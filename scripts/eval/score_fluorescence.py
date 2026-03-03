#!/usr/bin/env python3
"""Score protein sequences with SaProt fluorescence regressor.

Supported inputs:
- JSONL (one object per line)
- JSON (list or {'data': [...]})
- TSV (with sequence column)
- FASTA
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

try:
    import torch
except Exception:  # pragma: no cover - optional dependency for CLI help
    torch = None


DEFAULT_BASE_MODEL_ID = "westlake-repl/SaProt_650M_AF2"
DEFAULT_ADAPTER_ID = "SaProtHub/Model-Fluorescence-650M"
DEFAULT_FALLBACK_FIELDS = ("parsed_protein", "protein", "sequence", "seq")


def infer_input_format(path: str, declared: str) -> str:
    if declared != "auto":
        return declared
    lower = path.lower()
    if lower.endswith(".jsonl"):
        return "jsonl"
    if lower.endswith(".json"):
        return "json"
    if lower.endswith(".tsv"):
        return "tsv"
    if lower.endswith((".fasta", ".fa", ".faa", ".fas")):
        return "fasta"

    with open(path, "r", encoding="utf-8") as fin:
        for line in fin:
            s = line.strip()
            if not s:
                continue
            if s.startswith(">"):
                return "fasta"
            if "\t" in s and not s.startswith("{") and not s.startswith("["):
                return "tsv"
            try:
                obj = json.loads(s)
                if isinstance(obj, list):
                    return "json"
                return "jsonl"
            except json.JSONDecodeError:
                return "fasta"
    return "jsonl"


def extract_sequence(row: Dict[str, object], seq_field: str) -> str:
    value = row.get(seq_field)
    if isinstance(value, str) and value:
        return value
    for key in DEFAULT_FALLBACK_FIELDS:
        value = row.get(key)
        if isinstance(value, str) and value:
            return value
    raise KeyError(f"Missing sequence field '{seq_field}' and fallbacks.")


def read_jsonl(path: str, seq_field: str, max_samples: int) -> Tuple[List[Dict[str, object]], List[str]]:
    rows: List[Dict[str, object]] = []
    seqs: List[str] = []
    with open(path, "r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            try:
                seq = extract_sequence(obj, seq_field)
            except KeyError:
                continue
            rows.append(obj)
            seqs.append(seq)
            if max_samples > 0 and len(rows) >= max_samples:
                break
    return rows, seqs


def read_json(path: str, seq_field: str, max_samples: int) -> Tuple[List[Dict[str, object]], List[str]]:
    with open(path, "r", encoding="utf-8") as fin:
        data = json.load(fin)
    if isinstance(data, dict):
        records = data.get("data", [])
    elif isinstance(data, list):
        records = data
    else:
        raise ValueError("JSON input must be a list or {'data': [...]}.")

    rows: List[Dict[str, object]] = []
    seqs: List[str] = []
    for obj in records:
        try:
            seq = extract_sequence(obj, seq_field)
        except KeyError:
            continue
        rows.append(obj)
        seqs.append(seq)
        if max_samples > 0 and len(rows) >= max_samples:
            break
    return rows, seqs


def read_tsv(path: str, seq_field: str, max_samples: int) -> Tuple[List[Dict[str, str]], List[str], List[str]]:
    rows: List[Dict[str, str]] = []
    seqs: List[str] = []
    with open(path, "r", encoding="utf-8", newline="") as fin:
        reader = csv.DictReader(fin, delimiter="\t")
        fieldnames = reader.fieldnames or []
        for row in reader:
            try:
                seq = extract_sequence(row, seq_field)
            except KeyError:
                continue
            rows.append(row)
            seqs.append(seq)
            if max_samples > 0 and len(rows) >= max_samples:
                break
    return rows, seqs, fieldnames


def read_fasta(path: str, max_samples: int) -> Tuple[List[Dict[str, str]], List[str]]:
    rows: List[Dict[str, str]] = []
    seqs: List[str] = []
    current_id: Optional[str] = None
    chunks: List[str] = []

    def flush() -> None:
        nonlocal current_id, chunks
        if current_id is None or not chunks:
            return
        seq = "".join(chunks)
        rows.append({"id": current_id, "sequence": seq})
        seqs.append(seq)
        current_id = None
        chunks = []

    with open(path, "r", encoding="utf-8") as fin:
        for line in fin:
            s = line.strip()
            if not s:
                continue
            if s.startswith(">"):
                flush()
                current_id = s[1:].strip() or f"seq_{len(rows)+1:05d}"
                if max_samples > 0 and len(rows) >= max_samples:
                    break
            else:
                if current_id is None:
                    current_id = f"seq_{len(rows)+1:05d}"
                chunks.append(s)
        flush()

    if max_samples > 0:
        rows = rows[:max_samples]
        seqs = seqs[:max_samples]
    return rows, seqs


def load_model(base_model_id: str, adapter_id: str, cache_root: str, device):
    from peft import PeftModel
    from transformers import AutoConfig, AutoModelForSequenceClassification, EsmTokenizer

    base_cache = os.path.join(cache_root, "saprot_base")
    adapter_cache = os.path.join(cache_root, "saprot_adapter")
    os.makedirs(base_cache, exist_ok=True)
    os.makedirs(adapter_cache, exist_ok=True)

    tokenizer = EsmTokenizer.from_pretrained(base_model_id, cache_dir=base_cache)
    config = AutoConfig.from_pretrained(base_model_id, cache_dir=base_cache)
    config.num_labels = 1
    config.problem_type = "regression"

    base_model = AutoModelForSequenceClassification.from_pretrained(
        base_model_id,
        config=config,
        cache_dir=base_cache,
    )
    model = PeftModel.from_pretrained(base_model, adapter_id, cache_dir=adapter_cache)
    model.to(device)
    model.eval()
    return tokenizer, model


def _no_grad():
    if torch is None:
        def passthrough(func):
            return func
        return passthrough
    return torch.no_grad()


@_no_grad()
def predict_batch(seqs: List[str], tokenizer, model, device: torch.device) -> List[float]:
    formatted = ["".join([aa + "#" for aa in s]) if "#" not in s else s for s in seqs]
    enc = tokenizer(
        formatted,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=1024,
    )
    enc = {k: v.to(device) for k, v in enc.items()}
    outputs = model(**enc)
    logits = outputs.logits.squeeze(-1)
    if logits.ndim == 0:
        logits = logits.unsqueeze(0)
    return [float(x) for x in logits.detach().cpu().tolist()]


def predict_all(seqs: List[str], tokenizer, model, device: torch.device, batch_size: int) -> List[float]:
    preds: List[float] = []
    for start in range(0, len(seqs), batch_size):
        preds.extend(predict_batch(seqs[start : start + batch_size], tokenizer, model, device))
    return preds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict fluorescence score for protein sequences.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--input-format", choices=["auto", "jsonl", "json", "tsv", "fasta"], default="auto")
    parser.add_argument("--seq-field", default="parsed_protein")
    parser.add_argument("--output-field", default="predict_label")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--base-model-id", default=DEFAULT_BASE_MODEL_ID)
    parser.add_argument("--adapter-id", default=DEFAULT_ADAPTER_ID)
    parser.add_argument("--cache-root", default=os.path.expanduser("~/.cache/stride/models"))
    default_device = "cpu"
    if torch is not None and torch.cuda.is_available():
        default_device = "cuda"
    parser.add_argument("--device", default=default_device)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fmt = infer_input_format(args.input, args.input_format)

    if fmt == "jsonl":
        rows, seqs = read_jsonl(args.input, args.seq_field, args.max_samples)
        fieldnames = None
    elif fmt == "json":
        rows, seqs = read_json(args.input, args.seq_field, args.max_samples)
        fieldnames = None
    elif fmt == "tsv":
        rows, seqs, fieldnames = read_tsv(args.input, args.seq_field, args.max_samples)
    else:
        rows, seqs = read_fasta(args.input, args.max_samples)
        fieldnames = None

    print(f"Input format: {fmt}")
    print(f"Loaded {len(seqs)} valid sequences from {args.input}")

    if args.dry_run:
        print("Dry-run enabled: model loading and scoring skipped.")
        return

    if torch is None:
        raise ModuleNotFoundError(
            "The 'torch' package is required. Install dependencies from requirements/train.txt."
        )

    if not seqs:
        raise ValueError("No valid sequences were found in the input.")

    device = torch.device(args.device)
    tokenizer, model = load_model(args.base_model_id, args.adapter_id, args.cache_root, device)
    preds = predict_all(seqs, tokenizer, model, device, args.batch_size)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if fmt == "jsonl":
        with open(out_path, "w", encoding="utf-8") as fout:
            for row, pred in zip(rows, preds):
                row[args.output_field] = pred
                fout.write(json.dumps(row, ensure_ascii=True) + "\n")
    elif fmt == "json":
        for row, pred in zip(rows, preds):
            row[args.output_field] = pred
        with open(out_path, "w", encoding="utf-8") as fout:
            json.dump(rows, fout, ensure_ascii=True, indent=2)
    elif fmt == "tsv":
        if args.output_field not in (fieldnames or []):
            fieldnames = (fieldnames or []) + [args.output_field]
        with open(out_path, "w", encoding="utf-8", newline="") as fout:
            writer = csv.DictWriter(fout, fieldnames=fieldnames, delimiter="\t")
            writer.writeheader()
            for row, pred in zip(rows, preds):
                row[args.output_field] = str(pred)
                writer.writerow(row)
    else:
        with open(out_path, "w", encoding="utf-8") as fout:
            for row, pred in zip(rows, preds):
                row[args.output_field] = pred
                fout.write(json.dumps(row, ensure_ascii=True) + "\n")

    print(f"Wrote predictions to {out_path}")


if __name__ == "__main__":
    main()
