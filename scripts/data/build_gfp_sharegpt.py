#!/usr/bin/env python3
"""Convert GFP delta-label JSONL into instruction-tuning datasets.

Input rows are expected to include:
- protein
- reason
- stage (train/valid/test)
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, Iterator, Tuple


DEFAULT_WT_SEQUENCE = (
    "SKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTLSYGVQCF"
    "SRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLE"
    "YNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNE"
    "KRDHMVLLEFVTAAGITHGMDELYK"
)

SYSTEM_PROMPT = (
    "You are a protein engineer optimizing the fluorescence of a Green Fluorescent "
    "Protein (GFP) variant. We use a wild-type (WT) amino-acid sequence as the "
    "baseline. Your goal is to propose sequence edits that improve fluorescence."
)

USER_TEMPLATE = (
    "Background:\n"
    "- WT is the reference GFP sequence used as the baseline fluorescence.\n"
    "- You must propose mutation actions that convert WT to a brighter mutant.\n\n"
    "Output format requirements (STRICT):\n"
    "1) Put ONLY the mutation actions inside <think>...</think>.\n"
    "2) After </think>, output ONLY a JSON object with exactly one key:\n"
    "   {{\"protein\": \"<MUTATED_SEQUENCE>\"}}\n\n"
    "WT sequence:\n{wt_seq}\n"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build ShareGPT or Alpaca data from GFP delta dataset.")
    parser.add_argument("--input-jsonl", required=True, help="Input delta-label JSONL")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--format", choices=["sharegpt", "alpaca"], default="sharegpt")
    parser.add_argument("--wt-seq", default=DEFAULT_WT_SEQUENCE, help="WT sequence")
    parser.add_argument("--prefix", default="gfp_optimization", help="Output filename prefix")
    return parser.parse_args()


def iter_jsonl(path: Path) -> Iterator[Tuple[int, Dict[str, object]]]:
    with path.open("r", encoding="utf-8") as fin:
        for idx, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield idx, json.loads(line)
            except json.JSONDecodeError:
                continue


def wrap_assistant(reason: str, mutant: str) -> str:
    reason_text = (reason or "").strip()
    if "<think>" not in reason_text:
        reason_text = f"<think>\n{reason_text}\n</think>"
    return f"{reason_text}\n{json.dumps({'protein': mutant}, ensure_ascii=True)}"


def to_sharegpt(wt_seq: str, mutant: str, reason: str, stage: str) -> Dict[str, object]:
    return {
        "conversations": [
            {"from": "system", "value": SYSTEM_PROMPT},
            {"from": "human", "value": USER_TEMPLATE.format(wt_seq=wt_seq)},
            {"from": "gpt", "value": wrap_assistant(reason, mutant)},
        ],
        "stage": stage,
    }


def to_alpaca(wt_seq: str, mutant: str, reason: str, stage: str) -> Dict[str, object]:
    return {
        "instruction": f"{SYSTEM_PROMPT}\n\n{USER_TEMPLATE.format(wt_seq=wt_seq)}",
        "input": "",
        "output": wrap_assistant(reason, mutant),
        "stage": stage,
    }


def select_builder(fmt: str):
    if fmt == "alpaca":
        return to_alpaca
    return to_sharegpt


def stage_bucket(stage: str) -> str:
    stage = (stage or "").strip().lower()
    if stage in {"train"}:
        return "train"
    if stage in {"valid", "validation", "eval", "dev"}:
        return "valid"
    if stage in {"test"}:
        return "test"
    return "other"


def main() -> None:
    args = parse_args()
    in_path = Path(args.input_jsonl)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    builder = select_builder(args.format)

    files = {
        "all": (out_dir / f"{args.prefix}_{args.format}_all.jsonl").open("w", encoding="utf-8"),
        "train": (out_dir / f"{args.prefix}_{args.format}_train.jsonl").open("w", encoding="utf-8"),
        "valid": (out_dir / f"{args.prefix}_{args.format}_valid.jsonl").open("w", encoding="utf-8"),
        "test": (out_dir / f"{args.prefix}_{args.format}_test.jsonl").open("w", encoding="utf-8"),
    }

    counts = {"all": 0, "train": 0, "valid": 0, "test": 0}
    try:
        for _, row in iter_jsonl(in_path):
            mutant = str(row.get("protein", "")).strip()
            reason = str(row.get("reason", "")).strip()
            stage = str(row.get("stage", "")).strip()
            if not mutant or not reason:
                continue

            record = builder(args.wt_seq, mutant, reason, stage)
            line = json.dumps(record, ensure_ascii=True) + "\n"
            files["all"].write(line)
            counts["all"] += 1

            bucket = stage_bucket(stage)
            if bucket in files:
                files[bucket].write(line)
                counts[bucket] += 1
    finally:
        for f in files.values():
            f.close()

    print(
        f"Wrote all/train/valid/test = "
        f"{counts['all']}/{counts['train']}/{counts['valid']}/{counts['test']} to {out_dir}"
    )


if __name__ == "__main__":
    main()
