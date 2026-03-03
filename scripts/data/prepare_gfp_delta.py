#!/usr/bin/env python3
"""Build a GFP delta-label dataset from a TAPE arrow file.

Each output row contains:
- protein: mutant sequence
- label: delta fluorescence relative to WT
- stage: split tag if available
- reason: edit operations from WT to mutant
"""

import argparse
import json
from typing import Dict, List

try:
    from datasets import Dataset
except Exception:  # pragma: no cover - optional dependency for CLI help
    Dataset = None


DEFAULT_WT_SEQUENCE = (
    "SKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTLSYGVQCF"
    "SRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLE"
    "YNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNE"
    "KRDHMVLLEFVTAAGITHGMDELYK"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate GFP delta-label JSONL from arrow data.")
    parser.add_argument("--input-arrow", required=True, help="Path to dataset-fluorescence-tape-*.arrow")
    parser.add_argument("--output-jsonl", required=True, help="Path to write delta dataset JSONL")
    parser.add_argument(
        "--wt-seq",
        default=DEFAULT_WT_SEQUENCE,
        help="WT sequence used as reference baseline",
    )
    parser.add_argument(
        "--min-delta",
        type=float,
        default=0.0,
        help="Keep rows with delta label >= min-delta",
    )
    parser.add_argument(
        "--include-wt",
        action="store_true",
        help="Keep WT row with delta=0 in output",
    )
    return parser.parse_args()


def build_reason(wt_seq: str, mutant_seq: str) -> str:
    if mutant_seq == wt_seq:
        return "wild type"

    if len(mutant_seq) == len(wt_seq):
        edits: List[str] = []
        for idx, (wt_aa, aa) in enumerate(zip(wt_seq, mutant_seq), start=1):
            if aa != wt_aa:
                edits.append(f"replace {wt_aa} to {aa} at position {idx}")
        return "; ".join(edits) if edits else "no change"

    # Fallback for unequal-length sequences.
    return "length mismatch to WT"


def main() -> None:
    args = parse_args()
    if Dataset is None:
        raise ModuleNotFoundError(
            "The 'datasets' package is required. Install dependencies from requirements/train.txt."
        )

    dataset = Dataset.from_file(args.input_arrow)
    wt_seq = args.wt_seq.strip()

    wt_label = None
    for row in dataset:
        if row.get("protein") == wt_seq:
            wt_label = row.get("label")
            break
    if wt_label is None:
        raise ValueError("WT sequence not found in input arrow dataset.")

    kept = 0
    with open(args.output_jsonl, "w", encoding="utf-8") as fout:
        for row in dataset:
            protein = row.get("protein", "")
            if not protein:
                continue

            label = float(row.get("label", 0.0))
            stage = row.get("stage", "")

            if protein == wt_seq:
                if not args.include_wt:
                    continue
                delta = 0.0
            else:
                delta = label - float(wt_label)

            if delta < args.min_delta:
                continue

            out: Dict[str, object] = {
                "protein": protein,
                "label": delta,
                "stage": stage,
                "reason": build_reason(wt_seq, protein),
            }
            fout.write(json.dumps(out, ensure_ascii=True) + "\n")
            kept += 1

    print(f"Wrote {kept} rows to {args.output_jsonl}")


if __name__ == "__main__":
    main()
