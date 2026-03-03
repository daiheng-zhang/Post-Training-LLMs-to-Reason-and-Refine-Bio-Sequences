#!/usr/bin/env python3
"""Generate synthetic GFP variants by random edit actions."""

import argparse
import random
from pathlib import Path
from typing import List, Tuple


DEFAULT_WT_SEQUENCE = (
    "SKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTLSYGVQCF"
    "SRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLE"
    "YNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNE"
    "KRDHMVLLEFVTAAGITHGMDELYK"
)
AA_LIST = list("ACDEFGHIKLMNPQRSTVWY")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate random GFP edits into TSV.")
    parser.add_argument("--output", required=True, help="Output TSV path")
    parser.add_argument("--num-seq", type=int, default=1000, help="Number of synthetic sequences")
    parser.add_argument("--min-actions", type=int, default=1)
    parser.add_argument("--max-actions", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wt-seq", default=DEFAULT_WT_SEQUENCE)
    return parser.parse_args()


def add_aa(seq: str, rng: random.Random) -> Tuple[str, str]:
    pos = rng.randint(0, len(seq))
    aa = rng.choice(AA_LIST)
    return seq[:pos] + aa + seq[pos:], f"add {aa} at position {pos + 1}"


def remove_aa(seq: str, rng: random.Random) -> Tuple[str, str]:
    if len(seq) <= 1:
        return seq, "remove skipped (sequence too short)"
    pos = rng.randint(0, len(seq) - 1)
    aa = seq[pos]
    return seq[:pos] + seq[pos + 1 :], f"remove {aa} at position {pos + 1}"


def replace_aa(seq: str, rng: random.Random) -> Tuple[str, str]:
    pos = rng.randint(0, len(seq) - 1)
    old_aa = seq[pos]
    new_aa = rng.choice(AA_LIST)
    return seq[:pos] + new_aa + seq[pos + 1 :], f"replace {old_aa} to {new_aa} at position {pos + 1}"


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    actions = [add_aa, remove_aa, replace_aa]

    with output_path.open("w", encoding="utf-8") as fout:
        fout.write("sequence_id\tsequence\tactions\n")
        for idx in range(args.num_seq):
            seq = args.wt_seq
            logs: List[str] = []
            num_actions = rng.randint(args.min_actions, args.max_actions)
            for _ in range(num_actions):
                fn = rng.choice(actions)
                seq, msg = fn(seq, rng)
                logs.append(msg)
            fout.write(f"gfp_aug_{idx:06d}\t{seq}\t{'; '.join(logs)}\n")

    print(f"Wrote {args.num_seq} rows to {output_path}")


if __name__ == "__main__":
    main()
