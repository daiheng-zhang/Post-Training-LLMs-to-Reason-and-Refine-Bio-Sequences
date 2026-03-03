#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Standalone sampling script for FULL Edit Flows (ins/del/sub).

Example:
  python sample_full.py --ckpt_dir ./ckpt_full --n_samples 8 --n_steps 200
  python sample_full.py --ckpt_dir ./ckpt_full --output_path samples.jsonl
  # (optional) control max length or alignment for action diff:
  python sample_full.py --ckpt_dir ./ckpt_full --max_len 400 --diff_align_mode random_pad
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import List, Tuple

import torch

from model import (
    AA_VOCAB,
    ID_TO_AA,
    BOS_ID,
    PAD_ID,
    VOCAB_SIZE,
    ALIGN_BLANK,
    FullEditFlowsTransformer,
    encode_protein,
    decode_protein,
    align_pair,
    sample_full,
)


def _auto_device(pref: str | None) -> str:
    if pref and pref.lower() != "auto":
        return pref
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_model(config: dict, ckpt_path: Path, device: str) -> FullEditFlowsTransformer:
    max_seq_len = int(config.get("max_seq_len", 0) or (int(config.get("wt_len", 0)) + 1 + 50))
    model = FullEditFlowsTransformer(
        vocab_size=int(config.get("VOCAB_SIZE", VOCAB_SIZE)),
        aa_vocab_size=len(config.get("AA_VOCAB", AA_VOCAB)),
        max_seq_len=max_seq_len,
        hidden_dim=int(config.get("hidden_dim", 256)),
        num_layers=int(config.get("num_layers", 6)),
        num_heads=int(config.get("num_heads", 8)),
        dropout=float(config.get("dropout", 0.1)),
    ).to(device)

    checkpoint = torch.load(ckpt_path, map_location=device)
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()
    return model


def _id_to_aa(tid: int) -> str:
    if tid == BOS_ID:
        return "<BOS>"
    if tid == PAD_ID:
        return "<PAD>"
    return ID_TO_AA.get(tid, "?")


def diff_to_actions_full(
    wt_ids: List[int],
    mut_ids: List[int],
    diff_align_mode: str = "nw",
    diff_seed: int = 0,
) -> Tuple[str, int]:
    """
    Generate a human-readable edit script (ins/del/sub) using alignment.
    Positions are 1-based on WT amino acids (excluding BOS).
    Insert anchor uses 'after position k' where k=0 means 'after BOS'.
    """
    rng = random.Random(diff_seed)
    z0, z1 = align_pair(wt_ids, mut_ids, mode=diff_align_mode, rng=rng)

    actions: List[str] = []

    wt_idx = -1              # index over z0 non-blank tokens (includes BOS as wt_idx=0)
    last_anchor = 0          # anchor position in WT-token index space (0 means BOS)

    for a0, a1 in zip(z0, z1):
        if a0 != ALIGN_BLANK:
            wt_idx += 1
            last_anchor = max(last_anchor, wt_idx)

        if a0 == a1:
            continue

        # INSERT: WT has blank, mutant has token
        if a0 == ALIGN_BLANK and a1 != ALIGN_BLANK:
            aa = _id_to_aa(a1)
            actions.append(f"insert {aa} after position {last_anchor}")

        # DELETE: WT has token, mutant has blank
        elif a0 != ALIGN_BLANK and a1 == ALIGN_BLANK:
            if wt_idx > 0:  # skip BOS
                aa = _id_to_aa(a0)
                actions.append(f"delete {aa} at position {wt_idx}")

        # SUB: both tokens but different
        else:
            if wt_idx > 0:  # skip BOS
                aa0 = _id_to_aa(a0)
                aa1 = _id_to_aa(a1)
                actions.append(f"replace {aa0} to {aa1} at position {wt_idx}")

    return ("; ".join(actions) if actions else "no edits", len(actions))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", type=str, required=True, help="Directory containing config.json and weights")
    ap.add_argument("--ckpt_name", type=str, default="model_final.pt", help="Checkpoint filename inside ckpt_dir")
    ap.add_argument("--wt", type=str, default=None, help="WT sequence override (falls back to config wt_seq)")
    ap.add_argument("--n_samples", type=int, default=8)
    ap.add_argument("--n_steps", type=int, default=200)
    ap.add_argument("--device", type=str, default="auto", help="'auto', 'cuda', or 'cpu'")
    ap.add_argument("--max_len", type=int, default=None, help="Hard cap on generated length (including BOS)")
    ap.add_argument("--diff_align_mode", type=str, default="nw", choices=["nw", "random_pad"], help="Alignment mode for diff display")
    ap.add_argument("--diff_seed", type=int, default=0, help="Seed for diff alignment when using random_pad")
    ap.add_argument("--seed", type=int, default=None, help="Random seed for sampling reproducibility")
    ap.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Path to save results as jsonl (relative paths use current working directory)",
    )

    args = ap.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    ckpt_dir = Path(args.ckpt_dir)
    config_path = ckpt_dir / "config.json"
    ckpt_path = ckpt_dir / args.ckpt_name

    if not config_path.exists():
        raise FileNotFoundError(f"Config not found at {config_path}")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    wt_seq = (args.wt or cfg.get("wt_seq") or "").strip().upper()
    if not wt_seq:
        raise ValueError("WT sequence missing; pass --wt or ensure 'wt_seq' is in config.json")

    device = _auto_device(args.device)
    print(f"Loading model from {ckpt_path} on device={device}")
    model = load_model(cfg, ckpt_path, device=device)

    # Decide a safe max_len cap
    cfg_max_seq_len = int(cfg.get("max_seq_len", getattr(model, "max_seq_len", 0) or model.pos_emb.num_embeddings))
    model_cap = int(getattr(model, "max_seq_len", model.pos_emb.num_embeddings))
    max_len_cap = args.max_len if args.max_len is not None else cfg_max_seq_len
    max_len_cap = min(max_len_cap, model_cap)
    if max_len_cap <= 0:
        max_len_cap = model_cap

    wt_ids: List[int] = encode_protein(wt_seq, add_bos=True)
    wt_batch = [wt_ids for _ in range(args.n_samples)]

    # Note: sample_full returns List[List[int]] (each includes BOS id)
    print(f"Sampling {args.n_samples} seqs with n_steps={args.n_steps}, max_len_cap={max_len_cap} ...")
    samples_ids = sample_full(
        model=model,
        wt_ids_list=wt_batch,
        n_steps=args.n_steps,
        device=device,
        max_len=max_len_cap,
    )

    output_path = Path(args.output_path).expanduser() if args.output_path else None
    out_file = None

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        out_file = output_path.open("w", encoding="utf-8")

    for i, seq_ids in enumerate(samples_ids):
        seq = decode_protein(seq_ids, remove_bos=True)
        actions, n_edits = diff_to_actions_full(
            wt_ids,
            seq_ids,
            diff_align_mode=args.diff_align_mode,
            diff_seed=args.diff_seed,
        )

        result_obj = {
            "id": i,
            "protein": seq,
            "actions": actions,
            "n_edits": n_edits,
            "length": len(seq),
        }

        print(f"\nSample {i}:")
        print("actions:", actions)
        print(json.dumps({"protein": seq}))

        if out_file:
            out_file.write(json.dumps(result_obj) + "\n")

    if out_file:
        out_file.close()
        print(f"\nSaved {args.n_samples} samples to {output_path}")


if __name__ == "__main__":
    main()
