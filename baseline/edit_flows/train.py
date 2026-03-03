#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Training entrypoint for full insertion/deletion/substitution Edit Flows on GFP-style data.
Supports Single GPU and DDP (Distributed Data Parallel).

Usage (Single GPU):
  python train.py --data_jsonl train.jsonl --out_dir ./ckpt --wt MSK...

Usage (DDP - e.g., 4 GPUs):
  torchrun --nproc_per_node=4 train.py --data_jsonl train.jsonl --out_dir ./ckpt --wt MSK...
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

# wandb is optional; only required when --use_wandb is passed
try:
    import wandb  # type: ignore
except Exception:  # pragma: no cover - import guard
    wandb = None

from model import (
    AA_VOCAB,
    BOS_ID,
    PAD_ID,
    VOCAB_SIZE,
    FullEditFlowsTransformer,
    encode_protein,
    decode_protein,
    editflows_full_loss_batch,
    sample_full,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _norm_stage(s: Optional[str]) -> Optional[str]:
    if s is None:
        return None
    s2 = str(s).strip().lower()
    if s2 in ("all", "any", "none", "*", ""):
        return None
    return str(s)


_WT_RE = re.compile(r"WT sequence:\s*\n([A-Za-z]+)", re.MULTILINE)
_PROTEIN_JSON_RE = re.compile(r'\{\s*"protein"\s*:\s*"([A-Za-z]+)"\s*\}')


def extract_wt_from_human_msg(text: str) -> Optional[str]:
    m = _WT_RE.search(text)
    if not m:
        return None
    return m.group(1).strip().upper()


def extract_mutant_from_gpt_msg(text: str) -> Optional[str]:
    # Fast regex path
    m = _PROTEIN_JSON_RE.search(text)
    if m:
        return m.group(1).strip().upper()

    # Fallback: try to find the last JSON object and parse it
    idx = text.rfind("{")
    if idx == -1:
        return None
    try:
        obj = json.loads(text[idx:])
        if isinstance(obj, dict) and "protein" in obj:
            return str(obj["protein"]).strip().upper()
    except Exception:
        return None
    return None


@dataclass
class DatasetStats:
    n_total: int
    n_used: int
    n_bad_parse: int
    n_bad_vocab: int
    wt_len: int
    avg_mutations: float


class ShareGPTProteinPairDataset(Dataset):
    """
    Produces (x0, x1) token id lists for variable-length edit flows.

    We ignore the natural-language 'think' actions and rely on
    the final sequence string {"protein": "..."}.
    """
    def __init__(
        self,
        data_jsonl: str,
        stage: str = "train",
        wt_override: Optional[str] = None,
        require_same_wt: bool = True,
        max_examples: Optional[int] = None,
    ):
        super().__init__()
        self.data_jsonl = str(data_jsonl)
        self.stage = stage
        self.wt_seq: Optional[str] = wt_override.strip().upper() if wt_override else None
        self.require_same_wt = require_same_wt

        mutants: List[str] = []

        n_total = 0
        n_used = 0
        n_bad_parse = 0
        n_bad_vocab = 0

        with open(self.data_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                n_total += 1

                try:
                    rec = json.loads(line)
                except Exception:
                    n_bad_parse += 1
                    continue

                if stage is not None and rec.get("stage", None) != stage:
                    continue

                conv = rec.get("conversations", [])
                if not isinstance(conv, list) or len(conv) == 0:
                    n_bad_parse += 1
                    continue

                # find WT in human msg
                wt_here = None
                for msg in conv:
                    if msg.get("from") in ("human", "user"):
                        wt_here = extract_wt_from_human_msg(msg.get("value", ""))
                        if wt_here:
                            break

                # find mutant in gpt msg
                mut_here = None
                for msg in conv:
                    if msg.get("from") in ("gpt", "assistant"):
                        mut_here = extract_mutant_from_gpt_msg(msg.get("value", ""))
                        if mut_here:
                            break

                if mut_here is None:
                    # some datasets might store it directly
                    if isinstance(rec.get("protein", None), str):
                        mut_here = rec["protein"].strip().upper()

                if wt_override is None:
                    if wt_here is None:
                        n_bad_parse += 1
                        continue
                    if self.wt_seq is None:
                        self.wt_seq = wt_here
                    elif require_same_wt and wt_here != self.wt_seq:
                        # inconsistent WT across examples
                        n_bad_parse += 1
                        continue

                if mut_here is None:
                    n_bad_parse += 1
                    continue

                if self.wt_seq is None:
                    n_bad_parse += 1
                    continue

                # vocab check (allow variable length but only standard AA chars)
                mut_up = mut_here.strip().upper()
                if any(ch not in AA_VOCAB for ch in mut_up):
                    n_bad_vocab += 1
                    continue

                mutants.append(mut_up)
                n_used += 1
                if max_examples is not None and n_used >= max_examples:
                    break

        if self.wt_seq is None:
            raise ValueError("Could not determine WT sequence. Pass --wt explicitly or ensure it's in the human prompt.")

        self.mutants = mutants

        # pre-tokenize for speed
        self._x1_ids: List[List[int]] = [encode_protein(s, add_bos=True) for s in self.mutants]
        self._wt_ids: List[int] = encode_protein(self.wt_seq, add_bos=True)

        # stats
        mut_counts = []
        wt = self.wt_seq
        for s in self.mutants:
            diff = sum(a != b for a, b in zip(wt, s)) + abs(len(wt) - len(s))
            mut_counts.append(diff)
        avg_mut = float(sum(mut_counts) / max(1, len(mut_counts)))

        self.stats = DatasetStats(
            n_total=n_total,
            n_used=n_used,
            n_bad_parse=n_bad_parse,
            n_bad_vocab=n_bad_vocab,
            wt_len=len(self.wt_seq),
            avg_mutations=avg_mut,
        )

    @property
    def wt_ids(self) -> List[int]:
        return self._wt_ids

    def __len__(self) -> int:
        return len(self._x1_ids)

    def __getitem__(self, idx: int) -> Tuple[List[int], List[int]]:
        # returns (WT, Mutant) lists
        return self._wt_ids, self._x1_ids[idx]


def collate_lists(batch: List[Tuple[List[int], List[int]]]) -> Tuple[List[List[int]], List[List[int]]]:
    # batch is list of (x0_list, x1_list)
    x0s = [item[0] for item in batch]
    x1s = [item[1] for item in batch]
    return x0s, x1s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_jsonl", type=str, required=True, help="ShareGPT-style JSONL training file")
    ap.add_argument(
        "--stage",
        type=str,
        default="train",
        help="stage field to filter on (use 'all' to disable filtering)",
    )
    ap.add_argument("--valid_jsonl", type=str, default=None, help="Optional validation JSONL file")
    ap.add_argument(
        "--valid_stage",
        type=str,
        default="valid",
        help="stage field to filter on for validation (use 'all' to disable filtering)",
    )
    ap.add_argument("--eval_every", type=int, default=1, help="Run validation every N epochs (requires --valid_jsonl)")
    ap.add_argument(
        "--eval_every_steps",
        type=int,
        default=0,
        help="Run validation every N steps (0 disables; requires --valid_jsonl)",
    )
    ap.add_argument("--save_every", type=int, default=1, help="Save checkpoint every N epochs (0 disables)")
    ap.add_argument("--save_best", action="store_true", help="If set, save best-by-val-loss checkpoint")
    ap.add_argument("--wt", type=str, default=None, help="WT sequence override (optional)")
    ap.add_argument("--out_dir", type=str, required=True, help="Output directory for checkpoints")
    ap.add_argument("--max_examples", type=int, default=None, help="Optional cap on dataset size")

    # model
    ap.add_argument("--hidden_dim", type=int, default=256)
    ap.add_argument("--num_layers", type=int, default=6)
    ap.add_argument("--num_heads", type=int, default=8)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--max_len_buffer", type=int, default=50, help="Extra positional slots beyond observed max length")
    ap.add_argument(
        "--align_mode",
        type=str,
        default="nw",
        choices=["nw", "random_pad"],
        help="Auxiliary alignment strategy for edit flow training",
    )
    ap.add_argument("--kappa_pow", type=int, default=3, help="Exponent for kappa(t) scheduling (kappa=t^kappa_pow)")

    # training
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # sampling demo
    ap.add_argument("--demo_samples", type=int, default=8)
    ap.add_argument("--demo_steps", type=int, default=200)

    # wandb
    ap.add_argument("--use_wandb", action="store_true", help="Enable Weights & Biases logging")
    ap.add_argument("--wandb_project", type=str, default="gfp_editflows", help="WandB project name")
    ap.add_argument("--wandb_entity", type=str, default=None, help="WandB entity/user (optional)")
    ap.add_argument("--wandb_run_name", type=str, default=None, help="Optional run name")

    args = ap.parse_args()
    # --- DDP Init ---
    is_ddp = False
    rank = 0
    world_size = 1
    local_rank = 0

    if "LOCAL_RANK" in os.environ:
        is_ddp = True
        local_rank = int(os.environ["LOCAL_RANK"])
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        torch.cuda.set_device(local_rank)
        args.device = f"cuda:{local_rank}"
        set_seed(args.seed)
    else:
        set_seed(args.seed)

    device = torch.device(args.device)

    args.stage = _norm_stage(args.stage)
    args.valid_stage = _norm_stage(args.valid_stage)

    out_dir = Path(args.out_dir)
    if rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)

    ds = ShareGPTProteinPairDataset(
        data_jsonl=args.data_jsonl,
        stage=args.stage,
        wt_override=args.wt,
        max_examples=args.max_examples,
    )
    if rank == 0:
        print("Dataset stats:", asdict(ds.stats))
    wt_ids = ds.wt_ids
    wt_len = len(ds.wt_seq)

    max_mut_len = max((len(s) for s in ds.mutants), default=wt_len)
    base_len = max(wt_len, max_mut_len) + 1  # +1 for BOS
    max_seq_len = base_len + max(args.max_len_buffer, 0)

    if is_ddp:
        train_sampler = DistributedSampler(ds, num_replicas=world_size, rank=rank, shuffle=True)
    else:
        train_sampler = None

    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=0,
        collate_fn=collate_lists,
        drop_last=True,
    )

    # Optional validation loader
    valid_dl = None
    valid_ds = None
    if args.valid_jsonl is not None:
        valid_ds = ShareGPTProteinPairDataset(
            data_jsonl=args.valid_jsonl,
            stage=args.valid_stage,
            wt_override=ds.wt_seq,          # force same WT as train
            require_same_wt=True,
            max_examples=None,
        )
        if rank == 0:
            print("Valid dataset stats:", asdict(valid_ds.stats))
        if is_ddp:
            valid_sampler = DistributedSampler(valid_ds, num_replicas=world_size, rank=rank, shuffle=False)
        else:
            valid_sampler = None
        valid_dl = DataLoader(
            valid_ds,
            batch_size=args.batch_size,
            shuffle=False,
            sampler=valid_sampler,
            num_workers=0,
            collate_fn=collate_lists,
            drop_last=False,
        )

    model = FullEditFlowsTransformer(
        vocab_size=VOCAB_SIZE,
        aa_vocab_size=len(AA_VOCAB),
        max_seq_len=max_seq_len,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
    ).to(device)

    if is_ddp:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if rank == 0:
        cfg = vars(args).copy()
        cfg.update({
            "AA_VOCAB": AA_VOCAB,
            "BOS_ID": BOS_ID,
            "PAD_ID": PAD_ID,
            "VOCAB_SIZE": VOCAB_SIZE,
            "wt_seq": ds.wt_seq,
            "wt_len": len(ds.wt_seq),
            "max_seq_len": max_seq_len,
            "align_mode": args.align_mode,
            "kappa_pow": args.kappa_pow,
            "world_size": world_size,
            "is_ddp": is_ddp,
        })
        with open(out_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)

    wandb_run = None
    if args.use_wandb and rank == 0:
        if wandb is None:
            raise ImportError("wandb is not installed; install with `pip install wandb` or disable --use_wandb")
        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name,
            config=cfg,
            dir=str(out_dir),
        )

    model.train()
    global_step = 0
    best_val = float("inf")

    @torch.no_grad()
    def run_eval() -> float:
        assert valid_dl is not None
        model.eval()
        losses = []
        for x0_list, x1_list in valid_dl:
            loss = editflows_full_loss_batch(
                model,
                x0_list,
                x1_list,
                device,
                align_mode=args.align_mode,
                kappa_pow=args.kappa_pow,
            )
            losses.append(loss.detach().float())
        if len(losses) == 0:
            local_avg = torch.tensor(0.0, device=device)
        else:
            local_avg = torch.stack(losses).mean()
        if is_ddp:
            dist.all_reduce(local_avg, op=dist.ReduceOp.AVG)
        model.train()
        return local_avg.item()

    def maybe_run_eval(tag: str) -> None:
        nonlocal best_val
        if valid_dl is None:
            return
        val_loss = run_eval()
        if rank == 0:
            print(f"[valid] {tag}: loss={val_loss:.4f}")
            if wandb_run is not None:
                wandb.log(
                    {"valid/loss": val_loss, "valid/tag": tag, "valid/epoch": epoch + 1, "global_step": global_step},
                    step=global_step,
                )
            if args.save_best and val_loss < best_val:
                best_val = val_loss
                best_path = out_dir / "model_best.pt"
                state_dict = model.module.state_dict() if is_ddp else model.state_dict()
                torch.save(
                    {"model": state_dict, "epoch": epoch + 1, "step": global_step, "val_loss": best_val},
                    best_path,
                )
                print(f"[saved best] {best_path} (val_loss={best_val:.4f})")
                if wandb_run is not None:
                    wandb.summary["best_val_loss"] = best_val

    try:
        for epoch in range(args.epochs):
            if is_ddp and train_sampler is not None:
                train_sampler.set_epoch(epoch)

            if rank == 0:
                pbar = tqdm(dl, desc=f"epoch {epoch+1}/{args.epochs}", leave=False)
                iter_loader = pbar
            else:
                iter_loader = dl

            for x0_list, x1_list in iter_loader:
                loss = editflows_full_loss_batch(
                    model,
                    x0_list,
                    x1_list,
                    device,
                    align_mode=args.align_mode,
                    kappa_pow=args.kappa_pow,
                )

                optim.zero_grad(set_to_none=True)
                loss.backward()
                if args.grad_clip is not None and args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optim.step()

                global_step += 1

                if rank == 0:
                    pbar.set_postfix({"loss": f"{loss.item():.4f}"})
                    if wandb_run is not None:
                        wandb.log(
                            {
                                "train/loss": loss.item(),
                                "train/epoch": epoch + 1,
                                "global_step": global_step,
                                "lr": optim.param_groups[0]["lr"],
                            },
                            step=global_step,
                        )

                if (
                    valid_dl is not None
                    and args.eval_every_steps is not None
                    and args.eval_every_steps > 0
                    and (global_step % args.eval_every_steps == 0)
                ):
                    maybe_run_eval(tag=f"step {global_step}")

            if rank == 0 and args.save_every > 0 and ((epoch + 1) % args.save_every == 0):
                ckpt_path = out_dir / f"model_epoch{epoch+1}.pt"
                state_dict = model.module.state_dict() if is_ddp else model.state_dict()
                torch.save({"model": state_dict, "epoch": epoch + 1, "step": global_step}, ckpt_path)
                print(f"[saved] {ckpt_path}")

            # validation
            if valid_dl is not None and args.eval_every > 0 and ((epoch + 1) % args.eval_every == 0):
                maybe_run_eval(tag=f"epoch {epoch+1}")

            if is_ddp:
                dist.barrier()
    finally:
        if rank == 0 and wandb_run is not None:
            wandb_run.finish()
        if is_ddp:
            dist.destroy_process_group()

    # sampling demo (rank 0 only)
    if rank == 0:
        print("\n=== Sampling demo (full edit flows) ===")
        inference_model = model.module if is_ddp else model
        inference_model.eval()
        wts = [wt_ids for _ in range(args.demo_samples)]
        samples = sample_full(
            model=inference_model,
            wt_ids_list=wts,
            n_steps=args.demo_steps,
            device=device,
        )

        for i, seq_ids in enumerate(samples):
            mut_seq = decode_protein(seq_ids, remove_bos=True)
            print(f"\nSample {i}:")
            print(json.dumps({"protein": mut_seq}))

        state_dict = model.module.state_dict() if is_ddp else model.state_dict()
        torch.save({"model": state_dict, "epoch": args.epochs, "step": global_step}, out_dir / "model_final.pt")
        print(f"\n[saved] {out_dir / 'model_final.pt'}")


if __name__ == "__main__":
    main()
