#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Shared components for FULL Edit Flows (Ins/Del/Sub):
- Needleman-Wunsch Alignment
- Full Model Definition
- Variable-length Sampling
"""

from __future__ import annotations

import math
import random
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Vocabulary ------------------------------------------------------------- #

AA_VOCAB = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_ID = {a: i for i, a in enumerate(AA_VOCAB)}
ID_TO_AA = {i: a for a, i in AA_TO_ID.items()}

BOS_ID = len(AA_VOCAB)         # 20
PAD_ID = len(AA_VOCAB) + 1     # 21
VOCAB_SIZE = len(AA_VOCAB) + 2 

# Special constant for Alignment (not in model vocab)
ALIGN_BLANK = -1 

def encode_protein(seq: str, add_bos: bool = True) -> List[int]:
    # Returns List[int] because standard collation handles padding later
    seq = seq.strip().upper()
    ids = [BOS_ID] if add_bos else []
    for ch in seq:
        if ch in AA_TO_ID:
            ids.append(AA_TO_ID[ch])
    return ids

def decode_protein(ids: List[int], remove_bos: bool = True) -> str:
    out = []
    for j, tid in enumerate(ids):
        if remove_bos and tid == BOS_ID: continue
        if tid == PAD_ID: continue
        if tid in ID_TO_AA:
            out.append(ID_TO_AA[tid])
    return "".join(out)

# --- 1. Alignment Logic ----------------------------------------------------- #

def align_pair_nw(x0: List[int], x1: List[int]) -> Tuple[List[int], List[int]]:
    """
    Needleman-Wunsch alignment on token lists *without* BOS.
    Cost: match=0, sub=1, indel=1.
    Returns aligned z0, z1 using ALIGN_BLANK.
    """
    n, m = len(x0), len(x1)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    ptr = [[0] * (m + 1) for _ in range(n + 1)]  # 0:diag, 1:up, 2:left

    for i in range(1, n + 1):
        dp[i][0] = i
        ptr[i][0] = 1
    for j in range(1, m + 1):
        dp[0][j] = j
        ptr[0][j] = 2

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost_match = 0 if x0[i - 1] == x1[j - 1] else 1
            s_diag = dp[i - 1][j - 1] + cost_match
            s_up = dp[i - 1][j] + 1
            s_left = dp[i][j - 1] + 1

            best = min(s_diag, s_up, s_left)
            dp[i][j] = best
            ptr[i][j] = 0 if best == s_diag else (1 if best == s_up else 2)

    z0, z1 = [], []
    i, j = n, m
    while i > 0 or j > 0:
        move = ptr[i][j]
        if move == 0:
            z0.append(x0[i - 1])
            z1.append(x1[j - 1])
            i -= 1
            j -= 1
        elif move == 1:
            z0.append(x0[i - 1])
            z1.append(ALIGN_BLANK)
            i -= 1
        else:
            z0.append(ALIGN_BLANK)
            z1.append(x1[j - 1])
            j -= 1

    return z0[::-1], z1[::-1]


def align_pair_random_pad(x0: List[int], x1: List[int], rng: random.Random) -> Tuple[List[int], List[int]]:
    """
    Randomly insert blanks to match lengths, preserving order.
    Alignment length = max(len(x0), len(x1)).
    """
    n, m = len(x0), len(x1)
    L = max(n, m)

    def pad_to(x: List[int], L: int) -> List[int]:
        if len(x) == L:
            return list(x)
        token_pos = sorted(rng.sample(range(L), len(x)))
        pos_set = set(token_pos)
        out: List[int] = []
        xi = 0
        for i in range(L):
            if i in pos_set:
                out.append(x[xi])
                xi += 1
            else:
                out.append(ALIGN_BLANK)
        return out

    return pad_to(x0, L), pad_to(x1, L)


def align_pair(x0_full: List[int], x1_full: List[int], mode: str = "nw", rng: Optional[random.Random] = None) -> Tuple[List[int], List[int]]:
    """
    Align full sequences that include BOS. BOS is forced to align at position 0.
    """
    assert len(x0_full) >= 1 and x0_full[0] == BOS_ID
    assert len(x1_full) >= 1 and x1_full[0] == BOS_ID

    x0 = x0_full[1:]
    x1 = x1_full[1:]
    rng = rng or random.Random()

    if mode == "nw":
        z0_body, z1_body = align_pair_nw(x0, x1)
    elif mode == "random_pad":
        z0_body, z1_body = align_pair_random_pad(x0, x1, rng)
    else:
        raise ValueError(f"Unknown align mode: {mode}")

    return [BOS_ID] + z0_body, [BOS_ID] + z1_body

# --- Model ------------------------------------------------------------------ #

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.dim() == 1: t = t.unsqueeze(-1)
        half = self.dim // 2
        freqs = torch.exp(torch.arange(half, device=t.device) * (-math.log(10000.0) / max(1, half - 1)))
        args = t * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1: emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb

class FullEditFlowsTransformer(nn.Module):
    """
    Full Edit Flows model.
    Outputs:
      - lambdas: ins, del, sub (all non-negative)
      - logits: q_ins, q_sub
    """
    def __init__(self, vocab_size, aa_vocab_size, max_seq_len, hidden_dim=256, num_layers=6, num_heads=8, dropout=0.1):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.token_emb = nn.Embedding(vocab_size, hidden_dim)
        self.pos_emb = nn.Embedding(max_seq_len, hidden_dim)
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        enc_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim*4, dropout=dropout, activation="gelu", batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.ln = nn.LayerNorm(hidden_dim)

        # Rate heads (lambda)
        self.lam_ins = nn.Linear(hidden_dim, 1)
        self.lam_del = nn.Linear(hidden_dim, 1)
        self.lam_sub = nn.Linear(hidden_dim, 1)

        # Distribution heads (Q)
        self.q_ins = nn.Linear(hidden_dim, aa_vocab_size)
        self.q_sub = nn.Linear(hidden_dim, aa_vocab_size)

    def forward(self, x_t, t, pad_mask=None):
        B, L = x_t.shape
        tok = self.token_emb(x_t)
        pos = self.pos_emb(torch.arange(L, device=x_t.device).unsqueeze(0).expand(B, -1))
        time = self.time_mlp(t).unsqueeze(1).expand(-1, L, -1)
        
        h = self.ln(self.encoder(tok + pos + time, src_key_padding_mask=pad_mask))

        # Rates: use softplus for positivity
        lam_ins = F.softplus(self.lam_ins(h)).squeeze(-1)
        lam_del = F.softplus(self.lam_del(h)).squeeze(-1)
        lam_sub = F.softplus(self.lam_sub(h)).squeeze(-1)

        q_ins = self.q_ins(h)
        q_sub = self.q_sub(h)
        
        # Masking rules for Full Edit Flows:
        # 1. Never delete/replace BOS (pos 0)
        # 2. Never insert/delete/replace PAD
        if pad_mask is not None:
            lam_ins = lam_ins.masked_fill(pad_mask, 0.0)
            lam_del = lam_del.masked_fill(pad_mask, 0.0)
            lam_sub = lam_sub.masked_fill(pad_mask, 0.0)
        
        lam_del[:, 0] = 0.0 
        lam_sub[:, 0] = 0.0
        
        return lam_ins, lam_del, lam_sub, q_ins, q_sub

# --- Loss (Eq 23) ----------------------------------------------------------- #

def editflows_full_loss_batch(
    model,
    x0_list,
    x1_list,
    device,
    align_mode: str = "nw",
    kappa_pow: int = 3,
    seed: Optional[int] = None,
):
    """
    Monte-Carlo estimate of Edit Flows Eq.(23) with on-the-fly alignment.
    Always trains full Ins/Del/Sub heads; if the data has no indels, the
    optimal solution naturally drives lam_ins/lam_del toward zero.
    """
    B = len(x0_list)
    t = torch.rand(B, device=device)

    k = t ** kappa_pow
    kd = (kappa_pow * (t ** (kappa_pow - 1))) if kappa_pow > 0 else torch.ones_like(t)
    w = kd / (1.0 - k + 1e-8)

    rng = random.Random(seed)

    xt_list = []
    edits_batch = []  # List of list of (op, pos, target_tok)

    for b in range(B):
        z0, z1 = align_pair(x0_list[b], x1_list[b], mode=align_mode, rng=rng)

        zt = []
        kb = float(k[b].item())
        for a0, a1 in zip(z0, z1):
            if a0 == a1:
                zt.append(a0)
            else:
                zt.append(a1 if rng.random() < kb else a0)

        xt = [tk for tk in zt if tk != ALIGN_BLANK]
        xt_list.append(xt)

        cur_xt_idx = 0
        sample_edits = []

        for i in range(len(zt)):
            is_blank_zt = zt[i] == ALIGN_BLANK

            if zt[i] == z1[i]:
                if not is_blank_zt:
                    cur_xt_idx += 1
                continue

            if zt[i] == ALIGN_BLANK and z1[i] != ALIGN_BLANK:
                pos_in_xt = max(0, cur_xt_idx - 1)
                if z1[i] < len(AA_VOCAB):
                    sample_edits.append(("ins", pos_in_xt, z1[i]))

            elif zt[i] != ALIGN_BLANK and z1[i] == ALIGN_BLANK:
                if cur_xt_idx != 0:
                    sample_edits.append(("del", cur_xt_idx, None))
                cur_xt_idx += 1

            else:
                if cur_xt_idx != 0 and z1[i] < len(AA_VOCAB):
                    sample_edits.append(("sub", cur_xt_idx, z1[i]))
                cur_xt_idx += 1

        edits_batch.append(sample_edits)

    max_len = max(len(x) for x in xt_list)
    x_ids = torch.full((B, max_len), PAD_ID, dtype=torch.long, device=device)
    pad_mask = torch.ones((B, max_len), dtype=torch.bool, device=device)

    for b in range(B):
        l = len(xt_list[b])
        x_ids[b, :l] = torch.tensor(xt_list[b], device=device)
        pad_mask[b, :l] = False

    lam_ins, lam_del, lam_sub, q_ins, q_sub = model(x_ids, t, pad_mask)

    rate_sum = (lam_ins + lam_del + lam_sub).masked_fill(pad_mask, 0.0).sum(dim=1)

    log_correct = torch.zeros(B, device=device)

    for b in range(B):
        s = 0.0
        for (op, pos, target_tok) in edits_batch[b]:
            if pos >= len(xt_list[b]):
                continue

            eps = 1e-8
            if op == "ins":
                s += torch.log(lam_ins[b, pos] + eps) + F.log_softmax(q_ins[b, pos], dim=-1)[target_tok]
            elif op == "del":
                s += torch.log(lam_del[b, pos] + eps)
            elif op == "sub":
                s += torch.log(lam_sub[b, pos] + eps) + F.log_softmax(q_sub[b, pos], dim=-1)[target_tok]
        log_correct[b] = s

    loss = rate_sum - w * log_correct
    return loss.mean()


# --- Sampling (Full Euler) -------------------------------------------------- #

@torch.no_grad()
def sample_full(model, wt_ids_list, n_steps=100, device="cuda", max_len: Optional[int] = None):
    """
    Full Euler sampling that always allows ins/del/sub. If sequences
    grow beyond positional embedding capacity, insertions are disabled.
    """
    model.eval()
    current_seqs = [list(wt) for wt in wt_ids_list]
    B = len(current_seqs)
    h = 1.0 / n_steps

    if max_len is None:
        max_len = getattr(model, "max_seq_len", None)
    if max_len is None:
        max_len = model.pos_emb.num_embeddings

    for step in range(n_steps):
        t_val = step / n_steps

        lens = [len(s) for s in current_seqs]
        if max(lens) > max_len:
            current_seqs = [s[:max_len] for s in current_seqs]
            lens = [len(s) for s in current_seqs]

        maxL = max(lens)
        x_tensor = torch.full((B, maxL), PAD_ID, dtype=torch.long, device=device)
        pad_mask = torch.ones((B, maxL), dtype=torch.bool, device=device)
        t_tensor = torch.full((B, 1), t_val, device=device)

        for b in range(B):
            x_tensor[b, :lens[b]] = torch.tensor(current_seqs[b], device=device)
            pad_mask[b, :lens[b]] = False

        lam_ins, lam_del, lam_sub, q_ins, q_sub = model(x_tensor, t_tensor, pad_mask)

        new_seqs = []
        for b in range(B):
            seq = current_seqs[b]
            L = len(seq)

            p_ins = (h * lam_ins[b, :L]).clamp(0, 0.9).cpu()
            p_del = (h * lam_del[b, :L]).clamp(0, 0.9).cpu()
            p_sub = (h * lam_sub[b, :L]).clamp(0, 0.9).cpu()

            if L >= max_len:
                p_ins[:] = 0.0

            do_del = torch.bernoulli(p_del).bool()
            do_sub = torch.bernoulli(p_sub).bool() & (~do_del)
            do_ins = torch.bernoulli(p_ins).bool()

            next_seq = []
            for i in range(L):
                if do_del[i]:
                    pass
                elif do_sub[i]:
                    tok = torch.multinomial(F.softmax(q_sub[b, i], dim=-1), 1).item()
                    next_seq.append(tok)
                else:
                    next_seq.append(seq[i])

                if do_ins[i] and len(next_seq) < max_len:
                    tok = torch.multinomial(F.softmax(q_ins[b, i], dim=-1), 1).item()
                    next_seq.append(tok)

            new_seqs.append(next_seq)
        current_seqs = new_seqs

    return current_seqs
