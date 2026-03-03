import re
import random
from typing import List, Dict, Any, Tuple, Optional

# --- 1) Regex SMILES tokenizer (Schwaller et al. atom-wise pattern) ---
SMI_REGEX = re.compile(
    r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|/|:|~|@|\?|>>?|\*|\$|\%[0-9]{2}|[0-9])"
)

def tokenize(smiles: str) -> List[str]:
    toks = SMI_REGEX.findall(smiles)
    # Ensure the regex covers the full string; otherwise indices are unreliable.
    if "".join(toks) != smiles:
        raise ValueError(
            "Tokenizer did not fully cover SMILES.\n"
            f"SMILES: {smiles}\n"
            f"JOIN(tokens): {''.join(toks)}\n"
            f"Tokens: {toks}"
        )
    return toks

# A lightweight "sampleable token set": from input/output plus basic symbols.
BASE_VOCAB = ["B","Br","C","Cl","N","O","S","P","F","I",
              "b","c","n","o","s","p",
              "(",")",".","=","#","-","+","\\","/",":","~","@","?","*","$",">>"] + [str(i) for i in range(10)]

def build_vocab(src: List[str], tgt: List[str]) -> List[str]:
    return sorted(set(src) | set(tgt) | set(BASE_VOCAB))

# --- 2) Utility: weighted sampling ---
def _normalize(weights: Dict[str, float]) -> Dict[str, float]:
    s = sum(max(0.0, v) for v in weights.values())
    if s <= 0:
        raise ValueError(f"Non-positive total weight: {weights}")
    return {k: max(0.0, v) / s for k, v in weights.items()}

def weighted_choice(weights: Dict[str, float], rng: random.Random) -> Tuple[str, Dict[str, float]]:
    p = _normalize(weights)
    r = rng.random()
    cum = 0.0
    for k, w in p.items():
        cum += w
        if r <= cum:
            return k, p
    # fallback
    last = next(reversed(p))
    return last, p

# --- 3) Apply one edit action (INSERT/DELETE/REPLACE) ---
def apply_action(tokens: List[str], act: Dict[str, Any]) -> List[str]:
    op = act["op"]
    pos = act["pos"]

    if op == "INSERT":
        tokens.insert(pos, act["token"])
    elif op == "DELETE":
        tokens.pop(pos)
    elif op == "REPLACE":
        tokens[pos] = act["to"]
    else:
        raise ValueError(f"Unknown op: {op}")
    return tokens

def replay_actions(input_smiles: str, actions: List[Dict[str, Any]]) -> str:
    cur = tokenize(input_smiles)
    for a in actions:
        cur = apply_action(cur, a)
    return "".join(cur)

# --- 4) Repair (project current tokens back to target) using edit-distance DP ---
def dp_edit_distance(src: List[str], tgt: List[str]) -> List[List[int]]:
    n, m = len(src), len(tgt)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if src[i - 1] == tgt[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,      # delete
                dp[i][j - 1] + 1,      # insert
                dp[i - 1][j - 1] + cost  # match/replace
            )
    return dp

def backtrace_alignment_ops(
    src: List[str],
    tgt: List[str],
    dp: List[List[int]],
    rng: Optional[random.Random] = None,
    stochastic_ties: bool = False
) -> List[Tuple[str, Optional[int], Optional[int]]]:
    """
    Returns a forward alignment op list: (MATCH/REPLACE/DELETE/INSERT, src_index, tgt_index)
    """
    i, j = len(src), len(tgt)
    rev = []
    while i > 0 or j > 0:
        candidates = []

        if i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            candidates.append(("DELETE", i - 1, None))
        if j > 0 and dp[i][j] == dp[i][j - 1] + 1:
            candidates.append(("INSERT", None, j - 1))
        if i > 0 and j > 0:
            cost = 0 if src[i - 1] == tgt[j - 1] else 1
            if dp[i][j] == dp[i - 1][j - 1] + cost:
                candidates.append(("MATCH" if cost == 0 else "REPLACE", i - 1, j - 1))

        if not candidates:
            raise RuntimeError("Backtrace failed.")

        if stochastic_ties and rng is not None and len(candidates) > 1:
            pick = rng.choice(candidates)
        else:
            # deterministic priority: MATCH/REPLACE > DELETE > INSERT
            prio = {"MATCH": 0, "REPLACE": 0, "DELETE": 1, "INSERT": 2}
            pick = sorted(candidates, key=lambda x: prio[x[0]])[0]

        rev.append(pick)
        typ, si, tj = pick
        if typ in ("MATCH", "REPLACE"):
            i -= 1
            j -= 1
        elif typ == "DELETE":
            i -= 1
        else:  # INSERT
            j -= 1

    return rev[::-1]

def alignment_to_actions(
    aln_ops: List[Tuple[str, Optional[int], Optional[int]]],
    src_tokens: List[str],
    tgt_tokens: List[str]
) -> List[Dict[str, Any]]:
    """
    Convert alignment ops into executable actions with *current* indices.
    """
    cur = src_tokens[:]
    pos = 0
    actions = []

    for typ, si, tj in aln_ops:
        if typ == "MATCH":
            pos += 1

        elif typ == "INSERT":
            tok = tgt_tokens[tj]
            actions.append({"op": "INSERT", "pos": pos, "token": tok})
            cur.insert(pos, tok)
            pos += 1

        elif typ == "DELETE":
            tok = cur[pos]
            actions.append({"op": "DELETE", "pos": pos, "token": tok})
            cur.pop(pos)

        elif typ == "REPLACE":
            old = cur[pos]
            new = tgt_tokens[tj]
            actions.append({"op": "REPLACE", "pos": pos, "from": old, "to": new})
            cur[pos] = new
            pos += 1

        else:
            raise ValueError(f"Unknown alignment op: {typ}")

    return actions

def repair_script(
    cur: List[str],
    tgt: List[str],
    rng: Optional[random.Random],
    stochastic_ties: bool
) -> List[Dict[str, Any]]:
    dp = dp_edit_distance(cur, tgt)
    aln = backtrace_alignment_ops(cur, tgt, dp, rng=rng, stochastic_ties=stochastic_ties)
    return alignment_to_actions(aln, cur, tgt)

# --- 5) Noise step: token also random, allow deviation ---
def sample_noise_action(
    cur: List[str],
    vocab: List[str],
    rng: random.Random,
    op_weights: Dict[str, float]
) -> Tuple[Dict[str, Any], Dict[str, float]]:
    op, op_p = weighted_choice(op_weights, rng)

    if len(cur) == 0 and op in ("DELETE", "REPLACE"):
        op = "INSERT"
        op_p = {"INSERT": 1.0, "DELETE": 0.0, "REPLACE": 0.0}

    if op == "INSERT":
        pos = rng.randint(0, len(cur))
        tok = rng.choice(vocab)
        return {"op": "INSERT", "pos": pos, "token": tok}, op_p

    if op == "DELETE":
        pos = rng.randint(0, len(cur) - 1)
        return {"op": "DELETE", "pos": pos, "token": cur[pos]}, op_p

    if op == "REPLACE":
        pos = rng.randint(0, len(cur) - 1)
        tok = rng.choice(vocab)
        return {"op": "REPLACE", "pos": pos, "from": cur[pos], "to": tok}, op_p

    raise ValueError(op)

# --- 6) Main: diffusion-like trajectory (noise + guided drift + final projection) ---
def diffusion_like_bridge(
    input_smiles: str,
    output_smiles: str,
    noise_step_prob: float = 0.40,
    noise_op_weights: Optional[Dict[str, float]] = None,
    max_steps: int = 60,
    seed: int = 7
) -> Tuple[List[Dict[str, Any]], str]:
    """
    - With prob noise_step_prob: do a NOISE edit (token random, pos random) -> can deviate from target.
    - Otherwise: do one GUIDED step (first action of a minimal repair script; tie-breaking can be random).
    - Finally: force a deterministic REPAIR to guarantee endpoint == output_smiles.

    Positions are always *current* token indices (indices auto-update after each step).
    """
    if noise_op_weights is None:
        noise_op_weights = {"INSERT": 0.4, "DELETE": 0.3, "REPLACE": 0.3}

    src = tokenize(input_smiles)
    tgt = tokenize(output_smiles)
    vocab = build_vocab(src, tgt)
    rng = random.Random(seed)

    cur = src[:]
    actions: List[Dict[str, Any]] = []

    # stochastic trajectory
    for _ in range(max_steps):
        if cur == tgt:
            break

        if rng.random() < noise_step_prob:
            act, op_p = sample_noise_action(cur, vocab, rng, noise_op_weights)
            act = dict(act)
            act["mode"] = "NOISE"
            cur = apply_action(cur, act)
            actions.append(act)
        else:
            # guided drift (random tie-breaking for variety)
            rs = repair_script(cur, tgt, rng=rng, stochastic_ties=True)
            if not rs:
                break
            act = dict(rs[0])  # take one step
            act["mode"] = "GUIDED"
            cur = apply_action(cur, act)
            actions.append(act)

    # hard projection to endpoint (deterministic)
    if cur != tgt:
        rs = repair_script(cur, tgt, rng=None, stochastic_ties=False)
        for act0 in rs:
            act = dict(act0)
            act["mode"] = "REPAIR"
            cur = apply_action(cur, act)
            actions.append(act)

    final_smiles = "".join(cur)
    if final_smiles != output_smiles:
        raise AssertionError(
            "Final SMILES mismatch!\n"
            f"Expected: {output_smiles}\n"
            f"Got:      {final_smiles}"
        )
    return actions, final_smiles

def actions_to_thinking(
    actions: List[Dict[str, Any]],
    max_lines: int = 120,
    *,
    one_indexed: bool = False,
    show_mode: bool = False
) -> str:
    return actions_to_natural_language(
        actions,
        max_lines=max_lines,
        one_indexed=one_indexed,
        show_mode=show_mode
    )

def action_to_sentence(
    a: Dict[str, Any],
    *,
    one_indexed: bool = False,
    show_mode: bool = False
) -> str:
    pos = a["pos"] + (1 if one_indexed else 0)
    prefix = f"[{a.get('mode', '?')}] " if show_mode else ""

    op = a["op"]
    if op == "INSERT":
        tok = a.get("token", "?")
        return f"{prefix}insert {tok} at position {pos}"
    if op == "DELETE":
        tok = a.get("token", "?")
        return f"{prefix}delete {tok} at position {pos}"
    if op == "REPLACE":
        old = a.get("from", "?")
        new = a.get("to", "?")
        return f"{prefix}replace {old} with {new} at position {pos}"
    return f"{prefix}unknown op={op} at position {pos}"

def actions_to_natural_language(
    actions: List[Dict[str, Any]],
    max_lines: int = 120,
    *,
    one_indexed: bool = False,
    show_mode: bool = False
) -> str:
    lines = [
        action_to_sentence(a, one_indexed=one_indexed, show_mode=show_mode)
        for a in actions[:max_lines]
    ]
    if len(actions) > max_lines:
        lines.append(f"... ({len(actions) - max_lines} more steps)")
    return "\n".join(lines)

# --- 7) Test with your data ---
if __name__ == "__main__":
    input_smiles  = "COc1ccccc1S[C@@H](C#N)Cc1ccc(C)cc1"
    output_smiles = "COc1ccccc1S[C@@H](C#N)Cn1cc(-c2ccc(C)cc2)nc1C"

    actions, final_smiles = diffusion_like_bridge(
        input_smiles,
        output_smiles,
        noise_step_prob=0.55,  # Higher means more deviation/generative feel, but final repair still reaches output.
        noise_op_weights={"INSERT": 0.45, "DELETE": 0.25, "REPLACE": 0.30},
        max_steps=40,
        seed=42
    )

    print("=== Diffusion-like edit trajectory (token random allowed) ===")
    print(actions_to_thinking(actions, max_lines=200))
    print("\nFinal:", final_smiles)

    replayed = replay_actions(input_smiles, actions)
    print("Replay:", replayed)

    if final_smiles == output_smiles and replayed == output_smiles:
        print("\nPASS: final == target and replay == target")
    else:
        print("\nFAIL")
