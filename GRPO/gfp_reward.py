import json
import os
import re
from collections import OrderedDict
from typing import Any, Dict, Iterable, List, Optional, Tuple


AA_ALPHABET = set("ACDEFGHIKLMNPQRSTVWY")

THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
WT_RE = re.compile(r"WT sequence:\\s*([A-Z]+)")

ACTION_SPLIT_RE = re.compile(r"[;\\n]+")
REPLACE_RE = re.compile(
    r"replace\\s+([A-Za-z])\\s+to\\s+([A-Za-z])\\s+at\\s+position\\s+(\\d+)",
    re.IGNORECASE,
)
REMOVE_RE = re.compile(r"remove\\s+([A-Za-z])\\s+at\\s+position\\s+(\\d+)", re.IGNORECASE)
ADD_RE = re.compile(r"add\\s+([A-Za-z])\\s+at\\s+position\\s+(\\d+)", re.IGNORECASE)


def _get_env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _get_env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


FORMAT_WEIGHT = _get_env_float("GFP_FORMAT_WEIGHT", 0.1)
CONSISTENCY_WEIGHT = _get_env_float("GFP_CONSISTENCY_WEIGHT", 1.0)
FLUORESCENCE_WEIGHT = _get_env_float("GFP_FLUORESCENCE_WEIGHT", 1.0)
FLUORESCENCE_SCALE = _get_env_float("GFP_FLUORESCENCE_SCALE", 1.0)

REWARD_BATCH_SIZE = _get_env_int("GFP_REWARD_BATCH_SIZE", 8)
REWARD_CACHE_SIZE = _get_env_int("GFP_REWARD_CACHE_SIZE", 4096)

SAPROT_BASE_ID = os.getenv("GFP_SAPROT_BASE_ID", "westlake-repl/SaProt_650M_AF2")
SAPROT_ADAPTER_ID = os.getenv("GFP_SAPROT_ADAPTER_ID", "SaProtHub/Model-Fluorescence-650M")
SAPROT_CACHE_ROOT = os.getenv(
    "GFP_SAPROT_CACHE_ROOT",
    os.path.expanduser("~/.cache/stride/models"),
)


_TOKENIZER = None
_MODEL = None
_DEVICE = None
_CACHE: "OrderedDict[str, float]" = OrderedDict()


def _messages_to_prompt(messages: Any) -> str:
    if not messages:
        return ""
    if isinstance(messages, dict):
        return str(messages.get("content", ""))
    if isinstance(messages, list):
        parts = []
        for msg in messages:
            if isinstance(msg, dict):
                content = msg.get("content")
                if content:
                    parts.append(content)
            else:
                parts.append(str(msg))
        return "\n".join(parts)
    return str(messages)


def _prompt_from_sample(sample: Optional[Dict[str, Any]]) -> str:
    if not sample:
        return ""
    if "messages" in sample:
        return _messages_to_prompt(sample.get("messages"))
    system = sample.get("system")
    conversations = sample.get("conversations") or []
    user_prompt = ""
    for msg in conversations:
        role = msg.get("from") or msg.get("role")
        if role in {"human", "user"}:
            user_prompt = msg.get("value") or msg.get("content") or ""
            break
    if system and user_prompt:
        return f"{system}\n{user_prompt}"
    return user_prompt or system or ""


def _resolve_samples(kwargs: Dict[str, Any]) -> Iterable[Optional[Dict[str, Any]]]:
    for key in ("data_samples", "samples", "batch"):
        value = kwargs.get(key)
        if isinstance(value, list):
            return value
    return []


def _strip_code_fences(text: str) -> str:
    if "```" not in text:
        return text
    lines = text.strip().splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        return "\n".join(lines).strip()
    return text


def _extract_json_block(text: str) -> Optional[str]:
    if not text:
        return None
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for idx in range(start, len(text)):
        char = text[idx]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : idx + 1]
    return None


def extract_content(completion: str) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    if not completion:
        return None, None
    think_match = THINK_RE.search(completion)
    think_content = think_match.group(1).strip() if think_match else None
    remainder = completion[think_match.end() :] if think_match else completion
    remainder = _strip_code_fences(remainder.strip())
    json_block = _extract_json_block(remainder)
    if not json_block and think_match:
        json_block = _extract_json_block(completion[think_match.start() :])
    json_obj = None
    if json_block:
        try:
            json_obj = json.loads(json_block)
        except Exception:
            json_obj = None
    return think_content, json_obj


def _is_valid_sequence(seq: str) -> bool:
    if not seq or not isinstance(seq, str):
        return False
    return all(residue in AA_ALPHABET for residue in seq)


def parse_actions(think_text: Optional[str]) -> Optional[List[Dict[str, Any]]]:
    if not think_text:
        return None
    parts = [part.strip() for part in ACTION_SPLIT_RE.split(think_text) if part.strip()]
    if not parts:
        return None
    actions: List[Dict[str, Any]] = []
    for part in parts:
        part = part.lstrip("-*").strip().rstrip(".")
        match = REPLACE_RE.fullmatch(part)
        if match:
            old = match.group(1).upper()
            new = match.group(2).upper()
            pos = int(match.group(3))
            if old not in AA_ALPHABET or new not in AA_ALPHABET or pos <= 0:
                return None
            actions.append({"op": "sub", "old": old, "new": new, "pos": pos})
            continue
        match = REMOVE_RE.fullmatch(part)
        if match:
            old = match.group(1).upper()
            pos = int(match.group(2))
            if old not in AA_ALPHABET or pos <= 0:
                return None
            actions.append({"op": "del", "old": old, "pos": pos})
            continue
        match = ADD_RE.fullmatch(part)
        if match:
            new = match.group(1).upper()
            pos = int(match.group(2))
            if new not in AA_ALPHABET or pos <= 0:
                return None
            actions.append({"op": "ins", "new": new, "pos": pos})
            continue
        return None
    return actions


def apply_edits_to_wt(wt_seq: str, actions: List[Dict[str, Any]]) -> Optional[str]:
    seq_list = list(wt_seq)
    last_pos = float("inf")
    last_op = None
    insert_offsets: Dict[int, int] = {}
    for act in actions:
        pos = act["pos"]
        if pos > last_pos:
            return None
        if pos == last_pos and (act["op"] != "ins" or last_op != "ins"):
            return None
        last_pos = pos
        last_op = act["op"]

        idx = pos - 1
        if act["op"] in {"sub", "del"}:
            if idx < 0 or idx >= len(seq_list):
                return None
            if seq_list[idx] != act["old"]:
                return None
        if act["op"] == "sub":
            seq_list[idx] = act["new"]
        elif act["op"] == "del":
            seq_list.pop(idx)
        elif act["op"] == "ins":
            if pos > len(seq_list) + 1:
                return None
            offset = insert_offsets.get(pos, 0)
            insert_idx = idx + offset
            if insert_idx < 0:
                return None
            if insert_idx > len(seq_list):
                insert_idx = len(seq_list)
            seq_list.insert(insert_idx, act["new"])
            insert_offsets[pos] = offset + 1
    return "".join(seq_list)


def extract_wt_from_prompt(prompt: str) -> str:
    match = WT_RE.search(prompt)
    if match:
        return match.group(1).strip()
    return ""


def _get_device() -> str:
    device = os.getenv("GFP_REWARD_DEVICE")
    if device:
        return device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _load_fluorescence_model():
    global _TOKENIZER, _MODEL, _DEVICE
    if _MODEL is not None and _TOKENIZER is not None and _DEVICE is not None:
        return _TOKENIZER, _MODEL, _DEVICE
    import torch
    from transformers import AutoConfig, AutoModelForSequenceClassification, EsmTokenizer
    from peft import PeftModel

    base_dir = os.path.join(SAPROT_CACHE_ROOT, "saprot_base")
    adapter_dir = os.path.join(SAPROT_CACHE_ROOT, "saprot_adapter")
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(adapter_dir, exist_ok=True)

    tokenizer = EsmTokenizer.from_pretrained(SAPROT_BASE_ID, cache_dir=base_dir)
    config = AutoConfig.from_pretrained(SAPROT_BASE_ID, cache_dir=base_dir)
    config.num_labels = 1
    config.problem_type = "regression"

    base_model = AutoModelForSequenceClassification.from_pretrained(
        SAPROT_BASE_ID,
        config=config,
        cache_dir=base_dir,
    )

    model = PeftModel.from_pretrained(
        base_model,
        SAPROT_ADAPTER_ID,
        cache_dir=adapter_dir,
    )

    device = torch.device(_get_device())
    model.to(device)
    if device.type == "cuda" and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model.eval()

    _TOKENIZER = tokenizer
    _MODEL = model
    _DEVICE = device
    return _TOKENIZER, _MODEL, _DEVICE


def _format_saprot_sequence(seq: str) -> str:
    if "#" in seq:
        return seq
    return "".join([aa + "#" for aa in seq])


def _predict_batch(sequences: List[str]) -> List[float]:
    if not sequences:
        return []
    tokenizer, model, device = _load_fluorescence_model()
    import torch

    formatted = [_format_saprot_sequence(seq) for seq in sequences]
    enc = tokenizer(
        formatted,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=1024,
    )
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        outputs = model(**enc)
        logits = outputs.logits.squeeze(-1)
    if logits.ndim == 0:
        logits = logits.unsqueeze(0)
    return [float(val) for val in logits.detach().cpu().tolist()]


def _predict_fluorescence(sequences: List[str]) -> List[float]:
    preds: List[float] = []
    for start in range(0, len(sequences), REWARD_BATCH_SIZE):
        batch = sequences[start : start + REWARD_BATCH_SIZE]
        preds.extend(_predict_batch(batch))
    return preds


def _cache_get(seq: str) -> Optional[float]:
    if seq in _CACHE:
        _CACHE.move_to_end(seq)
        return _CACHE[seq]
    return None


def _cache_set(seq: str, value: float) -> None:
    _CACHE[seq] = value
    _CACHE.move_to_end(seq)
    if len(_CACHE) > REWARD_CACHE_SIZE:
        _CACHE.popitem(last=False)


def _predict_with_cache(sequences: List[str]) -> List[float]:
    preds: List[Optional[float]] = [None] * len(sequences)
    missing: List[str] = []
    missing_indices: List[int] = []
    for idx, seq in enumerate(sequences):
        cached = _cache_get(seq)
        if cached is not None:
            preds[idx] = cached
        else:
            missing.append(seq)
            missing_indices.append(idx)
    if missing:
        new_preds = _predict_fluorescence(missing)
        for idx, seq, pred in zip(missing_indices, missing, new_preds):
            preds[idx] = pred
            _cache_set(seq, pred)
    return [float(pred) if pred is not None else 0.0 for pred in preds]


def compute_gfp_rewards(prompts: List[str], completions: List[str], **kwargs: Any) -> List[float]:
    rewards: List[float] = [0.0 for _ in completions]
    if not completions:
        return rewards

    samples = list(_resolve_samples(kwargs))
    format_scores: List[float] = []
    consistency_scores: List[float] = []
    mutant_seqs: List[str] = []
    wt_seqs: List[str] = []
    valid_indices: List[int] = []

    for idx, (prompt, completion) in enumerate(zip(prompts, completions)):
        sample = samples[idx] if idx < len(samples) else None
        if not prompt:
            prompt = _prompt_from_sample(sample)

        think, json_obj = extract_content(completion or "")
        actions = parse_actions(think)
        if not think or json_obj is None or actions is None:
            format_scores.append(0.0)
            consistency_scores.append(0.0)
            continue

        if not isinstance(json_obj, dict) or set(json_obj.keys()) != {"protein"}:
            format_scores.append(0.0)
            consistency_scores.append(0.0)
            continue

        mutant = json_obj.get("protein")
        if not isinstance(mutant, str) or not _is_valid_sequence(mutant):
            format_scores.append(0.0)
            consistency_scores.append(0.0)
            continue

        format_scores.append(1.0)

        wt_seq = extract_wt_from_prompt(prompt)
        if not wt_seq:
            consistency_scores.append(0.0)
            continue

        reconstructed = apply_edits_to_wt(wt_seq, actions)
        if reconstructed is None or reconstructed != mutant:
            consistency_scores.append(0.0)
            continue

        consistency_scores.append(1.0)
        if FLUORESCENCE_WEIGHT != 0.0:
            mutant_seqs.append(mutant)
            wt_seqs.append(wt_seq)
            valid_indices.append(idx)

    if FLUORESCENCE_WEIGHT != 0.0 and mutant_seqs:
        mut_scores = _predict_with_cache(mutant_seqs)
        wt_scores = _predict_with_cache(wt_seqs)
    else:
        mut_scores = []
        wt_scores = []

    score_map: Dict[int, float] = {}
    for idx, mut_score, wt_score in zip(valid_indices, mut_scores, wt_scores):
        score_map[idx] = (mut_score - wt_score) * FLUORESCENCE_SCALE

    for idx in range(len(completions)):
        format_score = format_scores[idx] if idx < len(format_scores) else 0.0
        consistency_score = consistency_scores[idx] if idx < len(consistency_scores) else 0.0
        fluorescence_score = score_map.get(idx, 0.0)
        rewards[idx] = (
            format_score * FORMAT_WEIGHT
            + consistency_score * CONSISTENCY_WEIGHT
            + fluorescence_score * FLUORESCENCE_WEIGHT
        )

    return rewards
