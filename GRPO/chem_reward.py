import os
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, Descriptors, QED, rdMolDescriptors


def _get_env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


STABILITY_PENALTY = _get_env_float("CHEM_STABILITY_PENALTY", -0.5)

STABILITY_THRESHOLDS = {
    "logp": 0.5,
    "qed": 0.1,
    "tpsa": 10.0,
    "hba": 1.0,
    "hbd": 1.0,
}


TASK_RULES = {
    101: {"objs": [("logp", "minimize")], "strict": [0.5], "loose": [0.0]},
    102: {"objs": [("logp", "maximize")], "strict": [0.5], "loose": [0.0]},
    103: {"objs": [("qed", "maximize")], "strict": [0.1], "loose": [0.0]},
    104: {"objs": [("qed", "minimize")], "strict": [0.1], "loose": [0.0]},
    105: {"objs": [("tpsa", "minimize")], "strict": [10.0], "loose": [0.0]},
    106: {"objs": [("tpsa", "maximize")], "strict": [10.0], "loose": [0.0]},
    107: {"objs": [("hba", "maximize")], "strict": [1.0], "loose": [0.0]},
    108: {"objs": [("hbd", "maximize")], "strict": [1.0], "loose": [0.0]},
    201: {
        "objs": [("logp", "minimize"), ("hba", "maximize")],
        "strict": [0.5, 1.0],
        "loose": [0.0, 0.0],
    },
    202: {
        "objs": [("logp", "maximize"), ("hba", "maximize")],
        "strict": [0.5, 1.0],
        "loose": [0.0, 0.0],
    },
    203: {
        "objs": [("logp", "minimize"), ("hbd", "maximize")],
        "strict": [0.5, 1.0],
        "loose": [0.0, 0.0],
    },
    204: {
        "objs": [("logp", "maximize"), ("hbd", "maximize")],
        "strict": [0.5, 1.0],
        "loose": [0.0, 0.0],
    },
    205: {
        "objs": [("logp", "minimize"), ("tpsa", "minimize")],
        "strict": [0.5, 10.0],
        "loose": [0.0, 0.0],
    },
    206: {
        "objs": [("logp", "minimize"), ("tpsa", "maximize")],
        "strict": [0.5, 10.0],
        "loose": [0.0, 0.0],
    },
}

PROMPT_PATTERNS: List[Tuple[str, int]] = [
    ("more soluble in water and with more hydrogen bond acceptors", 201),
    ("less soluble in water and with more hydrogen bond acceptors", 202),
    ("more soluble in water and with more hydrogen bond donors", 203),
    ("less soluble in water and with more hydrogen bond donors", 204),
    ("more soluble in water and with higher permeability", 205),
    ("more soluble in water and with lower permeability", 206),
    ("more soluble in water", 101),
    ("less soluble in water", 102),
    ("more like a drug", 103),
    ("less like a drug", 104),
    ("higher permeability", 105),
    ("lower permeability", 106),
    ("more hydrogen bond acceptors", 107),
    ("more hydrogen bond donors", 108),
]


def extract_solution(completion_text: str) -> str:
    text_no_think = re.sub(r"<think>.*?</think>", "", completion_text, flags=re.DOTALL)
    json_match = re.search(r'{"smiles"\s*:\s*"([^"]+)"}', text_no_think)
    if json_match:
        return json_match.group(1).strip()
    lines = [line.strip() for line in text_no_think.splitlines() if line.strip()]
    return lines[-1] if lines else ""


def extract_input_smiles(prompt: str) -> Optional[str]:
    match = re.search(r"input_smiles:\s*([^\s]+)", prompt)
    if match:
        return match.group(1).strip()
    match = re.search(r"Input molecule \\(SMILES\\):\\s*\\n([^\\s]+)", prompt)
    if match:
        return match.group(1).strip()
    lines = [line.strip() for line in prompt.splitlines()]
    for idx, line in enumerate(lines):
        if line.startswith("input_smiles"):
            for j in range(idx + 1, len(lines)):
                if lines[j]:
                    return lines[j]
        if line.startswith("Input molecule (SMILES)"):
            for j in range(idx + 1, len(lines)):
                if lines[j]:
                    return lines[j]
    return None


def _extract_task_id_from_sample(sample: Dict[str, Any]) -> Optional[int]:
    for key in ("task_id", "task", "taskid"):
        if key in sample:
            try:
                return int(sample[key])
            except (TypeError, ValueError):
                return None
    for container_key in ("meta", "metadata", "reward_model", "extra"):
        container = sample.get(container_key)
        if isinstance(container, dict):
            for key in ("task_id", "task"):
                if key in container:
                    try:
                        return int(container[key])
                    except (TypeError, ValueError):
                        return None
    return None


def resolve_task_id(prompt: str, sample: Optional[Dict[str, Any]]) -> Optional[int]:
    if sample:
        task_id = _extract_task_id_from_sample(sample)
        if task_id is not None:
            return task_id
    match = re.search(r"task_id\s*[:=]\s*(\d+)", prompt, flags=re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None


def resolve_task_rule(prompt: str, task_id: Optional[int]) -> Dict[str, Any]:
    if task_id in TASK_RULES:
        return TASK_RULES[task_id]
    prompt_lower = prompt.lower()
    for phrase, mapped_id in PROMPT_PATTERNS:
        if phrase in prompt_lower:
            return TASK_RULES[mapped_id]
    return TASK_RULES[107]


def get_property(mol: Chem.Mol, name: str) -> float:
    if name == "hba":
        return float(rdMolDescriptors.CalcNumHBA(mol))
    if name == "hbd":
        return float(rdMolDescriptors.CalcNumHBD(mol))
    if name == "logp":
        return float(Descriptors.MolLogP(mol))
    if name == "qed":
        return float(QED.qed(mol))
    if name == "tpsa":
        return float(rdMolDescriptors.CalcTPSA(mol))
    raise ValueError(f"Unknown property: {name}")


def score_property(
    mol_in: Chem.Mol,
    mol_out: Chem.Mol,
    rule: Dict[str, Any],
) -> float:
    objs = rule["objs"]
    strict = rule["strict"]
    loose = rule["loose"]
    meet_strict = []
    meet_loose = []
    for (prop, direction), strict_thr, loose_thr in zip(objs, strict, loose):
        val_in = get_property(mol_in, prop)
        val_out = get_property(mol_out, prop)
        change = val_out - val_in
        if direction != "maximize":
            change = -change
        meet_strict.append(change >= strict_thr)
        meet_loose.append(change > loose_thr)
    if meet_strict and all(meet_strict):
        return 1.0
    if meet_loose and all(meet_loose):
        return 0.5
    return 0.0


def score_similarity(mol_in: Chem.Mol, mol_out: Chem.Mol) -> float:
    fp_in = AllChem.GetMorganFingerprintAsBitVect(mol_in, 2, nBits=2048)
    fp_out = AllChem.GetMorganFingerprintAsBitVect(mol_out, 2, nBits=2048)
    tanimoto = DataStructs.TanimotoSimilarity(fp_in, fp_out)
    if tanimoto > 0.65:
        return 1.0
    if tanimoto >= 0.4:
        return 0.5
    return 0.0


def score_stability_violations(
    mol_in: Chem.Mol,
    mol_out: Chem.Mol,
    rule: Dict[str, Any],
) -> int:
    target_props = {prop for prop, _ in rule.get("objs", [])}
    violations = 0
    for prop, threshold in STABILITY_THRESHOLDS.items():
        if prop in target_props:
            continue
        try:
            val_in = get_property(mol_in, prop)
            val_out = get_property(mol_out, prop)
        except Exception:
            continue
        if abs(val_out - val_in) >= threshold:
            violations += 1
    return violations


def _resolve_samples(kwargs: Dict[str, Any]) -> Iterable[Optional[Dict[str, Any]]]:
    for key in ("data_samples", "samples", "batch"):
        value = kwargs.get(key)
        if isinstance(value, list):
            return value
    return []


def _is_primary_process() -> bool:
    rank = os.environ.get("RANK")
    if rank is not None:
        try:
            return int(rank) == 0
        except ValueError:
            return False
    local_rank = os.environ.get("LOCAL_RANK")
    if local_rank is not None:
        try:
            return int(local_rank) == 0
        except ValueError:
            return False
    return True


def _maybe_log_wandb(metrics: Dict[str, float], step: Optional[int]) -> None:
    if not metrics or not _is_primary_process():
        return
    if os.environ.get("WANDB_MODE", "").strip().lower() == "disabled":
        return
    try:
        import wandb
    except Exception:
        return
    if wandb.run is None:
        return
    if step is None:
        wandb.log(metrics)
    else:
        wandb.log(metrics, step=step)


def compute_chem_rewards(prompts: List[str], completions: List[str], **kwargs: Any) -> List[float]:
    rewards = []
    validity_scores = []
    property_scores = []
    similarity_scores = []
    stability_scores = []
    stability_penalties = []
    samples = list(_resolve_samples(kwargs))
    task_ids = kwargs.get("task_ids")
    step = kwargs.get("global_step") or kwargs.get("step")
    step_value = None
    if step is not None:
        try:
            step_value = int(step)
        except (TypeError, ValueError):
            step_value = None

    for idx, (prompt, completion) in enumerate(zip(prompts, completions)):
        r_validity = 0.0
        r_property = 0.0
        r_similarity = 0.0
        n_violations = 0
        sample = samples[idx] if idx < len(samples) else None
        task_id = None
        if isinstance(task_ids, list) and idx < len(task_ids):
            task_id = task_ids[idx]
        if task_id is None:
            task_id = resolve_task_id(prompt, sample)
        hint_text = prompt
        if sample:
            for key in ("task_instruction", "instruction", "prompt", "task_prompt"):
                value = sample.get(key)
                if isinstance(value, str) and value:
                    hint_text = f"{hint_text}\n{value}"
        rule = resolve_task_rule(hint_text, task_id)

        input_smiles = extract_input_smiles(prompt)
        pred_smiles = extract_solution(completion)
        mol_out = Chem.MolFromSmiles(pred_smiles) if pred_smiles else None

        if mol_out is None:
            rewards.append(0.0)
            validity_scores.append(r_validity)
            property_scores.append(r_property)
            similarity_scores.append(r_similarity)
            stability_scores.append(n_violations)
            stability_penalties.append(0.0)
            continue

        r_validity = 1.0
        if not input_smiles:
            rewards.append(0.0)
            validity_scores.append(r_validity)
            property_scores.append(r_property)
            similarity_scores.append(r_similarity)
            stability_scores.append(n_violations)
            stability_penalties.append(0.0)
            continue

        mol_in = Chem.MolFromSmiles(input_smiles)
        if mol_in is None:
            rewards.append(0.0)
            validity_scores.append(r_validity)
            property_scores.append(r_property)
            similarity_scores.append(r_similarity)
            stability_scores.append(n_violations)
            stability_penalties.append(0.0)
            continue

        try:
            r_similarity = score_similarity(mol_in, mol_out)
        except Exception:
            r_similarity = 0.0

        try:
            r_property = score_property(mol_in, mol_out, rule)
        except Exception:
            r_property = 0.0

        try:
            n_violations = score_stability_violations(mol_in, mol_out, rule)
        except Exception:
            n_violations = 0

        r_stability_penalty = STABILITY_PENALTY if n_violations > 0 else 0.0
        final_reward = (r_validity * r_property * r_similarity) + r_stability_penalty
        rewards.append(final_reward)
        validity_scores.append(r_validity)
        property_scores.append(r_property)
        similarity_scores.append(r_similarity)
        stability_scores.append(n_violations)
        stability_penalties.append(r_stability_penalty)

    if rewards:
        _maybe_log_wandb(
            {
                "chem_reward/validity": sum(validity_scores) / len(validity_scores),
                "chem_reward/property": sum(property_scores) / len(property_scores),
                "chem_reward/similarity": sum(similarity_scores) / len(similarity_scores),
                "chem_reward/stability_violations": sum(stability_scores) / len(stability_scores),
                "chem_reward/stability_penalty": sum(stability_penalties)
                / len(stability_penalties),
                "chem_reward/final_reward": sum(rewards) / len(rewards),
            },
            step=step_value,
        )

    return rewards
