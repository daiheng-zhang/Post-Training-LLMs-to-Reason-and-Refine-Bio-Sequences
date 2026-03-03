import os
import sys
from typing import Any, List, Optional

from swift.plugin import ORM, orms

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from chem_reward import compute_chem_rewards


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


def _normalize_prompts(messages: Any, completions_len: int) -> List[str]:
    if isinstance(messages, list) and messages and isinstance(messages[0], list):
        prompts = [_messages_to_prompt(msgs) for msgs in messages]
        if len(prompts) == completions_len:
            return prompts
        if prompts and completions_len % len(prompts) == 0:
            repeat = completions_len // len(prompts)
            expanded: List[str] = []
            for prompt in prompts:
                expanded.extend([prompt] * repeat)
            return expanded
        return (prompts * completions_len)[:completions_len]
    prompt = _messages_to_prompt(messages)
    return [prompt for _ in range(completions_len)]


def _normalize_task_ids(task_id: Any, completions_len: int) -> Optional[List[int]]:
    if task_id is None:
        return None
    if isinstance(task_id, list):
        ids = [int(value) for value in task_id]
        if len(ids) == completions_len:
            return ids
        if ids and completions_len % len(ids) == 0:
            repeat = completions_len // len(ids)
            expanded: List[int] = []
            for value in ids:
                expanded.extend([value] * repeat)
            return expanded
        return (ids * completions_len)[:completions_len]
    return [int(task_id) for _ in range(completions_len)]


class ChemReward(ORM):
    def __call__(self, completions: List[str], messages=None, task_id=None, **kwargs: Any) -> List[float]:
        if completions is None:
            completions = []
        prompts = _normalize_prompts(messages, len(completions))
        task_ids = _normalize_task_ids(task_id, len(completions))
        filtered_kwargs = {key: value for key, value in kwargs.items() if key != "task_ids"}
        return compute_chem_rewards(
            prompts=prompts,
            completions=completions,
            task_ids=task_ids,
            **filtered_kwargs,
        )


orms["chem_reward"] = ChemReward


def _patch_ref_adapter_activation() -> None:
    try:
        from swift.llm.train import rlhf as rlhf_mod
    except Exception:
        return
    if getattr(rlhf_mod, "_patched_ref_adapter_activation", False):
        return
    original_prepare = rlhf_mod.SwiftRLHF.prepare_model

    def _prepare_with_default_adapter(cls, args, model, *, template=None, train_dataset=None, task_type=None):
        model = original_prepare(
            args,
            model,
            template=template,
            train_dataset=train_dataset,
            task_type=task_type,
        )
        if getattr(args, "ref_adapters", None) and hasattr(model, "set_adapter"):
            try:
                model.set_adapter("default")
            except Exception:
                pass
        return model

    rlhf_mod.SwiftRLHF.prepare_model = classmethod(_prepare_with_default_adapter)
    rlhf_mod._patched_ref_adapter_activation = True


_patch_ref_adapter_activation()
