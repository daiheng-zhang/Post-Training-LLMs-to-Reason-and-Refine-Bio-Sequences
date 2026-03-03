import os
import sys
from typing import Any, List, Optional

from swift.plugin import ORM, orms

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from gfp_reward import compute_gfp_rewards


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


class GFPReward(ORM):
    def __call__(self, completions: List[str], messages=None, task_id=None, **kwargs: Any) -> List[float]:
        if completions is None:
            completions = []
        prompts = _normalize_prompts(messages, len(completions))
        task_ids = _normalize_task_ids(task_id, len(completions))
        filtered_kwargs = {key: value for key, value in kwargs.items() if key != "task_ids"}
        return compute_gfp_rewards(
            prompts=prompts,
            completions=completions,
            task_ids=task_ids,
            **filtered_kwargs,
        )


orms["gfp_reward"] = GFPReward
