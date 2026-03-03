#!/usr/bin/env python3
"""Convert ShareGPT records into ms-swift RLHF dataset format."""

import argparse
import json
import os
import random
from typing import Any, Dict, List, Optional, Tuple


def load_records(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as fin:
        text = fin.read()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        data = []
        with open(path, "r", encoding="utf-8") as fin:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                data.append(json.loads(line))

    if isinstance(data, dict):
        data = data.get("data", [data])
    if not isinstance(data, list):
        raise ValueError("Unsupported JSON format: expected a list or {'data': [...]}.")
    return data


def first_message(conversations: List[Dict[str, Any]], roles: List[str]) -> Optional[str]:
    for msg in conversations:
        role = msg.get("from") or msg.get("role")
        if role in roles:
            return msg.get("value") or msg.get("content")
    return None


def convert_record(item: Dict[str, Any], ability: str) -> Dict[str, Any]:
    system_prompt = item.get("system")
    conversations = item.get("conversations") or []
    if not system_prompt and conversations:
        system_prompt = first_message(conversations, ["system"]) or ""

    user_prompt = first_message(conversations, ["human", "user"])
    assistant_value = first_message(conversations, ["gpt", "assistant"]) or ""
    if not user_prompt:
        raise ValueError("Record is missing a user prompt.")

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})

    out = {
        "messages": messages,
        "ground_truth": assistant_value,
        "ability": ability,
    }
    if "task_id" in item:
        out["task_id"] = item["task_id"]
    return out


def split_records(records: List[Dict[str, Any]], val_ratio: float, seed: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if val_ratio <= 0:
        return records, []
    rng = random.Random(seed)
    indices = list(range(len(records)))
    rng.shuffle(indices)
    cutoff = int(len(indices) * (1 - val_ratio))
    train_idx = set(indices[:cutoff])
    train = [records[i] for i in range(len(records)) if i in train_idx]
    valid = [records[i] for i in range(len(records)) if i not in train_idx]
    return train, valid


def dump_jsonl(records: List[Dict[str, Any]], path: str) -> None:
    with open(path, "w", encoding="utf-8") as fout:
        for row in records:
            fout.write(json.dumps(row, ensure_ascii=True) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert ShareGPT JSON/JSONL to ms-swift JSONL.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-name", default="train.jsonl")
    parser.add_argument("--valid-name", default="valid.jsonl")
    parser.add_argument("--val-ratio", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--shuffle-before-limit", action="store_true")
    parser.add_argument("--ability", default="chemistry")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_records(args.input)

    if args.max_records and args.max_records > 0 and len(records) > args.max_records:
        if args.shuffle_before_limit:
            rng = random.Random(args.seed)
            rng.shuffle(records)
        records = records[: args.max_records]

    converted = [convert_record(item, args.ability) for item in records]
    train, valid = split_records(converted, args.val_ratio, args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    train_path = os.path.join(args.output_dir, args.train_name)
    valid_path = os.path.join(args.output_dir, args.valid_name)

    dump_jsonl(train, train_path)
    if valid:
        dump_jsonl(valid, valid_path)

    print(f"Wrote {len(train)} train records to {train_path}")
    if valid:
        print(f"Wrote {len(valid)} valid records to {valid_path}")


if __name__ == "__main__":
    main()
