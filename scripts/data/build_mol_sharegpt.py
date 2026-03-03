#!/usr/bin/env python3
import argparse
import json
import math
import os

try:
    import pyarrow.parquet as pq
except Exception:  # pragma: no cover - optional dependency for CLI help
    pq = None


SYSTEM_PROMPT = (
    "You are a molecular SMILES editing assistant.\n\n"
    "You will be given:\n"
    "- a natural-language optimization instruction\n"
    "- an input_smiles\n\n"
    "Your goal is to produce a chemically valid output_smiles that better satisfies the instruction "
    "while staying as similar as possible to the input_smiles. Preserve stereochemistry unless a change is necessary.\n\n"
    "### Action Constraints\n"
    "The edit trajectory is strictly limited to the following three operations (use ONLY these action keywords):\n"
    "- INSERT\n"
    "- DELETE\n"
    "- REPLACE\n"
    "Do not use any other action keywords.\n\n"
    "### Edit Line Grammar (must match exactly)\n"
    "- INSERT <token> at position <int>\n"
    "- DELETE <token> at position <int>\n"
    "- REPLACE <old_token> with <new_token> at position <int>\n\n"
    "Edits are applied sequentially. Positions refer to the current SMILES string after previous edits.\n\n"
    "### Response Format\n"
    "- Put the full edit trajectory inside <think>...</think>, one operation per line.\n"
    "- After </think>, output ONLY the final output_smiles as a single SMILES string on its own line.\n"
    "- Do not output the trajectory outside <think>.\n"
    "- Do not add any extra commentary."
)


VERB_MAP = {
    "insert": "INSERT",
    "delete": "DELETE",
    "replace": "REPLACE",
    "inserted": "INSERT",
    "deleted": "DELETE",
    "replaced": "REPLACE",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert edit_traj parquet files into ShareGPT JSONL format "
            "for LLaMA-Factory training."
        )
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing *.edit_traj.parquet files.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for JSONL files (default: parent of input-dir).",
    )
    parser.add_argument(
        "--train-name",
        default="train-00000-of-00001.edit_traj.parquet",
        help="Train parquet filename inside input-dir.",
    )
    parser.add_argument(
        "--val-name",
        default="validation-00000-of-00001.edit_traj.parquet",
        help="Validation parquet filename inside input-dir.",
    )
    parser.add_argument(
        "--train-output",
        default="train_sharegpt_edit_traj.jsonl",
        help="Output JSONL filename for training data.",
    )
    parser.add_argument(
        "--val-output",
        default="val_sharegpt_edit_traj.jsonl",
        help="Output JSONL filename for validation data.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10000,
        help="Rows per parquet batch.",
    )
    parser.add_argument(
        "--include-meta",
        action="store_true",
        help="Include the original parquet row as a 'meta' field.",
    )
    return parser.parse_args()


def normalize_edit_traj(edit_traj: str) -> str:
    if not edit_traj:
        return ""
    lines = []
    for raw in str(edit_traj).splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = line.split(maxsplit=1)
        verb = parts[0].lower() if parts else ""
        rest = parts[1] if len(parts) > 1 else ""
        mapped = VERB_MAP.get(verb)
        if mapped:
            line = mapped + (f" {rest}" if rest else "")
        lines.append(line)
    return "\n".join(lines)


def sanitize_json(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: sanitize_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [sanitize_json(v) for v in value]
    return value


def build_sharegpt_record(rec: dict, include_meta: bool) -> dict:
    prompt = rec.get("prompt", "")
    input_smiles = rec.get("input_smiles", "")
    output_smiles = rec.get("output_smiles", "")
    edit_traj = normalize_edit_traj(rec.get("edit_traj", ""))

    user_prompt = (
        f"{prompt}\n\n"
        f"input_smiles:\n{input_smiles}\n\n"
        "Please return:\n"
        "- edit_traj (using only INSERT/DELETE/REPLACE)\n"
        "- output_smiles"
    )

    assistant_value = f"<think>\n{edit_traj}\n</think>\n{output_smiles}"

    record = {
        "task_id": rec.get("task_id"),
        "system": SYSTEM_PROMPT,
        "conversations": [
            {"from": "human", "value": user_prompt},
            {"from": "gpt", "value": assistant_value},
        ],
    }

    if include_meta:
        record["meta"] = sanitize_json(rec)

    return record


def convert_parquet(
    input_path: str,
    output_path: str,
    batch_size: int,
    include_meta: bool,
) -> int:
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Parquet file not found: {input_path}")

    parquet_file = pq.ParquetFile(input_path)
    schema_names = parquet_file.schema.names
    required = ["prompt", "input_smiles", "output_smiles", "edit_traj"]
    missing = [name for name in required if name not in schema_names]
    if missing:
        raise ValueError(f"Missing columns in parquet: {missing}")

    if include_meta:
        columns = None
    else:
        columns = list(required)
        if "task_id" in schema_names:
            columns.append("task_id")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    count = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for batch in parquet_file.iter_batches(
            columns=columns, batch_size=batch_size
        ):
            rows = batch.to_pylist()
            for rec in rows:
                sharegpt = build_sharegpt_record(rec, include_meta=include_meta)
                f.write(json.dumps(sharegpt, ensure_ascii=True) + "\n")
                count += 1
    return count


def main() -> None:
    args = parse_args()
    if pq is None:
        raise ModuleNotFoundError(
            "The 'pyarrow' package is required. Install dependencies from requirements/train.txt."
        )
    input_dir = args.input_dir
    output_dir = args.output_dir or os.path.dirname(input_dir.rstrip(os.sep))

    train_input = os.path.join(input_dir, args.train_name)
    val_input = os.path.join(input_dir, args.val_name)
    train_output = os.path.join(output_dir, args.train_output)
    val_output = os.path.join(output_dir, args.val_output)

    train_count = convert_parquet(
        train_input,
        train_output,
        batch_size=args.batch_size,
        include_meta=args.include_meta,
    )
    val_count = convert_parquet(
        val_input,
        val_output,
        batch_size=args.batch_size,
        include_meta=args.include_meta,
    )

    print(f"Wrote {train_count} records to {train_output}")
    print(f"Wrote {val_count} records to {val_output}")


if __name__ == "__main__":
    main()
