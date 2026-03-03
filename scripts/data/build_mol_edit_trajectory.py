#!/usr/bin/env python3
import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Optional, Tuple

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except Exception:  # pragma: no cover - optional dependency for CLI help
    pa = None
    pq = None

from mol_edit_trajectory_utils import diffusion_like_bridge, actions_to_thinking


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Add an edit trajectory column to a parquet file by processing "
            "input/output SMILES pairs."
        )
    )
    parser.add_argument("--input", required=True, help="Input parquet path.")
    parser.add_argument(
        "--output",
        default=None,
        help="Output parquet path (default: <input>.<traj-col>.parquet).",
    )
    parser.add_argument(
        "--input-col",
        default="input_smiles",
        help="Column name for input SMILES.",
    )
    parser.add_argument(
        "--output-col",
        default="output_smiles",
        help="Column name for output SMILES.",
    )
    parser.add_argument(
        "--traj-col",
        default="edit_traj",
        help="New column name for the edit trajectory.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=5000,
        help="Rows per parquet batch.",
    )
    parser.add_argument(
        "--format",
        choices=["thinking", "actions_json"],
        default="thinking",
        help="Output format for the trajectory column.",
    )
    parser.add_argument(
        "--max-lines",
        type=int,
        default=1000000,
        help="Max lines to keep in thinking format.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Base RNG seed (row index is added for determinism).",
    )
    parser.add_argument(
        "--noise-step-prob",
        type=float,
        default=0.40,
        help="Probability of a noise edit step.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=60,
        help="Max stochastic steps before repair.",
    )
    parser.add_argument(
        "--show-mode",
        action="store_true",
        help="Include NOISE/GUIDED/REPAIR tags in thinking format.",
    )
    parser.add_argument(
        "--one-indexed",
        action="store_true",
        help="Use 1-based positions in thinking format.",
    )
    parser.add_argument(
        "--on-error",
        choices=["raise", "skip", "empty"],
        default="skip",
        help="How to handle trajectory failures.",
    )
    parser.add_argument(
        "--compression",
        default="snappy",
        help="Parquet compression codec.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of worker processes for trajectory generation.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output file if it exists.",
    )
    return parser.parse_args()


def default_output_path(input_path: str, traj_col: str) -> str:
    base, ext = os.path.splitext(input_path)
    if ext.lower() != ".parquet":
        return f"{input_path}.{traj_col}.parquet"
    return f"{base}.{traj_col}{ext}"


def build_traj(
    input_smiles: str,
    output_smiles: str,
    *,
    seed: int,
    noise_step_prob: float,
    max_steps: int,
    fmt: str,
    max_lines: int,
    show_mode: bool,
    one_indexed: bool,
) -> str:
    actions, _ = diffusion_like_bridge(
        input_smiles,
        output_smiles,
        noise_step_prob=noise_step_prob,
        max_steps=max_steps,
        seed=seed,
    )
    if fmt == "actions_json":
        return json.dumps(actions, ensure_ascii=True)
    return actions_to_thinking(
        actions,
        max_lines=max_lines,
        one_indexed=one_indexed,
        show_mode=show_mode,
    )


def _build_traj_worker(args: Tuple[str, str, int, float, int, str, int, bool, bool]) -> str:
    inp, out, seed, noise_step_prob, max_steps, fmt, max_lines, show_mode, one_indexed = args
    return build_traj(
        inp,
        out,
        seed=seed,
        noise_step_prob=noise_step_prob,
        max_steps=max_steps,
        fmt=fmt,
        max_lines=max_lines,
        show_mode=show_mode,
        one_indexed=one_indexed,
    )


def main() -> None:
    args = parse_args()
    if pa is None or pq is None:
        raise ModuleNotFoundError(
            "The 'pyarrow' package is required. Install dependencies from requirements/train.txt."
        )
    input_path = args.input
    output_path = args.output or default_output_path(input_path, args.traj_col)

    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Parquet file not found: {input_path}")

    if os.path.exists(output_path) and not args.overwrite:
        raise FileExistsError(f"Output already exists: {output_path}")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    parquet_file = pq.ParquetFile(input_path)
    schema_names = parquet_file.schema.names
    missing = [name for name in (args.input_col, args.output_col) if name not in schema_names]
    if missing:
        raise ValueError(f"Missing columns in parquet: {missing}")
    if args.traj_col in schema_names:
        raise ValueError(f"Trajectory column already exists: {args.traj_col}")

    writer: Optional[pq.ParquetWriter] = None
    error_count = 0
    error_samples: List[str] = []
    row_offset = 0
    num_workers = max(1, args.num_workers)

    executor: Optional[ProcessPoolExecutor] = None
    if num_workers > 1:
        executor = ProcessPoolExecutor(max_workers=num_workers)

    try:
        for batch in parquet_file.iter_batches(batch_size=args.batch_size):
            input_idx = batch.schema.get_field_index(args.input_col)
            output_idx = batch.schema.get_field_index(args.output_col)
            inputs = batch.column(input_idx).to_pylist()
            outputs = batch.column(output_idx).to_pylist()

            traj_values: List[Optional[str]] = [None] * len(inputs)
            if executor is None:
                for i, (inp, out) in enumerate(zip(inputs, outputs)):
                    row_id = row_offset + i
                    if inp is None or out is None:
                        continue
                    try:
                        traj = build_traj(
                            inp,
                            out,
                            seed=args.seed + row_id,
                            noise_step_prob=args.noise_step_prob,
                            max_steps=args.max_steps,
                            fmt=args.format,
                            max_lines=args.max_lines,
                            show_mode=args.show_mode,
                            one_indexed=args.one_indexed,
                        )
                        traj_values[i] = traj
                    except Exception as exc:  # noqa: BLE001
                        error_count += 1
                        if len(error_samples) < 5:
                            error_samples.append(f"row {row_id}: {exc}")
                        if args.on_error == "raise":
                            raise
                        if args.on_error == "empty":
                            traj_values[i] = ""
                        else:
                            traj_values[i] = None
            else:
                futures = {}
                for i, (inp, out) in enumerate(zip(inputs, outputs)):
                    row_id = row_offset + i
                    if inp is None or out is None:
                        continue
                    worker_args = (
                        inp,
                        out,
                        args.seed + row_id,
                        args.noise_step_prob,
                        args.max_steps,
                        args.format,
                        args.max_lines,
                        args.show_mode,
                        args.one_indexed,
                    )
                    futures[executor.submit(_build_traj_worker, worker_args)] = (i, row_id)

                for fut in as_completed(futures):
                    i, row_id = futures[fut]
                    try:
                        traj_values[i] = fut.result()
                    except Exception as exc:  # noqa: BLE001
                        error_count += 1
                        if len(error_samples) < 5:
                            error_samples.append(f"row {row_id}: {exc}")
                        if args.on_error == "raise":
                            raise
                        if args.on_error == "empty":
                            traj_values[i] = ""
                        else:
                            traj_values[i] = None

            table = pa.Table.from_batches([batch])
            table = table.append_column(args.traj_col, pa.array(traj_values, type=pa.string()))

            if writer is None:
                writer = pq.ParquetWriter(output_path, table.schema, compression=args.compression)
            writer.write_table(table)
            row_offset += len(traj_values)
    finally:
        if executor is not None:
            executor.shutdown()

    if writer is not None:
        writer.close()

    print(f"Wrote {row_offset} rows to {output_path}")
    if error_count:
        print(f"Encountered {error_count} trajectory errors", file=sys.stderr)
        for msg in error_samples:
            print(msg, file=sys.stderr)


if __name__ == "__main__":
    main()
