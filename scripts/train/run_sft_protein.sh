#!/usr/bin/env bash
set -euo pipefail

DRY_RUN=0
EXTRA_ARGS=()
for arg in "$@"; do
  if [[ "$arg" == "--dry-run" ]]; then
    DRY_RUN=1
  else
    EXTRA_ARGS+=("$arg")
  fi
done

CONFIG_PATH="${CONFIG_PATH:-configs/sft/protein_sft_qwen3_14b_qlora.yml}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-Qwen/Qwen3-14B}"
DATASET_DIR="${DATASET_DIR:-data/gfp/sharegpt}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/gfp/sft}"

CMD=(
  llamafactory-cli train "$CONFIG_PATH"
  "model_name_or_path=${MODEL_NAME_OR_PATH}"
  "dataset_dir=${DATASET_DIR}"
  "output_dir=${OUTPUT_DIR}"
)
CMD+=("${EXTRA_ARGS[@]}")

if [[ "$DRY_RUN" -eq 1 ]]; then
  printf 'Dry-run command:\n%s\n' "${CMD[*]}"
  exit 0
fi

exec "${CMD[@]}"
