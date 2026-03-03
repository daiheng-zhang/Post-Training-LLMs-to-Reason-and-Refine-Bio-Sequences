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

export WANDB_PROJECT="${WANDB_PROJECT:-stride-gfp}"
CMD=(python GRPO/train_grpo.py --config-name gfp_grpo)
CMD+=("${EXTRA_ARGS[@]}")

if [[ "$DRY_RUN" -eq 1 ]]; then
  printf 'Dry-run command:\nSTRIDE_DRY_RUN=1 %s\n' "${CMD[*]}"
  exit 0
fi

exec "${CMD[@]}"
