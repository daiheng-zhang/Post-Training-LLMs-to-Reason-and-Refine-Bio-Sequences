#!/bin/bash
#SBATCH --job-name=stride-rl
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.log
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

# Example:
# bash scripts/train/run_rl_molecule.sh
bash scripts/train/run_rl_molecule.sh
