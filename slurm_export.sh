#!/bin/bash
#SBATCH --job-name=miniport_export
#SBATCH --time=02:00:00
#SBATCH --mem=16GB
#SBATCH --cpus-per-task=8
# This cluster exposes only gpu-gen; CUDA stays hidden from the CPU merge.
#SBATCH --partition=gpu-gen
#SBATCH --gres=gpu:rtx6000:1
#SBATCH --output=logs/export-%j.out
#SBATCH --error=logs/export-%j-err.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
export CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu TOKENIZERS_PARALLELISM=false
MINIPORT_VENV="${MINIPORT_VENV:-.venv-gpu}"
source "${MINIPORT_VENV}/bin/activate"
python -u -m miniport.export_model --checkpoint "${CHECKPOINT:-artifacts/gpu-run/checkpoint}" \
    --output "${EXPORT_DIR:-artifacts/export}"
