#!/bin/bash
#SBATCH --job-name=miniport_gpu_eval
#SBATCH --time=06:00:00
#SBATCH --mem=16GB
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu-gen
#SBATCH --gres=gpu:rtx6000:1
#SBATCH --output=logs/gpu-eval-%j.out
#SBATCH --error=logs/gpu-eval-%j-err.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
MINIPORT_VENV="${MINIPORT_VENV:-.venv-gpu}"
CHECKPOINT="${CHECKPOINT:-artifacts/gpu-run/checkpoint}"
EVAL_MANIFEST="${EVAL_MANIFEST:-artifacts/private-gpu-eval/manifest.json}"
EVAL_OUTPUT="${EVAL_OUTPUT:-artifacts/gpu-eval-update8}"
CHALLENGES="${CHALLENGES:-artifacts/private-gpu-eval/challenges}"
source "${MINIPORT_VENV}/bin/activate"
python -m pip install --no-deps -e .
if [ ! -f "${EVAL_MANIFEST}" ]; then
    mkdir -p "$(dirname "${EVAL_MANIFEST}")"
    python -m miniport.cli provision-eval "${EVAL_MANIFEST}"
fi
BASELINE_ARGS=()
if [ -n "${BASELINE_RESULTS:-}" ]; then
    BASELINE_ARGS+=(--baseline-results "${BASELINE_RESULTS}")
fi
python -u -m miniport.gpu_eval --checkpoint "${CHECKPOINT}" \
    --manifest "${EVAL_MANIFEST}" --output "${EVAL_OUTPUT}" --challenges "${CHALLENGES}" \
    --task mlp-heldout --task layernorm-heldout --task attention-heldout --seed 42 "${BASELINE_ARGS[@]}"
