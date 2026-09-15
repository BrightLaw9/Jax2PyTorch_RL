#!/bin/bash
#SBATCH --job-name=miniport_teacher_value
#SBATCH --time=10:00:00
#SBATCH --mem=16GB
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu-gen
#SBATCH --gres=gpu:rtx6000:1
#SBATCH --output=logs/teacher-%j.out
#SBATCH --error=logs/teacher-%j-err.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
MINIPORT_VENV="${MINIPORT_VENV:-.venv-gpu}"
TEACHER_CONFIG="${TEACHER_CONFIG:-configs/teacher-value.json}"
source "${MINIPORT_VENV}/bin/activate"
python -m pip install --no-deps -e .
ARGS=(--config "${TEACHER_CONFIG}")
if [ "${RESUME:-0}" = "1" ]; then
    ARGS+=(--resume)
fi
python -u -m miniport.teacher_collect "${ARGS[@]}"
