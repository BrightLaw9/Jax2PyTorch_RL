#!/bin/bash
#SBATCH --job-name=miniport_train
#SBATCH --time=10:00:00
#SBATCH --mem=16GB
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu-gen
#SBATCH --gres=gpu:rtx6000:1
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j-err.out

set -euo pipefail
# Before sbatch: mkdir -p logs
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
# Cluster-specific modules, if required:
# module load python/3.11 cuda/12.6
MINIPORT_VENV="${MINIPORT_VENV:-.venv-gpu}"
TRAIN_CONFIG="${TRAIN_CONFIG:-configs/train.json}"
if [ ! -x "${MINIPORT_VENV}/bin/python" ]; then
    python3 -m venv "${MINIPORT_VENV}"
fi
source "${MINIPORT_VENV}/bin/activate"
python -m pip install --upgrade pip
python -m pip install torch==2.7.1 --index-url "${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu126}"
python -m pip install -r requirements-gpu.txt
python -m pip install --no-deps -e .
python -u -c 'import torch; print("torch:", torch.__version__, "CUDA:", torch.version.cuda); assert torch.cuda.is_available(), "CUDA unavailable: check driver and wheel compatibility"; print("GPU:", torch.cuda.get_device_name(0))'
TRAIN_ARGS=(--config "${TRAIN_CONFIG}")
if [ "${RESUME:-0}" = "1" ]; then
    TRAIN_ARGS+=(--resume)
fi
if [ "${COLLECT_ONLY:-0}" = "1" ]; then
    TRAIN_ARGS+=(--collect-only)
fi
python -u -m miniport.train "${TRAIN_ARGS[@]}"
# Export is a separate job, so CPU merge does not compete with live CUDA state.
