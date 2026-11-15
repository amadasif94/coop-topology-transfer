#!/bin/bash
#SBATCH --job-name=A1_no_shared_kpi
#SBATCH --partition=dgx
#SBATCH --qos=freetier
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=/network/rit/lab/dahlin_lab/amad/graph_marl_dess/results/ablations/ieee33/slurm_logs/A1_a0style_%A_%a.out
#SBATCH --error=/network/rit/lab/dahlin_lab/amad/graph_marl_dess/results/ablations/ieee33/slurm_logs/A1_a0style_%A_%a.err

set -euo pipefail

BASE_DIR="/network/rit/lab/dahlin_lab/amad/graph_marl_dess"
CONDA_ENV="/network/rit/lab/dahlin_lab/amad/amad_thesis/env/thesis_conda"

# Positional arguments avoid passing custom variables through sbatch --export.
STAGE="${1:-launch_check}"
JOB_MAP_PATH="${2:-${BASE_DIR}/experiments/no_shared_kpi/cluster_job_map.json}"

RUN_ID="${SLURM_ARRAY_TASK_ID:-0}"
SEED=$((42 + RUN_ID))

cd "${BASE_DIR}"

echo "============================================================"
echo "JOB ID       = ${SLURM_JOB_ID}"
echo "ARRAY TASK   = ${RUN_ID}"
echo "STAGE        = ${STAGE}"
echo "SEED         = ${SEED}"
echo "HOSTNAME     = $(hostname)"
echo "START TIME   = $(date)"
echo "JOB MAP      = ${JOB_MAP_PATH}"
echo "============================================================"

test -f "${JOB_MAP_PATH}"

# Very small test: no Python, no Conda, no training.
if [[ "${STAGE}" == "launch_check" ]]; then
    echo "LAUNCH_CHECK_PASS"
    exit 0
fi

# Match original A0 temporary directory configuration.
TMP_BASE="/network/rit/dgx/dgx_dahlin_lab/amad/graph_marl_dess/tmp"
PIP_CACHE_DIR_BASE="/network/rit/dgx/dgx_dahlin_lab/amad/graph_marl_dess/pip_cache"

mkdir -p "${TMP_BASE}" "${PIP_CACHE_DIR_BASE}"

export TMPDIR="${TMP_BASE}"
export TEMP="${TMP_BASE}"
export TMP="${TMP_BASE}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR_BASE}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1

source /network/rit/lab/dahlin_lab/anaconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV}"

export PYTHONPATH="${BASE_DIR}:${PYTHONPATH:-}"

echo "PYTHON = $(which python)"
python --version

# Check imports and device availability without training.
if [[ "${STAGE}" == "preflight" ]]; then
    python - <<'PY'
import torch
from configs.ieee33_config import IEEE33_CONFIG
from environments.dess_env import DESSEnv
from experiments.no_shared_kpi import slurm_dispatch

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")
print("A1 imports successful")
print("PREFLIGHT_PASS")
PY
    exit 0
fi

if [[ "${STAGE}" != "train" ]]; then
    echo "ERROR: Unknown stage ${STAGE}"
    exit 1
fi

# Full training must use the allocated GPU.
python - <<'PY'
import torch
if not torch.cuda.is_available():
    raise RuntimeError("Training requested a GPU but CUDA is unavailable")
print("CUDA_TRAINING_PREFLIGHT_PASS")
PY

echo "Starting A1 No Shared KPI training"
echo "Run: ${RUN_ID}"
echo "Seed: ${SEED}"
echo "Total steps: 200000"

python -u -m experiments.no_shared_kpi.slurm_dispatch \
    --variant no_shared_kpi \
    --stage train \
    --run "${RUN_ID}" \
    --job-map "${JOB_MAP_PATH}"

echo "A1_TRAINING_COMPLETED"
echo "END TIME = $(date)"
