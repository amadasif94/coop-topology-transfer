#!/bin/bash
#SBATCH --job-name=A2_eval
#SBATCH --partition=dgx
#SBATCH --qos=freetier
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=/network/rit/lab/dahlin_lab/amad/graph_marl_dess/results/ablations/ieee33/slurm_logs/A2_eval_%A_%a.out
#SBATCH --error=/network/rit/lab/dahlin_lab/amad/graph_marl_dess/results/ablations/ieee33/slurm_logs/A2_eval_%A_%a.err

set -euo pipefail

BASE_DIR="/network/rit/lab/dahlin_lab/amad/graph_marl_dess"
CONDA_ENV="/network/rit/lab/dahlin_lab/amad/amad_thesis/env/thesis_conda"

# Receive configuration as positional arguments.
# Do not require STAGE or JOB_MAP through sbatch --export.
STAGE="${1:?Specify launch_check, preflight, sweep, select, or evaluate}"
JOB_MAP_PATH="${2:?Specify the absolute JOB_MAP path}"

RUN_ID="${SLURM_ARRAY_TASK_ID:-0}"
SEED=$((42 + RUN_ID))

cd "${BASE_DIR}"

echo "============================================================"
echo "SLURM_JOB_ID        = ${SLURM_JOB_ID}"
echo "ARRAY_TASK_ID       = ${RUN_ID}"
echo "VARIANT             = no_centralized_critic"
echo "STAGE               = ${STAGE}"
echo "SEED                = ${SEED}"
echo "HOSTNAME            = $(hostname)"
echo "JOB_MAP             = ${JOB_MAP_PATH}"
echo "START_TIME          = $(date)"
echo "============================================================"

if [[ ! -f "${JOB_MAP_PATH}" ]]; then
    echo "ERROR: JOB_MAP not found: ${JOB_MAP_PATH}"
    exit 1
fi

case "${STAGE}" in
    launch_check|preflight|sweep|select|evaluate)
        ;;
    *)
        echo "ERROR: Unknown evaluation stage: ${STAGE}"
        exit 1
        ;;
esac

# Check batch execution without loading Python.
if [[ "${STAGE}" == "launch_check" ]]; then
    echo "LAUNCH_CHECK_PASS"
    exit 0
fi

# Match the successful A0 environment structure.
TMP_BASE="/network/rit/dgx/dgx_dahlin_lab/amad/graph_marl_dess/tmp"
PIP_CACHE_BASE="/network/rit/dgx/dgx_dahlin_lab/amad/graph_marl_dess/pip_cache"

JOB_TMP="${TMP_BASE}/eval_${SLURM_JOB_ID}_${RUN_ID}"
JOB_PIP_CACHE="${PIP_CACHE_BASE}/eval_${SLURM_JOB_ID}_${RUN_ID}"

mkdir -p "${JOB_TMP}" "${JOB_PIP_CACHE}"

export TMPDIR="${JOB_TMP}"
export TEMP="${JOB_TMP}"
export TMP="${JOB_TMP}"
export PIP_CACHE_DIR="${JOB_PIP_CACHE}"
export RAY_TMPDIR="${JOB_TMP}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1

source /network/rit/lab/dahlin_lab/anaconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV}"

export PYTHONPATH="${BASE_DIR}:${PYTHONPATH:-}"

echo "PYTHON = $(which python)"
python --version

# Check imports and environment without evaluation.
if [[ "${STAGE}" == "preflight" ]]; then
    python - <<'PY'
import torch

from configs.ieee33_config import IEEE33_CONFIG
from environments.dess_env import DESSEnv
from experiments.no_shared_kpi import slurm_dispatch

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("A2 evaluation imports successful")
print("PREFLIGHT_PASS")
PY
    exit 0
fi

echo "============================================================"
echo "Launching IEEE33 A2 evaluation stage"
echo "Variant: no_centralized_critic"
echo "Stage: ${STAGE}"
echo "Run: ${RUN_ID}"
echo "============================================================"

python -u -m experiments.no_shared_kpi.slurm_dispatch \
    --variant no_centralized_critic \
    --stage "${STAGE}" \
    --run "${RUN_ID}" \
    --job-map "${JOB_MAP_PATH}"

echo "============================================================"
echo "A2_EVAL_STAGE_COMPLETED"
echo "END TIME = $(date)"
echo "============================================================"
