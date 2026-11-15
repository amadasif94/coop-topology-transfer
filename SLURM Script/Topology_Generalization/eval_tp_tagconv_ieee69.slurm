#!/bin/bash
#SBATCH --job-name=tp_tag69_best
#SBATCH --partition=batch
#SBATCH --array=0-2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=/network/rit/lab/dahlin_lab/amad/graph_marl_dess/results/logs/slurm/tp_tag69_best_%A_%a.out
#SBATCH --error=/network/rit/lab/dahlin_lab/amad/graph_marl_dess/results/logs/slurm/tp_tag69_best_%A_%a.err

set -euo pipefail

BASE_DIR="/network/rit/lab/dahlin_lab/amad/graph_marl_dess"
SLURM_LOG_DIR="${BASE_DIR}/results/logs/slurm"
TMP_BASE="${BASE_DIR}/tmp"
PIP_CACHE_DIR_BASE="${BASE_DIR}/pip_cache"

CONDA_SH="/network/rit/lab/dahlin_lab/anaconda3/etc/profile.d/conda.sh"
CONDA_ENV="/network/rit/lab/dahlin_lab/amad/amad_thesis/env/thesis_conda"

mkdir -p "${SLURM_LOG_DIR}" "${TMP_BASE}" "${PIP_CACHE_DIR_BASE}"

JOB_TMP="${TMP_BASE}/job_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
JOB_PIP_CACHE="${PIP_CACHE_DIR_BASE}/job_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

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
export VECLIB_MAXIMUM_THREADS=1

MODEL="tagconv"
GNN_TYPE="tagconv"
RUN_ID="${SLURM_ARRAY_TASK_ID}"

CHECKPOINTS=(
  "${BASE_DIR}/results/models/IEEE69/tagconv/run_0_best.pt"
  "${BASE_DIR}/results/models/IEEE69/tagconv/run_1_best.pt"
  "${BASE_DIR}/results/models/IEEE69/tagconv/run_2_best.pt"
)

RUN_NAMES=(
  "tagconv_ieee69_tp_full_eval_run0_best"
  "tagconv_ieee69_tp_full_eval_run1_best"
  "tagconv_ieee69_tp_full_eval_run2_best"
)

CHECKPOINT_PATH="${CHECKPOINTS[${RUN_ID}]}"
RUN_NAME="${RUN_NAMES[${RUN_ID}]}"

echo "============================================================"
echo "SLURM_JOB_ID         = ${SLURM_JOB_ID}"
echo "SLURM_ARRAY_JOB_ID   = ${SLURM_ARRAY_JOB_ID}"
echo "SLURM_ARRAY_TASK_ID  = ${SLURM_ARRAY_TASK_ID}"
echo "MODEL                = ${MODEL}"
echo "GNN_TYPE             = ${GNN_TYPE}"
echo "CASE                 = ieee69"
echo "RUN_ID               = ${RUN_ID}"
echo "RUN_NAME             = ${RUN_NAME}"
echo "CHECKPOINT_PATH      = ${CHECKPOINT_PATH}"
echo "HOSTNAME             = $(hostname)"
echo "START TIME           = $(date)"
echo "CPUS REQUESTED       = ${SLURM_CPUS_PER_TASK}"
echo "TMPDIR               = ${TMPDIR}"
echo "============================================================"

source "${CONDA_SH}"

conda activate "${CONDA_ENV}" || {
    echo "ERROR: conda activate failed"
    exit 127
}

export PYTHONPATH="${BASE_DIR}:${PYTHONPATH:-}"

echo "PYTHON = $(which python)"
python --version

cd "${BASE_DIR}"

echo "============================================================"
echo "Preflight import/checkpoint test"
echo "============================================================"

python - <<PY
from pathlib import Path
import torch

ckpt = Path("${CHECKPOINT_PATH}")

print("CUDA available:", torch.cuda.is_available())
print("Checkpoint exists:", ckpt.exists())
print("Checkpoint path:", ckpt)

if not ckpt.exists():
    raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

from configs.ieee69_config import IEEE69_CONFIG
from environments.dess_env import DESSEnv
from models.maddpg import GraphAwareMADDPG
from experiments.topology_generalization.topology_utils_ieee69 import build_topology_config

print("IEEE69 TAGConv topology-eval import/checkpoint test passed")
PY

echo "============================================================"
echo "Launching IEEE69 TAGConv topology generalization evaluation"
echo "============================================================"

PYTHONPATH=. python -u experiments/topology_generalization/run_tp_eval_ieee69.py \
  --checkpoint "${CHECKPOINT_PATH}" \
  --gnn_type "${GNN_TYPE}" \
  --topologies TP1 TP2 TP3 TP4 TP5 TP6 TP7 TP8 TP9 \
  --episodes 31 \
  --include_zero \
  --include_random \
  --run_name "${RUN_NAME}"

rc=$?

echo "============================================================"
echo "END TIME      = $(date)"
echo "EXIT CODE     = ${rc}"
echo "============================================================"

exit ${rc}