#!/bin/bash
#SBATCH --job-name=tp_gat69_best
#SBATCH --partition=dgx
#SBATCH --qos=freetier
#SBATCH --array=0-2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --gres=gpu:1
#SBATCH --time=1-00:00:00
#SBATCH --output=/network/rit/lab/dahlin_lab/amad/graph_marl_dess/results/logs/slurm/tp_gat69_best_%A_%a.out
#SBATCH --error=/network/rit/lab/dahlin_lab/amad/graph_marl_dess/results/logs/slurm/tp_gat69_best_%A_%a.err

set -euo pipefail

BASE_DIR="/network/rit/lab/dahlin_lab/amad/graph_marl_dess"
SLURM_LOG_DIR="${BASE_DIR}/results/logs/slurm"
TMP_BASE="/network/rit/dgx/dgx_dahlin_lab/amad/graph_marl_dess/tmp"
PIP_CACHE_DIR_BASE="/network/rit/dgx/dgx_dahlin_lab/amad/graph_marl_dess/pip_cache"

mkdir -p "${SLURM_LOG_DIR}" "${TMP_BASE}" "${PIP_CACHE_DIR_BASE}"

export TMPDIR="${TMP_BASE}"
export TEMP="${TMP_BASE}"
export TMP="${TMP_BASE}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR_BASE}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

MODEL="gat"
GNN_TYPE="gat"

CHECKPOINTS=(
  "${BASE_DIR}/results/models/IEEE69/gat/run_0_best.pt"
  "${BASE_DIR}/results/models/IEEE69/gat/run_1_best.pt"
  "${BASE_DIR}/results/models/IEEE69/gat/run_2_best.pt"
)

RUN_NAMES=(
  "gat_ieee69_tp_full_eval_run0_best"
  "gat_ieee69_tp_full_eval_run1_best"
  "gat_ieee69_tp_full_eval_run2_best"
)

CHECKPOINT_PATH="${CHECKPOINTS[${SLURM_ARRAY_TASK_ID}]}"
RUN_NAME="${RUN_NAMES[${SLURM_ARRAY_TASK_ID}]}"

echo "============================================================"
echo "SLURM_JOB_ID         = ${SLURM_JOB_ID}"
echo "SLURM_ARRAY_JOB_ID   = ${SLURM_ARRAY_JOB_ID}"
echo "SLURM_ARRAY_TASK_ID  = ${SLURM_ARRAY_TASK_ID}"
echo "MODEL                = ${MODEL}"
echo "GNN_TYPE             = ${GNN_TYPE}"
echo "RUN_NAME             = ${RUN_NAME}"
echo "CHECKPOINT_PATH      = ${CHECKPOINT_PATH}"
echo "HOSTNAME             = $(hostname)"
echo "CUDA_VISIBLE_DEVICES = ${CUDA_VISIBLE_DEVICES:-unset}"
echo "START TIME           = $(date)"
echo "============================================================"

source /network/rit/lab/dahlin_lab/anaconda3/etc/profile.d/conda.sh
conda activate /network/rit/lab/dahlin_lab/amad/amad_thesis/env/thesis_conda

export PYTHONPATH="${BASE_DIR}:${PYTHONPATH:-}"

echo "PYTHON = $(which python)"
python --version

cd "${BASE_DIR}"

echo "============================================================"
echo "Preflight CUDA/import/checkpoint test"
echo "============================================================"

python - <<PY
from pathlib import Path
import torch

ckpt = Path("${CHECKPOINT_PATH}")

print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")
print("Checkpoint exists:", ckpt.exists())
print("Checkpoint path:", ckpt)

if not ckpt.exists():
    raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

from environments.dess_env import DESSEnv
from models.maddpg import GraphAwareMADDPG
from configs.ieee69_config import IEEE69_CONFIG
from experiments.topology_generalization.topology_utils_ieee69 import build_topology_config

print("IEEE69 GAT topology-eval import/checkpoint test passed")
PY

echo "============================================================"
echo "Launching IEEE69 GAT topology evaluation"
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
echo "PYTHON EXIT   = ${rc}"
echo "============================================================"

exit ${rc}