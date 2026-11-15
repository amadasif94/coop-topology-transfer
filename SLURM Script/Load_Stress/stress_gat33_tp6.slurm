#!/bin/bash
#SBATCH --job-name=stress_gat33_tp6
#SBATCH --partition=batch
#SBATCH --array=0-2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=61G
#SBATCH --time=2-00:00:00
#SBATCH --output=/network/rit/lab/dahlin_lab/amad/graph_marl_dess/results/logs/slurm/stress_gat33_tp6_%A_%a.out
#SBATCH --error=/network/rit/lab/dahlin_lab/amad/graph_marl_dess/results/logs/slurm/stress_gat33_tp6_%A_%a.err

set -euo pipefail

# ============================================================
# Paths
# ============================================================

BASE_DIR="/network/rit/lab/dahlin_lab/amad/graph_marl_dess"
SLURM_LOG_DIR="${BASE_DIR}/results/logs/slurm"
TMP_BASE="${BASE_DIR}/tmp"
PIP_CACHE_DIR_BASE="${BASE_DIR}/pip_cache"

CONDA_SH="/network/rit/lab/dahlin_lab/anaconda3/etc/profile.d/conda.sh"
CONDA_ENV="/network/rit/lab/dahlin_lab/amad/amad_thesis/env/thesis_conda"

mkdir -p \
    "${SLURM_LOG_DIR}" \
    "${TMP_BASE}" \
    "${PIP_CACHE_DIR_BASE}"

# Use a separate temporary directory for each array task.
JOB_TMP="${TMP_BASE}/stress_gat33_tp6_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
JOB_PIP_CACHE="${PIP_CACHE_DIR_BASE}/stress_gat33_tp6_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

mkdir -p "${JOB_TMP}" "${JOB_PIP_CACHE}"

export TMPDIR="${JOB_TMP}"
export TEMP="${JOB_TMP}"
export TMP="${JOB_TMP}"
export PIP_CACHE_DIR="${JOB_PIP_CACHE}"
export RAY_TMPDIR="${JOB_TMP}"

# Prevent numerical libraries from spawning excessive threads.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

# ============================================================
# Experiment configuration
# ============================================================

MODEL="gat"
GNN_TYPE="gat"
TOPOLOGY="TP6"
LOAD_SCALE="1.3"
EPISODES="31"

CHECKPOINTS=(
  "${BASE_DIR}/results/models/IEEE33/gat/run_0.pt"
  "${BASE_DIR}/results/models/IEEE33/gat/run_1.pt"
  "${BASE_DIR}/results/models/IEEE33/gat/run_2.pt"
)

RUN_NAMES=(
  "gat_ieee33_tp6_load1p3_run0"
  "gat_ieee33_tp6_load1p3_run1"
  "gat_ieee33_tp6_load1p3_run2"
)

CHECKPOINT_PATH="${CHECKPOINTS[${SLURM_ARRAY_TASK_ID}]}"
RUN_NAME="${RUN_NAMES[${SLURM_ARRAY_TASK_ID}]}"

# ============================================================
# Job information
# ============================================================

echo "============================================================"
echo "IEEE33 GAT TP6 LOAD-STRESS EVALUATION"
echo "============================================================"
echo "SLURM_JOB_ID         = ${SLURM_JOB_ID}"
echo "SLURM_ARRAY_JOB_ID   = ${SLURM_ARRAY_JOB_ID}"
echo "SLURM_ARRAY_TASK_ID  = ${SLURM_ARRAY_TASK_ID}"
echo "MODEL                = ${MODEL}"
echo "GNN_TYPE             = ${GNN_TYPE}"
echo "CASE                 = ieee33"
echo "TOPOLOGY             = ${TOPOLOGY}"
echo "LOAD SCALE           = ${LOAD_SCALE}"
echo "EPISODES             = ${EPISODES}"
echo "RUN_NAME             = ${RUN_NAME}"
echo "CHECKPOINT_PATH      = ${CHECKPOINT_PATH}"
echo "HOSTNAME             = $(hostname)"
echo "START TIME           = $(date)"
echo "CPUS REQUESTED       = ${SLURM_CPUS_PER_TASK}"
echo "TMPDIR               = ${TMPDIR}"
echo "============================================================"

# ============================================================
# Activate environment
# ============================================================

source "${CONDA_SH}"

conda activate "${CONDA_ENV}" || {
    echo "ERROR: Conda environment activation failed."
    exit 127
}

export PYTHONPATH="${BASE_DIR}:${PYTHONPATH:-}"

cd "${BASE_DIR}"

echo "PYTHON = $(which python)"
python --version

# ============================================================
# Preflight test
# ============================================================

echo "============================================================"
echo "Preflight import, checkpoint, and configuration test"
echo "============================================================"

python - <<PY
from pathlib import Path

import torch

from configs.ieee33_config import IEEE33_CONFIG
from environments.dess_env import DESSEnv
from models.maddpg import GraphAwareMADDPG
from experiments.topology_generalization.topology_utils import (
    build_topology_config,
)

checkpoint = Path("${CHECKPOINT_PATH}")

print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("Evaluation mode: CPU-compatible HPC batch mode")
print("Checkpoint exists:", checkpoint.exists())
print("Checkpoint path:", checkpoint)

if not checkpoint.exists():
    raise FileNotFoundError(
        f"Checkpoint does not exist: {checkpoint}"
    )

data_scaling = IEEE33_CONFIG.get("data_scaling", {})

actual_load_scale = float(
    data_scaling.get("load_scaling_factor", 1.0)
)

expected_load_scale = float("${LOAD_SCALE}")

print("Expected load scaling factor:", expected_load_scale)
print("Configured load scaling factor:", actual_load_scale)

if abs(actual_load_scale - expected_load_scale) > 1e-12:
    raise RuntimeError(
        "Incorrect IEEE33 load scaling factor. "
        f"Expected {expected_load_scale}, "
        f"but configuration contains {actual_load_scale}."
    )

print("IEEE33 GAT TP6 stress-test preflight passed.")
PY

# ============================================================
# Run evaluation
# ============================================================

echo "============================================================"
echo "Launching IEEE33 GAT TP6 load-stress evaluation"
echo "Topology: ${TOPOLOGY}"
echo "Load scaling factor: ${LOAD_SCALE}"
echo "Episodes: ${EPISODES}"
echo "Checkpoint: ${CHECKPOINT_PATH}"
echo "Run name: ${RUN_NAME}"
echo "============================================================"

PYTHONPATH=. python -u \
  experiments/topology_generalization/run_tp_eval.py \
  --checkpoint "${CHECKPOINT_PATH}" \
  --gnn_type "${GNN_TYPE}" \
  --topologies "${TOPOLOGY}" \
  --episodes "${EPISODES}" \
  --run_name "${RUN_NAME}" \
  --cpu

rc=$?

echo "============================================================"
echo "END TIME      = $(date)"
echo "PYTHON EXIT   = ${rc}"
echo "============================================================"

exit ${rc}