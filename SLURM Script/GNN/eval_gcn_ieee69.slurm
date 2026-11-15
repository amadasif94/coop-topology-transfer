#!/bin/bash
#SBATCH --job-name=eval_gcn69
#SBATCH --partition=batch
#SBATCH --array=0-2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=/network/rit/lab/dahlin_lab/amad/graph_marl_dess/results/logs/slurm/eval_gcn_ieee69_%A_%a.out
#SBATCH --error=/network/rit/lab/dahlin_lab/amad/graph_marl_dess/results/logs/slurm/eval_gcn_ieee69_%A_%a.err

set -euo pipefail

# F2/F16: checkpoint-selection sweep on the TP1 VALIDATION schedule S-SEL-v1 only (never the
# December test split). zero/random are evaluated once per schedule (validation_baselines.slurm).
# Select with scripts/select_checkpoint.py; final tests: final_test_learned.slurm.

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

MODEL="gcn"
RUN_ID="${SLURM_ARRAY_TASK_ID}"

CHECKPOINT_DIR="${BASE_DIR}/results/checkpoints/maddpg_ieee69/${MODEL}/${MODEL}_run_${RUN_ID}"

MIN_STEP=10000
MAX_STEP=200000

echo "============================================================"
echo "SLURM_JOB_ID         = ${SLURM_JOB_ID}"
echo "SLURM_ARRAY_JOB_ID   = ${SLURM_ARRAY_JOB_ID}"
echo "SLURM_ARRAY_TASK_ID  = ${SLURM_ARRAY_TASK_ID}"
echo "MODEL                = ${MODEL}"
echo "CASE                 = ieee69"
echo "RUN_ID               = ${RUN_ID}"
echo "CHECKPOINT_DIR       = ${CHECKPOINT_DIR}"
echo "MIN_STEP             = ${MIN_STEP}"
echo "MAX_STEP             = ${MAX_STEP}"
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

cd "${BASE_DIR}"

echo "PYTHON = $(which python)"
python --version

echo "============================================================"
echo "Preflight import/env/checkpoint-dir test"
echo "============================================================"

python - <<PY
from pathlib import Path
import torch

ckpt_dir = Path("${CHECKPOINT_DIR}")

print("CUDA available:", torch.cuda.is_available())
print("Checkpoint dir exists:", ckpt_dir.exists())
print("Checkpoint dir:", ckpt_dir)

if not ckpt_dir.exists():
    raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_dir}")

from configs.ieee69_config import IEEE69_CONFIG
from environments.dess_env import DESSEnv
from models.maddpg import GraphAwareMADDPG

env = DESSEnv(config=IEEE69_CONFIG, mode="val", seed=1)
obs, info = env.reset(seed=1)

print("IEEE69 eval import/env test passed")
print("num_buses:", env.num_buses)
print("num_agents:", env.num_agents)
print("obs x shape:", obs["x"].shape)
print("agent_obs shape:", obs["agent_obs"].shape)
print("initial feasible:", info["feasible"])
print("initial converged:", info["converged"])
PY

echo "============================================================"
echo "Launching IEEE69 GCN evaluation"
echo "Policies: maddpg (zero/random: validation_baselines.slurm)"
echo "Checkpoint range: ${MIN_STEP} to ${MAX_STEP}"
echo "============================================================"

shopt -s nullglob

FOUND_ANY=0
EVALUATED_ANY=0

for CKPT in "${CHECKPOINT_DIR}"/maddpg_step_*.pt
do
    FOUND_ANY=1

    STEP=$(basename "${CKPT}" | sed 's/maddpg_step_//; s/.pt//')

    if [ "${STEP}" -ge "${MIN_STEP}" ] && [ "${STEP}" -le "${MAX_STEP}" ]; then
        EVALUATED_ANY=1

        RUN_NAME="selection_ieee69/${MODEL}/run_${RUN_ID}/step_${STEP}"

        echo "------------------------------------------------------------"
        echo "Evaluating checkpoint:"
        echo "${CKPT}"
        echo "STEP     = ${STEP}"
        echo "RUN_NAME = ${RUN_NAME}"
        echo "Policies = maddpg (zero/random: validation_baselines.slurm)"
        echo "------------------------------------------------------------"

        PYTHONPATH=. python -u scripts/run_eval_ieee69.py \
            --checkpoint "${CKPT}" \
            --gnn_type "${MODEL}" \
            --mode val \
            --episodes schedule \
            --run_purpose checkpoint_selection \
            --run_name "${RUN_NAME}" \
            --cpu
    else
        echo "Skipping checkpoint outside range: ${CKPT}"
    fi
done

if [ "${FOUND_ANY}" -eq 0 ]; then
    echo "WARNING: No checkpoint files found in:"
    echo "${CHECKPOINT_DIR}"
fi

if [ "${EVALUATED_ANY}" -eq 0 ]; then
    echo "WARNING: No checkpoints evaluated in range ${MIN_STEP}-${MAX_STEP}."
fi

rc=$?

echo "============================================================"
echo "END TIME      = $(date)"
echo "PYTHON EXIT   = ${rc}"
echo "============================================================"

exit "${rc}"