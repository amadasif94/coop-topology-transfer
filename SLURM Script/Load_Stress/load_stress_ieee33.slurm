#!/bin/bash
#SBATCH --job-name=LS33
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=/network/rit/lab/dahlin_lab/amad/graph_marl_dess/results/load_stress/ieee33/slurm_logs/LS33_%A_%a.out
#SBATCH --error=/network/rit/lab/dahlin_lab/amad/graph_marl_dess/results/load_stress/ieee33/slurm_logs/LS33_%A_%a.err

# IEEE33 load-stress evaluation of the frozen A0/A1/A2 checkpoints and the zero-action baseline.
# Create the log directory before the first submission:
#   mkdir -p results/load_stress/ieee33/slurm_logs
# Usage (array index = training run; zero and summary use --array=0 only):
#   sbatch --array=0-2 "SLURM Script/load_stress_ieee33.slurm" identity a0 <JOB_MAP>
#   sbatch --array=0-2 "SLURM Script/load_stress_ieee33.slurm" smoke    a0 <JOB_MAP>
#   sbatch --array=0-2 "SLURM Script/load_stress_ieee33.slurm" full     a0 <JOB_MAP> [SCALES]
#   sbatch --array=0   "SLURM Script/load_stress_ieee33.slurm" summary  -  <JOB_MAP>
# Stages: launch_check | preflight | identity (1.0x + identity validation) | smoke (1.3x, TP1, 2 days)
#         | full (default scales "1.1 1.2 1.3", run in sequence) | summary. Gates are enforced in Python.

set -euo pipefail

BASE_DIR="/network/rit/lab/dahlin_lab/amad/graph_marl_dess"
CONDA_ENV="/network/rit/lab/dahlin_lab/amad/amad_thesis/env/thesis_conda"
OUTPUT_ROOT="results/load_stress/ieee33"

STAGE="${1:?Specify launch_check, preflight, identity, smoke, full, or summary}"
MODEL="${2:?Specify a0, a1, a2, zero (or - for summary)}"
JOB_MAP_PATH="${3:?Specify the absolute JOB_MAP path}"
SCALES="${4:-1.1 1.2 1.3}"
RUN_ID="${SLURM_ARRAY_TASK_ID:-0}"

cd "${BASE_DIR}"

echo "============================================================"
echo "SLURM_JOB_ID        = ${SLURM_JOB_ID:-none}"
echo "ARRAY_TASK_ID       = ${RUN_ID}"
echo "STAGE               = ${STAGE}"
echo "MODEL               = ${MODEL}"
echo "SCALES              = ${SCALES}"
echo "HOSTNAME            = $(hostname)"
echo "JOB_MAP             = ${JOB_MAP_PATH}"
echo "START_TIME          = $(date)"
echo "============================================================"

if [[ ! -f "${JOB_MAP_PATH}" ]]; then
    echo "ERROR: JOB_MAP not found: ${JOB_MAP_PATH}"
    exit 1
fi

case "${STAGE}" in
    launch_check|preflight|identity|smoke|full|summary) ;;
    *) echo "ERROR: Unknown stage: ${STAGE}"; exit 1 ;;
esac
if [[ "${STAGE}" != "summary" ]]; then
    case "${MODEL}" in
        a0|a1|a2|zero) ;;
        *) echo "ERROR: Unknown model: ${MODEL}"; exit 1 ;;
    esac
fi
if [[ ( "${MODEL}" == "zero" || "${STAGE}" == "summary" ) && "${RUN_ID}" != "0" ]]; then
    echo "ERROR: ${STAGE}/${MODEL} runs once; submit with --array=0"
    exit 1
fi

if [[ "${STAGE}" == "launch_check" ]]; then
    echo "LAUNCH_CHECK_PASS"
    exit 0
fi

TMP_BASE="${SLURM_TMPDIR:-/tmp}"
PIP_CACHE_BASE="${SLURM_TMPDIR:-/tmp}"
JOB_TMP="${TMP_BASE}/ls33_${SLURM_JOB_ID:-local}_${RUN_ID}"
JOB_PIP_CACHE="${PIP_CACHE_BASE}/ls33_${SLURM_JOB_ID:-local}_${RUN_ID}"
mkdir -p "${JOB_TMP}" "${JOB_PIP_CACHE}"
export TMPDIR="${JOB_TMP}" TEMP="${JOB_TMP}" TMP="${JOB_TMP}" PIP_CACHE_DIR="${JOB_PIP_CACHE}" RAY_TMPDIR="${JOB_TMP}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 BLIS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1

source /network/rit/lab/dahlin_lab/anaconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV}"
export PYTHONPATH="${BASE_DIR}:${PYTHONPATH:-}"
echo "PYTHON = $(which python)"
python --version

EVAL=(python -u -m experiments.load_stress.evaluate_load_stress_ieee33
      --model "${MODEL}" --run "${RUN_ID}" --job-map "${JOB_MAP_PATH}" --output-root "${OUTPUT_ROOT}")
LABEL_DIR() {
    case "${MODEL}" in
        a0) echo "a0_original" ;; a1) echo "a1_no_shared_kpi" ;;
        a2) echo "a2_no_centralized_critic" ;; zero) echo "zero_action" ;;
    esac
}

case "${STAGE}" in
    preflight)
        python - <<'PY'
import torch
from experiments.load_stress import evaluate_load_stress_ieee33, validate_load_stress_identity, summarize_load_stress
print("PyTorch:", torch.__version__, "CUDA available:", torch.cuda.is_available())
print("PREFLIGHT_PASS")
PY
        ;;
    identity)
        "${EVAL[@]}" --scale 1.0
        RUN_DIR="${OUTPUT_ROOT}/$(LABEL_DIR)/run_${RUN_ID}/load_x100"
        if [[ "${MODEL}" == "zero" ]]; then
            # Optional for zero: only observable when an existing 1.0x zero-action final run is recorded.
            python -u -m experiments.load_stress.validate_load_stress_identity --run-dir "${RUN_DIR}" \
                || echo "ZERO_IDENTITY_NOT_PASSED (not a gate for the zero-action baseline)"
        else
            python -u -m experiments.load_stress.validate_load_stress_identity --run-dir "${RUN_DIR}"
        fi
        ;;
    smoke)
        "${EVAL[@]}" --scale 1.3 --smoke
        ;;
    full)
        for SCALE in ${SCALES}; do
            echo "---- scale ${SCALE} ----"
            "${EVAL[@]}" --scale "${SCALE}"
        done
        ;;
    summary)
        STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
        python -u -m experiments.load_stress.summarize_load_stress --root "${OUTPUT_ROOT}" \
            --output "${OUTPUT_ROOT}/summary_${STAMP}"
        ;;
esac

echo "============================================================"
echo "LS33_STAGE_COMPLETED ${STAGE} ${MODEL}"
echo "END TIME = $(date)"
echo "============================================================"
