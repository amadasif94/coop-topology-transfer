#!/bin/bash
#SBATCH --job-name=LS69
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=2-00:00:00
#SBATCH --output=/network/rit/lab/dahlin_lab/amad/graph_marl_dess/results/load_stress/ieee69/slurm_logs/LS69_%A_%a.out
#SBATCH --error=/network/rit/lab/dahlin_lab/amad/graph_marl_dess/results/load_stress/ieee69/slurm_logs/LS69_%A_%a.err

# IEEE69 load-stress evaluation of the frozen A0/A1/A2 checkpoints, TP1-TP10, 31 December days (CPU, batch).
# Create the log directory before the first submission:
#   mkdir -p results/load_stress/ieee69/slurm_logs
# Usage (array index = training run 0-2; summary uses --array=0):
#   sbatch --array=0-2 "SLURM Script/load_stress_ieee69.slurm" full     a0 <JOB_MAP> [SCALES]
#   sbatch --array=0-2 "SLURM Script/load_stress_ieee69.slurm" identity a0 <JOB_MAP>   # optional, non-blocking
#   sbatch --array=0-2 "SLURM Script/load_stress_ieee69.slurm" pilot    a0 <JOB_MAP>   # optional, non-blocking
#   sbatch --array=0   "SLURM Script/load_stress_ieee69.slurm" summary  -  <JOB_MAP>
# Stages: launch_check | preflight | full (default scales "1.1 1.2 1.3", in sequence) | identity (1.0x +
#         optional identity validation) | pilot (1.3x, TP1, 2 days) | summary. Checks are enforced in Python.

set -euo pipefail

BASE_DIR="/network/rit/lab/dahlin_lab/amad/graph_marl_dess"
CONDA_ENV="/network/rit/lab/dahlin_lab/amad/amad_thesis/env/thesis_conda"
OUTPUT_ROOT="results/load_stress/ieee69"

STAGE="${1:?Specify launch_check, preflight, full, identity, pilot, or summary}"
MODEL="${2:?Specify a0, a1, a2 (or - for summary)}"
JOB_MAP_PATH="${3:?Specify the absolute IEEE69 JOB_MAP path}"
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
    launch_check|preflight|full|identity|pilot|summary) ;;
    *) echo "ERROR: Unknown stage: ${STAGE}"; exit 1 ;;
esac
if [[ "${STAGE}" != "summary" ]]; then
    case "${MODEL}" in
        a0|a1|a2) ;;
        *) echo "ERROR: Unknown model: ${MODEL} (zero-action is not part of the IEEE69 stress study)"; exit 1 ;;
    esac
elif [[ "${RUN_ID}" != "0" ]]; then
    echo "ERROR: summary runs once; submit with --array=0"
    exit 1
fi
for SCALE in ${SCALES}; do
    case "${SCALE}" in 1.1|1.2|1.3) ;; *) echo "ERROR: stress scale must be 1.1, 1.2 or 1.3: ${SCALE}"; exit 1 ;; esac
done

if [[ "${STAGE}" == "launch_check" ]]; then
    echo "LAUNCH_CHECK_PASS"
    exit 0
fi

# CPU node: node-local scratch only, no DGX paths; one thread per numerical library.
JOB_TMP="${TMPDIR:-/tmp}/ls69_${SLURM_JOB_ID:-local}_${RUN_ID}"
mkdir -p "${JOB_TMP}"
export TMPDIR="${JOB_TMP}" TEMP="${JOB_TMP}" TMP="${JOB_TMP}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 BLIS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 TORCH_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
trap 'rm -rf "${JOB_TMP}"' EXIT

source /network/rit/lab/dahlin_lab/anaconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV}"
export PYTHONPATH="${BASE_DIR}:${PYTHONPATH:-}"
echo "PYTHON = $(which python)"
python --version

EVAL=(python -u -m experiments.load_stress.evaluate_load_stress_ieee69
      --model "${MODEL}" --run "${RUN_ID}" --job-map "${JOB_MAP_PATH}" --output-root "${OUTPUT_ROOT}" --cpu)
LABEL_DIR() {
    case "${MODEL}" in
        a0) echo "a0_original" ;; a1) echo "a1_no_shared_kpi" ;; a2) echo "a2_no_centralized_critic" ;;
    esac
}

case "${STAGE}" in
    preflight)
        python - <<'PY'
import torch
torch.set_num_threads(1)
from experiments.load_stress import evaluate_load_stress_ieee69, validate_load_stress_identity_ieee69, summarize_load_stress_ieee69
print("PyTorch:", torch.__version__, "threads:", torch.get_num_threads())
print("PREFLIGHT_PASS")
PY
        ;;
    full)
        for SCALE in ${SCALES}; do
            echo "---- scale ${SCALE} ----"
            "${EVAL[@]}" --scale "${SCALE}"
        done
        ;;
    identity)
        "${EVAL[@]}" --scale 1.0
        python -u -m experiments.load_stress.validate_load_stress_identity_ieee69 \
            --run-dir "${OUTPUT_ROOT}/$(LABEL_DIR)/run_${RUN_ID}/load_x100" \
            || echo "IDENTITY_NOT_PASSED: inspect identity_report.json; a FAIL report blocks later stress runs"
        ;;
    pilot)
        "${EVAL[@]}" --scale 1.3 --pilot || echo "PILOT_NOT_PASSED (informational; full runs are not blocked)"
        ;;
    summary)
        STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
        python -u -m experiments.load_stress.summarize_load_stress_ieee69 --root "${OUTPUT_ROOT}" \
            --output "${OUTPUT_ROOT}/summary_${STAMP}"
        ;;
esac

echo "============================================================"
echo "LS69_STAGE_COMPLETED ${STAGE} ${MODEL}"
echo "END TIME = $(date)"
echo "============================================================"
