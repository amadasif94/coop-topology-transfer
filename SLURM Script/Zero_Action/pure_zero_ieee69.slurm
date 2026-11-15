#!/bin/bash
#SBATCH --job-name=PZ69
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=1-00:00:00
#SBATCH --output=/network/rit/lab/dahlin_lab/amad/graph_marl_dess/results/load_stress/ieee69/pure_zero_no_backup/slurm_logs/PZ69_%A_%a.out
#SBATCH --error=/network/rit/lab/dahlin_lab/amad/graph_marl_dess/results/load_stress/ieee69/pure_zero_no_backup/slurm_logs/PZ69_%A_%a.err

# IEEE69 PURE zero-action reference (nine batteries 0 kW, no backup), TP1-TP10, 31 December days (CPU, batch).
# A0/A1/A2 load-stress results are not touched. Create the log directory before the first submission:
#   mkdir -p results/load_stress/ieee69/pure_zero_no_backup/slurm_logs
# Usage:
#   sbatch "SLURM Script/pure_zero_ieee69.slurm" smoke <JOB_MAP>                    # 1.3x, TP1+TP10, 2 days
#   sbatch --array=0-3 "SLURM Script/pure_zero_ieee69.slurm" full <JOB_MAP>         # index -> 1.0 1.1 1.2 1.3
#   sbatch "SLURM Script/pure_zero_ieee69.slurm" crosscheck <JOB_MAP>               # 1.0x only; optional
#   sbatch "SLURM Script/pure_zero_ieee69.slurm" summary <JOB_MAP>
# Stages: launch_check | preflight | smoke | full | crosscheck | summary. Checks are enforced in Python.

set -euo pipefail

BASE_DIR="/network/rit/lab/dahlin_lab/amad/graph_marl_dess"
CONDA_ENV="/network/rit/lab/dahlin_lab/amad/amad_thesis/env/thesis_conda"
OUTPUT_ROOT="results/load_stress/ieee69/pure_zero_no_backup"
SCALES=(1.0 1.1 1.2 1.3)

STAGE="${1:?Specify launch_check, preflight, smoke, full, crosscheck, or summary}"
JOB_MAP_PATH="${2:?Specify the absolute IEEE69 JOB_MAP path}"
INDEX="${SLURM_ARRAY_TASK_ID:-0}"

cd "${BASE_DIR}"

echo "============================================================"
echo "SLURM_JOB_ID        = ${SLURM_JOB_ID:-none}"
echo "ARRAY_TASK_ID       = ${INDEX}"
echo "STAGE               = ${STAGE}"
echo "HOSTNAME            = $(hostname)"
echo "JOB_MAP             = ${JOB_MAP_PATH}"
echo "START_TIME          = $(date)"
echo "============================================================"

if [[ ! -f "${JOB_MAP_PATH}" ]]; then
    echo "ERROR: JOB_MAP not found: ${JOB_MAP_PATH}"
    exit 1
fi
case "${STAGE}" in
    launch_check|preflight|smoke|full|crosscheck|summary) ;;
    *) echo "ERROR: Unknown stage: ${STAGE}"; exit 1 ;;
esac
if [[ "${STAGE}" == "full" ]]; then
    if [[ ! "${INDEX}" =~ ^[0-3]$ ]]; then
        echo "ERROR: full uses --array=0-3 (index -> scale 1.0 1.1 1.2 1.3): ${INDEX}"
        exit 1
    fi
elif [[ "${INDEX}" != "0" ]]; then
    echo "ERROR: ${STAGE} runs once; submit without --array or with --array=0"
    exit 1
fi
SCALE="${SCALES[${INDEX}]}"
echo "SCALE               = ${SCALE}"

if [[ "${STAGE}" == "launch_check" ]]; then
    echo "LAUNCH_CHECK_PASS"
    exit 0
fi

# CPU node: node-local scratch only, no DGX paths; one thread per numerical library.
JOB_TMP="${TMPDIR:-/tmp}/pz69_${SLURM_JOB_ID:-local}_${INDEX}"
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

EVAL=(python -u -m experiments.load_stress.evaluate_pure_zero_ieee69 --job-map "${JOB_MAP_PATH}" --output-root "${OUTPUT_ROOT}")

case "${STAGE}" in
    preflight)
        python - <<'PY'
from experiments.load_stress import evaluate_pure_zero_ieee69, validate_pure_zero_ieee69, summarize_pure_zero_ieee69
print("PREFLIGHT_PASS")
PY
        ;;
    smoke)
        "${EVAL[@]}" --scale 1.3 --smoke
        ;;
    full)
        "${EVAL[@]}" --scale "${SCALE}"
        ;;
    crosscheck)
        python -u -m experiments.load_stress.validate_pure_zero_ieee69 --pure-dir "${OUTPUT_ROOT}/load_x100" \
            || echo "CROSS_CHECK_NOT_PASSED: inspect cross_check_report.json"
        ;;
    summary)
        STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
        python -u -m experiments.load_stress.summarize_pure_zero_ieee69 --root "${OUTPUT_ROOT}" \
            --output "${OUTPUT_ROOT}/summary_${STAMP}"
        ;;
esac

echo "============================================================"
echo "PZ69_STAGE_COMPLETED ${STAGE} ${SCALE}"
echo "END TIME = $(date)"
echo "============================================================"
