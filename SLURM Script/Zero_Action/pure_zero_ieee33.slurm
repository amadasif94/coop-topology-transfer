#!/bin/bash
#SBATCH --job-name=PZ33
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=/network/rit/lab/dahlin_lab/amad/graph_marl_dess/results/load_stress/ieee33/pure_zero_no_backup/slurm_logs/PZ33_%j.out
#SBATCH --error=/network/rit/lab/dahlin_lab/amad/graph_marl_dess/results/load_stress/ieee33/pure_zero_no_backup/slurm_logs/PZ33_%j.err

# IEEE33 PURE zero-action reference (all batteries 0 kW, no backup), TP1-TP7, 31 December days (CPU, batch).
# The existing zero-action + backup runs are not touched. Create the log directory before the first submission:
#   mkdir -p results/load_stress/ieee33/pure_zero_no_backup/slurm_logs
# Usage (single job, no array):
#   sbatch "SLURM Script/pure_zero_ieee33.slurm" full       <JOB_MAP> [SCALES]   # default "1.0 1.1 1.2 1.3"
#   sbatch "SLURM Script/pure_zero_ieee33.slurm" pilot      <JOB_MAP>            # 1.3x, TP1, 2 days; optional
#   sbatch "SLURM Script/pure_zero_ieee33.slurm" crosscheck <JOB_MAP> [SCALES]   # optional, non-blocking
#   sbatch "SLURM Script/pure_zero_ieee33.slurm" summary    <JOB_MAP>
# Stages: launch_check | preflight | full | pilot | crosscheck | summary. Checks are enforced in Python.

set -euo pipefail

BASE_DIR="/network/rit/lab/dahlin_lab/amad/graph_marl_dess"
CONDA_ENV="/network/rit/lab/dahlin_lab/amad/amad_thesis/env/thesis_conda"
OUTPUT_ROOT="results/load_stress/ieee33/pure_zero_no_backup"

STAGE="${1:?Specify launch_check, preflight, full, pilot, crosscheck, or summary}"
JOB_MAP_PATH="${2:?Specify the absolute IEEE33 JOB_MAP path}"
SCALES="${3:-1.0 1.1 1.2 1.3}"

cd "${BASE_DIR}"

echo "============================================================"
echo "SLURM_JOB_ID        = ${SLURM_JOB_ID:-none}"
echo "STAGE               = ${STAGE}"
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
    launch_check|preflight|full|pilot|crosscheck|summary) ;;
    *) echo "ERROR: Unknown stage: ${STAGE}"; exit 1 ;;
esac
for SCALE in ${SCALES}; do
    case "${SCALE}" in 1.0|1.1|1.2|1.3) ;; *) echo "ERROR: load scale must be 1.0, 1.1, 1.2 or 1.3: ${SCALE}"; exit 1 ;; esac
done

if [[ "${STAGE}" == "launch_check" ]]; then
    echo "LAUNCH_CHECK_PASS"
    exit 0
fi

# CPU node: node-local scratch only, no DGX paths; one thread per numerical library.
JOB_TMP="${TMPDIR:-/tmp}/pz33_${SLURM_JOB_ID:-local}"
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

EVAL=(python -u -m experiments.load_stress.evaluate_pure_zero_ieee33 --job-map "${JOB_MAP_PATH}" --output-root "${OUTPUT_ROOT}")
LABEL() { python -c "import sys; from experiments.load_stress.stress_env import scale_label; print(scale_label(float(sys.argv[1])))" "$1"; }

case "${STAGE}" in
    preflight)
        python - <<'PY'
from experiments.load_stress import evaluate_pure_zero_ieee33, validate_pure_zero_ieee33, summarize_pure_zero_ieee33
print("PREFLIGHT_PASS")
PY
        ;;
    full)
        for SCALE in ${SCALES}; do
            echo "---- scale ${SCALE} ----"
            "${EVAL[@]}" --scale "${SCALE}"
        done
        ;;
    pilot)
        "${EVAL[@]}" --scale 1.3 --pilot || echo "PILOT_NOT_PASSED (informational; full runs are not blocked)"
        ;;
    crosscheck)
        for SCALE in ${SCALES}; do
            python -u -m experiments.load_stress.validate_pure_zero_ieee33 --pure-dir "${OUTPUT_ROOT}/$(LABEL "${SCALE}")" \
                || echo "CROSS_CHECK_NOT_PASSED ${SCALE}: inspect cross_check_report.json"
        done
        ;;
    summary)
        STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
        python -u -m experiments.load_stress.summarize_pure_zero_ieee33 --root "${OUTPUT_ROOT}" \
            --output "${OUTPUT_ROOT}/summary_${STAMP}"
        ;;
esac

echo "============================================================"
echo "PZ33_STAGE_COMPLETED ${STAGE}"
echo "END TIME = $(date)"
echo "============================================================"
