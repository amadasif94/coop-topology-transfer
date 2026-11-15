#!/bin/bash
#SBATCH --job-name=FMPC
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/network/rit/lab/dahlin_lab/amad/graph_marl_dess/results/slurm_logs/fmpc/FMPC_%x_%A_%a.out
#SBATCH --error=/network/rit/lab/dahlin_lab/amad/graph_marl_dess/results/slurm_logs/fmpc/FMPC_%x_%A_%a.err

# Launcher for the F-MPC forecasting pipeline, F-MPC and PQ-RBC (maintained). Separate from "SLURM Script/marl.slurm",
# which is unchanged (its 'summary' stage combines all results). The defaults above fit the training/forecast array
# tasks; pass other resources on the sbatch command line (they override #SBATCH), e.g. for F-MPC runs:
#   sbatch --cpus-per-task=4 --mem=32G --time=3-00:00:00 "SLURM Script/forecast_mpc.slurm" fmpc ieee69 test --scales 1.0
# Stages (all write under $MARL_OUTPUT_ROOT, default results/soc_feasible_affine_v2; nothing is ever overwritten):
#   mkdir -p results/slurm_logs/fmpc                                                              (once)
#   sbatch "SLURM Script/forecast_mpc.slurm" check        <feeder>                # AutoGluon + focused tests
# Forecast stages = monthly expanding-window walk-forward cutoffs: oct (fit Jan-Sep), nov (fit Jan-Oct), dec (fit Jan-Nov).
#   sbatch --array=0-5  "SLURM Script/forecast_mpc.slurm" fc_train    ieee33 <oct|nov|dec>   # 66 models in 6 tasks
#   sbatch --array=0-11 "SLURM Script/forecast_mpc.slurm" fc_train    ieee69 <oct|nov|dec>   # 138 models in 12 tasks
#   sbatch --array=0-5  "SLURM Script/forecast_mpc.slurm" fc_generate ieee33 <oct|nov|dec>
#   sbatch --array=0-11 "SLURM Script/forecast_mpc.slurm" fc_generate ieee69 <oct|nov|dec>
#   sbatch "SLURM Script/forecast_mpc.slurm" fc_status   <feeder> <oct|nov|dec>      # completeness of models + forecasts
#   sbatch "SLURM Script/forecast_mpc.slurm" fc_validate <feeder> <val|oct|nov|dec>  # val = oct + nov, per month + pooled
#   sbatch "SLURM Script/forecast_mpc.slurm" fmpc        <feeder> <val|test> [--scales ...] [--max-days N] [--days ...]
#   sbatch "SLURM Script/forecast_mpc.slurm" pq_freeze   <feeder>
#   sbatch "SLURM Script/forecast_mpc.slurm" pqrbc       <feeder> <val|test> [--scales ...] [--max-days N] [--days ...]
# Array tasks: task K of N (--array=0-(N-1)) trains / forecasts the K-th contiguous chunk of the feeder's targets
# (load_node_1..N, pv_node_1..N). To resubmit only some tasks, keep N fixed with FMPC_CHUNKS, e.g.
#   sbatch --array=3-5 --export=ALL,FMPC_CHUNKS=6 "SLURM Script/forecast_mpc.slurm" fc_train ieee33 oct
# Arrays always pass --skip-completed: a target with a record is skipped; a saved predictor without a record is loaded,
# validated and completed (train.recover_target); an incomplete, unloadable, incompatible or recently written folder,
# and a forecast file without a record, are refused (the task fails, nothing is deleted).
# Read-only check of saved predictors without a record (AutoGluon needed; compute node; no fit, writes nothing):
#   sbatch --cpus-per-task=2 --mem=8G --time=00:30:00 "SLURM Script/forecast_mpc.slurm" fc_verify_saved <feeder> <stage>
# Status without PyTorch (also on the login node):
#   python -m evaluation.forecast_mpc.status --feeder <feeder> --stage <stage> --output-root <root> [--chunks N]

set -euo pipefail

BASE_DIR="/network/rit/lab/dahlin_lab/amad/graph_marl_dess"
CONDA_ENV="/network/rit/lab/dahlin_lab/amad/amad_thesis/env/thesis_conda"

STAGE="${1:?stage: check fc_train fc_verify_saved fc_generate fc_status fc_validate fmpc pq_freeze pqrbc}"
FEEDER="${2:?feeder: ieee33 or ieee69}"
OUTPUT_ROOT="${MARL_OUTPUT_ROOT:-results/soc_feasible_affine_v2}"
case "${OUTPUT_ROOT}" in
    results/soc_feasible|results/soc_feasible/|*/results/soc_feasible|*/results/soc_feasible/)
        echo "ERROR: ${OUTPUT_ROOT} holds the earlier zero-preserving soc_feasible results; use a fresh root"; exit 1 ;;
esac
case "${FEEDER}" in ieee33|ieee69) ;; *) echo "ERROR: unknown feeder ${FEEDER}"; exit 1 ;; esac
ARG3="-"
case "${STAGE}" in
    fc_train|fc_generate|fc_status|fc_verify_saved)
        ARG3="${3:?forecast stage: oct, nov or dec}"
        case "${ARG3}" in oct|nov|dec) ;; *) echo "ERROR: forecast stage must be oct, nov or dec"; exit 1 ;; esac
        EXTRA=("${@:4}") ;;
    fc_validate)
        ARG3="${3:?forecast stage: val (= oct nov), oct, nov or dec}"
        case "${ARG3}" in val|oct|nov|dec) ;; *) echo "ERROR: forecast stage must be val, oct, nov or dec"; exit 1 ;; esac
        EXTRA=("${@:4}") ;;
    fmpc|pqrbc)
        ARG3="${3:?split: val or test}"
        case "${ARG3}" in val|test) ;; *) echo "ERROR: split must be val or test"; exit 1 ;; esac
        EXTRA=("${@:4}") ;;
    check|pq_freeze) EXTRA=("${@:3}") ;;
    *) echo "ERROR: unknown stage ${STAGE}"; exit 1 ;;
esac
CHUNK_ARGS=()
case "${STAGE}" in
    fc_train|fc_generate)
        if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
            if [ -n "${FMPC_CHUNKS:-}" ]; then
                if [ "${SLURM_ARRAY_TASK_ID}" -ge "${FMPC_CHUNKS}" ]; then
                    echo "ERROR: array task ${SLURM_ARRAY_TASK_ID} outside 0..$(( FMPC_CHUNKS - 1 ))"; exit 1
                fi
                CHUNKS="${FMPC_CHUNKS}"
            elif [ "${SLURM_ARRAY_TASK_MIN:-0}" != "0" ] || [ "$(( SLURM_ARRAY_TASK_MAX + 1 ))" != "${SLURM_ARRAY_TASK_COUNT}" ]; then
                echo "ERROR: use a contiguous array 0-(N-1), or set FMPC_CHUNKS=N to resubmit a subset of tasks"; exit 1
            else
                CHUNKS="${SLURM_ARRAY_TASK_COUNT}"
            fi
            CHUNK_ARGS=(--chunk "${SLURM_ARRAY_TASK_ID}" --chunks "${CHUNKS}" --skip-completed)
        else
            CHUNK_ARGS=(--skip-completed)
        fi ;;
esac

cd "${BASE_DIR}"
echo "STAGE=${STAGE} FEEDER=${FEEDER} ARG=${ARG3} OUTPUT_ROOT=${OUTPUT_ROOT} CHUNK=${CHUNK_ARGS[*]:-} EXTRA=${EXTRA[*]:-} HOST=$(hostname) CPUS=${SLURM_CPUS_PER_TASK:-?} START=$(date)"

JOB_TMP="${TMPDIR:-/tmp}/fmpc_${SLURM_JOB_ID:-local}_${SLURM_ARRAY_TASK_ID:-0}"
mkdir -p "${JOB_TMP}"
export TMPDIR="${JOB_TMP}" TEMP="${JOB_TMP}" TMP="${JOB_TMP}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}" PYTHONDONTWRITEBYTECODE=1
trap 'rm -rf "${JOB_TMP}"' EXIT

source /network/rit/lab/dahlin_lab/anaconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV}"
export PYTHONPATH="${BASE_DIR}:${PYTHONPATH:-}"
python --version

case "${STAGE}" in
    check)
        python -c "import autogluon.tabular as a, numpy, pandas, scipy; print('autogluon.tabular', a.__version__, 'numpy', numpy.__version__, 'pandas', pandas.__version__, 'scipy', scipy.__version__)"
        python -m unittest tests.test_forecast_mpc tests.test_pq_rbc tests.test_pf_mpc -v ;;
    fc_train)    python -u -m evaluation.forecast_mpc.train    --feeder "${FEEDER}" --stage "${ARG3}" --output-root "${OUTPUT_ROOT}" "${CHUNK_ARGS[@]}" "${EXTRA[@]}" ;;
    fc_verify_saved) python -u -m evaluation.forecast_mpc.train --feeder "${FEEDER}" --stage "${ARG3}" --output-root "${OUTPUT_ROOT}" --verify-saved "${EXTRA[@]}" ;;
    fc_generate) python -u -m evaluation.forecast_mpc.generate --feeder "${FEEDER}" --stage "${ARG3}" --output-root "${OUTPUT_ROOT}" "${CHUNK_ARGS[@]}" "${EXTRA[@]}" ;;
    fc_status)
        rc=0
        python -u -m evaluation.forecast_mpc.status --feeder "${FEEDER}" --stage "${ARG3}" --output-root "${OUTPUT_ROOT}" "${EXTRA[@]}" || rc=1
        echo "FMPC_STATUS_RC ${rc}" ;;
    fc_validate)
        if [ "${ARG3}" = "val" ]; then VALIDATE_STAGES=(oct nov); else VALIDATE_STAGES=("${ARG3}"); fi
        python -u -m evaluation.forecast_mpc.validate --feeder "${FEEDER}" --stage "${VALIDATE_STAGES[@]}" --output-root "${OUTPUT_ROOT}" "${EXTRA[@]}" ;;
    fmpc)        python -u -m evaluation.forecast_mpc.run      --feeder "${FEEDER}" --split "${ARG3}" --output-root "${OUTPUT_ROOT}" "${EXTRA[@]}" ;;
    pq_freeze)   python -u -m evaluation.rule_based.pq_rbc freeze --feeder "${FEEDER}" --output-root "${OUTPUT_ROOT}" "${EXTRA[@]}" ;;
    pqrbc)       python -u -m evaluation.rule_based.pq_rbc run    --feeder "${FEEDER}" --split "${ARG3}" --output-root "${OUTPUT_ROOT}" "${EXTRA[@]}" ;;
esac
echo "FMPC_STAGE_COMPLETED ${STAGE} ${FEEDER} ${ARG3} task_${SLURM_ARRAY_TASK_ID:-0} END=$(date)"
