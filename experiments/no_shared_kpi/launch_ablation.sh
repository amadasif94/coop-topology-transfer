#!/bin/bash
# Shared batch body. Called by the four NEW ablation launchers, never by A0.
set -euo pipefail
VARIANT=${1:?variant required}
STAGE=${2:?stage required as a positional argument}
JOB_MAP=${3:?job-map path required as a positional argument}
RUN_ID=${4:?run required}
case "$VARIANT" in no_shared_kpi|no_centralized_critic) ;; *) exit 64 ;; esac
case "$RUN_ID" in 0|1|2) ;; *) printf 'Invalid run: %s\n' "$RUN_ID" >&2; exit 64 ;; esac
case "$STAGE" in launch_check|preflight|validate|train|sweep|select|evaluate|analysis) ;;
  *) printf 'Invalid stage: %s\n' "$STAGE" >&2; exit 64 ;; esac

BASE_DIR=/network/rit/lab/dahlin_lab/amad/graph_marl_dess
cd "$BASE_DIR"
test -r "$JOB_MAP" || { printf 'Unreadable job map: %s\n' "$JOB_MAP" >&2; exit 66; }
printf 'ABLATION_SHELL_READY variant=%s stage=%s run=%s seed=%s host=%s\n' \
  "$VARIANT" "$STAGE" "$RUN_ID" "$((42 + RUN_ID))" "$(hostname)"
printf 'JOB_MAP=%s\nCUDA_VISIBLE_DEVICES=%s\n' "$JOB_MAP" "${CUDA_VISIBLE_DEVICES:-unset}"
if [[ "$STAGE" == launch_check ]]; then
  printf 'LAUNCH_CHECK_PASS: Bash entered, repository and job map readable. No Conda/Python/training executed.\n'
  exit 0
fi

# Same filesystem locations as working A0; new subdirectories avoid touching its cache files.
TMP_BASE=/network/rit/dgx/dgx_dahlin_lab/amad/graph_marl_dess/tmp
PIP_CACHE_BASE=/network/rit/dgx/dgx_dahlin_lab/amad/graph_marl_dess/pip_cache
export TMPDIR="$TMP_BASE/ablations/${SLURM_JOB_ID:-manual_${BASHPID}}"
export TEMP="$TMPDIR" TMP="$TMPDIR" PIP_CACHE_DIR="$PIP_CACHE_BASE/ablations"
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 BLIS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
# Controls subsequent job steps only; cannot repair an earlier slurmd retrieval failure.
export SLURM_EXPORT_ENV=ALL
source /network/rit/lab/dahlin_lab/anaconda3/etc/profile.d/conda.sh
conda activate /network/rit/lab/dahlin_lab/amad/amad_thesis/env/thesis_conda
export PYTHONPATH="$BASE_DIR:${PYTHONPATH:-}"
printf 'CONDA_READY python=%s\n' "$(command -v python)"
python --version
python -u -m experiments.no_shared_kpi.launcher_preflight --run "$RUN_ID" --job-map "$JOB_MAP"
if [[ "$STAGE" == preflight ]]; then exit 0; fi
printf 'DISPATCH_BEGIN variant=%s stage=%s run=%s\n' "$VARIANT" "$STAGE" "$RUN_ID"
exec python -u -m experiments.no_shared_kpi.slurm_dispatch \
  --variant "$VARIANT" --stage "$STAGE" --run "$RUN_ID" --job-map "$JOB_MAP"
