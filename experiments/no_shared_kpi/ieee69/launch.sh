#!/bin/bash
# Shared IEEE69 A1/A2 launcher. The active IEEE33 scripts are not imported/edited.
set -euo pipefail
VARIANT=${1:?variant required}
STAGE=${2:-launch_check}
BASE_DIR=/network/rit/lab/dahlin_lab/amad/graph_marl_dess
JOB_MAP_PATH=${3:-$BASE_DIR/experiments/no_shared_kpi/ieee69/cluster_job_map.json}
RUN_ID=${SLURM_ARRAY_TASK_ID:-0}
case "$VARIANT" in no_shared_kpi|no_centralized_critic) ;; *) exit 64 ;; esac
case "$STAGE" in launch_check|preflight|validate|train|sweep|select|evaluate) ;; *) exit 64 ;; esac
printf 'ABLATION_BATCH_ENTERED feeder=IEEE69 job=%s run=%s stage=%s\n' "${SLURM_JOB_ID:-manual}" "$RUN_ID" "$STAGE"
cd "$BASE_DIR"
test -r "$JOB_MAP_PATH"
if [[ "$STAGE" == launch_check ]]; then
    echo LAUNCH_CHECK_PASS
    exit 0
fi
TMP_BASE=/network/rit/dgx/dgx_dahlin_lab/amad/graph_marl_dess/tmp
CACHE_BASE=/network/rit/dgx/dgx_dahlin_lab/amad/graph_marl_dess/pip_cache
export TMPDIR="$TMP_BASE/ieee69_ablations/${SLURM_JOB_ID:-manual}_${RUN_ID}"
export TEMP="$TMPDIR" TMP="$TMPDIR"
export PIP_CACHE_DIR="$CACHE_BASE/ieee69_ablations/${SLURM_JOB_ID:-manual}_${RUN_ID}"
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 BLIS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
source /network/rit/lab/dahlin_lab/anaconda3/etc/profile.d/conda.sh
conda activate /network/rit/lab/dahlin_lab/amad/amad_thesis/env/thesis_conda
export PYTHONPATH="$BASE_DIR:${PYTHONPATH:-}"
python --version
if [[ "$STAGE" == train ]]; then
    python - <<'PY'
import torch
if not torch.cuda.is_available():
    raise RuntimeError('Full training requires its allocated GPU; no silent CPU fallback')
print('CUDA_TRAINING_PREFLIGHT_PASS', torch.cuda.get_device_name(0))
PY
fi
exec python -u -m experiments.no_shared_kpi.ieee69.slurm_dispatch \
    --variant "$VARIANT" --stage "$STAGE" --run "$RUN_ID" --job-map "$JOB_MAP_PATH"
