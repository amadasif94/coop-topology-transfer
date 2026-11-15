#!/bin/bash
#SBATCH --job-name=tp_tag33_best
#SBATCH --partition=batch
#SBATCH --array=0-2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=/network/rit/lab/dahlin_lab/amad/graph_marl_dess/results/logs/slurm/tp_tagconv33_best_%A_%a.out
#SBATCH --error=/network/rit/lab/dahlin_lab/amad/graph_marl_dess/results/logs/slurm/tp_tagconv33_best_%A_%a.err

set -euo pipefail

# IEEE33 TAGCONV-MADDPG final zero-shot evaluation; TP1-TP7, 31 test episodes.
# Each array task uses one of three independently frozen run selections.
# Frozen TP1-validation selected steps: pending selection.
# DO NOT submit before all three TAGConv selection.json files and best.pt copies exist.

BASE_DIR="/network/rit/lab/dahlin_lab/amad/graph_marl_dess"
CONDA_SH="/network/rit/lab/dahlin_lab/anaconda3/etc/profile.d/conda.sh"
CONDA_ENV="/network/rit/lab/dahlin_lab/amad/amad_thesis/env/thesis_conda"
MODEL="tagconv"
RUN_ID="${SLURM_ARRAY_TASK_ID:?Must run as a SLURM array task}"

cd "${BASE_DIR}"

# Private temporary/cache directories for simultaneous array tasks.
JOB_TMP="${BASE_DIR}/tmp/job_${SLURM_JOB_ID}_${RUN_ID}"
JOB_PIP_CACHE="${BASE_DIR}/pip_cache/job_${SLURM_JOB_ID}_${RUN_ID}"
mkdir -p "${BASE_DIR}/results/logs/slurm" "${JOB_TMP}" "${JOB_PIP_CACHE}"
export TMPDIR="${JOB_TMP}" TEMP="${JOB_TMP}" TMP="${JOB_TMP}"
export PIP_CACHE_DIR="${JOB_PIP_CACHE}" RAY_TMPDIR="${JOB_TMP}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1

source "${CONDA_SH}"
conda activate "${CONDA_ENV}"
export PYTHONPATH="${BASE_DIR}:${PYTHONPATH:-}"

# Frozen validation selection: one record per model and independent seed.
# Avoid choosing a different record if a run was reselected later.
shopt -s nullglob
SELECTION_FILES=("${BASE_DIR}/results/selections/ieee33/${MODEL}/run_${RUN_ID}/"*/selection.json)
shopt -u nullglob
if (( ${#SELECTION_FILES[@]} != 1 )); then
    echo "ERROR: Expected one frozen selection.json for ${MODEL} run ${RUN_ID}; found ${#SELECTION_FILES[@]}" >&2
    exit 1
fi
SELECTION_PATH="${SELECTION_FILES[0]}"
BEST_CHECKPOINT="${BASE_DIR}/results/models/IEEE33/${MODEL}/run_${RUN_ID}_best.pt"
RUN_NAME="${MODEL}_ieee33_tp_full_eval_run${RUN_ID}_best"

# select_checkpoint.py writes its path inside selection['selected'].
# Validate the original and copied checkpoint against the immutable
# selected SHA-256, and pass the ORIGINAL checkpoint to the evaluator.
CHECKPOINT_PATH="$(python - "${SELECTION_PATH}" "${MODEL}" "${RUN_ID}" "${BEST_CHECKPOINT}" <<'PY_CHECK'
import hashlib
import json
import sys
from pathlib import Path

selection_path, model, run_id, best_path = sys.argv[1:]
selection = json.loads(Path(selection_path).read_text(encoding='utf-8'))
required = {
    'feeder': 'ieee33', 'model': model, 'run': f'run_{run_id}',
    'topology': 'TP1', 'data_split': 'val',
    'run_purpose': 'checkpoint_selection'
}
for key, expected in required.items():
    if selection.get(key) != expected:
        raise SystemExit(f'ERROR: selection.json {key} mismatch: {selection.get(key)!r} != {expected!r}')
chosen = selection['selected']
checkpoint = Path(chosen['checkpoint_path'])
expected_hash = chosen['checkpoint_sha256']

def sha256(path):
    if not path.is_file():
        raise SystemExit(f'ERROR: Required checkpoint missing: {path}')
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

for path in (checkpoint, Path(best_path)):
    if sha256(path) != expected_hash:
        raise SystemExit(f'ERROR: Frozen checkpoint SHA-256 mismatch: {path}')
print(checkpoint)
PY_CHECK
)"

echo "============================================================"
echo "IEEE33 TAGCONV FINAL ZERO-SHOT TOPOLOGY EVALUATION"
echo "SLURM_JOB_ID   = ${SLURM_JOB_ID}"
echo "RUN_ID         = ${RUN_ID}"
echo "HOST           = $(hostname)"
echo "CHECKPOINT     = ${CHECKPOINT_PATH}"
echo "BEST COPY      = ${BEST_CHECKPOINT}"
echo "SELECTION      = ${SELECTION_PATH}"
echo "RUN_NAME       = ${RUN_NAME}"
echo "MODE           = test / final"
echo "TOPOLOGIES     = TP1 TP2 TP3 TP4 TP5 TP6 TP7"
echo "EPISODES       = 31"
echo "START          = $(date)"
echo "============================================================"

# Held-out TP2-TP7 are authorized only with test + final + frozen selection.
python -u experiments/topology_generalization/run_tp_eval.py \
  --checkpoint "${CHECKPOINT_PATH}" \
  --selection "${SELECTION_PATH}" \
  --gnn_type "${MODEL}" \
  --topologies TP1 TP2 TP3 TP4 TP5 TP6 TP7 \
  --episodes 31 \
  --mode test \
  --final \
  --include_zero \
  --include_random \
  --run_name "${RUN_NAME}" \
  --cpu


echo "COMPLETED IEEE33 TAGCONV run ${RUN_ID} at $(date)"
