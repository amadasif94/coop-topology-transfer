#!/bin/bash
#SBATCH --job-name=final_test_learned
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --output=results/logs/slurm/final_test_learned_%j.out
#SBATCH --error=results/logs/slurm/final_test_learned_%j.err

# F2: FINAL TEST of ONE learned controller on the December test split (explicit --final).
# Requires the frozen selection.json written by scripts/select_checkpoint.py; the evaluated
# checkpoint is taken from it and verified by sha256 inside the Python evaluator (which enforces
# the same rules when run outside SLURM). TP2 onward are zero-shot assessments only.
#
# Submit from the repository root (log paths are relative to it):
#   sbatch --export=ALL,FEEDER=ieee33,MODEL=gcn,SELECTION=results/selections/.../selection.json "SLURM Script/final_test_learned.slurm"
# Required:  FEEDER (ieee33|ieee69), MODEL (gcn|gat|tagconv|mlp), SELECTION, GMARL_CONDA_SH, GMARL_CONDA_ENV
# Optional:  TOPOLOGIES (default "TP1"; e.g. "TP1 TP2 TP3 TP4 TP5 TP6 TP7"), EPISODES (default all),
#            RUN_NAME (default final/<feeder>/<model>/<selection id>), GMARL_REPO (default submit dir)

set -euo pipefail

: "${FEEDER:?set FEEDER=ieee33|ieee69}"
: "${MODEL:?set MODEL=gcn|gat|tagconv|mlp}"
: "${SELECTION:?set SELECTION=<frozen selection.json>}"
: "${GMARL_CONDA_SH:?set GMARL_CONDA_SH=<conda.sh to source>}"
: "${GMARL_CONDA_ENV:?set GMARL_CONDA_ENV=<conda environment>}"
REPO_DIR="${GMARL_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
TOPOLOGIES="${TOPOLOGIES:-TP1}"
EPISODES="${EPISODES:-all}"

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
source "${GMARL_CONDA_SH}"
conda activate "${GMARL_CONDA_ENV}"
cd "${REPO_DIR}"
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"

SELECTION_ID=$(python - "${SELECTION}" "${FEEDER}" "${MODEL}" <<'PY'
import json, sys
rec = json.load(open(sys.argv[1]))
if rec.get("feeder") != sys.argv[2] or rec.get("model") != sys.argv[3]:
    sys.exit(f"selection is for {rec.get('feeder')}/{rec.get('model')}, not {sys.argv[2]}/{sys.argv[3]}")
print(rec["selection_id"])
PY
)
RUN_NAME="${RUN_NAME:-final/${FEEDER}/${MODEL}/${SELECTION_ID}}"

case "${FEEDER}:${MODEL}" in
  ieee33:mlp)      SCRIPT=experiments/no_gnn_maddpg/run_tp_eval_mlp.py;          ARCH=() ;;
  ieee69:mlp)      SCRIPT=experiments/no_gnn_maddpg/run_tp_eval_mlp_ieee69.py;   ARCH=() ;;
  ieee33:gcn|ieee33:gat|ieee33:tagconv) SCRIPT=experiments/topology_generalization/run_tp_eval.py;        ARCH=(--gnn_type "${MODEL}") ;;
  ieee69:gcn|ieee69:gat|ieee69:tagconv) SCRIPT=experiments/topology_generalization/run_tp_eval_ieee69.py; ARCH=(--gnn_type "${MODEL}") ;;
  *) echo "unsupported FEEDER/MODEL ${FEEDER}/${MODEL}"; exit 2 ;;
esac

echo "FINAL TEST  feeder=${FEEDER} model=${MODEL} selection=${SELECTION} (${SELECTION_ID})"
echo "            topologies=${TOPOLOGIES} episodes=${EPISODES} run_name=${RUN_NAME} script=${SCRIPT}"
# shellcheck disable=SC2086
python -u "${SCRIPT}" ${ARCH[@]+"${ARCH[@]}"} \
    --final \
    --selection "${SELECTION}" \
    --topologies ${TOPOLOGIES} \
    --episodes "${EPISODES}" \
    --run_name "${RUN_NAME}" \
    --cpu
