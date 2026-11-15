#!/bin/bash
#SBATCH --job-name=tp1_tp10_replay
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=results/logs/slurm/tp1_tp10_replay_%j.out
#SBATCH --error=results/logs/slurm/tp1_tp10_replay_%j.err

# IEEE69 TP1 -> TP1 identity replay and TP1 -> TP10 open-loop action replay (CPU; no actor inference).
# Submit from the repository root. Order is enforced by the code (TP10 needs PASS reports):
#   STAGE=check                                   syntax + imports + TP10 registration
#   STAGE=identity  MODEL SELECTION [MAX_EPISODES=1] RUN_ID     replay TP1 + identity validation
#   STAGE=tp10      MODEL SELECTION FEEDBACK_JOB IDENTITY_REPORT [SMOKE_REPORT] [MAX_EPISODES=1] RUN_ID
#                                                 replay TP10 + target validation
#   STAGE=validate  KIND=identity|target MODEL REPLAY_DIR       re-run a validation only
#   STAGE=compare   COMPARE_OUT COMPARE_PAIRS="mlp 0 <replay dir> job:99786 ; gcn 0 ..."   paired analysis
# Required for every stage: GMARL_CONDA_SH, GMARL_CONDA_ENV.
set -euo pipefail
: "${STAGE:?}"; : "${GMARL_CONDA_SH:?}"; : "${GMARL_CONDA_ENV:?}"
REPO_DIR="${GMARL_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 MPLBACKEND=Agg
source "${GMARL_CONDA_SH}"
conda activate "${GMARL_CONDA_ENV}"
cd "${REPO_DIR}"
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"
REPLAY_ROOT=results/topology_generalization/csv_ieee69/action_replay
NEW_FILES=(experiments/topology_generalization/replay_utils_ieee69.py
           experiments/topology_generalization/run_tp1_action_replay_ieee69.py
           experiments/no_gnn_maddpg/run_tp1_action_replay_mlp_ieee69.py
           experiments/topology_generalization/validate_tp1_replay_identity_ieee69.py
           experiments/topology_generalization/compare_tp10_replay_vs_feedback_ieee69.py)

evaluator() { if [ "$1" = mlp ]; then echo experiments/no_gnn_maddpg/run_tp1_action_replay_mlp_ieee69.py;
              else echo experiments/topology_generalization/run_tp1_action_replay_ieee69.py; fi; }
arch_args() { if [ "$1" = mlp ]; then :; else echo "--gnn_type $1"; fi; }
selection_id() { python - "$1" "$2" <<'PY'
import json, sys
rec = json.load(open(sys.argv[1]))
if rec.get("feeder") != "ieee69" or rec.get("model") != sys.argv[2]:
    sys.exit(f"selection is for {rec.get('feeder')}/{rec.get('model')}, not ieee69/{sys.argv[2]}")
print(rec["selection_id"])
PY
}
episodes_args() { if [ -n "${MAX_EPISODES:-}" ]; then echo "--max_episodes ${MAX_EPISODES}"; fi; }

case "${STAGE}" in
  check)
    python -m py_compile "${NEW_FILES[@]}"
    python - <<'PY'
import importlib
for m in ("experiments.topology_generalization.replay_utils_ieee69",
          "experiments.topology_generalization.run_tp1_action_replay_ieee69",
          "experiments.no_gnn_maddpg.run_tp1_action_replay_mlp_ieee69",
          "experiments.topology_generalization.validate_tp1_replay_identity_ieee69",
          "experiments.topology_generalization.compare_tp10_replay_vs_feedback_ieee69"):
    importlib.import_module(m)
    print("import OK", m)
from experiments.topology_generalization.tp_configs_ieee69 import get_topology_case
from experiments.topology_generalization.replay_utils_ieee69 import TP10
c = get_topology_case("TP10")
ok = sorted(map(tuple, c["open_edges"])) == TP10["open_edges"] and sorted(map(tuple, c["close_edges"])) == TP10["close_edges"]
print("TP10 registered:", c, "matches specification:", ok)
raise SystemExit(0 if ok else 5)
PY
    ;;
  identity|tp10)
    : "${MODEL:?}"; : "${SELECTION:?}"; : "${RUN_ID:?set RUN_ID, e.g. identity_smoke}"
    SEL_ID=$(selection_id "${SELECTION}" "${MODEL}")
    TARGET=TP1; KIND=identity; EXTRA=()
    if [ "${STAGE}" = tp10 ]; then
      : "${FEEDBACK_JOB:?}"; : "${IDENTITY_REPORT:?}"
      TARGET=TP10; KIND=target; EXTRA=(--feedback_job "${FEEDBACK_JOB}" --identity_report "${IDENTITY_REPORT}")
      if [ -n "${SMOKE_REPORT:-}" ]; then EXTRA+=(--smoke_report "${SMOKE_REPORT}"); fi
    elif [ -n "${PRE_TP10_BACKUP:-}" ]; then EXTRA=(--pre_tp10_backup "${PRE_TP10_BACKUP}"); fi
    # shellcheck disable=SC2046
    python -u "$(evaluator "${MODEL}")" $(arch_args "${MODEL}") --selection "${SELECTION}" --target_topology "${TARGET}" \
        --run_id "${RUN_ID}" $(episodes_args) ${EXTRA[@]+"${EXTRA[@]}"}
    python -u experiments/topology_generalization/validate_tp1_replay_identity_ieee69.py --kind "${KIND}" --model "${MODEL}" \
        --replay_dir "${REPLAY_ROOT}/${TARGET}/${MODEL}/${SEL_ID}/${RUN_ID}"
    ;;
  validate)
    : "${KIND:?}"; : "${MODEL:?}"; : "${REPLAY_DIR:?}"
    python -u experiments/topology_generalization/validate_tp1_replay_identity_ieee69.py --kind "${KIND}" --model "${MODEL}" --replay_dir "${REPLAY_DIR}"
    ;;
  compare)
    : "${COMPARE_OUT:?}"; : "${COMPARE_PAIRS:?}"
    ARGS=()
    IFS=';' read -ra PAIRS <<< "${COMPARE_PAIRS}"
    for p in "${PAIRS[@]}"; do read -ra F <<< "${p}"; ARGS+=(--pair "${F[@]}"); done
    python -u experiments/topology_generalization/compare_tp10_replay_vs_feedback_ieee69.py "${ARGS[@]}" --out "${COMPARE_OUT}" \
        ${TERMINAL_ENERGY_VALUATION:+--terminal_energy_valuation "${TERMINAL_ENERGY_VALUATION}"}
    ;;
  *) echo "unknown STAGE ${STAGE}"; exit 2 ;;
esac
