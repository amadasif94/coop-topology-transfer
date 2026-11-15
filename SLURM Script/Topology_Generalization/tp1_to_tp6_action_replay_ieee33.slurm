#!/bin/bash
#SBATCH --job-name=ieee33_tp6_replay
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=results/logs/slurm/ieee33_tp6_replay_%j.out
#SBATCH --error=results/logs/slurm/ieee33_tp6_replay_%j.err

# IEEE33 TP1 -> TP1 identity replay and TP1 -> TP6 fixed-action replay (CPU; no actor inference), plus analyses.
# Submit from the repository root. Execution order is enforced by the code (TP6 needs PASS reports):
#   STAGE=check                                           syntax + imports + TP6 definition
#   STAGE=config    OUT                                   IEEE33 configuration report (base vs saved final runs)
#   STAGE=zero      OUT [EXTRA_DIR]                       zero-action completeness + network stress (no simulation)
# Original evaluations: results/topology_generalization/csv/<model>_ieee33_tp_full_eval_run<run>_best/ (run from the selection).
#   STAGE=identity  MODEL SELECTION RUN_ID [MAX_EPISODES=1]                   replay TP1 + identity validation
#   STAGE=tp6       MODEL SELECTION RUN_ID IDENTITY_REPORT [SMOKE_REPORT] [MAX_EPISODES=1]   replay TP6 + target validation
#   STAGE=validate  KIND=identity|target MODEL REPLAY_DIR                     re-run one validation
#   STAGE=clipping  OUT CLIP_PAIRS="gcn 0 <identity dir> <tp6 dir> ; ..."     physical-clipping diagnostics
#   STAGE=compare   OUT COMPARE_PAIRS="gcn 0 <tp6 full dir> ; ..."            paired replay - feedback analysis
# Required for every stage: GMARL_CONDA_SH, GMARL_CONDA_ENV.
set -euo pipefail
: "${STAGE:?}"; : "${GMARL_CONDA_SH:?}"; : "${GMARL_CONDA_ENV:?}"
REPO_DIR="${GMARL_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 MPLBACKEND=Agg
source "${GMARL_CONDA_SH}"
conda activate "${GMARL_CONDA_ENV}"
cd "${REPO_DIR}"
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"
REPLAY_ROOT=results/topology_generalization/csv/action_replay_ieee33
TG=experiments/topology_generalization
NEW_FILES=("${TG}/replay_utils_ieee33.py" "${TG}/run_tp1_action_replay_ieee33.py" experiments/no_gnn_maddpg/run_tp1_action_replay_mlp_ieee33.py
           "${TG}/validate_tp1_replay_identity_ieee33.py" "${TG}/compare_tp6_replay_vs_feedback_ieee33.py"
           "${TG}/diagnose_physical_clipping_ieee33.py" "${TG}/zero_action_stress_ieee33.py" "${TG}/inspect_ieee33_replay_config.py")

evaluator() { if [ "$1" = mlp ]; then echo experiments/no_gnn_maddpg/run_tp1_action_replay_mlp_ieee33.py;
              else echo "${TG}/run_tp1_action_replay_ieee33.py"; fi; }
arch_args() { if [ "$1" != mlp ]; then echo "--gnn_type $1"; fi; }
selection_id() { python - "$1" "$2" <<'PY'
import json, sys
rec = json.load(open(sys.argv[1]))
if rec.get("feeder") != "ieee33" or rec.get("model") != sys.argv[2]:
    sys.exit(f"selection is for {rec.get('feeder')}/{rec.get('model')}, not ieee33/{sys.argv[2]}")
print(rec["selection_id"])
PY
}
episodes_args() { if [ -n "${MAX_EPISODES:-}" ]; then echo "--max_episodes ${MAX_EPISODES}"; fi; }
split_pairs() { local -n _out=$1; IFS=';' read -ra P <<< "$2"; for p in "${P[@]}"; do read -ra F <<< "${p}"; _out+=(--pair "${F[@]}"); done; }

case "${STAGE}" in
  check)
    python -m py_compile "${NEW_FILES[@]}"
    python - <<'PY'
import importlib
for m in ("replay_utils_ieee33", "run_tp1_action_replay_ieee33", "validate_tp1_replay_identity_ieee33",
          "compare_tp6_replay_vs_feedback_ieee33", "diagnose_physical_clipping_ieee33", "zero_action_stress_ieee33",
          "inspect_ieee33_replay_config"):
    importlib.import_module("experiments.topology_generalization." + m); print("import OK", m)
importlib.import_module("experiments.no_gnn_maddpg.run_tp1_action_replay_mlp_ieee33"); print("import OK run_tp1_action_replay_mlp_ieee33")
from experiments.topology_generalization.tp_configs import get_topology_case
from experiments.topology_generalization.replay_utils_ieee33 import TP6
c = get_topology_case("TP6")
ok = sorted(map(tuple, c["open_edges"])) == TP6["open_edges"] and sorted(map(tuple, c["close_edges"])) == TP6["close_edges"]
print("TP6:", c, "matches specification:", ok)
raise SystemExit(0 if ok else 5)
PY
    ;;
  config)
    : "${OUT:?}"; python -u "${TG}/inspect_ieee33_replay_config.py" --out "${OUT}" ;;
  zero)
    : "${OUT:?}"; ZX=(); if [ -n "${EXTRA_DIR:-}" ]; then ZX=(--extra_dir "${EXTRA_DIR}"); fi
    python -u "${TG}/zero_action_stress_ieee33.py" --out "${OUT}" ${ZX[@]+"${ZX[@]}"} ;;
  identity|tp6)
    : "${MODEL:?}"; : "${SELECTION:?}"; : "${RUN_ID:?set RUN_ID, e.g. identity_smoke}"
    SEL_ID=$(selection_id "${SELECTION}" "${MODEL}")
    TARGET=TP1; KIND=identity; EXTRA=()
    if [ "${STAGE}" = tp6 ]; then
      : "${IDENTITY_REPORT:?}"; TARGET=TP6; KIND=target; EXTRA=(--identity_report "${IDENTITY_REPORT}")
      if [ -n "${SMOKE_REPORT:-}" ]; then EXTRA+=(--smoke_report "${SMOKE_REPORT}"); fi
    fi
    # shellcheck disable=SC2046
    python -u "$(evaluator "${MODEL}")" $(arch_args "${MODEL}") --selection "${SELECTION}" --target_topology "${TARGET}" \
        --run_id "${RUN_ID}" $(episodes_args) ${EXTRA[@]+"${EXTRA[@]}"}
    python -u "${TG}/validate_tp1_replay_identity_ieee33.py" --kind "${KIND}" --model "${MODEL}" \
        --replay_dir "${REPLAY_ROOT}/${TARGET}/${MODEL}/${SEL_ID}/${RUN_ID}"
    ;;
  validate)
    : "${KIND:?}"; : "${MODEL:?}"; : "${REPLAY_DIR:?}"
    python -u "${TG}/validate_tp1_replay_identity_ieee33.py" --kind "${KIND}" --model "${MODEL}" --replay_dir "${REPLAY_DIR}" ;;
  clipping)
    : "${OUT:?}"; : "${CLIP_PAIRS:?}"; ARGS=(); split_pairs ARGS "${CLIP_PAIRS}"
    python -u "${TG}/diagnose_physical_clipping_ieee33.py" "${ARGS[@]}" --out "${OUT}" ;;
  compare)
    : "${OUT:?}"; : "${COMPARE_PAIRS:?}"; ARGS=(); split_pairs ARGS "${COMPARE_PAIRS}"
    python -u "${TG}/compare_tp6_replay_vs_feedback_ieee33.py" "${ARGS[@]}" --out "${OUT}" ;;
  *) echo "unknown STAGE ${STAGE}"; exit 2 ;;
esac
