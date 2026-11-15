#!/bin/bash
#SBATCH --job-name=final_test_baselines
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --output=results/logs/slurm/final_test_baselines_%j.out
#SBATCH --error=results/logs/slurm/final_test_baselines_%j.err

# F2: FINAL TEST of the NON-LEARNING baselines on the December test split (explicit --final).
# No selection.json is used; the manifests record selection provenance 'not_applicable'.
# The Python evaluators enforce the same rules when run outside SLURM.
#
# Submit from the repository root:
#   sbatch --export=ALL,FEEDER=ieee33,CONTROLLER=zero_random "SLURM Script/final_test_baselines.slurm"
# Required:  FEEDER (ieee33|ieee69), CONTROLLER (zero_random|mpc|smpc|both), GMARL_CONDA_SH, GMARL_CONDA_ENV
# Optional:  TOPOLOGIES (default "TP1"), EPISODES (default all), RUN_NAME, SCENARIOS (SMPC, default 100),
#            HORIZON (default 96), GMARL_REPO (default submit dir)

set -euo pipefail

: "${FEEDER:?set FEEDER=ieee33|ieee69}"
: "${CONTROLLER:?set CONTROLLER=zero_random|mpc|smpc|both}"
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

echo "FINAL TEST (baselines) feeder=${FEEDER} controller=${CONTROLLER} topologies=${TOPOLOGIES} episodes=${EPISODES}"
# shellcheck disable=SC2086
if [ "${CONTROLLER}" = "zero_random" ]; then
  case "${FEEDER}" in
    ieee33) SCRIPT=experiments/topology_generalization/run_tp_eval.py ;;
    ieee69) SCRIPT=experiments/topology_generalization/run_tp_eval_ieee69.py ;;
    *) echo "unsupported FEEDER ${FEEDER}"; exit 2 ;;
  esac
  python -u "${SCRIPT}" \
      --baselines_only \
      --final \
      --topologies ${TOPOLOGIES} \
      --episodes "${EPISODES}" \
      --run_name "${RUN_NAME:-final/${FEEDER}/zero_random/${SLURM_JOB_ID:-local}}" \
      --cpu
else
  case "${FEEDER}" in
    ieee33) SCRIPT=experiments/mpc_smpc/optimization/run_ieee33_optimization_baselines.py ;;
    ieee69) SCRIPT=experiments/mpc_smpc/optimization/run_ieee69_optimization_baselines.py ;;
    *) echo "unsupported FEEDER ${FEEDER}"; exit 2 ;;
  esac
  python -u "${SCRIPT}" \
      --controller "${CONTROLLER}" \
      --final \
      --topologies ${TOPOLOGIES} \
      --episodes "${EPISODES}" \
      --horizon "${HORIZON:-96}" \
      --scenarios "${SCENARIOS:-100}" \
      --print_every 10
fi
