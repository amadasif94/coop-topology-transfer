#!/bin/bash
#SBATCH --job-name=FMPC_GRB
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=3-00:00:00
# ABSOLUTE log paths (independent of the sbatch submission directory). Create the folder BEFORE sbatch:
#   mkdir -p /network/rit/lab/dahlin_lab/amad/graph_marl_dess/logs/fmpc_gurobi
#SBATCH --output=/network/rit/lab/dahlin_lab/amad/graph_marl_dess/logs/fmpc_gurobi/FMPC_GRB_%A_%a.out
#SBATCH --error=/network/rit/lab/dahlin_lab/amad/graph_marl_dess/logs/fmpc_gurobi/FMPC_GRB_%A_%a.err
#
# Gurobi F-MPC, one December topology per array task (IEEE33: TP1-TP7, IEEE69: TP1-TP10):
#   sbatch                      fmpc_gurobi.slurm smoke ieee69 130 TP1     # first December day only (_smoke1 folder)
#   sbatch --array=1-10         fmpc_gurobi.slurm task  ieee69 130         # IEEE69, all 31 days, TP<array index>
#   sbatch --array=1-7          fmpc_gurobi.slurm task  ieee33 130         # IEEE33
#   sbatch                      fmpc_gurobi.slurm bench ieee69 130 TP1     # capture steps 0 1 48 + HiGHS/Gurobi comparison
# Status (login node, no Slurm job): python -m evaluation.forecast_mpc.gurobi_run status --feeder ieee69 --scale 1.3
# Gurobi Threads = SLURM_CPUS_PER_TASK (set by --cpus-per-task above); Seed 42; MIPGap 1e-4; TimeLimit 60 s per call.
# Optional: export GUROBI_MODULE=<module name> before sbatch if gurobipy/licence come from an environment module.
set -eo pipefail
cd /network/rit/lab/dahlin_lab/amad/graph_marl_dess
source /network/rit/lab/dahlin_lab/anaconda3/etc/profile.d/conda.sh
conda activate /network/rit/lab/dahlin_lab/amad/amad_thesis/env/thesis_conda
if [ -n "${GUROBI_MODULE:-}" ]; then module load "${GUROBI_MODULE}"; fi
set -u
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTHONDONTWRITEBYTECODE=1
SCALE="$(python -c "import sys; print(int(sys.argv[1]) / 100)" "${3:?load percent, e.g. 130}")"
case "${1:-}" in
    smoke) exec python -u -m evaluation.forecast_mpc.gurobi_run task --feeder "${2:?feeder}" --scale "${SCALE}" \
               --topology "${4:?topology}" --max-days 1 ;;
    task)  exec python -u -m evaluation.forecast_mpc.gurobi_run task --feeder "${2:?feeder}" --scale "${SCALE}" \
               --topology "TP${SLURM_ARRAY_TASK_ID:?submit with --array (1-7 ieee33, 1-10 ieee69)}" ;;
    bench) OUT="/network/rit/lab/dahlin_lab/amad/graph_marl_dess/results/fmpc_gurobi_benchmark/${SLURM_JOB_ID}"
           python -u -m evaluation.forecast_mpc.gurobi_benchmark capture --feeder "${2:?feeder}" --scale "${SCALE}" \
               --topology "${4:-TP1}" --steps 0 1 48 --out "${OUT}"
           exec python -u -m evaluation.forecast_mpc.gurobi_benchmark compare --out "${OUT}" --repeats 3 ;;
    *) echo "Expected: smoke FEEDER PCT TP | task FEEDER PCT (with --array) | bench FEEDER PCT [TP]" >&2; exit 2 ;;
esac
