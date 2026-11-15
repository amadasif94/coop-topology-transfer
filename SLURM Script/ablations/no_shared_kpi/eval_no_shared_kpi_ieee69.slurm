#!/bin/bash
#SBATCH --job-name=A1_eval_ieee69
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=/network/rit/lab/dahlin_lab/amad/graph_marl_dess/results/ablations/ieee69/slurm_logs/A1_eval_%A_%a.out
#SBATCH --error=/network/rit/lab/dahlin_lab/amad/graph_marl_dess/results/ablations/ieee69/slurm_logs/A1_eval_%A_%a.err

set -euo pipefail

STAGE="${1:?Missing stage}"
JOB_MAP="${2:?Missing job map}"
RUN_ID="${SLURM_ARRAY_TASK_ID:?Missing run ID}"

cd /network/rit/lab/dahlin_lab/amad/graph_marl_dess

source /network/rit/lab/dahlin_lab/anaconda3/etc/profile.d/conda.sh
conda activate /network/rit/lab/dahlin_lab/amad/amad_thesis/env/thesis_conda

export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1

export TMPDIR="${SLURM_TMPDIR:-/tmp}/ieee69_${SLURM_JOB_ID}_${RUN_ID}"
mkdir -p "$TMPDIR"
export TMP="$TMPDIR"
export TEMP="$TMPDIR"

exec python -u -m experiments.no_shared_kpi.ieee69.slurm_dispatch \
    --variant no_shared_kpi \
    --stage "$STAGE" \
    --run "$RUN_ID" \
    --job-map "$JOB_MAP"
