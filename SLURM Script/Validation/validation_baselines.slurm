#!/bin/bash
#SBATCH --job-name=validation_baselines
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=results/logs/slurm/validation_baselines_%j.out
#SBATCH --error=results/logs/slurm/validation_baselines_%j.err

# F2/F16: zero and random on the SAME TP1 validation schedule (S-SEL-v1, all 61 days) used for
# checkpoint selection, evaluated ONCE instead of at every checkpoint. Never the December split.
#
# Submit from the repository root:
#   sbatch --export=ALL,FEEDER=ieee33 "SLURM Script/validation_baselines.slurm"
# Required:  FEEDER (ieee33|ieee69), GMARL_CONDA_SH, GMARL_CONDA_ENV
# Optional:  RUN_NAME, GMARL_REPO (default submit dir)

set -euo pipefail

: "${FEEDER:?set FEEDER=ieee33|ieee69}"
: "${GMARL_CONDA_SH:?set GMARL_CONDA_SH=<conda.sh to source>}"
: "${GMARL_CONDA_ENV:?set GMARL_CONDA_ENV=<conda environment>}"
REPO_DIR="${GMARL_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
source "${GMARL_CONDA_SH}"
conda activate "${GMARL_CONDA_ENV}"
cd "${REPO_DIR}"
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"

case "${FEEDER}" in
  ieee33) SCRIPT=scripts/run_eval_ieee33.py ;;
  ieee69) SCRIPT=scripts/run_eval_ieee69.py ;;
  *) echo "unsupported FEEDER ${FEEDER}"; exit 2 ;;
esac

python -u "${SCRIPT}" \
    --baselines_only \
    --mode val \
    --run_purpose baseline_validation \
    --episodes schedule \
    --run_name "${RUN_NAME:-validation_baselines/${FEEDER}/${SLURM_JOB_ID:-local}}" \
    --cpu
