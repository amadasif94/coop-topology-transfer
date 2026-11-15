#!/bin/bash
#SBATCH --job-name=agg_tp6_stress
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=results/logs/slurm/aggregate_tp6_stress_%j.out
#SBATCH --error=results/logs/slurm/aggregate_tp6_stress_%j.err

set -euo pipefail

BASE_DIR="/network/rit/lab/dahlin_lab/amad/graph_marl_dess"

source /network/rit/lab/dahlin_lab/anaconda3/etc/profile.d/conda.sh
conda activate /network/rit/lab/dahlin_lab/amad/amad_thesis/env/thesis_conda

cd "${BASE_DIR}"

export PYTHONPATH="${BASE_DIR}"

python scripts/aggregate_stress_ieee33_tp6.py