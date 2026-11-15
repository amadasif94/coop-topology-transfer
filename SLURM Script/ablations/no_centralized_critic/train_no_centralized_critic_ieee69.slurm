#!/bin/bash
#SBATCH --job-name=A2_train_ieee69
#SBATCH --partition=dgx
#SBATCH --qos=freetier
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --export=ALL
#SBATCH --output=/network/rit/lab/dahlin_lab/amad/graph_marl_dess/results/ablations/ieee69/slurm_logs/A2_train_%A_%a.out
#SBATCH --error=/network/rit/lab/dahlin_lab/amad/graph_marl_dess/results/ablations/ieee69/slurm_logs/A2_train_%A_%a.err

# Match current working IEEE33 pattern: resources/array supplied with sbatch.
# Positional arguments: STAGE JOB_MAP. No custom --export assignments.
set -euo pipefail
exec /bin/bash /network/rit/lab/dahlin_lab/amad/graph_marl_dess/experiments/no_shared_kpi/ieee69/launch.sh no_centralized_critic "$@"
