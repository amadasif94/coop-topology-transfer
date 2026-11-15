#!/bin/bash
#SBATCH --job-name=stress_plot
#SBATCH --output=logs/stress_plot_%j.out
#SBATCH --error=logs/stress_plot_%j.err
#SBATCH --time=00:15:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=1

set -euo pipefail

cd /network/rit/lab/dahlin_lab/amad/graph_marl_dess

mkdir -p logs

export PYTHONPATH=.
export MPLBACKEND=Agg
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

PYTHON=/network/rit/lab/dahlin_lab/amad/amad_thesis/env/thesis_conda/bin/python

echo "Host: $(hostname)"
echo "Python: $PYTHON"
echo "SLURM job ID: ${SLURM_JOB_ID:-unknown}"
free -h || true

echo "Testing plotting imports..."
"$PYTHON" -c "import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt; import fontTools; print('Matplotlib and fontTools import successful')"

echo "Generating stress-test figure and table..."
"$PYTHON" scripts/plot_stress_divergence_final.py

echo "Completed successfully."
