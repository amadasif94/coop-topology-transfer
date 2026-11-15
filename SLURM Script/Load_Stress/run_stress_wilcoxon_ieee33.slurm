#!/bin/bash
#SBATCH --job-name=stress_wilcoxon
#SBATCH --output=logs/stress_wilcoxon_%j.out
#SBATCH --error=logs/stress_wilcoxon_%j.err
#SBATCH --time=00:15:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=1

set -euo pipefail

cd /network/rit/lab/dahlin_lab/amad/graph_marl_dess

mkdir -p logs

export PYTHONPATH=.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

PYTHON=/network/rit/lab/dahlin_lab/amad/amad_thesis/env/thesis_conda/bin/python

echo "Host: $(hostname)"
echo "Python: $PYTHON"
echo "SLURM job ID: ${SLURM_JOB_ID:-unknown}"
echo "Memory information:"
free -h || true

echo "Testing NumPy and SciPy imports..."
"$PYTHON" -c "import numpy; print('NumPy:', numpy.__version__)"
"$PYTHON" -c "import scipy; print('SciPy:', scipy.__version__)"
"$PYTHON" -c "from scipy.stats import wilcoxon; print('Wilcoxon import successful')"

echo "Running stress-test statistical analysis..."
"$PYTHON" scripts/run_stress_wilcoxon_ieee33.py

echo "Completed successfully."
