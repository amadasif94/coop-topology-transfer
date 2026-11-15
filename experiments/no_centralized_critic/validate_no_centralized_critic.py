"""Run deterministic tests and optional real AC training/evaluation smoke."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.no_shared_kpi.validation import main
if __name__ == '__main__': main('no_centralized_critic')
