"""no_shared_kpi: reuse the controlled ablation runner."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.no_shared_kpi.runner import main
if __name__ == '__main__': main('no_shared_kpi', 'train')
