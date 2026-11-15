"""HISTORICAL COMPATIBILITY SHIM for the B3/B4 entry points (experiments/b3) and their tests.

The maintained implementation is configs.experiments; maintained modules never import this module. Kept here:
the historical names, the historical run layout (<root>/<feeder>/b3_<controller>/run_<k>), the historical default output
root (results/<experiment id>) and the historical B3 source inventory (b3_sources), so existing commands, manifests
and results stay reproducible.
"""
from pathlib import Path

from configs.experiments import *                                     # noqa: F401,F403 (historical re-export)
from configs.experiments import ROOT, sha

# Every B3 library module outside experiments/b3, as recorded in historical B3/B4 manifests.
B3_LIBRARY_MODULES = ('environments/battery_feasibility.py', 'environments/feasibility_env.py',
                      'training/bounded_replay.py', 'training/feasibility_supervision.py',
                      'training/feasibility_trainer.py', 'models/feasibility_agents.py',
                      'utility/feasibility_metrics.py', 'utility/experiment_setup.py',
                      'environments/local_reward_env.py', 'models/independent_local_agents.py')


def b3_sources():
    """sha256 of every B3 source file (library modules + experiments/b3/*.py), recorded in every B3 manifest."""
    paths = [ROOT / p for p in B3_LIBRARY_MODULES] + sorted((ROOT / 'experiments' / 'b3').glob('*.py'))
    return {p.relative_to(ROOT).as_posix(): sha(p) for p in paths}


def run_dir(output_root, feeder, label, run):
    """Historical layout <root>/<feeder>/<label>/run_<run>."""
    return Path(output_root) / feeder / label / f'run_{run}'


def output_root_for(experiment, output_root=None):
    """Historical default results/<experiment>; a non-b3 experiment is never written into a results/b3 tree."""
    root = Path(output_root or f'results/{experiment}')
    if experiment != 'b3' and 'b3' in [part.lower() for part in root.resolve().parts]:
        raise ValueError(f'experiment {experiment} must not write into a b3 results tree: {root}')
    return str(root)
