"""Shared helpers for the maintained tests (no dependency on historical modules or historical test files)."""
import importlib.util
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
HAS_PYG = importlib.util.find_spec('torch_geometric') is not None
SERIES = dict(ieee33=['data/time_series/processed_15min_smartds.csv',
                      'audit_tmp/replica/data/time_series/processed_15min_smartds.csv'],
              ieee69=['data/time_series/processed_15min_smartds_ieee69.csv',
                      'audit_tmp/replica/data/time_series/processed_15min_smartds_ieee69.csv'])
EDGE = torch.tensor([[i for i in range(11)] + [i + 1 for i in range(11)], [i + 1 for i in range(11)] + [i for i in range(11)]])
CONTROLLER_SPECS = ([('mlp', v) for v in ('A0', 'A1', 'A2', 'A3')] +
                    ([('gcn', None), ('gat', None), ('tagconv', None)] if HAS_PYG else []))


def series(feeder, root=ROOT):
    """Processed 15-minute time series of `feeder` (data/time_series on RIT, or the local audit_tmp/replica copy)."""
    return next((str(Path(root) / p) for p in SERIES[feeder] if (Path(root) / p).is_file()), None)


def batch(n, B=16, lower=-1.0, upper=1.0, seed=0):
    g = torch.Generator().manual_seed(seed)
    lo = lower if torch.is_tensor(lower) else torch.full((B, n), float(lower))
    hi = upper if torch.is_tensor(upper) else torch.full((B, n), float(upper))
    return dict(agent_obs=torch.rand(B, n, 10, generator=g), next_agent_obs=torch.rand(B, n, 10, generator=g),
                x=torch.rand(B, 12, 4, generator=g), next_x=torch.rand(B, 12, 4, generator=g), edge_index=EDGE,
                actions=torch.rand(B, n, 1, generator=g) * 2 - 1, rewards=torch.randn(B, n, generator=g),
                dones=torch.zeros(B, 1), feasible_action_lower=lo, feasible_action_upper=hi)
