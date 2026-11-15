"""Shared run plumbing of the non-learned reference controllers F-MPC and PQ-RBC. Maintained.

The environment is built exactly as for PF-MPC and the learned controllers (topology_config(..., experiment=<protocol>),
make_marl_env(..., mode, 42, 0.0, scale)): SoC0 = 0.50 at every day start, physical battery limiter, directional AC
network backup, the protocol's terminal valuation and the same evaluators and table writers. Splits:
  test  the 31 December days, every topology (final-test authorization; learned=False, so no checkpoint selection);
  val   validation days, TP1 only (run_purpose 'development'); used to check a controller before any December run.
Output folders (never overwritten): <root>/<feeder>/<label>/load_x<pct>[_smoke<N>] (test) and
<root>/<feeder>/<label>_validation/load_x<pct>_<days> (val), so validation runs are never summarised with test runs.
"""
from pathlib import Path

from configs.experiments import TOPOLOGIES, feeder_config, sha, topology_config
from environments.marl_env import make_marl_env
from experiments.load_stress.stress_env import scale_label
from utility.evaluation_protocol import authorize, episode_plan, protocol_config

SEED = 42                     # as PF-MPC and the zero reference: identical days and reset seeds
SPLITS = ('val', 'test')
TEST_DAYS = 31


def split_protocol(split, topologies):
    if split == 'test':
        return authorize(mode='test', final=True, run_purpose='final_test', learned=False, topologies=topologies)
    if split == 'val':
        return authorize(mode='val', final=False, run_purpose='development', learned=False, topologies=topologies)
    raise ValueError(f'unknown split {split!r}')


def resolve_topologies(feeder, split, requested):
    tps = [t.upper() for t in (requested or (TOPOLOGIES[feeder] if split == 'test' else ['TP1']))]
    unknown = sorted(set(tps) - set(TOPOLOGIES[feeder]))
    if unknown:
        raise ValueError(f'unknown topologies {unknown}')
    if split == 'val' and tps != ['TP1']:
        raise ValueError('validation-day runs use TP1 only (held-out topologies are reserved for the December test)')
    return tps


def reference_env(feeder, topology, out, experiment, protocol, scale, split, time_series=None):
    cfg = protocol_config(topology_config(feeder, topology, Path(out) / 'network_inputs', time_series=time_series,
                                          experiment=experiment), protocol)
    return make_marl_env(cfg, split, SEED, 0.0, scale)


def day_plan(env, protocol, split, max_days=None, days=None):
    """(start, seed) pairs: all days of the split, the first max_days, or the listed day positions."""
    starts, seeds, _ = episode_plan(env, 'all', SEED, protocol)
    if split == 'test' and len(starts) != TEST_DAYS:
        raise SystemExit(f'Expected {TEST_DAYS} December test days, found {len(starts)}')
    plan = list(zip(starts, seeds))
    if days:
        bad = [d for d in days if not 0 <= d < len(plan)]
        if bad:
            raise ValueError(f'--days positions {bad} outside 0..{len(plan) - 1}')
        return [plan[d] for d in days]
    return plan[:max_days or None]


def output_dir(root, feeder, label, split, scale, max_days=None, days=None):
    if split == 'test':
        if days:
            raise ValueError('--days is for validation-day runs; December runs use all days or --max-days')
        return Path(root) / feeder / label / (scale_label(scale) + (f'_smoke{max_days}' if max_days else ''))
    tag = f'days{"-".join(str(d) for d in days)}' if days else (f'first{max_days}' if max_days else 'all')
    return Path(root) / feeder / f'{label}_validation' / f'{scale_label(scale)}_{tag}'


def env_series_sha256(env):
    return sha(env.config['paths']['processed_time_series'])


def check_reference_env(env, scale, feeder):
    """Nominal PV, prices and the stress factor on load only (the environment's _get_profiles convention), and the
    configured series: the original per-day energy accounting (experiments.no_shared_kpi energy_rows, used by
    write_tables) reads prices from the feeder configuration's processed_time_series, so the environment must use it."""
    configured = Path(feeder_config(feeder)['paths']['processed_time_series']).resolve()
    if Path(env.config['paths']['processed_time_series']).resolve() != configured:
        raise ValueError(f'environment series {env.config["paths"]["processed_time_series"]} is not the configured '
                         f'{configured} used by the energy accounting')
    if (env.load_scaling_factor, env.pv_scaling_factor, env.price_scaling_factor) != (scale, 1.0, 1.0):
        raise ValueError(f'environment scaling load={env.load_scaling_factor} pv={env.pv_scaling_factor} '
                         f'price={env.price_scaling_factor}; expected load={scale}, pv=1.0, price=1.0')
    if env.doe_enabled:
        raise ValueError('reference controllers run with DOE off (maintained environments)')
