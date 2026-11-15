"""Load-stressed IEEE69 environments; the IEEE69 counterpart of stress_env.py.

Same rule as IEEE33: the ORIGINAL primary_config checks run on the unchanged 1.0 configuration, then only
data_scaling.load_scaling_factor is changed in an in-memory copy. The IEEE69 feeder/agent/interface checks
of experiments/no_shared_kpi/ieee69/runner.make_primary_env and its TP10 runtime evidence hook are applied
unchanged, and the A0 comparison is the IEEE69 one (experiments/no_shared_kpi/ieee69/protocol.py).

Kept outside experiments/no_shared_kpi/ieee69 and experiments/no_centralized_critic/ieee69: their *.py files
are part of the IEEE69 ablation source inventory embedded in every A1/A2 checkpoint.
"""
from copy import deepcopy

from environments.primary_marl import validate_primary_env
from experiments.load_stress.stress_env import PV_SCALE, check_scale, scenario, stress_config
from experiments.no_shared_kpi.ieee69 import protocol as p69


def make_stress_env_ieee69(config, mode, seed, scale):
    from environments.dess_env import DESSEnv
    from experiments.no_shared_kpi.ieee69.tp10_provenance import runtime_evidence
    env = DESSEnv(config=stress_config(config, scale), mode=mode, seed=seed)
    validate_primary_env(env, scenario(scale))
    # Same checks as the IEEE69 ablation make_primary_env.
    if env.num_buses != 69 or env.num_agents != 9 or list(env.dess_buses) != p69.DESS_BUSES:
        raise ValueError('Wrong IEEE69 feeder/agent ordering')
    if env.agent_obs_dim != 10 or env.episode_length != 96 or env.delta_t_hours != .25:
        raise ValueError('Wrong IEEE69 observation/time interface')
    if env.load_scaling_factor != check_scale(scale) or env.pv_scaling_factor != PV_SCALE:
        raise ValueError(f'Environment scaling load={env.load_scaling_factor} pv={env.pv_scaling_factor} '
                         f'differs from the requested load={scale} pv={PV_SCALE}')
    env.tp10_extension_equivalence = runtime_evidence(env)
    return env


def compare_stress_environment_ieee69(record, baseline, scale):
    """ORIGINAL IEEE69 A0 comparison, after checking that load_scaling_factor is the only scaling change."""
    scale = check_scale(scale)
    matches = [e for e in baseline['environments'] if e['topology'] == record['topology'] and e['mode'] == record['mode']]
    if len(matches) != 1:
        raise ValueError(f'Require one A0 environment for {record["topology"]}/{record["mode"]}')
    reference = matches[0]['data_scaling']
    expected = dict(deepcopy(reference), load_scaling_factor=scale)
    if record.get('data_scaling') != expected:
        raise ValueError(f'data_scaling {record.get("data_scaling")} differs from A0 {reference} '
                         f'by more than load_scaling_factor={scale}')
    p69.compare_environment(dict(deepcopy(record), data_scaling=deepcopy(reference)), baseline)
    return dict(a0_data_scaling=reference, stressed_data_scaling=expected)
