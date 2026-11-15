"""Load-stressed IEEE33 environments for frozen-policy sensitivity evaluation.

The only deviation from the A0 final-test environment is data_scaling.load_scaling_factor.
The source configuration (scale 1.0) first passes the ORIGINAL primary_config checks; the
scale is applied to an in-memory deep copy. Nothing on disk is changed.

This folder must stay outside experiments/no_shared_kpi and experiments/no_centralized_critic:
their *.py files form the ablation source inventory embedded in every A1/A2 checkpoint.
"""
from copy import deepcopy

from environments.primary_marl import primary_config, validate_primary_env
from experiments.no_shared_kpi.protocol import compare_environment

ALLOWED_SCALES = (1.0, 1.1, 1.2, 1.3)
PV_SCALE = 1.0


def check_scale(scale):
    scale = float(scale)
    if scale not in ALLOWED_SCALES:
        raise ValueError(f'Load scale {scale} not in approved set {ALLOWED_SCALES}')
    return scale


def scenario(scale):
    return 'production' if check_scale(scale) == 1.0 else 'stress'


def scale_label(scale):
    return f'load_x{int(round(check_scale(scale) * 100))}'


def stress_config(config, scale):
    """ORIGINAL primary_config at 1.0, then only load_scaling_factor changed in a copy."""
    scale = check_scale(scale)
    if config['data_scaling']['load_scaling_factor'] != 1.0 or config['data_scaling']['pv_scaling_factor'] != 1.0:
        raise ValueError('Source configuration must keep load and PV scaling at 1.0')
    cfg = deepcopy(primary_config(config))
    cfg['data_scaling']['load_scaling_factor'] = scale
    return cfg


def make_stress_env(config, mode, seed, scale):
    from environments.dess_env import DESSEnv
    env = DESSEnv(config=stress_config(config, scale), mode=mode, seed=seed)
    validate_primary_env(env, scenario(scale))
    if env.load_scaling_factor != check_scale(scale) or env.pv_scaling_factor != PV_SCALE:
        raise ValueError(f'Environment scaling load={env.load_scaling_factor} pv={env.pv_scaling_factor} '
                         f'differs from the requested load={scale} pv={PV_SCALE}')
    return env


def compare_stress_environment(record, baseline, scale):
    """ORIGINAL A0 comparison, after checking that load_scaling_factor is the only scaling change."""
    scale = check_scale(scale)
    matches = [e for e in baseline['environments'] if e['topology'] == record['topology'] and e['mode'] == record['mode']]
    if not matches:
        raise ValueError(f'No A0 environment for {record["topology"]}/{record["mode"]}')
    reference = matches[0]['data_scaling']
    expected = dict(deepcopy(reference), load_scaling_factor=scale)
    if record.get('data_scaling') != expected:
        raise ValueError(f'data_scaling {record.get("data_scaling")} differs from A0 {reference} '
                         f'by more than load_scaling_factor={scale}')
    compare_environment(dict(deepcopy(record), data_scaling=deepcopy(reference)), baseline)
    return dict(a0_data_scaling=reference, stressed_data_scaling=expected)
