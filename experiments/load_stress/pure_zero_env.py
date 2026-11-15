"""Pure zero-action IEEE33 reference environment: every battery at exactly 0 kW, NO backup intervention.

Built from the load-stress configuration (ORIGINAL primary_config at 1.0, then load_scaling_factor), with exactly
two settings changed in an in-memory copy:
    backup.backup_mode   directional_incremental -> legacy_scaling
    constraints.handling hard_feasibility        -> accept_unconstrained
Any handling other than hard_feasibility makes DESSEnv execute the requested dispatch as-is, so a violating 0 kW
dispatch is executed and measured, never corrected. Power flow, ratings, normalization, prices, DOE (off), battery
physics and input files are unchanged.

Without directional backup DESSEnv would raise on a power-flow solver exception instead of recording it; this
instance sets _robust_pf=True so failures are recorded exactly as in production runs (the day ends, marked failed).
A reference only: not a controller, and its executed dispatch may violate limits.
"""
from copy import deepcopy

from experiments.load_stress.stress_env import check_scale, stress_config

BACKUP_MODE, HANDLING = 'legacy_scaling', 'accept_unconstrained'
CHANGED_KEYS = (('backup', 'backup_mode'), ('constraints', 'handling'))
LABEL = 'pure_zero_no_backup'
NOTE = ('Reference only: no controller and no backup. Every battery executes exactly 0 kW and the resulting dispatch '
        'is executed even when it violates voltage or current limits.')


def pure_zero_config(config, scale):
    cfg = deepcopy(stress_config(config, scale))
    cfg['backup']['backup_mode'] = BACKUP_MODE
    cfg.setdefault('constraints', {})['handling'] = HANDLING
    return cfg


def flatten(d, prefix=()):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict) and v:
            out.update(flatten(v, prefix + (k,)))
        else:
            out[prefix + (k,)] = v
    return out


def config_differences(a, b):
    fa, fb = flatten(a), flatten(b)
    return sorted(k for k in set(fa) | set(fb) if fa.get(k, '<missing>') != fb.get(k, '<missing>'))


def check_only_backup_disabled(pure_cfg, stress_cfg):
    diff = config_differences(pure_cfg, stress_cfg)
    allowed = sorted(CHANGED_KEYS)
    if [d for d in diff if d not in allowed]:
        raise ValueError(f'Pure zero config differs from the stress config beyond {allowed}: {diff}')
    return [list(d) for d in diff]


def make_pure_zero_env(config, mode, seed, scale):
    from environments.dess_env import DESSEnv
    from environments.primary_marl import validate_network
    scale = check_scale(scale)
    cfg = pure_zero_config(config, scale)
    differences = check_only_backup_disabled(cfg, stress_config(config, scale))
    env = DESSEnv(config=cfg, mode=mode, seed=seed)
    env._robust_pf = True
    validate_network(env, 'production' if scale == 1.0 else 'stress')
    if (env.backup_mode, env.constraint_handling, env.doe_enabled) != (BACKUP_MODE, HANDLING, False):
        raise ValueError(f'Pure zero env: backup={env.backup_mode} handling={env.constraint_handling} doe={env.doe_enabled}')
    if env.load_scaling_factor != scale or env.pv_scaling_factor != 1.0:
        raise ValueError(f'Pure zero env scaling load={env.load_scaling_factor} pv={env.pv_scaling_factor}')
    env.pure_zero_config_differences = differences
    return env
