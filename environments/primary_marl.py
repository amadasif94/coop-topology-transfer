"""Explicit primary MARL protocol; library defaults remain regression-compatible.

Copies the supplied production configuration, changing only the seven approved
DOE/backup settings. Never repairs scaling or CSV ratings silently.
"""
import copy
import json
import numpy as np

DOE = dict(doe_enabled=False, doe_w=0.015, doe_delta_c=0.010,
           pf_failure_mode="logged_fallback")
BACKUP = dict(backup_mode="directional_incremental", pf_budget=20,
              refinement_steps=4, voltage_scale_pu=0.01, relative_current_scale=0.05,
              residual_penalty_enabled=False, residual_penalty_coefficient=0.0)


def primary_config(config):
    cfg = copy.deepcopy(config)
    cfg.setdefault("doe", {}).update(DOE)
    cfg.setdefault("backup", {}).update(BACKUP)
    for name in ("load_scaling_factor", "pv_scaling_factor"):
        if cfg["data_scaling"][name] != 1.0:
            raise ValueError(f"Primary MARL requires {name}=1.0; fix the source configuration")
    return cfg


def validate_primary_env(env, scenario="production"):
    checks = dict(doe_enabled=env.doe_enabled, doe_w=env.doe_w,
                  doe_delta_c=env.doe_delta_c, pf_failure_mode=env.doe_pf_failure_mode)
    def matches(actual, expected):
        if isinstance(expected, bool):
            return actual is expected
        if isinstance(expected, float):
            return actual is not None and bool(np.isclose(actual, expected, rtol=0, atol=1e-12))
        return actual == expected
    for key, expected in DOE.items():
        if not matches(checks[key], expected):
            raise ValueError(f"Primary MARL {key}: expected {expected!r}, got {checks[key]!r}")
    if env.backup_mode != 'directional_incremental':
        raise ValueError(f"Primary MARL backup_mode: got {env.backup_mode!r}")
    for key, expected in BACKUP.items():
        if not matches(env.backup_cfg.get(key), expected):
            raise ValueError(f"Primary MARL backup {key}: expected {expected!r}, got {env.backup_cfg.get(key)!r}")
    validate_network(env, scenario)
    effective = effective_current_ratings_a(env)
    print("PRIMARY_MARL " + json.dumps(dict(mode=env.mode, doe=checks,
          backup=env.backup_cfg, current_limit_min_a=float(effective.min()),
          current_limit_max_a=float(effective.max())), sort_keys=True), flush=True)


def effective_current_ratings_a(env):
    effective = np.array(env.grid.get_line_current_limits_pu(), dtype=float, copy=True)
    effective *= env.config["network"]["s_base_kva"] / (np.sqrt(3)*env.config["network"]["v_base_kv"])
    return effective


def validate_network(env, scenario="production"):
    if scenario not in ('production', 'synthetic', 'stress'):
        raise ValueError('Explicit scenario must be production, synthetic, or stress')
    for key in ('load_scaling_factor', 'pv_scaling_factor'):
        if scenario == 'production' and env.config['data_scaling'][key] != 1.0:
            raise ValueError(f'Production requires {key}=1.0')
    effective = effective_current_ratings_a(env)
    if not effective.size or not np.isfinite(effective).all() or np.any(effective <= 0):
        raise ValueError('Invalid effective active-branch current ratings')
    if scenario != 'production':
        return
    lines = env.grid.in_service_line_df
    ratings = np.asarray(lines["max_current_a"], dtype=float)
    if not ratings.size or not np.allclose(ratings, 255, rtol=0, atol=1e-6):
        raise ValueError("Primary MARL requires active CSV current ratings of 255 A")
    if effective.shape != ratings.shape or not np.allclose(effective, ratings, rtol=0, atol=1e-6):
        raise ValueError("Effective current ratings differ from the production CSV")


def make_primary_env(config, mode="train", seed=None):
    from environments.dess_env import DESSEnv
    env = DESSEnv(config=primary_config(config), mode=mode, seed=seed)
    validate_primary_env(env)
    return env
