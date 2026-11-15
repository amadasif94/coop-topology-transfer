"""The single place where the evaluation pipeline builds a zero-action reference environment.

Bug in the original zero-action baselines: the policy requested 0 kW, but the directional network backup could
still move the batteries off zero whenever the 0 kW dispatch violated a network limit, so the "zero-action"
reference sometimes used the batteries. Here a zero policy ALWAYS runs in the pure zero-action environment of
experiments/load_stress/pure_zero_env(_ieee69): the original primary configuration with exactly two in-memory
changes (backup.backup_mode -> legacy_scaling, constraints.handling -> accept_unconstrained), which executes the
0 kW dispatch as-is. Power flow, loads, PV, curtailment, prices, violations and PF failures are measured
unchanged; a violating day is reported as violating, never corrected and never dropped. Learned controllers are
untouched and keep the full network backup. assert_zero_dispatch() checks every step: requested, SoC-feasible
(physically limited) and executed battery power exactly 0 kW, no backup activation, one control power flow.
"""
import json

import numpy as np

from experiments.load_stress.evaluate_pure_zero_ieee33 import check_steps


def make_zero_action_env(feeder, config, mode, seed, scale=1.0):
    """The original DESSEnv (never MarlDESSEnv): a policy action of 0 is executed as 0 kW. Under the affine soc_feasible
    mapping an action of 0 would mean the MIDPOINT of the available power interval, so the zero-action reference must
    never pass through that mapping; this is checked here and, per step, by assert_zero_dispatch."""
    if feeder == 'ieee69':
        from experiments.load_stress.pure_zero_env_ieee69 import make_pure_zero_env_ieee69
        env = make_pure_zero_env_ieee69(config, mode, seed, scale)
    else:
        from experiments.load_stress.pure_zero_env import make_pure_zero_env
        env = make_pure_zero_env(config, mode, seed, scale)
    from environments.marl_env import MarlDESSEnv
    if isinstance(env, MarlDESSEnv) or getattr(env, 'action_mapping', 'rated') != 'rated':
        raise RuntimeError('zero-action reference must execute 0 kW, not a SOC-feasible mapped action')
    return env


def assert_zero_dispatch(frame, num_agents):
    """Raise unless every dispatch step executed exactly 0 kW (requested, SoC-feasible and executed)."""
    result = check_steps(frame, num_agents)
    steps = frame[frame['record_type'] == 'dispatch_attempt'] if 'record_type' in frame else frame
    for column in ('physical_limited_power_kw', 'executed_battery_power_kw'):
        if column in steps:
            values = np.concatenate([np.asarray(json.loads(v) if isinstance(v, str) else v, dtype=float)
                                     for v in steps[column].dropna()]) if steps[column].notna().any() else np.zeros(0)
            if np.count_nonzero(values):
                raise AssertionError(f'Zero-action reference: {column} is not exactly 0 kW')
            result[f'{column}_nonzero'] = 0
    return result
