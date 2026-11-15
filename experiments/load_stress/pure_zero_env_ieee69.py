"""Pure zero-action IEEE69 reference environment: all nine batteries at exactly 0 kW, NO backup intervention.

The IEEE69 counterpart of pure_zero_env.py. The configuration is built the same way (ORIGINAL primary_config at
1.0, which also forces DOE off: the raw IEEE69 config has doe_enabled=True; then load_scaling_factor; then
exactly backup.backup_mode -> legacy_scaling and constraints.handling -> accept_unconstrained in a copy), and the
IEEE69 feeder/agent/interface checks of make_stress_env_ieee69 are applied unchanged (69 buses, nine agents at
DESS_BUSES, 10 observation features, 96 x 15-minute steps).

The TP10 runtime-evidence hook is not called: it applies only to TP1 train/validation environments (it returns
None for every test environment) and would rebuild a backup-enabled primary environment. The December
evaluation is test mode; the backup-enabled control environment built beside it runs the hook as usual.
"""
from experiments.load_stress.pure_zero_env import make_pure_zero_env
from experiments.no_shared_kpi.ieee69 import protocol as p69


def make_pure_zero_env_ieee69(config, mode, seed, scale):
    env = make_pure_zero_env(config, mode, seed, scale)
    if env.num_buses != 69 or env.num_agents != 9 or list(env.dess_buses) != p69.DESS_BUSES:
        raise ValueError('Wrong IEEE69 feeder/agent ordering')
    if env.agent_obs_dim != 10 or env.episode_length != 96 or env.delta_t_hours != .25:
        raise ValueError('Wrong IEEE69 observation/time interface')
    env.tp10_extension_equivalence = None
    return env
