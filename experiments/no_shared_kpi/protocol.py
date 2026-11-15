"""Fail-closed A0 matching and file identity, independent of PyTorch."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LOCAL_FEATURES = ('time', 'price', 'soc', 'voltage_pu', 'load_kw', 'pv_kw', 'previous_action')
FULL_FEATURES = LOCAL_FEATURES + ('feeder_grid_import_kw', 'feeder_voltage_deviation', 'feeder_curtailment_kw')
# Evidence: original SLURM file. Production values are still read/checked against each A0 manifest.
TRAIN_SETTINGS = dict(total_steps=200000, warmup_steps=10000, buffer_size=100000,
    batch_size=128, actor_lr=5e-5, critic_lr=5e-4, gamma=.99, tau=.005,
    noise_std=.05, eval_every=5000, save_every=5000, eval_episodes=10)
VARIANTS = ('no_shared_kpi', 'no_centralized_critic')
COST_DEFINITION = 'cost - day_mean_price_per_MWh * (terminal_SOC_kWh - initial_SOC_kWh) / 1000'


def inventory_adjusted_cost(raw_cost, initial_kwh, terminal_kwh, mean_price_per_mwh):
    """Historical stored-side convention; physical efficiencies already affect SOC."""
    return raw_cost - mean_price_per_mwh * (terminal_kwh-initial_kwh)/1000.0


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_json(path, data):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(data, stream, indent=2, allow_nan=False, default=str)


def result_root(variant, run):
    if variant not in VARIANTS or run not in (0, 1, 2):
        raise ValueError('Unknown ablation or run')
    return ROOT/'results/ablations/ieee33'/variant/f'run_{run}'


def load_baseline(path, run, training=True):
    m = read_json(path)
    if m.get('status') != 'finished' or m.get('controller', '').upper() != 'MLP':
        raise ValueError('A0 must be a finished MLP run manifest')
    if training:
        if m.get('mode') != 'train' or int(m.get('seed', -1)) != 42 + run:
            raise ValueError('A0 training split/seed differs from original run-specific seed 42+run')
        args = m.get('arguments', {})
        for key, expected in TRAIN_SETTINGS.items():
            if args.get(key) != expected:
                raise ValueError(f'A0 {key}: expected observed SLURM value {expected}, got {args.get(key)}')
    else:
        p = m.get('evaluation_protocol', {})
        if p.get('data_split') != 'test' or p.get('final_test') is not True:
            raise ValueError('A0 evaluation must be an authorized December final test')
        if p.get('topologies') != [f'TP{i}' for i in range(1, 8)]:
            raise ValueError('A0 final manifest must cover TP1-TP7 in order')
        if f'run_{run}_' not in (p.get('selection') or {}).get('selection_id', ''):
            raise ValueError('A0 final selection does not identify the requested run')
    envs = m.get('environments', [])
    if not envs or any(e.get('feeder') != 'IEEE33' for e in envs):
        raise ValueError('A0 IEEE33 environments missing')
    for e in envs:
        if e.get('agent_obs_features') != list(FULL_FEATURES):
            raise ValueError('A0 observation schema differs: current source has 10 features, not legacy 11')
        if e.get('doe', {}).get('enabled') is not False or e.get('backup_mode') != 'directional_incremental':
            raise ValueError('A0 must use DOE OFF and directional backup, as this controlled protocol requires')
        if training and e.get('topology') != 'TP1':
            raise ValueError('A0 training/periodic validation must use TP1')
        if not training and (e.get('mode') != 'test' or int(m.get('arguments', {}).get('episodes', -1)) != 31 or e.get('expected_steps_per_episode') != 96):
            raise ValueError('A0 final evaluation must plan 31 test episodes of 96 steps')
    return m


def compare_environment(current, baseline):
    """Compare instantiated env manifest to SAME topology/split A0; paths may move, bytes may not."""
    candidates = [e for e in baseline['environments'] if e['topology'] == current['topology']
                  and e['mode'] == current['mode']]
    if not candidates:
        raise ValueError(f'No A0 environment for {current["topology"]}/{current["mode"]}')
    b = candidates[0]
    keys = ('agent_obs_features', 'interface_identity', 'normalization_identity', 'data_scaling',
            'doe', 'backup', 'network', 'reward', 'battery_constraints', 'active_current_ratings_a',
            'voltage_tolerance_pu', 'current_relative_tolerance', 'expected_steps_per_episode',
            'available_episode_start_indices')
    for key in keys:
        if key not in b or current.get(key) != b[key]:
            raise ValueError(f'A0 environment mismatch: {key}')
    for key, rec in b['input_files'].items():
        now = current['input_files'].get(key, {})
        if rec.get('exists') and (not now.get('exists') or rec.get('sha256') != now.get('sha256')):
            raise ValueError(f'A0 input bytes differ: {key}')


def feature_spec(variant, agents=5):
    return dict(variant=variant, full_features=list(FULL_FEATURES),
        actor_features=list(LOCAL_FEATURES if variant == VARIANTS[0] else FULL_FEATURES),
        actor_input_dim=7 if variant == VARIANTS[0] else 10,
        critic_input_dim=agents*11 if variant == VARIANTS[0] else 11,
        critic_scope='joint_full_observations_and_actions' if variant == VARIANTS[0] else 'own_observation_and_action',
        num_agents=agents, action_dim=1, actor_hidden_dims=[256,256], critic_hidden_dims=[256,256],
        activation='ReLU', output_activation='tanh', shared_reward='unchanged_DESSEnv')


def source_identity():
    files = {}
    for folder in VARIANTS:
        for p in sorted((ROOT/'experiments'/folder).glob('*.py')):
            files[p.relative_to(ROOT).as_posix()] = sha(p)
    for relative in ('experiments/no_gnn_maddpg/maddpg_mlp.py', 'experiments/no_gnn_maddpg/mlp_actor.py',
        'experiments/no_gnn_maddpg/mlp_critic.py', 'experiments/no_gnn_maddpg/trainer_mlp.py',
        'experiments/no_gnn_maddpg/run_tp_eval_mlp.py', 'scripts/select_checkpoint.py',
        'experiments/topology_generalization/compare_tp6_replay_vs_feedback_ieee33.py'):
        files[relative] = sha(ROOT/relative)
    return files
