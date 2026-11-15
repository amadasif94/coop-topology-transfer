"""Fail-closed evaluation identities, independent of the training checkpoint code."""
import hashlib
import json
from pathlib import Path

CLASSES = ('original_maddpg', 'b3_post_tanh', 'b3_pre_tanh', 'soc_feasible')
INVARIANTS = ('feeder', 'interface_identity', 'normalization_identity', 'battery_constraints',
              'reward', 'doe', 'backup', 'network', 'voltage_tolerance_pu',
              'current_relative_tolerance', 'expected_steps_per_episode')
TOPOLOGY_INPUTS = {'line_data', 'gridtensor_line_data'}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def require_equal(a, b, label):
    if a != b:
        raise ValueError(f'Scientific provenance mismatch: {label}')


def formulation(checkpoint):
    schema = checkpoint.get('schema')
    if schema in ('b3-checkpoint/2', 'graph-marl-dess-checkpoint/1'):     # historical / maintained label schema
        if checkpoint.get('action_mapping', 'rated') == 'soc_feasible':
            # SOC-feasible action mapping, trained without the SoC-feasibility supervision loss.
            if checkpoint.get('feasibility_formulation') != 'none' or checkpoint.get('feasibility_supervision') is not False:
                raise ValueError('Contradictory soc_feasible checkpoint metadata')
            return 'soc_feasible'
        if checkpoint.get('feasibility_formulation') != 'pre_tanh_squared_hinge':
            raise ValueError('Unknown corrected B3 formulation')
        return 'b3_pre_tanh'
    if schema == 'b3-checkpoint/1':
        if checkpoint.get('feasibility_formulation') not in (None, 'post_tanh_action_space_squared_hinge'):
            raise ValueError('Contradictory historical B3 formulation')
        return 'b3_post_tanh'
    if schema is None or schema in ('ieee33-mlp-ablation/1', 'ieee69-mlp-ablation/1'):
        if checkpoint.get('feasibility_formulation') is not None or str(checkpoint.get('controller', '')).startswith('b3_'):
            raise ValueError('Ambiguous original checkpoint with B3 metadata')
        if 'actor' not in checkpoint or 'critics' not in checkpoint:
            raise ValueError('Unrecognized original checkpoint')
        return 'original_maddpg'
    raise ValueError(f'Unknown checkpoint schema {schema}')


def model_shapes(checkpoint):
    return {group: {k: list(v.shape) for k, v in checkpoint[group].items()} for group in ('actor', 'critics')}


def environment_contract(entry):
    for key in INVARIANTS + ('data_scaling', 'input_files', 'active_current_ratings_a'):
        if key not in entry or entry[key] is None:
            raise ValueError('Unverified environment metadata: '+key)
    files = {}
    for key, rec in entry['input_files'].items():
        if rec.get('exists'):
            if not rec.get('sha256'):
                raise ValueError('Unverified input hash: '+key)
            files[key] = rec['sha256']
        else:
            files[key] = None
    if not files.get('processed_time_series'):
        raise ValueError('Missing actual time-series fingerprint')
    result = {k: entry[k] for k in INVARIANTS}
    # Paths and normalization digest may vary across hosts; scales themselves may not.
    result['normalization_identity'] = {k:v for k,v in entry['normalization_identity'].items() if k != 'sha256'}
    result.update(data_scaling=entry['data_scaling'], input_hashes=files,
                  ratings=entry['active_current_ratings_a'], topology=entry.get('topology', 'TP1'))
    return result


def verify_environment(training, actual, scale=1.0, topology='TP1'):
    """Only approved load scaling and generated topology line files may differ.

    Caller must first validate current TP1 at scale 1 against historical TP1.
    Thus 'allow topology' cannot authorize an arbitrary replacement base network.
    """
    if scale not in (1., 1.1, 1.2, 1.3):
        raise ValueError('Unauthorized load multiplier')
    for key in INVARIANTS:
        require_equal(training[key], actual[key], key)
    expected = dict(training['data_scaling'], load_scaling_factor=scale)
    require_equal(expected, actual['data_scaling'], 'data scaling')
    require_equal(actual['topology'], topology, 'topology label')
    require_equal(sorted(training['ratings']), sorted(actual['ratings']), 'effective current ratings')
    require_equal(set(training['input_hashes']), set(actual['input_hashes']), 'input inventory')
    for key, value in training['input_hashes'].items():
        if topology != 'TP1' and key in TOPOLOGY_INPUTS:
            if not actual['input_hashes'][key]:
                raise ValueError('Missing topology snapshot: '+key)
            continue
        require_equal(value, actual['input_hashes'][key], 'input '+key)


def result_identity(experiment_class, controller, feeder, run, checkpoint, training_manifest,
                    shapes, model_config, coefficients, protocol, contracts, plans, selection=None):
    if experiment_class not in CLASSES:
        raise ValueError('Unknown experiment class')
    return dict(schema='matched-evaluation/1', experiment_class=experiment_class, controller=controller,
        feeder=feeder, run=int(run), seed=42+int(run), checkpoint_sha256=sha(checkpoint),
        training_manifest_sha256=sha(training_manifest), architecture=shapes, model_config=model_config,
        coefficients=coefficients, protocol=protocol, environments=contracts, episode_plans=plans,
        selection_sha256=sha(selection) if selection else None,
        reporting_sources={p.name:sha(p) for p in sorted(Path(__file__).parent.glob('*.py'))})


def verify_finished(directory, expected):
    directory = Path(directory)
    m = json.loads((directory/'run_manifest.json').read_text())
    if m.get('status') != 'finished':
        raise ValueError('Existing incomplete result; choose a new output directory')
    require_equal(m.get('scientific_identity'), expected, 'finished result identity (missing means unverified)')
    for name in ('step_metrics.csv', 'episode_summary.csv', 'aggregate_summary.csv', 'scientific_episodes.csv'):
        recorded = m.get('report_sha256', {}).get(name)
        if not recorded or not (directory/name).is_file() or sha(directory/name) != recorded:
            raise ValueError('Changed/missing result artifact: '+name)
    return m


def paired_identity(a, b):
    require_equal(a.get('experiment_class'), 'original_maddpg', 'baseline formulation')
    require_equal(b.get('experiment_class'), 'b3_pre_tanh', 'primary B3 formulation')
    for key in ('schema','controller','feeder','run','seed','architecture','model_config','environments','episode_plans','protocol'):
        if key not in a or key not in b or a[key] is None or b[key] is None:
            raise ValueError('Unverified paired metadata: '+key)
        if isinstance(a[key], dict) and (not a[key] or not b[key]):
            raise ValueError('Empty paired metadata: '+key)
        require_equal(a.get(key), b.get(key), 'paired '+key)
    if not a.get('selection_sha256') or not b.get('selection_sha256'):
        raise ValueError('Both policies require frozen validation selections')
