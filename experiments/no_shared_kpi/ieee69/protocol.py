"""IEEE69 A0 compatibility checks. Fail closed rather than invent baseline records."""
from pathlib import Path
from experiments.no_shared_kpi.protocol import (
    ROOT, FULL_FEATURES, VARIANTS, TRAIN_SETTINGS, read_json, write_json, sha,
    feature_spec as shared_feature_spec, compare_environment as shared_compare,
    inventory_adjusted_cost, COST_DEFINITION)

DESS_BUSES = [13, 15, 17, 19, 21, 23, 25, 26, 64]
TOPOLOGIES = [f'TP{i}' for i in range(1, 8)] + ['TP10']


def feature_spec(variant, agents=9):
    if agents != 9 or variant not in VARIANTS:
        raise ValueError('IEEE69 requires nine agents and a known ablation')
    return shared_feature_spec(variant, agents)


def result_root(variant, run):
    if variant not in VARIANTS or run not in (0, 1, 2):
        raise ValueError('Unknown ablation/run')
    return ROOT/'results/ablations/ieee69'/variant/f'run_{run}'


def load_baseline(path, run, training=True):
    m = read_json(path)
    if m.get('status') != 'finished' or m.get('controller', '').upper() != 'MLP':
        raise ValueError('Require a real finished A0 MLP manifest')
    if training:
        if m.get('mode') != 'train' or m.get('seed') != 42 + run:
            raise ValueError('IEEE69 A0 training seed/split mismatch')
        for key, expected in TRAIN_SETTINGS.items():
            if m.get('arguments', {}).get(key) != expected:
                raise ValueError(f'IEEE69 A0 training setting differs: {key}')
    else:
        p = m.get('evaluation_protocol', {})
        from experiments.topology_generalization.tp_configs_ieee69 import list_topology_cases
        cases=p.get('topologies',[])
        if p.get('data_split') != 'test' or p.get('final_test') is not True or not cases or len(set(cases))!=len(cases) or not set(cases)<=set(list_topology_cases()):
            raise ValueError('Require an authorized A0 IEEE69 final manifest with supported topology cases')
        if f'run_{run}_' not in (p.get('selection') or {}).get('selection_id', ''):
            raise ValueError('Wrong A0 final evaluation run')
        episodes = m.get('arguments', {}).get('episodes')
        # 'all' is historical metadata, not proof of episode completeness.
        # Final references() must verify all 31 x 96 December rows per topology.
        if not ((type(episodes) is int and episodes == 31) or
                (type(episodes) is str and episodes in ('31', 'all'))):
            raise ValueError('Require verified A0 31-day final evaluation')
    envs = m.get('environments', [])
    if not envs or any(e.get('feeder') != 'IEEE69' for e in envs):
        raise ValueError('IEEE69 environment records missing')
    if training and {e['mode'] for e in envs} != {'train', 'val'}:
        raise ValueError('Require training and periodic-validation environment records')
    if not training and {e['topology'] for e in envs} != set(m['evaluation_protocol']['topologies']):
        raise ValueError('Missing A0 final topology records')
    for e in envs:
        if e.get('agent_obs_features') != list(FULL_FEATURES):
            raise ValueError('Expected the verified ten-feature actor schema')
        if e.get('interface_identity', {}).get('dess_buses') != DESS_BUSES:
            raise ValueError('IEEE69 battery placement/order mismatch')
        if len(e.get('battery_constraints', [])) != 9:
            raise ValueError('Expected nine battery configurations')
        if e.get('doe', {}).get('enabled') is not False or e.get('backup_mode') != 'directional_incremental':
            raise ValueError('Verified IEEE69 A0 uses DOE OFF and directional backup')
        if e.get('expected_steps_per_episode') != 96:
            raise ValueError('Expected 96-step episodes')
        if training and e.get('topology') != 'TP1':
            raise ValueError('Training and validation must use TP1')
        if not training and e.get('mode') != 'test':
            raise ValueError('Final evaluation must use test split')
    return m


def compare_environment(current, baseline):
    if current.get('feeder') != 'IEEE69' or current.get('interface_identity', {}).get('dess_buses') != DESS_BUSES:
        raise ValueError('Wrong instantiated IEEE69 environment')
    from copy import deepcopy
    from .tp10_provenance import APPROVED, recover, digest
    candidates=[e for e in baseline['environments'] if e['topology']==current['topology'] and e['mode']==current['mode']]
    if len(candidates)!=1: raise ValueError('Require one exact A0 topology/split reference')
    reference=candidates[0]
    checked=deepcopy(current)
    for key,record in reference['input_files'].items():
        now=current['input_files'].get(key,{})
        if not record.get('exists') or now.get('sha256')==record.get('sha256'): continue
        if key not in APPROVED or current['topology']!='TP1' or current['mode'] not in ('train','val'):
            raise ValueError('Unapproved A0 input difference: '+key)
        proof=current.get('tp10_extension_equivalence',{})
        if not proof.get('runtime_verified') or proof.get('mode')!=current['mode']:
            raise ValueError('TP10 extension requires a passing paired runtime equivalence proof')
        raw=Path(now['path']).read_bytes()
        recover(raw,key,record['sha256'])
        if digest(raw)!=now['sha256'] or proof.get('files',{}).get(key,{}).get('current_sha256')!=now['sha256']:
            raise ValueError('TP10 proof does not describe loaded input: '+key)
        # Private comparison view ONLY. Actual manifest and historical hashes stay intact.
        checked['input_files'][key]['sha256']=record['sha256']
    return shared_compare(checked, baseline)


def source_identity():
    # No top-level additions: IEEE33's existing glob-based identity stays stable.
    from experiments.no_shared_kpi.protocol import source_identity as original_identity
    files = original_identity()
    for variant in VARIANTS:
        for p in sorted((ROOT/'experiments'/variant/'ieee69').glob('*.py')):
            files[p.relative_to(ROOT).as_posix()] = sha(p)
    for rel in ('configs/ieee69_config.py',
                'experiments/no_gnn_maddpg/run_tp_eval_mlp_ieee69.py',
                'experiments/no_gnn_maddpg/run_train_mlp_ieee69.py',
                'experiments/topology_generalization/topology_utils_ieee69.py',
                'experiments/topology_generalization/tp_configs_ieee69.py'):
        files[rel] = sha(ROOT/rel)
    return files


def portable(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def verify_map(path, run):
    m = read_json(path)
    if m.get('feeder') != 'IEEE69':
        raise ValueError('Refusing a non-IEEE69 job map')
    entry = m['runs'][str(run)]
    for key in ('baseline_training_manifest', 'baseline_evaluation_manifest'):
        if entry.get(key) and sha(entry[key]) != entry[key+'_sha256']:
            raise ValueError('Job-map baseline changed: '+key)
    load_baseline(entry['baseline_training_manifest'], run)
    for record in entry.get('baseline_evaluation_manifests',[]):
        if sha(record['path'])!=record['sha256']: raise ValueError('A0 final manifest changed')
        load_baseline(record['path'],run,training=False)
    return entry
