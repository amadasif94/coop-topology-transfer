"""Separate IEEE69 map, discovered by manifest content; no guessed checkpoint paths."""
import argparse
from pathlib import Path
from .protocol import ROOT, VARIANTS, read_json, write_json, sha, portable, load_baseline, result_root


def discover(root, training):
    found = {}
    for path in Path(root).rglob('run_manifest.json'):
        m = read_json(path)
        if m.get('controller', '').upper() != 'MLP' or m.get('status') != 'finished': continue
        if not m.get('environments') or any(e.get('feeder') != 'IEEE69' for e in m['environments']): continue
        for run in range(3):
            try: load_baseline(path, run, training)
            except (ValueError, KeyError, TypeError): continue
            if training:
                if run in found: raise ValueError(f'Ambiguous A0 training manifests for run {run}: {found[run]}, {path}')
                found[run] = path
            else:
                found.setdefault(run,[]).append(path)
    return found


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--a0-training-root', type=Path, required=True)
    parser.add_argument('--a0-evaluation-root', type=Path)
    parser.add_argument('--discover-ablation-selections', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    a = parser.parse_args()
    if not any(a.output.resolve().is_relative_to((ROOT/'experiments'/v/'ieee69').resolve()) for v in VARIANTS):
        parser.error('Write IEEE69 maps only inside the nested ieee69 experiment packages')
    training = discover(a.a0_training_root, True)
    if set(training) != {0,1,2}: raise ValueError(f'Require three verified IEEE69 training manifests; found {training}')
    final = discover(a.a0_evaluation_root, False) if a.a0_evaluation_root else {}
    if a.a0_evaluation_root and set(final) != {0,1,2}:
        raise ValueError('Requested A0 final manifest discovery did not find all three IEEE69 runs')
    result = dict(feeder='IEEE69', runs={})
    for run in range(3):
        entry = dict(baseline_training_manifest=portable(training[run]),
                     baseline_training_manifest_sha256=sha(training[run]), selections={})
        if run in final:
            from .final_evaluation import references
            references(final[run],run)  # verify coverage, schedules, snapshots before binding
            entry['baseline_evaluation_manifests']=[dict(path=portable(x),sha256=sha(x)) for x in final[run]]
        if a.discover_ablation_selections:
            from utility.evaluation_protocol import verify_selection
            for v in VARIANTS:
                paths = list((result_root(v,run)/'selections').rglob('selection.json'))
                if len(paths) != 1: raise ValueError(f'{v}/{run}: expected exactly one frozen selection; found {len(paths)}')
                rec = read_json(paths[0])
                if rec.get('feeder') != 'ieee69' or rec.get('model') != v or rec.get('run') != f'run_{run}':
                    raise ValueError('Wrong selection identity')
                verify_selection(paths[0])
                entry['selections'][v] = portable(paths[0])
        result['runs'][str(run)] = entry
    write_json(a.output, result)
    print('Verified IEEE69 A0 manifests. Wrote:', a.output)
    print('Final baseline runs:', len(final), '; final evaluation requires all baselines and bound ablation selections.')


if __name__ == '__main__': main()
