"""Identity gate: the 1.0x load-stress rerun must reproduce the existing final evaluation step by step.

    python -u -m experiments.load_stress.validate_load_stress_identity --run-dir <output-root>/<label>/run_<r>/load_x100

Reference: step_metrics.csv next to the reference final manifest recorded by the evaluator (learned models),
or the existing 1.0x zero-action final run (zero). Alignment per topology on (start_index, step); all of
TP1-TP7, 31 episodes x 96 steps. Fields, exact/close classes and tolerances are those of the IEEE33
replay identity validator (validate_tp1_replay_identity_ieee33.py, imported unchanged). Timing columns are
never compared. Writes identity_report.json and identity_metrics.csv into the run directory (never overwrites).
"""
import argparse
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from experiments.load_stress.evaluate_load_stress_ieee33 import TOPOLOGIES
from experiments.no_shared_kpi.protocol import read_json, sha, write_json
from experiments.topology_generalization.replay_utils_ieee33 import EXPECTED_EPISODES, read_filtered
from experiments.topology_generalization.validate_tp1_replay_identity_ieee33 import (
    OPTIONAL_CLOSE, OPTIONAL_EXACT, OPTIONAL_VECTOR_CLOSE, REQUIRED_CLOSE, REQUIRED_EXACT, REQUIRED_VECTOR_CLOSE,
    REQUIRED_VECTOR_EXACT, compare, key_report)

GROUPS = [('exact', REQUIRED_EXACT, True), ('close', REQUIRED_CLOSE, True), ('vector_exact', REQUIRED_VECTOR_EXACT, True),
          ('vector_close', REQUIRED_VECTOR_CLOSE, True), ('exact', OPTIONAL_EXACT, False), ('close', OPTIONAL_CLOSE, False),
          ('vector_close', OPTIONAL_VECTOR_CLOSE, False)]
COLUMNS = [c for _, cs, _ in GROUPS for c in cs]


def reference_csv(stress):
    if stress['model'] == 'zero':
        ref = stress.get('zero_identity_reference')
        if not ref:
            raise SystemExit('No existing 1.0x zero-action final run is recorded; zero identity is not observable')
        return Path(ref['path']).parent / 'step_metrics.csv', ref
    return Path(stress['reference_manifest']['path']).parent / 'step_metrics.csv', stress['reference_manifest']


def compare_topology(tp, policy, ref_csv, new_csv, atol, rtol):
    src, _ = read_filtered(ref_csv, tp, policy, ['start_index', 'step', *COLUMNS])
    new, _ = read_filtered(new_csv, tp, policy, ['start_index', 'step', *COLUMNS])
    keys = dict(reference=key_report(src, f'reference {tp}'), rerun=key_report(new, f'1.0x rerun {tp}'))
    merged = src.merge(new, on=['start_index', 'step'], how='outer', suffixes=('__src', '__rep'), indicator=True)
    missing = dict(only_in_reference=int((merged['_merge'] == 'left_only').sum()),
                   only_in_rerun=int((merged['_merge'] == 'right_only').sum()))
    both = merged[merged['_merge'] == 'both']
    rows = []
    for kind, cs, required in GROUPS:
        for c in cs:
            if c not in src.columns or c not in new.columns:
                rows.append(dict(topology_case=tp, metric=c, kind=kind, required=required, in_reference=c in src.columns,
                                 in_rerun=c in new.columns, status='MISSING' if required else 'NOT OBSERVABLE'))
                continue
            n, bad, mx, mean = compare(both[f'{c}__src'], both[f'{c}__rep'], kind, atol, rtol)
            rows.append(dict(topology_case=tp, metric=c, kind=kind, required=required, in_reference=True, in_rerun=True,
                             matched_steps=n, n_exceed=bad, max_abs_err=mx, mean_abs_err=mean,
                             status='PASS' if bad == 0 and n > 0 else 'FAIL'))
    table = pd.DataFrame(rows)
    ok_keys = (not any(missing.values()) and all(not k['duplicate_keys'] and not k['incomplete_episodes']
               and k['episodes'] == EXPECTED_EPISODES for k in keys.values()))
    ok = ok_keys and len(both) > 0 and not table['status'].isin(['FAIL', 'MISSING']).any()
    return ('PASS' if ok else 'FAIL'), table, dict(keys=keys, missing=missing, matched_rows=int(len(both)))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--run-dir', required=True, type=Path)
    p.add_argument('--atol', type=float, default=1e-8)
    p.add_argument('--rtol', type=float, default=1e-8)
    a = p.parse_args(argv)
    manifest_path = a.run_dir / 'run_manifest.json'
    man = read_json(manifest_path)
    stress = man.get('load_stress') or {}
    if man.get('status') != 'finished' or stress.get('load_scaling_factor') != 1.0 or stress.get('smoke'):
        raise SystemExit('Identity validation needs a finished, non-smoke 1.0x load-stress run')
    ref_csv, ref_record = reference_csv(stress)
    if sha(ref_record['path']) != ref_record['sha256']:
        raise SystemExit('Reference manifest changed since the rerun was made')
    new_csv = a.run_dir / 'step_metrics.csv'
    tables, per_tp, status = [], {}, 'PASS'
    for tp in TOPOLOGIES:
        s, table, detail = compare_topology(tp, stress['policy'], ref_csv, new_csv, a.atol, a.rtol)
        tables.append(table)
        per_tp[tp] = dict(detail, status=s, failing_metrics=table.loc[table.status.isin(['FAIL', 'MISSING']), 'metric'].tolist())
        print(f'IDENTITY {tp}: {s}', flush=True)
        status = status if s == 'PASS' else 'FAIL'
    pd.concat(tables, ignore_index=True).to_csv(a.run_dir / 'identity_metrics.csv', index=False)
    report = dict(status=status, model=stress['model'], run=stress['run'], policy=stress['policy'],
        evaluation_manifest_sha256=sha(manifest_path), checkpoint_sha256=(man.get('checkpoint_file') or {}).get('sha256'),
        reference_manifest=ref_record, reference_step_metrics=dict(path=str(ref_csv.resolve()), sha256=sha(ref_csv)),
        rerun_step_metrics_sha256=sha(new_csv), atol=a.atol, rtol=a.rtol, topologies=per_tp,
        checked_utc=datetime.now(timezone.utc).isoformat())
    write_json(a.run_dir / 'identity_report.json', report)
    print('IDENTITY', status, a.run_dir / 'identity_report.json', flush=True)
    if status != 'PASS':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
