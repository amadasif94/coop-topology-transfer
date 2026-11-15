"""OPTIONAL IEEE69 cross-check: 1.0x pure zero-action run vs existing zero-action + backup rows in the A0 finals.

    python -u -m experiments.load_stress.validate_pure_zero_ieee69 \
        --pure-dir results/load_stress/ieee69/pure_zero_no_backup/load_x100

Reference per topology: the step_metrics.csv beside that topology's A0 group manifest (recorded in the pure run's
manifest), rows with policy == 'zero'. They exist only if the A0 final ran with --include_zero. In those rows the
first (pre-backup) power flow of every step is the 0 kW dispatch, so backup_initial_* / doe_candidate_* must equal
the pure run's on every matched (start_index, step). Status per topology: PASS, FAIL, or NOT_OBSERVABLE (no zero
rows, or the reference predates those columns). Overall: PASS, PASS_PARTIAL (some NOT_OBSERVABLE), FAIL (exit 1),
NOT_AVAILABLE (nothing observable). Stressed scales have no IEEE69 zero + backup reference; the --smoke run of the
evaluator performs the same comparison against a fresh backup-enabled control. Never overwrites; read only otherwise.
"""
import argparse
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from experiments.load_stress.evaluate_pure_zero_ieee69 import TOPOLOGIES
from experiments.load_stress.validate_pure_zero_ieee33 import FLAGS, KEYS, NUMERIC, compare, read_steps
from experiments.no_shared_kpi.protocol import read_json, sha, write_json


def reference_rows(path, tp):
    df = pd.read_csv(path, usecols=lambda c: c in {*KEYS, *NUMERIC, *FLAGS, 'policy', 'record_type', 'backup_activated',
                     'feasible'}, low_memory=False)
    df = df[(df['policy'] == 'zero') & (df['topology_case'] == tp)]
    if 'record_type' in df:
        df = df[df['record_type'] == 'dispatch_attempt']
    return df


def overall(statuses):
    if any(s == 'FAIL' for s in statuses.values()):
        return 'FAIL'
    if not any(s == 'PASS' for s in statuses.values()):
        return 'NOT_AVAILABLE'
    return 'PASS' if all(s == 'PASS' for s in statuses.values()) else 'PASS_PARTIAL'


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--pure-dir', required=True, type=Path)
    p.add_argument('--atol', type=float, default=1e-9)
    a = p.parse_args(argv)
    man = read_json(a.pure_dir / 'run_manifest.json')
    info = man.get('pure_zero') or {}
    if man.get('status') != 'finished' or info.get('feeder') != 'IEEE69' or info.get('smoke') \
            or info.get('load_scaling_factor') != 1.0:
        raise SystemExit('Cross-check needs a finished, non-smoke IEEE69 1.0x pure zero run')
    if (a.pure_dir / 'cross_check_report.json').exists():
        raise SystemExit('cross_check_report.json already exists; refusing to overwrite')
    for ref in info['a0_references'].values():
        if sha(ref['path']) != ref['sha256']:
            raise SystemExit('A0 reference manifest changed since the pure run: ' + ref['path'])
    pure = read_steps(a.pure_dir)
    tables, per_tp, statuses = [], {}, {}
    for tp in TOPOLOGIES:
        ref_csv = Path(info['a0_references'][tp]['path']).parent / 'step_metrics.csv'
        header = set(pd.read_csv(ref_csv, nrows=0).columns)
        missing = [c for c in [*NUMERIC, *FLAGS, 'policy', 'backup_activated'] if c not in header]
        ref = reference_rows(ref_csv, tp) if not missing else pd.DataFrame()
        if missing or ref.empty:
            statuses[tp] = 'NOT_OBSERVABLE'
            per_tp[tp] = dict(status='NOT_OBSERVABLE', reference=str(ref_csv),
                              reason=f'reference lacks {missing}' if missing else 'no zero-policy rows')
        else:
            table, coverage = compare(pure[pure.topology_case == tp], ref, a.atol)
            ok = (table['status'] == 'PASS').all() and coverage['pure_violating_steps'] == coverage['reference_backup_activations']
            statuses[tp] = 'PASS' if ok else 'FAIL'
            per_tp[tp] = dict(status=statuses[tp], reference=str(ref_csv), reference_sha256=sha(ref_csv), coverage=coverage,
                              failing_metrics=table.loc[table.status != 'PASS', 'metric'].tolist())
            tables.append(table.assign(topology_case=tp))
        print(f'CROSS_CHECK {tp}: {statuses[tp]}', flush=True)
    status = overall(statuses)
    if tables:
        pd.concat(tables, ignore_index=True).to_csv(a.pure_dir / 'cross_check_metrics.csv', index=False)
    write_json(a.pure_dir / 'cross_check_report.json', dict(status=status, feeder='IEEE69', load_scaling_factor=1.0,
        pure_manifest_sha256=sha(a.pure_dir / 'run_manifest.json'), pure_step_metrics_sha256=sha(a.pure_dir / 'step_metrics.csv'),
        atol=a.atol, topologies=per_tp, not_observable_topologies=[t for t, s in statuses.items() if s == 'NOT_OBSERVABLE'],
        checked_utc=datetime.now(timezone.utc).isoformat()))
    print('CROSS_CHECK', status, flush=True)
    if status == 'FAIL':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
