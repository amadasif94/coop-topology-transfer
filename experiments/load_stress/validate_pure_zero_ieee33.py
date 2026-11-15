"""Cross-check: pure zero-action run vs the existing zero-action + backup run at the same load scale.

    python -u -m experiments.load_stress.validate_pure_zero_ieee33 \
        --pure-dir results/load_stress/ieee33/pure_zero_no_backup/load_x<pct> \
        [--reference-dir results/load_stress/ieee33/zero_action/run_0/load_x<pct>]

In the zero + backup run the first (pre-backup) power flow of every step is the 0 kW dispatch, so its
backup_initial_* / doe_candidate_* columns must equal the pure run's on every step both runs reached, aligned on
(topology_case, start_index, step). Read only for both runs; writes cross_check_report.json and
cross_check_metrics.csv into the pure run directory (never overwrites). Status:
  PASS           every compared column identical on every matched step, and the violating-step counts agree
  FAIL           any mismatch (exit 1)
  NOT_AVAILABLE  the reference run is absent or unfinished (exit 0; nothing compared)
Informational only: the pure-zero evaluation and summary do not depend on it.
"""
import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.load_stress.evaluate_load_stress_ieee33 import run_dir as stress_run_dir
from experiments.load_stress.summarize_load_stress import as_bool
from experiments.no_shared_kpi.protocol import read_json, sha, write_json

KEYS = ['topology_case', 'start_index', 'step']
NUMERIC = ['backup_initial_score', 'backup_initial_min_voltage', 'backup_initial_max_voltage',
           'backup_initial_max_line_current', 'backup_initial_undervoltage_count', 'backup_initial_overvoltage_count',
           'backup_initial_overloaded_lines', 'backup_initial_max_voltage_violation',
           'backup_initial_max_line_current_violation']
FLAGS = ['doe_candidate_voltage_violation', 'doe_candidate_current_violation', 'doe_candidate_network_violation']


def read_steps(d):
    df = pd.read_csv(Path(d) / 'step_metrics.csv', usecols=lambda c: c in {*KEYS, *NUMERIC, *FLAGS, 'record_type',
                     'backup_activated', 'feasible'}, low_memory=False)
    return df[df['record_type'] == 'dispatch_attempt'] if 'record_type' in df else df


def same(a, b, atol):
    a, b = pd.to_numeric(a, errors='coerce').to_numpy(float), pd.to_numeric(b, errors='coerce').to_numpy(float)
    finite = np.isfinite(a) & np.isfinite(b)
    ok = np.where(finite, np.abs(a - b) <= atol, (np.isnan(a) & np.isnan(b)) | (a == b))
    return ok, float(np.max(np.abs(a - b)[finite], initial=0.0))


def compare(pure, ref, atol):
    merged = pure.merge(ref, on=KEYS, how='outer', suffixes=('__pure', '__ref'), indicator=True)
    both = merged[merged['_merge'] == 'both']
    rows = []
    for c in NUMERIC + FLAGS:
        if f'{c}__pure' not in both or f'{c}__ref' not in both:
            rows.append(dict(metric=c, status='MISSING'))
            continue
        if c in FLAGS:
            ok = (both[f'{c}__pure'].isna() & both[f'{c}__ref'].isna()) | (as_bool(both[f'{c}__pure']) == as_bool(both[f'{c}__ref']))
            ok = ok.fillna(False).astype(bool)  # known vs unknown flag is a mismatch, not a crash
            mx = float('nan')
        else:
            ok, mx = same(both[f'{c}__pure'], both[f'{c}__ref'], atol)
        bad = int((~np.asarray(ok)).sum())
        rows.append(dict(metric=c, matched_steps=int(len(both)), n_mismatch=bad, max_abs_err=mx,
                         status='PASS' if bad == 0 and len(both) else 'FAIL'))
    pure_violations = int((~as_bool(both['feasible__pure'])).sum())
    ref_activations = int(as_bool(both['backup_activated__ref']).sum())
    coverage = dict(matched_steps=int(len(both)), only_in_pure=int((merged['_merge'] == 'left_only').sum()),
                    only_in_reference=int((merged['_merge'] == 'right_only').sum()),
                    pure_violating_steps=pure_violations, reference_backup_activations=ref_activations)
    return pd.DataFrame(rows), coverage


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--pure-dir', required=True, type=Path)
    p.add_argument('--reference-dir', type=Path)
    p.add_argument('--atol', type=float, default=1e-9)
    a = p.parse_args(argv)
    man = read_json(a.pure_dir / 'run_manifest.json')
    info = man.get('pure_zero') or {}
    if man.get('status') != 'finished' or info.get('pilot'):
        raise SystemExit('Cross-check needs a finished, non-pilot pure zero run')
    if (a.pure_dir / 'cross_check_report.json').exists():
        raise SystemExit('cross_check_report.json already exists; refusing to overwrite')
    scale = info['load_scaling_factor']
    ref_dir = a.reference_dir or stress_run_dir('results/load_stress/ieee33', 'zero', 0, scale)
    ref_manifest = Path(ref_dir) / 'run_manifest.json'
    report = dict(feeder='IEEE33', load_scaling_factor=scale, pure_dir=str(a.pure_dir.resolve()),
                  pure_manifest_sha256=sha(a.pure_dir / 'run_manifest.json'), reference_dir=str(Path(ref_dir).resolve()),
                  atol=a.atol, checked_utc=datetime.now(timezone.utc).isoformat())
    ref = read_json(ref_manifest) if ref_manifest.is_file() else {}
    if ref.get('status') != 'finished' or (ref.get('load_stress') or {}).get('load_scaling_factor') != scale:
        write_json(a.pure_dir / 'cross_check_report.json', dict(report, status='NOT_AVAILABLE',
                   reason='zero-action + backup run at this scale is absent or unfinished'))
        print('CROSS_CHECK NOT_AVAILABLE', ref_dir, flush=True)
        return
    table, coverage = compare(read_steps(a.pure_dir), read_steps(ref_dir), a.atol)
    ok = (not table['status'].isin(['FAIL', 'MISSING']).any()
          and coverage['pure_violating_steps'] == coverage['reference_backup_activations'])
    status = 'PASS' if ok else 'FAIL'
    table.to_csv(a.pure_dir / 'cross_check_metrics.csv', index=False)
    write_json(a.pure_dir / 'cross_check_report.json', dict(report, status=status, coverage=coverage,
               reference_manifest_sha256=sha(ref_manifest), pure_step_metrics_sha256=sha(a.pure_dir / 'step_metrics.csv'),
               reference_step_metrics_sha256=sha(Path(ref_dir) / 'step_metrics.csv'),
               failing_metrics=table.loc[table.status != 'PASS', 'metric'].tolist()))
    print('CROSS_CHECK', status, coverage, flush=True)
    if status == 'FAIL':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
