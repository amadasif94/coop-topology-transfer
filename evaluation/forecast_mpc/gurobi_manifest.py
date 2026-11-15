"""Consolidated run_manifest.json for the Gurobi F-MPC results of one feeder and load factor. Maintained.

    python -m evaluation.forecast_mpc.gurobi_manifest --feeder ieee33 [--scales 1.0 1.1 1.2 1.3] [--check-only]
        [--results-root results/soc_feasible_affine_v2] [--production-root results/fmpc_gurobi]
        [--replace-existing "REASON"]

The Gurobi F-MPC runs publish one topology per task (evaluation.forecast_mpc.gurobi_run: <TP>/completion.json +
tables/). For the final analyses they were copied to
    <results-root>/<feeder>/forecast_mpc/load_x<pct>/gurobi_topologies/<TP>/{completion.json, tables/}
and their CSV tables concatenated into <results-root>/<feeder>/forecast_mpc/load_x<pct>/*.csv. This tool writes the
consolidated run_manifest.json of that load-factor folder, with status 'finished' ONLY after every check passes:
  per topology (all of gurobi_run.TOPOLOGIES[feeder]; none missing, none extra)
    completion.json present, schema of gurobi_run, identity_sha256 = sha256 of its identity; identity = this feeder,
    topology and load factor, full December (31 days, no smoke), the production Gurobi configuration (60 s, 1e-4,
    incumbents accepted, seed 42); every published table file matches the receipt's sha256 (copy intact);
    gurobi_verification from Gurobi invocation records with highs_calls 0 and solves = 31 x 96; tables/run_manifest.json
    finished, same feeder/topology/load factor, Gurobi solver block equal to the receipt's; table structure and day /
    step coverage (ieee69_array.validate_tables), every episode completed, every fmpc_solves row solver_backend gurobi;
    if the production copy <production-root>/<feeder>/load_x<pct>/<TP>/completion.json exists it must be identical.
  forecasts: every December forecast record still has the sha256 recorded in the identity and its npz matches the
    record (forecasts preserved).
  consolidated tables (aggregate_summary, episode_summary, fmpc_days, fmpc_solves, step_metrics, terminal_soc_summary):
    exactly the union of the topology tables, row for row (value-equal, numbers compared as floats; a column a
    topology does not have must be empty), no duplicate, missing or foreign row, every fmpc_solves row Gurobi.
  sources: the recorded F-MPC source hashes are compared with the files present now and the differences are REPORTED
    (source_files_differing_now); they do not change the published results, so they do not block.
The manifest lists every topology's receipt (path, sha256, attempt, verification, Gurobi version, code hash, source
revision); a field is recorded as shared only if identical for all topologies, otherwise per topology. It is written
atomically (complete temporary file, then a no-overwrite hard link). An existing manifest is accepted only if it is
this tool's manifest of exactly the same receipts and tables; anything else is refused unless --replace-existing
REASON is given, in which case the old file is kept as run_manifest.replaced_<UTC>.json and the reason recorded.
Standard library only (login-node safe). Nothing else is written, moved or deleted; results are never regenerated.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import uuid

from evaluation.forecast_mpc.gurobi_run import (GUROBI_FMPC_CONFIG, REPO, SCHEMA, TEST_DAYS, TOPOLOGIES, fsync_dir,
                                                json_sha, read_json, require, sha)
from evaluation.forecast_mpc.ieee69_array import TABLES, inventory, rows, validate_tables

csv.field_size_limit(sys.maxsize if sys.maxsize < 2 ** 31 else 2 ** 31 - 1)
CONSOLIDATED_SCHEMA = 'graph-marl-dess-fmpc-gurobi-consolidated/1'
FORECAST_SCHEMA = 'graph-marl-dess-forecast/2'            # evaluation.forecast_mpc.common (kept import-free)
STEPS = 96
SCALES = (1.0, 1.1, 1.2, 1.3)


def utc():
    return datetime.now(timezone.utc).isoformat()


def label(scale):
    return f'load_x{int(round(float(scale) * 100))}'


def flag(value):
    return str(value).strip().lower() in ('true', '1', '1.0')


def canonical(row):
    """Row digest that ignores empty cells and compares numbers as floats (CSV round trips may write 5 as 5.0)."""
    out = []
    for k, v in sorted(row.items()):
        if k is None or v is None or v == '':
            continue
        try:
            v = repr(float(v))
        except ValueError:
            pass
        out.append((k, v))
    return hashlib.sha256(json.dumps(out).encode()).hexdigest()


def key_of(name, row):
    if name == 'aggregate_summary':
        return (row['topology_case'],)
    k = (row['topology_case'], int(float(row['start_index'])))
    return k + (int(float(row['step'])),) if name in ('step_metrics', 'fmpc_solves') else k


# ----------------------------------------------------------------------------------------------- per topology
def check_topology(feeder, scale, tp, topo, prod):
    """Validate one published Gurobi topology; returns (provenance entry, environment entry, forecast identity)."""
    receipt_path = topo / 'completion.json'
    require(receipt_path.is_file(), f'{tp}: no completion.json in {topo} (not published: failed, incomplete or '
                                    'not copied)')
    receipt = read_json(receipt_path)
    require(receipt.get('schema') == SCHEMA, f'{tp}: completion.json schema {receipt.get("schema")!r}')
    identity = receipt.get('identity') or {}
    require(receipt.get('identity_sha256') == json_sha(identity), f'{tp}: completion.json identity hash mismatch')
    exp, g = identity.get('expected') or {}, identity.get('gurobi') or {}
    require(exp.get('feeder') == feeder and exp.get('topology') == tp, f'{tp}: receipt is for '
            f'{exp.get("feeder")}/{exp.get("topology")}')
    require(float(exp.get('scale', -1)) == float(scale), f'{tp}: receipt load factor {exp.get("scale")}, expected {scale}')
    starts = [int(s) for s in exp.get('intended_start_indices') or []]
    require(not exp.get('max_days') and len(starts) == TEST_DAYS and starts == exp.get('december_start_indices'),
            f'{tp}: receipt is not a full {TEST_DAYS}-day December run (smoke or partial)')
    require(all(g.get(k) == v for k, v in GUROBI_FMPC_CONFIG.items()), f'{tp}: Gurobi configuration {g}')
    tables = topo / 'tables'
    require(tables.is_dir() and inventory(tables) == receipt.get('files'),
            f'{tp}: published tables differ from completion.json (copy changed or incomplete)')
    v = receipt.get('gurobi_verification') or {}
    solves = TEST_DAYS * STEPS
    require(v.get('verified') == 'gurobi_invocation_records', f'{tp}: no Gurobi invocation-record verification')
    require(v.get('highs_calls') == 0, f'{tp}: highs_calls = {v.get("highs_calls")}')
    require(v.get('solves') == solves and v.get('lp_calls') == solves and int(v.get('gurobi_calls', 0)) >= solves,
            f'{tp}: verification {v} does not cover {solves} solved decisions')
    manifest = read_json(tables / 'run_manifest.json')
    env, = manifest.get('environments') or [None]
    require(manifest.get('status') == 'finished' and env is not None and env.get('feeder') == feeder.upper()
            and str(env.get('topology')).upper() == tp and env.get('intended_episodes') == TEST_DAYS,
            f'{tp}: topology run_manifest.json is not the finished {feeder}/{tp} run')
    r = manifest.get('forecast_mpc_record') or {}
    require(float((env.get('data_scaling') or {}).get('load_scaling_factor', -1)) == float(scale)
            and float(r.get('load_scaling_factor', -1)) == float(scale), f'{tp}: manifest load factor differs')
    require(r.get('solver_backend') == g and r.get('allow_non_optimal') is True
            and (r.get('settings') or {}).get('time_limit_s') == GUROBI_FMPC_CONFIG['time_limit_s']
            and (r.get('settings') or {}).get('mip_rel_gap') == GUROBI_FMPC_CONFIG['mip_rel_gap'],
            f'{tp}: manifest solver block differs from the receipt')
    validate_tables(tables, tp, starts)
    for e in rows(tables, 'episode_summary'):
        require(int(e['completed_episodes']) == 1, f'{tp}: episode {e["start_index"]} did not complete')
    labels = [s.get('solver_backend') for s in rows(tables, 'fmpc_solves')]
    require(len(labels) == solves and set(labels) == {'gurobi'}, f'{tp}: fmpc_solves.csv rows not all Gurobi')
    production = None
    if prod is not None:
        p = prod / 'completion.json'
        production = dict(path=str(p), present=p.is_file())
        if p.is_file():
            require(sha(p) == sha(receipt_path), f'{tp}: copied completion.json differs from the production {p}')
            production['identical'] = True
    differing = sorted(k for k, h in (r.get('source_inventory') or {}).items()
                       if not (REPO / k).is_file() or sha(REPO / k) != h)
    entry = dict(topology=tp, completion_json=str(receipt_path), completion_sha256=sha(receipt_path),
                 identity_sha256=receipt['identity_sha256'], attempt_id=receipt.get('attempt_id'),
                 published_at_utc=receipt.get('published_at_utc'), gurobi_verification=v,
                 gurobi_version=g.get('gurobi_version'), threads=g.get('threads'),
                 code_sha256=identity.get('code_sha256'), source_revision=manifest.get('source_revision'),
                 source_revision_status=manifest.get('source_revision_status'),
                 topology_manifest_sha256=sha(tables / 'run_manifest.json'),
                 input_files_sha256=exp.get('input_files_sha256'), series_sha256=exp.get('series_sha256'),
                 forecast_directory=exp.get('forecast_directory'), forecasts_sha256=json_sha(exp.get('forecasts')),
                 settings=exp.get('settings'), seed=manifest.get('seed'), production_copy=production,
                 source_files_differing_now=differing)
    return entry, env, (exp.get('forecast_directory'), exp.get('forecasts'))


def check_forecasts(directory, forecasts, feeder):
    """The December forecasts recorded in the identities are still the same files (record sha256, npz sha256)."""
    d = Path(directory)
    n = dict(ieee33=33, ieee69=69)[feeder]
    names = [f'{k}_node_{i}' for k in ('load', 'pv') for i in range(1, n + 1)]
    require(sorted(forecasts or {}) == sorted(names), f'forecast identity does not list the {2 * n} {feeder} targets')
    for t in names:
        record, npz = d / f'{t}.json', d / f'{t}.npz'
        require(record.is_file() and npz.is_file(), f'forecast {t} missing in {d}')
        require(sha(record) == forecasts[t], f'forecast record {record} changed since the Gurobi runs')
        content = read_json(record)
        require(content.get('schema') == FORECAST_SCHEMA and content.get('npz_sha256') == sha(npz),
                f'forecast {npz} does not match its record')


# ----------------------------------------------------------------------------------------------- consolidated tables
def check_consolidated(folder, sources):
    """Every consolidated table = the union of the topology tables, row for row. Returns {table: rows, sha256}."""
    out = {}
    for name in TABLES:
        expected = {}
        for tp, tables in sources.items():
            for r in rows(tables, name):
                expected[key_of(name, r)] = canonical(r)
        path = folder / f'{name}.csv'
        require(path.is_file(), f'consolidated {path.name} missing in {folder}')
        seen = set()
        for r in rows(folder, name):
            try:
                k = key_of(name, r)
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f'{path.name}: unreadable row key ({exc})') from exc
            require(k not in seen, f'{path.name}: duplicate row {k}')
            seen.add(k)
            require(k in expected, f'{path.name}: row {k} is not in the published topology tables')
            require(canonical(r) == expected[k], f'{path.name}: row {k} differs from the published topology table')
            if name == 'fmpc_solves':
                require(r.get('solver_backend') == 'gurobi', f'{path.name}: row {k} not solved by Gurobi')
        lost = sorted(set(expected) - seen)
        require(not lost, f'{path.name}: {len(lost)} published row(s) missing, e.g. {lost[:3]}')
        out[name] = dict(rows=len(seen), sha256=sha(path))
    return out


# ----------------------------------------------------------------------------------------------- manifest
def shared(entries, field):
    values = [json.dumps(e.get(field), sort_keys=True, default=str) for e in entries]
    return json.loads(values[0]) if len(set(values)) == 1 else None


def build(feeder, scale, results_root, production_root):
    folder = Path(results_root) / feeder / 'forecast_mpc' / label(scale)
    require(folder.is_dir(), f'{folder} does not exist')
    base = folder / 'gurobi_topologies'
    expected = TOPOLOGIES[feeder]
    present = sorted(p.name for p in base.iterdir() if p.is_dir()) if base.is_dir() else []
    extra = sorted(set(present) - set(expected))
    require(not extra, f'{base}: unexpected topology folders {extra}')
    entries, envs, sources, forecasts = [], [], {}, set()
    for tp in expected:
        prod = Path(production_root) / feeder / label(scale) / tp if production_root else None
        entry, env, fc = check_topology(feeder, scale, tp, base / tp, prod)
        entries.append(entry)
        envs.append(env)
        sources[tp] = base / tp / 'tables'
        forecasts.add(json.dumps(fc, sort_keys=True))
    require(len(forecasts) == 1, 'the topologies were run on different December forecasts')
    directory, identity = json.loads(forecasts.pop())
    check_forecasts(directory, identity, feeder)
    tables = check_consolidated(folder, sources)
    provenance_sha256 = json_sha(dict(topologies={e['topology']: e['completion_sha256'] for e in entries},
                                      tables={k: v['sha256'] for k, v in tables.items()}))
    fields = ('gurobi_version', 'threads', 'code_sha256', 'source_revision', 'settings', 'seed', 'series_sha256',
              'input_files_sha256', 'forecasts_sha256', 'forecast_directory')
    common = {f: shared(entries, f) for f in fields}
    totals = {k: sum(int(e['gurobi_verification'][k]) for e in entries)
              for k in ('solves', 'gurobi_calls', 'lp_calls', 'milp_calls', 'highs_calls')}
    manifest = dict(
        schema=CONSOLIDATED_SCHEMA, status='finished', kind='consolidated Gurobi F-MPC topology results',
        controller='FORECAST_MPC', mode='test', feeder=feeder, load_scaling_factor=float(scale),
        topologies=list(expected), days_per_topology=TEST_DAYS, decisions_per_day=STEPS,
        seed=common['seed'], solver_backend=dict(GUROBI_FMPC_CONFIG, gurobi_version=common['gurobi_version'],
                                                 threads=common['threads']),
        shared_identity={f: common[f] for f in fields if common[f] is not None},
        differs_between_topologies=sorted(f for f in fields if common[f] is None),
        topology_records=entries, environments=envs, consolidated_tables=tables, totals=totals,
        provenance_sha256=provenance_sha256, forecasts_verified=dict(directory=directory, targets=len(identity)),
        checks=['completion.json schema, identity hash, feeder/topology/load factor, 31-day December identity',
                'production Gurobi configuration (60 s, 1e-4, incumbents accepted, seed 42)',
                'published table sha256 = completion.json files (copy intact)',
                'gurobi_verification: invocation records, highs_calls 0, solves 2976 per topology',
                'topology run_manifest.json finished, same identity and Gurobi solver block',
                'day/step coverage and uniqueness, every episode completed, every solve Gurobi',
                'consolidated tables = union of the topology tables, row for row',
                'December forecast records and npz unchanged'],
        note='Metadata only: generated from the verified per-topology Gurobi publications; no evaluation was rerun '
             'and no result file was changed. Per-topology source revisions and code hashes are listed in '
             'topology_records; a field appears in shared_identity only if identical for all topologies.',
        generated_by=dict(tool='evaluation/forecast_mpc/gurobi_manifest.py', tool_sha256=sha(Path(__file__)),
                          host=platform.node(), python=platform.python_version(), at_utc=utc()))
    return folder, manifest


def write_manifest(folder, manifest, replace_reason=None):
    """Atomic, never a silent overwrite. Returns 'written', 'replaced' or 'verified_existing'."""
    path = folder / 'run_manifest.json'
    outcome = 'written'
    if path.exists() or path.is_symlink():
        try:
            old = read_json(path)
        except (OSError, ValueError):
            old = None
        if old and old.get('schema') == CONSOLIDATED_SCHEMA and old.get('status') == 'finished' \
                and old.get('provenance_sha256') == manifest['provenance_sha256']:
            return 'verified_existing'
        require(replace_reason, f'{path} exists and is not the consolidated manifest of these receipts and tables; '
                                'refusing to overwrite (inspect it; pass --replace-existing "REASON" to replace it)')
        kept = folder / f'run_manifest.replaced_{datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")}.json'
        require(not kept.exists(), f'{kept} exists')
        manifest['replaces'] = dict(previous_sha256=sha(path), previous_kept_as=str(kept), reason=replace_reason,
                                    previous_schema=(old or {}).get('schema'), previous_status=(old or {}).get('status'))
        os.rename(path, kept)
        outcome = 'replaced'
    text = json.dumps(manifest, indent=2, sort_keys=True, default=str, allow_nan=False)
    temp = folder / f'.run_manifest.json.{os.getpid()}.{uuid.uuid4().hex[:12]}.tmp'
    try:
        with temp.open('x', encoding='utf-8') as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temp, path)                       # complete file appears at once; fails if path appeared meanwhile
    finally:
        try:
            temp.unlink()
        except OSError:
            pass
    fsync_dir(folder)
    return outcome


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--feeder', required=True, choices=sorted(TOPOLOGIES))
    p.add_argument('--scales', type=float, nargs='+', default=list(SCALES), choices=SCALES)
    p.add_argument('--results-root', default='results/soc_feasible_affine_v2')
    p.add_argument('--production-root', default='results/fmpc_gurobi',
                   help='original Gurobi publications to cross-check (skipped per topology if absent; "" = never)')
    p.add_argument('--check-only', action='store_true', help='validate and report; write nothing')
    p.add_argument('--replace-existing', metavar='REASON', help='replace an existing different manifest (kept)')
    a = p.parse_args(argv)
    failed = 0
    for scale in a.scales:
        try:
            folder, manifest = build(a.feeder, scale, a.results_root, a.production_root or None)
            outcome = 'check_ok' if a.check_only else write_manifest(folder, manifest, a.replace_existing)
        except (ValueError, OSError, KeyError) as exc:
            failed += 1
            print(f'FMPC_MANIFEST_REFUSED {a.feeder} x{scale}: {type(exc).__name__}: {exc}', flush=True)
            continue
        t = manifest['totals']
        summary = (f'topologies={len(manifest["topologies"])} solves={t["solves"]} gurobi_calls={t["gurobi_calls"]} '
                   f'highs_calls={t["highs_calls"]} rows=' + ','.join(f'{k}:{v["rows"]}' for k, v in
                                                                     manifest['consolidated_tables'].items()))
        differing = sorted({f for e in manifest['topology_records'] for f in e['source_files_differing_now']})
        if differing:
            print(f'FMPC_MANIFEST_NOTE {a.feeder} x{scale}: source files differ now from the recorded hashes '
                  f'(reported, not blocking): {differing}', flush=True)
        print(f'FMPC_MANIFEST_{outcome.upper()} {folder / "run_manifest.json"} {summary}', flush=True)
    return 1 if failed else 0


if __name__ == '__main__':
    raise SystemExit(main())
