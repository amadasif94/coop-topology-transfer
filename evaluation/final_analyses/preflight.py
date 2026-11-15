"""Read-only preflight of the inputs of the four final analyses. Python standard library only (login-node safe; no
PyTorch, no AutoGluon, no environment). Writes nothing.

    python -m evaluation.final_analyses.preflight --results-root results/soc_feasible_affine_v2 [--feeders ieee33 ieee69]

Checks, per feeder, and prints one OK / MISSING / NOTE line per item, then a summary; exit code 1 if anything needed is
missing:
  bootstrap        finished test runs of A0-A3 (seeds 0-2), PF-MPC, F-MPC, PQ-RBC, zero-action at x1.0-x1.3; each
                   episode_summary.csv has the metric columns and all maintained topologies x 31 days; the F-MPC
                   folders' fmpc_solves.csv shows Gurobi on all 96 decisions of every episode
  soc_diagnostics  A0/A3 step_metrics.csv headers: soc_agent_<i> for every battery, executed power, terminal shortfall
  action_replay    A0/A3 step_metrics.csv has marl_power_soc_mapped_kw; whether the target topology (IEEE33 TP6/TP7,
                   IEEE69 TP10) is already in the authoritative test tables (reused) or must be generated (NOTE)
  timing           frozen selections of A0-A3, F-MPC 'dec' models and forecasts complete, PQ-RBC thresholds file
  code             the final_analyses modules and generate.py's issue_steps argument are installed
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

csv.field_size_limit(sys.maxsize if sys.maxsize < 2 ** 31 else 2 ** 31 - 1)
TOPOLOGIES = dict(ieee33=[f'TP{i}' for i in range(1, 8)], ieee69=[f'TP{i}' for i in range(1, 11)])
AGENTS = dict(ieee33=5, ieee69=9)                       # batteries (soc_agent_<i> columns)
BUSES = dict(ieee33=33, ieee69=69)                      # F-MPC forecast targets: load and PV of every bus
MODEL_SCHEMA = 'graph-marl-dess-forecast-model/2'       # evaluation.forecast_mpc.common
FORECAST_SCHEMA = 'graph-marl-dess-forecast/2'
SAVED_ARTIFACTS = ('predictor.pkl', 'learner.pkl', 'metadata.json', 'version.txt')   # evaluation.forecast_mpc.train
OPTIONAL_COLUMNS = {'zero_action_no_backup': ('b3_terminal_shortfall_mean',)}       # common.OPTIONAL_METRIC_COLUMNS
TARGETS = dict(ieee33=('TP6', 'TP7'), ieee69=('TP10',))
SCALES = dict(load_x100=1.0, load_x110=1.1, load_x120=1.2, load_x130=1.3)
VARIANTS = dict(mlp_a0='A0', mlp_a1='A1', mlp_a2='A2', mlp_a3='A3')
REFERENCES = ('pf_mpc', 'forecast_mpc', 'price_quantile_rbc', 'zero_action_no_backup')
EPISODE_COLUMNS = ('controller', 'topology_case', 'start_index', 'adjusted_energy_cost', 'raw_energy_cost',
                   'total_reward_mean', 'final_network_feasibility_rate', 'mean_voltage_deviation',
                   'b3_terminal_shortfall_mean')
CODE = ('evaluation/final_analyses/__init__.py', 'evaluation/final_analyses/common.py',
        'evaluation/final_analyses/bootstrap.py', 'evaluation/final_analyses/timing.py',
        'evaluation/final_analyses/soc_diagnostics.py', 'evaluation/final_analyses/action_replay.py',
        'evaluation/final_analyses/preflight.py', 'SLURM Script/final_analyses.slurm')


class Report:
    def __init__(self):
        self.missing = 0

    def line(self, status, analysis, text):
        if status == 'MISSING':
            self.missing += 1
        print(f'{status:8s} [{analysis}] {text}', flush=True)


def finished(d):
    m = d / 'run_manifest.json'
    try:
        return m.is_file() and json.loads(m.read_text()).get('status') == 'finished'
    except (OSError, ValueError):
        return False


def header(path):
    with open(path, newline='', encoding='utf-8') as f:
        return next(csv.reader(f), [])


def episodes(path):
    with open(path, newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    cols = set(rows[0]) if rows else set(header(path))
    days = {}
    for r in rows:
        days.setdefault(r.get('topology_case'), set()).add(r.get('start_index'))
    return cols, {k: len(v) for k, v in days.items()}


def check_episode_table(rep, d, feeder, analysis, rl, controller=None):
    p = d / 'episode_summary.csv'
    if not p.is_file():
        rep.line('MISSING', analysis, f'{p}')
        return set()
    cols, days = episodes(p)
    optional = OPTIONAL_COLUMNS.get(controller, ())
    lost = [c for c in EPISODE_COLUMNS + (('run_id',) if rl else ()) if c not in cols and c not in optional]
    if lost:
        rep.line('MISSING', analysis, f'{p}: columns {lost}')
    absent_optional = [c for c in optional if c not in cols]
    if absent_optional:
        rep.line('NOTE', analysis, f'{p}: optional {absent_optional} absent; comparisons of that metric with '
                                   f'{controller} are reported as not estimable')
    tps = [t for t in TOPOLOGIES[feeder] if t not in days]
    short = {t: n for t, n in days.items() if n != 31}
    if tps or short:
        rep.line('NOTE' if analysis == 'replay' else 'MISSING', analysis,
                 f'{p}: topologies missing {tps}; days != 31: {short}')
    return set(days)


def check_fmpc_backend(rep, d):
    """The consolidated F-MPC folder holds the Gurobi results: fmpc_solves.csv has solver_backend = 'gurobi' on every
    row and 96 rows per episode of episode_summary.csv (same rule as common.fmpc_backend_problems)."""
    p, e = d / 'fmpc_solves.csv', d / 'episode_summary.csv'
    if not p.is_file() or not e.is_file():
        rep.line('MISSING', 'bootstrap', f'{p}: missing (F-MPC solver cannot be established)')
        return
    with open(p, newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    with open(e, newline='', encoding='utf-8') as f:
        days = {(r.get('topology_case'), r.get('start_index')) for r in csv.DictReader(f)}
    labels = {r.get('solver_backend') for r in rows}
    per = {}
    for r in rows:
        k = (r.get('topology_case'), r.get('start_index'))
        per[k] = per.get(k, 0) + 1
    ok_backend = labels == {'gurobi'}
    ok_count = set(per) == days and all(n == 96 for n in per.values())
    rep.line('OK' if ok_backend and ok_count else 'MISSING', 'bootstrap',
             f'{p}: ' + ('Gurobi on all ' + str(len(rows)) + ' decisions' if ok_backend else
                         f'not Gurobi results (solver_backend values {sorted(map(str, labels))[:3]}; HiGHS-era or '
                         'not consolidated)') + ('' if ok_count else f'; solves do not match the {len(days)} episodes'))


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def dir_sha256(path):
    """Same rule as evaluation.forecast_mpc.common.dir_sha256 (AutoGluon log files excluded)."""
    path, h = Path(path), hashlib.sha256()
    files = [p for p in path.rglob('*') if p.is_file() and p.suffix != '.log' and 'logs' not in p.relative_to(path).parts]
    for f in sorted(files):
        h.update(f.relative_to(path).as_posix().encode() + b'\0' + sha256_file(f).encode() + b'\n')
    return h.hexdigest()


def read_record(path):
    try:
        return json.loads(Path(path).read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None


def check_fmpc(rep, root, feeder, deep=False):
    """The exact December F-MPC artifacts the timing benchmark loads (one per bus and quantity: load_node_1..N,
    pv_node_1..N; N = number of BUSES, not batteries), each validated against its record:
      models/dec/<target>.json    schema /2, feeder, stage 'dec', target, cutoff 2018-12-01 00:00:00, and the folder
                                  models/dec/<target>/ with a finished AutoGluon save (predictor.pkl, learner.pkl,
                                  metadata.json, version.txt, non-empty models/); --deep also re-hashes the folder
                                  against model_dir_sha256 (generate.load_predictor checks it again when loading)
      forecasts/dec/<target>.json schema /2, feeder, stage 'dec', month 2018-12, 31 days, the same series and model
                                  hash as the model record, and <target>.npz whose sha256 equals npz_sha256
    Other files in those folders (leaderboards, extra JSON) are reported as a NOTE, never counted."""
    base = root / feeder / 'forecast_mpc'
    names = [f'{k}_node_{i}' for k in ('load', 'pv') for i in range(1, BUSES[feeder] + 1)]
    model_problems, forecast_problems, series = [], [], set()
    for t in names:
        rec = read_record(base / 'models' / 'dec' / f'{t}.json')
        if rec is None:
            model_problems.append(f'{t}: record missing or unreadable')
        else:
            bad = [k for k, v in (('schema', MODEL_SCHEMA), ('feeder', feeder), ('stage', 'dec'), ('target', t),
                                  ('cutoff_time', '2018-12-01 00:00:00')) if rec.get(k) != v]
            folder = base / 'models' / 'dec' / t
            lost = [a for a in SAVED_ARTIFACTS if not (folder / a).is_file()]
            if not (folder / 'models').is_dir() or not any(p.is_file() for p in (folder / 'models').rglob('*')):
                lost.append('models/')
            if bad or lost:
                model_problems.append(f'{t}: record fields {bad} wrong, save files missing {lost}')
            elif deep and dir_sha256(folder) != rec.get('model_dir_sha256'):
                model_problems.append(f'{t}: predictor folder differs from model_dir_sha256')
            series.add(rec.get('series_sha256'))
        frec = read_record(base / 'forecasts' / 'dec' / f'{t}.json')
        npz = base / 'forecasts' / 'dec' / f'{t}.npz'
        if frec is None or not npz.is_file():
            forecast_problems.append(f'{t}: record or npz missing')
            continue
        bad = [k for k, v in (('schema', FORECAST_SCHEMA), ('feeder', feeder), ('stage', 'dec'), ('target', t),
                              ('month', '2018-12'), ('days', 31)) if frec.get(k) != v]
        if rec is not None:
            bad += [k for k in ('series_sha256', 'model_dir_sha256') if frec.get(k) != rec.get(k)]
        if sha256_file(npz) != frec.get('npz_sha256'):
            bad.append('npz_sha256 (file changed or truncated)')
        if bad:
            forecast_problems.append(f'{t}: {bad}')
        series.add(frec.get('series_sha256'))
    for kind, problems in (('models', model_problems), ('forecasts', forecast_problems)):
        rep.line('MISSING' if problems else 'OK', 'timing', f'F-MPC {kind}/dec: {len(names) - len(problems)}/{len(names)} '
                 f'targets valid' + (f'; e.g. {problems[:3]}' if problems else ''))
        d = base / kind / 'dec'
        extra = sorted(p.name for p in d.glob('*.json') if p.stem not in names) if d.is_dir() else []
        if extra:
            rep.line('NOTE', 'timing', f'F-MPC {kind}/dec: {len(extra)} unrelated JSON file(s) ignored, e.g. {extra[:3]}')
    if len(series - {None}) > 1:
        rep.line('MISSING', 'timing', f'F-MPC dec records refer to {len(series - {None})} different time series')


def run(root, feeders, deep=False):
    rep, root = Report(), Path(root)
    for f in CODE:
        rep.line('OK' if Path(f).is_file() else 'MISSING', 'code', f)
    gen = Path('evaluation/forecast_mpc/generate.py')
    rep.line('OK' if gen.is_file() and 'issue_steps' in gen.read_text(encoding='utf-8') else 'MISSING', 'code',
             'generate.py has the issue_steps argument')
    for feeder in feeders:
        print(f'===== {feeder}', flush=True)
        n = AGENTS[feeder]
        for key, variant in VARIANTS.items():
            for k in (0, 1, 2):
                sel = sorted((root / feeder / 'selections' / feeder / key / f'seed_{k}').glob('*/selection.json'))
                rep.line('OK' if sel else 'MISSING', 'timing/replay', f'{key} seed {k}: frozen selection '
                         f'{"found" if sel else "absent"}')
                for name in SCALES:
                    d = root / feeder / 'mlp' / variant / f'seed_{k}' / 'test' / name
                    if not finished(d):
                        rep.line('MISSING', 'bootstrap', f'{d}: not finished')
                        continue
                    tps = check_episode_table(rep, d, feeder, 'bootstrap', True)
                    if key in ('mlp_a0', 'mlp_a3'):
                        cols = set(header(d / 'step_metrics.csv')) if (d / 'step_metrics.csv').is_file() else set()
                        need = [f'soc_agent_{i}' for i in range(n)] + ['marl_terminal_soc_shortfall', 'marl_terminal_completed']
                        lost = [c for c in need if c not in cols]
                        if 'executed_battery_power_kw' not in cols and f'dess_power_kw_agent_{n - 1}' not in cols:
                            lost.append('executed_battery_power_kw')
                        rep.line('MISSING' if lost else 'OK', 'soc_diagnostics', f'{d.relative_to(root)}: step columns'
                                 + (f' missing {lost}' if lost else ''))
                        rep.line('OK' if 'marl_power_soc_mapped_kw' in cols and 'TP1' in tps else 'MISSING', 'replay',
                                 f'{d.relative_to(root)}: TP1 source requests (marl_power_soc_mapped_kw)')
                        for t in TARGETS[feeder]:
                            rep.line('OK' if t in tps else 'NOTE', 'replay',
                                     f'{d.relative_to(root)}: closed-loop {t} ' + ('present (reused)' if t in tps else
                                     'absent: action_replay will run an evaluation-only closed loop first'))
        for c in REFERENCES:
            for name in SCALES:
                d = root / feeder / c / name
                if not finished(d):
                    rep.line('MISSING', 'bootstrap', f'{d}: not finished')
                    continue
                check_episode_table(rep, d, feeder, 'bootstrap', False, controller=c)
                if c == 'forecast_mpc':
                    check_fmpc_backend(rep, d)
        check_fmpc(rep, root, feeder, deep)
        th = root / feeder / 'price_quantile_rbc' / 'thresholds.json'
        rep.line('OK' if th.is_file() else 'MISSING', 'timing', f'{th}')
    print(f'PREFLIGHT {"COMPLETE" if rep.missing == 0 else "INCOMPLETE"}: {rep.missing} missing item(s)', flush=True)
    return rep.missing


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--results-root', default='results/soc_feasible_affine_v2')
    p.add_argument('--feeders', nargs='+', default=['ieee33', 'ieee69'], choices=sorted(TOPOLOGIES))
    p.add_argument('--deep', action='store_true', help='also re-hash every F-MPC predictor folder (slower)')
    a = p.parse_args(argv)
    raise SystemExit(1 if run(a.results_root, a.feeders, a.deep) else 0)


if __name__ == '__main__':
    main()
