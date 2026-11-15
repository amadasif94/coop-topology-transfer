"""Shared helpers of the final paper analyses. Maintained.

Authoritative inputs (read only), under the results root (default results/soc_feasible_affine_v2):
    <root>/<feeder>/mlp/<A0|A1|A2|A3>/seed_<k>/test/load_x<pct>/   learned controllers (run_id = seed k)
    <root>/<feeder>/pf_mpc/load_x<pct>/                             PF-MPC (single deterministic run)
    <root>/<feeder>/forecast_mpc/load_x<pct>/                       F-MPC (single deterministic run)
    <root>/<feeder>/price_quantile_rbc/load_x<pct>/                 PQ-RBC (single deterministic run)
    <root>/<feeder>/zero_action_no_backup/load_x<pct>/              zero-action reference (single run)
Only folders whose run_manifest.json says status 'finished' are used; smoke folders (load_x<pct>_smoke<N>) never.
F-MPC folders hold the CONSOLIDATED Gurobi results (gurobi_topologies/TP*/ merged into load_x<pct>/); load_episodes
refuses one whose fmpc_solves.csv does not show Gurobi on every decision (fmpc_backend_problems).
Outputs go under results/final_paper_analyses/ (default); guard_output refuses anything inside the results root or any
results/soc_feasible* tree, so authoritative results can never be overwritten.
"""
from datetime import datetime, timezone
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd

from configs.experiments import TOPOLOGIES, RunLayout
from evaluation.evaluate import ZERO_LABEL
from experiments.load_stress.stress_env import ALLOWED_SCALES, scale_label

RESULTS_ROOT = 'results/soc_feasible_affine_v2'
OUT_ROOT = 'results/final_paper_analyses'
RL = ('mlp_a0', 'mlp_a1', 'mlp_a2', 'mlp_a3')
REFERENCES = ('pf_mpc', 'forecast_mpc', 'price_quantile_rbc', ZERO_LABEL)
CONTROLLERS = RL + REFERENCES
DISPLAY = dict(mlp_a0='A0', mlp_a1='A1', mlp_a2='A2', mlp_a3='A3', pf_mpc='PF-MPC', forecast_mpc='F-MPC',
               price_quantile_rbc='PQ-RBC', **{ZERO_LABEL: 'Zero-action'})
SEEDS = (0, 1, 2)
# Episode columns a controller's tables may legitimately lack. The pure zero-action reference (evaluation.zero) is not
# a MarlDESSEnv episode, so it records no terminal SoC shortfall: comparisons of that metric with it are 'not
# estimable'; every other column stays mandatory for every controller.
OPTIONAL_METRIC_COLUMNS = {ZERO_LABEL: ('b3_terminal_shortfall_mean',)}
SCALES = tuple(ALLOWED_SCALES)
STEPS = 96
VARIANT = dict(mlp_a0='A0', mlp_a1='A1', mlp_a2='A2', mlp_a3='A3')


def guard_output(path, results_root=RESULTS_ROOT):
    """Refuse an output location inside the authoritative results (the results root or any results/soc_feasible*)."""
    p = Path(path).resolve()
    root = Path(results_root).resolve()
    if p == root or root in p.parents:
        raise ValueError(f'{path} lies inside the authoritative results root {results_root}; refused')
    parts = [x.lower() for x in p.parts]
    for i, part in enumerate(parts[:-1]):
        if part == 'results' and parts[i + 1].startswith('soc_feasible'):
            raise ValueError(f'{path} lies inside an authoritative results/soc_feasible* tree; refused')
    return p


def stamp():
    return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')


def scale_of(name):
    """load_x110 -> 1.1 (folder names written by experiments.load_stress.stress_env.scale_label)."""
    for s in SCALES:
        if name == scale_label(s):
            return s
    return None


def _finished(d):
    m = d / 'run_manifest.json'
    return m.is_file() and json.loads(m.read_text()).get('status') == 'finished'


def run_dirs(results_root, feeder, controllers=CONTROLLERS, scales=SCALES):
    """[(controller, seed or None, scale, dir)] of finished authoritative runs, plus exclusions."""
    root, layout = Path(results_root), RunLayout(results_root)
    found, excluded = [], []
    for c in controllers:
        if c in RL:
            cands = [(k, d) for k in SEEDS for d in sorted((root / feeder / 'mlp' / VARIANT[c] / f'seed_{k}' / 'test').glob('load_x*'))]
        else:
            label = ZERO_LABEL if c == ZERO_LABEL else c
            base = layout.zero_dir(feeder) if c == ZERO_LABEL else root / feeder / label
            cands = [(None, d) for d in sorted(base.glob('load_x*'))]
        for seed, d in cands:
            s = scale_of(d.name)
            if s is None:
                excluded.append(dict(dir=str(d), reason='not a full run folder (smoke or unknown name)'))
                continue
            if s not in scales:
                continue
            if not _finished(d):
                excluded.append(dict(dir=str(d), reason='manifest missing or not finished'))
                continue
            found.append((c, seed, s, d))
    return found, excluded


def series_dates(feeder, time_series=None):
    """{start_index: calendar date} from the environment's own series (the date of every evaluation day)."""
    from evaluation.forecast_mpc.common import load_series
    s = load_series(feeder, time_series)
    return s, (lambda idx: [str(s.timestamps[int(i)].date()) for i in idx])


FMPC_BACKENDS = ('gurobi', 'highs')
FMPC_BACKEND = 'gurobi'          # the paper's F-MPC: evaluation.forecast_mpc.gurobi_run, consolidated per load factor


def fmpc_backend_problems(d, episodes, backend=FMPC_BACKEND):
    """[] when <d>/fmpc_solves.csv shows that every decision of every episode in `episodes` (episode_summary.csv of
    the same folder) was solved by `backend`, else the problems. 'gurobi': column solver_backend present and 'gurobi'
    on every row (written only by gurobi_run's verified controller; the HiGHS-era tables have no such column).
    'highs': the legacy tables (no Gurobi row). Either way exactly STEPS solve rows per (topology, day) of the episode
    table and no solves of other days, so a partly consolidated folder cannot pass."""
    p = Path(d) / 'fmpc_solves.csv'
    if backend not in FMPC_BACKENDS:
        raise ValueError(f'F-MPC backend must be one of {FMPC_BACKENDS}')
    if not p.is_file():
        return [f'{p}: missing; the F-MPC solver of this folder cannot be established']
    header = set(pd.read_csv(p, nrows=0).columns)
    s = pd.read_csv(p, usecols=[c for c in ('topology_case', 'start_index', 'step', 'solver_backend') if c in header],
                    low_memory=False)
    problems = []
    labels = s['solver_backend'].astype(str) if 'solver_backend' in s else None
    if backend == 'gurobi':
        if labels is None:
            problems.append(f'{p}: no solver_backend column: HiGHS-era F-MPC table, not the Gurobi results')
        elif (labels != 'gurobi').any():
            problems.append(f'{p}: {int((labels != "gurobi").sum())} solve rows not solved by Gurobi '
                            f'({sorted(set(labels[labels != "gurobi"]))[:3]})')
    elif labels is not None and (labels == 'gurobi').any():
        problems.append(f'{p}: {int((labels == "gurobi").sum())} Gurobi solve rows in a HiGHS F-MPC folder')
    per = s.groupby([s.topology_case.astype(str), s.start_index.astype(int)]).size()
    want = pd.MultiIndex.from_frame(episodes[['topology_case', 'start_index']].astype(dict(topology_case=str,
                                                                                         start_index=int)))
    lost = [k for k in want if k not in per.index]
    short = {f'{k[0]}@{k[1]}': int(n) for k, n in per.items() if k in want and n != STEPS}
    extra = [k for k in per.index if k not in want]
    if lost or short or extra:
        problems.append(f'{p}: solves do not match episode_summary.csv: {len(lost)} day(s) without solves, '
                        f'{len(short)} with != {STEPS} solves {dict(list(short.items())[:3])}, {len(extra)} extra day(s)')
    return problems


def load_episodes(results_root, feeder, columns, controllers=CONTROLLERS, scales=SCALES, time_series=None,
                  fmpc_backend=FMPC_BACKEND):
    """Episode-level table: controller, seed (RL) / NaN (single-run references), scale, topology_case, start_index,
    date and the requested metric columns (missing columns are NaN and listed in the returned info). Every F-MPC
    folder must prove its solver (fmpc_backend_problems); a mismatch stops the run, never a silent substitution."""
    runs, excluded = run_dirs(results_root, feeder, controllers, scales)
    _, date_of = series_dates(feeder, time_series)
    frames, missing, optional, fmpc_checked = [], {}, {}, []
    for c, seed, s, d in runs:
        ep = pd.read_csv(d / 'episode_summary.csv')
        if c == 'forecast_mpc':
            bad = fmpc_backend_problems(d, ep, fmpc_backend)
            if bad:
                raise ValueError(f'F-MPC folder is not the {fmpc_backend} result: ' + '; '.join(bad))
            fmpc_checked.append(str(d))
        if 'controller' in ep and set(ep.controller.dropna().astype(str)) - {c}:
            raise ValueError(f'{d}: episode_summary controller {set(ep.controller)} is not {c}')
        if 'load_scaling_factor' in ep and not np.allclose(ep.load_scaling_factor.astype(float), s):
            raise ValueError(f'{d}: load_scaling_factor column differs from the folder scale {s}')
        if c in RL and 'run_id' in ep and not (ep.run_id.astype(float) == seed).all():
            raise ValueError(f'{d}: run_id differs from seed {seed}')
        out = pd.DataFrame(dict(controller=c, seed=np.nan if seed is None else float(seed), scale=s,
                                topology_case=ep.topology_case.astype(str), start_index=ep.start_index.astype(int)))
        for col in columns:
            if col in ep:
                out[col] = pd.to_numeric(ep[col], errors='coerce')
            else:
                out[col] = np.nan
                if col in OPTIONAL_METRIC_COLUMNS.get(c, ()):
                    optional.setdefault(col, {}).setdefault(c, []).append(str(d))     # not estimable, not an error
                else:
                    missing.setdefault(col, []).append(str(d))
        dup = out.duplicated(['topology_case', 'start_index'])
        if dup.any():
            raise ValueError(f'{d}: repeated (topology, start_index) rows; refusing to pair')
        out['source_dir'] = str(d)
        frames.append(out)
    if not frames:
        raise SystemExit(f'no finished runs under {results_root}/{feeder}')
    table = pd.concat(frames, ignore_index=True)
    table['date'] = date_of(table.start_index)
    unknown = sorted(set(table.topology_case) - set(TOPOLOGIES[feeder]))
    if unknown:
        raise ValueError(f'unknown topologies {unknown}')
    return table, dict(runs=[str(r[3]) for r in runs], excluded=excluded, missing_columns=missing,
                       unavailable_optional_columns=optional, fmpc_backend=fmpc_backend,
                       fmpc_backend_verified=fmpc_checked)


TEST_DAYS = 31


def coverage_problems(units, feeder, scales, controllers, info=None):
    """Every expected (controller, load factor) with its seeds (3 for RL, 1 otherwise), every maintained topology
    of the feeder and all 31 December days; plus metric columns missing from any run. [] when complete."""
    problems = []
    for c in controllers:
        for s in scales:
            u = units[(units.controller == c) & (units.scale == s)]
            if u.empty:
                problems.append(f'{c} x{s}: no finished run')
                continue
            seeds = sorted(int(k) for k in u.seed.dropna().unique())
            if c in RL and seeds != list(SEEDS):
                problems.append(f'{c} x{s}: seeds {seeds}, expected {list(SEEDS)}')
            for k, g in (u.groupby('seed') if c in RL else [(None, u)]):
                tps = sorted(set(TOPOLOGIES[feeder]) - set(g.topology_case))
                if tps:
                    problems.append(f'{c} x{s} seed {k}: topologies missing {tps}')
                days = g.groupby('topology_case').start_index.nunique()
                short = days[days != TEST_DAYS]
                if len(short):
                    problems.append(f'{c} x{s} seed {k}: days per topology {short.to_dict()} (expected {TEST_DAYS})')
    for col, dirs in ((info or {}).get('missing_columns') or {}).items():
        problems.append(f'column {col} missing in {len(dirs)} run(s), e.g. {dirs[0]}')
    return problems


def json_list(values, n):
    """Decode a column of JSON lists (or NaN) into an [rows, n] float matrix."""
    out = np.full((len(values), n), np.nan)
    for i, v in enumerate(values):
        if isinstance(v, str) and v.startswith('['):
            a = np.asarray(json.loads(v), dtype=float).reshape(-1)
            if a.size == n:
                out[i] = a
    return out


def environment_info(threads=None):
    """Hardware / software description recorded with timing results."""
    import os
    info = dict(host=platform.node(), platform=platform.platform(), python=platform.python_version(),
                processor=platform.processor(), cpu_count=os.cpu_count(),
                affinity=len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None,
                omp_threads=os.environ.get('OMP_NUM_THREADS'), mkl_threads=os.environ.get('MKL_NUM_THREADS'),
                openblas_threads=os.environ.get('OPENBLAS_NUM_THREADS'),
                cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'), requested_threads=threads,
                slurm_job=os.environ.get('SLURM_JOB_ID'), slurm_cpus=os.environ.get('SLURM_CPUS_PER_TASK'),
                slurm_partition=os.environ.get('SLURM_JOB_PARTITION'))
    try:
        for line in Path('/proc/cpuinfo').read_text().splitlines():
            if line.lower().startswith('model name'):
                info['cpu_model'] = line.split(':', 1)[1].strip()
                break
    except OSError:
        info['cpu_model'] = platform.processor() or None
    for name in ('numpy', 'scipy', 'pandas', 'torch'):
        try:
            info[f'{name}_version'] = __import__(name).__version__
        except ImportError:
            info[f'{name}_version'] = None
    try:
        import torch
        info.update(torch_threads=torch.get_num_threads(), torch_cuda_available=bool(torch.cuda.is_available()))
    except ImportError:
        pass
    return info


def selected_checkpoint(results_root, feeder, key, seed):
    """(checkpoint path, protocol, training record) of the frozen selection used by the authoritative test run."""
    from configs.experiments import run_block
    from evaluation.evaluate import EvaluationSpec, training_record
    from utility.evaluation_protocol import authorize, existing_selections, selection_key_dir
    layout = RunLayout(results_root)
    records = existing_selections(selection_key_dir(layout.selections_dir(feeder), feeder, layout.label(key),
                                                    layout.run_key(seed)))
    if not records:
        raise SystemExit(f'no frozen selection for {feeder}/{key}/seed_{seed}')
    protocol = authorize(mode='test', final=True, run_purpose='final_test', learned=True,
                         topologies=TOPOLOGIES[feeder], selection=str(records[-1]))
    spec = EvaluationSpec(feeder, key, seed, 'soc_feasible', layout, cpu=True)
    return protocol['selection']['checkpoint_path'], protocol, run_block(training_record(spec))


def rl_test_env(feeder, topology, out, protocol, record, seed, scale):
    """The authoritative test environment of a learned controller (evaluation.evaluate.test): same configuration,
    seed 42 + k, lambda_terminal from the training record, load factor `scale`."""
    from configs.experiments import topology_config
    from environments.marl_env import make_marl_env
    from utility.evaluation_protocol import protocol_config
    cfg = protocol_config(topology_config(feeder, topology, Path(out) / 'network_inputs', experiment='soc_feasible'),
                          protocol)
    return make_marl_env(cfg, 'test', 42 + int(seed), record['lambda_terminal'], scale)
