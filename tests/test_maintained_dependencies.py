"""The maintained pipeline must not depend on the historical B3/B4 code, the compatibility shims, or results/b3|b4.

    python -m unittest tests.test_maintained_dependencies -v

The historical directories experiments/b3, experiments/b4 and experiments/b3_reporting may be absent (deleted from the
maintained repository) or present; every test here passes either way and never requires them.

StaticImportClosure  AST import closure (module-level AND function-level imports, relative imports, `from pkg import
                     submodule`, parent-package __init__ files) of every maintained entry point and library module
                     (training, selection/test/zero, summary, PF-MPC, F-MPC incl. Gurobi, PQ-RBC, reporting and the
                     final paper analyses). Fails on any IMPORTED NAME of a historical module (experiments.b3*,
                     experiments.b4*, experiments.b3_reporting*) or compatibility shim, whether or not that module's
                     file exists, any string naming such a module, and any results/b3 or results/b4 path literal.
RuntimeIsolation     copies the repository code WITHOUT experiments/b3, experiments/b4, experiments/b3_reporting and the
                     four shim files into a temporary directory (the real repository is never changed), then in a
                     separate `python -I` process with an import blocker for those names runs a
                     lightweight smoke of the maintained entry points: training.train.main (205 steps), evaluation.
                     evaluate select / test / zero (one validation day each; no December test data), evaluation.
                     summarize, and --help of evaluation.oracle.run / evaluation.reporting.{evaluate,compare}. It finally
                     checks that no module was loaded from the real repository and no historical module was imported.
"""
import ast
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest

from tests.support import HAS_PYG, ROOT, series

ENTRY_POINTS = ('training.train', 'evaluation.evaluate', 'evaluation.summarize', 'evaluation.zero',
                'evaluation.oracle.run', 'evaluation.reporting.evaluate', 'evaluation.reporting.compare',
                'evaluation.forecast_mpc.train', 'evaluation.forecast_mpc.generate', 'evaluation.forecast_mpc.validate',
                'evaluation.forecast_mpc.run', 'evaluation.rule_based.pq_rbc',
                'evaluation.forecast_mpc.gurobi_run', 'evaluation.forecast_mpc.gurobi_benchmark',
                'evaluation.forecast_mpc.gurobi_manifest', 'evaluation.final_analyses.bootstrap',
                'evaluation.final_analyses.soc_diagnostics', 'evaluation.final_analyses.timing',
                'evaluation.final_analyses.action_replay', 'evaluation.final_analyses.preflight')
LIBRARY = ('configs.experiments', 'environments.marl_env', 'environments.local_reward_env',
           'environments.battery_feasibility', 'models.controllers', 'models.independent_local_agents',
           'training.controller_trainer', 'training.bounded_replay', 'training.feasibility_supervision',
           'utility.feasibility_metrics')
HISTORICAL_PACKAGES = ('experiments.b3', 'experiments.b4', 'experiments.b3_reporting')
SHIMS = ('environments.feasibility_env', 'models.feasibility_agents', 'training.feasibility_trainer',
         'utility.experiment_setup')
HISTORICAL_DIRS = ('experiments/b3', 'experiments/b4', 'experiments/b3_reporting')
RESULTS_LITERAL = re.compile(r'results[\\/]+b[34](?![0-9A-Za-z_])')


def forbidden(module):
    return module in SHIMS or any(module == p or module.startswith(p + '.') for p in HISTORICAL_PACKAGES)


def module_file(module, root=ROOT):
    base = Path(root, *module.split('.'))
    for candidate in (base.with_suffix('.py'), base / '__init__.py'):
        if candidate.is_file():
            return candidate
    return None


def imports_of(module, path):
    """Every module name `path` may import (absolute names; local and third-party)."""
    tree = ast.parse(path.read_text(encoding='utf-8'), str(path))
    package = module if path.name == '__init__.py' else module.rpartition('.')[0]
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                parts = package.split('.')
                base = '.'.join(parts[:len(parts) - node.level + 1])
                target = f'{base}.{node.module}' if node.module else base
            else:
                target = node.module
            found.add(target)
            found.update(f'{target}.{alias.name}' for alias in node.names)       # `from pkg import submodule`
    return found


def closure(roots, root=ROOT):
    """({module: file} of every repository module reachable from `roots`, {imported module: importers},
    {imported NAME: importers}). The names include imports whose files do not exist (e.g. a deleted historical
    package), so a reference to an absent module is still visible to the checks."""
    seen, edges, names, todo = {}, {}, {}, list(roots)
    while todo:
        module = todo.pop()
        if module in seen:
            continue
        path = module_file(module, root)
        if path is None:
            continue
        seen[module] = path
        parts = module.split('.')
        imported = imports_of(module, path)
        for name in imported:
            names.setdefault(name, set()).add(module)
        targets = imported | {'.'.join(parts[:i]) for i in range(1, len(parts))}   # parent __init__
        for target in targets:
            if module_file(target, root) is not None:
                edges.setdefault(target, set()).add(module)
                todo.append(target)
    return seen, edges, names


def docstring_nodes(tree):
    ids = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
                ids.add(id(first.value))
    return ids


class StaticImportClosure(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.modules, cls.edges, cls.names = closure(ENTRY_POINTS + LIBRARY)

    def test_entry_points_and_library_exist(self):
        for module in ENTRY_POINTS + LIBRARY:
            self.assertIn(module, self.modules, module)

    def test_no_historical_module_or_shim_in_the_closure(self):
        bad = {m: sorted(self.edges.get(m, ())) for m in self.modules if forbidden(m)}
        self.assertEqual(bad, {}, 'maintained code imports historical modules or shims (module: importers)')

    def test_no_import_of_a_historical_or_shim_name(self):
        """Also catches imports of historical packages that are no longer on disk (no file, so not in the closure)."""
        bad = {n: sorted(importers) for n, importers in self.names.items() if forbidden(n)}
        self.assertEqual(bad, {}, 'maintained code imports historical modules or shims (name: importers)')

    def test_no_dynamic_reference_to_historical_modules(self):
        bad = []
        for module, path in self.modules.items():
            tree = ast.parse(path.read_text(encoding='utf-8'))
            docs = docstring_nodes(tree)
            for node in ast.walk(tree):
                if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docs:
                    name = node.value.strip()
                    if re.fullmatch(r'[A-Za-z_][\w.]*', name) and forbidden(name):
                        bad.append(f'{module}:{node.lineno} {name!r}')
        self.assertEqual(bad, [])

    def test_no_results_b3_or_b4_path_literal(self):
        bad = []
        for module, path in self.modules.items():
            tree = ast.parse(path.read_text(encoding='utf-8'))
            docs = docstring_nodes(tree)
            for node in ast.walk(tree):
                if (isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docs
                        and RESULTS_LITERAL.search(node.value)):
                    bad.append(f'{module}:{node.lineno} {node.value!r}')
        self.assertEqual(bad, [])

    def test_maintained_source_inventory_matches_files(self):
        from configs.experiments import MAINTAINED_SOURCES, source_inventory
        files = {p.resolve() for p in self.modules.values()}
        for rel in MAINTAINED_SOURCES:
            self.assertTrue((ROOT / rel).is_file(), rel)
            self.assertIn((ROOT / rel).resolve(), files, rel)
        self.assertEqual(set(source_inventory()), set(MAINTAINED_SOURCES))

    def test_maintained_default_protocol_and_output_roots(self):
        from configs.experiments import MAINTAINED_PROTOCOL, maintained_output_root, protocol_id
        from training.train import parse_args
        self.assertEqual(MAINTAINED_PROTOCOL, 'soc_feasible')
        args = parse_args(['--feeder', 'ieee33', '--controller', 'mlp', '--variant', 'A0', '--seed', '0'])
        self.assertEqual((args.protocol, args.experiment, Path(args.root)), ('soc_feasible', 'soc_feasible',
                                                                               Path('results/soc_feasible')))
        self.assertEqual(protocol_id('rated_feasibility_supervised'), 'b4')
        self.assertEqual(protocol_id('rated_quadratic_terminal'), 'b3')
        for bad in ('results/b3', 'results/b4/x', 'results\\b4'):
            with self.assertRaises(ValueError):
                maintained_output_root('soc_feasible', bad)
        with self.assertRaises(SystemExit):        # MLP needs a variant; GNNs take none
            parse_args(['--feeder', 'ieee33', '--controller', 'mlp', '--seed', '0'])
        with self.assertRaises(SystemExit):
            parse_args(['--feeder', 'ieee33', '--controller', 'gcn', '--variant', 'A1', '--seed', '0'])

    def test_scanner_detects_historical_dependencies(self):
        """Control (self-contained, needs no historical directory): a throw-away package that imports deleted
        historical modules, a shim and a results/b4 path must be flagged by the same scanner, so a pass is not
        vacuous."""
        with tempfile.TemporaryDirectory(prefix='dependency-scanner-') as tmp:
            pkg = Path(tmp) / 'probe'
            pkg.mkdir()
            (pkg / '__init__.py').write_text('')
            (pkg / 'entry.py').write_text(textwrap.dedent('''
                from experiments.b3.train import main
                import experiments.b4.oracle.run
                def late():
                    from experiments.b3_reporting import metrics
                    from utility import experiment_setup
                    return 'results/b4/x'
            '''))
            modules, _, names = closure(('probe.entry',), root=Path(tmp))
            self.assertIn('probe.entry', modules)
            flagged = {n for n in names if forbidden(n)}
            for name in ('experiments.b3.train', 'experiments.b4.oracle.run', 'experiments.b3_reporting.metrics',
                         'utility.experiment_setup'):
                self.assertIn(name, flagged)
            literals = [n.value for n in ast.walk(ast.parse((pkg / 'entry.py').read_text()))
                        if isinstance(n, ast.Constant) and isinstance(n.value, str)]
            self.assertTrue(any(RESULTS_LITERAL.search(v) for v in literals))

    def test_no_maintained_module_lives_in_a_historical_directory(self):
        """Holds whether the historical directories are present or deleted."""
        inside = sorted(m for m, p in self.modules.items()
                        if any(Path(p).resolve().is_relative_to((ROOT / rel).resolve()) for rel in HISTORICAL_DIRS))
        self.assertEqual(inside, [])


COPY_DIRS = ('configs', 'environments', 'evaluation', 'experiments', 'models', 'scripts', 'training', 'utility', 'data')
SKIP_DIRS = {Path(p) for p in HISTORICAL_DIRS} | {Path('data/processed')}
SHIM_FILES = [Path(*m.split('.')).with_suffix('.py') for m in SHIMS]

DRIVER = r'''
import importlib.abc, json, sys
from pathlib import Path
COPY, REAL, OUT, S33 = (Path(p) for p in sys.argv[1:5])
GNN = sys.argv[5] == '1'
sys.path.insert(0, str(COPY))
HISTORICAL = ('experiments.b3', 'experiments.b4', 'experiments.b3_reporting')
SHIMS = %(shims)r
def forbidden(name):
    return name in SHIMS or any(name == p or name.startswith(p + '.') for p in HISTORICAL)
class Blocker(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if forbidden(name):
            raise ImportError(f'BLOCKED historical import: {name}')
        return None
sys.meta_path.insert(0, Blocker())
for d in ('experiments/b3', 'experiments/b4', 'experiments/b3_reporting'):
    assert not (COPY / d).exists(), d

from unittest.mock import patch
from configs.ieee33_config import IEEE33_CONFIG
IEEE33_CONFIG['paths']['processed_time_series'] = S33
from configs.experiments import RunLayout, feeder_config
from environments.marl_env import make_marl_env
import evaluation.evaluate as evaluate
from evaluation.summarize import summarize
from training.train import main as train
from utility.evaluation_protocol import authorize as real_authorize

layout = RunLayout(OUT)
val_starts = list(make_marl_env(feeder_config('ieee33', experiment='soc_feasible'), 'val', 42, 0.0).episode_start_indices[:31])
real_run = evaluate.run_episodes
def one_day(feeder, env, agent, policy, tp, plan):
    return real_run(feeder, env, agent, policy, tp, list(plan)[:1])
def validation_days(env, episodes, seed, protocol):
    return val_starts, [seed + i for i in range(31)], None
controllers = [('mlp', 'A0', 'mlp_a0')] + ([('gcn', None, 'gcn')] if GNN else [])
for family, variant, key in controllers:
    argv = ['--feeder', 'ieee33', '--controller', family, '--seed', '0', '--output-root', str(OUT), '--time-series', str(S33),
            '--total-steps', '205', '--warmup-steps', '200', '--buffer-size', '400', '--batch-size', '16',
            '--eval-every', '1000000000', '--save-every', '205', '--cpu'] + (['--variant', variant] if variant else [])
    out = train(argv)
    checkpoint = out / 'checkpoints' / 'maddpg_step_205.pt'
    spec = evaluate.EvaluationSpec('ieee33', key, 0, 'soc_feasible', layout, cpu=True)
    with patch.object(evaluate, 'run_episodes', one_day), patch('scripts.select_checkpoint.main', return_value=None):
        evaluate.select(spec)
    def authorize(**kw):
        protocol = real_authorize(mode='test', final=True, run_purpose='final_test', learned=False, topologies=kw['topologies'])
        protocol['selection'] = dict(checkpoint_path=str(checkpoint), selection_sha256='isolation-test')
        return protocol
    with patch.object(evaluate, 'run_episodes', one_day), patch.object(evaluate, 'authorize', side_effect=authorize), \
            patch.object(evaluate, 'existing_selections', return_value=[Path('isolation-test')]), \
            patch.object(evaluate, 'episode_plan', side_effect=validation_days), \
            patch.dict(evaluate.TOPOLOGIES, {'ieee33': ['TP1', 'TP2']}):
        evaluate.test(spec, [1.0])
with patch.object(evaluate, 'run_episodes', one_day), patch.object(evaluate, 'episode_plan', side_effect=validation_days), \
        patch.dict(evaluate.TOPOLOGIES, {'ieee33': ['TP1']}):
    evaluate.zero('ieee33', 'soc_feasible', layout, [1.0])
summary = summarize('ieee33', layout)
import evaluation.oracle.run, evaluation.reporting.evaluate, evaluation.reporting.compare  # noqa: E401
for m in (evaluation.oracle.run, evaluation.reporting.evaluate, evaluation.reporting.compare):
    try:
        m.main(['--help'])
    except SystemExit as exc:
        assert exc.code == 0, m.__name__
loaded = {n: getattr(m, '__file__', None) for n, m in list(sys.modules.items())}
outside = sorted(n for n, f in loaded.items() if f and Path(f).resolve().is_relative_to(REAL.resolve())
                 and not Path(f).resolve().is_relative_to(COPY.resolve()))
historical = sorted(n for n in loaded if forbidden(n))
print('ISOLATION_RESULT ' + json.dumps(dict(outside=outside, historical=historical, summary=str(summary),
                                            controllers=[c[2] for c in controllers])))
''' % dict(shims=SHIMS)


@unittest.skipUnless(series('ieee33'), 'no processed time series')
class RuntimeIsolation(unittest.TestCase):
    def test_maintained_entry_points_run_without_historical_code(self):
        with tempfile.TemporaryDirectory(prefix='maintained-isolation-') as tmp:
            copy, out = Path(tmp) / 'repo', Path(tmp) / 'results' / 'soc_feasible'
            for top in COPY_DIRS:
                for src in (ROOT / top).rglob('*'):
                    rel = src.relative_to(ROOT)
                    if (src.is_dir() or '__pycache__' in rel.parts or
                            any(rel == s or s in rel.parents for s in SKIP_DIRS) or rel in SHIM_FILES):
                        continue
                    (copy / rel).parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, copy / rel)
            for rel in HISTORICAL_DIRS:                 # absent in the copy; the real repository is never changed
                self.assertFalse((copy / rel).exists())
            for rel in SHIM_FILES:
                self.assertFalse((copy / rel).exists())
            driver = Path(tmp) / 'driver.py'
            driver.write_text(textwrap.dedent(DRIVER))
            env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', TEMP=tmp, TMP=tmp, TMPDIR=tmp)
            env.pop('PYTHONPATH', None)
            proc = subprocess.run([sys.executable, '-I', str(driver), str(copy), str(ROOT), str(out), series('ieee33'),
                                   '1' if HAS_PYG else '0'], cwd=tmp, env=env, capture_output=True, text=True,
                                  timeout=3600)
            tail = '\n'.join((proc.stdout + proc.stderr).splitlines()[-40:])
            self.assertEqual(proc.returncode, 0, tail)
            line = next(l for l in proc.stdout.splitlines() if l.startswith('ISOLATION_RESULT '))
            result = json.loads(line.split(' ', 1)[1])
            self.assertEqual(result['outside'], [], 'modules loaded from the real repository')
            self.assertEqual(result['historical'], [])
            self.assertTrue((out / 'ieee33' / 'mlp' / 'A0' / 'seed_0' / 'test' / 'load_x100' / 'run_manifest.json').is_file())
            self.assertTrue((out / 'ieee33' / 'zero_action_no_backup' / 'load_x100' / 'run_manifest.json').is_file())
            self.assertTrue(Path(result['summary']).is_dir())
            for marker in ('TRAINING_DONE', 'TEST_DONE', 'ZERO_DONE'):
                self.assertIn(marker, proc.stdout)


if __name__ == '__main__':
    unittest.main()
