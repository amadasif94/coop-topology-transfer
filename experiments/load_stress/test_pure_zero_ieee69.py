"""Unit tests for the IEEE69 pure zero-action reference (no checkpoints, no job map).

    python -m unittest experiments.load_stress.test_pure_zero_ieee69 -v

The last test builds a real IEEE69 TP10 environment pair and is skipped unless the local replica time series
exists (audit_tmp/replica, patched in memory only).
"""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from configs.ieee69_config import IEEE69_CONFIG
from experiments.load_stress import evaluate_pure_zero_ieee69 as P
from experiments.load_stress import summarize_pure_zero_ieee69 as S
from experiments.load_stress import validate_pure_zero_ieee69 as V
from experiments.load_stress.evaluate_load_stress_ieee33 import StressError
from experiments.load_stress.evaluate_load_stress_ieee69 import run_dir as controller_run_dir
from experiments.load_stress.pure_zero_env import check_only_backup_disabled, pure_zero_config
from experiments.load_stress.stress_env import stress_config
from experiments.load_stress.test_pure_zero_ieee33 import write_run
from experiments.load_stress.validate_pure_zero_ieee33 import FLAGS, NUMERIC
from experiments.no_shared_kpi.protocol import ROOT, sha

REPLICA = ROOT / 'audit_tmp/replica/data/time_series/processed_15min_smartds_ieee69.csv'


class Config(unittest.TestCase):
    def test_only_backup_settings_change_and_doe_stays_off(self):
        before = deepcopy(IEEE69_CONFIG)
        self.assertTrue(IEEE69_CONFIG['doe']['doe_enabled'])            # raw IEEE69 config: DOE on
        cfg = pure_zero_config(IEEE69_CONFIG, 1.3)
        self.assertEqual(IEEE69_CONFIG, before)
        self.assertFalse(cfg['doe']['doe_enabled'])                       # forced off by primary_config, as in A0
        self.assertEqual(check_only_backup_disabled(cfg, stress_config(IEEE69_CONFIG, 1.3)),
                         [['backup', 'backup_mode'], ['constraints', 'handling']])


class Guards(unittest.TestCase):
    def test_output_root_and_arguments(self):
        for bad in ('results/load_stress/ieee69/a0_original/run_0', 'results/load_stress/ieee69',
                    'results/load_stress/ieee33/pure_zero_no_backup', 'results/load_stress/elsewhere'):
            with self.assertRaises(StressError):
                P.check_output_root(ROOT / bad, [])
        P.check_output_root(ROOT / P.DEFAULT_OUTPUT_ROOT, [])
        self.assertEqual(P.run_dir('r', 1.3, smoke=True).name, 'load_x130_smoke')
        with self.assertRaises(SystemExit):
            P.parse_args(['--scale', '1.1', '--smoke', '--job-map', 'j.json'])
        self.assertTrue(P.parse_args(['--scale', '1.3', '--smoke', '--job-map', 'j.json']).smoke)

    def test_smoke_status(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(P.smoke_status(d)['status'], 'NOT_PERFORMED')
            s = P.run_dir(d, 1.3, smoke=True)
            s.mkdir(parents=True)
            (s / 'run_manifest.json').write_text('{}')
            report = dict(status='PASS', evaluation_manifest_sha256=sha(s / 'run_manifest.json'))
            (s / 'smoke_check.json').write_text(json.dumps(report))
            self.assertEqual(P.smoke_status(d)['status'], 'PASS')
            (s / 'smoke_check.json').write_text(json.dumps(dict(report, status='FAIL')))
            with self.assertRaises(P.PureZeroError):
                P.smoke_status(d)
            (s / 'smoke_check.json').write_text(json.dumps(report))
            (s / 'run_manifest.json').write_text('{"changed": 1}')
            with self.assertRaises(P.PureZeroError):
                P.smoke_status(d)


def write_pure(d, references, feasible):
    write_run(d, 1.0, feasible, 10.0)
    m = json.loads((d / 'run_manifest.json').read_text())
    m['pure_zero'].update(feeder='IEEE69', smoke=False, a0_references=references)
    (d / 'run_manifest.json').write_text(json.dumps(m))


class CrossCheckAndSummary(unittest.TestCase):
    def a0_final(self, root, feasible, include_zero=True):
        root.mkdir(parents=True)
        (root / 'run_manifest.json').write_text('{}')
        rows = []
        for policy in ['mlp_maddpg'] + (['zero'] if include_zero else []):
            for s in (0, 96):
                for step, f in enumerate(feasible):
                    rows.append(dict(topology_case='TP1', policy=policy, start_index=s, step=step, record_type='dispatch_attempt',
                                     feasible=True, backup_activated=not f, **{c: (0.0 if f else 1.5) for c in NUMERIC},
                                     **{c: not f for c in FLAGS}))
        pd.DataFrame(rows).to_csv(root / 'step_metrics.csv', index=False)
        return {tp: dict(path=str(root / 'run_manifest.json'), sha256=sha(root / 'run_manifest.json')) for tp in P.TOPOLOGIES}

    def test_cross_check(self):
        with tempfile.TemporaryDirectory() as d:
            refs = self.a0_final(Path(d) / 'a0', [True, False, True])
            pure = Path(d) / 'pure'
            write_pure(pure, refs, [True, False, True])
            V.main(['--pure-dir', str(pure)])
            report = json.loads((pure / 'cross_check_report.json').read_text())
            self.assertEqual(report['status'], 'PASS_PARTIAL')              # only TP1 has zero rows here
            self.assertEqual(len(report['not_observable_topologies']), 9)
            bad = Path(d) / 'bad'
            write_pure(bad, refs, [True, True, True])
            with self.assertRaises(SystemExit):
                V.main(['--pure-dir', str(bad)])
            none_refs = self.a0_final(Path(d) / 'a0_nozero', [True], include_zero=False)
            none = Path(d) / 'none'
            write_pure(none, none_refs, [True])
            V.main(['--pure-dir', str(none)])
            self.assertEqual(json.loads((none / 'cross_check_report.json').read_text())['status'], 'NOT_AVAILABLE')

    def test_summary(self):
        with tempfile.TemporaryDirectory() as d:
            root, controllers = Path(d) / 'pure', Path(d) / 'stress'
            write_run(P.run_dir(root, 1.3), 1.3, [True, False], 20.0)
            for r in (0, 1):
                write_run(controller_run_dir(controllers, 'a0', r, 1.3), 1.3, [True, True], 15.0 + r, pure=False)
            S.main(['--root', str(root), '--output', str(Path(d) / 'out'), '--controller-root', str(controllers),
                    '--bootstrap', '100'])
            absolute = pd.read_csv(Path(d) / 'out' / 'absolute_by_scale.csv')
            self.assertAlmostEqual(absolute.query("metric == 'violating_step_rate'").iloc[0]['mean'], 0.5)
            by_model = pd.read_csv(Path(d) / 'out' / 'controllers_minus_pure_by_model.csv')
            row = by_model.query("model == 'a0' and metric == 'total_energy_cost'").iloc[0]
            self.assertAlmostEqual(row.mean_delta, -4.5)
            inc = json.loads((Path(d) / 'out' / 'inclusion.json').read_text())
            self.assertEqual((inc['missing_scales'], inc['episodes'], inc['smoke_check']), ([1.0, 1.1, 1.2], 2, 'NOT_PERFORMED'))


@unittest.skipUnless(REPLICA.is_file(), 'local replica IEEE69 time series not available')
class RealEnvironment(unittest.TestCase):
    def test_tp10_pure_zero_dispatch_and_identical_initial_power_flow(self):
        from experiments.load_stress.evaluate_pure_zero_ieee33 import check_steps, compare_to_control
        from experiments.load_stress.pure_zero_env_ieee69 import make_pure_zero_env_ieee69
        from experiments.load_stress.stress_env_ieee69 import make_stress_env_ieee69
        from experiments.load_stress.validate_pure_zero_ieee33 import compare
        from experiments.no_gnn_maddpg.run_tp_eval_mlp_ieee69 import evaluate_one_episode
        from experiments.no_shared_kpi.ieee69.runner import RunManifest, energy_rows
        from experiments.topology_generalization.topology_utils_ieee69 import build_topology_config
        with tempfile.TemporaryDirectory() as d:
            cfg = build_topology_config(case_name='TP10', output_root=Path(d) / 'net', validate=True)
            cfg['paths']['processed_time_series'] = REPLICA.resolve()        # local test only, in memory
            pure, control = make_pure_zero_env_ieee69(cfg, 'val', 0, 1.3), make_stress_env_ieee69(cfg, 'val', 0, 1.3)
            run = RunManifest(Path(d) / 'run', 'ZERO', 0, 'val', None, {})
            run.add_environment(control, topology='TP10', scenario='stress')
            control_record = run.data['environments'].pop()
            run.add_environment(pure, topology='TP10', primary=False, scenario='stress')
            diff = compare_to_control(run.data['environments'][-1], control_record)
            start = int(pure.episode_start_indices[0])
            fp = pd.DataFrame(evaluate_one_episode(pure, None, 'zero', 'TP10', 0, start, 0))
            fc = pd.DataFrame(evaluate_one_episode(control, None, 'zero', 'TP10', 0, start, 0))
            self.assertEqual(len(energy_rows(fp, pure)), 1)
        self.assertEqual(diff['backup_differences'], ['backup_mode'])
        self.assertEqual(check_steps(fp, 9)['dispatch_steps'], 96)
        table, coverage = compare(fp, fc, 1e-9)
        self.assertEqual(set(table.status), {'PASS'}, table.to_string())
        self.assertEqual(coverage['pure_violating_steps'], coverage['reference_backup_activations'])
        self.assertEqual(int(fp['zero_battery_feasible'].isna().sum()), 0)
        print('TP10 1.3x replica day: pure violating steps', coverage['pure_violating_steps'],
              'pure cost', round(float(fp.energy_cost.sum()), 3), 'backup cost', round(float(fc.energy_cost.sum()), 3))


if __name__ == '__main__':
    unittest.main()
