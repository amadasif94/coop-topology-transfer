"""Unit tests for the IEEE33 pure zero-action reference (no checkpoints, no job map).

    python -m unittest experiments.load_stress.test_pure_zero_ieee33 -v

The last test builds a real IEEE33 TP1 environment and is skipped unless the local replica time series exists
(audit_tmp/replica, patched in memory only).
"""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from configs.ieee33_config import IEEE33_CONFIG
from experiments.load_stress import evaluate_pure_zero_ieee33 as P
from experiments.load_stress import summarize_pure_zero_ieee33 as S
from experiments.load_stress import validate_pure_zero_ieee33 as V
from experiments.load_stress.evaluate_load_stress_ieee33 import StressError, run_dir as stress_run_dir
from experiments.load_stress.pure_zero_env import check_only_backup_disabled, pure_zero_config
from experiments.load_stress.stress_env import stress_config
from experiments.no_shared_kpi.protocol import ROOT

REPLICA = ROOT / 'audit_tmp/replica/data/time_series/processed_15min_smartds.csv'


def steps(n_agents=5, power=0.0, rows=4, **extra):
    d = dict(record_type='dispatch_attempt', backup_activated=False, backup_direction='unchanged', backup_pf_evaluations=1)
    for k in ('requested_action', 'accepted_action', 'dess_power_kw'):
        for i in range(n_agents):
            d[f'{k}_agent_{i}'] = power
    d.update(extra)
    return pd.DataFrame([d] * rows)


class Config(unittest.TestCase):
    def test_only_backup_settings_change(self):
        before = deepcopy(IEEE33_CONFIG)
        cfg = pure_zero_config(IEEE33_CONFIG, 1.2)
        self.assertEqual(IEEE33_CONFIG, before)
        self.assertEqual((cfg['backup']['backup_mode'], cfg['constraints']['handling']), ('legacy_scaling', 'accept_unconstrained'))
        self.assertEqual(cfg['data_scaling']['load_scaling_factor'], 1.2)
        self.assertEqual(check_only_backup_disabled(cfg, stress_config(IEEE33_CONFIG, 1.2)),
                         [['backup', 'backup_mode'], ['constraints', 'handling']])
        cfg['doe']['doe_enabled'] = True
        with self.assertRaisesRegex(ValueError, 'beyond'):
            check_only_backup_disabled(cfg, stress_config(IEEE33_CONFIG, 1.2))

    def test_control_comparison(self):
        control = dict(topology='TP1', primary_marl=True, configuration_sha256='a', backup_mode='directional_incremental',
                       backup=dict(backup_mode='directional_incremental', pf_budget=20), network={'s': 1})
        pure = dict(control, primary_marl=False, configuration_sha256='b', backup_mode='legacy_scaling',
                    backup=dict(backup_mode='legacy_scaling', pf_budget=20))
        self.assertEqual(P.compare_to_control(pure, control)['backup_differences'], ['backup_mode'])
        with self.assertRaises(P.PureZeroError):
            P.compare_to_control(dict(pure, network={'s': 2}), control)
        with self.assertRaises(P.PureZeroError):
            P.compare_to_control(dict(pure, backup=dict(backup_mode='legacy_scaling', pf_budget=10)), control)


class Guards(unittest.TestCase):
    def test_step_checks(self):
        self.assertEqual(P.check_steps(steps(), 5)['dispatch_steps'], 4)
        for bad in (steps(power=0.1), steps(backup_activated=True), steps(backup_direction='legacy_scaling'),
                    steps(backup_pf_evaluations=2)):
            with self.assertRaises(P.PureZeroError):
                P.check_steps(bad, 5)

    def test_output_root_and_arguments(self):
        for bad in ('results/load_stress/ieee33/zero_action', 'results/load_stress/ieee33/zero_action/run_0/load_x110',
                    'results/load_stress/ieee33', 'results/ablations/ieee33/x'):
            with self.assertRaises(StressError):
                P.check_output_root(bad, [])
        P.check_output_root(P.DEFAULT_OUTPUT_ROOT, [])
        self.assertEqual(P.run_dir('r', 1.1, pilot=True).name, 'load_x110_pilot')
        with self.assertRaises(SystemExit):
            P.parse_args(['--scale', '1.15', '--job-map', 'j.json'])


def write_run(d, scale, feasible, cost, extra=None, pure=True):
    d.mkdir(parents=True)
    starts = [0, 96]
    manifest = dict(status='finished')
    manifest['pure_zero' if pure else 'load_stress'] = dict(load_scaling_factor=scale, pilot=False,
                                                            episodes=dict(planned=2, complete=2))
    (d / 'run_manifest.json').write_text(json.dumps(manifest))
    pd.DataFrame(dict(topology_case='TP1', start_index=starts, total_energy_cost=cost, adjusted_energy_cost=cost,
                      feasible_rate=float(np.mean(feasible)), complete=True)).to_csv(d / 'episode_summary.csv', index=False)
    rows = []
    for s in starts:
        for step, f in enumerate(feasible):
            row = dict(topology_case='TP1', start_index=s, step=step, record_type='dispatch_attempt', converged=True,
                       feasible=f if pure else True, backup_activated=not f, executed_network_violation=not f if pure else False,
                       energy_cost=cost / len(feasible), **{c: (0.0 if f else 1.5) for c in V.NUMERIC},
                       **{c: not f for c in V.FLAGS})
            rows.append(dict(row, **(extra or {})))
    pd.DataFrame(rows).to_csv(d / 'step_metrics.csv', index=False)


class CrossCheckAndSummary(unittest.TestCase):
    def test_cross_check(self):
        with tempfile.TemporaryDirectory() as d:
            pure, ref = Path(d) / 'pure', Path(d) / 'ref'
            write_run(pure, 1.3, [True, False, True], 30.0)
            write_run(ref, 1.3, [True, False, True], 28.0, pure=False)
            V.main(['--pure-dir', str(pure), '--reference-dir', str(ref)])
            self.assertEqual(json.loads((pure / 'cross_check_report.json').read_text())['status'], 'PASS')
            with self.assertRaises(SystemExit):                    # never overwrites
                V.main(['--pure-dir', str(pure), '--reference-dir', str(ref)])
            bad = Path(d) / 'bad'
            write_run(bad, 1.3, [True, True, True], 30.0)
            with self.assertRaises(SystemExit):
                V.main(['--pure-dir', str(bad), '--reference-dir', str(ref)])
            self.assertEqual(json.loads((bad / 'cross_check_report.json').read_text())['status'], 'FAIL')
            none = Path(d) / 'none'
            write_run(none, 1.3, [True], 1.0)
            V.main(['--pure-dir', str(none), '--reference-dir', str(Path(d) / 'absent')])
            self.assertEqual(json.loads((none / 'cross_check_report.json').read_text())['status'], 'NOT_AVAILABLE')

    def test_summary(self):
        with tempfile.TemporaryDirectory() as d:
            root, zb = Path(d) / 'pure', Path(d) / 'stress'
            for s in (1.0, 1.3):
                write_run(P.run_dir(root, s), s, [True, False], 10.0 * s)
                write_run(stress_run_dir(zb, 'zero', 0, s), s, [True, False], 10.0 * s - 1.0, pure=False)
            S.main(['--root', str(root), '--output', str(Path(d) / 'out'), '--zero-backup-root', str(zb), '--bootstrap', '100'])
            absolute = pd.read_csv(Path(d) / 'out' / 'absolute_by_scale.csv')
            row = absolute.query("scale == 1.3 and metric == 'violating_step_rate'").iloc[0]
            self.assertAlmostEqual(row['mean'], 0.5)
            delta = pd.read_csv(Path(d) / 'out' / 'zero_backup_minus_pure_summary.csv')
            self.assertAlmostEqual(delta.query("scale == 1.0 and metric == 'total_energy_cost'").iloc[0].mean_delta, -1.0)
            inc = json.loads((Path(d) / 'out' / 'inclusion.json').read_text())
            self.assertEqual((inc['missing_scales'], inc['episodes'], inc['failed_days']), ([1.1, 1.2], 4, 0))


@unittest.skipUnless(REPLICA.is_file(), 'local replica time series not available')
class RealEnvironment(unittest.TestCase):
    def test_pure_zero_dispatch_and_identical_initial_power_flow(self):
        from experiments.load_stress.pure_zero_env import make_pure_zero_env
        from experiments.load_stress.stress_env import make_stress_env
        from experiments.no_gnn_maddpg.run_tp_eval_mlp import evaluate_one_episode
        from experiments.topology_generalization.topology_utils import build_topology_config
        with tempfile.TemporaryDirectory() as d:
            cfg = build_topology_config(case_name='TP1', output_root=Path(d), validate=True)
            cfg['paths']['processed_time_series'] = REPLICA.resolve()        # local test only, in memory
            pure, control = make_pure_zero_env(cfg, 'val', 0, 1.3), make_stress_env(cfg, 'val', 0, 1.3)
            start = int(pure.episode_start_indices[0])
            fp = pd.DataFrame(evaluate_one_episode(pure, None, 'zero', 'TP1', 0, start, 0))
            fc = pd.DataFrame(evaluate_one_episode(control, None, 'zero', 'TP1', 0, start, 0))
        self.assertEqual(P.check_steps(fp, pure.num_agents)['dispatch_steps'], 96)
        table, coverage = V.compare(fp, fc, 1e-9)
        self.assertEqual(set(table.status), {'PASS'}, table.to_string())
        self.assertEqual(coverage['pure_violating_steps'], coverage['reference_backup_activations'])
        self.assertEqual(int(fp['zero_battery_feasible'].isna().sum()), 0)


if __name__ == '__main__':
    unittest.main()
