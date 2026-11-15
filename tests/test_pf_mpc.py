"""PF-MPC (evaluation.oracle.run, --replan never, direct-kW dispatch). Validation days only (never December test data);
environment tests need data/time_series (RIT) or the local replica and are skipped without one.
Run: python -m unittest tests.test_pf_mpc -v
"""
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd

from tests.support import series


def lists(column):
    return np.array([json.loads(v) for v in column], dtype=float)


def make_env(feeder='ieee33', experiment='soc_feasible', scale=1.0):
    from configs.experiments import feeder_config
    from environments.marl_env import make_marl_env
    return make_marl_env(feeder_config(feeder, series(feeder), experiment), 'val', 42, 0.0, scale)


class Summarize(unittest.TestCase):
    """Synthetic result tree: PF-MPC is a single deterministic run, never averaged like seeds; smoke runs excluded."""

    def write_run(self, d, controller, starts, mapping):
        d.mkdir(parents=True)
        (d / 'run_manifest.json').write_text(json.dumps(dict(status='finished')))
        pd.DataFrame(dict(controller=controller, load_scaling_factor=1.0, topology_case='TP1', run_id=np.nan,
                          start_index=starts, adjusted_energy_cost=[10.0 + i for i in range(len(starts))],
                          action_mapping=mapping, action_mapping_version=mapping)).to_csv(d / 'episode_summary.csv',
                                                                                        index=False)

    def test_single_run_reference_and_smoke_exclusion(self):
        from configs.experiments import RunLayout
        from evaluation.summarize import summarize
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.write_run(root / 'ieee33' / 'pf_mpc' / 'load_x100', 'pf_mpc', [0, 96, 192], 'direct_kw')
            self.write_run(root / 'ieee33' / 'pf_mpc' / 'load_x100_smoke1', 'pf_mpc', [0], 'direct_kw')
            out = summarize('ieee33', RunLayout(root), optimization_references=True)
            by = pd.read_csv(out / 'by_controller.csv')
            row = by[by.controller == 'pf_mpc'].iloc[0]
            self.assertEqual(row.runs, 1)
            self.assertTrue(np.isnan(row.adjusted_energy_cost_between_run_sd))
            self.assertAlmostEqual(row.adjusted_energy_cost_mean, 11.0)
            self.assertEqual(row.result_type, 'deterministic_optimization_single_run')
            inclusion = json.loads((out / 'inclusion.json').read_text())
            self.assertTrue(any('smoke' in e['dir'] for e in inclusion['excluded']))

    def test_repeated_reference_days_refused(self):
        from configs.experiments import RunLayout
        from evaluation.summarize import summarize
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.write_run(root / 'ieee33' / 'pf_mpc' / 'load_x100', 'pf_mpc', [0, 0], 'direct_kw')
            with self.assertRaisesRegex(SystemExit, 'single deterministic optimization'):
                summarize('ieee33', RunLayout(root), optimization_references=True)

    def test_historical_summary_ignores_pf_mpc(self):
        from configs.experiments import RunLayout
        from evaluation.summarize import summarize
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.write_run(root / 'ieee33' / 'pf_mpc' / 'load_x100', 'pf_mpc', [0], 'direct_kw')
            with self.assertRaisesRegex(SystemExit, 'No finished test/zero runs'):
                summarize('ieee33', RunLayout(root))           # default (historical shim): unchanged inputs


class Statistics(unittest.TestCase):
    def test_dispatch_statistics(self):
        from evaluation.oracle.run import dispatch_statistics
        dumps = lambda rows: [json.dumps(r) for r in rows]
        frame = pd.DataFrame(dict(topology_case='TP1', start_index=0, step=[0, 1], record_type='dispatch_attempt',
                                  marl_planned_power_kw=dumps([[10.0, -20.0], [5.0, 0.0]]),
                                  physical_limited_power_kw=dumps([[10.0, -20.0], [3.0, 0.0]]),
                                  final_battery_power_kw=dumps([[10.0, -20.0], [1.0, 0.0]]),
                                  final_network_feasible=[True, False], backup_activated=[False, True],
                                  episode_failed=[False, False]))
        r = dispatch_statistics(frame, 0.25).iloc[0]
        self.assertAlmostEqual(r.max_abs_planned_minus_executed_kw, 4.0)
        self.assertAlmostEqual(r.max_abs_planned_minus_limiter_kw, 2.0)
        self.assertAlmostEqual(r.max_abs_backup_change_kw, 2.0)
        self.assertEqual(r.steps_executed_differs_from_plan, 1)
        self.assertAlmostEqual(r.ac_network_feasible_step_rate, 0.5)
        self.assertEqual(r.ac_backup_activation_steps, 1)

    def test_cli_defaults(self):
        from evaluation.oracle import run as pf
        captured = {}
        with mock.patch.object(pf, 'run', lambda args, p: captured.setdefault('args', args)):
            pf.main(['--feeder', 'ieee33', '--output-root', 'results/x'])
            self.assertEqual(captured['args'].replan, 'never')
            self.assertFalse(captured['args'].allow_non_optimal)
        self.assertEqual(pf.PF_LABELS, dict(never='pf_mpc', on_deviation='pf_mpc_replan_on_deviation'))


@unittest.skipUnless(series('ieee33'), 'no processed time series')
class DirectKw(unittest.TestCase):
    def reset(self, env, k=3):
        from utility.evaluation_reporting import reset_for_evaluation
        reset_for_evaluation(env, int(env.episode_start_indices[k]), 42)

    def test_policy_path_unchanged_by_default(self):
        from environments.battery_feasibility import soc_feasible_action
        env = make_env()
        self.assertEqual(env.dispatch_mode, 'policy')
        self.reset(env)
        a = np.array([0.3, -0.5, 0.0, 1.0, -1.0], dtype=np.float32)
        expected = soc_feasible_action(a.copy(), env.batteries)
        np.testing.assert_array_equal(env._prepare_action(a), expected)
        _, _, _, _, info = env.step(a)
        for key in ('marl_planned_power_kw', 'marl_dispatch_source', 'marl_planned_minus_executed_kw'):
            self.assertNotIn(key, info)
        self.assertEqual(info['marl_action_mapping'], 'soc_feasible')
        with self.assertRaises(ValueError):
            env.set_dispatch_mode('kw')

    def test_direct_kw_executes_planned_power_and_keeps_limiter(self):
        env = make_env()
        env.set_dispatch_mode('direct_kw')
        self.reset(env)
        planned = np.array([50.0, -60.0, 0.0, 30.5, -20.25])
        _, _, _, _, info = env.step(planned.copy())
        self.assertEqual(info['marl_dispatch_source'], 'direct_kw')
        self.assertEqual(info['marl_action_mapping'], 'direct_kw')
        np.testing.assert_array_equal(info['marl_planned_power_kw'], planned)
        if not info['backup_activated']:
            np.testing.assert_allclose(info['final_battery_power_kw'], planned, atol=1e-3)
        np.testing.assert_allclose(info['marl_planned_minus_executed_kw'], planned - info['final_battery_power_kw'])
        # physical limiter stays active: rating and SoC limits
        env.batteries[0].soc = env.batteries[0].config.soc_min + 0.001
        _, _, _, _, info = env.step(np.array([500.0, -500.0, 0.0, 0.0, 0.0]))
        limiter = np.asarray(info['physical_limited_power_kw'])
        self.assertLess(limiter[0], 10.0)                                     # SoC-limited discharge
        self.assertAlmostEqual(limiter[1], -env.batteries[1].config.max_charge_kw, places=3)   # rating
        self.assertGreater(info['marl_planned_minus_limiter_kw'][0], 490.0)
        with self.assertRaises(ValueError):
            env.step(np.array([np.nan, 0, 0, 0, 0]))

    def test_network_backup_stays_active(self):
        env = make_env(scale=1.3)
        env.set_dispatch_mode('direct_kw')
        self.reset(env)
        charge = -np.array([b.config.max_charge_kw for b in env.batteries])
        for _ in range(env.episode_length):
            for b in env.batteries:
                b.soc = 0.5                                                   # keep full charging physically possible
            _, _, _, truncated, info = env.step(charge.copy())
            backup = np.abs(np.asarray(info['physical_limited_power_kw']) - np.asarray(info['final_battery_power_kw']))
            if info['backup_activated'] and backup.max() > 1e-3:
                np.testing.assert_allclose(info['marl_planned_minus_executed_kw'],
                                           charge - np.asarray(info['final_battery_power_kw']))
                self.assertGreater(np.abs(info['marl_planned_minus_executed_kw']).max(), 1e-3)
                return
            if truncated:
                break
        self.fail('no step where the network backup changed a full-charge dispatch at 1.3x load')


@unittest.skipUnless(series('ieee33'), 'no processed time series')
class OneDay(unittest.TestCase):
    """One validation day, IEEE33 TP1: direct-kW vs historical inverse-affine dispatch of the same full-day plan."""

    @classmethod
    def setUpClass(cls):
        from evaluation.evaluate import POLICY, run_episodes
        from evaluation.oracle import solver
        from evaluation.oracle.network import LinDistFlowNetwork
        from evaluation.oracle.run import OracleController
        cls.frames, cls.records, cls.calls = {}, {}, {}
        real = solver.solve_day
        for dispatch in ('policy', 'direct_kw'):
            env = make_env()
            start = int(env.episode_start_indices[3])
            calls = []
            with mock.patch('evaluation.oracle.run.solve_day', lambda *a, **k: calls.append(1) or real(*a, **k)):
                controller = OracleController(env, LinDistFlowNetwork(env), solver.OracleSettings(), 'TP1', 1.0,
                                              dispatch=dispatch, replan='never', allow_non_optimal=False,
                                              detailed=True)
                cls.frames[dispatch] = run_episodes('ieee33', env, controller, POLICY, 'TP1', [(start, 42)])
            controller.finish()
            cls.records[dispatch], cls.calls[dispatch] = controller.records[0], len(calls)

    def test_one_solve_per_day(self):
        for dispatch in ('policy', 'direct_kw'):
            self.assertEqual(self.calls[dispatch], 1)
            self.assertEqual(self.records[dispatch]['solves'], 1)
            self.assertEqual(self.records[dispatch]['replans'], 0)
            self.assertEqual(len(self.frames[dispatch]), 96)

    def test_direct_kw_matches_inverse_affine_dispatch(self):
        policy, direct = self.frames['policy'], self.frames['direct_kw']
        np.testing.assert_allclose(lists(direct.final_battery_power_kw), lists(policy.final_battery_power_kw),
                                   atol=1e-3)
        self.assertAlmostEqual(float(direct.energy_cost.sum()), float(policy.energy_cost.sum()), places=3)
        planned, executed = lists(direct.marl_planned_power_kw), lists(direct.final_battery_power_kw)
        backup = np.asarray(direct.backup_activated, dtype=bool)
        np.testing.assert_allclose(planned[~backup], executed[~backup], atol=1e-3)   # unchanged where no intervention
        self.assertNotIn('marl_planned_power_kw', policy)

    def test_solver_record(self):
        r = self.records['direct_kw']
        self.assertEqual(r['solver_outcomes'], 'optimal')
        self.assertTrue(r['proven_optimal'])
        self.assertEqual(r['plan_network_feasible'], not r['network_slack_used'])
        self.assertEqual(r['replan_mode'], 'never')


@unittest.skipUnless(series('ieee33'), 'no processed time series')
class SolverStatus(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from evaluation.oracle.network import LinDistFlowNetwork
        from evaluation.oracle.solver import day_data
        from utility.evaluation_reporting import reset_for_evaluation
        env = make_env()
        reset_for_evaluation(env, int(env.episode_start_indices[3]), 42)
        cls.net, cls.data = LinDistFlowNetwork(env), day_data(env, 0, env.start_of_day_soc)

    def patched(self, status, keep_x=True):
        from evaluation.oracle import solver
        real = solver.milp

        def fake(*args, **kwargs):
            res = real(*args, **kwargs)
            return SimpleNamespace(x=res.x if keep_x else None, status=status, message=f'forced status {status}',
                                   fun=res.fun)
        return mock.patch.object(solver, 'milp', fake)

    def test_time_limited_solution_refused_or_labelled(self):
        from evaluation.oracle.solver import OracleSettings, PlanError, solve_day
        with self.patched(1):
            with self.assertRaisesRegex(PlanError, 'not proven optimal'):
                solve_day(self.net, self.data, OracleSettings(), allow_non_optimal=False)
            plan = solve_day(self.net, self.data, OracleSettings(), allow_non_optimal=True)
        self.assertEqual(plan.solver_outcome, 'time_limit_feasible')
        self.assertFalse(plan.proven_optimal)
        self.assertEqual(plan.status, 'status_1')                 # historical label unchanged

    def test_no_solution_raises(self):
        from evaluation.oracle.solver import OracleSettings, PlanError, solve_day
        with self.patched(2, keep_x=False):
            for allow in (False, True):
                with self.assertRaisesRegex(PlanError, 'infeasible'):
                    solve_day(self.net, self.data, OracleSettings(), allow_non_optimal=allow)

    def test_slack_reported(self):
        from evaluation.oracle.solver import OracleSettings, solve_day
        plan = solve_day(self.net, self.data, OracleSettings())
        self.assertTrue(plan.proven_optimal)
        self.assertEqual(plan.network_slack_used, plan.voltage_slack > 1e-7 or plan.branch_slack > 1e-7)
        tight = solve_day(self.net, self.data, OracleSettings(voltage_margin_pu=0.049))   # band 0.999-1.001 pu
        self.assertTrue(tight.network_slack_used)
        self.assertGreater(tight.voltage_slack_max, 0.0)
        self.assertLessEqual(tight.voltage_slack_max, tight.voltage_slack + 1e-12)


if __name__ == '__main__':
    unittest.main()
