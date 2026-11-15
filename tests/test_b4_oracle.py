"""Perfect-foresight LinDistFlow benchmark (experiments.b4.oracle). Validation days only (never December test data);
uses data/time_series on RIT or the local audit_tmp/replica copy and is skipped without one.
Run: python -m unittest tests.test_b4_oracle -v
"""
import unittest

import numpy as np

from tests.test_b3 import series

FEEDERS = ('ieee33', 'ieee69')


@unittest.skipUnless(series('ieee33') and series('ieee69'), 'no processed time series')
class Oracle(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from environments.feasibility_env import make_b3_env
        from utility.experiment_setup import feeder_config
        cls.envs = {}
        for feeder in FEEDERS:
            env = make_b3_env(feeder_config(feeder, series(feeder), 'b4'), 'val', 42, 0.0)
            env.reset(seed=42)
            cls.envs[feeder] = env

    def test_lindistflow_matches_ac_power_flow_at_zero_dispatch(self):
        from experiments.b4.oracle.network import LinDistFlowNetwork
        for feeder, env in self.envs.items():
            net = LinDistFlowNetwork(env)
            self.assertEqual(net.num_lines, env.num_buses - 1)
            np.testing.assert_allclose(net.path.sum(1) >= 1, True)               # every line feeds >= 1 bus
            worst_v, worst_p = 0.0, 0.0
            for t in (0, 40, 72):                                                  # night, midday, evening
                load, kvar, pv, _, _ = env._get_profiles(env.episode_start_index + t)
                ac = env.power_flow.run_power_flow(load_kw=load, load_kvar=kvar, pv_kw=pv,
                                                   dess_power_kw=np.zeros(env.num_buses))
                p, q = (load - pv) / net.s_base_kva, kvar / net.s_base_kva
                worst_v = max(worst_v, float(np.max(np.abs(net.squared_voltage(p, q) - ac['voltage_pu_raw']))))
                fp, _ = net.branch_flow(p, q)
                ac_p = np.asarray(ac['line_active_power_kw'], float) / net.s_base_kva
                sign = np.where(env.power_flow.in_service_line_info['FROM'].to_numpy() - 1 == net.upstream, 1.0, -1.0)
                worst_p = max(worst_p, float(np.max(np.abs(fp - sign * ac_p)) / max(np.max(np.abs(ac_p)), 1e-9)))
            self.assertLess(worst_v, 0.01, feeder)     # lossless model: small voltage error
            self.assertLess(worst_p, 0.10, feeder)     # upstream flows differ by the neglected losses

    def day(self, feeder, first_step=0):
        from experiments.b4.oracle.solver import day_data
        env = self.envs[feeder]
        return day_data(env, first_step, env.b3_start_soc)

    def test_plan_respects_battery_model_and_b4_objective(self):
        from environments.battery import Battery
        from experiments.b4.oracle.network import LinDistFlowNetwork
        from experiments.b4.oracle.solver import OracleSettings, overlap_steps, solve_day
        from experiments.no_shared_kpi.protocol import inventory_adjusted_cost
        for feeder, env in self.envs.items():
            d = self.day(feeder)
            plan = solve_day(LinDistFlowNetwork(env), d, OracleSettings())
            self.assertEqual(plan.status, 'optimal')
            self.assertEqual(overlap_steps(plan), 0)
            self.assertEqual(plan.soc.shape, (97, env.num_agents))
            np.testing.assert_allclose(plan.soc[0], 0.5)
            self.assertTrue((plan.soc >= d.soc_min - 1e-7).all() and (plan.soc <= d.soc_max + 1e-7).all())
            # The plan's SoC trajectory is exactly what the environment's battery model produces for the net power.
            for i, b in enumerate(env.batteries):
                sim = Battery(b.config, env.delta_t_hours)
                sim.soc = float(plan.soc[0, i])
                for t in range(96):
                    p = plan.net_power_kw[t, i]
                    sim.step(p / b.config.max_discharge_kw if p >= 0 else p / b.config.max_charge_kw)
                    self.assertAlmostEqual(sim.get_soc(), plan.soc[t + 1, i], places=5)
            caps = d.capacity_kwh
            expected = inventory_adjusted_cost(plan.predicted_raw_cost, float(plan.soc[0] @ caps),
                                               float(plan.soc[-1] @ caps), env.day_mean_price)
            self.assertAlmostEqual(plan.predicted_adjusted_cost, expected, places=9)
            self.assertAlmostEqual(d.day_mean_price, float(np.mean(d.price)), places=9)
            self.assertLess(plan.solve_time_s, 120)

    def test_exclusive_milp_path(self):
        from experiments.b4.oracle.network import LinDistFlowNetwork
        from experiments.b4.oracle.solver import OracleSettings, overlap_steps, solve
        env = self.envs['ieee33']
        d = self.day('ieee33', first_step=80)                                  # rest-of-day window (16 steps)
        lp = solve(LinDistFlowNetwork(env), d, OracleSettings(), binary=False)
        milp = solve(LinDistFlowNetwork(env), d, OracleSettings(), binary=True)
        self.assertEqual(overlap_steps(milp), 0)
        self.assertTrue(milp.milp_used)
        self.assertEqual(milp.soc.shape, (17, env.num_agents))
        self.assertLessEqual(lp.objective, milp.objective + 1e-6)               # LP is a relaxation of the MILP

    def test_one_validation_day_replayed_in_ac_environment(self):
        from environments.feasibility_env import make_b3_env
        from experiments.b3.evaluate import POLICY, run_episodes
        from experiments.b4.oracle.network import LinDistFlowNetwork
        from experiments.b4.oracle.run import OracleController
        from experiments.b4.oracle.solver import OracleSettings
        from experiments.no_shared_kpi.protocol import inventory_adjusted_cost
        from utility.experiment_setup import feeder_config
        env = make_b3_env(feeder_config('ieee33', series('ieee33'), 'b4'), 'val', 42, 0.0)
        start = int(env.episode_start_indices[3])
        ctrl = OracleController(env, LinDistFlowNetwork(env), OracleSettings(), 'TP1', 1.0)
        oracle = run_episodes('ieee33', env, ctrl, POLICY, 'TP1', [(start, 42)])
        ctrl.finish()
        zero = run_episodes('ieee33', env, None, 'zero', 'TP1', [(start, 42)])
        caps = np.array([b.config.capacity_kwh for b in env.batteries])

        def adjusted(frame):
            soc = lambda row: np.array([row[f'soc_agent_{i}'] for i in range(env.num_agents)], float)
            e1 = float(soc(frame.iloc[-1]) @ caps)
            return inventory_adjusted_cost(float(frame.energy_cost.sum()), 0.5 * caps.sum(), e1, env.day_mean_price)
        self.assertEqual(len(oracle), 96)
        self.assertTrue(oracle.episode_completed.all() and not oracle.episode_failed.any())
        record = ctrl.records[0]
        self.assertEqual((record['start_index'], record['statuses']), (start, 'optimal'))
        self.assertGreaterEqual(record['solves'], 1)
        ac_adjusted = adjusted(oracle)
        self.assertLess(ac_adjusted, adjusted(zero))                            # optimised dispatch beats zero
        self.assertLess(abs(ac_adjusted - record['predicted_adjusted_energy_cost']),
                        0.10 * abs(record['predicted_adjusted_energy_cost']) + 5.0)   # model gap (losses) is small
        print(f"\nIEEE33 validation day {start}: AC adjusted {ac_adjusted:.2f} $, predicted "
              f"{record['predicted_adjusted_energy_cost']:.2f} $, zero {adjusted(zero):.2f} $, replans "
              f"{record['replans']}, backup steps {int(oracle.backup_activated.sum())}, MILP {record['milp_solves']}")


if __name__ == '__main__':
    unittest.main()
