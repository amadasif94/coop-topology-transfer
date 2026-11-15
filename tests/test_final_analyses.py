"""Final paper analyses on SYNTHETIC data (no research data, no training, no December evaluation).

    python -m unittest tests.test_final_analyses -v
"""
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd

from tests.synthetic_data import write_synthetic_series

TMP = None
SERIES = None


def setUpModule():
    global TMP, SERIES
    TMP = Path(tempfile.mkdtemp(prefix='final-analyses-'))
    SERIES = write_synthetic_series('ieee33', TMP / 'series' / 'synthetic_ieee33.csv', seed=3)


def tearDownModule():
    shutil.rmtree(TMP, ignore_errors=True)


def december_starts(n):
    from evaluation.forecast_mpc.common import load_series, stage_day_starts
    s = load_series('ieee33', SERIES)
    return stage_day_starts(s, 'dec')[:n], s


def finished(d):
    d.mkdir(parents=True, exist_ok=True)
    (d / 'run_manifest.json').write_text(json.dumps(dict(status='finished')))


class Bootstrap(unittest.TestCase):
    """Known synthetic effects: the estimates, pairing and seed handling must recover them exactly."""

    @classmethod
    def setUpClass(cls):
        cls.root = TMP / 'boot_results'
        starts, _ = december_starts(8)
        cls.starts = starts
        rng = np.random.default_rng(0)
        base = {int(s): 100 + 10 * rng.random() for s in starts}           # day-specific level shared by all
        offsets = dict(mlp_a0=[0, 0, 0], mlp_a1=[1, 1, 1], mlp_a2=[0, 3, 6], mlp_a3=[2, 2, 2])
        refs = dict(pf_mpc=-5.0, forecast_mpc=-2.0, price_quantile_rbc=4.0, zero_action_no_backup=10.0)

        def table(controller, run_id, shift, tp_shift):
            rows = []
            for tp in ('TP1', 'TP2'):
                for s in starts:
                    v = base[int(s)] + shift + (tp_shift if tp == 'TP2' else 0.0)
                    rows.append(dict(controller=controller, load_scaling_factor=1.0, topology_case=tp, run_id=run_id,
                                     start_index=int(s), adjusted_energy_cost=v, raw_energy_cost=v + 1,
                                     total_reward_mean=-v, final_network_feasibility_rate=1.0, mean_voltage_deviation=0.1,
                                     b3_terminal_shortfall_mean=0.0))
            return pd.DataFrame(rows).sample(frac=1.0, random_state=len(controller))   # shuffled row order
        for c, offs in offsets.items():
            for k, o in enumerate(offs):
                d = cls.root / 'ieee33' / 'mlp' / c[-2:].upper() / f'seed_{k}' / 'test' / 'load_x100'
                finished(d)
                table(c, k, o, 3.0 if c == 'mlp_a0' else 0.0).to_csv(d / 'episode_summary.csv', index=False)
        for c, o in refs.items():
            d = cls.root / 'ieee33' / c / 'load_x100'
            finished(d)
            table(c, np.nan, o, 0.0).to_csv(d / 'episode_summary.csv', index=False)
        smoke = cls.root / 'ieee33' / 'pf_mpc' / 'load_x100_smoke1'
        finished(smoke)
        table('pf_mpc', np.nan, 999.0, 0.0).to_csv(smoke / 'episode_summary.csv', index=False)

    def run_main(self, out, extra=('--allow-incomplete',)):
        from evaluation.final_analyses import bootstrap
        return bootstrap.main(['--feeder', 'ieee33', '--results-root', str(self.root), '--out-root', str(out),
                               '--scales', '1.0', '--time-series', str(SERIES), *extra])

    def test_missing_prerequisites_stop_the_analysis(self):
        """Only TP1/TP2 and 8 days exist here: without --allow-incomplete the analysis refuses and says why."""
        with self.assertRaisesRegex(SystemExit, 'missing prerequisites'):
            self.run_main(TMP / 'boot_refused', extra=())
        from evaluation.final_analyses.common import coverage_problems, load_episodes
        units, info = load_episodes(self.root, 'ieee33', ['adjusted_energy_cost'], time_series=SERIES)
        problems = coverage_problems(units[units.controller != 'forecast_mpc'], 'ieee33', [1.0],
                                     ('mlp_a0', 'forecast_mpc'), info)
        self.assertTrue(any(p.startswith('forecast_mpc x1.0: no finished run') for p in problems))
        self.assertTrue(any('topologies missing' in p and 'TP7' in p for p in problems))

    def test_known_effects_seed_mean_and_families(self):
        out = self.run_main(TMP / 'boot_out')
        c = pd.read_csv(out / 'comparisons.csv')
        prim = c[(c.family_role == 'primary')].set_index('comparison')
        self.assertEqual(set(prim.family), {'primary|ieee33'})
        # feeder scope = mean over TP1, TP2 of (A0 - B); A0 is +3 on TP2 only -> +1.5 on average
        self.assertAlmostEqual(prim.loc['A0 - A1', 'mean_diff'], 1.5 - 1.0)
        self.assertAlmostEqual(prim.loc['A0 - A2', 'mean_diff'], 1.5 - 3.0)     # A2 seed mean of 0, 3, 6
        self.assertAlmostEqual(prim.loc['A0 - PF-MPC', 'mean_diff'], 1.5 + 5.0)
        self.assertAlmostEqual(prim.loc['A1 - A3', 'mean_diff'], -1.0)
        self.assertEqual(prim.loc['A0 - PF-MPC', 'seeds_b'], 'deterministic')
        self.assertTrue((prim.n_dates == 8).all())
        self.assertAlmostEqual(prim.loc['A1 - A3', 'ci_low'], -1.0)           # constant difference -> degenerate CI
        self.assertLess(prim.loc['A1 - A3', 'p_signflip'], 0.02)
        self.assertTrue((prim.p_holm >= prim.p_signflip - 1e-12).all())
        per_seed = c[c.family_role == 'per_seed']
        a2 = per_seed[(per_seed.comparison == 'A0 - A2') & (per_seed.metric == 'adjusted_energy_cost')
                      & (per_seed.scale == 1.0)]
        self.assertAlmostEqual(float(a2[a2.seed.astype(str) == '2'].mean_diff.iloc[0]), 1.5 - 6.0)  # seed 2 vs seed 2
        tp = c[(c.scope == 'topology') & (c.comparison == 'A0 - A1') & (c.metric == 'adjusted_energy_cost')].set_index('topology')
        self.assertAlmostEqual(tp.loc['TP2', 'mean_diff'], 2.0)
        self.assertAlmostEqual(tp.loc['TP1', 'mean_diff'], -1.0)
        info = json.loads((out / 'inclusion.json').read_text())
        self.assertTrue(any('smoke' in e['dir'] for e in info['excluded']))     # smoke run never used

    def test_transfer_and_savings(self):
        out = self.run_main(TMP / 'boot_out_tr')
        t = pd.read_csv(out / 'transfer.csv')
        a0 = t[(t.controller == 'mlp_a0') & (t.seed == 'mean_of_3')].set_index('metric')
        self.assertAlmostEqual(a0.loc['adjusted_energy_cost', 'mean_diff'], 3.0)       # TP2 - TP1
        self.assertAlmostEqual(a0.loc['savings_vs_zero', 'mean_diff'], -3.0)          # savings fall by 3 on TP2
        self.assertTrue(bool(a0.loc['adjusted_energy_cost', 'degradation_detectable']))
        a1 = t[(t.controller == 'mlp_a1') & (t.seed == 'mean_of_3')].set_index('metric')
        self.assertAlmostEqual(a1.loc['adjusted_energy_cost', 'mean_diff'], 0.0)
        self.assertTrue(t.family.str.startswith('transfer').all())

    def test_missing_seed_drops_the_key_and_pairs_by_key_not_position(self):
        from evaluation.final_analyses.bootstrap import values
        from evaluation.final_analyses.common import load_episodes
        units, _ = load_episodes(self.root, 'ieee33', ['adjusted_energy_cost'], time_series=SERIES)
        u = units.drop(units[(units.controller == 'mlp_a2') & (units.seed == 1) &
                             (units.start_index == int(self.starts[0]))].index)
        v = values(u, 'mlp_a2', 'adjusted_energy_cost')
        self.assertEqual(len(v), 2 * len(self.starts) - 2)       # that date (both topologies) has 2 seeds: dropped
        self.assertTrue(units.date.str.startswith('2018-12').all())

    def test_holm_and_resampler(self):
        from evaluation.final_analyses.bootstrap import Resampler, holm, paired_stats
        np.testing.assert_allclose(holm([0.01, 0.04, 0.03, np.nan]), [0.03, 0.06, 0.06, np.nan])
        d, dates = np.array([1.0, -0.5, 2.0, 0.3, 0.8]), list('abcde')
        a = paired_stats(d, dates, Resampler(10000, 10000), 'lower')
        b = paired_stats(d, dates, Resampler(10000, 10000), 'lower')
        self.assertEqual((a['ci_low'], a['ci_high'], a['p_signflip']), (b['ci_low'], b['ci_high'], b['p_signflip']))
        self.assertAlmostEqual(paired_stats(np.zeros(6), list('abcdef'), Resampler(), 'lower')['p_signflip'], 1.0)

    def test_outputs_never_inside_authoritative_results(self):
        from evaluation.final_analyses.common import guard_output
        for bad in ('results/soc_feasible_affine_v2/x', 'results/soc_feasible/y', str(self.root / 'z')):
            with self.assertRaises(ValueError):
                guard_output(bad, self.root)
        guard_output('results/final_paper_analyses', self.root)


def step_table(n_days, starts, soc_fn, n=5):
    rows = []
    for d, s in enumerate(starts[:n_days]):
        for k in range(96):
            soc = soc_fn(d, k)
            row = dict(topology_case='TP1', start_index=int(s), step=k, record_type='dispatch_attempt',
                       executed_battery_power_kw=json.dumps([10.0 if k % 2 else -10.0] * n),
                       marl_terminal_completed=k == 95,
                       marl_terminal_soc_shortfall=json.dumps([0.25] * n) if k == 95 else None,
                       marl_start_of_day_soc=json.dumps([0.5] * n))
            row.update({f'soc_agent_{i}': soc[i] for i in range(n)})
            rows.append(row)
    return pd.DataFrame(rows)


class SocDiagnostics(unittest.TestCase):
    def batteries(self):
        return pd.DataFrame([dict(battery=i, bus=10 + i, soc_min=0.1, soc_max=0.9, capacity_kwh=500.0) for i in range(5)])

    def test_runs_streaks_reset_at_midnight_and_terminal_values(self):
        from evaluation.final_analyses.soc_diagnostics import day_metrics, runs_of
        self.assertEqual(runs_of(np.array([0, 1, 1, 0, 1, 1, 1, 0], bool)), (2, 3))
        starts, _ = december_starts(2)

        def soc(d, k):
            low = (d == 0 and (10 <= k < 20 or 50 <= k < 55 or k >= 90)) or (d == 1 and k < 6)
            return [0.1 if low else 0.5, 0.1005, 0.5, 0.5, 0.5]
        steps = step_table(2, starts, soc)
        rows = {(int(s), r['battery']): r for s, g in steps.groupby('start_index') for r in day_metrics(g, self.batteries())}
        r0 = rows[(int(starts[0]), 0)]
        self.assertEqual((r0['near_min_episodes'], r0['longest_streak_intervals']), (3, 10))
        self.assertAlmostEqual(r0['longest_streak_hours'], 2.5)
        self.assertAlmostEqual(r0['near_min_fraction'], 21 / 96)
        self.assertEqual(rows[(int(starts[1]), 0)]['longest_streak_intervals'], 6)   # 6 + 6 across midnight not merged
        self.assertEqual(rows[(int(starts[0]), 1)]['near_min_fraction'], 1.0)          # within SOC_NEAR = 0.001
        self.assertAlmostEqual(r0['terminal_shortfall'], 0.25)
        self.assertAlmostEqual(r0['charge_throughput_kwh'], 48 * 10 * 0.25)

    def test_end_to_end_tables_and_a3_vs_a0(self):
        from evaluation.final_analyses import soc_diagnostics
        root = TMP / 'soc_results'
        starts, _ = december_starts(3)
        for c, level in (('A0', 0.5), ('A3', 0.1)):
            for k in range(3):
                d = root / 'ieee33' / 'mlp' / c / f'seed_{k}' / 'test' / 'load_x100'
                finished(d)
                step_table(3, starts, lambda dd, kk, lv=level: [lv] * 5).to_csv(d / 'step_metrics.csv', index=False)
        with mock.patch.object(soc_diagnostics, 'battery_table', return_value=self.batteries()):
            out = soc_diagnostics.main(['--feeder', 'ieee33', '--results-root', str(root), '--out-root',
                                        str(TMP / 'soc_out'), '--scales', '1.0', '--time-series', str(SERIES)])
        day = pd.read_csv(out / 'battery_day.csv')
        self.assertEqual(len(day), 2 * 3 * 3 * 5)                                     # controllers x seeds x days x batteries
        cmp = pd.read_csv(out / 'a0_vs_a3_battery.csv')
        frac = cmp[(cmp.metric == 'near_min_fraction') & (cmp.seed == 'mean_of_3')]
        self.assertEqual(len(frac), 5)
        self.assertTrue(np.allclose(frac.mean_diff, 1.0))                             # A3 always near min, A0 never
        self.assertTrue((pd.read_csv(out / 'soc_profile.csv').step.max() == 95))
        self.assertIn('streak_bin', pd.read_csv(out / 'streak_distribution.csv'))


def synthetic_env(scale=1.0):
    from configs.experiments import feeder_config
    from environments.marl_env import make_marl_env
    return make_marl_env(feeder_config('ieee33', SERIES, 'soc_feasible'), 'val', 42, 0.0, scale)


class ActionReplay(unittest.TestCase):
    def test_targets_are_checked_not_substituted(self):
        from evaluation.final_analyses.action_replay import EXPERIMENTS, verify_target
        self.assertEqual(EXPERIMENTS, dict(ieee33=(('TP6', 'primary'), ('TP7', 'secondary')),
                                           ieee69=(('TP10', 'primary'),)))
        verify_target('ieee69', 'TP10')
        for feeder, tp in (('ieee33', 'TP10'), ('ieee69', 'TP6'), ('ieee69', 'TP11')):
            with self.assertRaises(SystemExit):
                verify_target(feeder, tp)

    def test_requested_kw_replay_keeps_limiter_and_backup(self):
        from evaluation.evaluate import POLICY, run_episodes
        from evaluation.final_analyses.action_replay import ReplayAgent
        from evaluation.final_analyses.common import json_list
        env = synthetic_env()
        start = int(env.episode_start_indices[2])
        request = np.full((96, env.num_agents), -400.0)                     # far beyond rating: limiter must cut
        request[48:] = 400.0
        agent = ReplayAgent(env, {start: request})
        self.assertEqual(env.dispatch_mode, 'direct_kw')
        frame = run_episodes('ieee33', env, agent, POLICY, 'TP1', [(start, 42)])
        steps = frame[frame.record_type == 'dispatch_attempt'].sort_values('step')
        planned = json_list(steps.marl_planned_power_kw.tolist(), env.num_agents)
        limiter = json_list(steps.physical_limited_power_kw.tolist(), env.num_agents)
        np.testing.assert_allclose(planned, request[:len(steps)])            # the recorded kW, unmodified
        self.assertTrue((np.abs(limiter) < np.abs(planned)).any())           # the target battery limiter acted
        self.assertIn('backup_activated', steps)                             # the AC network backup stage ran
        self.assertTrue((steps.marl_dispatch_source == 'direct_kw').all())

    def test_source_requests_and_compare_signs(self):
        from evaluation.final_analyses import action_replay
        root, out = TMP / 'replay_results', TMP / 'replay_out'
        starts, _ = december_starts(2)
        n = 5
        for k in (0, 1, 2):
            cdir = root / 'ieee33' / 'mlp' / 'A0' / f'seed_{k}' / 'test' / 'load_x100'
            finished(cdir)
            steps = []
            for tp in ('TP1', 'TP6'):
                for s in starts:
                    for st in range(96):
                        steps.append(dict(topology_case=tp, start_index=int(s), step=st, record_type='dispatch_attempt',
                                          marl_power_soc_mapped_kw=json.dumps([float(st)] * n),
                                          executed_battery_power_kw=json.dumps([0.0] * n), soc_after=json.dumps([0.5] * n)))
            pd.DataFrame(steps).to_csv(cdir / 'step_metrics.csv', index=False)
            ep = pd.DataFrame([dict(topology_case=tp, start_index=int(s), raw_energy_cost=10.0, adjusted_energy_cost=10.0,
                                    total_reward_mean=1.0) for tp in ('TP1', 'TP6') for s in starts])
            ep.to_csv(cdir / 'episode_summary.csv', index=False)
            rdir = out / 'action_replay' / 'ieee33' / 'mlp_a0' / f'seed_{k}' / 'TP1_to_TP6' / 'load_x100'
            finished(rdir)
            ep[ep.topology_case == 'TP6'].assign(adjusted_energy_cost=12.0).to_csv(rdir / 'episode_summary.csv', index=False)
            pd.DataFrame(steps)[lambda x: x.topology_case == 'TP6'].to_csv(rdir / 'step_metrics.csv', index=False)
        req, skipped, _ = action_replay.source_requests(root, 'ieee33', 'mlp_a0', 0, 1.0, n)
        self.assertEqual((sorted(req), skipped), (sorted(int(s) for s in starts), []))
        np.testing.assert_allclose(req[int(starts[0])][:, 0], np.arange(96))
        with mock.patch.object(action_replay, 'SCALES', (1.0,)):
            with self.assertRaisesRegex(SystemExit, 'missing or unfinished'):        # A3 and TP7 runs do not exist
                action_replay.compare(root, out, 'ieee33')
            res = action_replay.compare(root, out, 'ieee33', allow_incomplete=True)
        info = json.loads((res / 'inclusion.json').read_text())
        self.assertFalse(info['complete'])
        self.assertEqual(set(info['closed_loop_origin'].values()), {'authoritative_test_run'})
        paired = pd.read_csv(res / 'paired.csv')
        row = paired[(paired.metric == 'adjusted_energy_cost') & (paired.seed == 'mean_of_3') & (paired.target == 'TP6')]
        self.assertAlmostEqual(float(row.mean_diff.iloc[0]), -2.0)            # closed loop (10) - replay (12)
        self.assertIn('source_requested_kw', pd.read_csv(res / 'trajectories.csv', nrows=5))


class ReplayInputs(unittest.TestCase):
    """Closed-loop target data: reused only if the authoritative run holds the target topology; checkpoint check."""

    def test_closed_loop_source_selection_and_resumption(self):
        from evaluation.final_analyses import action_replay as ar
        root, base = TMP / 'cl_results', TMP / 'cl_out'
        for tps, k in ((('TP1', 'TP6'), 0), (('TP1',), 1)):
            d = root / 'ieee69' / 'mlp' / 'A0' / f'seed_{k}' / 'test' / 'load_x100'
            finished(d)
            pd.DataFrame(dict(topology_case=list(tps), start_index=[0] * len(tps))).to_csv(d / 'episode_summary.csv',
                                                                                         index=False)
        d0, origin0 = ar.closed_loop_dir(root, base, 'ieee69', 'mlp_a0', 0, 1.0, 'TP6')
        d1, origin1 = ar.closed_loop_dir(root, base, 'ieee69', 'mlp_a0', 1, 1.0, 'TP10')
        self.assertEqual((origin0, d0), ('authoritative_test_run', root / 'ieee69' / 'mlp' / 'A0' / 'seed_0' / 'test' / 'load_x100'))
        self.assertEqual(origin1, 'generated_evaluation_only')                # TP10 absent: never assumed to exist
        self.assertTrue(str(d1).endswith(str(Path('mlp_a0') / 'seed_1' / 'closed_loop_TP10' / 'load_x100')))
        self.assertTrue(ar._begin(d1))                                        # nothing there yet
        finished(d1)
        self.assertFalse(ar._begin(d1))                                       # finished: skipped on resubmission
        partial = base / 'partial'
        partial.mkdir(parents=True)
        (partial / 'run_manifest.json').write_text(json.dumps(dict(status='started')))
        with self.assertRaisesRegex(SystemExit, 'unfinished'):
            ar._begin(partial)

    def test_checkpoint_must_match_the_source_run(self):
        from evaluation.final_analyses import action_replay as ar
        root = TMP / 'ck_results'
        d = root / 'ieee33' / 'mlp' / 'A3' / 'seed_2' / 'test' / 'load_x100'
        d.mkdir(parents=True)
        (d / 'run_manifest.json').write_text(json.dumps(dict(status='finished', checkpoint_file=dict(sha256='a' * 64))))
        ok = ar.checkpoint_consistency(root, 'ieee33', 'mlp_a3', 2, 1.0, dict(selection=dict(checkpoint_sha256='a' * 64)))
        self.assertEqual(ok['checkpoint_match'], 'verified')
        with self.assertRaisesRegex(SystemExit, 'differs from the frozen selection'):
            ar.checkpoint_consistency(root, 'ieee33', 'mlp_a3', 2, 1.0, dict(selection=dict(checkpoint_sha256='b' * 64)))


class OptionalZeroShortfall(unittest.TestCase):
    """Zero-action tables lack b3_terminal_shortfall_mean: not estimable, never an error; other columns mandatory."""

    @classmethod
    def setUpClass(cls):
        Bootstrap.setUpClass()                                      # same synthetic tree, copied, then edited
        cls.root = TMP / 'boot_results_nozero'
        shutil.copytree(Bootstrap.root, cls.root)
        z = cls.root / 'ieee33' / 'zero_action_no_backup' / 'load_x100' / 'episode_summary.csv'
        pd.read_csv(z).drop(columns='b3_terminal_shortfall_mean').to_csv(z, index=False)

    def test_zero_without_shortfall_is_not_an_error_and_others_still_are(self):
        from evaluation.final_analyses.bootstrap import METRICS
        from evaluation.final_analyses.common import coverage_problems, load_episodes
        units, info = load_episodes(self.root, 'ieee33', [c for c, _ in METRICS.values()], time_series=SERIES)
        self.assertEqual(info['missing_columns'], {})
        self.assertIn('zero_action_no_backup', info['unavailable_optional_columns']['b3_terminal_shortfall_mean'])
        problems = coverage_problems(units, 'ieee33', [1.0], ('zero_action_no_backup',), info)
        self.assertFalse(any('column' in p for p in problems))
        pf = self.root / 'ieee33' / 'pf_mpc' / 'load_x100' / 'episode_summary.csv'   # a required column elsewhere
        original = pd.read_csv(pf)
        original.drop(columns='b3_terminal_shortfall_mean').to_csv(pf, index=False)
        try:
            units2, info2 = load_episodes(self.root, 'ieee33', [c for c, _ in METRICS.values()], time_series=SERIES)
            problems2 = coverage_problems(units2, 'ieee33', [1.0], ('pf_mpc',), info2)
            self.assertTrue(any(p.startswith('column b3_terminal_shortfall_mean missing') for p in problems2))
        finally:
            original.to_csv(pf, index=False)

    def test_comparisons_not_estimable_and_primary_family_unchanged(self):
        from evaluation.final_analyses import bootstrap
        out = bootstrap.main(['--feeder', 'ieee33', '--results-root', str(self.root), '--out-root',
                              str(TMP / 'boot_out_nozero'), '--scales', '1.0', '--time-series', str(SERIES),
                              '--allow-incomplete'])                 # only because the synthetic tree has 2 topologies
        c = pd.read_csv(out / 'comparisons.csv', keep_default_na=False, na_values=[''])
        zero = c[(c.comparison == 'A0 - Zero-action') & (c.metric == 'terminal_soc_shortfall')]
        self.assertTrue(len(zero) and zero.mean_diff.isna().all())
        self.assertTrue((zero.interpretation == 'not estimable').all())
        self.assertTrue(zero.not_estimable_reason.str.contains('not recorded for zero_action_no_backup').all())
        other = c[(c.comparison == 'A0 - Zero-action') & (c.metric == 'adjusted_energy_cost')]
        self.assertTrue(other.mean_diff.notna().all())                # zero-action stays in the cost comparisons
        prim = c[c.family_role == 'primary']
        self.assertEqual((len(prim), int(prim.family_tests.iloc[0])), (9, 9))   # primary Holm family unchanged
        sec = c[c.family == 'secondary|ieee33|x1.0|terminal_soc_shortfall']
        self.assertEqual(int(sec.family_tests.iloc[0]), 8)            # the not-estimable test is not counted
        info = json.loads((out / 'inclusion.json').read_text())
        self.assertIn('b3_terminal_shortfall_mean', info['unavailable_optional_columns'])


def write_fmpc_dec(root, feeder='ieee33', buses=33):
    """Synthetic December F-MPC artifacts in the real layout: records, predictor folders, forecasts with npz hashes."""
    import hashlib
    base = root / feeder / 'forecast_mpc'
    for kind in ('models', 'forecasts'):
        (base / kind / 'dec').mkdir(parents=True, exist_ok=True)
    for t in [f'{k}_node_{i}' for k in ('load', 'pv') for i in range(1, buses + 1)]:
        folder = base / 'models' / 'dec' / t
        (folder / 'models' / 'Best').mkdir(parents=True)
        for name in ('predictor.pkl', 'learner.pkl', 'metadata.json', 'version.txt'):
            (folder / name).write_text('x')
        (folder / 'models' / 'Best' / 'model.pkl').write_text('m')
        common = dict(feeder=feeder, stage='dec', target=t, series_sha256='s' * 64, model_dir_sha256='m' * 64)
        (base / 'models' / 'dec' / f'{t}.json').write_text(json.dumps(dict(
            schema='graph-marl-dess-forecast-model/2', cutoff_time='2018-12-01 00:00:00', **common)))
        npz = base / 'forecasts' / 'dec' / f'{t}.npz'
        npz.write_bytes(t.encode())
        (base / 'forecasts' / 'dec' / f'{t}.json').write_text(json.dumps(dict(
            schema='graph-marl-dess-forecast/2', month='2018-12', days=31,
            npz_sha256=hashlib.sha256(t.encode()).hexdigest(), **common)))
        (base / 'models' / 'dec' / f'{t}.leaderboard.csv').write_text('model\nBest\n')
    (base / 'forecasts' / 'dec' / 'unrelated_manifest.json').write_text('{}')
    return base


class PreflightArtifacts(unittest.TestCase):
    def run_preflight(self, root):
        import contextlib
        import io
        from evaluation.final_analyses import preflight
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            preflight.check_fmpc(rep := preflight.Report(), Path(root), 'ieee33')
        return buf.getvalue(), rep.missing

    def test_fmpc_targets_are_the_buses_and_each_is_validated(self):
        root = TMP / 'pre_fmpc'
        base = write_fmpc_dec(root)
        text, missing = self.run_preflight(root)
        self.assertEqual(missing, 0, text)
        self.assertIn('F-MPC models/dec: 66/66 targets valid', text)            # 2 x 33 buses, not 2 x 5 batteries
        self.assertIn('F-MPC forecasts/dec: 66/66 targets valid', text)
        self.assertIn('unrelated JSON file(s) ignored', text)                   # never counted
        (base / 'forecasts' / 'dec' / 'pv_node_7.npz').write_bytes(b'truncated')
        (base / 'models' / 'dec' / 'load_node_3' / 'learner.pkl').unlink()
        rec = base / 'models' / 'dec' / 'pv_node_9.json'
        rec.write_text(json.dumps(dict(json.loads(rec.read_text()), stage='nov')))
        text, missing = self.run_preflight(root)
        self.assertEqual(missing, 2, text)
        self.assertIn('F-MPC models/dec: 64/66 targets valid', text)
        self.assertIn('F-MPC forecasts/dec: 65/66 targets valid', text)
        self.assertIn('npz_sha256', text)

    def test_zero_action_shortfall_note_but_pf_mpc_shortfall_required(self):
        import contextlib
        import io
        from evaluation.final_analyses import preflight
        root = TMP / 'pre_cols'
        cols = [c for c in preflight.EPISODE_COLUMNS if c != 'b3_terminal_shortfall_mean']
        for c in ('zero_action_no_backup', 'pf_mpc'):
            d = root / c
            finished(d)
            pd.DataFrame([{k: (c if k == 'controller' else 1) for k in cols} | dict(topology_case=t, start_index=s)
                          for t in preflight.TOPOLOGIES['ieee33'] for s in range(31)]).to_csv(d / 'episode_summary.csv',
                                                                                             index=False)
        buf, rep = io.StringIO(), preflight.Report()
        with contextlib.redirect_stdout(buf):
            preflight.check_episode_table(rep, root / 'zero_action_no_backup', 'ieee33', 'bootstrap', False,
                                          controller='zero_action_no_backup')
            self.assertEqual(rep.missing, 0)
            preflight.check_episode_table(rep, root / 'pf_mpc', 'ieee33', 'bootstrap', False, controller='pf_mpc')
        self.assertEqual(rep.missing, 1)
        self.assertIn("optional ['b3_terminal_shortfall_mean'] absent", buf.getvalue())
        self.assertIn("columns ['b3_terminal_shortfall_mean']", buf.getvalue())


class Preflight(unittest.TestCase):
    def test_reports_missing_items_and_runs_without_torch(self):
        import subprocess
        import sys
        root = TMP / 'pre_results'
        d = root / 'ieee69' / 'mlp' / 'A0' / 'seed_0' / 'test' / 'load_x100'
        finished(d)
        pd.DataFrame([dict(controller='mlp_a0', run_id=0, topology_case=t, start_index=s, adjusted_energy_cost=1,
                           raw_energy_cost=1, total_reward_mean=1, final_network_feasibility_rate=1,
                           mean_voltage_deviation=1, b3_terminal_shortfall_mean=0)
                      for t in [f'TP{i}' for i in range(1, 10)] for s in range(31)]).to_csv(d / 'episode_summary.csv',
                                                                                            index=False)
        cols = [f'soc_agent_{i}' for i in range(9)] + ['marl_terminal_soc_shortfall', 'marl_terminal_completed',
                                                       'executed_battery_power_kw', 'marl_power_soc_mapped_kw']
        pd.DataFrame(columns=cols).to_csv(d / 'step_metrics.csv', index=False)
        repo = Path(__file__).resolve().parents[1]
        code = ('import sys, importlib.abc\n'
                'class B(importlib.abc.MetaPathFinder):\n'
                '    def find_spec(self, name, path=None, target=None):\n'
                '        if name.split(".")[0] in ("torch", "numpy", "pandas", "configs", "environments"):\n'
                '            raise ImportError("blocked " + name)\n'
                f'sys.meta_path.insert(0, B()); sys.path.insert(0, {str(repo)!r})\n'
                'from evaluation.final_analyses import preflight\n'
                f'preflight.main(["--results-root", {str(root)!r}, "--feeders", "ieee69"])\n')
        proc = subprocess.run([sys.executable, '-I', '-c', code], capture_output=True, text=True, timeout=120, cwd=repo)
        self.assertEqual(proc.returncode, 1, proc.stderr)
        out = proc.stdout
        self.assertIn('PREFLIGHT INCOMPLETE', out)
        self.assertIn("closed-loop TP10 absent: action_replay will run an evaluation-only closed loop first", out)
        self.assertIn("topologies missing ['TP10']", out)
        self.assertIn('[soc_diagnostics] ieee69', out.replace('\\', '/'))
        self.assertIn('MISSING  [timing] F-MPC models/dec: 0/138 targets valid', out)    # 2 x 69 buses
        self.assertNotIn('blocked', proc.stderr)


class Timing(unittest.TestCase):
    def test_single_issue_forecast_equals_the_full_causal_run(self):
        from evaluation.forecast_mpc.generate import recursive_forecast
        starts, s = december_starts(2)
        y = s.load[:, 4]
        mix = lambda X: 0.5 * X['y_lag_1'].to_numpy() + 0.5 * X['y_lag_96'].to_numpy()
        full, _ = recursive_forecast(y, s.price, s.timestamps, starts, mix)
        for k in (0, 17, 95):
            one, _ = recursive_forecast(y, s.price, s.timestamps, starts[1:2], mix, issue_steps=[k])
            np.testing.assert_array_equal(one[0, k], full[1, k])
            self.assertTrue(np.isnan(one[0, (k + 1) % 96]).all())

    def test_day_timer_covers_96_decisions_and_summary_refuses_mixed_hardware(self):
        from evaluation.final_analyses import timing
        from evaluation.rule_based.pq_rbc import PriceQuantileRBC
        env = synthetic_env()
        rbc = PriceQuantileRBC(env, 25.0, 40.0, 'TP1', 1.0)
        rows = timing.time_day(env, lambda obs: (rbc.select_action(obs), {}), int(env.episode_start_indices[0]), 42)
        self.assertEqual([r['step'] for r in rows], list(range(96)))
        self.assertTrue(all(r['controller_s'] >= 0 for r in rows))
        out = TMP / 'timing_out'
        for name, cpu, label in (('a_x', 'CPU A', 'x'), ('b_y', 'CPU B', 'y')):
            d = out / 'timing' / 'ieee33' / 'job_1' / name
            d.mkdir(parents=True)
            pd.DataFrame([dict(label=label, scale=1.0, controller_s=1.0)]).to_csv(d / 'per_day.csv', index=False)
            pd.DataFrame([dict(label=label, scale=1.0, controller_s=0.01)]).to_csv(d / 'per_decision.csv', index=False)
            (d / 'environment.json').write_text(json.dumps(dict(cpu_model=cpu, requested_threads=4, omp_threads='4',
                                                                label=label)))
        with self.assertRaisesRegex(SystemExit, 'different CPU models'):
            timing.summarize('ieee33', out, None, group='job_1')
        with self.assertRaisesRegex(SystemExit, 'no finished timing runs'):
            timing.summarize('ieee33', out, None, group='job_2')                   # groups never mix


if __name__ == '__main__':
    unittest.main()
