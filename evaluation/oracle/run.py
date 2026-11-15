"""Perfect-Foresight MPC (PF-MPC): Full-Day Deterministic LinDistFlow Optimization, replayed in the AC environment.
Maintained.

    python -u -m evaluation.oracle.run --feeder ieee69 [--protocol soc_feasible] [--output-root results/<root>]
                                       [--topologies TP1 ...] [--scales 1.0 ...] [--max-days N]
                                       [--replan never|on_deviation] [--allow-non-optimal]

For every (topology, load scale) the unchanged test environment is built exactly as for the learned controllers
(evaluation.evaluate.test: topology_config(..., experiment=<protocol>), make_marl_env(..., 'test', 42, 0.0, scale); the same 31
December days from episode_plan; SoC0 = 0.50; physical limiter and directional backup active). Every load scale has its
own environment, so every (day, topology, load scale) gets its own optimisation with that scale's realized load.

PRIMARY BENCHMARK (--replan never, default): at the first step of each day ONE full 96-step LP/MILP is solved with the
realized load, PV and price of that day (perfect information, evaluation.oracle.solver); the planned battery powers
are then replayed open loop, never re-optimised during the day. Dispatch is DIRECT kW (MarlDESSEnv 'direct_kw' mode):
the planned kW bypass the learned-policy action mapping (affine soc_feasible), while the physical battery limiter,
network backup, AC power flow and all accounting stay active. Optional --replan on_deviation (NOT the primary
benchmark): the rest of the day is re-optimised from the actual SoC whenever the executed SoC differs from the plan
(physical limiter or backup changed the dispatch); every re-solve is counted.

Solver: a solve that is not proven optimal (time/iteration limit) stops the run with an error unless
--allow-non-optimal is given; then it is used and labelled time_limit_feasible / proven_optimal = False. A failed
solve (no solution) always stops the run; no fallback dispatch is ever substituted. Network limits in the LP are soft;
pf_mpc_days.csv reports per day whether slack was needed (plan_network_feasible = False then) and its totals/maxima,
separately from the AC feasibility of the replayed day.

Outputs (never overwritten): <output-root>/<feeder>/pf_mpc/load_x<pct>[_smoke<N>]/ (the optional variant:
<output-root>/<feeder>/pf_mpc_replan_on_deviation/..., controller label of the same name) with the A0-A3 tables (run_manifest,
step_metrics with marl_planned_power_kw / marl_planned_minus_limiter_kw / marl_planned_minus_executed_kw,
episode_summary, aggregate_summary, terminal_soc_summary) plus pf_mpc_days.csv. Output root default results/<protocol>.

This is an information-advantaged optimization reference: a lossless, linearised network model with perfect
foresight, evaluated in the nonlinear AC environment. It is NOT a certified global AC optimum and not an online
controller. ZERO is not run here.
The historical command `python -m experiments.b4.oracle.run` (default experiment b4, root results/<experiment>,
'b4_oracle' manifest block, B4_ORACLE_* markers, label oracle_lindistflow, oracle_days.csv, policy-action dispatch
through the environment's action mapping, re-plan on deviation, time-limited plans accepted) is unchanged: it runs
this module with legacy=True.
"""
import argparse
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import scipy

from configs.experiments import (B4_FAMILY, FEEDERS, MAINTAINED_PROTOCOL, PROTOCOLS, TOPOLOGIES,
                                 maintained_output_root, protocol_id, protocol_name, source_inventory, topology_config)
from environments.marl_env import make_marl_env
from evaluation.evaluate import POLICY, feeder_tools, run_episodes, write_tables
from evaluation.oracle.network import LinDistFlowNetwork
from evaluation.oracle.solver import OracleSettings, day_data, policy_action, solve_day
from experiments.load_stress.stress_env import ALLOWED_SCALES, check_scale, scale_label, scenario
from utility.evaluation_protocol import authorize, episode_plan, protocol_config

LABEL = 'oracle_lindistflow'    # historical (legacy) label and folder
PF_LABEL = 'pf_mpc'             # maintained PF-MPC label and folder (primary benchmark, --replan never)
REPLAN_MODES = ('never', 'on_deviation')
PF_LABELS = dict(never=PF_LABEL, on_deviation='pf_mpc_replan_on_deviation')   # the optional variant never shares a folder
BENCHMARK_NAME = 'Perfect-Foresight MPC (PF-MPC): Full-Day Deterministic LinDistFlow Optimization'
SEED = 42                       # the ZERO reference's seed: identical days and reset seeds for matched comparisons
SOC_REPLAN_TOL = 1e-5           # executed vs planned SoC; larger differences trigger a re-plan (on_deviation only)
DISPATCH_TOL_KW = 1e-3          # |planned - executed| above this counts as a changed dispatch (float32 rounding ~1e-5 kW)
DISCLAIMER = ('Perfect-foresight LinDistFlow optimization benchmark (realized load, PV and prices of the day; '
              'lossless linearised network), evaluated in the nonlinear AC environment. Not a certified global '
              'AC optimum, not an online controller.')
PF_DISCLAIMER = (f'{BENCHMARK_NAME}. Information-advantaged optimization reference: realized load, PV and prices of '
                 'the whole day; lossless linearised network; planned kW replayed directly in the nonlinear AC '
                 'environment with the physical limiter and network backup active. Not a certified global AC optimum, '
                 'not a lower bound on AC cost, not an online controller, and a single deterministic run (not seeds).')


class OracleController:
    """Duck-typed agent for the existing evaluators: select_action(obs, noise_std) returns the planned dispatch.

    dispatch 'policy' (historical): a policy action that makes the env's action mapping execute the planned power.
    dispatch 'direct_kw' (PF-MPC): the planned kW; the env is switched to its direct_kw dispatch mode.
    replan 'on_deviation' (historical) or 'never' (PF-MPC primary: one solve per day)."""

    def __init__(self, env, net, settings, topology, scale, dispatch='policy', replan='on_deviation',
                 allow_non_optimal=True, detailed=False):
        if dispatch not in ('policy', 'direct_kw'):
            raise ValueError(f'unknown dispatch {dispatch!r}')
        if replan not in REPLAN_MODES:
            raise ValueError(f'unknown replan mode {replan!r}')
        self.env, self.net, self.settings = env, net, settings
        self.topology, self.scale = topology, scale
        self.dispatch, self.replan, self.allow_non_optimal, self.detailed = dispatch, replan, allow_non_optimal, detailed
        if dispatch == 'direct_kw':
            env.set_dispatch_mode('direct_kw')
        self.records, self.day, self.plan, self.data, self.window = [], None, None, None, 0

    def _solve(self, step, soc):
        self.data = day_data(self.env, step, soc)
        self.plan, self.window = solve_day(self.net, self.data, self.settings, self.allow_non_optimal), step
        d, p = self.day, self.plan
        d['solves'] += 1
        d['milp_solves'] += int(p.milp_used)
        d['lp_overlap_steps'] += p.overlap_steps_lp
        d['solve_time_s'] += p.solve_time_s
        d['statuses'].add(p.status)
        d['voltage_slack'] += p.voltage_slack
        d['branch_slack'] += p.branch_slack
        if self.detailed:
            d['solver_outcomes'].add(p.solver_outcome)
            d['solver_status_codes'].add(str(p.solver_status_code))
            d['proven_optimal'] = bool(d['proven_optimal'] and p.proven_optimal)
            d['voltage_slack_max'] = max(d['voltage_slack_max'], p.voltage_slack_max)
            d['branch_slack_max'] = max(d['branch_slack_max'], p.branch_slack_max)
            d['network_slack_used'] = bool(d['network_slack_used'] or p.network_slack_used)

    def select_action(self, obs, noise_std=0.0):
        env = self.env
        if env.protocol is None or env.protocol.get('name') not in B4_FAMILY:
            raise RuntimeError('the oracle runs only in a soc_feasible or rated_feasibility_supervised (b4) environment')
        if self.dispatch == 'direct_kw' and getattr(env, 'dispatch_mode', 'policy') != 'direct_kw':
            raise RuntimeError('PF-MPC direct-kW dispatch needs the environment in direct_kw mode')
        step = int(env.current_step)
        soc = np.array([b.get_soc() for b in env.batteries], dtype=float)
        if step == 0:
            self.finish()
            self.day = dict(topology_case=self.topology, load_scaling_factor=self.scale,
                            start_index=int(env.episode_start_index), day_mean_price=float(env.day_mean_price),
                            solves=0, milp_solves=0, lp_overlap_steps=0, replans=0, solve_time_s=0.0,
                            statuses=set(), voltage_slack=0.0, branch_slack=0.0, max_soc_deviation=0.0)
            if self.detailed:
                self.day.update(replan_mode=self.replan, dispatch=self.dispatch, solver_outcomes=set(),
                                solver_status_codes=set(), proven_optimal=True, voltage_slack_max=0.0,
                                branch_slack_max=0.0, network_slack_used=False, soc_deviation_steps=0)
            self._solve(0, soc)
            p = self.plan
            self.day.update(predicted_raw_energy_cost=p.predicted_raw_cost,
                            predicted_adjusted_energy_cost=p.predicted_adjusted_cost, predicted_objective=p.objective,
                            predicted_v_min=p.predicted_v_min, predicted_v_max=p.predicted_v_max,
                            predicted_branch_loading_max=p.predicted_branch_loading_max,
                            planned_throughput_kwh=float((p.charge_kw + p.discharge_kw).sum() * self.data.dt_hours))
            if self.detailed:
                self.day.update(planned_terminal_soc_mean=float(np.mean(p.soc[-1])))
        else:
            deviation = float(np.max(np.abs(soc - self.plan.soc[step - self.window])))
            self.day['max_soc_deviation'] = max(self.day['max_soc_deviation'], deviation)
            if deviation > SOC_REPLAN_TOL:
                if self.detailed:
                    self.day['soc_deviation_steps'] += 1
                if self.replan == 'on_deviation':
                    self.day['replans'] += 1
                    self._solve(step, soc)
        if self.dispatch == 'direct_kw':
            return np.asarray(self.plan.net_power_kw[step - self.window], dtype=float).copy()
        return policy_action(self.plan, self.data, step - self.window, env)

    def finish(self):
        if self.day is not None:
            for key in ('statuses', 'solver_outcomes', 'solver_status_codes'):
                if key in self.day:
                    self.day[key] = ';'.join(sorted(self.day[key]))
            if self.detailed:
                self.day['plan_network_feasible'] = not self.day['network_slack_used']
            self.records.append(self.day)
            self.day = None


def _arrays(g, column):
    if column not in g or not g[column].notna().all():
        return None
    values = [json.loads(v) if isinstance(v, str) else v for v in g[column]]
    return np.asarray(values, dtype=float)


def dispatch_statistics(frame, dt_hours):
    """Per day: planned vs physical-limiter vs executed battery power (kW) and the AC feasibility of the replay."""
    rows = []
    for (tp, start), g in frame.groupby(['topology_case', 'start_index']):
        if 'record_type' in g:
            g = g[g.record_type == 'dispatch_attempt']
        g = g.sort_values('step')
        r = dict(topology_case=tp, start_index=int(start), replay_steps=int(len(g)))
        planned, limiter = _arrays(g, 'marl_planned_power_kw'), _arrays(g, 'physical_limited_power_kw')
        executed = _arrays(g, 'final_battery_power_kw')
        if planned is not None and limiter is not None and executed is not None and \
                planned.shape == limiter.shape == executed.shape and planned.size:
            d_exec, d_lim, d_backup = np.abs(planned - executed), np.abs(planned - limiter), np.abs(limiter - executed)
            r.update(mean_abs_planned_minus_executed_kw=float(d_exec.mean()),
                     max_abs_planned_minus_executed_kw=float(d_exec.max()),
                     steps_executed_differs_from_plan=int((d_exec > DISPATCH_TOL_KW).any(axis=1).sum()),
                     mean_abs_planned_minus_limiter_kw=float(d_lim.mean()),
                     max_abs_planned_minus_limiter_kw=float(d_lim.max()),
                     steps_limiter_changed_plan=int((d_lim > DISPATCH_TOL_KW).any(axis=1).sum()),
                     mean_abs_backup_change_kw=float(d_backup.mean()), max_abs_backup_change_kw=float(d_backup.max()),
                     steps_backup_changed_dispatch=int((d_backup > DISPATCH_TOL_KW).any(axis=1).sum()),
                     replayed_planned_throughput_kwh=float(np.abs(planned).sum() * dt_hours),
                     executed_throughput_kwh=float(np.abs(executed).sum() * dt_hours))
        if 'final_network_feasible' in g:
            feasible = pd.to_numeric(g['final_network_feasible'].astype(float), errors='coerce')
            r['ac_network_feasible_step_rate'] = float(feasible.mean()) if len(g) else np.nan
        if 'backup_activated' in g:
            r['ac_backup_activation_steps'] = int(pd.to_numeric(g['backup_activated'].astype(float),
                                                                errors='coerce').fillna(0).sum())
        if 'episode_failed' in g:
            r['ac_episode_failed'] = bool(g['episode_failed'].astype(bool).any())
        rows.append(r)
    return pd.DataFrame(rows)


def settings_from(args):
    return OracleSettings(voltage_margin_pu=args.voltage_margin, branch_margin=args.branch_margin,
                          time_limit_s=args.time_limit, mip_rel_gap=args.mip_gap)


def run_scale(args, scale, settings, legacy=False, sources=None):
    """legacy=True: historical names ('b4_oracle' block with b3_sources, B4_ORACLE_* markers, oracle_lindistflow,
    oracle_days.csv) and historical behaviour (policy-action dispatch, re-plan on deviation, time-limited plans used).
    legacy=False: PF-MPC (pf_mpc, direct-kW dispatch, --replan, strict solver status unless --allow-non-optimal)."""
    if legacy:
        block, prefix, label, days_file = 'b4_oracle', 'B4_', LABEL, 'oracle_days.csv'
        dispatch, replan, allow_non_optimal = 'policy', 'on_deviation', True
    else:
        replan = args.replan
        block, prefix, label, days_file = 'pf_mpc_record', 'PF_MPC_', PF_LABELS[replan], 'pf_mpc_days.csv'
        dispatch, allow_non_optimal = 'direct_kw', bool(args.allow_non_optimal)
    sources = sources if sources is not None else source_inventory()
    feeder, topologies = args.feeder, args.topologies
    out = Path(args.output_root) / feeder / label / (scale_label(scale) + (f'_smoke{args.max_days}' if args.max_days else ''))
    run = feeder_tools(feeder)[3](out, label, SEED, 'test', None, vars(args))
    protocol = authorize(mode='test', final=True, run_purpose='final_test', learned=False, topologies=topologies)
    run.record_protocol(protocol, [label])
    run.data[block] = dict(controller=label, disclaimer=DISCLAIMER if legacy else PF_DISCLAIMER,
                           experiment=args.experiment, protocol=protocol_name(args.experiment),
                           action_mapping='soc_feasible' if args.experiment == 'soc_feasible' else 'rated',
                           load_scaling_factor=scale,
                           pv_scaling_factor=1.0, settings=settings.as_dict(), soc_replan_tolerance=SOC_REPLAN_TOL,
                           solver=f'scipy.optimize.milp (HiGHS), scipy {scipy.__version__}',
                           python_version=platform.python_version(),
                           **{('b3_sources' if legacy else 'source_inventory'): sources},
                           evaluation_policy_tag=f'{POLICY} (evaluator dispatch name; controller={label})',
                           days=f'first {args.max_days}' if args.max_days else 'all')
    if not legacy:
        run.data[block].update(
            benchmark=BENCHMARK_NAME, dispatch='direct_kw (planned kW bypass the policy action mapping; physical '
            'limiter, network backup and AC power flow active)', replan_mode=replan,
            primary_benchmark=replan == 'never', allow_non_optimal=allow_non_optimal,
            objective='adjusted energy cost (raw import cost - day mean price x stored-energy change), lossless model; '
                      'tie-breakers eps_curtailment/eps_throughput and soft network-limit slack penalties only',
            dispatch_tolerance_kw=DISPATCH_TOL_KW, deterministic_single_run=True)
    frames, envs, days, stats = [], {}, [], []
    for tp in topologies:
        cfg = protocol_config(topology_config(feeder, tp, out / 'network_inputs', experiment=args.experiment), protocol)
        env = make_marl_env(cfg, 'test', SEED, 0.0, scale)
        starts, seeds, _ = episode_plan(env, 'all', SEED, protocol)
        if len(starts) != 31:
            raise SystemExit('Expected 31 December test days')
        plan = list(zip(starts, seeds))[:args.max_days or None]
        run.add_environment(env, topology=tp, scenario=scenario(scale), intended_episodes=len(plan))
        controller = OracleController(env, LinDistFlowNetwork(env), settings, tp, scale, dispatch=dispatch,
                                      replan=replan, allow_non_optimal=allow_non_optimal, detailed=not legacy)
        frame = run_episodes(feeder, env, controller, POLICY, tp, plan)
        frames.append(frame)
        controller.finish()
        days.extend(controller.records)
        if not legacy:
            stats.append(dispatch_statistics(frame, env.delta_t_hours))
        envs[tp] = env
        print(f'{prefix}ORACLE_TOPOLOGY_DONE' if legacy else 'PF_MPC_TOPOLOGY_DONE', feeder, tp, scale_label(scale),
              len(plan), 'days', flush=True)
    write_tables(feeder, out, frames, envs, label, dict(run_id=None, load_scaling_factor=scale))
    episodes = pd.read_csv(out / 'episode_summary.csv')
    oracle = pd.DataFrame(days).merge(
        episodes[['topology_case', 'start_index', 'raw_energy_cost', 'adjusted_energy_cost']],
        on=['topology_case', 'start_index'], how='left', validate='one_to_one')
    oracle['adjusted_cost_model_gap'] = oracle['adjusted_energy_cost'] - oracle['predicted_adjusted_energy_cost']
    oracle['raw_cost_model_gap'] = oracle['raw_energy_cost'] - oracle['predicted_raw_energy_cost']
    if not legacy:
        oracle = oracle.merge(pd.concat(stats, ignore_index=True), on=['topology_case', 'start_index'], how='left',
                              validate='one_to_one')
        oracle.insert(0, 'controller', label)
    oracle.to_csv(out / days_file, index=False)
    summary = dict(
        days=int(len(oracle)), days_with_replans=int((oracle.replans > 0).sum()), milp_solves=int(oracle.milp_solves.sum()),
        days_with_network_slack=int(((oracle.voltage_slack > 0) | (oracle.branch_slack > 0)).sum()),
        mean_adjusted_cost_model_gap=float(oracle.adjusted_cost_model_gap.mean()))
    if not legacy:
        executed_diff = oracle.get('max_abs_planned_minus_executed_kw', pd.Series(dtype=float))
        summary.update(
            solves=int(oracle.solves.sum()), max_solves_per_day=int(oracle.solves.max()),
            days_not_proven_optimal=int((~oracle.proven_optimal.astype(bool)).sum()),
            days_plan_used_network_slack=int(oracle.network_slack_used.astype(bool).sum()),
            max_voltage_slack=float(oracle.voltage_slack_max.max()), max_branch_slack=float(oracle.branch_slack_max.max()),
            days_dispatch_changed_by_env=int((oracle.get('steps_executed_differs_from_plan',
                                                          pd.Series(0, index=oracle.index)) > 0).sum()),
            max_abs_planned_minus_executed_kw=float(executed_diff.max()) if len(executed_diff) else np.nan,
            mean_ac_network_feasible_step_rate=float(oracle.get('ac_network_feasible_step_rate',
                                                                  pd.Series(dtype=float)).mean()))
    run.data[block]['summary'] = summary
    run.finish()
    print(f'{prefix}ORACLE_DONE' if legacy else 'PF_MPC_DONE', out, flush=True)
    return out


def add_oracle_arguments(p):
    p.add_argument('--feeder', required=True, choices=FEEDERS)
    p.add_argument('--topologies', nargs='+', help='default: every topology of the feeder')
    p.add_argument('--scales', type=float, nargs='+', default=list(ALLOWED_SCALES))
    p.add_argument('--max-days', type=int, help='smoke test: first N of the 31 days (separate _smoke<N> folder)')
    defaults = OracleSettings()
    p.add_argument('--voltage-margin', type=float, default=defaults.voltage_margin_pu)
    p.add_argument('--branch-margin', type=float, default=defaults.branch_margin)
    p.add_argument('--time-limit', type=float, default=defaults.time_limit_s)
    p.add_argument('--mip-gap', type=float, default=defaults.mip_rel_gap)


def run(args, p, legacy=False, sources=None):
    args.topologies = [t.upper() for t in (args.topologies or TOPOLOGIES[args.feeder])]
    unknown = sorted(set(args.topologies) - set(TOPOLOGIES[args.feeder]))
    if unknown:
        p.error(f'unknown topologies {unknown}')
    if args.max_days is not None and not 1 <= args.max_days < 31:
        p.error('--max-days must be between 1 and 30 (omit it for all 31 days)')
    settings = settings_from(args)
    return [run_scale(args, check_scale(s), settings, legacy, sources) for s in args.scales]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_oracle_arguments(p)
    oracle_protocols = sorted(k for k, v in PROTOCOLS.items() if v in B4_FAMILY)
    p.add_argument('--protocol', choices=oracle_protocols, default=MAINTAINED_PROTOCOL,
                   help='environment protocol (batteries, terminal valuation); PF-MPC dispatches kW directly')
    p.add_argument('--output-root', help='default results/<protocol>')
    p.add_argument('--replan', choices=REPLAN_MODES, default='never',
                   help='never (default, primary PF-MPC: one full-day solve per day, open-loop replay) or '
                        'on_deviation (optional variant: re-optimise the rest of the day when the executed SoC '
                        'leaves the plan)')
    p.add_argument('--allow-non-optimal', action='store_true',
                   help='use time-limited solutions (labelled time_limit_feasible, not proven optimal) instead of '
                        'stopping the run')
    args = p.parse_args(argv)
    args.experiment = protocol_id(args.protocol)
    try:
        args.output_root = maintained_output_root(args.protocol, args.output_root)
    except ValueError as exc:
        p.error(str(exc))
    return run(args, p)


if __name__ == '__main__':
    main()
