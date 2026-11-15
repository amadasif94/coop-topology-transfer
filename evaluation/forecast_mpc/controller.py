"""F-MPC controller: rest-of-day receding-horizon LinDistFlow MPC on causal forecasts, direct kW dispatch. Maintained.

At every decision step k of a day (start row s, decision row t = s + k) the controller
  1. reads the CAUSAL forecasts issued at t for the rows t .. s + 95 (evaluation.forecast_mpc.generate; nominal load and
     PV), multiplies the LOAD by the environment's stress factor lambda (PV never), and derives reactive load with the
     environment's own rule (base kvar x total load / total base load, DESSEnv._get_profiles);
  2. takes the day-ahead prices of rows t .. s + 95 (published before the day; see known_day_prices), the measured SoC of
     every battery (the executed state, never a planned or future value), the day-start stored energy and the
     protocol's terminal valuation price;
  3. solves PF-MPC's unchanged optimisation model (evaluation.oracle.solver.solve_day: same LinDistFlow network, battery
     dynamics, limits, adjusted-cost objective, terminal valuation, slack formulation, HiGHS settings) over that
     shrinking horizon, and
  4. requests the FIRST planned battery power in kW (+ discharge, - charge) through the environment's direct_kw path,
     so the physical limiter and the AC network backup act exactly as for PF-MPC.
It never reads realized load or PV (no env._get_profiles, no evaluation.oracle.solver.day_data).

Solver failure (no solution; or, unless allow_non_optimal, a solution not proven optimal; or an MILP that still
overlaps): the step requests 0 kW for every battery through the same direct_kw path (limiter and backup unchanged);
no earlier plan is reused. Every solve is logged (time, feeder, topology, load factor, status, message, proven
optimality, fallback, runtime); backup activation and AC feasibility are joined from the step table by run.py.
Missing or nonfinite forecasts are a hard error, never a fallback.
"""
import time

import numpy as np
import pandas as pd

from evaluation.oracle.solver import DayData, battery_arrays, solve_day

FALLBACK = 'zero_kw_request'
PRICE_AVAILABILITY = ('dst_guard', 'full_day')
# Environment prices are the NYISO day-ahead prices placed by ROW POSITION (scripts/build_smartds_15min_timeseries_*).
# If the raw file follows daylight saving time (23-hour day on 2018-03-11, 25-hour day on 2018-11-04), the last hour of
# every environment day from 2018-03-11 to 2018-11-03 is 00:00 of the NEXT market day, whose day-ahead price is not
# published when the environment day starts. 'dst_guard' replaces it by the previous environment day's last hour (00:00
# of the same market day, already known) on those days, for the whole day; outside those days, and for December, both
# modes are identical. 'full_day' uses every price of the day (correct if the raw file has 24 rows on every day).
DST_SHIFT_FIRST, DST_SHIFT_LAST = pd.Timestamp('2018-03-11').date(), pd.Timestamp('2018-11-03').date()
LAST_HOUR = slice(92, 96)


def known_day_prices(env, mode):
    """(96 day-ahead prices of the current day as known at its start, substituted?)."""
    if mode not in PRICE_AVAILABILITY:
        raise ValueError(f'price availability must be one of {PRICE_AVAILABILITY}')
    ts = env.time_series
    start = int(env.episode_start_index)
    price = ts['df'][ts['price_col']].to_numpy(dtype=float) * float(env.price_scaling_factor)
    day = price[start:start + 96].copy()
    if len(day) != 96 or not np.isclose(day.mean(), env.day_mean_price, rtol=0, atol=1e-9):
        raise ValueError('day prices differ from the environment day mean price')
    date = pd.Timestamp(ts['df'][ts['datetime_col']].iloc[start]).date()
    if mode == 'dst_guard' and DST_SHIFT_FIRST <= date <= DST_SHIFT_LAST:
        if start < 4:
            raise ValueError('no previous day to take the guarded last-hour price from')
        day[LAST_HOUR] = price[start - 4:start]
        return day, True
    return day, False


def forecast_day_data(env, store, step, soc0, scale, price_mode='dst_guard'):
    """DayData of the shrinking horizon step .. 95 built ONLY from forecasts, known prices and measured state."""
    start = int(env.episode_start_index)
    if int(env.current_index) != start + int(step):
        raise RuntimeError('environment row and decision step disagree')
    load_nominal, pv = store.window(start, step)
    load_kw = float(scale) * load_nominal                        # stress factor on load only (DESSEnv._get_profiles)
    pv_kw = pv * float(env.pv_scaling_factor)                    # 1.0 (checked by run.py)
    base_kw = np.asarray(env.grid.get_base_load_kw(), dtype=float)
    base_kvar = np.asarray(env.grid.get_base_load_kvar(), dtype=float)
    load_kvar = base_kvar[None, :] * (load_kw.sum(axis=1) / float(base_kw.sum()))[:, None]
    prices, substituted = known_day_prices(env, price_mode)
    batteries = battery_arrays(env)
    data = DayData(load_kw=load_kw, load_kvar=load_kvar, pv_kw=pv_kw, price=prices[int(step):],
                   soc0=np.asarray(soc0, dtype=float),
                   day_mean_price=float(prices.mean()) if substituted else float(env.day_mean_price),
                   day_start_energy_kwh=float(np.asarray(env.start_of_day_soc) @ batteries['capacity_kwh']),
                   dt_hours=float(env.delta_t_hours), export_limit_kw=float(env.grid_export_limit_kw), **batteries)
    if load_kw.shape != (96 - int(step), env.num_buses) or pv_kw.shape != load_kw.shape:
        raise ValueError(f'forecast window shape {load_kw.shape}, expected {(96 - int(step), env.num_buses)}')
    return data, substituted


class ForecastMPCController:
    """Duck-typed agent for the existing evaluators: select_action(obs, noise_std) returns the battery kW request."""

    def __init__(self, env, net, settings, store, feeder, topology, scale, allow_non_optimal=False,
                 price_mode='dst_guard'):
        if price_mode not in PRICE_AVAILABILITY:
            raise ValueError(f'price availability must be one of {PRICE_AVAILABILITY}')
        self.env, self.net, self.settings, self.store = env, net, settings, store
        self.feeder, self.topology, self.scale = feeder, topology, float(scale)
        self.allow_non_optimal, self.price_mode = bool(allow_non_optimal), price_mode
        env.set_dispatch_mode('direct_kw')
        self.records, self.solves, self.day = [], [], None

    def _new_day(self):
        self.finish()
        env = self.env
        self.day = dict(topology_case=self.topology, load_scaling_factor=self.scale,
                        start_index=int(env.episode_start_index), day_mean_price=float(env.day_mean_price),
                        solves=0, failed_solves=0, fallback_steps=0, solves_not_proven_optimal=0, milp_solves=0,
                        solves_with_network_slack=0, solve_time_s=0.0, max_solve_time_s=0.0,
                        price_substituted=False)

    def select_action(self, obs, noise_std=0.0):
        env = self.env
        if getattr(env, 'dispatch_mode', 'policy') != 'direct_kw':
            raise RuntimeError('F-MPC needs the environment in direct_kw dispatch mode')
        step = int(env.current_step)
        if step == 0:
            self._new_day()
        soc = np.array([b.get_soc() for b in env.batteries], dtype=float)       # executed (measured) state
        data, substituted = forecast_day_data(env, self.store, step, soc, self.scale, self.price_mode)
        row = dict(feeder=self.feeder, topology_case=self.topology, load_scaling_factor=self.scale,
                   start_index=int(env.episode_start_index), step=step, decision_row=int(env.current_index),
                   date_time=str(env.time_series['df'][env.time_series['datetime_col']].iloc[int(env.current_index)]),
                   horizon_steps=int(len(data.price)), soc0=soc.tolist(), price_substituted=bool(substituted))
        started = time.perf_counter()
        try:
            plan = solve_day(self.net, data, self.settings, self.allow_non_optimal)
        except RuntimeError as exc:                                     # PlanError or an MILP that still overlaps
            request = np.zeros(env.num_agents, dtype=float)
            row.update(solve_ok=False, solver_outcome=getattr(exc, 'outcome', 'failed'), solver_status_code=None,
                       solver_message=f'{type(exc).__name__}: {exc}', proven_optimal=False, milp_used=None,
                       objective=np.nan, voltage_slack=np.nan, branch_slack=np.nan, network_slack_used=None,
                       fallback=FALLBACK)
        else:
            request = np.asarray(plan.net_power_kw[0], dtype=float).copy()
            row.update(solve_ok=True, solver_outcome=plan.solver_outcome, solver_status_code=plan.solver_status_code,
                       solver_message=plan.solver_message, proven_optimal=bool(plan.proven_optimal),
                       milp_used=bool(plan.milp_used), objective=float(plan.objective),
                       voltage_slack=float(plan.voltage_slack), branch_slack=float(plan.branch_slack),
                       network_slack_used=bool(plan.network_slack_used), fallback='none')
        elapsed = time.perf_counter() - started
        row.update(solve_time_s=elapsed, requested_kw=request.tolist())
        self.solves.append(row)
        d = self.day
        d['solves'] += 1
        d['solve_time_s'] += elapsed
        d['max_solve_time_s'] = max(d['max_solve_time_s'], elapsed)
        d['price_substituted'] = bool(d['price_substituted'] or substituted)
        if row['solve_ok']:
            d['solves_not_proven_optimal'] += int(not row['proven_optimal'])
            d['milp_solves'] += int(row['milp_used'])
            d['solves_with_network_slack'] += int(row['network_slack_used'])
        else:
            d['failed_solves'] += 1
            d['fallback_steps'] += 1
        return request

    def finish(self):
        if self.day is not None:
            self.records.append(self.day)
            self.day = None
