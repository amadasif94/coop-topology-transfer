"""Forecast-Based MPC (F-MPC): causal AutoGluon load/PV forecasts + rest-of-day receding-horizon LinDistFlow MPC.
Maintained. Separate from PF-MPC (evaluation.oracle), whose code and behaviour are unchanged; F-MPC only reuses its
optimisation model (evaluation.oracle.solver.solve_day / DayData) with forecast inputs.

common.py      stages, paths, the environment's own processed series, forecast files, hashes
features.py    the causal feature builder shared by training and recursive forecasting
train.py       entry point: one AutoGluon TabularPredictor per bus and quantity (load, PV) for a stage
generate.py    entry point: causal recursive multi-step forecasts for every decision step of a period
validate.py    entry point: forecast accuracy by horizon, persistence baselines, physical checks
controller.py  the F-MPC controller (re-optimises the rest of the day every 15 minutes, direct kW dispatch)
run.py         entry point: F-MPC evaluation in the unchanged AC environment

Monthly expanding-window walk-forward stages, each with its own models and the same frozen settings: 'oct' trains on
2018-01-01..09-30 and forecasts October, 'nov' trains on 2018-01-01..10-31 and forecasts November (both validation
months), 'dec' trains on 2018-01-01..11-30 and forecasts the December test days. Models are never refitted within a
month, and no December row is ever used for training. Every forecast issued at decision row t uses realized load/PV of rows < t only (row t itself is the interval
about to be dispatched and is forecast too), calendar features, and the day-ahead prices of the same day.
"""
