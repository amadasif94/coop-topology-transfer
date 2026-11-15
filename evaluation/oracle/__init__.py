"""Perfect-Foresight MPC (PF-MPC): Full-Day Deterministic LinDistFlow Optimization (not a certified AC optimum).
Maintained home of the former experiments.b4.oracle; runs in the soc_feasible (default) or
rated_feasibility_supervised protocol environment. Maintained runs (label pf_mpc) solve once per day and replay the
planned kW directly (MarlDESSEnv direct_kw mode); the historical entry point keeps its original behaviour.

network.py  LinDistFlow model of the environment's in-service radial network (topology-specific)
solver.py   one-day (or rest-of-day) LP / MILP with the protocol batteries and the adjusted-cost objective (SciPy HiGHS)
run.py      replays the optimized dispatch in the unchanged AC environment and writes the controller tables
"""
