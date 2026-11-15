"""Final paper analyses (Applied Energy). Maintained; read the authoritative results, write ONLY under
results/final_paper_analyses/ (common.guard_output refuses any path inside an authoritative results tree).

common.py          controller labels, discovery of finished authoritative runs, episode/step loading, date mapping,
                   selected-checkpoint and environment helpers
bootstrap.py       Task 1: paired calendar-day block bootstrap (controller comparisons, topology transfer, per seed)
timing.py          Task 2: CPU-only controller computation time per simulated day (A0-A3, PF-MPC, F-MPC, PQ-RBC)
soc_diagnostics.py Task 3: per-battery SoC depletion diagnostics, A3 vs A0
action_replay.py   Task 4: TP1 requested-kW action replay on an unseen topology vs closed-loop execution

Scope: MLP-MADDPG A0-A3, PF-MPC, F-MPC, PQ-RBC and the zero-action reference only.
"""
