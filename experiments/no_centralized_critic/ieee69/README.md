# IEEE69 A2 — independent critics

Nine original ten-feature actors; nine independent critics with eleven inputs
(own ten-feature observation and own scalar action). Original shared team reward,
IEEE69 physics/safety and training protocol are unchanged. This is an independent-
critic multi-agent actor–critic baseline, not centralized-critic MADDPG.

`train.py` and `evaluate.py` use the common IEEE69 runner in
`experiments/no_shared_kpi/ieee69/`. The top-level new IEEE69 SLURM scripts use
the shared IEEE69 dispatcher in that same package. No A2-only dispatcher exists.

See `experiments/no_shared_kpi/ieee69/README.md` for exact commands, A0 evidence,
pending metadata, test status, and safe additive transfer instructions. Do not
overwrite or change any existing IEEE33 files while transferring this extension.
