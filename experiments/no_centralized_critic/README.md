# A2: independent-critic multi-agent actor–critic

Keep both new folders when transferring this experiment. Shared orchestration,
compatibility checks, statistics and tests live in `../no_shared_kpi/`.

Five original ten-input MLP actors are retained. Each independent critic receives
only its own ten-feature observation and scalar action (eleven inputs). All actor,
critic, target and optimizer state is separate per agent. The original environment,
shared grid/KPI reward terms, local reward terms, battery physics, proposed-action
replay and safety mechanisms are unchanged. DOE is OFF to match A0.

Use `train_no_centralized_critic_ieee33.py`,
`evaluate_no_centralized_critic_ieee33.py`, and
`validate_no_centralized_critic.py`. Training uses seeds 42–44 and 200000 total steps
per run on TP1. Selection is validation-only; final evaluation is TP1–TP7 on December.

See `../no_shared_kpi/README.md` for scientific definitions and unchanged components,
`../no_shared_kpi/COMMANDS.md` for all stages, and
`../no_shared_kpi/IMPLEMENTATION_REPORT.md` for actual validation status and file list.
