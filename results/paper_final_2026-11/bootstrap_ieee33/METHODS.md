# Paired calendar-day bootstrap

```
Task 1: paired calendar-day block bootstrap of December 2018 controller comparisons. Maintained entry point.

    python -u -m evaluation.final_analyses.bootstrap --feeder ieee33 [--results-root results/soc_feasible_affine_v2]
                                                    [--out-root results/final_paper_analyses] [--resamples 10000]

Data: episode_summary.csv of every finished authoritative run (common.load_episodes): one row per (controller, seed,
load factor, topology, December day). Rows are matched on (topology, load factor, start_index); start_index identifies
the calendar day of the environment's series (its date is recorded), never the row position. Missing or non-finite
values drop that date for that comparison (counted in n_dates_dropped); nothing is imputed.

Seed handling (mandatory procedure):
  primary    for each RL controller, the arithmetic mean over the three training seeds per (date, topology, load
             factor); a key without all three seeds is dropped and counted. PF-MPC, F-MPC, PQ-RBC and the zero-action
             reference have ONE outcome per key and are used as they are (never duplicated).
  per_seed   the same comparisons for each seed k separately: an RL controller's seed k against the other RL
             controller's seed k (seed indices are paired as labels of independent runs; they share only the RNG seed
             42 + k), or against the single deterministic outcome.
Paired difference per date: d = Y_A - Y_B (A minus B, as named). Feeder scope: d_date = mean over all evaluated
topologies of (Y_A - Y_B) at that date, so a date is resampled with all its topologies (block); a date missing in any
topology is dropped. Topology scope: one topology.
Uncertainty: percentile bootstrap over the n December dates (B >= 10,000 resamples of date indices with replacement,
fixed RNG seed; the same index matrix for every comparison over the same date set). The intervals quantify variation
across evaluation days CONDITIONAL on the three trained RL seeds; they do not quantify uncertainty over hypothetical
future training seeds (see seed_variability.csv for between-seed spread). Deterministic references carry no run-to-run
uncertainty of their own.
p-values: NOT taken from the bootstrap distribution. A paired sign-flip randomization test on the date-level differences
(H0: the date-level differences are symmetric about zero; R = 10,000 random sign vectors, fixed seed; p = (1 + #{|mean
flipped| >= |mean observed|}) / (R + 1)). Holm correction is applied within prespecified families:
  primary          feeder scope, load factor 1.0, adjusted energy cost, the prespecified comparisons; one family per feeder
  secondary        feeder scope, one family per (feeder, load factor, metric) for every other load factor / metric
  per_topology     one family per (feeder, load factor, metric) over comparisons x topologies
  transfer         one family per (feeder, load factor, metric) over RL controllers x unseen topologies (TPk - TP1)
  per_seed         one family per (feeder, load factor, metric, seed); transfer_per_seed likewise
Holm's m counts only comparisons with a valid p-value (family_tests). A metric a controller does not record (the pure
zero-action reference has no terminal SoC shortfall; common.OPTIONAL_METRIC_COLUMNS) makes those comparisons 'not
estimable' (not_estimable_reason); they carry no p-value and do not enter any family. Every other column, and every
controller, seed, load factor, topology and December day, is required (the run stops otherwise).
F-MPC = the Gurobi results (--fmpc-backend gurobi, default): a folder whose fmpc_solves.csv does not show Gurobi on all
96 decisions of every episode stops the run (common.fmpc_backend_problems); inclusion.json lists the verified folders.
Effect sizes are always reported: mean and median paired difference, relative difference (% of B's mean), Cohen's dz
(mean / SD of date differences) and the fraction of dates on which A is better. A confidence interval containing zero is
not evidence of equivalence; non-inferiority is evaluated only for a margin given with --ni-margin (prespecified).
Outputs (new folder, never overwritten): <out-root>/bootstrap/<feeder>/<UTC>/ comparisons.csv, primary_table.csv,
transfer.csv, per_seed.csv, seed_variability.csv, inclusion.json, METHODS.md.

```
