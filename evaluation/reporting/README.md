# Strict matched reporting (`evaluation.reporting`)

**Maintained location (Phase 2, 2026-11).** This package is the maintained home of the former
`experiments.b3_reporting`; that path is now a compatibility shim that resolves to the same module objects, so the
historical commands below (`python -m experiments.b3_reporting.evaluate ...`) keep working unchanged. New work uses

    python -m evaluation.reporting.evaluate --help
    python -m evaluation.reporting.compare --help
    python -m unittest tests.test_reporting -v

The evaluator accepts maintained training manifests (`run_record` block, controller key such as `mlp_a0`, checkpoint
schema `graph-marl-dess-checkpoint/1`) and historical ones (`b3` block, `b3_mlp_a0`, `b3-checkpoint/2`). The experiment
class names `original_maddpg`, `b3_post_tanh`, `b3_pre_tanh` and `soc_feasible` are data identifiers already written into
existing result manifests and are kept verbatim. The text below is the historical documentation; the paths
`results/b3...` in its examples refer to existing historical results, which remain read-only.

# Evaluation-only original MADDPG / corrected B3 comparison (historical documentation)

## Scope after verification review

This is optional infrastructure for exploratory B3, not a prerequisite for the
main paper on original graph-based MADDPG zero-shot topology generalization.
It is NOT wired into the original MADDPG evaluators and does not repair their
reporting automatically. No existing training or shared reporting file was edited.
The original topology evaluators already call `augment_summaries`, which records
failed/incomplete episodes and economic inclusion/exclusion counts. Their old
tiny-threshold clipping rate remains a different metric from material corrections.
The separate original load-stress summarizer still converts missing booleans to
False and drops failure-count fields from its episode comparison table. These
main-paper concerns require separately approved changes; they are not fixed here.

Use this package instead of the legacy `experiments.b3.evaluate` and `summarize`
for the primary comparison. No existing file was replaced: the old reporting
module and B3 directory are recorded in running training source inventories.
Training does not import this new package. Existing legacy results stay intact.

## Scientific contract

* Classes are `original_maddpg`, `b3_post_tanh`, `b3_pre_tanh`. Primary paired
  comparison admits only original versus pre-tanh; unknown historical metadata
  is explicitly excluded. A directory name alone is never formulation evidence.
* Both original and B3 policies use their existing checkpoint loaders, including
  A1/A2 source-inventory compatibility. No gradient/update method is invoked.
* Selection remains the original TP1 61-day S-SEL-v1 eligibility/reward rule.
  Original reward and B3 reward retain their respective training objectives.
* Final tests use frozen selected checkpoints, the same 31 December dates/reset
  seeds, TP1–TP7 for IEEE33, and supported TP1–TP10 definitions for IEEE69.
* Baseline TP1 inputs must match the training manifest before generated topology
  line-file changes are allowed. Load scale may be 1.0/1.1/1.2/1.3; other physical,
  reward, safety, interface, normalization, and data settings must agree.
* Strict historical configuration hashing recovers omitted timestep metadata.
  An unprovable historical configuration is a blocker, not guessed evidence.
  In particular, an IEEE69 historical pre-TP10 base-file mismatch currently
  fails closed here even if another workflow has an approved migration. This
  package does not silently import that exception into baseline comparison.
* Inputs/checkpoints/selection/result artifacts are fingerprinted. A finished
  result is reused only on exact scientific identity and artifact checksum match.
  Existing nonempty or incompatible directories are never overwritten.
* The old B3 reporting entry points remain unchanged for source preservation.
  Do not mix their outputs into the primary comparison by renaming directories.

## A3 (`mlp_a3`, independent DTDE with local rewards)

* Accepted only as `--experiment-class b3_pre_tanh`; no original or post-tanh A3 exists.
* A3 is TRAINED on individual local rewards (`a3_local_e3_own_bus_tariff`). Here it is
  selected and tested exactly like every other controller: `make_b3_env`, the same
  TP1 61-day S-SEL-v1 rule, and the same system-level B3 reward as the selection
  metric (mean over batteries of the A0 reward plus the B3 terminal term). This is
  a frozen-checkpoint model-selection step on validation days only; no test data,
  no held-out topology. It must be disclosed as such.
* Metrics are unchanged. `compare` pairs original MADDPG with corrected B3 of the
  SAME controller, so A3 has no counterpart there (listed under missing_pairs);
  report A3 from its own `scientific_episodes.csv` / `summary()`, which use the
  identical metric definitions.

## Experiment b4 (`--experiment b4` at training)

* The evaluator reads the experiment from the training manifest (`b3.experiment`; absent = b3) and builds every
  selection/test environment with that configuration (b4 batteries; validation/test SoC0 = 0.50). The environment
  contract then fails closed on any mismatch, so b3 and b4 policies cannot be evaluated in each other's setting, and
  `compare` cannot pair a b4 result with an original (b3) one (different battery constraints).
* `adjusted_energy_cost` is unchanged: raw - p_day*(E_T-E_0)/1000. It is already two-sided and is exactly the
  convention of the b4 terminal reward (environments.feasibility_env.terminal_energy_value).

## Metrics and denominators

`scientific_episodes.csv` is authoritative for the new paired report. Legacy
columns and paper summaries are retained in the other CSVs for compatibility.

`policy_requested_power_kw` is the original pre-physical request, and
`physical_limited_power_kw` is the SoC/power-limited request before DOE/backup.
The ratio is sum(abs(request-physical))/sum(abs(request)) over jointly valid
battery decisions. The material rate is count(abs(request-physical)>1 kW) / valid
battery decisions. Neither uses accepted/requested_action as a surrogate.
Missing, malformed/infinite and valid observations have separate counts. Zero
requested-power sum yields an undefined ratio (NaN), even for a zero policy.

Raw cost is sum(price[$/MWh]*max(grid_import[kW],0)*dt[hours]/1000).
There is no export revenue. Prices come from the actual evaluated environment,
not reward units or a global IEEE33 price helper. Inventory-adjusted cost is
raw_cost - day_mean_price*(terminal_kWh-initial_kWh)/1000: stored-side valuation,
not an actual recharge transaction and not an efficiency-adjusted replacement cost.
Terminal shortfall is mean(max(0,s0-sT)/(s0-smin)), with zero room handled explicitly.

Incomplete days retain undefined economics. Reports include intended, completed,
failed, incomplete and missing-record days, plus each metric's valid/missing days.
Paired metrics use dates on which BOTH complete and have the metric, reporting
coverage separately for each metric. Full individual failure statistics remain
in `full_failure_and_metric_coverage.csv`. Across-seed means weight seeds equally;
sample SD uses ddof=1. A one-seed SD is undefined, never zero. A missing seed is
flagged; this is not a completed three-seed campaign. No significance claim is made.

## Validation (no training)

```bash
PYTHONDONTWRITEBYTECODE=1 python -m unittest experiments.b3_reporting.test_reporting -v
python -m experiments.b3_reporting.evaluate --help
python -m experiments.b3_reporting.compare --help
```

The unit tests need numpy/pandas, not Torch or feeder datasets. Actual checkpoint
loading and AC rollouts require the RIT environment and are NOT locally verified.

## Future evaluation commands — DO NOT run until authorized

From repository root. `ROOT` must be a NEW results root. The controller choices
are mlp_a0, mlp_a1, mlp_a2, mlp_a3 (b3_pre_tanh only), gcn, gat, tagconv; run indices 0,1,2 (seeds 42–44).

```bash
ROOT=results/b3_matched_reviewed
CONTROLLER=mlp_a0
RUN=0
B3_TRAIN="results/b3/ieee33/b3_${CONTROLLER}/run_${RUN}/training"

python -m experiments.b3_reporting.evaluate select \
  --experiment-class b3_pre_tanh --feeder ieee33 --controller "$CONTROLLER" --run "$RUN" \
  --training-manifest "$B3_TRAIN/run_manifest.json" --checkpoint-dir "$B3_TRAIN/checkpoints" \
  --output-root "$ROOT"
```

This prints the actual new frozen selection path. Set `B3_SELECTION` to that
printed path; do not guess a checkpoint or select one using December outcomes.

```bash
python -m experiments.b3_reporting.evaluate test \
  --experiment-class b3_pre_tanh --feeder ieee33 --controller "$CONTROLLER" --run "$RUN" \
  --training-manifest "$B3_TRAIN/run_manifest.json" --selection "$B3_SELECTION" \
  --scales 1.0 1.1 1.2 1.3 --output-root "$ROOT"
```

For original policies, use the actual training manifest and existing frozen
validation selection. A0 example manifest:
`results/checkpoints/maddpg_ieee33/mlp/mlp_run_0/run_manifest.json`.
A1/A2 manifests and selections must be supplied from their actual completed runs;
their locations are not assumed. No manual editing of manifests is required.

```bash
python -m experiments.b3_reporting.evaluate test \
  --experiment-class original_maddpg --feeder ieee33 --controller "$CONTROLLER" --run "$RUN" \
  --training-manifest "$ORIGINAL_MANIFEST" --selection "$ORIGINAL_SELECTION" \
  --scales 1.0 1.1 1.2 1.3 --output-root "$ROOT"
```

Only when an original frozen selection does not already exist, use the same
`select` command with `--experiment-class original_maddpg`, its real training
manifest and `--checkpoint-dir "$ORIGINAL_CHECKPOINT_DIR"`. Do not rerun an
existing original checkpoint sweep unnecessarily. The original selection loader
still verifies the selected checkpoint bytes, validation split, and schedule.

After repeating evaluation for all intended controllers and runs:

```bash
python -m experiments.b3_reporting.compare \
  --original "$ROOT"/ieee33/original_maddpg/*/run_*/test/load_x* \
  --b3 "$ROOT"/ieee33/b3_pre_tanh/*/run_*/test/load_x* \
  --output "$ROOT/comparison_v1"
```

To inspect historical output field availability without evaluation:

```bash
python -m experiments.b3_reporting.compare --inspect-historical "$HISTORICAL_RESULT_DIR"
```

Historical CSVs lacking original requested power cannot provide the new clipping
ratio from two already-limited fields. Header inspection identifies reconstruction
candidates but does not certify values/provenance. Unknown results are excluded;
the strict evaluator provides deterministic re-evaluation of existing weights.
No automatic importer rewrites or fabricates their metadata.

## Remaining deployment checks

Run real original and B3 checkpoint loading, TP1 validation and one unseen topology
in the RIT runtime only after approval. This implementation has not accessed RIT.
If historical source/manifest checks fail, investigate that exact discrepancy;
do not remove checks to obtain a table. Relocated time-series bytes can be provided
with `--time-series`, but the original selector itself still expects its canonical
configured data path when constructing S-SEL-v1.
