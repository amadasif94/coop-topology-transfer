# IEEE69 A1/A2 extension — validation pending on RIT

This is an additive extension of the existing IEEE33 ablation workflow. No
existing source, launchers, maps, checkpoints, datasets, or results were edited.
Both variants use the original IEEE69 environment and shared team reward.

## Why the Python entry points are nested

IEEE33 checkpoint identity hashes every `*.py` directly in both ablation folders.
Adding top-level IEEE69 Python files would change that identity during the six
active IEEE33 runs. Therefore all new Python modules live under `ieee69/`.
The four new SLURM launchers can remain at the requested top-level paths because
the checkpoint identity does not include SLURM files. Transfer ONLY these new
paths; do not synchronize/delete or overwrite the active IEEE33 folders wholesale.

The existing runner/model constructors are reused through private `FunctionType`
bindings, the same approach used by the existing ablation trainer factory.
IEEE69 bindings replace feeder-specific dependencies in private dictionaries;
they do not monkey-patch imported modules. A1 inherits the original centralized
update; A2 inherits the existing independent update. Existing schema text
`ieee33-mlp-ablation/1` is retained as a serialization format identifier; the
mandatory nine-agent specification and source identity distinguish IEEE69.

## Verified A0 evidence

All three local manifests under
`results/checkpoints/maddpg_ieee69/mlp/mlp_run_{0,1,2}/run_manifest.json`
are finished IEEE69 MLP runs, seeds 42/43/44, total_steps=200000. The 200k
checkpoint files exist. Weights were not loaded locally (PyTorch unavailable).
Settings verified from these manifests and the A0 IEEE69 training script:

- TP1 training and periodic validation; nine agents at zero-based buses
  `[13,15,17,19,21,23,25,26,64]` (one-based 14/16/18/20/22/24/26/27/65).
- 10 features; A1 retains indices 0–6 for actors, removes 7–9 (grid import,
  voltage deviation, curtailment). Full joint replay and centralized critics
  retain all features. A1 actor/critic dimensions: **7/99**; A2: **10/11**.
- Batteries: **1000 kWh, 250 kW** charge/discharge, efficiencies .95,
  SOC .1–.9, initial SOC .5. These are the actual baseline, not older 75 kW values.
- DOE **OFF**, directional backup ON, unchanged reward/physics. The primary
  environment factory applies the actual A0 DOE setting; direct config imports
  alone do not represent the effective environment. Manifest comparison checks
  actual instantiated values, input-file hashes, scaling, normalization,
  current ratings, reward and battery constraints before training/evaluation.
- Warmup 10000; replay 100000; batch 128; actor/critic LR 5e-5/5e-4;
  gamma .99; tau .005; noise .05; save/eval interval 5000. Original trainer
  counts warmup inside the 200000 total. Scheduled checkpoints are 15000 through
  200000 (38 candidates), plus the separately named final checkpoint.

## Validation and selection protocol

Source: `run_eval_mlp_ieee69.py`, `trainer_mlp.py`,
`utility/periodic_validation.py`, `utility/validation_schedule.py`, and
`scripts/select_checkpoint.py` (all imported/read only).

Periodic validation is **S-PER-v1**, eight fixed IEEE69 dates:
Oct 3/11/17/25 and Nov 3/5/11/13. The stored 200k validation manifests confirm
this, despite the CLI's legacy `eval_episodes=10` setting. The original trainer
is reused, so it retains the eight-day behavior.

Checkpoint selection uses **S-SEL-v1**, all 61 Oct–Nov days on TP1, 96 steps
each. Reset seed = 20261000 + chronological validation-day index. The original
selector requires completed, failure-free and safety-eligible results, maximizes
mean episode `total_reward_mean`, and breaks ties toward the earliest step.
Its 2000-resample reliability bootstrap and seed 20261021 remain unchanged.
No December or TP2+ data enters selection. `sweep` evaluates the 38 checkpoints
and then freezes a selection; `select` can select from an already completed sweep.
These procedures are verified from IEEE69-aware source; the actual A0 frozen
selection records are still required to verify which candidates it evaluated.

Final evaluation uses the original IEEE69 TP1–TP7 topology builder/evaluator,
December data, frozen ablation weights, baseline seed and full daily start list.
The A0 final manifest must confirm 31 days x 96 steps and matching input hashes.
Final evaluation is blocked without it. This is pending metadata verification,
not a claim that the original research lacks these records.

Original cost, safety, DOE/backup, clipping, failure, and action-stage CSV fields
are retained through the existing evaluator. Additional terminal SOC/energy
uses IEEE69 prices and actual executed states. Adjusted cost is
`raw_cost - mean_daily_price_per_MWh * (terminal_kWh - initial_kWh) / 1000`.
Stored-side energy is used with no extra efficiency factor; SOC already includes
physical efficiency. Invalid episodes retain missing adjusted costs.

## Transfer and local validation limits

Transfer the two new `ieee69/` packages plus the four `*_ieee69.slurm` files.
Use WinSCP binary transfer to retain LF shell scripts and checkpoint source hashes.
Do NOT overwrite the original `cluster_job_map.json` or any IEEE33 script.
The local map is an audit artifact; generate the production map on RIT.

Local static/model test status is saved in `validation_results.txt`.
Missing local PyTorch, Gymnasium and SciPy block real model/AC smoke tests.
Local Bash is blocked by WSL requiring an update. No SLURM job was submitted.
The preservation hash test captures this workspace's originals; if authoritative
RIT files differ, report the specific differences rather than overwriting them.

## RIT commands — run from repository root

```bash
cd /network/rit/lab/dahlin_lab/amad/graph_marl_dess
source /network/rit/lab/dahlin_lab/anaconda3/etc/profile.d/conda.sh
conda activate /network/rit/lab/dahlin_lab/amad/amad_thesis/env/thesis_conda
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 BLIS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1

bash -n experiments/no_shared_kpi/ieee69/launch.sh
for variant in no_shared_kpi no_centralized_critic; do
  bash -n "experiments/$variant/train_${variant}_ieee69.slurm" || exit 1
  bash -n "experiments/$variant/eval_${variant}_ieee69.slurm" || exit 1
done

python -m experiments.no_shared_kpi.ieee69.prepare_job_map \
  --a0-training-root results/checkpoints/maddpg_ieee69/mlp \
  --output experiments/no_shared_kpi/ieee69/cluster_job_map.json

python -m experiments.no_shared_kpi.ieee69.validation --require-runtime
```

On an allocated CPU node (not a full training job), run both real smoke tests:

```bash
STAMP=$(date +%Y%m%d_%H%M%S)
for variant in no_shared_kpi no_centralized_critic; do
  python -m experiments.no_shared_kpi.ieee69.validation \
    --require-runtime --variant "$variant" --run 0 \
    --baseline-training-manifest results/checkpoints/maddpg_ieee69/mlp/mlp_run_0/run_manifest.json \
    --real-smoke --output "results/ablations/ieee69/$variant/run_0/smoke_$STAMP" || break
done
```

Each smoke checks 16 transitions, 8 optimizer updates, proposed-action replay,
executed rewards/next observations, unchanged reward responses for equal actions,
checkpoint save/load, a 96-step validation evaluation and energy-accounting CSVs.
Zero PF failures are required. Dimensions, KPI masking and independent-critic
locality are checked by the model tests. No full training is run by validation.

## Scheduler diagnostic and full training (user submits only after validation)

The current working IEEE33 scripts receive resources through submission, unlike
the older packaged headers. IEEE69 preserves that behavior, positional stage/map,
plain `--export=ALL`, DGX/freetier, original Conda and filesystem setup. Full
training requests the A0 IEEE69 resources: 32 CPUs, 120G, one GPU, one day.
Both variants use the shared A1 **IEEE69** dispatcher, never a nonexistent A2 one.

```bash
mkdir -p results/ablations/ieee69/slurm_logs
JOB_MAP="$PWD/experiments/no_shared_kpi/ieee69/cluster_job_map.json"
SBATCH_BIN=$(type -P sbatch)

env -u SBATCH_GET_USER_ENV -u SBATCH_EXPORT -u SLURM_EXPORT_ENV \
  "$SBATCH_BIN" --export=ALL --array=0 --cpus-per-task=1 --mem=2G --time=00:05:00 \
  experiments/no_shared_kpi/train_no_shared_kpi_ieee69.slurm launch_check "$JOB_MAP"

# Run only after the diagnostic and BOTH real smoke tests pass:
env -u SBATCH_GET_USER_ENV -u SBATCH_EXPORT -u SLURM_EXPORT_ENV \
  "$SBATCH_BIN" --export=ALL --array=0-2 --cpus-per-task=32 --mem=120G \
  --gres=gpu:1 --time=1-00:00:00 \
  experiments/no_shared_kpi/train_no_shared_kpi_ieee69.slurm train "$JOB_MAP"

env -u SBATCH_GET_USER_ENV -u SBATCH_EXPORT -u SLURM_EXPORT_ENV \
  "$SBATCH_BIN" --export=ALL --array=0-2 --cpus-per-task=32 --mem=120G \
  --gres=gpu:1 --time=1-00:00:00 \
  experiments/no_centralized_critic/train_no_centralized_critic_ieee69.slurm train "$JOB_MAP"
```

## Future selection and final evaluation

After training finishes, submit each three-run sweep. DGX resources below match
the training allocation; monitor sweep duration because 38 x 61 full episodes
per run may need a longer site-approved wall time. No scheduler success is claimed.

```bash
for variant in no_shared_kpi no_centralized_critic; do
  env -u SBATCH_GET_USER_ENV -u SBATCH_EXPORT -u SLURM_EXPORT_ENV \
    "$SBATCH_BIN" --export=ALL --array=0-2 --cpus-per-task=32 --mem=120G \
    --gres=gpu:1 --time=1-00:00:00 \
    "experiments/$variant/eval_${variant}_ieee69.slurm" sweep "$JOB_MAP"
done
```

`sweep` includes freezing the best checkpoint. Do not resubmit `select` if it
already succeeded. If only freezing remains after a completed sweep, use the
same command with stage `select`, or invoke the shared dispatcher directly:

```bash
python -m experiments.no_shared_kpi.ieee69.slurm_dispatch \
  --variant no_shared_kpi --stage select --run 0 --job-map "$JOB_MAP"
```

Once A0 final manifests are accessible, generate a NEW final map (never overwrite
the training map). Discovery is by IEEE69 manifest content, not guessed names:

```bash
python -m experiments.no_shared_kpi.ieee69.prepare_job_map \
  --a0-training-root results/checkpoints/maddpg_ieee69/mlp \
  --a0-evaluation-root results/topology_generalization \
  --discover-ablation-selections \
  --output experiments/no_shared_kpi/ieee69/cluster_job_map_final.json

FINAL_MAP="$PWD/experiments/no_shared_kpi/ieee69/cluster_job_map_final.json"
for variant in no_shared_kpi no_centralized_critic; do
  env -u SBATCH_GET_USER_ENV -u SBATCH_EXPORT -u SLURM_EXPORT_ENV \
    "$SBATCH_BIN" --export=ALL --array=0-2 --cpus-per-task=32 --mem=120G \
    --gres=gpu:1 --time=1-00:00:00 \
    "experiments/$variant/eval_${variant}_ieee69.slurm" evaluate "$FINAL_MAP"
done
```

All new results are under `results/ablations/ieee69/<variant>/run_<id>/`:
`training/`, `selection_sweep/`, `selections/`, `final_test/`. Existing nonempty
run directories and selection records are rejected, not overwritten. Interrupted
sweeps currently require reviewing completed directories before resubmission;
there is no newly introduced automatic resume behavior.

## Original IEEE69 records still needed locally

For each run 0/1/2, supply the **actual** A0 final-evaluation directory containing
`run_manifest.json`, `episode_summary.csv`, `aggregate_summary.csv`,
`step_metrics.csv`, and `network_inputs/`. Exact directory names are not yet
verified. Manifest JSON must identify IEEE69/MLP, TP1–TP7, final December protocol,
episode count, seeds/start indices, selected-checkpoint identity, and input hashes.

Also supply each actual frozen A0 `selection.json` (schema
`gmarl-checkpoint-selection/1`, feeder `ieee69`, model `mlp`, run `run_N`, TP1/val,
S-SEL-v1 schedule, candidate list, selected path/hash/step and metric), plus its
referenced selection-sweep manifests and summary CSVs. Expected selection parent
is `results/selections/ieee69/mlp/` by the original selector convention, but actual
versioned paths must be located on RIT. Do not generate replacement A0 selections.
The available 200k checkpoints do not establish which checkpoint was selected.

These records are needed to finish historical selection/final-evaluation parity
verification. Training manifests already suffice to implement training and smoke
validation; the extension is not certified ready for full research runs locally.
