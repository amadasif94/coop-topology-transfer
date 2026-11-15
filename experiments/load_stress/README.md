# IEEE33 load-stress evaluation (frozen A0 / A1 / A2 and zero action)

Post-hoc sensitivity analysis on the December test split, TP1–TP7, all 31 days, at 1.1×, 1.2× and 1.3× load.
It reuses the existing frozen selections and checkpoints. It never trains, selects or tunes anything, and its
results must not feed back into model choice. PV stays at 1.0×.

## What is (not) changed

- Only `data_scaling.load_scaling_factor` changes, and only in an in-memory deep copy. First, the original
  `primary_config` checks run on the unchanged 1.0 configuration.
- `configs/ieee33_config.py`, `primary_marl.py`, `dess_env.py`, all original evaluators, checkpoints,
  selections and the existing 1.0× results are untouched.
- Every environment goes through the original `compare_environment` against the A0 final manifest. The only
  accepted difference is `load_scaling_factor`. Interface, normalization, DOE (OFF), backup, network,
  reward, battery, 255 A ratings, tolerances, start indices and input-file bytes must all match.
- The normalizers keep the unscaled base load. Stressed load and feeder-import features therefore go beyond
  the 1.0× range, which is intended. `obs_diagnostics.csv` and the summary report by how much.
- This folder is deliberately outside `experiments/no_shared_kpi/` and `no_centralized_critic/`. Their
  `*.py` files form the source inventory embedded in every A1/A2 checkpoint.

## Files

| File | Role |
|---|---|
| `stress_env.py` | Builds the scaled environment and runs the A0 comparison with load scale as the only difference |
| `evaluate_load_stress_ieee33.py` | Evaluates one model × run × scale; enforces the gates |
| `validate_load_stress_identity.py` | 1.0× rerun vs the existing final `step_metrics.csv` (exact / 1e-8) |
| `summarize_load_stress.py` | Change vs 1.0×, policy − zero, out-of-distribution table; run-aware bootstrap |
| `test_load_stress.py` | Fast unit tests (no power flow) |
| `../../SLURM Script/load_stress_ieee33.slurm` | Array launcher (index = training run) |

Outputs go to `results/load_stress/ieee33/<label>/run_<r>/load_x{100,110,120,130}[_smoke]/`. Labels are
`a0_original`, `a1_no_shared_kpi`, `a2_no_centralized_critic` and `zero_action` (zero: `run_0` only).
Directories that aren't empty, and paths under `results/topology_generalization` or `results/ablations`,
are refused.

## Gates (per model and run, enforced in Python)

1. **identity (OPTIONAL):** a 1.0× rerun, then `validate_load_stress_identity` on all of TP1–TP7, 31 × 96 steps.
   If no identity report exists, stress runs still go ahead. Their manifest then records
   `gates.identity = {performed: false, status: NOT_PERFORMED}` and `identity_verified: false`, and the
   summary marks those runs `identity_check = NOT_PERFORMED`. An identity report that exists but isn't a
   valid PASS (FAIL, a changed 1.0× manifest, or a different checkpoint) still blocks the stress runs.
   For zero the check is NOT_APPLICABLE as a gate.
2. **smoke (optional):** 1.3×, TP1, first 2 days. Writes `smoke_check.json`.
3. **full:** 1.1, 1.2, 1.3. Records the smoke status (`NOT_PERFORMED` when absent) but does not require it
   (changed 2026-10-29; earlier runs required a smoke PASS).

The summary's "change vs 1.0×" and out-of-distribution tables use the 1.0× runs from this tree
(`load_x100`). Without the `identity` stage those tables are empty. The 1.0× runs can be submitted at the
same time as the stress runs.

## Job map

Use the existing A1/A2 job map (`prepare_job_map.py`). The evaluator reads
`runs[<r>].baseline_evaluation_manifest` (A0 final), `baseline_training_manifest` and `selections[<variant>]`.
The A0 selection is taken from the A0 final manifest and re-verified. A1/A2 references default to
`results/ablations/ieee33/<variant>/run_<r>/final_test/run_manifest.json`.

## Run order on RIT

```bash
mkdir -p results/load_stress/ieee33/slurm_logs
JM=/abs/path/job_map.json
sbatch --array=0-2 "SLURM Script/load_stress_ieee33.slurm" preflight a0 $JM
sbatch --array=0-2 "SLURM Script/load_stress_ieee33.slurm" identity a0 $JM   # optional; likewise a1, a2; zero --array=0
sbatch --array=0-2 "SLURM Script/load_stress_ieee33.slurm" smoke a0 $JM
sbatch --array=0-2 "SLURM Script/load_stress_ieee33.slurm" full  a0 $JM      # optional 4th arg, e.g. "1.1"
sbatch --array=0   "SLURM Script/load_stress_ieee33.slurm" summary - $JM
```

Each full stage runs 3 × 217 episodes per array task (each scale is a separate output directory). Run the
identity stage on the same partition and device setup as the original finals (the launcher matches the A1
eval launcher: `dgx`, no GPU requested). A CPU/GPU difference in actor inference would show up as an
identity FAIL; it would not be silently tolerated.

## Disclosure

Report this as a post-hoc load-stress sensitivity analysis of frozen policies on the already used December
test split, with normalizers fixed at the 1.0× reference.
