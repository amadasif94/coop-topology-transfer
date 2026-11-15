# IEEE69 load-stress evaluation (frozen A0 / A1 / A2)

Post-hoc sensitivity analysis on the December 2018 test split: TP1–TP10, 31 days, runs 0–2, load 1.1× / 1.2× /
1.3×, PV 1.0×. It uses the existing frozen IEEE69 selections only: no retraining, selection or tuning, and no
zero-action baseline. 3 controllers × 3 runs × 3 scales × 10 topologies × 31 days = **8,370 daily episodes**.

## How references are matched

- **Job map:** `experiments/no_shared_kpi/ieee69/cluster_job_map_final.json` (on RIT). It is verified by the
  original `ieee69.protocol.verify_map`: the sha256 of the training manifest and of every A0 final manifest
  must match.
- **A0 references:** `runs[r].baseline_evaluation_manifests` lists one manifest per A0 topology group
  (e.g. TP1–TP7, TP8–TP9, TP10). They are read by the original `ieee69.final_evaluation.references()`:
  - each group's environments are deduplicated;
  - each topology must come from exactly one group, with 31 × 96 complete December rows at 15-minute steps;
  - the archived network snapshot must be hash-matched and must equal the topology definition;
  - the plan is start index plus seed = manifest seed + episode id;
  - all groups must share one checkpoint.
- **Extra checks added by the evaluator:**
  - all ten topologies must be present;
  - all A0 groups must share one frozen selection;
  - the frozen selection file is unchanged.
- **Per-topology environment:** uses that group's archived A0 network bytes. Start indices and timestamps must
  match the A0 reference.
- **A1/A2:** `runs[r].selections[<variant>]`, plus the reference
  `results/ablations/ieee69/<variant>/run_<r>/final_test/run_manifest.json`. Its selection, checkpoint,
  ablation, run and per-topology episode plans must agree; a topology missing from it is recorded.
- **Checkpoint loading:** A0 uses the original `MLPMADDPG.load`, with provenance. A1/A2 use the IEEE69
  test-mode loader in `ieee69/models.py`: sidecar sha, release-pinned source inventory, provenance.
- **Environment comparison:** `ieee69.protocol.compare_environment` against the A0 record for that topology.
  The only accepted difference is `data_scaling.load_scaling_factor`. Interface, 9 DESS buses, normalization
  (unscaled base load), battery, reward, DOE OFF, directional backup, ratings and input bytes are unchanged.

## Optional checks (never blocking)

- **identity:** a 1.0× rerun, then `validate_load_stress_identity_ieee69`. The report is `PASS`, `PASS_PARTIAL`
  (a topology is absent from the reference) or `FAIL`.
  - With no report, stress manifests record `gates.identity.status = NOT_PERFORMED` and `identity_verified: false`.
  - An existing `FAIL` or stale report is refused rather than ignored. Delete it deliberately to proceed.
- **pilot:** 1.3×, TP1, 2 days, writing `pilot_check.json`. Full runs only record its status.

## Files

- `stress_env_ieee69.py`
- `evaluate_load_stress_ieee69.py`
- `validate_load_stress_identity_ieee69.py`
- `summarize_load_stress_ieee69.py`
- `test_load_stress_ieee69.py`
- `../../SLURM Script/load_stress_ieee69.slurm`: CPU `batch` partition, node-local TMPDIR, one thread per
  numerical library

The new files reuse `stress_env.py`, `evaluate_load_stress_ieee33.py` and `summarize_load_stress.py`, which
are not modified. Outputs go only under `results/load_stress/ieee69/<label>/run_<r>/load_x<pct>[_pilot]/`.
Directories that aren't empty are refused.
