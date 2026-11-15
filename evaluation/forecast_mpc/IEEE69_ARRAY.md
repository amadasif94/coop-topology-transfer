# IEEE69 December F-MPC on the batch cluster

Run from the repository root on the independent HPC scheduler. The wrapper calls the
existing `evaluation.forecast_mpc.run.main` once per topology and forwards the completed
1.3× reference's protocol, solver options, price mode, and saved December forecast directory.
It does not fit forecasts. The original result tree is read-only.

## Verify before production

```bash
cd /network/rit/lab/dahlin_lab/amad/graph_marl_dess
source /network/rit/lab/dahlin_lab/anaconda3/etc/profile.d/conda.sh
conda activate /network/rit/lab/dahlin_lab/amad/amad_thesis/env/thesis_conda
python -B -m unittest tests.test_fmpc_ieee69_array -v
bash -n 'SLURM Script/fmpc_ieee69_array.slurm'
sbatch 'SLURM Script/fmpc_ieee69_array.slurm' smoke
```

The smoke is one complete day (96 real solves of shrinking horizon), December 1 on TP1
at 1.3×. It streams the existing TP1/day rows from the completed reference and checks all
five comparable CSVs, including requests, physical dispatch, SoC, costs, feasibility,
solver outcomes and fallbacks. Wall-clock fields are excluded. Numeric comparisons use
absolute tolerance 1e-8 and relative tolerance 1e-9; strings and missing values must agree.
The sixth CSV is a topology aggregate over different day counts and is checked for coverage
only. Review `results/fmpc_ieee69_topology_tasks/smoke/TP1/tables/reference_comparison.json`.
An absent or failed comparison blocks full tasks. A runtime or provenance mismatch should
be investigated before using either campaign's numbers.

## Run only separately approved, outstanding work

For example, **if** all ten topologies at 1.0× still need independent computation:

```bash
job=$(sbatch --parsable --array=1-10%2 'SLURM Script/fmpc_ieee69_array.slurm' task 100)
sbatch --dependency="afterok:${job%%;*}" 'SLURM Script/fmpc_ieee69_array.slurm' aggregate 100
```

100, 110, 120, and 130 denote load percent. The array index is the topology number.
Choose an array subset for outstanding tasks (e.g. `--array=3,5,10%2`); aggregation
requires all ten **published shards for that scale**. A completed legacy scale can
remain in its original location. The workflow does not import completion messages from
ongoing legacy jobs and does not need a new full 130 run. Slurm output goes to
`FMPC69_<job>_<task>.out/.err` in the submission directory.

## Persistence and recovery

Results are under `results/fmpc_ieee69_topology_tasks/load_x<PCT>/TP<N>/tables/`.
Each task uses its own stable hidden attempt directory. After validation it hard-links
the results into a private publication directory and atomically renames that directory.
Keep the attempt directory: the original run manifest references its input snapshots.
The batch filesystem must support POSIX advisory locks and same-filesystem directory
rename. If hard links are unavailable, publication uses checksum-verified copies.
A timeout leaves an unpublished attempt and releases its task lock.
Rerun that topology from its first day. Other completed topologies are hash-verified and
reused. There is no within-topology restart. Failed AC episodes are retained if their
recorded dispatch attempts and solver rows agree. An optimizer fallback is a recorded
controller outcome and does not invalidate a shard.

Aggregation streams the six original CSV tables in numeric topology order without
rounding or re-averaging topology statistics. Columns that occur only in some shards
are retained. Output is published at `load_x<PCT>/aggregate/tables/`; its manifest
identifies the ten exact completion records. Per-topology solver summaries and
environment manifests remain in the shard directories. Invalid or incomplete publications
raise an error rather than being silently replaced.
