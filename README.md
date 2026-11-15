# The Role of Cooperation in Zero-Shot Topology Transfer for Multi-Agent Battery Storage Control

**Official code, pretrained controllers, and experimental results accompanying the research paper**

**Muhammad Amad Asif, Huraiz Ahmed Wasiq, and Nathan Dahlin**

*The Role of Cooperation in Zero-Shot Topology Transfer for Multi-Agent Battery Storage Control.* Submitted to *Applied Energy*, 2026.

**[Download pretrained MADDPG controllers](https://github.com/amadasif94/coop-topology-transfer/releases/tag/v1.0.0-checkpoints)** | **[All releases](https://github.com/amadasif94/coop-topology-transfer/releases)**

---

## Overview

This repository contains the implementation, experimental configurations, selected trained controllers, and evaluation results for a systematic study of **cooperation and zero-shot topology transfer in multi-agent battery energy storage control**.

Distributed energy storage systems (DESS) are coordinated using Multi-Agent Deep Deterministic Policy Gradient (MADDPG) in reconfigurable radial distribution networks.

Controllers are trained on a single feeder topology (TP1) and subsequently evaluated on previously unseen network reconfigurations **without retraining or policy adaptation**.

The study examines three complementary dimensions of cooperation:

1. **Information sharing:** Whether battery agents receive feeder-level key performance indicators (KPIs).
2. **Critic architecture:** Whether learning uses centralized or independent critics.
3. **Reward sharing:** Whether agents are trained with shared and local reward components or exclusively local rewards.

Four controller variants (A0–A3) progressively remove these cooperation mechanisms to isolate their effects on zero-shot topology transfer.

**Research question:** Does cooperation determine whether multi-agent battery controllers transfer to unseen feeder topologies, or primarily influence the economic value retained after transfer?

**Main finding:** All four variants demonstrate zero-shot transfer to unseen feeder topologies. Cooperation primarily determines how much economic value is retained after transfer, rather than whether transfer occurs.

---

## Controller Variants

| Variant | Actor observations | Critic architecture | Training reward |
|---|---|---|---|
| **A0 — Full cooperation** | Local measurements + feeder KPIs | Centralized | Shared + local |
| **A1 — No shared KPIs** | Local measurements only | Centralized | Shared + local |
| **A2 — Independent critics** | Local measurements + feeder KPIs | Independent | Shared + local |
| **A3 — Full independence** | Local measurements only | Independent | Local only |

### Cooperation mechanisms

**A0 — Full cooperation**

The baseline MADDPG controller uses feeder-level KPI observations, centralized critics, and shared reward information.

**A1 — No shared KPIs**

Feeder KPIs are removed from the actors' observations. Centralized critics and the shared reward structure are retained. During training, the centralized critics continue to receive the full observation information.

**A2 — Independent critics**

Centralized critics are replaced by independent critics. Feeder-level KPI observations and the shared reward structure are retained.

**A3 — Full independence**

Each battery agent uses local observations, an independent critic, and a locally defined training reward.

A3 uses `environments/local_reward_env.py` during training. Checkpoint selection and final evaluation use the common team-reward environment, ensuring that evaluation rewards are comparable across all four variants.

### Shared control infrastructure

All controllers use the same underlying feeder environment, battery dynamics, operational constraints, SOC-feasible action mapping, and network safety mechanism.

This controlled experimental design isolates the effects of cooperation from changes in the physical simulation and action-feasibility handling.

---

## Experimental Setup

### Distribution networks

Experiments are conducted on two radial distribution networks.

| Property | IEEE 33-bus | IEEE 69-bus |
|---|---|---|
| Number of battery agents | 5 | 9 |
| Energy capacity per battery | 700 kWh | 850 kWh |
| Battery power limits | ±140 kW | ±170 kW |
| Training topology | TP1 | TP1 |
| Unseen evaluation topologies | TP2–TP7 | TP2–TP10 |
| Number of unseen topologies | 6 | 9 |
| Independent training seeds | 42, 43, 44 | 42, 43, 44 |
| Control interval | 15 minutes | 15 minutes |
| Final test period | December 2018 | December 2018 |

### Battery operating parameters

The battery configurations include:

| Parameter | Value |
|---|---|
| Charging efficiency | 0.95 |
| Discharging efficiency | 0.95 |
| Minimum SOC | 0.10 |
| Maximum SOC | 0.90 |
| Initial SOC for validation/testing | 0.50 |
| Control interval | 15 minutes |
| Decisions per day | 96 |

The feeder reconfiguration cases and tie-line impedances are provided in `data/network/` and documented in the accompanying manuscript (see the tie-line impedance table in the paper).

The experimental tie-line parameters should be used rather than substituting values from other published versions of the IEEE test feeders.

### Dataset partitioning

A chronological partition of the 2018 dataset is used.

| Period | Purpose |
|---|---|
| January–September 2018 | Training |
| October–November 2018 | Validation |
| December 2018 | Final testing |

The final test period comprises 31 days.

### Topology-transfer evaluation

All four controllers are trained exclusively on TP1.

Checkpoint selection is performed using TP1 validation data. The selected policies are then evaluated without retraining on the December test period across the training topology and unseen feeder reconfigurations.

- **IEEE33:** TP1–TP7, including six unseen topologies.
- **IEEE69:** TP1–TP10, including nine unseen topologies.

TP1 is retained as an in-distribution evaluation reference.

### Load-stress evaluation

The selected policies are additionally tested under increasing demand levels.

| Scenario | Load multiplier |
|---|---|
| Nominal | 1.0× |
| Stress 1 | 1.1× |
| Stress 2 | 1.2× |
| Stress 3 | 1.3× |

Demand scaling is applied without retraining the controllers.

PV generation and electricity-price profiles are not scaled.

### SOC feasibility and network safety

Battery actions are implemented using the SOC-feasible affine action mapping (`affine_v2`), which maps normalized policy outputs to physically admissible charging or discharging powers.

The feeder environment uses an AC power-flow implementation to calculate operating conditions and assess network constraints.

A directional backup safety mechanism attempts to correct voltage and branch-current violations through adjustments to battery dispatch.

These physical constraints and safety mechanisms are applied consistently across the four cooperation variants.

---

## Training Configuration

The maintained A0–A3 experiments use the following common learning configuration.

| Parameter | Value |
|---|---|
| Reinforcement learning algorithm | MADDPG |
| Controller architecture | MLP |
| Training steps | 200,000 |
| Random exploration warm-up | 10,000 steps |
| Replay buffer capacity | 100,000 |
| Batch size | 128 |
| Actor learning rate | 5 × 10⁻⁵ |
| Critic learning rate | 5 × 10⁻⁴ |
| Discount factor (γ) | 1.0 |
| Target-network update coefficient (τ) | 0.005 |
| Exploration noise standard deviation | 0.05 |
| Hidden-layer dimensions | 256 × 256 |
| Random seeds | 42, 43, 44 |

The cooperation variants differ in their information sharing, critic structure, and reward formulation, while retaining the common training protocol.

---

## Repository Structure

The repository is organized as follows:

| Directory | Description |
|---|---|
| `configs/` | Feeder configurations, battery parameters, and experimental settings |
| `data/network/` | IEEE 33-bus and IEEE 69-bus network data and topology definitions |
| `data/time_series/` | Processed demand, PV, and electricity-price time series |
| `environments/` | Battery dynamics, AC power flow, SOC-feasible action mapping, network safety, and reward environments |
| `models/` | MADDPG actor/critic implementations and controller interfaces |
| `training/` | Maintained training pipeline and experience replay |
| `evaluation/` | Checkpoint selection, testing, optimization benchmarks, and statistical analyses |
| `experiments/` | Cooperation-variant implementations and supporting experimental utilities |
| `utility/` | Power-flow utilities, data handling, experiment provenance, and run manifests |
| `scripts/` | Data-processing and experimental support scripts |
| `tests/` | Implementation and evaluation tests |
| `SLURM Script/` | HPC job submission scripts |
| `results/` | Selected experimental results, evaluation records, forecasts, and final analyses |
| `requirements.txt` | Python software dependencies |

The maintained training and evaluation entry points are:

```text
training/train.py
evaluation/evaluate.py
```

Some shared modules retain historical graph-based controller implementations for compatibility and experiment provenance. The four MADDPG cooperation variants are the controllers studied in the accompanying manuscript.

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/amadasif94/coop-topology-transfer.git
cd coop-topology-transfer
```

### 2. Create a Python environment

The experiments were conducted using Python **3.10.20**.

```bash
python3.10 -m venv .venv
source .venv/bin/activate
```

### 3. Install the required packages

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The core MADDPG experiments use PyTorch, NumPy, pandas, SciPy, and Gymnasium.

The forecasting and Gurobi-based optimization workflows require additional packages, as described below.

### Verified software versions

The following versions were confirmed using the RIT research environment and the saved experiment records.

| Package | Version |
|---|---|
| Python | **3.10.20** |
| PyTorch | **2.9.1** (RIT CUDA build: `2.9.1+cu128`) |
| NumPy | **2.2.5** |
| pandas | **2.3.3** |
| SciPy | **1.15.3** |
| Gymnasium | **0.29.1** |
| Matplotlib | **3.10.8** |
| PyArrow | **20.0.0** |
| AutoGluon Tabular | **1.5.0** |
| Gurobi / gurobipy | **13.0.1** |

AutoGluon 1.5.0 was verified against the saved training records, forecast-generation records, and predictor version files for all 66 IEEE33 and 138 IEEE69 forecasting targets.

Gurobi 13.0.1 was verified from the recorded solver versions in all 28 IEEE33 and 40 IEEE69 final F-MPC task records.

For F-MPC, install the appropriate Gurobi Python package and obtain a valid license:

```bash
python -m pip install gurobipy==13.0.1
```

Regenerating the forecasting models additionally requires:

```bash
python -m pip install autogluon.tabular==1.5.0
```

The main A0–A3 MADDPG controllers do not require AutoGluon or Gurobi for training and policy evaluation.

A compatible CPU or GPU build of PyTorch should be selected for the execution platform.

---

## Reproducing the Experiments

The experimental workflow consists of:

1. Training the MADDPG variants on TP1.
2. Selecting checkpoints using the validation period.
3. Evaluating zero-shot topology transfer.
4. Computing benchmark and statistical comparisons.

Run the following commands from the repository root.

Use a fresh output directory when reproducing training experiments.

### 1. Train a MADDPG controller

Example: Train A0 on IEEE33 using seed index 0, corresponding to random seed 42.

```bash
python -u -m training.train \
  --feeder ieee33 \
  --controller mlp \
  --variant A0 \
  --seed 0 \
  --output-root results/repro_affine_v2
```

The maintained training interface supports all four variants through the same entry point.

| Argument | Supported values |
|---|---|
| `--feeder` | `ieee33`, `ieee69` |
| `--controller` | `mlp` |
| `--variant` | `A0`, `A1`, `A2`, `A3` |
| `--seed` | `0`, `1`, `2` |

The seed index maps to the actual random seed as follows:

| CLI seed index | Random seed |
|---|---|
| 0 | 42 |
| 1 | 43 |
| 2 | 44 |

For example, to train A3 on IEEE69 using seed index 1:

```bash
python -u -m training.train \
  --feeder ieee69 \
  --controller mlp \
  --variant A3 \
  --seed 1 \
  --output-root results/repro_affine_v2
```

A3's local-reward environment is selected automatically through the controller configuration.

### 2. Select a trained checkpoint

After training, run checkpoint selection using TP1 validation data:

```bash
python -u -m evaluation.evaluate select \
  --feeder ieee33 \
  --controller mlp \
  --variant A0 \
  --seed 0 \
  --output-root results/repro_affine_v2
```

Checkpoint selection evaluates saved training checkpoints on the validation period and records the selected model and its associated provenance.

The selection records include the training step, checkpoint path, and SHA-256 checksum.

### 3. Evaluate topology transfer

Evaluate the selected controller without retraining:

```bash
python -u -m evaluation.evaluate test \
  --feeder ieee33 \
  --controller mlp \
  --variant A0 \
  --seed 0 \
  --scales 1.0 1.1 1.2 1.3 \
  --output-root results/repro_affine_v2
```

The maintained evaluator processes every configured topology for the selected feeder:

- IEEE33: TP1–TP7
- IEEE69: TP1–TP10

Evaluation covers all 31 December test days.

Change `--variant`, `--seed`, and `--feeder` to reproduce the remaining controller experiments.

**Important:** These commands reproduce the workflow using checkpoints and manifests generated locally by the training and selection stages. The original published checkpoint-selection records contain absolute RIT filesystem paths. Evaluating the downloaded pretrained checkpoints on another machine requires resolving those paths to the extracted model files while retaining the recorded SHA-256 integrity checks.

### 4. Evaluate the Idle baseline

```bash
python -u -m evaluation.evaluate zero \
  --feeder ieee33 \
  --scales 1.0 1.1 1.2 1.3 \
  --output-root results/repro_affine_v2
```

The internal command is named `zero`; this benchmark is reported as **Idle** in the manuscript.

---

## Benchmark Controllers

The MADDPG variants are compared with optimization-based and rule-based reference controllers.

| Benchmark | Description | Solver |
|---|---|---|
| **PF-MPC** | Perfect-foresight model predictive control | SciPy / HiGHS |
| **F-MPC** | Forecast-based receding-horizon model predictive control | Gurobi |
| **RBC** | Price-quantile rule-based battery control | Rule-based |
| **Idle** | Batteries remain at 0 kW | None |

### Perfect-foresight MPC (PF-MPC)

```bash
python -u -m evaluation.oracle.run \
  --feeder ieee33 \
  --output-root results/repro_affine_v2
```

PF-MPC uses a perfect-foresight optimization formulation with a LinDistFlow network approximation and SciPy/HiGHS.

### Forecast-based MPC (F-MPC)

Example F-MPC task:

```bash
python -u -m evaluation.forecast_mpc.gurobi_run task \
  --feeder ieee33 \
  --scale 1.0 \
  --topology TP1 \
  --output-root results/fmpc_gurobi
```

F-MPC uses Gurobi 13.0.1 and frozen forecast inputs generated by the forecasting pipeline.

Reproducing the full benchmark requires the relevant feeder, topology, and load-factor combinations.

### Rule-based control (RBC)

The manuscript reports this benchmark as **RBC**. The implementation uses the `pq_rbc` module.

Freeze the rule-based parameters:

```bash
python -u -m evaluation.rule_based.pq_rbc freeze \
  --feeder ieee33 \
  --output-root results/repro_affine_v2
```

Evaluate on the test period:

```bash
python -u -m evaluation.rule_based.pq_rbc run \
  --feeder ieee33 \
  --split test \
  --output-root results/repro_affine_v2
```

### Idle

The Idle benchmark sets battery dispatch to zero and is evaluated using:

```bash
python -u -m evaluation.evaluate zero \
  --feeder ieee33 \
  --scales 1.0 1.1 1.2 1.3 \
  --output-root results/repro_affine_v2
```

---

## Forecast-Based MPC Data

The F-MPC benchmark uses forecasts produced by an AutoGluon Tabular forecasting pipeline.

One `TabularPredictor` is trained for each forecast target using historical observations, calendar information, and electricity-price features.

The forecasting procedure uses expanding training windows and generates December forecasts using models trained on data available through November 2018.

The associated source modules include:

```text
evaluation/forecast_mpc/train.py
evaluation/forecast_mpc/generate.py
evaluation/forecast_mpc/gurobi_run.py
```

Frozen December forecast arrays are stored in the feeder-specific results directories:

```text
results/soc_feasible_affine_v2/<feeder>/forecast_mpc/forecasts/dec/
```

The forecast artifacts include:

- `.npz` arrays containing the generated forecasts.
- `.json` records containing forecast metadata and provenance.
- Dataset integrity information used by the F-MPC evaluation pipeline.

The frozen forecast artifacts permit reuse of the generated predictions without retraining the underlying forecasting models.

Regenerating identical forecasts requires the original saved AutoGluon predictors and matching input data. Retraining a predictor may produce different forecasts.

---

## Statistical Analysis

The repository includes statistical and diagnostic analysis utilities for evaluating controller performance.

These are implemented under:

```text
evaluation/final_analyses/
```

### Bootstrap analysis

```bash
python -u -m evaluation.final_analyses.bootstrap \
  --feeder ieee33 \
  --results-root results/soc_feasible_affine_v2 \
  --out-root results/final_paper_analyses
```

### Battery SOC diagnostics

```bash
python -u -m evaluation.final_analyses.soc_diagnostics \
  --feeder ieee33 \
  --results-root results/soc_feasible_affine_v2 \
  --out-root results/final_paper_analyses
```

Change the feeder argument to `ieee69` for the corresponding IEEE69 analyses.

The analysis scripts consume completed evaluation records and do not require retraining.

Selected final-paper analysis outputs are provided in the repository's `results/` directory.

---

## Pretrained MADDPG Controllers

**[Download the pretrained A0–A3 controllers](https://github.com/amadasif94/coop-topology-transfer/releases/tag/v1.0.0-checkpoints)**

**Release:** `v1.0.0-checkpoints`

**Archive:** `coop_topology_transfer_24_maddpg_checkpoints.tar`

The release contains **24 selected MADDPG controller checkpoints**:

| Variant | IEEE33 | IEEE69 | Total |
|---|---:|---:|---:|
| A0 — Full cooperation | 3 | 3 | 6 |
| A1 — No shared KPIs | 3 | 3 | 6 |
| A2 — Independent critics | 3 | 3 | 6 |
| A3 — Full independence | 3 | 3 | 6 |
| **Total** | **12** | **12** | **24** |

### Archive contents

The archive contains:

- **24** selected PyTorch model checkpoints (`.pt`)
- **24** checkpoint-selection records (`selection.json`)
- **1** checkpoint manifest (`checkpoint_manifest.json`)

The manifest provides the feeder, controller variant, seed directory, selected training step, checkpoint size, and SHA-256 checksum.

Historical GCN, GAT, and TAGConv checkpoint files are excluded.

### Download and extract

Download the archive from the GitHub Release and extract it:

```bash
tar -xf coop_topology_transfer_24_maddpg_checkpoints.tar
```

The archive preserves the experiment-relative paths and selection metadata.

These pretrained controllers support the zero-shot topology-transfer study without requiring new training runs, subject to the checkpoint-path resolution requirement described in the evaluation section.

---

## Data Sources

### Load and photovoltaic generation

Load and photovoltaic generation profiles originate from the National Renewable Energy Laboratory's Synthetic Models for Advanced, Realistic Testing: Distribution Systems (SMART-DS).

[NREL SMART-DS](https://www.nrel.gov/grid/smart-ds.html)

### Electricity prices

Electricity-price data originate from the New York Independent System Operator (NYISO), using 2018 day-ahead locational-based marginal prices.

### Distribution network models

The experiments use IEEE 33-bus and IEEE 69-bus radial distribution test feeders.

The network parameters, topology definitions, and experimentally adopted tie-line impedances are provided in:

```text
data/network/
```

The network files distributed with the repository are the reference inputs for reproducing the reported experiments.

Users should consult NREL, NYISO, and the original feeder-data sources for applicable attribution and redistribution conditions. Source-data terms are separate from the software license.

---

## Citation

If you use the code, pretrained controllers, experimental data, or results from this repository, please cite:

**Muhammad Amad Asif, Huraiz Ahmed Wasiq, and Nathan Dahlin.**

*The Role of Cooperation in Zero-Shot Topology Transfer for Multi-Agent Battery Storage Control.*

Submitted to *Applied Energy*, 2026.

```bibtex
@unpublished{asif2026cooperation,
  title  = {The Role of Cooperation in Zero-Shot Topology Transfer for Multi-Agent Battery Storage Control},
  author = {Asif, Muhammad Amad and Wasiq, Huraiz Ahmed and Dahlin, Nathan},
  year   = {2026},
  note   = {Submitted to Applied Energy}
}
```

The citation will be updated with the journal article details and DOI when available.

---

## License

This project is licensed under the MIT License. See `LICENSE`.

---

## Contact

**Muhammad Amad Asif**

University at Albany, State University of New York

For implementation questions, reproducibility issues, or research-related inquiries, please open a GitHub issue in this repository.
