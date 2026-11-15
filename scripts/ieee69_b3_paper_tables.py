from pathlib import Path
import numpy as np
import pandas as pd

OUT = Path("results/b3/ieee69/analysis")
SOURCE = OUT / "ieee69_b3_all_metrics_seed_summary.csv"

s = pd.read_csv(SOURCE)

EXPECTED = {
    "ZERO": 1, "RANDOM": 3,
    "MLP_A0": 3, "MLP_A1": 3, "MLP_A2": 3,
    "GCN": 2, "GAT": 2, "TAGCONV": 3,
}
TOPS = {f"TP{i}" for i in range(1, 11)} | {"ALL"}
LOADS = (100, 110, 120, 130)

# Validate data coverage before writing reports.
assert len(s) == 352, "Unexpected summary row count"
assert not s.duplicated(["load", "controller", "topology"]).any()
assert set(s.topology) == TOPS
assert set(s.controller) == set(EXPECTED)
assert set(np.rint(s.load * 100).astype(int)) == set(LOADS)

for model, n in EXPECTED.items():
    part = s[s.controller == model]
    assert len(part) == 44, f"{model}: incomplete coverage"
    assert (part.n_runs == n).all(), f"{model}: wrong seed count"

# IEEE33-compatible detailed report columns.
DETAIL = {
    "load": "Load",
    "topology": "Topology",
    "controller": "Controller",
    "n_runs": "Runs",
    "raw_cost_usd_day_mean": "Raw_cost_USD_day",
    "raw_cost_usd_day_sd": "Raw_cost_SD",
    "adjusted_cost_usd_day_mean": "Adjusted_cost_USD_day",
    "adjusted_cost_usd_day_sd": "Adjusted_cost_SD",
    "adjusted_cost_usd_day_ci95_halfwidth":
        "Adjusted_cost_CI95_halfwidth",
    "savings_vs_zero_usd_day_mean": "Savings_vs_zero_USD_day",
    "savings_vs_zero_usd_day_sd": "Savings_SD",
    "gap_vs_tp1_usd_day_mean": "Generalization_gap_USD_day",
    "savings_retention_vs_tp1_usd_day_mean":
        "Savings_retention_USD_day",
    "voltage_deviation_mean": "Voltage_deviation",
    "voltage_deviation_sd": "Voltage_deviation_SD",
    "worst_min_voltage_pu_mean": "Worst_min_voltage_pu",
    "network_violation_pct_mean": "Network_violation_pct",
    "zero_network_violation_pct_mean": "Zero_violation_pct",
    "backup_activation_pct_mean": "Backup_rate_pct",
    "power_removed_pct_mean":
        "Physical_limiter_power_removed_pct",
    "soc_shortfall_pp_mean": "Terminal_SOC_shortfall_pp",
}

ECONOMIC = {
    "topology": "topology",
    "controller": "controller",
    "adjusted_cost_usd_day_mean": "Net_cost_mean_USD_day",
    "adjusted_cost_usd_day_sd": "Net_cost_run_SD",
    "adjusted_cost_usd_day_ci95_halfwidth":
        "Net_cost_CI95_halfwidth",
    "savings_vs_zero_usd_day_mean": "Savings_vs_zero_USD_day",
    "savings_vs_zero_usd_day_sd": "Savings_run_SD",
    "gap_vs_tp1_usd_day_mean": "Cost_gap_vs_TP1_USD_day",
    "gap_vs_tp1_usd_day_sd": "Cost_gap_run_SD",
    "savings_retention_vs_tp1_usd_day_mean":
        "Savings_retention_USD_day",
    "savings_retention_vs_tp1_usd_day_sd":
        "Savings_retention_run_SD",
}

SAFETY = {
    "topology": "topology",
    "controller": "controller",
    "voltage_deviation_mean": "Voltage_deviation_pu",
    "worst_min_voltage_pu_mean": "Worst_min_voltage_pu",
    "network_violation_pct_mean": "Network_violation_pct",
    "zero_network_violation_pct_mean": "Zero_violation_pct",
    "backup_activation_pct_mean": "Backup_activation_pct",
}

BATTERY = {
    "topology": "topology",
    "controller": "controller",
    "power_removed_pct_mean": "Physical_power_removed_pct",
    "soc_shortfall_pp_mean": "Terminal_SOC_shortfall_pp",
}

NET = {
    "load": "load",
    "controller": "controller",
    "topology": "topology",
    "adjusted_cost_usd_day_mean": "net_mean",
    "adjusted_cost_usd_day_sd": "net_sd",
    "raw_cost_usd_day_mean": "raw_mean",
    "n_runs": "n_runs",
    "adjusted_cost_usd_day_ci95_halfwidth": "ci95_halfwidth",
}

# Verify that every requested source column exists.
needed = set(DETAIL) | set(ECONOMIC) | set(SAFETY) | set(BATTERY) | set(NET)
missing = needed - set(s.columns)
if missing:
    raise ValueError(f"Missing source columns: {sorted(missing)}")

reports = {}

def project(data, mapping, decimals=None):
    result = data[list(mapping)].rename(columns=mapping).copy()
    if decimals is not None:
        result = result.round(decimals)
    return result

# Detailed combined report.
reports["ieee69_complete_detailed_results.csv"] = project(
    s, DETAIL, 5
)

# Net daily cost report.
net = project(s, NET)
net["load"] = np.rint(net["load"] * 100).astype(int)
net["ci95_lower"] = net.net_mean - net.ci95_halfwidth
net["ci95_upper"] = net.net_mean + net.ci95_halfwidth
reports["ieee69_b3_net_daily_cost_sd_ci.csv"] = net

# Four load-specific report families.
for load in LOADS:
    subset = s[np.isclose(s.load, load / 100)].copy()
    assert len(subset) == 88

    reports[f"ieee69_complete_detailed_load_x{load}.csv"] = (
        project(subset, DETAIL, 5)
    )
    reports[f"ieee69_paper_economic_load_x{load}.csv"] = (
        project(subset, ECONOMIC, 4)
    )
    reports[f"ieee69_paper_voltage_safety_load_x{load}.csv"] = (
        project(subset, SAFETY, 4)
    )
    reports[f"ieee69_paper_battery_intervention_load_x{load}.csv"] = (
        project(subset, BATTERY, 4)
    )

assert len(reports) == 18

# Do not overwrite existing reports.
existing = [name for name in reports if (OUT / name).exists()]
if existing:
    raise FileExistsError(f"Reports already exist: {existing}")

for name, frame in reports.items():
    frame.to_csv(OUT / name, index=False)
    print(f"CREATED {name}: {len(frame)} rows")

print("\nSUCCESS: 18 IEEE69 reports created.")
print("Source full-metric CSVs remain unchanged.")
