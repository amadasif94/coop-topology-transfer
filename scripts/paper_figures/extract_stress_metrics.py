from pathlib import Path
import numpy as np
import pandas as pd


# ============================================================
# 1. Paths and experiment configuration
# ============================================================

PROJECT = Path(
    "/network/rit/lab/dahlin_lab/amad/graph_marl_dess"
)

SOURCE = (
    PROJECT
    / "results/soc_feasible_affine_v2/ieee33"
)

OUTPUT = (
    PROJECT
    / "results/paper_final_2026-11/extra/"
      "stress_metrics_by_topology.csv"
)

LOADS = {
    1.0: "load_x100",
    1.1: "load_x110",
    1.2: "load_x120",
    1.3: "load_x130",
}

TOPOLOGIES = [
    f"TP{i}" for i in range(1, 8)
]

LEARNED = ["A0", "A1", "A2", "A3"]

BENCHMARK_DIRS = {
    "PF-MPC": "pf_mpc",
    "F-MPC": "forecast_mpc",
    "RBC": "price_quantile_rbc",
    "Idle": "zero_action_no_backup",
}

SEEDS = (0, 1, 2)

EXPECTED_EPISODES = 31

REQUIRED = [
    "topology_case",
    "episodes",
    "final_network_feasibility_count",
    "final_network_feasibility_measured_steps",
    "backup_activation_count",
    "backup_activation_measured_steps",
]


# ============================================================
# 2. Read one final test aggregate
# ============================================================

def read_aggregate(path):

    if not path.is_file():
        raise FileNotFoundError(
            f"Missing final evaluation file:\n{path}"
        )

    df = pd.read_csv(path)

    missing = [
        c for c in REQUIRED
        if c not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns in {path}:\n{missing}"
        )

    if len(df) != len(TOPOLOGIES):
        raise ValueError(
            f"Expected seven topology rows in {path}; "
            f"found {len(df)}"
        )

    if df["topology_case"].duplicated().any():
        raise ValueError(
            f"Duplicate topologies in {path}"
        )

    if set(df["topology_case"]) != set(TOPOLOGIES):
        raise ValueError(
            f"Unexpected topology coverage in {path}"
        )

    for col in REQUIRED[1:]:
        df[col] = pd.to_numeric(
            df[col],
            errors="raise"
        )

        if not np.isfinite(df[col]).all():
            raise ValueError(
                f"Non-finite values in {col}: {path}"
            )

    if not (df["episodes"] == EXPECTED_EPISODES).all():
        raise ValueError(
            f"Expected {EXPECTED_EPISODES} episodes "
            f"per topology in {path}"
        )

    for count_col, denominator_col in [
        (
            "final_network_feasibility_count",
            "final_network_feasibility_measured_steps",
        ),
        (
            "backup_activation_count",
            "backup_activation_measured_steps",
        ),
    ]:
        if not (df[denominator_col] > 0).all():
            raise ValueError(
                f"Invalid denominator in {path}: "
                f"{denominator_col}"
            )

        if (
            (df[count_col] < 0)
            | (df[count_col] > df[denominator_col])
        ).any():
            raise ValueError(
                f"Invalid count in {path}: {count_col}"
            )

    # Check reported rates against the raw counts.
    comparisons = [
        (
            "final_network_feasibility_rate",
            "final_network_feasibility_count",
            "final_network_feasibility_measured_steps",
        ),
        (
            "backup_activation_rate",
            "backup_activation_count",
            "backup_activation_measured_steps",
        ),
    ]

    for rate_col, count_col, denominator_col in comparisons:

        if rate_col in df.columns:

            reported = pd.to_numeric(
                df[rate_col],
                errors="raise"
            )

            calculated = (
                df[count_col] / df[denominator_col]
            )

            if not np.allclose(
                reported,
                calculated,
                rtol=0,
                atol=1e-5
            ):
                raise ValueError(
                    f"Reported rate disagrees with "
                    f"counts in {path}: {rate_col}"
                )

    return df.set_index("topology_case")


# ============================================================
# 3. Resolve final evaluation paths
# ============================================================

def paths_for(controller, load_directory):

    if controller in LEARNED:

        return [
            SOURCE
            / "mlp"
            / controller
            / f"seed_{seed}"
            / "test"
            / load_directory
            / "aggregate_summary.csv"
            for seed in SEEDS
        ]

    return [
        SOURCE
        / BENCHMARK_DIRS[controller]
        / load_directory
        / "aggregate_summary.csv"
    ]


# ============================================================
# 4. Aggregate correctly across evaluation seeds
# ============================================================

def process_case(controller, load, load_directory):

    paths = paths_for(controller, load_directory)

    frames = [
        read_aggregate(path)
        for path in paths
    ]

    rows = []

    for topology in TOPOLOGIES:

        # Pool measured steps across all seeds.
        feasible_count = sum(
            frame.loc[
                topology,
                "final_network_feasibility_count"
            ]
            for frame in frames
        )

        feasible_steps = sum(
            frame.loc[
                topology,
                "final_network_feasibility_measured_steps"
            ]
            for frame in frames
        )

        feasible_rate = (
            feasible_count / feasible_steps
        )

        if controller == "Idle":

            # Idle has no backup safety controller.
            activation_rate = np.nan

        else:

            activation_count = sum(
                frame.loc[
                    topology,
                    "backup_activation_count"
                ]
                for frame in frames
            )

            activation_steps = sum(
                frame.loc[
                    topology,
                    "backup_activation_measured_steps"
                ]
                for frame in frames
            )

            activation_rate = (
                activation_count / activation_steps
            )

        rows.append({
            "feeder": "ieee33",
            "load_scale": load,
            "topology_case": topology,
            "controller": controller,
            "feasible_rate": feasible_rate,
            "backup_activation_rate": activation_rate,
        })

    return rows


# ============================================================
# 5. Main extraction and validation
# ============================================================

def main():

    controllers = (
        LEARNED + list(BENCHMARK_DIRS)
    )

    # Check all source files before processing.
    expected_paths = []

    for load, load_directory in LOADS.items():
        for controller in controllers:
            expected_paths.extend(
                paths_for(
                    controller,
                    load_directory
                )
            )

    missing = [
        path for path in expected_paths
        if not path.is_file()
    ]

    if missing:
        print(
            f"Missing {len(missing)} "
            "final test aggregate files:"
        )

        for path in missing:
            print(path)

        raise FileNotFoundError(
            "Extraction stopped. "
            "Existing evaluation files were not modified."
        )

    print(
        f"Found all {len(expected_paths)} "
        "final evaluation aggregate files."
    )

    records = []

    for load, load_directory in LOADS.items():

        for controller in controllers:

            records.extend(
                process_case(
                    controller,
                    load,
                    load_directory
                )
            )

    result = pd.DataFrame(records)

    # 8 controllers × 4 loads × 7 topologies.
    expected_rows = (
        len(controllers)
        * len(LOADS)
        * len(TOPOLOGIES)
    )

    if len(result) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} records, "
            f"found {len(result)}."
        )

    keys = [
        "feeder",
        "load_scale",
        "topology_case",
        "controller"
    ]

    if result.duplicated(keys).any():
        raise ValueError(
            "Duplicate output records detected."
        )

    if not result["feasible_rate"].between(
        0, 1
    ).all():
        raise ValueError(
            "Invalid final feasible rates."
        )

    active = result.loc[
        result["controller"] != "Idle",
        "backup_activation_rate"
    ]

    if not active.between(0, 1).all():
        raise ValueError(
            "Invalid backup activation rates."
        )

    result = result.sort_values(
        [
            "load_scale",
            "controller",
            "topology_case"
        ]
    ).reset_index(drop=True)

    # Write only after all checks succeed.
    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    result.to_csv(
        OUTPUT,
        index=False,
        float_format="%.10f",
        na_rep=""
    )

    print("\nSUCCESS")
    print("Output:", OUTPUT)
    print("Rows:", len(result))

    # Print summaries so we can check them
    # against the existing manuscript results.
    summary = (
        result.groupby(
            ["load_scale", "controller"],
            as_index=False
        )
        .agg(
            feasible_rate=("feasible_rate", "mean"),
            backup_activation_rate=(
                "backup_activation_rate",
                "mean"
            )
        )
    )

    summary["feasible_pct"] = (
        100 * summary["feasible_rate"]
    )

    summary["activation_pct"] = (
        100 * summary["backup_activation_rate"]
    )

    print("\nSTRESS SUMMARY (%)")
    print(
        summary[
            [
                "load_scale",
                "controller",
                "feasible_pct",
                "activation_pct"
            ]
        ].round(3).to_string(index=False)
    )

    print(
        "\nIdle activation is undefined; "
        "its violation rate is calculated "
        "by the plotting script."
    )


if __name__ == "__main__":
    main()
