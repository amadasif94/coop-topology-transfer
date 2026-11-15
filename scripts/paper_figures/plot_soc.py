
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


# ============================================================
# 1. Input and output paths
# ============================================================

ROOT = Path(
    "/network/rit/lab/dahlin_lab/amad/graph_marl_dess/"
    "results/paper_final_2026-11/extra"
)

DATA_PATH = ROOT / "soc_trajectories.csv"

PDF_PATH = ROOT / "fig_soc.pdf"
PNG_PATH = ROOT / "fig_soc.png"


# ============================================================
# 2. Experimental configuration
# ============================================================

# IMPORTANT:
# Battery IDs are assumed to follow the order below.
# Verify these bus mappings against the authoritative
# dess_buses.csv before final manuscript submission.

FEEDERS = [
    (
        "ieee33",
        "IEEE 33-bus",
        [12, 16, 25, 30, 33]
    ),
    (
        "ieee69",
        "IEEE 69-bus",
        [14, 16, 18, 20, 22, 24, 26, 27, 65]
    )
]

CONTROLLERS = [
    ("mlp_a0", "A0 (cooperative)"),
    ("mlp_a3", "A3 (independent)")
]

PANEL_LABELS = {
    ("ieee33", "mlp_a0"): "(a)",
    ("ieee33", "mlp_a3"): "(b)",
    ("ieee69", "mlp_a0"): "(c)",
    ("ieee69", "mlp_a3"): "(d)"
}

EXPECTED_SEEDS = {"0", "1", "2"}

EXPECTED_DAYS = 31
EXPECTED_STEPS = 96

# 15-minute dispatch intervals.
STEP_HOURS = 0.25

# SOC operating bounds.
SOC_MIN = 0.10
SOC_MAX = 0.90

# Light bands prevent visual clutter when
# multiple battery trajectories overlap.
BAND_ALPHA = 0.07


# ============================================================
# 3. Load and validate SOC trajectories
# ============================================================

def load_data(path: Path) -> pd.DataFrame:

    if not path.is_file():
        raise FileNotFoundError(
            f"SOC trajectory file not found: {path}"
        )

    df = pd.read_csv(path)

    required_columns = [
        "feeder",
        "controller",
        "battery_id",
        "time_step",
        "seed",
        "aggregation",
        "mean_soc",
        "p10_soc",
        "p90_soc",
        "n_days",
        "n_seeds"
    ]

    missing = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    # --------------------------------------------------------
    # Normalize column types
    # --------------------------------------------------------

    df["battery_id"] = pd.to_numeric(
        df["battery_id"],
        errors="raise"
    ).astype(int)

    df["time_step"] = pd.to_numeric(
        df["time_step"],
        errors="raise"
    ).astype(int)

    df["mean_soc"] = pd.to_numeric(
        df["mean_soc"],
        errors="raise"
    )

    df["p10_soc"] = pd.to_numeric(
        df["p10_soc"],
        errors="coerce"
    )

    df["p90_soc"] = pd.to_numeric(
        df["p90_soc"],
        errors="coerce"
    )

    df["n_days"] = pd.to_numeric(
        df["n_days"],
        errors="raise"
    )

    df["n_seeds"] = pd.to_numeric(
        df["n_seeds"],
        errors="raise"
    )

    df["seed"] = df["seed"].astype(str)

    # --------------------------------------------------------
    # Separate seed-averaged means from daily quantiles
    # --------------------------------------------------------

    mean_df = df.loc[
        df["aggregation"].eq(
            "mean_of_3_seed_profiles"
        )
        & df["seed"].eq("mean_of_3")
    ].copy()

    band_df = df.loc[
        df["aggregation"].eq(
            "per_seed_day_quantiles"
        )
        & df["seed"].isin(EXPECTED_SEEDS)
    ].copy()

    if mean_df.empty:
        raise ValueError(
            "Missing mean_of_3_seed_profiles data."
        )

    if band_df.empty:
        raise ValueError(
            "Missing per_seed_day_quantiles data."
        )

    # --------------------------------------------------------
    # Validate observations
    # --------------------------------------------------------

    keys = [
        "feeder",
        "controller",
        "battery_id",
        "time_step"
    ]

    if mean_df.duplicated(keys).any():
        raise ValueError(
            "Duplicate mean-of-three SOC records."
        )

    if band_df.duplicated(
        keys + ["seed"]
    ).any():
        raise ValueError(
            "Duplicate per-seed SOC quantile records."
        )

    if not (
        mean_df["n_days"] == EXPECTED_DAYS
    ).all():
        raise ValueError(
            "Unexpected test-day count in mean profiles."
        )

    if not (
        band_df["n_days"] == EXPECTED_DAYS
    ).all():
        raise ValueError(
            "Unexpected test-day count in seed quantiles."
        )

    if not (
        mean_df["n_seeds"] == 3
    ).all():
        raise ValueError(
            "Mean profiles must represent three seeds."
        )

    if not (
        band_df["n_seeds"] == 1
    ).all():
        raise ValueError(
            "Per-seed quantiles must represent one seed."
        )

    if not np.isfinite(
        mean_df["mean_soc"]
    ).all():
        raise ValueError(
            "Non-finite mean SOC values detected."
        )

    if not np.isfinite(
        band_df[
            ["p10_soc", "p90_soc"]
        ].to_numpy()
    ).all():
        raise ValueError(
            "Non-finite percentile values detected."
        )

    if (
        band_df["p10_soc"]
        > band_df["p90_soc"]
    ).any():
        raise ValueError(
            "Some SOC 10th percentiles exceed "
            "the 90th percentiles."
        )

    # --------------------------------------------------------
    # Require exactly three seed quantiles per
    # feeder/controller/battery/time-step combination
    # --------------------------------------------------------

    seed_counts = (
        band_df.groupby(keys)["seed"]
        .nunique()
    )

    if not (
        seed_counts == len(EXPECTED_SEEDS)
    ).all():
        raise ValueError(
            "Incomplete three-seed percentile coverage."
        )

    # --------------------------------------------------------
    # Average seed-specific percentile boundaries
    # --------------------------------------------------------

    # These are NOT exact pooled percentiles.
    # They represent the arithmetic mean of the
    # three seed-specific daily percentile values.

    band_avg = (
        band_df.groupby(
            keys,
            as_index=False
        )
        .agg(
            p10_soc=("p10_soc", "mean"),
            p90_soc=("p90_soc", "mean")
        )
    )

    # --------------------------------------------------------
    # Merge with seed-averaged SOC mean
    # --------------------------------------------------------

    merged = pd.merge(
        mean_df[
            keys + ["mean_soc"]
        ],
        band_avg,
        on=keys,
        how="left",
        validate="one_to_one"
    )

    if merged[
        ["mean_soc", "p10_soc", "p90_soc"]
    ].isna().any().any():
        raise ValueError(
            "Missing SOC trajectories after merging."
        )

    # --------------------------------------------------------
    # Validate each panel
    # --------------------------------------------------------

    for feeder, _, buses in FEEDERS:

        expected_batteries = set(
            range(len(buses))
        )

        for controller, _ in CONTROLLERS:

            subset = merged.loc[
                merged["feeder"].eq(feeder)
                & merged["controller"].eq(controller)
            ]

            expected_rows = (
                len(buses) * EXPECTED_STEPS
            )

            if len(subset) != expected_rows:
                raise ValueError(
                    f"Unexpected row count for "
                    f"{feeder}, {controller}: "
                    f"{len(subset)} "
                    f"(expected {expected_rows})"
                )

            observed_batteries = set(
                subset["battery_id"]
            )

            if (
                observed_batteries
                != expected_batteries
            ):
                raise ValueError(
                    f"Battery coverage mismatch for "
                    f"{feeder}, {controller}."
                )

            for battery_id in expected_batteries:

                battery_data = subset.loc[
                    subset["battery_id"].eq(
                        battery_id
                    )
                ]

                observed_steps = set(
                    battery_data["time_step"]
                )

                expected_steps = set(
                    range(EXPECTED_STEPS)
                )

                if observed_steps != expected_steps:
                    raise ValueError(
                        f"Missing time steps for "
                        f"{feeder}, {controller}, "
                        f"battery {battery_id}."
                    )

    # SOC values must represent fractions.
    values = merged[
        ["mean_soc", "p10_soc", "p90_soc"]
    ].to_numpy(dtype=float)

    if (
        np.any(values < -1e-6)
        or np.any(values > 1.0 + 1e-6)
    ):
        raise ValueError(
            "SOC values outside the physical range [0, 1]."
        )

    print(
        f"Validated {len(merged)} "
        "seed-averaged SOC observations."
    )

    return merged


# ============================================================
# 4. Publication-quality figure
# ============================================================

def main():

    df = load_data(DATA_PATH)

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": [
            "Arial",
            "Helvetica",
            "DejaVu Sans"
        ],
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.6,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "figure.facecolor": "white",
        "savefig.facecolor": "white"
    })

    # --------------------------------------------------------
    # Colors for the individual batteries
    # --------------------------------------------------------

    tab10 = plt.get_cmap("tab10")

    # --------------------------------------------------------
    # Four panels: 2 feeders × 2 controllers
    # --------------------------------------------------------

    fig, axes = plt.subplots(
        nrows=2,
        ncols=2,
        figsize=(7.48, 6.6),
        sharex=True,
        sharey=True,
        constrained_layout=False
    )

    # Dispatch time steps correspond to
    # 00:00, 00:15, ..., 23:45.
    x_hours = (
        np.arange(EXPECTED_STEPS)
        * STEP_HOURS
    )

    # --------------------------------------------------------
    # Draw four SOC panels
    # --------------------------------------------------------

    for row, (
        feeder,
        feeder_title,
        buses
    ) in enumerate(FEEDERS):

        for col, (
            controller,
            controller_title
        ) in enumerate(CONTROLLERS):

            ax = axes[row, col]

            panel_label = PANEL_LABELS[
                (feeder, controller)
            ]

            ax.set_title(
                f"{panel_label} "
                f"{feeder_title} — "
                f"{controller_title}",
                fontsize=9,
                fontweight="semibold",
                pad=8
            )

            subset = df.loc[
                df["feeder"].eq(feeder)
                & df["controller"].eq(controller)
            ]

            # ----------------------------------------
            # Plot each battery
            # ----------------------------------------

            for battery_id, bus in enumerate(buses):

                battery_data = (
                    subset.loc[
                        subset["battery_id"].eq(
                            battery_id
                        )
                    ]
                    .sort_values("time_step")
                )

                color = tab10(
                    battery_id % 10
                )

                y_mean = battery_data[
                    "mean_soc"
                ].to_numpy(dtype=float)

                y_p10 = battery_data[
                    "p10_soc"
                ].to_numpy(dtype=float)

                y_p90 = battery_data[
                    "p90_soc"
                ].to_numpy(dtype=float)

                # ------------------------------------
                # Light SOC percentile band
                # ------------------------------------

                ax.fill_between(
                    x_hours,
                    y_p10,
                    y_p90,
                    color=color,
                    alpha=BAND_ALPHA,
                    linewidth=0,
                    zorder=1
                )

                # ------------------------------------
                # Mean SOC trajectory
                # ------------------------------------

                ax.plot(
                    x_hours,
                    y_mean,
                    color=color,
                    linewidth=1.6,
                    label=f"Bus {bus}",
                    zorder=3
                )

            # ----------------------------------------
            # Physical SOC limits
            # ----------------------------------------

            ax.axhline(
                SOC_MIN,
                color="#999999",
                linestyle="--",
                linewidth=0.8,
                zorder=0
            )

            ax.axhline(
                SOC_MAX,
                color="#999999",
                linestyle="--",
                linewidth=0.8,
                zorder=0
            )

            # ----------------------------------------
            # Axes
            # ----------------------------------------

            # A small horizontal margin keeps the
            # rightmost 24:00 label fully visible.
            ax.set_xlim(-0.4, 24.4)

            ax.set_ylim(0.08, 0.92)

            ax.set_xticks(
                [0, 6, 12, 18, 24]
            )

            ax.set_xticklabels([
                "00:00",
                "06:00",
                "12:00",
                "18:00",
                "24:00"
            ])

            ax.set_yticks([
                0.1,
                0.3,
                0.5,
                0.7,
                0.9
            ])

            ax.set_axisbelow(True)

            ax.grid(
                axis="y",
                linestyle=":",
                linewidth=0.7,
                alpha=0.35
            )

            if col == 0:
                ax.set_ylabel(
                    "State of charge"
                )

            if row == 1:
                ax.set_xlabel(
                    "Time of day"
                )

    # ========================================================
    # 5. Battery legends
    # ========================================================

    # IEEE33: five battery agents.
    handles_33 = [
        Line2D(
            [0],
            [0],
            color=tab10(i),
            linewidth=1.8,
            label=f"Bus {bus}"
        )
        for i, bus in enumerate(
            FEEDERS[0][2]
        )
    ]

    # IEEE69: nine battery agents.
    handles_69 = [
        Line2D(
            [0],
            [0],
            color=tab10(i),
            linewidth=1.8,
            label=f"Bus {bus}"
        )
        for i, bus in enumerate(
            FEEDERS[1][2]
        )
    ]

    # --------------------------------------------------------
    # Attach IEEE33 legend beneath the first row
    # --------------------------------------------------------

    axes[0, 0].legend(
        handles=handles_33,
        loc="upper center",
        bbox_to_anchor=(1.07, -0.06),
        ncol=5,
        frameon=False,
        columnspacing=1.2,
        handlelength=2.0
    )

    # --------------------------------------------------------
    # Attach IEEE69 legend beneath the second row
    # --------------------------------------------------------

    axes[1, 0].legend(
        handles=handles_69,
        loc="upper center",
        bbox_to_anchor=(1.07, -0.20),
        ncol=5,
        frameon=False,
        columnspacing=1.2,
        handlelength=2.0
    )

    # ========================================================
    # 6. Final layout
    # ========================================================

    fig.subplots_adjust(
        left=0.10,
        right=0.97,
        top=0.95,
        bottom=0.14,
        hspace=0.42,
        wspace=0.18
    )

    # ========================================================
    # 7. Export publication-quality files
    # ========================================================

    # Vector PDF for Overleaf.
    fig.savefig(
        PDF_PATH,
        format="pdf",
        bbox_inches="tight",
        pad_inches=0.08
    )

    # High-resolution PNG for inspection.
    fig.savefig(
        PNG_PATH,
        format="png",
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.08
    )

    plt.close(fig)

    # ========================================================
    # 8. Report outputs
    # ========================================================

    print("\nSOC figure generated successfully.")
    print("PDF:", PDF_PATH)
    print("PNG:", PNG_PATH)

    print("Feeders:", [
        feeder[0]
        for feeder in FEEDERS
    ])

    print("Controllers:", [
        controller[0]
        for controller in CONTROLLERS
    ])

    print(
        "Percentile band opacity:",
        BAND_ALPHA
    )

    print(
        "Mean SOC: average of three seed profiles."
    )

    print(
        "Bands: mean of seed-specific "
        "daily 10th and 90th percentiles."
    )

    print(
        "Experimental data unchanged."
    )


if __name__ == "__main__":
    main()
