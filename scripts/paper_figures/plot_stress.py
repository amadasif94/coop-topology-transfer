
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

DATA_PATH = ROOT / "stress_metrics_by_topology.csv"

PDF_PATH = ROOT / "fig_stress.pdf"
PNG_PATH = ROOT / "fig_stress.png"


# ============================================================
# 2. Experimental configuration
# ============================================================

FEEDER = "ieee33"

LOADS = (1.0, 1.1, 1.2, 1.3)

TOPOLOGIES = tuple(
    f"TP{i}" for i in range(1, 8)
)

CONTROLLERS = (
    "A0",
    "A1",
    "A2",
    "A3",
    "PF-MPC",
    "F-MPC",
    "RBC",
    "Idle"
)

METRICS = (
    "feasible_rate",
    "backup_activation_rate"
)


# ============================================================
# 3. Consistent controller styles
# ============================================================

STYLES = {
    "A0": {
        "color": "#1565C0",
        "marker": "o",
        "label": "A0"
    },
    "A1": {
        "color": "#E67E22",
        "marker": "s",
        "label": "A1"
    },
    "A2": {
        "color": "#2E8B57",
        "marker": "^",
        "label": "A2"
    },
    "A3": {
        "color": "#C0392B",
        "marker": "D",
        "label": "A3"
    },
    "PF-MPC": {
        "color": "#343A55",
        "marker": "o",
        "label": "PF-MPC"
    },
    "F-MPC": {
        "color": "#8056B3",
        "marker": "s",
        "label": "F-MPC"
    },
    "RBC": {
        "color": "#777777",
        "marker": "^",
        "label": "RBC"
    },
    "Idle": {
        "color": "#333333",
        "marker": "x",
        "label": "Idle"
    }
}


# ============================================================
# 4. Load and validate evaluation results
# ============================================================

def load_data(path: Path) -> pd.DataFrame:

    if not path.is_file():
        raise FileNotFoundError(
            f"\nMissing input file:\n{path}\n\n"
            "The file must be generated from the existing "
            "final evaluation episode summaries.\n"
            "Do not estimate feasibility or activation rates "
            "from savings_by_topology.csv."
        )

    df = pd.read_csv(path)

    required_columns = [
        "feeder",
        "load_scale",
        "topology_case",
        "controller",
        "feasible_rate",
        "backup_activation_rate"
    ]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # Convert numerical fields.
    df["load_scale"] = pd.to_numeric(
        df["load_scale"],
        errors="raise"
    )

    for metric in METRICS:
        df[metric] = pd.to_numeric(
            df[metric],
            errors="raise"
        )

    # --------------------------------------------------------
    # Select IEEE33 and the required load factors
    # --------------------------------------------------------

    load_mask = np.zeros(
        len(df),
        dtype=bool
    )

    for load in LOADS:
        load_mask |= np.isclose(
            df["load_scale"].to_numpy(dtype=float),
            load,
            rtol=0.0,
            atol=1e-9
        )

    df = df.loc[
        df["feeder"].eq(FEEDER)
        & df["controller"].isin(CONTROLLERS)
        & load_mask
    ].copy()

    if df.empty:
        raise ValueError(
            "No matching IEEE33 stress evaluation data."
        )

    # Normalize floating-point load values.
    for load in LOADS:

        mask = np.isclose(
            df["load_scale"],
            load,
            rtol=0.0,
            atol=1e-9
        )

        df.loc[mask, "load_scale"] = load

    # --------------------------------------------------------
    # Validate unique topology/controller/load records
    # --------------------------------------------------------

    keys = [
        "feeder",
        "load_scale",
        "topology_case",
        "controller"
    ]

    duplicates = df.duplicated(
        keys,
        keep=False
    )

    if duplicates.any():

        raise ValueError(
            "Duplicate evaluation records:\n"
            + df.loc[
                duplicates,
                keys
            ].to_string(index=False)
        )

    # --------------------------------------------------------
    # Validate complete topology coverage
    # --------------------------------------------------------

    expected_topologies = set(TOPOLOGIES)

    for load in LOADS:

        for controller in CONTROLLERS:

            subset = df.loc[
                df["load_scale"].eq(load)
                & df["controller"].eq(controller)
            ]

            observed = set(
                subset["topology_case"].dropna()
            )

            if (
                len(subset) != len(TOPOLOGIES)
                or observed != expected_topologies
            ):

                raise ValueError(
                    f"Incomplete evaluation coverage:\n"
                    f"Controller: {controller}\n"
                    f"Load: {load:.1f}x\n"
                    f"Found: {len(subset)} records\n"
                    f"Missing: "
                    f"{sorted(expected_topologies - observed)}\n"
                    f"Unexpected: "
                    f"{sorted(observed - expected_topologies)}"
                )

    # --------------------------------------------------------
    # Validate measured rates
    # --------------------------------------------------------

    # Feasible rates are required for all controllers,
    # including Idle.
    feasible = df["feasible_rate"]

    if not np.isfinite(feasible).all():
        raise ValueError(
            "Non-finite feasible rates detected."
        )

    if not feasible.between(0.0, 1.0).all():
        raise ValueError(
            "Feasible rates must be fractions in [0, 1]."
        )

    # Backup activation rates are required only for
    # controllers that actually use the safety layer.
    #
    # Idle has no safety-layer activation metric.
    active = df.loc[
        ~df["controller"].eq("Idle"),
        "backup_activation_rate"
    ]

    if not np.isfinite(active).all():
        raise ValueError(
            "Non-finite safety-layer activation rates "
            "detected for dispatch controllers."
        )

    if not active.between(0.0, 1.0).all():
        raise ValueError(
            "Safety-layer activation rates must be "
            "fractions in [0, 1]."
        )

    print(
        f"Validated {len(df)} IEEE33 "
        "topology-level evaluation records."
    )

    return df


# ============================================================
# 5. Aggregate across feeder topologies
# ============================================================

def aggregate_results(df: pd.DataFrame) -> pd.DataFrame:

    # Each topology has equal weight.
    #
    # Input rows must already represent correct averages
    # across evaluation days and seeds.
    #
    # Idle backup activation is ignored, because
    # no safety layer is active for Idle.

    summary = (
        df.groupby(
            ["controller", "load_scale"],
            as_index=False
        )
        .agg(
            feasible_rate=(
                "feasible_rate",
                "mean"
            ),
            backup_activation_rate=(
                "backup_activation_rate",
                "mean"
            ),
            n_topologies=(
                "topology_case",
                "nunique"
            )
        )
    )

    if not (
        summary["n_topologies"] == len(TOPOLOGIES)
    ).all():

        raise ValueError(
            "Incomplete topology aggregation."
        )

    print("\nAggregated stress evaluation metrics:")
    print(summary.to_string(index=False))

    return summary


# ============================================================
# 6. Generate publication-quality figure
# ============================================================

def main():

    df = load_data(DATA_PATH)

    summary = aggregate_results(df)

    # --------------------------------------------------------
    # Formatting consistent with Figures 1 and 2
    # --------------------------------------------------------

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
        "lines.linewidth": 1.7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "figure.facecolor": "white",
        "savefig.facecolor": "white"
    })

    # Two-panel, full-width journal figure.
    fig, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(7.48, 3.4),
        constrained_layout=False
    )

    panel_specs = [
        (
            "feasible_rate",
            "(a) Feasible rate after safety layer",
            "Feasible rate (%)"
        ),
        (
            "backup_activation_rate",
            "(b) Safety-layer activation",
            "Activation / violation rate (%)"
        )
    ]

    # --------------------------------------------------------
    # Plot both panels
    # --------------------------------------------------------

    for ax, (metric, title, ylabel) in zip(
        axes,
        panel_specs
    ):

        for controller in CONTROLLERS:

            subset = (
                summary.loc[
                    summary["controller"].eq(controller)
                ]
                .set_index("load_scale")
            )

            ordered = subset.loc[list(LOADS)]

            style = STYLES[controller]

            # ------------------------------------------------
            # Special treatment for Idle
            # ------------------------------------------------

            if (
                controller == "Idle"
                and metric == "backup_activation_rate"
            ):

                # Idle has no safety layer.
                #
                # Panel (b) instead displays the fraction
                # of Idle observations violating network
                # limits, computed from its feasible rate.
                values = (
                    1.0
                    - ordered[
                        "feasible_rate"
                    ].to_numpy(dtype=float)
                ) * 100.0

            else:

                values = (
                    ordered[
                        metric
                    ].to_numpy(dtype=float)
                    * 100.0
                )

            # Idle is dashed in both panels.
            linestyle = (
                "--"
                if controller == "Idle"
                else "-"
            )

            # ------------------------------------------------
            # Validate against the selected axis range
            # ------------------------------------------------

            if metric == "feasible_rate":
                lower, upper = 80.0, 101.0
            else:
                lower, upper = 0.0, 20.0

            if (
                np.any(values < lower)
                or np.any(values > upper)
            ):

                raise ValueError(
                    f"{controller}, {metric}: "
                    f"values fall outside the selected "
                    f"axis range [{lower}, {upper}].\n"
                    f"Values: {values}\n"
                    "Review the axis limits before "
                    "generating the figure."
                )

            # ------------------------------------------------
            # Plot controller series
            # ------------------------------------------------

            ax.plot(
                LOADS,
                values,
                color=style["color"],
                marker=style["marker"],
                linestyle=linestyle,
                markersize=4.5,
                linewidth=1.7,
                label=style["label"],
                zorder=3
            )

        # ----------------------------------------------------
        # Panel titles and axes
        # ----------------------------------------------------

        ax.set_title(
            title,
            fontsize=9,
            fontweight="semibold",
            pad=12
        )

        ax.set_xlabel(
            "Load factor"
        )

        ax.set_ylabel(
            ylabel
        )

        ax.set_xticks(LOADS)

        ax.set_xticklabels([
            f"{load:.1f}×"
            for load in LOADS
        ])

        ax.set_xlim(
            0.975,
            1.325
        )

        # Focused ranges make the measured changes visible.
        if metric == "feasible_rate":

            ax.set_ylim(80, 101)

            ax.set_yticks(
                np.arange(80, 101, 5)
            )

        else:

            ax.set_ylim(0, 20)

            ax.set_yticks(
                np.arange(0, 21, 5)
            )

        ax.set_axisbelow(True)

        ax.grid(
            axis="y",
            linestyle=":",
            linewidth=0.7,
            alpha=0.35
        )

    # ========================================================
    # 7. Shared legend
    # ========================================================

    handles = {}

    for controller in CONTROLLERS:

        style = STYLES[controller]

        handles[controller] = Line2D(
            [],
            [],
            color=style["color"],
            marker=style["marker"],
            linestyle=(
                "--"
                if controller == "Idle"
                else "-"
            ),
            markersize=4.5,
            linewidth=1.7,
            label=style["label"]
        )

    # Matplotlib fills multi-column legends vertically.
    #
    # This ordering produces:
    #
    # Row 1: A0, A1, A2, A3
    # Row 2: PF-MPC, F-MPC, RBC, Idle

    legend_order = (
        "A0", "PF-MPC",
        "A1", "F-MPC",
        "A2", "RBC",
        "A3", "Idle"
    )

    fig.legend(
        handles=[
            handles[name]
            for name in legend_order
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        ncol=4,
        frameon=False,
        columnspacing=1.5,
        handlelength=2.0
    )

    # ========================================================
    # 8. Layout
    # ========================================================

    fig.subplots_adjust(
        left=0.09,
        right=0.98,
        top=0.87,
        bottom=0.26,
        wspace=0.28
    )

    # ========================================================
    # 9. Save vector PDF and high-resolution PNG
    # ========================================================

    fig.savefig(
        PDF_PATH,
        format="pdf"
    )

    fig.savefig(
        PNG_PATH,
        format="png",
        dpi=600
    )

    plt.close(fig)

    # ========================================================
    # 10. Report
    # ========================================================

    print("\nFigure 3 generated successfully.")
    print("PDF:", PDF_PATH)
    print("PNG:", PNG_PATH)
    print("Feeder:", FEEDER)
    print("Load factors:", LOADS)
    print("Topologies:", len(TOPOLOGIES))
    print("Controllers:", len(CONTROLLERS))
    print("Idle activation replaced by violation reference.")
    print("Original evaluation data unchanged.")


if __name__ == "__main__":
    main()
