
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

DATA_PATH = ROOT / "savings_by_topology.csv"

PDF_PATH = ROOT / "fig_pfstress.pdf"
PNG_PATH = ROOT / "fig_pfstress.png"


# ============================================================
# 2. Figure configuration
# ============================================================

FEEDER = "ieee33"
FEEDER_TITLE = "IEEE 33-bus"

TOPOLOGIES = [
    "TP1", "TP2", "TP3", "TP4",
    "TP5", "TP6", "TP7"
]

LOADS = [1.0, 1.3]

CONTROLLERS = (
    "A0",
    "A1",
    "A2",
    "A3",
    "PF-MPC",
    "F-MPC"
)

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
    }
}


# ============================================================
# 3. Load and validate experimental results
# ============================================================

def load_data(path: Path) -> pd.DataFrame:

    if not path.is_file():
        raise FileNotFoundError(
            f"Input CSV not found: {path}"
        )

    df = pd.read_csv(path)

    required_columns = [
        "feeder",
        "controller",
        "topology_case",
        "load_scale",
        "saving_vs_idle"
    ]

    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    df["load_scale"] = pd.to_numeric(
        df["load_scale"],
        errors="raise"
    )

    df["saving_vs_idle"] = pd.to_numeric(
        df["saving_vs_idle"],
        errors="raise"
    )

    # Retain IEEE33, the two loading conditions,
    # and the six controllers used in this figure.
    load_mask = np.isclose(
        df["load_scale"].to_numpy(),
        1.0,
        rtol=0,
        atol=1e-9
    ) | np.isclose(
        df["load_scale"].to_numpy(),
        1.3,
        rtol=0,
        atol=1e-9
    )

    df = df.loc[
        (df["feeder"] == FEEDER)
        & df["controller"].isin(CONTROLLERS)
        & df["topology_case"].isin(TOPOLOGIES)
        & load_mask
    ].copy()

    if df.empty:
        raise ValueError(
            "No matching IEEE33 results were found."
        )

    # Validate unique results.
    keys = [
        "feeder",
        "controller",
        "topology_case",
        "load_scale"
    ]

    duplicate_mask = df.duplicated(
        keys,
        keep=False
    )

    if duplicate_mask.any():
        raise ValueError(
            "Duplicate experimental results:\n"
            + df.loc[
                duplicate_mask,
                keys
            ].to_string(index=False)
        )

    # Every controller must have all seven
    # topologies at both loads.
    expected_set = set(TOPOLOGIES)

    for load in LOADS:

        for controller in CONTROLLERS:

            subset = df.loc[
                np.isclose(
                    df["load_scale"],
                    load,
                    rtol=0,
                    atol=1e-9
                )
                & df["controller"].eq(controller)
            ]

            observed_set = set(
                subset["topology_case"]
            )

            if (
                len(subset) != len(TOPOLOGIES)
                or observed_set != expected_set
            ):
                raise ValueError(
                    f"Incomplete coverage: "
                    f"controller={controller}, "
                    f"load={load}. "
                    f"Found {len(subset)} rows. "
                    f"Missing: "
                    f"{sorted(expected_set - observed_set)}"
                )

    if not np.isfinite(
        df["saving_vs_idle"]
    ).all():
        raise ValueError(
            "Non-finite savings values detected."
        )

    return df


# ============================================================
# 4. Publication-quality plotting
# ============================================================

def main():

    df = load_data(DATA_PATH)

    print(
        f"Validated {len(df)} experimental results."
    )

    # Keep formatting consistent with Figure 1.
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

    # Two-panel figure at journal print width.
    fig, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(7.48, 3.25),
        sharey=True,
        constrained_layout=False
    )

    x = np.arange(len(TOPOLOGIES))

    # Common y-axis limits across both loading conditions.
    # This allows direct comparison of absolute savings.
    y_values = df[
        "saving_vs_idle"
    ].to_numpy(dtype=float)

    y_min = min(
        -5.0,
        np.floor(y_values.min() / 5.0) * 5.0
    )

    y_max = np.ceil(
        y_values.max() / 5.0
    ) * 5.0

    y_limits = (
        y_min - 2.0,
        y_max + 2.0
    )

    panel_labels = {
        1.0: "(a)",
        1.3: "(b)"
    }

    for ax, load in zip(axes, LOADS):

        subset = df.loc[
            np.isclose(
                df["load_scale"],
                load,
                rtol=0,
                atol=1e-9
            )
        ].copy()

        # Panel title.
        ax.set_title(
            f"{panel_labels[load]} "
            f"{FEEDER_TITLE} — "
            f"{load:.1f}× load",
            fontweight="semibold",
            fontsize=9,
            pad=16
        )

        # ----------------------------------------
        # TP1: training topology
        # ----------------------------------------

        ax.axvspan(
            -0.45,
            0.45,
            color="#E9EEF4",
            zorder=0
        )

        ax.axvline(
            0.5,
            color="#999999",
            linestyle=":",
            linewidth=0.8,
            zorder=1
        )

        # Same annotation as Figure 1.
        ax.text(
            0,
            1.015,
            "Training TP",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=7,
            color="#555555",
            clip_on=False
        )

        # Idle baseline: zero savings.
        ax.axhline(
            0,
            color="#888888",
            linestyle="--",
            linewidth=0.9,
            zorder=1
        )

        # ----------------------------------------
        # Controller savings
        # ----------------------------------------

        for controller in CONTROLLERS:

            series = (
                subset.loc[
                    subset["controller"].eq(controller)
                ]
                .set_index("topology_case")
            )

            values = series.loc[
                TOPOLOGIES,
                "saving_vs_idle"
            ].to_numpy(dtype=float)

            style = STYLES[controller]

            ax.plot(
                x,
                values,
                color=style["color"],
                marker=style["marker"],
                linewidth=1.7,
                markersize=4.2,
                label=style["label"],
                zorder=3
            )

        # ----------------------------------------
        # Axes and grid
        # ----------------------------------------

        ax.set_xticks(x)
        ax.set_xticklabels(TOPOLOGIES)

        ax.set_xlim(
            -0.45,
            len(TOPOLOGIES) - 0.55
        )

        ax.set_ylim(*y_limits)

        ax.set_xlabel(
            "Feeder topology"
        )

        ax.set_axisbelow(True)

        ax.grid(
            axis="y",
            linestyle=":",
            linewidth=0.7,
            alpha=0.35
        )

    axes[0].set_ylabel(
        "Saving vs. idle ($/day)"
    )

    # ========================================================
    # 5. Shared legend
    # ========================================================

    legend_handles = [
        Line2D(
            [],
            [],
            color=STYLES[name]["color"],
            marker=STYLES[name]["marker"],
            linewidth=1.7,
            markersize=4.5,
            label=STYLES[name]["label"]
        )
        for name in CONTROLLERS
    ]

    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=6,
        frameon=False,
        columnspacing=1.2,
        handlelength=2.0
    )

    # ========================================================
    # 6. Figure spacing
    # ========================================================

    fig.subplots_adjust(
        left=0.09,
        right=0.99,
        top=0.88,
        bottom=0.22,
        wspace=0.18
    )

    # ========================================================
    # 7. Export
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

    print("\nFigure 2 generated successfully.")
    print("PDF:", PDF_PATH)
    print("PNG:", PNG_PATH)
    print("Feeder:", FEEDER)
    print("Loads:", LOADS)
    print("Topologies:", len(TOPOLOGIES))
    print("Controllers:", len(CONTROLLERS))
    print("Original experimental data unchanged.")


if __name__ == "__main__":
    main()
