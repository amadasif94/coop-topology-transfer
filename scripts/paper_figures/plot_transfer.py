from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


# 1. Input and output paths

ROOT = Path(
    "/network/rit/lab/dahlin_lab/amad/graph_marl_dess/"
    "results/paper_final_2026-11/extra"
)

DATA_PATH = ROOT / "savings_by_topology.csv"
PDF_PATH = ROOT / "fig_transfer.pdf"
PNG_PATH = ROOT / "fig_transfer.png"


# 2. Figure settings

FEEDERS = (
    ("ieee33", "IEEE 33-bus", 7),
    ("ieee69", "IEEE 69-bus", 10),
)

LEARNED = ("A0", "A1", "A2", "A3")
BENCHMARKS = ("PF-MPC", "F-MPC", "RBC")
PLOTTED_CONTROLLERS = LEARNED + BENCHMARKS
ALL_CONTROLLERS = PLOTTED_CONTROLLERS + ("Idle",)

LEARNED_Y_LIMITS = (-5, 35)
LEARNED_Y_TICKS = (0, 10, 20, 30)

# Dictionary keys match the controller names in the CSV.
STYLES = {
    "A0":     {"color": "#1565C0", "marker": "o", "label": "A0"},
    "A1":     {"color": "#E67E22", "marker": "s", "label": "A1"},
    "A2":     {"color": "#2E8B57", "marker": "^", "label": "A2"},
    "A3":     {"color": "#C0392B", "marker": "D", "label": "A3"},
    "PF-MPC": {"color": "#343A55", "marker": "o", "label": "PF-MPC"},
    "F-MPC":  {"color": "#8056B3", "marker": "s", "label": "F-MPC"},
    "RBC":    {"color": "#777777", "marker": "^", "label": "RBC"},
}


def load_savings(path):
    """Read nominal-load savings and verify one result per plotted case."""
    if not path.is_file():
        raise FileNotFoundError(f"Input file not found: {path}")

    data = pd.read_csv(path)

    keys = ["feeder", "controller", "topology_case"]
    required = keys + ["load_scale", "saving_vs_idle"]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    load_scale = pd.to_numeric(data["load_scale"], errors="raise")
    nominal = np.isclose(load_scale, 1.0, rtol=0.0, atol=1e-9)
    feeder_ids = [feeder for feeder, _, _ in FEEDERS]

    selected = data.loc[
        nominal
        & data["feeder"].isin(feeder_ids)
        & data["controller"].isin(ALL_CONTROLLERS),
        keys + ["saving_vs_idle"],
    ].copy()

    selected["saving_vs_idle"] = pd.to_numeric(
        selected["saving_vs_idle"], errors="raise"
    )

    invalid = ~np.isfinite(selected["saving_vs_idle"])
    if invalid.any():
        raise ValueError(
            "Missing or infinite savings:\n"
            + selected.loc[invalid].to_string(index=False)
        )

    duplicates = selected.duplicated(keys, keep=False)
    if duplicates.any():
        raise ValueError(
            "Duplicate nominal-load results:\n"
            + selected.loc[duplicates, keys].to_string(index=False)
        )

    for feeder, _, n_topologies in FEEDERS:
        expected = {f"TP{i}" for i in range(1, n_topologies + 1)}

        for controller in ALL_CONTROLLERS:
            rows = selected.loc[
                selected["feeder"].eq(feeder)
                & selected["controller"].eq(controller),
                "topology_case",
            ]
            observed = set(rows.dropna())

            if len(rows) != n_topologies or observed != expected:
                raise ValueError(
                    f"Invalid topology coverage for {feeder}, {controller}. "
                    f"Expected {n_topologies} rows, found {len(rows)}. "
                    f"Missing: {sorted(expected - observed)}. "
                    f"Unexpected: {sorted(observed - expected)}. "
                    f"Unlabeled rows: {rows.isna().sum()}."
                )

    idle = selected.loc[selected["controller"].eq("Idle"), "saving_vs_idle"]
    if not np.allclose(idle, 0.0, rtol=0.0, atol=1e-9):
        raise ValueError("Idle savings differ from the plotted zero baseline.")

    # A fixed y-axis must not hide a result if the input CSV changes.
    lower, upper = LEARNED_Y_LIMITS
    outside_axis = (
        selected["controller"].isin(LEARNED)
        & ~selected["saving_vs_idle"].between(lower, upper)
    )
    if outside_axis.any():
        raise ValueError(
            f"Learned-controller savings fall outside [{lower}, {upper}] $/day. "
            "Review LEARNED_Y_LIMITS before plotting:\n"
            + selected.loc[outside_axis].to_string(index=False)
        )

    return selected.set_index(keys)["saving_vs_idle"]


def main():
    savings = load_savings(DATA_PATH)
    print(f"Validated {len(savings)} nominal-load results.")

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    })

    # Independent axes let the benchmark panels use their full data ranges.
    fig, axes = plt.subplots(2, 2, figsize=(7.48, 5.4))

    for col, (feeder, feeder_title, n_topologies) in enumerate(FEEDERS):
        topologies = [f"TP{i}" for i in range(1, n_topologies + 1)]
        x = np.arange(n_topologies)

        for row, controllers in enumerate((LEARNED, BENCHMARKS)):
            ax = axes[row, col]
            panel = "abcd"[2 * row + col]
            learned_panel = row == 0
            group_title = "Learned controllers" if learned_panel else "Benchmarks"

            ax.set_title(
                f"({panel}) {feeder_title} — {group_title}",
                fontweight="semibold",
                pad=16,
            )

            # Highlight TP1 and separate it from the reconfigured feeders.
            ax.axvspan(-0.45, 0.45, color="#E9EEF4", zorder=0)
            ax.axvline(
                0.5, color="#999999", linestyle=":", linewidth=0.8, zorder=1
            )
            ax.text(
                0, 1.015,
                "Training TP" if learned_panel else "Base TP",
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="bottom",
                fontsize=7,
                color="#555555",
                clip_on=False,
            )

            ax.axhline(
                0, color="#888888", linestyle="--", linewidth=0.9, zorder=2
            )

            for controller in controllers:
                series = savings.xs(
                    (feeder, controller), level=("feeder", "controller")
                )

                # Explicit ordering keeps TP10 after TP9.
                values = series.loc[topologies].to_numpy(dtype=float)

                ax.plot(
                    x, values,
                    **STYLES[controller],
                    markersize=4.2,
                    linewidth=1.7,
                    zorder=3,
                )

            ax.set_xticks(x)
            ax.set_xticklabels(topologies)
            ax.set_xlim(-0.45, n_topologies - 0.55)

            if learned_panel:
                ax.set_ylim(*LEARNED_Y_LIMITS)
                ax.set_yticks(LEARNED_Y_TICKS)
            else:
                ax.set_xlabel("Feeder topology")

            if col == 0:
                ax.set_ylabel("Savings vs. idle ($/day)")

            ax.set_axisbelow(True)
            ax.grid(axis="y", linestyle=":", linewidth=0.7, alpha=0.35)

    # Arrange the two legend rows by panel: learned controllers, then benchmarks.
    handles = {
        name: Line2D([], [], **STYLES[name], markersize=4.5, linewidth=1.7)
        for name in PLOTTED_CONTROLLERS
    }
    handles["Idle"] = Line2D(
        [], [],
        color="#888888",
        linestyle="--",
        linewidth=0.9,
        label="Idle (zero savings)",
    )

    # Matplotlib fills a multi-column legend down each column.
    legend_order = ("A0", "PF-MPC", "A1", "F-MPC", "A2", "RBC", "A3", "Idle")
    fig.legend(
        handles=[handles[name] for name in legend_order],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        ncol=4,
        frameon=False,
        columnspacing=1.5,
        handlelength=2.0,
    )

    fig.subplots_adjust(
        left=0.10,
        right=0.98,
        top=0.91,
        bottom=0.19,
        hspace=0.53,
        wspace=0.25,
    )

    # Preserve the 7.48 × 5.4 inch canvas in both exports.
    fig.savefig(PDF_PATH, format="pdf")
    fig.savefig(PNG_PATH, format="png", dpi=600)
    plt.close(fig)

    print(f"PDF: {PDF_PATH}")
    print(f"PNG: {PNG_PATH}")


if __name__ == "__main__":
    main()