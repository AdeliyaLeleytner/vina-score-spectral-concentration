#!/usr/bin/env python3
"""Visualize complementarity of SEA-style and residual-docking target graphs.

The figure deliberately uses a deterministic KLIFS-grouped circular layout.
Node proximity is therefore an annotation/layout choice, not an inferred
biological distance.  Only the top 10% (19/190) edges from each graph are drawn
in the network panel; the scatter panel retains every target pair.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
import numpy as np
import pandas as pd
from scipy import stats


PACKAGE = Path(__file__).resolve().parents[1]
DEFAULT_SEA = (
    PACKAGE / "results/sea_graph_comparison/kinase_primary_exact_sea_graph.csv"
)
DEFAULT_PAIRS = PACKAGE / "results/replicated_pair_retrieval/target_pairs.csv"
DEFAULT_ANNOTATIONS = (
    PACKAGE / "results/klifs_pocket_control/target_annotations.csv"
)
DEFAULT_FIGURE = PACKAGE / "figures/sea_residual_graph_complementarity"
DEFAULT_RESULTS = PACKAGE / "results/sea_graph_visualization"
DEFAULT_PANEL_METRICS = (
    PACKAGE / "results/sea_graph_panel_transfer/panel_metrics.csv"
)

FIGURE_WIDTH = 7.20
FIGURE_HEIGHT = 6.35
TOP_EDGE_COUNT = 19
RELEASE_TIMESTAMP = datetime(2026, 8, 3, tzinfo=timezone.utc)

INK = "#202124"
PALE = "#E8EAED"
SEA_COLOR = "#D97706"
RESIDUAL_COLOR = "#2563A6"
SHARED_COLOR = "#7A5195"
POSITIVE_COLOR = "#111111"
GROUP_ORDER = ("AGC", "CAMK", "CMGC", "Other", "STE", "TK")
GROUP_COLORS = {
    "TK": "#80B1D3",
    "AGC": "#FDB462",
    "CAMK": "#B3DE69",
    "CMGC": "#BC80BD",
    "STE": "#FB8072",
    "Other": "#BDBDBD",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def use_paper_style() -> None:
    matplotlib.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": INK,
            "axes.labelcolor": INK,
            "axes.titlecolor": INK,
            "axes.titlesize": 8.3,
            "font.family": "serif",
            "font.serif": ["Charter", "Times New Roman", "DejaVu Serif"],
            "font.size": 8,
            "mathtext.fontset": "cm",
            "xtick.color": INK,
            "ytick.color": INK,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.labelsize": 8,
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.hashsalt": "sea-residual-graph-20260803",
            "savefig.facecolor": "white",
            "savefig.transparent": False,
        }
    )


def percentile_ranks(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    return (stats.rankdata(array, method="average") - 0.5) / len(array)


def canonical_pair(first: str, second: str) -> tuple[str, str]:
    return tuple(sorted((str(first), str(second))))


def prepare_edges(
    sea: pd.DataFrame,
    target_pairs: pd.DataFrame,
    *,
    top_edge_count: int = TOP_EDGE_COUNT,
) -> pd.DataFrame:
    """Join the two graphs and assign deterministic top-edge classes."""

    required_sea = {"target_a", "target_b", "sea_raw_score_no_exact"}
    required_pairs = {
        "target_a",
        "target_b",
        "centered_docking_pair_percentile",
        "primary_replicated_positive",
    }
    if missing := sorted(required_sea - set(sea.columns)):
        raise ValueError(f"SEA graph is missing columns: {missing}")
    if missing := sorted(required_pairs - set(target_pairs.columns)):
        raise ValueError(f"target-pair table is missing columns: {missing}")

    def with_key(frame: pd.DataFrame) -> pd.DataFrame:
        result = frame.copy()
        keys = [canonical_pair(a, b) for a, b in zip(result.target_a, result.target_b)]
        result["target_a"] = [key[0] for key in keys]
        result["target_b"] = [key[1] for key in keys]
        return result

    sea = with_key(sea)
    target_pairs = with_key(target_pairs)
    frame = target_pairs.merge(
        sea[["target_a", "target_b", "sea_raw_score_no_exact"]],
        on=["target_a", "target_b"],
        how="inner",
        validate="one_to_one",
    ).sort_values(["target_a", "target_b"], kind="stable").reset_index(drop=True)
    if len(frame) != len(sea) or len(frame) != len(target_pairs):
        raise ValueError("SEA and docking target-pair supports do not match exactly")
    if not 0 < top_edge_count <= len(frame):
        raise ValueError("top_edge_count must lie within the target-pair support")

    frame["sea_percentile"] = percentile_ranks(frame.sea_raw_score_no_exact)
    frame["residual_percentile"] = pd.to_numeric(
        frame.centered_docking_pair_percentile, errors="raise"
    )
    experimental = frame.primary_replicated_positive
    if experimental.dtype == bool:
        frame["experimental_positive"] = experimental
    else:
        mapped = experimental.astype(str).str.lower().map({"true": True, "false": False})
        if mapped.isna().any():
            raise ValueError("primary_replicated_positive is not Boolean")
        frame["experimental_positive"] = mapped.astype(bool)

    stable_order = frame.sort_values(
        ["sea_raw_score_no_exact", "target_a", "target_b"],
        ascending=[False, True, True],
        kind="stable",
    ).index
    residual_order = frame.sort_values(
        ["residual_percentile", "target_a", "target_b"],
        ascending=[False, True, True],
        kind="stable",
    ).index
    frame["sea_top"] = False
    frame["residual_top"] = False
    frame.loc[stable_order[:top_edge_count], "sea_top"] = True
    frame.loc[residual_order[:top_edge_count], "residual_top"] = True
    frame["edge_class"] = np.select(
        [
            frame.sea_top & frame.residual_top,
            frame.sea_top,
            frame.residual_top,
        ],
        ["shared", "sea_only", "residual_only"],
        default="not_top_decile",
    )
    return frame


def grouped_circular_layout(
    annotations: pd.DataFrame,
    *,
    group_order: Sequence[str] = GROUP_ORDER,
    group_gap_degrees: float = 10.0,
) -> pd.DataFrame:
    """Place targets on one circle with visible gaps between KLIFS groups."""

    required = {"target", "klifs_group", "klifs_family"}
    if missing := sorted(required - set(annotations.columns)):
        raise ValueError(f"target annotations are missing columns: {missing}")
    frame = annotations[list(required)].copy()
    unknown = sorted(set(frame.klifs_group.astype(str)) - set(group_order))
    if unknown:
        raise ValueError(f"KLIFS groups lack layout positions: {unknown}")
    order_lookup = {name: index for index, name in enumerate(group_order)}
    frame["group_order"] = frame.klifs_group.map(order_lookup)
    frame = frame.sort_values(
        ["group_order", "klifs_family", "target"], kind="stable"
    ).reset_index(drop=True)

    total_gap = group_gap_degrees * len(group_order)
    node_step = (360.0 - total_gap) / len(frame)
    angle = 92.0
    angles: list[float] = []
    previous_group: str | None = None
    for row in frame.itertuples(index=False):
        group = str(row.klifs_group)
        if previous_group is not None and group != previous_group:
            angle -= group_gap_degrees
        angles.append(angle)
        angle -= node_step
        previous_group = group
    radians = np.deg2rad(angles)
    frame["angle_degrees"] = angles
    frame["x"] = np.cos(radians)
    frame["y"] = np.sin(radians)
    return frame.drop(columns="group_order")


def _draw_edge(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str,
    experimental_positive: bool,
    width: float,
) -> None:
    if experimental_positive:
        halo = FancyArrowPatch(
            start,
            end,
            arrowstyle="-",
            connectionstyle="arc3,rad=0",
            linewidth=width + 1.15,
            color=POSITIVE_COLOR,
            alpha=0.90,
            zorder=1,
        )
        ax.add_patch(halo)
    edge = FancyArrowPatch(
        start,
        end,
        arrowstyle="-",
        connectionstyle="arc3,rad=0",
        linewidth=width,
        linestyle="-",
        color=color,
        alpha=0.95 if experimental_positive else 0.68,
        zorder=2 if experimental_positive else 1.5,
    )
    ax.add_patch(edge)


def draw_network(
    ax: plt.Axes,
    edges: pd.DataFrame,
    layout: pd.DataFrame,
    *,
    method: str,
) -> None:
    if method not in {"sea", "residual"}:
        raise ValueError("method must be 'sea' or 'residual'")
    positions = {
        row.target: (float(row.x), float(row.y))
        for row in layout.itertuples(index=False)
    }
    selected_column = "sea_top" if method == "sea" else "residual_top"
    rank_column = "sea_percentile" if method == "sea" else "residual_percentile"
    method_color = SEA_COLOR if method == "sea" else RESIDUAL_COLOR
    method_label = "SEA-style" if method == "sea" else "Residual docking"
    plotted = edges.loc[edges[selected_column]].copy()
    # Draw non-replicated edges first so experimentally supported edges remain legible.
    plotted = plotted.sort_values(
        ["experimental_positive", "edge_class", "target_a", "target_b"],
        kind="stable",
    )
    for row in plotted.itertuples(index=False):
        rank = float(getattr(row, rank_column))
        width = 0.75 + 1.45 * max(0.0, min(1.0, (rank - 0.90) / 0.10))
        _draw_edge(
            ax,
            positions[row.target_a],
            positions[row.target_b],
            color=SHARED_COLOR if row.edge_class == "shared" else method_color,
            experimental_positive=bool(row.experimental_positive),
            width=width,
        )

    for row in layout.itertuples(index=False):
        ax.scatter(
            row.x,
            row.y,
            s=57,
            facecolor="white",
            edgecolor=GROUP_COLORS[str(row.klifs_group)],
            linewidth=1.35,
            zorder=4,
        )
        radius = 1.14
        label_x = radius * np.cos(np.deg2rad(row.angle_degrees))
        label_y = radius * np.sin(np.deg2rad(row.angle_degrees))
        horizontal = "left" if label_x >= 0 else "right"
        ax.text(
            label_x,
            label_y,
            str(row.target),
            ha=horizontal,
            va="center",
            fontsize=5.9,
            color=INK,
            zorder=5,
        )

    for group in GROUP_ORDER:
        part = layout.loc[layout.klifs_group.eq(group)]
        if part.empty:
            continue
        angles = np.deg2rad(part.angle_degrees.to_numpy(float))
        mean_angle = np.arctan2(np.mean(np.sin(angles)), np.mean(np.cos(angles)))
        group_x = 1.62 * np.cos(mean_angle)
        group_y = 1.62 * np.sin(mean_angle)
        if group == "CMGC":
            group_x += 0.03
            group_y += 0.16
        ax.text(
            group_x,
            group_y,
            group,
            ha="center",
            va="center",
            fontsize=6.0,
            fontweight="bold",
            color=GROUP_COLORS[group],
        )

    legend = [
        Line2D([0], [0], color=method_color, lw=1.8, label=f"{method_label} only"),
        Line2D([0], [0], color=SHARED_COLOR, lw=1.8, label="Shared"),
        Line2D([0], [0], color=INK, lw=2.7, label="Replicated positive"),
    ]
    ax.legend(
        handles=legend,
        frameon=False,
        fontsize=5.8,
        ncol=3,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.07),
        handlelength=1.7,
        columnspacing=0.7,
    )
    ax.set_xlim(-1.72, 1.72)
    ax.set_ylim(-1.72, 1.72)
    ax.set_aspect("equal")
    ax.axis("off")
    positive_hits = int(plotted.experimental_positive.sum())
    ax.set_title(
        f"{method_label}: top 19 edges\n"
        f"{positive_hits}/15 replicated experimental positives",
        pad=2,
    )


def draw_scatter(ax: plt.Axes, edges: pd.DataFrame) -> None:
    styles = {
        "not_top_decile": ("#B9BDC4", 13, 0.48),
        "sea_only": (SEA_COLOR, 23, 0.88),
        "residual_only": (RESIDUAL_COLOR, 23, 0.88),
        "shared": (SHARED_COLOR, 31, 0.95),
    }
    for category in ("not_top_decile", "sea_only", "residual_only", "shared"):
        part = edges.loc[edges.edge_class.eq(category)]
        color, size, alpha = styles[category]
        ax.scatter(
            part.sea_percentile,
            part.residual_percentile,
            s=size,
            color=color,
            alpha=alpha,
            linewidth=0,
            zorder=2,
        )
    positives = edges.loc[edges.experimental_positive]
    ax.scatter(
        positives.sea_percentile,
        positives.residual_percentile,
        s=48,
        facecolor="none",
        edgecolor=POSITIVE_COLOR,
        linewidth=0.9,
        zorder=3,
        label="Replicated experimental edge",
    )

    threshold = (len(edges) - TOP_EDGE_COUNT + 0.5) / len(edges)
    ax.axvline(threshold, color=INK, lw=0.7, ls=(0, (2, 2)), alpha=0.55)
    ax.axhline(threshold, color=INK, lw=0.7, ls=(0, (2, 2)), alpha=0.55)
    rho = stats.spearmanr(edges.sea_percentile, edges.residual_percentile).statistic
    ax.text(
        0.04,
        0.96,
        rf"Spearman $\rho$ = {rho:.3f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.0,
    )
    ax.text(
        0.04,
        0.04,
        "Top-19 overlap: 4 edges\nJaccard = 0.118",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=6.3,
        color=INK,
    )
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xticks([0, 0.5, 1.0])
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_xlabel("SEA-style edge percentile")
    ax.set_ylabel("Residual-docking edge percentile")
    ax.grid(color=PALE, linewidth=0.65, zorder=0)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(
        frameon=False,
        fontsize=6.1,
        loc="lower right",
        handletextpad=0.3,
    )
    ax.set_title("All 190 target pairs", pad=3)


def draw_held_out_auroc(ax: plt.Axes, panel_metrics: pd.DataFrame) -> None:
    required = {"panel", "split", "method", "roc_auc"}
    if missing := sorted(required - set(panel_metrics.columns)):
        raise ValueError(f"panel metrics are missing columns: {missing}")
    held_out = panel_metrics.loc[panel_metrics.split.eq("held_out")].copy()
    panel_order = ("PKIS2", "DAVIS", "KiRHub")
    method_order = ("SEA", "Fusion", "Residual")
    expected = {(panel, method) for panel in panel_order for method in method_order}
    observed = set(zip(held_out.panel.astype(str), held_out.method.astype(str)))
    if observed != expected:
        raise ValueError("held-out panel metrics do not contain the expected 3 x 3 grid")

    means = (
        held_out.groupby("method", sort=False).roc_auc.mean().reindex(method_order)
    )
    y_base = {"PKIS2": 3.0, "DAVIS": 2.0, "KiRHub": 1.0, "Mean": 0.0}
    offsets = {"SEA": 0.18, "Fusion": 0.0, "Residual": -0.18}
    colors = {"SEA": SEA_COLOR, "Fusion": "#2A9D8F", "Residual": RESIDUAL_COLOR}
    markers = {"SEA": "s", "Fusion": "D", "Residual": "o"}
    labels = {"SEA": "SEA-style", "Fusion": "Frozen fusion", "Residual": "Residual"}

    for panel in panel_order:
        values = held_out.loc[held_out.panel.eq(panel)].set_index("method").roc_auc
        ax.plot(
            [float(values.min()), float(values.max())],
            [y_base[panel], y_base[panel]],
            color=PALE,
            lw=2.0,
            zorder=0,
        )
    for method in method_order:
        for panel in panel_order:
            value = float(
                held_out.loc[
                    held_out.panel.eq(panel) & held_out.method.eq(method), "roc_auc"
                ].iloc[0]
            )
            ax.scatter(
                value,
                y_base[panel] + offsets[method],
                s=27,
                marker=markers[method],
                color=colors[method],
                edgecolor="white",
                linewidth=0.45,
                zorder=2,
            )
        mean = float(means[method])
        ax.scatter(
            mean,
            y_base["Mean"] + offsets[method],
            s=38,
            marker=markers[method],
            color=colors[method],
            edgecolor=INK,
            linewidth=0.55,
            zorder=3,
            label=labels[method],
        )
        ax.text(
            mean + 0.008,
            y_base["Mean"] + offsets[method],
            f"{mean:.3f}",
            ha="left",
            va="center",
            fontsize=5.8,
            color=colors[method],
        )

    ax.axvline(0.5, color=INK, lw=0.7, ls=(0, (2, 2)), alpha=0.55)
    ax.axhline(0.5, color=PALE, lw=0.8)
    ax.set_xlim(0.46, 0.82)
    ax.set_ylim(-0.48, 3.48)
    ax.set_yticks([3, 2, 1, 0], labels=[*panel_order, "Mean"])
    ax.set_xlabel("Top-decile target-pair AUROC")
    ax.grid(axis="x", color=PALE, linewidth=0.65, zorder=0)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.legend(
        frameon=False,
        fontsize=6.0,
        loc="upper left",
        ncol=3,
        handletextpad=0.25,
        columnspacing=0.7,
    )
    ax.text(
        0.02,
        0.02,
        r"Fusion weight selected on PKIS1: $w_{residual}=0.55$"
        "\nShared target labels; descriptive panel mean",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=6.0,
        color=INK,
    )
    ax.set_title("Held-out edge retrieval", pad=3)


def make_figure(
    edges: pd.DataFrame,
    layout: pd.DataFrame,
    panel_metrics: pd.DataFrame,
) -> plt.Figure:
    use_paper_style()
    figure = plt.figure(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))
    grid = figure.add_gridspec(
        2,
        2,
        height_ratios=[1.12, 0.88],
        hspace=0.22,
        wspace=0.25,
    )
    sea_ax = figure.add_subplot(grid[0, 0])
    residual_ax = figure.add_subplot(grid[0, 1])
    scatter_ax = figure.add_subplot(grid[1, 0])
    auroc_ax = figure.add_subplot(grid[1, 1])
    draw_network(sea_ax, edges, layout, method="sea")
    draw_network(residual_ax, edges, layout, method="residual")
    draw_scatter(scatter_ax, edges)
    draw_held_out_auroc(auroc_ax, panel_metrics)
    for label, ax, x, y in (
        ("a", sea_ax, -0.05, 1.03),
        ("b", residual_ax, -0.05, 1.03),
        ("c", scatter_ax, -0.14, 1.09),
        ("d", auroc_ax, -0.14, 1.09),
    ):
        ax.text(
            x,
            y,
            label,
            transform=ax.transAxes,
            fontsize=9,
            fontweight="bold",
            ha="left",
            va="top",
        )
    figure.suptitle(
        "SEA-style chemistry and residual docking yield complementary target graphs",
        x=0.5,
        y=0.990,
        fontsize=9.2,
    )
    figure.subplots_adjust(left=0.07, right=0.985, top=0.925, bottom=0.075)
    return figure


def save_figure(figure: plt.Figure, base: Path, *, dpi: int = 400) -> dict[str, str]:
    base.parent.mkdir(parents=True, exist_ok=True)
    pdf = base.with_suffix(".pdf")
    png = base.with_suffix(".png")
    svg = base.with_suffix(".svg")
    common = {
        "Title": "SEA-style and residual-docking target-graph complementarity",
        "Author": "Adeliya Leleytner; Victor Safronov; Maxim Fedorov",
        "Creator": "make_sea_graph_figure.py",
    }
    figure.savefig(
        pdf,
        metadata={
            **common,
            "CreationDate": RELEASE_TIMESTAMP,
            "ModDate": RELEASE_TIMESTAMP,
        },
    )
    figure.savefig(png, dpi=dpi, metadata={"Software": common["Creator"]})
    figure.savefig(
        svg,
        metadata={
            "Title": common["Title"],
            "Creator": common["Creator"],
            "Date": RELEASE_TIMESTAMP.date().isoformat(),
        },
    )
    return {path.suffix.lstrip("."): sha256_file(path) for path in (pdf, png, svg)}


def summarize(edges: pd.DataFrame) -> dict[str, Any]:
    top_union = edges.loc[edges.sea_top | edges.residual_top]
    sea_top = edges.loc[edges.sea_top]
    residual_top = edges.loc[edges.residual_top]
    shared = edges.loc[edges.sea_top & edges.residual_top]
    sea_top_34 = edges.nlargest(len(top_union), "sea_percentile", keep="first")
    residual_top_34 = edges.nlargest(
        len(top_union), "residual_percentile", keep="first"
    )
    return {
        "target_pair_count": int(len(edges)),
        "top_edges_per_graph": int(TOP_EDGE_COUNT),
        "sea_residual_spearman": float(
            stats.spearmanr(edges.sea_percentile, edges.residual_percentile).statistic
        ),
        "sea_only_top_edges": int((edges.edge_class == "sea_only").sum()),
        "residual_only_top_edges": int(
            (edges.edge_class == "residual_only").sum()
        ),
        "shared_top_edges": int(len(shared)),
        "top_edge_jaccard": float(len(shared) / len(top_union)),
        "experimental_positive_pairs": int(edges.experimental_positive.sum()),
        "experimental_positives_in_sea_top19": int(
            sea_top.experimental_positive.sum()
        ),
        "experimental_positives_in_residual_top19": int(
            residual_top.experimental_positive.sum()
        ),
        "experimental_positives_in_union34": int(
            top_union.experimental_positive.sum()
        ),
        "experimental_positives_in_sea_top34": int(
            sea_top_34.experimental_positive.sum()
        ),
        "experimental_positives_in_residual_top34": int(
            residual_top_34.experimental_positive.sum()
        ),
        "shared_top_edge_pairs": [
            f"{row.target_a}--{row.target_b}"
            for row in shared.sort_values(["target_a", "target_b"]).itertuples(
                index=False
            )
        ],
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sea-graph", type=Path, default=DEFAULT_SEA)
    parser.add_argument("--target-pairs", type=Path, default=DEFAULT_PAIRS)
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--panel-metrics", type=Path, default=DEFAULT_PANEL_METRICS)
    parser.add_argument("--figure-base", type=Path, default=DEFAULT_FIGURE)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--dpi", type=int, default=400)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    sea = pd.read_csv(args.sea_graph)
    pairs = pd.read_csv(args.target_pairs)
    annotations = pd.read_csv(args.annotations)
    panel_metrics = pd.read_csv(args.panel_metrics)
    edges = prepare_edges(sea, pairs)
    layout = grouped_circular_layout(annotations)

    expected_targets = set(layout.target)
    observed_targets = set(edges.target_a) | set(edges.target_b)
    if expected_targets != observed_targets:
        raise ValueError("target annotations and target-pair graph do not match")

    figure = make_figure(edges, layout, panel_metrics)
    figure_hashes = save_figure(figure, args.figure_base, dpi=args.dpi)
    plt.close(figure)

    args.results_dir.mkdir(parents=True, exist_ok=True)
    edge_path = args.results_dir / "edge_classification.csv"
    layout_path = args.results_dir / "node_layout.csv"
    summary_path = args.results_dir / "summary.json"
    edges.to_csv(edge_path, index=False)
    layout.to_csv(layout_path, index=False)
    summary = {
        "analysis_status": "science-only publication visualization",
        "layout_contract": (
            "Deterministic circle grouped by KLIFS group and sorted by family/target; "
            "node distance is not an inferred biological similarity."
        ),
        "edge_contract": (
            "Network draws exactly the highest 19 of 190 edges in each graph; "
            "solid black-outlined edges are replicated experimental positives."
        ),
        "statistics": summarize(edges),
        "sources": {
            "sea_graph": {
                "path": str(args.sea_graph.relative_to(PACKAGE)),
                "sha256": sha256_file(args.sea_graph),
            },
            "target_pairs": {
                "path": str(args.target_pairs.relative_to(PACKAGE)),
                "sha256": sha256_file(args.target_pairs),
            },
            "target_annotations": {
                "path": str(args.annotations.relative_to(PACKAGE)),
                "sha256": sha256_file(args.annotations),
            },
            "panel_metrics": {
                "path": str(args.panel_metrics.relative_to(PACKAGE)),
                "sha256": sha256_file(args.panel_metrics),
            },
        },
        "outputs": {
            edge_path.name: sha256_file(edge_path),
            layout_path.name: sha256_file(layout_path),
            **{
                args.figure_base.with_suffix(f".{suffix}").name: digest
                for suffix, digest in figure_hashes.items()
            },
        },
    }
    temporary = summary_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, summary_path)
    print(json.dumps(summary["statistics"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
