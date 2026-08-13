#!/usr/bin/env python3
"""Graphical abstract for the v5 Journal of Cheminformatics manuscript.

The abstract is drawn from the real Docking-44 matrix and frozen calibration results,
not from schematic stand-ins. The score panel shows a fixed 400-ligand random
subsample ordered by each ligand's mean across targets, so the shared axis is visible
as a vertical gradient. The target map uses the manuscript's raw-unit two-way
centring and is hierarchically ordered.

Output: ``figures/v5/graphical_abstract.{pdf,png}``
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import numpy as np
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform

PACKAGE = Path(__file__).resolve().parents[1]
RESULTS = PACKAGE / "results"
OUT = PACKAGE / "figures" / "v5"

INK = "#1F2328"
MUTED = "#8C9196"
CLAY = "#C75D43"
TEAL = "#2E7C8A"
SLATE = "#566B7E"
CRIMSON = "#A32A2E"

DOCKING44_TARGETS = (
    "1m2z", "1pbq", "1xoq", "2rh1", "2vt4", "2ydo", "2z5x", "3b66",
    "3kk6", "3ln1", "3rze", "4djh", "4ey7", "4iar", "4mqs", "4n6h",
    "5cxv", "5i71", "5tvn", "5u09", "5va1", "6cm4", "6kpf", "6kux",
    "6lqa", "6pdj", "6x3x", "6y1z", "7f8y", "7kwe", "7ljd", "7wc9",
    "7xnk", "7ym8", "8e9y", "8ef6", "8fhs", "8pjk", "8st0", "8wty",
    "8xvk", "8yn3", "9eo4", "V1A",
)
DISPLAY_LIGANDS = 400
DISPLAY_SEED = 20260811


def load_matrix() -> np.ndarray:
    frame = pd.read_csv(
        PACKAGE / "data" / "frozen" / "df_final_v4.csv.gz",
        usecols=list(DOCKING44_TARGETS),
    )
    values = frame[list(DOCKING44_TARGETS)].apply(pd.to_numeric, errors="coerce")
    values = values.clip(upper=0.0)                    # documented preprocessing rule
    values = values.fillna(values.mean(axis=0))        # target-mean imputation
    return values.to_numpy(dtype=np.float64)


def residual_correlation(matrix: np.ndarray) -> np.ndarray:
    column_centred = matrix - matrix.mean(axis=0, keepdims=True)
    residual = column_centred - column_centred.mean(axis=1, keepdims=True)
    return np.corrcoef(residual, rowvar=False)


def clustered_order(correlation: np.ndarray) -> np.ndarray:
    distance = np.clip(1.0 - correlation, 0.0, 2.0)
    np.fill_diagonal(distance, 0.0)
    return leaves_list(linkage(squareform(distance, checks=False), method="average"))


def main() -> None:
    mpl.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8.0,
        "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.facecolor": "white",
    })

    matrix = load_matrix()
    correlation = residual_correlation(matrix)
    order = clustered_order(correlation)

    rng = np.random.default_rng(DISPLAY_SEED)
    rows = rng.choice(matrix.shape[0], size=DISPLAY_LIGANDS, replace=False)
    display = matrix[rows][:, order]
    display = display[np.argsort(display.mean(axis=1))]   # order by the shared axis

    recovery = pd.read_csv(RESULTS / "calibration_panel_recovery" / "summary.csv")
    recovery = recovery[recovery["calibration_ligands"] == 200].set_index("dataset")
    recovery_rows = [
        ("Docking-44", float(recovery.loc["Docking-44", "geometry_spearman_mean"]), CLAY, "D"),
        ("DOCKSTRING-58", float(recovery.loc["DOCKSTRING-58", "geometry_spearman_mean"]), TEAL, "o"),
    ]

    figure = Figure(figsize=(7.0, 2.55), layout="constrained", facecolor="white")
    FigureCanvasAgg(figure)
    grid = figure.add_gridspec(
        1, 5, width_ratios=[1.02, 0.30, 1.02, 0.30, 1.06], wspace=0.05,
    )

    # ---- 1. the score matrix, with its shared axis visible as a gradient ---- #
    ax = figure.add_subplot(grid[0, 0])
    ax.pcolormesh(display, cmap="cividis", shading="flat", rasterized=False, lw=0)
    ax.invert_yaxis()
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("docking score matrix", fontsize=8.2, pad=4, color=INK)
    ax.set_xlabel("44 targets", fontsize=7.2, labelpad=2, color=MUTED)
    ax.text(-0.055, 0.5, "ligands", transform=ax.transAxes, rotation=90,
            fontsize=7.2, color=MUTED, ha="center", va="center")
    for spine in ax.spines.values():
        spine.set_color(MUTED); spine.set_linewidth(0.6)
    ax.annotate(
        "one near-uniform axis\ncarries most variance",
        xy=(0.5, 0.06), xytext=(0.5, -0.30), xycoords="axes fraction",
        textcoords="axes fraction", fontsize=7.2, color=CRIMSON,
        ha="center", va="top",
    )

    # ---- 2. the operation ------------------------------------------------- #
    ax = figure.add_subplot(grid[0, 1]); ax.axis("off")
    ax.text(0.5, 0.60, "\u2212", fontsize=16, color=CLAY, ha="center", va="center",
            fontweight="bold")
    ax.text(0.5, 0.40, "each\nligand's\nmean", fontsize=7.2, color=INK,
            ha="center", va="top", linespacing=1.35)

    # ---- 3. the residual target map --------------------------------------- #
    ax = figure.add_subplot(grid[0, 2])
    ax.pcolormesh(correlation[np.ix_(order, order)], cmap="RdBu_r", vmin=-1, vmax=1,
                  shading="flat", rasterized=False, lw=0)
    ax.invert_yaxis(); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("residual target map", fontsize=8.2, pad=4, color=INK)
    ax.set_xlabel("44 targets", fontsize=7.2, labelpad=2, color=MUTED)
    for spine in ax.spines.values():
        spine.set_color(MUTED); spine.set_linewidth(0.6)
    ax.annotate(
        "more concentrated than\nfour matched nulls",
        xy=(0.5, 0.06), xytext=(0.5, -0.30), xycoords="axes fraction",
        textcoords="axes fraction", fontsize=7.2, color=CRIMSON,
        ha="center", va="top",
    )

    # ---- 4. the consequence ----------------------------------------------- #
    ax = figure.add_subplot(grid[0, 3]); ax.axis("off")
    ax.annotate("", xy=(0.92, 0.56), xytext=(0.08, 0.56), xycoords="axes fraction",
                textcoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", lw=1.2, color=INK, mutation_scale=9))

    ax = figure.add_subplot(grid[0, 4])
    ax.set_xlim(0.84, 1.0); ax.set_ylim(-0.55, 1.55)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_yticks([])
    ax.tick_params(axis="x", labelsize=7.2)
    ax.grid(axis="x", color="#EDEBE6", lw=0.6); ax.set_axisbelow(True)
    for y, (label, value, colour, marker) in zip((1.0, 0.0), recovery_rows):
        ax.plot([0.84, value], [y, y], color="#E1E4E7", lw=1.2, zorder=1)
        ax.plot(value, y, marker=marker, ms=5.5, color=colour,
                mec="white", mew=0.7, zorder=3)
        ax.text(0.842, y + 0.20, label, fontsize=7.1, color=INK,
                ha="left", va="bottom")
    ax.set_xlabel("200-ligand map agreement, Spearman $\\rho$",
                  fontsize=7.2, color=INK, labelpad=2)
    ax.set_title("source-library pilot recovery", fontsize=8.2, pad=4, color=INK)

    OUT.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        OUT / "graphical_abstract.pdf",
        facecolor="white",
        metadata={"CreationDate": None},
    )
    figure.savefig(OUT / "graphical_abstract.png", dpi=400, facecolor="white")
    print("wrote", OUT / "graphical_abstract.pdf")


if __name__ == "__main__":
    main()
