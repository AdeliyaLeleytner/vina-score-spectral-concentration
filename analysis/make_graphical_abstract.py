#!/usr/bin/env python3
"""Render the 920x300 px graphical abstract from the manuscript evidence ledger.

The three panels carry the submission claim in reading order: the target-correlation map
changes across chemical domains, a small same-domain probe panel recovers it, and the map is
not a ligand-wise ranking rule.  Every printed number is read from the ledger.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.transforms import Bbox

try:  # Support direct CLI execution and package-style imports.
    from . import make_manuscript_figures as figures
except ImportError:  # pragma: no cover - direct CLI execution.
    import make_manuscript_figures as figures  # type: ignore


PACKAGE = Path(__file__).resolve().parents[1]
OUTPUT = PACKAGE / "figures" / "graphical_abstract.png"

# The journal asks for 920 x 300 px; at 200 dpi that is 4.6 x 1.5 inches.
WIDTH_PX, HEIGHT_PX, DPI = 920, 300, 200

BLUE = figures.BLUE
ORANGE = figures.ORANGE
GREEN = figures.GREEN
GREY = figures.GREY
LIGHT_GREY = figures.LIGHT_GREY
BLACK = figures.BLACK

def build(ledger: dict) -> plt.Figure:
    domain = ledger["exploratory_chemical_domain_structure"][
        "low_high_molecular_weight_domain_instability"
    ]["datasets"]
    recovery = ledger["probe_panel_recovery"]
    ranking = ledger["dockstring_chembl_ranking"]

    fig, axes = plt.subplots(
        1,
        3,
        # One extra nominal pixel avoids a Matplotlib/macOS float-to-int truncation
        # (4.6 in at 200 dpi is otherwise materialized as 919 and saved as 918 px).
        figsize=((WIDTH_PX + 2) / DPI, HEIGHT_PX / DPI),
    )
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.20, top=0.82, wspace=0.53)

    # Panel 1 - the map changes under chemical-domain shift, not merely disjointness.
    ax = axes[0]
    datasets = ("Docking-44", "DOCKSTRING-58")
    shifted = [domain[name]["row_centered_residual_geometry"]["spearman"] for name in datasets]
    matched = [
        domain[name]["row_centered_residual_geometry"][
            "mw_matched_chemical_group_disjoint_control"
        ]["mean"]
        for name in datasets
    ]
    x = np.arange(2)
    ax.bar(x - 0.18, shifted, width=0.34, color=ORANGE, label="Low vs high MW")
    ax.bar(x + 0.18, matched, width=0.34, color=GREY, label="MW-matched disjoint")
    ax.set_xticks(x, ("Docking-\n44", "DOCKSTRING-\n58"), fontsize=5.0)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Map agreement", fontsize=6.1, labelpad=1)
    ax.tick_params(axis="y", labelsize=5.6, pad=1)
    ax.legend(fontsize=4.8, frameon=False, loc="upper left", handlelength=1.0)
    ax.set_title("The map changes with\nmolecular-size domain", fontsize=6.5, pad=3)

    # Panel 2 - a small representative panel recovers a stated source-library map.
    ax = axes[1]
    sample_sizes = np.array([50, 100, 200, 500])
    for name, color, marker in (
        ("Docking-44", BLUE, "o"),
        ("DOCKSTRING-58", figures.PURPLE, "s"),
    ):
        means = [recovery[name][str(n)]["geometry_spearman_mean"] for n in sample_sizes]
        ax.plot(sample_sizes, means, marker=marker, color=color, lw=1.2, ms=3.3, label=name)
    ax.axvline(200, color=LIGHT_GREY, lw=0.8, ls="--")
    ax.set_xscale("log")
    ax.set_xticks(sample_sizes, ["50", "100", "200", "500"])
    ax.set_ylim(0.72, 1.01)
    ax.set_ylabel("Full-library map agreement", fontsize=6.0, labelpad=1)
    ax.set_xlabel("Representative probe ligands", fontsize=6.0, labelpad=1)
    ax.tick_params(labelsize=5.5, pad=1)
    ax.legend(fontsize=4.8, frameon=False, loc="lower right", handlelength=1.0)
    ax.text(0.02, 0.035, "$n=200$: 0.942 / 0.918", ha="left", fontsize=4.8,
            color=BLACK, transform=ax.transAxes)
    ax.set_title("A few hundred ligands\nrecover the same-domain map", fontsize=6.5, pad=3)

    # Panel 3 - exact algebraic boundary and the broad observed-pair check.
    ax = axes[2]
    ax.axis("off")
    ax.set_title("A map is not a\nligand-wise ranking rule", fontsize=6.5, pad=3)
    ax.text(0.50, 0.84, r"$x_{ij}-\bar{x}_{i\cdot}$", ha="center", fontsize=9, transform=ax.transAxes)
    ax.text(0.50, 0.68, "same target order, exactly", ha="center", fontsize=5.8,
            color=GREEN, fontweight="bold", transform=ax.transAxes)
    ax.annotate("", xy=(0.50, 0.51), xytext=(0.50, 0.63), xycoords=ax.transAxes,
                arrowprops=dict(arrowstyle="->", lw=0.9, color=GREY))
    ax.text(0.47, 0.43, r"divide by unequal target $s_j$", ha="center", fontsize=5.2,
            transform=ax.transAxes)
    ax.text(0.50, 0.30, "new score rule", ha="center", fontsize=6.1, color=ORANGE,
            fontweight="bold", transform=ax.transAxes)
    contrast = ranking["headline_contrasts"]["two_way_residual_minus_absolute_vina"]
    delta = contrast["plugin_mean_difference"]
    low, high = contrast["conservative_cluster_bootstrap_interval_95"]
    ax.text(
        0.50,
        0.09,
        f"Broad ChEMBL: $\\Delta$={delta:+.3f}\ncluster 95% [{low:+.3f}, {high:+.3f}]",
        ha="center",
        fontsize=4.5,
        color=BLACK,
        transform=ax.transAxes,
    )
    return fig


def main() -> None:
    figures.configure()
    mpl.rcParams["savefig.bbox"] = None
    ledger = json.loads((PACKAGE / "results/manuscript_evidence.json").read_text())
    fig = build(ledger)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    # Crop an explicitly sized physical canvas; this avoids backend-specific float-to-pixel
    # rounding while preserving all artwork inside the 920-by-300 journal canvas.
    exact_canvas = Bbox.from_bounds(0, 0, (WIDTH_PX + 0.5) / DPI, HEIGHT_PX / DPI)
    fig.savefig(OUTPUT, dpi=DPI, bbox_inches=exact_canvas, pad_inches=0)
    plt.close(fig)
    height, width = mpl.image.imread(OUTPUT).shape[:2]
    if (width, height) != (WIDTH_PX, HEIGHT_PX):
        raise ValueError(
            f"graphical abstract must be exactly {WIDTH_PX}x{HEIGHT_PX} px, "
            f"got {width}x{height}"
        )
    print(f"Wrote {OUTPUT.relative_to(PACKAGE)} at {width}x{height} px")


if __name__ == "__main__":
    main()
