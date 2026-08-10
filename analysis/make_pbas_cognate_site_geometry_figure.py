#!/usr/bin/env python3
"""Plot the frozen pre-docking geometry for cognate-site validation design v2.

This script is deliberately outcome-blind: it reads only the design-time target
table and never opens a docking output, score table, or spectral-analysis file.
It creates a target-level figure showing three geometric quantities for the six
primary small-molecule sites:

1. exact bound-ligand heavy-atom capture in the published PBAS box and in the
   same-size box recentered on the experimental ligand centroid;
2. the distance between those two box centers; and
3. the paired change in exact contact-site C-alpha capture.

The exact plotted values are exported to CSV.  A machine-readable manifest
records the frozen input and output checksums, the no-outcome boundary, target
ordering, and visual encodings.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.text import Text
import numpy as np
import pandas as pd


PACKAGE = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (
    PACKAGE / "results" / "pbas_cognate_site_validation_design" / "targets.csv"
)
DEFAULT_RESULTS = PACKAGE / "results" / "pbas_cognate_site_validation_geometry"
DEFAULT_PLOTTED = DEFAULT_RESULTS / "plotted_primary_six.csv"
DEFAULT_MANIFEST = DEFAULT_RESULTS / "figure_manifest.json"
DEFAULT_STEM = PACKAGE / "figures" / "pbas_cognate_site_pre_docking_geometry"

# A fixed timestamp prevents PDF metadata from breaking byte-level determinism.
RELEASE_TIMESTAMP = datetime(2026, 8, 2, tzinfo=timezone.utc)

# Okabe--Ito palette. Shape and fill redundantly identify the two box centers.
PUBLISHED = "#0072B2"
COGNATE = "#D55E00"
POSITIVE = "#009E73"
NEGATIVE = "#CC79A7"
INK = "#202124"
MID_GREY = "#6F777D"
LIGHT_GREY = "#D9DEE2"
GRID = "#E6EAED"
GROSS_FILL = "#FCE8E3"
PARTIAL_FILL = "#FFF3D6"

TARGET_NAMES = {
    "O00408": "PDE2A",
    "P01116": "KRAS",
    "P08173": "CHRM4",
    "P22102": "GART",
    "Q96LA8": "PRMT6",
    "Q9NVS9": "PNPO",
}

CLASS_LABELS = {
    "gross_published_center_miss": "gross miss",
    "partial_published_site_capture": "partial capture",
    "non_gross_published_site_capture": "non-gross",
}

REQUIRED_COLUMNS = {
    "uniprot_accession",
    "pdb_id",
    "bound_ligand_comp_id",
    "bound_ligand_heavy_atoms",
    "geometric_context_class",
    "center_shift_distance_angstrom",
    "published_bound_ligand_fraction_inside_box",
    "experimental_center_bound_ligand_fraction_inside_box",
    "paired_bound_ligand_fraction_inside_box_difference",
    "paired_contact_site_af_ca_fraction_inside_box_difference",
    "in_core_small_molecule_six_primary",
    "selection_used_new_docking_scores",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _portable(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(PACKAGE.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def _validate_and_prepare(source: pd.DataFrame) -> pd.DataFrame:
    missing = REQUIRED_COLUMNS.difference(source.columns)
    if missing:
        raise ValueError(f"Design target table lacks columns: {sorted(missing)}")

    primary = source.loc[source["in_core_small_molecule_six_primary"].eq(True)].copy()
    if len(primary) != 6:
        raise ValueError(f"Expected exactly six primary targets; observed {len(primary)}")
    if primary["uniprot_accession"].duplicated().any():
        raise ValueError("Primary target accessions are not unique")
    if set(primary["uniprot_accession"]) != set(TARGET_NAMES):
        raise ValueError(
            "Frozen primary accessions changed: "
            f"observed={sorted(primary['uniprot_accession'])}"
        )
    if primary["selection_used_new_docking_scores"].fillna(True).astype(bool).any():
        raise ValueError("The frozen design no longer certifies outcome-blind selection")

    numeric_columns = [
        "center_shift_distance_angstrom",
        "published_bound_ligand_fraction_inside_box",
        "experimental_center_bound_ligand_fraction_inside_box",
        "paired_bound_ligand_fraction_inside_box_difference",
        "paired_contact_site_af_ca_fraction_inside_box_difference",
    ]
    numeric = primary[numeric_columns].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError("Primary geometry contains a non-finite plotted value")
    fractions = primary[
        [
            "published_bound_ligand_fraction_inside_box",
            "experimental_center_bound_ligand_fraction_inside_box",
        ]
    ].to_numpy(dtype=float)
    if ((fractions < 0) | (fractions > 1)).any():
        raise ValueError("Bound-ligand capture fractions must lie in [0, 1]")
    recomputed = (
        primary["experimental_center_bound_ligand_fraction_inside_box"]
        - primary["published_bound_ligand_fraction_inside_box"]
    )
    if not np.allclose(
        recomputed,
        primary["paired_bound_ligand_fraction_inside_box_difference"],
        rtol=0,
        atol=1e-12,
    ):
        raise ValueError("Stored bound-ligand capture differences do not recompute")
    if (recomputed < -1e-12).any():
        raise ValueError("Cognate-centered box worsens exact ligand capture in primary panel")

    primary["target_name"] = primary["uniprot_accession"].map(TARGET_NAMES)
    primary["geometry_class_label"] = primary["geometric_context_class"].map(
        CLASS_LABELS
    )
    if primary["geometry_class_label"].isna().any():
        bad = sorted(primary.loc[primary["geometry_class_label"].isna(), "geometric_context_class"])
        raise ValueError(f"Unknown geometric context classes: {bad}")

    # Sorting is a frozen geometric convention, not an outcome-driven choice:
    # worst published exact-ligand capture at the top, then accession for ties.
    primary = primary.sort_values(
        ["published_bound_ligand_fraction_inside_box", "uniprot_accession"],
        ascending=[True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    primary.insert(0, "display_order", np.arange(1, len(primary) + 1))
    primary["target_display_label"] = primary.apply(
        lambda row: (
            f"{row['target_name']} · {row['uniprot_accession']}\n"
            f"{str(row['pdb_id']).upper()}:{row['bound_ligand_comp_id']}  "
            f"[{row['geometry_class_label']}]"
        ),
        axis=1,
    )
    primary["contact_site_tradeoff"] = (
        primary["paired_contact_site_af_ca_fraction_inside_box_difference"] < -0.10
    )

    plotted_columns = [
        "display_order",
        "target_name",
        "uniprot_accession",
        "pdb_id",
        "bound_ligand_comp_id",
        "bound_ligand_heavy_atoms",
        "geometric_context_class",
        "geometry_class_label",
        "target_display_label",
        "center_shift_distance_angstrom",
        "published_bound_ligand_fraction_inside_box",
        "experimental_center_bound_ligand_fraction_inside_box",
        "paired_bound_ligand_fraction_inside_box_difference",
        "paired_contact_site_af_ca_fraction_inside_box_difference",
        "contact_site_tradeoff",
    ]
    return primary[plotted_columns]


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8.0,
            "axes.labelsize": 8.2,
            "axes.titlesize": 9.0,
            "xtick.labelsize": 7.4,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.2,
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )


def _shade_geometry_rows(axes: list[plt.Axes], frame: pd.DataFrame) -> None:
    for y, row in frame.iterrows():
        if row["geometric_context_class"] == "gross_published_center_miss":
            color = GROSS_FILL
        elif row["geometric_context_class"] == "partial_published_site_capture":
            color = PARTIAL_FILL
        else:
            continue
        for ax in axes:
            ax.axhspan(y - 0.43, y + 0.43, color=color, zorder=0, linewidth=0)


def _assert_rendered_layout(
    figure: plt.Figure,
    contact_axis: plt.Axes,
    contact_value_labels: list[Text],
) -> None:
    """Fail loudly if labels escape the raster or panel-C plotting area."""
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    canvas = figure.bbox
    tolerance_px = 1.0
    for artist in figure.findobj(match=Text):
        if not artist.get_visible() or not artist.get_text().strip():
            continue
        bounds = artist.get_window_extent(renderer=renderer)
        if (
            bounds.x0 < canvas.x0 - tolerance_px
            or bounds.x1 > canvas.x1 + tolerance_px
            or bounds.y0 < canvas.y0 - tolerance_px
            or bounds.y1 > canvas.y1 + tolerance_px
        ):
            raise ValueError(
                f"Rendered text escapes the figure canvas: {artist.get_text()!r}; "
                f"text_bounds={bounds.bounds}, canvas_bounds={canvas.bounds}"
            )
    contact_bounds = contact_axis.get_window_extent(renderer=renderer)
    for artist in contact_value_labels:
        bounds = artist.get_window_extent(renderer=renderer)
        if (
            bounds.x0 < contact_bounds.x0 - tolerance_px
            or bounds.x1 > contact_bounds.x1 + tolerance_px
            or bounds.y0 < contact_bounds.y0 - tolerance_px
            or bounds.y1 > contact_bounds.y1 + tolerance_px
        ):
            raise ValueError(
                "Panel-C value label escapes its axes: "
                f"{artist.get_text()!r}; text_bounds={bounds.bounds}, "
                f"axes_bounds={contact_bounds.bounds}"
            )


def make_figure(frame: pd.DataFrame) -> plt.Figure:
    _style()
    figure = plt.figure(figsize=(7.85, 4.15), constrained_layout=False)
    grid = figure.add_gridspec(
        1,
        3,
        width_ratios=(3.75, 1.25, 1.85),
        left=0.215,
        right=0.988,
        bottom=0.235,
        top=0.79,
        wspace=0.22,
    )
    ax_capture = figure.add_subplot(grid[0, 0])
    ax_shift = figure.add_subplot(grid[0, 1], sharey=ax_capture)
    ax_contact = figure.add_subplot(grid[0, 2], sharey=ax_capture)
    axes = [ax_capture, ax_shift, ax_contact]
    y = np.arange(len(frame), dtype=float)

    _shade_geometry_rows(axes, frame)

    published = frame["published_bound_ligand_fraction_inside_box"].to_numpy(float)
    cognate = frame["experimental_center_bound_ligand_fraction_inside_box"].to_numpy(float)
    for yi, x0, x1 in zip(y, published, cognate, strict=True):
        ax_capture.plot(
            [x0, x1],
            [yi, yi],
            color=MID_GREY,
            linewidth=1.35,
            solid_capstyle="round",
            zorder=2,
        )
    ax_capture.scatter(
        published,
        y,
        s=34,
        marker="o",
        facecolor="white",
        edgecolor=PUBLISHED,
        linewidth=1.35,
        zorder=4,
    )
    ax_capture.scatter(
        cognate,
        y,
        s=35,
        marker="D",
        facecolor=COGNATE,
        edgecolor="white",
        linewidth=0.55,
        zorder=5,
    )
    ax_capture.set_xlim(-0.035, 1.045)
    ax_capture.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax_capture.set_xlabel("Exact bound-ligand heavy atoms inside box (fraction)")
    ax_capture.set_title("A   Same-size box: ligand capture", loc="left", fontweight="bold")
    ax_capture.set_yticks(y, frame["target_display_label"])
    ax_capture.tick_params(axis="y", length=0, pad=7)
    ax_capture.xaxis.grid(True, color=GRID, linewidth=0.65, zorder=0)

    for yi, x0, x1 in zip(y, published, cognate, strict=True):
        delta = x1 - x0
        midpoint = (x0 + x1) / 2
        if delta >= 0.13:
            ax_capture.text(
                midpoint,
                yi - 0.20,
                f"+{delta:.2f}",
                ha="center",
                va="bottom",
                fontsize=6.7,
                color=INK,
                zorder=6,
            )

    shift = frame["center_shift_distance_angstrom"].to_numpy(float)
    ax_shift.hlines(y, 0, shift, color=LIGHT_GREY, linewidth=1.5, zorder=1)
    ax_shift.scatter(shift, y, s=24, color=INK, marker="o", zorder=3)
    for yi, value in zip(y, shift, strict=True):
        ax_shift.text(
            value + 0.55,
            yi,
            f"{value:.1f}",
            ha="left",
            va="center",
            fontsize=7.0,
            color=INK,
        )
    ax_shift.set_xlim(0, 23.6)
    ax_shift.set_xticks([0, 10, 20])
    ax_shift.set_xlabel("Distance (Å)")
    ax_shift.set_title("B   Center shift", loc="left", fontweight="bold")
    ax_shift.xaxis.grid(True, color=GRID, linewidth=0.65, zorder=0)
    ax_shift.tick_params(axis="y", left=False, labelleft=False)

    contact = frame[
        "paired_contact_site_af_ca_fraction_inside_box_difference"
    ].to_numpy(float)
    contact_value_labels: list[Text] = []
    ax_contact.axvline(0, color=INK, linewidth=0.75, zorder=1)
    for yi, value in zip(y, contact, strict=True):
        color = POSITIVE if value >= 0 else NEGATIVE
        marker = ">" if value >= 0 else "<"
        ax_contact.hlines(yi, 0, value, color=color, linewidth=2.0, zorder=2)
        ax_contact.scatter(
            value,
            yi,
            s=30,
            marker=marker,
            facecolor=color,
            edgecolor="white",
            linewidth=0.45,
            zorder=4,
        )
        offset = 0.018 if value >= 0 else -0.018
        contact_value_labels.append(
            ax_contact.text(
                value + offset,
                yi,
                f"{value:+.2f}",
                ha="left" if value >= 0 else "right",
                va="center",
                fontsize=6.8,
                color=INK,
                clip_on=True,
            )
        )
    # These limits intentionally include the full rendered extents of the
    # extreme -0.21 and +0.41 value labels, not merely their marker centers.
    ax_contact.set_xlim(-0.39, 0.59)
    ax_contact.set_xticks([-0.2, 0, 0.2, 0.4])
    ax_contact.set_xlabel("Δ contact-site Cα capture")
    ax_contact.set_title("C   Contact-site Δ", loc="left", fontweight="bold")
    ax_contact.xaxis.grid(True, color=GRID, linewidth=0.65, zorder=0)
    ax_contact.tick_params(axis="y", left=False, labelleft=False)

    gart = frame.loc[frame["target_name"].eq("GART")]
    if len(gart) != 1 or not bool(gart["contact_site_tradeoff"].iloc[0]):
        raise ValueError("Expected one predeclared GART contact-site trade-off")
    gart_y = float(gart.index[0])
    gart_x = float(
        gart["paired_contact_site_af_ca_fraction_inside_box_difference"].iloc[0]
    )
    ax_contact.annotate(
        "GART trade-off",
        xy=(gart_x, gart_y),
        xytext=(-0.285, gart_y + 0.62),
        ha="left",
        va="center",
        fontsize=6.8,
        fontweight="bold",
        color=NEGATIVE,
        arrowprops={
            "arrowstyle": "-|>",
            "color": NEGATIVE,
            "linewidth": 0.7,
            "shrinkA": 1,
            "shrinkB": 3,
        },
        zorder=6,
    )

    for ax in axes:
        ax.set_ylim(len(frame) - 0.5, -0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="x", length=3, width=0.65, color=MID_GREY)

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor="white",
            markeredgecolor=PUBLISHED,
            markeredgewidth=1.35,
            markersize=5.8,
            label="Published PBAS center",
        ),
        Line2D(
            [0],
            [0],
            marker="D",
            linestyle="none",
            markerfacecolor=COGNATE,
            markeredgecolor="white",
            markeredgewidth=0.5,
            markersize=5.5,
            label="Cognate-ligand center",
        ),
    ]
    figure.legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(0.215, 0.905),
        frameon=False,
        ncol=2,
        handletextpad=0.5,
        columnspacing=1.2,
    )
    figure.text(
        0.215,
        0.965,
        "Pre-docking site-center geometry · frozen cognate validation design v2",
        ha="left",
        va="top",
        fontsize=10.2,
        fontweight="bold",
        color=INK,
    )
    figure.text(
        0.215,
        0.925,
        "Experimental centers use the bound-ligand centroid; receptor and box dimensions are fixed.",
        ha="left",
        va="top",
        fontsize=7.6,
        color=MID_GREY,
    )
    figure.text(
        0.215,
        0.043,
        "Shaded rows denote geometry-defined gross or partial published-site misses.\n"
        "Panel C shows exact cognate-contact Cα capture; negative values mark trade-offs.\n"
        "Geometry and ordering were frozen before docking; no docking scores were used.",
        ha="left",
        va="bottom",
        fontsize=6.9,
        color=MID_GREY,
        linespacing=1.35,
    )
    _assert_rendered_layout(figure, ax_contact, contact_value_labels)
    return figure


def _write_manifest(
    *,
    input_path: Path,
    plotted_path: Path,
    png_path: Path,
    pdf_path: Path,
    manifest_path: Path,
    frame: pd.DataFrame,
) -> None:
    manifest: dict[str, Any] = {
        "analysis": "pre-docking cognate-site geometry figure",
        "design_version": 2,
        "design_status": "outcome_blind_no_new_docking_score_read",
        "claim_boundary": (
            "This figure establishes the magnitude and direction of the frozen "
            "site-center perturbation only. It does not show docking quality, "
            "pose correctness, affinity accuracy, or a spectral outcome."
        ),
        "input": {
            "path": _portable(input_path),
            "sha256": _sha256(input_path),
        },
        "selection": {
            "field": "in_core_small_molecule_six_primary",
            "required_value": True,
            "sort": "ascending published exact bound-ligand heavy-atom capture; accession tie-break",
            "n_targets": int(len(frame)),
            "targets_in_display_order": frame["uniprot_accession"].tolist(),
        },
        "encodings": {
            "published_center": {"color": PUBLISHED, "marker": "open circle"},
            "cognate_center": {"color": COGNATE, "marker": "filled diamond"},
            "positive_contact_change": {"color": POSITIVE, "marker": "right triangle"},
            "negative_contact_change": {"color": NEGATIVE, "marker": "left triangle"},
            "gross_miss_row_fill": GROSS_FILL,
            "partial_capture_row_fill": PARTIAL_FILL,
        },
        "outputs": {
            "plotted_csv": {"path": _portable(plotted_path), "sha256": _sha256(plotted_path)},
            "png": {"path": _portable(png_path), "sha256": _sha256(png_path)},
            "pdf": {"path": _portable(pdf_path), "sha256": _sha256(pdf_path)},
        },
        "script": {
            "path": _portable(Path(__file__)),
            "sha256": _sha256(Path(__file__)),
        },
        "fixed_pdf_timestamp": RELEASE_TIMESTAMP.isoformat(),
        "rendered_layout_validation": {
            "all_visible_text_within_canvas": True,
            "panel_c_value_labels_within_axes": True,
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--plotted-csv", type=Path, default=DEFAULT_PLOTTED)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-stem", type=Path, default=DEFAULT_STEM)
    args = parser.parse_args()

    if not args.input.is_file():
        raise FileNotFoundError(args.input)
    source = pd.read_csv(args.input)
    plotted = _validate_and_prepare(source)

    args.plotted_csv.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.output_stem.parent.mkdir(parents=True, exist_ok=True)
    plotted.to_csv(args.plotted_csv, index=False, float_format="%.12g")

    figure = make_figure(plotted)
    png_path = args.output_stem.with_suffix(".png")
    pdf_path = args.output_stem.with_suffix(".pdf")
    figure.savefig(
        png_path,
        dpi=400,
        facecolor="white",
        metadata={
            "Software": "Matplotlib",
            "Title": "Pre-docking cognate-site validation geometry",
            "Description": "Outcome-blind frozen design v2; no docking scores used",
        },
    )
    figure.savefig(
        pdf_path,
        facecolor="white",
        metadata={
            "Title": "Pre-docking cognate-site validation geometry",
            "Author": "",
            "Subject": "Outcome-blind frozen design v2; no docking scores used",
            "Creator": "Matplotlib",
            "Producer": "Matplotlib",
            "CreationDate": RELEASE_TIMESTAMP,
            "ModDate": RELEASE_TIMESTAMP,
        },
    )
    plt.close(figure)

    _write_manifest(
        input_path=args.input,
        plotted_path=args.plotted_csv,
        png_path=png_path,
        pdf_path=pdf_path,
        manifest_path=args.manifest,
        frame=plotted,
    )
    print(f"Wrote {png_path}")
    print(f"Wrote {pdf_path}")
    print(f"Wrote {args.plotted_csv}")
    print(f"Wrote {args.manifest}")


if __name__ == "__main__":
    main()
