#!/usr/bin/env python3
"""Figures for the v5 (post-audit) Journal of Cheminformatics manuscript.

Every plotted value is read from a frozen file under ``results/``.  Nothing is
recomputed, smoothed, or imputed, and no favourable sensitivity is selected: where
a contrast exists under more than one experimental transform, both are drawn.

Figure 1  ``figures/v5/fig1_shared_axis.{pdf,png}``
    The near-uniform leading component across eight scoring outputs on one
    identical fixed-pose block, ordered by feature class.
    Source: ``results/nonvina_scorer_transport/scorer_spectral_metrics.csv``

Figure 3  ``figures/v5/fig3_external_agreement.{pdf,png}``
    Raw versus residual docking target-map agreement with measured pharmacology,
    including the KiRHub non-replication.
    Sources: ``results/pdsp_counterscreen_retrieval/global_retrieval_metrics.csv``,
    ``results/spd_external_validation/geometry_metrics.csv``,
    ``results/klifs_pocket_control/locked_endpoint_retrieval.csv``,
    ``results/kirhub_external_validation/summary.json``
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import numpy as np
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

PACKAGE = Path(__file__).resolve().parents[1]
RESULTS = PACKAGE / "results"
OUT = PACKAGE / "figures" / "v5"

WIDTH = 6.30  # article text width, A4 with 2.5 cm margins

# Project palette used by the released figure drivers. Clay = docking / caution,
# teal = experimental / positive, slate = neutral, crimson = annotation only.
INK = "#1F2328"
MUTED = "#8C9196"
HAIR = "#C9CBC8"
CLAY = "#C75D43"
TEAL = "#2E7C8A"
SAGE = "#5E9A6B"
SAND = "#D29A3A"
SLATE = "#566B7E"
CRIMSON = "#A32A2E"  # annotation accent, never a data series

# Feature class -> (colour, marker). Hue AND marker, so the grouping survives
# greyscale printing and colour-vision deficiency.
CLASSES = {
    "interaction fingerprint": (TEAL, "o"),
    "learned contact net": (SAGE, "s"),
    "empirical terms": (SLATE, "^"),
    "atom-pair counts": (CLAY, "D"),
}

SCORERS = [
    # (csv key, display label, feature class)
    ("plecscore_linear", "PLECscore", "interaction fingerprint"),
    ("nnscore", "NNScore 2.0", "learned contact net"),
    ("vinardo", "Vinardo", "empirical terms"),
    ("released_vina", "AutoDock Vina", "empirical terms"),
    ("smina_vina", "smina (Vina fn.)", "empirical terms"),
    ("rfscore_v3", "RF-Score v3", "atom-pair counts"),
    ("rfscore_v1", "RF-Score v1", "atom-pair counts"),
    ("rfscore_v2", "RF-Score v2", "atom-pair counts"),
]


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.0,
            "axes.titlesize": 8.4,
            "axes.labelsize": 8.2,
            "axes.linewidth": 0.7,
            "xtick.labelsize": 7.4,
            "ytick.labelsize": 7.4,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "legend.fontsize": 7.0,
            "legend.frameon": False,
            "lines.linewidth": 1.3,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )


def new_figure(width: float, height: float) -> Figure:
    figure = Figure(figsize=(width, height), layout="constrained", facecolor="white")
    FigureCanvasAgg(figure)
    return figure


def clean(axis, *, grid: str = "x") -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_visible(False)
    if grid:
        axis.grid(axis=grid, color="#EDEBE6", linewidth=0.6, zorder=0)
    axis.set_axisbelow(True)
    axis.tick_params(axis="y", length=0)


def panel_label(axis, text: str, *, x: float = -0.02, y: float = 1.04) -> None:
    axis.text(
        x, y, text, transform=axis.transAxes,
        fontsize=9.5, fontweight="bold", va="bottom", ha="right", color=INK,
    )


def save(figure: Figure, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        OUT / f"{stem}.pdf",
        facecolor="white",
        metadata={"CreationDate": None},
    )
    figure.savefig(OUT / f"{stem}.png", dpi=400, facecolor="white")
    print(f"wrote {OUT / stem}.pdf/.png")


# --------------------------------------------------------------------------- #
# Figure 1 — the shared axis across eight scoring outputs
# --------------------------------------------------------------------------- #

def figure_one() -> None:
    frame = pd.read_csv(RESULTS / "nonvina_scorer_transport" / "scorer_spectral_metrics.csv")
    frame = frame.set_index("scorer")

    rows = list(reversed(SCORERS))  # bottom-to-top so PLECscore reads at the top
    ypos = np.arange(len(rows), dtype=float)

    figure = new_figure(WIDTH, 3.55)
    axes = figure.subplots(1, 3, sharey=True, width_ratios=[1.0, 0.92, 0.98])
    ax_pc1, ax_cos, ax_pr = axes

    for axis in axes:
        clean(axis)
        axis.set_ylim(-0.75, len(rows) - 0.15)

    labels = []
    for index, (key, label, klass) in enumerate(rows):
        colour, marker = CLASSES[klass]
        record = frame.loc[key]
        labels.append(label)

        # (a) PC1 variance fraction, with the cluster-bootstrap interval.
        ax_pc1.plot(
            [record["raw_pc1_fraction_ci_low"], record["raw_pc1_fraction_ci_high"]],
            [ypos[index]] * 2, color=colour, lw=1.5, solid_capstyle="round", zorder=2,
        )
        ax_pc1.plot(
            record["raw_pc1_fraction"], ypos[index], marker=marker, ms=5.0,
            color=colour, mec="white", mew=0.7, zorder=3,
        )

        # (b) cosine of the PC1 loading vector with the uniform target direction.
        ax_cos.plot(
            [record["pc1_uniform_cosine_ci_low"], record["pc1_uniform_cosine_ci_high"]],
            [ypos[index]] * 2, color=colour, lw=1.5, solid_capstyle="round", zorder=2,
        )
        ax_cos.plot(
            record["pc1_uniform_cosine"], ypos[index], marker=marker, ms=5.0,
            color=colour, mec="white", mew=0.7, zorder=3,
        )

        # (c) participation ratio, raw -> after within-ligand centring.
        raw_pr = record["raw_participation_ratio"]
        cen_pr = record["row_centered_participation_ratio"]
        ax_pr.annotate(
            "", xy=(cen_pr, ypos[index]), xytext=(raw_pr, ypos[index]),
            arrowprops=dict(arrowstyle="-|>", lw=1.1, color=colour,
                            shrinkA=2.2, shrinkB=0.5, mutation_scale=7),
            zorder=2,
        )
        ax_pr.plot(raw_pr, ypos[index], marker=marker, ms=4.4, color="white",
                   mec=colour, mew=1.2, zorder=3)

    ax_pc1.set_yticks(ypos)
    ax_pc1.set_yticklabels(labels)

    # Recorded descriptive gates, drawn as reference lines with a horizontal label above
    # the axis so nothing is rotated into the data area.
    def gate(axis, value: str | float, text: str, ha: str = "left", dx: float = 0.0):
        axis.axvline(value, color=CRIMSON, lw=0.9, ls=(0, (4, 2)), zorder=1)
        axis.annotate(
            text, xy=(value, 1.0), xytext=(value + dx, 1.022),
            xycoords=("data", "axes fraction"), textcoords=("data", "axes fraction"),
            color=CRIMSON, fontsize=7.0, ha=ha, va="bottom",
        )

    gate(ax_pc1, 0.50, "gate 0.50", dx=0.004)
    gate(ax_cos, 0.95, "gate 0.95", dx=0.0012)
    gate(ax_pr, 4.5, "gate $P/2$", dx=0.09)

    ax_pc1.set_xlabel("PC1 variance fraction")
    ax_pc1.set_xlim(0.37, 0.72)
    ax_pc1.set_xticks([0.4, 0.5, 0.6, 0.7])
    ax_cos.set_xlabel("cos with uniform vector")
    ax_cos.set_xlim(0.936, 0.996)
    ax_cos.set_xticks([0.95, 0.97, 0.99])
    ax_pr.set_xlabel("participation ratio")
    ax_pr.set_xlim(1.5, 8.6)
    ax_pr.set_xticks([2, 4, 6, 8])

    panel_label(ax_pc1, "a", x=-0.60, y=1.10)
    panel_label(ax_cos, "b", x=-0.06, y=1.10)
    panel_label(ax_pr, "c", x=-0.06, y=1.10)

    # The one output that falls outside the quantitative gates, named next to it
    # in the empty band between its interval and the gate line.
    ax_pc1.annotate(
        "outside gate", xy=(0.428, ypos[-1]), xytext=(0.527, ypos[-1]),
        color=CRIMSON, fontsize=7.0, ha="left", va="center",
        arrowprops=dict(arrowstyle="-", lw=0.6, color=CRIMSON, shrinkA=1.5, shrinkB=1.5),
    )
    # Panel c reads left to right: open marker is raw, arrowhead is centred.
    for value_col, text in (
        ("raw_participation_ratio", "raw"),
        ("row_centered_participation_ratio", "centred"),
    ):
        ax_pr.text(
            frame.loc["rfscore_v2", value_col], -0.52, text,
            fontsize=7.0, color=SLATE, ha="center", va="center",
        )

    handles = [
        Line2D([], [], marker=marker, ms=4.6, color=colour, ls="none",
               mec="white", mew=0.6, label=name)
        for name, (colour, marker) in CLASSES.items()
    ]
    figure.legend(
        handles=handles, loc="outside lower center", ncol=4,
        handletextpad=0.35, columnspacing=1.6,
    )
    save(figure, "fig1_shared_axis")


# --------------------------------------------------------------------------- #
# Figure 3 — agreement with measured pharmacology, raw versus residual
# --------------------------------------------------------------------------- #

def _spd_rows() -> list[tuple[str, float, float]]:
    """All SPD endpoints under both experimental transforms.

    Both transforms are drawn because a raw-vs-residual docking contrast is only
    interpretable against a *fixed* experimental geometry; reporting one transform
    only would select a support.
    """
    frame = pd.read_csv(RESULTS / "spd_external_validation" / "geometry_metrics.csv")
    frame = frame[
        (frame["docking_geometry"] == "nonoverlap__local_12")
        & (frame["min_pair_support"] == 40)
        & (frame["compound_identity"] == "connectivity_key")
    ]
    wanted = [
        ("floor_at_bound", float("nan"), "raw", "SPD bound  ·  raw assay"),
        ("floor_at_bound", float("nan"), "column_z_row_center", "SPD bound  ·  centred assay"),
        ("censor_aware_binary", 10.0, "raw", "SPD 10 µM  ·  raw assay"),
        ("censor_aware_binary", 10.0, "column_z_row_center", "SPD 10 µM  ·  centred assay"),
        ("censor_aware_binary", 30.0, "raw", "SPD 30 µM  ·  raw assay"),
        ("censor_aware_binary", 30.0, "column_z_row_center", "SPD 30 µM  ·  centred assay"),
    ]
    out = []
    for endpoint, threshold, transform, label in wanted:
        subset = frame[
            (frame["endpoint"] == endpoint)
            & (frame["experimental_transform"] == transform)
        ]
        if endpoint == "censor_aware_binary":
            subset = subset[subset["threshold_uM"] == threshold]
        if subset.empty:
            raise ValueError(f"no SPD row for {endpoint}/{threshold}/{transform}")
        record = subset.iloc[0]
        out.append((label, float(record["raw_spearman"]), float(record["residual_spearman"])))
    return out


def figure_three() -> None:
    # --- panel a: target-map agreement (Spearman rho) --------------------- #
    pdsp = pd.read_csv(RESULTS / "pdsp_counterscreen_retrieval" / "global_retrieval_metrics.csv")
    pdsp = pdsp[pdsp["top_fraction"] == 0.1].set_index("predictor")
    rho_rows = [(
        "PDSP quantified Ki\n(21 targets, 148 pairs)",
        float(pdsp.loc["raw_docking", "continuous_spearman"]),
        float(pdsp.loc["residual_docking", "continuous_spearman"]),
    )]
    rho_rows.extend(_spd_rows())

    # --- panel b: top-decile retrieval (ROC-AUC) -------------------------- #
    kinase = pd.read_csv(RESULTS / "klifs_pocket_control" / "locked_endpoint_retrieval.csv")
    kinase = kinase.set_index("predictor")
    kir = json.loads((RESULTS / "kirhub_external_validation" / "summary.json").read_text())
    kir_metrics = kir["KiRHub_specific_top10_boundary"]["metrics"]

    auc_rows = [
        ("PDSP quantified Ki",
         float(pdsp.loc["raw_docking", "roc_auc"]),
         float(pdsp.loc["residual_docking", "roc_auc"])),
        ("Kinase frozen endpoint\n(20 targets, 190 pairs)",
         float(kinase.loc["raw_Vina_target_geometry", "roc_auc"]),
         float(kinase.loc["centered_Vina_target_geometry", "roc_auc"])),
        ("KiRHub 2026",
         float(kir_metrics["raw_docking"]["roc_auc"]),
         float(kir_metrics["centered_docking"]["roc_auc"])),
    ]

    figure = new_figure(WIDTH, 3.9)
    ax_rho, ax_auc = figure.subplots(1, 2, width_ratios=[1.0, 0.92])

    def draw(axis, rows, xlabel, xlim):
        clean(axis)
        ypos = np.arange(len(rows), dtype=float)[::-1]
        for y, (label, raw, res) in zip(ypos, rows):
            improves = res > raw
            colour = TEAL if improves else CRIMSON
            axis.annotate(
                "", xy=(res, y), xytext=(raw, y),
                arrowprops=dict(arrowstyle="-|>", lw=1.35, color=colour,
                                shrinkA=1.5, shrinkB=1.0, mutation_scale=8),
                zorder=2,
            )
            axis.plot(raw, y, "o", ms=4.6, color="white", mec=MUTED, mew=1.3, zorder=3)
            axis.plot(res, y, "o", ms=4.8, color=colour, mec="white", mew=0.7, zorder=4)
        axis.set_yticks(ypos)
        axis.set_yticklabels([r[0] for r in rows])
        axis.set_xlabel(xlabel)
        axis.set_xlim(*xlim)
        axis.set_ylim(-0.7, len(rows) - 0.3)
        return ypos

    y_rho = draw(ax_rho, rho_rows, "target-map agreement, Spearman $\\rho$", (-0.02, 0.62))
    y_auc = draw(ax_auc, auc_rows, "top-decile retrieval, ROC-AUC", (0.30, 0.90))

    # Centred docking approaches the structural baselines on ROC-AUC without
    # establishing equivalence or superiority. Drawn so the figure says so.
    kinase_row = y_auc[1]
    for value, name in (
        (float(kinase.loc["receptor_domain_sequence_identity", "roc_auc"]), "sequence"),
        (float(kinase.loc["klifs_pocket_identity", "roc_auc"]), "pocket"),
    ):
        ax_auc.plot([value, value], [kinase_row - 0.17, kinase_row + 0.17],
                    color=INK, lw=1.1, zorder=5)
    ax_auc.annotate(
        "sequence and pocket\nidentity baselines",
        xy=(0.713, kinase_row + 0.17), xytext=(0.745, kinase_row + 0.62),
        fontsize=7.0, color=INK, ha="center", va="bottom",
        arrowprops=dict(arrowstyle="-", lw=0.6, color=INK, shrinkA=1.0, shrinkB=1.0),
    )
    ax_auc.axvline(0.5, color=HAIR, lw=0.8, ls=(0, (3, 2)), zorder=1)
    ax_auc.text(0.505, y_auc[0] + 0.36, "chance", color=MUTED, fontsize=7.0, ha="left")

    panel_label(ax_rho, "a", x=-0.78)
    panel_label(ax_auc, "b", x=-0.60)

    # Name the one result that moves the wrong way, on the figure.
    ax_auc.annotate(
        "non-replication:\ncentring lowers\nboth metrics",
        xy=(0.7436, y_auc[-1]), xytext=(0.535, y_auc[-1] + 0.42),
        fontsize=7.0, color=CRIMSON, ha="center", va="bottom",
        arrowprops=dict(arrowstyle="-", lw=0.7, color=CRIMSON,
                        shrinkA=1.0, shrinkB=2.0),
    )

    handles = [
        Line2D([], [], marker="o", ms=4.6, ls="none", color="white", mec=MUTED,
               mew=1.3, label="raw docking map"),
        Line2D([], [], marker="o", ms=4.8, ls="none", color=TEAL, mec="white",
               mew=0.6, label="row-centred docking map"),
    ]
    figure.legend(handles=handles, loc="outside lower center", ncol=2,
                  handletextpad=0.35, columnspacing=1.8)
    save(figure, "fig3_external_agreement")


# --------------------------------------------------------------------------- #
# Figure 2 — matched-null calibration and ligand-domain dependence
# --------------------------------------------------------------------------- #

NULL_LABELS = [
    ("additive_gaussian", "additive Gaussian\n(target-specific variances)"),
    ("empirical_residual_column_permutation", "independent column\npermutation"),
    ("row_norm_preserving_random_direction", "random direction, each\nligand row norm preserved"),
]
DATASETS = [("docking44", "Docking-44", CLAY, "D"), ("dockstring58", "DOCKSTRING-58", TEAL, "o")]


def _mw_conditioned_null_repeats() -> pd.DataFrame:
    """Return the declared ten-bin molecular-weight null, and no control rows.

    The source file also holds a size-matched random-bin diagnostic.  Pooling the
    two designs would plot a mixture that corresponds to neither analysis.
    """
    frame = pd.read_csv(
        RESULTS / "public_residual_null_audit" / "mw_conditional_null_repeats.csv"
    )
    frame = frame[
        frame["requested_mw_bins"].eq(10)
        & frame["bin_design"].eq("molecular_weight_quantile")
    ].copy()
    counts = frame.groupby("dataset", sort=True).size().to_dict()
    expected = {key: 500 for key, *_ in DATASETS}
    if counts != expected:
        raise ValueError(f"unexpected ten-bin molecular-weight null support: {counts}")
    return frame


def figure_two() -> None:
    audit = json.loads((RESULTS / "public_residual_null_audit" / "summary.json").read_text())
    repeats = _mw_conditioned_null_repeats()

    figure = new_figure(WIDTH, 3.3)
    ax_null, ax_domain = figure.subplots(1, 2, width_ratios=[1.16, 0.84])

    # --- panel a: observed residual dimension against four matched nulls ---- #
    clean(ax_null)
    rows = [n for _, n in NULL_LABELS] + ["molecular-weight-conditioned\ntarget marginals"] + ["observed"]
    ypos = np.arange(len(rows), dtype=float)[::-1]

    for dataset, label, colour, marker in DATASETS:
        node = audit["datasets"][dataset]
        targets = node["full_processed_shape"][1]
        scale = float(targets - 1)  # residual dimension is bounded above by P-1
        offset = 0.16 if dataset == "docking44" else -0.16

        values = []
        for key, _ in NULL_LABELS:
            null = node["nulls"][key]["null_residual"]
            values.append((null["median"] / scale, [v / scale for v in null["interval_95"]]))
        mw = repeats[repeats["dataset"] == dataset]["participation_ratio"]
        values.append((float(mw.median()) / scale,
                       [float(mw.quantile(0.025)) / scale, float(mw.quantile(0.975)) / scale]))
        values.append((node["observed_residual_pr"] / scale, None))

        for y, (point, interval) in zip(ypos, values):
            if interval is not None:
                ax_null.plot(interval, [y + offset] * 2, color=colour, lw=1.4,
                             solid_capstyle="round", zorder=2)
            ax_null.plot(point, y + offset, marker=marker, ms=5.0, color=colour,
                         mec="white", mew=0.7, zorder=3,
                         label=label if y == ypos[-1] else None)

    ax_null.set_yticks(ypos)
    ax_null.set_yticklabels(rows)
    ax_null.set_xlabel("residual dimension / ($P-1$)")
    ax_null.set_xlim(0, 1.05)
    ax_null.axhline(0.5, color=HAIR, lw=0.8)
    ax_null.set_ylim(-0.6, len(rows) - 0.4)
    ax_null.legend(loc="upper left", bbox_to_anchor=(0.01, 0.99), handletextpad=0.35)
    panel_label(ax_null, "a", x=-0.52, y=1.04)

    # --- panel b: does the map transport across the molecular-size domain? -- #
    clean(ax_domain)
    observed = pd.read_csv(RESULTS / "public_chemical_domain_controls" / "observed_low_high_mw_contrasts.csv")
    observed = observed[observed["transformation"] == "row_centered_residual"]
    control = pd.read_csv(RESULTS / "public_chemical_domain_controls" / "mw_matched_group_disjoint_summary.csv")
    control = control[
        (control["transformation"] == "row_centered_residual")
        & (control["control_type"] == "mw_matched_chemical_group_disjoint")
    ]

    names = {"docking44": "Docking-44", "dockstring58": "DOCKSTRING-58"}
    ypos2 = np.array([1.0, 0.0])
    for y, (dataset, label, colour, marker) in zip(ypos2, DATASETS):
        obs = float(observed[observed["dataset"] == names[dataset]]["geometry_spearman"].iloc[0])
        ctl = control[control["dataset"] == names[dataset]].iloc[0]
        ax_domain.plot(
            [float(ctl["geometry_spearman_q025"]), float(ctl["geometry_spearman_q975"])],
            [y] * 2, color=MUTED, lw=3.0, solid_capstyle="round", alpha=0.5, zorder=2,
        )
        ax_domain.plot(float(ctl["geometry_spearman_median"]), y, marker="|", ms=9,
                       color=INK, mew=1.2, zorder=3)
        ax_domain.plot(obs, y, marker=marker, ms=6.0, color=colour, mec="white",
                       mew=0.8, zorder=4)

    ax_domain.set_yticks(ypos2)
    ax_domain.set_yticklabels([label for _, label, _, _ in DATASETS])
    ax_domain.set_xlabel("map agreement, $\\rho$")
    ax_domain.set_xlim(0, 1.05)
    ax_domain.set_ylim(-0.65, 1.65)
    ax_domain.annotate(
        "size-matched,\nchemical-group-disjoint\nhalves",
        xy=(0.95, ypos2[0] + 0.13), xytext=(0.42, ypos2[0] + 0.34),
        fontsize=7.0, color=INK, ha="center", va="bottom",
        arrowprops=dict(arrowstyle="-", lw=0.6, color=INK, shrinkA=1.0, shrinkB=1.5),
    )
    ax_domain.text(0.135, ypos2[1] - 0.42, "observed low/high split",
                   fontsize=7.0, color=INK, ha="left", va="center")
    panel_label(ax_domain, "b", x=-0.42, y=1.04)

    save(figure, "fig2_residual_structure")


# --------------------------------------------------------------------------- #
# Figure 4 — map-recovery cost and an exploratory five-mode comparison
# --------------------------------------------------------------------------- #

def figure_four() -> None:
    recovery = pd.read_csv(RESULTS / "calibration_panel_recovery" / "summary.csv")
    core = pd.read_csv(RESULTS / "biological_core_modes" / "panel_metrics.csv")

    figure = new_figure(WIDTH, 3.2)
    ax_cost, ax_core = figure.subplots(1, 2, width_ratios=[1.0, 1.0])

    # --- panel a: how many docked ligands the map needs -------------------- #
    clean(ax_cost, grid="both")
    for name, colour, marker in (("Docking-44", CLAY, "D"), ("DOCKSTRING-58", TEAL, "o")):
        frame = recovery[recovery["dataset"] == name].sort_values("calibration_ligands")
        x = frame["calibration_ligands"].to_numpy(dtype=float)
        ax_cost.fill_between(x, frame["geometry_spearman_q025"], frame["geometry_spearman_q975"],
                             color=colour, alpha=0.16, lw=0, zorder=1)
        ax_cost.plot(x, frame["geometry_spearman_mean"], color=colour, lw=1.4,
                     marker=marker, ms=3.6, mec="white", mew=0.6, zorder=3, label=name)
    ax_cost.set_xscale("log")
    ax_cost.set_xlabel("docked calibration ligands (log scale)")
    ax_cost.set_ylabel("agreement with the full-matrix map, $\\rho$")
    recovery_floor = float(recovery["geometry_spearman_q025"].min()) - 0.02
    ax_cost.set_ylim(recovery_floor, 1.005)
    ax_cost.axvline(200, color=CRIMSON, lw=0.9, ls=(0, (4, 2)), zorder=2)
    ax_cost.annotate(
        "200 ligands:\n$\\rho$ = 0.92–0.94\nDOCKSTRING: 1,300$\\times$ fewer cells",
        xy=(200, 0.845), xytext=(255, 0.775),
        fontsize=7.0, color=CRIMSON, ha="left", va="center",
        arrowprops=dict(arrowstyle="-", lw=0.6, color=CRIMSON, shrinkA=1.0, shrinkB=1.5),
    )
    ax_cost.legend(loc="lower right", handletextpad=0.4, bbox_to_anchor=(1.0, 0.0))
    panel_label(ax_cost, "a", x=-0.16, y=1.04)

    # --- panel b: exploratory five-mode comparison ------------------------- #
    clean(ax_core)
    support = "exact_connectivity_excluded"
    core = core[core["reference_support"] == support].copy()
    panels = ["DAVIS", "PKIS2", "PKIS1", "KiRHub"]
    metrics = [
        ("normalized_low_mode_minus_full_continuous_spearman", "$\\rho$", TEAL, ""),
        ("normalized_low_mode_minus_full_roc_auc", "ROC-AUC", SLATE, "///"),
        ("normalized_low_mode_minus_full_average_precision", "avg. precision", SAND, ""),
    ]
    height = 0.26
    ypos = np.arange(len(panels), dtype=float)[::-1]
    for offset, (column, label, colour, hatch) in zip((height, 0.0, -height), metrics):
        values = []
        for panel in panels:
            row = core[core["panel"] == panel]
            if len(row) != 1:
                raise ValueError(
                    f"expected one {support!r} row for {panel}, found {len(row)}"
                )
            values.append(float(row[column].iloc[0]))
        ax_core.barh(ypos + offset, values, height=height * 0.92, color=colour,
                     edgecolor="white", lw=0.5, hatch=hatch, zorder=3, label=label)
    ax_core.axvline(0, color=INK, lw=0.8, zorder=4)
    ax_core.set_yticks(ypos)
    ax_core.set_yticklabels(panels)
    ax_core.set_xlabel("five-mode truncation $-$ full residual map")
    ax_core.set_ylim(-0.6, len(panels) - 0.05)
    ax_core.legend(loc="lower left", handletextpad=0.4, labelspacing=0.3)
    ax_core.text(
        0.40, 0.98,
        "average precision is the metric\nthat does not follow",
        transform=ax_core.transAxes, fontsize=7.0, color=INK, ha="left", va="top",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.92, pad=1.5),
    )
    panel_label(ax_core, "b", x=-0.14, y=1.04)

    save(figure, "fig4_cost_and_core")


if __name__ == "__main__":
    configure_style()
    figure_one()
    figure_two()
    figure_three()
    figure_four()
