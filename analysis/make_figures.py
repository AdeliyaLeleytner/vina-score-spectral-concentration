#!/usr/bin/env python3
"""Generate the five vector figures for the dimensionality manuscript.

Every plotted value is read from a versioned source table or cached analysis
result in this repository.  The script deliberately does not touch the live
``reproduce/`` selectivity analysis, whose estimand and readiness differ from
this manuscript.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PACKAGE = Path(__file__).resolve().parents[1]
LEGACY_ROOT = PACKAGE.parent
FROZEN = PACKAGE / "data" / "frozen"
OUT = PACKAGE / "figures"
WIDTH = 6.30  # article text width under the manuscript's A4/2.5-cm geometry
PALETTE = {
    "ink": "#1A1A1A",
    "blue": "#2C6FAD",
    "orange": "#E07A3E",
    "teal": "#3E8E8A",
    "plum": "#7D5BA6",
    "grey": "#9AA0A6",
    "pale": "#EEF2F6",
}
BLUE = PALETTE["blue"]
ORANGE = PALETTE["orange"]
TEAL = PALETTE["teal"]
PLUM = PALETTE["plum"]
GREY = PALETTE["grey"]
INK = PALETTE["ink"]
PALE = PALETTE["pale"]
RELEASE_TIMESTAMP = datetime(2026, 8, 1, tzinfo=timezone.utc)


def use_paper_style() -> None:
    """Apply the final-size manuscript style without an external style package."""
    matplotlib.rcParams.update({
        "figure.constrained_layout.use": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "axes.titlecolor": INK,
        "axes.titlesize": 8,
        "axes.titleweight": "normal",
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
        "lines.linewidth": 1.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.facecolor": "white",
        "savefig.transparent": False,
    })


def audit_fig(fig: plt.Figure, compare: list[plt.Axes] | None = None) -> bool:
    """Fail on the mechanical errors most likely to make a print figure unusable."""
    errors: list[str] = []
    width = float(fig.get_size_inches()[0])
    if abs(width - WIDTH) > 0.01:
        errors.append(f"figure width {width:.3f} in does not match {WIDTH:.2f} in")
    for text_artist in fig.findobj(match=matplotlib.text.Text):
        if text_artist.get_text().strip() and text_artist.get_fontsize() < 6:
            errors.append(f"text smaller than 6 pt: {text_artist.get_text()!r}")
    for ax in fig.axes:
        if ax.get_xscale() == "log" and "log" not in ax.get_xlabel().lower():
            errors.append("logarithmic x axis is not declared in its label")
        if ax.get_yscale() == "log" and "log" not in ax.get_ylabel().lower():
            errors.append("logarithmic y axis is not declared in its label")
    if compare:
        xlims = {tuple(np.round(ax.get_xlim(), 8)) for ax in compare}
        ylims = {tuple(np.round(ax.get_ylim(), 8)) for ax in compare}
        if len(xlims) > 1 or len(ylims) > 1:
            errors.append("compared axes do not share limits")
    if errors:
        raise AssertionError("; ".join(sorted(set(errors))))
    return True


def save(fig: plt.Figure, path: str, dpi: int = 400) -> None:
    """Write an exact-size vector PDF and a same-size PNG preview."""
    base = Path(path)
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        base.with_suffix(".pdf"),
        metadata={
            "Title": base.stem,
            "Author": "Adeliya Leleytner; Victor Safronov; Maxim Fedorov",
            "Creator": "make_figures.py",
            "CreationDate": RELEASE_TIMESTAMP,
            "ModDate": RELEASE_TIMESTAMP,
        },
    )
    fig.savefig(base.with_suffix(".png"), dpi=dpi)
    width, height = fig.get_size_inches()
    print(f"wrote {base.with_suffix('.pdf')} ({width:.3f} × {height:.3f} in)")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def source_path(relative: str | Path) -> Path:
    relative = Path(relative)
    candidates = [FROZEN / relative, Path(f"{FROZEN / relative}.gz")]
    if os.environ.get("JCHEMINF_REQUIRE_BUNDLED") != "1":
        candidates.append(LEGACY_ROOT / relative)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Frozen input not found: {relative}")


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.14, 1.05, label, transform=ax.transAxes, ha="left", va="top",
            fontsize=9, fontweight="bold", color=INK)


def clean(ax: plt.Axes) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#E1E5E9", linewidth=0.55, zorder=0)
    ax.tick_params(length=2.5, width=0.6)


def fig1(evidence: dict) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(WIDTH, 4.55),
                             gridspec_kw={"height_ratios": [1.0, 1.05]})
    (a, b), (c, d) = axes
    panels = [
        (a, "Docking-44", evidence["centering_ladder"]["docking44"]["raw"], BLUE),
        (b, "DOCKSTRING-58", evidence["centering_ladder"]["dockstring58"]["raw"], ORANGE),
    ]
    for ax, title, rec, color in panels:
        eig = np.asarray(rec["top10_eigenvalues"], dtype=float)
        frac = eig / rec["n_targets"]
        x = np.arange(1, len(frac) + 1)
        ax.plot(x, frac, marker="o", ms=3.2, lw=1.1, color=color)
        ax.fill_between(x, frac, 1e-3, color=color, alpha=0.10)
        ax.set_yscale("log")
        ax.set_xlim(0.7, 10.3)
        ax.set_ylim(0.003, 1.0)
        ax.set_xticks([1, 2, 4, 6, 8, 10])
        ax.set_xlabel("eigenvalue rank")
        ax.set_title(title, loc="left")
        ax.text(0.98, 0.93,
                f"$r_{{PR}}$ = {rec['participation_ratio']:.2f}\n"
                f"PC1 = {100 * rec['pc1_fraction']:.0f}%",
                transform=ax.transAxes, ha="right", va="top", fontsize=7.5)
        clean(ax)
    a.set_ylabel("variance fraction (log scale)")
    b.set_ylabel("variance fraction (log scale)")
    a.sharey(b)

    uncertainty = evidence["target_panel_uncertainty"]

    def plot_target_curve(ax: plt.Axes, surface: str) -> None:
        for prefix, color, name in [
            ("docking44", BLUE, "Docking-44"),
            ("dockstring58", ORANGE, "DOCKSTRING-58"),
        ]:
            records = uncertainty[f"{prefix}_unstratified_subsamples"]
            xvals = np.asarray([row["n_targets"] for row in records])
            med = np.asarray([row[surface]["median"] for row in records])
            lo = np.asarray([row[surface]["q025"] for row in records])
            hi = np.asarray([row[surface]["q975"] for row in records])
            ax.fill_between(xvals, lo, hi, color=color, alpha=0.12)
            ax.plot(xvals, med, marker="o", ms=3, color=color, label=name)
            stratified = uncertainty[f"{prefix}_family_stratified_subsamples"]
            sx = np.asarray([row["n_targets"] for row in stratified])
            sm = np.asarray([row[surface]["median"] for row in stratified])
            ax.plot(sx, sm, marker="s", ms=2.6, ls="--", color=color, alpha=0.95)
        ax.set_xlim(2, 60)
        ax.set_xlabel("targets sampled")
        ax.set_ylabel("PR effective dimension")
        clean(ax)

    plot_target_curve(c, "raw")
    c.set_ylim(0.8, 3.2)
    c.set_title("column-standardized surface", loc="left")
    c.text(0.02, 0.95, "solid: random\ndashed: family-stratified",
           transform=c.transAxes, va="top", fontsize=6.5)
    plot_target_curve(d, "interaction")
    d.set_ylim(0.5, 20.5)
    d.set_title("two-way-centered residual surface", loc="left")
    d.legend(frameon=False, fontsize=6.5, loc="upper left")
    for ax, label in zip([a, b, c, d], "abcd"):
        panel_label(ax, label)
    fig.subplots_adjust(left=0.10, right=0.985, bottom=0.11, top=0.96, wspace=0.30, hspace=0.42)
    assert audit_fig(fig, compare=[a, b])
    save(fig, str(OUT / "fig1_raw_collapse"))
    plt.close(fig)


def fig_s3_scoring_controls(evidence: dict) -> None:
    dock = load_json(source_path("negative_results_paper/analysis/rank_robustness_summary.json"))
    ds = load_json(source_path("negative_results_paper/analysis/review_dockstring_rmt_summary.json"))["complete_data"]
    rf1 = load_json(source_path("negative_results_paper/analysis/rescore_ml_rfscore_summary.json"))
    rf2 = load_json(source_path("negative_results_paper/analysis/rescore_ml_rfscore_v2_summary.json"))
    expanded = load_json(source_path("negative_results_paper/analysis/rmt_panel_v4_contrast.json"))
    rows = [
        ("Vina, Docking-44", dock["pearson"]["eff_rank"], *dock["cluster_bootstrap_eff_rank"]["ci95"], "12,651 × 44"),
        ("Vina, DOCKSTRING", ds["effective_rank"], *ds["effective_rank_boot95"], "260,060 × 58"),
        ("RF-Score, Vina poses", rf1["eff_rank_rfscore"], *rf1["bootstrap95_eff_rank"]["rfscore"], "41 × 14"),
        ("RF-Score, Vinardo poses", rf2["eff_rank_rfscore_vinardoposes"], *rf2["eff_rank_rfscore_boot95"], "35 × 20"),
        ("Vina, Docking-47", expanded["panel_47_with_new_hepatic"]["eff_rank_PR"], np.nan, np.nan, "12,651 × 47"),
    ]
    frame = pd.DataFrame(rows, columns=["label", "rank", "lo", "hi", "shape"])
    fig, ax = plt.subplots(figsize=(WIDTH, 3.05))
    y = np.arange(len(frame))[::-1]
    for yi, row in zip(y, frame.itertuples()):
        color = ORANGE if "RF-Score" in row.label else BLUE
        if np.isfinite(row.lo):
            ax.plot([row.lo, row.hi], [yi, yi], color=GREY, lw=1.5, zorder=1)
            ax.plot([row.lo, row.hi], [yi, yi], marker="|", color=GREY, lw=0, ms=7)
        ax.plot(row.rank, yi, marker="o", color=color, ms=6, mec="white", mew=0.7, zorder=3)
        ax.text(row.rank + 0.055, yi, f"{row.rank:.2f}   {row.shape}", va="center", fontsize=7)
    ax.axvspan(1.0, 2.5, color=PALE, zorder=-2)
    ax.set_yticks(y, frame.label)
    ax.set_xlim(0.9, 3.05)
    ax.set_xlabel("participation-ratio effective dimension")
    ax.set_ylabel("score surface")
    clean(ax)
    fig.subplots_adjust(left=0.34, right=0.98, bottom=0.20, top=0.95)
    assert audit_fig(fig)
    save(fig, str(OUT / "figS3_scoring_controls"))
    plt.close(fig)


def fig2(evidence: dict) -> None:
    phys = load_json(source_path("negative_results_paper/analysis/physchem_depth_summary.json"))
    robust = load_json(source_path("negative_results_paper/analysis/rank_robustness_summary.json"))
    transforms = [
        ("column-standardized", phys["effective_rank"]["raw_docking"]["eff_rank"]),
        ("linear residuals", phys["effective_rank"]["size_residualized"]["eff_rank"]),
        ("nonlinear residuals", robust["nonlinear_size_removed"]["eff_rank"]),
        ("17–22 heavy atoms", robust["within_size_band"]["eff_rank"]),
        ("score / heavy atoms", robust["ligand_efficiency"]["eff_rank"]),
    ]
    r2 = phys["mean_docking_is_size"]
    controls = [
        ("7 descriptors", r2["R2_on_7_size_descriptors"]),
        ("heavy atoms", r2["R2_on_HeavyAtoms_only"]),
        ("Labute ASA", r2["R2_on_LabuteASA_only"]),
        ("cLogP", r2["R2_on_clogP_only"]),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(WIDTH, 5.25),
                             gridspec_kw={"height_ratios": [0.78, 1.22]})
    (a, b), (c, d) = axes
    ya = np.arange(len(transforms))[::-1]
    for yi, (name, val) in zip(ya, transforms):
        a.plot([1, val], [yi, yi], color=GREY, lw=1.1)
        a.plot(val, yi, "o", ms=5.7, color=ORANGE if "residual" in name else BLUE, mec="white", mew=0.6)
        a.text(val + 0.07, yi, f"{val:.2f}", va="center", fontsize=7)
    a.axvline(2, color=GREY, ls="--", lw=0.8)
    a.set_yticks(ya, [x[0] for x in transforms])
    a.set_xlim(0.8, 3.9)
    a.set_xlabel("effective dimension")
    a.set_ylabel("matrix transformation")
    clean(a)

    yb = np.arange(len(controls))[::-1]
    vals = np.asarray([x[1] for x in controls])
    b.barh(yb, vals, color=[ORANGE, BLUE, BLUE, BLUE], height=0.57)
    for yi, val in zip(yb, vals):
        b.text(val + 0.018, yi, f"{val:.2f}", va="center", fontsize=7)
    b.set_yticks(yb, [x[0] for x in controls])
    b.set_xlim(0, 0.70)
    b.set_xlabel("$R^2$ for mean docking score")
    b.set_ylabel("ligand descriptors")
    clean(b)
    def loading_panel(ax: plt.Axes, key: str, title: str, color: str) -> None:
        record = evidence["pc1_axis"][key]
        frame = pd.DataFrame({
            "target": record["targets"],
            "family": record["families"],
            "loading": record["loadings"],
        })
        family_order = (
            frame.groupby("family").loading.median().sort_values().index.tolist()
        )
        for position, family in enumerate(family_order):
            subset = frame[frame.family.eq(family)].sort_values("loading")
            jitter = np.linspace(-0.18, 0.18, len(subset)) if len(subset) > 1 else np.array([0.0])
            ax.scatter(subset.loading, position + jitter, color=color, s=17,
                       edgecolors="white", linewidths=0.35, zorder=3)
        minimum = frame.loc[frame.loading.idxmin()]
        minimum_position = family_order.index(minimum.family)
        ax.annotate(str(minimum.target), (minimum.loading, minimum_position),
                    xytext=(5, -7), textcoords="offset points", fontsize=6.3,
                    arrowprops={"arrowstyle": "-", "color": GREY, "lw": 0.4})
        ax.axvline(0, color=INK, lw=0.7, ls="--")
        ax.set_yticks(np.arange(len(family_order)), family_order)
        ax.set_xlim(-0.04, 0.18)
        ax.set_xlabel("PC1 target loading")
        ax.set_ylabel("target family")
        ax.set_title(title, loc="left")
        ax.text(0.02, 0.97,
                f"positive: {record['n_positive_loadings']}/{len(frame)}\n"
                f"corr(PC1, raw row mean)={record['correlation_with_raw_per_ligand_mean']:.3f}",
                transform=ax.transAxes, ha="left", va="top", fontsize=6.2,
                bbox={"boxstyle": "round,pad=0.15", "fc": "white", "ec": "none", "alpha": 0.88})
        clean(ax)

    loading_panel(c, "docking44", "Docking-44 loadings", BLUE)
    loading_panel(d, "dockstring58", "DOCKSTRING-58 loadings", ORANGE)
    for ax, label in zip([a, b, c, d], "abcd"):
        panel_label(ax, label)
    fig.subplots_adjust(left=0.23, right=0.985, bottom=0.095, top=0.965,
                        wspace=0.70, hspace=0.42)
    assert audit_fig(fig)
    save(fig, str(OUT / "fig2_shared_axis"))
    plt.close(fig)


def fig3(evidence: dict) -> None:
    paired = evidence["matched_raw_and_interaction_bootstrap"]
    imputation = evidence["matched_target_preference_benchmark"]["multiple_imputation"][
        "participation_ratio"
    ]
    fig, (a, b) = plt.subplots(1, 2, figsize=(WIDTH, 3.25),
                              gridspec_kw={"width_ratios": [0.9, 1.1]})
    surfaces = ["raw", "interaction"]
    labels = ["column-standardized", "residual"]
    x = np.arange(2)
    docking = np.asarray([paired[key]["docking_participation_ratio"] for key in surfaces])
    experiment = np.asarray([paired[key]["experimental_participation_ratio"] for key in surfaces])
    a.plot(x, docking, "o-", color=BLUE, markerfacecolor="white", markeredgewidth=1.1)
    a.plot(x, experiment, "s-", color=ORANGE, markerfacecolor="white", markeredgewidth=1.1)
    for position, value in zip(x, docking):
        a.text(position - 0.04, value - 0.16, f"{value:.2f}", ha="right", va="top", fontsize=7)
    for position, value in zip(x, experiment):
        a.text(position + 0.04, value + 0.13, f"{value:.2f}", ha="left", va="bottom", fontsize=7)
    a.set_xticks(x, labels, rotation=12)
    a.set_xlim(-0.35, 1.35)
    a.set_ylim(1.45, 4.65)
    a.set_ylabel("PR effective dimension")
    a.set_xlabel("matched 74 × 6 surface")
    a.text(0.04, 0.96, "docking", transform=a.transAxes, color=BLUE,
           va="top", fontsize=7.2)
    a.text(0.31, 0.96, "experiment", transform=a.transAxes, color=ORANGE,
           va="top", fontsize=7.2)
    clean(a)

    y = np.array([1, 0])
    for yi, key, mi_key in zip(y, surfaces, ["raw_difference", "residual_difference"]):
        plugin = paired[key]["plugin_difference_experiment_minus_docking"]
        lo, hi = paired[key]["bootstrap_difference_95_interval"]
        mi = imputation[mi_key]
        b.plot(mi["interval_95"], [yi, yi], color=ORANGE, lw=5.5, alpha=0.24,
               solid_capstyle="butt", zorder=1)
        b.plot([lo, hi], [yi, yi], color=INK, lw=1.2, zorder=2)
        b.plot([lo, hi], [yi, yi], marker="|", color=INK, lw=0, ms=7, zorder=2)
        b.plot(plugin, yi, "o", color=BLUE, ms=5.5, mec="white", mew=0.55, zorder=3)
        b.text(plugin + 0.06, yi + 0.12, f"Δ={plugin:.2f}", fontsize=7, va="bottom")
    b.axvline(0, color=GREY, ls="--", lw=0.8)
    b.set_yticks(y, labels)
    b.set_ylim(-0.55, 1.55)
    b.set_xlim(-0.35, 2.85)
    b.set_xlabel("experiment − docking PR dimension")
    b.set_ylabel("surface")
    b.text(0.98, 0.96, "black: paired ligand bootstrap\norange: 95% across 100 imputations",
           transform=b.transAxes, ha="right", va="top", fontsize=6.4)
    clean(b)
    panel_label(a, "a"); panel_label(b, "b")
    fig.subplots_adjust(left=0.12, right=0.985, bottom=0.22, top=0.94, wspace=0.52)
    assert audit_fig(fig)
    save(fig, str(OUT / "fig3_matched_experiment"))
    plt.close(fig)


def fig_s4_learned_affinity(evidence: dict) -> None:
    centers = evidence["centering_ladder"]
    ladder = [
        ("Docking-44", centers["docking44"]["raw"]["participation_ratio"], centers["docking44"]["interaction"]["participation_ratio"]),
        ("DOCKSTRING", centers["dockstring58"]["raw"]["participation_ratio"], centers["dockstring58"]["interaction"]["participation_ratio"]),
        ("RF-Score/Vina", centers["rfscore_v1_14targets"]["raw"]["participation_ratio"], centers["rfscore_v1_14targets"]["interaction"]["participation_ratio"]),
        ("RF-Score/Vinardo", centers["rfscore_v1_vinardo_20targets"]["raw"]["participation_ratio"], centers["rfscore_v1_vinardo_20targets"]["interaction"]["participation_ratio"]),
        ("DAVIS experiment", centers["davis_experiment_floor_excluded"]["raw"]["participation_ratio"], centers["davis_experiment_floor_excluded"]["interaction"]["participation_ratio"]),
        ("Boltz-2", centers["boltz2_davis"]["raw"]["participation_ratio"], centers["boltz2_davis"]["interaction"]["participation_ratio"]),
    ]
    fig, (a, b) = plt.subplots(1, 2, figsize=(WIDTH, 3.35),
                              gridspec_kw={"width_ratios": [1.2, 0.9]})
    colors = [BLUE, ORANGE, BLUE, ORANGE, BLUE, ORANGE]
    markers = ["o", "s", "^", "D", "P", "X"]
    for (name, raw, inter), color, marker in zip(ladder, colors, markers):
        a.plot([0, 1], [raw, inter], color=GREY, lw=0.9, alpha=0.85)
        a.scatter([0, 1], [raw, inter], s=31, color=color, marker=marker,
                  edgecolors="white", linewidths=0.5, zorder=3)
        a.text(1.04, inter, name, fontsize=6.6, va="center", color=INK)
    a.set_xticks([0, 1], ["column-standardized", "residual"])
    a.set_xlim(-0.18, 1.85)
    a.set_ylim(0, 20.2)
    a.set_ylabel("PR effective dimension")
    a.set_xlabel("surface transformation")
    clean(a)

    target_counts = np.asarray([44, 58, 14, 20, 442, 442], dtype=float)
    raw = np.asarray([row[1] for row in ladder]) / target_counts
    residual = np.asarray([row[2] for row in ladder]) / (target_counts - 1)
    names = [row[0] for row in ladder]
    y = np.arange(len(names))[::-1]
    b.barh(y + 0.16, raw, height=0.28, color=BLUE, label="column-standardized")
    b.barh(y - 0.16, residual, height=0.28, color=ORANGE, label="residual")
    b.set_yticks(y, names)
    b.set_xlim(0, max(0.5, residual.max() * 1.15))
    b.set_xlabel("PR dimension / algebraic maximum")
    b.set_ylabel("matrix")
    b.legend(frameon=False, fontsize=6.5, loc="lower right")
    clean(b)
    for ax, label in zip([a, b], "ab"):
        panel_label(ax, label)
    fig.subplots_adjust(left=0.105, right=0.985, bottom=0.18, top=0.94, wspace=0.67)
    assert audit_fig(fig)
    save(fig, str(OUT / "figS4_learned_affinity"))
    plt.close(fig)


def fig4(evidence: dict) -> None:
    centers = evidence["centering_ladder"]
    dock_boot = evidence["ligand_sampling_sensitivity"][
        "docking44_butina_cluster_surface_bootstrap"
    ]
    dockstring = evidence["ligand_sampling_sensitivity"]["dockstring_scaffold_bootstrap"]
    additive = evidence["additive_main_effect_null"]
    empirical = evidence["empirical_residual_permutation_null"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(WIDTH, 3.35),
                              gridspec_kw={"width_ratios": [1.05, 0.95]})
    x = np.array([0, 1])
    for key, label, color, marker in [
        ("docking44", "Docking-44", BLUE, "o"),
        ("dockstring58", "DOCKSTRING-58", ORANGE, "s"),
    ]:
        raw = centers[key]["raw"]["participation_ratio"]
        residual = centers[key]["interaction"]["participation_ratio"]
        a.plot(x, [raw, residual], color=color, lw=1.1, alpha=0.85)
        a.scatter(x, [raw, residual], color=color, marker=marker, s=37,
                  edgecolors="white", linewidths=0.55, zorder=3)
        if key == "docking44":
            raw_interval = dock_boot["bootstrap"]["raw"]["interval_95"]
            residual_interval = dock_boot["bootstrap"]["residual"]["interval_95"]
        else:
            raw_interval = [
                dockstring["support_point_estimates"]["raw"]["minimum"],
                dockstring["support_point_estimates"]["raw"]["maximum"],
            ]
            residual_interval = [
                dockstring["support_point_estimates"]["residual"]["minimum"],
                dockstring["support_point_estimates"]["residual"]["maximum"],
            ]
        a.errorbar(x, [raw, residual],
                   yerr=[[raw - raw_interval[0], residual - residual_interval[0]],
                         [raw_interval[1] - raw, residual_interval[1] - residual]],
                   fmt="none", ecolor=color, elinewidth=1.0, capsize=2, zorder=2)
        a.text(1.04, residual, label, fontsize=6.8, va="center", color=color)
    a.set_xticks(x, ["column-standardized", "residual"], rotation=12)
    a.set_xlim(-0.25, 1.62)
    a.set_ylim(0, 20.2)
    a.set_ylabel("PR effective dimension")
    a.set_xlabel("surface transformation")
    a.text(0.02, 0.96, "Docking-44: Butina bootstrap\nDOCKSTRING: five support estimates",
           transform=a.transAxes, ha="left", va="top", fontsize=6.2)
    clean(a)

    y = np.array([1, 0])
    for yi, key, label, maximum, color, marker in [
        (1, "docking44", "Docking-44", 43, BLUE, "o"),
        (0, "dockstring58", "DOCKSTRING-58", 57, ORANGE, "s"),
    ]:
        observed = centers[key]["interaction"]["participation_ratio"] / maximum
        null = additive[key]["null_residual"]
        null_median = null["median"] / maximum
        null_interval = np.asarray(null["interval_95"]) / maximum
        empirical_null = empirical[key]["null_residual"]
        empirical_median = empirical_null["median"] / maximum
        empirical_interval = np.asarray(empirical_null["interval_95"]) / maximum
        b.plot([observed, null_median], [yi, yi], color=GREY, lw=1.0)
        b.plot(null_interval, [yi + 0.08, yi + 0.08], color=INK, lw=2.0, alpha=0.42)
        b.plot(empirical_interval, [yi - 0.08, yi - 0.08], color=GREY, lw=2.0, alpha=0.60)
        b.scatter(observed, yi, color=color, marker=marker, s=40,
                  edgecolors="white", linewidths=0.55, zorder=3)
        b.scatter(null_median, yi + 0.08, facecolors="white", edgecolors=INK,
                  marker=marker, s=36, linewidths=0.8, zorder=3)
        b.scatter(empirical_median, yi - 0.08, color=GREY,
                  marker="x", s=30, linewidths=0.9, zorder=3)
        b.text(observed + 0.025, yi + 0.13,
               f"observed {centers[key]['interaction']['participation_ratio']:.1f}/{maximum}",
               fontsize=6.4, color=color)
    b.set_yticks(y, ["Docking-44", "DOCKSTRING-58"])
    b.set_xlim(0.12, 1.04)
    b.set_ylim(-0.55, 1.55)
    b.set_xlabel("residual PR / algebraic maximum")
    b.set_ylabel("matrix")
    b.text(0.98, 0.96, "filled: observed\nopen: Gaussian additive null\nx: empirical residual null",
           transform=b.transAxes, ha="right", va="top", fontsize=6.3)
    clean(b)
    panel_label(a, "a"); panel_label(b, "b")
    fig.subplots_adjust(left=0.12, right=0.985, bottom=0.22, top=0.94, wspace=0.48)
    assert audit_fig(fig)
    save(fig, str(OUT / "fig4_centering_null"))
    plt.close(fig)


def fig5(evidence: dict) -> None:
    benchmark = evidence["expanded_target_preference_benchmark"]["primary_all_exact"]
    representations = benchmark["representations"]
    rows = [
        ("two-way residual", "two_way_residual", TEAL, "^"),
        ("column-standardized", "column_standardized", ORANGE, "o"),
        ("absolute Vina", "absolute_vina", BLUE, "s"),
        ("docking target prior", "docking_target_prior", GREY, "D"),
        ("cohort experimental prior", "cohort_experimental_target_prior", PLUM, "P"),
    ]
    fig, (a, b) = plt.subplots(1, 2, figsize=(WIDTH, 3.45),
                              gridspec_kw={"width_ratios": [1.08, 0.92]})
    y = np.arange(len(rows))[::-1]
    for yi, (label, key, color, marker) in zip(y, rows):
        record = representations[key]
        observed = record["mean_per_ligand_pairwise_accuracy"]
        interval = record["scaffold_cluster_bootstrap"]["interval_95"]
        a.plot(interval, [yi, yi], color=color, lw=1.25, zorder=2)
        a.plot(interval, [yi, yi], marker="|", color=color, lw=0, ms=7, zorder=2)
        a.scatter(observed, yi, color=color, marker=marker, s=43,
                  edgecolors="white", linewidths=0.55, zorder=4)
        if "ligand_identity_shuffle_null" in record:
            null = record["ligand_identity_shuffle_null"]
            a.plot(null["interval_95"], [yi - 0.17, yi - 0.17], color=GREY,
                   lw=3.8, alpha=0.33, solid_capstyle="butt", zorder=1)
            a.plot(null["mean"], yi - 0.17, "x", color=INK, ms=4.5, mew=0.8, zorder=3)
            a.text(0.678, yi - 0.27,
                   f"perm. p={null['one_sided_empirical_p_observed_at_least_as_large']:.3f}",
                   ha="right", va="center", fontsize=6.0)
    a.axvline(0.5, color=INK, ls="--", lw=0.75)
    a.set_yticks(y, [row[0] for row in rows])
    a.set_xlim(0.43, 0.682)
    a.set_ylim(-0.55, len(rows) - 0.45)
    a.set_xlabel("pairwise target-preference accuracy")
    a.set_ylabel("operational representation")
    clean(a)

    coverage = benchmark["coverage_sensitivity"]
    minimums = np.asarray(sorted(int(value) for value in coverage), dtype=int)
    coverage_rows = [
        ("two-way residual", "two_way_residual", TEAL, "^", "-"),
        ("column-standardized", "column_standardized", ORANGE, "o", "--"),
        ("absolute Vina", "absolute_vina", BLUE, "s", (0, (3, 1, 1, 1))),
        ("docking target prior", "docking_target_prior", GREY, "D", ":"),
        ("cohort experimental prior", "cohort_experimental_target_prior", PLUM, "P", "-."),
    ]
    for label, key, color, marker, linestyle in coverage_rows:
        values = [
            coverage[str(minimum)]["representations"][key][
                "mean_per_ligand_pairwise_accuracy"
            ]
            for minimum in minimums
        ]
        b.plot(minimums, values, color=color, marker=marker, ls=linestyle,
               lw=1.0, ms=4.2, label=label)
        intervals = np.asarray([
            coverage[str(minimum)]["representations"][key][
                "scaffold_cluster_bootstrap"
            ]["interval_95"]
            for minimum in minimums
        ])
        b.vlines(minimums, intervals[:, 0], intervals[:, 1], color=color,
                 lw=0.55, alpha=0.30, zorder=1)
    b.axhline(0.5, color=INK, ls="--", lw=0.7)
    b.text(1.72, 0.474, "n:", ha="left", va="bottom", fontsize=6.0, color=INK)
    for minimum in minimums:
        b.text(minimum, 0.474, f"{coverage[str(minimum)]['n_ligands']}",
               ha="center", va="bottom", fontsize=6.0, color=INK)
    b.set_xticks(minimums)
    b.set_xlim(1.7, 7.55)
    b.set_ylim(0.47, 0.705)
    b.set_xlabel("minimum observed targets per ligand", fontsize=7.2)
    b.set_ylabel("pairwise accuracy")
    last_values = {
        key: coverage[str(minimums[-1])]["representations"][key][
            "mean_per_ligand_pairwise_accuracy"
        ]
        for _, key, _, _, _ in coverage_rows
    }
    direct_labels = {
        "two_way_residual": "residual",
        "column_standardized": "standardized",
        "absolute_vina": "absolute",
        "docking_target_prior": "docking prior",
        "cohort_experimental_target_prior": "cohort prior",
    }
    direct_offsets = {
        "two_way_residual": 0.0,
        "column_standardized": 0.0,
        "absolute_vina": -0.006,
        "docking_target_prior": 0.006,
        "cohort_experimental_target_prior": 0.0,
    }
    for label, key, color, _, _ in coverage_rows:
        b.text(6.16, last_values[key] + direct_offsets[key], direct_labels[key],
               color=color, fontsize=6.0,
               ha="left", va="center")
    clean(b)
    panel_label(a, "a"); panel_label(b, "b")
    fig.subplots_adjust(left=0.22, right=0.985, bottom=0.21, top=0.94, wspace=0.52)
    assert audit_fig(fig)
    save(fig, str(OUT / "fig5_target_preference"))
    plt.close(fig)


def fig_s2(evidence: dict) -> None:
    uncertainty = evidence["target_panel_uncertainty"]
    fig, axes = plt.subplots(2, 2, figsize=(WIDTH, 4.5))
    for row_index, (prefix, title, color) in enumerate([
        ("docking44", "Docking-44", BLUE),
        ("dockstring58", "DOCKSTRING-58", ORANGE),
    ]):
        jackknife = uncertainty[f"{prefix}_jackknife"]
        for col_index, (surface, label) in enumerate([
            ("raw", "column-standardized"),
            ("interaction", "two-way-centered residual"),
        ]):
            ax = axes[row_index, col_index]
            record = jackknife[surface]
            names = list(record["leave_one_target_out"])
            values = np.asarray(list(record["leave_one_target_out"].values()))
            order = np.argsort(values)
            ax.plot(np.arange(len(values)), values[order], "o", ms=2.7, color=color)
            ax.axhline(record["full_panel_on_jackknife_support"], color=INK, lw=0.8, ls="--")
            influential = record["most_influential_target"]
            position = int(np.flatnonzero(np.asarray(names)[order] == influential)[0])
            at_right = position >= len(values) - 4
            ax.annotate(influential, (position, values[order][position]),
                        xytext=(-4 if at_right else 4, 5), textcoords="offset points",
                        ha="right" if at_right else "left", fontsize=6.5)
            ax.set_title(f"{title}: {label}", loc="left")
            ax.set_xlabel("left-out target (ordered)")
            ax.set_ylabel("PR effective dimension")
            clean(ax)
    for ax, label in zip(axes.flat, "abcd"):
        panel_label(ax, label)
    fig.subplots_adjust(left=0.11, right=0.985, bottom=0.11, top=0.96,
                        wspace=0.31, hspace=0.40)
    assert audit_fig(fig)
    save(fig, str(OUT / "figS2_target_jackknife"))
    plt.close(fig)


def fig_s5_dti(evidence: dict) -> None:
    arms = pd.read_csv(
        PACKAGE / "results/dti_all_scorers_sensitivity.csv"
    )
    assoc = evidence["dti_associations"]["all_20_scorers_sensitivity"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(WIDTH, 3.55))
    offset_maps = {
        "raw_effective_rank": {
            1: (10, -12), 2: (12, 1), 3: (-14, -12), 4: (11, 10),
            5: (10, 12), 6: (-12, 15), 7: (-12, -16), 8: (10, -14),
            9: (10, 12), 10: (-12, 14), 11: (14, 7), 12: (-16, 15),
            13: (12, 11), 14: (14, 10), 15: (12, -20), 16: (-10, -14),
            17: (-14, 18), 18: (10, -22), 19: (14, 21), 20: (12, 10),
        },
        "interaction_effective_rank": {
            1: (10, -13), 2: (12, 4), 3: (-15, -12), 4: (-14, -13),
            5: (10, 10), 6: (-12, 15), 7: (-12, -17), 8: (10, -14),
            9: (10, 10), 10: (-12, 15), 11: (12, 8), 12: (-12, 15),
            13: (12, -12), 14: (12, 10), 15: (10, -15), 16: (-12, 15),
            17: (-10, 19), 18: (-12, -18), 19: (11, 19), 20: (-13, -6),
        },
    }
    for ax, rank_col, stats_key, xlab in [
        (a, "raw_effective_rank", "raw_rank_vs_selectivity_p_at_5",
         "column-standardized PR dimension"),
        (b, "interaction_effective_rank", "interaction_rank_vs_selectivity_p_at_5",
         "residual PR dimension"),
    ]:
        colors = np.where(arms.clears_chance, BLUE, GREY)
        ax.scatter(arms[rank_col], arms.selectivity_p_at_5, c=colors, s=56,
                   edgecolors="white", linewidths=0.45)
        for plot_id, row in enumerate(arms.itertuples(index=False), start=1):
            offset = offset_maps[rank_col][plot_id]
            ax.annotate(str(plot_id),
                        (getattr(row, rank_col), row.selectivity_p_at_5),
                        xytext=offset, textcoords="offset points", fontsize=6.0,
                        ha="center", va="center", color=INK,
                        bbox={"boxstyle": "round,pad=0.08", "fc": "white",
                              "ec": "none", "alpha": 0.82},
                        arrowprops={"arrowstyle": "-", "color": GREY, "lw": 0.35},
                        zorder=4)
        rec = assoc[stats_key]
        ax.text(0.98, 0.97,
                f"all 20 arms: Spearman r_s={rec['spearman_rho']:.2f}, p={rec['spearman_p']:.3f}",
                transform=ax.transAxes, ha="right", va="top", fontsize=6.5,
                bbox={"boxstyle": "round,pad=0.15", "fc": "white",
                      "ec": "none", "alpha": 0.90})
        ax.set_xlabel(xlab)
        ax.set_ylabel("selectivity precision at 5")
        clean(ax)
    a.scatter([], [], color=BLUE, s=28, label="above chance")
    a.scatter([], [], color=GREY, s=28, label="at chance")
    a.legend(frameon=False, fontsize=6.5, loc="center right")
    for ax, label in zip([a, b], "ab"):
        panel_label(ax, label)
    fig.subplots_adjust(left=0.10, right=0.985, bottom=0.18, top=0.94, wspace=0.34)
    assert audit_fig(fig)
    save(fig, str(OUT / "figS5_dti_all20"))
    plt.close(fig)


def fig_s1(evidence: dict) -> None:
    """Full column-standardized and residual scree plots for the Supplement."""
    fig, axes = plt.subplots(2, 2, figsize=(WIDTH, 4.55), sharey=True)
    for row, (key, name, color) in enumerate([
        ("docking44", "Docking-44", BLUE),
        ("dockstring58", "DOCKSTRING-58", ORANGE),
    ]):
        for col, (surface, label) in enumerate([
            ("raw", "column-standardized"),
            ("interaction", "two-way-centered residual"),
        ]):
            ax = axes[row, col]
            rec = evidence["centering_ladder"][key][surface]
            eig = np.asarray(rec["eigenvalues"], dtype=float) / rec["n_targets"]
            x = np.arange(1, len(eig) + 1)
            ax.plot(x, eig, marker="o", ms=2.4, color=color, lw=0.95)
            ax.set_yscale("log")
            ax.set_ylim(1e-5, 1.0)
            ax.set_xlim(0.5, len(eig) + 0.5)
            ax.set_title(f"{name}: {label}", loc="left")
            ax.set_xlabel("eigenvalue rank")
            ax.set_ylabel("variance fraction (log scale)")
            ax.text(0.97, 0.93, f"$r_{{PR}}$={rec['participation_ratio']:.2f}",
                    transform=ax.transAxes, ha="right", va="top", fontsize=7.5)
            clean(ax)
    for ax, label in zip(axes.flat, "abcd"):
        panel_label(ax, label)
    fig.subplots_adjust(left=0.10, right=0.985, bottom=0.11, top=0.96, wspace=0.26, hspace=0.38)
    assert audit_fig(fig)
    save(fig, str(OUT / "figS1_full_spectra"))
    plt.close(fig)


def fig_s6_dockstring_supports(evidence: dict) -> None:
    records = evidence["ligand_sampling_sensitivity"]["dockstring_scaffold_bootstrap"][
        "supports"
    ]
    fig, (a, b) = plt.subplots(1, 2, figsize=(WIDTH, 3.25))
    y = np.arange(len(records))[::-1]
    for ax, metric, label, color in [
        (a, "raw", "column-standardized PR", BLUE),
        (b, "residual", "residual PR", ORANGE),
    ]:
        for yi, record in zip(y, records):
            value = record["point_estimate"][metric]
            interval = record["scaffold_cluster_bootstrap"]["bootstrap"][metric][
                "interval_95"
            ]
            ax.plot(interval, [yi, yi], color=GREY, lw=1.2)
            ax.plot(interval, [yi, yi], marker="|", color=GREY, lw=0, ms=7)
            ax.plot(value, yi, "o", color=color, ms=5.5, mec="white", mew=0.5)
        ax.set_yticks(y, [f"support {record['support_number']}" for record in records])
        ax.set_xlabel(label)
        ax.set_ylabel("independent 15,000-molecule support")
        clean(ax)
    panel_label(a, "a"); panel_label(b, "b")
    fig.subplots_adjust(left=0.21, right=0.985, bottom=0.20, top=0.94, wspace=0.55)
    assert audit_fig(fig)
    save(fig, str(OUT / "figS6_dockstring_supports"))
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    use_paper_style()
    evidence = load_json(PACKAGE / "results/evidence_summary.json")
    fig1(evidence)
    fig2(evidence)
    fig3(evidence)
    fig4(evidence)
    fig5(evidence)
    fig_s1(evidence)
    fig_s2(evidence)
    fig_s3_scoring_controls(evidence)
    fig_s4_learned_affinity(evidence)
    fig_s5_dti(evidence)
    fig_s6_dockstring_supports(evidence)
    print(f"Wrote manuscript figures to {OUT}")


if __name__ == "__main__":
    main()
