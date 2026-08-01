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
    fig.savefig(base.with_suffix(".pdf"))
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


def fig2(evidence: dict) -> None:
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
    save(fig, str(OUT / "fig2_scoring_controls"))
    plt.close(fig)


def fig3() -> None:
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
    fig, (a, b) = plt.subplots(1, 2, figsize=(WIDTH, 3.25), gridspec_kw={"width_ratios": [1.1, 0.9]})
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
    panel_label(a, "a"); panel_label(b, "b")
    fig.subplots_adjust(left=0.24, right=0.98, bottom=0.18, top=0.94, wspace=0.63)
    assert audit_fig(fig)
    save(fig, str(OUT / "fig3_general_axis"))
    plt.close(fig)


def matched_matrices() -> tuple[pd.DataFrame, pd.DataFrame]:
    targets = ["5va1", "6cm4", "7wc9", "8pjk", "3rze", "7ym8"]
    dock = pd.read_csv(source_path("negative_results_paper/analysis/honest_dock_scores.csv"))
    exp = pd.read_csv(source_path("negative_results_paper/analysis/chembl_affinity_long.csv")).rename(columns={"pdb": "target", "pchembl": "pexp"})
    dmat = dock[dock.target.isin(targets)].pivot_table(index="inchikey", columns="target", values="dock")
    emat = exp[exp.target.isin(targets)].pivot_table(index="inchikey", columns="target", values="pexp")
    dmat = dmat.reindex(columns=targets)
    emat = emat.reindex(columns=targets)
    common = dmat.dropna().index.intersection(emat[emat.notna().sum(axis=1) >= 5].index)
    dmat = dmat.loc[common]
    emat = emat.loc[common].fillna(emat.loc[common].median())
    return dmat, emat


def fig4(evidence: dict) -> None:
    controls = evidence["experimental_sensitivity"]["activity_level_curated_matched_blocks"]
    pairs = [
        ("exact relation\nmedian", controls["all_exact_median"]),
        ("human binding\nall endpoints", controls["human_binding_all_endpoints"]),
        ("human binding\nKi/Kd", controls["human_binding_Ki_Kd"]),
    ]
    fig, ax = plt.subplots(figsize=(WIDTH, 3.15))
    y = np.arange(len(pairs))[::-1]
    for yi, (label, record) in zip(y, pairs):
        dock = record["matched_docking_participation_ratio"]
        experiment = record["experimental_participation_ratio"]
        ax.plot([dock, experiment], [yi, yi], color=GREY, lw=1.3, zorder=1)
        ax.scatter(dock, yi, s=46, color=BLUE, edgecolors="white", linewidths=0.7, zorder=3)
        ax.scatter(experiment, yi, s=46, color=ORANGE, edgecolors="white", linewidths=0.7, zorder=3)
        ax.text(dock - 0.08, yi + 0.16, f"{dock:.2f}", ha="right", fontsize=7)
        ax.text(experiment + 0.08, yi + 0.16, f"{experiment:.2f}", ha="left", fontsize=7)
        ax.text(4.35, yi, f"N={record['n_ligands']}, P={record['n_targets']}",
                va="center", fontsize=7, color=INK)
    ax.set_yticks(y, [item[0] for item in pairs])
    ax.set_xlim(1.35, 5.05)
    ax.set_ylim(-0.55, len(pairs) - 0.45)
    ax.set_xlabel("column-standardized PR effective dimension")
    ax.set_ylabel("matched ChEMBL curation")
    ax.text(0.02, 0.96, "docking", color=BLUE, transform=ax.transAxes,
            ha="left", va="top", fontsize=7.5)
    ax.text(0.16, 0.96, "experiment", color=ORANGE, transform=ax.transAxes,
            ha="left", va="top", fontsize=7.5)
    clean(ax)
    fig.subplots_adjust(left=0.27, right=0.98, bottom=0.20, top=0.95)
    assert audit_fig(fig)
    save(fig, str(OUT / "fig4_matched_experiment"))
    plt.close(fig)


def fig5(evidence: dict) -> None:
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
    save(fig, str(OUT / "fig5_centering_residual"))
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


def fig_s3(evidence: dict) -> None:
    arms = pd.read_csv(
        PACKAGE / "results/dti_all_scorers_sensitivity.csv"
    )
    assoc = evidence["dti_associations"]["all_20_scorers_sensitivity"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(WIDTH, 3.55))
    for ax, rank_col, stats_key, xlab in [
        (a, "raw_effective_rank", "raw_rank_vs_selectivity_p_at_5",
         "column-standardized PR dimension"),
        (b, "interaction_effective_rank", "interaction_rank_vs_selectivity_p_at_5",
         "residual PR dimension"),
    ]:
        colors = np.where(arms.clears_chance, BLUE, GREY)
        ax.scatter(arms[rank_col], arms.selectivity_p_at_5, c=colors, s=56,
                   edgecolors="white", linewidths=0.45)
        label_offsets = {
            1: (7, -8), 2: (8, 10), 3: (-9, -8), 4: (-8, -8),
            5: (7, 8),
            6: (-9, 8), 7: (-8, -10), 11: (8, 8), 12: (-11, 15),
            8: (8, 10), 9: (7, 8), 10: (-8, 10), 13: (8, -8),
            14: (8, 8), 15: (8, -11), 16: (-8, 10), 17: (-8, 8),
            18: (9, -16), 19: (10, 16), 20: (8, 8),
        }
        for plot_id, row in enumerate(arms.itertuples(index=False), start=1):
            offset = label_offsets.get(plot_id, (4, 4))
            ax.annotate(str(plot_id),
                        (getattr(row, rank_col), row.selectivity_p_at_5),
                        xytext=offset, textcoords="offset points", fontsize=6.0,
                        ha="center", va="center", color=INK,
                        arrowprops={"arrowstyle": "-", "color": GREY, "lw": 0.35},
                        zorder=4)
        rec = assoc[stats_key]
        ax.text(0.03, 0.97,
                f"all 20 arms: Spearman r_s={rec['spearman_rho']:.2f}, p={rec['spearman_p']:.3f}",
                transform=ax.transAxes, ha="left", va="top", fontsize=6.5)
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
    save(fig, str(OUT / "figS3_dti_all20"))
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


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    use_paper_style()
    evidence = load_json(PACKAGE / "results/evidence_summary.json")
    fig1(evidence)
    fig2(evidence)
    fig3()
    fig4(evidence)
    fig5(evidence)
    fig_s1(evidence)
    fig_s2(evidence)
    fig_s3(evidence)
    print(f"Wrote manuscript figures to {OUT}")


if __name__ == "__main__":
    main()
