#!/usr/bin/env python3
"""Generate the five main figures for the estimand-aware JoC manuscript."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


PACKAGE = Path(__file__).resolve().parents[1]
LEDGER = PACKAGE / "results/manuscript_evidence.json"
DESCRIPTOR_CONTROLS = (
    PACKAGE / "results/descriptor_rank_matched_controls/summary.json"
)
FIGURES = PACKAGE / "figures"

BLUE = "#2166AC"
ORANGE = "#D6604D"
GREEN = "#1B9E77"
PURPLE = "#7570B3"
GOLD = "#B8860B"
GREY = "#6B7280"
LIGHT_GREY = "#D1D5DB"
BLACK = "#111827"
RELEASE_TIMESTAMP = datetime(2026, 8, 10, tzinfo=timezone.utc)

def configure() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.5,
            "axes.labelsize": 8,
            "axes.titlesize": 8.5,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 6.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.7,
            "lines.linewidth": 1.4,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def load() -> dict:
    return json.loads(LEDGER.read_text())


def load_descriptor_controls() -> dict:
    """Load the fixed-20, rank-matched ligand-feature control artifact."""
    return json.loads(DESCRIPTOR_CONTROLS.read_text())


def panel_label(
    ax: plt.Axes, label: str, *, x: float = -0.13, y: float = 1.07
) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        va="top",
    )


def save(fig: plt.Figure, stem: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        FIGURES / f"{stem}.pdf",
        bbox_inches="tight",
        metadata={
            "Title": stem,
            "Author": "Adeliya Leleytner; Victor Safronov; Maxim Fedorov",
            "Creator": "make_manuscript_figures.py",
            "CreationDate": RELEASE_TIMESTAMP,
            "ModDate": RELEASE_TIMESTAMP,
        },
    )
    fig.savefig(FIGURES / f"{stem}.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def figure1(ledger: dict) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.25, 5.8), constrained_layout=True)
    datasets = ("Docking-44", "DOCKSTRING-58")
    for ax, dataset, label in zip(axes[0], datasets, ("a", "b"), strict=True):
        block = ledger["spectral"][dataset]["correlation"]
        for surface, color in (("raw", BLUE), ("residual", ORANGE)):
            eigenvalues = np.asarray(block[surface]["eigenvalues"], dtype=float)
            # Row centering introduces one algebraically zero mode.  Omitting only
            # floating-point zeros keeps the log-spectrum readable without
            # changing any spectral summary.
            eigenvalues = eigenvalues[eigenvalues > 1e-12]
            ax.plot(
                np.arange(1, len(eigenvalues) + 1),
                eigenvalues / eigenvalues.sum(),
                marker="o" if len(eigenvalues) < 50 else None,
                markersize=2,
                color=color,
                label="Uncentred" if surface == "raw" else "Within-ligand centred",
            )
        ax.set_yscale("log")
        ax.set_xlabel("Principal-component index")
        ax.set_ylabel("Variance fraction")
        ax.set_title(f"{dataset} spectrum")
        ax.grid(axis="y", alpha=0.2)
        ax.legend(frameon=False)
        panel_label(ax, label)

    ax = axes[1, 0]
    x = np.arange(2)
    width = 0.24
    for offset, key, color, legend in (
        (-width, "raw", BLUE, "Uncentred"),
        (0.0, "residual", ORANGE, "Centred"),
    ):
        values = [
            ledger["spectral"][dataset]["correlation"][key]["participation_ratio"]
            for dataset in datasets
        ]
        ax.bar(x + offset, values, width, color=color, label=legend)
        for xpos, value in zip(x + offset, values, strict=True):
            ax.text(xpos, value + 1.1, f"{value:.1f}", ha="center", va="bottom")
    null_values = [
        ledger["spectral"][dataset]["row_norm_preserving_null"]["null_residual"][
            "median"
        ]
        for dataset in datasets
    ]
    ax.bar(x + width, null_values, width, color=LIGHT_GREY, edgecolor=GREY, label="Row-norm null")
    for xpos, value in zip(x + width, null_values, strict=True):
        ax.text(xpos, value + 1.1, f"{value:.1f}", ha="center", va="bottom")
    ax.set_xticks(x, ["Docking-44", "DOCKSTRING-58"])
    ax.set_ylabel("PR effective dimension")
    ax.set_ylim(0, 66)
    ax.legend(
        frameon=False,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        fontsize=6.1,
    )
    panel_label(ax, "c")

    ax = axes[1, 1]
    for dataset, marker, color in (
        ("Docking-44", "o", BLUE),
        ("DOCKSTRING-58", "s", PURPLE),
    ):
        block = ledger["spectral"][dataset]["correlation"]
        values = [
            block["raw"]["mean_squared_offdiagonal_correlation"],
            block["residual"]["mean_squared_offdiagonal_correlation"],
        ]
        ax.plot([0, 1], values, marker=marker, color=color, label=dataset)
        for xpos, value in zip((0, 1), values, strict=True):
            ax.text(xpos, value + 0.018, f"{value:.3f}", ha="center", color=color)
    ax.set_xticks([0, 1], ["Uncentred", "Centred"])
    ax.set_ylabel(r"Mean squared target correlation, $\overline{r^2}$")
    ax.set_ylim(0, 0.61)
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False, loc="upper right")
    ax.text(
        0.50,
        0.87,
        r"$r_{\mathrm{PR}}=P/[1+(P-1)\overline{r^2}]$",
        transform=ax.transAxes,
        va="top",
        ha="center",
        fontsize=7.3,
    )
    panel_label(ax, "d")
    save(fig, "fig1_spectral_geometry")


def figure2(ledger: dict) -> None:
    panels = ["DAVIS", "PKIS2", "PKIS1", "KiRHub"]
    colors = [BLUE, ORANGE, GREEN, PURPLE]
    fixed_block = ledger["fixed20_estimand_decomposition"]
    fixed = fixed_block["panels"]
    fixed_matched = fixed_block["transformation_matched_independent_column_null"]["panels"]
    # Three panels.  The conditional-increment heat map that used to sit here as panel b
    # is dropped: Table 1 already reports all four increments for all four panels
    # numerically, so the heat map was a second rendering of the same numbers.
    fig, axes = plt.subplots(1, 3, figsize=(7.25, 3.05), constrained_layout=True)

    ax = axes[0]
    # Short docking/experiment stage names; the caption spells the pairs out in full.
    stages = ["Raw /\nraw", "Raw /\ncentred", "Centred /\ncentred"]
    for panel, color in zip(panels, colors, strict=True):
        block = fixed[panel]
        values = [
            block["docking_raw__experimental_raw"],
            block["docking_raw__experimental_centered"],
            block["docking_centered__experimental_centered"],
        ]
        ax.plot(range(3), values, marker="o", color=color, label=panel)
    ax.axhline(0, color=LIGHT_GREY, lw=0.8)
    ax.set_xticks(range(3), stages, fontsize=6.6)
    ax.set_xlabel("Docking / experiment transform", fontsize=7)
    ax.set_ylabel("Target-geometry concordance\n" + r"(Spearman $\rho$)")
    ax.set_ylim(-0.28, 0.39)
    ax.legend(frameon=False, ncol=2, fontsize=6.2)
    ax.grid(axis="y", alpha=0.2)
    panel_label(ax, "a")

    ax = axes[1]
    for panel, color in zip(panels, colors, strict=True):
        block = fixed[panel]
        raw = block["docking_raw__experimental_centered"]
        centred = block["docking_centered__experimental_centered"]
        p_value = block["paired_qap_for_docking_increment"]["one_sided_p_positive_delta"]
        matched_p = fixed_matched[panel][
            "two_way_centered_experimental_target_geometry"
        ]["p_observed_gain_at_least_as_large"]
        ax.plot(
            [0, 1],
            [raw, centred],
            color=color,
            marker="o",
            alpha=0.9,
            label=f"{panel}: $p_L$={p_value:.3f}; $p_M$={matched_p:.3f}",
        )
    ax.set_xticks([0, 1], ["Raw docking", "Centred docking"])
    ax.set_ylabel("Concordance with fixed\ncentred experiment")
    ax.set_xlim(-0.13, 1.13)
    ax.set_ylim(0.09, 0.35)
    ax.grid(axis="y", alpha=0.2)
    ax.legend(
        frameon=False,
        fontsize=5.7,
        loc="lower center",
        handlelength=1.3,
        handletextpad=0.5,
        labelspacing=0.32,
        borderpad=0.1,
    )
    panel_label(ax, "b")

    ax = axes[2]
    cross = ledger["experimental_cross_panel"]
    values = [cross["raw_mean_pairwise_spearman"], cross["centered_mean_pairwise_spearman"]]
    bars = ax.bar([0, 1], values, color=[BLUE, ORANGE], width=0.58)
    for bar, value in zip(bars, values, strict=True):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.018, f"{value:.3f}", ha="center")
    ax.set_xticks([0, 1], ["Raw", "Centred"])
    ax.set_ylim(0, 0.9)
    ax.set_ylabel("Mean pairwise agreement among\nfour experimental panels")
    matched = cross["transformation_matched_independent_column_null"]
    p_label_gain = cross["paired_global_network_qap"][
        "paired_global_network_qap_p_positive_gain"
    ]
    p_matched_gain = matched["p_observed_gain_at_least_as_large"]
    p_matched_absolute = matched["p_observed_centered_at_least_as_large"]
    ax.text(
        0.5,
        0.26,
        "Centred-minus-raw gain:\n"
        f"target-label p={p_label_gain:.3f}\n"
        f"matched p={p_matched_gain:.3f}\n"
        f"absolute centred,\nmatched p={p_matched_absolute:.4f}",
        ha="center",
        va="center",
        fontsize=6.0,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.95, pad=2.0),
    )
    ax.text(
        0.5,
        0.89,
        "Strong cross-panel agreement;\nits increment is not isolated",
        ha="center",
        va="top",
        fontsize=6.2,
    )
    panel_label(ax, "c")
    save(fig, "fig2_estimand_decomposition")


def figure3_legacy_structural_boundary(ledger: dict) -> None:
    # Three panels.  The continuous target-pair geometry heat map that used to sit here as
    # panel a is dropped: Supplementary Table S13 already reports the same per-panel
    # sequence, pocket and Vina concordances numerically.
    fig, axes = plt.subplots(1, 3, figsize=(7.25, 3.05), constrained_layout=True)

    ax = axes[0]
    baselines = ledger["structural_controls"]["locked_pair_baselines"]
    order = [
        "receptor_domain_sequence_identity",
        "klifs_pocket_identity",
        "raw_Vina_target_geometry",
        "centered_Vina_target_geometry",
    ]
    labels = ["Sequence", "KLIFS pocket", "Raw Vina", "Centred Vina"]
    x = np.arange(len(order))
    auc = [baselines[key]["roc_auc"] for key in order]
    ap = [baselines[key]["average_precision"] for key in order]
    ax.bar(x - 0.18, auc, 0.36, color=BLUE, label="AUROC")
    ax.bar(x + 0.18, ap, 0.36, color=ORANGE, label="Average precision")
    ax.axhline(15 / 190, color=GREY, ls="--", lw=0.8, label="AP prevalence")
    ax.set_xticks(x, labels, rotation=25, ha="right")
    ax.set_ylim(0, 0.96)
    ax.set_ylabel("Locked 15-of-190 pair retrieval")
    ax.legend(
        frameon=False,
        fontsize=6.0,
        loc="upper left",
        handlelength=1.4,
        handletextpad=0.5,
        labelspacing=0.3,
        borderpad=0.1,
    )
    panel_label(ax, "a")

    ax = axes[1]
    structural = ledger["structural_controls"]
    labels = ["Three-panel mean", "KiRHub"]
    rhos = []
    p_unrestricted = []
    p_group = []
    for key in ("three_panel_mean", "KiRHub"):
        unrestricted = structural["unrestricted_partial_qap"][key]
        restricted = structural["within_KLIFS_group_partial_qap"][key]
        rhos.append(unrestricted["partial_spearman"])
        p_unrestricted.append(unrestricted["target_label_qap_p_positive"])
        p_group.append(restricted["restricted_qap_p_positive"])
    bars = ax.bar(range(2), rhos, color=[GREEN, PURPLE], width=0.58)
    for index, bar in enumerate(bars):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.012,
            f"unrestricted\np={p_unrestricted[index]:.3f}\nwithin-group\np={p_group[index]:.3f}",
            ha="center",
            va="bottom",
            fontsize=6.2,
        )
    ax.set_xticks(range(2), labels, fontsize=6.6)
    ax.set_ylim(0, 0.44)
    ax.set_ylabel("Partial Spearman rho\n(sequence + KLIFS pocket adjusted)")
    panel_label(ax, "b")

    ax = axes[2]
    spd = ledger["broad_family_censored_sensitivity"]["key_results"]
    endpoints = ["Released bound", "10 uM binary", "30 uM binary"]
    raw = [
        spd["primary_floor_at_bound"]["raw_spearman"],
        spd["censor_aware_binary"]["10_uM"]["raw_spearman"],
        spd["censor_aware_binary"]["30_uM"]["raw_spearman"],
    ]
    centred = [
        spd["primary_floor_at_bound"]["residual_spearman"],
        spd["censor_aware_binary"]["10_uM"]["residual_spearman"],
        spd["censor_aware_binary"]["30_uM"]["residual_spearman"],
    ]
    p_gain = {row["endpoint"]: row["p_positive"] for row in spd["qap"] if row["metric"] == "residual_minus_raw" and row["permutation_scheme"] == "unrestricted"}
    x = np.arange(3)
    ax.bar(x - 0.17, raw, 0.34, color=BLUE, label="Raw")
    ax.bar(x + 0.17, centred, 0.34, color=ORANGE, label="Centred")
    for i, key in enumerate(("floor_at_bound", "censor_aware_binary_10uM", "censor_aware_binary_30uM")):
        ax.text(i, max(raw[i], centred[i]) + 0.022, f"gain p={p_gain[key]:.3f}", ha="center", fontsize=6.0)
    ax.set_xticks(x, endpoints, rotation=20, ha="right", fontsize=6.6)
    ax.set_ylim(0, 0.62)
    ax.set_ylabel("Mixed-target concordance")
    ax.legend(
        frameon=False,
        loc="upper right",
        fontsize=6.2,
        handlelength=1.2,
        handletextpad=0.5,
        labelspacing=0.3,
        borderpad=0.1,
    )
    ax.text(
        0.02,
        0.97,
        "91.6% right-censored;\nfamily-preserving\ngain unresolved",
        transform=ax.transAxes,
        va="top",
        fontsize=6.0,
    )
    panel_label(ax, "c")
    save(fig, "fig3_structural_boundary")


def figure2_support_recovery(ledger: dict) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.25, 5.9), constrained_layout=True)
    domain = ledger["exploratory_chemical_domain_structure"][
        "low_high_molecular_weight_domain_instability"
    ]["datasets"]
    ax = axes[0, 0]
    datasets = ["Docking-44", "DOCKSTRING-58"]
    x = np.arange(2)
    raw = [domain[name]["raw_surface_geometry"] for name in datasets]
    residual = [domain[name]["row_centered_residual_geometry"] for name in datasets]
    adjusted = [
        domain[name]["strict_descriptor_adjusted_residual_sensitivity"]
        for name in datasets
    ]
    within_key = "restriction_matched_within_band_reproducibility"
    group_key = "mw_stratified_chemical_group_disjoint"
    within_low = [row[within_key]["low_mw"][group_key] for row in residual]
    within_high = [row[within_key]["high_mw"][group_key] for row in residual]

    def errorbar_rows(
        offset: float,
        values: list[float],
        intervals: list[list[float]],
        *,
        color: str,
        marker: str,
        label: str,
    ) -> None:
        yerr = np.asarray(
            [
                [value - bounds[0] for value, bounds in zip(values, intervals, strict=True)],
                [bounds[1] - value for value, bounds in zip(values, intervals, strict=True)],
            ]
        )
        ax.errorbar(
            x + offset,
            values,
            yerr=yerr,
            fmt=marker,
            color=color,
            capsize=3,
            label=label,
            zorder=3,
        )

    errorbar_rows(
        -0.30,
        [row["spearman"] for row in raw],
        [row["chemical_group_bootstrap"]["central_95_percent_interval"] for row in raw],
        color=GOLD,
        marker="v",
        label="Raw, low vs high MW",
    )
    errorbar_rows(
        -0.15,
        [row["spearman"] for row in residual],
        [row["chemical_group_bootstrap"]["central_95_percent_interval"] for row in residual],
        color=ORANGE,
        marker="D",
        label="Residual, low vs high MW",
    )
    errorbar_rows(
        0.0,
        [row["geometry_spearman"] for row in adjusted],
        [row["chemical_group_bootstrap_central_95_percent_interval"] for row in adjusted],
        color=PURPLE,
        marker="^",
        label="After held-out descriptor removal",
    )
    errorbar_rows(
        0.15,
        [row["geometry_spearman_mean"] for row in within_low],
        [row["central_95_percent_repeated_split_sensitivity_range"] for row in within_low],
        color=BLUE,
        marker="s",
        label="Within-low-MW group-disjoint",
    )
    errorbar_rows(
        0.30,
        [row["geometry_spearman_mean"] for row in within_high],
        [row["central_95_percent_repeated_split_sensitivity_range"] for row in within_high],
        color=GREEN,
        marker="o",
        label="Within-high-MW group-disjoint",
    )
    ax.set_xticks(x, datasets)
    ax.set_ylim(0.1, 1.04)
    ax.set_ylabel("Between-support map agreement")
    ax.legend(frameon=False, fontsize=5.8, loc="lower right")
    panel_label(ax, "a")

    ax = axes[0, 1]
    chemistry = ledger["chemical_context"]["panels"]
    panels = ["DAVIS", "PKIS2", "PKIS1"]
    x = np.arange(3)
    observed = [chemistry[p]["observed"]["geometry_spearman"] for p in panels]
    null_mean = [chemistry[p]["random_disjoint_null"]["mean"] for p in panels]
    low = [chemistry[p]["random_disjoint_null"]["q025"] for p in panels]
    high = [chemistry[p]["random_disjoint_null"]["q975"] for p in panels]
    ax.errorbar(
        x - 0.12,
        null_mean,
        yerr=[np.asarray(null_mean) - low, np.asarray(high) - null_mean],
        fmt="o",
        color=GREY,
        capsize=3,
        label="Random disjoint supports",
    )
    ax.scatter(
        x + 0.12,
        observed,
        color=ORANGE,
        marker="D",
        label="Low- vs high-MW quartiles",
        zorder=3,
    )
    for i, panel in enumerate(panels):
        probability = chemistry[panel]["random_disjoint_null"][
            "lower_tail_monte_carlo_p"
        ]
        ax.text(
            i + 0.12,
            observed[i] - 0.06,
            f"p={probability:.3f}",
            ha="center",
            fontsize=6.6,
        )
    ax.set_xticks(x, panels)
    ax.set_ylim(0.15, 0.98)
    ax.set_ylabel("Between-support map agreement")
    ax.legend(frameon=False, fontsize=6.2)
    panel_label(ax, "b")

    ax = axes[1, 0]
    recovery = ledger["probe_panel_recovery"]
    for dataset, color, marker in (("DOCKSTRING-58", PURPLE, "s"), ("Docking-44", BLUE, "o")):
        sizes = np.asarray([50, 100, 200, 500, 1000, 2000, 5000])
        means = np.asarray([recovery[dataset][str(size)]["geometry_spearman_mean"] for size in sizes])
        low = np.asarray([recovery[dataset][str(size)]["geometry_spearman_q025"] for size in sizes])
        high = np.asarray([recovery[dataset][str(size)]["geometry_spearman_q975"] for size in sizes])
        ax.plot(sizes, means, color=color, marker=marker, label=dataset)
        ax.fill_between(sizes, low, high, color=color, alpha=0.15)
    ax.set_xscale("log")
    ax.set_xticks([50, 100, 200, 500, 1000, 2000, 5000], ["50", "100", "200", "500", "1k", "2k", "5k"])
    ax.set_ylim(0.62, 1.01)
    ax.set_xlabel("Random same-distribution probe ligands")
    ax.set_ylabel("Agreement with full-library target map")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2)
    panel_label(ax, "c")

    ax = axes[1, 1]
    x = np.arange(2)
    width = 0.32
    for offset, dataset, color in ((-width / 2, "DOCKSTRING-58", PURPLE), (width / 2, "Docking-44", BLUE)):
        values = [
            recovery[dataset]["chemical_group_holdout_200"]["mean_geometry_spearman_to_held_out_scaffolds"],
            recovery[dataset]["chemical_group_holdout_500"]["mean_geometry_spearman_to_held_out_scaffolds"],
        ]
        ax.bar(x + offset, values, width, color=color, label=dataset)
        for xpos, value in zip(x + offset, values, strict=True):
            ax.text(xpos, value + 0.006, f"{value:.3f}", ha="center", fontsize=6.5)
    ax.set_xticks(x, ["200", "500"])
    ax.set_ylim(0.82, 1.0)
    ax.set_xlabel("Probe ligands")
    ax.set_ylabel("Chemical-group-held-out agreement")
    ax.legend(frameon=False)
    panel_label(ax, "d")
    save(fig, "fig4_support_recovery")


def figure3(ledger: dict) -> None:
    """Rank-matched feature controls on the fixed, de-leaked 20-target support."""
    del ledger  # This figure is traced to the dedicated reviewer-control artifact.
    controls = load_descriptor_controls()
    nuisance = controls.get("physicochemical_row_permutation_nuisance_ensemble")
    if nuisance is None:
        raise KeyError(
            "descriptor controls lack the 20-seed row-permutation nuisance ensemble"
        )

    metrics = controls["predictive_metrics"]
    geometry = controls["panel_geometry_agreement"]
    morgan = controls["morgan_count_plus_size_rp_7_seed_ensemble"]
    morgan_members = morgan["members_detail"]
    nuisance_members = nuisance["members_detail"]
    phys_key = "physicochemical_7"
    morgan_key = "morgan_count_plus_size_rp_7_locked_seed_20260910"
    hash_key = "stable_hash_nuisance_7"

    fig, axes = plt.subplots(2, 2, figsize=(7.25, 5.85), constrained_layout=True)
    seed_jitter = np.linspace(-0.16, 0.16, len(morgan_members))
    nuisance_jitter = np.linspace(-0.16, 0.16, len(nuisance_members))
    category_x = np.arange(4, dtype=float)
    category_labels = [
        "Physchem\n7-D",
        "Count-Morgan\n" + r"20 $\times$ 7-D",
        "Stable hash\n7-D",
        "Permuted physchem\n" + r"20 $\times$ 7-D",
    ]

    def seed_panel(
        ax: plt.Axes,
        metric_key: str,
        ylabel: str,
        *,
        multiplier: float = 1.0,
    ) -> None:
        phys_value = multiplier * metrics[phys_key][metric_key]
        locked_value = multiplier * metrics[morgan_key][metric_key]
        hash_value = multiplier * metrics[hash_key][metric_key]
        morgan_values = multiplier * np.asarray(
            [row["predictive_metrics"][metric_key] for row in morgan_members],
            dtype=float,
        )
        nuisance_values = multiplier * np.asarray(
            [row[metric_key] for row in nuisance_members], dtype=float
        )

        ax.axhline(0, color=LIGHT_GREY, lw=0.8, zorder=0)
        ax.scatter(category_x[0], phys_value, color=BLUE, marker="o", s=39, zorder=4)
        ax.scatter(
            category_x[1] + seed_jitter,
            morgan_values,
            color=ORANGE,
            marker="o",
            s=13,
            alpha=0.62,
            edgecolor="none",
            zorder=2,
        )
        ax.vlines(
            category_x[1],
            float(morgan_values.min()),
            float(morgan_values.max()),
            color=ORANGE,
            lw=1.0,
            zorder=1,
        )
        ax.scatter(
            category_x[1],
            locked_value,
            color=BLACK,
            marker="D",
            s=30,
            zorder=5,
            label="Locked Morgan seed",
        )
        ax.scatter(
            category_x[2],
            hash_value,
            facecolor="white",
            edgecolor=PURPLE,
            marker="X",
            linewidth=1.1,
            s=43,
            zorder=4,
        )
        ax.scatter(
            category_x[3] + nuisance_jitter,
            nuisance_values,
            facecolor="none",
            edgecolor=GREY,
            marker="s",
            linewidth=0.7,
            s=15,
            alpha=0.75,
            zorder=2,
        )
        ax.vlines(
            category_x[3],
            float(nuisance_values.min()),
            float(nuisance_values.max()),
            color=GREY,
            lw=1.0,
            zorder=1,
        )
        ax.set_xticks(category_x, category_labels, fontsize=6.2)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.18)

    ax = axes[0, 0]
    seed_panel(
        ax,
        "mean_out_of_fold_target_r2",
        r"Mean out-of-fold target $R^2$",
    )
    ax.set_title("Held-out predictive performance", fontsize=8.2, pad=3)
    ax.legend(frameon=False, fontsize=6.0, loc="upper right")
    panel_label(ax, "a", x=-0.10, y=1.02)

    ax = axes[0, 1]
    seed_panel(
        ax,
        "relative_reduction_in_mean_squared_offdiagonal_target_correlation",
        "Operational reduction in mean\nsquared target correlation (%)",
        multiplier=100.0,
    )
    ax.set_title(
        "Operational correlation reduction",
        fontsize=8.2,
        pad=3,
    )
    ax.text(
        0.98,
        0.97,
        "non-additive; not an explained share",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=5.7,
        color=GREY,
    )
    panel_label(ax, "b", x=-0.10, y=1.02)

    ax = axes[1, 0]
    panels = ["DAVIS", "PKIS2", "PKIS1", "KiRHub"]
    panel_style = {
        "DAVIS": (BLUE, "o"),
        "PKIS2": (ORANGE, "s"),
        "PKIS1": (GREEN, "^"),
        "KiRHub": (PURPLE, "D"),
    }
    basis_records = [
        ("Physicochemical", phys_key, 38),
        ("Locked count-Morgan", morgan_key, 34),
        ("Stable hash", hash_key, 42),
    ]
    for panel in panels:
        color, marker = panel_style[panel]
        for _, basis_key, size in basis_records:
            ax.scatter(
                metrics[basis_key]["mean_out_of_fold_target_r2"],
                geometry[basis_key][panel],
                color=color,
                marker=marker,
                s=size,
                edgecolor="white",
                linewidth=0.45,
                zorder=4,
            )
        ax.scatter(
            [row["mean_out_of_fold_target_r2"] for row in nuisance_members],
            [row["panel_geometry_agreement"][panel] for row in nuisance_members],
            facecolor="none",
            edgecolor=color,
            marker=marker,
            linewidth=0.45,
            s=11,
            alpha=0.35,
            zorder=1,
        )
        ax.scatter([], [], color=color, marker=marker, label=panel, s=27)

    for name, basis_key, _ in basis_records:
        amplitude = metrics[basis_key]["mean_out_of_fold_target_r2"]
        mean_agreement = float(
            np.mean([geometry[basis_key][panel] for panel in panels])
        )
        offsets = {
            "Physicochemical": (4, -13),
            "Locked count-Morgan": (-47, 9),
            "Stable hash": (5, 7),
        }
        ax.annotate(
            name,
            (amplitude, mean_agreement),
            xytext=offsets[name],
            textcoords="offset points",
            fontsize=5.8,
            color=BLACK,
            arrowprops=dict(arrowstyle="-", color=GREY, lw=0.5),
        )
    ax.axvline(0, color=LIGHT_GREY, lw=0.8)
    ax.set_xlabel(r"Mean out-of-fold target $R^2$")
    ax.set_ylabel("Component-map agreement with\n" + r"experiment (Spearman $\rho$)")
    ax.set_title("Map agreement is not feature-specific evidence", fontsize=7.5, pad=3)
    ax.legend(frameon=False, ncol=2, fontsize=5.9, loc="lower right")
    ax.grid(alpha=0.16)
    panel_label(ax, "c", x=-0.10, y=1.02)

    ax = axes[1, 1]
    ax.set_axis_off()
    box_style = dict(
        boxstyle="round,pad=0.35",
        facecolor="#F3F4F6",
        edgecolor=GREY,
        lw=0.8,
    )
    nodes = [
        (0.10, 0.63, "Ligand basis\n" + r"$Z: n\times7$"),
        (0.42, 0.63, "Target-wise fit\n" + r"$B: 7\times20$"),
        (0.75, 0.63, "Component\n$C=ZB$"),
        (0.75, 0.28, "Target map\n$corr(C)$"),
    ]
    for xpos, ypos, label in nodes:
        ax.text(
            xpos,
            ypos,
            label,
            transform=ax.transAxes,
            ha="center",
            va="center",
            bbox=box_style,
            fontsize=6.6,
        )
    arrow_style = dict(
        arrowstyle="->", color=BLACK, lw=0.9, shrinkA=3, shrinkB=3
    )
    ax.annotate(
        "",
        xy=(0.28, 0.63),
        xytext=(0.20, 0.63),
        xycoords=ax.transAxes,
        arrowprops=arrow_style,
    )
    ax.annotate(
        "",
        xy=(0.61, 0.63),
        xytext=(0.53, 0.63),
        xycoords=ax.transAxes,
        arrowprops=arrow_style,
    )
    ax.annotate(
        "",
        xy=(0.75, 0.39),
        xytext=(0.75, 0.52),
        xycoords=ax.transAxes,
        arrowprops=arrow_style,
    )
    ax.text(
        0.36,
        0.15,
        r"Near-zero predictive performance does not force $corr(C)=0$." "\n"
        "Target-specific fits can orient even nuisance bases.",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=6.7,
        color=BLACK,
    )
    ax.set_title("Why orientation alone is non-identifying", fontsize=7.5, pad=3)
    panel_label(ax, "d", x=-0.10, y=1.02)
    save(fig, "fig4_feature_controls")


def figure4(ledger: dict) -> None:
    """External estimand, structural-boundary and censoring controls."""
    fig, axes = plt.subplots(2, 2, figsize=(7.25, 5.9), constrained_layout=True)

    ax = axes[0, 0]
    factorial = ledger["fixed20_estimand_decomposition"]["panels"]
    panels = ["DAVIS", "PKIS2", "PKIS1", "KiRHub"]
    styles = {
        "DAVIS": (BLUE, "o", "-"),
        "PKIS2": (ORANGE, "s", "--"),
        "PKIS1": (GREEN, "^", "-."),
        "KiRHub": (PURPLE, "D", ":"),
    }
    xpos = np.asarray([0.0, 1.0, 2.35, 3.35])
    keys = [
        "docking_raw__experimental_raw",
        "docking_centered__experimental_raw",
        "docking_raw__experimental_centered",
        "docking_centered__experimental_centered",
    ]
    ax.axvspan(-0.30, 1.30, color="#F3F4F6", zorder=0)
    ax.axvspan(2.05, 3.65, color="#F8FAFC", zorder=0)
    for panel in panels:
        color, marker, linestyle = styles[panel]
        values = [factorial[panel][key] for key in keys]
        ax.plot(
            xpos[:2],
            values[:2],
            color=color,
            marker=marker,
            ls=linestyle,
            label=panel,
        )
        ax.plot(
            xpos[2:],
            values[2:],
            color=color,
            marker=marker,
            ls=linestyle,
        )
    ax.axhline(0, color=LIGHT_GREY, lw=0.8)
    ax.set_xticks(
        xpos,
        ["Raw\ndocking", "Centred\ndocking", "Raw\ndocking", "Centred\ndocking"],
        fontsize=6.2,
    )
    ax.text(0.5, 0.985, "Raw experimental\nendpoint", transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=6.3)
    ax.text(2.85, 0.985, "Centred experimental\nendpoint", transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=6.3)
    ax.set_ylabel("Map agreement with experiment\n" + r"(Spearman $\rho$)")
    ax.set_ylim(-0.26, 0.38)
    ax.legend(frameon=False, ncol=2, fontsize=5.9, loc="lower right")
    ax.grid(axis="y", alpha=0.18)
    panel_label(ax, "a", x=-0.10, y=1.02)

    ax = axes[0, 1]
    baselines = ledger["structural_controls"]["locked_pair_baselines"]
    baseline_keys = [
        "receptor_domain_sequence_identity",
        "klifs_pocket_identity",
        "raw_Vina_target_geometry",
        "centered_Vina_target_geometry",
    ]
    baseline_labels = ["Sequence", "KLIFS pocket", "Raw Vina", "Centred Vina"]
    x = np.arange(len(baseline_keys))
    auc = [baselines[key]["roc_auc"] for key in baseline_keys]
    average_precision = [baselines[key]["average_precision"] for key in baseline_keys]
    ax.bar(x - 0.18, auc, 0.36, color=BLUE, label="AUROC")
    ax.bar(x + 0.18, average_precision, 0.36, color=ORANGE, label="Average precision")
    ax.axhline(15 / 190, color=GREY, ls="--", lw=0.8, label="AP prevalence")
    ax.set_xticks(x, baseline_labels, rotation=20, ha="right", fontsize=6.3)
    ax.set_ylim(0, 0.88)
    ax.set_ylabel("Locked co-selective-pair retrieval")
    ax.set_title("15 positive pairs among 190 fixed target pairs", fontsize=6.8, pad=3)
    ax.legend(
        frameon=False,
        fontsize=5.9,
        loc="upper left",
        handlelength=1.3,
        handletextpad=0.4,
        labelspacing=0.25,
    )
    panel_label(ax, "b", x=-0.10, y=1.02)

    ax = axes[1, 0]
    structural = ledger["structural_controls"]
    endpoint_keys = ["three_panel_mean", "KiRHub"]
    endpoint_labels = ["Three-panel mean", "KiRHub"]
    y = np.arange(2, dtype=float)
    for index, endpoint in enumerate(endpoint_keys):
        unrestricted = structural["unrestricted_partial_qap"][endpoint]
        restricted = structural["within_KLIFS_group_partial_qap"][endpoint]
        ax.hlines(
            y[index] - 0.11,
            unrestricted["null_q025"],
            unrestricted["null_q975"],
            color=BLUE,
            lw=1.5,
        )
        ax.scatter(
            [unrestricted["null_q025"], unrestricted["null_q975"]],
            [y[index] - 0.11] * 2,
            color=BLUE,
            marker="|",
            s=52,
            zorder=3,
        )
        ax.hlines(
            y[index] + 0.11,
            restricted["null_q05"],
            restricted["null_q95"],
            color=ORANGE,
            lw=2.1,
        )
        ax.scatter(
            restricted["null_median"],
            y[index] + 0.11,
            color=ORANGE,
            marker="s",
            s=25,
            zorder=4,
        )
        ax.scatter(
            restricted["partial_spearman"],
            y[index],
            color=BLACK,
            marker="D",
            s=31,
            zorder=5,
        )
        ax.annotate(
            f"critical {restricted['empirical_one_sided_alpha_0_05_critical_partial_spearman']:.3f}",
            (restricted["null_q95"], y[index] + 0.11),
            xytext=(2, 4),
            textcoords="offset points",
            fontsize=5.5,
            color=ORANGE,
        )
    ax.axvline(0, color=LIGHT_GREY, lw=0.8)
    ax.set_yticks(y, endpoint_labels, fontsize=6.5)
    ax.invert_yaxis()
    ax.set_xlim(-0.14, 0.47)
    ax.set_xlabel(r"Sequence + KLIFS-adjusted partial Spearman $\rho$")
    ax.set_title("Target-label randomization resolution", fontsize=7.2, pad=3)
    ax.text(
        0.02,
        0.50,
        "Blue: unrestricted null, central 95%\n"
        "Orange: within-group null, 5--95%; square = median\n"
        "Black diamond: observed partial rho",
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=5.6,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.88, pad=1.5),
    )
    panel_label(ax, "c", x=-0.10, y=1.02)

    ax = axes[1, 1]
    davis = ledger["davis_fixed20_censoring_sensitivity"]
    filtered = davis["exclude_targets_over_90_percent_floor_sensitivity"]
    support_x = np.arange(2)
    series = [
        (
            "Experiment: continuous vs floor status",
            [
                davis["continuous_vs_above_floor_status_geometry_spearman"],
                filtered["continuous_vs_above_floor_status_geometry_spearman"],
            ],
            GREEN,
            "^",
            "-.",
        ),
        (
            "Centred Vina vs continuous",
            [
                davis["centered_docking_vs_continuous_davis"]["spearman"],
                filtered["centered_docking_vs_continuous_davis"]["spearman"],
            ],
            BLUE,
            "o",
            "-",
        ),
        (
            "Centred Vina vs floor status",
            [
                davis["centered_docking_vs_above_floor_status_davis"]["spearman"],
                filtered["centered_docking_vs_above_floor_status_davis"]["spearman"],
            ],
            ORANGE,
            "s",
            "--",
        ),
    ]
    for label, values, color, marker, linestyle in series:
        ax.plot(
            support_x,
            values,
            color=color,
            marker=marker,
            ls=linestyle,
            label=label,
        )
    ax.set_xticks(
        support_x,
        ["Fixed 20 targets", "16 targets\n(4 high-floor targets removed)"],
        fontsize=6.4,
    )
    ax.set_ylim(0.12, 0.94)
    ax.set_ylabel(r"Target-map agreement (Spearman $\rho$)")
    ax.set_title("DAVIS reporting-floor sensitivity", fontsize=7.2, pad=3)
    ax.legend(frameon=False, fontsize=5.6, loc="center right")
    ax.grid(axis="y", alpha=0.18)
    panel_label(ax, "d", x=-0.10, y=1.02)
    save(fig, "fig3_external_boundary")


def _conservative_interval(record: dict, level: str = "interval_95") -> tuple[float, float]:
    intervals = [
        record["scaffold_cluster_bootstrap"][level],
        record["butina_cluster_bootstrap"][level],
    ]
    return min(x[0] for x in intervals), max(x[1] for x in intervals)


def _dense_cluster_interval(record: dict) -> tuple[float, float]:
    intervals = [
        record["uncertainty"]["murcko_cluster_bootstrap"]["interval_95"],
        record["uncertainty"]["butina_cluster_bootstrap"]["interval_95"],
    ]
    return min(x[0] for x in intervals), max(x[1] for x in intervals)


def figure5(ledger: dict) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.25, 5.7), constrained_layout=True)
    broad = ledger["dockstring_chembl_ranking"]
    broad_benchmark = broad["benchmark"]
    broad_representations = broad_benchmark["representations"]
    sparse = ledger["ligand_wise_ranking_boundary"]

    # The primary display is the broad DOCKSTRING--ChEMBL benchmark.  Priors and
    # docking-score representations are shown together, but their roles remain explicit.
    ax = axes[0, 0]
    keys = [
        "docking_target_prior",
        "experimental_target_prior",
        "cohort_experimental_target_prior",
        "absolute_vina",
        "column_standardized",
        "two_way_residual",
    ]
    labels = [
        "Docking\nprior",
        "External\nprior",
        "Cohort\nprior",
        "Absolute\nVina",
        "Column\nz-score",
        "Scaled\nresidual",
    ]
    colors = [PURPLE, GOLD, GREEN, BLUE, GREY, ORANGE]
    values = [
        broad_representations[key]["mean_per_ligand_pairwise_accuracy"] for key in keys
    ]
    intervals = [
        _conservative_interval(broad_representations[key]) for key in keys
    ]
    yerr = np.asarray([[value - low for value, (low, high) in zip(values, intervals, strict=True)], [high - value for value, (low, high) in zip(values, intervals, strict=True)]])
    ax.errorbar(range(len(keys)), values, yerr=yerr, fmt="none", ecolor=BLACK, capsize=3, lw=1)
    ax.scatter(range(len(keys)), values, c=colors, s=35, zorder=3)
    ax.axhline(0.5, color=LIGHT_GREY, ls="--", lw=0.8)
    ax.set_xticks(range(len(keys)), labels, fontsize=6.4)
    ax.set_ylim(0.44, 0.64)
    ax.set_ylabel("Observed-pair concordance")
    evaluated = broad_representations["absolute_vina"]["evaluated_ligands"]
    ax.set_title(
        "Broad primary DOCKSTRING–ChEMBL\n"
        f"{evaluated:,} evaluated ligands; "
        f"{broad['support']['targets']} targets; "
        f"{broad['support']['non_tied_within_ligand_target_pairs']:,} pairs",
        fontsize=6.8,
        pad=2,
    )
    panel_label(ax, "a")
    ax.texts[-1].set_x(-0.17)

    # Paired residual-minus-absolute differences for the coverage primary and the
    # cleaner human-binding Ki/Kd arm. Chemical-support sampling intervals and
    # target-panel jackknife approximations are deliberately separate.
    ax = axes[0, 1]
    clean = broad["sensitivities"]["human_binding_Ki_Kd"]
    contrast_key = "two_way_residual_minus_absolute_vina"
    rows = [
        (
            "All endpoints",
            broad["headline_contrasts"][contrast_key],
            broad_benchmark["target_jackknife_paired_contrasts"][contrast_key],
        ),
        (
            "Human binding Ki/Kd",
            clean["contrasts"][contrast_key],
            clean["target_jackknife_paired_contrasts"][contrast_key],
        ),
    ]
    labels = [f"{label}\nResidual - absolute" for label, _, _ in rows]
    values = [record["plugin_mean_difference"] for _, record, _ in rows]
    sampling_intervals = [
        tuple(record["conservative_cluster_bootstrap_interval_95"])
        for _, record, _ in rows
    ]
    target_intervals = [
        tuple(target["jackknife_normal_95_interval"])
        for _, _, target in rows
    ]
    y = np.arange(len(rows))
    ax.hlines(
        y - 0.10,
        [low for low, _ in sampling_intervals],
        [high for _, high in sampling_intervals],
        color=BLUE,
        lw=2.1,
        label="Chemical-cluster 95% interval",
    )
    ax.hlines(
        y + 0.10,
        [low for low, _ in target_intervals],
        [high for _, high in target_intervals],
        color=ORANGE,
        lw=1.2,
        label="Target-jackknife 95% approximation",
    )
    ax.scatter(values, y, color=BLACK, s=28, zorder=3)
    ax.axvline(0, color=GREY, ls="--", lw=0.8)
    ax.set_yticks(y, labels, fontsize=6.2)
    ax.invert_yaxis()
    all_bounds = [value for interval in sampling_intervals + target_intervals for value in interval]
    span_low = min(all_bounds + values)
    span_high = max(all_bounds + values)
    padding = max(0.008, 0.08 * (span_high - span_low))
    ax.set_xlim(span_low - padding, span_high + padding)
    ax.set_xlabel("Paired accuracy difference")
    ax.set_title("Endpoint contract and target composition", fontsize=7, pad=2)
    ax.legend(frameon=False, loc="center left", fontsize=5.9)
    panel_label(ax, "b")
    ax.texts[-1].set_x(-0.17)

    # The earlier Docking-44 support is now an explicitly labelled sensitivity rather
    # than the primary display.  Both contrasts use identical interval semantics.
    ax = axes[1, 0]
    support_rows = []
    for contrast_name, contrast_label in (
        ("two_way_residual_minus_absolute_vina", "Residual - absolute"),
        ("two_way_residual_minus_column_standardized", "Residual - column-z"),
    ):
        for benchmark_label, source, color, marker in (
            ("Broad primary", broad_benchmark, BLUE, "o"),
            ("Sparse sensitivity", sparse, GREY, "s"),
        ):
            contrast = source["paired_comparisons"][contrast_name]
            low, high = _conservative_interval(contrast)
            support_rows.append(
                (
                    f"{contrast_label}\n{benchmark_label}",
                    contrast["plugin_mean_difference"],
                    low,
                    high,
                    color,
                    marker,
                )
            )
    for y, (label, value, low, high, color, marker) in enumerate(support_rows):
        ax.hlines(y, low, high, color=color, lw=1.6)
        ax.scatter(value, y, color=color, marker=marker, s=29, zorder=3)
    ax.axvline(0, color=GREY, ls="--", lw=0.8)
    ax.set_yticks(range(len(support_rows)), [row[0] for row in support_rows], fontsize=6.2)
    ax.invert_yaxis()
    ax.set_xlim(-0.075, 0.075)
    ax.set_xlabel("Paired accuracy difference (cluster 95% interval)")
    ax.set_title("Chemical-support sensitivity", fontsize=7, pad=2)
    ax.grid(axis="x", alpha=0.2)
    panel_label(ax, "c")
    ax.texts[-1].set_x(-0.17)

    # Dense external panels retain their role as complementary estimation checks.
    # Post-hoc equivalence-margin curves are kept in the Supplement, not in the main
    # figure.
    ax = axes[1, 1]
    dense = ledger["dense_ligand_wise_benchmarks"]
    ax.axvline(0, color=BLACK, ls="--", lw=0.8)
    for y, (panel, color, marker) in enumerate(
        (("DAVIS", BLUE, "o"), ("PKIS2", ORANGE, "s"))
    ):
        contrast = dense[panel]["paired_comparisons"][
            "two_way_residual_minus_absolute_vina"
        ]
        low_95, high_95 = _dense_cluster_interval(contrast)
        ax.hlines(y, low_95, high_95, color=color, lw=2.0)
        ax.scatter(
            contrast["plugin_mean_difference"], y, color=color, marker=marker, s=32, zorder=3
        )
    ax.set_yticks([0, 1], ["DAVIS", "PKIS2"])
    ax.invert_yaxis()
    ax.set_xlim(-0.045, 0.045)
    ax.set_xlabel("Residual - absolute concordance")
    ax.set_title("Dense external checks", fontsize=7, pad=2)
    panel_label(ax, "d")
    ax.texts[-1].set_x(-0.17)
    save(fig, "fig5_ranking_boundary")


def main() -> None:
    configure()
    ledger = load()
    figure1(ledger)
    figure2_support_recovery(ledger)
    figure3(ledger)
    figure4(ledger)
    figure5(ledger)


if __name__ == "__main__":
    main()
