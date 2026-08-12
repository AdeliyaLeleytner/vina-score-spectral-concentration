#!/usr/bin/env python3
"""Render the four public-only main figures for the revised manuscript.

The figure producer has an intentionally narrow evidence boundary.  It reads
the two redistributed Vina matrices and the three public revision result
directories that are assembled into ``results/public_core_evidence.json``.
No historical manuscript ledger or source-restricted endpoint is imported.

Every panel is descriptive on a fixed target panel.  Resampling ranges are
labelled as distributions across ligand supports rather than target-population
confidence intervals, and no significance-star encoding is used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib as mpl  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.backends.backend_agg import FigureCanvasAgg  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from scipy.cluster import hierarchy  # noqa: E402
from scipy.spatial.distance import squareform  # noqa: E402


PACKAGE = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PACKAGE / "figures" / "public_core"

INPUT_PATHS = {
    "public_core": PACKAGE / "results" / "public_core_evidence.json",
    "docking44": PACKAGE / "data" / "frozen" / "df_final_v4.csv.gz",
    "dockstring58": PACKAGE / "data" / "frozen" / "dockstring-dataset.tsv.gz",
    "residual_nulls": (
        PACKAGE / "results" / "public_residual_null_audit" / "summary.json"
    ),
    "mw_sweep": (
        PACKAGE
        / "results"
        / "revision_map_decision_sensitivities"
        / "mw_threshold_sweep.csv"
    ),
    "mw_decile_pairs": (
        PACKAGE
        / "results"
        / "revision_map_decision_sensitivities"
        / "mw_decile_pair_distances.csv"
    ),
    "mw_decile_trends": (
        PACKAGE
        / "results"
        / "revision_map_decision_sensitivities"
        / "mw_decile_continuous_trends.csv"
    ),
    "chemical_domain_boundary": (
        PACKAGE
        / "results"
        / "public_chemical_domain_controls"
        / "causal_boundary_summary.csv"
    ),
    "chemical_domain_global_controls": (
        PACKAGE
        / "results"
        / "public_chemical_domain_controls"
        / "mw_matched_group_disjoint_summary.csv"
    ),
    "chemical_domain_within_controls": (
        PACKAGE
        / "results"
        / "public_chemical_domain_controls"
        / "within_band_reproducibility_summary.csv"
    ),
    "chemical_domain_support_seeds": (
        PACKAGE
        / "results"
        / "public_chemical_domain_controls"
        / "dockstring_support_seed_contrasts.csv"
    ),
    "descriptor_domain_extremes": (
        PACKAGE
        / "results"
        / "public_descriptor_domain_specificity"
        / "descriptor_extremes.csv"
    ),
    "descriptor_domain_random_controls": (
        PACKAGE
        / "results"
        / "public_descriptor_domain_specificity"
        / "descriptor_random_disjoint_summary.csv"
    ),
    "recovery_edges": (
        PACKAGE
        / "results"
        / "revision_map_decision_sensitivities"
        / "recovery_edge_summary.csv"
    ),
    "recovery_decisions": (
        PACKAGE
        / "results"
        / "revision_map_decision_sensitivities"
        / "recovery_decision_summary.csv"
    ),
    "panel_selector_metrics": (
        PACKAGE
        / "results"
        / "public_panel_selector_controls"
        / "panel_metrics.csv.gz"
    ),
    "panel_selector_direct": (
        PACKAGE
        / "results"
        / "public_panel_selector_robustness"
        / "direct_residual_summary.csv"
    ),
    "experimental_cross_panel": (
        PACKAGE
        / "results"
        / "experimental_map_reliability"
        / "cross_panel_geometry.csv"
    ),
    "experimental_reliability": (
        PACKAGE
        / "results"
        / "experimental_map_reliability"
        / "map_reliability.csv"
    ),
    "same_support_geometry": (
        PACKAGE
        / "results"
        / "experimental_map_reliability"
        / "same_support_geometry.csv"
    ),
    "same_support_bootstrap": (
        PACKAGE
        / "results"
        / "experimental_map_reliability"
        / "same_support_bootstrap.csv"
    ),
    "same_support_splits": (
        PACKAGE
        / "results"
        / "experimental_map_reliability"
        / "same_support_split_half.csv"
    ),
    "anastassiadis_splits": (
        PACKAGE
        / "results"
        / "public_anastassiadis_panel_validation"
        / "split_half_reliability.csv"
    ),
    "experimental_panel_transfer": (
        PACKAGE
        / "results"
        / "public_anastassiadis_panel_validation"
        / "same_endpoint_panel_transfer.csv"
    ),
    "anastassiadis_map_concordance": (
        PACKAGE
        / "results"
        / "public_anastassiadis_panel_validation"
        / "map_concordance.csv"
    ),
}

FIGURE_STEMS = (
    "fig1_spectra_residual",
    "fig2_support_domain",
    "fig3_pilot_decision",
    "fig4_experimental_boundary",
)

# Okabe--Ito categorical palette, with neutral greys for baselines/nulls.
COLORS = {
    "Docking-44": "#0072B2",
    "DOCKSTRING-58": "#D55E00",
    "DAVIS": "#009E73",
    "PKIS2": "#CC79A7",
    "HotSpot": "#E69F00",
    "orange": "#E69F00",
    "sky": "#56B4E9",
    "black": "#222222",
    "grey": "#777777",
    "light_grey": "#D3D3D3",
}

DATASET_ORDER = ("Docking-44", "DOCKSTRING-58")
PANEL_ORDER = ("DAVIS", "PKIS2")
EXPERIMENTAL_PANEL_ORDER = ("DAVIS", "PKIS2", "Anastassiadis")
RESIDUAL_CMAP = LinearSegmentedColormap.from_list(
    "accessible_residual",
    ["#0072B2", "#F7F7F7", "#D55E00"],
    N=256,
)


@dataclass(frozen=True)
class SpectralPanel:
    name: str
    targets: tuple[str, ...]
    rows: int
    raw_correlation: np.ndarray
    residual_correlation: np.ndarray
    raw_eigenvalues: np.ndarray
    residual_eigenvalues: np.ndarray
    pc1_loading: np.ndarray
    pc1_uniform_cosine: float


def configure_style() -> None:
    """Apply one compact, print-oriented style without pyplot state calls."""
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.0,
            "axes.titlesize": 8.7,
            "axes.labelsize": 8.2,
            "axes.linewidth": 0.7,
            "xtick.labelsize": 7.2,
            "ytick.labelsize": 7.2,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "legend.fontsize": 7.0,
            "legend.frameon": False,
            "lines.linewidth": 1.4,
            "lines.markersize": 4.2,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )


def new_figure(width: float, height: float) -> Figure:
    figure = Figure(figsize=(width, height), layout="constrained", facecolor="white")
    FigureCanvasAgg(figure)
    return figure


def panel_label(
    axis: Any,
    label: str,
    *,
    x: float = -0.10,
    y: float = 1.06,
) -> None:
    axis.text(
        x,
        y,
        label,
        transform=axis.transAxes,
        fontsize=10,
        fontweight="bold",
        va="bottom",
        ha="left",
    )


def clean_axis(axis: Any, *, grid: str | None = "y") -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    if grid:
        axis.grid(axis=grid, color="#E5E5E5", linewidth=0.55, zorder=0)
    axis.set_axisbelow(True)


def save_figure(figure: Figure, output: Path, stem: str) -> tuple[Path, Path]:
    output.mkdir(parents=True, exist_ok=True)
    pdf = output / f"{stem}.pdf"
    png = output / f"{stem}.png"
    figure.savefig(pdf, facecolor="white")
    figure.savefig(png, dpi=300, facecolor="white")
    figure.clear()
    return pdf, png


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"expected an object in {path}")
    return payload


def require_columns(frame: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"{name} is missing columns: {sorted(missing)}")


def finite_frame(frame: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    require_columns(frame, columns, name)
    values = frame[list(columns)].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"{name} contains non-finite plotted values")


def quantile_range(values: Iterable[float]) -> tuple[float, float, float]:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if len(array) < 2:
        raise ValueError("a plotted resampling distribution needs at least two values")
    return (
        float(np.quantile(array, 0.025)),
        float(np.median(array)),
        float(np.quantile(array, 0.975)),
    )


def load_vina_matrix(
    name: str,
    targets: tuple[str, ...],
) -> np.ndarray:
    """Read and preprocess one redistributed Vina matrix."""
    if name == "Docking-44":
        frame = pd.read_csv(INPUT_PATHS["docking44"], usecols=list(targets))
        values = frame.apply(pd.to_numeric, errors="coerce")
        values = values.fillna(values.mean(axis=0))
        matrix = values.to_numpy(dtype=np.float64)
        expected_rows = 12_651
    elif name == "DOCKSTRING-58":
        frame = pd.read_csv(
            INPUT_PATHS["dockstring58"],
            sep="\t",
            usecols=list(targets),
        )
        frame = frame.dropna(axis=0, how="any")
        matrix = frame.to_numpy(dtype=np.float64)
        expected_rows = 260_060
    else:
        raise ValueError(f"unknown Vina panel: {name}")
    if matrix.shape != (expected_rows, len(targets)) or not np.isfinite(matrix).all():
        raise ValueError(f"unexpected preprocessed shape for {name}: {matrix.shape}")
    return np.minimum(matrix, 0.0)


def correlation_spectrum(correlation: np.ndarray) -> np.ndarray:
    values = np.linalg.eigvalsh((correlation + correlation.T) / 2.0)
    values = np.clip(values[::-1], 0.0, None)
    return values / values.sum()


def participation_ratio_from_spectrum(spectrum: np.ndarray) -> float:
    values = np.asarray(spectrum, dtype=float)
    return float(np.square(values.sum()) / np.square(values).sum())


def spectral_panel(name: str, targets: tuple[str, ...]) -> SpectralPanel:
    matrix = load_vina_matrix(name, targets)
    raw = np.corrcoef(matrix, rowvar=False)
    column_mean = matrix.mean(axis=0, keepdims=True)
    row_mean = matrix.mean(axis=1, keepdims=True)
    residual = matrix - column_mean - row_mean + float(matrix.mean())
    residual_correlation = np.corrcoef(residual, rowvar=False)
    raw_values, raw_vectors = np.linalg.eigh((raw + raw.T) / 2.0)
    leading = raw_vectors[:, np.argmax(raw_values)]
    uniform = np.ones(len(targets), dtype=float) / np.sqrt(len(targets))
    if np.dot(leading, uniform) < 0:
        leading = -leading
    output = SpectralPanel(
        name=name,
        targets=targets,
        rows=len(matrix),
        raw_correlation=raw,
        residual_correlation=residual_correlation,
        raw_eigenvalues=correlation_spectrum(raw),
        residual_eigenvalues=correlation_spectrum(residual_correlation),
        pc1_loading=leading,
        pc1_uniform_cosine=float(np.dot(leading, uniform)),
    )
    del matrix, residual
    return output


def clustered_order(correlation: np.ndarray) -> np.ndarray:
    distance = np.sqrt(np.clip(2.0 * (1.0 - correlation), 0.0, None))
    np.fill_diagonal(distance, 0.0)
    condensed = squareform(distance, checks=False)
    linkage = hierarchy.linkage(condensed, method="average", optimal_ordering=True)
    return hierarchy.leaves_list(linkage)


def plot_loading_axis(axis: Any, panels: dict[str, SpectralPanel]) -> None:
    for name in DATASET_ORDER:
        payload = panels[name]
        ordered = np.sort(payload.pc1_loading)
        rank = np.linspace(0.0, 1.0, len(ordered))
        axis.plot(rank, ordered, color=COLORS[name], marker="o", markersize=2.4)
        uniform = 1.0 / np.sqrt(len(ordered))
        axis.axhline(uniform, color=COLORS[name], linestyle=":", linewidth=0.8)
        axis.text(
            1.015,
            ordered[-1],
            name.replace("DOCKSTRING-", "DS-"),
            color=COLORS[name],
            fontsize=6.5,
            va="center",
        )
    axis.axhline(0.0, color=COLORS["grey"], linewidth=0.7)
    axis.set_xlim(-0.02, 1.12)
    axis.set_xlabel("Target rank within panel")
    axis.set_ylabel("PC1 loading")
    axis.set_title("Shared favorable-score axis")
    axis.set_ylim(-0.028, 0.178)
    axis.text(
        0.98,
        0.035,
        "cos(uniform): "
        + f"D44 {panels['Docking-44'].pc1_uniform_cosine:.3f}\n"
        + f"DS58 {panels['DOCKSTRING-58'].pc1_uniform_cosine:.3f}",
        transform=axis.transAxes,
        fontsize=6.5,
        ha="right",
        va="bottom",
    )
    clean_axis(axis)


def plot_spectra(axis: Any, panels: dict[str, SpectralPanel]) -> None:
    for name in DATASET_ORDER:
        payload = panels[name]
        for label, spectrum, linestyle in (
            ("score", payload.raw_eigenvalues, "-"),
            ("residual", payload.residual_eigenvalues, "--"),
        ):
            positive = spectrum > 1e-12
            axis.plot(
                np.arange(1, len(spectrum) + 1)[positive],
                spectrum[positive],
                color=COLORS[name],
                linestyle=linestyle,
                label=f"{name}, {label}",
            )
    axis.set_yscale("log")
    axis.set_ylim(1e-4, 1.0)
    axis.set_xlim(1, 58)
    axis.set_xlabel("Principal component")
    axis.set_ylabel("Variance fraction (log scale)")
    axis.set_title("Centering spreads spectral variance")
    axis.legend(ncol=2, loc="upper right")
    clean_axis(axis, grid="both")


def plot_residual_heatmap(axis: Any, payload: SpectralPanel) -> Any:
    order = clustered_order(payload.residual_correlation)
    matrix = payload.residual_correlation[np.ix_(order, order)]
    image = axis.imshow(
        matrix,
        cmap=RESIDUAL_CMAP,
        vmin=-0.65,
        vmax=0.65,
        interpolation="nearest",
        rasterized=True,
    )
    step = 7 if len(order) <= 44 else 9
    ticks = np.arange(0, len(order), step)
    labels = [payload.targets[index] for index in order[ticks]]
    axis.set_xticks(ticks, labels, rotation=90, fontsize=6.0)
    axis.set_yticks(ticks, labels, fontsize=6.0)
    axis.set_title(f"{payload.name} residual map")
    axis.set_xlabel("Targets (clustered)")
    axis.set_ylabel("")
    return image


def plot_null_pr(axis: Any, null_summary: dict[str, Any]) -> None:
    null_keys = (
        ("additive_gaussian", "Additive"),
        ("empirical_residual_column_permutation", "Column perm."),
        ("row_norm_preserving_random_direction", "Row norm"),
        ("molecular_weight_conditional_permutation", "MW-cond."),
    )
    dataset_keys = (("docking44", "Docking-44"), ("dockstring58", "DOCKSTRING-58"))
    y_positions: list[float] = []
    y_labels: list[str] = []
    for group, (key, display) in enumerate(dataset_keys):
        dataset = null_summary["datasets"][key]
        base = 8.0 - group * 4.5
        color = COLORS[display]
        maximum = int(dataset["nulls"]["additive_gaussian"]["n_targets"]) - 1
        observed_raw = float(dataset["observed_residual_pr"])
        observed = observed_raw / maximum
        axis.scatter(observed, base, color=color, marker="D", s=28, zorder=4)
        axis.text(
            observed + 0.025,
            base,
            f"{observed_raw:.1f}/{maximum}",
            va="center",
            fontsize=6.5,
        )
        y_positions.append(base)
        y_labels.append("Observed")
        for offset, (null_key, label) in enumerate(null_keys, start=1):
            record = dataset["nulls"][null_key]
            distribution = record.get("null_residual")
            if distribution is None:
                distribution = record["mw_10_bins"][
                    "null_distributions_by_bin_design"
                ]["molecular_weight_quantile"]["participation_ratio"]
            median = float(distribution["median"]) / maximum
            lower, upper = (
                float(value) / maximum
                for value in distribution["interval_95"]
            )
            y = base - 0.72 * offset
            axis.errorbar(
                median,
                y,
                xerr=np.asarray([[median - lower], [upper - median]]),
                color=color,
                marker="o",
                markerfacecolor="white",
                markeredgewidth=0.9,
                capsize=2.0,
                linewidth=1.0,
                zorder=3,
            )
            y_positions.append(y)
            y_labels.append(label)
        axis.text(
            -0.02,
            base + 0.46,
            display,
            transform=axis.get_yaxis_transform(),
            ha="left",
            va="bottom",
            color=color,
            fontweight="bold",
            fontsize=6.7,
        )
    axis.set_yticks(y_positions, y_labels)
    axis.set_xlim(0, 1.04)
    axis.set_ylim(0.0, 8.8)
    axis.set_xlabel("Residual PR / (P − 1)")
    axis.set_title("Residual PR vs nulls")
    axis.text(
        0.98,
        0.02,
        "n=12,000; medians\nbars: 95% null ranges",
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=6.7,
        color=COLORS["grey"],
    )
    clean_axis(axis)


def make_figure_1(output: Path, core: dict[str, Any], nulls: dict[str, Any]) -> tuple[Path, Path]:
    panels: dict[str, SpectralPanel] = {}
    for name in DATASET_ORDER:
        targets = tuple(core["large_vina_panels"][name]["support"]["target_order"])
        panels[name] = spectral_panel(name, targets)

    figure = new_figure(6.3, 7.0)
    grid = figure.add_gridspec(2, 6, height_ratios=(0.78, 1.15))
    loading_axis = figure.add_subplot(grid[0, :2])
    spectra_axis = figure.add_subplot(grid[0, 2:])
    heatmap_44 = figure.add_subplot(grid[1, :2])
    heatmap_58 = figure.add_subplot(grid[1, 2:4])
    null_axis = figure.add_subplot(grid[1, 4:])

    plot_loading_axis(loading_axis, panels)
    plot_spectra(spectra_axis, panels)
    image = plot_residual_heatmap(heatmap_44, panels["Docking-44"])
    plot_residual_heatmap(heatmap_58, panels["DOCKSTRING-58"])
    colorbar = figure.colorbar(
        image,
        ax=[heatmap_44, heatmap_58],
        orientation="horizontal",
        fraction=0.06,
        pad=0.02,
        shrink=0.72,
    )
    colorbar.set_label("Residual target correlation")
    colorbar.ax.tick_params(labelsize=6)
    plot_null_pr(null_axis, nulls)

    for axis, label in zip(
        (loading_axis, spectra_axis, heatmap_44, heatmap_58, null_axis),
        "abcde",
        strict=True,
    ):
        panel_label(
            axis,
            label,
            x=-0.13 if label in {"c", "d"} else -0.10,
            y=1.13 if label in {"c", "d"} else 1.06,
        )
    return save_figure(figure, output, FIGURE_STEMS[0])


def plot_domain_controls(
    axis: Any,
    boundary: pd.DataFrame,
    global_controls: pd.DataFrame,
    within_controls: pd.DataFrame,
    support_seeds: pd.DataFrame,
) -> None:
    transform = "row_centered_residual"
    observed = boundary.loc[boundary["transformation"] == transform].copy()
    global_rows = global_controls.loc[
        (global_controls["transformation"] == transform)
        & (
            global_controls["control_type"]
            == "mw_matched_chemical_group_disjoint"
        )
    ].copy()
    within_rows = within_controls.loc[
        (within_controls["transformation"] == transform)
        & (
            within_controls["control_type"]
            == "mw_matched_chemical_group_disjoint"
        )
    ].copy()
    if len(observed) != 2 or len(global_rows) != 2 or len(within_rows) != 4:
        raise ValueError("unexpected residual chemical-domain control support")

    categories = (
        "Observed: low vs high quartiles",
        "Control: MW-matched full support",
        "Control: within low quartile",
        "Control: within high quartile",
    )
    row_positions = np.arange(3, -1, -1, dtype=float)
    offsets = {"Docking-44": 0.10, "DOCKSTRING-58": -0.10}
    for name in DATASET_ORDER:
        color = COLORS[name]
        offset = offsets[name]
        observed_row = observed.loc[observed["dataset"] == name]
        global_row = global_rows.loc[global_rows["dataset"] == name]
        if len(observed_row) != 1 or len(global_row) != 1:
            raise ValueError(f"missing chemical-domain contrast for {name}")
        observed_value = float(
            observed_row.iloc[0]["observed_low_high_geometry_spearman"]
        )
        axis.scatter(
            observed_value,
            row_positions[0] + offset,
            color=color,
            marker="D",
            s=29,
            zorder=4,
        )
        axis.text(
            observed_value + 0.025,
            row_positions[0] + offset,
            f"{observed_value:.3f}",
            color=color,
            fontsize=6.4,
            va="center",
        )
        if name == "DOCKSTRING-58":
            seed_values = support_seeds.loc[
                (support_seeds["dataset"] == name)
                & (support_seeds["transformation"] == transform),
                "geometry_spearman",
            ].to_numpy(dtype=float)
            if len(seed_values) != 6:
                raise ValueError("expected six DOCKSTRING residual support draws")
            axis.scatter(
                seed_values,
                np.full(len(seed_values), row_positions[0] - 0.29),
                color=color,
                marker="|",
                s=65,
                linewidths=0.9,
                zorder=3,
            )
            axis.text(
                0.43,
                row_positions[0] - 0.29,
                "six seeded n=15,000 supports",
                color=color,
                fontsize=5.7,
                va="center",
            )
        control_rows = [global_row.iloc[0]]
        for band in ("low_mw", "high_mw"):
            record = within_rows.loc[
                (within_rows["dataset"] == name)
                & (within_rows["mw_band"] == band)
            ]
            if len(record) != 1:
                raise ValueError(f"missing {band} reproducibility control for {name}")
            control_rows.append(record.iloc[0])
        for category, record in enumerate(control_rows, start=1):
            median = float(record["geometry_spearman_median"])
            lower = float(record["geometry_spearman_q025"])
            upper = float(record["geometry_spearman_q975"])
            horizontal_interval_point(
                axis,
                median,
                row_positions[category] + offset,
                lower,
                upper,
                color=color,
                marker="o",
                filled=False,
            )
            axis.text(
                0.43,
                row_positions[category] + offset,
                f"{median:.3f} [{lower:.3f}–{upper:.3f}]",
                color=color,
                fontsize=5.8,
                va="center",
            )

    axis.set_yticks(row_positions, categories)
    axis.tick_params(axis="y", labelsize=6.0)
    axis.set_xlim(0.1, 1.025)
    axis.set_ylim(-0.38, 3.38)
    axis.set_xlabel("Residual-map agreement, ρ")
    axis.set_title("Quartile contrast vs controls")
    clean_axis(axis, grid="x")


def plot_threshold_sweep(axis: Any, sweep: pd.DataFrame) -> None:
    transform = "row_centered_residual"
    subset = sweep.loc[sweep["transformation"] == transform].copy()
    for name in DATASET_ORDER:
        rows = subset.loc[subset["dataset"] == name].sort_values("tail_fraction_per_side")
        if len(rows) != 7:
            raise ValueError(f"expected seven MW thresholds for {name}, {transform}")
        x = 100.0 * rows["tail_fraction_per_side"].to_numpy(dtype=float)
        observed = rows["geometry_spearman"].to_numpy(dtype=float)
        control = rows["control_geometry_spearman_mean"].to_numpy(dtype=float)
        lower = rows["control_geometry_spearman_q025"].to_numpy(dtype=float)
        upper = rows["control_geometry_spearman_q975"].to_numpy(dtype=float)
        axis.fill_between(x, lower, upper, color=COLORS[name], alpha=0.12, linewidth=0)
        axis.plot(x, control, color=COLORS[name], linestyle=":", linewidth=1.0)
        axis.plot(x, observed, color=COLORS[name], marker="o", label=name)
    axis.set_xlim(9, 41)
    axis.set_ylim(0.0, 1.03)
    axis.set_xticks(np.arange(10, 41, 5))
    axis.set_xlabel("Ligands per MW tail (%)")
    axis.set_ylabel("Residual-map agreement, ρ")
    axis.set_title("Tail-definition sensitivity")
    axis.axvline(25.0, color=COLORS["light_grey"], linewidth=0.8, zorder=0)
    axis.text(
        25.6,
        0.69,
        "quartiles in (a)\n(±1 boundary ligand)",
        fontsize=6.0,
        color=COLORS["grey"],
    )
    axis.text(
        0.02,
        0.04,
        "Solid: observed tails\n"
        "Random-row control: mean (dotted)\n"
        "Band: 2.5–97.5% range; groups may recur",
        transform=axis.transAxes,
        fontsize=6.6,
        va="bottom",
    )
    clean_axis(axis)


def plot_decile_distance(axis: Any, pairs: pd.DataFrame, trends: pd.DataFrame) -> None:
    transform = "row_centered_residual"
    subset = pairs.loc[pairs["transformation"] == transform].copy()
    for name in DATASET_ORDER:
        rows = subset.loc[subset["dataset"] == name]
        if len(rows) != 45:
            raise ValueError(f"expected 45 decile pairs for {name}, {transform}")
        x = rows["median_mw_separation"].to_numpy(dtype=float)
        y = rows["geometry_dissimilarity"].to_numpy(dtype=float)
        marker_size = 7.0 + 3.5 * rows["decile_separation"].to_numpy(dtype=float)
        axis.scatter(
            x,
            y,
            color=COLORS[name],
            s=marker_size,
            alpha=0.58,
            edgecolors="none",
            rasterized=True,
        )
        coefficients = np.polyfit(x, y, deg=1)
        line_x = np.linspace(x.min(), x.max(), 100)
        axis.plot(line_x, np.polyval(coefficients, line_x), color=COLORS[name])
        record = trends.loc[
            (trends["dataset"] == name) & (trends["transformation"] == transform)
        ]
        if len(record) != 1:
            raise ValueError(f"missing MW decile trend for {name}, {transform}")
    axis.set_xlabel("Median-MW separation (Da)")
    axis.set_ylabel("Map dissimilarity, 1 − ρ")
    axis.set_ylim(0.0, 1.15)
    axis.set_title("Ten-bin MW continuum")
    axis.text(
        0.98,
        0.04,
        "Size: bin-index separation\n"
        "45 dependent pairs; descriptive fits",
        transform=axis.transAxes,
        fontsize=6.7,
        ha="right",
        va="bottom",
    )
    clean_axis(axis)


def plot_descriptor_domain_specificity(
    axis: Any,
    descriptors: pd.DataFrame,
    controls: pd.DataFrame,
) -> None:
    rows = descriptors.loc[
        descriptors["transformation"] == "row_centered_residual"
    ].copy()
    order = (
        "molecular_weight",
        "heavy_atoms",
        "labute_asa",
        "tpsa",
        "clogp",
        "rotatable_bonds",
        "ring_count",
    )
    labels = (
        "Mol. weight",
        "Heavy atoms",
        "Labute ASA",
        "TPSA",
        "cLogP",
        "Rotatable",
        "Ring count",
    )
    y_positions = np.arange(len(order) - 1, -1, -1, dtype=float)
    offsets = {"Docking-44": 0.10, "DOCKSTRING-58": -0.10}
    if len(rows) != 14:
        raise ValueError("expected seven residual descriptor contrasts per dataset")
    if len(controls) != 14:
        raise ValueError("expected one matched-size random control per contrast")
    for descriptor, y in zip(order, y_positions, strict=True):
        descriptor_rows = rows.loc[rows["descriptor"] == descriptor]
        if set(descriptor_rows["dataset"]) != set(DATASET_ORDER):
            raise ValueError(f"missing descriptor-domain contrast for {descriptor}")
        values: list[float] = []
        for name in DATASET_ORDER:
            observed_row = descriptor_rows.loc[descriptor_rows["dataset"] == name]
            control_row = controls.loc[
                (controls["dataset"] == name)
                & (controls["descriptor"] == descriptor)
            ]
            if len(observed_row) != 1 or len(control_row) != 1:
                raise ValueError(f"non-unique descriptor/control row for {name}/{descriptor}")
            value = float(observed_row["geometry_spearman"].iloc[0])
            control = control_row.iloc[0]
            values.append(value)
            axis.plot(
                [value, float(control["random_disjoint_median"])],
                [y + offsets[name], y + offsets[name]],
                color=COLORS[name],
                alpha=0.24,
                linewidth=0.7,
                zorder=1,
            )
            axis.scatter(
                value,
                y + offsets[name],
                color=COLORS[name],
                s=24,
                zorder=3,
            )
            horizontal_interval_point(
                axis,
                float(control["random_disjoint_median"]),
                y + offsets[name],
                float(control["random_disjoint_q025"]),
                float(control["random_disjoint_q975"]),
                color=COLORS[name],
                filled=False,
                zorder=2,
            )
        axis.plot(
            values,
            [y + offsets[name] for name in DATASET_ORDER],
            color=COLORS["light_grey"],
            linewidth=0.8,
            zorder=1,
        )
    axis.axhspan(3.5, 6.45, color=COLORS["light_grey"], alpha=0.20, zorder=0)
    axis.text(
        0.93,
        6.30,
        "size-related axes",
        color=COLORS["grey"],
        fontsize=5.8,
        ha="right",
        va="top",
    )
    axis.set_yticks(y_positions, labels)
    axis.tick_params(axis="y", labelsize=6.5)
    axis.set_xlim(0.1, 1.005)
    axis.set_ylim(-0.45, 6.45)
    axis.set_xlabel("Tail residual-map agreement, ρ")
    axis.set_title("Descriptor tails vs random")
    clean_axis(axis, grid="x")


def make_figure_2(
    output: Path,
    sweep: pd.DataFrame,
    pairs: pd.DataFrame,
    trends: pd.DataFrame,
    boundary: pd.DataFrame,
    global_controls: pd.DataFrame,
    within_controls: pd.DataFrame,
    support_seeds: pd.DataFrame,
    descriptors: pd.DataFrame,
    descriptor_controls: pd.DataFrame,
) -> tuple[Path, Path]:
    finite_frame(
        sweep,
        (
            "tail_fraction_per_side",
            "geometry_spearman",
            "control_geometry_spearman_mean",
            "control_geometry_spearman_q025",
            "control_geometry_spearman_q975",
        ),
        "MW threshold sweep",
    )
    finite_frame(
        pairs,
        ("median_mw_separation", "geometry_dissimilarity"),
        "MW decile pairs",
    )
    finite_frame(
        boundary,
        ("observed_low_high_geometry_spearman",),
        "chemical-domain boundary",
    )
    for frame, name in (
        (global_controls, "global group-disjoint controls"),
        (within_controls, "within-band group-disjoint controls"),
    ):
        finite_frame(
            frame,
            (
                "geometry_spearman_median",
                "geometry_spearman_q025",
                "geometry_spearman_q975",
            ),
            name,
        )
    finite_frame(support_seeds, ("geometry_spearman",), "support-seed contrasts")
    finite_frame(descriptors, ("geometry_spearman",), "descriptor-domain contrasts")
    finite_frame(
        descriptor_controls,
        (
            "random_disjoint_median",
            "random_disjoint_q025",
            "random_disjoint_q975",
        ),
        "descriptor-domain matched-size random controls",
    )
    figure = new_figure(6.3, 6.0)
    axes = figure.subplots(2, 2, squeeze=False)
    plot_domain_controls(
        axes[0, 0],
        boundary,
        global_controls,
        within_controls,
        support_seeds,
    )
    plot_threshold_sweep(axes[0, 1], sweep)
    plot_decile_distance(axes[1, 0], pairs, trends)
    plot_descriptor_domain_specificity(
        axes[1, 1], descriptors, descriptor_controls
    )
    handles = [
        Line2D([0], [0], color=COLORS[name], marker="o", label=name)
        for name in DATASET_ORDER
    ]
    handles.extend(
        [
            Line2D(
                [0],
                [0],
                color=COLORS["black"],
                marker="D",
                linestyle="none",
                label="Observed",
            ),
            Line2D(
                [0],
                [0],
                color=COLORS["black"],
                marker="o",
                markerfacecolor="white",
                linestyle="none",
                label="Control median/split range",
            ),
        ]
    )
    figure.legend(handles=handles, loc="outside upper center", ncol=4)
    for axis, label in zip(axes.flat, "abcd", strict=True):
        panel_label(axis, label)
    return save_figure(figure, output, FIGURE_STEMS[1])


def interval_point(
    axis: Any,
    x: float,
    mean: float,
    lower: float,
    upper: float,
    *,
    color: str,
    marker: str = "o",
    filled: bool = True,
    zorder: int = 3,
) -> None:
    axis.errorbar(
        x,
        mean,
        yerr=np.asarray([[mean - lower], [upper - mean]]),
        color=color,
        marker=marker,
        markerfacecolor=color if filled else "white",
        markeredgecolor=color,
        markeredgewidth=0.9,
        capsize=2.2,
        linewidth=1.0,
        zorder=zorder,
    )


def horizontal_interval_point(
    axis: Any,
    x: float,
    y: float,
    lower: float,
    upper: float,
    *,
    color: str,
    marker: str = "o",
    filled: bool = True,
    zorder: int = 3,
) -> None:
    axis.errorbar(
        x,
        y,
        xerr=np.asarray([[x - lower], [upper - x]]),
        color=color,
        marker=marker,
        markerfacecolor=color if filled else "white",
        markeredgecolor=color,
        markeredgewidth=0.9,
        capsize=2.2,
        linewidth=1.0,
        zorder=zorder,
    )


def primary_recovery_tables(
    edges: pd.DataFrame,
    decisions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    primary_edges = edges.loc[
        edges["reference_scope"] == "row_disjoint_complement_map"
    ].copy()
    primary_decisions = decisions.loc[
        (decisions["reference_scope"] == "row_disjoint_complement_map")
        & (decisions["panel_targets_k"] == 8)
    ].copy()
    expected = {(name, n) for name in DATASET_ORDER for n in (200, 500)}
    for name, frame in (("edge", primary_edges), ("decision", primary_decisions)):
        observed = set(
            zip(
                frame["dataset"].astype(str),
                frame["calibration_ligands"].astype(int),
                strict=True,
            )
        )
        if observed != expected or len(frame) != 4:
            raise ValueError(f"unexpected primary {name}-recovery support: {observed}")
    return primary_edges, primary_decisions


def plot_global_recovery(axis: Any, edges: pd.DataFrame) -> None:
    for dataset_index, name in enumerate(DATASET_ORDER):
        rows = edges.loc[edges["dataset"] == name].sort_values("calibration_ligands")
        x = rows["calibration_ligands"].to_numpy(dtype=float) + (-8.0 if dataset_index == 0 else 8.0)
        mean = rows["geometry_spearman_mean"].to_numpy(dtype=float)
        lower = rows["geometry_spearman_q025"].to_numpy(dtype=float)
        upper = rows["geometry_spearman_q975"].to_numpy(dtype=float)
        axis.plot(x, mean, color=COLORS[name], alpha=0.55)
        for point_x, point, lo, hi in zip(x, mean, lower, upper, strict=True):
            interval_point(axis, point_x, point, lo, hi, color=COLORS[name])
    axis.set_xticks([200, 500])
    axis.set_xlim(150, 550)
    axis.set_ylim(0.84, 1.0)
    axis.set_xlabel("Calibration ligands")
    axis.set_ylabel("Pilot vs row-disjoint map Spearman ρ")
    axis.set_title("Global map recovery")
    axis.text(
        0.03,
        0.05,
        "Points: means; bars: 2.5–97.5%\nacross 100 calibration supports",
        transform=axis.transAxes,
        fontsize=6.6,
    )
    clean_axis(axis)


def plot_edge_recovery(axis: Any, edges: pd.DataFrame) -> None:
    metrics = (
        ("edge_sign_agreement", "Edge sign", "o", "-"),
        ("top_positive_10_precision", "Top +10", "^", "--"),
        ("top_negative_10_precision", "Top −10", "s", ":"),
    )
    for dataset_index, name in enumerate(DATASET_ORDER):
        rows = edges.loc[edges["dataset"] == name].sort_values("calibration_ligands")
        x = rows["calibration_ligands"].to_numpy(dtype=float) + (-8.0 if dataset_index == 0 else 8.0)
        for metric_index, (prefix, _label, marker, linestyle) in enumerate(metrics):
            metric_x = x + (-4.0 + 4.0 * metric_index)
            mean = rows[f"{prefix}_mean"].to_numpy(dtype=float)
            lower = rows[f"{prefix}_q025"].to_numpy(dtype=float)
            upper = rows[f"{prefix}_q975"].to_numpy(dtype=float)
            axis.plot(metric_x, mean, color=COLORS[name], linestyle=linestyle, alpha=0.75)
            for point_x, point, lo, hi in zip(metric_x, mean, lower, upper, strict=True):
                interval_point(
                    axis,
                    point_x,
                    point,
                    lo,
                    hi,
                    color=COLORS[name],
                    marker=marker,
                    filled=prefix == "edge_sign_agreement",
                )
    axis.set_xticks([200, 500])
    axis.set_xlim(150, 550)
    axis.set_ylim(0.1, 1.01)
    axis.set_xlabel("Calibration ligands")
    axis.set_ylabel("Recovered edge fraction")
    axis.set_title("Signs stabilize before extreme edges")
    metric_handles = [
        Line2D([0], [0], color=COLORS["black"], marker=marker, linestyle=linestyle, label=label)
        for _prefix, label, marker, linestyle in metrics
    ]
    axis.legend(handles=metric_handles, loc="lower right")
    clean_axis(axis)


def plot_panel_decision(axis: Any, decisions: pd.DataFrame) -> None:
    for dataset_index, name in enumerate(DATASET_ORDER):
        rows = decisions.loc[decisions["dataset"] == name].sort_values("calibration_ligands")
        x = rows["calibration_ligands"].to_numpy(dtype=float) + (-8.0 if dataset_index == 0 else 8.0)
        specifications = (
            (
                "cluster_adjusted_rand_index",
                "o",
                True,
                "-",
                1.0,
                0.0,
            ),
            (
                "sample_vs_fixed_full_source_panel_medoid_overlap",
                "s",
                False,
                "--",
                8.0,
                8.0 / (44.0 if name == "Docking-44" else 58.0),
            ),
        )
        for metric_index, (
            prefix,
            marker,
            filled,
            linestyle,
            denominator,
            chance,
        ) in enumerate(specifications):
            metric_x = x + (-3.0 if metric_index == 0 else 3.0)
            mean = rows[f"{prefix}_mean"].to_numpy(dtype=float) / denominator
            lower = rows[f"{prefix}_q025"].to_numpy(dtype=float) / denominator
            upper = rows[f"{prefix}_q975"].to_numpy(dtype=float) / denominator
            if chance:
                mean = (mean - chance) / (1.0 - chance)
                lower = (lower - chance) / (1.0 - chance)
                upper = (upper - chance) / (1.0 - chance)
            axis.plot(metric_x, mean, color=COLORS[name], linestyle=linestyle, alpha=0.75)
            for point_x, point, lo, hi in zip(metric_x, mean, lower, upper, strict=True):
                interval_point(
                    axis,
                    point_x,
                    point,
                    lo,
                    hi,
                    color=COLORS[name],
                    marker=marker,
                    filled=filled,
                )
    axis.set_xticks([200, 500])
    axis.set_xlim(150, 550)
    axis.axhline(0.0, color=COLORS["grey"], linewidth=0.7)
    axis.set_ylim(-0.2, 1.02)
    axis.set_xlabel("Calibration ligands")
    axis.set_ylabel("Chance-adjusted decision recovery")
    axis.set_title("k=8 decision recovery")
    axis.legend(
        handles=[
            Line2D([0], [0], color=COLORS["black"], marker="o", label="Cluster ARI"),
            Line2D(
                [0],
                [0],
                color=COLORS["black"],
                marker="s",
                markerfacecolor="white",
                linestyle="--",
                label="Adjusted medoid overlap",
            ),
        ],
        loc="lower right",
    )
    clean_axis(axis)


def plot_coverage_vs_random(axis: Any, decisions: pd.DataFrame) -> None:
    records = decisions.sort_values(["dataset", "calibration_ligands"])
    x = np.arange(len(records), dtype=float)
    for position, (_, row) in zip(x, records.iterrows(), strict=True):
        name = str(row["dataset"])
        color = COLORS[name]
        sample = float(row["sample_map_medoids_mean_nearest_distance_on_reference_map_mean"])
        sample_lower = float(row["sample_map_medoids_mean_nearest_distance_on_reference_map_q025"])
        sample_upper = float(row["sample_map_medoids_mean_nearest_distance_on_reference_map_q975"])
        random_median = float(row["random_mean_nearest_distance_median_mean"])
        fixed = float(row["fixed_full_source_panel_mean_nearest_distance_on_reference_map_mean"])
        denominator = random_median - fixed
        if denominator <= 0:
            raise ValueError("random coverage must be worse than full-map coverage")
        remaining = (sample - fixed) / denominator
        remaining_lower = (sample_lower - fixed) / denominator
        remaining_upper = (sample_upper - fixed) / denominator
        interval_point(
            axis,
            position,
            remaining,
            remaining_lower,
            remaining_upper,
            color=color,
            marker="o",
        )
    labels = [
        f"{str(row.dataset).replace('DOCKSTRING-', 'DS-')}\nn={int(row.calibration_ligands)}"
        for row in records.itertuples(index=False)
    ]
    axis.set_xticks(x, labels)
    axis.axhline(0.0, color=COLORS["black"], linewidth=0.8, linestyle="-")
    axis.axhline(1.0, color=COLORS["grey"], linewidth=0.8, linestyle="--")
    axis.set_ylim(-0.05, 1.05)
    axis.set_ylabel("Random-to-full gap remaining")
    axis.set_title("k=8 coverage gap remaining")
    axis.legend(
        handles=[
            Line2D([0], [0], color=COLORS["black"], linestyle="-", label="Full-map medoids (0)"),
            Line2D([0], [0], color=COLORS["grey"], linestyle="--", label="Random-panel median (1)"),
        ],
        loc="upper right",
    )
    clean_axis(axis)


def plot_omitted_target_reconstruction(
    axis: Any,
    panel_metrics: pd.DataFrame,
) -> None:
    """Show how much of each omitted score surface is reconstructed."""
    rows = panel_metrics.loc[
        panel_metrics["pilot_ligands"].eq(500)
        & panel_metrics["selector"].eq("pilot_residual_r2_exact_kmedoids")
        & panel_metrics["predictor"].eq("multivariate_ridge_gcv")
        & panel_metrics["panel_k"].isin((4, 6, 8, 10, 12))
        & panel_metrics["surface"].isin(("raw", "raw_row_centered_residual"))
    ].copy()
    expected = 2 * 5 * 5 * 2
    if len(rows) != expected:
        raise ValueError(
            f"unexpected omitted-target reconstruction support: {len(rows)}"
        )
    surface_styles = {
        "raw": ("score", "-", "o", True),
        "raw_row_centered_residual": ("residual", "--", "s", False),
    }
    for dataset_index, name in enumerate(DATASET_ORDER):
        color = COLORS[name]
        for surface, (_label, linestyle, marker, filled) in surface_styles.items():
            selected = rows.loc[
                rows["dataset"].eq(name) & rows["surface"].eq(surface)
            ]
            summary = selected.groupby("panel_k", sort=True)[
                "omitted_only_variance_weighted_r2"
            ].agg(["median", "min", "max"])
            if list(summary.index) != [4, 6, 8, 10, 12]:
                raise ValueError(f"incomplete panel-size grid for {name}, {surface}")
            x = summary.index.to_numpy(dtype=float) + (
                -0.10 if dataset_index == 0 else 0.10
            )
            y = summary["median"].to_numpy(dtype=float)
            axis.plot(x, y, color=color, linestyle=linestyle, alpha=0.85)
            for point_x, point, lower, upper in zip(
                x,
                y,
                summary["min"].to_numpy(dtype=float),
                summary["max"].to_numpy(dtype=float),
                strict=True,
            ):
                interval_point(
                    axis,
                    point_x,
                    point,
                    lower,
                    upper,
                    color=color,
                    marker=marker,
                    filled=filled,
                )
    axis.set_xticks([4, 6, 8, 10, 12])
    axis.set_xlim(3.35, 12.65)
    axis.set_ylim(0.0, 0.96)
    axis.set_xlabel("Targets scored per ligand (k)")
    axis.set_ylabel("Omitted-target variance-weighted $R^2$")
    axis.set_title("Omitted-target reconstruction")
    axis.legend(
        handles=[
            Line2D(
                [0],
                [0],
                color=COLORS["black"],
                marker="o",
                label="Score surface",
            ),
            Line2D(
                [0],
                [0],
                color=COLORS["black"],
                marker="s",
                markerfacecolor="white",
                linestyle="--",
                label="Residual surface",
            ),
        ],
        loc="center right",
    )
    axis.text(
        0.02,
        0.04,
        "Medians and fold ranges; five chemical-group-disjoint folds",
        transform=axis.transAxes,
        fontsize=6.4,
    )
    clean_axis(axis)


def plot_k8_residual_controls(
    axis: Any,
    panel_metrics: pd.DataFrame,
    direct_summary: pd.DataFrame,
) -> None:
    """Diagnose whether residual reconstruction is one-factor or pipeline-bound."""
    rows = panel_metrics.loc[
        panel_metrics["pilot_ligands"].eq(500)
        & panel_metrics["panel_k"].eq(8)
        & panel_metrics["selector"].eq("pilot_residual_r2_exact_kmedoids")
        & panel_metrics["surface"].eq("raw_row_centered_residual")
        & panel_metrics["predictor"].isin(
            ("multivariate_ridge_gcv", "selected_target_pc1")
        )
    ].copy()
    if len(rows) != 2 * 5 * 2 or len(direct_summary) != 2:
        raise ValueError("unexpected k=8 residual-control support")
    methods = (
        ("multivariate_ridge_gcv", "Derived Ridge", "o", True, -0.16),
        ("direct_residual", "Direct residual Ridge", "D", False, 0.0),
        ("selected_target_pc1", "Selected-target PC1", "s", False, 0.16),
    )
    for dataset_index, name in enumerate(DATASET_ORDER):
        color = COLORS[name]
        direct = direct_summary.loc[direct_summary["dataset"].eq(name)]
        if len(direct) != 1:
            raise ValueError(f"missing direct-residual summary for {name}")
        for method, _label, marker, filled, offset in methods:
            if method == "direct_residual":
                point = float(direct.iloc[0]["direct_ridge_variance_weighted_r2_median"])
                lower = float(direct.iloc[0]["direct_ridge_variance_weighted_r2_min"])
                upper = float(direct.iloc[0]["direct_ridge_variance_weighted_r2_max"])
            else:
                values = rows.loc[
                    rows["dataset"].eq(name) & rows["predictor"].eq(method),
                    "omitted_only_variance_weighted_r2",
                ].to_numpy(dtype=float)
                if len(values) != 5:
                    raise ValueError(f"incomplete k=8 {method} support for {name}")
                lower, point, upper = (
                    float(np.min(values)),
                    float(np.median(values)),
                    float(np.max(values)),
                )
            interval_point(
                axis,
                dataset_index + offset,
                point,
                lower,
                upper,
                color=color,
                marker=marker,
                filled=filled,
            )
    axis.set_xticks([0, 1], ["Docking-44", "DOCKSTRING-58"])
    axis.set_xlim(-0.48, 1.48)
    axis.set_ylim(0.0, 0.36)
    axis.set_ylabel("Residual omitted-target $R^2$")
    axis.set_title("Direct and derived residual fits agree")
    axis.legend(
        handles=[
            Line2D(
                [0],
                [0],
                color=COLORS["black"],
                marker=marker,
                markerfacecolor=(COLORS["black"] if filled else "white"),
                linestyle="none",
                label=label,
            )
            for _method, label, marker, filled, _offset in methods
        ],
        loc="lower left",
    )
    clean_axis(axis)


def make_figure_3(
    output: Path,
    edges: pd.DataFrame,
    decisions: pd.DataFrame,
    panel_metrics: pd.DataFrame,
    direct_summary: pd.DataFrame,
) -> tuple[Path, Path]:
    primary_edges, primary_decisions = primary_recovery_tables(edges, decisions)
    figure = new_figure(6.3, 5.7)
    axes = figure.subplots(2, 2, squeeze=False)
    plot_global_recovery(axes[0, 0], primary_edges)
    plot_omitted_target_reconstruction(axes[0, 1], panel_metrics)
    plot_coverage_vs_random(axes[1, 0], primary_decisions)
    plot_k8_residual_controls(axes[1, 1], panel_metrics, direct_summary)
    dataset_handles = [
        Line2D([0], [0], color=COLORS[name], marker="o", label=name)
        for name in DATASET_ORDER
    ]
    figure.legend(handles=dataset_handles, loc="outside upper center", ncol=2)
    for axis, label in zip(axes.flat, "abcd", strict=True):
        panel_label(axis, label)
    return save_figure(figure, output, FIGURE_STEMS[2])


def primary_experimental_splits(
    reliability: pd.DataFrame,
    same_splits: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    full = reliability.loc[
        (reliability["support"] == "released_full_panel")
        & (reliability["transform"] == "two_way_centered")
        & (reliability["estimator"] == "empirical")
        & (reliability["split_unit"] == "butina_cluster")
        & (reliability["valid_split"].astype(bool))
    ].copy()
    matched = same_splits.loc[
        (same_splits["support"] == "informative_exact_matches_primary")
        & (same_splits["transform"] == "two_way_centered")
        & (same_splits["split_unit"] == "butina_cluster")
        & (same_splits["valid_split"].astype(bool))
    ].copy()
    for name, frame in (("full experimental", full), ("same-support", matched)):
        counts = frame.groupby("panel").size().to_dict()
        if counts != {"DAVIS": 2_000, "PKIS2": 2_000}:
            raise ValueError(f"unexpected {name} split support: {counts}")
    return full, matched


def plot_experimental_map_reproducibility(
    axis: Any,
    reliability: pd.DataFrame,
    anastassiadis_splits: pd.DataFrame,
) -> None:
    positions = {"DAVIS": 0.0, "PKIS2": 1.0}
    for panel in PANEL_ORDER:
        values = reliability.loc[
            reliability["panel"] == panel, "split_map_spearman"
        ].to_numpy(dtype=float)
        lower, median, upper = quantile_range(values)
        interval_point(
            axis,
            positions[panel],
            median,
            lower,
            upper,
            color=COLORS[panel],
            marker="o",
        )
    hotspot_values = anastassiadis_splits[
        "cross_half_target_map_spearman"
    ].to_numpy(dtype=float)
    if len(hotspot_values) != 2_000:
        raise ValueError("unexpected Anastassiadis split-half support")
    lower, median, upper = quantile_range(hotspot_values)
    interval_point(
        axis,
        2.0,
        median,
        lower,
        upper,
        color=COLORS["HotSpot"],
        marker="o",
    )
    axis.set_xticks([0.0, 1.0, 2.0], ["DAVIS", "PKIS2", "HotSpot"])
    axis.set_xlim(-0.45, 2.45)
    axis.set_ylim(0.40, 1.0)
    axis.set_ylabel("Experimental map edge-rank ρ")
    axis.set_title("Experimental-map repeatability")
    axis.text(
        0.02,
        0.04,
        "Bars: 2.5–97.5% across 2,000 disjoint-half splits\n"
        "DAVIS/PKIS2: Butina clusters; HotSpot: ligand rows",
        transform=axis.transAxes,
        fontsize=6.3,
    )
    clean_axis(axis)


def plot_experimental_panel_transfer(
    axis: Any,
    transfer: pd.DataFrame,
) -> None:
    rows = transfer.loc[
        transfer["experimental_endpoint"].eq("metric_row_centered_residual")
        & transfer["objective"].eq("signed_1_minus_r")
    ].copy()
    if len(rows) != 15:
        raise ValueError(f"unexpected primary panel-transfer support: {len(rows)}")
    colors = {
        "DAVIS": COLORS["DAVIS"],
        "PKIS2": COLORS["PKIS2"],
        "Anastassiadis": COLORS["HotSpot"],
    }
    for panel in EXPERIMENTAL_PANEL_ORDER:
        panel_rows = rows.loc[rows["evaluation_map"].eq(panel)].sort_values(
            "targets_selected_k"
        )
        if list(panel_rows["targets_selected_k"]) != [4, 6, 8, 10, 12]:
            raise ValueError(f"incomplete panel-transfer k grid for {panel}")
        x = panel_rows["targets_selected_k"].to_numpy(dtype=float)
        raw_contrast = panel_rows[
            "all_optimal_pair_coverage_loss_improvement_min_conservative"
        ].to_numpy(dtype=float)
        sequence_contrast = panel_rows[
            "sequence_minus_residual_selected_coverage_loss_improvement"
        ].to_numpy(dtype=float)
        axis.plot(
            x,
            raw_contrast,
            color=colors[panel],
            marker="o",
            linestyle="-",
            label=("HotSpot" if panel == "Anastassiadis" else panel),
        )
        axis.plot(
            x,
            sequence_contrast,
            color=colors[panel],
            marker="x",
            linestyle=":",
            linewidth=0.9,
            alpha=0.72,
        )
    axis.axhline(0.0, color=COLORS["black"], linewidth=0.8)
    axis.set_xticks([4, 6, 8, 10, 12])
    axis.set_xlim(3.4, 12.6)
    axis.set_ylim(-0.06, 0.19)
    axis.set_xlabel("Targets selected (k)")
    axis.set_ylabel("Coverage-loss improvement")
    axis.set_title("Residual panels lower co-response loss")
    axis.legend(
        handles=[
            Line2D(
                [0],
                [0],
                color=COLORS["black"],
                marker="o",
                linestyle="-",
                label="Best raw − worst residual",
            ),
            Line2D(
                [0],
                [0],
                color=COLORS["grey"],
                marker="x",
                linestyle=":",
                label="Sequence − residual",
            ),
        ],
        loc="upper right",
    )
    axis.text(
        0.02,
        0.04,
        "Positive values favour the residual-selected panel",
        transform=axis.transAxes,
        fontsize=6.4,
    )
    clean_axis(axis)


def plot_same_support_geometry(
    axis: Any,
    geometry: pd.DataFrame,
    bootstrap: pd.DataFrame,
) -> None:
    rows = geometry.loc[
        (geometry["support"] == "informative_exact_matches_primary")
    ].copy()
    samples = bootstrap.loc[
        (bootstrap["support"] == "informative_exact_matches_primary")
        & (bootstrap["bootstrap_unit"] == "butina_cluster")
    ].copy()
    if set(rows["panel"]) != set(PANEL_ORDER) or len(rows) != 4:
        raise ValueError("unexpected exact same-support geometry table")
    categories: list[str] = []
    position = 0
    for panel in PANEL_ORDER:
        for transform, transform_label in (("raw", "score"), ("two_way_centered", "residual")):
            row = rows.loc[
                (rows["panel"] == panel) & (rows["transform"] == transform)
            ].iloc[0]
            draws = samples.loc[
                (samples["panel"] == panel) & (samples["transform"] == transform)
            ]
            if len(draws) != 2_000:
                raise ValueError(f"unexpected bootstrap support for {panel}, {transform}")
            broad = float(row["broad_support_empirical_spearman"])
            same = float(row["same_support_empirical_spearman"])
            broad_lower, _broad_median, broad_upper = quantile_range(
                draws["broad_support_spearman"]
            )
            same_lower, _same_median, same_upper = quantile_range(
                draws["same_support_spearman"]
            )
            color = COLORS[panel]
            interval_point(
                axis,
                position - 0.08,
                broad,
                broad_lower,
                broad_upper,
                color=color,
                marker="s",
                filled=False,
            )
            interval_point(
                axis,
                position + 0.08,
                same,
                same_lower,
                same_upper,
                color=color,
                marker="o",
                filled=True,
            )
            axis.plot(
                [position - 0.08, position + 0.08],
                [broad, same],
                color=color,
                linewidth=1.0,
                zorder=2,
            )
            categories.append(f"{panel}\n{transform_label}")
            position += 1
    axis.axhline(0.0, color=COLORS["grey"], linewidth=0.7)
    axis.set_xticks(np.arange(4), categories)
    axis.set_xlim(-0.45, 3.45)
    axis.set_ylim(-0.3, 0.48)
    axis.set_ylabel("Docking–experiment map Spearman ρ")
    axis.set_title("Vina concordance by ligand support")
    axis.legend(
        handles=[
            Line2D(
                [0],
                [0],
                color=COLORS["black"],
                marker="s",
                markerfacecolor="white",
                linestyle="none",
                label="De-leaked broad Vina support",
            ),
            Line2D([0], [0], color=COLORS["black"], marker="o", linestyle="none", label="Exact same ligands"),
        ],
        loc="lower right",
    )
    axis.text(
        0.02,
        0.97,
        "Bars: 2.5–97.5% across 2,000 Butina-cluster bootstraps",
        transform=axis.transAxes,
        va="top",
        fontsize=6.5,
    )
    clean_axis(axis)


def plot_matched_half_maps(axis: Any, matched: pd.DataFrame) -> None:
    metrics = (
        ("experimental_split_map_spearman", "Experiment self", "o", True),
        ("docking_split_map_spearman", "Vina self", "s", False),
        ("cross_half_cross_modal_spearman", "Vina vs experiment", "D", True),
    )
    offsets = (-0.19, 0.0, 0.19)
    for panel_index, panel in enumerate(PANEL_ORDER):
        rows = matched.loc[matched["panel"] == panel]
        for offset, (column, _label, marker, filled) in zip(offsets, metrics, strict=True):
            lower, median, upper = quantile_range(rows[column].to_numpy(dtype=float))
            color = COLORS[panel] if column != "cross_half_cross_modal_spearman" else COLORS["black"]
            interval_point(
                axis,
                panel_index + offset,
                median,
                lower,
                upper,
                color=color,
                marker=marker,
                filled=filled,
            )
    axis.set_xticks([0, 1], ["DAVIS\n56 exact ligands", "PKIS2\n154 exact ligands"])
    axis.set_xlim(-0.48, 1.48)
    axis.set_ylim(-0.05, 0.9)
    axis.set_ylabel("Half-map edge-rank ρ")
    axis.set_title("Matched-support half-map stability")
    axis.legend(
        handles=[
            Line2D([0], [0], color=COLORS["black"], marker=marker, markerfacecolor="white" if not filled else COLORS["black"], linestyle="none", label=label)
            for _column, label, marker, filled in metrics
        ],
        loc="lower right",
    )
    axis.text(
        0.02,
        0.96,
        "Cluster-disjoint halves; bars are split-distribution ranges",
        transform=axis.transAxes,
        va="top",
        fontsize=6.5,
    )
    clean_axis(axis)


def plot_cross_assay_map_concordance(
    axis: Any,
    davis_pkis2: pd.DataFrame,
    anastassiadis: pd.DataFrame,
) -> None:
    """Compare raw and row-centred map agreement across assay technologies."""
    pair_values: list[tuple[str, float, float]] = []
    dp = davis_pkis2.set_index("transform")
    if not {"raw", "two_way_centered"} <= set(dp.index):
        raise ValueError("DAVIS--PKIS2 map concordance is incomplete")
    pair_values.append(
        (
            "DAVIS--PKIS2",
            float(dp.loc["raw", "edge_spearman"]),
            float(dp.loc["two_way_centered", "edge_spearman"]),
        )
    )
    for comparison, label in (("DAVIS", "DAVIS--HotSpot"), ("PKIS2", "PKIS2--HotSpot")):
        rows = anastassiadis.loc[anastassiadis["comparison_panel"].eq(comparison)]
        indexed = rows.set_index("anastassiadis_transformation")
        if not {"raw_correlation", "row_centered_correlation"} <= set(indexed.index):
            raise ValueError(f"HotSpot map concordance is incomplete for {comparison}")
        pair_values.append(
            (
                label,
                float(indexed.loc["raw_correlation", "target_pair_spearman"]),
                float(
                    indexed.loc[
                        "row_centered_correlation", "target_pair_spearman"
                    ]
                ),
            )
        )
    for position, (_label, raw, residual) in enumerate(pair_values):
        axis.plot(
            [position - 0.09, position + 0.09],
            [raw, residual],
            color=COLORS["light_grey"],
            linewidth=1.2,
            zorder=1,
        )
        axis.scatter(
            position - 0.09,
            raw,
            facecolor="white",
            edgecolor=COLORS["grey"],
            marker="s",
            s=26,
            zorder=3,
        )
        axis.scatter(
            position + 0.09,
            residual,
            color=COLORS["black"],
            marker="o",
            s=28,
            zorder=3,
        )
    axis.set_xticks(
        np.arange(3),
        ["DAVIS–\nPKIS2", "DAVIS–\nHotSpot", "PKIS2–\nHotSpot"],
    )
    axis.set_xlim(-0.45, 2.45)
    axis.set_ylim(0.45, 0.90)
    axis.set_ylabel("Cross-assay edge-rank ρ")
    axis.set_title("Cross-assay map agreement")
    axis.legend(
        handles=[
            Line2D(
                [0],
                [0],
                color=COLORS["grey"],
                marker="s",
                markerfacecolor="white",
                linestyle="none",
                label="Raw response map",
            ),
            Line2D(
                [0],
                [0],
                color=COLORS["black"],
                marker="o",
                linestyle="none",
                label="Row-centred map",
            ),
        ],
        loc="lower right",
    )
    clean_axis(axis)


def make_figure_4(
    output: Path,
    reliability: pd.DataFrame,
    cross_panel: pd.DataFrame,
    geometry: pd.DataFrame,
    bootstrap: pd.DataFrame,
    same_splits: pd.DataFrame,
    anastassiadis_splits: pd.DataFrame,
    panel_transfer: pd.DataFrame,
    anastassiadis_map_concordance: pd.DataFrame,
) -> tuple[Path, Path]:
    full, _matched = primary_experimental_splits(reliability, same_splits)
    figure = new_figure(6.3, 5.7)
    axes = figure.subplots(2, 2, squeeze=False)
    plot_experimental_map_reproducibility(
        axes[0, 0], full, anastassiadis_splits
    )
    plot_experimental_panel_transfer(axes[0, 1], panel_transfer)
    plot_same_support_geometry(axes[1, 0], geometry, bootstrap)
    plot_cross_assay_map_concordance(
        axes[1, 1], cross_panel, anastassiadis_map_concordance
    )
    figure.legend(
        handles=[
            Line2D([0], [0], color=COLORS["DAVIS"], marker="o", label="DAVIS"),
            Line2D([0], [0], color=COLORS["PKIS2"], marker="o", label="PKIS2"),
            Line2D(
                [0], [0], color=COLORS["HotSpot"], marker="o", label="HotSpot"
            ),
        ],
        loc="outside upper center",
        ncol=3,
    )
    for axis, label in zip(axes.flat, "abcd", strict=True):
        panel_label(axis, label)
    return save_figure(figure, output, FIGURE_STEMS[3])


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def relative_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PACKAGE.resolve()))
    except ValueError:
        return str(path.resolve())


def metadata_payload(output_paths: Iterable[Path]) -> dict[str, Any]:
    captions = {
        FIGURE_STEMS[0]: (
            "Shared and residual spectral structure of the two public Vina panels. "
            "(a) Target loadings on the leading score-surface principal component, "
            "oriented toward the positive uniform vector; dotted lines mark uniform "
            "loadings and the inset reports cosine similarity to that vector. "
            "(b) Correlation-spectrum variance fractions before and after two-way "
            "centering; the algebraically zero residual mode is omitted from the "
            "logarithmic display. (c) Hierarchically ordered residual target--target "
            "correlation map for Docking-44. (d) The corresponding residual map for "
            "DOCKSTRING-58. (e) Observed "
            "residual participation-ratio dimension, normalized by its P-1 maximum, "
            "against additive-Gaussian, column-permutation, row-norm-preserving, "
            "and ten-bin molecular-weight-conditioned raw-score nulls on the same "
            "deterministic n=12,000 ligand support. Null points "
            "are medians and bars are 2.5--97.5% simulation ranges."
        ),
        FIGURE_STEMS[1]: (
            "Chemical-domain dependence of residual Vina target-correlation maps. "
            "(a) Diamonds show the observed edge-rank Spearman correlation between "
            "low- and high-molecular-weight quartile maps (0.205 for Docking-44 "
            "and 0.351 for DOCKSTRING-58). Open circles and bars "
            "show medians and 2.5--97.5% ranges across 200 whole-chemical-group-"
            "disjoint splits for the full support after MW matching and separately "
            "within each MW quartile; the global matched medians were 0.992 and "
            "0.994. Chemical groups are inherited source cluster labels "
            "for Docking-44 and Bemis--Murcko scaffolds, with acyclic compounds as "
            "singletons, for DOCKSTRING-58. The small DOCKSTRING ticks show the "
            "observed residual contrast in six seeded 15,000-ligand supports; "
            "Docking-44 uses all 12,651 ligands. These repeated-split composition "
            "ranges and seeded-support values are not confidence intervals. "
            "(b) Residual-map agreement across "
            "10--40% low/high tail definitions; dotted lines and bands are means "
            "and 2.5--97.5% ranges across 100 equal-size random-row-disjoint control "
            "supports; unlike panel a, chemical groups may recur across their halves. "
            "The quartile definition differs from the 25% sweep point by at most one "
            "boundary ligand because the producers use different tie rules. "
            "(c) Pairwise map dissimilarity among ten equal-size MW bins versus "
            "their median-MW separation. Marker size encodes bin-index separation; "
            "the fits are descriptive because the 45 pairs reuse ten maps. "
            "(d) Residual low/high threshold-tail map agreement along seven ligand-"
            "descriptor axes. Filled points are observed contrasts; open points "
            "and bars are medians and 2.5--97.5% ranges from 200 matched-tail-size "
            "random-row-disjoint controls. Every observed contrast lies below its "
            "control range, with the largest changes along the correlated molecular-"
            "size axes. Inclusive thresholds retain ties, so discrete descriptors "
            "can have expanded tails. All descriptor comparisons were selected post "
            "hoc and are descriptive; these panels diagnose support dependence but "
            "do not identify a causal molecular property or establish biological "
            "target relationships."
        ),
        FIGURE_STEMS[2]: (
            "Recovery and compression of source-library Vina maps. (a) Global "
            "edge-rank agreement between maps estimated from 200 or 500 random "
            "calibration ligands and their row-disjoint complement maps. Points are "
            "means and bars are 2.5--97.5% ranges over 100 supports. (b) Median "
            "omitted-target variance-weighted R2 for score and row-centred residual "
            "surfaces when k=4--12 selected raw target scores are observed. Lines "
            "join medians and bars span five chemical-group-disjoint folds. "
            "(c) Pilot-medoid coverage expressed as the fraction "
            "of the random-to-full-map nearest-distance gap remaining. These are "
            "Vina score-compression diagnostics, not affinity predictions or a "
            "biological target optimum. (d) At k=8, multivariate Ridge residual "
            "reconstruction after raw-score prediction, direct residual-outcome "
            "Ridge using the same selected raw predictors, and a selected-target "
            "PC1 baseline. Points are fold medians and bars span five folds; the "
            "direct fit is an artifact check rather than a deployment policy."
        ),
        FIGURE_STEMS[3]: (
            "Experimental-map reproducibility, assay-value-blind panel transfer, "
            "and the docking--experiment boundary on 21 fixed kinase targets. "
            "(a) Two-way-centred experimental edge-map agreement across 2,000 "
            "disjoint halves of DAVIS, PKIS2 and the Anastassiadis HotSpot panel. "
            "DAVIS and PKIS2 keep Butina clusters within a half; HotSpot uses ligand "
            "rows. (b) Signed co-response coverage-loss contrasts over k. Solid "
            "circles are the conservative best-raw minus worst-residual contrast over "
            "every numerically optimal Vina panel; positive values favour the "
            "residual selector. Dotted crosses compare the sequence-identity panel "
            "with the lexicographic residual panel. Vina panels were selected without "
            "assay values after excluding all plausible evaluation-compound "
            "connectivities from the calibration support. "
            "(c) Score- and residual-map agreement between Vina "
            "and experiment on 56 exact DAVIS and 154 exact PKIS2 Standard InChIKey "
            "matches, compared with de-leaked broad-support Vina maps; bars are "
            "2.5--97.5% ranges across 2,000 Butina-cluster bootstraps. (d) Paired "
            "raw and row-centred edge-map agreement across DAVIS--PKIS2, DAVIS--"
            "HotSpot and PKIS2--HotSpot assay pairs. All comparisons are conditional "
            "on the fixed target panel; cross-panel experimental agreement is not "
            "an attenuation-corrected noise ceiling."
        ),
    }
    outputs = {
        relative_path(path): {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in output_paths
    }
    inputs = {
        key: {
            "path": relative_path(path),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for key, path in INPUT_PATHS.items()
    }
    return {
        "schema_version": "1.0",
        "producer": "analysis/make_public_manuscript_figures.py",
        "scope": (
            "Public-only main figures; all inferential statements are conditional "
            "on the fixed target panels and stated ligand supports."
        ),
        "uncertainty_contract": (
            "Error bars labelled as ranges are empirical 2.5--97.5% resampling or "
            "simulation ranges, not target-superpopulation confidence intervals."
        ),
        "captions": captions,
        "inputs": inputs,
        "outputs": outputs,
    }


def load_figure_inputs() -> dict[str, Any]:
    missing = [relative_path(path) for path in INPUT_PATHS.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing public figure inputs: {missing}")
    return {
        "core": load_json(INPUT_PATHS["public_core"]),
        "nulls": load_json(INPUT_PATHS["residual_nulls"]),
        "mw_sweep": pd.read_csv(INPUT_PATHS["mw_sweep"]),
        "mw_decile_pairs": pd.read_csv(INPUT_PATHS["mw_decile_pairs"]),
        "mw_decile_trends": pd.read_csv(INPUT_PATHS["mw_decile_trends"]),
        "chemical_domain_boundary": pd.read_csv(
            INPUT_PATHS["chemical_domain_boundary"]
        ),
        "chemical_domain_global_controls": pd.read_csv(
            INPUT_PATHS["chemical_domain_global_controls"]
        ),
        "chemical_domain_within_controls": pd.read_csv(
            INPUT_PATHS["chemical_domain_within_controls"]
        ),
        "chemical_domain_support_seeds": pd.read_csv(
            INPUT_PATHS["chemical_domain_support_seeds"]
        ),
        "descriptor_domain_extremes": pd.read_csv(
            INPUT_PATHS["descriptor_domain_extremes"]
        ),
        "descriptor_domain_random_controls": pd.read_csv(
            INPUT_PATHS["descriptor_domain_random_controls"]
        ),
        "recovery_edges": pd.read_csv(INPUT_PATHS["recovery_edges"]),
        "recovery_decisions": pd.read_csv(INPUT_PATHS["recovery_decisions"]),
        "panel_selector_metrics": pd.read_csv(
            INPUT_PATHS["panel_selector_metrics"]
        ),
        "panel_selector_direct": pd.read_csv(
            INPUT_PATHS["panel_selector_direct"]
        ),
        "experimental_cross_panel": pd.read_csv(INPUT_PATHS["experimental_cross_panel"]),
        "experimental_reliability": pd.read_csv(INPUT_PATHS["experimental_reliability"]),
        "same_support_geometry": pd.read_csv(INPUT_PATHS["same_support_geometry"]),
        "same_support_bootstrap": pd.read_csv(INPUT_PATHS["same_support_bootstrap"]),
        "same_support_splits": pd.read_csv(INPUT_PATHS["same_support_splits"]),
        "anastassiadis_splits": pd.read_csv(INPUT_PATHS["anastassiadis_splits"]),
        "experimental_panel_transfer": pd.read_csv(
            INPUT_PATHS["experimental_panel_transfer"]
        ),
        "anastassiadis_map_concordance": pd.read_csv(
            INPUT_PATHS["anastassiadis_map_concordance"]
        ),
    }


def make_all_figures(output: Path = DEFAULT_OUTPUT) -> list[Path]:
    configure_style()
    data = load_figure_inputs()
    created: list[Path] = []
    created.extend(make_figure_1(output, data["core"], data["nulls"]))
    created.extend(
        make_figure_2(
            output,
            data["mw_sweep"],
            data["mw_decile_pairs"],
            data["mw_decile_trends"],
            data["chemical_domain_boundary"],
            data["chemical_domain_global_controls"],
            data["chemical_domain_within_controls"],
            data["chemical_domain_support_seeds"],
            data["descriptor_domain_extremes"],
            data["descriptor_domain_random_controls"],
        )
    )
    created.extend(
        make_figure_3(
            output,
            data["recovery_edges"],
            data["recovery_decisions"],
            data["panel_selector_metrics"],
            data["panel_selector_direct"],
        )
    )
    created.extend(
        make_figure_4(
            output,
            data["experimental_reliability"],
            data["experimental_cross_panel"],
            data["same_support_geometry"],
            data["same_support_bootstrap"],
            data["same_support_splits"],
            data["anastassiadis_splits"],
            data["experimental_panel_transfer"],
            data["anastassiadis_map_concordance"],
        )
    )
    metadata = metadata_payload(created)
    metadata_path = output / "figure_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    created.append(metadata_path)
    return created


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    paths = make_all_figures(arguments.output)
    for path in paths:
        print(relative_path(path))


if __name__ == "__main__":
    main()
