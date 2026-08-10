#!/usr/bin/env python3
"""Plot the falsification boundary of the spectral docking-QC criterion.

The figure is intentionally generated from the frozen replicate-level output of
``spectral_qc_criterion_simulation.py``.  It does not rerun the simulation or
select thresholds from outcomes.  The three panels show:

1. spectral expansion versus simulated target-preference accuracy;
2. the two plotted axes of the predeclared residual-coherence gate; and
3. the paired accuracy loss after an exactly spectrum-preserving target-label
   permutation.

The plotted medians and central 95% simulation intervals are also written in a
machine-readable CSV.  A JSON manifest records inputs, thresholds, styling, and
output checksums.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


PACKAGE = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = PACKAGE / "results" / "spectral_qc_criterion_simulation"
DEFAULT_SOURCE = DEFAULT_RESULTS / "replicate_metrics.csv"
DEFAULT_RULES = DEFAULT_RESULTS / "predeclared_rules.json"
DEFAULT_SUMMARY = DEFAULT_RESULTS / "falsification_figure_summary.csv"
DEFAULT_MANIFEST = DEFAULT_RESULTS / "falsification_figure_manifest.json"
DEFAULT_STEM = PACKAGE / "figures" / "spectral_qc_criterion_falsification"

WIDTH = 7.15  # approximately 182 mm, a full two-column journal figure
HEIGHT = 3.45
RELEASE_TIMESTAMP = datetime(2026, 8, 2, tzinfo=timezone.utc)
INK = "#1A1A1A"
GRID = "#DCE1E5"
GATE_FILL = "#D9F0E8"

# Okabe--Ito colors plus grey; marker redundancy preserves category identity in
# greyscale and for readers with color-vision deficiencies.
REGIME_STYLE: dict[str, dict[str, Any]] = {
    "good_correlated_interactions": {
        "label": "Informative, correlated",
        "short_label": "informative correlated",
        "color": "#0072B2",
        "marker": "o",
        "size": 6.8,
        "zorder": 5,
    },
    "generic_ligand_effect_independent_noise": {
        "label": "Independent artifact",
        "short_label": "independent artifact",
        "color": "#E69F00",
        "marker": "s",
        "size": 7.5,
        "zorder": 3,
    },
    "target_correlated_artifact": {
        "label": "Correlated artifact",
        "short_label": "correlated artifact",
        "color": "#D55E00",
        "marker": "D",
        "size": 5.8,
        "zorder": 6,
    },
    "shuffled_box_labels": {
        "label": "Shuffled target labels",
        "short_label": "shuffled labels",
        "color": "#CC79A7",
        "marker": "X",
        "size": 6.6,
        "zorder": 8,
    },
    "seed_unstable_noise": {
        "label": "Seed-unstable noise",
        "short_label": "seed-unstable",
        "color": "#6F777D",
        "marker": "v",
        "size": 6.5,
        "zorder": 4,
    },
    "good_high_dimensional_interactions": {
        "label": "Informative, high-dimensional",
        "short_label": "informative high-D",
        "color": "#009E73",
        "marker": "^",
        "size": 6.3,
        "zorder": 7,
    },
}

REGIME_ORDER = tuple(REGIME_STYLE)
OBSERVABLE_INVARIANCE_METRICS = (
    "raw_pr",
    "residual_pr",
    "expansion_log2",
    "raw_seed_spearman",
    "residual_seed_spearman",
    "residual_pr_null_median",
    "residual_pr_null_ratio",
    "residual_pr_null_lower_tail_p",
    "coherence_composite_score",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _portable_path(path: Path) -> str:
    """Prefer repository-relative paths while supporting explicit external inputs."""
    resolved = path.resolve()
    try:
        return resolved.relative_to(PACKAGE.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def _load_rules(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict) or not isinstance(value.get("primary"), dict):
        raise ValueError(f"Malformed predeclared rule file: {path}")
    required = {
        "minimum_median_pairwise_residual_seed_spearman",
        "maximum_residual_pr_over_row_norm_null_median",
        "maximum_empirical_lower_tail_p",
    }
    missing = required.difference(value["primary"])
    if missing:
        raise ValueError(f"Missing primary-rule fields in {path}: {sorted(missing)}")
    return value


def _validate_replicates(frame: pd.DataFrame) -> None:
    required = {
        "replicate",
        "regime",
        "truth_informative",
        "expansion_log2",
        "residual_seed_spearman",
        "residual_pr_null_ratio",
        "residual_pr_null_lower_tail_p",
        "preference_accuracy",
        *OBSERVABLE_INVARIANCE_METRICS,
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing replicate columns: {sorted(missing)}")
    observed_regimes = set(frame["regime"])
    expected_regimes = set(REGIME_ORDER)
    if observed_regimes != expected_regimes:
        raise ValueError(
            "Unexpected simulation regimes: "
            f"missing={sorted(expected_regimes - observed_regimes)}, "
            f"extra={sorted(observed_regimes - expected_regimes)}"
        )
    counts = frame.groupby("regime", sort=False)["replicate"].nunique()
    if counts.nunique() != 1 or int(counts.iloc[0]) < 2:
        raise ValueError(f"Regimes are not balanced paired replicates: {counts.to_dict()}")
    if frame.duplicated(["replicate", "regime"]).any():
        raise ValueError("Each replicate/regime combination must be unique")
    numeric = frame[
        [
            "expansion_log2",
            "residual_seed_spearman",
            "residual_pr_null_ratio",
            "residual_pr_null_lower_tail_p",
            "preference_accuracy",
            *OBSERVABLE_INVARIANCE_METRICS,
        ]
    ].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError("Replicate metrics contain non-finite plotted values")


def _quantiles(values: Iterable[float]) -> tuple[float, float, float]:
    array = np.asarray(tuple(values), dtype=float)
    return tuple(float(value) for value in np.quantile(array, [0.025, 0.5, 0.975]))


def _paired_frames(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    indexed = frame.set_index(["replicate", "regime"]).sort_index()
    good = indexed.xs("good_correlated_interactions", level="regime")
    shuffled = indexed.xs("shuffled_box_labels", level="regime")
    if not good.index.equals(shuffled.index):
        raise ValueError("Good and shuffled regimes do not contain identical replicates")
    return good, shuffled


def build_plotted_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Return the exact medians and central 95% intervals shown in the figure."""
    records: list[dict[str, Any]] = []
    for regime in REGIME_ORDER:
        subset = frame.loc[frame["regime"].eq(regime)]
        for panel, x_metric, y_metric in (
            ("a", "expansion_log2", "preference_accuracy"),
            ("b", "residual_seed_spearman", "residual_pr_null_ratio"),
        ):
            x_low, x_median, x_high = _quantiles(subset[x_metric])
            y_low, y_median, y_high = _quantiles(subset[y_metric])
            p_low, p_median, p_high = _quantiles(
                subset["residual_pr_null_lower_tail_p"]
            )
            records.append(
                {
                    "panel": panel,
                    "series": regime,
                    "display_label": REGIME_STYLE[regime]["label"],
                    "n_paired_replicates": int(len(subset)),
                    "x_metric": x_metric,
                    "x_q025": x_low,
                    "x_median": x_median,
                    "x_q975": x_high,
                    "y_metric": y_metric,
                    "y_q025": y_low,
                    "y_median": y_median,
                    "y_q975": y_high,
                    "null_p_q025": p_low,
                    "null_p_median": p_median,
                    "null_p_q975": p_high,
                    "paired_difference_metric": "",
                    "paired_difference_q025": np.nan,
                    "paired_difference_median": np.nan,
                    "paired_difference_q975": np.nan,
                    "maximum_absolute_observable_difference": np.nan,
                }
            )

    good, shuffled = _paired_frames(frame)
    difference = shuffled["preference_accuracy"] - good["preference_accuracy"]
    x_low, x_median, x_high = _quantiles(good["preference_accuracy"])
    y_low, y_median, y_high = _quantiles(shuffled["preference_accuracy"])
    d_low, d_median, d_high = _quantiles(difference)
    maximum_observable_difference = max(
        float(np.max(np.abs(good[metric] - shuffled[metric])))
        for metric in OBSERVABLE_INVARIANCE_METRICS
    )
    records.append(
        {
            "panel": "c",
            "series": "shuffled_minus_correct_target_labels",
            "display_label": "Paired target-label permutation",
            "n_paired_replicates": int(len(good)),
            "x_metric": "good_preference_accuracy",
            "x_q025": x_low,
            "x_median": x_median,
            "x_q975": x_high,
            "y_metric": "shuffled_preference_accuracy",
            "y_q025": y_low,
            "y_median": y_median,
            "y_q975": y_high,
            "null_p_q025": np.nan,
            "null_p_median": np.nan,
            "null_p_q975": np.nan,
            "paired_difference_metric": "shuffled_minus_good_preference_accuracy",
            "paired_difference_q025": d_low,
            "paired_difference_median": d_median,
            "paired_difference_q975": d_high,
            "maximum_absolute_observable_difference": maximum_observable_difference,
        }
    )
    return pd.DataFrame(records)


def _use_style() -> None:
    matplotlib.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": INK,
            "axes.labelcolor": INK,
            "axes.titlecolor": INK,
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7.2,
            "axes.labelsize": 7.5,
            "axes.titlesize": 7.7,
            "xtick.labelsize": 6.7,
            "ytick.labelsize": 6.7,
            "legend.fontsize": 6.6,
            "axes.linewidth": 0.7,
            "lines.linewidth": 1.0,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
            "savefig.transparent": False,
        }
    )


def _clean(ax: plt.Axes) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=2.5, width=0.6)
    ax.grid(color=GRID, linewidth=0.5, alpha=0.8, zorder=0)


def _panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.17,
        1.08,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        fontweight="bold",
        color=INK,
    )


def _summary_record(summary: pd.DataFrame, panel: str, series: str) -> pd.Series:
    selected = summary.loc[
        summary["panel"].eq(panel) & summary["series"].eq(series)
    ]
    if len(selected) != 1:
        raise ValueError(f"Expected one summary row for panel {panel}, series {series}")
    return selected.iloc[0]


def _cross_interval(
    ax: plt.Axes,
    record: pd.Series,
    style: dict[str, Any],
) -> None:
    x = float(record["x_median"])
    y = float(record["y_median"])
    color = str(style["color"])
    ax.plot(
        [record["x_q025"], record["x_q975"]],
        [y, y],
        color=color,
        alpha=0.56,
        linewidth=1.0,
        zorder=int(style["zorder"]) - 1,
    )
    ax.plot(
        [x, x],
        [record["y_q025"], record["y_q975"]],
        color=color,
        alpha=0.56,
        linewidth=1.0,
        zorder=int(style["zorder"]) - 1,
    )
    # A thin white halo keeps nearby marker outlines legible without moving any
    # estimate.  The X marker deliberately has no halo: in panel b it is exactly
    # superposed on the blue circle, so both symbols must remain visible.
    if style["marker"] != "X":
        ax.plot(
            x,
            y,
            marker=style["marker"],
            markersize=float(style["size"]) + 1.6,
            markerfacecolor="white",
            markeredgecolor="white",
            markeredgewidth=1.6,
            linestyle="none",
            zorder=int(style["zorder"]),
        )
    ax.plot(
        x,
        y,
        marker=style["marker"],
        markersize=float(style["size"]),
        markerfacecolor=color,
        markeredgecolor="white",
        markeredgewidth=0.65,
        linestyle="none",
        zorder=int(style["zorder"]) + 1,
    )


def make_figure(
    frame: pd.DataFrame,
    summary: pd.DataFrame,
    rules: dict[str, Any],
) -> plt.Figure:
    _use_style()
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, HEIGHT))
    ax_a, ax_b, ax_c = axes

    # Panel a: expansion is neither sufficient nor necessary for simulated truth.
    for regime in REGIME_ORDER:
        _cross_interval(
            ax_a,
            _summary_record(summary, "a", regime),
            REGIME_STYLE[regime],
        )
    expansion_threshold = float(
        rules["expansion_only_comparator"]["minimum_expansion_log2"]
    )
    ax_a.axvline(
        expansion_threshold,
        color=INK,
        linestyle=(0, (3, 2)),
        linewidth=0.8,
        zorder=1,
    )
    ax_a.axhline(
        0.5,
        color="#7A8288",
        linestyle=(0, (2, 2)),
        linewidth=0.8,
        zorder=1,
    )
    ax_a.text(
        expansion_threshold + 0.04,
        0.985,
        "doubling rule",
        transform=ax_a.get_xaxis_transform(),
        ha="left",
        va="top",
        fontsize=6.2,
        color="#4F565B",
    )
    ax_a.text(
        0.98,
        0.505,
        "chance",
        transform=ax_a.get_yaxis_transform(),
        ha="right",
        va="bottom",
        fontsize=6.2,
        color="#4F565B",
    )
    ax_a.set_xlim(0.75, 4.27)
    ax_a.set_ylim(0.38, 0.985)
    ax_a.set_xlabel(r"Expansion  $E=\log_2(\mathrm{PR}_{res}/\mathrm{PR}_{raw})$")
    ax_a.set_ylabel("Simulated target-preference accuracy")
    ax_a.set_title("Expansion does not identify accuracy", loc="left")
    _clean(ax_a)

    # Panel b: the predeclared matrix-coherence gate, explicitly not accuracy.
    primary = rules["primary"]
    rho_threshold = float(
        primary["minimum_median_pairwise_residual_seed_spearman"]
    )
    ratio_threshold = float(
        primary["maximum_residual_pr_over_row_norm_null_median"]
    )
    p_threshold = float(primary["maximum_empirical_lower_tail_p"])
    ax_b.axvspan(rho_threshold, 1.03, color=GATE_FILL, alpha=0.75, zorder=-3)
    ax_b.axhspan(ratio_threshold, 1.10, color="white", alpha=0.92, zorder=-2)
    ax_b.axvline(
        rho_threshold,
        color=INK,
        linestyle=(0, (3, 2)),
        linewidth=0.8,
        zorder=1,
    )
    ax_b.axhline(
        ratio_threshold,
        color=INK,
        linestyle=(0, (3, 2)),
        linewidth=0.8,
        zorder=1,
    )
    for regime in REGIME_ORDER:
        _cross_interval(
            ax_b,
            _summary_record(summary, "b", regime),
            REGIME_STYLE[regime],
        )
    ax_b.text(
        0.98,
        0.035,
        "plotted gate region\n"
        + rf"(full gate also null $p\leq${p_threshold:.2f})",
        transform=ax_b.transAxes,
        ha="right",
        va="bottom",
        fontsize=6.2,
        color="#276B58",
    )
    ax_b.set_xlim(-0.055, 1.025)
    ax_b.set_ylim(0.14, 1.075)
    ax_b.set_xlabel(r"Residual seed reproducibility  $\rho$")
    ax_b.set_ylabel("Residual PR / row-norm-null PR")
    ax_b.set_title("Coherence gate accepts artifacts", loc="left")
    _clean(ax_b)

    # Panel c: exact observable invariance, paired with a large accuracy loss.
    good, shuffled = _paired_frames(frame)
    ax_c.scatter(
        good["preference_accuracy"],
        shuffled["preference_accuracy"],
        s=13,
        marker="o",
        facecolor=REGIME_STYLE["shuffled_box_labels"]["color"],
        edgecolor="white",
        linewidth=0.28,
        alpha=0.42,
        rasterized=True,
        zorder=2,
    )
    panel_c = _summary_record(
        summary, "c", "shuffled_minus_correct_target_labels"
    )
    ax_c.plot(
        [panel_c["x_q025"], panel_c["x_q975"]],
        [panel_c["y_median"], panel_c["y_median"]],
        color=INK,
        linewidth=1.2,
        zorder=5,
    )
    ax_c.plot(
        [panel_c["x_median"], panel_c["x_median"]],
        [panel_c["y_q025"], panel_c["y_q975"]],
        color=INK,
        linewidth=1.2,
        zorder=5,
    )
    ax_c.plot(
        panel_c["x_median"],
        panel_c["y_median"],
        marker="D",
        markersize=5.2,
        color=INK,
        markeredgecolor="white",
        markeredgewidth=0.55,
        linestyle="none",
        zorder=6,
    )
    ax_c.plot([0.38, 1.0], [0.38, 1.0], color="#A8ADB1", lw=0.8, ls=(0, (2, 2)))
    maximum_difference = float(panel_c["maximum_absolute_observable_difference"])
    ax_c.text(
        0.045,
        0.95,
        "same observable metrics\n"
        + rf"max $|\Delta|={maximum_difference:.1e}$",
        transform=ax_c.transAxes,
        ha="left",
        va="top",
        fontsize=6.4,
        color="#5A306A",
    )
    ax_c.text(
        0.045,
        0.05,
        rf"median paired $\Delta$ = {panel_c['paired_difference_median']:.3f}"
        + "\n"
        + rf"95%: {panel_c['paired_difference_q025']:.3f} to "
        + rf"{panel_c['paired_difference_q975']:.3f}",
        transform=ax_c.transAxes,
        ha="left",
        va="bottom",
        fontsize=6.4,
        color=INK,
    )
    ax_c.set_xlim(0.38, 1.0)
    ax_c.set_ylim(0.38, 1.0)
    ax_c.set_aspect("equal", adjustable="box")
    ax_c.set_xlabel("Accuracy with correct target labels")
    ax_c.set_ylabel("Accuracy after target-label shuffle")
    ax_c.set_title("Spectra cannot detect label mismatch", loc="left")
    _clean(ax_c)

    for axis, label in zip(axes, "abc"):
        _panel_label(axis, label)

    legend_handles = [
        Line2D(
            [],
            [],
            marker=style["marker"],
            markersize=5.4,
            markerfacecolor=style["color"],
            markeredgecolor="white",
            markeredgewidth=0.5,
            color="none",
            label=style["label"],
        )
        for style in REGIME_STYLE.values()
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.006),
        ncol=3,
        frameon=False,
        columnspacing=1.4,
        handletextpad=0.45,
    )
    fig.text(
        0.5,
        0.985,
        "Observable spectral coherence is neither necessary nor sufficient for simulated docking accuracy",
        ha="center",
        va="top",
        fontsize=8.2,
        fontweight="bold",
        color=INK,
    )
    fig.text(
        0.5,
        0.145,
        "Markers are medians; bars are central 95% intervals across paired simulation replicates.",
        ha="center",
        va="bottom",
        fontsize=6.3,
        color="#4F565B",
    )
    fig.subplots_adjust(left=0.067, right=0.992, bottom=0.270, top=0.86, wspace=0.39)
    return fig


def _audit_figure(fig: plt.Figure) -> None:
    if not np.allclose(fig.get_size_inches(), [WIDTH, HEIGHT], atol=1e-8):
        raise AssertionError(f"Unexpected figure size: {fig.get_size_inches()}")
    for artist in fig.findobj(match=matplotlib.text.Text):
        if artist.get_text().strip() and artist.get_fontsize() < 6.0:
            raise AssertionError(f"Text below 6 pt: {artist.get_text()!r}")
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    canvas = fig.bbox
    clipped: list[str] = []
    for artist in fig.findobj(match=matplotlib.text.Text):
        if not artist.get_visible() or not artist.get_text().strip():
            continue
        box = artist.get_window_extent(renderer=renderer)
        if (
            box.x0 < canvas.x0 - 2
            or box.y0 < canvas.y0 - 2
            or box.x1 > canvas.x1 + 2
            or box.y1 > canvas.y1 + 2
        ):
            clipped.append(artist.get_text().replace("\n", " / "))
    if clipped:
        raise AssertionError("Figure text is clipped: " + "; ".join(clipped[:8]))


def _write_outputs(
    fig: plt.Figure,
    stem: Path,
    summary: pd.DataFrame,
    summary_path: Path,
    manifest_path: Path,
    source_path: Path,
    rules_path: Path,
    rules: dict[str, Any],
) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False, float_format="%.12g")
    pdf_path = stem.with_suffix(".pdf")
    png_path = stem.with_suffix(".png")
    fig.savefig(
        pdf_path,
        metadata={
            "Title": "Spectral-QC criterion falsification simulation",
            "Author": "Adeliya Leleytner; Victor Safronov; Maxim Fedorov",
            "Creator": Path(__file__).name,
            "CreationDate": RELEASE_TIMESTAMP,
            "ModDate": RELEASE_TIMESTAMP,
        },
    )
    fig.savefig(
        png_path,
        dpi=400,
        metadata={
            "Title": "Spectral-QC criterion falsification simulation",
            "Author": "Adeliya Leleytner; Victor Safronov; Maxim Fedorov",
            "Software": Path(__file__).name,
        },
    )
    if pdf_path.stat().st_size < 10_000 or png_path.stat().st_size < 40_000:
        raise AssertionError(
            "Figure output appears incomplete: "
            f"{pdf_path.stat().st_size} PDF bytes; {png_path.stat().st_size} PNG bytes"
        )
    manifest = {
        "analysis": "spectral_qc_criterion_falsification_figure",
        "source": {"path": _portable_path(source_path), "sha256": _sha256(source_path)},
        "predeclared_rules": {
            "path": _portable_path(rules_path),
            "sha256": _sha256(rules_path),
            "primary": rules["primary"],
            "expansion_only_comparator": rules["expansion_only_comparator"],
        },
        "plotted_summary": {
            "path": _portable_path(summary_path),
            "sha256": _sha256(summary_path),
            "interval": "central 95% across paired simulation replicates",
        },
        "regime_order": list(REGIME_ORDER),
        "regime_style": REGIME_STYLE,
        "outputs": {
            "pdf": {"path": _portable_path(pdf_path), "sha256": _sha256(pdf_path)},
            "png": {"path": _portable_path(png_path), "sha256": _sha256(png_path)},
        },
        "runtime": {
            "python": platform.python_version(),
            "matplotlib": matplotlib.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "release_timestamp": RELEASE_TIMESTAMP.isoformat(),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--rules", type=Path, default=DEFAULT_RULES)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--figure-stem", type=Path, default=DEFAULT_STEM)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.source)
    _validate_replicates(frame)
    rules = _load_rules(args.rules)
    summary = build_plotted_summary(frame)
    panel_c = _summary_record(
        summary, "c", "shuffled_minus_correct_target_labels"
    )
    if float(panel_c["maximum_absolute_observable_difference"]) > 1e-12:
        raise AssertionError(
            "Good and shuffled regimes are not observably invariant to numerical tolerance"
        )
    fig = make_figure(frame, summary, rules)
    _audit_figure(fig)
    _write_outputs(
        fig,
        args.figure_stem,
        summary,
        args.summary,
        args.manifest,
        args.source,
        args.rules,
        rules,
    )
    plt.close(fig)
    print(
        f"Wrote {args.figure_stem.with_suffix('.pdf')}, "
        f"{args.figure_stem.with_suffix('.png')}, and {args.summary}"
    )


if __name__ == "__main__":
    main()
