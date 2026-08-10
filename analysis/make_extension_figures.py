#!/usr/bin/env python3
"""Generate figures for the dense operational and proteome-wide extensions.

The script is deliberately separate from :mod:`make_figures`: the extension
analyses are still being evaluated as scientific additions and should not alter
the manuscript's established figure build until that decision is made.

Inputs
------
``results/dense_davis_benchmark.json``
    Dense 59-ligand x 21-target DAVIS--DOCKSTRING benchmark.
``results/dense_pkis2_benchmark.json``
    Dense 154-ligand x 21-target PKIS2--DOCKSTRING benchmark.
``results/pbas_proteome/``
    PBAS summary, target-subsampling, and component-association outputs.

Outputs
-------
``figures/figS10_dense_operational_extensions.{pdf,png}``
``figures/figS11_pbas_boundary_audit.{pdf,png}``

Every plotted estimate is read from those frozen result files.  The code does
not recalculate a benchmark, choose a favourable sensitivity, or infer missing
confidence intervals.  In particular, outcome-conditioned binder--binder
analyses are visually marked as exploratory and no performance advantage is
claimed when an interval crosses zero.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:  # Support both direct execution and package-style import.
    from .make_figures import (
        BLUE,
        GREY,
        INK,
        ORANGE,
        PLUM,
        WIDTH,
        audit_fig,
        clean,
        panel_label,
        use_paper_style,
    )
except ImportError:
    from make_figures import (  # type: ignore
        BLUE,
        GREY,
        INK,
        ORANGE,
        PLUM,
        WIDTH,
        audit_fig,
        clean,
        panel_label,
        use_paper_style,
    )


PACKAGE = Path(__file__).resolve().parents[1]
RESULTS = PACKAGE / "results"
OUT = PACKAGE / "figures"
DAVIS_PATH = RESULTS / "dense_davis_benchmark.json"
PKIS2_PATH = RESULTS / "dense_pkis2_benchmark.json"
PBAS_DIR = RESULTS / "pbas_proteome"
PBAS_SUMMARY_PATH = PBAS_DIR / "summary.json"
PBAS_SUBSAMPLING_PATH = PBAS_DIR / "target_subsampling.csv"
PBAS_ASSOCIATIONS_PATH = PBAS_DIR / "component_associations.csv"
PBAS_EIGENVALUES_PATH = PBAS_DIR / "top_eigenvalues.csv"
PBAS_MISSINGNESS_PATH = PBAS_DIR / "target_missingness.csv"
RELEASE_TIMESTAMP = datetime(2026, 8, 1, tzinfo=timezone.utc)

CORE_REPRESENTATIONS = (
    "absolute_vina",
    "target_centered_unscaled",
    "column_standardized",
    "target_centered_residual_scaled",
    "two_way_residual",
)
BINARY_REPRESENTATIONS = (*CORE_REPRESENTATIONS, "docking_target_prior")
REPRESENTATION_LABELS = {
    "absolute_vina": "Absolute\nVina",
    "target_centered_unscaled": "Target-\ncentered",
    "column_standardized": "Column\n$z$ score",
    "target_centered_residual_scaled": "Centered /\nresidual SD",
    "two_way_residual": "Two-way\nresidual",
    "docking_target_prior": "Target\nprior",
}
BINARY_LABELS = {
    "absolute_vina": "Abs.\nVina",
    "target_centered_unscaled": "Target-\ncentered",
    "column_standardized": "Column\n$z$",
    "target_centered_residual_scaled": "Centered /\nresid. SD",
    "two_way_residual": "Two-way\nresid.",
    "docking_target_prior": "Target\nprior",
}


def _require_files(paths: Iterable[Path], retry_seconds: float = 10.0) -> None:
    """Require all analysis products, waiting once for concurrently built JSONs.

    A single short retry is useful when ``make`` launches the two dense
    benchmarks immediately before this script.  Failure after that retry is
    explicit and lists both the missing files and the commands that create them.
    """

    paths = tuple(paths)
    missing = [path for path in paths if not path.is_file()]
    if missing:
        print(
            "Extension figure inputs are not complete; retrying once in "
            f"{retry_seconds:g} seconds: "
            + ", ".join(str(path) for path in missing),
            flush=True,
        )
        time.sleep(retry_seconds)
        missing = [path for path in paths if not path.is_file()]
    if missing:
        formatted = "\n  - ".join(str(path) for path in missing)
        raise FileNotFoundError(
            "Required extension-analysis outputs are missing:\n  - "
            f"{formatted}\n"
            "Run analysis/dense_davis_benchmark.py, "
            "analysis/dense_pkis2_benchmark.py, and "
            "analysis/pbas_proteome_analysis.py before plotting."
        )


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except json.JSONDecodeError as error:
        raise ValueError(f"Malformed JSON input {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _finite(value: Any, context: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Expected a numeric value for {context}; observed {value!r}") from error
    if not np.isfinite(number):
        raise ValueError(f"Expected a finite value for {context}; observed {number}")
    return number


def _path(record: dict[str, Any], keys: Iterable[str], context: str) -> Any:
    value: Any = record
    walked: list[str] = []
    for key in keys:
        walked.append(key)
        if not isinstance(value, dict) or key not in value:
            raise KeyError(
                f"Missing required field {'.'.join(walked)!r} while reading {context}"
            )
        value = value[key]
    return value


def _interval(record: dict[str, Any], method: str, context: str) -> tuple[float, float]:
    uncertainty = _path(record, ("uncertainty", method, "interval_95"), context)
    if not isinstance(uncertainty, list) or len(uncertainty) != 2:
        raise ValueError(f"Malformed 95% interval for {context}: {uncertainty!r}")
    low = _finite(uncertainty[0], f"{context} lower interval")
    high = _finite(uncertainty[1], f"{context} upper interval")
    if low > high:
        raise ValueError(f"Reversed 95% interval for {context}: {low}, {high}")
    return low, high


def _binary_interval(
    record: dict[str, Any], dataset: str, method: str = "murcko_cluster_bootstrap"
) -> tuple[float, float]:
    if dataset == "DAVIS":
        return _interval(record, method, "DAVIS binary AUC")
    uncertainty = _path(
        record,
        ("roc_auc_uncertainty", method, "interval_95"),
        "PKIS2 binary AUC",
    )
    if not isinstance(uncertainty, list) or len(uncertainty) != 2:
        raise ValueError(f"Malformed PKIS2 ROC-AUC interval: {uncertainty!r}")
    return (
        _finite(uncertainty[0], "PKIS2 binary AUC lower interval"),
        _finite(uncertainty[1], "PKIS2 binary AUC upper interval"),
    )


def _validate_dense_reports(davis: dict[str, Any], pkis2: dict[str, Any]) -> None:
    for label, report, expected_ligands in (
        ("DAVIS", davis, 59),
        ("PKIS2", pkis2, 154),
    ):
        if not report.get("release_eligible_identity_scan", False):
            raise ValueError(
                f"{label} JSON was made with a smoke-test identity scan; "
                "publication figures require the full recomputed identity scan"
            )
        ligands = _path(
            report,
            ("source_support", "primary_full_key_ligands"),
            f"{label} support",
        )
        targets = _path(
            report, ("source_support", "shared_targets"), f"{label} support"
        )
        if int(ligands) != expected_ligands or int(targets) != 21:
            raise ValueError(
                f"Unexpected {label} dense support: {ligands} ligands x {targets} targets"
            )


def _save(fig: plt.Figure, stem: str, dpi: int = 400) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pdf = OUT / f"{stem}.pdf"
    png = OUT / f"{stem}.png"
    fig.savefig(
        pdf,
        metadata={
            "Title": stem,
            "Author": "Adeliya Leleytner; Victor Safronov; Maxim Fedorov",
            "Creator": "make_extension_figures.py",
            "CreationDate": RELEASE_TIMESTAMP,
            "ModDate": RELEASE_TIMESTAMP,
        },
    )
    fig.savefig(png, dpi=dpi)
    if pdf.stat().st_size < 5_000 or png.stat().st_size < 20_000:
        raise AssertionError(
            f"Figure output appears incomplete: {pdf.stat().st_size} PDF bytes, "
            f"{png.stat().st_size} PNG bytes"
        )
    width, height = fig.get_size_inches()
    print(f"wrote {pdf} ({width:.3f} x {height:.3f} in)")


def _audit_text_within_canvas(fig: plt.Figure, tolerance_pixels: float = 2.0) -> None:
    """Fail when a visible text artist is clipped by the figure canvas."""

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    figure_box = fig.bbox
    failures: list[str] = []
    for artist in fig.findobj(match=matplotlib.text.Text):
        if not artist.get_visible() or not artist.get_text().strip():
            continue
        box = artist.get_window_extent(renderer=renderer)
        if (
            box.x0 < figure_box.x0 - tolerance_pixels
            or box.y0 < figure_box.y0 - tolerance_pixels
            or box.x1 > figure_box.x1 + tolerance_pixels
            or box.y1 > figure_box.y1 + tolerance_pixels
        ):
            failures.append(artist.get_text().replace("\n", " / "))
    if failures:
        raise AssertionError(
            "text extends beyond the figure canvas: " + "; ".join(failures[:8])
        )


def _errorbar(
    ax: plt.Axes,
    x: float,
    y: float,
    interval: tuple[float, float],
    *,
    color: str,
    marker: str,
    zorder: int = 3,
    markersize: float = 4.5,
) -> None:
    low, high = interval
    ax.errorbar(
        x,
        y,
        yerr=np.asarray([[y - low], [high - y]]),
        fmt=marker,
        color=color,
        markerfacecolor=color,
        markeredgecolor="white",
        markeredgewidth=0.45,
        markersize=markersize,
        elinewidth=0.75,
        capsize=1.8,
        capthick=0.7,
        zorder=zorder,
    )


def _horizontal_interval(
    ax: plt.Axes,
    estimate: float,
    y: float,
    interval: tuple[float, float],
    *,
    color: str,
    marker: str,
) -> None:
    low, high = interval
    ax.plot([low, high], [y, y], color=color, lw=1.35, solid_capstyle="round", zorder=2)
    ax.plot([low, high], [y, y], marker="|", color=color, lw=0, ms=5.5, zorder=2)
    ax.plot(
        estimate,
        y,
        marker=marker,
        ms=5.1,
        color=color,
        markeredgecolor="white",
        markeredgewidth=0.5,
        zorder=3,
    )


def _dense_primary_values(
    davis: dict[str, Any], pkis2: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    davis_values = []
    davis_binder = []
    pkis_values = []
    for representation in CORE_REPRESENTATIONS:
        d_record = _path(
            davis,
            (
                "primary",
                "representations",
                representation,
                "0.0",
                "all_informative",
            ),
            "DAVIS primary representation",
        )
        binder_record = _path(
            davis,
            (
                "primary",
                "representations",
                representation,
                "0.0",
                "both_uncensored",
            ),
            "DAVIS exploratory binder-binder representation",
        )
        p_record = _path(
            pkis2,
            (
                "primary",
                "representations",
                representation,
                "absolute_difference_gt_10",
            ),
            "PKIS2 primary representation",
        )
        davis_values.append(
            _finite(
                d_record["mean_per_ligand_pairwise_concordance"],
                f"DAVIS {representation} concordance",
            )
        )
        davis_binder.append(
            _finite(
                binder_record["mean_per_ligand_pairwise_concordance"],
                f"DAVIS {representation} binder concordance",
            )
        )
        pkis_values.append(
            _finite(
                p_record["mean_per_ligand_pairwise_concordance"],
                f"PKIS2 {representation} concordance",
            )
        )
    return (
        np.asarray(davis_values),
        np.asarray(pkis_values),
        np.asarray(davis_binder),
    )


def _dense_contrast_records(
    davis: dict[str, Any], pkis2: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    return {
        "DAVIS": _path(
            davis,
            (
                "primary",
                "paired_comparisons",
                "two_way_residual_minus_absolute_vina",
                "0.0",
                "all_informative",
            ),
            "DAVIS primary paired contrast",
        ),
        "PKIS2": _path(
            pkis2,
            (
                "primary",
                "paired_comparisons",
                "two_way_residual_minus_absolute_vina",
                "absolute_difference_gt_10",
            ),
            "PKIS2 primary paired contrast",
        ),
    }


def _dense_binary_records(
    davis: dict[str, Any], pkis2: dict[str, Any]
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    return (
        _path(
            davis,
            ("primary", "co_primary_hit_detection_auc"),
            "DAVIS hit-detection AUC",
        ),
        _path(
            pkis2,
            ("primary", "binary_activity_at_65_percent"),
            "PKIS2 binary activity AUC",
        ),
    )


def _calibration_contrast_values(
    davis: dict[str, Any], pkis2: dict[str, Any]
) -> dict[str, np.ndarray]:
    """Extract residual-minus-absolute contrasts across reference supports."""

    output: dict[str, np.ndarray] = {}

    davis_calibration = _path(
        davis,
        ("molecular_size_matched_reference_calibration",),
        "DAVIS molecular-size-matched calibration sensitivity",
    )
    if davis_calibration.get("status") != "complete" or not davis_calibration.get(
        "release_result_available", False
    ):
        raise ValueError(
            "DAVIS molecular-size-matched reference sensitivity is unavailable; "
            "a full identity scan is required"
        )
    davis_values = [
        _finite(
            _path(
                davis,
                (
                    "primary",
                    "paired_comparisons",
                    "two_way_residual_minus_absolute_vina",
                    "0.0",
                    "all_informative",
                    "plugin_mean_difference",
                ),
                "DAVIS full-reference calibration contrast",
            ),
            "DAVIS full-reference calibration contrast",
        )
    ]
    davis_neighbours = _path(
        davis_calibration,
        ("neighbour_count_sensitivity",),
        "DAVIS neighbour-count calibration sensitivity",
    )
    for count in ("50", "100", "250"):
        davis_values.append(
            _finite(
                _path(
                    davis_neighbours,
                    (
                        count,
                        "representations",
                        "paired_comparisons",
                        "two_way_residual_minus_absolute_vina",
                    ),
                    f"DAVIS {count}-neighbour calibration contrast",
                ),
                f"DAVIS {count}-neighbour calibration contrast",
            )
        )
    output["DAVIS"] = np.asarray(davis_values)

    pkis_calibration = _path(
        pkis2,
        ("molecular_size_matched_reference_calibration",),
        "PKIS2 molecular-size-matched calibration sensitivity",
    )
    if pkis_calibration.get("status") != "complete" or not pkis_calibration.get(
        "release_result_available", False
    ):
        raise ValueError(
            "PKIS2 molecular-size-matched reference sensitivity is unavailable; "
            "a full identity scan is required"
        )
    pkis_values = [
        _finite(
            _path(
                pkis2,
                (
                    "primary",
                    "paired_comparisons",
                    "two_way_residual_minus_absolute_vina",
                    "absolute_difference_gt_10",
                    "plugin_mean_difference",
                ),
                "PKIS2 full-reference calibration contrast",
            ),
            "PKIS2 full-reference calibration contrast",
        )
    ]
    pkis_neighbours = _path(
        pkis_calibration,
        ("by_neighbour_count",),
        "PKIS2 neighbour-count calibration sensitivity",
    )
    for count in ("50", "100", "250"):
        pkis_values.append(
            _finite(
                _path(
                    pkis_neighbours,
                    (
                        count,
                        "representations",
                        "paired_comparisons",
                        "two_way_residual_minus_absolute_vina",
                        "absolute_difference_gt_10",
                    ),
                    f"PKIS2 {count}-neighbour calibration contrast",
                ),
                f"PKIS2 {count}-neighbour calibration contrast",
            )
        )
    output["PKIS2"] = np.asarray(pkis_values)
    return output


def dense_operational_figure(davis: dict[str, Any], pkis2: dict[str, Any]) -> None:
    """Show the dense operational results without promoting secondary strata."""

    _validate_dense_reports(davis, pkis2)
    fig, axes = plt.subplots(2, 2, figsize=(WIDTH, 5.15))
    a, b, c, d = axes.flat

    # a: primary representation estimates and the explicitly secondary DAVIS stratum.
    x = np.arange(len(CORE_REPRESENTATIONS), dtype=float)
    davis_values, pkis_values, davis_binder = _dense_primary_values(davis, pkis2)
    a.axhline(0.5, color=GREY, lw=0.75, ls=":", zorder=0)
    a.plot(
        x,
        davis_values,
        marker="o",
        ms=4.5,
        color=BLUE,
        label="DAVIS primary: pKd, floor-floor excluded",
    )
    a.plot(
        x,
        pkis_values,
        marker="s",
        ms=4.2,
        color=ORANGE,
        label=r"PKIS2 primary: |$\Delta$ inhibition| > 10 pp",
    )
    a.plot(
        x,
        davis_binder,
        marker="^",
        ms=3.9,
        color=GREY,
        ls="--",
        label="DAVIS both-uncensored (exploratory)",
    )
    a.set_xticks(x, [REPRESENTATION_LABELS[value] for value in CORE_REPRESENTATIONS])
    a.set_ylabel("equal-ligand concordance")
    a.set_title("Dense observed-pair target preference", loc="left")
    a.set_ylim(
        min(0.44, float(davis_binder.min()) - 0.015),
        max(0.565, float(max(davis_values.max(), pkis_values.max())) + 0.015),
    )
    a.legend(frameon=False, fontsize=6.0, loc="best", handlelength=1.6)
    a.text(0.98, 0.04, "dotted line: chance", transform=a.transAxes,
           ha="right", va="bottom", fontsize=6.0, color=INK)
    clean(a)

    # b: paired chemical-cluster intervals for the primary contrast.
    contrasts = _dense_contrast_records(davis, pkis2)
    rows = [
        ("DAVIS - Murcko", "DAVIS", "murcko_cluster_bootstrap", BLUE, "o"),
        ("DAVIS - Butina", "DAVIS", "butina_cluster_bootstrap", BLUE, "s"),
        ("PKIS2 - Murcko", "PKIS2", "murcko_cluster_bootstrap", ORANGE, "o"),
        ("PKIS2 - Butina", "PKIS2", "butina_cluster_bootstrap", ORANGE, "s"),
    ]
    all_limits: list[float] = []
    for y, (label, dataset, method, color, marker) in enumerate(rows[::-1]):
        record = contrasts[dataset]
        estimate = _finite(record["plugin_mean_difference"], f"{dataset} contrast")
        interval = _interval(record, method, f"{dataset} {method} contrast")
        all_limits.extend(interval)
        _horizontal_interval(
            b, estimate, float(y), interval, color=color, marker=marker
        )
    labels = [row[0] for row in rows[::-1]]
    b.set_yticks(np.arange(len(labels)), labels)
    b.axvline(0.0, color=INK, lw=0.75, ls=":", zorder=0)
    bound = max(0.02, max(abs(value) for value in all_limits) * 1.18)
    b.set_xlim(-bound, bound)
    tick = min(0.02, 0.65 * bound)
    b.set_xticks([-tick, 0.0, tick], [f"{-tick:.2f}", "0.00", f"{tick:.2f}"])
    b.set_xlabel(r"paired $\Delta$ concordance" "\n(two-way residual - absolute Vina)")
    b.set_title("Primary contrasts: 95% cluster bootstrap", loc="left")
    clean(b)
    b.grid(axis="x", color="#E1E5E9", linewidth=0.55, zorder=0)
    b.grid(axis="y", visible=False)

    # c: binary panels use assay-appropriate definitions and cluster intervals.
    davis_binary, pkis_binary = _dense_binary_records(davis, pkis2)
    xb = np.arange(len(BINARY_REPRESENTATIONS), dtype=float)
    offset = 0.10
    binary_interval_limits: list[float] = []
    for index, representation in enumerate(BINARY_REPRESENTATIONS):
        d_record = davis_binary[representation]
        p_record = pkis_binary[representation]
        d_value = _finite(
            d_record["mean_per_ligand_auc"], f"DAVIS {representation} hit AUC"
        )
        p_value = _finite(
            p_record["mean_per_ligand_roc_auc"], f"PKIS2 {representation} ROC AUC"
        )
        davis_interval = _binary_interval(d_record, "DAVIS")
        pkis_interval = _binary_interval(p_record, "PKIS2")
        binary_interval_limits.extend((*davis_interval, *pkis_interval))
        _errorbar(
            c,
            index - offset,
            d_value,
            davis_interval,
            color=BLUE,
            marker="o",
        )
        _errorbar(
            c,
            index + offset,
            p_value,
            pkis_interval,
            color=ORANGE,
            marker="s",
        )
    c.axhline(0.5, color=GREY, lw=0.75, ls=":", zorder=0)
    c.set_xticks(
        xb, [BINARY_LABELS[value] for value in BINARY_REPRESENTATIONS]
    )
    c.tick_params(axis="x", labelsize=6.0)
    c.set_ylabel("mean per-ligand ROC AUC")
    c.set_title("Binary AUC: 95% Murcko-cluster bootstrap", loc="left")
    c.text(
        0.02,
        0.97,
        "DAVIS (blue): uncensored vs pKd=5 floor",
        transform=c.transAxes,
        va="top",
        fontsize=6.0,
        color=BLUE,
    )
    c.text(
        0.02,
        0.90,
        "PKIS2 (orange): inhibition >= 65% vs < 65%",
        transform=c.transAxes,
        va="top",
        fontsize=6.0,
        color=ORANGE,
    )
    c.set_ylim(
        min(0.38, min(binary_interval_limits) - 0.01),
        max(0.67, max(binary_interval_limits) + 0.01),
    )
    clean(c)

    # d: reference-support sensitivity of learned target offsets and scales.
    calibration = _calibration_contrast_values(davis, pkis2)
    xd = np.arange(4, dtype=float)
    styles = {"DAVIS": (BLUE, "o"), "PKIS2": (ORANGE, "s")}
    d.axhline(0.0, color=INK, lw=0.75, ls=":", zorder=0)
    for dataset, values in calibration.items():
        color, marker = styles[dataset]
        d.plot(
            xd,
            values,
            marker=marker,
            ms=4.2,
            color=color,
            label=dataset,
        )
    d.set_xticks(xd, ["Full\nreference", "50 NN", "100 NN", "250 NN"])
    d.set_ylabel(r"paired $\Delta$ concordance" "\n(residual - absolute Vina)")
    d.set_title("Calibration-support sensitivity", loc="left")
    d.text(
        0.02,
        0.97,
        "MW / heavy-atom nearest-neighbour reference;\npoint estimates, no resampled CI",
        transform=d.transAxes,
        va="top",
        fontsize=6.0,
    )
    values = np.concatenate(list(calibration.values()))
    bound = max(0.003, 1.25 * float(np.max(np.abs(values))))
    d.set_ylim(-bound, bound)
    d.legend(frameon=False, fontsize=6.0, loc="lower right")
    clean(d)

    for ax, label in zip(axes.flat, "abcd"):
        panel_label(ax, label)
    fig.subplots_adjust(
        left=0.12,
        right=0.985,
        bottom=0.095,
        top=0.94,
        wspace=0.38,
        hspace=0.47,
    )
    assert audit_fig(fig)
    _audit_text_within_canvas(fig)
    _save(fig, "figS10_dense_operational_extensions")
    plt.close(fig)


def _pbas_variant_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    variants = _path(summary, ("variants",), "PBAS variants")
    specifications = [
        (
            "clip, zero-fill (corr.)",
            "clipped_fill_zero_primary",
            "correlation_spectrum",
        ),
        (
            "clip, zero-fill (cov.)",
            "clipped_fill_zero_primary",
            "covariance_spectrum",
        ),
        (
            "clip, target-mean (corr.)",
            "clipped_target_mean_imputation_sensitivity",
            "correlation_spectrum",
        ),
        (
            "clip, target-mean (cov.)",
            "clipped_target_mean_imputation_sensitivity",
            "covariance_spectrum",
        ),
        (
            "no clip, zero-fill (corr.)",
            "unclipped_fill_zero_sensitivity",
            "correlation_spectrum",
        ),
        (
            "no clip, zero-fill (cov.)",
            "unclipped_fill_zero_sensitivity",
            "covariance_spectrum",
        ),
    ]
    rows: list[dict[str, Any]] = []
    for label, variant, spectrum in specifications:
        variant_record = _path(variants, (variant,), f"PBAS {variant}")
        raw = _finite(
            _path(
                variant_record,
                (
                    "surfaces",
                    "column_centered_raw",
                    spectrum,
                    "participation_ratio",
                ),
                f"PBAS {variant} raw {spectrum}",
            ),
            f"PBAS {variant} raw PR",
        )
        residual = _finite(
            _path(
                variant_record,
                (
                    "surfaces",
                    "two_way_residual",
                    spectrum,
                    "participation_ratio",
                ),
                f"PBAS {variant} residual {spectrum}",
            ),
            f"PBAS {variant} residual PR",
        )
        rows.append({"label": label, "raw": raw, "residual": residual})
    filtered = _path(
        summary,
        ("drop_high_missing_target_sensitivity",),
        "PBAS high-missing-target sensitivity",
    )
    rows.append(
        {
            "label": (
                f"low-missing targets (corr.)\n"
                f"P={int(filtered['retained_targets']):,}"
            ),
            "raw": _finite(
                _path(
                    filtered,
                    (
                        "surfaces",
                        "column_centered_raw",
                        "correlation_spectrum",
                        "participation_ratio",
                    ),
                    "PBAS low-missing raw PR",
                ),
                "PBAS low-missing raw PR",
            ),
            "residual": _finite(
                _path(
                    filtered,
                    (
                        "surfaces",
                        "two_way_residual",
                        "correlation_spectrum",
                        "participation_ratio",
                    ),
                    "PBAS low-missing residual PR",
                ),
                "PBAS low-missing residual PR",
            ),
        }
    )
    return rows


def _pbas_primary_pr(summary: dict[str, Any], surface: str) -> float:
    return _finite(
        _path(
            summary,
            (
                "variants",
                "clipped_fill_zero_primary",
                "surfaces",
                surface,
                "correlation_spectrum",
                "participation_ratio",
            ),
            f"PBAS primary {surface} PR",
        ),
        f"PBAS primary {surface} PR",
    )


def _pbas_association_matrix(associations: pd.DataFrame) -> np.ndarray:
    required = {
        "variant",
        "surface",
        "spectrum",
        "component",
        "covariate",
        "absolute_correlation",
    }
    missing = required - set(associations.columns)
    if missing:
        raise ValueError(
            f"PBAS component association table lacks columns: {sorted(missing)}"
        )
    selected = associations[
        associations.variant.eq("clipped_fill_zero_primary")
        & associations.spectrum.eq("correlation")
        & associations.component.isin([1, 2])
    ].copy()
    surface_order = ("column_centered_raw", "two_way_residual")
    covariate_order = (
        "preprocessed_per_drug_mean",
        "preprocessed_per_drug_zero_fraction",
    )
    rows = []
    for surface in surface_order:
        for component in (1, 2):
            row = []
            for covariate in covariate_order:
                matches = selected[
                    selected.surface.eq(surface)
                    & selected.component.eq(component)
                    & selected.covariate.eq(covariate)
                ]
                if len(matches) != 1:
                    raise ValueError(
                        "Expected one PBAS association for "
                        f"{surface}, PC{component}, {covariate}; found {len(matches)}"
                    )
                row.append(
                    _finite(
                        matches.iloc[0].absolute_correlation,
                        f"PBAS {surface} PC{component} association",
                    )
                )
            rows.append(row)
    return np.asarray(rows, dtype=float)


def pbas_boundary_figure(
    summary: dict[str, Any],
    subsampling: pd.DataFrame,
    associations: pd.DataFrame,
) -> None:
    """Show why the proteome-wide surface is a boundary case, not replication."""

    if not _path(summary, ("source", "checksum_verified"), "PBAS source"):
        raise ValueError("PBAS figure requires a checksum-verified source archive")
    n_ligands = int(_path(summary, ("source", "n_ligands"), "PBAS source"))
    n_targets = int(_path(summary, ("source", "n_targets"), "PBAS source"))
    if (n_ligands, n_targets) != (7582, 19135):
        raise ValueError(
            f"Unexpected PBAS source support: {n_ligands:,} x {n_targets:,}"
        )

    fig, axes = plt.subplots(2, 2, figsize=(WIDTH, 5.05))
    a, b, c, d = axes.flat

    # a: analysis choices, with paired raw/residual values for every variant.
    variant_rows = _pbas_variant_rows(summary)
    y = np.arange(len(variant_rows))[::-1]
    raw = np.asarray([row["raw"] for row in variant_rows], dtype=float)
    residual = np.asarray([row["residual"] for row in variant_rows], dtype=float)
    for yi, raw_value, residual_value in zip(y, raw, residual):
        a.plot(
            [raw_value, residual_value],
            [yi, yi],
            color=GREY,
            lw=0.85,
            zorder=1,
        )
        a.plot(raw_value, yi, marker="o", color=BLUE, ms=4.4, zorder=2)
        a.plot(residual_value, yi, marker="s", color=ORANGE, ms=4.1, zorder=2)
    a.set_yticks(y, [row["label"] for row in variant_rows])
    a.tick_params(axis="y", labelsize=6.0)
    a.set_xlabel("PR effective dimension")
    a.set_title("Preprocessing / estimand sensitivity", loc="left")
    a.plot([], [], marker="o", color=BLUE, ls="none", label="raw")
    a.plot([], [], marker="s", color=ORANGE, ls="none", label="two-way residual")
    a.legend(frameon=False, fontsize=6.0, loc="lower right")
    a.text(
        0.98,
        0.98,
        "Primary: raw 3.50; residual 3.28",
        transform=a.transAxes,
        ha="right",
        va="top",
        fontsize=6.2,
    )
    clean(a)
    a.grid(axis="x", color="#E1E5E9", linewidth=0.55, zorder=0)
    a.grid(axis="y", visible=False)

    # b: same-N,P independent-column reference; logarithmic separation is essential.
    observed = np.asarray(
        [
            _pbas_primary_pr(summary, "column_centered_raw"),
            _pbas_primary_pr(summary, "two_way_residual"),
        ]
    )
    null = _path(
        summary,
        ("independent_column_null_baseline",),
        "PBAS finite-aspect null",
    )
    expected = np.asarray(
        [
            _finite(
                null["raw_participation_ratio_baseline"], "PBAS raw null PR"
            ),
            _finite(
                null["residual_participation_ratio_baseline"],
                "PBAS residual null PR",
            ),
        ]
    )
    xb = np.arange(2, dtype=float)
    b.plot(xb, observed, marker="o", color=PLUM, ms=5, label="observed")
    b.plot(
        xb,
        expected,
        marker="^",
        color=GREY,
        ms=5,
        ls="--",
        label="independent-column baseline",
    )
    b.set_yscale("log")
    b.set_yticks(
        [3, 10, 100, 1_000, 5_000],
        ["3", "10", "100", "1,000", "5,000"],
    )
    b.set_xticks(xb, ["Raw", "Two-way\nresidual"])
    b.set_xlim(-0.45, 1.45)
    b.set_ylabel("PR effective dimension (log scale)")
    b.set_title("Finite-aspect null reference", loc="left")
    b.text(
        0.5,
        0.51,
        f"N={n_ligands:,}, P={n_targets:,}\nnull PR approx {expected.mean():,.0f}",
        transform=b.transAxes,
        ha="center",
        va="center",
        fontsize=6.2,
    )
    b.legend(frameon=False, fontsize=6.0, loc="upper right")
    clean(b)

    # c: target-count supports show that the direction is not a full-panel artefact.
    required_subsampling = {
        "target_count",
        "raw_correlation_participation_ratio",
        "residual_correlation_participation_ratio",
    }
    missing = required_subsampling - set(subsampling.columns)
    if missing:
        raise ValueError(f"PBAS target subsampling lacks columns: {sorted(missing)}")
    grouped = subsampling.groupby("target_count", sort=True)
    target_counts = np.asarray(sorted(grouped.groups), dtype=float)
    for field, color, marker, label in (
        (
            "raw_correlation_participation_ratio",
            BLUE,
            "o",
            "raw",
        ),
        (
            "residual_correlation_participation_ratio",
            ORANGE,
            "s",
            "two-way residual",
        ),
    ):
        median = grouped[field].median().reindex(target_counts.astype(int)).to_numpy()
        low = grouped[field].min().reindex(target_counts.astype(int)).to_numpy()
        high = grouped[field].max().reindex(target_counts.astype(int)).to_numpy()
        c.fill_between(target_counts, low, high, color=color, alpha=0.12)
        c.plot(target_counts, median, marker=marker, color=color, ms=3.8, label=label)
    c.plot(
        [n_targets],
        [observed[0]],
        marker="o",
        color=BLUE,
        ms=5.0,
        markerfacecolor="white",
        markeredgewidth=1.0,
    )
    c.plot(
        [n_targets],
        [observed[1]],
        marker="s",
        color=ORANGE,
        ms=4.8,
        markerfacecolor="white",
        markeredgewidth=1.0,
    )
    c.set_xscale("log")
    c.set_xlim(35, 30_000)
    c.set_xticks([100, 1_000, 10_000])
    c.set_xlabel("targets sampled (log scale)")
    c.set_ylabel("correlation-spectrum PR")
    c.set_title("Target-count sensitivity", loc="left")
    c.text(
        0.98,
        0.04,
        "bands: min-max over fixed-seed replicates\nopen symbols: full P=19,135 panel",
        transform=c.transAxes,
        ha="right",
        va="bottom",
        fontsize=6.0,
    )
    c.legend(frameon=False, fontsize=6.0, loc="upper left")
    clean(c)

    # d: component signs are arbitrary, hence absolute correlations are plotted.
    association_matrix = _pbas_association_matrix(associations)
    image = d.imshow(
        association_matrix,
        cmap="cividis",
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
        aspect="auto",
    )
    d.set_xticks([0, 1], ["drug-row\nmean", "zero\nfraction"])
    d.set_yticks(
        np.arange(4),
        ["Raw PC1", "Raw PC2", "Residual PC1", "Residual PC2"],
    )
    for row in range(association_matrix.shape[0]):
        for column in range(association_matrix.shape[1]):
            value = association_matrix[row, column]
            d.text(
                column,
                row,
                f"{value:.2f}",
                ha="center",
                va="center",
                color="white" if value > 0.52 else INK,
                fontsize=7.0,
            )
    d.set_title("PC associations with row covariates ($|r|$)", loc="left")
    colorbar = fig.colorbar(image, ax=d, fraction=0.046, pad=0.04)
    colorbar.ax.tick_params(labelsize=6.0, length=2)
    d.tick_params(length=0)

    for ax, label in zip(axes.flat, "abcd"):
        panel_label(ax, label)
    fig.subplots_adjust(
        left=0.18,
        right=0.97,
        bottom=0.105,
        top=0.94,
        wspace=0.43,
        hspace=0.42,
    )
    assert audit_fig(fig)
    _audit_text_within_canvas(fig)
    _save(fig, "figS11_pbas_boundary_audit")
    plt.close(fig)


def main() -> None:
    required = (
        DAVIS_PATH,
        PKIS2_PATH,
        PBAS_SUMMARY_PATH,
        PBAS_SUBSAMPLING_PATH,
        PBAS_ASSOCIATIONS_PATH,
        PBAS_EIGENVALUES_PATH,
        PBAS_MISSINGNESS_PATH,
    )
    _require_files(required)
    use_paper_style()
    davis = _load_json(DAVIS_PATH)
    pkis2 = _load_json(PKIS2_PATH)
    pbas = _load_json(PBAS_SUMMARY_PATH)
    subsampling = pd.read_csv(PBAS_SUBSAMPLING_PATH)
    associations = pd.read_csv(PBAS_ASSOCIATIONS_PATH)
    # Parse the auxiliary PBAS tables too: this catches truncated/incomplete runs
    # even though the current panels do not need their individual rows.
    if pd.read_csv(PBAS_EIGENVALUES_PATH).empty:
        raise ValueError(f"PBAS eigenvalue table is empty: {PBAS_EIGENVALUES_PATH}")
    if pd.read_csv(PBAS_MISSINGNESS_PATH).empty:
        raise ValueError(f"PBAS missingness table is empty: {PBAS_MISSINGNESS_PATH}")
    dense_operational_figure(davis, pkis2)
    pbas_boundary_figure(pbas, subsampling, associations)
    print(f"Wrote extension figures to {OUT}")


if __name__ == "__main__":
    main()
