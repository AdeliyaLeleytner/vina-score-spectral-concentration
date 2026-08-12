#!/usr/bin/env python3
"""Revision analyses for support transport and decision-level map recovery.

This module deliberately keeps two questions separate.

1. Threshold robustness: do target-correlation maps change progressively with
   molecular-weight support location, rather than only at one post-hoc
   quartile boundary?
2. Operational recovery: when a residual map is estimated from 200 or 500
   representative ligands, are its strongest edges, signs, clusters and a
   concrete non-redundant target-panel decision recovered as well as its global
   edge ordering?

All analyses are descriptive, outcome-blind and conditional on the tested Vina
panels.  The molecular-weight analysis remains post hoc.  The target-panel
decision optimizes coverage of the *Vina correlation distance* and is not an
experimental target-selection validation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy import stats
from scipy.cluster.hierarchy import cut_tree, linkage
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.spatial.distance import squareform
from scipy.sparse import coo_matrix
from sklearn.metrics import adjusted_rand_score

try:  # Direct script execution and package-style import are both supported.
    from .residual_mechanism_analysis import (
        DESCRIPTOR_NAMES,
        AnalysisConfig,
        DatasetBundle,
        load_docking44,
        load_dockstring,
        two_way_center,
    )
except ImportError:  # pragma: no cover - direct CLI execution.
    from residual_mechanism_analysis import (  # type: ignore
        DESCRIPTOR_NAMES,
        AnalysisConfig,
        DatasetBundle,
        load_docking44,
        load_dockstring,
        two_way_center,
    )


PACKAGE = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PACKAGE / "results" / "revision_map_decision_sensitivities"
DEFAULT_SEED = 202_608_10
DEFAULT_RECOVERY_SIZES = (200, 500)
DEFAULT_CLUSTER_K = (6, 8, 10)
DEFAULT_TAIL_FRACTIONS = (0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40)
DEFAULT_EDGE_K = (10, 25)
DEFAULT_MW_NULL_REPETITIONS = 100
DEFAULT_COMPLEMENT_RANDOM_PANELS = 200


def correlation_pr(correlation: np.ndarray) -> float:
    """Participation-ratio dimension of a finite correlation matrix."""
    correlation = np.asarray(correlation, dtype=np.float64)
    if (
        correlation.ndim != 2
        or correlation.shape[0] != correlation.shape[1]
        or not np.isfinite(correlation).all()
    ):
        raise ValueError("correlation PR requires a finite square matrix")
    return float(np.trace(correlation) ** 2 / np.square(correlation).sum())


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float) and not np.isfinite(value):
        raise ValueError("non-finite value cannot be serialized")
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(json_ready(payload), indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def upper(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("upper triangle requires a square matrix")
    return matrix[np.triu_indices(len(matrix), k=1)]


def target_geometry(matrix: np.ndarray, transformation: str) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] < 3 or matrix.shape[1] < 2:
        raise ValueError("target geometry requires at least 3 rows and 2 columns")
    if not np.isfinite(matrix).all():
        raise ValueError("target geometry contains non-finite values")
    if transformation == "raw":
        transformed = matrix
    elif transformation == "row_centered_residual":
        transformed = two_way_center(matrix)
    else:
        raise ValueError(f"unknown transformation: {transformation}")
    geometry = np.corrcoef(transformed, rowvar=False)
    geometry = np.clip((geometry + geometry.T) / 2.0, -1.0, 1.0)
    np.fill_diagonal(geometry, 1.0)
    return geometry


def correlation_from_sufficient_statistics(
    rows: int,
    column_sum: np.ndarray,
    cross_product: np.ndarray,
) -> np.ndarray:
    """Correlation matrix from row count, column sums and X'X."""
    column_sum = np.asarray(column_sum, dtype=np.float64)
    cross_product = np.asarray(cross_product, dtype=np.float64)
    if (
        rows < 3
        or column_sum.ndim != 1
        or cross_product.shape != (len(column_sum), len(column_sum))
    ):
        raise ValueError("invalid sufficient statistics for target correlation")
    covariance_numerator = cross_product - np.outer(column_sum, column_sum) / rows
    variance = np.diag(covariance_numerator)
    if np.any(variance <= 1e-14):
        raise ValueError("sufficient statistics contain a constant target")
    correlation = covariance_numerator / np.sqrt(np.outer(variance, variance))
    correlation = np.clip((correlation + correlation.T) / 2.0, -1.0, 1.0)
    np.fill_diagonal(correlation, 1.0)
    return correlation


def row_centered_statistics(
    matrix: np.ndarray,
    chunk_size: int = 50_000,
) -> tuple[np.ndarray, np.ndarray]:
    """Accumulate row-centred column sums and cross-products without a full copy."""
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2 or min(matrix.shape) < 2 or not np.isfinite(matrix).all():
        raise ValueError("row-centred statistics require a finite matrix")
    column_sum = np.zeros(matrix.shape[1], dtype=np.float64)
    cross_product = np.zeros((matrix.shape[1], matrix.shape[1]), dtype=np.float64)
    for start in range(0, len(matrix), chunk_size):
        chunk = matrix[start : start + chunk_size]
        residual = chunk - chunk.mean(axis=1, keepdims=True)
        column_sum += residual.sum(axis=0)
        cross_product += residual.T @ residual
    return column_sum, cross_product


def residual_subset_statistics(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.asarray(matrix, dtype=np.float64)
    residual = matrix - matrix.mean(axis=1, keepdims=True)
    return residual.sum(axis=0), residual.T @ residual


def stable_top_mask(values: np.ndarray, count: int, *, largest: bool) -> np.ndarray:
    """Select exactly ``count`` entries, breaking numerical ties by pair order."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("edge values must be a finite vector")
    if count < 1 or count > len(values):
        raise ValueError("edge count is outside the available pair range")
    order = np.argsort(values, kind="mergesort")
    chosen = order[-count:] if largest else order[:count]
    mask = np.zeros(len(values), dtype=bool)
    mask[chosen] = True
    return mask


def set_overlap(first: np.ndarray, second: np.ndarray) -> dict[str, float]:
    first = np.asarray(first, dtype=bool)
    second = np.asarray(second, dtype=bool)
    if first.shape != second.shape or first.ndim != 1:
        raise ValueError("set masks must have the same one-dimensional shape")
    intersection = int(np.sum(first & second))
    union = int(np.sum(first | second))
    return {
        "intersection": intersection,
        "precision": float(intersection / max(1, int(second.sum()))),
        "recall": float(intersection / max(1, int(first.sum()))),
        "jaccard": float(intersection / max(1, union)),
    }


def map_comparison(first: np.ndarray, second: np.ndarray) -> dict[str, float]:
    first_values = upper(first)
    second_values = upper(second)
    rho = float(stats.spearmanr(first_values, second_values).statistic)
    if not np.isfinite(rho):
        raise ValueError("map agreement is not finite")
    return {
        "geometry_spearman": rho,
        "geometry_dissimilarity": 1.0 - rho,
        "edge_correlation_rmse": float(
            np.sqrt(np.mean(np.square(first_values - second_values)))
        ),
        "edge_sign_agreement": float(
            np.mean(np.sign(first_values) == np.sign(second_values))
        ),
        "first_pr": correlation_pr(first),
        "second_pr": correlation_pr(second),
    }


def quantile_bin_ids(values: np.ndarray, bins: int) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic value-threshold bins; a boundary tie enters the upper bin."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all() or bins < 2:
        raise ValueError("quantile binning requires a finite vector and >=2 bins")
    edges = np.quantile(values, np.linspace(0.0, 1.0, bins + 1))
    if np.any(np.diff(edges) <= 0):
        raise ValueError("descriptor cannot define the requested distinct quantiles")
    identifiers = np.searchsorted(edges[1:-1], values, side="right")
    if set(np.unique(identifiers)) != set(range(bins)):
        raise ValueError("at least one requested quantile bin is empty")
    return identifiers.astype(np.int16), edges.astype(np.float64)


def mw_threshold_sweep(
    bundle: DatasetBundle,
    fractions: Iterable[float] = DEFAULT_TAIL_FRACTIONS,
) -> pd.DataFrame:
    """Compare low/high MW tails across a grid of non-overlapping cutoffs."""
    mw = bundle.descriptor_matrix[:, DESCRIPTOR_NAMES.index("molecular_weight")]
    records: list[dict[str, Any]] = []
    for fraction in fractions:
        if not 0 < fraction < 0.5:
            raise ValueError("tail fractions must lie in (0, 0.5)")
        order = np.lexsort((np.arange(len(mw)), mw))
        count = int(np.floor(fraction * len(mw)))
        if count < 3 or 2 * count >= len(mw):
            raise ValueError("tail threshold produced invalid supports")
        low_index = np.sort(order[:count])
        high_index = np.sort(order[-count:])
        if np.intersect1d(low_index, high_index).size:
            raise AssertionError("MW tails overlap")
        low_threshold = float(mw[order[count - 1]])
        high_threshold = float(mw[order[-count]])
        for transformation in ("raw", "row_centered_residual"):
            comparison = map_comparison(
                target_geometry(bundle.matrix[low_index], transformation),
                target_geometry(bundle.matrix[high_index], transformation),
            )
            records.append(
                {
                    "dataset": bundle.name,
                    "transformation": transformation,
                    "tail_fraction_per_side": float(fraction),
                    "low_threshold_inclusive": float(low_threshold),
                    "high_threshold_inclusive": float(high_threshold),
                    "low_ligands": int(count),
                    "high_ligands": int(count),
                    "low_median_mw": float(np.median(mw[low_index])),
                    "high_median_mw": float(np.median(mw[high_index])),
                    "median_mw_separation": float(
                        np.median(mw[high_index]) - np.median(mw[low_index])
                    ),
                    **comparison,
                }
            )
    return pd.DataFrame.from_records(records)


def mw_equal_n_random_disjoint_controls(
    bundle: DatasetBundle,
    repetitions: int,
    seed: int,
    fractions: Iterable[float] = DEFAULT_TAIL_FRACTIONS,
) -> pd.DataFrame:
    """Finite-N controls matched separately to every exact-rank MW tail size."""
    if repetitions < 2:
        raise ValueError("MW controls require at least two repetitions")
    fractions = tuple(float(value) for value in fractions)
    row_centered = bundle.matrix - bundle.matrix.mean(axis=1, keepdims=True)
    rng = np.random.default_rng(seed)
    records: list[dict[str, Any]] = []
    for repetition in range(repetitions):
        order = rng.permutation(len(bundle.matrix))
        for fraction in fractions:
            count = int(np.floor(fraction * len(bundle.matrix)))
            first_index = np.sort(order[:count])
            second_index = np.sort(order[count : 2 * count])
            if np.intersect1d(first_index, second_index).size:
                raise AssertionError("random MW-control supports overlap")
            for transformation, surface in (
                ("raw", bundle.matrix),
                ("row_centered_residual", row_centered),
            ):
                comparison = map_comparison(
                    target_geometry(surface[first_index], "raw"),
                    target_geometry(surface[second_index], "raw"),
                )
                records.append(
                    {
                        "dataset": bundle.name,
                        "transformation": transformation,
                        "tail_fraction_per_side": fraction,
                        "ligands_per_support": count,
                        "repetition": repetition,
                        **comparison,
                    }
                )
    return pd.DataFrame.from_records(records)


def attach_mw_control_summary(
    observed: pd.DataFrame,
    controls: pd.DataFrame,
) -> pd.DataFrame:
    keys = ["dataset", "transformation", "tail_fraction_per_side"]
    summary = (
        controls.groupby(keys, as_index=False)
        .geometry_spearman.agg(
            control_geometry_spearman_mean="mean",
            control_geometry_spearman_q025=lambda values: np.quantile(values, 0.025),
            control_geometry_spearman_q975=lambda values: np.quantile(values, 0.975),
            control_repetitions="count",
        )
    )
    merged = observed.merge(summary, on=keys, how="left", validate="one_to_one")
    if merged.control_geometry_spearman_mean.isna().any():
        raise ValueError("MW control summary did not cover every observed threshold")
    merged["observed_minus_control_mean_geometry_spearman"] = (
        merged.geometry_spearman - merged.control_geometry_spearman_mean
    )
    merged["observed_below_control_q025"] = (
        merged.geometry_spearman < merged.control_geometry_spearman_q025
    )
    return merged


def mw_decile_continuum(
    bundle: DatasetBundle,
    bins: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Non-overlapping MW-bin maps and their continuous separation relationship."""
    mw = bundle.descriptor_matrix[:, DESCRIPTOR_NAMES.index("molecular_weight")]
    order = np.lexsort((np.arange(len(mw)), mw))
    rows_per_bin = len(mw) // bins
    if rows_per_bin < 3:
        raise ValueError("too few rows for fixed-size MW bins")
    used_rows = rows_per_bin * bins
    bin_indices = [
        np.sort(order[index * rows_per_bin : (index + 1) * rows_per_bin])
        for index in range(bins)
    ]
    bin_records: list[dict[str, Any]] = []
    pair_records: list[dict[str, Any]] = []
    trend_records: list[dict[str, Any]] = []
    for transformation in ("raw", "row_centered_residual"):
        maps: list[np.ndarray] = []
        medians: list[float] = []
        for bin_index in range(bins):
            selected = bin_indices[bin_index]
            geometry = target_geometry(bundle.matrix[selected], transformation)
            maps.append(geometry)
            medians.append(float(np.median(mw[selected])))
            values = upper(geometry)
            bin_records.append(
                {
                    "dataset": bundle.name,
                    "transformation": transformation,
                    "mw_decile": bin_index + 1,
                    "ligands": int(len(selected)),
                    "lower_mw_inclusive": float(np.min(mw[selected])),
                    "upper_mw_inclusive": float(np.max(mw[selected])),
                    "median_molecular_weight": medians[-1],
                    "rows_excluded_after_equal_n_binning": int(
                        len(mw) - used_rows
                    ),
                    "target_correlation_pr": correlation_pr(geometry),
                    "mean_signed_target_correlation": float(values.mean()),
                    "mean_absolute_target_correlation": float(np.abs(values).mean()),
                }
            )
        for first in range(bins):
            for second in range(first + 1, bins):
                pair_records.append(
                    {
                        "dataset": bundle.name,
                        "transformation": transformation,
                        "first_mw_decile": first + 1,
                        "second_mw_decile": second + 1,
                        "decile_separation": second - first,
                        "first_median_molecular_weight": medians[first],
                        "second_median_molecular_weight": medians[second],
                        "median_mw_separation": medians[second] - medians[first],
                        **map_comparison(maps[first], maps[second]),
                    }
                )
        selected = pd.DataFrame.from_records(pair_records)
        selected = selected.loc[
            (selected.dataset == bundle.name)
            & (selected.transformation == transformation)
        ]
        separation = selected.median_mw_separation.to_numpy(dtype=np.float64)
        dissimilarity = selected.geometry_dissimilarity.to_numpy(dtype=np.float64)
        slope, intercept = np.polyfit(separation, dissimilarity, deg=1)
        trend_records.append(
            {
                "dataset": bundle.name,
                "transformation": transformation,
                "mw_bins": bins,
                "bin_pairs": int(len(selected)),
                "spearman_mw_separation_vs_map_dissimilarity": float(
                    stats.spearmanr(separation, dissimilarity).statistic
                ),
                "descriptive_ols_dissimilarity_per_100_da": float(100.0 * slope),
                "descriptive_ols_intercept": float(intercept),
                "minimum_pair_agreement": float(selected.geometry_spearman.min()),
                "maximum_pair_agreement": float(selected.geometry_spearman.max()),
                "inference": (
                    "descriptive only; the 45 pairwise comparisons share ten maps and "
                    "are not independent observations"
                ),
            }
        )
    return (
        pd.DataFrame.from_records(bin_records),
        pd.DataFrame.from_records(pair_records),
        pd.DataFrame.from_records(trend_records),
    )


def correlation_distance(geometry: np.ndarray) -> np.ndarray:
    geometry = np.asarray(geometry, dtype=np.float64)
    distance = np.clip(1.0 - geometry, 0.0, 2.0)
    distance = (distance + distance.T) / 2.0
    np.fill_diagonal(distance, 0.0)
    return distance


def dependency_distance(geometry: np.ndarray) -> np.ndarray:
    """Unsigned redundancy distance used only for panel compression.

    Both a strongly positive and a strongly negative target-profile correlation
    imply linear redundancy for a probe-panel coverage decision.  Signed
    ``1-r`` remains the estimand for cluster recovery and edge interpretation.
    """
    geometry = np.asarray(geometry, dtype=np.float64)
    distance = np.clip(1.0 - np.abs(geometry), 0.0, 1.0)
    distance = (distance + distance.T) / 2.0
    np.fill_diagonal(distance, 0.0)
    return distance


def hierarchical_labels(distance: np.ndarray, k: int) -> np.ndarray:
    distance = np.asarray(distance, dtype=np.float64)
    if distance.ndim != 2 or distance.shape[0] != distance.shape[1]:
        raise ValueError("clustering requires a square distance matrix")
    if not 2 <= k < len(distance):
        raise ValueError("cluster count must lie between 2 and targets-1")
    condensed = squareform(distance, checks=True)
    tree = linkage(condensed, method="average", optimal_ordering=True)
    labels = cut_tree(tree, n_clusters=[k]).reshape(-1).astype(np.int16)
    if len(np.unique(labels)) != k:
        raise RuntimeError("hierarchical cut did not produce the requested clusters")
    return labels


def panel_coverage_objective(distance: np.ndarray, medoids: np.ndarray) -> dict[str, float]:
    distance = np.asarray(distance, dtype=np.float64)
    medoids = np.asarray(medoids, dtype=np.int64)
    nearest = np.min(distance[:, medoids], axis=1)
    return {
        "mean_nearest_distance": float(nearest.mean()),
        "maximum_nearest_distance": float(nearest.max()),
    }


def deterministic_pam(
    distance: np.ndarray,
    k: int,
    target_names: list[str],
) -> np.ndarray:
    """Deterministic BUILD + SWAP k-medoids for a small target panel."""
    distance = np.asarray(distance, dtype=np.float64)
    n = len(distance)
    if distance.shape != (n, n) or len(target_names) != n or not 1 <= k < n:
        raise ValueError("invalid distance, target labels or medoid count")
    candidate_order = sorted(range(n), key=lambda index: target_names[index])
    selected: list[int] = []

    def objective(indices: Iterable[int]) -> float:
        panel = np.asarray(sorted(indices), dtype=np.int64)
        return float(np.min(distance[:, panel], axis=1).sum())

    while len(selected) < k:
        candidates = [index for index in candidate_order if index not in selected]
        scored = [(objective([*selected, candidate]), candidate) for candidate in candidates]
        _, chosen = min(scored, key=lambda item: (item[0], target_names[item[1]]))
        selected.append(chosen)

    tolerance = 1e-12
    while True:
        current = objective(selected)
        best_value = current
        best_panel = sorted(selected)
        for outgoing in sorted(selected, key=lambda index: target_names[index]):
            for incoming in candidate_order:
                if incoming in selected:
                    continue
                proposal = sorted((set(selected) - {outgoing}) | {incoming})
                value = objective(proposal)
                proposal_names = tuple(target_names[index] for index in proposal)
                best_names = tuple(target_names[index] for index in best_panel)
                if value < best_value - tolerance or (
                    abs(value - best_value) <= tolerance and proposal_names < best_names
                ):
                    best_value = value
                    best_panel = proposal
        if best_value >= current - tolerance:
            break
        selected = best_panel
    return np.asarray(sorted(selected), dtype=np.int64)


def exact_k_medoids(distance: np.ndarray, k: int) -> np.ndarray:
    """Solve the full-map discrete k-medoids coverage problem exactly.

    Assignment variables are continuous because, conditional on binary medoid
    indicators, their simplex optimum is integral.  Only the target-selection
    indicators are declared binary.  A negligible deterministic index penalty
    breaks exact objective ties without changing any displayed coverage value.
    """
    distance = np.asarray(distance, dtype=np.float64)
    n = len(distance)
    if distance.shape != (n, n) or not np.isfinite(distance).all():
        raise ValueError("exact k-medoids requires a finite square distance matrix")
    if not 1 <= k < n:
        raise ValueError("medoid count must lie between 1 and targets-1")
    assignment_variables = n * n
    variables = assignment_variables + n
    objective = np.zeros(variables, dtype=np.float64)
    objective[:assignment_variables] = distance.reshape(-1)
    objective[assignment_variables:] = 1e-12 * np.arange(n, dtype=np.float64)
    integrality = np.zeros(variables, dtype=np.int8)
    integrality[assignment_variables:] = 1

    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []
    lower: list[float] = []
    upper_bounds: list[float] = []
    constraint = 0
    # Every target is assigned to exactly one selected representative.
    for target in range(n):
        for candidate in range(n):
            rows.append(constraint)
            columns.append(target * n + candidate)
            values.append(1.0)
        lower.append(1.0)
        upper_bounds.append(1.0)
        constraint += 1
    # An assignment can be positive only when its representative is selected.
    for target in range(n):
        for candidate in range(n):
            rows.extend((constraint, constraint))
            columns.extend(
                (target * n + candidate, assignment_variables + candidate)
            )
            values.extend((1.0, -1.0))
            lower.append(-np.inf)
            upper_bounds.append(0.0)
            constraint += 1
    for candidate in range(n):
        rows.append(constraint)
        columns.append(assignment_variables + candidate)
        values.append(1.0)
    lower.append(float(k))
    upper_bounds.append(float(k))
    constraint += 1

    matrix = coo_matrix(
        (values, (rows, columns)), shape=(constraint, variables), dtype=np.float64
    ).tocsr()
    result = milp(
        c=objective,
        integrality=integrality,
        bounds=Bounds(np.zeros(variables), np.ones(variables)),
        constraints=LinearConstraint(
            matrix,
            np.asarray(lower, dtype=np.float64),
            np.asarray(upper_bounds, dtype=np.float64),
        ),
        options={"presolve": True, "mip_rel_gap": 0.0},
    )
    if not result.success or result.x is None:
        raise RuntimeError(f"exact k-medoids optimization failed: {result.message}")
    selected = np.flatnonzero(result.x[assignment_variables:] > 0.5)
    if len(selected) != k:
        raise RuntimeError("exact k-medoids returned the wrong panel size")
    return selected.astype(np.int64)


def random_panel_controls(
    distance: np.ndarray,
    k: int,
    repeats: int,
    seed: int,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    panels = np.asarray(
        [np.sort(rng.choice(len(distance), size=k, replace=False)) for _ in range(repeats)],
        dtype=np.int16,
    )
    return evaluate_random_panels(distance, panels)


def evaluate_random_panels(
    distance: np.ndarray,
    panels: np.ndarray,
) -> dict[str, np.ndarray]:
    panels = np.asarray(panels, dtype=np.int64)
    if panels.ndim != 2 or panels.shape[1] < 1:
        raise ValueError("random panels must be a nonempty two-dimensional array")
    repeats = len(panels)
    means = np.empty(repeats, dtype=np.float64)
    maxima = np.empty(repeats, dtype=np.float64)
    for repetition, panel in enumerate(panels):
        coverage = panel_coverage_objective(distance, panel)
        means[repetition] = coverage["mean_nearest_distance"]
        maxima[repetition] = coverage["maximum_nearest_distance"]
    return {
        "mean_nearest_distance": means,
        "maximum_nearest_distance": maxima,
        "panels": panels,
    }


def edge_recovery_metrics(
    reference: np.ndarray,
    estimate: np.ndarray,
    edge_counts: tuple[int, ...] = DEFAULT_EDGE_K,
) -> dict[str, float]:
    reference_values = upper(reference)
    estimate_values = upper(estimate)
    result = map_comparison(reference, estimate)
    absolute_cut = float(np.quantile(np.abs(reference_values), 0.90))
    strong = np.abs(reference_values) >= absolute_cut
    result["reference_abs_top10pct_edges"] = int(strong.sum())
    result["reference_abs_top10pct_sign_agreement"] = float(
        np.mean(np.sign(reference_values[strong]) == np.sign(estimate_values[strong]))
    )
    for count in edge_counts:
        if count > len(reference_values):
            continue
        for direction, largest in (("positive", True), ("negative", False)):
            reference_mask = stable_top_mask(reference_values, count, largest=largest)
            estimate_mask = stable_top_mask(estimate_values, count, largest=largest)
            overlap = set_overlap(reference_mask, estimate_mask)
            for metric, value in overlap.items():
                result[f"top_{direction}_{count}_{metric}"] = value
    return result


def analyze_recovery(
    *,
    name: str,
    matrix: np.ndarray,
    targets: list[str],
    sizes: tuple[int, ...],
    repetitions: int,
    seed: int,
    cluster_k: tuple[int, ...],
    random_panel_repeats: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    matrix = np.asarray(matrix, dtype=np.float64)
    reference = target_geometry(matrix, "row_centered_residual")
    reference_signed_distance = correlation_distance(reference)
    reference_dependency_distance = dependency_distance(reference)
    reference_labels = {
        k: hierarchical_labels(reference_signed_distance, k) for k in cluster_k
    }
    reference_medoids = {
        k: exact_k_medoids(reference_dependency_distance, k) for k in cluster_k
    }
    random_controls = {
        k: random_panel_controls(
            reference_dependency_distance,
            k,
            random_panel_repeats,
            seed + 10_000 + k,
        )
        for k in cluster_k
    }
    edge_records: list[dict[str, Any]] = []
    decision_records: list[dict[str, Any]] = []
    selection_records: list[dict[str, Any]] = []
    full_residual_sum, full_residual_cross = row_centered_statistics(matrix)
    rng = np.random.default_rng(seed)
    for size in sizes:
        if not 3 <= size < len(matrix):
            raise ValueError("recovery size must lie between 3 and full support-1")
        for repetition in range(repetitions):
            selected_rows = np.sort(rng.choice(len(matrix), size=size, replace=False))
            selected_sum, selected_cross = residual_subset_statistics(
                matrix[selected_rows]
            )
            estimate = correlation_from_sufficient_statistics(
                size, selected_sum, selected_cross
            )
            complement = correlation_from_sufficient_statistics(
                len(matrix) - size,
                full_residual_sum - selected_sum,
                full_residual_cross - selected_cross,
            )
            for reference_scope, comparison_reference in (
                ("full_source_map_including_calibration_rows", reference),
                ("row_disjoint_complement_map", complement),
            ):
                edge_records.append(
                    {
                        "dataset": name,
                        "calibration_ligands": int(size),
                        "repetition": repetition,
                        "reference_scope": reference_scope,
                        **edge_recovery_metrics(comparison_reference, estimate),
                    }
                )
            estimate_signed_distance = correlation_distance(estimate)
            estimate_dependency_distance = dependency_distance(estimate)
            complement_signed_distance = correlation_distance(complement)
            complement_dependency_distance = dependency_distance(complement)
            for k in cluster_k:
                estimated_labels = hierarchical_labels(estimate_signed_distance, k)
                estimated_medoids = deterministic_pam(
                    estimate_dependency_distance, k, targets
                )
                full_panel = reference_medoids[k]
                full_set = set(full_panel.tolist())
                estimated_set = set(estimated_medoids.tolist())
                complement_labels = hierarchical_labels(
                    complement_signed_distance, k
                )
                for reference_scope, signed_labels, evaluation_distance, controls in (
                    (
                        "full_source_map_including_calibration_rows",
                        reference_labels[k],
                        reference_dependency_distance,
                        random_controls[k],
                    ),
                    (
                        "row_disjoint_complement_map",
                        complement_labels,
                        complement_dependency_distance,
                        evaluate_random_panels(
                            complement_dependency_distance,
                            random_controls[k]["panels"][
                                : min(
                                    DEFAULT_COMPLEMENT_RANDOM_PANELS,
                                    len(random_controls[k]["panels"]),
                                )
                            ],
                        ),
                    ),
                ):
                    full_coverage = panel_coverage_objective(
                        evaluation_distance, full_panel
                    )
                    estimated_coverage = panel_coverage_objective(
                        evaluation_distance, estimated_medoids
                    )
                    decision_records.append(
                        {
                            "dataset": name,
                            "calibration_ligands": int(size),
                            "repetition": repetition,
                            "panel_targets_k": k,
                            "reference_scope": reference_scope,
                            "cluster_adjusted_rand_index": float(
                                adjusted_rand_score(signed_labels, estimated_labels)
                            ),
                            "sample_vs_fixed_full_source_panel_medoid_overlap": len(
                                full_set & estimated_set
                            ),
                            "sample_vs_fixed_full_source_panel_medoid_jaccard": float(
                                len(full_set & estimated_set)
                                / len(full_set | estimated_set)
                            ),
                            "fixed_full_source_panel_mean_nearest_distance_on_reference_map": (
                                full_coverage["mean_nearest_distance"]
                            ),
                            "sample_map_medoids_mean_nearest_distance_on_reference_map": (
                                estimated_coverage["mean_nearest_distance"]
                            ),
                            "sample_minus_fixed_full_source_panel_mean_nearest_distance_on_reference_map": float(
                                estimated_coverage["mean_nearest_distance"]
                                - full_coverage["mean_nearest_distance"]
                            ),
                            "sample_map_medoids_maximum_nearest_distance_on_reference_map": (
                                estimated_coverage["maximum_nearest_distance"]
                            ),
                            "random_panels_on_reference_map": int(
                                len(controls["mean_nearest_distance"])
                            ),
                            "fraction_random_panels_with_mean_coverage_no_worse_than_sample": float(
                                np.mean(
                                    controls["mean_nearest_distance"]
                                    <= estimated_coverage["mean_nearest_distance"]
                                )
                            ),
                            "random_mean_nearest_distance_median": float(
                                np.median(controls["mean_nearest_distance"])
                            ),
                            "random_mean_nearest_distance_q025": float(
                                np.quantile(
                                    controls["mean_nearest_distance"], 0.025
                                )
                            ),
                            "random_mean_nearest_distance_q975": float(
                                np.quantile(
                                    controls["mean_nearest_distance"], 0.975
                                )
                            ),
                        }
                    )
                for selection_source, panel in (
                    ("full_map", full_panel),
                    ("sample_map", estimated_medoids),
                ):
                    for target_index in panel:
                        selection_records.append(
                            {
                                "dataset": name,
                                "calibration_ligands": int(size),
                                "repetition": repetition,
                                "panel_targets_k": k,
                                "selection_source": selection_source,
                                "target_index": int(target_index),
                                "target": targets[int(target_index)],
                            }
                        )
    metadata = {
        "dataset": name,
        "full_ligands": int(len(matrix)),
        "targets": int(matrix.shape[1]),
        "target_order": targets,
        "reference_residual_pr": correlation_pr(reference),
        "random_panel_repeats_per_k": int(random_panel_repeats),
        "row_disjoint_complement_random_panels_per_k": int(
            min(DEFAULT_COMPLEMENT_RANDOM_PANELS, random_panel_repeats)
        ),
        "full_map_medoids": {
            str(k): [targets[index] for index in reference_medoids[k]]
            for k in cluster_k
        },
        "clustering": (
            "average-linkage hierarchical clustering of signed distance 1-r, cut to fixed k"
        ),
        "panel_selection": (
            "sample maps use deterministic BUILD+SWAP k-medoids; the full-map reference "
            "is the exact mixed-integer k-medoids optimum. Both minimize total nearest "
            "dependency distance 1-|r|, so positive and negative linear dependence both "
            "count as redundancy"
        ),
        "row_disjoint_reference": (
            "for every replicate, the complement map is computed from every source row "
            "not used in that calibration panel by subtracting exact row-centred sufficient "
            "statistics; edge and cluster metrics compare the calibration map directly with "
            "this row-disjoint reference. Panel coverage evaluates both the sample-selected "
            "panel and the one fixed full-source exact-optimum panel on that complement; the "
            "difference is not regret relative to a complement-specific optimum"
        ),
    }
    return (
        pd.DataFrame.from_records(edge_records),
        pd.DataFrame.from_records(decision_records),
        pd.DataFrame.from_records(selection_records),
        metadata,
    )


def summarize_replicates(
    frame: pd.DataFrame,
    identifiers: list[str],
) -> pd.DataFrame:
    excluded = set(identifiers) | {"repetition"}
    metrics = [
        column
        for column in frame.columns
        if column not in excluded and pd.api.types.is_numeric_dtype(frame[column])
    ]
    records: list[dict[str, Any]] = []
    for keys, group in frame.groupby(identifiers, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        record: dict[str, Any] = dict(zip(identifiers, keys, strict=True))
        record["repetitions"] = int(len(group))
        for metric in metrics:
            values = group[metric].to_numpy(dtype=np.float64)
            record[f"{metric}_mean"] = float(values.mean())
            record[f"{metric}_q025"] = float(np.quantile(values, 0.025))
            record[f"{metric}_q975"] = float(np.quantile(values, 0.975))
        records.append(record)
    return pd.DataFrame.from_records(records)


def threshold_summary(frame: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for (dataset, transformation), group in frame.groupby(
        ["dataset", "transformation"], sort=True
    ):
        records.append(
            {
                "dataset": dataset,
                "transformation": transformation,
                "thresholds_tested": int(len(group)),
                "minimum_tail_fraction": float(group.tail_fraction_per_side.min()),
                "maximum_tail_fraction": float(group.tail_fraction_per_side.max()),
                "minimum_geometry_spearman": float(group.geometry_spearman.min()),
                "maximum_geometry_spearman": float(group.geometry_spearman.max()),
                "median_geometry_spearman": float(group.geometry_spearman.median()),
                "minimum_equal_n_random_control_mean": float(
                    group.control_geometry_spearman_mean.min()
                ),
                "maximum_equal_n_random_control_mean": float(
                    group.control_geometry_spearman_mean.max()
                ),
                "minimum_observed_minus_control_mean": float(
                    group.observed_minus_control_mean_geometry_spearman.min()
                ),
                "maximum_observed_minus_control_mean": float(
                    group.observed_minus_control_mean_geometry_spearman.max()
                ),
                "observed_below_control_q025_at_every_threshold": bool(
                    group.observed_below_control_q025.all()
                ),
            }
        )
    return pd.DataFrame.from_records(records)


def build_readme(
    threshold: pd.DataFrame,
    trends: pd.DataFrame,
    edge_summary: pd.DataFrame,
    decision_summary: pd.DataFrame,
    repetitions: int,
) -> str:
    lines = [
        "# Threshold-robust support shift and decision-level map recovery",
        "",
        "This revision-only artifact is exploratory and outcome-blind. It does not",
        "modify the manuscript or establish biological validity of a Vina map.",
        "",
        "## Molecular-weight support continuum",
        "",
    ]
    for row in threshold.itertuples(index=False):
        lines.append(
            f"- {row.dataset}, {row.transformation}: low/high-tail map agreement ranged "
            f"from {row.minimum_geometry_spearman:.3f} to "
            f"{row.maximum_geometry_spearman:.3f} across "
            f"{row.thresholds_tested} thresholds ({row.minimum_tail_fraction:.2f}--"
            f"{row.maximum_tail_fraction:.2f} per tail); equal-N random-disjoint means "
            f"ranged {row.minimum_equal_n_random_control_mean:.3f}--"
            f"{row.maximum_equal_n_random_control_mean:.3f}, and every observed value "
            f"fell below its matched control 2.5th percentile: "
            f"{row.observed_below_control_q025_at_every_threshold}."
        )
    lines.extend(["", "Across ten non-overlapping MW bins:", ""])
    for row in trends.itertuples(index=False):
        lines.append(
            f"- {row.dataset}, {row.transformation}: Spearman(MW separation, "
            f"map dissimilarity) = "
            f"{row.spearman_mw_separation_vs_map_dissimilarity:.3f}."
        )
    lines.extend(
        [
            "",
            "These are descriptive pairwise trends. The 45 bin pairs reuse ten maps, so",
            "they are not treated as independent inferential observations.",
            "",
            "## Recovery of edges and decisions",
            "",
            f"Each size uses {repetitions} simple-random representative panels. Selected",
            "numbers below are replicate means; CSV files retain central 95% replicate",
            "ranges.",
            "",
        ]
    )
    held_out_edges = edge_summary.loc[
        edge_summary.reference_scope.eq("row_disjoint_complement_map")
    ]
    for row in held_out_edges.itertuples(index=False):
        lines.append(
            f"- {row.dataset}, n={row.calibration_ligands}, row-disjoint complement: "
            f"global rho "
            f"{row.geometry_spearman_mean:.3f}; all-edge sign agreement "
            f"{row.edge_sign_agreement_mean:.3f}; top-positive-10 precision "
            f"{row.top_positive_10_precision_mean:.3f}; top-negative-10 precision "
            f"{row.top_negative_10_precision_mean:.3f}."
        )
    lines.extend(["", "For the fixed k=8 decision example:", ""])
    selected_decisions = decision_summary.loc[
        decision_summary.panel_targets_k.eq(8)
        & decision_summary.reference_scope.eq("row_disjoint_complement_map")
    ]
    for row in selected_decisions.itertuples(index=False):
        lines.append(
            f"- {row.dataset}, n={row.calibration_ligands}, row-disjoint complement: "
            f"cluster ARI "
            f"{row.cluster_adjusted_rand_index_mean:.3f}; overlap with the fixed "
            f"full-source panel "
            f"{row.sample_vs_fixed_full_source_panel_medoid_overlap_mean:.2f}/8; "
            f"sample-minus-fixed-full-source-panel coverage difference, with both "
            f"evaluated on the complement, "
            f"{row.sample_minus_fixed_full_source_panel_mean_nearest_distance_on_reference_map_mean:.4f}; "
            f"fraction of random "
            f"panels with no worse mean coverage "
            f"{row.fraction_random_panels_with_mean_coverage_no_worse_than_sample_mean:.3f}."
        )
    lines.extend(
        [
            "",
            "The medoid exercise is a decision-level demonstration for compressing the",
            "tested Vina target map. It does not show that the selected targets maximize",
            "biological coverage or experimental selectivity. On the row-disjoint",
            "complement, the comparator is the fixed full-source panel, not a separately",
            "optimized complement panel; the displayed difference is therefore not",
            "held-out optimality regret.",
            "",
            "## Reproduction",
            "",
            "```bash",
            ".venv/bin/python analysis/revision_map_decision_sensitivities.py \\",
            f"  --repetitions {repetitions} --sizes 200,500 \\",
            "  --mw-null-repetitions 100 \\",
            "  --cluster-k 6,8,10 \\",
            "  --output-dir results/revision_map_decision_sensitivities",
            ".venv/bin/python -m pytest -q analysis/test_revision_map_decision_sensitivities.py",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def parse_int_tuple(value: str) -> tuple[int, ...]:
    result = tuple(int(item) for item in value.split(",") if item)
    if not result or len(set(result)) != len(result):
        raise argparse.ArgumentTypeError("expected distinct comma-separated integers")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=parse_int_tuple, default=DEFAULT_RECOVERY_SIZES)
    parser.add_argument("--cluster-k", type=parse_int_tuple, default=DEFAULT_CLUSTER_K)
    parser.add_argument("--repetitions", type=int, default=100)
    parser.add_argument("--random-panel-repeats", type=int, default=10_000)
    parser.add_argument(
        "--mw-null-repetitions", type=int, default=DEFAULT_MW_NULL_REPETITIONS
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--dockstring-sample-size", type=int, default=15_000)
    parser.add_argument("--dockstring-sample-seed", type=int, default=71)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if (
        args.repetitions < 2
        or args.random_panel_repeats < 100
        or args.mw_null_repetitions < 2
    ):
        raise ValueError(
            "use >=2 recovery/MW-null repeats and >=100 random panel controls"
        )
    docking44 = load_docking44()
    dockstring = load_dockstring(
        AnalysisConfig(
            dockstring_sample_size=args.dockstring_sample_size,
            dockstring_sample_seed=args.dockstring_sample_seed,
        )
    )
    bundles = (docking44, dockstring)

    threshold_observed = pd.concat(
        [mw_threshold_sweep(bundle) for bundle in bundles], ignore_index=True
    )
    threshold_controls = pd.concat(
        [
            mw_equal_n_random_disjoint_controls(
                bundle,
                args.mw_null_repetitions,
                args.seed + 50_000 + 1000 * index,
            )
            for index, bundle in enumerate(bundles)
        ],
        ignore_index=True,
    )
    threshold = attach_mw_control_summary(threshold_observed, threshold_controls)
    threshold_overview = threshold_summary(threshold)
    decile_parts = [mw_decile_continuum(bundle) for bundle in bundles]
    deciles = pd.concat([part[0] for part in decile_parts], ignore_index=True)
    decile_pairs = pd.concat([part[1] for part in decile_parts], ignore_index=True)
    decile_trends = pd.concat([part[2] for part in decile_parts], ignore_index=True)

    recovery_parts = []
    for dataset_index, (bundle, matrix) in enumerate(
        (
            (docking44, docking44.matrix),
            (dockstring, dockstring.spectral_matrix),
        )
    ):
        recovery_parts.append(
            analyze_recovery(
                name=bundle.name,
                matrix=matrix,
                targets=bundle.targets,
                sizes=args.sizes,
                repetitions=args.repetitions,
                seed=args.seed + 1000 * dataset_index,
                cluster_k=args.cluster_k,
                random_panel_repeats=args.random_panel_repeats,
            )
        )
    edge_metrics = pd.concat([part[0] for part in recovery_parts], ignore_index=True)
    decision_metrics = pd.concat(
        [part[1] for part in recovery_parts], ignore_index=True
    )
    selections = pd.concat([part[2] for part in recovery_parts], ignore_index=True)
    edge_summary = summarize_replicates(
        edge_metrics, ["dataset", "calibration_ligands", "reference_scope"]
    )
    decision_summary = summarize_replicates(
        decision_metrics,
        ["dataset", "calibration_ligands", "panel_targets_k", "reference_scope"],
    )

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    tables = {
        "mw_threshold_sweep.csv": threshold,
        "mw_equal_n_random_disjoint_controls.csv": threshold_controls,
        "mw_threshold_sweep_summary.csv": threshold_overview,
        "mw_decile_maps.csv": deciles,
        "mw_decile_pair_distances.csv": decile_pairs,
        "mw_decile_continuous_trends.csv": decile_trends,
        "recovery_edge_metrics.csv": edge_metrics,
        "recovery_edge_summary.csv": edge_summary,
        "recovery_decision_metrics.csv": decision_metrics,
        "recovery_decision_summary.csv": decision_summary,
        "recovery_medoid_selections.csv": selections,
    }
    for filename, frame in tables.items():
        frame.to_csv(output / filename, index=False, float_format="%.17g")

    metadata = {
        "schema_version": "1.2.0",
        "analysis_status": "exploratory_revision_sensitivity",
        "seed": int(args.seed),
        "molecular_weight_support": {
            "Docking-44": docking44.support_rule,
            "DOCKSTRING-58": dockstring.support_rule,
            "threshold_tail_fractions": list(DEFAULT_TAIL_FRACTIONS),
            "equal_n_random_disjoint_repetitions_per_threshold": int(
                args.mw_null_repetitions
            ),
            "quantile_bins": 10,
            "threshold_rule": (
                "exact rank tails with equal N; sort by molecular weight then source-row "
                "index, so boundary ties are deterministic"
            ),
            "continuum_rule": (
                "ten non-overlapping equal-N rank windows; any N mod 10 highest-rank "
                "remainder rows are excluded and counted in mw_decile_maps.csv"
            ),
            "chronology": (
                "molecular weight was selected after preliminary inspection; the sweep "
                "and continuous decile analysis were added in response to review"
            ),
        },
        "recovery": {
            "sizes": list(args.sizes),
            "repetitions_per_size": int(args.repetitions),
            "sampling": "simple random without replacement from each full source support",
            "cluster_k_sensitivity": list(args.cluster_k),
            "edge_counts": list(DEFAULT_EDGE_K),
            "random_panel_repeats_per_dataset_and_k": int(
                args.random_panel_repeats
            ),
            "datasets": [part[3] for part in recovery_parts],
        },
        "claim_boundary": (
            "These analyses show threshold robustness and within-source recovery of Vina "
            "target-map features and a Vina-defined panel-compression decision. They do "
            "not identify molecular weight as causal, establish transport to a shifted "
            "chemical library, validate docking poses, or establish biological optimality "
            "of the selected medoids."
        ),
    }
    write_json(output / "metadata.json", metadata)
    (output / "README.md").write_text(
        build_readme(
            threshold_overview,
            decile_trends,
            edge_summary,
            decision_summary,
            args.repetitions,
        )
    )
    checksum_files = [*tables, "metadata.json", "README.md"]
    write_json(
        output / "output_checksums.json",
        {
            "algorithm": "sha256",
            "files": {
                filename: sha256_file(output / filename)
                for filename in sorted(checksum_files)
            },
        },
    )
    print(threshold_overview.to_string(index=False))
    print(decile_trends.to_string(index=False))
    print(edge_summary.to_string(index=False))
    print(
        decision_summary.loc[decision_summary.panel_targets_k.eq(8)].to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()
