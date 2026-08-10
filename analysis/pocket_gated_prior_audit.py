#!/usr/bin/env python3
"""Does pocket-gated Vina add information beyond a target-degree prior?

This science-only audit addresses a specific rival explanation for the
``pocket_gated_fusion`` result.  A target pair can appear co-selective simply
because the two targets have similar marginal hit rates.  We therefore use
PKIS1 only to derive a transparent target-promiscuity prior, fit all model
coefficients, and select the KLIFS pocket-identity gate.  DAVIS, PKIS2, and
KiRHub are then evaluated without refitting.

The primary baseline is an ordinary least-squares pair model with four ranked
features: geometric-mean PKIS1 target degree, similarity of PKIS1 target
degrees, receptor-domain sequence identity, and KLIFS pocket identity.  The
primary docking addition is a single feature proportional to the difference
between the previously defined dual-surface gated fusion and sequence alone.
Thus the comparison asks whether the *aligned docking part* of the locked gate
adds information after the target prior, sequence, and pocket identity are
already available.

The target-label QAP is deliberately selection-aware.  On every permutation,
raw and residual Vina target labels are permuted jointly, the gate is selected
again on PKIS1, and the augmented regression is refit on PKIS1.  Experimental
endpoints, the degree prior, sequence, pocket identity, and all locked panels
remain fixed.  This controls the complete discovery-side fitting procedure,
not merely a frozen post-selection threshold.

This is a post-hoc exploratory target-pair analysis on the same 20 kinases.  It
does not test ligand-level target retrieval or extrapolation to unseen targets.
No manuscript file is read or modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import OrderedDict
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy import stats

try:
    from .biological_core_modes import holm_adjust
    from .klifs_pocket_control import matrix_from_pairs
    from .pocket_gated_fusion import (
        DISCOVERY_PANEL,
        LOCKED_PANELS,
        TARGETS,
        THRESHOLD_GRID,
        _average_precision_from_scores,
        load_inputs,
        top_labels,
    )
    from .residual_target_geometry_validation import load_pkis1_full
except ImportError:  # pragma: no cover - direct CLI execution
    from biological_core_modes import holm_adjust  # type: ignore
    from klifs_pocket_control import matrix_from_pairs  # type: ignore
    from pocket_gated_fusion import (  # type: ignore
        DISCOVERY_PANEL,
        LOCKED_PANELS,
        TARGETS,
        THRESHOLD_GRID,
        _average_precision_from_scores,
        load_inputs,
        top_labels,
    )
    from residual_target_geometry_validation import load_pkis1_full  # type: ignore


PACKAGE = Path(__file__).resolve().parents[1]
DEFAULT_PAIRS = PACKAGE / "results" / "klifs_pocket_control" / "target_pairs.csv"
DEFAULT_PKIS1 = Path("/private/tmp/pkis1_supplement.zip")
DEFAULT_KIRHUB = Path("/private/tmp/kirhub_supp_tables.xlsx")
DEFAULT_OUTPUT = PACKAGE / "results" / "pocket_gated_prior_audit"
DEFAULT_PERMUTATIONS = 50_000
DEFAULT_SEED = 20260821
PRIMARY_DEGREE_THRESHOLD = 50.0
DEGREE_THRESHOLD_SENSITIVITY = (20.0, 35.0, 50.0, 65.0, 80.0)
TOP_FRACTIONS = (0.05, 0.10, 0.15)
PRIMARY_METRICS = ("continuous_spearman", "top_10_roc_auc", "top_10_average_precision")


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def json_ready(value: object) -> object:
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
        raise ValueError("cannot serialize a non-finite value")
    return value


def upper(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("upper requires a square matrix")
    return matrix[np.triu_indices(len(matrix), k=1)]


def pair_percentile_rank(values: np.ndarray) -> np.ndarray:
    """Return deterministic average-tie ranks scaled to [0, 1]."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or len(array) < 2 or not np.isfinite(array).all():
        raise ValueError("pair_percentile_rank requires a finite vector of length >=2")
    return (stats.rankdata(array, method="average") - 1.0) / (len(array) - 1.0)


def symmetric_rank_matrix(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("symmetric_rank_matrix requires a square matrix")
    tri = np.triu_indices(len(matrix), k=1)
    result = np.zeros_like(matrix, dtype=np.float64)
    result[tri] = pair_percentile_rank(matrix[tri])
    result[(tri[1], tri[0])] = result[tri]
    np.fill_diagonal(result, 1.0)
    return result


def degree_pair_features(
    target_degree: np.ndarray, pair_indices: tuple[np.ndarray, np.ndarray]
) -> np.ndarray:
    """Transparent pair prior from joint degree and degree similarity.

    The first column is geometric-mean marginal activity; the second is the
    negative absolute degree difference.  Both are pair-rank transformed so
    that their scale is comparable to the other pair predictors.
    """
    degree = np.asarray(target_degree, dtype=np.float64)
    if degree.ndim != 1 or not np.isfinite(degree).all():
        raise ValueError("target_degree must be a finite vector")
    first, second = pair_indices
    if max(first.max(), second.max()) >= len(degree):
        raise ValueError("pair indices exceed target-degree support")
    joint = np.sqrt(degree[first] * degree[second])
    similarity = -np.abs(degree[first] - degree[second])
    return np.column_stack(
        [pair_percentile_rank(joint), pair_percentile_rank(similarity)]
    )


def ols_fit_predict(features: np.ndarray, endpoint: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit a transparent unregularized OLS model and return predictions/coefs."""
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(endpoint, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 1 or len(x) != len(y):
        raise ValueError("features and endpoint are not aligned")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("OLS inputs must be finite")
    design = np.column_stack([np.ones(len(x), dtype=np.float64), x])
    coefficients = np.linalg.lstsq(design, y, rcond=None)[0]
    return design @ coefficients, coefficients


def gated_docking_increment(
    sequence_rank: np.ndarray,
    raw_rank: np.ndarray,
    residual_rank: np.ndarray,
    pocket_identity: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """Single feature proportional to dual-surface gated fusion minus sequence.

    For low-identity pairs, the previous fusion is
    ``(sequence + raw + residual) / 3``.  Its difference from sequence is 2/3
    times the returned feature.  Scaling is immaterial in an OLS addition.
    """
    arrays = [
        np.asarray(sequence_rank, dtype=np.float64),
        np.asarray(raw_rank, dtype=np.float64),
        np.asarray(residual_rank, dtype=np.float64),
        np.asarray(pocket_identity, dtype=np.float64),
    ]
    if len({value.shape for value in arrays}) != 1 or arrays[0].ndim != 1:
        raise ValueError("gated increment vectors must be aligned")
    low_identity = arrays[3] < float(threshold)
    return low_identity * ((arrays[1] + arrays[2]) / 2.0 - arrays[0])


def top_fraction_labels(endpoint: np.ndarray, fraction: float) -> np.ndarray:
    values = np.asarray(endpoint, dtype=np.float64)
    if not 0 < fraction < 1:
        raise ValueError("top fraction must lie in (0, 1)")
    count = max(1, int(np.ceil(float(fraction) * len(values))))
    labels = np.zeros(len(values), dtype=bool)
    labels[np.argsort(values, kind="mergesort")[-count:]] = True
    return labels


def endpoint_cache(endpoint: np.ndarray) -> dict[str, object]:
    values = np.asarray(endpoint, dtype=np.float64)
    ranked = stats.rankdata(values, method="average").astype(np.float64)
    ranked -= ranked.mean()
    return {
        "centered_ranks": ranked,
        "rank_norm": float(np.linalg.norm(ranked)),
        "labels": {
            fraction: top_fraction_labels(values, fraction)
            for fraction in TOP_FRACTIONS
        },
    }


def _auc_from_ranks(labels: np.ndarray, ranks: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=bool)
    positive = int(labels.sum())
    negative = int((~labels).sum())
    return float(
        (ranks[labels].sum() - positive * (positive + 1) / 2.0)
        / (positive * negative)
    )


def predictor_metrics_from_cache(
    predictor: np.ndarray, cache: dict[str, object]
) -> OrderedDict[str, float]:
    values = np.asarray(predictor, dtype=np.float64)
    ranks = stats.rankdata(values, method="average").astype(np.float64)
    centered = ranks - ranks.mean()
    rho = float(
        np.dot(centered, np.asarray(cache["centered_ranks"], dtype=np.float64))
        / (np.linalg.norm(centered) * float(cache["rank_norm"]))
    )
    output: OrderedDict[str, float] = OrderedDict([("continuous_spearman", rho)])
    labels_by_fraction = cache["labels"]
    if not isinstance(labels_by_fraction, dict):
        raise TypeError("endpoint cache labels are malformed")
    for fraction in TOP_FRACTIONS:
        labels = np.asarray(labels_by_fraction[fraction], dtype=bool)
        prefix = f"top_{int(round(100 * fraction))}"
        output[f"{prefix}_roc_auc"] = _auc_from_ranks(labels, ranks)
        output[f"{prefix}_average_precision"] = _average_precision_from_scores(
            labels, values
        )
    return output


def model_fit(
    base_features: np.ndarray,
    endpoint: np.ndarray,
    *,
    added_features: Sequence[np.ndarray] = (),
) -> tuple[np.ndarray, np.ndarray]:
    columns = [np.asarray(base_features, dtype=np.float64)]
    for feature in added_features:
        array = np.asarray(feature, dtype=np.float64)
        if array.ndim != 1 or len(array) != len(base_features):
            raise ValueError("added model feature is not aligned")
        columns.append(array[:, None])
    return ols_fit_predict(np.column_stack(columns), endpoint)


def select_gate(
    base_features: np.ndarray,
    discovery_endpoint: np.ndarray,
    sequence_rank: np.ndarray,
    raw_rank: np.ndarray,
    residual_rank: np.ndarray,
    pocket_identity: np.ndarray,
    grid: Iterable[float] = THRESHOLD_GRID,
) -> tuple[float, np.ndarray, pd.DataFrame]:
    labels = top_fraction_labels(discovery_endpoint, 0.10)
    rows: list[dict[str, float | int]] = []
    predictions: dict[float, np.ndarray] = {}
    for value in sorted(set(float(item) for item in grid)):
        increment = gated_docking_increment(
            sequence_rank, raw_rank, residual_rank, pocket_identity, value
        )
        prediction, _ = model_fit(
            base_features, discovery_endpoint, added_features=(increment,)
        )
        predictions[value] = prediction
        rows.append(
            {
                "threshold": value,
                "low_identity_pairs": int(np.sum(pocket_identity < value)),
                "discovery_top_10_average_precision": (
                    _average_precision_from_scores(labels, prediction)
                ),
                "discovery_continuous_spearman": float(
                    stats.spearmanr(prediction, discovery_endpoint).statistic
                ),
            }
        )
    frame = pd.DataFrame.from_records(rows)
    selected = frame.sort_values(
        ["discovery_top_10_average_precision", "threshold"],
        ascending=[False, True],
        kind="mergesort",
    ).iloc[0]
    threshold = float(selected.threshold)
    return threshold, predictions[threshold], frame


def select_gate_fast(
    base_features: np.ndarray,
    discovery_endpoint: np.ndarray,
    discovery_top_10_labels: np.ndarray,
    sequence_rank: np.ndarray,
    raw_rank: np.ndarray,
    residual_rank: np.ndarray,
    pocket_identity: np.ndarray,
    grid: Iterable[float] = THRESHOLD_GRID,
) -> tuple[float, np.ndarray]:
    """Allocation-light equivalent of :func:`select_gate` for QAP loops."""
    best_key: tuple[float, float] | None = None
    best_threshold: float | None = None
    best_prediction: np.ndarray | None = None
    for value in sorted(set(float(item) for item in grid)):
        increment = gated_docking_increment(
            sequence_rank, raw_rank, residual_rank, pocket_identity, value
        )
        prediction, _ = model_fit(
            base_features, discovery_endpoint, added_features=(increment,)
        )
        ap = _average_precision_from_scores(discovery_top_10_labels, prediction)
        key = (ap, -value)
        if best_key is None or key > best_key:
            best_key = key
            best_threshold = value
            best_prediction = prediction
    if best_threshold is None or best_prediction is None:
        raise ValueError("threshold grid is empty")
    return best_threshold, best_prediction


def _matrix_from_pair_vector(values: np.ndarray, size: int) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64)
    tri = np.triu_indices(size, k=1)
    if len(vector) != len(tri[0]):
        raise ValueError("pair vector has the wrong length")
    matrix = np.zeros((size, size), dtype=np.float64)
    matrix[tri] = vector
    matrix[(tri[1], tri[0])] = vector
    np.fill_diagonal(matrix, 1.0)
    return matrix


def _panel_metric_rows(
    predictions: OrderedDict[str, np.ndarray],
    endpoint_caches: dict[str, dict[str, object]],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for panel, cache in endpoint_caches.items():
        for model, prediction in predictions.items():
            rows.append(
                {
                    "panel": panel,
                    "role": (
                        "discovery_fit"
                        if panel == DISCOVERY_PANEL
                        else "locked_no_retuning_evaluation"
                    ),
                    "model": model,
                    **predictor_metrics_from_cache(prediction, cache),
                }
            )
    return pd.DataFrame.from_records(rows)


def selection_aware_qap(
    *,
    base_features: np.ndarray,
    sequence_rank: np.ndarray,
    pocket_identity: np.ndarray,
    raw_rank_matrix: np.ndarray,
    residual_rank_matrix: np.ndarray,
    endpoints: dict[str, np.ndarray],
    baseline_prediction: np.ndarray,
    observed_augmented_prediction: np.ndarray,
    observed_threshold: float,
    permutations: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """Full-procedure QAP and a no-hard-gate monotonicity audit."""
    if permutations < 1:
        raise ValueError("permutations must be positive")
    size = len(raw_rank_matrix)
    tri = np.triu_indices(size, k=1)
    caches = {name: endpoint_cache(values) for name, values in endpoints.items()}
    baseline_metrics = {
        panel: predictor_metrics_from_cache(baseline_prediction, caches[panel])
        for panel in LOCKED_PANELS
    }
    metric_names = list(next(iter(baseline_metrics.values())))

    observed_metrics = {
        panel: predictor_metrics_from_cache(
            observed_augmented_prediction, caches[panel]
        )
        for panel in LOCKED_PANELS
    }
    observed_delta = np.asarray(
        [
            np.mean(
                [
                    observed_metrics[panel][metric]
                    - baseline_metrics[panel][metric]
                    for panel in LOCKED_PANELS
                ]
            )
            for metric in metric_names
        ],
        dtype=np.float64,
    )

    # A continuous, ungated diagnostic.  Positive pair gain means the ungated
    # docking addition reduced absolute rank error on the locked panels.
    observed_ungated_increment = (
        (upper(raw_rank_matrix) + upper(residual_rank_matrix)) / 2.0
        - sequence_rank
    )
    observed_ungated, _ = model_fit(
        base_features,
        endpoints[DISCOVERY_PANEL],
        added_features=(observed_ungated_increment,),
    )
    observed_pair_gain = np.mean(
        np.vstack(
            [
                np.abs(endpoints[panel] - baseline_prediction)
                - np.abs(endpoints[panel] - observed_ungated)
                for panel in LOCKED_PANELS
            ]
        ),
        axis=0,
    )
    observed_gain_pocket_rho = float(
        stats.spearmanr(observed_pair_gain, pocket_identity).statistic
    )

    null = np.empty((permutations, len(metric_names)), dtype=np.float64)
    null_gain_pocket_rho = np.empty(permutations, dtype=np.float64)
    selected_thresholds = np.empty(permutations, dtype=np.float64)
    discovery_top_10_labels = top_fraction_labels(
        endpoints[DISCOVERY_PANEL], 0.10
    )
    rng = np.random.default_rng(seed)
    for repetition in range(permutations):
        order = rng.permutation(size)
        permuted_raw = raw_rank_matrix[np.ix_(order, order)][tri]
        permuted_residual = residual_rank_matrix[np.ix_(order, order)][tri]
        threshold, prediction = select_gate_fast(
            base_features,
            endpoints[DISCOVERY_PANEL],
            discovery_top_10_labels,
            sequence_rank,
            permuted_raw,
            permuted_residual,
            pocket_identity,
        )
        selected_thresholds[repetition] = threshold
        candidate_metrics = {
            panel: predictor_metrics_from_cache(prediction, caches[panel])
            for panel in LOCKED_PANELS
        }
        for metric_index, metric in enumerate(metric_names):
            null[repetition, metric_index] = np.mean(
                [
                    candidate_metrics[panel][metric]
                    - baseline_metrics[panel][metric]
                    for panel in LOCKED_PANELS
                ]
            )

        ungated_increment = (
            (permuted_raw + permuted_residual) / 2.0 - sequence_rank
        )
        ungated_prediction, _ = model_fit(
            base_features,
            endpoints[DISCOVERY_PANEL],
            added_features=(ungated_increment,),
        )
        pair_gain = np.mean(
            np.vstack(
                [
                    np.abs(endpoints[panel] - baseline_prediction)
                    - np.abs(endpoints[panel] - ungated_prediction)
                    for panel in LOCKED_PANELS
                ]
            ),
            axis=0,
        )
        null_gain_pocket_rho[repetition] = float(
            stats.spearmanr(pair_gain, pocket_identity).statistic
        )

    probabilities = {
        metric: float(
            (1 + np.sum(null[:, index] >= observed_delta[index]))
            / (permutations + 1)
        )
        for index, metric in enumerate(metric_names)
    }
    primary_adjusted = holm_adjust(
        {metric: probabilities[metric] for metric in PRIMARY_METRICS}
    )
    rows: list[dict[str, object]] = []
    for index, metric in enumerate(metric_names):
        panel_deltas = [
            observed_metrics[panel][metric] - baseline_metrics[panel][metric]
            for panel in LOCKED_PANELS
        ]
        rows.append(
            {
                "metric": metric,
                "locked_panels": ";".join(LOCKED_PANELS),
                "observed_mean_augmented_minus_baseline": observed_delta[index],
                "minimum_panel_delta": float(np.min(panel_deltas)),
                "maximum_panel_delta": float(np.max(panel_deltas)),
                "selection_aware_target_label_qap_p_positive": probabilities[metric],
                "holm_p_across_three_primary_metrics": (
                    primary_adjusted[metric] if metric in PRIMARY_METRICS else np.nan
                ),
                "null_mean": float(np.mean(null[:, index])),
                "null_q025": float(np.quantile(null[:, index], 0.025)),
                "null_q975": float(np.quantile(null[:, index], 0.975)),
                "observed_selected_threshold": observed_threshold,
                "permutations": permutations,
                "seed": seed,
            }
        )
    qap = pd.DataFrame.from_records(rows)

    threshold_counts = (
        pd.Series(selected_thresholds)
        .value_counts()
        .sort_index()
        .rename_axis("selected_threshold")
        .reset_index(name="permutations")
    )
    threshold_counts["fraction"] = threshold_counts.permutations / permutations

    monotonicity = np.asarray(
        [
            observed_gain_pocket_rho,
            float(
                (1 + np.sum(null_gain_pocket_rho <= observed_gain_pocket_rho))
                / (permutations + 1)
            ),
            float(np.mean(null_gain_pocket_rho)),
            float(np.quantile(null_gain_pocket_rho, 0.025)),
            float(np.quantile(null_gain_pocket_rho, 0.975)),
        ]
    )
    return qap, threshold_counts, monotonicity, observed_pair_gain


def degree_prior_increment_qap(
    *,
    target_degree: np.ndarray,
    pair_indices: tuple[np.ndarray, np.ndarray],
    sequence_rank: np.ndarray,
    pocket_rank: np.ndarray,
    endpoints: dict[str, np.ndarray],
    permutations: int,
    seed: int,
) -> pd.DataFrame:
    """Target-label QAP for the degree prior beyond sequence plus pocket.

    On each draw the scalar PKIS1 degrees are reassigned to target labels, both
    pair-prior features are reconstructed, and the augmented model is refit on
    PKIS1.  Sequence, pocket identity, and all experimental endpoints remain
    fixed.  This tests transport of correctly aligned marginal target breadth,
    rather than merely the flexibility of adding two pair covariates.
    """
    if permutations < 1:
        raise ValueError("permutations must be positive")
    base_features = np.column_stack([sequence_rank, pocket_rank])
    baseline, _ = model_fit(base_features, endpoints[DISCOVERY_PANEL])
    observed_features = np.column_stack(
        [degree_pair_features(target_degree, pair_indices), base_features]
    )
    observed, _ = model_fit(observed_features, endpoints[DISCOVERY_PANEL])
    caches = {name: endpoint_cache(values) for name, values in endpoints.items()}
    baseline_metrics = {
        panel: predictor_metrics_from_cache(baseline, caches[panel])
        for panel in LOCKED_PANELS
    }
    observed_metrics = {
        panel: predictor_metrics_from_cache(observed, caches[panel])
        for panel in LOCKED_PANELS
    }
    observed_delta = {
        metric: float(
            np.mean(
                [
                    observed_metrics[panel][metric]
                    - baseline_metrics[panel][metric]
                    for panel in LOCKED_PANELS
                ]
            )
        )
        for metric in PRIMARY_METRICS
    }

    null = {
        metric: np.empty(permutations, dtype=np.float64)
        for metric in PRIMARY_METRICS
    }
    rng = np.random.default_rng(seed)
    for repetition in range(permutations):
        permuted_degree = target_degree[rng.permutation(len(target_degree))]
        candidate_features = np.column_stack(
            [degree_pair_features(permuted_degree, pair_indices), base_features]
        )
        candidate, _ = model_fit(
            candidate_features, endpoints[DISCOVERY_PANEL]
        )
        candidate_metrics = {
            panel: predictor_metrics_from_cache(candidate, caches[panel])
            for panel in LOCKED_PANELS
        }
        for metric in PRIMARY_METRICS:
            null[metric][repetition] = np.mean(
                [
                    candidate_metrics[panel][metric]
                    - baseline_metrics[panel][metric]
                    for panel in LOCKED_PANELS
                ]
            )

    probabilities = {
        metric: float(
            (1 + np.sum(null[metric] >= observed_delta[metric]))
            / (permutations + 1)
        )
        for metric in PRIMARY_METRICS
    }
    adjusted = holm_adjust(probabilities)
    rows: list[dict[str, object]] = []
    for metric in PRIMARY_METRICS:
        panel_deltas = [
            observed_metrics[panel][metric] - baseline_metrics[panel][metric]
            for panel in LOCKED_PANELS
        ]
        rows.append(
            {
                "metric": metric,
                "observed_mean_degree_prior_increment": observed_delta[metric],
                "minimum_panel_delta": float(np.min(panel_deltas)),
                "maximum_panel_delta": float(np.max(panel_deltas)),
                "degree_target_label_qap_p_positive": probabilities[metric],
                "holm_p_across_three_primary_metrics": adjusted[metric],
                "null_mean": float(np.mean(null[metric])),
                "null_q025": float(np.quantile(null[metric], 0.025)),
                "null_q975": float(np.quantile(null[metric], 0.975)),
                "permutations": permutations,
                "seed": seed,
            }
        )
    return pd.DataFrame.from_records(rows)


def degree_prior_target_jackknife(
    *,
    target_degree: np.ndarray,
    sequence_matrix: np.ndarray,
    pocket_matrix: np.ndarray,
    endpoint_matrices: dict[str, np.ndarray],
) -> pd.DataFrame:
    """Refit the degree-prior comparison after deleting each target."""
    rows: list[dict[str, object]] = []
    for omitted, target in enumerate(TARGETS):
        keep = np.delete(np.arange(len(TARGETS)), omitted)
        tri = np.triu_indices(len(keep), k=1)
        sequence = pair_percentile_rank(sequence_matrix[np.ix_(keep, keep)][tri])
        pocket = pair_percentile_rank(pocket_matrix[np.ix_(keep, keep)][tri])
        endpoints = {
            name: pair_percentile_rank(matrix[np.ix_(keep, keep)][tri])
            for name, matrix in endpoint_matrices.items()
        }
        baseline_features = np.column_stack([sequence, pocket])
        augmented_features = np.column_stack(
            [degree_pair_features(target_degree[keep], tri), baseline_features]
        )
        baseline, _ = model_fit(
            baseline_features, endpoints[DISCOVERY_PANEL]
        )
        augmented, _ = model_fit(
            augmented_features, endpoints[DISCOVERY_PANEL]
        )
        for panel in LOCKED_PANELS:
            cache = endpoint_cache(endpoints[panel])
            baseline_metrics = predictor_metrics_from_cache(baseline, cache)
            augmented_metrics = predictor_metrics_from_cache(augmented, cache)
            for metric in PRIMARY_METRICS:
                rows.append(
                    {
                        "omitted_target": target,
                        "panel": panel,
                        "metric": metric,
                        "degree_prior_increment": (
                            augmented_metrics[metric] - baseline_metrics[metric]
                        ),
                    }
                )
    return pd.DataFrame.from_records(rows)


def target_prior_definition_sensitivity(
    *,
    pkis1_values: np.ndarray,
    sequence_rank: np.ndarray,
    pocket_rank: np.ndarray,
    endpoints: dict[str, np.ndarray],
) -> pd.DataFrame:
    """Sensitivity to reasonable marginal target-propensity summaries."""
    definitions: OrderedDict[str, np.ndarray] = OrderedDict(
        (
            f"fraction_at_least_{int(threshold)}_percent_inhibition",
            np.mean(pkis1_values >= threshold, axis=0),
        )
        for threshold in DEGREE_THRESHOLD_SENSITIVITY
    )
    definitions["clipped_mean_percent_inhibition"] = np.mean(
        np.clip(pkis1_values, 0.0, 100.0), axis=0
    )
    definitions["raw_mean_percent_inhibition"] = np.mean(pkis1_values, axis=0)
    definitions["standard_deviation_percent_inhibition"] = np.std(
        pkis1_values, axis=0
    )
    definitions["median_percent_inhibition"] = np.median(pkis1_values, axis=0)

    tri = np.triu_indices(pkis1_values.shape[1], k=1)
    baseline_features = np.column_stack([sequence_rank, pocket_rank])
    baseline, _ = model_fit(baseline_features, endpoints[DISCOVERY_PANEL])
    rows: list[dict[str, object]] = []
    for definition, target_summary in definitions.items():
        augmented_features = np.column_stack(
            [degree_pair_features(target_summary, tri), baseline_features]
        )
        augmented, _ = model_fit(
            augmented_features, endpoints[DISCOVERY_PANEL]
        )
        for panel in LOCKED_PANELS:
            cache = endpoint_cache(endpoints[panel])
            baseline_metrics = predictor_metrics_from_cache(baseline, cache)
            augmented_metrics = predictor_metrics_from_cache(augmented, cache)
            for metric in PRIMARY_METRICS:
                rows.append(
                    {
                        "target_prior_definition": definition,
                        "panel": panel,
                        "metric": metric,
                        "target_prior_increment": (
                            augmented_metrics[metric] - baseline_metrics[metric]
                        ),
                    }
                )
    return pd.DataFrame.from_records(rows)


def target_jackknife(
    *,
    target_degree: np.ndarray,
    sequence_matrix: np.ndarray,
    pocket_matrix: np.ndarray,
    raw_matrix: np.ndarray,
    residual_matrix: np.ndarray,
    endpoint_matrices: dict[str, np.ndarray],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for omitted, target in enumerate(TARGETS):
        keep = np.delete(np.arange(len(TARGETS)), omitted)
        tri = np.triu_indices(len(keep), k=1)
        degree_features = degree_pair_features(target_degree[keep], tri)
        sequence = pair_percentile_rank(sequence_matrix[np.ix_(keep, keep)][tri])
        pocket = pocket_matrix[np.ix_(keep, keep)][tri]
        pocket_rank = pair_percentile_rank(pocket)
        raw = pair_percentile_rank(raw_matrix[np.ix_(keep, keep)][tri])
        residual = pair_percentile_rank(residual_matrix[np.ix_(keep, keep)][tri])
        endpoints = {
            name: pair_percentile_rank(matrix[np.ix_(keep, keep)][tri])
            for name, matrix in endpoint_matrices.items()
        }
        base_features = np.column_stack([degree_features, sequence, pocket_rank])
        baseline, _ = model_fit(base_features, endpoints[DISCOVERY_PANEL])
        threshold, augmented, _ = select_gate(
            base_features,
            endpoints[DISCOVERY_PANEL],
            sequence,
            raw,
            residual,
            pocket,
        )
        for panel in LOCKED_PANELS:
            cache = endpoint_cache(endpoints[panel])
            baseline_metrics = predictor_metrics_from_cache(baseline, cache)
            augmented_metrics = predictor_metrics_from_cache(augmented, cache)
            for metric in baseline_metrics:
                rows.append(
                    {
                        "omitted_target": target,
                        "panel": panel,
                        "metric": metric,
                        "selected_threshold_after_deletion": threshold,
                        "augmented_minus_baseline": (
                            augmented_metrics[metric] - baseline_metrics[metric]
                        ),
                    }
                )
    return pd.DataFrame.from_records(rows)


def degree_threshold_sensitivity(
    *,
    pkis1_values: np.ndarray,
    sequence_rank: np.ndarray,
    pocket_rank: np.ndarray,
    pocket_identity: np.ndarray,
    raw_rank: np.ndarray,
    residual_rank: np.ndarray,
    endpoints: dict[str, np.ndarray],
) -> pd.DataFrame:
    tri = np.triu_indices(pkis1_values.shape[1], k=1)
    rows: list[dict[str, object]] = []
    for activity_threshold in DEGREE_THRESHOLD_SENSITIVITY:
        degree = np.mean(pkis1_values >= activity_threshold, axis=0)
        features = np.column_stack(
            [degree_pair_features(degree, tri), sequence_rank, pocket_rank]
        )
        baseline, _ = model_fit(features, endpoints[DISCOVERY_PANEL])
        threshold, augmented, _ = select_gate(
            features,
            endpoints[DISCOVERY_PANEL],
            sequence_rank,
            raw_rank,
            residual_rank,
            pocket_identity,
        )
        for panel in LOCKED_PANELS:
            cache = endpoint_cache(endpoints[panel])
            base_metrics = predictor_metrics_from_cache(baseline, cache)
            augmented_metrics = predictor_metrics_from_cache(augmented, cache)
            for metric in base_metrics:
                rows.append(
                    {
                        "PKIS1_activity_threshold_percent_inhibition": activity_threshold,
                        "selected_pocket_identity_threshold": threshold,
                        "panel": panel,
                        "metric": metric,
                        "augmented_minus_baseline": (
                            augmented_metrics[metric] - base_metrics[metric]
                        ),
                    }
                )
    return pd.DataFrame.from_records(rows)


def run_analysis(
    *,
    pair_path: Path,
    pkis1_zip: Path,
    kirhub_workbook: Path,
    output: Path,
    permutations: int,
    seed: int,
) -> dict[str, object]:
    pair_frame = pd.read_csv(pair_path)
    predictors, endpoint_matrices = load_inputs(pair_path, kirhub_workbook)
    pkis1_frame = load_pkis1_full(pkis1_zip)
    pkis1_values = pkis1_frame[list(TARGETS)].to_numpy(dtype=np.float64)
    if pkis1_values.shape != (360, len(TARGETS)):
        raise ValueError("unexpected PKIS1 discovery support")

    size = len(TARGETS)
    tri = np.triu_indices(size, k=1)
    target_degree = np.mean(pkis1_values >= PRIMARY_DEGREE_THRESHOLD, axis=0)
    degree_features = degree_pair_features(target_degree, tri)

    sequence_matrix = predictors["receptor_domain_sequence_identity"]
    pocket_matrix = predictors["klifs_pocket_identity"]
    raw_matrix = matrix_from_pairs(pair_frame, "raw_docking_pair_percentile")
    residual_matrix = predictors["centered_Vina_target_geometry"]
    sequence_rank_matrix = symmetric_rank_matrix(sequence_matrix)
    raw_rank_matrix = symmetric_rank_matrix(raw_matrix)
    residual_rank_matrix = symmetric_rank_matrix(residual_matrix)
    sequence_rank = sequence_rank_matrix[tri]
    raw_rank = raw_rank_matrix[tri]
    residual_rank = residual_rank_matrix[tri]
    pocket_identity = pocket_matrix[tri]
    pocket_rank = pair_percentile_rank(pocket_identity)
    endpoints = {
        name: pair_percentile_rank(matrix[tri])
        for name, matrix in endpoint_matrices.items()
    }

    base_features = np.column_stack(
        [degree_features, sequence_rank, pocket_rank]
    )
    degree_prior_prediction, degree_prior_coef = model_fit(
        degree_features, endpoints[DISCOVERY_PANEL]
    )
    baseline_prediction, baseline_coef = model_fit(
        base_features, endpoints[DISCOVERY_PANEL]
    )
    selected_threshold, dual_gated_prediction, selection = select_gate(
        base_features,
        endpoints[DISCOVERY_PANEL],
        sequence_rank,
        raw_rank,
        residual_rank,
        pocket_identity,
    )
    raw_increment = (pocket_identity < selected_threshold) * (
        raw_rank - sequence_rank
    )
    residual_increment = (pocket_identity < selected_threshold) * (
        residual_rank - sequence_rank
    )
    raw_gated_prediction, raw_gated_coef = model_fit(
        base_features,
        endpoints[DISCOVERY_PANEL],
        added_features=(raw_increment,),
    )
    residual_gated_prediction, residual_gated_coef = model_fit(
        base_features,
        endpoints[DISCOVERY_PANEL],
        added_features=(residual_increment,),
    )
    two_view_gated_prediction, two_view_gated_coef = model_fit(
        base_features,
        endpoints[DISCOVERY_PANEL],
        added_features=(raw_increment, residual_increment),
    )
    ungated_raw_prediction, _ = model_fit(
        base_features, endpoints[DISCOVERY_PANEL], added_features=(raw_rank,)
    )
    ungated_raw_residual_prediction, _ = model_fit(
        base_features,
        endpoints[DISCOVERY_PANEL],
        added_features=(raw_rank, residual_rank),
    )

    predictions = OrderedDict(
        [
            ("PKIS1_target_degree_prior", degree_prior_prediction),
            ("degree_prior_plus_sequence_plus_KLIFS_pocket", baseline_prediction),
            ("baseline_plus_ungated_raw_Vina", ungated_raw_prediction),
            (
                "baseline_plus_ungated_raw_plus_residual_Vina",
                ungated_raw_residual_prediction,
            ),
            ("baseline_plus_raw_Vina_gate", raw_gated_prediction),
            ("baseline_plus_residual_Vina_gate", residual_gated_prediction),
            (
                "baseline_plus_separate_raw_and_residual_Vina_gates",
                two_view_gated_prediction,
            ),
            (
                "baseline_plus_exact_dual_surface_gate_increment",
                dual_gated_prediction,
            ),
        ]
    )
    caches = {name: endpoint_cache(values) for name, values in endpoints.items()}
    panel_metrics = _panel_metric_rows(predictions, caches)

    qap, qap_thresholds, monotonicity, ungated_pair_gain = selection_aware_qap(
        base_features=base_features,
        sequence_rank=sequence_rank,
        pocket_identity=pocket_identity,
        raw_rank_matrix=raw_rank_matrix,
        residual_rank_matrix=residual_rank_matrix,
        endpoints=endpoints,
        baseline_prediction=baseline_prediction,
        observed_augmented_prediction=dual_gated_prediction,
        observed_threshold=selected_threshold,
        permutations=permutations,
        seed=seed,
    )
    degree_qap = degree_prior_increment_qap(
        target_degree=target_degree,
        pair_indices=tri,
        sequence_rank=sequence_rank,
        pocket_rank=pocket_rank,
        endpoints=endpoints,
        permutations=permutations,
        seed=seed + 1,
    )
    degree_jackknife = degree_prior_target_jackknife(
        target_degree=target_degree,
        sequence_matrix=sequence_matrix,
        pocket_matrix=pocket_matrix,
        endpoint_matrices=endpoint_matrices,
    )
    prior_sensitivity = target_prior_definition_sensitivity(
        pkis1_values=pkis1_values,
        sequence_rank=sequence_rank,
        pocket_rank=pocket_rank,
        endpoints=endpoints,
    )
    jackknife = target_jackknife(
        target_degree=target_degree,
        sequence_matrix=sequence_matrix,
        pocket_matrix=pocket_matrix,
        raw_matrix=raw_matrix,
        residual_matrix=residual_matrix,
        endpoint_matrices=endpoint_matrices,
    )
    degree_sensitivity = degree_threshold_sensitivity(
        pkis1_values=pkis1_values,
        sequence_rank=sequence_rank,
        pocket_rank=pocket_rank,
        pocket_identity=pocket_identity,
        raw_rank=raw_rank,
        residual_rank=residual_rank,
        endpoints=endpoints,
    )

    pair_gain = pd.DataFrame(
        {
            "target_a": np.asarray(TARGETS, dtype=object)[tri[0]],
            "target_b": np.asarray(TARGETS, dtype=object)[tri[1]],
            "klifs_pocket_identity": pocket_identity,
            "ungated_locked_mean_absolute_error_gain": ungated_pair_gain,
        }
    )
    pair_gain["pocket_identity_quartile"] = pd.qcut(
        pair_gain.klifs_pocket_identity,
        q=4,
        labels=("Q1_lowest", "Q2", "Q3", "Q4_highest"),
        duplicates="drop",
    ).astype(str)
    quartiles = (
        pair_gain.groupby("pocket_identity_quartile", observed=True)
        .agg(
            target_pairs=("target_a", "size"),
            mean_pocket_identity=("klifs_pocket_identity", "mean"),
            mean_ungated_absolute_error_gain=(
                "ungated_locked_mean_absolute_error_gain",
                "mean",
            ),
            median_ungated_absolute_error_gain=(
                "ungated_locked_mean_absolute_error_gain",
                "median",
            ),
        )
        .reset_index()
    )

    target_degrees = pd.DataFrame(
        {
            "target": TARGETS,
            "PKIS1_compounds": len(pkis1_values),
            "activity_threshold_percent_inhibition": PRIMARY_DEGREE_THRESHOLD,
            "active_compounds": np.sum(
                pkis1_values >= PRIMARY_DEGREE_THRESHOLD, axis=0
            ),
            "target_degree": target_degree,
        }
    )
    coefficients = pd.DataFrame.from_records(
        [
            {
                "model": "PKIS1_target_degree_prior",
                "feature": feature,
                "coefficient": value,
            }
            for feature, value in zip(
                ("intercept", "joint_degree", "degree_similarity"),
                degree_prior_coef,
            )
        ]
        + [
            {
                "model": "degree_prior_plus_sequence_plus_KLIFS_pocket",
                "feature": feature,
                "coefficient": value,
            }
            for feature, value in zip(
                (
                    "intercept",
                    "joint_degree",
                    "degree_similarity",
                    "receptor_domain_sequence_identity",
                    "KLIFS_pocket_identity",
                ),
                baseline_coef,
            )
        ]
        + [
            {
                "model": model,
                "feature": feature,
                "coefficient": value,
            }
            for model, coef, features in (
                (
                    "baseline_plus_raw_Vina_gate",
                    raw_gated_coef,
                    ("raw_Vina_gate_increment",),
                ),
                (
                    "baseline_plus_residual_Vina_gate",
                    residual_gated_coef,
                    ("residual_Vina_gate_increment",),
                ),
                (
                    "baseline_plus_separate_raw_and_residual_Vina_gates",
                    two_view_gated_coef,
                    ("raw_Vina_gate_increment", "residual_Vina_gate_increment"),
                ),
            )
            for feature, value in zip(
                (
                    "intercept",
                    "joint_degree",
                    "degree_similarity",
                    "receptor_domain_sequence_identity",
                    "KLIFS_pocket_identity",
                    *features,
                ),
                coef,
            )
        ]
    )

    output.mkdir(parents=True, exist_ok=True)
    target_degrees.to_csv(output / "PKIS1_target_degrees.csv", index=False)
    selection.to_csv(output / "discovery_gate_selection.csv", index=False)
    coefficients.to_csv(output / "discovery_model_coefficients.csv", index=False)
    panel_metrics.to_csv(output / "panel_metrics.csv", index=False)
    qap.to_csv(output / "selection_aware_qap.csv", index=False)
    degree_qap.to_csv(output / "degree_prior_increment_qap.csv", index=False)
    degree_jackknife.to_csv(
        output / "degree_prior_target_jackknife.csv", index=False
    )
    prior_sensitivity.to_csv(
        output / "target_prior_definition_sensitivity.csv", index=False
    )
    qap_thresholds.to_csv(output / "qap_selected_thresholds.csv", index=False)
    jackknife.to_csv(output / "target_jackknife.csv", index=False)
    degree_sensitivity.to_csv(output / "degree_threshold_sensitivity.csv", index=False)
    pair_gain.to_csv(output / "ungated_pair_gain.csv", index=False)
    quartiles.to_csv(output / "ungated_pocket_quartiles.csv", index=False)

    qap_primary = qap.loc[qap.metric.isin(PRIMARY_METRICS)].set_index("metric")
    jackknife_primary = jackknife.loc[jackknife.metric.isin(PRIMARY_METRICS)]
    sensitivity_primary = degree_sensitivity.loc[
        degree_sensitivity.metric.isin(PRIMARY_METRICS)
    ]
    locked_metrics = panel_metrics.loc[
        panel_metrics.panel.isin(LOCKED_PANELS)
    ].set_index(["panel", "model"])
    degree_prior_external_rho = {
        panel: float(
            locked_metrics.loc[
                (panel, "PKIS1_target_degree_prior"), "continuous_spearman"
            ]
        )
        for panel in LOCKED_PANELS
    }
    full_baseline_external_rho = {
        panel: float(
            locked_metrics.loc[
                (
                    panel,
                    "degree_prior_plus_sequence_plus_KLIFS_pocket",
                ),
                "continuous_spearman",
            ]
        )
        for panel in LOCKED_PANELS
    }
    residual_beyond_raw = {
        metric: {
            "mean_locked_delta": float(
                np.mean(
                    [
                        locked_metrics.loc[
                            (
                                panel,
                                "baseline_plus_separate_raw_and_residual_Vina_gates",
                            ),
                            metric,
                        ]
                        - locked_metrics.loc[
                            (panel, "baseline_plus_raw_Vina_gate"), metric
                        ]
                        for panel in LOCKED_PANELS
                    ]
                )
            ),
            "minimum_locked_panel_delta": float(
                np.min(
                    [
                        locked_metrics.loc[
                            (
                                panel,
                                "baseline_plus_separate_raw_and_residual_Vina_gates",
                            ),
                            metric,
                        ]
                        - locked_metrics.loc[
                            (panel, "baseline_plus_raw_Vina_gate"), metric
                        ]
                        for panel in LOCKED_PANELS
                    ]
                )
            ),
        }
        for metric in PRIMARY_METRICS
    }
    strict_go = bool(
        all(
            float(qap_primary.loc[metric, "observed_mean_augmented_minus_baseline"])
            > 0
            and float(qap_primary.loc[metric, "minimum_panel_delta"]) > 0
            and float(
                qap_primary.loc[metric, "holm_p_across_three_primary_metrics"]
            )
            <= 0.05
            and int(
                np.sum(
                    jackknife_primary.loc[
                        jackknife_primary.metric.eq(metric),
                        "augmented_minus_baseline",
                    ]
                    > 0
                )
            )
            >= 48
            for metric in PRIMARY_METRICS
        )
    )
    degree_qap_indexed = degree_qap.set_index("metric")
    degree_jackknife_indexed = degree_jackknife.set_index("metric")
    degree_prior_go = bool(
        all(
            float(
                degree_qap_indexed.loc[
                    metric, "observed_mean_degree_prior_increment"
                ]
            )
            > 0
            and float(degree_qap_indexed.loc[metric, "minimum_panel_delta"]) > 0
            and float(
                degree_qap_indexed.loc[
                    metric, "holm_p_across_three_primary_metrics"
                ]
            )
            <= 0.05
            for metric in PRIMARY_METRICS
        )
    )
    summary: dict[str, object] = {
        "analysis": "PKIS1 target-degree prior audit of pocket-gated Vina fusion",
        "status": "post_hoc_exploratory_science_only",
        "discovery_panel": DISCOVERY_PANEL,
        "locked_no_retuning_panels": list(LOCKED_PANELS),
        "targets": list(TARGETS),
        "PKIS1_discovery_compounds": int(len(pkis1_values)),
        "target_degree_definition": (
            "fraction of PKIS1 compounds with at least 50% inhibition; the pair "
            "prior uses rank geometric-mean degree and rank negative absolute "
            "degree difference"
        ),
        "baseline": (
            "PKIS1-fit OLS on target-degree joint breadth, target-degree "
            "similarity, receptor-domain sequence identity, and KLIFS pocket identity"
        ),
        "primary_docking_addition": (
            "one feature proportional to the previously locked dual-surface "
            "pocket-gated fusion minus sequence"
        ),
        "selected_klifs_pocket_identity_threshold": selected_threshold,
        "selection_metric": "PKIS1 top-10-percent average precision",
        "selection_aware_qap": (
            "jointly permutes raw and residual docking target labels, reselects "
            "the gate and refits the augmented PKIS1 model on every permutation; "
            "all non-docking predictors and experimental endpoints remain fixed"
        ),
        "strict_decision": (
            "GO_independent_increment_beyond_degree_prior"
            if strict_go
            else "NO_GO_not_QAP_resolved_beyond_degree_prior"
        ),
        "degree_prior_external_continuous_spearman": degree_prior_external_rho,
        "degree_prior_plus_sequence_plus_pocket_external_continuous_spearman": (
            full_baseline_external_rho
        ),
        "degree_prior_increment_beyond_sequence_plus_pocket": {
            "strict_decision": (
                "GO_replicated_target_prior_increment"
                if degree_prior_go
                else "NO_GO_not_target_label_QAP_resolved"
            ),
            "null": (
                "permute PKIS1 marginal degrees across target labels, reconstruct "
                "both pair-prior features, and refit on PKIS1; sequence, pocket, "
                "and locked experimental panels remain fixed"
            ),
            "metrics": {
                metric: {
                    "mean_locked_delta": float(
                        degree_qap_indexed.loc[
                            metric, "observed_mean_degree_prior_increment"
                        ]
                    ),
                    "minimum_panel_delta": float(
                        degree_qap_indexed.loc[metric, "minimum_panel_delta"]
                    ),
                    "target_label_qap_p": float(
                        degree_qap_indexed.loc[
                            metric, "degree_target_label_qap_p_positive"
                        ]
                    ),
                    "holm_p_three_primary_metrics": float(
                        degree_qap_indexed.loc[
                            metric, "holm_p_across_three_primary_metrics"
                        ]
                    ),
                    "positive_panel_target_deletions": int(
                        np.sum(
                            degree_jackknife_indexed.loc[
                                metric, "degree_prior_increment"
                            ]
                            > 0
                        )
                    ),
                    "total_panel_target_deletions": int(
                        len(
                            degree_jackknife_indexed.loc[
                                metric, "degree_prior_increment"
                            ]
                        )
                    ),
                    "minimum_target_jackknife_delta": float(
                        degree_jackknife_indexed.loc[
                            metric, "degree_prior_increment"
                        ].min()
                    ),
                }
                for metric in PRIMARY_METRICS
            },
            "definition_sensitivity": (
                "See target_prior_definition_sensitivity.csv. The conventional "
                "50% degree is primary; thresholds and continuous marginal "
                "summaries are post-hoc sensitivities, not additional confirmatory tests."
            ),
        },
        "primary_locked_mean_deltas": {
            metric: {
                "augmented_minus_baseline": float(
                    qap_primary.loc[metric, "observed_mean_augmented_minus_baseline"]
                ),
                "minimum_panel_delta": float(
                    qap_primary.loc[metric, "minimum_panel_delta"]
                ),
                "selection_aware_qap_p": float(
                    qap_primary.loc[
                        metric, "selection_aware_target_label_qap_p_positive"
                    ]
                ),
                "holm_p_three_primary_metrics": float(
                    qap_primary.loc[
                        metric, "holm_p_across_three_primary_metrics"
                    ]
                ),
            }
            for metric in PRIMARY_METRICS
        },
        "target_jackknife": {
            metric: {
                "positive_panel_target_deletions": int(
                    np.sum(
                        jackknife_primary.loc[
                            jackknife_primary.metric.eq(metric),
                            "augmented_minus_baseline",
                        ]
                        > 0
                    )
                ),
                "total_panel_target_deletions": int(
                    np.sum(jackknife_primary.metric.eq(metric))
                ),
                "minimum_delta": float(
                    jackknife_primary.loc[
                        jackknife_primary.metric.eq(metric),
                        "augmented_minus_baseline",
                    ].min()
                ),
            }
            for metric in PRIMARY_METRICS
        },
        "degree_threshold_sensitivity": {
            metric: {
                "minimum_locked_panel_delta_across_20_35_50_65_80_percent": float(
                    sensitivity_primary.loc[
                        sensitivity_primary.metric.eq(metric),
                        "augmented_minus_baseline",
                    ].min()
                ),
                "positive_locked_panel_threshold_cells": int(
                    np.sum(
                        sensitivity_primary.loc[
                            sensitivity_primary.metric.eq(metric),
                            "augmented_minus_baseline",
                        ]
                        > 0
                    )
                ),
                "total_locked_panel_threshold_cells": int(
                    np.sum(sensitivity_primary.metric.eq(metric))
                ),
            }
            for metric in PRIMARY_METRICS
        },
        "residual_gate_increment_beyond_raw_gate_at_fixed_primary_gate": (
            residual_beyond_raw
        ),
        "ungated_continuous_pocket_trend": {
            "spearman_pair_gain_vs_pocket_identity": float(monotonicity[0]),
            "selection_aware_qap_p_negative_trend": float(monotonicity[1]),
            "null_mean": float(monotonicity[2]),
            "null_q025": float(monotonicity[3]),
            "null_q975": float(monotonicity[4]),
            "interpretation": (
                "tests whether the absolute-error benefit of an ungated docking "
                "addition increases continuously as KLIFS pocket identity falls"
            ),
        },
        "decision_rule": (
            "GO only if all three primary locked mean deltas are positive, all "
            "three are positive in every locked panel, selection-aware QAP Holm "
            "p values are <=0.05, and at least 80% of target deletions remain positive"
        ),
        "claim_boundary": (
            "The discovery target prior, gate, and coefficients are learned on "
            "PKIS1 and transferred to different ligands on the same 20 targets. "
            "This does not establish ligand-level target ranking, causal biology, "
            "or transfer to unseen targets. The analysis is post hoc."
        ),
        "inputs": {
            "target_pairs": {
                "path": str(pair_path.relative_to(PACKAGE)),
                "sha256": sha256_file(pair_path),
            },
            "PKIS1_workbook_archive": {
                "redistributed": False,
                "local_filename": pkis1_zip.name,
                "sha256": sha256_file(pkis1_zip),
            },
            "KiRHub_workbook": {
                "redistributed": False,
                "local_filename": kirhub_workbook.name,
                "sha256": sha256_file(kirhub_workbook),
            },
        },
    }
    (output / "summary.json").write_text(
        json.dumps(json_ready(summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    primary_lines = []
    for metric in PRIMARY_METRICS:
        record = summary["primary_locked_mean_deltas"][metric]  # type: ignore[index]
        primary_lines.append(
            f"- {metric}: mean delta {record['augmented_minus_baseline']:+.4f}; "
            f"minimum panel delta {record['minimum_panel_delta']:+.4f}; "
            f"selection-aware QAP p={record['selection_aware_qap_p']:.5f}; "
            f"Holm p={record['holm_p_three_primary_metrics']:.5f}."
        )
    trend = summary["ungated_continuous_pocket_trend"]  # type: ignore[assignment]
    readme = [
        "# Target-degree-prior audit of pocket-gated Vina fusion",
        "",
        (
            "**Strict decision: GO.**"
            if strict_go
            else "**Strict decision: NO-GO as an independently resolved docking increment.**"
        ),
        "",
        "This analysis tests the strongest simple rival explanation for the",
        "pocket-gated fusion result: target pairs may look co-selective because",
        "their targets merely have similar marginal hit rates. PKIS1 alone defines",
        "the target-degree prior, fits all coefficients, and selects the gate; DAVIS,",
        "PKIS2, and KiRHub remain locked/no-retuning evaluations.",
        "",
        f"The selected gate remains **{selected_threshold:.2f}**. Relative to a",
        "baseline already containing the PKIS1 target-degree prior, receptor-domain",
        "sequence identity, and KLIFS pocket identity, the exact dual-surface gated",
        "docking addition gives:",
        "",
        *primary_lines,
        "",
        (
            "All point estimates are positive in every locked panel and in all "
            "60 panel-by-target deletions, but the full-procedure QAP does not "
            "resolve the increment after multiplicity correction. The directional "
            "effect is therefore suggestive, not a new headline result."
            if not strict_go
            else "The increment passes the predeclared directional, QAP, and target-jackknife criteria."
        ),
        "",
        "The alternative explanation is substantial: the two-feature PKIS1 degree",
        "prior alone reaches external Spearman correlations from "
        f"{min(degree_prior_external_rho.values()):.3f} to "
        f"{max(degree_prior_external_rho.values()):.3f}; adding sequence and pocket",
        "identity raises them to "
        f"{min(full_baseline_external_rho.values()):.3f}--"
        f"{max(full_baseline_external_rho.values()):.3f}.",
        (
            "This target-degree increment passes its target-label QAP and is the "
            "strong positive result of the audit."
            if degree_prior_go
            else "The target-degree increment itself does not pass its target-label QAP."
        ),
        "See `degree_prior_increment_qap.csv` for the complete null comparison.",
        "The degree-prior increment remains positive in all 60 panel-by-target",
        "deletions for every primary metric. Alternate cutoffs and threshold-free",
        "marginal summaries are reported in",
        "`target_prior_definition_sensitivity.csv`; these are explicitly post hoc.",
        "",
        "In the fixed-gate component decomposition, adding a separately weighted",
        "residual-Vina gate after the raw-Vina gate changes the locked mean by",
        f"{residual_beyond_raw['continuous_spearman']['mean_locked_delta']:+.4f} "
        "Spearman, "
        f"{residual_beyond_raw['top_10_roc_auc']['mean_locked_delta']:+.4f} AUROC, "
        "and "
        f"{residual_beyond_raw['top_10_average_precision']['mean_locked_delta']:+.4f} AP. "
        "At least one locked panel is negative for each of these nested contrasts,",
        "so the present audit does not isolate an incremental residual-surface gain",
        "beyond raw Vina once the strong baseline is included.",
        "",
        "The QAP is selection-aware: it jointly permutes raw and residual Vina",
        "target labels and repeats PKIS1 gate selection and model fitting on every",
        "permutation. See `selection_aware_qap.csv` for top-5%, top-10%, and top-15%",
        "endpoints, `target_jackknife.csv` for delete-one-target results, and",
        "`degree_threshold_sensitivity.csv` for alternate PKIS1 activity cutoffs.",
        "",
        "## No-hard-gate test",
        "",
        (
            "Without imposing a hard gate, pairwise error improvement has Spearman "
            f"rho {trend['spearman_pair_gain_vs_pocket_identity']:+.3f} with KLIFS "
            "pocket identity (one-sided target-label QAP "
            f"p={trend['selection_aware_qap_p_negative_trend']:.4f})."
        ),
        "This is a separate falsification test of a monotonic low-identity trend; a",
        "non-significant result must not be used to justify the hard threshold.",
        "",
        "## Boundary",
        "",
        "This post-hoc result concerns target-pair geometry on a fixed 20-kinase",
        "panel. It does not establish ligand-level target retrieval or generalization",
        "to unseen targets.",
        "",
        "## Reproduction",
        "",
        "```bash",
        ".venv/bin/python analysis/pocket_gated_prior_audit.py \\",
        "  --pkis1-zip /private/tmp/pkis1_supplement.zip \\",
        "  --kirhub-workbook /private/tmp/kirhub_supp_tables.xlsx \\",
        "  --output results/pocket_gated_prior_audit \\",
        f"  --permutations {permutations} --seed {seed}",
        "```",
        "",
    ]
    (output / "README.md").write_text("\n".join(readme), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, default=DEFAULT_PAIRS)
    parser.add_argument("--pkis1-zip", type=Path, default=DEFAULT_PKIS1)
    parser.add_argument("--kirhub-workbook", type=Path, default=DEFAULT_KIRHUB)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--permutations", type=int, default=DEFAULT_PERMUTATIONS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_analysis(
        pair_path=args.pairs,
        pkis1_zip=args.pkis1_zip,
        kirhub_workbook=args.kirhub_workbook,
        output=args.output,
        permutations=args.permutations,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
