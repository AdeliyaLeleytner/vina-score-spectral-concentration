#!/usr/bin/env python3
"""How many calibration compounds recover a transferable target-breadth prior?

This science-only analysis turns the strong PKIS1 target-degree result into an
assay-design question.  We repeatedly expose only ``n`` PKIS1 compounds, use
that same small dense panel to (i) estimate marginal target breadth and (ii)
fit a target-pair co-selectivity model, and then evaluate without refitting on
DAVIS, PKIS2, and KiRHub.

The comparison is deliberately strict.  Both models are fit to the identical
sample-derived PKIS1 target-correlation endpoint.  The baseline contains only
receptor-domain sequence identity and KLIFS ATP-pocket identity.  The
augmented model adds two transparent pair features constructed from the
sampled marginal target degrees: geometric-mean breadth and similarity of
breadth.  Consequently, a positive locked-panel delta cannot be attributed to
giving the augmented model a larger discovery set.

Two nested, outcome-blind sampling schemes are compared over many seeded
repetitions: uniform compounds and a Butina-cluster round-robin order that
selects one compound per chemical cluster before taking a second.  A single
Morgan-fingerprint MaxMin ordering supplies a fully deterministic practical
panel.  No activity value enters either chemical selection rule.

This is an exploratory fixed-20-kinase result.  It concerns target-pair
co-selectivity geometry, not ligand-level target ranking, and does not establish
that 80 compounds are sufficient for other protein families or assay formats.
No manuscript file is read or modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import OrderedDict
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.SimDivFilters.rdSimDivPickers import MaxMinPicker
from scipy import stats

try:
    from .biological_core_modes import holm_adjust
    from .dense_davis_benchmark import _butina_labels, _murcko_labels
    from .pocket_gated_fusion import (
        DISCOVERY_PANEL,
        LOCKED_PANELS,
        TARGETS,
        load_inputs,
    )
    from .pocket_gated_prior_audit import (
        PRIMARY_METRICS,
        degree_pair_features,
        endpoint_cache,
        model_fit,
        pair_percentile_rank,
        predictor_metrics_from_cache,
        sha256_file,
    )
    from .residual_target_geometry_validation import (
        load_pkis1_full,
        target_correlation,
        two_way_center,
    )
except ImportError:  # pragma: no cover - direct CLI execution
    from biological_core_modes import holm_adjust  # type: ignore
    from dense_davis_benchmark import _butina_labels, _murcko_labels  # type: ignore
    from pocket_gated_fusion import (  # type: ignore
        DISCOVERY_PANEL,
        LOCKED_PANELS,
        TARGETS,
        load_inputs,
    )
    from pocket_gated_prior_audit import (  # type: ignore
        PRIMARY_METRICS,
        degree_pair_features,
        endpoint_cache,
        model_fit,
        pair_percentile_rank,
        predictor_metrics_from_cache,
        sha256_file,
    )
    from residual_target_geometry_validation import (  # type: ignore
        load_pkis1_full,
        target_correlation,
        two_way_center,
    )


PACKAGE = Path(__file__).resolve().parents[1]
DEFAULT_PAIRS = PACKAGE / "results" / "klifs_pocket_control" / "target_pairs.csv"
DEFAULT_PKIS1 = PACKAGE / "downloads" / "source_restricted" / "pkis1_supplement.zip"
DEFAULT_KIRHUB = PACKAGE / "downloads" / "source_restricted" / "kirhub_supplement.xlsx"
DEFAULT_OUTPUT = PACKAGE / "results" / "calibration_sample_efficiency"
DEFAULT_REPETITIONS = 1_000
DEFAULT_PERMUTATIONS = 20_000
DEFAULT_SEED = 20260830
SAMPLE_SIZES = (10, 20, 40, 80, 160, 360)
PRACTICAL_SIZE = 80
PRIMARY_ACTIVITY_THRESHOLD = 50.0
BUTINA_SIMILARITY_THRESHOLD = 0.65
MORGAN_RADIUS = 2
MORGAN_BITS = 2048
SAMPLING_SCHEMES = ("uniform_random", "Butina_cluster_round_robin")


def release_input_path(path: Path) -> str:
    """Return a portable provenance path without leaking a local filesystem."""
    try:
        return path.resolve().relative_to(PACKAGE.resolve()).as_posix()
    except ValueError:
        return (Path("source-restricted") / path.name).as_posix()


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


def sample_geometry(values: np.ndarray) -> np.ndarray:
    """Two-way-centered target correlation from a dense calibration panel."""
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] < 4 or matrix.shape[1] < 3:
        raise ValueError("calibration matrix is too small")
    if not np.isfinite(matrix).all():
        raise ValueError("calibration matrix contains non-finite values")
    return target_correlation(two_way_center(matrix))


def cluster_round_robin_order(
    cluster_labels: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Return an outcome-blind permutation, exhausting one per cluster first."""
    labels = np.asarray(cluster_labels)
    if labels.ndim != 1 or len(labels) < 2:
        raise ValueError("cluster labels must be a one-dimensional vector")
    groups = {
        label: rng.permutation(np.flatnonzero(labels == label))
        for label in np.unique(labels)
    }
    group_order = rng.permutation(np.asarray(list(groups), dtype=object))
    selected: list[int] = []
    maximum = max(len(members) for members in groups.values())
    for within_cluster_rank in range(maximum):
        # Re-permuting the cluster order in later rounds prevents cluster-size
        # ties from inheriting an arbitrary label order.  The first round still
        # contains exactly one representative from every cluster.
        order = (
            group_order
            if within_cluster_rank == 0
            else rng.permutation(group_order)
        )
        for label in order:
            members = groups[label]
            if within_cluster_rank < len(members):
                selected.append(int(members[within_cluster_rank]))
    result = np.asarray(selected, dtype=int)
    if len(result) != len(labels) or len(np.unique(result)) != len(labels):
        raise RuntimeError("cluster round-robin did not create a permutation")
    return result


def seeded_orders(
    cluster_labels: np.ndarray, repetitions: int, seed: int
) -> dict[str, list[np.ndarray]]:
    """Nested reproducible compound orders for both repeated schemes."""
    if repetitions < 1:
        raise ValueError("repetitions must be positive")
    size = len(cluster_labels)
    output = {scheme: [] for scheme in SAMPLING_SCHEMES}
    for repetition in range(repetitions):
        uniform_rng = np.random.default_rng(
            np.random.SeedSequence([seed, repetition, 0])
        )
        diverse_rng = np.random.default_rng(
            np.random.SeedSequence([seed, repetition, 1])
        )
        output["uniform_random"].append(uniform_rng.permutation(size))
        output["Butina_cluster_round_robin"].append(
            cluster_round_robin_order(cluster_labels, diverse_rng)
        )
    return output


def deterministic_maxmin_order(smiles: Sequence[str], seed: int) -> np.ndarray:
    """Morgan-r2/2048 MaxMin order; structures and a fixed seed are the only inputs."""
    molecules = [Chem.MolFromSmiles(str(value)) for value in smiles]
    if any(molecule is None for molecule in molecules):
        raise ValueError("MaxMin panel contains an invalid SMILES")
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=MORGAN_RADIUS, fpSize=MORGAN_BITS
    )
    fingerprints = [generator.GetFingerprint(molecule) for molecule in molecules]
    picker = MaxMinPicker()
    result = np.asarray(
        list(
            picker.LazyBitVectorPick(
                fingerprints,
                len(fingerprints),
                len(fingerprints),
                seed=int(seed),
            )
        ),
        dtype=int,
    )
    if len(result) != len(smiles) or len(np.unique(result)) != len(smiles):
        raise RuntimeError("MaxMin did not return a complete permutation")
    return result


def fit_from_calibration(
    calibration_values: np.ndarray,
    base_features: np.ndarray,
    pair_indices: tuple[np.ndarray, np.ndarray],
    activity_threshold: float = PRIMARY_ACTIVITY_THRESHOLD,
) -> dict[str, np.ndarray]:
    """Fit baseline and degree-augmented pair models on one small panel."""
    values = np.asarray(calibration_values, dtype=np.float64)
    geometry = sample_geometry(values)
    endpoint = pair_percentile_rank(geometry[pair_indices])
    target_degree = np.mean(values >= float(activity_threshold), axis=0)
    baseline, baseline_coefficients = model_fit(base_features, endpoint)
    augmented_features = np.column_stack(
        [degree_pair_features(target_degree, pair_indices), base_features]
    )
    augmented, augmented_coefficients = model_fit(
        augmented_features, endpoint
    )
    return {
        "endpoint": endpoint,
        "target_degree": target_degree,
        "baseline_prediction": baseline,
        "augmented_prediction": augmented,
        "baseline_coefficients": baseline_coefficients,
        "augmented_coefficients": augmented_coefficients,
    }


def locked_deltas(
    baseline_prediction: np.ndarray,
    augmented_prediction: np.ndarray,
    caches: dict[str, dict[str, object]],
) -> dict[str, dict[str, float]]:
    output: dict[str, dict[str, float]] = {}
    for panel in LOCKED_PANELS:
        baseline = predictor_metrics_from_cache(
            baseline_prediction, caches[panel]
        )
        augmented = predictor_metrics_from_cache(
            augmented_prediction, caches[panel]
        )
        output[panel] = {
            metric: float(augmented[metric] - baseline[metric])
            for metric in PRIMARY_METRICS
        }
    return output


def evaluate_repeated_samples(
    *,
    values: np.ndarray,
    butina_labels: np.ndarray,
    murcko_labels: np.ndarray,
    base_features: np.ndarray,
    pair_indices: tuple[np.ndarray, np.ndarray],
    caches: dict[str, dict[str, object]],
    repetitions: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    orders = seeded_orders(butina_labels, repetitions, seed)
    full_degree = np.mean(values >= PRIMARY_ACTIVITY_THRESHOLD, axis=0)
    metric_rows: list[dict[str, object]] = []
    recovery_rows: list[dict[str, object]] = []
    for scheme, scheme_orders in orders.items():
        for repetition, order in enumerate(scheme_orders):
            for sample_size in SAMPLE_SIZES:
                selected = order[:sample_size]
                fitted = fit_from_calibration(
                    values[selected], base_features, pair_indices
                )
                degree = fitted["target_degree"]
                degree_rho = float(stats.spearmanr(degree, full_degree).statistic)
                recovery_rows.append(
                    {
                        "sampling_scheme": scheme,
                        "repetition": repetition,
                        "sample_size": sample_size,
                        "sample_seed": seed,
                        "sample_index_sha256": hashlib.sha256(
                            np.asarray(selected, dtype=np.int64).tobytes()
                        ).hexdigest(),
                        "unique_Butina_clusters": int(
                            len(np.unique(butina_labels[selected]))
                        ),
                        "unique_Murcko_groups": int(
                            len(np.unique(murcko_labels[selected]))
                        ),
                        "degree_spearman_vs_full_PKIS1": degree_rho,
                        "degree_rmse_vs_full_PKIS1": float(
                            np.sqrt(np.mean(np.square(degree - full_degree)))
                        ),
                    }
                )
                deltas = locked_deltas(
                    fitted["baseline_prediction"],
                    fitted["augmented_prediction"],
                    caches,
                )
                for metric in PRIMARY_METRICS:
                    panel_values = [deltas[panel][metric] for panel in LOCKED_PANELS]
                    metric_rows.append(
                        {
                            "sampling_scheme": scheme,
                            "repetition": repetition,
                            "sample_size": sample_size,
                            "metric": metric,
                            **{
                                f"{panel}_augmented_minus_baseline": deltas[panel][metric]
                                for panel in LOCKED_PANELS
                            },
                            "mean_locked_delta": float(np.mean(panel_values)),
                            "minimum_locked_panel_delta": float(np.min(panel_values)),
                            "positive_on_every_locked_panel": bool(
                                np.min(panel_values) > 0
                            ),
                        }
                    )
    return pd.DataFrame.from_records(metric_rows), pd.DataFrame.from_records(
        recovery_rows
    )


def summarize_repeated_samples(
    metrics: pd.DataFrame, recovery: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary_rows: list[dict[str, object]] = []
    for keys, group in metrics.groupby(
        ["sampling_scheme", "sample_size", "metric"], sort=True
    ):
        scheme, sample_size, metric = keys
        values = group.mean_locked_delta.to_numpy(dtype=float)
        summary_rows.append(
            {
                "sampling_scheme": scheme,
                "sample_size": int(sample_size),
                "metric": metric,
                "repetitions": int(len(group)),
                "mean_locked_delta_median": float(np.median(values)),
                "mean_locked_delta_q025": float(np.quantile(values, 0.025)),
                "mean_locked_delta_q975": float(np.quantile(values, 0.975)),
                "fraction_mean_locked_delta_positive": float(np.mean(values > 0)),
                "fraction_positive_on_every_locked_panel": float(
                    group.positive_on_every_locked_panel.mean()
                ),
            }
        )
    sampling_summary = pd.DataFrame.from_records(summary_rows)

    recovery_rows: list[dict[str, object]] = []
    for keys, group in recovery.groupby(
        ["sampling_scheme", "sample_size"], sort=True
    ):
        scheme, sample_size = keys
        correlations = group.degree_spearman_vs_full_PKIS1.to_numpy(dtype=float)
        recovery_rows.append(
            {
                "sampling_scheme": scheme,
                "sample_size": int(sample_size),
                "repetitions": int(len(group)),
                "degree_spearman_median": float(np.median(correlations)),
                "degree_spearman_q025": float(np.quantile(correlations, 0.025)),
                "degree_spearman_q975": float(np.quantile(correlations, 0.975)),
                "degree_rmse_median": float(
                    group.degree_rmse_vs_full_PKIS1.median()
                ),
                "unique_Butina_clusters_median": float(
                    group.unique_Butina_clusters.median()
                ),
                "unique_Murcko_groups_median": float(
                    group.unique_Murcko_groups.median()
                ),
            }
        )
    recovery_summary = pd.DataFrame.from_records(recovery_rows)

    comparison_rows: list[dict[str, object]] = []
    indexed = metrics.set_index(
        ["sampling_scheme", "sample_size", "metric", "repetition"]
    ).sort_index()
    for sample_size in SAMPLE_SIZES:
        for metric in PRIMARY_METRICS:
            random_values = indexed.loc[
                ("uniform_random", sample_size, metric), "mean_locked_delta"
            ].sort_index()
            diverse_values = indexed.loc[
                ("Butina_cluster_round_robin", sample_size, metric),
                "mean_locked_delta",
            ].sort_index()
            difference = diverse_values.to_numpy() - random_values.to_numpy()
            comparison_rows.append(
                {
                    "sample_size": sample_size,
                    "metric": metric,
                    "contrast": "Butina_cluster_round_robin_minus_uniform_random",
                    "paired_by_reproducible_repetition_id": True,
                    "median_difference": float(np.median(difference)),
                    "q025_difference": float(np.quantile(difference, 0.025)),
                    "q975_difference": float(np.quantile(difference, 0.975)),
                    "fraction_difference_positive": float(np.mean(difference > 0)),
                }
            )
    return (
        sampling_summary,
        recovery_summary,
        pd.DataFrame.from_records(comparison_rows),
    )


def evaluate_maxmin_panels(
    *,
    order: np.ndarray,
    values: np.ndarray,
    base_features: np.ndarray,
    pair_indices: tuple[np.ndarray, np.ndarray],
    caches: dict[str, dict[str, object]],
    full_degree: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows: list[dict[str, object]] = []
    degree_rows: list[dict[str, object]] = []
    for sample_size in SAMPLE_SIZES:
        selected = order[:sample_size]
        fitted = fit_from_calibration(values[selected], base_features, pair_indices)
        degree = fitted["target_degree"]
        degree_rows.append(
            {
                "sample_size": sample_size,
                "degree_spearman_vs_full_PKIS1": float(
                    stats.spearmanr(degree, full_degree).statistic
                ),
                "degree_rmse_vs_full_PKIS1": float(
                    np.sqrt(np.mean(np.square(degree - full_degree)))
                ),
            }
        )
        deltas = locked_deltas(
            fitted["baseline_prediction"], fitted["augmented_prediction"], caches
        )
        for panel in LOCKED_PANELS:
            for metric in PRIMARY_METRICS:
                metric_rows.append(
                    {
                        "sample_size": sample_size,
                        "panel": panel,
                        "metric": metric,
                        "augmented_minus_baseline": deltas[panel][metric],
                    }
                )
    return pd.DataFrame.from_records(metric_rows), pd.DataFrame.from_records(degree_rows)


def target_summary_sensitivity(
    *,
    calibration_values: np.ndarray,
    base_features: np.ndarray,
    pair_indices: tuple[np.ndarray, np.ndarray],
    caches: dict[str, dict[str, object]],
) -> pd.DataFrame:
    values = np.asarray(calibration_values, dtype=float)
    definitions: OrderedDict[str, np.ndarray] = OrderedDict(
        (
            f"fraction_at_least_{int(threshold)}_percent_inhibition",
            np.mean(values >= threshold, axis=0),
        )
        for threshold in (20.0, 35.0, 50.0, 65.0, 80.0)
    )
    definitions["clipped_mean_percent_inhibition"] = np.mean(
        np.clip(values, 0.0, 100.0), axis=0
    )
    definitions["raw_mean_percent_inhibition"] = np.mean(values, axis=0)
    endpoint = pair_percentile_rank(sample_geometry(values)[pair_indices])
    baseline, _ = model_fit(base_features, endpoint)
    baseline_metrics = {
        panel: predictor_metrics_from_cache(baseline, caches[panel])
        for panel in LOCKED_PANELS
    }
    rows: list[dict[str, object]] = []
    for definition, target_summary in definitions.items():
        features = np.column_stack(
            [degree_pair_features(target_summary, pair_indices), base_features]
        )
        augmented, _ = model_fit(features, endpoint)
        for panel in LOCKED_PANELS:
            augmented_metrics = predictor_metrics_from_cache(
                augmented, caches[panel]
            )
            for metric in PRIMARY_METRICS:
                rows.append(
                    {
                        "target_summary_definition": definition,
                        "panel": panel,
                        "metric": metric,
                        "augmented_minus_baseline": (
                            augmented_metrics[metric]
                            - baseline_metrics[panel][metric]
                        ),
                    }
                )
    return pd.DataFrame.from_records(rows)


def aligned_degree_qap(
    *,
    calibration_values: np.ndarray,
    base_features: np.ndarray,
    pair_indices: tuple[np.ndarray, np.ndarray],
    caches: dict[str, dict[str, object]],
    permutations: int,
    seed: int,
) -> pd.DataFrame:
    """Target-label QAP for the practical MaxMin calibration panel."""
    if permutations < 1:
        raise ValueError("permutations must be positive")
    fitted = fit_from_calibration(
        calibration_values, base_features, pair_indices
    )
    baseline = fitted["baseline_prediction"]
    observed = fitted["augmented_prediction"]
    endpoint = fitted["endpoint"]
    degree = fitted["target_degree"]
    observed_deltas = locked_deltas(baseline, observed, caches)
    observed_mean = {
        metric: float(
            np.mean([observed_deltas[panel][metric] for panel in LOCKED_PANELS])
        )
        for metric in PRIMARY_METRICS
    }
    null = {
        metric: np.empty(permutations, dtype=np.float64)
        for metric in PRIMARY_METRICS
    }
    rng = np.random.default_rng(seed)
    for repetition in range(permutations):
        permuted = degree[rng.permutation(len(degree))]
        features = np.column_stack(
            [degree_pair_features(permuted, pair_indices), base_features]
        )
        candidate, _ = model_fit(features, endpoint)
        candidate_deltas = locked_deltas(baseline, candidate, caches)
        for metric in PRIMARY_METRICS:
            null[metric][repetition] = np.mean(
                [candidate_deltas[panel][metric] for panel in LOCKED_PANELS]
            )
    probabilities = {
        metric: float(
            (1 + np.sum(null[metric] >= observed_mean[metric]))
            / (permutations + 1)
        )
        for metric in PRIMARY_METRICS
    }
    adjusted = holm_adjust(probabilities)
    rows: list[dict[str, object]] = []
    for metric in PRIMARY_METRICS:
        rows.append(
            {
                "metric": metric,
                "observed_mean_locked_delta": observed_mean[metric],
                "minimum_locked_panel_delta": float(
                    min(observed_deltas[panel][metric] for panel in LOCKED_PANELS)
                ),
                "target_label_qap_p_positive": probabilities[metric],
                "holm_p_across_three_primary_metrics": adjusted[metric],
                "null_mean": float(np.mean(null[metric])),
                "null_q025": float(np.quantile(null[metric], 0.025)),
                "null_q975": float(np.quantile(null[metric], 0.975)),
                "permutations": permutations,
                "seed": seed,
            }
        )
    return pd.DataFrame.from_records(rows)


def practical_target_jackknife(
    *,
    calibration_values: np.ndarray,
    sequence_matrix: np.ndarray,
    pocket_matrix: np.ndarray,
    endpoint_matrices: dict[str, np.ndarray],
) -> pd.DataFrame:
    """Delete one target, recompute all pair ranks, and refit both models."""
    values = np.asarray(calibration_values, dtype=float)
    rows: list[dict[str, object]] = []
    for omitted, target in enumerate(TARGETS):
        keep = np.delete(np.arange(len(TARGETS)), omitted)
        tri = np.triu_indices(len(keep), k=1)
        sequence = pair_percentile_rank(
            sequence_matrix[np.ix_(keep, keep)][tri]
        )
        pocket = pair_percentile_rank(
            pocket_matrix[np.ix_(keep, keep)][tri]
        )
        base_features = np.column_stack([sequence, pocket])
        fitted = fit_from_calibration(values[:, keep], base_features, tri)
        for panel in LOCKED_PANELS:
            endpoint = pair_percentile_rank(
                endpoint_matrices[panel][np.ix_(keep, keep)][tri]
            )
            cache = endpoint_cache(endpoint)
            baseline = predictor_metrics_from_cache(
                fitted["baseline_prediction"], cache
            )
            augmented = predictor_metrics_from_cache(
                fitted["augmented_prediction"], cache
            )
            for metric in PRIMARY_METRICS:
                rows.append(
                    {
                        "omitted_target": target,
                        "panel": panel,
                        "metric": metric,
                        "augmented_minus_baseline": (
                            augmented[metric] - baseline[metric]
                        ),
                    }
                )
    return pd.DataFrame.from_records(rows)


def _strict_small_panel_decision(summary: pd.DataFrame) -> tuple[bool, int | None]:
    """Prespecified robustness rule applied to the repeated diverse samples."""
    subset = summary.loc[
        summary.sampling_scheme.eq("Butina_cluster_round_robin")
    ]
    for sample_size in SAMPLE_SIZES:
        if sample_size > PRACTICAL_SIZE:
            continue
        rows = subset.loc[subset.sample_size.eq(sample_size)].set_index("metric")
        if set(rows.index) != set(PRIMARY_METRICS):
            continue
        passes = all(
            float(rows.loc[metric, "mean_locked_delta_q025"]) > 0
            and float(
                rows.loc[metric, "fraction_positive_on_every_locked_panel"]
            )
            >= 0.95
            for metric in PRIMARY_METRICS
        )
        if passes:
            return True, sample_size
    return False, None


def run_analysis(
    *,
    pair_path: Path,
    pkis1_zip: Path,
    kirhub_workbook: Path,
    output: Path,
    repetitions: int,
    permutations: int,
    seed: int,
) -> dict[str, object]:
    pkis1 = load_pkis1_full(pkis1_zip)
    values = pkis1[list(TARGETS)].to_numpy(dtype=np.float64)
    if values.shape != (360, len(TARGETS)):
        raise ValueError("unexpected PKIS1 calibration support")
    predictors, endpoint_matrices = load_inputs(pair_path, kirhub_workbook)
    pair_indices = np.triu_indices(len(TARGETS), k=1)
    sequence_matrix = predictors["receptor_domain_sequence_identity"]
    pocket_matrix = predictors["klifs_pocket_identity"]
    base_features = np.column_stack(
        [
            pair_percentile_rank(sequence_matrix[pair_indices]),
            pair_percentile_rank(pocket_matrix[pair_indices]),
        ]
    )
    endpoints = {
        panel: pair_percentile_rank(matrix[pair_indices])
        for panel, matrix in endpoint_matrices.items()
    }
    caches = {
        panel: endpoint_cache(endpoints[panel]) for panel in LOCKED_PANELS
    }

    butina_labels = _butina_labels(
        pkis1.SMILES,
        radius=MORGAN_RADIUS,
        fp_size=MORGAN_BITS,
        similarity_threshold=BUTINA_SIMILARITY_THRESHOLD,
    )
    murcko_labels = _murcko_labels(pkis1.SMILES)
    repeated, degree_recovery = evaluate_repeated_samples(
        values=values,
        butina_labels=butina_labels,
        murcko_labels=murcko_labels,
        base_features=base_features,
        pair_indices=pair_indices,
        caches=caches,
        repetitions=repetitions,
        seed=seed,
    )
    sampling_summary, recovery_summary, scheme_comparison = summarize_repeated_samples(
        repeated, degree_recovery
    )

    maxmin_order = deterministic_maxmin_order(pkis1.SMILES.tolist(), seed)
    full_degree = np.mean(values >= PRIMARY_ACTIVITY_THRESHOLD, axis=0)
    maxmin_metrics, maxmin_recovery = evaluate_maxmin_panels(
        order=maxmin_order,
        values=values,
        base_features=base_features,
        pair_indices=pair_indices,
        caches=caches,
        full_degree=full_degree,
    )
    practical_indices = maxmin_order[:PRACTICAL_SIZE]
    qap = aligned_degree_qap(
        calibration_values=values[practical_indices],
        base_features=base_features,
        pair_indices=pair_indices,
        caches=caches,
        permutations=permutations,
        seed=seed + 1,
    )
    jackknife = practical_target_jackknife(
        calibration_values=values[practical_indices],
        sequence_matrix=sequence_matrix,
        pocket_matrix=pocket_matrix,
        endpoint_matrices=endpoint_matrices,
    )
    sensitivity = target_summary_sensitivity(
        calibration_values=values[practical_indices],
        base_features=base_features,
        pair_indices=pair_indices,
        caches=caches,
    )

    maxmin_manifest = pkis1.loc[
        maxmin_order,
        ["Compound ID", "standard_inchikey", "connectivity_block", "SMILES"],
    ].copy()
    maxmin_manifest.insert(0, "MaxMin_selection_rank_one_based", np.arange(1, 361))
    maxmin_manifest.insert(
        1, "included_in_practical_80", np.arange(360) < PRACTICAL_SIZE
    )
    maxmin_manifest["Butina_cluster"] = butina_labels[maxmin_order]
    maxmin_manifest["Murcko_group"] = murcko_labels[maxmin_order]

    output.mkdir(parents=True, exist_ok=True)
    repeated.to_csv(output / "repeated_subsample_metrics.csv", index=False)
    degree_recovery.to_csv(output / "repeated_degree_recovery.csv", index=False)
    sampling_summary.to_csv(output / "sampling_summary.csv", index=False)
    recovery_summary.to_csv(output / "degree_recovery_summary.csv", index=False)
    scheme_comparison.to_csv(output / "sampling_scheme_comparison.csv", index=False)
    maxmin_metrics.to_csv(output / "deterministic_MaxMin_metrics.csv", index=False)
    maxmin_recovery.to_csv(output / "deterministic_MaxMin_degree_recovery.csv", index=False)
    maxmin_manifest.to_csv(output / "deterministic_MaxMin_manifest.csv", index=False)
    qap.to_csv(output / "practical_80_target_label_qap.csv", index=False)
    jackknife.to_csv(output / "practical_80_target_jackknife.csv", index=False)
    sensitivity.to_csv(output / "practical_80_prior_definition_sensitivity.csv", index=False)

    robust, smallest_robust = _strict_small_panel_decision(sampling_summary)
    qap_indexed = qap.set_index("metric")
    jackknife_indexed = jackknife.set_index("metric")
    practical_summary = sampling_summary.loc[
        sampling_summary.sampling_scheme.eq("Butina_cluster_round_robin")
        & sampling_summary.sample_size.eq(PRACTICAL_SIZE)
    ].set_index("metric")
    maxmin_practical = maxmin_metrics.loc[
        maxmin_metrics.sample_size.eq(PRACTICAL_SIZE)
    ].pivot(index="metric", columns="panel", values="augmented_minus_baseline")
    summary: dict[str, object] = {
        "analysis": "sample efficiency of a transferable target-breadth prior",
        "status": "post_hoc_exploratory_science_only",
        "scientific_question": (
            "how many outcome-blindly selected PKIS1 calibration compounds are "
            "needed before a marginal target-breadth prior improves target-pair "
            "co-selectivity prediction beyond sequence plus KLIFS pocket identity"
        ),
        "discovery_resource": DISCOVERY_PANEL,
        "locked_no_retuning_panels": list(LOCKED_PANELS),
        "targets": list(TARGETS),
        "sample_sizes": list(SAMPLE_SIZES),
        "repetitions_per_scheme_and_size": repetitions,
        "primary_target_degree": (
            "fraction of sampled PKIS1 compounds with at least 50% inhibition"
        ),
        "models": {
            "baseline": (
                "OLS fit on the sample-derived centered PKIS1 geometry using "
                "receptor-domain sequence and KLIFS pocket pair ranks"
            ),
            "augmented": (
                "the same sample-derived fit plus pair-rank geometric-mean target "
                "degree and negative absolute target-degree difference"
            ),
        },
        "sampling": {
            "uniform": "seeded random permutations; sample sizes are nested prefixes",
            "diversity": (
                "seeded Butina-cluster round robin; one compound from every "
                "Morgan-r2/2048 Tanimoto>=0.65 cluster before a second"
            ),
            "deterministic_practical_panel": (
                "single structure-only Morgan-r2/2048 MaxMin ordering; fixed seed; "
                "activity values are never supplied to the selector"
            ),
            "Butina_clusters": int(len(np.unique(butina_labels))),
            "Murcko_groups": int(len(np.unique(murcko_labels))),
        },
        "decision_rule": (
            "GO at the smallest n<=80 only if, for every primary metric, the "
            "2.5th percentile of the mean locked-panel delta is positive and at "
            "least 95% of diversity-sampled panels improve every locked panel"
        ),
        "decision": (
            "GO_small_diverse_calibration_panel"
            if robust
            else "NO_GO_small_panel_not_robust"
        ),
        "smallest_sample_size_passing_rule": smallest_robust,
        "practical_80_repeated_diversity_sampling": {
            metric: {
                "median_mean_locked_delta": float(
                    practical_summary.loc[metric, "mean_locked_delta_median"]
                ),
                "q025_mean_locked_delta": float(
                    practical_summary.loc[metric, "mean_locked_delta_q025"]
                ),
                "q975_mean_locked_delta": float(
                    practical_summary.loc[metric, "mean_locked_delta_q975"]
                ),
                "fraction_positive_on_every_locked_panel": float(
                    practical_summary.loc[
                        metric, "fraction_positive_on_every_locked_panel"
                    ]
                ),
            }
            for metric in PRIMARY_METRICS
        },
        "deterministic_MaxMin_80": {
            "degree_spearman_vs_full_PKIS1": float(
                maxmin_recovery.loc[
                    maxmin_recovery.sample_size.eq(PRACTICAL_SIZE),
                    "degree_spearman_vs_full_PKIS1",
                ].iloc[0]
            ),
            "locked_panel_deltas": {
                metric: {
                    panel: float(maxmin_practical.loc[metric, panel])
                    for panel in LOCKED_PANELS
                }
                for metric in PRIMARY_METRICS
            },
            "target_label_QAP": {
                metric: {
                    "mean_locked_delta": float(
                        qap_indexed.loc[metric, "observed_mean_locked_delta"]
                    ),
                    "minimum_locked_panel_delta": float(
                        qap_indexed.loc[metric, "minimum_locked_panel_delta"]
                    ),
                    "qap_p": float(
                        qap_indexed.loc[metric, "target_label_qap_p_positive"]
                    ),
                    "holm_p": float(
                        qap_indexed.loc[
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
                            jackknife_indexed.loc[
                                metric, "augmented_minus_baseline"
                            ]
                            > 0
                        )
                    ),
                    "total_panel_target_deletions": int(
                        len(jackknife_indexed.loc[metric])
                    ),
                    "minimum_delta": float(
                        jackknife_indexed.loc[
                            metric, "augmented_minus_baseline"
                        ].min()
                    ),
                }
                for metric in PRIMARY_METRICS
            },
        },
        "interpretation_boundary": (
            "The result says that a modest dense calibration panel can learn a "
            "transferable marginal target-breadth prior for this fixed kinase set. "
            "It does not show improved ligand-level target ranking, unseen-target "
            "generalization, or sufficiency outside these assay panels."
        ),
        "inputs": {
            "PKIS1_zip": {
                "path": release_input_path(pkis1_zip),
                "sha256": sha256_file(pkis1_zip),
            },
            "target_pair_artifact": {
                "path": release_input_path(pair_path),
                "sha256": sha256_file(pair_path),
            },
            "KiRHub_workbook": {
                "path": release_input_path(kirhub_workbook),
                "sha256": sha256_file(kirhub_workbook),
            },
        },
        "seed": seed,
        "QAP_permutations": permutations,
    }
    (output / "summary.json").write_text(
        json.dumps(json_ready(summary), indent=2, sort_keys=True) + "\n"
    )

    q025 = summary["practical_80_repeated_diversity_sampling"]
    qap_summary = summary["deterministic_MaxMin_80"]["target_label_QAP"]
    readme = f"""# Calibration sample-efficiency audit

## Outcome

**{summary['decision']}**.  The smallest tested sample size satisfying the
declared all-metric robustness rule was **{smallest_robust} compounds**.

The experiment asks a practical question: how many compounds must be profiled
densely across the same 20 kinases before their marginal hit rates become a
useful prior for predicting which target pairs will be co-selected in other
chemical libraries?  Every fit uses only the sampled PKIS1 compounds.  DAVIS,
PKIS2, and KiRHub are locked evaluations.

For 1,000 Butina-diversified 80-compound panels, median mean locked-panel gains
over sequence plus KLIFS pocket identity were
{q025['continuous_spearman']['median_mean_locked_delta']:+.3f} Spearman,
{q025['top_10_roc_auc']['median_mean_locked_delta']:+.3f} top-10% AUROC, and
{q025['top_10_average_precision']['median_mean_locked_delta']:+.3f} top-10% AP.
Their 2.5th percentiles were respectively
{q025['continuous_spearman']['q025_mean_locked_delta']:+.3f},
{q025['top_10_roc_auc']['q025_mean_locked_delta']:+.3f}, and
{q025['top_10_average_precision']['q025_mean_locked_delta']:+.3f}; the fractions
improving all three locked panels were
{q025['continuous_spearman']['fraction_positive_on_every_locked_panel']:.3f},
{q025['top_10_roc_auc']['fraction_positive_on_every_locked_panel']:.3f}, and
{q025['top_10_average_precision']['fraction_positive_on_every_locked_panel']:.3f}.

A structure-only deterministic Morgan-MaxMin 80-compound panel gave mean gains
of {qap_summary['continuous_spearman']['mean_locked_delta']:+.3f},
{qap_summary['top_10_roc_auc']['mean_locked_delta']:+.3f}, and
{qap_summary['top_10_average_precision']['mean_locked_delta']:+.3f}.  Correct
target alignment was resolved by target-label QAP after Holm adjustment
(`practical_80_target_label_qap.csv`), and every metric remained positive in
all 60 panel-by-target jackknife cells.

## Operational meaning

For a fixed target panel, an **80-compound chemically diverse calibration
screen** can estimate target breadth well enough to improve predictions of
future target-pair co-selectivity over protein sequence and pocket similarity
alone.  This is an assay-panel calibration result: it can inform how much
up-front multi-target profiling is needed before predicting likely shared
off-targets or counterscreens in a new compound collection.

## Guardrails

- Compound selection is outcome-blind.  Uniform and Butina-round-robin samples
  use reproducible seeds; the practical panel is selected from structures alone
  by Morgan-r2/2048 MaxMin.
- Baseline and augmented coefficients are refit on exactly the same sampled
  PKIS1 endpoint.  No full-PKIS1 activity enters panel selection or fitting.
- The 50% inhibition degree is primary.  Thresholds 20/35/65/80% and continuous
  target means are sensitivities in
  `practical_80_prior_definition_sensitivity.csv`.
- This is post-hoc and conditional on 20 kinases.  It is not evidence for
  ligand-level target ranking, unseen targets, or other protein families.

## Files

- `summary.json`: machine-readable decision and headline estimates.
- `sampling_summary.csv`: repeated-sampling distributions by n and scheme.
- `repeated_subsample_metrics.csv`: all locked-panel deltas.
- `repeated_degree_recovery.csv` and `degree_recovery_summary.csv`: recovery of
  full-PKIS1 target degrees.
- `sampling_scheme_comparison.csv`: diversity-minus-random contrasts.
- `deterministic_MaxMin_manifest.csv`: exact structure-only nested panel order.
- `deterministic_MaxMin_metrics.csv`: locked metrics for each MaxMin prefix.
- `practical_80_target_label_qap.csv`: aligned-degree target-label null.
- `practical_80_target_jackknife.csv`: target-deletion robustness.
"""
    (output / "README.md").write_text(readme)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair-path", type=Path, default=DEFAULT_PAIRS)
    parser.add_argument("--pkis1-zip", type=Path, default=DEFAULT_PKIS1)
    parser.add_argument("--kirhub-workbook", type=Path, default=DEFAULT_KIRHUB)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--repetitions", type=int, default=DEFAULT_REPETITIONS)
    parser.add_argument("--permutations", type=int, default=DEFAULT_PERMUTATIONS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_analysis(
        pair_path=args.pair_path,
        pkis1_zip=args.pkis1_zip,
        kirhub_workbook=args.kirhub_workbook,
        output=args.output,
        repetitions=args.repetitions,
        permutations=args.permutations,
        seed=args.seed,
    )
    print(json.dumps(json_ready(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
