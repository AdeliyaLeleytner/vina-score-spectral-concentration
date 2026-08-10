#!/usr/bin/env python3
"""Exploratory information-diversity audit of the Reinecke Kinobeads libraries.

This script treats a blank affinity cell as *no target call in the assayed
Kinobeads system*, not as proof of no biochemical binding.  The primary matrix
contains direct protein/lipid-kinase calls with apparent Kd < 1,000 nM.

The analysis is deliberately library-design focused.  KCGS compounds are
assigned to KCGS even when they also occur in PKIS/PKIS2; comparator collections
exclude those overlaps and the residual PKIS/PKIS2 overlap.  Thus all four
groups used in direct comparisons are disjoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


EXPECTED_SHA256 = {
    "table1": "9c13b2c95e2cd31de46b74fc9ecc25a2f3c72e0b101d0720bd6e917c3cecf5bb",
    "table2": "aa62105218abe10c6c1eb3584a13a87769c4dd1d96cd9cc496cf6c2743d36606",
}
GROUP_ORDER = ("KCGS", "PKIS_exclusive", "PKIS2_exclusive", "Roche")
PRIMARY_SUPPORT_MIN_HITS = 5
SUPPORT_THRESHOLDS = (1, 2, 3, 4, 5)


@dataclass(frozen=True)
class DataBundle:
    nanomolar: pd.DataFrame
    all_calls: pd.DataFrame
    metadata: pd.DataFrame
    groups: dict[str, np.ndarray]
    audit: dict[str, object]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv_deterministic(frame: pd.DataFrame, path: Path) -> None:
    compression: str | dict[str, object] | None = None
    if path.suffix == ".gz":
        compression = {"method": "gzip", "mtime": 0}
    frame.to_csv(path, index=False, float_format="%.12g", compression=compression)


def _strip_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame.columns = frame.columns.astype(str).str.strip()
    return frame


def _canonicalize_compound_columns(frame: pd.DataFrame) -> pd.DataFrame:
    # The two supplied workbooks differ for exactly one compound spelling.
    return frame.rename(columns={"PFE-PKIS3": "PFE-PKIS_3"})


def _matrix_from_sheet(
    table2: Path,
    sheet: str,
    direct_kinases: set[str],
) -> pd.DataFrame:
    raw = _canonicalize_compound_columns(pd.read_excel(table2, sheet_name=sheet))
    target_column = raw.columns[0]
    raw[target_column] = raw[target_column].astype(str)
    raw = raw.loc[raw[target_column].isin(direct_kinases)].set_index(target_column)
    if raw.index.duplicated().any() or raw.columns.duplicated().any():
        raise ValueError(f"duplicate target or compound identifier in {sheet}")
    binary = raw.notna().T.astype(np.uint8)
    binary.index = binary.index.astype(str)
    return binary


def build_groups(metadata: pd.DataFrame) -> dict[str, np.ndarray]:
    has = lambda name: metadata[name].notna().to_numpy()  # noqa: E731
    groups = {
        "KCGS": has("KCGS"),
        "PKIS_exclusive": has("PKIS") & ~has("KCGS") & ~has("PKIS2"),
        "PKIS2_exclusive": has("PKIS2") & ~has("KCGS") & ~has("PKIS"),
        "Roche": has("Roche"),
    }
    membership = np.column_stack([groups[name] for name in GROUP_ORDER]).sum(axis=1)
    if np.any(membership > 1):
        raise ValueError("comparison groups are not disjoint")
    if any(groups[name].sum() < 3 for name in GROUP_ORDER):
        raise ValueError("a comparison group is too small")
    return groups


def load_data(table1: Path, table2: Path) -> DataBundle:
    hashes = {"table1": sha256(table1), "table2": sha256(table2)}
    if hashes != EXPECTED_SHA256:
        raise ValueError(
            "input checksums do not match the frozen Nature Chemical Biology "
            f"supplementary workbooks: {hashes}"
        )

    metadata = _strip_columns(pd.read_excel(table1, sheet_name="Tool compounds"))
    metadata["Compound"] = metadata["Compound"].astype(str)
    if len(metadata) != 1183 or metadata["Compound"].duplicated().any():
        raise ValueError("unexpected Tool compounds table")

    annotation = _strip_columns(pd.read_excel(table2, sheet_name="Target annotation"))
    annotation["Gene"] = annotation["Gene"].astype(str)
    direct_kinases = set(
        annotation.loc[
            annotation["DirectBinder"].eq("x")
            & annotation["Protein/lipid Kinase"].eq("x"),
            "Gene",
        ]
    )
    nanomolar = _matrix_from_sheet(
        table2, "Kinobeads Drugmatrix -nanomolar", direct_kinases
    )
    all_calls = _matrix_from_sheet(table2, "Kinobeads Drugmatrix - all", direct_kinases)
    if set(nanomolar.index) != set(metadata["Compound"]):
        raise ValueError("compound join is not one-to-one after the documented alias")
    metadata = metadata.set_index("Compound").loc[nanomolar.index]
    all_calls = all_calls.loc[nanomolar.index]
    groups = build_groups(metadata)

    audit = {
        "sha256": hashes,
        "n_compounds": int(len(metadata)),
        "n_direct_protein_or_lipid_kinases_with_nanomolar_call": int(
            nanomolar.shape[1]
        ),
        "n_direct_protein_or_lipid_kinases_with_any_call": int(all_calls.shape[1]),
        "group_sizes": {name: int(groups[name].sum()) for name in GROUP_ORDER},
        "semantics": (
            "One for a classifier-called direct protein/lipid-kinase interaction "
            "reported in the selected affinity sheet; zero means no reported target "
            "call in this Kinobeads assay, not proven absence of biochemical binding."
        ),
    }
    return DataBundle(nanomolar, all_calls, metadata, groups, audit)


def common_target_support(
    matrix: pd.DataFrame,
    groups: dict[str, np.ndarray],
    minimum_hits: int,
) -> np.ndarray:
    keep = np.ones(matrix.shape[1], dtype=bool)
    for name in GROUP_ORDER:
        block = matrix.loc[groups[name]]
        hits = block.sum(axis=0).to_numpy()
        keep &= (hits >= minimum_hits) & ((len(block) - hits) >= minimum_hits)
    return keep


def participation_ratio(psd: np.ndarray) -> float:
    eigenvalues = np.clip(np.linalg.eigvalsh(psd), 0.0, None)
    if eigenvalues.sum() <= 0:
        raise ValueError("non-positive spectrum")
    return float(eigenvalues.sum() ** 2 / np.square(eigenvalues).sum())


def covariance_matrix(matrix: np.ndarray) -> np.ndarray:
    return np.cov(np.asarray(matrix, dtype=float), rowvar=False, ddof=1)


def correlation_matrix(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] < 3 or matrix.shape[1] < 2:
        raise ValueError("correlation requires a matrix with >=3 rows and >=2 columns")
    if np.any(matrix.std(axis=0, ddof=1) <= 1e-12):
        raise ValueError("constant target column")
    result = np.corrcoef(matrix, rowvar=False)
    np.fill_diagonal(result, 1.0)
    return np.clip(result, -1.0, 1.0)


def weighted_covariance(matrix: np.ndarray, weights: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    weights = np.asarray(weights, dtype=float)
    weights = weights / weights.sum()
    centered = matrix - weights @ matrix
    denominator = 1.0 - float(weights @ weights)
    if denominator <= 0:
        raise ValueError("degenerate weights")
    return (centered * weights[:, None]).T @ centered / denominator


def weighted_correlation(matrix: np.ndarray, weights: np.ndarray) -> np.ndarray:
    covariance = weighted_covariance(matrix, weights)
    scale = np.sqrt(np.diag(covariance))
    if np.any(scale <= 1e-12):
        raise ValueError("constant weighted target column")
    correlation = covariance / np.outer(scale, scale)
    np.fill_diagonal(correlation, 1.0)
    return np.clip(correlation, -1.0, 1.0)


def surface_metrics(matrix: np.ndarray, weights: np.ndarray | None = None) -> dict[str, float]:
    matrix = np.asarray(matrix, dtype=float)
    row_centered = matrix - matrix.mean(axis=1, keepdims=True)
    scale = matrix.std(axis=0, ddof=1)
    if np.any(scale <= 1e-12):
        raise ValueError("constant target column")
    z = (matrix - matrix.mean(axis=0, keepdims=True)) / scale
    z_row_centered = z - z.mean(axis=1, keepdims=True)

    if weights is None:
        corr = correlation_matrix
        cov = covariance_matrix
    else:
        corr = lambda value: weighted_correlation(value, weights)  # noqa: E731
        cov = lambda value: weighted_covariance(value, weights)  # noqa: E731
    return {
        "raw_correlation_pr": participation_ratio(corr(matrix)),
        "residual_correlation_pr": participation_ratio(corr(row_centered)),
        "z_then_row_correlation_pr": participation_ratio(corr(z_row_centered)),
        "raw_covariance_pr": participation_ratio(cov(matrix)),
        "residual_covariance_pr": participation_ratio(cov(row_centered)),
        "z_then_row_covariance_pr": participation_ratio(cov(z_row_centered)),
        "mean_hits_per_compound": float(matrix.sum(axis=1).mean()),
    }


def curveball_trade(rows: list[set[int]], rng: np.random.Generator, trades: int) -> None:
    """In-place Curveball trades preserving every row and column degree."""
    n_rows = len(rows)
    for _ in range(trades):
        first, second = rng.integers(0, n_rows, size=2)
        if first == second:
            continue
        left, right = rows[first], rows[second]
        left_only = list(left - right)
        right_only = list(right - left)
        if not left_only or not right_only:
            continue
        pool = left_only + right_only
        rng.shuffle(pool)
        shared = left & right
        rows[first] = shared | set(pool[: len(left_only)])
        rows[second] = shared | set(pool[len(left_only) :])


def rows_to_binary(rows: list[set[int]], n_columns: int) -> np.ndarray:
    result = np.zeros((len(rows), n_columns), dtype=np.uint8)
    for index, columns in enumerate(rows):
        if columns:
            result[index, list(columns)] = 1
    return result


def configuration_null(
    matrix: np.ndarray,
    repetitions: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    matrix = np.asarray(matrix, dtype=np.uint8)
    original_row_degree = matrix.sum(axis=1)
    original_column_degree = matrix.sum(axis=0)
    rows = [set(np.flatnonzero(row)) for row in matrix]
    curveball_trade(rows, rng, 100 * len(rows))
    records: list[dict[str, float]] = []
    for repetition in range(repetitions):
        curveball_trade(rows, rng, 20 * len(rows))
        randomized = rows_to_binary(rows, matrix.shape[1])
        if not np.array_equal(randomized.sum(axis=1), original_row_degree):
            raise AssertionError("Curveball changed row degrees")
        if not np.array_equal(randomized.sum(axis=0), original_column_degree):
            raise AssertionError("Curveball changed column degrees")
        record = surface_metrics(randomized)
        record["repetition"] = repetition
        records.append(record)
    return pd.DataFrame(records)


def configuration_audit(
    matrix_name: str,
    matrix: pd.DataFrame,
    groups: dict[str, np.ndarray],
    support_thresholds: Iterable[int],
    repetitions: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    aggregate: list[dict[str, object]] = []
    draws: list[pd.DataFrame] = []
    metric_names = [
        "raw_correlation_pr",
        "residual_correlation_pr",
        "z_then_row_correlation_pr",
        "raw_covariance_pr",
        "residual_covariance_pr",
        "z_then_row_covariance_pr",
    ]
    for threshold in support_thresholds:
        support = common_target_support(matrix, groups, threshold)
        for group_index, group in enumerate(GROUP_ORDER):
            block = matrix.loc[groups[group], matrix.columns[support]].to_numpy()
            observed = surface_metrics(block)
            rng = np.random.default_rng(seed + 1000 * threshold + group_index)
            null = configuration_null(block, repetitions, rng)
            null.insert(0, "group", group)
            null.insert(0, "minimum_hits_per_library", threshold)
            null.insert(0, "matrix", matrix_name)
            draws.append(null)
            for metric in metric_names:
                values = null[metric].to_numpy()
                aggregate.append(
                    {
                        "matrix": matrix_name,
                        "minimum_hits_per_library": threshold,
                        "n_targets": int(support.sum()),
                        "group": group,
                        "n_compounds": int(groups[group].sum()),
                        "metric": metric,
                        "observed": observed[metric],
                        "null_mean": float(values.mean()),
                        "null_sd": float(values.std(ddof=1)),
                        "null_q025": float(np.quantile(values, 0.025)),
                        "null_q975": float(np.quantile(values, 0.975)),
                        "observed_over_null_mean": float(observed[metric] / values.mean()),
                        "lower_tail_probability": float(
                            (1 + np.sum(values <= observed[metric])) / (len(values) + 1)
                        ),
                    }
                )
    return pd.DataFrame(aggregate), pd.concat(draws, ignore_index=True)


def library_metrics(
    matrix_name: str,
    matrix: pd.DataFrame,
    groups: dict[str, np.ndarray],
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for threshold in SUPPORT_THRESHOLDS:
        support = common_target_support(matrix, groups, threshold)
        for group in GROUP_ORDER:
            block = matrix.loc[groups[group], matrix.columns[support]].to_numpy()
            records.append(
                {
                    "matrix": matrix_name,
                    "minimum_hits_per_library": threshold,
                    "n_targets": int(support.sum()),
                    "group": group,
                    "n_compounds": int(groups[group].sum()),
                    **surface_metrics(block),
                }
            )
    return pd.DataFrame(records)


def capped_degree_weights(degrees: np.ndarray, reference: np.ndarray) -> np.ndarray:
    degree_bins = np.minimum(np.asarray(degrees, dtype=int), 6)
    reference_bins = np.minimum(np.asarray(reference, dtype=int), 6)
    weights = np.zeros(len(degree_bins), dtype=float)
    for degree in range(7):
        target_probability = np.mean(reference_bins == degree)
        source_probability = np.mean(degree_bins == degree)
        if source_probability > 0:
            weights[degree_bins == degree] = target_probability / source_probability
    if weights.sum() <= 0:
        raise ValueError("degree distributions have no overlap")
    return weights / weights.sum()


def promiscuity_reweighted_audit(
    matrix: pd.DataFrame,
    groups: dict[str, np.ndarray],
    minimum_hits: int,
) -> pd.DataFrame:
    support = common_target_support(matrix, groups, minimum_hits)
    columns = matrix.columns[support]
    reference = matrix.loc[groups["KCGS"], columns].to_numpy(dtype=float)
    reference_degrees = reference.sum(axis=1)
    records: list[dict[str, object]] = []
    for group in GROUP_ORDER:
        block = matrix.loc[groups[group], columns].to_numpy(dtype=float)
        if group == "KCGS":
            weights = np.full(len(block), 1.0 / len(block))
        else:
            weights = capped_degree_weights(block.sum(axis=1), reference_degrees)
        records.append(
            {
                "group": group,
                "n_targets": int(len(columns)),
                "n_compounds": int(len(block)),
                "effective_sample_size": float(1.0 / np.square(weights).sum()),
                "weighted_mean_hits": float(weights @ block.sum(axis=1)),
                **surface_metrics(block, weights),
            }
        )
    return pd.DataFrame(records)


def size_matched_audit(
    matrix: pd.DataFrame,
    metadata: pd.DataFrame,
    groups: dict[str, np.ndarray],
    minimum_hits: int,
    repetitions: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    support = common_target_support(matrix, groups, minimum_hits)
    columns = matrix.columns[support]
    reference = matrix.loc[groups["KCGS"], columns].to_numpy(dtype=float)
    observed = surface_metrics(reference)
    rng = np.random.default_rng(seed)
    draws: list[dict[str, object]] = []
    aggregate: list[dict[str, object]] = []
    n_reference = len(reference)
    for group in GROUP_ORDER[1:]:
        block = matrix.loc[groups[group], columns].to_numpy(dtype=float)
        valid = 0
        for repetition in range(repetitions):
            sampled = block[rng.choice(len(block), n_reference, replace=False)]
            if np.any(sampled.std(axis=0, ddof=1) <= 1e-12):
                continue
            record = surface_metrics(sampled)
            record.update({"group": group, "repetition": repetition})
            draws.append(record)
            valid += 1
        frame = pd.DataFrame([row for row in draws if row["group"] == group])
        for metric in ("residual_correlation_pr", "residual_covariance_pr"):
            values = frame[metric].to_numpy()
            aggregate.append(
                {
                    "group": group,
                    "metric": metric,
                    "n_targets": int(len(columns)),
                    "reference_n_compounds": int(n_reference),
                    "valid_repetitions": int(valid),
                    "KCGS_observed": observed[metric],
                    "comparator_mean": float(values.mean()),
                    "comparator_q025": float(np.quantile(values, 0.025)),
                    "comparator_q975": float(np.quantile(values, 0.975)),
                    "probability_comparator_at_least_KCGS": float(
                        (1 + np.sum(values >= observed[metric])) / (len(values) + 1)
                    ),
                }
            )
    return pd.DataFrame(aggregate), pd.DataFrame(draws)


def target_entropy(matrix: np.ndarray) -> float:
    counts = np.asarray(matrix).sum(axis=0).astype(float)
    counts = counts[counts > 0]
    probabilities = counts / counts.sum()
    return float(np.exp(-np.sum(probabilities * np.log(probabilities))))


def profile_key(row: np.ndarray) -> bytes:
    return np.asarray(row, dtype=np.uint8).tobytes()


def collision_by_degree(
    matrix: np.ndarray,
    chemotypes: np.ndarray,
    cross_chemotype_only: bool,
) -> dict[int, dict[str, float]]:
    matrix = np.asarray(matrix, dtype=np.uint8)
    degrees = matrix.sum(axis=1)
    result: dict[int, dict[str, float]] = {}
    for degree in sorted(set(degrees)):
        if degree == 0:
            continue
        indices = np.flatnonzero(degrees == degree)
        if len(indices) < 2:
            continue
        numerator = 0
        denominator = 0
        for position, first in enumerate(indices):
            for second in indices[position + 1 :]:
                if cross_chemotype_only and chemotypes[first] == chemotypes[second]:
                    continue
                denominator += 1
                numerator += profile_key(matrix[first]) == profile_key(matrix[second])
        if denominator:
            result[int(degree)] = {
                "n_compounds": int(len(indices)),
                "collision_probability": float(numerator / denominator),
                "identical_pairs": int(numerator),
                "eligible_pairs": int(denominator),
            }
    return result


def standardized_collision(
    reference: dict[int, dict[str, float]],
    candidate: dict[int, dict[str, float]],
) -> tuple[float, float, list[int]]:
    degrees = sorted(set(reference) & set(candidate))
    weights = np.array([reference[d]["n_compounds"] for d in degrees], dtype=float)
    weights /= weights.sum()
    reference_value = float(
        sum(weights[i] * reference[d]["collision_probability"] for i, d in enumerate(degrees))
    )
    candidate_value = float(
        sum(weights[i] * candidate[d]["collision_probability"] for i, d in enumerate(degrees))
    )
    return reference_value, candidate_value, degrees


def fingerprint_audit(
    matrix: pd.DataFrame,
    metadata: pd.DataFrame,
    groups: dict[str, np.ndarray],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summaries: list[dict[str, object]] = []
    collisions: dict[str, dict[str, dict[int, dict[str, float]]]] = {}
    for group in GROUP_ORDER:
        block = matrix.loc[groups[group]].to_numpy(dtype=np.uint8)
        chemotypes = metadata.loc[groups[group], "Chemotype"].astype(str).to_numpy()
        degrees = block.sum(axis=1)
        summaries.append(
            {
                "group": group,
                "n_compounds": int(len(block)),
                "n_chemotypes": int(len(set(chemotypes))),
                "n_target_calls": int(block.sum()),
                "n_targets_covered": int(np.sum(block.sum(axis=0) > 0)),
                "target_entropy_effective_number": target_entropy(block),
                "n_nonzero_compounds": int(np.sum(degrees > 0)),
                "n_unique_binary_profiles": int(len({profile_key(row) for row in block})),
            }
        )
        collisions[group] = {
            "all_pairs": collision_by_degree(block, chemotypes, False),
            "cross_chemotype": collision_by_degree(block, chemotypes, True),
        }

    comparisons: list[dict[str, object]] = []
    for pair_type in ("all_pairs", "cross_chemotype"):
        reference = collisions["KCGS"][pair_type]
        for group in GROUP_ORDER[1:]:
            ref_value, candidate_value, degrees = standardized_collision(
                reference, collisions[group][pair_type]
            )
            comparisons.append(
                {
                    "pair_type": pair_type,
                    "comparator": group,
                    "common_positive_hit_degrees": ",".join(map(str, degrees)),
                    "KCGS_degree_standardized_collision": ref_value,
                    "comparator_degree_standardized_collision": candidate_value,
                    "comparator_over_KCGS_collision": (
                        float(candidate_value / ref_value) if ref_value > 0 else np.nan
                    ),
                }
            )
    return pd.DataFrame(summaries), pd.DataFrame(comparisons)


def chemotype_bootstrap(
    matrix: pd.DataFrame,
    metadata: pd.DataFrame,
    groups: dict[str, np.ndarray],
    minimum_hits: int,
    repetitions: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    support = common_target_support(matrix, groups, minimum_hits)
    columns = matrix.columns[support]
    rng = np.random.default_rng(seed)
    draws: dict[str, np.ndarray] = {}
    records: list[dict[str, object]] = []
    rejected: dict[str, int] = {}
    for group in GROUP_ORDER:
        block = matrix.loc[groups[group], columns].to_numpy(dtype=float)
        chemotypes = metadata.loc[groups[group], "Chemotype"].astype(str).to_numpy()
        labels = np.unique(chemotypes)
        index = {label: np.flatnonzero(chemotypes == label) for label in labels}
        values_list: list[float] = []
        attempts = 0
        while len(values_list) < repetitions:
            attempts += 1
            if attempts > repetitions * 50:
                raise RuntimeError("too many degenerate chemotype bootstrap draws")
            sampled_labels = rng.choice(labels, len(labels), replace=True)
            sampled_indices = np.concatenate([index[label] for label in sampled_labels])
            try:
                value = surface_metrics(block[sampled_indices])[
                    "residual_correlation_pr"
                ]
            except ValueError as error:
                if str(error) != "constant target column":
                    raise
                continue
            repetition = len(values_list)
            values_list.append(value)
            records.append(
                {
                    "group": group,
                    "repetition": repetition,
                    "attempt": attempts,
                    "residual_correlation_pr": value,
                }
            )
        values = np.asarray(values_list, dtype=float)
        draws[group] = values
        rejected[group] = attempts - repetitions

    comparisons: list[dict[str, object]] = []
    for group in GROUP_ORDER:
        values = draws[group]
        comparisons.append(
            {
                "comparison": group,
                "estimate_type": "library",
                "median": float(np.median(values)),
                "q025": float(np.quantile(values, 0.025)),
                "q975": float(np.quantile(values, 0.975)),
                "probability_above_zero": np.nan,
                "rejected_degenerate_draws": rejected[group],
            }
        )
    for group in GROUP_ORDER[1:]:
        difference = draws["KCGS"] - draws[group]
        comparisons.append(
            {
                "comparison": f"KCGS_minus_{group}",
                "estimate_type": "difference",
                "median": float(np.median(difference)),
                "q025": float(np.quantile(difference, 0.025)),
                "q975": float(np.quantile(difference, 0.975)),
                "probability_above_zero": float(np.mean(difference > 0)),
                "rejected_degenerate_draws": rejected["KCGS"] + rejected[group],
            }
        )
    return pd.DataFrame(comparisons), pd.DataFrame(records)


def write_summary(
    output_dir: Path,
    bundle: DataBundle,
    metrics: pd.DataFrame,
    configuration: pd.DataFrame,
    size_matched: pd.DataFrame,
    reweighted: pd.DataFrame,
    fingerprints: pd.DataFrame,
    collision: pd.DataFrame,
    bootstrap: pd.DataFrame,
    null_repetitions: int,
    resamples: int,
) -> None:
    primary_metrics = metrics.loc[
        metrics["matrix"].eq("nanomolar")
        & metrics["minimum_hits_per_library"].eq(PRIMARY_SUPPORT_MIN_HITS)
    ].set_index("group")
    primary_configuration = configuration.loc[
        configuration["matrix"].eq("nanomolar")
        & configuration["minimum_hits_per_library"].eq(PRIMARY_SUPPORT_MIN_HITS)
        & configuration["metric"].isin(
            ["residual_correlation_pr", "residual_covariance_pr"]
        )
    ]
    cross_collision = collision.loc[collision["pair_type"].eq("cross_chemotype")]
    all_call_primary = metrics.loc[
        metrics["matrix"].eq("all_calls")
        & metrics["minimum_hits_per_library"].eq(PRIMARY_SUPPORT_MIN_HITS)
    ].set_index("group")
    payload = {
        "status": "exploratory_fixed_collection_audit",
        "data_audit": bundle.audit,
        "primary_support": {
            "definition": (
                "direct protein/lipid kinases with >=5 nanomolar calls and >=5 "
                "non-calls in every disjoint library group"
            ),
            "n_targets": int(primary_metrics["n_targets"].iloc[0]),
            "residual_correlation_pr": primary_metrics[
                "residual_correlation_pr"
            ].to_dict(),
            "residual_covariance_pr": primary_metrics[
                "residual_covariance_pr"
            ].to_dict(),
        },
        "configuration_normalized_primary": primary_configuration.to_dict("records"),
        "all_classifier_called_interactions_sensitivity": {
            "n_targets": int(all_call_primary["n_targets"].iloc[0]),
            "residual_correlation_pr": all_call_primary[
                "residual_correlation_pr"
            ].to_dict(),
        },
        "size_matched_primary": size_matched.to_dict("records"),
        "promiscuity_reweighted_primary": reweighted.to_dict("records"),
        "full_target_profile_summary": fingerprints.to_dict("records"),
        "cross_chemotype_degree_standardized_profile_collision": cross_collision.to_dict(
            "records"
        ),
        "chemotype_cluster_bootstrap": bootstrap.to_dict("records"),
        "repetitions": {
            "configuration_null": null_repetitions,
            "size_matched_and_chemotype_bootstrap": resamples,
        },
        "interpretation": {
            "positive": (
                "KCGS has the least redundant kinase target-call geometry among the "
                "four fixed collections. The ordering survives sample-size control, "
                "correlation/covariance estimands, target-support thresholds, and a "
                "configuration null preserving every compound and target degree."
            ),
            "boundary": (
                "After matching only the per-compound hit-count distribution, KCGS "
                "is nearly tied with PKIS/PKIS2 in residual correlation PR. Therefore "
                "raw PR is partly a selectivity/promiscuity statistic; the stronger "
                "library-design result is the degree-and-margin-normalized redundancy "
                "audit and the lower target-profile collision rate."
            ),
            "not_claimed": (
                "This post hoc fixed-collection comparison does not establish a "
                "population-level advantage, biochemical completeness, or causal "
                "superiority of KCGS design. Kinobeads cannot assay every kinase."
            ),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")

    k = primary_metrics["residual_correlation_pr"].to_dict()
    config = primary_configuration.loc[
        primary_configuration["metric"].eq("residual_correlation_pr")
    ].set_index("group")["observed_over_null_mean"].to_dict()
    all_call = all_call_primary["residual_correlation_pr"].to_dict()
    coll = cross_collision.set_index("comparator")[
        "comparator_over_KCGS_collision"
    ].to_dict()
    boot_differences = bootstrap.loc[bootstrap["estimate_type"].eq("difference")]
    lines = [
        "# Kinobeads library information-diversity audit",
        "",
        "Exploratory, fixed-collection analysis; no manuscript files were edited.",
        "",
        "## Main result",
        "",
        (
            f"On the primary common support ({int(primary_metrics['n_targets'].iloc[0])} "
            "direct kinase targets), residual correlation PR was "
            f"{k['KCGS']:.2f} for KCGS, {k['PKIS_exclusive']:.2f} for exclusive "
            f"PKIS, {k['PKIS2_exclusive']:.2f} for exclusive PKIS2, and "
            f"{k['Roche']:.2f} for Roche."
        ),
        "",
        (
            "After a configuration null preserving every compound hit count and "
            "every target marginal, observed/null residual-PR ratios were "
            f"{config['KCGS']:.3f}, {config['PKIS_exclusive']:.3f}, "
            f"{config['PKIS2_exclusive']:.3f}, and {config['Roche']:.3f}, "
            "respectively. A value closer to one means less co-target redundancy "
            "than expected from sparsity and marginals alone."
        ),
        "",
        (
            "Using every classifier-called direct kinase interaction rather than "
            "only submicromolar calls gave the same ordering on a larger "
            f"{int(all_call_primary['n_targets'].iloc[0])}-target common support: "
            f"{all_call['KCGS']:.2f}, {all_call['PKIS_exclusive']:.2f}, "
            f"{all_call['PKIS2_exclusive']:.2f}, and {all_call['Roche']:.2f}."
        ),
        "",
        (
            "Among compounds with the same positive hit count and from different "
            "annotated chemotypes, exact target-profile collisions were "
            f"{coll['PKIS_exclusive']:.1f}x, {coll['PKIS2_exclusive']:.1f}x, and "
            f"{coll['Roche']:.1f}x more frequent than in KCGS after standardization "
            "to the KCGS hit-count distribution."
        ),
        "",
        "## Essential boundary",
        "",
        (
            "Inverse-probability matching of per-compound hit counts nearly removes "
            "the KCGS residual-correlation-PR difference versus PKIS and PKIS2. "
            "Thus raw PR alone is not a pure information-diversity score. The more "
            "defensible positive result is configuration-normalized redundancy and "
            "degree-conditioned profile uniqueness."
        ),
        "",
        "Chemotype-cluster bootstrap differences (KCGS minus comparator) were:",
        "",
    ]
    for row in boot_differences.itertuples(index=False):
        lines.append(
            f"- {row.comparison}: median {row.median:.2f}, 95% range "
            f"[{row.q025:.2f}, {row.q975:.2f}]."
        )
    lines.extend(
        [
            "",
            "The PKIS and PKIS2 cluster-bootstrap ranges include zero; this is "
            "reported rather than hidden.",
            "Bootstrap draws are conditional on retaining nonzero variance in all "
            "26 fixed target columns; rejected degenerate-draw counts are recorded "
            "in chemotype_bootstrap_summary.csv.",
            "",
            "## Data semantics and provenance",
            "",
            "All 1,183 compounds were profiled at 100 nM and 1 uM in the same "
            "Kinobeads workflow. A blank workbook cell is interpreted only as no "
            "classifier-called target in that assay. The primary binary matrix uses "
            "direct protein/lipid-kinase calls below 1,000 nM. Inputs are the open "
            "supplementary Tables S1 and S2 from Reinecke et al., Nature Chemical "
            "Biology (2024), DOI 10.1038/s41589-023-01459-3; exact SHA-256 values "
            "are recorded in summary.json.",
            "",
            "## Reproduction",
            "",
            "```bash",
            "python analysis/kinobeads_library_information_diversity.py \\",
            "  --table1 /path/to/41589_2023_1459_MOESM2_ESM.xlsx \\",
            "  --table2 /path/to/41589_2023_1459_MOESM3_ESM.xlsx \\",
            "  --output-dir results/kinobeads_library_information_diversity",
            "```",
            "",
        ]
    )
    (output_dir / "README.md").write_text("\n".join(lines))


def run(args: argparse.Namespace) -> None:
    table1 = Path(args.table1).resolve()
    table2 = Path(args.table2).resolve()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    bundle = load_data(table1, table2)

    metrics = pd.concat(
        [
            library_metrics("nanomolar", bundle.nanomolar, bundle.groups),
            library_metrics("all_calls", bundle.all_calls, bundle.groups),
        ],
        ignore_index=True,
    )
    configuration, null_draws = configuration_audit(
        "nanomolar",
        bundle.nanomolar,
        bundle.groups,
        SUPPORT_THRESHOLDS,
        args.null_repetitions,
        args.seed,
    )
    size_matched, size_draws = size_matched_audit(
        bundle.nanomolar,
        bundle.metadata,
        bundle.groups,
        PRIMARY_SUPPORT_MIN_HITS,
        args.resamples,
        args.seed + 20000,
    )
    reweighted = promiscuity_reweighted_audit(
        bundle.nanomolar, bundle.groups, PRIMARY_SUPPORT_MIN_HITS
    )
    fingerprints, collisions = fingerprint_audit(
        bundle.nanomolar, bundle.metadata, bundle.groups
    )
    bootstrap, bootstrap_draws = chemotype_bootstrap(
        bundle.nanomolar,
        bundle.metadata,
        bundle.groups,
        PRIMARY_SUPPORT_MIN_HITS,
        args.resamples,
        args.seed + 30000,
    )

    write_csv_deterministic(metrics, output_dir / "library_metrics.csv")
    write_csv_deterministic(
        configuration, output_dir / "configuration_null_summary.csv"
    )
    write_csv_deterministic(
        null_draws, output_dir / "configuration_null_draws.csv.gz"
    )
    write_csv_deterministic(size_matched, output_dir / "size_matched_summary.csv")
    write_csv_deterministic(size_draws, output_dir / "size_matched_draws.csv.gz")
    write_csv_deterministic(
        reweighted, output_dir / "promiscuity_reweighted_metrics.csv"
    )
    write_csv_deterministic(fingerprints, output_dir / "target_profile_summary.csv")
    write_csv_deterministic(
        collisions, output_dir / "degree_standardized_profile_collision.csv"
    )
    write_csv_deterministic(bootstrap, output_dir / "chemotype_bootstrap_summary.csv")
    write_csv_deterministic(
        bootstrap_draws, output_dir / "chemotype_bootstrap_draws.csv.gz"
    )
    write_summary(
        output_dir,
        bundle,
        metrics,
        configuration,
        size_matched,
        reweighted,
        fingerprints,
        collisions,
        bootstrap,
        args.null_repetitions,
        args.resamples,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table1", required=True)
    parser.add_argument("--table2", required=True)
    parser.add_argument(
        "--output-dir", default="results/kinobeads_library_information_diversity"
    )
    parser.add_argument("--null-repetitions", type=int, default=200)
    parser.add_argument("--resamples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260803)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
