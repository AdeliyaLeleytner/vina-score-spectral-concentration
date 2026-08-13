#!/usr/bin/env python3
"""Audit ligand-support dependence in a dense ligand-by-target matrix.

This module is a standalone, reusable research deliverable.  It deliberately
does not import manuscript-specific analysis code or frozen-data loaders.  The
input contract is explicit, missing and malformed values fail closed by
default, stochastic analyses are seeded, and every emitted artifact is covered
by a SHA-256 manifest.

The audit distinguishes three questions:

* What is the target-correlation spectrum of the supplied surface?
* How does the map change after removal of row and column main effects?
* How stable is the map under a stated change or subsampling of ligand support?

Neither correlation nor panel compression validates affinity prediction,
target retrieval, or biological mechanism.  The generated contracts repeat
that boundary next to the corresponding outputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd


SCHEMA_VERSION = "1.0.0"
TRANSFORMS = ("raw", "two_way_centered")
CORE_OUTPUTS = (
    "preprocessing_contract.json",
    "spectra.json",
    "target_correlation_raw.csv",
    "target_correlation_two_way_centered.csv",
    "signed_edges.csv",
)
BIOLOGICAL_BOUNDARY = (
    "Target correlations describe score co-variation on the supplied ligand "
    "support. They do not establish affinity accuracy, causal mechanism, target "
    "retrieval, or transport to unmeasured proteins or chemical space."
)
PANEL_BOUNDARY = (
    "The k-target panel minimizes redundancy under 1-|r| on the observed score "
    "map only. It is not a biologically validated target panel and does not "
    "optimize assay coverage, protein-family diversity, safety, or retrieval "
    "performance."
)


class AuditError(ValueError):
    """Raised when an explicit audit contract is violated."""


@dataclass(frozen=True)
class AuditConfig:
    input_path: Path
    output_dir: Path
    target_columns: tuple[str, ...] | None = None
    target_regex: str | None = None
    ligand_id: str | None = None
    domain_column: str | None = None
    imputation: str = "error"
    domain_quantiles: int | None = None
    pilot_sizes: tuple[int, ...] = ()
    pilot_repeats: int = 100
    top_edge_count: int = 10
    panel_k: int | None = None
    seed: int = 0


@dataclass(frozen=True)
class LoadedMatrix:
    frame_rows: int
    target_names: tuple[str, ...]
    matrix: np.ndarray
    domain: np.ndarray | None
    source_contract: dict[str, Any]
    selection_contract: dict[str, Any]
    missing_contract: dict[str, Any]
    ligand_contract: dict[str, Any]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


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
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        raise AuditError("non-finite value cannot be serialized")
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(json_ready(payload), indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    frame.to_csv(
        path,
        index=False,
        lineterminator="\n",
        float_format="%.17g",
    )


def parse_name_list(value: str, *, label: str) -> tuple[str, ...]:
    raw = value.split(",")
    names = tuple(item.strip() for item in raw)
    if not names or any(not item for item in names):
        raise AuditError(f"{label} must be a non-empty comma-separated list")
    if len(names) != len(set(names)):
        raise AuditError(f"{label} contains duplicate names")
    return names


def parse_integer_list(value: str, *, label: str) -> tuple[int, ...]:
    names = parse_name_list(value, label=label)
    try:
        values = tuple(int(item) for item in names)
    except ValueError as exc:
        raise AuditError(f"{label} must contain integers") from exc
    if any(item <= 0 for item in values):
        raise AuditError(f"{label} must contain positive integers")
    if len(values) != len(set(values)):
        raise AuditError(f"{label} contains duplicate values")
    return values


def detect_separator(path: Path) -> tuple[str, str]:
    name = path.name.lower()
    if name.endswith(".csv") or name.endswith(".csv.gz"):
        return ",", "CSV"
    if name.endswith(".tsv") or name.endswith(".tsv.gz"):
        return "\t", "TSV"
    raise AuditError("input must have a .csv, .tsv, .csv.gz, or .tsv.gz suffix")


def select_targets(
    columns: Sequence[str],
    *,
    explicit: tuple[str, ...] | None,
    pattern_text: str | None,
) -> tuple[str, ...]:
    if (explicit is None) == (pattern_text is None):
        raise AuditError("select targets with exactly one of target_columns or target_regex")
    column_set = set(columns)
    if explicit is not None:
        missing = [name for name in explicit if name not in column_set]
        if missing:
            raise AuditError(f"target columns not found: {missing}")
        selected = explicit
    else:
        try:
            pattern = re.compile(str(pattern_text))
        except re.error as exc:
            raise AuditError(f"invalid target regex: {exc}") from exc
        selected = tuple(name for name in columns if pattern.search(str(name)))
    if len(selected) < 3:
        raise AuditError("at least three target columns are required")
    return selected


def numeric_frame_fail_closed(
    raw: pd.DataFrame,
    *,
    imputation: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    converted = raw.apply(pd.to_numeric, errors="coerce")
    malformed = converted.isna() & ~raw.isna()
    malformed_count = int(malformed.to_numpy().sum())
    if malformed_count:
        examples: list[dict[str, Any]] = []
        for row_index, column_index in np.argwhere(malformed.to_numpy())[:5]:
            examples.append(
                {
                    "source_row_zero_based": int(row_index),
                    "target": str(raw.columns[column_index]),
                    "value": str(raw.iat[row_index, column_index]),
                }
            )
        raise AuditError(
            f"target matrix contains {malformed_count} nonnumeric cells; "
            f"examples={examples}"
        )

    missing_by_target = {
        str(column): int(count)
        for column, count in converted.isna().sum().items()
        if int(count) > 0
    }
    missing_count = int(converted.isna().to_numpy().sum())
    imputation_values: dict[str, float] = {}
    if missing_count and imputation == "error":
        raise AuditError(
            f"target matrix contains {missing_count} missing cells; choose an explicit "
            "imputation policy to proceed"
        )
    if missing_count:
        if imputation == "target-mean":
            values = converted.mean(axis=0)
        elif imputation == "target-median":
            values = converted.median(axis=0)
        else:
            raise AuditError(f"unsupported imputation policy: {imputation}")
        if values.isna().any():
            empty = values.index[values.isna()].tolist()
            raise AuditError(f"cannot impute all-missing target columns: {empty}")
        converted = converted.fillna(values)
        imputation_values = {str(key): float(value) for key, value in values.items()}

    matrix = converted.to_numpy(dtype=np.float64)
    if not np.isfinite(matrix).all():
        raise AuditError("target matrix contains infinite values")
    standard_deviations = matrix.std(axis=0, ddof=1)
    constant = [
        str(converted.columns[index])
        for index, value in enumerate(standard_deviations)
        if not value > 0
    ]
    if constant:
        raise AuditError(f"target columns must have non-zero variance: {constant}")
    return converted, {
        "policy": imputation,
        "missing_cells_before_imputation": missing_count,
        "missing_cells_by_target": missing_by_target,
        "imputation_values": imputation_values,
        "nonnumeric_cells": malformed_count,
        "finite_after_preprocessing": True,
    }


def load_matrix(config: AuditConfig) -> LoadedMatrix:
    input_path = config.input_path.expanduser().resolve()
    if not input_path.is_file():
        raise AuditError(f"input is not a regular file: {input_path}")
    separator, format_name = detect_separator(input_path)
    frame = pd.read_csv(input_path, sep=separator)
    if len(frame) < 4:
        raise AuditError("at least four ligand rows are required")
    if frame.columns.duplicated().any():
        raise AuditError("input contains duplicate column names")

    target_names = select_targets(
        tuple(str(column) for column in frame.columns),
        explicit=config.target_columns,
        pattern_text=config.target_regex,
    )
    role_columns = {name for name in (config.ligand_id, config.domain_column) if name}
    missing_roles = sorted(role_columns - set(frame.columns))
    if missing_roles:
        raise AuditError(f"declared role columns not found: {missing_roles}")
    overlap = sorted(set(target_names) & role_columns)
    if overlap:
        raise AuditError(f"target and role columns overlap: {overlap}")
    if config.ligand_id and config.domain_column == config.ligand_id:
        raise AuditError("ligand_id and domain_column must be different columns")

    numeric, missing_contract = numeric_frame_fail_closed(
        frame.loc[:, list(target_names)], imputation=config.imputation
    )

    ligand_contract: dict[str, Any] = {"column": config.ligand_id, "provided": False}
    if config.ligand_id:
        identifiers = frame[config.ligand_id]
        if identifiers.isna().any():
            raise AuditError("ligand identifier column contains missing values")
        ligand_contract = {
            "column": config.ligand_id,
            "provided": True,
            "unique_values": int(identifiers.nunique(dropna=False)),
            "duplicate_rows": int(identifiers.duplicated(keep=False).sum()),
            "is_unique": bool(identifiers.is_unique),
        }

    domain: np.ndarray | None = None
    if config.domain_column:
        raw_domain = frame[config.domain_column]
        numeric_domain = pd.to_numeric(raw_domain, errors="coerce")
        malformed_domain = numeric_domain.isna() & ~raw_domain.isna()
        if malformed_domain.any():
            raise AuditError("domain column contains nonnumeric values")
        if numeric_domain.isna().any():
            raise AuditError("domain column contains missing values")
        domain = numeric_domain.to_numpy(dtype=np.float64)
        if not np.isfinite(domain).all():
            raise AuditError("domain column contains infinite values")
        if np.ptp(domain) <= 0:
            raise AuditError("domain column must have non-zero range")

    return LoadedMatrix(
        frame_rows=int(len(frame)),
        target_names=target_names,
        matrix=numeric.to_numpy(dtype=np.float64),
        domain=domain,
        source_contract={
            "path": str(input_path),
            "bytes": input_path.stat().st_size,
            "sha256": sha256_file(input_path),
            "format": format_name,
            "compression": "gzip" if input_path.name.lower().endswith(".gz") else "none",
            "rows": int(len(frame)),
            "columns": int(len(frame.columns)),
        },
        selection_contract={
            "mode": "explicit_columns" if config.target_columns is not None else "regex",
            "target_regex": config.target_regex,
            "target_columns": list(target_names),
            "targets": len(target_names),
            "domain_column": config.domain_column,
        },
        missing_contract=missing_contract,
        ligand_contract=ligand_contract,
    )


def row_center(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    return matrix - matrix.mean(axis=1, keepdims=True)


def two_way_center(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    return (
        matrix
        - matrix.mean(axis=0, keepdims=True)
        - matrix.mean(axis=1, keepdims=True)
        + matrix.mean()
    )


def transform_surface(matrix: np.ndarray, transform: str) -> np.ndarray:
    if transform == "raw":
        return np.asarray(matrix, dtype=np.float64)
    if transform == "two_way_centered":
        return two_way_center(matrix)
    raise AuditError(f"unknown transformation: {transform}")


def target_correlation(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] < 3 or matrix.shape[1] < 3:
        raise AuditError("correlation maps require at least three rows and three targets")
    if not np.isfinite(matrix).all():
        raise AuditError("correlation input must be finite")
    standard_deviations = matrix.std(axis=0, ddof=1)
    if np.any(standard_deviations <= 0):
        raise AuditError("a transformed target column has zero variance")
    correlation = np.corrcoef(matrix, rowvar=False)
    if not np.isfinite(correlation).all():
        raise AuditError("target correlation map is non-finite")
    correlation = np.clip((correlation + correlation.T) / 2.0, -1.0, 1.0)
    np.fill_diagonal(correlation, 1.0)
    return correlation


def upper_triangle(matrix: np.ndarray) -> np.ndarray:
    return np.asarray(matrix)[np.triu_indices(matrix.shape[0], k=1)]


def spectrum(correlation: np.ndarray) -> dict[str, Any]:
    correlation = np.asarray(correlation, dtype=np.float64)
    targets = correlation.shape[0]
    eigenvalues = np.maximum(np.linalg.eigvalsh(correlation), 0.0)[::-1]
    total = float(eigenvalues.sum())
    probabilities = eigenvalues / total
    positive = probabilities > 0
    edges = upper_triangle(correlation)
    mean_r2 = float(np.square(edges).mean())
    pr = float(total**2 / np.square(eigenvalues).sum())
    pr_identity = float(targets / (1.0 + (targets - 1) * mean_r2))
    return {
        "targets": int(targets),
        "participation_ratio_dimension": pr,
        "mean_squared_off_diagonal_correlation": mean_r2,
        "pr_from_exact_mean_r2_identity": pr_identity,
        "pr_identity_absolute_error": abs(pr - pr_identity),
        "entropy_effective_dimension": float(
            np.exp(-np.sum(probabilities[positive] * np.log(probabilities[positive])))
        ),
        "pc1_variance_fraction": float(probabilities[0]),
        "components_for_90_percent": int(
            np.searchsorted(np.cumsum(probabilities), 0.90) + 1
        ),
        "mean_correlation": float(edges.mean()),
        "mean_absolute_correlation": float(np.abs(edges).mean()),
        "negative_edge_fraction": float(np.mean(edges < 0)),
        "eigenvalues": eigenvalues,
        "variance_fractions": probabilities,
        "identity_note": (
            "For a P-target correlation matrix, PR equals "
            "P/[1+(P-1) mean_{i!=j}(r_ij^2)]."
        ),
    }


def correlation_frame(correlation: np.ndarray, targets: Sequence[str]) -> pd.DataFrame:
    frame = pd.DataFrame(correlation, index=targets, columns=targets)
    frame.insert(0, "target", list(targets))
    return frame.reset_index(drop=True)


def edge_frame(
    correlations: dict[str, np.ndarray], targets: Sequence[str]
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for transform in TRANSFORMS:
        correlation = correlations[transform]
        for first in range(len(targets)):
            for second in range(first + 1, len(targets)):
                value = float(correlation[first, second])
                rows.append(
                    {
                        "transform": transform,
                        "target_index_a": first,
                        "target_index_b": second,
                        "target_a": targets[first],
                        "target_b": targets[second],
                        "correlation": value,
                        "absolute_correlation": abs(value),
                        "sign": "positive" if value > 0 else "negative" if value < 0 else "zero",
                    }
                )
    return pd.DataFrame(rows)


def average_ranks(values: np.ndarray) -> np.ndarray:
    return pd.Series(np.asarray(values, dtype=np.float64)).rank(method="average").to_numpy()


def spearman_correlation(first: np.ndarray, second: np.ndarray) -> float:
    first_rank = average_ranks(first)
    second_rank = average_ranks(second)
    if first_rank.std(ddof=1) <= 0 or second_rank.std(ddof=1) <= 0:
        raise AuditError("Spearman map comparison is undefined for a constant edge vector")
    return float(np.corrcoef(first_rank, second_rank)[0, 1])


def top_indices(values: np.ndarray, count: int, *, largest: bool) -> np.ndarray:
    order = np.argsort(-values if largest else values, kind="stable")
    return order[:count]


def precision_between(first: np.ndarray, second: np.ndarray) -> float:
    return float(len(set(first.tolist()) & set(second.tolist())) / len(first))


def compare_edge_vectors(
    first: np.ndarray,
    second: np.ndarray,
    *,
    top_edge_count: int,
) -> dict[str, float]:
    if len(first) != len(second) or top_edge_count > len(first):
        raise AuditError("edge-comparison dimensions or top-edge count are invalid")
    return {
        "global_edge_spearman": spearman_correlation(first, second),
        "edge_sign_agreement_fraction": float(np.mean(np.sign(first) == np.sign(second))),
        "mean_absolute_edge_difference": float(np.mean(np.abs(first - second))),
        "root_mean_squared_edge_difference": float(np.sqrt(np.mean(np.square(first - second)))),
        "top_positive_edge_precision": precision_between(
            top_indices(first, top_edge_count, largest=True),
            top_indices(second, top_edge_count, largest=True),
        ),
        "top_negative_edge_precision": precision_between(
            top_indices(first, top_edge_count, largest=False),
            top_indices(second, top_edge_count, largest=False),
        ),
    }


def quantile_domain_analysis(
    matrix: np.ndarray,
    domain: np.ndarray,
    targets: Sequence[str],
    *,
    quantiles: int,
    top_edge_count: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if quantiles < 2:
        raise AuditError("domain_quantiles must be at least two")
    boundaries = np.quantile(domain, np.linspace(0.0, 1.0, quantiles + 1))
    if np.any(np.diff(boundaries) <= 0):
        raise AuditError(
            "domain quantile boundaries are not strictly increasing; reduce the number "
            "of bins or use a less discrete domain"
        )
    bin_index = np.searchsorted(boundaries[1:-1], domain, side="right")
    counts = np.bincount(bin_index, minlength=quantiles)
    if np.any(counts < 3):
        raise AuditError("each domain quantile must contain at least three rows")

    map_vectors: dict[tuple[str, int], np.ndarray] = {}
    edge_rows: list[dict[str, Any]] = []
    for transform in TRANSFORMS:
        for bin_number in range(quantiles):
            mask = bin_index == bin_number
            transformed = transform_surface(matrix[mask], transform)
            correlation = target_correlation(transformed)
            vector = upper_triangle(correlation)
            map_vectors[(transform, bin_number)] = vector
            edge_cursor = 0
            for first in range(len(targets)):
                for second in range(first + 1, len(targets)):
                    edge_rows.append(
                        {
                            "transform": transform,
                            "domain_bin": bin_number + 1,
                            "rows": int(mask.sum()),
                            "domain_quantile_lower": float(boundaries[bin_number]),
                            "domain_quantile_upper": float(boundaries[bin_number + 1]),
                            "domain_observed_min": float(domain[mask].min()),
                            "domain_observed_max": float(domain[mask].max()),
                            "target_a": targets[first],
                            "target_b": targets[second],
                            "correlation": float(vector[edge_cursor]),
                        }
                    )
                    edge_cursor += 1

    comparison_rows: list[dict[str, Any]] = []
    for transform in TRANSFORMS:
        for first_bin in range(quantiles):
            for second_bin in range(first_bin + 1, quantiles):
                metrics = compare_edge_vectors(
                    map_vectors[(transform, first_bin)],
                    map_vectors[(transform, second_bin)],
                    top_edge_count=top_edge_count,
                )
                comparison_rows.append(
                    {
                        "transform": transform,
                        "domain_bin_a": first_bin + 1,
                        "domain_bin_b": second_bin + 1,
                        "rows_a": int(counts[first_bin]),
                        "rows_b": int(counts[second_bin]),
                        **metrics,
                    }
                )
    contract = {
        "column_is_numeric_and_complete": True,
        "quantile_bins": quantiles,
        "quantile_boundaries": boundaries,
        "rows_per_bin": counts,
        "boundary_rule": (
            "Interior boundaries use numpy quantiles and values equal to an interior "
            "boundary enter the higher-numbered bin."
        ),
        "interpretation_boundary": BIOLOGICAL_BOUNDARY,
    }
    return pd.DataFrame(edge_rows), pd.DataFrame(comparison_rows), contract


def distribution_summary(values: Iterable[float]) -> dict[str, float | int | list[float]]:
    array = np.asarray(list(values), dtype=np.float64)
    if not len(array) or not np.isfinite(array).all():
        raise AuditError("cannot summarize an empty or non-finite distribution")
    return {
        "n": int(len(array)),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "interval_95": [
            float(np.quantile(array, 0.025)),
            float(np.quantile(array, 0.975)),
        ],
    }


def pilot_recovery_analysis(
    matrix: np.ndarray,
    *,
    sizes: Sequence[int],
    repeats: int,
    top_edge_count: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    rows = len(matrix)
    if repeats <= 0:
        raise AuditError("pilot_repeats must be positive")
    if not sizes:
        raise AuditError("pilot recovery requires at least one size")
    for size in sizes:
        if size < 3 or rows - size < 3:
            raise AuditError(
                f"pilot size {size} must leave at least three sampled and three "
                "row-disjoint reference rows"
            )
    rng = np.random.default_rng(seed)
    replicate_rows: list[dict[str, Any]] = []
    for size in sizes:
        for repeat in range(repeats):
            permutation = rng.permutation(rows)
            sample = np.sort(permutation[:size])
            reference = np.sort(permutation[size:])
            support_hash = hashlib.sha256(sample.astype("<i8").tobytes()).hexdigest()
            for transform in TRANSFORMS:
                sample_map = target_correlation(
                    transform_surface(matrix[sample], transform)
                )
                reference_map = target_correlation(
                    transform_surface(matrix[reference], transform)
                )
                metrics = compare_edge_vectors(
                    upper_triangle(sample_map),
                    upper_triangle(reference_map),
                    top_edge_count=top_edge_count,
                )
                replicate_rows.append(
                    {
                        "transform": transform,
                        "pilot_rows": size,
                        "reference_rows": rows - size,
                        "repeat": repeat,
                        "sample_index_sha256": support_hash,
                        **metrics,
                    }
                )
    replicates = pd.DataFrame(replicate_rows)
    metrics = (
        "global_edge_spearman",
        "edge_sign_agreement_fraction",
        "mean_absolute_edge_difference",
        "root_mean_squared_edge_difference",
        "top_positive_edge_precision",
        "top_negative_edge_precision",
    )
    summary_rows: list[dict[str, Any]] = []
    for transform in TRANSFORMS:
        for size in sizes:
            selected = replicates.loc[
                replicates["transform"].eq(transform)
                & replicates["pilot_rows"].eq(size)
            ]
            row: dict[str, Any] = {
                "transform": transform,
                "pilot_rows": size,
                "reference_rows": rows - size,
                "repeats": repeats,
                "reference_scope": "row_disjoint_complement_map",
            }
            for metric in metrics:
                summary = distribution_summary(selected[metric])
                row[f"{metric}_mean"] = summary["mean"]
                row[f"{metric}_median"] = summary["median"]
                row[f"{metric}_q025"] = summary["interval_95"][0]
                row[f"{metric}_q975"] = summary["interval_95"][1]
            summary_rows.append(row)
    contract = {
        "seed": seed,
        "repeats": repeats,
        "pilot_sizes": list(sizes),
        "top_edge_count": top_edge_count,
        "reference_scope": "row_disjoint_complement_map",
        "sampling": "simple random rows without replacement within each repeat",
        "interpretation_boundary": (
            "Recovery quantifies reconstruction of this score-derived map on held-out "
            "rows. It is not an experimental validation of target selection."
        ),
    }
    return replicates, pd.DataFrame(summary_rows), contract


def k_medoids(distance: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, float]:
    distance = np.asarray(distance, dtype=np.float64)
    items = len(distance)
    if distance.shape != (items, items) or k < 1 or k > items:
        raise AuditError("panel_k must be between one and the number of targets")
    if not np.isfinite(distance).all() or np.any(distance < -1e-12):
        raise AuditError("medoid distance matrix must be finite and nonnegative")
    if not np.allclose(distance, distance.T, atol=1e-12):
        raise AuditError("medoid distance matrix must be symmetric")

    medoids = [int(np.argmin(distance.sum(axis=1)))]
    while len(medoids) < k:
        nearest = distance[:, medoids].min(axis=1)
        nearest[np.asarray(medoids)] = -np.inf
        medoids.append(int(np.argmax(nearest)))
    medoids = sorted(medoids)

    def objective(candidate: Sequence[int]) -> float:
        return float(distance[:, np.asarray(candidate)].min(axis=1).sum())

    current = objective(medoids)
    for _ in range(100):
        best = tuple(medoids)
        best_objective = current
        non_medoids = [index for index in range(items) if index not in medoids]
        for outgoing in medoids:
            for incoming in non_medoids:
                candidate = tuple(sorted((set(medoids) - {outgoing}) | {incoming}))
                candidate_objective = objective(candidate)
                if candidate_objective < best_objective - 1e-12 or (
                    abs(candidate_objective - best_objective) <= 1e-12
                    and candidate < best
                ):
                    best = candidate
                    best_objective = candidate_objective
        if best_objective >= current - 1e-12:
            break
        medoids = list(best)
        current = best_objective
    else:  # pragma: no cover - the finite swap search should converge well before this.
        raise AuditError("k-medoids did not converge within 100 swap iterations")

    medoid_array = np.asarray(sorted(medoids), dtype=int)
    assignments = np.argmin(distance[:, medoid_array], axis=1)
    for cluster_index, medoid in enumerate(medoid_array):
        assignments[medoid] = cluster_index
    return medoid_array, assignments, current


def target_panel_analysis(
    correlations: dict[str, np.ndarray],
    targets: Sequence[str],
    *,
    panel_k: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    summaries: dict[str, Any] = {}
    for transform in TRANSFORMS:
        distance = 1.0 - np.abs(correlations[transform])
        np.fill_diagonal(distance, 0.0)
        medoids, assignments, objective = k_medoids(distance, panel_k)
        summaries[transform] = {
            "panel_k": panel_k,
            "medoid_targets": [targets[index] for index in medoids],
            "sum_nearest_medoid_distance": objective,
            "distance": "1 - absolute target correlation",
        }
        for target_index, cluster_index in enumerate(assignments):
            medoid_index = int(medoids[cluster_index])
            rows.append(
                {
                    "transform": transform,
                    "cluster": int(cluster_index + 1),
                    "medoid_target": targets[medoid_index],
                    "member_target": targets[target_index],
                    "member_is_medoid": bool(target_index == medoid_index),
                    "distance_to_medoid": float(distance[target_index, medoid_index]),
                }
            )
    return pd.DataFrame(rows), {
        "method": "deterministic farthest-first initialization followed by PAM swaps",
        "transforms": summaries,
        "biological_boundary_warning": PANEL_BOUNDARY,
    }


def row_centering_invariance_record(matrix: np.ndarray) -> dict[str, Any]:
    centered = row_center(matrix)
    reference_before = matrix[:, 1:] - matrix[:, [0]]
    reference_after = centered[:, 1:] - centered[:, [0]]
    return {
        "operation": "subtract the within-row arithmetic mean without target scaling",
        "theoretical_result": (
            "Within each row, every pairwise target difference and therefore every "
            "within-row target rank is unchanged in exact arithmetic."
        ),
        "maximum_absolute_reference_difference_error": float(
            np.max(np.abs(reference_before - reference_after))
        ),
        "important_distinction": (
            "This invariance applies to row-centering alone. Target-specific scaling "
            "or other column-wise transformations can change within-row rankings."
        ),
    }


def validate_config(config: AuditConfig) -> None:
    if config.imputation not in {"error", "target-mean", "target-median"}:
        raise AuditError("imputation must be error, target-mean, or target-median")
    if config.seed < 0:
        raise AuditError("seed must be nonnegative")
    if config.top_edge_count <= 0:
        raise AuditError("top_edge_count must be positive")
    if config.domain_quantiles is not None and config.domain_column is None:
        raise AuditError("domain_quantiles requires domain_column")
    if config.domain_column is not None and config.domain_quantiles is None:
        raise AuditError("domain_column requires domain_quantiles")
    output = config.output_dir.expanduser().resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise AuditError("output directory must not exist or must be empty")


def run_audit(config: AuditConfig) -> dict[str, Any]:
    validate_config(config)
    loaded = load_matrix(config)
    edge_count = len(loaded.target_names) * (len(loaded.target_names) - 1) // 2
    if (config.pilot_sizes or config.domain_quantiles is not None) and (
        config.top_edge_count > edge_count
    ):
        raise AuditError(
            f"top_edge_count={config.top_edge_count} exceeds {edge_count} target-pair edges"
        )
    if config.panel_k is not None and not 1 <= config.panel_k <= len(loaded.target_names):
        raise AuditError("panel_k must be between one and the number of targets")

    correlations = {
        transform: target_correlation(transform_surface(loaded.matrix, transform))
        for transform in TRANSFORMS
    }
    spectra = {
        "schema_version": SCHEMA_VERSION,
        "surfaces": {
            transform: spectrum(correlations[transform]) for transform in TRANSFORMS
        },
        "interpretation_boundary": BIOLOGICAL_BOUNDARY,
    }
    contracts: dict[str, Any] = {}
    optional_frames: dict[str, pd.DataFrame] = {}
    optional_json: dict[str, dict[str, Any]] = {}

    if config.domain_quantiles is not None:
        if loaded.domain is None:  # Defensive; validate_config/load_matrix enforce this.
            raise AuditError("domain values are unavailable")
        bin_edges, comparisons, contract = quantile_domain_analysis(
            loaded.matrix,
            loaded.domain,
            loaded.target_names,
            quantiles=config.domain_quantiles,
            top_edge_count=config.top_edge_count,
        )
        optional_frames["domain_bin_edges.csv"] = bin_edges
        optional_frames["domain_map_comparisons.csv"] = comparisons
        contracts["quantile_domain_analysis"] = contract

    if config.pilot_sizes:
        replicates, summary, contract = pilot_recovery_analysis(
            loaded.matrix,
            sizes=config.pilot_sizes,
            repeats=config.pilot_repeats,
            top_edge_count=config.top_edge_count,
            seed=config.seed,
        )
        optional_frames["pilot_recovery_replicates.csv"] = replicates
        optional_frames["pilot_recovery_summary.csv"] = summary
        contracts["pilot_recovery"] = contract

    if config.panel_k is not None:
        assignments, panel_summary = target_panel_analysis(
            correlations,
            loaded.target_names,
            panel_k=config.panel_k,
        )
        optional_frames["target_panel_medoids.csv"] = assignments
        optional_json["target_panel_summary.json"] = panel_summary
        contracts["target_panel"] = panel_summary

    script_path = Path(__file__).resolve()
    preprocessing = {
        "schema_version": SCHEMA_VERSION,
        "generated_by": {
            "path": str(script_path),
            "sha256": sha256_file(script_path),
        },
        "source": loaded.source_contract,
        "selection": loaded.selection_contract,
        "ligand_identifier": loaded.ligand_contract,
        "missing_data": loaded.missing_contract,
        "seed": config.seed,
        "surface_definitions": {
            "raw": (
                "Pearson target correlations across the supplied ligand rows; Pearson "
                "correlation internally centers and scales each target column."
            ),
            "two_way_centered": (
                "Subtract row means and column means and add the grand mean, then "
                "compute Pearson target correlations."
            ),
        },
        "row_centering_rank_invariance": row_centering_invariance_record(loaded.matrix),
        "optional_analyses": contracts,
        "interpretation_boundary": BIOLOGICAL_BOUNDARY,
    }

    # All numerical work and validation complete before the output directory is
    # created, so a failed contract does not leave a plausible partial audit.
    output_dir = config.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "preprocessing_contract.json", preprocessing)
    write_json(output_dir / "spectra.json", spectra)
    write_csv(
        output_dir / "target_correlation_raw.csv",
        correlation_frame(correlations["raw"], loaded.target_names),
    )
    write_csv(
        output_dir / "target_correlation_two_way_centered.csv",
        correlation_frame(correlations["two_way_centered"], loaded.target_names),
    )
    write_csv(output_dir / "signed_edges.csv", edge_frame(correlations, loaded.target_names))
    for filename, frame in optional_frames.items():
        write_csv(output_dir / filename, frame)
    for filename, payload in optional_json.items():
        write_json(output_dir / filename, payload)

    artifact_names = sorted(
        [*CORE_OUTPUTS, *optional_frames.keys(), *optional_json.keys()]
    )
    checksum_payload = {
        "algorithm": "sha256",
        "schema_version": SCHEMA_VERSION,
        "files": {
            filename: {
                "bytes": (output_dir / filename).stat().st_size,
                "sha256": sha256_file(output_dir / filename),
            }
            for filename in artifact_names
        },
    }
    write_json(output_dir / "output_checksums.json", checksum_payload)
    return {
        "output_dir": str(output_dir),
        "rows": loaded.frame_rows,
        "targets": len(loaded.target_names),
        "artifacts": [*artifact_names, "output_checksums.json"],
        "raw_pr_dimension": spectra["surfaces"]["raw"][
            "participation_ratio_dimension"
        ],
        "two_way_centered_pr_dimension": spectra["surfaces"][
            "two_way_centered"
        ]["participation_ratio_dimension"],
        "interpretation_boundary": BIOLOGICAL_BOUNDARY,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="dense CSV/TSV/CSV.GZ/TSV.GZ table")
    parser.add_argument("--output-dir", type=Path, required=True)
    target_group = parser.add_mutually_exclusive_group(required=True)
    target_group.add_argument(
        "--target-columns",
        help="comma-separated target columns in the required output order",
    )
    target_group.add_argument(
        "--target-regex",
        help="regular expression searched against input column names",
    )
    parser.add_argument("--ligand-id", help="optional ligand identifier column")
    parser.add_argument("--domain-column", help="optional complete numeric domain column")
    parser.add_argument(
        "--imputation",
        choices=("error", "target-mean", "target-median"),
        default="error",
        help="missing-target policy; malformed nonnumeric values always fail",
    )
    parser.add_argument(
        "--domain-quantiles",
        type=int,
        help="compare maps across this many quantiles of --domain-column",
    )
    parser.add_argument(
        "--pilot-sizes",
        help="comma-separated row counts for repeated held-out map recovery",
    )
    parser.add_argument("--pilot-repeats", type=int, default=100)
    parser.add_argument(
        "--top-edge-count",
        type=int,
        default=10,
        help="number of strongest positive and negative edges in support comparisons",
    )
    parser.add_argument(
        "--panel-k",
        type=int,
        help="optional number of medoid targets under distance 1-|r|",
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser


def config_from_args(args: argparse.Namespace) -> AuditConfig:
    return AuditConfig(
        input_path=args.input,
        output_dir=args.output_dir,
        target_columns=(
            parse_name_list(args.target_columns, label="target_columns")
            if args.target_columns
            else None
        ),
        target_regex=args.target_regex,
        ligand_id=args.ligand_id,
        domain_column=args.domain_column,
        imputation=args.imputation,
        domain_quantiles=args.domain_quantiles,
        pilot_sizes=(
            parse_integer_list(args.pilot_sizes, label="pilot_sizes")
            if args.pilot_sizes
            else ()
        ),
        pilot_repeats=args.pilot_repeats,
        top_edge_count=args.top_edge_count,
        panel_k=args.panel_k,
        seed=args.seed,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    try:
        config = config_from_args(parser.parse_args(argv))
        summary = run_audit(config)
    except (AuditError, OSError, pd.errors.ParserError) as exc:
        print(f"target-map-audit: error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(json_ready(summary), indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
