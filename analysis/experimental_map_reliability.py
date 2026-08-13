#!/usr/bin/env python3
"""Reliability limits and same-support geometry for public kinase panels.

This analysis answers two related reviewer questions using only the fully
redistributed public kinase panels.

1. How reproducible are the DAVIS and PKIS2 target-correlation maps under
   ligand sampling, and are the previously reported docking--experiment
   concordances close to a sampling-noise limit?
2. What is the docking--experiment geometry concordance when both maps are
   estimated on exactly the same compounds and the same 21 targets?

The primary geometry is the strict upper triangle of the target correlation
matrix after two-way centring.  Random disjoint split halves estimate map
repeatability directly at the observed half-panel size.  We deliberately do not
apply Spearman--Brown or correction for attenuation: a nonlinear correlation
map is not a summed parallel-test score, and docking and experiment cannot be
assumed to be two noisy measurements of one latent construct.  Cluster-disjoint
splits, paired ligand/Butina bootstraps, edge-sign stability, top-decile
stability, a cross-panel experimental benchmark and OAS covariance shrinkage
are complementary diagnostics.

DAVIS has 59 exact full-Standard-InChIKey matches to DOCKSTRING, but three
matched profiles are constant at the released pKd=5 censoring floor.  The
primary same-support DAVIS estimand therefore uses the 56 informative profiles;
the 59-row exact-match surface is retained as a sensitivity.  All 154 exact
PKIS2 matches are nonconstant.

All inputs used here are redistributed in the repository.  The module is
dependency-closed on the public DAVIS, PKIS2 and DOCKSTRING inputs and does not
read or import any historical source-restricted analysis branch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import warnings
from collections import OrderedDict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import scipy
from scipy import stats
from sklearn.covariance import OAS

# Several older analysis modules support direct-script imports only.  Add the
# local analysis directory explicitly so this new module behaves identically
# when imported by pytest and when executed as a script.
ANALYSIS_DIR = Path(__file__).resolve().parent
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

import dense_davis_benchmark as dense_davis  # noqa: E402
import dense_pkis2_benchmark as dense_pkis2  # noqa: E402


PACKAGE = ANALYSIS_DIR.parent
DEFAULT_OUTPUT = PACKAGE / "results" / "experimental_map_reliability"
DEFAULT_DOCKSTRING = PACKAGE / "data" / "frozen" / "dockstring-dataset.tsv.gz"
DEFAULT_DAVIS = PACKAGE / "data" / "frozen" / "davis_complete.tab.gz"
DEFAULT_PKIS2 = PACKAGE / "data" / "frozen" / "pkis2_s4.xlsx"
DEFAULT_IDENTITY_CONTRACT = (
    PACKAGE / "data" / "frozen" / "dockstring_identity_contract_2026-08-03.csv.gz"
)
TARGETS = tuple(dense_davis.TARGET_MAP)
if tuple(dense_pkis2.TARGET_MAP) != TARGETS:
    raise RuntimeError("public DAVIS and PKIS2 target orders no longer agree")
TARGET_PAIRS = len(TARGETS) * (len(TARGETS) - 1) // 2
PRIMARY_TRANSFORM = "two_way_centered"
TOP_FRACTION = 0.10
DEFAULT_SPLIT_REPEATS = 2_000
DEFAULT_BOOTSTRAPS = 2_000
DEFAULT_SEED = 20260810
PANEL_SEED_STRIDE = 1_000_000

OUTPUT_FILES = (
    "README.md",
    "cross_panel_geometry.csv",
    "edge_stability.csv",
    "map_reliability.csv",
    "same_support_bootstrap.csv",
    "same_support_geometry.csv",
    "same_support_split_half.csv",
    "summary.json",
)


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return [json_ready(item) for item in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def describe(values: Iterable[float]) -> dict[str, Any]:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    if not len(array):
        return {
            "n": 0,
            "mean": None,
            "median": None,
            "standard_deviation": None,
            "interval_95": [None, None],
            "minimum": None,
            "maximum": None,
        }
    return {
        "n": int(len(array)),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "standard_deviation": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        "interval_95": [
            float(np.quantile(array, 0.025)),
            float(np.quantile(array, 0.975)),
        ],
        "minimum": float(array.min()),
        "maximum": float(array.max()),
    }


def informative_rows(matrix: np.ndarray, tolerance: float = 1e-12) -> np.ndarray:
    """Rows with at least one target contrast on the observed assay scale."""
    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("informative-row filtering requires a finite 2-D matrix")
    return np.ptp(values, axis=1) > tolerance


def two_way_center(matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("two-way centring requires a finite 2-D matrix")
    return (
        values
        - values.mean(axis=0, keepdims=True)
        - values.mean(axis=1, keepdims=True)
        + values.mean()
    )


def target_correlation(matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 3 or values.shape[1] < 2:
        raise ValueError("target correlation requires >=3 rows and >=2 targets")
    if not np.isfinite(values).all():
        raise ValueError("target correlation input contains non-finite values")
    scales = values.std(axis=0, ddof=1)
    if np.any(scales <= 1e-14):
        raise ValueError("target correlation matrix contains a constant target")
    correlation = np.corrcoef(values, rowvar=False)
    correlation = np.clip((correlation + correlation.T) / 2.0, -1.0, 1.0)
    np.fill_diagonal(correlation, 1.0)
    return correlation


def transformed_surface(matrix: np.ndarray, transform: str) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("geometry requires a finite 2-D matrix")
    if transform == "raw":
        return values
    if transform == "two_way_centered":
        return two_way_center(values)
    raise ValueError(f"unknown transform: {transform}")


def covariance_to_correlation(covariance: np.ndarray) -> np.ndarray:
    covariance = np.asarray(covariance, dtype=np.float64)
    diagonal = np.diag(covariance)
    if np.any(diagonal <= 1e-15) or not np.isfinite(covariance).all():
        raise ValueError("covariance matrix has a nonpositive or invalid diagonal")
    scale = np.sqrt(diagonal)
    correlation = covariance / np.outer(scale, scale)
    correlation = np.clip((correlation + correlation.T) / 2.0, -1.0, 1.0)
    np.fill_diagonal(correlation, 1.0)
    return correlation


def target_map(
    matrix: np.ndarray,
    transform: str = PRIMARY_TRANSFORM,
    estimator: str = "empirical",
) -> np.ndarray:
    surface = transformed_surface(matrix, transform)
    if estimator == "empirical":
        return target_correlation(surface)
    if estimator == "oas":
        return covariance_to_correlation(OAS().fit(surface).covariance_)
    raise ValueError(f"unknown covariance estimator: {estimator}")


def upper(matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("upper triangle requires a square matrix")
    return values[np.triu_indices(len(values), k=1)]


def map_spearman(first: np.ndarray, second: np.ndarray) -> float:
    value = stats.spearmanr(upper(first), upper(second)).statistic
    if not np.isfinite(value):
        raise ValueError("map Spearman correlation is not finite")
    return float(value)


def deterministic_top_indices(values: np.ndarray, fraction: float = TOP_FRACTION) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64)
    if vector.ndim != 1 or not np.isfinite(vector).all():
        raise ValueError("top-edge selection requires a finite vector")
    if not 0.0 < fraction < 1.0:
        raise ValueError("top fraction must lie in (0, 1)")
    count = max(1, int(np.ceil(fraction * len(vector))))
    return np.sort(np.argsort(vector, kind="mergesort")[-count:])


def set_overlap(first: np.ndarray, second: np.ndarray) -> tuple[float, float]:
    a = set(np.asarray(first, dtype=int).tolist())
    b = set(np.asarray(second, dtype=int).tolist())
    if not a or not b:
        raise ValueError("edge sets must be nonempty")
    intersection = len(a & b)
    return float(intersection / min(len(a), len(b))), float(intersection / len(a | b))


def cluster_disjoint_halves(
    labels: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Randomized greedy split with no chemical cluster crossing the halves."""
    labels = np.asarray(labels, dtype=object)
    unique = np.unique(labels)
    if len(unique) < 4:
        raise ValueError("cluster-disjoint splitting requires at least four clusters")
    members = {label: np.flatnonzero(labels == label) for label in unique}
    order = unique[rng.permutation(len(unique))]
    first: list[int] = []
    second: list[int] = []
    for label in order:
        destination = first if len(first) <= len(second) else second
        destination.extend(members[label].tolist())
    if not first or not second:
        raise RuntimeError("cluster split produced an empty half")
    return np.asarray(first, dtype=int), np.asarray(second, dtype=int)


def split_indices(
    rows: int,
    split_unit: str,
    rng: np.random.Generator,
    cluster_labels: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    if split_unit == "ligand":
        order = rng.permutation(rows)
        half = rows // 2
        return order[:half], order[half : 2 * half]
    if split_unit == "butina_cluster":
        if cluster_labels is None or len(cluster_labels) != rows:
            raise ValueError("cluster labels are required for a cluster split")
        return cluster_disjoint_halves(cluster_labels, rng)
    raise ValueError(f"unknown split unit: {split_unit}")


def split_half_diagnostics(
    matrix: np.ndarray,
    *,
    panel: str,
    support: str,
    transform: str,
    estimator: str,
    split_unit: str,
    repeats: int,
    seed: int,
    cluster_labels: np.ndarray | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Independent-half map reliability plus edge-level decision stability."""
    values = np.asarray(matrix, dtype=np.float64)
    if len(values) < 8 or repeats < 1:
        raise ValueError("split-half analysis requires >=8 rows and >=1 repeat")
    full = target_map(values, transform, estimator)
    full_top = deterministic_top_indices(upper(full))
    rng = np.random.default_rng(seed)
    records: list[dict[str, Any]] = []
    for repetition in range(repeats):
        first, second = split_indices(
            len(values), split_unit, rng, cluster_labels
        )
        try:
            map_first = target_map(values[first], transform, estimator)
            map_second = target_map(values[second], transform, estimator)
        except ValueError as error:
            records.append(
                {
                    "panel": panel,
                    "support": support,
                    "transform": transform,
                    "estimator": estimator,
                    "split_unit": split_unit,
                    "repetition": repetition,
                    "first_rows": int(len(first)),
                    "second_rows": int(len(second)),
                    "valid_split": False,
                    "invalid_reason": str(error),
                    "split_map_spearman": np.nan,
                    "edge_sign_agreement_fraction": np.nan,
                    "target_label_null_sign_agreement_fraction": np.nan,
                    "sign_agreement_above_target_label_null": np.nan,
                    "top_decile_precision_between_halves": np.nan,
                    "target_label_null_top_decile_precision": np.nan,
                    "top_decile_precision_above_target_label_null": np.nan,
                    "top_decile_jaccard_between_halves": np.nan,
                    "target_label_null_top_decile_jaccard": np.nan,
                    "mean_half_to_full_top_decile_precision": np.nan,
                    "mean_half_to_full_map_spearman": np.nan,
                }
            )
            continue
        edge_first = upper(map_first)
        edge_second = upper(map_second)
        reliability = map_spearman(map_first, map_second)
        top_first = deterministic_top_indices(edge_first)
        top_second = deterministic_top_indices(edge_second)
        top_precision, top_jaccard = set_overlap(top_first, top_second)
        target_order = rng.permutation(map_second.shape[0])
        null_second = map_second[np.ix_(target_order, target_order)]
        null_edges = upper(null_second)
        null_top = deterministic_top_indices(null_edges)
        null_precision, null_jaccard = set_overlap(top_first, null_top)
        sign_agreement = float(
            np.mean(np.sign(edge_first) == np.sign(edge_second))
        )
        null_sign_agreement = float(
            np.mean(np.sign(edge_first) == np.sign(null_edges))
        )
        first_full_precision, _ = set_overlap(top_first, full_top)
        second_full_precision, _ = set_overlap(top_second, full_top)
        records.append(
            {
                "panel": panel,
                "support": support,
                "transform": transform,
                "estimator": estimator,
                "split_unit": split_unit,
                "repetition": repetition,
                "first_rows": int(len(first)),
                "second_rows": int(len(second)),
                "valid_split": True,
                "invalid_reason": "",
                "split_map_spearman": reliability,
                "edge_sign_agreement_fraction": sign_agreement,
                "target_label_null_sign_agreement_fraction": null_sign_agreement,
                "sign_agreement_above_target_label_null": float(
                    sign_agreement - null_sign_agreement
                ),
                "top_decile_precision_between_halves": top_precision,
                "target_label_null_top_decile_precision": null_precision,
                "top_decile_precision_above_target_label_null": float(
                    top_precision - null_precision
                ),
                "top_decile_jaccard_between_halves": top_jaccard,
                "target_label_null_top_decile_jaccard": null_jaccard,
                "mean_half_to_full_top_decile_precision": float(
                    (first_full_precision + second_full_precision) / 2.0
                ),
                "mean_half_to_full_map_spearman": float(
                    (
                        map_spearman(map_first, full)
                        + map_spearman(map_second, full)
                    )
                    / 2.0
                ),
            }
        )
    frame = pd.DataFrame.from_records(records)
    summary = {
        "panel": panel,
        "support": support,
        "ligands": int(len(values)),
        "targets": int(values.shape[1]),
        "target_pairs": int(values.shape[1] * (values.shape[1] - 1) // 2),
        "transform": transform,
        "estimator": estimator,
        "split_unit": split_unit,
        "repeats": int(repeats),
        "valid_repeats": int(frame.valid_split.sum()),
        "invalid_repeats": int((~frame.valid_split).sum()),
        "valid_fraction": float(frame.valid_split.mean()),
        "seed": int(seed),
        "split_map_spearman": describe(frame.split_map_spearman),
        "edge_sign_agreement_fraction": describe(
            frame.edge_sign_agreement_fraction
        ),
        "target_label_null_sign_agreement_fraction": describe(
            frame.target_label_null_sign_agreement_fraction
        ),
        "sign_agreement_above_target_label_null": describe(
            frame.sign_agreement_above_target_label_null
        ),
        "top_decile_precision_between_halves": describe(
            frame.top_decile_precision_between_halves
        ),
        "target_label_null_top_decile_precision": describe(
            frame.target_label_null_top_decile_precision
        ),
        "top_decile_precision_above_target_label_null": describe(
            frame.top_decile_precision_above_target_label_null
        ),
        "top_decile_jaccard_between_halves": describe(
            frame.top_decile_jaccard_between_halves
        ),
        "target_label_null_top_decile_jaccard": describe(
            frame.target_label_null_top_decile_jaccard
        ),
        "mean_half_to_full_top_decile_precision": describe(
            frame.mean_half_to_full_top_decile_precision
        ),
        "mean_half_to_full_map_spearman": describe(
            frame.mean_half_to_full_map_spearman
        ),
        "interpretation": (
            "The disjoint-half Spearman is the repeatability diagnostic at the "
            "observed half-panel size. No Spearman--Brown extrapolation is made. "
            "Sign and top-decile overlap are compared with complete target-label "
            "relabeling because row centring induces a shared compositional "
            "constraint. Half-to-full quantities are recovery diagnostics, not "
            "independent reliability estimates."
        ),
    }
    return summary, records


def bootstrap_indices(
    rows: int,
    unit: str,
    rng: np.random.Generator,
    cluster_labels: np.ndarray | None,
) -> np.ndarray:
    if unit == "ligand":
        return rng.integers(0, rows, size=rows)
    if unit == "butina_cluster":
        if cluster_labels is None or len(cluster_labels) != rows:
            raise ValueError("cluster bootstrap requires aligned cluster labels")
        labels = np.asarray(cluster_labels)
        unique = np.unique(labels)
        sampled = unique[rng.integers(0, len(unique), size=len(unique))]
        indices = np.concatenate([np.flatnonzero(labels == label) for label in sampled])
        if len(indices) < 3:
            raise RuntimeError("cluster bootstrap produced fewer than three rows")
        return indices
    raise ValueError(f"unknown bootstrap unit: {unit}")


def bootstrap_edge_stability(
    matrix: np.ndarray,
    targets: Sequence[str],
    *,
    panel: str,
    support: str,
    transform: str,
    unit: str,
    repeats: int,
    seed: int,
    cluster_labels: np.ndarray | None,
) -> tuple[list[dict[str, Any]], dict[str, Any], np.ndarray]:
    """Bootstrap perturbation stability, deliberately not called reliability."""
    values = np.asarray(matrix, dtype=np.float64)
    full = target_map(values, transform, "empirical")
    full_edges = upper(full)
    full_top = deterministic_top_indices(full_edges)
    top_count = len(full_top)
    tri = np.triu_indices(len(targets), k=1)
    sign_match = np.zeros(len(full_edges), dtype=np.int64)
    top_frequency = np.zeros(len(full_edges), dtype=np.int64)
    map_agreement = np.full(repeats, np.nan, dtype=np.float64)
    top_precision = np.full(repeats, np.nan, dtype=np.float64)
    top_jaccard = np.full(repeats, np.nan, dtype=np.float64)
    rng = np.random.default_rng(seed)
    valid_repeats = 0
    for repetition in range(repeats):
        rows = bootstrap_indices(len(values), unit, rng, cluster_labels)
        try:
            boot = target_map(values[rows], transform, "empirical")
        except ValueError:
            continue
        edges = upper(boot)
        selected = deterministic_top_indices(edges)
        sign_match += np.sign(edges) == np.sign(full_edges)
        top_frequency[selected] += 1
        map_agreement[repetition] = map_spearman(boot, full)
        top_precision[repetition], top_jaccard[repetition] = set_overlap(
            selected, full_top
        )
        valid_repeats += 1
    if valid_repeats == 0:
        raise ValueError("every edge-stability bootstrap replicate was degenerate")
    edge_rows = [
        {
            "panel": panel,
            "support": support,
            "transform": transform,
            "bootstrap_unit": unit,
            "target_a": targets[first],
            "target_b": targets[second],
            "full_map_correlation": float(full_edges[pair]),
            "full_map_sign": int(np.sign(full_edges[pair])),
            "full_map_top_decile": bool(pair in set(full_top.tolist())),
            "bootstrap_sign_matches_full_fraction": float(
                sign_match[pair] / valid_repeats
            ),
            "bootstrap_top_decile_inclusion_fraction": float(
                top_frequency[pair] / valid_repeats
            ),
            "repeats": int(repeats),
            "valid_repeats": int(valid_repeats),
            "seed": int(seed),
        }
        for pair, (first, second) in enumerate(zip(*tri, strict=True))
    ]
    edge_frame = pd.DataFrame.from_records(edge_rows)
    full_top_rows = edge_frame[edge_frame.full_map_top_decile]
    summary = {
        "panel": panel,
        "support": support,
        "ligands": int(len(values)),
        "transform": transform,
        "bootstrap_unit": unit,
        "repeats": int(repeats),
        "valid_repeats": int(valid_repeats),
        "invalid_repeats": int(repeats - valid_repeats),
        "seed": int(seed),
        "top_decile_edges": int(top_count),
        "bootstrap_map_to_full_spearman": describe(map_agreement),
        "bootstrap_top_decile_precision_against_full": describe(top_precision),
        "bootstrap_top_decile_jaccard_against_full": describe(top_jaccard),
        "edges_with_sign_match_at_least_0_90": int(
            (edge_frame.bootstrap_sign_matches_full_fraction >= 0.90).sum()
        ),
        "full_top_decile_edges_with_inclusion_at_least_0_80": int(
            (full_top_rows.bootstrap_top_decile_inclusion_fraction >= 0.80).sum()
        ),
        "interpretation": (
            "Bootstrap maps overlap the plugin map and therefore quantify "
            "perturbation stability, not independent-map reliability; the "
            "disjoint split-half result supplies the latter."
        ),
    }
    return edge_rows, summary, np.asarray(top_frequency / valid_repeats, dtype=float)


def paired_geometry_bootstrap(
    docking: np.ndarray,
    experimental: np.ndarray,
    broad_map: np.ndarray,
    *,
    panel: str,
    support: str,
    transform: str,
    unit: str,
    repeats: int,
    seed: int,
    cluster_labels: np.ndarray | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Paired n-out-of-n bootstrap of both same-support maps."""
    docking = np.asarray(docking, dtype=np.float64)
    experimental = np.asarray(experimental, dtype=np.float64)
    if docking.shape != experimental.shape:
        raise ValueError("same-support matrices must have identical shapes")
    point_docking = target_map(docking, transform)
    point_experimental = target_map(experimental, transform)
    point_same = map_spearman(point_docking, point_experimental)
    point_broad = map_spearman(broad_map, point_experimental)
    rng = np.random.default_rng(seed)
    records: list[dict[str, Any]] = []
    for repetition in range(repeats):
        rows = bootstrap_indices(len(docking), unit, rng, cluster_labels)
        try:
            docking_map = target_map(docking[rows], transform)
            experimental_map = target_map(experimental[rows], transform)
            same = map_spearman(docking_map, experimental_map)
            broad = map_spearman(broad_map, experimental_map)
            valid = True
            invalid_reason = ""
        except ValueError as error:
            same = np.nan
            broad = np.nan
            valid = False
            invalid_reason = str(error)
        records.append(
            {
                "panel": panel,
                "support": support,
                "transform": transform,
                "bootstrap_unit": unit,
                "repetition": repetition,
                "sampled_rows": int(len(rows)),
                "valid_replicate": valid,
                "invalid_reason": invalid_reason,
                "same_support_spearman": same,
                "broad_support_spearman": broad,
                "same_minus_broad": float(same - broad) if valid else np.nan,
            }
        )
    frame = pd.DataFrame.from_records(records)
    summary = {
        "panel": panel,
        "support": support,
        "ligands": int(len(docking)),
        "targets": int(docking.shape[1]),
        "transform": transform,
        "bootstrap_unit": unit,
        "method": (
            "paired n-out-of-n nonparametric bootstrap; the same sampled "
            "ligands/clusters re-estimate docking and experimental maps"
        ),
        "repeats": int(repeats),
        "valid_repeats": int(frame.valid_replicate.sum()),
        "invalid_repeats": int((~frame.valid_replicate).sum()),
        "seed": int(seed),
        "point_estimate": {
            "same_support_spearman": point_same,
            "broad_support_spearman": point_broad,
            "same_minus_broad": float(point_same - point_broad),
        },
        "same_support_spearman": describe(frame.same_support_spearman),
        "broad_support_spearman": describe(frame.broad_support_spearman),
        "same_minus_broad": describe(frame.same_minus_broad),
        "fraction_same_above_broad": float(
            np.mean(frame.loc[frame.valid_replicate, "same_minus_broad"] > 0.0)
        ),
        "interval_note": (
            "Percentile bootstrap sampling ranges are reported for this nonlinear "
            "rank statistic; they are not target-superpopulation intervals."
        ),
    }
    return summary, records


def paired_same_support_split(
    docking: np.ndarray,
    experimental: np.ndarray,
    *,
    panel: str,
    support: str,
    transform: str,
    split_unit: str,
    repeats: int,
    seed: int,
    cluster_labels: np.ndarray | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Joint half-map repeatability and shared-support optimism diagnostic."""
    docking = np.asarray(docking, dtype=np.float64)
    experimental = np.asarray(experimental, dtype=np.float64)
    if docking.shape != experimental.shape:
        raise ValueError("same-support matrices must align")
    point = map_spearman(
        target_map(docking, transform), target_map(experimental, transform)
    )
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for repetition in range(repeats):
        first, second = split_indices(
            len(docking), split_unit, rng, cluster_labels
        )
        try:
            docking_first = target_map(docking[first], transform)
            docking_second = target_map(docking[second], transform)
            experimental_first = target_map(experimental[first], transform)
            experimental_second = target_map(experimental[second], transform)
        except ValueError as error:
            rows.append(
                {
                    "panel": panel,
                    "support": support,
                    "transform": transform,
                    "split_unit": split_unit,
                    "repetition": repetition,
                    "first_rows": int(len(first)),
                    "second_rows": int(len(second)),
                    "valid_split": False,
                    "invalid_reason": str(error),
                    "docking_split_map_spearman": np.nan,
                    "experimental_split_map_spearman": np.nan,
                    "same_half_cross_modal_spearman": np.nan,
                    "cross_half_cross_modal_spearman": np.nan,
                    "same_minus_cross_half": np.nan,
                    "cross_half_target_label_null_spearman": np.nan,
                    "cross_half_spearman_above_target_label_null": np.nan,
                    "experimental_self_top_decile_precision": np.nan,
                    "same_half_cross_modal_top_decile_precision": np.nan,
                    "cross_half_cross_modal_top_decile_precision": np.nan,
                }
            )
            continue
        docking_reliability = map_spearman(docking_first, docking_second)
        experimental_reliability = map_spearman(
            experimental_first, experimental_second
        )
        same_half = float(
            (
                map_spearman(docking_first, experimental_first)
                + map_spearman(docking_second, experimental_second)
            )
            / 2.0
        )
        cross_half = float(
            (
                map_spearman(docking_first, experimental_second)
                + map_spearman(docking_second, experimental_first)
            )
            / 2.0
        )
        experimental_self_top, _ = set_overlap(
            deterministic_top_indices(upper(experimental_first)),
            deterministic_top_indices(upper(experimental_second)),
        )
        same_top_first, _ = set_overlap(
            deterministic_top_indices(upper(docking_first)),
            deterministic_top_indices(upper(experimental_first)),
        )
        same_top_second, _ = set_overlap(
            deterministic_top_indices(upper(docking_second)),
            deterministic_top_indices(upper(experimental_second)),
        )
        cross_top_first, _ = set_overlap(
            deterministic_top_indices(upper(docking_first)),
            deterministic_top_indices(upper(experimental_second)),
        )
        cross_top_second, _ = set_overlap(
            deterministic_top_indices(upper(docking_second)),
            deterministic_top_indices(upper(experimental_first)),
        )
        first_order = rng.permutation(docking.shape[1])
        second_order = rng.permutation(docking.shape[1])
        null_first = experimental_first[np.ix_(first_order, first_order)]
        null_second = experimental_second[np.ix_(second_order, second_order)]
        cross_null = float(
            (
                map_spearman(docking_first, null_second)
                + map_spearman(docking_second, null_first)
            )
            / 2.0
        )
        rows.append(
            {
                "panel": panel,
                "support": support,
                "transform": transform,
                "split_unit": split_unit,
                "repetition": repetition,
                "first_rows": int(len(first)),
                "second_rows": int(len(second)),
                "valid_split": True,
                "invalid_reason": "",
                "docking_split_map_spearman": docking_reliability,
                "experimental_split_map_spearman": experimental_reliability,
                "same_half_cross_modal_spearman": same_half,
                "cross_half_cross_modal_spearman": cross_half,
                "same_minus_cross_half": float(same_half - cross_half),
                "cross_half_target_label_null_spearman": cross_null,
                "cross_half_spearman_above_target_label_null": float(
                    cross_half - cross_null
                ),
                "experimental_self_top_decile_precision": experimental_self_top,
                "same_half_cross_modal_top_decile_precision": float(
                    (same_top_first + same_top_second) / 2.0
                ),
                "cross_half_cross_modal_top_decile_precision": float(
                    (cross_top_first + cross_top_second) / 2.0
                ),
            }
        )
    frame = pd.DataFrame.from_records(rows)
    experimental_self = float(np.nanmedian(frame.experimental_split_map_spearman))
    cross_modal = float(np.nanmedian(frame.cross_half_cross_modal_spearman))
    summary = {
        "panel": panel,
        "support": support,
        "ligands": int(len(docking)),
        "targets": int(docking.shape[1]),
        "transform": transform,
        "split_unit": split_unit,
        "repeats": int(repeats),
        "valid_repeats": int(frame.valid_split.sum()),
        "invalid_repeats": int((~frame.valid_split).sum()),
        "seed": int(seed),
        "plugin_same_support_spearman": point,
        "docking_split_map_spearman": describe(frame.docking_split_map_spearman),
        "experimental_split_map_spearman": describe(
            frame.experimental_split_map_spearman
        ),
        "same_half_cross_modal_spearman": describe(
            frame.same_half_cross_modal_spearman
        ),
        "cross_half_cross_modal_spearman": describe(
            frame.cross_half_cross_modal_spearman
        ),
        "same_minus_cross_half": describe(frame.same_minus_cross_half),
        "cross_half_target_label_null_spearman": describe(
            frame.cross_half_target_label_null_spearman
        ),
        "cross_half_spearman_above_target_label_null": describe(
            frame.cross_half_spearman_above_target_label_null
        ),
        "experimental_self_top_decile_precision": describe(
            frame.experimental_self_top_decile_precision
        ),
        "same_half_cross_modal_top_decile_precision": describe(
            frame.same_half_cross_modal_top_decile_precision
        ),
        "cross_half_cross_modal_top_decile_precision": describe(
            frame.cross_half_cross_modal_top_decile_precision
        ),
        "median_cross_half_spearman_divided_by_experimental_self_repeatability": (
            float(cross_modal / experimental_self)
            if abs(experimental_self) > 1e-12
            else None
        ),
        "interpretation": (
            "The experimental half--half map agreement is a direct repeatability "
            "benchmark at half the observed ligand count; it is not extrapolated. "
            "The cross-modal ratio is descriptive and not bounded by one. The "
            "same-minus-cross-half contrast estimates optimism from sharing the "
            "same finite ligand draw, and the target-label null handles the "
            "dependence among map edges."
        ),
    }
    return summary, rows


def panel_clusters(smiles: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    clean = pd.Series(smiles).reset_index(drop=True)
    return (
        dense_davis._murcko_labels(clean),  # noqa: SLF001
        dense_davis._butina_labels(clean),  # noqa: SLF001
    )


def cluster_support_summary(labels: np.ndarray) -> dict[str, Any]:
    """Describe the chemical resampling units used by a panel analysis."""
    _, counts = np.unique(np.asarray(labels), return_counts=True)
    return {
        "clusters": int(len(counts)),
        "singleton_clusters": int(np.sum(counts == 1)),
        "singleton_ligand_fraction": float(np.sum(counts[counts == 1]) / counts.sum()),
        "median_cluster_size": float(np.median(counts)),
        "maximum_cluster_size": int(counts.max()),
    }


def molecular_descriptor_matrix(smiles: pd.Series) -> np.ndarray:
    records: list[tuple[float, float]] = []
    for value in pd.Series(smiles).astype(str):
        molecule = dense_davis.Chem.MolFromSmiles(value)
        if molecule is None:
            raise ValueError("an experimental-panel SMILES is invalid")
        records.append(
            (
                float(dense_davis.Descriptors.MolWt(molecule)),
                float(dense_davis.Descriptors.MolLogP(molecule)),
            )
        )
    return np.asarray(records, dtype=float)


def matched_selection_audit(
    panel_name: str,
    panel: pd.DataFrame,
    matched_keys: Sequence[str],
) -> dict[str, Any]:
    matched_mask = panel.standard_inchikey.astype(str).isin(set(matched_keys)).to_numpy()
    if not matched_mask.any() or matched_mask.all():
        raise ValueError("matched-selection audit requires matched and unmatched rows")
    descriptors = molecular_descriptor_matrix(panel.smiles)
    names = ("molecular_weight", "clogp")
    rows: dict[str, Any] = {}
    for index, name in enumerate(names):
        selected = descriptors[matched_mask, index]
        remainder = descriptors[~matched_mask, index]
        pooled = np.sqrt(
            (
                (len(selected) - 1) * selected.var(ddof=1)
                + (len(remainder) - 1) * remainder.var(ddof=1)
            )
            / (len(selected) + len(remainder) - 2)
        )
        rows[name] = {
            "matched_mean": float(selected.mean()),
            "unmatched_mean": float(remainder.mean()),
            "matched_minus_unmatched_standardized_mean_difference": float(
                (selected.mean() - remainder.mean()) / pooled
            ),
        }
    output: dict[str, Any] = {
        "matched_rows": int(matched_mask.sum()),
        "matched_unique_full_standard_inchikeys": int(
            panel.loc[matched_mask, "standard_inchikey"].nunique()
        ),
        "unmatched_rows": int((~matched_mask).sum()),
        "match_fraction": float(matched_mask.mean()),
        "descriptors": rows,
    }
    if panel_name == "DAVIS":
        values = panel[list(TARGETS)].to_numpy(dtype=float)
        floor = np.isclose(values, 5.0, rtol=0.0, atol=1e-12)
        output["pkd5_floor_cell_fraction"] = {
            "matched": float(floor[matched_mask].mean()),
            "unmatched": float(floor[~matched_mask].mean()),
        }
    else:
        output["pkd5_floor_cell_fraction"] = None
    return output


def load_dockstring_support(
    dockstring_path: Path = DEFAULT_DOCKSTRING,
    identity_contract_path: Path = DEFAULT_IDENTITY_CONTRACT,
) -> pd.DataFrame:
    """Public complete DOCKSTRING support aligned to its frozen identity contract."""
    source = pd.read_csv(dockstring_path, sep="\t")
    score_columns = [
        column for column in source.columns if column not in {"inchikey", "smiles"}
    ]
    complete = ~source[score_columns].isna().any(axis=1)
    frame = source.loc[complete, ["inchikey", "smiles", *TARGETS]].copy()
    frame.insert(0, "source_row_index", frame.index.to_numpy(dtype=np.int64))
    frame = frame.reset_index(drop=True)
    if len(frame) != 260_060 or frame[list(TARGETS)].isna().any().any():
        raise ValueError("unexpected complete DOCKSTRING support")
    contract = pd.read_csv(
        identity_contract_path,
        usecols=[
            "source_row_index",
            "complete_support_row_index",
            "raw_inchikey",
            "raw_connectivity",
        ],
    )
    if len(contract) != len(frame):
        raise ValueError("DOCKSTRING identity contract has unexpected support")
    if not np.array_equal(
        contract.source_row_index.to_numpy(dtype=np.int64),
        frame.source_row_index.to_numpy(dtype=np.int64),
    ):
        raise ValueError("identity contract does not align to DOCKSTRING rows")
    if not np.array_equal(
        contract.complete_support_row_index.to_numpy(dtype=np.int64),
        np.arange(len(contract), dtype=np.int64),
    ):
        raise ValueError("identity contract support indices are not contiguous")
    frame["standard_inchikey"] = contract.raw_inchikey.to_numpy(dtype=object)
    frame["connectivity_block"] = contract.raw_connectivity.to_numpy(dtype=object)
    return frame


def load_davis_panel(davis_path: Path = DEFAULT_DAVIS) -> pd.DataFrame:
    """Public dense 72 x 21 DAVIS pKd block with recomputed molecular identity."""
    raw = pd.read_csv(
        davis_path,
        sep="\t",
        usecols=["drug_name", "protein", "compound_iso_smiles", "y"],
    )
    molecules = raw[["drug_name", "compound_iso_smiles"]].drop_duplicates()
    if molecules.drug_name.duplicated().any() or len(molecules) != 72:
        raise ValueError("unexpected DAVIS molecule table")
    selected = raw[raw.protein.isin(dense_davis.TARGET_MAP.values())]
    matrix = selected.pivot(index="drug_name", columns="protein", values="y").reindex(
        index=molecules.drug_name,
        columns=[dense_davis.TARGET_MAP[target] for target in TARGETS],
    )
    matrix.columns = list(TARGETS)
    if matrix.shape != (72, 21) or matrix.isna().any().any():
        raise ValueError("expected a dense DAVIS 72 x 21 block")
    identity = dense_davis._identity_table(  # noqa: SLF001
        molecules, "compound_iso_smiles", scan="full"
    )
    panel = pd.DataFrame(matrix.to_numpy(dtype=float), columns=list(TARGETS))
    panel["standard_inchikey"] = identity.standard_inchikey.to_numpy(dtype=object)
    panel["connectivity_block"] = identity.connectivity_block.to_numpy(dtype=object)
    panel["smiles"] = identity.compound_iso_smiles.to_numpy(dtype=object)
    return panel


def load_pkis2_panel(pkis2_path: Path = DEFAULT_PKIS2) -> pd.DataFrame:
    """Public dense 645 x 21 PKIS2 percentage-displacement block."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Unknown extension is not supported and will be removed",
            category=UserWarning,
            module=r"openpyxl\.worksheet\._reader",
        )
        raw = pd.read_excel(pkis2_path, sheet_name=dense_pkis2.PKIS2_SHEET)
    required_metadata = ["Regno", "Compound", "Chemotype", "Smiles"]
    if len(raw.columns) != 413:
        raise ValueError("unexpected PKIS2 workbook width")
    raw = raw.dropna(subset=required_metadata).copy()
    if len(raw) != 645 or raw[required_metadata].isna().any().any():
        raise ValueError("unexpected populated PKIS2 support")
    identity = dense_davis._identity_table(raw, "Smiles", scan="full")  # noqa: SLF001
    experimental = identity[list(dense_pkis2.TARGET_MAP.values())].apply(
        pd.to_numeric, errors="coerce"
    )
    experimental.columns = list(TARGETS)
    if experimental.isna().any().any():
        raise ValueError("selected PKIS2 block is not dense/numeric")
    if ((experimental < 0.0) | (experimental > 100.0)).any().any():
        raise ValueError("a PKIS2 percentage lies outside [0, 100]")
    panel = experimental.reset_index(drop=True).astype(float)
    panel["standard_inchikey"] = identity.standard_inchikey.to_numpy(dtype=object)
    panel["connectivity_block"] = identity.connectivity_block.to_numpy(dtype=object)
    panel["smiles"] = identity.Smiles.to_numpy(dtype=object)
    return panel


def match_panel(
    panel: pd.DataFrame,
    dockstring: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.DataFrame]:
    """Exact full-Standard-InChIKey intersection with median duplicate collapse."""
    common = sorted(
        set(panel.standard_inchikey.dropna())
        & set(dockstring.standard_inchikey.dropna())
    )
    if not common:
        raise ValueError("no exact panel x DOCKSTRING molecular overlap")
    panel_rows = panel[panel.standard_inchikey.isin(common)]
    docking_rows = dockstring[dockstring.standard_inchikey.isin(common)]
    experimental = panel_rows.groupby("standard_inchikey", sort=True)[list(TARGETS)].median()
    docking = (
        docking_rows.groupby("standard_inchikey", sort=True)[list(TARGETS)]
        .median()
        .reindex(experimental.index)
    )
    smiles = (
        panel_rows.groupby("standard_inchikey", sort=True)["smiles"]
        .first()
        .reindex(experimental.index)
    )
    provenance = pd.DataFrame(
        {
            "standard_inchikey": experimental.index.to_numpy(dtype=object),
            "panel_rows": panel_rows.groupby("standard_inchikey", sort=True)
            .size()
            .reindex(experimental.index)
            .to_numpy(dtype=int),
            "dockstring_rows": docking_rows.groupby("standard_inchikey", sort=True)
            .size()
            .reindex(experimental.index)
            .to_numpy(dtype=int),
        }
    )
    if docking.isna().any().any() or experimental.isna().any().any():
        raise ValueError("matched public panel is not dense")
    return docking, experimental, smiles, provenance


def load_public_panels() -> OrderedDict[str, pd.DataFrame]:
    return OrderedDict(
        [
            ("DAVIS", load_davis_panel()),
            ("PKIS2", load_pkis2_panel()),
        ]
    )


def broad_reference(
    dockstring: pd.DataFrame, panels: dict[str, pd.DataFrame]
) -> tuple[np.ndarray, dict[str, Any]]:
    """De-leaked public reference, excluding connectivity from both panels."""
    excluded = set()
    for panel in panels.values():
        excluded.update(panel.connectivity_block.dropna().astype(str))
    keep = ~dockstring.connectivity_block.astype(str).isin(excluded)
    matrix = np.minimum(
        dockstring.loc[keep, list(TARGETS)].to_numpy(dtype=np.float64), 0.0
    )
    return matrix, {
        "definition": (
            "complete DOCKSTRING support after excluding every row whose raw "
            "InChIKey connectivity block occurs in DAVIS or PKIS2"
        ),
        "rows": int(len(matrix)),
        "excluded_rows": int((~keep).sum()),
        "positive_scores_clipped_to_zero": True,
    }


def cross_panel_map_record(
    first: np.ndarray,
    second: np.ndarray,
    *,
    transform: str,
    permutations: int,
    seed: int,
) -> dict[str, Any]:
    """Independent-panel concordance with a complete target-label QAP null."""
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    if first.shape != second.shape or first.shape != (len(TARGETS), len(TARGETS)):
        raise ValueError("cross-panel maps do not match the fixed target support")
    first_edges = upper(first)
    second_edges = upper(second)
    observed = map_spearman(first, second)
    first_top = deterministic_top_indices(first_edges)
    second_top = deterministic_top_indices(second_edges)
    observed_precision, observed_jaccard = set_overlap(first_top, second_top)
    observed_sign = float(np.mean(np.sign(first_edges) == np.sign(second_edges)))
    null_rho = np.empty(permutations, dtype=float)
    null_precision = np.empty(permutations, dtype=float)
    null_jaccard = np.empty(permutations, dtype=float)
    null_sign = np.empty(permutations, dtype=float)
    rng = np.random.default_rng(seed)
    for repetition in range(permutations):
        order = rng.permutation(len(TARGETS))
        permuted = second[np.ix_(order, order)]
        permuted_edges = upper(permuted)
        permuted_top = deterministic_top_indices(permuted_edges)
        null_rho[repetition] = map_spearman(first, permuted)
        null_precision[repetition], null_jaccard[repetition] = set_overlap(
            first_top, permuted_top
        )
        null_sign[repetition] = np.mean(
            np.sign(first_edges) == np.sign(permuted_edges)
        )
    return {
        "panel_a": "DAVIS",
        "panel_b": "PKIS2",
        "support_a": "67 informative released profiles",
        "support_b": "644 informative released profiles",
        "targets": len(TARGETS),
        "target_pairs": TARGET_PAIRS,
        "transform": transform,
        "edge_spearman": observed,
        "target_label_qap_p_two_sided": float(
            (1 + np.sum(np.abs(null_rho) >= abs(observed))) / (permutations + 1)
        ),
        "target_label_qap_null_spearman_median": float(np.median(null_rho)),
        "top_decile_precision": observed_precision,
        "top_decile_precision_null_median": float(np.median(null_precision)),
        "top_decile_precision_above_null": float(
            observed_precision - np.median(null_precision)
        ),
        "top_decile_jaccard": observed_jaccard,
        "top_decile_jaccard_null_median": float(np.median(null_jaccard)),
        "edge_sign_agreement_fraction": observed_sign,
        "edge_sign_agreement_null_median": float(np.median(null_sign)),
        "edge_sign_agreement_above_null": float(observed_sign - np.median(null_sign)),
        "permutations": int(permutations),
        "seed": int(seed),
        "interpretation": (
            "Independent-panel agreement is an external reproducibility benchmark, "
            "not a formal noise ceiling, because assay technology and chemical "
            "support differ between panels."
        ),
    }


def analyse(
    *,
    split_repeats: int = DEFAULT_SPLIT_REPEATS,
    bootstraps: int = DEFAULT_BOOTSTRAPS,
    seed: int = DEFAULT_SEED,
) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
    if split_repeats < 10 or bootstraps < 10:
        raise ValueError("production analysis requires at least ten repeats")
    dockstring = load_dockstring_support()
    panels = load_public_panels()
    broad_matrix, broad_metadata = broad_reference(dockstring, panels)
    broad_maps = {
        transform: target_map(broad_matrix, transform)
        for transform in ("raw", PRIMARY_TRANSFORM)
    }

    map_reliability_rows: list[dict[str, Any]] = []
    edge_rows: list[dict[str, Any]] = []
    same_geometry_rows: list[dict[str, Any]] = []
    same_bootstrap_rows: list[dict[str, Any]] = []
    same_split_rows: list[dict[str, Any]] = []
    cross_panel_rows: list[dict[str, Any]] = []
    panel_summaries: OrderedDict[str, dict[str, Any]] = OrderedDict()

    for panel_index, (panel_name, panel) in enumerate(panels.items()):
        panel_seed = seed + panel_index * PANEL_SEED_STRIDE
        full_values = panel[list(TARGETS)].to_numpy(dtype=np.float64)
        full_smiles = panel.smiles.reset_index(drop=True)
        full_informative = informative_rows(full_values)
        full_murcko, full_butina = panel_clusters(full_smiles)

        docking_frame, experimental_frame, matched_smiles, provenance = match_panel(
            panel, dockstring
        )
        docking_values = np.minimum(
            docking_frame.to_numpy(dtype=np.float64), 0.0
        )
        experimental_values = experimental_frame.to_numpy(dtype=np.float64)
        match_informative = informative_rows(experimental_values)
        matched_murcko, matched_butina = panel_clusters(matched_smiles)

        support_specs = (
            (
                "released_full_panel",
                full_values,
                full_butina,
                "map exactly as estimated on every released panel row",
            ),
            (
                "informative_profiles",
                full_values[full_informative],
                full_butina[full_informative],
                "constant assay profiles excluded",
            ),
        )
        reliability_summary: list[dict[str, Any]] = []
        edge_summary: list[dict[str, Any]] = []
        reliability_counter = 0
        for support_name, support_values, support_clusters, support_note in support_specs:
            for transform in (PRIMARY_TRANSFORM, "raw"):
                for split_unit in ("ligand", "butina_cluster"):
                    item, records = split_half_diagnostics(
                        support_values,
                        panel=panel_name,
                        support=support_name,
                        transform=transform,
                        estimator="empirical",
                        split_unit=split_unit,
                        repeats=split_repeats,
                        seed=panel_seed + 10_000 + reliability_counter,
                        cluster_labels=support_clusters,
                    )
                    support_map = target_map(support_values, transform)
                    observed = map_spearman(broad_maps[transform], support_map)
                    item.update(
                        {
                            "support_note": support_note,
                            "broad_docking_vs_experimental_spearman": observed,
                            "comparison_boundary": (
                                "This cross-support value is reported beside, not "
                                "normalized by, experimental split-half repeatability. "
                                "The two quantities have different chemical-support "
                                "contracts."
                            ),
                        }
                    )
                    reliability_summary.append(item)
                    for record in records:
                        map_reliability_rows.append(
                            {
                                **record,
                                "broad_docking_vs_experimental_spearman": observed,
                            }
                        )
                    reliability_counter += 1

            for unit_index, unit in enumerate(("ligand", "butina_cluster")):
                rows, item, _ = bootstrap_edge_stability(
                    support_values,
                    TARGETS,
                    panel=panel_name,
                    support=support_name,
                    transform=PRIMARY_TRANSFORM,
                    unit=unit,
                    repeats=bootstraps,
                    seed=panel_seed + 100_000 + 10 * reliability_counter + unit_index,
                    cluster_labels=support_clusters,
                )
                edge_rows.extend(rows)
                edge_summary.append(item)

        primary_docking = docking_values[match_informative]
        primary_experimental = experimental_values[match_informative]
        primary_clusters = matched_butina[match_informative]
        matched_specs = (
            (
                "informative_exact_matches_primary",
                primary_docking,
                primary_experimental,
                primary_clusters,
            ),
            (
                "all_exact_matches_sensitivity",
                docking_values,
                experimental_values,
                matched_butina,
            ),
        )
        same_summary: list[dict[str, Any]] = []
        same_bootstrap_summary: list[dict[str, Any]] = []
        same_split_summary: list[dict[str, Any]] = []
        for support_index, (
            support_name,
            support_docking,
            support_experimental,
            support_clusters,
        ) in enumerate(matched_specs):
            for transform_index, transform in enumerate((PRIMARY_TRANSFORM, "raw")):
                empirical_docking = target_map(support_docking, transform, "empirical")
                empirical_experimental = target_map(
                    support_experimental, transform, "empirical"
                )
                oas_docking = target_map(support_docking, transform, "oas")
                oas_experimental = target_map(support_experimental, transform, "oas")
                empirical_same = map_spearman(
                    empirical_docking, empirical_experimental
                )
                broad_same = map_spearman(
                    broad_maps[transform], empirical_experimental
                )
                oas_same = map_spearman(oas_docking, oas_experimental)
                point = {
                    "panel": panel_name,
                    "support": support_name,
                    "ligands": int(len(support_docking)),
                    "targets": len(TARGETS),
                    "target_pairs": TARGET_PAIRS,
                    "transform": transform,
                    "same_support_empirical_spearman": empirical_same,
                    "broad_support_empirical_spearman": broad_same,
                    "same_minus_broad_empirical": float(empirical_same - broad_same),
                    "same_support_oas_spearman": oas_same,
                    "oas_minus_empirical": float(oas_same - empirical_same),
                    "molecular_identity": (
                        "exact full Standard InChIKey; duplicates collapsed by "
                        "the cell-wise median"
                    ),
                    "interpretation": (
                        "Same chemical support removes a support mismatch but does "
                        "not certify biological validity; OAS is a finite-sample "
                        "covariance sensitivity."
                    ),
                }
                same_geometry_rows.append(point)
                same_summary.append(point)
                for unit_index, unit in enumerate(("ligand", "butina_cluster")):
                    item, records = paired_geometry_bootstrap(
                        support_docking,
                        support_experimental,
                        broad_maps[transform],
                        panel=panel_name,
                        support=support_name,
                        transform=transform,
                        unit=unit,
                        repeats=bootstraps,
                        seed=(
                            panel_seed
                            + 200_000
                            + 10_000 * support_index
                            + 100 * transform_index
                            + unit_index
                        ),
                        cluster_labels=support_clusters,
                    )
                    same_bootstrap_summary.append(item)
                    same_bootstrap_rows.extend(records)

            # Direct half-map repeatability is reported for the primary estimand.
            for split_index, split_unit in enumerate(("ligand", "butina_cluster")):
                item, records = paired_same_support_split(
                    support_docking,
                    support_experimental,
                    panel=panel_name,
                    support=support_name,
                    transform=PRIMARY_TRANSFORM,
                    split_unit=split_unit,
                    repeats=split_repeats,
                    seed=(
                        panel_seed
                        + 300_000
                        + 10_000 * support_index
                        + split_index
                    ),
                    cluster_labels=support_clusters,
                )
                same_split_summary.append(item)
                same_split_rows.extend(records)

        panel_summaries[panel_name] = {
            "released_panel_ligands": int(len(full_values)),
            "released_panel_constant_profiles": int((~full_informative).sum()),
            "released_panel_informative_profiles": int(full_informative.sum()),
            "exact_matches": int(len(docking_values)),
            "exact_matches_constant_experimental_profiles": int(
                (~match_informative).sum()
            ),
            "primary_same_support_informative_ligands": int(
                match_informative.sum()
            ),
            "matched_provenance_rows": int(len(provenance)),
            "matched_selection_audit": matched_selection_audit(
                panel_name, panel, experimental_frame.index.astype(str).tolist()
            ),
            "chemical_cluster_support": {
                "method": (
                    "RDKit Morgan radius-2, 2048-bit fingerprints with Butina "
                    "clustering at Tanimoto similarity 0.65 (distance 0.35), as "
                    "redistributed public-panel loader"
                ),
                "released_full_panel": cluster_support_summary(full_butina),
                "released_informative_profiles": cluster_support_summary(
                    full_butina[full_informative]
                ),
                "exact_matches_after_duplicate_collapse": cluster_support_summary(
                    matched_butina
                ),
                "primary_same_support_informative": cluster_support_summary(
                    primary_clusters
                ),
            },
            "targets": len(TARGETS),
            "target_pairs": TARGET_PAIRS,
            "map_reliability": reliability_summary,
            "edge_stability": edge_summary,
            "same_support_geometry": same_summary,
            "same_support_bootstrap": same_bootstrap_summary,
            "same_support_reliability": same_split_summary,
        }

    # Independent experimental cross-panel agreement is a direct external
    # reproducibility benchmark.  It is not called a ceiling because the panels
    # differ in assay technology and chemical support.
    informative_panel_values = {
        name: panel[list(TARGETS)].to_numpy(dtype=float)[
            informative_rows(panel[list(TARGETS)].to_numpy(dtype=float))
        ]
        for name, panel in panels.items()
    }
    for transform_index, transform in enumerate((PRIMARY_TRANSFORM, "raw")):
        first = target_map(informative_panel_values["DAVIS"], transform)
        second = target_map(informative_panel_values["PKIS2"], transform)
        cross_panel_rows.append(
            cross_panel_map_record(
                first,
                second,
                transform=transform,
                permutations=9_999,
                seed=seed + 5_000_000 + transform_index,
            )
        )
    summary: dict[str, Any] = {
        "analysis": (
            "experimental target-map reliability and exact same-support "
            "docking--experiment geometry"
        ),
        "status": "post_hoc_reviewer_requested_boundary_analysis",
        "primary_estimand": (
            "Spearman concordance of the 210 target-pair correlations after "
            "two-way centring, estimated on the same informative exact-match "
            "ligands and the same 21 targets"
        ),
        "central_data_boundary": (
            "Only repository-redistributed DAVIS, PKIS2 and DOCKSTRING source "
            "tables are used for the central results."
        ),
        "broad_reference": broad_metadata,
        "panels": panel_summaries,
        "cross_panel_experimental_geometry": cross_panel_rows,
        "methods_boundary": {
            "split_half": (
                "disjoint halves are the repeatability estimate; chemical-cluster "
                "splits never place one Butina cluster in both halves"
            ),
            "full_support_reliability": (
                "not extrapolated from halves: Spearman--Brown is inapplicable to "
                "this nonlinear map statistic"
            ),
            "bootstrap": (
                "paired perturbation/sampling ranges conditional on each fixed "
                "target panel; not target-superpopulation intervals"
            ),
            "oas": (
                "Oracle Approximating Shrinkage covariance converted to a "
                "correlation matrix; finite-sample estimator sensitivity"
            ),
            "noise_ceiling": (
                "no classical attenuation ceiling is reported because docking "
                "and experiment cannot be assumed to measure one latent construct"
            ),
        },
        "parameters": {
            "targets": list(TARGETS),
            "target_pairs": TARGET_PAIRS,
            "top_fraction": TOP_FRACTION,
            "split_repeats": int(split_repeats),
            "bootstraps": int(bootstraps),
            "seed": int(seed),
            "butina_definition": (
                "RDKit Morgan radius 2, 2048 bits; Butina Tanimoto similarity "
                ">=0.65 (distance <=0.35)"
            ),
        },
        "primary_input_sha256": {
            "DAVIS": sha256_file(DEFAULT_DAVIS),
            "PKIS2": sha256_file(DEFAULT_PKIS2),
            "DOCKSTRING": sha256_file(DEFAULT_DOCKSTRING),
            "DOCKSTRING_identity_contract": sha256_file(DEFAULT_IDENTITY_CONTRACT),
        },
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit_learn_OAS": "installed pinned environment",
        },
    }
    tables = {
        "cross_panel_geometry.csv": pd.DataFrame.from_records(cross_panel_rows),
        "edge_stability.csv": pd.DataFrame.from_records(edge_rows),
        "map_reliability.csv": pd.DataFrame.from_records(map_reliability_rows),
        "same_support_bootstrap.csv": pd.DataFrame.from_records(
            same_bootstrap_rows
        ),
        "same_support_geometry.csv": pd.DataFrame.from_records(same_geometry_rows),
        "same_support_split_half.csv": pd.DataFrame.from_records(same_split_rows),
    }
    return json_ready(summary), tables


def write_readme(output_dir: Path, summary: dict[str, Any]) -> None:
    davis = summary["panels"]["DAVIS"]
    pkis2 = summary["panels"]["PKIS2"]
    lines = [
        "# Experimental map reliability and same-support geometry",
        "",
        "This reviewer-requested analysis uses only fully redistributed DAVIS, "
        "PKIS2 and DOCKSTRING inputs for its central results.",
        "",
        "## Fixed support",
        "",
        f"- DAVIS: {davis['exact_matches']} exact matches, of which "
        f"{davis['primary_same_support_informative_ligands']} are informative; "
        "the excluded matched rows are constant at the pKd=5 floor.",
        f"- PKIS2: {pkis2['exact_matches']} exact informative matches.",
        f"- Both panels: {summary['parameters']['target_pairs']} edges over "
        f"{len(summary['parameters']['targets'])} shared targets.",
        "",
        "## Interpretation guardrails",
        "",
        "Disjoint split halves estimate map repeatability directly at the "
        "half-panel ligand count; no Spearman--Brown extrapolation or attenuation "
        "correction is made. Sign and top-decile overlap are compared with a "
        "target-label null. Bootstrap-to-full agreement is called perturbation "
        "stability, not reliability. OAS is a shrinkage sensitivity. Same-support "
        "concordance removes chemical-support mismatch but is not a certificate "
        "of biological validity.",
    ]
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_outputs(
    output_dir: Path, summary: dict[str, Any], tables: dict[str, pd.DataFrame]
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in tables.items():
        frame.to_csv(output_dir / name, index=False)
    write_readme(output_dir, summary)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    checksums = {
        name: {
            "sha256": sha256_file(output_dir / name),
            "bytes": int((output_dir / name).stat().st_size),
        }
        for name in OUTPUT_FILES
    }
    (output_dir / "output_checksums.json").write_text(
        json.dumps(checksums, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--split-repeats", type=int, default=DEFAULT_SPLIT_REPEATS)
    parser.add_argument("--bootstraps", type=int, default=DEFAULT_BOOTSTRAPS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    arguments = parser.parse_args()
    summary, tables = analyse(
        split_repeats=arguments.split_repeats,
        bootstraps=arguments.bootstraps,
        seed=arguments.seed,
    )
    write_outputs(arguments.output, summary, tables)
    for panel in ("DAVIS", "PKIS2"):
        primary = [
            row
            for row in summary["panels"][panel]["same_support_geometry"]
            if row["support"] == "informative_exact_matches_primary"
            and row["transform"] == PRIMARY_TRANSFORM
        ][0]
        print(
            panel,
            primary["ligands"],
            primary["same_support_empirical_spearman"],
            primary["broad_support_empirical_spearman"],
        )


if __name__ == "__main__":
    main()
