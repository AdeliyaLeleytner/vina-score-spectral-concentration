#!/usr/bin/env python3
"""External validation and interpretation of residual target geometry.

This analysis is deliberately separate from the manuscript evidence builder.  It asks
whether the target--target correlation geometry exposed by two-way centering of the
DOCKSTRING score matrix agrees with two dense experimental kinase panels (DAVIS and
PKIS2).  It also evaluates, explicitly as exploratory, a score-blind correction for the
leading target-specific molecular-weight slope.

Primary geometry supports
-------------------------
* DOCKSTRING: all 260,060 rows complete over the released 58-target panel, restricted
  to the 21 kinase targets common to DAVIS and PKIS2.  Positive scores are clipped to
  zero.  Every row whose recomputed Standard-InChI connectivity block occurs in either
  complete experimental ligand panel is excluded.
* DAVIS: the complete 72 x 21 pKd block.  The released 10,000 nM cap remains pKd=5;
  no missing-value imputation is performed.
* PKIS2: the complete 645 x 21 single-concentration percentage-inhibition block.

Molecule-matched 59 x 21 DAVIS and 154 x 21 PKIS2 surfaces are sensitivity analyses,
not the primary geometry estimand.  Geometry concordance is a target-level statistic:
Spearman correlation between the 210 strict-upper-triangle entries of two target
correlation matrices.  QAP inference jointly permutes target labels (rows and columns).

The molecular-weight correction is exploratory and is fit only on the excluded-reference
DOCKSTRING surface.  No experimental endpoint enters its fit.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import warnings
import zipfile
from collections import OrderedDict
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats

try:  # Direct script execution and package-style import are both supported.
    from . import build_evidence as evidence
    from . import dense_davis_benchmark as davis
    from . import dense_pkis2_benchmark as pkis2
except ImportError:  # pragma: no cover - exercised by direct CLI execution.
    import build_evidence as evidence  # type: ignore
    import dense_davis_benchmark as davis  # type: ignore
    import dense_pkis2_benchmark as pkis2  # type: ignore


PACKAGE = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PACKAGE / "results" / "residual_target_geometry"
DEFAULT_SEED = 20260802
PKIS1_DOI = "10.1038/nbt.3374"
PKIS1_URL = (
    "https://media.springernature.com/original/springer-static/esm/"
    "art%3A10.1038%2Fnbt.3374/MediaObjects/41587_2016_BFnbt3374_MOESM5_ESM.zip"
)
PKIS1_SHA256 = "1ffbe7fd0b4fc1ef72f2434a2b247d15c0f2b4d3c34668f112f4b4c9e4ead7dc"
PKIS1_SMILES_MEMBER = (
    "Supplementary Files/1. PKIS ID codes SMILES and References.xlsx"
)
PKIS1_HEATMAP_MEMBER = "Supplementary Files/3. PKIS Nanosyn Assay Heatmaps.xlsx"

TARGETS = tuple(davis.TARGET_MAP)
if TARGETS != tuple(pkis2.TARGET_MAP):  # Fail closed if either benchmark drifts.
    raise RuntimeError("DAVIS and PKIS2 no longer expose the same ordered 21 targets")

TRANSFORMS: OrderedDict[str, str] = OrderedDict(
    [
        ("raw", "raw target correlation without row centering"),
        (
            "center_then_correlation",
            "two-way center raw values, then compute target correlations",
        ),
        (
            "z_before_center",
            "column z-score raw values, two-way center, then compute correlations",
        ),
        (
            "rank_normal_before_center",
            "column inverse-normal ranks, two-way center, then compute correlations",
        ),
    ]
)

PKIS1_TARGET_MAP: OrderedDict[str, str] = OrderedDict(
    [
        ("ABL1", "ABL1"),
        ("AKT1", "AKT1"),
        ("AKT2", "AKT2"),
        ("CDK2", "CDK2/cyclinA"),
        ("CSF1R", "FMS"),
        ("EGFR", "EGFR"),
        ("FGFR1", "FGFR1"),
        ("IGF1R", "IGF1R"),
        ("JAK2", "JAK2"),
        ("KDR", "KDR"),
        ("KIT", "KIT"),
        ("LCK", "LCK"),
        ("MAP2K1", "MEK1"),
        ("MAPK1", "MAPK1"),
        ("MAPK14", "P38α"),
        ("MAPKAPK2", "MAPKAPK2"),
        ("MET", "MET"),
        ("PLK1", "PLK1"),
        ("ROCK1", "ROCK1"),
        ("SRC", "SRC"),
    ]
)


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def two_way_center(matrix: np.ndarray) -> np.ndarray:
    """Remove least-squares ligand and target main effects from a dense matrix."""
    x = np.asarray(matrix, dtype=np.float64)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError("two-way centering requires a finite two-dimensional matrix")
    return (
        x
        - x.mean(axis=0, keepdims=True)
        - x.mean(axis=1, keepdims=True)
        + x.mean()
    )


def column_zscore(matrix: np.ndarray) -> np.ndarray:
    x = np.asarray(matrix, dtype=np.float64)
    scale = x.std(axis=0, ddof=1)
    if np.any(scale <= 0) or not np.isfinite(scale).all():
        raise ValueError("a target column has zero or invalid standard deviation")
    return (x - x.mean(axis=0, keepdims=True)) / scale


def column_rank_normal_scores(matrix: np.ndarray) -> np.ndarray:
    """Inverse-normal scores using average ranks independently within each target."""
    x = np.asarray(matrix, dtype=np.float64)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError("rank normalization requires a finite 2-D matrix")
    n = len(x)
    if n < 3:
        raise ValueError("rank normalization requires at least three ligands")
    out = np.empty_like(x)
    for column in range(x.shape[1]):
        ranks = stats.rankdata(x[:, column], method="average")
        out[:, column] = stats.norm.ppf((ranks - 0.5) / n)
    return out


def transformed_surface(matrix: np.ndarray, transform: str) -> np.ndarray:
    x = np.asarray(matrix, dtype=np.float64)
    if transform == "raw":
        return x
    if transform == "center_then_correlation":
        return two_way_center(x)
    if transform == "z_before_center":
        return two_way_center(column_zscore(x))
    if transform == "rank_normal_before_center":
        return two_way_center(column_rank_normal_scores(x))
    raise ValueError(f"unknown transform: {transform}")


def target_correlation(matrix: np.ndarray) -> np.ndarray:
    x = np.asarray(matrix, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] < 3 or x.shape[1] < 2:
        raise ValueError("target correlation requires at least 3 ligands and 2 targets")
    if not np.isfinite(x).all():
        raise ValueError("target correlation matrix contains non-finite input")
    scale = x.std(axis=0, ddof=1)
    if np.any(scale <= 1e-14):
        raise ValueError("target correlation matrix contains a constant target")
    corr = np.corrcoef(x, rowvar=False)
    corr = np.clip(corr, -1.0, 1.0)
    np.fill_diagonal(corr, 1.0)
    return corr


def geometry_correlation(matrix: np.ndarray, transform: str) -> np.ndarray:
    return target_correlation(transformed_surface(matrix, transform))


def upper_triangle(matrix: np.ndarray) -> np.ndarray:
    x = np.asarray(matrix, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] != x.shape[1]:
        raise ValueError("upper triangle requires a square matrix")
    return x[np.triu_indices(len(x), k=1)]


def geometry_concordance(first: np.ndarray, second: np.ndarray) -> float:
    if np.asarray(first).shape != np.asarray(second).shape:
        raise ValueError("geometry matrices must have the same shape")
    value = stats.spearmanr(upper_triangle(first), upper_triangle(second)).statistic
    if not np.isfinite(value):
        raise ValueError("geometry concordance is not finite")
    return float(value)


def _rank_matrix(matrix: np.ndarray) -> np.ndarray:
    """Symmetric matrix holding the ranks of strict-upper-triangle entries."""
    x = np.asarray(matrix, dtype=np.float64)
    tri = np.triu_indices(len(x), k=1)
    ranks = stats.rankdata(x[tri], method="average")
    out = np.zeros_like(x)
    out[tri] = ranks
    out[(tri[1], tri[0])] = ranks
    return out


def qap_test(
    docking_geometry: np.ndarray,
    experimental_geometry: np.ndarray,
    permutations: int,
    seed: int,
) -> dict:
    """One-sided target-label QAP test for positive geometry concordance."""
    if permutations < 1:
        raise ValueError("QAP requires at least one permutation")
    p = len(docking_geometry)
    if docking_geometry.shape != (p, p) or experimental_geometry.shape != (p, p):
        raise ValueError("QAP inputs must be equally sized square matrices")
    tri = np.triu_indices(p, k=1)
    docking_ranks = _rank_matrix(docking_geometry)
    experimental_ranks = stats.rankdata(
        experimental_geometry[tri], method="average"
    ).astype(np.float64)
    experimental_ranks -= experimental_ranks.mean()
    experimental_norm = float(np.linalg.norm(experimental_ranks))
    docking_vector = docking_ranks[tri]
    docking_centered = docking_vector - docking_vector.mean()
    docking_norm = float(np.linalg.norm(docking_centered))
    if experimental_norm <= 0 or docking_norm <= 0:
        raise ValueError("QAP input has constant off-diagonal geometry")
    observed = float(
        np.dot(docking_centered, experimental_ranks)
        / (docking_norm * experimental_norm)
    )
    rng = np.random.default_rng(seed)
    null = np.empty(permutations, dtype=np.float64)
    for repetition in range(permutations):
        order = rng.permutation(p)
        vector = docking_ranks[np.ix_(order, order)][tri]
        vector = vector - vector.mean()
        null[repetition] = np.dot(vector, experimental_ranks) / (
            np.linalg.norm(vector) * experimental_norm
        )
    return {
        "observed_spearman": observed,
        "permutations": int(permutations),
        "one_sided_p_positive": float(
            (1 + np.sum(null >= observed)) / (permutations + 1)
        ),
        "null_interval_95": [
            float(np.quantile(null, 0.025)),
            float(np.quantile(null, 0.975)),
        ],
        "null_median": float(np.median(null)),
        "seed": int(seed),
    }


def _ranked_geometry_concordance(
    rank_matrix: np.ndarray,
    experimental_ranks: np.ndarray,
    experimental_norm: float,
    triangle: tuple[np.ndarray, np.ndarray],
) -> float:
    vector = rank_matrix[triangle]
    vector = vector - vector.mean()
    norm = float(np.linalg.norm(vector))
    if norm <= 0 or experimental_norm <= 0:
        raise ValueError("paired QAP input has constant off-diagonal geometry")
    return float(np.dot(vector, experimental_ranks) / (norm * experimental_norm))


def fixed_experimental_geometry_paired_qap(
    raw_docking_geometry: np.ndarray,
    centered_docking_geometry: np.ndarray,
    fixed_experimental_geometry: np.ndarray,
    permutations: int,
    seed: int,
) -> dict:
    """Paired QAP for two docking predictors of one fixed experimental endpoint.

    The experimental geometry is never changed.  The same target-label permutation
    is applied to both docking geometries in each draw, so the reported contrast is
    the uniquely interpretable effect of changing the docking representation.
    """
    if permutations < 1:
        raise ValueError("paired QAP requires at least one permutation")
    p = len(raw_docking_geometry)
    matrices = (raw_docking_geometry, centered_docking_geometry, fixed_experimental_geometry)
    if any(np.asarray(matrix).shape != (p, p) for matrix in matrices):
        raise ValueError("paired QAP inputs must be equally sized square matrices")
    tri = np.triu_indices(p, k=1)
    raw_ranks = _rank_matrix(raw_docking_geometry)
    centered_ranks = _rank_matrix(centered_docking_geometry)
    experimental_ranks = stats.rankdata(
        fixed_experimental_geometry[tri], method="average"
    ).astype(float)
    experimental_ranks -= experimental_ranks.mean()
    experimental_norm = float(np.linalg.norm(experimental_ranks))
    observed_raw = _ranked_geometry_concordance(
        raw_ranks, experimental_ranks, experimental_norm, tri
    )
    observed_centered = _ranked_geometry_concordance(
        centered_ranks, experimental_ranks, experimental_norm, tri
    )
    observed_delta = observed_centered - observed_raw
    rng = np.random.default_rng(seed)
    null = np.empty(permutations, dtype=np.float64)
    for repetition in range(permutations):
        order = rng.permutation(p)
        raw_permuted = raw_ranks[np.ix_(order, order)]
        centered_permuted = centered_ranks[np.ix_(order, order)]
        null[repetition] = _ranked_geometry_concordance(
            centered_permuted, experimental_ranks, experimental_norm, tri
        ) - _ranked_geometry_concordance(
            raw_permuted, experimental_ranks, experimental_norm, tri
        )
    return {
        "fixed_experimental_endpoint": "two-way-centered experimental target geometry",
        "raw_docking_concordance": float(observed_raw),
        "centered_docking_concordance": float(observed_centered),
        "centered_minus_raw_docking": float(observed_delta),
        "permutations": int(permutations),
        "one_sided_p_positive_delta": float(
            (1 + np.sum(null >= observed_delta)) / (permutations + 1)
        ),
        "null_interval_95": [
            float(np.quantile(null, 0.025)),
            float(np.quantile(null, 0.975)),
        ],
        "null_median": float(np.median(null)),
        "seed": int(seed),
    }


def matched_estimand_transition_qap(
    raw_docking_geometry: np.ndarray,
    raw_experimental_geometry: np.ndarray,
    centered_docking_geometry: np.ndarray,
    centered_experimental_geometry: np.ndarray,
    permutations: int,
    seed: int,
) -> dict:
    """QAP for the joint raw/raw to centered/centered estimand transition.

    This changes both sides of the comparison and therefore must not be interpreted
    as a gain caused by centering the docking surface.  It is retained only as an
    explicitly labelled matched-estimand transition sensitivity.
    """
    if permutations < 1:
        raise ValueError("paired QAP requires at least one permutation")
    matrices = (
        raw_docking_geometry,
        raw_experimental_geometry,
        centered_docking_geometry,
        centered_experimental_geometry,
    )
    p = len(raw_docking_geometry)
    if any(np.asarray(matrix).shape != (p, p) for matrix in matrices):
        raise ValueError("paired QAP inputs must be equally sized square matrices")
    tri = np.triu_indices(p, k=1)
    raw_docking_ranks = _rank_matrix(raw_docking_geometry)
    centered_docking_ranks = _rank_matrix(centered_docking_geometry)
    raw_experimental_ranks = stats.rankdata(
        raw_experimental_geometry[tri], method="average"
    ).astype(float)
    centered_experimental_ranks = stats.rankdata(
        centered_experimental_geometry[tri], method="average"
    ).astype(float)
    raw_experimental_ranks -= raw_experimental_ranks.mean()
    centered_experimental_ranks -= centered_experimental_ranks.mean()
    raw_experimental_norm = float(np.linalg.norm(raw_experimental_ranks))
    centered_experimental_norm = float(np.linalg.norm(centered_experimental_ranks))

    observed_raw = _ranked_geometry_concordance(
        raw_docking_ranks, raw_experimental_ranks, raw_experimental_norm, tri
    )
    observed_centered = _ranked_geometry_concordance(
        centered_docking_ranks,
        centered_experimental_ranks,
        centered_experimental_norm,
        tri,
    )
    observed_delta = observed_centered - observed_raw
    rng = np.random.default_rng(seed)
    null = np.empty(permutations, dtype=np.float64)
    for repetition in range(permutations):
        order = rng.permutation(p)
        raw_permuted = raw_docking_ranks[np.ix_(order, order)]
        centered_permuted = centered_docking_ranks[np.ix_(order, order)]
        null[repetition] = _ranked_geometry_concordance(
            centered_permuted,
            centered_experimental_ranks,
            centered_experimental_norm,
            tri,
        ) - _ranked_geometry_concordance(
            raw_permuted, raw_experimental_ranks, raw_experimental_norm, tri
        )
    return {
        "warning": (
            "both docking and experimental estimands change; this is not a "
            "centering-gain estimate"
        ),
        "raw_concordance": float(observed_raw),
        "centered_concordance": float(observed_centered),
        "centered_centered_minus_raw_raw": float(observed_delta),
        "permutations": int(permutations),
        "one_sided_p_positive_transition": float(
            (1 + np.sum(null >= observed_delta)) / (permutations + 1)
        ),
        "null_interval_95": [
            float(np.quantile(null, 0.025)),
            float(np.quantile(null, 0.975)),
        ],
        "null_median": float(np.median(null)),
        "seed": int(seed),
    }


def target_jackknife(
    docking_geometry: np.ndarray, experimental_geometry: np.ndarray
) -> tuple[dict, np.ndarray]:
    p = len(docking_geometry)
    records = np.empty(p, dtype=np.float64)
    for deleted in range(p):
        keep = np.arange(p) != deleted
        records[deleted] = geometry_concordance(
            docking_geometry[np.ix_(keep, keep)],
            experimental_geometry[np.ix_(keep, keep)],
        )
    return (
        {
            "minimum": float(records.min()),
            "median": float(np.median(records)),
            "maximum": float(records.max()),
        },
        records,
    )


def describe(values: np.ndarray) -> dict:
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    if not len(x):
        raise ValueError("cannot describe an empty numeric vector")
    return {
        "n": int(len(x)),
        "mean": float(x.mean()),
        "median": float(np.median(x)),
        "standard_deviation": float(x.std(ddof=1)) if len(x) > 1 else 0.0,
        "interval_95": [
            float(np.quantile(x, 0.025)),
            float(np.quantile(x, 0.975)),
        ],
        "interval_90": [
            float(np.quantile(x, 0.05)),
            float(np.quantile(x, 0.95)),
        ],
        "minimum": float(x.min()),
        "maximum": float(x.max()),
    }


def holm_adjust(p_values: Iterable[float]) -> np.ndarray:
    """Holm family-wise-error adjusted p-values in original order."""
    values = np.asarray(tuple(p_values), dtype=np.float64)
    if values.ndim != 1 or not len(values):
        raise ValueError("Holm adjustment requires at least one p-value")
    if np.any(~np.isfinite(values)) or np.any((values < 0) | (values > 1)):
        raise ValueError("Holm adjustment requires finite p-values in [0, 1]")
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    adjusted_sorted = np.maximum.accumulate(
        (len(values) - np.arange(len(values))) * sorted_values
    )
    adjusted = np.empty_like(values)
    adjusted[order] = np.minimum(adjusted_sorted, 1.0)
    return adjusted


def _cluster_members(labels: np.ndarray) -> list[np.ndarray]:
    labels = np.asarray(labels)
    return [np.flatnonzero(labels == label) for label in np.unique(labels)]


def weighted_two_way_center(matrix: np.ndarray, ligand_weights: np.ndarray) -> np.ndarray:
    x = np.asarray(matrix, dtype=np.float64)
    weights = np.asarray(ligand_weights, dtype=np.float64)
    if weights.shape != (len(x),) or np.any(weights <= 0):
        raise ValueError("Bayesian-bootstrap weights must be positive and align to rows")
    normalized = weights / weights.sum()
    column_mean = normalized @ x
    row_mean = x.mean(axis=1)
    grand_mean = float(normalized @ row_mean)
    return x - column_mean[None, :] - row_mean[:, None] + grand_mean


def weighted_target_correlation(
    matrix: np.ndarray, ligand_weights: np.ndarray
) -> np.ndarray:
    x = np.asarray(matrix, dtype=np.float64)
    weights = np.asarray(ligand_weights, dtype=np.float64)
    normalized = weights / weights.sum()
    centered = x - normalized @ x
    covariance = (centered * normalized[:, None]).T @ centered
    scale = np.sqrt(np.diag(covariance))
    if np.any(scale <= 1e-14):
        raise ValueError("weighted target correlation contains a constant target")
    correlation = covariance / np.outer(scale, scale)
    correlation = np.clip(correlation, -1.0, 1.0)
    np.fill_diagonal(correlation, 1.0)
    return correlation


def geometry_bootstrap(
    docking_geometry: np.ndarray,
    experimental_matrix: np.ndarray,
    transform: str,
    repeats: int,
    seed: int,
    cluster_labels: np.ndarray | None = None,
) -> dict:
    """Bayesian bootstrap ligands or clusters, keeping targets fixed.

    Positive exponential weights avoid undefined target correlations in sparse
    DAVIS resamples (ordinary multinomial resampling can omit every non-floor
    observation for a target).  For clusters, one weight is drawn per cluster and
    inherited by all member ligands.
    """
    if repeats < 1:
        raise ValueError("bootstrap requires at least one repeat")
    x = np.asarray(experimental_matrix, dtype=np.float64)
    rng = np.random.default_rng(seed)
    records = np.empty(repeats, dtype=np.float64)
    if transform != "center_then_correlation":
        raise ValueError("weighted geometry bootstrap is defined for centered geometry")
    if cluster_labels is None:
        cluster_count = None
        cluster_index = None
    else:
        labels = np.asarray(cluster_labels)
        if labels.shape != (len(x),):
            raise ValueError("cluster labels do not match experimental ligands")
        _, cluster_index = np.unique(labels, return_inverse=True)
        cluster_count = int(cluster_index.max() + 1)
    for repetition in range(repeats):
        if cluster_index is None:
            weights = rng.exponential(size=len(x))
        else:
            weights = rng.exponential(size=cluster_count)[cluster_index]
        centered = weighted_two_way_center(x, weights)
        experimental_geometry = weighted_target_correlation(centered, weights)
        records[repetition] = geometry_concordance(
            docking_geometry, experimental_geometry
        )
    return {
        "method": "Bayesian bootstrap with iid Exp(1) weights",
        "unit": "ligand" if cluster_labels is None else "chemical cluster",
        "clusters": cluster_count,
        "fixed_target_panel": int(x.shape[1]),
        "seed": int(seed),
        **describe(records),
    }


def projection_matched_independent_column_null(
    reference_matrix: np.ndarray,
    experimental_geometry: np.ndarray,
    transform: str,
    repeats: int,
    seed: int,
    full_reference_observed: float,
) -> dict:
    """Destroy ligand-wise target coordination, then apply the identical projection.

    Independent row permutations within every target preserve the empirical target
    marginals.  Applying the requested transform after permutation retains any geometry
    induced mechanically by column preprocessing and the row-zero-sum projection.
    """
    if repeats < 1:
        raise ValueError("projection null requires at least one repeat")
    x = np.asarray(reference_matrix, dtype=np.float64)
    if transform == "raw":
        base = x
    elif transform == "center_then_correlation":
        base = x
    elif transform == "z_before_center":
        base = column_zscore(x)
    elif transform == "rank_normal_before_center":
        base = column_rank_normal_scores(x)
    else:
        raise ValueError(f"unknown transform: {transform}")
    rng = np.random.default_rng(seed)
    matched_support_observed = geometry_concordance(
        geometry_correlation(x, transform), experimental_geometry
    )
    null = np.empty(repeats, dtype=np.float64)
    work = np.empty_like(base)
    for repetition in range(repeats):
        for column in range(base.shape[1]):
            work[:, column] = base[rng.permutation(len(base)), column]
        projected = work if transform == "raw" else two_way_center(work)
        null[repetition] = geometry_concordance(
            target_correlation(projected), experimental_geometry
        )
    return {
        "null": "independent within-target ligand permutations followed by the identical transform",
        "reference_ligands": int(len(x)),
        "repeats": int(repeats),
        "seed": int(seed),
        "matched_support_observed": float(matched_support_observed),
        "full_reference_observed_descriptive": float(full_reference_observed),
        "one_sided_p_observed_positive": float(
            (1 + np.sum(null >= matched_support_observed)) / (repeats + 1)
        ),
        **describe(null),
    }


def load_pkis2_full() -> pd.DataFrame:
    """Load and validate all 645 populated PKIS2 rows without rescanning DOCKSTRING."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Unknown extension is not supported and will be removed",
            category=UserWarning,
            module=r"openpyxl\.worksheet\._reader",
        )
        frame = pd.read_excel(pkis2.DEFAULT_PKIS2, sheet_name=pkis2.PKIS2_SHEET)
    if len(frame.columns) != 413:
        raise ValueError("PKIS2 workbook no longer contains 413 columns")
    metadata = ["Regno", "Compound", "Chemotype", "Smiles"]
    required = [*metadata, *pkis2.TARGET_MAP.values()]
    missing = [column for column in required if column not in frame]
    if missing:
        raise ValueError(f"PKIS2 workbook is missing required columns: {missing}")
    frame = frame.dropna(subset=metadata).copy()
    if len(frame) != 645:
        raise ValueError(f"expected 645 populated PKIS2 rows, observed {len(frame)}")
    frame.insert(0, "pkis2_row", np.arange(len(frame), dtype=int))
    frame.insert(1, "source_excel_row", np.arange(2, len(frame) + 2, dtype=int))
    frame = pkis2._identity_table(frame, "Smiles", scan="full")
    experimental = frame[list(pkis2.TARGET_MAP.values())].apply(
        pd.to_numeric, errors="coerce"
    )
    experimental.columns = list(TARGETS)
    if experimental.isna().any().any():
        raise ValueError("selected PKIS2 645 x 21 block is not dense")
    if ((experimental < 0) | (experimental > 100)).any().any():
        raise ValueError("selected PKIS2 percentage is outside [0,100]")
    frame[list(TARGETS)] = experimental
    return frame


def load_pkis1_full(zip_path: Path, cdk2_construct: str = "CDK2/cyclinA") -> pd.DataFrame:
    """Load the public PKIS1 1-uM Nanosyn panel from its original supplement.

    The source archive is validated byte-for-byte and is never copied into the
    reproducibility package.  Duplicate compound IDs are technical duplicate rows
    with identical SMILES; their inhibition values are aggregated by the median.
    """
    path = Path(zip_path)
    if sha256_file(path) != PKIS1_SHA256:
        raise ValueError("PKIS1 supplement SHA256 does not match the documented source")
    if cdk2_construct not in {"CDK2/cyclinA", "CDK2/cyclinE"}:
        raise ValueError("unsupported PKIS1 CDK2 construct")
    target_map = PKIS1_TARGET_MAP.copy()
    target_map["CDK2"] = cdk2_construct
    with zipfile.ZipFile(path) as archive:
        missing = [
            member
            for member in (PKIS1_SMILES_MEMBER, PKIS1_HEATMAP_MEMBER)
            if member not in archive.namelist()
        ]
        if missing:
            raise ValueError(f"PKIS1 supplement is missing members: {missing}")
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Unknown extension is not supported and will be removed",
                category=UserWarning,
            )
            structures = pd.read_excel(
                io.BytesIO(archive.read(PKIS1_SMILES_MEMBER)), sheet_name="Sheet1"
            )
            heatmap = pd.read_excel(
                io.BytesIO(archive.read(PKIS1_HEATMAP_MEMBER)),
                sheet_name="PKIS HeatMap 1uM",
                header=2,
            )
    required_structure = {"Compound ID", "SMILES"}
    required_heatmap = {"Compound ID", *target_map.values()}
    if not required_structure.issubset(structures) or not required_heatmap.issubset(
        heatmap
    ):
        raise ValueError("PKIS1 workbook schema differs from the documented release")
    structure_pairs = structures[["Compound ID", "SMILES"]].dropna().drop_duplicates()
    if structure_pairs["Compound ID"].duplicated().any():
        raise ValueError("a PKIS1 compound ID maps to more than one SMILES")
    values = heatmap[["Compound ID", *target_map.values()]].copy()
    for column in target_map.values():
        values[column] = pd.to_numeric(values[column], errors="coerce")
    values = values.groupby("Compound ID", sort=True, as_index=False).median(
        numeric_only=True
    )
    values = values.merge(structure_pairs, on="Compound ID", how="left", validate="1:1")
    values = values.dropna(subset=["SMILES", *target_map.values()]).reset_index(drop=True)
    experiment = values[list(target_map.values())].copy()
    experiment.columns = list(target_map)
    if len(values) != 360 or experiment.isna().any().any():
        raise ValueError(
            f"expected a complete PKIS1 360 x 20 block, observed {experiment.shape}"
        )
    identity = davis._identity_table(values, "SMILES", scan="full")
    identity[list(target_map)] = experiment
    identity.attrs["cdk2_construct"] = cdk2_construct
    return identity


def murcko_scaffold_keys(smiles: pd.Series) -> np.ndarray:
    """Canonical cyclic Murcko keys; acyclic molecules receive an empty key."""
    keys: list[str] = []
    for value in smiles.astype(str):
        molecule = davis.Chem.MolFromSmiles(value)
        if molecule is None:
            raise ValueError(f"RDKit could not parse a SMILES for scaffold exclusion: {value!r}")
        keys.append(davis.MurckoScaffold.MurckoScaffoldSmiles(mol=molecule))
    return np.asarray(keys, dtype=object)


def paired_cluster_bootstrap(
    differences: np.ndarray,
    smiles: pd.Series,
    repeats: int,
    seed: int,
) -> dict:
    murcko = davis._murcko_labels(smiles.reset_index(drop=True))
    butina = davis._butina_labels(smiles.reset_index(drop=True))
    return davis._uncertainty(
        np.asarray(differences, dtype=np.float64),
        murcko,
        butina,
        repeats,
        seed,
    )


def fit_molecular_weight_component(
    reference_scores: np.ndarray, molecular_weight: np.ndarray
) -> dict:
    scores = np.asarray(reference_scores, dtype=np.float64)
    mw = np.asarray(molecular_weight, dtype=np.float64)
    if scores.shape[0] != len(mw) or not np.isfinite(mw).all():
        raise ValueError("reference scores and molecular weights do not align")
    mean = float(mw.mean())
    sd = float(mw.std(ddof=1))
    if sd <= 0:
        raise ValueError("reference molecular weight has zero variance")
    z = (mw - mean) / sd
    slopes = (z @ scores) / (z @ z)
    centered = slopes - slopes.mean()
    corrected_reference = scores - z[:, None] * centered[None, :]
    return {
        "molecular_weight_mean": mean,
        "molecular_weight_sd": sd,
        "target_mean": scores.mean(axis=0),
        "target_sd": scores.std(axis=0, ddof=1),
        "corrected_target_sd": corrected_reference.std(axis=0, ddof=1),
        "raw_target_slopes_per_mw_sd": slopes,
        "centered_target_slopes_per_mw_sd": centered,
        "maximum_centered_slope_sum_error": float(abs(centered.sum())),
    }


def apply_molecular_weight_correction(
    scores: np.ndarray, molecular_weight: np.ndarray, fit: dict
) -> np.ndarray:
    x = np.asarray(scores, dtype=np.float64)
    mw = np.asarray(molecular_weight, dtype=np.float64)
    z = (mw - fit["molecular_weight_mean"]) / fit["molecular_weight_sd"]
    slopes = np.asarray(fit["centered_target_slopes_per_mw_sd"], dtype=np.float64)
    if x.shape != (len(mw), len(slopes)):
        raise ValueError("evaluation scores, molecular weights and slopes do not align")
    return x - z[:, None] * slopes[None, :]


def _descriptor_molecular_weight(smiles: pd.Series) -> np.ndarray:
    frame = evidence.molecular_descriptor_frame(smiles.reset_index(drop=True))
    if frame.molecular_weight.isna().any():
        raise ValueError("evaluation panel contains an invalid molecular structure")
    return frame.molecular_weight.to_numpy(dtype=np.float64)


def _metric_vector_davis(
    scores: np.ndarray, experimental: np.ndarray, censored: np.ndarray
) -> tuple[dict, np.ndarray]:
    return davis._pairwise_metrics(
        scores, experimental, censored, margin=0.0, stratum="all_informative"
    )


def _metric_vector_pkis2(
    scores: np.ndarray, experimental: np.ndarray
) -> tuple[dict, np.ndarray]:
    summary, vector, _ = pkis2._pairwise_concordance(
        scores, experimental, margin=10.0
    )
    return summary, vector


def slope_correction_panel(
    dataset: str,
    docking_scores: pd.DataFrame,
    experimental: pd.DataFrame,
    smiles: pd.Series,
    fit: dict,
    bootstrap_repeats: int,
    seed: int,
    censored: pd.DataFrame | None = None,
) -> tuple[dict, list[dict]]:
    raw = np.minimum(docking_scores.to_numpy(dtype=np.float64), 0.0)
    exp = experimental.to_numpy(dtype=np.float64)
    mw = _descriptor_molecular_weight(smiles)
    corrected = apply_molecular_weight_correction(raw, mw, fit)
    target_mean = np.asarray(fit["target_mean"], dtype=np.float64)
    target_sd = np.asarray(fit["target_sd"], dtype=np.float64)
    corrected_sd = np.asarray(fit["corrected_target_sd"], dtype=np.float64)
    representations = OrderedDict(
        [
            ("absolute", raw),
            ("absolute_mw_corrected", corrected),
            ("target_centered", raw - target_mean[None, :]),
            ("target_centered_mw_corrected", corrected - target_mean[None, :]),
            ("column_z", (raw - target_mean[None, :]) / target_sd[None, :]),
            (
                "column_z_mw_corrected_original_sd",
                (corrected - target_mean[None, :]) / target_sd[None, :],
            ),
            (
                "column_z_mw_corrected_refit_sd",
                (corrected - target_mean[None, :]) / corrected_sd[None, :],
            ),
        ]
    )
    vectors: dict[str, np.ndarray] = {}
    estimates: dict[str, dict] = {}
    censor_array = (
        censored.to_numpy(dtype=bool) if censored is not None else None
    )
    for name, surface in representations.items():
        if dataset == "DAVIS":
            if censor_array is None:
                raise ValueError("DAVIS correction analysis requires censor flags")
            summary, vector = _metric_vector_davis(surface, exp, censor_array)
        elif dataset == "PKIS2":
            summary, vector = _metric_vector_pkis2(surface, exp)
        else:
            raise ValueError(f"unknown correction panel: {dataset}")
        vectors[name] = vector
        estimates[name] = summary

    contrasts = OrderedDict(
        [
            ("absolute", ("absolute_mw_corrected", "absolute")),
            (
                "target_centered",
                ("target_centered_mw_corrected", "target_centered"),
            ),
            (
                "column_z_original_sd",
                ("column_z_mw_corrected_original_sd", "column_z"),
            ),
            (
                "column_z_refit_sd",
                ("column_z_mw_corrected_refit_sd", "column_z"),
            ),
        ]
    )
    contrast_records: dict[str, dict] = {}
    jackknife_rows: list[dict] = []
    for contrast, (after, before) in contrasts.items():
        difference = vectors[after] - vectors[before]
        uncertainty = paired_cluster_bootstrap(
            difference,
            smiles,
            bootstrap_repeats,
            seed + 100 * list(contrasts).index(contrast),
        )
        target_deleted: list[float] = []
        for deleted, target in enumerate(TARGETS):
            keep = np.arange(len(TARGETS)) != deleted
            if dataset == "DAVIS":
                before_summary, _ = _metric_vector_davis(
                    representations[before][:, keep],
                    exp[:, keep],
                    censor_array[:, keep],  # type: ignore[index]
                )
                after_summary, _ = _metric_vector_davis(
                    representations[after][:, keep],
                    exp[:, keep],
                    censor_array[:, keep],  # type: ignore[index]
                )
            else:
                before_summary, _ = _metric_vector_pkis2(
                    representations[before][:, keep], exp[:, keep]
                )
                after_summary, _ = _metric_vector_pkis2(
                    representations[after][:, keep], exp[:, keep]
                )
            value = float(
                after_summary["mean_per_ligand_pairwise_concordance"]
                - before_summary["mean_per_ligand_pairwise_concordance"]
            )
            target_deleted.append(value)
            jackknife_rows.append(
                {
                    "analysis": "mw_slope_correction",
                    "dataset": dataset,
                    "transform": contrast,
                    "deleted_target": target,
                    "estimate": value,
                }
            )
        contrast_records[contrast] = {
            "after_minus_before_plugin": float(np.nanmean(difference)),
            "after": after,
            "before": before,
            "paired_uncertainty": uncertainty,
            "target_delete_one": {
                "minimum": float(np.min(target_deleted)),
                "median": float(np.median(target_deleted)),
                "maximum": float(np.max(target_deleted)),
            },
        }
    return (
        {
            "status": "exploratory; defined after inspection of residual modes",
            "dataset": dataset,
            "n_ligands": int(len(raw)),
            "n_targets": int(raw.shape[1]),
            "endpoint": (
                "all-informative pKd target pairs; both-floor pairs excluded; margin 0"
                if dataset == "DAVIS"
                else "percentage-inhibition target pairs differing by more than 10 points"
            ),
            "representations": estimates,
            "contrasts": contrast_records,
        },
        jackknife_rows,
    )


def _observed_pairwise_accuracy(
    scores: np.ndarray, experimental: np.ndarray
) -> tuple[dict, np.ndarray]:
    scores = np.asarray(scores, dtype=np.float64)
    experimental = np.asarray(experimental, dtype=np.float64)
    records = np.full(len(scores), np.nan)
    evaluated_pairs = 0
    for row, (score_row, exp_row) in enumerate(zip(scores, experimental)):
        observed = np.flatnonzero(np.isfinite(exp_row))
        values: list[float] = []
        for position, first in enumerate(observed):
            for second in observed[position + 1 :]:
                exp_difference = exp_row[first] - exp_row[second]
                if abs(exp_difference) <= 1e-12:
                    continue
                score_difference = score_row[first] - score_row[second]
                if abs(score_difference) <= 1e-12:
                    values.append(0.5)
                else:
                    values.append(
                        float(np.sign(exp_difference) == -np.sign(score_difference))
                    )
        if values:
            records[row] = float(np.mean(values))
            evaluated_pairs += len(values)
    return (
        {
            "mean_per_ligand_pairwise_concordance": float(np.nanmean(records)),
            "evaluated_ligands": int(np.isfinite(records).sum()),
            "evaluated_pairs": int(evaluated_pairs),
        },
        records,
    )


def chembl_negative_sensitivity(
    bootstrap_repeats: int, seed: int
) -> dict:
    """Apply the same correction to the exact-relation matched 74 x 6 block."""
    columns = [*evidence.DOCK44, "Canonical SMILES", "Cleaned SMILES"]
    canonical = pd.read_csv(
        evidence.source_path("df_final_v4.csv"), usecols=columns
    )
    canonical["analysis_smiles"] = canonical["Cleaned SMILES"].fillna(
        canonical["Canonical SMILES"]
    )
    canonical["inchikey"] = evidence.full_inchikeys(canonical["analysis_smiles"])
    docking_frame = canonical[evidence.DOCK44].apply(
        pd.to_numeric, errors="coerce"
    ).clip(upper=0)
    docking_frame = docking_frame.fillna(docking_frame.mean())
    matched_dock, matched_observed, _ = evidence.matched_experimental_matrices()
    matched_smiles = pd.read_csv(
        evidence.source_path(
            "negative_results_paper/analysis/exp_positive_control_smiles.csv"
        )
    ).set_index("inchikey")["smiles"].reindex(matched_dock.index)
    if matched_smiles.isna().any():
        raise ValueError("matched ChEMBL panel is missing ligand SMILES")
    keep = ~canonical.inchikey.isin(set(matched_dock.index))
    target_columns = list(matched_dock.columns)
    reference_scores = docking_frame.loc[keep, target_columns].to_numpy(dtype=float)
    reference_mw = _descriptor_molecular_weight(
        canonical.loc[keep, "analysis_smiles"]
    )
    fit = fit_molecular_weight_component(reference_scores, reference_mw)
    raw = matched_dock.to_numpy(dtype=float)
    corrected = apply_molecular_weight_correction(
        raw, _descriptor_molecular_weight(matched_smiles), fit
    )
    target_mean = np.asarray(fit["target_mean"])
    surfaces = {
        "absolute": raw,
        "absolute_mw_corrected": corrected,
        "target_centered": raw - target_mean[None, :],
        "target_centered_mw_corrected": corrected - target_mean[None, :],
    }
    estimates: dict[str, dict] = {}
    vectors: dict[str, np.ndarray] = {}
    for name, surface in surfaces.items():
        estimates[name], vectors[name] = _observed_pairwise_accuracy(
            surface, matched_observed.to_numpy(dtype=float)
        )
    contrasts = {}
    for contrast, after, before in (
        ("absolute", "absolute_mw_corrected", "absolute"),
        (
            "target_centered",
            "target_centered_mw_corrected",
            "target_centered",
        ),
    ):
        difference = vectors[after] - vectors[before]
        contrasts[contrast] = {
            "after_minus_before_plugin": float(np.nanmean(difference)),
            "paired_uncertainty": paired_cluster_bootstrap(
                difference,
                matched_smiles,
                bootstrap_repeats,
                seed + (0 if contrast == "absolute" else 100),
            ),
        }
    return {
        "status": "negative generalization sensitivity; not a co-primary endpoint",
        "n_ligands": int(len(raw)),
        "n_targets": int(raw.shape[1]),
        "experimental_handling": "observed exact-relation median cells only; no imputation",
        "representations": estimates,
        "contrasts": contrasts,
    }


def _geometry_rows(
    dataset: str,
    support: str,
    target_names: tuple[str, ...] | list[str],
    docking_matrix: np.ndarray,
    experimental_matrix: np.ndarray,
    qap_permutations: int,
    bootstrap_repeats: int,
    seed: int,
    smiles: pd.Series | None,
    projection_reference: np.ndarray | None,
    projection_null_repeats: int,
) -> tuple[dict, list[dict], list[dict]]:
    target_names = tuple(target_names)
    if docking_matrix.shape[1] != len(target_names):
        raise ValueError("target names do not align with the docking geometry")
    records: dict[str, dict] = {}
    csv_rows: list[dict] = []
    jackknife_rows: list[dict] = []
    geometry_pairs: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    murcko = davis._murcko_labels(smiles) if smiles is not None else None
    butina = davis._butina_labels(smiles) if smiles is not None else None
    for transform_index, transform in enumerate(TRANSFORMS):
        docking_geometry = geometry_correlation(docking_matrix, transform)
        experimental_geometry = geometry_correlation(experimental_matrix, transform)
        geometry_pairs[transform] = (docking_geometry, experimental_geometry)
        concordance = geometry_concordance(docking_geometry, experimental_geometry)
        qap = qap_test(
            docking_geometry,
            experimental_geometry,
            qap_permutations,
            seed + 1000 * transform_index,
        )
        jackknife, jackknife_values = target_jackknife(
            docking_geometry, experimental_geometry
        )
        for target, value in zip(target_names, jackknife_values):
            jackknife_rows.append(
                {
                    "analysis": "geometry_concordance",
                    "dataset": dataset,
                    "support": support,
                    "transform": transform,
                    "deleted_target": target,
                    "estimate": float(value),
                }
            )
        record: dict = {
            "description": TRANSFORMS[transform],
            "concordance_spearman": concordance,
            "qap": qap,
            "target_delete_one": jackknife,
        }
        if transform == "center_then_correlation" and smiles is not None:
            record["experimental_support_uncertainty"] = {
                "bayesian_ligand_bootstrap": geometry_bootstrap(
                    docking_geometry,
                    experimental_matrix,
                    transform,
                    bootstrap_repeats,
                    seed + 10000,
                ),
                "bayesian_murcko_cluster_bootstrap": geometry_bootstrap(
                    docking_geometry,
                    experimental_matrix,
                    transform,
                    bootstrap_repeats,
                    seed + 11000,
                    murcko,
                ),
                "bayesian_butina_cluster_bootstrap": geometry_bootstrap(
                    docking_geometry,
                    experimental_matrix,
                    transform,
                    bootstrap_repeats,
                    seed + 12000,
                    butina,
                ),
            }
        if projection_reference is not None:
            record["projection_matched_independent_column_null"] = (
                projection_matched_independent_column_null(
                    projection_reference,
                    experimental_geometry,
                    transform,
                    projection_null_repeats,
                    seed + 20000 + 1000 * transform_index,
                    concordance,
                )
            )
        records[transform] = record
        boot = record.get("experimental_support_uncertainty", {})
        csv_rows.append(
            {
                "dataset": dataset,
                "support": support,
                "n_ligands": int(len(experimental_matrix)),
                "n_targets": int(experimental_matrix.shape[1]),
                "transform": transform,
                "concordance_spearman": concordance,
                "qap_one_sided_p": qap["one_sided_p_positive"],
                "target_loo_min": jackknife["minimum"],
                "target_loo_median": jackknife["median"],
                "target_loo_max": jackknife["maximum"],
                "bayesian_ligand_low": boot.get("bayesian_ligand_bootstrap", {}).get(
                    "interval_95", [None, None]
                )[0],
                "bayesian_ligand_high": boot.get("bayesian_ligand_bootstrap", {}).get(
                    "interval_95", [None, None]
                )[1],
                "bayesian_murcko_low": boot.get(
                    "bayesian_murcko_cluster_bootstrap", {}
                ).get("interval_95", [None, None])[0],
                "bayesian_murcko_high": boot.get(
                    "bayesian_murcko_cluster_bootstrap", {}
                ).get("interval_95", [None, None])[1],
                "bayesian_butina_low": boot.get(
                    "bayesian_butina_cluster_bootstrap", {}
                ).get("interval_95", [None, None])[0],
                "bayesian_butina_high": boot.get(
                    "bayesian_butina_cluster_bootstrap", {}
                ).get("interval_95", [None, None])[1],
                "projection_null_p": record.get(
                    "projection_matched_independent_column_null", {}
                ).get("one_sided_p_observed_positive"),
            }
        )
    raw_docking, raw_experimental = geometry_pairs["raw"]
    centered_docking, centered_experimental = geometry_pairs[
        "center_then_correlation"
    ]
    cross_transform_geometry = {
        "docking_raw__experimental_raw": {
            "concordance_spearman": geometry_concordance(
                raw_docking, raw_experimental
            ),
            "qap": records["raw"]["qap"],
        },
        "docking_raw__experimental_centered": {
            "concordance_spearman": geometry_concordance(
                raw_docking, centered_experimental
            ),
            "qap": qap_test(
                raw_docking,
                centered_experimental,
                qap_permutations,
                seed + 91_000,
            ),
        },
        "docking_centered__experimental_raw": {
            "concordance_spearman": geometry_concordance(
                centered_docking, raw_experimental
            ),
            "qap": qap_test(
                centered_docking,
                raw_experimental,
                qap_permutations,
                seed + 92_000,
            ),
        },
        "docking_centered__experimental_centered": {
            "concordance_spearman": geometry_concordance(
                centered_docking, centered_experimental
            ),
            "qap": records["center_then_correlation"]["qap"],
        },
    }
    fixed_endpoint = fixed_experimental_geometry_paired_qap(
        raw_docking,
        centered_docking,
        centered_experimental,
        qap_permutations,
        seed + 93_000,
    )
    matched_transition = matched_estimand_transition_qap(
        raw_docking,
        raw_experimental,
        centered_docking,
        centered_experimental,
        qap_permutations,
        seed + 94_000,
    )
    records["docking_by_experimental_transform_2x2"] = cross_transform_geometry
    records["fixed_centered_experimental_paired_qap"] = fixed_endpoint
    records["matched_estimand_transition_qap"] = matched_transition
    for row in csv_rows:
        if row["transform"] == "center_then_correlation":
            row["raw_docking_vs_centered_experimental"] = fixed_endpoint[
                "raw_docking_concordance"
            ]
            row["centered_minus_raw_docking_fixed_experimental"] = fixed_endpoint[
                "centered_minus_raw_docking"
            ]
            row["paired_qap_fixed_experimental_p"] = fixed_endpoint[
                "one_sided_p_positive_delta"
            ]
            row["matched_estimand_transition"] = matched_transition[
                "centered_centered_minus_raw_raw"
            ]
            row["matched_estimand_transition_p"] = matched_transition[
                "one_sided_p_positive_transition"
            ]
        else:
            row["raw_docking_vs_centered_experimental"] = None
            row["centered_minus_raw_docking_fixed_experimental"] = None
            row["paired_qap_fixed_experimental_p"] = None
            row["matched_estimand_transition"] = None
            row["matched_estimand_transition_p"] = None
    return records, csv_rows, jackknife_rows


def reference_support_sensitivity(
    reference: np.ndarray,
    panels: list[tuple[str, tuple[int, ...], np.ndarray]],
    support_size: int,
    support_seeds: tuple[int, ...],
) -> tuple[dict, pd.DataFrame]:
    """Repeat geometry estimation on independently drawn score-blind supports."""
    if support_size < 100 or support_size > len(reference):
        raise ValueError("invalid reference-support sensitivity size")
    rows: list[dict] = []
    for support_seed in support_seeds:
        rng = np.random.default_rng(support_seed)
        indices = np.sort(rng.choice(len(reference), support_size, replace=False))
        sampled = reference[indices]
        for dataset, target_indices, experimental in panels:
            target_sample = sampled[:, target_indices]
            for transform in TRANSFORMS:
                rho = geometry_concordance(
                    geometry_correlation(target_sample, transform),
                    geometry_correlation(experimental, transform),
                )
                rows.append(
                    {
                        "dataset": dataset,
                        "reference_support_size": int(support_size),
                        "reference_support_seed": int(support_seed),
                        "transform": transform,
                        "concordance_spearman": float(rho),
                    }
                )
    frame = pd.DataFrame(rows)
    summary: dict[str, dict] = {}
    for (dataset, transform), group in frame.groupby(["dataset", "transform"]):
        values = group.concordance_spearman.to_numpy(dtype=float)
        summary.setdefault(dataset, {})[transform] = describe(values)
    return summary, frame


def experimental_cross_panel_concordance(
    panels: list[tuple[str, tuple[str, ...], np.ndarray]],
    qap_permutations: int,
    seed: int,
) -> dict:
    """Quantify whether experimental panels themselves recover the same topology."""
    report: dict[str, dict] = {}
    for first_index, (first_name, first_targets, first_matrix) in enumerate(panels):
        for second_index in range(first_index + 1, len(panels)):
            second_name, second_targets, second_matrix = panels[second_index]
            common = tuple(target for target in first_targets if target in second_targets)
            first_columns = [first_targets.index(target) for target in common]
            second_columns = [second_targets.index(target) for target in common]
            key = f"{first_name}_vs_{second_name}"
            transform_records: dict[str, dict] = {}
            for transform_index, transform in enumerate(TRANSFORMS):
                first_geometry = geometry_correlation(
                    first_matrix[:, first_columns], transform
                )
                second_geometry = geometry_correlation(
                    second_matrix[:, second_columns], transform
                )
                transform_records[transform] = {
                    "concordance_spearman": geometry_concordance(
                        first_geometry, second_geometry
                    ),
                    "qap": qap_test(
                        first_geometry,
                        second_geometry,
                        qap_permutations,
                        seed
                        + first_index * 100_000
                        + second_index * 10_000
                        + transform_index * 1_000,
                    ),
                }
            report[key] = {
                "first_panel": first_name,
                "second_panel": second_name,
                "n_common_targets": int(len(common)),
                "common_targets": list(common),
                "transforms": transform_records,
            }
    return report


def _retrieval_metrics(labels: np.ndarray, scores: np.ndarray) -> dict:
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    positives = int(labels.sum())
    negatives = int(len(labels) - positives)
    if positives < 1 or negatives < 1:
        raise ValueError("retrieval metrics require both target-pair classes")
    ranks = stats.rankdata(scores, method="average")
    roc_auc = (
        float(ranks[labels == 1].sum()) - positives * (positives + 1) / 2
    ) / (positives * negatives)
    order = np.argsort(scores, kind="mergesort")[::-1]
    ordered_scores = scores[order]
    ordered_labels = labels[order]
    threshold_indices = np.r_[
        np.flatnonzero(np.diff(ordered_scores)), len(ordered_scores) - 1
    ]
    true_positives = np.cumsum(ordered_labels)[threshold_indices]
    predicted_positives = threshold_indices + 1
    precision = true_positives / predicted_positives
    recall = true_positives / positives
    average_precision = float(np.sum(np.diff(np.r_[0.0, recall]) * precision))
    return {
        "roc_auc": float(roc_auc),
        "average_precision": average_precision,
    }


def retrieval_support_bootstrap(
    raw_docking_geometry: np.ndarray,
    centered_docking_geometry: np.ndarray,
    experimental_matrix: np.ndarray,
    fraction: float,
    repeats: int,
    seed: int,
    cluster_labels: np.ndarray | None = None,
) -> dict:
    """Re-estimate experimental co-selectivity labels in every bootstrap draw.

    Docking predictors are fixed.  Positive target-pair labels are reconstructed
    from the weighted, two-way-centered experimental geometry in each Bayesian
    ligand or chemical-cluster bootstrap, so uncertainty includes instability of
    the endpoint definition itself rather than conditioning on plugin labels.
    """
    if repeats < 1:
        raise ValueError("retrieval bootstrap requires at least one repeat")
    x = np.asarray(experimental_matrix, dtype=np.float64)
    p = x.shape[1]
    if raw_docking_geometry.shape != (p, p) or centered_docking_geometry.shape != (
        p,
        p,
    ):
        raise ValueError("retrieval bootstrap geometries do not align to targets")
    tri = np.triu_indices(p, k=1)
    raw_scores = raw_docking_geometry[tri]
    centered_scores = centered_docking_geometry[tri]
    positive_count = max(1, int(np.ceil(fraction * len(raw_scores))))
    if cluster_labels is None:
        cluster_index = None
        cluster_count = None
    else:
        labels = np.asarray(cluster_labels)
        if labels.shape != (len(x),):
            raise ValueError("retrieval-bootstrap cluster labels do not align to ligands")
        _, cluster_index = np.unique(labels, return_inverse=True)
        cluster_count = int(cluster_index.max() + 1)
    metrics = {
        representation: {
            metric: np.empty(repeats, dtype=np.float64)
            for metric in ("roc_auc", "average_precision")
        }
        for representation in ("raw_docking", "centered_docking", "centered_minus_raw")
    }
    rng = np.random.default_rng(seed)
    for repetition in range(repeats):
        if cluster_index is None:
            weights = rng.exponential(size=len(x))
        else:
            weights = rng.exponential(size=cluster_count)[cluster_index]
        centered_experimental = weighted_two_way_center(x, weights)
        experimental_geometry = weighted_target_correlation(
            centered_experimental, weights
        )
        experimental_scores = experimental_geometry[tri]
        endpoint = np.zeros(len(experimental_scores), dtype=int)
        endpoint[
            np.argsort(experimental_scores, kind="mergesort")[-positive_count:]
        ] = 1
        raw = _retrieval_metrics(endpoint, raw_scores)
        centered = _retrieval_metrics(endpoint, centered_scores)
        for metric in ("roc_auc", "average_precision"):
            metrics["raw_docking"][metric][repetition] = raw[metric]
            metrics["centered_docking"][metric][repetition] = centered[metric]
            metrics["centered_minus_raw"][metric][repetition] = (
                centered[metric] - raw[metric]
            )
    result: dict[str, dict] = {}
    for representation, values in metrics.items():
        result[representation] = {}
        for metric, draws in values.items():
            result[representation][metric] = {
                **describe(draws),
                "probability_positive": float(np.mean(draws > 0))
                if representation == "centered_minus_raw"
                else None,
            }
    return {
        "method": "Bayesian bootstrap with iid Exp(1) weights",
        "unit": "ligand" if cluster_labels is None else "chemical cluster",
        "clusters": cluster_count,
        "repeats": int(repeats),
        "seed": int(seed),
        "fixed_target_panel": int(p),
        "positive_target_pairs_per_draw": int(positive_count),
        "endpoint_recomputed_each_draw": True,
        "docking_predictors_fixed": True,
        "metrics": result,
    }


def target_pair_retrieval(
    docking_matrix: np.ndarray,
    experimental_matrix: np.ndarray,
    fractions: tuple[float, ...],
    qap_permutations: int,
    seed: int,
    bootstrap_repeats: int = 0,
    smiles: pd.Series | None = None,
) -> dict:
    """Retrieve experimentally co-selective target pairs from docking geometry.

    The endpoint is held fixed: positives are target pairs in the upper tail of the
    *two-way-centered experimental* target correlations.  Raw and centered docking
    correlations are alternative predictors of the same labels.  Target-label QAP
    provides inference for the primary 10% endpoint and its paired improvement.
    """
    if 0.10 not in fractions:
        raise ValueError("target-pair retrieval requires a prespecified 10% endpoint")
    raw_geometry = geometry_correlation(docking_matrix, "raw")
    centered_geometry = geometry_correlation(
        docking_matrix, "center_then_correlation"
    )
    experimental_geometry = geometry_correlation(
        experimental_matrix, "center_then_correlation"
    )
    p = len(raw_geometry)
    tri = np.triu_indices(p, k=1)
    raw_scores = raw_geometry[tri]
    centered_scores = centered_geometry[tri]
    experimental_scores = experimental_geometry[tri]
    point_estimates: dict[str, dict] = {}
    primary_labels: np.ndarray | None = None
    for fraction in fractions:
        positive_count = max(1, int(np.ceil(fraction * len(experimental_scores))))
        labels = np.zeros(len(experimental_scores), dtype=int)
        labels[np.argsort(experimental_scores, kind="mergesort")[-positive_count:]] = 1
        key = f"top_{int(round(100 * fraction))}_percent"
        point_estimates[key] = {
            "positive_target_pairs": int(positive_count),
            "total_target_pairs": int(len(labels)),
            "prevalence": float(labels.mean()),
            "raw_docking": _retrieval_metrics(labels, raw_scores),
            "centered_docking": _retrieval_metrics(labels, centered_scores),
        }
        point_estimates[key]["centered_minus_raw"] = {
            metric: float(
                point_estimates[key]["centered_docking"][metric]
                - point_estimates[key]["raw_docking"][metric]
            )
            for metric in ("roc_auc", "average_precision")
        }
        if fraction == 0.10:
            primary_labels = labels
    if primary_labels is None:  # Defensive; checked above.
        raise RuntimeError("primary target-pair labels were not constructed")
    observed_raw = _retrieval_metrics(primary_labels, raw_scores)
    observed_centered = _retrieval_metrics(primary_labels, centered_scores)
    observed_delta = {
        metric: observed_centered[metric] - observed_raw[metric]
        for metric in ("roc_auc", "average_precision")
    }
    rng = np.random.default_rng(seed)
    null_raw = {metric: np.empty(qap_permutations) for metric in observed_delta}
    null_centered = {metric: np.empty(qap_permutations) for metric in observed_delta}
    null_delta = {metric: np.empty(qap_permutations) for metric in observed_delta}
    for repetition in range(qap_permutations):
        order = rng.permutation(p)
        raw_permuted = raw_geometry[np.ix_(order, order)][tri]
        centered_permuted = centered_geometry[np.ix_(order, order)][tri]
        raw_metrics = _retrieval_metrics(primary_labels, raw_permuted)
        centered_metrics = _retrieval_metrics(primary_labels, centered_permuted)
        for metric in observed_delta:
            null_raw[metric][repetition] = raw_metrics[metric]
            null_centered[metric][repetition] = centered_metrics[metric]
            null_delta[metric][repetition] = (
                centered_metrics[metric] - raw_metrics[metric]
            )
    inference: dict[str, dict] = {}
    for metric in observed_delta:
        inference[metric] = {
            "raw_observed": float(observed_raw[metric]),
            "raw_qap_p_greater_than_null": float(
                (1 + np.sum(null_raw[metric] >= observed_raw[metric]))
                / (qap_permutations + 1)
            ),
            "centered_observed": float(observed_centered[metric]),
            "centered_qap_p_greater_than_null": float(
                (1 + np.sum(null_centered[metric] >= observed_centered[metric]))
                / (qap_permutations + 1)
            ),
            "centered_minus_raw": float(observed_delta[metric]),
            "paired_qap_p_positive_delta": float(
                (1 + np.sum(null_delta[metric] >= observed_delta[metric]))
                / (qap_permutations + 1)
            ),
            "paired_delta_null_interval_95": [
                float(np.quantile(null_delta[metric], 0.025)),
                float(np.quantile(null_delta[metric], 0.975)),
            ],
        }
    support_uncertainty = None
    if bootstrap_repeats:
        murcko = davis._murcko_labels(smiles) if smiles is not None else None
        butina = davis._butina_labels(smiles) if smiles is not None else None
        support_uncertainty = {
            "bayesian_ligand_bootstrap": retrieval_support_bootstrap(
                raw_geometry,
                centered_geometry,
                experimental_matrix,
                0.10,
                bootstrap_repeats,
                seed + 10_000,
            )
        }
        if murcko is not None and butina is not None:
            support_uncertainty.update(
                {
                    "bayesian_murcko_cluster_bootstrap": retrieval_support_bootstrap(
                        raw_geometry,
                        centered_geometry,
                        experimental_matrix,
                        0.10,
                        bootstrap_repeats,
                        seed + 20_000,
                        murcko,
                    ),
                    "bayesian_butina_cluster_bootstrap": retrieval_support_bootstrap(
                        raw_geometry,
                        centered_geometry,
                        experimental_matrix,
                        0.10,
                        bootstrap_repeats,
                        seed + 30_000,
                        butina,
                    ),
                }
            )
    return {
        "endpoint": (
            "upper-tail two-way-centered experimental target correlations; raw "
            "and centered docking predict the same fixed labels"
        ),
        "point_estimates": point_estimates,
        "primary_top_10_percent_target_label_qap": {
            "permutations": int(qap_permutations),
            "seed": int(seed),
            "metrics": inference,
        },
        "primary_top_10_percent_support_uncertainty": support_uncertainty,
    }


def run_analysis(
    *,
    qap_permutations: int,
    bootstrap_repeats: int,
    projection_null_repeats: int,
    projection_reference_size: int,
    reference_support_size: int,
    reference_support_seeds: tuple[int, ...],
    seed: int,
    include_chembl: bool,
    pkis1_zip: Path | None,
    include_scaffold_exclusion: bool,
) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # One full identity scan supplies the complete DOCKSTRING reference and DAVIS.
    dockstring, davis_identity, davis_experiment, davis_censored = davis._load_inputs(
        davis.DEFAULT_DOCKSTRING, davis.DEFAULT_DAVIS, identity_scan="full"
    )
    pkis_frame = load_pkis2_full()
    pkis_experiment = pkis_frame[list(TARGETS)].copy()

    pkis1_frame = load_pkis1_full(pkis1_zip) if pkis1_zip is not None else None
    pkis1_targets = tuple(PKIS1_TARGET_MAP)
    pkis1_indices = tuple(TARGETS.index(target) for target in pkis1_targets)
    pkis1_experiment = (
        pkis1_frame[list(pkis1_targets)].copy()
        if pkis1_frame is not None
        else None
    )

    excluded_blocks = set(davis_identity.connectivity_block.dropna()) | set(
        pkis_frame.connectivity_block.dropna()
    )
    if pkis1_frame is not None:
        excluded_blocks |= set(pkis1_frame.connectivity_block.dropna())
    reference_keep = ~dockstring.connectivity_block.isin(excluded_blocks)
    reference_frame = dockstring.loc[reference_keep].copy()
    reference = np.minimum(
        reference_frame[list(TARGETS)].to_numpy(dtype=np.float64), 0.0
    )
    if len(reference) < 259000:
        raise ValueError("unexpectedly many DOCKSTRING rows were excluded")

    davis_smiles = (
        davis_identity.set_index("drug_name")
        .loc[davis_experiment.index, "compound_iso_smiles"]
        .reset_index(drop=True)
    )
    pkis_smiles = pkis_frame["Smiles"].reset_index(drop=True)
    pkis1_smiles = (
        pkis1_frame["SMILES"].reset_index(drop=True)
        if pkis1_frame is not None
        else None
    )

    rng = np.random.default_rng(seed + 50000)
    if projection_reference_size < 100 or projection_reference_size > len(reference):
        raise ValueError("invalid projection-null reference size")
    projection_indices = np.sort(
        rng.choice(len(reference), projection_reference_size, replace=False)
    )
    projection_reference = reference[projection_indices]

    geometry: dict[str, dict] = {}
    geometry_csv: list[dict] = []
    jackknife_csv: list[dict] = []
    panel_specs: list[
        tuple[str, tuple[str, ...], tuple[int, ...], np.ndarray, pd.Series]
    ] = [
        (
            "DAVIS",
            TARGETS,
            tuple(range(len(TARGETS))),
            davis_experiment.to_numpy(dtype=float),
            davis_smiles,
        ),
        (
            "PKIS2",
            TARGETS,
            tuple(range(len(TARGETS))),
            pkis_experiment.to_numpy(dtype=float),
            pkis_smiles,
        ),
    ]
    if pkis1_experiment is not None and pkis1_smiles is not None:
        panel_specs.append(
            (
                "PKIS1",
                pkis1_targets,
                pkis1_indices,
                pkis1_experiment.to_numpy(dtype=float),
                pkis1_smiles,
            )
        )
    for index, (dataset, target_names, target_indices, experimental, smiles) in enumerate(
        panel_specs
    ):
        records, rows, jackknife = _geometry_rows(
            dataset,
            "full_experimental_panel",
            target_names,
            reference[:, target_indices],
            experimental,
            qap_permutations,
            bootstrap_repeats,
            seed + 100000 * index,
            smiles,
            projection_reference[:, target_indices],
            projection_null_repeats,
        )
        geometry.setdefault(dataset, {})["full_experimental_panel"] = records
        geometry_csv.extend(rows)
        jackknife_csv.extend(jackknife)

    # The only multiplicity family designated as primary is the centered-geometry
    # QAP across the three full external panels.  Transform variants, target subsets,
    # and thresholded retrieval endpoints remain explicitly secondary sensitivities.
    primary_names = [dataset for dataset, _, _, _, _ in panel_specs]
    primary_p = [
        geometry[dataset]["full_experimental_panel"]["center_then_correlation"][
            "qap"
        ]["one_sided_p_positive"]
        for dataset in primary_names
    ]
    primary_holm = holm_adjust(primary_p)
    primary_geometry_multiplicity = {
        "family": (
            "full-panel centered target-geometry QAP tests only; all transforms, "
            "support restrictions, and retrieval thresholds are sensitivities"
        ),
        "method": "Holm family-wise-error adjustment",
        "number_of_tests": int(len(primary_names)),
        "tests": {},
    }
    for dataset, nominal, adjusted in zip(
        primary_names, primary_p, primary_holm, strict=True
    ):
        qap = geometry[dataset]["full_experimental_panel"][
            "center_then_correlation"
        ]["qap"]
        qap["holm_adjusted_p_across_primary_panels"] = float(adjusted)
        primary_geometry_multiplicity["tests"][dataset] = {
            "nominal_p": float(nominal),
            "holm_adjusted_p": float(adjusted),
        }
        for row in geometry_csv:
            if (
                row["dataset"] == dataset
                and row["support"] == "full_experimental_panel"
                and row["transform"] == "center_then_correlation"
            ):
                row["primary_holm_adjusted_p"] = float(adjusted)

    # DAVIS censoring/target-quality checks.  These are target-panel sensitivities,
    # not replacements for the complete 72 x 21 primary experimental surface.
    target_quality = davis._target_quality_table(davis_experiment, davis_censored)
    for threshold_index, threshold in enumerate((5, 10)):
        selected_targets = tuple(
            target_quality.loc[
                target_quality.uncensored_cells >= threshold, "target"
            ].tolist()
        )
        target_indices = tuple(TARGETS.index(target) for target in selected_targets)
        support = f"targets_with_at_least_{threshold}_uncensored_DAVIS_ligands"
        records, rows, jackknife = _geometry_rows(
            "DAVIS",
            support,
            selected_targets,
            reference[:, target_indices],
            davis_experiment[list(selected_targets)].to_numpy(dtype=float),
            qap_permutations,
            bootstrap_repeats,
            seed + 900_000 + threshold_index * 100_000,
            None,
            None,
            projection_null_repeats,
        )
        geometry["DAVIS"][support] = records
        geometry_csv.extend(rows)
        jackknife_csv.extend(jackknife)

    binary_davis = (~davis_censored).astype(float).to_numpy(dtype=float)
    records, rows, jackknife = _geometry_rows(
        "DAVIS",
        "binary_uncensored_activity_surface",
        TARGETS,
        reference,
        binary_davis,
        qap_permutations,
        bootstrap_repeats,
        seed + 1_100_000,
        None,
        None,
        projection_null_repeats,
    )
    geometry["DAVIS"]["binary_uncensored_activity_surface"] = records
    geometry_csv.extend(rows)
    jackknife_csv.extend(jackknife)

    if pkis1_frame is not None and pkis1_zip is not None:
        pkis1_cycline = load_pkis1_full(pkis1_zip, cdk2_construct="CDK2/cyclinE")
        records, rows, jackknife = _geometry_rows(
            "PKIS1",
            "CDK2_cyclinE_construct_sensitivity",
            pkis1_targets,
            reference[:, pkis1_indices],
            pkis1_cycline[list(pkis1_targets)].to_numpy(dtype=float),
            qap_permutations,
            bootstrap_repeats,
            seed + 1_200_000,
            None,
            None,
            projection_null_repeats,
        )
        geometry["PKIS1"]["CDK2_cyclinE_construct_sensitivity"] = records
        geometry_csv.extend(rows)
        jackknife_csv.extend(jackknife)

    experimental_cross_panel = experimental_cross_panel_concordance(
        [
            (dataset, target_names, experimental)
            for dataset, target_names, _, experimental, _ in panel_specs
        ],
        qap_permutations,
        seed + 1_250_000,
    )
    experimental_cross_panel.update(
        experimental_cross_panel_concordance(
            [
                ("DAVIS_continuous_pKd", TARGETS, davis_experiment.to_numpy(float)),
                ("DAVIS_binary_uncensored", TARGETS, binary_davis),
            ],
            qap_permutations,
            seed + 1_275_000,
        )
    )

    scaffold_reference_rows_excluded = None
    if include_scaffold_exclusion:
        experimental_scaffolds = set(
            key
            for key in np.concatenate(
                [
                    murcko_scaffold_keys(davis_smiles),
                    murcko_scaffold_keys(pkis_smiles),
                    *(
                        [murcko_scaffold_keys(pkis1_smiles)]
                        if pkis1_smiles is not None
                        else []
                    ),
                ]
            )
            if key
        )
        reference_scaffolds = murcko_scaffold_keys(reference_frame["smiles"])
        scaffold_keep = ~np.isin(reference_scaffolds, list(experimental_scaffolds))
        scaffold_reference = reference[scaffold_keep]
        scaffold_reference_rows_excluded = int((~scaffold_keep).sum())
        if len(scaffold_reference) < 200_000:
            raise ValueError("same-scaffold exclusion unexpectedly removed most reference rows")
        for index, (
            dataset,
            target_names,
            target_indices,
            experimental,
            _,
        ) in enumerate(panel_specs):
            support = "exact_identity_and_same_Murcko_scaffold_excluded_reference"
            records, rows, jackknife = _geometry_rows(
                dataset,
                support,
                target_names,
                scaffold_reference[:, target_indices],
                experimental,
                qap_permutations,
                bootstrap_repeats,
                seed + 1_300_000 + index * 100_000,
                None,
                None,
                projection_null_repeats,
            )
            geometry[dataset][support] = records
            geometry_csv.extend(rows)
            jackknife_csv.extend(jackknife)

    reference_support_summary, reference_support_frame = reference_support_sensitivity(
        reference,
        [
            (dataset, target_indices, experimental)
            for dataset, _, target_indices, experimental, _ in panel_specs
        ],
        reference_support_size,
        reference_support_seeds,
    )
    pair_retrieval = {
        dataset: target_pair_retrieval(
            reference[:, target_indices],
            experimental,
            fractions=(0.05, 0.10, 0.20),
            qap_permutations=qap_permutations,
            seed=seed + 1_600_000 + index * 100_000,
            bootstrap_repeats=bootstrap_repeats,
            smiles=smiles,
        )
        for index, (dataset, _, target_indices, experimental, smiles) in enumerate(
            panel_specs
        )
    }

    # Molecule-matched sensitivities use the same leakage-excluded reference geometry.
    matched_dock_davis, matched_exp_davis, matched_censor_davis, matched_smiles_davis = (
        davis._matched_surface(
            dockstring,
            davis_identity,
            davis_experiment,
            davis_censored,
            "standard_inchikey",
            "median",
        )
    )
    matched_dock_pkis, matched_exp_pkis, matched_smiles_pkis = pkis2._matched_surface(
        dockstring, pkis_frame, "standard_inchikey"
    )
    # Geometry is estimated on the full reference; matched experimental ligand supports
    # change only the experimental geometry.  This avoids evaluating the trivial
    # correlation of docking and experiment on the same small ligand sample.
    for index, (dataset, experimental, smiles) in enumerate(
        (
            (
                "DAVIS",
                matched_exp_davis.to_numpy(dtype=float),
                matched_smiles_davis.smiles,
            ),
            (
                "PKIS2",
                matched_exp_pkis.to_numpy(dtype=float),
                matched_smiles_pkis.smiles,
            ),
        )
    ):
        records, rows, jackknife = _geometry_rows(
            dataset,
            "molecule_matched_experimental_sensitivity",
            TARGETS,
            reference,
            experimental,
            qap_permutations,
            bootstrap_repeats,
            seed + 300000 + 100000 * index,
            smiles.reset_index(drop=True),
            None,
            projection_null_repeats,
        )
        geometry[dataset]["molecule_matched_experimental_sensitivity"] = records
        geometry_csv.extend(rows)
        jackknife_csv.extend(jackknife)

    fit = fit_molecular_weight_component(
        reference,
        reference_frame.molecular_weight.to_numpy(dtype=np.float64),
    )
    correction: dict[str, dict] = {}
    correction_rows: list[dict] = []
    correction_jackknife: list[dict] = []
    correction["DAVIS"], rows = slope_correction_panel(
        "DAVIS",
        matched_dock_davis,
        matched_exp_davis,
        matched_smiles_davis.smiles,
        fit,
        bootstrap_repeats,
        seed + 600000,
        matched_censor_davis,
    )
    correction_jackknife.extend(rows)
    correction["PKIS2"], rows = slope_correction_panel(
        "PKIS2",
        matched_dock_pkis,
        matched_exp_pkis,
        matched_smiles_pkis.smiles,
        fit,
        bootstrap_repeats,
        seed + 700000,
    )
    correction_jackknife.extend(rows)
    for dataset, record in correction.items():
        for contrast, values in record["contrasts"].items():
            murcko = values["paired_uncertainty"]["murcko_cluster_bootstrap"]
            butina = values["paired_uncertainty"]["butina_cluster_bootstrap"]
            correction_rows.append(
                {
                    "dataset": dataset,
                    "contrast": contrast,
                    "after_minus_before": values["after_minus_before_plugin"],
                    "murcko_low": murcko["interval_95"][0],
                    "murcko_high": murcko["interval_95"][1],
                    "murcko_clusters": murcko["n_clusters"],
                    "butina_low": butina["interval_95"][0],
                    "butina_high": butina["interval_95"][1],
                    "butina_clusters": butina["n_clusters"],
                    "target_loo_min": values["target_delete_one"]["minimum"],
                    "target_loo_median": values["target_delete_one"]["median"],
                    "target_loo_max": values["target_delete_one"]["maximum"],
                }
            )

    fit_json = {
        key: (
            [float(value) for value in np.asarray(item)]
            if isinstance(item, np.ndarray)
            else item
        )
        for key, item in fit.items()
    }
    report: dict = {
        "analysis": "residual target-geometry external validation",
        "status": "exploratory extension; manuscript unchanged",
        "seed": int(seed),
        "targets": list(TARGETS),
        "sources": {
            "dockstring": str(davis.DEFAULT_DOCKSTRING.relative_to(PACKAGE)),
            "dockstring_sha256": sha256_file(davis.DEFAULT_DOCKSTRING),
            "davis": str(davis.DEFAULT_DAVIS.relative_to(PACKAGE)),
            "davis_sha256": sha256_file(davis.DEFAULT_DAVIS),
            "pkis2": str(pkis2.DEFAULT_PKIS2.relative_to(PACKAGE)),
            "pkis2_sha256": sha256_file(pkis2.DEFAULT_PKIS2),
        },
        "configuration": {
            "qap_permutations": int(qap_permutations),
            "bootstrap_repeats": int(bootstrap_repeats),
            "projection_null_repeats": int(projection_null_repeats),
            "projection_reference_size": int(projection_reference_size),
            "reference_support_sensitivity_size": int(reference_support_size),
            "reference_support_sensitivity_seeds": [
                int(value) for value in reference_support_seeds
            ],
            "positive_score_handling": "clip to zero before all DOCKSTRING analyses",
            "reference_exclusion": (
                "union of all included experimental-panel Standard-InChI connectivity blocks"
            ),
            "same_murcko_scaffold_exclusion_sensitivity": bool(
                include_scaffold_exclusion
            ),
            "geometry_concordance": (
                "Spearman correlation of strict-upper-triangle target correlations "
                "(210 entries for 21-target panels; 190 for PKIS1)"
            ),
        },
        "support": {
            "dockstring_complete_rows_before_exclusion": int(len(dockstring)),
            "excluded_connectivity_blocks": int(len(excluded_blocks)),
            "dockstring_rows_excluded": int((~reference_keep).sum()),
            "dockstring_reference_rows": int(len(reference)),
            "davis_full_shape": list(davis_experiment.shape),
            "pkis2_full_shape": list(pkis_experiment.shape),
            "davis_matched_shape": list(matched_exp_davis.shape),
            "pkis2_matched_shape": list(matched_exp_pkis.shape),
            "matched_full_inchikey_overlap_between_davis_and_pkis2": int(
                len(set(matched_exp_davis.index) & set(matched_exp_pkis.index))
            ),
            "davis_floor_cells_pkd5": int(davis_censored.to_numpy().sum()),
            "davis_uncensored_target_counts": {
                row.target: int(row.uncensored_cells)
                for row in target_quality.itertuples()
            },
            "same_murcko_scaffold_reference_rows_excluded": scaffold_reference_rows_excluded,
        },
        "geometry": geometry,
        "primary_centered_geometry_multiplicity": primary_geometry_multiplicity,
        "experimental_cross_panel_geometry": experimental_cross_panel,
        "co_selective_target_pair_retrieval": pair_retrieval,
        "independent_reference_support_sensitivity": reference_support_summary,
        "molecular_weight_component_fit": fit_json,
        "molecular_weight_slope_correction": correction,
    }
    if pkis1_frame is not None and pkis1_zip is not None:
        report["sources"]["pkis1"] = {
            "doi": PKIS1_DOI,
            "url": PKIS1_URL,
            "local_input_filename": pkis1_zip.name,
            "sha256": sha256_file(pkis1_zip),
            "heatmap_member": PKIS1_HEATMAP_MEMBER,
            "smiles_member": PKIS1_SMILES_MEMBER,
            "primary_cdk2_construct": "CDK2/cyclinA",
            "sensitivity_cdk2_construct": "CDK2/cyclinE",
            "redistribution": "source archive is not copied into this package",
        }
        report["support"]["pkis1_full_shape"] = list(pkis1_experiment.shape)
    if include_chembl:
        report["chembl_74x6_negative_sensitivity"] = chembl_negative_sensitivity(
            bootstrap_repeats, seed + 800000
        )
    jackknife_csv.extend(correction_jackknife)
    return (
        report,
        pd.DataFrame(geometry_csv),
        pd.DataFrame(correction_rows),
        pd.DataFrame(jackknife_csv),
        reference_support_frame,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--qap-permutations", type=int, default=20_000)
    parser.add_argument("--bootstrap-repeats", type=int, default=5_000)
    parser.add_argument("--projection-null-repeats", type=int, default=500)
    parser.add_argument("--projection-reference-size", type=int, default=15_000)
    parser.add_argument("--reference-support-size", type=int, default=15_000)
    parser.add_argument(
        "--reference-support-seeds",
        default="11,29,47,71,97",
        help="comma-separated seeds for independent score-blind reference supports",
    )
    parser.add_argument(
        "--pkis1-zip",
        type=Path,
        default=None,
        help="optional original PKIS1 supplementary ZIP (validated by SHA256)",
    )
    parser.add_argument(
        "--skip-scaffold-exclusion",
        action="store_true",
        help="omit the slower same-Murcko-scaffold exclusion sensitivity",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--skip-chembl",
        action="store_true",
        help="omit the slower matched ChEMBL negative sensitivity",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    support_seeds = tuple(
        int(value.strip())
        for value in args.reference_support_seeds.split(",")
        if value.strip()
    )
    if not support_seeds:
        raise ValueError("at least one reference-support seed is required")
    report, geometry, correction, jackknife, reference_support = run_analysis(
        qap_permutations=args.qap_permutations,
        bootstrap_repeats=args.bootstrap_repeats,
        projection_null_repeats=args.projection_null_repeats,
        projection_reference_size=args.projection_reference_size,
        reference_support_size=args.reference_support_size,
        reference_support_seeds=support_seeds,
        seed=args.seed,
        include_chembl=not args.skip_chembl,
        pkis1_zip=args.pkis1_zip,
        include_scaffold_exclusion=not args.skip_scaffold_exclusion,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    geometry.to_csv(args.output_dir / "geometry_concordance.csv", index=False)
    correction.to_csv(args.output_dir / "mw_slope_correction.csv", index=False)
    jackknife.to_csv(args.output_dir / "target_jackknife.csv", index=False)
    reference_support.to_csv(
        args.output_dir / "reference_support_sensitivity.csv", index=False
    )
    print(f"Wrote {args.output_dir / 'summary.json'}")
    for dataset in report["geometry"]:
        record = report["geometry"][dataset]["full_experimental_panel"]
        print(
            f"{dataset}: raw rho={record['raw']['concordance_spearman']:.3f}; "
            "centered rho="
            f"{record['center_then_correlation']['concordance_spearman']:.3f}"
        )


if __name__ == "__main__":
    main()
