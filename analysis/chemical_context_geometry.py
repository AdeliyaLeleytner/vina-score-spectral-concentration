#!/usr/bin/env python3
"""Chemical-domain dependence of Vina target-correlation geometry.

This science-only, explicitly post-hoc analysis asks whether the target--target
relationship network inferred from docking scores is stable across ligand
chemical space.  Molecular weight is the primary stratifier because preliminary
inspection suggested a size-dependent effect; all inferential quantities in this
file are therefore exploratory rather than prospective confirmation.

The primary comparison is the Spearman agreement between target-correlation
matrices in groups at the inclusive empirical 25th- and 75th-percentile molecular-weight
thresholds.  A random,
disjoint-support null asks how much disagreement is expected from finite ligand
sampling alone.  A chemical-group bootstrap resamples whole frozen Butina groups
(Docking-44) or Bemis--Murcko/scaffold groups (DOCKSTRING).  Seven descriptor
stratifiers, raw versus row-centered surfaces, molecular-weight quintiles, and a
strict group-held-out linear descriptor-removal analysis are sensitivities.

No manuscript files are read or modified.  Experimental target-pair retrieval by
molecular-weight bin is a descriptive reuse of a previously frozen endpoint and
is not used to select a stratifier, transformation, or threshold.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from scipy import stats

try:  # Direct script execution and package-style import are both supported.
    from .replicated_pair_retrieval import retrieval_metrics
    from .residual_mechanism_analysis import (
        DESCRIPTOR_NAMES,
        AnalysisConfig,
        DatasetBundle,
        grouped_descriptor_decomposition,
        load_docking44,
        load_dockstring,
    )
    from .residual_target_geometry_validation import PKIS1_TARGET_MAP
except ImportError:  # pragma: no cover - direct CLI execution.
    from replicated_pair_retrieval import retrieval_metrics  # type: ignore
    from residual_mechanism_analysis import (  # type: ignore
        DESCRIPTOR_NAMES,
        AnalysisConfig,
        DatasetBundle,
        grouped_descriptor_decomposition,
        load_docking44,
        load_dockstring,
    )
    from residual_target_geometry_validation import PKIS1_TARGET_MAP  # type: ignore


PACKAGE = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PACKAGE / "results" / "chemical_context_geometry"
DEFAULT_ENDPOINT = PACKAGE / "results" / "replicated_pair_retrieval" / "target_pairs.csv"
DEFAULT_RANDOM_REPETITIONS = 500
DEFAULT_GROUP_BOOTSTRAPS = 300
DEFAULT_GROUP_DISJOINT_REPETITIONS = 200
DEFAULT_SEED = 202_608_09
DEFAULT_DOCKSTRING_SUPPORT_SEEDS = (71, 72, 73, 74, 75, DEFAULT_SEED)
PRIMARY_DESCRIPTOR = "molecular_weight"
PRIMARY_EXTREME_FRACTION = 0.25
QUINTILE_COUNT = 5
TOP_PAIR_FRACTION = 0.10


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
        return json_ready(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float) and not np.isfinite(value):
        raise ValueError("non-finite value cannot be serialized to strict JSON")
    return value


def dataframe_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert a frame to JSON-ready records while preserving explicit nulls."""
    clean = frame.astype(object).where(pd.notna(frame), None)
    return clean.to_dict(orient="records")


def row_center(matrix: np.ndarray) -> np.ndarray:
    """Remove each ligand's mean across targets.

    Subsequent target correlation removes column means, so this produces exactly
    the same target-correlation matrix as explicit two-way centering.
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2 or min(matrix.shape) < 2 or not np.isfinite(matrix).all():
        raise ValueError("row centering requires a finite two-dimensional matrix")
    return matrix - matrix.mean(axis=1, keepdims=True)


def target_correlation(
    matrix: np.ndarray, weights: np.ndarray | None = None
) -> np.ndarray:
    """Target correlation, optionally treating integer weights as row repeats."""
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] < 3 or matrix.shape[1] < 2:
        raise ValueError("target correlation requires at least 3 rows and 2 targets")
    if not np.isfinite(matrix).all():
        raise ValueError("target-correlation input contains non-finite values")
    if weights is None:
        centered = matrix - matrix.mean(axis=0, keepdims=True)
        covariance = centered.T @ centered
    else:
        weights = np.asarray(weights, dtype=np.float64)
        if weights.shape != (len(matrix),) or np.any(weights < 0):
            raise ValueError("weights must be nonnegative and match matrix rows")
        total = float(weights.sum())
        if total <= 2:
            raise ValueError("weighted target correlation has fewer than 3 rows")
        weighted_sum = weights @ matrix
        covariance = matrix.T @ (matrix * weights[:, None])
        covariance -= np.outer(weighted_sum, weighted_sum) / total
    variance = np.diag(covariance)
    if np.any(variance <= 1e-14) or not np.isfinite(variance).all():
        raise ValueError("target correlation contains a constant target")
    scale = np.sqrt(variance)
    correlation = covariance / np.outer(scale, scale)
    correlation = np.clip((correlation + correlation.T) / 2.0, -1.0, 1.0)
    np.fill_diagonal(correlation, 1.0)
    return correlation


def upper_triangle(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("upper triangle requires a square matrix")
    return matrix[np.triu_indices(len(matrix), k=1)]


def correlation_pr(correlation: np.ndarray) -> float:
    correlation = np.asarray(correlation, dtype=np.float64)
    return float(np.trace(correlation) ** 2 / np.square(correlation).sum())


def strongest_pair_mask(values: np.ndarray, fraction: float = TOP_PAIR_FRACTION) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if not 0 < fraction < 1 or values.ndim != 1:
        raise ValueError("top-pair fraction requires a one-dimensional vector")
    count = max(1, int(np.ceil(fraction * len(values))))
    mask = np.zeros(len(values), dtype=bool)
    # Stable sorting makes tie handling deterministic.
    mask[np.argsort(values, kind="mergesort")[-count:]] = True
    return mask


def geometry_comparison(first: np.ndarray, second: np.ndarray) -> dict[str, float]:
    first_values = upper_triangle(first)
    second_values = upper_triangle(second)
    rho = float(stats.spearmanr(first_values, second_values).statistic)
    if not np.isfinite(rho):
        raise ValueError("geometry agreement is not finite")
    first_top = strongest_pair_mask(first_values)
    second_top = strongest_pair_mask(second_values)
    union = int(np.sum(first_top | second_top))
    return {
        "geometry_spearman": rho,
        "geometry_dissimilarity": 1.0 - rho,
        "sign_flip_fraction": float(
            np.mean(np.sign(first_values) != np.sign(second_values))
        ),
        "top_positive_10pct_pair_jaccard": float(
            np.sum(first_top & second_top) / union
        ),
        "first_pr": correlation_pr(first),
        "second_pr": correlation_pr(second),
        "first_mean_signed_correlation": float(first_values.mean()),
        "second_mean_signed_correlation": float(second_values.mean()),
        "first_mean_absolute_correlation": float(np.abs(first_values).mean()),
        "second_mean_absolute_correlation": float(np.abs(second_values).mean()),
    }


def extreme_masks(
    descriptor: np.ndarray, fraction: float = PRIMARY_EXTREME_FRACTION
) -> tuple[np.ndarray, np.ndarray, float, float]:
    descriptor = np.asarray(descriptor, dtype=np.float64)
    if descriptor.ndim != 1 or not np.isfinite(descriptor).all():
        raise ValueError("descriptor must be a finite vector")
    if not 0 < fraction < 0.5:
        raise ValueError("extreme fraction must lie in (0, 0.5)")
    low_threshold, high_threshold = np.quantile(
        descriptor, [fraction, 1.0 - fraction]
    )
    if low_threshold >= high_threshold:
        raise ValueError("descriptor quantiles do not define disjoint extremes")
    low = descriptor <= low_threshold
    high = descriptor >= high_threshold
    if np.any(low & high) or min(int(low.sum()), int(high.sum())) < 3:
        raise ValueError("descriptor extremes are invalid")
    return low, high, float(low_threshold), float(high_threshold)


def quantile_bin_ids(descriptor: np.ndarray, bins: int = QUINTILE_COUNT) -> tuple[np.ndarray, np.ndarray]:
    descriptor = np.asarray(descriptor, dtype=np.float64)
    if bins < 2 or descriptor.ndim != 1 or not np.isfinite(descriptor).all():
        raise ValueError("quantile binning requires a finite vector and >=2 bins")
    edges = np.quantile(descriptor, np.linspace(0.0, 1.0, bins + 1))
    if np.any(np.diff(edges) <= 0):
        raise ValueError("descriptor has tied quantiles and cannot define requested bins")
    identifiers = np.searchsorted(edges[1:-1], descriptor, side="left")
    if set(np.unique(identifiers)) != set(range(bins)):
        raise ValueError("not every requested quantile bin is represented")
    return identifiers.astype(np.int16), edges


def surface_variants(
    bundle: DatasetBundle, folds: int
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Return raw, row-centered, and strict OOF descriptor-adjusted surfaces."""
    decomposition = grouped_descriptor_decomposition(
        bundle.matrix, bundle.descriptor_matrix, bundle.groups, folds
    )
    surfaces = {
        "raw": np.asarray(bundle.matrix, dtype=np.float64),
        "row_centered_residual": row_center(bundle.matrix),
        "group_heldout_descriptor_adjusted_residual": np.asarray(
            decomposition["oof_error"], dtype=np.float64
        ),
    }
    return surfaces, {
        "descriptor_decomposition_metrics": decomposition["metrics"],
        "fold_diagnostics": decomposition["fold_rows"],
    }


def descriptor_extreme_analysis(
    bundle: DatasetBundle, surfaces: dict[str, np.ndarray]
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for descriptor_index, descriptor_name in enumerate(DESCRIPTOR_NAMES):
        descriptor = bundle.descriptor_matrix[:, descriptor_index]
        low, high, low_threshold, high_threshold = extreme_masks(descriptor)
        transformations = ["raw", "row_centered_residual"]
        if descriptor_name == PRIMARY_DESCRIPTOR:
            transformations.append("group_heldout_descriptor_adjusted_residual")
        for transformation in transformations:
            first = target_correlation(surfaces[transformation][low])
            second = target_correlation(surfaces[transformation][high])
            records.append(
                {
                    "dataset": bundle.name,
                    "descriptor": descriptor_name,
                    "transformation": transformation,
                    "analysis_status": "exploratory_post_hoc",
                    "extreme_fraction_per_tail": PRIMARY_EXTREME_FRACTION,
                    "low_threshold_inclusive": low_threshold,
                    "high_threshold_inclusive": high_threshold,
                    "low_ligands": int(low.sum()),
                    "high_ligands": int(high.sum()),
                    "low_descriptor_median": float(np.median(descriptor[low])),
                    "high_descriptor_median": float(np.median(descriptor[high])),
                    **geometry_comparison(first, second),
                }
            )
    return pd.DataFrame.from_records(records)


def descriptor_correlation_table(bundle: DatasetBundle) -> pd.DataFrame:
    """Pairwise Spearman correlations among the seven stratifying descriptors."""
    correlation = stats.spearmanr(bundle.descriptor_matrix, axis=0).statistic
    correlation = np.asarray(correlation, dtype=np.float64)
    if correlation.shape != (len(DESCRIPTOR_NAMES), len(DESCRIPTOR_NAMES)):
        raise ValueError("unexpected molecular-descriptor correlation shape")
    records: list[dict[str, Any]] = []
    for first in range(len(DESCRIPTOR_NAMES)):
        for second in range(first + 1, len(DESCRIPTOR_NAMES)):
            records.append(
                {
                    "dataset": bundle.name,
                    "descriptor_a": DESCRIPTOR_NAMES[first],
                    "descriptor_b": DESCRIPTOR_NAMES[second],
                    "spearman": float(correlation[first, second]),
                }
            )
    return pd.DataFrame.from_records(records)


def random_disjoint_support_null(
    bundle: DatasetBundle,
    surfaces: dict[str, np.ndarray],
    repetitions: int,
    seed: int,
) -> pd.DataFrame:
    """Finite-support null using two random, nonoverlapping ligand subsets."""
    if repetitions < 1:
        raise ValueError("at least one random repetition is required")
    descriptor = bundle.descriptor_matrix[:, DESCRIPTOR_NAMES.index(PRIMARY_DESCRIPTOR)]
    low, high, _, _ = extreme_masks(descriptor)
    low_count, high_count = int(low.sum()), int(high.sum())
    if low_count + high_count > len(descriptor):
        raise ValueError("random null cannot construct disjoint supports")
    rng = np.random.default_rng(seed)
    records: list[dict[str, Any]] = []
    for repetition in range(repetitions):
        selected = rng.choice(
            len(descriptor), size=low_count + high_count, replace=False
        )
        first_index = selected[:low_count]
        second_index = selected[low_count:]
        if np.intersect1d(first_index, second_index).size:
            raise AssertionError("random supports overlap")
        for transformation, surface in surfaces.items():
            first = target_correlation(surface[first_index])
            second = target_correlation(surface[second_index])
            records.append(
                {
                    "dataset": bundle.name,
                    "transformation": transformation,
                    "repetition": repetition,
                    "first_ligands": low_count,
                    "second_ligands": high_count,
                    **geometry_comparison(first, second),
                }
            )
    return pd.DataFrame.from_records(records)


def _group_codes(groups: Iterable[object]) -> tuple[np.ndarray, np.ndarray]:
    values = pd.Series([str(value) for value in groups], dtype="string")
    codes, uniques = pd.factorize(values, sort=True)
    if np.any(codes < 0):
        raise ValueError("chemical groups contain missing values")
    return codes.astype(np.int64), np.asarray(uniques.astype(str), dtype=object)


def group_bootstrap_primary(
    bundle: DatasetBundle,
    surfaces: dict[str, np.ndarray],
    repetitions: int,
    seed: int,
) -> pd.DataFrame:
    """Paired cluster bootstrap of the primary low/high-MW comparison.

    Each chemical group is sampled once per draw from the union of groups in
    the two domains.  A selected group's multiplicity is applied to all of its
    molecules in both domains, preserving paired group-level dependence.
    """
    if repetitions < 1:
        raise ValueError("at least one bootstrap repetition is required")
    descriptor = bundle.descriptor_matrix[:, DESCRIPTOR_NAMES.index(PRIMARY_DESCRIPTOR)]
    low, high, _, _ = extreme_masks(descriptor)
    codes, unique_groups = _group_codes(bundle.groups)
    relevant_group_codes = np.unique(codes[low | high])
    if len(relevant_group_codes) < 2:
        raise ValueError("fewer than two chemical groups in MW extremes")
    rng = np.random.default_rng(seed)
    records: list[dict[str, Any]] = []
    transformations = (
        "raw",
        "row_centered_residual",
        "group_heldout_descriptor_adjusted_residual",
    )
    for repetition in range(repetitions):
        sampled = rng.choice(
            relevant_group_codes, size=len(relevant_group_codes), replace=True
        )
        multiplicity = np.bincount(sampled, minlength=len(unique_groups))
        row_weights = multiplicity[codes].astype(np.float64)
        low_weights = row_weights[low]
        high_weights = row_weights[high]
        if low_weights.sum() <= 2 or high_weights.sum() <= 2:
            raise RuntimeError("cluster bootstrap produced an empty MW domain")
        for transformation in transformations:
            surface = surfaces[transformation]
            first = target_correlation(surface[low], low_weights)
            second = target_correlation(surface[high], high_weights)
            records.append(
                {
                    "dataset": bundle.name,
                    "transformation": transformation,
                    "repetition": repetition,
                    "chemical_groups_in_extremes": int(len(relevant_group_codes)),
                    "weighted_low_ligand_rows": int(low_weights.sum()),
                    "weighted_high_ligand_rows": int(high_weights.sum()),
                    **geometry_comparison(first, second),
                }
            )
    return pd.DataFrame.from_records(records)


def balanced_group_split(
    group_sizes: np.ndarray,
    group_descriptor_medians: np.ndarray,
    rng: np.random.Generator,
    strata: int = 10,
) -> np.ndarray:
    """Assign whole groups to two size-balanced halves within descriptor strata."""
    group_sizes = np.asarray(group_sizes, dtype=np.int64)
    group_descriptor_medians = np.asarray(
        group_descriptor_medians, dtype=np.float64
    )
    if (
        group_sizes.ndim != 1
        or group_sizes.shape != group_descriptor_medians.shape
        or np.any(group_sizes < 1)
        or not np.isfinite(group_descriptor_medians).all()
    ):
        raise ValueError("group sizes and descriptor medians are invalid")
    if strata < 2 or len(group_sizes) < 2 * strata:
        raise ValueError("too few groups for requested descriptor strata")
    edges = np.quantile(
        group_descriptor_medians, np.linspace(0.0, 1.0, strata + 1)
    )
    if np.any(np.diff(edges) <= 0):
        raise ValueError("group medians cannot define requested descriptor strata")
    stratum_ids = np.searchsorted(
        edges[1:-1], group_descriptor_medians, side="left"
    )
    assignment = np.full(len(group_sizes), -1, dtype=np.int8)
    for stratum in range(strata):
        members = rng.permutation(np.flatnonzero(stratum_ids == stratum))
        totals = [0, 0]
        tie_side = int(rng.integers(0, 2))
        for group in members:
            if totals[0] < totals[1]:
                side = 0
            elif totals[1] < totals[0]:
                side = 1
            else:
                side = tie_side
                tie_side = 1 - tie_side
            assignment[group] = side
            totals[side] += int(group_sizes[group])
    if np.any(assignment < 0) or not set(assignment) == {0, 1}:
        raise AssertionError("group split did not assign two complete halves")
    return assignment


def mw_matched_group_disjoint_control(
    bundle: DatasetBundle,
    surfaces: dict[str, np.ndarray],
    repetitions: int,
    seed: int,
) -> pd.DataFrame:
    """Network agreement across chemically disjoint, MW-stratified halves.

    Frozen chemical groups are assigned whole to one half.  Assignment is done
    separately within deciles of group-median molecular weight and greedily
    balances ligand-row counts, so the two halves have closely matched MW
    distributions but no shared group labels.
    """
    if repetitions < 1:
        raise ValueError("at least one group-disjoint repetition is required")
    molecular_weight = bundle.descriptor_matrix[
        :, DESCRIPTOR_NAMES.index(PRIMARY_DESCRIPTOR)
    ]
    codes, unique_groups = _group_codes(bundle.groups)
    group_frame = pd.DataFrame(
        {"group_code": codes, "molecular_weight": molecular_weight}
    )
    grouped = group_frame.groupby("group_code", sort=True).molecular_weight.agg(
        ["size", "median"]
    )
    if len(grouped) != len(unique_groups) or not np.array_equal(
        grouped.index.to_numpy(dtype=int), np.arange(len(unique_groups))
    ):
        raise AssertionError("chemical-group summary does not align with group codes")
    group_sizes = grouped["size"].to_numpy(dtype=np.int64)
    group_medians = grouped["median"].to_numpy(dtype=np.float64)
    rng = np.random.default_rng(seed)
    records: list[dict[str, Any]] = []
    for repetition in range(repetitions):
        assignment = balanced_group_split(
            group_sizes, group_medians, rng, strata=10
        )
        first_mask = assignment[codes] == 0
        second_mask = assignment[codes] == 1
        first_groups = set(codes[first_mask].tolist())
        second_groups = set(codes[second_mask].tolist())
        if first_groups.intersection(second_groups):
            raise AssertionError("chemical-group leakage between control halves")
        common = {
            "dataset": bundle.name,
            "repetition": repetition,
            "first_ligands": int(first_mask.sum()),
            "second_ligands": int(second_mask.sum()),
            "first_chemical_groups": int(len(first_groups)),
            "second_chemical_groups": int(len(second_groups)),
            "chemical_group_overlap": 0,
            "first_median_molecular_weight": float(
                np.median(molecular_weight[first_mask])
            ),
            "second_median_molecular_weight": float(
                np.median(molecular_weight[second_mask])
            ),
            "absolute_median_mw_difference": float(
                abs(
                    np.median(molecular_weight[first_mask])
                    - np.median(molecular_weight[second_mask])
                )
            ),
            "mw_ks_statistic": float(
                stats.ks_2samp(
                    molecular_weight[first_mask], molecular_weight[second_mask]
                ).statistic
            ),
        }
        for transformation, surface in surfaces.items():
            records.append(
                {
                    **common,
                    "transformation": transformation,
                    **geometry_comparison(
                        target_correlation(surface[first_mask]),
                        target_correlation(surface[second_mask]),
                    ),
                }
            )
    return pd.DataFrame.from_records(records)


def within_mw_band_reproducibility_controls(
    bundle: DatasetBundle,
    surfaces: dict[str, np.ndarray],
    random_repetitions: int,
    group_disjoint_repetitions: int,
    seed: int,
) -> pd.DataFrame:
    """Restriction-matched map reproducibility within each extreme MW band.

    The low- and high-MW threshold groups are analysed separately.  The
    row-random control partitions the rows of one band into equal-size,
    nonoverlapping halves; chemical groups may occur in both halves.  The
    chemical-group-disjoint control assigns every frozen group represented in
    that band wholly to one half, within deciles of the group's median MW and
    while greedily balancing row counts.  Thus neither control compares across
    MW bands, and the latter also prevents scaffold/Butina-group leakage.

    Repeated-split quantiles are composition-sensitivity ranges, not confidence
    intervals for a population parameter.  Only the raw and row-centred
    surfaces are evaluated because these are the two headline geometries.
    """
    if random_repetitions < 1 or group_disjoint_repetitions < 1:
        raise ValueError("within-band controls require positive repetition counts")
    required_surfaces = ("raw", "row_centered_residual")
    if any(name not in surfaces for name in required_surfaces):
        raise ValueError("within-band controls require raw and residual surfaces")

    molecular_weight = bundle.descriptor_matrix[
        :, DESCRIPTOR_NAMES.index(PRIMARY_DESCRIPTOR)
    ]
    low, high, _, _ = extreme_masks(molecular_weight)
    global_codes, _ = _group_codes(bundle.groups)
    records: list[dict[str, Any]] = []

    def append_comparisons(
        *,
        band_name: str,
        control_type: str,
        repetition: int,
        split_seed: int,
        first_index: np.ndarray,
        second_index: np.ndarray,
        unassigned_ligands: int,
    ) -> None:
        first_groups = set(global_codes[first_index].tolist())
        second_groups = set(global_codes[second_index].tolist())
        overlap = len(first_groups.intersection(second_groups))
        if control_type == "mw_stratified_chemical_group_disjoint" and overlap:
            raise AssertionError("chemical-group leakage in within-band control")
        common = {
            "dataset": bundle.name,
            "mw_band": band_name,
            "control_type": control_type,
            "repetition": repetition,
            "split_seed": split_seed,
            "first_ligands": int(len(first_index)),
            "second_ligands": int(len(second_index)),
            "unassigned_ligands": int(unassigned_ligands),
            "first_chemical_groups": int(len(first_groups)),
            "second_chemical_groups": int(len(second_groups)),
            "chemical_group_overlap": int(overlap),
            "first_median_molecular_weight": float(
                np.median(molecular_weight[first_index])
            ),
            "second_median_molecular_weight": float(
                np.median(molecular_weight[second_index])
            ),
            "absolute_median_mw_difference": float(
                abs(
                    np.median(molecular_weight[first_index])
                    - np.median(molecular_weight[second_index])
                )
            ),
            "mw_ks_statistic": float(
                stats.ks_2samp(
                    molecular_weight[first_index],
                    molecular_weight[second_index],
                ).statistic
            ),
        }
        for transformation in required_surfaces:
            surface = surfaces[transformation]
            records.append(
                {
                    **common,
                    "transformation": transformation,
                    **geometry_comparison(
                        target_correlation(surface[first_index]),
                        target_correlation(surface[second_index]),
                    ),
                }
            )

    for band_index, (band_name, band_mask) in enumerate(
        (("low_mw", low), ("high_mw", high))
    ):
        band_index_values = np.flatnonzero(band_mask)
        half_size = len(band_index_values) // 2
        if half_size < 3:
            raise ValueError(f"{band_name} has too few rows for two correlations")

        row_seed = seed + band_index
        row_rng = np.random.default_rng(row_seed)
        for repetition in range(random_repetitions):
            order = row_rng.permutation(band_index_values)
            first_index = order[:half_size]
            second_index = order[half_size : 2 * half_size]
            if np.intersect1d(first_index, second_index).size:
                raise AssertionError("row-random within-band halves overlap")
            append_comparisons(
                band_name=band_name,
                control_type="row_random_disjoint",
                repetition=repetition,
                split_seed=row_seed,
                first_index=first_index,
                second_index=second_index,
                unassigned_ligands=len(band_index_values) - 2 * half_size,
            )

        # Re-factor groups after restriction so the assignment vector is dense
        # and every group summary is computed from the same MW band being tested.
        restricted_codes, restricted_groups = _group_codes(
            np.asarray(bundle.groups, dtype=object)[band_mask]
        )
        group_frame = pd.DataFrame(
            {
                "group_code": restricted_codes,
                "molecular_weight": molecular_weight[band_mask],
            }
        )
        grouped = group_frame.groupby(
            "group_code", sort=True
        ).molecular_weight.agg(["size", "median"])
        if len(grouped) != len(restricted_groups) or not np.array_equal(
            grouped.index.to_numpy(dtype=int), np.arange(len(restricted_groups))
        ):
            raise AssertionError("within-band group summary is misaligned")
        group_sizes = grouped["size"].to_numpy(dtype=np.int64)
        group_medians = grouped["median"].to_numpy(dtype=np.float64)
        group_seed = seed + 100 + band_index
        group_rng = np.random.default_rng(group_seed)
        for repetition in range(group_disjoint_repetitions):
            assignment = balanced_group_split(
                group_sizes,
                group_medians,
                group_rng,
                strata=10,
            )
            first_index = band_index_values[assignment[restricted_codes] == 0]
            second_index = band_index_values[assignment[restricted_codes] == 1]
            append_comparisons(
                band_name=band_name,
                control_type="mw_stratified_chemical_group_disjoint",
                repetition=repetition,
                split_seed=group_seed,
                first_index=first_index,
                second_index=second_index,
                unassigned_ligands=0,
            )

    return pd.DataFrame.from_records(records)


def summarize_within_mw_band_controls(controls: pd.DataFrame) -> pd.DataFrame:
    """Summarize repeated within-band partitions without CI terminology."""
    required = {
        "dataset",
        "mw_band",
        "control_type",
        "transformation",
        "geometry_spearman",
    }
    if not required.issubset(controls.columns) or controls.empty:
        raise ValueError("within-band control table is incomplete")
    records: list[dict[str, Any]] = []
    for keys, group in controls.groupby(
        ["dataset", "mw_band", "control_type", "transformation"],
        sort=True,
    ):
        rho = group.geometry_spearman.to_numpy(dtype=np.float64)
        records.append(
            {
                "dataset": keys[0],
                "mw_band": keys[1],
                "control_type": keys[2],
                "transformation": keys[3],
                "repetitions": int(len(group)),
                "geometry_spearman_mean": float(rho.mean()),
                "geometry_spearman_median": float(np.median(rho)),
                "geometry_spearman_q025": float(np.quantile(rho, 0.025)),
                "geometry_spearman_q975": float(np.quantile(rho, 0.975)),
                "first_ligands_mean": float(group.first_ligands.mean()),
                "second_ligands_mean": float(group.second_ligands.mean()),
                "chemical_group_overlap_mean": float(
                    group.chemical_group_overlap.mean()
                ),
                "chemical_group_overlap_maximum": int(
                    group.chemical_group_overlap.max()
                ),
                "absolute_median_mw_difference_mean": float(
                    group.absolute_median_mw_difference.mean()
                ),
                "mw_ks_statistic_mean": float(group.mw_ks_statistic.mean()),
                "interval_interpretation": (
                    "central 95% repeated-split composition-sensitivity range; "
                    "not a population confidence interval"
                ),
            }
        )
    return pd.DataFrame.from_records(records)


def dockstring_support_seed_sensitivity(
    bundle: DatasetBundle,
    dockstring_path: Path,
    sample_size: int,
    seeds: tuple[int, ...],
) -> pd.DataFrame:
    """Repeat the MW-extreme contrast on independently sampled source supports."""
    if bundle.name != "DOCKSTRING-58":
        raise ValueError("support-seed sensitivity is defined only for DOCKSTRING")
    source = pd.read_csv(dockstring_path, sep="\t")
    score_columns = [
        column for column in source if column not in {"inchikey", "smiles"}
    ]
    complete = source[score_columns].notna().all(axis=1)
    smiles = source.loc[complete, "smiles"].reset_index(drop=True).astype(str)
    if len(smiles) != len(bundle.spectral_matrix):
        raise ValueError("complete DOCKSTRING rows do not align with spectral matrix")
    if sample_size > len(smiles):
        raise ValueError("support sensitivity sample exceeds complete rows")
    records: list[dict[str, Any]] = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        selected = np.sort(rng.choice(len(smiles), sample_size, replace=False))
        weights: list[float] = []
        for row_index, value in enumerate(smiles.iloc[selected]):
            molecule = Chem.MolFromSmiles(value)
            if molecule is None:
                raise ValueError(
                    f"invalid DOCKSTRING SMILES in support seed {seed}, row {row_index}"
                )
            weights.append(float(Descriptors.MolWt(molecule)))
        molecular_weight = np.asarray(weights, dtype=np.float64)
        low, high, low_threshold, high_threshold = extreme_masks(molecular_weight)
        sampled_matrix = bundle.spectral_matrix[selected]
        for transformation, surface in (
            ("raw", sampled_matrix),
            ("row_centered_residual", row_center(sampled_matrix)),
        ):
            comparison = geometry_comparison(
                target_correlation(surface[low]), target_correlation(surface[high])
            )
            records.append(
                {
                    "dataset": bundle.name,
                    "sample_seed": int(seed),
                    "is_inherited_primary_support_seed": bool(seed == 71),
                    "sample_ligands": int(sample_size),
                    "transformation": transformation,
                    "low_ligands": int(low.sum()),
                    "high_ligands": int(high.sum()),
                    "low_threshold_inclusive": low_threshold,
                    "high_threshold_inclusive": high_threshold,
                    **comparison,
                }
            )
    return pd.DataFrame.from_records(records)


def load_fixed_endpoint(
    target_order: list[str], endpoint_path: Path
) -> tuple[np.ndarray, np.ndarray, list[str]] | None:
    common_targets = list(PKIS1_TARGET_MAP)
    if not all(target in target_order for target in common_targets):
        return None
    endpoint = pd.read_csv(endpoint_path)
    required = {"target_a", "target_b", "primary_replicated_positive"}
    if not required.issubset(endpoint.columns):
        raise ValueError("fixed endpoint file lacks required columns")
    lookup = {
        tuple(sorted((str(row.target_a), str(row.target_b)))): bool(
            row.primary_replicated_positive
        )
        for row in endpoint.itertuples()
    }
    target_indices = np.asarray(
        [target_order.index(target) for target in common_targets], dtype=np.int64
    )
    tri = np.triu_indices(len(common_targets), k=1)
    labels: list[bool] = []
    for first, second in zip(*tri, strict=True):
        key = tuple(sorted((common_targets[first], common_targets[second])))
        if key not in lookup:
            raise ValueError(f"fixed endpoint lacks target pair {key}")
        labels.append(lookup[key])
    return target_indices, np.asarray(labels, dtype=bool), common_targets


def mw_quintile_analysis(
    bundle: DatasetBundle,
    surfaces: dict[str, np.ndarray],
    endpoint_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    descriptor = bundle.descriptor_matrix[:, DESCRIPTOR_NAMES.index(PRIMARY_DESCRIPTOR)]
    bin_ids, edges = quantile_bin_ids(descriptor, QUINTILE_COUNT)
    endpoint = load_fixed_endpoint(bundle.targets, endpoint_path)
    bin_records: list[dict[str, Any]] = []
    pair_records: list[dict[str, Any]] = []
    pair_detail_records: list[dict[str, Any]] = []
    for transformation, surface in surfaces.items():
        geometries: list[np.ndarray] = []
        medians: list[float] = []
        for bin_index in range(QUINTILE_COUNT):
            mask = bin_ids == bin_index
            geometry = target_correlation(surface[mask])
            geometries.append(geometry)
            medians.append(float(np.median(descriptor[mask])))
            values = upper_triangle(geometry)
            record: dict[str, Any] = {
                "dataset": bundle.name,
                "transformation": transformation,
                "mw_quintile": bin_index + 1,
                "ligands": int(mask.sum()),
                "lower_edge": float(edges[bin_index]),
                "upper_edge": float(edges[bin_index + 1]),
                "median_molecular_weight": medians[-1],
                "target_correlation_pr": correlation_pr(geometry),
                "mean_signed_target_correlation": float(values.mean()),
                "mean_absolute_target_correlation": float(np.abs(values).mean()),
            }
            if endpoint is not None:
                target_indices, labels, _ = endpoint
                endpoint_surface = surface[mask][:, target_indices]
                # The fixed endpoint is defined on these 20 targets.  Re-centering
                # within that panel matches the panel-restricted estimand used by the
                # existing experimental-geometry analyses; for an already row-centered
                # surface this exactly cancels the full-panel row offset.
                if transformation != "raw":
                    endpoint_surface = row_center(endpoint_surface)
                endpoint_geometry = target_correlation(endpoint_surface)
                metrics = retrieval_metrics(labels, upper_triangle(endpoint_geometry))
                record.update(
                    {
                        "fixed_endpoint_roc_auc": float(metrics["roc_auc"]),
                        "fixed_endpoint_average_precision": float(
                            metrics["average_precision"]
                        ),
                        "fixed_endpoint_positive_pairs": int(labels.sum()),
                        "fixed_endpoint_total_pairs": int(len(labels)),
                    }
                )
            bin_records.append(record)

        target_first, target_second = np.triu_indices(len(bundle.targets), k=1)
        for first_bin in range(QUINTILE_COUNT):
            for second_bin in range(first_bin + 1, QUINTILE_COUNT):
                comparison = geometry_comparison(
                    geometries[first_bin], geometries[second_bin]
                )
                pair_records.append(
                    {
                        "dataset": bundle.name,
                        "transformation": transformation,
                        "first_mw_quintile": first_bin + 1,
                        "second_mw_quintile": second_bin + 1,
                        "first_median_molecular_weight": medians[first_bin],
                        "second_median_molecular_weight": medians[second_bin],
                        "median_molecular_weight_separation": abs(
                            medians[first_bin] - medians[second_bin]
                        ),
                        **comparison,
                    }
                )
                if first_bin == 0 and second_bin == QUINTILE_COUNT - 1:
                    first_values = upper_triangle(geometries[first_bin])
                    second_values = upper_triangle(geometries[second_bin])
                    first_top = strongest_pair_mask(first_values)
                    second_top = strongest_pair_mask(second_values)
                    for index, (target_a, target_b) in enumerate(
                        zip(target_first, target_second, strict=True)
                    ):
                        pair_detail_records.append(
                            {
                                "dataset": bundle.name,
                                "transformation": transformation,
                                "target_a": bundle.targets[target_a],
                                "target_b": bundle.targets[target_b],
                                "low_mw_quintile_correlation": first_values[index],
                                "high_mw_quintile_correlation": second_values[index],
                                "high_minus_low_correlation": (
                                    second_values[index] - first_values[index]
                                ),
                                "sign_flip": bool(
                                    np.sign(first_values[index])
                                    != np.sign(second_values[index])
                                ),
                                "top_positive_10pct_in_low_mw": bool(first_top[index]),
                                "top_positive_10pct_in_high_mw": bool(second_top[index]),
                            }
                        )
    return (
        pd.DataFrame.from_records(bin_records),
        pd.DataFrame.from_records(pair_records),
        pd.DataFrame.from_records(pair_detail_records),
    )


def summarize_primary(
    observed: pd.DataFrame,
    random_null: pd.DataFrame,
    group_bootstrap: pd.DataFrame,
    group_disjoint: pd.DataFrame,
) -> pd.DataFrame:
    primary = observed.loc[observed.descriptor.eq(PRIMARY_DESCRIPTOR)].copy()
    records: list[dict[str, Any]] = []
    for row in primary.itertuples(index=False):
        null = random_null.loc[
            (random_null.dataset == row.dataset)
            & (random_null.transformation == row.transformation)
        ]
        bootstrap = group_bootstrap.loc[
            (group_bootstrap.dataset == row.dataset)
            & (group_bootstrap.transformation == row.transformation)
        ]
        disjoint = group_disjoint.loc[
            (group_disjoint.dataset == row.dataset)
            & (group_disjoint.transformation == row.transformation)
        ]
        record = row._asdict()
        null_rho = null.geometry_spearman.to_numpy(dtype=np.float64)
        record.update(
            {
                "random_disjoint_repetitions": int(len(null_rho)),
                "random_disjoint_geometry_spearman_mean": float(null_rho.mean()),
                "random_disjoint_geometry_spearman_q025": float(
                    np.quantile(null_rho, 0.025)
                ),
                "random_disjoint_geometry_spearman_q975": float(
                    np.quantile(null_rho, 0.975)
                ),
                "random_minus_observed_geometry_spearman": float(
                    null_rho.mean() - row.geometry_spearman
                ),
                "random_lower_tail_probability": float(
                    (1 + np.sum(null_rho <= row.geometry_spearman))
                    / (len(null_rho) + 1)
                ),
            }
        )
        if len(bootstrap):
            bootstrap_rho = bootstrap.geometry_spearman.to_numpy(dtype=np.float64)
            record.update(
                {
                    "chemical_group_bootstrap_repetitions": int(len(bootstrap_rho)),
                    "chemical_group_bootstrap_geometry_spearman_q025": float(
                        np.quantile(bootstrap_rho, 0.025)
                    ),
                    "chemical_group_bootstrap_geometry_spearman_median": float(
                        np.median(bootstrap_rho)
                    ),
                    "chemical_group_bootstrap_geometry_spearman_q975": float(
                        np.quantile(bootstrap_rho, 0.975)
                    ),
                }
            )
        if len(disjoint):
            disjoint_rho = disjoint.geometry_spearman.to_numpy(dtype=np.float64)
            record.update(
                {
                    "mw_stratified_group_disjoint_repetitions": int(
                        len(disjoint_rho)
                    ),
                    "mw_stratified_group_disjoint_geometry_spearman_mean": float(
                        disjoint_rho.mean()
                    ),
                    "mw_stratified_group_disjoint_geometry_spearman_q025": float(
                        np.quantile(disjoint_rho, 0.025)
                    ),
                    "mw_stratified_group_disjoint_geometry_spearman_q975": float(
                        np.quantile(disjoint_rho, 0.975)
                    ),
                    "mw_stratified_group_disjoint_mw_ks_mean": float(
                        disjoint.mw_ks_statistic.mean()
                    ),
                    "mw_stratified_group_disjoint_absolute_median_mw_difference_mean": float(
                        disjoint.absolute_median_mw_difference.mean()
                    ),
                }
            )
        records.append(record)
    return pd.DataFrame.from_records(records)


def quintile_trend_summary(pair_metrics: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for (dataset, transformation), group in pair_metrics.groupby(
        ["dataset", "transformation"], sort=True
    ):
        rho = float(
            stats.spearmanr(
                group.median_molecular_weight_separation,
                group.geometry_dissimilarity,
            ).statistic
        )
        records.append(
            {
                "dataset": dataset,
                "transformation": transformation,
                "mw_quintile_pairs": int(len(group)),
                "spearman_median_mw_separation_vs_geometry_dissimilarity": rho,
                "analysis_status": "descriptive_post_hoc",
            }
        )
    return pd.DataFrame.from_records(records)


def quintile_association_summary(bin_metrics: pd.DataFrame) -> pd.DataFrame:
    """Descriptive across-bin associations; five bins preclude formal inference."""
    records: list[dict[str, Any]] = []
    for (dataset, transformation), group in bin_metrics.groupby(
        ["dataset", "transformation"], sort=True
    ):
        group = group.sort_values("mw_quintile")
        record: dict[str, Any] = {
            "dataset": dataset,
            "transformation": transformation,
            "mw_quintiles": int(len(group)),
            "spearman_median_mw_vs_target_correlation_pr": float(
                stats.spearmanr(
                    group.median_molecular_weight, group.target_correlation_pr
                ).statistic
            ),
            "analysis_status": "descriptive_post_hoc_five_bins",
        }
        if group.fixed_endpoint_roc_auc.notna().all():
            record.update(
                {
                    "spearman_target_correlation_pr_vs_fixed_endpoint_roc_auc": float(
                        stats.spearmanr(
                            group.target_correlation_pr,
                            group.fixed_endpoint_roc_auc,
                        ).statistic
                    ),
                    "spearman_target_correlation_pr_vs_fixed_endpoint_average_precision": float(
                        stats.spearmanr(
                            group.target_correlation_pr,
                            group.fixed_endpoint_average_precision,
                        ).statistic
                    ),
                }
            )
        records.append(record)
    return pd.DataFrame.from_records(records)


def _readme(
    primary: pd.DataFrame,
    trends: pd.DataFrame,
    bins: pd.DataFrame,
    support_sensitivity: pd.DataFrame,
    within_band_summary: pd.DataFrame,
    metadata: dict[str, Any],
) -> str:
    lines = [
        "# Chemical-context dependence of target geometry",
        "",
        "This is a science-only, explicitly post-hoc analysis. Molecular weight was",
        "chosen after preliminary inspection, so the reported probabilities are",
        "calibration diagnostics, not confirmatory hypothesis tests.",
        "",
        "## Primary result",
        "",
    ]
    for dataset in ("DOCKSTRING-58", "Docking-44"):
        raw = primary.loc[
            primary.dataset.eq(dataset) & primary.transformation.eq("raw")
        ].iloc[0]
        row = primary.loc[
            primary.dataset.eq(dataset)
            & primary.transformation.eq("row_centered_residual")
        ].iloc[0]
        adjusted = primary.loc[
            primary.dataset.eq(dataset)
            & primary.transformation.eq(
                "group_heldout_descriptor_adjusted_residual"
            )
        ].iloc[0]
        lines.append(
            f"- **{dataset}:** low-versus-high MW threshold-group geometry Spearman changes "
            f"from {raw.geometry_spearman:.3f} on the raw surface to "
            f"{row.geometry_spearman:.3f} after row centering; random disjoint residual "
            "supports "
            f"mean = {row.random_disjoint_geometry_spearman_mean:.3f} "
            f"(lower-tail Monte Carlo probability "
            f"{row.random_lower_tail_probability:.4f}). The correlation is "
            f"{adjusted.geometry_spearman:.3f} after strict chemical-group-held-out "
            "linear removal of seven ligand descriptors. MW-stratified halves with "
            "no shared chemical groups retain geometry agreement "
            f"{row.mw_stratified_group_disjoint_geometry_spearman_mean:.3f}."
        )
        within = within_band_summary.loc[
            within_band_summary.dataset.eq(dataset)
            & within_band_summary.transformation.eq("row_centered_residual")
            & within_band_summary.control_type.eq(
                "mw_stratified_chemical_group_disjoint"
            )
        ].set_index("mw_band")
        lines.append(
            "  Restriction-matched, chemical-group-disjoint splits reproduce the "
            "residual map within the low- and high-MW domains with mean Spearman "
            f"{within.loc['low_mw', 'geometry_spearman_mean']:.3f} and "
            f"{within.loc['high_mw', 'geometry_spearman_mean']:.3f}, respectively; "
            "their repeated-split ranges are composition sensitivities, not "
            "population confidence intervals."
        )
    lines.extend(
        [
            "",
            "Across five MW bins, increasing separation in median MW tracks increasing",
            "dissimilarity between residual target networks:",
            "",
        ]
    )
    for row in trends.loc[
        trends.transformation.eq("row_centered_residual")
    ].itertuples(index=False):
        lines.append(
            f"- {row.dataset}: Spearman = "
            f"{row.spearman_median_mw_separation_vs_geometry_dissimilarity:.3f}."
        )
    support_rho = support_sensitivity.loc[
        support_sensitivity.transformation.eq("row_centered_residual"),
        "geometry_spearman",
    ]
    lines.extend(
        [
            "",
            "Across six independently sampled 15,000-ligand DOCKSTRING supports, the",
            f"primary residual agreement ranges from {support_rho.min():.3f} to "
            f"{support_rho.max():.3f}.",
        ]
    )
    endpoint = bins.loc[
        bins.dataset.eq("DOCKSTRING-58")
        & bins.transformation.eq("row_centered_residual")
    ].dropna(subset=["fixed_endpoint_roc_auc"])
    if len(endpoint):
        low = endpoint.sort_values("mw_quintile").iloc[0]
        high = endpoint.sort_values("mw_quintile").iloc[-1]
        lines.extend(
            [
                "",
                "On the reused 20-target experimental co-selectivity endpoint, the low-",
                f"MW quintile gives AUROC/AP {low.fixed_endpoint_roc_auc:.3f}/"
                f"{low.fixed_endpoint_average_precision:.3f}, versus "
                f"{high.fixed_endpoint_roc_auc:.3f}/"
                f"{high.fixed_endpoint_average_precision:.3f} in the high-MW quintile.",
                f"Over the same bins, residual PR rises from "
                f"{low.target_correlation_pr:.2f} to {high.target_correlation_pr:.2f}; "
                "thus higher effective dimension is not a monotone fidelity signal in",
                "this descriptive within-dataset comparison.",
                "This comparison is descriptive and was not prospectively specified.",
            ]
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "The result supports a *conditional geometry* interpretation: the inferred",
            "target network depends on which chemical domain probes the proteins. It does",
            "not show that molecular weight is causal, that either domain is biologically",
            "correct, or that target ranking for individual ligands improves. Correlated",
            "size descriptors, scaffold composition, Vina's score form, and pocket fit can",
            "all contribute. The primary descriptor-removal analysis uses one inherited",
            "fixed 15,000-ligand DOCKSTRING support, supplemented by six support seeds;",
            "Docking-44 provides cross-panel replication but differs in targets, receptors,",
            "and chemical library.",
            "",
            "## Reproduction",
            "",
            "```bash",
            "./.venv/bin/python analysis/chemical_context_geometry.py",
            "./.venv/bin/python -m pytest -q analysis/test_chemical_context_geometry.py",
            "```",
            "",
            f"Random disjoint repetitions: {metadata['random_disjoint_repetitions']}; "
            f"chemical-group bootstrap repetitions: "
            f"{metadata['chemical_group_bootstrap_repetitions']}; MW-stratified "
            f"group-disjoint repetitions: "
            f"{metadata['mw_stratified_group_disjoint_repetitions']}; seed: "
            f"{metadata['seed']}.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--endpoint", type=Path, default=DEFAULT_ENDPOINT)
    parser.add_argument(
        "--random-repetitions", type=int, default=DEFAULT_RANDOM_REPETITIONS
    )
    parser.add_argument(
        "--group-bootstraps", type=int, default=DEFAULT_GROUP_BOOTSTRAPS
    )
    parser.add_argument(
        "--group-disjoint-repetitions",
        type=int,
        default=DEFAULT_GROUP_DISJOINT_REPETITIONS,
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--dockstring-sample-size", type=int, default=15_000)
    parser.add_argument("--dockstring-sample-seed", type=int, default=71)
    parser.add_argument(
        "--dockstring-support-seeds",
        default=",".join(map(str, DEFAULT_DOCKSTRING_SUPPORT_SEEDS)),
    )
    args = parser.parse_args()
    if args.random_repetitions < 200:
        raise ValueError("the frozen random-split calibration requires >=200 repeats")
    if args.group_bootstraps < 100:
        raise ValueError("the frozen group bootstrap requires >=100 repeats")
    if args.group_disjoint_repetitions < 200:
        raise ValueError("the frozen group-disjoint control requires >=200 repeats")

    bundles = [
        load_dockstring(
            AnalysisConfig(
                dockstring_sample_size=args.dockstring_sample_size,
                dockstring_sample_seed=args.dockstring_sample_seed,
            )
        ),
        load_docking44(),
    ]
    support_seeds = tuple(
        int(value) for value in args.dockstring_support_seeds.split(",") if value
    )
    if args.dockstring_sample_seed not in support_seeds:
        raise ValueError("support-seed sensitivity must include the primary sample seed")
    observed_frames: list[pd.DataFrame] = []
    random_frames: list[pd.DataFrame] = []
    bootstrap_frames: list[pd.DataFrame] = []
    group_disjoint_frames: list[pd.DataFrame] = []
    within_band_control_frames: list[pd.DataFrame] = []
    bin_frames: list[pd.DataFrame] = []
    bin_pair_frames: list[pd.DataFrame] = []
    pair_detail_frames: list[pd.DataFrame] = []
    descriptor_correlation_frames: list[pd.DataFrame] = []
    decomposition_metadata: dict[str, Any] = {}

    for dataset_index, bundle in enumerate(bundles):
        surfaces, decomposition = surface_variants(bundle, args.folds)
        decomposition_metadata[bundle.name] = decomposition
        observed_frames.append(descriptor_extreme_analysis(bundle, surfaces))
        descriptor_correlation_frames.append(descriptor_correlation_table(bundle))
        random_frames.append(
            random_disjoint_support_null(
                bundle,
                surfaces,
                args.random_repetitions,
                args.seed + dataset_index * 10,
            )
        )
        bootstrap_frames.append(
            group_bootstrap_primary(
                bundle,
                surfaces,
                args.group_bootstraps,
                args.seed + dataset_index * 10 + 1,
            )
        )
        group_disjoint_frames.append(
            mw_matched_group_disjoint_control(
                bundle,
                surfaces,
                args.group_disjoint_repetitions,
                args.seed + dataset_index * 10 + 2,
            )
        )
        within_band_control_frames.append(
            within_mw_band_reproducibility_controls(
                bundle,
                surfaces,
                args.random_repetitions,
                args.group_disjoint_repetitions,
                args.seed + dataset_index * 10 + 3,
            )
        )
        bins, bin_pairs, pair_details = mw_quintile_analysis(
            bundle, surfaces, args.endpoint
        )
        bin_frames.append(bins)
        bin_pair_frames.append(bin_pairs)
        pair_detail_frames.append(pair_details)

    observed = pd.concat(observed_frames, ignore_index=True)
    random_null = pd.concat(random_frames, ignore_index=True)
    group_bootstrap = pd.concat(bootstrap_frames, ignore_index=True)
    group_disjoint = pd.concat(group_disjoint_frames, ignore_index=True)
    within_band_controls = pd.concat(
        within_band_control_frames, ignore_index=True
    )
    within_band_summary = summarize_within_mw_band_controls(
        within_band_controls
    )
    bin_metrics = pd.concat(bin_frames, ignore_index=True)
    bin_pair_metrics = pd.concat(bin_pair_frames, ignore_index=True)
    pair_details = pd.concat(pair_detail_frames, ignore_index=True)
    descriptor_correlations = pd.concat(
        descriptor_correlation_frames, ignore_index=True
    )
    primary = summarize_primary(
        observed, random_null, group_bootstrap, group_disjoint
    )
    trends = quintile_trend_summary(bin_pair_metrics)
    associations = quintile_association_summary(bin_metrics)
    support_sensitivity = dockstring_support_seed_sensitivity(
        bundles[0],
        PACKAGE / "data" / "frozen" / "dockstring-dataset.tsv.gz",
        args.dockstring_sample_size,
        support_seeds,
    )

    metadata: dict[str, Any] = {
        "analysis_status": "exploratory_post_hoc_science_only",
        "primary_stratifier": PRIMARY_DESCRIPTOR,
        "primary_stratifier_selection": (
            "selected after preliminary inspection; not prospectively prespecified"
        ),
        "primary_extreme_fraction_per_tail": PRIMARY_EXTREME_FRACTION,
        "random_disjoint_repetitions": args.random_repetitions,
        "chemical_group_bootstrap_repetitions": args.group_bootstraps,
        "mw_stratified_group_disjoint_repetitions": (
            args.group_disjoint_repetitions
        ),
        "within_mw_band_random_repetitions": args.random_repetitions,
        "within_mw_band_group_disjoint_repetitions": (
            args.group_disjoint_repetitions
        ),
        "seed": args.seed,
        "dockstring_support_sensitivity_seeds": support_seeds,
        "descriptor_names_in_order": DESCRIPTOR_NAMES,
        "transformations": {
            "raw": "unmodified clipped/imputed Vina score surface",
            "row_centered_residual": (
                "within-ligand target mean removed; target correlation then removes "
                "column offsets, making this correlation-equivalent to two-way centering"
            ),
            "group_heldout_descriptor_adjusted_residual": (
                "five-fold group-held-out residual score error after fold-local target "
                "offsets/scales, descriptor scaling, and seven-descriptor linear fit"
            ),
        },
        "random_null": (
            "two simple-random, nonoverlapping supports matching the observed low/high "
            "observed low/high MW threshold-group row counts; finite-support calibration, not a chemical-group null"
        ),
        "group_bootstrap": {
            "scheme": (
                "paired nonparametric cluster bootstrap; sample the union of groups in "
                "both MW domains with replacement and apply each group multiplicity to "
                "all of its rows in both domains"
            ),
            "Docking-44_unit": bundles[1].group_definition,
            "DOCKSTRING-58_unit": bundles[0].group_definition,
        },
        "mw_stratified_group_disjoint_control": {
            "scheme": (
                "assign every frozen chemical group wholly to one of two halves; "
                "assignment occurs within deciles of group-median MW and greedily "
                "balances ligand-row counts"
            ),
            "purpose": (
                "distinguish MW-domain drift from generic chemically disjoint library "
                "composition under closely matched MW distributions"
            ),
            "Docking-44_unit": bundles[1].group_definition,
            "DOCKSTRING-58_unit": bundles[0].group_definition,
        },
        "within_mw_band_reproducibility_control": {
            "scheme": (
                "split the low- and high-MW threshold groups independently; "
                "compare equal-size row-random disjoint halves and whole-chemical-"
                "group halves assigned within deciles of within-band group-median MW"
            ),
            "purpose": (
                "distinguish genuine low-versus-high map drift from instability "
                "induced by estimating target correlations on an MW-restricted support"
            ),
            "transformations": ["raw", "row_centered_residual"],
            "interval_interpretation": (
                "central 95% repeated-split composition-sensitivity range; not a "
                "population confidence interval"
            ),
            "Docking-44_unit": bundles[1].group_definition,
            "DOCKSTRING-58_unit": bundles[0].group_definition,
        },
        "fixed_endpoint": {
            "path": str(args.endpoint.relative_to(PACKAGE)),
            "sha256": sha256_file(args.endpoint),
            "targets": list(PKIS1_TARGET_MAP),
            "role": "descriptive_reuse_only",
            "selection_independence": (
                "not used to choose MW, bin edges, transformations, or target-pair threshold"
            ),
            "panel_transform": (
                "non-raw surfaces are row-centered again after restriction to the fixed "
                "20-target panel, matching the panel-restricted residual estimand"
            ),
        },
        "datasets": {
            bundle.name: {
                "ligands": int(len(bundle.matrix)),
                "targets": int(bundle.matrix.shape[1]),
                "support_rule": bundle.support_rule,
                "chemical_groups": int(len(np.unique(bundle.groups))),
                "group_definition": bundle.group_definition,
                "missing_cells_before_preprocessing": int(
                    bundle.missing_cells_before_preprocessing
                ),
                "positive_cells_clipped": int(bundle.positive_cells_clipped),
            }
            for bundle in bundles
        },
        "descriptor_decomposition": decomposition_metadata,
        "claim_boundary": (
            "The analysis establishes reproducible chemical-domain dependence of the "
            "Vina-derived target network on the tested supports. It does not identify a "
            "causal descriptor, validate pose correctness, establish biological truth, "
            "or demonstrate improved per-ligand target retrieval."
        ),
    }

    metadata.update(
        {
            "primary_results": dataframe_records(primary),
            "within_mw_band_reproducibility_summary": dataframe_records(
                within_band_summary
            ),
            "mw_quintile_geometry_trends": dataframe_records(trends),
            "mw_quintile_associations": dataframe_records(associations),
            "dockstring_support_seed_sensitivity": support_sensitivity.loc[
                support_sensitivity.transformation.eq("row_centered_residual")
            ].pipe(dataframe_records),
            "size_descriptor_replication": observed.loc[
                observed.transformation.eq("row_centered_residual")
                & observed.descriptor.isin(
                    ["heavy_atoms", "molecular_weight", "labute_asa"]
                )
            ].pipe(dataframe_records),
        }
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = [
        args.output_dir / "descriptor_extremes.csv",
        args.output_dir / "descriptor_correlations.csv",
        args.output_dir / "primary_mw_contrasts.csv",
        args.output_dir / "random_disjoint_null.csv",
        args.output_dir / "chemical_group_bootstrap.csv",
        args.output_dir / "mw_stratified_group_disjoint_control.csv",
        args.output_dir / "mw_quintile_metrics.csv",
        args.output_dir / "mw_quintile_pair_geometry.csv",
        args.output_dir / "mw_quintile_trends.csv",
        args.output_dir / "mw_quintile_associations.csv",
        args.output_dir / "mw_extreme_target_pairs.csv",
        args.output_dir / "dockstring_support_seed_sensitivity.csv",
        args.output_dir / "within_mw_band_reproducibility_controls.csv",
        args.output_dir / "within_mw_band_control_summary.csv",
        args.output_dir / "summary.json",
        args.output_dir / "README.md",
    ]
    observed.to_csv(output_paths[0], index=False)
    descriptor_correlations.to_csv(
        output_paths[1], index=False
    )
    primary.to_csv(output_paths[2], index=False)
    random_null.to_csv(output_paths[3], index=False)
    group_bootstrap.to_csv(output_paths[4], index=False)
    group_disjoint.to_csv(output_paths[5], index=False)
    bin_metrics.to_csv(output_paths[6], index=False)
    bin_pair_metrics.to_csv(output_paths[7], index=False)
    trends.to_csv(output_paths[8], index=False)
    associations.to_csv(
        output_paths[9], index=False
    )
    pair_details.to_csv(output_paths[10], index=False)
    support_sensitivity.to_csv(
        output_paths[11], index=False
    )
    within_band_controls.to_csv(output_paths[12], index=False)
    within_band_summary.to_csv(output_paths[13], index=False)
    output_paths[14].write_text(
        json.dumps(json_ready(metadata), indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    output_paths[15].write_text(
        _readme(
            primary,
            trends,
            bin_metrics,
            support_sensitivity,
            within_band_summary,
            metadata,
        )
    )
    checksum_payload = {
        "algorithm": "sha256",
        "scope": "all frozen chemical-context outputs except this checksum manifest",
        "files": {path.name: sha256_file(path) for path in output_paths},
    }
    (args.output_dir / "output_checksums.json").write_text(
        json.dumps(checksum_payload, indent=2, sort_keys=True) + "\n"
    )
    print(
        primary[
            [
                "dataset",
                "transformation",
                "geometry_spearman",
                "random_disjoint_geometry_spearman_mean",
                "random_lower_tail_probability",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
