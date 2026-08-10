#!/usr/bin/env python3
"""Is the descriptor surface target-blind?  A shared-coefficient control.

``residual_mechanism_analysis.grouped_descriptor_decomposition`` -- reused by
``descriptor_component_geometry`` -- regresses the whole ligand-by-target residual
surface on ``[1, 7 standardized ligand descriptors]`` inside every training fold with a
single ``lstsq`` call.  That call returns an ``(8, n_targets)`` coefficient array, i.e. a
*separate slope vector for every target*.  The descriptor matrix never sees the protein,
but the fitted slopes do: each column is estimated from that target's own docking column,
so pocket volume, contact area and receptor composition can enter the predicted surface
through them.  A surface built that way is not entitled to the label "target-blind".

This script builds the control that separates the two ingredients.  Everything is held
fixed -- the same de-leaked DOCKSTRING support, the same fixed common-20 kinase panel, the
same five Bemis-Murcko group-held-out folds, the same fold-local target offsets, residual
target scales and descriptor standardization -- and only the coefficient structure varies:

* ``observed_residual``               -- the out-of-fold within-ligand residual surface;
* ``descriptor_component_target_specific``  -- the published arm, one slope vector per
  target (the ``(8, 20)`` array above);
* ``descriptor_component_target_blind`` -- one slope vector for the entire panel, fitted
  by pooled out-of-fold least squares over all 20 target columns at once.

The target-blind arm is degenerate *by construction*, and this script says so rather than
reporting its number as a discovery.  With one shared coefficient vector ``b`` the
prediction for ligand ``i`` at target ``j`` is ``d(i) . b`` for every ``j``: all 20
predicted target columns are literally the same vector of ligand scores, so all 190
target-pair correlations equal +1 exactly and the Spearman agreement with any experimental
panel is *undefined* (a constant vector has no ranks), not zero.  Confirming that is a
proof, not a measurement.

The informative quantities are therefore contrasts, and this script reports four:

1. an exact orthogonal variance split of the target-specific descriptor component into its
   shared part and its between-target coefficient-heterogeneity part (the cross term
   vanishes because the pooled coefficient vector is the mean of the per-target vectors);
2. the spread of the fitted per-target slopes for each of the seven descriptors;
3. a *coefficient-rank ladder*: the per-target coefficient deviations truncated to rank
   ``k = 0..7``, which asks how many effective degrees of between-target slope freedom are needed
   before the descriptor surface reproduces its experimental agreement (``k = 0`` is the
   target-blind arm; the nominal eighth intercept singular direction is numerical noise);
4. a slope-permutation null -- the control a reviewer actually wants -- which keeps the
   fitted slope vectors and destroys only their pairing with targets.  With one permutation
   shared across folds this is *algebraically identical* to the target-label QAP already
   reported (permuting coefficient columns permutes the predicted surface columns, so the
   correlation matrix is merely relabelled); the script verifies that identity numerically
   and also runs the fold-independent variant, which is not a relabelling.

Experimental geometry is read from the released 190-edge ledger rather than recomputed, so
this script needs no source-restricted panel workbook.
"""

from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold
from scipy import stats
from sklearn.model_selection import GroupKFold

try:  # Support direct CLI execution and package-style imports.
    from . import descriptor_component_geometry as dcg
    from . import residual_mechanism_analysis as mechanism
    from . import residual_target_geometry_validation as geometry
except ImportError:  # pragma: no cover - direct CLI execution.
    import descriptor_component_geometry as dcg  # type: ignore
    import residual_mechanism_analysis as mechanism  # type: ignore
    import residual_target_geometry_validation as geometry  # type: ignore


RDLogger.DisableLog("rdApp.*")

PACKAGE = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PACKAGE / "results" / "target_blind_descriptor_control"
DEFAULT_LEDGER = (
    PACKAGE / "results" / "released_pair_geometry_ledger" / "target_pair_geometry.csv"
)
DEFAULT_PUBLISHED_ARM = (
    PACKAGE / "results" / "descriptor_component_geometry" / "summary.json"
)
DEFAULT_PKIS1_ZIP = dcg.DEFAULT_PKIS1_ZIP
DEFAULT_SEED = 20260814
DEFAULT_FOLDS = 5
DEFAULT_QAP_PERMUTATIONS = 49_999
DEFAULT_SLOPE_PERMUTATIONS = 5_000
DEFAULT_BOOTSTRAP_REPEATS = 5_000
BOOTSTRAP_CHUNK = 200
EFFECTIVE_SVD_RTOL = 1e-10

TARGETS20 = dcg.TARGETS20
PANELS = dcg.PANELS
COEFFICIENT_NAMES = ("intercept",) + tuple(mechanism.DESCRIPTOR_NAMES)

ARM_OBSERVED = "observed_residual"
ARM_SPECIFIC = "descriptor_component_target_specific"
ARM_BLIND = "descriptor_component_target_blind"
ARM_LABELS = OrderedDict(
    [
        (ARM_OBSERVED, "out-of-fold within-ligand residual surface"),
        (
            ARM_SPECIFIC,
            "out-of-fold descriptor prediction with a separate fitted slope vector per "
            "target (the published descriptor component)",
        ),
        (
            ARM_BLIND,
            "out-of-fold descriptor prediction with one slope vector shared by the whole "
            "20-target panel (pooled least squares); contains no target-specific "
            "parameter of any kind",
        ),
    ]
)


# --------------------------------------------------------------------------------------
# inputs
# --------------------------------------------------------------------------------------


def load_ledger(path: Path) -> pd.DataFrame:
    """Read the released 190-edge ledger and confirm its locked upper-triangle order."""
    frame = pd.read_csv(path)
    expected = [
        (first, second)
        for index, first in enumerate(TARGETS20)
        for second in TARGETS20[index + 1 :]
    ]
    if list(zip(frame.target_a, frame.target_b)) != expected:
        raise ValueError(f"{path} is not in the locked upper-triangle target order")
    required = ["docking_local_20_centered_correlation"] + [
        f"{panel}_experimental_centered_correlation" for panel in PANELS
    ]
    missing = [name for name in required if name not in frame.columns]
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
    return frame


def generic_framework_keys(murcko_keys: np.ndarray) -> tuple[np.ndarray, int]:
    """Coarser chemical clusters: Bemis-Murcko *generic* frameworks.

    Butina clustering is not available at this support.  The package's own helper
    (``dense_davis_benchmark._butina_labels``) materialises the complete pairwise Tanimoto
    distance list, which is 3.37e10 entries for 259,579 ligands.  The generic Murcko
    framework -- the scaffold with every atom type and bond order erased -- is a strictly
    coarser chemical grouping than the Murcko scaffold used for the holdout folds and
    plays the conservative role Butina plays in the smaller-cohort analyses.  A scaffold
    RDKit cannot genericise keeps a key of its own, which only makes the clustering finer,
    so the fallback cannot inflate the reported uncertainty.
    """
    mapping: dict[str, str] = {}
    failures = 0
    for key in np.unique(murcko_keys):
        text = str(key)
        if text.startswith("ACYCLIC_SINGLETON:"):
            mapping[text] = text
            continue
        molecule = Chem.MolFromSmiles(text)
        if molecule is None:  # pragma: no cover - scaffold SMILES come from RDKit
            mapping[text] = f"GENERIC_FAILED:{text}"
            failures += 1
            continue
        try:
            generic = MurckoScaffold.MakeScaffoldGeneric(molecule)
            mapping[text] = Chem.MolToSmiles(generic)
        except Exception:  # pragma: no cover - RDKit valence corner cases
            mapping[text] = f"GENERIC_FAILED:{text}"
            failures += 1
    keys = np.asarray([mapping[str(key)] for key in murcko_keys], dtype=object)
    return keys, failures


# --------------------------------------------------------------------------------------
# decomposition
# --------------------------------------------------------------------------------------


def shared_and_specific_decomposition(
    matrix: np.ndarray,
    descriptors: np.ndarray,
    groups: np.ndarray,
    n_splits: int,
) -> dict[str, object]:
    """Fit both coefficient structures in one pass over the identical folds.

    The target-specific arm is exactly ``grouped_descriptor_decomposition``'s estimator
    (verified against it downstream).  The target-blind arm stacks all target columns into
    a single response and fits one coefficient vector; because every target shares the same
    design, that pooled solution equals the row-mean of the per-target coefficient array,
    which is what makes the variance split below exactly orthogonal.
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    descriptors = np.asarray(descriptors, dtype=np.float64)
    groups = np.asarray(groups, dtype=object)
    if len(matrix) != len(descriptors) or len(matrix) != len(groups):
        raise ValueError("matrix, descriptors and groups must have the same rows")
    n_targets = matrix.shape[1]

    oof_surface = np.full_like(matrix, np.nan)
    oof_specific = np.full_like(matrix, np.nan)
    oof_blind = np.full_like(matrix, np.nan)
    fold_index = np.full(len(matrix), -1, dtype=np.int64)
    coefficients = np.empty((n_splits, len(COEFFICIENT_NAMES), n_targets))
    pooled = np.empty((n_splits, len(COEFFICIENT_NAMES)))
    gram = np.empty((n_splits, n_targets, n_targets))
    column_sums = np.empty((n_splits, n_targets))
    design_gram = np.empty((n_splits, len(COEFFICIENT_NAMES), len(COEFFICIENT_NAMES)))
    design_sums = np.empty((n_splits, len(COEFFICIENT_NAMES)))
    fold_rows: list[dict[str, object]] = []
    pooled_versus_mean = 0.0

    splitter = GroupKFold(n_splits=n_splits)
    for fold, (train_index, test_index) in enumerate(
        splitter.split(descriptors, groups=groups)
    ):
        if set(groups[train_index].tolist()) & set(groups[test_index].tolist()):
            raise AssertionError("chemical group leakage between train and test fold")
        train_surface, test_surface, scale_record = mechanism.fold_local_residual_transform(
            matrix[train_index], matrix[test_index]
        )
        descriptor_mean = descriptors[train_index].mean(axis=0)
        descriptor_sd = descriptors[train_index].std(axis=0, ddof=1)
        if np.any(descriptor_sd <= 1e-12):
            raise ValueError("training fold has a constant descriptor")
        train_design = np.column_stack(
            [
                np.ones(len(train_index)),
                (descriptors[train_index] - descriptor_mean) / descriptor_sd,
            ]
        )
        test_design = np.column_stack(
            [
                np.ones(len(test_index)),
                (descriptors[test_index] - descriptor_mean) / descriptor_sd,
            ]
        )
        specific = np.linalg.lstsq(train_design, train_surface, rcond=None)[0]
        # Pooled least squares over the stacked target columns.  Identical designs make
        # the normal equations collapse onto the across-target mean response.
        blind = np.linalg.lstsq(
            train_design, train_surface.mean(axis=1), rcond=None
        )[0]
        pooled_versus_mean = max(
            pooled_versus_mean, float(np.abs(blind - specific.mean(axis=1)).max())
        )

        oof_surface[test_index] = test_surface
        oof_specific[test_index] = test_design @ specific
        oof_blind[test_index] = np.outer(test_design @ blind, np.ones(n_targets))
        fold_index[test_index] = fold
        coefficients[fold] = specific
        pooled[fold] = blind
        design_gram[fold] = test_design.T @ test_design
        design_sums[fold] = test_design.sum(axis=0)
        gram[fold] = specific.T @ design_gram[fold] @ specific
        column_sums[fold] = specific.T @ design_sums[fold]
        fold_rows.append(
            {
                "fold": fold + 1,
                "train_ligands": int(len(train_index)),
                "test_ligands": int(len(test_index)),
                "train_groups": int(len(set(groups[train_index].tolist()))),
                "test_groups": int(len(set(groups[test_index].tolist()))),
                **scale_record,
            }
        )

    if (fold_index < 0).any():
        raise AssertionError("every ligand must occur in exactly one test fold")
    for name, surface in (
        ("observed", oof_surface),
        ("specific", oof_specific),
        ("blind", oof_blind),
    ):
        if not np.isfinite(surface).all():
            raise AssertionError(f"the out-of-fold {name} surface was not filled")

    return {
        "oof_surface": oof_surface,
        "oof_specific": oof_specific,
        "oof_blind": oof_blind,
        "fold_index": fold_index,
        "coefficients": coefficients,
        "pooled_coefficients": pooled,
        "gram": gram,
        "column_sums": column_sums,
        "design_gram": design_gram,
        "design_sums": design_sums,
        "fold_rows": fold_rows,
        "pooled_equals_mean_of_per_target_max_abs_difference": pooled_versus_mean,
    }


def variance_split(
    oof_specific: np.ndarray, oof_blind: np.ndarray, fold_index: np.ndarray
) -> dict[str, object]:
    """Exact orthogonal split of the descriptor component's total sum of squares."""
    shared_ss = float(np.square(oof_blind).sum())
    heterogeneity = oof_specific - oof_blind
    heterogeneity_ss = float(np.square(heterogeneity).sum())
    total_ss = float(np.square(oof_specific).sum())
    cross = float(2.0 * (oof_blind * heterogeneity).sum())
    per_fold = []
    for fold in np.unique(fold_index):
        mask = fold_index == fold
        shared_fold = float(np.square(oof_blind[mask]).sum())
        hetero_fold = float(np.square(heterogeneity[mask]).sum())
        per_fold.append(
            {
                "fold": int(fold) + 1,
                "shared_sum_of_squares": shared_fold,
                "heterogeneity_sum_of_squares": hetero_fold,
                "heterogeneity_share": hetero_fold / (shared_fold + hetero_fold),
            }
        )
    return {
        "definition": (
            "the target-specific descriptor component splits as d(i).b + d(i).(B_j - b) "
            "with b the shared (pooled) coefficient vector; because b is the mean of the "
            "per-target vectors the cross term is exactly zero"
        ),
        "total_sum_of_squares": total_ss,
        "shared_coefficient_sum_of_squares": shared_ss,
        "between_target_coefficient_heterogeneity_sum_of_squares": heterogeneity_ss,
        "cross_term_sum_of_squares": cross,
        "cross_term_relative_to_total": abs(cross) / total_ss,
        "shared_share_of_descriptor_component_variance": shared_ss
        / (shared_ss + heterogeneity_ss),
        "heterogeneity_share_of_descriptor_component_variance": heterogeneity_ss
        / (shared_ss + heterogeneity_ss),
        "per_fold": per_fold,
    }


def out_of_fold_target_r2(observed: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    """Per-target out-of-fold R^2, the same statistic the published arm reports."""
    residual = np.square(observed - prediction).sum(axis=0)
    total = np.square(observed - observed.mean(axis=0)).sum(axis=0)
    return 1.0 - residual / total


def coefficient_spread(coefficients: np.ndarray) -> pd.DataFrame:
    """Across-target spread of every fitted slope, averaged over folds and per fold."""
    fold_mean = coefficients.mean(axis=0)  # (8, 20)
    rows: list[dict[str, object]] = []
    for index, name in enumerate(COEFFICIENT_NAMES):
        values = fold_mean[index]
        per_fold_sd = coefficients[:, index, :].std(axis=1, ddof=1)
        rows.append(
            {
                "coefficient": name,
                "shared_target_blind_value": float(values.mean()),
                "across_target_sd": float(values.std(ddof=1)),
                "across_target_minimum": float(values.min()),
                "across_target_maximum": float(values.max()),
                "across_target_range": float(values.max() - values.min()),
                "heterogeneity_to_shared_ratio": (
                    float(values.std(ddof=1) / abs(values.mean()))
                    if abs(values.mean()) > 1e-12
                    else None
                ),
                "minimum_fold_across_target_sd": float(per_fold_sd.min()),
                "maximum_fold_across_target_sd": float(per_fold_sd.max()),
            }
        )
    return pd.DataFrame.from_records(rows)


# --------------------------------------------------------------------------------------
# geometry helpers
# --------------------------------------------------------------------------------------


def _triangle() -> tuple[np.ndarray, np.ndarray]:
    return np.triu_indices(len(TARGETS20), k=1)


def correlation_from_moments(
    cross_products: np.ndarray, sums: np.ndarray, weight: float
) -> np.ndarray:
    """Correlation matrix from weighted second moments (matches np.corrcoef exactly)."""
    covariance = cross_products / weight - np.outer(sums, sums) / weight**2
    scale = np.sqrt(np.diag(covariance))
    if np.any(scale <= 1e-14):
        raise ValueError("a target column has negligible variance")
    correlation = np.clip(covariance / np.outer(scale, scale), -1.0, 1.0)
    np.fill_diagonal(correlation, 1.0)
    return correlation


def experimental_rank_vectors(ledger: pd.DataFrame) -> "OrderedDict[str, np.ndarray]":
    """Centred, unit-norm rank vectors so Spearman is a single dot product."""
    vectors: OrderedDict[str, np.ndarray] = OrderedDict()
    for panel in PANELS:
        ranks = stats.rankdata(
            ledger[f"{panel}_experimental_centered_correlation"].to_numpy(dtype=np.float64),
            method="average",
        )
        ranks = ranks - ranks.mean()
        vectors[panel] = ranks / np.linalg.norm(ranks)
    return vectors


def spearman_against(edges: np.ndarray, rank_vector: np.ndarray) -> float | None:
    ranks = stats.rankdata(edges, method="average")
    ranks = ranks - ranks.mean()
    norm = float(np.linalg.norm(ranks))
    if norm <= 1e-12:
        return None  # constant predictor: Spearman is undefined, not zero
    return float(ranks @ rank_vector / norm)


def degeneracy_report(correlation: np.ndarray) -> dict[str, object]:
    triangle = _triangle()
    edges = correlation[triangle]
    rounded = np.round(edges, 12)
    return {
        "target_pairs": int(len(edges)),
        "distinct_edge_values_at_1e-12": int(len(np.unique(rounded))),
        "minimum_edge": float(edges.min()),
        "maximum_edge": float(edges.max()),
        "maximum_absolute_deviation_from_plus_one": float(np.abs(edges - 1.0).max()),
        "is_degenerate": bool(len(np.unique(rounded)) == 1),
    }


# --------------------------------------------------------------------------------------
# slope-permutation nulls
# --------------------------------------------------------------------------------------


def slope_permutation_nulls(
    gram: np.ndarray,
    column_sums: np.ndarray,
    n_rows: int,
    observed_geometry: np.ndarray,
    rank_vectors: "OrderedDict[str, np.ndarray]",
    observed: dict[str, float],
    permutations: int,
    seed: int,
) -> tuple[dict[str, object], pd.DataFrame]:
    """Keep the fitted slope vectors; destroy only their pairing with targets.

    Permuting the columns of the per-fold coefficient array permutes the columns of the
    predicted surface, so the whole permuted geometry is obtainable from per-fold 20x20
    second-moment matrices without rebuilding a 259,579-row surface.  Two variants:

    ``shared_permutation``  one permutation used in every fold.  This is exactly a
    relabelling of the target axis, hence algebraically identical to the target-label QAP;
    the identity is verified numerically and reported.

    ``fold_independent_permutation``  an independent permutation per fold.  This is not a
    relabelling: each target column becomes a different target's slope vector on each
    fold's block of ligands, so cross-fold coherence of the slope-to-target pairing is
    destroyed as well.
    """
    triangle = _triangle()
    n_targets = gram.shape[1]
    rng = np.random.default_rng(seed)
    variants = ("shared_permutation", "fold_independent_permutation")
    null = {variant: {panel: np.empty(permutations) for panel in PANELS} for variant in variants}
    relabel_check = 0.0

    for repetition in range(permutations):
        order = rng.permutation(n_targets)
        cross = gram[:, order][:, :, order].sum(axis=0)
        sums = column_sums[:, order].sum(axis=0)
        shared_geometry = correlation_from_moments(cross, sums, float(n_rows))
        relabel_check = max(
            relabel_check,
            float(
                np.abs(
                    shared_geometry - observed_geometry[np.ix_(order, order)]
                ).max()
            ),
        )
        edges = shared_geometry[triangle]
        for panel in PANELS:
            null["shared_permutation"][panel][repetition] = spearman_against(
                edges, rank_vectors[panel]
            )

        cross = np.zeros((n_targets, n_targets))
        sums = np.zeros(n_targets)
        for fold in range(len(gram)):
            fold_order = rng.permutation(n_targets)
            cross += gram[fold][np.ix_(fold_order, fold_order)]
            sums += column_sums[fold][fold_order]
        edges = correlation_from_moments(cross, sums, float(n_rows))[triangle]
        for panel in PANELS:
            null["fold_independent_permutation"][panel][repetition] = spearman_against(
                edges, rank_vectors[panel]
            )

    rows: list[dict[str, object]] = []
    report: dict[str, object] = {
        "permutations": int(permutations),
        "seed": int(seed),
        "unit": "the fitted 8-vector of regression coefficients of one target",
        "shared_permutation_is_a_target_relabelling": {
            "maximum_absolute_geometry_difference_from_direct_relabelling": relabel_check,
            "interpretation": (
                "identical to machine precision, so the shared-permutation null is the "
                "target-label QAP already reported for this arm and is not independent "
                "evidence"
            ),
        },
        "panels": {},
    }
    for variant in variants:
        report["panels"][variant] = {}
        for panel in PANELS:
            values = null[variant][panel]
            point = observed[panel]
            record = {
                "observed_spearman": point,
                "null_mean": float(values.mean()),
                "null_sd": float(values.std(ddof=1)),
                "null_interval_95": [
                    float(np.quantile(values, 0.025)),
                    float(np.quantile(values, 0.975)),
                ],
                "one_sided_p_positive": float(
                    (1 + np.sum(values >= point)) / (permutations + 1)
                ),
            }
            report["panels"][variant][panel] = record
            rows.append({"null": variant, "panel": panel, **record})
    return report, pd.DataFrame.from_records(rows)


# --------------------------------------------------------------------------------------
# coefficient-rank ladder
# --------------------------------------------------------------------------------------


def coefficient_rank_ladder(
    coefficients: np.ndarray,
    gram_design: list[np.ndarray],
    design_sums: list[np.ndarray],
    fold_sizes: list[int],
    rank_vectors: "OrderedDict[str, np.ndarray]",
    total_sum_of_squares: float,
) -> tuple[list[dict[str, object]], pd.DataFrame, dict[str, object]]:
    """Agreement as a function of the rank of the between-target coefficient deviation.

    ``k = 0`` is the target-blind arm.  Each extra unit of rank is one more direction in
    which targets are allowed to respond differently to the same seven descriptors.
    """
    triangle = _triangle()
    n_targets = coefficients.shape[2]
    n_rows = float(sum(fold_sizes))
    ladder: list[dict[str, object]] = []
    rows: list[dict[str, object]] = []
    # The design has an intercept plus seven descriptors, so the nominal coefficient
    # deviation matrix is 8 x 20.  The fold-local residual transform makes every training
    # target mean zero, however, and the fitted intercept deviations are only numerical
    # noise (~1e-12 in this artifact).  Calling that noise an eighth scientific direction
    # made the old ladder one rung too long.  Use an explicit relative tolerance and report
    # the discarded singular value rather than relying on np.linalg.matrix_rank's
    # machine-epsilon default.
    spectra: list[dict[str, object]] = []
    effective_ranks: list[int] = []
    nominal_max_rank = min(coefficients.shape[1], n_targets - 1)
    for fold in range(len(coefficients)):
        specific = coefficients[fold]
        deviation = specific - specific.mean(axis=1, keepdims=True)
        singular = np.linalg.svd(deviation, compute_uv=False)
        threshold = float(singular[0] * EFFECTIVE_SVD_RTOL)
        effective = int(np.sum(singular > threshold))
        effective_ranks.append(effective)
        spectra.append(
            {
                "fold": fold + 1,
                "singular_values": singular.tolist(),
                "relative_tolerance": EFFECTIVE_SVD_RTOL,
                "absolute_threshold": threshold,
                "effective_rank": effective,
                "largest_discarded_singular_value": (
                    float(singular[effective]) if effective < len(singular) else 0.0
                ),
            }
        )
    max_rank = max(effective_ranks)
    for rank in range(max_rank + 1):
        cross = np.zeros((n_targets, n_targets))
        sums = np.zeros(n_targets)
        retained = 0.0
        reconstruction_error = 0.0
        for fold in range(len(coefficients)):
            specific = coefficients[fold]
            shared = specific.mean(axis=1)
            deviation = specific - shared[:, None]
            if rank == 0:
                approximation = np.zeros_like(deviation)
            else:
                left, singular, right = np.linalg.svd(deviation, full_matrices=False)
                approximation = (left[:, :rank] * singular[:rank]) @ right[:rank]
            reconstruction_error = max(
                reconstruction_error,
                float(np.max(np.abs(deviation - approximation))),
            )
            truncated = shared[:, None] + approximation
            cross_fold = truncated.T @ gram_design[fold] @ truncated
            cross += cross_fold
            sums += truncated.T @ design_sums[fold]
            retained += float(np.trace(cross_fold))
        try:
            correlation = correlation_from_moments(cross, sums, n_rows)
        except ValueError:
            correlation = None
        record: dict[str, object] = {
            "coefficient_deviation_rank": int(rank),
            "variance_share_of_full_descriptor_component": retained / total_sum_of_squares,
            "maximum_absolute_coefficient_reconstruction_error": reconstruction_error,
        }
        if correlation is None:
            record["degenerate"] = True
            record["distinct_edge_values_at_1e-12"] = 0
            for panel in PANELS:
                record[f"{panel}_spearman"] = None
        else:
            edges = correlation[triangle]
            record["degenerate"] = bool(len(np.unique(np.round(edges, 12))) == 1)
            record["distinct_edge_values_at_1e-12"] = int(
                len(np.unique(np.round(edges, 12)))
            )
            for panel in PANELS:
                record[f"{panel}_spearman"] = spearman_against(edges, rank_vectors[panel])
        ladder.append(record)
        rows.append(record)
    diagnostics = {
        "nominal_maximum_rank_from_matrix_shape": int(nominal_max_rank),
        "effective_maximum_rank": int(max_rank),
        "effective_rank_relative_tolerance": EFFECTIVE_SVD_RTOL,
        "per_fold_singular_spectra": spectra,
        "interpretation": (
            "the apparent eighth direction is a fold-local intercept residual at about "
            "1e-12; rank 7 reconstructs the fitted target-specific coefficient surface "
            "to the reported numerical tolerance"
        ),
    }
    return ladder, pd.DataFrame.from_records(rows), diagnostics


# --------------------------------------------------------------------------------------
# chemical-cluster bootstrap
# --------------------------------------------------------------------------------------


def cluster_moment_representation(
    surface: np.ndarray, cluster_index: np.ndarray, n_clusters: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-cluster second moments, so a weighted correlation costs one matrix product."""
    n_targets = surface.shape[1]
    cross = np.empty((n_clusters, n_targets * n_targets))
    for first in range(n_targets):
        for second in range(first, n_targets):
            values = np.bincount(
                cluster_index,
                weights=surface[:, first] * surface[:, second],
                minlength=n_clusters,
            )
            cross[:, first * n_targets + second] = values
            cross[:, second * n_targets + first] = values
    sums = np.column_stack(
        [
            np.bincount(cluster_index, weights=surface[:, column], minlength=n_clusters)
            for column in range(n_targets)
        ]
    )
    counts = np.bincount(cluster_index, minlength=n_clusters).astype(np.float64)
    return cross, sums, counts


def cluster_bootstrap(
    surfaces: "OrderedDict[str, np.ndarray]",
    cluster_labels: np.ndarray,
    rank_vectors: "OrderedDict[str, np.ndarray]",
    shared_component: np.ndarray,
    heterogeneity_component: np.ndarray,
    repeats: int,
    seed: int,
) -> dict[str, object]:
    """Bayesian bootstrap over chemical clusters, keeping the target panel fixed.

    Exponential cluster weights are inherited by every member ligand, exactly as in
    ``residual_target_geometry_validation.geometry_bootstrap``; the fast path here is
    validated against that function on the first draws.
    """
    labels = np.asarray(cluster_labels)
    _, cluster_index = np.unique(labels, return_inverse=True)
    n_clusters = int(cluster_index.max() + 1)
    triangle = _triangle()
    n_targets = len(TARGETS20)

    moments = OrderedDict(
        (name, cluster_moment_representation(surface, cluster_index, n_clusters))
        for name, surface in surfaces.items()
    )
    shared_rowsum = np.bincount(
        cluster_index, weights=np.square(shared_component).sum(axis=1), minlength=n_clusters
    )
    hetero_rowsum = np.bincount(
        cluster_index,
        weights=np.square(heterogeneity_component).sum(axis=1),
        minlength=n_clusters,
    )

    draws = {
        name: {panel: np.empty(repeats) for panel in PANELS} for name in surfaces
    }
    heterogeneity_share = np.empty(repeats)
    validation = 0.0

    rng = np.random.default_rng(seed)
    position = 0
    while position < repeats:
        size = min(BOOTSTRAP_CHUNK, repeats - position)
        weights = rng.exponential(size=(size, n_clusters))
        for name, (cross, sums, counts) in moments.items():
            block_cross = weights @ cross
            block_sums = weights @ sums
            block_weight = weights @ counts
            for row in range(size):
                correlation = correlation_from_moments(
                    block_cross[row].reshape(n_targets, n_targets),
                    block_sums[row],
                    float(block_weight[row]),
                )
                edges = correlation[triangle]
                for panel in PANELS:
                    draws[name][panel][position + row] = spearman_against(
                        edges, rank_vectors[panel]
                    )
                if position + row < 3:
                    reference = geometry.weighted_target_correlation(
                        surfaces[name], weights[row][cluster_index]
                    )
                    validation = max(
                        validation, float(np.abs(reference - correlation).max())
                    )
        block_shared = weights @ shared_rowsum
        block_hetero = weights @ hetero_rowsum
        heterogeneity_share[position : position + size] = block_hetero / (
            block_shared + block_hetero
        )
        position += size

    contrast = {
        panel: draws[ARM_OBSERVED][panel] - draws[ARM_SPECIFIC][panel] for panel in PANELS
    }

    report: dict[str, object] = {
        "method": "Bayesian bootstrap with iid Exp(1) weights, one weight per cluster",
        "unit": "chemical cluster",
        "clusters": n_clusters,
        "repeats": int(repeats),
        "seed": int(seed),
        "fixed_target_panel": len(TARGETS20),
        "fast_path_maximum_absolute_deviation_from_weighted_target_correlation": validation,
        "arms": {
            name: {panel: geometry.describe(draws[name][panel]) for panel in PANELS}
            for name in surfaces
        },
        "observed_minus_target_specific": {
            panel: geometry.describe(contrast[panel]) for panel in PANELS
        },
        "heterogeneity_share_of_descriptor_component_variance": geometry.describe(
            heterogeneity_share
        ),
    }
    return report


def conservative_union(intervals: list[list[float]]) -> list[float]:
    return [
        float(min(interval[0] for interval in intervals)),
        float(max(interval[1] for interval in intervals)),
    ]


# --------------------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------------------


def evaluate(
    block: np.ndarray,
    descriptors: np.ndarray,
    murcko: np.ndarray,
    ledger: pd.DataFrame,
    *,
    folds: int,
    qap_permutations: int,
    slope_permutations: int,
    bootstrap_repeats: int,
    seed: int,
    published_arm: Path | None,
) -> tuple[dict[str, object], dict[str, pd.DataFrame]]:
    triangle = _triangle()
    rank_vectors = experimental_rank_vectors(ledger)
    decomposition = shared_and_specific_decomposition(block, descriptors, murcko, folds)

    surfaces = OrderedDict(
        [
            (ARM_OBSERVED, decomposition["oof_surface"]),
            (ARM_SPECIFIC, decomposition["oof_specific"]),
            (ARM_BLIND, decomposition["oof_blind"]),
        ]
    )

    # --- verification against the published estimator -------------------------------
    published = mechanism.grouped_descriptor_decomposition(
        block, descriptors, murcko, folds
    )
    reproduction = {
        "published_estimator": (
            "residual_mechanism_analysis.grouped_descriptor_decomposition"
        ),
        "maximum_absolute_difference_observed_surface": float(
            np.abs(published["oof_surface"] - surfaces[ARM_OBSERVED]).max()
        ),
        "maximum_absolute_difference_descriptor_component": float(
            np.abs(published["oof_prediction"] - surfaces[ARM_SPECIFIC]).max()
        ),
    }
    if reproduction["maximum_absolute_difference_descriptor_component"] > 1e-9:
        raise AssertionError(
            "the target-specific arm does not reproduce the published descriptor component"
        )
    del published

    geometries: "OrderedDict[str, np.ndarray]" = OrderedDict()
    for name, surface in surfaces.items():
        # target_correlation refuses a constant column; the shared-coefficient surface has
        # 20 identical (non-constant) columns, which it accepts and returns as all ones.
        matrix = np.clip(geometry.target_correlation(surface), -1.0, 1.0)
        np.fill_diagonal(matrix, 1.0)
        geometries[name] = matrix

    ledger_docking = ledger["docking_local_20_centered_correlation"].to_numpy(
        dtype=np.float64
    )
    ledger_geometry = np.eye(len(TARGETS20))
    ledger_geometry[triangle] = ledger_docking
    ledger_geometry[(triangle[1], triangle[0])] = ledger_docking
    reconstruction = geometry.geometry_concordance(
        geometries[ARM_OBSERVED], ledger_geometry
    )
    if reconstruction < 0.99:
        raise ValueError(
            "the reconstructed fixed-20 docking geometry does not match the released "
            f"ledger (Spearman {reconstruction:.4f}); support or target order drifted"
        )

    # --- point agreement --------------------------------------------------------------
    agreement: "OrderedDict[str, dict[str, float | None]]" = OrderedDict()
    qap: dict[str, dict[str, object]] = {}
    for arm_index, (name, matrix) in enumerate(geometries.items()):
        edges = matrix[triangle]
        agreement[name] = {
            panel: spearman_against(edges, rank_vectors[panel]) for panel in PANELS
        }
        qap[name] = {}
        degenerate = len(np.unique(np.round(edges, 12))) == 1
        for panel_index, panel in enumerate(PANELS):
            if degenerate:
                qap[name][panel] = None
                continue
            experimental = np.eye(len(TARGETS20))
            values = ledger[f"{panel}_experimental_centered_correlation"].to_numpy(
                dtype=np.float64
            )
            experimental[triangle] = values
            experimental[(triangle[1], triangle[0])] = values
            test = geometry.qap_test(
                matrix,
                experimental,
                qap_permutations,
                seed + panel_index * 1_000_000 + arm_index * 10_000,
            )
            qap[name][panel] = {
                "spearman": test["observed_spearman"],
                "one_sided_p_positive": test["one_sided_p_positive"],
                "null_interval_95": test["null_interval_95"],
                "seed": test["seed"],
            }

    blind_degeneracy = degeneracy_report(geometries[ARM_BLIND])
    if not blind_degeneracy["is_degenerate"]:
        raise AssertionError(
            "the shared-coefficient arm must produce one distinct target-pair value"
        )

    # --- variance split, slopes, ladder ------------------------------------------------
    split = variance_split(
        surfaces[ARM_SPECIFIC], surfaces[ARM_BLIND], decomposition["fold_index"]
    )
    spread = coefficient_spread(decomposition["coefficients"])

    specific_r2 = out_of_fold_target_r2(surfaces[ARM_OBSERVED], surfaces[ARM_SPECIFIC])
    blind_r2 = out_of_fold_target_r2(surfaces[ARM_OBSERVED], surfaces[ARM_BLIND])
    predictive = {
        ARM_SPECIFIC: {
            "mean_out_of_fold_target_r2": float(specific_r2.mean()),
            "median_out_of_fold_target_r2": float(np.median(specific_r2)),
            "minimum_out_of_fold_target_r2": float(specific_r2.min()),
            "maximum_out_of_fold_target_r2": float(specific_r2.max()),
            "targets_with_positive_r2": int((specific_r2 > 0).sum()),
        },
        ARM_BLIND: {
            "mean_out_of_fold_target_r2": float(blind_r2.mean()),
            "median_out_of_fold_target_r2": float(np.median(blind_r2)),
            "minimum_out_of_fold_target_r2": float(blind_r2.min()),
            "maximum_out_of_fold_target_r2": float(blind_r2.max()),
            "targets_with_positive_r2": int((blind_r2 > 0).sum()),
        },
        "interpretation": (
            "R^2 is against the same out-of-fold residual surface, so the two arms differ "
            "only in whether the seven descriptors are allowed a separate slope vector per "
            "target; a negative value means the arm predicts worse than that target's own "
            "out-of-fold mean"
        ),
    }
    predictive_frame = pd.DataFrame.from_records(
        [
            {
                "target": target,
                "target_specific_out_of_fold_r2": float(specific_r2[index]),
                "target_blind_out_of_fold_r2": float(blind_r2[index]),
                "difference": float(specific_r2[index] - blind_r2[index]),
            }
            for index, target in enumerate(TARGETS20)
        ]
    )

    fold_sizes = [int(row["test_ligands"]) for row in decomposition["fold_rows"]]
    ladder, ladder_frame, ladder_diagnostics = coefficient_rank_ladder(
        decomposition["coefficients"],
        decomposition["design_gram"],
        decomposition["design_sums"],
        fold_sizes,
        rank_vectors,
        split["total_sum_of_squares"],
    )

    # --- slope permutation nulls --------------------------------------------------------
    null_report, null_frame = slope_permutation_nulls(
        decomposition["gram"],
        decomposition["column_sums"],
        len(block),
        geometries[ARM_SPECIFIC],
        rank_vectors,
        {panel: agreement[ARM_SPECIFIC][panel] for panel in PANELS},
        slope_permutations,
        seed + 500_000,
    )

    # --- chemical-cluster bootstrap -----------------------------------------------------
    generic, generic_failures = generic_framework_keys(murcko)
    bootstrap_surfaces = OrderedDict(
        [
            (ARM_OBSERVED, surfaces[ARM_OBSERVED]),
            (ARM_SPECIFIC, surfaces[ARM_SPECIFIC]),
        ]
    )
    heterogeneity_component = surfaces[ARM_SPECIFIC] - surfaces[ARM_BLIND]
    bootstrap = OrderedDict(
        [
            (
                "murcko_scaffold_clusters",
                cluster_bootstrap(
                    bootstrap_surfaces,
                    murcko,
                    rank_vectors,
                    surfaces[ARM_BLIND],
                    heterogeneity_component,
                    bootstrap_repeats,
                    seed + 700_000,
                ),
            ),
            (
                "generic_murcko_framework_clusters",
                cluster_bootstrap(
                    bootstrap_surfaces,
                    generic,
                    rank_vectors,
                    surfaces[ARM_BLIND],
                    heterogeneity_component,
                    bootstrap_repeats,
                    seed + 800_000,
                ),
            ),
        ]
    )

    conservative: dict[str, object] = {"arms": {}, "observed_minus_target_specific": {}}
    for name in bootstrap_surfaces:
        conservative["arms"][name] = {
            panel: conservative_union(
                [bootstrap[definition]["arms"][name][panel]["interval_90"] for definition in bootstrap]
            )
            for panel in PANELS
        }
    conservative["observed_minus_target_specific"] = {
        panel: conservative_union(
            [
                bootstrap[definition]["observed_minus_target_specific"][panel]["interval_90"]
                for definition in bootstrap
            ]
        )
        for panel in PANELS
    }
    conservative["heterogeneity_share_of_descriptor_component_variance"] = (
        conservative_union(
            [
                bootstrap[definition][
                    "heterogeneity_share_of_descriptor_component_variance"
                ]["interval_90"]
                for definition in bootstrap
            ]
        )
    )

    # --- optional cross-check against the released sibling artifact ----------------------
    published_check: dict[str, object] = {"available": False}
    if published_arm is not None and published_arm.exists():
        released = json.loads(published_arm.read_text())["experimental_panel_results"]
        deviations = {
            panel: {
                "observed_residual": abs(
                    released[panel]["observed_residual"]["spearman"]
                    - agreement[ARM_OBSERVED][panel]
                ),
                "descriptor_component": abs(
                    released[panel]["descriptor_component"]["spearman"]
                    - agreement[ARM_SPECIFIC][panel]
                ),
            }
            for panel in PANELS
        }
        worst = max(
            value for record in deviations.values() for value in record.values()
        )
        published_check = {
            "available": True,
            "artifact": str(published_arm.relative_to(PACKAGE)),
            "maximum_absolute_spearman_deviation": float(worst),
            "per_panel_absolute_deviation": deviations,
        }
        if worst > 1e-9:
            raise AssertionError(
                "the reference arms do not reproduce results/descriptor_component_geometry"
            )

    # --- tables --------------------------------------------------------------------------
    panel_rows: list[dict[str, object]] = []
    for name in geometries:
        for panel in PANELS:
            record: dict[str, object] = {
                "arm": name,
                "panel": panel,
                "degenerate_geometry": bool(name == ARM_BLIND),
                "spearman": agreement[name][panel],
                "target_label_qap_p_positive": (
                    qap[name][panel]["one_sided_p_positive"]
                    if qap[name][panel] is not None
                    else None
                ),
            }
            if name in bootstrap_surfaces:
                for definition in bootstrap:
                    interval = bootstrap[definition]["arms"][name][panel]["interval_90"]
                    record[f"{definition}_lower_90"] = interval[0]
                    record[f"{definition}_upper_90"] = interval[1]
                union = conservative["arms"][name][panel]
                record["conservative_union_lower_90"] = union[0]
                record["conservative_union_upper_90"] = union[1]
            panel_rows.append(record)

    pair_rows = []
    for edge in range(len(ledger)):
        row = {
            "target_a": ledger.target_a.iloc[edge],
            "target_b": ledger.target_b.iloc[edge],
        }
        for name, matrix in geometries.items():
            row[f"docking_{name}_correlation"] = matrix[triangle][edge]
        for panel in PANELS:
            row[f"{panel}_experimental_centered_correlation"] = ledger[
                f"{panel}_experimental_centered_correlation"
            ].iloc[edge]
        pair_rows.append(row)

    coefficient_rows = []
    for fold in range(folds):
        for target_index, target in enumerate(TARGETS20):
            record = {"fold": fold + 1, "target": target}
            for index, name in enumerate(COEFFICIENT_NAMES):
                record[name] = float(
                    decomposition["coefficients"][fold, index, target_index]
                )
            coefficient_rows.append(record)
        record = {"fold": fold + 1, "target": "SHARED_TARGET_BLIND"}
        for index, name in enumerate(COEFFICIENT_NAMES):
            record[name] = float(decomposition["pooled_coefficients"][fold, index])
        coefficient_rows.append(record)

    summary: dict[str, object] = {
        "analysis": (
            "target-blind (shared-coefficient) control for the descriptor component of "
            "the fixed common-20 kinase residual surface"
        ),
        "question": (
            "The descriptor model fits a separate slope vector per target, so target "
            "identity can enter the 'descriptor' surface through those slopes. How much "
            "of the descriptor component's variance, and of its agreement with "
            "experimental kinase co-response, survives when the coefficients are shared "
            "across the whole panel and the model is therefore genuinely target-blind?"
        ),
        "status": "exploratory_post_hoc_reviewer_requested_control",
        "arms": dict(ARM_LABELS),
        "descriptors": list(mechanism.DESCRIPTOR_NAMES),
        "coefficients_per_target": list(COEFFICIENT_NAMES),
        "targets": list(TARGETS20),
        "target_pairs": int(len(ledger)),
        "headline": {
            "target_blind_geometry_is_degenerate_by_construction": True,
            "target_blind_distinct_target_pair_values": blind_degeneracy[
                "distinct_edge_values_at_1e-12"
            ],
            "target_blind_spearman_with_every_panel": None,
            "why": (
                "with one shared slope vector b the prediction for ligand i at target j is "
                "d(i).b for every j, so all 20 predicted target columns are the same vector "
                "of ligand scores, all 190 target-pair correlations equal +1 exactly, and "
                "Spearman against any experimental panel is undefined rather than zero"
            ),
            "share_of_descriptor_component_variance_needing_per_target_slopes": split[
                "heterogeneity_share_of_descriptor_component_variance"
            ],
            "mean_out_of_fold_target_r2_target_specific": predictive[ARM_SPECIFIC][
                "mean_out_of_fold_target_r2"
            ],
            "mean_out_of_fold_target_r2_target_blind": predictive[ARM_BLIND][
                "mean_out_of_fold_target_r2"
            ],
            "conclusion": (
                "the descriptor component's target-pair geometry is carried entirely by "
                "the fitted per-target slope vectors; the surface is not target-blind"
            ),
        },
        "target_blind_degeneracy": blind_degeneracy,
        "out_of_fold_predictive_r2": predictive,
        "descriptor_component_variance_split": split,
        "panel_agreement": {
            name: {
                panel: {
                    "spearman": agreement[name][panel],
                    "target_label_qap": qap[name][panel],
                }
                for panel in PANELS
            }
            for name in geometries
        },
        "slope_permutation_null": null_report,
        "coefficient_rank_ladder": {
            "definition": (
                "the per-target coefficient array is replaced by its shared mean plus the "
                "rank-k truncation of its between-target deviation, fold-locally; k=0 is "
                "the target-blind arm and effective k=7 reconstructs the full published "
                "arm; the nominal eighth intercept direction is reported as numerical noise"
            ),
            "diagnostics": ladder_diagnostics,
            "rungs": ladder,
        },
        "coefficient_rank_7_svd_comparator": {
            "role": (
                "predictive coefficient-space comparator built only from the seven ligand "
                "descriptors; this is distinct from the outcome-derived rank-7 SVD "
                "transported-subspace oracle reconstruction reported by "
                "descriptor_rank_matched_controls"
            ),
            "rung": ladder[-1],
        },
        "chemical_cluster_bootstrap": {
            "definitions": {
                "murcko_scaffold_clusters": (
                    "RDKit Bemis-Murcko scaffolds, the same grouping used for the holdout "
                    "folds; every acyclic molecule is a singleton"
                ),
                "generic_murcko_framework_clusters": (
                    "Bemis-Murcko generic frameworks (atom types and bond orders erased), "
                    "a strictly coarser chemical grouping"
                ),
            },
            "generic_framework_failures": int(generic_failures),
            "variance_share_note": (
                "the bootstrapped heterogeneity share reweights the per-ligand shared and "
                "heterogeneity sums of squares; the cross term is exactly zero only at the "
                "fitted (unit) weights, so the resampled share is defined as "
                "H(w) / (S(w) + H(w))"
            ),
            "butina_note": (
                "Butina clustering is not computable at this support: the package helper "
                "dense_davis_benchmark._butina_labels materialises the complete pairwise "
                "Tanimoto distance list, which is 3.37e10 entries for 259,579 ligands. The "
                "generic Murcko framework is used as the coarse second definition and the "
                "reported interval is the conservative union of the two, in place of the "
                "Murcko/Butina union used on the smaller cohorts."
            ),
            "per_definition": bootstrap,
            "conservative_union_90": conservative,
        },
        "verification": {
            "reference_arm_reproduces_published_estimator": reproduction,
            "observed_arm_versus_released_ledger_spearman": reconstruction,
            "released_ledger": str(DEFAULT_LEDGER.relative_to(PACKAGE)),
            "pooled_equals_mean_of_per_target_coefficients_max_abs_difference": (
                decomposition["pooled_equals_mean_of_per_target_max_abs_difference"]
            ),
            "published_descriptor_component_geometry_cross_check": published_check,
        },
        "configuration": {
            "folds": int(folds),
            "group_definition": (
                "RDKit Bemis-Murcko scaffolds; every acyclic molecule is a singleton"
            ),
            "fold_summary": decomposition["fold_rows"],
            "qap_permutations": int(qap_permutations),
            "slope_permutations": int(slope_permutations),
            "bootstrap_repeats": int(bootstrap_repeats),
            "fixed_endpoint": "two-way-centered experimental target correlation",
            "experimental_geometry_source": str(DEFAULT_LEDGER.relative_to(PACKAGE)),
        },
        "seeds": {
            "base": int(seed),
            "target_label_qap": (
                "base + 1,000,000*panel_index + 10,000*arm_index"
            ),
            "slope_permutation": int(seed + 500_000),
            "murcko_cluster_bootstrap": int(seed + 700_000),
            "generic_framework_cluster_bootstrap": int(seed + 800_000),
            "fold_assignment": "GroupKFold is deterministic and takes no seed",
        },
        "claim_boundary": (
            "This control does NOT show that the residual target geometry is target-blind; "
            "it shows the opposite, and it does not rescue that wording. The shared-"
            "coefficient arm is degenerate BY CONSTRUCTION, not empirically null: one slope "
            "vector makes all 20 predicted target columns identical, so its 190 target-pair "
            "correlations are all exactly +1 and its Spearman agreement with experiment is "
            "undefined, not zero. Reporting that number as evidence would be circular. This "
            "analysis also does NOT test whether some genuinely target-blind model built "
            "from protein-side features (sequence, pocket geometry, contact area) could "
            "reproduce the geometry: no such model is fitted here, so nothing is concluded "
            "about it. The slope-permutation null with one permutation shared across folds "
            "is algebraically identical to the target-label QAP already reported and is "
            "therefore a restatement of that test rather than independent evidence; only "
            "the fold-independent variant adds anything, and it too keeps the fitted slope "
            "marginals. The variance split is a decomposition of a fitted surface, not a "
            "causal attribution, and the rank ladder is descriptive: rank k is a property "
            "of the fitted coefficient array, not a count of physical mechanisms. All "
            "numbers are post hoc on one fixed 20-kinase panel and one de-leaked DOCKSTRING "
            "ligand support; QAP probabilities and bootstrap intervals are conditional on "
            "that panel and that support. Butina clustering is infeasible at 259,579 "
            "ligands, so the conservative interval unions Murcko-scaffold and generic-"
            "Murcko-framework bootstraps rather than Murcko and Butina."
        ),
    }
    tables = {
        "panel_agreement.csv": pd.DataFrame.from_records(panel_rows),
        "target_pair_geometry.csv": pd.DataFrame.from_records(pair_rows),
        "coefficient_spread.csv": spread,
        "target_predictive_r2.csv": predictive_frame,
        "fold_target_coefficients.csv": pd.DataFrame.from_records(coefficient_rows),
        "coefficient_rank_ladder.csv": ladder_frame,
        "slope_permutation_null.csv": null_frame,
    }
    return summary, tables


def run(
    *,
    pkis1_zip: Path,
    ledger_path: Path,
    published_arm: Path | None,
    folds: int,
    qap_permutations: int,
    slope_permutations: int,
    bootstrap_repeats: int,
    seed: int,
) -> tuple[dict[str, object], dict[str, pd.DataFrame]]:
    reference, score_columns, smiles, support = dcg.load_deleaked_reference(pkis1_zip)
    score_index = {target: position for position, target in enumerate(score_columns)}
    block = np.ascontiguousarray(reference[:, [score_index[target] for target in TARGETS20]])
    del reference
    descriptors = mechanism.molecular_descriptors(smiles)
    murcko = mechanism.scaffold_keys(smiles)
    if not np.isfinite(descriptors).all():  # molecular_descriptors already enforces this
        raise AssertionError("validated descriptor matrix unexpectedly became non-finite")

    ledger = load_ledger(ledger_path)
    summary, tables = evaluate(
        block,
        descriptors,
        murcko,
        ledger,
        folds=folds,
        qap_permutations=qap_permutations,
        slope_permutations=slope_permutations,
        bootstrap_repeats=bootstrap_repeats,
        seed=seed,
        published_arm=published_arm,
    )
    summary["support"] = {
        "ligands": int(len(block)),
        "targets": int(block.shape[1]),
        "ligand_by_target_cells": int(block.size),
        "chemical_groups_murcko": int(len(np.unique(murcko))),
        "target_pairs": int(len(ledger)),
        **support,
    }
    return summary, tables


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pkis1-zip", type=Path, default=DEFAULT_PKIS1_ZIP)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--published-arm", type=Path, default=DEFAULT_PUBLISHED_ARM)
    parser.add_argument("--folds", type=int, default=DEFAULT_FOLDS)
    parser.add_argument("--qap-permutations", type=int, default=DEFAULT_QAP_PERMUTATIONS)
    parser.add_argument(
        "--slope-permutations", type=int, default=DEFAULT_SLOPE_PERMUTATIONS
    )
    parser.add_argument(
        "--bootstrap-repeats", type=int, default=DEFAULT_BOOTSTRAP_REPEATS
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    summary, tables = run(
        pkis1_zip=args.pkis1_zip,
        ledger_path=args.ledger,
        published_arm=args.published_arm,
        folds=args.folds,
        qap_permutations=args.qap_permutations,
        slope_permutations=args.slope_permutations,
        bootstrap_repeats=args.bootstrap_repeats,
        seed=args.seed,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    mechanism.write_json(args.output / "summary.json", summary)
    for name, frame in tables.items():
        mechanism.write_csv(frame, args.output / name)
    print(tables["panel_agreement.csv"].to_string(index=False))


if __name__ == "__main__":
    main()
