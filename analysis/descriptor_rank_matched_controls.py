#!/usr/bin/env python3
"""Rank/dimension-matched controls for the fixed-20 descriptor geometry.

The published physicochemical arm uses seven ligand features but fits a separate
coefficient vector for every target.  Its agreement with experimental target-pair geometry
could therefore reflect either the chosen physicochemical basis or the generic capacity of
seven target-indexed ligand features.  This analysis keeps the support, targets, folds,
fold-local score transformation, feature standardization, and target-wise least-squares
estimator fixed while changing only the seven-dimensional ligand basis.

Predictive bases
----------------
``physicochemical_7``
    The seven published RDKit descriptors.
``morgan_count_plus_size_rp_7_locked_seed_20260910``
    The deterministically locked first-seed representative of a 20-seed ensemble.  Every member contains explicit
    total Morgan occurrence count plus six seeded Rademacher projections of the raw hashed
    radius-2 count fingerprint.  There is no ligand-wise normalization, so size/magnitude
    and repeated environments are retained.  All 20 members receive point estimates;
    their range is algorithmic sensitivity, not uncertainty.
``morgan_normalized_bit_rp_7_seed_a`` and ``..._seed_b``
    Two secondary normalization-sensitivity projections of L2-normalized Morgan bit
    fingerprints.  They are not the principal basis-specificity controls.
``stable_hash_nuisance_7``
    Seven deterministic pseudorandom coordinates derived from a keyed BLAKE2b hash of the
    ligand SMILES.  They preserve duplicate identity but contain no smooth chemical basis;
    this is a negative-control basis, not a molecular representation proposed for use.

Nuisance-projection ensemble
----------------------------
Twenty deterministic row permutations of one standardized copy of the seven-dimensional
physicochemical matrix preserve the descriptor cloud exactly while destroying which ligand
receives which row.  Each receives the identical fold/scaling/target-wise OLS estimator.
Their per-seed held-out predictive performance, operational reduction and four panel-map agreements
are reported as an algorithmic distribution without confidence-interval terminology.

All predictive bases are fitted with the identical five-fold Murcko GroupKFold contract.  Their
primary estimand is the *relative reduction in mean squared off-diagonal target correlation
after subtracting fixed out-of-fold predictions*.  It is an operational correlation-ratio
contrast, not a fraction of correlation explained, not a variance decomposition, and not a
causal attribution.

Transported-subspace oracle comparator
--------------------------------------
``outcome_rank7_svd_transported_oracle`` is deliberately not a predictor from ligand
features.  In each fold, the seven leading target directions are estimated from the training residual
surface and each held-out docking row is projected onto them using that row's docking
outcomes.  It is a transported-subspace oracle reconstruction comparator: test rows are
never used to fit the target directions, but their outcomes are used for projection.  It is
neither deployable nor a mathematical capacity ceiling for arbitrary rank-7 methods.

Paired design sensitivity uses a tractable coarsened chemical-group block bootstrap.  Individual
Murcko (or generic-Murcko) groups are greedily packed into balanced blocks, and iid Exp(1)
weights are drawn for those blocks.  The same blocks and weights are used for every arm, so
physicochemical-minus-control contrasts are paired.  Coarsening makes the full 259,579-row
support tractable and induces common weights within unrelated packed groups; these intervals
are sensitivity intervals, not a substitute for an individual-cluster full-refit bootstrap.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import OrderedDict
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import GroupKFold

try:  # Support direct execution and package imports.
    from . import descriptor_component_geometry as component
    from . import descriptor_geometry_decomposition as edge_decomposition
    from . import residual_mechanism_analysis as mechanism
    from . import residual_target_geometry_validation as geometry
    from . import target_blind_descriptor_control as target_blind
except ImportError:  # pragma: no cover
    import descriptor_component_geometry as component  # type: ignore
    import descriptor_geometry_decomposition as edge_decomposition  # type: ignore
    import residual_mechanism_analysis as mechanism  # type: ignore
    import residual_target_geometry_validation as geometry  # type: ignore
    import target_blind_descriptor_control as target_blind  # type: ignore


RDLogger.DisableLog("rdApp.*")

PACKAGE = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PACKAGE / "results" / "descriptor_rank_matched_controls"
DEFAULT_LEDGER = target_blind.DEFAULT_LEDGER
DEFAULT_PKIS1_ZIP = component.DEFAULT_PKIS1_ZIP
DEFAULT_FOLDS = 5
DEFAULT_FEATURES = 7
DEFAULT_NORMALIZED_MORGAN_SEEDS = (20260821, 20260822)
DEFAULT_COUNT_MORGAN_SEEDS = tuple(range(20260910, 20260930))
DEFAULT_NUISANCE_PERMUTATION_SEEDS = tuple(range(20261010, 20261030))
DEFAULT_HASH_SEED = 20260823
DEFAULT_BOOTSTRAP_SEED = 20260824
DEFAULT_BOOTSTRAP_REPEATS = 5_000
DEFAULT_BOOTSTRAP_BLOCKS = 512
BOOTSTRAP_CHUNK = 250

TARGETS20 = component.TARGETS20
PANELS = component.PANELS

PHYSICOCHEMICAL = "physicochemical_7"
MORGAN_A = "morgan_normalized_bit_rp_7_seed_a"
MORGAN_B = "morgan_normalized_bit_rp_7_seed_b"
COUNT_MORGAN_LOCKED = "morgan_count_plus_size_rp_7_locked_seed_20260910"
HASH_NUISANCE = "stable_hash_nuisance_7"
ORACLE = "outcome_rank7_svd_transported_oracle"
OBSERVED = "observed_residual"
PREDICTIVE_BASES = (
    PHYSICOCHEMICAL,
    COUNT_MORGAN_LOCKED,
    MORGAN_A,
    MORGAN_B,
    HASH_NUISANCE,
)


def generic_murcko_acyclic_topology_keys(
    smiles: Iterable[str],
    murcko_keys: np.ndarray,
    generic_murcko_keys: np.ndarray,
) -> tuple[np.ndarray, dict[str, object]]:
    """Coarse holdout keys that do not leave every acyclic ligand a singleton.

    Cyclic molecules use their atom/bond-agnostic Bemis--Murcko framework.  Because an
    acyclic molecule has an empty Murcko framework, its *whole heavy-atom graph* is made
    generic instead (atom types and bond orders are erased while connectivity and branching
    are retained).  Thus related acyclic topologies can be held out together.  A rare RDKit
    genericisation failure falls back to the exact InChI connectivity block and is counted.
    """
    values = [str(value) for value in smiles]
    murcko = np.asarray(murcko_keys, dtype=str)
    generic = np.asarray(generic_murcko_keys, dtype=str)
    if len(values) != len(murcko) or len(murcko) != len(generic):
        raise ValueError("SMILES and scaffold-key arrays must have the same length")

    output = generic.astype(object, copy=True)
    acyclic = np.char.startswith(murcko, "ACYCLIC_SINGLETON:")
    failures = 0
    for row in np.flatnonzero(acyclic):
        molecule = Chem.MolFromSmiles(values[row])
        if molecule is None:  # pragma: no cover - validated by upstream loaders
            raise ValueError(f"invalid SMILES at row {row}")
        try:
            topology = MurckoScaffold.MakeScaffoldGeneric(molecule)
            key = Chem.MolToSmiles(topology, canonical=True)
            if not key:
                raise ValueError("empty generic graph")
            output[row] = f"ACYCLIC_GENERIC_TOPOLOGY:{key}"
        except Exception:  # pragma: no cover - defensive RDKit fallback
            inchikey = Chem.MolToInchiKey(molecule)
            if not inchikey or len(inchikey) < 14:
                raise ValueError(f"could not construct an acyclic key at row {row}")
            output[row] = f"ACYCLIC_CONNECTIVITY:{inchikey[:14]}"
            failures += 1

    _, counts = np.unique(output, return_counts=True)
    acyclic_keys = output[acyclic]
    acyclic_unique = len(np.unique(acyclic_keys)) if len(acyclic_keys) else 0
    return output, {
        "definition": (
            "generic Bemis-Murcko framework for cyclic ligands; generic whole-molecule "
            "heavy-atom connectivity topology for acyclic ligands"
        ),
        "ligands": int(len(output)),
        "acyclic_ligands": int(acyclic.sum()),
        "groups": int(len(counts)),
        "acyclic_groups": int(acyclic_unique),
        "singleton_groups": int(np.sum(counts == 1)),
        "largest_group_ligands": int(counts.max()),
        "genericisation_failures_using_exact_connectivity_fallback": int(failures),
    }


def morgan_random_projection_features(
    smiles: Iterable[str],
    normalized_bit_seeds: tuple[int, int],
    count_seeds: tuple[int, ...],
    *,
    features: int = DEFAULT_FEATURES,
    fp_size: int = 2_048,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    """Normalized-bit controls plus a multi-seed count-preserving RP ensemble.

    All projections use independent Rademacher matrices.  The two historical secondary
    controls divide the binary fingerprint by its L2 norm.  The primary ensemble instead
    projects raw hashed Morgan occurrence counts without ligand-wise normalization, so
    fingerprint magnitude and repeated-environment counts are retained.
    """
    if len(normalized_bit_seeds) != 2 or len(set(normalized_bit_seeds)) != 2:
        raise ValueError("exactly two distinct normalized-bit seeds are required")
    if len(count_seeds) < 20 or len(set(count_seeds)) != len(count_seeds):
        raise ValueError("at least twenty distinct count-projection seeds are required")
    values = [str(value) for value in smiles]
    normalized_projection = np.column_stack(
        [
            np.random.default_rng(seed)
            .choice((-1.0, 1.0), size=(fp_size, features))
            .astype(np.float32)
            / np.sqrt(features)
            for seed in normalized_bit_seeds
        ]
    )
    count_projection = np.column_stack(
        [
            np.random.default_rng(seed)
            .choice((-1.0, 1.0), size=(fp_size, features - 1))
            .astype(np.float32)
            / np.sqrt(features - 1)
            for seed in count_seeds
        ]
    )
    normalized_output = np.empty(
        (len(values), 2 * features), dtype=np.float32
    )
    count_output = np.empty(
        (len(values), len(count_seeds), features), dtype=np.float32
    )
    nonzero_counts = np.empty(len(values), dtype=np.int32)
    occurrence_counts = np.empty(len(values), dtype=np.int32)
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=fp_size)
    for row, text in enumerate(values):
        molecule = Chem.MolFromSmiles(text)
        if molecule is None:
            raise ValueError(f"invalid SMILES at row {row}")
        elements = generator.GetCountFingerprint(molecule).GetNonzeroElements()
        indices = np.fromiter(elements.keys(), dtype=np.int32)
        counts = np.fromiter(elements.values(), dtype=np.float64)
        if not len(indices):
            raise ValueError(f"empty Morgan fingerprint at row {row}")
        nonzero_counts[row] = len(indices)
        occurrence_counts[row] = int(counts.sum())
        normalized_output[row] = normalized_projection[indices].sum(axis=0) / np.sqrt(
            len(indices)
        )
        count_output[row, :, 0] = counts.sum()
        count_output[row, :, 1:] = (
            counts @ count_projection[indices]
        ).reshape(len(count_seeds), features - 1)
    first, second = normalized_output[:, :features], normalized_output[:, features:]
    ensemble = count_output
    return first, second, ensemble, {
        "fingerprint": "hashed Morgan count fingerprint, radius 2, 2048 bins",
        "normalized_bit_secondary_controls": {
            "definition": (
                "binary nonzero bins divided by sqrt(number of nonzero bins) before "
                "projection; retained only as secondary normalization sensitivity"
            ),
            "seeds": [int(seed) for seed in normalized_bit_seeds],
        },
        "count_preserving_primary_ensemble": {
            "definition": (
                "one explicit total-occurrence-count coordinate plus six raw hashed "
                "Morgan count random-projection coordinates, without ligand-wise "
                "normalization; the size/magnitude direction and repeated environments "
                "are therefore retained"
            ),
            "seeds": [int(seed) for seed in count_seeds],
            "members": int(len(count_seeds)),
            "locked_representative_seed": int(count_seeds[0]),
        },
        "minimum_nonzero_bins": int(nonzero_counts.min()),
        "median_nonzero_bins": float(np.median(nonzero_counts)),
        "maximum_nonzero_bins": int(nonzero_counts.max()),
        "minimum_total_occurrences": int(occurrence_counts.min()),
        "median_total_occurrences": float(np.median(occurrence_counts)),
        "maximum_total_occurrences": int(occurrence_counts.max()),
    }


def stable_hash_nuisance_features(
    smiles: Iterable[str], seed: int, *, features: int = DEFAULT_FEATURES
) -> np.ndarray:
    """Deterministic identity-hash nuisance basis with approximately unit variance."""
    values = [str(value) for value in smiles]
    output = np.empty((len(values), features), dtype=np.float64)
    key = int(seed).to_bytes(16, byteorder="little", signed=False)
    for row, text in enumerate(values):
        digest = hashlib.blake2b(
            text.encode("utf-8"), digest_size=8 * features, key=key
        ).digest()
        words = np.frombuffer(digest, dtype="<u8")
        uniform = (words >> np.uint64(11)).astype(np.float64) * (2.0**-53)
        output[row] = (uniform - 0.5) * np.sqrt(12.0)
    return output


def array_sha256(values: np.ndarray, dtype: str) -> str:
    """Portable checksum of a contiguous array in an explicit little-endian dtype."""
    canonical = np.ascontiguousarray(np.asarray(values, dtype=dtype))
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


def standardized_descriptor_matrix(features: np.ndarray) -> np.ndarray:
    """Globally standardize a descriptor matrix before nuisance row permutation.

    The predictive estimator still applies its usual training-fold-only scaling.  This
    global affine transform is therefore prediction-invariant, but makes explicit that all
    nuisance members permute the exact same standardized seven-dimensional cloud.
    """
    values = np.asarray(features, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != DEFAULT_FEATURES:
        raise ValueError("physicochemical nuisance source must be an n-by-7 matrix")
    scale = values.std(axis=0, ddof=1)
    if np.any(scale <= 1e-12):
        raise ValueError("physicochemical nuisance source has a constant column")
    return np.ascontiguousarray((values - values.mean(axis=0)) / scale)


def row_permuted_nuisance_features(
    standardized_features: np.ndarray, seed: int
) -> tuple[np.ndarray, np.ndarray, dict[str, str]]:
    """One deterministic row-permutation nuisance basis plus auditable checksums."""
    values = np.asarray(standardized_features, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != DEFAULT_FEATURES:
        raise ValueError("standardized nuisance source must be an n-by-7 matrix")
    permutation = np.random.default_rng(seed).permutation(len(values)).astype(
        np.int64, copy=False
    )
    permuted = np.ascontiguousarray(values[permutation])
    return permuted, permutation, {
        "permutation_index_sha256": array_sha256(permutation, "<i8"),
        "permuted_feature_matrix_sha256": array_sha256(permuted, "<f8"),
    }


def _effective_rank(matrix: np.ndarray, relative_tolerance: float = 1e-10) -> int:
    singular = np.linalg.svd(matrix, compute_uv=False)
    return int(np.sum(singular > singular[0] * relative_tolerance))


def fit_predictive_bases(
    block: np.ndarray,
    bases: "OrderedDict[str, np.ndarray]",
    groups: np.ndarray,
    folds: int,
    *,
    compute_oracle: bool = True,
) -> dict[str, object]:
    """Fit every seven-dimensional basis on exactly the same GroupKFold splits."""
    block = np.asarray(block, dtype=np.float64)
    groups = np.asarray(groups, dtype=object)
    for name, features in bases.items():
        if features.shape != (len(block), DEFAULT_FEATURES):
            raise ValueError(f"{name} is not an n-by-7 feature matrix")
        if not np.isfinite(features).all():
            raise ValueError(f"{name} contains non-finite features")

    observed = np.full_like(block, np.nan)
    predictions = OrderedDict(
        (name, np.full_like(block, np.nan)) for name in bases
    )
    capacity = np.full_like(block, np.nan) if compute_oracle else None
    fold_index = np.full(len(block), -1, dtype=np.int16)
    fold_rows: list[dict[str, object]] = []
    coefficient_ranks: dict[str, list[int]] = {name: [] for name in bases}

    splitter = GroupKFold(n_splits=folds)
    for fold, (train, test) in enumerate(splitter.split(block, groups=groups), start=1):
        if set(groups[train].tolist()) & set(groups[test].tolist()):
            raise AssertionError("chemical group leakage")
        train_surface, test_surface, scale = mechanism.fold_local_residual_transform(
            block[train], block[test]
        )
        observed[test] = test_surface
        fold_index[test] = fold - 1

        for name, features in bases.items():
            mean = features[train].mean(axis=0)
            sd = features[train].std(axis=0, ddof=1)
            if np.any(sd <= 1e-12):
                raise ValueError(f"{name} has a constant training feature in fold {fold}")
            train_design = np.column_stack(
                [np.ones(len(train)), (features[train] - mean) / sd]
            )
            test_design = np.column_stack(
                [np.ones(len(test)), (features[test] - mean) / sd]
            )
            coefficients = np.linalg.lstsq(train_design, train_surface, rcond=None)[0]
            predictions[name][test] = test_design @ coefficients
            deviation = coefficients[1:] - coefficients[1:].mean(axis=1, keepdims=True)
            coefficient_ranks[name].append(_effective_rank(deviation))

        fold_record: dict[str, object] = {
            "fold": fold,
            "train_ligands": int(len(train)),
            "test_ligands": int(len(test)),
            "train_groups": int(len(np.unique(groups[train]))),
            "test_groups": int(len(np.unique(groups[test]))),
            **scale,
        }
        if compute_oracle:
            # Transported-subspace oracle: target directions come only from training rows,
            # but the held-out row's docking outcomes are needed for its projection.
            covariance = train_surface.T @ train_surface
            eigenvalues, eigenvectors = np.linalg.eigh(covariance)
            basis = eigenvectors[:, -DEFAULT_FEATURES:]
            assert capacity is not None
            capacity[test] = test_surface @ basis @ basis.T
            fold_record.update(
                {
                    "rank7_training_energy_share": float(
                        eigenvalues[-DEFAULT_FEATURES:].sum() / eigenvalues.sum()
                    ),
                    "rank7_test_reconstruction_r2": float(
                        1.0
                        - np.square(test_surface - capacity[test]).sum()
                        / np.square(test_surface).sum()
                    ),
                }
            )
        fold_rows.append(fold_record)

    if not np.isfinite(observed).all():
        raise AssertionError("OOF surfaces were not filled")
    if capacity is not None and not np.isfinite(capacity).all():
        raise AssertionError("oracle OOF surface was not filled")
    if any(not np.isfinite(surface).all() for surface in predictions.values()):
        raise AssertionError("an OOF predictive surface was not filled")
    return {
        "observed": observed,
        "predictions": predictions,
        "capacity": capacity,
        "fold_index": fold_index,
        "fold_rows": fold_rows,
        "coefficient_effective_ranks": coefficient_ranks,
    }


def predictive_metrics(observed: np.ndarray, prediction: np.ndarray) -> dict[str, object]:
    """Precisely named operational metrics for one fixed OOF prediction surface."""
    error = observed - prediction
    before = mechanism.correlation_moments(observed)
    after = mechanism.correlation_moments(error)
    before_r2 = before["mean_squared_offdiagonal_correlation"]
    after_r2 = after["mean_squared_offdiagonal_correlation"]
    target_sst = np.square(observed - observed.mean(axis=0)).sum(axis=0)
    target_sse = np.square(error).sum(axis=0)
    target_r2 = 1.0 - target_sse / target_sst
    return {
        "estimand": (
            "relative reduction in mean squared off-diagonal target correlation after "
            "subtracting the fixed out-of-fold prediction surface"
        ),
        "mean_squared_offdiagonal_target_correlation_before_subtraction": float(before_r2),
        "mean_squared_offdiagonal_target_correlation_after_subtraction": float(after_r2),
        "absolute_reduction_in_mean_squared_offdiagonal_target_correlation": float(
            before_r2 - after_r2
        ),
        "relative_reduction_in_mean_squared_offdiagonal_target_correlation": float(
            (before_r2 - after_r2) / before_r2
        ),
        "mean_out_of_fold_target_r2": float(target_r2.mean()),
        "median_out_of_fold_target_r2": float(np.median(target_r2)),
        "minimum_out_of_fold_target_r2": float(target_r2.min()),
        "maximum_out_of_fold_target_r2": float(target_r2.max()),
        "not_an_explained_fraction": (
            "the correlation-ratio contrast is non-additive and is not a fraction of "
            "correlation, covariance, or variance explained"
        ),
    }


def describe_algorithmic_seed_distribution(values: np.ndarray) -> dict[str, object]:
    """Describe seed sensitivity without CI-like interval field names."""
    values = np.asarray(values, dtype=np.float64)
    return {
        "members": int(len(values)),
        "mean": float(values.mean()),
        "standard_deviation_across_seeds": float(values.std(ddof=1)),
        "minimum": float(values.min()),
        "median": float(np.median(values)),
        "maximum": float(values.max()),
        "central_90_percent_seed_range": [
            float(np.quantile(values, 0.05)),
            float(np.quantile(values, 0.95)),
        ],
        "central_95_percent_seed_range": [
            float(np.quantile(values, 0.025)),
            float(np.quantile(values, 0.975)),
        ],
        "not_an_uncertainty_interval": (
            "quantiles summarize deterministic algorithmic seed sensitivity only"
        ),
    }


def describe_conditional_sensitivity(values: np.ndarray) -> dict[str, object]:
    """Describe fixed-OOF reweighting draws without confidence-interval terminology."""
    values = np.asarray(values, dtype=np.float64)
    return {
        "n": int(len(values)),
        "mean": float(values.mean()),
        "standard_deviation": float(values.std(ddof=1)),
        "minimum": float(values.min()),
        "median": float(np.median(values)),
        "maximum": float(values.max()),
        "conditional_coarsened_sensitivity_interval_90": [
            float(np.quantile(values, 0.05)),
            float(np.quantile(values, 0.95)),
        ],
        "conditional_coarsened_sensitivity_interval_95": [
            float(np.quantile(values, 0.025)),
            float(np.quantile(values, 0.975)),
        ],
        "not_a_confidence_interval": True,
    }


def evaluate_count_morgan_ensemble(
    block: np.ndarray,
    ensemble_features: np.ndarray,
    seeds: tuple[int, ...],
    groups: np.ndarray,
    ledger: pd.DataFrame,
    folds: int,
) -> tuple[dict[str, object], dict[str, pd.DataFrame]]:
    """Point/distribution controls for a deterministically locked count-RP seed ensemble.

    Members are evaluated one at a time to avoid retaining twenty 259,579-by-20 OOF
    prediction surfaces.  Every member nevertheless receives the identical folds,
    fold-local transforms, seven feature dimensions and per-target least-squares fit.
    The seed distribution is algorithmic sensitivity, not sampling uncertainty.
    """
    if ensemble_features.shape != (len(block), len(seeds), DEFAULT_FEATURES):
        raise ValueError("count Morgan ensemble has an unexpected shape")
    rank_vectors = target_blind.experimental_rank_vectors(ledger)
    triangle = np.triu_indices(len(TARGETS20), k=1)
    member_records: list[dict[str, object]] = []
    geometry_rows: list[dict[str, object]] = []
    association_rows: list[dict[str, object]] = []
    for member, seed in enumerate(seeds):
        name = f"morgan_count_plus_size_rp_7_seed_{seed}"
        fitted = fit_predictive_bases(
            block,
            OrderedDict([(name, ensemble_features[:, member, :])]),
            groups,
            folds,
            compute_oracle=False,
        )
        observed = fitted["observed"]
        prediction = fitted["predictions"][name]
        metrics = predictive_metrics(observed, prediction)
        predicted_edges = geometry.target_correlation(prediction)[triangle]
        remainder_edges = geometry.target_correlation(observed - prediction)[triangle]
        component_remainder_spearman = float(
            pd.Series(predicted_edges).corr(
                pd.Series(remainder_edges), method="spearman"
            )
        )
        component_remainder_pearson = float(
            np.corrcoef(predicted_edges, remainder_edges)[0, 1]
        )
        agreement: dict[str, float] = OrderedDict()
        for panel in PANELS:
            value = target_blind.spearman_against(
                predicted_edges, rank_vectors[panel]
            )
            if value is None:
                raise ValueError(f"count Morgan seed {seed} has degenerate geometry")
            agreement[panel] = value
            experimental = ledger[
                f"{panel}_experimental_centered_correlation"
            ].to_numpy(dtype=np.float64)
            fit = edge_decomposition.joint_edge_regression(
                experimental, predicted_edges, remainder_edges
            )
            association_rows.append(
                {
                    "seed": int(seed),
                    "panel": panel,
                    "component_remainder_spearman": component_remainder_spearman,
                    "component_remainder_pearson": component_remainder_pearson,
                    "partial_spearman_component": edge_decomposition.partial_spearman(
                        experimental, predicted_edges, remainder_edges
                    ),
                    "partial_spearman_remainder": edge_decomposition.partial_spearman(
                        experimental, remainder_edges, predicted_edges
                    ),
                    "standardized_beta_component": fit["beta_first"],
                    "standardized_beta_remainder": fit["beta_second"],
                    "joint_rank_r2": fit["r2_joint"],
                    "semipartial_rank_r2_component": fit[
                        "semipartial_r2_first"
                    ],
                    "semipartial_rank_r2_remainder": fit[
                        "semipartial_r2_second"
                    ],
                }
            )
            geometry_rows.append(
                {"seed": int(seed), "panel": panel, "spearman": value}
            )
        member_records.append(
            {
                "seed": int(seed),
                "locked_representative": bool(member == 0),
                "coefficient_effective_ranks_by_fold": fitted[
                    "coefficient_effective_ranks"
                ][name],
                "predictive_metrics": metrics,
                "panel_geometry_agreement": agreement,
                "component_remainder_spearman": component_remainder_spearman,
                "component_remainder_pearson": component_remainder_pearson,
            }
        )
        del fitted, observed, prediction

    metric_rows = [
        {
            "seed": record["seed"],
            "locked_representative": record["locked_representative"],
            **{
                key: value
                for key, value in record["predictive_metrics"].items()
                if isinstance(value, (int, float))
            },
        }
        for record in member_records
    ]
    return {
        "members": int(len(seeds)),
        "seeds": [int(seed) for seed in seeds],
        "locked_representative_seed": int(seeds[0]),
        "representative_selection": (
            "first seed in ascending order, deterministically locked before inspecting "
            "this ensemble run and not selected by performance"
        ),
        "interpretation": (
            "the across-seed distribution is algorithmic random-projection sensitivity, "
            "not a confidence interval or chemical-space uncertainty distribution"
        ),
        "members_detail": member_records,
    }, {
        "morgan_count_rp_ensemble_metrics.csv": pd.DataFrame.from_records(
            metric_rows
        ),
        "morgan_count_rp_ensemble_panel_geometry.csv": pd.DataFrame.from_records(
            geometry_rows
        ),
        "morgan_count_rp_ensemble_partial_associations.csv": pd.DataFrame.from_records(
            association_rows
        ),
    }


def evaluate_row_permutation_nuisance_ensemble(
    block: np.ndarray,
    physical_features: np.ndarray,
    seeds: tuple[int, ...],
    groups: np.ndarray,
    ledger: pd.DataFrame,
    folds: int,
) -> tuple[dict[str, object], pd.DataFrame]:
    """Evaluate shuffled-physicochemical nuisance bases on the locked OOF design.

    Every member is a row permutation of one globally standardized seven-descriptor
    matrix.  It therefore retains the feature cloud exactly while destroying the
    ligand-feature assignment.  The ordinary estimator subsequently performs its same
    training-fold-only scaling, fold-local outcome transform and target-wise OLS.
    """
    if len(seeds) < 20 or len(set(seeds)) != len(seeds):
        raise ValueError("at least twenty distinct nuisance-permutation seeds are required")
    standardized = standardized_descriptor_matrix(physical_features)
    source_checksum = array_sha256(standardized, "<f8")
    rank_vectors = target_blind.experimental_rank_vectors(ledger)
    triangle = np.triu_indices(len(TARGETS20), k=1)
    records: list[dict[str, object]] = []
    observed_reference: np.ndarray | None = None
    maximum_observed_difference = 0.0
    for seed in seeds:
        name = f"physicochemical_row_permutation_nuisance_7_seed_{seed}"
        permuted, permutation, checksums = row_permuted_nuisance_features(
            standardized, seed
        )
        fitted = fit_predictive_bases(
            block,
            OrderedDict([(name, permuted)]),
            groups,
            folds,
            compute_oracle=False,
        )
        observed = fitted["observed"]
        if observed_reference is None:
            observed_reference = observed.copy()
        else:
            maximum_observed_difference = max(
                maximum_observed_difference,
                float(np.max(np.abs(observed - observed_reference))),
            )
        prediction = fitted["predictions"][name]
        metrics = predictive_metrics(observed, prediction)
        predicted_edges = geometry.target_correlation(prediction)[triangle]
        panel_agreement: dict[str, float] = OrderedDict()
        for panel in PANELS:
            value = target_blind.spearman_against(
                predicted_edges, rank_vectors[panel]
            )
            if value is None:
                raise ValueError(
                    f"row-permutation nuisance seed {seed} has degenerate geometry"
                )
            panel_agreement[panel] = value
        records.append(
            {
                "seed": int(seed),
                **checksums,
                "permutation_is_bijection": bool(
                    np.array_equal(np.sort(permutation), np.arange(len(permutation)))
                ),
                "coefficient_effective_ranks_by_fold": fitted[
                    "coefficient_effective_ranks"
                ][name],
                "mean_out_of_fold_target_r2": metrics[
                    "mean_out_of_fold_target_r2"
                ],
                "relative_reduction_in_mean_squared_offdiagonal_target_correlation": (
                    metrics[
                        "relative_reduction_in_mean_squared_offdiagonal_target_correlation"
                    ]
                ),
                "panel_geometry_agreement": panel_agreement,
            }
        )
        del fitted, observed, prediction, permuted, permutation

    mean_r2 = np.asarray(
        [record["mean_out_of_fold_target_r2"] for record in records],
        dtype=np.float64,
    )
    operational = np.asarray(
        [
            record[
                "relative_reduction_in_mean_squared_offdiagonal_target_correlation"
            ]
            for record in records
        ],
        dtype=np.float64,
    )
    panel_distributions: dict[str, object] = OrderedDict()
    for panel in PANELS:
        panel_distributions[panel] = describe_algorithmic_seed_distribution(
            np.asarray(
                [record["panel_geometry_agreement"][panel] for record in records],
                dtype=np.float64,
            )
        )
    table_rows = [
        {
            "seed": record["seed"],
            "permutation_index_sha256": record["permutation_index_sha256"],
            "permuted_feature_matrix_sha256": record[
                "permuted_feature_matrix_sha256"
            ],
            "mean_out_of_fold_target_r2": record["mean_out_of_fold_target_r2"],
            "relative_reduction_in_mean_squared_offdiagonal_target_correlation": record[
                "relative_reduction_in_mean_squared_offdiagonal_target_correlation"
            ],
            **{
                f"{panel}_component_map_spearman": record[
                    "panel_geometry_agreement"
                ][panel]
                for panel in PANELS
            },
        }
        for record in records
    ]
    return {
        "members": int(len(records)),
        "seeds": [int(seed) for seed in seeds],
        "seed_selection": (
            "fixed ascending integer range chosen for this post-hoc nuisance extension; "
            "no member was selected by performance"
        ),
        "source": (
            "row permutations of one globally standardized copy of the same seven "
            "physicochemical descriptor matrix; the fit then repeats the primary "
            "training-fold-only standardization"
        ),
        "source_standardized_feature_matrix_sha256": source_checksum,
        "checksums": (
            "SHA-256 over contiguous explicit little-endian int64 permutation indices "
            "and float64 permuted matrices"
        ),
        "same_oof_observed_surface_maximum_absolute_difference": float(
            maximum_observed_difference
        ),
        "members_detail": records,
        "algorithmic_seed_distributions": {
            "mean_out_of_fold_target_r2": describe_algorithmic_seed_distribution(
                mean_r2
            ),
            "relative_reduction_in_mean_squared_offdiagonal_target_correlation": (
                describe_algorithmic_seed_distribution(operational)
            ),
            "panel_geometry_agreement": panel_distributions,
        },
        "interpretation": (
            "Across-seed ranges diagnose the algorithmic nuisance-projection mechanism; "
            "they are distributions over fixed row permutations, not confidence "
            "intervals or sampling uncertainty."
        ),
    }, pd.DataFrame.from_records(table_rows)


def refresh_interpretive_fields(summary: dict[str, object]) -> dict[str, object]:
    """Synchronize interpretation-only fields from already computed numeric results.

    This helper deliberately derives every quoted negative-control value from the numeric
    fields in ``summary``.  It therefore supports a fast deterministic metadata refresh
    without rerunning the existing fingerprint and OOF analyses.
    """
    estimands = summary["estimands"]
    if "paired_uncertainty" in estimands:
        estimands["paired_design_sensitivity"] = estimands.pop("paired_uncertainty")
    metrics = summary["predictive_metrics"][HASH_NUISANCE]
    agreement = summary["panel_geometry_agreement"][HASH_NUISANCE]
    summary["negative_control_interpretation"] = {
        "stable_hash_mean_out_of_fold_target_r2": metrics[
            "mean_out_of_fold_target_r2"
        ],
        "stable_hash_relative_reduction_in_mean_squared_offdiagonal_target_correlation": (
            metrics[
                "relative_reduction_in_mean_squared_offdiagonal_target_correlation"
            ]
        ),
        "stable_hash_panel_geometry_agreement": agreement,
        "conclusion": (
            "The stable-hash basis has essentially zero ligand-level held-out predictive "
            "performance yet appreciable scale-free target-pair geometry agreement. "
            "Experimental geometry of a target-specific fitted component cannot by "
            "itself support feature-specific orientation without held-out-performance "
            "and negative-feature controls; the stable-hash arm falsifies that "
            "feature-specific interpretation here."
        ),
        "nuisance_projection_extension": (
            "Twenty seeded row permutations of the same standardized physicochemical "
            "matrix were evaluated with the identical OOF estimator; see "
            "physicochemical_row_permutation_nuisance_ensemble."
            if "physicochemical_row_permutation_nuisance_ensemble" in summary
            else "The requested multi-seed nuisance extension has not yet been attached."
        ),
    }
    boundary_addition = (
        " The stable-hash basis has essentially zero ligand-level held-out predictive performance "
        "but appreciable scale-free target-pair geometry agreement. Thus a target-specific "
        "fitted component's experimental geometry cannot by itself support feature-specific "
        "orientation without held-out-performance and negative-feature controls; the hash "
        "arm falsifies that interpretation here."
    )
    if boundary_addition.strip() not in summary["claim_boundary"]:
        summary["claim_boundary"] += boundary_addition
    return summary


def feature_diagnostics(
    bases: "OrderedDict[str, np.ndarray]", physical: np.ndarray
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    physical_z = (physical - physical.mean(axis=0)) / physical.std(axis=0, ddof=1)
    for name, features in bases.items():
        z = (features - features.mean(axis=0)) / features.std(axis=0, ddof=1)
        cross = z.T @ physical_z / (len(z) - 1)
        gram = z.T @ z / (len(z) - 1)
        rows.append(
            {
                "basis": name,
                "dimensions": int(features.shape[1]),
                "matrix_rank": int(np.linalg.matrix_rank(z)),
                "maximum_absolute_pearson_with_published_descriptors": float(
                    np.abs(cross).max()
                ),
                "maximum_absolute_within_basis_correlation": float(
                    np.abs(gram - np.eye(len(gram))).max()
                ),
            }
        )
    return pd.DataFrame.from_records(rows)


def balanced_group_blocks(
    labels: np.ndarray, requested_blocks: int, seed: int
) -> tuple[np.ndarray, dict[str, object]]:
    """Pack intact chemical groups into approximately row-balanced bootstrap blocks."""
    _, inverse, sizes = np.unique(
        np.asarray(labels, dtype=object), return_inverse=True, return_counts=True
    )
    blocks = min(int(requested_blocks), len(sizes))
    rng = np.random.default_rng(seed)
    jitter = rng.random(len(sizes))
    order = np.lexsort((jitter, -sizes))
    loads = np.zeros(blocks, dtype=np.int64)
    assignment = np.empty(len(sizes), dtype=np.int32)
    for group in order:
        block = int(np.argmin(loads))
        assignment[group] = block
        loads[block] += sizes[group]
    row_blocks = assignment[inverse]
    return row_blocks, {
        "chemical_groups": int(len(sizes)),
        "bootstrap_blocks": int(blocks),
        "minimum_block_ligands": int(loads.min()),
        "median_block_ligands": float(np.median(loads)),
        "maximum_block_ligands": int(loads.max()),
        "assignment_seed": int(seed),
    }


def block_moments(
    surface: np.ndarray, row_blocks: np.ndarray, blocks: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    triangle = np.triu_indices(surface.shape[1])
    products = np.column_stack(
        [
            np.bincount(
                row_blocks,
                weights=surface[:, first] * surface[:, second],
                minlength=blocks,
            )
            for first, second in zip(*triangle)
        ]
    )
    sums = np.column_stack(
        [
            np.bincount(row_blocks, weights=surface[:, column], minlength=blocks)
            for column in range(surface.shape[1])
        ]
    )
    counts = np.bincount(row_blocks, minlength=blocks).astype(np.float64)
    return products, sums, counts


def bootstrap_surface_agreement(
    surface: np.ndarray,
    row_blocks: np.ndarray,
    rank_vectors: "OrderedDict[str, np.ndarray]",
    repeats: int,
    seed: int,
) -> dict[str, np.ndarray]:
    """Bayesian bootstrap draws for one surface using a reproducible block partition."""
    blocks = int(row_blocks.max() + 1)
    products, sums, counts = block_moments(surface, row_blocks, blocks)
    target_triangle = np.triu_indices(surface.shape[1], k=1)
    moment_triangle = np.triu_indices(surface.shape[1])
    draws = {panel: np.empty(repeats, dtype=np.float64) for panel in PANELS}
    rng = np.random.default_rng(seed)
    position = 0
    while position < repeats:
        size = min(BOOTSTRAP_CHUNK, repeats - position)
        weights = rng.exponential(size=(size, blocks))
        weighted_products = weights @ products
        weighted_sums = weights @ sums
        weighted_counts = weights @ counts
        for offset in range(size):
            cross = np.zeros((surface.shape[1], surface.shape[1]), dtype=np.float64)
            cross[moment_triangle] = weighted_products[offset]
            cross[(moment_triangle[1], moment_triangle[0])] = weighted_products[offset]
            correlation = target_blind.correlation_from_moments(
                cross, weighted_sums[offset], float(weighted_counts[offset])
            )
            edges = correlation[target_triangle]
            for panel in PANELS:
                value = target_blind.spearman_against(edges, rank_vectors[panel])
                if value is None:
                    raise ValueError("bootstrap produced a degenerate geometry")
                draws[panel][position + offset] = value
        position += size
    return draws


def paired_block_bootstrap(
    surfaces: "OrderedDict[str, np.ndarray]",
    groupings: "OrderedDict[str, np.ndarray]",
    rank_vectors: "OrderedDict[str, np.ndarray]",
    repeats: int,
    requested_blocks: int,
    seed: int,
) -> tuple[dict[str, object], pd.DataFrame]:
    """Paired physicochemical-minus-control geometry intervals."""
    per_grouping: dict[str, object] = OrderedDict()
    rows: list[dict[str, object]] = []
    all_draws: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    for grouping_index, (grouping, labels) in enumerate(groupings.items()):
        assignment_seed = seed + grouping_index * 1_000_000
        draw_seed = seed + grouping_index * 1_000_000 + 100_000
        row_blocks, block_report = balanced_group_blocks(
            labels, requested_blocks, assignment_seed
        )
        draws = {
            name: bootstrap_surface_agreement(
                surface, row_blocks, rank_vectors, repeats, draw_seed
            )
            for name, surface in surfaces.items()
        }
        all_draws[grouping] = draws
        per_grouping[grouping] = {
            **block_report,
            "repeats": int(repeats),
            "draw_seed": int(draw_seed),
            "arms": {
                name: {
                    panel: describe_conditional_sensitivity(draws[name][panel])
                    for panel in PANELS
                }
                for name in surfaces
            },
        }

    contrasts: dict[str, object] = OrderedDict()
    for control in tuple(name for name in surfaces if name != PHYSICOCHEMICAL):
        contrasts[control] = {}
        for panel in PANELS:
            intervals = []
            for grouping, draws in all_draws.items():
                difference = (
                    draws[PHYSICOCHEMICAL][panel] - draws[control][panel]
                )
                described = describe_conditional_sensitivity(difference)
                intervals.append(
                    described[
                        "conditional_coarsened_sensitivity_interval_95"
                    ]
                )
                rows.append(
                    {
                        "control": control,
                        "panel": panel,
                        "grouping": grouping,
                        "physicochemical_minus_control_point": np.nan,
                        "bootstrap_mean": described["mean"],
                        "bootstrap_median": described["median"],
                        "conditional_coarsened_sensitivity_low95": described[
                            "conditional_coarsened_sensitivity_interval_95"
                        ][0],
                        "conditional_coarsened_sensitivity_high95": described[
                            "conditional_coarsened_sensitivity_interval_95"
                        ][1],
                    }
                )
            contrasts[control][panel] = {
                "conservative_union_sensitivity_interval_95": (
                    target_blind.conservative_union(intervals)
                )
            }
    return {
        "method": (
            "paired Bayesian bootstrap with iid Exp(1) weights over balanced blocks of "
            "intact chemical groups"
        ),
        "coarsening_limitation": (
            "individual chemical groups are packed into balanced blocks for tractability; "
            "unrelated groups in the same block share a weight, so this is a coarsened "
            "chemical-group sensitivity interval rather than an individual-cluster "
            "full-refit confidence interval"
        ),
        "per_grouping": per_grouping,
        "physicochemical_minus_control": contrasts,
    }, pd.DataFrame.from_records(rows)


def mean_squared_correlation_from_moments(
    weighted_products: np.ndarray,
    weighted_sums: np.ndarray,
    weighted_count: float,
    targets: int,
) -> float:
    """Mean squared off-diagonal correlation from packed raw moments."""
    moment_triangle = np.triu_indices(targets)
    cross = np.zeros((targets, targets), dtype=np.float64)
    cross[moment_triangle] = weighted_products
    cross[(moment_triangle[1], moment_triangle[0])] = weighted_products
    correlation = target_blind.correlation_from_moments(
        cross, weighted_sums, weighted_count
    )
    edges = correlation[np.triu_indices(targets, k=1)]
    return float(np.mean(np.square(edges)))


def paired_operational_reduction_bootstrap(
    observed: np.ndarray,
    predictions: "OrderedDict[str, np.ndarray]",
    groupings: "OrderedDict[str, np.ndarray]",
    *,
    repeats: int,
    requested_blocks: int,
    seed: int,
) -> tuple[dict[str, object], pd.DataFrame]:
    """Paired conditional bootstrap of the exact correlation-reduction contrast.

    In every draw the same Exp(1) block weights are used for the observed denominator and
    every arm's error numerator.  The learned OOF prediction surfaces remain fixed.
    """
    per_grouping: dict[str, object] = OrderedDict()
    contrast_rows: list[dict[str, object]] = []
    point = {
        name: predictive_metrics(observed, prediction)[
            "relative_reduction_in_mean_squared_offdiagonal_target_correlation"
        ]
        for name, prediction in predictions.items()
    }
    for grouping_index, (grouping, labels) in enumerate(groupings.items()):
        assignment_seed = seed + grouping_index * 1_000_000
        draw_seed = seed + grouping_index * 1_000_000 + 100_000
        row_blocks, block_report = balanced_group_blocks(
            labels, requested_blocks, assignment_seed
        )
        blocks = int(block_report["bootstrap_blocks"])
        observed_moments = block_moments(observed, row_blocks, blocks)
        error_moments = OrderedDict(
            (
                name,
                block_moments(observed - prediction, row_blocks, blocks),
            )
            for name, prediction in predictions.items()
        )
        draws = {
            name: np.empty(repeats, dtype=np.float64) for name in predictions
        }
        rng = np.random.default_rng(draw_seed)
        position = 0
        while position < repeats:
            size = min(BOOTSTRAP_CHUNK, repeats - position)
            weights = rng.exponential(size=(size, blocks))
            before_products = weights @ observed_moments[0]
            before_sums = weights @ observed_moments[1]
            before_counts = weights @ observed_moments[2]
            before = np.empty(size, dtype=np.float64)
            for offset in range(size):
                before[offset] = mean_squared_correlation_from_moments(
                    before_products[offset],
                    before_sums[offset],
                    float(before_counts[offset]),
                    observed.shape[1],
                )
            for name, moments in error_moments.items():
                after_products = weights @ moments[0]
                after_sums = weights @ moments[1]
                after_counts = weights @ moments[2]
                for offset in range(size):
                    after = mean_squared_correlation_from_moments(
                        after_products[offset],
                        after_sums[offset],
                        float(after_counts[offset]),
                        observed.shape[1],
                    )
                    draws[name][position + offset] = 1.0 - after / before[offset]
            position += size

        arm_report = {
            name: {
                "point": float(point[name]),
                **describe_conditional_sensitivity(values),
            }
            for name, values in draws.items()
        }
        contrasts: dict[str, object] = OrderedDict()
        for control in predictions:
            if control == PHYSICOCHEMICAL:
                continue
            difference = draws[PHYSICOCHEMICAL] - draws[control]
            described = describe_conditional_sensitivity(difference)
            contrasts[control] = {
                "point": float(point[PHYSICOCHEMICAL] - point[control]),
                **described,
            }
            contrast_rows.append(
                {
                    "grouping": grouping,
                    "control": control,
                    "physicochemical_minus_control_point": (
                        point[PHYSICOCHEMICAL] - point[control]
                    ),
                    "bootstrap_mean": described["mean"],
                    "bootstrap_median": described["median"],
                    "conditional_coarsened_sensitivity_low95": described[
                        "conditional_coarsened_sensitivity_interval_95"
                    ][0],
                    "conditional_coarsened_sensitivity_high95": described[
                        "conditional_coarsened_sensitivity_interval_95"
                    ][1],
                }
            )
        per_grouping[grouping] = {
            **block_report,
            "repeats": int(repeats),
            "draw_seed": int(draw_seed),
            "arms": arm_report,
            "physicochemical_minus_control": contrasts,
        }

    conservative: dict[str, object] = OrderedDict()
    for control in predictions:
        if control == PHYSICOCHEMICAL:
            continue
        intervals = [
            per_grouping[grouping]["physicochemical_minus_control"][control][
                "conditional_coarsened_sensitivity_interval_95"
            ]
            for grouping in groupings
        ]
        conservative[control] = {
            "point": float(point[PHYSICOCHEMICAL] - point[control]),
            "conservative_union_sensitivity_interval_95": (
                target_blind.conservative_union(intervals)
            ),
        }
    return {
        "estimand": (
            "1 - weighted mean squared off-diagonal error correlation / weighted mean "
            "squared off-diagonal observed correlation"
        ),
        "method": (
            "paired Bayesian bootstrap with iid Exp(1) weights over balanced blocks of "
            "intact chemical groups; every arm and both parts of its ratio use the same "
            "draw weights"
        ),
        "conditioning": (
            "OOF prediction surfaces, experimental edges and target panel are fixed; "
            "predictive models are not refitted inside bootstrap draws"
        ),
        "coarsening_limitation": (
            "unrelated chemical groups packed in the same balanced block share a weight; "
            "intervals are conditional coarsened sensitivity intervals, not individual-"
            "cluster, chemical-space or target-population confidence intervals"
        ),
        "per_grouping": per_grouping,
        "physicochemical_minus_control_conservative_union": conservative,
    }, pd.DataFrame.from_records(contrast_rows)


def evaluate(
    block: np.ndarray,
    bases: "OrderedDict[str, np.ndarray]",
    count_morgan_ensemble_features: np.ndarray,
    count_morgan_seeds: tuple[int, ...],
    groups: np.ndarray,
    ledger: pd.DataFrame,
    generic_groups: np.ndarray,
    topology_holdout_groups: np.ndarray,
    *,
    folds: int,
    bootstrap_repeats: int,
    bootstrap_blocks: int,
    bootstrap_seed: int,
) -> tuple[dict[str, object], dict[str, pd.DataFrame]]:
    fitted = fit_predictive_bases(block, bases, groups, folds)
    observed = fitted["observed"]
    surfaces: "OrderedDict[str, np.ndarray]" = OrderedDict(fitted["predictions"])
    surfaces[ORACLE] = fitted["capacity"]
    rank_vectors = target_blind.experimental_rank_vectors(ledger)
    triangle = np.triu_indices(len(TARGETS20), k=1)

    matrices = OrderedDict(
        [(OBSERVED, geometry.target_correlation(observed))]
        + [(name, geometry.target_correlation(surface)) for name, surface in surfaces.items()]
    )
    agreements: dict[str, dict[str, float]] = OrderedDict()
    for name, matrix in matrices.items():
        edges = matrix[triangle]
        agreements[name] = {}
        for panel in PANELS:
            value = target_blind.spearman_against(edges, rank_vectors[panel])
            if value is None:
                raise ValueError(f"{name} produced a degenerate geometry")
            agreements[name][panel] = value

    metrics = {
        name: predictive_metrics(observed, surface)
        for name, surface in fitted["predictions"].items()
    }
    capacity_error = observed - fitted["capacity"]
    capacity_metrics = predictive_metrics(observed, fitted["capacity"])
    capacity_metrics["role"] = (
        "outcome-derived fold-local rank-7 transported-subspace oracle reconstruction; "
        "not an OOF ligand-feature predictor because each test row's docking outcomes "
        "are used for projection, and not a universal rank-7 capacity ceiling"
    )

    ensemble, ensemble_tables = evaluate_count_morgan_ensemble(
        block,
        count_morgan_ensemble_features,
        count_morgan_seeds,
        groups,
        ledger,
        folds,
    )
    member_details = ensemble["members_detail"]
    operational_key = (
        "relative_reduction_in_mean_squared_offdiagonal_target_correlation"
    )
    ensemble_operational = np.asarray(
        [
            member["predictive_metrics"][operational_key]
            for member in member_details
        ],
        dtype=np.float64,
    )
    physical_operational = float(metrics[PHYSICOCHEMICAL][operational_key])
    algorithmic_distributions: dict[str, object] = {
        "relative_reduction_in_mean_squared_offdiagonal_target_correlation": {
            "count_morgan_seed_distribution": describe_algorithmic_seed_distribution(
                ensemble_operational
            ),
            "count_morgan_seed_minimum": float(ensemble_operational.min()),
            "count_morgan_seed_maximum": float(ensemble_operational.max()),
            "physicochemical_point": physical_operational,
            "physicochemical_minus_count_morgan_distribution": (
                describe_algorithmic_seed_distribution(
                    physical_operational - ensemble_operational
                )
            ),
            "physicochemical_exceeds_all_ensemble_members": bool(
                physical_operational > ensemble_operational.max()
            ),
            "ensemble_members_at_least_as_large_as_physicochemical": int(
                np.sum(ensemble_operational >= physical_operational)
            ),
        },
        "panel_geometry_agreement": OrderedDict(),
    }
    for panel in PANELS:
        values = np.asarray(
            [
                member["panel_geometry_agreement"][panel]
                for member in member_details
            ],
            dtype=np.float64,
        )
        physical = float(agreements[PHYSICOCHEMICAL][panel])
        algorithmic_distributions["panel_geometry_agreement"][panel] = {
            "count_morgan_seed_distribution": describe_algorithmic_seed_distribution(
                values
            ),
            "count_morgan_seed_minimum": float(values.min()),
            "count_morgan_seed_maximum": float(values.max()),
            "physicochemical_point": physical,
            "physicochemical_minus_count_morgan_distribution": (
                describe_algorithmic_seed_distribution(physical - values)
            ),
            "physicochemical_exceeds_all_ensemble_members": bool(
                physical > values.max()
            ),
            "ensemble_members_at_least_as_large_as_physicochemical": int(
                np.sum(values >= physical)
            ),
        }
    ensemble["algorithmic_distributions"] = algorithmic_distributions
    locked_member = member_details[0]
    ensemble["locked_representative_reproduction"] = {
        "maximum_absolute_predictive_metric_difference": float(
            max(
                abs(
                    float(locked_member["predictive_metrics"][key])
                    - float(metrics[COUNT_MORGAN_LOCKED][key])
                )
                for key in locked_member["predictive_metrics"]
                if isinstance(locked_member["predictive_metrics"][key], (int, float))
            )
        ),
        "maximum_absolute_panel_geometry_difference": float(
            max(
                abs(
                    locked_member["panel_geometry_agreement"][panel]
                    - agreements[COUNT_MORGAN_LOCKED][panel]
                )
                for panel in PANELS
            )
        ),
    }

    # Recompute the established physical estimator rather than trusting an in-memory alias.
    # The historical artifact stores geometry summaries, not its ligand-level OOF surface.
    established = mechanism.grouped_descriptor_decomposition(
        block, bases[PHYSICOCHEMICAL], groups, folds
    )
    physical_reproduction = {
        "reference": "residual_mechanism_analysis.grouped_descriptor_decomposition",
        "artifact_limitation": (
            "the established result artifact does not serialize ligand-level OOF values; "
            "the established implementation is therefore rerun on the locked support"
        ),
        "maximum_absolute_difference_observed_surface": float(
            np.abs(established["oof_surface"] - observed).max()
        ),
        "maximum_absolute_difference_descriptor_prediction": float(
            np.abs(
                established["oof_prediction"]
                - fitted["predictions"][PHYSICOCHEMICAL]
            ).max()
        ),
    }
    if (
        physical_reproduction[
            "maximum_absolute_difference_descriptor_prediction"
        ]
        > 1e-9
    ):
        raise AssertionError("physical OOF prediction does not reproduce the established estimator")
    del established

    # Predictive sensitivity: refit every ligand-feature arm under a grouping that groups
    # acyclic analogues by generic heavy-atom topology instead of treating them as
    # independent singleton scaffolds.  This is one deterministic GroupKFold partition,
    # so it is reported as a sensitivity analysis rather than as inferential uncertainty.
    sensitivity_fit = fit_predictive_bases(
        block, bases, topology_holdout_groups, folds
    )
    sensitivity_surfaces = OrderedDict(sensitivity_fit["predictions"])
    sensitivity_matrices = OrderedDict(
        [(OBSERVED, geometry.target_correlation(sensitivity_fit["observed"]))]
        + [
            (name, geometry.target_correlation(surface))
            for name, surface in sensitivity_surfaces.items()
        ]
    )
    sensitivity_agreements: dict[str, dict[str, float]] = OrderedDict()
    for name, matrix in sensitivity_matrices.items():
        sensitivity_agreements[name] = {}
        edges = matrix[triangle]
        for panel in PANELS:
            value = target_blind.spearman_against(edges, rank_vectors[panel])
            if value is None:
                raise ValueError(f"{name} sensitivity geometry is degenerate")
            sensitivity_agreements[name][panel] = value
    sensitivity_metrics = {
        name: predictive_metrics(sensitivity_fit["observed"], surface)
        for name, surface in sensitivity_surfaces.items()
    }

    # Component/remainder dependence and partial associations are retained for every
    # predictive basis so none of the separate agreements is mistaken for an additive
    # decomposition.
    edge_rows: list[dict[str, object]] = []
    dependence: dict[str, object] = OrderedDict()
    for name, prediction in fitted["predictions"].items():
        predicted_edges = matrices[name][triangle]
        remainder_edges = geometry.target_correlation(observed - prediction)[triangle]
        dependence[name] = {
            "component_remainder_spearman": float(
                pd.Series(predicted_edges).corr(pd.Series(remainder_edges), method="spearman")
            ),
            "component_remainder_pearson": float(
                np.corrcoef(predicted_edges, remainder_edges)[0, 1]
            ),
            "panels": {},
        }
        for panel in PANELS:
            experimental = ledger[
                f"{panel}_experimental_centered_correlation"
            ].to_numpy(dtype=np.float64)
            fit = edge_decomposition.joint_edge_regression(
                experimental, predicted_edges, remainder_edges
            )
            record = {
                "partial_spearman_component": edge_decomposition.partial_spearman(
                    experimental, predicted_edges, remainder_edges
                ),
                "partial_spearman_remainder": edge_decomposition.partial_spearman(
                    experimental, remainder_edges, predicted_edges
                ),
                "standardized_beta_component": fit["beta_first"],
                "standardized_beta_remainder": fit["beta_second"],
                "joint_rank_r2": fit["r2_joint"],
                "semipartial_rank_r2_component": fit["semipartial_r2_first"],
                "semipartial_rank_r2_remainder": fit["semipartial_r2_second"],
            }
            dependence[name]["panels"][panel] = record
            edge_rows.append({"basis": name, "panel": panel, **record})

    bootstrap_groupings = OrderedDict(
        [
            ("murcko_scaffold", groups),
            ("murcko_generic_framework", generic_groups),
            (
                "generic_murcko_or_acyclic_topology",
                topology_holdout_groups,
            ),
        ]
    )
    bootstrap, contrast_frame = paired_block_bootstrap(
        surfaces,
        bootstrap_groupings,
        rank_vectors,
        bootstrap_repeats,
        bootstrap_blocks,
        bootstrap_seed,
    )
    reduction_bootstrap, reduction_contrast_frame = (
        paired_operational_reduction_bootstrap(
            observed,
            surfaces,
            bootstrap_groupings,
            repeats=bootstrap_repeats,
            requested_blocks=bootstrap_blocks,
            seed=bootstrap_seed + 10_000_000,
        )
    )
    for control in surfaces:
        if control == PHYSICOCHEMICAL:
            continue
        for panel in PANELS:
            point = agreements[PHYSICOCHEMICAL][panel] - agreements[control][panel]
            bootstrap["physicochemical_minus_control"][control][panel][
                "point"
            ] = point
            mask = contrast_frame.control.eq(control) & contrast_frame.panel.eq(panel)
            contrast_frame.loc[mask, "physicochemical_minus_control_point"] = point

    agreement_rows = [
        {"arm": arm, "panel": panel, "spearman": value}
        for arm, panel_values in agreements.items()
        for panel, value in panel_values.items()
    ]
    metric_rows = [
        {
            "basis": name,
            **{
                key: value
                for key, value in record.items()
                if isinstance(value, (int, float))
            },
        }
        for name, record in {**metrics, ORACLE: capacity_metrics}.items()
    ]

    summary: dict[str, object] = {
        "analysis": "rank/dimension-matched controls for fixed-20 descriptor geometry",
        "status": "exploratory_post_hoc_reviewer_requested_controls",
        "estimands": {
            "predictive_score_structure": (
                "relative reduction in mean squared off-diagonal target correlation after "
                "subtracting a fixed group-held-out prediction; not an explained fraction"
            ),
            "experimental_geometry": (
                "Spearman correlation between the 190 predicted target-pair edges and a "
                "fixed centered experimental edge vector"
            ),
            "paired_design_sensitivity": (
                "physicochemical-minus-control Spearman difference under coarsened "
                "chemical-group block reweighting, with the target panel and experimental "
                "edges fixed"
            ),
        },
        "arms": {
            PHYSICOCHEMICAL: "seven published RDKit physicochemical descriptors",
            COUNT_MORGAN_LOCKED: (
                "deterministically locked first-seed representative of the 20-seed 7-D "
                "unnormalized "
                "Morgan count-fingerprint random-projection ensemble"
            ),
            MORGAN_A: (
                "first secondary 7-D projection of normalized Morgan bit fingerprints"
            ),
            MORGAN_B: (
                "second secondary 7-D projection of normalized Morgan bit fingerprints"
            ),
            HASH_NUISANCE: "stable 7-D ligand-identity hash nuisance basis",
            ORACLE: (
                "outcome-derived fold-local rank-7 transported-subspace oracle; neither "
                "predictive nor a universal capacity ceiling"
            ),
        },
        "predictive_metrics": metrics,
        "outcome_rank7_svd_transported_oracle": {
            **capacity_metrics,
            "folds": fitted["fold_rows"],
        },
        "panel_geometry_agreement": agreements,
        "component_remainder_and_partial_associations": dependence,
        "morgan_count_plus_size_rp_7_seed_ensemble": ensemble,
        "coefficient_effective_ranks_by_fold": fitted["coefficient_effective_ranks"],
        "generic_murcko_acyclic_topology_predictive_sensitivity": {
            "role": (
                "all predictive ligand-feature arms refitted under one deterministic "
                "five-fold grouping that clusters acyclic ligands by generic heavy-atom "
                "topology"
            ),
            "uncertainty_boundary": (
                "single deterministic GroupKFold sensitivity; differences from the primary "
                "Murcko split have no calibrated confidence interval"
            ),
            "panel_geometry_agreement": sensitivity_agreements,
            "predictive_metrics": sensitivity_metrics,
            "coefficient_effective_ranks_by_fold": sensitivity_fit[
                "coefficient_effective_ranks"
            ],
            "folds": sensitivity_fit["fold_rows"],
        },
        "paired_chemical_group_block_bootstrap": bootstrap,
        "operational_reduction_paired_block_bootstrap": reduction_bootstrap,
        "configuration": {
            "folds": int(folds),
            "features_per_predictive_basis": DEFAULT_FEATURES,
            "holdout_group": (
                "RDKit Bemis-Murcko scaffold; every acyclic molecule is a singleton"
            ),
            "bootstrap_repeats": int(bootstrap_repeats),
            "requested_bootstrap_blocks": int(bootstrap_blocks),
            "bootstrap_seed": int(bootstrap_seed),
        },
        "verification": {
            "physical_oof_reproduction": physical_reproduction,
            "physical_prediction_reproduction_max_abs_difference": physical_reproduction[
                "maximum_absolute_difference_descriptor_prediction"
            ],
            "observed_surface_is_identical_across_all_arms": True,
            "every_predictive_basis_has_seven_columns": True,
            "transported_oracle_uses_test_outcomes_for_projection": True,
            "locked_count_morgan_ensemble_reproduction": ensemble[
                "locked_representative_reproduction"
            ],
        },
        "claim_boundary": (
            "These controls do NOT identify a causal physicochemical mechanism. Morgan "
            "random projections retain chemical topology, the stable-hash basis is only a "
            "negative control, and every predictive arm still fits separate coefficients "
            "for every target. The rank-7 SVD arm is not a predictor: held-out docking "
            "outcomes are used for projection. Relative reduction in mean squared target "
            "correlation is a non-additive operational contrast, not a share of correlation "
            "explained. Bootstrap intervals reweight coarsened blocks of chemical groups "
            "with the experimental geometries and target panel fixed; they are sensitivity "
            "intervals, not experimental- or target-population confidence intervals. The "
            "20-seed count-Morgan range is algorithmic projection sensitivity, not "
            "uncertainty. It tests basis specificity only for this fixed common-20, "
            "de-leaked DOCKSTRING geometry and is not extrapolated to the Docking-44 or "
            "DOCKSTRING-58 headline supports. Because the same 20 targets and 190 dependent "
            "dyads are fixed, physicochemical-versus-RP experimental-agreement differences "
            "are descriptive and conditional, not inferential significance tests. "
            "Physicochemical feature specificity is not claimed solely from a finite "
            "random-projection ensemble, especially wherever an ensemble member matches "
            "or exceeds the physicochemical point."
        ),
    }
    summary = refresh_interpretive_fields(summary)
    tables = {
        "panel_geometry_agreement.csv": pd.DataFrame.from_records(agreement_rows),
        "predictive_metrics.csv": pd.DataFrame.from_records(metric_rows),
        "paired_geometry_differences.csv": contrast_frame,
        "paired_operational_reduction_differences.csv": reduction_contrast_frame,
        "component_remainder_partial_associations.csv": pd.DataFrame.from_records(
            edge_rows
        ),
        "fold_oracle_reconstruction_diagnostics.csv": pd.DataFrame.from_records(
            fitted["fold_rows"]
        ),
        "topology_holdout_sensitivity_panel_geometry.csv": pd.DataFrame.from_records(
            [
                {"arm": arm, "panel": panel, "spearman": value}
                for arm, panel_values in sensitivity_agreements.items()
                for panel, value in panel_values.items()
            ]
        ),
        "topology_holdout_sensitivity_predictive_metrics.csv": pd.DataFrame.from_records(
            [
                {
                    "basis": name,
                    **{
                        key: value
                        for key, value in record.items()
                        if isinstance(value, (int, float))
                    },
                }
                for name, record in sensitivity_metrics.items()
            ]
        ),
        "topology_holdout_sensitivity_folds.csv": pd.DataFrame.from_records(
            sensitivity_fit["fold_rows"]
        ),
    }
    tables.update(ensemble_tables)
    return summary, tables


def run(
    *,
    pkis1_zip: Path,
    ledger_path: Path,
    folds: int,
    normalized_morgan_seeds: tuple[int, int],
    count_morgan_seeds: tuple[int, ...],
    nuisance_permutation_seeds: tuple[int, ...],
    hash_seed: int,
    bootstrap_repeats: int,
    bootstrap_blocks: int,
    bootstrap_seed: int,
) -> tuple[dict[str, object], dict[str, pd.DataFrame]]:
    if not count_morgan_seeds or count_morgan_seeds[0] != DEFAULT_COUNT_MORGAN_SEEDS[0]:
        raise ValueError(
            "the locked count-Morgan representative must remain the deterministic first "
            f"seed {DEFAULT_COUNT_MORGAN_SEEDS[0]}"
        )
    reference, score_columns, smiles, support = component.load_deleaked_reference(
        pkis1_zip
    )
    score_index = {target: position for position, target in enumerate(score_columns)}
    block = np.ascontiguousarray(
        reference[:, [score_index[target] for target in TARGETS20]]
    )
    del reference
    physical = mechanism.molecular_descriptors(smiles)
    groups = mechanism.scaffold_keys(smiles)
    morgan_a, morgan_b, count_morgan_ensemble, morgan_report = (
        morgan_random_projection_features(
            smiles, normalized_morgan_seeds, count_morgan_seeds
        )
    )
    nuisance = stable_hash_nuisance_features(smiles, hash_seed)
    if not all(
        np.isfinite(array).all()
        for array in (
            physical,
            morgan_a,
            morgan_b,
            count_morgan_ensemble,
            nuisance,
        )
    ):
        raise AssertionError("validated feature construction returned non-finite values")
    bases = OrderedDict(
        [
            (PHYSICOCHEMICAL, physical),
            (COUNT_MORGAN_LOCKED, count_morgan_ensemble[:, 0, :]),
            (MORGAN_A, morgan_a),
            (MORGAN_B, morgan_b),
            (HASH_NUISANCE, nuisance),
        ]
    )
    ledger = target_blind.load_ledger(ledger_path)
    generic, generic_failures = target_blind.generic_framework_keys(groups)
    topology_holdout, topology_report = generic_murcko_acyclic_topology_keys(
        smiles, groups, generic
    )
    summary, tables = evaluate(
        block,
        bases,
        count_morgan_ensemble,
        count_morgan_seeds,
        groups,
        ledger,
        generic,
        topology_holdout,
        folds=folds,
        bootstrap_repeats=bootstrap_repeats,
        bootstrap_blocks=bootstrap_blocks,
        bootstrap_seed=bootstrap_seed,
    )
    nuisance_ensemble, nuisance_table = evaluate_row_permutation_nuisance_ensemble(
        block,
        physical,
        nuisance_permutation_seeds,
        groups,
        ledger,
        folds,
    )
    summary["physicochemical_row_permutation_nuisance_ensemble"] = nuisance_ensemble
    tables["physicochemical_row_permutation_nuisance_ensemble.csv"] = nuisance_table
    summary = refresh_interpretive_fields(summary)
    diagnostics = feature_diagnostics(bases, physical)
    summary["support"] = {
        **support,
        "ligands": int(len(block)),
        "targets": int(block.shape[1]),
        "cells": int(block.size),
        "murcko_groups": int(len(np.unique(groups))),
        "generic_murcko_groups": int(len(np.unique(generic))),
        "generic_framework_failures": int(generic_failures),
        "generic_murcko_acyclic_topology_groups": int(
            len(np.unique(topology_holdout))
        ),
    }
    summary["generic_murcko_acyclic_topology_grouping"] = topology_report
    summary["feature_construction"] = {
        "morgan_random_projections": morgan_report,
        "stable_hash_nuisance_seed": int(hash_seed),
        "stable_hash_nuisance_definition": (
            "keyed BLAKE2b digest of the exact input SMILES mapped to seven centered "
            "uniform coordinates; duplicates receive identical coordinates"
        ),
        "diagnostics": diagnostics.to_dict(orient="records"),
    }
    tables["feature_diagnostics.csv"] = diagnostics
    return summary, tables


def run_nuisance_extension(
    *,
    pkis1_zip: Path,
    ledger_path: Path,
    folds: int,
    nuisance_permutation_seeds: tuple[int, ...],
) -> tuple[dict[str, object], pd.DataFrame, int]:
    """Compute only the row-permutation ensemble for an existing full artifact."""
    reference, score_columns, smiles, _ = component.load_deleaked_reference(pkis1_zip)
    score_index = {target: position for position, target in enumerate(score_columns)}
    block = np.ascontiguousarray(
        reference[:, [score_index[target] for target in TARGETS20]]
    )
    del reference
    physical = mechanism.molecular_descriptors(smiles)
    groups = mechanism.scaffold_keys(smiles)
    ledger = target_blind.load_ledger(ledger_path)
    report, table = evaluate_row_permutation_nuisance_ensemble(
        block,
        physical,
        nuisance_permutation_seeds,
        groups,
        ledger,
        folds,
    )
    return report, table, len(block)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pkis1-zip", type=Path, default=DEFAULT_PKIS1_ZIP)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--folds", type=int, default=DEFAULT_FOLDS)
    parser.add_argument(
        "--normalized-morgan-seeds",
        type=int,
        nargs=2,
        default=DEFAULT_NORMALIZED_MORGAN_SEEDS,
        metavar=("SEED_A", "SEED_B"),
    )
    parser.add_argument(
        "--count-morgan-seeds",
        type=int,
        nargs="+",
        default=DEFAULT_COUNT_MORGAN_SEEDS,
    )
    parser.add_argument(
        "--nuisance-permutation-seeds",
        type=int,
        nargs="+",
        default=DEFAULT_NUISANCE_PERMUTATION_SEEDS,
    )
    parser.add_argument("--hash-seed", type=int, default=DEFAULT_HASH_SEED)
    parser.add_argument(
        "--bootstrap-repeats", type=int, default=DEFAULT_BOOTSTRAP_REPEATS
    )
    parser.add_argument("--bootstrap-blocks", type=int, default=DEFAULT_BOOTSTRAP_BLOCKS)
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--nuisance-extension-only",
        action="store_true",
        help=(
            "compute and attach only the row-permutation nuisance ensemble to an "
            "existing full artifact"
        ),
    )
    args = parser.parse_args()
    if args.nuisance_extension_only:
        summary_path = args.output / "summary.json"
        if not summary_path.exists():
            raise FileNotFoundError(
                "--nuisance-extension-only requires an existing summary.json"
            )
        summary = json.loads(summary_path.read_text())
        nuisance_report, nuisance_table, ligands = run_nuisance_extension(
            pkis1_zip=args.pkis1_zip,
            ledger_path=args.ledger,
            folds=args.folds,
            nuisance_permutation_seeds=tuple(args.nuisance_permutation_seeds),
        )
        if int(summary["support"]["ligands"]) != ligands:
            raise ValueError("existing summary support differs from nuisance extension")
        if int(summary["configuration"]["folds"]) != args.folds:
            raise ValueError("existing summary fold count differs from nuisance extension")
        summary["physicochemical_row_permutation_nuisance_ensemble"] = nuisance_report
        summary = refresh_interpretive_fields(summary)
        mechanism.write_json(summary_path, summary)
        mechanism.write_csv(
            nuisance_table,
            args.output / "physicochemical_row_permutation_nuisance_ensemble.csv",
        )
        print(nuisance_table.to_string(index=False))
        return
    summary, tables = run(
        pkis1_zip=args.pkis1_zip,
        ledger_path=args.ledger,
        folds=args.folds,
        normalized_morgan_seeds=tuple(args.normalized_morgan_seeds),
        count_morgan_seeds=tuple(args.count_morgan_seeds),
        nuisance_permutation_seeds=tuple(args.nuisance_permutation_seeds),
        hash_seed=args.hash_seed,
        bootstrap_repeats=args.bootstrap_repeats,
        bootstrap_blocks=args.bootstrap_blocks,
        bootstrap_seed=args.bootstrap_seed,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    mechanism.write_json(args.output / "summary.json", summary)
    for name, frame in tables.items():
        mechanism.write_csv(frame, args.output / name)
    print(tables["panel_geometry_agreement.csv"].to_string(index=False))


if __name__ == "__main__":
    main()
