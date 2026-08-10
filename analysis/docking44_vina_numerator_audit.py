#!/usr/bin/env python3
"""Audit Vina's intermolecular numerator and torsional normalisation in Docking-44.

The frozen Docking-44 score matrix does not retain poses or the five individual
Vina terms.  A separate upstream artifact, ``residue_matrices/``, does retain a
weighted per-residue intermolecular-energy matrix for 43 of the 44 primary
targets.  Summing each residue matrix therefore supplies a derived
intermolecular numerator on the same 12,651 ligands.

This script answers a deliberately narrow question: how much of the raw-to-
two-way-centred PR change is already present in that numerator, and how much is
associated with Vina's ligand-specific torsional normalisation?

Important support boundary
--------------------------
``V1A`` is excluded because no exact ``V1A_residue_matrix.csv`` exists.
``9uwl_residue_matrix.csv`` is not substituted: there is no scalar ``9uwl``
column in the frozen table, so doing so would silently compare different data
keys.  All headline comparisons below use the exact 43-target overlap and the
scalar matrix's original missing/censoring mask.

The inferred factor is a diagnostic reconstruction, not independent evidence:

    q_i = median_j(E_ij / S_ij),   S_ij approximately E_ij / q_i,

where the median uses finite, strictly negative scalar-score cells.  The
rotatable-bond factor ``1 + 0.05846 * rotatable_bonds`` is reported separately
as the non-circular ligand-descriptor sensitivity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from scipy import stats


RDLogger.DisableLog("rdApp.*")

PACKAGE = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PACKAGE.parent
DEFAULT_SCALAR_TABLE = PACKAGE / "data" / "frozen" / "df_final_v4.csv.gz"
DEFAULT_RESIDUE_DIR = WORKSPACE_ROOT / "residue_matrices"
DEFAULT_OUTPUT_DIR = PACKAGE / "results" / "docking44_vina_numerator"

ROT_WEIGHT = 0.05846

DOCK44 = (
    "1m2z", "1pbq", "1xoq", "2rh1", "2vt4", "2ydo", "2z5x", "3b66",
    "3kk6", "3ln1", "3rze", "4djh", "4ey7", "4iar", "4mqs", "4n6h",
    "5cxv", "5i71", "5tvn", "5u09", "5va1", "6cm4", "6kpf", "6kux",
    "6lqa", "6pdj", "6x3x", "6y1z", "7f8y", "7kwe", "7ljd", "7wc9",
    "7xnk", "7ym8", "8e9y", "8ef6", "8fhs", "8pjk", "8st0", "8wty",
    "8xvk", "8yn3", "9eo4", "V1A",
)

NUMERATOR_TARGETS = tuple(target for target in DOCK44 if target != "V1A")

SUPPORT_CAVEAT = (
    "The numerator analysis is conditional on the exact 43-target overlap. V1A "
    "is excluded because no exact V1A residue matrix exists. The available "
    "9uwl residue matrix is not substituted because the frozen scalar table has "
    "no 9uwl score column. Residue row sums are derived upstream values rather "
    "than a lossless pose-level decomposition; the scalar observation/censoring "
    "mask is therefore preserved before every matched comparison."
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def portable_path(path: Path) -> str:
    path = path.resolve()
    for root in (WORKSPACE_ROOT.resolve(), PACKAGE.resolve()):
        try:
            return str(path.relative_to(root))
        except ValueError:
            pass
    return str(path)


def mean_impute_columns(matrix: np.ndarray) -> np.ndarray:
    """Fill non-finite cells with the finite mean of their target column."""
    work = np.asarray(matrix, dtype=np.float64).copy()
    means = np.nanmean(np.where(np.isfinite(work), work, np.nan), axis=0)
    if not np.isfinite(means).all():
        raise ValueError("at least one target column has no finite values")
    rows, columns = np.where(~np.isfinite(work))
    work[rows, columns] = means[columns]
    return work


def two_way_center(matrix: np.ndarray) -> np.ndarray:
    """Remove least-squares row and column main effects from a complete matrix."""
    work = np.asarray(matrix, dtype=np.float64)
    if not np.isfinite(work).all():
        raise ValueError("two-way centering requires a complete matrix")
    return (
        work
        - work.mean(axis=0, keepdims=True)
        - work.mean(axis=1, keepdims=True)
        + work.mean()
    )


def correlation_pr(matrix: np.ndarray) -> dict[str, Any]:
    """Return PR and companion summaries for the column-correlation spectrum."""
    work = np.asarray(matrix, dtype=np.float64)
    if not np.isfinite(work).all():
        raise ValueError("correlation PR requires a complete matrix")
    standard_deviation = work.std(axis=0, ddof=1)
    keep = standard_deviation > 1e-12
    if int(keep.sum()) < 2:
        raise ValueError("fewer than two nonconstant columns remain")
    work = work[:, keep]
    z = (work - work.mean(axis=0)) / work.std(axis=0, ddof=1)
    correlation = (z.T @ z) / (len(z) - 1)
    np.fill_diagonal(correlation, 1.0)
    eigenvalues = np.clip(np.linalg.eigvalsh(correlation)[::-1], 0.0, None)
    weights = eigenvalues / eigenvalues.sum()
    upper = correlation[np.triu_indices_from(correlation, k=1)]
    return {
        "n_ligands": int(len(work)),
        "n_targets": int(work.shape[1]),
        "participation_ratio": float(
            eigenvalues.sum() ** 2 / np.square(eigenvalues).sum()
        ),
        "pc1_fraction": float(weights[0]),
        "mean_squared_offdiagonal_correlation": float(np.mean(np.square(upper))),
        "mean_absolute_offdiagonal_correlation": float(np.mean(np.abs(upper))),
        "eigenvalues": [float(value) for value in eigenvalues],
    }


def surface_metrics(matrix_with_missing: np.ndarray) -> dict[str, Any]:
    """Use target-mean imputation, then compare raw and centred PR."""
    complete = mean_impute_columns(matrix_with_missing)
    raw = correlation_pr(complete)
    residual = correlation_pr(two_way_center(complete))
    raw_pr = raw["participation_ratio"]
    residual_pr = residual["participation_ratio"]
    return {
        "raw_pr": raw_pr,
        "residual_pr": residual_pr,
        "pr_increase": residual_pr - raw_pr,
        "residual_to_raw_ratio": residual_pr / raw_pr,
        "raw_pc1_fraction": raw["pc1_fraction"],
        "residual_pc1_fraction": residual["pc1_fraction"],
        "raw_mean_squared_correlation": raw[
            "mean_squared_offdiagonal_correlation"
        ],
        "residual_mean_squared_correlation": residual[
            "mean_squared_offdiagonal_correlation"
        ],
        "n_ligands": raw["n_ligands"],
        "n_targets": raw["n_targets"],
    }


def apply_scalar_support(
    numerator: np.ndarray,
    scalar_scores: np.ndarray,
    *,
    clip_upper_zero: bool,
) -> np.ndarray:
    """Apply the frozen scalar matrix's missing and zero-censoring support.

    A scalar-missing cell remains missing even if an upstream residue value is
    available.  A scalar zero-censored cell is fixed to zero.  This prevents the
    residue artifact from silently changing the analysis support.
    """
    numerator = np.asarray(numerator, dtype=np.float64)
    scalar_scores = np.asarray(scalar_scores, dtype=np.float64)
    if numerator.shape != scalar_scores.shape:
        raise ValueError("numerator and scalar score matrices must have equal shape")
    matched = numerator.copy()
    observed = np.isfinite(scalar_scores)
    matched[~observed] = np.nan
    if clip_upper_zero:
        matched = np.minimum(matched, 0.0)
    matched[observed & (scalar_scores == 0.0)] = 0.0
    return matched


def infer_ligand_factor(
    numerator: np.ndarray,
    scalar_scores: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Infer a robust row-specific E/S factor from observed negative cells."""
    valid = (
        np.isfinite(numerator)
        & np.isfinite(scalar_scores)
        & (scalar_scores < 0.0)
    )
    ratio = np.full_like(numerator, np.nan, dtype=np.float64)
    ratio[valid] = numerator[valid] / scalar_scores[valid]
    factor = np.nanmedian(ratio, axis=1)
    if not np.isfinite(factor).all() or np.any(factor <= 0.0):
        raise ValueError("could not infer a positive finite factor for every ligand")
    return factor, ratio


def validate_target_contract(residue_dir: Path) -> list[Path]:
    """Resolve exactly 43 files and reject a silent V1A/9uwl substitution."""
    if "V1A" in NUMERATOR_TARGETS or "9uwl" in NUMERATOR_TARGETS:
        raise AssertionError("invalid exact-overlap target contract")
    paths = [residue_dir / f"{target}_residue_matrix.csv" for target in NUMERATOR_TARGETS]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "missing exact target residue matrices: " + ", ".join(missing)
        )
    return paths


def load_inputs(
    scalar_table: Path,
    residue_dir: Path,
    *,
    compute_hashes: bool,
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    """Load scalar scores and exact-name residue sums, validating ligand IDs."""
    residue_paths = validate_target_contract(residue_dir)
    usecols = [
        "ligand_id",
        "Cleaned SMILES",
        "rotatable_bonds",
        *NUMERATOR_TARGETS,
    ]
    scalar = pd.read_csv(scalar_table, usecols=usecols)
    if scalar["ligand_id"].duplicated().any():
        raise ValueError("scalar ligand_id values are not unique")
    expected_ids = ("ligand_" + scalar["ligand_id"].astype(str)).to_numpy()

    numerator = np.empty((len(scalar), len(NUMERATOR_TARGETS)), dtype=np.float64)
    manifest_rows: list[dict[str, Any]] = [{
        "kind": "scalar_table",
        "target": "__all__",
        "path": portable_path(scalar_table),
        "size_bytes": scalar_table.stat().st_size,
        "sha256": sha256_file(scalar_table) if compute_hashes else "not_computed",
        "n_rows": int(len(scalar)),
        "n_numeric_columns": int(len(NUMERATOR_TARGETS)),
    }]

    for column, (target, path) in enumerate(zip(NUMERATOR_TARGETS, residue_paths)):
        frame = pd.read_csv(path)
        if "ligand_id" not in frame.columns:
            raise ValueError(f"{path} has no ligand_id column")
        observed_ids = frame.pop("ligand_id").astype(str).to_numpy()
        if not np.array_equal(observed_ids, expected_ids):
            raise ValueError(f"ligand IDs/order do not match in {path}")
        values = frame.apply(pd.to_numeric, errors="coerce").to_numpy(np.float64)
        if not np.isfinite(values).all():
            raise ValueError(f"non-finite residue energies found in {path}")
        numerator[:, column] = values.sum(axis=1)
        manifest_rows.append({
            "kind": "residue_matrix",
            "target": target,
            "path": portable_path(path),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path) if compute_hashes else "not_computed",
            "n_rows": int(len(frame)),
            "n_numeric_columns": int(frame.shape[1]),
        })

    return scalar, numerator, pd.DataFrame(manifest_rows)


def target_reconstruction(
    numerator: np.ndarray,
    scalar_scores: np.ndarray,
    factor: np.ndarray,
) -> pd.DataFrame:
    reconstructed = numerator / factor[:, None]
    rows: list[dict[str, Any]] = []
    for column, target in enumerate(NUMERATOR_TARGETS):
        valid = np.isfinite(scalar_scores[:, column]) & (
            scalar_scores[:, column] < 0.0
        )
        observed = scalar_scores[valid, column]
        predicted = reconstructed[valid, column]
        rows.append({
            "target": target,
            "n_observed_negative_cells": int(valid.sum()),
            "pearson_reconstructed_vs_scalar": float(
                stats.pearsonr(predicted, observed).statistic
            ),
            "spearman_reconstructed_vs_scalar": float(
                stats.spearmanr(predicted, observed).statistic
            ),
            "mean_absolute_error_kcal_mol": float(np.mean(np.abs(predicted - observed))),
            "root_mean_squared_error_kcal_mol": float(
                np.sqrt(np.mean(np.square(predicted - observed)))
            ),
        })
    return pd.DataFrame(rows)


def leave_one_target_out_reconstruction(
    numerator: np.ndarray,
    scalar_scores: np.ndarray,
) -> pd.DataFrame:
    """Reconstruct each target using a row factor estimated from other targets."""
    valid = (
        np.isfinite(numerator)
        & np.isfinite(scalar_scores)
        & (scalar_scores < 0.0)
    )
    ratios = np.full_like(numerator, np.nan, dtype=np.float64)
    ratios[valid] = numerator[valid] / scalar_scores[valid]
    rows: list[dict[str, Any]] = []
    for column, target in enumerate(NUMERATOR_TARGETS):
        training_ratios = ratios.copy()
        training_ratios[:, column] = np.nan
        held_out_factor = np.nanmedian(training_ratios, axis=1)
        evaluate = (
            valid[:, column]
            & np.isfinite(held_out_factor)
            & (held_out_factor > 0.0)
        )
        observed = scalar_scores[evaluate, column]
        predicted = numerator[evaluate, column] / held_out_factor[evaluate]
        rows.append(
            {
                "target": target,
                "n_held_out_observed_negative_cells": int(evaluate.sum()),
                "pearson_leave_one_target_out": float(
                    stats.pearsonr(predicted, observed).statistic
                ),
                "spearman_leave_one_target_out": float(
                    stats.spearmanr(predicted, observed).statistic
                ),
                "mean_absolute_error_leave_one_target_out_kcal_mol": float(
                    np.mean(np.abs(predicted - observed))
                ),
                "root_mean_squared_error_leave_one_target_out_kcal_mol": float(
                    np.sqrt(np.mean(np.square(predicted - observed)))
                ),
            }
        )
    return pd.DataFrame(rows)


def additive_factor_toy(
    matched_numerator: np.ndarray,
    factor: np.ndarray,
) -> dict[str, Any]:
    """Show that row scaling of a purely additive numerator creates one mode."""
    complete = mean_impute_columns(matched_numerator)
    grand = complete.mean()
    row_effect = complete.mean(axis=1) - grand
    target_effect = complete.mean(axis=0) - grand
    additive = grand + row_effect[:, None] + target_effect[None, :]
    centered_before = two_way_center(additive)
    centered_after = two_way_center(additive / factor[:, None])
    singular_values = np.linalg.svd(centered_after, compute_uv=False)
    tolerance = singular_values[0] * 1e-10
    numerical_rank = int(np.sum(singular_values > tolerance))
    return {
        "maximum_absolute_centered_value_before_factor": float(
            np.max(np.abs(centered_before))
        ),
        "centered_numerical_rank_after_factor": numerical_rank,
        "centered_correlation_pr_after_factor": correlation_pr(centered_after)[
            "participation_ratio"
        ],
        "leading_singular_values_after_factor": [
            float(value) for value in singular_values[:5]
        ],
        "rank_bound": (
            "For E=a1^T+1b^T+Gamma and S=DE, H_N S H_P="
            "H_N D Gamma H_P+(H_N D1)(b^T H_P). Row scaling therefore "
            "adds at most one centered algebraic mode to a k-mode Gamma."
        ),
    }


def summarise_distribution(values: Iterable[float]) -> dict[str, Any]:
    array = np.asarray(list(values), dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "standard_deviation": float(array.std(ddof=1)),
        "interval_95": [
            float(np.quantile(array, 0.025)),
            float(np.quantile(array, 0.975)),
        ],
        "minimum": float(array.min()),
        "maximum": float(array.max()),
    }


def permute_within_groups(
    values: np.ndarray, groups: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Permute a ligand factor while preserving an exact grouping variable."""
    values = np.asarray(values, dtype=np.float64)
    groups = np.asarray(groups)
    if values.ndim != 1 or groups.shape != values.shape:
        raise ValueError("values and groups must be aligned one-dimensional arrays")
    permuted = values.copy()
    for group in np.unique(groups):
        indices = np.flatnonzero(groups == group)
        permuted[indices] = rng.permutation(values[indices])
    return permuted


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
    return value


def run_analysis(
    scalar_table: Path,
    residue_dir: Path,
    output_dir: Path,
    *,
    shuffle_repeats: int,
    seed: int,
    compute_hashes: bool,
) -> dict[str, Any]:
    if shuffle_repeats < 1:
        raise ValueError("shuffle_repeats must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)
    scalar, numerator, manifest = load_inputs(
        scalar_table, residue_dir, compute_hashes=compute_hashes
    )
    scalar_scores = scalar.loc[:, NUMERATOR_TARGETS].apply(
        pd.to_numeric, errors="coerce"
    ).to_numpy(np.float64)
    scalar_primary = np.minimum(scalar_scores, 0.0)
    factor, ratios = infer_ligand_factor(numerator, scalar_scores)

    matched_numerator_unclipped = apply_scalar_support(
        numerator, scalar_scores, clip_upper_zero=False
    )
    matched_numerator = apply_scalar_support(
        numerator, scalar_scores, clip_upper_zero=True
    )
    inferred_integer_torsions = np.rint((factor - 1.0) / ROT_WEIGHT)
    integer_factor = 1.0 + ROT_WEIGHT * inferred_integer_torsions
    rotatable_bonds = pd.to_numeric(
        scalar["rotatable_bonds"], errors="coerce"
    ).to_numpy(np.float64)
    if not np.isfinite(rotatable_bonds).all() or np.any(rotatable_bonds < 0):
        raise ValueError("rotatable_bonds must be finite and non-negative")
    descriptor_factor = 1.0 + ROT_WEIGHT * rotatable_bonds
    heavy_atoms_list: list[int] = []
    for index, smiles in enumerate(scalar["Cleaned SMILES"].astype(str)):
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            raise ValueError(f"invalid Cleaned SMILES at scalar row {index}")
        heavy_atoms_list.append(molecule.GetNumHeavyAtoms())
    heavy_atoms = np.asarray(heavy_atoms_list, dtype=np.float64)
    if np.any(heavy_atoms <= 0):
        raise ValueError("RDKit heavy-atom counts must be positive")

    surfaces = {
        "normalized_scalar_score": scalar_primary,
        "unnormalized_numerator_matched_mask": matched_numerator,
        "unnormalized_numerator_matched_mask_no_clip": matched_numerator_unclipped,
        "numerator_divided_by_inferred_q": matched_numerator / factor[:, None],
        "numerator_divided_by_integer_q": matched_numerator / integer_factor[:, None],
        "numerator_divided_by_rotatable_bond_q": (
            matched_numerator / descriptor_factor[:, None]
        ),
    }
    surface_rows = []
    for name, matrix in surfaces.items():
        surface_rows.append({
            "support": "43_target_mean_imputed_primary",
            "surface": name,
            **surface_metrics(matrix),
        })

    complete_rows = np.isfinite(scalar_scores).all(axis=1)
    for name in (
        "normalized_scalar_score",
        "unnormalized_numerator_matched_mask",
        "numerator_divided_by_rotatable_bond_q",
    ):
        surface_rows.append({
            "support": "43_target_complete_case_sensitivity",
            "surface": name,
            **surface_metrics(surfaces[name][complete_rows]),
        })
    surface_frame = pd.DataFrame(surface_rows)

    rng = np.random.default_rng(seed)
    shuffle_rows: list[dict[str, Any]] = []
    for repetition in range(shuffle_repeats):
        for scheme, permuted_factor in (
            ("global", rng.permutation(factor)),
            (
                "within_exact_heavy_atom_count",
                permute_within_groups(factor, heavy_atoms, rng),
            ),
        ):
            metrics = surface_metrics(matched_numerator / permuted_factor[:, None])
            shuffle_rows.append(
                {"scheme": scheme, "repetition": repetition, **metrics}
            )
    shuffle_frame = pd.DataFrame(shuffle_rows)

    reconstruction = target_reconstruction(numerator, scalar_scores, factor)
    held_out_reconstruction = leave_one_target_out_reconstruction(
        numerator, scalar_scores
    )
    ratio_counts = np.sum(np.isfinite(ratios), axis=1)
    ratio_absolute_deviation = np.abs(ratios - factor[:, None])
    q_frame = pd.DataFrame({
        "ligand_id": scalar["ligand_id"],
        "q_inferred_median_ratio": factor,
        "n_targets_used_for_q": ratio_counts,
        "median_absolute_target_ratio_deviation": np.nanmedian(
            ratio_absolute_deviation, axis=1
        ),
        "inferred_torsions_continuous": (factor - 1.0) / ROT_WEIGHT,
        "inferred_torsions_nearest_integer": inferred_integer_torsions.astype(int),
        "rotatable_bonds_frozen_descriptor": rotatable_bonds,
        "heavy_atoms_frozen_descriptor": heavy_atoms,
        "q_from_rotatable_bonds": descriptor_factor,
    })

    score_observed = np.isfinite(scalar_scores)
    sign_discordant = score_observed & (scalar_scores < 0.0) & (numerator > 0.0)
    inferred_observed = surface_frame.loc[
        (surface_frame.support == "43_target_mean_imputed_primary")
        & (surface_frame.surface == "numerator_divided_by_inferred_q")
    ].iloc[0]
    descriptor_observed = surface_frame.loc[
        (surface_frame.support == "43_target_mean_imputed_primary")
        & (surface_frame.surface == "numerator_divided_by_rotatable_bond_q")
    ].iloc[0]
    numerator_observed = surface_frame.loc[
        (surface_frame.support == "43_target_mean_imputed_primary")
        & (surface_frame.surface == "unnormalized_numerator_matched_mask")
    ].iloc[0]
    score_observed_metrics = surface_frame.loc[
        (surface_frame.support == "43_target_mean_imputed_primary")
        & (surface_frame.surface == "normalized_scalar_score")
    ].iloc[0]

    q_pearson = stats.pearsonr(factor, rotatable_bonds)
    q_spearman = stats.spearmanr(factor, rotatable_bonds)
    summary: dict[str, Any] = {
        "analysis": "Docking-44 Vina intermolecular numerator and torsion audit",
        "support": {
            "primary_scalar_targets": len(DOCK44),
            "exact_numerator_overlap_targets": len(NUMERATOR_TARGETS),
            "included_targets": list(NUMERATOR_TARGETS),
            "excluded_target": "V1A",
            "excluded_target_reason": (
                "No exact V1A_residue_matrix.csv exists; 9uwl is not substituted."
            ),
            "n_ligands": int(len(scalar)),
            "missing_scalar_cells": int((~score_observed).sum()),
            "rows_with_any_missing_scalar_score": int((~score_observed).any(axis=1).sum()),
            "complete_case_rows": int(complete_rows.sum()),
            "zero_censored_scalar_cells": int((scalar_scores == 0.0).sum()),
            "positive_numerator_cells_before_mask_or_clip": int((numerator > 0.0).sum()),
            "positive_numerator_but_negative_scalar_cells": int(sign_discordant.sum()),
            "support_caveat": SUPPORT_CAVEAT,
        },
        "headline": {
            "normalized_score_raw_pr": float(score_observed_metrics.raw_pr),
            "normalized_score_residual_pr": float(score_observed_metrics.residual_pr),
            "numerator_raw_pr": float(numerator_observed.raw_pr),
            "numerator_residual_pr": float(numerator_observed.residual_pr),
            "inferred_q_score_raw_pr": float(inferred_observed.raw_pr),
            "inferred_q_score_residual_pr": float(inferred_observed.residual_pr),
            "rotatable_bond_q_score_raw_pr": float(descriptor_observed.raw_pr),
            "rotatable_bond_q_score_residual_pr": float(
                descriptor_observed.residual_pr
            ),
            "residual_pr_amplification_by_inferred_q": float(
                inferred_observed.residual_pr - numerator_observed.residual_pr
            ),
            "pr_increase_amplification_by_inferred_q": float(
                inferred_observed.pr_increase - numerator_observed.pr_increase
            ),
            "descriptive_fraction_of_inferred_q_pr_increase_due_to_amplification": float(
                (
                    inferred_observed.pr_increase
                    - numerator_observed.pr_increase
                )
                / inferred_observed.pr_increase
            ),
            "interpretation": (
                "Most of the centred effective dimension is already present in the "
                "matched intermolecular numerator. Ligand-specific torsional "
                "normalisation amplifies the residual PR but is not its sole origin. "
                "The fraction is descriptive rather than an additive causal variance "
                "decomposition."
            ),
        },
        "factor_diagnostics": {
            "definition": "q_i = median_j(E_ij / S_ij) on finite S_ij < 0 cells",
            "vina_rot_weight": ROT_WEIGHT,
            "q_quantiles": {
                str(probability): float(np.quantile(factor, probability))
                for probability in (0.0, 0.01, 0.25, 0.5, 0.75, 0.99, 1.0)
            },
            "q_vs_rotatable_bonds_pearson": float(q_pearson.statistic),
            "q_vs_rotatable_bonds_spearman": float(q_spearman.statistic),
            "fraction_nearest_integer_torsions_equal_frozen_rotatable_bonds": float(
                np.mean(inferred_integer_torsions == rotatable_bonds)
            ),
            "circularity_caveat": (
                "The inferred q uses the scalar score being reconstructed. The "
                "rotatable-bond q surface is the independent descriptor sensitivity."
            ),
        },
        "target_reconstruction": {
            "median_target_pearson": float(
                reconstruction.pearson_reconstructed_vs_scalar.median()
            ),
            "minimum_target_pearson": float(
                reconstruction.pearson_reconstructed_vs_scalar.min()
            ),
            "maximum_target_pearson": float(
                reconstruction.pearson_reconstructed_vs_scalar.max()
            ),
            "median_target_mae_kcal_mol": float(
                reconstruction.mean_absolute_error_kcal_mol.median()
            ),
            "leave_one_target_out_median_pearson": float(
                held_out_reconstruction.pearson_leave_one_target_out.median()
            ),
            "leave_one_target_out_minimum_pearson": float(
                held_out_reconstruction.pearson_leave_one_target_out.min()
            ),
            "leave_one_target_out_median_mae_kcal_mol": float(
                held_out_reconstruction[
                    "mean_absolute_error_leave_one_target_out_kcal_mol"
                ].median()
            ),
            "leave_one_target_out_definition": (
                "For target j, each ligand q_i is the median E_ik/S_ik over k != j; "
                "the held-out target cell is never used to estimate its factor"
            ),
        },
        "q_shuffle_sensitivity": {
            "repeats": shuffle_repeats,
            "seed": seed,
            "global": {
                "raw_pr": summarise_distribution(
                    shuffle_frame.loc[shuffle_frame.scheme.eq("global"), "raw_pr"]
                ),
                "residual_pr": summarise_distribution(
                    shuffle_frame.loc[
                        shuffle_frame.scheme.eq("global"), "residual_pr"
                    ]
                ),
            },
            "within_exact_heavy_atom_count": {
                "raw_pr": summarise_distribution(
                    shuffle_frame.loc[
                        shuffle_frame.scheme.eq("within_exact_heavy_atom_count"),
                        "raw_pr",
                    ]
                ),
                "residual_pr": summarise_distribution(
                    shuffle_frame.loc[
                        shuffle_frame.scheme.eq("within_exact_heavy_atom_count"),
                        "residual_pr",
                    ]
                ),
            },
            "observed_inferred_q_residual_pr": float(inferred_observed.residual_pr),
            "observed_minus_shuffle_median_residual_pr": float(
                inferred_observed.residual_pr
                - shuffle_frame.loc[
                    shuffle_frame.scheme.eq("global"), "residual_pr"
                ].median()
            ),
            "observed_minus_exact_heavy_atom_shuffle_median_residual_pr": float(
                inferred_observed.residual_pr
                - shuffle_frame.loc[
                    shuffle_frame.scheme.eq("within_exact_heavy_atom_count"),
                    "residual_pr",
                ].median()
            ),
            "interpretation": (
                "These are descriptive ligand-factor permutation sensitivities, not "
                "calibrated biological nulls or inferential p-values. The exact-heavy-"
                "atom scheme preserves molecular-size composition while disrupting "
                "factor alignment within each size stratum. If only the global shuffle "
                "lowers residual PR, the apparent amplification should be attributed "
                "primarily to factor--molecular-size association rather than a finer "
                "within-size alignment with residual target patterns."
            ),
        },
        "additive_factor_toy": additive_factor_toy(matched_numerator, factor),
        "equations": {
            "score_reconstruction": "S_ij approximately E_ij / q_i",
            "vina_factor": "q_i approximately 1 + 0.05846 N_rot,i",
            "centering_identity": (
                "H_N D E H_P = H_N D Gamma H_P + (H_N D 1)(b^T H_P)"
            ),
        },
        "outputs": {
            "surface_metrics": "surface_metrics.csv",
            "q_factors": "q_factors.csv",
            "q_shuffle_null": "q_shuffle_null.csv",
            "target_reconstruction": "target_reconstruction.csv",
            "leave_one_target_out_reconstruction": (
                "target_reconstruction_leave_one_out.csv"
            ),
            "input_manifest": "input_manifest.csv",
        },
    }

    surface_frame.to_csv(output_dir / "surface_metrics.csv", index=False)
    q_frame.to_csv(output_dir / "q_factors.csv", index=False)
    shuffle_frame.to_csv(output_dir / "q_shuffle_null.csv", index=False)
    reconstruction.to_csv(output_dir / "target_reconstruction.csv", index=False)
    held_out_reconstruction.to_csv(
        output_dir / "target_reconstruction_leave_one_out.csv", index=False
    )
    manifest.to_csv(output_dir / "input_manifest.csv", index=False)
    (output_dir / "analysis_summary.json").write_text(
        json.dumps(json_ready(summary), indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scalar-table", type=Path, default=DEFAULT_SCALAR_TABLE)
    parser.add_argument("--residue-dir", type=Path, default=DEFAULT_RESIDUE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--shuffle-repeats", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--skip-input-hashes",
        action="store_true",
        help="skip SHA-256 computation for faster development runs",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_analysis(
        args.scalar_table,
        args.residue_dir,
        args.output_dir,
        shuffle_repeats=args.shuffle_repeats,
        seed=args.seed,
        compute_hashes=not args.skip_input_hashes,
    )
    print(json.dumps(json_ready(summary["headline"]), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
