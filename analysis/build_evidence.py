#!/usr/bin/env python3
"""Deprecated legacy evidence builder; direct execution is disabled.

The current release builder is ``analysis/build_manuscript_evidence.py``.
This module is retained only to keep its historical analysis helpers readable.

The script deliberately keeps three objects separate:

1. the raw ligand-by-target score surface;
2. the two-way-centred ligand-target interaction surface; and
3. external performance metrics such as affinity correlation or selectivity P@5.

It does not consume the live OKL analysis under ``reproduce/``.  That study has a
different estimand and is not submission-ready.
"""

from __future__ import annotations

import hashlib
import json
import os
import warnings
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import numpy as np
import pandas as pd
from scipy import stats
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, Descriptors, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge, LinearRegression
from sklearn.model_selection import KFold

RDLogger.DisableLog("rdApp.*")


PACKAGE = Path(__file__).resolve().parents[1]
LEGACY_ROOT = PACKAGE.parent
FROZEN = PACKAGE / "data" / "frozen"
OUT = PACKAGE / "results"
SEED = 0

DOCK44 = [
    "1m2z", "1pbq", "1xoq", "2rh1", "2vt4", "2ydo", "2z5x", "3b66",
    "3kk6", "3ln1", "3rze", "4djh", "4ey7", "4iar", "4mqs", "4n6h",
    "5cxv", "5i71", "5tvn", "5u09", "5va1", "6cm4", "6kpf", "6kux",
    "6lqa", "6pdj", "6x3x", "6y1z", "7f8y", "7kwe", "7ljd", "7wc9",
    "7xnk", "7ym8", "8e9y", "8ef6", "8fhs", "8pjk", "8st0", "8wty",
    "8xvk", "8yn3", "9eo4", "V1A",
]
MATCHED_TARGETS = ["5va1", "6cm4", "7wc9", "8pjk", "3rze", "7ym8"]


def source_path(relative: str | Path) -> Path:
    """Resolve a release-local frozen input, with a legacy-tree fallback for development."""
    relative = Path(relative)
    candidates = [FROZEN / relative, Path(f"{FROZEN / relative}.gz")]
    if os.environ.get("JCHEMINF_REQUIRE_BUNDLED") != "1":
        candidates.append(LEGACY_ROOT / relative)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Frozen input not found: {relative}")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def two_way_center(x: np.ndarray) -> np.ndarray:
    """Remove the least-squares row and column main effects from a complete matrix."""
    x = np.asarray(x, dtype=np.float64)
    return x - x.mean(axis=0, keepdims=True) - x.mean(axis=1, keepdims=True) + x.mean()


def target_correlation_matrix(x: np.ndarray) -> np.ndarray:
    """Target correlation matrix after finite-value mean imputation and z-scoring."""
    x = np.asarray(x, dtype=np.float64).copy()
    keep = np.nanstd(x, axis=0, ddof=1) > 1e-12
    x = x[:, keep]
    if np.isnan(x).any():
        means = np.nanmean(x, axis=0)
        rows, cols = np.where(np.isnan(x))
        x[rows, cols] = means[cols]
    z = (x - x.mean(axis=0)) / x.std(axis=0, ddof=1)
    corr = (z.T @ z) / (len(z) - 1)
    np.fill_diagonal(corr, 1.0)
    return np.clip(corr, -1.0, 1.0)


def spectrum(x: np.ndarray) -> np.ndarray:
    """Eigenvalues of the target correlation matrix after column standardisation."""
    eig = np.linalg.eigvalsh(target_correlation_matrix(x))[::-1]
    return np.clip(eig, 0.0, None)


def rank_summary(x: np.ndarray) -> dict:
    corr = target_correlation_matrix(x)
    eig = np.clip(np.linalg.eigvalsh(corr)[::-1], 0.0, None)
    report = rank_summary_from_eigenvalues(eig, np.asarray(x).shape)
    upper = corr[np.triu_indices(len(corr), 1)]
    report.update({
        "mean_absolute_offdiagonal_correlation": float(np.mean(np.abs(upper))),
        "median_absolute_offdiagonal_correlation": float(np.median(np.abs(upper))),
        "mean_offdiagonal_correlation": float(np.mean(upper)),
        "offdiagonal_correlation_interval_90": [
            float(np.quantile(upper, 0.05)),
            float(np.quantile(upper, 0.95)),
        ],
        "fraction_positive_offdiagonal_correlations": float(np.mean(upper > 0)),
    })
    return report


def rank_summary_from_eigenvalues(eig: np.ndarray, shape: tuple[int, int]) -> dict:
    weights = eig / eig.sum()
    weights_pos = weights[weights > 0]
    cumulative = np.cumsum(weights)
    p = len(eig)
    mean_squared_offdiag = (
        float((np.square(eig).sum() - p) / (p * (p - 1))) if p > 1 else 0.0
    )
    return {
        "n_ligands": int(shape[0]),
        "n_targets": int(shape[1]),
        "participation_ratio": float(eig.sum() ** 2 / np.square(eig).sum()),
        "entropy_rank": float(np.exp(-np.sum(weights_pos * np.log(weights_pos)))),
        "pc1_fraction": float(weights[0]),
        "pcs_for_80pct": int(np.searchsorted(cumulative, 0.80) + 1),
        "pcs_for_90pct": int(np.searchsorted(cumulative, 0.90) + 1),
        "mean_squared_offdiagonal_correlation": max(mean_squared_offdiag, 0.0),
        "numerical_zero_eigenvalues_at_1e-10": int(np.sum(eig < 1e-10)),
        "eigenvalues": [float(v) for v in eig],
        "top10_eigenvalues": [float(v) for v in eig[:10]],
    }


def covariance_rank_summary(x: np.ndarray) -> dict:
    """Effective-dimension summaries without target-wise variance equalisation."""
    x = np.asarray(x, dtype=np.float64)
    covariance = np.cov(x, rowvar=False, ddof=1)
    eig = np.clip(np.linalg.eigvalsh(covariance)[::-1], 0.0, None)
    weights = eig / eig.sum()
    positive = weights[weights > 0]
    cumulative = np.cumsum(weights)
    target_sd = x.std(axis=0, ddof=1)
    return {
        "n_ligands": int(x.shape[0]),
        "n_targets": int(x.shape[1]),
        "participation_ratio": float(eig.sum() ** 2 / np.square(eig).sum()),
        "entropy_rank": float(np.exp(-np.sum(positive * np.log(positive)))),
        "pc1_fraction": float(weights[0]),
        "pcs_for_80pct": int(np.searchsorted(cumulative, 0.80) + 1),
        "pcs_for_90pct": int(np.searchsorted(cumulative, 0.90) + 1),
        "trace": float(eig.sum()),
        "target_standard_deviation": {
            "minimum": float(target_sd.min()),
            "median": float(np.median(target_sd)),
            "maximum": float(target_sd.max()),
            "coefficient_of_variation": float(target_sd.std(ddof=1) / target_sd.mean()),
        },
        "eigenvalues": [float(value) for value in eig],
    }


def correlation_covariance_sensitivity(x: np.ndarray) -> dict:
    """Compare equal-target-variance and score-scale spectral estimands."""
    x = np.asarray(x, dtype=np.float64)
    residual = two_way_center(x)
    p = x.shape[1]
    raw_ceiling = min(x.shape[0] - 1, p)
    residual_ceiling = min(x.shape[0] - 1, p - 1)
    correlation = {"raw": rank_summary(x), "residual": rank_summary(residual)}
    covariance = {
        "raw": covariance_rank_summary(x),
        "residual": covariance_rank_summary(residual),
    }
    for estimand in (correlation, covariance):
        estimand["raw"]["algebraic_ceiling"] = int(raw_ceiling)
        estimand["residual"]["algebraic_ceiling"] = int(residual_ceiling)
        estimand["raw"]["dimension_fraction_of_ceiling"] = (
            estimand["raw"]["participation_ratio"] / raw_ceiling
        )
        estimand["residual"]["dimension_fraction_of_ceiling"] = (
            estimand["residual"]["participation_ratio"] / residual_ceiling
        )
    return {
        "correlation": correlation,
        "covariance": covariance,
        "interpretation": (
            "The correlation spectrum is primary because it gives every target equal variance. "
            "The covariance spectrum retains target-specific score variance; its participation "
            "ratio is invariant only to a single global multiplicative rescaling."
        ),
    }


def centering_ladder(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=np.float64)
    interaction = two_way_center(x)
    total_ss = float(np.square(x - x.mean()).sum())
    interaction_ss = float(np.square(interaction).sum())
    return {
        "raw": rank_summary(x),
        "interaction": rank_summary(interaction),
        "interaction_fraction_of_grand_centered_sum_squares": interaction_ss / total_ss,
    }


def pc1_loading_summary(
    matrix: np.ndarray, target_names: list[str], families: list[str]
) -> dict:
    """Target loadings and score correlations for the leading standardized direction."""
    matrix = np.asarray(matrix, dtype=np.float64)
    z = (matrix - matrix.mean(axis=0)) / matrix.std(axis=0, ddof=1)
    eigenvalues, eigenvectors = np.linalg.eigh((z.T @ z) / (len(z) - 1))
    loading = eigenvectors[:, -1]
    if loading.mean() < 0:
        loading = -loading
    scores = z @ loading
    uniform = np.ones(len(loading), dtype=np.float64) / np.sqrt(len(loading))
    uniform_cosine = float(np.dot(loading, uniform))
    return {
        "targets": list(target_names),
        "families": list(families),
        "loadings": [float(value) for value in loading],
        "minimum_loading": float(loading.min()),
        "maximum_loading": float(loading.max()),
        "n_positive_loadings": int(np.sum(loading > 0)),
        "n_negative_loadings": int(np.sum(loading < 0)),
        "cosine_with_uniform_target_vector": uniform_cosine,
        "squared_cosine_with_uniform_target_vector": uniform_cosine ** 2,
        "angle_from_uniform_target_vector_degrees": float(
            np.degrees(np.arccos(np.clip(uniform_cosine, -1.0, 1.0)))
        ),
        "loading_coefficient_of_variation": float(
            loading.std(ddof=1) / loading.mean()
        ),
        "pc1_eigenvalue": float(eigenvalues[-1]),
        "correlation_with_raw_per_ligand_mean": float(
            stats.pearsonr(scores, matrix.mean(axis=1)).statistic
        ),
        "correlation_with_column_standardized_per_ligand_mean": float(
            stats.pearsonr(scores, z.mean(axis=1)).statistic
        ),
        "orientation": "sign chosen so the mean target loading is positive",
    }


def molecular_descriptor_frame(smiles: pd.Series) -> pd.DataFrame:
    """Compact, interpretable RDKit descriptor panel for support comparisons."""
    rows: list[dict[str, float]] = []
    for value in smiles.astype(str):
        molecule = Chem.MolFromSmiles(value)
        if molecule is None:
            rows.append({
                "molecular_weight": np.nan,
                "heavy_atoms": np.nan,
                "clogp": np.nan,
                "tpsa": np.nan,
                "rotatable_bonds": np.nan,
                "ring_count": np.nan,
            })
            continue
        rows.append({
            "molecular_weight": float(Descriptors.MolWt(molecule)),
            "heavy_atoms": float(molecule.GetNumHeavyAtoms()),
            "clogp": float(Crippen.MolLogP(molecule)),
            "tpsa": float(Descriptors.TPSA(molecule)),
            "rotatable_bonds": float(Descriptors.NumRotatableBonds(molecule)),
            "ring_count": float(rdMolDescriptors.CalcNumRings(molecule)),
        })
    return pd.DataFrame(rows, index=smiles.index)


def target_physicochemical_slopes(
    matrix: np.ndarray,
    target_names: list[str],
    families: list[str],
    smiles: pd.Series,
    sample_size: int,
    seed: int,
    imputed_cells_by_target: np.ndarray | pd.Series | None = None,
) -> dict:
    """Quantify common and target-specific molecular-size/flexibility slopes."""
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape[1] != len(target_names) or len(target_names) != len(families):
        raise ValueError("target metadata do not match the score matrix")
    if len(smiles) != len(matrix):
        raise ValueError("SMILES do not match the score matrix")
    if imputed_cells_by_target is None:
        imputed_cells = np.zeros(len(target_names), dtype=np.int64)
    else:
        imputed_cells = np.asarray(imputed_cells_by_target, dtype=np.int64)
        if imputed_cells.shape != (len(target_names),):
            raise ValueError("imputed-cell counts do not match target metadata")
        if np.any(imputed_cells < 0) or np.any(imputed_cells > len(matrix)):
            raise ValueError("invalid target-level imputed-cell count")

    rng = np.random.default_rng(seed)
    if len(matrix) > sample_size:
        support = np.sort(rng.choice(len(matrix), sample_size, replace=False))
    else:
        support = np.arange(len(matrix))
    x = matrix[support]
    descriptors = molecular_descriptor_frame(
        pd.Series(smiles).iloc[support].reset_index(drop=True)
    )[["molecular_weight", "heavy_atoms", "rotatable_bonds"]]
    if descriptors.isna().any().any():
        raise ValueError("physicochemical slope support contains invalid molecules")

    residual = two_way_center(x)
    residual_z = (residual - residual.mean(axis=0)) / residual.std(axis=0, ddof=1)
    eigenvalues, eigenvectors = np.linalg.eigh(
        (residual_z.T @ residual_z) / (len(residual_z) - 1)
    )
    residual_pc1 = eigenvectors[:, -1]

    descriptor_scales = {
        "molecular_weight": 100.0,
        "heavy_atoms": 1.0,
        "rotatable_bonds": 1.0,
    }
    descriptor_units = {
        "molecular_weight": "kcal/mol per 100 Da",
        "heavy_atoms": "kcal/mol per heavy atom",
        "rotatable_bonds": "kcal/mol per RDKit rotatable bond",
    }
    records: dict[str, dict] = {}
    per_target: dict[str, dict[str, float]] = {target: {} for target in target_names}
    molecular_weight_vector: np.ndarray | None = None

    for descriptor_name in descriptors.columns:
        values = descriptors[descriptor_name].to_numpy(dtype=np.float64)
        scale = descriptor_scales[descriptor_name]
        scaled_values = values / scale
        design = np.column_stack([np.ones(len(values)), scaled_values])
        raw_slopes = np.asarray([
            np.linalg.lstsq(design, x[:, column], rcond=None)[0][1]
            for column in range(x.shape[1])
        ])
        residual_direct = np.asarray([
            np.linalg.lstsq(design, residual[:, column], rcond=None)[0][1]
            for column in range(x.shape[1])
        ])
        residual_from_identity = raw_slopes - raw_slopes.mean()

        standardized_values = (values - values.mean()) / values.std(ddof=1)
        residual_correlations = (
            residual_z.T @ standardized_values
        ) / (len(standardized_values) - 1)
        norm = np.linalg.norm(residual_correlations)
        alignment = float(abs(np.dot(residual_pc1, residual_correlations)) / norm)
        pc1_score_correlation = float(
            abs(np.dot(residual_pc1, residual_correlations)) / np.sqrt(eigenvalues[-1])
        )
        if descriptor_name == "molecular_weight":
            molecular_weight_vector = residual_correlations

        records[descriptor_name] = {
            "target_slope_unit": descriptor_units[descriptor_name],
            "target_descriptor_correlation_unit": "dimensionless Pearson correlation",
            "raw_target_slope": describe_distribution(raw_slopes),
            "residual_target_slope": describe_distribution(residual_direct),
            "residual_target_descriptor_correlation": describe_distribution(
                residual_correlations
            ),
            "cosine_with_residual_pc1_loading": alignment,
            "absolute_correlation_with_residual_pc1_scores": pc1_score_correlation,
            "maximum_absolute_error_in_two_way_slope_identity": float(
                np.max(np.abs(residual_direct - residual_from_identity))
            ),
        }
        for index, target in enumerate(target_names):
            per_target[target][f"{descriptor_name}_raw_slope"] = float(raw_slopes[index])
            per_target[target][f"{descriptor_name}_residual_slope"] = float(
                residual_direct[index]
            )
            per_target[target][f"{descriptor_name}_residual_correlation"] = float(
                residual_correlations[index]
            )

    if molecular_weight_vector is None:
        raise AssertionError("molecular-weight slope vector was not computed")
    if np.dot(residual_pc1, molecular_weight_vector) < 0:
        residual_pc1 = -residual_pc1

    conditional_design = np.column_stack([
        np.ones(len(descriptors)),
        descriptors["molecular_weight"].to_numpy(dtype=np.float64) / 100.0,
        descriptors["rotatable_bonds"].to_numpy(dtype=np.float64),
    ])
    conditional_coefficients = np.asarray([
        np.linalg.lstsq(conditional_design, x[:, column], rcond=None)[0]
        for column in range(x.shape[1])
    ])
    for index, target in enumerate(target_names):
        per_target[target]["imputed_cells"] = int(imputed_cells[index])
        per_target[target]["imputed_fraction"] = float(
            imputed_cells[index] / len(matrix)
        )
        per_target[target]["conditional_mw_slope_per_100_da"] = float(
            conditional_coefficients[index, 1]
        )
        per_target[target]["conditional_rotatable_bond_slope"] = float(
            conditional_coefficients[index, 2]
        )
        per_target[target]["residual_pc1_loading"] = float(residual_pc1[index])

    return {
        "input_n": int(len(matrix)),
        "support_n": int(len(support)),
        "support_rule": (
            "all ligands when N <= sample_size; otherwise a sorted simple random sample "
            f"without replacement using seed {seed}"
        ),
        "imputed_cells_by_target": {
            target: int(imputed_cells[index])
            for index, target in enumerate(target_names)
        },
        "target_names": list(target_names),
        "families": list(families),
        "descriptor_slopes": records,
        "conditional_model": {
            "formula": "score ~ 1 + molecular_weight/100 + RDKit_NumRotatableBonds",
            "molecular_weight_slope": describe_distribution(
                conditional_coefficients[:, 1]
            ),
            "rotatable_bond_slope": describe_distribution(
                conditional_coefficients[:, 2]
            ),
            "targets_with_positive_rotatable_bond_slope": int(
                np.sum(conditional_coefficients[:, 2] > 0)
            ),
            "interpretation_boundary": (
                "RDKit rotatable-bond count is a proxy, not Vina's internal num_tors. "
                "The result is consistent with a flexibility adjustment but is not a "
                "decomposition of the Vina score."
            ),
        },
        "per_target": per_target,
    }


def residual_structure_characterization(
    matrix: np.ndarray,
    target_names: list[str],
    families: list[str],
    smiles: pd.Series,
    sample_size: int = 15000,
    family_permutations: int = 5000,
    seed: int = SEED,
) -> dict:
    """Describe residual target correlations, family grouping and ligand PC chemistry."""
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape[1] != len(target_names) or len(target_names) != len(families):
        raise ValueError("target metadata does not match residual matrix")
    if len(smiles) != len(matrix):
        raise ValueError("SMILES do not match residual matrix rows")
    rng = np.random.default_rng(seed)
    if len(matrix) > sample_size:
        support = np.sort(rng.choice(len(matrix), sample_size, replace=False))
    else:
        support = np.arange(len(matrix))
    work = matrix[support]
    residual = two_way_center(work)
    raw_corr = target_correlation_matrix(work)
    residual_corr = target_correlation_matrix(residual)

    def correlation_record(corr: np.ndarray) -> dict:
        values = corr[np.triu_indices(len(corr), 1)]
        return {
            "mean": float(values.mean()),
            "mean_absolute": float(np.abs(values).mean()),
            "mean_squared": float(np.square(values).mean()),
            "median": float(np.median(values)),
            "median_absolute": float(np.median(np.abs(values))),
            "interval_90": [
                float(np.quantile(values, 0.05)),
                float(np.quantile(values, 0.95)),
            ],
            "fraction_positive": float(np.mean(values > 0)),
            "values": [float(value) for value in values],
        }

    distance = np.clip(1.0 - residual_corr, 0.0, 2.0)
    np.fill_diagonal(distance, 0.0)
    linkage = hierarchy.linkage(squareform(distance, checks=False), method="average")
    order = hierarchy.leaves_list(linkage)

    family_array = np.asarray(families, dtype=object)
    first, second = np.triu_indices(len(family_array), 1)
    same_family = family_array[first] == family_array[second]
    pair_abs = np.abs(residual_corr[first, second])
    observed_family_difference = float(
        pair_abs[same_family].mean() - pair_abs[~same_family].mean()
    )
    permuted_family_differences = np.empty(family_permutations, dtype=float)
    for repetition in range(family_permutations):
        permuted = rng.permutation(family_array)
        same = permuted[first] == permuted[second]
        permuted_family_differences[repetition] = (
            pair_abs[same].mean() - pair_abs[~same].mean()
        )

    z_residual = (
        residual - residual.mean(axis=0)
    ) / residual.std(axis=0, ddof=1)
    eigenvalues, eigenvectors = np.linalg.eigh(residual_corr)
    order_eigen = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order_eigen]
    eigenvectors = eigenvectors[:, order_eigen]
    descriptor_frame = molecular_descriptor_frame(smiles.iloc[support].reset_index(drop=True))
    mode_records: list[dict] = []
    for mode in range(min(3, len(target_names))):
        loadings = eigenvectors[:, mode]
        scores = z_residual @ loadings
        descriptor_correlations = {}
        for descriptor in descriptor_frame:
            rho = stats.spearmanr(scores, descriptor_frame[descriptor], nan_policy="omit")
            descriptor_correlations[descriptor] = {
                "spearman_rho": float(rho.statistic),
                "p_value": float(rho.pvalue),
            }
        top = np.argsort(np.abs(loadings))[::-1][:10]
        mode_records.append({
            "mode": mode + 1,
            "eigenvalue": float(eigenvalues[mode]),
            "variance_fraction": float(eigenvalues[mode] / len(target_names)),
            "largest_absolute_target_loadings": [
                {
                    "target": target_names[index],
                    "family": families[index],
                    "loading": float(loadings[index]),
                }
                for index in top
            ],
            "ligand_descriptor_correlations": descriptor_correlations,
        })

    return {
        "sample_n": int(len(work)),
        "n_targets": int(len(target_names)),
        "target_names": list(target_names),
        "families": list(families),
        "raw_target_correlation": [[float(value) for value in row] for row in raw_corr],
        "residual_target_correlation": [
            [float(value) for value in row] for row in residual_corr
        ],
        "residual_cluster_order": [target_names[index] for index in order],
        "raw_correlation_distribution": correlation_record(raw_corr),
        "residual_correlation_distribution": correlation_record(residual_corr),
        "family_association": {
            "within_family_pairs": int(same_family.sum()),
            "between_family_pairs": int((~same_family).sum()),
            "mean_absolute_within_family": float(pair_abs[same_family].mean()),
            "mean_absolute_between_family": float(pair_abs[~same_family].mean()),
            "difference_within_minus_between": observed_family_difference,
            "label_permutations": int(family_permutations),
            "one_sided_p_for_positive_difference": float(
                (1 + np.sum(permuted_family_differences >= observed_family_difference))
                / (family_permutations + 1)
            ),
            "permutation_interval_95": [
                float(np.quantile(permuted_family_differences, 0.025)),
                float(np.quantile(permuted_family_differences, 0.975)),
            ],
            "scope": "curated broad family labels on the fixed target panel",
        },
        "leading_residual_modes": mode_records,
    }


def pivot_complete_common(long: pd.DataFrame, value: str, common_values: list[str]) -> np.ndarray:
    panels = {
        column: long.pivot(index="ligand_id", columns="target", values=column).dropna(axis=0, how="any")
        for column in common_values
    }
    common = panels[common_values[0]].index
    for column in common_values[1:]:
        common = common.intersection(panels[column].index)
    return panels[value].loc[common].to_numpy(dtype=np.float64)


def subsample_targets(
    x: np.ndarray,
    counts: list[int],
    repeats: int,
    seed: int,
) -> list[dict]:
    rng = np.random.default_rng(seed)
    p = x.shape[1]
    z = (x - x.mean(axis=0)) / x.std(axis=0, ddof=1)
    corr = (z.T @ z) / (len(z) - 1)
    rows = []
    for count in counts:
        if count > p:
            continue
        vals = []
        for _ in range(repeats):
            cols = rng.choice(p, size=count, replace=False)
            eig = np.linalg.eigvalsh(corr[np.ix_(cols, cols)])[::-1]
            eig = np.clip(eig, 0.0, None)
            vals.append(float(eig.sum() ** 2 / np.square(eig).sum()))
        rows.append({
            "n_targets": count,
            "median": float(np.median(vals)),
            "q025": float(np.quantile(vals, 0.025)),
            "q975": float(np.quantile(vals, 0.975)),
        })
    return rows


def _family_quotas(
    families: np.ndarray,
    count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Allocate a target subset across families while retaining broad coverage.

    At least one target is drawn from every family when ``count`` permits. Remaining
    places follow the observed family proportions by a randomized largest-remainder
    allocation. The returned quotas never exceed the targets available in a family.
    """
    labels, sizes = np.unique(families, return_counts=True)
    if count < len(labels):
        raise ValueError("stratified target sampling requires count >= number of families")
    quotas = np.ones(len(labels), dtype=int)
    remaining = count - len(labels)
    capacity = sizes - quotas
    if remaining:
        ideal = remaining * sizes / sizes.sum()
        add = np.minimum(np.floor(ideal).astype(int), capacity)
        quotas += add
        remaining -= int(add.sum())
    while remaining:
        eligible = np.flatnonzero(quotas < sizes)
        if not len(eligible):
            break
        weights = sizes[eligible] / sizes[eligible].sum()
        chosen = int(rng.choice(eligible, p=weights))
        quotas[chosen] += 1
        remaining -= 1
    return quotas


def subsample_target_surfaces(
    x: np.ndarray,
    counts: list[int],
    repeats: int,
    seed: int,
    families: list[str] | None = None,
    sample_size: int | None = 12000,
) -> list[dict]:
    """Subsample target panels and evaluate raw and residual surfaces together.

    A fixed ligand subsample is used when the matrix exceeds ``sample_size`` so that
    variation along the curves reflects target composition rather than a changing
    ligand sample. The full-panel point is always evaluated on all ligand rows.
    """
    x = np.asarray(x, dtype=np.float64)
    rng = np.random.default_rng(seed)
    n, p = x.shape
    if sample_size is not None and n > sample_size:
        fixed_index = rng.choice(n, size=sample_size, replace=False)
        work = x[fixed_index]
    else:
        work = x
    family_array = np.asarray(families, dtype=object) if families is not None else None
    if family_array is not None and len(family_array) != p:
        raise ValueError("one family label is required for every target")
    labels = np.unique(family_array) if family_array is not None else np.asarray([])

    rows: list[dict] = []
    for count in counts:
        if count > p:
            continue
        raw_values: list[float] = []
        interaction_values: list[float] = []
        if count == p:
            selections = [np.arange(p)]
        else:
            selections = []
            for _ in range(repeats):
                if family_array is None:
                    cols = rng.choice(p, size=count, replace=False)
                else:
                    quotas = _family_quotas(family_array, count, rng)
                    pieces = []
                    for label, quota in zip(labels, quotas):
                        candidates = np.flatnonzero(family_array == label)
                        pieces.append(rng.choice(candidates, size=int(quota), replace=False))
                    cols = np.concatenate(pieces)
                selections.append(np.asarray(cols, dtype=int))
        for cols in selections:
            matrix = x[:, cols] if count == p else work[:, cols]
            raw_values.append(rank_summary(matrix)["participation_ratio"])
            interaction_values.append(
                rank_summary(two_way_center(matrix))["participation_ratio"]
            )

        def summarize(values: list[float]) -> dict:
            array = np.asarray(values, dtype=float)
            return {
                "median": float(np.median(array)),
                "q025": float(np.quantile(array, 0.025)),
                "q975": float(np.quantile(array, 0.975)),
            }

        rows.append({
            "n_targets": count,
            "raw": summarize(raw_values),
            "interaction": summarize(interaction_values),
            "n_repeats": len(selections),
            "stratified_by_family": family_array is not None,
            "fixed_ligand_sample_n_for_subsets": int(len(work)),
            "full_panel_uses_all_ligands": count == p,
        })
    return rows


def target_jackknife(
    x: np.ndarray,
    target_names: list[str],
    sample_size: int = 15000,
    seed: int = SEED,
) -> dict:
    """Delete-one-target jackknife for raw and two-way-centered PR dimensions."""
    x = np.asarray(x, dtype=np.float64)
    rng = np.random.default_rng(seed)
    if len(x) > sample_size:
        work = x[rng.choice(len(x), size=sample_size, replace=False)]
    else:
        work = x
    p = x.shape[1]
    if p != len(target_names):
        raise ValueError("target names do not match matrix columns")
    report: dict[str, dict] = {}
    for surface, transform in [
        ("raw", lambda value: value),
        ("interaction", two_way_center),
    ]:
        full_all_ligands = rank_summary(transform(x))["participation_ratio"]
        full = rank_summary(transform(work))["participation_ratio"]
        leave_one_out = {}
        for column, name in enumerate(target_names):
            keep = np.arange(p) != column
            leave_one_out[name] = rank_summary(transform(work[:, keep]))[
                "participation_ratio"
            ]
        values = np.asarray(list(leave_one_out.values()), dtype=float)
        loo_mean = float(values.mean())
        jackknife_estimate = float(p * full - (p - 1) * loo_mean)
        standard_error = float(
            np.sqrt((p - 1) / p * np.square(values - loo_mean).sum())
        )
        report[surface] = {
            "full_panel_all_ligands": full_all_ligands,
            "full_panel_on_jackknife_support": full,
            "jackknife_support_n": int(len(work)),
            "leave_one_target_out": leave_one_out,
            "leave_one_out_min": float(values.min()),
            "leave_one_out_median": float(np.median(values)),
            "leave_one_out_max": float(values.max()),
            "jackknife_bias_corrected_estimate": jackknife_estimate,
            "jackknife_standard_error": standard_error,
            "jackknife_normal_95_interval": [
                jackknife_estimate - 1.96 * standard_error,
                jackknife_estimate + 1.96 * standard_error,
            ],
            "most_influential_target": target_names[
                int(np.argmax(np.abs(values - full)))
            ],
        }
    report["estimand"] = (
        "delete-one-target composition sensitivity; the normal interval is a "
        "jackknife approximation for the observed target population"
    )
    return report


def scaffold_keys(smiles: pd.Series) -> np.ndarray:
    """Bemis--Murcko scaffold keys; acyclic molecules remain singleton clusters."""
    keys: list[str] = []
    for index, value in enumerate(smiles.astype(str)):
        molecule = Chem.MolFromSmiles(value)
        if molecule is None:
            keys.append(f"INVALID:{index}")
            continue
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=molecule)
        keys.append(scaffold if scaffold else f"ACYCLIC:{index}")
    return np.asarray(keys, dtype=object)


def describe_distribution(values: list[float] | np.ndarray) -> dict:
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "standard_deviation": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        "interval_95": [
            float(np.quantile(array, 0.025)),
            float(np.quantile(array, 0.975)),
        ],
        "interval_90": [
            float(np.quantile(array, 0.05)),
            float(np.quantile(array, 0.95)),
        ],
        "minimum": float(array.min()),
        "maximum": float(array.max()),
    }


def surface_pr_triplet(matrix: np.ndarray) -> tuple[float, float, float, float]:
    raw = rank_summary(matrix)["participation_ratio"]
    residual = rank_summary(two_way_center(matrix))["participation_ratio"]
    return raw, residual, residual - raw, residual / raw


def surface_bootstrap(
    matrix: np.ndarray,
    repeats: int,
    seed: int,
    cluster_labels: np.ndarray | None = None,
) -> dict:
    """Paired bootstrap for raw PR, residual PR, and their change."""
    matrix = np.asarray(matrix, dtype=np.float64)
    rng = np.random.default_rng(seed)
    members: list[np.ndarray] | None = None
    if cluster_labels is not None:
        _, inverse = np.unique(np.asarray(cluster_labels), return_inverse=True)
        members = [np.flatnonzero(inverse == value) for value in range(inverse.max() + 1)]
    values = {"raw": [], "residual": [], "difference": [], "ratio": []}
    sample_sizes: list[int] = []
    for _ in range(repeats):
        if members is None:
            index = rng.integers(0, len(matrix), len(matrix))
        else:
            sampled = rng.integers(0, len(members), len(members))
            index = np.concatenate([members[value] for value in sampled])
        sample_sizes.append(int(len(index)))
        raw, residual, difference, ratio = surface_pr_triplet(matrix[index])
        values["raw"].append(raw)
        values["residual"].append(residual)
        values["difference"].append(difference)
        values["ratio"].append(ratio)
    raw, residual, difference, ratio = surface_pr_triplet(matrix)
    return {
        "plugin": {
            "raw": raw,
            "residual": residual,
            "difference_residual_minus_raw": difference,
            "ratio_residual_over_raw": ratio,
        },
        "bootstrap": {key: describe_distribution(record) for key, record in values.items()},
        "repeats": repeats,
        "n_clusters": int(len(members)) if members is not None else None,
        "resampled_n": {
            "minimum": int(np.min(sample_sizes)),
            "median": float(np.median(sample_sizes)),
            "maximum": int(np.max(sample_sizes)),
        },
    }


def dockstring_scaffold_bootstrap(
    x: np.ndarray,
    smiles: pd.Series,
    repeats: int = 100,
    sample_size: int = 15000,
    supports: int = 5,
    seed: int = SEED,
) -> dict:
    """Molecule and scaffold bootstrap on several independent chemical supports."""
    x = np.asarray(x, dtype=np.float64)
    rng = np.random.default_rng(seed)
    if len(x) != len(smiles):
        raise ValueError("DOCKSTRING matrix and SMILES support differ")
    support_records = []
    for support_number in range(supports):
        support_seed = int(rng.integers(0, np.iinfo(np.int32).max))
        support_rng = np.random.default_rng(support_seed)
        support = support_rng.choice(len(x), size=min(sample_size, len(x)), replace=False)
        matrix = x[support]
        keys = scaffold_keys(smiles.iloc[support].reset_index(drop=True))
        molecule = surface_bootstrap(
            matrix, repeats=repeats, seed=support_seed + 1
        )
        scaffold = surface_bootstrap(
            matrix, repeats=repeats, seed=support_seed + 2, cluster_labels=keys
        )
        support_records.append({
            "support_number": support_number + 1,
            "seed": support_seed,
            "n_molecules": int(len(matrix)),
            "n_scaffold_clusters": int(scaffold["n_clusters"]),
            "acyclic_singletons": int(
                np.sum(np.char.startswith(keys.astype(str), "ACYCLIC:"))
            ),
            "point_estimate": molecule["plugin"],
            "molecule_bootstrap": molecule,
            "scaffold_cluster_bootstrap": scaffold,
        })
    aggregate = {}
    for key in ["raw", "residual", "difference_residual_minus_raw", "ratio_residual_over_raw"]:
        aggregate[key] = describe_distribution([
            record["point_estimate"][key] for record in support_records
        ])
    return {
        "supports": support_records,
        "support_point_estimates": aggregate,
        "n_supports": supports,
        "support_n": int(min(sample_size, len(x))),
        "bootstrap_repeats_per_support": repeats,
        "scope": (
            "chemical-support and scaffold-cluster sensitivity across five independently "
            "seeded 15,000-molecule supports; conditional on the 58 observed targets"
        ),
    }


def subsample_ligands(
    x: np.ndarray,
    counts: list[int],
    repeats: int,
    seed: int,
) -> list[dict]:
    rng = np.random.default_rng(seed)
    n = x.shape[0]
    rows = []
    for count in counts:
        if count > n:
            continue
        vals = []
        for _ in range(repeats):
            idx = rng.choice(n, size=count, replace=False)
            vals.append(rank_summary(x[idx])["participation_ratio"])
        rows.append({
            "n_ligands": count,
            "median": float(np.median(vals)),
            "q025": float(np.quantile(vals, 0.025)),
            "q975": float(np.quantile(vals, 0.975)),
        })
    return rows


def paired_surface_bootstrap(
    docking: np.ndarray, experiment: np.ndarray, repeats: int = 2000, seed: int = SEED
) -> dict:
    """Paired ligand bootstrap for raw and interaction PR contrasts."""
    docking = np.asarray(docking, dtype=np.float64)
    experiment = np.asarray(experiment, dtype=np.float64)
    rng = np.random.default_rng(seed)
    records = {"raw": [], "interaction": []}
    for _ in range(repeats):
        index = rng.integers(0, len(docking), len(docking))
        for surface, transform in [
            ("raw", lambda value: value),
            ("interaction", two_way_center),
        ]:
            dock_rank = rank_summary(transform(docking[index]))["participation_ratio"]
            exp_rank = rank_summary(transform(experiment[index]))["participation_ratio"]
            records[surface].append(exp_rank - dock_rank)

    report = {}
    for surface, transform in [
        ("raw", lambda value: value),
        ("interaction", two_way_center),
    ]:
        values = np.asarray(records[surface])
        dock_rank = rank_summary(transform(docking))["participation_ratio"]
        exp_rank = rank_summary(transform(experiment))["participation_ratio"]
        report[surface] = {
            "docking_participation_ratio": dock_rank,
            "experimental_participation_ratio": exp_rank,
            "plugin_difference_experiment_minus_docking": exp_rank - dock_rank,
            "bootstrap_mean_difference": float(values.mean()),
            "bootstrap_median_difference": float(np.median(values)),
            "bootstrap_difference_95_interval": [
                float(np.quantile(values, 0.025)),
                float(np.quantile(values, 0.975)),
            ],
        }
    report["repeats"] = repeats
    report["resampling_unit"] = "ligand row, paired across docking and experiment"
    return report


def parallel_analysis(
    matrix: np.ndarray,
    sample_size: int = 12000,
    repeats: int = 500,
    series: int = 5,
    seed: int = SEED,
) -> dict:
    """Descriptive rank-wise and family-wise permutation envelopes."""
    matrix = np.asarray(matrix, dtype=np.float64)
    rng = np.random.default_rng(seed)
    if len(matrix) > sample_size:
        index = rng.choice(len(matrix), sample_size, replace=False)
        observed_matrix = matrix[index]
    else:
        observed_matrix = matrix
    if repeats % series:
        raise ValueError("parallel-analysis repeats must divide evenly into series")
    observed = {
        "raw": spectrum(observed_matrix),
        "interaction": spectrum(two_way_center(observed_matrix)),
    }
    nulls = {
        "raw": np.empty((repeats, observed_matrix.shape[1]), dtype=float),
        "interaction": np.empty((repeats, observed_matrix.shape[1]), dtype=float),
    }
    for repetition in range(repeats):
        permuted = np.column_stack([
            rng.permutation(observed_matrix[:, column])
            for column in range(observed_matrix.shape[1])
        ])
        nulls["raw"][repetition] = spectrum(permuted)
        nulls["interaction"][repetition] = spectrum(two_way_center(permuted))

    reports = {}
    per_series = repeats // series
    for surface in ["raw", "interaction"]:
        null_95 = np.quantile(nulls[surface], 0.95, axis=0)
        null_mean = nulls[surface].mean(axis=0)
        null_sd = nulls[surface].std(axis=0, ddof=1)
        valid = null_sd > 1e-12
        max_studentized = np.max(
            (nulls[surface][:, valid] - null_mean[valid]) / null_sd[valid], axis=1
        )
        simultaneous_critical = float(np.quantile(max_studentized, 0.95))
        simultaneous_envelope = null_mean + simultaneous_critical * null_sd
        series_counts = []
        for index in range(series):
            block = nulls[surface][index * per_series:(index + 1) * per_series]
            block_95 = np.quantile(block, 0.95, axis=0)
            series_counts.append(int(np.sum(observed[surface] > block_95)))
        reports[surface] = {
            "n_modes_above_rankwise_95pct_null": int(
                np.sum(observed[surface] > null_95)
            ),
            "n_modes_above_simultaneous_95pct_envelope": int(
                np.sum(observed[surface] > simultaneous_envelope)
            ),
            "independent_series_mode_counts": series_counts,
            "per_series_repeats": per_series,
            "observed_eigenvalues": [float(value) for value in observed[surface]],
            "rankwise_null_95pct": [float(value) for value in null_95],
            "simultaneous_studentized_95pct_critical_value": simultaneous_critical,
            "simultaneous_95pct_envelope": [
                float(value) for value in simultaneous_envelope
            ],
        }
    reports["sample_n"] = int(len(observed_matrix))
    reports["repeats"] = repeats
    reports["independent_series"] = series
    reports["null"] = (
        "independently permute every target column, then apply the same raw or "
        "two-way-centered transformation; rank-wise counts are descriptive, while the "
        "studentized maximum-deviation envelope controls the family-wise error rate over ranks"
    )
    return reports


def fixed_ligand_support_indices(
    n_rows: int,
    sample_size: int,
    seed: int,
) -> tuple[np.ndarray, np.random.Generator, dict]:
    """Select and fingerprint one deterministic ligand support for residual nulls.

    The returned generator is the same generator used for sampling.  Continuing from
    that state preserves the historical simulation stream while making the sampled
    support an explicit, machine-checkable part of every null artifact.  The digest is
    computed from sorted little-endian int64 row indices, so it identifies the support
    set independently of row order and platform byte order.
    """

    if n_rows < 1:
        raise ValueError("residual-null input must contain at least one ligand row")
    if sample_size < 1:
        raise ValueError("residual-null sample_size must be positive")
    rng = np.random.default_rng(seed)
    if n_rows > sample_size:
        support = rng.choice(n_rows, sample_size, replace=False)
        method = "uniform_without_replacement"
    else:
        support = np.arange(n_rows, dtype=np.int64)
        method = "all_rows"
    canonical = np.asarray(np.sort(support), dtype="<i8")
    metadata = {
        "method": method,
        "seed": int(seed),
        "source_rows": int(n_rows),
        "sample_rows": int(len(support)),
        "support_index_sha256": hashlib.sha256(canonical.tobytes()).hexdigest(),
        "digest_contract": "sha256(sorted zero-based row indices encoded as little-endian int64)",
    }
    return np.asarray(support, dtype=np.int64), rng, metadata


def additive_main_effect_null(
    matrix: np.ndarray,
    sample_size: int = 12000,
    repeats: int = 500,
    seed: int = SEED,
) -> dict:
    """Fitted additive Gaussian null with target-specific residual variances.

    The fitted grand, ligand, and target effects are retained. Independent Gaussian
    residuals use the observed residual standard deviation of each target. The same
    two-way centering and column standardisation are then applied to every simulation.
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    support, rng, support_selection = fixed_ligand_support_indices(
        len(matrix), sample_size, seed
    )
    work = matrix[support]
    grand = float(work.mean())
    ligand_effect = work.mean(axis=1) - grand
    target_effect = work.mean(axis=0) - grand
    residual = two_way_center(work)
    residual_sd = residual.std(axis=0, ddof=1)
    raw_values: list[float] = []
    residual_values: list[float] = []
    for _ in range(repeats):
        noise = rng.normal(0.0, residual_sd, size=work.shape)
        simulated = (
            grand + ligand_effect[:, None] + target_effect[None, :] + noise
        )
        raw, centered, _, _ = surface_pr_triplet(simulated)
        raw_values.append(raw)
        residual_values.append(centered)
    observed_raw, observed_residual, observed_difference, observed_ratio = surface_pr_triplet(work)
    residual_array = np.asarray(residual_values)
    return {
        "sample_n": int(len(work)),
        "n_targets": int(work.shape[1]),
        "repeats": repeats,
        "support_selection": support_selection,
        "observed": {
            "raw": observed_raw,
            "residual": observed_residual,
            "difference_residual_minus_raw": observed_difference,
            "ratio_residual_over_raw": observed_ratio,
        },
        "null_raw": describe_distribution(raw_values),
        "null_residual": describe_distribution(residual_values),
        "empirical_lower_tail_p_for_residual_pr": float(
            (1 + np.sum(residual_array <= observed_residual)) / (repeats + 1)
        ),
        "model": (
            "X_ij = fitted grand + fitted ligand effect_i + fitted target effect_j + "
            "independent Gaussian residual with the observed target-specific residual variance"
        ),
        "interpretation": (
            "The null asks whether two-way centering alone creates the observed residual "
            "concentration. A residual PR below this null indicates correlated residual structure."
        ),
    }


def empirical_residual_permutation_null(
    matrix: np.ndarray,
    sample_size: int = 12000,
    repeats: int = 500,
    seed: int = SEED,
    cluster_labels: np.ndarray | None = None,
    cluster_repeats: int = 0,
) -> dict:
    """Nonparametric null that breaks cross-target residual correspondence.

    Each target column of the observed two-way-centered residual matrix is independently
    permuted across ligands before the same centering and column standardisation are
    reapplied.  The permutation step exactly preserves every empirical target-wise residual
    marginal without a Gaussian assumption; the subsequent transformation-matched
    re-centering can modify those marginals slightly.  An optional one-ligand-per-cluster
    sensitivity compares observed and permuted residual PR on matched chemically
    de-redundant supports.
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    support, rng, support_selection = fixed_ligand_support_indices(
        len(matrix), sample_size, seed
    )
    work = matrix[support]
    residual = two_way_center(work)
    observed_residual = rank_summary(residual)["participation_ratio"]
    null_values: list[float] = []
    for _ in range(repeats):
        permuted = np.column_stack([
            rng.permutation(residual[:, column])
            for column in range(residual.shape[1])
        ])
        null_values.append(
            rank_summary(two_way_center(permuted))["participation_ratio"]
        )
    null_array = np.asarray(null_values, dtype=float)
    report = {
        "sample_n": int(len(work)),
        "n_targets": int(work.shape[1]),
        "repeats": repeats,
        "support_selection": support_selection,
        "observed_residual": float(observed_residual),
        "null_residual": describe_distribution(null_array),
        "empirical_lower_tail_p_for_residual_pr": float(
            (1 + np.sum(null_array <= observed_residual)) / (repeats + 1)
        ),
        "model": (
            "independently permute every observed two-way-centered residual target column "
            "across ligands, then repeat two-way centering and column standardisation"
        ),
        "preserves_at_permutation_step": (
            "the empirical non-Gaussian marginal distribution of each target residual on "
            "the fixed support; transformation-matched re-centering can modify these "
            "marginals slightly"
        ),
        "does_not_preserve": (
            "cross-target residual correspondence or ligand-specific residual scale"
        ),
    }
    if cluster_labels is not None and cluster_repeats:
        labels = np.asarray(cluster_labels)
        if len(labels) != len(matrix):
            raise ValueError("cluster labels do not match empirical-null matrix rows")
        _, inverse = np.unique(labels, return_inverse=True)
        members = [
            np.flatnonzero(inverse == value)
            for value in range(int(inverse.max()) + 1)
        ]
        observed_cluster_values: list[float] = []
        null_cluster_values: list[float] = []
        paired_differences: list[float] = []
        for _ in range(cluster_repeats):
            index = np.asarray([
                member[rng.integers(0, len(member))]
                for member in members
            ])
            cluster_residual = two_way_center(matrix[index])
            cluster_observed = rank_summary(cluster_residual)["participation_ratio"]
            cluster_permuted = np.column_stack([
                rng.permutation(cluster_residual[:, column])
                for column in range(cluster_residual.shape[1])
            ])
            cluster_null = rank_summary(two_way_center(cluster_permuted))[
                "participation_ratio"
            ]
            observed_cluster_values.append(cluster_observed)
            null_cluster_values.append(cluster_null)
            paired_differences.append(cluster_null - cluster_observed)
        report["one_ligand_per_cluster_sensitivity"] = {
            "clusters": int(len(members)),
            "repeats": int(cluster_repeats),
            "observed_residual": describe_distribution(observed_cluster_values),
            "matched_empirical_null_residual": describe_distribution(null_cluster_values),
            "paired_null_minus_observed": describe_distribution(paired_differences),
            "scope": (
                "one randomly selected ligand per supplied chemical cluster in every "
                "realization; observed and independently column-permuted residual PR use "
                "the identical de-redundant support"
            ),
        }
    return report


def row_norm_preserving_residual_null(
    matrix: np.ndarray,
    sample_size: int = 12000,
    repeats: int = 500,
    seed: int = SEED,
) -> dict:
    """Random-direction null preserving each observed centered residual row norm.

    Every simulated row is a random Gaussian direction projected into the target-zero-sum
    subspace and then rescaled to the corresponding observed residual Euclidean norm. This
    retains ligand-specific residual scale exactly while destroying target alignment.
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    support, rng, support_selection = fixed_ligand_support_indices(
        len(matrix), sample_size, seed
    )
    work = matrix[support]
    residual = two_way_center(work)
    row_norms = np.linalg.norm(residual, axis=1)
    observed_residual = rank_summary(residual)["participation_ratio"]
    null_values = np.empty(repeats, dtype=float)
    for repetition in range(repeats):
        direction = rng.normal(size=residual.shape)
        direction -= direction.mean(axis=1, keepdims=True)
        norms = np.linalg.norm(direction, axis=1)
        simulated = direction * np.divide(
            row_norms,
            norms,
            out=np.zeros_like(row_norms),
            where=norms > 0,
        )[:, None]
        null_values[repetition] = rank_summary(simulated)["participation_ratio"]
    return {
        "sample_n": int(len(work)),
        "n_targets": int(work.shape[1]),
        "repeats": int(repeats),
        "support_selection": support_selection,
        "observed_residual": float(observed_residual),
        "null_residual": describe_distribution(null_values),
        "empirical_lower_tail_p_for_residual_pr": float(
            (1 + np.sum(null_values <= observed_residual)) / (repeats + 1)
        ),
        "model": (
            "independent random directions in the target-zero-sum subspace, with every "
            "simulated residual row rescaled to its observed Euclidean norm"
        ),
        "preserves_exactly": (
            "fixed ligand support, target count, zero sum within every row, and every "
            "ligand-specific residual row norm before column standardization"
        ),
        "does_not_preserve": (
            "target-specific marginal distributions, residual variances, or cross-target alignment"
        ),
        "interpretation": (
            "A residual PR below this null cannot be explained solely by heterogeneous "
            "ligand-specific residual magnitude plus the zero-row-sum constraint."
        ),
    }


def residualized_rank_correlation(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> dict:
    """Partial Spearman correlation of x and y after linear removal of ranked z."""
    rx, ry, rz = (stats.rankdata(v) for v in (x, y, z))
    design = np.column_stack([np.ones(len(rz)), rz])
    ex = rx - design @ np.linalg.lstsq(design, rx, rcond=None)[0]
    ey = ry - design @ np.linalg.lstsq(design, ry, rcond=None)[0]
    r, p = stats.pearsonr(ex, ey)
    return {"rho": float(r), "p_value": float(p)}


def participation_ratio_from_correlation(corr: np.ndarray) -> float:
    """Participation ratio from a possibly pairwise-complete correlation matrix."""
    corr = np.asarray(corr, dtype=np.float64)
    return float(np.trace(corr) ** 2 / np.square(corr).sum())


def observed_additive_least_squares_imputation(
    values: np.ndarray,
    tolerance: float = 1e-12,
    max_iter: int = 1000,
) -> tuple[np.ndarray, dict]:
    """Fill missing cells with the fitted observed-cell additive surface."""
    x = np.asarray(values, dtype=np.float64)
    observed = np.isfinite(x)
    if np.any(observed.sum(axis=1) == 0) or np.any(observed.sum(axis=0) == 0):
        raise ValueError("additive imputation requires an observation in every row and column")
    grand = float(np.nanmean(x))
    row_effect = np.zeros(x.shape[0], dtype=np.float64)
    column_effect = np.zeros(x.shape[1], dtype=np.float64)

    for iteration in range(1, max_iter + 1):
        previous = np.concatenate([[grand], row_effect, column_effect])
        row_effect = np.nanmean(x - grand - column_effect[None, :], axis=1)
        column_effect = np.nanmean(x - grand - row_effect[:, None], axis=0)
        shift = float(np.average(column_effect, weights=observed.sum(axis=0)))
        column_effect -= shift
        grand += shift
        current = np.concatenate([[grand], row_effect, column_effect])
        if np.max(np.abs(current - previous)) < tolerance:
            break
    else:
        raise RuntimeError("observed-cell additive least squares did not converge")

    fitted = grand + row_effect[:, None] + column_effect[None, :]
    completed = np.where(observed, x, fitted)
    return completed, {
        "iterations": int(iteration),
        "tolerance": float(tolerance),
        "observed_cells": int(observed.sum()),
        "imputed_cells": int((~observed).sum()),
        "assumption": (
            "Each missing ligand-target interaction residual is set to zero after fitting "
            "the additive ligand and target effects on observed cells."
        ),
    }


def preprocessing_sensitivity(
    docking_frame: pd.DataFrame,
    smiles: pd.Series | None = None,
    butina_labels: pd.Series | np.ndarray | None = None,
    unclipped_docking_frame: pd.DataFrame | None = None,
) -> dict:
    """Quantify the preprocessing alternatives requested during internal review."""
    numeric = docking_frame.apply(pd.to_numeric, errors="coerce")
    mean_filled = numeric.fillna(numeric.mean())
    median_filled = numeric.fillna(numeric.median())
    complete = numeric.dropna(axis=0, how="any")
    additive_filled_values, additive_fit = observed_additive_least_squares_imputation(
        numeric.to_numpy(dtype=np.float64)
    )

    x = mean_filled.to_numpy(dtype=np.float64)
    interaction = two_way_center(x)
    z = (x - x.mean(axis=0)) / x.std(axis=0, ddof=1)
    robust_scale = np.median(np.abs(x - np.median(x, axis=0)), axis=0) * 1.4826
    robust_scale[robust_scale < 1e-12] = 1.0
    robust_z = (x - np.median(x, axis=0)) / robust_scale

    ranked = numeric.rank(axis=0, method="average", na_option="keep")
    pairwise_corr = numeric.corr(min_periods=100).to_numpy(dtype=np.float64)
    spearman_corr = ranked.corr(min_periods=100).to_numpy(dtype=np.float64)

    def missing_ladder(values: np.ndarray | pd.DataFrame, method: str, assumption: str) -> dict:
        array = np.asarray(values, dtype=np.float64)
        raw_pr = rank_summary(array)["participation_ratio"]
        residual_pr = rank_summary(two_way_center(array))["participation_ratio"]
        return {
            "method": method,
            "assumption": assumption,
            "n_ligands": int(array.shape[0]),
            "n_targets": int(array.shape[1]),
            "raw_participation_ratio": raw_pr,
            "residual_participation_ratio": residual_pr,
            "residual_minus_raw": residual_pr - raw_pr,
            "residual_to_raw_ratio": residual_pr / raw_pr,
            "residual_fraction_of_algebraic_ceiling": residual_pr / (array.shape[1] - 1),
        }

    missing_methods = {
        "target_mean": missing_ladder(
            mean_filled,
            "target mean imputation",
            "missing scores equal the observed target mean",
        ),
        "target_median": missing_ladder(
            median_filled,
            "target median imputation",
            "missing scores equal the observed target median",
        ),
        "observed_additive_least_squares": {
            **missing_ladder(
                additive_filled_values,
                "observed-cell additive least-squares imputation",
                additive_fit["assumption"],
            ),
            "fit": additive_fit,
        },
        "complete_case": missing_ladder(
            complete,
            "complete-case deletion",
            "changes chemical support and is not a like-for-like imputation analysis",
        ),
        "mean_imputed_restricted_to_complete_rows": missing_ladder(
            mean_filled.loc[complete.index],
            "primary mean-imputed matrix restricted to complete rows",
            "equals complete-case input exactly because these rows contain no missing cells",
        ),
    }
    most_missing_target = str(numeric.isna().sum(axis=0).idxmax())
    without_target = numeric.drop(columns=[most_missing_target])
    without_target_mean = without_target.fillna(without_target.mean())
    without_target_median = without_target.fillna(without_target.median())
    without_target_complete = without_target.dropna(axis=0, how="any")
    missing_without_most_affected_target = {
        "removed_target": most_missing_target,
        "removed_target_missing_cells": int(numeric[most_missing_target].isna().sum()),
        "target_mean": missing_ladder(
            without_target_mean,
            "target mean imputation after target removal",
            "removes the target with the largest missing-cell count",
        ),
        "target_median": missing_ladder(
            without_target_median,
            "target median imputation after target removal",
            "removes the target with the largest missing-cell count",
        ),
        "complete_case": missing_ladder(
            without_target_complete,
            "complete cases after target removal",
            "changes both chemical and target support",
        ),
    }

    rng = np.random.default_rng(SEED + 80)
    random_equal_support = []
    for _ in range(100):
        indices = rng.choice(len(mean_filled), len(complete), replace=False)
        random_equal_support.append(
            rank_summary(two_way_center(mean_filled.iloc[indices].to_numpy(dtype=np.float64)))[
                "participation_ratio"
            ]
        )

    report = {
        "input_characterization": {
            "n_ligands": int(len(numeric)),
            "n_targets": int(numeric.shape[1]),
            "missing_cells": int(numeric.isna().to_numpy().sum()),
            "missing_fraction": float(numeric.isna().to_numpy().mean()),
            "rows_with_any_missing": int(numeric.isna().any(axis=1).sum()),
            "complete_rows": int(len(complete)),
            "zero_after_primary_clipping_cells": int((numeric.to_numpy() == 0).sum()),
            "zero_after_primary_clipping_fraction": float((numeric.to_numpy() == 0).mean()),
            "strictly_positive_cells_after_primary_clipping": int(
                (numeric.to_numpy() > 0).sum()
            ),
        },
        "raw_participation_ratio": {
            "target_mean_imputation_primary": rank_summary(mean_filled)["participation_ratio"],
            "target_median_imputation": rank_summary(median_filled)["participation_ratio"],
            "complete_case": rank_summary(complete)["participation_ratio"],
            "target_mean_imputed_matrix_restricted_to_complete_case_rows": rank_summary(
                mean_filled.loc[complete.index]
            )["participation_ratio"],
            "complete_case_n": int(len(complete)),
            "pairwise_complete_correlation": participation_ratio_from_correlation(pairwise_corr),
            "spearman_pairwise_correlation": participation_ratio_from_correlation(spearman_corr),
        },
        "interaction_participation_ratio": {
            "center_raw_then_standardize_primary": rank_summary(interaction)["participation_ratio"],
            "standardize_then_center": rank_summary(two_way_center(z))["participation_ratio"],
            "robust_scale_then_center": rank_summary(two_way_center(robust_z))["participation_ratio"],
        },
        "missing_value_residual_sensitivity": {
            "methods": missing_methods,
            "excluded_method": {
                "method": "nearest-neighbour imputation",
                "reason": (
                    "It constructs each missing score from the other target scores and can "
                    "therefore induce the cross-target dependence being measured; it is also "
                    "computationally disproportionate for this sensitivity analysis."
                ),
            },
            "without_most_affected_target": missing_without_most_affected_target,
            "random_mean_imputed_supports_matched_to_complete_case_n": {
                "repeats": len(random_equal_support),
                "seed": SEED + 80,
                "residual_participation_ratio": describe_distribution(
                    random_equal_support
                ),
                "complete_case_residual_participation_ratio": missing_methods[
                    "complete_case"
                ]["residual_participation_ratio"],
                "interpretation": (
                    "This separates the effect of reducing N from the chemical-support "
                    "selection induced by complete-case deletion."
                ),
            },
            "interpretation_boundary": (
                "Imputation methods encode different assumptions about an unobserved "
                "ligand-target interaction. Their spread is a model-sensitivity range, "
                "not a confidence interval. Pairwise-complete residual correlations are "
                "not used as a spectrum because their correlation matrix is not guaranteed "
                "to be positive semidefinite."
            ),
        },
        "invariance_note": (
            "For a raw complete matrix, the correlation spectrum is algebraically invariant to "
            "non-zero affine rescaling of individual columns. Robust scaling matters only when "
            "performed before row-effect removal."
        ),
    }
    if unclipped_docking_frame is not None:
        unclipped_numeric = unclipped_docking_frame.apply(pd.to_numeric, errors="coerce")
        if unclipped_numeric.shape != numeric.shape:
            raise ValueError("unclipped Docking-44 frame does not match processed input")
        report["input_characterization"].update({
            "strictly_positive_cells_before_primary_clipping": int(
                (unclipped_numeric.to_numpy() > 0).sum()
            ),
            "strictly_positive_fraction_before_primary_clipping": float(
                (unclipped_numeric.to_numpy() > 0).mean()
            ),
        })
    if smiles is not None:
        descriptor_frame = molecular_descriptor_frame(pd.Series(smiles).reset_index(drop=True))
        complete_mask = ~numeric.isna().any(axis=1).to_numpy()
        descriptor_comparison = {}
        for descriptor in descriptor_frame:
            complete_values = descriptor_frame.loc[complete_mask, descriptor].dropna().to_numpy()
            incomplete_values = descriptor_frame.loc[~complete_mask, descriptor].dropna().to_numpy()
            pooled_sd = np.sqrt(
                ((len(complete_values) - 1) * complete_values.var(ddof=1)
                 + (len(incomplete_values) - 1) * incomplete_values.var(ddof=1))
                / (len(complete_values) + len(incomplete_values) - 2)
            )
            descriptor_comparison[descriptor] = {
                "complete_median": float(np.median(complete_values)),
                "complete_iqr": [
                    float(np.quantile(complete_values, 0.25)),
                    float(np.quantile(complete_values, 0.75)),
                ],
                "incomplete_median": float(np.median(incomplete_values)),
                "incomplete_iqr": [
                    float(np.quantile(incomplete_values, 0.25)),
                    float(np.quantile(incomplete_values, 0.75)),
                ],
                "standardized_mean_difference_incomplete_minus_complete": float(
                    (incomplete_values.mean() - complete_values.mean()) / pooled_sd
                ),
            }
        missing_per_row = numeric.isna().sum(axis=1).to_numpy()
        target_missing = numeric.isna().sum(axis=0).sort_values(ascending=False)
        chemistry = {
            "complete_rows": int(complete_mask.sum()),
            "incomplete_rows": int((~complete_mask).sum()),
            "missing_cells_per_incomplete_row": describe_distribution(
                missing_per_row[~complete_mask]
            ),
            "descriptor_comparison": descriptor_comparison,
            "targets_with_most_missing_cells": {
                str(target): int(value) for target, value in target_missing.head(10).items()
            },
            "interpretation_boundary": (
                "complete-case deletion changes chemical support; descriptor differences "
                "describe selection but do not identify the missingness mechanism"
            ),
        }
        if butina_labels is not None:
            labels = np.asarray(butina_labels)
            chemistry["butina_cluster_support"] = {
                "clusters_total": int(len(np.unique(labels))),
                "clusters_among_complete_rows": int(len(np.unique(labels[complete_mask]))),
                "clusters_among_incomplete_rows": int(len(np.unique(labels[~complete_mask]))),
            }
        report["complete_case_chemical_support"] = chemistry
    return report


def matched_experimental_matrices() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Rebuild the primary exact-relation-median 74-by-6 comparison."""
    dock = pd.read_csv(source_path("negative_results_paper/analysis/honest_dock_scores.csv"))
    dmat = dock[dock.target.isin(MATCHED_TARGETS)].pivot_table(
        index="inchikey", columns="target", values="dock"
    ).reindex(columns=MATCHED_TARGETS)
    activities = pd.read_csv(
        source_path("negative_results_paper/analysis/chembl_activities_full.csv")
    )
    activities = activities[
        activities.panel_key.isin(MATCHED_TARGETS)
        & activities.pchembl_value.notna()
        & activities.standard_relation.eq("=")
        & activities.standard_type.isin(["Ki", "Kd", "IC50", "EC50"])
    ].copy()
    smiles_to_key = {}
    for smiles in activities.canonical_smiles.dropna().unique():
        molecule = Chem.MolFromSmiles(str(smiles))
        smiles_to_key[smiles] = Chem.MolToInchiKey(molecule) if molecule is not None else None
    activities["inchikey"] = activities.canonical_smiles.map(smiles_to_key)
    emat = activities.pivot_table(
        index="inchikey",
        columns="panel_key",
        values="pchembl_value",
        aggfunc="median",
    ).reindex(columns=MATCHED_TARGETS)
    common = dmat.dropna().index.intersection(emat[emat.notna().sum(axis=1) >= 5].index)
    dmat = dmat.loc[common]
    emat_observed = emat.loc[common]
    emat_filled = emat_observed.fillna(emat_observed.median())
    return dmat, emat_observed, emat_filled


def leave_one_ligand_out_docking_representations(
    docking: np.ndarray, experiment_observed: np.ndarray
) -> dict[str, np.ndarray]:
    """Construct operational score representations without held-out-ligand leakage."""
    docking = np.asarray(docking, dtype=np.float64)
    experiment_observed = np.asarray(experiment_observed, dtype=np.float64)
    n, p = docking.shape
    outputs = {
        key: np.empty((n, p), dtype=np.float64)
        for key in [
            "absolute_vina",
            "column_standardized",
            "two_way_residual",
            "docking_target_prior",
            "experimental_target_prior",
        ]
    }
    for held_out in range(n):
        train = np.arange(n) != held_out
        train_matrix = docking[train]
        target_mean = train_matrix.mean(axis=0)
        target_sd = train_matrix.std(axis=0, ddof=1)
        grand = float(train_matrix.mean())
        train_residual = (
            train_matrix
            - target_mean[None, :]
            - train_matrix.mean(axis=1)[:, None]
            + grand
        )
        residual_sd = train_residual.std(axis=0, ddof=1)
        test = docking[held_out]
        outputs["absolute_vina"][held_out] = test
        outputs["column_standardized"][held_out] = (test - target_mean) / target_sd
        outputs["two_way_residual"][held_out] = (
            test - target_mean - test.mean() + grand
        ) / residual_sd
        outputs["docking_target_prior"][held_out] = target_mean
        outputs["experimental_target_prior"][held_out] = -np.nanmean(
            experiment_observed[train], axis=0
        )
    return outputs


def preference_metrics(
    scores: np.ndarray,
    experiment: np.ndarray,
    tie_tolerance: float = 0.0,
) -> dict:
    """Within-ligand target-preference metrics; lower scores predict higher pChEMBL."""
    scores = np.asarray(scores, dtype=np.float64)
    experiment = np.asarray(experiment, dtype=np.float64)
    per_ligand_accuracy = np.full(len(experiment), np.nan)
    per_ligand_spearman = np.full(len(experiment), np.nan)
    per_ligand_top1 = np.full(len(experiment), np.nan)
    per_ligand_pair_count = np.zeros(len(experiment), dtype=int)
    per_ligand_correct = np.zeros(len(experiment), dtype=float)
    total_correct = 0.0
    total_pairs = 0
    excluded_ties = 0
    predicted_score_ties = 0
    for ligand in range(len(experiment)):
        observed = np.flatnonzero(
            np.isfinite(experiment[ligand]) & np.isfinite(scores[ligand])
        )
        correct = 0.0
        pairs = 0
        for first in range(len(observed)):
            for second in range(first + 1, len(observed)):
                i, j = observed[first], observed[second]
                experimental_difference = experiment[ligand, i] - experiment[ligand, j]
                if abs(experimental_difference) <= tie_tolerance:
                    excluded_ties += 1
                    continue
                score_difference = scores[ligand, i] - scores[ligand, j]
                if score_difference == 0:
                    correct += 0.5
                    predicted_score_ties += 1
                else:
                    correct += int(
                        np.sign(experimental_difference) == -np.sign(score_difference)
                    )
                pairs += 1
        if pairs:
            per_ligand_accuracy[ligand] = correct / pairs
            per_ligand_pair_count[ligand] = pairs
            per_ligand_correct[ligand] = correct
        total_correct += correct
        total_pairs += pairs
        if len(observed) >= 3:
            per_ligand_spearman[ligand] = stats.spearmanr(
                -scores[ligand, observed], experiment[ligand, observed]
            ).statistic
        if len(observed):
            per_ligand_top1[ligand] = float(
                observed[np.argmin(scores[ligand, observed])]
                == observed[np.argmax(experiment[ligand, observed])]
            )
    finite_accuracy = np.isfinite(per_ligand_accuracy)
    finite_spearman = np.isfinite(per_ligand_spearman)
    finite_top1 = np.isfinite(per_ligand_top1)
    return {
        "per_ligand_pairwise_accuracy": per_ligand_accuracy,
        "per_ligand_spearman": per_ligand_spearman,
        "per_ligand_top1": per_ligand_top1,
        "per_ligand_pair_count": per_ligand_pair_count,
        "per_ligand_correct": per_ligand_correct,
        "mean_per_ligand_pairwise_accuracy": (
            float(np.nanmean(per_ligand_accuracy)) if finite_accuracy.any() else float("nan")
        ),
        "pair_weighted_accuracy": (
            float(total_correct / total_pairs) if total_pairs else float("nan")
        ),
        "mean_per_ligand_spearman": (
            float(np.nanmean(per_ligand_spearman)) if finite_spearman.any() else float("nan")
        ),
        "top1_accuracy": (
            float(np.nanmean(per_ligand_top1)) if finite_top1.any() else float("nan")
        ),
        "evaluated_ligands": int(finite_accuracy.sum()),
        "spearman_ligands": int(finite_spearman.sum()),
        "evaluated_pairs": int(total_pairs),
        "excluded_ties": int(excluded_ties),
        "predicted_score_ties_half_credit": int(predicted_score_ties),
    }


def bootstrap_vector_mean(
    values: np.ndarray,
    repeats: int,
    seed: int,
    cluster_labels: np.ndarray | None = None,
) -> dict:
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    records = []
    if cluster_labels is None:
        for _ in range(repeats):
            index = rng.integers(0, len(values), len(values))
            records.append(float(np.nanmean(values[index])))
        n_clusters = None
    else:
        _, inverse = np.unique(np.asarray(cluster_labels), return_inverse=True)
        members = [np.flatnonzero(inverse == value) for value in range(inverse.max() + 1)]
        for _ in range(repeats):
            sampled = rng.integers(0, len(members), len(members))
            index = np.concatenate([members[value] for value in sampled])
            records.append(float(np.nanmean(values[index])))
        n_clusters = len(members)
    report = describe_distribution(records)
    report["plugin_mean"] = float(np.nanmean(values))
    report["n_clusters"] = n_clusters
    return report


def target_preference_benchmark(
    docking: pd.DataFrame,
    experiment_observed: pd.DataFrame,
    smiles: pd.Series,
    bootstrap_repeats: int = 5000,
    shuffle_repeats: int = 5000,
    imputations: int = 100,
    seed: int = SEED,
) -> dict:
    """Reverse-docking preference benchmark with target-prior controls."""
    docking_values = docking.to_numpy(dtype=np.float64)
    experiment_values = experiment_observed.to_numpy(dtype=np.float64)
    score_matrices = leave_one_ligand_out_docking_representations(
        docking_values, experiment_values
    )
    keys = scaffold_keys(smiles.reindex(docking.index).reset_index(drop=True))
    reports = {}
    metric_cache = {}
    for offset, (name, scores) in enumerate(score_matrices.items()):
        metric = preference_metrics(scores, experiment_values)
        tolerance = preference_metrics(scores, experiment_values, tie_tolerance=0.10)
        metric_cache[name] = metric
        reports[name] = {
            key: value
            for key, value in metric.items()
            if not key.startswith("per_ligand_")
        }
        reports[name]["ligand_bootstrap"] = bootstrap_vector_mean(
            metric["per_ligand_pairwise_accuracy"],
            repeats=bootstrap_repeats,
            seed=seed + 10 + offset,
        )
        reports[name]["scaffold_cluster_bootstrap"] = bootstrap_vector_mean(
            metric["per_ligand_pairwise_accuracy"],
            repeats=bootstrap_repeats,
            seed=seed + 20 + offset,
            cluster_labels=keys,
        )
        reports[name]["tie_tolerance_0.10"] = {
            "mean_per_ligand_pairwise_accuracy": tolerance[
                "mean_per_ligand_pairwise_accuracy"
            ],
            "pair_weighted_accuracy": tolerance["pair_weighted_accuracy"],
            "evaluated_pairs": tolerance["evaluated_pairs"],
            "excluded_ties": tolerance["excluded_ties"],
        }

    comparisons = {}
    for first, second in [
        ("two_way_residual", "column_standardized"),
        ("two_way_residual", "absolute_vina"),
        ("column_standardized", "absolute_vina"),
        ("absolute_vina", "docking_target_prior"),
    ]:
        difference = (
            metric_cache[first]["per_ligand_pairwise_accuracy"]
            - metric_cache[second]["per_ligand_pairwise_accuracy"]
        )
        label = f"{first}_minus_{second}"
        comparisons[label] = {
            "plugin_mean_difference": float(np.nanmean(difference)),
            "ligand_bootstrap": bootstrap_vector_mean(
                difference, bootstrap_repeats, seed + 100 + len(comparisons)
            ),
            "scaffold_cluster_bootstrap": bootstrap_vector_mean(
                difference,
                bootstrap_repeats,
                seed + 200 + len(comparisons),
                cluster_labels=keys,
            ),
        }
    residual_column_difference = (
        metric_cache["two_way_residual"]["per_ligand_pairwise_accuracy"]
        - metric_cache["column_standardized"]["per_ligand_pairwise_accuracy"]
    )
    standard_error = float(
        np.nanstd(residual_column_difference, ddof=1)
        / np.sqrt(np.isfinite(residual_column_difference).sum())
    )
    comparisons["two_way_residual_minus_column_standardized"][
        "normal_approx_80pct_power_mde_two_sided_alpha_0.05"
    ] = float((stats.norm.ppf(0.975) + stats.norm.ppf(0.80)) * standard_error)

    rng = np.random.default_rng(seed + 300)
    for name in ["absolute_vina", "column_standardized", "two_way_residual"]:
        null_values = []
        for _ in range(shuffle_repeats):
            shuffled = score_matrices[name][rng.permutation(len(docking_values))]
            null_values.append(
                preference_metrics(shuffled, experiment_values)[
                    "mean_per_ligand_pairwise_accuracy"
                ]
            )
        null_array = np.asarray(null_values)
        observed = reports[name]["mean_per_ligand_pairwise_accuracy"]
        reports[name]["ligand_identity_shuffle_null"] = {
            **describe_distribution(null_array),
            "one_sided_empirical_p_observed_at_least_as_large": float(
                (1 + np.sum(null_array >= observed)) / (shuffle_repeats + 1)
            ),
            "repeats": shuffle_repeats,
        }

    imputation_records = {
        name: {"pairwise_accuracy": [], "spearman": [], "top1": []}
        for name in ["absolute_vina", "column_standardized", "two_way_residual"]
    }
    imputation_pr = {"raw": [], "residual": [], "raw_difference": [], "residual_difference": []}
    observed_min = np.nanmin(experiment_values, axis=0)
    observed_max = np.nanmax(experiment_values, axis=0)
    for imputation in range(imputations):
        imputer = IterativeImputer(
            estimator=BayesianRidge(),
            sample_posterior=True,
            max_iter=20,
            random_state=seed + 400 + imputation,
            min_value=observed_min,
            max_value=observed_max,
        )
        completed = imputer.fit_transform(experiment_values)
        exp_raw = rank_summary(completed)["participation_ratio"]
        exp_residual = rank_summary(two_way_center(completed))["participation_ratio"]
        dock_raw = rank_summary(docking_values)["participation_ratio"]
        dock_residual = rank_summary(two_way_center(docking_values))["participation_ratio"]
        imputation_pr["raw"].append(exp_raw)
        imputation_pr["residual"].append(exp_residual)
        imputation_pr["raw_difference"].append(exp_raw - dock_raw)
        imputation_pr["residual_difference"].append(exp_residual - dock_residual)
        for name in imputation_records:
            metric = preference_metrics(score_matrices[name], completed)
            imputation_records[name]["pairwise_accuracy"].append(
                metric["mean_per_ligand_pairwise_accuracy"]
            )
            imputation_records[name]["spearman"].append(
                metric["mean_per_ligand_spearman"]
            )
            imputation_records[name]["top1"].append(metric["top1_accuracy"])

    target_docking_mean = docking.mean(axis=0).to_numpy(dtype=float)
    target_experimental_mean = experiment_observed.mean(axis=0).to_numpy(dtype=float)
    return {
        "n_ligands": int(len(docking)),
        "n_targets": int(docking.shape[1]),
        "observed_experimental_cells": int(np.isfinite(experiment_values).sum()),
        "missing_experimental_cells": int(np.isnan(experiment_values).sum()),
        "primary_estimand": (
            "mean per-ligand pairwise target-preference accuracy over observed exact-relation "
            "median pChEMBL cells; exact experimental ties excluded"
        ),
        "sign_convention": (
            "fixed a priori from the Vina energy convention: more negative docking scores "
            "predict higher pChEMBL"
        ),
        "cross_fitting": (
            "leave one ligand out; target means, grand mean, raw target standard deviations, "
            "and residual target standard deviations estimated only from the other 73 ligands"
        ),
        "representations": reports,
        "paired_comparisons": comparisons,
        "matched_scaffold_clusters": int(len(np.unique(keys))),
        "target_mean_alignment": {
            "targets": [str(value) for value in docking.columns],
            "minus_mean_docking_score": [float(value) for value in -target_docking_mean],
            "mean_experimental_pchembl": [float(value) for value in target_experimental_mean],
            "spearman_rho_minus_docking_mean_vs_experimental_mean": float(
                stats.spearmanr(-target_docking_mean, target_experimental_mean).statistic
            ),
            "two_sided_p_value": float(
                stats.spearmanr(-target_docking_mean, target_experimental_mean).pvalue
            ),
            "n_targets": int(docking.shape[1]),
        },
        "multiple_imputation": {
            "method": (
                "100 posterior draws from chained Bayesian-ridge regressions; imputed values "
                "bounded by the observed range of each target"
            ),
            "n_imputations": imputations,
            "participation_ratio": {
                key: describe_distribution(value) for key, value in imputation_pr.items()
            },
            "downstream_metrics": {
                name: {
                    metric: describe_distribution(values)
                    for metric, values in record.items()
                }
                for name, record in imputation_records.items()
            },
        },
        "boundary": (
            "This six-target benchmark tests operational target ranking on one matched ChEMBL "
            "support; it does not estimate reverse-docking accuracy for other panels."
        ),
    }


def full_inchikeys(smiles: pd.Series) -> pd.Series:
    """Generate version-pinned full InChIKeys from a SMILES series."""
    mapping: dict[str, str | None] = {}
    for value in smiles.dropna().astype(str).unique():
        molecule = Chem.MolFromSmiles(value)
        mapping[value] = Chem.MolToInchiKey(molecule) if molecule is not None else None
    return smiles.astype("string").map(mapping)


def external_reference_score_representations(
    docking_reference: pd.DataFrame,
    evaluation_ids: pd.Index,
    evaluation_targets: list[str],
    experimental_reference: pd.DataFrame,
) -> tuple[dict[str, np.ndarray], dict]:
    """Fit score transformations outside the evaluated ligand set.

    Docking offsets and scales are estimated from the complete Docking-44 reference after
    removing every evaluated ligand.  The experimental target prior is estimated from the
    exact-relation ChEMBL matrix after the same batch holdout.  No experimental values from
    evaluated ligands enter any score representation.
    """
    evaluation_ids = pd.Index(evaluation_ids)
    docking_train_raw = docking_reference.drop(index=evaluation_ids, errors="ignore")
    target_imputation_mean = docking_train_raw.mean(axis=0)
    docking_train = docking_train_raw.fillna(target_imputation_mean)
    docking_test_raw = docking_reference.loc[evaluation_ids]
    docking_test = docking_test_raw.fillna(target_imputation_mean)
    target_mean = docking_train.mean(axis=0)
    target_sd = docking_train.std(axis=0, ddof=1)
    grand_mean = float(docking_train.to_numpy(dtype=float).mean())
    train_residual = (
        docking_train
        - target_mean
        - docking_train.mean(axis=1).to_numpy()[:, None]
        + grand_mean
    )
    residual_sd = train_residual.std(axis=0, ddof=1)
    test_residual = (
        docking_test
        - target_mean
        - docking_test.mean(axis=1).to_numpy()[:, None]
        + grand_mean
    )
    experimental_train = experimental_reference.drop(index=evaluation_ids, errors="ignore")
    experimental_target_mean = experimental_train.mean(axis=0)
    selected = list(evaluation_targets)
    n = len(docking_test)
    score_matrices = {
        "absolute_vina": docking_test[selected].to_numpy(dtype=float),
        "target_centered_unscaled": (
            docking_test[selected] - target_mean[selected]
        ).to_numpy(dtype=float),
        "two_way_centered_unscaled": test_residual[selected].to_numpy(dtype=float),
        "column_standardized": (
            (docking_test[selected] - target_mean[selected]) / target_sd[selected]
        ).to_numpy(dtype=float),
        "target_centered_residual_scaled": (
            (docking_test[selected] - target_mean[selected]) / residual_sd[selected]
        ).to_numpy(dtype=float),
        "two_way_residual": (
            test_residual[selected] / residual_sd[selected]
        ).to_numpy(dtype=float),
        "docking_target_prior": np.tile(target_mean[selected].to_numpy(dtype=float), (n, 1)),
        "experimental_target_prior": np.tile(
            -experimental_target_mean[selected].to_numpy(dtype=float), (n, 1)
        ),
    }
    nonfinite = {
        name: int((~np.isfinite(values)).sum())
        for name, values in score_matrices.items()
        if not np.isfinite(values).all()
    }
    if nonfinite:
        raise ValueError(
            f"Expanded preference benchmark contains non-finite score values: {nonfinite}"
        )
    return score_matrices, {
        "docking_training_ligands": int(len(docking_train)),
        "experimental_prior_training_ligands": int(len(experimental_train)),
        "evaluation_ligands_excluded_from_both_references": int(len(evaluation_ids)),
        "docking_training_cells_imputed_from_external_target_means": int(
            docking_train_raw.isna().sum().sum()
        ),
        "docking_evaluation_cells_imputed_from_external_target_means": int(
            docking_test_raw.isna().sum().sum()
        ),
        "row_effect_for_evaluated_ligand": "mean over all 44 Docking-44 target scores",
        "ranking_decomposition": (
            "target_centered_unscaled and two_way_centered_unscaled differ only by a "
            "within-ligand constant and must induce identical target rankings; "
            "target_centered_residual_scaled isolates the residual-scale choice, while "
            "two_way_residual additionally contains the ligand-offset by inverse-scale tilt"
        ),
    }


def mean_pairwise_preference_accuracy(
    scores: np.ndarray,
    experiment: np.ndarray,
    tie_tolerance: float = 0.0,
) -> float:
    """Fast equal-ligand accuracy used inside permutation loops."""
    ligand_values: list[float] = []
    for ligand in range(len(experiment)):
        observed = np.flatnonzero(
            np.isfinite(experiment[ligand]) & np.isfinite(scores[ligand])
        )
        correct = 0.0
        pairs = 0
        for first in range(len(observed)):
            for second in range(first + 1, len(observed)):
                i, j = observed[first], observed[second]
                experimental_difference = experiment[ligand, i] - experiment[ligand, j]
                if abs(experimental_difference) <= tie_tolerance:
                    continue
                score_difference = scores[ligand, i] - scores[ligand, j]
                correct += (
                    0.5
                    if score_difference == 0
                    else int(
                        np.sign(experimental_difference)
                        == -np.sign(score_difference)
                    )
                )
                pairs += 1
        if pairs:
            ligand_values.append(correct / pairs)
    return float(np.mean(ligand_values)) if ligand_values else float("nan")


def preference_pair_arrays(experiment: np.ndarray) -> tuple[np.ndarray, ...]:
    """Flatten within-ligand target pairs for fast permutation evaluation."""
    ligands: list[int] = []
    first_targets: list[int] = []
    second_targets: list[int] = []
    for ligand in range(len(experiment)):
        observed = np.flatnonzero(np.isfinite(experiment[ligand]))
        for first in range(len(observed)):
            for second in range(first + 1, len(observed)):
                ligands.append(ligand)
                first_targets.append(int(observed[first]))
                second_targets.append(int(observed[second]))
    ligand_array = np.asarray(ligands, dtype=int)
    first_array = np.asarray(first_targets, dtype=int)
    second_array = np.asarray(second_targets, dtype=int)
    truth = np.sign(
        experiment[ligand_array, first_array]
        - experiment[ligand_array, second_array]
    )
    return ligand_array, first_array, second_array, truth


def accuracy_from_pair_arrays(
    scores: np.ndarray,
    pair_arrays: tuple[np.ndarray, ...],
    truth: np.ndarray | None = None,
) -> float:
    """Equal-ligand accuracy from flattened target-pair arrays."""
    ligand, first, second, original_truth = pair_arrays
    comparison = original_truth if truth is None else np.asarray(truth)
    valid = comparison != 0
    if not valid.any():
        return float("nan")
    score_difference = (
        scores[ligand[valid], first[valid]]
        - scores[ligand[valid], second[valid]]
    )
    predicted = -np.sign(score_difference)
    correct = np.where(
        score_difference == 0,
        0.5,
        (predicted == comparison[valid]).astype(float),
    )
    correct_by_ligand = np.bincount(
        ligand[valid], weights=correct, minlength=len(scores)
    )
    pairs_by_ligand = np.bincount(ligand[valid], minlength=len(scores))
    keep = pairs_by_ligand > 0
    return float(np.mean(correct_by_ligand[keep] / pairs_by_ligand[keep]))


def paired_preference_comparison(
    first: np.ndarray,
    second: np.ndarray,
    bootstrap_repeats: int,
    seed: int,
    scaffold_labels: np.ndarray,
    butina_labels: np.ndarray | None = None,
) -> dict:
    """Paired ligand- and scaffold-cluster uncertainty for an accuracy contrast."""
    difference = np.asarray(first, float) - np.asarray(second, float)
    report = {
        "plugin_mean_difference": float(np.nanmean(difference)),
        "ligand_bootstrap": bootstrap_vector_mean(
            difference, bootstrap_repeats, seed
        ),
        "scaffold_cluster_bootstrap": bootstrap_vector_mean(
            difference,
            bootstrap_repeats,
            seed + 1,
            cluster_labels=scaffold_labels,
        ),
    }
    if butina_labels is not None:
        report["butina_cluster_bootstrap"] = bootstrap_vector_mean(
            difference,
            bootstrap_repeats,
            seed + 2,
            cluster_labels=butina_labels,
        )
    cluster_intervals_90 = [
        report["scaffold_cluster_bootstrap"]["interval_90"]
    ]
    if "butina_cluster_bootstrap" in report:
        cluster_intervals_90.append(
            report["butina_cluster_bootstrap"]["interval_90"]
        )
    conservative_interval_90 = [
        float(min(interval[0] for interval in cluster_intervals_90)),
        float(max(interval[1] for interval in cluster_intervals_90)),
    ]
    report["post_hoc_equivalence_sensitivity"] = {
        "conservative_cluster_bootstrap_interval_90": conservative_interval_90,
        "margins": {
            f"{margin:.2f}": {
                "equivalence_established": bool(
                    conservative_interval_90[0] > -margin
                    and conservative_interval_90[1] < margin
                ),
                "criterion": (
                    "the conservative union of Murcko- and Butina-cluster 90% bootstrap "
                    "intervals lies strictly inside the symmetric margin"
                ),
            }
            for margin in [0.02, 0.05]
        },
        "status": (
            "post hoc sensitivity; margins were declared during revision rather than "
            "prospectively registered"
        ),
    }
    finite = difference[np.isfinite(difference)]
    if len(finite) > 1:
        standard_error = float(finite.std(ddof=1) / np.sqrt(len(finite)))
        report["normal_approx_80pct_power_mde_two_sided_alpha_0.05"] = float(
            (stats.norm.ppf(0.975) + stats.norm.ppf(0.80)) * standard_error
        )
    return report


def expanded_preference_metrics(
    score_matrices: dict[str, np.ndarray],
    experiment_observed: pd.DataFrame,
    smiles: pd.Series,
    bootstrap_repeats: int,
    permutation_repeats: int,
    seed: int,
    full_sensitivities: bool,
    butina_labels: np.ndarray | None = None,
) -> dict:
    """Observed-cell target-preference benchmark on a sparse multi-target overlap."""
    experiment = experiment_observed.to_numpy(dtype=float)
    scaffold_labels = scaffold_keys(smiles.reindex(experiment_observed.index))
    if butina_labels is not None:
        butina_labels = np.asarray(butina_labels)
        if len(butina_labels) != len(experiment_observed):
            raise ValueError("Butina labels do not match operational benchmark rows")
    reports: dict[str, dict] = {}
    metric_cache: dict[str, dict] = {}
    margins = [0.0, 0.10, 0.50, 1.00]
    for offset, (name, scores) in enumerate(score_matrices.items()):
        metric = preference_metrics(scores, experiment)
        metric_cache[name] = metric
        reports[name] = {
            key: value
            for key, value in metric.items()
            if not key.startswith("per_ligand_")
        }
        reports[name]["ligand_bootstrap"] = bootstrap_vector_mean(
            metric["per_ligand_pairwise_accuracy"],
            bootstrap_repeats,
            seed + 10 + offset,
        )
        reports[name]["scaffold_cluster_bootstrap"] = bootstrap_vector_mean(
            metric["per_ligand_pairwise_accuracy"],
            bootstrap_repeats,
            seed + 20 + offset,
            cluster_labels=scaffold_labels,
        )
        if butina_labels is not None:
            reports[name]["butina_cluster_bootstrap"] = bootstrap_vector_mean(
                metric["per_ligand_pairwise_accuracy"],
                bootstrap_repeats,
                seed + 30 + offset,
                cluster_labels=butina_labels,
            )
        reports[name]["experimental_margin_sensitivity"] = {
            f"{margin:.2f}": {
                key: value
                for key, value in preference_metrics(
                    scores, experiment, tie_tolerance=margin
                ).items()
                if key
                in {
                    "mean_per_ligand_pairwise_accuracy",
                    "pair_weighted_accuracy",
                    "evaluated_ligands",
                    "evaluated_pairs",
                    "excluded_ties",
                }
            }
            for margin in margins
        }

    comparisons = {}
    for offset, (first, second) in enumerate([
        ("two_way_residual", "column_standardized"),
        ("two_way_residual", "absolute_vina"),
        ("two_way_residual", "docking_target_prior"),
        ("two_way_residual", "experimental_target_prior"),
        ("two_way_residual", "cohort_experimental_target_prior"),
        ("absolute_vina", "docking_target_prior"),
        ("two_way_centered_unscaled", "target_centered_unscaled"),
        ("two_way_residual", "target_centered_residual_scaled"),
        ("target_centered_residual_scaled", "target_centered_unscaled"),
        ("column_standardized", "target_centered_unscaled"),
    ]):
        comparisons[f"{first}_minus_{second}"] = paired_preference_comparison(
            metric_cache[first]["per_ligand_pairwise_accuracy"],
            metric_cache[second]["per_ligand_pairwise_accuracy"],
            bootstrap_repeats,
            seed + 100 + 4 * offset,
            scaffold_labels,
            butina_labels=butina_labels,
        )

    rng = np.random.default_rng(seed + 300)
    pair_arrays = preference_pair_arrays(experiment)
    pair_ligand, pair_first, pair_second, _ = pair_arrays
    null_names = ["absolute_vina", "column_standardized", "two_way_residual"]
    for name in null_names:
        scores = score_matrices[name]
        identity_null = []
        outcome_null = []
        for _ in range(permutation_repeats):
            identity_null.append(
                accuracy_from_pair_arrays(
                    scores[rng.permutation(len(scores))], pair_arrays
                )
            )
            permuted = experiment.copy()
            for ligand in range(len(permuted)):
                observed = np.flatnonzero(np.isfinite(permuted[ligand]))
                permuted[ligand, observed] = rng.permutation(permuted[ligand, observed])
            permuted_truth = np.sign(
                permuted[pair_ligand, pair_first]
                - permuted[pair_ligand, pair_second]
            )
            outcome_null.append(
                accuracy_from_pair_arrays(scores, pair_arrays, truth=permuted_truth)
            )
        observed = reports[name]["mean_per_ligand_pairwise_accuracy"]
        identity_array = np.asarray(identity_null, dtype=float)
        outcome_array = np.asarray(outcome_null, dtype=float)
        reports[name]["ligand_identity_shuffle_null"] = {
            **describe_distribution(identity_array),
            "one_sided_empirical_p_observed_at_least_as_large": float(
                (1 + np.sum(identity_array >= observed)) / (permutation_repeats + 1)
            ),
            "repeats": permutation_repeats,
        }
        reports[name]["within_ligand_outcome_permutation_null"] = {
            **describe_distribution(outcome_array),
            "one_sided_empirical_p_observed_at_least_as_large": float(
                (1 + np.sum(outcome_array >= observed)) / (permutation_repeats + 1)
            ),
            "repeats": permutation_repeats,
        }

    if full_sensitivities:
        _, scaffold_inverse = np.unique(scaffold_labels, return_inverse=True)
        scaffold_members = [
            np.flatnonzero(scaffold_inverse == cluster)
            for cluster in range(int(scaffold_inverse.max()) + 1)
        ]
        support_repeats = 25
        permutations_per_support = max(200, permutation_repeats // support_repeats)
        for name in null_names:
            support_observed: list[float] = []
            support_p_values: list[float] = []
            for _ in range(support_repeats):
                index = np.asarray([
                    member[rng.integers(0, len(member))]
                    for member in scaffold_members
                ])
                support_scores = score_matrices[name][index]
                support_experiment = experiment[index]
                support_pairs = preference_pair_arrays(support_experiment)
                observed = accuracy_from_pair_arrays(support_scores, support_pairs)
                null = np.asarray([
                    accuracy_from_pair_arrays(
                        support_scores[rng.permutation(len(support_scores))],
                        support_pairs,
                    )
                    for _ in range(permutations_per_support)
                ])
                support_observed.append(observed)
                support_p_values.append(
                    float((1 + np.sum(null >= observed)) / (len(null) + 1))
                )
            reports[name]["one_ligand_per_murcko_cluster_identity_permutation"] = {
                "clusters": int(len(scaffold_members)),
                "support_repeats": int(support_repeats),
                "permutations_per_support": int(permutations_per_support),
                "observed_accuracy": describe_distribution(support_observed),
                "one_sided_p_value_across_supports": describe_distribution(support_p_values),
                "scope": (
                    "one randomly selected ligand per Bemis-Murcko cluster on each support; "
                    "ligand identities are then permuted within that de-redundant support"
                ),
            }

    coverage = experiment_observed.notna().sum(axis=1).to_numpy()
    coverage_sensitivity = {}
    for minimum in [2, 3, 5, 6]:
        keep = coverage >= minimum
        if not keep.any():
            continue
        coverage_reports = {}
        for offset, (name, scores) in enumerate(score_matrices.items()):
            metric = preference_metrics(scores[keep], experiment[keep])
            report = {
                key: value
                for key, value in metric.items()
                if key
                in {
                    "mean_per_ligand_pairwise_accuracy",
                    "pair_weighted_accuracy",
                    "evaluated_ligands",
                    "evaluated_pairs",
                    "predicted_score_ties_half_credit",
                }
            }
            if full_sensitivities:
                report["scaffold_cluster_bootstrap"] = bootstrap_vector_mean(
                    metric["per_ligand_pairwise_accuracy"],
                    bootstrap_repeats,
                    seed + 500 + 30 * minimum + offset,
                    cluster_labels=scaffold_labels[keep],
                )
            coverage_reports[name] = report
        coverage_sensitivity[str(minimum)] = {
            "n_ligands": int(keep.sum()),
            "observed_cells": int(np.isfinite(experiment[keep]).sum()),
            "representations": coverage_reports,
        }

    target_jackknife = {}
    target_jackknife_contrasts = {}
    if full_sensitivities:
        per_target_metrics: dict[str, dict[str, np.ndarray]] = {}
        for column, target in enumerate(experiment_observed.columns):
            masked = experiment.copy()
            masked[:, column] = np.nan
            per_target_metrics[str(target)] = {
                name: preference_metrics(scores, masked)[
                    "per_ligand_pairwise_accuracy"
                ]
                for name, scores in score_matrices.items()
            }
        for name in score_matrices:
            leave_one_out = {
                target: float(np.nanmean(metrics[name]))
                for target, metrics in per_target_metrics.items()
            }
            values = np.asarray(list(leave_one_out.values()), dtype=float)
            target_jackknife[name] = {
                "full_panel": reports[name]["mean_per_ligand_pairwise_accuracy"],
                "leave_one_target_out": leave_one_out,
                "minimum": float(np.nanmin(values)),
                "median": float(np.nanmedian(values)),
                "maximum": float(np.nanmax(values)),
                "most_influential_target": list(leave_one_out)[
                    int(
                        np.nanargmax(
                            np.abs(
                                values
                                - reports[name]["mean_per_ligand_pairwise_accuracy"]
                            )
                        )
                    )
                ],
            }
        for first, second in [
            ("two_way_residual", "absolute_vina"),
            ("two_way_residual", "column_standardized"),
        ]:
            key = f"{first}_minus_{second}"
            leave_one_out = {}
            for target, metrics in per_target_metrics.items():
                difference = metrics[first] - metrics[second]
                leave_one_out[target] = float(np.nanmean(difference))
            values = np.asarray(list(leave_one_out.values()), dtype=float)
            full = comparisons[key]["plugin_mean_difference"]
            mean_loo = float(values.mean())
            jackknife_estimate = float(len(values) * full - (len(values) - 1) * mean_loo)
            standard_error = float(
                np.sqrt((len(values) - 1) / len(values) * np.square(values - mean_loo).sum())
            )
            target_jackknife_contrasts[key] = {
                "full_panel_paired_difference": float(full),
                "leave_one_target_out": leave_one_out,
                "minimum": float(values.min()),
                "median": float(np.median(values)),
                "maximum": float(values.max()),
                "jackknife_bias_corrected_estimate": jackknife_estimate,
                "jackknife_standard_error": standard_error,
                "jackknife_normal_95_interval": [
                    jackknife_estimate - 1.96 * standard_error,
                    jackknife_estimate + 1.96 * standard_error,
                ],
                "most_influential_target": list(leave_one_out)[
                    int(np.argmax(np.abs(values - full)))
                ],
                "scope": (
                    "delete-one-target composition sensitivity over the fixed observed panel; "
                    "the normal interval is a jackknife approximation, not design-based "
                    "inference to a random target superpopulation"
                ),
            }

    observation_mask = experiment_observed.notna().to_numpy(dtype=int)
    pair_support_matrix = observation_mask.T @ observation_mask
    pair_support = pair_support_matrix[np.triu_indices(pair_support_matrix.shape[0], 1)]
    nonzero_pair_support = pair_support[pair_support > 0]
    return {
        "n_ligands": int(len(experiment_observed)),
        "n_targets": int(experiment_observed.shape[1]),
        "observed_experimental_cells": int(np.isfinite(experiment).sum()),
        "observed_fraction": float(np.isfinite(experiment).mean()),
        "evaluated_target_pairs_with_any_ligand": int(len(nonzero_pair_support)),
        "target_pair_coobservations": {
            "minimum_nonzero": int(nonzero_pair_support.min()),
            "median_nonzero": float(np.median(nonzero_pair_support)),
            "maximum": int(nonzero_pair_support.max()),
            "pairs_with_at_least_5_ligands": int(np.sum(pair_support >= 5)),
            "pairs_with_at_least_10_ligands": int(np.sum(pair_support >= 10)),
            "pairs_with_at_least_20_ligands": int(np.sum(pair_support >= 20)),
        },
        "primary_estimand": (
            "mean per-ligand pairwise target-preference accuracy over observed exact-relation "
            "median pChEMBL cells; exact experimental ties excluded"
        ),
        "spectrally_motivated_representation": "two_way_residual",
        "multiplicity_note": (
            "Two-way residual Vina is the representation motivated by the spectral analysis, "
            "not a prospectively registered endpoint. Absolute, centered, scaled and prior "
            "representations are reported to decompose its operational behavior."
        ),
        "representations": reports,
        "paired_comparisons": comparisons,
        "coverage_sensitivity": coverage_sensitivity,
        "target_jackknife": target_jackknife,
        "target_jackknife_paired_contrasts": target_jackknife_contrasts,
        "scaffold_clusters": int(len(np.unique(scaffold_labels))),
        "uncertainty_unit": (
            "ligand and Bemis-Murcko scaffold-cluster bootstrap; target deletion is a "
            "composition sensitivity rather than a confidence interval"
        ),
    }


def same_endpoint_pair_sensitivity(
    score_matrices: dict[str, np.ndarray],
    endpoint_experiments: dict[str, pd.DataFrame],
    smiles: pd.Series,
    butina_labels: np.ndarray,
    bootstrap_repeats: int,
    seed: int,
) -> dict:
    """Aggregate only within-ligand target pairs measured with the same endpoint type.

    The ligand cohort, target columns and docking transformations remain fixed to the broad
    operational benchmark.  Endpoint-specific per-ligand accuracies are first computed for
    Ki, Kd, IC50 and EC50 separately and then averaged with equal endpoint weight within each
    ligand, followed by equal ligand weight.
    """
    representation_names = [
        "absolute_vina",
        "column_standardized",
        "two_way_residual",
    ]
    scaffold_labels = scaffold_keys(smiles)
    endpoint_reports: dict[str, dict] = {}
    per_representation: dict[str, list[np.ndarray]] = {
        name: [] for name in representation_names
    }
    for endpoint, frame in endpoint_experiments.items():
        experiment = frame.to_numpy(dtype=float)
        endpoint_report = {
            "observed_cells": int(np.isfinite(experiment).sum()),
            "ligands_with_at_least_two_endpoint_matched_targets": int(
                (np.isfinite(experiment).sum(axis=1) >= 2).sum()
            ),
            "targets_with_any_observation": int(
                np.isfinite(experiment).any(axis=0).sum()
            ),
            "representations": {},
        }
        for name in representation_names:
            metric = preference_metrics(score_matrices[name], experiment)
            per_representation[name].append(
                metric["per_ligand_pairwise_accuracy"]
            )
            endpoint_report["representations"][name] = {
                key: value
                for key, value in metric.items()
                if key
                in {
                    "mean_per_ligand_pairwise_accuracy",
                    "pair_weighted_accuracy",
                    "evaluated_ligands",
                    "evaluated_pairs",
                    "excluded_ties",
                    "predicted_score_ties_half_credit",
                }
            }
        endpoint_reports[endpoint] = endpoint_report

    pooled_vectors: dict[str, np.ndarray] = {}
    representations: dict[str, dict] = {}
    for offset, name in enumerate(representation_names):
        values = np.column_stack(per_representation[name])
        finite = np.isfinite(values)
        endpoint_count = finite.sum(axis=1)
        pooled = np.full(len(values), np.nan)
        keep = endpoint_count > 0
        pooled[keep] = np.nansum(values[keep], axis=1) / endpoint_count[keep]
        pooled_vectors[name] = pooled
        representations[name] = {
            "mean_per_ligand_pairwise_accuracy": float(np.nanmean(pooled)),
            "evaluated_ligands": int(np.isfinite(pooled).sum()),
            "ligand_bootstrap": bootstrap_vector_mean(
                pooled, bootstrap_repeats, seed + 10 + offset
            ),
            "murcko_scaffold_cluster_bootstrap": bootstrap_vector_mean(
                pooled,
                bootstrap_repeats,
                seed + 20 + offset,
                cluster_labels=scaffold_labels,
            ),
            "butina_cluster_bootstrap": bootstrap_vector_mean(
                pooled,
                bootstrap_repeats,
                seed + 30 + offset,
                cluster_labels=butina_labels,
            ),
        }
    paired = {
        "two_way_residual_minus_absolute_vina": paired_preference_comparison(
            pooled_vectors["two_way_residual"],
            pooled_vectors["absolute_vina"],
            bootstrap_repeats,
            seed + 100,
            scaffold_labels,
            butina_labels=butina_labels,
        ),
        "two_way_residual_minus_column_standardized": paired_preference_comparison(
            pooled_vectors["two_way_residual"],
            pooled_vectors["column_standardized"],
            bootstrap_repeats,
            seed + 110,
            scaffold_labels,
            butina_labels=butina_labels,
        ),
    }
    total_instances = sum(
        report["representations"]["absolute_vina"]["evaluated_pairs"]
        for report in endpoint_reports.values()
    )
    duplicate_collapsed_vectors: dict[str, np.ndarray] = {}
    unique_pair_counts = np.zeros(len(smiles), dtype=int)
    for name in representation_names:
        scores = score_matrices[name]
        per_ligand = np.full(len(smiles), np.nan)
        for ligand in range(len(smiles)):
            pair_values: dict[tuple[int, int], list[float]] = {}
            for frame in endpoint_experiments.values():
                values = frame.iloc[ligand].to_numpy(dtype=float)
                observed = np.flatnonzero(np.isfinite(values))
                for first_position in range(len(observed)):
                    for second_position in range(first_position + 1, len(observed)):
                        first = int(observed[first_position])
                        second = int(observed[second_position])
                        truth = np.sign(values[first] - values[second])
                        if truth == 0:
                            continue
                        score_difference = scores[ligand, first] - scores[ligand, second]
                        correct = (
                            0.5
                            if score_difference == 0
                            else float(-np.sign(score_difference) == truth)
                        )
                        pair_values.setdefault((first, second), []).append(correct)
            if pair_values:
                per_ligand[ligand] = float(
                    np.mean([np.mean(values) for values in pair_values.values()])
                )
                if name == representation_names[0]:
                    unique_pair_counts[ligand] = len(pair_values)
        duplicate_collapsed_vectors[name] = per_ligand

    duplicate_collapsed = {
        "estimand": (
            "for each ligand and unordered target pair, correctness is averaged across "
            "eligible identical-endpoint strata before equal-pair and equal-ligand averaging"
        ),
        "evaluated_ligands": int(np.sum(unique_pair_counts > 0)),
        "unique_ligand_target_pair_instances": int(unique_pair_counts.sum()),
        "representations": {},
    }
    for offset, name in enumerate(representation_names):
        vector = duplicate_collapsed_vectors[name]
        duplicate_collapsed["representations"][name] = {
            "mean_per_ligand_pairwise_accuracy": float(np.nanmean(vector)),
            "ligand_bootstrap": bootstrap_vector_mean(
                vector, bootstrap_repeats, seed + 210 + offset
            ),
            "murcko_scaffold_cluster_bootstrap": bootstrap_vector_mean(
                vector,
                bootstrap_repeats,
                seed + 220 + offset,
                cluster_labels=scaffold_labels,
            ),
            "butina_cluster_bootstrap": bootstrap_vector_mean(
                vector,
                bootstrap_repeats,
                seed + 230 + offset,
                cluster_labels=butina_labels,
            ),
        }
    duplicate_collapsed["paired_comparisons"] = {
        "two_way_residual_minus_absolute_vina": paired_preference_comparison(
            duplicate_collapsed_vectors["two_way_residual"],
            duplicate_collapsed_vectors["absolute_vina"],
            bootstrap_repeats,
            seed + 240,
            scaffold_labels,
            butina_labels=butina_labels,
        ),
        "two_way_residual_minus_column_standardized": paired_preference_comparison(
            duplicate_collapsed_vectors["two_way_residual"],
            duplicate_collapsed_vectors["column_standardized"],
            bootstrap_repeats,
            seed + 250,
            scaffold_labels,
            butina_labels=butina_labels,
        ),
    }
    return {
        "estimand": (
            "equal-endpoint mean within ligand followed by equal-ligand mean; every target "
            "comparison uses two cells with the identical ChEMBL standard_type"
        ),
        "fixed_support": {
            "ligands": int(len(smiles)),
            "targets": int(next(iter(endpoint_experiments.values())).shape[1]),
            "endpoint_types": list(endpoint_experiments),
            "endpoint_specific_pair_instances": int(total_instances),
        },
        "representations": representations,
        "paired_comparisons": paired,
        "by_endpoint": endpoint_reports,
        "duplicate_collapsed_unique_target_pairs": duplicate_collapsed,
        "boundary": (
            "the analysis avoids cross-endpoint target comparisons but remains observational "
            "and can count the same target pair once in more than one endpoint type"
        ),
    }


def expanded_target_preference_benchmark(
    canonical: pd.DataFrame,
    docking_reference: pd.DataFrame,
    bootstrap_repeats: int = 5000,
    permutation_repeats: int = 5000,
    seed: int = SEED,
) -> dict:
    """Build the strict Docking-44 x ChEMBL observed-pair operational benchmark."""
    analysis = Path("negative_results_paper/analysis")
    required = [
        "canonical_smiles",
        "pchembl_value",
        "standard_relation",
        "standard_type",
        "panel_key",
        "target_organism",
        "assay_type",
    ]
    activities = pd.read_csv(source_path(analysis / "chembl_activities_full.csv"), usecols=required)
    activities = activities[
        activities.panel_key.isin(DOCK44)
        & activities.pchembl_value.notna()
        & activities.standard_relation.eq("=")
        & activities.standard_type.isin(["Ki", "Kd", "IC50", "EC50"])
    ].copy()
    activities["inchikey"] = full_inchikeys(activities.canonical_smiles)
    activities = activities[activities.inchikey.notna()].copy()

    docking_smiles = canonical.set_index("inchikey")["analysis_smiles"]
    docking_butina = canonical.set_index("inchikey")["Butina_clusters"]
    if not docking_smiles.index.is_unique or not docking_reference.index.is_unique:
        raise ValueError("Docking-44 full InChIKeys are not unique")

    def prepare_variant(
        frame: pd.DataFrame,
        variant_seed: int,
        full_sensitivities: bool,
    ) -> tuple[dict, pd.DataFrame, dict[str, np.ndarray]]:
        experimental_reference = frame.pivot_table(
            index="inchikey",
            columns="panel_key",
            values="pchembl_value",
            aggfunc="median",
        ).reindex(columns=DOCK44)
        matched_any = experimental_reference.index.intersection(docking_reference.index)
        matched = experimental_reference.loc[matched_any]
        matched_coverage = matched.notna().sum(axis=1)
        evaluation_ids = matched.index[matched_coverage >= 2]
        experiment = matched.loc[evaluation_ids]
        evaluation_targets = experiment.columns[experiment.notna().any(axis=0)].tolist()
        experiment = experiment[evaluation_targets]
        score_matrices, fit_support = external_reference_score_representations(
            docking_reference,
            evaluation_ids,
            evaluation_targets,
            experimental_reference,
        )
        experiment_values = experiment.to_numpy(dtype=float)
        cohort_scaffolds = scaffold_keys(docking_smiles.reindex(experiment.index))
        evaluation_butina = docking_butina.reindex(experiment.index).to_numpy()
        global_prior = score_matrices["experimental_target_prior"][0]
        cohort_prior = np.empty_like(experiment_values)
        for held_out in range(len(experiment_values)):
            training = cohort_scaffolds != cohort_scaffolds[held_out]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                mean = np.nanmean(experiment_values[training], axis=0)
            cohort_prior[held_out] = np.where(
                np.isfinite(mean), -mean, global_prior
            )
        score_matrices["cohort_experimental_target_prior"] = cohort_prior
        benchmark = expanded_preference_metrics(
            score_matrices,
            experiment,
            docking_smiles,
            bootstrap_repeats=(bootstrap_repeats if full_sensitivities else 2000),
            permutation_repeats=(permutation_repeats if full_sensitivities else 1000),
            seed=variant_seed,
            full_sensitivities=full_sensitivities,
            butina_labels=evaluation_butina,
        )
        benchmark["fit_support"] = fit_support
        benchmark["fit_support"]["cohort_experimental_target_prior"] = (
            "leave-one-Bemis-Murcko-scaffold-cluster-out target mean within the broadly "
            "profiled cohort; targets with no remaining value fall back to the external "
            "ChEMBL prior"
        )
        benchmark["matched_support_before_minimum_two_target_filter"] = {
            "n_ligands": int(len(matched)),
            "n_targets_with_any_observation": int(matched.notna().any(axis=0).sum()),
            "observed_cells": int(matched.notna().sum().sum()),
            "observed_fraction_over_nonempty_targets": float(
                matched.notna().sum().sum()
                / (len(matched) * matched.notna().any(axis=0).sum())
            ),
            "ligands_by_minimum_observed_targets": {
                str(minimum): int((matched_coverage >= minimum).sum())
                for minimum in [1, 2, 3, 4, 5, 6, 8, 10]
            },
        }
        return benchmark, experiment, score_matrices

    primary, primary_experiment, primary_scores = prepare_variant(
        activities, seed, True
    )
    human_binding_frame = activities[
        activities.target_organism.eq("Homo sapiens")
        & activities.assay_type.eq("B")
    ]
    human_binding, _, _ = prepare_variant(
        human_binding_frame, seed + 1000, False
    )
    human_kikd_frame = human_binding_frame[
        human_binding_frame.standard_type.isin(["Ki", "Kd"])
    ]
    human_binding_kikd, _, _ = prepare_variant(
        human_kikd_frame, seed + 2000, True
    )

    endpoint_experiments = {
        endpoint: activities[activities.standard_type.eq(endpoint)]
        .pivot_table(
            index="inchikey",
            columns="panel_key",
            values="pchembl_value",
            aggfunc="median",
        )
        .reindex(index=primary_experiment.index, columns=primary_experiment.columns)
        for endpoint in ["Ki", "Kd", "IC50", "EC50"]
    }
    same_endpoint = same_endpoint_pair_sensitivity(
        primary_scores,
        endpoint_experiments,
        docking_smiles.reindex(primary_experiment.index),
        docking_butina.reindex(primary_experiment.index).to_numpy(),
        bootstrap_repeats=bootstrap_repeats,
        seed=seed + 3000,
    )

    primary_ids = primary_experiment.index
    cell_groups = activities[
        activities.inchikey.isin(primary_ids)
        & activities.panel_key.isin(primary_experiment.columns)
    ].groupby(["inchikey", "panel_key"])["pchembl_value"]
    cell_range = (cell_groups.max() - cell_groups.min()).unstack("panel_key").reindex(
        index=primary_ids, columns=primary_experiment.columns
    )
    cell_count = cell_groups.size().unstack("panel_key").reindex(
        index=primary_ids, columns=primary_experiment.columns
    )
    low_dispersion_experiment = primary_experiment.mask(cell_range > 1.0)
    low_dispersion_keep = low_dispersion_experiment.notna().sum(axis=1) >= 2
    dispersion_metrics = {
        name: {
            key: value
            for key, value in preference_metrics(
                scores[low_dispersion_keep.to_numpy()],
                low_dispersion_experiment.loc[low_dispersion_keep].to_numpy(dtype=float),
            ).items()
            if key
            in {
                "mean_per_ligand_pairwise_accuracy",
                "pair_weighted_accuracy",
                "evaluated_ligands",
                "evaluated_pairs",
            }
        }
        for name, scores in primary_scores.items()
    }

    experimental_all = activities.pivot_table(
        index="inchikey",
        columns="panel_key",
        values="pchembl_value",
        aggfunc="median",
    ).reindex(columns=DOCK44)
    experimental_coverage = experimental_all.notna().sum(axis=1)
    connectivity_experiment = experimental_all.copy()
    connectivity_experiment.index = connectivity_experiment.index.str.split("-").str[0]
    connectivity_experiment = connectivity_experiment.groupby(level=0).median()
    connectivity_docking = docking_reference.copy()
    connectivity_docking.index = connectivity_docking.index.str.split("-").str[0]
    connectivity_docking = connectivity_docking.groupby(level=0).median()
    connectivity_common = connectivity_experiment.index.intersection(
        connectivity_docking.index
    )
    connectivity_coverage = connectivity_experiment.loc[connectivity_common].notna().sum(axis=1)

    return {
        "matching_rule": (
            "RDKit full InChIKey from Docking-44 analysis SMILES and ChEMBL canonical "
            "SMILES; exact standard relation; Ki, Kd, IC50 or EC50; median per cell"
        ),
        "all_exact_activity_support": {
            "source_records": int(len(activities)),
            "unique_ligands": int(len(experimental_all)),
            "observed_cells": int(experimental_all.notna().sum().sum()),
            "targets_with_data": int(experimental_all.notna().any(axis=0).sum()),
            "ligands_by_minimum_observed_targets": {
                str(minimum): int((experimental_coverage >= minimum).sum())
                for minimum in [1, 2, 3, 4, 5, 6, 8, 10]
            },
        },
        "connectivity_key_sensitivity": {
            "matched_ligands_with_any_activity": int(len(connectivity_common)),
            "matched_ligands_with_at_least_two_targets": int(
                (connectivity_coverage >= 2).sum()
            ),
            "not_primary_because": (
                "the connectivity block ignores stereochemical and protonation layers of "
                "the full InChIKey"
            ),
        },
        "primary_all_exact": primary,
        "co_primary_operational_estimands": {
            "status": (
                "declared during revision rather than prospectively registered; both are "
                "reported without selecting between them by outcome"
            ),
            "broad_coverage": "primary_all_exact",
            "assay_restricted": "assay_sensitivities/human_binding_Ki_Kd",
        },
        "assay_sensitivities": {
            "human_binding_all_endpoints": human_binding,
            "human_binding_Ki_Kd": human_binding_kikd,
            "same_endpoint_pairs_on_fixed_primary_support": same_endpoint,
        },
        "within_cell_dispersion_sensitivity": {
            "cells_with_multiple_source_records": int((cell_count > 1).sum().sum()),
            "cells_with_range_above_1_pchembl": int((cell_range > 1.0).sum().sum()),
            "ligands_retained_with_at_least_two_cells": int(low_dispersion_keep.sum()),
            "representations": dispersion_metrics,
        },
        "boundary": (
            "The sparse observed-pair benchmark tests ordinal target preferences but does "
            "not support a second experimental eigenspectrum or unmeasured-target retrieval claim."
        ),
    }


def experimental_sensitivity(dmat: pd.DataFrame, emat_observed: pd.DataFrame) -> dict:
    """Collect frozen ChEMBL controls and run a reproducibility-calibrated noise test."""
    analysis = Path("negative_results_paper/analysis")
    heterogeneity = read_json(source_path(analysis / "review_assay_heterogeneity_summary.json"))
    imputation = read_json(source_path(analysis / "exp_imputation_robustness_summary.json"))
    imputation["interpretation"] = (
        "The experimental estimate remains above the docking reference under all tested "
        "missing-data models, but its magnitude is model-dependent: 2.644 with low-rank "
        "SoftImpute and 5.721--8.592 with KNN, PPCA or multiple imputation. The fully "
        "observed 31-by-6 sub-block has participation rank 3.608."
    )
    provenance = read_json(source_path(analysis / "dd_chembl_provenance_summary.json"))
    shrinkage = read_json(source_path(analysis / "exp_ledoitwolf_summary.json"))

    sigma = float(provenance["sigma_independent"])
    exp_sd = emat_observed.std(axis=0, ddof=1)
    standardized_noise = sigma / exp_sd
    z_dock = (dmat - dmat.mean(axis=0)) / dmat.std(axis=0, ddof=1)
    rng = np.random.default_rng(SEED)
    noisy_rank = []
    for _ in range(2000):
        noisy = z_dock.to_numpy(dtype=np.float64) + rng.normal(
            0.0, standardized_noise.to_numpy(dtype=np.float64), size=z_dock.shape
        )
        noisy_rank.append(rank_summary(noisy)["participation_ratio"])
    noise_sweep = {}
    for multiplier in [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
        ranks = []
        for _ in range(100):
            noisy = z_dock.to_numpy(dtype=np.float64) + rng.normal(
                0.0,
                multiplier * standardized_noise.to_numpy(dtype=np.float64),
                size=z_dock.shape,
            )
            ranks.append(rank_summary(noisy)["participation_ratio"])
        noise_sweep[str(multiplier)] = {
            "mean_participation_ratio": float(np.mean(ranks)),
            "median_participation_ratio": float(np.median(ranks)),
        }

    targets = list(dmat.columns)
    activities = pd.read_csv(source_path(analysis / "chembl_activities_full.csv"))
    activities = activities[
        activities.panel_key.isin(targets)
        & activities.pchembl_value.notna()
        & activities.standard_relation.eq("=")
        & activities.standard_type.isin(["Ki", "Kd", "IC50", "EC50"])
    ].copy()
    unique_smiles = activities.canonical_smiles.dropna().unique()
    smiles_to_key = {}
    for smiles in unique_smiles:
        molecule = Chem.MolFromSmiles(str(smiles))
        smiles_to_key[smiles] = Chem.MolToInchiKey(molecule) if molecule is not None else None
    activities["inchikey"] = activities.canonical_smiles.map(smiles_to_key)

    def curated_matched_block(frame: pd.DataFrame) -> dict:
        matrix = frame.pivot_table(
            index="inchikey", columns="panel_key", values="pchembl_value", aggfunc="median"
        ).reindex(columns=targets)
        common = dmat.index.intersection(matrix[matrix.notna().sum(axis=1) >= 5].index)
        observed = matrix.loc[common]
        filled = observed.fillna(observed.median())
        experimental_ladder = centering_ladder(filled.to_numpy(dtype=np.float64))
        docking_ladder = centering_ladder(dmat.loc[common].to_numpy(dtype=np.float64))
        return {
            "n_ligands": int(len(common)),
            "n_targets": int(len(targets)),
            "missing_fraction": float(observed.isna().to_numpy().mean()),
            "experimental_participation_ratio": experimental_ladder["raw"]["participation_ratio"],
            "experimental_entropy_rank": experimental_ladder["raw"]["entropy_rank"],
            "experimental_residual_participation_ratio": experimental_ladder["interaction"]["participation_ratio"],
            "matched_docking_participation_ratio": docking_ladder["raw"]["participation_ratio"],
            "matched_docking_entropy_rank": docking_ladder["raw"]["entropy_rank"],
            "matched_docking_residual_participation_ratio": docking_ladder["interaction"]["participation_ratio"],
        }

    curated_matched = {
        "selection": (
            "exact standard relation; Ki/Kd/IC50/EC50; full InChIKey; median over repeated "
            "compound-target activities; at least five of six targets"
        ),
        "all_exact_median": curated_matched_block(activities),
        "human_binding_all_endpoints": curated_matched_block(
            activities[
                activities.target_organism.eq("Homo sapiens")
                & activities.assay_type.eq("B")
            ]
        ),
        "human_binding_Ki_Kd": curated_matched_block(
            activities[
                activities.target_organism.eq("Homo sapiens")
                & activities.assay_type.eq("B")
                & activities.standard_type.isin(["Ki", "Kd"])
            ]
        ),
    }

    smiles = pd.read_csv(source_path(analysis / "exp_positive_control_smiles.csv")).set_index("inchikey")
    smiles = smiles.reindex(dmat.index)
    descriptor_rows = []
    for value in smiles.smiles:
        molecule = Chem.MolFromSmiles(str(value))
        descriptor_rows.append([
            molecule.GetNumHeavyAtoms(),
            Descriptors.MolWt(molecule),
            rdMolDescriptors.CalcLabuteASA(molecule),
            Descriptors.TPSA(molecule),
            Crippen.MolLogP(molecule),
            Descriptors.NumRotatableBonds(molecule),
            rdMolDescriptors.CalcNumRings(molecule),
        ])
    descriptor_matrix = np.asarray(descriptor_rows, dtype=np.float64)
    descriptor_matrix = (
        descriptor_matrix - descriptor_matrix.mean(axis=0)
    ) / descriptor_matrix.std(axis=0, ddof=1)
    dock_values = dmat.to_numpy(dtype=np.float64)
    linear_residual = np.empty_like(dock_values)
    nonlinear_residual = np.empty_like(dock_values)
    folds = KFold(5, shuffle=True, random_state=SEED)
    for column in range(dock_values.shape[1]):
        linear = LinearRegression().fit(descriptor_matrix, dock_values[:, column])
        linear_residual[:, column] = (
            dock_values[:, column] - linear.predict(descriptor_matrix)
        )
        predictions = np.zeros(len(dock_values), dtype=np.float64)
        for train, test in folds.split(descriptor_matrix):
            model = HistGradientBoostingRegressor(
                max_iter=200, learning_rate=0.05, max_depth=4, random_state=SEED
            ).fit(descriptor_matrix[train], dock_values[train, column])
            predictions[test] = model.predict(descriptor_matrix[test])
        nonlinear_residual[:, column] = dock_values[:, column] - predictions

    matched_physchem = {
        "n_ligands": int(len(dmat)),
        "n_targets": int(dmat.shape[1]),
        "descriptors": [
            "heavy atoms", "molecular weight", "Labute ASA", "TPSA", "cLogP",
            "rotatable bonds", "ring count",
        ],
        "raw_docking_participation_ratio": rank_summary(dock_values)["participation_ratio"],
        "linear_residual_participation_ratio": rank_summary(linear_residual)["participation_ratio"],
        "cross_fitted_histgbm_residual_participation_ratio": rank_summary(
            nonlinear_residual
        )["participation_ratio"],
        "cross_fitting": "five-fold shuffled KFold, seed 0; one HistGBM per target",
    }

    return {
        "curated_assay_blocks": heterogeneity,
        "missing_data_estimators": imputation,
        "ledoit_wolf_sensitivity": shrinkage,
        "independent_between_document_repeatability": provenance["noise_ladder"]["different documents"],
        "reproducibility_calibrated_noise_injection_74x6": {
            "measurement_sigma_pchembl": sigma,
            "interpretation_of_sigma": (
                "Estimated from repeated compound-target records reported in different documents; "
                "it includes inter-laboratory and condition variability and is therefore a conservative "
                "measurement-noise scale."
            ),
            "experimental_target_sd_pchembl": {
                str(k): float(v) for k, v in exp_sd.items()
            },
            "noise_sd_as_fraction_of_experimental_target_sd": {
                str(k): float(v) for k, v in standardized_noise.items()
            },
            "median_standardized_noise_sd": float(standardized_noise.median()),
            "docking_rank_without_noise": rank_summary(dmat)["participation_ratio"],
            "docking_rank_with_noise_mean": float(np.mean(noisy_rank)),
            "docking_rank_with_noise_median": float(np.median(noisy_rank)),
            "docking_rank_with_noise_95_interval": [
                float(np.quantile(noisy_rank, 0.025)),
                float(np.quantile(noisy_rank, 0.975)),
            ],
            "n_simulations": len(noisy_rank),
            "calibrated_noise_multiplier_sweep": noise_sweep,
        },
        "activity_level_curated_matched_blocks": curated_matched,
        "matched_physicochemical_residualization": matched_physchem,
    }


DTI_PROVENANCE = {
    "boltz_s3": {
        "display_name": "Boltz-2 deployed affinity head",
        "representation_and_head": "co-folded complex; deployed affinity score",
        "training_data": "foundation-model training; no DAVIS fitting in this analysis",
        "evaluation_regime": "external zero-shot DAVIS",
        "oof_status": "not applicable (no DAVIS fitting)",
        "marker_regime": "external",
        "caveat": "foundation-model training corpus may contain related structures or affinities",
    },
    "s1_readout": {
        "display_name": "Boltz-2 frozen representation + ridge",
        "representation_and_head": "frozen pre-pairformer representation; RidgeCV",
        "training_data": "DAVIS training folds",
        "evaluation_regime": "six-fold unseen-target GroupKFold",
        "oof_status": "out-of-fold by target",
        "marker_regime": "davis_oof",
        "caveat": "shares Boltz-2 representation with boltz_s3",
    },
    "chembert_davisoof": {
        "display_name": "ChemBERTa + AAC, DAVIS OOF",
        "representation_and_head": "ChemBERTa drug embedding + amino-acid composition; MLP",
        "training_data": "DAVIS training folds",
        "evaluation_regime": "six-fold unseen-target GroupKFold",
        "oof_status": "out-of-fold by target",
        "marker_regime": "davis_oof",
        "caveat": "paired with chembert_kiba2davis",
    },
    "chembert_kiba2davis": {
        "display_name": "ChemBERTa + AAC, KIBA to DAVIS",
        "representation_and_head": "ChemBERTa drug embedding + amino-acid composition; MLP",
        "training_data": "KIBA",
        "evaluation_regime": "zero-shot transfer to DAVIS",
        "oof_status": "no DAVIS fitting",
        "marker_regime": "kiba_transfer",
        "caveat": "paired with chembert_davisoof",
    },
    "conplex_kiba2davis": {
        "display_name": "ConPLex",
        "representation_and_head": "protein language model + contrastive DTI score",
        "training_data": "pretrained on BindingDB",
        "evaluation_regime": "zero-shot DAVIS inference",
        "oof_status": "no DAVIS fitting",
        "marker_regime": "external",
        "caveat": "possible training-data overlap with DAVIS proteins",
    },
    "esm_target_davisoof": {
        "display_name": "Morgan + ESM-2, DAVIS OOF",
        "representation_and_head": "Morgan drug features + ESM-2 target embedding; MLP",
        "training_data": "DAVIS training folds",
        "evaluation_regime": "six-fold unseen-target GroupKFold",
        "oof_status": "out-of-fold by target",
        "marker_regime": "davis_oof",
        "caveat": "paired with esm_target_kiba2davis",
    },
    "esm_target_kiba2davis": {
        "display_name": "Morgan + ESM-2, KIBA to DAVIS",
        "representation_and_head": "Morgan drug features + ESM-2 target embedding; MLP",
        "training_data": "KIBA",
        "evaluation_regime": "zero-shot transfer to DAVIS",
        "oof_status": "no DAVIS fitting",
        "marker_regime": "kiba_transfer",
        "caveat": "paired with esm_target_davisoof",
    },
    "histgbm_davisoof": {
        "display_name": "Morgan + AAC gradient boosting",
        "representation_and_head": "Morgan + physicochemical drug features and amino-acid composition; HistGBM",
        "training_data": "DAVIS training folds",
        "evaluation_regime": "six-fold unseen-target GroupKFold",
        "oof_status": "out-of-fold by target",
        "marker_regime": "davis_oof",
        "caveat": "feature family overlaps several comparator arms",
    },
    "mlp_davisoof": {
        "display_name": "Morgan + AAC MLP, DAVIS OOF",
        "representation_and_head": "Morgan + physicochemical drug features and amino-acid composition; MLP",
        "training_data": "DAVIS training folds",
        "evaluation_regime": "six-fold unseen-target GroupKFold",
        "oof_status": "out-of-fold by target",
        "marker_regime": "davis_oof",
        "caveat": "paired with mlp_kiba2davis",
    },
    "mlp_kiba2davis": {
        "display_name": "Morgan + AAC MLP, KIBA to DAVIS",
        "representation_and_head": "Morgan + physicochemical drug features and amino-acid composition; MLP",
        "training_data": "KIBA",
        "evaluation_regime": "zero-shot transfer to DAVIS",
        "oof_status": "no DAVIS fitting",
        "marker_regime": "kiba_transfer",
        "caveat": "paired with mlp_davisoof",
    },
    "rf_davisoof": {
        "display_name": "Morgan + AAC random forest, DAVIS OOF",
        "representation_and_head": "Morgan + physicochemical drug features and amino-acid composition; random forest",
        "training_data": "DAVIS training folds",
        "evaluation_regime": "six-fold unseen-target GroupKFold",
        "oof_status": "out-of-fold by target",
        "marker_regime": "davis_oof",
        "caveat": "paired with rf_kiba2davis",
    },
    "rf_kiba2davis": {
        "display_name": "Morgan + AAC random forest, KIBA to DAVIS",
        "representation_and_head": "Morgan + physicochemical drug features and amino-acid composition; random forest",
        "training_data": "KIBA",
        "evaluation_regime": "zero-shot transfer to DAVIS",
        "oof_status": "no DAVIS fitting",
        "marker_regime": "kiba_transfer",
        "caveat": "paired with rf_davisoof",
    },
}


def dti_associations(leaderboard: dict) -> dict:
    primary = leaderboard["headline_reorder"]["primary"]["arms_ranked"]
    all_scorers = leaderboard["headline_reorder"]["sensitivity_all_scorers"]["arms_ranked"]
    arms = leaderboard["arms"]
    frame = pd.DataFrame([
        {
            "arm": name,
            "kind": arms[name]["kind"],
            **DTI_PROVENANCE[name],
            "affinity_pearson": arms[name]["affinity_pearson"],
            "raw_effective_rank": arms[name]["raw_effrank"],
            "interaction_effective_rank": arms[name]["interaction_effrank"],
            "selectivity_p_at_5": arms[name]["prec_at_5"]["mean"],
            "selectivity_p_at_5_ci_low": arms[name]["prec_at_5"]["ci95"][0],
            "selectivity_p_at_5_ci_high": arms[name]["prec_at_5"]["ci95"][1],
            "selectivity_auprc": arms[name]["auprc"]["mean"],
        }
        for name in primary
    ])
    frame.to_csv(OUT / "dti_primary_arms.csv", index=False)
    frame.to_csv(OUT / "dti_arm_provenance.csv", index=False)
    all_frame = pd.DataFrame([
        {
            "arm": name,
            "affinity_pearson": arms[name]["affinity_pearson"],
            "raw_effective_rank": arms[name]["raw_effrank"],
            "interaction_effective_rank": arms[name]["interaction_effrank"],
            "selectivity_p_at_5": arms[name]["prec_at_5"]["mean"],
            "selectivity_auprc": arms[name]["auprc"]["mean"],
            "clears_chance": name in primary,
        }
        for name in all_scorers
    ])
    all_frame.to_csv(OUT / "dti_all_scorers_sensitivity.csv", index=False)

    def corr(table: pd.DataFrame, a: str, b: str) -> dict:
        pearson = stats.pearsonr(table[a], table[b])
        spearman = stats.spearmanr(table[a], table[b])
        kendall = stats.kendalltau(table[a], table[b])
        return {
            "pearson_r": float(pearson.statistic),
            "pearson_p": float(pearson.pvalue),
            "spearman_rho": float(spearman.statistic),
            "spearman_p": float(spearman.pvalue),
            "kendall_tau": float(kendall.statistic),
            "kendall_p": float(kendall.pvalue),
        }

    raw = frame["raw_effective_rank"].to_numpy()
    interaction = frame["interaction_effective_rank"].to_numpy()
    selectivity = frame["selectivity_p_at_5"].to_numpy()
    affinity = frame["affinity_pearson"].to_numpy()

    loo = []
    for i, arm in enumerate(frame["arm"]):
        keep = np.arange(len(frame)) != i
        loo.append({
            "left_out": arm,
            "raw_rank_vs_p_at_5_spearman": float(stats.spearmanr(raw[keep], selectivity[keep]).statistic),
        })

    rng = np.random.default_rng(SEED)
    boot_rho = []
    for _ in range(10000):
        index = rng.integers(0, len(frame), len(frame))
        value = stats.spearmanr(raw[index], selectivity[index]).statistic
        if np.isfinite(value):
            boot_rho.append(value)
    critical_t = stats.t.ppf(0.975, len(frame) - 2)
    detectable_r = float(np.sqrt(critical_t**2 / (critical_t**2 + len(frame) - 2)))

    all_raw = all_frame["raw_effective_rank"].to_numpy()
    all_interaction = all_frame["interaction_effective_rank"].to_numpy()
    all_selectivity = all_frame["selectivity_p_at_5"].to_numpy()
    all_affinity = all_frame["affinity_pearson"].to_numpy()

    davis = pd.read_csv(source_path("davis_complete.tab"), sep="\t")
    measured_per_ligand = (
        davis.assign(measured=davis["y"] > 5.0)
        .groupby("drug_name", sort=False)["measured"]
        .sum()
    )
    eligible_counts = measured_per_ligand[measured_per_ligand >= 12].to_numpy(dtype=int)
    relevant_counts = np.ceil(0.10 * eligible_counts).astype(int)
    p_at_5_ceiling = np.minimum(relevant_counts, 5) / 5

    return {
        "primary_exploratory_panel": "all 20 scorer arms",
        "n_all_arms": int(len(all_frame)),
        "n_outcome_restricted_sensitivity_arms": int(len(frame)),
        "n_primary_arms": len(frame),  # legacy field name retained for machine compatibility
        "raw_rank_vs_selectivity_p_at_5": corr(frame, "raw_effective_rank", "selectivity_p_at_5"),
        "interaction_rank_vs_selectivity_p_at_5": corr(frame, "interaction_effective_rank", "selectivity_p_at_5"),
        "raw_rank_vs_selectivity_auprc": corr(frame, "raw_effective_rank", "selectivity_auprc"),
        "interaction_rank_vs_selectivity_auprc": corr(
            frame, "interaction_effective_rank", "selectivity_auprc"
        ),
        "affinity_vs_selectivity_p_at_5": corr(frame, "affinity_pearson", "selectivity_p_at_5"),
        "raw_rank_vs_affinity": corr(frame, "raw_effective_rank", "affinity_pearson"),
        "partial_spearman_raw_rank_vs_p_at_5_given_affinity": residualized_rank_correlation(
            raw, selectivity, affinity
        ),
        "partial_spearman_interaction_rank_vs_p_at_5_given_affinity": residualized_rank_correlation(
            interaction, selectivity, affinity
        ),
        "leave_one_arm_out": loo,
        "primary_raw_rank_spearman_bootstrap_95_interval": [
            float(np.quantile(boot_rho, 0.025)),
            float(np.quantile(boot_rho, 0.975)),
        ],
        "primary_raw_rank_leave_one_out_range": [
            float(min(item["raw_rank_vs_p_at_5_spearman"] for item in loo)),
            float(max(item["raw_rank_vs_p_at_5_spearman"] for item in loo)),
        ],
        "approximate_two_sided_alpha_0.05_detectable_absolute_correlation": detectable_r,
        "all_20_scorers_sensitivity": {
            "n_arms": len(all_frame),
            "n_at_chance": int((~all_frame.clears_chance).sum()),
            "raw_rank_vs_selectivity_p_at_5": corr(
                all_frame, "raw_effective_rank", "selectivity_p_at_5"
            ),
            "interaction_rank_vs_selectivity_p_at_5": corr(
                all_frame, "interaction_effective_rank", "selectivity_p_at_5"
            ),
            "raw_rank_vs_selectivity_auprc": corr(
                all_frame, "raw_effective_rank", "selectivity_auprc"
            ),
            "interaction_rank_vs_selectivity_auprc": corr(
                all_frame, "interaction_effective_rank", "selectivity_auprc"
            ),
            "affinity_vs_selectivity_p_at_5": corr(
                all_frame, "affinity_pearson", "selectivity_p_at_5"
            ),
            "partial_spearman_raw_rank_vs_p_at_5_given_affinity":
                residualized_rank_correlation(all_raw, all_selectivity, all_affinity),
            "partial_spearman_interaction_rank_vs_p_at_5_given_affinity":
                residualized_rank_correlation(
                    all_interaction, all_selectivity, all_affinity
                ),
            "selection_warning": (
                "The outcome-restricted 12-arm sensitivity excludes eight chance-level arms on an "
                "outcome-related criterion. The 20-arm sensitivity is required to show "
                "the dependence of the association on that restriction."
            ),
        },
        "precision_at_5_support": {
            "n_eligible_ligands": int(len(eligible_counts)),
            "measured_targets_per_eligible_ligand": {
                "minimum": int(eligible_counts.min()),
                "median": float(np.median(eligible_counts)),
                "mean": float(eligible_counts.mean()),
                "maximum": int(eligible_counts.max()),
            },
            "relevant_targets_per_eligible_ligand": {
                "minimum": int(relevant_counts.min()),
                "median": float(np.median(relevant_counts)),
                "mean": float(relevant_counts.mean()),
                "maximum": int(relevant_counts.max()),
            },
            "mean_attainable_precision_at_5_ceiling": float(p_at_5_ceiling.mean()),
            "n_ligands_with_ceiling_one": int(np.sum(p_at_5_ceiling == 1.0)),
            "n_ligands_with_ceiling_below_one": int(np.sum(p_at_5_ceiling < 1.0)),
            "ceiling_definition": (
                "min(ceil(0.10 * n_measured_targets), 5) / 5; ties at the experimental "
                "decile threshold can only increase the number labelled relevant"
            ),
        },
        "provenance_warning": (
            "The 20 arms are heterogeneous model/configuration results, not independent draws. "
            "Association statistics are descriptive and must not be interpreted as population-level inference."
        ),
    }


def _legacy_main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    canonical = pd.read_csv(
        source_path("df_final_v4.csv"),
        usecols=DOCK44
        + ["Butina_clusters", "Canonical SMILES", "Cleaned SMILES"],
    )
    canonical["analysis_smiles"] = canonical["Cleaned SMILES"].fillna(
        canonical["Canonical SMILES"]
    )
    canonical["inchikey"] = full_inchikeys(canonical["analysis_smiles"])
    if canonical.inchikey.isna().any() or not canonical.inchikey.is_unique:
        raise ValueError("Docking-44 analysis SMILES do not produce unique full InChIKeys")
    docking_frame = canonical[DOCK44].apply(pd.to_numeric, errors="coerce")
    docking = docking_frame.clip(upper=0)
    docking = docking.fillna(docking.mean()).to_numpy(dtype=np.float64)
    docking_reference = pd.DataFrame(
        docking_frame.clip(upper=0).to_numpy(dtype=float),
        index=pd.Index(canonical.inchikey, name="inchikey"),
        columns=DOCK44,
    )

    dockstring_df = pd.read_csv(source_path("dockstring-dataset.tsv"), sep="\t")
    dockstring_cols = [c for c in dockstring_df.columns if c not in {"inchikey", "smiles"}]
    dockstring_unclipped_numeric = dockstring_df[dockstring_cols].apply(
        pd.to_numeric, errors="coerce"
    )
    dockstring_complete = ~dockstring_unclipped_numeric.isna().any(axis=1)
    dockstring_smiles = dockstring_df.loc[dockstring_complete, "smiles"].reset_index(drop=True)
    dockstring_unclipped = dockstring_unclipped_numeric.loc[dockstring_complete].to_numpy(
        dtype=np.float64
    )
    dockstring = np.minimum(dockstring_unclipped, 0.0)

    family_manifest = pd.read_csv(PACKAGE / "data/target_families.csv")
    dock44_families = (
        family_manifest[family_manifest.dataset.eq("Docking-44")]
        .set_index("target")
        .loc[DOCK44, "family"]
        .tolist()
    )
    dockstring_families = (
        family_manifest[family_manifest.dataset.eq("DOCKSTRING-58")]
        .set_index("target")
        .loc[dockstring_cols, "family"]
        .tolist()
    )

    rf1_long = pd.read_csv(source_path("negative_results_paper/analysis/rescore_ml_rfscore_scores.csv"))
    rf2_long = pd.read_csv(source_path("negative_results_paper/analysis/rescore_ml_rfscore_v2_scores.csv"))
    rf1 = pivot_complete_common(rf1_long, "rfscore", ["rfscore", "smina_vina", "orig_vina"])
    rf2 = pivot_complete_common(rf2_long, "rfscore", ["rfscore", "orig_vina"])

    leaderboard = read_json(source_path("project_clean/results/dti_leaderboard_summary.json"))
    matched_dock, matched_exp_observed, matched_exp = matched_experimental_matrices()

    dock44_ladder = centering_ladder(docking)
    dockstring_ladder = centering_ladder(dockstring)
    rf1_ladder = centering_ladder(rf1)
    rf2_ladder = centering_ladder(rf2)
    matched_dock_ladder = centering_ladder(matched_dock.to_numpy(dtype=np.float64))
    matched_exp_ladder = centering_ladder(matched_exp.to_numpy(dtype=np.float64))
    experimental_controls = experimental_sensitivity(matched_dock, matched_exp_observed)
    exact_matched = experimental_controls["activity_level_curated_matched_blocks"][
        "all_exact_median"
    ]
    matched_smiles = pd.read_csv(
        source_path("negative_results_paper/analysis/exp_positive_control_smiles.csv")
    ).set_index("inchikey")["smiles"]
    if matched_smiles.reindex(matched_dock.index).isna().any():
        raise ValueError("Matched ChEMBL benchmark is missing ligand SMILES")
    docking44_surface_bootstrap = surface_bootstrap(
        docking,
        repeats=500,
        seed=SEED + 50,
        cluster_labels=canonical["Butina_clusters"].to_numpy(),
    )
    dockstring_scaffold_sensitivity = dockstring_scaffold_bootstrap(
        dockstring,
        dockstring_smiles,
        repeats=100,
        sample_size=15000,
        supports=5,
        seed=SEED + 20,
    )
    target_preference = target_preference_benchmark(
        matched_dock,
        matched_exp_observed,
        matched_smiles,
        bootstrap_repeats=5000,
        shuffle_repeats=5000,
        imputations=100,
        seed=SEED,
    )
    expanded_preference = expanded_target_preference_benchmark(
        canonical,
        docking_reference,
        bootstrap_repeats=5000,
        permutation_repeats=5000,
        seed=SEED + 5000,
    )
    residual_characterization = {
        "docking44": residual_structure_characterization(
            docking,
            DOCK44,
            dock44_families,
            canonical["analysis_smiles"],
            sample_size=12000,
            family_permutations=5000,
            seed=SEED + 70,
        ),
        "dockstring58": residual_structure_characterization(
            dockstring,
            dockstring_cols,
            dockstring_families,
            dockstring_smiles,
            sample_size=15000,
            family_permutations=5000,
            seed=SEED + 71,
        ),
    }
    spectral_estimand_sensitivity = {
        "docking44": correlation_covariance_sensitivity(docking),
        "dockstring58": correlation_covariance_sensitivity(dockstring),
    }
    dock44_missing_counts = (
        docking_frame.isna().sum().reindex(DOCK44).to_numpy(dtype=np.int64)
    )
    docking44_target_slopes = target_physicochemical_slopes(
        docking,
        DOCK44,
        dock44_families,
        canonical["analysis_smiles"],
        sample_size=len(docking),
        seed=SEED + 70,
        imputed_cells_by_target=dock44_missing_counts,
    )
    most_imputed_index = int(np.argmax(dock44_missing_counts))
    most_imputed_target = DOCK44[most_imputed_index]
    keep_slope_targets = np.arange(len(DOCK44)) != most_imputed_index
    exclusion_slopes = target_physicochemical_slopes(
        docking[:, keep_slope_targets],
        [target for index, target in enumerate(DOCK44) if keep_slope_targets[index]],
        [family for index, family in enumerate(dock44_families) if keep_slope_targets[index]],
        canonical["analysis_smiles"],
        sample_size=len(docking),
        seed=SEED + 70,
        imputed_cells_by_target=dock44_missing_counts[keep_slope_targets],
    )
    exclusion_axis = pc1_loading_summary(
        docking[:, keep_slope_targets],
        exclusion_slopes["target_names"],
        exclusion_slopes["families"],
    )
    docking44_target_slopes["most_imputed_target_exclusion_sensitivity"] = {
        "removed_target": most_imputed_target,
        "removed_target_imputed_cells": int(dock44_missing_counts[most_imputed_index]),
        "removed_target_imputed_fraction": float(
            dock44_missing_counts[most_imputed_index] / len(docking)
        ),
        "remaining_n_targets": int(keep_slope_targets.sum()),
        "shared_axis_cosine_with_uniform_target_vector": exclusion_axis[
            "cosine_with_uniform_target_vector"
        ],
        "descriptor_slopes": exclusion_slopes["descriptor_slopes"],
        "conditional_model": exclusion_slopes["conditional_model"],
        "interpretation": (
            "This excludes the target whose scores required the most imputations while "
            "retaining the primary mean-imputed 44-target analysis separately."
        ),
    }
    physicochemical_target_slopes = {
        "docking44": docking44_target_slopes,
        "dockstring58": target_physicochemical_slopes(
            dockstring,
            dockstring_cols,
            dockstring_families,
            dockstring_smiles,
            sample_size=15000,
            seed=SEED + 71,
            imputed_cells_by_target=np.zeros(len(dockstring_cols), dtype=np.int64),
        ),
    }

    dataset_rows = [
        {
            "dataset": "Docking-44",
            "score_or_scorer": "Vina-GPU 2.0",
            "pose_source": "source orthosteric docking protocol",
            "n_ligands": len(docking),
            "n_targets": docking.shape[1],
            "missing_fraction_before_preprocessing": float(docking_frame.isna().to_numpy().mean()),
            "preprocessing": "positive/source-censored scores to 0; target-mean imputation; column z-score",
            "raw_pr": dock44_ladder["raw"]["participation_ratio"],
            "interaction_pr": dock44_ladder["interaction"]["participation_ratio"],
            "raw_entropy_rank": dock44_ladder["raw"]["entropy_rank"],
            "uncertainty": "Butina-cluster bootstrap; target leave-one-out and subsampling",
        },
        {
            "dataset": "DOCKSTRING-58",
            "score_or_scorer": "AutoDock Vina",
            "pose_source": "DOCKSTRING release protocol",
            "n_ligands": len(dockstring),
            "n_targets": dockstring.shape[1],
            "missing_fraction_before_preprocessing": 0.00002246671248373814,
            "preprocessing": "positive scores to 0; complete rows; column z-score",
            "raw_pr": dockstring_ladder["raw"]["participation_ratio"],
            "interaction_pr": dockstring_ladder["interaction"]["participation_ratio"],
            "raw_entropy_rank": dockstring_ladder["raw"]["entropy_rank"],
            "uncertainty": (
                "scaffold-cluster bootstrap on five independent chemical supports; "
                "target jackknife"
            ),
        },
        {
            "dataset": "RF-Score/Vina-pose sensitivity",
            "score_or_scorer": "RF-Score v1",
            "pose_source": "smina/Vina",
            "n_ligands": len(rf1),
            "n_targets": rf1.shape[1],
            "missing_fraction_before_preprocessing": 0.0,
            "preprocessing": "complete common block; column z-score",
            "raw_pr": rf1_ladder["raw"]["participation_ratio"],
            "interaction_pr": rf1_ladder["interaction"]["participation_ratio"],
            "raw_entropy_rank": rf1_ladder["raw"]["entropy_rank"],
            "uncertainty": "ligand bootstrap; small fixed panel",
        },
        {
            "dataset": "RF-Score/Vinardo-pose sensitivity",
            "score_or_scorer": "RF-Score v1",
            "pose_source": "Vinardo",
            "n_ligands": len(rf2),
            "n_targets": rf2.shape[1],
            "missing_fraction_before_preprocessing": 0.0,
            "preprocessing": "complete common block; column z-score",
            "raw_pr": rf2_ladder["raw"]["participation_ratio"],
            "interaction_pr": rf2_ladder["interaction"]["participation_ratio"],
            "raw_entropy_rank": rf2_ladder["raw"]["entropy_rank"],
            "uncertainty": "ligand bootstrap; small fixed panel",
        },
        {
            "dataset": "Matched docking",
            "score_or_scorer": "AutoDock Vina",
            "pose_source": "matched redocking protocol",
            "n_ligands": len(matched_dock),
            "n_targets": matched_dock.shape[1],
            "missing_fraction_before_preprocessing": 0.0,
            "preprocessing": "complete matched block; column z-score",
            "raw_pr": exact_matched["matched_docking_participation_ratio"],
            "interaction_pr": exact_matched["matched_docking_residual_participation_ratio"],
            "raw_entropy_rank": exact_matched["matched_docking_entropy_rank"],
            "uncertainty": "same ligand-target support as exact-relation median ChEMBL block",
        },
        {
            "dataset": "Matched ChEMBL affinity",
            "score_or_scorer": "experimental pChEMBL",
            "pose_source": "not applicable",
            "n_ligands": exact_matched["n_ligands"],
            "n_targets": exact_matched["n_targets"],
            "missing_fraction_before_preprocessing": exact_matched["missing_fraction"],
            "preprocessing": "exact relation; median over repeats; target-median imputation; column z-score",
            "raw_pr": exact_matched["experimental_participation_ratio"],
            "interaction_pr": exact_matched["experimental_residual_participation_ratio"],
            "raw_entropy_rank": exact_matched["experimental_entropy_rank"],
            "uncertainty": "same observed support; assay and imputation sensitivities",
        },
    ]
    pd.DataFrame(dataset_rows).to_csv(OUT / "dataset_summary.csv", index=False)

    out = {
        "definition": {
            "effective_dimension": "participation ratio of the column-standardized correlation spectrum",
            "interaction_surface": "X - column_mean - row_mean + grand_mean",
            "participation_ratio_identity": "P / (1 + (P - 1) * mean_squared_offdiagonal_correlation)",
            "interaction_rank_constraint": (
                "Two-way centering gives each interaction row sum zero. Subsequent non-zero "
                "column scaling preserves one exact linear dependence, so interaction rank is at most P-1."
            ),
            "seed": SEED,
        },
        "centering_ladder": {
            "docking44": dock44_ladder,
            "dockstring58": dockstring_ladder,
            "rfscore_v1_14targets": rf1_ladder,
            "rfscore_v1_vinardo_20targets": rf2_ladder,
            "matched_docking_74x6": matched_dock_ladder,
            "matched_experiment_74x6": matched_exp_ladder,
            "davis_experiment_floor_excluded": {
                "raw": {
                    "participation_ratio": leaderboard["arms"]["experiment"]["raw_effrank"],
                },
                "interaction": {
                    "participation_ratio": leaderboard["arms"]["experiment"]["interaction_effrank"],
                },
                "source": "project_clean/results/dti_leaderboard_summary.json",
            },
            "boltz2_davis": {
                "raw": {
                    "participation_ratio": leaderboard["arms"]["boltz_s3"]["raw_effrank"],
                },
                "interaction": {
                    "participation_ratio": leaderboard["arms"]["boltz_s3"]["interaction_effrank"],
                },
                "source": "project_clean/results/dti_leaderboard_summary.json",
            },
        },
        "pc1_axis": {
            "docking44": pc1_loading_summary(docking, DOCK44, dock44_families),
            "dockstring58": pc1_loading_summary(
                dockstring, dockstring_cols, dockstring_families
            ),
        },
        "spectral_estimand_sensitivity": spectral_estimand_sensitivity,
        "physicochemical_target_slopes": physicochemical_target_slopes,
        "target_panel_uncertainty": {
            "docking44_jackknife": target_jackknife(
                docking, DOCK44, sample_size=15000, seed=SEED
            ),
            "dockstring58_jackknife": target_jackknife(
                dockstring, dockstring_cols, sample_size=15000, seed=SEED + 30
            ),
            "docking44_unstratified_subsamples": subsample_target_surfaces(
                docking,
                [4, 6, 8, 12, 20, 32, 44],
                repeats=200,
                seed=SEED,
            ),
            "docking44_family_stratified_subsamples": subsample_target_surfaces(
                docking,
                [12, 20, 32, 44],
                repeats=200,
                seed=SEED + 10,
                families=dock44_families,
            ),
            "dockstring58_unstratified_subsamples": subsample_target_surfaces(
                dockstring,
                [4, 6, 8, 12, 20, 32, 44, 58],
                repeats=200,
                seed=SEED + 1,
            ),
            "dockstring58_family_stratified_subsamples": subsample_target_surfaces(
                dockstring,
                [6, 8, 12, 20, 32, 44, 58],
                repeats=200,
                seed=SEED + 11,
                families=dockstring_families,
            ),
            "family_manifest": "data/target_families.csv",
            "interpretation": (
                "Intervals across target subsets describe composition sensitivity, not a "
                "probability sample of all drug targets. Stratification retains every broad "
                "family when the requested subset size permits."
            ),
        },
        "ligand_sampling_sensitivity": {
            "docking44_ligand_subsamples": subsample_ligands(
                docking, [100, 300, 1000, 3000, 10000, len(docking)], repeats=50, seed=SEED + 2
            ),
            "dockstring58_ligand_subsamples": subsample_ligands(
                dockstring, [100, 300, 1000, 3000, 10000, 50000], repeats=50, seed=SEED + 3
            ),
            "docking44_butina_cluster_surface_bootstrap": docking44_surface_bootstrap,
            "dockstring_scaffold_bootstrap": dockstring_scaffold_sensitivity,
        },
        "parallel_analysis_common_protocol": {
            "docking44": parallel_analysis(
                docking, sample_size=12000, repeats=500, series=5, seed=SEED
            ),
            "dockstring58": parallel_analysis(
                dockstring, sample_size=12000, repeats=500, series=5, seed=SEED + 1
            ),
        },
        "additive_main_effect_null": {
            "docking44": additive_main_effect_null(
                docking, sample_size=12000, repeats=500, seed=SEED + 60
            ),
            "dockstring58": additive_main_effect_null(
                dockstring, sample_size=12000, repeats=500, seed=SEED + 61
            ),
        },
        "empirical_residual_permutation_null": {
            "docking44": empirical_residual_permutation_null(
                docking,
                sample_size=12000,
                repeats=500,
                seed=SEED + 60,
                cluster_labels=canonical["Butina_clusters"].to_numpy(),
                cluster_repeats=200,
            ),
            "dockstring58": empirical_residual_permutation_null(
                dockstring,
                sample_size=12000,
                repeats=500,
                seed=SEED + 61,
            ),
        },
        "row_norm_preserving_residual_null": {
            "docking44": row_norm_preserving_residual_null(
                docking, sample_size=12000, repeats=500, seed=SEED + 60
            ),
            "dockstring58": row_norm_preserving_residual_null(
                dockstring, sample_size=12000, repeats=500, seed=SEED + 61
            ),
        },
        "residual_structure_characterization": residual_characterization,
        "matched_raw_and_interaction_bootstrap": paired_surface_bootstrap(
            matched_dock.to_numpy(dtype=np.float64),
            matched_exp.to_numpy(dtype=np.float64),
            repeats=2000,
            seed=SEED,
        ),
        "matched_target_preference_benchmark": target_preference,
        "expanded_target_preference_benchmark": expanded_preference,
        "preprocessing_sensitivity": {
            **preprocessing_sensitivity(
                docking_frame.clip(upper=0),
                smiles=canonical["analysis_smiles"],
                butina_labels=canonical["Butina_clusters"],
                unclipped_docking_frame=docking_frame,
            ),
            "dockstring_clipping": {
                "complete_rows": int(len(dockstring_unclipped)),
                "strictly_positive_cells": int(np.sum(dockstring_unclipped > 0)),
                "strictly_positive_fraction": float(np.mean(dockstring_unclipped > 0)),
                "clipped": {
                    "raw_pr": dockstring_ladder["raw"]["participation_ratio"],
                    "residual_pr": dockstring_ladder["interaction"]["participation_ratio"],
                },
                "unclipped": {
                    "raw_pr": rank_summary(dockstring_unclipped)["participation_ratio"],
                    "residual_pr": rank_summary(two_way_center(dockstring_unclipped))[
                        "participation_ratio"
                    ],
                },
            },
        },
        "experimental_sensitivity": experimental_controls,
        "dti_associations": dti_associations(leaderboard),
        "frozen_source_summaries": {
            "docking44_bootstrap_and_physchem": read_json(
                source_path("negative_results_paper/analysis/rank_robustness_summary.json")
            ),
            "dockstring_bootstrap_and_null": read_json(
                source_path("negative_results_paper/analysis/review_dockstring_rmt_summary.json")
            ),
            "rfscore_vina_pose": read_json(
                source_path("negative_results_paper/analysis/rescore_ml_rfscore_summary.json")
            ),
            "rfscore_vinardo_pose": read_json(
                source_path("negative_results_paper/analysis/rescore_ml_rfscore_v2_summary.json")
            ),
            "target_expansion": read_json(
                source_path("negative_results_paper/analysis/rmt_panel_v4_contrast.json")
            ),
            "physicochemical_axis": read_json(
                source_path("negative_results_paper/analysis/physchem_depth_summary.json")
            ),
            "pocket_loading": read_json(
                source_path("negative_results_paper/analysis/pc1_physics_summary.json")
            ),
            "matched_bootstrap": read_json(
                source_path("negative_results_paper/analysis/chembl_matched_stats_summary.json")
            ),
            "larger_experimental_block": read_json(
                source_path("negative_results_paper/analysis/review_matched_exp_docking_summary.json")
            ),
            "panel_rigor": read_json(
                source_path("project_clean/results/review_panel_rigor_summary.json")
            ),
        },
    }

    metric_rows = []
    for dataset, key in [("Docking-44", "docking44"), ("DOCKSTRING-58", "dockstring58")]:
        for surface, label in [("raw", "column-standardized"), ("interaction", "residual")]:
            record = out["centering_ladder"][key][surface]
            metric_rows.append({
                "dataset": dataset,
                "surface": label,
                "n_targets": record["n_targets"],
                "participation_ratio": record["participation_ratio"],
                "mean_squared_offdiagonal_correlation": record[
                    "mean_squared_offdiagonal_correlation"
                ],
                "mean_absolute_offdiagonal_correlation": record[
                    "mean_absolute_offdiagonal_correlation"
                ],
                "pc1_fraction": record["pc1_fraction"],
                "entropy_rank": record["entropy_rank"],
            })
    pd.DataFrame(metric_rows).to_csv(OUT / "spectral_metric_comparison.csv", index=False)

    target_order_rows = []
    for dataset, key in [("Docking-44", "docking44"), ("DOCKSTRING-58", "dockstring58")]:
        record = out["residual_structure_characterization"][key]
        family_by_target = dict(zip(record["target_names"], record["families"]))
        for position, target in enumerate(record["residual_cluster_order"], start=1):
            target_order_rows.append({
                "dataset": dataset,
                "cluster_order": position,
                "target": target,
                "family": family_by_target[target],
            })
    pd.DataFrame(target_order_rows).to_csv(OUT / "residual_target_order.csv", index=False)

    chemistry_rows = []
    for descriptor, record in out["preprocessing_sensitivity"][
        "complete_case_chemical_support"
    ]["descriptor_comparison"].items():
        chemistry_rows.append({"descriptor": descriptor, **record})
    pd.DataFrame(chemistry_rows).to_csv(
        OUT / "complete_case_chemical_support.csv", index=False
    )

    missing_rows = []
    missing_sensitivity = out["preprocessing_sensitivity"][
        "missing_value_residual_sensitivity"
    ]
    for key, record in missing_sensitivity["methods"].items():
        missing_rows.append({"analysis": "all_targets", "method_key": key, **{
            field: value for field, value in record.items() if field != "fit"
        }})
    removed = missing_sensitivity["without_most_affected_target"]
    for key in ("target_mean", "target_median", "complete_case"):
        missing_rows.append({
            "analysis": f"without_{removed['removed_target']}",
            "method_key": key,
            **removed[key],
        })
    pd.DataFrame(missing_rows).to_csv(
        OUT / "missing_residual_sensitivity.csv", index=False
    )

    covariance_rows = []
    for dataset, key in (("Docking-44", "docking44"), ("DOCKSTRING-58", "dockstring58")):
        sensitivity = spectral_estimand_sensitivity[key]
        for estimand in ("correlation", "covariance"):
            for surface in ("raw", "residual"):
                record = sensitivity[estimand][surface]
                covariance_rows.append({
                    "dataset": dataset,
                    "estimand": estimand,
                    "surface": surface,
                    "n_ligands": record["n_ligands"],
                    "n_targets": record["n_targets"],
                    "algebraic_ceiling": record["algebraic_ceiling"],
                    "participation_ratio": record["participation_ratio"],
                    "dimension_fraction_of_ceiling": record[
                        "dimension_fraction_of_ceiling"
                    ],
                    "entropy_rank": record["entropy_rank"],
                    "pc1_fraction": record["pc1_fraction"],
                    "pcs_for_90pct": record["pcs_for_90pct"],
                    "target_sd_cv": (
                        record.get("target_standard_deviation", {})
                        .get("coefficient_of_variation", np.nan)
                    ),
                })
    pd.DataFrame(covariance_rows).to_csv(
        OUT / "correlation_covariance_sensitivity.csv", index=False
    )

    slope_rows = []
    for dataset, key in (("Docking-44", "docking44"), ("DOCKSTRING-58", "dockstring58")):
        record = physicochemical_target_slopes[key]
        family_by_target = dict(zip(record["target_names"], record["families"]))
        for target in record["target_names"]:
            slope_rows.append({
                "dataset": dataset,
                "target": target,
                "family": family_by_target[target],
                **record["per_target"][target],
            })
    pd.DataFrame(slope_rows).to_csv(
        OUT / "target_physicochemical_slopes.csv", index=False
    )

    slope_sensitivity_rows = []
    for dataset, key in (("Docking-44", "docking44"), ("DOCKSTRING-58", "dockstring58")):
        record = physicochemical_target_slopes[key]
        target_sets = [("primary", "", record)]
        if key == "docking44":
            exclusion = record["most_imputed_target_exclusion_sensitivity"]
            target_sets.append((
                "without_most_imputed_target",
                exclusion["removed_target"],
                exclusion,
            ))
        for target_set, removed_target, target_set_record in target_sets:
            conditional = target_set_record["conditional_model"]
            for descriptor, descriptor_record in target_set_record[
                "descriptor_slopes"
            ].items():
                slope_sensitivity_rows.append({
                    "dataset": dataset,
                    "target_set": target_set,
                    "removed_target": removed_target,
                    "descriptor": descriptor,
                    "loading_vector_cosine": descriptor_record[
                        "cosine_with_residual_pc1_loading"
                    ],
                    "absolute_descriptor_residual_pc1_score_correlation":
                        descriptor_record[
                            "absolute_correlation_with_residual_pc1_scores"
                        ],
                    "median_raw_target_slope": descriptor_record[
                        "raw_target_slope"
                    ]["median"],
                    "median_residual_target_slope": descriptor_record[
                        "residual_target_slope"
                    ]["median"],
                    "conditional_mw_slope_median": conditional[
                        "molecular_weight_slope"
                    ]["median"],
                    "conditional_rotatable_bond_slope_median": conditional[
                        "rotatable_bond_slope"
                    ]["median"],
                    "targets_with_positive_conditional_rotatable_bond_slope":
                        conditional["targets_with_positive_rotatable_bond_slope"],
                })
    pd.DataFrame(slope_sensitivity_rows).to_csv(
        OUT / "physicochemical_slope_sensitivity.csv", index=False
    )

    (OUT / "evidence_summary.json").write_text(
        json.dumps(out, indent=2, allow_nan=False) + "\n"
    )
    print(f"Wrote {OUT / 'evidence_summary.json'}")
    print(
        "Headline PR: "
        f"Docking-44 {dock44_ladder['raw']['participation_ratio']:.3f} -> "
        f"{dock44_ladder['interaction']['participation_ratio']:.3f}; "
        f"DOCKSTRING-58 {dockstring_ladder['raw']['participation_ratio']:.3f} -> "
        f"{dockstring_ladder['interaction']['participation_ratio']:.3f}"
    )


def main() -> None:
    """Stop the obsolete DTI/RF evidence workflow from overwriting release outputs."""

    raise SystemExit(
        "analysis/build_evidence.py is a deprecated legacy DTI/RF workflow and "
        "is intentionally disabled. Use analysis/build_manuscript_evidence.py "
        "for the current release evidence ledger."
    )


if __name__ == "__main__":
    main()
