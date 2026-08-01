#!/usr/bin/env python3
"""Build the evidence ledger for the Journal of Cheminformatics paper.

The script deliberately keeps three objects separate:

1. the raw ligand-by-target score surface;
2. the two-way-centred ligand-target interaction surface; and
3. external performance metrics such as affinity correlation or selectivity P@5.

It does not consume the live OKL analysis under ``reproduce/``.  That study has a
different estimand and is not submission-ready.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import numpy as np
import pandas as pd
from scipy import stats
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


def spectrum(x: np.ndarray) -> np.ndarray:
    """Eigenvalues of the target correlation matrix after column standardisation."""
    x = np.asarray(x, dtype=np.float64)
    keep = np.nanstd(x, axis=0, ddof=1) > 1e-12
    x = x[:, keep]
    if np.isnan(x).any():
        means = np.nanmean(x, axis=0)
        rows, cols = np.where(np.isnan(x))
        x[rows, cols] = means[cols]
    z = (x - x.mean(axis=0)) / x.std(axis=0, ddof=1)
    eig = np.linalg.eigvalsh((z.T @ z) / (len(z) - 1))[::-1]
    return np.clip(eig, 0.0, None)


def rank_summary(x: np.ndarray) -> dict:
    eig = spectrum(x)
    return rank_summary_from_eigenvalues(eig, np.asarray(x).shape)


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
    return {
        "targets": list(target_names),
        "families": list(families),
        "loadings": [float(value) for value in loading],
        "minimum_loading": float(loading.min()),
        "maximum_loading": float(loading.max()),
        "n_positive_loadings": int(np.sum(loading > 0)),
        "n_negative_loadings": int(np.sum(loading < 0)),
        "pc1_eigenvalue": float(eigenvalues[-1]),
        "correlation_with_raw_per_ligand_mean": float(
            stats.pearsonr(scores, matrix.mean(axis=1)).statistic
        ),
        "correlation_with_column_standardized_per_ligand_mean": float(
            stats.pearsonr(scores, z.mean(axis=1)).statistic
        ),
        "orientation": "sign chosen so the mean target loading is positive",
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
    rng = np.random.default_rng(seed)
    if len(matrix) > sample_size:
        support = rng.choice(len(matrix), sample_size, replace=False)
        work = matrix[support]
    else:
        work = matrix
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


def preprocessing_sensitivity(docking_frame: pd.DataFrame) -> dict:
    """Quantify the preprocessing alternatives requested during internal review."""
    numeric = docking_frame.apply(pd.to_numeric, errors="coerce")
    mean_filled = numeric.fillna(numeric.mean())
    median_filled = numeric.fillna(numeric.median())
    complete = numeric.dropna(axis=0, how="any")

    x = mean_filled.to_numpy(dtype=np.float64)
    interaction = two_way_center(x)
    z = (x - x.mean(axis=0)) / x.std(axis=0, ddof=1)
    robust_scale = np.median(np.abs(x - np.median(x, axis=0)), axis=0) * 1.4826
    robust_scale[robust_scale < 1e-12] = 1.0
    robust_z = (x - np.median(x, axis=0)) / robust_scale

    ranked = numeric.rank(axis=0, method="average", na_option="keep")
    pairwise_corr = numeric.corr(min_periods=100).to_numpy(dtype=np.float64)
    spearman_corr = ranked.corr(min_periods=100).to_numpy(dtype=np.float64)

    return {
        "input_characterization": {
            "n_ligands": int(len(numeric)),
            "n_targets": int(numeric.shape[1]),
            "missing_cells": int(numeric.isna().to_numpy().sum()),
            "missing_fraction": float(numeric.isna().to_numpy().mean()),
            "rows_with_any_missing": int(numeric.isna().any(axis=1).sum()),
            "complete_rows": int(len(complete)),
            "exact_zero_or_source_censored_cells": int((numeric.to_numpy() == 0).sum()),
            "exact_zero_or_source_censored_fraction": float((numeric.to_numpy() == 0).mean()),
            "strictly_positive_cells_in_processed_file": int((numeric.to_numpy() > 0).sum()),
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
        "invariance_note": (
            "For a raw complete matrix, the correlation spectrum is algebraically invariant to "
            "non-zero affine rescaling of individual columns. Robust scaling matters only when "
            "performed before row-effect removal."
        ),
    }


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
    total_correct = 0
    total_pairs = 0
    excluded_ties = 0
    for ligand in range(len(experiment)):
        observed = np.flatnonzero(np.isfinite(experiment[ligand]))
        correct = 0
        pairs = 0
        for first in range(len(observed)):
            for second in range(first + 1, len(observed)):
                i, j = observed[first], observed[second]
                experimental_difference = experiment[ligand, i] - experiment[ligand, j]
                if abs(experimental_difference) <= tie_tolerance:
                    excluded_ties += 1
                    continue
                score_difference = scores[ligand, i] - scores[ligand, j]
                correct += int(
                    np.sign(experimental_difference) == -np.sign(score_difference)
                )
                pairs += 1
        if pairs:
            per_ligand_accuracy[ligand] = correct / pairs
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
    return {
        "per_ligand_pairwise_accuracy": per_ligand_accuracy,
        "per_ligand_spearman": per_ligand_spearman,
        "per_ligand_top1": per_ligand_top1,
        "mean_per_ligand_pairwise_accuracy": float(np.nanmean(per_ligand_accuracy)),
        "pair_weighted_accuracy": float(total_correct / total_pairs),
        "mean_per_ligand_spearman": float(np.nanmean(per_ligand_spearman)),
        "top1_accuracy": float(np.nanmean(per_ligand_top1)),
        "evaluated_pairs": int(total_pairs),
        "excluded_ties": int(excluded_ties),
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

    davis = pd.read_csv(source_path("project_clean/data/processed/davis_boltz2/davis_complete.tab"), sep="\t")
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


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    canonical = pd.read_csv(source_path("df_final_v4.csv"), usecols=DOCK44 + ["Butina_clusters"])
    docking_frame = canonical[DOCK44].apply(pd.to_numeric, errors="coerce")
    docking = docking_frame.clip(upper=0)
    docking = docking.fillna(docking.mean()).to_numpy(dtype=np.float64)

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
        "matched_raw_and_interaction_bootstrap": paired_surface_bootstrap(
            matched_dock.to_numpy(dtype=np.float64),
            matched_exp.to_numpy(dtype=np.float64),
            repeats=2000,
            seed=SEED,
        ),
        "matched_target_preference_benchmark": target_preference,
        "preprocessing_sensitivity": {
            **preprocessing_sensitivity(docking_frame),
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

    (OUT / "evidence_summary.json").write_text(json.dumps(out, indent=2) + "\n")
    print(f"Wrote {OUT / 'evidence_summary.json'}")
    print(
        "Headline PR: "
        f"Docking-44 {dock44_ladder['raw']['participation_ratio']:.3f} -> "
        f"{dock44_ladder['interaction']['participation_ratio']:.3f}; "
        f"DOCKSTRING-58 {dockstring_ladder['raw']['participation_ratio']:.3f} -> "
        f"{dockstring_ladder['interaction']['participation_ratio']:.3f}"
    )


if __name__ == "__main__":
    main()
