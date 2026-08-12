#!/usr/bin/env python3
"""Strict-public Docking-44 missing-data and chemical-support sensitivity.

The producer reads one frozen row-level score table and does not import local
analysis modules or consume any evidence ledger.  It separates imputation from
chemical-support selection, fits an observed-cell additive model as a
missing-aware sensitivity, and compares the complete-case residual PR with
equal-size random ligand supports.  Chemical differences are reported as
descriptive effect sizes without null-hypothesis tests.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rdkit import Chem, rdBase


PACKAGE = Path(__file__).resolve().parents[1]
FROZEN = PACKAGE / "data" / "frozen"
DEFAULT_INPUT = FROZEN / "df_final_v4.csv.gz"
DEFAULT_OUTPUT = PACKAGE / "results" / "public_docking44_missing_data_sensitivity"
DEFAULT_RANDOM_CONTROLS = 1_000
DEFAULT_SEED = 202_608_11
DOCK44 = [
    "1m2z", "1pbq", "1xoq", "2rh1", "2vt4", "2ydo", "2z5x", "3b66",
    "3kk6", "3ln1", "3rze", "4djh", "4ey7", "4iar", "4mqs", "4n6h",
    "5cxv", "5i71", "5tvn", "5u09", "5va1", "6cm4", "6kpf", "6kux",
    "6lqa", "6pdj", "6x3x", "6y1z", "7f8y", "7kwe", "7ljd", "7wc9",
    "7xnk", "7ym8", "8e9y", "8ef6", "8fhs", "8pjk", "8st0", "8wty",
    "8xvk", "8yn3", "9eo4", "V1A",
]
DESCRIPTOR_COLUMNS = {
    "molecular_weight": "MW, g/mol",
    "rotatable_bonds": "rotatable_bonds",
    "logp": "logP",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def index_sha256(indices: np.ndarray) -> str:
    canonical = np.asarray(np.sort(indices), dtype="<i8")
    return hashlib.sha256(canonical.tobytes()).hexdigest()


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
        raise ValueError("non-finite value cannot be serialized")
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(json_ready(payload), indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def two_way_center(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2 or min(matrix.shape) < 2 or not np.isfinite(matrix).all():
        raise ValueError("two-way centering requires a finite two-dimensional matrix")
    return (
        matrix
        - matrix.mean(axis=0, keepdims=True)
        - matrix.mean(axis=1, keepdims=True)
        + matrix.mean()
    )


def spectrum_summary(matrix: np.ndarray) -> dict[str, Any]:
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2 or len(matrix) < 3 or not np.isfinite(matrix).all():
        raise ValueError("spectral summary requires at least three finite rows")
    standard_deviations = matrix.std(axis=0, ddof=1)
    if np.any(standard_deviations <= 1e-12):
        raise ValueError("target column has negligible variance")
    standardized = (matrix - matrix.mean(axis=0)) / standard_deviations
    correlation = standardized.T @ standardized / (len(matrix) - 1)
    correlation = (correlation + correlation.T) / 2.0
    eigenvalues = np.clip(np.linalg.eigvalsh(correlation)[::-1], 0.0, None)
    weights = eigenvalues / eigenvalues.sum()
    positive = weights[weights > 0]
    cumulative = np.cumsum(weights)
    upper = correlation[np.triu_indices(len(correlation), 1)]
    return {
        "participation_ratio": float(
            eigenvalues.sum() ** 2 / np.square(eigenvalues).sum()
        ),
        "entropy_rank": float(np.exp(-np.sum(positive * np.log(positive)))),
        "pc1_fraction": float(weights[0]),
        "pcs_for_90pct": int(np.searchsorted(cumulative, 0.90) + 1),
        "mean_absolute_offdiagonal_correlation": float(np.mean(np.abs(upper))),
    }


def pr_from_sufficient_statistics(
    rows: int,
    column_sum: np.ndarray,
    cross_product: np.ndarray,
) -> float:
    covariance = (
        cross_product - np.outer(column_sum, column_sum) / rows
    ) / (rows - 1)
    standard_deviations = np.sqrt(np.clip(np.diag(covariance), 0.0, None))
    if np.any(standard_deviations <= 1e-12):
        raise ValueError("random support has a target with negligible variance")
    correlation = covariance / np.outer(standard_deviations, standard_deviations)
    correlation = (correlation + correlation.T) / 2.0
    np.fill_diagonal(correlation, 1.0)
    eigenvalues = np.clip(np.linalg.eigvalsh(correlation), 0.0, None)
    return float(eigenvalues.sum() ** 2 / np.square(eigenvalues).sum())


def observed_cell_additive_residual(
    matrix_with_nan: np.ndarray,
    *,
    tolerance: float = 1e-12,
    maximum_iterations: int = 10_000,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Observed-cell least-squares additive residual with neutral completion.

    Alternating exact updates solve the normal equations for one row intercept
    and one target effect.  The fit uses observed cells only.  Unobserved cells
    are assigned residual zero, their fitted expectation under the additive
    model; this is a residual completion, not raw-score imputation.
    """
    matrix = np.asarray(matrix_with_nan, dtype=np.float64)
    if matrix.ndim != 2 or min(matrix.shape) < 2:
        raise ValueError("matrix must be two-dimensional")
    observed = np.isfinite(matrix)
    row_counts = observed.sum(axis=1)
    column_counts = observed.sum(axis=0)
    if np.any(row_counts == 0) or np.any(column_counts == 0):
        raise ValueError("every row and target must contain an observed score")
    target_effect = np.nanmean(matrix, axis=0)
    target_effect -= target_effect.mean()
    row_effect = np.zeros(len(matrix), dtype=np.float64)
    converged = False
    maximum_update = float("inf")
    for iteration in range(1, maximum_iterations + 1):
        row_effect = np.where(
            observed, matrix - target_effect[None, :], 0.0
        ).sum(axis=1) / row_counts
        updated_target = np.where(
            observed, matrix - row_effect[:, None], 0.0
        ).sum(axis=0) / column_counts
        shift = float(updated_target.mean())
        updated_target -= shift
        row_effect += shift
        maximum_update = float(np.max(np.abs(updated_target - target_effect)))
        target_effect = updated_target
        if maximum_update <= tolerance:
            converged = True
            break
    if not converged:
        raise RuntimeError("observed-cell additive least squares did not converge")
    fitted_residual = np.zeros_like(matrix)
    fitted_residual[observed] = (
        matrix[observed]
        - np.broadcast_to(row_effect[:, None], matrix.shape)[observed]
        - np.broadcast_to(target_effect[None, :], matrix.shape)[observed]
    )
    row_normal_error = float(
        np.max(np.abs(fitted_residual.sum(axis=1) / row_counts))
    )
    column_normal_error = float(
        np.max(np.abs(fitted_residual.sum(axis=0) / column_counts))
    )
    diagnostics = {
        "iterations": int(iteration),
        "converged": converged,
        "tolerance": float(tolerance),
        "maximum_target_effect_update": maximum_update,
        "maximum_absolute_observed_row_residual_mean": row_normal_error,
        "maximum_absolute_observed_target_residual_mean": column_normal_error,
        "unobserved_residual_completion": 0.0,
        "objective": "unweighted least squares over observed cells",
    }
    return fitted_residual, diagnostics


def equal_n_random_support_controls(
    primary_imputed: np.ndarray,
    *,
    selected_rows: int,
    repetitions: int,
    seed: int,
) -> np.ndarray:
    """Random equal-N PRs from the primary full-support processed matrix."""
    matrix = np.asarray(primary_imputed, dtype=np.float64)
    if not 3 <= selected_rows < len(matrix) or repetitions < 2:
        raise ValueError("invalid equal-N random-support configuration")
    row_centered = matrix - matrix.mean(axis=1, keepdims=True)
    full_sum = row_centered.sum(axis=0)
    full_cross = row_centered.T @ row_centered
    omitted_rows = len(matrix) - selected_rows
    rng = np.random.default_rng(seed)
    values = np.empty(repetitions, dtype=np.float64)
    for repetition in range(repetitions):
        omitted = rng.choice(len(matrix), size=omitted_rows, replace=False)
        omitted_matrix = row_centered[omitted]
        values[repetition] = pr_from_sufficient_statistics(
            selected_rows,
            full_sum - omitted_matrix.sum(axis=0),
            full_cross - omitted_matrix.T @ omitted_matrix,
        )
    return values


def describe(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        raise ValueError("descriptor has no finite values")
    median = float(np.median(finite))
    return {
        "n": int(len(finite)),
        "mean": float(finite.mean()),
        "standard_deviation": float(finite.std(ddof=1)) if len(finite) > 1 else 0.0,
        "q025": float(np.quantile(finite, 0.025)),
        "q10": float(np.quantile(finite, 0.10)),
        "q25": float(np.quantile(finite, 0.25)),
        "median": median,
        "q75": float(np.quantile(finite, 0.75)),
        "q90": float(np.quantile(finite, 0.90)),
        "q975": float(np.quantile(finite, 0.975)),
        "interquartile_range": float(
            np.quantile(finite, 0.75) - np.quantile(finite, 0.25)
        ),
        "median_absolute_deviation": float(np.median(np.abs(finite - median))),
    }


def cliffs_delta(excluded: np.ndarray, complete: np.ndarray) -> float:
    excluded = np.asarray(excluded, dtype=np.float64)
    complete = np.sort(np.asarray(complete, dtype=np.float64))
    excluded = excluded[np.isfinite(excluded)]
    complete = complete[np.isfinite(complete)]
    if len(excluded) == 0 or len(complete) == 0:
        raise ValueError("Cliff's delta requires two nonempty finite groups")
    lower = np.searchsorted(complete, excluded, side="left").sum()
    upper = (len(complete) - np.searchsorted(complete, excluded, side="right")).sum()
    return float((lower - upper) / (len(excluded) * len(complete)))


def chemistry_effect_size(
    complete: np.ndarray,
    excluded: np.ndarray,
) -> dict[str, float]:
    complete = np.asarray(complete, dtype=np.float64)
    excluded = np.asarray(excluded, dtype=np.float64)
    complete = complete[np.isfinite(complete)]
    excluded = excluded[np.isfinite(excluded)]
    first = describe(complete)
    second = describe(excluded)
    pooled_variance = (
        (len(complete) - 1) * np.var(complete, ddof=1)
        + (len(excluded) - 1) * np.var(excluded, ddof=1)
    ) / (len(complete) + len(excluded) - 2)
    pooled_sd = float(np.sqrt(pooled_variance))
    mean_iqr = (first["interquartile_range"] + second["interquartile_range"]) / 2
    median_difference = second["median"] - first["median"]
    return {
        "mean_difference_excluded_minus_complete": second["mean"] - first["mean"],
        "median_difference_excluded_minus_complete": median_difference,
        "standardized_mean_difference_excluded_minus_complete": (
            (second["mean"] - first["mean"]) / pooled_sd if pooled_sd > 0 else 0.0
        ),
        "median_difference_over_mean_group_iqr": (
            median_difference / mean_iqr if mean_iqr > 0 else 0.0
        ),
        "cliffs_delta_excluded_vs_complete": cliffs_delta(excluded, complete),
    }


def load_input(path: Path) -> dict[str, Any]:
    columns = DOCK44 + [
        "Cleaned SMILES",
        "Canonical SMILES",
        *DESCRIPTOR_COLUMNS.values(),
    ]
    frame = pd.read_csv(path, usecols=columns)
    numeric = frame[DOCK44].apply(pd.to_numeric, errors="coerce")
    original = numeric.to_numpy(dtype=np.float64)
    positive_cells = int(np.nansum(original > 0))
    clipped = np.minimum(original, 0.0)
    complete = np.isfinite(clipped).all(axis=1)
    descriptors = {
        name: pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=np.float64)
        for name, column in DESCRIPTOR_COLUMNS.items()
    }
    smiles = frame["Cleaned SMILES"].fillna(frame["Canonical SMILES"]).astype(str)
    heavy_atoms = np.full(len(frame), np.nan, dtype=np.float64)
    invalid_smiles = 0
    for index, value in enumerate(smiles):
        molecule = Chem.MolFromSmiles(value)
        if molecule is None:
            invalid_smiles += 1
        else:
            heavy_atoms[index] = molecule.GetNumHeavyAtoms()
    descriptors["heavy_atoms"] = heavy_atoms
    return {
        "matrix": clipped,
        "complete_mask": complete,
        "descriptors": descriptors,
        "positive_cells_clipped": positive_cells,
        "invalid_smiles_for_heavy_atoms": invalid_smiles,
    }


def analyze(
    loaded: dict[str, Any],
    *,
    random_controls: int,
    seed: int,
) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
    matrix = np.asarray(loaded["matrix"], dtype=np.float64)
    complete_mask = np.asarray(loaded["complete_mask"], dtype=bool)
    complete_indices = np.flatnonzero(complete_mask)
    excluded_indices = np.flatnonzero(~complete_mask)
    target_means = np.nanmean(matrix, axis=0)
    target_medians = np.nanmedian(matrix, axis=0)
    mean_imputed = np.where(np.isfinite(matrix), matrix, target_means[None, :])
    median_imputed = np.where(np.isfinite(matrix), matrix, target_medians[None, :])
    complete_observed = matrix[complete_mask]
    primary_restricted = mean_imputed[complete_mask]

    surfaces: list[dict[str, Any]] = []

    def add_complete_surface(name: str, values: np.ndarray, interpretation: str) -> None:
        raw = spectrum_summary(values)
        residual = spectrum_summary(two_way_center(values))
        surfaces.append(
            {
                "analysis": name,
                "ligands": int(len(values)),
                "observed_cells_used": int(np.isfinite(values).sum()),
                "raw_pr": raw["participation_ratio"],
                "residual_pr": residual["participation_ratio"],
                "residual_entropy_rank": residual["entropy_rank"],
                "residual_pc1_fraction": residual["pc1_fraction"],
                "residual_pcs_for_90pct": residual["pcs_for_90pct"],
                "residual_mean_absolute_correlation": residual[
                    "mean_absolute_offdiagonal_correlation"
                ],
                "interpretation": interpretation,
            }
        )

    add_complete_surface(
        "target_mean_imputation_full_support",
        mean_imputed,
        "primary full-support preprocessing",
    )
    add_complete_surface(
        "target_median_imputation_full_support",
        median_imputed,
        "same full support; changes only imputed values",
    )
    add_complete_surface(
        "complete_rows_observed_scores",
        complete_observed,
        "complete-case support; no imputation is applied",
    )
    add_complete_surface(
        "target_mean_imputed_restricted_to_complete_rows",
        primary_restricted,
        "primary processed matrix restricted to exactly the complete-case rows",
    )
    wls_residual, wls_diagnostics = observed_cell_additive_residual(matrix)
    wls_summary = spectrum_summary(wls_residual)
    surfaces.append(
        {
            "analysis": "observed_cell_wls_zero_residual_completion",
            "ligands": int(len(matrix)),
            "observed_cells_used": int(np.isfinite(matrix).sum()),
            "raw_pr": np.nan,
            "residual_pr": wls_summary["participation_ratio"],
            "residual_entropy_rank": wls_summary["entropy_rank"],
            "residual_pc1_fraction": wls_summary["pc1_fraction"],
            "residual_pcs_for_90pct": wls_summary["pcs_for_90pct"],
            "residual_mean_absolute_correlation": wls_summary[
                "mean_absolute_offdiagonal_correlation"
            ],
            "interpretation": (
                "additive effects fitted only on observed cells; missing residuals are "
                "completed by their fitted additive expectation of zero"
            ),
        }
    )
    spectral = pd.DataFrame.from_records(surfaces)
    primary_pr = float(
        spectral.loc[
            spectral.analysis.eq("target_mean_imputation_full_support"), "residual_pr"
        ].iloc[0]
    )
    spectral["residual_pr_difference_vs_primary"] = spectral.residual_pr - primary_pr
    spectral["residual_pr_ratio_vs_primary"] = spectral.residual_pr / primary_pr

    equality_error = float(np.max(np.abs(complete_observed - primary_restricted)))
    if equality_error != 0.0:
        raise RuntimeError("complete rows changed under target-mean imputation")

    controls = equal_n_random_support_controls(
        mean_imputed,
        selected_rows=len(complete_indices),
        repetitions=random_controls,
        seed=seed,
    )
    complete_pr = float(
        spectral.loc[
            spectral.analysis.eq("complete_rows_observed_scores"), "residual_pr"
        ].iloc[0]
    )
    control_frame = pd.DataFrame(
        {
            "repetition": np.arange(random_controls, dtype=int),
            "random_equal_n_residual_pr": controls,
            "random_minus_full_primary_residual_pr": controls - primary_pr,
            "complete_case_minus_random_residual_pr": complete_pr - controls,
        }
    )
    random_summary = describe(controls)
    random_summary.update(
        {
            "repetitions": int(random_controls),
            "seed": int(seed),
            "rows_per_random_support": int(len(complete_indices)),
            "observed_complete_case_residual_pr": complete_pr,
            "observed_complete_case_minus_full_primary": complete_pr - primary_pr,
            "observed_complete_case_minus_random_mean": (
                complete_pr - random_summary["mean"]
            ),
            "observed_complete_case_minus_random_median": (
                complete_pr - random_summary["median"]
            ),
            "fraction_random_supports_at_or_below_complete_case": float(
                np.mean(controls <= complete_pr)
            ),
            "interpretation": (
                "descriptive equal-N support-selection diagnostic, not a population "
                "p-value; every random support is sampled from the primary mean-imputed "
                "full ligand set without replacement"
            ),
        }
    )

    chemistry_summary_records: list[dict[str, Any]] = []
    chemistry_effect_records: list[dict[str, Any]] = []
    descriptor_units = {
        "molecular_weight": "g/mol",
        "heavy_atoms": "count",
        "rotatable_bonds": "count",
        "logp": "dimensionless",
    }
    descriptor_provenance = {
        "molecular_weight": "frozen column MW, g/mol",
        "heavy_atoms": "RDKit GetNumHeavyAtoms from frozen cleaned/canonical SMILES",
        "rotatable_bonds": "frozen column rotatable_bonds",
        "logp": "frozen column logP",
    }
    for descriptor, values in loaded["descriptors"].items():
        values = np.asarray(values, dtype=np.float64)
        for group, mask in (
            ("complete_rows", complete_mask),
            ("excluded_incomplete_rows", ~complete_mask),
        ):
            stats = describe(values[mask])
            chemistry_summary_records.append(
                {
                    "descriptor": descriptor,
                    "unit": descriptor_units[descriptor],
                    "provenance": descriptor_provenance[descriptor],
                    "group": group,
                    "group_rows": int(mask.sum()),
                    "finite_descriptor_rows": stats.pop("n"),
                    **stats,
                }
            )
        chemistry_effect_records.append(
            {
                "descriptor": descriptor,
                "unit": descriptor_units[descriptor],
                "direction": "excluded_incomplete_rows minus complete_rows",
                **chemistry_effect_size(values[complete_mask], values[~complete_mask]),
            }
        )
    chemistry_summary = pd.DataFrame.from_records(chemistry_summary_records)
    chemistry_effects = pd.DataFrame.from_records(chemistry_effect_records)

    missing_counts = np.sum(~np.isfinite(matrix), axis=0)
    if matrix.shape[1] > len(DOCK44):
        raise ValueError("matrix contains more targets than the declared Docking-44 panel")
    missing_by_target = pd.DataFrame(
        {
            "target": DOCK44[: matrix.shape[1]],
            "missing_cells": missing_counts.astype(int),
            "fraction_rows_missing": missing_counts / len(matrix),
        }
    ).sort_values(["missing_cells", "target"], ascending=[False, True])

    summary = {
        "schema_version": "1.0.0",
        "analysis_status": "standalone_public_revision_sensitivity",
        "producer": "analysis/public_docking44_missing_data_sensitivity.py",
        "matrix": {
            "rows": int(len(matrix)),
            "targets": int(matrix.shape[1]),
            "total_cells": int(matrix.size),
            "missing_cells": int((~np.isfinite(matrix)).sum()),
            "missing_cell_fraction": float((~np.isfinite(matrix)).mean()),
            "rows_with_any_missing_score": int((~complete_mask).sum()),
            "complete_rows": int(complete_mask.sum()),
            "positive_cells_clipped_to_zero": int(loaded["positive_cells_clipped"]),
            "complete_row_index_sha256": index_sha256(complete_indices),
            "excluded_row_index_sha256": index_sha256(excluded_indices),
        },
        "spectral_sensitivity": {
            row["analysis"]: {
                key: json_ready(value)
                for key, value in row.items()
                if key != "analysis" and not (isinstance(value, float) and np.isnan(value))
            }
            for row in spectral.to_dict(orient="records")
        },
        "complete_row_identity_check": {
            "maximum_absolute_score_difference_between_observed_complete_rows_and_primary_imputed_restricted_rows": equality_error,
            "interpretation": (
                "these two surfaces are algebraically identical; their comparison verifies "
                "that the complete-case contrast is a support change, not an imputation change"
            ),
        },
        "observed_cell_wls_diagnostics": wls_diagnostics,
        "equal_n_random_support_control": random_summary,
        "chemistry": {
            "effect_size_direction": "excluded incomplete rows minus complete rows",
            "invalid_smiles_for_heavy_atom_calculation": int(
                loaded["invalid_smiles_for_heavy_atoms"]
            ),
            "rdkit_version": rdBase.rdkitVersion,
            "no_hypothesis_tests": True,
        },
        "data_dictionary": {
            "observational_unit": "one ligand row scored against 44 targets",
            "missingness": (
                "a missing docking-score cell; mechanism is not assumed to be random"
            ),
            "primary_estimand": (
                "target-correlation participation-ratio dimension of the two-way-centred "
                "target-mean-imputed surface"
            ),
            "equal_n_control_estimand": (
                "distribution of residual PR across random ligand subsets having exactly "
                "the complete-case row count, drawn from the primary full processed surface"
            ),
        },
        "claim_boundary": (
            "The analysis diagnoses sensitivity of one fixed Docking-44 Vina matrix to its "
            "observed missingness pattern. Equal-N controls separate unusual complete-case "
            "support from sample-size reduction descriptively, but are not a population "
            "randomization test because missingness was not randomized. Chemical effect sizes "
            "describe association with inclusion and do not identify a causal missingness "
            "mechanism. Zero residual completion is an additive-model sensitivity and does not "
            "recover unknown docking scores."
        ),
    }
    tables = {
        "spectral_sensitivity.csv": spectral,
        "equal_n_random_controls.csv": control_frame,
        "chemistry_group_summary.csv": chemistry_summary,
        "chemistry_effect_sizes.csv": chemistry_effects,
        "missingness_by_target.csv": missing_by_target,
    }
    return summary, tables


def build_readme(summary: dict[str, Any]) -> str:
    spectral = summary["spectral_sensitivity"]
    primary = spectral["target_mean_imputation_full_support"]["residual_pr"]
    median = spectral["target_median_imputation_full_support"]["residual_pr"]
    complete = spectral["complete_rows_observed_scores"]["residual_pr"]
    missing_aware = spectral["observed_cell_wls_zero_residual_completion"][
        "residual_pr"
    ]
    control = summary["equal_n_random_support_control"]
    return f"""# Docking-44 missing-data residual sensitivity

This strict-public artifact is generated directly from the frozen row-level
Docking-44 table. It does not read an evidence ledger or import a local analysis
builder.

## Result

- Target-mean full-support residual PR: {primary:.6f}
- Target-median full-support residual PR: {median:.6f}
- Complete-row residual PR: {complete:.6f}
- Observed-cell additive-WLS residual PR with zero residual completion: {missing_aware:.6f}
- Equal-N random-support residual PR median and central 95% range:
  {control['median']:.6f} [{control['q025']:.6f}, {control['q975']:.6f}].
- Complete-case minus random-support mean: {control['observed_complete_case_minus_random_mean']:.6f}

The complete-row and mean-imputed-restricted surfaces are exactly identical.
Accordingly, their difference from the primary surface is caused by ligand-support
selection, not by imputed values surviving on those rows. Chemical differences are
reported as raw shifts, standardized mean differences, robust median/IQR shifts,
and Cliff's delta; no hypothesis tests are used.

The observed-cell additive sensitivity fits row and target effects only to observed
cells and completes unobserved residuals with zero. It is not an estimate of the
missing raw docking scores.

## Reproduce

```bash
.venv/bin/python analysis/public_docking44_missing_data_sensitivity.py \\
  --random-controls {control['repetitions']} \\
  --output-dir results/public_docking44_missing_data_sensitivity
.venv/bin/python -m pytest -q \\
  analysis/test_public_docking44_missing_data_sensitivity.py
```
"""


def write_artifact(
    output: Path,
    summary: dict[str, Any],
    tables: dict[str, pd.DataFrame],
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "summary.json", summary)
    for filename, frame in tables.items():
        frame.to_csv(
            output / filename,
            index=False,
            float_format="%.17g",
            na_rep="",
        )
    (output / "README.md").write_text(build_readme(summary), encoding="utf-8")
    checksums = {
        path.name: sha256_file(path)
        for path in sorted(output.iterdir())
        if path.is_file() and path.name != "output_checksums.json"
    }
    write_json(
        output / "output_checksums.json",
        {
            "algorithm": "sha256",
            "scope": "all output files except this checksum manifest",
            "files": checksums,
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--random-controls", type=int, default=DEFAULT_RANDOM_CONTROLS
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.random_controls < 2:
        raise ValueError("random-controls must be at least two")
    loaded = load_input(args.input)
    summary, tables = analyze(
        loaded,
        random_controls=args.random_controls,
        seed=args.seed,
    )
    summary["input"] = {
        "path": str(args.input.resolve().relative_to(PACKAGE)),
        "sha256": sha256_file(args.input),
    }
    write_artifact(args.output_dir, summary, tables)
    print(args.output_dir / "summary.json")


if __name__ == "__main__":
    main()
