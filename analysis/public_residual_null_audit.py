#!/usr/bin/env python3
"""Build a standalone public-only audit of residual-spectrum nulls.

The producer reads only the two frozen row-level Vina score tables.  It does not
read or copy any historical evidence ledger.  For each matrix, one deterministic
12,000-ligand support is shared by the additive Gaussian, empirical residual-column
permutation, row-norm-preserving random-direction, and molecular-weight-conditional
permutation nulls.  The conditional null preserves each target's empirical processed
raw-score distribution within deterministic molecular-weight bins at the permutation
step before row centring.
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
from rdkit.Chem import Descriptors

PACKAGE = Path(__file__).resolve().parents[1]
FROZEN = PACKAGE / "data" / "frozen"
DEFAULT_OUTPUT = PACKAGE / "results" / "public_residual_null_audit"
DEFAULT_SAMPLE_SIZE = 12_000
DEFAULT_REPEATS = 500
DATASET_SEEDS = {"docking44": 60, "dockstring58": 61}
MW_BIN_COUNTS = (1, 5, 10, 20)
PRIMARY_MW_BIN_COUNT = 10
DOCK44 = [
    "1m2z", "1pbq", "1xoq", "2rh1", "2vt4", "2ydo", "2z5x", "3b66",
    "3kk6", "3ln1", "3rze", "4djh", "4ey7", "4iar", "4mqs", "4n6h",
    "5cxv", "5i71", "5tvn", "5u09", "5va1", "6cm4", "6kpf", "6kux",
    "6lqa", "6pdj", "6x3x", "6y1z", "7f8y", "7kwe", "7ljd", "7wc9",
    "7xnk", "7ym8", "8e9y", "8ef6", "8fhs", "8pjk", "8st0", "8wty",
    "8xvk", "8yn3", "9eo4", "V1A",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
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
        raise ValueError("matrix must be a finite two-dimensional array")
    return (
        matrix
        - matrix.mean(axis=0, keepdims=True)
        - matrix.mean(axis=1, keepdims=True)
        + matrix.mean()
    )


def participation_ratio(matrix: np.ndarray) -> float:
    matrix = np.asarray(matrix, dtype=np.float64)
    standard_deviations = matrix.std(axis=0, ddof=1)
    if np.any(standard_deviations <= 1e-12):
        raise ValueError("target column has negligible variance")
    standardized = (matrix - matrix.mean(axis=0)) / standard_deviations
    correlation = standardized.T @ standardized / (len(standardized) - 1)
    correlation = (correlation + correlation.T) / 2.0
    eigenvalues = np.clip(np.linalg.eigvalsh(correlation), 0.0, None)
    return float(eigenvalues.sum() ** 2 / np.square(eigenvalues).sum())


def target_correlation(matrix: np.ndarray) -> np.ndarray:
    """Return the target Pearson-correlation map after row centring."""

    matrix = np.asarray(matrix, dtype=np.float64)
    row_centered = matrix - matrix.mean(axis=1, keepdims=True)
    standardized = (
        row_centered - row_centered.mean(axis=0, keepdims=True)
    ) / row_centered.std(axis=0, ddof=1, keepdims=True)
    correlation = standardized.T @ standardized / (len(standardized) - 1)
    correlation = np.clip((correlation + correlation.T) / 2.0, -1.0, 1.0)
    np.fill_diagonal(correlation, 1.0)
    return correlation


def map_summary(matrix: np.ndarray, observed_map: np.ndarray | None = None) -> dict[str, float]:
    """Summarise one residual target map without thresholding its edges."""

    correlation = target_correlation(matrix)
    edges = correlation[np.triu_indices(len(correlation), k=1)]
    eigenvalues = np.clip(np.linalg.eigvalsh(correlation), 0.0, None)
    result = {
        "participation_ratio": float(
            eigenvalues.sum() ** 2 / np.square(eigenvalues).sum()
        ),
        "pc1_fraction": float(eigenvalues[-1] / eigenvalues.sum()),
        "mean_off_diagonal_correlation": float(np.mean(edges)),
        "mean_absolute_off_diagonal_correlation": float(np.mean(np.abs(edges))),
        "rms_off_diagonal_correlation": float(np.sqrt(np.mean(np.square(edges)))),
        "maximum_absolute_off_diagonal_correlation": float(np.max(np.abs(edges))),
    }
    if observed_map is not None:
        observed_edges = observed_map[
            np.triu_indices(len(observed_map), k=1)
        ]
        result["spearman_to_observed_map"] = float(
            pd.Series(edges).corr(pd.Series(observed_edges), method="spearman")
        )
        result["edge_sign_agreement_to_observed_map"] = float(
            np.mean(np.sign(edges) == np.sign(observed_edges))
        )
        for count in (10, 25):
            effective_count = min(count, len(edges))
            observed_top = np.argpartition(
                np.abs(observed_edges), -effective_count
            )[-effective_count:]
            simulated_top = np.argpartition(np.abs(edges), -effective_count)[
                -effective_count:
            ]
            intersection = len(set(observed_top.tolist()) & set(simulated_top.tolist()))
            result[f"top_absolute_edge_jaccard_{count}"] = float(
                intersection / (2 * effective_count - intersection)
            )
    return result


def molecular_weights(smiles: pd.Series, source_name: str) -> np.ndarray:
    """Compute RDKit average molecular weights from the released row SMILES."""

    values = np.empty(len(smiles), dtype=np.float64)
    for position, text in enumerate(smiles.astype(str)):
        molecule = Chem.MolFromSmiles(text)
        if molecule is None:
            raise ValueError(f"{source_name}: invalid SMILES at row {position}")
        values[position] = Descriptors.MolWt(molecule)
    if not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError(f"{source_name}: invalid molecular weights")
    return values


def molecular_weight_bins(
    weights: np.ndarray, requested_bins: int
) -> tuple[np.ndarray, dict[str, Any]]:
    """Make deterministic quantile-edge bins while keeping exact MW ties together."""

    weights = np.asarray(weights, dtype=np.float64)
    if weights.ndim != 1 or len(weights) < requested_bins or not np.isfinite(weights).all():
        raise ValueError("molecular weights must be a finite one-dimensional array")
    probabilities = np.linspace(0.0, 1.0, requested_bins + 1)
    quantiles = np.quantile(weights, probabilities)
    internal_edges = np.unique(quantiles[1:-1])
    labels = np.searchsorted(internal_edges, weights, side="left").astype(np.int64)
    sizes = np.bincount(labels, minlength=len(internal_edges) + 1)
    if np.any(sizes < 2):
        raise ValueError("molecular-weight bin contains fewer than two ligands")
    return labels, {
        "requested_bins": int(requested_bins),
        "actual_bins": int(len(sizes)),
        "internal_edges_g_per_mol": internal_edges.tolist(),
        "bin_sizes": sizes.tolist(),
        "minimum_bin_size": int(sizes.min()),
        "maximum_bin_size": int(sizes.max()),
        "bin_label_vector_sha256": hashlib.sha256(
            np.asarray(labels, dtype="<i2").tobytes()
        ).hexdigest(),
        "bin_label_digest_contract": (
            "sha256 support-order bin labels encoded as little-endian int16"
        ),
        "bins": [
            {
                "bin_index": int(label),
                "ligands": int(np.sum(labels == label)),
                "mw_minimum_g_per_mol": float(weights[labels == label].min()),
                "mw_median_g_per_mol": float(np.median(weights[labels == label])),
                "mw_maximum_g_per_mol": float(weights[labels == label].max()),
            }
            for label in range(len(sizes))
        ],
        "tie_contract": (
            "quantile edges are deduplicated and np.searchsorted(side='left') "
            "assigns every exactly tied RDKit MolWt value to one bin"
        ),
    }


def molecular_weight_conditional_permutation_null(
    matrix: np.ndarray,
    weights: np.ndarray,
    *,
    sample_size: int,
    repeats: int,
    seed: int,
    requested_bins: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Permute each processed raw target column within fixed MW bins."""

    support, rng, support_selection = fixed_ligand_support_indices(
        len(matrix), sample_size, seed
    )
    work = np.asarray(matrix, dtype=np.float64)[support]
    support_weights = np.asarray(weights, dtype=np.float64)[support]
    residual = two_way_center(work)
    observed_map = target_correlation(residual)
    observed_summary = map_summary(residual)
    labels, bin_metadata = molecular_weight_bins(support_weights, requested_bins)
    bin_indices = [np.flatnonzero(labels == label) for label in np.unique(labels)]
    records: list[dict[str, Any]] = []
    designs = ("molecular_weight_quantile",) if requested_bins == 1 else (
        "molecular_weight_quantile",
        "size_matched_random_partition",
    )
    for repetition in range(repeats):
        for design in designs:
            design_labels = labels
            if design == "size_matched_random_partition":
                design_labels = labels[rng.permutation(len(labels))]
            design_indices = [
                np.flatnonzero(design_labels == label)
                for label in np.unique(design_labels)
            ]
            permuted = np.empty_like(work)
            for column in range(work.shape[1]):
                for indices in design_indices:
                    permuted[indices, column] = work[
                        rng.permutation(indices), column
                    ]
            simulated = permuted - permuted.mean(axis=1, keepdims=True)
            records.append(
                {
                    "bin_design": design,
                    "requested_mw_bins": int(requested_bins),
                    "repetition": int(repetition),
                    **map_summary(simulated, observed_map),
                }
            )
    table = pd.DataFrame.from_records(records)
    metrics = [
        "participation_ratio",
        "pc1_fraction",
        "mean_off_diagonal_correlation",
        "mean_absolute_off_diagonal_correlation",
        "rms_off_diagonal_correlation",
        "maximum_absolute_off_diagonal_correlation",
        "spearman_to_observed_map",
        "edge_sign_agreement_to_observed_map",
        "top_absolute_edge_jaccard_10",
        "top_absolute_edge_jaccard_25",
    ]
    distributions = {
        design: {
            metric: describe_distribution(
                table.loc[table.bin_design.eq(design), metric].to_numpy()
            )
            for metric in metrics
        }
        for design in designs
    }
    primary_values = table.loc[
        table.bin_design.eq("molecular_weight_quantile"), "participation_ratio"
    ].to_numpy()
    return {
        "sample_n": int(len(work)),
        "n_targets": int(work.shape[1]),
        "repeats": int(repeats),
        "support_selection": support_selection,
        "molecular_weight": {
            "definition": "RDKit Descriptors.MolWt from the released row SMILES",
            "rdkit_version": rdBase.rdkitVersion,
            "support_minimum_g_per_mol": float(support_weights.min()),
            "support_median_g_per_mol": float(np.median(support_weights)),
            "support_maximum_g_per_mol": float(support_weights.max()),
            **bin_metadata,
        },
        "observed": observed_summary,
        "null_distributions_by_bin_design": distributions,
        "empirical_lower_tail_p_for_residual_pr": float(
            (1 + np.sum(primary_values <= observed_summary["participation_ratio"]))
            / (repeats + 1)
        ),
        "model": (
            "independently permute each processed raw target-score column within "
            "deterministic molecular-weight bins, then row-centre and target-scale"
        ),
        "preserves_at_permutation_step": (
            "fixed ligand support and each target's exact empirical raw-score marginal "
            "within each molecular-weight bin, including bin-specific location, scale "
            "and non-Gaussian shape"
        ),
        "does_not_preserve": (
            "continuous within-bin molecular-weight trends, cross-target correspondence, "
            "or ligand main effects and residual row norms; row centring changes target "
            "marginals. The size-matched random partition preserves only bin sizes."
        ),
    }, table


def describe_distribution(values: list[float] | np.ndarray) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or len(array) < 1 or not np.isfinite(array).all():
        raise ValueError("distribution must contain finite values")
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


def fixed_ligand_support_indices(
    n_rows: int,
    sample_size: int,
    seed: int,
) -> tuple[np.ndarray, np.random.Generator, dict[str, Any]]:
    if n_rows < 1 or sample_size < 1:
        raise ValueError("support dimensions must be positive")
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
        "digest_contract": (
            "sha256(sorted zero-based row indices encoded as little-endian int64)"
        ),
    }
    return np.asarray(support, dtype=np.int64), rng, metadata


def additive_main_effect_null(
    matrix: np.ndarray,
    *,
    sample_size: int,
    repeats: int,
    seed: int,
) -> dict[str, Any]:
    support, rng, support_selection = fixed_ligand_support_indices(
        len(matrix), sample_size, seed
    )
    work = np.asarray(matrix, dtype=np.float64)[support]
    grand = float(work.mean())
    ligand_effect = work.mean(axis=1) - grand
    target_effect = work.mean(axis=0) - grand
    residual = two_way_center(work)
    residual_sd = residual.std(axis=0, ddof=1)
    raw_values = np.empty(repeats, dtype=np.float64)
    residual_values = np.empty(repeats, dtype=np.float64)
    for repetition in range(repeats):
        noise = rng.normal(0.0, residual_sd, size=work.shape)
        simulated = (
            grand + ligand_effect[:, None] + target_effect[None, :] + noise
        )
        raw_values[repetition] = participation_ratio(simulated)
        residual_values[repetition] = participation_ratio(two_way_center(simulated))
    observed_raw = participation_ratio(work)
    observed_residual = participation_ratio(residual)
    return {
        "sample_n": int(len(work)),
        "n_targets": int(work.shape[1]),
        "repeats": int(repeats),
        "support_selection": support_selection,
        "observed": {
            "raw": observed_raw,
            "residual": observed_residual,
            "difference_residual_minus_raw": observed_residual - observed_raw,
            "ratio_residual_over_raw": observed_residual / observed_raw,
        },
        "null_raw": describe_distribution(raw_values),
        "null_residual": describe_distribution(residual_values),
        "empirical_lower_tail_p_for_residual_pr": float(
            (1 + np.sum(residual_values <= observed_residual)) / (repeats + 1)
        ),
        "model": (
            "fitted grand, ligand and target effects plus independent Gaussian noise "
            "with the observed target-specific residual standard deviations"
        ),
    }


def empirical_residual_permutation_null(
    matrix: np.ndarray,
    *,
    sample_size: int,
    repeats: int,
    seed: int,
) -> dict[str, Any]:
    support, rng, support_selection = fixed_ligand_support_indices(
        len(matrix), sample_size, seed
    )
    work = np.asarray(matrix, dtype=np.float64)[support]
    residual = two_way_center(work)
    observed_residual = participation_ratio(residual)
    null_values = np.empty(repeats, dtype=np.float64)
    for repetition in range(repeats):
        permuted = np.column_stack(
            [rng.permutation(residual[:, column]) for column in range(residual.shape[1])]
        )
        null_values[repetition] = participation_ratio(two_way_center(permuted))
    return {
        "sample_n": int(len(work)),
        "n_targets": int(work.shape[1]),
        "repeats": int(repeats),
        "support_selection": support_selection,
        "observed_residual": observed_residual,
        "null_residual": describe_distribution(null_values),
        "empirical_lower_tail_p_for_residual_pr": float(
            (1 + np.sum(null_values <= observed_residual)) / (repeats + 1)
        ),
        "model": (
            "independently permute each observed two-way-centred residual target "
            "column across ligands, then repeat two-way centring and target scaling"
        ),
        "preserves_at_permutation_step": (
            "the empirical marginal distribution of every target residual"
        ),
        "does_not_preserve": (
            "cross-target residual correspondence or ligand-specific residual scale"
        ),
    }


def row_norm_preserving_residual_null(
    matrix: np.ndarray,
    *,
    sample_size: int,
    repeats: int,
    seed: int,
) -> dict[str, Any]:
    support, rng, support_selection = fixed_ligand_support_indices(
        len(matrix), sample_size, seed
    )
    work = np.asarray(matrix, dtype=np.float64)[support]
    residual = two_way_center(work)
    row_norms = np.linalg.norm(residual, axis=1)
    observed_residual = participation_ratio(residual)
    null_values = np.empty(repeats, dtype=np.float64)
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
        null_values[repetition] = participation_ratio(simulated)
    return {
        "sample_n": int(len(work)),
        "n_targets": int(work.shape[1]),
        "repeats": int(repeats),
        "support_selection": support_selection,
        "observed_residual": observed_residual,
        "null_residual": describe_distribution(null_values),
        "empirical_lower_tail_p_for_residual_pr": float(
            (1 + np.sum(null_values <= observed_residual)) / (repeats + 1)
        ),
        "model": (
            "independent random directions in the target-zero-sum subspace, each "
            "rescaled to its observed ligand-specific residual Euclidean norm"
        ),
        "preserves_exactly": (
            "fixed ligand support, target count, zero row sums and every residual row norm"
        ),
        "does_not_preserve": (
            "target residual marginals, target variances or cross-target alignment"
        ),
    }


def load_row_level_matrices(
    docking_path: Path,
    dockstring_path: Path,
) -> tuple[
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, dict[str, Any]],
]:
    """Load exactly the two frozen row-level score tables and preprocess them."""
    docking_frame = pd.read_csv(
        docking_path,
        usecols=DOCK44 + ["Cleaned SMILES", "Canonical SMILES"],
    )
    docking_numeric = docking_frame[DOCK44].apply(pd.to_numeric, errors="coerce")
    docking_missing = int(docking_numeric.isna().to_numpy().sum())
    docking_positive = int((docking_numeric.to_numpy(dtype=float) > 0).sum())
    docking_clipped = docking_numeric.clip(upper=0)
    docking_matrix = docking_clipped.fillna(docking_clipped.mean()).to_numpy(
        dtype=np.float64
    )
    docking_smiles = docking_frame["Cleaned SMILES"].fillna(
        docking_frame["Canonical SMILES"]
    )
    docking_weights = molecular_weights(docking_smiles, "docking44")

    dockstring_frame = pd.read_csv(dockstring_path, sep="\t")
    dockstring_targets = [
        column
        for column in dockstring_frame.columns
        if column not in {"inchikey", "smiles"}
    ]
    dockstring_numeric = dockstring_frame[dockstring_targets].apply(
        pd.to_numeric, errors="coerce"
    )
    dockstring_missing = int(dockstring_numeric.isna().to_numpy().sum())
    complete = ~dockstring_numeric.isna().any(axis=1)
    dockstring_complete = dockstring_numeric.loc[complete].to_numpy(dtype=np.float64)
    dockstring_positive = int((dockstring_complete > 0).sum())
    dockstring_matrix = np.minimum(dockstring_complete, 0.0)
    dockstring_weights = molecular_weights(
        dockstring_frame.loc[complete, "smiles"].reset_index(drop=True),
        "dockstring58",
    )

    matrices = {
        "docking44": docking_matrix,
        "dockstring58": dockstring_matrix,
    }
    weights = {
        "docking44": docking_weights,
        "dockstring58": dockstring_weights,
    }
    for name, matrix in matrices.items():
        if matrix.ndim != 2 or min(matrix.shape) < 3 or not np.isfinite(matrix).all():
            raise ValueError(f"{name}: invalid processed score matrix")
    preprocessing = {
        "docking44": {
            "source_rows": int(len(docking_numeric)),
            "targets": int(len(DOCK44)),
            "target_order": list(DOCK44),
            "missing_cells_before_preprocessing": docking_missing,
            "positive_cells_clipped_to_zero": docking_positive,
            "rule": "clip positive scores to zero, then target-mean impute missing cells",
        },
        "dockstring58": {
            "source_rows": int(len(dockstring_numeric)),
            "complete_rows_used_as_sampling_frame": int(complete.sum()),
            "rows_excluded_for_any_missing_target": int((~complete).sum()),
            "targets": int(len(dockstring_targets)),
            "target_order": dockstring_targets,
            "missing_cells_before_complete-case_filter": dockstring_missing,
            "positive_cells_clipped_to_zero_on_complete_rows": dockstring_positive,
            "rule": "retain complete target rows, then clip positive scores to zero",
        },
    }
    return matrices, weights, preprocessing


def analyze_matrix(
    matrix: np.ndarray,
    weights: np.ndarray,
    *,
    sample_size: int,
    repeats: int,
    seed: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Run all nulls and enforce their identical-support contract."""
    common = {"sample_size": sample_size, "repeats": repeats, "seed": seed}
    additive = additive_main_effect_null(matrix, **common)
    empirical = empirical_residual_permutation_null(matrix, **common)
    row_norm = row_norm_preserving_residual_null(matrix, **common)
    conditional_blocks: dict[str, Any] = {}
    conditional_tables: list[pd.DataFrame] = []
    for bin_count in MW_BIN_COUNTS:
        block, table = molecular_weight_conditional_permutation_null(
            matrix,
            weights,
            requested_bins=bin_count,
            **common,
        )
        conditional_blocks[f"mw_{bin_count}_bins"] = block
        conditional_tables.append(table)
    observed_pr = conditional_blocks["mw_1_bins"]["observed"][
        "participation_ratio"
    ]
    unstratified_median = conditional_blocks["mw_1_bins"][
        "null_distributions_by_bin_design"
    ]["molecular_weight_quantile"]["participation_ratio"]["median"]
    observed_gap = unstratified_median - observed_pr
    if observed_gap <= 0:
        raise RuntimeError("unstratified raw-score permutation is not above observed PR")
    for block in conditional_blocks.values():
        comparisons: dict[str, Any] = {}
        for design, distributions in block[
            "null_distributions_by_bin_design"
        ].items():
            pr_distribution = distributions["participation_ratio"]
            comparisons[design] = {
                "analytic_row_centered_independence_pr": int(matrix.shape[1] - 1),
                "observed_minus_null_mean_in_null_sd": float(
                    (observed_pr - pr_distribution["mean"])
                    / pr_distribution["standard_deviation"]
                ),
                "explained_share_of_observed_pr_gap_relative_to_one_bin": float(
                    (unstratified_median - pr_distribution["median"])
                    / observed_gap
                ),
            }
        block["comparisons"] = comparisons
    blocks = {
        "additive_gaussian": additive,
        "empirical_residual_column_permutation": empirical,
        "row_norm_preserving_random_direction": row_norm,
        "molecular_weight_conditional_permutation": conditional_blocks,
    }
    support_hashes = {
        block["support_selection"]["support_index_sha256"]
        for block in (
            additive,
            empirical,
            row_norm,
            *conditional_blocks.values(),
        )
    }
    if len(support_hashes) != 1:
        raise RuntimeError("residual nulls did not use one identical ligand support")
    observed = np.asarray(
        [
            additive["observed"]["residual"],
            empirical["observed_residual"],
            row_norm["observed_residual"],
            *(
                block["observed"]["participation_ratio"]
                for block in conditional_blocks.values()
            ),
        ],
        dtype=float,
    )
    if np.ptp(observed) > 1e-12:
        raise RuntimeError("observed residual PR differs across the shared-support nulls")
    result = {
        "full_processed_shape": [int(value) for value in matrix.shape],
        "support_selection": additive["support_selection"],
        "observed_residual_pr": float(observed[0]),
        "nulls": blocks,
    }
    table = pd.concat(conditional_tables, ignore_index=True)
    return result, table


def build_artifact(
    matrices: dict[str, np.ndarray],
    molecular_weight_arrays: dict[str, np.ndarray],
    preprocessing: dict[str, dict[str, Any]],
    *,
    sample_size: int,
    repeats: int,
    input_records: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], pd.DataFrame]:
    if sample_size < 3 or repeats < 2:
        raise ValueError("sample size must be >=3 and repeats must be >=2")
    dataset_reports: dict[str, Any] = {}
    repeat_tables: list[pd.DataFrame] = []
    for name in ("docking44", "dockstring58"):
        report, repeats_table = analyze_matrix(
            matrices[name],
            molecular_weight_arrays[name],
            sample_size=sample_size,
            repeats=repeats,
            seed=DATASET_SEEDS[name],
        )
        report["preprocessing"] = preprocessing[name]
        dataset_reports[name] = report
        repeats_table.insert(0, "dataset", name)
        repeat_tables.append(repeats_table)
    payload = {
        "schema_version": "1.0.0",
        "analysis_status": "standalone_public_reproducibility_artifact",
        "producer": "analysis/public_residual_null_audit.py",
        "inputs": input_records,
        "configuration": {
            "sample_size_per_dataset": int(sample_size),
            "repetitions_per_null": int(repeats),
            "dataset_seeds": DATASET_SEEDS,
            "support_contract": (
                "within each dataset all nulls use the same deterministic simple "
                "random support without replacement and expose its index SHA-256"
            ),
            "molecular_weight_bin_counts": list(MW_BIN_COUNTS),
            "primary_molecular_weight_bin_count": PRIMARY_MW_BIN_COUNT,
        },
        "datasets": dataset_reports,
        "claim_boundary": (
            "These nulls test whether residual spectral concentration can be explained "
            "by an additive independent-noise surface, target-wise empirical residual "
            "marginals without cross-target correspondence, or ligand-specific residual "
            "row magnitude without target alignment, or target-specific residual "
            "marginals conditional on coarse molecular-weight bins without cross-target "
            "alignment. They do not establish biological "
            "correctness, docking quality, or transport beyond the two fixed Vina panels."
        ),
    }
    return payload, pd.concat(repeat_tables, ignore_index=True)


def write_artifact(
    output: Path, payload: dict[str, Any], conditional_repeats: pd.DataFrame
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    summary_path = output / "summary.json"
    write_json(summary_path, payload)
    conditional_repeats.to_csv(
        output / "mw_conditional_null_repeats.csv",
        index=False,
        float_format="%.17g",
    )
    bin_rows: list[dict[str, Any]] = []
    for dataset, report in payload["datasets"].items():
        for key, block in report["nulls"][
            "molecular_weight_conditional_permutation"
        ].items():
            metadata = block["molecular_weight"]
            for record in metadata["bins"]:
                bin_rows.append(
                    {
                        "dataset": dataset,
                        "null_key": key,
                        "requested_mw_bins": metadata["requested_bins"],
                        "actual_mw_bins": metadata["actual_bins"],
                        "support_index_sha256": block["support_selection"][
                            "support_index_sha256"
                        ],
                        **record,
                    }
                )
    pd.DataFrame.from_records(bin_rows).to_csv(
        output / "mw_conditional_bin_contract.csv",
        index=False,
        float_format="%.17g",
    )
    readme = """# Public residual-null audit

This standalone artifact is produced directly from the two frozen row-level Vina
score tables. It does not consume any historical evidence ledger. All nulls
for a dataset share one fingerprinted deterministic ligand support.

Reproduce with:

```bash
.venv/bin/python analysis/public_residual_null_audit.py \\
  --sample-size 12000 --repeats 500 \\
  --output-dir results/public_residual_null_audit
.venv/bin/python -m pytest -q analysis/test_public_residual_null_audit.py
```
"""
    (output / "README.md").write_text(readme, encoding="utf-8")
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
    parser.add_argument(
        "--docking44",
        type=Path,
        default=FROZEN / "df_final_v4.csv.gz",
    )
    parser.add_argument(
        "--dockstring58",
        type=Path,
        default=FROZEN / "dockstring-dataset.tsv.gz",
    )
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    matrices, weights, preprocessing = load_row_level_matrices(
        args.docking44, args.dockstring58
    )
    inputs = {
        "docking44": {
            "path": str(args.docking44.resolve().relative_to(PACKAGE)),
            "sha256": sha256_file(args.docking44),
        },
        "dockstring58": {
            "path": str(args.dockstring58.resolve().relative_to(PACKAGE)),
            "sha256": sha256_file(args.dockstring58),
        },
    }
    payload, conditional_repeats = build_artifact(
        matrices,
        weights,
        preprocessing,
        sample_size=args.sample_size,
        repeats=args.repeats,
        input_records=inputs,
    )
    write_artifact(args.output_dir, payload, conditional_repeats)
    for name, report in payload["datasets"].items():
        medians = {
            "additive_gaussian": report["nulls"]["additive_gaussian"]["null_residual"]["median"],
            "empirical_residual_column_permutation": report["nulls"]["empirical_residual_column_permutation"]["null_residual"]["median"],
            "row_norm_preserving_random_direction": report["nulls"]["row_norm_preserving_random_direction"]["null_residual"]["median"],
            "mw_conditional_primary": report["nulls"]["molecular_weight_conditional_permutation"][f"mw_{PRIMARY_MW_BIN_COUNT}_bins"]["null_distributions_by_bin_design"]["molecular_weight_quantile"]["participation_ratio"]["median"],
        }
        print(
            f"{name}: observed residual PR={report['observed_residual_pr']:.6f}; "
            f"null medians={medians}; support="
            f"{report['support_selection']['support_index_sha256']}"
        )
    print(args.output_dir / "summary.json")


if __name__ == "__main__":
    main()
