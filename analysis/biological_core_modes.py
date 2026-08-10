#!/usr/bin/env python3
"""Outcome-blind truncation of the DOCKSTRING residual target geometry.

This science-only analysis asks whether the leading residual docking modes form a
more transferable target-relationship geometry than the complete residual
correlation matrix.  The mode count is selected from docking scores alone as the
smallest number of eigenvalues explaining at least 50% of residual correlation
trace.  Experimental panels are never passed to the selection function.

The analysis is explicitly post hoc.  The 50% rule and the low-mode hypothesis
were formulated after prior inspection of these datasets, so the so-called
validation panels below are locked/no-retuning evaluations, not prospective
confirmatory data.
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
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from scipy import stats

try:  # Direct script execution and package-style import are both supported.
    from . import dense_davis_benchmark as davis
    from . import kirhub_external_validation as kirhub
    from . import replicated_pair_retrieval as retrieval
    from . import residual_target_geometry_validation as geometry
except ImportError:  # pragma: no cover - direct CLI execution.
    import dense_davis_benchmark as davis  # type: ignore
    import kirhub_external_validation as kirhub  # type: ignore
    import replicated_pair_retrieval as retrieval  # type: ignore
    import residual_target_geometry_validation as geometry  # type: ignore


PACKAGE = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PACKAGE / "results" / "biological_core_modes"
DEFAULT_SEED = 20260808
DEFAULT_VARIANCE_THRESHOLD = 0.50
DEFAULT_SCAFFOLD_FOLDS = 5
DEFAULT_QAP_PERMUTATIONS = 50_000
DEFAULT_RANDOM_SUBSPACES = 20_000
PRIMARY_TOP_FRACTION = 0.10
TARGETS = tuple(kirhub.TARGETS)

PANEL_ROLES = OrderedDict(
    [
        ("PKIS1", "post_hoc_discovery"),
        ("DAVIS", "locked_no_retuning_evaluation"),
        ("PKIS2", "locked_no_retuning_evaluation"),
        ("KiRHub", "locked_no_retuning_evaluation"),
    ]
)


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def holm_adjust(values: dict[str, float]) -> dict[str, float]:
    """Holm-adjust a named family while preserving monotonicity."""
    ordered = sorted(values, key=values.get)
    total = len(ordered)
    adjusted: dict[str, float] = {}
    running = 0.0
    for index, name in enumerate(ordered):
        candidate = min(1.0, (total - index) * float(values[name]))
        running = max(running, candidate)
        adjusted[name] = running
    return adjusted


def residual_correlation_from_covariance(covariance: np.ndarray) -> np.ndarray:
    """Correlation after per-row target centering, using only raw cross-moments."""
    covariance = np.asarray(covariance, dtype=np.float64)
    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
        raise ValueError("covariance must be square")
    if not np.isfinite(covariance).all() or not np.allclose(
        covariance, covariance.T, atol=1e-10
    ):
        raise ValueError("covariance must be finite and symmetric")
    p = len(covariance)
    centering = np.eye(p) - np.ones((p, p), dtype=np.float64) / p
    residual_covariance = centering @ covariance @ centering
    scale = np.sqrt(np.maximum(np.diag(residual_covariance), 0.0))
    if np.any(scale <= 1e-14):
        raise ValueError("row-centered covariance contains a constant target")
    correlation = residual_covariance / np.outer(scale, scale)
    correlation = np.clip((correlation + correlation.T) / 2.0, -1.0, 1.0)
    np.fill_diagonal(correlation, 1.0)
    return correlation


def residual_correlation(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2 or len(matrix) < 3 or not np.isfinite(matrix).all():
        raise ValueError("residual correlation requires a finite ligand x target matrix")
    return residual_correlation_from_covariance(np.cov(matrix, rowvar=False, ddof=1))


def ordered_eigendecomposition(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("eigendecomposition requires a square matrix")
    eigenvalues, eigenvectors = np.linalg.eigh((matrix + matrix.T) / 2.0)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    if eigenvalues[-1] < -1e-7:
        raise ValueError("geometry is not positive semidefinite")
    eigenvalues = np.maximum(eigenvalues, 0.0)
    return eigenvalues, eigenvectors


def docking_only_mode_count(
    correlation: np.ndarray, threshold: float = DEFAULT_VARIANCE_THRESHOLD
) -> dict:
    """Select k from docking alone; no experimental argument exists by design."""
    if not 0 < threshold <= 1:
        raise ValueError("variance threshold must lie in (0, 1]")
    eigenvalues, _ = ordered_eigendecomposition(correlation)
    total = float(eigenvalues.sum())
    if total <= 0:
        raise ValueError("correlation trace is non-positive")
    cumulative = np.cumsum(eigenvalues) / total
    selected = int(np.searchsorted(cumulative, threshold, side="left") + 1)
    return {
        "selected_k": selected,
        "selected_cumulative_variance_fraction": float(cumulative[selected - 1]),
        "previous_cumulative_variance_fraction": (
            float(cumulative[selected - 2]) if selected > 1 else 0.0
        ),
        "threshold": float(threshold),
        "positive_eigenvalues": int(np.sum(eigenvalues > 1e-10)),
        "eigenvalues": eigenvalues,
    }


def truncated_geometry(correlation: np.ndarray, modes: int) -> np.ndarray:
    """Return the unnormalized rank-k spectral kernel.

    The diagonal contains target-specific communality (variance captured by the
    retained modes), so this object is not itself a correlation matrix.  The
    historical function name is retained for compatibility with the exploratory
    calibration scripts; new analyses should label the result explicitly as a
    spectral kernel and call :func:`unit_diagonal_correlation` when a correlation
    geometry is the estimand.
    """
    eigenvalues, eigenvectors = ordered_eigendecomposition(correlation)
    if modes < 1 or modes > len(eigenvalues):
        raise ValueError("invalid number of modes")
    approximation = (
        eigenvectors[:, :modes] * eigenvalues[:modes]
    ) @ eigenvectors[:, :modes].T
    return (approximation + approximation.T) / 2.0


def unit_diagonal_correlation(spectral_kernel: np.ndarray) -> np.ndarray:
    """Normalize a PSD spectral kernel to a unit-diagonal correlation matrix."""
    kernel = np.asarray(spectral_kernel, dtype=np.float64)
    if kernel.ndim != 2 or kernel.shape[0] != kernel.shape[1]:
        raise ValueError("spectral kernel must be square")
    if not np.isfinite(kernel).all() or not np.allclose(
        kernel, kernel.T, atol=1e-10
    ):
        raise ValueError("spectral kernel must be finite and symmetric")
    diagonal = np.diag(kernel)
    if np.any(diagonal <= 1e-14):
        raise ValueError("spectral kernel has a non-positive diagonal entry")
    scale = np.sqrt(diagonal)
    correlation = kernel / np.outer(scale, scale)
    correlation = np.clip((correlation + correlation.T) / 2.0, -1.0, 1.0)
    np.fill_diagonal(correlation, 1.0)
    return correlation


def subspace_stability(
    first: np.ndarray, second: np.ndarray, modes: int
) -> dict[str, float]:
    _, first_vectors = ordered_eigendecomposition(first)
    _, second_vectors = ordered_eigendecomposition(second)
    if modes < 1 or modes > min(first_vectors.shape[1], second_vectors.shape[1]):
        raise ValueError("invalid subspace dimension")
    singular_values = np.linalg.svd(
        first_vectors[:, :modes].T @ second_vectors[:, :modes],
        compute_uv=False,
    )
    singular_values = np.clip(singular_values, 0.0, 1.0)
    return {
        "mean_squared_canonical_correlation": float(np.mean(singular_values**2)),
        "minimum_canonical_correlation": float(np.min(singular_values)),
        "maximum_principal_angle_degrees": float(
            np.degrees(np.arccos(np.min(singular_values)))
        ),
    }


def scaffold_fold_ids(
    group_keys: Iterable[str], folds: int, seed: int
) -> np.ndarray:
    """Assign whole scaffold groups to deterministic, mutually disjoint folds."""
    if folds < 2:
        raise ValueError("at least two scaffold folds are required")
    cache: dict[str, int] = {}
    output: list[int] = []
    for raw in group_keys:
        key = str(raw)
        if key not in cache:
            digest = hashlib.sha256(f"{seed}|{key}".encode("utf-8")).digest()
            cache[key] = int.from_bytes(digest[:8], "big") % folds
        output.append(cache[key])
    return np.asarray(output, dtype=np.int16)


def _dockstring_identity_and_scaffolds(
    smiles: pd.Series, reported_inchikey: pd.Series
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    connectivity: list[str] = []
    scaffolds: list[str] = []
    groups: list[str] = []
    full_key_string_differences = 0
    connectivity_mismatches = 0
    for index, (smiles_value, reported) in enumerate(
        zip(smiles.astype(str), reported_inchikey.astype(str))
    ):
        molecule = Chem.MolFromSmiles(smiles_value)
        if molecule is None:
            raise ValueError(f"invalid DOCKSTRING SMILES at retained row {index}")
        key = Chem.MolToInchiKey(molecule)
        if not key or len(key) < 14:
            raise ValueError(f"could not generate InChIKey at retained row {index}")
        full_key_string_differences += int(key != reported)
        connectivity_mismatches += int(key[:14] != reported[:14])
        block = key[:14]
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=molecule)
        group = f"MURCKO:{scaffold}" if scaffold else f"ACYCLIC:{block}"
        connectivity.append(block)
        scaffolds.append(scaffold)
        groups.append(group)
    return (
        np.asarray(connectivity, dtype=object),
        np.asarray(scaffolds, dtype=object),
        np.asarray(groups, dtype=object),
        int(full_key_string_differences),
        int(connectivity_mismatches),
    )


def _davis_identity() -> pd.DataFrame:
    source = pd.read_csv(
        davis.DEFAULT_DAVIS,
        sep="\t",
        usecols=["drug_name", "compound_iso_smiles"],
    ).drop_duplicates()
    if source.drug_name.duplicated().any() or len(source) != 72:
        raise ValueError("unexpected DAVIS structure table")
    return davis._identity_table(  # noqa: SLF001
        source, "compound_iso_smiles", scan="full"
    )


def load_inputs(
    pkis1_zip: Path, kirhub_workbook: Path
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray], dict]:
    """Load frozen inputs and construct exact-only and scaffold-conservative masks."""
    source = pd.read_csv(davis.DEFAULT_DOCKSTRING, sep="\t")
    score_columns = [
        column for column in source.columns if column not in {"inchikey", "smiles"}
    ]
    complete = ~source[score_columns].isna().any(axis=1)
    retained = source.loc[complete, ["inchikey", "smiles", *TARGETS]].copy()
    if len(retained) != 260_060 or retained[list(TARGETS)].isna().any().any():
        raise ValueError("unexpected complete DOCKSTRING support")
    (
        connectivity,
        scaffolds,
        groups,
        full_key_string_differences,
        connectivity_mismatches,
    ) = (
        _dockstring_identity_and_scaffolds(retained.smiles, retained.inchikey)
    )

    pkis2_frame = geometry.load_pkis2_full()
    pkis1_frame = geometry.load_pkis1_full(Path(pkis1_zip))
    davis_identity = _davis_identity()

    excluded_blocks = set(davis_identity.connectivity_block.dropna())
    excluded_blocks |= set(pkis2_frame.connectivity_block.dropna())
    excluded_blocks |= set(pkis1_frame.connectivity_block.dropna())
    exact_keep = ~np.isin(connectivity, np.asarray(sorted(excluded_blocks), dtype=object))

    experimental_scaffolds: set[str] = set()
    for series in (
        davis_identity.compound_iso_smiles,
        pkis2_frame.Smiles,
        pkis1_frame.SMILES,
    ):
        experimental_scaffolds |= {
            str(value)
            for value in geometry.murcko_scaffold_keys(series)
            if str(value)
        }
    scaffold_keep = exact_keep & ~np.isin(
        scaffolds, np.asarray(sorted(experimental_scaffolds), dtype=object)
    )

    unclipped = retained[list(TARGETS)].to_numpy(dtype=np.float64)
    clipped = np.minimum(unclipped, 0.0)
    reference = clipped[exact_keep]
    conservative_reference = clipped[scaffold_keep]
    reference_groups = groups[exact_keep]

    davis_experiment = retrieval._load_davis_experiment()  # noqa: SLF001
    pkis2_experiment = pkis2_frame[list(TARGETS)].to_numpy(dtype=np.float64)
    pkis1_experiment = pkis1_frame[list(TARGETS)].to_numpy(dtype=np.float64)
    kirhub_frame = kirhub.load_kirhub(Path(kirhub_workbook))
    panels = {
        "PKIS1": pkis1_experiment,
        "DAVIS": davis_experiment[list(TARGETS)].to_numpy(dtype=np.float64),
        "PKIS2": pkis2_experiment,
        "KiRHub": kirhub_frame[list(TARGETS)].to_numpy(dtype=np.float64),
    }
    provenance = {
        "complete_dockstring_rows": int(len(retained)),
        "exact_connectivity_excluded_rows": int(np.sum(~exact_keep)),
        "exact_reference_rows": int(len(reference)),
        "same_cyclic_murcko_additionally_excluded_rows": int(
            np.sum(exact_keep & ~scaffold_keep)
        ),
        "conservative_reference_rows": int(len(conservative_reference)),
        "experimental_cyclic_murcko_scaffolds": int(len(experimental_scaffolds)),
        "reference_scaffold_groups": int(len(set(reference_groups))),
        "reference_cyclic_murcko_groups": int(
            len({value for value in reference_groups if str(value).startswith("MURCKO:")})
        ),
        "reference_acyclic_connectivity_groups": int(
            len({value for value in reference_groups if str(value).startswith("ACYCLIC:")})
        ),
        "reported_vs_recomputed_connectivity_block_mismatches": (
            connectivity_mismatches
        ),
        "reported_vs_recomputed_full_inchikey_string_differences": (
            full_key_string_differences
        ),
        "dockstring_inchikey_boundary": (
            "DOCKSTRING supplied keys systematically use a different suffix from "
            "current RDKit Standard InChIKeys; identity exclusion and acyclic "
            "grouping use the recomputed 14-character connectivity block."
        ),
        "positive_scores_clipped_to_zero": int(np.sum(unclipped > 0)),
        "kirhub_chemical_exclusion_boundary": (
            "KiRHub provides compound names but no structures; exact or scaffold "
            "exclusion against KiRHub was impossible."
        ),
    }
    return reference, conservative_reference, reference_groups, panels, provenance


def panel_endpoint(matrix: np.ndarray) -> np.ndarray:
    return residual_correlation(np.asarray(matrix, dtype=np.float64))


def _candidate_metrics(
    candidate: np.ndarray,
    experimental_geometry: np.ndarray,
    labels: np.ndarray,
) -> dict[str, float]:
    tri = np.triu_indices(len(candidate), k=1)
    retrieval_metrics = retrieval.retrieval_metrics(labels, candidate[tri])
    return {
        "continuous_spearman": float(
            geometry.geometry_concordance(candidate, experimental_geometry)
        ),
        "roc_auc": float(retrieval_metrics["roc_auc"]),
        "average_precision": float(retrieval_metrics["average_precision"]),
    }


def panel_metrics(
    full_geometry: np.ndarray,
    normalized_low_geometry: np.ndarray,
    spectral_kernel: np.ndarray,
    experimental_geometry: np.ndarray,
    top_fraction: float = PRIMARY_TOP_FRACTION,
) -> dict[str, float]:
    tri = np.triu_indices(len(full_geometry), k=1)
    labels = retrieval.upper_tail_labels(experimental_geometry[tri], top_fraction)
    candidates = OrderedDict(
        [
            ("full", full_geometry),
            ("normalized_low_mode", normalized_low_geometry),
            ("spectral_kernel", spectral_kernel),
        ]
    )
    values = {
        name: _candidate_metrics(candidate, experimental_geometry, labels)
        for name, candidate in candidates.items()
    }
    record: dict[str, float] = {
        "positive_pairs": int(np.sum(labels)),
        "total_pairs": int(len(labels)),
    }
    for representation, metrics in values.items():
        for metric, value in metrics.items():
            record[f"{representation}_{metric}"] = value
            if representation != "full":
                record[f"{representation}_minus_full_{metric}"] = (
                    value - values["full"][metric]
                )
    return record


def _metric_from_rank_and_order(
    labels: np.ndarray, ranks: np.ndarray, descending_order: np.ndarray
) -> tuple[float, float]:
    labels = np.asarray(labels, dtype=bool)
    positives = int(labels.sum())
    negatives = int(len(labels) - positives)
    roc_auc = (
        float(ranks[labels].sum()) - positives * (positives + 1) / 2
    ) / (positives * negatives)
    ordered = labels[descending_order].astype(np.int64)
    precision = np.cumsum(ordered) / (np.arange(len(ordered)) + 1)
    average_precision = float(np.sum(ordered * precision) / positives)
    return float(roc_auc), average_precision


def eigenvalue_matched_random_subspace_null(
    full_geometry: np.ndarray,
    normalized_low_geometry: np.ndarray,
    spectral_kernel: np.ndarray,
    panel_geometries: dict[str, np.ndarray],
    modes: int,
    repeats: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Representation-matched Haar nulls for both low-mode score matrices.

    Each draw first preserves the observed leading eigenvalues in a Haar-random
    frame inside the estimable target subspace.  That draw is evaluated both as
    an unnormalized spectral kernel and after the same unit-diagonal
    normalization used for the primary low-mode correlation.
    """
    if repeats < 1:
        raise ValueError("random-subspace repeats must be positive")
    eigenvalues, eigenvectors = ordered_eigendecomposition(full_geometry)
    positive = eigenvalues > 1e-10
    basis = eigenvectors[:, positive]
    if modes >= basis.shape[1]:
        raise ValueError("random-subspace null needs a nontrivial discarded subspace")
    retained_values = eigenvalues[:modes]
    tri = np.triu_indices(len(full_geometry), k=1)
    endpoint_rank_vectors: dict[str, np.ndarray] = {}
    endpoint_norms: dict[str, float] = {}
    labels: dict[str, np.ndarray] = {}
    observed: dict[str, dict[str, float]] = {}
    for name, endpoint in panel_geometries.items():
        endpoint_ranks = stats.rankdata(endpoint[tri], method="average").astype(float)
        endpoint_ranks -= endpoint_ranks.mean()
        endpoint_rank_vectors[name] = endpoint_ranks
        endpoint_norms[name] = float(np.linalg.norm(endpoint_ranks))
        labels[name] = retrieval.upper_tail_labels(
            endpoint[tri], PRIMARY_TOP_FRACTION
        )
        observed[name] = panel_metrics(
            full_geometry,
            normalized_low_geometry,
            spectral_kernel,
            endpoint,
        )

    metrics = ("continuous_spearman", "roc_auc", "average_precision")
    representations = OrderedDict(
        [
            ("normalized_low_mode_correlation", "normalized_low_mode"),
            ("communality_weighted_spectral_kernel", "spectral_kernel"),
        ]
    )
    null = {
        name: {
            representation: {
                metric: np.empty(repeats, dtype=np.float64) for metric in metrics
            }
            for representation in representations
        }
        for name in panel_geometries
    }
    rng = np.random.default_rng(seed)
    for repetition in range(repeats):
        random_matrix = rng.normal(size=(basis.shape[1], modes))
        random_frame, _ = np.linalg.qr(random_matrix, mode="reduced")
        vectors = basis @ random_frame
        random_kernel = (vectors * retained_values) @ vectors.T
        random_candidates = {
            "normalized_low_mode_correlation": unit_diagonal_correlation(
                random_kernel
            ),
            "communality_weighted_spectral_kernel": random_kernel,
        }
        for representation, candidate in random_candidates.items():
            pair_values = candidate[tri]
            ranks = stats.rankdata(pair_values, method="average").astype(float)
            descending = np.argsort(pair_values, kind="mergesort")[::-1]
            centered_ranks = ranks - ranks.mean()
            rank_norm = float(np.linalg.norm(centered_ranks))
            for name in panel_geometries:
                null[name][representation]["continuous_spearman"][repetition] = (
                    float(
                        np.dot(centered_ranks, endpoint_rank_vectors[name])
                        / (rank_norm * endpoint_norms[name])
                    )
                )
                auc, ap = _metric_from_rank_and_order(
                    labels[name], ranks, descending
                )
                null[name][representation]["roc_auc"][repetition] = auc
                null[name][representation]["average_precision"][repetition] = ap

    rows: list[dict] = []
    validation_names = [
        name for name, role in PANEL_ROLES.items() if role.startswith("locked")
    ]
    full_keys = {
        "continuous_spearman": "full_continuous_spearman",
        "roc_auc": "full_roc_auc",
        "average_precision": "full_average_precision",
    }
    for representation, observed_prefix in representations.items():
        observed_keys = {
            metric: f"{observed_prefix}_{metric}" for metric in metrics
        }
        for name in panel_geometries:
            for metric in metrics:
                values = null[name][representation][metric]
                observed_low = observed[name][observed_keys[metric]]
                observed_full = observed[name][full_keys[metric]]
                rows.append(
                    {
                        "representation": representation,
                        "primary_representation": bool(
                            representation == "normalized_low_mode_correlation"
                        ),
                        "panel": name,
                        "analysis_role": PANEL_ROLES[name],
                        "metric": metric,
                        "observed_candidate": observed_low,
                        "observed_full": observed_full,
                        "observed_candidate_minus_full": (
                            observed_low - observed_full
                        ),
                        "random_mean": float(values.mean()),
                        "random_median": float(np.median(values)),
                        "random_interval_95_low": float(
                            np.quantile(values, 0.025)
                        ),
                        "random_interval_95_high": float(
                            np.quantile(values, 0.975)
                        ),
                        "one_sided_p_random_at_least_observed_candidate": float(
                            (1 + np.sum(values >= observed_low)) / (repeats + 1)
                        ),
                        "repeats": int(repeats),
                        "seed": int(seed),
                    }
                )
        for metric in metrics:
            values = np.mean(
                np.vstack(
                    [null[name][representation][metric] for name in validation_names]
                ),
                axis=0,
            )
            observed_low = float(
                np.mean(
                    [
                        observed[name][observed_keys[metric]]
                        for name in validation_names
                    ]
                )
            )
            observed_full = float(
                np.mean(
                    [observed[name][full_keys[metric]] for name in validation_names]
                )
            )
            rows.append(
                {
                    "representation": representation,
                    "primary_representation": bool(
                        representation == "normalized_low_mode_correlation"
                    ),
                    "panel": "LOCKED_VALIDATION_MEAN",
                    "analysis_role": "omnibus_locked_no_retuning_evaluation",
                    "metric": metric,
                    "observed_candidate": observed_low,
                    "observed_full": observed_full,
                    "observed_candidate_minus_full": observed_low - observed_full,
                    "random_mean": float(values.mean()),
                    "random_median": float(np.median(values)),
                    "random_interval_95_low": float(np.quantile(values, 0.025)),
                    "random_interval_95_high": float(np.quantile(values, 0.975)),
                    "one_sided_p_random_at_least_observed_candidate": float(
                        (1 + np.sum(values >= observed_low)) / (repeats + 1)
                    ),
                    "repeats": int(repeats),
                    "seed": int(seed),
                }
            )
    summary = pd.DataFrame(rows)
    spectrum_check = pd.DataFrame(
        {
            "retained_mode": np.arange(1, modes + 1, dtype=int),
            "retained_eigenvalue": retained_values,
        }
    )
    return summary, spectrum_check


def paired_qap_records(
    full_geometry: np.ndarray,
    normalized_low_geometry: np.ndarray,
    spectral_kernel: np.ndarray,
    panel_geometries: dict[str, np.ndarray],
    permutations: int,
    seed: int,
) -> pd.DataFrame:
    candidates = OrderedDict(
        [
            ("normalized_low_mode_correlation", normalized_low_geometry),
            ("communality_weighted_spectral_kernel", spectral_kernel),
        ]
    )
    rows: list[dict] = []
    for representation_index, (representation, candidate) in enumerate(
        candidates.items()
    ):
        representation_seed = seed + representation_index * 100_000
        for panel_index, (name, endpoint) in enumerate(panel_geometries.items()):
            continuous_seed = representation_seed + panel_index
            continuous = geometry.fixed_experimental_geometry_paired_qap(
                full_geometry,
                candidate,
                endpoint,
                permutations,
                continuous_seed,
            )
            rows.append(
                {
                    "representation": representation,
                    "primary_representation": bool(
                        representation == "normalized_low_mode_correlation"
                    ),
                    "panel": name,
                    "analysis_role": PANEL_ROLES[name],
                    "metric": "continuous_spearman",
                    "full": continuous["raw_docking_concordance"],
                    "candidate": continuous["centered_docking_concordance"],
                    "candidate_minus_full": continuous[
                        "centered_minus_raw_docking"
                    ],
                    "paired_target_label_qap_one_sided_p": continuous[
                        "one_sided_p_positive_delta"
                    ],
                    "paired_delta_null_interval_95_low": continuous[
                        "null_interval_95"
                    ][0],
                    "paired_delta_null_interval_95_high": continuous[
                        "null_interval_95"
                    ][1],
                    "permutations": int(permutations),
                    "seed": int(continuous_seed),
                }
            )
            tri = np.triu_indices(len(endpoint), k=1)
            labels = retrieval.upper_tail_labels(
                endpoint[tri], PRIMARY_TOP_FRACTION
            )
            ranked_seed = representation_seed + 10_000 + panel_index
            ranked = retrieval.fixed_endpoint_qap(
                labels,
                full_geometry,
                candidate,
                permutations,
                ranked_seed,
            )
            for metric in ("roc_auc", "average_precision"):
                record = ranked["metrics"][metric]
                rows.append(
                    {
                        "representation": representation,
                        "primary_representation": bool(
                            representation == "normalized_low_mode_correlation"
                        ),
                        "panel": name,
                        "analysis_role": PANEL_ROLES[name],
                        "metric": metric,
                        "full": record["raw_docking"],
                        "candidate": record["centered_docking"],
                        "candidate_minus_full": record["centered_minus_raw"],
                        "paired_target_label_qap_one_sided_p": record[
                            "paired_target_label_qap_p_positive_gain"
                        ],
                        "paired_delta_null_interval_95_low": record[
                            "paired_delta_null_interval_95"
                        ][0],
                        "paired_delta_null_interval_95_high": record[
                            "paired_delta_null_interval_95"
                        ][1],
                        "permutations": int(permutations),
                        "seed": int(ranked_seed),
                    }
                )
    frame = pd.DataFrame(rows)
    validation = frame.analysis_role.str.startswith("locked")
    frame["holm_p_within_representation_9_locked_tests"] = np.nan
    for representation in candidates:
        selected = validation & frame.representation.eq(representation)
        p_values = {
            f"{row.panel}|{row.metric}": float(
                row.paired_target_label_qap_one_sided_p
            )
            for row in frame.loc[selected].itertuples(index=False)
        }
        adjusted = holm_adjust(p_values)
        for index, row in frame.loc[selected].iterrows():
            frame.loc[
                index, "holm_p_within_representation_9_locked_tests"
            ] = adjusted[f"{row.panel}|{row.metric}"]
    all_p_values = {
        f"{row.representation}|{row.panel}|{row.metric}": float(
            row.paired_target_label_qap_one_sided_p
        )
        for row in frame.loc[validation].itertuples(index=False)
    }
    adjusted_all = holm_adjust(all_p_values)
    frame["holm_p_across_both_representations_18_locked_tests"] = np.nan
    for index, row in frame.loc[validation].iterrows():
        frame.loc[
            index, "holm_p_across_both_representations_18_locked_tests"
        ] = adjusted_all[f"{row.representation}|{row.panel}|{row.metric}"]
    return frame


def omnibus_validation_qap(
    full_geometry: np.ndarray,
    normalized_low_geometry: np.ndarray,
    spectral_kernel: np.ndarray,
    panel_geometries: dict[str, np.ndarray],
    permutations: int,
    seed: int,
) -> pd.DataFrame:
    validation = {
        name: endpoint
        for name, endpoint in panel_geometries.items()
        if PANEL_ROLES[name].startswith("locked")
    }
    p = len(full_geometry)
    tri = np.triu_indices(p, k=1)
    endpoint_ranks: dict[str, np.ndarray] = {}
    endpoint_norms: dict[str, float] = {}
    endpoint_labels: dict[str, np.ndarray] = {}
    for name, endpoint in validation.items():
        ranks = stats.rankdata(endpoint[tri], method="average").astype(float)
        ranks -= ranks.mean()
        endpoint_ranks[name] = ranks
        endpoint_norms[name] = float(np.linalg.norm(ranks))
        endpoint_labels[name] = retrieval.upper_tail_labels(
            endpoint[tri], PRIMARY_TOP_FRACTION
        )

    def metrics(matrix: np.ndarray, name: str) -> np.ndarray:
        values = matrix[tri]
        ranks = stats.rankdata(values, method="average").astype(float)
        centered = ranks - ranks.mean()
        rho = float(
            np.dot(centered, endpoint_ranks[name])
            / (np.linalg.norm(centered) * endpoint_norms[name])
        )
        ranked = retrieval.retrieval_metrics(endpoint_labels[name], values)
        return np.asarray([rho, ranked["roc_auc"], ranked["average_precision"]])

    candidates = OrderedDict(
        [
            ("normalized_low_mode_correlation", normalized_low_geometry),
            ("communality_weighted_spectral_kernel", spectral_kernel),
        ]
    )
    observed = {
        representation: np.mean(
            np.vstack(
                [
                    metrics(candidate, name) - metrics(full_geometry, name)
                    for name in validation
                ]
            ),
            axis=0,
        )
        for representation, candidate in candidates.items()
    }
    rng = np.random.default_rng(seed)
    null = {
        representation: np.empty((permutations, 3), dtype=np.float64)
        for representation in candidates
    }
    for repetition in range(permutations):
        order = rng.permutation(p)
        full_permuted = full_geometry[np.ix_(order, order)]
        for representation, candidate in candidates.items():
            candidate_permuted = candidate[np.ix_(order, order)]
            null[representation][repetition] = np.mean(
                np.vstack(
                    [
                        metrics(candidate_permuted, name)
                        - metrics(full_permuted, name)
                        for name in validation
                    ]
                ),
                axis=0,
            )
    rows = []
    metrics_order = ("continuous_spearman", "roc_auc", "average_precision")
    for representation in candidates:
        for index, metric in enumerate(metrics_order):
            values = null[representation][:, index]
            rows.append(
                {
                    "representation": representation,
                    "primary_representation": bool(
                        representation == "normalized_low_mode_correlation"
                    ),
                    "metric": metric,
                    "locked_validation_panels": ",".join(validation),
                    "observed_mean_candidate_minus_full": float(
                        observed[representation][index]
                    ),
                    "one_sided_p_positive_mean_delta": float(
                        (
                            1
                            + np.sum(
                                values >= observed[representation][index]
                            )
                        )
                        / (permutations + 1)
                    ),
                    "null_interval_95_low": float(np.quantile(values, 0.025)),
                    "null_interval_95_high": float(np.quantile(values, 0.975)),
                    "permutations": int(permutations),
                    "seed": int(seed),
                }
            )
    frame = pd.DataFrame(rows)
    frame["holm_p_within_representation_3_metrics"] = np.nan
    for representation in candidates:
        selected = frame.representation.eq(representation)
        adjusted = holm_adjust(
            {
                str(row.metric): float(row.one_sided_p_positive_mean_delta)
                for row in frame.loc[selected].itertuples(index=False)
            }
        )
        for index, row in frame.loc[selected].iterrows():
            frame.loc[
                index, "holm_p_within_representation_3_metrics"
            ] = adjusted[str(row.metric)]
    adjusted_all = holm_adjust(
        {
            f"{row.representation}|{row.metric}": float(
                row.one_sided_p_positive_mean_delta
            )
            for row in frame.itertuples(index=False)
        }
    )
    frame["holm_p_across_both_representations_6_metrics"] = [
        adjusted_all[f"{row.representation}|{row.metric}"]
        for row in frame.itertuples(index=False)
    ]
    return frame


def target_jackknife(
    reference_covariance: np.ndarray,
    panel_covariances: dict[str, np.ndarray],
    threshold: float,
) -> pd.DataFrame:
    rows: list[dict] = []
    p = len(reference_covariance)
    for dropped in range(p):
        keep = np.asarray([index for index in range(p) if index != dropped])
        docking = residual_correlation_from_covariance(
            reference_covariance[np.ix_(keep, keep)]
        )
        selected = docking_only_mode_count(docking, threshold)
        spectral_kernel = truncated_geometry(docking, selected["selected_k"])
        normalized_low = unit_diagonal_correlation(spectral_kernel)
        for name, covariance in panel_covariances.items():
            endpoint = residual_correlation_from_covariance(
                covariance[np.ix_(keep, keep)]
            )
            record = panel_metrics(
                docking, normalized_low, spectral_kernel, endpoint
            )
            rows.append(
                {
                    "dropped_target": TARGETS[dropped],
                    "remaining_targets": int(len(keep)),
                    "selected_k": int(selected["selected_k"]),
                    "selected_cumulative_variance_fraction": selected[
                        "selected_cumulative_variance_fraction"
                    ],
                    "panel": name,
                    "analysis_role": PANEL_ROLES[name],
                    **record,
                }
            )
    return pd.DataFrame(rows)


def summarize_jackknife(frame: pd.DataFrame) -> dict:
    summary: dict[str, dict] = {}
    for panel, part in frame.groupby("panel", sort=False):
        summary[panel] = {
            "selected_k_values": sorted(part.selected_k.unique().astype(int).tolist()),
        }
        for metric in (
            "normalized_low_mode_minus_full_continuous_spearman",
            "normalized_low_mode_minus_full_roc_auc",
            "normalized_low_mode_minus_full_average_precision",
            "spectral_kernel_minus_full_continuous_spearman",
            "spectral_kernel_minus_full_roc_auc",
            "spectral_kernel_minus_full_average_precision",
        ):
            values = part[metric].to_numpy(dtype=float)
            summary[panel][metric] = {
                "positive_drops": int(np.sum(values > 0)),
                "total_drops": int(len(values)),
                "minimum": float(values.min()),
                "median": float(np.median(values)),
                "maximum": float(values.max()),
            }
    return summary


def scaffold_support_records(
    reference: np.ndarray,
    group_keys: np.ndarray,
    panel_geometries: dict[str, np.ndarray],
    folds: int,
    threshold: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fold_ids = scaffold_fold_ids(group_keys, folds, seed)
    full = residual_correlation(reference)
    selected = docking_only_mode_count(full, threshold)
    locked_k = int(selected["selected_k"])
    kernel_full = truncated_geometry(full, locked_k)
    normalized_full = unit_diagonal_correlation(kernel_full)
    support_rows: list[dict] = []
    panel_rows: list[dict] = []
    fold_geometries: dict[int, np.ndarray] = {}
    fold_kernels: dict[int, np.ndarray] = {}
    fold_normalized: dict[int, np.ndarray] = {}
    for fold in range(folds):
        mask = fold_ids == fold
        geometry_fold = residual_correlation(reference[mask])
        fold_geometries[fold] = geometry_fold
        selected_fold = docking_only_mode_count(geometry_fold, threshold)
        kernel_fold = truncated_geometry(
            geometry_fold, selected_fold["selected_k"]
        )
        normalized_fold = unit_diagonal_correlation(kernel_fold)
        fold_kernels[fold] = kernel_fold
        fold_normalized[fold] = normalized_fold
        stability = subspace_stability(full, geometry_fold, locked_k)
        support_rows.append(
            {
                "fold": fold,
                "ligands": int(mask.sum()),
                "scaffold_groups": int(len(set(group_keys[mask]))),
                "selected_k": int(selected_fold["selected_k"]),
                "selected_cumulative_variance_fraction": selected_fold[
                    "selected_cumulative_variance_fraction"
                ],
                "full_geometry_spearman_to_all_rows": geometry.geometry_concordance(
                    geometry_fold, full
                ),
                "normalized_low_mode_correlation_spearman_to_all_rows": geometry.geometry_concordance(
                    normalized_fold, normalized_full
                ),
                "spectral_kernel_spearman_to_all_rows": geometry.geometry_concordance(
                    kernel_fold, kernel_full
                ),
                **stability,
            }
        )
        for panel, endpoint in panel_geometries.items():
            panel_rows.append(
                {
                    "fold": fold,
                    "panel": panel,
                    "analysis_role": PANEL_ROLES[panel],
                    "selected_k": int(selected_fold["selected_k"]),
                    **panel_metrics(
                        geometry_fold,
                        normalized_fold,
                        kernel_fold,
                        endpoint,
                    ),
                }
            )
    pair_rows: list[dict] = []
    for first in range(folds):
        for second in range(first + 1, folds):
            stability = subspace_stability(
                fold_geometries[first], fold_geometries[second], locked_k
            )
            pair_rows.append(
                {
                    "fold_a": first,
                    "fold_b": second,
                    "locked_k": locked_k,
                    "full_geometry_spearman": geometry.geometry_concordance(
                        fold_geometries[first], fold_geometries[second]
                    ),
                    "normalized_low_mode_correlation_spearman": geometry.geometry_concordance(
                        fold_normalized[first], fold_normalized[second]
                    ),
                    "spectral_kernel_spearman": geometry.geometry_concordance(
                        fold_kernels[first], fold_kernels[second]
                    ),
                    **stability,
                }
            )
    return (
        pd.DataFrame(support_rows),
        pd.DataFrame(pair_rows),
        pd.DataFrame(panel_rows),
    )


def _frame_records_with_support(
    support: str,
    full: np.ndarray,
    normalized_low: np.ndarray,
    spectral_kernel: np.ndarray,
    panel_geometries: dict[str, np.ndarray],
    selection: dict,
) -> list[dict]:
    return [
        {
            "reference_support": support,
            "panel": panel,
            "analysis_role": PANEL_ROLES[panel],
            "selected_k": int(selection["selected_k"]),
            "selected_cumulative_variance_fraction": selection[
                "selected_cumulative_variance_fraction"
            ],
            **panel_metrics(
                full, normalized_low, spectral_kernel, endpoint
            ),
        }
        for panel, endpoint in panel_geometries.items()
    ]


def target_pair_geometry_frame(
    full_correlation: np.ndarray,
    normalized_low_correlation: np.ndarray,
    communality_weighted_kernel: np.ndarray,
    targets: Iterable[str] = TARGETS,
) -> pd.DataFrame:
    """Return aligned pair scores for downstream science-only audits."""
    target_order = [str(target) for target in targets]
    matrices = (
        np.asarray(full_correlation, dtype=np.float64),
        np.asarray(normalized_low_correlation, dtype=np.float64),
        np.asarray(communality_weighted_kernel, dtype=np.float64),
    )
    expected = (len(target_order), len(target_order))
    if any(matrix.shape != expected for matrix in matrices):
        raise ValueError("target-pair geometry matrices do not match target order")
    rows: list[dict[str, object]] = []
    for first, target_a in enumerate(target_order):
        for second in range(first + 1, len(target_order)):
            rows.append(
                {
                    "target_a": target_a,
                    "target_b": target_order[second],
                    "full_residual_correlation": float(
                        full_correlation[first, second]
                    ),
                    "normalized_low_mode_correlation": float(
                        normalized_low_correlation[first, second]
                    ),
                    "communality_weighted_spectral_kernel": float(
                        communality_weighted_kernel[first, second]
                    ),
                }
            )
    return pd.DataFrame.from_records(rows)


def write_outputs(
    output: Path,
    frames: dict[str, pd.DataFrame],
    summary: dict,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    output_paths: dict[str, dict[str, str]] = {}
    for stem, frame in frames.items():
        path = output / f"{stem}.csv"
        frame.to_csv(path, index=False, float_format="%.15g")
        output_paths[stem] = {
            "path": path.name,
            "sha256": sha256_file(path),
        }
    summary["outputs"] = output_paths
    summary_path = output / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    readme = f"""# Leading residual docking modes: science-only audit

This artifact was generated by `analysis/biological_core_modes.py`.  It tests
whether the leading DOCKSTRING residual target modes transfer to experimental
kinase target geometry better than the complete residual correlation matrix.

## Two explicitly distinct low-mode representations

The **primary** representation is the rank-k reconstruction after
unit-diagonal renormalization, so it remains a target correlation matrix.  The
secondary unnormalized reconstruction is reported as a
**communality-weighted spectral kernel**: its diagonal is the target-specific
variance captured by the retained modes, and it must not be called a
correlation matrix.  Continuous geometry concordance is primary, target-pair
AUROC is secondary, and average-precision effects are explicitly mixed across
panels rather than a general positive result.

## Frozen analysis boundary

The mode count is selected without an experimental input: it is the smallest
number explaining at least {summary['selection_rule']['variance_threshold']:.0%}
of the residual correlation trace.  The resulting value is
**k={summary['selection_rule']['selected_k']}**
({summary['selection_rule']['selected_cumulative_variance_fraction']:.3f} of
trace).  PKIS1 is labelled discovery; DAVIS, PKIS2, and KiRHub are evaluated
without further retuning.

This is **post-hoc exploratory evidence, not prospective confirmation**.  The
hypothesis and the 50% convention were formulated after these resources had
already been inspected.  The docking-only function prevents numerical endpoint
leakage in this rerun, but it cannot undo historical analyst exposure.  A new
external panel is required for a confirmatory claim.

The Haar random-subspace analysis is representation matched: every
eigenvalue-matched random kernel is evaluated both before and after the same
unit-diagonal normalization.  It tests whether the observed low-mode
orientation is unusual relative to random orientations; the paired QAP files,
not the Haar null, contain the direct candidate-versus-full contrasts.

## Reproduce

```bash
python analysis/biological_core_modes.py \\
  --pkis1-zip /tmp/pkis1_supplement.zip \\
  --kirhub-workbook /tmp/kirhub_supp_tables.xlsx \\
  --output results/biological_core_modes \\
  --qap-permutations {summary['inference']['qap_permutations']} \\
  --random-subspaces {summary['inference']['random_subspaces']} \\
  --seed {summary['inference']['seed']}
```

No compound-level experimental values are written.  The KiRHub workbook is
validated by its frozen SHA256 but is not redistributed under its NC-ND license.
"""
    (output / "README.md").write_text(readme, encoding="utf-8")


def run_analysis(
    *,
    pkis1_zip: Path,
    kirhub_workbook: Path,
    output: Path,
    variance_threshold: float,
    scaffold_folds: int,
    qap_permutations: int,
    random_subspaces: int,
    seed: int,
) -> dict:
    if qap_permutations < 99 or random_subspaces < 99:
        raise ValueError("release analysis requires at least 99 null draws")
    reference, conservative, groups, panels, provenance = load_inputs(
        pkis1_zip, kirhub_workbook
    )
    panel_geometries = {
        name: panel_endpoint(matrix) for name, matrix in panels.items()
    }

    full = residual_correlation(reference)
    selection = docking_only_mode_count(full, variance_threshold)
    spectral_kernel = truncated_geometry(full, selection["selected_k"])
    normalized_low = unit_diagonal_correlation(spectral_kernel)
    conservative_full = residual_correlation(conservative)
    conservative_selection = docking_only_mode_count(
        conservative_full, variance_threshold
    )
    conservative_kernel = truncated_geometry(
        conservative_full, conservative_selection["selected_k"]
    )
    conservative_normalized = unit_diagonal_correlation(conservative_kernel)

    panel_rows = _frame_records_with_support(
        "exact_connectivity_excluded",
        full,
        normalized_low,
        spectral_kernel,
        panel_geometries,
        selection,
    )
    panel_rows += _frame_records_with_support(
        "exact_and_same_cyclic_murcko_excluded",
        conservative_full,
        conservative_normalized,
        conservative_kernel,
        panel_geometries,
        conservative_selection,
    )
    panel_frame = pd.DataFrame(panel_rows)

    support_frame, support_pair_frame, fold_panel_frame = scaffold_support_records(
        reference,
        groups,
        panel_geometries,
        scaffold_folds,
        variance_threshold,
        seed,
    )
    qap_frame = paired_qap_records(
        full,
        normalized_low,
        spectral_kernel,
        panel_geometries,
        qap_permutations,
        seed + 100_000,
    )
    omnibus_frame = omnibus_validation_qap(
        full,
        normalized_low,
        spectral_kernel,
        panel_geometries,
        qap_permutations,
        seed + 200_000,
    )
    random_frame, retained_spectrum = eigenvalue_matched_random_subspace_null(
        full,
        normalized_low,
        spectral_kernel,
        panel_geometries,
        selection["selected_k"],
        random_subspaces,
        seed + 300_000,
    )
    reference_covariance = np.cov(reference, rowvar=False, ddof=1)
    panel_covariances = {
        name: np.cov(matrix, rowvar=False, ddof=1)
        for name, matrix in panels.items()
    }
    jackknife_frame = target_jackknife(
        reference_covariance, panel_covariances, variance_threshold
    )

    eigenvalues = np.asarray(selection["eigenvalues"], dtype=float)
    summary = {
        "analysis": "post-hoc low-order biological core in residual docking geometry",
        "analysis_status": (
            "exploratory_post_hoc_with_locked_no_retuning_evaluation; "
            "not_prospective_confirmation"
        ),
        "strongest_honest_claim": (
            "Under an algorithmically outcome-blind 50%-trace truncation, leading "
            "residual DOCKSTRING modes, renormalized to a unit-diagonal target "
            "correlation, show higher continuous experimental-geometry concordance "
            "and target-pair AUROC than the complete residual correlation on all four "
            "examined kinase panels. Average-precision effects are mixed across "
            "panels and are not a general result. "
            "Because the hypothesis and cutoff were formulated after prior data "
            "inspection, this is a replicated exploratory result that requires a "
            "new external panel for confirmation."
        ),
        "selection_rule": {
            "definition": (
                "smallest k whose leading docking residual-correlation eigenvalues "
                "explain at least the fixed trace fraction"
            ),
            "endpoint_access_by_selection_function": "none",
            "historical_status": (
                "50% convention fixed for this artifact but chosen post hoc after "
                "prior project-wide data inspection"
            ),
            "variance_threshold": float(variance_threshold),
            "selected_k": int(selection["selected_k"]),
            "selected_cumulative_variance_fraction": selection[
                "selected_cumulative_variance_fraction"
            ],
            "previous_cumulative_variance_fraction": selection[
                "previous_cumulative_variance_fraction"
            ],
            "positive_eigenvalues": int(selection["positive_eigenvalues"]),
            "full_eigenvalues": eigenvalues.tolist(),
            "conservative_same_scaffold_exclusion_selected_k": int(
                conservative_selection["selected_k"]
            ),
            "conservative_same_scaffold_exclusion_cumulative_fraction": (
                conservative_selection["selected_cumulative_variance_fraction"]
            ),
        },
        "representation_estimands": {
            "primary": {
                "name": "normalized_low_mode_correlation",
                "definition": (
                    "rank-k spectral reconstruction divided by the square root of "
                    "its diagonal outer product; unit diagonal"
                ),
                "primary_metric": "continuous_spearman",
                "secondary_metric": "roc_auc",
                "mixed_not_general": "average_precision",
            },
            "secondary": {
                "name": "communality_weighted_spectral_kernel",
                "definition": (
                    "unnormalized rank-k reconstruction; its diagonal is the "
                    "target-specific variance captured by the retained modes"
                ),
                "interpretation_boundary": (
                    "a retrieval score that combines angular similarity with "
                    "target communality, not a correlation matrix"
                ),
            },
        },
        "discovery_validation_boundary": {
            "PKIS1": PANEL_ROLES["PKIS1"],
            "DAVIS": PANEL_ROLES["DAVIS"],
            "PKIS2": PANEL_ROLES["PKIS2"],
            "KiRHub": PANEL_ROLES["KiRHub"],
            "qualification": (
                "locked means no tuning after the rule was frozen in this script; "
                "these are not uninspected prospective datasets"
            ),
        },
        "support": provenance,
        "scaffold_disjoint_stability": {
            "folds": int(scaffold_folds),
            "assignment": (
                "SHA256 assignment of whole nonempty Bemis-Murcko groups; acyclic "
                "molecules grouped by recomputed InChIKey connectivity block"
            ),
            "selected_k_values": sorted(
                support_frame.selected_k.unique().astype(int).tolist()
            ),
            "mean_squared_canonical_correlation_range": [
                float(support_frame.mean_squared_canonical_correlation.min()),
                float(support_frame.mean_squared_canonical_correlation.max()),
            ],
            "pairwise_mean_squared_canonical_correlation_range": [
                float(
                    support_pair_frame.mean_squared_canonical_correlation.min()
                ),
                float(
                    support_pair_frame.mean_squared_canonical_correlation.max()
                ),
            ],
        },
        "primary_panel_metrics": {
            row.panel: {
                key: (int(value) if key in {"selected_k", "positive_pairs", "total_pairs"} else float(value))
                for key, value in row._asdict().items()
                if key not in {"Index", "panel", "reference_support", "analysis_role"}
            }
            for row in panel_frame.loc[
                panel_frame.reference_support.eq("exact_connectivity_excluded")
            ].itertuples()
        },
        "target_jackknife": summarize_jackknife(jackknife_frame),
        "inference": {
            "qap_permutations": int(qap_permutations),
            "random_subspaces": int(random_subspaces),
            "random_subspace_definition": (
                "Haar-random k-frame inside the estimable non-null target subspace, "
                "with the observed leading k eigenvalues preserved exactly in the "
                "kernel; every random kernel is additionally unit-diagonal-normalized "
                "for the representation-matched primary null"
            ),
            "seed": int(seed),
            "multiplicity": (
                "Holm correction is reported within each representation and across "
                "both representations; omnibus mean-delta QAPs are corrected across "
                "their three metrics and across both representations"
            ),
        },
        "provenance": {
            "dockstring": {
                "path": str(davis.DEFAULT_DOCKSTRING.relative_to(PACKAGE)),
                "sha256": sha256_file(davis.DEFAULT_DOCKSTRING),
            },
            "davis": {
                "path": str(davis.DEFAULT_DAVIS.relative_to(PACKAGE)),
                "sha256": sha256_file(davis.DEFAULT_DAVIS),
            },
            "pkis2": {
                "path": str(geometry.pkis2.DEFAULT_PKIS2.relative_to(PACKAGE)),
                "sha256": sha256_file(geometry.pkis2.DEFAULT_PKIS2),
            },
            "pkis1": {
                "local_filename": Path(pkis1_zip).name,
                "sha256": sha256_file(pkis1_zip),
            },
            "kirhub": {
                "local_filename": Path(kirhub_workbook).name,
                "sha256": sha256_file(kirhub_workbook),
                "expected_sha256": kirhub.KIRHUB_SHA256,
                "license": kirhub.KIRHUB_LICENSE,
            },
            "targets": list(TARGETS),
            "target_order_sha256": canonical_json_sha256(list(TARGETS)),
        },
    }
    frames = {
        "target_pair_geometries": target_pair_geometry_frame(
            full, normalized_low, spectral_kernel
        ),
        "panel_metrics": panel_frame,
        "scaffold_support_stability": support_frame,
        "scaffold_support_pairwise_stability": support_pair_frame,
        "scaffold_support_panel_metrics": fold_panel_frame,
        "paired_target_qap": qap_frame,
        "omnibus_validation_qap": omnibus_frame,
        "eigenvalue_matched_random_subspace_null": random_frame,
        "retained_eigenvalue_spectrum": retained_spectrum,
        "target_jackknife": jackknife_frame,
    }
    write_outputs(Path(output), frames, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pkis1-zip", type=Path, required=True)
    parser.add_argument("--kirhub-workbook", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--variance-threshold", type=float, default=DEFAULT_VARIANCE_THRESHOLD
    )
    parser.add_argument("--scaffold-folds", type=int, default=DEFAULT_SCAFFOLD_FOLDS)
    parser.add_argument(
        "--qap-permutations", type=int, default=DEFAULT_QAP_PERMUTATIONS
    )
    parser.add_argument(
        "--random-subspaces", type=int, default=DEFAULT_RANDOM_SUBSPACES
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    summary = run_analysis(
        pkis1_zip=args.pkis1_zip,
        kirhub_workbook=args.kirhub_workbook,
        output=args.output,
        variance_threshold=args.variance_threshold,
        scaffold_folds=args.scaffold_folds,
        qap_permutations=args.qap_permutations,
        random_subspaces=args.random_subspaces,
        seed=args.seed,
    )
    print(
        "Selected k={selected_k} at cumulative fraction {fraction:.6f}; "
        "wrote {output}".format(
            selected_k=summary["selection_rule"]["selected_k"],
            fraction=summary["selection_rule"][
                "selected_cumulative_variance_fraction"
            ],
            output=args.output,
        )
    )


if __name__ == "__main__":
    main()
