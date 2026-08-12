#!/usr/bin/env python3
"""Outcome-blind calibration-size audit for residual-signed Vina panels.

For each calibration size, random DOCKSTRING rows select an exact k=8 panel by
minimising mean nearest signed distance (1-r) on the row-centred Vina target
map.  The selected panel is then evaluated, without retuning, on fixed DAVIS
and PKIS2 experimental maps against the exact C(21,8) panel distribution.

Experimental values are never used for Vina panel selection.  This analysis
separates the ligand count needed to estimate the broad map from the ligand
count needed to make a stable discrete target-panel decision.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ANALYSIS_DIR = Path(__file__).resolve().parent
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

import experimental_map_reliability as reliability  # noqa: E402
import public_experimental_boundary_controls as boundary  # noqa: E402


PACKAGE = ANALYSIS_DIR.parent
DEFAULT_OUTPUT = PACKAGE / "results" / "public_vina_panel_calibration_transfer"
SEQUENCE_FASTA = PACKAGE / "data" / "frozen" / "dockstring_kinase_receptors.fasta"
SEQUENCE_MANIFEST = (
    PACKAGE / "data" / "frozen" / "dockstring_kinase_receptors_manifest.csv"
)
SIZES = (200, 500, 2_000, 5_000, 15_000, 50_000)
REPETITIONS = 100
SEED = 202_609_01
MULTIPLICITY_PERMUTATIONS = 20_000
SEQUENCE_QAP_PERMUTATIONS = 100_000
CLUSTER_MULTIPLIER_REPETITIONS = 500
OPTIMUM_ABS_TOLERANCE = 1e-12
K = 8
K_GRID = (4, 6, 8, 10, 12)
OBJECTIVES = (
    "signed_1_minus_r",
    "absolute_1_minus_abs_r",
    "squared_1_minus_r2",
)
TARGETS = tuple(reliability.TARGETS)
OUTPUT_FILES = (
    "README.md",
    "calibration_replicates.csv",
    "calibration_summary.csv",
    "full_source_k_sensitivity.csv",
    "full_source_k_selections.csv",
    "same_endpoint_centering_comparison.csv",
    "same_endpoint_panel_selections.csv",
    "same_endpoint_cluster_multiplier_bootstrap.csv",
    "sequence_geometry_control.csv",
    "panel_selections.csv",
    "summary.json",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1 << 20):
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
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, float_format="%.12g")
    temporary.replace(path)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(json_ready(payload), indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def experimental_reference() -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    panels = reliability.load_public_panels()
    matrices: dict[str, np.ndarray] = {}
    support: dict[str, Any] = {}
    for name, frame in panels.items():
        values = frame[list(TARGETS)].to_numpy(float)
        informative = reliability.informative_rows(values)
        values = values[informative]
        matrices[name] = values
        support[name] = {
            "released_rows": int(len(frame)),
            "informative_rows": int(len(values)),
        }
    return matrices, support


def panel_metrics(
    distribution: np.ndarray, selected_index: int
) -> dict[str, float | int]:
    value = float(distribution[selected_index])
    oracle = float(distribution.min())
    median = float(np.median(distribution))
    denominator = median - oracle
    return {
        "mean_nearest_distance": value,
        "oracle_mean_nearest_distance": oracle,
        "absolute_loss_vs_oracle": float(value - oracle),
        "fraction_all_panels_no_worse": float(
            np.mean(distribution <= value + 1e-15)
        ),
        "exact_rank_best_is_1": int(1 + np.sum(distribution < value - 1e-15)),
        "median_headroom_captured": (
            float((median - value) / denominator) if denominator > 1e-15 else np.nan
        ),
    }


def global_alignment_identity(first: str, second: str, aligner: Any) -> float:
    """Identity over all columns of one deterministic optimal alignment."""
    alignment = aligner.align(first, second)[0]
    matches = 0
    for (first_start, first_end), (second_start, second_end) in zip(
        alignment.aligned[0], alignment.aligned[1]
    ):
        matches += sum(
            left == right
            for left, right in zip(
                first[first_start:first_end], second[second_start:second_end]
            )
        )
    coordinates = alignment.coordinates
    columns = sum(
        max(
            int(coordinates[0, index + 1] - coordinates[0, index]),
            int(coordinates[1, index + 1] - coordinates[1, index]),
        )
        for index in range(coordinates.shape[1] - 1)
    )
    if columns < 1:
        raise ValueError("empty global sequence alignment")
    return float(matches / columns)


def symmetric_global_alignment_identity(
    first: str, second: str, aligner: Any
) -> float:
    """Order-invariant mean over both tied-alignment orientations."""
    return float(
        0.5
        * (
            global_alignment_identity(first, second, aligner)
            + global_alignment_identity(second, first, aligner)
        )
    )


def load_sequence_identity() -> tuple[np.ndarray, dict[str, Any]]:
    """Global BLOSUM62 identities for the exact frozen DOCKSTRING receptors."""
    try:
        import Bio
        from Bio import Align
        from Bio.Align import substitution_matrices
    except ImportError as error:  # pragma: no cover - dependency gate
        raise RuntimeError("Biopython is required for the sequence baseline") from error
    sequences: dict[str, str] = {}
    current: str | None = None
    for line in SEQUENCE_FASTA.read_text(encoding="utf-8").splitlines():
        if line.startswith(">"):
            current = line[1:].split()[0]
            if current in sequences:
                raise ValueError(f"duplicate FASTA target: {current}")
            sequences[current] = ""
        elif current is not None:
            sequences[current] += line.strip()
    if set(sequences) != set(TARGETS):
        raise ValueError("frozen receptor FASTA does not match the 21-target panel")
    manifest = pd.read_csv(SEQUENCE_MANIFEST)
    if set(manifest.target) != set(TARGETS):
        raise ValueError("receptor manifest does not match the 21-target panel")
    for row in manifest.itertuples(index=False):
        sequence = sequences[row.target]
        if len(sequence) != int(row.sequence_length):
            raise ValueError(f"sequence length mismatch for {row.target}")
        observed = hashlib.sha256(sequence.encode("ascii")).hexdigest()
        if observed != row.sequence_sha256:
            raise ValueError(f"sequence checksum mismatch for {row.target}")

    aligner = Align.PairwiseAligner()
    aligner.mode = "global"
    aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
    aligner.open_gap_score = -10.0
    aligner.extend_gap_score = -0.5
    identity = np.eye(len(TARGETS), dtype=np.float64)
    for first in range(len(TARGETS)):
        for second in range(first + 1, len(TARGETS)):
            first_sequence = sequences[TARGETS[first]]
            second_sequence = sequences[TARGETS[second]]
            value = symmetric_global_alignment_identity(
                first_sequence, second_sequence, aligner
            )
            identity[first, second] = identity[second, first] = value
    metadata = {
        "source": "exact receptor-domain sequences parsed from DOCKSTRING 0.3.4 PDBQT resources",
        "fasta_sha256": sha256_file(SEQUENCE_FASTA),
        "manifest_sha256": sha256_file(SEQUENCE_MANIFEST),
        "alignment": "global BLOSUM62; gap-open -10; gap-extend -0.5",
        "tie_symmetrization": (
            "mean identity from the first optimal alignment in both sequence "
            "orientations; removes arbitrary target-order dependence under tied "
            "optimal BLOSUM62 alignments"
        ),
        "identity_denominator": "all alignment columns, including gaps",
        "residue_normalization": (
            "defined by the frozen manifest/fetch producer; every loaded sequence "
            "is checked against its manifest length and SHA-256"
        ),
        "biopython_version": Bio.__version__,
    }
    return identity, metadata


def pearson_edges(first: np.ndarray, second: np.ndarray) -> float:
    left = boundary.upper(first)
    right = boundary.upper(second)
    return float(np.corrcoef(left, right)[0, 1])


def sequence_geometry_control(
    broad_vina: np.ndarray,
    *,
    permutations: int,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Shared-label QAP for raw/residual Vina agreement with receptor sequence."""
    sequence_identity, provenance = load_sequence_identity()
    raw = reliability.target_correlation(broad_vina)
    residual = boundary.target_map(broad_vina)
    observed_raw = pearson_edges(raw, sequence_identity)
    observed_residual = pearson_edges(residual, sequence_identity)
    observed_difference = observed_residual - observed_raw
    null_raw = np.empty(permutations, dtype=np.float64)
    null_residual = np.empty(permutations, dtype=np.float64)
    rng = np.random.default_rng(seed)
    for repetition in range(permutations):
        permutation = rng.permutation(len(TARGETS))
        permuted = sequence_identity[np.ix_(permutation, permutation)]
        null_raw[repetition] = pearson_edges(raw, permuted)
        null_residual[repetition] = pearson_edges(residual, permuted)
    null_difference = null_residual - null_raw
    rows = []
    for metric, observed, null in (
        ("raw_vina_vs_sequence_edge_pearson", observed_raw, null_raw),
        (
            "row_centered_residual_vina_vs_sequence_edge_pearson",
            observed_residual,
            null_residual,
        ),
        (
            "residual_minus_raw_sequence_edge_pearson",
            observed_difference,
            null_difference,
        ),
    ):
        rows.append(
            {
                "metric": metric,
                "observed": observed,
                "target_label_qap_p_one_sided_at_least_observed": float(
                    (1 + np.sum(null >= observed)) / (permutations + 1)
                ),
                "null_mean": float(np.mean(null)),
                "null_q025": float(np.quantile(null, 0.025)),
                "null_q975": float(np.quantile(null, 0.975)),
                "permutations": permutations,
                "seed": seed,
            }
        )
    metadata = {
        "provenance": provenance,
        "statistic": "Pearson correlation across the 210 unique target-pair edges",
        "qap_contract": (
            "one common target-label permutation of the fixed sequence-identity map "
            "is used for raw, residual and residual-minus-raw statistics in every "
            "replicate; Vina maps remain fixed"
        ),
        "boundary": (
            "The control tests fixed-panel sequence-related geometry. It does not "
            "show that Vina is superior to sequence or that either map predicts "
            "affinity for unseen targets."
        ),
    }
    return pd.DataFrame.from_records(rows), metadata


def coverage_value(distance: np.ndarray, selected: np.ndarray) -> float:
    """Mean distance from every target to its nearest selected target."""
    selected = np.asarray(selected, dtype=np.int16)
    return float(np.asarray(distance, dtype=np.float64)[:, selected].min(axis=1).mean())


def exact_percentile(sorted_distribution: np.ndarray, value: float) -> float:
    """Fraction of exactly enumerated panels no worse than ``value``."""
    return float(
        np.searchsorted(sorted_distribution, value + 1e-15, side="right")
        / len(sorted_distribution)
    )


def lexicographic_argmin(distribution: np.ndarray) -> int:
    """First exact minimizer in the enumerator's lexicographic panel order."""
    values = np.asarray(distribution, dtype=np.float64)
    if values.ndim != 1 or len(values) < 1 or not np.isfinite(values).all():
        raise ValueError("panel objective distribution must be finite and nonempty")
    return int(np.argmin(values))


def numerical_optimal_indices(
    distribution: np.ndarray, *, absolute_tolerance: float = OPTIMUM_ABS_TOLERANCE
) -> np.ndarray:
    """Complete enumerated optimum set under a declared absolute tolerance."""
    values = np.asarray(distribution, dtype=np.float64)
    selected = lexicographic_argmin(values)
    if absolute_tolerance < 0.0 or not np.isfinite(absolute_tolerance):
        raise ValueError("optimum tolerance must be finite and nonnegative")
    return np.flatnonzero(values <= values[selected] + absolute_tolerance)


def selection_tie_diagnostics(distribution: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(distribution, dtype=np.float64)
    selected_index = lexicographic_argmin(values)
    minimum = float(values[selected_index])
    ordered = np.partition(values, 1) if len(values) > 1 else values
    return {
        "selected_lexicographic_index": selected_index,
        "minimum_objective": minimum,
        "gap_to_second_panel_objective": (
            float(ordered[1] - minimum) if len(values) > 1 else np.nan
        ),
        "exact_minimizer_count": int(np.sum(values == minimum)),
        "panels_within_absolute_1e_12_of_minimum": int(
            len(numerical_optimal_indices(values))
        ),
        "panels_within_absolute_1e_8_of_minimum": int(
            np.sum(values <= minimum + 1e-8)
        ),
    }


def normalized_trapezoid_auc(values: np.ndarray, grid: np.ndarray) -> float:
    """Trapezoidal mean over a strictly increasing grid (NumPy 1.26 compatible)."""
    values = np.asarray(values, dtype=np.float64)
    grid = np.asarray(grid, dtype=np.float64)
    if values.ndim != 1 or grid.shape != values.shape or len(grid) < 2:
        raise ValueError("trapezoid inputs must be aligned one-dimensional arrays")
    if not np.isfinite(values).all() or not np.isfinite(grid).all():
        raise ValueError("trapezoid inputs must be finite")
    width = float(grid[-1] - grid[0])
    if width <= 0.0 or np.any(np.diff(grid) <= 0.0):
        raise ValueError("trapezoid grid must be strictly increasing")
    return float(np.trapz(values, grid) / width)


def same_endpoint_centering_comparison(
    broad_vina: np.ndarray,
    experimental_matrices: dict[str, np.ndarray],
    *,
    permutations: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Compare raw- and residual-Vina selectors on one residual endpoint.

    Both selectors are evaluated on the *same* experimental target map in each
    cell, so the contrast isolates the Vina selector transformation rather than
    changing both selection and evaluation endpoints.  The metric residual map
    is primary; column-ranking each experimental target before row centring is a
    monotone/censoring sensitivity.
    """
    if permutations < 100:
        raise ValueError("same-endpoint test requires at least 100 permutations")
    vina_maps = {
        "raw_vina": reliability.target_correlation(broad_vina),
        "row_centered_residual_vina": boundary.target_map(broad_vina),
    }
    sequence_identity, _ = load_sequence_identity()
    sequence_distance = boundary.distance_from_map(
        sequence_identity, "signed_1_minus_r"
    )
    endpoint_maps: dict[str, dict[str, np.ndarray]] = {
        "metric_row_centered_residual": {
            name: boundary.target_map(matrix)
            for name, matrix in experimental_matrices.items()
        },
        "column_rank_then_row_centered": {
            name: boundary.column_rank_target_map(matrix)
            for name, matrix in experimental_matrices.items()
        },
    }
    records: list[dict[str, Any]] = []
    selections: list[dict[str, Any]] = []
    selector_diagnostics: list[dict[str, Any]] = []
    # Each payload supports both the observed paired contrast and a shared
    # target-label permutation without repeatedly enumerating panel indices.
    permutation_cells: dict[
        tuple[str, str], list[dict[str, Any]]
    ] = {
        (endpoint, objective): []
        for endpoint in endpoint_maps
        for objective in OBJECTIVES
    }
    for k in K_GRID:
        panels = boundary.all_panels(n_targets=len(TARGETS), k=k)
        sequence_distribution = boundary.coverage_distribution(
            sequence_distance, panels
        )
        sequence_primary_index = lexicographic_argmin(sequence_distribution)
        sequence_optimal_indices = numerical_optimal_indices(sequence_distribution)
        sequence_selected = panels[sequence_primary_index].copy()
        for member_rank, panel_index in enumerate(sequence_optimal_indices):
            for target_index in panels[panel_index]:
                selections.append(
                    {
                        "objective": "sequence_distance_1_minus_identity",
                        "targets_selected_k": k,
                        "selector": "receptor_sequence_identity",
                        "numerical_optimum_member_rank": member_rank,
                        "panel_index_lexicographic": int(panel_index),
                        "is_lexicographic_primary": bool(
                            panel_index == sequence_primary_index
                        ),
                        "target_index": int(target_index),
                        "target": TARGETS[int(target_index)],
                    }
                )
        selector_diagnostics.append(
            {
                "objective": "sequence_distance_1_minus_identity",
                "targets_selected_k": k,
                "selector": "receptor_sequence_identity",
                **selection_tie_diagnostics(sequence_distribution),
            }
        )
        fixed_selected: dict[tuple[str, str], np.ndarray] = {}
        fixed_optimal_panels: dict[tuple[str, str], np.ndarray] = {}
        for objective in OBJECTIVES:
            for selector_name, vina_map in vina_maps.items():
                vina_distance = boundary.distance_from_map(vina_map, objective)
                vina_distribution = boundary.coverage_distribution(vina_distance, panels)
                primary_index = lexicographic_argmin(vina_distribution)
                optimal_indices = numerical_optimal_indices(vina_distribution)
                selected = panels[primary_index].copy()
                fixed_selected[(objective, selector_name)] = selected
                fixed_optimal_panels[(objective, selector_name)] = panels[
                    optimal_indices
                ].copy()
                for member_rank, panel_index in enumerate(optimal_indices):
                    for target_index in panels[panel_index]:
                        selections.append(
                            {
                                "objective": objective,
                                "targets_selected_k": k,
                                "selector": selector_name,
                                "numerical_optimum_member_rank": member_rank,
                                "panel_index_lexicographic": int(panel_index),
                                "is_lexicographic_primary": bool(
                                    panel_index == primary_index
                                ),
                                "target_index": int(target_index),
                                "target": TARGETS[int(target_index)],
                            }
                        )
                selector_diagnostics.append(
                    {
                        "objective": objective,
                        "targets_selected_k": k,
                        "selector": selector_name,
                        **selection_tie_diagnostics(vina_distribution),
                    }
                )
        for endpoint_name, panel_maps in endpoint_maps.items():
            for objective in OBJECTIVES:
                raw_selected = fixed_selected[(objective, "raw_vina")]
                residual_selected = fixed_selected[
                    (objective, "row_centered_residual_vina")
                ]
                raw_optimal_panels = fixed_optimal_panels[
                    (objective, "raw_vina")
                ]
                residual_optimal_panels = fixed_optimal_panels[
                    (objective, "row_centered_residual_vina")
                ]
                for panel_name, experiment_map in panel_maps.items():
                    distance = boundary.distance_from_map(experiment_map, objective)
                    distribution = boundary.coverage_distribution(distance, panels)
                    sorted_distribution = np.sort(distribution)
                    oracle = float(sorted_distribution[0])
                    raw_value = coverage_value(distance, raw_selected)
                    residual_value = coverage_value(distance, residual_selected)
                    sequence_value = coverage_value(distance, sequence_selected)
                    raw_percentile = exact_percentile(sorted_distribution, raw_value)
                    residual_percentile = exact_percentile(
                        sorted_distribution, residual_value
                    )
                    sequence_percentile = exact_percentile(
                        sorted_distribution, sequence_value
                    )
                    raw_optimal_values = np.asarray(
                        [
                            coverage_value(distance, panel)
                            for panel in raw_optimal_panels
                        ],
                        dtype=np.float64,
                    )
                    residual_optimal_values = np.asarray(
                        [
                            coverage_value(distance, panel)
                            for panel in residual_optimal_panels
                        ],
                        dtype=np.float64,
                    )
                    raw_optimal_percentiles = np.asarray(
                        [
                            exact_percentile(sorted_distribution, value)
                            for value in raw_optimal_values
                        ]
                    )
                    residual_optimal_percentiles = np.asarray(
                        [
                            exact_percentile(sorted_distribution, value)
                            for value in residual_optimal_values
                        ]
                    )
                    pair_improvements = (
                        raw_optimal_values[:, None]
                        - residual_optimal_values[None, :]
                    )
                    pair_percentile_improvements = (
                        raw_optimal_percentiles[:, None]
                        - residual_optimal_percentiles[None, :]
                    )
                    records.append(
                        {
                            "experimental_endpoint": endpoint_name,
                            "objective": objective,
                            "targets_selected_k": k,
                            "evaluation_map": panel_name,
                            "all_possible_panels": int(len(panels)),
                            "raw_vina_selected_mean_nearest_distance": raw_value,
                            "residual_vina_selected_mean_nearest_distance": residual_value,
                            "sequence_selected_mean_nearest_distance": sequence_value,
                            "experimental_in_sample_oracle_mean_nearest_distance": oracle,
                            "raw_vina_selected_excess_loss_over_in_sample_oracle": float(
                                raw_value - oracle
                            ),
                            "residual_vina_selected_excess_loss_over_in_sample_oracle": float(
                                residual_value - oracle
                            ),
                            "sequence_selected_excess_loss_over_in_sample_oracle": float(
                                sequence_value - oracle
                            ),
                            "raw_minus_residual_selected_coverage_loss_improvement": float(
                                raw_value - residual_value
                            ),
                            "raw_vina_selected_exact_fraction_all_panels_no_worse": raw_percentile,
                            "residual_vina_selected_exact_fraction_all_panels_no_worse": residual_percentile,
                            "sequence_selected_exact_fraction_all_panels_no_worse": sequence_percentile,
                            "raw_minus_residual_selected_exact_percentile_improvement": float(
                                raw_percentile - residual_percentile
                            ),
                            "residual_selector_has_lower_coverage_loss": bool(
                                residual_value < raw_value - 1e-15
                            ),
                            "raw_vina_numerical_optimal_panel_count": int(
                                len(raw_optimal_panels)
                            ),
                            "residual_vina_numerical_optimal_panel_count": int(
                                len(residual_optimal_panels)
                            ),
                            "raw_optimal_experimental_excess_loss_min": float(
                                raw_optimal_values.min() - oracle
                            ),
                            "raw_optimal_experimental_excess_loss_max": float(
                                raw_optimal_values.max() - oracle
                            ),
                            "residual_optimal_experimental_excess_loss_min": float(
                                residual_optimal_values.min() - oracle
                            ),
                            "residual_optimal_experimental_excess_loss_max": float(
                                residual_optimal_values.max() - oracle
                            ),
                            "all_optimal_pair_coverage_loss_improvement_min_conservative": float(
                                pair_improvements.min()
                            ),
                            "all_optimal_pair_coverage_loss_improvement_mean": float(
                                pair_improvements.mean()
                            ),
                            "all_optimal_pair_coverage_loss_improvement_max": float(
                                pair_improvements.max()
                            ),
                            "all_optimal_pair_exact_percentile_improvement_min_conservative": float(
                                pair_percentile_improvements.min()
                            ),
                            "all_optimal_pair_exact_percentile_improvement_mean": float(
                                pair_percentile_improvements.mean()
                            ),
                            "all_optimal_pair_exact_percentile_improvement_max": float(
                                pair_percentile_improvements.max()
                            ),
                            "residual_minus_sequence_selected_coverage_loss": float(
                                residual_value - sequence_value
                            ),
                            "sequence_minus_residual_selected_coverage_loss_improvement": float(
                                sequence_value - residual_value
                            ),
                            "residual_minus_sequence_selected_exact_percentile": float(
                                residual_percentile - sequence_percentile
                            ),
                        }
                    )
                    permutation_cells[(endpoint_name, objective)].append(
                        {
                            "evaluation_map": panel_name,
                            "targets_selected_k": k,
                            "distance": distance,
                            "sorted_distribution": sorted_distribution,
                            "raw_selected": raw_selected,
                            "residual_selected": residual_selected,
                            "raw_optimal_panels": raw_optimal_panels,
                            "residual_optimal_panels": residual_optimal_panels,
                        }
                    )

    frame = pd.DataFrame.from_records(records)
    observed: dict[tuple[str, str], dict[str, float]] = {}
    null_loss: dict[tuple[str, str], np.ndarray] = {}
    null_percentile: dict[tuple[str, str], np.ndarray] = {}
    null_conservative_tie_loss: dict[tuple[str, str], np.ndarray] = {}
    null_mean_tie_loss: dict[tuple[str, str], np.ndarray] = {}
    for key, cells in permutation_cells.items():
        subset = frame.loc[
            frame.experimental_endpoint.eq(key[0]) & frame.objective.eq(key[1])
        ]
        observed[key] = {
            "mean_coverage_loss_improvement": float(
                subset.raw_minus_residual_selected_coverage_loss_improvement.mean()
            ),
            "mean_exact_percentile_improvement": float(
                subset.raw_minus_residual_selected_exact_percentile_improvement.mean()
            ),
            "conservative_mean_coverage_loss_improvement_over_ties": float(
                subset.all_optimal_pair_coverage_loss_improvement_min_conservative.mean()
            ),
            "tie_averaged_mean_coverage_loss_improvement": float(
                subset.all_optimal_pair_coverage_loss_improvement_mean.mean()
            ),
        }
        null_loss[key] = np.empty(permutations, dtype=np.float64)
        null_percentile[key] = np.empty(permutations, dtype=np.float64)
        null_conservative_tie_loss[key] = np.empty(
            permutations, dtype=np.float64
        )
        null_mean_tie_loss[key] = np.empty(permutations, dtype=np.float64)

    rng = np.random.default_rng(seed)
    for repetition in range(permutations):
        permutation = rng.permutation(len(TARGETS))
        for key, cells in permutation_cells.items():
            loss_differences: list[float] = []
            percentile_differences: list[float] = []
            conservative_tie_differences: list[float] = []
            mean_tie_differences: list[float] = []
            for cell in cells:
                raw_selected = np.sort(permutation[cell["raw_selected"]])
                residual_selected = np.sort(permutation[cell["residual_selected"]])
                raw_value = coverage_value(cell["distance"], raw_selected)
                residual_value = coverage_value(cell["distance"], residual_selected)
                loss_differences.append(raw_value - residual_value)
                sorted_distribution = cell["sorted_distribution"]
                percentile_differences.append(
                    exact_percentile(sorted_distribution, raw_value)
                    - exact_percentile(sorted_distribution, residual_value)
                )
                raw_optimal_panels = np.sort(
                    permutation[cell["raw_optimal_panels"]], axis=1
                )
                residual_optimal_panels = np.sort(
                    permutation[cell["residual_optimal_panels"]], axis=1
                )
                raw_optimal_values = np.asarray(
                    [
                        coverage_value(cell["distance"], panel)
                        for panel in raw_optimal_panels
                    ]
                )
                residual_optimal_values = np.asarray(
                    [
                        coverage_value(cell["distance"], panel)
                        for panel in residual_optimal_panels
                    ]
                )
                conservative_tie_differences.append(
                    float(raw_optimal_values.min() - residual_optimal_values.max())
                )
                mean_tie_differences.append(
                    float(raw_optimal_values.mean() - residual_optimal_values.mean())
                )
            null_loss[key][repetition] = float(np.mean(loss_differences))
            null_percentile[key][repetition] = float(
                np.mean(percentile_differences)
            )
            null_conservative_tie_loss[key][repetition] = float(
                np.mean(conservative_tie_differences)
            )
            null_mean_tie_loss[key][repetition] = float(
                np.mean(mean_tie_differences)
            )

    inference: list[dict[str, Any]] = []
    for endpoint_name in endpoint_maps:
        endpoint_keys = [(endpoint_name, objective) for objective in OBJECTIVES]
        observed_max_loss = max(
            observed[key]["mean_coverage_loss_improvement"] for key in endpoint_keys
        )
        observed_max_percentile = max(
            observed[key]["mean_exact_percentile_improvement"]
            for key in endpoint_keys
        )
        observed_max_conservative_tie = max(
            observed[key][
                "conservative_mean_coverage_loss_improvement_over_ties"
            ]
            for key in endpoint_keys
        )
        observed_max_mean_tie = max(
            observed[key]["tie_averaged_mean_coverage_loss_improvement"]
            for key in endpoint_keys
        )
        null_max_loss = np.max(
            np.vstack([null_loss[key] for key in endpoint_keys]), axis=0
        )
        null_max_percentile = np.max(
            np.vstack([null_percentile[key] for key in endpoint_keys]), axis=0
        )
        null_max_conservative_tie = np.max(
            np.vstack(
                [null_conservative_tie_loss[key] for key in endpoint_keys]
            ),
            axis=0,
        )
        null_max_mean_tie = np.max(
            np.vstack([null_mean_tie_loss[key] for key in endpoint_keys]), axis=0
        )
        for key in endpoint_keys:
            loss_observed = observed[key]["mean_coverage_loss_improvement"]
            percentile_observed = observed[key]["mean_exact_percentile_improvement"]
            conservative_tie_observed = observed[key][
                "conservative_mean_coverage_loss_improvement_over_ties"
            ]
            mean_tie_observed = observed[key][
                "tie_averaged_mean_coverage_loss_improvement"
            ]
            loss_p = float(
                (1 + np.sum(null_loss[key] >= loss_observed)) / (permutations + 1)
            )
            percentile_p = float(
                (1 + np.sum(null_percentile[key] >= percentile_observed))
                / (permutations + 1)
            )
            conservative_tie_p = float(
                (
                    1
                    + np.sum(
                        null_conservative_tie_loss[key]
                        >= conservative_tie_observed
                    )
                )
                / (permutations + 1)
            )
            mean_tie_p = float(
                (1 + np.sum(null_mean_tie_loss[key] >= mean_tie_observed))
                / (permutations + 1)
            )
            mask = frame.experimental_endpoint.eq(key[0]) & frame.objective.eq(key[1])
            frame.loc[mask, "pooled_mean_coverage_loss_improvement_over_k_and_panels"] = loss_observed
            frame.loc[mask, "pooled_mean_exact_percentile_improvement_over_k_and_panels"] = percentile_observed
            frame.loc[mask, "target_label_p_one_sided_coverage_loss_improvement"] = loss_p
            frame.loc[mask, "target_label_p_one_sided_exact_percentile_improvement"] = percentile_p
            frame.loc[mask, "pooled_conservative_mean_coverage_loss_improvement_over_ties"] = conservative_tie_observed
            frame.loc[mask, "pooled_tie_averaged_mean_coverage_loss_improvement"] = mean_tie_observed
            frame.loc[mask, "target_label_p_one_sided_conservative_tie_improvement"] = conservative_tie_p
            frame.loc[mask, "target_label_p_one_sided_tie_averaged_improvement"] = mean_tie_p
            inference.append(
                {
                    "experimental_endpoint": key[0],
                    "objective": key[1],
                    "cells_averaged": len(permutation_cells[key]),
                    "mean_coverage_loss_improvement": loss_observed,
                    "wins_lower_loss": int(
                        frame.loc[mask].residual_selector_has_lower_coverage_loss.sum()
                    ),
                    "target_label_p_one_sided_coverage_loss_improvement": loss_p,
                    "mean_exact_percentile_improvement": percentile_observed,
                    "target_label_p_one_sided_exact_percentile_improvement": percentile_p,
                    "conservative_mean_coverage_loss_improvement_over_ties": conservative_tie_observed,
                    "conservative_tie_cells_with_positive_improvement": int(
                        np.sum(
                            frame.loc[
                                mask,
                                "all_optimal_pair_coverage_loss_improvement_min_conservative",
                            ]
                            > 0.0
                        )
                    ),
                    "target_label_p_one_sided_conservative_tie_improvement": conservative_tie_p,
                    "tie_averaged_mean_coverage_loss_improvement": mean_tie_observed,
                    "target_label_p_one_sided_tie_averaged_improvement": mean_tie_p,
                    "target_label_null_loss_q025": float(
                        np.quantile(null_loss[key], 0.025)
                    ),
                    "target_label_null_loss_q975": float(
                        np.quantile(null_loss[key], 0.975)
                    ),
                }
            )
        adjusted_loss_p = float(
            (1 + np.sum(null_max_loss >= observed_max_loss)) / (permutations + 1)
        )
        adjusted_percentile_p = float(
            (1 + np.sum(null_max_percentile >= observed_max_percentile))
            / (permutations + 1)
        )
        adjusted_conservative_tie_p = float(
            (
                1
                + np.sum(
                    null_max_conservative_tie >= observed_max_conservative_tie
                )
            )
            / (permutations + 1)
        )
        adjusted_mean_tie_p = float(
            (1 + np.sum(null_max_mean_tie >= observed_max_mean_tie))
            / (permutations + 1)
        )
        for record in inference:
            if record["experimental_endpoint"] == endpoint_name:
                record["max_over_three_objectives_p_coverage_loss"] = adjusted_loss_p
                record[
                    "max_over_three_objectives_p_exact_percentile"
                ] = adjusted_percentile_p
                record[
                    "max_over_three_objectives_p_conservative_tie_improvement"
                ] = adjusted_conservative_tie_p
                record[
                    "max_over_three_objectives_p_tie_averaged_improvement"
                ] = adjusted_mean_tie_p

    metadata = {
        "comparison": (
            "raw-Vina-selected versus row-centered-residual-Vina-selected fixed "
            "panels, both evaluated on the same residual experimental endpoint"
        ),
        "k_grid": list(K_GRID),
        "objectives": list(OBJECTIVES),
        "primary_endpoint": "metric_row_centered_residual",
        "primary_statistic": (
            "signed-1-r conservative best-raw-optimum minus worst-residual-optimum "
            "coverage-loss improvement, averaged over the prespecified k grid and "
            "both panels; numerical Vina optima use the declared absolute tolerance"
        ),
        "primary_inference": (
            "one-sided shared target-label permutation test for the conservative "
            "tie-robust coverage-loss contrast. Deterministic lexicographic and "
            "tie-averaged contrasts are co-reported; exact-percentile inference is "
            "scale-normalized sensitivity"
        ),
        "deterministic_tie_contract": {
            "primary_panel": (
                "first minimum in the complete lexicographic C(21,k) enumeration; "
                "no optimizer tolerance or outcome-dependent tie breaking"
            ),
            "numerical_optimum_set": (
                "every enumerated panel whose Vina objective is no more than "
                f"{OPTIMUM_ABS_TOLERANCE:g} above the enumerated minimum"
            ),
            "selector_diagnostics": selector_diagnostics,
            "tie_sensitivity": (
                "for every experimental endpoint cell, all raw-optimal crossed with "
                "all residual-optimal panels are evaluated. The conservative cell "
                "contrast is best raw minus worst residual; pooled inference averages "
                "that contrast across the fixed k grid and both panels under the same "
                "shared target-label null"
            ),
        },
        "outcome_blind_sequence_baseline": (
            "at every k, exact sequence-diverse panels minimize mean nearest "
            "distance 1-global-receptor-sequence-identity and are evaluated on the "
            "same experimental endpoints as both Vina selectors; sequence uses no "
            "experimental outcome"
        ),
        "column_rank_sensitivity": (
            "rank ligands independently within every experimental target before "
            "row centring; this removes target-wise monotone scale and reduces the "
            "influence of DAVIS floor censoring"
        ),
        "target_label_permutations": permutations,
        "target_label_seed": seed,
        "permutation_contract": (
            "one common permutation of the 21 target labels is applied to every "
            "fixed raw- and residual-Vina selected set across both panels, all k, "
            "and all three objectives in a replicate; experimental maps and exact "
            "all-panel distributions remain fixed"
        ),
        "inference": inference,
        "multiplicity": (
            "signed 1-r is the prespecified scientific objective. The max-over-three "
            "p-value is reported as a retrospective objective-choice sensitivity; "
            "metric residual and column-rank endpoints are not pooled as duplicate "
            "confirmatory tests"
        ),
        "boundary": (
            "This is fixed-target map coverage, not affinity prediction, compound "
            "ranking, or retrieval of unseen targets. The k-averaged comparison "
            "avoids selecting the exceptionally favorable inherited k=8 result. "
            "Sequence is a substantive baseline and no Vina-over-sequence "
            "superiority claim is made."
        ),
    }
    return frame, pd.DataFrame.from_records(selections), metadata


def summarize_same_endpoint_multiplier(frame: pd.DataFrame) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    group_columns = ["aggregation_scope", "evaluation_map", "targets_selected_k"]
    for keys, group in frame.groupby(group_columns, dropna=False, sort=True):
        improvement = group[
            "raw_minus_residual_selected_coverage_loss_improvement"
        ].to_numpy(float)
        percentile = group[
            "raw_minus_residual_selected_exact_percentile_improvement"
        ].to_numpy(float)
        sequence_improvement = (
            group["sequence_minus_residual_selected_coverage_loss_improvement"]
            .to_numpy(float)
            if "sequence_minus_residual_selected_coverage_loss_improvement"
            in group.columns
            else np.full(len(group), np.nan)
        )
        conservative_tie = (
            group[
                "all_optimal_pair_coverage_loss_improvement_min_conservative"
            ].to_numpy(float)
            if "all_optimal_pair_coverage_loss_improvement_min_conservative"
            in group.columns
            else np.full(len(group), np.nan)
        )
        mean_tie = (
            group["all_optimal_pair_coverage_loss_improvement_mean"].to_numpy(float)
            if "all_optimal_pair_coverage_loss_improvement_mean" in group.columns
            else np.full(len(group), np.nan)
        )
        records.append(
            {
                "aggregation_scope": keys[0],
                "evaluation_map": keys[1],
                "targets_selected_k": (
                    None if pd.isna(keys[2]) else int(keys[2])
                ),
                "repetitions": int(len(group)),
                "coverage_loss_improvement_mean": float(np.mean(improvement)),
                "coverage_loss_improvement_median": float(np.median(improvement)),
                "coverage_loss_improvement_q025": float(
                    np.quantile(improvement, 0.025)
                ),
                "coverage_loss_improvement_q975": float(
                    np.quantile(improvement, 0.975)
                ),
                "coverage_loss_improvement_win_fraction": float(
                    np.mean(improvement > 0.0)
                ),
                "exact_percentile_improvement_median": float(
                    np.median(percentile)
                ),
                "exact_percentile_improvement_q025": float(
                    np.quantile(percentile, 0.025)
                ),
                "exact_percentile_improvement_q975": float(
                    np.quantile(percentile, 0.975)
                ),
                "exact_percentile_improvement_win_fraction": float(
                    np.mean(percentile > 0.0)
                ),
                "sequence_minus_residual_coverage_loss_improvement_median": float(
                    np.nanmedian(sequence_improvement)
                    if np.isfinite(sequence_improvement).any()
                    else np.nan
                ),
                "sequence_minus_residual_coverage_loss_improvement_q025": float(
                    np.nanquantile(sequence_improvement, 0.025)
                    if np.isfinite(sequence_improvement).any()
                    else np.nan
                ),
                "sequence_minus_residual_coverage_loss_improvement_q975": float(
                    np.nanquantile(sequence_improvement, 0.975)
                    if np.isfinite(sequence_improvement).any()
                    else np.nan
                ),
                "residual_better_than_sequence_win_fraction": float(
                    np.mean(sequence_improvement > 0.0)
                    if np.isfinite(sequence_improvement).any()
                    else np.nan
                ),
                "conservative_tie_improvement_median": float(
                    np.nanmedian(conservative_tie)
                    if np.isfinite(conservative_tie).any()
                    else np.nan
                ),
                "conservative_tie_improvement_q025": float(
                    np.nanquantile(conservative_tie, 0.025)
                    if np.isfinite(conservative_tie).any()
                    else np.nan
                ),
                "conservative_tie_improvement_q975": float(
                    np.nanquantile(conservative_tie, 0.975)
                    if np.isfinite(conservative_tie).any()
                    else np.nan
                ),
                "conservative_tie_positive_fraction": float(
                    np.mean(conservative_tie > 0.0)
                    if np.isfinite(conservative_tie).any()
                    else np.nan
                ),
                "tie_averaged_improvement_median": float(
                    np.nanmedian(mean_tie)
                    if np.isfinite(mean_tie).any()
                    else np.nan
                ),
                "tie_averaged_improvement_q025": float(
                    np.nanquantile(mean_tie, 0.025)
                    if np.isfinite(mean_tie).any()
                    else np.nan
                ),
                "tie_averaged_improvement_q975": float(
                    np.nanquantile(mean_tie, 0.975)
                    if np.isfinite(mean_tie).any()
                    else np.nan
                ),
            }
        )
    return records


_WORKER_PANELS_BY_K: dict[int, np.ndarray] = {}
_WORKER_FIXED_SELECTED: dict[tuple[int, str], np.ndarray] = {}


def _initialize_multiplier_worker(
    fixed_selected: dict[tuple[int, str], np.ndarray],
) -> None:
    """Build exact panel arrays once in every spawned bootstrap worker."""
    global _WORKER_PANELS_BY_K, _WORKER_FIXED_SELECTED
    _WORKER_PANELS_BY_K = {
        k: boundary.all_panels(n_targets=len(TARGETS), k=k) for k in K_GRID
    }
    _WORKER_FIXED_SELECTED = fixed_selected


def _multiplier_bootstrap_chunk(
    panel_name: str,
    values: np.ndarray,
    inverse: np.ndarray,
    cluster_weight_rows: np.ndarray,
    repetition_start: int,
    source_clusters: int,
) -> list[dict[str, Any]]:
    """Compute a deterministic contiguous repetition chunk in one process."""
    records: list[dict[str, Any]] = []
    for offset, cluster_weights in enumerate(cluster_weight_rows):
        repetition = repetition_start + offset
        row_weights = cluster_weights[inverse]
        correlation = boundary.weighted_target_map(
            values, row_weights, row_centered=True
        )
        distance = boundary.distance_from_map(correlation, "signed_1_minus_r")
        for k, all_k in _WORKER_PANELS_BY_K.items():
            distribution = boundary.coverage_distribution(distance, all_k)
            oracle = float(distribution.min())
            raw_value = coverage_value(
                distance, _WORKER_FIXED_SELECTED[(k, "raw_vina")]
            )
            residual_value = coverage_value(
                distance,
                _WORKER_FIXED_SELECTED[(k, "row_centered_residual_vina")],
            )
            sequence_value = coverage_value(
                distance,
                _WORKER_FIXED_SELECTED[(k, "receptor_sequence_identity")],
            )
            raw_optimal_values = np.asarray(
                [
                    coverage_value(distance, panel)
                    for panel in _WORKER_FIXED_SELECTED[
                        (k, "raw_vina_numerical_optimal_panels")
                    ]
                ]
            )
            residual_optimal_values = np.asarray(
                [
                    coverage_value(distance, panel)
                    for panel in _WORKER_FIXED_SELECTED[
                        (k, "row_centered_residual_vina_numerical_optimal_panels")
                    ]
                ]
            )
            raw_percentile = float(np.mean(distribution <= raw_value + 1e-15))
            residual_percentile = float(
                np.mean(distribution <= residual_value + 1e-15)
            )
            sequence_percentile = float(
                np.mean(distribution <= sequence_value + 1e-15)
            )
            records.append(
                {
                    "resampling_scheme": "positive_exp1_cluster_multiplier",
                    "aggregation_scope": "per_k",
                    "evaluation_map": panel_name,
                    "repetition": repetition,
                    "targets_selected_k": k,
                    "source_ligands": int(len(values)),
                    "source_clusters": int(source_clusters),
                    "effective_row_sample_size": float(
                        np.square(row_weights.sum())
                        / np.square(row_weights).sum()
                    ),
                    "all_possible_panels": int(len(all_k)),
                    "experimental_in_sample_oracle_mean_nearest_distance": oracle,
                    "raw_vina_selected_excess_loss_over_in_sample_oracle": float(
                        raw_value - oracle
                    ),
                    "residual_vina_selected_excess_loss_over_in_sample_oracle": float(
                        residual_value - oracle
                    ),
                    "sequence_selected_excess_loss_over_in_sample_oracle": float(
                        sequence_value - oracle
                    ),
                    "raw_minus_residual_selected_coverage_loss_improvement": float(
                        raw_value - residual_value
                    ),
                    "raw_vina_selected_exact_fraction_all_panels_no_worse": raw_percentile,
                    "residual_vina_selected_exact_fraction_all_panels_no_worse": residual_percentile,
                    "sequence_selected_exact_fraction_all_panels_no_worse": sequence_percentile,
                    "raw_minus_residual_selected_exact_percentile_improvement": float(
                        raw_percentile - residual_percentile
                    ),
                    "all_optimal_pair_coverage_loss_improvement_min_conservative": float(
                        raw_optimal_values.min() - residual_optimal_values.max()
                    ),
                    "all_optimal_pair_coverage_loss_improvement_mean": float(
                        raw_optimal_values.mean() - residual_optimal_values.mean()
                    ),
                    "all_optimal_pair_coverage_loss_improvement_max": float(
                        raw_optimal_values.max() - residual_optimal_values.min()
                    ),
                    "residual_minus_sequence_selected_coverage_loss": float(
                        residual_value - sequence_value
                    ),
                    "sequence_minus_residual_selected_coverage_loss_improvement": float(
                        sequence_value - residual_value
                    ),
                    "residual_minus_sequence_selected_exact_percentile": float(
                        residual_percentile - sequence_percentile
                    ),
                }
            )
    return records


def same_endpoint_cluster_multiplier_bootstrap(
    broad_vina: np.ndarray,
    released_panels: dict[str, pd.DataFrame],
    *,
    repetitions: int,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Cluster multiplier uncertainty for the signed same-endpoint aggregate.

    Vina-selected panels are fixed once on the full de-leaked source. Independent
    Exp(1) weights are drawn for DAVIS and PKIS2 chemical clusters in every common
    repetition.  All selected panels are evaluated on the resulting weighted
    residual experimental maps.
    """
    if repetitions < 100:
        raise ValueError("cluster multiplier audit requires at least 100 repetitions")
    vina_maps = {
        "raw_vina": reliability.target_correlation(broad_vina),
        "row_centered_residual_vina": boundary.target_map(broad_vina),
    }
    panels_by_k = {
        k: boundary.all_panels(n_targets=len(TARGETS), k=k) for k in K_GRID
    }
    sequence_identity, _ = load_sequence_identity()
    sequence_distance = boundary.distance_from_map(
        sequence_identity, "signed_1_minus_r"
    )
    fixed_selected: dict[tuple[int, str], np.ndarray] = {}
    for k, all_k in panels_by_k.items():
        for selector, target_correlation in vina_maps.items():
            distribution = boundary.coverage_distribution(
                boundary.distance_from_map(
                    target_correlation, "signed_1_minus_r"
                ),
                all_k,
            )
            fixed_selected[(k, selector)] = all_k[
                lexicographic_argmin(distribution)
            ].copy()
            fixed_selected[(k, f"{selector}_numerical_optimal_panels")] = all_k[
                numerical_optimal_indices(distribution)
            ].copy()
        sequence_distribution = boundary.coverage_distribution(
            sequence_distance, all_k
        )
        fixed_selected[(k, "receptor_sequence_identity")] = all_k[
            lexicographic_argmin(sequence_distribution)
        ].copy()

    panel_payloads: dict[str, dict[str, Any]] = {}
    for panel_index, (panel_name, panel) in enumerate(released_panels.items()):
        values = panel[list(TARGETS)].to_numpy(float)
        informative = reliability.informative_rows(values)
        values = values[informative]
        smiles = panel.smiles.to_numpy(dtype=object)[informative]
        _, labels = reliability.panel_clusters(pd.Series(smiles))
        unique, inverse = np.unique(labels, return_inverse=True)
        panel_payloads[panel_name] = {
            "values": values,
            "unique": unique,
            "inverse": inverse,
            "rng": np.random.default_rng(seed + panel_index * 1_000_000),
        }

    # Pre-generate the exact weight draws in the parent process. This preserves
    # the sequential RNG contract while allowing exact panel enumeration to run
    # in parallel. Every worker builds the immutable C(21,k) arrays once.
    jobs: list[tuple[str, np.ndarray, np.ndarray, np.ndarray, int, int]] = []
    worker_count = min(4, repetitions)
    chunk_edges = np.linspace(0, repetitions, worker_count + 1, dtype=int)
    for panel_name, payload in panel_payloads.items():
        all_weights = payload["rng"].exponential(
            scale=1.0,
            size=(repetitions, len(payload["unique"])),
        )
        for start, stop in zip(chunk_edges[:-1], chunk_edges[1:]):
            if start == stop:
                continue
            jobs.append(
                (
                    panel_name,
                    payload["values"],
                    payload["inverse"],
                    all_weights[start:stop],
                    int(start),
                    int(len(payload["unique"])),
                )
            )
    cell_records: list[dict[str, Any]] = []
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=worker_count,
        initializer=_initialize_multiplier_worker,
        initargs=(fixed_selected,),
    ) as executor:
        futures = [executor.submit(_multiplier_bootstrap_chunk, *job) for job in jobs]
        for future in concurrent.futures.as_completed(futures):
            cell_records.extend(future.result())
    cells = pd.DataFrame.from_records(cell_records).sort_values(
        ["evaluation_map", "repetition", "targets_selected_k"],
        kind="mergesort",
        ignore_index=True,
    )
    numeric = [
        "experimental_in_sample_oracle_mean_nearest_distance",
        "raw_vina_selected_excess_loss_over_in_sample_oracle",
        "residual_vina_selected_excess_loss_over_in_sample_oracle",
        "sequence_selected_excess_loss_over_in_sample_oracle",
        "raw_minus_residual_selected_coverage_loss_improvement",
        "raw_vina_selected_exact_fraction_all_panels_no_worse",
        "residual_vina_selected_exact_fraction_all_panels_no_worse",
        "sequence_selected_exact_fraction_all_panels_no_worse",
        "raw_minus_residual_selected_exact_percentile_improvement",
        "all_optimal_pair_coverage_loss_improvement_min_conservative",
        "all_optimal_pair_coverage_loss_improvement_mean",
        "all_optimal_pair_coverage_loss_improvement_max",
        "residual_minus_sequence_selected_coverage_loss",
        "sequence_minus_residual_selected_coverage_loss_improvement",
        "residual_minus_sequence_selected_exact_percentile",
        "effective_row_sample_size",
    ]
    panel_mean = (
        cells.groupby(["evaluation_map", "repetition"], sort=True)[numeric]
        .mean()
        .reset_index()
    )
    panel_mean.insert(0, "resampling_scheme", "positive_exp1_cluster_multiplier")
    panel_mean.insert(1, "aggregation_scope", "panel_mean_over_k")
    panel_mean["targets_selected_k"] = np.nan
    support = cells.groupby("evaluation_map", sort=True)[
        ["source_ligands", "source_clusters"]
    ].first()
    panel_mean = panel_mean.join(support, on="evaluation_map")
    panel_mean["all_possible_panels"] = np.nan

    pooled = (
        cells.groupby("repetition", sort=True)[numeric].mean().reset_index()
    )
    pooled.insert(0, "resampling_scheme", "positive_exp1_cluster_multiplier")
    pooled.insert(1, "aggregation_scope", "pooled_mean_over_panels_and_k")
    pooled["evaluation_map"] = "DAVIS_PKIS2_independent_cluster_weights"
    pooled["targets_selected_k"] = np.nan
    pooled["source_ligands"] = int(
        sum(len(payload["values"]) for payload in panel_payloads.values())
    )
    pooled["source_clusters"] = int(
        sum(len(payload["unique"]) for payload in panel_payloads.values())
    )
    pooled["all_possible_panels"] = np.nan
    frame = pd.concat([cells, panel_mean, pooled], ignore_index=True, sort=False)
    metadata = {
        "repetitions": repetitions,
        "seed": seed,
        "objective": "signed_1_minus_r",
        "experimental_endpoint": "metric_row_centered_residual",
        "k_grid": list(K_GRID),
        "selector_contract": (
            "raw- and residual-Vina panels are fixed once at every k on the full "
            "de-leaked broad Vina source; sequence panels are fixed from global "
            "receptor identity; experimental values never enter selection"
        ),
        "resampling_contract": (
            "independent positive Exp(1) weights are assigned to every panel-specific "
            "Butina chemical cluster and inherited by all ligands in that cluster; "
            "all clusters remain represented and DAVIS/PKIS2 draws share only the "
            "repetition index"
        ),
        "primary_bootstrap_aggregate": (
            "mean raw-minus-residual selected-panel coverage-loss improvement over "
            "the five prespecified k values and both experimental panels"
        ),
        "tie_robust_bootstrap": (
            f"all Vina panels within absolute {OPTIMUM_ABS_TOLERANCE:g} of each "
            "enumerated selector minimum are propagated; conservative best-raw "
            "minus worst-residual and tie-averaged contrasts are summarized"
        ),
        "summary": summarize_same_endpoint_multiplier(frame),
        "boundary": (
            "Intervals perturb chemical support within each released panel. They do "
            "not sample target identity, assay technology, or the kinase universe."
        ),
    }
    return frame, metadata


def full_source_k_sensitivity(
    broad_vina: np.ndarray,
    experimental_maps: dict[str, np.ndarray],
    *,
    permutations: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Exact raw/residual transfer over the fixed prespecified k grid."""
    if permutations < 100:
        raise ValueError("choice-grid adjustment requires at least 100 permutations")
    records: list[dict[str, Any]] = []
    selections: list[dict[str, Any]] = []
    choice_cells: dict[tuple[str, int], dict[str, Any]] = {}
    for k in K_GRID:
        panels = boundary.all_panels(n_targets=len(TARGETS), k=k)
        maps = {
            "raw": reliability.target_correlation(broad_vina),
            "row_centered_residual": boundary.target_map(broad_vina),
        }
        for transformation, vina_map in maps.items():
            vina_distance = boundary.distance_from_map(vina_map, "signed_1_minus_r")
            vina_distribution = boundary.coverage_distribution(vina_distance, panels)
            selected_index = int(np.argmin(vina_distribution))
            selected = panels[selected_index]
            cell_payloads: dict[str, dict[str, np.ndarray]] = {}
            for panel_name in ("DAVIS", "PKIS2"):
                matrix = experimental_maps[panel_name]
                experiment_map = (
                    reliability.target_correlation(matrix)
                    if transformation == "raw"
                    else boundary.target_map(matrix)
                )
                experiment_distance = boundary.distance_from_map(
                    experiment_map, "signed_1_minus_r"
                )
                distribution = boundary.coverage_distribution(
                    experiment_distance, panels
                )
                metrics = panel_metrics(distribution, selected_index)
                cell_payloads[panel_name] = {
                    "distance": experiment_distance,
                    "sorted_distribution": np.sort(distribution),
                }
                records.append(
                    {
                        "targets_selected_k": k,
                        "transformation": transformation,
                        "evaluation_map": panel_name,
                        "all_possible_panels": int(len(panels)),
                        **metrics,
                    }
                )
            choice_cells[(transformation, k)] = {
                "selected": selected.astype(np.int16),
                "experimental": cell_payloads,
            }
            for target_index in selected:
                selections.append(
                    {
                        "targets_selected_k": k,
                        "transformation": transformation,
                        "target_index": int(target_index),
                        "target": TARGETS[int(target_index)],
                    }
                )
    frame = pd.DataFrame.from_records(records)
    paired = frame.pivot(
        index=["targets_selected_k", "evaluation_map"],
        columns="transformation",
        values=[
            "fraction_all_panels_no_worse",
            "absolute_loss_vs_oracle",
            "median_headroom_captured",
        ],
    )
    paired.columns = [f"{metric}__{transform}" for metric, transform in paired.columns]
    paired = paired.reset_index()
    paired["residual_minus_raw_fraction_all_panels_no_worse"] = (
        paired["fraction_all_panels_no_worse__row_centered_residual"]
        - paired["fraction_all_panels_no_worse__raw"]
    )
    paired["residual_minus_raw_absolute_loss"] = (
        paired["absolute_loss_vs_oracle__row_centered_residual"]
        - paired["absolute_loss_vs_oracle__raw"]
    )
    paired["residual_minus_raw_median_headroom_captured"] = (
        paired["median_headroom_captured__row_centered_residual"]
        - paired["median_headroom_captured__raw"]
    )
    frame = frame.merge(
        paired[
            [
                "targets_selected_k",
                "evaluation_map",
                "residual_minus_raw_fraction_all_panels_no_worse",
                "residual_minus_raw_absolute_loss",
                "residual_minus_raw_median_headroom_captured",
            ]
        ],
        on=["targets_selected_k", "evaluation_map"],
        how="left",
        validate="many_to_one",
    )
    # Choice-grid multiplicity adjustment.  One target-label permutation is
    # shared across every raw/residual x k cell in each replicate.  The cell
    # score is the worse (maximum) exact percentile across DAVIS and PKIS2; the
    # exploratory global statistic is the best (minimum) cell score.
    observed_cell_scores: dict[tuple[str, int], float] = {}
    null_cell_scores = {
        cell: np.empty(permutations, dtype=np.float64)
        for cell in choice_cells
    }
    rng = np.random.default_rng(seed)
    for cell, payload in choice_cells.items():
        selected = payload["selected"]
        observed_cell_scores[cell] = max(
            exact_percentile(
                payload["experimental"][panel]["sorted_distribution"],
                coverage_value(
                    payload["experimental"][panel]["distance"], selected
                ),
            )
            for panel in ("DAVIS", "PKIS2")
        )
    null_best = np.empty(permutations, dtype=np.float64)
    for repetition in range(permutations):
        permutation = rng.permutation(len(TARGETS))
        scores = []
        for cell, payload in choice_cells.items():
            permuted = np.sort(permutation[payload["selected"]])
            score = max(
                exact_percentile(
                    payload["experimental"][panel]["sorted_distribution"],
                    coverage_value(
                        payload["experimental"][panel]["distance"], permuted
                    ),
                )
                for panel in ("DAVIS", "PKIS2")
            )
            null_cell_scores[cell][repetition] = score
            scores.append(score)
        null_best[repetition] = min(scores)
    for (transformation, k), observed in observed_cell_scores.items():
        p_value = float(
            (1 + np.sum(null_cell_scores[(transformation, k)] <= observed))
            / (permutations + 1)
        )
        mask = frame.transformation.eq(transformation) & frame.targets_selected_k.eq(k)
        frame.loc[mask, "target_label_permutation_p_two_panel_max_percentile"] = p_value
        frame.loc[mask, "observed_two_panel_max_percentile"] = observed
    observed_best = min(observed_cell_scores.values())
    global_p = float(
        (1 + np.sum(null_best <= observed_best))
        / (permutations + 1)
    )
    integrated: list[dict[str, Any]] = []
    for (transformation, panel_name), group in frame.groupby(
        ["transformation", "evaluation_map"], sort=True
    ):
        ordered = group.sort_values("targets_selected_k")
        k_values = ordered.targets_selected_k.to_numpy(float)
        integrated.append(
            {
                "transformation": transformation,
                "evaluation_map": panel_name,
                "k_grid": list(K_GRID),
                "mean_exact_percentile_over_k": float(
                    ordered.fraction_all_panels_no_worse.mean()
                ),
                "trapezoid_auc_percentile_over_k_divided_by_k_range": float(
                    normalized_trapezoid_auc(
                        ordered.fraction_all_panels_no_worse.to_numpy(float),
                        k_values,
                    )
                ),
                "mean_absolute_loss_over_k": float(
                    ordered.absolute_loss_vs_oracle.mean()
                ),
                "mean_median_headroom_captured_over_k": float(
                    ordered.median_headroom_captured.mean()
                ),
            }
        )
    metadata = {
        "k_grid": list(K_GRID),
        "objective": "signed_1_minus_r",
        "enumeration": "complete C(21,k) enumeration independently at every k",
        "integrated_over_k": integrated,
        "choice_grid_adjustment": {
            "choice_grid": (
                "raw and row-centered-residual transformations crossed with "
                f"k={list(K_GRID)}, signed distance 1-r"
            ),
            "cell_statistic": (
                "maximum of the exact DAVIS and PKIS2 all-panel percentiles; smaller "
                "requires a panel to rank well on both experimental maps"
            ),
            "global_statistic": "minimum cell statistic over the full choice grid",
            "observed_best_cell_statistic": float(observed_best),
            "target_label_permutation_p_choice_grid_adjusted": global_p,
            "permutations": permutations,
            "seed": seed,
            "permutation_contract": (
                "one common permutation of the 21 target labels is applied to every "
                "fixed Vina-selected set in a replicate; experimental percentile "
                "distributions and all selection choices remain fixed"
            ),
            "role": "exploratory analysis-choice-multiplicity adjustment",
        },
        "interpretation": (
            "k=8 is one inherited illustrative compression size and is exceptional in "
            "the observed residual transfer. Claims about panel compression must use "
            "the full k grid or explicitly label k=8 as illustrative."
        ),
    }
    return frame, pd.DataFrame.from_records(selections), metadata


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "calibration_vs_full_map_spearman",
        "panel_overlap_with_full_source",
        "DAVIS_fraction_all_panels_no_worse",
        "PKIS2_fraction_all_panels_no_worse",
        "DAVIS_absolute_loss_vs_oracle",
        "PKIS2_absolute_loss_vs_oracle",
        "DAVIS_median_headroom_captured",
        "PKIS2_median_headroom_captured",
    ]
    records: list[dict[str, Any]] = []
    for size, group in frame.groupby("calibration_ligands", sort=True):
        record: dict[str, Any] = {
            "calibration_ligands": int(size),
            "repetitions": int(len(group)),
            "fraction_top5pct_DAVIS": float(
                np.mean(group.DAVIS_fraction_all_panels_no_worse <= 0.05)
            ),
            "fraction_top5pct_PKIS2": float(
                np.mean(group.PKIS2_fraction_all_panels_no_worse <= 0.05)
            ),
            "fraction_top5pct_both_experimental_maps": float(
                np.mean(
                    (group.DAVIS_fraction_all_panels_no_worse <= 0.05)
                    & (group.PKIS2_fraction_all_panels_no_worse <= 0.05)
                )
            ),
        }
        for metric in metrics:
            values = group[metric].to_numpy(float)
            record[f"{metric}_median"] = float(np.median(values))
            record[f"{metric}_q025"] = float(np.quantile(values, 0.025))
            record[f"{metric}_q975"] = float(np.quantile(values, 0.975))
        records.append(record)
    return pd.DataFrame.from_records(records)


def analyse(
    *,
    sizes: tuple[int, ...],
    repetitions: int,
    seed: int,
    multiplicity_permutations: int = MULTIPLICITY_PERMUTATIONS,
    sequence_qap_permutations: int = SEQUENCE_QAP_PERMUTATIONS,
    cluster_multiplier_repetitions: int = CLUSTER_MULTIPLIER_REPETITIONS,
) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
    if repetitions < 20:
        raise ValueError("calibration audit requires at least 20 repetitions")
    dockstring = reliability.load_dockstring_support()
    panels = reliability.load_public_panels()
    matrix, broad_meta = reliability.broad_reference(dockstring, panels)
    if max(sizes) >= len(matrix) or min(sizes) < 3:
        raise ValueError("calibration sizes must lie inside the de-leaked support")

    all_k8 = boundary.all_panels()
    panel_to_index = {tuple(panel.tolist()): index for index, panel in enumerate(all_k8)}
    full_map = boundary.target_map(matrix)
    full_distance = boundary.distance_from_map(full_map, "signed_1_minus_r")
    full_distribution = boundary.coverage_distribution(full_distance, all_k8)
    full_index = int(np.argmin(full_distribution))
    full_panel = all_k8[full_index]

    experimental_matrices, experimental_support = experimental_reference()
    experimental_distributions = {
        name: boundary.coverage_distribution(
            boundary.distance_from_map(
                boundary.target_map(matrix), "signed_1_minus_r"
            ),
            all_k8,
        )
        for name, matrix in experimental_matrices.items()
    }
    full_transfer = {
        name: panel_metrics(distribution, full_index)
        for name, distribution in experimental_distributions.items()
    }
    k_sensitivity, k_selections, k_metadata = full_source_k_sensitivity(
        matrix,
        experimental_matrices,
        permutations=multiplicity_permutations,
        seed=seed + 9_000_000,
    )
    same_endpoint, same_endpoint_selections, same_endpoint_metadata = (
        same_endpoint_centering_comparison(
            matrix,
            experimental_matrices,
            permutations=multiplicity_permutations,
            seed=seed + 10_000_000,
        )
    )
    sequence_control, sequence_control_metadata = sequence_geometry_control(
        matrix,
        permutations=sequence_qap_permutations,
        seed=seed + 15_000_000,
    )
    same_endpoint_bootstrap, same_endpoint_bootstrap_metadata = (
        same_endpoint_cluster_multiplier_bootstrap(
            matrix,
            panels,
            repetitions=cluster_multiplier_repetitions,
            seed=seed + 20_000_000,
        )
    )

    records: list[dict[str, Any]] = []
    selection_records: list[dict[str, Any]] = []
    rng = np.random.default_rng(seed)
    for size in sizes:
        for repetition in range(repetitions):
            selected_rows = np.sort(rng.choice(len(matrix), size=size, replace=False))
            calibration_map = boundary.target_map(matrix[selected_rows])
            calibration_distance = boundary.distance_from_map(
                calibration_map, "signed_1_minus_r"
            )
            calibration_distribution = boundary.coverage_distribution(
                calibration_distance, all_k8
            )
            calibration_index = int(np.argmin(calibration_distribution))
            selected_panel = all_k8[calibration_index]
            overlap = len(set(selected_panel.tolist()) & set(full_panel.tolist()))
            record: dict[str, Any] = {
                "calibration_ligands": int(size),
                "repetition": repetition,
                "calibration_vs_full_map_spearman": boundary.map_spearman(
                    calibration_map, full_map
                ),
                "panel_overlap_with_full_source": int(overlap),
                "panel_jaccard_with_full_source": float(overlap / (2 * K - overlap)),
                "calibration_panel_index_lexicographic": calibration_index,
                "top5pct_on_both_experimental_maps": True,
            }
            for name, distribution in experimental_distributions.items():
                for metric, value in panel_metrics(distribution, calibration_index).items():
                    record[f"{name}_{metric}"] = value
                record["top5pct_on_both_experimental_maps"] &= bool(
                    record[f"{name}_fraction_all_panels_no_worse"] <= 0.05
                )
            records.append(record)
            for target_index in selected_panel:
                selection_records.append(
                    {
                        "selection_source": "calibration_sample",
                        "calibration_ligands": int(size),
                        "repetition": repetition,
                        "target_index": int(target_index),
                        "target": TARGETS[int(target_index)],
                    }
                )
    for target_index in full_panel:
        selection_records.append(
            {
                "selection_source": "full_deleaked_source",
                "calibration_ligands": int(len(matrix)),
                "repetition": -1,
                "target_index": int(target_index),
                "target": TARGETS[int(target_index)],
            }
        )
    frame = pd.DataFrame.from_records(records)
    summary_frame = summarize(frame)
    summary = {
        "analysis": "outcome-blind Vina calibration-size target-panel transfer",
        "status": "strict_public_fixed_target_panel_geometry",
        "sizes": list(sizes),
        "repetitions_per_size": int(repetitions),
        "seed": int(seed),
        "selection": (
            "exact minimizer among all C(21,8) panels of mean nearest signed distance "
            "1-r on the row-centered calibration Vina map"
        ),
        "evaluation": (
            "exact panel rank, loss and headroom on fixed released-informative DAVIS "
            "and PKIS2 row-centered maps; experimental values never enter selection"
        ),
        "all_possible_panels": int(len(all_k8)),
        "full_source_panel": [TARGETS[index] for index in full_panel],
        "full_source_transfer": full_transfer,
        "full_source_k_sensitivity": k_metadata,
        "same_endpoint_centering_comparison": same_endpoint_metadata,
        "sequence_geometry_control": sequence_control_metadata,
        "same_endpoint_cluster_multiplier_bootstrap": (
            same_endpoint_bootstrap_metadata
        ),
        "experimental_support": experimental_support,
        "broad_vina_support": broad_meta,
        "calibration_summary": summary_frame.to_dict(orient="records"),
        "claim_boundary": (
            "The audit measures stability of one k=8 decision on the fixed 21-target "
            "panel. It does not measure affinity accuracy, compound ranking or transfer "
            "to unseen targets. Repetitions perturb ligands, not targets or assays."
        ),
        "inputs": {
            str(path.relative_to(PACKAGE)): sha256_file(path)
            for path in (
                reliability.DEFAULT_DOCKSTRING,
                reliability.DEFAULT_DAVIS,
                reliability.DEFAULT_PKIS2,
                reliability.DEFAULT_IDENTITY_CONTRACT,
                SEQUENCE_FASTA,
                SEQUENCE_MANIFEST,
            )
        },
    }
    return summary, {
        "calibration_replicates.csv": frame,
        "calibration_summary.csv": summary_frame,
        "full_source_k_sensitivity.csv": k_sensitivity,
        "full_source_k_selections.csv": k_selections,
        "same_endpoint_centering_comparison.csv": same_endpoint,
        "same_endpoint_panel_selections.csv": same_endpoint_selections,
        "same_endpoint_cluster_multiplier_bootstrap.csv": same_endpoint_bootstrap,
        "sequence_geometry_control.csv": sequence_control,
        "panel_selections.csv": pd.DataFrame.from_records(selection_records),
    }


def readme(summary: dict[str, Any]) -> str:
    return f"""# Vina panel calibration-size transfer

This strict-public analysis selects an exact residual-signed k=8 Vina target
panel from random calibration supports of {', '.join(map(str, summary['sizes']))}
ligands and evaluates the resulting decision on fixed DAVIS and PKIS2 maps.
Selection is outcome-blind: no experimental value is used until evaluation.

The same-endpoint control compares raw- and residual-Vina panel selectors over
k={list(K_GRID)} while evaluating both selectors on the same residual
experimental map. Shared target-label permutations and positive Exp(1)
chemical-cluster multipliers quantify target-label and chemical-support
uncertainty, respectively.

All {summary['all_possible_panels']:,} possible eight-target panels are enumerated.
Results are conditional on the fixed 21 targets and do not validate affinity or
compound ranking.

```bash
python analysis/public_vina_panel_calibration_transfer.py
pytest -q analysis/test_public_vina_panel_calibration_transfer.py
```
"""


def produce(
    output: Path,
    *,
    sizes: tuple[int, ...],
    repetitions: int,
    seed: int,
    multiplicity_permutations: int = MULTIPLICITY_PERMUTATIONS,
    sequence_qap_permutations: int = SEQUENCE_QAP_PERMUTATIONS,
    cluster_multiplier_repetitions: int = CLUSTER_MULTIPLIER_REPETITIONS,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    summary, tables = analyse(
        sizes=sizes,
        repetitions=repetitions,
        seed=seed,
        multiplicity_permutations=multiplicity_permutations,
        sequence_qap_permutations=sequence_qap_permutations,
        cluster_multiplier_repetitions=cluster_multiplier_repetitions,
    )
    for filename, frame in tables.items():
        write_csv(output / filename, frame)
    write_json(output / "summary.json", summary)
    (output / "README.md").write_text(readme(summary), encoding="utf-8")
    checksums = {
        filename: sha256_file(output / filename) for filename in OUTPUT_FILES
    }
    write_json(output / "output_checksums.json", checksums)


def parse_sizes(value: str) -> tuple[int, ...]:
    result = tuple(int(item) for item in value.split(","))
    if len(set(result)) != len(result):
        raise argparse.ArgumentTypeError("sizes must be unique")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sizes", type=parse_sizes, default=SIZES)
    parser.add_argument("--repetitions", type=int, default=REPETITIONS)
    parser.add_argument(
        "--multiplicity-permutations",
        type=int,
        default=MULTIPLICITY_PERMUTATIONS,
    )
    parser.add_argument(
        "--cluster-multiplier-repetitions",
        type=int,
        default=CLUSTER_MULTIPLIER_REPETITIONS,
    )
    parser.add_argument(
        "--sequence-qap-permutations",
        type=int,
        default=SEQUENCE_QAP_PERMUTATIONS,
    )
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    produce(
        args.output,
        sizes=args.sizes,
        repetitions=args.repetitions,
        seed=args.seed,
        multiplicity_permutations=args.multiplicity_permutations,
        sequence_qap_permutations=args.sequence_qap_permutations,
        cluster_multiplier_repetitions=args.cluster_multiplier_repetitions,
    )


if __name__ == "__main__":
    main()
