#!/usr/bin/env python3
"""Retrieve experimentally replicated co-selective target pairs.

This exploratory analysis keeps the experimental endpoint fixed while comparing
raw and two-way-centered DOCKSTRING target geometries.  A target pair is primary-
endpoint positive when it belongs to the upper 10% of the centered experimental
target-correlation geometry in at least two of three dense panels (DAVIS, PKIS2,
and PKIS1).  Target-label QAP permutes only the docking predictor labels.

The endpoint is intended as an operational panel-redundancy/co-selectivity check,
not as a ligand-level binding or target-retrieval benchmark.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats

try:  # Direct script execution and package-style import are both supported.
    from . import dense_davis_benchmark as davis
    from . import residual_target_geometry_validation as geometry
except ImportError:  # pragma: no cover - exercised by direct CLI execution.
    import dense_davis_benchmark as davis  # type: ignore
    import residual_target_geometry_validation as geometry  # type: ignore


PACKAGE = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PACKAGE / "results" / "replicated_pair_retrieval"
DEFAULT_SEED = 20260803
PRIMARY_FRACTION = 0.10
THRESHOLD_GRID = (0.05, 0.075, 0.10, 0.125, 0.15, 0.20, 0.25, 0.30)
OMNIBUS_THRESHOLDS = (0.05, 0.10, 0.15, 0.20, 0.25)
EXPERIMENTAL_TRANSFORMS = (
    "center_then_correlation",
    "z_before_center",
    "rank_normal_before_center",
)
STANDARD_AA3_TO_1 = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
}


def upper_triangle(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("upper_triangle requires a square matrix")
    return matrix[np.triu_indices(len(matrix), k=1)]


def percentile_ranks(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("percentile_ranks requires a finite one-dimensional vector")
    return (stats.rankdata(values, method="average") - 0.5) / len(values)


def upper_tail_labels(values: np.ndarray, fraction: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("upper_tail_labels requires a finite vector")
    if not 0 < fraction < 1:
        raise ValueError("fraction must lie strictly between zero and one")
    positive_count = max(1, int(np.ceil(fraction * len(values))))
    labels = np.zeros(len(values), dtype=bool)
    order = np.argsort(values, kind="mergesort")
    labels[order[-positive_count:]] = True
    return labels


def replicated_labels(
    panel_percentile_ranks: dict[str, np.ndarray],
    fraction: float,
    minimum_panels: int = 2,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    if len(panel_percentile_ranks) < minimum_panels:
        raise ValueError("minimum_panels exceeds the number of panels")
    panel_labels = {
        name: upper_tail_labels(values, fraction)
        for name, values in panel_percentile_ranks.items()
    }
    lengths = {len(values) for values in panel_labels.values()}
    if len(lengths) != 1:
        raise ValueError("panel vectors do not share a common target-pair support")
    count = np.sum(np.vstack(list(panel_labels.values())), axis=0)
    return count >= minimum_panels, panel_labels


def retrieval_metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=np.float64)
    if labels.shape != scores.shape or labels.ndim != 1:
        raise ValueError("labels and scores must be aligned vectors")
    positives = int(labels.sum())
    negatives = int(len(labels) - positives)
    if positives < 1 or negatives < 1:
        raise ValueError("retrieval requires both outcome classes")
    ranks = stats.rankdata(scores, method="average")
    roc_auc = (
        float(ranks[labels].sum()) - positives * (positives + 1) / 2
    ) / (positives * negatives)
    order = np.argsort(scores, kind="mergesort")[::-1]
    ordered = labels[order].astype(np.int64)
    precision = np.cumsum(ordered) / (np.arange(len(ordered)) + 1)
    average_precision = float(np.sum(ordered * precision) / positives)
    return {"roc_auc": float(roc_auc), "average_precision": average_precision}


def fixed_endpoint_qap(
    labels: np.ndarray,
    raw_geometry: np.ndarray,
    centered_geometry: np.ndarray,
    permutations: int,
    seed: int,
) -> dict:
    """Target-label QAP for two predictors of one unchanged endpoint."""
    labels = np.asarray(labels, dtype=bool)
    p = len(raw_geometry)
    if raw_geometry.shape != (p, p) or centered_geometry.shape != (p, p):
        raise ValueError("docking geometries must be aligned square matrices")
    if len(labels) != p * (p - 1) // 2:
        raise ValueError("endpoint length does not match target-pair support")
    if permutations < 1:
        raise ValueError("permutations must be positive")
    tri = np.triu_indices(p, k=1)
    observed_raw = retrieval_metrics(labels, raw_geometry[tri])
    observed_centered = retrieval_metrics(labels, centered_geometry[tri])
    metrics = tuple(observed_raw)
    observed_delta = {
        metric: observed_centered[metric] - observed_raw[metric]
        for metric in metrics
    }
    null_raw = {metric: np.empty(permutations) for metric in metrics}
    null_centered = {metric: np.empty(permutations) for metric in metrics}
    null_delta = {metric: np.empty(permutations) for metric in metrics}
    rng = np.random.default_rng(seed)
    for repetition in range(permutations):
        order = rng.permutation(p)
        raw_metrics = retrieval_metrics(
            labels, raw_geometry[np.ix_(order, order)][tri]
        )
        centered_metrics = retrieval_metrics(
            labels, centered_geometry[np.ix_(order, order)][tri]
        )
        for metric in metrics:
            null_raw[metric][repetition] = raw_metrics[metric]
            null_centered[metric][repetition] = centered_metrics[metric]
            null_delta[metric][repetition] = (
                centered_metrics[metric] - raw_metrics[metric]
            )
    result: dict[str, dict] = {}
    for metric in metrics:
        result[metric] = {
            "raw_docking": float(observed_raw[metric]),
            "raw_target_label_qap_p": float(
                (1 + np.sum(null_raw[metric] >= observed_raw[metric]))
                / (permutations + 1)
            ),
            "centered_docking": float(observed_centered[metric]),
            "centered_target_label_qap_p": float(
                (1 + np.sum(null_centered[metric] >= observed_centered[metric]))
                / (permutations + 1)
            ),
            "centered_minus_raw": float(observed_delta[metric]),
            "paired_target_label_qap_p_positive_gain": float(
                (1 + np.sum(null_delta[metric] >= observed_delta[metric]))
                / (permutations + 1)
            ),
            "paired_delta_null_interval_95": [
                float(np.quantile(null_delta[metric], 0.025)),
                float(np.quantile(null_delta[metric], 0.975)),
            ],
        }
    return {
        "permutations": int(permutations),
        "seed": int(seed),
        "metrics": result,
    }


def endpoint_metrics(
    labels: np.ndarray,
    raw_geometry: np.ndarray,
    centered_geometry: np.ndarray,
) -> dict:
    raw = retrieval_metrics(labels, upper_triangle(raw_geometry))
    centered = retrieval_metrics(labels, upper_triangle(centered_geometry))
    return {
        "positive_pairs": int(np.sum(labels)),
        "total_pairs": int(len(labels)),
        "prevalence": float(np.mean(labels)),
        "raw_docking": raw,
        "centered_docking": centered,
        "centered_minus_raw": {
            metric: float(centered[metric] - raw[metric]) for metric in raw
        },
    }


def correlation_pr(correlation: np.ndarray) -> float:
    correlation = np.asarray(correlation, dtype=np.float64)
    if correlation.ndim != 2 or correlation.shape[0] != correlation.shape[1]:
        raise ValueError("correlation_pr requires a square matrix")
    eigenvalues = np.linalg.eigvalsh(correlation)
    return float(eigenvalues.sum() ** 2 / np.sum(eigenvalues**2))


def leading_mode_reconstruction(correlation: np.ndarray) -> np.ndarray:
    """Return lambda_1 v_1 v_1^T without diagonal renormalization."""
    eigenvalues, eigenvectors = np.linalg.eigh(correlation)
    leading = int(np.argmax(eigenvalues))
    vector = eigenvectors[:, leading]
    return eigenvalues[leading] * np.outer(vector, vector)


def cumulative_mode_retrieval(
    labels: np.ndarray,
    correlation: np.ndarray,
    experimental_pair_ranks: dict[str, np.ndarray] | None = None,
) -> pd.DataFrame:
    eigenvalues, eigenvectors = np.linalg.eigh(correlation)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    tri = np.triu_indices(len(correlation), k=1)
    rows: list[dict] = []
    for count in range(1, len(correlation) + 1):
        approximation = (
            eigenvectors[:, :count] * eigenvalues[:count]
        ) @ eigenvectors[:, :count].T
        values = retrieval_metrics(labels, approximation[tri])
        row = {
            "cumulative_centered_modes": count,
            "cumulative_eigenvalue_fraction": float(
                np.sum(eigenvalues[:count]) / np.sum(eigenvalues)
            ),
            "roc_auc": values["roc_auc"],
            "average_precision": values["average_precision"],
        }
        if experimental_pair_ranks is not None:
            concordances = []
            for panel, ranks in experimental_pair_ranks.items():
                concordance = float(stats.spearmanr(approximation[tri], ranks).statistic)
                row[f"{panel}_continuous_geometry_spearman"] = concordance
                concordances.append(concordance)
            row["mean_continuous_geometry_spearman"] = float(
                np.mean(concordances)
            )
        rows.append(row)
    return pd.DataFrame(rows)


def holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    """Holm adjustment for a small named p-value family."""
    ordered = sorted(p_values, key=p_values.get)
    adjusted: dict[str, float] = {}
    running = 0.0
    total = len(ordered)
    for index, name in enumerate(ordered):
        value = min(1.0, (total - index) * p_values[name])
        running = max(running, value)
        adjusted[name] = running
    return adjusted


def _pdbqt_ca_sequence(path: Path) -> str:
    residues: list[str] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("ATOM") and line[12:16].strip() == "CA":
                residues.append(STANDARD_AA3_TO_1.get(line[17:20].strip(), "X"))
    if len(residues) < 100:
        raise ValueError(f"unexpectedly short receptor sequence in {path.name}")
    return "".join(residues)


def _global_sequence_identity(first: str, second: str, aligner: object) -> float:
    alignment = aligner.align(first, second)[0]  # type: ignore[attr-defined]
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
        raise ValueError("empty global alignment")
    return float(matches / columns)


def receptor_sequence_identity(
    targets: tuple[str, ...], dockstring_source_root: Path
) -> tuple[np.ndarray, pd.DataFrame, dict]:
    """Global identities of the exact receptor domains used by DOCKSTRING."""
    try:
        from Bio import Align
        from Bio.Align import substitution_matrices
        import Bio
    except ImportError as error:  # pragma: no cover - dependency gate.
        raise RuntimeError(
            "Biopython is required when --dockstring-source-root is supplied"
        ) from error
    target_dir = dockstring_source_root / "dockstring" / "resources" / "targets"
    if not target_dir.is_dir():
        raise ValueError("DOCKSTRING source root does not contain receptor resources")
    sequences: dict[str, str] = {}
    checksums: dict[str, str] = {}
    aggregate = hashlib.sha256()
    for target in targets:
        path = target_dir / f"{target}_target.pdbqt"
        if not path.is_file():
            raise ValueError(f"missing DOCKSTRING receptor PDBQT for {target}")
        sequences[target] = _pdbqt_ca_sequence(path)
        checksum = geometry.sha256_file(path)
        checksums[target] = checksum
        aggregate.update(target.encode("utf-8") + b"\0" + bytes.fromhex(checksum))
    aligner = Align.PairwiseAligner()
    aligner.mode = "global"
    aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
    aligner.open_gap_score = -10.0
    aligner.extend_gap_score = -0.5
    identity = np.eye(len(targets), dtype=np.float64)
    rows: list[dict] = []
    for first in range(len(targets)):
        for second in range(first + 1, len(targets)):
            value = _global_sequence_identity(
                sequences[targets[first]], sequences[targets[second]], aligner
            )
            identity[first, second] = identity[second, first] = value
            rows.append(
                {
                    "target_a": targets[first],
                    "target_b": targets[second],
                    "receptor_domain_sequence_identity": value,
                }
            )
    provenance = {
        "source": "DOCKSTRING receptor PDBQT CA sequences",
        "source_repository": "https://github.com/dockstring/dockstring",
        "source_paper_doi": "10.1021/acs.jcim.1c01334",
        "resource_aggregate_sha256": aggregate.hexdigest(),
        "individual_resource_sha256": checksums,
        "nonstandard_residue_handling": "map to X",
        "alignment": "global BLOSUM62; gap-open -10; gap-extend -0.5",
        "identity_denominator": "all alignment columns, including gaps",
        "biopython_version": Bio.__version__,
        "sequence_length_range": [
            int(min(map(len, sequences.values()))),
            int(max(map(len, sequences.values()))),
        ],
    }
    return identity, pd.DataFrame(rows), provenance


def _linear_residual(values: np.ndarray, covariate: np.ndarray) -> np.ndarray:
    design = np.column_stack([np.ones(len(covariate)), covariate])
    coefficients = np.linalg.lstsq(design, values, rcond=None)[0]
    return values - design @ coefficients


def sequence_identity_control(
    labels: np.ndarray,
    raw_geometry: np.ndarray,
    centered_geometry: np.ndarray,
    sequence_identity: np.ndarray,
    permutations: int,
    seed: int,
    low_identity_threshold: float = 0.40,
) -> dict:
    """Benchmark and condition on receptor-domain sequence identity."""
    p = len(raw_geometry)
    tri = np.triu_indices(p, k=1)
    sequence_scores = sequence_identity[tri]
    raw_scores = raw_geometry[tri]
    centered_scores = centered_geometry[tri]
    sequence_metrics = retrieval_metrics(labels, sequence_scores)
    raw_metrics = retrieval_metrics(labels, raw_scores)
    centered_metrics = retrieval_metrics(labels, centered_scores)
    sequence_ranks = percentile_ranks(sequence_scores)
    raw_ranks = percentile_ranks(raw_scores)
    centered_ranks = percentile_ranks(centered_scores)
    endpoint_residual = _linear_residual(labels.astype(float), sequence_ranks)
    partial_observed = {
        "raw_docking": float(
            np.corrcoef(
                _linear_residual(raw_ranks, sequence_ranks), endpoint_residual
            )[0, 1]
        ),
        "centered_docking": float(
            np.corrcoef(
                _linear_residual(centered_ranks, sequence_ranks), endpoint_residual
            )[0, 1]
        ),
    }
    ensemble_scores = 0.5 * sequence_ranks + 0.5 * centered_ranks
    ensemble_metrics = retrieval_metrics(labels, ensemble_scores)
    ensemble_delta = {
        metric: ensemble_metrics[metric] - sequence_metrics[metric]
        for metric in sequence_metrics
    }
    low_identity = sequence_scores < low_identity_threshold
    if int(labels[low_identity].sum()) < 2:
        raise ValueError("low-identity stratum has too few positive target pairs")
    low_raw = retrieval_metrics(labels[low_identity], raw_scores[low_identity])
    low_centered = retrieval_metrics(
        labels[low_identity], centered_scores[low_identity]
    )
    low_sequence = retrieval_metrics(
        labels[low_identity], sequence_scores[low_identity]
    )
    low_delta = {
        metric: low_centered[metric] - low_raw[metric] for metric in low_raw
    }

    rng = np.random.default_rng(seed)
    partial_null = {
        "raw_docking": np.empty(permutations),
        "centered_docking": np.empty(permutations),
    }
    ensemble_delta_null = {
        metric: np.empty(permutations) for metric in ensemble_delta
    }
    low_centered_null = {metric: np.empty(permutations) for metric in low_delta}
    low_delta_null = {metric: np.empty(permutations) for metric in low_delta}
    for repetition in range(permutations):
        order = rng.permutation(p)
        permuted_raw = raw_geometry[np.ix_(order, order)][tri]
        permuted_centered = centered_geometry[np.ix_(order, order)][tri]
        partial_null["raw_docking"][repetition] = np.corrcoef(
            _linear_residual(percentile_ranks(permuted_raw), sequence_ranks),
            endpoint_residual,
        )[0, 1]
        partial_null["centered_docking"][repetition] = np.corrcoef(
            _linear_residual(percentile_ranks(permuted_centered), sequence_ranks),
            endpoint_residual,
        )[0, 1]
        permuted_ensemble = 0.5 * sequence_ranks + 0.5 * percentile_ranks(
            permuted_centered
        )
        permuted_ensemble_metrics = retrieval_metrics(labels, permuted_ensemble)
        for metric in ensemble_delta:
            ensemble_delta_null[metric][repetition] = (
                permuted_ensemble_metrics[metric] - sequence_metrics[metric]
            )
        permuted_low_raw = retrieval_metrics(
            labels[low_identity], permuted_raw[low_identity]
        )
        permuted_low_centered = retrieval_metrics(
            labels[low_identity], permuted_centered[low_identity]
        )
        for metric in low_delta:
            low_centered_null[metric][repetition] = permuted_low_centered[metric]
            low_delta_null[metric][repetition] = (
                permuted_low_centered[metric] - permuted_low_raw[metric]
            )
    return {
        "standalone_baselines": {
            "sequence_identity": sequence_metrics,
            "raw_docking": raw_metrics,
            "centered_docking": centered_metrics,
            "boundary": (
                "Sequence identity is the stronger standalone baseline when its "
                "average precision exceeds centered docking."
            ),
        },
        "partial_rank_association_controlling_sequence_identity": {
            name: {
                "partial_correlation": value,
                "target_label_qap_p_positive": float(
                    (1 + np.sum(partial_null[name] >= value))
                    / (permutations + 1)
                ),
            }
            for name, value in partial_observed.items()
        },
        "equal_rank_sequence_centered_docking_ensemble": {
            "definition": "0.5 sequence-identity percentile + 0.5 centered-docking percentile",
            "point_estimate": ensemble_metrics,
            "minus_sequence_identity_alone": ensemble_delta,
            "target_label_qap_p_at_least_observed_delta": {
                metric: float(
                    (1 + np.sum(ensemble_delta_null[metric] >= value))
                    / (permutations + 1)
                )
                for metric, value in ensemble_delta.items()
            },
            "boundary": (
                "A null-tail probability for a negative observed delta is not "
                "evidence of a positive gain."
            ),
        },
        "low_homology_stratum": {
            "definition": f"receptor-domain sequence identity < {low_identity_threshold:.2f}",
            "comparison_is_exploratory": True,
            "target_pairs": int(low_identity.sum()),
            "positive_pairs": int(labels[low_identity].sum()),
            "sequence_identity": low_sequence,
            "raw_docking": low_raw,
            "centered_docking": low_centered,
            "centered_minus_raw": low_delta,
            "centered_target_label_qap_p": {
                metric: float(
                    (1 + np.sum(low_centered_null[metric] >= low_centered[metric]))
                    / (permutations + 1)
                )
                for metric in low_delta
            },
            "paired_target_label_qap_p_positive_gain": {
                metric: float(
                    (1 + np.sum(low_delta_null[metric] >= low_delta[metric]))
                    / (permutations + 1)
                )
                for metric in low_delta
            },
        },
        "permutations": int(permutations),
        "seed": int(seed),
    }


def target_delete_one(
    targets: tuple[str, ...],
    labels: np.ndarray,
    raw_geometry: np.ndarray,
    centered_geometry: np.ndarray,
) -> pd.DataFrame:
    p = len(targets)
    tri = np.triu_indices(p, k=1)
    label_matrix = np.zeros((p, p), dtype=bool)
    label_matrix[tri] = labels
    label_matrix[(tri[1], tri[0])] = labels
    rows: list[dict] = []
    for omitted in range(p):
        keep = np.array([index for index in range(p) if index != omitted])
        subset = np.ix_(keep, keep)
        subset_tri = np.triu_indices(p - 1, k=1)
        subset_labels = label_matrix[subset][subset_tri]
        raw = retrieval_metrics(subset_labels, raw_geometry[subset][subset_tri])
        centered = retrieval_metrics(
            subset_labels, centered_geometry[subset][subset_tri]
        )
        rows.append(
            {
                "omitted_target": targets[omitted],
                "remaining_positive_pairs": int(subset_labels.sum()),
                "raw_roc_auc": raw["roc_auc"],
                "centered_roc_auc": centered["roc_auc"],
                "centered_minus_raw_roc_auc": (
                    centered["roc_auc"] - raw["roc_auc"]
                ),
                "raw_average_precision": raw["average_precision"],
                "centered_average_precision": centered["average_precision"],
                "centered_minus_raw_average_precision": (
                    centered["average_precision"] - raw["average_precision"]
                ),
            }
        )
    return pd.DataFrame(rows)


def omnibus_threshold_qap(
    endpoint_labels: list[np.ndarray],
    raw_geometry: np.ndarray,
    centered_geometry: np.ndarray,
    permutations: int,
    seed: int,
) -> dict:
    """Average paired gain over a declared threshold family."""
    if not endpoint_labels:
        raise ValueError("at least one threshold endpoint is required")
    p = len(raw_geometry)
    tri = np.triu_indices(p, k=1)

    def mean_gain(raw: np.ndarray, centered: np.ndarray) -> np.ndarray:
        gains = []
        for labels in endpoint_labels:
            first = retrieval_metrics(labels, raw[tri])
            second = retrieval_metrics(labels, centered[tri])
            gains.append(
                [
                    second["roc_auc"] - first["roc_auc"],
                    second["average_precision"] - first["average_precision"],
                ]
            )
        return np.mean(np.asarray(gains, dtype=np.float64), axis=0)

    observed = mean_gain(raw_geometry, centered_geometry)
    null = np.empty((permutations, 2), dtype=np.float64)
    rng = np.random.default_rng(seed)
    for repetition in range(permutations):
        order = rng.permutation(p)
        null[repetition] = mean_gain(
            raw_geometry[np.ix_(order, order)],
            centered_geometry[np.ix_(order, order)],
        )
    return {
        "permutations": int(permutations),
        "seed": int(seed),
        "mean_centered_minus_raw": {
            "roc_auc": float(observed[0]),
            "average_precision": float(observed[1]),
        },
        "paired_target_label_qap_p_positive_gain": {
            "roc_auc": float(
                (1 + np.sum(null[:, 0] >= observed[0])) / (permutations + 1)
            ),
            "average_precision": float(
                (1 + np.sum(null[:, 1] >= observed[1])) / (permutations + 1)
            ),
        },
        "paired_delta_null_interval_95": {
            "roc_auc": [
                float(np.quantile(null[:, 0], 0.025)),
                float(np.quantile(null[:, 0], 0.975)),
            ],
            "average_precision": [
                float(np.quantile(null[:, 1], 0.025)),
                float(np.quantile(null[:, 1], 0.975)),
            ],
        },
    }


def _load_davis_experiment() -> pd.DataFrame:
    frame = pd.read_csv(
        davis.DEFAULT_DAVIS,
        sep="\t",
        usecols=["drug_name", "protein", "y"],
    )
    selected = frame[frame.protein.isin(davis.TARGET_MAP.values())]
    experiment = selected.pivot(index="drug_name", columns="protein", values="y")
    experiment = experiment.reindex(columns=list(davis.TARGET_MAP.values()))
    experiment.columns = list(davis.TARGET_MAP)
    if experiment.shape != (72, 21) or experiment.isna().any().any():
        raise ValueError("unexpected DAVIS experimental support")
    return experiment


def panel_rank_vectors(
    experimental_matrices: dict[str, np.ndarray], transform: str
) -> dict[str, np.ndarray]:
    return {
        name: percentile_ranks(
            upper_triangle(geometry.geometry_correlation(matrix, transform))
        )
        for name, matrix in experimental_matrices.items()
    }


def _numeric_range(values: Iterable[float]) -> dict[str, float]:
    array = np.asarray(list(values), dtype=np.float64)
    return {"minimum": float(array.min()), "maximum": float(array.max())}


def run_analysis(
    *,
    pkis1_zip: Path,
    qap_permutations: int,
    reference_support_size: int,
    reference_support_seeds: tuple[int, ...],
    seed: int,
    dockstring_source_root: Path | None,
) -> tuple[
    dict,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    if qap_permutations < 1:
        raise ValueError("qap_permutations must be positive")
    if reference_support_size < 100:
        raise ValueError("reference_support_size is unexpectedly small")

    # The full identity scan is intentionally shared with the validated geometry
    # analysis so the leakage exclusion is exactly aligned.
    dockstring, davis_identity, davis_experiment, _ = davis._load_inputs(
        davis.DEFAULT_DOCKSTRING, davis.DEFAULT_DAVIS, identity_scan="full"
    )
    pkis2_frame = geometry.load_pkis2_full()
    pkis1_frame = geometry.load_pkis1_full(pkis1_zip)
    targets = tuple(geometry.PKIS1_TARGET_MAP)
    excluded_blocks = set(davis_identity.connectivity_block.dropna())
    excluded_blocks |= set(pkis2_frame.connectivity_block.dropna())
    excluded_blocks |= set(pkis1_frame.connectivity_block.dropna())
    reference_keep = ~dockstring.connectivity_block.isin(excluded_blocks)
    reference_unclipped = dockstring.loc[reference_keep, list(targets)].to_numpy(
        dtype=np.float64
    )
    reference = np.minimum(reference_unclipped, 0.0)
    if len(reference) < 259_000:
        raise ValueError("unexpectedly many DOCKSTRING rows were excluded")

    # Reuse the rigorously ordered DAVIS block returned above.  The small direct
    # loader exists for tests and future reuse but is not needed in this path.
    experimental_matrices = {
        "DAVIS": davis_experiment[list(targets)].to_numpy(dtype=np.float64),
        "PKIS2": pkis2_frame[list(targets)].to_numpy(dtype=np.float64),
        "PKIS1": pkis1_frame[list(targets)].to_numpy(dtype=np.float64),
    }
    raw_geometry = geometry.geometry_correlation(reference, "raw")
    centered_geometry = geometry.geometry_correlation(
        reference, "center_then_correlation"
    )

    primary_ranks = panel_rank_vectors(
        experimental_matrices, "center_then_correlation"
    )
    primary_labels, primary_panel_labels = replicated_labels(
        primary_ranks, PRIMARY_FRACTION
    )
    primary_metrics = endpoint_metrics(
        primary_labels, raw_geometry, centered_geometry
    )
    primary_qap = fixed_endpoint_qap(
        primary_labels,
        raw_geometry,
        centered_geometry,
        qap_permutations,
        seed,
    )
    primary_holm = holm_adjust(
        {
            metric: record["paired_target_label_qap_p_positive_gain"]
            for metric, record in primary_qap["metrics"].items()
        }
    )

    # This post hoc but directly discriminating check asks whether the practical
    # signal is exhausted by one residual mode.  It compares the complete centered
    # geometry with lambda_1 v_1 v_1^T on the same fixed experimental labels.
    leading_mode = leading_mode_reconstruction(centered_geometry)
    full_vs_leading = fixed_endpoint_qap(
        primary_labels,
        leading_mode,
        centered_geometry,
        qap_permutations,
        seed + 50_000,
    )
    mode_frame = cumulative_mode_retrieval(
        primary_labels, centered_geometry, primary_ranks
    )
    mode_jackknife = target_delete_one(
        targets, primary_labels, leading_mode, centered_geometry
    )

    tri = np.triu_indices(len(targets), k=1)
    raw_percentiles = percentile_ranks(raw_geometry[tri])
    centered_percentiles = percentile_ranks(centered_geometry[tri])
    pair_rows: list[dict] = []
    for pair_index, (first, second) in enumerate(zip(*tri)):
        panel_hits = int(
            sum(labels[pair_index] for labels in primary_panel_labels.values())
        )
        pair_rows.append(
            {
                "target_a": targets[first],
                "target_b": targets[second],
                "primary_replicated_positive": bool(primary_labels[pair_index]),
                "number_of_panels_in_top_10_percent": panel_hits,
                "DAVIS_centered_pair_percentile": primary_ranks["DAVIS"][pair_index],
                "PKIS2_centered_pair_percentile": primary_ranks["PKIS2"][pair_index],
                "PKIS1_centered_pair_percentile": primary_ranks["PKIS1"][pair_index],
                "mean_experimental_centered_pair_percentile": float(
                    np.mean([values[pair_index] for values in primary_ranks.values()])
                ),
                "raw_docking_pair_percentile": raw_percentiles[pair_index],
                "centered_docking_pair_percentile": centered_percentiles[pair_index],
            }
        )
    pair_frame = pd.DataFrame(pair_rows).sort_values(
        ["primary_replicated_positive", "mean_experimental_centered_pair_percentile"],
        ascending=[False, False],
        kind="mergesort",
    )
    sequence_control = None
    sequence_provenance = None
    if dockstring_source_root is not None:
        sequence_matrix, sequence_pairs, sequence_provenance = (
            receptor_sequence_identity(targets, dockstring_source_root)
        )
        sequence_control = sequence_identity_control(
            primary_labels,
            raw_geometry,
            centered_geometry,
            sequence_matrix,
            qap_permutations,
            seed + 75_000,
        )
        pair_frame = pair_frame.merge(
            sequence_pairs, on=["target_a", "target_b"], how="left", validate="1:1"
        )
        if pair_frame.receptor_domain_sequence_identity.isna().any():
            raise ValueError("sequence identity did not align with target-pair output")

    threshold_rows: list[dict] = []
    threshold_labels: dict[float, np.ndarray] = {}
    for fraction in THRESHOLD_GRID:
        labels, _ = replicated_labels(primary_ranks, fraction)
        threshold_labels[fraction] = labels
        values = endpoint_metrics(labels, raw_geometry, centered_geometry)
        threshold_rows.append(
            {
                "experimental_upper_tail_fraction": fraction,
                "positive_pairs": values["positive_pairs"],
                "raw_roc_auc": values["raw_docking"]["roc_auc"],
                "centered_roc_auc": values["centered_docking"]["roc_auc"],
                "centered_minus_raw_roc_auc": values["centered_minus_raw"][
                    "roc_auc"
                ],
                "raw_average_precision": values["raw_docking"][
                    "average_precision"
                ],
                "centered_average_precision": values["centered_docking"][
                    "average_precision"
                ],
                "centered_minus_raw_average_precision": values[
                    "centered_minus_raw"
                ]["average_precision"],
            }
        )
    threshold_frame = pd.DataFrame(threshold_rows)
    omnibus = omnibus_threshold_qap(
        [threshold_labels[value] for value in OMNIBUS_THRESHOLDS],
        raw_geometry,
        centered_geometry,
        qap_permutations,
        seed + 100_000,
    )

    transform_sensitivity: dict[str, dict] = {}
    for transform_index, transform in enumerate(EXPERIMENTAL_TRANSFORMS):
        ranks = panel_rank_vectors(experimental_matrices, transform)
        labels, _ = replicated_labels(ranks, PRIMARY_FRACTION)
        transform_sensitivity[transform] = {
            "endpoint": endpoint_metrics(labels, raw_geometry, centered_geometry),
            "target_label_qap": fixed_endpoint_qap(
                labels,
                raw_geometry,
                centered_geometry,
                qap_permutations,
                seed + 200_000 + transform_index * 100_000,
            ),
            "jaccard_with_primary_labels": float(
                np.sum(labels & primary_labels) / np.sum(labels | primary_labels)
            ),
        }

    jackknife = target_delete_one(
        targets, primary_labels, raw_geometry, centered_geometry
    )

    # Leave-one-panel-out endpoints are defined from the other two panels.  Their
    # high held-out-panel AUROCs establish that the upper-tail endpoint replicates
    # across experimental technologies before docking is evaluated.
    loo_rows: list[dict] = []
    for held_out in experimental_matrices:
        training_panels = [name for name in experimental_matrices if name != held_out]
        consensus = np.mean(
            np.vstack([primary_ranks[name] for name in training_panels]), axis=0
        )
        labels = upper_tail_labels(consensus, PRIMARY_FRACTION)
        held_out_auc = retrieval_metrics(labels, primary_ranks[held_out])["roc_auc"]
        values = endpoint_metrics(labels, raw_geometry, centered_geometry)
        loo_rows.append(
            {
                "held_out_experimental_panel": held_out,
                "endpoint_training_panels": "+".join(training_panels),
                "positive_pairs": values["positive_pairs"],
                "held_out_experimental_roc_auc": held_out_auc,
                "raw_docking_roc_auc": values["raw_docking"]["roc_auc"],
                "centered_docking_roc_auc": values["centered_docking"]["roc_auc"],
                "centered_minus_raw_roc_auc": values["centered_minus_raw"][
                    "roc_auc"
                ],
                "raw_docking_average_precision": values["raw_docking"][
                    "average_precision"
                ],
                "centered_docking_average_precision": values["centered_docking"][
                    "average_precision"
                ],
                "centered_minus_raw_average_precision": values[
                    "centered_minus_raw"
                ]["average_precision"],
            }
        )
    loo_frame = pd.DataFrame(loo_rows)

    # Five independent DOCKSTRING supports test whether the result depends on a
    # single large-reference realization.
    support_rows: list[dict] = []
    for support_seed in reference_support_seeds:
        rng = np.random.default_rng(support_seed)
        indices = np.sort(
            rng.choice(len(reference), reference_support_size, replace=False)
        )
        sample = reference[indices]
        sample_raw = geometry.geometry_correlation(sample, "raw")
        sample_centered = geometry.geometry_correlation(
            sample, "center_then_correlation"
        )
        values = endpoint_metrics(primary_labels, sample_raw, sample_centered)
        support_rows.append(
            {
                "reference_support_seed": int(support_seed),
                "reference_support_size": int(reference_support_size),
                "raw_roc_auc": values["raw_docking"]["roc_auc"],
                "centered_roc_auc": values["centered_docking"]["roc_auc"],
                "centered_minus_raw_roc_auc": values["centered_minus_raw"][
                    "roc_auc"
                ],
                "raw_average_precision": values["raw_docking"][
                    "average_precision"
                ],
                "centered_average_precision": values["centered_docking"][
                    "average_precision"
                ],
                "centered_minus_raw_average_precision": values[
                    "centered_minus_raw"
                ]["average_precision"],
            }
        )
    support_frame = pd.DataFrame(support_rows)

    unclipped_raw = geometry.geometry_correlation(reference_unclipped, "raw")
    unclipped_centered = geometry.geometry_correlation(
        reference_unclipped, "center_then_correlation"
    )
    unclipped_sensitivity = endpoint_metrics(
        primary_labels, unclipped_raw, unclipped_centered
    )

    summary = {
        "analysis": "replicated experimental co-selective target-pair retrieval",
        "status": "exploratory extension; manuscript unchanged",
        "claim_boundary": (
            "The endpoint concerns co-selective target pairs on a fixed 20-kinase "
            "panel. It does not test ligand-level target retrieval, docking pose "
            "quality, or universal affinity accuracy."
        ),
        "primary_endpoint": {
            "definition": (
                "target pair is in the upper 10% of the two-way-centered target-"
                "correlation geometry in at least two of DAVIS, PKIS2, and PKIS1"
            ),
            "experimental_preprocessing": "two-way center raw endpoint values, then correlate target columns",
            "positive_pairs": int(primary_labels.sum()),
            "unanimous_positive_pairs": int(
                np.sum(np.sum(np.vstack(list(primary_panel_labels.values())), axis=0) == 3)
            ),
            "total_pairs": int(len(primary_labels)),
            "point_estimates": primary_metrics,
            "target_label_qap": primary_qap,
            "holm_adjusted_paired_gain_p_across_auc_and_ap": primary_holm,
        },
        "relation_to_rank_increase": {
            "raw_docking_correlation_pr": correlation_pr(raw_geometry),
            "centered_docking_correlation_pr": correlation_pr(centered_geometry),
            "interpretation": (
                "The PR increase and retrieval contrast occur on the same fixed "
                "20-target docking surface. Association does not establish that "
                "higher PR itself causes better retrieval."
            ),
            "full_centered_vs_leading_centered_mode": {
                "leading_mode_definition": (
                    "unrenormalized off-diagonal reconstruction lambda_1 v_1 v_1^T"
                ),
                "comparison_is_exploratory": True,
                "leading_mode_only": endpoint_metrics(
                    primary_labels, leading_mode, leading_mode
                )["raw_docking"],
                "full_centered_geometry": primary_metrics["centered_docking"],
                "full_minus_leading_mode": {
                    metric: float(
                        primary_metrics["centered_docking"][metric]
                        - endpoint_metrics(
                            primary_labels, leading_mode, leading_mode
                        )["raw_docking"][metric]
                    )
                    for metric in ("roc_auc", "average_precision")
                },
                "paired_target_label_qap": {
                    metric: {
                        "paired_p_positive_full_geometry_gain": record[
                            "paired_target_label_qap_p_positive_gain"
                        ],
                        "paired_delta_null_interval_95": record[
                            "paired_delta_null_interval_95"
                        ],
                    }
                    for metric, record in full_vs_leading["metrics"].items()
                },
                "target_delete_one_full_minus_leading_mode": {
                    "roc_auc": _numeric_range(
                        mode_jackknife.centered_minus_raw_roc_auc
                    ),
                    "average_precision": _numeric_range(
                        mode_jackknife.centered_minus_raw_average_precision
                    ),
                },
            },
        },
        "experimental_transform_sensitivity": transform_sensitivity,
        "threshold_family_omnibus": {
            "thresholds": list(OMNIBUS_THRESHOLDS),
            "interpretation": (
                "average paired gain across a declared threshold family; included "
                "to expose, rather than hide, upper-tail threshold dependence"
            ),
            **omnibus,
        },
        "target_delete_one": {
            "centered_minus_raw_roc_auc": _numeric_range(
                jackknife.centered_minus_raw_roc_auc
            ),
            "centered_minus_raw_average_precision": _numeric_range(
                jackknife.centered_minus_raw_average_precision
            ),
        },
        "leave_one_experimental_panel_out": {
            "held_out_experimental_roc_auc": _numeric_range(
                loo_frame.held_out_experimental_roc_auc
            ),
            "centered_minus_raw_roc_auc": _numeric_range(
                loo_frame.centered_minus_raw_roc_auc
            ),
            "centered_minus_raw_average_precision": _numeric_range(
                loo_frame.centered_minus_raw_average_precision
            ),
        },
        "independent_reference_supports": {
            "size": int(reference_support_size),
            "seeds": [int(value) for value in reference_support_seeds],
            "centered_minus_raw_roc_auc": _numeric_range(
                support_frame.centered_minus_raw_roc_auc
            ),
            "centered_minus_raw_average_precision": _numeric_range(
                support_frame.centered_minus_raw_average_precision
            ),
        },
        "unclipped_positive_score_sensitivity": unclipped_sensitivity,
        "support": {
            "targets": list(targets),
            "n_targets": int(len(targets)),
            "davis_ligands": int(len(experimental_matrices["DAVIS"])),
            "pkis2_ligands": int(len(experimental_matrices["PKIS2"])),
            "pkis1_ligands": int(len(experimental_matrices["PKIS1"])),
            "dockstring_reference_ligands": int(len(reference)),
            "excluded_connectivity_blocks": int(len(excluded_blocks)),
            "excluded_dockstring_rows": int((~reference_keep).sum()),
        },
        "configuration": {
            "qap_permutations": int(qap_permutations),
            "seed": int(seed),
            "primary_fraction": PRIMARY_FRACTION,
            "primary_minimum_panels": 2,
            "positive_score_handling": "clip to zero; unclipped full-support sensitivity reported",
            "reference_exclusion": "all experimental-panel Standard-InChI connectivity blocks",
        },
        "sources": {
            "dockstring": str(davis.DEFAULT_DOCKSTRING.relative_to(PACKAGE)),
            "dockstring_sha256": geometry.sha256_file(davis.DEFAULT_DOCKSTRING),
            "davis": str(davis.DEFAULT_DAVIS.relative_to(PACKAGE)),
            "davis_sha256": geometry.sha256_file(davis.DEFAULT_DAVIS),
            "pkis2": str(geometry.pkis2.DEFAULT_PKIS2.relative_to(PACKAGE)),
            "pkis2_sha256": geometry.sha256_file(geometry.pkis2.DEFAULT_PKIS2),
            "pkis1_doi": geometry.PKIS1_DOI,
            "pkis1_source_url": geometry.PKIS1_URL,
            "pkis1_local_filename": pkis1_zip.name,
            "pkis1_sha256": geometry.sha256_file(pkis1_zip),
        },
    }
    if sequence_control is not None and sequence_provenance is not None:
        summary["receptor_sequence_identity_control"] = {
            **sequence_control,
            "provenance": sequence_provenance,
        }
    return (
        summary,
        pair_frame,
        threshold_frame,
        jackknife,
        pd.concat(
            [
                support_frame.assign(analysis="reference_support"),
                loo_frame.assign(analysis="leave_one_panel_out"),
            ],
            ignore_index=True,
            sort=False,
        ),
        mode_frame,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pkis1-zip", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--qap-permutations", type=int, default=50_000)
    parser.add_argument("--reference-support-size", type=int, default=15_000)
    parser.add_argument(
        "--reference-support-seeds",
        default="11,29,47,71,97",
        help="comma-separated deterministic support seeds",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--dockstring-source-root",
        type=Path,
        help=(
            "optional DOCKSTRING source tree containing dockstring/resources/targets; "
            "enables receptor-sequence controls"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    support_seeds = tuple(
        int(value.strip())
        for value in args.reference_support_seeds.split(",")
        if value.strip()
    )
    summary, pairs, thresholds, jackknife, sensitivities, modes = run_analysis(
        pkis1_zip=args.pkis1_zip,
        qap_permutations=args.qap_permutations,
        reference_support_size=args.reference_support_size,
        reference_support_seeds=support_seeds,
        seed=args.seed,
        dockstring_source_root=args.dockstring_source_root,
    )
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    pairs.to_csv(output / "target_pairs.csv", index=False)
    thresholds.to_csv(output / "threshold_sensitivity.csv", index=False)
    jackknife.to_csv(output / "target_jackknife.csv", index=False)
    sensitivities.to_csv(output / "support_sensitivities.csv", index=False)
    modes.to_csv(output / "cumulative_mode_retrieval.csv", index=False)


if __name__ == "__main__":
    main()
