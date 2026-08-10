#!/usr/bin/env python3
"""Can chemistry-support stability qualify residual docking target edges?

This science-only audit treats the variability of each DOCKSTRING target
correlation across 50 mutually scaffold-disjoint 1,995-ligand supports as an
outcome-blind confidence feature.  A fixed, class-balanced ridge linear
probability model is fitted on PKIS1 only.  DAVIS, PKIS2, and KiRHub are locked
evaluations.  The candidate adds inverse between-support rank variability to a
strong baseline containing mean residual docking edge strength, receptor-domain
sequence identity, KLIFS pocket identity, and an outcome-blindly selected
80-compound PKIS1 target-breadth prior.

The primary target-label QAP keeps the already aligned edge-strength network
and all nondocking controls fixed and relabels only the stability network.  Raw
docking and row-disjoint random supports are explicit negative controls.

No manuscript file is read or modified.  The analysis is post hoc and is
designed to return an explicit NO-GO when stability is absorbed by edge
magnitude or fails to transfer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from collections import OrderedDict
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import scipy
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

try:
    from . import biological_core_modes as core
    from . import calibration_sample_efficiency as calibration
    from . import kirhub_external_validation as kirhub
    from .pocket_gated_prior_audit import degree_pair_features
except ImportError:  # pragma: no cover - direct CLI execution
    import biological_core_modes as core  # type: ignore
    import calibration_sample_efficiency as calibration  # type: ignore
    import kirhub_external_validation as kirhub  # type: ignore
    from pocket_gated_prior_audit import degree_pair_features  # type: ignore


PACKAGE = Path(__file__).resolve().parents[1]
DEFAULT_PAIR_PATH = PACKAGE / "results" / "klifs_pocket_control" / "target_pairs.csv"
DEFAULT_OUTPUT = PACKAGE / "results" / "edge_stability_confidence"
DEFAULT_SEED = core.DEFAULT_SEED
DEFAULT_SUPPORTS = 50
DEFAULT_SUPPORT_SIZE = 1_995
DEFAULT_PERMUTATIONS = 50_000
RIDGE_ALPHA = 1.0
LCB_MULTIPLIER = 1.645
CALIBRATION_SIZE = 80
CALIBRATION_SEED = calibration.DEFAULT_SEED
TOP_FRACTIONS = (0.05, 0.10, 0.15)
PRIMARY_FRACTION = 0.10
TARGETS = tuple(kirhub.TARGETS)
LOCKED_PANELS = ("DAVIS", "PKIS2", "KiRHub")
PANEL_ROLES = OrderedDict(
    [
        ("PKIS1", "post_hoc_discovery_model_fit_only"),
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


def upper(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("upper requires a square matrix")
    return matrix[np.triu_indices(len(matrix), k=1)]


def top_labels(values: np.ndarray, fraction: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("top_labels requires a finite vector")
    if not 0 < fraction < 1:
        raise ValueError("fraction must lie in (0, 1)")
    count = max(1, int(np.ceil(fraction * len(values))))
    labels = np.zeros(len(values), dtype=bool)
    labels[np.argsort(values, kind="mergesort")[-count:]] = True
    return labels


def zscore_columns(features: np.ndarray) -> np.ndarray:
    features = np.asarray(features, dtype=np.float64)
    if features.ndim != 2 or not np.isfinite(features).all():
        raise ValueError("features must be a finite matrix")
    scales = features.std(axis=0, ddof=0)
    if np.any(scales <= 1e-14):
        raise ValueError("features contain a constant column")
    return (features - features.mean(axis=0)) / scales


def balanced_ridge_scores(
    features: np.ndarray,
    discovery_labels: np.ndarray,
    alpha: float = RIDGE_ALPHA,
) -> np.ndarray:
    """Fixed class-balanced ridge linear-probability scores.

    The model is used only as a ranking rule.  It is intentionally simple and
    closed form so the complete discovery fit can be repeated inside every QAP
    relabelling rather than treating fitted coefficients as fixed.
    """
    labels = np.asarray(discovery_labels, dtype=bool)
    if labels.ndim != 1 or len(labels) != len(features):
        raise ValueError("labels and features are not aligned")
    positives = int(labels.sum())
    if positives == 0 or positives == len(labels):
        raise ValueError("both classes are required")
    standardized = zscore_columns(features)
    design = np.column_stack([np.ones(len(standardized)), standardized])
    prevalence = positives / len(labels)
    weights = np.where(labels, 1.0 / prevalence, 1.0 / (1.0 - prevalence))
    weights /= weights.mean()
    penalty = np.eye(design.shape[1], dtype=np.float64) * float(alpha)
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(
        design.T @ (design * weights[:, None]) + penalty,
        design.T @ (weights * labels.astype(np.float64)),
    )
    return design @ coefficients


def logistic_scores(
    features: np.ndarray, discovery_labels: np.ndarray
) -> np.ndarray:
    """Fixed C=1 class-balanced logistic sensitivity, with no tuning."""
    labels = np.asarray(discovery_labels, dtype=bool)
    standardized = StandardScaler().fit_transform(np.asarray(features, dtype=float))
    model = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        max_iter=10_000,
        solver="lbfgs",
        random_state=0,
    ).fit(standardized, labels)
    return model.predict_proba(standardized)[:, 1]


def binary_metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=np.float64)
    return {
        "roc_auc": float(roc_auc_score(labels, scores)),
        "average_precision": float(average_precision_score(labels, scores)),
    }


def fast_binary_metrics(
    labels: np.ndarray, scores: np.ndarray
) -> tuple[float, float]:
    """Tie-correct AUROC/AP used in the inner QAP loop."""
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=np.float64)
    positives = int(labels.sum())
    negatives = int(len(labels) - positives)
    ranks = stats.rankdata(scores, method="average")
    auc = (
        float(ranks[labels].sum()) - positives * (positives + 1) / 2
    ) / (positives * negatives)
    order = np.argsort(scores, kind="mergesort")[::-1]
    ordered = labels[order].astype(np.int64)
    precision = np.cumsum(ordered) / (np.arange(len(ordered)) + 1)
    ap = float(np.sum(ordered * precision) / positives)
    return float(auc), ap


def edge_indices(target_count: int) -> tuple[np.ndarray, np.ndarray]:
    return np.triu_indices(target_count, k=1)


def pair_label_map(target_count: int, order: np.ndarray) -> np.ndarray:
    tri = edge_indices(target_count)
    index = np.full((target_count, target_count), -1, dtype=np.int32)
    for pair, (first, second) in enumerate(zip(*tri)):
        index[first, second] = index[second, first] = pair
    order = np.asarray(order, dtype=int)
    if sorted(order.tolist()) != list(range(target_count)):
        raise ValueError("order must be a target permutation")
    return index[np.ix_(order, order)][tri]


def target_promiscuity_prior(
    discovery_geometry: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Continuous PKIS1 target propensity and its pairwise mean."""
    geometry = np.asarray(discovery_geometry, dtype=np.float64)
    if geometry.shape != (len(TARGETS), len(TARGETS)):
        raise ValueError("unexpected discovery geometry shape")
    target_propensity = (geometry.sum(axis=1) - np.diag(geometry)) / (
        len(geometry) - 1
    )
    tri = edge_indices(len(geometry))
    pair_prior = (target_propensity[tri[0]] + target_propensity[tri[1]]) / 2.0
    return target_propensity, pair_prior


def partial_rank_association(
    predictor: np.ndarray, endpoint: np.ndarray, controls: np.ndarray
) -> float:
    """Rank-linear partial association, reported descriptively."""
    predictor_rank = stats.rankdata(predictor, method="average")
    endpoint_rank = stats.rankdata(endpoint, method="average")
    control_ranks = np.column_stack(
        [
            np.ones(len(predictor_rank)),
            *[
                stats.rankdata(controls[:, column], method="average")
                for column in range(controls.shape[1])
            ],
        ]
    )
    predictor_residual = predictor_rank - control_ranks @ np.linalg.lstsq(
        control_ranks, predictor_rank, rcond=None
    )[0]
    endpoint_residual = endpoint_rank - control_ranks @ np.linalg.lstsq(
        control_ranks, endpoint_rank, rcond=None
    )[0]
    value = np.corrcoef(predictor_residual, endpoint_residual)[0, 1]
    if not np.isfinite(value):
        raise ValueError("partial rank association is not finite")
    return float(value)


def validate_pair_frame(frame: pd.DataFrame) -> None:
    required = {
        "target_a",
        "target_b",
        "receptor_domain_sequence_identity",
        "klifs_pocket_identity",
    }
    if not required <= set(frame.columns):
        raise ValueError("target-pair artifact is missing required columns")
    tri = edge_indices(len(TARGETS))
    expected = [(TARGETS[i], TARGETS[j]) for i, j in zip(*tri)]
    observed = list(zip(frame.target_a.astype(str), frame.target_b.astype(str)))
    if observed != expected:
        raise ValueError("target-pair rows do not match the fixed target order")


def _stable_group_hash(key: object, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}|{key}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def balanced_scaffold_support_ids(
    groups: np.ndarray,
    support_count: int,
    support_size: int,
    seed: int,
) -> np.ndarray:
    """Build maximally diverse, mutually group-disjoint ligand supports.

    Exactly one frozen-row representative is retained per chemical group. Group
    keys are then ordered by a seeded SHA256 hash and allocated round-robin.
    Thus every selected ligand contributes a distinct cyclic Murcko scaffold or
    acyclic connectivity block, and no group occurs in two supports. Rows not
    needed for the maximal balanced design retain ID -1.
    """
    values = np.asarray(groups, dtype=object)
    if values.ndim != 1 or support_count < 2 or support_size < 3:
        raise ValueError("invalid scaffold-support design")
    unique, first_indices = np.unique(values, return_index=True)
    needed = support_count * support_size
    if len(unique) < needed:
        raise ValueError(
            "one-representative-per-group design cannot fill the requested "
            f"supports: need {needed} groups but only {len(unique)} are available"
        )
    group_order = sorted(
        range(len(unique)), key=lambda index: _stable_group_hash(unique[index], seed)
    )
    selected_groups = np.asarray(group_order[:needed], dtype=int)
    selected_rows = first_indices[selected_groups]
    result = np.full(len(values), -1, dtype=np.int32)
    result[selected_rows] = np.tile(np.arange(support_count), support_size)
    observed = np.bincount(result[result >= 0], minlength=support_count)
    if not np.array_equal(observed, np.full(support_count, support_size)):
        raise RuntimeError("scaffold supports do not have the requested size")
    if len(np.unique(values[result >= 0])) != needed:
        raise RuntimeError("a chemical group occurs more than once in the design")
    return result


def random_row_support_ids(
    rows: int, support_count: int, support_size: int, seed: int
) -> np.ndarray:
    """Row-disjoint simple-random supports; scaffolds may cross supports."""
    needed = support_count * support_size
    if rows < needed:
        raise ValueError("not enough rows for random supports")
    selected = np.random.default_rng(seed).permutation(rows)[:needed]
    result = np.full(rows, -1, dtype=np.int32)
    result[selected] = np.repeat(np.arange(support_count), support_size)
    return result


def _surface_correlation(matrix: np.ndarray, surface: str) -> np.ndarray:
    if surface == "residual":
        return core.residual_correlation(matrix)
    if surface == "raw":
        correlation = np.corrcoef(np.asarray(matrix, dtype=np.float64), rowvar=False)
        correlation = np.clip((correlation + correlation.T) / 2.0, -1.0, 1.0)
        np.fill_diagonal(correlation, 1.0)
        return correlation
    raise ValueError(f"unknown docking surface: {surface}")


def support_features(
    reference: np.ndarray,
    groups: np.ndarray,
    support_ids: np.ndarray,
    support_count: int,
    *,
    support_kind: str,
    surface: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, float | int]]:
    """Summarize target edges across a fixed collection of ligand supports."""
    reference = np.asarray(reference, dtype=np.float64)
    groups = np.asarray(groups, dtype=object)
    support_ids = np.asarray(support_ids, dtype=np.int32)
    if len(reference) != len(groups) or len(reference) != len(support_ids):
        raise ValueError("reference, groups, and support IDs do not align")
    geometries = np.stack(
        [
            _surface_correlation(reference[support_ids == support], surface)
            for support in range(support_count)
        ]
    )
    tri = edge_indices(reference.shape[1])
    edge_values = geometries[:, tri[0], tri[1]]
    percentiles = np.stack(
        [
            (stats.rankdata(values, method="average") - 1.0) / (len(values) - 1.0)
            for values in edge_values
        ]
    )
    mean_edge = edge_values.mean(axis=0)
    mean_rank = percentiles.mean(axis=0)
    sd_edge = edge_values.std(axis=0, ddof=1)
    rank_sd = percentiles.std(axis=0, ddof=1)
    sign_agreement = np.mean(
        np.sign(edge_values) == np.sign(mean_edge)[None, :], axis=0
    )
    prefix = f"{support_kind}_{surface}"
    feature_rows = [
        {
            "target_a": TARGETS[first],
            "target_b": TARGETS[second],
            f"{prefix}_mean_correlation": float(mean_edge[pair]),
            f"{prefix}_mean_percentile_rank": float(mean_rank[pair]),
            f"{prefix}_between_support_sd": float(sd_edge[pair]),
            f"{prefix}_between_support_rank_sd": float(rank_sd[pair]),
            f"{prefix}_stability_confidence": float(-rank_sd[pair]),
            f"{prefix}_lcb_rank": float(
                mean_rank[pair] - LCB_MULTIPLIER * rank_sd[pair]
            ),
            f"{prefix}_sign_agreement_fraction": float(sign_agreement[pair]),
        }
        for pair, (first, second) in enumerate(zip(*tri))
    ]
    edge_rows = [
        {
            "support_kind": support_kind,
            "surface": surface,
            "support_id": support,
            "target_a": TARGETS[first],
            "target_b": TARGETS[second],
            "correlation": float(edge_values[support, pair]),
            "percentile_rank": float(percentiles[support, pair]),
        }
        for support in range(support_count)
        for pair, (first, second) in enumerate(zip(*tri))
    ]
    manifest_rows = []
    for support in range(support_count):
        mask = support_ids == support
        manifest_rows.append(
            {
                "support_kind": support_kind,
                "support_id": support,
                "rows": int(mask.sum()),
                "chemical_groups": int(len(np.unique(groups[mask]))),
            }
        )
    pairwise_rows = [
        {
            "support_kind": support_kind,
            "surface": surface,
            "support_a": first,
            "support_b": second,
            "edge_rank_spearman": float(
                stats.spearmanr(edge_values[first], edge_values[second]).statistic
            ),
        }
        for first in range(support_count)
        for second in range(first + 1, support_count)
    ]
    pairwise_values = np.asarray(
        [row["edge_rank_spearman"] for row in pairwise_rows], dtype=float
    )
    diagnostics: dict[str, float | int] = {
        "supports": int(support_count),
        "support_rows_minimum": int(
            min(row["rows"] for row in manifest_rows)
        ),
        "support_rows_maximum": int(
            max(row["rows"] for row in manifest_rows)
        ),
        "selected_rows": int(np.sum(support_ids >= 0)),
        "selected_chemical_groups": int(len(np.unique(groups[support_ids >= 0]))),
        "edge_rank_sd_median": float(np.median(rank_sd)),
        "edge_rank_sd_95th_percentile": float(np.quantile(rank_sd, 0.95)),
        "edges_with_identical_sign_in_all_supports": int(
            np.sum(sign_agreement == 1)
        ),
        "edges_total": int(len(mean_edge)),
        "support_pair_spearman_minimum": float(pairwise_values.min()),
        "support_pair_spearman_median": float(np.median(pairwise_values)),
        "support_pair_spearman_maximum": float(pairwise_values.max()),
    }
    return (
        pd.DataFrame.from_records(feature_rows),
        pd.DataFrame.from_records(edge_rows),
        pd.DataFrame.from_records(manifest_rows),
        pd.DataFrame.from_records(pairwise_rows),
        diagnostics,
    )


def calibration_breadth_features(
    pkis1_zip: Path,
) -> tuple[np.ndarray, pd.DataFrame, dict[str, object]]:
    """Frozen structure-only MaxMin-80 target-breadth controls."""
    frame = core.geometry.load_pkis1_full(Path(pkis1_zip))
    order = calibration.deterministic_maxmin_order(
        frame.SMILES.astype(str).tolist(), CALIBRATION_SEED
    )
    selected = order[:CALIBRATION_SIZE]
    values = frame[list(TARGETS)].to_numpy(dtype=np.float64)
    degree = np.mean(
        values[selected] >= calibration.PRIMARY_ACTIVITY_THRESHOLD,
        axis=0,
    )
    pair_features = degree_pair_features(degree, edge_indices(len(TARGETS)))
    target_frame = pd.DataFrame(
        {
            "target": TARGETS,
            "MaxMin_80_target_breadth": degree,
        }
    )
    metadata: dict[str, object] = {
        "compounds": CALIBRATION_SIZE,
        "selection": "Morgan-r2/2048 MaxMin; no activity used for selection",
        "selection_seed": CALIBRATION_SEED,
        "activity_threshold_percent_inhibition": (
            calibration.PRIMARY_ACTIVITY_THRESHOLD
        ),
        "selected_compound_id_sha256": hashlib.sha256(
            "\n".join(frame.iloc[selected]["Compound ID"].astype(str)).encode("utf-8")
        ).hexdigest(),
    }
    return pair_features, target_frame, metadata


def _all_metrics(
    endpoint_values: np.ndarray,
    labels: np.ndarray,
    scores: np.ndarray,
) -> dict[str, float]:
    output = binary_metrics(labels, scores)
    output["continuous_spearman"] = float(
        stats.spearmanr(scores, endpoint_values).statistic
    )
    return output


def evaluate_models(
    feature_frame: pd.DataFrame,
    pair_frame: pd.DataFrame,
    endpoints: dict[str, np.ndarray],
    breadth_features: np.ndarray,
) -> tuple[pd.DataFrame, np.ndarray]:
    base_features = np.column_stack(
        [
            feature_frame.scaffold_residual_mean_correlation,
            pair_frame.receptor_domain_sequence_identity,
            pair_frame.klifs_pocket_identity,
            breadth_features,
        ]
    )
    candidate_columns = OrderedDict(
        [
            (
                "scaffold_residual_stability_primary",
                feature_frame.scaffold_residual_stability_confidence.to_numpy(float),
            ),
            (
                "random_residual_stability_control",
                feature_frame.random_residual_stability_confidence.to_numpy(float),
            ),
            (
                "scaffold_raw_stability_control",
                feature_frame.scaffold_raw_stability_confidence.to_numpy(float),
            ),
        ]
    )
    endpoint_edges = {name: upper(matrix) for name, matrix in endpoints.items()}
    rows: list[dict[str, object]] = []
    for fraction in TOP_FRACTIONS:
        labels = {
            name: top_labels(values, fraction)
            for name, values in endpoint_edges.items()
        }
        ridge_base = balanced_ridge_scores(base_features, labels["PKIS1"])
        evaluated: list[tuple[str, str, np.ndarray, np.ndarray]] = []
        for candidate_name, values in candidate_columns.items():
            candidate_features = np.column_stack([base_features, values])
            evaluated.append(
                (
                    candidate_name,
                    "class_balanced_ridge",
                    ridge_base,
                    balanced_ridge_scores(candidate_features, labels["PKIS1"]),
                )
            )
        primary_features = np.column_stack(
            [base_features, candidate_columns["scaffold_residual_stability_primary"]]
        )
        evaluated.append(
            (
                "scaffold_residual_stability_primary",
                "class_balanced_logistic_sensitivity",
                logistic_scores(base_features, labels["PKIS1"]),
                logistic_scores(primary_features, labels["PKIS1"]),
            )
        )
        evaluated.append(
            (
                "fixed_outcome_blind_residual_LCB",
                "direct_no_fit",
                feature_frame.scaffold_residual_mean_percentile_rank.to_numpy(float),
                feature_frame.scaffold_residual_lcb_rank.to_numpy(float),
            )
        )
        for candidate_name, model, baseline, candidate in evaluated:
            for panel in PANEL_ROLES:
                base_metrics = _all_metrics(
                    endpoint_edges[panel], labels[panel], baseline
                )
                candidate_metrics = _all_metrics(
                    endpoint_edges[panel], labels[panel], candidate
                )
                for metric in ("continuous_spearman", "roc_auc", "average_precision"):
                    rows.append(
                        {
                            "top_fraction": fraction,
                            "positive_pairs": int(labels[panel].sum()),
                            "candidate_name": candidate_name,
                            "model": model,
                            "panel": panel,
                            "analysis_role": PANEL_ROLES[panel],
                            "metric": metric,
                            "baseline": base_metrics[metric],
                            "candidate_with_stability": candidate_metrics[metric],
                            "candidate_minus_baseline": (
                                candidate_metrics[metric] - base_metrics[metric]
                            ),
                        }
                    )
    return pd.DataFrame.from_records(rows), base_features


def primary_qap(
    feature_frame: pd.DataFrame,
    endpoints: dict[str, np.ndarray],
    base_features: np.ndarray,
    permutations: int,
    seed: int,
) -> pd.DataFrame:
    """Conditional target-label QAP for stability's incremental value.

    The baseline edge-strength network and all biological/chemical controls
    remain fixed. Only the candidate stability network is target relabelled,
    and the candidate is refitted on PKIS1 after every relabelling.
    """
    endpoint_edges = {name: upper(matrix) for name, matrix in endpoints.items()}
    labels = {
        name: top_labels(values, PRIMARY_FRACTION)
        for name, values in endpoint_edges.items()
    }
    confidence = feature_frame[
        "scaffold_residual_stability_confidence"
    ].to_numpy(dtype=float)
    base_features = np.asarray(base_features, dtype=np.float64)
    if base_features.shape[0] != len(confidence):
        raise ValueError("base features and stability network do not align")
    observed_base = balanced_ridge_scores(base_features, labels["PKIS1"])
    observed_candidate = balanced_ridge_scores(
        np.column_stack([base_features, confidence]), labels["PKIS1"]
    )
    base_metrics = {
        panel: _all_metrics(endpoint_edges[panel], labels[panel], observed_base)
        for panel in LOCKED_PANELS
    }
    candidate_metrics = {
        panel: _all_metrics(endpoint_edges[panel], labels[panel], observed_candidate)
        for panel in LOCKED_PANELS
    }
    metrics = ("continuous_spearman", "roc_auc", "average_precision")
    observed = {
        metric: float(
            np.mean(
                [
                    candidate_metrics[panel][metric] - base_metrics[panel][metric]
                    for panel in LOCKED_PANELS
                ]
            )
        )
        for metric in metrics
    }
    null = {metric: np.empty(permutations, dtype=np.float64) for metric in metrics}
    endpoint_ranks = {
        panel: stats.rankdata(endpoint_edges[panel], method="average")
        for panel in LOCKED_PANELS
    }
    rng = np.random.default_rng(seed)
    for repetition in range(permutations):
        mapping = pair_label_map(len(TARGETS), rng.permutation(len(TARGETS)))
        candidate = balanced_ridge_scores(
            np.column_stack([base_features, confidence[mapping]]), labels["PKIS1"]
        )
        candidate_rank = stats.rankdata(candidate, method="average")
        panel_deltas = {metric: [] for metric in metrics}
        for panel in LOCKED_PANELS:
            auc, ap = fast_binary_metrics(labels[panel], candidate)
            continuous = float(
                np.corrcoef(candidate_rank, endpoint_ranks[panel])[0, 1]
            )
            panel_deltas["continuous_spearman"].append(
                continuous - base_metrics[panel]["continuous_spearman"]
            )
            panel_deltas["roc_auc"].append(
                auc - base_metrics[panel]["roc_auc"]
            )
            panel_deltas["average_precision"].append(
                ap - base_metrics[panel]["average_precision"]
            )
        for metric in metrics:
            null[metric][repetition] = float(np.mean(panel_deltas[metric]))
    rows: list[dict[str, object]] = []
    for metric, observed_delta in observed.items():
        values = null[metric]
        p_at_least = float(
            (1 + np.sum(values >= observed_delta)) / (permutations + 1)
        )
        rows.append(
            {
                "top_fraction": PRIMARY_FRACTION,
                "metric": metric,
                "locked_panel_mean_candidate_minus_baseline": observed_delta,
                "null_mean": float(values.mean()),
                "null_median": float(np.median(values)),
                "null_interval_95_low": float(np.quantile(values, 0.025)),
                "null_interval_95_high": float(np.quantile(values, 0.975)),
                "target_label_qap_p_at_least_observed": p_at_least,
                "target_label_qap_p_positive_increment": (
                    p_at_least if observed_delta > 0 else 1.0
                ),
                "permutations": int(permutations),
                "seed": int(seed),
                "permutation_contract": (
                    "target-relabel only the scaffold-residual stability network; "
                    "keep mean residual docking edge, receptor sequence, KLIFS "
                    "pocket identity, outcome-blind MaxMin-80 target-breadth "
                    "features, and all endpoints fixed; refit the candidate on "
                    "PKIS1 after every relabelling"
                ),
            }
        )
    return pd.DataFrame.from_records(rows)


def target_jackknife(
    feature_frame: pd.DataFrame,
    endpoints: dict[str, np.ndarray],
    base_features: np.ndarray,
) -> pd.DataFrame:
    tri = edge_indices(len(TARGETS))
    confidence = feature_frame[
        "scaffold_residual_stability_confidence"
    ].to_numpy(dtype=float)
    base_features = np.asarray(base_features, dtype=np.float64)
    endpoint_edges = {name: upper(matrix) for name, matrix in endpoints.items()}
    rows: list[dict[str, object]] = []
    for dropped, target in enumerate(TARGETS):
        keep = (tri[0] != dropped) & (tri[1] != dropped)
        base = base_features[keep]
        candidate = np.column_stack([base, confidence[keep]])
        labels = {
            panel: top_labels(values[keep], PRIMARY_FRACTION)
            for panel, values in endpoint_edges.items()
        }
        baseline_score = balanced_ridge_scores(base, labels["PKIS1"])
        candidate_score = balanced_ridge_scores(candidate, labels["PKIS1"])
        for panel in LOCKED_PANELS:
            baseline_metrics = _all_metrics(
                endpoint_edges[panel][keep], labels[panel], baseline_score
            )
            candidate_metrics = _all_metrics(
                endpoint_edges[panel][keep], labels[panel], candidate_score
            )
            for metric in ("continuous_spearman", "roc_auc", "average_precision"):
                rows.append(
                    {
                        "dropped_target": target,
                        "panel": panel,
                        "retained_targets": len(TARGETS) - 1,
                        "retained_pairs": int(keep.sum()),
                        "metric": metric,
                        "baseline": baseline_metrics[metric],
                        "candidate_with_stability": candidate_metrics[metric],
                        "candidate_minus_baseline": (
                            candidate_metrics[metric] - baseline_metrics[metric]
                        ),
                    }
                )
    return pd.DataFrame.from_records(rows)


def partial_associations(
    feature_frame: pd.DataFrame,
    pair_frame: pd.DataFrame,
    endpoints: dict[str, np.ndarray],
    breadth_features: np.ndarray,
) -> pd.DataFrame:
    controls = np.column_stack(
        [
            feature_frame.scaffold_residual_mean_correlation,
            pair_frame.receptor_domain_sequence_identity,
            pair_frame.klifs_pocket_identity,
            breadth_features,
        ]
    )
    candidates = OrderedDict(
        [
            (
                "scaffold_residual_stability_primary",
                feature_frame.scaffold_residual_stability_confidence,
            ),
            (
                "random_residual_stability_control",
                feature_frame.random_residual_stability_confidence,
            ),
            (
                "scaffold_raw_stability_control",
                feature_frame.scaffold_raw_stability_confidence,
            ),
        ]
    )
    rows: list[dict[str, object]] = []
    for name, confidence in candidates.items():
        for panel, endpoint in endpoints.items():
            rows.append(
                {
                    "candidate_name": name,
                    "panel": panel,
                    "analysis_role": PANEL_ROLES[panel],
                    "partial_spearman_stability_vs_continuous_experimental_geometry": (
                        partial_rank_association(
                            confidence.to_numpy(dtype=float), upper(endpoint), controls
                        )
                    ),
                    "controls": (
                        "mean scaffold-support residual docking edge; receptor-domain "
                        "sequence identity; KLIFS pocket identity; two outcome-blind "
                        "MaxMin-80 target-breadth pair features"
                    ),
                }
            )
    return pd.DataFrame.from_records(rows)


def aligned_conservative_reference(
    reference: np.ndarray,
    groups: np.ndarray,
    expected_conservative: np.ndarray,
    pkis1_zip: Path,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """Recover group IDs for the conservative, chemistry-disjoint reference."""
    pkis2_frame = core.geometry.load_pkis2_full()
    pkis1_frame = core.geometry.load_pkis1_full(Path(pkis1_zip))
    davis_identity = core._davis_identity()  # noqa: SLF001
    experimental_scaffolds: set[str] = set()
    for series in (
        davis_identity.compound_iso_smiles,
        pkis2_frame.Smiles,
        pkis1_frame.SMILES,
    ):
        experimental_scaffolds |= {
            str(value)
            for value in core.geometry.murcko_scaffold_keys(series)
            if str(value)
        }
    excluded_groups = {f"MURCKO:{value}" for value in experimental_scaffolds}
    keep = ~np.isin(groups, np.asarray(sorted(excluded_groups), dtype=object))
    aligned_reference = np.asarray(reference)[keep]
    aligned_groups = np.asarray(groups, dtype=object)[keep]
    if aligned_reference.shape != expected_conservative.shape or not np.array_equal(
        aligned_reference, expected_conservative
    ):
        raise RuntimeError("could not align conservative reference with group IDs")
    return aligned_reference, aligned_groups, {
        "experimental_cyclic_scaffold_groups_excluded": int(
            len(np.unique(np.asarray(groups, dtype=object)[~keep]))
        ),
        "conservative_rows": int(len(aligned_reference)),
        "conservative_chemical_groups": int(len(np.unique(aligned_groups))),
    }


def write_outputs(
    output: Path, frames: dict[str, pd.DataFrame], summary: dict
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, dict[str, object]] = {}
    for stem, frame in frames.items():
        path = output / f"{stem}.csv"
        frame.to_csv(path, index=False, float_format="%.15g")
        manifest[path.name] = {"rows": int(len(frame)), "sha256": sha256_file(path)}
    summary["output_files"] = manifest
    summary_path = output / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    configuration = summary["configuration"]
    readme = f"""# Chemistry-support edge stability: {summary['decision']}

This science-only artifact tests whether a residual docking target edge that is
stable across chemistry-disjoint ligand supports is more likely to reproduce as
experimental target co-selectivity.  The prespecified design uses
{configuration['scaffold_supports']} mutually scaffold-disjoint DOCKSTRING
supports of exactly {configuration['ligands_per_support']} ligands.  It also
includes row-disjoint random-support and uncentered raw-score controls.

## Result

**{summary['decision']}.** {summary['strongest_honest_conclusion']}

The primary candidate adds inverse between-support percentile-rank variability
to a baseline containing mean residual docking edge strength, receptor-domain
sequence identity, KLIFS ATP-pocket identity, and two target-breadth pair
features estimated from an outcome-blind MaxMin selection of 80 PKIS1 compounds.
PKIS1 fits the fixed class-balanced ridge score. DAVIS, PKIS2, and KiRHub are
locked evaluations.  Conditional QAP relabels only the stability network while
holding edge strength, all controls, and endpoints fixed.

The GO rule was fixed before this run: locked-panel mean improvements of at
least +0.05 Spearman, +0.07 AUROC, and +0.04 average precision; no negative
locked-panel delta; one-sided conditional-QAP p < 0.05 for every metric; and a
positive locked mean after at least 18 of 20 target deletions for every metric.
An initial whole-group 50 x 2,000 allocator was rejected before inference when
an outcome-blind audit showed that it overselected dominant analogue series.
The released 50 x 1,995 size is the mathematical maximum for 99,755 groups with
one representative per group; the model, endpoints, statistics, and GO
thresholds were not changed in response to the result.

## Reproduce

```bash
python analysis/edge_stability_confidence.py \\
  --pkis1-zip /tmp/pkis1_supplement.zip \\
  --kirhub-workbook /private/tmp/kirhub_supp_tables.xlsx \\
  --output results/edge_stability_confidence \\
  --supports {configuration['scaffold_supports']} \\
  --support-size {configuration['ligands_per_support']} \\
  --permutations {configuration['qap_permutations']}
```

The external PKIS1 and KiRHub inputs are checksum/provenance-validated by the
reused project loaders. Compound-level experimental values and KiRHub-derived
pair values are not written.
"""
    (output / "README.md").write_text(readme, encoding="utf-8")


def run_analysis(
    *,
    pkis1_zip: Path,
    kirhub_workbook: Path,
    pair_path: Path,
    output: Path,
    supports: int,
    support_size: int,
    permutations: int,
    seed: int,
) -> dict:
    if supports < 10 or support_size < 100 or permutations < 99:
        raise ValueError(
            "release analysis requires >=10 supports, >=100 rows/support, "
            "and >=99 permutations"
        )
    reference, conservative, groups, panels, provenance = core.load_inputs(
        Path(pkis1_zip), Path(kirhub_workbook)
    )
    reference, groups, conservative_alignment = aligned_conservative_reference(
        reference, groups, conservative, Path(pkis1_zip)
    )
    endpoints = {name: core.panel_endpoint(matrix) for name, matrix in panels.items()}
    pair_frame = pd.read_csv(pair_path)
    validate_pair_frame(pair_frame)

    scaffold_ids = balanced_scaffold_support_ids(
        groups, supports, support_size, seed
    )
    random_ids = random_row_support_ids(
        len(reference), supports, support_size, seed + 1
    )
    feature_frame: pd.DataFrame | None = None
    edge_frames: list[pd.DataFrame] = []
    manifest_frames: list[pd.DataFrame] = []
    pairwise_frames: list[pd.DataFrame] = []
    stability_diagnostics: dict[str, dict[str, float | int]] = {}
    for support_kind, support_ids in (
        ("scaffold", scaffold_ids),
        ("random", random_ids),
    ):
        for surface in ("residual", "raw"):
            features, edges, manifest, pairwise, diagnostics = support_features(
                reference,
                groups,
                support_ids,
                supports,
                support_kind=support_kind,
                surface=surface,
            )
            feature_frame = (
                features
                if feature_frame is None
                else feature_frame.merge(
                    features, on=["target_a", "target_b"], validate="1:1"
                )
            )
            edge_frames.append(edges)
            manifest_frames.append(manifest)
            pairwise_frames.append(pairwise)
            stability_diagnostics[f"{support_kind}_{surface}"] = diagnostics
    if feature_frame is None or len(feature_frame) != 190:
        raise RuntimeError("target-edge feature construction failed")
    edge_frame = pd.concat(edge_frames, ignore_index=True)
    manifest_frame = pd.concat(manifest_frames, ignore_index=True).drop_duplicates(
        ["support_kind", "support_id"]
    )
    pairwise_frame = pd.concat(pairwise_frames, ignore_index=True)

    scaffold_group_support_counts = (
        pd.DataFrame(
            {"group": groups[scaffold_ids >= 0], "support": scaffold_ids[scaffold_ids >= 0]}
        )
        .drop_duplicates()
        .groupby("group")
        .size()
    )
    random_group_support_counts = (
        pd.DataFrame(
            {"group": groups[random_ids >= 0], "support": random_ids[random_ids >= 0]}
        )
        .drop_duplicates()
        .groupby("group")
        .size()
    )
    if int(np.sum(scaffold_group_support_counts > 1)) != 0:
        raise RuntimeError("a scaffold group leaked across primary supports")

    breadth_features, breadth_frame, breadth_metadata = (
        calibration_breadth_features(Path(pkis1_zip))
    )
    model_frame, base_features = evaluate_models(
        feature_frame, pair_frame, endpoints, breadth_features
    )
    qap_frame = primary_qap(
        feature_frame,
        endpoints,
        base_features,
        permutations,
        seed + 100_000,
    )
    jackknife_frame = target_jackknife(
        feature_frame, endpoints, base_features
    )
    partial_frame = partial_associations(
        feature_frame, pair_frame, endpoints, breadth_features
    )

    primary = model_frame[
        (model_frame.top_fraction == PRIMARY_FRACTION)
        & (model_frame.candidate_name == "scaffold_residual_stability_primary")
        & (model_frame.model == "class_balanced_ridge")
        & model_frame.panel.isin(LOCKED_PANELS)
    ]
    primary_metrics = ("continuous_spearman", "roc_auc", "average_precision")
    primary_deltas = {
        metric: {
            panel: float(
                primary[(primary.metric == metric) & (primary.panel == panel)]
                .candidate_minus_baseline.iloc[0]
            )
            for panel in LOCKED_PANELS
        }
        for metric in primary_metrics
    }
    locked_means = {
        metric: float(np.mean(list(primary_deltas[metric].values())))
        for metric in primary_metrics
    }
    jackknife_summary: dict[str, dict[str, float | int]] = {}
    for metric in primary_metrics:
        locked_mean = (
            jackknife_frame[jackknife_frame.metric == metric]
            .groupby("dropped_target")
            .candidate_minus_baseline.mean()
        )
        jackknife_summary[metric] = {
            "positive_target_drops": int(np.sum(locked_mean > 0)),
            "total_target_drops": int(len(locked_mean)),
            "minimum_locked_panel_mean_delta": float(locked_mean.min()),
            "median_locked_panel_mean_delta": float(locked_mean.median()),
            "maximum_locked_panel_mean_delta": float(locked_mean.max()),
        }
    qap_positive_p = {
        row.metric: float(row.target_label_qap_p_positive_increment)
        for row in qap_frame.itertuples(index=False)
    }
    go_thresholds = {
        "continuous_spearman": 0.05,
        "roc_auc": 0.07,
        "average_precision": 0.04,
    }
    go_checks = {
        "locked_mean_effect_thresholds": all(
            locked_means[metric] >= go_thresholds[metric]
            for metric in primary_metrics
        ),
        "all_locked_panels_nonnegative": all(
            value >= 0
            for metric in primary_metrics
            for value in primary_deltas[metric].values()
        ),
        "conditional_qap_p_below_0_05": all(
            qap_positive_p[metric] < 0.05 for metric in primary_metrics
        ),
        "at_least_18_of_20_positive_target_drops": all(
            jackknife_summary[metric]["positive_target_drops"] >= 18
            for metric in primary_metrics
        ),
    }
    decision = "GO" if all(go_checks.values()) else "NO-GO"
    if decision == "GO":
        conclusion = (
            "Across the prespecified chemistry-disjoint supports, inverse edge "
            "rank variability adds a transferable co-selectivity signal beyond "
            "mean docking strength and the fixed biological/chemical controls."
        )
    else:
        conclusion = (
            f"In the prespecified {supports} x {support_size:,} design, inverse "
            "between-support edge "
            "rank variability does not meet the locked-panel effect, conditional-"
            "QAP, and target-jackknife criteria for incremental co-selectivity "
            "information beyond mean docking strength and fixed controls."
        )
    control_summary: dict[str, dict[str, float]] = {}
    for candidate in (
        "random_residual_stability_control",
        "scaffold_raw_stability_control",
        "fixed_outcome_blind_residual_LCB",
    ):
        selected = model_frame[
            (model_frame.top_fraction == PRIMARY_FRACTION)
            & (model_frame.candidate_name == candidate)
            & model_frame.panel.isin(LOCKED_PANELS)
            & (
                (model_frame.model == "class_balanced_ridge")
                | (model_frame.model == "direct_no_fit")
            )
        ]
        control_summary[candidate] = {
            metric: float(
                selected[selected.metric == metric].candidate_minus_baseline.mean()
            )
            for metric in primary_metrics
        }
    try:
        pair_path_display = str(Path(pair_path).relative_to(PACKAGE))
    except ValueError:
        pair_path_display = str(Path(pair_path))
    summary = {
        "analysis": "chemistry-support edge stability as a confidence score for target-pair co-selectivity",
        "decision": decision,
        "strongest_honest_conclusion": conclusion,
        "claim_boundary": (
            "This is one fixed post-hoc test on a 20-kinase Vina panel. An initial "
            "whole-group 50 x 2,000 allocator was rejected before inference after "
            "an outcome-blind audit found dominant-analogue-series imbalance. The "
            "released 50 x 1,995 design is the maximum possible with one representative "
            "from each of 99,755 groups. No endpoint, statistic, model hyperparameter, "
            "or GO threshold changed in response to the result. A NO-GO rejects this "
            "confidence feature; it does not prove that prospective domain-shift "
            "uncertainty is never useful."
        ),
        "analysis_status": (
            "post_hoc_science_only_decisive_"
            + decision.lower().replace("-", "_")
        ),
        "configuration": {
            "targets": list(TARGETS),
            "scaffold_supports": int(supports),
            "ligands_per_support": int(support_size),
            "scaffold_support_seed": int(seed),
            "random_support_seed": int(seed + 1),
            "primary_top_fraction": PRIMARY_FRACTION,
            "sensitivity_top_fractions": list(TOP_FRACTIONS),
            "primary_stability": "negative SD of within-support edge percentile rank",
            "fixed_lcb_multiplier": LCB_MULTIPLIER,
            "primary_model": "class-balanced ridge linear probability ranking score",
            "ridge_alpha_fixed_without_tuning": RIDGE_ALPHA,
            "logistic_sensitivity_C_fixed_without_tuning": 1.0,
            "discovery_panel": "PKIS1",
            "locked_panels": list(LOCKED_PANELS),
            "qap_permutations": int(permutations),
            "qap_seed": int(seed + 100_000),
        },
        "support_integrity": {
            **conservative_alignment,
            "primary_support_definition": (
                "one frozen-row representative from each cyclic Murcko scaffold "
                "or acyclic connectivity group; group keys assigned by seeded "
                "SHA256 order"
            ),
            "primary_support_boundary": (
                "Murcko/connectivity-disjoint, not globally Butina-disjoint"
            ),
            "maximum_balanced_support_size_at_fixed_support_count": int(
                len(np.unique(groups)) // supports
            ),
            "primary_selected_rows": int(np.sum(scaffold_ids >= 0)),
            "primary_unused_rows": int(np.sum(scaffold_ids < 0)),
            "primary_selected_groups": int(
                len(np.unique(groups[scaffold_ids >= 0]))
            ),
            "primary_unused_groups": int(
                len(np.unique(groups)) - len(np.unique(groups[scaffold_ids >= 0]))
            ),
            "primary_groups_present_in_more_than_one_support": int(
                np.sum(scaffold_group_support_counts > 1)
            ),
            "random_control_groups_present_in_more_than_one_support": int(
                np.sum(random_group_support_counts > 1)
            ),
        },
        "stability_diagnostics": stability_diagnostics,
        "calibration_target_breadth": breadth_metadata,
        "prespecified_go_rule": {
            "locked_panel_mean_minimum_effects": go_thresholds,
            "every_locked_panel_delta_nonnegative": True,
            "conditional_qap_one_sided_p_maximum": 0.05,
            "minimum_positive_target_drops_out_of_20": 18,
        },
        "go_rule_checks": go_checks,
        "primary_locked_panel_deltas": primary_deltas,
        "primary_locked_panel_mean_deltas": locked_means,
        "primary_qap": qap_frame.to_dict(orient="records"),
        "target_jackknife": jackknife_summary,
        "negative_and_direct_controls_locked_mean_deltas": control_summary,
        "continuous_partial_associations": partial_frame.to_dict(orient="records"),
        "input_provenance": {
            **provenance,
            "target_pair_path": pair_path_display,
            "target_pair_sha256": sha256_file(pair_path),
        },
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "script_sha256": sha256_file(Path(__file__)),
        },
    }
    write_outputs(
        output,
        {
            "edge_stability_features": feature_frame,
            "support_edge_values": edge_frame,
            "support_manifest": manifest_frame,
            "support_pairwise_stability": pairwise_frame,
            "calibration_target_breadth": breadth_frame,
            "panel_metrics": model_frame,
            "primary_target_label_qap": qap_frame,
            "target_jackknife": jackknife_frame,
            "partial_rank_associations": partial_frame,
        },
        summary,
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pkis1-zip", type=Path, required=True)
    parser.add_argument("--kirhub-workbook", type=Path, required=True)
    parser.add_argument("--target-pairs", type=Path, default=DEFAULT_PAIR_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--supports", type=int, default=DEFAULT_SUPPORTS)
    parser.add_argument("--support-size", type=int, default=DEFAULT_SUPPORT_SIZE)
    parser.add_argument("--permutations", type=int, default=DEFAULT_PERMUTATIONS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_analysis(
        pkis1_zip=args.pkis1_zip,
        kirhub_workbook=args.kirhub_workbook,
        pair_path=args.target_pairs,
        output=args.output,
        supports=args.supports,
        support_size=args.support_size,
        permutations=args.permutations,
        seed=args.seed,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
