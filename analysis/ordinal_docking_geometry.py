#!/usr/bin/env python3
"""Test an ordinal within-ligand docking fingerprint on a locked kinase panel.

This science-only analysis compares four prespecified DOCKSTRING target-map
representations on one chemically de-leaked 20-kinase support.  The primary
question is whether discarding score magnitudes and retaining only each ligand's
within-row target ordering improves transfer to experimental target--target
co-response geometries.

The experimental endpoints are never used to fit a representation.  Target-label
QAP applies the same label permutation to the ordinal and standard-centered
predictors, and target delete-one results expose dependence on individual kinases.
No ligand-level data or identifiers are written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import OrderedDict
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
from scipy import stats

try:  # Direct CLI and package-style execution are both supported.
    from . import dense_davis_benchmark as davis
    from . import kirhub_external_validation as kirhub
    from . import replicated_pair_retrieval as retrieval
    from . import residual_target_geometry_validation as geometry
except ImportError:  # pragma: no cover - direct CLI path.
    import dense_davis_benchmark as davis  # type: ignore
    import kirhub_external_validation as kirhub  # type: ignore
    import replicated_pair_retrieval as retrieval  # type: ignore
    import residual_target_geometry_validation as geometry  # type: ignore


PACKAGE = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PACKAGE / "results" / "ordinal_docking_geometry"
DEFAULT_SEED = 20260805
TARGETS = tuple(kirhub.TARGETS)

REPRESENTATIONS: OrderedDict[str, str] = OrderedDict(
    [
        (
            "raw_column_z",
            "target-wise z-score of clipped Vina scores; no row projection",
        ),
        (
            "standard_centered",
            "target-wise z-score of clipped Vina scores, then subtract row mean",
        ),
        (
            "within_ligand_ordinal",
            "within-row Vina ranks, target-wise z-score, then subtract row mean",
        ),
        (
            "heavy_atom_efficiency",
            "Vina score divided by heavy-atom count, target-wise z-score, then subtract row mean",
        ),
    ]
)


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def row_center(matrix: np.ndarray) -> np.ndarray:
    x = np.asarray(matrix, dtype=np.float64)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError("row_center requires a finite two-dimensional matrix")
    return x - x.mean(axis=1, keepdims=True)


def within_row_ranks(matrix: np.ndarray) -> np.ndarray:
    """Average ranks within each ligand; lower Vina scores receive lower ranks.

    Reversing all ranks would multiply the final surface by -1 and therefore
    leave its target-correlation geometry unchanged.
    """
    x = np.asarray(matrix, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] < 2 or not np.isfinite(x).all():
        raise ValueError("within-row ranks require a finite two-dimensional matrix")
    return np.asarray(stats.rankdata(x, method="average", axis=1), dtype=np.float64)


def transformed_surfaces(
    scores: np.ndarray,
    heavy_atom_count: np.ndarray,
) -> OrderedDict[str, np.ndarray]:
    x = np.asarray(scores, dtype=np.float64)
    heavy = np.asarray(heavy_atom_count, dtype=np.float64)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError("scores must be a finite two-dimensional matrix")
    if heavy.shape != (len(x),) or not np.isfinite(heavy).all() or np.any(heavy <= 0):
        raise ValueError("heavy-atom counts must be finite and positive per ligand")

    raw_z = geometry.column_zscore(x)
    ordinal = within_row_ranks(x)
    ligand_efficiency = x / heavy[:, None]
    return OrderedDict(
        [
            ("raw_column_z", raw_z),
            ("standard_centered", row_center(raw_z)),
            (
                "within_ligand_ordinal",
                row_center(geometry.column_zscore(ordinal)),
            ),
            (
                "heavy_atom_efficiency",
                row_center(geometry.column_zscore(ligand_efficiency)),
            ),
        ]
    )


def representation_geometries(
    scores: np.ndarray,
    heavy_atom_count: np.ndarray,
) -> OrderedDict[str, np.ndarray]:
    return OrderedDict(
        (name, geometry.target_correlation(surface))
        for name, surface in transformed_surfaces(scores, heavy_atom_count).items()
    )


def _rank_matrix(matrix: np.ndarray) -> np.ndarray:
    x = np.asarray(matrix, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] != x.shape[1] or not np.isfinite(x).all():
        raise ValueError("rank matrix requires a finite square matrix")
    tri = np.triu_indices(len(x), k=1)
    ranks = stats.rankdata(x[tri], method="average").astype(np.float64)
    out = np.zeros_like(x)
    out[tri] = ranks
    out[(tri[1], tri[0])] = ranks
    return out


def _standardized_ranks(values: np.ndarray) -> np.ndarray:
    ranks = stats.rankdata(np.asarray(values, dtype=np.float64), method="average")
    ranks = ranks - ranks.mean()
    norm = float(np.linalg.norm(ranks))
    if norm <= 0:
        raise ValueError("rank vector is constant")
    return ranks / norm


def geometry_spearman(first: np.ndarray, second: np.ndarray) -> float:
    return geometry.geometry_concordance(first, second)


def continuous_metrics(
    docking_geometries: Mapping[str, np.ndarray],
    experimental_geometries: Mapping[str, np.ndarray],
) -> pd.DataFrame:
    rows: list[dict] = []
    for representation, predictor in docking_geometries.items():
        estimates = []
        for panel, endpoint in experimental_geometries.items():
            estimate = geometry_spearman(predictor, endpoint)
            estimates.append(estimate)
            rows.append(
                {
                    "endpoint": panel,
                    "metric": "spearman_geometry_concordance",
                    "representation": representation,
                    "estimate": estimate,
                }
            )
        rows.append(
            {
                "endpoint": "equal_weight_mean_of_four_panels",
                "metric": "mean_spearman_geometry_concordance",
                "representation": representation,
                "estimate": float(np.mean(estimates)),
            }
        )
    return pd.DataFrame(rows)


def replicated_endpoint_metrics(
    docking_geometries: Mapping[str, np.ndarray], labels: np.ndarray
) -> pd.DataFrame:
    rows: list[dict] = []
    for representation, predictor in docking_geometries.items():
        estimates = retrieval.retrieval_metrics(
            labels, geometry.upper_triangle(predictor)
        )
        for metric, estimate in estimates.items():
            rows.append(
                {
                    "endpoint": "replicated_upper_10pct_in_at_least_two_old_panels",
                    "metric": metric,
                    "representation": representation,
                    "estimate": float(estimate),
                }
            )
    return pd.DataFrame(rows)


def paired_target_label_qap(
    standard_centered: np.ndarray,
    ordinal: np.ndarray,
    experimental_geometries: Mapping[str, np.ndarray],
    replicated_labels: np.ndarray,
    permutations: int,
    seed: int,
) -> pd.DataFrame:
    """Paired target-label QAP for ordinal-minus-standard-centered gains."""
    if permutations < 1:
        raise ValueError("QAP requires at least one permutation")
    p = len(standard_centered)
    if standard_centered.shape != (p, p) or ordinal.shape != (p, p):
        raise ValueError("predictor geometries must be aligned square matrices")
    if any(np.asarray(value).shape != (p, p) for value in experimental_geometries.values()):
        raise ValueError("experimental geometries do not align with predictors")
    pair_count = p * (p - 1) // 2
    labels = np.asarray(replicated_labels, dtype=bool)
    if labels.shape != (pair_count,):
        raise ValueError("replicated labels do not align with target pairs")

    tri = np.triu_indices(p, k=1)
    standard_rank_matrix = _rank_matrix(standard_centered)
    ordinal_rank_matrix = _rank_matrix(ordinal)
    experimental_ranks = {
        name: _standardized_ranks(endpoint[tri])
        for name, endpoint in experimental_geometries.items()
    }

    def predictor_rank_vector(rank_matrix: np.ndarray, order: np.ndarray | None) -> np.ndarray:
        selected = rank_matrix if order is None else rank_matrix[np.ix_(order, order)]
        return _standardized_ranks(selected[tri])

    standard_observed_vector = predictor_rank_vector(standard_rank_matrix, None)
    ordinal_observed_vector = predictor_rank_vector(ordinal_rank_matrix, None)
    observed_rows: list[dict] = []
    observed_deltas: dict[tuple[str, str], float] = {}
    null: dict[tuple[str, str], np.ndarray] = {}

    panel_standard: dict[str, float] = {}
    panel_ordinal: dict[str, float] = {}
    for panel, endpoint_ranks in experimental_ranks.items():
        panel_standard[panel] = float(np.dot(standard_observed_vector, endpoint_ranks))
        panel_ordinal[panel] = float(np.dot(ordinal_observed_vector, endpoint_ranks))
        key = (panel, "spearman_geometry_concordance")
        observed_deltas[key] = panel_ordinal[panel] - panel_standard[panel]
        null[key] = np.empty(permutations, dtype=np.float64)

    mean_key = (
        "equal_weight_mean_of_four_panels",
        "mean_spearman_geometry_concordance",
    )
    observed_deltas[mean_key] = float(
        np.mean(list(panel_ordinal.values())) - np.mean(list(panel_standard.values()))
    )
    null[mean_key] = np.empty(permutations, dtype=np.float64)

    observed_retrieval_standard = retrieval.retrieval_metrics(
        labels, standard_centered[tri]
    )
    observed_retrieval_ordinal = retrieval.retrieval_metrics(labels, ordinal[tri])
    for metric in ("roc_auc", "average_precision"):
        key = ("replicated_upper_10pct_in_at_least_two_old_panels", metric)
        observed_deltas[key] = (
            observed_retrieval_ordinal[metric] - observed_retrieval_standard[metric]
        )
        null[key] = np.empty(permutations, dtype=np.float64)

    rng = np.random.default_rng(seed)
    for repetition in range(permutations):
        order = rng.permutation(p)
        standard_vector = predictor_rank_vector(standard_rank_matrix, order)
        ordinal_vector = predictor_rank_vector(ordinal_rank_matrix, order)
        panel_deltas = []
        for panel, endpoint_ranks in experimental_ranks.items():
            delta = float(
                np.dot(ordinal_vector, endpoint_ranks)
                - np.dot(standard_vector, endpoint_ranks)
            )
            null[(panel, "spearman_geometry_concordance")][repetition] = delta
            panel_deltas.append(delta)
        null[mean_key][repetition] = float(np.mean(panel_deltas))

        standard_scores = standard_centered[np.ix_(order, order)][tri]
        ordinal_scores = ordinal[np.ix_(order, order)][tri]
        standard_retrieval = retrieval.retrieval_metrics(labels, standard_scores)
        ordinal_retrieval = retrieval.retrieval_metrics(labels, ordinal_scores)
        for metric in ("roc_auc", "average_precision"):
            key = ("replicated_upper_10pct_in_at_least_two_old_panels", metric)
            null[key][repetition] = (
                ordinal_retrieval[metric] - standard_retrieval[metric]
            )

    for key, observed_delta in observed_deltas.items():
        endpoint, metric = key
        if endpoint in panel_standard:
            standard_estimate = panel_standard[endpoint]
            ordinal_estimate = panel_ordinal[endpoint]
        elif key == mean_key:
            standard_estimate = float(np.mean(list(panel_standard.values())))
            ordinal_estimate = float(np.mean(list(panel_ordinal.values())))
        else:
            standard_estimate = float(observed_retrieval_standard[metric])
            ordinal_estimate = float(observed_retrieval_ordinal[metric])
        values = null[key]
        p_positive = float(
            (1 + np.sum(values >= observed_delta)) / (permutations + 1)
        )
        p_negative = float(
            (1 + np.sum(values <= observed_delta)) / (permutations + 1)
        )
        observed_rows.append(
            {
                "endpoint": endpoint,
                "metric": metric,
                "standard_centered": standard_estimate,
                "within_ligand_ordinal": ordinal_estimate,
                "ordinal_minus_standard_centered": observed_delta,
                "permutations": int(permutations),
                "one_sided_p_positive_gain": p_positive,
                "one_sided_p_negative_gain": p_negative,
                "two_sided_p_difference": min(1.0, 2.0 * min(p_positive, p_negative)),
                "null_q025": float(np.quantile(values, 0.025)),
                "null_median": float(np.median(values)),
                "null_q975": float(np.quantile(values, 0.975)),
                "seed": int(seed),
            }
        )
    return pd.DataFrame(observed_rows)


def target_delete_one(
    targets: tuple[str, ...],
    docking_geometries: Mapping[str, np.ndarray],
    experimental_geometries: Mapping[str, np.ndarray],
    replicated_labels: np.ndarray,
) -> pd.DataFrame:
    p = len(targets)
    tri = np.triu_indices(p, k=1)
    label_matrix = np.zeros((p, p), dtype=bool)
    label_matrix[tri] = np.asarray(replicated_labels, dtype=bool)
    label_matrix[(tri[1], tri[0])] = label_matrix[tri]
    rows: list[dict] = []
    for deleted, target in enumerate(targets):
        keep = np.arange(p) != deleted
        sub_tri = np.triu_indices(p - 1, k=1)
        panel_estimates: dict[str, dict[str, float]] = {}
        for panel, endpoint in experimental_geometries.items():
            endpoint_sub = endpoint[np.ix_(keep, keep)]
            estimates = {
                representation: geometry_spearman(
                    predictor[np.ix_(keep, keep)], endpoint_sub
                )
                for representation, predictor in docking_geometries.items()
            }
            panel_estimates[panel] = estimates
            rows.append(
                {
                    "omitted_target": target,
                    "endpoint": panel,
                    "metric": "spearman_geometry_concordance",
                    **estimates,
                    "ordinal_minus_standard_centered": (
                        estimates["within_ligand_ordinal"]
                        - estimates["standard_centered"]
                    ),
                }
            )
        mean_estimates = {
            representation: float(
                np.mean(
                    [values[representation] for values in panel_estimates.values()]
                )
            )
            for representation in docking_geometries
        }
        rows.append(
            {
                "omitted_target": target,
                "endpoint": "equal_weight_mean_of_four_panels",
                "metric": "mean_spearman_geometry_concordance",
                **mean_estimates,
                "ordinal_minus_standard_centered": (
                    mean_estimates["within_ligand_ordinal"]
                    - mean_estimates["standard_centered"]
                ),
            }
        )

        labels_sub = label_matrix[np.ix_(keep, keep)][sub_tri]
        for metric in ("roc_auc", "average_precision"):
            estimates = {
                representation: retrieval.retrieval_metrics(
                    labels_sub, predictor[np.ix_(keep, keep)][sub_tri]
                )[metric]
                for representation, predictor in docking_geometries.items()
            }
            rows.append(
                {
                    "omitted_target": target,
                    "endpoint": "replicated_upper_10pct_in_at_least_two_old_panels",
                    "metric": metric,
                    **estimates,
                    "ordinal_minus_standard_centered": (
                        estimates["within_ligand_ordinal"]
                        - estimates["standard_centered"]
                    ),
                }
            )
    return pd.DataFrame(rows)


def _range(values: pd.Series) -> dict[str, float | int]:
    x = values.to_numpy(dtype=np.float64)
    return {
        "minimum": float(x.min()),
        "median": float(np.median(x)),
        "maximum": float(x.max()),
        "positive_deletions": int(np.sum(x > 0)),
        "total_deletions": int(len(x)),
    }


def classify_result(
    metric_frame: pd.DataFrame,
    qap_frame: pd.DataFrame,
    jackknife_frame: pd.DataFrame,
) -> tuple[str, dict]:
    continuous = metric_frame[
        metric_frame.endpoint.isin(("DAVIS", "PKIS2", "PKIS1", "KiRHub"))
        & metric_frame.metric.eq("spearman_geometry_concordance")
    ].pivot(index="endpoint", columns="representation", values="estimate")
    panel_deltas = (
        continuous["within_ligand_ordinal"] - continuous["standard_centered"]
    )
    omnibus = qap_frame[
        qap_frame.endpoint.eq("equal_weight_mean_of_four_panels")
        & qap_frame.metric.eq("mean_spearman_geometry_concordance")
    ].iloc[0]
    replicated = qap_frame[
        qap_frame.endpoint.eq(
            "replicated_upper_10pct_in_at_least_two_old_panels"
        )
        & qap_frame.metric.eq("roc_auc")
    ].iloc[0]
    jackknife_mean = jackknife_frame[
        jackknife_frame.endpoint.eq("equal_weight_mean_of_four_panels")
    ].ordinal_minus_standard_centered

    criteria = {
        "positive_in_at_least_three_of_four_continuous_panels": bool(
            np.sum(panel_deltas > 0) >= 3
        ),
        "equal_weight_four_panel_qap_p_below_0_05": bool(
            omnibus.one_sided_p_positive_gain < 0.05
        ),
        "all_target_deletions_positive_for_four_panel_mean": bool(
            np.all(jackknife_mean > 0)
        ),
        "replicated_endpoint_auc_gain_positive": bool(
            replicated.ordinal_minus_standard_centered > 0
        ),
        "replicated_endpoint_auc_qap_p_below_0_05": bool(
            replicated.one_sided_p_positive_gain < 0.05
        ),
    }
    if all(criteria.values()):
        verdict = "GO: ordinal geometry is a robust stronger cross-panel representation"
    elif (
        criteria["positive_in_at_least_three_of_four_continuous_panels"]
        and criteria["equal_weight_four_panel_qap_p_below_0_05"]
    ):
        verdict = (
            "ENDPOINT-SPECIFIC: ordinal geometry improves continuous cross-panel "
            "alignment but fails at least one robustness or replicated-endpoint gate"
        )
    else:
        verdict = (
            "NO-GO: ordinal geometry is not a reproducibly stronger cross-panel "
            "representation than standard centering"
        )
    return verdict, {
        "criteria": criteria,
        "continuous_panel_deltas": {
            str(name): float(value) for name, value in panel_deltas.items()
        },
    }


def load_analysis_inputs(
    pkis1_zip: Path, kirhub_workbook: Path
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], np.ndarray, dict]:
    """Load one de-leaked docking support and four fixed experimental panels."""
    dockstring, davis_identity, davis_experiment, _ = davis._load_inputs(  # noqa: SLF001
        davis.DEFAULT_DOCKSTRING,
        davis.DEFAULT_DAVIS,
        identity_scan="full",
    )
    pkis2_frame = geometry.load_pkis2_full()
    pkis1_frame = geometry.load_pkis1_full(pkis1_zip)
    excluded_blocks = set(davis_identity.connectivity_block.dropna())
    excluded_blocks |= set(pkis2_frame.connectivity_block.dropna())
    excluded_blocks |= set(pkis1_frame.connectivity_block.dropna())
    keep = ~dockstring.connectivity_block.isin(excluded_blocks)
    scores_unclipped = dockstring.loc[keep, list(TARGETS)].to_numpy(dtype=np.float64)
    scores = np.minimum(scores_unclipped, 0.0)
    heavy = dockstring.loc[keep, "heavy_atom_count"].to_numpy(dtype=np.float64)
    if len(scores) < 259_000:
        raise ValueError("unexpectedly many DOCKSTRING rows were excluded")

    experimental_matrices = {
        "DAVIS": davis_experiment[list(TARGETS)].to_numpy(dtype=np.float64),
        "PKIS2": pkis2_frame[list(TARGETS)].to_numpy(dtype=np.float64),
        "PKIS1": pkis1_frame[list(TARGETS)].to_numpy(dtype=np.float64),
        "KiRHub": kirhub.load_kirhub(
            kirhub_workbook, cdk2_construct="cyclin_A"
        )[list(TARGETS)].to_numpy(dtype=np.float64),
    }
    old_panel_ranks = retrieval.panel_rank_vectors(
        {name: experimental_matrices[name] for name in ("DAVIS", "PKIS2", "PKIS1")},
        "center_then_correlation",
    )
    replicated_labels, _ = retrieval.replicated_labels(
        old_panel_ranks, retrieval.PRIMARY_FRACTION, minimum_panels=2
    )
    provenance = {
        "complete_dockstring_rows_before_exclusion": int(len(dockstring)),
        "reference_rows_after_old_panel_connectivity_exclusion": int(len(scores)),
        "excluded_connectivity_blocks": int(len(excluded_blocks)),
        "excluded_dockstring_rows": int((~keep).sum()),
        "positive_score_cells_clipped_to_zero": int(np.sum(scores_unclipped > 0)),
        "chemical_exclusion": (
            "all Standard-InChI connectivity blocks in DAVIS, PKIS2, and PKIS1"
        ),
        "kirhub_identity_boundary": (
            "KiRHub Table S4 exposes compound names but no structures; residual "
            "KiRHub chemical overlap with DOCKSTRING cannot be excluded by structure"
        ),
    }
    return scores, heavy, experimental_matrices, replicated_labels, provenance


def run_analysis(
    *,
    pkis1_zip: Path,
    kirhub_workbook: Path,
    qap_permutations: int,
    seed: int,
) -> tuple[dict, dict[str, pd.DataFrame]]:
    scores, heavy, experimental_matrices, labels, provenance = load_analysis_inputs(
        pkis1_zip, kirhub_workbook
    )
    docking_geometries = representation_geometries(scores, heavy)
    experimental_geometries = {
        name: geometry.geometry_correlation(matrix, "center_then_correlation")
        for name, matrix in experimental_matrices.items()
    }
    metrics = pd.concat(
        [
            continuous_metrics(docking_geometries, experimental_geometries),
            replicated_endpoint_metrics(docking_geometries, labels),
        ],
        ignore_index=True,
    )
    qap = paired_target_label_qap(
        docking_geometries["standard_centered"],
        docking_geometries["within_ligand_ordinal"],
        experimental_geometries,
        labels,
        qap_permutations,
        seed,
    )
    jackknife = target_delete_one(
        TARGETS, docking_geometries, experimental_geometries, labels
    )

    tri = np.triu_indices(len(TARGETS), k=1)
    pair_rows: list[dict] = []
    label_matrix = np.zeros((len(TARGETS), len(TARGETS)), dtype=bool)
    label_matrix[tri] = labels
    for first, second in zip(*tri):
        row = {
            "target_a": TARGETS[first],
            "target_b": TARGETS[second],
            "replicated_endpoint_positive": bool(label_matrix[first, second]),
        }
        for name, predictor in docking_geometries.items():
            row[f"docking_{name}"] = float(predictor[first, second])
        for name, endpoint in experimental_geometries.items():
            row[f"experimental_{name}_centered"] = float(endpoint[first, second])
        pair_rows.append(row)
    pair_frame = pd.DataFrame(pair_rows)

    verdict, verdict_details = classify_result(metrics, qap, jackknife)
    jackknife_summary = {
        f"{row.endpoint}__{row.metric}": _range(
            jackknife[
                jackknife.endpoint.eq(row.endpoint)
                & jackknife.metric.eq(row.metric)
            ].ordinal_minus_standard_centered
        )
        for row in jackknife[["endpoint", "metric"]].drop_duplicates().itertuples()
    }
    summary = {
        "analysis": "ordinal within-ligand docking target-map representation",
        "status": "science-only exploratory test; manuscript unchanged",
        "verdict": verdict,
        "verdict_details": verdict_details,
        "prespecified_representations": REPRESENTATIONS,
        "primary_question": (
            "Does within-ligand target order, stripped of score magnitude, improve "
            "transfer over the standard column-z-plus-row-centered Vina geometry?"
        ),
        "claim_boundary": (
            "All results concern target-pair co-response geometry on a fixed "
            "20-kinase panel. They do not establish ligand-level target ranking, "
            "unseen-target retrieval, affinity calibration, or docking pose quality."
        ),
        "continuous_endpoint": (
            "Spearman concordance with each experimental target-correlation geometry "
            "after two-way centering the experimental surface"
        ),
        "replicated_endpoint": (
            "target pair in the upper 10% of centered experimental geometry in at "
            "least two of DAVIS, PKIS2, and PKIS1"
        ),
        "paired_inference": (
            "same target-label permutation applied to ordinal and standard-centered "
            "predictors; experimental endpoints remain fixed"
        ),
        "target_delete_one_ordinal_minus_standard_centered": jackknife_summary,
        "support": {
            **provenance,
            "targets": list(TARGETS),
            "n_targets": len(TARGETS),
            "target_pairs": int(len(TARGETS) * (len(TARGETS) - 1) // 2),
            "replicated_positive_pairs": int(labels.sum()),
            "experimental_ligands": {
                name: int(len(matrix)) for name, matrix in experimental_matrices.items()
            },
            "heavy_atom_count_range": [float(heavy.min()), float(heavy.max())],
        },
        "configuration": {
            "qap_permutations": int(qap_permutations),
            "seed": int(seed),
            "positive_score_handling": "clip to zero before all four representations",
            "ordinal_ties": "average within-row ranks",
            "continuous_panel_aggregation": "unweighted mean of four panel-specific Spearman correlations",
        },
        "spd_boundary": (
            "The frozen SPD panel has only one target (EGFR) in common with this "
            "20-kinase panel, so it cannot test the same target-pair geometry and "
            "was not duplicated here."
        ),
        "sources": {
            "dockstring_sha256": geometry.sha256_file(davis.DEFAULT_DOCKSTRING),
            "davis_sha256": geometry.sha256_file(davis.DEFAULT_DAVIS),
            "pkis2_sha256": geometry.sha256_file(geometry.pkis2.DEFAULT_PKIS2),
            "pkis1_sha256": geometry.sha256_file(pkis1_zip),
            "kirhub_sha256": kirhub.sha256_file(kirhub_workbook),
        },
    }
    return summary, {
        "representation_metrics.csv": metrics,
        "paired_target_label_qap.csv": qap,
        "target_jackknife.csv": jackknife,
        "target_pairs.csv": pair_frame,
    }


def write_outputs(
    output_dir: Path, summary: dict, frames: Mapping[str, pd.DataFrame]
) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    summary_path = output / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    for filename, frame in frames.items():
        frame.to_csv(output / filename, index=False)
    checksums = {
        path.name: sha256_file(path)
        for path in sorted(output.iterdir())
        if path.is_file() and path.name not in {"README.md", "output_checksums.json"}
    }
    (output / "output_checksums.json").write_text(
        json.dumps(checksums, indent=2) + "\n", encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pkis1-zip", type=Path, required=True)
    parser.add_argument("--kirhub-workbook", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--qap-permutations", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary, frames = run_analysis(
        pkis1_zip=args.pkis1_zip,
        kirhub_workbook=args.kirhub_workbook,
        qap_permutations=args.qap_permutations,
        seed=args.seed,
    )
    write_outputs(args.output_dir, summary, frames)
    print(json.dumps({"verdict": summary["verdict"]}, indent=2))


if __name__ == "__main__":
    main()
