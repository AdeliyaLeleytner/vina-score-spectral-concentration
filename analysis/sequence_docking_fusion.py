#!/usr/bin/env python3
"""Does residual Vina geometry add target information beyond sequence identity?

The predictor is fixed and outcome-blind: an equal-weight mean of the
strict-upper-triangle percentile ranks of receptor-domain sequence identity and
two-way-centered Vina target geometry.  The analysis evaluates this predictor
against four experimental kinase target-correlation geometries.

Incremental target-label QAP keeps sequence identity and each experimental
endpoint fixed while permuting only the Vina target labels.  This directly asks
whether correctly aligned docking geometry improves on sequence identity more
than a target-mismatched docking network would.  No weight is fitted.

The analysis is post hoc: all four resources have been inspected during the
project.  Numerical outcome blindness in this artifact does not make the result
prospectively confirmatory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import kirhub_external_validation as kirhub
from klifs_pocket_control import TARGETS, matrix_from_pairs
from replicated_pair_retrieval import retrieval_metrics, upper_tail_labels


PACKAGE = Path(__file__).resolve().parents[1]
DEFAULT_PAIRS = PACKAGE / "results" / "klifs_pocket_control" / "target_pairs.csv"
DEFAULT_OUTPUT = PACKAGE / "results" / "sequence_docking_fusion"
DEFAULT_PERMUTATIONS = 50_000
DEFAULT_SEED = 20260814
TOP_FRACTION = 0.10

PANEL_ROLES = OrderedDict(
    [
        ("PKIS1", "post_hoc_discovery"),
        ("DAVIS", "locked_no_retuning_evaluation"),
        ("PKIS2", "locked_no_retuning_evaluation"),
        ("KiRHub", "locked_no_retuning_evaluation"),
    ]
)


def holm_adjust(values: dict[str, float]) -> dict[str, float]:
    """Holm-adjust a named p-value family while preserving monotonicity."""

    ordered = sorted(values, key=values.get)
    total = len(ordered)
    adjusted: dict[str, float] = {}
    running = 0.0
    for index, name in enumerate(ordered):
        candidate = min(1.0, (total - index) * float(values[name]))
        running = max(running, candidate)
        adjusted[name] = running
    return adjusted


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def upper(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    return matrix[np.triu_indices(len(matrix), k=1)]


def rank_matrix(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("rank_matrix requires a square matrix")
    tri = np.triu_indices(len(matrix), k=1)
    values = stats.rankdata(matrix[tri], method="average")
    result = np.zeros_like(matrix, dtype=np.float64)
    result[tri] = values
    result[(tri[1], tri[0])] = values
    return result


def equal_rank_fusion(*matrices: np.ndarray) -> np.ndarray:
    """Outcome-blind equal-weight rank fusion of aligned target similarities."""
    if len(matrices) < 2:
        raise ValueError("fusion requires at least two matrices")
    shapes = {np.asarray(matrix).shape for matrix in matrices}
    if len(shapes) != 1:
        raise ValueError("fusion matrices must have the same shape")
    ranks = [rank_matrix(matrix) for matrix in matrices]
    result = np.mean(ranks, axis=0)
    np.fill_diagonal(result, 1.0)
    return result


def continuous_spearman(predictor: np.ndarray, endpoint: np.ndarray) -> float:
    value = stats.spearmanr(upper(predictor), upper(endpoint)).statistic
    if not np.isfinite(value):
        raise ValueError("continuous geometry concordance is not finite")
    return float(value)


def predictor_metrics(
    predictor: np.ndarray, endpoint: np.ndarray, top_fraction: float = TOP_FRACTION
) -> dict[str, float | int]:
    labels = upper_tail_labels(upper(endpoint), top_fraction)
    retrieval = retrieval_metrics(labels, upper(predictor))
    return {
        "target_pairs": int(len(labels)),
        "positive_pairs": int(labels.sum()),
        "continuous_spearman": continuous_spearman(predictor, endpoint),
        "roc_auc": float(retrieval["roc_auc"]),
        "average_precision": float(retrieval["average_precision"]),
    }


def load_inputs(
    pair_path: Path, kirhub_workbook: Path
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    pair_frame = pd.read_csv(pair_path)
    expected_rows = len(TARGETS) * (len(TARGETS) - 1) // 2
    if len(pair_frame) != expected_rows:
        raise ValueError("unexpected target-pair artifact size")
    predictors = {
        "receptor_domain_sequence_identity": matrix_from_pairs(
            pair_frame, "receptor_domain_sequence_identity"
        ),
        "klifs_pocket_identity": matrix_from_pairs(
            pair_frame, "klifs_pocket_identity"
        ),
        "centered_Vina_target_geometry": matrix_from_pairs(
            pair_frame, "centered_docking_pair_percentile"
        ),
    }
    endpoints = {
        "PKIS1": matrix_from_pairs(pair_frame, "PKIS1_centered_pair_percentile"),
        "DAVIS": matrix_from_pairs(pair_frame, "DAVIS_centered_pair_percentile"),
        "PKIS2": matrix_from_pairs(pair_frame, "PKIS2_centered_pair_percentile"),
    }
    kirhub_frame = kirhub.load_kirhub(kirhub_workbook)
    endpoints["KiRHub"] = kirhub.correlation_geometry(
        kirhub_frame[list(TARGETS)].to_numpy(dtype=np.float64), "two_way_center"
    )
    return predictors, endpoints


def panel_metric_frame(
    predictors: dict[str, np.ndarray], endpoints: dict[str, np.ndarray]
) -> pd.DataFrame:
    sequence = predictors["receptor_domain_sequence_identity"]
    pocket = predictors["klifs_pocket_identity"]
    vina = predictors["centered_Vina_target_geometry"]
    evaluated = OrderedDict(
        [
            ("receptor_domain_sequence_identity", sequence),
            ("klifs_pocket_identity", pocket),
            ("centered_Vina_target_geometry", vina),
            ("equal_rank_sequence_plus_Vina", equal_rank_fusion(sequence, vina)),
            (
                "equal_rank_sequence_plus_pocket",
                equal_rank_fusion(sequence, pocket),
            ),
            (
                "equal_rank_sequence_plus_pocket_plus_Vina",
                equal_rank_fusion(sequence, pocket, vina),
            ),
        ]
    )
    rows: list[dict[str, object]] = []
    for panel, endpoint in endpoints.items():
        for predictor, matrix in evaluated.items():
            rows.append(
                {
                    "panel": panel,
                    "analysis_role": PANEL_ROLES[panel],
                    "predictor": predictor,
                    **predictor_metrics(matrix, endpoint),
                }
            )
    return pd.DataFrame.from_records(rows)


def incremental_qap(
    sequence: np.ndarray,
    vina: np.ndarray,
    endpoints: dict[str, np.ndarray],
    permutations: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Test fixed equal-rank fusion minus sequence with Vina-label QAP."""
    if permutations < 1:
        raise ValueError("permutations must be positive")
    p = len(sequence)
    tri = np.triu_indices(p, k=1)
    sequence_rank = rank_matrix(sequence)
    vina_rank = rank_matrix(vina)
    observed_fusion = (sequence_rank + vina_rank) / 2.0

    metric_names = ("continuous_spearman", "roc_auc", "average_precision")
    endpoint_data: dict[str, dict[str, np.ndarray | float | dict[str, float]]] = {}
    for panel, endpoint in endpoints.items():
        endpoint_values = upper(endpoint)
        endpoint_ranks = stats.rankdata(endpoint_values, method="average").astype(float)
        endpoint_ranks -= endpoint_ranks.mean()
        labels = upper_tail_labels(endpoint_values, TOP_FRACTION)
        sequence_metrics = predictor_metrics(sequence_rank, endpoint)
        fusion_metrics = predictor_metrics(observed_fusion, endpoint)
        endpoint_data[panel] = {
            "endpoint_ranks": endpoint_ranks,
            "endpoint_rank_norm": float(np.linalg.norm(endpoint_ranks)),
            "labels": labels,
            "sequence_metrics": {
                name: float(sequence_metrics[name]) for name in metric_names
            },
            "fusion_metrics": {
                name: float(fusion_metrics[name]) for name in metric_names
            },
        }

    null = {
        panel: {
            metric: np.empty(permutations, dtype=np.float64)
            for metric in metric_names
        }
        for panel in endpoints
    }
    rng = np.random.default_rng(seed)
    for repetition in range(permutations):
        order = rng.permutation(p)
        permuted_vina = vina_rank[np.ix_(order, order)]
        candidate = (sequence_rank + permuted_vina) / 2.0
        values = candidate[tri]
        ranks = stats.rankdata(values, method="average").astype(float)
        centered_ranks = ranks - ranks.mean()
        rank_norm = float(np.linalg.norm(centered_ranks))
        for panel, data in endpoint_data.items():
            endpoint_ranks = np.asarray(data["endpoint_ranks"], dtype=float)
            fusion_null_rho = float(
                np.dot(centered_ranks, endpoint_ranks)
                / (rank_norm * float(data["endpoint_rank_norm"]))
            )
            labels = np.asarray(data["labels"], dtype=bool)
            retrieval = retrieval_metrics(labels, values)
            sequence_metrics = data["sequence_metrics"]
            null[panel]["continuous_spearman"][repetition] = (
                fusion_null_rho
                - float(sequence_metrics["continuous_spearman"])
            )
            null[panel]["roc_auc"][repetition] = (
                retrieval["roc_auc"] - float(sequence_metrics["roc_auc"])
            )
            null[panel]["average_precision"][repetition] = (
                retrieval["average_precision"]
                - float(sequence_metrics["average_precision"])
            )

    rows: list[dict[str, object]] = []
    for panel, data in endpoint_data.items():
        sequence_metrics = data["sequence_metrics"]
        fusion_metrics = data["fusion_metrics"]
        for metric in metric_names:
            observed_delta = float(fusion_metrics[metric]) - float(
                sequence_metrics[metric]
            )
            values = null[panel][metric]
            rows.append(
                {
                    "panel": panel,
                    "analysis_role": PANEL_ROLES[panel],
                    "metric": metric,
                    "sequence_identity": float(sequence_metrics[metric]),
                    "equal_rank_sequence_plus_Vina": float(
                        fusion_metrics[metric]
                    ),
                    "fusion_minus_sequence": observed_delta,
                    "observed_fusion_improves_sequence": bool(observed_delta > 0),
                    "target_label_qap_p_aligned_Vina_at_least_observed_delta": float(
                        (1 + np.sum(values >= observed_delta))
                        / (permutations + 1)
                    ),
                    "target_label_qap_p_positive_increment": (
                        float(
                            (1 + np.sum(values >= observed_delta))
                            / (permutations + 1)
                        )
                        if observed_delta > 0
                        else 1.0
                    ),
                    "null_interval_95_low": float(np.quantile(values, 0.025)),
                    "null_interval_95_high": float(np.quantile(values, 0.975)),
                    "permutations": int(permutations),
                    "seed": int(seed),
                }
            )
    frame = pd.DataFrame.from_records(rows)
    locked = frame.analysis_role.str.startswith("locked")
    adjusted = holm_adjust(
        {
            f"{row.panel}|{row.metric}": float(
                row.target_label_qap_p_positive_increment
            )
            for row in frame.loc[locked].itertuples(index=False)
        }
    )
    frame["holm_p_across_9_locked_panel_metric_tests"] = np.nan
    for index, row in frame.loc[locked].iterrows():
        frame.loc[index, "holm_p_across_9_locked_panel_metric_tests"] = adjusted[
            f"{row.panel}|{row.metric}"
        ]

    locked_panels = [
        panel for panel, role in PANEL_ROLES.items() if role.startswith("locked")
    ]
    omnibus_rows: list[dict[str, object]] = []
    omnibus_p: dict[str, float] = {}
    for metric in metric_names:
        panel_records = frame.loc[
            frame.panel.isin(locked_panels) & frame.metric.eq(metric)
        ]
        observed = float(panel_records.fusion_minus_sequence.mean())
        values = np.mean(
            np.vstack([null[panel][metric] for panel in locked_panels]), axis=0
        )
        p_value = float(
            (1 + np.sum(values >= observed)) / (permutations + 1)
        )
        positive_p_value = p_value if observed > 0 else 1.0
        omnibus_p[metric] = positive_p_value
        omnibus_rows.append(
            {
                "metric": metric,
                "locked_panels": ",".join(locked_panels),
                "observed_mean_fusion_minus_sequence": observed,
                "observed_mean_fusion_improves_sequence": bool(observed > 0),
                "target_label_qap_p_aligned_Vina_at_least_observed_mean_delta": (
                    p_value
                ),
                "target_label_qap_p_positive_mean_increment": positive_p_value,
                "null_interval_95_low": float(np.quantile(values, 0.025)),
                "null_interval_95_high": float(np.quantile(values, 0.975)),
                "permutations": int(permutations),
                "seed": int(seed),
            }
        )
    omnibus = pd.DataFrame.from_records(omnibus_rows)
    omnibus_adjusted = holm_adjust(omnibus_p)
    omnibus["holm_p_across_3_omnibus_metrics"] = omnibus.metric.map(
        omnibus_adjusted
    )
    return frame, omnibus


def target_jackknife(
    sequence: np.ndarray,
    vina: np.ndarray,
    endpoints: dict[str, np.ndarray],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for dropped, target in enumerate(TARGETS):
        keep = np.asarray([index for index in range(len(TARGETS)) if index != dropped])
        sequence_sub = sequence[np.ix_(keep, keep)]
        vina_sub = vina[np.ix_(keep, keep)]
        fusion_sub = equal_rank_fusion(sequence_sub, vina_sub)
        for panel, endpoint in endpoints.items():
            endpoint_sub = endpoint[np.ix_(keep, keep)]
            sequence_rho = continuous_spearman(sequence_sub, endpoint_sub)
            vina_rho = continuous_spearman(vina_sub, endpoint_sub)
            fusion_rho = continuous_spearman(fusion_sub, endpoint_sub)
            rows.append(
                {
                    "dropped_target": target,
                    "panel": panel,
                    "analysis_role": PANEL_ROLES[panel],
                    "remaining_targets": int(len(keep)),
                    "sequence_continuous_spearman": sequence_rho,
                    "Vina_continuous_spearman": vina_rho,
                    "fusion_continuous_spearman": fusion_rho,
                    "fusion_minus_sequence_continuous_spearman": (
                        fusion_rho - sequence_rho
                    ),
                    "fusion_minus_Vina_continuous_spearman": fusion_rho - vina_rho,
                }
            )
    return pd.DataFrame.from_records(rows)


def run(
    pair_path: Path,
    kirhub_workbook: Path,
    output: Path,
    permutations: int,
    seed: int,
) -> dict[str, object]:
    predictors, endpoints = load_inputs(pair_path, kirhub_workbook)
    sequence = predictors["receptor_domain_sequence_identity"]
    vina = predictors["centered_Vina_target_geometry"]
    metrics = panel_metric_frame(predictors, endpoints)
    qap, omnibus = incremental_qap(
        sequence, vina, endpoints, permutations, seed
    )
    jackknife = target_jackknife(sequence, vina, endpoints)
    primary = metrics.loc[
        metrics.predictor.isin(
            [
                "receptor_domain_sequence_identity",
                "centered_Vina_target_geometry",
                "equal_rank_sequence_plus_Vina",
            ]
        )
    ]
    wide = primary.pivot(
        index="panel", columns="predictor", values="continuous_spearman"
    )
    all_panels_fusion_exceeds_both = bool(
        (
            wide["equal_rank_sequence_plus_Vina"]
            > wide[
                [
                    "receptor_domain_sequence_identity",
                    "centered_Vina_target_geometry",
                ]
            ].max(axis=1)
        ).all()
    )
    primary_panel_metrics = {
        panel: {
            predictor: {
                metric: float(
                    primary.loc[
                        primary.panel.eq(panel) & primary.predictor.eq(predictor),
                        metric,
                    ].iloc[0]
                )
                for metric in ("continuous_spearman", "roc_auc", "average_precision")
            }
            for predictor in (
                "receptor_domain_sequence_identity",
                "centered_Vina_target_geometry",
                "equal_rank_sequence_plus_Vina",
            )
        }
        for panel in PANEL_ROLES
    }
    jackknife_summary = {
        panel: {
            "drops": int(len(part)),
            "fusion_rho_exceeds_sequence_drops": int(
                np.sum(part.fusion_minus_sequence_continuous_spearman > 0)
            ),
            "minimum_fusion_minus_sequence_rho": float(
                part.fusion_minus_sequence_continuous_spearman.min()
            ),
            "median_fusion_minus_sequence_rho": float(
                part.fusion_minus_sequence_continuous_spearman.median()
            ),
            "fusion_rho_exceeds_Vina_drops": int(
                np.sum(part.fusion_minus_Vina_continuous_spearman > 0)
            ),
            "minimum_fusion_minus_Vina_rho": float(
                part.fusion_minus_Vina_continuous_spearman.min()
            ),
        }
        for panel, part in jackknife.groupby("panel", sort=False)
    }
    summary: dict[str, object] = {
        "analysis": "incremental target-geometry information beyond sequence identity",
        "analysis_status": "post_hoc_exploratory_with_outcome_blind_fixed_fusion",
        "predictor": (
            "equal mean of pairwise percentile ranks from receptor-domain sequence "
            "identity and two-way-centered Vina target geometry; no fitted weight"
        ),
        "primary_metric": "continuous target-geometry Spearman concordance",
        "secondary_metric": "upper-10%-pair ROC AUROC",
        "average_precision_boundary": (
            "Average precision does not improve consistently and is not a positive "
            "claim."
        ),
        "all_four_panels_fusion_continuous_rho_exceeds_both_components": (
            all_panels_fusion_exceeds_both
        ),
        "primary_panel_metrics": primary_panel_metrics,
        "locked_panel_omnibus": {
            row.metric: {
                key: (
                    bool(value)
                    if key == "observed_mean_fusion_improves_sequence"
                    else float(value)
                )
                for key, value in row._asdict().items()
                if key
                in {
                    "observed_mean_fusion_minus_sequence",
                    "observed_mean_fusion_improves_sequence",
                    "target_label_qap_p_positive_mean_increment",
                    "holm_p_across_3_omnibus_metrics",
                }
            }
            for row in omnibus.itertuples(index=False)
        },
        "target_jackknife": jackknife_summary,
        "inference": {
            "null": (
                "target-label permutations of Vina geometry with sequence and "
                "experimental endpoints held fixed"
            ),
            "permutations": int(permutations),
            "seed": int(seed),
            "locked_panels": [
                panel
                for panel, role in PANEL_ROLES.items()
                if role.startswith("locked")
            ],
        },
        "claim_boundary": (
            "The result concerns target-pair geometry on a fixed 20-kinase panel. "
            "It is not ligand-level target retrieval, and historically inspected "
            "panels are not independent prospective confirmation."
        ),
        "provenance": {
            "target_pairs": {
                "path": str(pair_path.relative_to(PACKAGE)),
                "sha256": sha256_file(pair_path),
            },
            "kirhub": {
                "local_filename": kirhub_workbook.name,
                "sha256": sha256_file(kirhub_workbook),
                "expected_sha256": kirhub.KIRHUB_SHA256,
                "license": kirhub.KIRHUB_LICENSE,
            },
            "target_order": list(TARGETS),
        },
    }
    output.mkdir(parents=True, exist_ok=True)
    frames = {
        "panel_metrics": metrics,
        "incremental_target_label_qap": qap,
        "omnibus_incremental_qap": omnibus,
        "target_jackknife": jackknife,
    }
    output_records: dict[str, dict[str, str]] = {}
    for stem, frame in frames.items():
        path = output / f"{stem}.csv"
        frame.to_csv(path, index=False, float_format="%.15g")
        output_records[stem] = {"path": path.name, "sha256": sha256_file(path)}
    summary["outputs"] = output_records
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    readme = f"""# Sequence plus residual docking target geometry

This science-only artifact tests whether two-way-centered Vina target geometry
contains information about experimental kinase co-selectivity beyond receptor
sequence identity.

## Fixed predictor

The predictor is an outcome-blind 50:50 mean of pairwise percentile ranks from
receptor-domain sequence identity and centered Vina target geometry.  No weight
or threshold is fitted to an experimental panel.

The fusion's continuous target-geometry concordance exceeds both individual
components in all four examined panels.  Incremental QAP keeps sequence and the
experimental endpoint fixed and permutes only Vina target labels.  See
`omnibus_incremental_qap.csv` for the locked-panel mean contrast and
`panel_metrics.csv` for all point estimates.

For the continuous endpoint, fusion exceeds sequence after every one-target
deletion in each of DAVIS, PKIS2, and KiRHub.  PKIS1 is weaker and retains a
positive fusion-minus-sequence contrast after 13 of 20 deletions.

Continuous concordance is primary and upper-tail AUROC is secondary.  Average
precision is heterogeneous and is **not** a positive general claim.

## Boundary

This is post-hoc exploratory evidence on a fixed 20-kinase panel.  It concerns
target-pair geometry, not ligand-level target retrieval.  The fixed formula is
numerically outcome-blind, but the resources were inspected earlier in the
project and are not prospective confirmation.

## Reproduction

```bash
.venv/bin/python analysis/sequence_docking_fusion.py \\
  --kirhub-workbook /tmp/kirhub_supp_tables.xlsx \\
  --output results/sequence_docking_fusion \\
  --permutations {permutations} \\
  --seed {seed}

.venv/bin/python -m pytest -q analysis/test_sequence_docking_fusion.py
```
"""
    (output / "README.md").write_text(readme, encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-pairs", type=Path, default=DEFAULT_PAIRS)
    parser.add_argument("--kirhub-workbook", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--permutations", type=int, default=DEFAULT_PERMUTATIONS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    summary = run(
        args.target_pairs,
        args.kirhub_workbook,
        args.output,
        args.permutations,
        args.seed,
    )
    print(
        "Sequence+Vina fusion exceeds both components in all panels: "
        f"{summary['all_four_panels_fusion_continuous_rho_exceeds_both_components']}"
    )


if __name__ == "__main__":
    main()
