#!/usr/bin/env python3
"""Censoring-aware sensitivity for the fixed 20-kinase DAVIS geometry.

The released DAVIS-Complete Kd values are capped at 10,000 nM, which becomes
a pKd floor at 5.  This script asks whether the fixed-panel conclusions persist
when DAVIS is represented only by above-floor status and when targets with very
little above-floor support are removed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import average_precision_score, roc_auc_score

import dense_davis_benchmark as davis
import kirhub_external_validation as kirhub


PACKAGE = Path(__file__).resolve().parents[1]
DEFAULT_DAVIS = PACKAGE / "data/frozen/davis_complete.tab.gz"
DEFAULT_TARGET_PAIRS = PACKAGE / "results/klifs_pocket_control/target_pairs.csv"
DEFAULT_OUTPUT = PACKAGE / "results/davis_fixed20_censoring_sensitivity"
TARGETS = tuple(kirhub.TARGETS)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def upper_triangle(matrix: np.ndarray) -> np.ndarray:
    return np.asarray(matrix, dtype=np.float64)[np.triu_indices(len(matrix), k=1)]


def geometry(matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float64)
    values = values - values.mean(axis=1, keepdims=True)
    return np.corrcoef(values, rowvar=False)


def percentile_ranks(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return (stats.rankdata(values, method="average") - 0.5) / len(values)


def spearman_edges(first: np.ndarray, second: np.ndarray) -> float:
    return float(stats.spearmanr(upper_triangle(first), upper_triangle(second)).statistic)


def qap_spearman(
    predictor: np.ndarray,
    endpoint: np.ndarray,
    *,
    permutations: int,
    seed: int,
) -> dict:
    observed = spearman_edges(predictor, endpoint)
    rng = np.random.default_rng(seed)
    exceedances = 0
    for _ in range(permutations):
        order = rng.permutation(len(endpoint))
        permuted = endpoint[np.ix_(order, order)]
        exceedances += spearman_edges(predictor, permuted) >= observed
    return {
        "spearman": observed,
        "permutations": int(permutations),
        "seed": int(seed),
        "one_sided_target_label_qap_p_positive": float(
            (1 + exceedances) / (permutations + 1)
        ),
    }


def endpoint_qap(
    labels: np.ndarray,
    predictor: np.ndarray,
    *,
    permutations: int,
    seed: int,
) -> dict:
    labels = np.asarray(labels, dtype=bool)
    observed_auc = float(roc_auc_score(labels, upper_triangle(predictor)))
    observed_ap = float(average_precision_score(labels, upper_triangle(predictor)))
    rng = np.random.default_rng(seed)
    auc_exceedances = 0
    ap_exceedances = 0
    for _ in range(permutations):
        order = rng.permutation(len(predictor))
        values = upper_triangle(predictor[np.ix_(order, order)])
        auc_exceedances += roc_auc_score(labels, values) >= observed_auc
        ap_exceedances += average_precision_score(labels, values) >= observed_ap
    return {
        "positive_pairs": int(labels.sum()),
        "total_pairs": int(len(labels)),
        "roc_auc": observed_auc,
        "average_precision": observed_ap,
        "permutations": int(permutations),
        "seed": int(seed),
        "roc_auc_target_label_qap_p_positive": float(
            (1 + auc_exceedances) / (permutations + 1)
        ),
        "average_precision_target_label_qap_p_positive": float(
            (1 + ap_exceedances) / (permutations + 1)
        ),
    }


def load_davis(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, sep="\t", usecols=["drug_name", "protein", "y"])
    target_map = {target: davis.TARGET_MAP[target] for target in TARGETS}
    selected = frame.loc[frame.protein.isin(target_map.values())]
    matrix = selected.pivot(index="drug_name", columns="protein", values="y")
    matrix = matrix.reindex(columns=list(target_map.values()))
    matrix.columns = list(target_map)
    if matrix.shape != (72, 20) or matrix.isna().any().any():
        raise ValueError(f"unexpected DAVIS support: {matrix.shape}")
    return matrix


def finalize_existing_release(
    *,
    davis_path: Path,
    target_pairs_path: Path,
    output: Path,
    permutations: int,
    seed: int,
) -> dict:
    """Add the local-only >90%-floor sensitivity to an existing release result.

    The KiRHub workbook is source-restricted and is not needed to recompute this
    reviewer-requested DAVIS sensitivity.  Existing KiRHub aggregate validation
    fields are preserved byte-for-value from the frozen summary.
    """

    summary_path = output / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(summary_path)
    summary = json.loads(summary_path.read_text())
    matrix = load_davis(davis_path)
    pair_frame = ordered_pair_frame(target_pairs_path)
    docking = np.eye(len(TARGETS), dtype=np.float64)
    docking[np.triu_indices(len(TARGETS), k=1)] = pair_frame[
        "centered_docking_pair_percentile"
    ].to_numpy(dtype=np.float64)
    docking = docking + docking.T - np.eye(len(TARGETS), dtype=np.float64)
    continuous = geometry(matrix.to_numpy(dtype=np.float64))
    status = geometry(matrix.gt(5.0).astype(float).to_numpy(dtype=np.float64))
    floor_counts = matrix.eq(5.0).sum(axis=0)
    floor_fractions = floor_counts / len(matrix)
    keep = floor_fractions.le(0.90)
    retained = [target for target in TARGETS if keep[target]]
    removed = [target for target in TARGETS if not keep[target]]
    indices = np.asarray([TARGETS.index(target) for target in retained], dtype=int)
    predictor = docking[np.ix_(indices, indices)]
    continuous_subset = continuous[np.ix_(indices, indices)]
    status_subset = status[np.ix_(indices, indices)]
    summary["measurement_boundary"][
        "targets_above_90_percent_at_floor_names"
    ] = removed
    summary["exclude_targets_over_90_percent_floor_sensitivity"] = {
        "rule": (
            "exclude targets with more than 90% of the 72 released DAVIS "
            "values at the pKd=5 reporting floor"
        ),
        "threshold_is_strict": True,
        "maximum_retained_fraction_at_floor": 0.90,
        "targets_retained": len(retained),
        "retained_target_names": retained,
        "targets_removed": len(removed),
        "removed_target_names": removed,
        "continuous_vs_above_floor_status_geometry_spearman": spearman_edges(
            continuous_subset, status_subset
        ),
        "centered_docking_vs_continuous_davis": qap_spearman(
            predictor,
            continuous_subset,
            permutations=permutations,
            seed=seed + 10,
        ),
        "centered_docking_vs_above_floor_status_davis": qap_spearman(
            predictor,
            status_subset,
            permutations=permutations,
            seed=seed + 404,
        ),
        "inference_boundary": (
            "Target-label QAP is conditional on this fixed retained 16-target "
            "panel. The binary endpoint tests co-response in exceeding the reporting "
            "floor, not quantitative affinity, and the exclusion rule is a sensitivity "
            "analysis rather than a missing-data correction."
        ),
    }
    target_censoring = pd.DataFrame(
        {
            "target": list(TARGETS),
            "cells": len(matrix),
            "cells_at_pkd_5_floor": [int(floor_counts[target]) for target in TARGETS],
            "fraction_at_pkd_5_floor": [
                float(floor_fractions[target]) for target in TARGETS
            ],
            "cells_above_pkd_5_floor": [
                int(len(matrix) - floor_counts[target]) for target in TARGETS
            ],
            "excluded_by_gt_90_percent_floor_rule": [
                bool(not keep[target]) for target in TARGETS
            ],
        }
    )
    target_censoring.to_csv(output / "target_censoring.csv", index=False)
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    return summary


def ordered_pair_frame(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    expected = [
        (TARGETS[first], TARGETS[second])
        for first in range(len(TARGETS))
        for second in range(first + 1, len(TARGETS))
    ]
    observed = list(zip(frame.target_a, frame.target_b, strict=True))
    if observed != expected:
        raise ValueError("target-pair table is not in fixed upper-triangle order")
    return frame


def run(
    *,
    davis_path: Path,
    target_pairs_path: Path,
    kirhub_workbook: Path,
    permutations: int,
    seed: int,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    davis_matrix = load_davis(davis_path)
    pair_frame = ordered_pair_frame(target_pairs_path)
    docking = np.eye(len(TARGETS), dtype=np.float64)
    docking[np.triu_indices(len(TARGETS), k=1)] = pair_frame[
        "centered_docking_pair_percentile"
    ].to_numpy(dtype=np.float64)
    docking = docking + docking.T - np.eye(len(TARGETS), dtype=np.float64)

    continuous_geometry = geometry(davis_matrix.to_numpy(dtype=np.float64))
    above_floor = davis_matrix.gt(5.0).astype(float)
    status_geometry = geometry(above_floor.to_numpy(dtype=np.float64))
    floor_counts = davis_matrix.eq(5.0).sum(axis=0)
    above_floor_counts = len(davis_matrix) - floor_counts
    floor_fractions = floor_counts / len(davis_matrix)
    target_censoring = pd.DataFrame(
        {
            "target": list(TARGETS),
            "cells": len(davis_matrix),
            "cells_at_pkd_5_floor": [int(floor_counts[target]) for target in TARGETS],
            "fraction_at_pkd_5_floor": [
                float(floor_fractions[target]) for target in TARGETS
            ],
            "cells_above_pkd_5_floor": [
                int(above_floor_counts[target]) for target in TARGETS
            ],
            "excluded_by_gt_90_percent_floor_rule": [
                bool(floor_fractions[target] > 0.90) for target in TARGETS
            ],
        }
    )

    quality_rows: list[dict] = []
    for minimum_above_floor in (0, 5, 10, 15):
        keep = above_floor_counts >= minimum_above_floor
        names = [target for target in TARGETS if keep[target]]
        indices = np.array([TARGETS.index(target) for target in names], dtype=int)
        predictor = docking[np.ix_(indices, indices)]
        endpoint = continuous_geometry[np.ix_(indices, indices)]
        result = qap_spearman(
            predictor,
            endpoint,
            permutations=permutations,
            seed=seed + minimum_above_floor,
        )
        quality_rows.append(
            {
                "minimum_values_above_pkd_floor_per_target": minimum_above_floor,
                "targets_retained": len(names),
                "target_names": ";".join(names),
                **result,
            }
        )

    status_result = qap_spearman(
        docking,
        status_geometry,
        permutations=permutations,
        seed=seed + 101,
    )

    # This is the reviewer-requested, named sensitivity.  It is deliberately
    # separated from the threshold sweep above: it removes exactly the targets
    # whose released DAVIS values are at the pKd=5 reporting floor in more than
    # 90% of the 72 compounds, and evaluates both the continuous pKd endpoint
    # and the coarse above-floor-status endpoint on the identical 16 targets.
    retain_90 = floor_fractions.le(0.90)
    retained_90_names = [target for target in TARGETS if retain_90[target]]
    removed_90_names = [target for target in TARGETS if not retain_90[target]]
    retained_90_indices = np.asarray(
        [TARGETS.index(target) for target in retained_90_names], dtype=int
    )
    docking_90 = docking[np.ix_(retained_90_indices, retained_90_indices)]
    continuous_90 = continuous_geometry[
        np.ix_(retained_90_indices, retained_90_indices)
    ]
    status_90 = status_geometry[np.ix_(retained_90_indices, retained_90_indices)]
    explicit_gt90_sensitivity = {
        "rule": (
            "exclude targets with more than 90% of the 72 released DAVIS "
            "values at the pKd=5 reporting floor"
        ),
        "threshold_is_strict": True,
        "maximum_retained_fraction_at_floor": 0.90,
        "targets_retained": len(retained_90_names),
        "retained_target_names": retained_90_names,
        "targets_removed": len(removed_90_names),
        "removed_target_names": removed_90_names,
        "continuous_vs_above_floor_status_geometry_spearman": spearman_edges(
            continuous_90, status_90
        ),
        "centered_docking_vs_continuous_davis": qap_spearman(
            docking_90,
            continuous_90,
            permutations=permutations,
            seed=seed + 10,
        ),
        "centered_docking_vs_above_floor_status_davis": qap_spearman(
            docking_90,
            status_90,
            permutations=permutations,
            seed=seed + 404,
        ),
        "inference_boundary": (
            "Target-label QAP is conditional on this fixed retained 16-target "
            "panel. The binary endpoint tests co-response in exceeding the reporting "
            "floor, not quantitative affinity, and the exclusion rule is a sensitivity "
            "analysis rather than a missing-data correction."
        ),
    }

    status_percentile = percentile_ranks(upper_triangle(status_geometry))
    alternative_count = (
        np.vstack(
            [
                status_percentile >= 0.9,
                pair_frame.PKIS2_centered_pair_percentile.to_numpy() >= 0.9,
                pair_frame.PKIS1_centered_pair_percentile.to_numpy() >= 0.9,
            ]
        ).sum(axis=0)
        >= 2
    )
    primary_labels = pair_frame.primary_replicated_positive.to_numpy(dtype=bool)

    kirhub_frame = kirhub.load_kirhub(kirhub_workbook)
    kirhub_values = kirhub_frame[list(TARGETS)].to_numpy(dtype=np.float64)
    kirhub_geometry = kirhub.correlation_geometry(kirhub_values, "two_way_center")
    alternative_validation = endpoint_qap(
        alternative_count,
        kirhub_geometry,
        permutations=permutations,
        seed=seed + 202,
    )

    summary = {
        "analysis": "fixed-20 DAVIS censoring sensitivity",
        "status": "exploratory sensitivity; not preregistered",
        "measurement_boundary": {
            "ligands": int(len(davis_matrix)),
            "targets": int(davis_matrix.shape[1]),
            "cells": int(davis_matrix.size),
            "cells_at_pkd_5_floor": int(davis_matrix.eq(5.0).to_numpy().sum()),
            "fraction_at_pkd_5_floor": float(davis_matrix.eq(5.0).to_numpy().mean()),
            "targetwise_floor_fraction_range": [
                float(davis_matrix.eq(5.0).mean(axis=0).min()),
                float(davis_matrix.eq(5.0).mean(axis=0).max()),
            ],
            "targets_above_90_percent_at_floor": int(
                (davis_matrix.eq(5.0).mean(axis=0) > 0.90).sum()
            ),
            "targets_above_90_percent_at_floor_names": removed_90_names,
        },
        "continuous_vs_above_floor_status_geometry_spearman": spearman_edges(
            continuous_geometry, status_geometry
        ),
        "centered_docking_vs_continuous_davis": quality_rows[0],
        "centered_docking_vs_above_floor_status_davis": status_result,
        "minimum_above_floor_target_sensitivities": quality_rows[1:],
        "exclude_targets_over_90_percent_floor_sensitivity": (
            explicit_gt90_sensitivity
        ),
        "alternative_endpoint": {
            "definition": (
                "top-decile centred co-response in at least two of binary DAVIS "
                "above-floor status, PKIS2 and PKIS1"
            ),
            "positive_pairs": int(alternative_count.sum()),
            "primary_positive_pairs": int(primary_labels.sum()),
            "shared_positive_pairs": int(np.sum(alternative_count & primary_labels)),
            "primary_pairs_retained_fraction": float(
                np.sum(alternative_count & primary_labels) / primary_labels.sum()
            ),
            "KiRHub_centered_geometry_validation": alternative_validation,
        },
        "interpretation": (
            "DAVIS supports a repeatable coarse co-response or measurement-limit-status "
            "geometry, but its fixed-20 values should not be read as a finely resolved "
            "quantitative-affinity surface."
        ),
        "provenance": {
            "davis_path": str(davis_path.relative_to(PACKAGE)),
            "davis_sha256": sha256_file(davis_path),
            "target_pairs_path": str(target_pairs_path.relative_to(PACKAGE)),
            "target_pairs_sha256": sha256_file(target_pairs_path),
            "kirhub_workbook_sha256": sha256_file(kirhub_workbook),
            "kirhub_workbook_redistributed": False,
        },
    }
    return summary, pd.DataFrame(quality_rows), target_censoring


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--davis", type=Path, default=DEFAULT_DAVIS)
    parser.add_argument("--target-pairs", type=Path, default=DEFAULT_TARGET_PAIRS)
    parser.add_argument("--kirhub-workbook", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--permutations", type=int, default=49_999)
    parser.add_argument("--seed", type=int, default=20_260_809)
    parser.add_argument(
        "--finalize-existing",
        action="store_true",
        help="recompute the local DAVIS-only sensitivity without the KiRHub workbook",
    )
    args = parser.parse_args()

    if args.finalize_existing:
        summary = finalize_existing_release(
            davis_path=args.davis,
            target_pairs_path=args.target_pairs,
            output=args.output,
            permutations=args.permutations,
            seed=args.seed,
        )
        print(
            json.dumps(
                summary["exclude_targets_over_90_percent_floor_sensitivity"],
                indent=2,
            )
        )
        return
    if args.kirhub_workbook is None:
        parser.error("--kirhub-workbook is required unless --finalize-existing is used")

    summary, quality, target_censoring = run(
        davis_path=args.davis,
        target_pairs_path=args.target_pairs,
        kirhub_workbook=args.kirhub_workbook,
        permutations=args.permutations,
        seed=args.seed,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )
    quality.to_csv(args.output / "target_quality_sensitivity.csv", index=False)
    target_censoring.to_csv(args.output / "target_censoring.csv", index=False)
    print(args.output)


if __name__ == "__main__":
    main()
