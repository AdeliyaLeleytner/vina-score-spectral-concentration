#!/usr/bin/env python3
"""Validate dense DAVIS and PKIS2 operational-benchmark outputs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd


PACKAGE = Path(__file__).resolve().parents[1]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def load_strict_json(path: Path) -> dict:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-standard JSON constant {value} in {path}")

    return json.loads(path.read_text(), parse_constant=reject_constant)


def close(first: float, second: float, tolerance: float = 1e-12) -> bool:
    return math.isclose(float(first), float(second), rel_tol=tolerance, abs_tol=tolerance)


def validate_davis(results: Path, *, validate_csvs: bool = True) -> None:
    report = load_strict_json(results / "dense_davis_benchmark.json")
    support = report["source_support"]
    require(report["release_eligible_identity_scan"], "DAVIS used a smoke identity scan")
    require(support["primary_full_key_ligands"] == 59, "DAVIS ligand support changed")
    require(support["shared_targets"] == 21, "DAVIS target support changed")
    require(support["primary_experimental_cells"] == 1_239, "DAVIS cells changed")
    require(support["primary_floor_cells_pkd_5"] == 847, "DAVIS floor count changed")
    require(support["primary_uncensored_cells"] == 392, "DAVIS uncensored count changed")

    primary = report["primary"]
    invariant = primary["unscaled_row_centering_ranking_invariant"]
    require(close(invariant["difference"], 0.0), "DAVIS row-centering invariant failed")
    for representation, record in primary["co_primary_hit_detection_auc"].items():
        require(
            close(record["identity_check_against_floor_vs_uncensored_concordance"], 0.0),
            f"DAVIS hit-AUC identity failed for {representation}",
        )
    contrast = primary["paired_comparisons"][
        "two_way_residual_minus_absolute_vina"
    ]["0.0"]["all_informative"]["plugin_mean_difference"]
    residual = primary["representations"]["two_way_residual"]["0.0"][
        "all_informative"
    ]["mean_per_ligand_pairwise_concordance"]
    absolute = primary["representations"]["absolute_vina"]["0.0"][
        "all_informative"
    ]["mean_per_ligand_pairwise_concordance"]
    require(close(contrast, residual - absolute), "DAVIS paired contrast drifted")

    calibration = report["molecular_size_matched_reference_calibration"]
    require(calibration["status"] == "complete", "DAVIS calibration sensitivity missing")
    full_smd = calibration[
        "standardized_mean_difference_full_reference_minus_evaluation"
    ]
    matched_smd = calibration[
        "standardized_mean_difference_matched_reference_minus_evaluation"
    ]
    require(max(abs(value) for value in full_smd) > 0.4, "DAVIS full reference mismatch vanished")
    require(max(abs(value) for value in matched_smd) < 0.02, "DAVIS matched reference is imbalanced")

    expected_rows = {
        "dense_davis_primary_metrics.csv": 114,
        "dense_davis_target_quality.csv": 21,
        "dense_davis_primary_molecule_mapping.csv": 59,
    }
    if validate_csvs:
        for filename, expected in expected_rows.items():
            require(len(pd.read_csv(results / filename)) == expected, f"{filename} row count changed")


def validate_pkis2(results: Path, *, validate_csvs: bool = True) -> None:
    report = load_strict_json(results / "dense_pkis2_benchmark.json")
    support = report["source_support"]
    require(report["release_eligible_identity_scan"], "PKIS2 used a smoke identity scan")
    require(support["primary_full_key_ligands"] == 154, "PKIS2 ligand support changed")
    require(support["shared_targets"] == 21, "PKIS2 target support changed")
    require(support["primary_experimental_cells"] == 3_234, "PKIS2 cells changed")
    require(report["assay_semantics"]["not_an_affinity_measurement"], "PKIS2 assay was mislabeled")

    primary = report["primary"]
    invariant = primary["unscaled_row_centering_ranking_invariant"]
    require(close(invariant["difference"], 0.0), "PKIS2 row-centering invariant failed")
    metric = primary["representations"]["absolute_vina"][
        "absolute_difference_gt_10"
    ]
    require(metric["evaluated_ligands"] == 154, "PKIS2 primary ligand count changed")
    require(metric["evaluated_pairs"] == 16_417, "PKIS2 primary pair count changed")
    binary = primary["binary_activity_at_65_percent"]
    require(
        binary["absolute_vina"]["evaluated_ligands_with_both_classes"] == 123,
        "PKIS2 binary ligand count changed",
    )
    require(
        binary["absolute_vina"]["positive_negative_pairs"] == 5_426,
        "PKIS2 binary pair count changed",
    )
    for representation, record in binary.items():
        require(
            close(
                record[
                    "identity_check_auc_vs_active_inactive_pairwise_max_abs_difference"
                ],
                0.0,
            ),
            f"PKIS2 activity-AUC identity failed for {representation}",
        )
    for label, record in report["molecular_identity_sensitivities"].items():
        require(
            record["all_evaluation_blocks_excluded_from_reference"],
            f"PKIS2 reference leakage flag failed for {label}",
        )

    both_active = primary["exploratory_both_active_pairwise_concordance"]
    both_active_absolute = both_active["absolute_vina"]
    require(
        both_active_absolute["experimental_margin_name"]
        == "absolute_difference_gt_10"
        and close(
            both_active_absolute["experimental_margin_percentage_points"],
            10.0,
        ),
        "PKIS2 both-active primary margin drifted",
    )
    require(
        both_active_absolute["evaluated_ligands"] == 57
        and both_active_absolute["evaluated_pairs"] == 256,
        "PKIS2 both-active primary support changed",
    )
    exact_non_tie = primary[
        "secondary_exact_non_tie_both_active_pairwise_concordance"
    ]
    exact_non_tie_absolute = exact_non_tie["absolute_vina"]
    require(
        exact_non_tie_absolute["experimental_margin_name"] == "exact_non_ties"
        and close(
            exact_non_tie_absolute["experimental_margin_percentage_points"],
            0.0,
        ),
        "PKIS2 exact-non-tie both-active sensitivity drifted",
    )
    require(
        exact_non_tie_absolute["evaluated_ligands"] == 66
        and exact_non_tie_absolute["evaluated_pairs"] == 591,
        "PKIS2 exact-non-tie both-active support changed",
    )

    calibration = report["molecular_size_matched_reference_calibration"]
    require(calibration["status"] == "complete", "PKIS2 calibration sensitivity missing")
    full_smd = calibration[
        "standardized_mean_difference_full_reference_minus_evaluation"
    ]
    require(max(abs(value) for value in full_smd) > 0.1, "PKIS2 full reference mismatch vanished")
    for neighbours, record in calibration["by_neighbour_count"].items():
        matched_smd = record[
            "standardized_mean_difference_matched_reference_minus_evaluation"
        ]
        require(
            max(abs(value) for value in matched_smd) < 0.01,
            f"PKIS2 {neighbours}-neighbour reference is imbalanced",
        )

    expected_rows = {
        "dense_pkis2_primary_metrics.csv": 24,
        "dense_pkis2_paired_comparisons.csv": 28,
        "dense_pkis2_primary_per_ligand_metrics.csv": 3_696,
        "dense_pkis2_binary_activity_metrics.csv": 6,
        "dense_pkis2_binary_activity_per_ligand_metrics.csv": 924,
        "dense_pkis2_paired_binary_comparisons.csv": 14,
        "dense_pkis2_exploratory_both_active.csv": 6,
        "dense_pkis2_exploratory_both_active_paired_comparisons.csv": 7,
        "dense_pkis2_secondary_exact_non_tie_both_active.csv": 6,
        "dense_pkis2_secondary_exact_non_tie_both_active_paired_comparisons.csv": 7,
        "dense_pkis2_target_quality.csv": 21,
        "dense_pkis2_target_pair_support.csv": 210,
        "dense_pkis2_target_jackknife.csv": 792,
        "dense_pkis2_molecular_size_matched_reference_calibration.csv": 3,
        "dense_pkis2_molecular_identity_sensitivities.csv": 4,
        "dense_pkis2_duplicate_sensitivities.csv": 7,
        "dense_pkis2_primary_molecule_mapping.csv": 154,
    }
    if validate_csvs:
        for filename, expected in expected_rows.items():
            require(len(pd.read_csv(results / filename)) == expected, f"{filename} row count changed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=PACKAGE / "results")
    parser.add_argument(
        "--include-detailed-csvs",
        action="store_true",
        help=(
            "also validate optional detailed CSV exports; the release archive validates "
            "the frozen report JSONs by default"
        ),
    )
    args = parser.parse_args()
    validate_davis(args.results_dir, validate_csvs=args.include_detailed_csvs)
    validate_pkis2(args.results_dir, validate_csvs=args.include_detailed_csvs)
    print("DENSE_BENCHMARK_VALIDATION_OK")


if __name__ == "__main__":
    main()
