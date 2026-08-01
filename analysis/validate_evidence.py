#!/usr/bin/env python3
"""Validate mathematical and cross-analysis invariants in the evidence ledger."""

from __future__ import annotations

import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "results" / "evidence_summary.json"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def close(first: float, second: float, tolerance: float = 1e-10) -> bool:
    return math.isclose(float(first), float(second), rel_tol=tolerance, abs_tol=tolerance)


def main() -> None:
    evidence = json.loads(EVIDENCE.read_text())

    for dataset in ("docking44", "dockstring58"):
        ladder = evidence["centering_ladder"][dataset]
        for surface in ("raw", "interaction"):
            record = ladder[surface]
            targets = int(record["n_targets"])
            mean_r2 = float(record["mean_squared_offdiagonal_correlation"])
            implied_pr = targets / (1.0 + (targets - 1.0) * mean_r2)
            require(
                close(record["participation_ratio"], implied_pr),
                f"{dataset}/{surface}: PR does not equal its mean-r^2 identity",
            )
            require(
                close(sum(record["eigenvalues"]), targets),
                f"{dataset}/{surface}: correlation eigenvalues do not sum to target count",
            )
        require(
            ladder["interaction"]["numerical_zero_eigenvalues_at_1e-10"] >= 1,
            f"{dataset}: two-way-centered spectrum lacks its structural zero",
        )

    for dataset, record in evidence["row_norm_preserving_residual_null"].items():
        repeats = int(record["repeats"])
        require(
            record["observed_residual"] < record["null_residual"]["minimum"],
            f"{dataset}: observed residual PR is not below every row-norm null realization",
        )
        require(
            close(
                record["empirical_lower_tail_p_for_residual_pr"],
                1.0 / (repeats + 1.0),
            ),
            f"{dataset}: row-norm-null lower-tail probability is inconsistent",
        )

    for dataset, record in evidence["residual_structure_characterization"].items():
        for surface in ("raw", "residual"):
            distribution = record[f"{surface}_correlation_distribution"]
            values = distribution["values"]
            require(
                close(
                    distribution["mean_absolute"],
                    sum(abs(value) for value in values) / len(values),
                ),
                f"{dataset}/{surface}: stored correlation summary does not match values",
            )

    expanded = evidence["expanded_target_preference_benchmark"]
    require(
        expanded["co_primary_operational_estimands"]["broad_coverage"]
        == "primary_all_exact",
        "broad operational estimand is not declared in the ledger",
    )
    broad = expanded["primary_all_exact"]
    representations = broad["representations"]
    require(
        close(
            representations["target_centered_unscaled"][
                "mean_per_ligand_pairwise_accuracy"
            ],
            representations["two_way_centered_unscaled"][
                "mean_per_ligand_pairwise_accuracy"
            ],
        ),
        "unscaled target-centered and two-way-centered rankings are not invariant",
    )
    for comparison, first, second in (
        (
            "two_way_residual_minus_absolute_vina",
            "two_way_residual",
            "absolute_vina",
        ),
        (
            "two_way_residual_minus_column_standardized",
            "two_way_residual",
            "column_standardized",
        ),
    ):
        implied = (
            representations[first]["mean_per_ligand_pairwise_accuracy"]
            - representations[second]["mean_per_ligand_pairwise_accuracy"]
        )
        require(
            close(
                broad["paired_comparisons"][comparison]["plugin_mean_difference"],
                implied,
            ),
            f"{comparison}: paired contrast does not match representation means",
        )

    require(
        expanded["assay_sensitivities"]["human_binding_Ki_Kd"]["n_targets"] == 30,
        "human binding Ki/Kd target count drifted",
    )
    print("Evidence invariants validated")


if __name__ == "__main__":
    main()
