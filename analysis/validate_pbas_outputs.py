#!/usr/bin/env python3
"""Validate the machine-readable output of the optional PBAS spectral audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


PACKAGE = Path(__file__).resolve().parents[1]
EXPECTED_SHA256 = "774aa84179c5707e40ce0d054722ff85a2ab5a1a2de6b561b8b55139c8626dea"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PACKAGE / "results" / "pbas_proteome",
    )
    args = parser.parse_args()
    summary_path = args.output_dir / "summary.json"
    report = json.loads(summary_path.read_text())

    source = report["source"]
    require(source["n_ligands"] == 7_582, "PBAS ligand count changed")
    require(source["n_targets"] == 19_135, "PBAS target count changed")
    require(source["sha256"] == EXPECTED_SHA256, "PBAS source checksum changed")
    require(source["checksum_verified"], "PBAS checksum was not verified")
    require(source["missing_cells"] == 4_841_504, "PBAS missing-cell count changed")
    require(source["positive_cells"] == 22_354_721, "PBAS positive-score count changed")

    for variant, variant_report in report["variants"].items():
        for surface, surface_report in variant_report["surfaces"].items():
            for spectrum in ("correlation_spectrum", "covariance_spectrum"):
                metrics = surface_report[spectrum]
                implied_pr = (
                    metrics["trace"] ** 2 / metrics["squared_frobenius_norm"]
                )
                require(
                    abs(metrics["participation_ratio"] - implied_pr) < 1e-9,
                    f"{variant}/{surface}/{spectrum}: PR identity failed",
                )
                require(
                    1 <= metrics["participation_ratio"]
                    <= metrics["algebraic_rank_upper_bound"] * (1 + 2e-5),
                    f"{variant}/{surface}/{spectrum}: PR outside rank bounds",
                )

    require(
        len(pd.read_csv(args.output_dir / "target_missingness.csv")) == 19_135,
        "PBAS target-missingness table has the wrong length",
    )
    subsampling = pd.read_csv(args.output_dir / "target_subsampling.csv")
    design = report["target_subsampling"]
    require(
        len(subsampling)
        == len(design["target_counts"]) * design["repeats_per_target_count"],
        "PBAS target-subsampling table has the wrong length",
    )
    for surface, record in report["numerical_validation"]["surfaces"].items():
        require(
            record["relative_absolute_error"] < 2e-5,
            f"PBAS {surface} float32/float64 cross-check failed",
        )

    print("PBAS_OUTPUT_VALIDATION_OK")


if __name__ == "__main__":
    main()
