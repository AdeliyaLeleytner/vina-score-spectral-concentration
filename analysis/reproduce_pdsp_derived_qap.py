#!/usr/bin/env python3
"""Reproduce the reported PDSP paired-QAP table from released aggregates.

The compound-level PDSP export is not redistributed.  The release instead
contains the target-pair endpoint, sequence/family covariates and the derived
Docking-44 overlap-exclusion mask needed for the reported graph-level comparison.
This script combines those files with the redistributed Docking-44 matrix,
rebuilds the de-overlapped raw and two-way-centred docking maps, reruns the
declared permutations on both primary and complete-case support, and checks the
support-threshold sweep.

It audits the inference reported in the paper.  It does not reproduce the
upstream selection and aggregation of compound-level PDSP measurements; that
step requires the checksum-pinned source export described in
``external_sources.csv``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype


PACKAGE = Path(__file__).resolve().parents[1]
ANALYSIS = PACKAGE / "analysis"
if str(ANALYSIS) not in sys.path:
    sys.path.insert(0, str(ANALYSIS))

import pdsp_counterscreen_retrieval as audit  # noqa: E402


DEFAULT_PAIRS = PACKAGE / "results" / "pdsp_counterscreen_retrieval" / "target_pairs.csv"
DEFAULT_EXCLUSIONS = (
    PACKAGE / "results" / "pdsp_counterscreen_retrieval" / "excluded_docking44_ligand_ids.csv"
)
DEFAULT_EXPECTED = PACKAGE / "results" / "pdsp_counterscreen_retrieval" / "paired_qap.csv"
DEFAULT_COMPLETE_METRICS = (
    PACKAGE
    / "results"
    / "pdsp_counterscreen_retrieval"
    / "complete_case_support_metrics.csv"
)
DEFAULT_COMPLETE_QAP = (
    PACKAGE
    / "results"
    / "pdsp_counterscreen_retrieval"
    / "complete_case_support_qap.csv"
)
DEFAULT_SUPPORT_SWEEP = (
    PACKAGE
    / "results"
    / "pdsp_counterscreen_retrieval"
    / "minimum_pair_support_sensitivity.csv"
)
DEFAULT_DOCKING = PACKAGE / "data" / "frozen" / "df_final_v4.csv.gz"
KEYS = ("permutation_scheme", "contrast", "metric")
def released_context(
    *,
    target_pairs: Path = DEFAULT_PAIRS,
    exclusions: Path = DEFAULT_EXCLUSIONS,
    docking: Path = DEFAULT_DOCKING,
) -> tuple[pd.DataFrame, set[int], list[str], dict[str, np.ndarray]]:
    """Load and verify the released aggregate inputs shared by all checks."""
    pairs = pd.read_csv(target_pairs)
    required = {
        "target_a",
        "target_b",
        "experimental_spearman",
        "raw_docking",
        "residual_docking",
        "full_sequence_identity",
        "same_curated_family",
    }
    missing = required - set(pairs.columns)
    if missing:
        raise ValueError(f"target-pair table is missing columns: {sorted(missing)}")
    exclusion_frame = pd.read_csv(exclusions)
    if set(exclusion_frame.columns) != {"ligand_id"}:
        raise ValueError("exclusion table must contain only ligand_id")
    if exclusion_frame["ligand_id"].duplicated().any():
        raise ValueError("exclusion table contains duplicate ligand IDs")
    excluded_ligand_ids = set(exclusion_frame["ligand_id"].astype(int))
    targets = sorted(set(pairs["target_a"]) | set(pairs["target_b"]))
    docking_targets, matrices = audit.docking_geometries(
        docking,
        transform=audit.RESIDUAL_TRANSFORM,
        excluded_ligand_ids=excluded_ligand_ids,
    )
    aligned = {
        name: audit.align_docking_matrix(docking_targets, matrix, targets)
        for name, matrix in matrices.items()
    }

    target_index = {target: index for index, target in enumerate(targets)}
    first = np.asarray([target_index[value] for value in pairs["target_a"]], dtype=int)
    second = np.asarray([target_index[value] for value in pairs["target_b"]], dtype=int)
    for name, matrix in aligned.items():
        np.testing.assert_allclose(
            matrix[first, second],
            pairs[name].to_numpy(dtype=float),
            rtol=0.0,
            atol=2e-15,
            err_msg=f"released {name} edges do not match Docking-44",
        )

    return pairs, excluded_ligand_ids, targets, aligned


def reproduce(
    *,
    target_pairs: Path = DEFAULT_PAIRS,
    exclusions: Path = DEFAULT_EXCLUSIONS,
    docking: Path = DEFAULT_DOCKING,
    permutations: int = audit.DEFAULT_QAP_PERMUTATIONS,
    seed: int = audit.DEFAULT_SEED,
) -> pd.DataFrame:
    """Return primary paired-QAP rows from released aggregate/public inputs."""
    pairs, _, targets, aligned = released_context(
        target_pairs=target_pairs,
        exclusions=exclusions,
        docking=docking,
    )
    return audit.qap_paired_gains(pairs, targets, aligned, permutations, seed)


def reproduce_complete_case(
    *,
    target_pairs: Path = DEFAULT_PAIRS,
    exclusions: Path = DEFAULT_EXCLUSIONS,
    docking: Path = DEFAULT_DOCKING,
    permutations: int = audit.DEFAULT_QAP_PERMUTATIONS,
    seed: int = audit.DEFAULT_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return complete-case metrics and QAP from released inputs."""
    pairs, excluded, targets, _ = released_context(
        target_pairs=target_pairs,
        exclusions=exclusions,
        docking=docking,
    )
    return audit.complete_case_support_sensitivity(
        docking,
        pairs,
        targets,
        excluded_ligand_ids=excluded,
        transform=audit.RESIDUAL_TRANSFORM,
        permutations=permutations,
        seed=seed,
    )


def reproduce_support_sweep(
    target_pairs: Path = DEFAULT_PAIRS,
) -> pd.DataFrame:
    """Return the support-threshold sweep from the released pair aggregates."""
    return audit.minimum_pair_support_sensitivity(pd.read_csv(target_pairs))


def verify_table(
    reproduced: pd.DataFrame,
    expected_path: Path,
    keys: tuple[str, ...],
) -> None:
    """Fail if a reproduced table differs from its frozen report artifact."""
    expected = pd.read_csv(expected_path)
    if set(reproduced.columns) != set(expected.columns):
        raise AssertionError("reproduced table columns differ")
    reproduced = reproduced[expected.columns].sort_values(
        list(keys), kind="mergesort"
    ).reset_index(drop=True)
    expected = expected.sort_values(list(keys), kind="mergesort").reset_index(drop=True)
    if len(reproduced) != len(expected):
        raise AssertionError("reproduced table row count differs")
    for column in expected.columns:
        if is_numeric_dtype(expected[column]):
            np.testing.assert_allclose(
                reproduced[column].to_numpy(dtype=float),
                expected[column].to_numpy(dtype=float),
                rtol=0.0,
                atol=5e-13,
                equal_nan=True,
                err_msg=f"reproduced column differs: {column}",
            )
        elif reproduced[column].astype(str).tolist() != expected[column].astype(str).tolist():
            raise AssertionError(f"reproduced text column differs: {column}")


def verify(reproduced: pd.DataFrame, expected_path: Path = DEFAULT_EXPECTED) -> None:
    """Fail if primary paired-QAP rows differ from the frozen table."""
    verify_table(reproduced, expected_path, KEYS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-pairs", type=Path, default=DEFAULT_PAIRS)
    parser.add_argument("--exclusions", type=Path, default=DEFAULT_EXCLUSIONS)
    parser.add_argument("--docking", type=Path, default=DEFAULT_DOCKING)
    parser.add_argument("--expected", type=Path, default=DEFAULT_EXPECTED)
    parser.add_argument("--complete-metrics", type=Path, default=DEFAULT_COMPLETE_METRICS)
    parser.add_argument("--complete-qap", type=Path, default=DEFAULT_COMPLETE_QAP)
    parser.add_argument("--support-sweep", type=Path, default=DEFAULT_SUPPORT_SWEEP)
    parser.add_argument("--permutations", type=int, default=audit.DEFAULT_QAP_PERMUTATIONS)
    parser.add_argument("--seed", type=int, default=audit.DEFAULT_SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = reproduce(
        target_pairs=args.target_pairs,
        exclusions=args.exclusions,
        docking=args.docking,
        permutations=args.permutations,
        seed=args.seed,
    )
    verify(result, args.expected)
    complete_metrics, complete_qap = reproduce_complete_case(
        target_pairs=args.target_pairs,
        exclusions=args.exclusions,
        docking=args.docking,
        permutations=args.permutations,
        seed=args.seed,
    )
    verify_table(complete_metrics, args.complete_metrics, ("predictor",))
    verify_table(complete_qap, args.complete_qap, KEYS)
    support_sweep = reproduce_support_sweep(args.target_pairs)
    verify_table(
        support_sweep,
        args.support_sweep,
        ("minimum_pair_support",),
    )
    print(
        f"verified {len(result)} primary and {len(complete_qap)} complete-case "
        f"paired-QAP rows plus support sensitivities from released target-pair "
        f"aggregates ({args.permutations} permutations per scheme)"
    )


if __name__ == "__main__":
    main()
