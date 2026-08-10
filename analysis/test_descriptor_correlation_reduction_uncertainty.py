"""Contract tests for repeated-partition/support descriptor uncertainty."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


PACKAGE = Path(__file__).resolve().parents[1]
RESULTS = PACKAGE / "results" / "descriptor_correlation_reduction_uncertainty"
METRIC = "relative_mean_squared_offdiagonal_target_correlation_reduction"


def load_summary() -> dict:
    return json.loads((RESULTS / "summary.json").read_text())


def test_reference_points_exact() -> None:
    headline = load_summary()["headline_reference_reproduction"]
    assert np.isclose(headline["Docking-44"], 0.4687472762980258, atol=1e-12)
    assert np.isclose(
        headline["DOCKSTRING-58_seed_71"], 0.5360024559043216, atol=1e-12
    )


def test_multiple_score_blind_supports_and_repeated_group_partitions() -> None:
    summary = load_summary()
    seeds = summary["dockstring_pool"]["score_blind_support_seeds"]
    assert len(seeds) >= 5
    assert 71 in seeds
    repeats = summary["configuration"]["repeated_partitions_per_support"]
    assert repeats >= 10
    frame = pd.read_csv(RESULTS / "repeated_partition_metrics.csv")
    dockstring = frame.loc[frame.dataset.eq("DOCKSTRING-58")]
    assert dockstring.support_seed.nunique() == len(seeds)
    assert len(dockstring) == len(seeds) * (repeats + 1)
    assert frame[METRIC].between(-1.0, 1.0).all()
    assert (frame.chemical_groups >= 5).all()


def test_empirical_intervals_are_named_as_sensitivity_not_confidence() -> None:
    summary = load_summary()
    assert "not_an_explained_fraction" in summary
    assert "not calibrated confidence intervals" in summary["claim_boundary"]
    for dataset in ("Docking-44", "DOCKSTRING-58"):
        record = summary["sensitivity"][dataset]
        interval = record["all_repeated_group_partitions"]["empirical_interval_95"]
        assert interval[0] < interval[1]
        if dataset == "DOCKSTRING-58":
            support_interval = record["between_score_blind_support_means"][
                "empirical_interval_95"
            ]
            assert support_interval[0] < support_interval[1]


def test_every_repeated_partition_is_score_blind_and_group_held_out() -> None:
    summary = load_summary()
    assert "scores are not inspected" in summary["dockstring_pool"]["selection_rule"]
    assert summary["configuration"][
        "every_learned_quantity_refit_inside_training_fold"
    ] is True
    frame = pd.read_csv(RESULTS / "repeated_partition_metrics.csv")
    repeated = frame.loc[~frame.partition.eq("deterministic_GroupKFold")]
    assert repeated.partition_seed.notna().all()
    assert (repeated.minimum_fold_ligands > 0).all()
    assert (repeated.maximum_fold_ligands >= repeated.minimum_fold_ligands).all()


def test_support_overlap_table_is_complete_and_bounded() -> None:
    summary = load_summary()
    seeds = summary["dockstring_pool"]["score_blind_support_seeds"]
    frame = pd.read_csv(RESULTS / "dockstring_support_overlap.csv")
    assert len(frame) == len(seeds) * (len(seeds) - 1) // 2
    assert frame.jaccard.between(0.0, 1.0).all()
    assert (frame.shared_ligands > 0).all()


def test_central_estimand_has_paired_conditional_block_bootstrap() -> None:
    summary = load_summary()
    bootstrap = summary["conditional_paired_chemical_group_block_bootstrap"]
    assert "same draw" in bootstrap["method"]
    assert "not refitted" in bootstrap["conditioning"]
    assert "coarsened" in bootstrap["coarsening_limitation"]
    records = bootstrap["per_dataset_support"]
    assert "Docking-44" in records
    assert "DOCKSTRING-58_seed_71" in records
    assert len(records) >= 6
    for record in records.values():
        low, high = record["conditional_coarsened_sensitivity_interval_95"]
        assert low < record["point"] < high
        assert record["repeats"] == 5_000
        assert record["blocks"] <= 512

    frame = pd.read_csv(RESULTS / "conditional_block_bootstrap.csv")
    assert len(frame) == len(records)
    assert (
        frame.conditional_coarsened_sensitivity_low95
        < frame.conditional_coarsened_sensitivity_high95
    ).all()
