from __future__ import annotations

import json
from pathlib import Path

import davis_fixed20_censoring_sensitivity as analysis


PACKAGE = Path(__file__).resolve().parents[1]


def test_fixed20_measurement_boundary_and_frozen_sensitivity() -> None:
    matrix = analysis.load_davis(analysis.DEFAULT_DAVIS)
    assert matrix.shape == (72, 20)
    assert int(matrix.eq(5.0).to_numpy().sum()) == 965
    assert int((matrix.eq(5.0).mean(axis=0) > 0.90).sum()) == 4
    assert set(matrix.columns[matrix.eq(5.0).mean(axis=0) > 0.90]) == {
        "AKT1",
        "AKT2",
        "MAPK1",
        "MAPKAPK2",
    }

    pair_frame = analysis.ordered_pair_frame(analysis.DEFAULT_TARGET_PAIRS)
    assert len(pair_frame) == 190

    summary = json.loads(
        (
            PACKAGE
            / "results/davis_fixed20_censoring_sensitivity/summary.json"
        ).read_text()
    )
    assert round(summary["continuous_vs_above_floor_status_geometry_spearman"], 3) == 0.885
    assert (
        summary["alternative_endpoint"]["shared_positive_pairs"] == 13
    )
    validation = summary["alternative_endpoint"][
        "KiRHub_centered_geometry_validation"
    ]
    assert round(validation["roc_auc"], 3) == 0.956
    assert round(validation["average_precision"], 3) == 0.673

    explicit = summary["exclude_targets_over_90_percent_floor_sensitivity"]
    assert explicit["targets_retained"] == 16
    assert explicit["removed_target_names"] == [
        "AKT1",
        "AKT2",
        "MAPK1",
        "MAPKAPK2",
    ]
    assert round(
        explicit["centered_docking_vs_continuous_davis"]["spearman"], 3
    ) == 0.210
    assert round(
        explicit["centered_docking_vs_above_floor_status_davis"]["spearman"],
        3,
    ) == 0.197
    target_table = Path(
        PACKAGE / "results/davis_fixed20_censoring_sensitivity/target_censoring.csv"
    )
    assert target_table.is_file()
