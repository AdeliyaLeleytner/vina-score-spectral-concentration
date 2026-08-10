from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from fixed20_estimand_decomposition import (
    DEFAULT_OUTPUT,
    RAW_EXPERIMENT_QAP_SEED_OFFSET,
    decompose,
    paired_docking_geometries,
    transformation_matched_independent_column_null,
)


def test_fixed_endpoint_decomposition_identity() -> None:
    rng = np.random.default_rng(4)
    docking = rng.normal(size=(500, 6))
    experiment = rng.normal(size=(80, 6))
    raw = np.corrcoef(docking, rowvar=False)
    centered = np.corrcoef(
        docking - docking.mean(axis=1, keepdims=True), rowvar=False
    )
    result = decompose(raw, centered, experiment, permutations=19, seed=8)
    total = result["matched_estimand_total_change"]
    pieces = (
        result["experimental_estimand_increment_with_raw_docking"]
        + result["docking_transform_increment_with_centered_experiment"]
    )
    assert abs(total - pieces) < 1e-12
    reverse_pieces = (
        result["docking_transform_increment_with_raw_experiment"]
        + result["experimental_estimand_increment_with_centered_docking"]
    )
    assert abs(total - reverse_pieces) < 1e-12
    shapley = result["path_averaged_descriptive_attribution"]
    assert abs(total - shapley["experimental_estimand"] - shapley["docking_transform"]) < 1e-12
    assert abs(
        result["two_factor_interaction_on_spearman_scale"]
        - result["docking_transform_increment_with_centered_experiment"]
        + result["docking_transform_increment_with_raw_experiment"]
    ) < 1e-12
    centered_qap = result["paired_qap_for_docking_increment"]
    raw_qap = result["paired_qap_for_docking_increment_with_raw_experiment"]
    assert centered_qap["centered_minus_raw_docking"] == pytest.approx(
        result["docking_transform_increment_with_centered_experiment"]
    )
    assert raw_qap["centered_minus_raw_docking"] == pytest.approx(
        result["docking_transform_increment_with_raw_experiment"]
    )
    assert centered_qap["fixed_experimental_endpoint"] == (
        "two-way-centered experimental target geometry"
    )
    assert raw_qap["fixed_experimental_endpoint"] == "raw experimental target geometry"
    assert centered_qap["seed"] == 8
    assert raw_qap["seed"] == 8 + RAW_EXPERIMENT_QAP_SEED_OFFSET
    assert raw_qap["seed"] != centered_qap["seed"]
    assert len(raw_qap["null_interval_95"]) == 2
    assert raw_qap["null_interval_95"][0] <= raw_qap["null_interval_95"][1]


def test_no_change_gives_zero_docking_increment() -> None:
    rng = np.random.default_rng(9)
    predictor = np.corrcoef(rng.normal(size=(100, 5)), rowvar=False)
    experiment = rng.normal(size=(60, 5))
    result = decompose(predictor, predictor, experiment, permutations=9, seed=2)
    assert abs(result["docking_transform_increment_with_centered_experiment"]) < 1e-12
    assert abs(result["docking_transform_increment_with_raw_experiment"]) < 1e-12
    assert (
        result["paired_qap_for_docking_increment_with_raw_experiment"]
        ["centered_minus_raw_docking"]
        == 0.0
    )


def test_covariance_projection_matches_materialized_two_way_centering() -> None:
    rng = np.random.default_rng(12)
    matrix = rng.normal(size=(300, 20))
    raw, centered = paired_docking_geometries(matrix)
    expected_raw = np.corrcoef(matrix, rowvar=False)
    expected_centered = np.corrcoef(
        matrix - matrix.mean(axis=1, keepdims=True), rowvar=False
    )
    assert np.allclose(raw, expected_raw, atol=2e-15, rtol=0)
    assert np.allclose(centered, expected_centered, atol=2e-15, rtol=0)


def test_transformation_matched_null_is_paired_and_seed_deterministic() -> None:
    rng = np.random.default_rng(14)
    reference = rng.normal(size=(240, 20))
    experiments = {
        "first": rng.normal(size=(60, 20)),
        "second": rng.normal(size=(75, 20)),
    }
    first_summary, first_frame = transformation_matched_independent_column_null(
        reference,
        experiments,
        reference_size=120,
        repeats=7,
        support_seed=21,
        permutation_seed=22,
    )
    second_summary, second_frame = transformation_matched_independent_column_null(
        reference,
        experiments,
        reference_size=120,
        repeats=7,
        support_seed=21,
        permutation_seed=22,
    )
    assert first_summary == second_summary
    assert first_frame.equals(second_frame)
    assert len(first_frame) == 7 * 2 * 2
    assert (
        first_frame.centered_minus_raw_docking
        == first_frame.centered_docking_concordance
        - first_frame.raw_docking_concordance
    ).all()
    assert first_frame.groupby(
        ["panel", "fixed_experimental_endpoint"]
    ).size().eq(7).all()
    for endpoint_records in first_summary["panels"].values():
        assert set(endpoint_records) == {
            "raw_experimental_target_geometry",
            "two_way_centered_experimental_target_geometry",
        }
        for record in endpoint_records.values():
            assert 0 < record["p_observed_gain_at_least_as_large"] <= 1
            assert len(record["null_centered_minus_raw_interval_95"]) == 2


def test_frozen_raw_experiment_qap_contract() -> None:
    summary_path = DEFAULT_OUTPUT / "summary.json"
    table_path = DEFAULT_OUTPUT / "decomposition.csv"
    assert summary_path.exists()
    assert table_path.exists()
    summary = json.loads(summary_path.read_text())
    table = pd.read_csv(table_path).set_index("panel")
    expected = {
        "DAVIS": (0.01512, -0.15221036885407235, 0.1528050492723956),
        "PKIS2": (0.00152, -0.1702362363150459, 0.1712159106317276),
        "PKIS1": (0.00536, -0.17460055024122262, 0.17320780135329583),
        "KiRHub": (0.00008, -0.14168475289445254, 0.1472671206812842),
    }
    p_column = "paired_qap_docking_increment_with_raw_experiment_p_positive"
    lower_column = (
        "paired_qap_docking_increment_with_raw_experiment_null_lower_95"
    )
    upper_column = (
        "paired_qap_docking_increment_with_raw_experiment_null_upper_95"
    )
    for panel_index, (panel, (expected_p, expected_lower, expected_upper)) in enumerate(
        expected.items()
    ):
        result = summary["panels"][panel]
        qap = result["paired_qap_for_docking_increment_with_raw_experiment"]
        assert qap["fixed_experimental_endpoint"] == "raw experimental target geometry"
        assert qap["centered_minus_raw_docking"] == pytest.approx(
            result["docking_transform_increment_with_raw_experiment"]
        )
        assert qap["permutations"] == 49_999
        assert qap["seed"] == (
            summary["configuration"]["seed"]
            + panel_index * 1_000_000
            + RAW_EXPERIMENT_QAP_SEED_OFFSET
        )
        assert qap["one_sided_p_positive_delta"] == expected_p
        assert qap["null_interval_95"] == pytest.approx(
            [expected_lower, expected_upper], abs=1e-15
        )
        assert table.loc[panel, p_column] == expected_p
        assert table.loc[panel, lower_column] == pytest.approx(expected_lower)
        assert table.loc[panel, upper_column] == pytest.approx(expected_upper)


def test_frozen_transformation_matched_null_contract() -> None:
    summary_path = DEFAULT_OUTPUT / "summary.json"
    null_path = DEFAULT_OUTPUT / "transformation_matched_null.csv"
    assert summary_path.exists()
    assert null_path.exists()
    summary = json.loads(summary_path.read_text())
    matched = summary["transformation_matched_independent_column_null"]
    frame = pd.read_csv(null_path)
    assert matched["reference_ligands_full"] == 259_579
    assert matched["reference_ligands_fixed_subsample"] == 15_000
    assert matched["repeats"] == 2_000
    assert len(frame) == 2_000 * 4 * 2
    assert set(frame.panel) == {"DAVIS", "PKIS2", "PKIS1", "KiRHub"}
    assert set(frame.fixed_experimental_endpoint) == {
        "raw_experimental_target_geometry",
        "two_way_centered_experimental_target_geometry",
    }
    expected = {
        "DAVIS": {
            "raw_experimental_target_geometry": (
                0.16838256198961632,
                0.16249349376932737,
                0.022579362017609468,
                0.0029985007496251873,
            ),
            "two_way_centered_experimental_target_geometry": (
                0.016391764752194615,
                0.010245509060609614,
                0.02607027516435067,
                0.6901549225387307,
            ),
        },
        "PKIS2": {
            "raw_experimental_target_geometry": (
                0.2485708162203065,
                0.24284620801567625,
                -0.06668872005493666,
                0.0004997501249375312,
            ),
            "two_way_centered_experimental_target_geometry": (
                0.04367638117982567,
                0.03410095920429354,
                0.005010559557707532,
                0.17191404297851073,
            ),
        },
        "PKIS1": {
            "raw_experimental_target_geometry": (
                0.21795326011363492,
                0.21509620473522378,
                -0.019963955263376597,
                0.0004997501249375312,
            ),
            "two_way_centered_experimental_target_geometry": (
                0.0008607907202561604,
                -0.005848828003691631,
                -0.004706463366094119,
                0.512743628185907,
            ),
        },
        "KiRHub": {
            "raw_experimental_target_geometry": (
                0.2644447068806397,
                0.2609980448503458,
                0.02594360333643883,
                0.0004997501249375312,
            ),
            "two_way_centered_experimental_target_geometry": (
                0.045476693478897934,
                0.041052019227825254,
                0.0015894973034680949,
                0.07296351824087956,
            ),
        },
    }
    for panel, endpoint_records in expected.items():
        for endpoint, (
            full_delta,
            support_delta,
            null_delta_mean,
            expected_p,
        ) in endpoint_records.items():
            record = matched["panels"][panel][endpoint]
            assert record[
                "full_reference_observed_centered_minus_raw_docking"
            ] == pytest.approx(full_delta, abs=1e-15)
            assert record[
                "matched_support_observed_centered_minus_raw_docking"
            ] == pytest.approx(support_delta, abs=1e-15)
            assert record["null_centered_minus_raw_mean"] == pytest.approx(
                null_delta_mean, abs=1e-15
            )
            assert record["p_observed_gain_at_least_as_large"] == expected_p
