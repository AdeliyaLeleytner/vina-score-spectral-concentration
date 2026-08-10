import json

import numpy as np
import pytest

import centering_panel_sensitivity as sensitivity


def test_row_and_two_way_centering_have_identical_target_correlations() -> None:
    rng = np.random.default_rng(17)
    matrix = rng.normal(size=(250, 9))
    indices = (1, 3, 4, 7)
    for all_columns in (False, True):
        correlation, discrepancy = sensitivity.centered_geometry(
            matrix, indices, center_over_all_columns=all_columns
        )
        assert correlation.shape == (4, 4)
        assert discrepancy < 1e-12


def test_centering_panel_can_change_extracted_geometry() -> None:
    rng = np.random.default_rng(23)
    latent = rng.normal(size=(400, 1))
    selected = rng.normal(size=(400, 3))
    other = latent @ np.array([[2.0, -1.0, 0.5]]) + rng.normal(
        scale=0.2, size=(400, 3)
    )
    matrix = np.column_stack([selected, other])
    kinase, _ = sensitivity.centered_geometry(
        matrix, (0, 1, 2), center_over_all_columns=False
    )
    full, _ = sensitivity.centered_geometry(
        matrix, (0, 1, 2), center_over_all_columns=True
    )
    assert not np.allclose(kinase, full)


def test_locked_endpoint_order_and_count() -> None:
    labels = sensitivity.load_locked_labels(sensitivity.DEFAULT_LOCKED_PAIRS)
    assert labels.shape == (190,)
    assert int(labels.sum()) == 15


def test_frozen_output_contract_if_present() -> None:
    path = sensitivity.DEFAULT_OUTPUT / "summary.json"
    if not path.exists():
        pytest.skip("centering-panel sensitivity has not been generated")
    with path.open() as handle:
        summary = json.load(handle)
    identity = summary["row_only_vs_two_way_correlation_identity"]
    assert identity["all_below_1e_minus_12"] is True
    assert max(identity["maximum_absolute_differences"].values()) < 1e-12
    locked = summary["locked_15_pair_retrieval"]
    assert "15 of 190" in locked["endpoint"]
    assert [row["positive_pairs"] for row in locked["point_estimates"]] == [15, 15, 15]
    agreement = summary["docking_geometry_agreement_between_centering_panels"]
    assert agreement["20_target_block"][
        "spearman_of_target_pair_correlations"
    ] > 0.95
    matched = summary["projection_matched_panel_delta_null"]
    assert set(matched["panels"]) == {"DAVIS", "PKIS2", "PKIS1", "KiRHub"}
    assert set(matched["locked_15_pair_retrieval"]) == {
        "roc_auc",
        "average_precision",
    }
