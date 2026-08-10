from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from matched_support_geometry import (
    DEFAULT_OUTPUT,
    MATCHING_DESCRIPTORS,
    OUTPUT_FILES,
    SENSITIVITY_TARGETS,
    TARGETS,
    attenuation_correction,
    paired_support_bootstrap,
    paired_support_qap,
    propensity_matched_positions,
    split_half_reliability,
    standardized_mean_differences,
    union_interval,
)
from residual_target_geometry_validation import (
    geometry_concordance,
    geometry_correlation,
)


# ---------------------------------------------------------------------------
# Unit behaviour
# ---------------------------------------------------------------------------


def test_paired_support_qap_recovers_the_support_contrast() -> None:
    rng = np.random.default_rng(3)
    experiment = rng.normal(size=(80, 8))
    endpoint = geometry_correlation(experiment, "center_then_correlation")
    broad = geometry_correlation(rng.normal(size=(4_000, 8)), "center_then_correlation")
    matched = geometry_correlation(rng.normal(size=(60, 8)), "center_then_correlation")
    result = paired_support_qap(broad, matched, endpoint, 99, 11)
    assert result["matched_support_concordance"] == pytest.approx(
        geometry_concordance(matched, endpoint), abs=1e-12
    )
    assert result["broad_support_concordance"] == pytest.approx(
        geometry_concordance(broad, endpoint), abs=1e-12
    )
    assert result["matched_minus_broad_support"] == pytest.approx(
        result["matched_support_concordance"] - result["broad_support_concordance"],
        abs=1e-12,
    )
    assert 0.0 < result["one_sided_p_positive_matched_minus_broad"] <= 1.0


def test_paired_bootstrap_is_seed_deterministic_and_reports_its_displacement() -> None:
    rng = np.random.default_rng(7)
    docking = rng.normal(size=(40, 6))
    experiment = docking + rng.normal(scale=2.0, size=(40, 6))
    broad = geometry_correlation(rng.normal(size=(3_000, 6)), "center_then_correlation")
    endpoint = geometry_correlation(experiment, "center_then_correlation")
    matched_geometry = geometry_correlation(docking, "center_then_correlation")
    points = {
        "matched_support_spearman": geometry_concordance(matched_geometry, endpoint),
        "broad_support_spearman": geometry_concordance(broad, endpoint),
    }
    points["matched_minus_broad_support"] = (
        points["matched_support_spearman"] - points["broad_support_spearman"]
    )
    first = paired_support_bootstrap(docking, experiment, broad, 64, 5, None, "ligand", points)
    second = paired_support_bootstrap(docking, experiment, broad, 64, 5, None, "ligand", points)
    assert first == second
    for name, value in points.items():
        assert first["point_estimate"][name] == pytest.approx(value, abs=1e-12)
        assert first["resampled_median_minus_point_estimate"][name] == pytest.approx(
            first[name]["median"] - value, abs=1e-12
        )
    lower, upper = first["matched_minus_broad_support"]["interval_95"]
    assert lower <= first["matched_minus_broad_support"]["median"] <= upper
    with pytest.raises(ValueError):
        paired_support_bootstrap(
            docking, experiment, broad, 4, 5, None, "ligand", {"wrong_key": 0.0}
        )


def test_split_half_reliability_is_high_for_a_reproducible_geometry() -> None:
    rng = np.random.default_rng(19)
    loadings = rng.normal(size=(3, 7))
    scores = rng.normal(size=(600, 3))
    matrix = scores @ loadings + 0.05 * rng.normal(size=(600, 7))
    strong = split_half_reliability(matrix, 24, 2)
    weak = split_half_reliability(rng.normal(size=(600, 7)), 24, 2)
    assert strong["spearman_brown_reliability"]["median"] > 0.9
    assert weak["spearman_brown_reliability"]["median"] < strong[
        "spearman_brown_reliability"
    ]["median"]
    assert strong["half_size"] == 300


def test_propensity_matching_removes_the_descriptor_imbalance() -> None:
    rng = np.random.default_rng(23)
    pool = rng.normal(size=(4_000, len(MATCHING_DESCRIPTORS)))
    anchors = rng.normal(loc=1.5, scale=0.3, size=(40, len(MATCHING_DESCRIPTORS)))
    selections, _, scale = propensity_matched_positions(pool, anchors, (5, 10))
    before = standardized_mean_differences(pool, anchors, scale)
    after = standardized_mean_differences(pool[selections[10]], anchors, scale)
    assert max(abs(value) for value in after.values()) < 0.1 * max(
        abs(value) for value in before.values()
    )
    assert len(selections[10]) == 400
    assert set(selections[5]).issubset(set(selections[10]))


def test_union_interval_and_attenuation_correction_are_conservative() -> None:
    assert union_interval([-0.1, 0.2], [-0.3, 0.15]) == [-0.3, 0.2]
    corrected = attenuation_correction(0.3, 0.36, 0.81)
    assert corrected["attenuation_factor_sqrt_reliability_product"] == pytest.approx(0.54)
    assert corrected["disattenuated_spearman"] == pytest.approx(0.3 / 0.54)
    assert corrected["disattenuated_spearman"] >= 0.3
    assert attenuation_correction(0.3, 0.0, 0.5)["status"].startswith("not_defined")


# ---------------------------------------------------------------------------
# Frozen artifact contract
# ---------------------------------------------------------------------------


def test_released_summary_reproduces_the_published_cross_support_values() -> None:
    """The broad-vs-full-panel arm must land on the manuscript's published number."""
    summary = json.loads((DEFAULT_OUTPUT / "summary.json").read_text())
    assert summary["analysis"] == "matched_support_geometry"
    assert summary["support"]["targets"] == len(TARGETS) == 20
    assert summary["support"]["target_pairs"] == 190
    assert summary["support"]["sensitivity_targets"] == len(SENSITIVITY_TARGETS) == 21
    assert summary["configuration"]["bootstrap_replicates"] == 5_000
    assert summary["configuration"]["target_label_qap_permutations"] == 49_999
    assert set(summary["support"]["panels"]) == {"DAVIS", "PKIS2"}
    assert summary["support"]["panels"]["DAVIS"]["matched_ligands"] == 59
    assert summary["support"]["panels"]["PKIS2"]["matched_ligands"] == 154
    assert "claim_boundary" in summary and summary["claim_boundary"].strip()
    for name, record in summary["panels"].items():
        assert record["frozen_mapping_check"]["recomputed_match_equals_frozen_mapping"]
        # Recomputing the manuscript's estimator on this de-leaked reference must
        # land on the published fixed-20 value to within 1e-3 Spearman.
        assert record["recomputed_broad_support_versus_full_panel_endpoint"] == pytest.approx(
            record["published_broad_support_versus_full_panel_endpoint"], abs=1e-3
        ), name
        assert record["matched_minus_broad_support"] == pytest.approx(
            record["matched_support_spearman"] - record["broad_support_spearman"],
            abs=1e-12,
        )


def test_released_contrast_is_unresolved_and_intervals_are_wide() -> None:
    """The reviewer's number: matched support neither beats nor loses to broad."""
    summary = json.loads((DEFAULT_OUTPUT / "summary.json").read_text())
    bootstrap = pd.read_csv(DEFAULT_OUTPUT / "support_contrast_bootstrap.csv")
    contrast = bootstrap[bootstrap.statistic == "matched_minus_broad_support"]
    assert len(contrast) == 2 * 3  # two panels x three resampling units
    assert (contrast.repeats == 5_000).all()
    # Every conservative chemical-cluster interval must straddle zero, and each
    # must be wide enough that a difference the size of the manuscript's own
    # low/high-molecular-weight contrast could not have been detected here.
    for name, record in summary["panels"].items():
        lower, upper = record["conservative_union_interval_95_murcko_and_butina"][
            "matched_minus_broad_support"
        ]
        assert lower < 0.0 < upper, name
        assert upper - lower > 0.15, name
        assert abs(record["matched_minus_broad_support"]) < 0.05, name
        qap = record["paired_target_label_qap_matched_minus_broad"]
        assert qap["permutations"] == 49_999
        assert 0.0 < qap["one_sided_p_positive_matched_minus_broad"] <= 1.0
    assert not any(
        summary["headline"]["resolved_by_conservative_cluster_bootstrap_union"].values()
    )
    assert summary["headline"]["verdict"].startswith("indistinguishable")


def test_released_propensity_matching_is_balanced_and_size_controlled() -> None:
    frame = pd.read_csv(DEFAULT_OUTPUT / "propensity_matching.csv")
    assert set(frame.descriptor) == set(MATCHING_DESCRIPTORS)
    assert frame.standardized_mean_difference_before.abs().max() > 0.1
    assert frame.standardized_mean_difference_after.abs().max() < 0.02
    summary = json.loads((DEFAULT_OUTPUT / "summary.json").read_text())
    for name, record in summary["panels"].items():
        propensity = record["propensity_matched_support"]
        primary = propensity["neighbour_count_sensitivity"][
            str(propensity["primary_neighbours_per_matched_ligand"])
        ]
        control = propensity["size_matched_random_support_control"]
        assert primary["support_rows_with_reuse"] == control["support_size_with_reuse"]
        assert (
            primary["maximum_absolute_standardized_mean_difference_after"]
            < primary["maximum_absolute_standardized_mean_difference_before"]
        ), name


def test_released_outputs_are_registered_and_checksummed() -> None:
    checksums = json.loads((DEFAULT_OUTPUT / "output_checksums.json").read_text())
    assert set(checksums) == set(OUTPUT_FILES)
    for name, digest in checksums.items():
        assert (DEFAULT_OUTPUT / name).exists(), name
        assert len(digest) == 64
    summary_text = (DEFAULT_OUTPUT / "summary.json").read_text()
    # Determinism guards: no wall-clock fields and no absolute paths.
    assert "runtime" not in summary_text
    assert "timestamp" not in summary_text
    assert "/Users/" not in summary_text
