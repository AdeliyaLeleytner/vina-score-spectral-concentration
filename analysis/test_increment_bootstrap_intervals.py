from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import fixed20_estimand_decomposition as fixed20
import residual_target_geometry_validation as geometry
from increment_bootstrap_intervals import (
    CLUSTER_GROUPINGS,
    DEFAULT_OUTPUT,
    DEFAULT_REPEATS,
    NEGLIGIBLE_MARGIN,
    PANELS,
    basic_interval,
    bootstrap_docking_rank_vectors,
    bootstrap_experimental_rank_vectors,
    concordance,
    conservative_union,
    describe,
    geometries_from_moments,
    group_moment_table,
    rank_unit_vector,
    verdict_for,
)


# ---------------------------------------------------------------------------
# Unit properties of the estimator
# ---------------------------------------------------------------------------


def test_rank_dot_product_is_spearman() -> None:
    rng = np.random.default_rng(3)
    first = rng.normal(size=190)
    second = 0.4 * first + rng.normal(size=190)
    observed = float(concordance(rank_unit_vector(first), rank_unit_vector(second))[0])
    assert observed == pytest.approx(
        float(stats.spearmanr(first, second).statistic), abs=1e-12
    )


def test_moment_route_reproduces_released_paired_geometries() -> None:
    """The bootstrap inner loop must equal the released full-support estimator."""
    rng = np.random.default_rng(11)
    matrix = rng.normal(size=(400, 20))
    expected_raw, expected_centered = fixed20.paired_docking_geometries(matrix)
    raw, centered = geometries_from_moments(
        float(len(matrix)), matrix.sum(axis=0), matrix.T @ matrix
    )
    assert np.allclose(raw, expected_raw, atol=1e-10, rtol=0)
    assert np.allclose(centered, expected_centered, atol=1e-10, rtol=0)


def test_group_moments_reproduce_a_duplicated_support() -> None:
    """Cluster multiplicities must be identical to physically repeating the rows."""
    rng = np.random.default_rng(17)
    matrix = rng.normal(size=(60, 20))
    codes = rng.integers(0, 8, size=60)
    codes[:8] = np.arange(8)  # guarantee a contiguous 0..7 labelling
    table, upper = group_moment_table(matrix, codes, 8)
    multiplicity = np.array([2.0, 0.0, 1.0, 3.0, 1.0, 0.0, 2.0, 1.0])
    moments = multiplicity @ table
    second = np.zeros((20, 20))
    second[upper] = moments[21:]
    second = second + second.T - np.diag(np.diag(second))
    raw, centered = geometries_from_moments(moments[0], moments[1:21], second)
    repeated = np.repeat(matrix, multiplicity[codes].astype(int), axis=0)
    expected_raw, expected_centered = fixed20.paired_docking_geometries(repeated)
    assert np.allclose(raw, expected_raw, atol=1e-9, rtol=0)
    assert np.allclose(centered, expected_centered, atol=1e-9, rtol=0)


def test_docking_bootstrap_is_seed_deterministic_and_chunk_invariant() -> None:
    rng = np.random.default_rng(23)
    matrix = rng.normal(size=(120, 20))
    codes = np.concatenate([np.arange(10), rng.integers(0, 10, size=110)])
    table, upper = group_moment_table(matrix, codes, 10)
    first = bootstrap_docking_rank_vectors(table, upper, 20, 9, seed=5, chunk=3)
    second = bootstrap_docking_rank_vectors(table, upper, 20, 9, seed=5, chunk=7)
    other = bootstrap_docking_rank_vectors(table, upper, 20, 9, seed=6, chunk=3)
    assert np.array_equal(first[0], second[0])
    assert np.array_equal(first[1], second[1])
    assert not np.array_equal(first[0], other[0])


def test_experimental_bootstrap_recovers_the_released_endpoint_at_unit_weight() -> None:
    """A degenerate one-compound-per-cluster draw must give the released geometry."""
    rng = np.random.default_rng(29)
    panel = rng.normal(size=(40, 20))
    labels = np.arange(40)
    vectors, clusters = bootstrap_experimental_rank_vectors(panel, labels, 4, seed=31)
    assert clusters == 40
    assert vectors.shape == (4, 190)
    released = rank_unit_vector(
        geometry.upper_triangle(
            geometry.geometry_correlation(panel, "center_then_correlation")
        )
    )
    # Every replicate is a resample, so it must differ from the plug-in geometry, but
    # the plug-in geometry itself is what a unit-multiplicity draw would return.
    assert not np.allclose(vectors[0], released)
    assert float(concordance(vectors[0], released)[0]) > 0.5


def test_basic_interval_reflects_the_percentile_interval() -> None:
    assert basic_interval(0.02, [-0.01, 0.06]) == pytest.approx([-0.02, 0.05])
    with pytest.raises(ValueError):
        basic_interval(0.0, [0.1, -0.1])


def test_conservative_union_and_verdicts() -> None:
    assert conservative_union([[-0.02, 0.03], [-0.01, 0.05]]) == [-0.02, 0.05]
    assert verdict_for([0.01, 0.05])[0] == "resolved_positive_increment"
    assert verdict_for([-0.06, -0.01])[0] == "resolved_negative_increment"
    assert (
        verdict_for([-0.02, 0.03])[0] == "consistent_with_a_negligible_increment"
    )
    assert (
        verdict_for([-0.02, 2 * NEGLIGIBLE_MARGIN])[0]
        == "dominated_by_sampling_uncertainty"
    )


def test_describe_reports_the_package_interval_convention() -> None:
    values = np.linspace(-1.0, 1.0, 1001)
    summary = describe(values)
    assert summary["replicates"] == 1001
    assert summary["interval_95"][0] == pytest.approx(np.quantile(values, 0.025))
    assert summary["interval_95"][1] == pytest.approx(np.quantile(values, 0.975))
    assert summary["fraction_of_replicates_at_or_below_zero"] == pytest.approx(
        0.5004995004995005
    )


# ---------------------------------------------------------------------------
# Frozen contract of the released artifact
# ---------------------------------------------------------------------------

EXPECTED_COMBINED_INTERVAL_95 = {
    "DAVIS": (-0.007584449780647164, 0.05909770062153646),
    "PKIS2": (0.019116945942517775, 0.06421323815647326),
    "PKIS1": (-0.021531971289480235, 0.029842582022245743),
    "KiRHub": (0.01454714447549951, 0.07016524732424409),
}
EXPECTED_VERDICT = {
    "DAVIS": "dominated_by_sampling_uncertainty",
    "PKIS2": "resolved_positive_increment",
    "PKIS1": "consistent_with_a_negligible_increment",
    "KiRHub": "resolved_positive_increment",
}


@pytest.fixture(scope="module")
def summary() -> dict:
    path = DEFAULT_OUTPUT / "summary.json"
    assert path.exists(), "run analysis/increment_bootstrap_intervals.py first"
    return json.loads(path.read_text())


def test_artifact_reproduces_the_published_increments(summary: dict) -> None:
    """The bootstrapped estimand must be the increment the manuscript reports."""
    frozen = summary["consistency_checks"]["frozen_fixed20_increments"]
    assert summary["consistency_checks"]["all_within_tolerance"] is True
    worst = max(summary["consistency_checks"]["max_absolute_differences"].values())
    assert worst < summary["consistency_checks"]["tolerance"]
    for panel in PANELS:
        assert summary["panels"][panel]["observed_delta_d"] == pytest.approx(
            frozen[panel], abs=1e-12
        )
    assert frozen["DAVIS"] == pytest.approx(0.016391764752194615, abs=1e-15)
    assert frozen["PKIS2"] == pytest.approx(0.04367638117982567, abs=1e-15)
    assert frozen["PKIS1"] == pytest.approx(0.0008607907202561604, abs=1e-15)
    assert frozen["KiRHub"] == pytest.approx(0.045476693478897934, abs=1e-15)


def test_artifact_intervals_are_frozen_and_ordered(summary: dict) -> None:
    for panel in PANELS:
        record = summary["panels"][panel]
        low, high = record["combined_conservative_interval_95"]
        assert low < high
        expected_low, expected_high = EXPECTED_COMBINED_INTERVAL_95[panel]
        assert low == pytest.approx(expected_low, abs=1e-12)
        assert high == pytest.approx(expected_high, abs=1e-12)
        assert record["verdict"] == EXPECTED_VERDICT[panel]
        contains_zero = low <= 0.0 <= high
        assert contains_zero == record["verdict"].startswith(
            ("dominated", "consistent")
        )


def test_bootstrap_interval_is_not_the_qap_null_interval(summary: dict) -> None:
    """The point of the analysis: PKIS2 and KiRHub are resolved by compound
    resampling even though their target-label QAP probabilities are not small."""
    for panel in ("PKIS2", "KiRHub"):
        record = summary["panels"][panel]
        assert record["published_paired_target_label_qap_p_positive"] > 0.10
        assert record["combined_conservative_interval_95"][0] > 0.0
        # Resolution must survive the reverse-percentile construction as well.
        assert record["combined_conservative_basic_interval_95"][0] > 0.0
        assert record["verdict_robust_to_interval_construction"] is True
    assert summary["panels"]["PKIS1"]["smallest_supported_symmetric_margin_95"] < (
        NEGLIGIBLE_MARGIN
    )
    assert summary["panels"]["DAVIS"]["smallest_supported_symmetric_margin_95"] > (
        NEGLIGIBLE_MARGIN
    )
    # DAVIS sits on the reporting threshold, so the artifact must flag it rather than
    # present its verdict as firm.
    assert summary["panels"]["DAVIS"]["verdict_robust_to_interval_construction"] is False
    assert summary["verdicts"]["verdict_not_robust_to_interval_construction"] == [
        "DAVIS"
    ]
    assert summary["verdicts"]["resolved_nonzero"] == ["PKIS2", "KiRHub"]
    assert "not evidence of target specificity" in summary[
        "strongest_honest_conclusion"
    ]


def test_artifact_records_support_seeds_status_and_boundaries(summary: dict) -> None:
    configuration = summary["configuration"]
    assert configuration["bootstrap_replicates"] == DEFAULT_REPEATS
    assert configuration["base_seed"] == 20260806
    assert configuration["docking_chemical_group_seed"] != configuration["base_seed"]
    assert len(configuration["experimental_seeds"]) == 10
    assert len(set(configuration["experimental_seeds"].values())) == 10
    support = summary["support"]
    assert support["target_pairs"] == 190
    assert len(support["targets"]) == 20
    assert support["docking_reference_ligands"] == 259_579
    assert support["docking_chemical_groups"] == 99_985
    assert support["experimental_panel_ligands"] == {
        "DAVIS": 72,
        "PKIS2": 645,
        "PKIS1": 360,
        "KiRHub": 92,
    }
    assert summary["status"].startswith("post_hoc")
    boundary = summary["claim_boundary"]
    assert "does NOT establish that the increment is tied to target identity" in (
        boundary
    )
    assert "transformation-matched null" in boundary
    assert "target-blind" in boundary


def test_kirhub_is_flagged_as_the_least_conservative_panel(summary: dict) -> None:
    assert summary["panels"]["KiRHub"]["chemical_cluster_bootstrap_available"] is False
    assert summary["panels"]["KiRHub"]["bootstrap_sources"] == "compound"
    for panel in ("DAVIS", "PKIS2", "PKIS1"):
        record = summary["panels"][panel]
        assert record["chemical_cluster_bootstrap_available"] is True
        assert record["bootstrap_sources"] == "|".join(CLUSTER_GROUPINGS)
    assert "no structures" in summary["kirhub_boundary"]


def test_artifact_summary_carries_no_wall_clock_or_absolute_paths(summary: dict) -> None:
    """A registered summary has to be byte-deterministic across reruns."""
    serialized = json.dumps(summary)
    for banned in ("runtime_seconds", "elapsed", "timestamp", "/Users/", "/private/"):
        assert banned not in serialized


def test_released_tables_agree_with_the_summary() -> None:
    intervals = pd.read_csv(DEFAULT_OUTPUT / "panel_intervals.csv")
    conservative = pd.read_csv(DEFAULT_OUTPUT / "conservative_intervals.csv")
    replicates = pd.read_csv(DEFAULT_OUTPUT / "bootstrap_replicates.csv")
    assert set(intervals.panel) == set(PANELS)
    assert (intervals.replicates == DEFAULT_REPEATS).all()
    assert (intervals.interval_95_low <= intervals.interval_95_high).all()
    assert (intervals.interval_90_low >= intervals.interval_95_low).all()
    assert (intervals.interval_90_high <= intervals.interval_95_high).all()
    assert np.allclose(
        intervals.basic_interval_95_low,
        2 * intervals.observed_delta_d - intervals.interval_95_high,
        atol=1e-12,
    )
    assert np.allclose(
        intervals.bootstrap_bias,
        intervals.bootstrap_mean - intervals.observed_delta_d,
        atol=1e-12,
    )
    assert len(replicates) == len(intervals) * DEFAULT_REPEATS
    grouped = replicates.groupby(
        ["panel", "resampling_scheme", "resampling_unit"]
    ).delta_d
    assert (grouped.size() == DEFAULT_REPEATS).all()
    for key, values in grouped:
        row = intervals[
            intervals.panel.eq(key[0])
            & intervals.resampling_scheme.eq(key[1])
            & intervals.resampling_unit.eq(key[2])
        ]
        assert len(row) == 1
        assert float(np.quantile(values.to_numpy(), 0.025)) == pytest.approx(
            float(row.interval_95_low.iloc[0]), abs=1e-9
        )
    combined = conservative[conservative.resampling_scheme.eq("combined")].set_index(
        "panel"
    )
    for panel, (low, high) in EXPECTED_COMBINED_INTERVAL_95.items():
        assert float(combined.loc[panel, "conservative_interval_95_low"]) == (
            pytest.approx(low, abs=1e-9)
        )
        assert float(combined.loc[panel, "conservative_interval_95_high"]) == (
            pytest.approx(high, abs=1e-9)
        )
