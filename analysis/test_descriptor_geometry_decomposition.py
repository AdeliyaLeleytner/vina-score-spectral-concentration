"""Contract tests for the descriptor-geometry decomposition artifact.

These read the released artifact rather than re-running the decomposition, which needs the
full DOCKSTRING surface and the PKIS1 supplement.  The
pure edge-level statistics are re-derived from the released 190-edge table, so the tests
check the numbers rather than only the file layout.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats


PACKAGE = Path(__file__).resolve().parents[1]
RESULTS = PACKAGE / "results" / "descriptor_geometry_decomposition"
PANELS = ("DAVIS", "PKIS2", "PKIS1", "KiRHub")
SURFACES = ("observed_residual", "descriptor_component", "descriptor_removed")

# This analysis was an exploratory mediation decomposition that was deliberately retired
# before the submission evidence ledger was frozen.  Keep its pure-code regression tests
# available to developers who regenerate the optional artifact, but do not make an absent,
# unregistered result bundle fail the submission package's full test discovery.
pytestmark = pytest.mark.skipif(
    not (RESULTS / "summary.json").exists(),
    reason="retired exploratory artifact is not part of the submission release",
)


def load_summary() -> dict:
    return json.loads((RESULTS / "summary.json").read_text())


def load_edges() -> pd.DataFrame:
    return pd.read_csv(RESULTS / "target_pair_edges.csv")


def rank_z(values: np.ndarray) -> np.ndarray:
    ranks = stats.rankdata(np.asarray(values, dtype=float), method="average")
    return (ranks - ranks.mean()) / ranks.std(ddof=1)


def test_support_seeds_and_claim_boundary() -> None:
    summary = load_summary()
    support = summary["support"]
    assert support["targets"] == 20
    assert support["target_pairs"] == 190
    assert support["ligands_after_descriptor_filter"] > 259_000
    assert support["cells"] == support["ligands_after_descriptor_filter"] * 20
    assert tuple(support["experimental_panels"]) == PANELS
    bootstrap = summary["chemical_group_bootstrap"]
    assert bootstrap["replicates"] == 5_000
    assert set(bootstrap["seeds"]) == set(bootstrap["groupings"])
    assert summary["configuration"]["seed"] > 0
    assert summary["status"] == "exploratory_post_hoc_science_only"
    assert bootstrap["mode"] == "fixed_oof_coarsened"
    assert "not individual-cluster" in bootstrap[
        "conditioning_and_coarsening_limitation"
    ]
    assert "does NOT show" in summary["claim_boundary"]
    assert "target-blind" in summary["claim_boundary"]


def test_every_reported_agreement_carries_a_conservative_union_interval() -> None:
    """Each of the three absolute agreements must have a two-grouping union interval."""
    summary = load_summary()
    statistics = summary["chemical_group_bootstrap"]["statistics"]
    groupings = summary["chemical_group_bootstrap"]["groupings"]
    assert len(groupings) == 2
    for panel in PANELS:
        for surface in SURFACES:
            key = f"{panel}::agreement_{surface}"
            entry = statistics[key]
            union = entry["conservative_union_sensitivity_interval_95"]
            point = summary["experimental_panel_results"][panel]["absolute_agreement"][
                surface
            ]["spearman"]
            assert union[0] < union[1], key
            assert union[0] <= point <= union[1], (key, point, union)
            for grouping in groupings:
                described = entry[grouping]
                assert described["n"] + described["failed_replicates"] == 5_000, key
                interval = described[
                    "conditional_coarsened_sensitivity_interval_95"
                ]
                assert described["not_a_confidence_interval"] is True
                assert union[0] <= interval[0], key
                assert union[1] >= interval[1], key


def test_the_two_components_are_not_orthogonal() -> None:
    """The reviewer's objection, checked against the released edge table.

    A ratio of the two agreements would only be interpretable if the components were
    orthogonal.  They are not: their 190-edge vectors are positively correlated, and the
    joint regression's semipartial R-squareds therefore do not sum to the joint R-squared.
    """
    summary = load_summary()
    frame = load_edges()
    assert len(frame) == 190
    predicted = frame["docking_descriptor_component_correlation"].to_numpy()
    remainder = frame["docking_descriptor_removed_correlation"].to_numpy()
    observed = stats.spearmanr(predicted, remainder).statistic
    reported = summary["component_versus_remainder_edge_correlation"]["spearman"]
    assert observed == pytest.approx(reported, abs=1e-12)
    assert reported > 0.2, reported
    for panel in PANELS:
        joint = summary["experimental_panel_results"][panel][
            "edge_level_joint_decomposition"
        ]
        total = joint["joint_rank_r2"]
        parts = joint["semipartial_rank_r2"]
        assert parts["descriptor_component"] + parts["descriptor_removed"] < total


def test_standardized_partial_coefficients_match_the_edge_table() -> None:
    """Recompute the joint rank regression from the released 190 edges."""
    summary = load_summary()
    frame = load_edges()
    predicted = rank_z(frame["docking_descriptor_component_correlation"].to_numpy())
    remainder = rank_z(frame["docking_descriptor_removed_correlation"].to_numpy())
    design = np.column_stack([np.ones(len(frame)), predicted, remainder])
    for panel in PANELS:
        response = rank_z(
            frame[f"{panel}_experimental_centered_correlation"].to_numpy()
        )
        coefficients = np.linalg.lstsq(design, response, rcond=None)[0]
        reported = summary["experimental_panel_results"][panel][
            "edge_level_joint_decomposition"
        ]["standardized_partial_coefficients"]
        assert coefficients[1] == pytest.approx(
            reported["descriptor_component"], abs=1e-10
        ), panel
        assert coefficients[2] == pytest.approx(
            reported["descriptor_removed"], abs=1e-10
        ), panel


def test_a_target_blind_descriptor_model_has_no_geometry_at_all() -> None:
    """The control that shows the descriptor component is not target-blind.

    One slope vector shared by all 20 targets produces 20 identical columns, so the
    190-edge geometry is constant and cannot agree with anything.  Every edge of the
    fitted descriptor component therefore comes from its per-target slopes.
    """
    diagnostic = load_summary()["per_target_slope_diagnostic"]
    control = diagnostic["target_blind_common_slope_control"]
    assert control["agreement_with_experiment_is_defined"] is False
    assert control["edge_standard_deviation"] < 1e-9
    assert control["minimum_offdiagonal_correlation"] > 1.0 - 1e-9
    assert diagnostic["descriptor_component_is_built_from_per_target_fitted_slopes"]
    slopes = diagnostic["pooled_descriptive_target_slopes"]
    assert len(slopes) == 7
    assert any(record["sign_changes_across_targets"] for record in slopes.values())


def test_the_retired_ratio_is_recorded_as_unstable_not_as_a_result() -> None:
    summary = load_summary()
    assert "mediated" in summary["replaces"]["retired_statistic"]
    statistics = summary["chemical_group_bootstrap"]["statistics"]
    widths = []
    for panel in PANELS:
        entry = statistics[f"{panel}::retired_mediated_share_ratio"]
        low, high = entry["conservative_union_sensitivity_interval_95"]
        widths.append(high - low)
        point = summary["experimental_panel_results"][panel][
            "retired_mediated_share_ratio"
        ]["value"]
        assert low <= point <= high, panel
    # The ratio's interval is wider than the interval of the absolute agreement it is
    # built from, which is the concrete reason the ratio is not a reportable statistic.
    for panel, width in zip(PANELS, widths):
        agreement = statistics[f"{panel}::agreement_descriptor_component"][
            "conservative_union_sensitivity_interval_95"
        ]
        assert width > agreement[1] - agreement[0], panel


def test_summary_is_free_of_wall_clock_and_absolute_paths() -> None:
    """Determinism guard: a registered summary must not carry timings or machine paths."""
    text = (RESULTS / "summary.json").read_text()
    assert "/Users/" not in text
    assert "/private/" not in text
    assert "runtime" not in text.lower()
    assert "elapsed" not in text.lower()
    assert "timestamp" not in text.lower()
