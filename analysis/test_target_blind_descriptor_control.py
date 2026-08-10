"""Contract tests for the target-blind (shared-coefficient) descriptor control.

These read the released artifact rather than re-running the control, which needs the full
DOCKSTRING surface, RDKit descriptors for 259,579 ligands and several minutes of work.

The point of the artifact is negative, so the tests are written to fail loudly if a future
edit turns the degenerate shared-coefficient arm into an apparently informative number.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


PACKAGE = Path(__file__).resolve().parents[1]
RESULTS = PACKAGE / "results" / "target_blind_descriptor_control"
PANELS = ("DAVIS", "PKIS2", "PKIS1", "KiRHub")
BLIND = "descriptor_component_target_blind"
SPECIFIC = "descriptor_component_target_specific"
OBSERVED = "observed_residual"


def load_summary() -> dict:
    return json.loads((RESULTS / "summary.json").read_text())


def test_shared_coefficient_geometry_is_degenerate_by_construction() -> None:
    """One slope vector for the panel means one distinct target-pair value, exactly +1.

    This is an algebraic consequence, not an empirical finding: the predicted column for
    every target is the same vector of ligand scores. If this ever stops holding, the
    shared-coefficient arm has silently stopped being shared.
    """
    summary = load_summary()
    degeneracy = summary["target_blind_degeneracy"]
    assert degeneracy["is_degenerate"] is True
    assert degeneracy["target_pairs"] == 190
    assert degeneracy["distinct_edge_values_at_1e-12"] == 1
    assert degeneracy["maximum_absolute_deviation_from_plus_one"] < 1e-12

    # Spearman against a constant vector is undefined, and must be recorded as undefined
    # rather than as zero or as any finite number.
    for panel in PANELS:
        assert summary["panel_agreement"][BLIND][panel]["spearman"] is None
        assert summary["panel_agreement"][BLIND][panel]["target_label_qap"] is None
    assert summary["headline"]["target_blind_spearman_with_every_panel"] is None

    frame = pd.read_csv(RESULTS / "target_pair_geometry.csv")
    assert len(frame) == 190
    blind_edges = frame[f"docking_{BLIND}_correlation"].to_numpy(dtype=float)
    assert np.allclose(blind_edges, 1.0, atol=1e-12)
    assert frame[f"docking_{SPECIFIC}_correlation"].nunique() == 190


def test_reference_arms_reproduce_the_published_descriptor_component() -> None:
    """The control's reference arm must be the published estimator, not a lookalike."""
    summary = load_summary()
    verification = summary["verification"]
    reproduction = verification["reference_arm_reproduces_published_estimator"]
    assert reproduction["maximum_absolute_difference_descriptor_component"] < 1e-9
    assert reproduction["maximum_absolute_difference_observed_surface"] < 1e-9
    assert verification["observed_arm_versus_released_ledger_spearman"] > 0.99
    assert (
        verification["pooled_equals_mean_of_per_target_coefficients_max_abs_difference"]
        < 1e-9
    )
    cross_check = verification["published_descriptor_component_geometry_cross_check"]
    if cross_check["available"]:
        assert cross_check["maximum_absolute_spearman_deviation"] < 1e-9


def test_descriptor_component_variance_needs_per_target_slopes() -> None:
    """Almost all of the fitted descriptor surface is coefficient heterogeneity."""
    summary = load_summary()
    split = summary["descriptor_component_variance_split"]
    shared = split["shared_share_of_descriptor_component_variance"]
    heterogeneity = split["heterogeneity_share_of_descriptor_component_variance"]
    assert abs(shared + heterogeneity - 1.0) < 1e-12
    assert heterogeneity > 0.95, heterogeneity
    # The split is exactly orthogonal because the pooled vector is the mean of the
    # per-target vectors; a non-vanishing cross term means that identity broke.
    assert split["cross_term_relative_to_total"] < 1e-10
    interval = summary["chemical_cluster_bootstrap"]["conservative_union_90"][
        "heterogeneity_share_of_descriptor_component_variance"
    ]
    assert interval[0] <= heterogeneity <= interval[1]
    assert interval[0] > 0.95


def test_target_blind_model_predicts_almost_nothing() -> None:
    """A genuinely target-blind descriptor fit barely beats each target's own mean."""
    summary = load_summary()
    predictive = summary["out_of_fold_predictive_r2"]
    specific = predictive[SPECIFIC]["mean_out_of_fold_target_r2"]
    blind = predictive[BLIND]["mean_out_of_fold_target_r2"]
    assert specific > 0.05, specific
    assert blind < 0.01, blind
    assert specific > 10 * blind

    frame = pd.read_csv(RESULTS / "target_predictive_r2.csv")
    assert len(frame) == 20
    assert np.allclose(
        frame["difference"],
        frame["target_specific_out_of_fold_r2"] - frame["target_blind_out_of_fold_r2"],
    )
    assert (frame["target_specific_out_of_fold_r2"] > 0).all()
    assert (
        frame["target_blind_out_of_fold_r2"]
        < frame["target_specific_out_of_fold_r2"]
    ).all()


def test_rank_ladder_starts_degenerate_and_ends_at_the_published_arm() -> None:
    summary = load_summary()
    rungs = summary["coefficient_rank_ladder"]["rungs"]
    ranks = [rung["coefficient_deviation_rank"] for rung in rungs]
    assert ranks == list(range(8))
    first, last = rungs[0], rungs[-1]
    assert first["degenerate"] is True
    assert all(first[f"{panel}_spearman"] is None for panel in PANELS)
    assert last["degenerate"] is False
    assert last["coefficient_deviation_rank"] == 7
    assert abs(last["variance_share_of_full_descriptor_component"] - 1.0) < 1e-9
    assert last["maximum_absolute_coefficient_reconstruction_error"] < 1e-9
    for panel in PANELS:
        published = summary["panel_agreement"][SPECIFIC][panel]["spearman"]
        assert abs(last[f"{panel}_spearman"] - published) < 1e-9, panel
    # Rank 1 keeps the fitted slope marginals but only one direction of target contrast;
    # if that already reproduced the published agreement the ladder would be uninformative.
    assert max(abs(rungs[1][f"{panel}_spearman"]) for panel in PANELS) < 0.15
    diagnostics = summary["coefficient_rank_ladder"]["diagnostics"]
    assert diagnostics["nominal_maximum_rank_from_matrix_shape"] == 8
    assert diagnostics["effective_maximum_rank"] == 7
    assert summary["coefficient_rank_7_svd_comparator"]["rung"] == last


def test_shared_slope_permutation_is_only_a_relabelling() -> None:
    """The one-permutation-per-run slope control is the target-label QAP, not new evidence."""
    summary = load_summary()
    null = summary["slope_permutation_null"]
    identity = null["shared_permutation_is_a_target_relabelling"]
    assert (
        identity["maximum_absolute_geometry_difference_from_direct_relabelling"] < 1e-10
    )
    assert null["permutations"] >= 5000
    frame = pd.read_csv(RESULTS / "slope_permutation_null.csv")
    assert set(frame["null"]) == {"shared_permutation", "fold_independent_permutation"}
    assert len(frame) == 2 * len(PANELS)
    assert frame["one_sided_p_positive"].between(0.0, 1.0).all()
    for variant in ("shared_permutation", "fold_independent_permutation"):
        # Both nulls centre near zero: destroying the slope-to-target pairing destroys the
        # agreement, which is the whole point of the control.
        assert abs(frame.loc[frame["null"].eq(variant), "null_mean"]).max() < 0.05


def test_reporting_contract() -> None:
    summary = load_summary()
    assert summary["status"] == "exploratory_post_hoc_reviewer_requested_control"
    assert summary["support"]["ligands"] > 259_000
    assert summary["support"]["targets"] == 20
    assert summary["support"]["ligand_by_target_cells"] == (
        summary["support"]["ligands"] * summary["support"]["targets"]
    )
    assert summary["target_pairs"] == 190
    assert summary["configuration"]["folds"] == 5
    assert summary["configuration"]["bootstrap_repeats"] == 5000
    assert isinstance(summary["seeds"]["base"], int)
    boundary = summary["claim_boundary"]
    assert "does NOT" in boundary
    assert "degenerate BY CONSTRUCTION" in boundary
    assert "question" in summary and summary["question"].strip()

    agreement = pd.read_csv(RESULTS / "panel_agreement.csv")
    assert len(agreement) == 3 * len(PANELS)
    for arm in (OBSERVED, SPECIFIC):
        rows = agreement[agreement["arm"].eq(arm)]
        assert rows["spearman"].notna().all()
        assert (
            rows["conservative_union_lower_90"] <= rows["spearman"]
        ).all()
        assert (
            rows["conservative_union_upper_90"] >= rows["spearman"]
        ).all()
    assert agreement[agreement["arm"].eq(BLIND)]["spearman"].isna().all()
