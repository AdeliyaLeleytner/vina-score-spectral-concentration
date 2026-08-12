from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import public_vina_panel_calibration_transfer as calibration


PACKAGE = Path(__file__).resolve().parents[1]
RESULTS = PACKAGE / "results" / "public_vina_panel_calibration_transfer"


def test_panel_metrics_exact_rank_and_headroom() -> None:
    distribution = np.asarray([0.5, 0.2, 0.3, 0.4])
    metrics = calibration.panel_metrics(distribution, 2)
    assert metrics["mean_nearest_distance"] == pytest.approx(0.3)
    assert metrics["oracle_mean_nearest_distance"] == pytest.approx(0.2)
    assert metrics["absolute_loss_vs_oracle"] == pytest.approx(0.1)
    assert metrics["fraction_all_panels_no_worse"] == pytest.approx(0.5)
    assert metrics["exact_rank_best_is_1"] == 2
    assert metrics["median_headroom_captured"] == pytest.approx(1 / 3)


def test_summary_counts_joint_top5() -> None:
    frame = pd.DataFrame(
        {
            "calibration_ligands": [200] * 3,
            "DAVIS_fraction_all_panels_no_worse": [0.01, 0.02, 0.10],
            "PKIS2_fraction_all_panels_no_worse": [0.01, 0.20, 0.01],
            "calibration_vs_full_map_spearman": [0.8, 0.9, 1.0],
            "panel_overlap_with_full_source": [4, 5, 6],
            "DAVIS_absolute_loss_vs_oracle": [0.1, 0.2, 0.3],
            "PKIS2_absolute_loss_vs_oracle": [0.1, 0.2, 0.3],
            "DAVIS_median_headroom_captured": [0.2, 0.3, 0.4],
            "PKIS2_median_headroom_captured": [0.2, 0.3, 0.4],
        }
    )
    result = calibration.summarize(frame).iloc[0]
    assert result.fraction_top5pct_DAVIS == pytest.approx(2 / 3)
    assert result.fraction_top5pct_PKIS2 == pytest.approx(2 / 3)
    assert result.fraction_top5pct_both_experimental_maps == pytest.approx(1 / 3)


def test_coverage_value_and_exact_percentile() -> None:
    distance = np.asarray(
        [
            [0.0, 0.1, 0.8],
            [0.1, 0.0, 0.4],
            [0.8, 0.4, 0.0],
        ]
    )
    assert calibration.coverage_value(distance, np.asarray([0])) == pytest.approx(
        0.3
    )
    distribution = np.asarray([0.1, 0.2, 0.2, 0.5])
    assert calibration.exact_percentile(distribution, 0.2) == pytest.approx(0.75)


def test_calibration_curve_integration_runs_under_pinned_numpy() -> None:
    grid = np.asarray([4.0, 6.0, 8.0, 10.0, 12.0])
    constant = np.full(len(grid), 0.25)
    assert calibration.normalized_trapezoid_auc(constant, grid) == pytest.approx(
        0.25
    )
    linear = grid.copy()
    assert calibration.normalized_trapezoid_auc(linear, grid) == pytest.approx(8.0)
    with pytest.raises(ValueError, match="strictly increasing"):
        calibration.normalized_trapezoid_auc(linear, grid[::-1])


def test_lexicographic_primary_and_complete_numerical_optimum_set() -> None:
    values = np.asarray([1.0, 0.2, 0.2 + 5e-13, 0.2 + 2e-12])
    assert calibration.lexicographic_argmin(values) == 1
    assert calibration.numerical_optimal_indices(values).tolist() == [1, 2]
    diagnostics = calibration.selection_tie_diagnostics(values)
    assert diagnostics["selected_lexicographic_index"] == 1
    assert diagnostics["exact_minimizer_count"] == 1
    assert diagnostics["panels_within_absolute_1e_12_of_minimum"] == 2


def test_full_source_choice_grid_honors_passed_seed_and_permutations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(calibration, "TARGETS", ("A", "B", "C"))
    monkeypatch.setattr(calibration, "K_GRID", (1, 2))
    rng = np.random.default_rng(71)
    broad = rng.normal(size=(80, 3))
    experimental = {
        "DAVIS": rng.normal(size=(40, 3)),
        "PKIS2": rng.normal(size=(50, 3)),
    }
    frame, selections, metadata = calibration.full_source_k_sensitivity(
        broad,
        experimental,
        permutations=101,
        seed=907,
    )
    contract = metadata["choice_grid_adjustment"]
    assert contract["permutations"] == 101
    assert contract["seed"] == 907
    assert len(frame) == 2 * 2 * 2
    assert len(selections) == sum((1, 2)) * 2
    p_values = frame.target_label_permutation_p_two_panel_max_percentile.unique()
    assert all(1 / 102 <= value <= 1 for value in p_values)


def test_same_endpoint_multiplier_summary_uses_positive_improvement() -> None:
    frame = pd.DataFrame(
        {
            "aggregation_scope": ["pooled"] * 4,
            "evaluation_map": ["both"] * 4,
            "targets_selected_k": [np.nan] * 4,
            "raw_minus_residual_selected_coverage_loss_improvement": [
                -0.1,
                0.1,
                0.2,
                0.3,
            ],
            "raw_minus_residual_selected_exact_percentile_improvement": [
                -0.2,
                0.1,
                0.2,
                0.4,
            ],
        }
    )
    summary = calibration.summarize_same_endpoint_multiplier(frame)[0]
    assert summary["coverage_loss_improvement_win_fraction"] == pytest.approx(0.75)
    assert summary["coverage_loss_improvement_mean"] == pytest.approx(0.125)


def test_symmetric_sequence_identity_is_order_invariant() -> None:
    from Bio import Align
    from Bio.Align import substitution_matrices

    aligner = Align.PairwiseAligner()
    aligner.mode = "global"
    aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
    aligner.open_gap_score = -10.0
    aligner.extend_gap_score = -0.5
    first = "MPEEERPLSASPHH"
    second = "MPEEPLASPHHH"
    forward = calibration.symmetric_global_alignment_identity(first, second, aligner)
    reverse = calibration.symmetric_global_alignment_identity(second, first, aligner)
    assert forward == pytest.approx(reverse, abs=0.0)
    matrix, metadata = calibration.load_sequence_identity()
    assert np.array_equal(matrix, matrix.T)
    assert "both sequence orientations" in metadata["tie_symmetrization"]


@pytest.mark.skipif(
    not (RESULTS / "summary.json").exists(), reason="production results not generated"
)
def test_production_shapes_and_outcome_blind_contract() -> None:
    summary = json.loads((RESULTS / "summary.json").read_text())
    replicates = pd.read_csv(RESULTS / "calibration_replicates.csv")
    summaries = pd.read_csv(RESULTS / "calibration_summary.csv")
    selections = pd.read_csv(RESULTS / "panel_selections.csv")
    k_sensitivity = pd.read_csv(RESULTS / "full_source_k_sensitivity.csv")
    k_selections = pd.read_csv(RESULTS / "full_source_k_selections.csv")
    same = pd.read_csv(RESULTS / "same_endpoint_centering_comparison.csv")
    same_selections = pd.read_csv(RESULTS / "same_endpoint_panel_selections.csv")
    multiplier = pd.read_csv(
        RESULTS / "same_endpoint_cluster_multiplier_bootstrap.csv"
    )
    sequence = pd.read_csv(RESULTS / "sequence_geometry_control.csv")
    assert summary["sizes"] == list(calibration.SIZES)
    assert len(replicates) == len(calibration.SIZES) * calibration.REPETITIONS
    assert len(summaries) == len(calibration.SIZES)
    assert len(selections) == len(replicates) * calibration.K + calibration.K
    assert replicates.groupby("calibration_ligands").size().eq(
        calibration.REPETITIONS
    ).all()
    assert replicates.top5pct_on_both_experimental_maps.dtype == bool
    assert "experimental values never enter selection" in summary["evaluation"]
    assert (replicates.panel_overlap_with_full_source.between(0, calibration.K)).all()
    assert len(k_sensitivity) == len(calibration.K_GRID) * 2 * 2
    assert len(k_selections) == sum(calibration.K_GRID) * 2
    assert len(same) == 2 * len(calibration.OBJECTIVES) * len(calibration.K_GRID) * 2
    primary_selections = same_selections.loc[
        same_selections.is_lexicographic_primary
    ]
    assert len(primary_selections) == (
        sum(calibration.K_GRID) * (2 * len(calibration.OBJECTIVES) + 1)
    )
    assert same_selections.numerical_optimum_member_rank.min() == 0
    assert len(multiplier) == calibration.CLUSTER_MULTIPLIER_REPETITIONS * 13
    assert set(multiplier.aggregation_scope) == {
        "per_k",
        "panel_mean_over_k",
        "pooled_mean_over_panels_and_k",
    }
    assert multiplier.raw_minus_residual_selected_coverage_loss_improvement.notna().all()
    assert multiplier.raw_minus_residual_selected_exact_percentile_improvement.notna().all()
    assert multiplier.all_optimal_pair_coverage_loss_improvement_min_conservative.notna().all()
    assert multiplier.all_optimal_pair_coverage_loss_improvement_mean.notna().all()
    assert len(sequence) == 3
    assert set(sequence.metric) == {
        "raw_vina_vs_sequence_edge_pearson",
        "row_centered_residual_vina_vs_sequence_edge_pearson",
        "residual_minus_raw_sequence_edge_pearson",
    }


@pytest.mark.skipif(
    not (RESULTS / "summary.json").exists(), reason="production results not generated"
)
def test_production_exact_panel_percentiles_and_checksums() -> None:
    summary = json.loads((RESULTS / "summary.json").read_text())
    assert summary["all_possible_panels"] == 203_490
    for panel in ("DAVIS", "PKIS2"):
        fraction = summary["full_source_transfer"][panel][
            "fraction_all_panels_no_worse"
        ]
        assert 0.0 < fraction <= 0.05
    inference = summary["same_endpoint_centering_comparison"]["inference"]
    primary = next(
        row
        for row in inference
        if row["experimental_endpoint"] == "metric_row_centered_residual"
        and row["objective"] == "signed_1_minus_r"
    )
    ranked = next(
        row
        for row in inference
        if row["experimental_endpoint"] == "column_rank_then_row_centered"
        and row["objective"] == "signed_1_minus_r"
    )
    assert primary["wins_lower_loss"] >= 9
    assert primary["mean_coverage_loss_improvement"] > 0.05
    assert "conservative_mean_coverage_loss_improvement_over_ties" in primary
    assert "target_label_p_one_sided_conservative_tie_improvement" in primary
    assert primary["conservative_mean_coverage_loss_improvement_over_ties"] > 0.04
    assert primary["conservative_tie_cells_with_positive_improvement"] >= 9
    assert "max_over_three_objectives_p_conservative_tie_improvement" in primary
    assert "max_over_three_objectives_p_tie_averaged_improvement" in primary
    assert ranked["wins_lower_loss"] == 10
    assert ranked["mean_coverage_loss_improvement"] > 0.05
    assert ranked["conservative_mean_coverage_loss_improvement_over_ties"] > 0.04
    tie_contract = summary["same_endpoint_centering_comparison"][
        "deterministic_tie_contract"
    ]
    assert "complete lexicographic" in tie_contract["primary_panel"]
    assert any(
        row["panels_within_absolute_1e_12_of_minimum"] > 1
        for row in tie_contract["selector_diagnostics"]
        if row["objective"] == "signed_1_minus_r"
    )
    sequence = pd.read_csv(RESULTS / "sequence_geometry_control.csv").set_index(
        "metric"
    )
    assert sequence.loc[
        "row_centered_residual_vina_vs_sequence_edge_pearson", "observed"
    ] > 0.15
    assert sequence.loc[
        "residual_minus_raw_sequence_edge_pearson",
        "target_label_qap_p_one_sided_at_least_observed",
    ] < 0.01
    pooled_bootstrap = next(
        row
        for row in summary["same_endpoint_cluster_multiplier_bootstrap"]["summary"]
        if row["aggregation_scope"] == "pooled_mean_over_panels_and_k"
    )
    assert pooled_bootstrap["coverage_loss_improvement_median"] > 0.0
    checksums = json.loads((RESULTS / "output_checksums.json").read_text())
    assert set(checksums) == set(calibration.OUTPUT_FILES)
    for filename, digest in checksums.items():
        assert calibration.sha256_file(RESULTS / filename) == digest
