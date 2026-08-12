from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import public_experimental_boundary_controls as boundary


PACKAGE = Path(__file__).resolve().parents[1]
RESULTS = PACKAGE / "results" / "public_experimental_boundary_controls"


def test_target_map_removes_row_offsets() -> None:
    rng = np.random.default_rng(11)
    matrix = rng.normal(size=(80, 6))
    offsets = rng.normal(size=(80, 1))
    assert np.allclose(boundary.target_map(matrix), boundary.target_map(matrix + offsets))


def test_rank_map_is_invariant_to_columnwise_monotone_transforms() -> None:
    rng = np.random.default_rng(12)
    matrix = rng.normal(size=(100, 5))
    transformed = np.column_stack(
        [np.exp(matrix[:, 0]), matrix[:, 1] ** 3, matrix[:, 2], 2 * matrix[:, 3] + 7, np.tanh(matrix[:, 4])]
    )
    assert np.allclose(
        boundary.column_rank_target_map(matrix),
        boundary.column_rank_target_map(transformed),
    )


def test_distance_contracts() -> None:
    correlation = np.asarray([[1.0, -0.5, 0.2], [-0.5, 1.0, 0.0], [0.2, 0.0, 1.0]])
    signed = boundary.distance_from_map(correlation, "signed_1_minus_r")
    absolute = boundary.distance_from_map(correlation, "absolute_1_minus_abs_r")
    squared = boundary.distance_from_map(correlation, "squared_1_minus_r2")
    assert signed[0, 1] == pytest.approx(1.5)
    assert absolute[0, 1] == pytest.approx(0.5)
    assert squared[0, 1] == pytest.approx(0.75)
    assert np.allclose(np.diag(signed), 0.0)


def test_all_panel_enumeration_and_coverage() -> None:
    panels = boundary.all_panels(n_targets=6, k=3)
    assert len(panels) == 20
    assert np.array_equal(panels[0], [0, 1, 2])
    assert np.array_equal(panels[-1], [3, 4, 5])
    distance = np.abs(np.subtract.outer(np.arange(6), np.arange(6))).astype(float)
    observed = boundary.coverage_distribution(distance, panels, batch_size=4)
    direct = np.asarray(
        [np.min(distance[:, panel], axis=1).mean() for panel in panels], dtype=float
    )
    assert np.allclose(observed, direct)


def test_draw_unique_candidates_enforces_rows_and_groups() -> None:
    candidates = [np.asarray([0, 1, 2]), np.asarray([1, 2, 3]), np.asarray([2, 3, 4])]
    groups = np.asarray(["a", "a", "b", "c", "d"], dtype=object)
    selected = boundary.draw_unique_candidates(candidates, np.random.default_rng(5))
    assert len(selected) == len(np.unique(selected)) == 3
    group_selected = boundary.draw_unique_candidates(
        candidates, np.random.default_rng(6), groups
    )
    assert len(np.unique(groups[group_selected])) == 3


def test_uniform_distinct_group_draw_uses_one_row_per_group() -> None:
    row_groups = np.asarray(["a", "a", "b", "c", "c", "c"], dtype=object)
    selected = boundary.uniform_distinct_group_draw(
        row_groups, 3, np.random.default_rng(9)
    )
    assert len(np.unique(row_groups[selected])) == 3


def test_qap_record_is_deterministic() -> None:
    rng = np.random.default_rng(20)
    first = boundary.target_map(rng.normal(size=(100, 7)))
    second = boundary.target_map(rng.normal(size=(120, 7)))
    a = boundary.qap_record(first, second, permutations=100, seed=17)
    b = boundary.qap_record(first, second, permutations=100, seed=17)
    assert a == b
    assert 1 / 101 <= a["target_label_qap_p_two_sided"] <= 1.0


def test_public_robustness_uses_fixed_estimable_binary_targets() -> None:
    panels = boundary.reliability.load_public_panels()
    davis, pkis2 = panels.values()
    frame = boundary.robust_cross_panel(
        davis, pkis2, qap_permutations=100, seed=23
    )
    assert len(frame) == 21
    exact = frame.loc[
        frame.support.eq("exact_dockstring_match_informative")
    ]
    assert set(exact.davis_ligands) == {56}
    assert set(exact.pkis2_ligands) == {154}
    binary = frame.loc[
        frame.analysis_family.eq("fully_crossed_binary_threshold_grid")
    ]
    assert len(binary) == 9
    assert binary.targets.nunique() == 1
    assert binary.target_names.nunique() == 1
    assert int(binary.targets.iloc[0]) >= 3


@pytest.mark.skipif(
    not (RESULTS / "summary.json").exists(), reason="production results not generated"
)
def test_production_result_shapes_and_contracts() -> None:
    overlap = pd.read_csv(RESULTS / "cross_panel_overlap.csv")
    sensitivity = pd.read_csv(RESULTS / "cross_panel_overlap_sensitivity.csv")
    robust = pd.read_csv(RESULTS / "cross_panel_robustness.csv")
    controls = pd.read_csv(RESULTS / "exact_support_control_replicates.csv")
    control_summary = pd.read_csv(RESULTS / "exact_support_control_summary.csv")
    coverage = pd.read_csv(RESULTS / "panel8_coverage.csv")
    panel_bootstrap = pd.read_csv(RESULTS / "panel8_cluster_bootstrap.csv")
    selections = pd.read_csv(RESULTS / "panel8_selections.csv")
    summary = json.loads((RESULTS / "summary.json").read_text())

    assert set(overlap.overlap_level[:4]) == {
        "full_standard_inchikey",
        "connectivity_block",
        "nonempty_bemis_murcko_scaffold",
        "mixed_panel_butina_cluster",
    }
    assert len(sensitivity) == 5
    assert len(robust) == 21
    assert set(controls.panel) == {"DAVIS", "PKIS2"}
    assert set(controls.control_design) == {
        "uniform_random_rows",
        "mw_20_quantile_matched",
        "seven_descriptor_nearest_256",
        "uniform_scaffold_distinct",
        "seven_descriptor_nearest_256_scaffold_distinct",
    }
    repeats = summary["configuration"]["control_repeats"]
    assert len(controls) == 2 * 2 * 5 * repeats
    assert len(control_summary) == 20
    assert set(control_summary.ligands) == {56, 154}
    assert len(coverage) == 14
    assert set(coverage.transformation) == {
        "raw",
        "row_centered_residual",
        "column_rank_then_row_centered_evaluation",
    }
    assert (coverage.all_possible_panels == 203_490).all()
    assert len(selections) == 2 * 3 * 2 * 3 * 8 + 2 * 2 * 8
    assert set(selections.selection_source) == {
        "Vina_selected",
        "cross_panel_experimental_selected",
        "in_sample_experimental_oracle",
    }
    assert len(panel_bootstrap) == 2 * 2 * summary["configuration"][
        "panel_cluster_bootstraps"
    ]
    assert set(panel_bootstrap.panel) == {"DAVIS", "PKIS2"}
    assert set(panel_bootstrap.resampling_scheme) == {
        "ordinary_cluster_resample_with_replacement",
        "positive_exp1_cluster_multiplier",
    }
    multiplier = panel_bootstrap.loc[
        panel_bootstrap.resampling_scheme.eq("positive_exp1_cluster_multiplier")
    ]
    assert set(multiplier.status) == {"ok"}
    assert np.isfinite(
        multiplier[
            [
                "raw_fixed_vina_excess_loss_over_in_sample_oracle",
                "residual_fixed_vina_excess_loss_over_in_sample_oracle",
                "residual_minus_raw_excess_loss_over_in_sample_oracle",
            ]
        ].to_numpy(float)
    ).all()
    assert summary["support"] == {
        "targets": 21,
        "target_pairs": 210,
        "davis_released_ligands": 72,
        "davis_informative_ligands": 67,
        "pkis2_released_ligands": 645,
        "pkis2_informative_ligands": 644,
    }


@pytest.mark.skipif(
    not (RESULTS / "summary.json").exists(), reason="production results not generated"
)
def test_production_overlap_counts_and_no_selected_threshold() -> None:
    overlap = pd.read_csv(RESULTS / "cross_panel_overlap.csv").set_index("overlap_level")
    assert overlap.loc["full_standard_inchikey", "shared_units"] == 3
    assert overlap.loc["connectivity_block", "shared_units"] == 6
    assert overlap.loc["nonempty_bemis_murcko_scaffold", "shared_units"] == 10
    robust = pd.read_csv(RESULTS / "cross_panel_robustness.csv")
    binary = robust.loc[robust.analysis_family.eq("fully_crossed_binary_threshold_grid")]
    assert len(binary) == 9
    assert set(binary.davis_threshold) == {5.0, 5.5, 6.0}
    assert set(binary.pkis2_threshold) == {50.0, 70.0, 80.0}


@pytest.mark.skipif(
    not (RESULTS / "summary.json").exists(), reason="production results not generated"
)
def test_output_checksums_cover_every_artifact() -> None:
    checksums = json.loads((RESULTS / "output_checksums.json").read_text())
    assert set(checksums) == set(boundary.OUTPUT_FILES)
    for filename, digest in checksums.items():
        assert boundary.sha256_file(RESULTS / filename) == digest
