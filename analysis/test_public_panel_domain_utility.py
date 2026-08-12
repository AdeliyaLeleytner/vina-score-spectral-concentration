from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import public_panel_domain_utility as utility


def test_calibration_transform_is_calibration_only() -> None:
    calibration = np.asarray(
        [
            [1.0, 2.0, np.nan],
            [2.0, 4.0, 4.0],
            [3.0, 6.0, 6.0],
            [4.0, 8.0, 8.0],
            [5.0, 10.0, 10.0],
            [6.0, 12.0, 12.0],
        ]
    )
    first_evaluation = np.full((4, 3), 100.0)
    second_evaluation = np.full((4, 3), -100.0)
    first_cal, first_eval, first_mean, first_sd = utility.calibration_transform(
        calibration, first_evaluation
    )
    second_cal, second_eval, second_mean, second_sd = utility.calibration_transform(
        calibration, second_evaluation
    )
    np.testing.assert_allclose(first_cal, second_cal)
    np.testing.assert_allclose(first_mean, second_mean)
    np.testing.assert_allclose(first_sd, second_sd)
    assert not np.allclose(first_eval, second_eval)
    assert first_mean[2] == pytest.approx(8.0)


def test_descriptor_standardization_is_calibration_only() -> None:
    rng = np.random.default_rng(91)
    calibration = rng.normal(size=(20, 7))
    first_evaluation = rng.normal(size=(10, 7))
    second_evaluation = first_evaluation + 1_000.0
    first_cal, first_eval, first_mean, first_sd = (
        utility.calibration_standardize_features(calibration, first_evaluation)
    )
    second_cal, second_eval, second_mean, second_sd = (
        utility.calibration_standardize_features(calibration, second_evaluation)
    )
    np.testing.assert_allclose(first_cal, second_cal)
    np.testing.assert_allclose(first_mean, second_mean)
    np.testing.assert_allclose(first_sd, second_sd)
    assert not np.allclose(first_eval, second_eval)


def test_residual_map_row_centers_raw_scores_before_column_scaling() -> None:
    rng = np.random.default_rng(101)
    latent = rng.normal(size=(300, 3))
    raw = np.column_stack(
        [
            0.2 * latent[:, 0] + 0.01 * rng.normal(size=300),
            4.0 * latent[:, 0] + latent[:, 1],
            20.0 * latent[:, 1] - 2.0 * latent[:, 2],
            0.5 * latent[:, 2] + latent[:, 0],
        ]
    )
    observed = utility.map_geometries(raw)["residual"]
    expected = utility.target_correlation(raw - raw.mean(axis=1, keepdims=True))
    np.testing.assert_allclose(observed, expected, atol=1e-12)

    standardized = (raw - raw.mean(axis=0)) / raw.std(axis=0, ddof=1)
    wrong_order = utility.target_correlation(
        standardized - standardized.mean(axis=1, keepdims=True)
    )
    assert np.linalg.norm(observed - wrong_order) > 0.1


def test_distance_objectives_are_distinct_and_bounded() -> None:
    correlation = np.asarray(
        [[1.0, -0.9, 0.5], [-0.9, 1.0, 0.0], [0.5, 0.0, 1.0]]
    )
    r2 = utility.dissimilarity(correlation, "residual_r2")
    absolute = utility.dissimilarity(correlation, "residual_absolute")
    signed = utility.dissimilarity(correlation, "residual_signed")
    assert r2[0, 1] == pytest.approx(0.19)
    assert absolute[0, 1] == pytest.approx(0.1)
    assert signed[0, 1] == pytest.approx(1.9)
    assert np.all(np.diag(r2) == 0.0)


def test_exact_k_medoids_matches_brute_force() -> None:
    distance = np.asarray(
        [
            [0.0, 0.1, 0.9, 1.0, 1.0],
            [0.1, 0.0, 0.8, 0.9, 1.0],
            [0.9, 0.8, 0.0, 0.2, 0.3],
            [1.0, 0.9, 0.2, 0.0, 0.1],
            [1.0, 1.0, 0.3, 0.1, 0.0],
        ]
    )
    selected = utility.exact_k_medoids(distance, 2)
    selected_loss = utility.coverage_loss(distance, selected)
    brute = min(
        utility.coverage_loss(distance, np.asarray(panel))
        for panel in itertools.combinations(range(len(distance)), 2)
    )
    assert selected_loss == pytest.approx(brute, abs=1e-12)


def test_ridge_learns_negative_dependence_without_evaluation_row_mean() -> None:
    rng = np.random.default_rng(13)
    latent_cal = rng.normal(size=(150, 2))
    latent_eval = rng.normal(size=(80, 2))

    def surface(latent: np.ndarray) -> np.ndarray:
        first, second = latent.T
        return np.column_stack(
            [
                first,
                second,
                -first + 0.02 * rng.normal(size=len(first)),
                0.5 * first - second + 0.02 * rng.normal(size=len(first)),
                -0.7 * second + 0.02 * rng.normal(size=len(first)),
            ]
        )

    calibration = surface(latent_cal)
    evaluation = surface(latent_eval)
    transform = utility.split_transform(calibration, evaluation)
    aggregates, targets = utility.fit_reconstruction(
        transform,
        np.asarray([0, 1]),
        ("A", "B", "C", "D", "E"),
    )
    repeated_aggregates, _ = utility.fit_reconstruction(
        transform,
        np.asarray([0, 1]),
        ("A", "B", "C", "D", "E"),
    )
    raw = next(
        row
        for row in aggregates
        if row["truth_surface"] == "calibration_standardized_raw"
        and row["predictive_model"] == "multivariate_ridge"
        and row["outcome_basis"] == "split_imputed_all_rows"
    )
    residual = next(
        row
        for row in aggregates
        if row["truth_surface"] == "raw_row_centered_residual_outcome"
        and row["predictive_model"] == "multivariate_ridge"
        and row["outcome_basis"] == "split_imputed_all_rows"
    )
    pc1_raw = next(
        row
        for row in aggregates
        if row["truth_surface"] == "calibration_standardized_raw"
        and row["predictive_model"] == "selected_target_pc1"
        and row["outcome_basis"] == "split_imputed_all_rows"
    )
    assert raw["variance_weighted_r2"] > 0.99
    assert residual["variance_weighted_r2"] > 0.95
    assert raw["median_ligand_profile_spearman"] > 0.9
    assert raw["full_profile_policy_variance_weighted_r2"] > 0.99
    assert residual["full_profile_policy_variance_weighted_r2"] > 0.95
    assert (
        raw["full_profile_policy_variance_weighted_r2"]
        > pc1_raw["full_profile_policy_variance_weighted_r2"]
    )
    repeated_pc1_raw = next(
        row
        for row in repeated_aggregates
        if row["truth_surface"] == "calibration_standardized_raw"
        and row["predictive_model"] == "selected_target_pc1"
        and row["outcome_basis"] == "split_imputed_all_rows"
    )
    assert pc1_raw["variance_weighted_r2"] == pytest.approx(
        repeated_pc1_raw["variance_weighted_r2"], abs=1e-14
    )
    assert raw["median_target_equal_budget_lower_5pct_overlap"] > 0.8
    assert raw[
        "full_profile_policy_median_target_equal_budget_lower_5pct_overlap"
    ] > 0.8
    assert raw["selected_targets"] + raw["omitted_targets"] == 5
    assert len(aggregates) == 8
    assert len(targets) == 8
    assert {row["outcome_basis"] for row in aggregates} == {
        "split_imputed_all_rows",
        "originally_complete_outcome_rows",
    }


def test_domain_comparator_is_row_disjoint_and_uses_common_truth(monkeypatch) -> None:
    rng = np.random.default_rng(17)
    matrix = rng.normal(size=(480, 10))
    dataset = SimpleNamespace(
        name="synthetic",
        matrix=matrix,
        targets=tuple(f"T{i}" for i in range(10)),
        molecular_weight=np.linspace(100.0, 600.0, len(matrix)),
    )
    metrics, reconstruction = utility.analyze_domain_shift(
        dataset, sizes=(20,), repetitions=2, seed=19
    )
    assert len(metrics) == 2 * 2 * 2 * 3
    assert (metrics.calibration_evaluation_overlap == 0).all()
    counts = metrics.groupby(
        ["domain", "calibration_ligands", "repetition", "transformation"]
    ).calibration_scope.nunique()
    assert (counts == 3).all()
    assert metrics.map_spearman_to_same_domain_complement.between(-1, 1).all()
    assert len(reconstruction) == 2 * 2 * 2 * 3
    assert set(reconstruction.truth_surface) == {
        "evaluation_standardized_raw",
        "evaluation_standardized_full_profile_residual",
    }
    assert (reconstruction.calibration_evaluation_overlap == 0).all()
    assert np.isfinite(
        reconstruction.full_profile_policy_variance_weighted_r2
    ).all()


def test_cost_break_even_matches_declared_formula() -> None:
    dataset = SimpleNamespace(name="D", matrix=np.zeros((1000, 44)))
    table = utility.cost_table(dataset, (200,), panel_k=8).iloc[0]
    assert table.prospective_break_even_future_ligands == pytest.approx(200 * 44 / 36)
    assert table.minimum_integer_future_ligands_for_savings == 245
    assert table.selected_workflow_score_cells == 200 * 44 + 800 * 8


def test_summary_keeps_random_draws_but_counts_calibration_repetitions() -> None:
    frame = pd.DataFrame(
        {
            "dataset": ["D"] * 6,
            "method": ["random"] * 6,
            "repetition": [0, 0, 0, 1, 1, 1],
            "random_panel_draw": [0, 1, 2, 0, 1, 2],
            "metric": np.arange(6.0),
        }
    )
    result = utility.summarize_replicates(frame, ["dataset", "method"]).iloc[0]
    assert result.records == 6
    assert result.independent_calibration_repetitions == 2
    assert result.metric_median == 2.5


def test_paired_reconstruction_contrasts_use_same_split_random_panels() -> None:
    rows = []
    for repetition in (0, 1):
        for draw, r2 in enumerate((0.1, 0.3, 0.5)):
            rows.append(
                {
                    "dataset": "D",
                    "calibration_ligands": 200,
                    "repetition": repetition,
                    "truth_surface": "residual",
                    "outcome_basis": "complete",
                    "predictive_model": "multivariate_ridge",
                    "panel_method": "random_panel",
                    "panel_information_scope": "outcome_blind_random",
                    "variance_weighted_r2": r2,
                    "median_target_r2": r2,
                    "pooled_calibration_standardized_rmse": 1.0 - r2,
                        "median_ligand_profile_spearman": r2,
                        "full_profile_policy_variance_weighted_r2": r2,
                        "full_profile_policy_pooled_calibration_standardized_rmse": 1.0 - r2,
                        "full_profile_policy_median_ligand_profile_spearman": r2,
                        "median_target_equal_budget_lower_5pct_overlap": r2,
                        "full_profile_policy_median_target_equal_budget_lower_5pct_overlap": r2,
                }
            )
        rows.append(
            {
                "dataset": "D",
                "calibration_ligands": 200,
                "repetition": repetition,
                "truth_surface": "residual",
                "outcome_basis": "complete",
                "predictive_model": "multivariate_ridge",
                "panel_method": "pilot_exact_residual_r2",
                "panel_information_scope": "pilot_only",
                "variance_weighted_r2": 0.6,
                "median_target_r2": 0.6,
                "pooled_calibration_standardized_rmse": 0.3,
                "median_ligand_profile_spearman": 0.6,
                "full_profile_policy_variance_weighted_r2": 0.6,
                "full_profile_policy_pooled_calibration_standardized_rmse": 0.3,
                "full_profile_policy_median_ligand_profile_spearman": 0.6,
                "median_target_equal_budget_lower_5pct_overlap": 0.6,
                "full_profile_policy_median_target_equal_budget_lower_5pct_overlap": 0.6,
            }
        )
    contrast = utility.reconstruction_random_contrasts(pd.DataFrame(rows))
    assert len(contrast) == 2
    assert (contrast.random_panels == 3).all()
    assert np.allclose(
        contrast.pilot_minus_random_median_variance_weighted_r2, 0.3
    )
    assert (
        contrast.fraction_random_panels_no_worse_variance_weighted_r2 == 0.0
    ).all()


def test_released_artifact_is_complete_and_checksum_valid() -> None:
    output = utility.DEFAULT_OUTPUT
    if not (output / "metadata.json").exists():
        pytest.skip("production artifact has not been generated")
    metadata = json.loads((output / "metadata.json").read_text())
    checksums = json.loads((output / "output_checksums.json").read_text())
    assert metadata["schema_version"] == "1.0.0"
    assert metadata["analysis_status"] == "strict_public_exploratory_utility_validation"
    assert metadata["panel_targets_k"] == 8
    assert set(metadata["datasets"]) == {"Docking-44", "DOCKSTRING-58"}
    assert "do not validate experimental affinity" in metadata["claim_boundary"]
    assert set(metadata["code_sha256"]) == {
        "analysis/public_panel_domain_utility.py",
        "analysis/test_public_panel_domain_utility.py",
    }
    for filename, expected in checksums["files"].items():
        observed = hashlib.sha256((output / filename).read_bytes()).hexdigest()
        assert observed == expected

    reconstruction = pd.read_csv(output / "reconstruction_metrics.csv")
    coverage = pd.read_csv(output / "panel_coverage_metrics.csv")
    domains = pd.read_csv(output / "mw_domain_recovery_metrics.csv")
    domain_reconstruction = pd.read_csv(
        output / "mw_domain_reconstruction_metrics.csv"
    )
    costs = pd.read_csv(output / "score_cell_costs.csv")
    assert set(reconstruction.truth_surface) == {
        "calibration_standardized_raw",
        "raw_row_centered_residual_outcome",
    }
    assert set(reconstruction.predictive_model) == {
        "multivariate_ridge",
        "selected_target_pc1",
    }
    assert set(reconstruction.outcome_basis) == {
        "split_imputed_all_rows",
        "originally_complete_outcome_rows",
    }
    assert set(coverage.objective) == set(utility.DISTANCE_OBJECTIVES)
    assert set(coverage.selection_source) == {
        "pilot_exact",
        "full_source_exact_diagnostic",
    }
    assert (coverage.held_out_true_regret >= -1e-12).all()
    assert coverage.fraction_random_panels_no_worse.between(0, 1).all()
    assert (reconstruction.calibration_evaluation_overlap == 0).all()
    assert (domains.calibration_evaluation_overlap == 0).all()
    assert set(domains.calibration_scope) == {
        "within_domain",
        "source_library_row_disjoint",
        "opposite_mw_domain",
    }
    assert set(domain_reconstruction.calibration_scope) == {
        "within_domain",
        "source_library_row_disjoint",
        "opposite_mw_domain",
    }
    assert (domain_reconstruction.calibration_evaluation_overlap == 0).all()
    assert np.isfinite(reconstruction.select_dtypes(include=[np.number])).all().all()
    assert "raw_originally_observed_cell_full_policy_variance_weighted_r2" in (
        reconstruction.columns
    )
    descriptor = pd.read_csv(output / "descriptor_baseline_metrics.csv")
    assert set(descriptor.predictive_model) == {"ligand_descriptor_ridge"}
    for filename in (
        "reconstruction_ridge_pc1_contrasts.csv",
        "reconstruction_raw_residual_contrasts.csv",
        "reconstruction_descriptor_contrasts.csv",
    ):
        assert (output / filename).exists()
    assert (
        costs.minimum_integer_future_ligands_for_savings
        == np.floor(costs.prospective_break_even_future_ligands).astype(int) + 1
    ).all()
