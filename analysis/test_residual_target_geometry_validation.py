from __future__ import annotations

import numpy as np

try:
    from .residual_target_geometry_validation import (
        apply_molecular_weight_correction,
        column_rank_normal_scores,
        fixed_experimental_geometry_paired_qap,
        fit_molecular_weight_component,
        geometry_concordance,
        holm_adjust,
        matched_estimand_transition_qap,
        qap_test,
        retrieval_support_bootstrap,
        target_correlation,
        target_pair_retrieval,
        transformed_surface,
        two_way_center,
        weighted_target_correlation,
        weighted_two_way_center,
    )
except ImportError:  # pragma: no cover
    from residual_target_geometry_validation import (  # type: ignore
        apply_molecular_weight_correction,
        column_rank_normal_scores,
        fixed_experimental_geometry_paired_qap,
        fit_molecular_weight_component,
        geometry_concordance,
        holm_adjust,
        matched_estimand_transition_qap,
        qap_test,
        retrieval_support_bootstrap,
        target_correlation,
        target_pair_retrieval,
        transformed_surface,
        two_way_center,
        weighted_target_correlation,
        weighted_two_way_center,
    )


def test_two_way_center_has_zero_row_and_column_means() -> None:
    rng = np.random.default_rng(1)
    centered = two_way_center(rng.normal(size=(40, 7)))
    assert np.max(np.abs(centered.mean(axis=0))) < 1e-12
    assert np.max(np.abs(centered.mean(axis=1))) < 1e-12


def test_rank_normal_scores_are_finite_and_column_standardized() -> None:
    x = np.column_stack(
        [np.arange(20), np.repeat(np.arange(10), 2), np.arange(20)[::-1]]
    )
    scores = column_rank_normal_scores(x)
    assert np.isfinite(scores).all()
    assert np.max(np.abs(scores.mean(axis=0))) < 1e-12
    assert np.all(scores.std(axis=0, ddof=1) > 0)


def test_geometry_concordance_is_one_for_identical_geometry() -> None:
    rng = np.random.default_rng(2)
    geometry = target_correlation(
        transformed_surface(rng.normal(size=(200, 8)), "center_then_correlation")
    )
    assert abs(geometry_concordance(geometry, geometry) - 1.0) < 1e-12


def test_qap_is_deterministic_and_detects_aligned_labels() -> None:
    rng = np.random.default_rng(3)
    latent = rng.normal(size=(300, 3))
    loadings = rng.normal(size=(3, 7))
    matrix = latent @ loadings + 0.1 * rng.normal(size=(300, 7))
    geometry = target_correlation(two_way_center(matrix))
    first = qap_test(geometry, geometry, permutations=499, seed=17)
    second = qap_test(geometry, geometry, permutations=499, seed=17)
    assert first == second
    assert first["observed_spearman"] > 0.999999
    assert first["one_sided_p_positive"] <= 0.01


def test_molecular_weight_component_recovers_centered_slopes() -> None:
    rng = np.random.default_rng(4)
    molecular_weight = rng.normal(350, 60, size=800)
    z = (molecular_weight - molecular_weight.mean()) / molecular_weight.std(ddof=1)
    slope = np.array([-1.0, -0.3, 0.2, 0.6, 0.5])
    slope -= slope.mean()
    target_offset = np.array([-8.0, -7.0, -6.0, -5.0, -4.0])
    scores = target_offset + z[:, None] * slope[None, :]
    scores += 0.02 * rng.normal(size=scores.shape)
    fit = fit_molecular_weight_component(scores, molecular_weight)
    recovered = np.asarray(fit["centered_target_slopes_per_mw_sd"])
    assert np.max(np.abs(recovered - slope)) < 0.01
    corrected = apply_molecular_weight_correction(scores, molecular_weight, fit)
    corrected_slopes = (z @ corrected) / (z @ z)
    assert np.ptp(corrected_slopes) < 0.01


def test_weighted_two_way_center_and_correlation_are_well_defined() -> None:
    rng = np.random.default_rng(5)
    matrix = rng.normal(size=(60, 9))
    weights = rng.exponential(size=len(matrix))
    centered = weighted_two_way_center(matrix, weights)
    normalized = weights / weights.sum()
    assert np.max(np.abs(normalized @ centered)) < 1e-12
    assert np.max(np.abs(centered.mean(axis=1))) < 1e-12
    correlation = weighted_target_correlation(centered, weights)
    assert np.isfinite(correlation).all()
    assert np.max(np.abs(np.diag(correlation) - 1.0)) < 1e-12


def test_fixed_endpoint_and_matched_transition_qap_are_distinct() -> None:
    rng = np.random.default_rng(6)
    raw_first = target_correlation(rng.normal(size=(300, 7)))
    raw_second = target_correlation(rng.normal(size=(300, 7)))
    centered = target_correlation(two_way_center(rng.normal(size=(300, 7))))
    first = matched_estimand_transition_qap(
        raw_first, raw_second, centered, centered, permutations=199, seed=23
    )
    second = matched_estimand_transition_qap(
        raw_first, raw_second, centered, centered, permutations=199, seed=23
    )
    assert first == second
    assert first["centered_centered_minus_raw_raw"] > 0.5
    fixed = fixed_experimental_geometry_paired_qap(
        raw_first, centered, centered, permutations=199, seed=23
    )
    assert fixed["centered_docking_concordance"] > 0.999999
    assert fixed["fixed_experimental_endpoint"].startswith("two-way-centered")


def test_holm_adjustment_preserves_original_order() -> None:
    adjusted = holm_adjust([0.03, 0.001, 0.02])
    np.testing.assert_allclose(adjusted, [0.04, 0.003, 0.04])


def test_target_pair_retrieval_uses_fixed_experimental_labels() -> None:
    rng = np.random.default_rng(7)
    latent = rng.normal(size=(500, 3))
    loadings = rng.normal(size=(3, 8))
    experiment = latent @ loadings + 0.1 * rng.normal(size=(500, 8))
    docking = latent @ loadings + 0.3 * rng.normal(size=(500, 8))
    result = target_pair_retrieval(
        docking,
        experiment,
        fractions=(0.05, 0.10, 0.20),
        qap_permutations=99,
        seed=31,
    )
    primary = result["point_estimates"]["top_10_percent"]
    assert primary["positive_target_pairs"] == 3
    assert primary["centered_docking"]["roc_auc"] > 0.8


def test_retrieval_bootstrap_recomputes_labels_and_is_deterministic() -> None:
    rng = np.random.default_rng(8)
    latent = rng.normal(size=(200, 2))
    loadings = rng.normal(size=(2, 7))
    experiment = latent @ loadings + 0.2 * rng.normal(size=(200, 7))
    raw = target_correlation(rng.normal(size=(300, 7)))
    centered = target_correlation(two_way_center(experiment))
    first = retrieval_support_bootstrap(
        raw, centered, experiment, fraction=0.10, repeats=29, seed=37
    )
    second = retrieval_support_bootstrap(
        raw, centered, experiment, fraction=0.10, repeats=29, seed=37
    )
    assert first == second
    assert first["endpoint_recomputed_each_draw"] is True
    assert first["positive_target_pairs_per_draw"] == 3
    assert first["metrics"]["centered_minus_raw"]["roc_auc"]["n"] == 29
