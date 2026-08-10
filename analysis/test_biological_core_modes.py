"""Unit tests for the science-only biological-core-mode audit."""

from __future__ import annotations

import inspect

import numpy as np

from analysis import biological_core_modes as core


def _synthetic_scores(seed: int = 7, ligands: int = 500, targets: int = 8) -> np.ndarray:
    rng = np.random.default_rng(seed)
    loadings = rng.normal(size=(targets, 3))
    factors = rng.normal(size=(ligands, 3))
    ligand_axis = rng.normal(size=(ligands, 1))
    target_offset = rng.normal(size=(1, targets))
    return ligand_axis + target_offset + factors @ loadings.T + 0.2 * rng.normal(
        size=(ligands, targets)
    )


def test_covariance_formula_matches_explicit_two_way_centering() -> None:
    scores = _synthetic_scores()
    explicit = core.geometry.geometry_correlation(scores, "center_then_correlation")
    moment = core.residual_correlation(scores)
    np.testing.assert_allclose(moment, explicit, atol=1e-12, rtol=1e-12)


def test_docking_only_selector_has_no_experimental_argument() -> None:
    signature = inspect.signature(core.docking_only_mode_count)
    assert tuple(signature.parameters) == ("correlation", "threshold")
    scores = _synthetic_scores()
    correlation = core.residual_correlation(scores)
    selected = core.docking_only_mode_count(correlation, 0.50)
    eigenvalues = selected["eigenvalues"]
    k = selected["selected_k"]
    assert eigenvalues[:k].sum() / eigenvalues.sum() >= 0.50
    if k > 1:
        assert eigenvalues[: k - 1].sum() / eigenvalues.sum() < 0.50


def test_truncation_has_requested_rank_and_leading_eigenvalues() -> None:
    correlation = core.residual_correlation(_synthetic_scores())
    eigenvalues, _ = core.ordered_eigendecomposition(correlation)
    spectral_kernel = core.truncated_geometry(correlation, 3)
    observed, _ = core.ordered_eigendecomposition(spectral_kernel)
    np.testing.assert_allclose(observed[:3], eigenvalues[:3], atol=1e-10)
    assert np.sum(observed > 1e-9) == 3
    assert not np.allclose(np.diag(spectral_kernel), 1.0)


def test_unit_diagonal_normalization_returns_a_correlation_matrix() -> None:
    correlation = core.residual_correlation(_synthetic_scores())
    spectral_kernel = core.truncated_geometry(correlation, 3)
    normalized = core.unit_diagonal_correlation(spectral_kernel)
    np.testing.assert_allclose(np.diag(normalized), 1.0, atol=1e-14)
    np.testing.assert_allclose(normalized, normalized.T, atol=1e-14)
    assert np.linalg.eigvalsh(normalized).min() > -1e-10


def test_subspace_stability_is_rotation_and_sign_invariant() -> None:
    correlation = core.residual_correlation(_synthetic_scores())
    result = core.subspace_stability(correlation, correlation, 3)
    assert np.isclose(result["mean_squared_canonical_correlation"], 1.0)
    assert np.isclose(result["minimum_canonical_correlation"], 1.0)
    assert result["maximum_principal_angle_degrees"] < 1e-5


def test_scaffold_fold_assignment_keeps_groups_whole_and_is_deterministic() -> None:
    groups = np.asarray(["A", "A", "B", "C", "B", "D", "D", "E"])
    first = core.scaffold_fold_ids(groups, folds=3, seed=11)
    second = core.scaffold_fold_ids(groups, folds=3, seed=11)
    np.testing.assert_array_equal(first, second)
    for group in set(groups):
        assert len(set(first[groups == group])) == 1


def test_panel_metrics_distinguish_normalized_correlation_and_kernel() -> None:
    correlation = core.residual_correlation(_synthetic_scores())
    metrics = core.panel_metrics(
        correlation,
        correlation,
        correlation,
        correlation,
    )
    for representation in ("normalized_low_mode", "spectral_kernel"):
        for metric in ("continuous_spearman", "roc_auc", "average_precision"):
            assert np.isclose(
                metrics[f"{representation}_minus_full_{metric}"], 0.0
            )


def test_random_subspace_null_preserves_retained_eigenvalues() -> None:
    correlation = core.residual_correlation(_synthetic_scores(targets=6))
    kernel = core.truncated_geometry(correlation, 2)
    normalized = core.unit_diagonal_correlation(kernel)
    endpoints = {
        "PKIS1": correlation,
        "DAVIS": correlation,
        "PKIS2": correlation,
        "KiRHub": correlation,
    }
    summary, spectrum = core.eigenvalue_matched_random_subspace_null(
        correlation,
        normalized,
        kernel,
        endpoints,
        modes=2,
        repeats=25,
        seed=13,
    )
    expected, _ = core.ordered_eigendecomposition(correlation)
    np.testing.assert_allclose(spectrum.retained_eigenvalue, expected[:2])
    assert set(summary.panel) == {
        "PKIS1",
        "DAVIS",
        "PKIS2",
        "KiRHub",
        "LOCKED_VALIDATION_MEAN",
    }
    assert set(summary.representation) == {
        "normalized_low_mode_correlation",
        "communality_weighted_spectral_kernel",
    }
    assert len(summary) == 2 * 5 * 3
    assert summary.one_sided_p_random_at_least_observed_candidate.between(
        0, 1
    ).all()


def test_qap_outputs_both_explicit_representations() -> None:
    correlation = core.residual_correlation(_synthetic_scores(targets=6))
    kernel = core.truncated_geometry(correlation, 2)
    normalized = core.unit_diagonal_correlation(kernel)
    endpoints = {
        name: correlation.copy() for name in core.PANEL_ROLES
    }
    paired = core.paired_qap_records(
        correlation,
        normalized,
        kernel,
        endpoints,
        permutations=19,
        seed=23,
    )
    omnibus = core.omnibus_validation_qap(
        correlation,
        normalized,
        kernel,
        endpoints,
        permutations=19,
        seed=29,
    )
    expected = {
        "normalized_low_mode_correlation",
        "communality_weighted_spectral_kernel",
    }
    assert set(paired.representation) == expected
    assert set(omnibus.representation) == expected
    assert len(paired) == 2 * len(endpoints) * 3
    assert len(omnibus) == 2 * 3
    assert paired.loc[
        paired.representation.eq("normalized_low_mode_correlation"),
        "primary_representation",
    ].all()


def test_target_jackknife_reselects_k_using_only_docking_covariance() -> None:
    scores = _synthetic_scores(targets=len(core.TARGETS))
    covariance = np.cov(scores, rowvar=False)
    panel_covariances = {
        name: covariance.copy() for name in core.PANEL_ROLES
    }
    frame = core.target_jackknife(covariance, panel_covariances, threshold=0.50)
    assert len(frame) == len(core.TARGETS) * len(core.PANEL_ROLES)
    assert frame.remaining_targets.eq(len(core.TARGETS) - 1).all()
    assert frame.selected_k.ge(1).all()


def test_target_pair_geometry_frame_is_complete_and_aligned() -> None:
    triangle = np.triu_indices(4, k=1)
    full = np.eye(4)
    full[triangle] = np.arange(6) / 10
    full[(triangle[1], triangle[0])] = np.arange(6) / 10
    normalized = full.copy()
    kernel = full * 0.5
    frame = core.target_pair_geometry_frame(
        full, normalized, kernel, targets=("A", "B", "C", "D")
    )
    assert len(frame) == 6
    assert frame.iloc[0].target_a == "A"
    assert frame.iloc[-1].target_b == "D"
    assert np.allclose(
        frame.normalized_low_mode_correlation,
        frame.full_residual_correlation,
    )


def test_holm_adjustment_is_monotone_in_ordered_p_values() -> None:
    raw = {"a": 0.001, "b": 0.02, "c": 0.03, "d": 0.5}
    adjusted = core.holm_adjust(raw)
    ordered = sorted(raw, key=raw.get)
    values = [adjusted[name] for name in ordered]
    assert values == sorted(values)
    assert all(adjusted[name] >= raw[name] for name in raw)
