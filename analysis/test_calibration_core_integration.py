from __future__ import annotations

import numpy as np

from calibration_core_integration import evaluate, summarize, unit_diagonal


def test_unit_diagonal_preserves_symmetry_and_sets_diagonal() -> None:
    vectors = np.asarray([[1.0, 0.0], [0.5, 1.0], [-0.2, 0.8]])
    kernel = vectors @ vectors.T
    correlation = unit_diagonal(kernel)
    assert np.allclose(correlation, correlation.T)
    assert np.allclose(np.diag(correlation), 1.0)
    assert np.linalg.eigvalsh(correlation).min() > -1e-10


def test_evaluate_is_tidy_and_outcome_labels_do_not_change_selection() -> None:
    rng = np.random.default_rng(7)
    latent = rng.normal(size=(180, 3))
    loadings = rng.normal(size=(3, 8))
    matrix = latent @ loadings + 0.4 * rng.normal(size=(180, 8))
    indices = np.arange(8)
    labels = np.zeros(28, dtype=bool)
    labels[:3] = True
    first, metadata = evaluate(
        matrix, indices, labels, (40,), 5, 11, 0.50
    )
    second, _ = evaluate(
        matrix, indices, ~labels, (40,), 5, 11, 0.50
    )
    assert len(first) == 10
    assert set(first.covariance_estimator) == {"empirical", "OAS"}
    assert metadata["full_selected_k"] >= 1
    assert np.array_equal(first.selected_k, second.selected_k)
    assert np.allclose(
        first.mean_squared_canonical_correlation,
        second.mean_squared_canonical_correlation,
    )


def test_summary_records_mode_count_distribution() -> None:
    rng = np.random.default_rng(13)
    matrix = rng.normal(size=(120, 6))
    labels = np.zeros(15, dtype=bool)
    labels[:2] = True
    records, _ = evaluate(matrix, np.arange(6), labels, (30,), 4, 5, 0.50)
    result = summarize(records)
    assert len(result) == 2
    assert result.repetitions.eq(4).all()
    assert result.selected_k_counts.str.contains(":").all()
