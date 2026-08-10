from __future__ import annotations

import numpy as np

import calibration_panel_recovery as calibration


def test_covariance_to_correlation_has_unit_diagonal() -> None:
    covariance = np.asarray([[4.0, 1.0], [1.0, 9.0]])
    correlation = calibration.covariance_to_correlation(covariance)
    assert np.allclose(np.diag(correlation), 1.0)
    assert np.isclose(correlation[0, 1], 1.0 / 6.0)


def test_correlation_pr_identity_equals_dimension() -> None:
    assert calibration.correlation_pr(np.eye(5)) == 5.0


def test_geometry_is_symmetric_and_deterministic() -> None:
    rng = np.random.default_rng(17)
    matrix = rng.normal(size=(80, 6))
    first = calibration.geometry(matrix, shrink=True)
    second = calibration.geometry(matrix, shrink=True)
    assert np.allclose(first, first.T)
    assert np.allclose(np.diag(first), 1.0)
    assert np.array_equal(first, second)


def test_top_fraction_labels_uses_ceiling() -> None:
    labels = calibration.top_fraction_labels(np.arange(21.0), fraction=0.10)
    assert int(labels.sum()) == 3
    assert labels[-3:].all()


def test_panel_metrics_recovers_identical_reference() -> None:
    rng = np.random.default_rng(31)
    matrix = rng.normal(size=(150, 8))
    reference = calibration.geometry(matrix)
    labels = calibration.top_fraction_labels(calibration.upper(reference))
    metrics = calibration.panel_metrics(matrix, reference, labels, None, None)
    assert np.isclose(metrics["geometry_spearman"], 1.0)
    assert np.isclose(metrics["reference_top10_pair_roc_auc"], 1.0)
    assert np.isclose(metrics["reference_top10_pair_average_precision"], 1.0)
