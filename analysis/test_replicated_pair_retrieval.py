import numpy as np
import pytest

import replicated_pair_retrieval as retrieval


def test_upper_tail_labels_has_exact_declared_count() -> None:
    labels = retrieval.upper_tail_labels(np.arange(10, dtype=float), 0.21)
    assert int(labels.sum()) == 3
    assert np.array_equal(np.flatnonzero(labels), np.array([7, 8, 9]))


def test_replicated_labels_requires_two_panels() -> None:
    ranks = {
        "first": np.array([0.9, 0.8, 0.1, 0.2]),
        "second": np.array([0.95, 0.1, 0.85, 0.2]),
        "third": np.array([0.99, 0.2, 0.1, 0.8]),
    }
    labels, panel_labels = retrieval.replicated_labels(
        ranks, fraction=0.25, minimum_panels=2
    )
    assert set(panel_labels) == set(ranks)
    assert np.array_equal(labels, np.array([True, False, False, False]))


def test_retrieval_metrics_perfect_and_reversed() -> None:
    labels = np.array([False, False, True, True])
    perfect = retrieval.retrieval_metrics(labels, np.array([0.0, 1.0, 2.0, 3.0]))
    reversed_metrics = retrieval.retrieval_metrics(
        labels, np.array([3.0, 2.0, 1.0, 0.0])
    )
    assert perfect["roc_auc"] == pytest.approx(1.0)
    assert perfect["average_precision"] == pytest.approx(1.0)
    assert reversed_metrics["roc_auc"] == pytest.approx(0.0)


def test_fixed_endpoint_qap_is_deterministic() -> None:
    raw = np.array(
        [
            [1.0, 0.1, 0.2, 0.3],
            [0.1, 1.0, 0.4, 0.5],
            [0.2, 0.4, 1.0, 0.9],
            [0.3, 0.5, 0.9, 1.0],
        ]
    )
    centered = raw.copy()
    centered[0, 1] = centered[1, 0] = 0.95
    labels = np.array([True, False, False, False, False, True])
    first = retrieval.fixed_endpoint_qap(labels, raw, centered, 100, seed=13)
    second = retrieval.fixed_endpoint_qap(labels, raw, centered, 100, seed=13)
    assert first == second
    assert first["metrics"]["roc_auc"]["centered_minus_raw"] > 0


def test_invalid_endpoint_fails_closed() -> None:
    with pytest.raises(ValueError):
        retrieval.retrieval_metrics(
            np.array([True, True]), np.array([0.1, 0.2])
        )


def test_pr_and_leading_mode_reconstruction() -> None:
    correlation = np.array([[1.0, 0.5], [0.5, 1.0]])
    assert retrieval.correlation_pr(correlation) == pytest.approx(1.6)
    leading = retrieval.leading_mode_reconstruction(correlation)
    assert np.allclose(leading, np.full((2, 2), 0.75))


def test_holm_adjustment_is_monotone_in_sorted_order() -> None:
    adjusted = retrieval.holm_adjust({"first": 0.01, "second": 0.03})
    assert adjusted["first"] == pytest.approx(0.02)
    assert adjusted["second"] == pytest.approx(0.03)
