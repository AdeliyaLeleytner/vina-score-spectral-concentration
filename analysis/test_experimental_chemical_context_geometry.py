from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ANALYSIS = Path(__file__).resolve().parent
if str(ANALYSIS) not in sys.path:
    sys.path.insert(0, str(ANALYSIS))

import experimental_chemical_context_geometry as subject


def test_row_center_and_residual_geometry_constraints() -> None:
    matrix = np.array(
        [
            [1.0, 4.0, 2.0],
            [3.0, 2.0, 8.0],
            [5.0, 9.0, 4.0],
            [2.0, 7.0, 6.0],
        ]
    )
    centered = subject.row_center(matrix)
    assert np.allclose(centered.mean(axis=1), 0.0)
    correlation = subject.residual_geometry(matrix)
    assert np.allclose(correlation, correlation.T)
    assert np.allclose(np.diag(correlation), 1.0)


def test_integer_weighted_correlation_equals_repeated_rows() -> None:
    matrix = np.array(
        [
            [1.0, 2.0, 4.0],
            [2.0, 5.0, 1.0],
            [8.0, 3.0, 7.0],
            [4.0, 9.0, 2.0],
        ]
    )
    weights = np.array([2, 1, 3, 1])
    centered = subject.row_center(matrix)
    observed = subject.target_correlation(centered, weights)
    repeated = subject.target_correlation(np.repeat(centered, weights, axis=0))
    assert np.allclose(observed, repeated)


def test_random_disjoint_null_is_deterministic() -> None:
    rng = np.random.default_rng(5)
    matrix = rng.normal(size=(30, 5))
    first = subject.random_disjoint_null(matrix, 7, 8, 12, 31)
    second = subject.random_disjoint_null(matrix, 7, 8, 12, 31)
    assert np.array_equal(first, second)
    assert np.isfinite(first).all()


def test_direction_qap_detects_identical_delta() -> None:
    rng = np.random.default_rng(13)
    matrix = rng.normal(size=(6, 6))
    delta = (matrix + matrix.T) / 2.0
    np.fill_diagonal(delta, 0.0)
    result = subject.direction_qap(delta, delta, 199, 17)
    assert result["direction_spearman"] == 1.0
    assert result["sign_agreement_fraction"] == 1.0
    assert 0.0 < result["target_label_qap_p_positive"] <= 0.05


def test_extreme_masks_are_disjoint() -> None:
    values = np.arange(40, dtype=float)
    low, high, low_cut, high_cut = subject.extreme_masks(values)
    assert not np.any(low & high)
    assert low.sum() == 10
    assert high.sum() == 10
    assert low_cut < high_cut


def test_holm_adjustment_preserves_order_and_monotonicity() -> None:
    adjusted = subject.holm_adjust([0.04, 0.01, 0.20])
    assert np.allclose(adjusted, [0.08, 0.03, 0.20])
