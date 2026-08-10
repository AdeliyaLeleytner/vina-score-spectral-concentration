from __future__ import annotations

import numpy as np

import pair_specific_normalization_transfer as transfer


def test_pair_cell_arrays_direction_and_ties() -> None:
    experiment = np.tile(np.arange(len(transfer.TARGETS)), (3, 1)).astype(float)
    scores = -experiment.copy()
    eligible, correct = transfer.pair_cell_arrays(scores, experiment, margin=0.0)
    assert eligible.all()
    assert np.all(correct == 1.0)
    tied_scores = np.zeros_like(scores)
    _, tied = transfer.pair_cell_arrays(tied_scores, experiment, margin=0.0)
    assert np.all(tied == 0.5)


def test_pair_cells_weighting_matches_row_repetition() -> None:
    rng = np.random.default_rng(4)
    experiment = rng.normal(size=(6, 20))
    absolute = rng.normal(size=(6, 20))
    residual = rng.normal(size=(6, 20))
    weights = np.asarray([2, 0, 1, 3, 0, 1], dtype=float)
    weighted = transfer.pair_cells(
        absolute, residual, experiment, 0.0, weights
    )
    repeated = np.repeat(np.arange(6), weights.astype(int))
    expanded = transfer.pair_cells(
        absolute[repeated], residual[repeated], experiment[repeated], 0.0
    )
    assert np.allclose(weighted.support, expanded.support)
    assert np.allclose(weighted.absolute_accuracy, expanded.absolute_accuracy)
    assert np.allclose(weighted.residual_accuracy, expanded.residual_accuracy)


def test_representations_remove_all_target_effects_in_fitted_reference() -> None:
    rng = np.random.default_rng(8)
    reference = rng.normal(size=(200, 21)) + np.linspace(-2, 2, 21)
    fit = transfer._fit_from_reference(reference)
    absolute, residual, pooled = transfer.representations(reference, fit)
    assert absolute.shape == (200, 20)
    assert residual.shape == (200, 20)
    assert pooled > 0
    # Residual columns are standardized on all 21 targets before restriction.
    all_mean = np.asarray(fit["target_mean"])
    all_sd = np.asarray(fit["residual_target_standard_deviation"])
    grand = fit["grand_mean"]
    full = reference - all_mean - reference.mean(1, keepdims=True) + grand
    full /= all_sd
    assert np.allclose(residual, full[:, transfer.TARGET_INDICES])


def test_vector_to_symmetric_roundtrip() -> None:
    values = np.arange(len(transfer.TRI[0]))
    matrix = transfer._vector_to_symmetric(values)
    assert np.array_equal(matrix[transfer.TRI], values)
    assert np.array_equal(matrix, matrix.T)


def test_aggregate_pair_weighting() -> None:
    accuracy = np.asarray([0.0, 1.0, 0.5])
    support = np.asarray([1.0, 3.0, 2.0])
    mask = np.ones(3, dtype=bool)
    assert transfer.aggregate_accuracy(
        accuracy, support, mask, pair_weighted=True
    ) == 4.0 / 6.0
    assert transfer.aggregate_accuracy(
        accuracy, support, mask, pair_weighted=False
    ) == 0.5


def test_prior_strata_exclude_ties_and_unsupported_pairs() -> None:
    prior = np.asarray(
        [
            [0.0, 0.5, 1.0, 0.0],
            [1.0, 0.0, 0.5, 1.0],
        ]
    )
    eligible = np.ones_like(prior, dtype=bool)
    pair_mask = np.asarray([True, True, False, True])
    reversal, routine, tied = transfer.prior_stratum_masks(
        prior, eligible, pair_mask
    )
    assert np.array_equal(
        reversal,
        np.asarray(
            [
                [True, False, False, True],
                [False, True, False, False],
            ]
        ),
    )
    assert np.array_equal(
        routine,
        np.asarray(
            [
                [False, False, False, False],
                [True, False, False, True],
            ]
        ),
    )
    assert np.array_equal(
        tied,
        np.asarray(
            [
                [False, True, False, False],
                [False, False, False, False],
            ]
        ),
    )
    assert not (reversal & routine).any()
    assert not (reversal & tied).any()
    assert not (routine & tied).any()
