import numpy as np
import pytest

import edge_stability_confidence as audit


def test_top_labels_has_exact_ceiling_count_and_is_deterministic():
    values = np.arange(190, dtype=float)
    labels = audit.top_labels(values, 0.10)
    assert labels.sum() == 19
    assert np.flatnonzero(labels).tolist() == list(range(171, 190))


def test_pair_label_map_matches_direct_matrix_permutation():
    p = 6
    tri = np.triu_indices(p, k=1)
    matrix = np.zeros((p, p), dtype=float)
    matrix[tri] = np.arange(len(tri[0]), dtype=float)
    matrix[(tri[1], tri[0])] = matrix[tri]
    order = np.asarray([3, 0, 5, 1, 4, 2])
    mapping = audit.pair_label_map(p, order)
    assert np.array_equal(matrix[np.ix_(order, order)][tri], matrix[tri][mapping])
    assert sorted(mapping.tolist()) == list(range(len(mapping)))


def test_balanced_ridge_is_invariant_to_affine_feature_rescaling():
    rng = np.random.default_rng(4)
    features = rng.normal(size=(80, 4))
    labels = np.zeros(80, dtype=bool)
    labels[np.argsort(features[:, 0] - features[:, 1])[-12:]] = True
    first = audit.balanced_ridge_scores(features, labels)
    transformed = features * np.asarray([2.0, -3.0, 7.0, 0.5]) + np.asarray(
        [4.0, 2.0, -10.0, 0.1]
    )
    second = audit.balanced_ridge_scores(transformed, labels)
    # Sign reversal of a standardized feature is absorbed by its coefficient.
    assert np.allclose(first, second, atol=1e-10)


def test_fast_binary_metrics_matches_sklearn_wrappers():
    labels = np.asarray([0, 1, 0, 1, 0, 0, 1], dtype=bool)
    scores = np.asarray([0.2, 0.9, 0.4, 0.6, 0.4, 0.1, 0.7])
    auc, ap = audit.fast_binary_metrics(labels, scores)
    expected = audit.binary_metrics(labels, scores)
    assert auc == pytest.approx(expected["roc_auc"])
    assert ap == pytest.approx(expected["average_precision"])


def test_partial_rank_association_removes_shared_control():
    rng = np.random.default_rng(9)
    control = np.linspace(-2, 2, 200)
    predictor = control + rng.normal(scale=0.05, size=len(control))
    endpoint = control + rng.normal(scale=0.05, size=len(control))
    marginal = np.corrcoef(predictor, endpoint)[0, 1]
    partial = audit.partial_rank_association(
        predictor, endpoint, control[:, None]
    )
    assert marginal > 0.99
    assert abs(partial) < 0.2


def test_balanced_scaffold_supports_are_exact_deterministic_and_disjoint():
    groups = np.asarray(
        [f"singleton_{index}" for index in range(50)]
        + [f"double_{index}" for index in range(5) for _ in range(2)],
        dtype=object,
    )
    first = audit.balanced_scaffold_support_ids(
        groups, support_count=3, support_size=10, seed=17
    )
    second = audit.balanced_scaffold_support_ids(
        groups, support_count=3, support_size=10, seed=17
    )
    assert np.array_equal(first, second)
    assert np.bincount(first[first >= 0], minlength=3).tolist() == [10, 10, 10]
    assert len(np.unique(groups[first >= 0])) == 30
    for group in np.unique(groups[first >= 0]):
        assert np.sum((groups == group) & (first >= 0)) == 1


def test_random_supports_are_exact_and_row_disjoint():
    support_ids = audit.random_row_support_ids(
        rows=100, support_count=4, support_size=12, seed=18
    )
    assert np.bincount(
        support_ids[support_ids >= 0], minlength=4
    ).tolist() == [12, 12, 12, 12]
    assert np.sum(support_ids >= 0) == 48


def test_residual_surface_removes_ligand_offsets_but_raw_surface_does_not():
    rng = np.random.default_rng(19)
    matrix = rng.normal(size=(200, 8))
    shifted = matrix + rng.normal(scale=4.0, size=(len(matrix), 1))
    residual_first = audit._surface_correlation(matrix, "residual")
    residual_second = audit._surface_correlation(shifted, "residual")
    raw_first = audit._surface_correlation(matrix, "raw")
    raw_second = audit._surface_correlation(shifted, "raw")
    assert np.allclose(residual_first, residual_second, atol=1e-12)
    assert not np.allclose(raw_first, raw_second, atol=1e-3)
