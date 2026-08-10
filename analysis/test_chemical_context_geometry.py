"""Unit tests for the science-only chemical-context geometry analysis."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from analysis import chemical_context_geometry as context


def _context_scores(seed: int = 13) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    ligands = 800
    targets = 8
    descriptor = np.linspace(100.0, 700.0, ligands)
    low_loading = rng.normal(size=(targets, 2))
    high_loading = rng.normal(size=(targets, 2))
    factors = rng.normal(size=(ligands, 2))
    interpolation = (descriptor - descriptor.min()) / np.ptp(descriptor)
    scores = np.empty((ligands, targets), dtype=np.float64)
    for index in range(ligands):
        loading = (
            (1.0 - interpolation[index]) * low_loading
            + interpolation[index] * high_loading
        )
        scores[index] = factors[index] @ loading.T
    scores += rng.normal(size=(ligands, 1))
    scores += 0.1 * rng.normal(size=scores.shape)
    return scores, descriptor


def test_row_center_is_correlation_equivalent_to_two_way_center() -> None:
    scores, _ = _context_scores()
    row_centered = context.row_center(scores)
    two_way = (
        scores
        - scores.mean(axis=0, keepdims=True)
        - scores.mean(axis=1, keepdims=True)
        + scores.mean()
    )
    np.testing.assert_allclose(
        context.target_correlation(row_centered),
        context.target_correlation(two_way),
        atol=1e-12,
        rtol=1e-12,
    )


def test_integer_weighted_correlation_matches_explicit_repeats() -> None:
    rng = np.random.default_rng(17)
    scores = rng.normal(size=(30, 6))
    weights = rng.integers(0, 4, size=len(scores))
    repeated = np.repeat(scores, weights, axis=0)
    np.testing.assert_allclose(
        context.target_correlation(scores, weights),
        context.target_correlation(repeated),
        atol=1e-12,
        rtol=1e-12,
    )


def test_extreme_masks_are_disjoint_and_inclusive() -> None:
    descriptor = np.arange(100, dtype=float)
    low, high, low_threshold, high_threshold = context.extreme_masks(descriptor)
    assert not np.any(low & high)
    assert low.sum() == 25
    assert high.sum() == 25
    assert np.all(descriptor[low] <= low_threshold)
    assert np.all(descriptor[high] >= high_threshold)


def test_quantile_bins_cover_every_row_exactly() -> None:
    descriptor = np.linspace(100.0, 800.0, 1001)
    identifiers, edges = context.quantile_bin_ids(descriptor, 5)
    assert len(identifiers) == len(descriptor)
    assert set(identifiers) == set(range(5))
    assert len(edges) == 6
    assert np.all(np.diff(edges) > 0)


def test_geometry_metrics_identify_context_shift() -> None:
    scores, descriptor = _context_scores()
    low, high, _, _ = context.extreme_masks(descriptor)
    residual = context.row_center(scores)
    observed = context.geometry_comparison(
        context.target_correlation(residual[low]),
        context.target_correlation(residual[high]),
    )
    rng = np.random.default_rng(19)
    random_rho = []
    for _ in range(20):
        order = rng.permutation(len(scores))
        first = order[: int(low.sum())]
        second = order[int(low.sum()) : int(low.sum() + high.sum())]
        random_rho.append(
            context.geometry_comparison(
                context.target_correlation(residual[first]),
                context.target_correlation(residual[second]),
            )["geometry_spearman"]
        )
    assert observed["geometry_spearman"] < np.quantile(random_rho, 0.05)
    assert 0 <= observed["sign_flip_fraction"] <= 1
    assert 0 <= observed["top_positive_10pct_pair_jaccard"] <= 1


def test_strongest_pair_mask_has_deterministic_ceiling_count() -> None:
    values = np.asarray([0.1, 0.5, 0.2, 0.5, -0.3, 0.0, 0.9])
    first = context.strongest_pair_mask(values, fraction=0.30)
    second = context.strongest_pair_mask(values, fraction=0.30)
    np.testing.assert_array_equal(first, second)
    assert first.sum() == 3


def test_group_factorization_is_deterministic_and_complete() -> None:
    groups = np.asarray(["B", "A", "B", "C", "A"], dtype=object)
    codes, unique = context._group_codes(groups)  # noqa: SLF001
    assert list(unique) == ["A", "B", "C"]
    assert np.all(codes >= 0)
    for group in unique:
        assert len(set(codes[groups == group])) == 1


def test_balanced_group_split_keeps_groups_whole_and_mw_stratified() -> None:
    sizes = np.asarray([3, 2, 4, 1] * 10)
    medians = np.linspace(100.0, 700.0, len(sizes))
    first = context.balanced_group_split(
        sizes, medians, np.random.default_rng(31), strata=10
    )
    second = context.balanced_group_split(
        sizes, medians, np.random.default_rng(31), strata=10
    )
    np.testing.assert_array_equal(first, second)
    assert set(first) == {0, 1}
    assert abs(int(sizes[first == 0].sum()) - int(sizes[first == 1].sum())) <= 10


def test_within_band_controls_are_restriction_matched_and_group_disjoint() -> None:
    scores, descriptor = _context_scores(seed=41)
    # Use 400 rows so every extreme band contains 50 two-row chemical groups,
    # enough to exercise the frozen ten-stratum assignment contract.
    scores = scores[:400]
    descriptor = descriptor[:400]
    descriptor_matrix = np.column_stack(
        [descriptor + 0.01 * column for column in range(7)]
    )
    bundle = SimpleNamespace(
        name="synthetic",
        descriptor_matrix=descriptor_matrix,
        groups=np.asarray([f"group-{index // 2}" for index in range(400)]),
    )
    surfaces = {
        "raw": scores,
        "row_centered_residual": context.row_center(scores),
    }
    first = context.within_mw_band_reproducibility_controls(
        bundle,
        surfaces,
        random_repetitions=4,
        group_disjoint_repetitions=3,
        seed=101,
    )
    second = context.within_mw_band_reproducibility_controls(
        bundle,
        surfaces,
        random_repetitions=4,
        group_disjoint_repetitions=3,
        seed=101,
    )
    pd.testing.assert_frame_equal(first, second)
    assert set(first.mw_band) == {"low_mw", "high_mw"}
    assert set(first.transformation) == {"raw", "row_centered_residual"}
    assert set(first.control_type) == {
        "row_random_disjoint",
        "mw_stratified_chemical_group_disjoint",
    }
    group_disjoint = first.loc[
        first.control_type.eq("mw_stratified_chemical_group_disjoint")
    ]
    assert (group_disjoint.chemical_group_overlap == 0).all()
    assert (group_disjoint.unassigned_ligands == 0).all()
    random = first.loc[first.control_type.eq("row_random_disjoint")]
    assert (random.first_ligands == random.second_ligands).all()
    assert random.chemical_group_overlap.max() > 0
    assert first.geometry_spearman.between(-1, 1).all()

    summary = context.summarize_within_mw_band_controls(first)
    assert len(summary) == 8
    assert set(summary.repetitions) == {3, 4}
    assert summary.interval_interpretation.str.contains(
        "not a population confidence interval", regex=False
    ).all()
