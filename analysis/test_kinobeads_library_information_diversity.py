from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

import kinobeads_library_information_diversity as subject


def test_curveball_preserves_both_degree_sequences() -> None:
    matrix = np.array(
        [
            [1, 1, 0, 0, 0],
            [0, 1, 1, 0, 0],
            [0, 0, 1, 1, 0],
            [1, 0, 0, 1, 1],
            [0, 1, 0, 0, 1],
        ],
        dtype=np.uint8,
    )
    rows = [set(np.flatnonzero(row)) for row in matrix]
    subject.curveball_trade(rows, np.random.default_rng(7), 1000)
    randomized = subject.rows_to_binary(rows, matrix.shape[1])
    assert np.array_equal(randomized.sum(axis=0), matrix.sum(axis=0))
    assert np.array_equal(randomized.sum(axis=1), matrix.sum(axis=1))
    assert set(np.unique(randomized)) <= {0, 1}


def test_uniform_weighted_geometry_matches_unweighted_geometry() -> None:
    rng = np.random.default_rng(11)
    matrix = rng.normal(size=(80, 7))
    weights = np.full(len(matrix), 1.0 / len(matrix))
    assert np.allclose(
        subject.weighted_correlation(matrix, weights),
        subject.correlation_matrix(matrix),
        atol=1e-12,
    )
    assert np.allclose(
        subject.weighted_covariance(matrix, weights),
        subject.covariance_matrix(matrix),
        atol=1e-12,
    )


def test_participation_ratio_bounds() -> None:
    assert subject.participation_ratio(np.eye(8)) == 8.0
    concentrated = np.full((8, 8), 1.0)
    assert np.isclose(subject.participation_ratio(concentrated), 1.0)


def test_common_support_requires_hits_and_nonhits_in_every_group() -> None:
    matrix = pd.DataFrame(
        {
            "a": [1, 1, 0, 0, 1, 0, 1, 0],
            "b": [1, 1, 1, 1, 0, 0, 0, 0],
            "c": [1, 0, 0, 0, 1, 0, 0, 0],
        }
    )
    groups = {
        "KCGS": np.array([1, 1, 1, 1, 0, 0, 0, 0], dtype=bool),
        "PKIS_exclusive": np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=bool),
        "PKIS2_exclusive": np.array([1, 1, 1, 1, 0, 0, 0, 0], dtype=bool),
        "Roche": np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=bool),
    }
    support = subject.common_target_support(matrix, groups, minimum_hits=1)
    assert support.tolist() == [True, False, True]


def test_build_groups_assigns_overlaps_only_to_kcgs() -> None:
    one_each = pd.DataFrame(
        {
            "KCGS": ["x", np.nan, np.nan, np.nan, np.nan],
            "PKIS": ["x", "x", np.nan, "x", np.nan],
            "PKIS2": [np.nan, np.nan, "x", "x", np.nan],
            "Roche": [np.nan, np.nan, np.nan, np.nan, "x"],
        }
    )
    metadata = pd.concat([one_each, one_each, one_each], ignore_index=True)
    groups = subject.build_groups(metadata)
    assert groups["KCGS"].sum() == 3
    assert groups["PKIS_exclusive"].sum() == 3
    assert groups["PKIS2_exclusive"].sum() == 3
    assert groups["Roche"].sum() == 3
    assert not np.any(
        np.column_stack([groups[name] for name in subject.GROUP_ORDER]).sum(axis=1) > 1
    )


def test_cross_chemotype_collision_excludes_same_chemotype_pair() -> None:
    matrix = np.array(
        [
            [1, 0, 0],
            [1, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
        ],
        dtype=np.uint8,
    )
    chemotypes = np.array(["a", "a", "b", "c"])
    all_pairs = subject.collision_by_degree(matrix, chemotypes, False)[1]
    cross = subject.collision_by_degree(matrix, chemotypes, True)[1]
    assert all_pairs["identical_pairs"] == 3
    assert all_pairs["eligible_pairs"] == 6
    assert cross["identical_pairs"] == 2
    assert cross["eligible_pairs"] == 5


def test_capped_degree_weights_match_reference_bins() -> None:
    candidate = np.array([0, 0, 1, 1, 2, 2, 3, 7, 10])
    reference = np.array([0, 1, 2, 6, 8, 9])
    weights = subject.capped_degree_weights(candidate, reference)
    candidate_bins = np.minimum(candidate, 6)
    reference_bins = np.minimum(reference, 6)
    for degree in sorted(set(candidate_bins) & set(reference_bins)):
        assert np.isclose(
            weights[candidate_bins == degree].sum(), np.mean(reference_bins == degree)
        )


def test_gzip_csv_writer_is_byte_deterministic(tmp_path: Path) -> None:
    frame = pd.DataFrame({"a": [1, 2], "b": [0.125, 0.25]})
    path = tmp_path / "draws.csv.gz"
    subject.write_csv_deterministic(frame, path)
    first = path.read_bytes()
    subject.write_csv_deterministic(frame, path)
    assert path.read_bytes() == first
