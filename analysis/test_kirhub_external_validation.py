from __future__ import annotations

import json

import numpy as np
import pytest

try:
    from . import kirhub_external_validation as kirhub
except ImportError:  # pragma: no cover
    import kirhub_external_validation as kirhub  # type: ignore


def test_name_normalization_is_exact_but_punctuation_insensitive() -> None:
    assert kirhub.normalize_drug_name("  Pazo-tanib ") == "pazo tanib"
    assert kirhub.normalize_drug_name("Pazo_tanib") == "pazo tanib"
    assert kirhub.normalize_drug_name("Pazopanib") != "pazo tanib"


def test_correlation_invariances_are_explicit() -> None:
    rng = np.random.default_rng(1)
    matrix = rng.normal(size=(100, 8))
    raw = kirhub.correlation_geometry(matrix, "raw")
    column_centered = kirhub.correlation_geometry(matrix, "column_center_only")
    row_centered = kirhub.correlation_geometry(matrix, "row_center_only")
    two_way = kirhub.correlation_geometry(matrix, "two_way_center")
    assert np.allclose(raw, column_centered, atol=1e-12)
    assert np.allclose(row_centered, two_way, atol=1e-12)


def test_mean_rank_geometry_preserves_upper_triangle() -> None:
    ranks = {
        "first": np.arange(6, dtype=float),
        "second": np.arange(6, dtype=float)[::-1],
    }
    matrix = kirhub.mean_rank_geometry(ranks, 4)
    assert np.allclose(kirhub.geometry.upper_triangle(matrix), 2.5)
    assert np.allclose(matrix, matrix.T)


def test_global_network_qap_is_deterministic() -> None:
    rng = np.random.default_rng(2)
    latent = rng.normal(size=(250, 3))
    loadings = rng.normal(size=(3, 7))
    matrices = {
        name: latent @ loadings + 0.1 * rng.normal(size=(250, 7))
        for name in ("a", "b", "c", "d")
    }
    raw = {
        name: kirhub.correlation_geometry(matrix, "raw")
        for name, matrix in matrices.items()
    }
    centered = {
        name: kirhub.correlation_geometry(matrix, "two_way_center")
        for name, matrix in matrices.items()
    }
    first = kirhub.paired_global_network_qap(raw, centered, 199, 11)
    second = kirhub.paired_global_network_qap(raw, centered, 199, 11)
    assert first == second
    assert first["raw_mean_pairwise_spearman"] > 0.95
    assert first["centered_mean_pairwise_spearman"] > 0.95


def test_partial_geometry_qap_is_deterministic() -> None:
    rng = np.random.default_rng(3)
    matrix = rng.normal(size=(200, 8))
    predictor = kirhub.correlation_geometry(matrix, "two_way_center")
    sequence = kirhub.correlation_geometry(
        rng.normal(size=(200, 8)), "two_way_center"
    )
    first = kirhub.partial_geometry_qap(predictor, predictor, sequence, 199, 13)
    second = kirhub.partial_geometry_qap(predictor, predictor, sequence, 199, 13)
    assert first == second
    assert first["partial_spearman"] > 0.999999


def test_subset_pair_vector_removes_incident_pairs() -> None:
    values = np.arange(10)  # five targets -> ten pairs
    keep = np.array([True, False, True, True, True])
    subset = kirhub.subset_pair_vector(values, keep)
    assert len(subset) == 6
    assert np.array_equal(subset, np.array([1, 2, 3, 7, 8, 9]))


def test_ligand_bootstrap_is_seed_deterministic() -> None:
    rng = np.random.default_rng(4)
    panels = {
        name: rng.normal(size=(40 + index, 6))
        for index, name in enumerate(("a", "b", "c", "d"))
    }
    first_summary, first_frame = kirhub.cross_panel_ligand_bootstrap(
        panels, 10, 17
    )
    second_summary, second_frame = kirhub.cross_panel_ligand_bootstrap(
        panels, 10, 17
    )
    assert first_summary == second_summary
    assert first_frame.equals(second_frame)


def test_transformation_matched_null_is_seed_deterministic() -> None:
    rng = np.random.default_rng(5)
    panels = {
        name: rng.normal(size=(45 + index, 6))
        for index, name in enumerate(("a", "b", "c", "KiRHub"))
    }
    labels = np.array(
        [
            True,
            False,
            False,
            True,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
        ]
    )
    first_summary, first_frame = (
        kirhub.transformation_matched_independent_column_null(
            panels, labels, 10, 19
        )
    )
    second_summary, second_frame = (
        kirhub.transformation_matched_independent_column_null(
            panels, labels, 10, 19
        )
    )
    assert first_summary == second_summary
    assert first_frame.equals(second_frame)


def test_release_summary_preserves_positive_result_and_boundary() -> None:
    with (kirhub.DEFAULT_OUTPUT / "summary.json").open() as handle:
        summary = json.load(handle)
    locked = summary["locked_primary_endpoint"]["KiRHub_validation"]
    assert locked["positive_pairs"] == 15
    assert locked["centered_KiRHub_geometry"]["roc_auc"] == pytest.approx(
        0.9634285714285714
    )
    matched = summary["experimental_cross_panel_agreement"][
        "transformation_matched_independent_column_null"
    ]
    assert matched["p_observed_centered_at_least_as_large"] < 0.001
    assert matched["p_observed_gain_at_least_as_large"] > 0.9
    pr = summary["KiRHub_PR_boundary"]
    assert (
        pr["two_way_centered_correlation_PR_dimension"]
        < pr["raw_correlation_PR_dimension"]
    )
    boundary = summary["KiRHub_specific_top10_boundary"]["metrics"]
    assert boundary["centered_minus_raw"]["roc_auc"] < 0
