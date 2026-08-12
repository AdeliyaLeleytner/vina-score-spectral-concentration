from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np

import revision_map_decision_sensitivities as revision


def test_stable_top_mask_selects_exact_count_with_ties() -> None:
    values = np.asarray([0.0, 1.0, 1.0, -1.0, -1.0])
    largest = revision.stable_top_mask(values, 2, largest=True)
    smallest = revision.stable_top_mask(values, 2, largest=False)
    assert int(largest.sum()) == int(smallest.sum()) == 2
    assert set(np.flatnonzero(largest)) == {1, 2}
    assert set(np.flatnonzero(smallest)) == {3, 4}


def test_edge_recovery_is_exact_for_identical_map() -> None:
    geometry = np.asarray(
        [
            [1.0, 0.8, -0.2, 0.1],
            [0.8, 1.0, -0.4, 0.3],
            [-0.2, -0.4, 1.0, 0.7],
            [0.1, 0.3, 0.7, 1.0],
        ]
    )
    metrics = revision.edge_recovery_metrics(geometry, geometry, edge_counts=(2,))
    assert metrics["geometry_spearman"] == 1.0
    assert metrics["edge_sign_agreement"] == 1.0
    assert metrics["reference_abs_top10pct_sign_agreement"] == 1.0
    assert metrics["top_positive_2_precision"] == 1.0
    assert metrics["top_negative_2_recall"] == 1.0


def test_quantile_bins_are_complete_and_boundary_ties_go_up() -> None:
    values = np.arange(100.0)
    identifiers, edges = revision.quantile_bin_ids(values, 10)
    assert set(identifiers) == set(range(10))
    assert len(edges) == 11
    # Search-right is the declared deterministic boundary contract.
    assert np.searchsorted(edges[1:-1], edges[1], side="right") == 1


def test_mw_continuum_detects_progressive_geometry_change() -> None:
    rng = np.random.default_rng(11)
    rows = 1_000
    mw = np.linspace(100.0, 600.0, rows)
    latent = rng.normal(size=rows)
    noise = rng.normal(scale=0.25, size=(rows, 4))
    loading = np.column_stack(
        [
            np.ones(rows),
            np.linspace(1.0, -1.0, rows),
            -np.ones(rows),
            np.linspace(-1.0, 1.0, rows),
        ]
    )
    matrix = latent[:, None] * loading + noise
    descriptors = np.zeros((rows, len(revision.DESCRIPTOR_NAMES)))
    descriptors[:, revision.DESCRIPTOR_NAMES.index("molecular_weight")] = mw
    bundle = SimpleNamespace(
        name="synthetic",
        matrix=matrix,
        descriptor_matrix=descriptors,
    )
    bins, pairs, trends = revision.mw_decile_continuum(bundle, bins=10)
    assert len(bins) == 20
    assert len(pairs) == 90
    residual = trends[trends.transformation.eq("row_centered_residual")].iloc[0]
    assert residual.spearman_mw_separation_vs_map_dissimilarity > 0.5


def test_sufficient_statistics_reproduce_and_subtract_residual_geometry() -> None:
    rng = np.random.default_rng(23)
    matrix = rng.normal(size=(120, 7))
    full_sum, full_cross = revision.row_centered_statistics(matrix, chunk_size=31)
    full = revision.correlation_from_sufficient_statistics(
        len(matrix), full_sum, full_cross
    )
    np.testing.assert_allclose(
        full,
        revision.target_geometry(matrix, "row_centered_residual"),
        atol=1e-12,
    )
    selected = np.arange(20)
    selected_sum, selected_cross = revision.residual_subset_statistics(matrix[selected])
    complement = revision.correlation_from_sufficient_statistics(
        len(matrix) - len(selected),
        full_sum - selected_sum,
        full_cross - selected_cross,
    )
    np.testing.assert_allclose(
        complement,
        revision.target_geometry(matrix[20:], "row_centered_residual"),
        atol=1e-12,
    )


def test_mw_controls_are_equal_n_and_cover_every_threshold() -> None:
    rng = np.random.default_rng(29)
    rows = 100
    matrix = rng.normal(size=(rows, 5))
    descriptors = np.zeros((rows, len(revision.DESCRIPTOR_NAMES)))
    descriptors[:, revision.DESCRIPTOR_NAMES.index("molecular_weight")] = np.linspace(
        100, 500, rows
    )
    bundle = SimpleNamespace(
        name="synthetic",
        matrix=matrix,
        descriptor_matrix=descriptors,
    )
    fractions = (0.1, 0.2)
    observed = revision.mw_threshold_sweep(bundle, fractions)
    controls = revision.mw_equal_n_random_disjoint_controls(
        bundle, repetitions=3, seed=31, fractions=fractions
    )
    assert len(controls) == 3 * len(fractions) * 2
    assert (controls.ligands_per_support == controls.tail_fraction_per_side * rows).all()
    merged = revision.attach_mw_control_summary(observed, controls)
    assert len(merged) == len(observed)
    assert (merged.control_repetitions == 3).all()


def test_deterministic_pam_covers_two_separated_blocks() -> None:
    distance = np.asarray(
        [
            [0.0, 0.1, 1.8, 1.9],
            [0.1, 0.0, 1.9, 1.8],
            [1.8, 1.9, 0.0, 0.1],
            [1.9, 1.8, 0.1, 0.0],
        ]
    )
    targets = ["A", "B", "C", "D"]
    first = revision.deterministic_pam(distance, 2, targets)
    second = revision.deterministic_pam(distance, 2, targets)
    assert np.array_equal(first, second)
    assert len(set(first).intersection({0, 1})) == 1
    assert len(set(first).intersection({2, 3})) == 1
    assert revision.panel_coverage_objective(distance, first)[
        "mean_nearest_distance"
    ] == 0.05
    exact = revision.exact_k_medoids(distance, 2)
    assert revision.panel_coverage_objective(distance, exact)[
        "mean_nearest_distance"
    ] == 0.05


def test_hierarchical_labels_recover_identical_partition() -> None:
    distance = np.asarray(
        [
            [0.0, 0.1, 1.8, 1.9],
            [0.1, 0.0, 1.9, 1.8],
            [1.8, 1.9, 0.0, 0.1],
            [1.9, 1.8, 0.1, 0.0],
        ]
    )
    labels = revision.hierarchical_labels(distance, 2)
    assert labels[0] == labels[1]
    assert labels[2] == labels[3]
    assert labels[0] != labels[2]


def test_released_revision_artifact_has_bounded_claims() -> None:
    metadata_path = revision.DEFAULT_OUTPUT / "metadata.json"
    if not metadata_path.exists():
        return
    metadata = json.loads(metadata_path.read_text())
    assert metadata["schema_version"] == "1.2.0"
    assert metadata["analysis_status"] == "exploratory_revision_sensitivity"
    assert metadata["recovery"]["sizes"] == [200, 500]
    assert metadata["molecular_weight_support"]["quantile_bins"] == 10
    assert "do not identify molecular weight as causal" in metadata["claim_boundary"]
