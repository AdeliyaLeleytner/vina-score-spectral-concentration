"""Tests for the public experimental-map reliability analysis."""

from __future__ import annotations

import json
import builtins
import sys
from pathlib import Path

import numpy as np
import pytest


ANALYSIS = Path(__file__).resolve().parent
PACKAGE = ANALYSIS.parent
if str(ANALYSIS) not in sys.path:
    sys.path.insert(0, str(ANALYSIS))

import experimental_map_reliability as reliability  # noqa: E402


def synthetic_surface(rows: int = 400, targets: int = 7, seed: int = 17) -> np.ndarray:
    rng = np.random.default_rng(seed)
    latent = rng.normal(size=(rows, 3))
    loadings = rng.normal(size=(3, targets))
    ligand_effect = rng.normal(scale=2.0, size=(rows, 1))
    target_effect = rng.normal(scale=1.0, size=(1, targets))
    return ligand_effect + target_effect + latent @ loadings + rng.normal(
        scale=0.15, size=(rows, targets)
    )


def test_informative_rows_excludes_only_constant_profiles() -> None:
    matrix = np.asarray(
        [
            [5.0, 5.0, 5.0],
            [5.0, 5.1, 5.0],
            [0.0, 0.0, 0.0],
            [1.0, 2.0, 3.0],
        ]
    )
    assert reliability.informative_rows(matrix).tolist() == [False, True, False, True]


def test_two_way_surface_has_zero_row_and_column_means() -> None:
    centered = reliability.transformed_surface(synthetic_surface(80, 6), "two_way_centered")
    assert np.max(np.abs(centered.mean(axis=0))) < 1e-12
    assert np.max(np.abs(centered.mean(axis=1))) < 1e-12


def test_oas_and_empirical_maps_are_valid_correlations() -> None:
    matrix = synthetic_surface(100, 8)
    for estimator in ("empirical", "oas"):
        observed = reliability.target_map(matrix, estimator=estimator)
        assert observed.shape == (8, 8)
        assert np.allclose(observed, observed.T)
        assert np.allclose(np.diag(observed), 1.0)
        assert np.max(np.abs(observed)) <= 1.0 + 1e-12


def test_top_selection_and_overlap_are_deterministic_with_ties() -> None:
    values = np.asarray([0.0, 1.0, 1.0, 2.0, 2.0])
    selected = reliability.deterministic_top_indices(values, fraction=0.4)
    assert selected.tolist() == [3, 4]
    precision, jaccard = reliability.set_overlap(selected, np.asarray([2, 4]))
    assert precision == pytest.approx(0.5)
    assert jaccard == pytest.approx(1.0 / 3.0)


def test_cluster_disjoint_split_never_leaks_a_cluster() -> None:
    labels = np.asarray(["a", "a", "b", "c", "c", "d", "e", "f"], dtype=object)
    first, second = reliability.cluster_disjoint_halves(
        labels, np.random.default_rng(8)
    )
    assert set(labels[first]).isdisjoint(set(labels[second]))
    assert sorted(np.r_[first, second].tolist()) == list(range(len(labels)))


def test_split_half_reliability_recovers_a_stable_synthetic_map() -> None:
    matrix = synthetic_surface(600, 7)
    labels = np.asarray([f"cluster-{index // 3}" for index in range(len(matrix))])
    summary, rows = reliability.split_half_diagnostics(
        matrix,
        panel="synthetic",
        support="all",
        transform="two_way_centered",
        estimator="empirical",
        split_unit="butina_cluster",
        repeats=40,
        seed=9,
        cluster_labels=labels,
    )
    assert len(rows) == 40
    assert summary["valid_repeats"] == 40
    assert summary["split_map_spearman"]["median"] > 0.95
    assert summary["top_decile_precision_between_halves"]["median"] >= 0.5


def test_same_support_identity_has_unit_concordance() -> None:
    matrix = synthetic_surface(240, 7)
    clusters = np.arange(len(matrix))
    broad = reliability.target_map(matrix)
    bootstrap, records = reliability.paired_geometry_bootstrap(
        matrix,
        matrix,
        broad,
        panel="synthetic",
        support="identical",
        transform="two_way_centered",
        unit="ligand",
        repeats=30,
        seed=11,
        cluster_labels=clusters,
    )
    assert len(records) == 30
    assert bootstrap["point_estimate"]["same_support_spearman"] == pytest.approx(1.0)
    assert bootstrap["same_support_spearman"]["minimum"] == pytest.approx(1.0)


def test_public_module_has_no_restricted_analysis_dependency() -> None:
    source = (ANALYSIS / "experimental_map_reliability.py").read_text(encoding="utf-8")
    forbidden = (
        "matched_support_geometry",
        "residual_target_geometry_validation",
        "pkis1",
        "kirhub",
        "released_pair_geometry_ledger",
    )
    assert not any(token in source.lower() for token in forbidden)


def test_public_analysis_never_opens_a_restricted_historical_path(monkeypatch) -> None:
    original_open = builtins.open
    forbidden = ("pkis1", "kirhub", "released_pair_geometry_ledger")

    def guarded_open(file, *args, **kwargs):
        path = str(file).lower()
        if any(token in path for token in forbidden):
            raise AssertionError(f"restricted historical path opened: {file}")
        return original_open(file, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", guarded_open)
    summary, tables = reliability.analyse(split_repeats=10, bootstraps=10, seed=31)
    assert set(summary["primary_input_sha256"]) == {
        "DAVIS",
        "PKIS2",
        "DOCKSTRING",
        "DOCKSTRING_identity_contract",
    }
    assert "secondary_input_sha256" not in summary
    assert set(tables) == {
        "cross_panel_geometry.csv",
        "edge_stability.csv",
        "map_reliability.csv",
        "same_support_bootstrap.csv",
        "same_support_geometry.csv",
        "same_support_split_half.csv",
    }


def test_frozen_output_contract_if_materialized() -> None:
    """Lock public support and headline values once production output exists."""
    path = PACKAGE / "results" / "experimental_map_reliability" / "summary.json"
    if not path.exists():
        pytest.skip("production reliability output has not been materialized")
    summary = json.loads(path.read_text(encoding="utf-8"))
    assert summary["central_data_boundary"].startswith("Only repository-redistributed")
    assert "secondary_input_sha256" not in summary
    assert "locked_endpoint_stability" not in summary
    assert summary["broad_reference"]["rows"] == 259_806
    assert len(summary["cross_panel_experimental_geometry"]) == 2
    expected = {"DAVIS": (59, 56), "PKIS2": (154, 154)}
    for panel, (exact, informative) in expected.items():
        record = summary["panels"][panel]
        assert record["exact_matches"] == exact
        assert record["primary_same_support_informative_ligands"] == informative
        primary = [
            item
            for item in record["same_support_geometry"]
            if item["support"] == "informative_exact_matches_primary"
            and item["transform"] == "two_way_centered"
        ]
        assert len(primary) == 1
        assert primary[0]["targets"] == 21
        assert primary[0]["target_pairs"] == 210
    assert summary["panels"]["DAVIS"]["same_support_geometry"][0][
        "same_support_empirical_spearman"
    ] == pytest.approx(0.30656542260433506, abs=1e-14)
    assert summary["panels"]["PKIS2"]["same_support_geometry"][0][
        "same_support_empirical_spearman"
    ] == pytest.approx(0.3260825480331591, abs=1e-14)
