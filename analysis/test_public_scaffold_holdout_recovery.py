from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import public_scaffold_holdout_recovery as mod


PACKAGE = Path(__file__).resolve().parents[1]
RESULTS = PACKAGE / "results" / "public_scaffold_holdout_recovery"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_target_map_is_invariant_to_row_offsets() -> None:
    rng = np.random.default_rng(4)
    matrix = rng.normal(size=(100, 7))
    offsets = rng.normal(size=(100, 1))
    observed = mod.target_correlation(matrix)
    shifted = mod.target_correlation(matrix + offsets)
    assert np.allclose(observed, shifted, atol=1e-12, rtol=0)


def test_group_fold_analysis_has_zero_leakage() -> None:
    rng = np.random.default_rng(5)
    matrix = rng.normal(size=(1_000, 6))
    groups = np.repeat(np.arange(100), 10).astype(object)
    smiles = pd.Series([f"{'C' * (2 + (index % 8))}O" for index in range(1_000)])
    _, fingerprints, _ = mod.chemical_groups_and_fingerprints(
        smiles, np.arange(len(smiles))
    )
    frame, similarity, folds = mod.analyze_dataset(
        "synthetic",
        matrix,
        groups,
        fingerprints,
        np.arange(len(matrix)),
        seed=9,
    )
    assert len(frame) == 2 * mod.FOLDS * len(mod.CALIBRATION_SIZES) * mod.REPETITIONS_PER_FOLD
    group = frame.loc[frame.holdout_design.eq("chemical_group")]
    assert group.chemical_group_overlap.max() == 0
    assert set(frame.holdout_design) == {"chemical_group", "matched_random_row"}
    assert np.isfinite(frame.geometry_spearman_to_holdout).all()
    assert len(similarity) == 2 * len(matrix)
    assert len(folds) == 2 * mod.FOLDS
    assert similarity.maximum_tanimoto_to_calibration_pool.between(0, 1).all()


def test_acyclic_duplicates_share_connectivity_group() -> None:
    smiles = pd.Series(["CCO", "OCC", "CCCO", "C1CC1", "C1(C)CC1"])
    groups, fingerprints, diagnostics = mod.chemical_groups_and_fingerprints(
        smiles, np.arange(len(smiles))
    )
    assert groups[0] == groups[1]
    assert groups[0].startswith("ACYCLIC_CONNECTIVITY_")
    assert groups[3].startswith("CYCLIC_MURCKO:")
    assert len(fingerprints) == len(smiles)
    assert diagnostics["acyclic_rows"] == 3
    assert diagnostics["duplicate_rows_beyond_one_per_group"] >= 1


def test_similarity_summary_reports_declared_thresholds() -> None:
    records = pd.DataFrame(
        {
            "dataset": ["x"] * 4,
            "holdout_design": ["chemical_group"] * 4,
            "fold": [1] * 4,
            "calibration_pool_ligands": [10] * 4,
            "maximum_tanimoto_to_calibration_pool": [0.4, 0.6, 0.8, 1.0],
        }
    )
    summary = mod.similarity_summary(records).iloc[0]
    assert summary.exact_pairwise_comparisons == 40
    assert summary["fraction_at_least_0.5"] == 0.75
    assert summary["fraction_at_least_0.9"] == 0.25


def test_imputation_is_calibration_pool_local() -> None:
    pool = np.array([[1.0, np.nan], [3.0, 5.0], [5.0, 7.0]])
    evaluation = np.array([[100.0, np.nan], [200.0, 101.0]])
    filled_pool, filled_evaluation, means = mod.fill_from_calibration_pool(
        pool, evaluation
    )
    assert np.allclose(means, [3.0, 6.0])
    assert filled_evaluation[0, 1] == 6.0
    changed = evaluation.copy()
    changed[1, 1] = 1_000_000.0
    _, _, changed_means = mod.fill_from_calibration_pool(pool, changed)
    assert np.array_equal(means, changed_means)
    assert np.isfinite(filled_pool).all()


def test_production_recovery_matches_frozen_contract() -> None:
    frame = pd.read_csv(RESULTS / "scaffold_holdout_summary.csv")
    lookup = frame.set_index(["dataset", "calibration_ligands"])
    assert np.isclose(
        lookup.loc[("DOCKSTRING-58", 200), "geometry_spearman_mean"],
        0.9143453532335925,
    )
    assert np.isclose(
        lookup.loc[("DOCKSTRING-58", 500), "geometry_spearman_mean"],
        0.9583082201385786,
    )
    assert np.isclose(
        lookup.loc[("Docking-44", 200), "geometry_spearman_mean"],
        0.9249615933122916,
    )
    assert np.isclose(
        lookup.loc[("Docking-44", 500), "geometry_spearman_mean"],
        0.9578565088096582,
    )
    assert (frame.repetitions == 125).all()
    assert frame.maximum_group_overlap.max() == 0

    random_summary = pd.read_csv(
        RESULTS / "matched_random_row_holdout_summary.csv"
    )
    paired = pd.read_csv(RESULTS / "chemical_group_minus_random_summary.csv")
    assert len(random_summary) == len(frame) == 4
    assert len(paired) == 4
    assert (paired.paired_repetitions == 125).all()
    random_lookup = random_summary.set_index(["dataset", "calibration_ligands"])
    assert np.isclose(
        random_lookup.loc[("Docking-44", 200), "geometry_spearman_mean"],
        0.9353939183306044,
    )
    assert np.isclose(
        random_lookup.loc[("Docking-44", 500), "geometry_spearman_mean"],
        0.9684045644643468,
    )


def test_production_support_and_checksums_are_authenticated() -> None:
    summary = json.loads((RESULTS / "summary.json").read_text())
    assert summary["analysis_status"] == "strict_public_chemical_group_holdout_sensitivity"
    assert summary["datasets"]["Docking-44"]["chemical_groups"] == 5_112
    assert summary["datasets"]["DOCKSTRING-58"]["chemical_groups"] == 11_903
    assert "calibration-pool target means" in summary["datasets"]["Docking-44"]["imputation_rule"]
    assert (
        summary["datasets"]["DOCKSTRING-58"]["selected_complete_row_index_sha256"]
        == "8afe6ccc883bd5d376847ddc4bfe0fff3b8cba1a518e4d8941f1e50a9356b11d"
    )
    manifest = json.loads((RESULTS / "output_checksums.json").read_text())
    for filename, expected in manifest["files"].items():
        assert _sha256(RESULTS / filename) == expected
    nearest = pd.read_csv(RESULTS / "heldout_maximum_morgan_tanimoto_summary.csv")
    assert "fraction_tanimoto_equal_one" in nearest
    assert "fraction_exact_match" not in nearest
    assert set(nearest.holdout_design) == {
        "chemical_group",
        "matched_random_row",
    }
    folds = pd.read_csv(RESULTS / "chemical_group_fold_diagnostics.csv")
    assert len(folds) == 2 * 2 * mod.FOLDS
    group_folds = folds.loc[folds.holdout_design.eq("chemical_group")]
    assert group_folds.chemical_group_overlap.max() == 0


def test_producer_has_no_restricted_branch_imports() -> None:
    source = (PACKAGE / "analysis" / "public_scaffold_holdout_recovery.py").read_text()
    assert "from residual_" not in source
    assert "from replicated_" not in source
    assert "PKIS1" not in source
    assert "KiRHub" not in source
