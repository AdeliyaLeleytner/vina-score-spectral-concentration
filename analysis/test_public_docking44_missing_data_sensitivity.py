from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import numpy as np

import public_docking44_missing_data_sensitivity as audit


def test_observed_cell_fit_matches_two_way_center_when_complete() -> None:
    matrix = np.random.default_rng(4).normal(size=(40, 7))
    residual, diagnostics = audit.observed_cell_additive_residual(matrix)
    np.testing.assert_allclose(residual, audit.two_way_center(matrix), atol=1e-11)
    assert diagnostics["converged"]
    assert diagnostics["maximum_absolute_observed_row_residual_mean"] < 1e-11
    assert diagnostics["maximum_absolute_observed_target_residual_mean"] < 1e-11


def test_observed_cell_fit_respects_mask_and_normal_equations() -> None:
    matrix = np.random.default_rng(7).normal(size=(50, 6))
    matrix[::4, 2] = np.nan
    matrix[3::7, 5] = np.nan
    residual, diagnostics = audit.observed_cell_additive_residual(matrix)
    assert np.all(residual[~np.isfinite(matrix)] == 0.0)
    assert diagnostics["maximum_absolute_observed_row_residual_mean"] < 1e-10
    assert diagnostics["maximum_absolute_observed_target_residual_mean"] < 1e-10


def test_equal_n_controls_are_deterministic_and_have_requested_length() -> None:
    matrix = np.random.default_rng(9).normal(size=(80, 8))
    first = audit.equal_n_random_support_controls(
        matrix, selected_rows=55, repetitions=6, seed=12
    )
    second = audit.equal_n_random_support_controls(
        matrix, selected_rows=55, repetitions=6, seed=12
    )
    np.testing.assert_array_equal(first, second)
    assert len(first) == 6
    assert np.isfinite(first).all()


def test_cliffs_delta_direction_and_ties() -> None:
    assert audit.cliffs_delta(np.asarray([3.0, 4.0]), np.asarray([1.0, 2.0])) == 1.0
    assert audit.cliffs_delta(np.asarray([1.0, 2.0]), np.asarray([3.0, 4.0])) == -1.0
    assert audit.cliffs_delta(np.asarray([1.0, 1.0]), np.asarray([1.0, 1.0])) == 0.0


def test_analyze_separates_imputation_from_complete_row_support() -> None:
    rng = np.random.default_rng(14)
    matrix = rng.normal(size=(90, 8))
    matrix[::5, 3] = np.nan
    loaded = {
        "matrix": matrix,
        "complete_mask": np.isfinite(matrix).all(axis=1),
        "descriptors": {
            "molecular_weight": rng.normal(300, 40, 90),
            "heavy_atoms": rng.integers(10, 35, 90).astype(float),
            "rotatable_bonds": rng.integers(0, 10, 90).astype(float),
            "logp": rng.normal(2, 1, 90),
        },
        "positive_cells_clipped": 0,
        "invalid_smiles_for_heavy_atoms": 0,
    }
    summary, tables = audit.analyze(loaded, random_controls=5, seed=17)
    spectral = tables["spectral_sensitivity.csv"].set_index("analysis")
    complete = spectral.loc["complete_rows_observed_scores"]
    restricted = spectral.loc["target_mean_imputed_restricted_to_complete_rows"]
    assert complete.residual_pr == restricted.residual_pr
    assert complete.raw_pr == restricted.raw_pr
    assert summary["complete_row_identity_check"][
        "maximum_absolute_score_difference_between_observed_complete_rows_and_primary_imputed_restricted_rows"
    ] == 0.0


def test_producer_ast_closure_and_no_ledger_dependency() -> None:
    source = Path(audit.__file__).read_text(encoding="utf-8")
    prohibited = (
        "build_evidence",
        "evidence_summary.json",
        "manuscript_evidence.json",
        "public_claim_reproducibility_ledger.json",
    )
    assert not any(name in source for name in prohibited)
    tree = ast.parse(source)
    imported_roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".")[0])
    assert imported_roots <= {
        "__future__",
        "argparse",
        "hashlib",
        "json",
        "pathlib",
        "typing",
        "numpy",
        "pandas",
        "rdkit",
    }


def test_released_artifact_and_checksums() -> None:
    output = audit.DEFAULT_OUTPUT
    summary_path = output / "summary.json"
    if not summary_path.exists():
        return
    summary = json.loads(summary_path.read_text())
    assert summary["schema_version"] == "1.0.0"
    assert summary["matrix"]["rows"] == 12_651
    assert summary["equal_n_random_support_control"]["repetitions"] == 1_000
    assert summary["chemistry"]["no_hypothesis_tests"] is True
    manifest = json.loads((output / "output_checksums.json").read_text())
    for filename, expected in manifest["files"].items():
        actual = hashlib.sha256((output / filename).read_bytes()).hexdigest()
        assert actual == expected
