from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import numpy as np

import public_chemical_domain_controls as domain


def test_spearman_and_geometry_identity() -> None:
    values = np.asarray([3.0, 1.0, 4.0, 2.0])
    assert np.isclose(domain.spearman(values, values), 1.0)
    assert np.isclose(domain.spearman(values, -values), -1.0)
    matrix = np.asarray(
        [[1.0, 0.5, -0.2], [0.5, 1.0, 0.1], [-0.2, 0.1, 1.0]]
    )
    comparison = domain.geometry_comparison(matrix, matrix)
    assert comparison["geometry_spearman"] == 1.0
    assert comparison["sign_flip_fraction"] == 0.0


def test_rank_strata_and_group_split_are_deterministic_and_disjoint() -> None:
    sizes = np.asarray([1, 2, 3, 1, 2] * 4)
    medians = np.linspace(100.0, 500.0, len(sizes))
    first = domain.balanced_group_split(
        sizes, medians, np.random.default_rng(9), strata=5
    )
    second = domain.balanced_group_split(
        sizes, medians, np.random.default_rng(9), strata=5
    )
    np.testing.assert_array_equal(first, second)
    assert set(first.tolist()) == {0, 1}


def test_murcko_grouping_makes_acyclic_molecules_singletons() -> None:
    smiles = np.asarray(["CCO", "CCO", "c1ccccc1", "c1ccccc1"])
    groups = domain.murcko_groups(smiles, np.asarray([10, 11, 12, 13]))
    assert groups[0] != groups[1]
    assert groups[2] == groups[3]


def test_group_controls_have_no_group_leakage() -> None:
    rng = np.random.default_rng(12)
    rows = 240
    targets = 6
    latent = rng.normal(size=(rows, 2))
    matrix = latent @ rng.normal(size=(2, targets)) + 0.2 * rng.normal(
        size=(rows, targets)
    )
    molecular_weight = np.linspace(100.0, 600.0, rows)
    groups = np.asarray([f"g{index // 3}" for index in range(rows)])
    global_control, within = domain.chemical_domain_controls(
        "synthetic",
        matrix,
        molecular_weight,
        groups,
        repetitions=3,
        seed=21,
    )
    assert (global_control.chemical_group_overlap == 0).all()
    group_within = within.loc[
        within.control_type.eq("mw_matched_chemical_group_disjoint")
    ]
    assert (group_within.chemical_group_overlap == 0).all()
    assert set(within.mw_band) == {"low_mw", "high_mw"}
    assert set(within.transformation) == {"raw", "row_centered_residual"}


def test_producer_ast_is_strict_public() -> None:
    source = Path(domain.__file__).read_text(encoding="utf-8")
    prohibited = (
        "chemical_context_geometry",
        "residual_mechanism_analysis",
        "build_evidence",
        "evidence_summary.json",
        "manuscript_evidence.json",
        "public_claim_reproducibility_ledger.json",
    )
    assert not any(name in source for name in prohibited)
    tree = ast.parse(source)
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".")[0])
    assert roots <= {
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


def test_released_artifact_contract_and_checksums() -> None:
    output = domain.DEFAULT_OUTPUT
    summary_path = output / "summary.json"
    if not summary_path.exists():
        return
    summary = json.loads(summary_path.read_text())
    assert summary["schema_version"] == "1.0.0"
    assert summary["configuration"]["dockstring_support_seeds"] == [
        71,
        72,
        73,
        74,
        75,
        20260809,
    ]
    assert summary["configuration"]["control_repetitions"] == 200
    manifest = json.loads((output / "output_checksums.json").read_text())
    for filename, expected in manifest["files"].items():
        assert hashlib.sha256((output / filename).read_bytes()).hexdigest() == expected
