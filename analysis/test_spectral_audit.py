from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from build_evidence import (
    additive_main_effect_null,
    empirical_residual_permutation_null,
    row_norm_preserving_residual_null,
)
from spectral_audit import map_recovery_curve


def test_residual_nulls_share_one_fingerprinted_support_when_seed_matches() -> None:
    matrix = np.random.default_rng(44).normal(size=(90, 7))
    common = {"sample_size": 40, "repeats": 3, "seed": 17}
    additive = additive_main_effect_null(matrix, **common)
    empirical = empirical_residual_permutation_null(matrix, **common)
    row_norm = row_norm_preserving_residual_null(matrix, **common)

    supports = [
        additive["support_selection"],
        empirical["support_selection"],
        row_norm["support_selection"],
    ]
    assert len({support["support_index_sha256"] for support in supports}) == 1
    assert all(support["sample_rows"] == 40 for support in supports)
    observed = [
        additive["observed"]["residual"],
        empirical["observed_residual"],
        row_norm["observed_residual"],
    ]
    assert max(observed) - min(observed) < 1e-12


def test_released_common_support_nulls_are_exposed_in_reports() -> None:
    results = Path(__file__).resolve().parents[1] / "results"
    evidence_path = results / "evidence_summary.json"
    registry_path = results / "reported_values.json"
    table_path = results / "supplement_tables.tex"
    if not all(path.exists() for path in (evidence_path, registry_path, table_path)):
        pytest.skip("released evidence reports have not been generated")

    evidence = json.loads(evidence_path.read_text())
    for dataset in ("docking44", "dockstring58"):
        blocks = [
            evidence[key][dataset]
            for key in (
                "additive_main_effect_null",
                "empirical_residual_permutation_null",
                "row_norm_preserving_residual_null",
            )
        ]
        assert len(
            {
                block["support_selection"]["support_index_sha256"]
                for block in blocks
            }
        ) == 1

    macros = json.loads(registry_path.read_text())["macros"]
    assert macros["DffCommonNullObservedPR"] == "9.295"
    assert macros["DsCommonNullObservedPR"] == "18.158"
    assert macros["DffRowNull"] == "42.628"
    assert macros["DsRowNull"] == "56.612"
    table = table_path.read_text()
    assert "Docking-44 & row-norm preserving & 9.295 & 42.628" in table
    assert "DOCKSTRING-58 & row-norm preserving & 18.158 & 56.612" in table


def test_map_recovery_curve_is_deterministic_and_support_bounded() -> None:
    rng = np.random.default_rng(74)
    latent = rng.normal(size=(240, 3))
    loadings = rng.normal(size=(3, 8))
    matrix = latent @ loadings + 0.25 * rng.normal(size=(240, 8))
    first = map_recovery_curve(matrix, [20, 60, 500], repeats=12, seed=9)
    second = map_recovery_curve(matrix, [20, 60, 500], repeats=12, seed=9)
    assert first == second
    assert set(first["points"]) == {"20", "60"}
    assert first["reference_ligands"] == 240
    assert first["targets"] == 8
    for point in first["points"].values():
        assert -1 <= point["q025_spearman"] <= point["mean_spearman"] <= 1
        assert -1 <= point["q975_spearman"] <= 1
    assert "not" in first["interpretation_boundary"]


def test_map_recovery_curve_rejects_invalid_contracts() -> None:
    matrix = np.arange(60, dtype=float).reshape(20, 3)
    with pytest.raises(ValueError, match="positive"):
        map_recovery_curve(matrix, [10], repeats=0, seed=1)
    with pytest.raises(ValueError, match="three target"):
        map_recovery_curve(matrix[:, :2], [10], repeats=2, seed=1)

    with pytest.raises(ValueError, match="three ligand"):
        map_recovery_curve(matrix[:2], [3], repeats=2, seed=1)

    valid = np.random.default_rng(2).normal(size=(20, 3))
    with pytest.raises(ValueError, match="no valid recovery size"):
        map_recovery_curve(valid, [2, 25], repeats=2, seed=1)

    degenerate = np.add.outer(
        np.arange(20, dtype=float), np.arange(3, dtype=float)
    )
    with pytest.raises(ValueError, match="non-zero variance"):
        map_recovery_curve(degenerate, [10], repeats=2, seed=1)


def test_cli_writes_strict_json_with_same_support_recovery(tmp_path: Path) -> None:
    rng = np.random.default_rng(15)
    latent = rng.normal(size=(80, 2))
    matrix = latent @ rng.normal(size=(2, 5)) + 0.2 * rng.normal(size=(80, 5))
    input_path = tmp_path / "scores.csv"
    output_path = tmp_path / "audit.json"
    pd.DataFrame(matrix, columns=[f"target_{index}" for index in range(5)]).to_csv(
        input_path, index=False
    )
    script = Path(__file__).with_name("spectral_audit.py")
    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            str(input_path),
            "--output",
            str(output_path),
            "--sample-size",
            "60",
            "--permutations",
            "5",
            "--bootstrap",
            "5",
            "--recovery-sizes",
            "20,50,500",
            "--recovery-repeats",
            "4",
            "--seed",
            "11",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    serialized = output_path.read_text()
    assert "NaN" not in serialized and "Infinity" not in serialized
    payload = json.loads(serialized)
    recovery = payload["same_support_map_recovery"]
    assert set(recovery["points"]) == {"20", "50"}
    assert recovery["reference_ligands"] == 80
    null_keys = (
        "additive_gaussian_null",
        "empirical_residual_permutation_null",
        "row_norm_preserving_residual_null",
    )
    assert len(
        {
            payload[key]["support_selection"]["support_index_sha256"]
            for key in null_keys
        }
    ) == 1
    assert str(output_path) in completed.stdout

    invalid_path = tmp_path / "scores_with_infinity.csv"
    invalid = matrix.copy()
    invalid[0, 0] = np.inf
    pd.DataFrame(invalid, columns=[f"target_{index}" for index in range(5)]).to_csv(
        invalid_path, index=False
    )
    rejected = subprocess.run(
        [
            sys.executable,
            str(script),
            str(invalid_path),
            "--output",
            str(tmp_path / "invalid.json"),
            "--permutations",
            "5",
            "--bootstrap",
            "5",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert rejected.returncode != 0
    assert "finite values" in rejected.stderr

    two_target_path = tmp_path / "two_targets.csv"
    pd.DataFrame(matrix[:, :2], columns=["target_a", "target_b"]).to_csv(
        two_target_path, index=False
    )
    two_target = subprocess.run(
        [
            sys.executable,
            str(script),
            str(two_target_path),
            "--output",
            str(tmp_path / "two_target.json"),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert two_target.returncode != 0
    assert "at least three target score columns" in two_target.stderr
