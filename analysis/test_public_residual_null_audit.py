from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np

import public_residual_null_audit as audit


def synthetic_matrix(seed: int, rows: int, targets: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    ligand = rng.normal(size=(rows, 1))
    interactions = rng.normal(size=(rows, 2)) @ rng.normal(size=(2, targets))
    return ligand + interactions + 0.2 * rng.normal(size=(rows, targets))


def test_three_nulls_share_identical_fingerprinted_support() -> None:
    report, conditional = audit.analyze_matrix(
        synthetic_matrix(3, 90, 7),
        np.linspace(100, 600, 90),
        sample_size=40,
        repeats=3,
        seed=17,
    )
    blocks = [
        report["nulls"]["additive_gaussian"],
        report["nulls"]["empirical_residual_column_permutation"],
        report["nulls"]["row_norm_preserving_random_direction"],
        *report["nulls"]["molecular_weight_conditional_permutation"].values(),
    ]
    supports = {
        block["support_selection"]["support_index_sha256"] for block in blocks
    }
    assert len(supports) == 1
    observed = [
        report["nulls"]["additive_gaussian"]["observed"]["residual"],
        report["nulls"]["empirical_residual_column_permutation"][
            "observed_residual"
        ],
        report["nulls"]["row_norm_preserving_random_direction"][
            "observed_residual"
        ],
    ]
    assert np.ptp(observed) < 1e-12
    expected_designs = 1 + 2 * (len(audit.MW_BIN_COUNTS) - 1)
    assert len(conditional) == 3 * expected_designs
    assert set(conditional.requested_mw_bins) == set(audit.MW_BIN_COUNTS)


def test_molecular_weight_bins_keep_exact_ties_together() -> None:
    weights = np.repeat(np.arange(10, dtype=float), 5)
    labels, metadata = audit.molecular_weight_bins(weights, requested_bins=5)
    for value in np.unique(weights):
        assert len(np.unique(labels[weights == value])) == 1
    assert sum(metadata["bin_sizes"]) == len(weights)


def test_conditional_null_reports_map_metrics() -> None:
    matrix = synthetic_matrix(8, 100, 8)
    block, table = audit.molecular_weight_conditional_permutation_null(
        matrix,
        np.linspace(100, 500, len(matrix)),
        sample_size=60,
        repeats=4,
        seed=23,
        requested_bins=5,
    )
    assert len(table) == 8
    assert block["molecular_weight"]["actual_bins"] == 5
    assert block["molecular_weight"]["minimum_bin_size"] >= 2
    assert {
        "participation_ratio",
        "pc1_fraction",
        "mean_off_diagonal_correlation",
        "mean_absolute_off_diagonal_correlation",
        "rms_off_diagonal_correlation",
        "maximum_absolute_off_diagonal_correlation",
        "spearman_to_observed_map",
        "edge_sign_agreement_to_observed_map",
        "top_absolute_edge_jaccard_10",
        "top_absolute_edge_jaccard_25",
    } <= set(table)


def test_artifact_is_strict_and_checksums_validate(tmp_path: Path) -> None:
    matrices = {
        "docking44": synthetic_matrix(4, 70, 6),
        "dockstring58": synthetic_matrix(5, 75, 7),
    }
    preprocessing = {
        name: {"rule": "synthetic test fixture"} for name in matrices
    }
    weights = {
        name: np.linspace(100, 500, len(matrix))
        for name, matrix in matrices.items()
    }
    payload, repeat_table = audit.build_artifact(
        matrices,
        weights,
        preprocessing,
        sample_size=50,
        repeats=3,
        input_records={
            "docking44": {"path": "fixture_a", "sha256": "a" * 64},
            "dockstring58": {"path": "fixture_b", "sha256": "b" * 64},
        },
    )
    audit.write_artifact(tmp_path, payload, repeat_table)
    serialized = (tmp_path / "summary.json").read_text()
    assert "NaN" not in serialized and "Infinity" not in serialized
    parsed = json.loads(serialized)
    assert parsed["analysis_status"] == "standalone_public_reproducibility_artifact"
    manifest = json.loads((tmp_path / "output_checksums.json").read_text())
    for filename, expected in manifest["files"].items():
        assert audit.sha256_file(tmp_path / filename) == expected


def test_producer_source_does_not_read_historical_ledgers() -> None:
    source = Path(audit.__file__).read_text(encoding="utf-8")
    prohibited = (
        "build_evidence",
        "evidence_summary.json",
        "manuscript_evidence.json",
        "public_claim_reproducibility_ledger.json",
    )
    assert not any(name in source for name in prohibited)


def test_producer_ast_import_closure_is_declared_public_dependencies_only() -> None:
    source = Path(audit.__file__).read_text(encoding="utf-8")
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


def test_released_artifact_has_public_input_only_contract() -> None:
    summary_path = audit.DEFAULT_OUTPUT / "summary.json"
    if not summary_path.exists():
        return
    payload = json.loads(summary_path.read_text())
    assert payload["schema_version"] == "1.0.0"
    assert payload["configuration"]["sample_size_per_dataset"] == 12_000
    assert set(payload["inputs"]) == {"docking44", "dockstring58"}
    assert all(
        record["path"].startswith("data/frozen/")
        for record in payload["inputs"].values()
    )
    for dataset in payload["datasets"].values():
        direct = [
            dataset["nulls"]["additive_gaussian"],
            dataset["nulls"]["empirical_residual_column_permutation"],
            dataset["nulls"]["row_norm_preserving_random_direction"],
            *dataset["nulls"]["molecular_weight_conditional_permutation"].values(),
        ]
        hashes = {
            null["support_selection"]["support_index_sha256"]
            for null in direct
        }
        assert len(hashes) == 1
    expected = {
        "docking44": (34.87523822972109, 0.23751160133616508),
        "dockstring58": (47.03913866728567, 0.2511928307519498),
    }
    for name, (median, explained_share) in expected.items():
        primary = payload["datasets"][name]["nulls"][
            "molecular_weight_conditional_permutation"
        ]["mw_10_bins"]
        observed = primary["null_distributions_by_bin_design"][
            "molecular_weight_quantile"
        ]["participation_ratio"]["median"]
        observed_share = primary["comparisons"][
            "molecular_weight_quantile"
        ]["explained_share_of_observed_pr_gap_relative_to_one_bin"]
        random_share = primary["comparisons"][
            "size_matched_random_partition"
        ]["explained_share_of_observed_pr_gap_relative_to_one_bin"]
        assert np.isclose(observed, median)
        assert np.isclose(observed_share, explained_share)
        assert abs(random_share) < 1e-3
        assert "processed raw target-score" in primary["model"]
    repeat_table = np.genfromtxt(
        audit.DEFAULT_OUTPUT / "mw_conditional_null_repeats.csv",
        delimiter=",",
        names=True,
        dtype=None,
        encoding="utf-8",
    )
    expected_designs = 1 + 2 * (len(audit.MW_BIN_COUNTS) - 1)
    assert len(repeat_table) == 2 * expected_designs * 500
