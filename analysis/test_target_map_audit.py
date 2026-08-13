from __future__ import annotations

import ast
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import target_map_audit as audit


SCRIPT = Path(audit.__file__).resolve()
FIXTURE = SCRIPT.parent / "fixtures" / "target_map_audit_synthetic.csv"
TARGETS = ("target_A", "target_B", "target_C", "target_D")


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_complete_cli(output_dir: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            str(FIXTURE),
            "--output-dir",
            str(output_dir),
            "--target-regex",
            "^target_",
            "--ligand-id",
            "ligand_id",
            "--domain-column",
            "molecular_weight",
            "--domain-quantiles",
            "2",
            "--pilot-sizes",
            "6",
            "--pilot-repeats",
            "8",
            "--top-edge-count",
            "2",
            "--panel-k",
            "2",
            "--seed",
            "17",
        ],
        check=True,
        capture_output=True,
        text=True,
    )


def test_module_has_no_local_or_legacy_imports() -> None:
    tree = ast.parse(SCRIPT.read_text())
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert imported <= {
        "__future__",
        "argparse",
        "dataclasses",
        "hashlib",
        "json",
        "math",
        "numpy",
        "pandas",
        "pathlib",
        "re",
        "sys",
        "typing",
    }
    lowered = SCRIPT.read_text().lower()
    for marker in (
        "build_evidence",
        "manuscript_evidence",
        "chembl",
        "pkis1",
        "kirhub",
        "fixed_pose",
    ):
        assert marker not in lowered


def test_row_centering_preserves_within_row_ranks_and_pr_identity_is_exact() -> None:
    frame = pd.read_csv(FIXTURE)
    matrix = frame[list(TARGETS)].to_numpy(dtype=np.float64)
    centered = audit.row_center(matrix)
    assert np.array_equal(
        np.argsort(matrix, axis=1, kind="stable"),
        np.argsort(centered, axis=1, kind="stable"),
    )
    for first in range(matrix.shape[1]):
        for second in range(first + 1, matrix.shape[1]):
            assert np.allclose(
                matrix[:, first] - matrix[:, second],
                centered[:, first] - centered[:, second],
                atol=1e-14,
            )

    for surface in (matrix, audit.two_way_center(matrix)):
        record = audit.spectrum(audit.target_correlation(surface))
        assert record["pr_identity_absolute_error"] < 1e-12
        assert abs(
            record["participation_ratio_dimension"]
            - record["pr_from_exact_mean_r2_identity"]
        ) < 1e-12


def test_complete_cli_emits_reconstructable_support_audit(tmp_path: Path) -> None:
    output = tmp_path / "audit"
    completed = run_complete_cli(output)
    stdout = json.loads(completed.stdout)
    assert stdout["rows"] == 20
    assert stdout["targets"] == 4

    expected = {
        *audit.CORE_OUTPUTS,
        "domain_bin_edges.csv",
        "domain_map_comparisons.csv",
        "pilot_recovery_replicates.csv",
        "pilot_recovery_summary.csv",
        "target_panel_medoids.csv",
        "target_panel_summary.json",
        "output_checksums.json",
    }
    assert {path.name for path in output.iterdir()} == expected

    checksums = json.loads((output / "output_checksums.json").read_text())
    assert checksums["algorithm"] == "sha256"
    assert set(checksums["files"]) == expected - {"output_checksums.json"}
    for filename, contract in checksums["files"].items():
        path = output / filename
        assert path.stat().st_size == contract["bytes"]
        assert file_hash(path) == contract["sha256"]

    preprocessing = json.loads((output / "preprocessing_contract.json").read_text())
    assert preprocessing["missing_data"]["policy"] == "error"
    assert preprocessing["selection"]["target_columns"] == list(TARGETS)
    assert "unchanged" in preprocessing["row_centering_rank_invariance"][
        "theoretical_result"
    ]

    spectra = json.loads((output / "spectra.json").read_text())
    for transform in audit.TRANSFORMS:
        assert spectra["surfaces"][transform]["pr_identity_absolute_error"] < 1e-12

    for filename in (
        "target_correlation_raw.csv",
        "target_correlation_two_way_centered.csv",
    ):
        correlation = pd.read_csv(output / filename).set_index("target")
        assert list(correlation.index) == list(TARGETS)
        assert np.allclose(correlation, correlation.T, atol=1e-14)
        assert np.allclose(np.diag(correlation), 1.0, atol=1e-14)

    comparisons = pd.read_csv(output / "domain_map_comparisons.csv")
    residual = comparisons.loc[
        comparisons["transform"].eq("two_way_centered")
    ].iloc[0]
    assert residual.global_edge_spearman < -0.8
    assert residual.edge_sign_agreement_fraction < 0.5

    recovery = pd.read_csv(output / "pilot_recovery_summary.csv")
    assert set(recovery.reference_scope) == {"row_disjoint_complement_map"}
    assert set(recovery["transform"]) == set(audit.TRANSFORMS)
    assert set(recovery.pilot_rows) == {6}

    panel = json.loads((output / "target_panel_summary.json").read_text())
    assert "not a biologically validated target panel" in panel[
        "biological_boundary_warning"
    ]
    assignments = pd.read_csv(output / "target_panel_medoids.csv")
    assert assignments.groupby("transform").member_is_medoid.sum().to_dict() == {
        "raw": 2,
        "two_way_centered": 2,
    }


def test_complete_cli_is_byte_deterministic_for_a_fixed_seed(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    run_complete_cli(first)
    run_complete_cli(second)
    assert {path.name for path in first.iterdir()} == {
        path.name for path in second.iterdir()
    }
    for first_path in first.iterdir():
        assert file_hash(first_path) == file_hash(second / first_path.name)


def test_explicit_columns_and_gzipped_tsv_are_supported(tmp_path: Path) -> None:
    frame = pd.read_csv(FIXTURE)
    compressed = tmp_path / "scores.tsv.gz"
    frame.to_csv(compressed, sep="\t", index=False, compression="gzip")
    output = tmp_path / "minimal"
    result = audit.run_audit(
        audit.AuditConfig(
            input_path=compressed,
            output_dir=output,
            target_columns=TARGETS,
            ligand_id="ligand_id",
            seed=11,
        )
    )
    assert result["rows"] == 20 and result["targets"] == 4
    contract = json.loads((output / "preprocessing_contract.json").read_text())
    assert contract["source"]["format"] == "TSV"
    assert contract["source"]["compression"] == "gzip"
    assert contract["selection"]["mode"] == "explicit_columns"
    assert {path.name for path in output.iterdir()} == {
        *audit.CORE_OUTPUTS,
        "output_checksums.json",
    }


def test_missing_and_malformed_targets_fail_closed(tmp_path: Path) -> None:
    frame = pd.read_csv(FIXTURE).iloc[:8].copy()
    malformed = frame.copy()
    malformed["target_A"] = malformed["target_A"].astype(object)
    malformed.loc[0, "target_A"] = "not-a-number"
    malformed_path = tmp_path / "malformed.csv"
    malformed.to_csv(malformed_path, index=False)
    malformed_output = tmp_path / "malformed-output"
    with pytest.raises(audit.AuditError, match="nonnumeric"):
        audit.run_audit(
            audit.AuditConfig(
                input_path=malformed_path,
                output_dir=malformed_output,
                target_columns=TARGETS,
                imputation="target-mean",
            )
        )
    assert not malformed_output.exists()

    missing = frame.copy()
    missing.loc[0, "target_A"] = np.nan
    missing_path = tmp_path / "missing.csv"
    missing.to_csv(missing_path, index=False)
    with pytest.raises(audit.AuditError, match="explicit imputation"):
        audit.run_audit(
            audit.AuditConfig(
                input_path=missing_path,
                output_dir=tmp_path / "missing-error",
                target_columns=TARGETS,
            )
        )

    imputed_output = tmp_path / "imputed"
    audit.run_audit(
        audit.AuditConfig(
            input_path=missing_path,
            output_dir=imputed_output,
            target_columns=TARGETS,
            imputation="target-mean",
        )
    )
    contract = json.loads((imputed_output / "preprocessing_contract.json").read_text())
    assert contract["missing_data"]["missing_cells_before_imputation"] == 1
    assert contract["missing_data"]["policy"] == "target-mean"


def test_invalid_roles_optional_analyses_and_output_state_are_rejected(
    tmp_path: Path,
) -> None:
    with pytest.raises(audit.AuditError, match="exactly one"):
        audit.run_audit(
            audit.AuditConfig(input_path=FIXTURE, output_dir=tmp_path / "none")
        )
    with pytest.raises(audit.AuditError, match="requires domain_column"):
        audit.run_audit(
            audit.AuditConfig(
                input_path=FIXTURE,
                output_dir=tmp_path / "domain",
                target_columns=TARGETS,
                domain_quantiles=2,
            )
        )
    with pytest.raises(audit.AuditError, match="leave at least three"):
        audit.run_audit(
            audit.AuditConfig(
                input_path=FIXTURE,
                output_dir=tmp_path / "pilot",
                target_columns=TARGETS,
                pilot_sizes=(19,),
                top_edge_count=2,
            )
        )

    occupied = tmp_path / "occupied"
    occupied.mkdir()
    (occupied / "keep.txt").write_text("user-owned\n")
    with pytest.raises(audit.AuditError, match="must be empty"):
        audit.run_audit(
            audit.AuditConfig(
                input_path=FIXTURE,
                output_dir=occupied,
                target_columns=TARGETS,
            )
        )
    assert (occupied / "keep.txt").read_text() == "user-owned\n"
