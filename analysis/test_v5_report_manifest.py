"""Integrity checks for the v5 report-level source manifest."""

from __future__ import annotations

import csv
import hashlib
from dataclasses import replace

import pytest

from analysis import build_v5_report_manifest as manifest


def test_discovery_covers_report_graph_without_generated_artifacts() -> None:
    rows = manifest.build_rows()
    by_path = {row.path: row for row in rows}

    required = {
        ".github/workflows/v5-report.yml",
        "Makefile",
        "manuscript_v5.tex",
        "supplementary_v5.tex",
        "references_v5.tex",
        "analysis/make_v5_figures.py",
        "analysis/make_v5_graphical_abstract.py",
        "analysis/pdsp_counterscreen_retrieval.py",
        "analysis/reproduce_pdsp_derived_qap.py",
        "analysis/test_reproduce_pdsp_derived_qap.py",
        "analysis/test_v5_manuscript_numbers.py",
        "analysis/test_v5_supplement_numbers.py",
        "data/frozen/df_final_v4.csv.gz",
        "results/nonvina_scorer_transport/scorer_spectral_metrics.csv",
        "results/pdsp_counterscreen_retrieval/excluded_docking44_ligand_ids.csv",
        "results/pdsp_counterscreen_retrieval/complete_case_support_metrics.csv",
        "results/pdsp_counterscreen_retrieval/complete_case_support_qap.csv",
        "results/pdsp_counterscreen_retrieval/minimum_pair_support_sensitivity.csv",
        "results/pdsp_counterscreen_retrieval/paired_qap.csv",
        "results/pdsp_counterscreen_retrieval/target_pairs.csv",
        "results/public_residual_null_audit/mw_conditional_null_repeats.csv",
        "results/residual_mie_boundary_audit/null_summary.csv",
        "results/v5_declarations.tex",
    }
    assert required <= by_path.keys()
    assert all(not row.path.endswith(tuple(manifest.FORBIDDEN_SOURCE_SUFFIXES)) for row in rows)
    assert manifest.MANIFEST.relative_to(manifest.ROOT).as_posix() not in by_path
    assert manifest.DIGEST.relative_to(manifest.ROOT).as_posix() not in by_path
    assert [row.path for row in rows] == sorted(by_path)
    assert len(rows) == len(by_path)
    assert {row.scope for row in rows} == {manifest.REPORT_SCOPE}


def test_saved_manifest_and_submission_art_are_current() -> None:
    assert manifest.verify_manifest(check_staged_assets=True)


def test_verifier_rejects_a_stale_row_even_with_a_matching_manifest_digest(tmp_path) -> None:
    rows = manifest.build_rows()
    altered = list(rows)
    altered[0] = replace(altered[0], sha256="0" * 64)
    payload = manifest.serialize(altered)
    manifest_path = tmp_path / manifest.MANIFEST.name
    digest_path = tmp_path / manifest.DIGEST.name
    manifest_path.write_bytes(payload)
    digest_path.write_text(
        f"{hashlib.sha256(payload).hexdigest()}  {manifest_path.name}\n",
        encoding="ascii",
    )

    with pytest.raises(manifest.ManifestError, match="stale"):
        manifest.verify_manifest(
            manifest_path,
            digest_path,
            check_staged_assets=False,
        )


def test_verifier_reports_a_missing_manifest_cleanly(tmp_path) -> None:
    digest_path = tmp_path / manifest.DIGEST.name
    digest_path.write_text(f"{'0' * 64}  {manifest.MANIFEST.name}\n", encoding="ascii")
    with pytest.raises(manifest.ManifestError, match="report manifest is missing"):
        manifest.verify_manifest(
            tmp_path / manifest.MANIFEST.name,
            digest_path,
            check_staged_assets=False,
        )


def test_nested_integrity_sidecar_is_checked(tmp_path) -> None:
    payload = tmp_path / "payload.csv"
    sidecar = tmp_path / "payload.sha256"
    payload.write_text("a,b\n1,2\n", encoding="utf-8")
    sidecar.write_text(f"{manifest.sha256(payload)}  {payload.name}\n", encoding="ascii")
    manifest.verify_digest_pair(payload, sidecar)

    payload.write_text("a,b\n1,3\n", encoding="utf-8")
    with pytest.raises(manifest.ManifestError, match="integrity sidecar is stale"):
        manifest.verify_digest_pair(payload, sidecar)


def test_saved_csv_has_the_canonical_schema_and_scope() -> None:
    with manifest.MANIFEST.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
        assert tuple(rows[0]) == manifest.MANIFEST_FIELDS
    assert {row["kind"] for row in rows} == {
        "build_contract",
        "frozen_input",
        "report_code",
        "report_metadata",
        "tex_source",
    }
    assert {row["scope"] for row in rows} == {manifest.REPORT_SCOPE}
