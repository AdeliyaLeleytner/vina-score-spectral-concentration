from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import public_anastassiadis_panel_validation as anast


def synthetic_source_frame() -> pd.DataFrame:
    compounds = [
        "SB 202474",
        "VEGF Receptor 2 Kinase Inhibitor II",
        *[f"compound_{index:03d}" for index in range(176)],
    ]
    cas = [f"{1_000_000 + index}-10-1" for index in range(178)]
    required = [
        alias for aliases in anast.TARGET_ALIASES.values() for alias in aliases
    ]
    source_targets = [
        *required,
        *[f"filler_kinase_{index:03d}" for index in range(300 - len(required))],
    ]
    body = np.full((300, 178), 90.0, dtype=np.float64)
    target_index = {name: index for index, name in enumerate(source_targets)}
    body[target_index["CDK2/cyclin A"]] = np.linspace(70.0, 100.0, 178)
    body[target_index["CDK2/cyclin E"]] = np.linspace(80.0, 110.0, 178)
    body[target_index["AKT1"], 0] = np.nan
    body[target_index["P38a/MAPK14"], 1] = np.nan
    filler_start = len(required)
    # Add 564 irrelevant missing cells, preserving exactly the two selected cells.
    flat = body[filler_start:].reshape(-1)
    flat[:564] = np.nan
    frame = pd.DataFrame(
        np.full((303, 179), np.nan, dtype=object),
        index=range(303),
        columns=range(179),
    )
    frame.iloc[0, 0] = (
        "Anastassiadis et al. NBT (2011) Supplementary Table 3: Complete "
        "pairwise kinase-compound activity dataset. Synthetic test suffix."
    )
    frame.iloc[1, 0] = "compound name:"
    frame.iloc[2, 0] = "compound CAS#:"
    frame.iloc[1, 1:] = compounds
    frame.iloc[2, 1:] = cas
    frame.iloc[3:, 0] = source_targets
    frame.iloc[3:, 1:] = body
    return frame


def patch_synthetic_hashes(monkeypatch: pytest.MonkeyPatch, frame: pd.DataFrame) -> None:
    numeric = frame.iloc[3:, 1:].apply(pd.to_numeric, errors="coerce")
    numeric.index = frame.iloc[3:, 0].astype(str).str.strip()
    columns = []
    for target in anast.TARGETS:
        aliases = anast.TARGET_ALIASES[target]
        columns.append(
            numeric.loc[list(aliases)].mean(axis=0, skipna=False).to_numpy(float)
        )
    remaining = np.ascontiguousarray(np.column_stack(columns))
    complete = np.ascontiguousarray(
        (100.0 - remaining)[np.isfinite(remaining).all(axis=1)]
    )
    monkeypatch.setattr(
        anast, "EXPECTED_REMAINING_178_SHA256", anast.array_sha256(remaining)
    )
    monkeypatch.setattr(
        anast,
        "EXPECTED_CDK2_MEAN_SHA256",
        anast.array_sha256(remaining[:, anast.TARGETS.index("CDK2")]),
    )
    monkeypatch.setattr(
        anast, "EXPECTED_COMPLETE_ACTIVITY_SHA256", anast.array_sha256(complete)
    )


def test_direct_frame_parser_builds_declared_176_by_21_primary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = synthetic_source_frame()
    patch_synthetic_hashes(monkeypatch, frame)
    parsed = anast.parse_source_frame(frame)
    primary = anast.activity_matrix(
        parsed,
        cdk2_context="mean_cyclin_A_and_E",
        support="complete_176",
        clipping="unclipped_activity",
    )
    assert primary.shape == (176, 21)
    assert np.isfinite(primary).all()
    cdk = parsed.remaining_by_cdk2_context["mean_cyclin_A_and_E"][:, 3]
    expected = 0.5 * (
        parsed.remaining_by_cdk2_context["cyclin_A_only"][:, 3]
        + parsed.remaining_by_cdk2_context["cyclin_E_only"][:, 3]
    )
    np.testing.assert_allclose(cdk, expected, rtol=0.0, atol=0.0)
    assert set(map(tuple, parsed.missing_records[["compound", "canonical_target"]].to_numpy())) == set(
        anast.EXPECTED_MISSING
    )
    direct = anast.boundary.target_map(primary)
    reversed_direction = anast.boundary.target_map(-primary)
    np.testing.assert_allclose(direct, reversed_direction, rtol=0.0, atol=1e-12)


def test_source_loader_requires_checksum_and_direct_xlrd_engine(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    frame = synthetic_source_frame()
    patch_synthetic_hashes(monkeypatch, frame)
    source = tmp_path / "official.xls"
    source.write_bytes(b"legacy-xls-placeholder")
    monkeypatch.setattr(anast, "sha256_file", lambda path: anast.SOURCE_SHA256)
    calls: list[tuple[str, object]] = []

    def fake_excel_file(path: Path, *, engine: str) -> SimpleNamespace:
        calls.append(("ExcelFile", engine))
        return SimpleNamespace(sheet_names=[anast.SOURCE_SHEET])

    def fake_read_excel(
        path: Path, *, sheet_name: str, header: None, engine: str
    ) -> pd.DataFrame:
        calls.append(("read_excel", engine))
        assert sheet_name == anast.SOURCE_SHEET
        assert header is None
        return frame

    monkeypatch.setattr(anast.pd, "ExcelFile", fake_excel_file)
    monkeypatch.setattr(anast.pd, "read_excel", fake_read_excel)
    parsed = anast.load_source(source)
    assert parsed.source_metadata["office_conversion_used"] is False
    assert calls == [("ExcelFile", "xlrd"), ("read_excel", "xlrd")]

    monkeypatch.setattr(anast, "sha256_file", lambda path: "0" * 64)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        anast.load_source(source)


def test_parser_fails_closed_if_selected_missing_cell_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = synthetic_source_frame()
    targets = frame.iloc[3:, 0].astype(str).str.strip().tolist()
    akt1_row = 3 + targets.index("AKT1")
    abl1_row = 3 + targets.index("ABL1")
    frame.iloc[akt1_row, 1] = 90.0
    frame.iloc[abl1_row, 1] = np.nan
    with pytest.raises(ValueError, match="selected-block missing cells changed"):
        anast.parse_source_frame(frame)


def test_selector_record_propagates_all_ties_conservatively() -> None:
    distance = np.asarray(
        [
            [0.0, 0.2, 0.8, 0.9],
            [0.2, 0.0, 0.7, 0.8],
            [0.8, 0.7, 0.0, 0.1],
            [0.9, 0.8, 0.1, 0.0],
        ]
    )
    panels = np.asarray(
        [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]], dtype=np.int16
    )
    raw_ties = np.asarray([[0, 1], [2, 3]], dtype=np.int16)
    residual_ties = np.asarray([[0, 2], [1, 3]], dtype=np.int16)
    record = anast._selector_record(distance, panels, raw_ties, residual_ties)
    raw = anast._losses_for_panels(distance, raw_ties)
    residual = anast._losses_for_panels(distance, residual_ties)
    assert record["raw_optimal_tie_count"] == 2
    assert record["residual_optimal_tie_count"] == 2
    assert record["tie_average_raw_minus_residual_loss"] == pytest.approx(
        raw.mean() - residual.mean()
    )
    assert record["conservative_best_raw_minus_worst_residual_loss"] == pytest.approx(
        raw.min() - residual.max()
    )


def test_three_panel_deleak_adds_hotspot_connectivity_exclusion() -> None:
    targets = list(anast.TARGETS)
    dockstring = pd.DataFrame(
        {
            "standard_inchikey": ["A" * 27, "B" * 27, "C" * 27, "D" * 27],
            "connectivity_block": ["A" * 14, "B" * 14, "C" * 14, "D" * 14],
            **{target: [-7.0, -8.0, -9.0, -10.0] for target in targets},
        }
    )
    public = {
        "DAVIS": pd.DataFrame({"connectivity_block": ["A" * 14]}),
        "PKIS2": pd.DataFrame({"connectivity_block": ["B" * 14]}),
    }
    primary, locked, metadata = anast.build_vina_references(
        dockstring, public, {"C" * 14}
    )
    assert locked.shape == (2, 21)
    assert primary.shape == (1, 21)
    assert metadata["additional_rows_removed_after_DAVIS_PKIS2_exclusion"] == 1
    assert metadata["primary_rows"] == 1


def test_target_leave_one_out_reselects_every_reduced_panel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(anast, "TARGETS", ("A", "B", "C", "D"))
    monkeypatch.setattr(anast, "K_GRID", (1, 2))
    rng = np.random.default_rng(17)
    broad = rng.normal(size=(40, 4))
    experiments = {
        "first": rng.normal(size=(20, 4)),
        "second": rng.normal(size=(24, 4)),
        "third": rng.normal(size=(28, 4)),
    }
    panel, maps, metadata = anast.target_leave_one_out_influence(
        broad, experiments
    )
    assert len(panel) == 4 * 2 * 3
    assert len(maps) == 4 * 3
    assert set(panel.omitted_target) == set(anast.TARGETS)
    assert set(panel.remaining_targets) == {3}
    assert metadata["panel_transfer"]["omitted_targets"] == 4
    assert all(
        row["omitted_targets"] == 4
        for row in metadata["cross_assay_map_concordance"]
    )


def test_output_writer_is_byte_deterministic(tmp_path: Path) -> None:
    summary = {
        "broad_vina_contract": {"primary_rows": 259641},
        "primary_mapping_sensitivity_summary": {
            "mean_lex_raw_minus_residual_loss_over_k": 0.1,
            "mean_conservative_best_raw_minus_worst_residual_loss_over_k": 0.08,
        },
    }
    table_names = set(anast.OUTPUT_FILES) - {"README.md", "summary.json"}
    tables = {
        name: pd.DataFrame({"row": [1, 2], "value": [0.125, 0.25]})
        for name in table_names
    }
    first = tmp_path / "first"
    second = tmp_path / "second"
    first_hashes = anast.write_outputs(first, summary, tables)
    second_hashes = anast.write_outputs(second, summary, tables)
    assert first_hashes == second_hashes
    assert (first / "bundle_checksums.sha256").read_bytes() == (
        second / "bundle_checksums.sha256"
    ).read_bytes()


def test_released_bundle_is_checksum_complete_and_contains_no_source_xls() -> None:
    output = anast.DEFAULT_OUTPUT
    if not (output / "summary.json").is_file():
        return
    summary = pd.read_json(output / "summary.json", typ="series")
    source_contract = summary["source_contract"]
    endpoint = summary["endpoint_contract"]
    assert source_contract["sha256"] == anast.SOURCE_SHA256
    assert source_contract["redistributed"] is False
    assert source_contract["office_conversion_used"] is False
    assert endpoint["complete_activity_matrix_sha256"] == (
        anast.EXPECTED_COMPLETE_ACTIVITY_SHA256
    )
    broad = summary["broad_vina_contract"]
    identity = summary["identity_contract"]
    assert broad["primary_rows"] == 259641
    assert broad["locked_DAVIS_PKIS2_only_rows"] == 259806
    assert broad["additional_rows_removed_after_DAVIS_PKIS2_exclusion"] == 165
    assert broad["primary_matrix_sha256"] == (
        "95227f6fbf70debdfd310540e0d912da12212b05aae14784c4de36a2fb53bc67"
    )
    assert identity["usable_structures"] == 155
    assert identity["unresolved_records_as_singleton_clusters"] == 21
    assert identity["whole_panel_clusters"] == 149
    assert not list(output.glob("*.xls*"))
    manifest = {}
    for line in (output / "bundle_checksums.sha256").read_text().splitlines():
        digest, filename = line.split("  ", 1)
        manifest[filename] = digest
    assert set(manifest) == set(anast.OUTPUT_FILES)
    for filename, digest in manifest.items():
        assert anast.sha256_file(output / filename) == digest

    selections = pd.read_csv(output / "optimal_panel_selections.csv")
    subset = selections.loc[
        selections.objective.eq("signed_1_minus_r")
        & selections.targets_selected_k.eq(12)
        & selections.selector.isin(
            ["raw_vina", "row_centered_residual_vina"]
        )
    ]
    counts = subset.groupby("selector").numerical_optimum_member_rank.nunique()
    assert counts.to_dict() == {"raw_vina": 2, "row_centered_residual_vina": 2}
