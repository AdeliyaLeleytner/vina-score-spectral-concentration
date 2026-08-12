from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from rdkit import Chem

import public_anastassiadis_identity_audit as audit


def _cas_for(index: int) -> str:
    first = str(100_000 + index)
    second = "00"
    body = first + second
    check = sum(
        multiplier * int(digit)
        for multiplier, digit in enumerate(reversed(body), start=1)
    ) % 10
    return f"{first}-{second}-{check}"


def _synthetic_workbook() -> pd.DataFrame:
    frame = pd.DataFrame(np.nan, index=range(303), columns=range(179), dtype=object)
    names = [f"Compound {index:03d}" for index in range(1, 179)]
    names[0] = "SB 202474"
    names[1] = "VEGF Receptor 2 Kinase Inhibitor II"
    frame.iloc[1, 1:179] = names
    frame.iloc[2, 1:179] = [_cas_for(index) for index in range(1, 179)]
    row_labels = [f"Target {index:03d}" for index in range(300)]
    aliases = [
        alias
        for source_aliases in audit.HOTSPOT_TARGET_MAP.values()
        for alias in source_aliases
    ]
    row_labels[: len(aliases)] = aliases
    frame.iloc[3:, 0] = row_labels
    frame.iloc[3:, 1:179] = 100.0
    lookup = {label: 3 + index for index, label in enumerate(row_labels)}
    frame.loc[lookup["AKT1"], 1] = np.nan
    frame.loc[lookup["P38a/MAPK14"], 2] = np.nan
    return frame


def _identity_row(index: int, smiles: str, *, usable: bool = True) -> dict:
    molecule = Chem.MolFromSmiles(smiles)
    assert molecule is not None
    key = Chem.MolToInchiKey(molecule)
    return {
        "workbook_compound_index_1based": index,
        "workbook_compound_name": f"compound-{index}",
        "workbook_cas": _cas_for(index),
        "complete_fixed21": True,
        "structure_usable": usable,
        "pubchem_smiles": smiles if usable else "",
        "rdkit_standard_inchikey": key if usable else "",
        "rdkit_connectivity": key[:14] if usable else "",
        "structure_fragment_count": 1 if usable else np.nan,
        "de_leakage_connectivity_blocks": key[:14],
    }


def test_cas_checksum_contract() -> None:
    assert audit.cas_checksum_valid("50-00-0")
    assert audit.cas_checksum_valid(_cas_for(17))
    assert not audit.cas_checksum_valid("50-00-1")
    assert not audit.cas_checksum_valid("not-a-cas")


def test_parse_workbook_identifies_exact_fixed21_missing_pairs() -> None:
    compounds, metadata = audit.parse_workbook_frame(_synthetic_workbook())
    assert len(compounds) == 178
    assert int(compounds.complete_fixed21.sum()) == 176
    observed = set(
        compounds.loc[
            ~compounds.complete_fixed21,
            ["workbook_compound_name", "missing_fixed21_targets"],
        ].itertuples(index=False, name=None)
    )
    assert observed == audit.EXPECTED_INCOMPLETE_PAIRS
    assert metadata["fixed21_workbook_aliases"]["CDK2"] == [
        "CDK2/cyclin A",
        "CDK2/cyclin E",
    ]


@pytest.mark.parametrize(
    ("cas", "name", "selected", "status", "ambiguous"),
    [
        ([1], [1], 1, "resolved_unique_cas_name_agrees", False),
        ([1], [], 1, "resolved_unique_cas_name_not_found", False),
        ([1], [2], None, "unresolved_cas_name_conflict", True),
        ([1, 2], [2, 3], 2, "resolved_unique_cas_name_intersection", False),
        ([1, 2], [1, 2], None, "unresolved_ambiguous_cas", True),
        ([], [3], 3, "resolved_unique_name_only", False),
        ([], [3, 4], None, "unresolved_ambiguous_name", True),
        ([], [], None, "unresolved_not_found", False),
    ],
)
def test_resolution_rule_is_fail_closed(
    cas: list[int],
    name: list[int],
    selected: int | None,
    status: str,
    ambiguous: bool,
) -> None:
    observed = audit.resolve_candidate_cids(cas, name)
    assert observed[:3] == (selected, status, ambiguous)


def test_butina_and_murcko_assignments_are_deterministic() -> None:
    identity = pd.DataFrame(
        [
            _identity_row(1, "c1ccccc1"),
            _identity_row(2, "Cc1ccccc1"),
            _identity_row(3, "CCO"),
            _identity_row(4, "CCN"),
        ]
    )
    first, first_summary = audit.chemical_group_assignments(identity)
    second, second_summary = audit.chemical_group_assignments(identity)
    pd.testing.assert_frame_equal(first, second)
    assert first_summary == second_summary
    assert first.butina_r2_2048_tanimoto50_cluster.notna().all()
    assert first.murcko_scaffold.notna().all()
    assert first_summary["usable_structures"] == 4


def test_overlap_distinguishes_exact_connectivity_and_candidate_exclusion() -> None:
    first = _identity_row(1, "F[C@H](Cl)Br")
    second = _identity_row(2, "F[C@@H](Cl)Br")
    unresolved = _identity_row(3, "CCO", usable=False)
    unresolved["de_leakage_connectivity_blocks"] = first["rdkit_connectivity"]
    identity = pd.DataFrame([first, second, unresolved])
    reference = pd.DataFrame(
        {
            "source_row": [7],
            "standard_inchikey": [first["rdkit_standard_inchikey"]],
            "connectivity": [first["rdkit_connectivity"]],
        }
    )
    records, summary, _ = audit.overlap_audit(identity, {"TEST": reference})
    assert records.exact_full_standard_inchikey_overlap.tolist() == [True, False, False]
    assert records.connectivity_overlap.tolist() == [True, True, False]
    assert records.conservative_de_leakage_connectivity_overlap.tolist() == [
        True,
        True,
        True,
    ]
    assert int(summary.iloc[0].connectivity_only_overlap) == 1
    assert int(summary.iloc[0].conservative_de_leakage_connectivity_overlap) == 3


def test_frozen_identity_contract_and_candidate_union() -> None:
    if not audit.DEFAULT_IDENTITY.exists():
        pytest.skip("identity-only PubChem crosswalk has not been generated")
    identity = audit.validate_frozen_identity(pd.read_csv(audit.DEFAULT_IDENTITY))
    assert int(identity.complete_fixed21.sum()) == 176
    assert int(identity.structure_usable.sum()) == 157
    unresolved = identity[~identity.structure_usable]
    assert len(unresolved) == 21
    assert unresolved.candidate_only_connectivity_blocks.str.len().gt(0).all()
    assert (
        unresolved.candidate_only_connectivity_blocks
        == unresolved.de_leakage_connectivity_blocks
    ).all()
    forbidden = {"activity", "percent_remaining_activity", "hotspot_value"}
    assert forbidden.isdisjoint(identity.columns)


def test_official_workbook_checksum_if_source_is_present() -> None:
    if not audit.DEFAULT_WORKBOOK.exists():
        pytest.skip("official source workbook is intentionally not redistributed")
    assert audit.sha256_file(audit.DEFAULT_WORKBOOK) == audit.WORKBOOK_SHA256
    compounds, _ = audit.load_workbook(audit.DEFAULT_WORKBOOK)
    assert len(compounds) == 178
    assert int(compounds.complete_fixed21.sum()) == 176


def test_atomic_csv_bytes_are_deterministic(tmp_path: Path) -> None:
    frame = pd.DataFrame({"b": [2, 1], "a": [0.5, np.nan]})
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    audit.write_csv(first, frame)
    audit.write_csv(second, frame)
    assert first.read_bytes() == second.read_bytes()
    assert hashlib.sha256(first.read_bytes()).hexdigest() == hashlib.sha256(
        second.read_bytes()
    ).hexdigest()
