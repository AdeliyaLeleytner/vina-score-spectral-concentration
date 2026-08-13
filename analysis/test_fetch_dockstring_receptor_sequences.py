from __future__ import annotations

import gzip
from pathlib import Path

import pytest

import fetch_dockstring_full_receptor_sequences as full_receptors
from fetch_dockstring_receptor_sequences import parse_pdbqt_sequence


PACKAGE = Path(__file__).resolve().parents[1]


def pdbqt_atom(serial: int, atom: str, residue: str, residue_id: int) -> str:
    """Return the fixed-width fields consumed by the receptor parser."""
    return (
        f"ATOM  {serial:5d} {atom:<4s} {residue:>3s} A{residue_id:4d}    "
        "   0.000   0.000   0.000  1.00  0.00           C"
    )


def test_parser_normalizes_modified_amino_acids_and_excludes_cofactors() -> None:
    lines: list[str] = []
    serial = 1
    for residue, residue_id, atoms in (
        ("ALA", 1, ("N", "CA", "C", "O")),
        ("HIE", 2, ("N", "CA", "C", "O", "ND1")),
        ("UNK", 3, ("N", "CA", "C", "O")),
        ("ADP", 901, ("N1", "C1'", "PA", "O1A")),
    ):
        for atom in atoms:
            lines.append(pdbqt_atom(serial, atom, residue, residue_id))
            serial += 1

    sequence, modified, unknown, excluded = parse_pdbqt_sequence(
        ("\n".join(lines) + "\n").encode("ascii")
    )

    assert sequence == "AHX"
    assert modified == ["HIE"]
    assert unknown == ["UNK"]
    assert excluded == ["ADP"]


def test_parser_rejects_a_file_without_protein_backbone() -> None:
    content = (
        pdbqt_atom(1, "MG", "MG1", 902) + "\n"
    ).encode("ascii")
    with pytest.raises(ValueError, match="no protein-backbone residues"):
        parse_pdbqt_sequence(content)


def test_full_receptor_order_matches_the_frozen_matrix_header() -> None:
    with gzip.open(
        PACKAGE / "data" / "frozen" / "dockstring-dataset.tsv.gz", "rt"
    ) as handle:
        header = handle.readline().rstrip("\n").split("\t")
    assert tuple(header[2:]) == full_receptors.TARGETS
    assert len(full_receptors.TARGETS) == len(set(full_receptors.TARGETS)) == 58
