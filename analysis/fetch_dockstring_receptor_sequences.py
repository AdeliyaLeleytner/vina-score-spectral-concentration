#!/usr/bin/env python3
"""Freeze receptor-domain sequences from the exact DOCKSTRING 0.3.4 wheel.

The source wheel is downloaded from PyPI, verified byte-for-byte, and is not copied
into the repository. Sequences are parsed from the official AutoDockTools PDBQT
receptors. A residue must contain the protein-backbone atoms N, CA and C; this
excludes crystallographic ligands, metals and cofactors that also occur as ATOM
records. Common protonation-state and covalent-modification aliases are translated
to their parent amino acid. Any other backbone-containing residue is represented as
``X`` and recorded in the manifest.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import urllib.request
import zipfile
from pathlib import Path


PACKAGE = Path(__file__).resolve().parents[1]
DEFAULT_FASTA = PACKAGE / "data" / "frozen" / "dockstring_kinase_receptors.fasta"
DEFAULT_MANIFEST = (
    PACKAGE / "data" / "frozen" / "dockstring_kinase_receptors_manifest.csv"
)
DOCKSTRING_VERSION = "0.3.4"
DOCKSTRING_LICENSE = "Apache-2.0"
WHEEL_URL = (
    "https://files.pythonhosted.org/packages/26/7b/"
    "4eab9b2d80f920b0fbaaf0c7cc42e948df6fc716b18cb5399bd3d5c768e7/"
    "dockstring-0.3.4-py3-none-any.whl"
)
WHEEL_SHA256 = "e0e61d4f6f5ff22b3cc2905ea01af5752fe0a14f93b4a148d81ffa1fac661cc5"

TARGETS = (
    "ABL1",
    "AKT1",
    "AKT2",
    "CDK2",
    "CSF1R",
    "EGFR",
    "FGFR1",
    "IGF1R",
    "JAK2",
    "KDR",
    "KIT",
    "LCK",
    "MAP2K1",
    "MAPK1",
    "MAPK14",
    "MAPKAPK2",
    "MET",
    "PLK1",
    "PTK2",
    "ROCK1",
    "SRC",
)

THREE_TO_ONE = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
}

# AutoDockTools/Amber residue aliases present in the exact receptor files.  The
# mapping is deliberately explicit rather than inferred from the first letter.
# It can therefore fail visibly if a future wheel introduces a new residue name.
MODIFIED_TO_ONE = {
    "HID": "H",
    "HIE": "H",
    "HIP": "H",
    "HIZ": "H",
    "TPO": "T",
    "PTR": "Y",
    "CYM": "C",
    "CYT": "C",
    "GLV": "E",
    "LEV": "L",
    "MEU": "M",
    # Additional aliases in the broader 58-target DOCKSTRING receptor set.
    "CYX": "C",
    "GLZ": "E",
    "GLH": "E",
    "GLO": "E",
    "DID": "D",
    "DIC": "D",
    "TYS": "Y",
}
BACKBONE_ATOMS = frozenset({"N", "CA", "C"})


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def parse_pdbqt_sequence(
    content: bytes,
) -> tuple[str, list[str], list[str], list[str]]:
    """Parse protein residues in file order and audit every noncanonical record.

    Returns the translated sequence, modified amino-acid aliases that were
    normalized, unknown backbone-containing residues represented as ``X``, and
    non-protein ATOM residues excluded because they lack a complete backbone.
    """
    ordered_keys: list[tuple[str, str, str, str]] = []
    residue_atoms: dict[tuple[str, str, str, str], set[str]] = {}
    for line in content.decode("ascii").splitlines():
        if not line.startswith("ATOM"):
            continue
        residue = line[17:20].strip().upper()
        # Some PDBQT resources blank the chain identifier and reuse a residue
        # number for a protein residue and a non-protein ATOM record. Including
        # the residue name prevents those records from being merged.
        key = (line[21:22], line[22:26], line[26:27], residue)
        atom = line[12:16].strip().upper()
        if key not in residue_atoms:
            ordered_keys.append(key)
            residue_atoms[key] = set()
        residue_atoms[key].add(atom)

    residues: list[str] = []
    modified: list[str] = []
    unknown_backbone: list[str] = []
    excluded_nonprotein: list[str] = []
    for key in ordered_keys:
        residue = key[3]
        if not BACKBONE_ATOMS.issubset(residue_atoms[key]):
            excluded_nonprotein.append(residue)
            continue
        if residue in THREE_TO_ONE:
            residues.append(THREE_TO_ONE[residue])
        elif residue in MODIFIED_TO_ONE:
            residues.append(MODIFIED_TO_ONE[residue])
            modified.append(residue)
        else:
            residues.append("X")
            unknown_backbone.append(residue)
    if not residues:
        raise ValueError("PDBQT contained no protein-backbone residues")
    return "".join(residues), modified, unknown_backbone, excluded_nonprotein


def download_wheel() -> bytes:
    with urllib.request.urlopen(WHEEL_URL) as response:  # noqa: S310 - pinned URL/hash
        content = response.read()
    observed = sha256_bytes(content)
    if observed != WHEEL_SHA256:
        raise ValueError(
            f"DOCKSTRING wheel SHA-256 mismatch: expected {WHEEL_SHA256}, got {observed}"
        )
    return content


def freeze_sequences(
    wheel: bytes,
    fasta_path: Path,
    manifest_path: Path,
    targets: tuple[str, ...] = TARGETS,
) -> list[dict[str, str | int]]:
    if not targets or len(set(targets)) != len(targets):
        raise ValueError("receptor target list must be non-empty and unique")
    records: list[dict[str, str | int]] = []
    fasta_lines: list[str] = []
    with zipfile.ZipFile(io.BytesIO(wheel)) as archive:
        for target in targets:
            member = f"dockstring/resources/targets/{target}_target.pdbqt"
            content = archive.read(member)
            (
                sequence,
                modified,
                unknown_backbone,
                excluded_nonprotein,
            ) = parse_pdbqt_sequence(content)
            fasta_lines.extend(
                [
                    (
                        f">{target} source=dockstring-{DOCKSTRING_VERSION} "
                        f"member={member} modified_to_parent=true "
                        f"unknown_backbone_as_X=true nonprotein_ATOM_excluded=true"
                    ),
                    sequence,
                ]
            )
            records.append(
                {
                    "target": target,
                    "dockstring_version": DOCKSTRING_VERSION,
                    "source_wheel_url": WHEEL_URL,
                    "source_wheel_sha256": WHEEL_SHA256,
                    "source_license": DOCKSTRING_LICENSE,
                    "source_member": member,
                    "source_pdbqt_sha256": sha256_bytes(content),
                    "sequence_length": len(sequence),
                    "sequence_sha256": sha256_bytes(sequence.encode("ascii")),
                    "modified_residue_count": len(modified),
                    "modified_residue_names": ";".join(sorted(set(modified))),
                    "unknown_backbone_residue_count": len(unknown_backbone),
                    "unknown_backbone_residue_names": ";".join(
                        sorted(set(unknown_backbone))
                    ),
                    "excluded_nonprotein_atom_residue_count": len(
                        excluded_nonprotein
                    ),
                    "excluded_nonprotein_atom_residue_names": ";".join(
                        sorted(set(excluded_nonprotein))
                    ),
                }
            )
    fasta_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    fasta_path.write_text("\n".join(fasta_lines) + "\n")
    with manifest_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fasta", type=Path, default=DEFAULT_FASTA)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = freeze_sequences(download_wheel(), args.fasta, args.manifest)
    print(f"Wrote {len(records)} receptor sequences to {args.fasta}")
    print(f"Wrote source checksums to {args.manifest}")


if __name__ == "__main__":
    main()
