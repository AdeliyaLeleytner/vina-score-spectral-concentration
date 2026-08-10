#!/usr/bin/env python3
"""Freeze receptor-domain sequences from the exact DOCKSTRING 0.3.4 wheel.

The source wheel is downloaded from PyPI, verified byte-for-byte, and is not copied
into the repository.  Sequences are parsed from the official AutoDockTools PDBQT
receptors.  Only the 20 canonical amino-acid residue names are translated; every
other ATOM residue is represented as ``X`` and recorded in the manifest.
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


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def parse_pdbqt_sequence(content: bytes) -> tuple[str, list[str]]:
    """Parse one residue per PDB chain/residue/insertion-code key, in file order."""
    residues: list[str] = []
    noncanonical: list[str] = []
    seen: set[tuple[str, str, str]] = set()
    for line in content.decode("ascii").splitlines():
        if not line.startswith("ATOM"):
            continue
        key = (line[21:22], line[22:26], line[26:27])
        if key in seen:
            continue
        seen.add(key)
        residue = line[17:20].strip()
        amino_acid = THREE_TO_ONE.get(residue, "X")
        residues.append(amino_acid)
        if amino_acid == "X":
            noncanonical.append(residue)
    if not residues:
        raise ValueError("PDBQT contained no ATOM residues")
    return "".join(residues), noncanonical


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
    wheel: bytes, fasta_path: Path, manifest_path: Path
) -> list[dict[str, str | int]]:
    records: list[dict[str, str | int]] = []
    fasta_lines: list[str] = []
    with zipfile.ZipFile(io.BytesIO(wheel)) as archive:
        for target in TARGETS:
            member = f"dockstring/resources/targets/{target}_target.pdbqt"
            content = archive.read(member)
            sequence, noncanonical = parse_pdbqt_sequence(content)
            fasta_lines.extend(
                [
                    (
                        f">{target} source=dockstring-{DOCKSTRING_VERSION} "
                        f"member={member} noncanonical_as_X=true"
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
                    "noncanonical_residue_count": len(noncanonical),
                    "noncanonical_residue_names": ";".join(sorted(set(noncanonical))),
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
