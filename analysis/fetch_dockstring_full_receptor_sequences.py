#!/usr/bin/env python3
"""Freeze all 58 receptor sequences used by the public DOCKSTRING matrix."""

from __future__ import annotations

from pathlib import Path

from fetch_dockstring_receptor_sequences import download_wheel, freeze_sequences


PACKAGE = Path(__file__).resolve().parents[1]
FASTA = PACKAGE / "data" / "frozen" / "dockstring_58_receptors.fasta"
MANIFEST = PACKAGE / "data" / "frozen" / "dockstring_58_receptors_manifest.csv"

# Frozen in the column order of dockstring-dataset.tsv.gz. Keeping the order
# explicit makes the sequence matrix independently auditable and fail-closed.
TARGETS = (
    "PPARD",
    "ABL1",
    "ADAM17",
    "ADRB1",
    "ADRB2",
    "AKT2",
    "MAOB",
    "CASP3",
    "DHFR",
    "ESR2",
    "PTK2",
    "FGFR1",
    "HMGCR",
    "HSP90AA1",
    "KIT",
    "MAPKAPK2",
    "MAP2K1",
    "NOS1",
    "PARP1",
    "PDE5A",
    "PGR",
    "PTPN1",
    "ROCK1",
    "AKT1",
    "AR",
    "CDK2",
    "CSF1R",
    "ESR1",
    "NR3C1",
    "IGF1R",
    "JAK2",
    "LCK",
    "MET",
    "MMP13",
    "PTGS2",
    "PPARA",
    "PPARG",
    "REN",
    "ADORA2A",
    "ACHE",
    "BACE1",
    "CA2",
    "CYP2C9",
    "CYP3A4",
    "HSD11B1",
    "DPP4",
    "DRD2",
    "DRD3",
    "EGFR",
    "F10",
    "GBA",
    "MAPK1",
    "MAPK14",
    "PLK1",
    "SRC",
    "THRB",
    "F2",
    "KDR",
)


def main() -> None:
    records = freeze_sequences(download_wheel(), FASTA, MANIFEST, TARGETS)
    if len(records) != 58:
        raise RuntimeError("the frozen full DOCKSTRING receptor set is not 58 targets")
    print(f"Wrote {len(records)} receptor sequences to {FASTA}")
    print(f"Wrote source checksums to {MANIFEST}")


if __name__ == "__main__":
    main()
