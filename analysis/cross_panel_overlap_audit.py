#!/usr/bin/env python3
"""Quantify chemical and target overlap between the two primary Vina panels."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold


RDLogger.DisableLog("rdApp.*")
PACKAGE = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def docking44_identities(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(
        path,
        usecols=["Cleaned SMILES", "Canonical SMILES"],
    )
    smiles = frame["Cleaned SMILES"].fillna(frame["Canonical SMILES"]).astype(str)
    rows: list[dict[str, str]] = []
    for row_index, value in enumerate(smiles):
        molecule = Chem.MolFromSmiles(value)
        if molecule is None:
            raise ValueError(f"invalid Docking-44 SMILES at row {row_index}")
        key = Chem.MolToInchiKey(molecule)
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=molecule)
        rows.append(
            {
                "full_inchikey": key,
                "connectivity_block": key[:14],
                "nonempty_murcko_scaffold": scaffold,
            }
        )
    return pd.DataFrame.from_records(rows)


def nonempty(values: pd.Series) -> set[str]:
    return {str(value) for value in values.dropna() if str(value)}


def analyze(
    docking44_path: Path,
    dockstring_identity_path: Path,
    target_mapping_path: Path,
    family_path: Path,
) -> dict[str, object]:
    docking44 = docking44_identities(docking44_path)
    dockstring = pd.read_csv(dockstring_identity_path)
    target_mapping = pd.read_csv(target_mapping_path)
    families = pd.read_csv(family_path)

    d44_full = set(docking44.full_inchikey)
    ds_full = set(dockstring.raw_inchikey.astype(str))
    d44_blocks = set(docking44.connectivity_block)
    ds_blocks = set(dockstring.raw_connectivity.astype(str))
    d44_scaffolds = nonempty(docking44.nonempty_murcko_scaffold)
    ds_scaffolds = nonempty(dockstring.raw_murcko_scaffold)

    shared_full = d44_full & ds_full
    shared_blocks = d44_blocks & ds_blocks
    shared_scaffolds = d44_scaffolds & ds_scaffolds
    d44_families = set(
        families.loc[families.dataset.eq("Docking-44"), "family"].astype(str)
    )
    ds_families = set(
        families.loc[families.dataset.eq("DOCKSTRING-58"), "family"].astype(str)
    )

    return {
        "analysis": "cross-panel chemical and target-support overlap audit",
        "support": {
            "docking44_ligands": int(len(docking44)),
            "dockstring_complete_ligands": int(len(dockstring)),
            "docking44_targets": 44,
            "dockstring_targets": 58,
        },
        "chemical_overlap": {
            "shared_full_standard_inchikeys": int(len(shared_full)),
            "shared_connectivity_blocks": int(len(shared_blocks)),
            "docking44_rows_in_shared_connectivity_blocks": int(
                docking44.connectivity_block.isin(shared_blocks).sum()
            ),
            "dockstring_rows_in_shared_connectivity_blocks": int(
                dockstring.raw_connectivity.astype(str).isin(shared_blocks).sum()
            ),
            "shared_nonempty_murcko_scaffolds": int(len(shared_scaffolds)),
            "docking44_rows_in_shared_nonempty_murcko_scaffolds": int(
                docking44.nonempty_murcko_scaffold.isin(shared_scaffolds).sum()
            ),
            "dockstring_rows_in_shared_nonempty_murcko_scaffolds": int(
                dockstring.raw_murcko_scaffold.astype(str).isin(shared_scaffolds).sum()
            ),
            "identity_note": (
                "Standard InChIKeys were recomputed from Docking-44 analysis SMILES; "
                "DOCKSTRING raw Standard InChIKeys and Murcko scaffolds come from the "
                "frozen identity contract. Connectivity blocks merge stereoisomers and "
                "protonation-state suffixes."
            ),
        },
        "target_overlap": {
            "mapped_target_identities": int(len(target_mapping)),
            "mapped_human_receptor_identities": int(
                target_mapping.docking44_structure_is_human.sum()
            ),
            "same_receptor_structure_used": int(
                target_mapping.same_receptor_structure_in_dockstring.sum()
            ),
            "mapped_pairs": target_mapping[
                ["docking44_target", "dockstring_target", "human_uniprot_accession"]
            ].to_dict(orient="records"),
            "broad_family_labels_docking44": sorted(d44_families),
            "broad_family_labels_dockstring": sorted(ds_families),
            "shared_broad_family_labels": sorted(d44_families & ds_families),
        },
        "interpretation": (
            "The panels were generated from different ligand collections and target "
            "lists, but they are not disjoint biological or chemical samples. Replication "
            "therefore means agreement across independently sourced matrices, not formal "
            "statistical independence."
        ),
        "sources": {
            "docking44": {
                "path": str(docking44_path.relative_to(PACKAGE)),
                "sha256": sha256_file(docking44_path),
            },
            "dockstring_identity_contract": {
                "path": str(dockstring_identity_path.relative_to(PACKAGE)),
                "sha256": sha256_file(dockstring_identity_path),
            },
            "target_mapping": {
                "path": str(target_mapping_path.relative_to(PACKAGE)),
                "sha256": sha256_file(target_mapping_path),
            },
            "target_families": {
                "path": str(family_path.relative_to(PACKAGE)),
                "sha256": sha256_file(family_path),
            },
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--docking44",
        type=Path,
        default=PACKAGE / "data/frozen/df_final_v4.csv.gz",
    )
    parser.add_argument(
        "--dockstring-identities",
        type=Path,
        default=PACKAGE / "data/frozen/dockstring_identity_contract_2026-08-03.csv.gz",
    )
    parser.add_argument(
        "--target-mapping",
        type=Path,
        default=PACKAGE / "results/residual_mechanism/cross_panel_target_mapping.csv",
    )
    parser.add_argument(
        "--target-families",
        type=Path,
        default=PACKAGE / "data/target_families.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PACKAGE / "results/cross_panel_overlap_audit/summary.json",
    )
    args = parser.parse_args()
    result = analyze(
        args.docking44,
        args.dockstring_identities,
        args.target_mapping,
        args.target_families,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
