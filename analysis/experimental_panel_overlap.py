#!/usr/bin/env python3
"""Audit chemical overlap among the fixed-20 experimental kinase panels.

DAVIS, PKIS2, and PKIS1 structures are compared at three explicitly distinct
levels: recomputed Standard InChIKey, its 14-character connectivity block, and
canonical nonempty Bemis--Murcko scaffold.  Acyclic ligands are counted but are
not collapsed into one shared pseudo-scaffold.  KiRHub has no structures in the
frozen source; its previously audited normalized-name overlap with DAVIS and
the associated geometry sensitivity are copied from the frozen KiRHub summary
after validating the expected fields.

This script is science-only.  It does not read or modify manuscript sources,
build rules, or data manifests.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from itertools import combinations
from pathlib import Path
from typing import Any

import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

import dense_davis_benchmark as davis
import residual_target_geometry_validation as geometry


PACKAGE = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PACKAGE / "results" / "experimental_panel_overlap"
DEFAULT_PKIS1_ZIP = Path("/tmp/pkis1_supplement.zip")
DEFAULT_KIRHUB_SUMMARY = PACKAGE / "results" / "kirhub_external_validation" / "summary.json"


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def murcko_fields(smiles: pd.Series) -> tuple[set[str], int]:
    """Return nonempty canonical scaffolds and the number of acyclic ligands."""
    scaffolds: set[str] = set()
    acyclic = 0
    for value in smiles.astype(str):
        molecule = Chem.MolFromSmiles(value)
        if molecule is None:
            raise ValueError("a panel SMILES cannot be parsed by RDKit")
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=molecule)
        if scaffold:
            scaffolds.add(scaffold)
        else:
            acyclic += 1
    return scaffolds, acyclic


def structure_summary(
    frame: pd.DataFrame, smiles_column: str
) -> tuple[dict[str, int], dict[str, set[str]]]:
    required = {"standard_inchikey", "connectivity_block", smiles_column}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"identity frame is missing required columns: {missing}")
    if frame[list(required)].isna().any().any():
        raise ValueError("identity frame contains missing structural identities")
    full_keys = set(frame.standard_inchikey.astype(str))
    connectivity = set(frame.connectivity_block.astype(str))
    scaffolds, acyclic = murcko_fields(frame[smiles_column])
    counts = {
        "ligand_rows": int(len(frame)),
        "unique_standard_inchikeys": int(len(full_keys)),
        "unique_connectivity_blocks": int(len(connectivity)),
        "unique_nonempty_murcko_scaffolds": int(len(scaffolds)),
        "acyclic_ligand_rows": int(acyclic),
        # This matches the project's clustering convention without implying
        # that different acyclic molecules share a scaffold.
        "murcko_clusters_with_acyclic_singletons": int(len(scaffolds) + acyclic),
    }
    sets = {
        "standard_inchikey": full_keys,
        "connectivity_block": connectivity,
        "nonempty_murcko_scaffold": scaffolds,
    }
    return counts, sets


def pairwise_overlap_rows(
    panel_sets: dict[str, dict[str, set[str]]]
) -> list[dict[str, int | str]]:
    rows: list[dict[str, int | str]] = []
    for first, second in combinations(panel_sets, 2):
        row: dict[str, int | str] = {
            "first_panel": first,
            "second_panel": second,
        }
        for identity_level in (
            "standard_inchikey",
            "connectivity_block",
            "nonempty_murcko_scaffold",
        ):
            row[f"shared_{identity_level}s"] = int(
                len(panel_sets[first][identity_level] & panel_sets[second][identity_level])
            )
        rows.append(row)
    return rows


def load_davis_structures() -> pd.DataFrame:
    raw = pd.read_csv(
        davis.DEFAULT_DAVIS,
        sep="\t",
        usecols=["drug_name", "compound_iso_smiles"],
    ).drop_duplicates()
    if raw.shape != (72, 2) or raw.drug_name.duplicated().any():
        raise ValueError("expected 72 unique DAVIS drug structures")
    return davis._identity_table(raw, "compound_iso_smiles", scan="full")


def load_kirhub_boundary(path: Path) -> dict[str, Any]:
    source = json.loads(Path(path).read_text(encoding="utf-8"))
    audit = source.get("chemical_independence_audit", {})
    overlap = audit.get("overlap_count")
    geometry_value = audit.get("after_excluding_overlap", {}).get(
        "geometry_spearman_to_old_three_panel_mean"
    )
    comparison = audit.get("comparison")
    boundary = audit.get("no_structure_claim")
    if overlap != 11:
        raise ValueError("frozen KiRHub exact normalized-name overlap is not 11")
    if geometry_value != 0.8165798649269518:
        raise ValueError("frozen KiRHub post-removal geometry sensitivity changed")
    if comparison != "exact normalized drug name against the DAVIS 72-drug panel":
        raise ValueError("frozen KiRHub name-comparison definition changed")
    if not isinstance(boundary, str) or "do not provide structures" not in boundary:
        raise ValueError("frozen KiRHub structure boundary is missing")
    return {
        "ligand_rows": 92,
        "structures_available": False,
        "unique_standard_inchikeys": None,
        "unique_connectivity_blocks": None,
        "unique_nonempty_murcko_scaffolds": None,
        "exact_normalized_name_overlap_with_DAVIS": int(overlap),
        "name_comparison_definition": comparison,
        "geometry_spearman_to_DAVIS_PKIS2_PKIS1_mean_after_removing_11_DAVIS_names": float(
            geometry_value
        ),
        "boundary": boundary,
        "pkis_name_boundary": (
            "PKIS1/PKIS2 project identifiers are not comparable to KiRHub "
            "clinical compound names; no string-based PKIS--KiRHub overlap is asserted."
        ),
        "source_summary": str(Path(path).relative_to(PACKAGE)),
        "source_summary_sha256": sha256_file(path),
        "value_origin": "copied_and_schema_validated_without_recomputation",
    }


def run_analysis(output_dir: Path, pkis1_zip: Path, kirhub_summary: Path) -> dict[str, Any]:
    if sha256_file(pkis1_zip) != geometry.PKIS1_SHA256:
        raise ValueError("PKIS1 source ZIP checksum mismatch")
    frames = {
        "DAVIS": (load_davis_structures(), "compound_iso_smiles"),
        "PKIS2": (geometry.load_pkis2_full(), "Smiles"),
        "PKIS1": (geometry.load_pkis1_full(pkis1_zip), "SMILES"),
    }
    panel_counts: dict[str, dict[str, int]] = {}
    panel_sets: dict[str, dict[str, set[str]]] = {}
    for name, (frame, smiles_column) in frames.items():
        panel_counts[name], panel_sets[name] = structure_summary(frame, smiles_column)

    pairwise = pairwise_overlap_rows(panel_sets)
    triple = {
        f"shared_{level}s": int(
            len(set.intersection(*(panel_sets[name][level] for name in panel_sets)))
        )
        for level in (
            "standard_inchikey",
            "connectivity_block",
            "nonempty_murcko_scaffold",
        )
    }
    summary = {
        "analysis": "experimental_panel_overlap",
        "identity_definitions": {
            "standard_inchikey": "RDKit Chem.MolToInchiKey recomputed from released SMILES",
            "connectivity_block": "first 14 characters of recomputed Standard InChIKey",
            "nonempty_murcko_scaffold": (
                "canonical Bemis--Murcko scaffold; empty/acyclic molecules excluded "
                "from cross-panel scaffold intersections"
            ),
            "acyclic_handling": (
                "each acyclic ligand is a singleton for within-panel cluster counts; "
                "acyclic ligands are not treated as one shared scaffold across panels"
            ),
        },
        "structured_panels": panel_counts,
        "pairwise_structural_overlap": pairwise,
        "three_panel_structural_overlap": triple,
        "KiRHub_boundary": load_kirhub_boundary(kirhub_summary),
        "source_sha256": {
            "DAVIS": sha256_file(davis.DEFAULT_DAVIS),
            "PKIS2": sha256_file(geometry.pkis2.DEFAULT_PKIS2),
            "PKIS1": sha256_file(pkis1_zip),
        },
        "interpretation_boundary": (
            "Structural overlap is quantified only for DAVIS, PKIS2, and PKIS1. "
            "The KiRHub statement is a validated reuse of an existing name-only "
            "sensitivity and makes no chemical-identity or scaffold-independence claim."
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--pkis1-zip", type=Path, default=DEFAULT_PKIS1_ZIP)
    parser.add_argument("--kirhub-summary", type=Path, default=DEFAULT_KIRHUB_SUMMARY)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(
        json.dumps(
            run_analysis(args.output_dir, args.pkis1_zip, args.kirhub_summary),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
