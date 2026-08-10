#!/usr/bin/env python3
"""Freeze the dual chemical-identity contract for complete DOCKSTRING rows.

The docking reference and the SEA-style chemistry graph use related but
distinct representations.  This builder records both, in original source-row
order, so downstream leakage masks do not depend on an implicit cross-contract
comparison or on re-running RDKit standardisation during every analysis.

``raw_*`` fields reproduce the current-RDKit identity helper used for the
DOCKSTRING docking reference.  ``standardized_*`` fields use the exact SEA
contract: RDKit ``FragmentParent`` followed by ``Uncharger`` and a cyclic
Bemis--Murcko scaffold.  The frozen source InChIKey is retained independently.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import multiprocessing as mp
import os
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd
from rdkit import Chem, rdBase
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Scaffolds import MurckoScaffold


PACKAGE = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = PACKAGE / "data/frozen/dockstring-dataset.tsv.gz"
DEFAULT_OUTPUT = (
    PACKAGE / "data/frozen/dockstring_identity_contract_2026-08-03.csv.gz"
)
DEFAULT_PROVENANCE = (
    PACKAGE / "data/frozen/dockstring_identity_contract_2026-08-03.provenance.json"
)
EXPECTED_SOURCE_SHA256 = (
    "e15a58258dbd613374e499bb17e5428a0df3f1f53b34758042f8e3cd3b53eb64"
)
EXPECTED_SOURCE_ROWS = 260_155
EXPECTED_COMPLETE_ROWS = 260_060


@lru_cache(maxsize=None)
def standardized_identity(smiles: str) -> tuple[str, str] | None:
    """Return the graph-contract connectivity key and cyclic Murcko scaffold.

    This is the identity-only subset of the chemistry-graph standardisation
    contract.  It lives here so this release builder does not depend on the
    exploratory, gitignored graph implementation.
    """

    if not isinstance(smiles, str) or not smiles.strip():
        return None
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return None
    try:
        parent = rdMolStandardize.FragmentParent(molecule)
        parent = rdMolStandardize.Uncharger().uncharge(parent)
        Chem.SanitizeMol(parent)
    except (ValueError, RuntimeError):
        return None
    if parent.GetNumAtoms() == 0:
        return None

    standardized_smiles = Chem.MolToSmiles(
        parent, canonical=True, isomericSmiles=True
    )
    inchi_key = Chem.MolToInchiKey(parent)
    if not standardized_smiles or len(inchi_key) < 14:
        return None

    scaffold_molecule = MurckoScaffold.GetScaffoldForMol(parent)
    scaffold = ""
    if scaffold_molecule.GetNumAtoms() and scaffold_molecule.GetRingInfo().NumRings():
        scaffold = Chem.MolToSmiles(
            scaffold_molecule,
            canonical=True,
            isomericSmiles=False,
        )
    return inchi_key[:14], scaffold


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def derive_identity(smiles: str) -> tuple[str, str, str, str, str]:
    """Return raw InChIKey/connectivity/scaffold and SEA identities."""

    molecule = Chem.MolFromSmiles(str(smiles))
    if molecule is None:
        raise ValueError("invalid DOCKSTRING SMILES")
    raw_inchikey = Chem.MolToInchiKey(molecule)
    if not raw_inchikey or len(raw_inchikey) < 14:
        raise ValueError("could not generate a raw Standard InChIKey")
    raw_scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=molecule)

    standardized = standardized_identity(str(smiles))
    if standardized is None:
        raise ValueError(
            "graph-contract standardization failed for a DOCKSTRING SMILES"
        )
    standardized_connectivity, standardized_scaffold = standardized
    return (
        raw_inchikey,
        raw_inchikey[:14],
        raw_scaffold,
        standardized_connectivity,
        standardized_scaffold,
    )


def atomic_deterministic_gzip_csv(frame: pd.DataFrame, path: Path) -> None:
    """Write a byte-stable gzip CSV (empty gzip filename and mtime zero)."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = tempfile.NamedTemporaryFile(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False
    )
    temporary_path = Path(temporary.name)
    temporary.close()
    try:
        with temporary_path.open("wb") as raw_handle:
            with gzip.GzipFile(
                filename="",
                mode="wb",
                compresslevel=9,
                fileobj=raw_handle,
                mtime=0,
            ) as compressed:
                with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as text:
                    frame.to_csv(text, index=False, lineterminator="\n")
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def atomic_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = tempfile.NamedTemporaryFile(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False
    )
    temporary_path = Path(temporary.name)
    try:
        with io.TextIOWrapper(temporary, encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def build_contract(
    source_path: Path,
    output_path: Path,
    provenance_path: Path,
    *,
    workers: int,
) -> dict[str, Any]:
    source_path = Path(source_path)
    source_sha256 = sha256_file(source_path)
    if source_sha256 != EXPECTED_SOURCE_SHA256:
        raise ValueError(
            "unexpected DOCKSTRING source SHA256: "
            f"{source_sha256}; expected {EXPECTED_SOURCE_SHA256}"
        )

    source = pd.read_csv(source_path, sep="\t")
    required = {"inchikey", "smiles"}
    if not required.issubset(source):
        raise ValueError("DOCKSTRING source lacks inchikey or smiles")
    score_columns = [column for column in source if column not in required]
    complete_mask = ~source[score_columns].isna().any(axis=1)
    retained = source.loc[complete_mask, ["inchikey", "smiles"]].copy()
    if len(source) != EXPECTED_SOURCE_ROWS or len(retained) != EXPECTED_COMPLETE_ROWS:
        raise ValueError(
            "unexpected DOCKSTRING shape: "
            f"{len(source):,} source and {len(retained):,} complete rows"
        )

    worker_count = max(1, int(workers))
    smiles = retained["smiles"].astype(str).tolist()
    if worker_count == 1:
        derived = list(map(derive_identity, smiles))
    else:
        context = mp.get_context("spawn")
        with context.Pool(worker_count) as pool:
            derived = list(pool.imap(derive_identity, smiles, chunksize=256))

    identity = pd.DataFrame.from_records(
        derived,
        columns=[
            "raw_inchikey",
            "raw_connectivity",
            "raw_murcko_scaffold",
            "standardized_connectivity",
            "standardized_cyclic_murcko_scaffold",
        ],
    )
    artifact = pd.DataFrame(
        {
            "source_row_index": retained.index.to_numpy(dtype="int64"),
            "complete_support_row_index": range(len(retained)),
            "reported_inchikey": retained["inchikey"].astype(str).to_numpy(),
        }
    )
    artifact["reported_connectivity"] = artifact["reported_inchikey"].str[:14]
    artifact = pd.concat([artifact, identity], axis=1)
    if artifact.isna().any().any():
        raise AssertionError("identity artifact contains missing values")

    atomic_deterministic_gzip_csv(artifact, output_path)
    output_sha256 = sha256_file(output_path)
    audit = {
        "reported_vs_raw_full_inchikey_differences": int(
            artifact["reported_inchikey"].ne(artifact["raw_inchikey"]).sum()
        ),
        "reported_vs_raw_connectivity_differences": int(
            artifact["reported_connectivity"].ne(artifact["raw_connectivity"]).sum()
        ),
        "raw_vs_standardized_connectivity_differences": int(
            artifact["raw_connectivity"]
            .ne(artifact["standardized_connectivity"])
            .sum()
        ),
        "raw_vs_standardized_scaffold_differences": int(
            artifact["raw_murcko_scaffold"]
            .ne(artifact["standardized_cyclic_murcko_scaffold"])
            .sum()
        ),
    }
    provenance: dict[str, Any] = {
        "artifact_date": "2026-08-03",
        "schema_version": 1,
        "source": {
            "path": str(source_path.relative_to(PACKAGE)),
            "sha256": source_sha256,
            "rows": int(len(source)),
            "complete_rows": int(len(retained)),
            "incomplete_rows_excluded": int((~complete_mask).sum()),
            "score_columns": int(len(score_columns)),
        },
        "software": {
            "rdkit_version": rdBase.rdkitVersion,
            "pandas_version": pd.__version__,
        },
        "identity_contract": {
            "reported_inchikey": "verbatim frozen DOCKSTRING source value",
            "raw_connectivity": (
                "first 14 characters of the Standard InChIKey generated after "
                "Chem.MolFromSmiles, without FragmentParent or Uncharger"
            ),
            "raw_murcko_scaffold": (
                "MurckoScaffold.MurckoScaffoldSmiles on the raw parsed molecule"
            ),
            "standardized_connectivity": (
                "first Standard-InChIKey block after FragmentParent then Uncharger"
            ),
            "standardized_cyclic_murcko_scaffold": (
                "non-isomeric Murcko scaffold after FragmentParent then Uncharger; "
                "empty for acyclic compounds"
            ),
        },
        "audit": audit,
        "output": {
            "path": str(Path(output_path).relative_to(PACKAGE)),
            "sha256": output_sha256,
            "bytes": int(Path(output_path).stat().st_size),
            "rows": int(len(artifact)),
            "columns": list(artifact.columns),
            "gzip_mtime": 0,
        },
    }
    atomic_json(provenance, provenance_path)
    return provenance


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--provenance", type=Path, default=DEFAULT_PROVENANCE)
    parser.add_argument(
        "--workers", type=int, default=min(6, os.cpu_count() or 1)
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    provenance = build_contract(
        args.source,
        args.output,
        args.provenance,
        workers=args.workers,
    )
    print(json.dumps(provenance, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
