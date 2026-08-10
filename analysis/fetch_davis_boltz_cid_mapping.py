#!/usr/bin/env python3
"""Build a safe DAVIS-name-to-Boltz-PubChem-CID mapping.

The legacy Boltz-2 arrays identify compounds by PubChem CID, whereas the frozen
DAVIS table identifies them by drug name and SMILES.  The original upstream
crosswalk is a Python pickle and is deliberately not loaded here.  Instead this
script asks PubChem for the Standard InChIKey of the 72 CIDs already present in
``affinity_out.npz`` and joins those keys to Standard InChIKeys recomputed from
the frozen DAVIS SMILES with RDKit.

Only the small, auditable CSV crosswalk is written.  No activity or prediction
values are transmitted.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem


PACKAGE = Path(__file__).resolve().parents[1]
DEFAULT_AFFINITY = (
    PACKAGE.parent
    / "project_clean"
    / "data"
    / "processed"
    / "davis_boltz2"
    / "affinity_out.npz"
)
DEFAULT_DAVIS = (
    PACKAGE
    / "data"
    / "frozen"
    / "project_clean"
    / "data"
    / "processed"
    / "davis_boltz2"
    / "davis_complete.tab.gz"
)
DEFAULT_OUTPUT = PACKAGE / "data" / "frozen" / "davis_boltz_cid_mapping.csv"
PUG_PROPERTY = "InChIKey,IsomericSMILES"

# Four upstream Boltz records are salts or a formulation/structure variant whose
# PubChem connectivity block differs from the neutral DAVIS graph.  The CIDs are
# resolved by the frozen upstream compound names and are kept explicit rather
# than inferred by a low-similarity nearest-neighbour rule.
NAME_TO_CID_OVERRIDES = {
    "AMG-706": "16097729",
    "BIBF-1120": "135423438",
    "PTK-787": "151193",
    "R406": "11984591",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def standard_inchikey(smiles: str) -> str:
    molecule = Chem.MolFromSmiles(str(smiles))
    if molecule is None:
        raise ValueError("RDKit could not parse a frozen DAVIS SMILES")
    key = Chem.MolToInchiKey(molecule)
    if not key:
        raise ValueError("RDKit could not generate a Standard InChIKey")
    return key


def fetch_pubchem_records(
    cids: list[str], timeout: float, cached_json: Path | None = None
) -> tuple[list[dict], str]:
    cid_string = ",".join(cids)
    url = (
        "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/"
        f"{cid_string}/property/{PUG_PROPERTY}/JSON"
    )
    if cached_json is not None:
        payload = json.loads(cached_json.read_text())
    else:
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "jcheminf-dimensionality-reproducibility/1.0"},
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.load(response)
    records = payload.get("PropertyTable", {}).get("Properties", [])
    return records, url


def build_mapping(
    affinity_path: Path,
    davis_path: Path,
    *,
    timeout: float = 120.0,
    pubchem_json: Path | None = None,
) -> tuple[pd.DataFrame, dict]:
    with np.load(affinity_path, allow_pickle=False) as archive:
        cids = sorted(set(archive["cids"].astype(str).tolist()), key=int)
    if len(cids) != 72:
        raise ValueError(f"Expected 72 Boltz CIDs, observed {len(cids)}")

    davis = pd.read_csv(
        davis_path,
        sep="\t",
        usecols=["drug_name", "compound_iso_smiles"],
    ).drop_duplicates()
    if len(davis) != 72 or davis.drug_name.duplicated().any():
        raise ValueError("Expected one frozen SMILES for each of 72 DAVIS drugs")
    davis["standard_inchikey"] = davis.compound_iso_smiles.map(standard_inchikey)
    if davis.standard_inchikey.duplicated().any():
        raise ValueError("Frozen DAVIS Standard InChIKeys are not unique")

    records, url = fetch_pubchem_records(cids, timeout, pubchem_json)
    pubchem = pd.DataFrame(records).rename(
        columns={"CID": "pubchem_cid", "InChIKey": "standard_inchikey"}
    )
    required = {"pubchem_cid", "standard_inchikey"}
    if not required.issubset(pubchem.columns):
        raise ValueError("PubChem response lacks CID or Standard InChIKey")
    pubchem["pubchem_cid"] = pubchem.pubchem_cid.astype(str)
    pubchem = pubchem[["pubchem_cid", "standard_inchikey"]].drop_duplicates()
    if len(pubchem) != 72 or set(pubchem.pubchem_cid) != set(cids):
        raise ValueError("PubChem did not return exactly the 72 requested CIDs")

    if pubchem.standard_inchikey.str[:14].duplicated().any():
        raise ValueError("PubChem connectivity blocks are not unique")
    exact_lookup = dict(zip(pubchem.standard_inchikey, pubchem.pubchem_cid))
    block_lookup = dict(
        zip(pubchem.standard_inchikey.str[:14], pubchem.pubchem_cid)
    )
    mapped_cids: list[str] = []
    methods: list[str] = []
    for record in davis.itertuples(index=False):
        key = str(record.standard_inchikey)
        if key in exact_lookup:
            mapped_cids.append(str(exact_lookup[key]))
            methods.append("exact_standard_inchikey")
        elif key[:14] in block_lookup:
            mapped_cids.append(str(block_lookup[key[:14]]))
            methods.append("exact_inchi_connectivity_block")
        elif str(record.drug_name) in NAME_TO_CID_OVERRIDES:
            mapped_cids.append(NAME_TO_CID_OVERRIDES[str(record.drug_name)])
            methods.append("explicit_upstream_name_salt_or_variant_override")
        else:
            raise ValueError(f"No auditable CID mapping for {record.drug_name}")
    davis["pubchem_cid"] = mapped_cids
    davis["match_method"] = methods
    if davis.pubchem_cid.duplicated().any():
        raise ValueError("The final DAVIS-to-CID mapping is not bijective")
    if set(davis.pubchem_cid) != set(cids):
        raise ValueError("The final DAVIS-to-CID mapping does not cover all Boltz CIDs")
    output = davis[
        ["drug_name", "standard_inchikey", "pubchem_cid", "match_method"]
    ].sort_values("standard_inchikey")
    metadata = {
        "retrieved_utc": datetime.now(timezone.utc).isoformat(),
        "source": "PubChem PUG REST",
        "request_url": url,
        "cached_response_sha256": sha256(pubchem_json) if pubchem_json else None,
        "join": (
            "exact Standard InChIKey, then exact connectivity block, with four "
            "explicitly named salt/formulation/structure-variant overrides"
        ),
        "match_method_counts": output.match_method.value_counts().to_dict(),
        "records": int(len(output)),
        "affinity_out_sha256": sha256(affinity_path),
        "davis_sha256": sha256(davis_path),
        "rdkit_version": Chem.rdBase.rdkitVersion,
    }
    return output, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--affinity", type=Path, default=DEFAULT_AFFINITY)
    parser.add_argument("--davis", type=Path, default=DEFAULT_DAVIS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument(
        "--pubchem-json",
        type=Path,
        default=None,
        help="Optional response previously downloaded from the recorded PUG REST URL",
    )
    arguments = parser.parse_args()

    mapping, metadata = build_mapping(
        arguments.affinity,
        arguments.davis,
        timeout=arguments.timeout,
        pubchem_json=arguments.pubchem_json,
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    mapping.to_csv(
        arguments.output,
        index=False,
        quoting=csv.QUOTE_MINIMAL,
        lineterminator="\n",
    )
    metadata_path = arguments.output.with_suffix(".provenance.json")
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(json.dumps({**metadata, "output": str(arguments.output)}, indent=2))


if __name__ == "__main__":
    main()
