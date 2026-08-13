#!/usr/bin/env python3
"""Chemical-identity and dependence audit for Anastassiadis et al. (2011).

This producer deliberately keeps the HotSpot assay matrix outside the release.
It reads the official Supplementary Table 3 workbook only to recover the 178
compound labels/CAS numbers and to identify the 176 compounds complete on the
fixed 21-kinase comparison support.  Chemical structures are resolved against
the official PubChem PUG REST service and frozen as an identity-only table.

The release audit then:

* reports every CAS/name query and every candidate PubChem CID;
* fails closed when the two queries leave more than one defensible CID;
* recomputes Standard InChIKeys with the pinned RDKit version;
* assigns Morgan-radius-2/2048-bit Butina clusters at Tanimoto >= 0.5 and
  Bemis--Murcko scaffolds; and
* quantifies full-Standard-InChIKey and connectivity overlap with the complete
  DOCKSTRING, DAVIS, and PKIS2 compound supports.

No HotSpot assay value is written to ``data/frozen`` or ``results``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import warnings
from collections import OrderedDict
from datetime import date
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.ML.Cluster import Butina


PACKAGE = Path(__file__).resolve().parents[1]
DEFAULT_WORKBOOK = Path(
    "/tmp/anastassiadis2011/41587_2011_BFnbt2017_MOESM23_ESM.xls"
)
DEFAULT_IDENTITY = (
    PACKAGE / "data" / "frozen" / "anastassiadis2011_pubchem_identity_2026-08-10.csv"
)
DEFAULT_PROVENANCE = DEFAULT_IDENTITY.with_suffix(".provenance.json")
DEFAULT_OUTPUT = PACKAGE / "results" / "public_anastassiadis_identity_audit"
DEFAULT_DOCKSTRING_IDENTITY = (
    PACKAGE / "data" / "frozen" / "dockstring_identity_contract_2026-08-03.csv.gz"
)
DEFAULT_DAVIS_IDENTITY = (
    PACKAGE / "data" / "frozen" / "davis_boltz_cid_mapping.csv"
)
DEFAULT_PKIS2 = PACKAGE / "data" / "frozen" / "pkis2_s4.xlsx"

WORKBOOK_SHA256 = "cd756bf2b6ad541a1781508c563caf0da6da876dfb71f2546fbff02e13d98684"
WORKBOOK_URL = (
    "https://media.springernature.com/original/springer-static/"
    "esm/art%3A10.1038%2Fnbt.2017/MediaObjects/"
    "41587_2011_BFnbt2017_MOESM23_ESM.xls"
)
ARTICLE_DOI = "10.1038/nbt.2017"
PUBCHEM_BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
PUBCHEM_DOWNLOAD_TERMS = "https://pubchem.ncbi.nlm.nih.gov/docs/downloads"
NCBI_MOLECULAR_DATA_POLICY = "https://www.ncbi.nlm.nih.gov/home/about/policies/"
ACCESS_DATE = date(2026, 8, 10).isoformat()

# Exact aliases for the fixed public 21-target comparison.  CDK2 is the
# per-compound arithmetic mean of the cyclin-A and cyclin-E rows in the panel
# validation, with ``skipna=False``.  Completeness therefore requires both
# source cells even though no assay value is exported by this identity audit.
HOTSPOT_TARGET_MAP: OrderedDict[str, tuple[str, ...]] = OrderedDict(
    [
        ("ABL1", ("ABL1",)),
        ("AKT1", ("AKT1",)),
        ("AKT2", ("AKT2",)),
        ("CDK2", ("CDK2/cyclin A", "CDK2/cyclin E")),
        ("CSF1R", ("FMS",)),
        ("EGFR", ("EGFR",)),
        ("FGFR1", ("FGFR1",)),
        ("IGF1R", ("IGF1R",)),
        ("JAK2", ("JAK2",)),
        ("KDR", ("KDR/VEGFR2",)),
        ("KIT", ("c-Kit",)),
        ("LCK", ("LCK",)),
        ("MAP2K1", ("MEK1",)),
        ("MAPK1", ("ERK2 MAPK1",)),
        ("MAPK14", ("P38a/MAPK14",)),
        ("MAPKAPK2", ("MAPKAPK2",)),
        ("MET", ("c-MET",)),
        ("PLK1", ("PLK1",)),
        ("PTK2", ("FAK/PTK2",)),
        ("ROCK1", ("ROCK1",)),
        ("SRC", ("c-SRC",)),
    ]
)

EXPECTED_INCOMPLETE_PAIRS = {
    ("SB 202474", "AKT1"),
    ("VEGF Receptor 2 Kinase Inhibitor II", "MAPK14"),
}

PKIS2_SHEET = "Table 4 - PKIS2 %Inh"
CAS_PATTERN = re.compile(r"^[0-9]{2,7}-[0-9]{2}-[0-9]$")
OUTPUT_FILES = (
    "README.md",
    "chemical_dependence_summary.csv",
    "cluster_assignments.csv",
    "complete21_support.csv",
    "cross_panel_overlap_records.csv",
    "cross_panel_overlap_summary.csv",
    "identity_resolution_audit.csv",
    "summary.json",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, float_format="%.12g", lineterminator="\n")
    temporary.replace(path)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(json_ready(payload), indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def normalize_label(value: Any) -> str:
    return " ".join(str(value).replace("\u00a0", " ").split()).strip()


def cas_checksum_valid(value: str) -> bool:
    cas = normalize_label(value)
    if not CAS_PATTERN.fullmatch(cas):
        return False
    digits = cas.replace("-", "")
    check = sum(
        multiplier * int(digit)
        for multiplier, digit in enumerate(reversed(digits[:-1]), start=1)
    ) % 10
    return check == int(digits[-1])


def parse_workbook_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Extract labels and fixed-21 completeness without exporting assay values."""
    if frame.shape[0] < 303 or frame.shape[1] < 179:
        raise ValueError(f"unexpected HotSpot workbook shape: {frame.shape}")
    names = [normalize_label(value) for value in frame.iloc[1, 1:179]]
    cas_numbers = [normalize_label(value) for value in frame.iloc[2, 1:179]]
    if len(set(names)) != 178 or len(set(cas_numbers)) != 178:
        raise ValueError("the 178 workbook compound name/CAS records are not unique")
    invalid_cas = [value for value in cas_numbers if not cas_checksum_valid(value)]
    if invalid_cas:
        raise ValueError(f"invalid CAS values in workbook: {invalid_cas}")

    row_labels = pd.Series(
        [normalize_label(value) for value in frame.iloc[3:, 0]],
        index=frame.index[3:],
    )
    if row_labels.duplicated().any():
        duplicates = row_labels[row_labels.duplicated(False)].tolist()
        raise ValueError(f"duplicate HotSpot target labels: {duplicates}")
    row_lookup = dict(zip(row_labels, row_labels.index))
    missing_aliases = [
        alias
        for aliases in HOTSPOT_TARGET_MAP.values()
        for alias in aliases
        if alias not in row_lookup
    ]
    if missing_aliases:
        raise ValueError(f"fixed-21 aliases absent from workbook: {missing_aliases}")

    target_rows = {
        target: [row_lookup[alias] for alias in aliases]
        for target, aliases in HOTSPOT_TARGET_MAP.items()
    }
    flat_target_rows = [row for rows in target_rows.values() for row in rows]
    fixed_values = frame.loc[flat_target_rows, list(range(1, 179))].apply(
        pd.to_numeric, errors="coerce"
    )
    records: list[dict[str, Any]] = []
    observed_missing_pairs: set[tuple[str, str]] = set()
    for compound_zero, column in enumerate(range(1, 179)):
        missing_targets = [
            target
            for target, rows in target_rows.items()
            if fixed_values.loc[rows, column].isna().any()
        ]
        for target in missing_targets:
            observed_missing_pairs.add((names[compound_zero], target))
        records.append(
            {
                "workbook_compound_index_1based": compound_zero + 1,
                "workbook_excel_column_1based": column + 1,
                "workbook_compound_name": names[compound_zero],
                "workbook_cas": cas_numbers[compound_zero],
                "complete_fixed21": not missing_targets,
                "missing_fixed21_targets": ";".join(missing_targets),
            }
        )
    compounds = pd.DataFrame(records)
    if int(compounds.complete_fixed21.sum()) != 176:
        raise ValueError(
            "expected 176 compounds complete on fixed 21 targets, observed "
            f"{int(compounds.complete_fixed21.sum())}"
        )
    if observed_missing_pairs != EXPECTED_INCOMPLETE_PAIRS:
        raise ValueError(
            "fixed-21 missing pairs differ from the audited contract: "
            f"{sorted(observed_missing_pairs)}"
        )
    metadata = {
        "workbook_rows": int(frame.shape[0]),
        "workbook_columns": int(frame.shape[1]),
        "compound_records": 178,
        "fixed21_targets": list(HOTSPOT_TARGET_MAP),
        "fixed21_workbook_aliases": {
            target: list(aliases) for target, aliases in HOTSPOT_TARGET_MAP.items()
        },
        "complete_fixed21_compounds": 176,
        "incomplete_fixed21_pairs": [list(item) for item in sorted(observed_missing_pairs)],
    }
    return compounds, metadata


def load_workbook(path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    path = Path(path)
    if sha256_file(path) != WORKBOOK_SHA256:
        raise ValueError("official Anastassiadis Supplementary Table 3 SHA-256 mismatch")
    frame = pd.read_excel(path, header=None)
    return parse_workbook_frame(frame)


class PubChemClient:
    """Bounded, serial PUG REST client with explicit rate limiting."""

    def __init__(
        self,
        *,
        min_interval_seconds: float = 0.22,
        timeout_seconds: float = 60.0,
        max_retries: int = 4,
    ) -> None:
        self.min_interval_seconds = float(min_interval_seconds)
        self.timeout_seconds = float(timeout_seconds)
        self.max_retries = int(max_retries)
        self._last_request = 0.0
        self.request_count = 0
        self.retry_count = 0

    def get_json(self, url: str) -> dict[str, Any]:
        for attempt in range(self.max_retries + 1):
            delay = self.min_interval_seconds - (time.monotonic() - self._last_request)
            if delay > 0:
                time.sleep(delay)
            request = urllib.request.Request(
                url,
                headers={
                    "Accept": "application/json",
                    "User-Agent": "jcheminf-dimensionality-identity-audit/1.0",
                },
            )
            self._last_request = time.monotonic()
            self.request_count += 1
            try:
                with urllib.request.urlopen(
                    request, timeout=self.timeout_seconds
                ) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                if not isinstance(payload, dict):
                    raise ValueError("PubChem returned a non-object JSON payload")
                return payload
            except urllib.error.HTTPError as error:
                if error.code == 404:
                    return {}
                if error.code not in {429, 500, 502, 503, 504} or attempt >= self.max_retries:
                    raise
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
                if attempt >= self.max_retries:
                    raise
            self.retry_count += 1
            time.sleep(float(2**attempt))
        raise RuntimeError("unreachable PubChem retry loop")

    def name_cids(self, query: str) -> tuple[list[int], str]:
        encoded = urllib.parse.quote(normalize_label(query), safe="")
        url = f"{PUBCHEM_BASE}/compound/name/{encoded}/cids/JSON"
        payload = self.get_json(url)
        cids = payload.get("IdentifierList", {}).get("CID", [])
        return sorted({int(value) for value in cids}), url

    def properties(self, cids: Sequence[int]) -> dict[int, dict[str, Any]]:
        properties: dict[int, dict[str, Any]] = {}
        for start in range(0, len(cids), 100):
            batch = [int(value) for value in cids[start : start + 100]]
            identifiers = ",".join(str(value) for value in batch)
            url = (
                f"{PUBCHEM_BASE}/compound/cid/{identifiers}/property/"
                "SMILES,ConnectivitySMILES,InChIKey,IUPACName/JSON"
            )
            payload = self.get_json(url)
            rows = payload.get("PropertyTable", {}).get("Properties", [])
            for row in rows:
                cid = int(row["CID"])
                if cid in properties:
                    raise ValueError(f"duplicate PubChem property row for CID {cid}")
                properties[cid] = dict(row)
        missing = sorted(set(map(int, cids)) - set(properties))
        if missing:
            raise ValueError(f"PubChem property retrieval omitted CIDs: {missing}")
        return properties


def resolve_candidate_cids(
    cas_cids: Sequence[int], name_cids: Sequence[int]
) -> tuple[int | None, str, bool, str]:
    """Select a CID only when the CAS/name evidence leaves one choice.

    CAS is the primary identifier because each workbook CAS passes its formal
    checksum.  A unique CAS hit is accepted when the text name is absent from
    PubChem or contains that same CID.  A disjoint name hit is a fail-closed
    conflict, rather than an excuse to choose whichever record is convenient.
    """
    cas = sorted(set(map(int, cas_cids)))
    name = sorted(set(map(int, name_cids)))
    intersection = sorted(set(cas) & set(name))
    if len(cas) == 1:
        if not name:
            return cas[0], "resolved_unique_cas_name_not_found", False, "cas_unique"
        if cas[0] in name:
            return cas[0], "resolved_unique_cas_name_agrees", False, "cas_name_agree"
        return None, "unresolved_cas_name_conflict", True, "disjoint_candidate_sets"
    if len(cas) > 1:
        if len(intersection) == 1:
            return (
                intersection[0],
                "resolved_unique_cas_name_intersection",
                False,
                "unique_intersection",
            )
        return None, "unresolved_ambiguous_cas", True, "nonunique_intersection"
    if len(name) == 1:
        return name[0], "resolved_unique_name_only", False, "cas_not_found"
    if len(name) > 1:
        return None, "unresolved_ambiguous_name", True, "multiple_name_candidates"
    return None, "unresolved_not_found", False, "no_candidate"


def rdkit_identity(smiles: str) -> dict[str, Any]:
    molecule = Chem.MolFromSmiles(str(smiles))
    if molecule is None:
        raise ValueError(f"RDKit could not parse PubChem SMILES: {smiles!r}")
    key = Chem.MolToInchiKey(molecule)
    if not key or len(key) != 27:
        raise ValueError(f"RDKit could not generate Standard InChIKey: {smiles!r}")
    return {
        "rdkit_standard_inchikey": key,
        "rdkit_connectivity": key[:14],
        "rdkit_canonical_isomeric_smiles": Chem.MolToSmiles(
            molecule, canonical=True, isomericSmiles=True
        ),
        "rdkit_canonical_nonisomeric_smiles": Chem.MolToSmiles(
            molecule, canonical=True, isomericSmiles=False
        ),
        "structure_fragment_count": len(Chem.GetMolFrags(molecule)),
    }


def retrieve_identity_table(
    compounds: pd.DataFrame, client: PubChemClient
) -> tuple[pd.DataFrame, dict[str, Any]]:
    query_rows: list[dict[str, Any]] = []
    for row in compounds.itertuples(index=False):
        cas_cids, cas_url = client.name_cids(row.workbook_cas)
        name_cids, name_url = client.name_cids(row.workbook_compound_name)
        selected, status, ambiguity, decision = resolve_candidate_cids(
            cas_cids, name_cids
        )
        query_rows.append(
            {
                **row._asdict(),
                "pubchem_cas_query": row.workbook_cas,
                "pubchem_cas_query_cids": ";".join(map(str, cas_cids)),
                "pubchem_name_query": row.workbook_compound_name,
                "pubchem_name_query_cids": ";".join(map(str, name_cids)),
                "pubchem_cas_query_url": cas_url,
                "pubchem_name_query_url": name_url,
                "pubchem_cid": selected,
                "resolution_status": status,
                "resolution_ambiguity": bool(ambiguity),
                "resolution_decision": decision,
                "pubchem_access_date": ACCESS_DATE,
            }
        )
    identity = pd.DataFrame(query_rows)
    selected_cids = sorted(
        {int(value) for value in identity.pubchem_cid.dropna().astype(int)}
    )
    # Retrieve properties for every candidate, not only the selected CID.  The
    # union of candidate connectivity blocks is a conservative exclusion key
    # for unresolved records: no plausible HotSpot identity may leak into a
    # DOCKSTRING calibration pool merely because exact identity was ambiguous.
    all_candidate_cids: set[int] = set(selected_cids)
    for column in ["pubchem_cas_query_cids", "pubchem_name_query_cids"]:
        for value in identity[column].fillna("").astype(str):
            all_candidate_cids.update(
                int(item) for item in value.split(";") if item.strip()
            )
    properties = client.properties(sorted(all_candidate_cids))
    enriched: list[dict[str, Any]] = []
    for record in identity.to_dict(orient="records"):
        cid_value = record["pubchem_cid"]
        candidate_cids = sorted(
            {
                int(item)
                for column in [
                    "pubchem_cas_query_cids",
                    "pubchem_name_query_cids",
                ]
                for item in str(record[column]).split(";")
                if item and item.lower() != "nan"
            }
        )
        candidate_keys = sorted(
            {
                str(properties[cid].get("InChIKey", ""))
                for cid in candidate_cids
                if str(properties[cid].get("InChIKey", ""))
            }
        )
        candidate_connectivities = sorted(
            {value[:14] for value in candidate_keys if len(value) >= 14}
        )
        record.update(
            {
                "candidate_pubchem_cids": ";".join(map(str, candidate_cids)),
                "candidate_pubchem_inchikeys": ";".join(candidate_keys),
                "candidate_connectivity_blocks": ";".join(
                    candidate_connectivities
                ),
            }
        )
        if pd.isna(cid_value):
            record.update(
                {
                    "pubchem_smiles": "",
                    "pubchem_connectivity_smiles": "",
                    "pubchem_inchikey": "",
                    "pubchem_iupac_name": "",
                    "rdkit_standard_inchikey": "",
                    "rdkit_connectivity": "",
                    "rdkit_canonical_isomeric_smiles": "",
                    "rdkit_canonical_nonisomeric_smiles": "",
                    "structure_fragment_count": np.nan,
                    "pubchem_rdkit_inchikey_agree": False,
                    "structure_usable": False,
                    "candidate_only_connectivity_blocks": ";".join(
                        candidate_connectivities
                    ),
                    "de_leakage_connectivity_blocks": ";".join(
                        candidate_connectivities
                    ),
                }
            )
            enriched.append(record)
            continue
        cid = int(cid_value)
        prop = properties[cid]
        smiles = str(prop.get("SMILES", ""))
        connectivity_smiles = str(prop.get("ConnectivitySMILES", ""))
        pubchem_key = str(prop.get("InChIKey", ""))
        if not smiles or not connectivity_smiles or not pubchem_key:
            raise ValueError(f"PubChem CID {cid} lacks a required structure property")
        computed = rdkit_identity(smiles)
        agree = computed["rdkit_standard_inchikey"] == pubchem_key
        record.update(
            {
                "pubchem_smiles": smiles,
                "pubchem_connectivity_smiles": connectivity_smiles,
                "pubchem_inchikey": pubchem_key,
                "pubchem_iupac_name": str(prop.get("IUPACName", "")),
                **computed,
                "pubchem_rdkit_inchikey_agree": bool(agree),
                "structure_usable": bool(agree and not record["resolution_ambiguity"]),
                "candidate_only_connectivity_blocks": "",
                "de_leakage_connectivity_blocks": computed[
                    "rdkit_connectivity"
                ],
            }
        )
        enriched.append(record)
    result = pd.DataFrame(enriched).sort_values(
        "workbook_compound_index_1based", kind="stable"
    )
    if len(result) != 178:
        raise RuntimeError("identity retrieval did not preserve 178 workbook records")
    metadata = {
        "pubchem_requests": client.request_count,
        "pubchem_retries": client.retry_count,
        "selected_unique_cids": len(selected_cids),
        "candidate_unique_cids": len(all_candidate_cids),
        "resolved_records": int(result.pubchem_cid.notna().sum()),
        "usable_structures": int(result.structure_usable.sum()),
        "ambiguous_records": int(result.resolution_ambiguity.sum()),
        "status_counts": {
            str(key): int(value)
            for key, value in result.resolution_status.value_counts().sort_index().items()
        },
    }
    return result, metadata


def validate_frozen_identity(identity: pd.DataFrame) -> pd.DataFrame:
    required = {
        "workbook_compound_index_1based",
        "workbook_compound_name",
        "workbook_cas",
        "complete_fixed21",
        "pubchem_cid",
        "resolution_status",
        "resolution_ambiguity",
        "pubchem_smiles",
        "pubchem_connectivity_smiles",
        "pubchem_inchikey",
        "rdkit_standard_inchikey",
        "rdkit_connectivity",
        "structure_usable",
        "candidate_pubchem_cids",
        "candidate_pubchem_inchikeys",
        "candidate_connectivity_blocks",
        "candidate_only_connectivity_blocks",
        "de_leakage_connectivity_blocks",
    }
    missing = sorted(required - set(identity.columns))
    if missing:
        raise ValueError(f"frozen identity table lacks columns: {missing}")
    if len(identity) != 178:
        raise ValueError(f"expected 178 identity rows, observed {len(identity)}")
    identity = identity.sort_values("workbook_compound_index_1based", kind="stable")
    if identity.workbook_compound_index_1based.tolist() != list(range(1, 179)):
        raise ValueError("identity rows are not a one-to-one ordered workbook mapping")
    for column in ["complete_fixed21", "resolution_ambiguity", "structure_usable"]:
        if identity[column].dtype != bool:
            normalized = identity[column].astype(str).str.lower().map(
                {"true": True, "false": False}
            )
            if normalized.isna().any():
                raise ValueError(f"invalid Boolean values in {column}")
            identity[column] = normalized.astype(bool)
    if int(identity.complete_fixed21.sum()) != 176:
        raise ValueError("frozen identity table no longer identifies 176 complete rows")
    usable = identity[identity.structure_usable]
    if usable.rdkit_standard_inchikey.str.len().ne(27).any():
        raise ValueError("a usable structure lacks a full Standard InChIKey")
    if usable.rdkit_connectivity.str.len().ne(14).any():
        raise ValueError("a usable structure lacks a connectivity block")
    unresolved = identity[~identity.structure_usable]
    if unresolved.de_leakage_connectivity_blocks.fillna("").str.len().eq(0).any():
        raise ValueError("an unresolved identity lacks conservative candidate blocks")
    if not (
        unresolved.de_leakage_connectivity_blocks.fillna("")
        == unresolved.candidate_only_connectivity_blocks.fillna("")
    ).all():
        raise ValueError("unresolved de-leakage blocks are not the candidate union")
    return identity.reset_index(drop=True)


def chemical_group_assignments(identity: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    usable = identity[identity.structure_usable].copy()
    molecules = [Chem.MolFromSmiles(value) for value in usable.pubchem_smiles]
    if any(molecule is None for molecule in molecules):
        raise ValueError("a frozen usable PubChem SMILES no longer parses with RDKit")
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fingerprints = [generator.GetFingerprint(molecule) for molecule in molecules]
    distances: list[float] = []
    for index in range(1, len(fingerprints)):
        similarities = DataStructs.BulkTanimotoSimilarity(
            fingerprints[index], fingerprints[:index]
        )
        distances.extend(1.0 - float(value) for value in similarities)
    clusters = Butina.ClusterData(
        distances,
        len(fingerprints),
        0.5,
        isDistData=True,
        reordering=True,
    )
    labels = np.full(len(usable), -1, dtype=int)
    for cluster_id, members in enumerate(clusters):
        labels[list(members)] = cluster_id
    if (labels < 0).any():
        raise RuntimeError("Butina did not assign every usable structure")
    usable["butina_r2_2048_tanimoto50_cluster"] = [
        f"B{value:03d}" for value in labels
    ]
    scaffolds: list[str] = []
    for molecule, connectivity in zip(molecules, usable.rdkit_connectivity):
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=molecule)
        scaffolds.append(scaffold if scaffold else f"ACYCLIC:{connectivity}")
    usable["murcko_scaffold"] = scaffolds
    butina_sizes = usable.butina_r2_2048_tanimoto50_cluster.value_counts()
    scaffold_sizes = usable.murcko_scaffold.value_counts()
    usable["butina_cluster_size"] = usable[
        "butina_r2_2048_tanimoto50_cluster"
    ].map(butina_sizes)
    usable["murcko_scaffold_size"] = usable.murcko_scaffold.map(scaffold_sizes)
    summary = {
        "usable_structures": int(len(usable)),
        "butina_clusters": int(len(butina_sizes)),
        "butina_singleton_clusters": int((butina_sizes == 1).sum()),
        "compounds_in_butina_singletons": int(butina_sizes[butina_sizes == 1].sum()),
        "murcko_groups": int(len(scaffold_sizes)),
        "murcko_singleton_groups": int((scaffold_sizes == 1).sum()),
        "compounds_in_murcko_singletons": int(scaffold_sizes[scaffold_sizes == 1].sum()),
        "acyclic_compounds": int(usable.murcko_scaffold.str.startswith("ACYCLIC:").sum()),
        "multi_fragment_compounds": int((usable.structure_fragment_count > 1).sum()),
        "butina_contract": {
            "fingerprint": "RDKit Morgan",
            "radius": 2,
            "fp_size": 2048,
            "similarity": "Tanimoto",
            "minimum_within_cluster_similarity_to_seed": 0.5,
            "distance_threshold": 0.5,
            "reordering": True,
            "input_order": "official workbook compound order",
        },
        "murcko_contract": (
            "RDKit MurckoScaffoldSmiles on the complete PubChem compound; "
            "acyclic compounds receive identity-specific ACYCLIC:<connectivity> labels"
        ),
    }
    return usable, summary


def load_reference_identities(
    dockstring_path: Path, davis_path: Path, pkis2_path: Path
) -> dict[str, pd.DataFrame]:
    dockstring = pd.read_csv(dockstring_path)
    if len(dockstring) != 260_060:
        raise ValueError(
            f"expected complete 260,060-row DOCKSTRING support, observed {len(dockstring)}"
        )
    dock = pd.DataFrame(
        {
            "source_row": dockstring.source_row_index.astype(int),
            "standard_inchikey": dockstring.raw_inchikey.astype(str),
            "connectivity": dockstring.raw_connectivity.astype(str),
        }
    )

    davis = pd.read_csv(davis_path)
    if len(davis) != 72 or davis.drug_name.nunique() != 72:
        raise ValueError("expected the complete 72-compound DAVIS identity support")
    dav = pd.DataFrame(
        {
            "source_row": np.arange(len(davis), dtype=int),
            "standard_inchikey": davis.standard_inchikey.astype(str),
            "connectivity": davis.standard_inchikey.astype(str).str[:14],
        }
    )

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Unknown extension is not supported and will be removed",
            category=UserWarning,
            module=r"openpyxl\.worksheet\._reader",
        )
        pkis2 = pd.read_excel(pkis2_path, sheet_name=PKIS2_SHEET)
    required = ["Regno", "Compound", "Chemotype", "Smiles"]
    pkis2 = pkis2.dropna(subset=required).copy()
    if len(pkis2) != 645:
        raise ValueError(f"expected complete 645-row PKIS2 support, observed {len(pkis2)}")
    keys: list[str] = []
    for smiles in pkis2.Smiles.astype(str):
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            raise ValueError(f"invalid released PKIS2 SMILES: {smiles!r}")
        key = Chem.MolToInchiKey(molecule)
        if not key:
            raise ValueError("RDKit could not identify a released PKIS2 structure")
        keys.append(key)
    pkis = pd.DataFrame(
        {
            "source_row": np.arange(len(pkis2), dtype=int),
            "standard_inchikey": keys,
            "connectivity": [value[:14] for value in keys],
        }
    )
    return {"DOCKSTRING": dock, "DAVIS": dav, "PKIS2": pkis}


def overlap_audit(
    identity: pd.DataFrame, references: dict[str, pd.DataFrame]
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    records: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    usable = identity[identity.structure_usable].copy()
    for panel, reference in references.items():
        full_index = reference.groupby("standard_inchikey").source_row.apply(list)
        connectivity_index = reference.groupby("connectivity").source_row.apply(list)
        for row in identity.itertuples(index=False):
            de_leakage_blocks = [
                value
                for value in str(row.de_leakage_connectivity_blocks).split(";")
                if value and value.lower() != "nan"
            ]
            conservative_rows = sorted(
                {
                    int(source_row)
                    for block in de_leakage_blocks
                    for source_row in connectivity_index.get(block, [])
                }
            )
            if not row.structure_usable:
                exact_rows: list[int] = []
                connectivity_rows: list[int] = []
            else:
                exact_rows = list(full_index.get(row.rdkit_standard_inchikey, []))
                connectivity_rows = list(
                    connectivity_index.get(row.rdkit_connectivity, [])
                )
            records.append(
                {
                    "panel": panel,
                    "workbook_compound_index_1based": row.workbook_compound_index_1based,
                    "workbook_compound_name": row.workbook_compound_name,
                    "workbook_cas": row.workbook_cas,
                    "complete_fixed21": row.complete_fixed21,
                    "structure_usable": row.structure_usable,
                    "rdkit_standard_inchikey": row.rdkit_standard_inchikey,
                    "rdkit_connectivity": row.rdkit_connectivity,
                    "exact_full_standard_inchikey_overlap": bool(exact_rows),
                    "connectivity_overlap": bool(connectivity_rows),
                    "connectivity_only_overlap": bool(
                        connectivity_rows and not exact_rows
                    ),
                    "conservative_de_leakage_connectivity_overlap": bool(
                        conservative_rows
                    ),
                    "matching_full_rows": ";".join(map(str, exact_rows)),
                    "matching_connectivity_rows": ";".join(
                        map(str, connectivity_rows)
                    ),
                    "matching_conservative_de_leakage_rows": ";".join(
                        map(str, conservative_rows)
                    ),
                }
            )
        panel_records = pd.DataFrame(records)
        panel_records = panel_records[panel_records.panel == panel]
        for support, mask in [
            ("all_178", np.ones(len(panel_records), dtype=bool)),
            ("complete_fixed21_176", panel_records.complete_fixed21.to_numpy(bool)),
        ]:
            subset = panel_records.loc[mask]
            summaries.append(
                {
                    "panel": panel,
                    "hotspot_support": support,
                    "hotspot_records": int(len(subset)),
                    "usable_hotspot_structures": int(subset.structure_usable.sum()),
                    "reference_rows": int(len(reference)),
                    "reference_unique_full_inchikeys": int(
                        reference.standard_inchikey.nunique()
                    ),
                    "reference_unique_connectivities": int(reference.connectivity.nunique()),
                    "exact_full_standard_inchikey_overlap": int(
                        subset.exact_full_standard_inchikey_overlap.sum()
                    ),
                    "connectivity_overlap": int(subset.connectivity_overlap.sum()),
                    "connectivity_only_overlap": int(
                        subset.connectivity_only_overlap.sum()
                    ),
                    "conservative_de_leakage_connectivity_overlap": int(
                        subset.conservative_de_leakage_connectivity_overlap.sum()
                    ),
                }
            )
    record_frame = pd.DataFrame(records).sort_values(
        ["panel", "workbook_compound_index_1based"], kind="stable"
    )
    summary_frame = pd.DataFrame(summaries).sort_values(
        ["panel", "hotspot_support"], kind="stable"
    )
    metadata = {
        panel: {
            "rows": int(len(frame)),
            "unique_full_standard_inchikeys": int(frame.standard_inchikey.nunique()),
            "unique_connectivities": int(frame.connectivity.nunique()),
        }
        for panel, frame in references.items()
    }
    return record_frame, summary_frame, metadata


def output_checksums(directory: Path) -> dict[str, str]:
    return {name: sha256_file(directory / name) for name in OUTPUT_FILES}


def make_readme(summary: dict[str, Any]) -> str:
    resolution = summary["identity_resolution"]
    groups = summary["chemical_dependence"]
    return f"""# Anastassiadis 2011 identity and chemical-dependence audit

This standalone audit maps all 178 compound name/CAS records from official
Supplementary Table 3 (DOI `{ARTICLE_DOI}`) to PubChem. It exports **no HotSpot
assay values**. The fixed 21-target completeness flag was computed in memory:
{summary['workbook']['complete_fixed21_compounds']} compounds are complete and
the two excluded compound--target cells are listed in `summary.json`.

## Identity contract

CAS and workbook name were queried independently against PubChem PUG REST on
{ACCESS_DATE}. Candidate lists, query URLs, the resolution status, and ambiguity
flags are retained in `identity_resolution_audit.csv`. A structure is usable only
when the resolution rule selects one CID and the PubChem InChIKey agrees exactly
with the Standard InChIKey recomputed from the returned PubChem SMILES by RDKit.
There are {resolution['usable_structures']} usable structures and
{resolution['ambiguous_records']} ambiguity-flagged records.
For every unresolved record, `candidate_only_connectivity_blocks` contains the
union from all CAS/name candidate CIDs. `de_leakage_connectivity_blocks` is the
field to use for conservative calibration-pool exclusion: one selected block
for resolved records and the complete candidate union otherwise.

## Dependence contract

Usable structures were fingerprinted as Morgan radius 2, 2048 bits. Butina
clustering used Tanimoto distance <= 0.5 (`reordering=True`) in official workbook
order. This gives {groups['butina_clusters']} clusters, including
{groups['butina_singleton_clusters']} singleton clusters. RDKit Bemis--Murcko
grouping gives {groups['murcko_groups']} groups, including
{groups['murcko_singleton_groups']} singletons. Acyclic structures receive
identity-specific `ACYCLIC:<connectivity>` labels rather than sharing an empty
scaffold label. For whole-panel multiplier resampling, each of the
{groups['unresolved_identity_singleton_groups']} unresolved records receives an
identity-specific singleton, giving {groups['total_cluster_multiplier_groups']}
groups across all 178 records.

## Cross-panel overlap

`cross_panel_overlap_summary.csv` reports exact full-Standard-InChIKey and
connectivity-block overlap with the complete 260,060-row DOCKSTRING, 72-compound
DAVIS, and 645-row PKIS2 supports. Connectivity-only hits are not called exact
chemical matches.

## Reuse boundary

NCBI states that it places no restrictions on use or distribution of molecular
database data, but also warns that submitters may retain third-party rights and
that NCBI cannot transfer those rights. PubChem likewise directs users to inspect
source-specific licensing. Accordingly, the frozen table is an identity/provenance
crosswalk with PubChem-derived structures, carries source attribution and policy
links, and contains no Anastassiadis assay measurements. It is not labelled CC0.

Primary policy pages:

* {PUBCHEM_DOWNLOAD_TERMS}
* {NCBI_MOLECULAR_DATA_POLICY}
"""


def run_audit(
    *,
    identity_path: Path,
    provenance_path: Path,
    output_dir: Path,
    dockstring_identity_path: Path,
    davis_identity_path: Path,
    pkis2_path: Path,
) -> dict[str, Any]:
    identity = validate_frozen_identity(pd.read_csv(identity_path))
    assignments, dependence = chemical_group_assignments(identity)
    assignment_values = assignments[
        [
            "workbook_compound_index_1based",
            "butina_r2_2048_tanimoto50_cluster",
            "butina_cluster_size",
            "murcko_scaffold",
            "murcko_scaffold_size",
        ]
    ]
    assignments_all = identity.merge(
        assignment_values,
        on="workbook_compound_index_1based",
        how="left",
        validate="one_to_one",
    )
    unresolved_label = assignments_all.workbook_compound_index_1based.map(
        lambda value: f"U{int(value):03d}"
    )
    assignments_all["cluster_multiplier_group"] = assignments_all[
        "butina_r2_2048_tanimoto50_cluster"
    ].fillna(unresolved_label)
    assignments_all["cluster_multiplier_group_source"] = np.where(
        assignments_all.structure_usable,
        "resolved_structure_butina",
        "unresolved_identity_specific_singleton",
    )
    dependence["unresolved_identity_singleton_groups"] = int(
        (~assignments_all.structure_usable).sum()
    )
    dependence["total_cluster_multiplier_groups"] = int(
        assignments_all.cluster_multiplier_group.nunique()
    )
    assignment_columns = [
        "workbook_compound_index_1based",
        "workbook_compound_name",
        "workbook_cas",
        "complete_fixed21",
        "pubchem_cid",
        "pubchem_smiles",
        "pubchem_inchikey",
        "rdkit_standard_inchikey",
        "rdkit_connectivity",
        "candidate_connectivity_blocks",
        "candidate_only_connectivity_blocks",
        "de_leakage_connectivity_blocks",
        "structure_fragment_count",
        "butina_r2_2048_tanimoto50_cluster",
        "butina_cluster_size",
        "murcko_scaffold",
        "murcko_scaffold_size",
        "cluster_multiplier_group",
        "cluster_multiplier_group_source",
    ]
    references = load_reference_identities(
        dockstring_identity_path, davis_identity_path, pkis2_path
    )
    overlap_records, overlap_summary, reference_metadata = overlap_audit(
        identity, references
    )
    provenance = json.loads(Path(provenance_path).read_text(encoding="utf-8"))
    resolution = {
        "records": 178,
        "resolved_records": int(identity.pubchem_cid.notna().sum()),
        "usable_structures": int(identity.structure_usable.sum()),
        "ambiguous_records": int(identity.resolution_ambiguity.sum()),
        "unresolved_records": int((~identity.structure_usable).sum()),
        "status_counts": {
            str(key): int(value)
            for key, value in identity.resolution_status.value_counts().sort_index().items()
        },
        "pubchem_rdkit_inchikey_disagreements": int(
            (
                identity.pubchem_cid.notna()
                & ~identity.pubchem_rdkit_inchikey_agree.astype(bool)
            ).sum()
        ),
    }
    workbook = dict(provenance["workbook_contract"])
    summary = {
        "analysis": "public_anastassiadis_identity_audit",
        "access_date": ACCESS_DATE,
        "article_doi": ARTICLE_DOI,
        "workbook": workbook,
        "identity_resolution": resolution,
        "chemical_dependence": dependence,
        "reference_supports": reference_metadata,
        "overlap": overlap_summary.to_dict(orient="records"),
        "cluster_multiplier_feasibility": {
            "feasible": bool(
                dependence["butina_clusters"] >= 20
                and dependence["usable_structures"] >= 100
            ),
            "recommended_unit": "Butina cluster",
            "recommended_scheme": (
                "independent Exp(1) multiplier per resolved-structure Butina "
                "cluster; all members inherit the cluster weight; each unresolved "
                "identity is an explicit singleton group"
            ),
            "limitations": (
                "cluster bootstrap quantifies dependence conditional on resolved "
                "structures and the fixed compound library; it does not make the "
                "178 commercial inhibitors a population sample"
            ),
        },
        "software": {
            "python": sys.version.split()[0],
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "rdkit": rdBase.rdkitVersion,
        },
        "input_checksums": {
            "identity_csv": sha256_file(identity_path),
            "identity_provenance": sha256_file(provenance_path),
            "dockstring_identity": sha256_file(dockstring_identity_path),
            "davis_identity": sha256_file(davis_identity_path),
            "pkis2_workbook": sha256_file(pkis2_path),
        },
        "redistribution_boundary": {
            "contains_hotspot_assay_values": False,
            "contains_identity_labels_from_official_supplement": True,
            "contains_pubchem_compound_properties": True,
            "license_label": "LicenseRef-NCBI-PubChem-molecular-data-policy",
            "not_asserted": "CC0 or transfer of third-party contributor rights",
            "pubchem_download_terms": PUBCHEM_DOWNLOAD_TERMS,
            "ncbi_molecular_data_policy": NCBI_MOLECULAR_DATA_POLICY,
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "identity_resolution_audit.csv", identity)
    write_csv(
        output_dir / "cluster_assignments.csv",
        assignments_all[assignment_columns],
    )
    write_csv(
        output_dir / "complete21_support.csv",
        assignments_all[
            [
                "workbook_compound_index_1based",
                "workbook_compound_name",
                "workbook_cas",
                "complete_fixed21",
                "missing_fixed21_targets",
                "structure_usable",
                "rdkit_standard_inchikey",
                "rdkit_connectivity",
                "candidate_connectivity_blocks",
                "candidate_only_connectivity_blocks",
                "de_leakage_connectivity_blocks",
                "butina_r2_2048_tanimoto50_cluster",
                "murcko_scaffold",
                "cluster_multiplier_group",
                "cluster_multiplier_group_source",
            ]
        ],
    )
    write_csv(output_dir / "cross_panel_overlap_records.csv", overlap_records)
    write_csv(output_dir / "cross_panel_overlap_summary.csv", overlap_summary)
    dependence_rows = [
        {"metric": key, "value": value}
        for key, value in dependence.items()
        if not isinstance(value, (dict, list))
    ]
    write_csv(output_dir / "chemical_dependence_summary.csv", pd.DataFrame(dependence_rows))
    write_json(output_dir / "summary.json", summary)
    (output_dir / "README.md").write_text(make_readme(summary), encoding="utf-8")
    write_json(output_dir / "output_checksums.json", output_checksums(output_dir))
    return summary


def refresh_identity(
    workbook_path: Path,
    identity_path: Path,
    provenance_path: Path,
    *,
    min_interval_seconds: float,
) -> dict[str, Any]:
    compounds, workbook_metadata = load_workbook(workbook_path)
    client = PubChemClient(min_interval_seconds=min_interval_seconds)
    identity, retrieval_metadata = retrieve_identity_table(compounds, client)
    write_csv(identity_path, identity)
    provenance = {
        "analysis": "Anastassiadis 2011 PubChem identity crosswalk",
        "article_doi": ARTICLE_DOI,
        "official_workbook_url": WORKBOOK_URL,
        "official_workbook_sha256": WORKBOOK_SHA256,
        "workbook_contract": workbook_metadata,
        "pubchem": {
            "base_url": PUBCHEM_BASE,
            "access_date": ACCESS_DATE,
            "identifier_queries": "independent PUG REST name lookups for workbook CAS and name",
            "properties": [
                "SMILES",
                "ConnectivitySMILES",
                "InChIKey",
                "IUPACName",
            ],
            "rate_limit_seconds_between_requests": min_interval_seconds,
            **retrieval_metadata,
        },
        "resolution_rule": {
            "primary_identifier": "checksum-valid workbook CAS",
            "unique_cas": (
                "accept if name is absent or contains the same CID; disjoint name "
                "candidates are an unresolved conflict"
            ),
            "multiple_cas": "accept only one-element CAS/name intersection",
            "cas_not_found": "accept only a unique exact PubChem name result",
            "downstream_structure_gate": (
                "one selected non-ambiguous CID and exact agreement between PubChem "
                "and RDKit Standard InChIKey"
            ),
            "conservative_de_leakage": (
                "resolved records use the selected RDKit connectivity block; "
                "unresolved records use the union of connectivity blocks from every "
                "CAS/name candidate PubChem CID"
            ),
        },
        "redistribution": {
            "contains_hotspot_assay_values": False,
            "pubchem_download_terms": PUBCHEM_DOWNLOAD_TERMS,
            "ncbi_molecular_data_policy": NCBI_MOLECULAR_DATA_POLICY,
            "notice": (
                "NCBI places no restrictions on molecular data but cannot transfer "
                "third-party rights; source-specific restrictions remain the user's "
                "responsibility. This identity crosswalk is not labelled CC0."
            ),
        },
        "software": {
            "python": sys.version.split()[0],
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "rdkit": rdBase.rdkitVersion,
        },
    }
    write_json(provenance_path, provenance)
    return provenance


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workbook", type=Path, default=DEFAULT_WORKBOOK)
    parser.add_argument("--identity", type=Path, default=DEFAULT_IDENTITY)
    parser.add_argument("--provenance", type=Path, default=DEFAULT_PROVENANCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--dockstring-identity", type=Path, default=DEFAULT_DOCKSTRING_IDENTITY
    )
    parser.add_argument("--davis-identity", type=Path, default=DEFAULT_DAVIS_IDENTITY)
    parser.add_argument("--pkis2", type=Path, default=DEFAULT_PKIS2)
    parser.add_argument(
        "--refresh-pubchem",
        action="store_true",
        help="query PubChem and replace the identity-only frozen crosswalk",
    )
    parser.add_argument(
        "--min-request-interval",
        type=float,
        default=0.22,
        help="minimum seconds between serial PubChem requests (default <5 requests/s)",
    )
    parser.add_argument("--quiet", action="store_true", help="suppress JSON stdout")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.refresh_pubchem:
        refresh_identity(
            args.workbook,
            args.identity,
            args.provenance,
            min_interval_seconds=args.min_request_interval,
        )
    summary = run_audit(
        identity_path=args.identity,
        provenance_path=args.provenance,
        output_dir=args.output,
        dockstring_identity_path=args.dockstring_identity,
        davis_identity_path=args.davis_identity,
        pkis2_path=args.pkis2,
    )
    if not args.quiet:
        print(json.dumps(json_ready(summary), indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
