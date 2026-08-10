#!/usr/bin/env python3
"""Fetch a frozen ChEMBL activity snapshot for the SEA-style graph audit.

The target list and filters are deliberately outcome-blind.  This program only
downloads and validates activity records; chemical standardisation and all
comparisons with experimental target geometry are performed by a separate
analysis script.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import os
import subprocess
import tempfile
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


API_URL = "https://www.ebi.ac.uk/chembl/api/data/activity.json"
PAGE_LIMIT = 1_000

# Exact human SINGLE PROTEIN mappings obtained through the ChEMBL target API
# using the listed UniProt accessions on 2026-08-03.
TARGETS: dict[str, dict[str, Any]] = {
    "ABL1": {"uniprot": "P00519", "chembl": "CHEMBL1862", "panels": "kinase20"},
    "AKT1": {"uniprot": "P31749", "chembl": "CHEMBL4282", "panels": "kinase20"},
    "AKT2": {"uniprot": "P31751", "chembl": "CHEMBL2431", "panels": "kinase20"},
    "CDK2": {"uniprot": "P24941", "chembl": "CHEMBL301", "panels": "kinase20"},
    "CSF1R": {"uniprot": "P07333", "chembl": "CHEMBL1844", "panels": "kinase20"},
    "EGFR": {"uniprot": "P00533", "chembl": "CHEMBL203", "panels": "kinase20,spd12"},
    "FGFR1": {"uniprot": "P11362", "chembl": "CHEMBL3650", "panels": "kinase20"},
    "IGF1R": {"uniprot": "P08069", "chembl": "CHEMBL1957", "panels": "kinase20"},
    "JAK2": {"uniprot": "O60674", "chembl": "CHEMBL2971", "panels": "kinase20"},
    "KDR": {"uniprot": "P35968", "chembl": "CHEMBL279", "panels": "kinase20"},
    "KIT": {"uniprot": "P10721", "chembl": "CHEMBL1936", "panels": "kinase20"},
    "LCK": {"uniprot": "P06239", "chembl": "CHEMBL258", "panels": "kinase20"},
    "MAP2K1": {"uniprot": "Q02750", "chembl": "CHEMBL3587", "panels": "kinase20"},
    "MAPK1": {"uniprot": "P28482", "chembl": "CHEMBL4040", "panels": "kinase20"},
    "MAPK14": {"uniprot": "Q16539", "chembl": "CHEMBL260", "panels": "kinase20"},
    "MAPKAPK2": {"uniprot": "P49137", "chembl": "CHEMBL2208", "panels": "kinase20"},
    "MET": {"uniprot": "P08581", "chembl": "CHEMBL3717", "panels": "kinase20"},
    "PLK1": {"uniprot": "P53350", "chembl": "CHEMBL3024", "panels": "kinase20"},
    "ROCK1": {"uniprot": "Q13464", "chembl": "CHEMBL3231", "panels": "kinase20"},
    "SRC": {"uniprot": "P12931", "chembl": "CHEMBL267", "panels": "kinase20"},
    "ADORA2A": {"uniprot": "P29274", "chembl": "CHEMBL251", "panels": "spd12"},
    "ADRB1": {"uniprot": "P08588", "chembl": "CHEMBL213", "panels": "spd12"},
    "ADRB2": {"uniprot": "P07550", "chembl": "CHEMBL210", "panels": "spd12"},
    "AR": {"uniprot": "P10275", "chembl": "CHEMBL1871", "panels": "spd12"},
    "DRD2": {"uniprot": "P14416", "chembl": "CHEMBL217", "panels": "spd12"},
    "ESR1": {"uniprot": "P03372", "chembl": "CHEMBL206", "panels": "spd12"},
    "ESR2": {"uniprot": "Q92731", "chembl": "CHEMBL242", "panels": "spd12"},
    "F2": {"uniprot": "P00734", "chembl": "CHEMBL204", "panels": "spd12"},
    "NR3C1": {"uniprot": "P04150", "chembl": "CHEMBL2034", "panels": "spd12"},
    "PGR": {"uniprot": "P06401", "chembl": "CHEMBL208", "panels": "spd12"},
    "PTGS2": {"uniprot": "P35354", "chembl": "CHEMBL230", "panels": "spd12"},
}

FIELDS = (
    "activity_id",
    "target_chembl_id",
    "molecule_chembl_id",
    "canonical_smiles",
    "pchembl_value",
    "standard_type",
    "standard_relation",
    "assay_chembl_id",
    "document_chembl_id",
    "data_validity_comment",
    "potential_duplicate",
)


def query_parameters(offset: int, target_chembl_id: str | None = None) -> dict[str, str | int]:
    target_filter = (
        {"target_chembl_id": target_chembl_id}
        if target_chembl_id is not None
        else {
            "target_chembl_id__in": ",".join(
                sorted({str(record["chembl"]) for record in TARGETS.values()})
            )
        }
    )
    return {
        **target_filter,
        "pchembl_value__gte": "6",
        "assay_type": "B",
        "standard_relation": "=",
        "data_validity_comment__isnull": "true",
        "potential_duplicate": "0",
        "limit": PAGE_LIMIT,
        "offset": offset,
        "only": ",".join(FIELDS),
    }


def page_url(offset: int, target_chembl_id: str) -> str:
    return f"{API_URL}?{urllib.parse.urlencode(query_parameters(offset, target_chembl_id))}"


def fetch_json(url: str, cache_path: Path) -> dict[str, Any]:
    if cache_path.exists():
        return json.loads(cache_path.read_bytes())
    completed = subprocess.run(
        [
            "curl",
            "-L",
            "--fail",
            "--retry",
            "5",
            "--retry-all-errors",
            "--connect-timeout",
            "15",
            "--max-time",
            "90",
            "-sS",
            url,
        ],
        check=True,
        capture_output=True,
    )
    payload = json.loads(completed.stdout)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_suffix(cache_path.suffix + ".tmp")
    temporary.write_bytes(completed.stdout)
    os.replace(temporary, cache_path)
    return payload


def fetch_all(workers: int, cache_dir: Path) -> tuple[list[dict[str, Any]], int]:
    target_ids = sorted({str(record["chembl"]) for record in TARGETS.values()})
    first_pages: dict[str, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                fetch_json,
                page_url(0, target_id),
                cache_dir / f"{target_id}_000000.json",
            ): target_id
            for target_id in target_ids
        }
        for future in as_completed(futures):
            first_pages[futures[future]] = future.result()

    totals = {
        target_id: int(first_pages[target_id]["page_meta"]["total_count"])
        for target_id in target_ids
    }
    total = sum(totals.values())
    records: list[dict[str, Any]] = []
    for target_id in target_ids:
        records.extend(first_pages[target_id]["activities"])

    tasks = [
        (target_id, offset)
        for target_id in target_ids
        for offset in range(PAGE_LIMIT, totals[target_id], PAGE_LIMIT)
    ]
    pages: dict[tuple[str, int], list[dict[str, Any]]] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                fetch_json,
                page_url(offset, target_id),
                cache_dir / f"{target_id}_{offset:06d}.json",
            ): (target_id, offset)
            for target_id, offset in tasks
        }
        completed_count = 0
        for future in as_completed(futures):
            target_id, offset = futures[future]
            page = future.result()
            if int(page["page_meta"]["offset"]) != offset:
                raise ValueError(f"unexpected offset for {target_id} page {offset}")
            if int(page["page_meta"]["total_count"]) != totals[target_id]:
                raise ValueError(f"ChEMBL total_count changed for {target_id}")
            pages[(target_id, offset)] = list(page["activities"])
            completed_count += 1
            if completed_count % 10 == 0 or completed_count == len(tasks):
                print(f"downloaded {completed_count}/{len(tasks)} continuation pages", flush=True)
    for key in sorted(pages):
        records.extend(pages[key])
    return records, total


def validate(records: list[dict[str, Any]], expected_total: int) -> list[dict[str, Any]]:
    if len(records) != expected_total:
        raise ValueError(f"expected {expected_total} records, received {len(records)}")
    activity_ids = [int(record["activity_id"]) for record in records]
    if len(set(activity_ids)) != len(activity_ids):
        raise ValueError("duplicate activity_id returned across pages")
    allowed_targets = {str(record["chembl"]) for record in TARGETS.values()}
    for record in records:
        missing = [field for field in FIELDS if field not in record]
        if missing:
            raise ValueError(f"activity {record.get('activity_id')} lacks fields {missing}")
        if record["target_chembl_id"] not in allowed_targets:
            raise ValueError(f"unexpected target {record['target_chembl_id']}")
        if record["standard_relation"] != "=":
            raise ValueError("non-exact standard relation escaped the server filter")
        if float(record["pchembl_value"]) < 6.0:
            raise ValueError("sub-threshold pChEMBL value escaped the server filter")
    return sorted(records, key=lambda record: int(record["activity_id"]))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv_gz(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        with temporary.open("wb") as raw:
            with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as zipped:
                with io.TextIOWrapper(zipped, encoding="utf-8", newline="") as text:
                    writer = csv.DictWriter(text, fieldnames=FIELDS, lineterminator="\n")
                    writer.writeheader()
                    writer.writerows({field: record.get(field) for field in FIELDS} for record in records)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path("/tmp/chembl_sea_page_cache"))
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    if not 1 <= args.workers <= 12:
        raise ValueError("--workers must be between 1 and 12")

    records, total = fetch_all(args.workers, args.cache_dir)
    records = validate(records, total)
    write_csv_gz(args.output, records)
    provenance = {
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "api_url": API_URL,
        "query": query_parameters(0),
        "target_contract": TARGETS,
        "record_count": len(records),
        "activity_id_min": min(int(record["activity_id"]) for record in records),
        "activity_id_max": max(int(record["activity_id"]) for record in records),
        "output_sha256": sha256_file(args.output),
        "notes": [
            "Targets were exact human SINGLE PROTEIN mappings from UniProt accessions.",
            "The API snapshot is not an experimental validation endpoint.",
            "Chemical standardisation, endpoint-type restrictions, and validation-panel leakage exclusions are downstream operations.",
        ],
    }
    write_json(args.provenance, provenance)
    print(json.dumps({"records": len(records), "sha256": provenance["output_sha256"]}))


if __name__ == "__main__":
    main()
