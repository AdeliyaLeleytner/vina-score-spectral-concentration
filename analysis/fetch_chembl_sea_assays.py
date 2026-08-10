#!/usr/bin/env python3
"""Fetch confidence-9 ChEMBL assay metadata for the SEA-style graph audit."""

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

try:
    from .fetch_chembl_sea_graph_data import TARGETS
except ImportError:  # pragma: no cover
    from fetch_chembl_sea_graph_data import TARGETS  # type: ignore


API_URL = "https://www.ebi.ac.uk/chembl/api/data/assay.json"
PAGE_LIMIT = 1_000
FIELDS = (
    "assay_chembl_id",
    "target_chembl_id",
    "assay_type",
    "confidence_score",
    "relationship_type",
    "assay_organism",
    "document_chembl_id",
)


def page_url(target_id: str, offset: int) -> str:
    query = urllib.parse.urlencode(
        {
            "target_chembl_id": target_id,
            "assay_type": "B",
            "confidence_score": "9",
            "limit": PAGE_LIMIT,
            "offset": offset,
            "only": ",".join(FIELDS),
        }
    )
    return f"{API_URL}?{query}"


def fetch_page(target_id: str, offset: int, cache_dir: Path) -> dict[str, Any]:
    cache = cache_dir / f"{target_id}_{offset:06d}.json"
    if cache.exists():
        return json.loads(cache.read_bytes())
    command = [
        "curl", "-L", "--fail", "--retry", "5", "--retry-all-errors",
        "--connect-timeout", "15", "--max-time", "90", "-sS",
        page_url(target_id, offset),
    ]
    completed = subprocess.run(command, check=True, capture_output=True)
    payload = json.loads(completed.stdout)
    cache.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache.with_suffix(".json.tmp")
    temporary.write_bytes(completed.stdout)
    os.replace(temporary, cache)
    return payload


def fetch_all(workers: int, cache_dir: Path) -> list[dict[str, Any]]:
    target_ids = sorted({str(record["chembl"]) for record in TARGETS.values()})
    first: dict[str, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(fetch_page, target, 0, cache_dir): target
            for target in target_ids
        }
        for future in as_completed(futures):
            first[futures[future]] = future.result()
    totals = {target: int(first[target]["page_meta"]["total_count"]) for target in target_ids}
    records = [record for target in target_ids for record in first[target]["assays"]]
    tasks = [
        (target, offset)
        for target in target_ids
        for offset in range(PAGE_LIMIT, totals[target], PAGE_LIMIT)
    ]
    pages: dict[tuple[str, int], list[dict[str, Any]]] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(fetch_page, target, offset, cache_dir): (target, offset)
            for target, offset in tasks
        }
        for count, future in enumerate(as_completed(futures), start=1):
            target, offset = futures[future]
            page = future.result()
            if int(page["page_meta"]["offset"]) != offset:
                raise ValueError(f"unexpected page offset for {target}")
            if int(page["page_meta"]["total_count"]) != totals[target]:
                raise ValueError(f"assay total changed during pagination for {target}")
            pages[(target, offset)] = list(page["assays"])
            if count % 10 == 0 or count == len(tasks):
                print(f"downloaded {count}/{len(tasks)} continuation pages", flush=True)
    for key in sorted(pages):
        records.extend(pages[key])
    expected = sum(totals.values())
    if len(records) != expected:
        raise ValueError(f"expected {expected} assays, received {len(records)}")
    unique = {str(record["assay_chembl_id"]): record for record in records}
    if len(unique) != len(records):
        raise ValueError("duplicate assay IDs across exact target queries")
    for record in records:
        if int(record["confidence_score"]) != 9 or record["assay_type"] != "B":
            raise ValueError("server-side assay filter was not respected")
    return sorted(records, key=lambda record: str(record["assay_chembl_id"]))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_gzip_csv(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(fd)
    temporary = Path(name)
    try:
        with temporary.open("wb") as raw:
            with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as zipped:
                with io.TextIOWrapper(zipped, encoding="utf-8", newline="") as text:
                    writer = csv.DictWriter(text, fieldnames=FIELDS, lineterminator="\n")
                    writer.writeheader()
                    writer.writerows({field: row.get(field) for field in FIELDS} for row in records)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path("/tmp/chembl_sea_assay_cache"))
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    records = fetch_all(args.workers, args.cache_dir)
    write_gzip_csv(args.output, records)
    provenance = {
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "api_url": API_URL,
        "filter": {"assay_type": "B", "confidence_score": 9},
        "record_count": len(records),
        "output_sha256": sha256(args.output),
    }
    args.provenance.write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"records": len(records), "sha256": provenance["output_sha256"]}))


if __name__ == "__main__":
    main()
