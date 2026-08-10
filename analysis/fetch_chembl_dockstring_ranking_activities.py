#!/usr/bin/env python3
"""Fetch a potency-floor-free ChEMBL activity snapshot for the DOCKSTRING ranking benchmark.

The frozen snapshot used by the chemistry-graph audit
(``data/frozen/chembl_sea_activities_2026-08-03.csv.gz``) is left-truncated at
``pchembl_value >= 6`` and restricted to ``assay_type=B``.  A within-ligand
target-ranking benchmark built on that snapshot would estimate ordering among
high-potency binding cells only, which is a different estimand from the
published sparse Docking-44 x ChEMBL benchmark
(``expanded_target_preference_benchmark`` in ``analysis/build_evidence.py``).

This program therefore re-pulls the same 31 human SINGLE PROTEIN targets under
the *published benchmark's* activity contract:

* every record with a non-null ``pchembl_value`` (no potency floor);
* exact standard relation ``=``;
* every assay type and every organism annotation retained in the file, so that
  the assay-restricted sensitivity of the published benchmark can be
  reproduced downstream by filtering rather than by re-querying.

Endpoint-type restriction (Ki/Kd/IC50/EC50), chemical standardisation, and all
matching against DOCKSTRING happen downstream in
``analysis/dockstring_chembl_ranking_benchmark.py``.  This program only
downloads and validates records.

The target contract (gene symbol -> UniProt accession -> ChEMBL target id) is
imported unchanged from ``analysis/fetch_chembl_sea_graph_data.py`` so the two
snapshots address exactly the same protein set.
"""

from __future__ import annotations

import argparse
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:  # Support direct CLI execution and package-style imports.
    from . import fetch_chembl_sea_graph_data as sea_fetch
except ImportError:  # pragma: no cover - direct CLI execution.
    import fetch_chembl_sea_graph_data as sea_fetch  # type: ignore


PACKAGE = Path(__file__).resolve().parents[1]
API_URL = sea_fetch.API_URL
PAGE_LIMIT = sea_fetch.PAGE_LIMIT
TARGETS: dict[str, dict[str, Any]] = sea_fetch.TARGETS

DEFAULT_OUTPUT = (
    PACKAGE / "data/frozen/chembl_dockstring_ranking_activities_2026-08-05.csv.gz"
)
DEFAULT_PROVENANCE = (
    PACKAGE
    / "data/frozen/chembl_dockstring_ranking_activities_2026-08-05.provenance.json"
)

# Superset of the published Docking-44 pull's analysis columns.  ``assay_type``
# and ``target_organism`` are retained so the published assay-restricted
# sensitivity is reproducible by filtering this file.
FIELDS = (
    "activity_id",
    "target_chembl_id",
    "molecule_chembl_id",
    "canonical_smiles",
    "pchembl_value",
    "standard_type",
    "standard_relation",
    "standard_value",
    "standard_units",
    "assay_chembl_id",
    "assay_type",
    "document_chembl_id",
    "target_organism",
    "data_validity_comment",
    "potential_duplicate",
)


def query_parameters(
    offset: int, target_chembl_id: str | None = None
) -> dict[str, str | int]:
    """Floor-free, assay-type-free activity query for one ChEMBL target."""
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
        "pchembl_value__isnull": "false",
        "standard_relation": "=",
        "limit": PAGE_LIMIT,
        "offset": offset,
        "only": ",".join(FIELDS),
    }


def page_url(offset: int, target_chembl_id: str) -> str:
    return f"{API_URL}?{urllib.parse.urlencode(query_parameters(offset, target_chembl_id))}"


def fetch_all(workers: int, cache_dir: Path) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Page every target in parallel, asserting server-side count stability."""
    target_ids = sorted({str(record["chembl"]) for record in TARGETS.values()})
    first_pages: dict[str, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                sea_fetch.fetch_json,
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
                sea_fetch.fetch_json,
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
                print(
                    f"downloaded {completed_count}/{len(tasks)} continuation pages",
                    flush=True,
                )
    for key in sorted(pages):
        records.extend(pages[key])
    return records, totals


def validate(
    records: list[dict[str, Any]], totals: dict[str, int]
) -> list[dict[str, Any]]:
    """Self-check the snapshot against the server counts and the query contract."""
    expected_total = sum(totals.values())
    if len(records) != expected_total:
        raise ValueError(f"expected {expected_total} records, received {len(records)}")
    activity_ids = [int(record["activity_id"]) for record in records]
    if len(set(activity_ids)) != len(activity_ids):
        raise ValueError("duplicate activity_id returned across pages")
    allowed_targets = {str(record["chembl"]) for record in TARGETS.values()}
    per_target: dict[str, int] = {target: 0 for target in allowed_targets}
    for record in records:
        missing = [field for field in FIELDS if field not in record]
        if missing:
            raise ValueError(
                f"activity {record.get('activity_id')} lacks fields {missing}"
            )
        target = str(record["target_chembl_id"])
        if target not in allowed_targets:
            raise ValueError(f"unexpected target {target}")
        if record["standard_relation"] != "=":
            raise ValueError("non-exact standard relation escaped the server filter")
        if record["pchembl_value"] in (None, ""):
            raise ValueError("null pChEMBL value escaped the server filter")
        per_target[target] += 1
    mismatched = {
        target: (per_target[target], totals[target])
        for target in allowed_targets
        if per_target[target] != totals[target]
    }
    if mismatched:
        raise ValueError(f"per-target record counts disagree with the API: {mismatched}")
    return sorted(records, key=lambda record: int(record["activity_id"]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--provenance", type=Path, default=DEFAULT_PROVENANCE)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("/tmp/chembl_dockstring_ranking_page_cache"),
    )
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if not 1 <= args.workers <= 12:
        raise ValueError("--workers must be between 1 and 12")

    records, totals = fetch_all(args.workers, args.cache_dir)
    records = validate(records, totals)
    _write_csv_gz(args.output, records)
    provenance = {
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "api_url": API_URL,
        "query": query_parameters(0),
        "target_contract": TARGETS,
        "record_count": len(records),
        "records_per_target": {target: int(count) for target, count in sorted(totals.items())},
        "activity_id_min": min(int(record["activity_id"]) for record in records),
        "activity_id_max": max(int(record["activity_id"]) for record in records),
        "output_sha256": sea_fetch.sha256_file(args.output),
        "differs_from_chembl_sea_activities_2026-08-03": [
            "no pchembl_value floor (the SEA snapshot is truncated at pchembl_value >= 6)",
            "no assay_type restriction (the SEA snapshot keeps assay_type=B only)",
            "no data_validity_comment or potential_duplicate server-side exclusion",
            "assay_type and target_organism retained as columns for downstream sensitivities",
        ],
        "notes": [
            "Targets are the same 31 human SINGLE PROTEIN mappings as the SEA snapshot.",
            "This activity contract matches the published Docking-44 x ChEMBL pull: "
            "non-null pChEMBL, exact standard relation, no potency floor.",
            "Endpoint-type restriction and chemical standardisation are downstream operations.",
        ],
    }
    sea_fetch.write_json(args.provenance, provenance)
    print(
        f"records={len(records)} sha256={provenance['output_sha256']}",
        flush=True,
    )


def _write_csv_gz(path: Path, records: list[dict[str, Any]]) -> None:
    """Deterministic gzip CSV writer using this module's wider field set."""
    import csv
    import gzip
    import io
    import os
    import tempfile

    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(handle)
    temporary = Path(temporary_name)
    try:
        with temporary.open("wb") as raw:
            with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as zipped:
                with io.TextIOWrapper(zipped, encoding="utf-8", newline="") as text:
                    writer = csv.DictWriter(
                        text, fieldnames=FIELDS, lineterminator="\n"
                    )
                    writer.writeheader()
                    writer.writerows(
                        {field: record.get(field) for field in FIELDS}
                        for record in records
                    )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
