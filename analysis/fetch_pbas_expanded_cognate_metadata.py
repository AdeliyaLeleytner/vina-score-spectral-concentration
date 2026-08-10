#!/usr/bin/env python3
"""Freeze the RCSB metadata used for the expanded PBAS cognate panel.

The PDB identifiers below are not selected by docking outcomes.  They are the
entries among the 92 previously downloaded exact-SIFTS-mapped PDB/AF pairs
that contain at least one locally eligible non-polymer instance according to
``pbas_pilot_input_audit.audit_target``.  This one-time network helper freezes
entry titles and non-polymer entity annotations from the primary RCSB Data API.
The panel-design script subsequently works offline from this snapshot and the
checksummed PDBe updated mmCIF coordinates.
"""

from __future__ import annotations

import argparse
import json
import os
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RCSB_GRAPHQL_ENDPOINT = "https://data.rcsb.org/graphql"
CANDIDATE_PDB_IDS = (
    "1NE7", "20ZX", "2AG4", "3SC0", "4ZZZ", "5FAI", "5IG6", "5WCI",
    "6C9H", "6HLO", "6LZZ", "6OL9", "6P2J", "6P5V", "6T4X", "6THA",
    "6Y9S", "6YG9", "7NR4", "8ROS", "9IAY", "9LRN", "9N09", "9NX6",
    "9PTU", "9R1F",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "results/pbas_expanded_cognate_panel_design/rcsb_metadata_snapshot.json",
    )
    return parser.parse_args()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    query = """
    query ExpandedCognateMetadata($ids: [String!]!) {
      entries(entry_ids: $ids) {
        rcsb_id
        struct { title }
        rcsb_accession_info { initial_release_date }
        exptl { method }
        nonpolymer_entities {
          rcsb_id
          pdbx_entity_nonpoly { comp_id entity_id name }
          rcsb_nonpolymer_entity { formula_weight pdbx_description pdbx_number_of_molecules }
        }
      }
    }
    """
    body = json.dumps(
        {"query": query, "variables": {"ids": list(CANDIDATE_PDB_IDS)}}
    ).encode("utf-8")
    request = urllib.request.Request(
        RCSB_GRAPHQL_ENDPOINT,
        data=body,
        headers={"Content-Type": "application/json", "User-Agent": "pbas-expanded-cognate-design/1.0"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if payload.get("errors"):
        raise RuntimeError(f"RCSB GraphQL errors: {payload['errors']}")
    entries = payload.get("data", {}).get("entries") or []
    by_id = {str(entry["rcsb_id"]).upper(): entry for entry in entries}
    if set(by_id) != set(CANDIDATE_PDB_IDS):
        raise ValueError(
            "RCSB response identifier mismatch: "
            f"missing={sorted(set(CANDIDATE_PDB_IDS) - set(by_id))}, "
            f"extra={sorted(set(by_id) - set(CANDIDATE_PDB_IDS))}"
        )
    normalized_entries: list[dict[str, Any]] = []
    for pdb_id in CANDIDATE_PDB_IDS:
        entry = by_id[pdb_id]
        nonpolymers = sorted(
            entry.get("nonpolymer_entities") or [],
            key=lambda row: (
                str((row.get("pdbx_entity_nonpoly") or {}).get("comp_id", "")),
                str(row.get("rcsb_id", "")),
            ),
        )
        normalized_entries.append({**entry, "nonpolymer_entities": nonpolymers})
    atomic_json(
        args.output,
        {
            "snapshot_role": "score-blind primary-metadata curation input",
            "retrieved_at_utc": datetime.now(UTC).isoformat(),
            "endpoint": RCSB_GRAPHQL_ENDPOINT,
            "candidate_set_definition": (
                "PDB entries among the 92 frozen exact-mapped PDB/AF pairs with at "
                "least one audit_target pilot-eligible non-polymer instance"
            ),
            "selection_used_docking_outcomes": False,
            "requested_pdb_ids": list(CANDIDATE_PDB_IDS),
            "query": query.strip(),
            "entries": normalized_entries,
        },
    )
    print(f"wrote {len(normalized_entries)} RCSB entries to {args.output}")


if __name__ == "__main__":
    main()
