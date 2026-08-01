#!/usr/bin/env python3
"""Verify every frozen input used by the manuscript against data_manifest.csv."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path


PACKAGE = Path(__file__).resolve().parents[1]
MANIFEST = PACKAGE / "data_manifest.csv"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    failures: list[str] = []
    with MANIFEST.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        path = PACKAGE / row["path"]
        if not path.exists():
            failures.append(f"missing: {row['path']}")
            continue
        size = path.stat().st_size
        if size != int(row["bytes"]):
            failures.append(f"size mismatch: {row['path']} ({size} != {row['bytes']})")
            continue
        observed = sha256(path)
        if observed != row["sha256"]:
            failures.append(f"checksum mismatch: {row['path']}")
    if failures:
        raise SystemExit("\n".join(failures))
    print(f"Verified {len(rows)} frozen inputs against {MANIFEST.name}")


if __name__ == "__main__":
    main()
