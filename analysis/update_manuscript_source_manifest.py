#!/usr/bin/env python3
"""Refresh byte counts/hashes in the manuscript source manifest deterministically."""

from __future__ import annotations

import csv
import hashlib
import os
from pathlib import Path

from build_result_bundle_checksums import MANIFEST_NAME, REQUIRED_BUNDLES


PACKAGE = Path(__file__).resolve().parents[1]
MANIFEST = PACKAGE / "results" / "manuscript_source_manifest.csv"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def required_bundle_manifests() -> list[str]:
    return [
        f"results/{bundle}/{MANIFEST_NAME}" for bundle in REQUIRED_BUNDLES
    ]


def refresh_manifest(path: Path = MANIFEST) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        original = list(csv.DictReader(handle))
    ordered_paths: list[str] = []
    seen: set[str] = set()
    for row in original:
        relative = row["path"]
        if relative in seen:
            raise ValueError(f"duplicate manuscript source path: {relative}")
        seen.add(relative)
        ordered_paths.append(relative)
    for relative in required_bundle_manifests():
        if relative not in seen:
            seen.add(relative)
            ordered_paths.append(relative)

    rows: list[dict[str, str]] = []
    for relative in ordered_paths:
        candidate = Path(relative)
        if candidate.is_absolute() or ".." in candidate.parts:
            raise ValueError(f"unsafe manuscript source path: {relative}")
        source = PACKAGE / candidate
        if not source.is_file() or source.is_symlink():
            raise FileNotFoundError(source)
        rows.append(
            {
                "path": candidate.as_posix(),
                "bytes": str(source.stat().st_size),
                "sha256": sha256_file(source),
            }
        )

    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("path", "bytes", "sha256"),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)
    return rows


def main() -> None:
    rows = refresh_manifest()
    print(f"refreshed {len(rows)} entries in {MANIFEST.relative_to(PACKAGE)}")


if __name__ == "__main__":
    main()
