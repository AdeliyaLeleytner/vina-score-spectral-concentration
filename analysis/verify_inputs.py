#!/usr/bin/env python3
"""Verify every frozen input used by the manuscript against data_manifest.csv."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from build_result_bundle_checksums import MANIFEST_NAME, REQUIRED_BUNDLES


PACKAGE = Path(__file__).resolve().parents[1]
MANIFESTS = (
    PACKAGE / "data_manifest.csv",
    PACKAGE / "results" / "manuscript_source_manifest.csv",
)
BUNDLE_MANIFESTS = tuple(
    PACKAGE / "results" / bundle / MANIFEST_NAME for bundle in REQUIRED_BUNDLES
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_manifest(manifest: Path) -> tuple[int, list[str]]:
    failures: list[str] = []
    with manifest.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        path = PACKAGE / row["path"]
        if not path.exists():
            failures.append(f"{manifest.name}: missing: {row['path']}")
            continue
        size = path.stat().st_size
        if size != int(row["bytes"]):
            failures.append(
                f"{manifest.name}: size mismatch: {row['path']} "
                f"({size} != {row['bytes']})"
            )
            continue
        observed = sha256(path)
        if observed != row["sha256"]:
            failures.append(f"{manifest.name}: checksum mismatch: {row['path']}")
    return len(rows), failures


def verify_bundle_manifest(manifest: Path) -> tuple[int, list[str]]:
    failures: list[str] = []
    if not manifest.is_file():
        return 0, [f"missing result-bundle manifest: {manifest.relative_to(PACKAGE)}"]
    try:
        payload = json.loads(manifest.read_text())
    except (json.JSONDecodeError, OSError) as error:
        return 0, [f"invalid result-bundle manifest {manifest.name}: {error}"]
    if payload.get("algorithm") != "sha256" or not isinstance(
        payload.get("files"), dict
    ):
        return 0, [f"invalid result-bundle schema: {manifest.relative_to(PACKAGE)}"]
    entries = payload["files"]
    expected_names = {
        path.name
        for path in manifest.parent.iterdir()
        if path.is_file() and path.name != MANIFEST_NAME
    }
    if set(entries) != expected_names:
        missing = sorted(expected_names - set(entries))
        extra = sorted(set(entries) - expected_names)
        failures.append(
            f"bundle file-set mismatch {manifest.parent.name}: "
            f"missing_from_manifest={missing}, absent_from_bundle={extra}"
        )
    for name, entry in entries.items():
        if Path(name).name != name or name in {"", ".", ".."}:
            failures.append(f"unsafe bundle filename {manifest.parent.name}/{name}")
            continue
        path = manifest.parent / name
        if not path.is_file() or path.is_symlink():
            failures.append(f"missing/non-regular bundle file: {path.relative_to(PACKAGE)}")
            continue
        if isinstance(entry, str):  # support older package manifests
            expected_hash = entry
            expected_bytes = None
        elif isinstance(entry, dict):
            expected_hash = entry.get("sha256")
            expected_bytes = entry.get("bytes")
        else:
            failures.append(f"invalid bundle entry: {manifest.parent.name}/{name}")
            continue
        if expected_bytes is not None and path.stat().st_size != int(expected_bytes):
            failures.append(f"bundle size mismatch: {path.relative_to(PACKAGE)}")
            continue
        if sha256(path) != expected_hash:
            failures.append(f"bundle checksum mismatch: {path.relative_to(PACKAGE)}")
    return len(entries), failures


def main() -> None:
    failures: list[str] = []
    verified: list[str] = []
    for manifest in MANIFESTS:
        count, manifest_failures = verify_manifest(manifest)
        failures.extend(manifest_failures)
        verified.append(f"{count} entries in {manifest.relative_to(PACKAGE)}")
    for manifest in BUNDLE_MANIFESTS:
        count, manifest_failures = verify_bundle_manifest(manifest)
        failures.extend(manifest_failures)
        verified.append(f"{count} files in {manifest.parent.relative_to(PACKAGE)}")
    if failures:
        raise SystemExit("\n".join(failures))
    print("Verified " + " and ".join(verified))


if __name__ == "__main__":
    main()
