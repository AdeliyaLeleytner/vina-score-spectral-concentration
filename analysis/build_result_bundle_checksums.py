#!/usr/bin/env python3
"""Build deterministic SHA-256 manifests for manuscript-facing result bundles.

The top-level manuscript source manifest authenticates each bundle manifest;
each bundle manifest in turn authenticates every regular file in its directory.
This two-level contract avoids an unwieldy top-level manifest while ensuring
that supporting CSV files cannot silently drift behind a frozen summary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


PACKAGE = Path(__file__).resolve().parents[1]
RESULTS = PACKAGE / "results"
MANIFEST_NAME = "output_checksums.json"

REQUIRED_BUNDLES = (
    "chemical_context_geometry",
    "descriptor_component_geometry",
    "descriptor_rank_matched_controls",
    "descriptor_correlation_reduction_uncertainty",
    "target_blind_descriptor_control",
    "fixed20_estimand_decomposition",
    "dockstring_chembl_ranking",
    "nonvina_scorer_transport",
    "equivalence_margin_sensitivity",
    "davis_fixed20_censoring_sensitivity",
    "strict_klifs_group_qap",
    "dockstring_vina_terms",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bundle_files(directory: Path) -> list[Path]:
    directory = directory.resolve()
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    files: list[Path] = []
    for path in sorted(directory.iterdir(), key=lambda value: value.name):
        if path.name == MANIFEST_NAME:
            continue
        if path.is_symlink():
            raise ValueError(f"result-bundle symlink is not allowed: {path}")
        if not path.is_file():
            raise ValueError(f"result bundles must be flat regular files: {path}")
        files.append(path)
    if not files:
        raise ValueError(f"empty result bundle: {directory}")
    return files


def build_bundle_manifest(directory: Path) -> dict[str, Any]:
    directory = directory.resolve()
    files = bundle_files(directory)
    return {
        "schema_version": "1.0.0",
        "algorithm": "sha256",
        "scope": "all regular files in this result directory except this manifest",
        "files": {
            path.name: {
                "bytes": int(path.stat().st_size),
                "sha256": sha256_file(path),
            }
            for path in files
        },
    }


def write_bundle_manifest(directory: Path) -> Path:
    directory = directory.resolve()
    payload = build_bundle_manifest(directory)
    destination = directory / MANIFEST_NAME
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, destination)
    return destination


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bundle",
        action="append",
        choices=REQUIRED_BUNDLES,
        help="result-directory name; repeat as needed (default: every required bundle)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bundles = tuple(args.bundle) if args.bundle else REQUIRED_BUNDLES
    for bundle in bundles:
        path = write_bundle_manifest(RESULTS / bundle)
        print(path.relative_to(PACKAGE))


if __name__ == "__main__":
    main()
