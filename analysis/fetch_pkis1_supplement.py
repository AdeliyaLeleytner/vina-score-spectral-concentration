#!/usr/bin/env python3
"""Fetch and checksum-validate the official PKIS1 supplementary archive.

The publisher archive is openly downloadable but is not redistributed by this package.
This helper writes only to the explicit destination chosen by the user and rejects any
payload whose SHA-256 differs from the source file used in the analyses.
"""

from __future__ import annotations

import argparse
import hashlib
import urllib.request
from pathlib import Path


URL = (
    "https://media.springernature.com/original/springer-static/esm/"
    "art%3A10.1038%2Fnbt.3374/MediaObjects/41587_2016_BFnbt3374_MOESM5_ESM.zip"
)
SHA256 = "1ffbe7fd0b4fc1ef72f2434a2b247d15c0f2b4d3c34668f112f4b4c9e4ead7dc"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="destination ZIP path outside git")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".part")
    urllib.request.urlretrieve(URL, temporary)
    observed = sha256_file(temporary)
    if observed != SHA256:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(
            f"PKIS1 SHA256 mismatch: expected {SHA256}, observed {observed}"
        )
    temporary.replace(args.output)
    print(f"Wrote {args.output} ({observed})")


if __name__ == "__main__":
    main()
