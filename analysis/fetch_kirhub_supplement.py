#!/usr/bin/env python3
"""Fetch and checksum-validate the official KiRHub supplementary workbook.

The associated article is CC BY-NC-ND 4.0, but the official KIRHub portal
separately states that the kinase-inhibition data are the property of Reaction
Biology Corporation and require proper acknowledgement.  This package makes no
claim that the article license independently licenses the workbook.  The helper
writes only to the explicit destination chosen by the user; do not commit the
downloaded workbook to this repository.
"""

from __future__ import annotations

import argparse
import hashlib
import urllib.request
from pathlib import Path


URL = (
    "https://media.springernature.com/original/springer-static/esm/"
    "art%3A10.1038%2Fs41587-026-03090-8/"
    "MediaObjects/41587_2026_3090_MOESM4_ESM.xlsx"
)
SHA256 = "d9eef358396b193834b0c4d48ccd8cadb43697a6458cdcb50415af5a7e4e0b03"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="destination XLSX path outside git")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".part")
    urllib.request.urlretrieve(URL, temporary)
    observed = sha256_file(temporary)
    if observed != SHA256:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(
            f"KiRHub SHA256 mismatch: expected {SHA256}, observed {observed}"
        )
    temporary.replace(args.output)
    print(f"Wrote {args.output} ({observed})")


if __name__ == "__main__":
    main()
