#!/usr/bin/env python3
"""Fetch the official Anastassiadis Supplementary Table 3 with a SHA gate."""

from __future__ import annotations

import argparse
import hashlib
import tempfile
import urllib.request
from pathlib import Path

from public_anastassiadis_panel_validation import SOURCE_SHA256, SOURCE_URL


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def fetch(output: Path) -> Path:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=output.name + ".", suffix=".tmp", dir=output.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        with urllib.request.urlopen(SOURCE_URL, timeout=120) as response:  # noqa: S310
            while block := response.read(1 << 20):
                handle.write(block)
    observed = sha256_file(temporary)
    if observed != SOURCE_SHA256:
        temporary.unlink(missing_ok=True)
        raise ValueError(
            f"download checksum mismatch: expected {SOURCE_SHA256}, observed {observed}"
        )
    temporary.replace(output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    path = fetch(args.output)
    print(f"verified {SOURCE_SHA256}  {path}")


if __name__ == "__main__":
    main()
