#!/usr/bin/env python3
"""Download and checksum the public Sawada et al. PBAS proteome-wide score archive."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import urllib.request
from pathlib import Path


URL = "https://yamanishi.cs.i.nagoya-u.ac.jp/pbas/2024-04-02/pbas_profiles.tsv.gz"
SHA256 = "774aa84179c5707e40ce0d054722ff85a2ab5a1a2de6b561b8b55139c8626dea"
BYTES = 415_234_459


def checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path, default=Path("downloads/pbas_profiles.tsv.gz")
    )
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        observed_size = args.output.stat().st_size
        observed_sha = checksum(args.output)
        if observed_size == BYTES and observed_sha == SHA256:
            print(f"Already downloaded and verified: {args.output}")
            return
        raise SystemExit(
            f"refusing to overwrite existing unverified file: {args.output} "
            f"(bytes={observed_size}, sha256={observed_sha})"
        )
    partial = args.output.with_suffix(args.output.suffix + ".partial")
    request = urllib.request.Request(URL, headers={"User-Agent": "jcheminf-pbas-audit/1"})
    with urllib.request.urlopen(request) as source, partial.open("wb") as sink:
        shutil.copyfileobj(source, sink, length=8 * 1024 * 1024)
    observed_size = partial.stat().st_size
    observed_sha = checksum(partial)
    if observed_size != BYTES or observed_sha != SHA256:
        raise SystemExit(
            f"download verification failed: bytes={observed_size}, sha256={observed_sha}"
        )
    partial.replace(args.output)
    print(f"Downloaded and verified {args.output} ({observed_size:,} bytes; {observed_sha})")


if __name__ == "__main__":
    main()
