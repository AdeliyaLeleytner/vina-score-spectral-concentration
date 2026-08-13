#!/usr/bin/env python3
"""Fetch and checksum-validate the CC BY 4.0 Novartis SPD export."""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import tempfile
from pathlib import Path
from typing import Callable


URL = (
    "https://zenodo.org/records/8103950/files/"
    "final_summarized_activity_data_pub.txt?download=1"
)
SHA256 = "7132723f85e746de2f8387d01dcde6ffff703c92561fda9751cbd6753e900240"
EXPECTED_BYTES = 16_060_609
REQUIRED_HEADER_FIELDS = {
    "inchi_key",
    "assay_group",
    "assay_group_name",
    "summarized prefix",
    "summarized IC50",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def validate(path: Path) -> None:
    path = Path(path)
    observed = sha256_file(path)
    if observed != SHA256:
        raise ValueError(f"SPD SHA256 mismatch: expected {SHA256}, observed {observed}")
    if path.stat().st_size != EXPECTED_BYTES:
        raise ValueError(
            f"SPD byte-count mismatch: expected {EXPECTED_BYTES}, observed {path.stat().st_size}"
        )
    with path.open("r", encoding="utf-8") as handle:
        fields = set(handle.readline().rstrip("\r\n").split("\t"))
    missing = REQUIRED_HEADER_FIELDS - fields
    if missing:
        raise ValueError(f"SPD export is missing required columns: {sorted(missing)}")


def download(url: str, destination: Path) -> None:
    subprocess.run(
        [
            "curl",
            "--proto",
            "=https",
            "--tlsv1.2",
            "--fail",
            "--location",
            "--silent",
            "--show-error",
            "--max-time",
            "120",
            "--output",
            str(destination),
            url,
        ],
        check=True,
    )


def fetch(
    output: Path,
    *,
    force: bool = False,
    downloader: Callable[[str, Path], None] = download,
) -> Path:
    output = Path(output)
    if output.exists() and not force:
        raise FileExistsError(f"refusing to replace {output}; pass --force to overwrite")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=output.name + ".", suffix=".part", dir=output.parent, delete=False
        ) as handle:
            temporary = Path(handle.name)
        downloader(URL, temporary)
        validate(temporary)
        temporary.replace(output)
    except Exception:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="destination TXT path outside git")
    parser.add_argument("--force", action="store_true", help="replace an existing destination")
    args = parser.parse_args()
    path = fetch(args.output, force=args.force)
    print(f"verified {SHA256}  {path}")


if __name__ == "__main__":
    main()
