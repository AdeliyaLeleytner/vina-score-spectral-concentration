#!/usr/bin/env python3
"""Fetch and checksum-validate the PDSP Ki database export.

The source is anonymously downloadable, but no redistribution licence was
located. This helper therefore writes only to the explicit destination chosen
by the user and rejects bytes that differ from the export used in this study.
Do not commit the downloaded table to this repository.
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import tempfile
from pathlib import Path
from typing import Callable


URL = "https://pdsp.unc.edu/databases/kiDownload/download.php"
SHA256 = "45c9a18ac30f1fad350d1dde186bc1f226c5a75d474ca50f50713852a5637ac6"
EXPECTED_BYTES = 23_845_904
EXPECTED_HEADER = (
    b"Number,Name,Unigene,Ligand ID, Ligand Name,SMILES,CAS,NSC,Hotligand,"
    b"species,source,ki Note,ki Val,Reference,Link"
)


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
        raise ValueError(f"PDSP SHA256 mismatch: expected {SHA256}, observed {observed}")
    if path.stat().st_size != EXPECTED_BYTES:
        raise ValueError(
            f"PDSP byte-count mismatch: expected {EXPECTED_BYTES}, "
            f"observed {path.stat().st_size}"
        )
    with path.open("rb") as handle:
        header = handle.readline().rstrip(b"\r\n")
    if header != EXPECTED_HEADER:
        raise ValueError("PDSP header differs from the frozen source contract")


def download(url: str, destination: Path) -> None:
    """Use the system TLS store through curl; never fall back to insecure HTTP."""
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
    parser.add_argument("output", type=Path, help="destination CSV path outside git")
    parser.add_argument("--force", action="store_true", help="replace an existing destination")
    args = parser.parse_args()
    path = fetch(args.output, force=args.force)
    print(f"verified {SHA256}  {path}")


if __name__ == "__main__":
    main()
