#!/usr/bin/env python3
"""Fetch the checksum-frozen UniProt table used by the SPD analysis.

UniProt's response order is not part of the biological input. The fetched rows
are therefore normalized to primary-gene order before the exact snapshot hash
is checked and written.
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import tempfile
import urllib.parse
from pathlib import Path
from typing import Callable


API_URL = "https://rest.uniprot.org/uniprotkb/search"
GENES = (
    "ADORA2A",
    "ADRB1",
    "ADRB2",
    "AR",
    "DRD2",
    "EGFR",
    "ESR1",
    "ESR2",
    "F2",
    "NR3C1",
    "PGR",
    "PTGS2",
)
FIELDS = "accession,gene_primary,sequence"
EXPECTED_HEADER = b"Entry\tGene Names (primary)\tSequence"
EXPECTED_SHA256 = "b0f6213a79286da247fb1bf74c036446fe61e1a3beff074187025fce35015689"
EXPECTED_BYTES = 8_130


def query_url() -> str:
    genes = " OR ".join(f"gene_exact:{gene}" for gene in GENES)
    parameters = (
        ("query", f"({genes}) AND (organism_id:9606) AND (reviewed:true)"),
        ("format", "tsv"),
        ("fields", FIELDS),
        ("size", "500"),
    )
    return f"{API_URL}?{urllib.parse.urlencode(parameters)}"


def normalize(payload: bytes) -> bytes:
    lines = payload.splitlines()
    if not lines or lines[0] != EXPECTED_HEADER:
        raise ValueError("UniProt response has an unexpected TSV header")
    records: list[tuple[str, bytes]] = []
    for line in lines[1:]:
        fields = line.split(b"\t")
        if len(fields) != 3:
            raise ValueError("UniProt response has a malformed row")
        records.append((fields[1].decode("ascii"), line))
    genes = [gene for gene, _ in records]
    if len(records) != len(GENES) or set(genes) != set(GENES) or len(set(genes)) != len(genes):
        raise ValueError("UniProt primary-gene labels do not exactly match the SPD panel")
    return b"\n".join([EXPECTED_HEADER, *(line for _, line in sorted(records))]) + b"\n"


def validate(payload: bytes) -> str:
    digest = hashlib.sha256(payload).hexdigest()
    if digest != EXPECTED_SHA256:
        raise ValueError(
            "UniProt normalized snapshot checksum changed: "
            f"expected {EXPECTED_SHA256}, received {digest}"
        )
    if len(payload) != EXPECTED_BYTES:
        raise ValueError(
            f"UniProt byte-count mismatch: expected {EXPECTED_BYTES}, observed {len(payload)}"
        )
    return digest


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
            "60",
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
        downloader(query_url(), temporary)
        normalized = normalize(temporary.read_bytes())
        validate(normalized)
        temporary.write_bytes(normalized)
        temporary.replace(output)
    except Exception:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="destination TSV path outside git")
    parser.add_argument("--force", action="store_true", help="replace an existing destination")
    parser.add_argument("--print-url", action="store_true", help="print the exact query and exit")
    args = parser.parse_args()
    if args.print_url:
        print(query_url())
        return
    path = fetch(args.output, force=args.force)
    print(f"verified {EXPECTED_SHA256}  {path}")


if __name__ == "__main__":
    main()
