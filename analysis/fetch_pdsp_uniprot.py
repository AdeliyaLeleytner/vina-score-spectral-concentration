#!/usr/bin/env python3
"""Fetch the checksum-frozen UniProt sequence table used by the PDSP audit.

The query is intentionally explicit: reviewed human records for the 27 gene
labels in the original PDSP mapping, sorted by accession, with only the four
fields consumed or retained by the analysis.  Bytes are written only after the
published SHA-256 and table shape have been verified.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


API_URL = "https://rest.uniprot.org/uniprotkb/search"
GENES = (
    "ADRA1A",
    "ADRA2A",
    "ADRB1",
    "ADRB2",
    "CNR1",
    "CNR2",
    "CHRM1",
    "CHRM2",
    "CHRM3",
    "DRD1",
    "DRD2",
    "HRH1",
    "HRH2",
    "HTR1A",
    "HTR1B",
    "HTR2A",
    "HTR2B",
    "HTR3A",
    "OPRD1",
    "OPRK1",
    "OPRM1",
    "SLC6A2",
    "SLC6A3",
    "SLC6A4",
    "ADORA2A",
    "NR3C1",
    "PTGS2",
)
FIELDS = "accession,gene_names,sequence,length"
EXPECTED_SHA256 = "051690a9647c3af3ad71708d935485c73bdbffc07772b0ec8c6f484e9c1b408a"
EXPECTED_HEADER = b"Entry\tGene Names\tSequence\tLength"
EXPECTED_RECORDS = 28


def query_url() -> str:
    genes = " OR ".join(f"gene_exact:{gene}" for gene in GENES)
    query = f"({genes}) AND (organism_id:9606) AND (reviewed:true)"
    parameters = (
        ("query", query),
        ("format", "tsv"),
        ("fields", FIELDS),
        ("size", "500"),
        ("sort", "accession asc"),
    )
    return f"{API_URL}?{urllib.parse.urlencode(parameters)}"


def validate(payload: bytes) -> str:
    """Return the digest after checking exact bytes and minimal TSV structure."""
    digest = hashlib.sha256(payload).hexdigest()
    if digest != EXPECTED_SHA256:
        raise ValueError(
            "UniProt response checksum changed: "
            f"expected {EXPECTED_SHA256}, received {digest}"
        )
    lines = payload.splitlines()
    if not lines or lines[0] != EXPECTED_HEADER:
        raise ValueError("UniProt response has an unexpected TSV header")
    if len(lines) - 1 != EXPECTED_RECORDS:
        raise ValueError(
            f"expected {EXPECTED_RECORDS} UniProt records, received {len(lines) - 1}"
        )
    return digest


def fetch(timeout: float = 60.0) -> bytes:
    request = urllib.request.Request(
        query_url(),
        headers={"User-Agent": "docking-target-geometry-provenance/5.0"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = response.read()
    except urllib.error.URLError as error:
        raise RuntimeError(f"UniProt request failed: {error}") from error
    validate(payload)
    return payload


def write_verified(payload: bytes, output: Path, *, force: bool = False) -> None:
    """Atomically write verified bytes, refusing an accidental overwrite."""
    validate(payload)
    output = Path(output)
    if output.exists() and not force:
        raise FileExistsError(f"output exists; pass --force to replace it: {output}")
    if not output.parent.is_dir():
        raise FileNotFoundError(f"output directory does not exist: {output.parent}")
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=output.parent,
            prefix=f".{output.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(payload)
            temporary_name = handle.name
        os.replace(temporary_name, output)
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="destination TSV")
    parser.add_argument("--force", action="store_true", help="replace an existing destination")
    parser.add_argument("--timeout", type=float, default=60.0, help="request timeout in seconds")
    parser.add_argument("--print-url", action="store_true", help="print the exact URL and exit")
    args = parser.parse_args()
    if not args.print_url and args.output is None:
        parser.error("--output is required unless --print-url is used")
    return args


def main() -> None:
    args = parse_args()
    if args.print_url:
        print(query_url())
        return
    payload = fetch(timeout=args.timeout)
    write_verified(payload, args.output, force=args.force)
    print(
        f"wrote {args.output} ({len(payload):,} bytes; "
        f"SHA-256 {EXPECTED_SHA256})"
    )


if __name__ == "__main__":
    main()
