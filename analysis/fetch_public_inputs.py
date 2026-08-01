#!/usr/bin/env python3
"""Optionally re-fetch a public upstream file for independent source comparison.

The release already contains the checksum-verified compressed DOCKSTRING table used by the
analysis. This utility downloads the original v1 table to a separate directory and never
silently replaces the frozen release artifact.
"""

from __future__ import annotations

import argparse
import json
import shutil
import urllib.request
from pathlib import Path


FIGSHARE_API = "https://api.figshare.com/v2/articles/16511577/versions/1"


def fetch_dockstring(destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(FIGSHARE_API) as response:
        metadata = json.load(response)
    files = metadata.get("files", [])
    matches = [item for item in files if item["name"] == "dockstring-dataset.tsv"]
    if len(matches) != 1:
        available = ", ".join(item["name"] for item in files)
        raise SystemExit(f"Expected dockstring-dataset.tsv; Figshare lists: {available}")
    output = destination / matches[0]["name"]
    with urllib.request.urlopen(matches[0]["download_url"]) as source, output.open("wb") as sink:
        shutil.copyfileobj(source, sink)
    print(f"Downloaded {output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dockstring", action="store_true")
    parser.add_argument("--destination", type=Path, default=Path("downloads"))
    args = parser.parse_args()
    if not args.dockstring:
        parser.error("select --dockstring")
    fetch_dockstring(args.destination)


if __name__ == "__main__":
    main()
