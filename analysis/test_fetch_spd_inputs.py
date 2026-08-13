from __future__ import annotations

import hashlib

import pytest

from analysis import fetch_spd_source as spd
from analysis import fetch_spd_uniprot as uniprot


def test_spd_fetch_is_atomic_and_checksum_gated(tmp_path, monkeypatch) -> None:
    content = ("\t".join(sorted(spd.REQUIRED_HEADER_FIELDS)) + "\n1\n").encode()
    monkeypatch.setattr(spd, "SHA256", hashlib.sha256(content).hexdigest())
    monkeypatch.setattr(spd, "EXPECTED_BYTES", len(content))

    def downloader(url: str, destination) -> None:
        assert url == spd.URL
        destination.write_bytes(content)

    output = tmp_path / "spd.txt"
    assert spd.fetch(output, downloader=downloader) == output
    with pytest.raises(FileExistsError, match="--force"):
        spd.fetch(output, downloader=downloader)
    spd.fetch(output, force=True, downloader=downloader)
    assert output.read_bytes() == content


def test_spd_uniprot_query_and_normalization_are_deterministic() -> None:
    url = urllib_parse(uniprot.query_url())
    assert f"{url.scheme}://{url.netloc}{url.path}" == uniprot.API_URL
    parameters = __import__("urllib.parse", fromlist=["parse_qs"]).parse_qs(url.query)
    assert parameters["fields"] == [uniprot.FIELDS]
    assert "sort" not in parameters

    records = [
        f"P{index:05d}\t{gene}\tAAAA".encode()
        for index, gene in enumerate(reversed(uniprot.GENES))
    ]
    payload = b"\n".join([uniprot.EXPECTED_HEADER, *records]) + b"\n"
    normalized = uniprot.normalize(payload)
    genes = [line.split(b"\t")[1].decode() for line in normalized.splitlines()[1:]]
    assert genes == sorted(uniprot.GENES)


def urllib_parse(url: str):
    import urllib.parse

    return urllib.parse.urlsplit(url)


def test_spd_uniprot_fetch_refuses_changed_snapshot(tmp_path, monkeypatch) -> None:
    records = [
        f"P{index:05d}\t{gene}\tAAAA".encode()
        for index, gene in enumerate(uniprot.GENES)
    ]
    raw = b"\n".join([uniprot.EXPECTED_HEADER, *reversed(records)]) + b"\n"
    normalized = uniprot.normalize(raw)
    monkeypatch.setattr(uniprot, "EXPECTED_SHA256", hashlib.sha256(normalized).hexdigest())
    monkeypatch.setattr(uniprot, "EXPECTED_BYTES", len(normalized))

    def downloader(_url: str, destination) -> None:
        destination.write_bytes(raw)

    output = tmp_path / "spd_uniprot.tsv"
    uniprot.fetch(output, downloader=downloader)
    assert output.read_bytes() == normalized

    monkeypatch.setattr(uniprot, "EXPECTED_SHA256", "0" * 64)
    with pytest.raises(ValueError, match="checksum changed"):
        uniprot.fetch(output, force=True, downloader=downloader)
