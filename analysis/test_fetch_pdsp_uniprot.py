from __future__ import annotations

import hashlib
import urllib.parse

import pytest

from analysis import fetch_pdsp_uniprot as fetcher


def test_query_contract_is_explicit_and_deterministic() -> None:
    parsed = urllib.parse.urlsplit(fetcher.query_url())
    parameters = urllib.parse.parse_qs(parsed.query)
    assert f"{parsed.scheme}://{parsed.netloc}{parsed.path}" == fetcher.API_URL
    assert parameters["format"] == ["tsv"]
    assert parameters["fields"] == [fetcher.FIELDS]
    assert parameters["size"] == ["500"]
    assert parameters["sort"] == ["accession asc"]
    query = parameters["query"][0]
    assert "organism_id:9606" in query and "reviewed:true" in query
    gene_clause = query.split("(", 1)[1].split(")", 1)[0]
    genes = [term.removeprefix("gene_exact:") for term in gene_clause.split(" OR ")]
    assert genes == list(fetcher.GENES)


def test_validation_fails_closed_on_changed_bytes() -> None:
    payload = b"Entry\tGene Names\tSequence\tLength\nP00000\tTEST\tAAAA\t4\n"
    assert hashlib.sha256(payload).hexdigest() != fetcher.EXPECTED_SHA256
    with pytest.raises(ValueError, match="checksum changed"):
        fetcher.validate(payload)


def test_atomic_writer_refuses_to_replace_without_force(tmp_path, monkeypatch) -> None:
    rows = [fetcher.EXPECTED_HEADER]
    rows.extend(
        f"P{index:05d}\tGENE{index}\tAAAA\t4".encode()
        for index in range(fetcher.EXPECTED_RECORDS)
    )
    payload = b"\n".join(rows) + b"\n"
    monkeypatch.setattr(fetcher, "EXPECTED_SHA256", hashlib.sha256(payload).hexdigest())
    output = tmp_path / "pdsp_uniprot.tsv"
    fetcher.write_verified(payload, output)
    assert output.read_bytes() == payload
    with pytest.raises(FileExistsError, match="--force"):
        fetcher.write_verified(payload, output)
    fetcher.write_verified(payload, output, force=True)
    assert output.read_bytes() == payload
