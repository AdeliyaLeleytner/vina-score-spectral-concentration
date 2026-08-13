from __future__ import annotations

import hashlib

import pytest

from analysis import fetch_pdsp_ki_database as fetcher


def payload() -> bytes:
    return fetcher.EXPECTED_HEADER + b"\n1,TEST\n"


def test_fetch_is_atomic_checksum_gated_and_refuses_overwrite(tmp_path, monkeypatch) -> None:
    content = payload()
    monkeypatch.setattr(fetcher, "SHA256", hashlib.sha256(content).hexdigest())
    monkeypatch.setattr(fetcher, "EXPECTED_BYTES", len(content))

    def downloader(url: str, destination) -> None:
        assert url == fetcher.URL
        destination.write_bytes(content)

    output = tmp_path / "KiDatabase.csv"
    assert fetcher.fetch(output, downloader=downloader) == output
    assert output.read_bytes() == content
    with pytest.raises(FileExistsError, match="--force"):
        fetcher.fetch(output, downloader=downloader)
    fetcher.fetch(output, force=True, downloader=downloader)
    assert output.read_bytes() == content


def test_validation_fails_closed_on_changed_bytes(tmp_path) -> None:
    changed = tmp_path / "changed.csv"
    changed.write_bytes(fetcher.EXPECTED_HEADER + b"\nchanged\n")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        fetcher.validate(changed)
