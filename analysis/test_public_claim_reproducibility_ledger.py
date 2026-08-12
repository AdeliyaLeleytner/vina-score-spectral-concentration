from __future__ import annotations

import hashlib
import json
from pathlib import Path


PACKAGE = Path(__file__).resolve().parents[1]
LEDGER_PATH = PACKAGE / "results/public_claim_reproducibility_ledger.json"
ALLOWED_LEVELS = {
    "row_level_reanalysis",
    "aggregate_recomputation",
    "checksum_authenticated_report_reconstruction",
}
ALLOWED_SCOPE_STATUSES = {
    "keep_main",
    "rebuild_davis_pkis2_only_before_use",
    "rebuild_davis_only_before_use",
    "rebuild_public_only_before_use",
    "remove_from_submission_report_layer",
}
REQUIRED_EXTERNAL_INPUTS = {
    "EXT-PKIS1",
    "EXT-KIRHUB",
    "EXT-KLIFS",
    "EXT-SPD",
    "EXT-POSES",
    "EXT-ODDT39",
    "EXT-UPSTREAM-DOCKING",
}
REQUIRED_CLAIMS = {
    "CLM-SPECTRAL",
    "CLM-CHEMICAL-DOMAIN",
    "CLM-PROBE-RECOVERY",
    "CLM-FEATURE-CONTROLS",
    "CLM-FIXED20-FACTORIAL",
    "CLM-LOCKED-ENDPOINT",
    "CLM-DAVIS-CENSORING",
    "CLM-SPD-SENSITIVITY",
    "CLM-BROAD-CHEMBL-RANKING",
    "CLM-DENSE-RANKING",
    "CLM-LEGACY-SPARSE",
    "CLM-NONVINA-RESCORING",
    "CLM-VINA-TERMS",
    "CLM-REPORT-REGISTRY",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_ledger() -> dict:
    return json.loads(LEDGER_PATH.read_text(encoding="utf-8"))


def assert_safe_regular_file(relative: str) -> Path:
    candidate = Path(relative)
    assert not candidate.is_absolute()
    assert ".." not in candidate.parts
    path = PACKAGE / candidate
    assert path.is_file(), relative
    assert not path.is_symlink(), relative
    return path


def test_claim_ledger_schema_and_coverage() -> None:
    ledger = load_ledger()
    assert ledger["schema_version"] == "1.0.0"
    interpretation = ledger["interpretation"]
    assert interpretation[
        "all_reported_numeric_claims_present_and_checksum_verifiable_in_public_frozen_artifacts"
    ]
    assert not interpretation[
        "all_reported_numeric_claims_recomputable_from_public_frozen_artifacts"
    ]
    assert not interpretation[
        "all_source_level_analyses_rerunnable_from_row_level_inputs_bundled_in_the_release"
    ]

    external = ledger["restricted_or_nonbundled_inputs"]
    external_ids = [record["id"] for record in external]
    assert len(external_ids) == len(set(external_ids))
    assert set(external_ids) == REQUIRED_EXTERNAL_INPUTS

    claims = ledger["claims"]
    claim_ids = [record["id"] for record in claims]
    assert len(claim_ids) == len(set(claim_ids))
    assert set(claim_ids) == REQUIRED_CLAIMS
    for claim in claims:
        assert claim["reproduction_level"] in ALLOWED_LEVELS
        assert claim["planned_scope_status"] in ALLOWED_SCOPE_STATUSES
        assert claim["manuscript_locations"]
        assert claim["public_inputs"]
        assert claim["analysis_scripts"]
        assert claim["public_outputs"]
        assert set(claim["external_input_ids"]).issubset(external_ids)
        if claim["reproduction_level"] == "row_level_reanalysis":
            assert not {
                "EXT-PKIS1",
                "EXT-KIRHUB",
                "EXT-KLIFS",
                "EXT-SPD",
                "EXT-POSES",
                "EXT-ODDT39",
            }.intersection(claim["external_input_ids"])

    scope = ledger["planned_public_only_submission_scope"]
    scope_ids = (
        set(scope["keep_main_claim_ids"])
        | set(scope["rebuild_before_use_claim_ids"])
        | set(scope["remove_from_submission_report_layer_claim_ids"])
    )
    assert scope_ids == set(claim_ids)
    claim_index = {claim["id"]: claim for claim in claims}
    for claim_id in scope["keep_main_claim_ids"]:
        claim = claim_index[claim_id]
        assert claim["planned_scope_status"] == "keep_main"
        assert claim["external_input_ids"] == []
    for claim_id in scope["remove_from_submission_report_layer_claim_ids"]:
        assert claim_index[claim_id]["planned_scope_status"] == (
            "remove_from_submission_report_layer"
        )

    linked_claims = {
        claim_id
        for record in external
        for claim_id in record["claim_ids"]
    }
    assert linked_claims.issubset(claim_ids)
    for record in external:
        assert record["claim_ids"]
        assert record["public_substitutes"]
        for relative in record["public_substitutes"]:
            assert_safe_regular_file(relative)


def test_every_declared_public_artifact_matches_its_checksum() -> None:
    ledger = load_ledger()
    checked: set[tuple[str, str]] = set()
    for claim in ledger["claims"]:
        for role in ("public_inputs", "public_outputs"):
            for artifact in claim[role]:
                key = (artifact["path"], artifact["sha256"])
                if key in checked:
                    continue
                path = assert_safe_regular_file(artifact["path"])
                assert sha256_file(path) == artifact["sha256"], artifact["path"]
                checked.add(key)
        for relative in claim["analysis_scripts"]:
            path = assert_safe_regular_file(relative)
            assert path.suffix == ".py", relative
    assert len(checked) >= 25


def test_restricted_claims_do_not_masquerade_as_bundled_row_level_reanalysis() -> None:
    ledger = load_ledger()
    external = {
        record["id"]: record
        for record in ledger["restricted_or_nonbundled_inputs"]
    }
    claims = {record["id"]: record for record in ledger["claims"]}
    for external_id in ("EXT-PKIS1", "EXT-KIRHUB", "EXT-KLIFS", "EXT-SPD"):
        for claim_id in external[external_id]["claim_ids"]:
            assert claims[claim_id]["reproduction_level"] != "row_level_reanalysis"

    fixed = claims["CLM-FIXED20-FACTORIAL"]
    assert fixed["reproduction_level"] == "aggregate_recomputation"
    assert any(
        artifact["path"]
        == "results/released_pair_geometry_ledger/target_pair_geometry.csv"
        for artifact in fixed["public_inputs"]
    )
    nonvina = claims["CLM-NONVINA-RESCORING"]
    assert nonvina["reproduction_level"] == (
        "checksum_authenticated_report_reconstruction"
    )


def test_aggregate_pair_ledger_is_now_directly_checksum_anchored() -> None:
    ledger = load_ledger()
    expected = "d7e7ee2bd9381595f344f61dbdf748f57cebc146e5bdb66ef577e915676c4ff6"
    declared = {
        artifact["sha256"]
        for claim in ledger["claims"]
        for artifact in claim["public_inputs"]
        if artifact["path"]
        == "results/released_pair_geometry_ledger/target_pair_geometry.csv"
    }
    assert declared == {expected}
    path = PACKAGE / "results/released_pair_geometry_ledger/target_pair_geometry.csv"
    assert sha256_file(path) == expected
