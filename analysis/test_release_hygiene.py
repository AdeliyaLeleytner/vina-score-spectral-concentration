from __future__ import annotations

import ast
import csv
import json
import re
from pathlib import Path

import pytest
from PIL import Image

import build_dockstring_identity_contract as identity_contract
import build_evidence
import make_figures
import sequence_docking_fusion as fusion
from build_result_bundle_checksums import MANIFEST_NAME, REQUIRED_BUNDLES
from verify_inputs import verify_bundle_manifest


ANALYSIS = Path(__file__).resolve().parent
PACKAGE = ANALYSIS.parent


def imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text())
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                modules.add(node.module)
            modules.update(alias.name for alias in node.names)
    return modules


@pytest.mark.parametrize(
    ("smiles", "expected"),
    [
        (
            "[Na+].CC(=O)[O-]",
            (
                "VMHLLURERBWHNL-UHFFFAOYSA-M",
                "VMHLLURERBWHNL",
                "",
                "QTBSBXVTEAMEQO",
                "",
            ),
        ),
        (
            "Oc1ccccc1",
            (
                "ISWSIDIOOBJBQZ-UHFFFAOYSA-N",
                "ISWSIDIOOBJBQZ",
                "c1ccccc1",
                "ISWSIDIOOBJBQZ",
                "c1ccccc1",
            ),
        ),
        (
            "CCO",
            (
                "LFQSCWFLJHTTHZ-UHFFFAOYSA-N",
                "LFQSCWFLJHTTHZ",
                "",
                "LFQSCWFLJHTTHZ",
                "",
            ),
        ),
    ],
)
def test_identity_builder_preserves_raw_and_standardized_contract(
    smiles: str, expected: tuple[str, str, str, str, str]
) -> None:
    assert identity_contract.derive_identity(smiles) == expected


def test_identity_builder_has_no_ignored_graph_module_import() -> None:
    imported = imported_modules(ANALYSIS / "build_dockstring_identity_contract.py")
    assert "sea_graph_chemistry" not in imported


def test_fusion_has_no_exploratory_core_import() -> None:
    imported = imported_modules(ANALYSIS / "sequence_docking_fusion.py")
    assert "biological_core_modes" not in imported


def test_fusion_holm_adjustment_preserves_named_family_results() -> None:
    assert fusion.holm_adjust({"middle": 0.04, "small": 0.01, "large": 0.20}) == {
        "small": pytest.approx(0.03),
        "middle": pytest.approx(0.08),
        "large": pytest.approx(0.20),
    }


@pytest.mark.parametrize(
    ("module", "replacement"),
    [
        (build_evidence, "analysis/build_manuscript_evidence.py"),
        (make_figures, "analysis/make_manuscript_figures.py"),
    ],
)
def test_legacy_main_paths_fail_before_creating_outputs(
    module: object, replacement: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "must-not-exist"
    monkeypatch.setattr(module, "OUT", output)
    with pytest.raises(SystemExit, match=re.escape(replacement)):
        module.main()
    assert not output.exists()


def test_manuscript_facing_result_bundles_have_complete_checksums() -> None:
    source_manifest = PACKAGE / "results/manuscript_source_manifest.csv"
    with source_manifest.open(newline="") as handle:
        registered = {row["path"] for row in csv.DictReader(handle)}
    for bundle in REQUIRED_BUNDLES:
        manifest = PACKAGE / "results" / bundle / MANIFEST_NAME
        assert manifest.is_file(), bundle
        count, failures = verify_bundle_manifest(manifest)
        assert count > 0, bundle
        assert failures == [], failures
        relative = manifest.relative_to(PACKAGE).as_posix()
        assert relative in registered, f"bundle manifest not source-registered: {relative}"


def test_fixed_pose_release_summaries_are_portable_and_explicit() -> None:
    vina = json.loads(
        (PACKAGE / "results/dockstring_vina_terms/analysis_summary.json").read_text()
    )
    transport = json.loads(
        (PACKAGE / "results/nonvina_scorer_transport/summary.json").read_text()
    )
    for payload in (vina, transport):
        serialized = json.dumps(payload)
        assert "/Users/" not in serialized
        assert "/private/tmp/" not in serialized
        assert "reproducibility_boundary" in payload
    assert not vina["reproducibility_boundary"]["independent_redocking_performed"]
    assert not transport["reproducibility_boundary"][
        "independent_redocking_per_scorer"
    ]


def test_graphical_abstract_uses_the_current_claim_and_exact_canvas() -> None:
    source = (ANALYSIS / "make_graphical_abstract.py").read_text()
    for retired_claim in (
        "Seven ligand descriptors alone",
        "reproduce the agreement",
        "equivalent within",
        "Normalization does not",
    ):
        assert retired_claim not in source
    assert "molecular-size domain" in source
    assert "same target order, exactly" in source
    graphical_abstract = PACKAGE / "figures/graphical_abstract.png"
    assert graphical_abstract.stat().st_size <= 150_000
    with Image.open(graphical_abstract) as rendered:
        assert rendered.size == (920, 300)
        rgba = rendered.convert("RGBA")
        assert all(
            rgba.getpixel(point) == (255, 255, 255, 255)
            for point in ((0, 0), (919, 0), (0, 299), (919, 299))
        )


def test_main_figure_builder_calls_every_manuscript_figure() -> None:
    source = (ANALYSIS / "make_manuscript_figures.py").read_text()
    main_block = source.split("def main() -> None:", 1)[1]
    for call in (
        "figure1(ledger)",
        "figure2_support_recovery(ledger)",
        "figure3(ledger)",
        "figure4(ledger)",
        "figure5(ledger)",
    ):
        assert call in main_block
    manuscript = (PACKAGE / "manuscript.tex").read_text()
    for stem in (
        "fig1_spectral_geometry",
        "fig4_support_recovery",
        "fig3_external_boundary",
        "fig4_feature_controls",
        "fig5_ranking_boundary",
    ):
        assert f"figures/{stem}.pdf" in manuscript
        assert f'save(fig, "{stem}")' in source


def test_broad_target_jackknife_is_described_as_fixed_score_reevaluation() -> None:
    manuscript = (PACKAGE / "manuscript.tex").read_text()
    provenance = (PACKAGE / "PROVENANCE.md").read_text()
    retired_false_claim = (
        "Delete-one-target jackknifing\nrefitted the complete transformation"
    )
    assert retired_false_claim not in manuscript
    assert "reevaluated the already externally fitted score" in manuscript
    assert "reevaluates the externally fitted\n  score representations" in provenance


def test_readme_endpoint_mix_percentages_match_the_frozen_diagnostic() -> None:
    summary = json.loads(
        (PACKAGE / "results/dockstring_chembl_ranking/summary.json").read_text()
    )
    support = summary["support"]
    mixing = summary["sensitivities"]["human_binding_Ki_Kd"][
        "endpoint_mixing_diagnostic"
    ]["fractions_of_all_informative_pairs"]
    readme = " ".join((PACKAGE / "README.md").read_text().split())
    assert (
        f"contains {support['evaluated_ligands']:,} informative ligands, "
        f"{support['targets']} targets and "
        f"{support['non_tied_within_ligand_target_pairs']:,} non-tied observed "
        "within-ligand comparisons"
        in readme
    )
    assert f"{100 * mixing['same_endpoint']:.1f}% of pairs" in readme
    assert (
        f"{100 * mixing['mixed_endpoint_Ki_Kd']:.1f}% were unambiguous Ki–Kd"
        in readme
    )
    assert (
        f"{100 * mixing['involves_pooled_Ki_Kd_cell']:.1f}% involved at least one "
        "cell pooling both endpoint types"
        in readme
    )
