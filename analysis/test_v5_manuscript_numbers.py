"""Submission checks for the v5 article.

The prose is intentionally free to improve.  These tests protect the parts that
must not drift during editing: the title, abstract contract, figure package and
the numerical claims that carry the argument.  Expected values are read from
the frozen result files rather than copied into a second narrative document.

Run with ``make verify-v5-numbers`` so that the staged figure package is also
rebuilt and checked.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
EXPECTED_TITLE = (
    "A shared ligand-wide score axis obscures library-dependent target geometry "
    "in docking matrices"
)

MAIN_SECTION_PATHS = sorted((ROOT / "sections").glob("*.tex"))
SI_SECTION_PATHS = sorted((ROOT / "si_sections").glob("*.tex"))


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def flat(text: str) -> str:
    """Make LaTeX line wrapping irrelevant to prose checks."""
    return re.sub(r"\s+", " ", text).strip()


def rows(relative: str) -> list[dict[str, str]]:
    with (RESULTS / relative).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def one(items: list[dict[str, str]], **where: str) -> dict[str, str]:
    matches = [row for row in items if all(row.get(key) == value for key, value in where.items())]
    assert len(matches) == 1, f"expected one row for {where}, found {len(matches)}"
    return matches[0]


def shown(value: str | float, digits: int) -> str:
    return f"{float(value):.{digits}f}"


def require_number(text: str, pattern: str, expected: str | float, digits: int, claim: str) -> None:
    """Require a number in its semantic sentence, not merely somewhere in the paper."""
    match = re.search(pattern, text, flags=re.IGNORECASE)
    assert match, f"could not find the {claim} claim"
    reported = float(match.group("value"))
    assert reported == pytest.approx(float(shown(expected, digits)), abs=0.5 * 10 ** (-digits)), (
        f"{claim}: article reports {reported}; frozen result rounds to {shown(expected, digits)}"
    )


@pytest.fixture(scope="module")
def main() -> str:
    sources = [ROOT / "manuscript_v5.tex", *MAIN_SECTION_PATHS]
    return flat("\n".join(read(path) for path in sources))


@pytest.fixture(scope="module")
def external() -> str:
    return flat(read(ROOT / "sections" / "04_results_external.tex"))


@pytest.fixture(scope="module")
def discussion() -> str:
    return flat(read(ROOT / "sections" / "06_discussion.tex"))


def abstract_words(source: str) -> list[str]:
    body = source.split(r"\textbf{Keywords:}", 1)[0]
    body = re.sub(r"%.*", " ", body)
    body = re.sub(r"\\(?:section\*|textbf|text|mathrm)\{([^{}]*)\}", r"\1", body)
    body = re.sub(r"\\[A-Za-z]+\*?", " ", body)
    body = re.sub(r"[$_{}\\]", " ", body)
    return re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*", body)


# ---------------------------------------------------------------------------
# Submission identity and readable front matter
# ---------------------------------------------------------------------------


def test_title_is_identical_in_all_submission_metadata() -> None:
    expected = flat(EXPECTED_TITLE)
    manuscript = flat(read(ROOT / "manuscript_v5.tex"))
    supplement = flat(read(ROOT / "supplementary_v5.tex"))
    citation = flat(read(ROOT / "CITATION.cff"))
    readme_first_line = read(ROOT / "README.md").splitlines()[0].strip()

    assert f"\\title{{{expected}}}" in manuscript
    assert f"pdftitle={{{expected}}}" in manuscript
    assert expected in supplement
    assert f'title: "{expected}"' in citation
    assert readme_first_line == f"# {expected}"


def test_abstract_has_one_clear_structured_contract() -> None:
    source = read(ROOT / "sections" / "00_abstract.tex")
    headings = ["Background.", "Results.", "Conclusions.", "Scientific Contribution."]
    positions = []
    for heading in headings:
        marker = rf"\textbf{{{heading}}}"
        assert source.count(marker) == 1, f"abstract must contain one {heading} heading"
        positions.append(source.index(marker))
    assert positions == sorted(positions), "abstract headings are out of order"
    assert len(abstract_words(source)) <= 350, "abstract exceeds the 350-word submission cap"

    contribution = source.split(r"\textbf{Scientific Contribution.}", 1)[1]
    contribution = contribution.split(r"\textbf{Keywords:}", 1)[0]
    plain = re.sub(r"\\[A-Za-z]+\*?(?:\[[^]]*\])?", " ", contribution)
    plain = re.sub(r"[$_{}\\]", " ", plain)
    sentences = [part for part in re.split(r"[.!?](?:\s|$)", flat(plain)) if part.strip()]
    assert 1 <= len(sentences) <= 3, (
        "Scientific Contribution must stay concise (one to three sentences)"
    )


def test_pdsp_two_sided_probability_definition_is_explicit() -> None:
    methods = flat(read(ROOT / "sections" / "02_methods.tex"))
    assert "twice the smaller of the add-one upper- and lower-tail probabilities" in methods
    assert "capped at one" in methods
    assert "Holm and Bonferroni adjustments within each 12-row PDSP QAP table" in methods


def test_article_has_exactly_four_report_figures() -> None:
    main_source = read(ROOT / "manuscript_v5.tex")
    supplement_source = "\n".join(read(path) for path in SI_SECTION_PATHS)
    main_figures = re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", main_source)

    assert main_figures == [
        "figures/v5/fig1_shared_axis.pdf",
        "figures/v5/fig2_residual_structure.pdf",
        "figures/v5/fig3_external_agreement.pdf",
        "figures/v5/fig4_cost_and_core.pdf",
    ]
    assert "fig_pocket_volume" not in main_source
    assert "fig_pocket_volume" not in supplement_source

    titles = re.findall(r"\\caption\{\\textbf\{(.+?)\}", main_source, flags=re.DOTALL)
    assert len(titles) == 4
    for title in titles:
        words = flat(title).split()
        assert len(words) <= 15, f"figure title exceeds 15 words: {flat(title)}"


def test_staged_submission_art_is_exact_and_within_limits() -> None:
    from PIL import Image

    stage = ROOT / "submission" / "figures"
    expected = {
        "Fig1.pdf",
        "Fig2.pdf",
        "Fig3.pdf",
        "Fig4.pdf",
        "GraphicalAbstract.pdf",
        "GraphicalAbstract_920x300.png",
    }
    assert {path.name for path in stage.iterdir() if path.is_file()} == expected
    for name in expected - {"GraphicalAbstract_920x300.png"}:
        assert (stage / name).stat().st_size > 1_000, f"{name} looks empty"

    graphical = stage / "GraphicalAbstract_920x300.png"
    assert graphical.stat().st_size <= 150_000
    with Image.open(graphical) as image:
        assert image.size == (920, 300)


def test_graphical_abstract_uses_canonical_transform_and_neutral_pilot_result() -> None:
    import numpy as np

    from analysis.make_v5_graphical_abstract import residual_correlation

    scores = np.array(
        [
            [-9.0, -3.0, -2.0],
            [-8.0, -1.0, -4.0],
            [-2.0, -7.0, -1.0],
            [-1.0, -9.0, -6.0],
            [-4.0, -5.0, -8.0],
        ]
    )
    column_centred = scores - scores.mean(axis=0, keepdims=True)
    two_way = column_centred - column_centred.mean(axis=1, keepdims=True)
    expected = np.corrcoef(two_way, rowvar=False)
    assert np.allclose(residual_correlation(scores), expected)

    source = read(ROOT / "analysis" / "make_v5_graphical_abstract.py")
    assert "calibration_panel_recovery" in source
    assert "certified_global_metrics" not in source


def test_figure_two_uses_only_the_declared_molecular_weight_null() -> None:
    from analysis.make_v5_figures import _mw_conditioned_null_repeats

    frame = _mw_conditioned_null_repeats()
    assert set(frame["bin_design"]) == {"molecular_weight_quantile"}
    assert set(frame["requested_mw_bins"]) == {10}
    medians = frame.groupby("dataset")["participation_ratio"].median().to_dict()
    assert medians == pytest.approx(
        {"docking44": 34.87523822972109, "dockstring58": 47.03913867341176}
    )


def test_figure_driver_preserves_support_uncertainty_and_accessibility_contracts() -> None:
    source = read(ROOT / "analysis" / "make_v5_figures.py")
    graphical = read(ROOT / "analysis" / "make_v5_graphical_abstract.py")

    assert 'support = "exact_connectivity_excluded"' in source
    assert 'core["reference_support"] == support' in source
    assert 'recovery["geometry_spearman_q025"].min()' in source
    assert "ax_cos.set_xlim(0.936, 0.996)" in source
    assert '"ROC-AUC", SLATE, "///"' in source
    assert 'metadata={"CreationDate": None}' in source
    assert 'metadata={"CreationDate": None}' in graphical
    assert "fig_pocket_volume" not in source

    core = rows("biological_core_modes/panel_metrics.csv")
    exact = [row for row in core if row["reference_support"] == "exact_connectivity_excluded"]
    assert sorted(row["panel"] for row in exact) == ["DAVIS", "KiRHub", "PKIS1", "PKIS2"]


def test_figure_three_names_the_assay_transform_unambiguously() -> None:
    source = read(ROOT / "analysis" / "make_v5_figures.py")
    for label in ("raw assay", "centred assay"):
        assert label in source
    assert "SPD bound  ·  raw map" not in source


# ---------------------------------------------------------------------------
# Claims read directly against frozen result artifacts
# ---------------------------------------------------------------------------


def test_fixed_pose_scorer_ranges_match_saved_metrics(main: str) -> None:
    metrics = rows("nonvina_scorer_transport/scorer_spectral_metrics.csv")
    assert len(metrics) == 8
    pc1 = [float(row["raw_pc1_fraction"]) for row in metrics]
    cosine = [float(row["pc1_uniform_cosine"]) for row in metrics]

    require_number(main, r"leading principal component .*? between (?P<value>[0-9.]+) and [0-9.]+", min(pc1), 3, "minimum PC1 fraction")
    require_number(main, r"leading principal component .*? between [0-9.]+ and (?P<value>[0-9.]+)", max(pc1), 3, "maximum PC1 fraction")
    require_number(main, r"loading vector has cosine (?P<value>[0-9.]+) to [0-9.]+", min(cosine), 3, "minimum uniform cosine")
    require_number(main, r"loading vector has cosine [0-9.]+ to (?P<value>[0-9.]+)", max(cosine), 3, "maximum uniform cosine")


def test_docking44_carrier_columns_and_source_panel_selection_are_explicit(main: str) -> None:
    from analysis.residual_mie_boundary_audit import DOCK44

    import pandas as pd

    header = pd.read_csv(RESULTS.parent / "data" / "frozen" / "df_final_v4.csv.gz", nrows=0)
    carrier_targets = [*DOCK44, "8pm6", "6itm", "9fzj"]
    assert all(target in header.columns for target in carrier_targets)
    assert len(DOCK44) == 44
    for token in ("8pm6", "6itm", "9fzj", "appended", "published source panel"):
        assert token in main


def test_pdsp_primary_and_certified_claims_match_saved_metrics(external: str) -> None:
    primary = rows("pdsp_counterscreen_retrieval/global_retrieval_metrics.csv")
    certified = rows("pdsp_counterscreen_retrieval/certified_global_metrics.csv")
    raw = one(primary, top_fraction="0.1", predictor="raw_docking")
    residual = one(primary, top_fraction="0.1", predictor="residual_docking")
    cert_raw = one(certified, top_fraction="0.1", predictor="raw_docking")
    cert_residual = one(certified, top_fraction="0.1", predictor="residual_docking")

    assert f"rho={shown(raw['continuous_spearman'], 3)}" in external
    assert shown(residual["continuous_spearman"], 3) in external
    assert shown(cert_raw["continuous_spearman"], 3) in external
    assert shown(cert_residual["continuous_spearman"], 3) in external


def test_pdsp_qap_overlap_and_support_boundaries_match_frozen_outputs(
    external: str, discussion: str
) -> None:
    qap = rows("pdsp_counterscreen_retrieval/paired_qap.csv")
    unrestricted = one(
        qap,
        permutation_scheme="all_target_labels",
        contrast="residual_minus_raw_joint_label",
        metric="continuous_spearman",
    )
    within_family = one(
        qap,
        permutation_scheme="within_curated_family",
        contrast="residual_minus_raw_joint_label",
        metric="continuous_spearman",
    )
    fusion = one(
        qap,
        permutation_scheme="within_curated_family",
        contrast="residual_increment_to_sequence_family",
        metric="continuous_spearman",
    )
    assert f"p={shown(unrestricted['two_sided_qap_p'], 4)}" in external
    assert f"p={shown(unrestricted['holm_adjusted_two_sided_qap_p'], 4)}" in external
    assert f"p={shown(within_family['two_sided_qap_p'], 4)}" in external
    assert f"p={shown(within_family['holm_adjusted_two_sided_qap_p'], 4)}" in external
    assert shown(fusion["observed_gain"], 3) in external
    assert float(fusion["holm_adjusted_two_sided_qap_p"]) >= 0.05

    summary = json.loads(read(RESULTS / "pdsp_counterscreen_retrieval" / "summary.json"))
    audit = summary["pdsp_audit"]
    exclusions = rows("pdsp_counterscreen_retrieval/excluded_docking44_ligand_ids.csv")
    assert len(exclusions) == audit["docking_compounds_excluded_for_pdsp_connectivity_overlap"]
    for value in (
        audit["exact_pdsp_docking_full_inchikey_overlap"],
        audit["exact_pdsp_docking_connectivity_block_overlap"],
        audit["docking_compounds_excluded_for_pdsp_connectivity_overlap"],
    ):
        assert str(value) in external

    complete = rows("pdsp_counterscreen_retrieval/complete_case_support_metrics.csv")
    complete_qap = rows("pdsp_counterscreen_retrieval/complete_case_support_qap.csv")
    sweep = rows("pdsp_counterscreen_retrieval/minimum_pair_support_sensitivity.csv")
    raw_complete = one(complete, predictor="raw_docking")
    residual_complete = one(complete, predictor="residual_docking")
    assert len(complete_qap) == 12 and len(sweep) == 6
    assert shown(raw_complete["continuous_spearman"], 3) in external
    assert shown(residual_complete["continuous_spearman"], 3) in external
    assert "support-sensitive" in external
    assert "support-sensitive" in discussion
    assert "retained support" not in external.lower()


def test_kinase_and_kirhub_boundaries_match_saved_results(external: str, discussion: str) -> None:
    kinase = rows("klifs_pocket_control/locked_endpoint_retrieval.csv")
    sequence = one(kinase, predictor="receptor_domain_sequence_identity")
    centered = one(kinase, predictor="centered_Vina_target_geometry")
    for value in (
        shown(centered["roc_auc"], 3),
        shown(centered["average_precision"], 3),
        shown(sequence["roc_auc"], 3),
        shown(sequence["average_precision"], 3),
    ):
        assert value in external
        assert value in discussion

    kirhub = json.loads(read(RESULTS / "kirhub_external_validation" / "summary.json"))
    metrics = kirhub["KiRHub_specific_top10_boundary"]["metrics"]
    for key in ("roc_auc", "average_precision"):
        raw = shown(metrics["raw_docking"][key], 3)
        centered_value = shown(metrics["centered_docking"][key], 3)
        assert raw in external and centered_value in external
        assert raw in discussion and centered_value in discussion
    assert "non-replication" in external.lower()


def test_safety_panel_claims_match_saved_summary(external: str) -> None:
    summary = json.loads(read(RESULTS / "spd_external_validation" / "summary.json"))
    bundle_readme = flat(read(RESULTS / "spd_external_validation" / "README.md"))
    floor = summary["key_results"]["primary_floor_at_bound"]
    rectangle = summary["key_results"]["common_support_rectangle"]
    ten = summary["key_results"]["censor_aware_binary"]["10_uM"]

    for value in (floor["raw_spearman"], floor["residual_spearman"]):
        assert shown(value, 3) in external
    for value in (rectangle["raw_spearman"], rectangle["residual_spearman"]):
        assert shown(value, 3) in external
    assert shown(rectangle["raw_partial_rank"], 3) in external
    assert shown(rectangle["residual_partial_rank"], 3) in external
    assert shown(ten["raw_spearman"], 3) in external
    assert shown(ten["residual_spearman"], 3) in external

    assert "frozen v5 manuscript evidence bundle" in bundle_readme
    assert "not yet part of the manuscript" not in bundle_readme
    for value in (
        floor["raw_spearman"],
        floor["residual_spearman"],
        floor["residual_minus_raw"],
        rectangle["raw_spearman"],
        rectangle["residual_spearman"],
        rectangle["residual_partial_rank"],
    ):
        assert shown(value, 3) in bundle_readme
    unrestricted = {
        (row["endpoint"], row["metric"]): float(row["p_positive"])
        for row in rows("spd_external_validation/target_label_qap.csv")
        if row["permutation_scheme"] == "unrestricted"
    }
    for key in (
        ("floor_at_bound", "residual_spearman"),
        ("floor_at_bound", "residual_minus_raw"),
        ("censor_aware_binary_10uM", "residual_minus_raw"),
        ("censor_aware_binary_30uM", "residual_minus_raw"),
    ):
        assert shown(unrestricted[key], 4) in bundle_readme


def test_recovery_domain_and_single_ligand_boundaries_match_artifacts(main: str, discussion: str) -> None:
    recovery = rows("calibration_panel_recovery/summary.csv")
    dockstring = one(recovery, dataset="DOCKSTRING-58", calibration_ligands="200")
    docking44 = one(recovery, dataset="Docking-44", calibration_ligands="200")
    assert shown(dockstring["geometry_spearman_mean"], 3) in main
    assert shown(docking44["geometry_spearman_mean"], 3) in main

    domain = json.loads(read(RESULTS / "public_chemical_domain_controls" / "summary.json"))
    residual_rows = {
        row["dataset"]: row
        for row in domain["key_boundary_results"]
        if row["transformation"] == "row_centered_residual"
    }
    for row in residual_rows.values():
        assert shown(row["observed_low_high_geometry_spearman"], 3) in main
        assert shown(row["global_mw_matched_group_disjoint_median"], 3) in main

    mie = rows("residual_mie_boundary_audit/observed_metrics.csv")
    absolute = one(mie, representation="absolute", weighting="compound_balanced", metric="top5")
    residual = one(mie, representation="two_way_residual", weighting="compound_balanced", metric="top5")
    assert f"{100 * float(residual['estimate']):.1f}\\%" in discussion
    assert f"{100 * float(absolute['estimate']):.1f}\\%" in discussion


# ---------------------------------------------------------------------------
# Boundaries, disclosure and provenance
# ---------------------------------------------------------------------------


def test_reviewer_requested_claim_boundaries_remain_explicit() -> None:
    abstract = flat(read(ROOT / "sections" / "00_abstract.tex"))
    methods = flat(read(ROOT / "sections" / "02_methods.tex"))
    axis = flat(read(ROOT / "sections" / "03_results_axis.tex"))
    external = flat(read(ROOT / "sections" / "04_results_external.tex"))
    cost = flat(read(ROOT / "sections" / "05_results_cost.tex"))
    discussion = flat(read(ROOT / "sections" / "06_discussion.tex"))
    manuscript = flat(read(ROOT / "manuscript_v5.tex"))
    s2 = flat(read(ROOT / "si_sections" / "S2_spectra_nulls.tex"))
    s3 = flat(read(ROOT / "si_sections" / "S3_transport.tex"))
    s5 = flat(read(ROOT / "si_sections" / "S5_recovery.tex"))
    s6 = flat(read(ROOT / "si_sections" / "S6_experimental.tex"))

    assert "alongside the raw one rather than in place of it" in abstract
    assert "no difference between their outputs can be attributed to a different selected pose" in methods
    assert "scorer-and-implementation sensitivity at fixed geometry" in methods
    assert "scorer- and implementation-specific at fixed geometry" in axis
    assert "reproduces part of the concentration" in axis
    assert "not identified as the cause" in axis
    assert "tracks a chemical-domain shift strongly aligned with molecular size" in axis
    assert "exact- and connectivity-de-overlapped test" in external
    assert "sharing exact identities or connectivity blocks were excluded" in manuscript
    assert "approaches the sequence-identity baseline" in discussion
    assert "without establishing equivalence" in s6
    assert "five-mode compression is exploratory" in cost
    assert all(token in cost for token in ("six-test", "sensitivity", "post hoc"))
    assert "exploratory subspace compression" in s5
    assert "map-construction and diagnostic stage" in cost
    assert "null constructions are supplied as separate analysis scripts" in cost
    assert "does not identify molecular weight as the operative cause" in s2

    narrative = " ".join((abstract, methods, axis, external, cost, discussion, manuscript, s2, s3, s5, s6))
    forbidden = (
        r"comes from (?:the )?scoring function(?: alone)?",
        r"coarse size accounts for",
        r"how much coarse size explains",
        r"reaches (?:the )?sequence(?:-identity| identity)",
        r"chemically non-overlapping",
        r"the non-overlapping DOCKSTRING support",
        r"\\subsection\{[^}]*experimental agreement",
    )
    for pattern in forbidden:
        assert not re.search(pattern, narrative, flags=re.IGNORECASE), pattern


def test_censoring_and_inference_boundaries_remain_explicit(main: str, discussion: str) -> None:
    censor = rows("pdsp_counterscreen_retrieval/censor_sensitivity_metrics.csv")
    restored = one(
        censor,
        sensitivity="all_allowed_rows_right_censored_at_reported_bound",
        top_fraction="0.1",
        predictor="residual_docking",
    )
    value = shown(restored["continuous_spearman"], 3)
    assert value in discussion
    assert value in main

    methods = read(ROOT / "sections" / "02_methods.tex")
    limitations = discussion.split("Scope and limitations", 1)[-1]
    for token in ("one-sided", "two-sided", "post hoc", "Holm", "Bonferroni", "12"):
        assert token in methods, f"Methods must declare {token}"
    for token in ("Holm", "Bonferroni", "12 correlated rows"):
        assert token in limitations, f"Limitations must retain {token}"


def test_non_human_structure_sensitivity_is_retained(main: str) -> None:
    from scipy import stats
    from sklearn.metrics import roc_auc_score

    from analysis.pdsp_counterscreen_retrieval import add_fixed_fusions, top_fraction_labels

    species = rows("docking_structure_species/structure_species.csv")
    non_human = sorted(row["gene"] for row in species if row["receptor_is_human"] == "False")
    assert non_human == ["ADRB1", "GRIN1", "HTR3A", "PTGS1", "PTGS2"]
    for token in ("non-human receptor chains", "119 pairs", "ADRB1", "HTR3A"):
        assert token in main

    import pandas as pd

    pairs = pd.read_csv(RESULTS / "pdsp_counterscreen_retrieval" / "target_pairs.csv")
    keep = ~pairs["target_a"].isin({"ADRB1", "HTR3A"}) & ~pairs["target_b"].isin(
        {"ADRB1", "HTR3A"}
    )
    clean = add_fixed_fusions(pairs.loc[keep].copy())
    assert len(clean) == 119
    base = stats.spearmanr(
        clean["experimental_spearman"], clean["equal_rank_sequence_family"]
    ).statistic
    fusion = stats.spearmanr(
        clean["experimental_spearman"], clean["equal_rank_sequence_family_residual"]
    ).statistic
    labels = top_fraction_labels(clean["experimental_spearman"].to_numpy(), 0.10)
    auc_gain = roc_auc_score(
        labels, clean["equal_rank_sequence_family_residual"]
    ) - roc_auc_score(labels, clean["equal_rank_sequence_family"])
    assert f"{fusion - base:+.3f}" in main
    assert f"{auc_gain:+.3f}" in main


def test_external_sources_and_declarations_are_auditable() -> None:
    sources = []
    with (ROOT / "external_sources.csv").open(newline="", encoding="utf-8") as handle:
        sources = list(csv.DictReader(handle))
    assert sources
    for source in sources:
        assert source["access"].strip()
        assert source["license"].strip()
        assert source["redistributed"] in {"yes", "no"}
        digest = source["sha256"].strip()
        integrity = source["integrity"].strip()
        if digest:
            assert re.fullmatch(r"[0-9a-f]{64}", digest), source["source"]
        if source["redistributed"] == "no":
            assert digest or integrity, source["source"]

    with (ROOT / "JOC_ACCESS_MATRIX.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        access_rows = list(csv.DictReader(handle))
    assert len(access_rows) == len(sources)
    assert {row["source_id"] for row in access_rows} == {
        source["source"] for source in sources
    }
    status_columns = [
        column for column in access_rows[0] if column.endswith("_status")
    ]
    assert status_columns
    assert "analysis_producer_present_status" in status_columns
    assert "source_producer_present_status" not in access_rows[0]
    for row in access_rows:
        assert row["required_artifact"].strip()
        assert row["evidence_locator"].strip()
        assert row["remaining_evidence_needed"].strip()
        assert row["verified_on"] == "2026-08-12"
        assert all(row[column] in {"VERIFIED", "UNKNOWN"} for column in status_columns)
        for locator in row["evidence_locator"].split(";"):
            local = locator.strip()
            if local.startswith(("analysis/", "data/", "results/", "sections/", "si_sections/")):
                assert (ROOT / local).is_file(), (row["source_id"], local)
    assert any(
        row[column] == "UNKNOWN"
        for row in access_rows
        for column in status_columns
    )
    by_source = {row["source_id"]: row for row in access_rows}
    assert by_source["smina 2020.12.10 executable"]["policy_compatible_license_status"] == "UNKNOWN"
    for column in (
        "release_includes_consumed_bytes_status",
        "consumed_digest_recorded_status",
        "checksum_gate_status",
    ):
        assert by_source["RCSB PDB structure metadata"][column] == "UNKNOWN"
    assert by_source["ChEMBL release 34"]["checksum_gate_status"] == "VERIFIED"

    declarations = read(ROOT / "results" / "v5_declarations.tex")
    for token in (
        "PDSP",
        r"\cite{pdspdatabase}",
        "8103950",
        "JOC_ACCESS_MATRIX.csv",
        "remain unknown",
        "Additional file 1",
    ):
        assert token in declarations
    references = read(ROOT / "references_v5.tex")
    assert "https://pdsp.unc.edu/databases/kiDownload/download.php" in references

    with (ROOT / "data_manifest.csv").open(newline="", encoding="utf-8") as handle:
        manifest = list(csv.DictReader(handle))
    crosswalk_rows = [
        row for row in manifest if "anastassiadis2011_pubchem_identity" in row["path"]
    ]
    assert len(crosswalk_rows) == 2
    assert {row["license"] for row in crosswalk_rows} == {
        "LicenseRef-NCBI-PubChem-molecular-data-policy"
    }
    license_notes = read(ROOT / "DATA_LICENSES.md")
    assert "crosswalk is not labelled CC0" in license_notes
    assert "derived crosswalk is CC0" not in license_notes


def test_submission_sources_never_reference_promotional_art() -> None:
    sources = [
        ROOT / "manuscript_v5.tex",
        ROOT / "supplementary_v5.tex",
        ROOT / "results" / "v5_declarations.tex",
        *MAIN_SECTION_PATHS,
        *SI_SECTION_PATHS,
    ]
    for path in sources:
        source = read(path)
        assert "promo/" not in source and "concept_art" not in source, path.name
