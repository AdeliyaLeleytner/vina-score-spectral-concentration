"""Submission checks for numerical and structural claims in the v5 Supplement.

The Supplement contains the detailed sensitivities that would make the main
article hard to read.  These checks bind those details to the frozen tables and
keep the seven-section navigation and release instructions intact.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
EXPECTED_TITLE = (
    "A shared ligand-wide score axis obscures library-dependent target geometry "
    "in docking matrices"
)
SI_PATHS = sorted((ROOT / "si_sections").glob("S*.tex"))


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def flat(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def csv_rows(relative: str) -> list[dict[str, str]]:
    with (RESULTS / relative).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def one(items: list[dict[str, str]], **where: str) -> dict[str, str]:
    matches = [row for row in items if all(row.get(key) == value for key, value in where.items())]
    assert len(matches) == 1, f"expected one row for {where}, found {len(matches)}"
    return matches[0]


def shown(value: str | float, digits: int) -> str:
    return f"{float(value):.{digits}f}"


def subsection(source: str, start: str, end: str | None = None) -> str:
    assert start in source, f"missing Supplement heading {start}"
    body = source.split(start, 1)[1]
    return body.split(end, 1)[0] if end and end in body else body


# ---------------------------------------------------------------------------
# Supplement navigation and release contract
# ---------------------------------------------------------------------------


def test_supplement_title_and_seven_section_navigation_are_locked() -> None:
    wrapper = flat(read(ROOT / "supplementary_v5.tex"))
    assert EXPECTED_TITLE in wrapper
    assert len(SI_PATHS) == 7
    assert [path.name for path in SI_PATHS] == [f"S{i}_{name}.tex" for i, name in enumerate(
        ["data", "spectra_nulls", "transport", "domain", "recovery", "experimental", "software"],
        start=1,
    )]

    inputs = re.findall(r"\\input\{(si_sections/S[1-7]_[^}]+\.tex)\}", wrapper)
    assert inputs == [f"si_sections/{path.name}" for path in SI_PATHS]
    for index, path in enumerate(SI_PATHS, start=1):
        source = read(path)
        assert source.count(r"\sisection{") == 1, path.name
        assert rf"\sisection{{S{index}." in source, path.name
        assert r"\section*{" not in source, path.name
        assert r"\subsection*{" not in source, path.name
        assert r"\subsection{" not in source, path.name


def test_supplement_documents_the_v5_build_and_pinned_xlrd() -> None:
    software = read(ROOT / "si_sections" / "S7_software.tex")
    for old_target in ("manuscript-v4", "supplement-v4"):
        assert old_target not in software
    assert "make PYTHON=.venv/bin/python all" in software
    for stage in ("regenerates", "stages", "graphical abstract", "checks", "compile"):
        assert stage in software.lower(), f"S7 does not explain that the default build {stage}"

    environment = read(ROOT / "environment.yml")
    assert re.search(r"^\s*-\s+xlrd=2\.0\.2\s*$", environment, flags=re.MULTILINE)
    assert re.search(r"xlrd\s*&\s*2\.0\.2", software)

    makefile = read(ROOT / "Makefile")
    assert re.search(r"^export SOURCE_DATE_EPOCH$", makefile, flags=re.MULTILINE)
    assert re.search(r"^release-check-v5:\s+all\s+full-test$", makefile, flags=re.MULTILINE)

    workflow = read(ROOT / ".github" / "workflows" / "v5-report.yml")
    for action in ("actions/checkout@v6", "actions/setup-python@v6", "actions/upload-artifact@v7"):
        assert action in workflow
    assert "make PYTHON=python release-check-v5" in workflow


def test_external_source_inventory_names_fixed_pose_toolchain_resources() -> None:
    with (ROOT / "external_sources.csv").open(newline="", encoding="utf-8") as handle:
        inventory = list(csv.DictReader(handle))
    assert inventory and all(None not in row for row in inventory)
    sources = {row["source"] for row in inventory}
    assert {
        "ODDT RF-Score v1-v3 resources (PDBbind 2016)",
        "DOCKSTRING pose archives",
        "DOCKSTRING receptor PDBQTs",
        "smina 2020.12.10 executable",
    } <= sources

    software = flat(read(ROOT / "si_sections" / "S7_software.tex"))
    assert "selected support and scorer seeds" in software
    assert "experimental-permutation seeds" in software
    assert "summary is authoritative for their versions and checksums" in software


def test_public_manifest_inventory_matches_the_saved_manifest() -> None:
    manifest = csv_rows("public_core_source_manifest.csv")
    counts: dict[str, int] = {}
    for row in manifest:
        counts[row["kind"]] = counts.get(row["kind"], 0) + 1
    assert len(manifest) == 149
    assert counts == {"input": 111, "code": 37, "output": 1}

    supplement = flat("\n".join(read(path) for path in SI_PATHS))
    assert re.search(r"149 (?:rows|records)", supplement)
    assert re.search(r"111 input(?:s| records)", supplement)
    assert re.search(r"37 code(?: files| records)", supplement)
    assert re.search(r"one output(?: file| record)?", supplement)


# ---------------------------------------------------------------------------
# Detailed numerical sensitivities
# ---------------------------------------------------------------------------


def test_pdsp_censoring_values_match_the_frozen_sensitivity_table() -> None:
    source = flat(read(ROOT / "si_sections" / "S6_experimental.tex"))
    text = subsection(source, "S6.4 PDSP support", "S6.5 SPD endpoints")
    metrics = csv_rows("pdsp_counterscreen_retrieval/censor_sensitivity_metrics.csv")

    # The readable prose reports residual censor sensitivities and points to the
    # CSV for the full raw/residual grid.  Bind every value it does state.
    contracts = [
        ("all_allowed_rows_right_censored_at_reported_bound", "residual_docking", 4),
        ("all_allowed_rows_quantified_vs_right_censored_binary", "residual_docking", 4),
        ("certified_right_censored_at_reported_bound", "residual_docking", 4),
        ("certified_quantified_vs_right_censored_binary", "residual_docking", 4),
    ]
    for sensitivity, predictor, digits in contracts:
        row = one(metrics, sensitivity=sensitivity, top_fraction="0.1", predictor=predictor)
        assert shown(row["continuous_spearman"], digits) in text, (sensitivity, predictor)

    for sensitivity in (
        "all_allowed_rows_right_censored_at_reported_bound",
        "all_allowed_rows_quantified_vs_right_censored_binary",
    ):
        row = one(metrics, sensitivity=sensitivity, top_fraction="0.1", predictor="residual_docking")
        assert shown(row["roc_auc"], 4) in text, (sensitivity, "residual_docking", "ROC-AUC")

    primary = one(
        csv_rows("pdsp_counterscreen_retrieval/global_retrieval_metrics.csv"),
        top_fraction="0.1",
        predictor="residual_docking",
    )
    restored = one(
        metrics,
        sensitivity="all_allowed_rows_right_censored_at_reported_bound",
        top_fraction="0.1",
        predictor="residual_docking",
    )
    assert shown(primary["continuous_spearman"], 4) in text
    assert shown(restored["continuous_spearman"], 4) in text


def test_pdsp_overlap_complete_case_and_qap_boundaries_are_frozen() -> None:
    import pandas as pd

    from analysis.residual_mie_boundary_audit import DOCK44

    source = flat(read(ROOT / "si_sections" / "S6_experimental.tex"))
    text = subsection(source, "S6.4 PDSP support", "S6.5 SPD endpoints")
    summary = json.loads(read(RESULTS / "pdsp_counterscreen_retrieval" / "summary.json"))
    audit = summary["pdsp_audit"]
    exclusions = csv_rows("pdsp_counterscreen_retrieval/excluded_docking44_ligand_ids.csv")
    assert len(exclusions) == audit["docking_compounds_excluded_for_pdsp_connectivity_overlap"]
    for value in (
        audit["exact_pdsp_docking_full_inchikey_overlap"],
        audit["exact_pdsp_docking_connectivity_block_overlap"],
        audit["docking_compounds_excluded_for_pdsp_connectivity_overlap"],
    ):
        assert str(value) in text

    complete = csv_rows("pdsp_counterscreen_retrieval/complete_case_support_metrics.csv")
    raw = one(complete, predictor="raw_docking")
    residual = one(complete, predictor="residual_docking")
    baseline = one(complete, predictor="equal_rank_sequence_family")
    fusion = one(complete, predictor="equal_rank_sequence_family_residual")
    for value in (
        raw["continuous_spearman"],
        residual["continuous_spearman"],
        float(fusion["continuous_spearman"]) - float(baseline["continuous_spearman"]),
    ):
        assert shown(value, 4) in text
    assert f"{int(float(raw['docking_rows'])):,}".replace(",", r"{,}") in text

    matrix = pd.read_csv(
        ROOT / "data" / "frozen" / "df_final_v4.csv.gz",
        usecols=["ligand_id", *DOCK44],
    )
    excluded = {int(row["ligand_id"]) for row in exclusions}
    kept = matrix.loc[~matrix["ligand_id"].isin(excluded)]
    missing = kept[list(DOCK44)].isna()
    deleted = missing.any(axis=1)
    assert int(deleted.sum()) == 3_302
    assert int((deleted & missing["4mqs"]).sum()) == 3_300
    assert "CHRM2" in text and "3{,}300" in text

    qap = csv_rows("pdsp_counterscreen_retrieval/paired_qap.csv")
    complete_qap = csv_rows("pdsp_counterscreen_retrieval/complete_case_support_qap.csv")
    assert len(qap) == len(complete_qap) == 12
    primary = one(
        qap,
        permutation_scheme="all_target_labels",
        contrast="residual_minus_raw_joint_label",
        metric="continuous_spearman",
    )
    within = one(
        qap,
        permutation_scheme="within_curated_family",
        contrast="residual_minus_raw_joint_label",
        metric="continuous_spearman",
    )
    for row in (primary,):
        assert shown(row["two_sided_qap_p"], 4) in text
        assert shown(row["holm_adjusted_two_sided_qap_p"], 4) in text
    whole_s6 = source
    assert shown(within["two_sided_qap_p"], 4) in whole_s6
    assert shown(within["holm_adjusted_two_sided_qap_p"], 4) in whole_s6
    assert "two-sided" in text and "post hoc" in text


def test_pdsp_minimum_pair_support_sweep_is_reported_without_dense_prose() -> None:
    source = flat(read(ROOT / "si_sections" / "S6_experimental.tex"))
    text = subsection(source, "S6.4 PDSP support", "S6.5 SPD endpoints")
    sweep = csv_rows("pdsp_counterscreen_retrieval/minimum_pair_support_sensitivity.csv")
    assert len(sweep) == 6
    for row in sweep:
        assert row["minimum_pair_support"] in text
        assert row["target_pairs"] in text
    assert "minimum_pair_support_sensitivity.csv" in text
    assert "do not propagate" in text or "does not propagate" in text


def test_spd_unrestricted_qap_probabilities_match_the_frozen_table() -> None:
    source = flat(read(ROOT / "si_sections" / "S6_experimental.tex"))
    text = subsection(source, "S6.5 SPD endpoints", "S6.6")
    qap = csv_rows("spd_external_validation/target_label_qap.csv")
    endpoints = (
        "floor_at_bound",
        "censor_aware_binary_10uM",
        "censor_aware_binary_30uM",
    )
    for endpoint in endpoints:
        residual = one(qap, permutation_scheme="unrestricted", metric="residual_spearman", endpoint=endpoint)
        increment = one(qap, permutation_scheme="unrestricted", metric="residual_minus_raw", endpoint=endpoint)
        assert shown(residual["p_positive"], 4) in text, (endpoint, "residual")
        assert shown(increment["p_positive"], 4) in text, (endpoint, "increment")
    assert "Agreement between DOCKSTRING target maps and SPD" in text
    assert "Agreement between Docking-44 target maps and SPD" not in text


def test_single_ligand_boundary_matches_observed_contrasts_and_null() -> None:
    source = flat(read(ROOT / "si_sections" / "S6_experimental.tex"))
    text = subsection(source, "S6.7", None)
    observed = csv_rows("residual_mie_boundary_audit/observed_metrics.csv")
    contrasts = csv_rows("residual_mie_boundary_audit/paired_contrasts.csv")
    nulls = csv_rows("residual_mie_boundary_audit/null_summary.csv")

    def observed_row(representation: str, weighting: str, metric: str) -> dict[str, str]:
        return one(observed, representation=representation, weighting=weighting, metric=metric)

    absolute_cb = observed_row("absolute", "compound_balanced", "top5")
    residual_cb = observed_row("two_way_residual", "compound_balanced", "top5")
    assert f"{100 * float(absolute_cb['estimate']):.1f}\\%" in text
    assert f"{100 * float(residual_cb['estimate']):.1f}\\%" in text

    for representation in ("absolute", "two_way_residual"):
        for metric in ("mean_rank_percentile", "top5"):
            row = observed_row(representation, "pair_weighted", metric)
            digits = 4 if metric == "mean_rank_percentile" else 3
            assert shown(row["estimate"], digits) in text, (representation, metric)

    mean_contrast = one(
        contrasts,
        comparator="two_way_residual",
        reference="absolute",
        weighting="compound_balanced",
        metric="mean_rank_percentile",
    )
    assert shown(mean_contrast["estimate"], 4) in text
    assert "Top-3 and top-5 retrieval both fall" in text
    assert "both cluster intervals exclude zero" in text

    for metric in ("mean_rank_percentile", "top5"):
        row = one(
            nulls,
            null_model="fixed_margin_bipartite_switch",
            representation="two_way_residual",
            weighting="compound_balanced",
            metric=metric,
        )
        assert shown(row["observed"], 4) in text
        assert shown(row["null_mean"], 4) in text
        assert shown(row["one_sided_empirical_p"], 4) in text


def test_pocket_result_and_figure_are_supplement_only() -> None:
    pocket_source = read(ROOT / "si_sections" / "S2_spectra_nulls.tex")
    main_source = read(ROOT / "manuscript_v5.tex")
    summary = json.loads(read(RESULTS / "residual_mechanism" / "analysis_summary.json"))
    pocket = summary["exploratory_pocket_volume"]

    assert main_source.count("fig_pocket_volume") == 0
    assert pocket_source.count(r"\includegraphics{figures/v5/fig_pocket_volume.pdf}") == 1
    assert shown(pocket["pocket_volume_vs_residual_mw_slope_spearman_rho"], 3) in pocket_source
    assert shown(pocket["within_family_permutation_two_sided_p"], 3) in pocket_source
    low, high = pocket["target_bootstrap_95_interval"]
    assert shown(low, 3) in pocket_source and shown(high, 3) in pocket_source
    assert str(pocket["n_targets_with_complete_pocket_volume"]) in pocket_source
