"""Focused contract tests for the public-only manuscript figure producer."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

import make_public_manuscript_figures as figures


FORBIDDEN_SOURCE_MARKERS = (
    "matched_support_geometry",
    "residual_target_geometry_validation",
    "PKIS1",
    "KiRHub",
    "locked_endpoint",
    "locked_ledger",
    "manuscript_evidence.json",
)

EXPECTED_INPUT_SUFFIXES = {
    "results/public_core_evidence.json",
    "data/frozen/df_final_v4.csv.gz",
    "data/frozen/dockstring-dataset.tsv.gz",
    "results/public_residual_null_audit/summary.json",
    "results/revision_map_decision_sensitivities/mw_threshold_sweep.csv",
    "results/revision_map_decision_sensitivities/mw_decile_pair_distances.csv",
    "results/revision_map_decision_sensitivities/mw_decile_continuous_trends.csv",
    "results/public_chemical_domain_controls/causal_boundary_summary.csv",
    "results/public_chemical_domain_controls/mw_matched_group_disjoint_summary.csv",
    "results/public_chemical_domain_controls/within_band_reproducibility_summary.csv",
    "results/public_chemical_domain_controls/dockstring_support_seed_contrasts.csv",
    "results/public_descriptor_domain_specificity/descriptor_extremes.csv",
    "results/public_descriptor_domain_specificity/descriptor_random_disjoint_summary.csv",
    "results/revision_map_decision_sensitivities/recovery_edge_summary.csv",
    "results/revision_map_decision_sensitivities/recovery_decision_summary.csv",
    "results/public_panel_selector_controls/panel_metrics.csv.gz",
    "results/public_panel_selector_robustness/direct_residual_summary.csv",
    "results/experimental_map_reliability/cross_panel_geometry.csv",
    "results/experimental_map_reliability/map_reliability.csv",
    "results/experimental_map_reliability/same_support_geometry.csv",
    "results/experimental_map_reliability/same_support_bootstrap.csv",
    "results/experimental_map_reliability/same_support_split_half.csv",
    "results/public_anastassiadis_panel_validation/split_half_reliability.csv",
    "results/public_anastassiadis_panel_validation/same_endpoint_panel_transfer.csv",
    "results/public_anastassiadis_panel_validation/map_concordance.csv",
}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_source_and_input_boundary_is_public_only() -> None:
    source = Path(figures.__file__).read_text(encoding="utf-8")
    for marker in FORBIDDEN_SOURCE_MARKERS:
        assert marker not in source
    observed = {
        str(path.resolve().relative_to(figures.PACKAGE.resolve()))
        for path in figures.INPUT_PATHS.values()
    }
    assert observed == EXPECTED_INPUT_SUFFIXES


def test_public_evidence_anchors_are_locked() -> None:
    data = figures.load_figure_inputs()
    core = data["core"]
    assert core["analysis_status"] == "submission_public_core"
    assert core["large_vina_panels"]["Docking-44"]["support"]["analysis_rows"] == 12_651
    assert core["large_vina_panels"]["DOCKSTRING-58"]["support"]["analysis_rows"] == 260_060

    nulls = data["nulls"]["datasets"]
    assert nulls["docking44"]["observed_residual_pr"] == pytest.approx(9.294969781550748)
    assert nulls["dockstring58"]["observed_residual_pr"] == pytest.approx(18.15792807754205)

    sweep = data["mw_sweep"]
    assert len(sweep) == 28
    assert sweep["observed_below_control_q025"].astype(bool).all()

    boundary = data["chemical_domain_boundary"].loc[
        data["chemical_domain_boundary"]["transformation"]
        == "row_centered_residual"
    ].set_index("dataset")
    assert boundary.loc[
        "Docking-44", "observed_low_high_geometry_spearman"
    ] == pytest.approx(0.20496480945389367)
    assert boundary.loc[
        "DOCKSTRING-58", "observed_low_high_geometry_spearman"
    ] == pytest.approx(0.3505978420709316)
    global_controls = data["chemical_domain_global_controls"].loc[
        data["chemical_domain_global_controls"]["transformation"]
        == "row_centered_residual"
    ].set_index("dataset")
    assert global_controls.loc[
        "Docking-44", "geometry_spearman_median"
    ] == pytest.approx(0.9917015350968347)
    assert global_controls.loc[
        "DOCKSTRING-58", "geometry_spearman_median"
    ] == pytest.approx(0.9944818581697781)
    within_controls = data["chemical_domain_within_controls"].loc[
        (data["chemical_domain_within_controls"]["transformation"]
         == "row_centered_residual")
        & (data["chemical_domain_within_controls"]["control_type"]
           == "mw_matched_chemical_group_disjoint")
    ]
    assert len(within_controls) == 4
    support_seeds = data["chemical_domain_support_seeds"]
    assert support_seeds["support_seed"].nunique() == 6
    assert support_seeds.groupby("support_seed").size().eq(2).all()
    descriptors = data["descriptor_domain_extremes"].loc[
        data["descriptor_domain_extremes"]["transformation"]
        == "row_centered_residual"
    ].pivot(index="descriptor", columns="dataset", values="geometry_spearman")
    assert descriptors.shape == (7, 2)
    assert descriptors.loc["labute_asa", "Docking-44"] == pytest.approx(
        0.200058200573905
    )
    assert descriptors.loc["clogp", "Docking-44"] == pytest.approx(
        0.8872465653635073
    )
    assert descriptors.loc["labute_asa", "DOCKSTRING-58"] == pytest.approx(
        0.32203190302999185
    )
    assert descriptors.loc["rotatable_bonds", "DOCKSTRING-58"] == pytest.approx(
        0.8314147147079953
    )
    descriptor_controls = data["descriptor_domain_random_controls"]
    assert len(descriptor_controls) == 14
    assert descriptor_controls["observed_below_control_q025"].astype(bool).all()
    assert descriptor_controls.loc[
        (descriptor_controls["dataset"] == "DOCKSTRING-58")
        & (descriptor_controls["descriptor"] == "ring_count"),
        "combined_support_fraction",
    ].iloc[0] == pytest.approx(1.0)

    edges, decisions = figures.primary_recovery_tables(
        data["recovery_edges"], data["recovery_decisions"]
    )
    dockstring_500 = edges.loc[
        (edges["dataset"] == "DOCKSTRING-58")
        & (edges["calibration_ligands"] == 500)
    ].iloc[0]
    assert dockstring_500["geometry_spearman_mean"] == pytest.approx(0.965432942441319)
    assert set(decisions["panel_targets_k"]) == {8}

    reconstruction = data["panel_selector_metrics"].loc[
        data["panel_selector_metrics"]["pilot_ligands"].eq(500)
        & data["panel_selector_metrics"]["panel_k"].eq(8)
        & data["panel_selector_metrics"]["selector"].eq(
            "pilot_residual_r2_exact_kmedoids"
        )
        & data["panel_selector_metrics"]["predictor"].eq(
            "multivariate_ridge_gcv"
        )
    ]
    medians = reconstruction.groupby(["dataset", "surface"])[
        "omitted_only_variance_weighted_r2"
    ].median()
    assert medians.loc[("Docking-44", "raw")] == pytest.approx(
        0.900474, abs=1e-6
    )
    assert medians.loc[
        ("Docking-44", "raw_row_centered_residual")
    ] == pytest.approx(0.277939, abs=1e-6)
    assert medians.loc[("DOCKSTRING-58", "raw")] == pytest.approx(
        0.697956, abs=1e-6
    )
    assert medians.loc[
        ("DOCKSTRING-58", "raw_row_centered_residual")
    ] == pytest.approx(0.219211, abs=1e-6)

    cross = data["experimental_cross_panel"].loc[
        data["experimental_cross_panel"]["transform"] == "two_way_centered"
    ].iloc[0]
    assert cross["edge_spearman"] == pytest.approx(0.8484034299449614)
    same = data["same_support_geometry"].loc[
        (data["same_support_geometry"]["support"] == "informative_exact_matches_primary")
        & (data["same_support_geometry"]["transform"] == "two_way_centered")
    ].set_index("panel")
    assert same.loc["DAVIS", "same_support_empirical_spearman"] == pytest.approx(
        0.30656542260433506
    )
    assert same.loc["PKIS2", "same_support_empirical_spearman"] == pytest.approx(
        0.3260825480331591
    )
    hotspot_splits = data["anastassiadis_splits"]
    assert len(hotspot_splits) == 2_000
    assert hotspot_splits["cross_half_target_map_spearman"].median() == pytest.approx(
        0.7938631585426297
    )
    transfer = data["experimental_panel_transfer"].loc[
        data["experimental_panel_transfer"]["experimental_endpoint"].eq(
            "metric_row_centered_residual"
        )
        & data["experimental_panel_transfer"]["objective"].eq(
            "signed_1_minus_r"
        )
    ]
    assert len(transfer) == 15
    conservative = transfer[
        "all_optimal_pair_coverage_loss_improvement_min_conservative"
    ]
    assert int((conservative > 0).sum()) == 14
    assert (
        transfer.loc[
            transfer["evaluation_map"].eq("Anastassiadis"),
            "all_optimal_pair_coverage_loss_improvement_min_conservative",
        ]
        > 0
    ).all()
    concordance = data["anastassiadis_map_concordance"].loc[
        data["anastassiadis_map_concordance"][
            "anastassiadis_transformation"
        ].eq("row_centered_correlation")
    ].set_index("comparison_panel")
    assert concordance.loc["DAVIS", "target_pair_spearman"] == pytest.approx(
        0.76506626324536
    )
    assert concordance.loc["PKIS2", "target_pair_spearman"] == pytest.approx(
        0.751496794549925
    )


@pytest.fixture(scope="module")
def rendered(tmp_path_factory: pytest.TempPathFactory) -> Path:
    output = tmp_path_factory.mktemp("public-figures")
    paths = figures.make_all_figures(output)
    assert len(paths) == 9
    return output


def test_all_vector_and_raster_outputs_are_valid(rendered: Path) -> None:
    expected = {
        *(f"{stem}.pdf" for stem in figures.FIGURE_STEMS),
        *(f"{stem}.png" for stem in figures.FIGURE_STEMS),
        "figure_metadata.json",
    }
    assert {path.name for path in rendered.iterdir()} == expected
    for stem in figures.FIGURE_STEMS:
        pdf = rendered / f"{stem}.pdf"
        png = rendered / f"{stem}.png"
        assert pdf.read_bytes().startswith(b"%PDF")
        assert pdf.stat().st_size > 15_000
        with Image.open(png) as image:
            assert image.format == "PNG"
            assert image.width >= 1_800
            assert image.height >= 1_300
            dpi = image.info.get("dpi")
            assert dpi is not None
            assert np.allclose(dpi, (300, 300), atol=1.0)


def test_metadata_records_exact_inputs_outputs_and_boundaries(rendered: Path) -> None:
    metadata = json.loads((rendered / "figure_metadata.json").read_text())
    assert metadata["schema_version"] == "1.0"
    assert "fixed target panels" in metadata["scope"]
    assert "not target-superpopulation confidence intervals" in metadata[
        "uncertainty_contract"
    ]
    assert set(metadata["inputs"]) == set(figures.INPUT_PATHS)
    assert set(metadata["captions"]) == set(figures.FIGURE_STEMS)
    expected_panels = {
        "fig1_spectra_residual": "abcde",
        "fig2_support_domain": "abcd",
        "fig3_pilot_decision": "abcd",
        "fig4_experimental_boundary": "abcd",
    }
    for stem, labels in expected_panels.items():
        caption = metadata["captions"][stem]
        assert all(f"({label})" in caption for label in labels)
    assert len(metadata["outputs"]) == 8
    for relative, record in metadata["outputs"].items():
        path = figures.PACKAGE / relative
        if not path.is_file():
            path = rendered / Path(relative).name
        assert path.stat().st_size == record["bytes"]
        assert sha256_file(path) == record["sha256"]


def test_primary_split_contract_is_cluster_disjoint() -> None:
    data = figures.load_figure_inputs()
    full, matched = figures.primary_experimental_splits(
        data["experimental_reliability"], data["same_support_splits"]
    )
    assert set(full["split_unit"]) == {"butina_cluster"}
    assert set(matched["split_unit"]) == {"butina_cluster"}
    assert full.groupby("panel").size().to_dict() == {"DAVIS": 2_000, "PKIS2": 2_000}
    assert matched.groupby("panel").size().to_dict() == {
        "DAVIS": 2_000,
        "PKIS2": 2_000,
    }
