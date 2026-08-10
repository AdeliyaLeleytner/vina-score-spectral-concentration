from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

import dockstring_chembl_ranking_benchmark as benchmark
import make_reported_results as reported


def test_gene_target_contract_is_a_bijection_onto_dockstring_columns() -> None:
    """Every ChEMBL target must resolve to exactly one DOCKSTRING gene symbol."""
    gene_to_chembl, chembl_to_gene = benchmark.gene_target_contract()
    assert len(gene_to_chembl) == len(chembl_to_gene) == 31
    assert set(chembl_to_gene.values()) == set(gene_to_chembl)
    for gene, chembl_id in gene_to_chembl.items():
        assert chembl_to_gene[chembl_id] == gene


def test_conservative_union_takes_the_widest_cluster_interval() -> None:
    comparison = {
        "plugin_mean_difference": 0.01,
        "scaffold_cluster_bootstrap": {
            "interval_90": [-0.03, 0.05],
            "interval_95": [-0.04, 0.06],
        },
        "butina_cluster_bootstrap": {
            "interval_90": [-0.04, 0.02],
            "interval_95": [-0.05, 0.03],
        },
    }
    interval_90, sources = benchmark.conservative_union(comparison, "interval_90")
    assert sources == ["scaffold_cluster_bootstrap", "butina_cluster_bootstrap"]
    assert interval_90 == [-0.04, 0.05]

    report = benchmark.contrast_equivalence_report(comparison)
    assert report["conservative_cluster_bootstrap_interval_95"] == [-0.05, 0.06]
    assert report["conservative_interval_width_90"] == pytest.approx(0.09)
    # The smallest symmetric margin the 90% interval supports is the larger of
    # the two absolute endpoints, here |-0.04| versus |0.05|.
    assert report[
        "smallest_supported_symmetric_equivalence_margin_90"
    ] == pytest.approx(0.05)
    assert report["margin_decisions_90"] == {
        "0.01": False,
        "0.02": False,
        "0.05": False,
        "0.10": True,
    }
    assert report["excludes_zero_90"] is False


def test_cohort_prior_holds_out_the_whole_scaffold_cluster() -> None:
    """The cohort prior must never see any ligand sharing the held-out scaffold."""
    experiment = pd.DataFrame(
        {
            "A": [7.0, 9.0, 5.0, np.nan],
            "B": [6.0, 8.0, np.nan, 4.0],
        },
        index=["l1", "l2", "l3", "l4"],
    )
    scaffolds = np.asarray(["s1", "s1", "s2", "s3"], dtype=object)
    global_prior = np.asarray([-1.0, -2.0])
    prior = benchmark.cohort_scaffold_prior(experiment, scaffolds, global_prior)

    # Ligands 1 and 2 share scaffold s1, so both are held out for either of them:
    # only l3 (A=5) and l4 (B=4) remain.
    np.testing.assert_allclose(prior[0], [-5.0, -4.0])
    np.testing.assert_allclose(prior[1], [-5.0, -4.0])
    # For l3 the remaining cohort is l1, l2 and l4.
    np.testing.assert_allclose(prior[2], [-8.0, -6.0])
    # Sign convention: the prior is negated pChEMBL so that lower is better,
    # matching the docking-score direction.
    assert (prior <= 0).all()


def test_cohort_prior_falls_back_to_the_external_prior_when_a_target_empties() -> None:
    experiment = pd.DataFrame(
        {"A": [7.0, np.nan], "B": [6.0, 5.0]},
        index=["l1", "l2"],
    )
    scaffolds = np.asarray(["s1", "s2"], dtype=object)
    global_prior = np.asarray([-1.25, -2.5])
    prior = benchmark.cohort_scaffold_prior(experiment, scaffolds, global_prior)
    # Holding out l1 leaves no observation of target A, so the external prior is used.
    np.testing.assert_allclose(prior[0], [-1.25, -5.0])
    np.testing.assert_allclose(prior[1], [-7.0, -6.0])


def test_build_support_applies_the_two_target_cohort_filter() -> None:
    activities = pd.DataFrame(
        {
            "inchikey": ["k1", "k1", "k2", "k3", "k3", "k4"],
            "gene": ["ABL1", "SRC", "ABL1", "ABL1", "SRC", "MET"],
            "pchembl_value": [7.0, 6.0, 8.0, 5.0, 9.0, 6.5],
        }
    )
    docking = pd.DataFrame(
        np.zeros((3, 3)),
        index=pd.Index(["k1", "k2", "k3"], name="inchikey"),
        columns=["ABL1", "SRC", "MET"],
    )
    reference, experiment, report = benchmark.build_support(
        activities, docking, ["ABL1", "SRC", "MET"]
    )
    # k4 has no docking row; k2 has only one observed target; MET is unobserved
    # in the surviving cohort and is therefore dropped from the evaluation panel.
    assert list(experiment.index) == ["k1", "k3"]
    assert list(experiment.columns) == ["ABL1", "SRC"]
    assert report["chembl_ligands_matched_to_dockstring"] == 3
    assert report["ligands_by_minimum_observed_targets"]["2"] == 2
    assert report["targets_dropped_for_no_observation_in_cohort"] == ["MET"]
    assert reference.loc["k4", "MET"] == pytest.approx(6.5)


def test_kikd_endpoint_mixing_diagnostic_keeps_dual_type_cells_separate() -> None:
    activities = pd.DataFrame(
        {
            "inchikey": [
                "l1", "l1", "l1",  # one Ki--Ki and two Ki--Kd pairs
                "l2", "l2", "l2",  # A pools Ki and Kd; A--B is separate
                "l3", "l3",        # exact median tie, excluded
            ],
            "gene": ["A", "B", "C", "A", "A", "B", "A", "B"],
            "pchembl_value": [7.0, 6.0, 5.0, 6.0, 8.0, 5.0, 6.0, 6.0],
            "standard_type": ["Ki", "Ki", "Kd", "Ki", "Kd", "Kd", "Ki", "Kd"],
        }
    )
    docking = pd.DataFrame(
        np.zeros((3, 3)), index=["l1", "l2", "l3"], columns=["A", "B", "C"]
    )
    result = benchmark.kikd_endpoint_mixing_diagnostic(
        activities, docking, ["A", "B", "C"]
    )

    assert result["informative_non_tied_pairs"] == 4
    assert result["excluded_exact_median_ties"] == 1
    assert result["endpoint_unambiguous_pairs"] == 3
    assert result["pair_counts"] == {
        "same_endpoint": 1,
        "same_endpoint_Ki_Ki": 1,
        "same_endpoint_Kd_Kd": 0,
        "mixed_endpoint_Ki_Kd": 2,
        "involves_pooled_Ki_Kd_cell": 1,
    }
    assert result["fractions_of_all_informative_pairs"] == pytest.approx(
        {
            "same_endpoint": 0.25,
            "mixed_endpoint_Ki_Kd": 0.50,
            "involves_pooled_Ki_Kd_cell": 0.25,
        }
    )
    assert result["fractions_among_endpoint_unambiguous_pairs"] == pytest.approx(
        {"same_endpoint": 1 / 3, "mixed_endpoint_Ki_Kd": 2 / 3}
    )
    assert result["observed_cell_endpoint_provenance"] == {
        "Ki_only": 3,
        "Kd_only": 3,
        "pooled_Ki_and_Kd": 1,
    }
    assert "not assigned" in result["cell_aggregation_contract"]


def test_released_summary_contract() -> None:
    """Assertions about the released artifact itself."""
    path = benchmark.DEFAULT_OUTPUT / "summary.json"
    if not path.exists():
        pytest.skip("the DOCKSTRING ChEMBL ranking benchmark has not been generated")
    summary = json.loads(path.read_text())

    assert summary["schema_version"] == "1.1.0"
    assert summary["status"] == "complete"
    assert summary["benchmark_id"] == "broad_dockstring_chembl_primary"
    assert summary["role"] == "primary_broad_support_observed_pair_benchmark"
    assert summary["seed"] == benchmark.DEFAULT_SEED
    assert summary["bootstrap"]["repeats"] == 5000
    assert summary["bootstrap"]["permutation_repeats"] == 5000
    assert "claim_boundary" in summary and len(summary["claim_boundary"]) > 200

    support = summary["support"]
    # The whole point of the rebuild: a DOCKSTRING-sized ligand cohort.
    assert support["docking_reference_ligands"] > 250_000
    assert support["ligands"] == support["retained_ligands"] == 6557
    assert support["evaluated_ligands"] == 6480
    assert support["retained_ligands_with_no_informative_non_tied_pair"] == 77
    assert support["targets"] == 31
    coverage = summary["matching"]["ligands_by_minimum_observed_targets"]
    assert support["ligands"] == coverage["2"]
    # Ligands with exactly one observed target contribute exactly one cell each,
    # so dropping them must remove that many cells and nothing else.
    assert support["observed_experimental_cells"] == (
        summary["matching"]["matched_observed_cells"] - (coverage["1"] - coverage["2"])
    )
    assert len(support["target_ids"]) == support["targets"]

    published = summary["published_docking44_reference"]
    assert published["benchmark_id"] == "sparse_docking44_chembl_sensitivity"
    assert published["role"] == "support_sensitivity_only"
    assert published["support"]["ligands"] == 137
    assert published["support"]["non_tied_within_ligand_pairs"] == 2522
    scale = summary["scale_comparison_with_published_docking44_benchmark"]
    assert scale["ligand_multiplier"] > 20.0
    assert scale["non_tied_pair_multiplier"] > 2.0

    # Larger support must not widen the uncertainty on the headline contrasts.
    for name in benchmark.HEADLINE_CONTRASTS:
        contrast = summary["headline_contrasts"][name]
        low_90, high_90 = contrast["conservative_cluster_bootstrap_interval_90"]
        low_95, high_95 = contrast["conservative_cluster_bootstrap_interval_95"]
        assert low_95 <= low_90 <= high_90 <= high_95
        assert low_90 <= contrast["plugin_mean_difference"] <= high_90
        assert contrast["conservative_interval_width_90"] == pytest.approx(
            high_90 - low_90
        )
        assert contrast[
            "smallest_supported_symmetric_equivalence_margin_90"
        ] == pytest.approx(max(abs(low_90), abs(high_90)))
        assert (
            contrast["conservative_interval_width_95"]
            < published["contrasts"][name]["conservative_interval_width_95"]
        )

    accuracy = summary["mean_per_ligand_pairwise_accuracy"]
    assert set(accuracy) == set(benchmark.REPRESENTATION_ORDER)
    # target_centered_unscaled and two_way_centered_unscaled differ only by a
    # within-ligand constant, so they must induce identical target rankings.
    assert accuracy["two_way_centered_unscaled"] == pytest.approx(
        accuracy["target_centered_unscaled"]
    )
    for value in accuracy.values():
        assert 0.0 <= value <= 1.0

    absolute_minus_prior = summary["benchmark"]["paired_comparisons"][
        "absolute_vina_minus_docking_target_prior"
    ]
    assert absolute_minus_prior["plugin_mean_difference"] == pytest.approx(
        accuracy["absolute_vina"] - accuracy["docking_target_prior"]
    )
    assert absolute_minus_prior["plugin_mean_difference"] == pytest.approx(
        0.01342231729813158
    )
    chemical_95 = reported.interval_union(
        absolute_minus_prior,
        ("scaffold_cluster_bootstrap", "butina_cluster_bootstrap"),
    )
    assert chemical_95 == pytest.approx(
        [-0.01130446556950808, 0.03802962307197403]
    )
    assert chemical_95[0] < 0 < chemical_95[1]

    ordering = summary["ranking_order"]
    assert ordering["best_tested_docking_score_representation"] == "absolute_vina"
    assert ordering["best_overall_representation"] == (
        "cohort_experimental_target_prior"
    )
    assert ordering[
        "absolute_vina_exceeds_cohort_experimental_target_prior"
    ] is False

    layers = summary["uncertainty_layers"]
    assert set(layers) == {
        "chemical_support_sampling",
        "target_panel_composition",
        "target_label_qap",
        "transformation_matched_mechanical_null",
    }
    for name in benchmark.HEADLINE_CONTRASTS:
        assert layers["chemical_support_sampling"][name][
            "conservative_union_interval_95"
        ] == summary["headline_contrasts"][name][
            "conservative_cluster_bootstrap_interval_95"
        ]
        assert layers["target_panel_composition"][name][
            "jackknife_normal_95_interval"
        ] == summary["benchmark"]["target_jackknife_paired_contrasts"][name][
            "jackknife_normal_95_interval"
        ]

    # Assay heterogeneity is exposed rather than hidden inside the coverage-primary
    # all-endpoint cell aggregation.  The endpoint-restricted arm has both chemical-
    # support and target-panel uncertainty; single-endpoint arms remain sensitivities.
    clean = summary["sensitivities"]["human_binding_Ki_Kd"]
    assert clean["support"]["retained_ligands"] == 1056
    assert clean["support"]["evaluated_ligands"] == 1049
    assert clean["support"]["targets"] == 31
    assert clean["support"]["observed_cells"] == 2783
    assert clean["support"]["non_tied_within_ligand_pairs"] == 3955
    clean_contrast = clean["contrasts"][
        "two_way_residual_minus_absolute_vina"
    ]
    assert clean_contrast["plugin_mean_difference"] == pytest.approx(
        clean["mean_per_ligand_pairwise_accuracy"]["two_way_residual"]
        - clean["mean_per_ligand_pairwise_accuracy"]["absolute_vina"]
    )
    assert clean_contrast["plugin_mean_difference"] < 0.0
    clean_chemical_95 = clean_contrast[
        "conservative_cluster_bootstrap_interval_95"
    ]
    clean_target_95 = clean["target_jackknife_paired_contrasts"][
        "two_way_residual_minus_absolute_vina"
    ]["jackknife_normal_95_interval"]
    assert clean_chemical_95[0] < clean_contrast["plugin_mean_difference"] < clean_chemical_95[1]
    assert clean_target_95[0] < clean_contrast["plugin_mean_difference"] < clean_target_95[1]
    assert clean_chemical_95[1] > 0.0
    assert clean_target_95[1] > 0.0
    mixing = clean["endpoint_mixing_diagnostic"]
    assert mixing["informative_non_tied_pairs"] == clean["support"][
        "non_tied_within_ligand_pairs"
    ]
    pair_counts = mixing["pair_counts"]
    assert mixing["excluded_exact_median_ties"] == 25
    assert mixing["endpoint_unambiguous_pairs"] == 3643
    assert pair_counts == {
        "same_endpoint": 3610,
        "same_endpoint_Ki_Ki": 1358,
        "same_endpoint_Kd_Kd": 2252,
        "mixed_endpoint_Ki_Kd": 33,
        "involves_pooled_Ki_Kd_cell": 312,
    }
    assert mixing["observed_cell_endpoint_provenance"] == {
        "Ki_only": 2008,
        "Kd_only": 704,
        "pooled_Ki_and_Kd": 71,
    }
    assert mixing["informative_non_tied_pairs"] == (
        pair_counts["same_endpoint"]
        + pair_counts["mixed_endpoint_Ki_Kd"]
        + pair_counts["involves_pooled_Ki_Kd_cell"]
    )
    assert pair_counts["same_endpoint"] == (
        pair_counts["same_endpoint_Ki_Ki"]
        + pair_counts["same_endpoint_Kd_Kd"]
    )
    assert sum(mixing["fractions_of_all_informative_pairs"].values()) == pytest.approx(1.0)
    assert sum(
        mixing["fractions_among_endpoint_unambiguous_pairs"].values()
    ) == pytest.approx(1.0)

    endpoints = summary["sensitivities"]["single_endpoint"]
    assert set(endpoints) == set(benchmark.ENDPOINTS)
    for endpoint, record in endpoints.items():
        assert record["reported_scope"].startswith("docking-derived representations")
        assert record["support"]["retained_ligands"] >= record["support"][
            "evaluated_ligands"
        ]
        contrast = record["contrasts"][
            "two_way_residual_minus_absolute_vina"
        ]
        assert contrast["plugin_mean_difference"] == pytest.approx(
            record["mean_per_ligand_pairwise_accuracy"]["two_way_residual"]
            - record["mean_per_ligand_pairwise_accuracy"]["absolute_vina"]
        )
    # The small Kd arm reverses the Ki direction.  This is a scientific result to
    # disclose, not average away into a claim of one universally best representation.
    assert endpoints["Ki"]["contrasts"][
        "two_way_residual_minus_absolute_vina"
    ]["plugin_mean_difference"] < 0.0
    assert endpoints["Kd"]["support"]["evaluated_ligands"] == 160
    assert endpoints["Kd"]["contrasts"][
        "two_way_residual_minus_absolute_vina"
    ]["plugin_mean_difference"] > 0.0


def test_released_tables_match_the_summary() -> None:
    path = benchmark.DEFAULT_OUTPUT / "summary.json"
    if not path.exists():
        pytest.skip("the DOCKSTRING ChEMBL ranking benchmark has not been generated")
    summary = json.loads(path.read_text())

    ligands = pd.read_csv(benchmark.DEFAULT_OUTPUT / "per_ligand_metrics.csv")
    assert len(ligands) == summary["support"]["ligands"]
    assert ligands.inchikey.is_unique
    assert (ligands.observed_targets >= 2).all()
    assert int(ligands.nontied_target_pairs.sum()) == (
        summary["support"]["non_tied_within_ligand_target_pairs"]
    )
    assert ligands.butina_cluster.nunique() == summary["support"]["butina_clusters"]
    assert ligands.murcko_scaffold.nunique() == (
        summary["support"]["murcko_scaffold_clusters"]
    )

    representations = pd.read_csv(
        benchmark.DEFAULT_OUTPUT / "representation_summary.csv"
    ).set_index("representation")
    for name, value in summary["mean_per_ligand_pairwise_accuracy"].items():
        assert representations.loc[
            name, "mean_per_ligand_pairwise_accuracy"
        ] == pytest.approx(value)

    contrasts = pd.read_csv(
        benchmark.DEFAULT_OUTPUT / "contrast_summary.csv"
    ).set_index("contrast")
    for name in benchmark.HEADLINE_CONTRASTS:
        assert contrasts.loc[name, "conservative_width_90"] == pytest.approx(
            summary["headline_contrasts"][name]["conservative_interval_width_90"]
        )
    targets = pd.read_csv(benchmark.DEFAULT_OUTPUT / "target_support.csv")
    assert list(targets.target) == summary["support"]["target_ids"]
    assert int(targets.ligands_observed.sum()) == (
        summary["support"]["observed_experimental_cells"]
    )
    # Docking scores are clipped at zero, so every mean must be non-positive.
    assert (targets.mean_dockstring_score_full_reference <= 0).all()

    endpoint_table = pd.read_csv(
        benchmark.DEFAULT_OUTPUT / "endpoint_sensitivity_summary.csv"
    ).set_index("analysis_arm")
    assert list(endpoint_table.index) == [
        "all_exact_relations_all_endpoints",
        "human_binding_Ki_Kd",
        "single_endpoint_Ki",
        "single_endpoint_Kd",
        "single_endpoint_IC50",
        "single_endpoint_EC50",
    ]
    assert endpoint_table.loc[
        "all_exact_relations_all_endpoints", "residual_minus_absolute"
    ] == pytest.approx(
        summary["headline_contrasts"][
            "two_way_residual_minus_absolute_vina"
        ]["plugin_mean_difference"]
    )
    assert endpoint_table.loc[
        "human_binding_Ki_Kd", "target_jackknife_95_low"
    ] == pytest.approx(
        summary["sensitivities"]["human_binding_Ki_Kd"][
            "target_jackknife_paired_contrasts"
        ]["two_way_residual_minus_absolute_vina"][
            "jackknife_normal_95_interval"
        ][0]
    )
    mixing = summary["sensitivities"]["human_binding_Ki_Kd"][
        "endpoint_mixing_diagnostic"
    ]
    assert endpoint_table.loc[
        "human_binding_Ki_Kd", "same_endpoint_pairs"
    ] == pytest.approx(mixing["pair_counts"]["same_endpoint"])
    assert endpoint_table.loc[
        "human_binding_Ki_Kd", "mixed_Ki_Kd_fraction_of_all_pairs"
    ] == pytest.approx(
        mixing["fractions_of_all_informative_pairs"]["mixed_endpoint_Ki_Kd"]
    )


def test_broad_absolute_minus_prior_is_exposed_in_generated_reports() -> None:
    registry_path = benchmark.PACKAGE / "results" / "reported_values.json"
    table_path = benchmark.PACKAGE / "results" / "supplement_tables.tex"
    if not registry_path.exists() or not table_path.exists():
        pytest.skip("generated manuscript reports have not been built")
    macros = json.loads(registry_path.read_text())["macros"]
    assert macros["BroadRankingAbsoluteMinusDockingPrior"] == "0.013"
    assert macros["BroadRankingAbsoluteMinusDockingPriorLow"] == "-0.011"
    assert macros["BroadRankingAbsoluteMinusDockingPriorHigh"] == "0.038"
    table = table_path.read_text()
    assert (
        "paired: absolute Vina minus docking target prior & +0.013 & -- & -- & "
        "6,480 & 25,214 & -0.011~--~0.038"
    ) in table


def test_reported_macros_cannot_mix_broad_and_sparse_benchmarks() -> None:
    """A ``Broad`` macro must never be populated from the 137-ligand object."""
    manuscript_path = benchmark.PACKAGE / "results" / "manuscript_evidence.json"
    if not manuscript_path.exists():
        pytest.skip("manuscript evidence has not been generated")
    manuscript = json.loads(manuscript_path.read_text())
    broad = manuscript["dockstring_chembl_ranking"]
    sparse = manuscript["ligand_wise_ranking_boundary"]
    _, _, registry = reported.build(manuscript)
    macros = registry["macros"]

    assert float(macros["BroadDockingPriorAccuracy"]) == pytest.approx(
        broad["mean_per_ligand_pairwise_accuracy"]["docking_target_prior"],
        abs=5e-4,
    )
    assert float(macros["BroadExternalPriorAccuracy"]) == pytest.approx(
        broad["mean_per_ligand_pairwise_accuracy"]["experimental_target_prior"],
        abs=5e-4,
    )
    assert float(macros["BroadCohortPriorAccuracy"]) == pytest.approx(
        broad["mean_per_ligand_pairwise_accuracy"][
            "cohort_experimental_target_prior"
        ],
        abs=5e-4,
    )
    assert float(macros["SparseRankingDockingPriorAccuracy"]) == pytest.approx(
        sparse["representations"]["docking_target_prior"][
            "mean_per_ligand_pairwise_accuracy"
        ],
        abs=5e-4,
    )
    assert macros["BroadRankingRetainedLigands"] == "6,557"
    assert macros["BroadRankingEvaluatedLigands"] == "6,480"
    assert macros["SparseRankingRetainedLigands"] == "137"
    assert macros["BroadHumanKiKdRetainedLigands"] == "1,056"
    assert float(macros["BroadHumanKiKdDelta"]) == pytest.approx(
        broad["sensitivities"]["human_binding_Ki_Kd"]["contrasts"][
            "two_way_residual_minus_absolute_vina"
        ]["plugin_mean_difference"],
        abs=5e-4,
    )
    for ambiguous in (
        "CohortPriorLow",
        "PairWeightedAbsoluteAccuracy",
        "AbsoluteIdentityShuffleP",
        "CoverageThreeDockingPriorAccuracy",
        "BenchmarkAccessibleBand",
        "BroadRankingAccessibleBand",
        "SparseRankingAccessibleBand",
    ):
        assert ambiguous not in macros
