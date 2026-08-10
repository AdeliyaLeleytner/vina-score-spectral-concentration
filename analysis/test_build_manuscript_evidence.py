from __future__ import annotations

from build_manuscript_evidence import build
from make_reported_results import build as build_reported


def test_headline_values_and_boundaries() -> None:
    ledger = build()
    spectral = ledger["spectral"]
    assert round(spectral["Docking-44"]["correlation"]["raw"]["participation_ratio"], 3) == 1.834
    assert round(spectral["DOCKSTRING-58"]["correlation"]["residual"]["participation_ratio"], 3) == 18.206
    panels = ledger["fixed20_estimand_decomposition"]["panels"]
    assert set(panels) == {"DAVIS", "PKIS2", "PKIS1", "KiRHub"}
    assert all(
        panel["paired_qap_for_docking_increment"]["one_sided_p_positive_delta"] > 0.1
        for panel in panels.values()
    )
    matched = ledger["fixed20_estimand_decomposition"][
        "transformation_matched_independent_column_null"
    ]["panels"]
    assert all(
        panel["raw_experimental_target_geometry"][
            "p_observed_gain_at_least_as_large"
        ]
        < 0.01
        for panel in matched.values()
    )
    assert all(
        panel["two_way_centered_experimental_target_geometry"][
            "p_observed_gain_at_least_as_large"
        ]
        > 0.05
        for panel in matched.values()
    )
    assert (
        matched["PKIS1"]["two_way_centered_experimental_target_geometry"][
            "matched_support_observed_centered_minus_raw_docking"
        ]
        < 0
    )
    assert all(
        0.55
        < panel["path_averaged_descriptive_attribution"][
            "experimental_fraction_of_total"
        ]
        < 0.72
        for panel in panels.values()
    )
    centering = ledger["centering_panel_sensitivity"]
    assert centering["target_pairs"] == 190
    assert set(centering["experimental_panel_results"]) == {
        "DAVIS",
        "PKIS2",
        "PKIS1",
        "KiRHub",
    }

    dockstring = ledger["large_matrix_input_audits"]["DOCKSTRING-58"]
    assert dockstring["release_rows"] == 260_155
    assert dockstring["analysis_rows"] == 260_060
    assert dockstring["source_missing_cells"] == 339
    assert dockstring["source_rows_with_any_missing"] == 95
    assert dockstring["source_strictly_positive_cells"] == 5_413
    assert dockstring["analysis_strictly_positive_cells"] == 5_393
    assert abs(
        dockstring["all_rows_after_imputation"]["target_mean"]
        ["residual_participation_ratio"]
        - 18.214353
    ) < 1e-6

    davis = ledger["experimental_measurement_audits"]["DAVIS"]
    assert davis["cells_at_pkd_floor"] == 965
    assert round(davis["fraction_at_pkd_floor"], 3) == 0.670
    assert davis["targets_above_90_percent_at_floor"] == 4


def test_ranking_invariance_is_exact() -> None:
    ranking = ledger = build()["ligand_wise_ranking_boundary"]
    assert {
        "docking_target_prior",
        "experimental_target_prior",
        "cohort_experimental_target_prior",
    }.issubset(ranking["representations"])
    assert round(
        ranking["representations"]["docking_target_prior"][
            "pair_weighted_accuracy"
        ],
        3,
    ) == 0.585
    assert round(
        ranking["representations"]["absolute_vina"]["pair_weighted_accuracy"],
        3,
    ) == 0.584
    assert (
        ranking["representations"]["target_centered_unscaled"]
        ["mean_per_ligand_pairwise_accuracy"]
        == ranking["representations"]["two_way_centered_unscaled"]
        ["mean_per_ligand_pairwise_accuracy"]
    )
    assert (
        ranking["paired_comparisons"]
        ["two_way_centered_unscaled_minus_target_centered_unscaled"]
        ["plugin_mean_difference"]
        == 0.0
    )


def test_dense_ligand_wise_benchmarks_are_registered() -> None:
    dense = build()["dense_ligand_wise_benchmarks"]
    assert dense["DAVIS"]["support"] == {
        "matched_ligands": 59,
        "targets": 21,
        "evaluated_ligands": 56,
        "evaluated_pairs": 6065,
    }
    assert dense["PKIS2"]["support"] == {
        "matched_ligands": 154,
        "targets": 21,
        "evaluated_ligands": 154,
        "evaluated_pairs": 16417,
    }
    for panel in ("DAVIS", "PKIS2"):
        record = dense[panel]
        assert "docking_target_prior" in record["representations"]
        absolute = record["representations"]["absolute_vina"][
            "mean_per_ligand_pairwise_concordance"
        ]
        residual = record["representations"]["two_way_residual"][
            "mean_per_ligand_pairwise_concordance"
        ]
        contrast = record["paired_comparisons"][
            "two_way_residual_minus_absolute_vina"
        ]["plugin_mean_difference"]
        assert abs((residual - absolute) - contrast) < 1e-12
        scale_step = record["paired_comparisons"][
            "target_centered_residual_scaled_minus_target_centered_unscaled"
        ]
        row_step = record["paired_comparisons"][
            "two_way_residual_minus_target_centered_residual_scaled"
        ]
        prior_step = record["paired_comparisons"][
            "absolute_vina_minus_docking_target_prior"
        ]
        offset_step = record["paired_comparisons"][
            "target_centered_unscaled_minus_absolute_vina"
        ]
        for method in ("murcko_cluster_bootstrap", "butina_cluster_bootstrap"):
            assert prior_step["uncertainty"][method]["interval_95"][0] < 0
            assert prior_step["uncertainty"][method]["interval_95"][1] > 0
            assert offset_step["uncertainty"][method]["interval_95"][0] < 0
            assert offset_step["uncertainty"][method]["interval_95"][1] > 0
            assert scale_step["uncertainty"][method]["interval_95"][1] < 0
            assert row_step["uncertainty"][method]["interval_95"][0] > 0
        assert record["operational_path_steps"]["status"].startswith(
            "exploratory"
        )
    assert dense["DAVIS"]["target_composition_sensitivity"][
        "jackknife_bias_corrected_normal_95_interval"
    ][0] < 0 < dense["DAVIS"]["target_composition_sensitivity"][
        "jackknife_bias_corrected_normal_95_interval"
    ][1]
    assert dense["PKIS2"]["target_composition_sensitivity"][
        "jackknife_normal_95_interval"
    ][0] < 0 < dense["PKIS2"]["target_composition_sensitivity"][
        "jackknife_normal_95_interval"
    ][1]
    davis_strata = dense["DAVIS"]["outcome_conditioned_strata"]
    assert davis_strata["both_uncensored"]["paired_comparisons"][
        "two_way_residual_minus_absolute_vina"
    ]["plugin_mean_difference"] > 0
    assert davis_strata["floor_vs_uncensored"]["paired_comparisons"][
        "two_way_residual_minus_absolute_vina"
    ]["plugin_mean_difference"] < 0
    pkis_active = dense["PKIS2"]["outcome_conditioned_strata"]["both_active"]
    assert pkis_active["experimental_margin_name"] == "absolute_difference_gt_10"
    assert pkis_active["experimental_margin_percentage_points"] == 10.0
    assert "greater than 10 percentage points" in pkis_active["status"]
    assert "unadjusted for multiplicity" in pkis_active["status"]
    pkis_exact_non_tie = dense["PKIS2"]["outcome_conditioned_strata"][
        "secondary_exact_non_tie_both_active"
    ]
    assert pkis_exact_non_tie["experimental_margin_name"] == "exact_non_ties"
    assert pkis_exact_non_tie["experimental_margin_percentage_points"] == 0.0
    assert "exact-non-tie sensitivity" in pkis_exact_non_tie["status"]


def test_ranking_centering_panel_sensitivity_is_registered() -> None:
    ledger = build()
    sensitivity = ledger["ranking_centering_panel_sensitivity"]
    assert sensitivity["support"]["targets"] == 38
    assert sensitivity["fit_contract"]["all_target_count"] == 44
    assert (
        sensitivity["ranking"]["changed_pair_predictions"] == 31
    )
    assert abs(
        sensitivity["ranking"]["local_38_minus_all_44"][
            "plugin_mean_difference"
        ]
    ) < 0.004


def test_exploratory_descriptor_residualization_contract_is_registered() -> None:
    ledger = build()
    assert ledger["schema_version"] == "3.1.0"
    exploratory = ledger["exploratory_chemical_domain_structure"]
    assert exploratory["analysis_status"] == "exploratory_post_hoc_science_only"
    descriptor = exploratory["descriptor_residualization"]
    assert descriptor["descriptor_names_in_order"] == [
        "heavy_atoms",
        "molecular_weight",
        "labute_asa",
        "tpsa",
        "clogp",
        "rotatable_bonds",
        "ring_count",
    ]
    expected = {
        "Docking-44": {
            "source_ligands": 12_651,
            "analysis_ligands": 12_651,
            "targets": 44,
            "groups": 8_150,
            "relative_r2_reduction": 0.4687472762980258,
            "pr_before": 9.294143343764464,
            "pr_after": 14.746371479452172,
        },
        "DOCKSTRING-58": {
            "source_ligands": 260_060,
            "analysis_ligands": 15_000,
            "targets": 58,
            "groups": 11_905,
            "relative_r2_reduction": 0.5360024559043216,
            "pr_before": 17.923478828457238,
            "pr_after": 28.46641503690876,
        },
    }
    for dataset, reference in expected.items():
        record = descriptor["datasets"][dataset]
        support = record["support"]
        folds = record["fold_contract"]
        metrics = record["metrics"]
        assert support["source_matrix_ligands"] == reference["source_ligands"]
        assert support["analysis_ligands"] == reference["analysis_ligands"]
        assert support["targets"] == reference["targets"]
        assert support["chemical_groups"] == reference["groups"]
        assert folds["folds"] == 5
        assert folds["group_overlap_counts"] == [0, 0, 0, 0, 0]
        assert all(
            folds[key]
            for key in (
                "strict_fold_local_target_offsets",
                "strict_fold_local_residual_target_scales",
                "strict_fold_local_descriptor_scaling",
                "strict_fold_local_regression_coefficients",
            )
        )
        assert (
            metrics[
                "out_of_fold_relative_mean_squared_target_correlation_reduction"
            ]
            == reference["relative_r2_reduction"]
        )
        assert metrics["out_of_fold_pr_before_descriptor_removal"] == reference[
            "pr_before"
        ]
        assert metrics["out_of_fold_pr_after_descriptor_removal"] == reference[
            "pr_after"
        ]
        assert record["identical_metrics_in_both_source_summaries"] is True


def test_exploratory_mw_domain_instability_contract_is_registered() -> None:
    ledger = build()
    exploratory = ledger["exploratory_chemical_domain_structure"]
    mw = exploratory["low_high_molecular_weight_domain_instability"]
    assert mw["analysis_status"] == "exploratory_post_hoc"
    assert "not prospectively prespecified" in mw["stratifier_selection"]
    assert mw["extreme_fraction_per_tail"] == 0.25
    assert mw["random_disjoint_control"]["repetitions"] == 500
    assert mw["chemical_group_bootstrap"]["repetitions"] == 300
    assert mw["mw_matched_chemical_group_disjoint_control"]["repetitions"] == 200

    expected = {
        "Docking-44": {
            "low": 3_163,
            "high": 3_163,
            "residual_spearman": 0.20496480945389367,
            "cluster_interval": [0.18538271714397053, 0.21868853738208616],
            "random_mean": 0.9914867245178385,
            "mw_matched_mean": 0.9911554422019292,
            "adjusted_spearman": 0.4892200836062725,
            "sign_flip_fraction": 0.4249471458773784,
            "top_pair_jaccard": 0.07954545454545454,
        },
        "DOCKSTRING-58": {
            "low": 3_751,
            "high": 3_750,
            "residual_spearman": 0.3505978420709316,
            "cluster_interval": [0.32647052757374273, 0.3629274450466302],
            "random_mean": 0.9903743812980054,
            "mw_matched_mean": 0.9944045608141542,
            "adjusted_spearman": 0.43669105310837264,
            "sign_flip_fraction": 0.3666061705989111,
            "top_pair_jaccard": 0.16083916083916083,
        },
    }
    for dataset, reference in expected.items():
        record = mw["datasets"][dataset]
        residual = record["row_centered_residual_geometry"]
        assert record["low_mw_domain"]["ligands"] == reference["low"]
        assert record["high_mw_domain"]["ligands"] == reference["high"]
        assert residual["spearman"] == reference["residual_spearman"]
        assert (
            residual["chemical_group_bootstrap"]["central_95_percent_interval"]
            == reference["cluster_interval"]
        )
        assert residual["random_disjoint_support_control"]["mean"] == reference[
            "random_mean"
        ]
        assert (
            residual["mw_matched_chemical_group_disjoint_control"]["mean"]
            == reference["mw_matched_mean"]
        )
        assert (
            record["strict_descriptor_adjusted_residual_sensitivity"][
                "geometry_spearman"
            ]
            == reference["adjusted_spearman"]
        )
        assert residual["target_pair_sign_flip_fraction"] == reference[
            "sign_flip_fraction"
        ]
        assert residual["top_10_percent_pair_jaccard"] == reference[
            "top_pair_jaccard"
        ]
    assert mw["datasets"]["DOCKSTRING-58"]["support_seed_sensitivity"] == {
        "sample_ligands_per_seed": [15_000],
        "seeds": [71, 72, 73, 74, 75, 20260809],
        "geometry_spearman_range": [
            0.3254038727710366,
            0.36102479369214213,
        ],
    }


def test_restriction_matched_within_band_controls_and_raw_geometry_are_registered() -> None:
    mw = build()["exploratory_chemical_domain_structure"][
        "low_high_molecular_weight_domain_instability"
    ]
    contract = mw["restriction_matched_within_band_reproducibility"]
    assert contract["row_random_repetitions_per_band"] == 500
    assert contract["chemical_group_disjoint_repetitions_per_band"] == 200
    assert "not a population confidence interval" in contract[
        "interval_interpretation"
    ]

    expected_raw = {
        "Docking-44": {
            "rho": 0.2272436943147387,
            "cluster_interval": [0.2085383735937504, 0.2458500896520591],
            "random_mean": 0.9952582146066784,
            "mw_matched_mean": 0.9952455648551028,
        },
        "DOCKSTRING-58": {
            "rho": 0.40947327442193726,
            "cluster_interval": [0.38351586677558297, 0.4320760115748485],
            "random_mean": 0.9961951379572254,
            "mw_matched_mean": 0.9977061042654459,
        },
    }
    expected_group_means = {
        "Docking-44": {
            "raw": {"low_mw": 0.988730857888295, "high_mw": 0.9890983174031232},
            "residual": {
                "low_mw": 0.9799025862578818,
                "high_mw": 0.9776386936437524,
            },
        },
        "DOCKSTRING-58": {
            "raw": {"low_mw": 0.9911555227745348, "high_mw": 0.9792790518204079},
            "residual": {
                "low_mw": 0.9783629176216376,
                "high_mw": 0.9655916393026598,
            },
        },
    }
    for dataset, expected in expected_raw.items():
        dataset_record = mw["datasets"][dataset]
        raw = dataset_record["raw_surface_geometry"]
        assert raw["spearman"] == expected["rho"]
        assert raw["chemical_group_bootstrap"][
            "central_95_percent_interval"
        ] == expected["cluster_interval"]
        assert raw["random_disjoint_support_control"]["mean"] == expected[
            "random_mean"
        ]
        assert raw["mw_matched_chemical_group_disjoint_control"][
            "mean"
        ] == expected["mw_matched_mean"]
        assert dataset_record["raw_surface_geometry_spearman"] == raw["spearman"]

        for surface_name, surface_key in (
            ("raw", "raw_surface_geometry"),
            ("residual", "row_centered_residual_geometry"),
        ):
            surface = dataset_record[surface_key]
            within = surface["restriction_matched_within_band_reproducibility"]
            for band in ("low_mw", "high_mw"):
                row_random = within[band]["row_random_disjoint"]
                group_disjoint = within[band][
                    "mw_stratified_chemical_group_disjoint"
                ]
                assert row_random["repetitions"] == 500
                assert group_disjoint["repetitions"] == 200
                assert group_disjoint["maximum_chemical_group_overlap"] == 0
                assert group_disjoint["geometry_spearman_mean"] == (
                    expected_group_means[dataset][surface_name][band]
                )
                assert surface["spearman"] < group_disjoint[
                    "central_95_percent_repeated_split_sensitivity_range"
                ][0]
                assert surface["spearman"] < row_random[
                    "central_95_percent_repeated_split_sensitivity_range"
                ][0]
                assert "not a population confidence interval" in group_disjoint[
                    "range_interpretation"
                ]

    _, _, registry = build_reported(build())
    macros = registry["macros"]
    assert macros["DffMWRawDomainRho"] == "0.227"
    assert macros["DsMWRawDomainRho"] == "0.409"
    assert macros["DffMWRawDomainControl"] == "0.995"
    assert macros["DsMWRawDomainControl"] == "0.998"
    assert macros["DffMWWithinLowResidualGroupDisjointMean"] == "0.980"
    assert macros["DffMWWithinHighResidualGroupDisjointMean"] == "0.978"
    assert macros["DsMWWithinLowResidualGroupDisjointMean"] == "0.978"
    assert macros["DsMWWithinHighResidualGroupDisjointMean"] == "0.966"


def test_post_hoc_descriptor_family_sensitivity_is_registered() -> None:
    exploratory = build()["exploratory_chemical_domain_structure"]
    family = exploratory["post_hoc_descriptor_family_stratification"]
    descriptor_order = [
        "heavy_atoms",
        "molecular_weight",
        "labute_asa",
        "tpsa",
        "clogp",
        "rotatable_bonds",
        "ring_count",
    ]
    assert family["analysis_status"] == (
        "exploratory_post_hoc_family_sensitivity"
    )
    assert "post hoc" in family["selection_boundary"]
    assert "boundary tie" in family["selection_boundary"]
    assert family["descriptor_names_in_order"] == descriptor_order
    assert family[
        "all_datasets_all_seven_below_mw_matched_control_mean"
    ]

    expected_rho = {
        "Docking-44": [
            0.2073273331886823,
            0.2049648094538936,
            0.200058200573905,
            0.75305974173389,
            0.8872465653635073,
            0.7885726518323949,
            0.703229858992242,
        ],
        "DOCKSTRING-58": [
            0.3609419485950078,
            0.3505978420709316,
            0.3220319030299918,
            0.7871035949644485,
            0.8221111056209311,
            0.8314147147079953,
            0.8027916167478151,
        ],
    }
    for dataset, expected in expected_rho.items():
        record = family["datasets"][dataset]
        descriptor_results = record["descriptor_results"]
        control = record["mw_matched_chemical_group_disjoint_control"][
            "mean"
        ]
        assert list(descriptor_results) == descriptor_order
        assert [
            descriptor_results[name]["residual_map_spearman"]
            for name in descriptor_order
        ] == expected
        assert all(
            descriptor_results[name]["source_analysis_status"]
            == "exploratory_post_hoc"
            for name in descriptor_order
        )
        assert all(
            descriptor_results[name]["below_mw_matched_control_mean"]
            and descriptor_results[name]["residual_map_spearman"] < control
            for name in descriptor_order
        )
        assert record[
            "all_seven_geometry_spearman_below_mw_matched_control_mean"
        ]
        assert set(record["three_lowest_geometry_spearman_descriptors"]) == {
            "heavy_atoms",
            "molecular_weight",
            "labute_asa",
        }
        assert record[
            "molecular_size_family_forms_three_lowest_geometry_agreements"
        ]

    docking44_mw = family["datasets"]["Docking-44"]["descriptor_results"][
        "molecular_weight"
    ]
    dockstring_mw = family["datasets"]["DOCKSTRING-58"][
        "descriptor_results"
    ]["molecular_weight"]
    assert docking44_mw["low_domain_residual_participation_ratio"] == (
        13.62457616857832
    )
    assert docking44_mw["high_domain_residual_participation_ratio"] == (
        10.147009176104564
    )
    assert dockstring_mw["low_domain_residual_participation_ratio"] == (
        19.552064499216343
    )
    assert dockstring_mw["high_domain_residual_participation_ratio"] == (
        23.799697948677796
    )
    assert (
        family["datasets"]["Docking-44"][
            "molecular_weight_residual_pr_direction"
        ]
        == "lower_in_high_mw_domain"
    )
    assert (
        family["datasets"]["DOCKSTRING-58"][
            "molecular_weight_residual_pr_direction"
        ]
        == "higher_in_high_mw_domain"
    )


def test_exploratory_source_hashes_are_registered() -> None:
    artifacts = build()["source_artifacts"]
    assert (
        artifacts["results/residual_mechanism/analysis_summary.json"]
        == "44396240f2ae25ec2767deee386279025dffba7cfb5430f0e7b141da6114189a"
    )
    assert (
        artifacts["results/chemical_context_geometry/summary.json"]
        == "1a8472369493859538044a7fb1c6dd479a2f7d3798ab5c09f1711a9212bec11e"
    )
    assert (
        artifacts[
            "results/chemical_context_geometry/descriptor_extremes.csv"
        ]
        == "f8dc7c8848a6c5a2ded6d9aefa01e4b2cf8d5c80ab0b4d39959138c8e178f42a"
    )
    assert (
        artifacts[
            "results/chemical_context_geometry/within_mw_band_reproducibility_controls.csv"
        ]
        == "dc06fb766a54e15de54cf0e7b142bc96b5114b8ece046b4534ca913dd76fcbcf"
    )
    assert (
        artifacts[
            "results/chemical_context_geometry/within_mw_band_control_summary.csv"
        ]
        == "c57471f565fe087826894745f4292a218ad965e97d8fce0489cce47bee594b96"
    )
    assert (
        artifacts[
            "results/chemical_context_geometry/output_checksums.json"
        ]
        == "58db93b29b5d0a3a10b7d1f962ad7c83bbcd8e49d8745f1753580f2472cfaf88"
    )


def test_qap_resolution_and_fixed_pose_boundaries_are_registered() -> None:
    ledger = build()
    resolution = ledger["fixed20_target_label_qap_resolution"]
    assert "not a power-based" in resolution["interpretation"]
    assert set(resolution["panels"]) == {"DAVIS", "PKIS2", "PKIS1", "KiRHub"}
    for panel in resolution["panels"].values():
        for record in panel.values():
            assert record[
                "empirical_two_sided_95_percent_absolute_resolution_bound"
            ] > 0
            assert record["absolute_observed_to_null_bound_ratio"] >= 0

    restricted = ledger["strict_KLIFS_group_QAP_resolution"]
    effective = restricted["exchangeability_diagnostics"][
        "effective_exchangeable_units"
    ]
    assert effective["block_size_vector"] == [11, 3, 3]
    assert effective["scalar_effective_sample_size"] is None

    vina = ledger["vina_fixed_pose_term_attribution"]
    assert not vina["reproducibility_boundary"]["independent_redocking_performed"]
    assert set(vina["si_ready_term_table"]) == {
        "gauss1",
        "gauss2",
        "repulsion",
        "hydrophobic",
        "hydrogen",
    }
    transport = ledger["nonvina_scorer_transport"]
    assert not transport["reproducibility_boundary"][
        "independent_redocking_per_scorer"
    ]
    equivalence = ledger["ranking_equivalence_margin_sensitivity"]
    assert "post hoc" in equivalence["claim_boundary"]
