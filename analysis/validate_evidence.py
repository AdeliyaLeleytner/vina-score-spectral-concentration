#!/usr/bin/env python3
"""Validate mathematical and cross-analysis invariants in the evidence ledger."""

from __future__ import annotations

import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "results" / "evidence_summary.json"
MANUSCRIPT_EVIDENCE = ROOT / "results" / "manuscript_evidence.json"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def close(first: float, second: float, tolerance: float = 1e-10) -> bool:
    return math.isclose(float(first), float(second), rel_tol=tolerance, abs_tol=tolerance)


def main() -> None:
    evidence = json.loads(EVIDENCE.read_text())
    manuscript = json.loads(MANUSCRIPT_EVIDENCE.read_text())

    for dataset in ("docking44", "dockstring58"):
        ladder = evidence["centering_ladder"][dataset]
        for surface in ("raw", "interaction"):
            record = ladder[surface]
            targets = int(record["n_targets"])
            mean_r2 = float(record["mean_squared_offdiagonal_correlation"])
            implied_pr = targets / (1.0 + (targets - 1.0) * mean_r2)
            require(
                close(record["participation_ratio"], implied_pr),
                f"{dataset}/{surface}: PR does not equal its mean-r^2 identity",
            )
            require(
                close(sum(record["eigenvalues"]), targets),
                f"{dataset}/{surface}: correlation eigenvalues do not sum to target count",
            )
        require(
            ladder["interaction"]["numerical_zero_eigenvalues_at_1e-10"] >= 1,
            f"{dataset}: two-way-centered spectrum lacks its structural zero",
        )

    null_blocks = {
        "additive Gaussian": evidence["additive_main_effect_null"],
        "empirical residual permutation": evidence[
            "empirical_residual_permutation_null"
        ],
        "row-norm preserving": evidence["row_norm_preserving_residual_null"],
    }
    for dataset in ("docking44", "dockstring58"):
        supports = {
            label: block[dataset]["support_selection"]
            for label, block in null_blocks.items()
        }
        digests = {
            support["support_index_sha256"] for support in supports.values()
        }
        observed = {
            "additive Gaussian": null_blocks["additive Gaussian"][dataset][
                "observed"
            ]["residual"],
            "empirical residual permutation": null_blocks[
                "empirical residual permutation"
            ][dataset]["observed_residual"],
            "row-norm preserving": null_blocks["row-norm preserving"][dataset][
                "observed_residual"
            ],
        }
        require(
            len(digests) == 1
            and all(support["sample_rows"] == 12000 for support in supports.values())
            and all(support["method"] == "uniform_without_replacement" for support in supports.values()),
            f"{dataset}: residual nulls do not use one identical deterministic 12k support",
        )
        require(
            max(observed.values()) - min(observed.values()) < 1e-10,
            f"{dataset}: observed residual PR differs across common-support nulls",
        )

    for dataset, record in evidence["row_norm_preserving_residual_null"].items():
        repeats = int(record["repeats"])
        require(
            record["observed_residual"] < record["null_residual"]["minimum"],
            f"{dataset}: observed residual PR is not below every row-norm null realization",
        )
        require(
            close(
                record["empirical_lower_tail_p_for_residual_pr"],
                1.0 / (repeats + 1.0),
            ),
            f"{dataset}: row-norm-null lower-tail probability is inconsistent",
        )

    for dataset, sensitivity in evidence["spectral_estimand_sensitivity"].items():
        for surface in ("raw", "residual"):
            correlation = sensitivity["correlation"][surface]
            correlation_eigenvalues = [
                float(value) for value in correlation["eigenvalues"]
            ]
            correlation_implied_pr = sum(correlation_eigenvalues) ** 2 / sum(
                value ** 2 for value in correlation_eigenvalues
            )
            require(
                close(correlation["participation_ratio"], correlation_implied_pr),
                f"{dataset}/{surface}: correlation PR does not match its eigenvalues",
            )
            covariance = sensitivity["covariance"][surface]
            eigenvalues = [float(value) for value in covariance["eigenvalues"]]
            implied_pr = sum(eigenvalues) ** 2 / sum(
                value ** 2 for value in eigenvalues
            )
            require(
                close(covariance["participation_ratio"], implied_pr),
                f"{dataset}/{surface}: covariance PR does not match its eigenvalues",
            )
            require(
                close(covariance["trace"], sum(eigenvalues)),
                f"{dataset}/{surface}: covariance trace does not match its eigenvalues",
            )
            require(
                min(eigenvalues) >= 0,
                f"{dataset}/{surface}: covariance spectrum contains a negative eigenvalue",
            )
        require(
            sensitivity["covariance"]["residual"]["participation_ratio"]
            <= sensitivity["covariance"]["residual"]["algebraic_ceiling"] + 1e-10,
            f"{dataset}: residual covariance PR exceeds its algebraic ceiling",
        )
        require(
            sensitivity["correlation"]["residual"]["participation_ratio"]
            <= sensitivity["correlation"]["residual"]["algebraic_ceiling"] + 1e-10,
            f"{dataset}: residual correlation PR exceeds its algebraic ceiling",
        )
        require(
            close(
                sensitivity["correlation"]["raw"]["participation_ratio"],
                evidence["centering_ladder"][dataset]["raw"][
                    "participation_ratio"
                ],
            )
            and close(
                sensitivity["correlation"]["residual"]["participation_ratio"],
                evidence["centering_ladder"][dataset]["interaction"][
                    "participation_ratio"
                ],
            ),
            f"{dataset}: correlation estimand drifted from the headline spectrum",
        )

    for dataset, axis in evidence["pc1_axis"].items():
        cosine = float(axis["cosine_with_uniform_target_vector"])
        require(0 <= cosine <= 1, f"{dataset}: invalid uniform-axis cosine")
        require(
            close(axis["squared_cosine_with_uniform_target_vector"], cosine ** 2),
            f"{dataset}: uniform-axis squared cosine drifted",
        )
        require(
            close(sum(float(value) ** 2 for value in axis["loadings"]), 1.0),
            f"{dataset}: PC1 target direction is not unit-normalized",
        )

    for dataset, slopes in evidence["physicochemical_target_slopes"].items():
        for descriptor, record in slopes["descriptor_slopes"].items():
            require(
                record["maximum_absolute_error_in_two_way_slope_identity"] < 1e-10,
                f"{dataset}/{descriptor}: two-way slope identity failed",
            )
            require(
                0 <= record["cosine_with_residual_pc1_loading"] <= 1,
                f"{dataset}/{descriptor}: invalid residual-loading cosine",
            )
            require(
                0 <= record["absolute_correlation_with_residual_pc1_scores"] <= 1,
                f"{dataset}/{descriptor}: invalid descriptor--PC1 score correlation",
            )
            require(
                "residual_target_descriptor_correlation" in record,
                f"{dataset}/{descriptor}: descriptor correlations are mislabelled",
            )
            correlation_distribution = record[
                "residual_target_descriptor_correlation"
            ]
            require(
                correlation_distribution["minimum"] >= -1
                and correlation_distribution["maximum"] <= 1,
                f"{dataset}/{descriptor}: target--descriptor correlation is out of range",
            )
        for target, record in slopes["per_target"].items():
            require(
                0 <= record["imputed_cells"] <= slopes["input_n"],
                f"{dataset}/{target}: invalid imputed-cell count",
            )
            require(
                close(
                    record["imputed_fraction"],
                    record["imputed_cells"] / slopes["input_n"],
                ),
                f"{dataset}/{target}: imputed fraction does not match its count",
            )

    dock44_slope_sensitivity = evidence["physicochemical_target_slopes"][
        "docking44"
    ]["most_imputed_target_exclusion_sensitivity"]
    require(
        dock44_slope_sensitivity["removed_target"] == "4mqs"
        and dock44_slope_sensitivity["removed_target_imputed_cells"] == 3352,
        "Docking-44 descriptor-slope exclusion target drifted",
    )
    require(
        dock44_slope_sensitivity["remaining_n_targets"] == 43,
        "Docking-44 descriptor-slope exclusion has the wrong target count",
    )

    missing = evidence["preprocessing_sensitivity"][
        "missing_value_residual_sensitivity"
    ]
    input_characterization = evidence["preprocessing_sensitivity"][
        "input_characterization"
    ]
    require(
        input_characterization["strictly_positive_cells_after_primary_clipping"] == 0,
        "Docking-44 preprocessing sensitivity was not run on the clipped surface",
    )
    require(
        close(
            missing["methods"]["target_mean"]["raw_participation_ratio"],
            evidence["centering_ladder"]["docking44"]["raw"][
                "participation_ratio"
            ],
        )
        and close(
            missing["methods"]["target_mean"]["residual_participation_ratio"],
            evidence["centering_ladder"]["docking44"]["interaction"][
                "participation_ratio"
            ],
        ),
        "Docking-44 primary missing-value analysis drifted from the headline surface",
    )
    require(
        missing["without_most_affected_target"]["removed_target"] == "4mqs",
        "Docking-44 most-missing target drifted",
    )
    require(
        close(
            missing["methods"]["complete_case"]["residual_participation_ratio"],
            missing["methods"]["mean_imputed_restricted_to_complete_rows"][
                "residual_participation_ratio"
            ],
        ),
        "complete-case and same-row mean-imputed residual analyses diverged",
    )
    for dataset, record in evidence["residual_structure_characterization"].items():
        for surface in ("raw", "residual"):
            distribution = record[f"{surface}_correlation_distribution"]
            values = distribution["values"]
            require(
                close(
                    distribution["mean_absolute"],
                    sum(abs(value) for value in values) / len(values),
                ),
                f"{dataset}/{surface}: stored correlation summary does not match values",
            )

    expanded = evidence["expanded_target_preference_benchmark"]
    require(
        expanded["co_primary_operational_estimands"]["broad_coverage"]
        == "primary_all_exact",
        "broad operational estimand is not declared in the ledger",
    )
    broad = expanded["primary_all_exact"]
    representations = broad["representations"]
    require(
        close(
            representations["target_centered_unscaled"][
                "mean_per_ligand_pairwise_accuracy"
            ],
            representations["two_way_centered_unscaled"][
                "mean_per_ligand_pairwise_accuracy"
            ],
        ),
        "unscaled target-centered and two-way-centered rankings are not invariant",
    )
    for comparison, first, second in (
        (
            "two_way_residual_minus_absolute_vina",
            "two_way_residual",
            "absolute_vina",
        ),
        (
            "two_way_residual_minus_column_standardized",
            "two_way_residual",
            "column_standardized",
        ),
    ):
        implied = (
            representations[first]["mean_per_ligand_pairwise_accuracy"]
            - representations[second]["mean_per_ligand_pairwise_accuracy"]
        )
        require(
            close(
                broad["paired_comparisons"][comparison]["plugin_mean_difference"],
                implied,
            ),
            f"{comparison}: paired contrast does not match representation means",
        )

    require(
        expanded["assay_sensitivities"]["human_binding_Ki_Kd"]["n_targets"] == 30,
        "human binding Ki/Kd target count drifted",
    )

    davis_censoring = manuscript["davis_fixed20_censoring_sensitivity"]
    over90 = davis_censoring[
        "exclude_targets_over_90_percent_floor_sensitivity"
    ]
    require(
        over90["targets_retained"] == 16
        and over90["targets_removed"] == 4
        and over90["removed_target_names"]
        == ["AKT1", "AKT2", "MAPK1", "MAPKAPK2"],
        "DAVIS >90%-floor target exclusion contract drifted",
    )
    require(
        -1
        <= over90["centered_docking_vs_continuous_davis"]["spearman"]
        <= 1
        and -1
        <= over90["centered_docking_vs_above_floor_status_davis"]["spearman"]
        <= 1,
        "DAVIS >90%-floor sensitivity has an invalid concordance",
    )

    fixed20_resolution = manuscript["fixed20_target_label_qap_resolution"]
    require(
        "not a power-based" in fixed20_resolution["interpretation"],
        "fixed-20 QAP resolution is misrepresented as a power MDE",
    )
    for panel, endpoints in fixed20_resolution["panels"].items():
        for endpoint, record in endpoints.items():
            interval = record["empirical_central_95_percent_null_interval"]
            bound = max(abs(interval[0]), abs(interval[1]))
            observed = record["observed_centered_minus_raw_docking_spearman"]
            require(
                close(
                    bound,
                    record[
                        "empirical_two_sided_95_percent_absolute_resolution_bound"
                    ],
                )
                and record["observed_exceeds_absolute_null_bound"]
                == (abs(observed) > bound),
                f"{panel}/{endpoint}: QAP resolution identity drifted",
            )

    restricted = manuscript["strict_KLIFS_group_QAP_resolution"]
    effective = restricted["exchangeability_diagnostics"][
        "effective_exchangeable_units"
    ]
    require(
        effective["movable_target_labels"] == 17
        and effective["movable_blocks"] == 3
        and effective["block_size_vector"] == [11, 3, 3]
        and effective["fixed_singletons"] == 3
        and effective["scalar_effective_sample_size"] is None,
        "restricted-QAP exchangeability units drifted",
    )
    for endpoint, records in restricted["primary_results"].items():
        for adjustment, record in records.items():
            require(
                record["one_sided_null_interval_90"][0]
                <= record[
                    "empirical_one_sided_alpha_0_05_critical_partial_spearman"
                ]
                == record["one_sided_null_interval_90"][1]
                and record[
                    "additional_partial_spearman_needed_to_cross_one_sided_critical"
                ]
                >= 0,
                f"{endpoint}/{adjustment}: restricted-QAP resolution drifted",
            )

    dense = manuscript["dense_ligand_wise_benchmarks"]
    for panel in ("DAVIS", "PKIS2"):
        record = dense[panel]
        representations = record["representations"]
        comparisons = record["paired_comparisons"]
        require(
            "docking_target_prior" in representations,
            f"{panel}: ligand-invariant target prior is missing",
        )
        require(
            close(
                comparisons["two_way_residual_minus_absolute_vina"][
                    "plugin_mean_difference"
                ],
                representations["two_way_residual"][
                    "mean_per_ligand_pairwise_concordance"
                ]
                - representations["absolute_vina"][
                    "mean_per_ligand_pairwise_concordance"
                ],
            ),
            f"{panel}: dense net residual contrast drifted",
        )
        scale_step = comparisons[
            "target_centered_residual_scaled_minus_target_centered_unscaled"
        ]
        row_step = comparisons[
            "two_way_residual_minus_target_centered_residual_scaled"
        ]
        prior_step = comparisons["absolute_vina_minus_docking_target_prior"]
        offset_step = comparisons["target_centered_unscaled_minus_absolute_vina"]
        for method in ("murcko_cluster_bootstrap", "butina_cluster_bootstrap"):
            require(
                prior_step["uncertainty"][method]["interval_95"][0] < 0
                < prior_step["uncertainty"][method]["interval_95"][1],
                f"{panel}: dense absolute-minus-prior interval no longer spans zero",
            )
            require(
                offset_step["uncertainty"][method]["interval_95"][0] < 0
                < offset_step["uncertainty"][method]["interval_95"][1],
                f"{panel}: target-offset interval no longer spans zero",
            )
            require(
                scale_step["uncertainty"][method]["interval_95"][1] < 0,
                f"{panel}: dense residual-SD scaling interval no longer excludes zero",
            )
            require(
                row_step["uncertainty"][method]["interval_95"][0] > 0,
                f"{panel}: dense row-effect correction interval no longer excludes zero",
            )
        require(
            record["operational_path_steps"]["status"].startswith(
                "exploratory post-hoc"
            )
            and "unadjusted for multiplicity"
            in record["operational_path_steps"]["status"],
            f"{panel}: dense path decomposition status drifted",
        )
        require(
            record["analysis_parameters"]["bootstrap_repeats"] == 5000
            and record["analysis_parameters"]["seed"] == 0,
            f"{panel}: dense analysis contract drifted",
        )
    davis_strata = dense["DAVIS"]["outcome_conditioned_strata"]
    require(
        davis_strata["both_uncensored"]["paired_comparisons"][
            "two_way_residual_minus_absolute_vina"
        ]["plugin_mean_difference"]
        > 0
        and davis_strata["floor_vs_uncensored"]["paired_comparisons"][
            "two_way_residual_minus_absolute_vina"
        ]["plugin_mean_difference"]
        < 0,
        "DAVIS outcome-conditioned sign reversal drifted",
    )
    pkis_active = dense["PKIS2"]["outcome_conditioned_strata"]["both_active"]
    pkis_exact_non_tie = dense["PKIS2"]["outcome_conditioned_strata"][
        "secondary_exact_non_tie_both_active"
    ]
    require(
        pkis_active["experimental_margin_name"]
        == "absolute_difference_gt_10"
        and pkis_active["experimental_margin_percentage_points"] == 10.0
        and "greater than 10 percentage points" in pkis_active["status"]
        and "unadjusted for multiplicity" in pkis_active["status"],
        "PKIS2 primary-margin both-active contract drifted",
    )
    require(
        pkis_exact_non_tie["experimental_margin_name"] == "exact_non_ties"
        and pkis_exact_non_tie["experimental_margin_percentage_points"] == 0.0
        and "exact-non-tie sensitivity" in pkis_exact_non_tie["status"],
        "PKIS2 secondary exact-non-tie both-active contract drifted",
    )

    exploratory = manuscript["exploratory_chemical_domain_structure"]
    require(
        exploratory["analysis_status"] == "exploratory_post_hoc_science_only",
        "chemical-domain evidence is not explicitly marked exploratory",
    )
    descriptor = exploratory["descriptor_residualization"]
    require(
        descriptor["descriptor_names_in_order"]
        == [
            "heavy_atoms",
            "molecular_weight",
            "labute_asa",
            "tpsa",
            "clogp",
            "rotatable_bonds",
            "ring_count",
        ],
        "descriptor residualization feature contract drifted",
    )
    for dataset, record in descriptor["datasets"].items():
        support = record["support"]
        folds = record["fold_contract"]
        metrics = record["metrics"]
        require(
            folds["folds"] == 5
            and len(folds["group_overlap_counts"]) == 5
            and set(folds["group_overlap_counts"]) == {0},
            f"{dataset}: descriptor residualization is not five-fold group-disjoint",
        )
        require(
            all(
                folds[key]
                for key in (
                    "strict_fold_local_target_offsets",
                    "strict_fold_local_residual_target_scales",
                    "strict_fold_local_descriptor_scaling",
                    "strict_fold_local_regression_coefficients",
                )
            ),
            f"{dataset}: a descriptor preprocessing or fit step is not fold-local",
        )
        require(
            sum(folds["test_ligands"]) == support["analysis_ligands"]
            and metrics["chemical_groups"] == support["chemical_groups"],
            f"{dataset}: descriptor support counts are inconsistent",
        )
        before_r2 = metrics[
            "out_of_fold_mean_squared_target_correlation_before_descriptor_removal"
        ]
        after_r2 = metrics[
            "out_of_fold_mean_squared_target_correlation_after_descriptor_removal"
        ]
        require(
            close(
                before_r2 - after_r2,
                metrics[
                    "out_of_fold_absolute_mean_squared_target_correlation_reduction"
                ],
            )
            and close(
                (before_r2 - after_r2) / before_r2,
                metrics[
                    "out_of_fold_relative_mean_squared_target_correlation_reduction"
                ],
            ),
            f"{dataset}: descriptor-correlation reduction identity drifted",
        )
        pr_before = metrics["out_of_fold_pr_before_descriptor_removal"]
        pr_after = metrics["out_of_fold_pr_after_descriptor_removal"]
        require(
            close(pr_after - pr_before, metrics["out_of_fold_pr_increase"])
            and close(pr_after / pr_before, metrics["out_of_fold_pr_ratio"]),
            f"{dataset}: descriptor PR-change identity drifted",
        )
        targets = support["targets"]
        require(
            close(pr_before, targets / (1 + (targets - 1) * before_r2))
            and close(pr_after, targets / (1 + (targets - 1) * after_r2)),
            f"{dataset}: descriptor PR does not match mean-r-squared",
        )
        require(
            record["identical_metrics_in_both_source_summaries"],
            f"{dataset}: duplicate descriptor summaries are not reconciled",
        )

    mw = exploratory["low_high_molecular_weight_domain_instability"]
    require(
        mw["analysis_status"] == "exploratory_post_hoc"
        and "not prospectively prespecified" in mw["stratifier_selection"],
        "molecular-weight stratification is not labelled post hoc",
    )
    require(
        mw["extreme_fraction_per_tail"] == 0.25
        and mw["random_disjoint_control"]["repetitions"] == 500
        and mw["chemical_group_bootstrap"]["repetitions"] == 300
        and mw["mw_matched_chemical_group_disjoint_control"]["repetitions"]
        == 200,
        "molecular-weight domain analysis contract drifted",
    )
    within_contract = mw["restriction_matched_within_band_reproducibility"]
    require(
        within_contract["row_random_repetitions_per_band"] == 500
        and within_contract[
            "chemical_group_disjoint_repetitions_per_band"
        ]
        == 200
        and "not a population confidence interval"
        in within_contract["interval_interpretation"],
        "restriction-matched within-band control contract drifted",
    )
    expected_raw_cross_band = {
        "Docking-44": 0.2272436943147387,
        "DOCKSTRING-58": 0.40947327442193726,
    }
    expected_within_group_means = {
        "Docking-44": {
            "raw_surface_geometry": {
                "low_mw": 0.988730857888295,
                "high_mw": 0.9890983174031232,
            },
            "row_centered_residual_geometry": {
                "low_mw": 0.9799025862578818,
                "high_mw": 0.9776386936437524,
            },
        },
        "DOCKSTRING-58": {
            "raw_surface_geometry": {
                "low_mw": 0.9911555227745348,
                "high_mw": 0.9792790518204079,
            },
            "row_centered_residual_geometry": {
                "low_mw": 0.9783629176216376,
                "high_mw": 0.9655916393026598,
            },
        },
    }
    for dataset, record in mw["datasets"].items():
        residual = record["row_centered_residual_geometry"]
        require(
            close(
                residual["dissimilarity_one_minus_spearman"],
                1 - residual["spearman"],
            ),
            f"{dataset}: MW-domain geometry dissimilarity identity drifted",
        )
        bootstrap = residual["chemical_group_bootstrap"]
        random_control = residual["random_disjoint_support_control"]
        mw_matched = residual["mw_matched_chemical_group_disjoint_control"]
        require(
            bootstrap["central_95_percent_interval"][0]
            <= residual["spearman"]
            <= bootstrap["central_95_percent_interval"][1],
            f"{dataset}: observed MW-domain geometry is outside its cluster interval",
        )
        require(
            residual["spearman"]
            < random_control["central_95_percent_interval"][0]
            and residual["spearman"]
            < mw_matched["central_95_percent_interval"][0],
            f"{dataset}: MW-domain drift no longer separates from its controls",
        )
        require(
            close(
                random_control["finite_support_lower_tail_probability"],
                1 / (random_control["repetitions"] + 1),
            ),
            f"{dataset}: random-support calibration probability drifted",
        )
        adjusted = record["strict_descriptor_adjusted_residual_sensitivity"]
        require(
            adjusted["geometry_spearman"]
            < mw_matched["central_95_percent_interval"][0],
            f"{dataset}: descriptor-adjusted MW-domain contrast no longer separates",
        )
        require(
            record["low_mw_domain"]["ligands"] >= 3
            and record["high_mw_domain"]["ligands"] >= 3
            and record["low_mw_domain"]["inclusive_upper_threshold"]
            < record["high_mw_domain"]["inclusive_lower_threshold"],
            f"{dataset}: MW-domain support is invalid",
        )
        raw = record["raw_surface_geometry"]
        require(
            close(raw["spearman"], expected_raw_cross_band[dataset])
            and close(record["raw_surface_geometry_spearman"], raw["spearman"])
            and close(
                raw["dissimilarity_one_minus_spearman"],
                1 - raw["spearman"],
            ),
            f"{dataset}: raw MW-domain geometry drifted",
        )
        require(
            raw["chemical_group_bootstrap"]["central_95_percent_interval"][0]
            <= raw["spearman"]
            <= raw["chemical_group_bootstrap"]["central_95_percent_interval"][1]
            and raw["spearman"]
            < raw["random_disjoint_support_control"][
                "central_95_percent_interval"
            ][0]
            and raw["spearman"]
            < raw["mw_matched_chemical_group_disjoint_control"][
                "central_95_percent_interval"
            ][0],
            f"{dataset}: raw MW-domain drift no longer separates from controls",
        )
        for surface_key in (
            "raw_surface_geometry",
            "row_centered_residual_geometry",
        ):
            surface = record[surface_key]
            within = surface[
                "restriction_matched_within_band_reproducibility"
            ]
            require(
                set(within) == {"low_mw", "high_mw"},
                f"{dataset}/{surface_key}: within-band domains drifted",
            )
            for band in ("low_mw", "high_mw"):
                require(
                    set(within[band])
                    == {
                        "row_random_disjoint",
                        "mw_stratified_chemical_group_disjoint",
                    },
                    f"{dataset}/{surface_key}/{band}: control set drifted",
                )
                row_random = within[band]["row_random_disjoint"]
                group_disjoint = within[band][
                    "mw_stratified_chemical_group_disjoint"
                ]
                require(
                    row_random["repetitions"] == 500
                    and group_disjoint["repetitions"] == 200
                    and group_disjoint["maximum_chemical_group_overlap"] == 0
                    and close(
                        group_disjoint["geometry_spearman_mean"],
                        expected_within_group_means[dataset][surface_key][band],
                    ),
                    f"{dataset}/{surface_key}/{band}: split contract drifted",
                )
                for control in (row_random, group_disjoint):
                    sensitivity = control[
                        "central_95_percent_repeated_split_sensitivity_range"
                    ]
                    require(
                        sensitivity[0]
                        <= control["geometry_spearman_median"]
                        <= sensitivity[1]
                        and surface["spearman"] < sensitivity[0]
                        and "not a population confidence interval"
                        in control["range_interpretation"],
                        f"{dataset}/{surface_key}/{band}: restriction-matched separation drifted",
                    )
    dockstring_seed = mw["datasets"]["DOCKSTRING-58"][
        "support_seed_sensitivity"
    ]
    require(
        len(dockstring_seed["seeds"]) == 6
        and dockstring_seed["sample_ligands_per_seed"] == [15000]
        and dockstring_seed["geometry_spearman_range"][0]
        <= mw["datasets"]["DOCKSTRING-58"]["row_centered_residual_geometry"][
            "spearman"
        ]
        <= dockstring_seed["geometry_spearman_range"][1],
        "DOCKSTRING MW-domain support-seed contract drifted",
    )

    descriptor_family = exploratory[
        "post_hoc_descriptor_family_stratification"
    ]
    expected_descriptor_order = [
        "heavy_atoms",
        "molecular_weight",
        "labute_asa",
        "tpsa",
        "clogp",
        "rotatable_bonds",
        "ring_count",
    ]
    expected_geometry = {
        "Docking-44": {
            "heavy_atoms": 0.2073273331886823,
            "molecular_weight": 0.20496480945389367,
            "labute_asa": 0.200058200573905,
            "tpsa": 0.75305974173389,
            "clogp": 0.8872465653635073,
            "rotatable_bonds": 0.7885726518323949,
            "ring_count": 0.703229858992242,
        },
        "DOCKSTRING-58": {
            "heavy_atoms": 0.3609419485950078,
            "molecular_weight": 0.3505978420709316,
            "labute_asa": 0.3220319030299918,
            "tpsa": 0.7871035949644485,
            "clogp": 0.8221111056209311,
            "rotatable_bonds": 0.8314147147079953,
            "ring_count": 0.8027916167478151,
        },
    }
    require(
        descriptor_family["analysis_status"]
        == "exploratory_post_hoc_family_sensitivity"
        and "post hoc" in descriptor_family["selection_boundary"]
        and "boundary tie" in descriptor_family["selection_boundary"],
        "descriptor-family sensitivity is not explicitly labelled post hoc",
    )
    require(
        descriptor_family["descriptor_names_in_order"]
        == expected_descriptor_order
        and set(descriptor_family["descriptor_families"])
        == set(expected_descriptor_order),
        "descriptor-family sensitivity does not contain the exact seven-descriptor set",
    )
    require(
        descriptor_family[
            "all_datasets_all_seven_below_mw_matched_control_mean"
        ],
        "not every descriptor split remains below its MW-matched control",
    )
    for dataset, expected in expected_geometry.items():
        record = descriptor_family["datasets"][dataset]
        results = record["descriptor_results"]
        control = record["mw_matched_chemical_group_disjoint_control"][
            "mean"
        ]
        require(
            set(results) == set(expected_descriptor_order)
            and all(
                value["source_analysis_status"] == "exploratory_post_hoc"
                for value in results.values()
            ),
            f"{dataset}: descriptor-family rows or post-hoc labels drifted",
        )
        require(
            all(
                close(results[name]["residual_map_spearman"], value)
                for name, value in expected.items()
            ),
            f"{dataset}: descriptor-family residual-map values drifted",
        )
        require(
            record[
                "all_seven_geometry_spearman_below_mw_matched_control_mean"
            ]
            and all(
                value["below_mw_matched_control_mean"]
                and value["residual_map_spearman"] < control
                for value in results.values()
            ),
            f"{dataset}: all-seven versus MW-matched-control statement failed",
        )
        require(
            set(record["three_lowest_geometry_spearman_descriptors"])
            == {"heavy_atoms", "molecular_weight", "labute_asa"}
            and record[
                "molecular_size_family_forms_three_lowest_geometry_agreements"
            ],
            f"{dataset}: molecular-size-family ordering drifted",
        )
    docking44_mw = descriptor_family["datasets"]["Docking-44"][
        "descriptor_results"
    ]["molecular_weight"]
    dockstring_mw = descriptor_family["datasets"]["DOCKSTRING-58"][
        "descriptor_results"
    ]["molecular_weight"]
    require(
        close(
            docking44_mw["low_domain_residual_participation_ratio"],
            13.62457616857832,
        )
        and close(
            docking44_mw["high_domain_residual_participation_ratio"],
            10.147009176104564,
        )
        and close(
            dockstring_mw["low_domain_residual_participation_ratio"],
            19.552064499216343,
        )
        and close(
            dockstring_mw["high_domain_residual_participation_ratio"],
            23.799697948677796,
        )
        and descriptor_family["datasets"]["Docking-44"][
            "molecular_weight_residual_pr_direction"
        ]
        == "lower_in_high_mw_domain"
        and descriptor_family["datasets"]["DOCKSTRING-58"][
            "molecular_weight_residual_pr_direction"
        ]
        == "higher_in_high_mw_domain",
        "MW-domain residual PR no longer moves in opposite directions",
    )

    require(
        manuscript["schema_version"] == "3.1.0",
        "manuscript evidence schema did not advance with the rank-matched controls",
    )
    feature_controls = manuscript["descriptor_rank_matched_controls"]
    predictive = feature_controls["predictive_metrics"]
    physchem = predictive["physicochemical_7"]
    nuisance_hash = predictive["stable_hash_nuisance_7"]
    require(
        physchem["mean_out_of_fold_target_r2"] > 0.05
        and physchem[
            "relative_reduction_in_mean_squared_offdiagonal_target_correlation"
        ]
        > 0.10,
        "physicochemical basis no longer predicts nontrivial held-out score amplitude",
    )
    require(
        abs(nuisance_hash["mean_out_of_fold_target_r2"]) < 1e-3
        and abs(
            nuisance_hash[
                "relative_reduction_in_mean_squared_offdiagonal_target_correlation"
            ]
        )
        < 1e-3
        and max(
            feature_controls["panel_geometry_agreement"][
                "stable_hash_nuisance_7"
            ].values()
        )
        > 0.30,
        "stable-hash amplitude--geometry negative control drifted",
    )
    morgan_distribution = feature_controls[
        "morgan_count_plus_size_rp_7_seed_ensemble"
    ]["algorithmic_distributions"][
        "relative_reduction_in_mean_squared_offdiagonal_target_correlation"
    ]
    require(
        morgan_distribution[
            "ensemble_members_at_least_as_large_as_physicochemical"
        ]
        >= 1,
        "rank-matched Morgan ensemble no longer falsifies basis specificity",
    )
    nuisance_distribution = feature_controls[
        "physicochemical_row_permutation_nuisance_ensemble"
    ]["algorithmic_seed_distributions"]
    nuisance_r2 = nuisance_distribution["mean_out_of_fold_target_r2"]
    nuisance_reduction = nuisance_distribution[
        "relative_reduction_in_mean_squared_offdiagonal_target_correlation"
    ]
    require(
        max(abs(nuisance_r2["minimum"]), abs(nuisance_r2["maximum"])) < 1e-3
        and max(
            abs(nuisance_reduction["minimum"]),
            abs(nuisance_reduction["maximum"]),
        )
        < 1e-3
        and max(
            panel["maximum"]
            for panel in nuisance_distribution["panel_geometry_agreement"].values()
        )
        > 0.45,
        "20-seed nuisance-projection amplitude--geometry separation drifted",
    )

    kikd = manuscript["dockstring_chembl_ranking"]["sensitivities"][
        "human_binding_Ki_Kd"
    ]
    broad_primary = manuscript["dockstring_chembl_ranking"]["benchmark"]
    broad_representations = broad_primary["representations"]
    absolute_minus_prior = broad_primary["paired_comparisons"][
        "absolute_vina_minus_docking_target_prior"
    ]
    require(
        close(
            absolute_minus_prior["plugin_mean_difference"],
            broad_representations["absolute_vina"][
                "mean_per_ligand_pairwise_accuracy"
            ]
            - broad_representations["docking_target_prior"][
                "mean_per_ligand_pairwise_accuracy"
            ],
        )
        and absolute_minus_prior["plugin_mean_difference"] > 0,
        "broad-primary absolute-minus-docking-prior contrast drifted",
    )
    for method in ("scaffold_cluster_bootstrap", "butina_cluster_bootstrap"):
        low, high = absolute_minus_prior[method]["interval_95"]
        require(
            low < 0 < high,
            f"broad-primary absolute-minus-docking-prior {method} interval no longer spans zero",
        )
    mixing = kikd["endpoint_mixing_diagnostic"]
    counts = mixing["pair_counts"]
    require(
        mixing["informative_non_tied_pairs"]
        == kikd["support"]["non_tied_within_ligand_pairs"]
        == 3955
        and mixing["endpoint_unambiguous_pairs"] == 3643
        and mixing["excluded_exact_median_ties"] == 25,
        "human binding Ki/Kd endpoint-provenance support drifted",
    )
    require(
        counts
        == {
            "same_endpoint": 3610,
            "same_endpoint_Ki_Ki": 1358,
            "same_endpoint_Kd_Kd": 2252,
            "mixed_endpoint_Ki_Kd": 33,
            "involves_pooled_Ki_Kd_cell": 312,
        }
        and mixing["observed_cell_endpoint_provenance"]
        == {"Ki_only": 2008, "Kd_only": 704, "pooled_Ki_and_Kd": 71},
        "human binding Ki/Kd endpoint-provenance classification drifted",
    )
    require(
        close(
            mixing["fractions_of_all_informative_pairs"]["same_endpoint"],
            counts["same_endpoint"] / 3955,
        )
        and close(
            mixing["fractions_among_endpoint_unambiguous_pairs"][
                "same_endpoint"
            ],
            counts["same_endpoint"] / 3643,
        ),
        "human binding Ki/Kd endpoint-provenance fractions drifted",
    )

    source_artifacts = manuscript["source_artifacts"]
    require(
        source_artifacts["results/residual_mechanism/analysis_summary.json"]
        == "44396240f2ae25ec2767deee386279025dffba7cfb5430f0e7b141da6114189a"
        and source_artifacts["results/chemical_context_geometry/summary.json"]
        == "1a8472369493859538044a7fb1c6dd479a2f7d3798ab5c09f1711a9212bec11e",
        "exploratory chemical-domain source hashes drifted",
    )
    require(
        source_artifacts[
            "results/chemical_context_geometry/descriptor_extremes.csv"
        ]
        == "f8dc7c8848a6c5a2ded6d9aefa01e4b2cf8d5c80ab0b4d39959138c8e178f42a",
        "descriptor-extreme source hash drifted",
    )
    require(
        source_artifacts[
            "results/chemical_context_geometry/within_mw_band_reproducibility_controls.csv"
        ]
        == "dc06fb766a54e15de54cf0e7b142bc96b5114b8ece046b4534ca913dd76fcbcf"
        and source_artifacts[
            "results/chemical_context_geometry/within_mw_band_control_summary.csv"
        ]
        == "c57471f565fe087826894745f4292a218ad965e97d8fce0489cce47bee594b96"
        and source_artifacts[
            "results/chemical_context_geometry/output_checksums.json"
        ]
        == "58db93b29b5d0a3a10b7d1f962ad7c83bbcd8e49d8745f1753580f2472cfaf88",
        "restriction-matched chemical-context source hashes drifted",
    )

    vina_terms = manuscript["vina_fixed_pose_term_attribution"]
    require(
        vina_terms["reproducibility_boundary"]["independent_redocking_performed"]
        is False
        and "fixed" in vina_terms["claim_boundary"],
        "Vina-term fixed-pose boundary is missing",
    )
    term_table = vina_terms["si_ready_term_table"]
    require(
        set(term_table)
        == {"gauss1", "gauss2", "repulsion", "hydrophobic", "hydrogen"},
        "Vina-term SI table is incomplete",
    )
    require(
        close(
            sum(
                record["shared_row_axis_covariance_share"]
                for record in term_table.values()
            ),
            1.0,
        )
        and close(
            sum(
                record["residual_frobenius_inner_product_share"]
                for record in term_table.values()
            ),
            1.0,
        ),
        "Vina-term linear attribution shares do not close",
    )
    transport = manuscript["nonvina_scorer_transport"]
    require(
        transport["reproducibility_boundary"]["independent_redocking_per_scorer"]
        is False
        and "Python 3.9"
        in transport["reproducibility_boundary"][
            "source_restricted_oddt_stage"
        ]["environment"]
        and "/Users/" not in json.dumps(transport)
        and "/private/tmp/" not in json.dumps(transport),
        "non-Vina fixed-pose/environment boundary is not portable",
    )
    equivalence = manuscript["ranking_equivalence_margin_sensitivity"]
    require(
        "post hoc" in equivalence["claim_boundary"]
        and equivalence["cross_benchmark_summary"][
            "smallest_grid_margin_passing_on_all_three"
        ]
        == 0.03,
        "equivalence-margin sensitivity boundary drifted",
    )
    print("Evidence invariants validated")


if __name__ == "__main__":
    main()
