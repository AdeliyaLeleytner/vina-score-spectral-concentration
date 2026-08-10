#!/usr/bin/env python3
"""Generate high-risk manuscript numbers and the main spectral comparison table."""

from __future__ import annotations

import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

# Reference symmetric equivalence margin for the dense ligand-wise benchmarks, on the
# per-ligand pairwise-concordance scale. It is NOT preregistered: it was chosen after the
# benchmarks were computed and is reported as one point on a margin sensitivity curve, not as
# a confirmatory threshold. For scale, the best benchmark-fitted reference reaches 0.592
# against a 0.5 chance level, so the attainable range is about 0.09 wide.
DENSE_EQUIVALENCE_MARGIN = 0.03

CLUSTER_BOOTSTRAPS = ("murcko_cluster_bootstrap", "butina_cluster_bootstrap")


def interval_union(record: dict, names: tuple[str, ...], level: str = "interval_95") -> list[float]:
    intervals = [record[name][level] for name in names]
    return [min(value[0] for value in intervals), max(value[1] for value in intervals)]


def smallest_equivalence_margin(interval_90: list[float]) -> float:
    """Smallest symmetric margin passing two one-sided tests at alpha = 0.05.

    A symmetric-margin TOST at alpha rejects both one-sided nulls exactly when the
    (1 - 2*alpha) interval lies strictly inside the margin, so the 90% interval is the
    right object here. Rounded up at three decimals so the reported margin stays
    conservative with respect to the interval it is derived from.
    """
    half_width = max(abs(interval_90[0]), abs(interval_90[1]))
    return math.ceil(half_width * 1000.0) / 1000.0


def smallest_grid_margin(interval_90: list[float], step: float = 0.005) -> float:
    """Smallest strict-passing margin on a regular post-hoc sensitivity grid."""
    reach = max(abs(interval_90[0]), abs(interval_90[1]))
    return (math.floor(reach / step) + 1) * step


def macro(name: str, value: str) -> str:
    return f"\\newcommand{{\\{name}}}{{{value}}}"


def build(manuscript: dict) -> tuple[str, str, dict]:
    spectral = manuscript["spectral"]
    covariance_rows = {
        (row["dataset"], row["estimand"], row["surface"]): row
        for row in manuscript["covariance_sensitivity"]
    }
    missing_rows = {
        (row["analysis"], row["method_key"]): row
        for row in manuscript["missing_residual_sensitivity"]
    }
    fixed = manuscript["fixed20_estimand_decomposition"]["panels"]
    fixed_matched = manuscript["fixed20_estimand_decomposition"][
        "transformation_matched_independent_column_null"
    ]["panels"]
    cross = manuscript["experimental_cross_panel"]
    structural = manuscript["structural_controls"]
    recovery = manuscript["probe_panel_recovery"]
    ranking = manuscript["ligand_wise_ranking_boundary"]
    dense_ranking = manuscript["dense_ligand_wise_benchmarks"]
    overlap = manuscript["cross_panel_overlap"]
    experimental_overlap = manuscript["experimental_panel_overlap"]
    experimental_pair_overlap = {
        (row["first_panel"], row["second_panel"]): row
        for row in experimental_overlap["pairwise_structural_overlap"]
    }
    physicochemical = manuscript["physicochemical_interpretation"]
    chemical_domain = manuscript["exploratory_chemical_domain_structure"]
    descriptor_decomposition = chemical_domain["descriptor_residualization"][
        "datasets"
    ]
    mw_domain = chemical_domain["low_high_molecular_weight_domain_instability"][
        "datasets"
    ]
    descriptor_family = chemical_domain[
        "post_hoc_descriptor_family_stratification"
    ]["datasets"]
    descriptor_family_ranges: dict[str, dict[str, float]] = {}
    for dataset in ("Docking-44", "DOCKSTRING-58"):
        descriptor_records = descriptor_family[dataset]["descriptor_results"]
        size_values = [
            record["residual_map_spearman"]
            for record in descriptor_records.values()
            if record["family"] == "molecular_size"
        ]
        other_values = [
            record["residual_map_spearman"]
            for record in descriptor_records.values()
            if record["family"] != "molecular_size"
        ]
        descriptor_family_ranges[dataset] = {
            "size_low": min(size_values),
            "size_high": max(size_values),
            "other_low": min(other_values),
            "other_high": max(other_values),
        }
    fixed_reference = manuscript["fixed20_reference_sensitivity"]
    input_audits = manuscript["large_matrix_input_audits"]
    ds_clipping = input_audits["DOCKSTRING-58"][
        "positive_score_clipping_sensitivity"
    ]
    ds_mw_seed_range = mw_domain["DOCKSTRING-58"]["support_seed_sensitivity"][
        "geometry_spearman_range"
    ]
    measurement_audits = manuscript["experimental_measurement_audits"]
    davis_censoring = manuscript["davis_fixed20_censoring_sensitivity"]
    kirhub_locked = manuscript["kirhub_locked_endpoint"]
    kirhub_metrics = kirhub_locked["target_label_qap"]["metrics"]
    kirhub_holm = kirhub_locked["holm_adjusted_centered_minus_raw_gain_p"]
    matched_cross_null = cross["transformation_matched_independent_column_null"]
    matched_kirhub = matched_cross_null["locked_old_endpoint_validation"]
    ranking_panel = manuscript["ranking_centering_panel_sensitivity"]["ranking"]
    transport = manuscript["nonvina_scorer_transport"]
    gauss2_attribution = manuscript["vina_fixed_pose_term_attribution"][
        "term_attribution"
    ]["gauss2"]
    transport_observed = transport["observed"]
    transport_summary = transport["prediction_summary"]
    broad = manuscript["dockstring_chembl_ranking"]
    broad_accuracy = broad["mean_per_ligand_pairwise_accuracy"]
    broad_benchmark = broad["benchmark"]
    broad_representations = broad_benchmark["representations"]
    broad_retained_ligands = int(
        broad["support"].get("retained_ligands", broad_benchmark["n_ligands"])
    )
    broad_evaluated_ligands = int(
        broad["support"].get(
            "evaluated_ligands",
            broad_representations["absolute_vina"]["evaluated_ligands"],
        )
    )
    sparse = ranking
    sparse_representations = sparse["representations"]
    broad_contrast = broad["headline_contrasts"]["two_way_residual_minus_absolute_vina"]
    broad_clean_kikd = broad["sensitivities"]["human_binding_Ki_Kd"]
    broad_single_endpoints = broad["sensitivities"]["single_endpoint"]
    broad_clean_kikd_mixing = broad_clean_kikd["endpoint_mixing_diagnostic"]
    broad_clean_kikd_mixing_counts = broad_clean_kikd_mixing["pair_counts"]
    broad_clean_kikd_mixing_all = broad_clean_kikd_mixing[
        "fractions_of_all_informative_pairs"
    ]
    broad_clean_kikd_mixing_unambiguous = broad_clean_kikd_mixing[
        "fractions_among_endpoint_unambiguous_pairs"
    ]
    broad_clean_kikd_contrast = broad_clean_kikd["contrasts"][
        "two_way_residual_minus_absolute_vina"
    ]
    broad_clean_kikd_target_interval = broad_clean_kikd[
        "target_jackknife_paired_contrasts"
    ]["two_way_residual_minus_absolute_vina"]["jackknife_normal_95_interval"]
    descriptor_geometry = manuscript["descriptor_mediated_geometry"]
    descriptor_geometry_range = descriptor_geometry["range_across_panels"]
    descriptor_controls = manuscript["descriptor_rank_matched_controls"]
    descriptor_predictive = descriptor_controls["predictive_metrics"]
    descriptor_panel_geometry = descriptor_controls["panel_geometry_agreement"]
    descriptor_morgan_ensemble = descriptor_controls[
        "morgan_count_plus_size_rp_7_seed_ensemble"
    ]
    descriptor_morgan_reduction_distribution = descriptor_morgan_ensemble[
        "algorithmic_distributions"
    ]["relative_reduction_in_mean_squared_offdiagonal_target_correlation"]
    descriptor_design = manuscript[
        "descriptor_correlation_reduction_uncertainty"
    ]["sensitivity"]
    descriptor_nuisance = descriptor_controls[
        "physicochemical_row_permutation_nuisance_ensemble"
    ]["algorithmic_seed_distributions"]
    ranking_residual_absolute = ranking["paired_comparisons"][
        "two_way_residual_minus_absolute_vina"
    ]
    ranking_residual_column = ranking["paired_comparisons"][
        "two_way_residual_minus_column_standardized"
    ]
    ranking_residual_absolute_union = interval_union(
        ranking_residual_absolute,
        ("scaffold_cluster_bootstrap", "butina_cluster_bootstrap"),
    )
    ranking_residual_column_union = interval_union(
        ranking_residual_column,
        ("scaffold_cluster_bootstrap", "butina_cluster_bootstrap"),
    )
    ranking_panel_delta_union = interval_union(
        ranking_panel["local_38_minus_all_44"],
        ("scaffold_cluster_bootstrap", "butina_cluster_bootstrap"),
    )
    ranking_panel_absolute_union = interval_union(
        ranking_panel["local_38_minus_absolute_vina"],
        ("scaffold_cluster_bootstrap", "butina_cluster_bootstrap"),
    )
    sparse_cohort_interval = interval_union(
        sparse_representations["cohort_experimental_target_prior"],
        ("scaffold_cluster_bootstrap", "butina_cluster_bootstrap"),
    )
    sparse_docking_prior_interval = interval_union(
        sparse_representations["docking_target_prior"],
        ("scaffold_cluster_bootstrap", "butina_cluster_bootstrap"),
    )
    sparse_external_prior_interval = interval_union(
        sparse_representations["experimental_target_prior"],
        ("scaffold_cluster_bootstrap", "butina_cluster_bootstrap"),
    )
    broad_prior_intervals = {
        name: interval_union(
            broad_representations[name],
            ("scaffold_cluster_bootstrap", "butina_cluster_bootstrap"),
        )
        for name in (
            "docking_target_prior",
            "experimental_target_prior",
            "cohort_experimental_target_prior",
        )
    }
    broad_absolute_minus_docking_prior = broad_benchmark["paired_comparisons"][
        "absolute_vina_minus_docking_target_prior"
    ]
    broad_absolute_minus_docking_prior_union = interval_union(
        broad_absolute_minus_docking_prior,
        ("scaffold_cluster_bootstrap", "butina_cluster_bootstrap"),
    )
    sparse_absolute = sparse_representations["absolute_vina"]
    sparse_residual = sparse_representations["two_way_residual"]
    dense_intervals: dict[str, list[float]] = {}
    dense_prior_intervals: dict[str, list[float]] = {}
    dense_offset_intervals: dict[str, list[float]] = {}
    dense_scale_intervals: dict[str, list[float]] = {}
    dense_row_intervals: dict[str, list[float]] = {}
    for panel in ("DAVIS", "PKIS2"):
        dense_intervals[panel] = interval_union(
            dense_ranking[panel]["paired_comparisons"][
                "two_way_residual_minus_absolute_vina"
            ]["uncertainty"],
            ("murcko_cluster_bootstrap", "butina_cluster_bootstrap"),
        )
        dense_prior_intervals[panel] = interval_union(
            dense_ranking[panel]["paired_comparisons"][
                "absolute_vina_minus_docking_target_prior"
            ]["uncertainty"],
            ("murcko_cluster_bootstrap", "butina_cluster_bootstrap"),
        )
        dense_offset_intervals[panel] = interval_union(
            dense_ranking[panel]["paired_comparisons"][
                "target_centered_unscaled_minus_absolute_vina"
            ]["uncertainty"],
            ("murcko_cluster_bootstrap", "butina_cluster_bootstrap"),
        )
        dense_scale_intervals[panel] = interval_union(
            dense_ranking[panel]["paired_comparisons"][
                "target_centered_residual_scaled_minus_target_centered_unscaled"
            ]["uncertainty"],
            ("murcko_cluster_bootstrap", "butina_cluster_bootstrap"),
        )
        dense_row_intervals[panel] = interval_union(
            dense_ranking[panel]["paired_comparisons"][
                "two_way_residual_minus_target_centered_residual_scaled"
            ]["uncertainty"],
            ("murcko_cluster_bootstrap", "butina_cluster_bootstrap"),
        )
    macros = {
        # Sparse Docking-44 sensitivity.  Every macro is explicitly namespaced so these
        # 137-ligand values cannot be mistaken for the broad DOCKSTRING--ChEMBL result.
        "SparseRankingCohortPriorLow": f"{sparse_cohort_interval[0]:.3f}",
        "SparseRankingCohortPriorHigh": f"{sparse_cohort_interval[1]:.3f}",
        "SparseRankingDockingPriorLow": f"{sparse_docking_prior_interval[0]:.3f}",
        "SparseRankingDockingPriorHigh": f"{sparse_docking_prior_interval[1]:.3f}",
        "SparseRankingExternalPriorLow": f"{sparse_external_prior_interval[0]:.3f}",
        "SparseRankingExternalPriorHigh": f"{sparse_external_prior_interval[1]:.3f}",
        "SparseRankingDockingPriorAccuracy": f"{sparse_representations['docking_target_prior']['mean_per_ligand_pairwise_accuracy']:.3f}",
        "SparseRankingExternalPriorAccuracy": f"{sparse_representations['experimental_target_prior']['mean_per_ligand_pairwise_accuracy']:.3f}",
        "SparseRankingCohortPriorAccuracy": f"{sparse_representations['cohort_experimental_target_prior']['mean_per_ligand_pairwise_accuracy']:.3f}",
        "SparseRankingPairWeightedAbsolute": f"{sparse_absolute['pair_weighted_accuracy']:.3f}",
        "SparseRankingPairWeightedResidual": f"{sparse_residual['pair_weighted_accuracy']:.3f}",
        "SparseRankingPairWeightedDockingPrior": f"{sparse_representations['docking_target_prior']['pair_weighted_accuracy']:.3f}",
        "SparseRankingPairWeightedExternalPrior": f"{sparse_representations['experimental_target_prior']['pair_weighted_accuracy']:.3f}",
        "SparseRankingPairWeightedCohortPrior": f"{sparse_representations['cohort_experimental_target_prior']['pair_weighted_accuracy']:.3f}",
        "SparseRankingAbsoluteIdentityShuffleP": f"{sparse_absolute['ligand_identity_shuffle_null']['one_sided_empirical_p_observed_at_least_as_large']:.3f}",
        "SparseRankingResidualIdentityShuffleP": f"{sparse_residual['ligand_identity_shuffle_null']['one_sided_empirical_p_observed_at_least_as_large']:.3f}",
        "SparseRankingAbsoluteDeduplicatedMedianP": f"{sparse_absolute['one_ligand_per_murcko_cluster_identity_permutation']['one_sided_p_value_across_supports']['median']:.3f}",
        "SparseRankingResidualDeduplicatedMedianP": f"{sparse_residual['one_ligand_per_murcko_cluster_identity_permutation']['one_sided_p_value_across_supports']['median']:.3f}",
        "SparseRankingCoverageThreeDockingPrior": f"{sparse['coverage_sensitivity']['3']['representations']['docking_target_prior']['mean_per_ligand_pairwise_accuracy']:.3f}",
        "SparseRankingCoverageThreeCohortPrior": f"{sparse['coverage_sensitivity']['3']['representations']['cohort_experimental_target_prior']['mean_per_ligand_pairwise_accuracy']:.3f}",
        "DescriptorGeometryObservedLow": f"{descriptor_geometry_range['observed_residual'][0]:.3f}",
        "DescriptorGeometryObservedHigh": f"{descriptor_geometry_range['observed_residual'][1]:.3f}",
        "DescriptorGeometryComponentLow": f"{descriptor_geometry_range['descriptor_component'][0]:.3f}",
        "DescriptorGeometryComponentHigh": f"{descriptor_geometry_range['descriptor_component'][1]:.3f}",
        "DescriptorGeometryRemovedLow": f"{descriptor_geometry_range['descriptor_removed'][0]:.3f}",
        "DescriptorGeometryRemovedHigh": f"{descriptor_geometry_range['descriptor_removed'][1]:.3f}",
        "DescriptorGeometryLigands": f"{descriptor_geometry['support']['dockstring_reference_rows']:,}",
        "DescriptorGeometryCorrelationReduction": (
            f"{100 * descriptor_geometry['decomposition_metrics']['out_of_fold_relative_mean_squared_target_correlation_reduction']:.1f}\\%"
        ),
        "FixedPhyschemOOFRtwo": (
            f"{descriptor_predictive['physicochemical_7']['mean_out_of_fold_target_r2']:.3f}"
        ),
        "FixedPhyschemCorrelationReduction": (
            f"{descriptor_predictive['physicochemical_7']['relative_reduction_in_mean_squared_offdiagonal_target_correlation']:.3f}"
        ),
        "FixedMorganOOFRtwo": (
            f"{descriptor_predictive['morgan_count_plus_size_rp_7_locked_seed_20260910']['mean_out_of_fold_target_r2']:.3f}"
        ),
        "FixedMorganCorrelationReduction": (
            f"{descriptor_predictive['morgan_count_plus_size_rp_7_locked_seed_20260910']['relative_reduction_in_mean_squared_offdiagonal_target_correlation']:.3f}"
        ),
        "FixedMorganReductionLow": (
            f"{descriptor_morgan_reduction_distribution['count_morgan_seed_minimum']:.3f}"
        ),
        "FixedMorganReductionHigh": (
            f"{descriptor_morgan_reduction_distribution['count_morgan_seed_maximum']:.3f}"
        ),
        "FixedMorganSeedsExceedPhyschem": str(
            descriptor_morgan_reduction_distribution[
                "ensemble_members_at_least_as_large_as_physicochemical"
            ]
        ),
        "FixedHashOOFRtwo": (
            f"{descriptor_predictive['stable_hash_nuisance_7']['mean_out_of_fold_target_r2']:.5f}"
        ),
        "FixedHashCorrelationReduction": (
            f"{descriptor_predictive['stable_hash_nuisance_7']['relative_reduction_in_mean_squared_offdiagonal_target_correlation']:.5f}"
        ),
        "FixedHashGeometryLow": (
            f"{min(descriptor_panel_geometry['stable_hash_nuisance_7'].values()):.3f}"
        ),
        "FixedHashGeometryHigh": (
            f"{max(descriptor_panel_geometry['stable_hash_nuisance_7'].values()):.3f}"
        ),
        "FixedPhyschemGeometryLow": (
            f"{min(descriptor_panel_geometry['physicochemical_7'].values()):.3f}"
        ),
        "FixedPhyschemGeometryHigh": (
            f"{max(descriptor_panel_geometry['physicochemical_7'].values()):.3f}"
        ),
        "NuisanceOOFRtwoLow": (
            f"{descriptor_nuisance['mean_out_of_fold_target_r2']['minimum']:.5f}"
        ),
        "NuisanceOOFRtwoHigh": (
            f"{descriptor_nuisance['mean_out_of_fold_target_r2']['maximum']:.5f}"
        ),
        "NuisanceCorrelationReductionLow": (
            f"{descriptor_nuisance['relative_reduction_in_mean_squared_offdiagonal_target_correlation']['minimum']:.5f}"
        ),
        "NuisanceCorrelationReductionHigh": (
            f"{descriptor_nuisance['relative_reduction_in_mean_squared_offdiagonal_target_correlation']['maximum']:.5f}"
        ),
        "NuisanceGeometryLow": (
            f"{min(record['minimum'] for record in descriptor_nuisance['panel_geometry_agreement'].values()):.3f}"
        ),
        "NuisanceGeometryHigh": (
            f"{max(record['maximum'] for record in descriptor_nuisance['panel_geometry_agreement'].values()):.3f}"
        ),
        "DffDescriptorDesignLow": (
            f"{descriptor_design['Docking-44']['all_repeated_group_partitions']['empirical_interval_95'][0]:.3f}"
        ),
        "DffDescriptorDesignHigh": (
            f"{descriptor_design['Docking-44']['all_repeated_group_partitions']['empirical_interval_95'][1]:.3f}"
        ),
        "DsDescriptorDesignLow": (
            f"{descriptor_design['DOCKSTRING-58']['all_repeated_group_partitions']['empirical_interval_95'][0]:.3f}"
        ),
        "DsDescriptorDesignHigh": (
            f"{descriptor_design['DOCKSTRING-58']['all_repeated_group_partitions']['empirical_interval_95'][1]:.3f}"
        ),
        "TransportLigands": f"{transport['support']['n_ligands']:,}",
        "TransportTargets": str(transport["support"]["n_targets"]),
        "TransportCells": f"{transport['support']['n_cells']:,}",
        "TransportNonVinaScorers": str(transport["support"]["n_non_vina_scorers"]),
        "TransportSupportingScorers": str(
            len(transport_summary["non_vina_scorers_supporting_the_prediction"])
        ),
        "TransportSminaReproduction": (
            f"{transport['smina_vina_reproduction_check']['minimum_pearson_r']:.3f}"
        ),
        "VinaGaussTwoSharedAxisShare": (
            f"{gauss2_attribution['shared_row_axis_covariance_share']:.3f}"
        ),
        "VinaGaussTwoResidualInnerProductShare": (
            f"{gauss2_attribution['residual_frobenius_inner_product_share']:.3f}"
        ),
        "TransportVinaRawPR": (
            f"{transport_observed['released_vina']['raw_participation_ratio']:.2f}"
        ),
        "TransportVinaRawPCOne": (
            f"{100 * transport_observed['released_vina']['raw_pc1_fraction']:.1f}\\%"
        ),
        "TransportVinardoRawPR": (
            f"{transport_observed['vinardo']['raw_participation_ratio']:.2f}"
        ),
        "TransportVinardoRawPCOne": (
            f"{100 * transport_observed['vinardo']['raw_pc1_fraction']:.1f}\\%"
        ),
        "TransportPlecRawPR": (
            f"{transport_observed['plecscore_linear']['raw_participation_ratio']:.2f}"
        ),
        "TransportPlecRawPCOne": (
            f"{100 * transport_observed['plecscore_linear']['raw_pc1_fraction']:.1f}\\%"
        ),
        "TransportPlecUniformCosine": (
            f"{transport_observed['plecscore_linear']['pc1_uniform_cosine']:.3f}"
        ),
        "DsUnclippedResidualPR": (
            f"{ds_clipping['unclipped']['residual_pr']:.3f}"
        ),
        "DsMWDomainSeedLow": f"{ds_mw_seed_range[0]:.3f}",
        "DsMWDomainSeedHigh": f"{ds_mw_seed_range[1]:.3f}",
        "SparseRankingRetainedLigands": f"{sparse['n_ligands']:,}",
        "SparseRankingEvaluatedLigands": f"{sparse_absolute['evaluated_ligands']:,}",
        "SparseRankingTargets": str(sparse["n_targets"]),
        "SparseRankingCells": f"{sparse['observed_experimental_cells']:,}",
        "SparseRankingPairs": f"{sparse_absolute['evaluated_pairs']:,}",
        "SparseRankingAbsolute": f"{sparse_absolute['mean_per_ligand_pairwise_accuracy']:.3f}",
        "SparseRankingColumn": f"{sparse_representations['column_standardized']['mean_per_ligand_pairwise_accuracy']:.3f}",
        "SparseRankingResidual": f"{sparse_residual['mean_per_ligand_pairwise_accuracy']:.3f}",
        "BroadRankingRetainedLigands": f"{broad_retained_ligands:,}",
        "BroadRankingEvaluatedLigands": f"{broad_evaluated_ligands:,}",
        "BroadRankingUninformativeLigands": f"{broad_retained_ligands - broad_evaluated_ligands:,}",
        "BroadRankingTargets": str(broad["support"]["targets"]),
        "BroadRankingCells": f"{broad['support']['observed_experimental_cells']:,}",
        "BroadRankingPairs": (
            f"{broad['support']['non_tied_within_ligand_target_pairs']:,}"
        ),
        "BroadRankingAbsolute": f"{broad_accuracy['absolute_vina']:.3f}",
        "BroadRankingResidual": f"{broad_accuracy['two_way_residual']:.3f}",
        "BroadRankingColumn": f"{broad_accuracy['column_standardized']:.3f}",
        "BroadRankingDockingPrior": f"{broad_accuracy['docking_target_prior']:.3f}",
        "BroadRankingExternalPrior": f"{broad_accuracy['experimental_target_prior']:.3f}",
        "BroadRankingCohortPrior": (
            f"{broad_accuracy['cohort_experimental_target_prior']:.3f}"
        ),
        "BroadRankingDockingPriorLow": f"{broad_prior_intervals['docking_target_prior'][0]:.3f}",
        "BroadRankingDockingPriorHigh": f"{broad_prior_intervals['docking_target_prior'][1]:.3f}",
        "BroadRankingExternalPriorLow": f"{broad_prior_intervals['experimental_target_prior'][0]:.3f}",
        "BroadRankingExternalPriorHigh": f"{broad_prior_intervals['experimental_target_prior'][1]:.3f}",
        "BroadRankingCohortPriorLow": f"{broad_prior_intervals['cohort_experimental_target_prior'][0]:.3f}",
        "BroadRankingCohortPriorHigh": f"{broad_prior_intervals['cohort_experimental_target_prior'][1]:.3f}",
        "BroadRankingAbsoluteMinusDockingPrior": (
            f"{broad_absolute_minus_docking_prior['plugin_mean_difference']:.3f}"
        ),
        "BroadRankingAbsoluteMinusDockingPriorLow": (
            f"{broad_absolute_minus_docking_prior_union[0]:.3f}"
        ),
        "BroadRankingAbsoluteMinusDockingPriorHigh": (
            f"{broad_absolute_minus_docking_prior_union[1]:.3f}"
        ),
        "BroadRankingDelta": f"{broad_contrast['plugin_mean_difference']:.3f}",
        "BroadRankingEquivalenceLow": (
            f"{broad_contrast['conservative_cluster_bootstrap_interval_90'][0]:.3f}"
        ),
        "BroadRankingEquivalenceHigh": (
            f"{broad_contrast['conservative_cluster_bootstrap_interval_90'][1]:.3f}"
        ),
        "BroadRankingSamplingLow": (
            f"{broad_contrast['conservative_cluster_bootstrap_interval_95'][0]:.3f}"
        ),
        "BroadRankingSamplingHigh": (
            f"{broad_contrast['conservative_cluster_bootstrap_interval_95'][1]:.3f}"
        ),
        "BroadRankingTargetJackknifeLow": (
            f"{broad_benchmark['target_jackknife_paired_contrasts']['two_way_residual_minus_absolute_vina']['jackknife_normal_95_interval'][0]:.3f}"
        ),
        "BroadRankingTargetJackknifeHigh": (
            f"{broad_benchmark['target_jackknife_paired_contrasts']['two_way_residual_minus_absolute_vina']['jackknife_normal_95_interval'][1]:.3f}"
        ),
        "BroadHumanKiKdRetainedLigands": (
            f"{broad_clean_kikd['support']['retained_ligands']:,}"
        ),
        "BroadHumanKiKdEvaluatedLigands": (
            f"{broad_clean_kikd['support']['evaluated_ligands']:,}"
        ),
        "BroadHumanKiKdTargets": str(broad_clean_kikd["support"]["targets"]),
        "BroadHumanKiKdCells": f"{broad_clean_kikd['support']['observed_cells']:,}",
        "BroadHumanKiKdPairs": (
            f"{broad_clean_kikd['support']['non_tied_within_ligand_pairs']:,}"
        ),
        "BroadHumanKiKdAbsolute": (
            f"{broad_clean_kikd['mean_per_ligand_pairwise_accuracy']['absolute_vina']:.3f}"
        ),
        "BroadHumanKiKdColumn": (
            f"{broad_clean_kikd['mean_per_ligand_pairwise_accuracy']['column_standardized']:.3f}"
        ),
        "BroadHumanKiKdResidual": (
            f"{broad_clean_kikd['mean_per_ligand_pairwise_accuracy']['two_way_residual']:.3f}"
        ),
        "BroadHumanKiKdDelta": (
            f"{broad_clean_kikd_contrast['plugin_mean_difference']:.3f}"
        ),
        "BroadHumanKiKdSamplingLow": (
            f"{broad_clean_kikd_contrast['conservative_cluster_bootstrap_interval_95'][0]:.3f}"
        ),
        "BroadHumanKiKdSamplingHigh": (
            f"{broad_clean_kikd_contrast['conservative_cluster_bootstrap_interval_95'][1]:.3f}"
        ),
        "BroadHumanKiKdTargetJackknifeLow": (
            f"{broad_clean_kikd_target_interval[0]:.3f}"
        ),
        "BroadHumanKiKdTargetJackknifeHigh": (
            f"{broad_clean_kikd_target_interval[1]:.3f}"
        ),
        "BroadHumanKiKdEndpointUnambiguousPairs": (
            f"{broad_clean_kikd_mixing['endpoint_unambiguous_pairs']:,}"
        ),
        "BroadHumanKiKdSameEndpointPairs": (
            f"{broad_clean_kikd_mixing_counts['same_endpoint']:,}"
        ),
        "BroadHumanKiKdSameKiPairs": (
            f"{broad_clean_kikd_mixing_counts['same_endpoint_Ki_Ki']:,}"
        ),
        "BroadHumanKiKdSameKdPairs": (
            f"{broad_clean_kikd_mixing_counts['same_endpoint_Kd_Kd']:,}"
        ),
        "BroadHumanKiKdMixedEndpointPairs": (
            f"{broad_clean_kikd_mixing_counts['mixed_endpoint_Ki_Kd']:,}"
        ),
        "BroadHumanKiKdPooledCellPairs": (
            f"{broad_clean_kikd_mixing_counts['involves_pooled_Ki_Kd_cell']:,}"
        ),
        "BroadHumanKiKdSameEndpointPercent": (
            f"{100 * broad_clean_kikd_mixing_all['same_endpoint']:.1f}\\%"
        ),
        "BroadHumanKiKdMixedEndpointPercent": (
            f"{100 * broad_clean_kikd_mixing_all['mixed_endpoint_Ki_Kd']:.1f}\\%"
        ),
        "BroadHumanKiKdPooledCellPercent": (
            f"{100 * broad_clean_kikd_mixing_all['involves_pooled_Ki_Kd_cell']:.1f}\\%"
        ),
        "BroadHumanKiKdSameAmongUnambiguousPercent": (
            f"{100 * broad_clean_kikd_mixing_unambiguous['same_endpoint']:.1f}\\%"
        ),
        "BroadRankingAttainedEquivalenceMargin": (
            f"{smallest_equivalence_margin(broad_contrast['conservative_cluster_bootstrap_interval_90']):.3f}"
        ),
        "BroadRankingSmallestGridMargin": (
            f"{smallest_grid_margin(broad_contrast['conservative_cluster_bootstrap_interval_90']):.3f}"
        ),
        # Broad priors must come from the broad object, never the 137-ligand sensitivity.
        "BroadDockingPriorAccuracy": f"{broad_accuracy['docking_target_prior']:.3f}",
        "BroadExternalPriorAccuracy": f"{broad_accuracy['experimental_target_prior']:.3f}",
        "BroadCohortPriorAccuracy": f"{broad_accuracy['cohort_experimental_target_prior']:.3f}",
        "DenseEquivalenceMargin": f"{DENSE_EQUIVALENCE_MARGIN:.2f}",
    }
    for panel, prefix in (("DAVIS", "DenseDavis"), ("PKIS2", "DensePkisTwo")):
        record = dense_ranking[panel]
        residual_contrast = record["paired_comparisons"][
            "two_way_residual_minus_absolute_vina"
        ]
        scale_contrast = record["paired_comparisons"][
            "target_centered_residual_scaled_minus_target_centered_unscaled"
        ]
        row_contrast = record["paired_comparisons"][
            "two_way_residual_minus_target_centered_residual_scaled"
        ]
        prior_contrast = record["paired_comparisons"][
            "absolute_vina_minus_docking_target_prior"
        ]
        offset_contrast = record["paired_comparisons"][
            "target_centered_unscaled_minus_absolute_vina"
        ]
        target_interval_key = (
            "jackknife_bias_corrected_normal_95_interval"
            if panel == "DAVIS"
            else "jackknife_normal_95_interval"
        )
        target_interval = record["target_composition_sensitivity"][
            target_interval_key
        ]
        macros.update(
            {
                f"{prefix}MatchedLigands": str(record["support"]["matched_ligands"]),
                f"{prefix}EvaluatedLigands": str(record["support"]["evaluated_ligands"]),
                f"{prefix}Targets": str(record["support"]["targets"]),
                f"{prefix}Pairs": f"{record['support']['evaluated_pairs']:,}",
                f"{prefix}Absolute": f"{record['representations']['absolute_vina']['mean_per_ligand_pairwise_concordance']:.3f}",
                f"{prefix}TargetPrior": f"{record['representations']['docking_target_prior']['mean_per_ligand_pairwise_concordance']:.3f}",
                f"{prefix}TargetPriorSpearman": f"{record['representations']['docking_target_prior']['mean_within_ligand_spearman']:.3f}",
                f"{prefix}TargetPriorTopOne": f"{record['representations']['docking_target_prior']['top1_accuracy_allowing_experimental_ties']:.3f}",
                f"{prefix}Residual": f"{record['representations']['two_way_residual']['mean_per_ligand_pairwise_concordance']:.3f}",
                f"{prefix}AbsolutePriorDelta": f"{prior_contrast['plugin_mean_difference']:.3f}",
                f"{prefix}AbsolutePriorDeltaLow": f"{dense_prior_intervals[panel][0]:.3f}",
                f"{prefix}AbsolutePriorDeltaHigh": f"{dense_prior_intervals[panel][1]:.3f}",
                f"{prefix}OffsetStep": f"{offset_contrast['plugin_mean_difference']:.3f}",
                f"{prefix}OffsetStepLow": f"{dense_offset_intervals[panel][0]:.3f}",
                f"{prefix}OffsetStepHigh": f"{dense_offset_intervals[panel][1]:.3f}",
                f"{prefix}Delta": f"{residual_contrast['plugin_mean_difference']:.3f}",
                f"{prefix}DeltaLow": f"{dense_intervals[panel][0]:.3f}",
                f"{prefix}DeltaHigh": f"{dense_intervals[panel][1]:.3f}",
                f"{prefix}ScalePenalty": f"{scale_contrast['plugin_mean_difference']:.3f}",
                f"{prefix}ScalePenaltyLow": f"{dense_scale_intervals[panel][0]:.3f}",
                f"{prefix}ScalePenaltyHigh": f"{dense_scale_intervals[panel][1]:.3f}",
                f"{prefix}RowCorrection": f"{row_contrast['plugin_mean_difference']:.3f}",
                f"{prefix}RowCorrectionLow": f"{dense_row_intervals[panel][0]:.3f}",
                f"{prefix}RowCorrectionHigh": f"{dense_row_intervals[panel][1]:.3f}",
                f"{prefix}TargetJackknifeLow": f"{target_interval[0]:.3f}",
                f"{prefix}TargetJackknifeHigh": f"{target_interval[1]:.3f}",
            }
        )
        # Equivalence reading of the same paired contrast (reviewer request M6). The 90%
        # conservative union is the TOST-relevant interval at alpha = 0.05.
        equivalence_interval = interval_union(
            residual_contrast["uncertainty"], CLUSTER_BOOTSTRAPS, level="interval_90"
        )
        attained_margin = smallest_equivalence_margin(equivalence_interval)
        exact_half_width = max(
            abs(equivalence_interval[0]), abs(equivalence_interval[1])
        )
        macros.update(
            {
                f"{prefix}EquivalenceLow": f"{equivalence_interval[0]:.3f}",
                f"{prefix}EquivalenceHigh": f"{equivalence_interval[1]:.3f}",
                # One-sided 95% upper confidence bound on any residual-rule gain.
                f"{prefix}GainUpperBound": f"{equivalence_interval[1]:.3f}",
                f"{prefix}EquivalenceHalfWidthExact": f"{exact_half_width:.6f}",
                f"{prefix}AttainedEquivalenceMargin": f"{attained_margin:.3f}",
                f"{prefix}SmallestGridMargin": f"{smallest_grid_margin(equivalence_interval):.3f}",
                f"{prefix}EquivalenceEstablished": (
                    "yes" if exact_half_width < DENSE_EQUIVALENCE_MARGIN else "no"
                ),
            }
        )
    davis_binder = dense_ranking["DAVIS"]["outcome_conditioned_strata"][
        "both_uncensored"
    ]
    davis_floor = dense_ranking["DAVIS"]["outcome_conditioned_strata"][
        "floor_vs_uncensored"
    ]
    pkis_active = dense_ranking["PKIS2"]["outcome_conditioned_strata"][
        "both_active"
    ]
    for block, stem in (
        (davis_binder, "DenseDavisBinder"),
        (davis_floor, "DenseDavisFloorDetection"),
        (pkis_active, "DensePkisTwoBothActive"),
    ):
        contrast = block["paired_comparisons"][
            "two_way_residual_minus_absolute_vina"
        ]
        contrast_interval = interval_union(
            contrast["uncertainty"],
            ("murcko_cluster_bootstrap", "butina_cluster_bootstrap"),
        )
        macros.update(
            {
                f"{stem}Ligands": str(
                    block["representations"]["absolute_vina"]["evaluated_ligands"]
                ),
                f"{stem}Pairs": f"{block['representations']['absolute_vina']['evaluated_pairs']:,}",
                f"{stem}Absolute": f"{block['representations']['absolute_vina']['mean_per_ligand_pairwise_concordance']:.3f}",
                f"{stem}Residual": f"{block['representations']['two_way_residual']['mean_per_ligand_pairwise_concordance']:.3f}",
                f"{stem}Delta": f"{contrast['plugin_mean_difference']:.3f}",
                f"{stem}DeltaLow": f"{contrast_interval[0]:.3f}",
                f"{stem}DeltaHigh": f"{contrast_interval[1]:.3f}",
            }
        )
    macros.update(
        {
            "DffRawPR": f"{spectral['Docking-44']['correlation']['raw']['participation_ratio']:.3f}",
            "DffResidualPR": f"{spectral['Docking-44']['correlation']['residual']['participation_ratio']:.3f}",
            "DsRawPR": f"{spectral['DOCKSTRING-58']['correlation']['raw']['participation_ratio']:.3f}",
            "DsResidualPR": f"{spectral['DOCKSTRING-58']['correlation']['residual']['participation_ratio']:.3f}",
            "DffRawCovariancePR": f"{covariance_rows[('Docking-44', 'covariance', 'raw')]['participation_ratio']:.3f}",
            "DffResidualCovariancePR": f"{covariance_rows[('Docking-44', 'covariance', 'residual')]['participation_ratio']:.3f}",
            "DsRawCovariancePR": f"{covariance_rows[('DOCKSTRING-58', 'covariance', 'raw')]['participation_ratio']:.3f}",
            "DsResidualCovariancePR": f"{covariance_rows[('DOCKSTRING-58', 'covariance', 'residual')]['participation_ratio']:.3f}",
            "DffMedianImputedResidualPR": f"{missing_rows[('all_targets', 'target_median')]['residual_participation_ratio']:.3f}",
            "DffObservedALSResidualPR": f"{missing_rows[('all_targets', 'observed_additive_least_squares')]['residual_participation_ratio']:.3f}",
            "DffCompleteCaseResidualPR": f"{missing_rows[('all_targets', 'complete_case')]['residual_participation_ratio']:.3f}",
            "DffRawMeanAbsR": f"{spectral['Docking-44']['correlation']['raw']['mean_absolute_offdiagonal_correlation']:.3f}",
            "DffResidualMeanAbsR": f"{spectral['Docking-44']['correlation']['residual']['mean_absolute_offdiagonal_correlation']:.3f}",
            "DsRawMeanAbsR": f"{spectral['DOCKSTRING-58']['correlation']['raw']['mean_absolute_offdiagonal_correlation']:.3f}",
            "DsResidualMeanAbsR": f"{spectral['DOCKSTRING-58']['correlation']['residual']['mean_absolute_offdiagonal_correlation']:.3f}",
            "DffPCOne": f"{100 * spectral['Docking-44']['correlation']['raw']['pc1_fraction']:.1f}\\%",
            "DsPCOne": f"{100 * spectral['DOCKSTRING-58']['correlation']['raw']['pc1_fraction']:.1f}\\%",
            "DffUniformCosine": f"{spectral['Docking-44']['pc1_axis']['cosine_with_uniform_target_vector']:.3f}",
            "DsUniformCosine": f"{spectral['DOCKSTRING-58']['pc1_axis']['cosine_with_uniform_target_vector']:.3f}",
            "DffCommonNullObservedPR": f"{spectral['Docking-44']['row_norm_preserving_null']['observed_residual']:.3f}",
            "DsCommonNullObservedPR": f"{spectral['DOCKSTRING-58']['row_norm_preserving_null']['observed_residual']:.3f}",
            "DffRowNull": f"{spectral['Docking-44']['row_norm_preserving_null']['null_residual']['median']:.3f}",
            "DsRowNull": f"{spectral['DOCKSTRING-58']['row_norm_preserving_null']['null_residual']['median']:.3f}",
            "CrossPanelRaw": f"{cross['raw_mean_pairwise_spearman']:.3f}",
            "CrossPanelCentered": f"{cross['centered_mean_pairwise_spearman']:.3f}",
            "CrossPanelMatchedNullP": f"{cross['transformation_matched_independent_column_null']['p_observed_gain_at_least_as_large']:.3f}",
            "CrossPanelMatchedAbsoluteCenteredP": f"{matched_cross_null['p_observed_centered_at_least_as_large']:.5f}",
            "CrossPanelMatchedRawNullMean": f"{matched_cross_null['null_raw_mean']:.4f}",
            # Reviewer request M4b: report the observed centring gain beside the purely
            # mechanical one, so the reader can see the ratio without reconstructing it.
            "CrossPanelObservedGain": f"{cross['centered_minus_raw']:.3f}",
            "CrossPanelMechanicalGain": (
                f"{matched_cross_null['null_centered_minus_raw_mean']:.3f}"
            ),
            "CrossPanelGainOverMechanical": (
                f"{cross['centered_minus_raw'] / matched_cross_null['null_centered_minus_raw_mean']:.2f}"
            ),
            "CrossPanelMatchedCenteredNullMean": f"{matched_cross_null['null_centered_mean']:.3f}",
            "CrossPanelGlobalGainP": f"{cross['paired_global_network_qap']['paired_global_network_qap_p_positive_gain']:.4f}",
            "KiRHubLockedAUROC": f"{manuscript['kirhub_locked_endpoint']['KiRHub_validation']['centered_KiRHub_geometry']['roc_auc']:.3f}",
            "KiRHubLockedAP": f"{manuscript['kirhub_locked_endpoint']['KiRHub_validation']['centered_KiRHub_geometry']['average_precision']:.3f}",
            "KiRHubGainAUROC": f"{kirhub_metrics['roc_auc']['centered_minus_raw_KiRHub_geometry']:.3f}",
            "KiRHubGainAP": f"{kirhub_metrics['average_precision']['centered_minus_raw_KiRHub_geometry']:.3f}",
            "KiRHubGainAUCLabelP": f"{kirhub_metrics['roc_auc']['paired_target_label_qap_p_positive_KiRHub_geometry_gain']:.5f}",
            "KiRHubGainAPLabelP": f"{kirhub_metrics['average_precision']['paired_target_label_qap_p_positive_KiRHub_geometry_gain']:.5f}",
            "KiRHubGainAUCHolmP": f"{kirhub_holm['roc_auc']:.5f}",
            "KiRHubGainAPHolmP": f"{kirhub_holm['average_precision']:.5f}",
            "KiRHubGainAUCMatchedP": f"{matched_kirhub['roc_auc']['p_observed_gain_at_least_as_large']:.3f}",
            "KiRHubGainAPMatchedP": f"{matched_kirhub['average_precision']['p_observed_gain_at_least_as_large']:.3f}",
            "KiRHubAbsoluteMatchedP": f"{matched_kirhub['roc_auc']['p_observed_centered_at_least_as_large']:.5f}",
            "DffFrozenZeroCells": str(input_audits['Docking-44']['cells_exactly_zero_in_frozen_export']),
            "DsReleaseRows": f"{input_audits['DOCKSTRING-58']['release_rows']:,}",
            "DsAnalysisRows": f"{input_audits['DOCKSTRING-58']['analysis_rows']:,}",
            "DsIncompleteRows": str(input_audits['DOCKSTRING-58']['source_rows_with_any_missing']),
            "DsMissingCells": str(input_audits['DOCKSTRING-58']['source_missing_cells']),
            "DsMissingPercent": f"{100 * input_audits['DOCKSTRING-58']['source_missing_fraction']:.5f}\\%",
            "DsIncompleteRowPercent": f"{100 * input_audits['DOCKSTRING-58']['source_rows_with_any_missing_fraction']:.4f}\\%",
            "DsReleasePositiveCells": f"{input_audits['DOCKSTRING-58']['source_strictly_positive_cells']:,}",
            "DsAnalysisPositiveCells": f"{input_audits['DOCKSTRING-58']['analysis_strictly_positive_cells']:,}",
            "DsMeanImputedRawPR": f"{input_audits['DOCKSTRING-58']['all_rows_after_imputation']['target_mean']['raw_participation_ratio']:.3f}",
            "DsMeanImputedResidualPR": f"{input_audits['DOCKSTRING-58']['all_rows_after_imputation']['target_mean']['residual_participation_ratio']:.3f}",
            "DsMedianImputedRawPR": f"{input_audits['DOCKSTRING-58']['all_rows_after_imputation']['target_median']['raw_participation_ratio']:.3f}",
            "DsMedianImputedResidualPR": f"{input_audits['DOCKSTRING-58']['all_rows_after_imputation']['target_median']['residual_participation_ratio']:.3f}",
            "DavisFloorCells": str(measurement_audits['DAVIS']['cells_at_pkd_floor']),
            "DavisFloorPercent": f"{100 * measurement_audits['DAVIS']['fraction_at_pkd_floor']:.1f}\\%",
            "DavisFloorRange": f"{100 * measurement_audits['DAVIS']['targetwise_fraction_at_floor_range'][0]:.1f}--{100 * measurement_audits['DAVIS']['targetwise_fraction_at_floor_range'][1]:.1f}\\%",
            "DavisHighFloorTargets": str(measurement_audits['DAVIS']['targets_above_90_percent_at_floor']),
            "KiRHubCeilingPercent": f"{100 * measurement_audits['KiRHub']['exact_100_ceiling_fraction']:.1f}\\%",
            "DavisStatusGeometryRho": f"{davis_censoring['continuous_vs_above_floor_status_geometry_spearman']:.3f}",
            "DavisStatusDockingRho": f"{davis_censoring['centered_docking_vs_above_floor_status_davis']['spearman']:.3f}",
            "DavisStatusDockingP": f"{davis_censoring['centered_docking_vs_above_floor_status_davis']['one_sided_target_label_qap_p_positive']:.5f}",
            "DavisCensorAltShared": str(davis_censoring['alternative_endpoint']['shared_positive_pairs']),
            "DavisCensorAltTotal": str(davis_censoring['alternative_endpoint']['positive_pairs']),
            "DavisCensorAltAUROC": f"{davis_censoring['alternative_endpoint']['KiRHub_centered_geometry_validation']['roc_auc']:.3f}",
            "DavisCensorAltAP": f"{davis_censoring['alternative_endpoint']['KiRHub_centered_geometry_validation']['average_precision']:.3f}",
            "ThreePanelPartialRho": f"{structural['unrestricted_partial_qap']['three_panel_mean']['partial_spearman']:.3f}",
            "ThreePanelUnrestrictedP": f"{structural['unrestricted_partial_qap']['three_panel_mean']['target_label_qap_p_positive']:.3f}",
            "ThreePanelRestrictedP": f"{structural['within_KLIFS_group_partial_qap']['three_panel_mean']['restricted_qap_p_positive']:.3f}",
            "KiRHubPartialRho": f"{structural['unrestricted_partial_qap']['KiRHub']['partial_spearman']:.3f}",
            "KiRHubUnrestrictedP": f"{structural['unrestricted_partial_qap']['KiRHub']['target_label_qap_p_positive']:.4f}",
            "KiRHubRestrictedP": f"{structural['within_KLIFS_group_partial_qap']['KiRHub']['restricted_qap_p_positive']:.3f}",
            "DsProbeTwoHundred": f"{recovery['DOCKSTRING-58']['200']['geometry_spearman_mean']:.3f}",
            "DffProbeTwoHundred": f"{recovery['Docking-44']['200']['geometry_spearman_mean']:.3f}",
            "DsProbeFiveHundred": f"{recovery['DOCKSTRING-58']['500']['geometry_spearman_mean']:.3f}",
            "DffProbeFiveHundred": f"{recovery['Docking-44']['500']['geometry_spearman_mean']:.3f}",
            "RankingAbsolute": f"{ranking['representations']['absolute_vina']['mean_per_ligand_pairwise_accuracy']:.3f}",
            "RankingColumn": f"{ranking['representations']['column_standardized']['mean_per_ligand_pairwise_accuracy']:.3f}",
            "RankingResidual": f"{ranking['representations']['two_way_residual']['mean_per_ligand_pairwise_accuracy']:.3f}",
            "RankingResidualAbsoluteLow": f"{ranking_residual_absolute_union[0]:.3f}",
            "RankingResidualAbsoluteHigh": f"{ranking_residual_absolute_union[1]:.3f}",
            "RankingResidualColumnLow": f"{ranking_residual_column_union[0]:.3f}",
            "RankingResidualColumnHigh": f"{ranking_residual_column_union[1]:.3f}",
            "RankingLocalThirtyEight": f"{ranking_panel['local_38_row_mean']['mean_per_ligand_pairwise_accuracy']:.3f}",
            "RankingLocalThirtyEightDelta": f"{ranking_panel['local_38_minus_all_44']['plugin_mean_difference']:.3f}",
            "RankingLocalThirtyEightDeltaLow": f"{ranking_panel_delta_union[0]:.4f}",
            "RankingLocalThirtyEightDeltaHigh": f"{ranking_panel_delta_union[1]:.4f}",
            "RankingLocalThirtyEightAbsoluteLow": f"{ranking_panel_absolute_union[0]:.3f}",
            "RankingLocalThirtyEightAbsoluteHigh": f"{ranking_panel_absolute_union[1]:.3f}",
            "RankingPanelChangedPairs": str(ranking_panel["changed_pair_predictions"]),
            "RankingPanelChangedLigands": str(ranking_panel["ligands_with_any_changed_pair_prediction"]),
            "SharedFullInchiKeys": str(overlap['chemical_overlap']['shared_full_standard_inchikeys']),
            "SharedConnectivityBlocks": str(overlap['chemical_overlap']['shared_connectivity_blocks']),
            "SharedMurckoScaffolds": str(overlap['chemical_overlap']['shared_nonempty_murcko_scaffolds']),
            "SharedTargetIdentities": str(overlap['target_overlap']['mapped_target_identities']),
            "SharedReceptorStructures": str(overlap['target_overlap']['same_receptor_structure_used']),
            "DavisPkisTwoConnectivityOverlap": str(experimental_pair_overlap[("DAVIS", "PKIS2")]["shared_connectivity_blocks"]),
            "DavisPkisOneConnectivityOverlap": str(experimental_pair_overlap[("DAVIS", "PKIS1")]["shared_connectivity_blocks"]),
            "PkisTwoPkisOneConnectivityOverlap": str(experimental_pair_overlap[("PKIS2", "PKIS1")]["shared_connectivity_blocks"]),
            "ExperimentalTripleConnectivityOverlap": str(experimental_overlap["three_panel_structural_overlap"]["shared_connectivity_blocks"]),
            "KiRHubDavisNameOverlap": str(experimental_overlap["KiRHub_boundary"]["exact_normalized_name_overlap_with_DAVIS"]),
            "KiRHubOverlapExcludedRho": f"{experimental_overlap['KiRHub_boundary']['geometry_spearman_to_DAVIS_PKIS2_PKIS1_mean_after_removing_11_DAVIS_names']:.3f}",
            "KiRHubOverlapExcludedAdjustedRho": f"{manuscript['kirhub_chemical_overlap_sensitivity']['after_excluding_overlap_partial_geometry_qap']['partial_spearman']:.3f}",
            "DffConditionalMWSlope": f"{physicochemical['Docking-44']['conditional_model']['molecular_weight_slope']['median']:.3f}",
            "DsConditionalMWSlope": f"{physicochemical['DOCKSTRING-58']['conditional_model']['molecular_weight_slope']['median']:.3f}",
            "DffConditionalRotSlope": f"{physicochemical['Docking-44']['conditional_model']['rotatable_bond_slope']['median']:.3f}",
            "DsConditionalRotSlope": f"{physicochemical['DOCKSTRING-58']['conditional_model']['rotatable_bond_slope']['median']:.3f}",
            "DffPositiveRotTargets": str(physicochemical['Docking-44']['conditional_model']['targets_with_positive_rotatable_bond_slope']),
            "DsPositiveRotTargets": str(physicochemical['DOCKSTRING-58']['conditional_model']['targets_with_positive_rotatable_bond_slope']),
            "DffMWResidualLoadingCosine": f"{physicochemical['Docking-44']['descriptor_slopes']['molecular_weight']['cosine_with_residual_pc1_loading']:.3f}",
            "DsMWResidualLoadingCosine": f"{physicochemical['DOCKSTRING-58']['descriptor_slopes']['molecular_weight']['cosine_with_residual_pc1_loading']:.3f}",
            "DffMWResidualScoreCorrelation": f"{physicochemical['Docking-44']['descriptor_slopes']['molecular_weight']['absolute_correlation_with_residual_pc1_scores']:.3f}",
            "DsMWResidualScoreCorrelation": f"{physicochemical['DOCKSTRING-58']['descriptor_slopes']['molecular_weight']['absolute_correlation_with_residual_pc1_scores']:.3f}",
            "DffDescriptorPRBefore": f"{descriptor_decomposition['Docking-44']['metrics']['out_of_fold_pr_before_descriptor_removal']:.3f}",
            "DffDescriptorPRAfter": f"{descriptor_decomposition['Docking-44']['metrics']['out_of_fold_pr_after_descriptor_removal']:.3f}",
            "DsDescriptorPRBefore": f"{descriptor_decomposition['DOCKSTRING-58']['metrics']['out_of_fold_pr_before_descriptor_removal']:.3f}",
            "DsDescriptorPRAfter": f"{descriptor_decomposition['DOCKSTRING-58']['metrics']['out_of_fold_pr_after_descriptor_removal']:.3f}",
            "DffDescriptorCorrelationReduction": f"{100 * descriptor_decomposition['Docking-44']['metrics']['out_of_fold_relative_mean_squared_target_correlation_reduction']:.1f}\\%",
            "DsDescriptorCorrelationReduction": f"{100 * descriptor_decomposition['DOCKSTRING-58']['metrics']['out_of_fold_relative_mean_squared_target_correlation_reduction']:.1f}\\%",
            "DffDescriptorPCOneRtwo": f"{descriptor_decomposition['Docking-44']['metrics']['out_of_fold_residual_pc1_r2']:.3f}",
            "DsDescriptorPCOneRtwo": f"{descriptor_decomposition['DOCKSTRING-58']['metrics']['out_of_fold_residual_pc1_r2']:.3f}",
            "DffMWDomainRho": f"{mw_domain['Docking-44']['row_centered_residual_geometry']['spearman']:.3f}",
            "DffMWDomainLow": f"{mw_domain['Docking-44']['row_centered_residual_geometry']['chemical_group_bootstrap']['central_95_percent_interval'][0]:.3f}",
            "DffMWDomainHigh": f"{mw_domain['Docking-44']['row_centered_residual_geometry']['chemical_group_bootstrap']['central_95_percent_interval'][1]:.3f}",
            "DffMWDomainControl": f"{mw_domain['Docking-44']['row_centered_residual_geometry']['mw_matched_chemical_group_disjoint_control']['mean']:.3f}",
            "DffMWDomainRandomControl": f"{mw_domain['Docking-44']['row_centered_residual_geometry']['random_disjoint_support_control']['mean']:.3f}",
            "DffMWDomainAdjusted": f"{mw_domain['Docking-44']['strict_descriptor_adjusted_residual_sensitivity']['geometry_spearman']:.3f}",
            "DffMWDomainSignFlip": f"{100 * mw_domain['Docking-44']['row_centered_residual_geometry']['target_pair_sign_flip_fraction']:.1f}\\%",
            "DffMWRawDomainRho": f"{mw_domain['Docking-44']['raw_surface_geometry']['spearman']:.3f}",
            "DffMWRawDomainLow": f"{mw_domain['Docking-44']['raw_surface_geometry']['chemical_group_bootstrap']['central_95_percent_interval'][0]:.3f}",
            "DffMWRawDomainHigh": f"{mw_domain['Docking-44']['raw_surface_geometry']['chemical_group_bootstrap']['central_95_percent_interval'][1]:.3f}",
            "DffMWRawDomainControl": f"{mw_domain['Docking-44']['raw_surface_geometry']['mw_matched_chemical_group_disjoint_control']['mean']:.3f}",
            "DffMWRawDomainRandomControl": f"{mw_domain['Docking-44']['raw_surface_geometry']['random_disjoint_support_control']['mean']:.3f}",
            "DsMWDomainRho": f"{mw_domain['DOCKSTRING-58']['row_centered_residual_geometry']['spearman']:.3f}",
            "DsMWDomainLow": f"{mw_domain['DOCKSTRING-58']['row_centered_residual_geometry']['chemical_group_bootstrap']['central_95_percent_interval'][0]:.3f}",
            "DsMWDomainHigh": f"{mw_domain['DOCKSTRING-58']['row_centered_residual_geometry']['chemical_group_bootstrap']['central_95_percent_interval'][1]:.3f}",
            "DsMWDomainControl": f"{mw_domain['DOCKSTRING-58']['row_centered_residual_geometry']['mw_matched_chemical_group_disjoint_control']['mean']:.3f}",
            "DsMWDomainRandomControl": f"{mw_domain['DOCKSTRING-58']['row_centered_residual_geometry']['random_disjoint_support_control']['mean']:.3f}",
            "DsMWDomainAdjusted": f"{mw_domain['DOCKSTRING-58']['strict_descriptor_adjusted_residual_sensitivity']['geometry_spearman']:.3f}",
            "DsMWDomainSignFlip": f"{100 * mw_domain['DOCKSTRING-58']['row_centered_residual_geometry']['target_pair_sign_flip_fraction']:.1f}\\%",
            "DsMWRawDomainRho": f"{mw_domain['DOCKSTRING-58']['raw_surface_geometry']['spearman']:.3f}",
            "DsMWRawDomainLow": f"{mw_domain['DOCKSTRING-58']['raw_surface_geometry']['chemical_group_bootstrap']['central_95_percent_interval'][0]:.3f}",
            "DsMWRawDomainHigh": f"{mw_domain['DOCKSTRING-58']['raw_surface_geometry']['chemical_group_bootstrap']['central_95_percent_interval'][1]:.3f}",
            "DsMWRawDomainControl": f"{mw_domain['DOCKSTRING-58']['raw_surface_geometry']['mw_matched_chemical_group_disjoint_control']['mean']:.3f}",
            "DsMWRawDomainRandomControl": f"{mw_domain['DOCKSTRING-58']['raw_surface_geometry']['random_disjoint_support_control']['mean']:.3f}",
            "DffSizeFamilyRhoLow": f"{descriptor_family_ranges['Docking-44']['size_low']:.3f}",
            "DffSizeFamilyRhoHigh": f"{descriptor_family_ranges['Docking-44']['size_high']:.3f}",
            "DsSizeFamilyRhoLow": f"{descriptor_family_ranges['DOCKSTRING-58']['size_low']:.3f}",
            "DsSizeFamilyRhoHigh": f"{descriptor_family_ranges['DOCKSTRING-58']['size_high']:.3f}",
            "DffOtherDescriptorRhoLow": f"{descriptor_family_ranges['Docking-44']['other_low']:.3f}",
            "DffOtherDescriptorRhoHigh": f"{descriptor_family_ranges['Docking-44']['other_high']:.3f}",
            "DsOtherDescriptorRhoLow": f"{descriptor_family_ranges['DOCKSTRING-58']['other_low']:.3f}",
            "DsOtherDescriptorRhoHigh": f"{descriptor_family_ranges['DOCKSTRING-58']['other_high']:.3f}",
            "DffMWLowDomainResidualPR": f"{descriptor_family['Docking-44']['descriptor_results']['molecular_weight']['low_domain_residual_participation_ratio']:.3f}",
            "DffMWHighDomainResidualPR": f"{descriptor_family['Docking-44']['descriptor_results']['molecular_weight']['high_domain_residual_participation_ratio']:.3f}",
            "DsMWLowDomainResidualPR": f"{descriptor_family['DOCKSTRING-58']['descriptor_results']['molecular_weight']['low_domain_residual_participation_ratio']:.3f}",
            "DsMWHighDomainResidualPR": f"{descriptor_family['DOCKSTRING-58']['descriptor_results']['molecular_weight']['high_domain_residual_participation_ratio']:.3f}",
            "FixedTwentyScaffoldRowsExcluded": f"{fixed_reference['same_cyclic_murcko_exclusion']['reference_rows_excluded']:,}",
            "FixedTwentyMaxScaffoldDelta": f"{max(abs(row['centered_delta_from_primary']) for row in fixed_reference['same_cyclic_murcko_exclusion']['panels'].values()):.4f}",
        }
    )
    for dataset, dataset_prefix in (
        ("Docking-44", "Dff"),
        ("DOCKSTRING-58", "Ds"),
    ):
        for surface_key, surface_prefix in (
            ("raw_surface_geometry", "Raw"),
            ("row_centered_residual_geometry", "Residual"),
        ):
            within = mw_domain[dataset][surface_key][
                "restriction_matched_within_band_reproducibility"
            ]
            for band, band_prefix in (("low_mw", "Low"), ("high_mw", "High")):
                for control_type, control_prefix in (
                    ("row_random_disjoint", "Random"),
                    (
                        "mw_stratified_chemical_group_disjoint",
                        "GroupDisjoint",
                    ),
                ):
                    record = within[band][control_type]
                    prefix = (
                        f"{dataset_prefix}MWWithin{band_prefix}{surface_prefix}"
                        f"{control_prefix}"
                    )
                    sensitivity = record[
                        "central_95_percent_repeated_split_sensitivity_range"
                    ]
                    macros.update(
                        {
                            f"{prefix}Mean": (
                                f"{record['geometry_spearman_mean']:.3f}"
                            ),
                            f"{prefix}Median": (
                                f"{record['geometry_spearman_median']:.3f}"
                            ),
                            f"{prefix}Low": f"{sensitivity[0]:.3f}",
                            f"{prefix}High": f"{sensitivity[1]:.3f}",
                        }
                    )
    endpoint_prefix = {
        "Ki": "Ki",
        "Kd": "Kd",
        "IC50": "ICFifty",
        "EC50": "ECFifty",
    }
    for endpoint, prefix in endpoint_prefix.items():
        record = broad_single_endpoints[endpoint]
        contrast = record["contrasts"][
            "two_way_residual_minus_absolute_vina"
        ]
        interval = contrast["conservative_cluster_bootstrap_interval_95"]
        macros.update(
            {
                f"BroadSingle{prefix}Absolute": (
                    f"{record['mean_per_ligand_pairwise_accuracy']['absolute_vina']:.3f}"
                ),
                f"BroadSingle{prefix}Delta": (
                    f"{contrast['plugin_mean_difference']:+.3f}"
                ),
                f"BroadSingle{prefix}Low": f"{interval[0]:.3f}",
                f"BroadSingle{prefix}High": f"{interval[1]:.3f}",
            }
        )
    panel_prefix = {"DAVIS": "Davis", "PKIS2": "PkisTwo", "PKIS1": "PkisOne", "KiRHub": "KiRHub"}
    for panel, prefix in panel_prefix.items():
        record = fixed[panel]
        matched_record = fixed_matched[panel]
        values = {
            f"{prefix}RawRaw": record["docking_raw__experimental_raw"],
            f"{prefix}RawCentered": record["docking_raw__experimental_centered"],
            f"{prefix}CenteredRaw": record["docking_centered__experimental_raw"],
            f"{prefix}CenteredCentered": record["docking_centered__experimental_centered"],
            f"{prefix}RawEndpointDockingIncrement": record["docking_transform_increment_with_raw_experiment"],
            f"{prefix}DockingIncrement": record["docking_transform_increment_with_centered_experiment"],
            f"{prefix}DockingP": record["paired_qap_for_docking_increment"]["one_sided_p_positive_delta"],
            f"{prefix}SequentialFraction": record["fraction_of_matched_change_before_docking_centering"],
            f"{prefix}PathAveragedExperimentalFraction": record["path_averaged_descriptive_attribution"]["experimental_fraction_of_total"],
        }
        for name, value in values.items():
            macros[name] = f"{value:.3f}"
        macros[f"{prefix}RawEndpointDockingP"] = (
            f"{record['paired_qap_for_docking_increment_with_raw_experiment']['one_sided_p_positive_delta']:.5f}"
        )
        macros[f"{prefix}RawEndpointDockingMatchedP"] = (
            f"{matched_record['raw_experimental_target_geometry']['p_observed_gain_at_least_as_large']:.5f}"
        )
        macros[f"{prefix}RawEndpointDockingMatchedDelta"] = (
            f"{matched_record['raw_experimental_target_geometry']['matched_support_observed_centered_minus_raw_docking']:.3f}"
        )
        macros[f"{prefix}DockingMatchedP"] = (
            f"{matched_record['two_way_centered_experimental_target_geometry']['p_observed_gain_at_least_as_large']:.3f}"
        )
        macros[f"{prefix}DockingMatchedDelta"] = (
            f"{matched_record['two_way_centered_experimental_target_geometry']['matched_support_observed_centered_minus_raw_docking']:.3f}"
        )
    macro_text = "% Generated by analysis/make_reported_results.py; do not edit.\n"
    macro_text += "\n".join(macro(name, value) for name, value in macros.items()) + "\n"

    table_lines = [
        "% Generated by analysis/make_reported_results.py; do not edit.",
        r"\begin{table}[t]",
        r"\centering\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\caption{\textbf{Complementary summaries of target dependence.} PR and mean squared off-diagonal correlation are algebraically equivalent for a correlation matrix; mean absolute correlation retains a more familiar correlation scale, while PC1 reports concentration in only the leading mode.}",
        r"\label{tab:spectralmetrics}",
        r"\begin{tabular}{llrrrr}",
        r"\toprule Matrix & surface & PR & mean $r^2$ & mean $|r|$ & PC1 \\",
        r"\midrule",
    ]
    for key, label in [("docking44", "Docking-44"), ("dockstring58", "DOCKSTRING-58")]:
        for surface, surface_label in [("raw", "column-standardized"), ("interaction", "residual")]:
            dataset = "Docking-44" if key == "docking44" else "DOCKSTRING-58"
            manuscript_surface = "raw" if surface == "raw" else "residual"
            record = manuscript["spectral"][dataset]["correlation"][manuscript_surface]
            table_lines.append(
                f"{label} & {surface_label} & {record['participation_ratio']:.2f} & "
                f"{record['mean_squared_offdiagonal_correlation']:.3f} & "
                f"{record['mean_absolute_offdiagonal_correlation']:.3f} & "
                f"{record['pc1_fraction']:.3f} \\\\"
            )
    table_lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    registry = {
        "sources": ["results/manuscript_evidence.json"],
        "macros": macros,
        "rounding": "round-half-even through Python fixed-point formatting",
    }
    return macro_text, "\n".join(table_lines), registry


def main() -> None:
    manuscript = json.loads((RESULTS / "manuscript_evidence.json").read_text())
    macro_text, table_text, registry = build(manuscript)
    (RESULTS / "evidence_macros.tex").write_text(macro_text)
    (RESULTS / "main_spectral_metrics_table.tex").write_text(table_text)
    (RESULTS / "reported_values.json").write_text(json.dumps(registry, indent=2) + "\n")
    print("Wrote evidence macros, main spectral table and reported-value registry")


if __name__ == "__main__":
    main()
