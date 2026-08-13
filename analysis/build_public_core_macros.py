#!/usr/bin/env python3
"""Generate manuscript LaTeX macros from the strict public-core evidence ledger."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


PACKAGE = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PACKAGE / "results" / "public_core_evidence.json"
DEFAULT_OUTPUT = PACKAGE / "results" / "public_core_macros.tex"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def one(records: list[dict[str, Any]], **selectors: Any) -> dict[str, Any]:
    selected = [
        record
        for record in records
        if all(record.get(key) == value for key, value in selectors.items())
    ]
    if len(selected) != 1:
        raise ValueError(f"expected one record for {selectors}, observed {len(selected)}")
    return selected[0]


def f3(value: float) -> str:
    return f"{float(value):.3f}"


def f4(value: float) -> str:
    return f"{float(value):.4f}"


def f6(value: float) -> str:
    return f"{float(value):.6f}"


def integer(value: int) -> str:
    return f"{int(value):,}"


def macro_values(ledger: dict[str, Any]) -> list[tuple[str, str]]:
    if ledger.get("schema_version") != "4.0.0":
        raise ValueError("public-core schema is not 4.0.0")
    panels = ledger["large_vina_panels"]
    d44 = panels["Docking-44"]
    ds = panels["DOCKSTRING-58"]
    domain = ledger["chemical_domain_controls"]["key_boundary_results"]
    descriptor_families = ledger["descriptor_domain_specificity"]["family_summary"]
    recovery = ledger["chemical_support_and_recovery"]
    scaffold = ledger["scaffold_holdout_recovery"]["key_results"]
    random_holdout = ledger["scaffold_holdout_recovery"]["matched_random_row_results"]
    selector_controls = ledger["target_panel_selector_controls"]
    selector_results = selector_controls["k8_omitted_target_results"]
    common_omitted = selector_controls["common_omitted_designed_vs_random"]
    experimental = ledger["experimental_map_boundary"]
    hotspot = ledger["hotspot_panel_validation"]
    overlap = ledger["cross_panel_overlap"]

    def domain_row(dataset: str) -> dict[str, Any]:
        return one(
            domain,
            dataset=dataset,
            transformation="row_centered_residual",
        )

    def family_row(dataset: str, family: str) -> dict[str, Any]:
        return one(
            descriptor_families,
            dataset=dataset,
            descriptor_family=family,
        )

    def pilot(dataset: str, size: int) -> dict[str, Any]:
        return recovery["pilot_recovery"][dataset][str(size)]

    def heldout(dataset: str, size: int) -> dict[str, Any]:
        return one(scaffold, dataset=dataset, calibration_ligands=size)

    def random_heldout(dataset: str, size: int) -> dict[str, Any]:
        return one(random_holdout, dataset=dataset, calibration_ligands=size)

    def selector_result(dataset: str) -> dict[str, Any]:
        return one(selector_results, dataset=dataset)

    def common_residual_result(dataset: str) -> dict[str, Any]:
        return one(
            common_omitted,
            dataset=dataset,
            surface="raw_row_centered_residual",
            training_outcome="raw_then_derived_residual",
        )

    null_datasets = ledger["residual_nulls"]["datasets"]
    d44_nulls = null_datasets["docking44"]["nulls"]
    ds_nulls = null_datasets["dockstring58"]["nulls"]
    conventional_null_names = (
        "additive_gaussian",
        "empirical_residual_column_permutation",
        "row_norm_preserving_random_direction",
    )
    d44_null_medians = [
        d44_nulls[name]["null_residual_pr_median"]
        for name in conventional_null_names
    ]
    ds_null_medians = [
        ds_nulls[name]["null_residual_pr_median"]
        for name in conventional_null_names
    ]
    d44_mw_null = d44_nulls["molecular_weight_conditional_permutation"][
        "mw_10_bins"
    ]
    ds_mw_null = ds_nulls["molecular_weight_conditional_permutation"][
        "mw_10_bins"
    ]
    cross = one(
        experimental["cross_panel_experimental_geometry"],
        transform="two_way_centered",
    )
    cross_raw = one(
        experimental["cross_panel_experimental_geometry"],
        transform="raw",
    )
    hotspot_primary = hotspot["panel_transfer"]["primary_inference"]
    hotspot_rank = hotspot["panel_transfer"]["column_rank_inference"]
    hotspot_joint = hotspot["panel_transfer"]["joint_k_averaged_inference"]
    hotspot_cluster = hotspot["chemical_cluster_multiplier"]["statistics"][
        "conservative_best_raw_minus_worst_residual_normalized_auc_over_k"
    ]
    target_loo = hotspot["target_leave_one_out_influence"]["panel_transfer"]

    def hotspot_map(panel: str, transform: str) -> dict[str, Any]:
        return one(
            hotspot["map_concordance"],
            comparison_panel=panel,
            anastassiadis_transformation=transform,
        )

    davis = experimental["panels"]["DAVIS"]
    pkis2 = experimental["panels"]["PKIS2"]
    missing = d44["missing_data_sensitivity"]

    def range_value(record: dict[str, Any]) -> str:
        return (
            f"{f3(record['geometry_spearman_minimum'])}--"
            f"{f3(record['geometry_spearman_maximum'])}"
        )

    values = [
        ("DffRows", integer(d44["support"]["analysis_rows"])),
        ("DffTargets", integer(d44["support"]["targets"])),
        ("DsRows", integer(ds["support"]["analysis_rows"])),
        ("DsTargets", integer(ds["support"]["targets"])),
        ("DffRawPR", f3(d44["column_standardized_surface"]["participation_ratio_dimension"])),
        ("DffResidualPR", f3(d44["two_way_centered_residual_surface"]["participation_ratio_dimension"])),
        ("DsRawPR", f3(ds["column_standardized_surface"]["participation_ratio_dimension"])),
        ("DsResidualPR", f3(ds["two_way_centered_residual_surface"]["participation_ratio_dimension"])),
        ("DffRawMeanRtwo", f3(d44["column_standardized_surface"]["mean_squared_target_correlation"])),
        ("DffResidualMeanRtwo", f3(d44["two_way_centered_residual_surface"]["mean_squared_target_correlation"])),
        ("DsRawMeanRtwo", f3(ds["column_standardized_surface"]["mean_squared_target_correlation"])),
        ("DsResidualMeanRtwo", f3(ds["two_way_centered_residual_surface"]["mean_squared_target_correlation"])),
        ("DffAxisCosine", f3(d44["shared_axis"]["cosine_similarity_with_uniform_axis"])),
        ("DsAxisCosine", f3(ds["shared_axis"]["cosine_similarity_with_uniform_axis"])),
        ("DffCovRawPR", f3(d44["covariance_sensitivity"]["unscaled_surface"]["participation_ratio_dimension"])),
        ("DffCovResidualPR", f3(d44["covariance_sensitivity"]["unscaled_two_way_centered_residual"]["participation_ratio_dimension"])),
        ("DsCovRawPR", f3(ds["covariance_sensitivity"]["unscaled_surface"]["participation_ratio_dimension"])),
        ("DsCovResidualPR", f3(ds["covariance_sensitivity"]["unscaled_two_way_centered_residual"]["participation_ratio_dimension"])),
        ("DffNullMedianLow", f3(min(d44_null_medians))),
        ("DffNullMedianHigh", f3(max(d44_null_medians))),
        ("DsNullMedianLow", f3(min(ds_null_medians))),
        ("DsNullMedianHigh", f3(max(ds_null_medians))),
        ("DffMWConditionedNullPR", f3(d44_mw_null["mw_conditioned_residual_pr"]["median"])),
        ("DsMWConditionedNullPR", f3(ds_mw_null["mw_conditioned_residual_pr"]["median"])),
        ("DffMWConditionedExplainedShare", f3(d44_mw_null["comparisons"]["molecular_weight_quantile"]["explained_share_of_observed_pr_gap_relative_to_one_bin"])),
        ("DsMWConditionedExplainedShare", f3(ds_mw_null["comparisons"]["molecular_weight_quantile"]["explained_share_of_observed_pr_gap_relative_to_one_bin"])),
        ("DffMWConditionedMapRho", f3(d44_mw_null["mw_conditioned_map_agreement"]["median"])),
        ("DsMWConditionedMapRho", f3(ds_mw_null["mw_conditioned_map_agreement"]["median"])),
        ("DffMWConditionedSignAgreement", f3(d44_mw_null["mw_conditioned_sign_agreement"]["median"])),
        ("DsMWConditionedSignAgreement", f3(ds_mw_null["mw_conditioned_sign_agreement"]["median"])),
        ("DffMWQuartileRho", f3(domain_row("Docking-44")["observed_low_high_geometry_spearman"])),
        ("DsMWQuartileRho", f3(domain_row("DOCKSTRING-58")["observed_low_high_geometry_spearman"])),
        ("DffMWMatchedControlRho", f3(domain_row("Docking-44")["global_mw_matched_group_disjoint_median"])),
        ("DsMWMatchedControlRho", f3(domain_row("DOCKSTRING-58")["global_mw_matched_group_disjoint_median"])),
        ("DffMWContinuumRho", f3(recovery["support_dependence"]["Docking-44"]["ten_bin_continuum"]["row_centered_residual"]["spearman_mw_separation_vs_map_dissimilarity"])),
        ("DsMWContinuumRho", f3(recovery["support_dependence"]["DOCKSTRING-58"]["ten_bin_continuum"]["row_centered_residual"]["spearman_mw_separation_vs_map_dissimilarity"])),
        ("DffSizeAxisRhoRange", range_value(family_row("Docking-44", "size_related"))),
        ("DsSizeAxisRhoRange", range_value(family_row("DOCKSTRING-58", "size_related"))),
        ("DffOtherAxisRhoRange", range_value(family_row("Docking-44", "other_coarse"))),
        ("DsOtherAxisRhoRange", range_value(family_row("DOCKSTRING-58", "other_coarse"))),
        ("DffProbeTwoHundred", f3(pilot("Docking-44", 200)["map_edges"]["geometry_spearman_mean"])),
        ("DffProbeFiveHundred", f3(pilot("Docking-44", 500)["map_edges"]["geometry_spearman_mean"])),
        ("DsProbeTwoHundred", f3(pilot("DOCKSTRING-58", 200)["map_edges"]["geometry_spearman_mean"])),
        ("DsProbeFiveHundred", f3(pilot("DOCKSTRING-58", 500)["map_edges"]["geometry_spearman_mean"])),
        ("DffProbeTwoHundredSign", f3(pilot("Docking-44", 200)["map_edges"]["edge_sign_agreement_mean"])),
        ("DffProbeFiveHundredSign", f3(pilot("Docking-44", 500)["map_edges"]["edge_sign_agreement_mean"])),
        ("DsProbeTwoHundredSign", f3(pilot("DOCKSTRING-58", 200)["map_edges"]["edge_sign_agreement_mean"])),
        ("DsProbeFiveHundredSign", f3(pilot("DOCKSTRING-58", 500)["map_edges"]["edge_sign_agreement_mean"])),
        ("DffScaffoldProbeTwoHundred", f3(heldout("Docking-44", 200)["geometry_spearman_mean"])),
        ("DffScaffoldProbeFiveHundred", f3(heldout("Docking-44", 500)["geometry_spearman_mean"])),
        ("DsScaffoldProbeTwoHundred", f3(heldout("DOCKSTRING-58", 200)["geometry_spearman_mean"])),
        ("DsScaffoldProbeFiveHundred", f3(heldout("DOCKSTRING-58", 500)["geometry_spearman_mean"])),
        ("DffRandomHoldoutTwoHundred", f3(random_heldout("Docking-44", 200)["geometry_spearman_mean"])),
        ("DffRandomHoldoutFiveHundred", f3(random_heldout("Docking-44", 500)["geometry_spearman_mean"])),
        ("DsRandomHoldoutTwoHundred", f3(random_heldout("DOCKSTRING-58", 200)["geometry_spearman_mean"])),
        ("DsRandomHoldoutFiveHundred", f3(random_heldout("DOCKSTRING-58", 500)["geometry_spearman_mean"])),
        ("DffSelectorRawRtwo", f3(selector_result("Docking-44")["raw_ridge_variance_weighted_r2_median"])),
        ("DsSelectorRawRtwo", f3(selector_result("DOCKSTRING-58")["raw_ridge_variance_weighted_r2_median"])),
        ("DffSelectorResidualRtwo", f3(selector_result("Docking-44")["derived_residual_ridge_variance_weighted_r2_median"])),
        ("DsSelectorResidualRtwo", f3(selector_result("DOCKSTRING-58")["derived_residual_ridge_variance_weighted_r2_median"])),
        ("DffSelectorDirectResidualRtwo", f3(selector_result("Docking-44")["direct_residual_ridge_variance_weighted_r2_median"])),
        ("DsSelectorDirectResidualRtwo", f3(selector_result("DOCKSTRING-58")["direct_residual_ridge_variance_weighted_r2_median"])),
        ("DffSelectorResidualPCOneRtwo", f3(selector_result("Docking-44")["derived_residual_pc1_variance_weighted_r2_median"])),
        ("DsSelectorResidualPCOneRtwo", f3(selector_result("DOCKSTRING-58")["derived_residual_pc1_variance_weighted_r2_median"])),
        ("DffSelectorDirectResidualDelta", f6(selector_result("Docking-44")["direct_minus_derived_ridge_variance_weighted_r2_median"])),
        ("DsSelectorDirectResidualDelta", f6(selector_result("DOCKSTRING-58")["direct_minus_derived_ridge_variance_weighted_r2_median"])),
        ("DffSelectorCommonResidualDelta", f3(common_residual_result("Docking-44")["designed_benefit_mean_r2_random_median_median"])),
        ("DsSelectorCommonResidualDelta", f3(common_residual_result("DOCKSTRING-58")["designed_benefit_mean_r2_random_median_median"])),
        ("DffPanelTwoHundredDelta", f4(pilot("Docking-44", 200)["eight_target_panel_decision"]["sample_minus_fixed_full_source_panel_mean_nearest_distance_on_reference_map_mean"])),
        ("DffPanelFiveHundredDelta", f4(pilot("Docking-44", 500)["eight_target_panel_decision"]["sample_minus_fixed_full_source_panel_mean_nearest_distance_on_reference_map_mean"])),
        ("DsPanelTwoHundredDelta", f4(pilot("DOCKSTRING-58", 200)["eight_target_panel_decision"]["sample_minus_fixed_full_source_panel_mean_nearest_distance_on_reference_map_mean"])),
        ("DsPanelFiveHundredDelta", f4(pilot("DOCKSTRING-58", 500)["eight_target_panel_decision"]["sample_minus_fixed_full_source_panel_mean_nearest_distance_on_reference_map_mean"])),
        ("ExperimentalCrossRho", f3(cross["edge_spearman"])),
        ("ExperimentalCrossRawRho", f3(cross_raw["edge_spearman"])),
        ("ExperimentalCrossQAP", f4(cross["target_label_qap_p_two_sided"])),
        ("DavisSplitRho", f3(davis["released_map_reliability"]["ligand"]["split_map_spearman"]["median"])),
        ("PkistwoSplitRho", f3(pkis2["released_map_reliability"]["ligand"]["split_map_spearman"]["median"])),
        ("DavisSameSupportLigands", integer(davis["primary_same_support_informative_ligands"])),
        ("PkistwoSameSupportLigands", integer(pkis2["primary_same_support_informative_ligands"])),
        ("DavisSameSupportRho", f3(davis["primary_same_support_geometry"]["two_way_centered"]["same_support_empirical_spearman"])),
        ("PkistwoSameSupportRho", f3(pkis2["primary_same_support_geometry"]["two_way_centered"]["same_support_empirical_spearman"])),
        ("DavisBroadSupportRho", f3(davis["primary_same_support_geometry"]["two_way_centered"]["broad_support_empirical_spearman"])),
        ("PkistwoBroadSupportRho", f3(pkis2["primary_same_support_geometry"]["two_way_centered"]["broad_support_empirical_spearman"])),
        ("HotspotLigands", integer(hotspot["experimental_support"]["Anastassiadis"]["ligands"])),
        ("HotspotTargets", integer(hotspot["experimental_support"]["Anastassiadis"]["targets"])),
        ("HotspotSplitRho", f3(hotspot["hotspot_reliability"]["median"])),
        ("HotspotSplitRhoLow", f3(hotspot["hotspot_reliability"]["q025"])),
        ("HotspotSplitRhoHigh", f3(hotspot["hotspot_reliability"]["q975"])),
        ("DavisHotspotRawRho", f3(hotspot_map("DAVIS", "raw_correlation")["target_pair_spearman"])),
        ("DavisHotspotResidualRho", f3(hotspot_map("DAVIS", "row_centered_correlation")["target_pair_spearman"])),
        ("PkistwoHotspotRawRho", f3(hotspot_map("PKIS2", "raw_correlation")["target_pair_spearman"])),
        ("PkistwoHotspotResidualRho", f3(hotspot_map("PKIS2", "row_centered_correlation")["target_pair_spearman"])),
        ("PanelTransferPositiveCells", integer(hotspot_primary["conservative_tie_cells_with_positive_improvement"])),
        ("PanelTransferTotalCells", integer(hotspot_primary["cells_averaged"])),
        ("PanelTransferConservativeMean", f4(hotspot_primary["conservative_mean_coverage_loss_improvement_over_ties"])),
        ("PanelTransferJointAUC", f4(hotspot_joint["normalized_trapezoid_auc_over_k"])),
        ("PanelTransferJointP", f4(hotspot_joint["max_over_three_objectives_fwer_p_for_this_objective"])),
        ("PanelRankPositiveCells", integer(hotspot_rank["conservative_tie_cells_with_positive_improvement"])),
        ("PanelRankConservativeMean", f4(hotspot_rank["conservative_mean_coverage_loss_improvement_over_ties"])),
        ("PanelRankP", f4(hotspot_rank["max_over_three_objectives_p_conservative_tie_improvement"])),
        ("PanelTargetLooPositive", integer(target_loo["joint_normalized_auc_positive_omissions"])),
        ("PanelTargetLooTotal", integer(target_loo["omitted_targets"])),
        ("PanelTargetLooAucLow", f4(target_loo["joint_normalized_auc_min"])),
        ("PanelTargetLooAucHigh", f4(target_loo["joint_normalized_auc_max"])),
        ("PanelTargetLooCellWinsLow", integer(target_loo["positive_cells_min"])),
        ("PanelTargetLooCellWinsHigh", integer(target_loo["positive_cells_max"])),
        ("HotspotClusterMedian", f4(hotspot_cluster["median"])),
        ("HotspotClusterLow", f4(hotspot_cluster["q025"])),
        ("HotspotClusterHigh", f4(hotspot_cluster["q975"])),
        ("HotspotClusterPositiveFraction", f3(hotspot_cluster["positive_fraction"])),
        ("DffMedianImputeResidualPR", f3(missing["variants"]["target_median_imputation"]["residual_pr"])),
        ("DffMissingAwareResidualPR", f3(missing["variants"]["observed_cell_wls_zero_residual_completion"]["residual_pr"])),
        ("DffCompleteCaseResidualPR", f3(missing["variants"]["complete_case"]["residual_pr"])),
        ("DffEqualNResidualPRLow", f3(missing["equal_n_random_support_control"]["q025"])),
        ("DffEqualNResidualPRHigh", f3(missing["equal_n_random_support_control"]["q975"])),
        ("SharedExactLigands", integer(overlap["chemical_overlap"]["shared_full_standard_inchikeys"])),
        ("SharedConnectivityBlocks", integer(overlap["chemical_overlap"]["shared_connectivity_blocks"])),
        ("SharedTargetIdentities", integer(overlap["target_overlap"]["mapped_target_identities"])),
        ("SharedReceptorStructures", integer(overlap["target_overlap"]["same_receptor_structure_used"])),
    ]
    names = [name for name, _ in values]
    if len(names) != len(set(names)):
        raise ValueError("duplicate LaTeX macro name")
    return values


def render(ledger: dict[str, Any], source_sha256: str) -> str:
    lines = [
        "% Auto-generated by analysis/build_public_core_macros.py; do not edit.",
        f"% Source: results/public_core_evidence.json sha256={source_sha256}",
    ]
    lines.extend(f"\\newcommand{{\\{name}}}{{{value}}}" for name, value in macro_values(ledger))
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    input_path = args.input.resolve()
    output_path = args.output.resolve()
    for path in (input_path, output_path):
        if PACKAGE.resolve() not in path.parents:
            raise PermissionError(f"path escaped the package: {path}")
    ledger = json.loads(input_path.read_text())
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render(ledger, sha256_file(input_path)))
    print(output_path.relative_to(PACKAGE))


if __name__ == "__main__":
    main()
