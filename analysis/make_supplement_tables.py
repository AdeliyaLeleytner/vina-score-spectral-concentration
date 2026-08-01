#!/usr/bin/env python3
"""Generate LaTeX tables for the supplementary information from frozen evidence."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def tex(value: object) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def main() -> None:
    evidence = json.loads((RESULTS / "evidence_summary.json").read_text())
    datasets = pd.read_csv(RESULTS / "dataset_summary.csv")
    arms = pd.read_csv(RESULTS / "dti_primary_arms.csv")
    all_arms = pd.read_csv(RESULTS / "dti_all_scorers_sensitivity.csv")
    lines: list[str] = []

    lines.extend([
        r"\begin{landscape}",
        r"\begin{table}[p]",
        r"\centering\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\caption{\textbf{Dataset and preprocessing summary.} PR denotes participation-ratio effective dimension. Listed procedures quantify sensitivity within the observed chemical and target panels, not superpopulation uncertainty.}",
        r"\label{tab:s1datasets}",
        r"\begin{tabular}{p{2.9cm}p{2.3cm}rrrp{4.4cm}rrrp{3.8cm}}",
        r"\toprule",
        r"Matrix & scorer / source & $N$ & $P$ & missing & preprocessing & column-standardized PR & residual PR & entropy rank & uncertainty \\",
        r"\midrule",
    ])
    for row in datasets.itertuples(index=False):
        lines.append(
            f"{tex(row.dataset)} & {tex(row.score_or_scorer)} & {row.n_ligands:,} & "
            f"{row.n_targets} & {100 * row.missing_fraction_before_preprocessing:.3f}\\% & "
            f"{tex(row.preprocessing)} & {row.raw_pr:.3f} & {row.interaction_pr:.3f} & "
            f"{row.raw_entropy_rank:.3f} & {tex(row.uncertainty)} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", r"\end{landscape}", ""])

    prep = evidence["preprocessing_sensitivity"]
    panel_rigor = evidence["frozen_source_summaries"]["panel_rigor"]
    zero_sensitivity = panel_rigor["zero_truncation"]
    raw = prep["raw_participation_ratio"]
    interaction = prep["interaction_participation_ratio"]
    prep_rows = [
        ("Column standardized", "target-mean imputation (primary)", raw["target_mean_imputation_primary"], "same 12,651 ligands"),
        ("Column standardized", "target-median imputation", raw["target_median_imputation"], "same support"),
        ("Column standardized", "complete cases", raw["complete_case"], f"{raw['complete_case_n']:,} ligands; support changes"),
        ("Column standardized", "primary mean-imputed matrix restricted to complete rows", raw["target_mean_imputed_matrix_restricted_to_complete_case_rows"], "same 9,297-row support; equals complete case"),
        ("Column standardized", "pairwise-complete Pearson correlation", raw["pairwise_complete_correlation"], "no matrix imputation"),
        ("Column standardized", "pairwise Spearman correlation", raw["spearman_pairwise_correlation"], "rank correlation"),
        ("Column standardized", "censored zeros treated as missing", zero_sensitivity["T4_censored_as_missing_meanimpute"]["eff_rank"], "111 cells; target-mean imputation"),
        ("Column standardized", "exclude ligands with censored zeros", zero_sensitivity["T3_nozero_compounds"]["eff_rank"], f"{zero_sensitivity['T3_nozero_compounds']['n']:,} ligands"),
        ("Column standardized", "exclude four most-censored targets", zero_sensitivity["T2_drop_top_decile_censored"]["eff_rank"], "changes the target estimand"),
        ("Residual", "center then standardize (primary)", interaction["center_raw_then_standardize_primary"], "closed-form two-way centering"),
        ("Residual", "standardize then center", interaction["standardize_then_center"], "order sensitivity"),
        ("Residual", "robust scale then center", interaction["robust_scale_then_center"], "median/IQR scaling"),
    ]
    lines.extend([
        r"\begin{table}[p]",
        r"\centering\small",
        r"\setlength{\tabcolsep}{3pt}",
        r"\caption{\textbf{Docking-44 preprocessing and censoring sensitivity.} Complete-case and target-exclusion results are not directly comparable because they change chemical or target support.}",
        r"\label{tab:s2preprocessing}",
        r"\begin{tabular}{p{2.8cm}p{5.0cm}rp{5.8cm}}",
        r"\toprule Surface & preprocessing & PR & note \\ \midrule",
    ])
    for surface, method, rank, note in prep_rows:
        lines.append(f"{tex(surface)} & {tex(method)} & {rank:.3f} & {tex(note)} \\\\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])

    exp = evidence["experimental_sensitivity"]
    blocks = exp["curated_assay_blocks"]["blocks"]
    imp = exp["missing_data_estimators"]
    noise = exp["reproducibility_calibrated_noise_injection_74x6"]
    curated = exp["activity_level_curated_matched_blocks"]
    paired = evidence["matched_raw_and_interaction_bootstrap"]
    benchmark = evidence["matched_target_preference_benchmark"]
    primary_mi = benchmark["multiple_imputation"]["participation_ratio"]
    physchem = exp["matched_physicochemical_residualization"]
    exp_rows = [
        ("ChEMBL all endpoints, frozen refetch A", blocks["all_types"]["n_compounds"], blocks["all_types"]["n_targets"], blocks["all_types"]["eff_rank_KNN"], "already human-only; KNN; 52.9% missing"),
        ("human binding Ki/Kd", blocks["Ki_Kd_human_binding"]["n_compounds"], blocks["Ki_Kd_human_binding"]["n_targets"], blocks["Ki_Kd_human_binding"]["eff_rank_KNN"], "assay type B; KNN"),
        ("IC50/EC50", blocks["IC50_EC50"]["n_compounds"], blocks["IC50_EC50"]["n_targets"], blocks["IC50_EC50"]["eff_rank_KNN"], "KNN"),
        ("fully observed sub-block", imp["fully_observed_subblock"]["shape"][0], imp["fully_observed_subblock"]["shape"][1], imp["fully_observed_subblock"]["eff_rank"], "no imputation"),
        ("matched exact-relation median", curated["all_exact_median"]["n_ligands"], 6, curated["all_exact_median"]["experimental_participation_ratio"], f"same-support docking {curated['all_exact_median']['matched_docking_participation_ratio']:.2f}"),
        ("matched human binding, all endpoints", curated["human_binding_all_endpoints"]["n_ligands"], 6, curated["human_binding_all_endpoints"]["experimental_participation_ratio"], f"same-support docking {curated['human_binding_all_endpoints']['matched_docking_participation_ratio']:.2f}"),
        ("matched human binding Ki/Kd", curated["human_binding_Ki_Kd"]["n_ligands"], 6, curated["human_binding_Ki_Kd"]["experimental_participation_ratio"], f"same-support docking {curated['human_binding_Ki_Kd']['matched_docking_participation_ratio']:.2f}"),
        ("ChEMBL all endpoints, frozen refetch B, KNN5", 212, 22, imp["experimental_eff_rank_by_method"]["knn5"], "different frozen extraction; 52.9% missing"),
        ("212 x 22, MICE", 212, 22, imp["experimental_eff_rank_by_method"]["multiple_imputation_MICE"]["mean"], "mean across imputations"),
        ("212 x 22, SoftImpute", 212, 22, imp["experimental_eff_rank_by_method"]["matrix_completion_softimpute"], "low-rank matrix completion"),
        ("212 x 22, PPCA-EM", 212, 22, imp["experimental_eff_rank_by_method"]["probabilistic_pca_em"], "probabilistic PCA"),
        ("matched docking + calibrated noise", 74, 6, noise["docking_rank_with_noise_median"], f"95% simulation interval {noise['docking_rank_with_noise_95_interval'][0]:.2f}--{noise['docking_rank_with_noise_95_interval'][1]:.2f}"),
        ("matched docking + 2.5x calibrated noise", 74, 6, noise["calibrated_noise_multiplier_sweep"]["2.5"]["median_participation_ratio"], "approximately reaches experimental raw PR"),
        ("matched docking, linear descriptor residual", 74, 6, physchem["linear_residual_participation_ratio"], "seven RDKit descriptors; same support"),
        ("matched docking, cross-fitted nonlinear residual", 74, 6, physchem["cross_fitted_histgbm_residual_participation_ratio"], "five-fold HistGBM; same support"),
        ("matched docking residual, exact-relation support", 74, 6, curated["all_exact_median"]["matched_docking_residual_participation_ratio"], "two-way centered"),
        ("matched experimental residual, exact relation median", 74, 6, curated["all_exact_median"]["experimental_residual_participation_ratio"], "two-way centered; primary curation"),
        ("matched experimental raw, 100 MICE draws", 74, 6, primary_mi["raw"]["median"], f"95% across imputations {primary_mi['raw']['interval_95'][0]:.2f}--{primary_mi['raw']['interval_95'][1]:.2f}"),
        ("matched experimental residual, 100 MICE draws", 74, 6, primary_mi["residual"]["median"], f"95% across imputations {primary_mi['residual']['interval_95'][0]:.2f}--{primary_mi['residual']['interval_95'][1]:.2f}"),
        ("paired raw experiment-docking contrast", 74, 6, paired["raw"]["plugin_difference_experiment_minus_docking"], f"ligand-bootstrap interval {paired['raw']['bootstrap_difference_95_interval'][0]:.2f}--{paired['raw']['bootstrap_difference_95_interval'][1]:.2f}"),
        ("paired residual experiment-docking contrast", 74, 6, paired["interaction"]["plugin_difference_experiment_minus_docking"], f"ligand-bootstrap interval {paired['interaction']['bootstrap_difference_95_interval'][0]:.2f}--{paired['interaction']['bootstrap_difference_95_interval'][1]:.2f}"),
    ]
    exp_groups = [
        ("assay and complete-support blocks", exp_rows[:7]),
        ("missing-data and paired-contrast controls", exp_rows[7:11] + exp_rows[17:]),
        ("noise, descriptor and residual controls", exp_rows[11:17]),
    ]
    for group_index, (group_label, group_rows) in enumerate(exp_groups):
        lines.extend([
            r"\begin{table}[p]",
            *([r"\ContinuedFloat"] if group_index else []),
            r"\centering\small",
            r"\setlength{\tabcolsep}{3pt}",
            r"\caption{\textbf{Experimental controls: " + group_label
            + r".} The noise scale was 0.493 pChEMBL, estimated from different-document repeats.}",
            *([r"\label{tab:s3experiment}"] if group_index == 0 else []),
            r"\begin{tabular}{p{5.1cm}rrrp{5.7cm}}",
            r"\toprule Analysis block & $N$ & $P$ & PR & note \\ \midrule",
        ])
        for name, n, p, rank, note in group_rows:
            lines.append(f"{tex(name)} & {n} & {p} & {rank:.3f} & {tex(note)} \\\\")
        lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])

    target_uncertainty = evidence["target_panel_uncertainty"]
    target_rows = []
    for label, key in [("Docking-44", "docking44_jackknife"),
                       ("DOCKSTRING-58", "dockstring58_jackknife")]:
        for surface, surface_label in [("raw", "column-standardized"),
                                       ("interaction", "two-way-centered residual")]:
            rec = target_uncertainty[key][surface]
            target_rows.append((
                label,
                surface_label,
                rec["full_panel_all_ligands"],
                rec["leave_one_out_min"],
                rec["leave_one_out_median"],
                rec["leave_one_out_max"],
                rec["most_influential_target"],
            ))
    parallel = evidence["parallel_analysis_common_protocol"]
    scaffold = evidence["ligand_sampling_sensitivity"]["dockstring_scaffold_bootstrap"]
    additive = evidence["additive_main_effect_null"]
    empirical = evidence["empirical_residual_permutation_null"]
    row_norm = evidence["row_norm_preserving_residual_null"]
    lines.extend([
        r"\begin{table}[htbp]",
        r"\centering\small",
        r"\setlength{\tabcolsep}{3pt}",
        r"\caption{\textbf{Target-panel and chemical-support sensitivities.} Leave-one-target-out values describe composition sensitivity within each observed panel; they are not superpopulation confidence intervals.}",
        r"\label{tab:s4target}",
        r"\begin{tabular}{llrrrrl}",
        r"\toprule Matrix & surface & full PR & LOTO min & LOTO median & LOTO max & target \\ \midrule",
    ])
    for label, surface, full, low, median, high, target in target_rows:
        lines.append(
            f"{tex(label)} & {tex(surface)} & {full:.3f} & {low:.3f} & {median:.3f} & "
            f"{high:.3f} & {tex(target)} \\\\"
        )
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\vspace{5pt}",
        (r"\parbox{0.94\textwidth}{\footnotesize Family-stratified target subsampling is shown "
         r"in Fig.~1c,d. The descriptive rank-wise and family-wise-error-controlled simultaneous mode counts "
         f"were both {parallel['docking44']['raw']['n_modes_above_simultaneous_95pct_envelope']} versus "
         f"{parallel['docking44']['interaction']['n_modes_above_simultaneous_95pct_envelope']} for Docking-44 and "
         f"{parallel['dockstring58']['raw']['n_modes_above_simultaneous_95pct_envelope']} versus "
         f"{parallel['dockstring58']['interaction']['n_modes_above_simultaneous_95pct_envelope']} for DOCKSTRING-58. "
         f"Across {scaffold['n_supports']} independent {scaffold['support_n']:,}-molecule DOCKSTRING supports, "
         f"raw point estimates ranged from {scaffold['support_point_estimates']['raw']['minimum']:.3f} to "
         f"{scaffold['support_point_estimates']['raw']['maximum']:.3f} and residual estimates from "
         f"{scaffold['support_point_estimates']['residual']['minimum']:.3f} to "
         f"{scaffold['support_point_estimates']['residual']['maximum']:.3f}; support-specific scaffold intervals "
         r"are plotted in Fig.~S6. The fitted additive null gave median residual PR dimensions "
         f"{additive['docking44']['null_residual']['median']:.2f} and "
         f"{additive['dockstring58']['null_residual']['median']:.2f}, compared with observed values "
         f"{additive['docking44']['observed']['residual']:.2f} and "
         f"{additive['dockstring58']['observed']['residual']:.2f}. The empirical residual-permutation null gave "
         f"medians {empirical['docking44']['null_residual']['median']:.2f} and "
         f"{empirical['dockstring58']['null_residual']['median']:.2f}. Across 200 matched one-ligand-per-Butina-cluster "
         f"Docking-44 supports, the observed median was "
         f"{empirical['docking44']['one_ligand_per_cluster_sensitivity']['observed_residual']['median']:.2f} versus "
         f"{empirical['docking44']['one_ligand_per_cluster_sensitivity']['matched_empirical_null_residual']['median']:.2f} under permutation. "
         f"The row-norm-preserving null medians were {row_norm['docking44']['null_residual']['median']:.2f} and "
         f"{row_norm['dockstring58']['null_residual']['median']:.2f}.}}"),
        r"\end{table}",
        "",
    ])

    expanded = evidence["expanded_target_preference_benchmark"]
    preference = expanded["primary_all_exact"]
    dense_preference = evidence["matched_target_preference_benchmark"]
    pref_rows = [
        ("absolute Vina", "absolute_vina"),
        ("docking target prior", "docking_target_prior"),
        ("external ChEMBL target prior", "experimental_target_prior"),
        ("cohort experimental target prior (scaffold held out)", "cohort_experimental_target_prior"),
        ("column-standardized Vina", "column_standardized"),
        ("two-way residual Vina", "two_way_residual"),
    ]
    lines.extend([
        r"\begin{table}[p]",
        r"\centering\scriptsize",
        r"\setlength{\tabcolsep}{2pt}",
        r"\caption{\textbf{Broad-coverage observed-pair ChEMBL target-preference benchmark.} This is reported with the human binding Ki/Kd assay-restricted estimand; both were designated during revision rather than preregistered. Accuracy averages within-ligand pairwise accuracies so that highly profiled ligands do not dominate; predicted score ties receive half credit. Intervals resample Bemis--Murcko scaffold clusters; the cohort experimental prior is fitted with the evaluation ligand's entire scaffold cluster held out. Identity shuffles are reported only for score-based representations.}",
        r"\label{tab:s5preference}",
        r"\begin{tabular}{p{3.7cm}>{\centering\arraybackslash}p{1.1cm}>{\centering\arraybackslash}p{1.8cm}>{\centering\arraybackslash}p{1.5cm}>{\centering\arraybackslash}p{1.0cm}>{\centering\arraybackslash}p{3.5cm}}",
        r"\toprule Representation & accuracy & scaffold 95\% CI & \shortstack{pair-\\weighted} & pairs & identity shuffle: mean / $p$ / cluster median $p$ \\ \midrule",
    ])
    for label, key in pref_rows:
        record = preference["representations"][key]
        ci = record["scaffold_cluster_bootstrap"]["interval_95"]
        if "ligand_identity_shuffle_null" in record:
            shuffled = record["ligand_identity_shuffle_null"]
            shuffle_text = (
                f"{shuffled['mean']:.3f} / "
                f"{shuffled['one_sided_empirical_p_observed_at_least_as_large']:.3f} / "
                f"{record['one_ligand_per_murcko_cluster_identity_permutation']['one_sided_p_value_across_supports']['median']:.3f}"
            )
        else:
            shuffle_text = "--"
        lines.append(
            f"{tex(label)} & {record['mean_per_ligand_pairwise_accuracy']:.3f} & "
            f"{ci[0]:.3f}--{ci[1]:.3f} & {record['pair_weighted_accuracy']:.3f} & "
            f"{record['evaluated_pairs']:,} & "
            f"{shuffle_text} \\\\"
        )
    residual_column = preference["paired_comparisons"][
        "two_way_residual_minus_column_standardized"
    ]
    residual_absolute = preference["paired_comparisons"][
        "two_way_residual_minus_absolute_vina"
    ]
    residual_cohort_prior = preference["paired_comparisons"][
        "two_way_residual_minus_cohort_experimental_target_prior"
    ]
    unscaled_identity = preference["paired_comparisons"][
        "two_way_centered_unscaled_minus_target_centered_unscaled"
    ]
    residual_scale_only = preference["representations"][
        "target_centered_residual_scaled"
    ]
    target_centered = preference["representations"]["target_centered_unscaled"]
    same_endpoint = expanded["assay_sensitivities"][
        "same_endpoint_pairs_on_fixed_primary_support"
    ]
    coverage_three = preference["coverage_sensitivity"]["3"]
    human_binding = expanded["assay_sensitivities"]["human_binding_all_endpoints"]
    human_kikd = expanded["assay_sensitivities"]["human_binding_Ki_Kd"]
    human_kikd_residual_absolute = human_kikd["paired_comparisons"][
        "two_way_residual_minus_absolute_vina"
    ]
    same_endpoint_residual_absolute = same_endpoint["paired_comparisons"][
        "two_way_residual_minus_absolute_vina"
    ]

    def union_cluster_interval(record: dict) -> tuple[float, float]:
        intervals = [
            record["scaffold_cluster_bootstrap"]["interval_95"],
            record["butina_cluster_bootstrap"]["interval_95"],
        ]
        return (
            min(interval[0] for interval in intervals),
            max(interval[1] for interval in intervals),
        )

    same_endpoint_residual_absolute_union = union_cluster_interval(
        same_endpoint_residual_absolute
    )
    human_kikd_residual_absolute_union = union_cluster_interval(
        human_kikd_residual_absolute
    )
    residual_column_union = union_cluster_interval(residual_column)
    residual_absolute_union = union_cluster_interval(residual_absolute)
    duplicate_collapsed = same_endpoint["duplicate_collapsed_unique_target_pairs"]
    broad_target_jackknife = preference["target_jackknife_paired_contrasts"][
        "two_way_residual_minus_absolute_vina"
    ]
    kikd_target_jackknife = human_kikd["target_jackknife_paired_contrasts"][
        "two_way_residual_minus_absolute_vina"
    ]
    dense_absolute = dense_preference["representations"]["absolute_vina"]
    dense_docking_prior = dense_preference["representations"]["docking_target_prior"]
    dense_experimental_prior = dense_preference["representations"]["experimental_target_prior"]
    dense_residual = dense_preference["representations"]["two_way_residual"]
    dense_standard = dense_preference["representations"]["column_standardized"]
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\vspace{4pt}",
        (r"\parbox{0.94\textwidth}{\footnotesize The residual-minus-column difference was "
         f"{residual_column['plugin_mean_difference']:+.3f} with a union of cluster-bootstrap intervals "
         f"{residual_column_union[0]:.3f}--"
         f"{residual_column_union[1]:.3f}; its conservative Murcko/Butina "
         f"90\\% interval was {residual_column['post_hoc_equivalence_sensitivity']['conservative_cluster_bootstrap_interval_90'][0]:.3f}--"
         f"{residual_column['post_hoc_equivalence_sensitivity']['conservative_cluster_bootstrap_interval_90'][1]:.3f}. "
         f"The residual-minus-absolute difference was {residual_absolute['plugin_mean_difference']:+.3f} "
         f"({residual_absolute_union[0]:.3f}--"
         f"{residual_absolute_union[1]:.3f}); its conservative 90\\% interval was "
         f"{residual_absolute['post_hoc_equivalence_sensitivity']['conservative_cluster_bootstrap_interval_90'][0]:.3f}--"
         f"{residual_absolute['post_hoc_equivalence_sensitivity']['conservative_cluster_bootstrap_interval_90'][1]:.3f}. "
         f"Post hoc equivalence was not established at $\\pm0.02$ for either contrast or at $\\pm0.05$ for residual versus absolute; "
         f"residual versus column-standardized met only the liberal $\\pm0.05$ criterion. Unscaled target-centered and two-way-centered "
         f"representations were rank-identical at accuracy {target_centered['mean_per_ligand_pairwise_accuracy']:.3f} "
         f"(difference {unscaled_identity['plugin_mean_difference']:+.3f}); residual-target scaling alone gave "
         f"{residual_scale_only['mean_per_ligand_pairwise_accuracy']:.3f}. On fixed support and same-endpoint pairs, absolute, "
         f"column-standardized and residual accuracies were "
         f"{same_endpoint['representations']['absolute_vina']['mean_per_ligand_pairwise_accuracy']:.3f}, "
         f"{same_endpoint['representations']['column_standardized']['mean_per_ligand_pairwise_accuracy']:.3f} and "
         f"{same_endpoint['representations']['two_way_residual']['mean_per_ligand_pairwise_accuracy']:.3f}; the residual-minus-absolute "
         f"cluster-interval union was {same_endpoint_residual_absolute_union[0]:.3f}--{same_endpoint_residual_absolute_union[1]:.3f}. "
         f"On human binding Ki/Kd, residual accuracy was {human_kikd['representations']['two_way_residual']['mean_per_ligand_pairwise_accuracy']:.3f} "
         f"and residual minus absolute was {human_kikd_residual_absolute_union[0]:.3f}--{human_kikd_residual_absolute_union[1]:.3f}. "
         f"Leave-one-target-out residual-minus-absolute differences ranged from {broad_target_jackknife['minimum']:.3f} to "
         f"{broad_target_jackknife['maximum']:.3f} on broad support and {kikd_target_jackknife['minimum']:.3f} to "
         f"{kikd_target_jackknife['maximum']:.3f} on Ki/Kd support. Duplicate-collapsed same-endpoint accuracies were "
         f"{duplicate_collapsed['representations']['absolute_vina']['mean_per_ligand_pairwise_accuracy']:.3f}, "
         f"{duplicate_collapsed['representations']['column_standardized']['mean_per_ligand_pairwise_accuracy']:.3f} and "
         f"{duplicate_collapsed['representations']['two_way_residual']['mean_per_ligand_pairwise_accuracy']:.3f}.}}"),
        r"\end{table}",
        "",
    ])

    assay_table_rows = [
        (
            "human binding Ki/Kd",
            human_kikd["representations"]["absolute_vina"]["evaluated_ligands"],
            human_kikd["representations"]["absolute_vina"]["evaluated_pairs"],
            human_kikd["representations"],
        ),
        (
            "same endpoint, equal endpoint weight",
            same_endpoint["representations"]["absolute_vina"]["evaluated_ligands"],
            same_endpoint["fixed_support"]["endpoint_specific_pair_instances"],
            same_endpoint["representations"],
        ),
    ]
    for endpoint, endpoint_block in same_endpoint["by_endpoint"].items():
        assay_table_rows.append((
            f"{endpoint} only",
            endpoint_block["representations"]["absolute_vina"]["evaluated_ligands"],
            endpoint_block["representations"]["absolute_vina"]["evaluated_pairs"],
            endpoint_block["representations"],
        ))
    assay_table_rows.append((
        "same endpoint, duplicate-collapsed pairs",
        duplicate_collapsed["evaluated_ligands"],
        duplicate_collapsed["unique_ligand_target_pair_instances"],
        duplicate_collapsed["representations"],
    ))
    lines.extend([
        r"\begin{table}[p]",
        r"\ContinuedFloat",
        r"\centering\small",
        r"\caption{\textbf{Assay-restricted and endpoint-specific operational estimands.} Endpoint-only rows can be very small and are descriptive. The duplicate-collapsed row gives every unique ligand--target-pair one contribution after averaging eligible endpoint strata.}",
        r"\begin{tabular}{p{5.2cm}rrrrr}",
        r"\toprule Analysis & ligands & pairs & absolute & column-standardized & residual \\ \midrule",
    ])
    for label, n_ligands, n_pairs, representations in assay_table_rows:
        lines.append(
            f"{tex(label)} & {n_ligands:,} & {n_pairs:,} & "
            f"{representations['absolute_vina']['mean_per_ligand_pairwise_accuracy']:.3f} & "
            f"{representations['column_standardized']['mean_per_ligand_pairwise_accuracy']:.3f} & "
            f"{representations['two_way_residual']['mean_per_ligand_pairwise_accuracy']:.3f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])

    lines.extend([
        r"\begin{table}[htbp]",
        r"\centering\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\caption{\textbf{All 20 exploratory DTI scorer arms.} Plot IDs identify points in Fig.~S5. The above-chance flag is outcome-related and is not used to define this complete panel.}",
        r"\label{tab:s6all20}",
        r"\begin{tabular}{rllrrrrr}",
        r"\toprule ID & arm & above chance & affinity $r$ & column-standardized PR & residual PR & P@5 & AUPRC \\ \midrule",
    ])
    for plot_id, row in enumerate(all_arms.itertuples(index=False), start=1):
        lines.append(
            f"{plot_id} & {tex(row.arm)} & {'yes' if row.clears_chance else 'no'} & "
            f"{row.affinity_pearson:.3f} & {row.raw_effective_rank:.2f} & "
            f"{row.interaction_effective_rank:.2f} & {row.selectivity_p_at_5:.3f} & "
            f"{row.selectivity_auprc:.3f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])

    arm_to_id = {arm: plot_id for plot_id, arm in enumerate(all_arms.arm, start=1)}
    lines.extend([
        r"\begin{landscape}",
        r"\scriptsize",
        r"\begin{longtable}{rp{4.5cm}p{5.3cm}p{5.4cm}p{5.0cm}}",
        r"\caption{\textbf{Provenance for the outcome-restricted 12-arm DTI sensitivity panel.} DAVIS OOF denotes six-fold GroupKFold by target; KIBA transfer denotes zero-shot evaluation on DAVIS after KIBA training. Metrics are listed for all arms in Table~S6.}\label{tab:s7dti}\\",
        r"\toprule",
        r"ID & arm & representation / head & training and evaluation & caveat \\",
        r"\midrule",
        r"\endfirsthead",
        r"\multicolumn{5}{l}{\textit{Supplementary Table S7 continued}}\\",
        r"\toprule",
        r"ID & arm & representation / head & training and evaluation & caveat \\",
        r"\midrule",
        r"\endhead",
        r"\midrule \multicolumn{5}{r}{\textit{continued on next page}}\\",
        r"\endfoot",
        r"\bottomrule",
        r"\endlastfoot",
    ])
    for row in arms.itertuples(index=False):
        lines.append(
            f"{arm_to_id[row.arm]} & {tex(row.display_name)} & {tex(row.representation_and_head)} & "
            f"{tex(row.training_data)}; {tex(row.evaluation_regime)} & {tex(row.caveat)} \\\\"
        )
    lines.extend([r"\end{longtable}", r"\end{landscape}", ""])

    dti = evidence["dti_associations"]
    all20 = dti["all_20_scorers_sensitivity"]
    association_rows = [
        ("12 above-chance", "column-standardized PR", "P@5", dti["raw_rank_vs_selectivity_p_at_5"]),
        ("12 above-chance", "residual PR", "P@5", dti["interaction_rank_vs_selectivity_p_at_5"]),
        ("12 above-chance", "column-standardized PR", "AUPRC", dti["raw_rank_vs_selectivity_auprc"]),
        ("12 above-chance", "residual PR", "AUPRC", dti["interaction_rank_vs_selectivity_auprc"]),
        ("all 20 scorers", "column-standardized PR", "P@5", all20["raw_rank_vs_selectivity_p_at_5"]),
        ("all 20 scorers", "residual PR", "P@5", all20["interaction_rank_vs_selectivity_p_at_5"]),
        ("all 20 scorers", "column-standardized PR", "AUPRC", all20["raw_rank_vs_selectivity_auprc"]),
        ("all 20 scorers", "residual PR", "AUPRC", all20["interaction_rank_vs_selectivity_auprc"]),
    ]
    lines.extend([
        r"\begin{table}[p]",
        r"\centering\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\caption{\textbf{Selection and estimator sensitivity of rank--selectivity associations.} The 12-arm panel excludes eight chance-level arms on an outcome-related criterion. All coefficients are descriptive across dependent model arms.}",
        r"\label{tab:s8selection}",
        r"\begin{tabular}{lllrrrrrr}",
        r"\toprule Model panel & PR variable & endpoint & Pearson $r$ & $p$ & Spearman $\rho$ & $p$ & Kendall $\tau$ & $p$ \\ \midrule",
    ])
    for panel, rank_name, endpoint, values in association_rows:
        lines.append(
            f"{tex(panel)} & {tex(rank_name)} & {tex(endpoint)} & "
            f"{values['pearson_r']:.3f} & {values['pearson_p']:.4f} & "
            f"{values['spearman_rho']:.3f} & {values['spearman_p']:.4f} & "
            f"{values['kendall_tau']:.3f} & {values['kendall_p']:.4f} \\\\"
        )
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\vspace{4pt}",
        (r"\parbox{0.93\textwidth}{\footnotesize For column-standardized PR versus P@5 in the 12-arm "
         f"subset, the arm-resampling 95\\% interval was {dti['primary_raw_rank_spearman_bootstrap_95_interval'][0]:.2f}--"
         f"{dti['primary_raw_rank_spearman_bootstrap_95_interval'][1]:.2f}, leave-one-arm-out values ranged from "
         f"{dti['primary_raw_rank_leave_one_out_range'][0]:.2f} to {dti['primary_raw_rank_leave_one_out_range'][1]:.2f}, "
         f"and the approximate two-sided 0.05 detectable absolute correlation was {dti['approximate_two_sided_alpha_0.05_detectable_absolute_correlation']:.2f}.}}"),
        r"\end{table}",
        "",
    ])

    protocol_rows = [
        (1, "Define matrix and preprocessing", "ligands, targets, missingness, clipping and scaling", "verify_inputs.py; spectral_audit.py"),
        (2, "Separate estimands", "column-standardized and two-way-centered residual surfaces", "centering_ladder"),
        (3, "Summarize concentration", "PR, entropy rank, PC1 fraction, components for 90% variance", "rank_summary"),
        (4, "Quantify chemical-support uncertainty", "paired ligand or scaffold-cluster intervals", "surface_bootstrap"),
        (5, "Probe target composition", "leave-one-target-out and uniform/family-stratified subsampling", "target_jackknife; target_subsampling"),
        (6, "Use transformation-matched nulls", "parallel analysis, Gaussian additive and empirical residual nulls", "parallel_analysis; additive_main_effect_null; empirical_residual_permutation_null"),
        (7, "Check ranking invariance", "decompose offsets, row effects and target-specific scales before task claims", "score-representation decomposition"),
        (8, "Validate operationally", "absolute/transformed scores, target priors, identity/outcome permutations, assay and coverage strata", "expanded operational benchmark"),
        (9, "State the boundary", "report uncertainty and equivalence status; do not equate dimension with accuracy", "manuscript interpretation rule"),
    ]
    lines.extend([
        r"\begin{landscape}",
        r"\begin{table}[p]",
        r"\centering\small",
        r"\setlength{\tabcolsep}{3pt}",
        r"\caption{\textbf{Reusable spectral-audit protocol.} Each step has a machine-readable implementation in the release. The generic command-line wrapper is \texttt{analysis/spectral\_audit.py}; task-level validation remains dataset-specific.}",
        r"\label{tab:s9protocol}",
        r"\begin{tabular}{r>{\raggedright\arraybackslash}p{4.7cm}>{\raggedright\arraybackslash}p{8.0cm}>{\raggedright\arraybackslash}p{8.2cm}}",
        r"\toprule Step & question / action & required output & release implementation \\ \midrule",
    ])
    for step, action, output, implementation in protocol_rows:
        lines.append(
            f"{step} & {tex(action)} & {tex(output)} & {tex(implementation)} \\\\"
        )
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        r"\end{landscape}",
        "",
    ])

    representation_rows = []
    for key, record in preference["representations"].items():
        representation_rows.append({
            "representation": key,
            "mean_per_ligand_pairwise_accuracy": record["mean_per_ligand_pairwise_accuracy"],
            "evaluated_ligands": record["evaluated_ligands"],
            "evaluated_pairs": record["evaluated_pairs"],
            "murcko_ci95_low": record["scaffold_cluster_bootstrap"]["interval_95"][0],
            "murcko_ci95_high": record["scaffold_cluster_bootstrap"]["interval_95"][1],
            "butina_ci95_low": record["butina_cluster_bootstrap"]["interval_95"][0],
            "butina_ci95_high": record["butina_cluster_bootstrap"]["interval_95"][1],
        })
    pd.DataFrame(representation_rows).to_csv(
        RESULTS / "operational_benchmark_representations.csv", index=False
    )

    assay_rows = []
    for label, block in [
        ("all exact endpoints", preference),
        ("human binding all endpoints", human_binding),
        ("human binding Ki/Kd", human_kikd),
    ]:
        for key in ["absolute_vina", "column_standardized", "two_way_residual"]:
            record = block["representations"][key]
            murcko = record["scaffold_cluster_bootstrap"]
            butina = record["butina_cluster_bootstrap"]
            assay_rows.append({
                "analysis": label,
                "representation": key,
                "n_ligands": block["n_ligands"],
                "n_targets": block["n_targets"],
                "evaluated_pairs": record["evaluated_pairs"],
                "mean_per_ligand_pairwise_accuracy": record["mean_per_ligand_pairwise_accuracy"],
                "murcko_ci95_low": murcko["interval_95"][0],
                "murcko_ci95_high": murcko["interval_95"][1],
                "butina_ci95_low": butina["interval_95"][0],
                "butina_ci95_high": butina["interval_95"][1],
            })
    for key, record in same_endpoint["representations"].items():
        murcko = record["murcko_scaffold_cluster_bootstrap"]
        butina = record["butina_cluster_bootstrap"]
        assay_rows.append({
            "analysis": "same-endpoint pairs on fixed primary support",
            "representation": key,
            "n_ligands": same_endpoint["fixed_support"]["ligands"],
            "n_targets": same_endpoint["fixed_support"]["targets"],
            "evaluated_pairs": same_endpoint["fixed_support"]["endpoint_specific_pair_instances"],
            "mean_per_ligand_pairwise_accuracy": record["mean_per_ligand_pairwise_accuracy"],
            "murcko_ci95_low": murcko["interval_95"][0],
            "murcko_ci95_high": murcko["interval_95"][1],
            "butina_ci95_low": butina["interval_95"][0],
            "butina_ci95_high": butina["interval_95"][1],
        })
    for endpoint, endpoint_block in same_endpoint["by_endpoint"].items():
        for key, record in endpoint_block["representations"].items():
            assay_rows.append({
                "analysis": f"endpoint-specific {endpoint}",
                "representation": key,
                "n_ligands": record["evaluated_ligands"],
                "n_targets": endpoint_block["targets_with_any_observation"],
                "evaluated_pairs": record["evaluated_pairs"],
                "mean_per_ligand_pairwise_accuracy": record[
                    "mean_per_ligand_pairwise_accuracy"
                ],
                "murcko_ci95_low": None,
                "murcko_ci95_high": None,
                "butina_ci95_low": None,
                "butina_ci95_high": None,
            })
    duplicate = same_endpoint["duplicate_collapsed_unique_target_pairs"]
    for key, record in duplicate["representations"].items():
        murcko = record["murcko_scaffold_cluster_bootstrap"]
        butina = record["butina_cluster_bootstrap"]
        assay_rows.append({
            "analysis": "same-endpoint duplicate-collapsed unique target pairs",
            "representation": key,
            "n_ligands": duplicate["evaluated_ligands"],
            "n_targets": same_endpoint["fixed_support"]["targets"],
            "evaluated_pairs": duplicate["unique_ligand_target_pair_instances"],
            "mean_per_ligand_pairwise_accuracy": record[
                "mean_per_ligand_pairwise_accuracy"
            ],
            "murcko_ci95_low": murcko["interval_95"][0],
            "murcko_ci95_high": murcko["interval_95"][1],
            "butina_ci95_low": butina["interval_95"][0],
            "butina_ci95_high": butina["interval_95"][1],
        })
    pd.DataFrame(assay_rows).to_csv(
        RESULTS / "operational_assay_sensitivities.csv", index=False
    )

    contrast_rows = []
    for label, block in [
        ("all exact endpoints", preference),
        ("human binding all endpoints", human_binding),
        ("human binding Ki/Kd", human_kikd),
        ("same-endpoint pairs on fixed primary support", same_endpoint),
    ]:
        for contrast in [
            "two_way_residual_minus_absolute_vina",
            "two_way_residual_minus_column_standardized",
        ]:
            record = block["paired_comparisons"][contrast]
            equivalence = record["post_hoc_equivalence_sensitivity"]
            contrast_rows.append({
                "analysis": label,
                "contrast": contrast,
                "plugin_mean_difference": record["plugin_mean_difference"],
                "murcko_ci95_low": record["scaffold_cluster_bootstrap"]["interval_95"][0],
                "murcko_ci95_high": record["scaffold_cluster_bootstrap"]["interval_95"][1],
                "butina_ci95_low": record["butina_cluster_bootstrap"]["interval_95"][0],
                "butina_ci95_high": record["butina_cluster_bootstrap"]["interval_95"][1],
                "conservative_cluster_ci90_low": equivalence["conservative_cluster_bootstrap_interval_90"][0],
                "conservative_cluster_ci90_high": equivalence["conservative_cluster_bootstrap_interval_90"][1],
                "equivalent_at_post_hoc_margin_0.02": equivalence["margins"]["0.02"]["equivalence_established"],
                "equivalent_at_post_hoc_margin_0.05": equivalence["margins"]["0.05"]["equivalence_established"],
                "normal_approximation_mde_80pct_power": record.get(
                    "normal_approx_80pct_power_mde_two_sided_alpha_0.05"
                ),
            })
    pd.DataFrame(contrast_rows).to_csv(
        RESULTS / "operational_benchmark_contrasts.csv", index=False
    )
    pd.DataFrame(
        protocol_rows,
        columns=["step", "action", "required_output", "release_implementation"],
    ).to_csv(RESULTS / "spectral_audit_protocol.csv", index=False)
    (RESULTS / "supplement_tables.tex").write_text("\n".join(lines))
    print(f"Wrote {RESULTS / 'supplement_tables.tex'}")


if __name__ == "__main__":
    main()
