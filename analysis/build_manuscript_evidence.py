#!/usr/bin/env python3
"""Build the compact, machine-readable ledger used by the JoC manuscript."""

from __future__ import annotations

import hashlib
import json
import math
import platform
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd


PACKAGE = Path(__file__).resolve().parents[1]
RESULTS = PACKAGE / "results"
OUTPUT = RESULTS / "manuscript_evidence.json"


DAVIS_FIXED20_TARGET_MAP = {
    "ABL1": "abl1",
    "AKT1": "akt1",
    "AKT2": "akt2",
    "CDK2": "cdk2",
    "CSF1R": "csf1r",
    "EGFR": "egfr",
    "FGFR1": "fgfr1",
    "IGF1R": "igf1r",
    "JAK2": "jak2_jh1domain_catalytic",
    "KDR": "vegfr2",
    "KIT": "kit",
    "LCK": "lck",
    "MAP2K1": "mek1",
    "MAPK1": "erk2",
    "MAPK14": "p38_alpha",
    "MAPKAPK2": "mapkapk2",
    "MET": "met",
    "PLK1": "plk1",
    "ROCK1": "rock1",
    "SRC": "src",
}


DESCRIPTOR_EXTREME_ORDER = (
    "heavy_atoms",
    "molecular_weight",
    "labute_asa",
    "tpsa",
    "clogp",
    "rotatable_bonds",
    "ring_count",
)

DESCRIPTOR_EXTREME_FAMILIES = {
    "heavy_atoms": "molecular_size",
    "molecular_weight": "molecular_size",
    "labute_asa": "molecular_size",
    "tpsa": "polarity",
    "clogp": "lipophilicity",
    "rotatable_bonds": "flexibility",
    "ring_count": "ring_topology",
}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_safe(value: object) -> object:
    """Replace non-finite tabular sentinels before strict JSON serialization."""
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def csv_record(frame: pd.DataFrame, **selectors: object) -> dict:
    selected = frame
    for column, value in selectors.items():
        selected = selected.loc[selected[column].eq(value)]
    if len(selected) != 1:
        raise ValueError(f"expected one row for {selectors}, observed {len(selected)}")
    return selected.iloc[0].to_dict()


def dictionary_record(records: list[dict], **selectors: object) -> dict:
    """Select exactly one JSON record by a set of equality constraints."""
    selected = [
        record
        for record in records
        if all(record.get(key) == value for key, value in selectors.items())
    ]
    if len(selected) != 1:
        raise ValueError(
            f"expected one JSON record for {selectors}, observed {len(selected)}"
        )
    return selected[0]


def participation_ratio(values: np.ndarray) -> float:
    """Correlation-spectrum participation ratio for a complete matrix."""
    correlation = np.corrcoef(np.asarray(values, dtype=np.float64), rowvar=False)
    eigenvalues = np.linalg.eigvalsh(correlation)
    return float(eigenvalues.sum() ** 2 / np.square(eigenvalues).sum())


@lru_cache(maxsize=1)
def dockstring_input_audit() -> dict:
    """Separate DOCKSTRING release missingness from the complete analysis support."""
    path = PACKAGE / "data/frozen/dockstring-dataset.tsv.gz"
    scores = pd.read_csv(path, sep="\t", usecols=range(2, 60)).to_numpy(
        dtype=np.float64
    )
    missing = np.isnan(scores)
    complete_rows = ~missing.any(axis=1)
    clipped = np.minimum(scores, 0.0)

    complete_values = clipped[complete_rows].copy()
    complete_raw_pr = participation_ratio(complete_values)
    complete_values -= complete_values.mean(axis=1, keepdims=True)
    complete_residual_pr = participation_ratio(complete_values)
    del complete_values

    imputation_results: dict[str, dict[str, float]] = {}
    for name, column_values in (
        ("target_mean", np.nanmean(clipped, axis=0)),
        ("target_median", np.nanmedian(clipped, axis=0)),
    ):
        completed = clipped.copy()
        missing_indices = np.where(np.isnan(completed))
        completed[missing_indices] = column_values[missing_indices[1]]
        raw_pr = participation_ratio(completed)
        completed -= completed.mean(axis=1, keepdims=True)
        imputation_results[name] = {
            "raw_participation_ratio": raw_pr,
            "residual_participation_ratio": participation_ratio(completed),
        }

    return {
        "release_rows": int(scores.shape[0]),
        "analysis_rows": int(complete_rows.sum()),
        "targets": int(scores.shape[1]),
        "release_score_cells": int(scores.size),
        "source_missing_cells": int(missing.sum()),
        "source_missing_fraction": float(missing.mean()),
        "source_rows_with_any_missing": int((~complete_rows).sum()),
        "source_rows_with_any_missing_fraction": float((~complete_rows).mean()),
        "analysis_missing_cells": 0,
        "analysis_missing_fraction": 0.0,
        "source_strictly_positive_cells": int(np.sum(scores > 0)),
        "analysis_strictly_positive_cells": int(
            np.sum(scores[complete_rows] > 0)
        ),
        "complete_case": {
            "raw_participation_ratio": complete_raw_pr,
            "residual_participation_ratio": complete_residual_pr,
        },
        "all_rows_after_imputation": imputation_results,
        "interpretation": (
            "The source table contains 95 incomplete rows (0.0365% of molecules). "
            "Complete-case deletion and target-wise mean or median imputation give "
            "numerically indistinguishable headline spectra."
        ),
    }


@lru_cache(maxsize=1)
def davis_fixed20_measurement_audit() -> dict:
    """Quantify the pKd=5 measurement-limit pile-up on the exact fixed panel."""
    path = PACKAGE / "data/frozen/davis_complete.tab.gz"
    frame = pd.read_csv(path, sep="\t", usecols=["drug_name", "protein", "y"])
    selected = frame.loc[frame.protein.isin(DAVIS_FIXED20_TARGET_MAP.values())]
    matrix = selected.pivot(index="drug_name", columns="protein", values="y")
    matrix = matrix.reindex(columns=list(DAVIS_FIXED20_TARGET_MAP.values()))
    matrix.columns = list(DAVIS_FIXED20_TARGET_MAP)
    if matrix.shape != (72, 20) or matrix.isna().any().any():
        raise ValueError(f"unexpected fixed-20 DAVIS support: {matrix.shape}")

    at_floor = matrix.eq(5.0)
    target_fractions = at_floor.mean(axis=0)
    target_counts = at_floor.sum(axis=0)
    high = target_fractions[target_fractions > 0.90]
    return {
        "ligands": int(matrix.shape[0]),
        "targets": int(matrix.shape[1]),
        "cells": int(matrix.size),
        "pkd_floor": 5.0,
        "corresponding_kd_nm_limit": 10_000,
        "cells_at_pkd_floor": int(at_floor.to_numpy().sum()),
        "fraction_at_pkd_floor": float(at_floor.to_numpy().mean()),
        "targetwise_fraction_at_floor_range": [
            float(target_fractions.min()),
            float(target_fractions.max()),
        ],
        "targets_above_90_percent_at_floor": int(len(high)),
        "high_floor_targets": {
            target: {
                "cells_at_floor": int(target_counts[target]),
                "fraction_at_floor": float(target_fractions[target]),
                "cells_above_floor": int(matrix.shape[0] - target_counts[target]),
            }
            for target in high.sort_values(ascending=False).index
        },
        "interpretation": (
            "The released 10,000-nM upper measurement bound becomes a pKd=5 "
            "floor. Correlations involving the most affected targets are based on "
            "few values above that floor."
        ),
    }


def build() -> dict:
    evidence = read_json(RESULTS / "evidence_summary.json")
    fixed20 = read_json(RESULTS / "fixed20_estimand_decomposition/summary.json")
    fixed20_reference = read_json(
        RESULTS / "fixed20_reference_sensitivity/summary.json"
    )
    ranking_centering_panel = read_json(
        RESULTS / "ranking_centering_panel_sensitivity/summary.json"
    )
    kirhub = read_json(RESULTS / "kirhub_external_validation/summary.json")
    davis_censoring = read_json(
        RESULTS / "davis_fixed20_censoring_sensitivity/summary.json"
    )
    strict_klifs = read_json(RESULTS / "strict_klifs_group_qap/summary.json")
    centering_fixed20 = read_json(
        RESULTS / "fixed20_centering_panel_sensitivity/summary.json"
    )
    centering_panel_specific = read_json(
        RESULTS / "centering_panel_sensitivity/summary.json"
    )
    chemistry = read_json(
        RESULTS / "experimental_chemical_context_geometry/summary.json"
    )
    experimental_overlap = read_json(
        RESULTS / "experimental_panel_overlap/summary.json"
    )
    spd = read_json(RESULTS / "spd_external_validation/summary.json")
    overlap = read_json(RESULTS / "cross_panel_overlap_audit/summary.json")
    pair_transfer = read_json(
        RESULTS / "pair_specific_normalization_transfer/summary.json"
    )
    dense_davis = read_json(RESULTS / "dense_davis_benchmark.json")
    dense_pkis2 = read_json(RESULTS / "dense_pkis2_benchmark.json")
    residual_mechanism = read_json(
        RESULTS / "residual_mechanism/analysis_summary.json"
    )
    chemical_context_geometry = read_json(
        RESULTS / "chemical_context_geometry/summary.json"
    )
    descriptor_component = read_json(
        RESULTS / "descriptor_component_geometry/summary.json"
    )
    descriptor_rank_controls = read_json(
        RESULTS / "descriptor_rank_matched_controls/summary.json"
    )
    descriptor_uncertainty = read_json(
        RESULTS / "descriptor_correlation_reduction_uncertainty/summary.json"
    )
    target_indexed_descriptor_control = read_json(
        RESULTS / "target_blind_descriptor_control/summary.json"
    )
    released_ledger = read_json(
        RESULTS / "released_pair_geometry_ledger/summary.json"
    )
    dockstring_chembl_ranking = read_json(
        RESULTS / "dockstring_chembl_ranking/summary.json"
    )
    nonvina_transport = read_json(
        RESULTS / "nonvina_scorer_transport/summary.json"
    )
    vina_terms = read_json(RESULTS / "dockstring_vina_terms/analysis_summary.json")
    equivalence_margin = read_json(
        RESULTS / "equivalence_margin_sensitivity/summary.json"
    )
    partial = pd.read_csv(RESULTS / "klifs_pocket_control/partial_geometry_qap.csv")
    retrieval = pd.read_csv(
        RESULTS / "klifs_pocket_control/locked_endpoint_retrieval.csv"
    )
    strict = pd.read_csv(
        RESULTS / "strict_klifs_group_qap/restricted_partial_qap.csv"
    )
    sequence = pd.read_csv(RESULTS / "sequence_docking_fusion/panel_metrics.csv")
    recovery = pd.read_csv(RESULTS / "calibration_panel_recovery/summary.csv")
    recovery_groups = pd.read_csv(
        RESULTS / "calibration_panel_recovery/scaffold_holdout_summary.csv"
    )
    missing = pd.read_csv(RESULTS / "missing_residual_sensitivity.csv")
    covariance = pd.read_csv(RESULTS / "correlation_covariance_sensitivity.csv")
    descriptor_extremes = pd.read_csv(
        RESULTS / "chemical_context_geometry/descriptor_extremes.csv"
    )

    docking44_input = evidence["preprocessing_sensitivity"][
        "input_characterization"
    ]
    dockstring_audit = dockstring_input_audit()
    dockstring_audit["positive_score_clipping_sensitivity"] = evidence[
        "preprocessing_sensitivity"
    ]["dockstring_clipping"]
    large_matrix_input_audits = {
        "Docking-44": {
            "release_rows": docking44_input["n_ligands"],
            "analysis_rows": docking44_input["n_ligands"],
            "targets": docking44_input["n_targets"],
            "source_missing_cells": docking44_input["missing_cells"],
            "source_missing_fraction": docking44_input["missing_fraction"],
            "source_rows_with_any_missing": docking44_input[
                "rows_with_any_missing"
            ],
            "analysis_missing_cells_after_target_mean_imputation": 0,
            "strictly_positive_cells_in_frozen_export": docking44_input[
                "strictly_positive_cells_before_primary_clipping"
            ],
            "cells_exactly_zero_in_frozen_export": docking44_input[
                "zero_after_primary_clipping_cells"
            ],
            "preprocessing": (
                "Upstream positive-score replacement was already encoded in the "
                "frozen export; target-mean imputation retained all rows."
            ),
        },
        "DOCKSTRING-58": dockstring_audit,
    }
    experimental_measurement_audits = {
        "DAVIS": davis_fixed20_measurement_audit(),
        "KiRHub": kirhub["KiRHub_measurement_audit"],
    }

    spectral: dict[str, object] = {}
    physicochemical: dict[str, object] = {}
    for key, label in (("docking44", "Docking-44"), ("dockstring58", "DOCKSTRING-58")):
        surfaces = evidence["spectral_estimand_sensitivity"][key]
        pc1 = evidence["pc1_axis"][key]
        null = evidence["row_norm_preserving_residual_null"][key]
        spectral[label] = {
            "correlation": surfaces["correlation"],
            "covariance": surfaces["covariance"],
            "pc1_axis": pc1,
            "row_norm_preserving_null": null,
        }
        slopes = evidence["physicochemical_target_slopes"][key]
        physicochemical[label] = {
            "support_n": slopes["support_n"],
            "support_rule": slopes["support_rule"],
            "conditional_model": slopes["conditional_model"],
            "descriptor_slopes": slopes["descriptor_slopes"],
            "most_imputed_target_exclusion_sensitivity": slopes.get(
                "most_imputed_target_exclusion_sensitivity"
            ),
        }

    structural = {
        # Pandas' JSON encoder maps missing secondary metrics to JSON null,
        # unlike ``to_dict`` which leaves non-standard NaN literals.
        "continuous_panel_metrics": json.loads(sequence.to_json(orient="records")),
        "unrestricted_partial_qap": {
            "three_panel_mean": csv_record(
                partial,
                endpoint="old_three_panel_mean",
                controls="sequence_plus_pocket",
            ),
            "KiRHub": csv_record(
                partial,
                endpoint="KiRHub",
                controls="sequence_plus_pocket",
            ),
        },
        "within_KLIFS_group_partial_qap": {
            "three_panel_mean": csv_record(
                strict,
                endpoint="old_three_panel_mean",
                controls="sequence_plus_KLIFS_pocket",
                permutation_scheme="all_non_singleton_KLIFS_groups",
            ),
            "KiRHub": csv_record(
                strict,
                endpoint="KiRHub",
                controls="sequence_plus_KLIFS_pocket",
                permutation_scheme="all_non_singleton_KLIFS_groups",
            ),
        },
        "locked_pair_baselines": {
            row.predictor: row._asdict()
            for row in retrieval.itertuples(index=False)
        },
    }

    recovery_selected: dict[str, dict[str, object]] = {}
    for dataset in ("DOCKSTRING-58", "Docking-44"):
        recovery_selected[dataset] = {
            str(size): csv_record(
                recovery, dataset=dataset, calibration_ligands=size
            )
            for size in (50, 100, 200, 500, 1000, 2000, 5000)
        }
        recovery_selected[dataset]["chemical_group_holdout_200"] = csv_record(
            recovery_groups, dataset=dataset, calibration_ligands=200
        )
        recovery_selected[dataset]["chemical_group_holdout_500"] = csv_record(
            recovery_groups, dataset=dataset, calibration_ligands=500
        )

    ranking = evidence["expanded_target_preference_benchmark"]["primary_all_exact"]
    ranking_selected = {
        "n_ligands": ranking["n_ligands"],
        "n_targets": ranking["n_targets"],
        "observed_experimental_cells": ranking["observed_experimental_cells"],
        "target_pair_coobservations": ranking["target_pair_coobservations"],
        "fit_support": ranking["fit_support"],
        "representations": {
            key: ranking["representations"][key]
            for key in (
                "absolute_vina",
                "target_centered_unscaled",
                "two_way_centered_unscaled",
                "column_standardized",
                "two_way_residual",
                "docking_target_prior",
                "experimental_target_prior",
                "cohort_experimental_target_prior",
            )
        },
        "paired_comparisons": {
            key: ranking["paired_comparisons"][key]
            for key in (
                "two_way_residual_minus_absolute_vina",
                "two_way_residual_minus_column_standardized",
                "two_way_residual_minus_docking_target_prior",
                "two_way_residual_minus_experimental_target_prior",
                "two_way_residual_minus_cohort_experimental_target_prior",
                "absolute_vina_minus_docking_target_prior",
                "two_way_centered_unscaled_minus_target_centered_unscaled",
            )
        },
        "coverage_sensitivity": ranking["coverage_sensitivity"],
    }

    dense_ranking = {
        "DAVIS": {
            "support": {
                "matched_ligands": dense_davis["source_support"][
                    "primary_full_key_ligands"
                ],
                "targets": dense_davis["source_support"]["shared_targets"],
                "evaluated_ligands": dense_davis["primary"]["representations"][
                    "absolute_vina"
                ]["0.0"]["all_informative"]["evaluated_ligands"],
                "evaluated_pairs": dense_davis["primary"]["representations"][
                    "absolute_vina"
                ]["0.0"]["all_informative"]["evaluated_pairs"],
            },
            "endpoint": (
                "pKd ordering after excluding exact pKd ties and pairs for which "
                "both measurements were at the pKd=5 reporting floor"
            ),
            "analysis_parameters": dense_davis["analysis_parameters"],
            "representations": {
                key: dense_davis["primary"]["representations"][key]["0.0"]
                ["all_informative"]
                for key in (
                    "docking_target_prior",
                    "absolute_vina",
                    "target_centered_unscaled",
                    "column_standardized",
                    "target_centered_residual_scaled",
                    "two_way_residual",
                )
            },
            "paired_comparisons": {
                key: value["0.0"]["all_informative"]
                for key, value in dense_davis["primary"][
                    "paired_comparisons"
                ].items()
            },
            "target_composition_sensitivity": dense_davis["target_jackknife"][
                "by_pair_stratum"
            ]["all_informative"],
            "operational_path_steps": dense_davis["primary"][
                "operational_path_steps"
            ],
            "outcome_conditioned_strata": {
                stratum: {
                    "status": dense_davis["primary"]["pair_strata_status"][
                        stratum
                    ],
                    "representations": {
                        key: dense_davis["primary"]["representations"][key][
                            "0.0"
                        ][stratum]
                        for key in (
                            "docking_target_prior",
                            "absolute_vina",
                            "target_centered_unscaled",
                            "target_centered_residual_scaled",
                            "two_way_residual",
                        )
                    },
                    "paired_comparisons": {
                        key: dense_davis["primary"]["paired_comparisons"][key][
                            "0.0"
                        ][stratum]
                        for key in (
                            "absolute_vina_minus_docking_target_prior",
                            "target_centered_unscaled_minus_absolute_vina",
                            "target_centered_residual_scaled_minus_target_centered_unscaled",
                            "two_way_residual_minus_target_centered_residual_scaled",
                            "two_way_residual_minus_absolute_vina",
                        )
                    },
                }
                for stratum in ("both_uncensored", "floor_vs_uncensored")
            },
        },
        "PKIS2": {
            "support": {
                "matched_ligands": dense_pkis2["source_support"][
                    "primary_full_key_ligands"
                ],
                "targets": dense_pkis2["source_support"]["shared_targets"],
                "evaluated_ligands": dense_pkis2["primary"]["representations"][
                    "absolute_vina"
                ]["absolute_difference_gt_10"]["evaluated_ligands"],
                "evaluated_pairs": dense_pkis2["primary"]["representations"][
                    "absolute_vina"
                ]["absolute_difference_gt_10"]["evaluated_pairs"],
            },
            "endpoint": (
                "single-dose percentage-inhibition ordering after excluding "
                "target differences of 10 percentage points or less"
            ),
            "analysis_parameters": dense_pkis2["analysis_parameters"],
            "representations": {
                key: dense_pkis2["primary"]["representations"][key][
                    "absolute_difference_gt_10"
                ]
                for key in (
                    "docking_target_prior",
                    "absolute_vina",
                    "target_centered_unscaled",
                    "column_standardized",
                    "target_centered_residual_scaled",
                    "two_way_residual",
                )
            },
            "paired_comparisons": {
                key: value["absolute_difference_gt_10"]
                for key, value in dense_pkis2["primary"][
                    "paired_comparisons"
                ].items()
            },
            "target_composition_sensitivity": dense_pkis2["target_jackknife"][
                "two_way_residual_minus_absolute_vina"
            ]["pairwise_concordance__absolute_difference_gt_10"],
            "operational_path_steps": dense_pkis2["primary"][
                "operational_path_steps"
            ],
            "outcome_conditioned_strata": {
                "both_active": {
                    "status": dense_pkis2["primary"]["both_active_status"],
                    "experimental_margin_name": "absolute_difference_gt_10",
                    "experimental_margin_percentage_points": 10.0,
                    "representations": dense_pkis2["primary"][
                        "exploratory_both_active_pairwise_concordance"
                    ],
                    "paired_comparisons": dense_pkis2["primary"][
                        "exploratory_both_active_paired_comparisons"
                    ],
                },
                "secondary_exact_non_tie_both_active": {
                    "status": dense_pkis2["primary"][
                        "secondary_exact_non_tie_both_active_status"
                    ],
                    "experimental_margin_name": "exact_non_ties",
                    "experimental_margin_percentage_points": 0.0,
                    "representations": dense_pkis2["primary"][
                        "secondary_exact_non_tie_both_active_pairwise_concordance"
                    ],
                    "paired_comparisons": dense_pkis2["primary"][
                        "secondary_exact_non_tie_both_active_paired_comparisons"
                    ],
                },
            },
        },
    }

    descriptor_residualization: dict[str, object] = {
        "analysis_status": "exploratory_predictive_decomposition",
        "descriptor_names_in_order": residual_mechanism["descriptor_names"],
        "method": (
            "Five-fold chemical-group-held-out linear prediction with target "
            "offsets, residual target scales, descriptor scaling and regression "
            "coefficients all fitted within each training fold. The reported "
            "surface pools out-of-fold residual score errors."
        ),
        "interpretation_boundary": (
            "A reduction in target-correlation concentration after descriptor "
            "removal is a predictive decomposition of docking-score structure, "
            "not causal attribution or validation of binding specificity."
        ),
        "datasets": {},
    }
    for dataset in ("Docking-44", "DOCKSTRING-58"):
        mechanism_dataset = residual_mechanism["datasets"][dataset]
        mechanism_metrics = mechanism_dataset["descriptor_decomposition"]
        context_decomposition = chemical_context_geometry[
            "descriptor_decomposition"
        ][dataset]
        context_metrics = context_decomposition[
            "descriptor_decomposition_metrics"
        ]
        if mechanism_metrics != context_metrics:
            raise ValueError(
                f"{dataset}: descriptor-decomposition summaries disagree"
            )
        fold_diagnostics = context_decomposition["fold_diagnostics"]
        if len(fold_diagnostics) != mechanism_metrics["folds"]:
            raise ValueError(f"{dataset}: descriptor fold count is inconsistent")
        descriptor_residualization["datasets"][dataset] = {
            "support": {
                "source_matrix_ligands": mechanism_dataset["n_ligands"],
                "analysis_ligands": mechanism_dataset[
                    "descriptor_support_ligands"
                ],
                "targets": mechanism_dataset["n_targets"],
                "chemical_groups": mechanism_metrics["chemical_groups"],
                "chemical_group_definition": mechanism_dataset[
                    "group_definition"
                ],
                "support_rule": mechanism_dataset["support_rule"],
            },
            "fold_contract": {
                "folds": mechanism_metrics["folds"],
                "group_overlap_counts": [
                    int(record["group_overlap_count"])
                    for record in fold_diagnostics
                ],
                "train_ligands": [
                    int(record["train_ligands"])
                    for record in fold_diagnostics
                ],
                "test_ligands": [
                    int(record["test_ligands"])
                    for record in fold_diagnostics
                ],
                "train_chemical_groups": [
                    int(record["train_groups"])
                    for record in fold_diagnostics
                ],
                "test_chemical_groups": [
                    int(record["test_groups"])
                    for record in fold_diagnostics
                ],
                "strict_fold_local_target_offsets": mechanism_metrics[
                    "strict_fold_local_target_offsets"
                ],
                "strict_fold_local_residual_target_scales": mechanism_metrics[
                    "strict_fold_local_residual_target_scales"
                ],
                "strict_fold_local_descriptor_scaling": mechanism_metrics[
                    "strict_fold_local_descriptor_scaling"
                ],
                "strict_fold_local_regression_coefficients": mechanism_metrics[
                    "strict_fold_local_regression_coefficients"
                ],
            },
            "metrics": mechanism_metrics,
            "identical_metrics_in_both_source_summaries": True,
        }

    mw_results: dict[str, object] = {}
    context_primary = chemical_context_geometry["primary_results"]
    within_band_records = chemical_context_geometry[
        "within_mw_band_reproducibility_summary"
    ]

    def within_band_payload(
        dataset: str, transformation: str
    ) -> dict[str, object]:
        bands: dict[str, object] = {}
        for band in ("low_mw", "high_mw"):
            controls: dict[str, object] = {}
            for control_type in (
                "row_random_disjoint",
                "mw_stratified_chemical_group_disjoint",
            ):
                record = dictionary_record(
                    within_band_records,
                    dataset=dataset,
                    mw_band=band,
                    control_type=control_type,
                    transformation=transformation,
                )
                controls[control_type] = {
                    "repetitions": int(record["repetitions"]),
                    "geometry_spearman_mean": record[
                        "geometry_spearman_mean"
                    ],
                    "geometry_spearman_median": record[
                        "geometry_spearman_median"
                    ],
                    "central_95_percent_repeated_split_sensitivity_range": [
                        record["geometry_spearman_q025"],
                        record["geometry_spearman_q975"],
                    ],
                    "mean_first_half_ligands": record[
                        "first_ligands_mean"
                    ],
                    "mean_second_half_ligands": record[
                        "second_ligands_mean"
                    ],
                    "mean_chemical_group_overlap": record[
                        "chemical_group_overlap_mean"
                    ],
                    "maximum_chemical_group_overlap": int(
                        record["chemical_group_overlap_maximum"]
                    ),
                    "mean_absolute_median_mw_difference": record[
                        "absolute_median_mw_difference_mean"
                    ],
                    "mean_mw_ks_statistic": record[
                        "mw_ks_statistic_mean"
                    ],
                    "range_interpretation": record[
                        "interval_interpretation"
                    ],
                }
            bands[band] = controls
        return bands

    for dataset in ("Docking-44", "DOCKSTRING-58"):
        raw = dictionary_record(
            context_primary, dataset=dataset, transformation="raw"
        )
        residual = dictionary_record(
            context_primary,
            dataset=dataset,
            transformation="row_centered_residual",
        )
        adjusted = dictionary_record(
            context_primary,
            dataset=dataset,
            transformation="group_heldout_descriptor_adjusted_residual",
        )
        correlated_size_records = {
            record["descriptor"]: {
                "geometry_spearman": record["geometry_spearman"],
                "sign_flip_fraction": record["sign_flip_fraction"],
                "top_10_percent_pair_jaccard": record[
                    "top_positive_10pct_pair_jaccard"
                ],
            }
            for record in chemical_context_geometry["size_descriptor_replication"]
            if record["dataset"] == dataset
            and record["transformation"] == "row_centered_residual"
        }
        quintile_trends = {
            record["transformation"]: {
                "mw_quintile_pairs": record["mw_quintile_pairs"],
                "spearman_median_mw_separation_vs_geometry_dissimilarity": record[
                    "spearman_median_mw_separation_vs_geometry_dissimilarity"
                ],
            }
            for record in chemical_context_geometry["mw_quintile_geometry_trends"]
            if record["dataset"] == dataset
            and record["transformation"]
            in (
                "row_centered_residual",
                "group_heldout_descriptor_adjusted_residual",
            )
        }
        context_support = chemical_context_geometry["datasets"][dataset]
        raw_within_band = within_band_payload(dataset, "raw")
        residual_within_band = within_band_payload(
            dataset, "row_centered_residual"
        )
        mw_results[dataset] = {
            "support": {
                "analysis_ligands": context_support["ligands"],
                "targets": context_support["targets"],
                "chemical_groups": context_support["chemical_groups"],
                "chemical_group_definition": context_support[
                    "group_definition"
                ],
                "support_rule": context_support["support_rule"],
            },
            "source_preprocessing_counts": {
                "missing_cells_before_preprocessing": context_support[
                    "missing_cells_before_preprocessing"
                ],
                "strictly_positive_cells_clipped_before_analysis": context_support[
                    "positive_cells_clipped"
                ],
                "scope_note": (
                    "For DOCKSTRING-58 these counts characterize the source or "
                    "complete-case matrix before the fixed 15,000-ligand support "
                    "was sampled; for Docking-44 they characterize the full "
                    "analysis support."
                ),
            },
            "low_mw_domain": {
                "ligands": residual["low_ligands"],
                "inclusive_upper_threshold": residual[
                    "low_threshold_inclusive"
                ],
                "median_molecular_weight": residual["low_descriptor_median"],
                "target_correlation_participation_ratio": residual["first_pr"],
                "mean_absolute_target_correlation": residual[
                    "first_mean_absolute_correlation"
                ],
            },
            "high_mw_domain": {
                "ligands": residual["high_ligands"],
                "inclusive_lower_threshold": residual[
                    "high_threshold_inclusive"
                ],
                "median_molecular_weight": residual["high_descriptor_median"],
                "target_correlation_participation_ratio": residual["second_pr"],
                "mean_absolute_target_correlation": residual[
                    "second_mean_absolute_correlation"
                ],
            },
            "row_centered_residual_geometry": {
                "spearman": residual["geometry_spearman"],
                "dissimilarity_one_minus_spearman": residual[
                    "geometry_dissimilarity"
                ],
                "target_pair_sign_flip_fraction": residual[
                    "sign_flip_fraction"
                ],
                "top_10_percent_pair_jaccard": residual[
                    "top_positive_10pct_pair_jaccard"
                ],
                "chemical_group_bootstrap": {
                    "repetitions": int(
                        residual[
                            "chemical_group_bootstrap_repetitions"
                        ]
                    ),
                    "median": residual[
                        "chemical_group_bootstrap_geometry_spearman_median"
                    ],
                    "central_95_percent_interval": [
                        residual[
                            "chemical_group_bootstrap_geometry_spearman_q025"
                        ],
                        residual[
                            "chemical_group_bootstrap_geometry_spearman_q975"
                        ],
                    ],
                },
                "random_disjoint_support_control": {
                    "repetitions": residual["random_disjoint_repetitions"],
                    "mean": residual["random_disjoint_geometry_spearman_mean"],
                    "central_95_percent_interval": [
                        residual["random_disjoint_geometry_spearman_q025"],
                        residual["random_disjoint_geometry_spearman_q975"],
                    ],
                    "finite_support_lower_tail_probability": residual[
                        "random_lower_tail_probability"
                    ],
                },
                "mw_matched_chemical_group_disjoint_control": {
                    "repetitions": residual[
                        "mw_stratified_group_disjoint_repetitions"
                    ],
                    "mean": residual[
                        "mw_stratified_group_disjoint_geometry_spearman_mean"
                    ],
                    "central_95_percent_interval": [
                        residual[
                            "mw_stratified_group_disjoint_geometry_spearman_q025"
                        ],
                        residual[
                            "mw_stratified_group_disjoint_geometry_spearman_q975"
                        ],
                    ],
                    "mean_mw_ks_distance": residual[
                        "mw_stratified_group_disjoint_mw_ks_mean"
                    ],
                    "mean_absolute_median_mw_difference": residual[
                        "mw_stratified_group_disjoint_absolute_median_mw_difference_mean"
                    ],
                },
                "restriction_matched_within_band_reproducibility": (
                    residual_within_band
                ),
            },
            "raw_surface_geometry_spearman": raw["geometry_spearman"],
            "raw_surface_geometry": {
                "spearman": raw["geometry_spearman"],
                "dissimilarity_one_minus_spearman": raw[
                    "geometry_dissimilarity"
                ],
                "target_pair_sign_flip_fraction": raw[
                    "sign_flip_fraction"
                ],
                "chemical_group_bootstrap": {
                    "repetitions": int(
                        raw["chemical_group_bootstrap_repetitions"]
                    ),
                    "median": raw[
                        "chemical_group_bootstrap_geometry_spearman_median"
                    ],
                    "central_95_percent_interval": [
                        raw["chemical_group_bootstrap_geometry_spearman_q025"],
                        raw["chemical_group_bootstrap_geometry_spearman_q975"],
                    ],
                },
                "random_disjoint_support_control": {
                    "repetitions": raw["random_disjoint_repetitions"],
                    "mean": raw[
                        "random_disjoint_geometry_spearman_mean"
                    ],
                    "central_95_percent_interval": [
                        raw["random_disjoint_geometry_spearman_q025"],
                        raw["random_disjoint_geometry_spearman_q975"],
                    ],
                    "finite_support_lower_tail_probability": raw[
                        "random_lower_tail_probability"
                    ],
                },
                "mw_matched_chemical_group_disjoint_control": {
                    "repetitions": raw[
                        "mw_stratified_group_disjoint_repetitions"
                    ],
                    "mean": raw[
                        "mw_stratified_group_disjoint_geometry_spearman_mean"
                    ],
                    "central_95_percent_interval": [
                        raw[
                            "mw_stratified_group_disjoint_geometry_spearman_q025"
                        ],
                        raw[
                            "mw_stratified_group_disjoint_geometry_spearman_q975"
                        ],
                    ],
                    "mean_mw_ks_distance": raw[
                        "mw_stratified_group_disjoint_mw_ks_mean"
                    ],
                    "mean_absolute_median_mw_difference": raw[
                        "mw_stratified_group_disjoint_absolute_median_mw_difference_mean"
                    ],
                },
                "restriction_matched_within_band_reproducibility": (
                    raw_within_band
                ),
            },
            "strict_descriptor_adjusted_residual_sensitivity": {
                "geometry_spearman": adjusted["geometry_spearman"],
                "chemical_group_bootstrap_central_95_percent_interval": [
                    adjusted[
                        "chemical_group_bootstrap_geometry_spearman_q025"
                    ],
                    adjusted[
                        "chemical_group_bootstrap_geometry_spearman_q975"
                    ],
                ],
            },
            "correlated_size_descriptor_sensitivity": correlated_size_records,
            "mw_quintile_distance_trends": quintile_trends,
        }

    dockstring_seed_records = chemical_context_geometry[
        "dockstring_support_seed_sensitivity"
    ]
    if any(
        record["transformation"] != "row_centered_residual"
        for record in dockstring_seed_records
    ):
        raise ValueError("DOCKSTRING MW support-seed sensitivity changed estimand")
    dockstring_seed_spearman = [
        float(record["geometry_spearman"])
        for record in dockstring_seed_records
    ]
    mw_results["DOCKSTRING-58"]["support_seed_sensitivity"] = {
        "sample_ligands_per_seed": sorted(
            {int(record["sample_ligands"]) for record in dockstring_seed_records}
        ),
        "seeds": [int(record["sample_seed"]) for record in dockstring_seed_records],
        "geometry_spearman_range": [
            min(dockstring_seed_spearman),
            max(dockstring_seed_spearman),
        ],
    }

    descriptor_family_datasets: dict[str, object] = {}
    for dataset in ("Docking-44", "DOCKSTRING-58"):
        rows = descriptor_extremes.loc[
            descriptor_extremes["dataset"].eq(dataset)
            & descriptor_extremes["transformation"].eq(
                "row_centered_residual"
            )
        ]
        if set(rows["descriptor"]) != set(DESCRIPTOR_EXTREME_ORDER):
            raise ValueError(
                f"{dataset}: residual descriptor-extreme set is incomplete"
            )
        if len(rows) != len(DESCRIPTOR_EXTREME_ORDER):
            raise ValueError(
                f"{dataset}: residual descriptor-extreme rows are not unique"
            )
        if set(rows["analysis_status"]) != {"exploratory_post_hoc"}:
            raise ValueError(
                f"{dataset}: descriptor-extreme status is not post hoc"
            )

        mw_matched = mw_results[dataset]["row_centered_residual_geometry"][
            "mw_matched_chemical_group_disjoint_control"
        ]
        records: dict[str, object] = {}
        for descriptor_name in DESCRIPTOR_EXTREME_ORDER:
            row = csv_record(
                rows,
                descriptor=descriptor_name,
            )
            records[descriptor_name] = {
                "family": DESCRIPTOR_EXTREME_FAMILIES[descriptor_name],
                "source_analysis_status": row["analysis_status"],
                "extreme_fraction_per_tail": float(
                    row["extreme_fraction_per_tail"]
                ),
                "low_threshold_inclusive": float(
                    row["low_threshold_inclusive"]
                ),
                "high_threshold_inclusive": float(
                    row["high_threshold_inclusive"]
                ),
                "low_ligands": int(row["low_ligands"]),
                "high_ligands": int(row["high_ligands"]),
                "low_descriptor_median": float(
                    row["low_descriptor_median"]
                ),
                "high_descriptor_median": float(
                    row["high_descriptor_median"]
                ),
                "low_domain_residual_participation_ratio": float(
                    row["first_pr"]
                ),
                "high_domain_residual_participation_ratio": float(
                    row["second_pr"]
                ),
                "residual_map_spearman": float(row["geometry_spearman"]),
                "target_pair_sign_flip_fraction": float(
                    row["sign_flip_fraction"]
                ),
                "top_10_percent_pair_jaccard": float(
                    row["top_positive_10pct_pair_jaccard"]
                ),
                "below_mw_matched_control_mean": bool(
                    row["geometry_spearman"] < mw_matched["mean"]
                ),
            }

        ordered_by_agreement = sorted(
            records,
            key=lambda descriptor_name: records[descriptor_name][
                "residual_map_spearman"
            ],
        )
        three_lowest = ordered_by_agreement[:3]
        molecular_size_descriptors = {
            name
            for name, family in DESCRIPTOR_EXTREME_FAMILIES.items()
            if family == "molecular_size"
        }
        descriptor_family_datasets[dataset] = {
            "mw_matched_chemical_group_disjoint_control": mw_matched,
            "descriptor_results": records,
            "three_lowest_geometry_spearman_descriptors": three_lowest,
            "molecular_size_family_forms_three_lowest_geometry_agreements": (
                set(three_lowest) == molecular_size_descriptors
            ),
            "all_seven_geometry_spearman_below_mw_matched_control_mean": all(
                record["below_mw_matched_control_mean"]
                for record in records.values()
            ),
            "molecular_weight_residual_pr_direction": (
                "lower_in_high_mw_domain"
                if records["molecular_weight"][
                    "high_domain_residual_participation_ratio"
                ]
                < records["molecular_weight"][
                    "low_domain_residual_participation_ratio"
                ]
                else "higher_in_high_mw_domain"
            ),
        }

    descriptor_family_stratification = {
        "analysis_status": "exploratory_post_hoc_family_sensitivity",
        "source_artifact": (
            "results/chemical_context_geometry/descriptor_extremes.csv"
        ),
        "transformation": "row_centered_residual",
        "extreme_fraction_per_tail": 0.25,
        "descriptor_names_in_order": list(DESCRIPTOR_EXTREME_ORDER),
        "descriptor_families": DESCRIPTOR_EXTREME_FAMILIES,
        "selection_boundary": (
            "All seven stratifiers came from the already inspected descriptor "
            "set; the family comparison was assembled post hoc and is "
            "descriptive rather than confirmatory. Groups use inclusive empirical "
            "25th- and 75th-percentile thresholds, so every boundary tie is retained."
        ),
        "size_family_interpretation": (
            "Heavy-atom count, molecular weight and Labute ASA form the three "
            "lowest residual-map agreements in both panels, consistent with a "
            "broad molecular-size family rather than a molecular-weight-only "
            "effect. Other descriptor splits also remain below the MW-matched "
            "chemical-group-disjoint control, so this ordering does not identify "
            "a unique mechanism."
        ),
        "effective_dimension_interpretation": (
            "Across the molecular-weight split, residual PR decreases from the "
            "low- to the high-MW domain in Docking-44 but increases in "
            "DOCKSTRING-58. Effective dimension therefore has no common monotone "
            "direction with molecular size in these two post-hoc comparisons."
        ),
        "all_datasets_all_seven_below_mw_matched_control_mean": all(
            record[
                "all_seven_geometry_spearman_below_mw_matched_control_mean"
            ]
            for record in descriptor_family_datasets.values()
        ),
        "datasets": descriptor_family_datasets,
    }

    exploratory_chemical_domain_structure = {
        "analysis_status": chemical_context_geometry["analysis_status"],
        "claim_boundary": chemical_context_geometry["claim_boundary"],
        "descriptor_residualization": descriptor_residualization,
        "low_high_molecular_weight_domain_instability": {
            "analysis_status": "exploratory_post_hoc",
            "primary_stratifier": chemical_context_geometry[
                "primary_stratifier"
            ],
            "stratifier_selection": chemical_context_geometry[
                "primary_stratifier_selection"
            ],
            "extreme_fraction_per_tail": chemical_context_geometry[
                "primary_extreme_fraction_per_tail"
            ],
            "geometry_definition": (
                "Spearman correlation between the strict upper triangles of "
                "the target-correlation matrices estimated in the lowest and "
                "highest molecular-weight domains."
            ),
            "transformations": chemical_context_geometry["transformations"],
            "random_disjoint_control": {
                "scheme": chemical_context_geometry["random_null"],
                "repetitions": chemical_context_geometry[
                    "random_disjoint_repetitions"
                ],
            },
            "chemical_group_bootstrap": {
                **chemical_context_geometry["group_bootstrap"],
                "repetitions": chemical_context_geometry[
                    "chemical_group_bootstrap_repetitions"
                ],
            },
            "mw_matched_chemical_group_disjoint_control": {
                **chemical_context_geometry[
                    "mw_stratified_group_disjoint_control"
                ],
                "repetitions": chemical_context_geometry[
                    "mw_stratified_group_disjoint_repetitions"
                ],
            },
            "restriction_matched_within_band_reproducibility": {
                **chemical_context_geometry[
                    "within_mw_band_reproducibility_control"
                ],
                "row_random_repetitions_per_band": (
                    chemical_context_geometry[
                        "within_mw_band_random_repetitions"
                    ]
                ),
                "chemical_group_disjoint_repetitions_per_band": (
                    chemical_context_geometry[
                        "within_mw_band_group_disjoint_repetitions"
                    ]
                ),
            },
            "seed": chemical_context_geometry["seed"],
            "datasets": mw_results,
        },
        "post_hoc_descriptor_family_stratification": (
            descriptor_family_stratification
        ),
        "source_agreement": (
            "The descriptor-decomposition metrics are exactly identical "
            "between the two serialized analysis summaries."
        ),
    }

    fixed20_qap_resolution: dict[str, object] = {
        "interpretation": (
            "For each fixed-panel paired target-label QAP, the reported value is "
            "the maximum absolute endpoint of its empirical central 95% null "
            "interval. It is a conservative randomization-resolution bound on this "
            "fixed matrix, not a power-based minimum detectable effect, confidence "
            "interval, or effective number of independent target pairs."
        ),
        "panels": {},
    }
    for panel, panel_record in fixed20["panels"].items():
        records: dict[str, object] = {}
        for label, source_key in (
            (
                "two_way_centered_experimental_geometry",
                "paired_qap_for_docking_increment",
            ),
            (
                "raw_experimental_geometry",
                "paired_qap_for_docking_increment_with_raw_experiment",
            ),
        ):
            record = panel_record[source_key]
            interval = [float(value) for value in record["null_interval_95"]]
            bound = max(abs(interval[0]), abs(interval[1]))
            observed = float(record["centered_minus_raw_docking"])
            records[label] = {
                "observed_centered_minus_raw_docking_spearman": observed,
                "empirical_central_95_percent_null_interval": interval,
                "empirical_two_sided_95_percent_absolute_resolution_bound": bound,
                "observed_exceeds_absolute_null_bound": abs(observed) > bound,
                "absolute_observed_to_null_bound_ratio": abs(observed) / bound,
            }
        fixed20_qap_resolution["panels"][panel] = records

    source_paths = [
        RESULTS / "evidence_summary.json",
        RESULTS / "fixed20_estimand_decomposition/summary.json",
        RESULTS / "fixed20_estimand_decomposition/decomposition.csv",
        RESULTS / "fixed20_estimand_decomposition/transformation_matched_null.csv",
        RESULTS / "fixed20_reference_sensitivity/summary.json",
        RESULTS / "fixed20_reference_sensitivity/support_concordance.csv",
        RESULTS / "ranking_centering_panel_sensitivity/summary.json",
        RESULTS / "kirhub_external_validation/summary.json",
        RESULTS / "davis_fixed20_censoring_sensitivity/summary.json",
        RESULTS
        / "davis_fixed20_censoring_sensitivity/target_quality_sensitivity.csv",
        RESULTS / "fixed20_centering_panel_sensitivity/summary.json",
        RESULTS / "centering_panel_sensitivity/summary.json",
        RESULTS / "experimental_chemical_context_geometry/summary.json",
        RESULTS / "experimental_panel_overlap/summary.json",
        RESULTS / "spd_external_validation/summary.json",
        RESULTS / "cross_panel_overlap_audit/summary.json",
        RESULTS / "pair_specific_normalization_transfer/summary.json",
        RESULTS / "klifs_pocket_control/partial_geometry_qap.csv",
        RESULTS / "klifs_pocket_control/locked_endpoint_retrieval.csv",
        RESULTS / "klifs_pocket_control/target_pairs.csv",
        RESULTS / "strict_klifs_group_qap/restricted_partial_qap.csv",
        RESULTS / "strict_klifs_group_qap/summary.json",
        RESULTS / "strict_klifs_group_qap/exchangeability_blocks.csv",
        RESULTS / "strict_klifs_group_qap/pair_strata.csv",
        RESULTS / "sequence_docking_fusion/panel_metrics.csv",
        RESULTS / "calibration_panel_recovery/summary.csv",
        RESULTS / "calibration_panel_recovery/scaffold_holdout_summary.csv",
        RESULTS / "missing_residual_sensitivity.csv",
        RESULTS / "correlation_covariance_sensitivity.csv",
        RESULTS / "dense_davis_benchmark.json",
        RESULTS / "dense_pkis2_benchmark.json",
        RESULTS / "residual_mechanism/analysis_summary.json",
        RESULTS / "residual_mechanism/cross_panel_target_mapping.csv",
        RESULTS / "chemical_context_geometry/summary.json",
        RESULTS / "chemical_context_geometry/descriptor_extremes.csv",
        RESULTS
        / "chemical_context_geometry/within_mw_band_reproducibility_controls.csv",
        RESULTS / "chemical_context_geometry/within_mw_band_control_summary.csv",
        RESULTS / "chemical_context_geometry/output_checksums.json",
        RESULTS / "descriptor_component_geometry/summary.json",
        RESULTS / "descriptor_rank_matched_controls/summary.json",
        RESULTS / "descriptor_correlation_reduction_uncertainty/summary.json",
        RESULTS / "target_blind_descriptor_control/summary.json",
        RESULTS / "released_pair_geometry_ledger/summary.json",
        RESULTS / "dockstring_chembl_ranking/summary.json",
        RESULTS / "nonvina_scorer_transport/summary.json",
        RESULTS / "dockstring_vina_terms/analysis_summary.json",
        RESULTS / "dockstring_vina_terms/si_vina_term_summary.csv",
        RESULTS / "equivalence_margin_sensitivity/summary.json",
        *sorted(
            path
            for directory in (
                RESULTS / "descriptor_rank_matched_controls",
                RESULTS / "descriptor_correlation_reduction_uncertainty",
                RESULTS / "target_blind_descriptor_control",
            )
            for path in directory.iterdir()
            if path.is_file() and path.name != "summary.json"
        ),
        *[
            RESULTS / bundle / "output_checksums.json"
            for bundle in (
                "descriptor_component_geometry",
                "fixed20_estimand_decomposition",
                "dockstring_chembl_ranking",
                "nonvina_scorer_transport",
                "equivalence_margin_sensitivity",
                "davis_fixed20_censoring_sensitivity",
                "strict_klifs_group_qap",
                "dockstring_vina_terms",
            )
        ],
    ]

    return {
        "schema_version": "3.1.0",
        "generated_by": "analysis/build_manuscript_evidence.py",
        "python_version": platform.python_version(),
        "ledger": "JoC manuscript evidence",
        "claim_boundary": (
            "Target-pair geometry is a library- and panel-conditional co-response "
            "estimand. It is not ligand-wise target ranking, affinity truth, pose "
            "accuracy, or a docking-quality score."
        ),
        "spectral": spectral,
        "large_matrix_input_audits": large_matrix_input_audits,
        "experimental_measurement_audits": experimental_measurement_audits,
        "physicochemical_interpretation": physicochemical,
        "missing_residual_sensitivity": missing.to_dict(orient="records"),
        "covariance_sensitivity": covariance.to_dict(orient="records"),
        "fixed20_estimand_decomposition": fixed20,
        "fixed20_reference_sensitivity": fixed20_reference,
        "experimental_cross_panel": kirhub[
            "experimental_cross_panel_agreement"
        ],
        "kirhub_locked_endpoint": kirhub["locked_primary_endpoint"],
        "davis_fixed20_censoring_sensitivity": davis_censoring,
        "fixed20_target_label_qap_resolution": fixed20_qap_resolution,
        "kirhub_chemical_overlap_sensitivity": kirhub["chemical_independence_audit"],
        "structural_controls": structural,
        "strict_KLIFS_group_QAP_resolution": strict_klifs,
        "centering_panel_sensitivity": centering_fixed20,
        "panel_specific_centering_sensitivity": centering_panel_specific,
        "chemical_context": chemistry,
        "experimental_panel_overlap": experimental_overlap,
        "probe_panel_recovery": recovery_selected,
        "broad_family_censored_sensitivity": spd,
        "ligand_wise_ranking_boundary": ranking_selected,
        "dense_ligand_wise_benchmarks": dense_ranking,
        "exploratory_chemical_domain_structure": (
            exploratory_chemical_domain_structure
        ),
        "ranking_centering_panel_sensitivity": ranking_centering_panel,
        "pair_specific_transfer_boundary": {
            "primary_transfers": pair_transfer["primary_transfers"],
            "operational_decision": pair_transfer["operational_decision"],
            "decision_reason": pair_transfer["decision_reason"],
        },
        "cross_panel_overlap": overlap,
        "descriptor_mediated_geometry": descriptor_component,
        "descriptor_rank_matched_controls": descriptor_rank_controls,
        "descriptor_correlation_reduction_uncertainty": descriptor_uncertainty,
        "target_indexed_descriptor_control": target_indexed_descriptor_control,
        "released_pair_geometry_ledger": released_ledger,
        "dockstring_chembl_ranking": dockstring_chembl_ranking,
        "nonvina_scorer_transport": nonvina_transport,
        "vina_fixed_pose_term_attribution": vina_terms,
        "ranking_equivalence_margin_sensitivity": equivalence_margin,
        "source_artifacts": {
            str(path.relative_to(PACKAGE)): sha256_file(path) for path in source_paths
        },
    }


def main() -> None:
    OUTPUT.write_text(json.dumps(json_safe(build()), indent=2, allow_nan=False) + "\n")
    print(OUTPUT)


if __name__ == "__main__":
    main()
