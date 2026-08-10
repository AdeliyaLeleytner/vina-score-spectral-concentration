#!/usr/bin/env python3
"""Report layer for the orphaned "Fixed-pose Vina term attribution" Methods subsection.

Reviewer point (Major Revision, cross-lab review).  manuscript.tex describes a 512-ligand
by 9-target term-by-term rescoring of released DOCKSTRING poses, but no Results paragraph
and no supplementary table ever prints its output.  Either report it or delete the
Methods text.  This script reports it.

The analysis itself was already run by analysis/dockstring_vina_term_decomposition.py and
its outputs are frozen under results/dockstring_vina_terms/.  Nothing there is touched.
This script reads four of those artifacts and emits one compact, machine-readable block
suitable for a supplementary table:

  * the score-reproduction validation that licenses the whole exercise (fraction of cells
    reproduced within 0.10 kcal/mol, cell-level Pearson, MAE, worst cell);
  * per Vina term, raw participation ratio, residual (within-ligand-centred)
    participation ratio, raw PC1 fraction, raw mean target correlation, and the term's
    share of the shared row-axis covariance with its scaffold-cluster bootstrap interval;
  * the same spectral columns for the three reference score constructions;
  * the leave-one-term-out arithmetic ablations of the released weighted sum;
  * an explicit assessment of whether these numbers support the claim that spectral
    concentration depends on which energy term dominates.

That last assessment is the point of the script and it is two-sided.  The term shares are
a decomposition of one fixed function, and a cross-scorer check on the same nine targets
contradicts the strong causal reading, so the honest claim is narrower than the Methods
paragraph implies.  See "claim_assessment" and "claim_boundary" in the output.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


PACKAGE = Path(__file__).resolve().parents[1]
RESULTS = PACKAGE / "results"
DEFAULT_TERMS_DIR = RESULTS / "dockstring_vina_terms"
DEFAULT_TRANSPORT = RESULTS / "nonvina_scorer_transport" / "summary.json"
DEFAULT_OUTPUT = RESULTS / "vina_term_attribution_report" / "summary.json"

# Vina 1.1.2 conformation-dependent terms in the fixed order used by the scoring
# function, with the published weights they enter the sum with.
TERM_ORDER = ("gauss1", "gauss2", "repulsion", "hydrophobic", "hydrogen")
TERM_WEIGHTS = {
    "gauss1": -0.035579,
    "gauss2": -0.005156,
    "repulsion": 0.840245,
    "hydrophobic": -0.035069,
    "hydrogen": -0.587439,
}
TERM_SIGN = {
    "gauss1": "attractive",
    "gauss2": "attractive",
    "repulsion": "repulsive",
    "hydrophobic": "attractive",
    "hydrogen": "attractive",
}

REFERENCE_ORDER = ("released_score", "rescored_post_total", "zero_rot_pre_total")
REFERENCE_LABEL = {
    "released_score": "released DOCKSTRING score",
    "rescored_post_total": "rescored total, after torsional normalisation",
    "zero_rot_pre_total": "rescored total, before torsional normalisation",
}

# Columns a supplementary table should print, in order.
TABLE_COLUMNS = (
    "component",
    "raw_pr",
    "residual_pr",
    "raw_pc1_fraction",
    "raw_mean_correlation",
    "shared_row_axis_covariance_share",
)
TABLE_DECIMALS = {
    "raw_pr": 2,
    "residual_pr": 2,
    "raw_pc1_fraction": 3,
    "raw_mean_correlation": 3,
    "shared_row_axis_covariance_share": 3,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def index_by(rows: list[dict[str, str]], key: str) -> dict[str, dict[str, str]]:
    return {row[key]: row for row in rows}


def spectral_columns(row: dict[str, str]) -> dict[str, float]:
    return {
        "raw_pr": float(row["raw_pr"]),
        "residual_pr": float(row["residual_pr"]),
        "pr_increase": float(row["pr_increase"]),
        "raw_pc1_fraction": float(row["raw_pc1_fraction"]),
        "residual_pc1_fraction": float(row["residual_pc1_fraction"]),
        "raw_mean_correlation": float(row["raw_mean_correlation"]),
        "residual_mean_absolute_correlation": float(
            row["residual_mean_absolute_correlation"]
        ),
    }


def display_row(component: str, values: dict[str, Any]) -> dict[str, str]:
    printed = {"component": component}
    for column in TABLE_COLUMNS[1:]:
        value = values.get(column)
        printed[column] = (
            "" if value is None else f"{value:.{TABLE_DECIMALS[column]}f}"
        )
    return printed


def build(
    summary: dict[str, Any],
    spectral_rows: list[dict[str, str]],
    ablation_rows: list[dict[str, str]],
    attribution_rows: list[dict[str, str]],
    transport: dict[str, Any],
    provenance: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    spectral = index_by(spectral_rows, "component")
    attribution = index_by(attribution_rows, "term")
    ablations = index_by(ablation_rows, "removed_term")
    bootstrap = summary["cluster_bootstrap"]
    bootstrap_metrics = bootstrap["metrics"]
    scope = summary["scope"]
    reproduction = summary["score_reproduction"]

    term_records: list[dict[str, Any]] = []
    for term in TERM_ORDER:
        post = spectral_columns(spectral[f"post_{term}"])
        pre = spectral_columns(spectral[f"pre_{term}"])
        share_key = f"{term}_shared_row_axis_share"
        share_ci = bootstrap_metrics[share_key]["ci95"]
        residual_share_ci = bootstrap_metrics[f"{term}_residual_ss_share"]["ci95"]
        term_records.append(
            {
                "term": term,
                "published_weight": TERM_WEIGHTS[term],
                "sign": TERM_SIGN[term],
                "weighted_term_after_torsional_normalisation": post,
                "weighted_term_before_torsional_normalisation": pre,
                "shared_row_axis_covariance_share": float(
                    attribution[term]["shared_row_axis_covariance_share"]
                ),
                "shared_row_axis_covariance_share_ci95": share_ci,
                "raw_pc1_bilinear_share": float(
                    attribution[term]["raw_pc1_bilinear_share"]
                ),
                "residual_frobenius_inner_product_share": float(
                    attribution[term]["residual_frobenius_inner_product_share"]
                ),
                "residual_frobenius_inner_product_share_ci95": residual_share_ci,
                "residual_pc1_bilinear_share": float(
                    attribution[term]["residual_pc1_bilinear_share"]
                ),
                "leave_this_term_out_of_the_sum": spectral_columns(ablations[term]),
            }
        )

    reference_records = [
        {
            "component": name,
            "label": REFERENCE_LABEL[name],
            **spectral_columns(spectral[name]),
        }
        for name in REFERENCE_ORDER
    ]

    table_rows: list[dict[str, str]] = []
    for record in reference_records:
        table_rows.append(display_row(record["label"], record))
    for record in term_records:
        values = dict(record["weighted_term_after_torsional_normalisation"])
        values["shared_row_axis_covariance_share"] = record[
            "shared_row_axis_covariance_share"
        ]
        table_rows.append(display_row(record["term"], values))

    post_raw_pr = {
        record["term"]: record["weighted_term_after_torsional_normalisation"]["raw_pr"]
        for record in term_records
    }
    post_pc1 = {
        record["term"]: record["weighted_term_after_torsional_normalisation"][
            "raw_pc1_fraction"
        ]
        for record in term_records
    }
    shares = {
        record["term"]: record["shared_row_axis_covariance_share"]
        for record in term_records
    }
    total_raw_pr = spectral_columns(spectral["rescored_post_total"])["raw_pr"]
    ablation_raw_pr = {
        term: spectral_columns(ablations[term])["raw_pr"] for term in TERM_ORDER
    }
    ablation_shift = {
        term: ablation_raw_pr[term] - total_raw_pr for term in TERM_ORDER
    }
    most_concentrated = min(post_raw_pr, key=lambda term: post_raw_pr[term])
    least_concentrated = max(post_raw_pr, key=lambda term: post_raw_pr[term])
    largest_share = max(shares, key=lambda term: shares[term])
    largest_ablation_shift = max(ablation_shift, key=lambda term: ablation_shift[term])

    vina = transport["observed"]["released_vina"]
    vinardo = transport["observed"]["vinardo"]

    concentration_spread = {
        "total_released_sum_raw_pr": total_raw_pr,
        "per_term_raw_pr": post_raw_pr,
        "per_term_raw_pc1_fraction": post_pc1,
        "raw_pr_range_across_terms": [
            post_raw_pr[most_concentrated],
            post_raw_pr[least_concentrated],
        ],
        "raw_pr_ratio_across_terms": (
            post_raw_pr[least_concentrated] / post_raw_pr[most_concentrated]
        ),
        "raw_pc1_fraction_range_across_terms": [
            min(post_pc1.values()),
            max(post_pc1.values()),
        ],
        "most_concentrated_term": most_concentrated,
        "least_concentrated_term": least_concentrated,
        "largest_shared_row_axis_share_term": largest_share,
        "shared_row_axis_covariance_shares": shares,
        "shared_row_axis_covariance_share_sum": sum(shares.values()),
        "leave_one_term_out_raw_pr": ablation_raw_pr,
        "leave_one_term_out_raw_pr_shift": ablation_shift,
        "largest_leave_one_out_raw_pr_shift_term": largest_ablation_shift,
        "largest_leave_one_out_raw_pr_shift": ablation_shift[largest_ablation_shift],
    }

    supports_descriptive = (
        concentration_spread["raw_pr_ratio_across_terms"] > 2.0
        and shares[largest_share] > 0.5
        and ablation_shift[largest_ablation_shift] > 1.0
    )
    refit_scorer_contradiction = (
        abs(vinardo["raw_participation_ratio"] - vina["raw_participation_ratio"])
        < ablation_shift[largest_ablation_shift]
    )

    return {
        "analysis_question": (
            "Does the frozen fixed-pose term-by-term rescoring of released DOCKSTRING "
            "poses support the claim that the spectral concentration of a Vina docking "
            "panel depends on which energy term dominates the score, and what exactly "
            "should a supplementary table print for it?"
        ),
        "status": "orphaned_methods_section_reported",
        "purpose": (
            "report layer only: it reads the frozen artifacts of "
            "analysis/dockstring_vina_term_decomposition.py, re-runs no docking, no "
            "rescoring and no resampling, and writes nothing under "
            "results/dockstring_vina_terms/"
        ),
        "support": {
            "n_ligands": scope["n_ligands"],
            "n_targets": scope["n_targets"],
            "n_cells": scope["n_cells"],
            "targets": list(scope["targets"]),
            "ligand_sample_rule": scope["sample_rule"],
            "pose_source": (
                "published DOCKSTRING top poses, Figshare DOI "
                f"{scope['figshare_doi']}"
            ),
            "ligand_conversion": scope["ligand_conversion"],
            "bootstrap_resampling_unit": bootstrap["resampling_unit"],
            "bootstrap_clusters": bootstrap["n_clusters"],
            "bootstrap_replicates": bootstrap["replicates"],
        },
        "seeds": {
            "primary_dockstring_support": 71,
            "term_subset_sample": 20260802,
            "scaffold_cluster_bootstrap": bootstrap["seed"],
            "torsional_factor_permutation": summary["torsional_normalisation"][
                "factor_permutation_null"
            ]["seed"],
            "this_script": (
                "none; the script is a deterministic read-and-tabulate over frozen "
                "artifacts and draws no random numbers"
            ),
        },
        "uncertainty_contract_deviation": {
            "package_contract": (
                "5,000 replicates and a conservative union of Murcko and Butina "
                "chemical-cluster intervals"
            ),
            "what_this_analysis_used": (
                f"{bootstrap['replicates']} replicates over "
                f"{bootstrap['n_clusters']} Bemis-Murcko scaffold clusters only; no "
                "Butina scheme and no conservative union were computed, and only 95% "
                "intervals were stored"
            ),
            "consequence": (
                "the term-share intervals below are narrower than a package-contract "
                "interval would be and must be reported as exploratory; the "
                "supplementary table should say so in its caption"
            ),
        },
        "score_reproduction": {
            "cells_reproduced_within_0.10_kcal_mol": reproduction[
                "cells_reproduced_within_0.10_kcal_mol"
            ],
            "n_cells": scope["n_cells"],
            "fraction_cells_reproduced_within_0.10_kcal_mol": reproduction[
                "fraction_cells_reproduced_within_0.10_kcal_mol"
            ],
            "ligands_reproduced_within_0.10_kcal_mol_on_all_targets": reproduction[
                "ligands_reproduced_within_0.10_kcal_mol_on_all_targets"
            ],
            "rescored_vs_released_cell_pearson": reproduction[
                "rescored_vs_released_cell_pearson"
            ],
            "rescored_minus_released_mae": reproduction["rescored_minus_released_mae"],
            "rescored_minus_released_mean": reproduction[
                "rescored_minus_released_mean"
            ],
            "rescored_minus_released_max_abs": reproduction[
                "rescored_minus_released_max_abs"
            ],
            "term_sum_max_abs_reconstruction_error": reproduction[
                "term_sum_max_abs_reconstruction_error"
            ],
            "quality_support_repeat": {
                "rule": summary["quality_support_sensitivity"]["score_roundtrip_rule"],
                "n_ligands": summary["quality_support_sensitivity"]["supports"][
                    "released_scores_reproduced_within_0.10_kcal_mol"
                ]["n_ligands"],
                "excluded_ligands": summary["quality_support_sensitivity"]["supports"][
                    "released_scores_reproduced_within_0.10_kcal_mol"
                ]["excluded_ligands"],
            },
        },
        "reference_constructions": reference_records,
        "terms": term_records,
        "supplementary_table": {
            "caption_facts": (
                f"Fixed-pose Vina term attribution on {scope['n_ligands']} DOCKSTRING "
                f"ligands x {scope['n_targets']} targets ({scope['n_cells']} cells). "
                "Rows are the three reference score constructions and the five weighted "
                "Vina terms after torsional normalisation. Shared row-axis covariance "
                "shares are exploratory and carry scaffold-cluster bootstrap intervals "
                f"from {bootstrap['replicates']} replicates only."
            ),
            "columns": list(TABLE_COLUMNS),
            "column_headers": [
                "component",
                "raw PR",
                "residual PR",
                "raw PC1 fraction",
                "raw mean correlation",
                "share of shared row-axis covariance",
            ],
            "rows": table_rows,
        },
        "concentration_spread": concentration_spread,
        "refit_scorer_cross_check": {
            "source": "results/nonvina_scorer_transport/summary.json",
            "why": (
                "the leave-one-term-out numbers drop a term from a fixed weighted sum "
                "without refitting the remaining weights; Vinardo is a scoring function "
                "that drops Vina's gauss2 and reweights repulsion with refitted weights, "
                "so it tests whether the arithmetic ablation generalises"
            ),
            "support_note": (
                "the transport block uses "
                f"{transport['support']['n_ligands']} ligands on the same "
                f"{transport['support']['n_targets']} targets and the same released "
                "poses, a different and larger ligand subset than the "
                f"{scope['n_ligands']}-ligand term block, so the comparison is between "
                "supports and not within one"
            ),
            "released_vina_raw_pr": vina["raw_participation_ratio"],
            "vinardo_raw_pr": vinardo["raw_participation_ratio"],
            "released_vina_raw_pc1_fraction": vina["raw_pc1_fraction"],
            "vinardo_raw_pc1_fraction": vinardo["raw_pc1_fraction"],
            "vinardo_minus_vina_raw_pr": (
                vinardo["raw_participation_ratio"] - vina["raw_participation_ratio"]
            ),
            "arithmetic_gauss2_ablation_raw_pr_shift": ablation_shift["gauss2"],
        },
        "claim_assessment": {
            "claim_under_test": (
                "spectral concentration of a docking panel depends on which energy term "
                "dominates the score"
            ),
            "descriptive_reading_supported": supports_descriptive,
            "descriptive_reading": (
                "supported as a decomposition of the released Vina sum: individual "
                f"weighted terms span raw PR {post_raw_pr[most_concentrated]:.2f} "
                f"({most_concentrated}) to {post_raw_pr[least_concentrated]:.2f} "
                f"({least_concentrated}) and raw PC1 fraction "
                f"{min(post_pc1.values()):.3f} to {max(post_pc1.values()):.3f} on the "
                "same poses and the same nine targets, so concentration is a property "
                "of the term and not of the panel alone. One attractive term, "
                f"{largest_share}, carries "
                f"{shares[largest_share]:.3f} of the shared row-axis covariance, and "
                f"removing it from the sum moves raw PR by "
                f"{ablation_shift['gauss2']:+.2f} from {total_raw_pr:.2f}."
            ),
            "causal_reading_supported": not refit_scorer_contradiction,
            "causal_reading": (
                "not supported. The ablations remove a term from a fixed weighted sum "
                "without refitting the remaining weights, and the poses are the "
                "published Vina poses rather than re-docked ones, so nothing here is a "
                "scoring-function redesign. The direct test contradicts the strong "
                f"reading: Vinardo, which drops gauss2 and reweights repulsion with "
                f"refitted weights, gives raw PR {vinardo['raw_participation_ratio']:.2f} "
                f"against Vina's {vina['raw_participation_ratio']:.2f} on the same "
                "targets and poses, a shift far smaller than the "
                f"{ablation_shift['gauss2']:+.2f} the arithmetic gauss2 ablation "
                "predicts. A function that removes the dominant term and refits does "
                "not decongest."
            ),
            "recommended_manuscript_wording": (
                "The term attribution should be reported as an exploratory attribution "
                "of the released Vina sum: which terms carry the shared ligand-wide "
                "axis (gauss2 dominant, repulsion opposing), not as evidence that "
                "changing the dominant term would change a panel's dimensionality. If "
                "the manuscript is unwilling to carry that qualification, the Methods "
                "subsection should be deleted rather than promoted."
            ),
        },
        "provenance": provenance,
        "claim_boundary": (
            "This is an attribution of retained poses, not a causal re-docking ablation: "
            "the original docked torsion trees were never published, poses are held "
            "fixed at the released Vina optimum, and every term matrix is an arithmetic "
            "piece of one already-computed score. It therefore does not show that a "
            "scoring function built around a different dominant term would produce a "
            "differently concentrated panel; the Vinardo cross-check argues the "
            "opposite. It is a nine-target, 512-ligand block of the DOCKSTRING support "
            "with no experimental endpoint attached, so it says nothing about binding "
            "accuracy, target selectivity or ranking performance, and the term-share "
            "intervals come from a single scaffold-cluster bootstrap that is weaker than "
            "the package's 5,000-replicate two-scheme contract."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--terms-dir", type=Path, default=DEFAULT_TERMS_DIR)
    parser.add_argument("--transport", type=Path, default=DEFAULT_TRANSPORT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    summary_path = args.terms_dir / "analysis_summary.json"
    spectral_path = args.terms_dir / "term_spectral_metrics.csv"
    ablation_path = args.terms_dir / "fixed_pose_term_ablations.csv"
    attribution_path = args.terms_dir / "term_attribution.csv"

    provenance = {
        "results/dockstring_vina_terms/analysis_summary.json": {
            "bytes": summary_path.stat().st_size,
            "sha256": sha256_file(summary_path),
        },
        "results/dockstring_vina_terms/term_spectral_metrics.csv": {
            "bytes": spectral_path.stat().st_size,
            "sha256": sha256_file(spectral_path),
        },
        "results/dockstring_vina_terms/fixed_pose_term_ablations.csv": {
            "bytes": ablation_path.stat().st_size,
            "sha256": sha256_file(ablation_path),
        },
        "results/dockstring_vina_terms/term_attribution.csv": {
            "bytes": attribution_path.stat().st_size,
            "sha256": sha256_file(attribution_path),
        },
        "results/nonvina_scorer_transport/summary.json": {
            "bytes": args.transport.stat().st_size,
            "sha256": sha256_file(args.transport),
        },
    }

    result = build(
        json.loads(summary_path.read_text()),
        read_csv_rows(spectral_path),
        read_csv_rows(ablation_path),
        read_csv_rows(attribution_path),
        json.loads(args.transport.read_text()),
        provenance,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
