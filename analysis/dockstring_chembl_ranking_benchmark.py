#!/usr/bin/env python3
"""Build the broad DOCKSTRING--ChEMBL observed-pair ranking benchmark.

The earlier sparse Docking-44 sensitivity benchmark
(``build_evidence.expanded_target_preference_benchmark``, frozen in
``results/evidence_summary.json`` under ``expanded_target_preference_benchmark``)
matches ChEMBL activities against the 12,651-ligand Docking-44 matrix and
retains 137 ligands, 38 targets, 691 observed cells and 2,522 non-tied
within-ligand target pairs.  The paper's *dense* exact-match checks (DAVIS,
PKIS2) instead use the 260,060-ligand DOCKSTRING matrix, so the headline sparse
benchmark runs on a score matrix roughly twenty times smaller than the one used
elsewhere in the same manuscript.

This program evaluates the identical estimand on the substantially broader
DOCKSTRING-matched chemical support.  It changes exactly
two things relative to the published benchmark:

1. the docking reference is the complete 260,060-row DOCKSTRING support with
   its 58 gene-symbol targets instead of the 12,651-row Docking-44 support with
   its 44 PDB-keyed targets; and
2. the ChEMBL pull is keyed on the 31 human SINGLE PROTEIN targets that map to
   DOCKSTRING gene symbols instead of on Docking-44 ``panel_key`` values.

Everything else is held to the published contract and, wherever possible, is
executed by the published functions themselves: the RDKit full-InChIKey
matching rule, exact standard relation, the Ki/Kd/IC50/EC50 endpoint set, the
median-per-cell aggregation, the at-least-two-observed-targets cohort filter,
the externally fitted score representations
(``build_evidence.external_reference_score_representations``), the
leave-one-Bemis-Murcko-cluster-out cohort prior, and the metric, bootstrap and
permutation machinery of ``build_evidence.expanded_preference_metrics``.

The ChEMBL snapshot is the potency-floor-free pull written by
``analysis/fetch_chembl_dockstring_ranking_activities.py``.  It is *not* the
chemistry-graph snapshot ``chembl_sea_activities_2026-08-03.csv.gz``, which is
left-truncated at ``pchembl_value >= 6`` and restricted to ``assay_type=B`` and
would therefore have estimated ordering among high-potency binding cells only.
The truncated contract is still reported, as a declared sensitivity, so the two
estimands can be compared directly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:  # Support direct CLI execution and package-style imports.
    from . import build_evidence as evidence
    from . import dense_davis_benchmark as dense_davis
    from . import fetch_chembl_dockstring_ranking_activities as ranking_fetch
except ImportError:  # pragma: no cover - direct CLI execution.
    import build_evidence as evidence  # type: ignore
    import dense_davis_benchmark as dense_davis  # type: ignore
    import fetch_chembl_dockstring_ranking_activities as ranking_fetch  # type: ignore


PACKAGE = Path(__file__).resolve().parents[1]
FROZEN = PACKAGE / "data" / "frozen"
DEFAULT_OUTPUT = PACKAGE / "results" / "dockstring_chembl_ranking"
DEFAULT_SEED = 20260805

DOCKSTRING_INPUT = FROZEN / "dockstring-dataset.tsv.gz"
IDENTITY_INPUT = FROZEN / "dockstring_identity_contract_2026-08-03.csv.gz"
ACTIVITY_INPUT = FROZEN / "chembl_dockstring_ranking_activities_2026-08-05.csv.gz"
PUBLISHED_EVIDENCE = PACKAGE / "results" / "evidence_summary.json"

EXPECTED_DOCKSTRING_ROWS = 260_155
EXPECTED_COMPLETE_ROWS = 260_060
ENDPOINTS = ("Ki", "Kd", "IC50", "EC50")

REPRESENTATION_ORDER = (
    "absolute_vina",
    "column_standardized",
    "two_way_residual",
    "target_centered_unscaled",
    "two_way_centered_unscaled",
    "target_centered_residual_scaled",
    "docking_target_prior",
    "experimental_target_prior",
    "cohort_experimental_target_prior",
)
HEADLINE_CONTRASTS = (
    "two_way_residual_minus_absolute_vina",
    "two_way_residual_minus_column_standardized",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def gene_target_contract() -> tuple[dict[str, str], dict[str, str]]:
    """Return the ChEMBL-id -> gene-symbol map used by the frozen activity pull."""
    gene_to_chembl = {
        gene: str(record["chembl"]) for gene, record in ranking_fetch.TARGETS.items()
    }
    chembl_to_gene: dict[str, str] = {}
    for gene, chembl_id in gene_to_chembl.items():
        if chembl_id in chembl_to_gene:
            raise ValueError(
                f"ChEMBL target {chembl_id} maps to more than one DOCKSTRING gene symbol"
            )
        chembl_to_gene[chembl_id] = gene
    return gene_to_chembl, chembl_to_gene


def load_docking_reference() -> tuple[pd.DataFrame, pd.Series, dict[str, Any]]:
    """Load the complete DOCKSTRING support keyed by frozen full InChIKey.

    The InChIKeys come from the frozen dual-identity contract
    (``raw_inchikey``), which is the current-RDKit standard key computed from
    the released DOCKSTRING SMILES.  DOCKSTRING's own ``inchikey`` column uses
    a non-standard third block (``...NA-N``) and matches no ChEMBL key at all,
    so it must not be used for the exact-match rule.
    """
    dockstring_all = pd.read_csv(DOCKSTRING_INPUT, sep="\t")
    if len(dockstring_all) != EXPECTED_DOCKSTRING_ROWS:
        raise ValueError(
            f"expected the {EXPECTED_DOCKSTRING_ROWS:,}-row DOCKSTRING release, "
            f"observed {len(dockstring_all):,}"
        )
    score_columns = [
        column
        for column in dockstring_all.columns
        if column not in {"inchikey", "smiles"}
    ]
    complete = ~dockstring_all[score_columns].isna().any(axis=1)
    dockstring = dockstring_all.loc[complete].copy()
    if len(dockstring) != EXPECTED_COMPLETE_ROWS:
        raise ValueError(
            f"expected the complete {EXPECTED_COMPLETE_ROWS:,}-row DOCKSTRING support, "
            f"observed {len(dockstring):,}"
        )

    identity = pd.read_csv(
        IDENTITY_INPUT, usecols=["source_row_index", "raw_inchikey"]
    )
    if len(identity) != EXPECTED_COMPLETE_ROWS:
        raise ValueError("the frozen DOCKSTRING identity contract changed length")
    if not np.array_equal(
        identity.source_row_index.to_numpy(dtype=int),
        dockstring.index.to_numpy(dtype=int),
    ):
        raise ValueError(
            "the frozen identity contract is not aligned with the complete DOCKSTRING rows"
        )
    if identity.raw_inchikey.isna().any():
        raise ValueError("the frozen identity contract contains a null full InChIKey")

    dockstring = dockstring.reset_index(drop=True)
    dockstring["inchikey"] = identity.raw_inchikey.to_numpy()
    # Vina energies carry pathological positive outliers; the package clips at 0
    # before every analysis.
    clipped = dockstring[score_columns].clip(upper=0.0)
    clipped.insert(0, "inchikey", dockstring.inchikey.to_numpy())

    duplicate_keys = int(dockstring.inchikey.duplicated().sum())
    reference = clipped.groupby("inchikey", sort=True).median()
    smiles = (
        dockstring.groupby("inchikey", sort=True).smiles.first().reindex(reference.index)
    )
    if not reference.index.is_unique:
        raise ValueError("DOCKSTRING full InChIKeys are not unique after aggregation")
    if reference.isna().to_numpy().any():
        raise ValueError("the aggregated DOCKSTRING reference contains missing scores")
    reference.index.name = "inchikey"

    report = {
        "released_rows": int(len(dockstring_all)),
        "complete_rows": int(EXPECTED_COMPLETE_ROWS),
        "unique_full_inchikeys": int(len(reference)),
        "rows_sharing_a_full_inchikey_with_an_earlier_row": duplicate_keys,
        "duplicate_resolution": (
            "median over every complete DOCKSTRING row sharing a full InChIKey, matching "
            "the median-per-cell rule the published benchmark applies to repeated ChEMBL "
            "measurements"
        ),
        "score_clipping": "scores clipped at an upper bound of 0.0 kcal/mol",
        "identity_rule": (
            "RDKit standard full InChIKey from the released DOCKSTRING SMILES "
            "(frozen identity contract column raw_inchikey); the released DOCKSTRING "
            "inchikey column uses a non-standard third block and cannot be matched"
        ),
        "targets": int(reference.shape[1]),
    }
    return reference, smiles, report


def load_activities(
    chembl_to_gene: dict[str, str],
    *,
    activity_path: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply the published activity contract to the floor-free ChEMBL snapshot."""
    columns = [
        "activity_id",
        "target_chembl_id",
        "canonical_smiles",
        "pchembl_value",
        "standard_type",
        "standard_relation",
        "assay_type",
        "target_organism",
    ]
    raw = pd.read_csv(activity_path, usecols=columns)
    source_records = int(len(raw))
    activities = raw[
        raw.pchembl_value.notna()
        & raw.standard_relation.eq("=")
        & raw.standard_type.isin(list(ENDPOINTS))
    ].copy()
    unexpected = set(activities.target_chembl_id.unique()) - set(chembl_to_gene)
    if unexpected:
        raise ValueError(f"activity snapshot contains unmapped targets: {sorted(unexpected)}")
    activities["gene"] = activities.target_chembl_id.map(chembl_to_gene)
    activities["inchikey"] = evidence.full_inchikeys(activities.canonical_smiles)
    unresolved = int(activities.inchikey.isna().sum())
    activities = activities[activities.inchikey.notna()].copy()

    report = {
        "source_records": source_records,
        "records_after_exact_relation_and_endpoint_filter": int(len(activities) + unresolved),
        "records_with_unparseable_smiles": unresolved,
        "analysis_records": int(len(activities)),
        "unique_ligands": int(activities.inchikey.nunique()),
        "targets_with_any_record": int(activities.gene.nunique()),
        "endpoint_counts": {
            str(key): int(value)
            for key, value in activities.standard_type.value_counts().items()
        },
        "minimum_pchembl": float(activities.pchembl_value.min()),
        "maximum_pchembl": float(activities.pchembl_value.max()),
        "fraction_below_pchembl_6": float((activities.pchembl_value < 6.0).mean()),
        "activity_contract": (
            "non-null pChEMBL; exact standard relation; Ki, Kd, IC50 or EC50; no potency "
            "floor and no assay-type restriction, matching the published Docking-44 pull"
        ),
    }
    return activities, report


def build_support(
    activities: pd.DataFrame,
    docking_reference: pd.DataFrame,
    target_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Pivot to median cells and apply the published cohort filter."""
    experimental_reference = activities.pivot_table(
        index="inchikey",
        columns="gene",
        values="pchembl_value",
        aggfunc="median",
    ).reindex(columns=target_columns)
    matched_ids = experimental_reference.index.intersection(docking_reference.index)
    matched = experimental_reference.loc[matched_ids]
    coverage = matched.notna().sum(axis=1)
    evaluation_ids = matched.index[coverage >= 2]
    experiment = matched.loc[evaluation_ids]
    evaluation_targets = experiment.columns[experiment.notna().any(axis=0)].tolist()
    experiment = experiment[evaluation_targets]
    if experiment.shape[0] < 2 or experiment.shape[1] < 2:
        raise ValueError("the DOCKSTRING x ChEMBL support is degenerate")

    dropped = [
        str(target) for target in target_columns if target not in evaluation_targets
    ]
    report = {
        "chembl_ligands_with_any_analysis_record": int(len(experimental_reference)),
        "chembl_ligands_matched_to_dockstring": int(len(matched)),
        "matched_observed_cells": int(matched.notna().sum().sum()),
        "ligands_by_minimum_observed_targets": {
            str(minimum): int((coverage >= minimum).sum())
            for minimum in [1, 2, 3, 4, 5, 6, 8, 10]
        },
        "targets_dropped_for_no_observation_in_cohort": dropped,
    }
    return experimental_reference, experiment, report


def kikd_endpoint_mixing_diagnostic(
    activities: pd.DataFrame,
    docking_reference: pd.DataFrame,
    target_columns: list[str],
) -> dict[str, Any]:
    """Describe endpoint provenance of the human binding Ki/Kd arm's pairs.

    The benchmark aggregates every ligand--target cell by the median over all
    eligible records, irrespective of whether the cell contains Ki, Kd, or
    both.  A cell containing both endpoint types therefore has no unique
    endpoint label.  We retain such cells in the unchanged benchmark estimand,
    but classify every informative comparison involving one as
    ``involves_pooled_Ki_Kd_cell`` rather than forcing it into the same- or
    mixed-endpoint categories.
    """
    observed_types = set(activities.standard_type.dropna().astype(str).unique())
    if not observed_types <= {"Ki", "Kd"}:
        raise ValueError(
            "the Ki/Kd mixing diagnostic received endpoint types outside Ki and Kd: "
            f"{sorted(observed_types - {'Ki', 'Kd'})}"
        )

    _, experiment, _ = build_support(
        activities, docking_reference, target_columns
    )
    cell_types = (
        activities.groupby(["inchikey", "gene"], sort=True)["standard_type"]
        .agg(lambda values: tuple(sorted(set(map(str, values)))))
        .unstack("gene")
        .reindex(index=experiment.index, columns=experiment.columns)
    )

    cell_counts = {"Ki_only": 0, "Kd_only": 0, "pooled_Ki_and_Kd": 0}
    for endpoint_set in cell_types.to_numpy(dtype=object).ravel():
        if not isinstance(endpoint_set, tuple):
            continue
        if endpoint_set == ("Ki",):
            cell_counts["Ki_only"] += 1
        elif endpoint_set == ("Kd",):
            cell_counts["Kd_only"] += 1
        elif endpoint_set == ("Kd", "Ki"):
            cell_counts["pooled_Ki_and_Kd"] += 1
        else:  # Defensive: every observed cell must satisfy the declared contract.
            raise ValueError(f"unexpected endpoint set in Ki/Kd cell: {endpoint_set}")

    pair_counts = {
        "same_endpoint": 0,
        "same_endpoint_Ki_Ki": 0,
        "same_endpoint_Kd_Kd": 0,
        "mixed_endpoint_Ki_Kd": 0,
        "involves_pooled_Ki_Kd_cell": 0,
    }
    excluded_exact_ties = 0
    values = experiment.to_numpy(dtype=float)
    types = cell_types.to_numpy(dtype=object)
    for ligand in range(len(experiment)):
        observed = np.flatnonzero(np.isfinite(values[ligand]))
        for first in range(len(observed)):
            for second in range(first + 1, len(observed)):
                i, j = observed[first], observed[second]
                if values[ligand, i] == values[ligand, j]:
                    excluded_exact_ties += 1
                    continue
                first_types = types[ligand, i]
                second_types = types[ligand, j]
                if not isinstance(first_types, tuple) or not isinstance(
                    second_types, tuple
                ):
                    raise ValueError(
                        "an observed Ki/Kd benchmark cell lacks endpoint provenance"
                    )
                if len(first_types) > 1 or len(second_types) > 1:
                    pair_counts["involves_pooled_Ki_Kd_cell"] += 1
                elif first_types == second_types == ("Ki",):
                    pair_counts["same_endpoint"] += 1
                    pair_counts["same_endpoint_Ki_Ki"] += 1
                elif first_types == second_types == ("Kd",):
                    pair_counts["same_endpoint"] += 1
                    pair_counts["same_endpoint_Kd_Kd"] += 1
                else:
                    pair_counts["mixed_endpoint_Ki_Kd"] += 1

    total = (
        pair_counts["same_endpoint"]
        + pair_counts["mixed_endpoint_Ki_Kd"]
        + pair_counts["involves_pooled_Ki_Kd_cell"]
    )
    classifiable = (
        pair_counts["same_endpoint"] + pair_counts["mixed_endpoint_Ki_Kd"]
    )
    if total <= 0 or classifiable <= 0:
        raise ValueError("the Ki/Kd endpoint-mixing diagnostic has no usable pairs")

    return {
        "informative_non_tied_pairs": int(total),
        "excluded_exact_median_ties": int(excluded_exact_ties),
        "endpoint_unambiguous_pairs": int(classifiable),
        "pair_counts": {key: int(value) for key, value in pair_counts.items()},
        "fractions_of_all_informative_pairs": {
            key: float(value / total)
            for key, value in pair_counts.items()
            if key
            in {
                "same_endpoint",
                "mixed_endpoint_Ki_Kd",
                "involves_pooled_Ki_Kd_cell",
            }
        },
        "fractions_among_endpoint_unambiguous_pairs": {
            "same_endpoint": float(pair_counts["same_endpoint"] / classifiable),
            "mixed_endpoint_Ki_Kd": float(
                pair_counts["mixed_endpoint_Ki_Kd"] / classifiable
            ),
        },
        "observed_cell_endpoint_provenance": {
            key: int(value) for key, value in cell_counts.items()
        },
        "cell_aggregation_contract": (
            "Each ligand-target cell is the median pChEMBL over all eligible human "
            "binding Ki and Kd records, matching the unchanged benchmark estimand. A "
            "cell containing records of both types remains pooled and is not assigned "
            "an artificial single endpoint label."
        ),
        "pair_classification_contract": (
            "Unordered within-ligand target pairs use the same observed cells and exact "
            "median-tie exclusion as the ranking benchmark. Ki-only/Ki-only and "
            "Kd-only/Kd-only pairs are same-endpoint; Ki-only/Kd-only pairs are mixed; "
            "pairs involving at least one pooled Ki-and-Kd cell are reported separately."
        ),
    }


def cohort_scaffold_prior(
    experiment: pd.DataFrame,
    cohort_scaffolds: np.ndarray,
    global_prior: np.ndarray,
) -> np.ndarray:
    """Leave-one-Bemis-Murcko-cluster-out cohort target prior (published rule)."""
    experiment_values = experiment.to_numpy(dtype=float)
    cohort_prior = np.empty_like(experiment_values)
    for held_out in range(len(experiment_values)):
        training = cohort_scaffolds != cohort_scaffolds[held_out]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            mean = np.nanmean(experiment_values[training], axis=0)
        cohort_prior[held_out] = np.where(np.isfinite(mean), -mean, global_prior)
    return cohort_prior


def score_representations(
    docking_reference: pd.DataFrame,
    experimental_reference: pd.DataFrame,
    experiment: pd.DataFrame,
    cohort_scaffolds: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Externally fitted representations plus the cohort prior, published contract."""
    evaluation_targets = experiment.columns.tolist()
    score_matrices, fit_support = evidence.external_reference_score_representations(
        docking_reference,
        experiment.index,
        evaluation_targets,
        experimental_reference,
    )
    score_matrices["cohort_experimental_target_prior"] = cohort_scaffold_prior(
        experiment,
        cohort_scaffolds,
        score_matrices["experimental_target_prior"][0],
    )
    # The shared helper hard-codes the Docking-44 row-effect wording.
    fit_support["row_effect_for_evaluated_ligand"] = (
        f"mean over all {docking_reference.shape[1]} DOCKSTRING target scores"
    )
    fit_support["cohort_experimental_target_prior"] = (
        "leave-one-Bemis-Murcko-scaffold-cluster-out target mean within the evaluated "
        "cohort; targets with no remaining value fall back to the external ChEMBL prior"
    )
    return score_matrices, fit_support


def conservative_union(
    comparison: dict[str, Any], level: str
) -> tuple[list[float], list[str]]:
    """Union of the Murcko- and Butina-cluster bootstrap intervals at one level."""
    sources = [
        name
        for name in ("scaffold_cluster_bootstrap", "butina_cluster_bootstrap")
        if name in comparison
    ]
    if not sources:
        raise ValueError("no chemical-cluster bootstrap present in the comparison")
    intervals = [comparison[name][level] for name in sources]
    return (
        [
            float(min(interval[0] for interval in intervals)),
            float(max(interval[1] for interval in intervals)),
        ],
        sources,
    )


def contrast_equivalence_report(comparison: dict[str, Any]) -> dict[str, Any]:
    """Two-level conservative intervals and the smallest supported margin."""
    union_90, sources = conservative_union(comparison, "interval_90")
    union_95, _ = conservative_union(comparison, "interval_95")
    smallest_margin_90 = float(max(abs(union_90[0]), abs(union_90[1])))
    smallest_margin_95 = float(max(abs(union_95[0]), abs(union_95[1])))
    return {
        "plugin_mean_difference": float(comparison["plugin_mean_difference"]),
        "bootstrap_sources": sources,
        "conservative_cluster_bootstrap_interval_90": union_90,
        "conservative_cluster_bootstrap_interval_95": union_95,
        "conservative_interval_width_90": float(union_90[1] - union_90[0]),
        "conservative_interval_width_95": float(union_95[1] - union_95[0]),
        "smallest_supported_symmetric_equivalence_margin_90": smallest_margin_90,
        "smallest_supported_symmetric_equivalence_margin_95": smallest_margin_95,
        "equivalence_reading": (
            "the 90% conservative interval supports a two-one-sided-test equivalence "
            "claim at alpha = 0.05 for any symmetric margin strictly larger than "
            f"{smallest_margin_90:.4f} accuracy units"
        ),
        "margin_decisions_90": {
            f"{margin:.2f}": bool(union_90[0] > -margin and union_90[1] < margin)
            for margin in (0.01, 0.02, 0.05, 0.10)
        },
        "excludes_zero_95": bool(union_95[0] > 0.0 or union_95[1] < 0.0),
        "excludes_zero_90": bool(union_90[0] > 0.0 or union_90[1] < 0.0),
        "status": (
            "post hoc equivalence sensitivity; the margins were declared during revision "
            "rather than prospectively registered"
        ),
    }


def uncertainty_layers(benchmark: dict[str, Any]) -> dict[str, Any]:
    """Keep sampling, target-composition and null-model questions distinct.

    Chemical-support intervals answer how the paired score contrast varies when
    ligands or chemical clusters are resampled.  Delete-one-target jackknives are
    instead a fixed-panel composition sensitivity.  Neither object is a target-label
    QAP or a transformation-matched mechanical null, which belong to the target-network
    analyses elsewhere in the package.
    """
    sampling: dict[str, Any] = {}
    target_composition: dict[str, Any] = {}
    target_count = int(benchmark["n_targets"])
    for name in HEADLINE_CONTRASTS:
        comparison = benchmark["paired_comparisons"][name]
        union_90, sources = conservative_union(comparison, "interval_90")
        union_95, _ = conservative_union(comparison, "interval_95")
        sampling[name] = {
            "estimand": "paired mean per-ligand observed-pair concordance difference",
            "chemical_cluster_schemes": sources,
            "conservative_union_interval_90": union_90,
            "conservative_union_interval_95": union_95,
            "interpretation": (
                "sampling uncertainty over the evaluated chemical support with target "
                f"labels and the {target_count}-target panel held fixed"
            ),
        }
        jackknife = benchmark["target_jackknife_paired_contrasts"][name]
        target_composition[name] = {
            "full_panel_paired_difference": jackknife[
                "full_panel_paired_difference"
            ],
            "leave_one_target_out_minimum": jackknife["minimum"],
            "leave_one_target_out_maximum": jackknife["maximum"],
            "jackknife_normal_95_interval": jackknife[
                "jackknife_normal_95_interval"
            ],
            "jackknife_standard_error": jackknife["jackknife_standard_error"],
            "most_influential_target": jackknife["most_influential_target"],
            "interpretation": jackknife["scope"],
        }
    return {
        "chemical_support_sampling": sampling,
        "target_panel_composition": target_composition,
        "target_label_qap": {
            "status": "not_run_for_this_ligand_wise_observed_pair_estimand",
            "interpretation": (
                "a target-label QAP addresses labelled target-network alignment and is "
                "not a confidence interval for ligand-wise ranking accuracy"
            ),
        },
        "transformation_matched_mechanical_null": {
            "status": "reported_in_the_separate_fixed20_geometry_analysis",
            "interpretation": (
                "the mechanical centring null addresses target-network geometry, not "
                "finite-chemical-support uncertainty in this ranking benchmark"
            ),
        },
    }


def ranking_order_summary(benchmark: dict[str, Any]) -> dict[str, Any]:
    """Name winners without conflating docking representations with fitted priors."""
    values = {
        name: float(record["mean_per_ligand_pairwise_accuracy"])
        for name, record in benchmark["representations"].items()
    }
    docking_names = (
        "absolute_vina",
        "column_standardized",
        "two_way_residual",
        "target_centered_unscaled",
        "two_way_centered_unscaled",
        "target_centered_residual_scaled",
    )
    prior_names = (
        "docking_target_prior",
        "experimental_target_prior",
        "cohort_experimental_target_prior",
    )
    best_docking = max(docking_names, key=values.__getitem__)
    best_prior = max(prior_names, key=values.__getitem__)
    best_overall = max(values, key=values.__getitem__)
    return {
        "best_tested_docking_score_representation": best_docking,
        "best_tested_docking_score_representation_accuracy": values[best_docking],
        "best_prior": best_prior,
        "best_prior_accuracy": values[best_prior],
        "best_overall_representation": best_overall,
        "best_overall_accuracy": values[best_overall],
        "absolute_vina_exceeds_cohort_experimental_target_prior": bool(
            values["absolute_vina"]
            > values["cohort_experimental_target_prior"]
        ),
        "interpretation": (
            "method ordering must be reported separately for docking-derived score "
            "representations and experimental/cohort target priors"
        ),
    }


def published_reference() -> dict[str, Any] | None:
    """Frozen Docking-44 numbers for a like-for-like comparison, if available."""
    if not PUBLISHED_EVIDENCE.exists():
        return None
    payload = json.loads(PUBLISHED_EVIDENCE.read_text())
    primary = payload["expanded_target_preference_benchmark"]["primary_all_exact"]
    reference: dict[str, Any] = {
        "benchmark_id": "sparse_docking44_chembl_sensitivity",
        "role": "support_sensitivity_only",
        "source": "results/evidence_summary.json :: "
        "expanded_target_preference_benchmark.primary_all_exact",
        "docking_matrix": "Docking-44 (12,651 ligands x 44 targets)",
        "support": {
            "ligands": int(primary["n_ligands"]),
            "targets": int(primary["n_targets"]),
            "observed_cells": int(primary["observed_experimental_cells"]),
            "non_tied_within_ligand_pairs": int(
                primary["representations"]["absolute_vina"]["evaluated_pairs"]
            ),
            "murcko_clusters": int(primary["scaffold_clusters"]),
        },
        "mean_per_ligand_pairwise_accuracy": {
            name: float(
                primary["representations"][name]["mean_per_ligand_pairwise_accuracy"]
            )
            for name in REPRESENTATION_ORDER
            if name in primary["representations"]
        },
        "contrasts": {},
    }
    for name in HEADLINE_CONTRASTS:
        comparison = primary["paired_comparisons"][name]
        union_90, _ = conservative_union(comparison, "interval_90")
        union_95, _ = conservative_union(comparison, "interval_95")
        reference["contrasts"][name] = {
            "plugin_mean_difference": float(comparison["plugin_mean_difference"]),
            "conservative_cluster_bootstrap_interval_90": union_90,
            "conservative_cluster_bootstrap_interval_95": union_95,
            "conservative_interval_width_90": float(union_90[1] - union_90[0]),
            "conservative_interval_width_95": float(union_95[1] - union_95[0]),
            "smallest_supported_symmetric_equivalence_margin_90": float(
                max(abs(union_90[0]), abs(union_90[1]))
            ),
        }
    return reference


def sensitivity_variant(
    frame: pd.DataFrame,
    docking_reference: pd.DataFrame,
    docking_smiles: pd.Series,
    target_columns: list[str],
    *,
    label: str,
    description: str,
    seed: int,
    bootstrap_repeats: int,
    permutation_repeats: int,
    full_sensitivities: bool = False,
    experimental_prior_reference: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Rerun the primary estimand on a restricted activity contract."""
    experimental_reference, experiment, support = build_support(
        frame, docking_reference, target_columns
    )
    cohort_scaffolds = evidence.scaffold_keys(docking_smiles.reindex(experiment.index))
    prior_reference = (
        experimental_reference
        if experimental_prior_reference is None
        else experimental_prior_reference
    )
    score_matrices, fit_support = score_representations(
        docking_reference, prior_reference, experiment, cohort_scaffolds
    )
    butina = dense_davis._butina_labels(docking_smiles.reindex(experiment.index))
    benchmark = evidence.expanded_preference_metrics(
        score_matrices,
        experiment,
        docking_smiles,
        bootstrap_repeats=bootstrap_repeats,
        permutation_repeats=permutation_repeats,
        seed=seed,
        full_sensitivities=full_sensitivities,
        butina_labels=butina,
    )
    absolute = benchmark["representations"]["absolute_vina"]
    result: dict[str, Any] = {
        "label": label,
        "description": description,
        "seed": int(seed),
        "bootstrap_repeats": int(bootstrap_repeats),
        "permutation_repeats": int(permutation_repeats),
        "support": {
            "ligands": int(benchmark["n_ligands"]),
            "retained_ligands": int(benchmark["n_ligands"]),
            "evaluated_ligands": int(absolute["evaluated_ligands"]),
            "retained_ligands_with_no_informative_non_tied_pair": int(
                benchmark["n_ligands"] - absolute["evaluated_ligands"]
            ),
            "targets": int(benchmark["n_targets"]),
            "observed_cells": int(benchmark["observed_experimental_cells"]),
            "non_tied_within_ligand_pairs": int(
                absolute["evaluated_pairs"]
            ),
            **support,
        },
        "mean_per_ligand_pairwise_accuracy": {
            name: float(
                benchmark["representations"][name]["mean_per_ligand_pairwise_accuracy"]
            )
            for name in REPRESENTATION_ORDER
        },
        "contrasts": {
            name: contrast_equivalence_report(benchmark["paired_comparisons"][name])
            for name in HEADLINE_CONTRASTS
        },
        "fit_support": fit_support,
        "claim_boundary": (
            "This restricted activity arm changes the experimental support and endpoint "
            "contract while retaining the externally fitted docking-score transformation. "
            "Its chemical-cluster intervals are conditional on the observed target labels."
        ),
    }
    if full_sensitivities:
        result["uncertainty_layers"] = uncertainty_layers(benchmark)
        result["target_jackknife_paired_contrasts"] = benchmark[
            "target_jackknife_paired_contrasts"
        ]
    return result


def endpoint_sensitivity_table(
    summary_support: dict[str, Any],
    summary_accuracy: dict[str, float],
    summary_contrasts: dict[str, Any],
    sensitivities: dict[str, Any],
) -> pd.DataFrame:
    """Compact machine-readable table for assay/endpoint heterogeneity."""

    def row(label: str, role: str, record: dict[str, Any]) -> dict[str, Any]:
        support = record["support"]
        accuracy = record["mean_per_ligand_pairwise_accuracy"]
        contrast = record["contrasts"]["two_way_residual_minus_absolute_vina"]
        target_interval = [float("nan"), float("nan")]
        if "target_jackknife_paired_contrasts" in record:
            target_interval = record["target_jackknife_paired_contrasts"][
                "two_way_residual_minus_absolute_vina"
            ]["jackknife_normal_95_interval"]
        mixing = record.get("endpoint_mixing_diagnostic")
        mixing_counts = mixing["pair_counts"] if mixing is not None else {}
        mixing_fractions = (
            mixing["fractions_of_all_informative_pairs"]
            if mixing is not None
            else {}
        )
        return {
            "analysis_arm": label,
            "role": role,
            "retained_ligands": int(
                support.get("retained_ligands", support["ligands"])
            ),
            "evaluated_ligands": int(
                support.get("evaluated_ligands", support["ligands"])
            ),
            "targets": int(support["targets"]),
            "observed_cells": int(support["observed_cells"]),
            "non_tied_pairs": int(support["non_tied_within_ligand_pairs"]),
            "absolute_vina": float(accuracy["absolute_vina"]),
            "column_standardized": float(accuracy["column_standardized"]),
            "two_way_residual": float(accuracy["two_way_residual"]),
            "residual_minus_absolute": float(contrast["plugin_mean_difference"]),
            "chemical_cluster_95_low": float(
                contrast["conservative_cluster_bootstrap_interval_95"][0]
            ),
            "chemical_cluster_95_high": float(
                contrast["conservative_cluster_bootstrap_interval_95"][1]
            ),
            "target_jackknife_95_low": float(target_interval[0]),
            "target_jackknife_95_high": float(target_interval[1]),
            "same_endpoint_pairs": mixing_counts.get("same_endpoint", pd.NA),
            "mixed_Ki_Kd_pairs": mixing_counts.get(
                "mixed_endpoint_Ki_Kd", pd.NA
            ),
            "pairs_involving_pooled_Ki_Kd_cell": mixing_counts.get(
                "involves_pooled_Ki_Kd_cell", pd.NA
            ),
            "same_endpoint_fraction_of_all_pairs": mixing_fractions.get(
                "same_endpoint", float("nan")
            ),
            "mixed_Ki_Kd_fraction_of_all_pairs": mixing_fractions.get(
                "mixed_endpoint_Ki_Kd", float("nan")
            ),
            "pooled_cell_fraction_of_all_pairs": mixing_fractions.get(
                "involves_pooled_Ki_Kd_cell", float("nan")
            ),
        }

    primary = {
        "support": {
            "ligands": summary_support["retained_ligands"],
            "retained_ligands": summary_support["retained_ligands"],
            "evaluated_ligands": summary_support["evaluated_ligands"],
            "targets": summary_support["targets"],
            "observed_cells": summary_support["observed_experimental_cells"],
            "non_tied_within_ligand_pairs": summary_support[
                "non_tied_within_ligand_target_pairs"
            ],
        },
        "mean_per_ligand_pairwise_accuracy": summary_accuracy,
        "contrasts": summary_contrasts,
    }
    rows = [row("all_exact_relations_all_endpoints", "coverage_primary", primary)]
    rows.append(
        row(
            "human_binding_Ki_Kd",
            "endpoint_restricted_sensitivity_with_target_jackknife",
            sensitivities["human_binding_Ki_Kd"],
        )
    )
    for endpoint in ENDPOINTS:
        rows.append(
            row(
                f"single_endpoint_{endpoint}",
                "single_endpoint_sensitivity",
                sensitivities["single_endpoint"][endpoint],
            )
        )
    return pd.DataFrame.from_records(rows)


def build_tables(
    benchmark: dict[str, Any],
    experiment: pd.DataFrame,
    docking_reference: pd.DataFrame,
    docking_smiles: pd.Series,
    score_matrices: dict[str, np.ndarray],
    murcko: np.ndarray,
    butina: np.ndarray,
) -> dict[str, pd.DataFrame]:
    """Per-ligand, per-representation, per-contrast and per-target release tables."""
    experiment_values = experiment.to_numpy(dtype=float)
    metrics = {
        name: evidence.preference_metrics(score_matrices[name], experiment_values)
        for name in REPRESENTATION_ORDER
    }
    ligand_table = pd.DataFrame(
        {
            "inchikey": experiment.index.astype(str),
            "smiles": docking_smiles.reindex(experiment.index).to_numpy(),
            "murcko_scaffold": murcko,
            "butina_cluster": butina,
            "observed_targets": experiment.notna().sum(axis=1).to_numpy(dtype=int),
            "nontied_target_pairs": metrics["absolute_vina"]["per_ligand_pair_count"],
        }
    )
    for name in REPRESENTATION_ORDER:
        ligand_table[f"accuracy_{name}"] = metrics[name]["per_ligand_pairwise_accuracy"]
    ligand_table["two_way_residual_minus_absolute_vina"] = (
        ligand_table.accuracy_two_way_residual - ligand_table.accuracy_absolute_vina
    )
    ligand_table["two_way_residual_minus_column_standardized"] = (
        ligand_table.accuracy_two_way_residual
        - ligand_table.accuracy_column_standardized
    )

    representation_rows = []
    for name in REPRESENTATION_ORDER:
        report = benchmark["representations"][name]
        row = {
            "representation": name,
            "mean_per_ligand_pairwise_accuracy": report[
                "mean_per_ligand_pairwise_accuracy"
            ],
            "pair_weighted_accuracy": report["pair_weighted_accuracy"],
            "mean_per_ligand_spearman": report["mean_per_ligand_spearman"],
            "top1_accuracy": report["top1_accuracy"],
            "evaluated_ligands": report["evaluated_ligands"],
            "evaluated_pairs": report["evaluated_pairs"],
            "predicted_score_ties_half_credit": report[
                "predicted_score_ties_half_credit"
            ],
        }
        for source, prefix in (
            ("ligand_bootstrap", "ligand"),
            ("scaffold_cluster_bootstrap", "murcko"),
            ("butina_cluster_bootstrap", "butina"),
        ):
            if source not in report:
                continue
            row[f"{prefix}_bootstrap_low_90"] = report[source]["interval_90"][0]
            row[f"{prefix}_bootstrap_high_90"] = report[source]["interval_90"][1]
            row[f"{prefix}_bootstrap_low_95"] = report[source]["interval_95"][0]
            row[f"{prefix}_bootstrap_high_95"] = report[source]["interval_95"][1]
        representation_rows.append(row)
    representation_table = pd.DataFrame(representation_rows)

    contrast_rows = []
    for name, comparison in benchmark["paired_comparisons"].items():
        report = contrast_equivalence_report(comparison)
        contrast_rows.append(
            {
                "contrast": name,
                "plugin_mean_difference": report["plugin_mean_difference"],
                "conservative_low_90": report[
                    "conservative_cluster_bootstrap_interval_90"
                ][0],
                "conservative_high_90": report[
                    "conservative_cluster_bootstrap_interval_90"
                ][1],
                "conservative_width_90": report["conservative_interval_width_90"],
                "conservative_low_95": report[
                    "conservative_cluster_bootstrap_interval_95"
                ][0],
                "conservative_high_95": report[
                    "conservative_cluster_bootstrap_interval_95"
                ][1],
                "conservative_width_95": report["conservative_interval_width_95"],
                "smallest_supported_symmetric_margin_90": report[
                    "smallest_supported_symmetric_equivalence_margin_90"
                ],
                "excludes_zero_90": report["excludes_zero_90"],
                "excludes_zero_95": report["excludes_zero_95"],
            }
        )
    contrast_table = pd.DataFrame(contrast_rows)

    target_table = pd.DataFrame(
        {
            "target": [str(column) for column in experiment.columns],
            "ligands_observed": experiment.notna().sum(axis=0).to_numpy(dtype=int),
            "mean_pchembl": experiment.mean(axis=0).to_numpy(dtype=float),
            "median_pchembl": experiment.median(axis=0).to_numpy(dtype=float),
            "mean_dockstring_score_full_reference": docking_reference[
                experiment.columns
            ]
            .mean(axis=0)
            .to_numpy(dtype=float),
            "mean_dockstring_score_evaluated_cohort": docking_reference.loc[
                experiment.index, experiment.columns
            ]
            .mean(axis=0)
            .to_numpy(dtype=float),
        }
    )
    return {
        "per_ligand_metrics": ligand_table,
        "representation_summary": representation_table,
        "contrast_summary": contrast_table,
        "target_support": target_table,
    }


def evaluate(
    *,
    activity_path: Path,
    seed: int,
    bootstrap_repeats: int,
    permutation_repeats: int,
    sensitivity_bootstrap_repeats: int,
    sensitivity_permutation_repeats: int,
    run_sensitivities: bool,
) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
    gene_to_chembl, chembl_to_gene = gene_target_contract()
    docking_reference, docking_smiles, docking_report = load_docking_reference()
    target_columns = [
        gene for gene in sorted(gene_to_chembl) if gene in docking_reference.columns
    ]
    missing_targets = sorted(set(gene_to_chembl) - set(target_columns))
    if missing_targets:
        raise ValueError(
            f"ChEMBL target contract lists genes absent from DOCKSTRING: {missing_targets}"
        )

    activities, activity_report = load_activities(
        chembl_to_gene, activity_path=activity_path
    )
    experimental_reference, experiment, support_report = build_support(
        activities, docking_reference, target_columns
    )
    evaluation_smiles = docking_smiles.reindex(experiment.index)
    if evaluation_smiles.isna().any():
        raise ValueError("an evaluated ligand has no DOCKSTRING SMILES")
    murcko = evidence.scaffold_keys(evaluation_smiles)
    butina = dense_davis._butina_labels(evaluation_smiles)

    score_matrices, fit_support = score_representations(
        docking_reference, experimental_reference, experiment, murcko
    )
    benchmark = evidence.expanded_preference_metrics(
        score_matrices,
        experiment,
        docking_smiles,
        bootstrap_repeats=bootstrap_repeats,
        permutation_repeats=permutation_repeats,
        seed=seed,
        full_sensitivities=True,
        butina_labels=butina,
    )
    benchmark["fit_support"] = fit_support

    non_tied_pairs = int(
        benchmark["representations"]["absolute_vina"]["evaluated_pairs"]
    )
    evaluated_ligands = int(
        benchmark["representations"]["absolute_vina"]["evaluated_ligands"]
    )
    retained_ligands = int(benchmark["n_ligands"])
    if evaluated_ligands > retained_ligands:
        raise ValueError("evaluated ligand count cannot exceed the retained cohort")
    contrasts = {
        name: contrast_equivalence_report(benchmark["paired_comparisons"][name])
        for name in HEADLINE_CONTRASTS
    }
    published = published_reference()

    scale_comparison: dict[str, Any] = {}
    if published is not None:
        scale_comparison = {
            "ligand_multiplier": float(
                len(experiment) / published["support"]["ligands"]
            ),
            "non_tied_pair_multiplier": float(
                non_tied_pairs / published["support"]["non_tied_within_ligand_pairs"]
            ),
            "observed_cell_multiplier": float(
                int(experiment.notna().sum().sum())
                / published["support"]["observed_cells"]
            ),
            "conservative_interval_width_ratio": {
                name: {
                    "level_90": float(
                        contrasts[name]["conservative_interval_width_90"]
                        / published["contrasts"][name]["conservative_interval_width_90"]
                    ),
                    "level_95": float(
                        contrasts[name]["conservative_interval_width_95"]
                        / published["contrasts"][name]["conservative_interval_width_95"]
                    ),
                }
                for name in HEADLINE_CONTRASTS
            },
        }

    tables = build_tables(
        benchmark,
        experiment,
        docking_reference,
        docking_smiles,
        score_matrices,
        murcko,
        butina,
    )

    sensitivities: dict[str, Any] = {}
    if run_sensitivities:
        floor_frame = activities[activities.pchembl_value >= 6.0]
        sensitivities["high_potency_cells_pchembl_at_least_6"] = sensitivity_variant(
            floor_frame,
            docking_reference,
            docking_smiles,
            target_columns,
            label="high_potency_cells_pchembl_at_least_6",
            description=(
                "the estimand the on-disk chemistry-graph snapshot would have supported: "
                "ordering restricted to cells with pChEMBL at least 6"
            ),
            seed=seed + 1000,
            bootstrap_repeats=sensitivity_bootstrap_repeats,
            permutation_repeats=sensitivity_permutation_repeats,
        )
        human_binding_frame = activities[
            activities.target_organism.eq("Homo sapiens")
            & activities.assay_type.eq("B")
        ]
        sensitivities["human_binding_all_endpoints"] = sensitivity_variant(
            human_binding_frame,
            docking_reference,
            docking_smiles,
            target_columns,
            label="human_binding_all_endpoints",
            description=(
                "mirror of the published assay-restricted co-primary: human organism "
                "annotation and assay_type B only"
            ),
            seed=seed + 2000,
            bootstrap_repeats=sensitivity_bootstrap_repeats,
            permutation_repeats=sensitivity_permutation_repeats,
        )

        # This endpoint-restricted, less heterogeneous arm is large enough for both
        # chemical-support uncertainty and a delete-one-target composition sensitivity.
        human_binding_kikd = human_binding_frame[
            human_binding_frame.standard_type.isin(["Ki", "Kd"])
        ]
        sensitivities["human_binding_Ki_Kd"] = sensitivity_variant(
            human_binding_kikd,
            docking_reference,
            docking_smiles,
            target_columns,
            label="human_binding_Ki_Kd",
            description=(
                "human SINGLE PROTEIN binding assays restricted to Ki or Kd; this "
                "endpoint-restricted arm receives both chemical-support and delete-one-target "
                "uncertainty analyses"
            ),
            seed=seed + 3000,
            bootstrap_repeats=bootstrap_repeats,
            permutation_repeats=sensitivity_permutation_repeats,
            full_sensitivities=True,
        )
        kikd_mixing = kikd_endpoint_mixing_diagnostic(
            human_binding_kikd, docking_reference, target_columns
        )
        if (
            kikd_mixing["informative_non_tied_pairs"]
            != sensitivities["human_binding_Ki_Kd"]["support"][
                "non_tied_within_ligand_pairs"
            ]
        ):
            raise ValueError(
                "Ki/Kd endpoint-provenance diagnostic does not reproduce the human-binding "
                "arm's informative-pair count"
            )
        sensitivities["human_binding_Ki_Kd"][
            "endpoint_mixing_diagnostic"
        ] = kikd_mixing

        # Separate endpoint arms expose rather than average heterogeneity between Ki,
        # Kd, IC50 and EC50.  The all-endpoint activity reference is supplied only to
        # make target-prior fallbacks finite in very small endpoint arms; the reported
        # docking-score contrasts do not use that prior.
        endpoint_variants: dict[str, Any] = {}
        for endpoint_index, endpoint in enumerate(ENDPOINTS):
            endpoint_variants[endpoint] = sensitivity_variant(
                activities[activities.standard_type.eq(endpoint)],
                docking_reference,
                docking_smiles,
                target_columns,
                label=f"single_endpoint_{endpoint}",
                description=(
                    f"within-{endpoint} activity support only; no target pair compares "
                    "different ChEMBL standard_type values"
                ),
                seed=seed + 4000 + 1000 * endpoint_index,
                bootstrap_repeats=sensitivity_bootstrap_repeats,
                permutation_repeats=max(200, sensitivity_permutation_repeats // 2),
                experimental_prior_reference=experimental_reference,
            )
            endpoint_variants[endpoint]["reported_scope"] = (
                "docking-derived representations and their paired contrasts; target-only "
                "priors are retained for pipeline completeness but are not interpreted"
            )
        sensitivities["single_endpoint"] = endpoint_variants

        endpoint_experiments = {
            endpoint: activities[activities.standard_type.eq(endpoint)]
            .pivot_table(
                index="inchikey",
                columns="gene",
                values="pchembl_value",
                aggfunc="median",
            )
            .reindex(index=experiment.index, columns=experiment.columns)
            for endpoint in ENDPOINTS
        }
        sensitivities["same_endpoint_pairs_on_primary_cohort"] = (
            evidence.same_endpoint_pair_sensitivity(
                score_matrices,
                endpoint_experiments,
                evaluation_smiles,
                butina,
                bootstrap_repeats=sensitivity_bootstrap_repeats,
                seed=seed + 9000,
            )
        )

    summary: dict[str, Any] = {
        "schema_version": "1.1.0",
        "analysis": "dockstring_chembl_ligand_wise_target_ranking_benchmark",
        "benchmark_id": "broad_dockstring_chembl_primary",
        "role": "primary_broad_support_observed_pair_benchmark",
        "analysis_question": (
            "Does the broad-support, sparse-cell ChEMBL target-ranking benchmark reach a "
            "resolved conclusion when it is evaluated on the 260,060-row DOCKSTRING "
            "matrix used by the paper's dense checks, instead of on the 12,651-row "
            "Docking-44 matrix? "
            "Specifically: on an approximately 48-fold larger matched ligand cohort, "
            "is the scaled two-way residual contrast against absolute Vina or "
            "column-standardized Vina resolved at the reported precision when ranking a "
            "ligand's measured targets?"
        ),
        "status": "complete",
        "seed": int(seed),
        "seeds": {
            "primary": int(seed),
            "high_potency_sensitivity": int(seed + 1000),
            "human_binding_sensitivity": int(seed + 2000),
            "human_binding_Ki_Kd_sensitivity": int(seed + 3000),
            "single_endpoint_sensitivities": {
                endpoint: int(seed + 4000 + 1000 * endpoint_index)
                for endpoint_index, endpoint in enumerate(ENDPOINTS)
            },
            "same_endpoint_pairs_on_primary_cohort": int(seed + 9000),
            "note": (
                "build_evidence.expanded_preference_metrics derives every bootstrap and "
                "permutation stream deterministically from the primary seed"
            ),
        },
        "support": {
            # ``ligands`` is retained as a backwards-compatible alias.  The explicit
            # names below prevent the 6,557 retained rows from being confused with the
            # 6,480 rows that contribute at least one non-tied target comparison.
            "ligands": retained_ligands,
            "retained_ligands": retained_ligands,
            "evaluated_ligands": evaluated_ligands,
            "retained_ligands_with_no_informative_non_tied_pair": (
                retained_ligands - evaluated_ligands
            ),
            "targets": int(benchmark["n_targets"]),
            "target_ids": [str(column) for column in experiment.columns],
            "observed_experimental_cells": int(
                benchmark["observed_experimental_cells"]
            ),
            "observed_fraction": float(benchmark["observed_fraction"]),
            "non_tied_within_ligand_target_pairs": non_tied_pairs,
            "excluded_exact_experimental_ties": int(
                benchmark["representations"]["absolute_vina"]["excluded_ties"]
            ),
            "murcko_scaffold_clusters": int(benchmark["scaffold_clusters"]),
            "butina_clusters": int(len(np.unique(butina))),
            "docking_reference_ligands": int(len(docking_reference)),
            "docking_reference_targets": int(docking_reference.shape[1]),
            "external_docking_training_ligands": int(
                fit_support["docking_training_ligands"]
            ),
            "external_experimental_prior_ligands": int(
                fit_support["experimental_prior_training_ligands"]
            ),
            "experimental_cell_contract": (
                "non-null pChEMBL; exact standard relation; Ki, Kd, IC50 or EC50; median "
                "pChEMBL per full-InChIKey and DOCKSTRING gene-symbol target cell; at "
                "least two observed targets per ligand"
            ),
        },
        "docking_reference": docking_report,
        "activity_snapshot": activity_report,
        "matching": support_report,
        "fit_contract": fit_support,
        "clustering_contract": {
            "murcko": (
                "Bemis-Murcko scaffold SMILES from build_evidence.scaffold_keys; acyclic "
                "molecules remain singletons"
            ),
            "butina": (
                "RDKit Morgan fingerprints, radius 2, 2048 bits; Butina clustering at "
                "Tanimoto similarity 0.65, computed here because DOCKSTRING ships no "
                "frozen Butina column (dense_davis_benchmark._butina_labels)"
            ),
            "butina_differs_from_published": (
                "the published Docking-44 benchmark reuses the frozen Butina_clusters "
                "column of df_final_v4; this rebuild recomputes Butina labels on the "
                "evaluated cohort under the package's DAVIS/PKIS2 clustering contract"
            ),
        },
        "estimands": {
            "primary": (
                "mean per-ligand pairwise target-preference accuracy over observed "
                "exact-relation median pChEMBL cells; exact experimental ties excluded; "
                "lower docking scores predict higher pChEMBL"
            ),
            "spectrally_motivated_representation": "two_way_residual",
            "multiplicity_note": benchmark["multiplicity_note"],
        },
        "mean_per_ligand_pairwise_accuracy": {
            name: float(
                benchmark["representations"][name]["mean_per_ligand_pairwise_accuracy"]
            )
            for name in REPRESENTATION_ORDER
        },
        "ranking_order": ranking_order_summary(benchmark),
        "headline_contrasts": contrasts,
        "uncertainty_layers": uncertainty_layers(benchmark),
        "scale_comparison_with_published_docking44_benchmark": scale_comparison,
        "published_docking44_reference": published,
        "benchmark": benchmark,
        "sensitivities": sensitivities,
        "bootstrap": {
            "repeats": int(bootstrap_repeats),
            "permutation_repeats": int(permutation_repeats),
            "units": (
                "ligands, Bemis-Murcko scaffold clusters, and recomputed Butina clusters; "
                "the reported interval is the conservative union of the two chemical-cluster "
                "intervals"
            ),
        },
        "claim_boundary": (
            "This benchmark measures only within-ligand ordering of the targets a ligand was "
            "actually measured against, on the ChEMBL cells that exist for 31 of DOCKSTRING's "
            "58 targets. It does NOT show that docking retrieves unmeasured targets, does NOT "
            "estimate absolute affinity, and does NOT establish a second experimental "
            "eigenspectrum. It does NOT license a general claim that residual centering helps "
            "or does not help selectivity ranking: the cohort is the sparse public-ChEMBL "
            "overlap, is dominated by ligands measured on exactly two targets, and its target "
            "set is 20 kinases plus 11 GPCR, nuclear-receptor and enzyme targets rather than a "
            "random protein sample. The activity snapshot is a single ChEMBL pull with no "
            "assay-heterogeneity, potency-censoring or publication-bias correction, and "
            "measurement error in the pChEMBL cells is not propagated. Butina labels are "
            "recomputed rather than inherited from the frozen Docking-44 clustering, so the "
            "conservative union is not byte-comparable with the published one. Finally, the "
            "DOCKSTRING and Docking-44 benchmarks share ChEMBL as a source and are not "
            "independent replications: they are the same estimand on two docking matrices. "
            "The human-binding Ki/Kd and single-endpoint arms expose endpoint dependence; "
            "their chemical-support intervals remain conditional on fixed, non-random "
            "target panels, and the Ki/Kd delete-one-target interval is reported separately."
        ),
    }
    if run_sensitivities:
        tables["endpoint_sensitivity_summary"] = endpoint_sensitivity_table(
            summary["support"],
            summary["mean_per_ligand_pairwise_accuracy"],
            summary["headline_contrasts"],
            sensitivities,
        )
    return summary, tables


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--activities", type=Path, default=ACTIVITY_INPUT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--bootstrap-repeats", type=int, default=5000)
    parser.add_argument("--permutation-repeats", type=int, default=5000)
    parser.add_argument("--sensitivity-bootstrap-repeats", type=int, default=2000)
    parser.add_argument("--sensitivity-permutation-repeats", type=int, default=1000)
    parser.add_argument(
        "--no-sensitivities",
        dest="run_sensitivities",
        action="store_false",
        help="skip the potency-floor and assay-restricted reruns",
    )
    parser.set_defaults(run_sensitivities=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary, tables = evaluate(
        activity_path=args.activities,
        seed=args.seed,
        bootstrap_repeats=args.bootstrap_repeats,
        permutation_repeats=args.permutation_repeats,
        sensitivity_bootstrap_repeats=args.sensitivity_bootstrap_repeats,
        sensitivity_permutation_repeats=args.sensitivity_permutation_repeats,
        run_sensitivities=args.run_sensitivities,
    )
    summary["frozen_inputs"] = {
        str(path.relative_to(PACKAGE)): sha256(path)
        for path in (DOCKSTRING_INPUT, IDENTITY_INPUT, args.activities)
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    for name, table in tables.items():
        table.to_csv(args.output / f"{name}.csv", index=False)
    print(
        json.dumps(
            {
                "support": summary["support"],
                "mean_per_ligand_pairwise_accuracy": summary[
                    "mean_per_ligand_pairwise_accuracy"
                ],
                "headline_contrasts": summary["headline_contrasts"],
                "scale_comparison": summary[
                    "scale_comparison_with_published_docking44_benchmark"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
