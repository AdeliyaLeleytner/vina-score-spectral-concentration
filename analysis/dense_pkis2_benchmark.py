#!/usr/bin/env python3
"""Dense PKIS2 x DOCKSTRING target-preference benchmark.

This benchmark matches PKIS2 compounds to the released DOCKSTRING structures
and evaluates the 21 kinase targets shared by the resources.  PKIS2 is a dense
single-concentration KINOMEscan displacement panel: its values are percentages
of inhibition/displacement at 1 micromolar, not equilibrium affinities.

The primary molecular identity is the full Standard InChIKey recomputed from
the released SMILES on both sides.  Duplicate PKIS2 measurements and duplicate
DOCKSTRING profiles are collapsed cell-wise by the median before analysis.
Every DOCKSTRING row whose recomputed Standard InChI connectivity block occurs
in the evaluation support is excluded before target offsets or scales are fit.

The primary endpoint is the equal-ligand mean of within-ligand pairwise target-
preference concordance for experimental differences exceeding 10 percentage
points.  Exact non-ties and 5- and 20-point margins, binary activity retrieval
at 65%, identity/duplicate sensitivities, chemical-cluster uncertainty, and a
delete-one-target analysis are reported as secondary analyses.
"""

from __future__ import annotations

import argparse
import json
import warnings
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.neighbors import NearestNeighbors

try:  # Direct script execution and package-style import are both supported.
    from .dense_davis_benchmark import (
        BASELINE_REPRESENTATIONS,
        CORE_REPRESENTATIONS,
        MATCH_VARIANTS,
        OPERATIONAL_PATH_STEPS,
        PAIRWISE_CONTRASTS,
        _butina_labels,
        _finite_float,
        _identity_table,
        _murcko_labels,
        _score_representations,
        _uncertainty,
    )
except ImportError:
    from dense_davis_benchmark import (  # type: ignore
        BASELINE_REPRESENTATIONS,
        CORE_REPRESENTATIONS,
        MATCH_VARIANTS,
        OPERATIONAL_PATH_STEPS,
        PAIRWISE_CONTRASTS,
        _butina_labels,
        _finite_float,
        _identity_table,
        _murcko_labels,
        _score_representations,
        _uncertainty,
    )


PACKAGE = Path(__file__).resolve().parents[1]
DEFAULT_DOCKSTRING = PACKAGE / "data" / "frozen" / "dockstring-dataset.tsv.gz"
DEFAULT_PKIS2 = PACKAGE / "data" / "frozen" / "pkis2_s4.xlsx"
PKIS2_SHEET = "Table 4 - PKIS2 %Inh"

# Ordered by the released DOCKSTRING target columns.  PKIS2 column names refer
# to the specific DiscoverX constructs in the source workbook; potentially
# ambiguous alternative constructs (for example autoinhibited KIT) are not used.
TARGET_MAP: OrderedDict[str, str] = OrderedDict(
    [
        ("ABL1", "ABL1-nonphosphorylated"),
        ("AKT1", "AKT1"),
        ("AKT2", "AKT2"),
        ("CDK2", "CDK2"),
        ("CSF1R", "CSF1R"),
        ("EGFR", "EGFR"),
        ("FGFR1", "FGFR1"),
        ("IGF1R", "IGF1R"),
        ("JAK2", "JAK2(JH1domain-catalytic)"),
        ("KDR", "VEGFR2"),
        ("KIT", "KIT"),
        ("LCK", "LCK"),
        ("MAP2K1", "MEK1"),
        ("MAPK1", "ERK2"),
        ("MAPK14", "p38-alpha"),
        ("MAPKAPK2", "MAPKAPK2"),
        ("MET", "MET"),
        ("PLK1", "PLK1"),
        ("PTK2", "FAK"),
        ("ROCK1", "ROCK1"),
        ("SRC", "SRC"),
    ]
)

MARGINS: OrderedDict[str, float] = OrderedDict(
    [
        ("exact_non_ties", 0.0),
        ("absolute_difference_gt_5", 5.0),
        ("absolute_difference_gt_10", 10.0),
        ("absolute_difference_gt_20", 20.0),
    ]
)
PRIMARY_MARGIN = "absolute_difference_gt_10"
ACTIVITY_THRESHOLD = 65.0
REPORTED_REPRESENTATIONS = (*CORE_REPRESENTATIONS, *BASELINE_REPRESENTATIONS)


def _aggregate_rows(
    frame: pd.DataFrame,
    identity_column: str,
    value_columns: list[str],
    method: str,
) -> pd.DataFrame:
    grouped = frame.groupby(identity_column, sort=True)[value_columns]
    if method == "median":
        return grouped.median()
    if method == "mean":
        return grouped.mean()
    if method == "first":
        return grouped.first()
    if method == "last":
        return grouped.last()
    raise ValueError(f"Unsupported duplicate aggregation: {method}")


def _load_inputs(
    dockstring_path: Path,
    pkis2_path: Path,
    identity_scan: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # The official workbook contains an unsupported Excel drawing/extension.
    # openpyxl drops that presentation-only object while preserving cell data.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Unknown extension is not supported and will be removed",
            category=UserWarning,
            module=r"openpyxl\.worksheet\._reader",
        )
        pkis2 = pd.read_excel(pkis2_path, sheet_name=PKIS2_SHEET)
    if len(pkis2.columns) != 413:
        raise ValueError(
            "Expected 7 PKIS2 metadata columns plus 406 assay columns; "
            f"observed {len(pkis2.columns)} total columns"
        )
    required_metadata = ["Regno", "Compound", "Chemotype", "Smiles"]
    missing_columns = [
        column
        for column in [*required_metadata, *TARGET_MAP.values()]
        if column not in pkis2.columns
    ]
    if missing_columns:
        raise ValueError(f"PKIS2 workbook is missing columns: {missing_columns}")
    # The published workbook contains one trailing formatting artefact with no
    # compound metadata (but a stray value in one non-selected assay column).
    pkis2 = pkis2.dropna(subset=required_metadata).copy()
    if len(pkis2) != 645:
        raise ValueError(f"Expected 645 populated PKIS2 rows, observed {len(pkis2)}")
    if pkis2[required_metadata].isna().any().any():
        raise ValueError("A populated PKIS2 row lacks required compound metadata")
    pkis2.insert(0, "pkis2_row", np.arange(len(pkis2), dtype=int))
    pkis2.insert(1, "source_excel_row", np.arange(2, len(pkis2) + 2, dtype=int))
    pkis2 = _identity_table(pkis2, "Smiles", scan="full")

    # Rename the experimental columns to the DOCKSTRING target names once, so
    # every downstream matrix has the same explicit order.
    experimental = pkis2[list(TARGET_MAP.values())].apply(
        pd.to_numeric, errors="coerce"
    )
    experimental.columns = list(TARGET_MAP)
    if experimental.isna().any().any():
        raise ValueError("The selected PKIS2 645 x 21 block is not fully numeric/dense")
    if ((experimental < 0.0) | (experimental > 100.0)).any().any():
        raise ValueError("A selected PKIS2 percentage is outside [0, 100]")
    pkis2[list(TARGET_MAP)] = experimental

    dockstring_all = pd.read_csv(dockstring_path, sep="\t")
    score_columns = [
        column
        for column in dockstring_all.columns
        if column not in {"inchikey", "smiles"}
    ]
    absent_targets = [target for target in TARGET_MAP if target not in score_columns]
    if absent_targets:
        raise ValueError(f"DOCKSTRING is missing target columns: {absent_targets}")
    complete = ~dockstring_all[score_columns].isna().any(axis=1)
    dockstring = dockstring_all.loc[
        complete, ["inchikey", "smiles", *TARGET_MAP]
    ].copy()
    if len(dockstring) != 260060 or dockstring[list(TARGET_MAP)].isna().any().any():
        raise ValueError(
            "Expected the complete 260,060-row DOCKSTRING support; "
            f"observed {len(dockstring):,} rows"
        )
    dockstring.insert(0, "dockstring_row", dockstring.index.to_numpy(dtype=int))
    pkis2_blocks = set(pkis2.connectivity_block)
    dockstring = _identity_table(
        dockstring,
        "smiles",
        scan=identity_scan,
        davis_connectivity_blocks=pkis2_blocks,
    )
    return dockstring, pkis2


def _matched_surface(
    dockstring: pd.DataFrame,
    pkis2: pd.DataFrame,
    identity_column: str,
    experimental_aggregation: str = "median",
    docking_aggregation: str = "median",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    targets = list(TARGET_MAP)
    common = sorted(
        set(pkis2[identity_column].dropna())
        & set(dockstring[identity_column].dropna())
    )
    if not common:
        raise ValueError(f"No PKIS2 x DOCKSTRING overlap under {identity_column}")
    pkis_common = pkis2[pkis2[identity_column].isin(common)]
    ds_common = dockstring[dockstring[identity_column].isin(common)]
    experiment = _aggregate_rows(
        pkis_common,
        identity_column,
        targets,
        experimental_aggregation,
    )
    docking = _aggregate_rows(
        ds_common,
        identity_column,
        targets,
        docking_aggregation,
    ).reindex(experiment.index)
    smiles = (
        pkis_common.groupby(identity_column, sort=True)["Smiles"]
        .first()
        .reindex(experiment.index)
        .to_frame("smiles")
    )
    if docking.isna().any().any() or experiment.isna().any().any():
        raise ValueError("A matched PKIS2 x DOCKSTRING surface is not dense")
    return docking, experiment, smiles


def _pairwise_concordance(
    scores: np.ndarray,
    experiment: np.ndarray,
    *,
    margin: float,
    pair_stratum: str = "all",
    activity_threshold: float = ACTIVITY_THRESHOLD,
) -> tuple[dict, np.ndarray, np.ndarray]:
    """Equal-ligand target-preference concordance on a dense panel.

    Higher PKIS2 percentage predicts target preference, whereas a lower Vina
    score predicts target preference.  Experimental ties and differences not
    exceeding ``margin`` are excluded; score ties receive half credit.
    """
    scores = np.asarray(scores, dtype=float)
    experiment = np.asarray(experiment, dtype=float)
    if scores.shape != experiment.shape or scores.ndim != 2:
        raise ValueError("Score and experiment matrices must have identical 2-D shape")
    if margin < 0:
        raise ValueError("The experimental difference margin cannot be negative")
    allowed = {"all", "both_active", "active_vs_inactive", "both_inactive"}
    if pair_stratum not in allowed:
        raise ValueError(f"Unknown pair stratum: {pair_stratum}")

    first, second = np.triu_indices(experiment.shape[1], k=1)
    per_ligand = np.full(len(experiment), np.nan)
    per_ligand_pairs = np.zeros(len(experiment), dtype=int)
    total_correct = 0.0
    score_ties = 0
    excluded_margin = 0
    excluded_stratum = 0
    for ligand, (score_row, experimental_row) in enumerate(zip(scores, experiment)):
        difference = experimental_row[first] - experimental_row[second]
        informative = np.abs(difference) > margin + 1e-12
        excluded_margin += int((~informative).sum())
        first_active = experimental_row[first] >= activity_threshold
        second_active = experimental_row[second] >= activity_threshold
        if pair_stratum == "both_active":
            stratum = first_active & second_active
        elif pair_stratum == "active_vs_inactive":
            stratum = first_active ^ second_active
        elif pair_stratum == "both_inactive":
            stratum = ~first_active & ~second_active
        else:
            stratum = np.ones_like(informative, dtype=bool)
        excluded_stratum += int((informative & ~stratum).sum())
        keep = informative & stratum
        if not keep.any():
            continue
        score_difference = score_row[first[keep]] - score_row[second[keep]]
        experimental_difference = difference[keep]
        tied = np.isclose(score_difference, 0.0, rtol=0, atol=1e-12)
        correct = float(tied.sum()) * 0.5
        correct += float(
            (
                np.sign(experimental_difference[~tied])
                == -np.sign(score_difference[~tied])
            ).sum()
        )
        pairs = int(keep.sum())
        per_ligand[ligand] = correct / pairs
        per_ligand_pairs[ligand] = pairs
        total_correct += correct
        score_ties += int(tied.sum())

    finite = np.isfinite(per_ligand)
    top1: list[float] = []
    spearman: list[float] = []
    if pair_stratum == "all":
        for score_row, experimental_row in zip(scores, experiment):
            best = np.flatnonzero(
                np.isclose(
                    experimental_row,
                    experimental_row.max(),
                    rtol=0,
                    atol=1e-12,
                )
            )
            top1.append(float(int(np.argmin(score_row)) in set(best.tolist())))
            if np.ptp(score_row) <= 1e-12 or np.ptp(experimental_row) <= 1e-12:
                spearman.append(np.nan)
            else:
                value = stats.spearmanr(-score_row, experimental_row).statistic
                spearman.append(float(value) if np.isfinite(value) else np.nan)
    total_pairs = int(per_ligand_pairs.sum())
    spearman_array = np.asarray(spearman, dtype=float)
    finite_spearman = np.isfinite(spearman_array)
    summary = {
        "mean_per_ligand_pairwise_concordance": (
            float(np.nanmean(per_ligand)) if finite.any() else None
        ),
        "pair_weighted_concordance": (
            float(total_correct / total_pairs) if total_pairs else None
        ),
        "evaluated_ligands": int(finite.sum()),
        "evaluated_pairs": total_pairs,
        "median_pairs_per_evaluated_ligand": (
            float(np.median(per_ligand_pairs[finite])) if finite.any() else None
        ),
        "minimum_pairs_per_evaluated_ligand": (
            int(per_ligand_pairs[finite].min()) if finite.any() else None
        ),
        "maximum_pairs_per_evaluated_ligand": (
            int(per_ligand_pairs[finite].max()) if finite.any() else None
        ),
        "excluded_by_experimental_margin": int(excluded_margin),
        "excluded_by_activity_stratum_after_margin": int(excluded_stratum),
        "predicted_score_ties_half_credit": int(score_ties),
        "chance_concordance": 0.5,
        "difference_from_chance_0_5": (
            float(np.nanmean(per_ligand) - 0.5) if finite.any() else None
        ),
        "mean_within_ligand_spearman": (
            float(spearman_array[finite_spearman].mean())
            if finite_spearman.any()
            else None
        ),
        "spearman_evaluated_ligands": int(finite_spearman.sum()),
        "top1_accuracy_allowing_experimental_ties": (
            _finite_float(np.mean(top1)) if top1 else None
        ),
        "top1_evaluated_ligands": int(len(top1)),
    }
    return summary, per_ligand, per_ligand_pairs


def _binary_activity_metrics(
    scores: np.ndarray,
    experiment: np.ndarray,
    threshold: float = ACTIVITY_THRESHOLD,
) -> tuple[dict, np.ndarray, np.ndarray]:
    scores = np.asarray(scores, dtype=float)
    experiment = np.asarray(experiment, dtype=float)
    if scores.shape != experiment.shape:
        raise ValueError("Score and experiment matrices must have identical shape")
    auc = np.full(len(scores), np.nan)
    average_precision = np.full(len(scores), np.nan)
    prevalence = np.full(len(scores), np.nan)
    positive_negative_pairs = 0
    for ligand, (score_row, experimental_row) in enumerate(zip(scores, experiment)):
        active = experimental_row >= threshold
        positives = int(active.sum())
        negatives = int((~active).sum())
        if not positives or not negatives:
            continue
        # Transformations can create sub-picounit floating-point differences
        # between conceptually tied scores.  Twelve-decimal rounding matches the
        # explicit 1e-12 tie convention used by pairwise concordance.
        prediction = np.round(-score_row, decimals=12)
        auc[ligand] = roc_auc_score(active, prediction)
        average_precision[ligand] = average_precision_score(active, prediction)
        prevalence[ligand] = positives / len(active)
        positive_negative_pairs += positives * negatives
    finite = np.isfinite(auc)
    summary = {
        "mean_per_ligand_roc_auc": float(np.nanmean(auc)) if finite.any() else None,
        "mean_per_ligand_average_precision": (
            float(np.nanmean(average_precision)) if finite.any() else None
        ),
        "mean_per_ligand_active_prevalence": (
            float(np.nanmean(prevalence)) if finite.any() else None
        ),
        "evaluated_ligands_with_both_classes": int(finite.sum()),
        "positive_negative_pairs": int(positive_negative_pairs),
        "positive_class": f"PKIS2 percentage inhibition >= {threshold:g}",
        "negative_class": f"PKIS2 percentage inhibition < {threshold:g}",
        "prediction_direction": "more-negative docking score predicts activity",
    }
    return summary, auc, average_precision


def _panel_analysis(
    reference: pd.DataFrame,
    docking: pd.DataFrame,
    experiment: pd.DataFrame,
    smiles: pd.Series,
    bootstrap_repeats: int,
    seed: int,
    *,
    include_uncertainty: bool,
) -> tuple[dict, dict[str, list[dict]]]:
    targets = list(experiment.columns)
    representations, fit = _score_representations(
        reference[targets].to_numpy(dtype=float),
        docking[targets].to_numpy(dtype=float),
    )
    experimental_values = experiment[targets].to_numpy(dtype=float)
    murcko = _murcko_labels(smiles)
    butina = _butina_labels(smiles, similarity_threshold=0.65)

    estimates: dict[str, dict] = {}
    vectors: dict[str, dict[str, np.ndarray]] = {}
    counts: dict[str, np.ndarray] = {}
    metric_rows: list[dict] = []
    per_ligand_rows: list[dict] = []
    identity_values = [str(value) for value in experiment.index]
    for representation_index, representation in enumerate(REPORTED_REPRESENTATIONS):
        estimates[representation] = {}
        vectors[representation] = {}
        for margin_index, (margin_name, margin) in enumerate(MARGINS.items()):
            summary, per_ligand, per_ligand_pairs = _pairwise_concordance(
                representations[representation],
                experimental_values,
                margin=margin,
                pair_stratum="all",
            )
            record = dict(summary)
            if include_uncertainty:
                record["uncertainty"] = _uncertainty(
                    per_ligand,
                    murcko,
                    butina,
                    bootstrap_repeats,
                    seed + 1000 * representation_index + 100 * margin_index,
                )
            estimates[representation][margin_name] = record
            vectors[representation][margin_name] = per_ligand
            counts.setdefault(margin_name, per_ligand_pairs)
            metric_rows.append(
                {
                    "representation": representation,
                    "experimental_margin_name": margin_name,
                    "experimental_margin_percentage_points": margin,
                    **summary,
                }
            )
            for ligand_index, identity in enumerate(identity_values):
                per_ligand_rows.append(
                    {
                        "evaluation_identity": identity,
                        "representation": representation,
                        "experimental_margin_name": margin_name,
                        "experimental_margin_percentage_points": margin,
                        "pairwise_concordance": _finite_float(
                            per_ligand[ligand_index]
                        ),
                        "evaluated_pairs": int(per_ligand_pairs[ligand_index]),
                    }
                )

    paired_comparisons: dict[str, dict] = {}
    paired_rows: list[dict] = []
    for contrast_index, (first, second) in enumerate(PAIRWISE_CONTRASTS):
        name = f"{first}_minus_{second}"
        paired_comparisons[name] = {}
        for margin_index, (margin_name, margin) in enumerate(MARGINS.items()):
            difference = vectors[first][margin_name] - vectors[second][margin_name]
            record = {
                "plugin_mean_difference": _finite_float(np.nanmean(difference))
            }
            if include_uncertainty:
                record["uncertainty"] = _uncertainty(
                    difference,
                    murcko,
                    butina,
                    bootstrap_repeats,
                    seed + 10000 + 1000 * contrast_index + 100 * margin_index,
                )
            paired_comparisons[name][margin_name] = record
            paired_rows.append(
                {
                    "contrast": name,
                    "experimental_margin_name": margin_name,
                    "experimental_margin_percentage_points": margin,
                    **record,
                }
            )

    binary: dict[str, dict] = {}
    binary_vectors: dict[str, dict[str, np.ndarray]] = {}
    binary_rows: list[dict] = []
    binary_per_ligand_rows: list[dict] = []
    for representation_index, representation in enumerate(REPORTED_REPRESENTATIONS):
        summary, auc, average_precision = _binary_activity_metrics(
            representations[representation], experimental_values
        )
        active_pair_summary, active_pair_vector, _ = _pairwise_concordance(
            representations[representation],
            experimental_values,
            margin=0.0,
            pair_stratum="active_vs_inactive",
        )
        auc_identity_difference = _finite_float(
            np.nanmax(np.abs(auc - active_pair_vector))
        )
        record = {
            **summary,
            "identity_check_auc_vs_active_inactive_pairwise_max_abs_difference": (
                auc_identity_difference
            ),
            "active_inactive_pairwise_summary": active_pair_summary,
        }
        if include_uncertainty:
            record["roc_auc_uncertainty"] = _uncertainty(
                auc,
                murcko,
                butina,
                bootstrap_repeats,
                seed + 30000 + 100 * representation_index,
            )
            record["average_precision_uncertainty"] = _uncertainty(
                average_precision,
                murcko,
                butina,
                bootstrap_repeats,
                seed + 30050 + 100 * representation_index,
            )
        binary[representation] = record
        binary_vectors[representation] = {
            "roc_auc": auc,
            "average_precision": average_precision,
        }
        binary_rows.append({"representation": representation, **record})
        for ligand_index, identity in enumerate(identity_values):
            binary_per_ligand_rows.append(
                {
                    "evaluation_identity": identity,
                    "representation": representation,
                    "roc_auc": _finite_float(auc[ligand_index]),
                    "average_precision": _finite_float(
                        average_precision[ligand_index]
                    ),
                }
            )

    paired_binary: dict[str, dict] = {}
    paired_binary_rows: list[dict] = []
    for contrast_index, (first, second) in enumerate(PAIRWISE_CONTRASTS):
        name = f"{first}_minus_{second}"
        paired_binary[name] = {}
        for metric_index, metric in enumerate(("roc_auc", "average_precision")):
            difference = (
                binary_vectors[first][metric] - binary_vectors[second][metric]
            )
            record = {
                "plugin_mean_difference": _finite_float(np.nanmean(difference))
            }
            if include_uncertainty:
                record["uncertainty"] = _uncertainty(
                    difference,
                    murcko,
                    butina,
                    bootstrap_repeats,
                    seed + 40000 + 1000 * contrast_index + 100 * metric_index,
                )
            paired_binary[name][metric] = record
            paired_binary_rows.append(
                {"contrast": name, "binary_metric": metric, **record}
            )

    def both_active_analysis(
        margin_name: str,
        *,
        representation_seed_offset: int,
        paired_seed_offset: int,
        status_prefix: str,
    ) -> tuple[dict[str, dict], dict[str, dict], list[dict], list[dict], str]:
        """Evaluate one explicitly labelled both-active margin contract."""
        margin = MARGINS[margin_name]
        status = (
            f"{status_prefix}: inclusion requires both PKIS2 target values to be "
            f"at least {ACTIVITY_THRESHOLD:g}% inhibition/displacement and their "
            f"absolute experimental difference to be greater than {margin:g} "
            "percentage points; intervals are unadjusted for multiplicity"
        )
        stratum: dict[str, dict] = {}
        rows: list[dict] = []
        vectors_for_margin: dict[str, np.ndarray] = {}
        for representation_index, representation in enumerate(
            REPORTED_REPRESENTATIONS
        ):
            summary, values, _ = _pairwise_concordance(
                representations[representation],
                experimental_values,
                margin=margin,
                pair_stratum="both_active",
            )
            record = {
                **summary,
                "pair_stratum": "both_active",
                "experimental_margin_name": margin_name,
                "experimental_margin_percentage_points": margin,
                "activity_threshold_percent": ACTIVITY_THRESHOLD,
                "status": status,
            }
            if include_uncertainty:
                record["uncertainty"] = _uncertainty(
                    values,
                    murcko,
                    butina,
                    bootstrap_repeats,
                    seed
                    + representation_seed_offset
                    + 100 * representation_index,
                )
            stratum[representation] = record
            vectors_for_margin[representation] = values
            rows.append({"representation": representation, **record})

        paired: dict[str, dict] = {}
        paired_rows_for_margin: list[dict] = []
        for contrast_index, (first, second) in enumerate(PAIRWISE_CONTRASTS):
            name = f"{first}_minus_{second}"
            difference = (
                vectors_for_margin[first] - vectors_for_margin[second]
            )
            record = {
                "plugin_mean_difference": _finite_float(np.nanmean(difference)),
                "pair_stratum": "both_active",
                "experimental_margin_name": margin_name,
                "experimental_margin_percentage_points": margin,
                "activity_threshold_percent": ACTIVITY_THRESHOLD,
                "evaluated_ligands": stratum[first]["evaluated_ligands"],
                "evaluated_pairs": stratum[first]["evaluated_pairs"],
                "status": status,
            }
            if include_uncertainty:
                record["uncertainty"] = _uncertainty(
                    difference,
                    murcko,
                    butina,
                    bootstrap_repeats,
                    seed + paired_seed_offset + 1000 * contrast_index,
                )
            paired[name] = record
            paired_rows_for_margin.append({"contrast": name, **record})
        return stratum, paired, rows, paired_rows_for_margin, status

    # The manuscript-facing both-active stratum uses the same strictly-greater-
    # than-10-percentage-point contract as the aggregate primary endpoint.
    (
        both_active,
        both_active_paired,
        both_active_rows,
        both_active_paired_rows,
        both_active_status,
    ) = both_active_analysis(
        PRIMARY_MARGIN,
        representation_seed_offset=70000,
        paired_seed_offset=80000,
        status_prefix="exploratory outcome-conditioned primary-margin analysis",
    )

    # Preserve the earlier exact-non-tie analysis as a clearly named secondary
    # sensitivity.  Its original seed offsets are retained for reproducibility.
    (
        exact_non_tie_both_active,
        exact_non_tie_both_active_paired,
        exact_non_tie_both_active_rows,
        exact_non_tie_both_active_paired_rows,
        exact_non_tie_both_active_status,
    ) = both_active_analysis(
        "exact_non_ties",
        representation_seed_offset=50000,
        paired_seed_offset=60000,
        status_prefix=(
            "secondary exploratory outcome-conditioned exact-non-tie sensitivity"
        ),
    )

    centered_summary, _, _ = _pairwise_concordance(
        representations["target_centered_unscaled"],
        experimental_values,
        margin=MARGINS[PRIMARY_MARGIN],
    )
    two_way_centered_summary, _, _ = _pairwise_concordance(
        representations["two_way_centered_unscaled"],
        experimental_values,
        margin=MARGINS[PRIMARY_MARGIN],
    )
    invariant_difference = float(
        centered_summary["mean_per_ligand_pairwise_concordance"]
        - two_way_centered_summary["mean_per_ligand_pairwise_concordance"]
    )
    if abs(invariant_difference) > 1e-12:
        raise AssertionError("Unscaled row centering changed within-ligand rankings")

    target_prior_primary = estimates["docking_target_prior"][PRIMARY_MARGIN]

    return (
        {
            "targets": targets,
            "fit": fit,
            "representations": estimates,
            "paired_comparisons": paired_comparisons,
            "docking_target_prior_baseline": {
                "definition": (
                    "ligand-independent target ordering obtained by assigning every "
                    "evaluation ligand the target means fitted on the leakage-free "
                    "DOCKSTRING reference pool"
                ),
                "primary_margin_name": PRIMARY_MARGIN,
                "primary_margin_percentage_points": MARGINS[PRIMARY_MARGIN],
                "mean_per_ligand_pairwise_concordance": target_prior_primary[
                    "mean_per_ligand_pairwise_concordance"
                ],
                "mean_within_ligand_spearman": target_prior_primary[
                    "mean_within_ligand_spearman"
                ],
                "top1_accuracy_allowing_experimental_ties": target_prior_primary[
                    "top1_accuracy_allowing_experimental_ties"
                ],
                "evaluated_ligands": target_prior_primary["evaluated_ligands"],
                "evaluated_pairs": target_prior_primary["evaluated_pairs"],
                "uncertainty": target_prior_primary.get("uncertainty"),
            },
            "operational_path_steps": {
                "status": (
                    "exploratory post-hoc decomposition; intervals are paired but "
                    "unadjusted for multiplicity, and the sequential steps are not "
                    "independent tests"
                ),
                "contrast_orientation": (
                    "paired mean within-ligand concordance of the first "
                    "representation minus the second"
                ),
                "primary_margin_name": PRIMARY_MARGIN,
                "primary_margin_percentage_points": MARGINS[PRIMARY_MARGIN],
                "contrast_keys": dict(OPERATIONAL_PATH_STEPS),
                "primary_results": {
                    step: paired_comparisons[contrast_key][PRIMARY_MARGIN]
                    for step, contrast_key in OPERATIONAL_PATH_STEPS.items()
                },
                "exploratory_both_active_results": {
                    step: both_active_paired[contrast_key]
                    for step, contrast_key in OPERATIONAL_PATH_STEPS.items()
                },
                "secondary_exact_non_tie_both_active_results": {
                    step: exact_non_tie_both_active_paired[contrast_key]
                    for step, contrast_key in OPERATIONAL_PATH_STEPS.items()
                },
            },
            "binary_activity_at_65_percent": binary,
            "paired_binary_comparisons": paired_binary,
            "exploratory_both_active_pairwise_concordance": both_active,
            "exploratory_both_active_paired_comparisons": both_active_paired,
            "both_active_status": both_active_status,
            "secondary_exact_non_tie_both_active_pairwise_concordance": (
                exact_non_tie_both_active
            ),
            "secondary_exact_non_tie_both_active_paired_comparisons": (
                exact_non_tie_both_active_paired
            ),
            "secondary_exact_non_tie_both_active_status": (
                exact_non_tie_both_active_status
            ),
            "chemical_clusters": {
                "murcko_clusters": int(len(pd.unique(murcko))),
                "butina_clusters": int(len(pd.unique(butina))),
                "murcko_definition": (
                    "RDKit Bemis-Murcko scaffolds; each acyclic compound is a singleton"
                ),
                "butina_definition": (
                    "RDKit Morgan fingerprints, radius 2, 2048 bits; Butina clustering "
                    "at Tanimoto similarity >= 0.65 (distance <= 0.35)"
                ),
            },
            "unscaled_row_centering_ranking_invariant": {
                "experimental_margin_percentage_points": MARGINS[PRIMARY_MARGIN],
                "target_centered_concordance": centered_summary[
                    "mean_per_ligand_pairwise_concordance"
                ],
                "two_way_centered_concordance": two_way_centered_summary[
                    "mean_per_ligand_pairwise_concordance"
                ],
                "difference": invariant_difference,
            },
        },
        {
            "primary_metrics": metric_rows,
            "paired_comparisons": paired_rows,
            "binary_activity_metrics": binary_rows,
            "binary_activity_per_ligand_metrics": binary_per_ligand_rows,
            "paired_binary_comparisons": paired_binary_rows,
            "exploratory_both_active": both_active_rows,
            "exploratory_both_active_paired_comparisons": (
                both_active_paired_rows
            ),
            "secondary_exact_non_tie_both_active": (
                exact_non_tie_both_active_rows
            ),
            "secondary_exact_non_tie_both_active_paired_comparisons": (
                exact_non_tie_both_active_paired_rows
            ),
            "primary_per_ligand_metrics": per_ligand_rows,
        },
    )


def _point_estimates(
    reference: pd.DataFrame,
    docking: pd.DataFrame,
    experiment: pd.DataFrame,
) -> dict:
    representations, _ = _score_representations(
        reference.to_numpy(dtype=float), docking.to_numpy(dtype=float)
    )
    values = experiment.to_numpy(dtype=float)
    result: dict[str, dict] = {}
    vectors: dict[str, dict[str, np.ndarray]] = {}
    for representation in REPORTED_REPRESENTATIONS:
        result[representation] = {"pairwise_concordance": {}}
        vectors[representation] = {}
        for margin_name, margin in MARGINS.items():
            summary, vector, _ = _pairwise_concordance(
                representations[representation], values, margin=margin
            )
            result[representation]["pairwise_concordance"][margin_name] = summary
            vectors[representation][margin_name] = vector
        binary, _, _ = _binary_activity_metrics(
            representations[representation], values
        )
        result[representation]["binary_activity_at_65_percent"] = binary
        both_active, _, _ = _pairwise_concordance(
            representations[representation],
            values,
            margin=MARGINS[PRIMARY_MARGIN],
            pair_stratum="both_active",
        )
        result[representation]["exploratory_both_active"] = both_active
        exact_non_tie_both_active, _, _ = _pairwise_concordance(
            representations[representation],
            values,
            margin=MARGINS["exact_non_ties"],
            pair_stratum="both_active",
        )
        result[representation][
            "secondary_exact_non_tie_both_active"
        ] = exact_non_tie_both_active
    result["paired_comparisons"] = {
        f"{first}_minus_{second}": {
            margin_name: _finite_float(
                np.nanmean(
                    vectors[first][margin_name] - vectors[second][margin_name]
                )
            )
            for margin_name in MARGINS
        }
        for first, second in PAIRWISE_CONTRASTS
    }
    return result


def _target_quality_table(experiment: pd.DataFrame) -> pd.DataFrame:
    records = []
    for target in experiment.columns:
        values = experiment[target].to_numpy(dtype=float)
        records.append(
            {
                "target": target,
                "pkis2_construct": TARGET_MAP[target],
                "n_ligands": int(len(values)),
                "minimum_percent_inhibition": float(values.min()),
                "median_percent_inhibition": float(np.median(values)),
                "mean_percent_inhibition": float(values.mean()),
                "maximum_percent_inhibition": float(values.max()),
                "standard_deviation": float(values.std(ddof=1)),
                "unique_values": int(len(np.unique(values))),
                "exact_zero_fraction": float(np.mean(values == 0.0)),
                "exact_100_fraction": float(np.mean(values == 100.0)),
                "active_at_65_count": int(np.sum(values >= ACTIVITY_THRESHOLD)),
                "active_at_65_fraction": float(
                    np.mean(values >= ACTIVITY_THRESHOLD)
                ),
            }
        )
    return pd.DataFrame(records)


def _pair_support_table(experiment: pd.DataFrame) -> pd.DataFrame:
    records = []
    for first_index, first in enumerate(experiment.columns[:-1]):
        for second in experiment.columns[first_index + 1 :]:
            first_values = experiment[first].to_numpy(dtype=float)
            second_values = experiment[second].to_numpy(dtype=float)
            difference = np.abs(first_values - second_values)
            first_active = first_values >= ACTIVITY_THRESHOLD
            second_active = second_values >= ACTIVITY_THRESHOLD
            record = {"target_1": first, "target_2": second}
            for margin_name, margin in MARGINS.items():
                record[f"informative_ligands__{margin_name}"] = int(
                    np.sum(difference > margin + 1e-12)
                )
            record["active_vs_inactive_ligands"] = int(
                np.sum(first_active ^ second_active)
            )
            record["both_active_unequal_ligands"] = int(
                np.sum(first_active & second_active & (difference > 1e-12))
            )
            records.append(record)
    return pd.DataFrame(records)


def _target_jackknife(
    reference: pd.DataFrame,
    docking: pd.DataFrame,
    experiment: pd.DataFrame,
) -> tuple[dict, list[dict]]:
    targets = list(experiment.columns)
    all_records: list[dict] = []

    def evaluate(
        kept_targets: list[str], deleted_target: str | None
    ) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, dict[str, np.ndarray]]]:
        representations, _ = _score_representations(
            reference[kept_targets].to_numpy(dtype=float),
            docking[kept_targets].to_numpy(dtype=float),
        )
        experimental = experiment[kept_targets].to_numpy(dtype=float)
        pair_vectors: dict[str, dict[str, np.ndarray]] = {}
        binary_vectors: dict[str, dict[str, np.ndarray]] = {}
        for representation in REPORTED_REPRESENTATIONS:
            pair_vectors[representation] = {}
            for margin_name, margin in MARGINS.items():
                summary, vector, _ = _pairwise_concordance(
                    representations[representation],
                    experimental,
                    margin=margin,
                )
                pair_vectors[representation][margin_name] = vector
                all_records.append(
                    {
                        "deleted_target": deleted_target or "NONE_FULL_PANEL",
                        "endpoint": "pairwise_concordance",
                        "experimental_margin_name": margin_name,
                        "representation": representation,
                        "estimate": summary[
                            "mean_per_ligand_pairwise_concordance"
                        ],
                        "evaluated_ligands": summary["evaluated_ligands"],
                        "evaluated_pairs": summary["evaluated_pairs"],
                    }
                )
            binary_summary, auc, average_precision = _binary_activity_metrics(
                representations[representation], experimental
            )
            binary_vectors[representation] = {
                "roc_auc": auc,
                "average_precision": average_precision,
            }
            for metric, field in (
                ("roc_auc", "mean_per_ligand_roc_auc"),
                ("average_precision", "mean_per_ligand_average_precision"),
            ):
                all_records.append(
                    {
                        "deleted_target": deleted_target or "NONE_FULL_PANEL",
                        "endpoint": metric,
                        "experimental_margin_name": None,
                        "representation": representation,
                        "estimate": binary_summary[field],
                        "evaluated_ligands": binary_summary[
                            "evaluated_ligands_with_both_classes"
                        ],
                        "evaluated_pairs": binary_summary[
                            "positive_negative_pairs"
                        ],
                    }
                )
        return pair_vectors, binary_vectors

    full_pair, full_binary = evaluate(targets, None)
    deleted: list[tuple[str, dict, dict]] = []
    for target in targets:
        keep = [value for value in targets if value != target]
        pair_vectors, binary_vectors = evaluate(keep, target)
        deleted.append((target, pair_vectors, binary_vectors))

    summaries: dict[str, dict] = {}
    for margin_name in MARGINS:
        full_difference = float(
            np.nanmean(
                full_pair["two_way_residual"][margin_name]
                - full_pair["absolute_vina"][margin_name]
            )
        )
        leave_one_out = np.asarray(
            [
                np.nanmean(
                    pair["two_way_residual"][margin_name]
                    - pair["absolute_vina"][margin_name]
                )
                for _, pair, _ in deleted
            ],
            dtype=float,
        )
        summaries[f"pairwise_concordance__{margin_name}"] = _jackknife_summary(
            full_difference, leave_one_out, targets
        )
    for metric in ("roc_auc", "average_precision"):
        full_difference = float(
            np.nanmean(
                full_binary["two_way_residual"][metric]
                - full_binary["absolute_vina"][metric]
            )
        )
        leave_one_out = np.asarray(
            [
                np.nanmean(
                    binary["two_way_residual"][metric]
                    - binary["absolute_vina"][metric]
                )
                for _, _, binary in deleted
            ],
            dtype=float,
        )
        summaries[metric] = _jackknife_summary(
            full_difference, leave_one_out, targets
        )
    return (
        {
            "two_way_residual_minus_absolute_vina": summaries,
            "interpretation": (
                "delete-one-target composition sensitivity; offsets, row means, "
                "target-specific raw scales, and residual scales are refit on each "
                "20-target panel. Normal intervals are descriptive because the 21 "
                "targets are a fixed, non-random panel"
            ),
        },
        all_records,
    )


def _jackknife_summary(
    full_estimate: float,
    leave_one_out: np.ndarray,
    targets_in_deletion_order: list[str],
) -> dict:
    values = np.asarray(leave_one_out, dtype=float)
    p = len(values)
    mean = float(values.mean())
    estimate = float(p * full_estimate - (p - 1) * mean)
    standard_error = float(
        np.sqrt((p - 1) / p * np.square(values - mean).sum())
    )
    influence = int(np.argmax(np.abs(values - full_estimate)))
    return {
        "full_panel_difference": float(full_estimate),
        "leave_one_target_out_minimum": float(values.min()),
        "leave_one_target_out_median": float(np.median(values)),
        "leave_one_target_out_maximum": float(values.max()),
        "jackknife_bias_corrected_difference": estimate,
        "jackknife_standard_error": standard_error,
        "jackknife_normal_95_interval": [
            estimate - 1.96 * standard_error,
            estimate + 1.96 * standard_error,
        ],
        "most_influential_deleted_target": targets_in_deletion_order[influence],
    }


def _mapping_table(
    dockstring: pd.DataFrame,
    pkis2: pd.DataFrame,
    primary_keys: set[str],
) -> list[dict]:
    targets = list(TARGET_MAP)
    records = []
    for key in sorted(primary_keys):
        pkis_rows = pkis2[pkis2.standard_inchikey.eq(key)]
        ds_rows = dockstring[dockstring.standard_inchikey.eq(key)]
        ds_score_range = (
            ds_rows[targets].max(axis=0) - ds_rows[targets].min(axis=0)
        ).max()
        experimental_range = (
            pkis_rows[targets].max(axis=0) - pkis_rows[targets].min(axis=0)
        ).max()
        records.append(
            {
                "standard_inchikey": key,
                "connectivity_block": key[:14],
                "pkis2_rows": int(len(pkis_rows)),
                "pkis2_excel_rows": ";".join(
                    str(value) for value in pkis_rows.source_excel_row.tolist()
                ),
                "pkis2_regno": ";".join(pkis_rows.Regno.astype(str).tolist()),
                "pkis2_compound": ";".join(pkis_rows.Compound.astype(str).tolist()),
                "pkis2_smiles": ";".join(pkis_rows.Smiles.astype(str).tolist()),
                "maximum_target_experimental_range_across_duplicate_rows": float(
                    experimental_range
                ),
                "dockstring_rows": int(len(ds_rows)),
                "dockstring_row_indices": ";".join(
                    str(value) for value in ds_rows.dockstring_row.tolist()
                ),
                "dockstring_reported_inchikeys": ";".join(
                    ds_rows.inchikey.astype(str).tolist()
                ),
                "maximum_target_score_range_across_duplicate_rows": float(
                    ds_score_range
                ),
            }
        )
    return records


def _size_matched_reference_sensitivity(
    dockstring: pd.DataFrame,
    evaluation_related: pd.Series,
    pkis2: pd.DataFrame,
    primary_docking: pd.DataFrame,
    primary_experiment: pd.DataFrame,
    *,
    neighbours_per_evaluation_ligand: tuple[int, ...] = (50, 100, 250),
) -> tuple[dict, list[dict]]:
    """Refit calibration on MW/heavy-atom nearest-reference supports.

    A neighbour can be selected for more than one evaluation ligand.  The rows
    with reuse are retained when target offsets and scales are fit, so each
    evaluation ligand contributes the same requested calibration mass.
    """
    descriptor_columns = ["molecular_weight", "heavy_atom_count"]
    neighbours = tuple(sorted(set(neighbours_per_evaluation_ligand)))
    if not neighbours or neighbours[0] < 1:
        raise ValueError("At least one positive neighbour count is required")
    pool = dockstring.loc[~evaluation_related].copy()
    if pool[descriptor_columns].isna().any().any():
        return (
            {
                "status": "not_run_in_nonexhaustive_identity_smoke_mode",
                "release_result_available": False,
            },
            [],
        )
    if neighbours[-1] > len(pool):
        raise ValueError("Requested more size-matched neighbours than reference rows")

    evaluation_descriptors = (
        pkis2[pkis2.standard_inchikey.isin(primary_docking.index)]
        .groupby("standard_inchikey", sort=True)[descriptor_columns]
        .median()
        .reindex(primary_docking.index)
    )
    if evaluation_descriptors.isna().any().any():
        raise ValueError("A primary PKIS2 ligand lacks size descriptors")
    pool_values = pool[descriptor_columns].to_numpy(dtype=float)
    evaluation_values = evaluation_descriptors.to_numpy(dtype=float)
    location = pool_values.mean(axis=0)
    scale = pool_values.std(axis=0, ddof=1)
    if (scale <= 0).any():
        raise ValueError("A reference matching descriptor has zero variance")
    pool_standardized = (pool_values - location) / scale
    evaluation_standardized = (evaluation_values - location) / scale
    nearest = NearestNeighbors(
        n_neighbors=neighbours[-1],
        algorithm="kd_tree",
        leaf_size=40,
        metric="euclidean",
        n_jobs=1,
    ).fit(pool_standardized)
    distances, positions = nearest.kneighbors(
        evaluation_standardized, return_distance=True
    )

    def standardized_mean_difference(values: np.ndarray) -> list[float]:
        return [
            float(value)
            for value in (
                (values.mean(axis=0) - evaluation_values.mean(axis=0)) / scale
            )
        ]

    records: list[dict] = []
    report_by_size: dict[str, dict] = {}
    for neighbour_count in neighbours:
        selected_positions = positions[:, :neighbour_count].reshape(-1)
        selected_distances = distances[:, :neighbour_count].reshape(-1)
        matched = pool.iloc[selected_positions]
        matched_reference = matched[list(TARGET_MAP)].clip(upper=0.0)
        point = _point_estimates(
            matched_reference,
            primary_docking.clip(upper=0.0),
            primary_experiment,
        )
        record = {
            "neighbours_per_evaluation_ligand": int(neighbour_count),
            "matched_reference_rows_with_reuse": int(len(matched)),
            "matched_reference_unique_rows": int(matched.dockstring_row.nunique()),
            "median_standardized_euclidean_neighbour_distance": float(
                np.median(selected_distances)
            ),
            "maximum_standardized_euclidean_neighbour_distance": float(
                selected_distances.max()
            ),
            "standardized_mean_difference_matched_reference_minus_evaluation": (
                standardized_mean_difference(
                    matched[descriptor_columns].to_numpy(dtype=float)
                )
            ),
            "representations": point,
        }
        report_by_size[str(neighbour_count)] = record
        records.append(record)
    return (
        {
            "status": "complete",
            "release_result_available": True,
            "matching_variables": descriptor_columns,
            "standardization_reference": (
                "complete DOCKSTRING pool after exclusion of every evaluation "
                "connectivity block"
            ),
            "sampling": (
                "exact KD-tree Euclidean nearest neighbours in reference-z-scored "
                "molecular-weight/heavy-atom space; reference rows may be reused "
                "across evaluation ligands"
            ),
            "reference_rows": int(len(pool)),
            "evaluation_ligands": int(len(evaluation_values)),
            "standardized_mean_difference_full_reference_minus_evaluation": (
                standardized_mean_difference(pool_values)
            ),
            "by_neighbour_count": report_by_size,
        },
        records,
    )


def dense_pkis2_dockstring_benchmark(
    dockstring_path: str | Path = DEFAULT_DOCKSTRING,
    pkis2_path: str | Path = DEFAULT_PKIS2,
    *,
    bootstrap_repeats: int = 5000,
    seed: int = 0,
    identity_scan: str = "full",
) -> dict:
    """Build the dense 21-target PKIS2 benchmark and return JSON-ready records."""
    if bootstrap_repeats < 1:
        raise ValueError("bootstrap_repeats must be positive")
    dockstring_path = Path(dockstring_path)
    pkis2_path = Path(pkis2_path)
    dockstring, pkis2 = _load_inputs(
        dockstring_path, pkis2_path, identity_scan
    )
    targets = list(TARGET_MAP)

    primary_keys = set(pkis2.standard_inchikey) & set(
        dockstring.standard_inchikey.dropna()
    )
    if not primary_keys:
        raise ValueError("No full-Standard-InChIKey PKIS2 overlap was found")
    primary_evaluation_blocks = set(
        pkis2.loc[
            pkis2.standard_inchikey.isin(primary_keys), "connectivity_block"
        ]
    )
    # Use one conservative, leakage-free reference for every identity
    # sensitivity.  This union protects graph/connectivity variants even if a
    # future source update creates a match outside the strict-key support.
    variant_evaluation_blocks: dict[str, set[str]] = {}
    evaluation_blocks: set[str] = set()
    for label, identity_column in MATCH_VARIANTS.items():
        common_variant = set(pkis2[identity_column].dropna()) & set(
            dockstring[identity_column].dropna()
        )
        blocks = set(
            pkis2.loc[
                pkis2[identity_column].isin(common_variant), "connectivity_block"
            ]
        )
        variant_evaluation_blocks[label] = blocks
        evaluation_blocks.update(blocks)
    evaluation_related = dockstring.connectivity_block.isin(evaluation_blocks)
    reference_unclipped = dockstring.loc[~evaluation_related, targets]
    reference_clipped = reference_unclipped.clip(upper=0.0)

    primary_docking, primary_experiment, primary_smiles = _matched_surface(
        dockstring,
        pkis2,
        "standard_inchikey",
        experimental_aggregation="median",
        docking_aggregation="median",
    )
    primary_docking_clipped = primary_docking.clip(upper=0.0)

    if identity_scan == "full":
        expected = {
            "pkis2_unique_standard_keys": 640,
            "primary_keys": 154,
            "primary_pkis2_rows": 155,
            "primary_dockstring_rows": 156,
            "evaluation_connectivity_blocks": 152,
            "excluded_dockstring_rows": 170,
        }
        observed = {
            "pkis2_unique_standard_keys": int(pkis2.standard_inchikey.nunique()),
            "primary_keys": int(len(primary_keys)),
            "primary_pkis2_rows": int(
                pkis2.standard_inchikey.isin(primary_keys).sum()
            ),
            "primary_dockstring_rows": int(
                dockstring.standard_inchikey.isin(primary_keys).sum()
            ),
            "evaluation_connectivity_blocks": int(len(evaluation_blocks)),
            "excluded_dockstring_rows": int(evaluation_related.sum()),
        }
        if observed != expected:
            raise AssertionError(
                f"Frozen PKIS2 identity audit changed: expected {expected}, "
                f"observed {observed}"
            )
        if primary_evaluation_blocks != evaluation_blocks:
            raise AssertionError(
                "An identity sensitivity adds connectivity blocks outside the "
                "strict full-key support; the conservative reference exclusion "
                "was applied, but frozen mapping expectations changed"
            )
    if primary_docking.shape != (154, 21) or primary_experiment.shape != (154, 21):
        raise AssertionError(
            "Expected a primary 154 x 21 surface; observed "
            f"{primary_docking.shape} and {primary_experiment.shape}"
        )

    primary, primary_tables = _panel_analysis(
        reference_clipped,
        primary_docking_clipped,
        primary_experiment,
        primary_smiles.smiles,
        bootstrap_repeats,
        seed,
        include_uncertainty=True,
    )
    primary_metric = primary["representations"]["absolute_vina"][PRIMARY_MARGIN]
    if (
        primary_metric["evaluated_ligands"] != 154
        or primary_metric["evaluated_pairs"] != 16417
    ):
        raise AssertionError(
            "Primary PKIS2 >10-point support changed: expected 154 ligands and "
            f"16,417 pairs, observed {primary_metric['evaluated_ligands']} and "
            f"{primary_metric['evaluated_pairs']}"
        )
    binary_metric = primary["binary_activity_at_65_percent"]["absolute_vina"]
    if (
        binary_metric["evaluated_ligands_with_both_classes"] != 123
        or binary_metric["positive_negative_pairs"] != 5426
    ):
        raise AssertionError("PKIS2 binary activity support changed")
    both_active_metric = primary["exploratory_both_active_pairwise_concordance"][
        "absolute_vina"
    ]
    exact_non_tie_both_active_metric = primary[
        "secondary_exact_non_tie_both_active_pairwise_concordance"
    ]["absolute_vina"]
    if (
        both_active_metric["evaluated_ligands"] != 57
        or both_active_metric["evaluated_pairs"] != 256
    ):
        raise AssertionError("PKIS2 primary-margin both-active support changed")
    if (
        exact_non_tie_both_active_metric["evaluated_ligands"] != 66
        or exact_non_tie_both_active_metric["evaluated_pairs"] != 591
    ):
        raise AssertionError("PKIS2 exact-non-tie both-active support changed")

    target_quality = _target_quality_table(primary_experiment)
    pair_support = _pair_support_table(primary_experiment)
    jackknife, jackknife_rows = _target_jackknife(
        reference_clipped,
        primary_docking_clipped,
        primary_experiment,
    )
    size_matched_calibration, size_matched_rows = (
        _size_matched_reference_sensitivity(
            dockstring,
            evaluation_related,
            pkis2,
            primary_docking,
            primary_experiment,
        )
    )

    identity_sensitivities: dict[str, dict] = {}
    identity_rows: list[dict] = []
    for label, identity_column in MATCH_VARIANTS.items():
        docking, experiment, _ = _matched_surface(
            dockstring,
            pkis2,
            identity_column,
            experimental_aggregation="median",
            docking_aggregation="median",
        )
        point = _point_estimates(
            reference_clipped,
            docking.clip(upper=0.0),
            experiment,
        )
        matched_pkis_rows = int(pkis2[identity_column].isin(docking.index).sum())
        matched_ds_rows = int(dockstring[identity_column].isin(docking.index).sum())
        matched_variant_blocks = variant_evaluation_blocks[label]
        if not matched_variant_blocks.issubset(evaluation_blocks):
            raise AssertionError(f"Reference leakage detected for {label}")
        record = {
            "match_variant": label,
            "identity_column": identity_column,
            "matched_ligands_after_identity_collapse": int(len(docking)),
            "matched_pkis2_rows_before_identity_collapse": matched_pkis_rows,
            "matched_dockstring_rows_before_identity_collapse": matched_ds_rows,
            "evaluation_connectivity_blocks": int(len(matched_variant_blocks)),
            "all_evaluation_blocks_excluded_from_reference": True,
            "representations": point,
        }
        identity_sensitivities[label] = record
        identity_rows.append(record)

    duplicate_sensitivities: dict[str, dict] = {}
    duplicate_rows: list[dict] = []
    duplicate_designs = [
        ("primary_median_both", "median", "median"),
        ("experimental_mean", "mean", "median"),
        ("experimental_first_source_row", "first", "median"),
        ("experimental_last_source_row", "last", "median"),
        ("dockstring_mean", "median", "mean"),
        ("dockstring_first_release_row", "median", "first"),
        ("dockstring_last_release_row", "median", "last"),
    ]
    for label, experimental_aggregation, docking_aggregation in duplicate_designs:
        docking, experiment, _ = _matched_surface(
            dockstring,
            pkis2,
            "standard_inchikey",
            experimental_aggregation=experimental_aggregation,
            docking_aggregation=docking_aggregation,
        )
        point = _point_estimates(
            reference_clipped,
            docking.clip(upper=0.0),
            experiment,
        )
        record = {
            "sensitivity": label,
            "experimental_duplicate_aggregation": experimental_aggregation,
            "dockstring_duplicate_aggregation": docking_aggregation,
            "representations": point,
        }
        duplicate_sensitivities[label] = record
        duplicate_rows.append(record)

    unclipped = _point_estimates(
        reference_unclipped,
        primary_docking,
        primary_experiment,
    )
    mapping_records = _mapping_table(dockstring, pkis2, primary_keys)

    experimental_values = primary_experiment.to_numpy(dtype=float)
    rounded_integer_fraction = float(
        np.mean(np.isclose(experimental_values, np.round(experimental_values)))
    )
    report = {
        "benchmark": (
            "dense_PKIS2_x_DOCKSTRING_21_target_observed_pair_concordance"
        ),
        "identity_scan": identity_scan,
        "release_eligible_identity_scan": bool(identity_scan == "full"),
        "analysis_parameters": {
            "seed": int(seed),
            "bootstrap_repeats": int(bootstrap_repeats),
            "identity_scan": identity_scan,
            "positive_score_handling": "clip values above zero to zero",
            "acyclic_cluster_rule": "recomputed Standard-InChI connectivity block",
        },
        "sources": {
            "pkis2_article_doi": "10.1371/journal.pone.0181585",
            "pkis2_article_url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC5540273/",
            "pkis2_s4_url": (
                "https://journals.plos.org/plosone/article/file?type=supplementary&"
                "id=info:doi/10.1371/journal.pone.0181585.s004"
            ),
            "pkis2_license": "Creative Commons Attribution (CC BY)",
            "pkis2_sheet": PKIS2_SHEET,
        },
        "source_support": {
            "pkis2_compound_rows": int(len(pkis2)),
            "pkis2_unique_recomputed_standard_inchikeys": int(
                pkis2.standard_inchikey.nunique()
            ),
            "pkis2_assay_columns_total": 406,
            "shared_targets": int(len(targets)),
            "shared_target_map": dict(TARGET_MAP),
            "dockstring_complete_rows": int(len(dockstring)),
            "evaluation_connectivity_blocks": int(len(evaluation_blocks)),
            "primary_full_key_connectivity_blocks": int(
                len(primary_evaluation_blocks)
            ),
            "dockstring_rows_excluded_from_reference": int(
                evaluation_related.sum()
            ),
            "reference_rows_after_evaluation_block_exclusion": int(
                len(reference_clipped)
            ),
            "primary_full_key_ligands": int(len(primary_docking)),
            "primary_full_key_pkis2_rows_before_median_collapse": int(
                pkis2.standard_inchikey.isin(primary_keys).sum()
            ),
            "primary_full_key_dockstring_rows_before_median_collapse": int(
                dockstring.standard_inchikey.isin(primary_keys).sum()
            ),
            "primary_experimental_cells": int(primary_experiment.size),
        },
        "assay_semantics": {
            "platform": "DiscoverX KINOMEscan competitive binding/displacement",
            "concentration": "1 micromolar",
            "measurement": "percentage inhibition/displacement",
            "preference_direction": "higher percentage indicates stronger displacement",
            "not_an_affinity_measurement": True,
            "exact_zero_cells": int(np.sum(experimental_values == 0.0)),
            "exact_100_cells": int(np.sum(experimental_values == 100.0)),
            "integer_valued_fraction": rounded_integer_fraction,
            "interpretation": (
                "zeros, 100s, and extensive integer rounding are treated as properties "
                "of a bounded single-concentration readout, not as thermodynamic censoring"
            ),
        },
        "score_handling": {
            "primary": (
                "positive DOCKSTRING scores are clipped to zero before fitting and "
                "evaluation, consistently with the primary DOCKSTRING surface"
            ),
            "sensitivity": (
                "the complete analysis is repeated without positive-score clipping"
            ),
        },
        "primary_estimand": (
            "equal-ligand mean within-ligand pairwise concordance on the dense 154 x 21 "
            "full-Standard-InChIKey support, restricted to experimental differences "
            "greater than 10 percentage points; higher PKIS2 percentage and lower Vina "
            "score indicate preference; score ties receive half credit"
        ),
        "primary_match": (
            "full Standard InChIKey recomputed with the pinned RDKit version from both "
            "released SMILES fields; duplicate PKIS2 rows and raw DOCKSTRING profiles "
            "are collapsed cell-wise by the median"
        ),
        "reference_fit_boundary": (
            "target offsets, raw target scales, residual target scales, and grand mean "
            "are fit on the complete DOCKSTRING 21-target surface after excluding all "
            "rows in the union of evaluation Standard-InChI connectivity blocks "
            "from all reported identity variants (152 blocks in this frozen release)"
        ),
        "primary": primary,
        "target_jackknife": jackknife,
        "molecular_size_matched_reference_calibration": size_matched_calibration,
        "molecular_identity_sensitivities": identity_sensitivities,
        "duplicate_measurement_and_profile_sensitivities": duplicate_sensitivities,
        "unclipped_score_sensitivity": unclipped,
        "boundary": (
            "This is observed-pair target-preference concordance with a dense, bounded, "
            "single-concentration KINOMEscan profile on 21 kinases already present in "
            "DOCKSTRING. It is neither an affinity benchmark nor retrieval of unmeasured "
            "targets. Target constructs and docking receptor structures are mapped but "
            "are not assumed to be biophysically identical"
        ),
        "tables": {
            **primary_tables,
            "target_quality": target_quality.to_dict(orient="records"),
            "target_pair_support": pair_support.to_dict(orient="records"),
            "target_jackknife": jackknife_rows,
            "molecular_size_matched_reference_calibration": size_matched_rows,
            "molecular_identity_sensitivities": identity_rows,
            "duplicate_sensitivities": duplicate_rows,
            "primary_molecule_mapping": mapping_records,
        },
    }
    json.dumps(report, allow_nan=False)
    return report


def write_outputs(report: dict, output_directory: str | Path) -> None:
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    (output / "dense_pkis2_benchmark.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    for name, records in report["tables"].items():
        pd.json_normalize(records, sep=".").to_csv(
            output / f"dense_pkis2_{name}.csv", index=False
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dockstring", type=Path, default=DEFAULT_DOCKSTRING)
    parser.add_argument("--pkis2", type=Path, default=DEFAULT_PKIS2)
    parser.add_argument("--output-dir", type=Path, default=PACKAGE / "results")
    parser.add_argument("--bootstrap-repeats", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--identity-scan",
        choices=["full", "reported_connectivity_prefilter"],
        default="full",
    )
    arguments = parser.parse_args()
    report = dense_pkis2_dockstring_benchmark(
        arguments.dockstring,
        arguments.pkis2,
        bootstrap_repeats=arguments.bootstrap_repeats,
        seed=arguments.seed,
        identity_scan=arguments.identity_scan,
    )
    write_outputs(report, arguments.output_dir)
    primary = report["primary"]["representations"]
    print(
        json.dumps(
            {
                "ligands": report["source_support"]["primary_full_key_ligands"],
                "targets": report["source_support"]["shared_targets"],
                "primary_margin_percentage_points": MARGINS[PRIMARY_MARGIN],
                "absolute_vina": primary["absolute_vina"][PRIMARY_MARGIN][
                    "mean_per_ligand_pairwise_concordance"
                ],
                "column_standardized": primary["column_standardized"][PRIMARY_MARGIN][
                    "mean_per_ligand_pairwise_concordance"
                ],
                "two_way_residual": primary["two_way_residual"][PRIMARY_MARGIN][
                    "mean_per_ligand_pairwise_concordance"
                ],
                "two_way_residual_minus_absolute_vina": report["primary"][
                    "paired_comparisons"
                ]["two_way_residual_minus_absolute_vina"][PRIMARY_MARGIN][
                    "plugin_mean_difference"
                ],
                "release_eligible_identity_scan": report[
                    "release_eligible_identity_scan"
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
