#!/usr/bin/env python3
"""Sensitivity of the operational ranking benchmark to its row-mean panel.

The primary operational benchmark evaluates 137 ligands on 38 Docking-44
targets.  Its released ``two_way_residual`` representation estimates docking
target offsets and residual scales on the external (evaluation-ligand-excluded)
Docking-44 reference, but computes each evaluation ligand's row mean over all
44 targets before extracting the 38 evaluated targets.

This script changes exactly one quantity: the evaluation-ligand row mean is
computed over the 38 evaluated targets instead of all 44 Docking-44 targets.
The evaluation ligands, observed experimental cells, target order, imputation
means, target offsets, grand mean, and residual target scales are held fixed.
Consequently, this is a one-factor sensitivity analysis, not a separately
refitted 38-target normalization.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:  # Support direct CLI execution and package-style imports.
    from . import build_evidence as evidence
except ImportError:  # pragma: no cover - direct CLI execution.
    import build_evidence as evidence  # type: ignore


PACKAGE = Path(__file__).resolve().parents[1]
FROZEN = PACKAGE / "data" / "frozen"
DEFAULT_OUTPUT = PACKAGE / "results" / "ranking_centering_panel_sensitivity"
DEFAULT_SEED = 20260806

CANONICAL_INPUT = FROZEN / "df_final_v4.csv.gz"
ACTIVITY_INPUT = (
    FROZEN / "negative_results_paper" / "analysis" / "chembl_activities_full.csv.gz"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finite_summary(values: np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if not len(array):
        raise ValueError("cannot summarize an empty finite array")
    return {
        "minimum": float(array.min()),
        "q025": float(np.quantile(array, 0.025)),
        "median": float(np.median(array)),
        "mean": float(array.mean()),
        "q975": float(np.quantile(array, 0.975)),
        "maximum": float(array.max()),
    }


def load_primary_support() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
]:
    """Reconstruct the frozen primary 137-by-38 support without live queries."""
    if not CANONICAL_INPUT.exists() or not ACTIVITY_INPUT.exists():
        missing = [
            str(path)
            for path in (CANONICAL_INPUT, ACTIVITY_INPUT)
            if not path.exists()
        ]
        raise FileNotFoundError(f"required frozen inputs are absent: {missing}")

    canonical = pd.read_csv(
        CANONICAL_INPUT,
        usecols=list(evidence.DOCK44)
        + ["Butina_clusters", "Canonical SMILES", "Cleaned SMILES"],
    )
    canonical["analysis_smiles"] = canonical["Cleaned SMILES"].fillna(
        canonical["Canonical SMILES"]
    )
    canonical["inchikey"] = evidence.full_inchikeys(canonical["analysis_smiles"])
    if canonical.inchikey.isna().any() or not canonical.inchikey.is_unique:
        raise ValueError("Docking-44 analysis SMILES do not yield unique full InChIKeys")

    docking_frame = canonical[list(evidence.DOCK44)].apply(pd.to_numeric, errors="coerce")
    docking_reference = pd.DataFrame(
        docking_frame.clip(upper=0).to_numpy(dtype=np.float64),
        index=pd.Index(canonical.inchikey, name="inchikey"),
        columns=evidence.DOCK44,
    )
    smiles = canonical.set_index("inchikey")["analysis_smiles"]
    butina = canonical.set_index("inchikey")["Butina_clusters"]

    required = [
        "canonical_smiles",
        "pchembl_value",
        "standard_relation",
        "standard_type",
        "panel_key",
    ]
    activities = pd.read_csv(ACTIVITY_INPUT, usecols=required)
    activities = activities[
        activities.panel_key.isin(evidence.DOCK44)
        & activities.pchembl_value.notna()
        & activities.standard_relation.eq("=")
        & activities.standard_type.isin(["Ki", "Kd", "IC50", "EC50"])
    ].copy()
    activities["inchikey"] = evidence.full_inchikeys(activities.canonical_smiles)
    activities = activities[activities.inchikey.notna()].copy()

    experimental_reference = activities.pivot_table(
        index="inchikey",
        columns="panel_key",
        values="pchembl_value",
        aggfunc="median",
    ).reindex(columns=evidence.DOCK44)
    matched_ids = experimental_reference.index.intersection(docking_reference.index)
    matched = experimental_reference.loc[matched_ids]
    evaluation_ids = matched.index[matched.notna().sum(axis=1) >= 2]
    experiment = matched.loc[evaluation_ids]
    evaluation_targets = experiment.columns[experiment.notna().any(axis=0)].tolist()
    experiment = experiment[evaluation_targets]

    if experiment.shape != (137, 38):
        raise ValueError(
            "frozen primary support changed: expected 137 ligands by 38 targets, "
            f"observed {experiment.shape}"
        )
    if int(experiment.notna().sum().sum()) != 691:
        raise ValueError("frozen primary support no longer contains 691 observed cells")
    return docking_reference, experimental_reference, experiment, smiles, butina


def externally_fixed_row_panel_scores(
    docking_reference: pd.DataFrame,
    evaluation_ids: pd.Index,
    evaluation_targets: list[str],
) -> tuple[dict[str, np.ndarray], dict[str, Any], pd.DataFrame]:
    """Return all-44 and local-target scores while holding fitted quantities fixed.

    All target-level quantities are estimated once, outside the evaluation
    ligand set, with the original all-44 training contract.  Only the row mean
    applied to an evaluation ligand changes between the two returned matrices.
    """
    evaluation_ids = pd.Index(evaluation_ids)
    selected = list(evaluation_targets)
    if not evaluation_ids.is_unique:
        raise ValueError("evaluation ligand identifiers must be unique")
    if len(selected) < 2 or len(set(selected)) != len(selected):
        raise ValueError("evaluation targets must be at least two unique columns")
    if not set(selected).issubset(docking_reference.columns):
        raise ValueError("evaluation targets are not a subset of the docking reference")

    train_raw = docking_reference.drop(index=evaluation_ids, errors="ignore")
    imputation_mean = train_raw.mean(axis=0)
    train = train_raw.fillna(imputation_mean)
    test_raw = docking_reference.loc[evaluation_ids]
    test = test_raw.fillna(imputation_mean)

    target_mean = train.mean(axis=0)
    grand_mean = float(train.to_numpy(dtype=np.float64).mean())
    train_row_mean_all = train.mean(axis=1)
    train_residual_all = (
        train
        - target_mean
        - train_row_mean_all.to_numpy(dtype=np.float64)[:, None]
        + grand_mean
    )
    residual_sd = train_residual_all.std(axis=0, ddof=1)
    if (residual_sd <= 0).any() or not np.isfinite(residual_sd).all():
        raise ValueError("external residual target scales must be finite and positive")

    row_mean_all = test.mean(axis=1)
    row_mean_local = test[selected].mean(axis=1)
    common = test[selected] - target_mean[selected] + grand_mean
    residual_all = common - row_mean_all.to_numpy(dtype=np.float64)[:, None]
    residual_local = common - row_mean_local.to_numpy(dtype=np.float64)[:, None]
    scale = residual_sd[selected]
    scores = {
        "all_44_row_mean": (residual_all / scale).to_numpy(dtype=np.float64),
        "local_evaluation_target_row_mean": (
            residual_local / scale
        ).to_numpy(dtype=np.float64),
    }

    row_table = pd.DataFrame(
        {
            "inchikey": evaluation_ids.astype(str),
            "row_mean_all_44": row_mean_all.to_numpy(dtype=np.float64),
            "row_mean_local_targets": row_mean_local.to_numpy(dtype=np.float64),
        }
    )
    row_table["local_minus_all_row_mean"] = (
        row_table.row_mean_local_targets - row_table.row_mean_all_44
    )

    metadata: dict[str, Any] = {
        "docking_training_ligands": int(len(train)),
        "evaluation_ligands": int(len(test)),
        "all_target_count": int(docking_reference.shape[1]),
        "local_target_count": int(len(selected)),
        "targets_excluded_from_local_row_mean": [
            str(target) for target in docking_reference.columns if target not in selected
        ],
        "docking_training_cells_imputed_from_external_target_means": int(
            train_raw.isna().sum().sum()
        ),
        "docking_evaluation_cells_imputed_from_external_target_means": int(
            test_raw.isna().sum().sum()
        ),
        "evaluation_imputed_cells_within_local_targets": int(
            test_raw[selected].isna().sum().sum()
        ),
        "evaluation_imputed_cells_in_six_context_targets": int(
            test_raw.drop(columns=selected).isna().sum().sum()
        ),
        "externally_fitted_quantities_held_fixed": [
            "per-target missing-score imputation means",
            "per-target docking offsets",
            "all-44 training grand mean",
            "per-target residual standard deviations fitted after all-44 training-row centering",
        ],
        "changed_quantity": (
            "evaluation-ligand row mean: all 44 Docking-44 targets versus the fixed "
            f"{len(selected)} evaluated targets"
        ),
        "not_a_refit": (
            "The local-target arm does not re-estimate target offsets, grand mean, or "
            "residual scales on a local-target training surface."
        ),
        "residual_scale_selected_targets": {
            "minimum": float(scale.min()),
            "median": float(scale.median()),
            "maximum": float(scale.max()),
        },
    }
    return scores, metadata, row_table


def _metric_without_vectors(metric: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in metric.items() if not key.startswith("per_ligand_")}


def _changed_pair_counts(
    first: np.ndarray,
    second: np.ndarray,
    experiment: np.ndarray,
) -> tuple[np.ndarray, int, int]:
    pair_arrays = evidence.preference_pair_arrays(experiment)
    ligand, target_a, target_b, truth = pair_arrays
    valid = truth != 0
    sign_first = np.sign(first[ligand, target_a] - first[ligand, target_b])
    sign_second = np.sign(second[ligand, target_a] - second[ligand, target_b])
    changed = valid & (sign_first != sign_second)
    per_ligand = np.bincount(
        ligand[changed], minlength=len(experiment)
    ).astype(int)
    return per_ligand, int(changed.sum()), int(valid.sum())


def evaluate(
    docking_reference: pd.DataFrame,
    experimental_reference: pd.DataFrame,
    experiment: pd.DataFrame,
    smiles: pd.Series,
    butina: pd.Series,
    *,
    bootstrap_repeats: int,
    seed: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Evaluate the isolated row-mean-panel perturbation."""
    if bootstrap_repeats < 1:
        raise ValueError("bootstrap repeats must be positive")
    evaluation_ids = experiment.index
    evaluation_targets = experiment.columns.tolist()
    scores, fit, rows = externally_fixed_row_panel_scores(
        docking_reference, evaluation_ids, evaluation_targets
    )

    released, released_fit = evidence.external_reference_score_representations(
        docking_reference,
        evaluation_ids,
        evaluation_targets,
        experimental_reference,
    )
    baseline_error = float(
        np.max(np.abs(scores["all_44_row_mean"] - released["two_way_residual"]))
    )
    if baseline_error > 1e-12:
        raise RuntimeError(
            "reconstructed all-44 arm does not reproduce the released contract: "
            f"maximum absolute error {baseline_error}"
        )

    experiment_values = experiment.to_numpy(dtype=np.float64)
    metrics = {
        name: evidence.preference_metrics(values, experiment_values)
        for name, values in scores.items()
    }
    fixed_context_metrics = {
        name: evidence.preference_metrics(released[name], experiment_values)
        for name in ("absolute_vina", "column_standardized")
    }
    murcko = evidence.scaffold_keys(smiles.reindex(evaluation_ids))
    butina_labels = butina.reindex(evaluation_ids).to_numpy()
    paired = evidence.paired_preference_comparison(
        metrics["local_evaluation_target_row_mean"]["per_ligand_pairwise_accuracy"],
        metrics["all_44_row_mean"]["per_ligand_pairwise_accuracy"],
        bootstrap_repeats,
        seed,
        murcko,
        butina_labels=butina_labels,
    )
    local_minus_absolute = evidence.paired_preference_comparison(
        metrics["local_evaluation_target_row_mean"]["per_ligand_pairwise_accuracy"],
        fixed_context_metrics["absolute_vina"]["per_ligand_pairwise_accuracy"],
        bootstrap_repeats,
        seed + 10,
        murcko,
        butina_labels=butina_labels,
    )
    local_minus_column = evidence.paired_preference_comparison(
        metrics["local_evaluation_target_row_mean"]["per_ligand_pairwise_accuracy"],
        fixed_context_metrics["column_standardized"]["per_ligand_pairwise_accuracy"],
        bootstrap_repeats,
        seed + 20,
        murcko,
        butina_labels=butina_labels,
    )

    changed_by_ligand, changed_pairs, eligible_pairs = _changed_pair_counts(
        scores["all_44_row_mean"],
        scores["local_evaluation_target_row_mean"],
        experiment_values,
    )
    rows["observed_targets"] = experiment.notna().sum(axis=1).to_numpy(dtype=int)
    rows["eligible_nontied_target_pairs"] = metrics["all_44_row_mean"][
        "per_ligand_pair_count"
    ]
    rows["changed_pair_predictions"] = changed_by_ligand
    rows["accuracy_all_44_row_mean"] = metrics["all_44_row_mean"][
        "per_ligand_pairwise_accuracy"
    ]
    rows["accuracy_local_target_row_mean"] = metrics[
        "local_evaluation_target_row_mean"
    ]["per_ligand_pairwise_accuracy"]
    rows["local_minus_all_accuracy"] = (
        rows.accuracy_local_target_row_mean - rows.accuracy_all_44_row_mean
    )
    rows["accuracy_absolute_vina"] = fixed_context_metrics["absolute_vina"][
        "per_ligand_pairwise_accuracy"
    ]
    rows["accuracy_column_standardized"] = fixed_context_metrics[
        "column_standardized"
    ]["per_ligand_pairwise_accuracy"]

    # The row-panel perturbation must be a row scalar before fixed target scaling.
    # Verify that multiplying the score difference by each target's scale gives
    # the same value across targets.  Recompute the scale from the external fit
    # explicitly to keep this identity independent of numerical division by a
    # potentially tiny row-mean difference.
    train_raw = docking_reference.drop(index=evaluation_ids, errors="ignore")
    train = train_raw.fillna(train_raw.mean(axis=0))
    target_mean = train.mean(axis=0)
    grand = float(train.to_numpy(dtype=np.float64).mean())
    train_residual = (
        train
        - target_mean
        - train.mean(axis=1).to_numpy(dtype=np.float64)[:, None]
        + grand
    )
    selected_scale = train_residual.std(axis=0, ddof=1)[evaluation_targets].to_numpy(
        dtype=np.float64
    )
    expected_score_delta = (
        -rows.local_minus_all_row_mean.to_numpy(dtype=np.float64)[:, None]
        / selected_scale[None, :]
    )
    observed_score_delta = (
        scores["local_evaluation_target_row_mean"] - scores["all_44_row_mean"]
    )
    score_delta_identity_error = float(
        np.max(np.abs(observed_score_delta - expected_score_delta))
    )
    if score_delta_identity_error > 1e-12:
        raise RuntimeError("row-mean score-difference identity failed")

    summary: dict[str, Any] = {
        "schema_version": "1.0.0",
        "analysis": (
            "operational 137-ligand/38-target ranking sensitivity to the evaluation-ligand "
            "row-mean target panel"
        ),
        "support": {
            "ligands": int(len(experiment)),
            "targets": int(experiment.shape[1]),
            "target_ids": evaluation_targets,
            "observed_experimental_cells": int(experiment.notna().sum().sum()),
            "eligible_nontied_within_ligand_target_pairs": eligible_pairs,
            "experimental_cell_contract": (
                "exact-relation Ki, Kd, IC50 or EC50; median pChEMBL per full-InChIKey "
                "and Docking-44 target cell; at least two observed targets per ligand"
            ),
        },
        "fit_contract": fit,
        "contract_audit": {
            "maximum_absolute_difference_from_released_all_44_two_way_residual": baseline_error,
            "maximum_absolute_error_in_fixed_scale_score_delta_identity": (
                score_delta_identity_error
            ),
            "released_fit_support_matches": {
                "docking_training_ligands": bool(
                    released_fit["docking_training_ligands"]
                    == fit["docking_training_ligands"]
                ),
                "docking_training_imputed_cells": bool(
                    released_fit[
                        "docking_training_cells_imputed_from_external_target_means"
                    ]
                    == fit[
                        "docking_training_cells_imputed_from_external_target_means"
                    ]
                ),
                "docking_evaluation_imputed_cells": bool(
                    released_fit[
                        "docking_evaluation_cells_imputed_from_external_target_means"
                    ]
                    == fit[
                        "docking_evaluation_cells_imputed_from_external_target_means"
                    ]
                ),
            },
        },
        "row_mean_change": {
            "all_44": finite_summary(rows.row_mean_all_44.to_numpy()),
            "local_38": finite_summary(rows.row_mean_local_targets.to_numpy()),
            "local_minus_all": finite_summary(rows.local_minus_all_row_mean.to_numpy()),
            "pearson_correlation": float(
                np.corrcoef(rows.row_mean_all_44, rows.row_mean_local_targets)[0, 1]
            ),
        },
        "ranking": {
            "all_44_row_mean": _metric_without_vectors(metrics["all_44_row_mean"]),
            "local_38_row_mean": _metric_without_vectors(
                metrics["local_evaluation_target_row_mean"]
            ),
            "local_38_minus_all_44": paired,
            "fixed_context_representations": {
                "absolute_vina": _metric_without_vectors(
                    fixed_context_metrics["absolute_vina"]
                ),
                "column_standardized": _metric_without_vectors(
                    fixed_context_metrics["column_standardized"]
                ),
            },
            "local_38_minus_absolute_vina": local_minus_absolute,
            "local_38_minus_column_standardized": local_minus_column,
            "changed_pair_predictions": changed_pairs,
            "fraction_of_eligible_pair_predictions_changed": float(
                changed_pairs / eligible_pairs
            ),
            "ligands_with_any_changed_pair_prediction": int(
                (changed_by_ligand > 0).sum()
            ),
        },
        "interpretation_boundary": (
            "This one-factor sensitivity measures dependence on six additional docking "
            "scores used only to estimate each evaluation ligand's row mean. It holds the "
            "38-target experimental support and all externally fitted target quantities "
            "fixed; it is not evidence about a newly fitted 38-target normalization or "
            "about unmeasured-target retrieval."
        ),
        "bootstrap": {
            "repeats": int(bootstrap_repeats),
            "seed": int(seed),
            "units": "ligands, cyclic Bemis-Murcko clusters, and frozen Butina clusters",
        },
    }
    return summary, rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstrap-repeats", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    docking, experimental, experiment, smiles, butina = load_primary_support()
    summary, ligand_table = evaluate(
        docking,
        experimental,
        experiment,
        smiles,
        butina,
        bootstrap_repeats=args.bootstrap_repeats,
        seed=args.seed,
    )
    summary["frozen_inputs"] = {
        str(CANONICAL_INPUT.relative_to(PACKAGE)): sha256(CANONICAL_INPUT),
        str(ACTIVITY_INPUT.relative_to(PACKAGE)): sha256(ACTIVITY_INPUT),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    ligand_table.to_csv(args.output / "ligand_sensitivity.csv", index=False)
    print(json.dumps(summary["ranking"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
