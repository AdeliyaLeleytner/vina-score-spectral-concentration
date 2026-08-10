#!/usr/bin/env python3
"""Sensitivity of the central descriptor-removal correlation-ratio contrast.

The headline 46.9% and 53.6% values are point estimates of an operational estimand:

    1 - mean(r_error^2) / mean(r_surface^2),

where both means run over off-diagonal target correlations and ``error`` is the residual
after subtracting group-held-out descriptor predictions.  This is *not* an additive share
of correlation explained.  The original analysis uses one deterministic GroupKFold
partition and, for DOCKSTRING, one score-blind random 15,000-ligand support.

This script quantifies two reproducibility axes without changing the estimator:

1. repeated, ligand-count-balanced random assignments of intact chemical groups to five
   folds, with every target offset, residual target scale, descriptor standardization and
   target-specific regression coefficient refitted inside every fold;
2. multiple score-blind simple-random 15,000-ligand DOCKSTRING supports, selected by row
   index without inspecting scores, each evaluated under the deterministic GroupKFold and
   the repeated partitions above.

The resulting percentile ranges are empirical design-sensitivity intervals.  They are not
confidence intervals for chemical space or target populations and do not include
experimental data, because this estimand is defined entirely on docking-score surfaces.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

try:  # Support direct execution and package imports.
    from . import residual_mechanism_analysis as mechanism
except ImportError:  # pragma: no cover
    import residual_mechanism_analysis as mechanism  # type: ignore


PACKAGE = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PACKAGE / "results" / "descriptor_correlation_reduction_uncertainty"
DEFAULT_FOLDS = 5
DEFAULT_REPEATED_PARTITIONS = 12
DEFAULT_PARTITION_SEED = 20260831
DEFAULT_DOCKSTRING_SUPPORT_SIZE = 15_000
DEFAULT_DOCKSTRING_SUPPORT_SEEDS = (71, 1701, 2701, 3701, 4701)
DEFAULT_BOOTSTRAP_REPEATS = 5_000
DEFAULT_BOOTSTRAP_BLOCKS = 512
DEFAULT_BOOTSTRAP_SEED = 20260901
BOOTSTRAP_CHUNK = 100


def balanced_random_group_folds(
    groups: np.ndarray, folds: int, seed: int
) -> tuple[np.ndarray, dict[str, object]]:
    """Assign intact groups to approximately ligand-balanced random folds."""
    _, inverse, sizes = np.unique(
        np.asarray(groups, dtype=object), return_inverse=True, return_counts=True
    )
    if len(sizes) < folds:
        raise ValueError("fewer groups than folds")
    rng = np.random.default_rng(seed)
    # Random order supplies the repeated-partition perturbation.  Greedy placement keeps
    # fold sizes close without using any score or descriptor value.
    order = rng.permutation(len(sizes))
    loads = np.zeros(folds, dtype=np.int64)
    assignment = np.empty(len(sizes), dtype=np.int16)
    for group in order:
        destination = int(np.argmin(loads))
        assignment[group] = destination
        loads[destination] += sizes[group]
    labels = assignment[inverse]
    return labels, {
        "seed": int(seed),
        "groups": int(len(sizes)),
        "minimum_fold_ligands": int(loads.min()),
        "median_fold_ligands": float(np.median(loads)),
        "maximum_fold_ligands": int(loads.max()),
    }


def _metrics(observed: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    error = observed - prediction
    before = mechanism.correlation_moments(observed)
    after = mechanism.correlation_moments(error)
    before_r2 = float(before["mean_squared_offdiagonal_correlation"])
    after_r2 = float(after["mean_squared_offdiagonal_correlation"])
    target_sst = np.square(observed - observed.mean(axis=0)).sum(axis=0)
    target_sse = np.square(error).sum(axis=0)
    target_r2 = 1.0 - target_sse / target_sst
    return {
        "mean_squared_offdiagonal_target_correlation_before": before_r2,
        "mean_squared_offdiagonal_target_correlation_after": after_r2,
        "absolute_mean_squared_offdiagonal_target_correlation_reduction": (
            before_r2 - after_r2
        ),
        "relative_mean_squared_offdiagonal_target_correlation_reduction": (
            (before_r2 - after_r2) / before_r2
        ),
        "mean_out_of_fold_target_r2": float(target_r2.mean()),
        "median_out_of_fold_target_r2": float(np.median(target_r2)),
    }


def fit_with_fold_labels(
    matrix: np.ndarray,
    descriptors: np.ndarray,
    groups: np.ndarray,
    fold_labels: np.ndarray,
) -> dict[str, float]:
    """Same fold-local estimator as grouped_descriptor_decomposition, explicit folds."""
    matrix = np.asarray(matrix, dtype=np.float64)
    descriptors = np.asarray(descriptors, dtype=np.float64)
    groups = np.asarray(groups, dtype=object)
    fold_labels = np.asarray(fold_labels, dtype=np.int16)
    if not (len(matrix) == len(descriptors) == len(groups) == len(fold_labels)):
        raise ValueError("all row-aligned inputs must have the same length")
    observed = np.full_like(matrix, np.nan)
    prediction = np.full_like(matrix, np.nan)
    unique_folds = np.unique(fold_labels)
    for fold in unique_folds:
        test = np.flatnonzero(fold_labels == fold)
        train = np.flatnonzero(fold_labels != fold)
        if set(groups[train].tolist()) & set(groups[test].tolist()):
            raise AssertionError("chemical group leakage")
        train_surface, test_surface, _ = mechanism.fold_local_residual_transform(
            matrix[train], matrix[test]
        )
        mean = descriptors[train].mean(axis=0)
        sd = descriptors[train].std(axis=0, ddof=1)
        if np.any(sd <= 1e-12):
            raise ValueError("constant descriptor in a training fold")
        train_design = np.column_stack(
            [np.ones(len(train)), (descriptors[train] - mean) / sd]
        )
        test_design = np.column_stack(
            [np.ones(len(test)), (descriptors[test] - mean) / sd]
        )
        coefficients = np.linalg.lstsq(train_design, train_surface, rcond=None)[0]
        observed[test] = test_surface
        prediction[test] = test_design @ coefficients
    if not np.isfinite(observed).all() or not np.isfinite(prediction).all():
        raise AssertionError("explicit-fold OOF surfaces were not filled")
    return _metrics(observed, prediction)


def deterministic_groupkfold_fit(
    matrix: np.ndarray, descriptors: np.ndarray, groups: np.ndarray, folds: int
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    decomposition = mechanism.grouped_descriptor_decomposition(
        matrix, descriptors, groups, folds
    )
    metrics = decomposition["metrics"]
    record = {
        "mean_squared_offdiagonal_target_correlation_before": metrics[
            "out_of_fold_mean_squared_target_correlation_before_descriptor_removal"
        ],
        "mean_squared_offdiagonal_target_correlation_after": metrics[
            "out_of_fold_mean_squared_target_correlation_after_descriptor_removal"
        ],
        "absolute_mean_squared_offdiagonal_target_correlation_reduction": metrics[
            "out_of_fold_absolute_mean_squared_target_correlation_reduction"
        ],
        "relative_mean_squared_offdiagonal_target_correlation_reduction": metrics[
            "out_of_fold_relative_mean_squared_target_correlation_reduction"
        ],
        "mean_out_of_fold_target_r2": metrics["mean_out_of_fold_target_r2"],
        "median_out_of_fold_target_r2": metrics["median_out_of_fold_target_r2"],
    }
    return record, decomposition["oof_surface"], decomposition["oof_prediction"]


def repeated_partition_rows(
    dataset: str,
    matrix: np.ndarray,
    descriptors: np.ndarray,
    groups: np.ndarray,
    *,
    folds: int,
    repeats: int,
    base_seed: int,
    support_seed: int | None,
) -> tuple[list[dict[str, object]], np.ndarray, np.ndarray]:
    rows: list[dict[str, object]] = []
    reference, observed, prediction = deterministic_groupkfold_fit(
        matrix, descriptors, groups, folds
    )
    rows.append(
        {
            "dataset": dataset,
            "support_seed": support_seed,
            "partition": "deterministic_GroupKFold",
            "partition_seed": None,
            "ligands": int(len(matrix)),
            "targets": int(matrix.shape[1]),
            "chemical_groups": int(len(np.unique(groups))),
            **reference,
        }
    )
    for repetition in range(repeats):
        seed = base_seed + repetition
        labels, balance = balanced_random_group_folds(groups, folds, seed)
        metrics = fit_with_fold_labels(matrix, descriptors, groups, labels)
        rows.append(
            {
                "dataset": dataset,
                "support_seed": support_seed,
                "partition": f"balanced_random_group_partition_{repetition + 1}",
                "partition_seed": seed,
                "ligands": int(len(matrix)),
                "targets": int(matrix.shape[1]),
                "chemical_groups": int(len(np.unique(groups))),
                **balance,
                **metrics,
            }
        )
    return rows, observed, prediction


def balanced_group_blocks(
    groups: np.ndarray, requested_blocks: int, seed: int
) -> tuple[np.ndarray, dict[str, object]]:
    """Pack intact chemical groups into reproducible ligand-balanced blocks."""
    _, inverse, sizes = np.unique(
        np.asarray(groups, dtype=object), return_inverse=True, return_counts=True
    )
    blocks = min(int(requested_blocks), len(sizes))
    rng = np.random.default_rng(seed)
    jitter = rng.random(len(sizes))
    order = np.lexsort((jitter, -sizes))
    loads = np.zeros(blocks, dtype=np.int64)
    assignment = np.empty(len(sizes), dtype=np.int32)
    for group in order:
        destination = int(np.argmin(loads))
        assignment[group] = destination
        loads[destination] += sizes[group]
    return assignment[inverse], {
        "chemical_groups": int(len(sizes)),
        "blocks": int(blocks),
        "minimum_block_ligands": int(loads.min()),
        "median_block_ligands": float(np.median(loads)),
        "maximum_block_ligands": int(loads.max()),
        "block_assignment_seed": int(seed),
    }


def block_raw_moments(
    surface: np.ndarray, row_blocks: np.ndarray, blocks: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Raw first and upper-triangle second moments for weighted correlations."""
    triangle = np.triu_indices(surface.shape[1])
    products = np.column_stack(
        [
            np.bincount(
                row_blocks,
                weights=surface[:, first] * surface[:, second],
                minlength=blocks,
            )
            for first, second in zip(*triangle)
        ]
    )
    sums = np.column_stack(
        [
            np.bincount(row_blocks, weights=surface[:, column], minlength=blocks)
            for column in range(surface.shape[1])
        ]
    )
    counts = np.bincount(row_blocks, minlength=blocks).astype(np.float64)
    return products, sums, counts


def weighted_mean_squared_offdiagonal_correlation(
    weighted_products: np.ndarray,
    weighted_sums: np.ndarray,
    weighted_count: float,
    targets: int,
) -> float:
    """Mean squared off-diagonal correlation reconstructed from raw moments."""
    triangle = np.triu_indices(targets)
    cross = np.zeros((targets, targets), dtype=np.float64)
    cross[triangle] = weighted_products
    cross[(triangle[1], triangle[0])] = weighted_products
    covariance = cross - np.outer(weighted_sums, weighted_sums) / weighted_count
    variances = np.diag(covariance)
    if np.any(variances <= 0):
        raise ValueError("non-positive weighted target variance")
    correlation = covariance / np.sqrt(np.outer(variances, variances))
    offdiagonal = correlation[np.triu_indices(targets, k=1)]
    return float(np.mean(np.square(offdiagonal)))


def paired_fixed_oof_block_bootstrap(
    observed: np.ndarray,
    prediction: np.ndarray,
    groups: np.ndarray,
    *,
    repeats: int,
    requested_blocks: int,
    assignment_seed: int,
    draw_seed: int,
) -> tuple[dict[str, object], np.ndarray]:
    """Paired conditional bootstrap of the exact operational reduction estimand.

    One Exp(1) weight vector is shared by the observed and error surfaces in every draw.
    Learned OOF predictions are fixed, so this propagates conditional ligand-support
    reweighting but not refit uncertainty.
    """
    row_blocks, block_report = balanced_group_blocks(
        groups, requested_blocks, assignment_seed
    )
    blocks = int(block_report["blocks"])
    observed_moments = block_raw_moments(observed, row_blocks, blocks)
    error_moments = block_raw_moments(observed - prediction, row_blocks, blocks)
    draws = np.empty(repeats, dtype=np.float64)
    rng = np.random.default_rng(draw_seed)
    position = 0
    while position < repeats:
        size = min(BOOTSTRAP_CHUNK, repeats - position)
        weights = rng.exponential(size=(size, blocks))
        observed_products = weights @ observed_moments[0]
        observed_sums = weights @ observed_moments[1]
        observed_counts = weights @ observed_moments[2]
        error_products = weights @ error_moments[0]
        error_sums = weights @ error_moments[1]
        error_counts = weights @ error_moments[2]
        for offset in range(size):
            before = weighted_mean_squared_offdiagonal_correlation(
                observed_products[offset],
                observed_sums[offset],
                float(observed_counts[offset]),
                observed.shape[1],
            )
            after = weighted_mean_squared_offdiagonal_correlation(
                error_products[offset],
                error_sums[offset],
                float(error_counts[offset]),
                observed.shape[1],
            )
            draws[position + offset] = 1.0 - after / before
        position += size
    point = _metrics(observed, prediction)[
        "relative_mean_squared_offdiagonal_target_correlation_reduction"
    ]
    described = describe(draws)
    return {
        **block_report,
        "repeats": int(repeats),
        "draw_seed": int(draw_seed),
        "point": float(point),
        "bootstrap_mean": described["mean"],
        "bootstrap_median": described["median"],
        "conditional_coarsened_sensitivity_interval_95": described[
            "empirical_interval_95"
        ],
        "minimum": described["minimum"],
        "maximum": described["maximum"],
    }, draws


def load_dockstring_complete_pool() -> tuple[np.ndarray, pd.Series, list[str], dict[str, int]]:
    path = mechanism.FROZEN / "dockstring-dataset.tsv.gz"
    frame = pd.read_csv(path, sep="\t")
    targets = [column for column in frame.columns if column not in {"inchikey", "smiles"}]
    numeric = frame[targets].apply(pd.to_numeric, errors="coerce")
    complete = ~numeric.isna().any(axis=1)
    complete_matrix = numeric.loc[complete].to_numpy(dtype=np.float64)
    positive = int((complete_matrix > 0).sum())
    matrix = np.minimum(complete_matrix, 0.0)
    smiles = frame.loc[complete, "smiles"].reset_index(drop=True).astype(str)
    return matrix, smiles, targets, {
        "source_rows": int(len(frame)),
        "complete_rows": int(len(matrix)),
        "missing_cells": int(numeric.isna().to_numpy().sum()),
        "positive_cells_clipped": positive,
    }


def describe(values: np.ndarray) -> dict[str, float | int | list[float]]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "n": int(len(values)),
        "mean": float(values.mean()),
        "standard_deviation": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        "minimum": float(values.min()),
        "median": float(np.median(values)),
        "maximum": float(values.max()),
        "empirical_interval_95": [
            float(np.quantile(values, 0.025)),
            float(np.quantile(values, 0.975)),
        ],
    }


def summarize_rows(frame: pd.DataFrame) -> tuple[dict[str, object], pd.DataFrame]:
    metric = "relative_mean_squared_offdiagonal_target_correlation_reduction"
    summary: dict[str, object] = OrderedDict()
    support_rows: list[dict[str, object]] = []
    for dataset in ("Docking-44", "DOCKSTRING-58"):
        selected = frame.loc[frame.dataset.eq(dataset)]
        deterministic = selected.loc[selected.partition.eq("deterministic_GroupKFold")]
        repeated = selected.loc[~selected.partition.eq("deterministic_GroupKFold")]
        record: dict[str, object] = {
            "deterministic_groupkfold": describe(deterministic[metric].to_numpy()),
            "all_repeated_group_partitions": describe(repeated[metric].to_numpy()),
        }
        if dataset == "DOCKSTRING-58":
            means = []
            for support_seed, group in selected.groupby("support_seed", sort=True):
                deterministic_value = float(
                    group.loc[group.partition.eq("deterministic_GroupKFold"), metric].iloc[0]
                )
                repeated_values = group.loc[
                    ~group.partition.eq("deterministic_GroupKFold"), metric
                ].to_numpy()
                row = {
                    "support_seed": int(support_seed),
                    "deterministic_groupkfold": deterministic_value,
                    "repeated_partition_mean": float(repeated_values.mean()),
                    "repeated_partition_minimum": float(repeated_values.min()),
                    "repeated_partition_maximum": float(repeated_values.max()),
                    "repeated_partition_sd": float(repeated_values.std(ddof=1)),
                }
                support_rows.append(row)
                means.append(row["repeated_partition_mean"])
            record["between_score_blind_support_means"] = describe(np.asarray(means))
        summary[dataset] = record
    return summary, pd.DataFrame.from_records(support_rows)


def run(
    *,
    folds: int,
    repeated_partitions: int,
    partition_seed: int,
    dockstring_support_size: int,
    dockstring_support_seeds: tuple[int, ...],
    bootstrap_repeats: int,
    bootstrap_blocks: int,
    bootstrap_seed: int,
) -> tuple[dict[str, object], dict[str, pd.DataFrame]]:
    rows: list[dict[str, object]] = []
    bootstrap_inputs: list[
        tuple[str, int | None, np.ndarray, np.ndarray, np.ndarray]
    ] = []
    docking = mechanism.load_docking44()
    docking_rows, docking_observed, docking_prediction = repeated_partition_rows(
        docking.name,
        docking.matrix,
        docking.descriptor_matrix,
        docking.groups,
        folds=folds,
        repeats=repeated_partitions,
        base_seed=partition_seed,
        support_seed=None,
    )
    rows.extend(docking_rows)
    bootstrap_inputs.append(
        (
            docking.name,
            None,
            docking_observed,
            docking_prediction,
            docking.groups,
        )
    )

    pool, smiles, targets, pool_report = load_dockstring_complete_pool()
    if dockstring_support_size > len(pool):
        raise ValueError("DOCKSTRING support size exceeds complete pool")
    support_indices: dict[int, np.ndarray] = {}
    for support_position, support_seed in enumerate(dockstring_support_seeds):
        rng = np.random.default_rng(support_seed)
        indices = np.sort(
            rng.choice(len(pool), dockstring_support_size, replace=False)
        )
        support_indices[support_seed] = indices
        support_smiles = smiles.iloc[indices].reset_index(drop=True)
        descriptors = mechanism.molecular_descriptors(support_smiles)
        groups = mechanism.scaffold_keys(support_smiles)
        support_rows, support_observed, support_prediction = repeated_partition_rows(
            "DOCKSTRING-58",
            pool[indices],
            descriptors,
            groups,
            folds=folds,
            repeats=repeated_partitions,
            base_seed=partition_seed + (support_position + 1) * 100_000,
            support_seed=support_seed,
        )
        rows.extend(support_rows)
        bootstrap_inputs.append(
            (
                "DOCKSTRING-58",
                support_seed,
                support_observed,
                support_prediction,
                groups,
            )
        )

    frame = pd.DataFrame.from_records(rows)
    sensitivity, support_frame = summarize_rows(frame)
    bootstrap_summary: dict[str, object] = OrderedDict()
    bootstrap_rows: list[dict[str, object]] = []
    for position, (dataset, support_seed, observed, prediction, groups) in enumerate(
        bootstrap_inputs
    ):
        key = dataset if support_seed is None else f"{dataset}_seed_{support_seed}"
        record, _ = paired_fixed_oof_block_bootstrap(
            observed,
            prediction,
            groups,
            repeats=bootstrap_repeats,
            requested_blocks=bootstrap_blocks,
            assignment_seed=bootstrap_seed + position * 1_000_000,
            draw_seed=bootstrap_seed + position * 1_000_000 + 100_000,
        )
        bootstrap_summary[key] = record
        bootstrap_rows.append(
            {
                "dataset": dataset,
                "support_seed": support_seed,
                **record,
                "conditional_coarsened_sensitivity_low95": record[
                    "conditional_coarsened_sensitivity_interval_95"
                ][0],
                "conditional_coarsened_sensitivity_high95": record[
                    "conditional_coarsened_sensitivity_interval_95"
                ][1],
            }
        )
    overlaps = []
    for first_index, first_seed in enumerate(dockstring_support_seeds):
        for second_seed in dockstring_support_seeds[first_index + 1 :]:
            overlap = len(
                np.intersect1d(
                    support_indices[first_seed], support_indices[second_seed], assume_unique=True
                )
            )
            overlaps.append(
                {
                    "first_support_seed": first_seed,
                    "second_support_seed": second_seed,
                    "shared_ligands": overlap,
                    "jaccard": overlap
                    / (
                        2 * dockstring_support_size - overlap
                    ),
                }
            )

    original_docking = frame.loc[
        frame.dataset.eq("Docking-44")
        & frame.partition.eq("deterministic_GroupKFold")
    ].iloc[0]
    original_dockstring = frame.loc[
        frame.dataset.eq("DOCKSTRING-58")
        & frame.support_seed.eq(71)
        & frame.partition.eq("deterministic_GroupKFold")
    ].iloc[0]
    summary: dict[str, object] = {
        "analysis": (
            "repeated chemical-group partitions and multiple score-blind DOCKSTRING "
            "supports for the descriptor-removal mean-squared-correlation contrast"
        ),
        "status": "exploratory_post_hoc_uncertainty_and_design_sensitivity",
        "estimand": (
            "relative reduction in mean squared off-diagonal target correlation after "
            "subtracting group-held-out, fold-locally fitted descriptor predictions"
        ),
        "not_an_explained_fraction": (
            "this is a ratio of two non-additive correlation summaries after residualizing "
            "a score surface; it is not a fraction of correlation, covariance or variance "
            "explained"
        ),
        "headline_reference_reproduction": {
            "Docking-44": float(
                original_docking[
                    "relative_mean_squared_offdiagonal_target_correlation_reduction"
                ]
            ),
            "DOCKSTRING-58_seed_71": float(
                original_dockstring[
                    "relative_mean_squared_offdiagonal_target_correlation_reduction"
                ]
            ),
        },
        "sensitivity": sensitivity,
        "conditional_paired_chemical_group_block_bootstrap": {
            "estimand": (
                "1 - weighted mean squared off-diagonal error correlation / weighted "
                "mean squared off-diagonal observed correlation"
            ),
            "method": (
                "paired Bayesian bootstrap with iid Exp(1) weights over balanced blocks "
                "of intact chemical groups; the same draw weights the observed and error "
                "surfaces"
            ),
            "conditioning": (
                "the deterministic group-held-out OOF prediction surface and target panel "
                "are fixed; predictions are not refitted inside bootstrap draws"
            ),
            "coarsening_limitation": (
                "individual chemical groups are packed into balanced blocks for tractable "
                "moment reweighting, so unrelated groups in one block share a weight; "
                "these are conditional coarsened intervals, not chemical-space or "
                "target-population confidence intervals"
            ),
            "per_dataset_support": bootstrap_summary,
        },
        "dockstring_pool": {
            **pool_report,
            "targets": int(len(targets)),
            "support_size": int(dockstring_support_size),
            "score_blind_support_seeds": [int(seed) for seed in dockstring_support_seeds],
            "selection_rule": (
                "sorted simple random sample without replacement from complete row indices; "
                "scores are not inspected during support selection"
            ),
        },
        "configuration": {
            "folds": int(folds),
            "repeated_partitions_per_support": int(repeated_partitions),
            "base_partition_seed": int(partition_seed),
            "group_definitions": {
                "Docking-44": "frozen source Butina clusters",
                "DOCKSTRING-58": (
                    "RDKit Bemis-Murcko scaffolds; every acyclic molecule is a singleton"
                ),
            },
            "every_learned_quantity_refit_inside_training_fold": True,
            "conditional_block_bootstrap_repeats": int(bootstrap_repeats),
            "conditional_block_bootstrap_requested_blocks": int(bootstrap_blocks),
            "conditional_block_bootstrap_seed": int(bootstrap_seed),
        },
        "claim_boundary": (
            "The empirical intervals quantify sensitivity to fold assignment and to a "
            "small set of score-blind 15,000-ligand supports. They are not calibrated "
            "confidence intervals for all chemical space, do not resample targets, and do "
            "not turn the non-additive mean-squared-correlation contrast into an explained "
            "fraction. The DOCKSTRING supports overlap at the rate expected for random "
            "samples from a 260,060-row pool and are sensitivity replicates rather than "
            "independent experiments."
        ),
    }
    return summary, {
        "repeated_partition_metrics.csv": frame,
        "dockstring_support_summary.csv": support_frame,
        "dockstring_support_overlap.csv": pd.DataFrame.from_records(overlaps),
        "conditional_block_bootstrap.csv": pd.DataFrame.from_records(bootstrap_rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds", type=int, default=DEFAULT_FOLDS)
    parser.add_argument(
        "--repeated-partitions", type=int, default=DEFAULT_REPEATED_PARTITIONS
    )
    parser.add_argument("--partition-seed", type=int, default=DEFAULT_PARTITION_SEED)
    parser.add_argument(
        "--dockstring-support-size", type=int, default=DEFAULT_DOCKSTRING_SUPPORT_SIZE
    )
    parser.add_argument(
        "--dockstring-support-seeds",
        type=int,
        nargs="+",
        default=DEFAULT_DOCKSTRING_SUPPORT_SEEDS,
    )
    parser.add_argument(
        "--bootstrap-repeats", type=int, default=DEFAULT_BOOTSTRAP_REPEATS
    )
    parser.add_argument("--bootstrap-blocks", type=int, default=DEFAULT_BOOTSTRAP_BLOCKS)
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    summary, tables = run(
        folds=args.folds,
        repeated_partitions=args.repeated_partitions,
        partition_seed=args.partition_seed,
        dockstring_support_size=args.dockstring_support_size,
        dockstring_support_seeds=tuple(args.dockstring_support_seeds),
        bootstrap_repeats=args.bootstrap_repeats,
        bootstrap_blocks=args.bootstrap_blocks,
        bootstrap_seed=args.bootstrap_seed,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    mechanism.write_json(args.output / "summary.json", summary)
    for name, frame in tables.items():
        mechanism.write_csv(frame, args.output / name)
    print(tables["dockstring_support_summary.csv"].to_string(index=False))


if __name__ == "__main__":
    main()
