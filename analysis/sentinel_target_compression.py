#!/usr/bin/env python3
"""Historical scaffold-held-out audit of Vina self-reconstruction.

The analysis asks whether a chemically held-out ligand's full DOCKSTRING score
profile can be reconstructed after docking it only against a small, training-
selected set of targets.  Sentinel targets are chosen from the training docking
matrix by column-pivoted QR.  No experimental outcome enters target selection,
model fitting, or hyperparameter choice.

Direct pivoted QR of column-standardized targets has an arbitrary equal-norm
first pivot and is not retained as a target-panel selection result.  This file
keeps the fair random-plus-descriptor controls for provenance, but the
experimental residual-DEIM analysis in ``experimental_sentinel_panel.py``
supersedes it.  Do not cite these self-reconstruction outputs as assay-panel
validation.  This is a science-only exploratory analysis and does not edit
manuscript files.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import qr
from scipy.stats import rankdata, spearmanr
from sklearn.model_selection import GroupKFold

from residual_mechanism_analysis import (
    AnalysisConfig,
    load_docking44,
    load_dockstring,
    two_way_center,
)


DEFAULT_K = (5, 10, 20)
PRIMARY_METRICS = (
    "full_target_r2",
    "full_mean_row_spearman",
    "full_top5_overlap",
    "held_out_target_r2",
    "held_out_mean_row_spearman",
    "held_out_top5_overlap",
)


def _standardize_train_test(
    train: np.ndarray, test: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    mean = train.mean(axis=0)
    sd = train.std(axis=0, ddof=1)
    sd = np.where(sd > 1e-12, sd, 1.0)
    return (train - mean) / sd, (test - mean) / sd


def _ridge_predict(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    alpha: float = 1e-6,
) -> np.ndarray:
    """Fixed-ridge multivariate prediction with fold-local feature scaling."""
    z_train, z_test = _standardize_train_test(train_x, test_x)
    x_mean = train_y.mean(axis=0)
    centered_y = train_y - x_mean
    gram = z_train.T @ z_train
    penalty = alpha * np.eye(gram.shape[0], dtype=np.float64)
    coef = np.linalg.solve(gram + penalty, z_train.T @ centered_y)
    return x_mean + z_test @ coef


def _select_qr_targets(train_matrix: np.ndarray, k: int) -> np.ndarray:
    z_train, _ = _standardize_train_test(train_matrix, train_matrix[:1])
    _, _, pivots = qr(z_train, mode="economic", pivoting=True, check_finite=False)
    return np.asarray(pivots[:k], dtype=int)


def _sample_family_diverse_targets(
    families: np.ndarray,
    k: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Random panel with maximal broad-family coverage at the requested size."""
    families = np.asarray(families, dtype=object)
    if not 1 <= k <= len(families):
        raise ValueError("k must be between one and the number of targets")
    labels = np.asarray(sorted(set(families.tolist())), dtype=object)
    chosen: list[int] = []
    if k < len(labels):
        selected_labels = rng.choice(labels, size=k, replace=False)
    else:
        selected_labels = labels
    for label in selected_labels:
        candidates = np.flatnonzero(families == label)
        chosen.append(int(rng.choice(candidates)))
    remaining = k - len(chosen)
    if remaining:
        available = np.setdiff1d(
            np.arange(len(families), dtype=int),
            np.asarray(chosen, dtype=int),
            assume_unique=False,
        )
        chosen.extend(rng.choice(available, size=remaining, replace=False).tolist())
    return np.sort(np.asarray(chosen, dtype=int))


def _mean_row_spearman(true: np.ndarray, pred: np.ndarray) -> float:
    true_rank = rankdata(true, axis=1, method="average")
    pred_rank = rankdata(pred, axis=1, method="average")
    true_rank -= true_rank.mean(axis=1, keepdims=True)
    pred_rank -= pred_rank.mean(axis=1, keepdims=True)
    numerator = np.sum(true_rank * pred_rank, axis=1)
    denominator = np.sqrt(
        np.sum(true_rank**2, axis=1) * np.sum(pred_rank**2, axis=1)
    )
    valid = denominator > 0
    return float(np.mean(numerator[valid] / denominator[valid]))


def _mean_topk_overlap(true: np.ndarray, pred: np.ndarray, top_k: int = 5) -> float:
    k = min(top_k, true.shape[1])
    true_top = np.argpartition(true, kth=k - 1, axis=1)[:, :k]
    pred_top = np.argpartition(pred, kth=k - 1, axis=1)[:, :k]
    overlap = [
        len(set(a.tolist()).intersection(b.tolist())) / k
        for a, b in zip(true_top, pred_top, strict=True)
    ]
    return float(np.mean(overlap))


def _pair_order_accuracy(true: np.ndarray, pred: np.ndarray) -> float:
    left, right = np.triu_indices(true.shape[1], k=1)
    true_delta = true[:, left] - true[:, right]
    pred_delta = pred[:, left] - pred[:, right]
    valid = (true_delta != 0) & (pred_delta != 0)
    if not np.any(valid):
        return float("nan")
    return float(np.mean(np.sign(true_delta[valid]) == np.sign(pred_delta[valid])))


def _geometry_spearman(true: np.ndarray, pred: np.ndarray) -> float:
    true_corr = np.corrcoef(true, rowvar=False)
    pred_corr = np.corrcoef(pred, rowvar=False)
    upper = np.triu_indices(true.shape[1], k=1)
    return float(spearmanr(true_corr[upper], pred_corr[upper]).statistic)


def _metrics(
    true_all: np.ndarray,
    pred_all: np.ndarray,
    held_out_targets: np.ndarray,
) -> dict[str, float]:
    true = true_all[:, held_out_targets]
    pred = pred_all[:, held_out_targets]
    residual = true - pred
    denominator = np.sum((true - true.mean(axis=0, keepdims=True)) ** 2)
    r2 = 1.0 - np.sum(residual**2) / denominator
    full_residual = true_all - pred_all
    full_denominator = np.sum(
        (true_all - true_all.mean(axis=0, keepdims=True)) ** 2
    )
    return {
        "held_out_target_r2": float(r2),
        "held_out_target_rmse": float(np.sqrt(np.mean(residual**2))),
        "held_out_mean_row_spearman": _mean_row_spearman(true, pred),
        "held_out_top5_overlap": _mean_topk_overlap(true, pred, top_k=5),
        "held_out_pair_order_accuracy": _pair_order_accuracy(true, pred),
        "held_out_target_geometry_spearman": _geometry_spearman(true, pred),
        "full_target_r2": float(
            1.0 - np.sum(full_residual**2) / full_denominator
        ),
        "full_target_rmse": float(np.sqrt(np.mean(full_residual**2))),
        "full_pair_order_accuracy": _pair_order_accuracy(true_all, pred_all),
        "full_target_geometry_spearman": _geometry_spearman(true_all, pred_all),
        "full_mean_row_spearman": _mean_row_spearman(true_all, pred_all),
        "full_top5_overlap": _mean_topk_overlap(true_all, pred_all, top_k=5),
    }


def _evaluate_feature_set(
    train_matrix: np.ndarray,
    test_matrix: np.ndarray,
    train_features: np.ndarray,
    test_features: np.ndarray,
    sentinels: np.ndarray,
    alpha: float = 1e-6,
) -> dict[str, float]:
    held_out = np.setdiff1d(
        np.arange(train_matrix.shape[1], dtype=int), sentinels, assume_unique=True
    )
    prediction = _ridge_predict(
        train_features, train_matrix[:, held_out], test_features, alpha=alpha
    )
    full_prediction = np.empty_like(test_matrix)
    full_prediction[:, held_out] = prediction
    # Sentinel scores are observed because those are the targets actually docked.
    full_prediction[:, sentinels] = test_matrix[:, sentinels]
    return _metrics(test_matrix, full_prediction, held_out)


def run_analysis(
    dataset: str,
    sample_size: int,
    sample_seed: int,
    folds: int,
    k_values: tuple[int, ...],
    random_repeats: int,
    random_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    if dataset == "dockstring":
        bundle = load_dockstring(
            AnalysisConfig(
                dockstring_sample_size=sample_size,
                dockstring_sample_seed=sample_seed,
                folds=folds,
            )
        )
    elif dataset == "docking44":
        bundle = load_docking44()
    else:
        raise ValueError(f"unknown dataset: {dataset}")
    matrix = np.asarray(bundle.matrix, dtype=np.float64)
    descriptors = np.asarray(bundle.descriptor_matrix, dtype=np.float64)
    groups = np.asarray(bundle.groups, dtype=object)
    families = np.asarray(bundle.families, dtype=object)
    splitter = GroupKFold(n_splits=folds)
    rng = np.random.default_rng(random_seed)
    records: list[dict[str, object]] = []
    selection_records: list[dict[str, object]] = []
    random_panels: dict[tuple[int, int], dict[str, np.ndarray]] = {}
    for k in k_values:
        for repeat in range(random_repeats):
            random_panels[(k, repeat)] = {
                "random_sentinels": np.sort(
                    rng.choice(matrix.shape[1], size=k, replace=False)
                ),
                "family_diverse_random_sentinels": _sample_family_diverse_targets(
                    families, k, rng
                ),
            }

    for fold, (train_index, test_index) in enumerate(
        splitter.split(matrix, groups=groups), start=1
    ):
        train = matrix[train_index]
        test = matrix[test_index]
        qr_orders = {
            "raw_qr": _select_qr_targets(train, max(k_values)),
            "residual_qr": _select_qr_targets(two_way_center(train), max(k_values)),
        }
        for strategy, qr_order in qr_orders.items():
            for rank, target_index in enumerate(qr_order, start=1):
                selection_records.append(
                    {
                        "fold": fold,
                        "selection_strategy": strategy,
                        "selection_rank": rank,
                        "target_index": int(target_index),
                        "target": bundle.targets[int(target_index)],
                        "family": bundle.families[int(target_index)],
                    }
                )

        for k in k_values:
            deterministic_panels = {
                "raw_qr_sentinels": qr_orders["raw_qr"][:k],
                "residual_qr_sentinels": qr_orders["residual_qr"][:k],
            }
            for panel_name, sentinels in deterministic_panels.items():
                model_features = {
                    panel_name: (train[:, sentinels], test[:, sentinels]),
                    f"{panel_name}_plus_descriptors": (
                        np.column_stack(
                            [train[:, sentinels], descriptors[train_index]]
                        ),
                        np.column_stack(
                            [test[:, sentinels], descriptors[test_index]]
                        ),
                    ),
                }
                if panel_name == "raw_qr_sentinels":
                    model_features["descriptors_for_qr_heldout"] = (
                        descriptors[train_index],
                        descriptors[test_index],
                    )
                for method, (train_features, test_features) in model_features.items():
                    metrics = _evaluate_feature_set(
                        train, test, train_features, test_features, sentinels
                    )
                    records.append(
                        {
                            "fold": fold,
                            "k": k,
                            "method": method,
                            "random_repeat": -1,
                            "sentinel_family_count": int(
                                len(set(families[sentinels].tolist()))
                            ),
                            "train_ligands": int(len(train_index)),
                            "test_ligands": int(len(test_index)),
                            "train_chemical_groups": int(
                                len(set(groups[train_index]))
                            ),
                            "test_chemical_groups": int(len(set(groups[test_index]))),
                            **metrics,
                        }
                    )

            for repeat in range(random_repeats):
                for panel_name, sentinels in random_panels[(k, repeat)].items():
                    feature_sets = {
                        f"{panel_name}_plus_descriptors": (
                            np.column_stack(
                                [train[:, sentinels], descriptors[train_index]]
                            ),
                            np.column_stack(
                                [test[:, sentinels], descriptors[test_index]]
                            ),
                        ),
                    }
                    for method, (train_features, test_features) in feature_sets.items():
                        metrics = _evaluate_feature_set(
                            train,
                            test,
                            train_features,
                            test_features,
                            sentinels,
                        )
                        records.append(
                            {
                                "fold": fold,
                                "k": k,
                                "method": method,
                                "random_repeat": repeat,
                                "sentinel_family_count": int(
                                    len(set(families[sentinels].tolist()))
                                ),
                                "train_ligands": int(len(train_index)),
                                "test_ligands": int(len(test_index)),
                                "train_chemical_groups": int(
                                    len(set(groups[train_index]))
                                ),
                                "test_chemical_groups": int(
                                    len(set(groups[test_index]))
                                ),
                                **metrics,
                            }
                        )

    records_frame = pd.DataFrame.from_records(records)
    selection_frame = pd.DataFrame.from_records(selection_records)
    metadata: dict[str, object] = {
        "analysis_status": "exploratory_science_only",
        "dataset": bundle.name,
        "sample_size": int(len(matrix)),
        "sample_seed": sample_seed if dataset == "dockstring" else None,
        "folds": folds,
        "chemical_split": bundle.group_definition,
        "k_values": list(k_values),
        "random_sentinel_repeats_per_fold": random_repeats,
        "random_seed": random_seed,
        "target_selection": (
            "column-pivoted QR of either the fold-local column-standardized raw "
            "training docking matrix or its fold-local two-way-centered residual"
        ),
        "predictor": (
            "fixed near-OLS ridge alpha=1e-6 after fold-local feature standardization"
        ),
        "outcome_leakage": (
            "none: only docking scores and molecular descriptors enter this analysis"
        ),
        "claim_boundary": (
            "reconstruction of Vina scores on held-out chemical scaffolds; not "
            "experimental affinity or target-retrieval validation"
        ),
    }
    stability_frame = selection_stability(selection_frame, k_values, folds)
    return records_frame, selection_frame, stability_frame, metadata


def summarize(records: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [
        column
        for column in records.columns
        if column.endswith(("_r2", "_rmse", "_spearman", "_overlap", "_accuracy"))
    ]
    grouped = records.groupby(["k", "method"], as_index=False)[metric_columns].mean()
    return grouped.sort_values(["k", "method"]).reset_index(drop=True)


def selection_stability(
    selections: pd.DataFrame,
    k_values: tuple[int, ...],
    folds: int,
) -> pd.DataFrame:
    """Pairwise fold Jaccard and consensus size for each QR strategy and k."""
    records: list[dict[str, object]] = []
    for strategy in sorted(selections["selection_strategy"].unique()):
        selected_strategy = selections[selections["selection_strategy"] == strategy]
        for k in k_values:
            fold_sets = {
                fold: set(
                    selected_strategy[
                        (selected_strategy["fold"] == fold)
                        & (selected_strategy["selection_rank"] <= k)
                    ]["target"].tolist()
                )
                for fold in range(1, folds + 1)
            }
            jaccards = []
            for left in range(1, folds + 1):
                for right in range(left + 1, folds + 1):
                    union = fold_sets[left] | fold_sets[right]
                    jaccards.append(
                        len(fold_sets[left] & fold_sets[right]) / len(union)
                    )
            frequencies: dict[str, int] = {}
            for values in fold_sets.values():
                for target in values:
                    frequencies[target] = frequencies.get(target, 0) + 1
            records.append(
                {
                    "selection_strategy": strategy,
                    "k": k,
                    "mean_pairwise_fold_jaccard": float(np.mean(jaccards)),
                    "min_pairwise_fold_jaccard": float(np.min(jaccards)),
                    "targets_selected_in_all_folds": int(
                        sum(count == folds for count in frequencies.values())
                    ),
                    "union_targets_across_folds": int(len(frequencies)),
                }
            )
    return pd.DataFrame.from_records(records)


def paired_qr_comparisons(records: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare each deterministic QR panel with repeat-matched random panels."""
    detail: list[dict[str, object]] = []
    qr_methods = (
        "raw_qr_sentinels_plus_descriptors",
        "residual_qr_sentinels_plus_descriptors",
    )
    random_methods = (
        "random_sentinels_plus_descriptors",
        "family_diverse_random_sentinels_plus_descriptors",
    )
    for k in sorted(records["k"].unique()):
        selected_k = records[records["k"] == k]
        for qr_method in qr_methods:
            qr = selected_k[selected_k["method"] == qr_method].set_index("fold")
            for random_method in random_methods:
                random = selected_k[selected_k["method"] == random_method]
                for repeat, repeated in random.groupby("random_repeat"):
                    repeated = repeated.set_index("fold")
                    if not qr.index.equals(repeated.index):
                        raise ValueError("QR and random controls do not share folds")
                    for metric in PRIMARY_METRICS:
                        qr_mean = float(qr[metric].mean())
                        random_mean = float(repeated[metric].mean())
                        detail.append(
                            {
                                "k": int(k),
                                "qr_method": qr_method,
                                "random_method": random_method,
                                "random_repeat": int(repeat),
                                "metric": metric,
                                "qr_fold_mean": qr_mean,
                                "random_fold_mean": random_mean,
                                "qr_minus_random": qr_mean - random_mean,
                            }
                        )
    detail_frame = pd.DataFrame.from_records(detail)
    summary_records: list[dict[str, object]] = []
    group_columns = ["k", "qr_method", "random_method", "metric"]
    for keys, group in detail_frame.groupby(group_columns, sort=True):
        deltas = group["qr_minus_random"].to_numpy(dtype=np.float64)
        summary_records.append(
            {
                **dict(zip(group_columns, keys, strict=True)),
                "qr_fold_mean": float(group["qr_fold_mean"].iloc[0]),
                "random_mean": float(group["random_fold_mean"].mean()),
                "mean_qr_minus_random": float(deltas.mean()),
                "random_2.5_percentile": float(
                    group["random_fold_mean"].quantile(0.025)
                ),
                "random_97.5_percentile": float(
                    group["random_fold_mean"].quantile(0.975)
                ),
                "delta_2.5_percentile": float(np.quantile(deltas, 0.025)),
                "delta_97.5_percentile": float(np.quantile(deltas, 0.975)),
                "one_sided_monte_carlo_p": float(
                    (1 + np.sum(deltas <= 0)) / (1 + len(deltas))
                ),
                "random_repeats": int(len(deltas)),
            }
        )
    return detail_frame, pd.DataFrame.from_records(summary_records)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", choices=("dockstring", "docking44"), default="dockstring"
    )
    parser.add_argument("--sample-size", type=int, default=15_000)
    parser.add_argument("--sample-seed", type=int, default=71)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--k", default=",".join(map(str, DEFAULT_K)))
    parser.add_argument("--random-repeats", type=int, default=200)
    parser.add_argument("--random-seed", type=int, default=202_608_03)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/sentinel_target_compression"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    k_values = tuple(int(value) for value in args.k.split(",") if value)
    maximum_targets = 58 if args.dataset == "dockstring" else 44
    if not k_values or min(k_values) < 1 or max(k_values) >= maximum_targets:
        raise ValueError(f"k values must lie between 1 and {maximum_targets - 1}")
    records, selections, stability, metadata = run_analysis(
        dataset=args.dataset,
        sample_size=args.sample_size,
        sample_seed=args.sample_seed,
        folds=args.folds,
        k_values=k_values,
        random_repeats=args.random_repeats,
        random_seed=args.random_seed,
    )
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    records.to_csv(output_dir / "fold_metrics.csv", index=False)
    selections.to_csv(output_dir / "sentinel_selections.csv", index=False)
    stability.to_csv(output_dir / "selection_stability.csv", index=False)
    summary = summarize(records)
    summary.to_csv(output_dir / "summary.csv", index=False)
    paired_detail, paired_summary = paired_qr_comparisons(records)
    paired_detail.to_csv(output_dir / "paired_qr_vs_random.csv", index=False)
    paired_summary.to_csv(output_dir / "paired_qr_vs_random_summary.csv", index=False)
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    print(paired_summary.to_string(index=False))


if __name__ == "__main__":
    main()
