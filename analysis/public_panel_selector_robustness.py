#!/usr/bin/env python3
"""Compact no-copy robustness controls for the public k=8 target panels.

This fail-closed companion reuses the five chemical-group-disjoint folds,
500-ligand pilots, exact residual-map panels and fixed random panels from
``public_panel_selector_controls``.  It adds two narrowly scoped controls:

1. designed-versus-random comparisons on the exact intersection of targets
   omitted by both panels, avoiding copied selected scores and differing target
   sets; and
2. an omitted-target-only comparison of the primary raw-score model followed
   by residual derivation against a secondary model trained directly on pilot
   residual targets.  Both Ridge-GCV and selected-target PC1 are reported.

The artifact is computational score reconstruction, not affinity validation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy import stats

try:
    from . import public_panel_selector_controls as selector
except ImportError:  # pragma: no cover - direct execution
    import public_panel_selector_controls as selector  # type: ignore


PACKAGE = Path(__file__).resolve().parents[1]
SOURCE_OUTPUT = PACKAGE / "results" / "public_panel_selector_controls"
DEFAULT_OUTPUT = PACKAGE / "results" / "public_panel_selector_robustness"
PILOT_SIZE = 500
PANEL_K = 8
FOLDS = 5
RANDOM_DRAWS = (0, 1, 2)
SOURCE_SEED = selector.DEFAULT_SEED
DESIGNED_SELECTOR = "pilot_residual_r2_exact_kmedoids"
RANDOM_SELECTOR = "fixed_random_panel"
METRICS = (
    "r2",
    "rmse",
    "pearson",
    "spearman",
    "equal_budget_lower_5pct_overlap",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(selector.json_ready(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def validate_source_artifact(source: Path) -> dict[str, str]:
    required = {
        "panel_metrics.csv.gz",
        "split_diagnostics.csv",
        "summary.json",
        "checksums.sha256",
    }
    if not source.is_dir() or not required <= {path.name for path in source.iterdir()}:
        raise FileNotFoundError("selector-control source artifact is incomplete")
    expected: dict[str, str] = {}
    for line in (source / "checksums.sha256").read_text(encoding="utf-8").splitlines():
        digest, filename = line.split("  ", maxsplit=1)
        expected[filename] = digest
    for filename in required - {"checksums.sha256"}:
        if filename not in expected or sha256_file(source / filename) != expected[filename]:
            raise ValueError(f"source artifact checksum mismatch: {filename}")
    return {filename: sha256_file(source / filename) for filename in sorted(required)}


def parse_panel(value: str, expected_k: int = PANEL_K) -> np.ndarray:
    panel = np.asarray([int(item) for item in str(value).split("|")], dtype=np.int64)
    if len(panel) != expected_k or len(np.unique(panel)) != expected_k:
        raise ValueError("source artifact contains an invalid panel")
    return np.sort(panel)


def panel_lookup(source_metrics: pd.DataFrame) -> dict[tuple[str, int, str, int], np.ndarray]:
    rows = source_metrics.loc[
        source_metrics.pilot_ligands.eq(PILOT_SIZE)
        & source_metrics.panel_k.eq(PANEL_K)
        & source_metrics.surface.eq("raw")
        & source_metrics.predictor.eq("multivariate_ridge_gcv")
        & (
            source_metrics.selector.eq(DESIGNED_SELECTOR)
            | (
                source_metrics.selector.eq(RANDOM_SELECTOR)
                & source_metrics.random_panel_draw.isin(RANDOM_DRAWS)
            )
        )
    ].copy()
    lookup: dict[tuple[str, int, str, int], np.ndarray] = {}
    for row in rows.itertuples(index=False):
        draw = int(row.random_panel_draw)
        key = (str(row.dataset), int(row.fold), str(row.selector), draw)
        if key in lookup:
            raise ValueError("source artifact panel key is duplicated")
        lookup[key] = parse_panel(str(row.panel_indices))
    expected = 2 * FOLDS * (1 + len(RANDOM_DRAWS))
    if len(lookup) != expected:
        raise ValueError(f"expected {expected} source panels, found {len(lookup)}")
    return lookup


def safe_correlation(first: np.ndarray, second: np.ndarray, *, rank: bool) -> float:
    if np.std(first) <= 1e-14 or np.std(second) <= 1e-14:
        return 0.0
    if rank:
        return float(stats.spearmanr(first, second).statistic)
    return float(np.corrcoef(first, second)[0, 1])


def target_metric_records(
    *,
    pilot_truth: np.ndarray,
    evaluation_truth: np.ndarray,
    prediction: np.ndarray,
    omitted: np.ndarray,
    targets: tuple[str, ...],
    base: dict[str, Any],
) -> list[dict[str, Any]]:
    if not (
        pilot_truth.shape[1]
        == evaluation_truth.shape[1]
        == prediction.shape[1]
        == len(omitted)
    ):
        raise ValueError("target metric matrices do not align")
    budget = max(1, int(np.ceil(0.05 * len(evaluation_truth))))
    records: list[dict[str, Any]] = []
    for local, target_index in enumerate(omitted):
        truth = evaluation_truth[:, local]
        fitted = prediction[:, local]
        sse = float(np.square(truth - fitted).sum())
        sst = float(np.square(truth - truth.mean()).sum())
        if sst <= 1e-14:
            raise ValueError("an evaluation target is constant")
        true_order = np.argsort(truth, kind="mergesort")[:budget]
        fitted_order = np.argsort(fitted, kind="mergesort")[:budget]
        threshold = float(np.quantile(pilot_truth[:, local], 0.05))
        true_tail = truth <= threshold
        predicted_tail = fitted <= threshold
        overlap = len(np.intersect1d(true_order, fitted_order)) / budget
        records.append(
            {
                **base,
                "target_index": int(target_index),
                "target": targets[int(target_index)],
                "r2": 1.0 - sse / sst,
                "rmse": float(np.sqrt(sse / len(truth))),
                "pearson": safe_correlation(truth, fitted, rank=False),
                "spearman": safe_correlation(truth, fitted, rank=True),
                "equal_budget_lower_5pct_overlap": float(overlap),
                "calibration_lower_5pct_threshold": threshold,
                "calibration_threshold_true_count": int(true_tail.sum()),
                "calibration_threshold_predicted_count": int(predicted_tail.sum()),
                "calibration_threshold_true_positive": int(
                    np.sum(true_tail & predicted_tail)
                ),
            }
        )
    return records


def derive_residual_prediction(
    transform: selector.PilotTransform,
    selected: np.ndarray,
    omitted: np.ndarray,
    predicted_raw_omitted: np.ndarray,
) -> np.ndarray:
    full_raw_z = np.empty_like(transform.evaluation_raw_z)
    full_raw_z[:, selected] = transform.evaluation_raw_z[:, selected]
    full_raw_z[:, omitted] = predicted_raw_omitted
    full_raw = full_raw_z * transform.raw_sds + transform.raw_means
    residual_raw = full_raw - full_raw.mean(axis=1, keepdims=True)
    residual_z = (residual_raw - transform.residual_means) / transform.residual_sds
    return residual_z[:, omitted]


def evaluate_panel(
    *,
    dataset: selector.Dataset,
    transform: selector.PilotTransform,
    panel: np.ndarray,
    panel_selector: str,
    random_draw: int,
    fold: int,
    direct: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    selected = np.sort(np.asarray(panel, dtype=np.int64))
    omitted = np.setdiff1d(np.arange(len(dataset.targets), dtype=np.int64), selected)
    x_pilot = transform.pilot_raw_z[:, selected]
    x_evaluation = transform.evaluation_raw_z[:, selected]
    raw_ridge, raw_alpha, raw_gcv = selector.ridge_gcv_predict(
        x_pilot, transform.pilot_raw_z[:, omitted], x_evaluation
    )
    raw_pc1 = selector.selected_pc1_predict(
        x_pilot, transform.pilot_raw_z[:, omitted], x_evaluation
    )
    predictions: list[tuple[str, str, np.ndarray, float, float]] = [
        ("raw_outcome", "multivariate_ridge_gcv", raw_ridge, raw_alpha, raw_gcv),
        ("raw_outcome", "selected_target_pc1", raw_pc1, -1.0, -1.0),
        (
            "raw_then_derived_residual",
            "multivariate_ridge_gcv",
            derive_residual_prediction(transform, selected, omitted, raw_ridge),
            raw_alpha,
            raw_gcv,
        ),
        (
            "raw_then_derived_residual",
            "selected_target_pc1",
            derive_residual_prediction(transform, selected, omitted, raw_pc1),
            -1.0,
            -1.0,
        ),
    ]
    if direct:
        direct_ridge, direct_alpha, direct_gcv = selector.ridge_gcv_predict(
            x_pilot, transform.pilot_residual_z[:, omitted], x_evaluation
        )
        direct_pc1 = selector.selected_pc1_predict(
            x_pilot, transform.pilot_residual_z[:, omitted], x_evaluation
        )
        predictions.extend(
            [
                (
                    "direct_residual_outcome",
                    "multivariate_ridge_gcv",
                    direct_ridge,
                    direct_alpha,
                    direct_gcv,
                ),
                (
                    "direct_residual_outcome",
                    "selected_target_pc1",
                    direct_pc1,
                    -1.0,
                    -1.0,
                ),
            ]
        )
    base = {
        "dataset": dataset.name,
        "fold": int(fold),
        "pilot_ligands": PILOT_SIZE,
        "panel_k": PANEL_K,
        "panel_selector": panel_selector,
        "random_panel_draw": int(random_draw),
        "panel_indices": "|".join(str(int(value)) for value in selected),
        "selected_targets": int(len(selected)),
        "omitted_targets": int(len(omitted)),
    }
    aggregate_records: list[dict[str, Any]] = []
    target_records: list[dict[str, Any]] = []
    for training_outcome, predictor, prediction, alpha, gcv in predictions:
        if training_outcome == "raw_outcome":
            surface = "raw"
            pilot_truth = transform.pilot_raw_z[:, omitted]
            truth = transform.evaluation_raw_z[:, omitted]
        else:
            surface = "raw_row_centered_residual"
            pilot_truth = transform.pilot_residual_z[:, omitted]
            truth = transform.evaluation_residual_z[:, omitted]
        metrics = selector.metric_bundle(truth, prediction)
        aggregate_records.append(
            {
                **base,
                "training_outcome": training_outcome,
                "surface": surface,
                "predictor": predictor,
                "ridge_alpha": float(alpha),
                "pilot_gcv": float(gcv),
                **metrics,
            }
        )
        # Common-omitted selector evidence is restricted to the primary raw model
        # and its derived residual, both using multivariate Ridge-GCV.
        if predictor == "multivariate_ridge_gcv" and training_outcome in {
            "raw_outcome",
            "raw_then_derived_residual",
        }:
            target_records.extend(
                target_metric_records(
                    pilot_truth=pilot_truth,
                    evaluation_truth=truth,
                    prediction=prediction,
                    omitted=omitted,
                    targets=dataset.targets,
                    base={
                        **base,
                        "training_outcome": training_outcome,
                        "surface": surface,
                        "predictor": predictor,
                    },
                )
            )
    return aggregate_records, target_records


def common_omitted_contrasts(targets: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    keys = ["dataset", "fold", "pilot_ligands", "surface", "training_outcome"]
    for key, group in targets.groupby(keys, sort=True):
        designed = group.loc[group.panel_selector.eq(DESIGNED_SELECTOR)]
        if designed.panel_indices.nunique() != 1:
            raise ValueError("designed target metrics are incomplete")
        designed = designed.set_index("target_index")
        for draw in RANDOM_DRAWS:
            random = group.loc[
                group.panel_selector.eq(RANDOM_SELECTOR)
                & group.random_panel_draw.eq(draw)
            ]
            if random.panel_indices.nunique() != 1:
                raise ValueError("random target metrics are incomplete")
            random = random.set_index("target_index")
            common = np.intersect1d(designed.index, random.index)
            p = int(group.selected_targets.iloc[0] + group.omitted_targets.iloc[0])
            if len(common) < p - 2 * PANEL_K:
                raise ValueError("common omitted intersection is unexpectedly small")
            record: dict[str, Any] = {
                **dict(zip(keys, key, strict=True)),
                "random_panel_draw": int(draw),
                "common_omitted_targets": int(len(common)),
                "total_targets": p,
                "designed_panel_indices": str(designed.panel_indices.iloc[0]),
                "random_panel_indices": str(random.panel_indices.iloc[0]),
            }
            for metric in METRICS:
                designed_values = designed.loc[common, metric].to_numpy(dtype=float)
                random_values = random.loc[common, metric].to_numpy(dtype=float)
                for statistic, function in (("mean", np.mean), ("median", np.median)):
                    designed_value = float(function(designed_values))
                    random_value = float(function(random_values))
                    record[f"designed_{statistic}_{metric}"] = designed_value
                    record[f"random_{statistic}_{metric}"] = random_value
                    record[f"designed_minus_random_{statistic}_{metric}"] = (
                        designed_value - random_value
                    )
                    record[f"designed_benefit_{statistic}_{metric}"] = (
                        random_value - designed_value
                        if metric == "rmse"
                        else designed_value - random_value
                    )
            records.append(record)
    return pd.DataFrame.from_records(records)


def summarize_random_within_split(contrasts: pd.DataFrame) -> pd.DataFrame:
    identifiers = [
        "dataset",
        "fold",
        "pilot_ligands",
        "surface",
        "training_outcome",
    ]
    metric_columns = [
        column for column in contrasts if column.startswith("designed_benefit_")
    ]
    records: list[dict[str, Any]] = []
    for key, group in contrasts.groupby(identifiers, sort=True):
        if set(group.random_panel_draw) != set(RANDOM_DRAWS):
            raise ValueError("a split lacks the three fixed random comparators")
        record: dict[str, Any] = {
            **dict(zip(identifiers, key, strict=True)),
            "random_panels": int(len(group)),
            "common_omitted_targets_min": int(group.common_omitted_targets.min()),
            "common_omitted_targets_max": int(group.common_omitted_targets.max()),
        }
        for column in metric_columns:
            values = group[column].to_numpy(dtype=float)
            record[f"{column}_random_median"] = float(np.median(values))
            record[f"{column}_fraction_positive"] = float(np.mean(values > 0.0))
        records.append(record)
    return pd.DataFrame.from_records(records)


def summarize_folds(frame: pd.DataFrame, identifiers: list[str]) -> pd.DataFrame:
    numeric = [
        column
        for column in frame
        if column not in {*identifiers, "fold"}
        and pd.api.types.is_numeric_dtype(frame[column])
    ]
    records: list[dict[str, Any]] = []
    for key, group in frame.groupby(identifiers, sort=True):
        if group.fold.nunique() != FOLDS:
            raise ValueError("summary lacks five independent chemical-group folds")
        if not isinstance(key, tuple):
            key = (key,)
        record: dict[str, Any] = {
            **dict(zip(identifiers, key, strict=True)),
            "folds": int(group.fold.nunique()),
        }
        for column in numeric:
            values = group[column].to_numpy(dtype=float)
            record[f"{column}_median"] = float(np.median(values))
            record[f"{column}_min"] = float(np.min(values))
            record[f"{column}_max"] = float(np.max(values))
        records.append(record)
    return pd.DataFrame.from_records(records)


def direct_residual_contrasts(metrics: pd.DataFrame) -> pd.DataFrame:
    designed = metrics.loc[
        metrics.panel_selector.eq(DESIGNED_SELECTOR)
        & metrics.surface.eq("raw_row_centered_residual")
    ]
    compared = (
        "variance_weighted_r2",
        "pooled_rmse",
        "median_target_pearson",
        "median_ligand_profile_pearson",
        "mean_ligand_profile_pearson",
    )
    records: list[dict[str, Any]] = []
    for (dataset, fold), group in designed.groupby(["dataset", "fold"], sort=True):
        indexed = group.set_index(["training_outcome", "predictor"])
        required = {
            ("raw_then_derived_residual", "multivariate_ridge_gcv"),
            ("raw_then_derived_residual", "selected_target_pc1"),
            ("direct_residual_outcome", "multivariate_ridge_gcv"),
            ("direct_residual_outcome", "selected_target_pc1"),
        }
        if set(indexed.index) != required:
            raise ValueError("direct-residual comparison is incomplete")
        record: dict[str, Any] = {"dataset": dataset, "fold": int(fold)}
        for metric in compared:
            derived_ridge = float(
                indexed.loc[
                    ("raw_then_derived_residual", "multivariate_ridge_gcv"), metric
                ]
            )
            direct_ridge = float(
                indexed.loc[
                    ("direct_residual_outcome", "multivariate_ridge_gcv"), metric
                ]
            )
            derived_pc1 = float(
                indexed.loc[
                    ("raw_then_derived_residual", "selected_target_pc1"), metric
                ]
            )
            direct_pc1 = float(
                indexed.loc[
                    ("direct_residual_outcome", "selected_target_pc1"), metric
                ]
            )
            for label, value in (
                ("derived_ridge", derived_ridge),
                ("direct_ridge", direct_ridge),
                ("derived_pc1", derived_pc1),
                ("direct_pc1", direct_pc1),
            ):
                record[f"{label}_{metric}"] = value
            if metric == "pooled_rmse":
                record[f"direct_minus_derived_ridge_benefit_{metric}"] = (
                    derived_ridge - direct_ridge
                )
                record[f"direct_ridge_minus_pc1_benefit_{metric}"] = (
                    direct_pc1 - direct_ridge
                )
                record[f"derived_ridge_minus_pc1_benefit_{metric}"] = (
                    derived_pc1 - derived_ridge
                )
            else:
                record[f"direct_minus_derived_ridge_benefit_{metric}"] = (
                    direct_ridge - derived_ridge
                )
                record[f"direct_ridge_minus_pc1_benefit_{metric}"] = (
                    direct_ridge - direct_pc1
                )
                record[f"derived_ridge_minus_pc1_benefit_{metric}"] = (
                    derived_ridge - derived_pc1
                )
        records.append(record)
    return pd.DataFrame.from_records(records)


def reproduce(
    datasets: list[selector.Dataset],
    source_metrics: pd.DataFrame,
    source_splits: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    lookup = panel_lookup(source_metrics)
    aggregate_records: list[dict[str, Any]] = []
    target_records: list[dict[str, Any]] = []
    split_records: list[dict[str, Any]] = []
    for dataset in datasets:
        complete = np.isfinite(dataset.matrix).all(axis=1)
        folds = selector.fixed_group_folds(dataset, FOLDS)
        for fold, (training, held) in enumerate(folds, start=1):
            complete_training = training[complete[training]]
            complete_held = held[complete[held]]
            rng = np.random.default_rng(
                SOURCE_SEED
                + (0 if dataset.name == "Docking-44" else 100_000_000)
                + 1_000_000 * fold
                + 10_000 * PILOT_SIZE
            )
            pilot = np.sort(rng.choice(complete_training, size=PILOT_SIZE, replace=False))
            pilot_hash = hashlib.sha256(np.asarray(pilot, dtype="<i8").tobytes()).hexdigest()
            evaluation_hash = hashlib.sha256(
                np.asarray(complete_held, dtype="<i8").tobytes()
            ).hexdigest()
            source_split = source_splits.loc[
                source_splits.dataset.eq(dataset.name)
                & source_splits.fold.eq(fold)
                & source_splits.pilot_ligands.eq(PILOT_SIZE)
            ]
            if len(source_split) != 1:
                raise ValueError("source split diagnostic is missing")
            source_row = source_split.iloc[0]
            if (
                pilot_hash != source_row.pilot_row_index_sha256
                or evaluation_hash != source_row.evaluation_row_index_sha256
            ):
                raise ValueError("reproduced fold rows do not match source artifact")
            transform = selector.pilot_transform(
                dataset.matrix[pilot], dataset.matrix[complete_held]
            )
            recomputed = selector.residual_map_medoids(transform, PANEL_K)
            designed = lookup[(dataset.name, fold, DESIGNED_SELECTOR, -1)]
            if not np.array_equal(recomputed, designed):
                raise ValueError("recomputed exact panel differs from source artifact")
            split_records.append(
                {
                    "dataset": dataset.name,
                    "fold": fold,
                    "pilot_ligands": PILOT_SIZE,
                    "evaluation_ligands": int(len(complete_held)),
                    "pilot_row_index_sha256": pilot_hash,
                    "evaluation_row_index_sha256": evaluation_hash,
                    "pilot_evaluation_row_overlap": int(
                        np.intersect1d(pilot, complete_held).size
                    ),
                    "pilot_evaluation_group_overlap": int(
                        len(set(dataset.groups[pilot]) & set(dataset.groups[complete_held]))
                    ),
                }
            )
            for panel_selector, draw, panel in [
                (DESIGNED_SELECTOR, -1, designed),
                *[
                    (
                        RANDOM_SELECTOR,
                        draw,
                        lookup[(dataset.name, fold, RANDOM_SELECTOR, draw)],
                    )
                    for draw in RANDOM_DRAWS
                ],
            ]:
                aggregate, targets = evaluate_panel(
                    dataset=dataset,
                    transform=transform,
                    panel=panel,
                    panel_selector=panel_selector,
                    random_draw=draw,
                    fold=fold,
                    direct=panel_selector == DESIGNED_SELECTOR,
                )
                aggregate_records.extend(aggregate)
                target_records.extend(targets)
    return (
        pd.DataFrame.from_records(aggregate_records),
        pd.DataFrame.from_records(target_records),
        pd.DataFrame.from_records(split_records),
    )


def validate_outputs(
    aggregate: pd.DataFrame,
    targets: pd.DataFrame,
    splits: pd.DataFrame,
    common_pairs: pd.DataFrame,
    common_splits: pd.DataFrame,
    direct: pd.DataFrame,
) -> None:
    if len(splits) != 2 * FOLDS or not (
        splits.pilot_evaluation_row_overlap.eq(0).all()
        and splits.pilot_evaluation_group_overlap.eq(0).all()
    ):
        raise ValueError("split contract failed")
    if len(common_pairs) != 2 * FOLDS * 2 * len(RANDOM_DRAWS):
        raise ValueError("common-omitted pair design is incomplete")
    if len(common_splits) != 2 * FOLDS * 2 or len(direct) != 2 * FOLDS:
        raise ValueError("fold-level robustness design is incomplete")
    for frame in (aggregate, targets, splits, common_pairs, common_splits, direct):
        numeric = frame.select_dtypes(include=[np.number])
        if numeric.empty or not np.isfinite(numeric.to_numpy(dtype=float)).all():
            raise ValueError("a robustness table contains non-finite values")


def write_artifact(
    output: Path,
    *,
    tables: dict[str, pd.DataFrame],
    source: Path,
    source_hashes: dict[str, str],
    datasets: list[selector.Dataset],
    overwrite: bool,
) -> None:
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    try:
        for filename, table in tables.items():
            table.to_csv(staging / filename, index=False, float_format="%.17g")
        summary = {
            "analysis_status": "strict_public_k8_selector_robustness_control",
            "producer": "analysis/public_panel_selector_robustness.py",
            "producer_sha256": sha256_file(Path(__file__)),
            "test_sha256": sha256_file(
                PACKAGE / "analysis" / "test_public_panel_selector_robustness.py"
            ),
            "source_artifact": str(source.relative_to(PACKAGE)),
            "source_artifact_sha256": source_hashes,
            "parameters": {
                "pilot_ligands": PILOT_SIZE,
                "panel_k": PANEL_K,
                "folds": FOLDS,
                "random_panel_draws": list(RANDOM_DRAWS),
                "source_seed": SOURCE_SEED,
                "ridge_alphas": list(selector.RIDGE_ALPHAS),
            },
            "contracts": {
                "common_omitted_primary": (
                    "designed and random predictions are compared only on the exact "
                    "intersection of targets omitted by both k=8 panels; no selected "
                    "truth is copied into these target-level contrasts"
                ),
                "random_inference": (
                    "three pre-existing fixed random panels per fold are descriptive "
                    "Monte Carlo controls, first aggregated within fold"
                ),
                "direct_residual_secondary": (
                    "selected raw target scores are the only inputs; direct residual "
                    "Ridge/PC1 and raw-to-derived residual Ridge/PC1 use identical exact "
                    "panels, folds, omitted targets and evaluation rows"
                ),
                "primary_deployment_model_unchanged": (
                    "the manuscript deployment model remains raw-score prediction followed "
                    "by residual derivation; direct residual training is sensitivity only"
                ),
            },
            "dataset_input_sha256": {
                dataset.name: sha256_file(dataset.source_path) for dataset in datasets
            },
            "table_rows": {filename: int(len(table)) for filename, table in tables.items()},
            "limitations": [
                "These are Vina score-reconstruction controls, not affinity validation.",
                "Five chemical-group folds are descriptive composition sensitivities.",
                "Only three fixed random panels per fold enter the common-omitted comparison.",
                "Direct residual training is not the deployable primary policy.",
            ],
        }
        write_json(staging / "summary.json", summary)
        (staging / "README.md").write_text(
            """# k=8 target-panel robustness controls

This compact artifact adds two controls to `public_panel_selector_controls`:
designed-versus-random comparisons on the same common omitted targets, and a
secondary direct-residual training sensitivity.  It reuses the exact five
chemical-group folds, 500-ligand pilots, k=8 panels and first three fixed random
panels from the checksummed source artifact.

The common-omitted analysis contains no copied selected truths.  The direct
residual analysis uses selected raw scores as its only predictors and is not a
replacement for the primary raw-score deployment model.
""",
            encoding="utf-8",
        )
        checksum_lines = []
        for path in sorted(staging.iterdir()):
            if path.name != "checksums.sha256":
                checksum_lines.append(f"{sha256_file(path)}  {path.name}")
        (staging / "checksums.sha256").write_text(
            "\n".join(checksum_lines) + "\n", encoding="utf-8"
        )
        if output.exists():
            if not overwrite:
                raise FileExistsError(f"output exists; pass --overwrite: {output}")
            backup = output.with_name(f".{output.name}.previous-{os.getpid()}")
            output.rename(backup)
            try:
                staging.rename(output)
            except Exception:
                backup.rename(output)
                raise
            shutil.rmtree(backup)
        else:
            staging.rename(output)
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def run(args: argparse.Namespace) -> None:
    source_hashes = validate_source_artifact(args.source)
    source_metrics = pd.read_csv(args.source / "panel_metrics.csv.gz")
    source_splits = pd.read_csv(args.source / "split_diagnostics.csv")
    datasets = selector.load_datasets(requested=("Docking-44", "DOCKSTRING-58"))
    aggregate, target_metrics, splits = reproduce(
        datasets, source_metrics, source_splits
    )
    common_pairs = common_omitted_contrasts(target_metrics)
    common_splits = summarize_random_within_split(common_pairs)
    common_summary = summarize_folds(
        common_splits, ["dataset", "surface", "training_outcome"]
    )
    direct = direct_residual_contrasts(aggregate)
    direct_summary = summarize_folds(direct, ["dataset"])
    validate_outputs(
        aggregate, target_metrics, splits, common_pairs, common_splits, direct
    )
    tables = {
        "panel_metrics.csv": aggregate,
        "primary_target_metrics.csv": target_metrics,
        "split_diagnostics.csv": splits,
        "common_omitted_pair_contrasts.csv": common_pairs,
        "common_omitted_split_contrasts.csv": common_splits,
        "common_omitted_summary.csv": common_summary,
        "direct_residual_contrasts.csv": direct,
        "direct_residual_summary.csv": direct_summary,
    }
    write_artifact(
        args.output,
        tables=tables,
        source=args.source,
        source_hashes=source_hashes,
        datasets=datasets,
        overwrite=args.overwrite,
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "common_pairs": len(common_pairs),
                "direct_folds": len(direct),
                "tables": len(tables),
            },
            sort_keys=True,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE_OUTPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
