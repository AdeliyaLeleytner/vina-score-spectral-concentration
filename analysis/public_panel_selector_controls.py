#!/usr/bin/env python3
"""Leakage-resistant target-panel selector controls on public Vina matrices.

The analysis deliberately separates target selection from evaluation.  Five
chemical-group folds are formed once.  A fully observed pilot is sampled only
from four folds; the fifth fold is never used for imputation, scaling, target
selection, hyperparameter selection, or model fitting.  On evaluation ligands,
the predictor receives only the raw scores for the selected targets.

Two pilot-only selectors are compared at several panel sizes:

* exact k-medoids on 1-r^2 from the raw-row-centred residual target map;
* column-pivoted QR on the pilot raw-score correlation representation.

Each selector is evaluated against at least 50 fixed random panels.  Ridge-GCV
predicts omitted raw target scores.  A full raw profile is constructed by
copying the selected observed raw scores into the prediction.  Only then is a
secondary residual profile formed by row-centring in raw score units and using
pilot residual column scales.  A selected-target PC1 predictor and a seven-
descriptor ridge are explicit baselines.  Common all-target profile metrics are
primary; metrics over selector-specific omitted targets are secondary.

This producer is intentionally standalone.  It imports the frozen public data
loaders and chemical-group definitions, but does not depend on any result from
``public_panel_domain_utility.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy import linalg, stats
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix
from sklearn.model_selection import GroupKFold

try:
    from . import public_scaffold_holdout_recovery as group_source
    from .residual_mechanism_analysis import DESCRIPTOR_NAMES, molecular_descriptors
except ImportError:  # pragma: no cover - direct execution
    import public_scaffold_holdout_recovery as group_source  # type: ignore
    from residual_mechanism_analysis import (  # type: ignore
        DESCRIPTOR_NAMES,
        molecular_descriptors,
    )


PACKAGE = Path(__file__).resolve().parents[1]
FROZEN = PACKAGE / "data" / "frozen"
DEFAULT_DOCKING44 = FROZEN / "df_final_v4.csv.gz"
DEFAULT_DOCKSTRING = FROZEN / "dockstring-dataset.tsv.gz"
DEFAULT_OUTPUT = PACKAGE / "results" / "public_panel_selector_controls"
DEFAULT_PILOT_SIZES = (500,)
DEFAULT_KS = (4, 6, 8, 10, 12)
DEFAULT_FOLDS = 5
DEFAULT_RANDOM_PANELS = 50
DEFAULT_SEED = 202_608_28
RIDGE_ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0)
PRIMARY_METRIC = "full_profile_variance_weighted_r2"


@dataclass(frozen=True)
class Dataset:
    name: str
    matrix: np.ndarray
    groups: np.ndarray
    descriptors: np.ndarray
    targets: tuple[str, ...]
    source_indices: np.ndarray
    source_path: Path
    loader_metadata: dict[str, Any]


@dataclass(frozen=True)
class PilotTransform:
    pilot_raw: np.ndarray
    evaluation_raw: np.ndarray
    pilot_raw_z: np.ndarray
    evaluation_raw_z: np.ndarray
    raw_means: np.ndarray
    raw_sds: np.ndarray
    pilot_residual_z: np.ndarray
    evaluation_residual_z: np.ndarray
    residual_means: np.ndarray
    residual_sds: np.ndarray


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float) and not np.isfinite(value):
        raise ValueError("non-finite value cannot be serialized")
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(json_ready(payload), indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def load_datasets(
    docking44_path: Path = DEFAULT_DOCKING44,
    dockstring_path: Path = DEFAULT_DOCKSTRING,
    requested: Iterable[str] = ("Docking-44", "DOCKSTRING-58"),
) -> list[Dataset]:
    requested_set = set(requested)
    unknown = requested_set - {"Docking-44", "DOCKSTRING-58"}
    if unknown:
        raise ValueError(f"unknown datasets: {sorted(unknown)}")
    datasets: list[Dataset] = []
    if "Docking-44" in requested_set:
        matrix, groups, _, source_indices, metadata = group_source.load_docking44(
            docking44_path
        )
        frame = pd.read_csv(
            docking44_path,
            usecols=["Cleaned SMILES", "Canonical SMILES"],
        )
        smiles = frame["Cleaned SMILES"].fillna(frame["Canonical SMILES"]).astype(str)
        descriptors = molecular_descriptors(smiles)
        datasets.append(
            Dataset(
                name="Docking-44",
                matrix=np.asarray(matrix, dtype=np.float64),
                groups=np.asarray(groups, dtype=object),
                descriptors=np.asarray(descriptors, dtype=np.float64),
                targets=tuple(group_source.DOCKING44_TARGETS),
                source_indices=np.asarray(source_indices, dtype=np.int64),
                source_path=docking44_path,
                loader_metadata=metadata,
            )
        )
    if "DOCKSTRING-58" in requested_set:
        matrix, groups, _, source_indices, metadata = group_source.load_dockstring(
            dockstring_path
        )
        header = pd.read_csv(dockstring_path, sep="\t", nrows=0)
        targets = tuple(
            column for column in header.columns if column not in {"inchikey", "smiles"}
        )
        smiles = pd.read_csv(dockstring_path, sep="\t", usecols=["smiles"])[
            "smiles"
        ]
        selected_smiles = smiles.iloc[source_indices].astype(str).reset_index(drop=True)
        descriptors = molecular_descriptors(selected_smiles)
        datasets.append(
            Dataset(
                name="DOCKSTRING-58",
                matrix=np.asarray(matrix, dtype=np.float64),
                groups=np.asarray(groups, dtype=object),
                descriptors=np.asarray(descriptors, dtype=np.float64),
                targets=targets,
                source_indices=np.asarray(source_indices, dtype=np.int64),
                source_path=dockstring_path,
                loader_metadata=metadata,
            )
        )
    for dataset in datasets:
        n, p = dataset.matrix.shape
        if not (
            len(dataset.groups)
            == len(dataset.descriptors)
            == len(dataset.source_indices)
            == n
        ):
            raise ValueError(f"{dataset.name}: row metadata do not align")
        if p != len(dataset.targets) or dataset.descriptors.shape[1] != len(
            DESCRIPTOR_NAMES
        ):
            raise ValueError(f"{dataset.name}: target or descriptor metadata mismatch")
    return datasets


def pilot_transform(pilot_raw: np.ndarray, evaluation_raw: np.ndarray) -> PilotTransform:
    """Fit raw and residual scaling on a fully observed pilot only."""

    pilot_raw = np.asarray(pilot_raw, dtype=np.float64)
    evaluation_raw = np.asarray(evaluation_raw, dtype=np.float64)
    if (
        pilot_raw.ndim != 2
        or evaluation_raw.ndim != 2
        or pilot_raw.shape[1] != evaluation_raw.shape[1]
        or len(pilot_raw) < 20
        or not np.isfinite(pilot_raw).all()
        or not np.isfinite(evaluation_raw).all()
    ):
        raise ValueError("pilot and evaluation must be aligned, finite score matrices")
    raw_means = pilot_raw.mean(axis=0)
    raw_sds = pilot_raw.std(axis=0, ddof=1)
    if np.any(raw_sds <= 1e-12):
        raise ValueError("a pilot raw target is constant")
    pilot_raw_z = (pilot_raw - raw_means) / raw_sds
    evaluation_raw_z = (evaluation_raw - raw_means) / raw_sds
    pilot_residual_raw = pilot_raw - pilot_raw.mean(axis=1, keepdims=True)
    evaluation_residual_raw = evaluation_raw - evaluation_raw.mean(
        axis=1, keepdims=True
    )
    residual_means = pilot_residual_raw.mean(axis=0)
    residual_sds = pilot_residual_raw.std(axis=0, ddof=1)
    if np.any(residual_sds <= 1e-12):
        raise ValueError("a pilot residual target is constant")
    return PilotTransform(
        pilot_raw=pilot_raw,
        evaluation_raw=evaluation_raw,
        pilot_raw_z=pilot_raw_z,
        evaluation_raw_z=evaluation_raw_z,
        raw_means=raw_means,
        raw_sds=raw_sds,
        pilot_residual_z=(pilot_residual_raw - residual_means) / residual_sds,
        evaluation_residual_z=(evaluation_residual_raw - residual_means)
        / residual_sds,
        residual_means=residual_means,
        residual_sds=residual_sds,
    )


def target_correlation(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    cross = centered.T @ centered
    variance = np.diag(cross)
    if matrix.ndim != 2 or len(matrix) < 3 or np.any(variance <= 1e-14):
        raise ValueError("target correlation requires a nonconstant matrix")
    result = cross / np.sqrt(np.outer(variance, variance))
    result = np.clip((result + result.T) / 2.0, -1.0, 1.0)
    np.fill_diagonal(result, 1.0)
    return result


def exact_k_medoids(distance: np.ndarray, k: int) -> np.ndarray:
    """Exact discrete p-median with deterministic target-index tie breaking."""

    distance = np.asarray(distance, dtype=np.float64)
    p = len(distance)
    if (
        distance.shape != (p, p)
        or not np.isfinite(distance).all()
        or np.any(distance < -1e-12)
        or not 1 <= k < p
    ):
        raise ValueError("invalid k-medoids problem")
    assignments = p * p
    variables = assignments + p
    objective = np.zeros(variables, dtype=np.float64)
    objective[:assignments] = distance.reshape(-1)
    objective[assignments:] = 1e-12 * np.arange(p)
    integrality = np.zeros(variables, dtype=np.int8)
    integrality[assignments:] = 1
    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []
    lower: list[float] = []
    upper: list[float] = []
    constraint = 0
    for target in range(p):
        for representative in range(p):
            rows.append(constraint)
            columns.append(target * p + representative)
            values.append(1.0)
        lower.append(1.0)
        upper.append(1.0)
        constraint += 1
    for target in range(p):
        for representative in range(p):
            rows.extend((constraint, constraint))
            columns.extend((target * p + representative, assignments + representative))
            values.extend((1.0, -1.0))
            lower.append(-np.inf)
            upper.append(0.0)
            constraint += 1
    for representative in range(p):
        rows.append(constraint)
        columns.append(assignments + representative)
        values.append(1.0)
    lower.append(float(k))
    upper.append(float(k))
    constraint += 1
    matrix = coo_matrix(
        (values, (rows, columns)), shape=(constraint, variables), dtype=np.float64
    ).tocsr()
    result = milp(
        c=objective,
        integrality=integrality,
        bounds=Bounds(np.zeros(variables), np.ones(variables)),
        constraints=LinearConstraint(
            matrix,
            np.asarray(lower, dtype=np.float64),
            np.asarray(upper, dtype=np.float64),
        ),
        options={"presolve": True, "mip_rel_gap": 0.0},
    )
    if not result.success or result.x is None:
        raise RuntimeError(f"exact k-medoids failed: {result.message}")
    panel = np.flatnonzero(result.x[assignments:] > 0.5).astype(np.int64)
    if len(panel) != k:
        raise RuntimeError("exact k-medoids returned an invalid panel")
    return panel


def residual_map_medoids(transform: PilotTransform, k: int) -> np.ndarray:
    correlation = target_correlation(transform.pilot_residual_z)
    distance = np.maximum(1.0 - np.square(correlation), 0.0)
    np.fill_diagonal(distance, 0.0)
    return exact_k_medoids(distance, k)


def pivoted_qr_panel(pilot_raw_z: np.ndarray, k: int) -> np.ndarray:
    """Select target columns by rank-revealing QR of the pilot correlation design."""

    pilot_raw_z = np.asarray(pilot_raw_z, dtype=np.float64)
    if pilot_raw_z.ndim != 2 or not 1 <= k < pilot_raw_z.shape[1]:
        raise ValueError("invalid pivoted-QR request")
    # Columns, not ligand rows, are pivoted.  Pilot column standardisation makes
    # this a correlation-scale reconstruction design rather than a variance-scale
    # preference for targets with wide score ranges.
    _, _, pivots = linalg.qr(
        pilot_raw_z,
        mode="economic",
        pivoting=True,
        check_finite=True,
    )
    panel = np.sort(np.asarray(pivots[:k], dtype=np.int64))
    if len(np.unique(panel)) != k:
        raise RuntimeError("pivoted QR returned duplicate targets")
    return panel


def fixed_random_panels(
    p: int, k: int, count: int, rng: np.random.Generator
) -> list[np.ndarray]:
    if count < 1 or not 1 <= k < p:
        raise ValueError("invalid random-panel request")
    panels: set[tuple[int, ...]] = set()
    while len(panels) < count:
        panels.add(tuple(sorted(int(x) for x in rng.choice(p, k, replace=False))))
    return [np.asarray(panel, dtype=np.int64) for panel in sorted(panels)]


def ridge_gcv_predict(
    x_pilot: np.ndarray,
    y_pilot: np.ndarray,
    x_evaluation: np.ndarray,
    alphas: Iterable[float] = RIDGE_ALPHAS,
) -> tuple[np.ndarray, float, float]:
    """Multi-output ridge with pilot-only generalized cross-validation."""

    x_pilot = np.asarray(x_pilot, dtype=np.float64)
    y_pilot = np.asarray(y_pilot, dtype=np.float64)
    x_evaluation = np.asarray(x_evaluation, dtype=np.float64)
    if (
        x_pilot.ndim != 2
        or y_pilot.ndim != 2
        or x_evaluation.ndim != 2
        or len(x_pilot) != len(y_pilot)
        or x_pilot.shape[1] != x_evaluation.shape[1]
        or not np.isfinite(x_pilot).all()
        or not np.isfinite(y_pilot).all()
        or not np.isfinite(x_evaluation).all()
    ):
        raise ValueError("invalid ridge matrices")
    u, singular, vt = np.linalg.svd(x_pilot, full_matrices=False)
    projected = u.T @ y_pilot
    orthogonal_sse = max(
        0.0, float(np.square(y_pilot).sum() - np.square(projected).sum())
    )
    candidates: list[tuple[float, float]] = []
    n, outputs = y_pilot.shape
    for alpha in alphas:
        alpha = float(alpha)
        if alpha <= 0:
            raise ValueError("ridge alphas must be positive")
        shrink_residual = alpha / (np.square(singular) + alpha)
        sse = orthogonal_sse + float(
            np.square(shrink_residual[:, None] * projected).sum()
        )
        degrees = float(np.sum(np.square(singular) / (np.square(singular) + alpha)))
        denominator = max(1e-12, 1.0 - degrees / n)
        gcv = (sse / (n * outputs)) / np.square(denominator)
        candidates.append((gcv, alpha))
    best_gcv, best_alpha = min(candidates, key=lambda pair: (pair[0], pair[1]))
    weights = vt.T @ (
        (singular / (np.square(singular) + best_alpha))[:, None] * projected
    )
    prediction = x_evaluation @ weights
    if not np.isfinite(prediction).all():
        raise RuntimeError("ridge prediction is non-finite")
    return prediction, best_alpha, best_gcv


def selected_pc1_predict(
    x_pilot: np.ndarray, y_pilot: np.ndarray, x_evaluation: np.ndarray
) -> np.ndarray:
    """Pilot-only rank-one predictor derived from the selected target scores."""

    _, _, vt = np.linalg.svd(x_pilot, full_matrices=False)
    loading = vt[0].copy()
    anchor = int(np.argmax(np.abs(loading)))
    if loading[anchor] < 0:
        loading *= -1.0
    score_pilot = x_pilot @ loading
    score_evaluation = x_evaluation @ loading
    denominator = float(score_pilot @ score_pilot)
    if denominator <= 1e-12:
        raise ValueError("selected-target PC1 is degenerate")
    beta = (score_pilot @ y_pilot) / denominator
    return score_evaluation[:, None] * beta[None, :]


def row_profile_pearson(truth: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    truth_centered = truth - truth.mean(axis=1, keepdims=True)
    prediction_centered = prediction - prediction.mean(axis=1, keepdims=True)
    denominator = np.sqrt(
        np.square(truth_centered).sum(axis=1)
        * np.square(prediction_centered).sum(axis=1)
    )
    return np.divide(
        np.sum(truth_centered * prediction_centered, axis=1),
        denominator,
        out=np.zeros(len(truth), dtype=np.float64),
        where=denominator > 1e-14,
    )


def metric_bundle(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    truth = np.asarray(truth, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    if truth.shape != prediction.shape or not np.isfinite(truth).all() or not np.isfinite(
        prediction
    ).all():
        raise ValueError("metric matrices must be aligned and finite")
    sse_columns = np.square(truth - prediction).sum(axis=0)
    sst_columns = np.square(truth - truth.mean(axis=0, keepdims=True)).sum(axis=0)
    if np.any(sst_columns <= 1e-14):
        raise ValueError("an evaluation outcome target is constant")
    target_pearson = [
        0.0
        if np.std(prediction[:, j]) <= 1e-14
        else float(np.corrcoef(truth[:, j], prediction[:, j])[0, 1])
        for j in range(truth.shape[1])
    ]
    row_pearson = row_profile_pearson(truth, prediction)
    return {
        "variance_weighted_r2": float(1.0 - sse_columns.sum() / sst_columns.sum()),
        "pooled_rmse": float(np.sqrt(np.mean(np.square(truth - prediction)))),
        "median_target_pearson": float(np.median(target_pearson)),
        "median_ligand_profile_pearson": float(np.median(row_pearson)),
        "mean_ligand_profile_pearson": float(np.mean(row_pearson)),
    }


def prediction_records(
    *,
    transform: PilotTransform,
    panel: np.ndarray,
    target_names: tuple[str, ...],
    selector: str,
    predictor: str,
    random_draw: int,
    base: dict[str, Any],
) -> list[dict[str, Any]]:
    selected = np.sort(np.asarray(panel, dtype=np.int64))
    omitted = np.setdiff1d(np.arange(len(target_names), dtype=np.int64), selected)
    x_pilot = transform.pilot_raw_z[:, selected]
    x_evaluation = transform.evaluation_raw_z[:, selected]
    if predictor == "multivariate_ridge_gcv":
        # Only omitted targets enter the fitted outcome and GCV loss.  Including
        # selected targets would reward the trivial identity relation even though
        # those columns are copied from observation at deployment.
        predicted_omitted, alpha, gcv = ridge_gcv_predict(
            x_pilot, transform.pilot_raw_z[:, omitted], x_evaluation
        )
    elif predictor == "selected_target_pc1":
        predicted_omitted = selected_pc1_predict(
            x_pilot, transform.pilot_raw_z[:, omitted], x_evaluation
        )
        alpha = -1.0
        gcv = -1.0
    else:
        raise ValueError(f"unknown predictor: {predictor}")
    full_raw_z = np.empty_like(transform.evaluation_raw_z)
    full_raw_z[:, omitted] = predicted_omitted
    full_raw_z[:, selected] = x_evaluation
    # The copy is tested in raw score units, not only standardized units.
    full_raw = full_raw_z * transform.raw_sds + transform.raw_means
    if not np.allclose(
        full_raw[:, selected], transform.evaluation_raw[:, selected], atol=1e-12
    ):
        raise AssertionError("selected observed raw scores were not copied exactly")
    full_residual_raw = full_raw - full_raw.mean(axis=1, keepdims=True)
    full_residual_z = (
        full_residual_raw - transform.residual_means
    ) / transform.residual_sds
    raw_full = metric_bundle(transform.evaluation_raw_z, full_raw_z)
    residual_full = metric_bundle(transform.evaluation_residual_z, full_residual_z)
    raw_omitted = metric_bundle(
        transform.evaluation_raw_z[:, omitted], predicted_omitted
    )
    residual_omitted = metric_bundle(
        transform.evaluation_residual_z[:, omitted], full_residual_z[:, omitted]
    )
    records: list[dict[str, Any]] = []
    for surface, full_metrics, omitted_metrics in (
        ("raw", raw_full, raw_omitted),
        ("raw_row_centered_residual", residual_full, residual_omitted),
    ):
        record: dict[str, Any] = {
            **base,
            "selector": selector,
            "predictor": predictor,
            "random_panel_draw": int(random_draw),
            "surface": surface,
            "panel_indices": "|".join(str(int(value)) for value in selected),
            "panel_targets": "|".join(target_names[int(value)] for value in selected),
            "selected_targets": int(len(selected)),
            "omitted_targets": int(len(omitted)),
            "ridge_alpha": float(alpha),
            "pilot_gcv": float(gcv),
        }
        record.update({f"full_profile_{key}": value for key, value in full_metrics.items()})
        record.update(
            {f"omitted_only_{key}": value for key, value in omitted_metrics.items()}
        )
        records.append(record)
    return records


def descriptor_records(
    *,
    pilot_descriptors: np.ndarray,
    evaluation_descriptors: np.ndarray,
    transform: PilotTransform,
    base: dict[str, Any],
) -> list[dict[str, Any]]:
    means = pilot_descriptors.mean(axis=0)
    sds = pilot_descriptors.std(axis=0, ddof=1)
    if np.any(sds <= 1e-12):
        raise ValueError("a pilot descriptor is constant")
    x_pilot = (pilot_descriptors - means) / sds
    x_evaluation = (evaluation_descriptors - means) / sds
    predicted_raw_z, alpha, gcv = ridge_gcv_predict(
        x_pilot, transform.pilot_raw_z, x_evaluation
    )
    predicted_raw = predicted_raw_z * transform.raw_sds + transform.raw_means
    predicted_residual_raw = predicted_raw - predicted_raw.mean(axis=1, keepdims=True)
    predicted_residual_z = (
        predicted_residual_raw - transform.residual_means
    ) / transform.residual_sds
    records: list[dict[str, Any]] = []
    for surface, truth, prediction in (
        ("raw", transform.evaluation_raw_z, predicted_raw_z),
        (
            "raw_row_centered_residual",
            transform.evaluation_residual_z,
            predicted_residual_z,
        ),
    ):
        metrics = metric_bundle(truth, prediction)
        records.append(
            {
                **base,
                "selector": "descriptor_only_no_docking_panel",
                "predictor": "seven_descriptor_ridge_gcv",
                "random_panel_draw": -1,
                "surface": surface,
                "panel_indices": "",
                "panel_targets": "",
                "selected_targets": 0,
                "omitted_targets": int(truth.shape[1]),
                "ridge_alpha": float(alpha),
                "pilot_gcv": float(gcv),
                **{f"full_profile_{key}": value for key, value in metrics.items()},
                **{
                    f"omitted_only_{key}": value
                    for key, value in metrics.items()
                },
            }
        )
    return records


def fixed_group_folds(dataset: Dataset, folds: int) -> list[tuple[np.ndarray, np.ndarray]]:
    if folds != DEFAULT_FOLDS:
        raise ValueError("the frozen scientific design requires exactly five folds")
    splitter = GroupKFold(n_splits=folds)
    result: list[tuple[np.ndarray, np.ndarray]] = []
    for train, held in splitter.split(dataset.matrix, groups=dataset.groups):
        if set(dataset.groups[train]) & set(dataset.groups[held]):
            raise RuntimeError("chemical-group leakage")
        result.append(
            (np.asarray(train, dtype=np.int64), np.asarray(held, dtype=np.int64))
        )
    return result


def analyze_dataset(
    dataset: Dataset,
    *,
    pilot_sizes: tuple[int, ...],
    ks: tuple[int, ...],
    folds: int,
    max_folds: int | None,
    random_panels: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if random_panels < 50 and max_folds is None:
        raise ValueError("production requires at least 50 random panels per split and k")
    complete = np.isfinite(dataset.matrix).all(axis=1)
    split_records: list[dict[str, Any]] = []
    metric_records: list[dict[str, Any]] = []
    selection_records: list[dict[str, Any]] = []
    all_folds = fixed_group_folds(dataset, folds)
    if max_folds is not None:
        if not 1 <= max_folds <= folds:
            raise ValueError("invalid max-folds diagnostic limit")
        all_folds = all_folds[:max_folds]
    for fold, (training, held) in enumerate(all_folds, start=1):
        complete_training = training[complete[training]]
        complete_held = held[complete[held]]
        if set(dataset.groups[complete_training]) & set(dataset.groups[complete_held]):
            raise RuntimeError("complete-row filtering introduced group overlap")
        if len(complete_held) < 100:
            raise ValueError("too few complete held-out outcomes")
        for pilot_size in pilot_sizes:
            if pilot_size > len(complete_training):
                raise ValueError("pilot size exceeds the complete training pool")
            rng = np.random.default_rng(
                seed
                + (0 if dataset.name == "Docking-44" else 100_000_000)
                + 1_000_000 * fold
                + 10_000 * pilot_size
            )
            pilot = np.sort(
                rng.choice(complete_training, size=pilot_size, replace=False)
            )
            if np.intersect1d(pilot, complete_held).size:
                raise RuntimeError("pilot and held-out rows overlap")
            transform = pilot_transform(dataset.matrix[pilot], dataset.matrix[complete_held])
            base = {
                "dataset": dataset.name,
                "fold": int(fold),
                "pilot_ligands": int(pilot_size),
                "evaluation_ligands": int(len(complete_held)),
                "pilot_complete_rows": True,
                "evaluation_complete_rows": True,
                "pilot_evaluation_row_overlap": 0,
                "pilot_evaluation_group_overlap": 0,
            }
            split_records.append(
                {
                    **base,
                    "training_pool_rows": int(len(training)),
                    "complete_training_pool_rows": int(len(complete_training)),
                    "held_fold_rows_before_complete_filter": int(len(held)),
                    "pilot_groups": int(len(set(dataset.groups[pilot]))),
                    "evaluation_groups": int(len(set(dataset.groups[complete_held]))),
                    "pilot_missing_cells": int(np.isnan(dataset.matrix[pilot]).sum()),
                    "evaluation_missing_cells": int(
                        np.isnan(dataset.matrix[complete_held]).sum()
                    ),
                    "pilot_row_index_sha256": hashlib.sha256(
                        np.asarray(pilot, dtype="<i8").tobytes()
                    ).hexdigest(),
                    "evaluation_row_index_sha256": hashlib.sha256(
                        np.asarray(complete_held, dtype="<i8").tobytes()
                    ).hexdigest(),
                }
            )
            metric_records.extend(
                descriptor_records(
                    pilot_descriptors=dataset.descriptors[pilot],
                    evaluation_descriptors=dataset.descriptors[complete_held],
                    transform=transform,
                    base={**base, "panel_k": 0},
                )
            )
            for k in ks:
                if not 1 <= k < len(dataset.targets):
                    raise ValueError("panel k is invalid")
                selectors = {
                    "pilot_residual_r2_exact_kmedoids": residual_map_medoids(
                        transform, k
                    ),
                    "pilot_raw_correlation_pivoted_qr": pivoted_qr_panel(
                        transform.pilot_raw_z, k
                    ),
                }
                random_rng = np.random.default_rng(
                    seed
                    + (0 if dataset.name == "Docking-44" else 100_000_000)
                    + 1_000_000 * fold
                    + 10_000 * pilot_size
                    + 100 * k
                    + 37
                )
                controls = fixed_random_panels(
                    len(dataset.targets), k, random_panels, random_rng
                )
                for selector, panel in selectors.items():
                    for target_index in panel:
                        selection_records.append(
                            {
                                **base,
                                "panel_k": int(k),
                                "selector": selector,
                                "target_index": int(target_index),
                                "target": dataset.targets[int(target_index)],
                            }
                        )
                    for predictor in (
                        "multivariate_ridge_gcv",
                        "selected_target_pc1",
                    ):
                        metric_records.extend(
                            prediction_records(
                                transform=transform,
                                panel=panel,
                                target_names=dataset.targets,
                                selector=selector,
                                predictor=predictor,
                                random_draw=-1,
                                base={**base, "panel_k": int(k)},
                            )
                        )
                for draw, panel in enumerate(controls):
                    metric_records.extend(
                        prediction_records(
                            transform=transform,
                            panel=panel,
                            target_names=dataset.targets,
                            selector="fixed_random_panel",
                            predictor="multivariate_ridge_gcv",
                            random_draw=draw,
                            base={**base, "panel_k": int(k)},
                        )
                    )
    metrics = pd.DataFrame.from_records(metric_records)
    splits = pd.DataFrame.from_records(split_records)
    selections = pd.DataFrame.from_records(selection_records)
    return metrics, splits, selections


def paired_contrasts(metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Paired common-target contrasts; omitted-only values are not compared."""

    designed = metrics.loc[
        metrics.selector.isin(
            [
                "pilot_residual_r2_exact_kmedoids",
                "pilot_raw_correlation_pivoted_qr",
            ]
        )
    ]
    baseline_rows: list[dict[str, Any]] = []
    selector_rows: list[dict[str, Any]] = []
    keys = ["dataset", "fold", "pilot_ligands", "panel_k", "surface"]
    for values, frame in metrics.loc[metrics.panel_k.gt(0)].groupby(keys, sort=True):
        base = dict(zip(keys, values, strict=True))
        random = frame.loc[
            frame.selector.eq("fixed_random_panel")
            & frame.predictor.eq("multivariate_ridge_gcv")
        ]
        if len(random) < 1:
            raise ValueError("split lacks random controls")
        descriptor = metrics.loc[
            metrics.dataset.eq(base["dataset"])
            & metrics.fold.eq(base["fold"])
            & metrics.pilot_ligands.eq(base["pilot_ligands"])
            & metrics.panel_k.eq(0)
            & metrics.surface.eq(base["surface"])
            & metrics.selector.eq("descriptor_only_no_docking_panel")
        ]
        if len(descriptor) != 1:
            raise ValueError("split lacks a unique descriptor baseline")
        random_values = random[PRIMARY_METRIC].to_numpy(dtype=np.float64)
        for selector in (
            "pilot_residual_r2_exact_kmedoids",
            "pilot_raw_correlation_pivoted_qr",
        ):
            ridge = frame.loc[
                frame.selector.eq(selector)
                & frame.predictor.eq("multivariate_ridge_gcv")
            ]
            pc1 = frame.loc[
                frame.selector.eq(selector) & frame.predictor.eq("selected_target_pc1")
            ]
            if len(ridge) != 1 or len(pc1) != 1:
                raise ValueError("designed selector does not have one ridge and PC1 record")
            observed = float(ridge[PRIMARY_METRIC].iloc[0])
            baseline_rows.append(
                {
                    **base,
                    "selector": selector,
                    "metric": PRIMARY_METRIC,
                    "higher_is_better": True,
                    "designed_value": observed,
                    "random_median": float(np.median(random_values)),
                    "designed_minus_random_median": float(
                        observed - np.median(random_values)
                    ),
                    "fraction_random_panels_no_worse": float(
                        np.mean(random_values >= observed)
                    ),
                    "selected_panel_pc1_value": float(pc1[PRIMARY_METRIC].iloc[0]),
                    "ridge_minus_selected_pc1": float(
                        observed - pc1[PRIMARY_METRIC].iloc[0]
                    ),
                    "descriptor_only_value": float(descriptor[PRIMARY_METRIC].iloc[0]),
                    "ridge_minus_descriptor_only": float(
                        observed - descriptor[PRIMARY_METRIC].iloc[0]
                    ),
                    "random_panels": int(len(random)),
                }
            )
        qr = designed.loc[
            designed.dataset.eq(base["dataset"])
            & designed.fold.eq(base["fold"])
            & designed.pilot_ligands.eq(base["pilot_ligands"])
            & designed.panel_k.eq(base["panel_k"])
            & designed.surface.eq(base["surface"])
            & designed.selector.eq("pilot_raw_correlation_pivoted_qr")
            & designed.predictor.eq("multivariate_ridge_gcv")
        ]
        medoids = designed.loc[
            designed.dataset.eq(base["dataset"])
            & designed.fold.eq(base["fold"])
            & designed.pilot_ligands.eq(base["pilot_ligands"])
            & designed.panel_k.eq(base["panel_k"])
            & designed.surface.eq(base["surface"])
            & designed.selector.eq("pilot_residual_r2_exact_kmedoids")
            & designed.predictor.eq("multivariate_ridge_gcv")
        ]
        if len(qr) != 1 or len(medoids) != 1:
            raise ValueError("selector contrast is not paired")
        selector_rows.append(
            {
                **base,
                "comparison": "pivoted_qr_minus_residual_kmedoids",
                "metric": PRIMARY_METRIC,
                "higher_is_better": True,
                "pivoted_qr_value": float(qr[PRIMARY_METRIC].iloc[0]),
                "residual_kmedoids_value": float(medoids[PRIMARY_METRIC].iloc[0]),
                "pivoted_qr_minus_residual_kmedoids": float(
                    qr[PRIMARY_METRIC].iloc[0] - medoids[PRIMARY_METRIC].iloc[0]
                ),
            }
        )
    return (
        pd.DataFrame.from_records(baseline_rows),
        pd.DataFrame.from_records(selector_rows),
    )


def summarize_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "dataset",
        "pilot_ligands",
        "panel_k",
        "selector",
        "predictor",
        "surface",
    ]
    rows: list[dict[str, Any]] = []
    numeric_metrics = [
        "full_profile_variance_weighted_r2",
        "full_profile_pooled_rmse",
        "full_profile_median_target_pearson",
        "full_profile_median_ligand_profile_pearson",
        "omitted_only_variance_weighted_r2",
        "omitted_only_pooled_rmse",
        "omitted_only_median_target_pearson",
        "omitted_only_median_ligand_profile_pearson",
    ]
    for values, frame in metrics.groupby(keys, sort=True):
        record = dict(zip(keys, values, strict=True))
        record["records"] = int(len(frame))
        record["folds"] = int(frame.fold.nunique())
        for metric in numeric_metrics:
            observed = frame[metric].to_numpy(dtype=np.float64)
            record[f"{metric}_median"] = float(np.median(observed))
            record[f"{metric}_minimum"] = float(np.min(observed))
            record[f"{metric}_maximum"] = float(np.max(observed))
        rows.append(record)
    return pd.DataFrame.from_records(rows)


def summarize_contrasts(contrasts: pd.DataFrame) -> pd.DataFrame:
    keys = ["dataset", "pilot_ligands", "panel_k", "surface", "selector"]
    rows: list[dict[str, Any]] = []
    for values, frame in contrasts.groupby(keys, sort=True):
        record = dict(zip(keys, values, strict=True))
        record["folds"] = int(frame.fold.nunique())
        for metric in (
            "designed_minus_random_median",
            "fraction_random_panels_no_worse",
            "ridge_minus_selected_pc1",
            "ridge_minus_descriptor_only",
        ):
            observed = frame[metric].to_numpy(dtype=np.float64)
            record[f"{metric}_median"] = float(np.median(observed))
            record[f"{metric}_minimum"] = float(np.min(observed))
            record[f"{metric}_maximum"] = float(np.max(observed))
        rows.append(record)
    return pd.DataFrame.from_records(rows)


def summarize_selector_pairs(contrasts: pd.DataFrame) -> pd.DataFrame:
    keys = ["dataset", "pilot_ligands", "panel_k", "surface", "comparison"]
    rows: list[dict[str, Any]] = []
    for values, frame in contrasts.groupby(keys, sort=True):
        observed = frame.pivoted_qr_minus_residual_kmedoids.to_numpy(dtype=np.float64)
        rows.append(
            {
                **dict(zip(keys, values, strict=True)),
                "folds": int(frame.fold.nunique()),
                "pivoted_qr_minus_residual_kmedoids_median": float(
                    np.median(observed)
                ),
                "pivoted_qr_minus_residual_kmedoids_minimum": float(
                    np.min(observed)
                ),
                "pivoted_qr_minus_residual_kmedoids_maximum": float(
                    np.max(observed)
                ),
            }
        )
    return pd.DataFrame.from_records(rows)


def validate_production(
    metrics: pd.DataFrame,
    splits: pd.DataFrame,
    selections: pd.DataFrame,
    *,
    datasets: list[Dataset],
    pilot_sizes: tuple[int, ...],
    ks: tuple[int, ...],
    folds: int,
    random_panels: int,
) -> None:
    if metrics.empty or splits.empty or selections.empty:
        raise RuntimeError("selector control produced an empty table")
    if random_panels < 50:
        raise RuntimeError("production random-panel count is below 50")
    if not (
        splits.pilot_complete_rows.all()
        and splits.evaluation_complete_rows.all()
        and splits.pilot_evaluation_row_overlap.eq(0).all()
        and splits.pilot_evaluation_group_overlap.eq(0).all()
        and splits.pilot_missing_cells.eq(0).all()
        and splits.evaluation_missing_cells.eq(0).all()
    ):
        raise RuntimeError("split contract failed")
    for dataset in datasets:
        subset = metrics.loc[metrics.dataset.eq(dataset.name)]
        if subset.fold.nunique() != folds:
            raise RuntimeError(f"{dataset.name}: not all fixed folds were evaluated")
        for pilot_size in pilot_sizes:
            for k in ks:
                frame = subset.loc[
                    subset.pilot_ligands.eq(pilot_size) & subset.panel_k.eq(k)
                ]
                for fold in range(1, folds + 1):
                    random = frame.loc[
                        frame.fold.eq(fold)
                        & frame.selector.eq("fixed_random_panel")
                        & frame.predictor.eq("multivariate_ridge_gcv")
                        & frame.surface.eq("raw")
                    ]
                    if len(random) != random_panels or random.panel_indices.nunique() != len(
                        random
                    ):
                        raise RuntimeError("random panel count or uniqueness failed")
    numeric = metrics.select_dtypes(include=[np.number]).to_numpy(dtype=np.float64)
    if not np.isfinite(numeric).all():
        raise RuntimeError("metrics contain non-finite values")


def write_artifact(
    output: Path,
    *,
    metrics: pd.DataFrame,
    splits: pd.DataFrame,
    selections: pd.DataFrame,
    baseline_contrasts: pd.DataFrame,
    selector_pair_contrasts: pd.DataFrame,
    metric_summary: pd.DataFrame,
    baseline_contrast_summary: pd.DataFrame,
    selector_pair_summary: pd.DataFrame,
    datasets: list[Dataset],
    parameters: dict[str, Any],
    overwrite: bool,
) -> None:
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent)
    )
    try:
        tables = {
            "panel_metrics.csv.gz": metrics,
            "split_diagnostics.csv": splits,
            "panel_selections.csv": selections,
            "paired_baseline_contrasts.csv": baseline_contrasts,
            "paired_selector_contrasts.csv": selector_pair_contrasts,
            "selector_metric_summary.csv": metric_summary,
            "baseline_contrast_summary.csv": baseline_contrast_summary,
            "selector_pair_summary.csv": selector_pair_summary,
        }
        for filename, table in tables.items():
            table.to_csv(staging / filename, index=False)
        input_hashes = {
            str(dataset.source_path.relative_to(PACKAGE)): sha256_file(
                dataset.source_path
            )
            for dataset in datasets
        }
        payload = {
            "analysis_status": "strict_public_selector_and_panel_size_control",
            "producer": "analysis/public_panel_selector_controls.py",
            "producer_sha256": sha256_file(Path(__file__)),
            "input_sha256": input_hashes,
            "parameters": parameters,
            "contracts": {
                "split": (
                    "five chemical-group-disjoint folds; fully observed pilot sampled "
                    "from four folds; complete held-out outcomes only"
                ),
                "information": (
                    "pilot alone determines scaling, selectors, hyperparameters and fits; "
                    "held-out predictors are selected raw target scores only"
                ),
                "core_estimand": (
                    "impute none because pilot and outcomes are complete; row-centre raw "
                    "scores before residual column scaling"
                ),
                "full_profile_policy": (
                    "predict all raw targets, copy selected observed raw scores, then "
                    "derive residual profile by raw-unit row centring"
                ),
                "primary_comparison": (
                    "all-P full-profile metrics on a common complete held-out truth set"
                ),
                "secondary_comparison": (
                    "omitted-only metrics are selector-specific and not used for direct "
                    "cross-selector claims"
                ),
            },
            "dataset_metadata": {
                dataset.name: dataset.loader_metadata for dataset in datasets
            },
            "table_rows": {filename: int(len(table)) for filename, table in tables.items()},
            "key_results": {
                "designed_panel_baselines": baseline_contrast_summary.to_dict(
                    orient="records"
                ),
                "selector_pair": selector_pair_summary.to_dict(orient="records"),
            },
            "limitations": [
                "The two matrices use Vina-family scores and fixed target panels.",
                "Full-profile reconstruction is score compression, not experimental affinity validation.",
                "The five fold ranges are descriptive composition sensitivities, not population confidence intervals.",
                "The descriptor baseline uses seven inexpensive two-dimensional ligand descriptors only.",
            ],
        }
        write_json(staging / "summary.json", payload)
        (staging / "README.md").write_text(
            """# Public panel-selector and panel-size controls

This fail-closed artifact compares a residual-map exact k-medoids selector with
a reconstruction-aware pivoted-QR selector, 50 fixed random panels per split,
a selected-target PC1 predictor, and a seven-descriptor ridge baseline.  Five
chemical-group folds are evaluated at every reported panel size.  The primary
endpoint is a common all-target profile; omitted-target endpoints are secondary
because different panels omit different targets.

The pilot and held-out evaluation rows are chemically group-disjoint and fully
observed.  Only pilot rows determine preprocessing, selection and fitted models.
Residual profiles are formed by row-centring raw scores before pilot residual
column scaling.

Reproduce with:

```bash
.venv/bin/python analysis/public_panel_selector_controls.py --overwrite
.venv/bin/python -m pytest -q analysis/test_public_panel_selector_controls.py
```
""",
            encoding="utf-8",
        )
        checksum_lines = []
        for path in sorted(staging.iterdir()):
            if path.name == "checksums.sha256":
                continue
            checksum_lines.append(f"{sha256_file(path)}  {path.name}")
        (staging / "checksums.sha256").write_text(
            "\n".join(checksum_lines) + "\n", encoding="utf-8"
        )
        if output.exists():
            if not overwrite:
                raise FileExistsError(f"output exists; pass --overwrite: {output}")
            if output.parent != DEFAULT_OUTPUT.resolve().parent and output.name.startswith(
                "."
            ):
                raise ValueError("refusing unsafe overwrite target")
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
    datasets = load_datasets(
        args.docking44,
        args.dockstring,
        requested=args.datasets,
    )
    metric_tables: list[pd.DataFrame] = []
    split_tables: list[pd.DataFrame] = []
    selection_tables: list[pd.DataFrame] = []
    for dataset in datasets:
        metrics, splits, selections = analyze_dataset(
            dataset,
            pilot_sizes=tuple(args.pilot_sizes),
            ks=tuple(args.ks),
            folds=args.folds,
            max_folds=args.max_folds,
            random_panels=args.random_panels,
            seed=args.seed,
        )
        metric_tables.append(metrics)
        split_tables.append(splits)
        selection_tables.append(selections)
    metrics = pd.concat(metric_tables, ignore_index=True)
    splits = pd.concat(split_tables, ignore_index=True)
    selections = pd.concat(selection_tables, ignore_index=True)
    baseline_contrasts, selector_pair_contrasts = paired_contrasts(metrics)
    metric_summary = summarize_metrics(metrics)
    baseline_contrast_summary = summarize_contrasts(baseline_contrasts)
    selector_pair_summary = summarize_selector_pairs(selector_pair_contrasts)
    if args.max_folds is None:
        validate_production(
            metrics,
            splits,
            selections,
            datasets=datasets,
            pilot_sizes=tuple(args.pilot_sizes),
            ks=tuple(args.ks),
            folds=args.folds,
            random_panels=args.random_panels,
        )
    write_artifact(
        args.output,
        metrics=metrics,
        splits=splits,
        selections=selections,
        baseline_contrasts=baseline_contrasts,
        selector_pair_contrasts=selector_pair_contrasts,
        metric_summary=metric_summary,
        baseline_contrast_summary=baseline_contrast_summary,
        selector_pair_summary=selector_pair_summary,
        datasets=datasets,
        parameters={
            "pilot_sizes": list(args.pilot_sizes),
            "panel_sizes": list(args.ks),
            "folds": args.folds,
            "max_folds_diagnostic": args.max_folds,
            "random_panels_per_split_and_k": args.random_panels,
            "seed": args.seed,
            "ridge_alphas": list(RIDGE_ALPHAS),
            "descriptor_names": list(DESCRIPTOR_NAMES),
            "primary_metric": PRIMARY_METRIC,
        },
        overwrite=args.overwrite,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docking44", type=Path, default=DEFAULT_DOCKING44)
    parser.add_argument("--dockstring", type=Path, default=DEFAULT_DOCKSTRING)
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=("Docking-44", "DOCKSTRING-58"),
        default=("Docking-44", "DOCKSTRING-58"),
    )
    parser.add_argument(
        "--pilot-sizes", nargs="+", type=int, default=DEFAULT_PILOT_SIZES
    )
    parser.add_argument("--ks", nargs="+", type=int, default=DEFAULT_KS)
    parser.add_argument("--folds", type=int, default=DEFAULT_FOLDS)
    parser.add_argument(
        "--max-folds",
        type=int,
        default=None,
        help="diagnostic only; production omits this option",
    )
    parser.add_argument("--random-panels", type=int, default=DEFAULT_RANDOM_PANELS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
