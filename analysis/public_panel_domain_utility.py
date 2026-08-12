#!/usr/bin/env python3
"""Strict-public utility tests for Vina target-correlation maps.

This producer asks three deliberately operational questions.

1. Can a target panel selected from a 200- or 500-ligand pilot reconstruct the
   omitted target scores of row-disjoint future ligands when only the selected
   target scores are available at prediction time?
2. When target-map coverage is the objective, how far is a pilot-selected exact
   k-medoids panel from the exact optimum on the row-disjoint complement?
3. For low- and high-molecular-weight domains, is a pilot sampled inside the
   future domain more faithful to a common row-disjoint domain map than an
   equal-size source-library or opposite-domain pilot?

All preprocessing and predictive-model fitting is calibration-only.  The
evaluation all-target row mean is used solely to define a residual *outcome*;
it is never a predictor.  Full-source and complement-specific panels are
explicitly non-deployable diagnostic ceilings.  The analysis uses only the two
public Vina matrices shipped in the repository and makes no claim about
experimental affinity, pose quality, or biological target coverage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from rdkit import rdBase
from scipy import stats
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix
from sklearn.linear_model import Ridge, RidgeCV

try:
    from .residual_mechanism_analysis import (
        DESCRIPTOR_NAMES,
        DOCK44,
        molecular_descriptors,
    )
except ImportError:  # pragma: no cover - direct execution
    from residual_mechanism_analysis import (  # type: ignore
        DESCRIPTOR_NAMES,
        DOCK44,
        molecular_descriptors,
    )


PACKAGE = Path(__file__).resolve().parents[1]
FROZEN = PACKAGE / "data" / "frozen"
DEFAULT_DOCKING44 = FROZEN / "df_final_v4.csv.gz"
DEFAULT_DOCKSTRING = FROZEN / "dockstring-dataset.tsv.gz"
DEFAULT_OUTPUT = PACKAGE / "results" / "public_panel_domain_utility"
DEFAULT_SIZES = (200, 500)
DEFAULT_REPETITIONS = 20
DEFAULT_PANEL_K = 8
DEFAULT_RANDOM_RECONSTRUCTION_PANELS = 3
DEFAULT_RANDOM_COVERAGE_PANELS = 1_000
DEFAULT_SEED = 202_608_24
DOCKSTRING_SUPPORT_SIZE = 15_000
DOCKSTRING_SUPPORT_SEED = 71
RIDGE_ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0)
DISTANCE_OBJECTIVES = ("residual_r2", "residual_absolute", "residual_signed")
PREDICTIVE_PANEL_METHODS = (
    "residual_r2",
    "residual_absolute",
    "residual_signed",
    "raw_r2",
)


@dataclass(frozen=True)
class PublicDataset:
    name: str
    matrix: np.ndarray
    targets: tuple[str, ...]
    molecular_weight: np.ndarray
    descriptor_matrix: np.ndarray
    support_rule: str
    input_paths: tuple[Path, ...]


@dataclass(frozen=True)
class SplitTransform:
    calibration_raw: np.ndarray
    evaluation_raw: np.ndarray
    calibration_raw_z: np.ndarray
    evaluation_raw_z: np.ndarray
    calibration_residual_raw: np.ndarray
    evaluation_residual_raw: np.ndarray
    calibration_residual_z: np.ndarray
    evaluation_residual_z: np.ndarray
    raw_means: np.ndarray
    raw_sds: np.ndarray
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


def load_docking44(path: Path = DEFAULT_DOCKING44) -> PublicDataset:
    columns = list(DOCK44) + ["Cleaned SMILES", "Canonical SMILES"]
    frame = pd.read_csv(path, usecols=columns)
    numeric = frame[list(DOCK44)].apply(pd.to_numeric, errors="coerce").clip(upper=0)
    matrix = numeric.to_numpy(dtype=np.float64)
    if np.any(np.all(np.isnan(matrix), axis=0)):
        raise ValueError("Docking-44 contains an entirely missing target")
    smiles = frame["Cleaned SMILES"].fillna(frame["Canonical SMILES"]).astype(str)
    descriptor_matrix = molecular_descriptors(smiles)
    molecular_weight = descriptor_matrix[
        :, DESCRIPTOR_NAMES.index("molecular_weight")
    ]
    return PublicDataset(
        name="Docking-44",
        matrix=matrix,
        targets=tuple(DOCK44),
        molecular_weight=molecular_weight,
        descriptor_matrix=descriptor_matrix,
        support_rule=(
            "all 12,651 rows; positive scores clipped at zero; split-local target-mean "
            "imputation"
        ),
        input_paths=(path,),
    )


def load_fixed_dockstring(path: Path = DEFAULT_DOCKSTRING) -> PublicDataset:
    if path != DEFAULT_DOCKSTRING:
        raise ValueError("the strict-public loader accepts only the frozen DOCKSTRING path")
    frame = pd.read_csv(path, sep="\t")
    targets = [column for column in frame if column not in {"inchikey", "smiles"}]
    numeric = frame[targets].apply(pd.to_numeric, errors="coerce")
    complete = ~numeric.isna().any(axis=1)
    complete_matrix = np.minimum(
        numeric.loc[complete].to_numpy(dtype=np.float64), 0.0
    )
    complete_smiles = frame.loc[complete, "smiles"].reset_index(drop=True).astype(str)
    rng = np.random.default_rng(DOCKSTRING_SUPPORT_SEED)
    support = np.sort(
        rng.choice(
            len(complete_matrix), size=DOCKSTRING_SUPPORT_SIZE, replace=False
        )
    )
    matrix = complete_matrix[support]
    descriptor_matrix = molecular_descriptors(complete_smiles.iloc[support])
    molecular_weight = descriptor_matrix[
        :, DESCRIPTOR_NAMES.index("molecular_weight")
    ]
    return PublicDataset(
        name="DOCKSTRING-58",
        matrix=matrix,
        targets=tuple(targets),
        molecular_weight=np.asarray(molecular_weight, dtype=np.float64),
        descriptor_matrix=descriptor_matrix,
        support_rule=(
            "sorted simple random sample without replacement of 15,000 complete "
            "ligands; NumPy seed 71; positive scores clipped at zero"
        ),
        input_paths=(path,),
    )


def calibration_transform(
    calibration: np.ndarray, evaluation: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Impute and standardize both arrays from calibration rows only."""

    calibration = np.asarray(calibration, dtype=np.float64)
    evaluation = np.asarray(evaluation, dtype=np.float64)
    if calibration.ndim != 2 or evaluation.ndim != 2:
        raise ValueError("calibration and evaluation must be matrices")
    if calibration.shape[1] != evaluation.shape[1] or len(calibration) < 6:
        raise ValueError("split matrices have incompatible dimensions")
    means = np.nanmean(calibration, axis=0)
    if not np.isfinite(means).all():
        raise ValueError("a calibration target is entirely missing")
    filled_calibration = np.where(np.isnan(calibration), means, calibration)
    filled_evaluation = np.where(np.isnan(evaluation), means, evaluation)
    if not np.isfinite(filled_calibration).all() or not np.isfinite(
        filled_evaluation
    ).all():
        raise ValueError("split-local imputation failed")
    standard_deviations = filled_calibration.std(axis=0, ddof=1)
    if np.any(standard_deviations <= 1e-12):
        raise ValueError("a calibration target has negligible variance")
    return (
        (filled_calibration - means) / standard_deviations,
        (filled_evaluation - means) / standard_deviations,
        means,
        standard_deviations,
    )


def calibration_standardize_features(
    calibration: np.ndarray, evaluation: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Standardize complete predictor features using calibration rows only."""

    calibration = np.asarray(calibration, dtype=np.float64)
    evaluation = np.asarray(evaluation, dtype=np.float64)
    if (
        calibration.ndim != 2
        or evaluation.ndim != 2
        or calibration.shape[1] != evaluation.shape[1]
        or len(calibration) < 6
        or not np.isfinite(calibration).all()
        or not np.isfinite(evaluation).all()
    ):
        raise ValueError("feature split matrices are invalid")
    means = calibration.mean(axis=0)
    sds = calibration.std(axis=0, ddof=1)
    if np.any(sds <= 1e-12):
        raise ValueError("a calibration feature is constant")
    return (
        (calibration - means) / sds,
        (evaluation - means) / sds,
        means,
        sds,
    )


def split_transform(
    calibration: np.ndarray, evaluation: np.ndarray
) -> SplitTransform:
    """Core estimand: impute raw scores, row-center raw, then scale columns."""

    raw_calibration_z, raw_evaluation_z, raw_means, raw_sds = calibration_transform(
        calibration, evaluation
    )
    calibration_raw = raw_calibration_z * raw_sds + raw_means
    evaluation_raw = raw_evaluation_z * raw_sds + raw_means
    calibration_residual_raw = calibration_raw - calibration_raw.mean(
        axis=1, keepdims=True
    )
    evaluation_residual_raw = evaluation_raw - evaluation_raw.mean(
        axis=1, keepdims=True
    )
    residual_means = calibration_residual_raw.mean(axis=0)
    residual_sds = calibration_residual_raw.std(axis=0, ddof=1)
    if np.any(residual_sds <= 1e-12):
        raise ValueError("a raw-row-centred residual target has negligible variance")
    return SplitTransform(
        calibration_raw=calibration_raw,
        evaluation_raw=evaluation_raw,
        calibration_raw_z=raw_calibration_z,
        evaluation_raw_z=raw_evaluation_z,
        calibration_residual_raw=calibration_residual_raw,
        evaluation_residual_raw=evaluation_residual_raw,
        calibration_residual_z=(
            calibration_residual_raw - residual_means
        )
        / residual_sds,
        evaluation_residual_z=(evaluation_residual_raw - residual_means) / residual_sds,
        raw_means=raw_means,
        raw_sds=raw_sds,
        residual_means=residual_means,
        residual_sds=residual_sds,
    )


def target_correlation(surface: np.ndarray) -> np.ndarray:
    surface = np.asarray(surface, dtype=np.float64)
    if surface.ndim != 2 or len(surface) < 3 or not np.isfinite(surface).all():
        raise ValueError("target correlation requires a finite matrix with >=3 rows")
    centered = surface - surface.mean(axis=0, keepdims=True)
    cross = centered.T @ centered
    variance = np.diag(cross)
    if np.any(variance <= 1e-14):
        raise ValueError("target correlation contains a constant target")
    result = cross / np.sqrt(np.outer(variance, variance))
    result = np.clip((result + result.T) / 2.0, -1.0, 1.0)
    np.fill_diagonal(result, 1.0)
    return result


def map_geometries(filled_raw: np.ndarray) -> dict[str, np.ndarray]:
    """Raw correlation and core raw-row-centred residual correlation maps."""

    residual = filled_raw - filled_raw.mean(axis=1, keepdims=True)
    return {
        "raw": target_correlation(filled_raw),
        "residual": target_correlation(residual),
    }


def dissimilarity(correlation: np.ndarray, objective: str) -> np.ndarray:
    correlation = np.asarray(correlation, dtype=np.float64)
    if objective.endswith("_r2"):
        distance = 1.0 - np.square(correlation)
    elif objective.endswith("_absolute"):
        distance = 1.0 - np.abs(correlation)
    elif objective.endswith("_signed"):
        distance = 1.0 - correlation
    else:
        raise ValueError(f"unknown panel objective: {objective}")
    distance = np.maximum((distance + distance.T) / 2.0, 0.0)
    np.fill_diagonal(distance, 0.0)
    return distance


def objective_map(geometries: dict[str, np.ndarray], objective: str) -> np.ndarray:
    surface = "raw" if objective.startswith("raw_") else "residual"
    return dissimilarity(geometries[surface], objective)


def exact_k_medoids(distance: np.ndarray, k: int) -> np.ndarray:
    """Exact discrete p-median solution using HiGHS MILP."""

    distance = np.asarray(distance, dtype=np.float64)
    n = len(distance)
    if (
        distance.shape != (n, n)
        or not np.isfinite(distance).all()
        or np.any(distance < -1e-12)
        or not 1 <= k < n
    ):
        raise ValueError("invalid exact k-medoids problem")
    assignments = n * n
    variables = assignments + n
    coefficients = np.zeros(variables, dtype=np.float64)
    coefficients[:assignments] = distance.reshape(-1)
    coefficients[assignments:] = 1e-12 * np.arange(n)
    integrality = np.zeros(variables, dtype=np.int8)
    integrality[assignments:] = 1
    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []
    lower: list[float] = []
    upper: list[float] = []
    constraint = 0
    for target in range(n):
        for representative in range(n):
            rows.append(constraint)
            columns.append(target * n + representative)
            values.append(1.0)
        lower.append(1.0)
        upper.append(1.0)
        constraint += 1
    for target in range(n):
        for representative in range(n):
            rows.extend((constraint, constraint))
            columns.extend(
                (target * n + representative, assignments + representative)
            )
            values.extend((1.0, -1.0))
            lower.append(-np.inf)
            upper.append(0.0)
            constraint += 1
    for representative in range(n):
        rows.append(constraint)
        columns.append(assignments + representative)
        values.append(1.0)
    lower.append(float(k))
    upper.append(float(k))
    constraint += 1
    constraints = coo_matrix(
        (values, (rows, columns)), shape=(constraint, variables), dtype=np.float64
    ).tocsr()
    result = milp(
        c=coefficients,
        integrality=integrality,
        bounds=Bounds(np.zeros(variables), np.ones(variables)),
        constraints=LinearConstraint(
            constraints,
            np.asarray(lower, dtype=np.float64),
            np.asarray(upper, dtype=np.float64),
        ),
        options={"presolve": True, "mip_rel_gap": 0.0},
    )
    if not result.success or result.x is None:
        raise RuntimeError(f"exact k-medoids failed: {result.message}")
    selected = np.flatnonzero(result.x[assignments:] > 0.5).astype(np.int64)
    if len(selected) != k:
        raise RuntimeError("exact k-medoids returned the wrong panel size")
    return selected


def coverage_loss(distance: np.ndarray, panel: np.ndarray) -> float:
    distance = np.asarray(distance, dtype=np.float64)
    panel = np.asarray(panel, dtype=np.int64)
    if panel.ndim != 1 or not len(panel):
        raise ValueError("panel is empty")
    return float(np.min(distance[:, panel], axis=1).mean())


def map_spearman(first: np.ndarray, second: np.ndarray) -> float:
    indices = np.triu_indices(len(first), k=1)
    value = float(stats.spearmanr(first[indices], second[indices]).statistic)
    if not np.isfinite(value):
        raise ValueError("map agreement is non-finite")
    return value


def random_panels(
    targets: int, k: int, repeats: int, rng: np.random.Generator
) -> np.ndarray:
    if repeats < 1 or not 1 <= k < targets:
        raise ValueError("invalid random-panel request")
    return np.asarray(
        [np.sort(rng.choice(targets, size=k, replace=False)) for _ in range(repeats)],
        dtype=np.int16,
    )


def finite_pearson(first: np.ndarray, second: np.ndarray) -> float:
    if np.std(first) <= 1e-14 or np.std(second) <= 1e-14:
        return 0.0
    return float(np.corrcoef(first, second)[0, 1])


def finite_spearman(first: np.ndarray, second: np.ndarray) -> float:
    if np.std(first) <= 1e-14 or np.std(second) <= 1e-14:
        return 0.0
    return float(stats.spearmanr(first, second).statistic)


def predictive_metrics(
    truth: np.ndarray,
    prediction: np.ndarray,
    target_names: Iterable[str],
) -> tuple[dict[str, float], pd.DataFrame]:
    truth = np.asarray(truth, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    names = tuple(target_names)
    if truth.shape != prediction.shape or truth.shape[1] != len(names):
        raise ValueError("truth, predictions and target names do not align")
    records: list[dict[str, Any]] = []
    total_sse = 0.0
    total_sst = 0.0
    for column, name in enumerate(names):
        observed = truth[:, column]
        fitted = prediction[:, column]
        sse = float(np.square(observed - fitted).sum())
        sst = float(np.square(observed - observed.mean()).sum())
        if sst <= 1e-14:
            raise ValueError("an evaluation target is constant")
        total_sse += sse
        total_sst += sst
        records.append(
            {
                "target": name,
                "r2": 1.0 - sse / sst,
                "calibration_standardized_rmse": float(
                    np.sqrt(np.mean(np.square(observed - fitted)))
                ),
                "pearson": finite_pearson(observed, fitted),
                "spearman": finite_spearman(observed, fitted),
            }
        )
    frame = pd.DataFrame.from_records(records)
    aggregate = {
        "variance_weighted_r2": 1.0 - total_sse / total_sst,
        "mean_target_r2": float(frame.r2.mean()),
        "median_target_r2": float(frame.r2.median()),
        "pooled_calibration_standardized_rmse": float(
            np.sqrt(np.mean(np.square(truth - prediction)))
        ),
        "mean_target_pearson": float(frame.pearson.mean()),
        "median_target_pearson": float(frame.pearson.median()),
        "mean_target_spearman": float(frame.spearman.mean()),
        "fraction_targets_positive_r2": float(np.mean(frame.r2 > 0.0)),
    }
    return aggregate, frame


def predictive_metrics_observed_cells(
    truth: np.ndarray, prediction: np.ndarray, observed: np.ndarray
) -> dict[str, float]:
    """Aggregate targetwise errors using only originally observed outcome cells."""

    if truth.shape != prediction.shape or truth.shape != observed.shape:
        raise ValueError("observed-cell metric arrays do not align")
    total_sse = 0.0
    total_sst = 0.0
    target_r2: list[float] = []
    target_rmse: list[float] = []
    for column in range(truth.shape[1]):
        mask = observed[:, column]
        if mask.sum() < 3:
            raise ValueError("too few observed outcomes for a target")
        target_truth = truth[mask, column]
        target_prediction = prediction[mask, column]
        sse = float(np.square(target_truth - target_prediction).sum())
        sst = float(np.square(target_truth - target_truth.mean()).sum())
        if sst <= 1e-14:
            raise ValueError("an observed-cell target is constant")
        total_sse += sse
        total_sst += sst
        target_r2.append(1.0 - sse / sst)
        target_rmse.append(float(np.sqrt(sse / mask.sum())))
    return {
        "variance_weighted_r2": 1.0 - total_sse / total_sst,
        "mean_target_r2": float(np.mean(target_r2)),
        "median_target_r2": float(np.median(target_r2)),
        "mean_target_calibration_standardized_rmse": float(np.mean(target_rmse)),
        "observed_cells": int(observed.sum()),
        "possible_cells": int(observed.size),
    }


def row_profile_metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    if truth.shape != prediction.shape:
        raise ValueError("profile matrices do not align")

    def row_pearson(first: np.ndarray, second: np.ndarray) -> np.ndarray:
        first_centered = first - first.mean(axis=1, keepdims=True)
        second_centered = second - second.mean(axis=1, keepdims=True)
        denominator = np.sqrt(
            np.square(first_centered).sum(axis=1)
            * np.square(second_centered).sum(axis=1)
        )
        return np.divide(
            np.sum(first_centered * second_centered, axis=1),
            denominator,
            out=np.zeros(len(first), dtype=np.float64),
            where=denominator > 1e-14,
        )

    pearson = row_pearson(truth, prediction)
    truth_rank = stats.rankdata(truth, axis=1, method="average")
    prediction_rank = stats.rankdata(prediction, axis=1, method="average")
    spearman = row_pearson(truth_rank, prediction_rank)
    return {
        "median_ligand_profile_pearson": float(np.median(pearson)),
        "mean_ligand_profile_pearson": float(np.mean(pearson)),
        "median_ligand_profile_spearman": float(np.median(spearman)),
        "mean_ligand_profile_spearman": float(np.mean(spearman)),
    }


def lower_tail_metrics(
    calibration_truth: np.ndarray,
    evaluation_truth: np.ndarray,
    evaluation_prediction: np.ndarray,
    fraction: float = 0.05,
) -> tuple[dict[str, float], pd.DataFrame]:
    """Lower-tail retrieval using pilot thresholds and equal held-out budgets."""

    if not 0 < fraction < 0.5 or calibration_truth.shape[1] != evaluation_truth.shape[1]:
        raise ValueError("invalid favorable-tail inputs")
    thresholds = np.quantile(calibration_truth, fraction, axis=0)
    truth = evaluation_truth <= thresholds
    predicted = evaluation_prediction <= thresholds
    true_positive = np.sum(truth & predicted, axis=0)
    predicted_positive = np.sum(predicted, axis=0)
    actual_positive = np.sum(truth, axis=0)
    precision = np.divide(
        true_positive,
        predicted_positive,
        out=np.zeros_like(true_positive, dtype=np.float64),
        where=predicted_positive > 0,
    )
    recall = np.divide(
        true_positive,
        actual_positive,
        out=np.zeros_like(true_positive, dtype=np.float64),
        where=actual_positive > 0,
    )
    budget = max(1, int(np.ceil(fraction * len(evaluation_truth))))
    budget_overlap = np.empty(evaluation_truth.shape[1], dtype=np.float64)
    for column in range(evaluation_truth.shape[1]):
        true_order = np.argsort(evaluation_truth[:, column], kind="mergesort")[:budget]
        predicted_order = np.argsort(
            evaluation_prediction[:, column], kind="mergesort"
        )[:budget]
        budget_overlap[column] = len(np.intersect1d(true_order, predicted_order)) / budget
    frame = pd.DataFrame(
        {
            "calibration_threshold_lower_5pct": thresholds,
            "calibration_threshold_lower_5pct_true_count": actual_positive,
            "calibration_threshold_lower_5pct_predicted_count": predicted_positive,
            "calibration_threshold_lower_5pct_true_positive": true_positive,
            "calibration_threshold_lower_5pct_precision": precision,
            "calibration_threshold_lower_5pct_recall": recall,
            "equal_budget_lower_5pct_count": budget,
            "equal_budget_lower_5pct_overlap": budget_overlap,
        }
    )
    aggregate = {
        "median_target_calibration_threshold_lower_5pct_precision": float(
            np.median(precision)
        ),
        "median_target_calibration_threshold_lower_5pct_recall": float(
            np.median(recall)
        ),
        "micro_calibration_threshold_lower_5pct_precision": float(
            true_positive.sum() / max(1, predicted_positive.sum())
        ),
        "micro_calibration_threshold_lower_5pct_recall": float(
            true_positive.sum() / max(1, actual_positive.sum())
        ),
        "median_target_equal_budget_lower_5pct_overlap": float(
            np.median(budget_overlap)
        ),
        "mean_target_equal_budget_lower_5pct_overlap": float(
            np.mean(budget_overlap)
        ),
        "fraction_targets_with_no_calibration_threshold_evaluation_members": float(
            np.mean(actual_positive == 0)
        ),
    }
    return aggregate, frame


def fit_reconstruction(
    transform: SplitTransform,
    panel: np.ndarray,
    target_names: tuple[str, ...],
    calibration_observed: np.ndarray | None = None,
    evaluation_observed: np.ndarray | None = None,
) -> tuple[list[dict[str, Any]], list[pd.DataFrame]]:
    """Predict raw and residual target profiles from selected raw target scores."""

    selected = np.asarray(panel, dtype=np.int64)
    omitted = np.setdiff1d(np.arange(transform.calibration_raw_z.shape[1]), selected)
    if calibration_observed is None:
        calibration_observed = np.ones_like(transform.calibration_raw_z, dtype=bool)
    if evaluation_observed is None:
        evaluation_observed = np.ones_like(transform.evaluation_raw_z, dtype=bool)
    calibration_observed = np.asarray(calibration_observed, dtype=bool)
    evaluation_observed = np.asarray(evaluation_observed, dtype=bool)
    if (
        calibration_observed.shape != transform.calibration_raw_z.shape
        or evaluation_observed.shape != transform.evaluation_raw_z.shape
    ):
        raise ValueError("original observation masks do not align")
    calibration_complete = calibration_observed.all(axis=1)
    evaluation_complete = evaluation_observed.all(axis=1)
    if calibration_complete.sum() < 20 or evaluation_complete.sum() < 20:
        raise ValueError("too few originally complete rows for reconstruction sensitivity")
    x_calibration = transform.calibration_raw_z[:, selected]
    x_evaluation = transform.evaluation_raw_z[:, selected]
    _, _, pc1_vt = np.linalg.svd(x_calibration, full_matrices=False)
    pc1_loading = pc1_vt[0]
    anchor = int(np.argmax(np.abs(pc1_loading)))
    if pc1_loading[anchor] < 0:
        pc1_loading = -pc1_loading
    pc1_calibration = x_calibration @ pc1_loading
    pc1_evaluation = x_evaluation @ pc1_loading
    pc1_denominator = float(pc1_calibration @ pc1_calibration)
    if pc1_denominator <= 1e-12:
        raise ValueError("selected-target PC1 has negligible pilot variance")

    def predictive_models(
        y_calibration: np.ndarray,
    ) -> dict[str, tuple[np.ndarray, np.ndarray, float]]:
        ridge = RidgeCV(
            alphas=np.asarray(RIDGE_ALPHAS, dtype=np.float64),
            fit_intercept=False,
            cv=None,
            gcv_mode="svd",
            scoring="neg_mean_squared_error",
        )
        ridge.fit(x_calibration, y_calibration[:, omitted])
        ridge_omitted = ridge.predict(x_evaluation)
        ridge_policy = Ridge(alpha=float(ridge.alpha_), fit_intercept=False)
        ridge_policy.fit(x_calibration, y_calibration)
        ridge_all = ridge_policy.predict(x_evaluation)
        pc1_beta_omitted = (
            pc1_calibration @ y_calibration[:, omitted]
        ) / pc1_denominator
        pc1_beta_all = (pc1_calibration @ y_calibration) / pc1_denominator
        return {
            "multivariate_ridge": (
                ridge_omitted,
                ridge_all,
                float(ridge.alpha_),
            ),
            "selected_target_pc1": (
                pc1_evaluation[:, None] * pc1_beta_omitted[None, :],
                pc1_evaluation[:, None] * pc1_beta_all[None, :],
                -1.0,
            ),
        }

    aggregates: list[dict[str, Any]] = []
    target_frames: list[pd.DataFrame] = []
    raw_predictions = predictive_models(transform.calibration_raw_z)
    for predictive_model, (
        predicted_raw_omitted,
        predicted_raw_all,
        ridge_alpha,
    ) in raw_predictions.items():
        raw_policy_prediction = predicted_raw_all.copy()
        raw_policy_prediction[:, selected] = x_evaluation
        observed_omitted_raw = predictive_metrics_observed_cells(
            transform.evaluation_raw_z[:, omitted],
            predicted_raw_omitted,
            evaluation_observed[:, omitted],
        )
        observed_policy_raw = predictive_metrics_observed_cells(
            transform.evaluation_raw_z,
            raw_policy_prediction,
            evaluation_observed,
        )
        observed_raw_metrics = {
            **{
                f"raw_originally_observed_cell_omitted_{key}": value
                for key, value in observed_omitted_raw.items()
            },
            **{
                f"raw_originally_observed_cell_full_policy_{key}": value
                for key, value in observed_policy_raw.items()
            },
        }
        predicted_raw = (
            raw_policy_prediction * transform.raw_sds + transform.raw_means
        )
        predicted_residual_raw = predicted_raw - predicted_raw.mean(
            axis=1, keepdims=True
        )
        residual_policy_prediction = (
            predicted_residual_raw - transform.residual_means
        ) / transform.residual_sds
        surface_models = (
            (
                "calibration_standardized_raw",
                transform.calibration_raw_z,
                transform.evaluation_raw_z,
                predicted_raw_omitted,
                raw_policy_prediction,
            ),
            (
                "raw_row_centered_residual_outcome",
                transform.calibration_residual_z,
                transform.evaluation_residual_z,
                residual_policy_prediction[:, omitted],
                residual_policy_prediction,
            ),
        )
        for (
            truth_surface,
            y_calibration,
            y_evaluation,
            predicted_omitted,
            policy_prediction,
        ) in surface_models:
            for outcome_basis, calibration_rows, evaluation_rows in (
                (
                    "split_imputed_all_rows",
                    np.ones(len(y_calibration), dtype=bool),
                    np.ones(len(y_evaluation), dtype=bool),
                ),
                (
                    "originally_complete_outcome_rows",
                    calibration_complete,
                    evaluation_complete,
                ),
            ):
                basis_truth = y_evaluation[evaluation_rows][:, omitted]
                basis_prediction = predicted_omitted[evaluation_rows]
                aggregate, target_frame = predictive_metrics(
                    basis_truth,
                    basis_prediction,
                    [target_names[index] for index in omitted],
                )
                zero_aggregate, _ = predictive_metrics(
                    basis_truth,
                    np.zeros_like(basis_truth),
                    [target_names[index] for index in omitted],
                )
                # Row-wise target-profile agreement is evaluated on omitted targets only,
                # so observed inputs cannot inflate it by being copied into prediction.
                aggregate.update(row_profile_metrics(basis_truth, basis_prediction))
                policy_truth = y_evaluation[evaluation_rows]
                basis_policy_prediction = policy_prediction[evaluation_rows]
                policy_aggregate, _ = predictive_metrics(
                    policy_truth,
                    basis_policy_prediction,
                    target_names,
                )
                policy_profile = row_profile_metrics(
                    policy_truth, basis_policy_prediction
                )
                for metric, value in policy_aggregate.items():
                    aggregate[f"full_profile_policy_{metric}"] = value
                for metric, value in policy_profile.items():
                    aggregate[f"full_profile_policy_{metric}"] = value
                policy_tail, _ = lower_tail_metrics(
                    y_calibration[calibration_rows],
                    policy_truth,
                    basis_policy_prediction,
                )
                for metric, value in policy_tail.items():
                    aggregate[f"full_profile_policy_{metric}"] = value
                tail_aggregate, tail_frame = lower_tail_metrics(
                    y_calibration[calibration_rows][:, omitted],
                    basis_truth,
                    basis_prediction,
                )
                aggregate.update(tail_aggregate)
                target_frame = pd.concat(
                    [
                        target_frame.reset_index(drop=True),
                        tail_frame.reset_index(drop=True),
                    ],
                    axis=1,
                )
                aggregate.update(
                    {
                        "truth_surface": truth_surface,
                        "outcome_basis": outcome_basis,
                        "predictive_model": predictive_model,
                        "ridge_alpha": ridge_alpha,
                        "latent_factors": int(
                            predictive_model == "selected_target_pc1"
                        ),
                        "selected_targets": int(len(selected)),
                        "omitted_targets": int(len(omitted)),
                        "calibration_rows_scored": int(calibration_rows.sum()),
                        "evaluation_rows_scored": int(evaluation_rows.sum()),
                        "zero_baseline_variance_weighted_r2": zero_aggregate[
                            "variance_weighted_r2"
                        ],
                        "zero_baseline_pooled_calibration_standardized_rmse": (
                            zero_aggregate[
                                "pooled_calibration_standardized_rmse"
                            ]
                        ),
                        **observed_raw_metrics,
                    }
                )
                target_frame.insert(0, "predictive_model", predictive_model)
                target_frame.insert(0, "outcome_basis", outcome_basis)
                target_frame.insert(0, "truth_surface", truth_surface)
                target_frames.append(target_frame)
                aggregates.append(aggregate)
    return aggregates, target_frames


def fit_descriptor_baseline(
    *,
    calibration_descriptors: np.ndarray,
    evaluation_descriptors: np.ndarray,
    transform: SplitTransform,
    target_names: tuple[str, ...],
    calibration_observed: np.ndarray,
    evaluation_observed: np.ndarray,
) -> list[dict[str, Any]]:
    """Pilot-only seven-descriptor chemistry baseline on the common P targets."""

    x_calibration, x_evaluation, _, _ = calibration_standardize_features(
        calibration_descriptors, evaluation_descriptors
    )
    model = RidgeCV(
        alphas=np.asarray(RIDGE_ALPHAS, dtype=np.float64),
        fit_intercept=False,
        cv=None,
        gcv_mode="svd",
        scoring="neg_mean_squared_error",
    )
    model.fit(x_calibration, transform.calibration_raw_z)
    predicted_raw_z = model.predict(x_evaluation)
    predicted_raw = predicted_raw_z * transform.raw_sds + transform.raw_means
    predicted_residual_raw = predicted_raw - predicted_raw.mean(axis=1, keepdims=True)
    predicted_residual_z = (
        predicted_residual_raw - transform.residual_means
    ) / transform.residual_sds
    observed_policy_raw = predictive_metrics_observed_cells(
        transform.evaluation_raw_z,
        predicted_raw_z,
        evaluation_observed,
    )
    observed_raw_metrics = {
        f"raw_originally_observed_cell_full_policy_{key}": value
        for key, value in observed_policy_raw.items()
    }
    calibration_complete = calibration_observed.all(axis=1)
    evaluation_complete = evaluation_observed.all(axis=1)
    records: list[dict[str, Any]] = []
    for truth_surface, y_calibration, y_evaluation, prediction in (
        (
            "calibration_standardized_raw",
            transform.calibration_raw_z,
            transform.evaluation_raw_z,
            predicted_raw_z,
        ),
        (
            "raw_row_centered_residual_outcome",
            transform.calibration_residual_z,
            transform.evaluation_residual_z,
            predicted_residual_z,
        ),
    ):
        for outcome_basis, calibration_rows, evaluation_rows in (
            (
                "split_imputed_all_rows",
                np.ones(len(y_calibration), dtype=bool),
                np.ones(len(y_evaluation), dtype=bool),
            ),
            (
                "originally_complete_outcome_rows",
                calibration_complete,
                evaluation_complete,
            ),
        ):
            truth = y_evaluation[evaluation_rows]
            fitted = prediction[evaluation_rows]
            aggregate, _ = predictive_metrics(truth, fitted, target_names)
            aggregate.update(row_profile_metrics(truth, fitted))
            tail, _ = lower_tail_metrics(
                y_calibration[calibration_rows], truth, fitted
            )
            aggregate.update(tail)
            records.append(
                {
                    "truth_surface": truth_surface,
                    "outcome_basis": outcome_basis,
                    "predictive_model": "ligand_descriptor_ridge",
                    "ridge_alpha": float(model.alpha_),
                    "descriptors": int(calibration_descriptors.shape[1]),
                    "calibration_rows_scored": int(calibration_rows.sum()),
                    "evaluation_rows_scored": int(evaluation_rows.sum()),
                    **observed_raw_metrics,
                    **aggregate,
                }
            )
    return records


def summarize_replicates(frame: pd.DataFrame, identifiers: list[str]) -> pd.DataFrame:
    excluded = set(identifiers) | {
        "repetition",
        "random_panel_draw",
        "panel_targets",
        "panel_indices",
    }
    metrics = [
        column
        for column in frame.columns
        if column not in excluded and pd.api.types.is_numeric_dtype(frame[column])
    ]
    records: list[dict[str, Any]] = []
    for keys, group in frame.groupby(identifiers, sort=True, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        record: dict[str, Any] = dict(zip(identifiers, keys, strict=True))
        record["records"] = int(len(group))
        record["independent_calibration_repetitions"] = int(group.repetition.nunique())
        for metric in metrics:
            values = group[metric].to_numpy(dtype=np.float64)
            record[f"{metric}_median"] = float(np.median(values))
            record[f"{metric}_q025"] = float(np.quantile(values, 0.025))
            record[f"{metric}_q975"] = float(np.quantile(values, 0.975))
        records.append(record)
    return pd.DataFrame.from_records(records)


def reconstruction_random_contrasts(frame: pd.DataFrame) -> pd.DataFrame:
    """Paired pilot-panel contrasts against random panels from the same split."""

    metrics = (
        "variance_weighted_r2",
        "median_target_r2",
        "pooled_calibration_standardized_rmse",
        "median_ligand_profile_spearman",
        "full_profile_policy_variance_weighted_r2",
        "full_profile_policy_pooled_calibration_standardized_rmse",
        "full_profile_policy_median_ligand_profile_spearman",
        "full_profile_policy_median_target_equal_budget_lower_5pct_overlap",
    )
    identifiers = [
        "dataset",
        "calibration_ligands",
        "repetition",
        "truth_surface",
        "outcome_basis",
        "predictive_model",
    ]
    records: list[dict[str, Any]] = []
    for keys, group in frame.groupby(identifiers, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        random = group.loc[group.panel_method.eq("random_panel")]
        pilots = group.loc[
            group.panel_information_scope.eq("pilot_only")
            & group.panel_method.ne("random_panel")
        ]
        if random.empty or pilots.empty:
            raise ValueError("a reconstruction split lacks pilot or random panels")
        for pilot in pilots.itertuples(index=False):
            record: dict[str, Any] = dict(zip(identifiers, keys, strict=True))
            record["panel_method"] = str(pilot.panel_method)
            record["random_panels"] = int(len(random))
            for metric in metrics:
                observed = float(getattr(pilot, metric))
                controls = random[metric].to_numpy(dtype=np.float64)
                record[f"pilot_{metric}"] = observed
                record[f"random_{metric}_median"] = float(np.median(controls))
                record[f"pilot_minus_random_median_{metric}"] = float(
                    observed - np.median(controls)
                )
                if metric.endswith("pooled_calibration_standardized_rmse"):
                    no_worse = controls <= observed
                else:
                    no_worse = controls >= observed
                record[f"fraction_random_panels_no_worse_{metric}"] = float(
                    np.mean(no_worse)
                )
            records.append(record)
    return pd.DataFrame.from_records(records)


def ridge_minus_pc1_contrasts(frame: pd.DataFrame) -> pd.DataFrame:
    """Paired multivariate increment beyond the selected-target one-factor model."""

    identifiers = [
        "dataset",
        "calibration_ligands",
        "repetition",
        "truth_surface",
        "outcome_basis",
        "panel_method",
        "panel_indices",
        "random_panel_draw",
    ]
    metrics = (
        "full_profile_policy_variance_weighted_r2",
        "full_profile_policy_pooled_calibration_standardized_rmse",
        "full_profile_policy_median_ligand_profile_spearman",
        "full_profile_policy_median_target_equal_budget_lower_5pct_overlap",
    )
    records: list[dict[str, Any]] = []
    for keys, group in frame.groupby(identifiers, sort=True, dropna=False):
        indexed = group.set_index("predictive_model")
        if set(indexed.index) != {"multivariate_ridge", "selected_target_pc1"}:
            raise ValueError("ridge/PC1 comparison is incomplete")
        record = dict(zip(identifiers, keys, strict=True))
        for metric in metrics:
            ridge = float(indexed.loc["multivariate_ridge", metric])
            pc1 = float(indexed.loc["selected_target_pc1", metric])
            record[f"ridge_{metric}"] = ridge
            record[f"selected_pc1_{metric}"] = pc1
            record[f"ridge_minus_selected_pc1_{metric}"] = ridge - pc1
        records.append(record)
    return pd.DataFrame.from_records(records)


def raw_minus_residual_contrasts(frame: pd.DataFrame) -> pd.DataFrame:
    """Paired surface contrast within one fixed panel and predictive model."""

    identifiers = [
        "dataset",
        "calibration_ligands",
        "repetition",
        "outcome_basis",
        "panel_method",
        "panel_indices",
        "random_panel_draw",
        "predictive_model",
    ]
    metrics = (
        "variance_weighted_r2",
        "pooled_calibration_standardized_rmse",
        "median_ligand_profile_spearman",
        "full_profile_policy_variance_weighted_r2",
        "full_profile_policy_pooled_calibration_standardized_rmse",
        "full_profile_policy_median_ligand_profile_spearman",
        "full_profile_policy_median_target_equal_budget_lower_5pct_overlap",
        "median_target_equal_budget_lower_5pct_overlap",
    )
    records: list[dict[str, Any]] = []
    for keys, group in frame.groupby(identifiers, sort=True, dropna=False):
        indexed = group.set_index("truth_surface")
        required = {
            "calibration_standardized_raw",
            "raw_row_centered_residual_outcome",
        }
        if set(indexed.index) != required:
            raise ValueError("raw/residual reconstruction comparison is incomplete")
        record = dict(zip(identifiers, keys, strict=True))
        for metric in metrics:
            raw = float(indexed.loc["calibration_standardized_raw", metric])
            residual = float(
                indexed.loc["raw_row_centered_residual_outcome", metric]
            )
            record[f"raw_{metric}"] = raw
            record[f"residual_{metric}"] = residual
            record[f"raw_minus_residual_{metric}"] = raw - residual
        records.append(record)
    return pd.DataFrame.from_records(records)


def panel_minus_descriptor_contrasts(
    reconstruction: pd.DataFrame, descriptor: pd.DataFrame
) -> pd.DataFrame:
    """Common-P policy comparison with the pilot-only ligand descriptor baseline."""

    panel = reconstruction.loc[
        reconstruction.predictive_model.eq("multivariate_ridge")
    ].copy()
    identifiers = [
        "dataset",
        "calibration_ligands",
        "repetition",
        "truth_surface",
        "outcome_basis",
    ]
    descriptor_indexed = descriptor.set_index(identifiers)
    metrics = (
        "full_profile_policy_variance_weighted_r2",
        "full_profile_policy_pooled_calibration_standardized_rmse",
        "full_profile_policy_median_ligand_profile_spearman",
        "full_profile_policy_median_target_equal_budget_lower_5pct_overlap",
    )
    records: list[dict[str, Any]] = []
    for row in panel.itertuples(index=False):
        key = tuple(getattr(row, field) for field in identifiers)
        if key not in descriptor_indexed.index:
            raise ValueError("descriptor comparator is missing a reconstruction split")
        baseline = descriptor_indexed.loc[key]
        record = {
            **dict(zip(identifiers, key, strict=True)),
            "panel_method": row.panel_method,
            "panel_indices": row.panel_indices,
            "random_panel_draw": row.random_panel_draw,
        }
        for metric in metrics:
            panel_value = float(getattr(row, metric))
            descriptor_value = float(baseline[metric.replace("full_profile_policy_", "")])
            record[f"panel_{metric}"] = panel_value
            record[f"descriptor_{metric}"] = descriptor_value
            record[f"panel_minus_descriptor_{metric}"] = (
                panel_value - descriptor_value
            )
        records.append(record)
    return pd.DataFrame.from_records(records)


def analyze_dataset(
    dataset: PublicDataset,
    *,
    sizes: tuple[int, ...],
    repetitions: int,
    panel_k: int,
    random_reconstruction_panels: int,
    random_coverage_panels: int,
    seed: int,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    matrix = np.asarray(dataset.matrix, dtype=np.float64)
    targets = dataset.targets
    target_count = matrix.shape[1]
    if target_count != len(targets) or not panel_k < target_count:
        raise ValueError("dataset target metadata does not align")

    full_transform = split_transform(matrix, matrix[:6])
    full_geometries = map_geometries(full_transform.calibration_raw)
    full_panels = {
        objective: exact_k_medoids(
            objective_map(full_geometries, objective), panel_k
        )
        for objective in PREDICTIVE_PANEL_METHODS
    }

    reconstruction_records: list[dict[str, Any]] = []
    target_records: list[pd.DataFrame] = []
    coverage_records: list[dict[str, Any]] = []
    selection_records: list[dict[str, Any]] = []
    descriptor_records: list[dict[str, Any]] = []
    rng = np.random.default_rng(seed)
    for size in sizes:
        if not 10 <= size < len(matrix) - 10:
            raise ValueError("calibration size is invalid")
        for repetition in range(repetitions):
            calibration_index = np.sort(rng.choice(len(matrix), size=size, replace=False))
            evaluation_index = np.setdiff1d(
                np.arange(len(matrix), dtype=np.int64), calibration_index
            )
            if np.intersect1d(calibration_index, evaluation_index).size:
                raise AssertionError("calibration and evaluation rows overlap")
            transform = split_transform(
                matrix[calibration_index], matrix[evaluation_index]
            )
            calibration_observed = np.isfinite(matrix[calibration_index])
            evaluation_observed = np.isfinite(matrix[evaluation_index])
            for descriptor_record in fit_descriptor_baseline(
                calibration_descriptors=dataset.descriptor_matrix[
                    calibration_index
                ],
                evaluation_descriptors=dataset.descriptor_matrix[evaluation_index],
                transform=transform,
                target_names=targets,
                calibration_observed=calibration_observed,
                evaluation_observed=evaluation_observed,
            ):
                descriptor_records.append(
                    {
                        "dataset": dataset.name,
                        "calibration_ligands": int(size),
                        "evaluation_ligands": int(len(evaluation_index)),
                        "repetition": repetition,
                        "calibration_evaluation_overlap": 0,
                        **descriptor_record,
                    }
                )
            calibration_geometries = map_geometries(transform.calibration_raw)
            # The held-out coverage target and exact complement oracle are defined on
            # the complement itself.  Evaluation-only imputation is metric definition,
            # never an input to pilot selection or prediction.
            evaluation_truth_transform = split_transform(
                matrix[evaluation_index], matrix[evaluation_index][:6]
            )
            evaluation_geometries = map_geometries(
                evaluation_truth_transform.calibration_raw
            )
            calibration_panels = {
                objective: exact_k_medoids(
                    objective_map(calibration_geometries, objective), panel_k
                )
                for objective in PREDICTIVE_PANEL_METHODS
            }
            generated_random_panels = random_panels(
                target_count, panel_k, random_reconstruction_panels, rng
            )
            reconstruction_panels: list[
                tuple[str, str, int, np.ndarray]
            ] = []
            for objective, panel in calibration_panels.items():
                reconstruction_panels.append(
                    (f"pilot_exact_{objective}", "pilot_only", -1, panel)
                )
            for objective, panel in full_panels.items():
                reconstruction_panels.append(
                    (
                        f"full_source_exact_{objective}",
                        "full_source_diagnostic_ceiling",
                        -1,
                        panel,
                    )
                )
            for draw, panel in enumerate(generated_random_panels):
                reconstruction_panels.append(
                    ("random_panel", "outcome_blind_random", draw, panel)
                )

            for method, information_scope, random_draw, panel in reconstruction_panels:
                aggregates, target_frames = fit_reconstruction(
                    transform,
                    panel,
                    targets,
                    calibration_observed,
                    evaluation_observed,
                )
                panel_names = "|".join(targets[index] for index in panel)
                panel_indices = "|".join(str(int(index)) for index in panel)
                for aggregate in aggregates:
                    reconstruction_records.append(
                        {
                            "dataset": dataset.name,
                            "calibration_ligands": int(size),
                            "evaluation_ligands": int(len(evaluation_index)),
                            "repetition": repetition,
                            "panel_method": method,
                            "panel_information_scope": information_scope,
                            "random_panel_draw": random_draw,
                            "panel_targets": panel_names,
                            "panel_indices": panel_indices,
                            "calibration_evaluation_overlap": 0,
                            **aggregate,
                        }
                    )
                for target_frame in target_frames:
                    target_frame.insert(0, "panel_indices", panel_indices)
                    target_frame.insert(0, "random_panel_draw", random_draw)
                    target_frame.insert(0, "panel_information_scope", information_scope)
                    target_frame.insert(0, "panel_method", method)
                    target_frame.insert(0, "repetition", repetition)
                    target_frame.insert(0, "calibration_ligands", size)
                    target_frame.insert(0, "dataset", dataset.name)
                    target_records.append(target_frame)

            control_panels = random_panels(
                target_count, panel_k, random_coverage_panels, rng
            )
            for objective in DISTANCE_OBJECTIVES:
                calibration_distance = objective_map(calibration_geometries, objective)
                evaluation_distance = objective_map(evaluation_geometries, objective)
                calibration_panel = calibration_panels[objective]
                evaluation_oracle = exact_k_medoids(evaluation_distance, panel_k)
                full_panel = full_panels[objective]
                oracle_loss = coverage_loss(evaluation_distance, evaluation_oracle)
                random_losses = np.asarray(
                    [coverage_loss(evaluation_distance, panel) for panel in control_panels]
                )
                for selection_source, panel in (
                    ("pilot_exact", calibration_panel),
                    ("full_source_exact_diagnostic", full_panel),
                ):
                    loss = coverage_loss(evaluation_distance, panel)
                    regret = loss - oracle_loss
                    if regret < -1e-9:
                        raise AssertionError("held-out regret is negative")
                    coverage_records.append(
                        {
                            "dataset": dataset.name,
                            "calibration_ligands": size,
                            "evaluation_ligands": int(len(evaluation_index)),
                            "repetition": repetition,
                            "objective": objective,
                            "selection_source": selection_source,
                            "panel_targets_k": panel_k,
                            "calibration_evaluation_overlap": 0,
                            "held_out_absolute_mean_nearest_loss": loss,
                            "held_out_exact_oracle_loss": oracle_loss,
                            "held_out_true_regret": max(0.0, regret),
                            "fraction_random_panels_no_worse": float(
                                np.mean(random_losses <= loss)
                            ),
                            "random_loss_median": float(np.median(random_losses)),
                            "random_loss_q025": float(np.quantile(random_losses, 0.025)),
                            "random_loss_q975": float(np.quantile(random_losses, 0.975)),
                            "random_panels": int(len(random_losses)),
                            "panel_targets": "|".join(targets[index] for index in panel),
                        }
                    )
                for selection_source, panel in (
                    ("pilot_exact", calibration_panel),
                    ("complement_exact_oracle", evaluation_oracle),
                    ("full_source_exact_diagnostic", full_panel),
                ):
                    for target_index in panel:
                        selection_records.append(
                            {
                                "dataset": dataset.name,
                                "calibration_ligands": size,
                                "repetition": repetition,
                                "objective": objective,
                                "selection_source": selection_source,
                                "target_index": int(target_index),
                                "target": targets[int(target_index)],
                            }
                        )

    reconstruction = pd.DataFrame.from_records(reconstruction_records)
    target_metrics = pd.concat(target_records, ignore_index=True)
    coverage = pd.DataFrame.from_records(coverage_records)
    selections = pd.DataFrame.from_records(selection_records)
    descriptor_baseline = pd.DataFrame.from_records(descriptor_records)
    costs = cost_table(dataset, sizes, panel_k)
    return (
        reconstruction,
        target_metrics,
        coverage,
        selections,
        costs,
        descriptor_baseline,
    )


def exact_rank_domains(values: np.ndarray) -> dict[str, np.ndarray]:
    values = np.asarray(values, dtype=np.float64)
    order = np.lexsort((np.arange(len(values)), values))
    count = len(values) // 4
    if count < 10 or 2 * count >= len(values):
        raise ValueError("molecular-weight domains are too small")
    low = np.sort(order[:count])
    high = np.sort(order[-count:])
    if np.intersect1d(low, high).size:
        raise AssertionError("molecular-weight domains overlap")
    return {"low_mw_rank_quartile": low, "high_mw_rank_quartile": high}


def fit_domain_reconstruction(
    *,
    calibration_raw: np.ndarray,
    evaluation_raw_complete: np.ndarray,
    calibration_z: np.ndarray,
    calibration_means: np.ndarray,
    calibration_sds: np.ndarray,
    evaluation_reference_means: np.ndarray,
    evaluation_reference_sds: np.ndarray,
    panel: np.ndarray,
    target_names: tuple[str, ...],
) -> list[dict[str, Any]]:
    """Deployable prediction on one common, originally complete domain truth set."""

    selected = np.asarray(panel, dtype=np.int64)
    omitted = np.setdiff1d(np.arange(calibration_z.shape[1]), selected)
    x_calibration = calibration_z[:, selected]
    calibration_complete = np.isfinite(calibration_raw).all(axis=1)
    if calibration_complete.sum() < 20 or not np.isfinite(evaluation_raw_complete).all():
        raise ValueError("domain reconstruction lacks complete outcome rows")
    filled_calibration = np.where(
        np.isnan(calibration_raw), calibration_means, calibration_raw
    )
    evaluation_z_pilot = (
        evaluation_raw_complete - calibration_means
    ) / calibration_sds
    x_evaluation = evaluation_z_pilot[:, selected]
    model = RidgeCV(
        alphas=np.asarray(RIDGE_ALPHAS, dtype=np.float64),
        fit_intercept=False,
        cv=None,
        gcv_mode="svd",
        scoring="neg_mean_squared_error",
    )
    model.fit(x_calibration, calibration_z[:, omitted])
    predicted_omitted_pilot_z = model.predict(x_evaluation)
    predicted_omitted_raw = (
        predicted_omitted_pilot_z * calibration_sds[omitted]
        + calibration_means[omitted]
    )
    # The common evaluation-reference standardization is metric-only. It is never
    # supplied to target selection or model fitting, and is identical for all three
    # calibration designs paired on these evaluation rows.
    truth_common_z = (
        evaluation_raw_complete - evaluation_reference_means
    ) / evaluation_reference_sds
    predicted_omitted_common_z = (
        predicted_omitted_raw - evaluation_reference_means[omitted]
    ) / evaluation_reference_sds[omitted]

    policy_model = Ridge(alpha=float(model.alpha_), fit_intercept=False)
    policy_model.fit(x_calibration, calibration_z)
    predicted_policy_raw = (
        policy_model.predict(x_evaluation) * calibration_sds + calibration_means
    )
    predicted_policy_raw[:, selected] = evaluation_raw_complete[:, selected]
    predicted_policy_common_z = (
        predicted_policy_raw - evaluation_reference_means
    ) / evaluation_reference_sds

    raw_aggregate, _ = predictive_metrics(
        truth_common_z[:, omitted],
        predicted_omitted_common_z,
        [target_names[index] for index in omitted],
    )
    raw_policy, _ = predictive_metrics(
        truth_common_z, predicted_policy_common_z, target_names
    )
    raw_aggregate.update(
        {
            f"full_profile_policy_{key}": value
            for key, value in raw_policy.items()
        }
    )
    raw_aggregate.update(
        {
            f"full_profile_policy_{key}": value
            for key, value in row_profile_metrics(
                truth_common_z, predicted_policy_common_z
            ).items()
        }
    )
    raw_aggregate.update(
        row_profile_metrics(
            truth_common_z[:, omitted], predicted_omitted_common_z
        )
    )
    raw_policy_tail, _ = lower_tail_metrics(
        filled_calibration[calibration_complete],
        evaluation_raw_complete,
        predicted_policy_raw,
    )
    raw_aggregate.update(
        {
            f"full_profile_policy_{key}": value
            for key, value in raw_policy_tail.items()
        }
    )
    raw_tail, _ = lower_tail_metrics(
        filled_calibration[calibration_complete][:, omitted],
        evaluation_raw_complete[:, omitted],
        predicted_omitted_raw,
    )
    raw_aggregate.update(raw_tail)
    raw_aggregate["truth_surface"] = "evaluation_standardized_raw"

    truth_residual_raw = evaluation_raw_complete - evaluation_raw_complete.mean(
        axis=1, keepdims=True
    )
    evaluation_residual_means = truth_residual_raw.mean(axis=0)
    evaluation_residual_sds = truth_residual_raw.std(axis=0, ddof=1)
    if np.any(evaluation_residual_sds <= 1e-12):
        raise ValueError("domain evaluation residual contains a constant target")
    truth_residual = (
        truth_residual_raw - evaluation_residual_means
    ) / evaluation_residual_sds
    predicted_residual_raw = predicted_policy_raw - predicted_policy_raw.mean(
        axis=1, keepdims=True
    )
    predicted_residual = (
        predicted_residual_raw - evaluation_residual_means
    ) / evaluation_residual_sds
    residual_aggregate, _ = predictive_metrics(
        truth_residual[:, omitted],
        predicted_residual[:, omitted],
        [target_names[index] for index in omitted],
    )
    residual_policy, _ = predictive_metrics(
        truth_residual, predicted_residual, target_names
    )
    residual_aggregate.update(
        {
            f"full_profile_policy_{key}": value
            for key, value in residual_policy.items()
        }
    )
    residual_aggregate.update(
        {
            f"full_profile_policy_{key}": value
            for key, value in row_profile_metrics(
                truth_residual, predicted_residual
            ).items()
        }
    )
    residual_aggregate.update(
        row_profile_metrics(
            truth_residual[:, omitted], predicted_residual[:, omitted]
        )
    )
    calibration_residual_raw = filled_calibration - filled_calibration.mean(
        axis=1, keepdims=True
    )
    calibration_residual = (
        calibration_residual_raw - evaluation_residual_means
    ) / evaluation_residual_sds
    residual_policy_tail, _ = lower_tail_metrics(
        calibration_residual[calibration_complete],
        truth_residual,
        predicted_residual,
    )
    residual_aggregate.update(
        {
            f"full_profile_policy_{key}": value
            for key, value in residual_policy_tail.items()
        }
    )
    residual_tail, _ = lower_tail_metrics(
        calibration_residual[calibration_complete][:, omitted],
        truth_residual[:, omitted],
        predicted_residual[:, omitted],
    )
    residual_aggregate.update(residual_tail)
    residual_aggregate["truth_surface"] = (
        "evaluation_standardized_full_profile_residual"
    )

    records: list[dict[str, Any]] = []
    for aggregate in (raw_aggregate, residual_aggregate):
        records.append(
            {
                **aggregate,
                "ridge_alpha": float(model.alpha_),
                "selected_targets": int(len(selected)),
                "omitted_targets": int(len(omitted)),
                "calibration_complete_rows_for_tail_threshold": int(
                    calibration_complete.sum()
                ),
                "evaluation_complete_rows": int(len(evaluation_raw_complete)),
            }
        )
    return records


def analyze_domain_shift(
    dataset: PublicDataset,
    *,
    sizes: tuple[int, ...],
    repetitions: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    matrix = dataset.matrix
    domains = exact_rank_domains(dataset.molecular_weight)
    rng = np.random.default_rng(seed)
    records: list[dict[str, Any]] = []
    reconstruction_records: list[dict[str, Any]] = []
    all_rows = np.arange(len(matrix), dtype=np.int64)
    for domain, domain_rows in domains.items():
        opposite_domain = (
            "high_mw_rank_quartile"
            if domain == "low_mw_rank_quartile"
            else "low_mw_rank_quartile"
        )
        opposite_rows = domains[opposite_domain]
        for size in sizes:
            if size >= len(domain_rows) - 10:
                raise ValueError("calibration size exhausts an MW domain")
            for repetition in range(repetitions):
                # Sample one large common truth support first. All three calibration
                # designs are then row-disjoint from exactly the same evaluation rows.
                evaluation_size = len(domain_rows) // 2
                evaluation = np.sort(
                    rng.choice(domain_rows, size=evaluation_size, replace=False)
                )
                within_pool = np.setdiff1d(domain_rows, evaluation)
                within = np.sort(rng.choice(within_pool, size=size, replace=False))
                source_pool = np.setdiff1d(all_rows, evaluation)
                source = np.sort(rng.choice(source_pool, size=size, replace=False))
                wrong_domain = np.sort(
                    rng.choice(opposite_rows, size=size, replace=False)
                )
                if np.intersect1d(within, evaluation).size or np.intersect1d(
                    source, evaluation
                ).size or np.intersect1d(wrong_domain, evaluation).size:
                    raise AssertionError("domain calibration overlaps evaluation")
                # Define the held-out truth from evaluation rows alone. Each pilot uses
                # its own calibration-only imputation/scaling before its map is formed.
                evaluation_transform = split_transform(
                    matrix[evaluation], matrix[evaluation][:6]
                )
                evaluation_maps = map_geometries(
                    evaluation_transform.calibration_raw
                )
                evaluation_complete_mask = np.isfinite(matrix[evaluation]).all(axis=1)
                evaluation_raw_complete = matrix[evaluation][evaluation_complete_mask]
                if len(evaluation_raw_complete) < 50:
                    raise ValueError("too few complete domain evaluation rows")
                evaluation_reference_means = evaluation_raw_complete.mean(axis=0)
                evaluation_reference_sds = evaluation_raw_complete.std(axis=0, ddof=1)
                if np.any(evaluation_reference_sds <= 1e-12):
                    raise ValueError("domain evaluation reference has a constant target")
                for calibration_scope, calibration_rows in (
                    ("within_domain", within),
                    ("source_library_row_disjoint", source),
                    ("opposite_mw_domain", wrong_domain),
                ):
                    calibration_transform_values = split_transform(
                        matrix[calibration_rows], matrix[evaluation][:6]
                    )
                    calibration_maps = map_geometries(
                        calibration_transform_values.calibration_raw
                    )
                    for transformation in ("raw", "residual"):
                        records.append(
                            {
                                "dataset": dataset.name,
                                "domain": domain,
                                "calibration_ligands": size,
                                "evaluation_ligands": int(len(evaluation)),
                                "repetition": repetition,
                                "calibration_scope": calibration_scope,
                                "transformation": transformation,
                                "calibration_evaluation_overlap": 0,
                                "within_vs_source_calibration_overlap": int(
                                    np.intersect1d(within, source).size
                                ),
                                "within_vs_opposite_calibration_overlap": 0,
                                "source_vs_opposite_calibration_overlap": int(
                                    np.intersect1d(source, wrong_domain).size
                                ),
                                "source_calibration_domain_fraction": float(
                                    np.mean(np.isin(source, domain_rows))
                                ),
                                "conditional_expected_source_domain_fraction": float(
                                    len(within_pool) / len(source_pool)
                                ),
                                "map_spearman_to_same_domain_complement": map_spearman(
                                    calibration_maps[transformation],
                                    evaluation_maps[transformation],
                                ),
                                "calibration_median_molecular_weight": float(
                                    np.median(
                                        dataset.molecular_weight[calibration_rows]
                                    )
                                ),
                                "evaluation_median_molecular_weight": float(
                                    np.median(dataset.molecular_weight[evaluation])
                                ),
                            }
                        )
                    panel = exact_k_medoids(
                        objective_map(calibration_maps, "residual_r2"),
                        DEFAULT_PANEL_K,
                    )
                    domain_reconstruction = fit_domain_reconstruction(
                        calibration_raw=matrix[calibration_rows],
                        evaluation_raw_complete=evaluation_raw_complete,
                        calibration_z=(
                            calibration_transform_values.calibration_raw_z
                        ),
                        calibration_means=calibration_transform_values.raw_means,
                        calibration_sds=calibration_transform_values.raw_sds,
                        evaluation_reference_means=evaluation_reference_means,
                        evaluation_reference_sds=evaluation_reference_sds,
                        panel=panel,
                        target_names=dataset.targets,
                    )
                    for domain_record in domain_reconstruction:
                        reconstruction_records.append(
                            {
                                "dataset": dataset.name,
                                "domain": domain,
                                "calibration_ligands": size,
                                "repetition": repetition,
                                "calibration_scope": calibration_scope,
                                "panel_method": "exact_residual_r2",
                                "panel_targets": "|".join(
                                    dataset.targets[index] for index in panel
                                ),
                                "calibration_evaluation_overlap": 0,
                                "source_calibration_domain_fraction": float(
                                    np.mean(np.isin(source, domain_rows))
                                ),
                                "conditional_expected_source_domain_fraction": float(
                                    len(within_pool) / len(source_pool)
                                ),
                                **domain_record,
                            }
                        )
    return (
        pd.DataFrame.from_records(records),
        pd.DataFrame.from_records(reconstruction_records),
    )


def paired_domain_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    identifiers = [
        "dataset",
        "domain",
        "calibration_ligands",
        "transformation",
        "repetition",
    ]
    pivot = metrics.pivot(
        index=identifiers,
        columns="calibration_scope",
        values="map_spearman_to_same_domain_complement",
    ).reset_index()
    required = {
        "within_domain",
        "source_library_row_disjoint",
        "opposite_mw_domain",
    }
    if not required.issubset(pivot.columns):
        raise ValueError("domain comparison is incomplete")
    pivot["within_minus_source_map_spearman"] = (
        pivot.within_domain - pivot.source_library_row_disjoint
    )
    pivot["within_minus_opposite_map_spearman"] = (
        pivot.within_domain - pivot.opposite_mw_domain
    )
    pivot["source_minus_opposite_map_spearman"] = (
        pivot.source_library_row_disjoint - pivot.opposite_mw_domain
    )
    return summarize_replicates(
        pivot,
        ["dataset", "domain", "calibration_ligands", "transformation"],
    )


def paired_domain_reconstruction_contrasts(metrics: pd.DataFrame) -> pd.DataFrame:
    identifiers = [
        "dataset",
        "domain",
        "calibration_ligands",
        "repetition",
        "truth_surface",
    ]
    compared = (
        "variance_weighted_r2",
        "pooled_calibration_standardized_rmse",
        "median_ligand_profile_spearman",
        "full_profile_policy_variance_weighted_r2",
        "full_profile_policy_pooled_calibration_standardized_rmse",
        "full_profile_policy_median_ligand_profile_spearman",
        "full_profile_policy_median_target_equal_budget_lower_5pct_overlap",
    )
    records: list[dict[str, Any]] = []
    for keys, group in metrics.groupby(identifiers, sort=True):
        record: dict[str, Any] = dict(zip(identifiers, keys, strict=True))
        indexed = group.set_index("calibration_scope")
        required = {
            "within_domain",
            "source_library_row_disjoint",
            "opposite_mw_domain",
        }
        if set(indexed.index) != required:
            raise ValueError("domain reconstruction comparison is incomplete")
        for metric in compared:
            within = float(indexed.loc["within_domain", metric])
            source = float(indexed.loc["source_library_row_disjoint", metric])
            opposite = float(indexed.loc["opposite_mw_domain", metric])
            record[f"within_domain_{metric}"] = within
            record[f"source_library_{metric}"] = source
            record[f"opposite_domain_{metric}"] = opposite
            record[f"within_minus_source_{metric}"] = within - source
            record[f"within_minus_opposite_{metric}"] = within - opposite
            record[f"source_minus_opposite_{metric}"] = source - opposite
        records.append(record)
    return pd.DataFrame.from_records(records)


def cost_table(
    dataset: PublicDataset, sizes: tuple[int, ...], panel_k: int
) -> pd.DataFrame:
    target_count = dataset.matrix.shape[1]
    records: list[dict[str, Any]] = []
    for size in sizes:
        future = len(dataset.matrix) - size
        pilot_cells = size * target_count
        future_selected_cells = future * panel_k
        future_full_cells = future * target_count
        break_even = size * target_count / (target_count - panel_k)
        records.append(
            {
                "dataset": dataset.name,
                "calibration_ligands": size,
                "targets": target_count,
                "selected_targets": panel_k,
                "row_disjoint_future_ligands": future,
                "pilot_score_cells": pilot_cells,
                "future_selected_panel_score_cells": future_selected_cells,
                "selected_workflow_score_cells": pilot_cells + future_selected_cells,
                "future_full_panel_score_cells": future_full_cells,
                "selected_workflow_fraction_of_future_full_scoring": float(
                    (pilot_cells + future_selected_cells) / future_full_cells
                ),
                "prospective_break_even_future_ligands": float(break_even),
                "minimum_integer_future_ligands_for_savings": int(np.floor(break_even) + 1),
                "break_even_formula": "n_calibration*P/(P-k)",
            }
        )
    return pd.DataFrame.from_records(records)


def build_readme(
    reconstruction_summary: pd.DataFrame,
    coverage_summary: pd.DataFrame,
    domain_summary: pd.DataFrame,
    domain_reconstruction_contrast_summary: pd.DataFrame,
    costs: pd.DataFrame,
) -> str:
    lines = [
        "# Deployable panel reconstruction and domain transport",
        "",
        "This strict-public artifact evaluates a concrete use of a Vina target map:",
        "select eight targets from a fully scored pilot, score each future ligand only",
        "on those eight targets, and reconstruct one full raw-score profile. The eight",
        "observed scores are copied into that profile; residual endpoints are derived",
        "from the same reconstructed raw profile, not from a second fitted model.",
        "All transforms and predictive models are pilot-only. Full-source panels and",
        "complement optima are labelled non-deployable diagnostics.",
        "",
        "## Reconstruction",
        "",
    ]
    selected = reconstruction_summary.loc[
        reconstruction_summary.panel_method.isin(
            ["pilot_exact_residual_r2", "random_panel"]
        )
        & reconstruction_summary.truth_surface.eq(
            "raw_row_centered_residual_outcome"
        )
        & reconstruction_summary.outcome_basis.eq(
            "originally_complete_outcome_rows"
        )
        & reconstruction_summary.predictive_model.eq("multivariate_ridge")
    ]
    for row in selected.itertuples(index=False):
        lines.append(
            f"- {row.dataset}, n={row.calibration_ligands}, {row.panel_method}: "
            f"median held-out full-profile residual variance-weighted R2 "
            f"{row.full_profile_policy_variance_weighted_r2_median:.3f}; median "
            f"per-ligand full-profile residual Spearman "
            f"{row.full_profile_policy_median_ligand_profile_spearman_median:.3f}."
        )
    lines.extend(["", "## Exact held-out coverage", ""])
    selected_coverage = coverage_summary.loc[
        coverage_summary.selection_source.eq("pilot_exact")
        & coverage_summary.objective.eq("residual_r2")
    ]
    for row in selected_coverage.itertuples(index=False):
        lines.append(
            f"- {row.dataset}, n={row.calibration_ligands}: median exact held-out "
            f"regret {row.held_out_true_regret_median:.4f}; median fraction of random "
            f"panels no worse {row.fraction_random_panels_no_worse_median:.3f}."
        )
    lines.extend(["", "## Molecular-weight domain shift", ""])
    for row in domain_summary.loc[domain_summary.transformation.eq("residual")].itertuples(
        index=False
    ):
        lines.append(
            f"- {row.dataset}, {row.domain}, n={row.calibration_ligands}: median paired "
            f"within-domain minus source-library map agreement "
            f"{row.within_minus_source_map_spearman_median:.3f}."
        )
    lines.extend(["", "On the same domain evaluation rows:", ""])
    selected_domain_reconstruction = domain_reconstruction_contrast_summary.loc[
        domain_reconstruction_contrast_summary.truth_surface.eq(
            "evaluation_standardized_full_profile_residual"
        )
    ]
    for row in selected_domain_reconstruction.itertuples(index=False):
        lines.append(
            f"- {row.dataset}, {row.domain}, n={row.calibration_ligands}: median "
            f"within-minus-source full-profile residual R2 "
            f"{row.within_minus_source_full_profile_policy_variance_weighted_r2_median:.3f}; "
            f"within-minus-opposite "
            f"{row.within_minus_opposite_full_profile_policy_variance_weighted_r2_median:.3f}."
        )
    lines.extend(["", "## Score-cell cost", ""])
    for row in costs.itertuples(index=False):
        lines.append(
            f"- {row.dataset}, n={row.calibration_ligands}: break-even after "
            f"{row.minimum_integer_future_ligands_for_savings} future ligands; on the "
            f"released row-disjoint complement the pilot-plus-eight-target workflow uses "
            f"{100*row.selected_workflow_fraction_of_future_full_scoring:.1f}% of the "
            "score cells required to score all future ligands against every target."
        )
    lines.extend(
        [
            "",
        "These are computational reconstruction and Vina-map compression endpoints.",
        "Common-P full-profile policy metrics are the primary comparison across panels;",
        "omitted-target metrics are directly comparable only for the same fixed panel.",
        "The selected-target PC1 and ligand-only seven-descriptor Ridge controls separate",
        "generic one-factor and chemistry-only predictability from multivariate panel value.",
        "They do not validate experimental target ranking, biological coverage, or",
            "docking quality, and they do not imply transport beyond the released source",
            "libraries or the two exact-rank molecular-weight domains.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_int_tuple(value: str) -> tuple[int, ...]:
    result = tuple(int(item) for item in value.split(",") if item)
    if not result or len(set(result)) != len(result):
        raise argparse.ArgumentTypeError("expected distinct comma-separated integers")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=parse_int_tuple, default=DEFAULT_SIZES)
    parser.add_argument("--repetitions", type=int, default=DEFAULT_REPETITIONS)
    parser.add_argument("--panel-k", type=int, default=DEFAULT_PANEL_K)
    parser.add_argument(
        "--random-reconstruction-panels",
        type=int,
        default=DEFAULT_RANDOM_RECONSTRUCTION_PANELS,
    )
    parser.add_argument(
        "--random-coverage-panels", type=int, default=DEFAULT_RANDOM_COVERAGE_PANELS
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--datasets",
        default="Docking-44,DOCKSTRING-58",
        help="comma-separated strict-public dataset names",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if (
        args.repetitions < 1
        or args.random_reconstruction_panels < 1
        or args.random_coverage_panels < 10
    ):
        raise ValueError("use >=1 repetitions/random reconstruction and >=10 controls")
    requested = tuple(item for item in args.datasets.split(",") if item)
    valid = {"Docking-44", "DOCKSTRING-58"}
    if not requested or not set(requested).issubset(valid):
        raise ValueError(f"datasets must be drawn from {sorted(valid)}")
    loaders = {
        "Docking-44": load_docking44,
        "DOCKSTRING-58": load_fixed_dockstring,
    }
    datasets = [loaders[name]() for name in requested]

    reconstruction_parts: list[pd.DataFrame] = []
    target_parts: list[pd.DataFrame] = []
    coverage_parts: list[pd.DataFrame] = []
    selection_parts: list[pd.DataFrame] = []
    cost_parts: list[pd.DataFrame] = []
    domain_parts: list[pd.DataFrame] = []
    domain_reconstruction_parts: list[pd.DataFrame] = []
    descriptor_parts: list[pd.DataFrame] = []
    for dataset_index, dataset in enumerate(datasets):
        parts = analyze_dataset(
            dataset,
            sizes=args.sizes,
            repetitions=args.repetitions,
            panel_k=args.panel_k,
            random_reconstruction_panels=args.random_reconstruction_panels,
            random_coverage_panels=args.random_coverage_panels,
            seed=args.seed + dataset_index * 1_000_000,
        )
        reconstruction_parts.append(parts[0])
        target_parts.append(parts[1])
        coverage_parts.append(parts[2])
        selection_parts.append(parts[3])
        cost_parts.append(parts[4])
        descriptor_parts.append(parts[5])
        domain_maps, domain_reconstruction = analyze_domain_shift(
            dataset,
            sizes=args.sizes,
            repetitions=args.repetitions,
            seed=args.seed + dataset_index * 1_000_000 + 500_000,
        )
        domain_parts.append(domain_maps)
        domain_reconstruction_parts.append(domain_reconstruction)
    reconstruction = pd.concat(reconstruction_parts, ignore_index=True)
    target_metrics = pd.concat(target_parts, ignore_index=True)
    coverage = pd.concat(coverage_parts, ignore_index=True)
    selections = pd.concat(selection_parts, ignore_index=True)
    costs = pd.concat(cost_parts, ignore_index=True)
    domain = pd.concat(domain_parts, ignore_index=True)
    domain_reconstruction = pd.concat(
        domain_reconstruction_parts, ignore_index=True
    )
    descriptor_baseline = pd.concat(descriptor_parts, ignore_index=True)
    reconstruction_summary = summarize_replicates(
        reconstruction,
        [
            "dataset",
            "calibration_ligands",
            "panel_method",
            "panel_information_scope",
            "truth_surface",
            "outcome_basis",
            "predictive_model",
        ],
    )
    reconstruction_contrasts = reconstruction_random_contrasts(reconstruction)
    reconstruction_contrast_summary = summarize_replicates(
        reconstruction_contrasts,
        [
            "dataset",
            "calibration_ligands",
            "truth_surface",
            "outcome_basis",
            "predictive_model",
            "panel_method",
        ],
    )
    ridge_pc1 = ridge_minus_pc1_contrasts(reconstruction)
    ridge_pc1_summary = summarize_replicates(
        ridge_pc1,
        [
            "dataset",
            "calibration_ligands",
            "truth_surface",
            "outcome_basis",
            "panel_method",
        ],
    )
    raw_residual = raw_minus_residual_contrasts(reconstruction)
    raw_residual_summary = summarize_replicates(
        raw_residual,
        [
            "dataset",
            "calibration_ligands",
            "outcome_basis",
            "panel_method",
            "predictive_model",
        ],
    )
    descriptor_summary = summarize_replicates(
        descriptor_baseline,
        [
            "dataset",
            "calibration_ligands",
            "truth_surface",
            "outcome_basis",
            "predictive_model",
        ],
    )
    panel_descriptor = panel_minus_descriptor_contrasts(
        reconstruction, descriptor_baseline
    )
    panel_descriptor_summary = summarize_replicates(
        panel_descriptor,
        [
            "dataset",
            "calibration_ligands",
            "truth_surface",
            "outcome_basis",
            "panel_method",
        ],
    )
    coverage_summary = summarize_replicates(
        coverage,
        ["dataset", "calibration_ligands", "objective", "selection_source"],
    )
    domain_summary = paired_domain_summary(domain)
    domain_reconstruction_summary = summarize_replicates(
        domain_reconstruction,
        [
            "dataset",
            "domain",
            "calibration_ligands",
            "truth_surface",
            "calibration_scope",
        ],
    )
    domain_reconstruction_contrasts = paired_domain_reconstruction_contrasts(
        domain_reconstruction
    )
    domain_reconstruction_contrast_summary = summarize_replicates(
        domain_reconstruction_contrasts,
        ["dataset", "domain", "calibration_ligands", "truth_surface"],
    )

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    tables = {
        "reconstruction_metrics.csv": reconstruction,
        "reconstruction_target_metrics.csv": target_metrics,
        "reconstruction_summary.csv": reconstruction_summary,
        "reconstruction_random_contrasts.csv": reconstruction_contrasts,
        "reconstruction_random_contrast_summary.csv": reconstruction_contrast_summary,
        "reconstruction_ridge_pc1_contrasts.csv": ridge_pc1,
        "reconstruction_ridge_pc1_contrast_summary.csv": ridge_pc1_summary,
        "reconstruction_raw_residual_contrasts.csv": raw_residual,
        "reconstruction_raw_residual_contrast_summary.csv": raw_residual_summary,
        "descriptor_baseline_metrics.csv": descriptor_baseline,
        "descriptor_baseline_summary.csv": descriptor_summary,
        "reconstruction_descriptor_contrasts.csv": panel_descriptor,
        "reconstruction_descriptor_contrast_summary.csv": panel_descriptor_summary,
        "panel_coverage_metrics.csv": coverage,
        "panel_coverage_summary.csv": coverage_summary,
        "panel_selections.csv": selections,
        "mw_domain_recovery_metrics.csv": domain,
        "mw_domain_recovery_summary.csv": domain_summary,
        "mw_domain_reconstruction_metrics.csv": domain_reconstruction,
        "mw_domain_reconstruction_summary.csv": domain_reconstruction_summary,
        "mw_domain_reconstruction_contrasts.csv": domain_reconstruction_contrasts,
        "mw_domain_reconstruction_contrast_summary.csv": (
            domain_reconstruction_contrast_summary
        ),
        "score_cell_costs.csv": costs,
    }
    for filename, frame in tables.items():
        if frame.empty:
            raise RuntimeError(f"refusing to write empty output {filename}")
        frame.to_csv(output / filename, index=False, float_format="%.17g")
    metadata = {
        "schema_version": "1.0.0",
        "analysis_status": "strict_public_exploratory_utility_validation",
        "seed": int(args.seed),
        "sizes": list(args.sizes),
        "repetitions_per_size": int(args.repetitions),
        "panel_targets_k": int(args.panel_k),
        "ridge": {
            "alphas": list(RIDGE_ALPHAS),
            "cross_validation": (
                "efficient generalized leave-one-out cross-validation within calibration "
                "rows (RidgeCV cv=None, gcv_mode=svd)"
            ),
            "predictors": (
                "calibration-standardized raw scores for selected targets only"
            ),
            "fitted_outcome": "calibration-standardized raw scores only",
            "deployment_policy": (
                "predict omitted raw scores, copy observed selected raw scores into the "
                "full profile, and derive the residual profile by raw-unit row centering "
                "followed by pilot residual-column scaling; no residual model is fitted"
            ),
            "outcome_bases": [
                "split-local-imputed complete matrices",
                "sensitivity restricted to rows whose original scores are complete for every target",
            ],
        },
        "ligand_descriptor_baseline": {
            "descriptor_names": list(DESCRIPTOR_NAMES),
            "descriptor_definition": (
                "RDKit heavy atoms, molecular weight, Labute ASA, TPSA, cLogP, "
                "rotatable bonds and ring count"
            ),
            "rdkit_version": rdBase.rdkitVersion,
            "preprocessing": (
                "descriptor means and sample SDs fitted on pilot rows only"
            ),
            "predictors_at_deployment": "seven ligand descriptors; no target scores",
        },
        "selected_target_pc1_baseline": {
            "definition": (
                "one deterministic pilot SVD factor fitted to the selected-target raw "
                "score block, followed by pilot-only univariate least-squares loadings"
            ),
            "purpose": (
                "tests whether multivariate panel reconstruction adds information beyond "
                "a shared selected-target score axis"
            ),
        },
        "estimand_and_comparison_contract": {
            "residual_order": (
                "split-local raw imputation, raw-unit row centering, then pilot-only "
                "residual-column centering and scaling"
            ),
            "primary_cross_panel_endpoint": (
                "common-P full-profile policy metrics; selected observed scores are copied"
            ),
            "omitted_target_endpoint": (
                "a direct predictive endpoint within one fixed panel, and for paired raw "
                "versus derived-residual comparisons; not a cross-panel primary metric"
            ),
            "docking44_observation_sensitivity": (
                "raw-score errors are additionally evaluated only on originally observed "
                "cells; residual sensitivity uses originally complete rows because every "
                "row mean depends on the full target profile"
            ),
        },
        "panel_objectives": {
            "residual_r2": "1-r^2; unsigned linear predictability with sign learned by ridge",
            "residual_absolute": "1-|r|; absolute-dependence sensitivity",
            "residual_signed": "1-r; positive-signed redundancy only",
            "raw_r2": "1-r^2 on the non-row-centred target correlation map",
        },
        "optimizer": (
            "the same exact HiGHS MILP p-median solver is used for pilot, full-source, "
            "and complement-specific panels"
        ),
        "domain_reconstruction": {
            "evaluation": (
                "one common half-domain row-disjoint support per paired replicate; "
                "metrics restricted to originally complete outcome rows"
            ),
            "calibration_designs": [
                "within_domain",
                "source_library_row_disjoint",
                "opposite_mw_domain",
            ],
            "metric_standardization": (
                "evaluation-reference means and SDs are used only to place truth and "
                "back-transformed predictions on a common reporting scale; selection, "
                "RidgeCV and prediction remain pilot-only"
            ),
        },
        "random_controls": {
            "reconstruction_panels_per_calibration": int(
                args.random_reconstruction_panels
            ),
            "coverage_panels_per_calibration": int(args.random_coverage_panels),
        },
        "datasets": {
            dataset.name: {
                "rows": int(len(dataset.matrix)),
                "targets": int(dataset.matrix.shape[1]),
                "support_rule": dataset.support_rule,
                "input_sha256": {
                    str(path.relative_to(PACKAGE)): sha256_file(path)
                    for path in dataset.input_paths
                },
            }
            for dataset in datasets
        },
        "code_sha256": {
            str(Path(__file__).resolve().relative_to(PACKAGE)): sha256_file(
                Path(__file__).resolve()
            ),
            "analysis/test_public_panel_domain_utility.py": sha256_file(
                PACKAGE / "analysis" / "test_public_panel_domain_utility.py"
            ),
        },
        "claim_boundary": (
            "The outputs test within-library computational score reconstruction, exact "
            "Vina-map coverage, and two post-hoc molecular-weight domains. They do not "
            "validate experimental affinity, pose quality, docking quality, biological "
            "target coverage, or transport to a new chemical population. Full-source "
            "and complement-specific panels are non-deployable diagnostic ceilings."
        ),
    }
    write_json(output / "metadata.json", metadata)
    (output / "README.md").write_text(
        build_readme(
            reconstruction_summary,
            coverage_summary,
            domain_summary,
            domain_reconstruction_contrast_summary,
            costs,
        ),
        encoding="utf-8",
    )
    checksum_files = [*tables, "metadata.json", "README.md"]
    write_json(
        output / "output_checksums.json",
        {
            "algorithm": "sha256",
            "files": {
                filename: sha256_file(output / filename)
                for filename in sorted(checksum_files)
            },
        },
    )
    print(
        json.dumps(
            {
                "output_dir": str(output),
                "datasets": list(requested),
                "reconstruction_records": int(len(reconstruction)),
                "coverage_records": int(len(coverage)),
                "domain_reconstruction_records": int(len(domain_reconstruction)),
                "tables": int(len(tables)),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
