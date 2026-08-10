#!/usr/bin/env python3
"""Recover residual target geometry from small random calibration panels.

The primary estimand is the Spearman agreement between the target-correlation
geometry estimated from an outcome-blind random ligand panel and the geometry of
the complete docking matrix.  OAS covariance shrinkage is evaluated as a
finite-sample correction for participation-ratio dimension.  For DOCKSTRING's
fixed 20-kinase support, the script also measures preservation of the previously
frozen experimental co-selective target-pair endpoint.

Only aggregate results are written; sampled molecule identifiers are not saved.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.covariance import OAS
from sklearn.model_selection import GroupKFold

from replicated_pair_retrieval import retrieval_metrics
from residual_mechanism_analysis import (
    AnalysisConfig,
    DatasetBundle,
    load_docking44,
    load_dockstring as load_dockstring_bundle,
    two_way_center,
)
from residual_target_geometry_validation import PKIS1_TARGET_MAP


PACKAGE = Path(__file__).resolve().parents[1]
DEFAULT_DOCKSTRING = PACKAGE / "data/frozen/dockstring-dataset.tsv.gz"
DEFAULT_ENDPOINT = PACKAGE / "results/replicated_pair_retrieval/target_pairs.csv"
DEFAULT_OUTPUT = PACKAGE / "results/calibration_panel_recovery"
DEFAULT_SIZES = (50, 100, 200, 500, 1_000, 2_000, 5_000)


def covariance_to_correlation(covariance: np.ndarray) -> np.ndarray:
    sd = np.sqrt(np.clip(np.diag(covariance), 1e-15, None))
    return covariance / np.outer(sd, sd)


def geometry(matrix: np.ndarray, shrink: bool = False) -> np.ndarray:
    centered = two_way_center(np.asarray(matrix, dtype=np.float64))
    if shrink:
        return covariance_to_correlation(OAS().fit(centered).covariance_)
    return np.corrcoef(centered, rowvar=False)


def correlation_pr(correlation: np.ndarray) -> float:
    return float(np.trace(correlation) ** 2 / np.sum(correlation**2))


def upper(matrix: np.ndarray) -> np.ndarray:
    return matrix[np.triu_indices(len(matrix), k=1)]


def top_fraction_labels(values: np.ndarray, fraction: float = 0.10) -> np.ndarray:
    labels = np.zeros(len(values), dtype=bool)
    count = int(np.ceil(fraction * len(values)))
    labels[np.argsort(values, kind="mergesort")[-count:]] = True
    return labels


def load_dockstring_matrix(path: Path) -> tuple[np.ndarray, list[str]]:
    frame = pd.read_csv(path, sep="\t")
    targets = [column for column in frame if column not in {"inchikey", "smiles"}]
    numeric = frame[targets].to_numpy(dtype=np.float64)
    complete = np.isfinite(numeric).all(axis=1)
    matrix = np.minimum(numeric[complete], 0.0)
    return matrix, targets


def scaffold_holdout_analysis(
    bundle: DatasetBundle,
    sizes: tuple[int, ...],
    repetitions: int,
    seed: int,
) -> pd.DataFrame:
    """Estimate geometry from training scaffolds and test against unseen scaffolds."""
    rng = np.random.default_rng(seed)
    records: list[dict[str, object]] = []
    splitter = GroupKFold(n_splits=5)
    for fold, (train_index, test_index) in enumerate(
        splitter.split(bundle.matrix, groups=bundle.groups), start=1
    ):
        train_groups = set(np.asarray(bundle.groups)[train_index].tolist())
        test_groups = set(np.asarray(bundle.groups)[test_index].tolist())
        if train_groups.intersection(test_groups):
            raise AssertionError("scaffold leakage in GroupKFold")
        held_out_geometry = geometry(bundle.matrix[test_index])
        held_out_upper = upper(held_out_geometry)
        held_out_pr = correlation_pr(held_out_geometry)
        for size in sizes:
            if size >= len(train_index):
                continue
            for repetition in range(repetitions):
                selected = rng.choice(train_index, size=size, replace=False)
                empirical = geometry(bundle.matrix[selected])
                shrunk = geometry(bundle.matrix[selected], shrink=True)
                records.append(
                    {
                        "dataset": bundle.name,
                        "fold": fold,
                        "calibration_ligands": size,
                        "repetition": repetition,
                        "train_scaffold_groups": len(train_groups),
                        "held_out_scaffold_groups": len(test_groups),
                        "held_out_ligands": int(len(test_index)),
                        "geometry_spearman_to_held_out_scaffolds": float(
                            spearmanr(upper(empirical), held_out_upper).statistic
                        ),
                        "oas_geometry_spearman_to_held_out_scaffolds": float(
                            spearmanr(upper(shrunk), held_out_upper).statistic
                        ),
                        "held_out_scaffold_pr": held_out_pr,
                        "empirical_pr": correlation_pr(empirical),
                        "oas_pr": correlation_pr(shrunk),
                    }
                )
    return pd.DataFrame.from_records(records)


def experimental_labels(
    target_order: list[str], endpoint_path: Path
) -> tuple[np.ndarray, np.ndarray]:
    common_targets = list(PKIS1_TARGET_MAP)
    indices = np.asarray([target_order.index(target) for target in common_targets])
    endpoint = pd.read_csv(endpoint_path)
    lookup = {
        tuple(sorted((str(row.target_a), str(row.target_b)))): bool(
            row.primary_replicated_positive
        )
        for row in endpoint.itertuples()
    }
    tri = np.triu_indices(len(common_targets), k=1)
    labels = np.asarray(
        [
            lookup[tuple(sorted((common_targets[i], common_targets[j])))]
            for i, j in zip(*tri, strict=True)
        ],
        dtype=bool,
    )
    return indices, labels


def panel_metrics(
    sample: np.ndarray,
    reference_geometry: np.ndarray,
    reference_tail_labels: np.ndarray,
    common_indices: np.ndarray | None,
    experimental_pair_labels: np.ndarray | None,
) -> dict[str, float]:
    empirical = geometry(sample, shrink=False)
    shrunk = geometry(sample, shrink=True)
    empirical_upper = upper(empirical)
    reference_upper = upper(reference_geometry)
    tail_auc, tail_ap = retrieval_metrics(reference_tail_labels, empirical_upper).values()
    metrics = {
        "geometry_spearman": float(
            spearmanr(empirical_upper, reference_upper).statistic
        ),
        "oas_geometry_spearman": float(
            spearmanr(upper(shrunk), reference_upper).statistic
        ),
        "empirical_pr": correlation_pr(empirical),
        "oas_pr": correlation_pr(shrunk),
        "reference_top10_pair_roc_auc": float(tail_auc),
        "reference_top10_pair_average_precision": float(tail_ap),
    }
    if common_indices is not None and experimental_pair_labels is not None:
        common_geometry = geometry(sample[:, common_indices], shrink=False)
        exp_auc, exp_ap = retrieval_metrics(
            experimental_pair_labels, upper(common_geometry)
        ).values()
        metrics.update(
            {
                "experimental_pair_roc_auc": float(exp_auc),
                "experimental_pair_average_precision": float(exp_ap),
            }
        )
    return metrics


def analyze_dataset(
    name: str,
    matrix: np.ndarray,
    target_order: list[str],
    sizes: tuple[int, ...],
    repetitions: int,
    seed: int,
    endpoint_path: Path | None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    reference = geometry(matrix)
    reference_upper = upper(reference)
    reference_labels = top_fraction_labels(reference_upper)
    common_indices: np.ndarray | None = None
    experimental_pair_labels: np.ndarray | None = None
    if endpoint_path is not None:
        common_indices, experimental_pair_labels = experimental_labels(
            target_order, endpoint_path
        )
    rng = np.random.default_rng(seed)
    records: list[dict[str, object]] = []
    for size in sizes:
        if size >= len(matrix):
            continue
        for repetition in range(repetitions):
            selected = rng.choice(len(matrix), size=size, replace=False)
            records.append(
                {
                    "dataset": name,
                    "calibration_ligands": size,
                    "repetition": repetition,
                    **panel_metrics(
                        matrix[selected],
                        reference,
                        reference_labels,
                        common_indices,
                        experimental_pair_labels,
                    ),
                }
            )
    full_experimental: dict[str, float] | None = None
    if common_indices is not None and experimental_pair_labels is not None:
        auc, ap = retrieval_metrics(
            experimental_pair_labels, upper(geometry(matrix[:, common_indices]))
        ).values()
        full_experimental = {
            "roc_auc": float(auc),
            "average_precision": float(ap),
        }
    metadata: dict[str, object] = {
        "dataset": name,
        "full_ligands": int(len(matrix)),
        "targets": int(matrix.shape[1]),
        "reference_residual_pr": correlation_pr(reference),
        "reference_experimental_pair_metrics": full_experimental,
        "sampling": "simple random without replacement; outcome-blind",
        "repetitions_per_size": repetitions,
        "seed": seed,
    }
    return pd.DataFrame.from_records(records), metadata


def summarize(records: pd.DataFrame) -> pd.DataFrame:
    identifier = {"dataset", "calibration_ligands", "repetition"}
    metrics = [column for column in records if column not in identifier]
    rows: list[dict[str, object]] = []
    for (dataset, size), group in records.groupby(
        ["dataset", "calibration_ligands"], sort=True
    ):
        row: dict[str, object] = {
            "dataset": dataset,
            "calibration_ligands": int(size),
        }
        for metric in metrics:
            values = group[metric].dropna().to_numpy(dtype=float)
            if not len(values):
                continue
            row[f"{metric}_mean"] = float(np.mean(values))
            row[f"{metric}_sd"] = float(np.std(values, ddof=1))
            row[f"{metric}_q025"] = float(np.quantile(values, 0.025))
            row[f"{metric}_q975"] = float(np.quantile(values, 0.975))
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dockstring", type=Path, default=DEFAULT_DOCKSTRING)
    parser.add_argument("--endpoint", type=Path, default=DEFAULT_ENDPOINT)
    parser.add_argument("--sizes", default=",".join(map(str, DEFAULT_SIZES)))
    parser.add_argument("--repetitions", type=int, default=100)
    parser.add_argument("--seed", type=int, default=202_608_03)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    sizes = tuple(int(value) for value in args.sizes.split(",") if value)
    dockstring_matrix, dockstring_targets = load_dockstring_matrix(args.dockstring)
    docking44 = load_docking44()
    ds_records, ds_metadata = analyze_dataset(
        "DOCKSTRING-58",
        dockstring_matrix,
        dockstring_targets,
        sizes,
        args.repetitions,
        args.seed,
        args.endpoint,
    )
    d44_records, d44_metadata = analyze_dataset(
        "Docking-44",
        docking44.matrix,
        docking44.targets,
        sizes,
        args.repetitions,
        args.seed + 1,
        None,
    )
    records = pd.concat([ds_records, d44_records], ignore_index=True)
    summary = summarize(records)
    dockstring_support = load_dockstring_bundle(
        AnalysisConfig(dockstring_sample_size=15_000, dockstring_sample_seed=71)
    )
    scaffold_records = pd.concat(
        [
            scaffold_holdout_analysis(
                dockstring_support,
                tuple(size for size in (200, 500) if size in sizes),
                25,
                args.seed + 2,
            ),
            scaffold_holdout_analysis(
                docking44,
                tuple(size for size in (200, 500) if size in sizes),
                25,
                args.seed + 3,
            ),
        ],
        ignore_index=True,
    )
    scaffold_summary = (
        scaffold_records.groupby(["dataset", "calibration_ligands"], as_index=False)
        .agg(
            mean_geometry_spearman_to_held_out_scaffolds=(
                "geometry_spearman_to_held_out_scaffolds",
                "mean",
            ),
            sd_geometry_spearman_to_held_out_scaffolds=(
                "geometry_spearman_to_held_out_scaffolds",
                "std",
            ),
            minimum_geometry_spearman_to_held_out_scaffolds=(
                "geometry_spearman_to_held_out_scaffolds",
                "min",
            ),
            maximum_geometry_spearman_to_held_out_scaffolds=(
                "geometry_spearman_to_held_out_scaffolds",
                "max",
            ),
        )
    )
    metadata = {
        "analysis_status": "exploratory_science_only",
        "estimand": (
            "recovery of the full-matrix two-way-centered target-correlation geometry"
        ),
        "finite_sample_correction": "Oracle Approximating Shrinkage (OAS)",
        "datasets": [ds_metadata, d44_metadata],
        "cost_examples": {
            dataset["dataset"]: {
                str(size): {
                    "full_cells": int(dataset["full_ligands"] * dataset["targets"]),
                    "pilot_cells": int(size * dataset["targets"]),
                    "cell_reduction_fraction": float(
                        1 - size / dataset["full_ligands"]
                    ),
                    "fold_reduction": float(dataset["full_ligands"] / size),
                }
                for size in (200, 500)
            }
            for dataset in (ds_metadata, d44_metadata)
        },
        "scaffold_holdout_sensitivity": {
            "split": "five-fold GroupKFold by the frozen dataset-specific chemical groups",
            "DOCKSTRING_support": (
                "fixed simple-random 15,000-ligand support, seed 71; Murcko groups"
            ),
            "calibration_sizes": [size for size in (200, 500) if size in sizes],
            "repetitions_per_fold": 25,
        },
        "claim_boundary": (
            "The calibration panel estimates Vina residual geometry for a representative "
            "source library. It does not reconstruct ligand-level scores, validate poses, "
            "or guarantee transfer to a chemically shifted deployment library."
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records.to_csv(args.output_dir / "replicate_metrics.csv", index=False)
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    scaffold_records.to_csv(
        args.output_dir / "scaffold_holdout_metrics.csv", index=False
    )
    scaffold_summary.to_csv(
        args.output_dir / "scaffold_holdout_summary.csv", index=False
    )
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
