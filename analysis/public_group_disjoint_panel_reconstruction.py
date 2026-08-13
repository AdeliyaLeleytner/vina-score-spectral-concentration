#!/usr/bin/env python3
"""Deployable k=8 score reconstruction on chemical-group-disjoint folds.

This compact strict-public sensitivity reuses the frozen score matrices and the
group definitions from ``public_scaffold_holdout_recovery.py``.  In each of five
fixed chemical-group folds, and in a fixed matched-random-row comparator, a
200- or 500-ligand pilot alone determines target imputation, target scaling, the
residual 1-r^2 map, an exact k=8 medoid panel, and a RidgeCV reconstruction of
the omitted target scores.  Prediction for the held-out fold receives only the
eight selected raw standardized target scores.  The all-target held-out row mean
is used only to define a secondary residual truth and is never a predictor.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

try:
    from . import public_panel_domain_utility as utility
    from . import public_scaffold_holdout_recovery as groups_mod
except ImportError:  # pragma: no cover - direct execution
    import public_panel_domain_utility as utility  # type: ignore
    import public_scaffold_holdout_recovery as groups_mod  # type: ignore


PACKAGE = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PACKAGE / "results" / "public_group_disjoint_panel_reconstruction"
DEFAULT_REPETITIONS_PER_FOLD = 5
DEFAULT_PANEL_K = 8
BASE_SEED = 202_608_26
PRIMARY_TRUTH = "calibration_standardized_raw"
PRIMARY_BASIS = "split_imputed_all_rows"
SUMMARY_METRICS = (
    "variance_weighted_r2",
    "pooled_calibration_standardized_rmse",
    "median_target_pearson",
    "median_target_spearman",
    "median_ligand_profile_pearson",
    "median_ligand_profile_spearman",
    "full_profile_policy_variance_weighted_r2",
    "full_profile_policy_pooled_calibration_standardized_rmse",
    "full_profile_policy_median_ligand_profile_pearson",
    "full_profile_policy_median_ligand_profile_spearman",
    "median_target_calibration_threshold_lower_5pct_precision",
    "median_target_calibration_threshold_lower_5pct_recall",
    "median_target_equal_budget_lower_5pct_overlap",
)


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
        raise ValueError("non-finite JSON value")
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(json_ready(payload), indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def fixed_splits(
    matrix: np.ndarray,
    groups: np.ndarray,
    *,
    seed: int,
) -> list[dict[str, Any]]:
    """Return the fixed group and matched-random designs used by the map audit."""

    result: list[dict[str, Any]] = []
    splitter = GroupKFold(n_splits=groups_mod.FOLDS)
    for fold, (group_pool, group_held) in enumerate(
        splitter.split(matrix, groups=groups), start=1
    ):
        if set(groups[group_pool]) & set(groups[group_held]):
            raise RuntimeError("chemical-group leakage")
        rng = np.random.default_rng(seed + 1_000_000 + 100_000 * fold)
        order = rng.permutation(len(matrix))
        random_held = order[: len(group_held)]
        random_pool = order[len(group_held) :]
        for design, pool, held in (
            ("chemical_group", group_pool, group_held),
            ("matched_random_row", random_pool, random_held),
        ):
            result.append(
                {
                    "holdout_design": design,
                    "fold": int(fold),
                    "pool_index": np.asarray(pool, dtype=np.int64),
                    "held_index": np.asarray(held, dtype=np.int64),
                    "pool_ligands": int(len(pool)),
                    "held_ligands": int(len(held)),
                    "pool_groups": int(len(set(groups[pool]))),
                    "held_groups": int(len(set(groups[held]))),
                    "group_overlap": int(len(set(groups[pool]) & set(groups[held]))),
                }
            )
    return result


def strip_residual_tail_metrics(record: dict[str, Any]) -> dict[str, Any]:
    if record["truth_surface"] != "raw_row_centered_residual_outcome":
        return record
    return {
        key: value
        for key, value in record.items()
        if "lower_5pct" not in key
    }


def analyze_dataset(
    dataset_name: str,
    matrix: np.ndarray,
    groups: np.ndarray,
    targets: tuple[str, ...],
    *,
    split_seed: int,
    sampling_seed: int,
    repetitions_per_fold: int,
    panel_k: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if repetitions_per_fold < 1 or panel_k >= matrix.shape[1]:
        raise ValueError("invalid repetition count or panel size")
    metric_records: list[dict[str, Any]] = []
    target_records: list[pd.DataFrame] = []
    selection_records: list[dict[str, Any]] = []
    fold_records: list[dict[str, Any]] = []
    for split in fixed_splits(matrix, groups, seed=split_seed):
        pool = split["pool_index"]
        held = split["held_index"]
        fold_records.append(
            {
                "dataset": dataset_name,
                **{key: value for key, value in split.items() if not key.endswith("_index")},
                "pool_held_row_overlap": int(len(np.intersect1d(pool, held))),
                "pool_missing_cells": int(np.isnan(matrix[pool]).sum()),
                "held_missing_cells": int(np.isnan(matrix[held]).sum()),
            }
        )
        for size in groups_mod.CALIBRATION_SIZES:
            if size >= len(pool):
                raise ValueError("pilot size exceeds pool")
            rng = np.random.default_rng(
                sampling_seed
                + 10_000_000 * split["fold"]
                + 100_000 * size
                + (0 if split["holdout_design"] == "chemical_group" else 1)
            )
            for repetition in range(repetitions_per_fold):
                pilot_local = np.sort(rng.choice(len(pool), size=size, replace=False))
                pilot = pool[pilot_local]
                if np.intersect1d(pilot, held).size:
                    raise RuntimeError("pilot and held-out folds overlap")
                transform = utility.split_transform(matrix[pilot], matrix[held])
                geometries = utility.map_geometries(transform.calibration_raw)
                distance = utility.objective_map(geometries, "residual_r2")
                panel = utility.exact_k_medoids(distance, panel_k)
                aggregates, target_frames = utility.fit_reconstruction(
                    transform,
                    panel,
                    targets,
                    np.isfinite(matrix[pilot]),
                    np.isfinite(matrix[held]),
                )
                base = {
                    "dataset": dataset_name,
                    "holdout_design": split["holdout_design"],
                    "fold": split["fold"],
                    "pilot_ligands": int(size),
                    "held_out_ligands": int(len(held)),
                    "repetition": int(repetition),
                    "panel_method": "pilot_exact_residual_r2",
                    "panel_k": int(panel_k),
                    "panel_indices": "|".join(str(int(value)) for value in panel),
                    "panel_targets": "|".join(targets[int(value)] for value in panel),
                    "pilot_held_row_overlap": 0,
                    "pilot_only_preprocessing": True,
                    "pilot_imputed_cells": int(np.isnan(matrix[pilot]).sum()),
                    "held_imputed_cells": int(np.isnan(matrix[held]).sum()),
                    "maximum_absolute_pilot_mean": float(
                        np.max(np.abs(transform.raw_means))
                    ),
                    "minimum_pilot_target_sd": float(np.min(transform.raw_sds)),
                }
                for aggregate, target_frame in zip(
                    aggregates, target_frames, strict=True
                ):
                    cleaned = strip_residual_tail_metrics(dict(aggregate))
                    metric_records.append({**base, **cleaned})
                    if aggregate["truth_surface"] == "raw_row_centered_residual_outcome":
                        tail_columns = [
                            column for column in target_frame if "lower_5pct" in column
                        ]
                        target_frame = target_frame.drop(columns=tail_columns)
                    target_frame = target_frame.copy()
                    for key, value in reversed(list(base.items())):
                        target_frame.insert(0, key, value)
                    target_frame["truth_surface"] = aggregate["truth_surface"]
                    target_frame["outcome_basis"] = aggregate["outcome_basis"]
                    target_records.append(target_frame)
                for target_index in panel:
                    selection_records.append(
                        {
                            **base,
                            "target_index": int(target_index),
                            "target": targets[int(target_index)],
                        }
                    )
    return (
        pd.DataFrame.from_records(metric_records),
        pd.concat(target_records, ignore_index=True),
        pd.DataFrame.from_records(selection_records),
        pd.DataFrame.from_records(fold_records),
    )


def summarize(metrics: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "dataset",
        "holdout_design",
        "pilot_ligands",
        "truth_surface",
        "outcome_basis",
        "predictive_model",
    ]
    rows: list[dict[str, Any]] = []
    for values, frame in metrics.groupby(keys, sort=True):
        record = dict(zip(keys, values, strict=True))
        record["replicates"] = int(len(frame))
        for metric in SUMMARY_METRICS:
            available = frame[metric].dropna() if metric in frame else pd.Series(dtype=float)
            if available.empty:
                continue
            record[f"{metric}_mean"] = float(available.mean())
            record[f"{metric}_median"] = float(available.median())
            record[f"{metric}_q025"] = float(np.quantile(available, 0.025))
            record[f"{metric}_q975"] = float(np.quantile(available, 0.975))
        rows.append(record)
    return pd.DataFrame.from_records(rows)


def descriptive_group_minus_random(metrics: pd.DataFrame) -> pd.DataFrame:
    index = [
        "dataset",
        "fold",
        "pilot_ligands",
        "repetition",
        "truth_surface",
        "outcome_basis",
        "predictive_model",
    ]
    rows: list[dict[str, Any]] = []
    for metric in SUMMARY_METRICS:
        if metric not in metrics:
            continue
        pivot = metrics.pivot_table(
            index=index,
            columns="holdout_design",
            values=metric,
            aggfunc="first",
        ).reset_index()
        if not {"chemical_group", "matched_random_row"} <= set(pivot):
            raise RuntimeError("descriptive comparator arm missing")
        pivot["difference"] = pivot.chemical_group - pivot.matched_random_row
        for values, frame in pivot.groupby(
            [
                "dataset",
                "pilot_ligands",
                "truth_surface",
                "outcome_basis",
                "predictive_model",
            ],
            sort=True,
        ):
            difference = frame.difference.dropna().to_numpy(dtype=float)
            if not len(difference):
                continue
            rows.append(
                {
                    "dataset": values[0],
                    "pilot_ligands": int(values[1]),
                    "truth_surface": values[2],
                    "outcome_basis": values[3],
                    "predictive_model": values[4],
                    "metric": metric,
                    "descriptive_replicates": int(len(difference)),
                    "group_minus_random_mean": float(difference.mean()),
                    "group_minus_random_median": float(np.median(difference)),
                    "group_minus_random_q025": float(np.quantile(difference, 0.025)),
                    "group_minus_random_q975": float(np.quantile(difference, 0.975)),
                }
            )
    return pd.DataFrame.from_records(rows)


def build(
    docking44_path: Path,
    dockstring_path: Path,
    *,
    repetitions_per_fold: int,
    panel_k: int,
) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
    d44 = groups_mod.load_docking44(docking44_path)
    ds = groups_mod.load_dockstring(dockstring_path)
    datasets = [
        ("Docking-44", d44[0], d44[1], tuple(groups_mod.DOCKING44_TARGETS), d44[4]),
        (
            "DOCKSTRING-58",
            ds[0],
            ds[1],
            tuple(
                column
                for column in pd.read_csv(dockstring_path, sep="\t", nrows=0).columns
                if column not in {"inchikey", "smiles"}
            ),
            ds[4],
        ),
    ]
    metric_parts: list[pd.DataFrame] = []
    target_parts: list[pd.DataFrame] = []
    selection_parts: list[pd.DataFrame] = []
    fold_parts: list[pd.DataFrame] = []
    dataset_metadata: dict[str, Any] = {}
    for position, (name, matrix, groups, targets, metadata) in enumerate(datasets):
        parts = analyze_dataset(
            name,
            matrix,
            groups,
            targets,
            split_seed=groups_mod.BASE_SEED + position,
            sampling_seed=BASE_SEED + position,
            repetitions_per_fold=repetitions_per_fold,
            panel_k=panel_k,
        )
        metric_parts.append(parts[0])
        target_parts.append(parts[1])
        selection_parts.append(parts[2])
        fold_parts.append(parts[3])
        dataset_metadata[name] = metadata
    metrics = pd.concat(metric_parts, ignore_index=True)
    target_metrics = pd.concat(target_parts, ignore_index=True)
    selections = pd.concat(selection_parts, ignore_index=True)
    fold_contract = pd.concat(fold_parts, ignore_index=True)
    summary_table = summarize(metrics)
    contrasts = descriptive_group_minus_random(metrics)
    summary = {
        "schema_version": "2.0.0",
        "analysis_status": "strict_public_group_disjoint_panel_reconstruction_sensitivity",
        "producer": "analysis/public_group_disjoint_panel_reconstruction.py",
        "configuration": {
            "folds": groups_mod.FOLDS,
            "pilot_sizes": list(groups_mod.CALIBRATION_SIZES),
            "repetitions_per_fold": int(repetitions_per_fold),
            "replicates_per_dataset_design_size": int(
                groups_mod.FOLDS * repetitions_per_fold
            ),
            "panel_k": int(panel_k),
            "panel_objective": "exact k-medoids on pilot residual distance 1-r^2",
            "base_seed": BASE_SEED,
            "primary_truth": PRIMARY_TRUTH,
            "primary_outcome_basis": PRIMARY_BASIS,
            "primary_deployment_contract": (
                "pilot-only target-mean imputation and scaling; eight selected raw "
                "standardized scores are the only held-out predictors; selected raw "
                "scores are copied into the full-profile policy output"
            ),
        },
        "datasets": dataset_metadata,
        "primary_results": summary_table.loc[
            summary_table.truth_surface.eq(PRIMARY_TRUTH)
            & summary_table.outcome_basis.eq(PRIMARY_BASIS)
            & summary_table.predictive_model.eq("multivariate_ridge")
        ].to_dict(orient="records"),
        "claim_boundary": (
            "This is a repeated-pilot sensitivity on fixed within-library held-out folds, "
            "not a confidence interval or evidence of transport to another collection. "
            "The chemical-group-minus-random contrast is descriptive because the two "
            "arms use different held-out supports, not a paired causal analogue-leak effect. "
            "The selected-target PC1 model is a declared low-rank comparator; multivariate "
            "ridge is primary. Residual reconstruction is a derived diagnostic; its lower tail is omitted "
            "and is not interpreted as favorable docking."
        ),
        "dependency_sources": {
            "public_panel_domain_utility.py": sha256_file(
                PACKAGE / "analysis" / "public_panel_domain_utility.py"
            ),
            "public_scaffold_holdout_recovery.py": sha256_file(
                PACKAGE / "analysis" / "public_scaffold_holdout_recovery.py"
            ),
        },
        "inputs": {
            "docking44": {
                "path": str(docking44_path.resolve().relative_to(PACKAGE)),
                "sha256": sha256_file(docking44_path),
            },
            "dockstring58": {
                "path": str(dockstring_path.resolve().relative_to(PACKAGE)),
                "sha256": sha256_file(dockstring_path),
            },
        },
    }
    return summary, {
        "reconstruction_metrics.csv": metrics,
        "reconstruction_target_metrics.csv": target_metrics,
        "panel_selections.csv": selections,
        "fold_contract.csv": fold_contract,
        "reconstruction_summary.csv": summary_table,
        "group_minus_random_descriptive.csv": contrasts,
    }


def write_artifact(
    output: Path,
    summary: dict[str, Any],
    tables: dict[str, pd.DataFrame],
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "summary.json", summary)
    for filename, frame in tables.items():
        frame.to_csv(output / filename, index=False, float_format="%.17g")
    (output / "README.md").write_text(
        """# Chemical-group-disjoint k=8 reconstruction sensitivity

For each of five fixed chemical-group folds and fixed matched-random-row folds,
five deterministic 200- and 500-ligand pilots independently fit preprocessing,
select an exact residual-1-r^2 k=8 panel, and fit omitted-target RidgeCV models.
Held-out prediction receives only the eight selected raw standardized scores.

The primary endpoint is calibration-standardized **raw** omitted-target score
reconstruction. Lower-tail retrieval is reported only for this raw-score target.
The residual truth is secondary and uses the all-target row mean only as an
outcome definition, never as a predictor. Group-minus-random differences are
descriptive because their held-out supports differ. Ranges are repeated-pilot
sensitivities conditional on the fixed datasets, not confidence intervals.

```bash
.venv/bin/python analysis/public_group_disjoint_panel_reconstruction.py
.venv/bin/python -m pytest -q analysis/test_public_group_disjoint_panel_reconstruction.py
```
""",
        encoding="utf-8",
    )
    checksums = {
        path.name: sha256_file(path)
        for path in sorted(output.iterdir())
        if path.is_file() and path.name != "output_checksums.json"
    }
    write_json(
        output / "output_checksums.json",
        {
            "algorithm": "sha256",
            "scope": "all output files except this checksum manifest",
            "files": checksums,
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--docking44", type=Path, default=groups_mod.DEFAULT_DOCKING44
    )
    parser.add_argument(
        "--dockstring", type=Path, default=groups_mod.DEFAULT_DOCKSTRING
    )
    parser.add_argument(
        "--repetitions-per-fold", type=int, default=DEFAULT_REPETITIONS_PER_FOLD
    )
    parser.add_argument("--panel-k", type=int, default=DEFAULT_PANEL_K)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary, tables = build(
        args.docking44,
        args.dockstring,
        repetitions_per_fold=args.repetitions_per_fold,
        panel_k=args.panel_k,
    )
    write_artifact(args.output_dir, summary, tables)
    print(pd.DataFrame(summary["primary_results"]).to_string(index=False))
    print(args.output_dir / "summary.json")


if __name__ == "__main__":
    main()
