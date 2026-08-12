#!/usr/bin/env python3
"""Strict-public chemical-group-held-out recovery of Vina residual target maps.

Five chemical-group folds are formed on each public score matrix.  Calibration
ligands are sampled only from four folds and their residual target map is
compared with the map estimated on the held-out chemical groups.  A matched
random-row-holdout design uses the same calibration and evaluation sizes, so the
incremental effect of holding out chemical groups can be described directly.
Any Docking-44 missing values are imputed from calibration-pool target means and
the same means are applied to the held-out fold.  The analysis does not
establish transport to a different chemical population.

Cyclic molecules are grouped by non-isomeric Bemis--Murcko scaffold. Acyclic
molecules, for which an empty Murcko scaffold is uninformative, are grouped by the
first Standard InChIKey connectivity block of a standardised parent rather than by
source row. Exact foldwise maximum Morgan-Tanimoto similarity from every held-out
molecule to the calibration pool quantifies remaining analogue proximity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import GroupKFold


PACKAGE = Path(__file__).resolve().parents[1]
FROZEN = PACKAGE / "data" / "frozen"
DEFAULT_DOCKING44 = FROZEN / "df_final_v4.csv.gz"
DEFAULT_DOCKSTRING = FROZEN / "dockstring-dataset.tsv.gz"
DEFAULT_OUTPUT = PACKAGE / "results" / "public_scaffold_holdout_recovery"
SUPPORT_SIZE = 15_000
SUPPORT_SEED = 71
CALIBRATION_SIZES = (200, 500)
FOLDS = 5
REPETITIONS_PER_FOLD = 25
BASE_SEED = 202_608_05
MORGAN_RADIUS = 2
MORGAN_BITS = 2_048
SIMILARITY_THRESHOLDS = (0.4, 0.5, 0.7, 0.8, 0.9)

DOCKING44_TARGETS = [
    "1m2z", "1pbq", "1xoq", "2rh1", "2vt4", "2ydo", "2z5x", "3b66",
    "3kk6", "3ln1", "3rze", "4djh", "4ey7", "4iar", "4mqs", "4n6h",
    "5cxv", "5i71", "5tvn", "5u09", "5va1", "6cm4", "6kpf", "6kux",
    "6lqa", "6pdj", "6x3x", "6y1z", "7f8y", "7kwe", "7ljd", "7wc9",
    "7xnk", "7ym8", "8e9y", "8ef6", "8fhs", "8pjk", "8st0", "8wty",
    "8xvk", "8yn3", "9eo4", "V1A",
]


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


def row_center(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    return matrix - matrix.mean(axis=1, keepdims=True)


def target_correlation(matrix: np.ndarray) -> np.ndarray:
    matrix = row_center(matrix)
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    covariance = centered.T @ centered
    variance = np.diag(covariance)
    if matrix.ndim != 2 or matrix.shape[0] < 3 or np.any(variance <= 1e-14):
        raise ValueError("target correlation requires a nonconstant matrix")
    scale = np.sqrt(variance)
    result = covariance / np.outer(scale, scale)
    result = np.clip((result + result.T) / 2.0, -1.0, 1.0)
    np.fill_diagonal(result, 1.0)
    return result


def upper_triangle(matrix: np.ndarray) -> np.ndarray:
    return matrix[np.triu_indices(len(matrix), k=1)]


def spearman(first: np.ndarray, second: np.ndarray) -> float:
    first_rank = pd.Series(first).rank(method="average").to_numpy(dtype=float)
    second_rank = pd.Series(second).rank(method="average").to_numpy(dtype=float)
    value = float(np.corrcoef(first_rank, second_rank)[0, 1])
    if not np.isfinite(value):
        raise ValueError("Spearman agreement is not finite")
    return value


def chemical_groups_and_fingerprints(
    smiles: pd.Series, source_indices: np.ndarray
) -> tuple[np.ndarray, list[Any], dict[str, Any]]:
    """Build reproducible chemical groups and achiral Morgan fingerprints."""

    groups: list[str] = []
    fingerprints: list[Any] = []
    cyclic_rows = 0
    acyclic_rows = 0
    multi_fragment_rows = 0
    inchi_fallback_groups = 0
    uncharger = rdMolStandardize.Uncharger()
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=MORGAN_RADIUS,
        fpSize=MORGAN_BITS,
        includeChirality=False,
    )
    for text, source_index in zip(smiles, source_indices, strict=True):
        molecule = Chem.MolFromSmiles(str(text))
        if molecule is None:
            raise ValueError(f"invalid SMILES at source row {source_index}")
        if len(Chem.GetMolFrags(molecule)) > 1:
            multi_fragment_rows += 1
        parent = rdMolStandardize.FragmentParent(
            rdMolStandardize.Cleanup(molecule)
        )
        parent = uncharger.uncharge(parent)
        Chem.RemoveStereochemistry(parent)
        for atom in parent.GetAtoms():
            atom.SetIsotope(0)
        Chem.SanitizeMol(parent)
        Chem.GetSymmSSSR(parent)
        scaffold_text = MurckoScaffold.MurckoScaffoldSmiles(mol=parent)
        if scaffold_text:
            scaffold = Chem.MolFromSmiles(scaffold_text)
            if scaffold is None:
                raise ValueError(f"invalid Murcko scaffold at source row {source_index}")
            identity = Chem.MolToSmiles(
                scaffold,
                canonical=True,
                isomericSmiles=False,
            )
            groups.append(f"CYCLIC_MURCKO:{identity}")
            cyclic_rows += 1
        else:
            inchi_key = Chem.MolToInchiKey(parent)
            if inchi_key and "-" in inchi_key:
                identity = inchi_key.split("-", maxsplit=1)[0]
                groups.append(f"ACYCLIC_CONNECTIVITY_INCHI:{identity}")
            else:
                identity = Chem.MolToSmiles(
                    parent,
                    canonical=True,
                    isomericSmiles=False,
                )
                groups.append(f"ACYCLIC_CONNECTIVITY_SMILES_FALLBACK:{identity}")
                inchi_fallback_groups += 1
            acyclic_rows += 1
        fingerprints.append(generator.GetFingerprint(parent))
    group_array = np.asarray(groups, dtype=object)
    counts = pd.Series(group_array).value_counts()
    diagnostics = {
        "chemical_groups": int(len(counts)),
        "cyclic_rows": int(cyclic_rows),
        "acyclic_rows": int(acyclic_rows),
        "cyclic_murcko_groups": int(
            sum(str(value).startswith("CYCLIC_MURCKO:") for value in counts.index)
        ),
        "acyclic_connectivity_groups": int(
            sum(
                str(value).startswith("ACYCLIC_CONNECTIVITY_")
                for value in counts.index
            )
        ),
        "singleton_groups": int((counts == 1).sum()),
        "largest_group_size": int(counts.max()),
        "median_group_size": float(counts.median()),
        "rows_in_ten_largest_groups": int(counts.head(10).sum()),
        "duplicate_rows_beyond_one_per_group": int(len(group_array) - len(counts)),
        "multi_fragment_rows_before_parent_standardization": int(
            multi_fragment_rows
        ),
        "inchi_connectivity_fallback_rows": int(inchi_fallback_groups),
        "group_namespace_collision_count": 0,
    }
    return group_array, fingerprints, diagnostics


def duplicate_group_score_diagnostics(
    matrix: np.ndarray, groups: np.ndarray
) -> dict[str, int]:
    """Count whether multi-row chemical groups also duplicate target-score rows."""

    memberships: dict[str, list[int]] = {}
    for index, group in enumerate(groups):
        memberships.setdefault(str(group), []).append(index)
    duplicated = [indices for indices in memberships.values() if len(indices) > 1]
    identical_groups = 0
    identical_rows = 0
    for indices in duplicated:
        rows = np.asarray(matrix[indices], dtype=np.float64)
        if np.allclose(rows, rows[[0]], equal_nan=True, atol=0.0, rtol=0.0):
            identical_groups += 1
            identical_rows += len(indices)
    return {
        "multirow_chemical_groups": int(len(duplicated)),
        "multirow_groups_with_identical_score_rows": int(identical_groups),
        "rows_in_multirow_groups_with_identical_score_rows": int(identical_rows),
    }


def load_docking44(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, list[Any], np.ndarray, dict[str, Any]]:
    frame = pd.read_csv(
        path,
        usecols=DOCKING44_TARGETS + ["Cleaned SMILES", "Canonical SMILES"],
    )
    numeric = frame[DOCKING44_TARGETS].apply(pd.to_numeric, errors="coerce")
    clipped = numeric.clip(upper=0)
    matrix = clipped.to_numpy(dtype=np.float64)
    smiles = (
        frame["Cleaned SMILES"]
        .fillna(frame["Canonical SMILES"])
        .astype(str)
    )
    source_indices = np.arange(len(frame), dtype=np.int64)
    groups, fingerprints, group_diagnostics = chemical_groups_and_fingerprints(
        smiles, source_indices
    )
    score_duplicate_diagnostics = duplicate_group_score_diagnostics(matrix, groups)
    return matrix, groups, fingerprints, source_indices, {
        "source_rows": int(len(frame)),
        "analysis_rows": int(len(frame)),
        "targets": int(len(DOCKING44_TARGETS)),
        **group_diagnostics,
        **score_duplicate_diagnostics,
        "group_definition": (
            "RDKit cleanup, largest-fragment parent, uncharging, isotope removal and "
            "stereochemistry removal; cyclic: canonical non-isomeric Bemis-Murcko "
            "scaffold; acyclic: first Standard InChIKey connectivity block, with "
            "canonical non-isomeric SMILES fallback on InChI failure"
        ),
        "missing_cells_before_target_mean_imputation": int(
            numeric.isna().to_numpy().sum()
        ),
        "imputation_rule": (
            "within each split, calibration-pool target means are applied to both "
            "the calibration pool and held-out rows"
        ),
    }


def fill_from_calibration_pool(
    calibration_pool: np.ndarray, evaluation: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Impute both arrays using target means estimated only on the pool."""

    calibration_pool = np.asarray(calibration_pool, dtype=np.float64)
    evaluation = np.asarray(evaluation, dtype=np.float64)
    means = np.nanmean(calibration_pool, axis=0)
    if not np.isfinite(means).all():
        raise ValueError("a target is entirely missing in a calibration pool")
    filled_pool = np.where(np.isnan(calibration_pool), means, calibration_pool)
    filled_evaluation = np.where(np.isnan(evaluation), means, evaluation)
    if not np.isfinite(filled_pool).all() or not np.isfinite(filled_evaluation).all():
        raise ValueError("fold-local imputation produced a non-finite score")
    return filled_pool, filled_evaluation, means


def load_dockstring(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, list[Any], np.ndarray, dict[str, Any]]:
    frame = pd.read_csv(path, sep="\t")
    targets = [column for column in frame if column not in {"inchikey", "smiles"}]
    numeric = frame[targets].apply(pd.to_numeric, errors="coerce")
    complete = ~numeric.isna().any(axis=1)
    complete_matrix = np.minimum(
        numeric.loc[complete].to_numpy(dtype=np.float64), 0.0
    )
    complete_smiles = frame.loc[complete, "smiles"].reset_index(drop=True).astype(str)
    complete_source_indices = np.flatnonzero(complete.to_numpy())
    rng = np.random.default_rng(SUPPORT_SEED)
    selected = np.sort(
        rng.choice(len(complete_matrix), size=SUPPORT_SIZE, replace=False)
    )
    source_indices = complete_source_indices[selected]
    groups, fingerprints, group_diagnostics = chemical_groups_and_fingerprints(
        complete_smiles.iloc[selected], complete_source_indices[selected]
    )
    score_duplicate_diagnostics = duplicate_group_score_diagnostics(
        complete_matrix[selected], groups
    )
    return complete_matrix[selected], groups, fingerprints, source_indices, {
        "source_rows": int(len(frame)),
        "complete_rows": int(complete.sum()),
        "analysis_rows": SUPPORT_SIZE,
        "targets": int(len(targets)),
        **group_diagnostics,
        **score_duplicate_diagnostics,
        "group_definition": (
            "RDKit cleanup, largest-fragment parent, uncharging, isotope removal and "
            "stereochemistry removal; cyclic: canonical non-isomeric Bemis-Murcko "
            "scaffold; acyclic: first Standard InChIKey connectivity block, with "
            "canonical non-isomeric SMILES fallback on InChI failure"
        ),
        "support_seed": SUPPORT_SEED,
        "selected_complete_row_index_sha256": hashlib.sha256(
            np.asarray(selected, dtype="<i8").tobytes()
        ).hexdigest(),
    }


def exact_maximum_tanimoto_to_pool(
    held_out_fingerprints: list[Any],
    pool_fingerprints: list[Any],
) -> np.ndarray:
    """Return every held-out ligand's exact maximum similarity to the full pool."""

    if not held_out_fingerprints or not pool_fingerprints:
        raise ValueError("fingerprint collections must be non-empty")
    maxima = np.empty(len(held_out_fingerprints), dtype=np.float64)
    for position, fingerprint in enumerate(held_out_fingerprints):
        maxima[position] = max(
            DataStructs.BulkTanimotoSimilarity(fingerprint, pool_fingerprints)
        )
    if not np.isfinite(maxima).all() or np.any((maxima < 0) | (maxima > 1)):
        raise ValueError("invalid Tanimoto similarities")
    return maxima


def similarity_summary(records: pd.DataFrame) -> pd.DataFrame:
    """Summarise exact calibration-pool-to-held-out nearest-neighbour similarity."""

    rows: list[dict[str, Any]] = []
    for (dataset, design, fold), frame in records.groupby(
        ["dataset", "holdout_design", "fold"], sort=True
    ):
        values = frame.maximum_tanimoto_to_calibration_pool.to_numpy(dtype=float)
        row: dict[str, Any] = {
            "dataset": dataset,
            "holdout_design": design,
            "fold": int(fold),
            "held_out_ligands": int(len(values)),
            "calibration_pool_ligands": int(frame.calibration_pool_ligands.iloc[0]),
            "exact_pairwise_comparisons": int(
                len(values) * int(frame.calibration_pool_ligands.iloc[0])
            ),
            "mean": float(values.mean()),
            "median": float(np.median(values)),
            "q025": float(np.quantile(values, 0.025)),
            "q05": float(np.quantile(values, 0.05)),
            "q25": float(np.quantile(values, 0.25)),
            "q75": float(np.quantile(values, 0.75)),
            "q975": float(np.quantile(values, 0.975)),
            "q95": float(np.quantile(values, 0.95)),
            "minimum": float(values.min()),
            "maximum": float(values.max()),
            "fraction_tanimoto_equal_one": float(
                np.mean(np.isclose(values, 1.0))
            ),
        }
        for threshold in SIMILARITY_THRESHOLDS:
            row[f"fraction_at_least_{threshold:.1f}"] = float(
                np.mean(values >= threshold)
            )
        rows.append(row)
    return pd.DataFrame.from_records(rows)


def analyze_dataset(
    dataset: str,
    matrix: np.ndarray,
    groups: np.ndarray,
    fingerprints: list[Any],
    source_indices: np.ndarray,
    *,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    records: list[dict[str, Any]] = []
    similarity_records: list[dict[str, Any]] = []
    fold_records: list[dict[str, Any]] = []
    if not (len(matrix) == len(groups) == len(fingerprints) == len(source_indices)):
        raise ValueError("matrix, groups, fingerprints and source indices must align")
    splitter = GroupKFold(n_splits=FOLDS)
    for fold, (train_index, held_out_index) in enumerate(
        splitter.split(matrix, groups=groups), start=1
    ):
        train_groups = set(groups[train_index].tolist())
        held_out_groups = set(groups[held_out_index].tolist())
        overlap = len(train_groups & held_out_groups)
        if overlap:
            raise RuntimeError("chemical-group leakage in GroupKFold")
        control_split_rng = np.random.default_rng(
            seed + 1_000_000 + 100_000 * fold
        )
        control_order = control_split_rng.permutation(len(matrix))
        control_held_index = control_order[: len(held_out_index)]
        control_pool_index = control_order[len(held_out_index) :]
        split_designs = {
            "chemical_group": (train_index, held_out_index),
            "matched_random_row": (control_pool_index, control_held_index),
        }
        for design, (pool_index, evaluation_index) in split_designs.items():
            pool_groups = set(groups[pool_index].tolist())
            evaluation_groups = set(groups[evaluation_index].tolist())
            design_overlap = len(pool_groups & evaluation_groups)
            maximum_similarities = exact_maximum_tanimoto_to_pool(
                [fingerprints[int(index)] for index in evaluation_index],
                [fingerprints[int(index)] for index in pool_index],
            )
            for local_index, similarity in zip(
                evaluation_index, maximum_similarities, strict=True
            ):
                similarity_records.append(
                    {
                        "dataset": dataset,
                        "holdout_design": design,
                        "fold": int(fold),
                        "analysis_row_index": int(local_index),
                        "source_row_index": int(source_indices[int(local_index)]),
                        "chemical_group": str(groups[int(local_index)]),
                        "calibration_pool_ligands": int(len(pool_index)),
                        "maximum_tanimoto_to_calibration_pool": float(similarity),
                    }
                )
            fold_records.append(
                {
                    "dataset": dataset,
                    "holdout_design": design,
                    "fold": int(fold),
                    "calibration_pool_ligands": int(len(pool_index)),
                    "held_out_ligands": int(len(evaluation_index)),
                    "calibration_chemical_groups": int(len(pool_groups)),
                    "held_out_chemical_groups": int(len(evaluation_groups)),
                    "chemical_group_overlap": int(design_overlap),
                    "calibration_pool_missing_cells": int(
                        np.isnan(matrix[pool_index]).sum()
                    ),
                    "held_out_missing_cells": int(
                        np.isnan(matrix[evaluation_index]).sum()
                    ),
                    "exact_fingerprint_comparisons": int(
                        len(pool_index) * len(evaluation_index)
                    ),
                    "maximum_tanimoto_median": float(
                        np.median(maximum_similarities)
                    ),
                    "maximum_tanimoto_q025": float(
                        np.quantile(maximum_similarities, 0.025)
                    ),
                    "maximum_tanimoto_q975": float(
                        np.quantile(maximum_similarities, 0.975)
                    ),
                }
            )
        train_matrix, held_out_matrix, train_means = fill_from_calibration_pool(
            matrix[train_index], matrix[held_out_index]
        )
        held_out_edges = upper_triangle(target_correlation(held_out_matrix))
        control_pool, control_held, control_means = fill_from_calibration_pool(
            matrix[control_pool_index], matrix[control_held_index]
        )
        control_held_edges = upper_triangle(target_correlation(control_held))
        train_missing_cells = int(np.isnan(matrix[train_index]).sum())
        held_out_missing_cells = int(np.isnan(matrix[held_out_index]).sum())
        control_pool_missing_cells = int(
            np.isnan(matrix[control_pool_index]).sum()
        )
        control_held_missing_cells = int(
            np.isnan(matrix[control_held_index]).sum()
        )
        for size in CALIBRATION_SIZES:
            if size >= len(train_index):
                raise ValueError("calibration size exceeds training fold")
            rng = np.random.default_rng(seed + 10_000 * fold + size)
            control_sample_rng = np.random.default_rng(
                seed + 2_000_000 + 10_000 * fold + size
            )
            for repetition in range(REPETITIONS_PER_FOLD):
                selected_local = rng.choice(
                    len(train_index), size=size, replace=False
                )
                sample_edges = upper_triangle(
                    target_correlation(train_matrix[selected_local])
                )
                records.append(
                    {
                        "dataset": dataset,
                        "holdout_design": "chemical_group",
                        "fold": int(fold),
                        "calibration_ligands": int(size),
                        "repetition": int(repetition),
                        "calibration_pool_ligands": int(len(train_index)),
                        "held_out_ligands": int(len(held_out_index)),
                        "calibration_chemical_groups": int(len(train_groups)),
                        "held_out_chemical_groups": int(len(held_out_groups)),
                        "chemical_group_overlap": int(overlap),
                        "fold_local_imputation": True,
                        "calibration_pool_imputed_cells": train_missing_cells,
                        "held_out_imputed_cells": held_out_missing_cells,
                        "maximum_absolute_imputation_mean": float(
                            np.max(np.abs(train_means))
                        ),
                        "geometry_spearman_to_holdout": spearman(
                            sample_edges, held_out_edges
                        ),
                    }
                )

                control_selected_local = control_sample_rng.choice(
                    len(control_pool_index), size=size, replace=False
                )
                control_sample = control_pool[control_selected_local]
                control_pool_groups = set(groups[control_pool_index].tolist())
                control_held_groups = set(groups[control_held_index].tolist())
                records.append(
                    {
                        "dataset": dataset,
                        "holdout_design": "matched_random_row",
                        "fold": int(fold),
                        "calibration_ligands": int(size),
                        "repetition": int(repetition),
                        "calibration_pool_ligands": int(len(control_pool_index)),
                        "held_out_ligands": int(len(control_held_index)),
                        "calibration_chemical_groups": int(len(control_pool_groups)),
                        "held_out_chemical_groups": int(len(control_held_groups)),
                        "chemical_group_overlap": int(
                            len(control_pool_groups & control_held_groups)
                        ),
                        "fold_local_imputation": True,
                        "calibration_pool_imputed_cells": control_pool_missing_cells,
                        "held_out_imputed_cells": control_held_missing_cells,
                        "maximum_absolute_imputation_mean": float(
                            np.max(np.abs(control_means))
                        ),
                        "geometry_spearman_to_holdout": spearman(
                            upper_triangle(target_correlation(control_sample)),
                            control_held_edges,
                        ),
                    }
                )
    return (
        pd.DataFrame.from_records(records),
        pd.DataFrame.from_records(similarity_records),
        pd.DataFrame.from_records(fold_records),
    )


def summarize(records: pd.DataFrame) -> pd.DataFrame:
    return records.groupby(
        ["dataset", "holdout_design", "calibration_ligands"], as_index=False
    ).agg(
        repetitions=("repetition", "size"),
        geometry_spearman_mean=("geometry_spearman_to_holdout", "mean"),
        geometry_spearman_sd=("geometry_spearman_to_holdout", "std"),
        geometry_spearman_q025=(
            "geometry_spearman_to_holdout",
            lambda values: float(np.quantile(values, 0.025)),
        ),
        geometry_spearman_q975=(
            "geometry_spearman_to_holdout",
            lambda values: float(np.quantile(values, 0.975)),
        ),
        geometry_spearman_minimum=("geometry_spearman_to_holdout", "min"),
        geometry_spearman_maximum=("geometry_spearman_to_holdout", "max"),
        maximum_group_overlap=("chemical_group_overlap", "max"),
    )


def summarize_by_fold(records: pd.DataFrame) -> pd.DataFrame:
    return records.groupby(
        ["dataset", "holdout_design", "fold", "calibration_ligands"],
        as_index=False,
    ).agg(
        repetitions=("repetition", "size"),
        geometry_spearman_mean=("geometry_spearman_to_holdout", "mean"),
        geometry_spearman_sd=("geometry_spearman_to_holdout", "std"),
        geometry_spearman_minimum=("geometry_spearman_to_holdout", "min"),
        geometry_spearman_maximum=("geometry_spearman_to_holdout", "max"),
    )


def paired_design_summary(records: pd.DataFrame) -> pd.DataFrame:
    index = ["dataset", "fold", "calibration_ligands", "repetition"]
    pivot = records.pivot(
        index=index,
        columns="holdout_design",
        values="geometry_spearman_to_holdout",
    ).reset_index()
    pivot["chemical_group_minus_matched_random_row"] = (
        pivot.chemical_group - pivot.matched_random_row
    )
    return pivot.groupby(
        ["dataset", "calibration_ligands"], as_index=False
    ).agg(
        paired_repetitions=("repetition", "size"),
        mean_difference=("chemical_group_minus_matched_random_row", "mean"),
        median_difference=("chemical_group_minus_matched_random_row", "median"),
        q025_difference=(
            "chemical_group_minus_matched_random_row",
            lambda values: float(np.quantile(values, 0.025)),
        ),
        q975_difference=(
            "chemical_group_minus_matched_random_row",
            lambda values: float(np.quantile(values, 0.975)),
        ),
    )


def build(
    docking44_path: Path,
    dockstring_path: Path,
) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
    (
        d44_matrix,
        d44_groups,
        d44_fingerprints,
        d44_source_indices,
        d44_support,
    ) = load_docking44(docking44_path)
    (
        ds_matrix,
        ds_groups,
        ds_fingerprints,
        ds_source_indices,
        ds_support,
    ) = load_dockstring(dockstring_path)
    d44_records, d44_similarity, d44_folds = analyze_dataset(
        "Docking-44",
        d44_matrix,
        d44_groups,
        d44_fingerprints,
        d44_source_indices,
        seed=BASE_SEED,
    )
    ds_records, ds_similarity, ds_folds = analyze_dataset(
        "DOCKSTRING-58",
        ds_matrix,
        ds_groups,
        ds_fingerprints,
        ds_source_indices,
        seed=BASE_SEED + 1,
    )
    records = pd.concat([d44_records, ds_records], ignore_index=True)
    similarity_records = pd.concat(
        [d44_similarity, ds_similarity], ignore_index=True
    )
    fold_diagnostics = pd.concat([d44_folds, ds_folds], ignore_index=True)
    nearest_neighbour_summary = similarity_summary(similarity_records)
    summary_table = summarize(records)
    by_fold_summary = summarize_by_fold(records)
    group_summary = summary_table.loc[
        summary_table.holdout_design.eq("chemical_group")
    ].reset_index(drop=True)
    random_summary = summary_table.loc[
        summary_table.holdout_design.eq("matched_random_row")
    ].reset_index(drop=True)
    paired_summary = paired_design_summary(records)
    if int(group_summary.maximum_group_overlap.max()) != 0:
        raise RuntimeError("group-disjoint recovery contains leakage")
    summary = {
        "schema_version": "1.0.0",
        "analysis_status": "strict_public_chemical_group_holdout_sensitivity",
        "producer": "analysis/public_scaffold_holdout_recovery.py",
        "configuration": {
            "folds": FOLDS,
            "calibration_sizes": list(CALIBRATION_SIZES),
            "repetitions_per_fold": REPETITIONS_PER_FOLD,
            "base_seed": BASE_SEED,
            "dockstring_support_size": SUPPORT_SIZE,
            "dockstring_support_seed": SUPPORT_SEED,
            "rdkit_version": rdBase.rdkitVersion,
            "morgan_fingerprint": {
                "implementation": "RDKit Morgan bit fingerprint",
                "radius": MORGAN_RADIUS,
                "bits": MORGAN_BITS,
                "include_chirality": False,
            },
            "nearest_neighbour_similarity": (
                "for each chemical-group-held-out ligand, exact maximum Morgan "
                "Tanimoto similarity to every ligand in the four-fold calibration pool"
            ),
            "tanimoto_equal_one_boundary": (
                "Tanimoto equal to one denotes identity of the declared finite Morgan "
                "bit vectors, not exact molecular identity; fingerprint collisions are possible"
            ),
            "fingerprint_cluster_sensitivity_boundary": (
                "No all-pairs fingerprint clustering was run: for n=15000, one "
                "dense Butina distance vector requires 112,492,500 pair distances, "
                "and repeating it at multiple cutoffs is not a lightweight strict-public "
                "sensitivity. Exact foldwise nearest-neighbour distributions and "
                "fractions above declared Tanimoto thresholds 0.4, 0.5, 0.7, 0.8 and 0.9 "
                "are reported instead; they diagnose analogue proximity but do not "
                "guarantee fingerprint-cluster-disjoint folds."
            ),
        },
        "datasets": {"Docking-44": d44_support, "DOCKSTRING-58": ds_support},
        "key_results": group_summary.to_dict(orient="records"),
        "matched_random_row_results": random_summary.to_dict(orient="records"),
        "paired_design_contrasts": paired_summary.to_dict(orient="records"),
        "chemical_group_fold_diagnostics": fold_diagnostics.to_dict(orient="records"),
        "nearest_neighbour_similarity_results": nearest_neighbour_summary.to_dict(
            orient="records"
        ),
        "recovery_by_fold": by_fold_summary.to_dict(orient="records"),
        "claim_boundary": (
            "The sensitivity tests recovery on chemical groups held out within the "
            "same source collection and compares it with a matched random-row "
            "holdout. Repeated-split differences describe the incremental group "
            "constraint; they do not establish equivalence or transport to a shifted "
            "deployment library, biological optimality, docking accuracy, or recovery "
            "of individual ligand scores."
        ),
        "inputs": {
            "docking44": {
                "path": str(docking44_path.resolve().relative_to(PACKAGE)),
                "sha256": sha256_file(docking44_path),
            },
            "dockstring": {
                "path": str(dockstring_path.resolve().relative_to(PACKAGE)),
                "sha256": sha256_file(dockstring_path),
            },
        },
    }
    return summary, {
        "scaffold_holdout_metrics.csv": records,
        "scaffold_holdout_summary.csv": group_summary,
        "matched_random_row_holdout_summary.csv": random_summary,
        "chemical_group_minus_random_summary.csv": paired_summary,
        "chemical_group_fold_diagnostics.csv": fold_diagnostics,
        "heldout_maximum_morgan_tanimoto.csv": similarity_records,
        "heldout_maximum_morgan_tanimoto_summary.csv": nearest_neighbour_summary,
        "scaffold_holdout_by_fold_summary.csv": by_fold_summary,
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
        "\n".join(
            [
                "# Strict-public chemical-group-held-out pilot recovery",
                "",
                "Calibration maps are estimated on four chemically grouped folds and",
                "compared with the residual Vina map on the fifth. Cyclic molecules use",
                "non-isomeric Bemis--Murcko groups; acyclic molecules use canonical",
                "non-isomeric full-connectivity identity, so duplicate acyclic structures",
                "cannot cross folds.",
                "Matched random-row holdouts use the same calibration and evaluation",
                "sizes. Docking-44 imputation is fold-local. All intervals are repeated-",
                "sample sensitivity ranges conditional on the fixed panel and source.",
                "For every held-out ligand, the exact maximum radius-2, 2048-bit Morgan",
                "Tanimoto similarity to the full calibration pool is reported. This",
                "diagnoses remaining analogue proximity but is not a fingerprint-cluster-",
                "disjoint split.",
                "A Tanimoto value of one means Morgan-bit-vector identity only and is",
                "not labelled as exact molecular identity.",
                "",
                "```bash",
                ".venv/bin/python analysis/public_scaffold_holdout_recovery.py",
                ".venv/bin/python -m pytest -q analysis/test_public_scaffold_holdout_recovery.py",
                "```",
                "",
            ]
        ),
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
    parser.add_argument("--docking44", type=Path, default=DEFAULT_DOCKING44)
    parser.add_argument("--dockstring", type=Path, default=DEFAULT_DOCKSTRING)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary, tables = build(args.docking44, args.dockstring)
    write_artifact(args.output_dir, summary, tables)
    print(args.output_dir / "summary.json")


if __name__ == "__main__":
    main()
