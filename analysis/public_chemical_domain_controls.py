#!/usr/bin/env python3
"""Strict-public chemical-domain controls for Vina target-correlation maps.

This producer recomputes, directly from the two frozen row-level score tables:

1. the low-versus-high molecular-weight quartile contrast on six independently
   sampled DOCKSTRING supports;
2. target-map agreement between whole-chemical-group-disjoint halves whose
   molecular-weight distributions are matched by group-median-MW strata; and
3. whole-group-disjoint and row-random split-half reproducibility separately
   inside the low- and high-MW bands of both large panels.

No local analysis module or evidence ledger is imported.  The controls distinguish
MW-domain dependence from generic chemical disjointness descriptively; they do not
identify molecular weight as a causal mechanism.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rdkit import Chem, rdBase
from rdkit.Chem import Descriptors
from rdkit.Chem.Scaffolds import MurckoScaffold


PACKAGE = Path(__file__).resolve().parents[1]
FROZEN = PACKAGE / "data" / "frozen"
DEFAULT_DOCKING44 = FROZEN / "df_final_v4.csv.gz"
DEFAULT_DOCKSTRING = FROZEN / "dockstring-dataset.tsv.gz"
DEFAULT_OUTPUT = PACKAGE / "results" / "public_chemical_domain_controls"
DEFAULT_SUPPORT_SIZE = 15_000
DEFAULT_SUPPORT_SEEDS = (71, 72, 73, 74, 75, 202_608_09)
DEFAULT_REPETITIONS = 200
DEFAULT_SEED = 202_608_12
EXTREME_FRACTION = 0.25
DOCK44 = [
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
    if matrix.ndim != 2 or min(matrix.shape) < 2 or not np.isfinite(matrix).all():
        raise ValueError("row centering requires a finite two-dimensional matrix")
    return matrix - matrix.mean(axis=1, keepdims=True)


def target_correlation(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] < 3 or matrix.shape[1] < 3:
        raise ValueError("target correlation requires >=3 rows and >=3 targets")
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    covariance = centered.T @ centered
    variance = np.diag(covariance)
    if np.any(variance <= 1e-14) or not np.isfinite(variance).all():
        raise ValueError("target correlation contains a constant target")
    scale = np.sqrt(variance)
    correlation = covariance / np.outer(scale, scale)
    correlation = np.clip((correlation + correlation.T) / 2.0, -1.0, 1.0)
    np.fill_diagonal(correlation, 1.0)
    return correlation


def upper_triangle(matrix: np.ndarray) -> np.ndarray:
    return np.asarray(matrix, dtype=np.float64)[
        np.triu_indices(len(matrix), k=1)
    ]


def spearman(first: np.ndarray, second: np.ndarray) -> float:
    first_rank = pd.Series(np.asarray(first, dtype=np.float64)).rank(
        method="average"
    ).to_numpy()
    second_rank = pd.Series(np.asarray(second, dtype=np.float64)).rank(
        method="average"
    ).to_numpy()
    value = float(np.corrcoef(first_rank, second_rank)[0, 1])
    if not np.isfinite(value):
        raise ValueError("Spearman agreement is not finite")
    return value


def correlation_pr(correlation: np.ndarray) -> float:
    correlation = np.asarray(correlation, dtype=np.float64)
    return float(np.trace(correlation) ** 2 / np.square(correlation).sum())


def strongest_positive_mask(values: np.ndarray, fraction: float = 0.10) -> np.ndarray:
    count = max(1, int(np.ceil(len(values) * fraction)))
    mask = np.zeros(len(values), dtype=bool)
    mask[np.argsort(values, kind="mergesort")[-count:]] = True
    return mask


def geometry_comparison(first: np.ndarray, second: np.ndarray) -> dict[str, float]:
    first_values = upper_triangle(first)
    second_values = upper_triangle(second)
    first_top = strongest_positive_mask(first_values)
    second_top = strongest_positive_mask(second_values)
    union = int(np.sum(first_top | second_top))
    rho = spearman(first_values, second_values)
    return {
        "geometry_spearman": rho,
        "geometry_dissimilarity": 1.0 - rho,
        "sign_flip_fraction": float(
            np.mean(np.sign(first_values) != np.sign(second_values))
        ),
        "top_positive_10pct_pair_jaccard": float(
            np.sum(first_top & second_top) / union
        ),
        "first_pr": correlation_pr(first),
        "second_pr": correlation_pr(second),
        "first_mean_absolute_correlation": float(np.mean(np.abs(first_values))),
        "second_mean_absolute_correlation": float(np.mean(np.abs(second_values))),
    }


def extreme_masks(
    molecular_weight: np.ndarray,
    fraction: float = EXTREME_FRACTION,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    molecular_weight = np.asarray(molecular_weight, dtype=np.float64)
    if molecular_weight.ndim != 1 or not np.isfinite(molecular_weight).all():
        raise ValueError("molecular weight must be one finite vector")
    low_threshold, high_threshold = np.quantile(
        molecular_weight, [fraction, 1.0 - fraction]
    )
    low = molecular_weight <= low_threshold
    high = molecular_weight >= high_threshold
    if low_threshold >= high_threshold or np.any(low & high):
        raise ValueError("MW extremes are not disjoint")
    return low, high, float(low_threshold), float(high_threshold)


def ks_statistic(first: np.ndarray, second: np.ndarray) -> float:
    first = np.sort(np.asarray(first, dtype=np.float64))
    second = np.sort(np.asarray(second, dtype=np.float64))
    points = np.unique(np.concatenate([first, second]))
    first_cdf = np.searchsorted(first, points, side="right") / len(first)
    second_cdf = np.searchsorted(second, points, side="right") / len(second)
    return float(np.max(np.abs(first_cdf - second_cdf)))


def group_codes(groups: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = pd.Series(np.asarray(groups, dtype=object).astype(str), dtype="string")
    codes, unique = pd.factorize(values, sort=True)
    if np.any(codes < 0):
        raise ValueError("chemical groups contain missing values")
    return codes.astype(np.int64), np.asarray(unique.astype(str), dtype=object)


def rank_strata(values: np.ndarray, strata: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if len(values) < 2 * strata or not np.isfinite(values).all():
        raise ValueError("too few groups for MW rank strata")
    order = np.lexsort((np.arange(len(values)), values))
    identifiers = np.empty(len(values), dtype=np.int16)
    identifiers[order] = np.minimum(
        strata - 1, np.arange(len(values)) * strata // len(values)
    )
    return identifiers


def balanced_group_split(
    group_sizes: np.ndarray,
    group_median_mw: np.ndarray,
    rng: np.random.Generator,
    *,
    strata: int = 10,
) -> np.ndarray:
    group_sizes = np.asarray(group_sizes, dtype=np.int64)
    group_median_mw = np.asarray(group_median_mw, dtype=np.float64)
    identifiers = rank_strata(group_median_mw, strata)
    assignment = np.full(len(group_sizes), -1, dtype=np.int8)
    for stratum in range(strata):
        members = rng.permutation(np.flatnonzero(identifiers == stratum))
        totals = [0, 0]
        tie_side = int(rng.integers(0, 2))
        for group in members:
            if totals[0] < totals[1]:
                side = 0
            elif totals[1] < totals[0]:
                side = 1
            else:
                side = tie_side
                tie_side = 1 - tie_side
            assignment[group] = side
            totals[side] += int(group_sizes[group])
    if np.any(assignment < 0) or set(assignment.tolist()) != {0, 1}:
        raise RuntimeError("group split did not create two complete halves")
    return assignment


def molecular_weights(smiles: np.ndarray) -> np.ndarray:
    values = np.empty(len(smiles), dtype=np.float64)
    for index, text in enumerate(smiles):
        molecule = Chem.MolFromSmiles(str(text))
        if molecule is None:
            raise ValueError(f"invalid SMILES at row offset {index}")
        values[index] = Descriptors.MolWt(molecule)
    return values


def murcko_groups(smiles: np.ndarray, source_indices: np.ndarray) -> np.ndarray:
    groups: list[str] = []
    for text, source_index in zip(smiles, source_indices, strict=True):
        molecule = Chem.MolFromSmiles(str(text))
        if molecule is None:
            groups.append(f"INVALID_SINGLETON:{int(source_index)}")
            continue
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=molecule)
        groups.append(
            scaffold if scaffold else f"ACYCLIC_SINGLETON:{int(source_index)}"
        )
    return np.asarray(groups, dtype=object)


def surfaces(matrix: np.ndarray) -> dict[str, np.ndarray]:
    return {"raw": matrix, "row_centered_residual": row_center(matrix)}


def low_high_contrast(
    dataset: str,
    matrix: np.ndarray,
    molecular_weight: np.ndarray,
    *,
    support_seed: int | None,
) -> list[dict[str, Any]]:
    low, high, low_threshold, high_threshold = extreme_masks(molecular_weight)
    records: list[dict[str, Any]] = []
    for transformation, surface in surfaces(matrix).items():
        records.append(
            {
                "dataset": dataset,
                "support_seed": support_seed,
                "support_ligands": int(len(matrix)),
                "transformation": transformation,
                "low_threshold_inclusive": low_threshold,
                "high_threshold_inclusive": high_threshold,
                "low_ligands": int(low.sum()),
                "high_ligands": int(high.sum()),
                "low_median_molecular_weight": float(
                    np.median(molecular_weight[low])
                ),
                "high_median_molecular_weight": float(
                    np.median(molecular_weight[high])
                ),
                **geometry_comparison(
                    target_correlation(surface[low]),
                    target_correlation(surface[high]),
                ),
            }
        )
    return records


def group_summary(groups: np.ndarray, molecular_weight: np.ndarray) -> tuple[
    np.ndarray, np.ndarray, np.ndarray
]:
    codes, unique = group_codes(groups)
    frame = pd.DataFrame(
        {"code": codes, "molecular_weight": molecular_weight}
    )
    grouped = frame.groupby("code", sort=True).molecular_weight.agg(["size", "median"])
    if len(grouped) != len(unique):
        raise RuntimeError("chemical-group summary is misaligned")
    return (
        codes,
        grouped["size"].to_numpy(dtype=np.int64),
        grouped["median"].to_numpy(dtype=np.float64),
    )


def append_split_comparisons(
    records: list[dict[str, Any]],
    *,
    dataset: str,
    matrix_surfaces: dict[str, np.ndarray],
    molecular_weight: np.ndarray,
    codes: np.ndarray,
    first_index: np.ndarray,
    second_index: np.ndarray,
    repetition: int,
    control_type: str,
    mw_band: str,
) -> None:
    first_groups = set(codes[first_index].tolist())
    second_groups = set(codes[second_index].tolist())
    overlap = len(first_groups & second_groups)
    if "group_disjoint" in control_type and overlap:
        raise RuntimeError("chemical-group leakage in group-disjoint split")
    common = {
        "dataset": dataset,
        "mw_band": mw_band,
        "control_type": control_type,
        "repetition": int(repetition),
        "first_ligands": int(len(first_index)),
        "second_ligands": int(len(second_index)),
        "first_chemical_groups": int(len(first_groups)),
        "second_chemical_groups": int(len(second_groups)),
        "chemical_group_overlap": int(overlap),
        "first_median_molecular_weight": float(
            np.median(molecular_weight[first_index])
        ),
        "second_median_molecular_weight": float(
            np.median(molecular_weight[second_index])
        ),
        "absolute_median_mw_difference": float(
            abs(
                np.median(molecular_weight[first_index])
                - np.median(molecular_weight[second_index])
            )
        ),
        "mw_ks_statistic": ks_statistic(
            molecular_weight[first_index], molecular_weight[second_index]
        ),
    }
    for transformation, surface in matrix_surfaces.items():
        records.append(
            {
                **common,
                "transformation": transformation,
                **geometry_comparison(
                    target_correlation(surface[first_index]),
                    target_correlation(surface[second_index]),
                ),
            }
        )


def chemical_domain_controls(
    dataset: str,
    matrix: np.ndarray,
    molecular_weight: np.ndarray,
    groups: np.ndarray,
    *,
    repetitions: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    codes, group_sizes, group_medians = group_summary(groups, molecular_weight)
    matrix_surfaces = surfaces(matrix)
    rng = np.random.default_rng(seed)
    global_records: list[dict[str, Any]] = []
    for repetition in range(repetitions):
        assignment = balanced_group_split(group_sizes, group_medians, rng)
        first = np.flatnonzero(assignment[codes] == 0)
        second = np.flatnonzero(assignment[codes] == 1)
        append_split_comparisons(
            global_records,
            dataset=dataset,
            matrix_surfaces=matrix_surfaces,
            molecular_weight=molecular_weight,
            codes=codes,
            first_index=first,
            second_index=second,
            repetition=repetition,
            control_type="mw_matched_chemical_group_disjoint",
            mw_band="full_support",
        )

    within_records: list[dict[str, Any]] = []
    low, high, _, _ = extreme_masks(molecular_weight)
    for band_offset, (band, mask) in enumerate((('low_mw', low), ('high_mw', high))):
        band_indices = np.flatnonzero(mask)
        half_size = len(band_indices) // 2
        row_rng = np.random.default_rng(seed + 100 + band_offset)
        for repetition in range(repetitions):
            order = row_rng.permutation(band_indices)
            append_split_comparisons(
                within_records,
                dataset=dataset,
                matrix_surfaces=matrix_surfaces,
                molecular_weight=molecular_weight,
                codes=codes,
                first_index=order[:half_size],
                second_index=order[half_size : 2 * half_size],
                repetition=repetition,
                control_type="row_random_disjoint",
                mw_band=band,
            )

        restricted_groups = np.asarray(groups, dtype=object)[mask]
        restricted_mw = molecular_weight[mask]
        restricted_codes, sizes, medians = group_summary(
            restricted_groups, restricted_mw
        )
        group_rng = np.random.default_rng(seed + 200 + band_offset)
        for repetition in range(repetitions):
            assignment = balanced_group_split(sizes, medians, group_rng)
            first = band_indices[assignment[restricted_codes] == 0]
            second = band_indices[assignment[restricted_codes] == 1]
            append_split_comparisons(
                within_records,
                dataset=dataset,
                matrix_surfaces=matrix_surfaces,
                molecular_weight=molecular_weight,
                codes=codes,
                first_index=first,
                second_index=second,
                repetition=repetition,
                control_type="mw_matched_chemical_group_disjoint",
                mw_band=band,
            )
    return (
        pd.DataFrame.from_records(global_records),
        pd.DataFrame.from_records(within_records),
    )


def summarize_controls(
    frame: pd.DataFrame,
    identifiers: list[str],
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for keys, group in frame.groupby(identifiers, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        record: dict[str, Any] = dict(zip(identifiers, keys, strict=True))
        record["repetitions"] = int(len(group))
        for metric in (
            "geometry_spearman",
            "sign_flip_fraction",
            "top_positive_10pct_pair_jaccard",
            "absolute_median_mw_difference",
            "mw_ks_statistic",
            "first_ligands",
            "second_ligands",
            "chemical_group_overlap",
        ):
            values = group[metric].to_numpy(dtype=np.float64)
            record[f"{metric}_mean"] = float(values.mean())
            record[f"{metric}_median"] = float(np.median(values))
            record[f"{metric}_q025"] = float(np.quantile(values, 0.025))
            record[f"{metric}_q975"] = float(np.quantile(values, 0.975))
        records.append(record)
    return pd.DataFrame.from_records(records)


def causal_boundary_table(
    observed: pd.DataFrame,
    global_summary: pd.DataFrame,
    within_summary: pd.DataFrame,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for row in observed.itertuples(index=False):
        global_row = global_summary.loc[
            global_summary.dataset.eq(row.dataset)
            & global_summary.transformation.eq(row.transformation)
        ].iloc[0]
        within = within_summary.loc[
            within_summary.dataset.eq(row.dataset)
            & within_summary.transformation.eq(row.transformation)
            & within_summary.control_type.eq(
                "mw_matched_chemical_group_disjoint"
            )
        ].set_index("mw_band")
        low = within.loc["low_mw"]
        high = within.loc["high_mw"]
        records.append(
            {
                "dataset": row.dataset,
                "transformation": row.transformation,
                "observed_low_high_geometry_spearman": row.geometry_spearman,
                "global_mw_matched_group_disjoint_median": global_row.geometry_spearman_median,
                "global_mw_matched_group_disjoint_q025": global_row.geometry_spearman_q025,
                "within_low_mw_group_disjoint_median": low.geometry_spearman_median,
                "within_low_mw_group_disjoint_q025": low.geometry_spearman_q025,
                "within_high_mw_group_disjoint_median": high.geometry_spearman_median,
                "within_high_mw_group_disjoint_q025": high.geometry_spearman_q025,
                "observed_below_global_group_disjoint_q025": bool(
                    row.geometry_spearman < global_row.geometry_spearman_q025
                ),
                "observed_below_both_within_band_group_disjoint_q025": bool(
                    row.geometry_spearman
                    < min(low.geometry_spearman_q025, high.geometry_spearman_q025)
                ),
                "interpretation": (
                    "descriptive separation from chemical-disjoint controls; not causal "
                    "attribution to molecular weight"
                ),
            }
        )
    return pd.DataFrame.from_records(records)


def load_docking44(path: Path) -> dict[str, Any]:
    frame = pd.read_csv(
        path,
        usecols=DOCK44
        + ["Cleaned SMILES", "Canonical SMILES", "Butina_clusters"],
    )
    numeric = frame[DOCK44].apply(pd.to_numeric, errors="coerce")
    clipped = numeric.clip(upper=0)
    matrix = clipped.fillna(clipped.mean()).to_numpy(dtype=np.float64)
    smiles = frame["Cleaned SMILES"].fillna(frame["Canonical SMILES"]).astype(str).to_numpy()
    groups = frame["Butina_clusters"].astype(str).to_numpy()
    if pd.isna(frame["Butina_clusters"]).any():
        raise ValueError("Docking-44 contains a missing Butina group")
    return {
        "dataset": "Docking-44",
        "matrix": matrix,
        "molecular_weight": molecular_weights(smiles),
        "groups": groups,
        "targets": DOCK44,
        "support_rule": "all 12,651 ligands; target-mean imputation after clipping",
        "missing_cells": int(numeric.isna().to_numpy().sum()),
        "positive_cells_clipped": int((numeric.to_numpy(dtype=float) > 0).sum()),
        "group_definition": "frozen source Butina_clusters",
    }


def load_dockstring_source(path: Path) -> dict[str, Any]:
    frame = pd.read_csv(path, sep="\t")
    targets = [column for column in frame if column not in {"inchikey", "smiles"}]
    numeric = frame[targets].apply(pd.to_numeric, errors="coerce")
    complete = ~numeric.isna().any(axis=1)
    matrix = np.minimum(
        numeric.loc[complete].to_numpy(dtype=np.float64), 0.0
    )
    smiles = frame.loc[complete, "smiles"].astype(str).reset_index(drop=True).to_numpy()
    source_indices = np.flatnonzero(complete.to_numpy())
    return {
        "matrix": matrix,
        "smiles": smiles,
        "source_indices": source_indices,
        "targets": targets,
        "source_rows": int(len(frame)),
        "complete_rows": int(complete.sum()),
        "missing_cells": int(numeric.isna().to_numpy().sum()),
        "positive_cells_clipped": int(
            (numeric.loc[complete].to_numpy(dtype=float) > 0).sum()
        ),
    }


def dockstring_support(
    source: dict[str, Any],
    *,
    size: int,
    seed: int,
    with_groups: bool,
) -> dict[str, Any]:
    if size > len(source["matrix"]):
        raise ValueError("DOCKSTRING support size exceeds complete rows")
    rng = np.random.default_rng(seed)
    selected = np.sort(rng.choice(len(source["matrix"]), size=size, replace=False))
    smiles = source["smiles"][selected]
    result = {
        "dataset": "DOCKSTRING-58",
        "matrix": source["matrix"][selected],
        "molecular_weight": molecular_weights(smiles),
        "targets": source["targets"],
        "support_seed": int(seed),
        "selected_complete_row_index_sha256": hashlib.sha256(
            np.asarray(selected, dtype="<i8").tobytes()
        ).hexdigest(),
    }
    if with_groups:
        result["groups"] = murcko_groups(
            smiles, source["source_indices"][selected]
        )
        result["group_definition"] = (
            "RDKit Bemis-Murcko scaffold; every acyclic molecule is a source-row singleton"
        )
    return result


def analyze(
    docking44: dict[str, Any],
    dockstring_source: dict[str, Any],
    *,
    support_size: int,
    support_seeds: tuple[int, ...],
    repetitions: int,
    seed: int,
) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
    seed_records: list[dict[str, Any]] = []
    primary_dockstring: dict[str, Any] | None = None
    support_metadata: list[dict[str, Any]] = []
    for support_seed in support_seeds:
        support = dockstring_support(
            dockstring_source,
            size=support_size,
            seed=support_seed,
            with_groups=support_seed == 71,
        )
        seed_records.extend(
            low_high_contrast(
                support["dataset"],
                support["matrix"],
                support["molecular_weight"],
                support_seed=support_seed,
            )
        )
        support_metadata.append(
            {
                "seed": int(support_seed),
                "selected_complete_row_index_sha256": support[
                    "selected_complete_row_index_sha256"
                ],
            }
        )
        if support_seed == 71:
            primary_dockstring = support
    if primary_dockstring is None:
        raise ValueError("support seeds must include primary seed 71")

    observed = pd.DataFrame.from_records(
        low_high_contrast(
            docking44["dataset"],
            docking44["matrix"],
            docking44["molecular_weight"],
            support_seed=None,
        )
        + [
            record
            for record in seed_records
            if record["support_seed"] == 71
        ]
    )
    global_frames: list[pd.DataFrame] = []
    within_frames: list[pd.DataFrame] = []
    for dataset_offset, bundle in enumerate((docking44, primary_dockstring)):
        global_control, within = chemical_domain_controls(
            bundle["dataset"],
            bundle["matrix"],
            bundle["molecular_weight"],
            bundle["groups"],
            repetitions=repetitions,
            seed=seed + 1000 * dataset_offset,
        )
        global_frames.append(global_control)
        within_frames.append(within)
    global_control = pd.concat(global_frames, ignore_index=True)
    within_control = pd.concat(within_frames, ignore_index=True)
    global_summary = summarize_controls(
        global_control,
        ["dataset", "control_type", "transformation"],
    )
    within_summary = summarize_controls(
        within_control,
        ["dataset", "mw_band", "control_type", "transformation"],
    )
    boundary = causal_boundary_table(observed, global_summary, within_summary)
    seed_frame = pd.DataFrame.from_records(seed_records)
    seed_summary_records: list[dict[str, Any]] = []
    for transformation, group in seed_frame.groupby("transformation", sort=True):
        seed_summary_records.append(
            {
                "dataset": "DOCKSTRING-58",
                "transformation": transformation,
                "support_seeds": int(len(group)),
                "geometry_spearman_minimum": float(group.geometry_spearman.min()),
                "geometry_spearman_median": float(group.geometry_spearman.median()),
                "geometry_spearman_maximum": float(group.geometry_spearman.max()),
                "sign_flip_fraction_minimum": float(group.sign_flip_fraction.min()),
                "sign_flip_fraction_maximum": float(group.sign_flip_fraction.max()),
            }
        )
    seed_summary = pd.DataFrame.from_records(seed_summary_records)

    summary = {
        "schema_version": "1.0.0",
        "analysis_status": "standalone_public_post_hoc_domain_sensitivity",
        "producer": "analysis/public_chemical_domain_controls.py",
        "configuration": {
            "dockstring_support_size": int(support_size),
            "dockstring_support_seeds": [int(value) for value in support_seeds],
            "primary_dockstring_support_seed": 71,
            "control_repetitions": int(repetitions),
            "control_seed": int(seed),
            "extreme_fraction_per_tail": EXTREME_FRACTION,
            "rdkit_version": rdBase.rdkitVersion,
        },
        "support_fingerprints": support_metadata,
        "datasets": {
            "Docking-44": {
                "ligands": int(len(docking44["matrix"])),
                "targets": int(docking44["matrix"].shape[1]),
                "chemical_groups": int(len(np.unique(docking44["groups"]))),
                "group_definition": docking44["group_definition"],
                "missing_cells_before_imputation": docking44["missing_cells"],
            },
            "DOCKSTRING-58": {
                "source_rows": dockstring_source["source_rows"],
                "complete_rows": dockstring_source["complete_rows"],
                "primary_support_ligands": int(len(primary_dockstring["matrix"])),
                "targets": int(primary_dockstring["matrix"].shape[1]),
                "chemical_groups_on_primary_support": int(
                    len(np.unique(primary_dockstring["groups"]))
                ),
                "group_definition": primary_dockstring["group_definition"],
            },
        },
        "control_estimands": {
            "mw_matched_group_disjoint": (
                "whole chemical groups assigned to two halves inside ten stable "
                "rank strata of group-median MW, greedily balancing ligand counts"
            ),
            "within_band_group_disjoint": (
                "the same whole-group split repeated separately after restricting "
                "to the low- or high-MW quartile"
            ),
            "within_band_row_random": (
                "equal-size nonoverlapping random row halves within one MW band; "
                "chemical groups may overlap"
            ),
        },
        "key_boundary_results": boundary.to_dict(orient="records"),
        "dockstring_seed_summary": seed_summary.to_dict(orient="records"),
        "claim_boundary": (
            "The controls show whether low-versus-high MW target-map disagreement is "
            "larger than disagreement caused by chemically disjoint but MW-matched "
            "supports and whether each restricted MW map is internally reproducible. "
            "Molecular weight was selected post hoc and is correlated with other ligand "
            "properties; these comparisons therefore do not establish MW as causal, "
            "validate docking poses, or establish biological target relationships."
        ),
    }
    tables = {
        "observed_low_high_mw_contrasts.csv": observed,
        "dockstring_support_seed_contrasts.csv": seed_frame,
        "dockstring_support_seed_summary.csv": seed_summary,
        "mw_matched_group_disjoint_controls.csv": global_control,
        "mw_matched_group_disjoint_summary.csv": global_summary,
        "within_band_reproducibility_controls.csv": within_control,
        "within_band_reproducibility_summary.csv": within_summary,
        "causal_boundary_summary.csv": boundary,
    }
    return summary, tables


def build_readme(summary: dict[str, Any]) -> str:
    lines = [
        "# Strict-public chemical-domain controls",
        "",
        "This artifact is recomputed directly from the frozen row-level Docking-44",
        "and DOCKSTRING score tables. It imports no local analysis builder and reads",
        "no evidence ledger.",
        "",
        "## Key comparison",
        "",
    ]
    for record in summary["key_boundary_results"]:
        lines.append(
            f"- {record['dataset']}, {record['transformation']}: observed low/high "
            f"rho={record['observed_low_high_geometry_spearman']:.3f}; global "
            f"MW-matched group-disjoint median="
            f"{record['global_mw_matched_group_disjoint_median']:.3f}; within-low/high "
            f"group-disjoint medians="
            f"{record['within_low_mw_group_disjoint_median']:.3f}/"
            f"{record['within_high_mw_group_disjoint_median']:.3f}."
        )
    lines.extend(
        [
            "",
            "Repeated-split ranges are composition-sensitivity ranges, not population",
            "confidence intervals. MW was selected post hoc; the result diagnoses",
            "chemical-domain dependence and does not identify a causal descriptor.",
            "",
            "## Reproduce",
            "",
            "```bash",
            ".venv/bin/python analysis/public_chemical_domain_controls.py \\",
            f"  --repetitions {summary['configuration']['control_repetitions']} \\",
            "  --output-dir results/public_chemical_domain_controls",
            ".venv/bin/python -m pytest -q analysis/test_public_chemical_domain_controls.py",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def write_artifact(
    output: Path,
    summary: dict[str, Any],
    tables: dict[str, pd.DataFrame],
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "summary.json", summary)
    for filename, frame in tables.items():
        frame.to_csv(output / filename, index=False, float_format="%.17g")
    (output / "README.md").write_text(build_readme(summary), encoding="utf-8")
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


def parse_seed_tuple(value: str) -> tuple[int, ...]:
    seeds = tuple(int(item) for item in value.split(",") if item)
    if not seeds or len(set(seeds)) != len(seeds):
        raise argparse.ArgumentTypeError("support seeds must be distinct integers")
    return seeds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docking44", type=Path, default=DEFAULT_DOCKING44)
    parser.add_argument("--dockstring", type=Path, default=DEFAULT_DOCKSTRING)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--support-size", type=int, default=DEFAULT_SUPPORT_SIZE)
    parser.add_argument(
        "--support-seeds",
        type=parse_seed_tuple,
        default=DEFAULT_SUPPORT_SEEDS,
    )
    parser.add_argument("--repetitions", type=int, default=DEFAULT_REPETITIONS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.repetitions < 2:
        raise ValueError("repetitions must be at least two")
    if tuple(args.support_seeds) != DEFAULT_SUPPORT_SEEDS:
        raise ValueError(
            "public production contract requires support seeds 71-75 and 20260809"
        )
    docking44 = load_docking44(args.docking44)
    dockstring_source = load_dockstring_source(args.dockstring)
    summary, tables = analyze(
        docking44,
        dockstring_source,
        support_size=args.support_size,
        support_seeds=tuple(args.support_seeds),
        repetitions=args.repetitions,
        seed=args.seed,
    )
    summary["inputs"] = {
        "docking44": {
            "path": str(args.docking44.resolve().relative_to(PACKAGE)),
            "sha256": sha256_file(args.docking44),
        },
        "dockstring": {
            "path": str(args.dockstring.resolve().relative_to(PACKAGE)),
            "sha256": sha256_file(args.dockstring),
        },
    }
    write_artifact(args.output_dir, summary, tables)
    print(args.output_dir / "summary.json")


if __name__ == "__main__":
    main()
