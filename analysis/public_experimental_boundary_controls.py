#!/usr/bin/env python3
"""Strict-public boundary controls for the DAVIS/PKIS2 target maps.

This producer asks four deliberately narrow questions using only redistributed
row-level inputs.

1. Is the exact-compound Vina--experiment map agreement unusual relative to
   equal-size DOCKSTRING supports, including coarse descriptor-matched and
   scaffold-distinct controls?
2. Does DAVIS--PKIS2 experimental-map agreement survive removal of the modest
   chemical overlap between the released panels?
3. Is cross-panel agreement visible under predeclared rank and binary-threshold
   transforms, rather than only on the native continuous assay scales?
4. Does a Vina-selected eight-target panel cover the two experimental maps
   better than the exact distribution over all C(21, 8) possible panels, and
   how far is it from each experimental-map oracle?

The target map throughout is the Pearson correlation matrix of row-centred
profiles unless an explicit rank/binary sensitivity is named.  Map agreement
is Spearman correlation over the 210 strict-upper-triangle edges.  These are
geometry diagnostics on observed targets, not evidence of biological networks,
binding truth, or target-retrieval performance.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.ML.Cluster import Butina
from scipy import stats
from sklearn.neighbors import NearestNeighbors


ANALYSIS_DIR = Path(__file__).resolve().parent
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

import experimental_map_reliability as reliability  # noqa: E402
from public_descriptor_domain_specificity import (  # noqa: E402
    DESCRIPTORS,
    molecular_descriptors,
)


PACKAGE = ANALYSIS_DIR.parent
DEFAULT_OUTPUT = PACKAGE / "results" / "public_experimental_boundary_controls"
DEFAULT_REPEATS = 1_000
DEFAULT_PANEL_BOOTSTRAPS = 500
DEFAULT_QAP = 10_000
DEFAULT_SEED = 202_608_31
K = 8
MW_BINS = 20
NEIGHBOURS = 256
MORGAN_RADIUS = 2
MORGAN_BITS = 2_048
BUTINA_SIMILARITY = 0.65
TARGETS = tuple(reliability.TARGETS)
TARGET_PAIRS = len(TARGETS) * (len(TARGETS) - 1) // 2

OUTPUT_FILES = (
    "README.md",
    "cross_panel_overlap.csv",
    "cross_panel_overlap_sensitivity.csv",
    "cross_panel_robustness.csv",
    "davis_censoring_edge_audit.csv",
    "exact_support_control_replicates.csv",
    "exact_support_control_summary.csv",
    "panel8_coverage.csv",
    "panel8_cluster_bootstrap.csv",
    "panel8_selections.csv",
    "summary.json",
)


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(chunk_size):
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
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def atomic_text(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    atomic_text(
        path,
        json.dumps(json_ready(payload), indent=2, sort_keys=True, allow_nan=False)
        + "\n",
    )


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, float_format="%.12g")
    temporary.replace(path)


def row_center(matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("row centring requires a finite matrix")
    return values - values.mean(axis=1, keepdims=True)


def target_map(matrix: np.ndarray) -> np.ndarray:
    return reliability.target_correlation(row_center(matrix))


def column_rank_target_map(matrix: np.ndarray) -> np.ndarray:
    """Rank ligands separately within each target, then remove row effects."""
    values = np.asarray(matrix, dtype=np.float64)
    ranks = np.column_stack(
        [stats.rankdata(values[:, column], method="average") for column in range(values.shape[1])]
    )
    return target_map(ranks)


def upper(matrix: np.ndarray) -> np.ndarray:
    return reliability.upper(np.asarray(matrix, dtype=np.float64))


def map_spearman(first: np.ndarray, second: np.ndarray) -> float:
    return reliability.map_spearman(first, second)


def murcko_labels(smiles: Sequence[str]) -> np.ndarray:
    labels: list[str] = []
    for index, value in enumerate(smiles):
        molecule = Chem.MolFromSmiles(str(value))
        if molecule is None:
            raise ValueError(f"invalid SMILES at row {index}")
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=molecule)
        # No panel here is acyclic, but the source-row singleton rule prevents
        # all acyclic structures from becoming one artificial mega-cluster.
        labels.append(scaffold if scaffold else f"ACYCLIC_ROW_{index}")
    return np.asarray(labels, dtype=object)


def fingerprints(smiles: Sequence[str]) -> list[Any]:
    generator = AllChem.GetMorganGenerator(radius=MORGAN_RADIUS, fpSize=MORGAN_BITS)
    output = []
    for index, value in enumerate(smiles):
        molecule = Chem.MolFromSmiles(str(value))
        if molecule is None:
            raise ValueError(f"invalid SMILES at row {index}")
        output.append(generator.GetFingerprint(molecule))
    return output


def combined_butina_labels(
    first_smiles: Sequence[str], second_smiles: Sequence[str]
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Cluster the combined released panels using a declared Morgan contract."""
    first_count = len(first_smiles)
    fps = fingerprints([*map(str, first_smiles), *map(str, second_smiles)])
    distances: list[float] = []
    for index in range(1, len(fps)):
        similarities = DataStructs.BulkTanimotoSimilarity(fps[index], fps[:index])
        distances.extend(1.0 - value for value in similarities)
    clusters = Butina.ClusterData(
        distances,
        len(fps),
        1.0 - BUTINA_SIMILARITY,
        isDistData=True,
        reordering=True,
    )
    labels = np.empty(len(fps), dtype=np.int64)
    for cluster_index, members in enumerate(clusters):
        labels[np.asarray(members, dtype=int)] = cluster_index
    first = labels[:first_count]
    second = labels[first_count:]
    first_set, second_set = set(first.tolist()), set(second.tolist())
    mixed = first_set & second_set
    metadata = {
        "clusters": int(len(clusters)),
        "mixed_panel_clusters": int(len(mixed)),
        "davis_rows_in_mixed_clusters": int(np.isin(first, list(mixed)).sum()),
        "pkis2_rows_in_mixed_clusters": int(np.isin(second, list(mixed)).sum()),
        "fingerprint": f"Morgan radius {MORGAN_RADIUS}, {MORGAN_BITS} bits",
        "butina_neighbor_similarity_cutoff": BUTINA_SIMILARITY,
        "cutoff_note": (
            "Butina membership is defined by center-neighbor connectivity at the "
            "declared cutoff; it does not guarantee that every within-cluster pair "
            "has similarity at least the cutoff"
        ),
        "algorithm": "RDKit Butina on the combined released panels",
    }
    return first, second, metadata


def cross_nearest_similarity(
    first_smiles: Sequence[str], second_smiles: Sequence[str]
) -> tuple[np.ndarray, np.ndarray]:
    first = fingerprints(first_smiles)
    second = fingerprints(second_smiles)
    first_max = np.asarray(
        [max(DataStructs.BulkTanimotoSimilarity(fp, second)) for fp in first], dtype=float
    )
    second_max = np.asarray(
        [max(DataStructs.BulkTanimotoSimilarity(fp, first)) for fp in second], dtype=float
    )
    return first_max, second_max


def qap_record(
    first: np.ndarray,
    second: np.ndarray,
    *,
    permutations: int,
    seed: int,
) -> dict[str, Any]:
    observed = map_spearman(first, second)
    rng = np.random.default_rng(seed)
    null = np.empty(permutations, dtype=np.float64)
    for repetition in range(permutations):
        order = rng.permutation(len(first))
        null[repetition] = map_spearman(first, second[np.ix_(order, order)])
    return {
        "edge_spearman": observed,
        "target_label_qap_p_two_sided": float(
            (1 + np.sum(np.abs(null) >= abs(observed))) / (permutations + 1)
        ),
        "qap_null_median": float(np.median(null)),
        "qap_permutations": int(permutations),
        "qap_seed": int(seed),
    }


def overlap_audit(
    davis: pd.DataFrame, pkis2: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, Any]]:
    davis_scaffold = murcko_labels(davis.smiles.astype(str))
    pkis2_scaffold = murcko_labels(pkis2.smiles.astype(str))
    shared_full = set(davis.standard_inchikey) & set(pkis2.standard_inchikey)
    shared_connectivity = set(davis.connectivity_block) & set(pkis2.connectivity_block)
    shared_scaffold = set(davis_scaffold) & set(pkis2_scaffold)
    davis_butina, pkis2_butina, butina_meta = combined_butina_labels(
        davis.smiles.astype(str), pkis2.smiles.astype(str)
    )
    mixed_butina = set(davis_butina.tolist()) & set(pkis2_butina.tolist())
    davis_nn, pkis2_nn = cross_nearest_similarity(
        davis.smiles.astype(str), pkis2.smiles.astype(str)
    )

    records = []
    for level, first_mask, second_mask, shared_units in (
        (
            "full_standard_inchikey",
            davis.standard_inchikey.isin(shared_full).to_numpy(),
            pkis2.standard_inchikey.isin(shared_full).to_numpy(),
            len(shared_full),
        ),
        (
            "connectivity_block",
            davis.connectivity_block.isin(shared_connectivity).to_numpy(),
            pkis2.connectivity_block.isin(shared_connectivity).to_numpy(),
            len(shared_connectivity),
        ),
        (
            "nonempty_bemis_murcko_scaffold",
            np.isin(davis_scaffold, list(shared_scaffold)),
            np.isin(pkis2_scaffold, list(shared_scaffold)),
            len(shared_scaffold),
        ),
        (
            "mixed_panel_butina_cluster",
            np.isin(davis_butina, list(mixed_butina)),
            np.isin(pkis2_butina, list(mixed_butina)),
            len(mixed_butina),
        ),
    ):
        records.append(
            {
                "overlap_level": level,
                "shared_units": int(shared_units),
                "davis_rows_in_shared_units": int(first_mask.sum()),
                "pkis2_rows_in_shared_units": int(second_mask.sum()),
            }
        )
    for panel, values in (("DAVIS", davis_nn), ("PKIS2", pkis2_nn)):
        records.append(
            {
                "overlap_level": f"cross_panel_morgan_nearest_{panel}",
                "shared_units": np.nan,
                "davis_rows_in_shared_units": int(np.sum(values >= BUTINA_SIMILARITY))
                if panel == "DAVIS"
                else np.nan,
                "pkis2_rows_in_shared_units": int(np.sum(values >= BUTINA_SIMILARITY))
                if panel == "PKIS2"
                else np.nan,
                "nearest_similarity_median": float(np.median(values)),
                "nearest_similarity_q025": float(np.quantile(values, 0.025)),
                "nearest_similarity_q975": float(np.quantile(values, 0.975)),
                "rows_with_nearest_similarity_ge_0_65": int(
                    np.sum(values >= BUTINA_SIMILARITY)
                ),
                "rows_with_nearest_similarity_ge_0_80": int(np.sum(values >= 0.80)),
            }
        )
    masks = {
        "none_DAVIS": np.zeros(len(davis), dtype=bool),
        "none_PKIS2": np.zeros(len(pkis2), dtype=bool),
        "full_standard_inchikey_DAVIS": davis.standard_inchikey.isin(shared_full).to_numpy(),
        "full_standard_inchikey_PKIS2": pkis2.standard_inchikey.isin(shared_full).to_numpy(),
        "connectivity_block_DAVIS": davis.connectivity_block.isin(shared_connectivity).to_numpy(),
        "connectivity_block_PKIS2": pkis2.connectivity_block.isin(shared_connectivity).to_numpy(),
        "nonempty_bemis_murcko_scaffold_DAVIS": np.isin(davis_scaffold, list(shared_scaffold)),
        "nonempty_bemis_murcko_scaffold_PKIS2": np.isin(pkis2_scaffold, list(shared_scaffold)),
        "mixed_panel_butina_cluster_DAVIS": np.isin(davis_butina, list(mixed_butina)),
        "mixed_panel_butina_cluster_PKIS2": np.isin(pkis2_butina, list(mixed_butina)),
    }
    return pd.DataFrame.from_records(records), masks, butina_meta


def overlap_sensitivity(
    davis: pd.DataFrame,
    pkis2: pd.DataFrame,
    masks: dict[str, np.ndarray],
    *,
    qap_permutations: int,
    seed: int,
) -> pd.DataFrame:
    records = []
    for index, exclusion in enumerate(
        (
            "none",
            "full_standard_inchikey",
            "connectivity_block",
            "nonempty_bemis_murcko_scaffold",
            "mixed_panel_butina_cluster",
        )
    ):
        first_values = davis.loc[
            ~masks[f"{exclusion}_DAVIS"], list(TARGETS)
        ].to_numpy(float)
        second_values = pkis2.loc[
            ~masks[f"{exclusion}_PKIS2"], list(TARGETS)
        ].to_numpy(float)
        first_values = first_values[reliability.informative_rows(first_values)]
        second_values = second_values[reliability.informative_rows(second_values)]
        record = {
            "exclusion": exclusion,
            "davis_ligands_after_exclusion": int(len(first_values)),
            "pkis2_ligands_after_exclusion": int(len(second_values)),
            **qap_record(
                target_map(first_values),
                target_map(second_values),
                permutations=qap_permutations,
                seed=seed + index,
            ),
        }
        records.append(record)
    return pd.DataFrame.from_records(records)


def robust_cross_panel(
    davis: pd.DataFrame,
    pkis2: pd.DataFrame,
    *,
    qap_permutations: int,
    seed: int,
) -> pd.DataFrame:
    first = davis[list(TARGETS)].to_numpy(float)
    second = pkis2[list(TARGETS)].to_numpy(float)
    first = first[reliability.informative_rows(first)]
    second = second[reliability.informative_rows(second)]
    records: list[dict[str, Any]] = []
    continuous = (
        ("released_informative", "native_row_centered_pearson", target_map(first), target_map(second)),
        (
            "released_informative",
            "column_rank_then_row_centered_pearson",
            column_rank_target_map(first),
            column_rank_target_map(second),
        ),
    )
    dockstring = reliability.load_dockstring_support()
    first_match = reliability.match_panel(davis, dockstring)[1].to_numpy(float)
    second_match = reliability.match_panel(pkis2, dockstring)[1].to_numpy(float)
    first_match = first_match[reliability.informative_rows(first_match)]
    second_match = second_match[reliability.informative_rows(second_match)]
    continuous = (*continuous,
        ("exact_dockstring_match_informative", "native_row_centered_pearson", target_map(first_match), target_map(second_match)),
        ("exact_dockstring_match_informative", "column_rank_then_row_centered_pearson", column_rank_target_map(first_match), column_rank_target_map(second_match)),
    )
    for index, (support, transform, first_map, second_map) in enumerate(continuous):
        records.append(
            {
                "analysis_family": "continuous_or_rank",
                "support": support,
                "transform": transform,
                "davis_threshold": np.nan,
                "pkis2_threshold": np.nan,
                "davis_ligands": int(len(first) if support == "released_informative" else len(first_match)),
                "pkis2_ligands": int(len(second) if support == "released_informative" else len(second_match)),
                "targets": len(TARGETS),
                "target_names": ";".join(TARGETS),
                **qap_record(
                    first_map,
                    second_map,
                    permutations=qap_permutations,
                    seed=seed + index,
                ),
            }
        )
    counter = len(continuous)
    floor = np.isclose(first, 5.0, rtol=0.0, atol=1e-12)
    for minimum_uncensored in (5, 10, 15, 20):
        eligible = (~floor).sum(axis=0) >= minimum_uncensored
        if eligible.sum() < 3:
            continue
        eligible_names = tuple(np.asarray(TARGETS, dtype=object)[eligible].tolist())
        for transform, map_function in (
            ("native_row_centered_pearson", target_map),
            ("column_rank_then_row_centered_pearson", column_rank_target_map),
        ):
            records.append(
                {
                    "analysis_family": "davis_minimum_uncensored_target_filter",
                    "support": "released_informative",
                    "transform": transform,
                    "davis_threshold": float(minimum_uncensored),
                    "pkis2_threshold": np.nan,
                    "davis_ligands": int(len(first)),
                    "pkis2_ligands": int(len(second)),
                    "targets": int(eligible.sum()),
                    "target_names": ";".join(eligible_names),
                    **qap_record(
                        map_function(first[:, eligible]),
                        map_function(second[:, eligible]),
                        permutations=qap_permutations,
                        seed=seed + counter,
                    ),
                }
            )
            counter += 1
    # Fix one estimable target set across the full threshold grid.  Every kept
    # target has at least five positives and five negatives in both panels at
    # every declared threshold, preventing changing edge support or constant
    # columns from selecting the result.
    feasible = np.ones(len(TARGETS), dtype=bool)
    for values, thresholds in ((first, (5.0, 5.5, 6.0)), (second, (50.0, 70.0, 80.0))):
        for threshold in thresholds:
            positives = (values > threshold).sum(axis=0)
            feasible &= (positives >= 5) & ((len(values) - positives) >= 5)
    if feasible.sum() < 3:
        raise ValueError("binary grid has fewer than three common estimable targets")
    feasible_names = tuple(np.asarray(TARGETS, dtype=object)[feasible].tolist())
    # Thresholds are a fully crossed, predeclared sensitivity grid.  No pair is
    # selected after inspecting concordance.
    for davis_threshold in (5.0, 5.5, 6.0):
        for pkis2_threshold in (50.0, 70.0, 80.0):
            first_binary = (first[:, feasible] > davis_threshold).astype(float)
            second_binary = (second[:, feasible] > pkis2_threshold).astype(float)
            records.append(
                {
                    "analysis_family": "fully_crossed_binary_threshold_grid",
                    "support": "released_informative_fixed_common_estimable_targets",
                    "transform": "binary_then_row_centered_pearson",
                    "davis_threshold": davis_threshold,
                    "pkis2_threshold": pkis2_threshold,
                    "davis_ligands": int(len(first_binary)),
                    "pkis2_ligands": int(len(second_binary)),
                    "targets": int(feasible.sum()),
                    "target_names": ";".join(feasible_names),
                    "davis_positive_cell_fraction": float(first_binary.mean()),
                    "pkis2_positive_cell_fraction": float(second_binary.mean()),
                    **qap_record(
                        target_map(first_binary),
                        target_map(second_binary),
                        permutations=qap_permutations,
                        seed=seed + counter,
                    ),
                }
            )
            counter += 1
    return pd.DataFrame.from_records(records)


def davis_censoring_edge_audit(davis: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Expose how released pKd=5 floor prevalence relates to DAVIS map edges."""
    values = davis[list(TARGETS)].to_numpy(float)
    informative = reliability.informative_rows(values)
    values = values[informative]
    floor = np.isclose(values, 5.0, rtol=0.0, atol=1e-12)
    continuous = target_map(values)
    ranked = column_rank_target_map(values)
    records: list[dict[str, Any]] = []
    for index, target in enumerate(TARGETS):
        records.append(
            {
                "record_type": "target",
                "target_a": target,
                "target_b": "",
                "target_a_floor_fraction": float(floor[:, index].mean()),
                "target_b_floor_fraction": np.nan,
                "target_a_uncensored_ligands": int((~floor[:, index]).sum()),
                "target_b_uncensored_ligands": np.nan,
                "jointly_uncensored_ligands": np.nan,
                "continuous_map_edge": np.nan,
                "rank_map_edge": np.nan,
                "sum_floor_fraction": np.nan,
                "absolute_floor_fraction_difference": np.nan,
            }
        )
    for first_index, second_index in itertools.combinations(range(len(TARGETS)), 2):
        first_floor = float(floor[:, first_index].mean())
        second_floor = float(floor[:, second_index].mean())
        records.append(
            {
                "record_type": "pair",
                "target_a": TARGETS[first_index],
                "target_b": TARGETS[second_index],
                "target_a_floor_fraction": first_floor,
                "target_b_floor_fraction": second_floor,
                "target_a_uncensored_ligands": int((~floor[:, first_index]).sum()),
                "target_b_uncensored_ligands": int((~floor[:, second_index]).sum()),
                "jointly_uncensored_ligands": int(
                    ((~floor[:, first_index]) & (~floor[:, second_index])).sum()
                ),
                "continuous_map_edge": float(continuous[first_index, second_index]),
                "rank_map_edge": float(ranked[first_index, second_index]),
                "sum_floor_fraction": first_floor + second_floor,
                "absolute_floor_fraction_difference": abs(first_floor - second_floor),
            }
        )
    frame = pd.DataFrame.from_records(records)
    pairs = frame.loc[frame.record_type.eq("pair")]
    design = np.column_stack(
        [
            np.ones(len(pairs)),
            pairs.sum_floor_fraction.to_numpy(float),
            pairs.absolute_floor_fraction_difference.to_numpy(float),
        ]
    )
    regressions: dict[str, Any] = {}
    for outcome in ("continuous_map_edge", "rank_map_edge"):
        response = pairs[outcome].to_numpy(float)
        coefficients = np.linalg.lstsq(design, response, rcond=None)[0]
        fitted = design @ coefficients
        total = np.square(response - response.mean()).sum()
        r2 = 1.0 - np.square(response - fitted).sum() / total
        regressions[outcome] = {
            "r_squared": float(r2),
            "intercept": float(coefficients[0]),
            "sum_floor_fraction_coefficient": float(coefficients[1]),
            "absolute_floor_fraction_difference_coefficient": float(coefficients[2]),
            "role": "descriptive in-sample censoring-association audit",
        }
    summary = {
        "released_floor": 5.0,
        "informative_ligands": int(len(values)),
        "target_floor_fraction_minimum": float(floor.mean(axis=0).min()),
        "target_floor_fraction_median": float(np.median(floor.mean(axis=0))),
        "target_floor_fraction_maximum": float(floor.mean(axis=0).max()),
        "targets_by_minimum_uncensored_count": {
            str(threshold): int(np.sum((~floor).sum(axis=0) >= threshold))
            for threshold in (5, 10, 15, 20)
        },
        "edge_regressions": regressions,
    }
    return frame, summary


def standardized_descriptors(
    pool: np.ndarray, query: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    median = np.median(pool, axis=0)
    scale = np.quantile(pool, 0.75, axis=0) - np.quantile(pool, 0.25, axis=0)
    scale[scale <= 1e-12] = 1.0
    return (pool - median) / scale, (query - median) / scale, median, scale


def mw_stratum_indices(pool_mw: np.ndarray, query_mw: np.ndarray) -> list[np.ndarray]:
    edges = np.unique(np.quantile(pool_mw, np.linspace(0.0, 1.0, MW_BINS + 1)))
    if len(edges) < 4:
        raise ValueError("MW support has too few distinct quantile boundaries")
    pool_bin = np.clip(np.searchsorted(edges[1:-1], pool_mw, side="right"), 0, len(edges) - 2)
    query_bin = np.clip(np.searchsorted(edges[1:-1], query_mw, side="right"), 0, len(edges) - 2)
    output = []
    for value in query_bin:
        candidates = np.flatnonzero(pool_bin == value)
        if not len(candidates):
            raise RuntimeError("an MW matching stratum is empty")
        output.append(candidates)
    return output


def draw_unique_candidates(
    candidates: Sequence[np.ndarray],
    rng: np.random.Generator,
    groups: np.ndarray | None = None,
) -> np.ndarray:
    order = rng.permutation(len(candidates))
    selected = np.empty(len(candidates), dtype=np.int64)
    used_rows: set[int] = set()
    used_groups: set[Any] = set()
    for query_index in order:
        options = np.asarray(candidates[query_index], dtype=np.int64)
        choice = None
        if len(options) > 1_024:
            # MW strata contain thousands of rows.  Random probing avoids
            # constructing a full permutation for every query and replicate;
            # collisions are rare at N<=154, with a complete fallback below.
            probe = options[
                rng.integers(0, len(options), size=min(1_024, len(options)))
            ]
        else:
            probe = options[rng.permutation(len(options))]
        for candidate in probe:
            group = None if groups is None else groups[candidate]
            if int(candidate) in used_rows or (groups is not None and group in used_groups):
                continue
            choice = int(candidate)
            break
        if choice is None and len(options) > 1_024:
            for candidate in options[rng.permutation(len(options))]:
                group = None if groups is None else groups[candidate]
                if int(candidate) in used_rows or (
                    groups is not None and group in used_groups
                ):
                    continue
                choice = int(candidate)
                break
        if choice is None:
            raise RuntimeError("matching candidate set could not satisfy uniqueness")
        selected[query_index] = choice
        used_rows.add(choice)
        if groups is not None:
            used_groups.add(groups[choice])
    return np.sort(selected)


def uniform_distinct_group_draw(
    row_groups: np.ndarray,
    size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Row-uniform sequential draw, rejecting repeated chemical groups."""
    row_groups = np.asarray(row_groups, dtype=object)
    if len(np.unique(row_groups)) < size:
        raise ValueError("fewer distinct chemical groups than requested rows")
    selected: list[int] = []
    used_groups: set[Any] = set()
    while len(selected) < size:
        needed = size - len(selected)
        candidates = rng.integers(0, len(row_groups), size=max(32, 3 * needed))
        for candidate in candidates:
            group = row_groups[int(candidate)]
            if group in used_groups:
                continue
            used_groups.add(group)
            selected.append(int(candidate))
            if len(selected) == size:
                break
    return np.sort(np.asarray(selected, dtype=np.int64))


def descriptor_balance(
    selected: np.ndarray, pool_z: np.ndarray, query_z: np.ndarray
) -> tuple[float, float, np.ndarray]:
    difference = pool_z[selected].mean(axis=0) - query_z.mean(axis=0)
    return (
        float(np.sqrt(np.mean(np.square(difference)))),
        float(np.max(np.abs(difference))),
        difference,
    )


def support_control_analysis(
    dockstring: pd.DataFrame,
    panels: OrderedDict[str, pd.DataFrame],
    *,
    repeats: int,
    qap_permutations: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    broad_values, broad_meta = reliability.broad_reference(dockstring, panels)
    excluded_connectivity = set()
    for panel in panels.values():
        excluded_connectivity.update(panel.connectivity_block.astype(str))
    keep = ~dockstring.connectivity_block.astype(str).isin(excluded_connectivity)
    pool = dockstring.loc[keep].reset_index(drop=True)
    if len(pool) != len(broad_values):
        raise RuntimeError("broad reference and de-leaked control pool disagree")
    pool_values = np.minimum(pool[list(TARGETS)].to_numpy(float), 0.0)
    pool_desc = molecular_descriptors(pool.smiles.astype(str))
    contract = pd.read_csv(
        reliability.DEFAULT_IDENTITY_CONTRACT,
        usecols=["complete_support_row_index", "raw_murcko_scaffold"],
    )
    if not np.array_equal(
        contract.complete_support_row_index.to_numpy(dtype=np.int64),
        np.arange(len(dockstring), dtype=np.int64),
    ):
        raise ValueError("identity-contract scaffolds do not align to DOCKSTRING")
    all_groups = contract.raw_murcko_scaffold.fillna("").astype(str).to_numpy(dtype=object)
    acyclic = all_groups == ""
    source_indices = dockstring.source_row_index.to_numpy(dtype=np.int64)
    all_groups[acyclic] = np.asarray(
        [f"ACYCLIC_SOURCE_ROW_{value}" for value in source_indices[acyclic]],
        dtype=object,
    )
    pool_groups = all_groups[keep.to_numpy()]

    records: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for panel_index, (panel_name, panel) in enumerate(panels.items()):
        docking, experimental, matched_smiles, _ = reliability.match_panel(panel, dockstring)
        docking_values = np.minimum(docking.to_numpy(float), 0.0)
        experimental_values = experimental.to_numpy(float)
        informative = reliability.informative_rows(experimental_values)
        docking_values = docking_values[informative]
        experimental_values = experimental_values[informative]
        query_smiles = matched_smiles.to_numpy(dtype=object)[informative]
        query_desc = molecular_descriptors(query_smiles)
        pool_z, query_z, _, _ = standardized_descriptors(pool_desc, query_desc)
        experimental_maps = {
            "raw": reliability.target_correlation(experimental_values),
            "row_centered_residual": target_map(experimental_values),
        }
        observed_maps = {
            "raw": reliability.target_correlation(docking_values),
            "row_centered_residual": target_map(docking_values),
        }
        observed = {
            transform: map_spearman(observed_maps[transform], experimental_maps[transform])
            for transform in experimental_maps
        }
        observed_qap = {
            transform: qap_record(
                observed_maps[transform],
                experimental_maps[transform],
                permutations=qap_permutations,
                seed=seed + panel_index * 1_000_000 + 800_000 + transform_index,
            )
            for transform_index, transform in enumerate(experimental_maps)
        }
        mw_candidates = mw_stratum_indices(pool_desc[:, 1], query_desc[:, 1])
        neighbours = NearestNeighbors(
            n_neighbors=min(NEIGHBOURS, len(pool_z)), algorithm="kd_tree", n_jobs=-1
        ).fit(pool_z)
        _, neighbour_indices = neighbours.kneighbors(query_z, return_distance=True)
        candidates = [row for row in neighbour_indices]
        rng = np.random.default_rng(seed + panel_index * 1_000_000)
        for repetition in range(repeats):
            positive_rows = rng.choice(
                len(pool), size=2 * len(query_desc), replace=False
            )
            positive_first = np.sort(positive_rows[: len(query_desc)])
            positive_second = np.sort(positive_rows[len(query_desc) :])
            positive_control_by_transform = {}
            for transform in experimental_maps:
                map_function = (
                    reliability.target_correlation
                    if transform == "raw"
                    else target_map
                )
                positive_control_by_transform[transform] = map_spearman(
                    map_function(pool_values[positive_first]),
                    map_function(pool_values[positive_second]),
                )
            designs = {
                "uniform_random_rows": np.sort(
                    rng.choice(len(pool), size=len(query_desc), replace=False)
                ),
                "mw_20_quantile_matched": draw_unique_candidates(mw_candidates, rng),
                "seven_descriptor_nearest_256": draw_unique_candidates(candidates, rng),
                "uniform_scaffold_distinct": uniform_distinct_group_draw(
                    pool_groups,
                    len(query_desc),
                    rng,
                ),
                "seven_descriptor_nearest_256_scaffold_distinct": draw_unique_candidates(
                    candidates, rng, pool_groups
                ),
            }
            for design, selected in designs.items():
                rms_balance, max_balance, descriptor_differences = descriptor_balance(
                    selected, pool_z, query_z
                )
                for transform in experimental_maps:
                    map_function = (
                        reliability.target_correlation
                        if transform == "raw"
                        else target_map
                    )
                    control = map_spearman(
                        map_function(pool_values[selected]), experimental_maps[transform]
                    )
                    records.append(
                        {
                            "panel": panel_name,
                            "transformation": transform,
                            "control_design": design,
                            "repetition": repetition,
                            "ligands": int(len(selected)),
                            "observed_exact_support_spearman": observed[transform],
                            "observed_target_label_qap_p_two_sided": observed_qap[transform][
                                "target_label_qap_p_two_sided"
                            ],
                            "observed_target_label_qap_permutations": qap_permutations,
                            "control_spearman": control,
                            "observed_minus_control_spearman": float(
                                observed[transform] - control
                            ),
                            "equal_n_row_disjoint_vina_self_spearman": positive_control_by_transform[
                                transform
                            ],
                            "descriptor_mean_difference_rms_iqr_units": rms_balance,
                            "descriptor_max_abs_mean_difference_iqr_units": max_balance,
                            "unique_murcko_source_groups": int(
                                len(np.unique(pool_groups[selected]))
                            ),
                            **{
                                f"mean_difference_iqr_units_{name}": float(value)
                                for name, value in zip(
                                    DESCRIPTORS, descriptor_differences, strict=True
                                )
                            },
                        }
                    )
        panel_frame = pd.DataFrame.from_records(
            [record for record in records if record["panel"] == panel_name]
        )
        for (transform, design), group in panel_frame.groupby(
            ["transformation", "control_design"], sort=True
        ):
            null = group.control_spearman.to_numpy(float)
            contrast = observed[transform] - null
            summaries.append(
                {
                    "panel": panel_name,
                    "transformation": transform,
                    "control_design": design,
                    "repetitions": int(len(group)),
                    "ligands": int(len(query_desc)),
                    "observed_exact_support_spearman": observed[transform],
                    "observed_target_label_qap_p_two_sided": observed_qap[transform][
                        "target_label_qap_p_two_sided"
                    ],
                    "control_spearman_median": float(np.median(null)),
                    "control_spearman_q025": float(np.quantile(null, 0.025)),
                    "control_spearman_q975": float(np.quantile(null, 0.975)),
                    "observed_percentile_among_controls": float(
                        (1 + np.sum(null <= observed[transform])) / (len(null) + 1)
                    ),
                    "observed_minus_control_median": float(np.median(contrast)),
                    "observed_minus_control_q025": float(np.quantile(contrast, 0.025)),
                    "observed_minus_control_q975": float(np.quantile(contrast, 0.975)),
                    "descriptor_balance_rms_median": float(
                        group.descriptor_mean_difference_rms_iqr_units.median()
                    ),
                    "unique_groups_median": float(group.unique_murcko_source_groups.median()),
                    "equal_n_row_disjoint_vina_self_spearman_median": float(
                        group.equal_n_row_disjoint_vina_self_spearman.median()
                    ),
                    "equal_n_row_disjoint_vina_self_spearman_q025": float(
                        group.equal_n_row_disjoint_vina_self_spearman.quantile(0.025)
                    ),
                    "equal_n_row_disjoint_vina_self_spearman_q975": float(
                        group.equal_n_row_disjoint_vina_self_spearman.quantile(0.975)
                    ),
                    **{
                        f"mean_difference_iqr_units_{name}_median": float(
                            group[f"mean_difference_iqr_units_{name}"].median()
                        )
                        for name in DESCRIPTORS
                    },
                }
            )
    metadata = {
        "pool": broad_meta,
        "control_pool_rows": int(len(pool)),
        "descriptor_names": list(DESCRIPTORS),
        "descriptor_standardization": "de-leaked pool median and IQR",
        "mw_matching": f"one candidate per query ligand from {MW_BINS} pool-quantile strata",
        "multidescriptor_matching": (
            f"random unique candidate among the {NEIGHBOURS} nearest pool molecules "
            "in seven-dimensional median/IQR-standardized descriptor space"
        ),
        "scaffold_distinct_sensitivity": (
            "at most one row per frozen nonempty Bemis-Murcko scaffold; each acyclic "
            "source row is its own group. This is a de-redundancy sensitivity, not a matched "
            "bootstrap of the observed support."
        ),
        "positive_control": (
            "map Spearman between two independent equal-N row-disjoint Vina supports "
            "from the same de-leaked library; this distinguishes map-estimation power "
            "from cross-modal alignment"
        ),
    }
    return (
        pd.DataFrame.from_records(records),
        pd.DataFrame.from_records(summaries),
        metadata,
    )


def distance_from_map(correlation: np.ndarray, objective: str) -> np.ndarray:
    values = np.asarray(correlation, dtype=np.float64)
    if objective == "signed_1_minus_r":
        distance = 1.0 - values
    elif objective == "absolute_1_minus_abs_r":
        distance = 1.0 - np.abs(values)
    elif objective == "squared_1_minus_r2":
        distance = 1.0 - np.square(values)
    else:
        raise ValueError(f"unknown objective: {objective}")
    distance = np.maximum((distance + distance.T) / 2.0, 0.0)
    np.fill_diagonal(distance, 0.0)
    return distance


def all_panels(n_targets: int = len(TARGETS), k: int = K) -> np.ndarray:
    return np.asarray(list(itertools.combinations(range(n_targets), k)), dtype=np.int16)


def coverage_distribution(
    distance: np.ndarray, panels: np.ndarray, batch_size: int = 4_096
) -> np.ndarray:
    values = np.empty(len(panels), dtype=np.float64)
    for start in range(0, len(panels), batch_size):
        stop = min(start + batch_size, len(panels))
        block = panels[start:stop]
        # distance[:, block] has shape targets x panels x selected targets.
        values[start:stop] = distance[:, block].min(axis=2).mean(axis=0)
    return values


def panel8_analysis(
    broad_vina: np.ndarray,
    davis: np.ndarray,
    pkis2: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    matrices = {
        "broad_deleaked_Vina": broad_vina,
        "DAVIS_experiment": davis,
        "PKIS2_experiment": pkis2,
    }
    panels = all_panels()
    expected = math.comb(len(TARGETS), K)
    if len(panels) != expected:
        raise RuntimeError("eight-target enumeration is incomplete")
    records: list[dict[str, Any]] = []
    selections: list[dict[str, Any]] = []
    objectives = (
        "signed_1_minus_r",
        "absolute_1_minus_abs_r",
        "squared_1_minus_r2",
    )
    for transformation, map_function in (
        ("raw", reliability.target_correlation),
        ("row_centered_residual", target_map),
    ):
        maps = {name: map_function(matrix) for name, matrix in matrices.items()}
        for objective in objectives:
            distances = {
                name: distance_from_map(item, objective) for name, item in maps.items()
            }
            distributions = {
                name: coverage_distribution(distance, panels)
                for name, distance in distances.items()
            }
            selected_indices = {
                name: int(np.argmin(distribution))
                for name, distribution in distributions.items()
            }
            vina_index = selected_indices["broad_deleaked_Vina"]
            vina_panel = panels[vina_index]
            for evaluation in ("DAVIS_experiment", "PKIS2_experiment"):
                transfer_source = (
                    "PKIS2_experiment"
                    if evaluation == "DAVIS_experiment"
                    else "DAVIS_experiment"
                )
                transfer_index = selected_indices[transfer_source]
                transfer_panel = panels[transfer_index]
                distribution = distributions[evaluation]
                oracle_index = selected_indices[evaluation]
                oracle_panel = panels[oracle_index]
                vina_value = float(distribution[vina_index])
                transfer_value = float(distribution[transfer_index])
                oracle_value = float(distribution[oracle_index])
                random_median = float(np.median(distribution))
                denominator = random_median - oracle_value
                captured = (
                    float((random_median - vina_value) / denominator)
                    if denominator > 1e-15
                    else np.nan
                )
                records.append(
                    {
                        "transformation": transformation,
                        "objective": objective,
                        "selection_map": "broad_deleaked_Vina",
                        "cross_panel_experimental_selection_map": transfer_source,
                        "evaluation_map": evaluation,
                        "targets_selected": K,
                        "all_possible_panels": int(len(panels)),
                        "vina_selected_mean_nearest_distance": vina_value,
                        "cross_panel_experimental_selected_mean_nearest_distance": transfer_value,
                        "experimental_oracle_mean_nearest_distance": oracle_value,
                        "absolute_vina_minus_oracle_loss": float(
                            vina_value - oracle_value
                        ),
                        "absolute_cross_panel_experimental_minus_oracle_loss": float(
                            transfer_value - oracle_value
                        ),
                        "vina_minus_cross_panel_experimental_transfer_distance": float(
                            vina_value - transfer_value
                        ),
                        "all_panel_median_mean_nearest_distance": random_median,
                        "all_panel_q025": float(np.quantile(distribution, 0.025)),
                        "all_panel_q975": float(np.quantile(distribution, 0.975)),
                        "fraction_all_panels_no_worse_than_vina": float(
                            np.mean(distribution <= vina_value + 1e-15)
                        ),
                        "fraction_all_panels_no_worse_than_cross_panel_experimental": float(
                            np.mean(distribution <= transfer_value + 1e-15)
                        ),
                        "fraction_all_panels_worse_than_or_equal_to_vina": float(
                            np.mean(distribution >= vina_value - 1e-15)
                        ),
                        "random_median_minus_vina_improvement": float(
                            random_median - vina_value
                        ),
                        "random_median_minus_cross_panel_experimental_improvement": float(
                            random_median - transfer_value
                        ),
                        "fraction_oracle_headroom_captured_vs_all_panel_median": captured,
                    }
                )
                for source, selected in (
                    ("Vina_selected", vina_panel),
                    ("cross_panel_experimental_selected", transfer_panel),
                    ("in_sample_experimental_oracle", oracle_panel),
                ):
                    for target_index in selected:
                        selections.append(
                            {
                                "transformation": transformation,
                                "objective": objective,
                                "evaluation_map": evaluation,
                                "selection_source": source,
                                "target_index": int(target_index),
                                "target": TARGETS[int(target_index)],
                            }
                        )
    # Prespecified censoring/assay-scale sensitivity: keep the signed residual
    # Vina panel fixed, but evaluate it on column-ranked experimental maps.
    residual_vina_distance = distance_from_map(
        target_map(broad_vina), "signed_1_minus_r"
    )
    residual_vina_distribution = coverage_distribution(
        residual_vina_distance, panels
    )
    residual_vina_index = int(np.argmin(residual_vina_distribution))
    residual_vina_panel = panels[residual_vina_index]
    for evaluation, matrix in (
        ("DAVIS_experiment", davis),
        ("PKIS2_experiment", pkis2),
    ):
        rank_distance = distance_from_map(
            column_rank_target_map(matrix), "signed_1_minus_r"
        )
        distribution = coverage_distribution(rank_distance, panels)
        oracle_index = int(np.argmin(distribution))
        oracle_panel = panels[oracle_index]
        vina_value = float(distribution[residual_vina_index])
        oracle_value = float(distribution[oracle_index])
        random_median = float(np.median(distribution))
        denominator = random_median - oracle_value
        records.append(
            {
                "transformation": "column_rank_then_row_centered_evaluation",
                "objective": "signed_1_minus_r",
                "selection_map": "broad_deleaked_Vina_row_centered_residual",
                "cross_panel_experimental_selection_map": "not_evaluated",
                "evaluation_map": evaluation,
                "targets_selected": K,
                "all_possible_panels": int(len(panels)),
                "vina_selected_mean_nearest_distance": vina_value,
                "cross_panel_experimental_selected_mean_nearest_distance": np.nan,
                "experimental_oracle_mean_nearest_distance": oracle_value,
                "absolute_vina_minus_oracle_loss": float(vina_value - oracle_value),
                "absolute_cross_panel_experimental_minus_oracle_loss": np.nan,
                "vina_minus_cross_panel_experimental_transfer_distance": np.nan,
                "all_panel_median_mean_nearest_distance": random_median,
                "all_panel_q025": float(np.quantile(distribution, 0.025)),
                "all_panel_q975": float(np.quantile(distribution, 0.975)),
                "fraction_all_panels_no_worse_than_vina": float(
                    np.mean(distribution <= vina_value + 1e-15)
                ),
                "fraction_all_panels_no_worse_than_cross_panel_experimental": np.nan,
                "fraction_all_panels_worse_than_or_equal_to_vina": float(
                    np.mean(distribution >= vina_value - 1e-15)
                ),
                "random_median_minus_vina_improvement": float(
                    random_median - vina_value
                ),
                "random_median_minus_cross_panel_experimental_improvement": np.nan,
                "fraction_oracle_headroom_captured_vs_all_panel_median": (
                    float((random_median - vina_value) / denominator)
                    if denominator > 1e-15
                    else np.nan
                ),
            }
        )
        for source, selected in (
            ("Vina_selected", residual_vina_panel),
            ("in_sample_experimental_oracle", oracle_panel),
        ):
            for target_index in selected:
                selections.append(
                    {
                        "transformation": "column_rank_then_row_centered_evaluation",
                        "objective": "signed_1_minus_r",
                        "evaluation_map": evaluation,
                        "selection_source": source,
                        "target_index": int(target_index),
                        "target": TARGETS[int(target_index)],
                    }
                )
    metadata = {
        "targets": len(TARGETS),
        "selected_targets": K,
        "all_possible_panels": expected,
        "enumeration": "complete lexicographic enumeration; no Monte Carlo panel sample",
        "coverage": "mean over all 21 targets of distance to the nearest selected target",
        "oracle": "minimum on the named experimental map among all C(21,8) panels",
        "oracle_boundary": (
            "the experimental oracle is selected and evaluated on the same finite map; "
            "it is an in-sample descriptive lower bound and can fit sampling or censoring "
            "noise. Consequently the normalized headroom ratio is secondary only"
        ),
        "cross_panel_experimental_transfer": (
            "the exact k=8 optimum selected on DAVIS is evaluated on PKIS2 and vice "
            "versa, so selection and evaluation use different chemical supports and "
            "assay technologies. This is still fixed-target map coverage, not biological "
            "or compound-ranking validation"
        ),
    }
    return pd.DataFrame.from_records(records), pd.DataFrame.from_records(selections), metadata


def summarize_panel8_bootstrap(frame: pd.DataFrame) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    metrics = [
        "raw_fixed_vina_excess_loss_over_in_sample_oracle",
        "raw_fraction_all_panels_no_worse",
        "residual_fixed_vina_excess_loss_over_in_sample_oracle",
        "residual_fraction_all_panels_no_worse",
        "residual_minus_raw_excess_loss_over_in_sample_oracle",
        "residual_minus_raw_fraction_no_worse",
    ]
    for (panel, scheme), group in frame.groupby(
        ["panel", "resampling_scheme"], sort=True
    ):
        successful = group.loc[group.status.eq("ok")]
        record: dict[str, Any] = {
            "panel": panel,
            "resampling_scheme": scheme,
            "repetitions": int(len(group)),
            "successful_repetitions": int(len(successful)),
            "failed_repetitions": int(len(group) - len(successful)),
            "resampling_unit": "panel-specific Butina chemical cluster",
        }
        for metric in metrics:
            values = successful[metric].to_numpy(float)
            record[f"{metric}_median"] = float(np.median(values))
            record[f"{metric}_q025"] = float(np.quantile(values, 0.025))
            record[f"{metric}_q975"] = float(np.quantile(values, 0.975))
        summaries.append(record)
    return summaries


def panel8_cluster_bootstrap(
    broad_vina: np.ndarray,
    panels: OrderedDict[str, pd.DataFrame],
    *,
    repetitions: int,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Cluster-bootstrap uncertainty for the fixed signed-distance Vina panel.

    The raw and residual Vina-selected panels are fixed once from the de-leaked
    broad Vina maps.  Each experimental panel resamples whole panel-specific
    Butina clusters with replacement; all member rows and multiplicities of a
    selected cluster are retained.  Selection therefore remains outcome-blind.
    """
    if repetitions < 100:
        raise ValueError("panel bootstrap requires at least 100 repetitions")
    all_k8 = all_panels()
    fixed_indices: dict[str, int] = {}
    for transformation, map_function in (
        ("raw", reliability.target_correlation),
        ("residual", target_map),
    ):
        distance = distance_from_map(
            map_function(broad_vina), "signed_1_minus_r"
        )
        distribution = coverage_distribution(distance, all_k8)
        fixed_indices[transformation] = int(np.argmin(distribution))

    records: list[dict[str, Any]] = []
    for panel_index, (panel_name, panel) in enumerate(panels.items()):
        values = panel[list(TARGETS)].to_numpy(float)
        informative = reliability.informative_rows(values)
        values = values[informative]
        smiles = panel.smiles.to_numpy(dtype=object)[informative]
        _, labels = reliability.panel_clusters(pd.Series(smiles))
        unique = np.unique(labels)
        members = {label: np.flatnonzero(labels == label) for label in unique}
        rng = np.random.default_rng(seed + panel_index * 1_000_000)
        for repetition in range(repetitions):
            sampled_labels = unique[
                rng.integers(0, len(unique), size=len(unique))
            ]
            rows = np.concatenate([members[label] for label in sampled_labels])
            bootstrap = values[rows]
            record: dict[str, Any] = {
                "panel": panel_name,
                "resampling_scheme": "ordinary_cluster_resample_with_replacement",
                "repetition": repetition,
                "status": "ok",
                "source_ligands": int(len(values)),
                "source_clusters": int(len(unique)),
                "drawn_clusters_with_replacement": int(len(sampled_labels)),
                "unique_clusters_drawn": int(len(np.unique(sampled_labels))),
                "bootstrap_rows_with_multiplicity": int(len(rows)),
            }
            try:
                for transformation, map_function in (
                    ("raw", reliability.target_correlation),
                    ("residual", target_map),
                ):
                    correlation = map_function(bootstrap)
                    distance = distance_from_map(correlation, "signed_1_minus_r")
                    distribution = coverage_distribution(distance, all_k8)
                    fixed_value = float(distribution[fixed_indices[transformation]])
                    oracle_value = float(distribution.min())
                    median_value = float(np.median(distribution))
                    record[f"{transformation}_fixed_vina_mean_nearest_distance"] = fixed_value
                    record[f"{transformation}_oracle_mean_nearest_distance"] = oracle_value
                    record[f"{transformation}_all_panel_median"] = median_value
                    record[
                        f"{transformation}_fixed_vina_excess_loss_over_in_sample_oracle"
                    ] = float(
                        fixed_value - oracle_value
                    )
                    record[f"{transformation}_fraction_all_panels_no_worse"] = float(
                        np.mean(distribution <= fixed_value + 1e-15)
                    )
                    record[f"{transformation}_exact_rank_best_is_1"] = int(
                        1 + np.sum(distribution < fixed_value - 1e-15)
                    )
                    denominator = median_value - oracle_value
                    record[f"{transformation}_median_headroom_captured"] = (
                        float((median_value - fixed_value) / denominator)
                        if denominator > 1e-15
                        else np.nan
                    )
                record[
                    "residual_minus_raw_excess_loss_over_in_sample_oracle"
                ] = float(
                    record[
                        "residual_fixed_vina_excess_loss_over_in_sample_oracle"
                    ]
                    - record["raw_fixed_vina_excess_loss_over_in_sample_oracle"]
                )
                record["residual_minus_raw_fraction_no_worse"] = float(
                    record["residual_fraction_all_panels_no_worse"]
                    - record["raw_fraction_all_panels_no_worse"]
                )
            except ValueError as error:
                record["status"] = "failed_constant_or_invalid_target"
                record["failure_message"] = str(error)
            records.append(record)
    frame = pd.DataFrame.from_records(records)
    metadata = {
        "repetitions_per_panel": int(repetitions),
        "objective": "signed_1_minus_r",
        "all_possible_panels": int(len(all_k8)),
        "selection_contract": (
            "raw and residual k=8 Vina panels fixed once on the de-leaked broad "
            "DOCKSTRING map before any experimental resampling"
        ),
        "resampling_contract": (
            "sample the panel-specific Butina clusters with replacement; concatenate "
            "every source row in each sampled cluster, preserving within-cluster rows "
            "and multiplicity when a cluster is drawn repeatedly"
        ),
        "summary": summarize_panel8_bootstrap(frame),
        "boundary": (
            "cluster-bootstrap intervals describe chemical-support perturbation within "
            "each released panel; they do not sample targets, assay technologies, or "
            "the universe of kinases"
        ),
    }
    return frame, metadata


def weighted_target_map(
    matrix: np.ndarray, weights: np.ndarray, *, row_centered: bool
) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if (
        values.ndim != 2
        or weights.shape != (len(values),)
        or not np.isfinite(values).all()
        or not np.isfinite(weights).all()
        or np.any(weights <= 0.0)
    ):
        raise ValueError("weighted target map requires finite values and positive weights")
    if row_centered:
        values = row_center(values)
    normalized = weights / weights.sum()
    mean = np.sum(values * normalized[:, None], axis=0)
    centered = values - mean
    covariance = (centered * normalized[:, None]).T @ centered
    variance = np.diag(covariance)
    if np.any(variance <= 1e-15):
        raise ValueError("weighted target map contains a constant target")
    result = covariance / np.sqrt(np.outer(variance, variance))
    result = np.clip((result + result.T) / 2.0, -1.0, 1.0)
    np.fill_diagonal(result, 1.0)
    return result


def panel8_cluster_multiplier_bootstrap(
    broad_vina: np.ndarray,
    panels: OrderedDict[str, pd.DataFrame],
    *,
    repetitions: int,
    seed: int,
) -> pd.DataFrame:
    """Positive Exp(1) cluster multipliers retain every chemical cluster."""
    all_k8 = all_panels()
    fixed_indices: dict[str, int] = {}
    for transformation, map_function in (
        ("raw", reliability.target_correlation),
        ("residual", target_map),
    ):
        distribution = coverage_distribution(
            distance_from_map(map_function(broad_vina), "signed_1_minus_r"), all_k8
        )
        fixed_indices[transformation] = int(np.argmin(distribution))
    records: list[dict[str, Any]] = []
    for panel_index, (panel_name, panel) in enumerate(panels.items()):
        values = panel[list(TARGETS)].to_numpy(float)
        informative = reliability.informative_rows(values)
        values = values[informative]
        smiles = panel.smiles.to_numpy(dtype=object)[informative]
        _, labels = reliability.panel_clusters(pd.Series(smiles))
        unique, inverse = np.unique(labels, return_inverse=True)
        rng = np.random.default_rng(seed + panel_index * 1_000_000)
        for repetition in range(repetitions):
            cluster_weights = rng.exponential(scale=1.0, size=len(unique))
            row_weights = cluster_weights[inverse]
            record: dict[str, Any] = {
                "panel": panel_name,
                "resampling_scheme": "positive_exp1_cluster_multiplier",
                "repetition": repetition,
                "status": "ok",
                "source_ligands": int(len(values)),
                "source_clusters": int(len(unique)),
                "drawn_clusters_with_replacement": np.nan,
                "unique_clusters_drawn": int(len(unique)),
                "bootstrap_rows_with_multiplicity": int(len(values)),
                "effective_row_sample_size": float(
                    np.square(row_weights.sum()) / np.square(row_weights).sum()
                ),
            }
            for transformation, row_centered in (("raw", False), ("residual", True)):
                correlation = weighted_target_map(
                    values, row_weights, row_centered=row_centered
                )
                distance = distance_from_map(correlation, "signed_1_minus_r")
                distribution = coverage_distribution(distance, all_k8)
                fixed_value = float(distribution[fixed_indices[transformation]])
                oracle_value = float(distribution.min())
                median_value = float(np.median(distribution))
                record[f"{transformation}_fixed_vina_mean_nearest_distance"] = fixed_value
                record[f"{transformation}_oracle_mean_nearest_distance"] = oracle_value
                record[f"{transformation}_all_panel_median"] = median_value
                record[
                    f"{transformation}_fixed_vina_excess_loss_over_in_sample_oracle"
                ] = float(fixed_value - oracle_value)
                record[f"{transformation}_fraction_all_panels_no_worse"] = float(
                    np.mean(distribution <= fixed_value + 1e-15)
                )
                record[f"{transformation}_exact_rank_best_is_1"] = int(
                    1 + np.sum(distribution < fixed_value - 1e-15)
                )
                denominator = median_value - oracle_value
                record[f"{transformation}_median_headroom_captured"] = (
                    float((median_value - fixed_value) / denominator)
                    if denominator > 1e-15
                    else np.nan
                )
            record[
                "residual_minus_raw_excess_loss_over_in_sample_oracle"
            ] = float(
                record["residual_fixed_vina_excess_loss_over_in_sample_oracle"]
                - record["raw_fixed_vina_excess_loss_over_in_sample_oracle"]
            )
            record["residual_minus_raw_fraction_no_worse"] = float(
                record["residual_fraction_all_panels_no_worse"]
                - record["raw_fraction_all_panels_no_worse"]
            )
            records.append(record)
    return pd.DataFrame.from_records(records)


def analyse(
    *,
    repeats: int = DEFAULT_REPEATS,
    qap_permutations: int = DEFAULT_QAP,
    panel_bootstraps: int = DEFAULT_PANEL_BOOTSTRAPS,
    seed: int = DEFAULT_SEED,
) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
    if repeats < 100 or qap_permutations < 100:
        raise ValueError("production controls require at least 100 repeats/permutations")
    dockstring = reliability.load_dockstring_support()
    panels = reliability.load_public_panels()
    davis, pkis2 = panels.values()

    overlap, masks, butina_meta = overlap_audit(davis, pkis2)
    overlap_geometry = overlap_sensitivity(
        davis, pkis2, masks, qap_permutations=qap_permutations, seed=seed + 100
    )
    robustness = robust_cross_panel(
        davis, pkis2, qap_permutations=qap_permutations, seed=seed + 10_000
    )
    censoring_edges, censoring_summary = davis_censoring_edge_audit(davis)
    controls, control_summary, control_meta = support_control_analysis(
        dockstring,
        panels,
        repeats=repeats,
        qap_permutations=qap_permutations,
        seed=seed + 100_000,
    )

    broad_values, broad_meta = reliability.broad_reference(dockstring, panels)
    davis_values = davis[list(TARGETS)].to_numpy(float)
    pkis2_values = pkis2[list(TARGETS)].to_numpy(float)
    davis_values = davis_values[reliability.informative_rows(davis_values)]
    pkis2_values = pkis2_values[reliability.informative_rows(pkis2_values)]
    coverage, selections, coverage_meta = panel8_analysis(
        broad_values, davis_values, pkis2_values
    )
    panel_bootstrap, panel_bootstrap_meta = panel8_cluster_bootstrap(
        broad_values,
        panels,
        repetitions=panel_bootstraps,
        seed=seed + 2_000_000,
    )
    multiplier_bootstrap = panel8_cluster_multiplier_bootstrap(
        broad_values,
        panels,
        repetitions=panel_bootstraps,
        seed=seed + 3_000_000,
    )
    panel_bootstrap = pd.concat(
        [panel_bootstrap, multiplier_bootstrap], ignore_index=True, sort=False
    )
    panel_bootstrap_meta["summary"] = summarize_panel8_bootstrap(panel_bootstrap)
    panel_bootstrap_meta["primary_uncertainty_scheme"] = (
        "positive_exp1_cluster_multiplier; every chemical cluster remains represented"
    )
    panel_bootstrap_meta["ordinary_bootstrap_role"] = (
        "sensitivity only; failed constant-target replicates are reported and summaries "
        "are conditional on successful replicates"
    )

    summary = {
        "analysis": "strict-public DAVIS/PKIS2 experimental-boundary controls",
        "status": "descriptive_external_boundary_not_biological_validation",
        "support": {
            "targets": len(TARGETS),
            "target_pairs": TARGET_PAIRS,
            "davis_released_ligands": int(len(davis)),
            "davis_informative_ligands": int(len(davis_values)),
            "pkis2_released_ligands": int(len(pkis2)),
            "pkis2_informative_ligands": int(len(pkis2_values)),
        },
        "chemical_overlap": {
            "table": overlap.to_dict(orient="records"),
            "combined_butina": butina_meta,
        },
        "overlap_exclusion_geometry": overlap_geometry.to_dict(orient="records"),
        "robust_cross_panel_geometry": robustness.to_dict(orient="records"),
        "davis_censoring_audit": censoring_summary,
        "exact_support_controls": {
            "summary": control_summary.to_dict(orient="records"),
            "metadata": control_meta,
        },
        "eight_target_panel": {
            "coverage": coverage.to_dict(orient="records"),
            "metadata": coverage_meta,
            "cluster_bootstrap": panel_bootstrap_meta,
        },
        "primary_estimands": {
            "target_map": (
                "Pearson correlations among targets after subtracting each ligand's "
                "mean across the fixed 21 targets"
            ),
            "map_agreement": "Spearman correlation across the 210 strict-upper-triangle edges",
            "exact_support_control": (
                "fixed experimental map compared with a Vina map from an equal-N "
                "de-leaked DOCKSTRING support; therefore it asks whether exact chemical "
                "support is unusually aligned, not whether Vina is accurate"
            ),
        },
        "claim_boundary": (
            "All results are conditional on the fixed 21 kinase targets and released "
            "assay panels. Cross-panel agreement is robustness across assay/support "
            "changes, not a biological network. Descriptor controls are coarse balance "
            "checks, not exchangeability proofs. The eight-target exercise evaluates "
            "raw and residual map coverage only. Even DAVIS-to-PKIS2 panel transfer does "
            "not validate compound ranking, affinity calibration, or unseen-target "
            "retrieval."
        ),
        "software": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "rdkit": rdBase.rdkitVersion,
        },
        "inputs": {
            str(path.relative_to(PACKAGE)): sha256_file(path)
            for path in (
                reliability.DEFAULT_DOCKSTRING,
                reliability.DEFAULT_DAVIS,
                reliability.DEFAULT_PKIS2,
                reliability.DEFAULT_IDENTITY_CONTRACT,
            )
        },
        "configuration": {
            "control_repeats": int(repeats),
            "qap_permutations": int(qap_permutations),
            "panel_cluster_bootstraps": int(panel_bootstraps),
            "seed": int(seed),
            "molecular_weight_bins": MW_BINS,
            "descriptor_neighbours": NEIGHBOURS,
        },
        "broad_vina_reference": broad_meta,
    }
    tables = {
        "cross_panel_overlap.csv": overlap,
        "cross_panel_overlap_sensitivity.csv": overlap_geometry,
        "cross_panel_robustness.csv": robustness,
        "davis_censoring_edge_audit.csv": censoring_edges,
        "exact_support_control_replicates.csv": controls,
        "exact_support_control_summary.csv": control_summary,
        "panel8_coverage.csv": coverage,
        "panel8_cluster_bootstrap.csv": panel_bootstrap,
        "panel8_selections.csv": selections,
    }
    return summary, tables


def readme(summary: dict[str, Any]) -> str:
    return f"""# Public experimental-boundary controls

This directory is generated by
`analysis/public_experimental_boundary_controls.py` from redistributed DAVIS,
PKIS2 and DOCKSTRING row-level inputs.

It reports: raw and row-centred equal-N support controls for exact-support Vina--experiment map
agreement; DAVIS--PKIS2 identity, connectivity, scaffold and Morgan overlap;
cross-panel agreement after overlap exclusions and under rank/threshold
sensitivities; a per-edge DAVIS floor-censoring audit; and exact C(21,8)
target-panel coverage distributions.

The primary target map is the Pearson correlation matrix after subtracting each
ligand's mean over the fixed targets.  Map agreement is edge Spearman.  The
rank sensitivity first ranks ligands within each target and then removes row
effects.  Binary thresholds are a fully crossed predeclared grid on one common
target set with at least five positives and five negatives in every grid cell.

These are fixed-panel geometry diagnostics.  They do not establish biological
networks, docking accuracy, affinity calibration, compound ranking, or
unobserved-target retrieval.

Configuration: {summary['configuration']['control_repeats']} equal-N repeats,
{summary['configuration']['qap_permutations']} target-label QAP permutations,
seed {summary['configuration']['seed']}.

## Reproduce

```bash
python analysis/public_experimental_boundary_controls.py
pytest -q analysis/test_public_experimental_boundary_controls.py
```
"""


def produce(
    output: Path,
    *,
    repeats: int,
    qap_permutations: int,
    panel_bootstraps: int,
    seed: int,
) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    summary, tables = analyse(
        repeats=repeats,
        qap_permutations=qap_permutations,
        panel_bootstraps=panel_bootstraps,
        seed=seed,
    )
    for filename, frame in tables.items():
        write_csv(output / filename, frame)
    write_json(output / "summary.json", summary)
    atomic_text(output / "README.md", readme(summary))
    checksums = {
        filename: sha256_file(output / filename)
        for filename in OUTPUT_FILES
        if filename != "README.md" and (output / filename).exists()
    }
    checksums["README.md"] = sha256_file(output / "README.md")
    write_json(output / "output_checksums.json", checksums)
    return summary


def finalize_existing(
    output: Path,
    *,
    repeats: int,
    qap_permutations: int,
    panel_bootstraps: int,
    seed: int,
) -> dict[str, Any]:
    """Finalize atomic CSVs after an interrupted metadata write.

    This recovery path performs no statistical recomputation.  It is useful
    when every atomic CSV completed but JSON serialization or documentation
    failed.  Public support/overlap metadata are cheaply re-audited from the
    frozen inputs, and every result record is loaded from the existing CSVs.
    """
    table_names = [
        filename for filename in OUTPUT_FILES if filename.endswith(".csv")
    ]
    missing = [filename for filename in table_names if not (output / filename).exists()]
    if missing:
        raise FileNotFoundError(f"cannot finalize; missing result tables: {missing}")
    tables = {filename: pd.read_csv(output / filename) for filename in table_names}
    dockstring = reliability.load_dockstring_support()
    panels = reliability.load_public_panels()
    davis, pkis2 = panels.values()
    _, _, butina_meta = overlap_audit(davis, pkis2)
    _, censoring_summary = davis_censoring_edge_audit(davis)
    broad_values, broad_meta = reliability.broad_reference(dockstring, panels)
    davis_values = davis[list(TARGETS)].to_numpy(float)
    pkis2_values = pkis2[list(TARGETS)].to_numpy(float)
    davis_values = davis_values[reliability.informative_rows(davis_values)]
    pkis2_values = pkis2_values[reliability.informative_rows(pkis2_values)]
    control_meta = {
        "pool": broad_meta,
        "control_pool_rows": int(len(broad_values)),
        "descriptor_names": list(DESCRIPTORS),
        "descriptor_standardization": "de-leaked pool median and IQR",
        "mw_matching": f"one candidate per query ligand from {MW_BINS} pool-quantile strata",
        "multidescriptor_matching": (
            f"random unique candidate among the {NEIGHBOURS} nearest pool molecules "
            "in seven-dimensional median/IQR-standardized descriptor space"
        ),
        "scaffold_distinct_sensitivity": (
            "at most one row per frozen nonempty Bemis-Murcko scaffold; each acyclic "
            "source row is its own group. This is a de-redundancy sensitivity, not a "
            "matched bootstrap of the observed support."
        ),
        "positive_control": (
            "map Spearman between two independent equal-N row-disjoint Vina supports "
            "from the same de-leaked library"
        ),
    }
    coverage_meta = {
        "targets": len(TARGETS),
        "selected_targets": K,
        "all_possible_panels": math.comb(len(TARGETS), K),
        "enumeration": "complete lexicographic enumeration; no Monte Carlo panel sample",
        "coverage": "mean over all 21 targets of distance to the nearest selected target",
        "oracle": "minimum on the named experimental map among all C(21,8) panels",
        "oracle_boundary": (
            "the experimental oracle is selected and evaluated on the same finite map; "
            "it is an in-sample descriptive lower bound"
        ),
        "cross_panel_experimental_transfer": (
            "the exact k=8 optimum selected on DAVIS is evaluated on PKIS2 and vice versa"
        ),
    }
    summary = {
        "analysis": "strict-public DAVIS/PKIS2 experimental-boundary controls",
        "status": "descriptive_external_boundary_not_biological_validation",
        "support": {
            "targets": len(TARGETS),
            "target_pairs": TARGET_PAIRS,
            "davis_released_ligands": int(len(davis)),
            "davis_informative_ligands": int(len(davis_values)),
            "pkis2_released_ligands": int(len(pkis2)),
            "pkis2_informative_ligands": int(len(pkis2_values)),
        },
        "chemical_overlap": {
            "table": tables["cross_panel_overlap.csv"].to_dict(orient="records"),
            "combined_butina": butina_meta,
        },
        "overlap_exclusion_geometry": tables[
            "cross_panel_overlap_sensitivity.csv"
        ].to_dict(orient="records"),
        "robust_cross_panel_geometry": tables[
            "cross_panel_robustness.csv"
        ].to_dict(orient="records"),
        "davis_censoring_audit": censoring_summary,
        "exact_support_controls": {
            "summary": tables["exact_support_control_summary.csv"].to_dict(
                orient="records"
            ),
            "metadata": control_meta,
        },
        "eight_target_panel": {
            "coverage": tables["panel8_coverage.csv"].to_dict(orient="records"),
            "metadata": coverage_meta,
            "cluster_bootstrap": {
                "repetitions_per_panel": int(panel_bootstraps),
                "objective": "signed_1_minus_r",
                "all_possible_panels": math.comb(len(TARGETS), K),
                "summary": summarize_panel8_bootstrap(
                    tables["panel8_cluster_bootstrap.csv"]
                ),
                "primary_uncertainty_scheme": "positive_exp1_cluster_multiplier",
                "ordinary_bootstrap_role": (
                    "sensitivity with explicit constant-target failure rate"
                ),
                "selection_contract": (
                    "raw and residual k=8 Vina panels fixed once on the de-leaked broad "
                    "DOCKSTRING map before experimental resampling"
                ),
                "boundary": (
                    "conditional chemical-cluster bootstrap within each released panel"
                ),
            },
        },
        "primary_estimands": {
            "target_map": (
                "Pearson correlations among targets after subtracting each ligand's "
                "mean across the fixed 21 targets"
            ),
            "map_agreement": "Spearman correlation across the 210 strict-upper-triangle edges",
            "exact_support_control": (
                "fixed experimental map versus an equal-N de-leaked DOCKSTRING Vina map"
            ),
        },
        "claim_boundary": (
            "All results are conditional on the fixed 21 kinase targets and released "
            "assay panels. These are geometry and target-panel coverage diagnostics, "
            "not biological networks, affinity calibration, compound ranking or "
            "unseen-target retrieval."
        ),
        "software": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "rdkit": rdBase.rdkitVersion,
        },
        "inputs": {
            str(path.relative_to(PACKAGE)): sha256_file(path)
            for path in (
                reliability.DEFAULT_DOCKSTRING,
                reliability.DEFAULT_DAVIS,
                reliability.DEFAULT_PKIS2,
                reliability.DEFAULT_IDENTITY_CONTRACT,
            )
        },
        "configuration": {
            "control_repeats": int(repeats),
            "qap_permutations": int(qap_permutations),
            "panel_cluster_bootstraps": int(panel_bootstraps),
            "seed": int(seed),
            "molecular_weight_bins": MW_BINS,
            "descriptor_neighbours": NEIGHBOURS,
        },
        "broad_vina_reference": broad_meta,
        "finalization": (
            "result CSVs were atomically completed by the default producer; metadata "
            "was reconstructed from those tables after a non-finite JSON serialization "
            "guard was corrected"
        ),
    }
    write_json(output / "summary.json", summary)
    atomic_text(output / "README.md", readme(summary))
    checksums = {
        filename: sha256_file(output / filename)
        for filename in OUTPUT_FILES
        if (output / filename).exists()
    }
    write_json(output / "output_checksums.json", checksums)
    return summary


def bootstrap_existing(
    output: Path,
    *,
    repeats: int,
    qap_permutations: int,
    panel_bootstraps: int,
    seed: int,
) -> dict[str, Any]:
    """Add cluster-bootstrap uncertainty to an already completed result bundle."""
    prerequisite = output / "panel8_coverage.csv"
    if not prerequisite.exists():
        raise FileNotFoundError("panel8 point results must exist before bootstrap recovery")
    dockstring = reliability.load_dockstring_support()
    panels = reliability.load_public_panels()
    broad_values, _ = reliability.broad_reference(dockstring, panels)
    davis, pkis2 = panels.values()
    davis_values = davis[list(TARGETS)].to_numpy(float)
    pkis2_values = pkis2[list(TARGETS)].to_numpy(float)
    davis_values = davis_values[reliability.informative_rows(davis_values)]
    pkis2_values = pkis2_values[reliability.informative_rows(pkis2_values)]
    coverage, selections, _ = panel8_analysis(
        broad_values, davis_values, pkis2_values
    )
    write_csv(output / "panel8_coverage.csv", coverage)
    write_csv(output / "panel8_selections.csv", selections)
    existing_bootstrap = output / "panel8_cluster_bootstrap.csv"
    ordinary = pd.read_csv(existing_bootstrap)
    if "resampling_scheme" not in ordinary.columns:
        ordinary.insert(
            2,
            "resampling_scheme",
            "ordinary_cluster_resample_with_replacement",
        )
    ordinary = ordinary.loc[
        ordinary.resampling_scheme.eq("ordinary_cluster_resample_with_replacement")
    ].copy()
    ordinary = ordinary.rename(
        columns={
            "raw_fixed_vina_loss": "raw_fixed_vina_excess_loss_over_in_sample_oracle",
            "residual_fixed_vina_loss": "residual_fixed_vina_excess_loss_over_in_sample_oracle",
            "residual_minus_raw_loss": "residual_minus_raw_excess_loss_over_in_sample_oracle",
        }
    )
    multiplier = panel8_cluster_multiplier_bootstrap(
        broad_values,
        panels,
        repetitions=panel_bootstraps,
        seed=seed + 3_000_000,
    )
    frame = pd.concat([ordinary, multiplier], ignore_index=True, sort=False)
    write_csv(output / "panel8_cluster_bootstrap.csv", frame)
    return finalize_existing(
        output,
        repeats=repeats,
        qap_permutations=qap_permutations,
        panel_bootstraps=panel_bootstraps,
        seed=seed,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--qap-permutations", type=int, default=DEFAULT_QAP)
    parser.add_argument(
        "--panel-bootstraps", type=int, default=DEFAULT_PANEL_BOOTSTRAPS
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--finalize-existing",
        action="store_true",
        help="finalize already-computed atomic CSVs without recomputing statistics",
    )
    parser.add_argument(
        "--bootstrap-existing",
        action="store_true",
        help="add only cluster-bootstrap panel uncertainty to existing point results",
    )
    args = parser.parse_args()
    if args.finalize_existing and args.bootstrap_existing:
        parser.error("choose only one recovery mode")
    function = (
        finalize_existing
        if args.finalize_existing
        else bootstrap_existing
        if args.bootstrap_existing
        else produce
    )
    summary = function(
        args.output,
        repeats=args.repeats,
        qap_permutations=args.qap_permutations,
        panel_bootstraps=args.panel_bootstraps,
        seed=args.seed,
    )
    print(json.dumps(json_ready(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
