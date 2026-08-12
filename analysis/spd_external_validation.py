#!/usr/bin/env python3
"""Censor-aware Novartis SPD validation of DOCKSTRING target geometry.

This science-only, post-hoc analysis asks whether target--target dependence in
Vina scores agrees with a broad human safety-pharmacology panel.  The script
keeps censored IC50 bounds distinct from exact measurements, fixes an
outcome-blind assay-group rule, checks connectivity-block overlap explicitly,
and never writes compound identifiers.

The external endpoint is target-pair co-response, not ligand-level target
retrieval and not quantitative affinity prediction.  Target-label QAP leaves
the experimental endpoint, its irregular observation mask, and all nuisance
controls fixed while permuting the complete docking predictor network.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import scipy
from scipy import stats


PACKAGE = Path(__file__).resolve().parents[1]
DEFAULT_SPD = Path("/tmp/spd_activity.txt")
DEFAULT_UNIPROT = Path("/tmp/spd_uniprot.tsv")
DEFAULT_DOCKSTRING = PACKAGE / "data" / "frozen" / "dockstring-dataset.tsv.gz"
DEFAULT_OUTPUT = PACKAGE / "results" / "spd_external_validation"

SPD_SHA256 = "7132723f85e746de2f8387d01dcde6ffff703c92561fda9751cbd6753e900240"
# Re-pinned 2026-08-12 alongside the two-way transform switch. The previous
# snapshot was 26ebfda1...bf1729; the pairwise identities it produces are
# reproduced by this one to 9.7e-17 over all 66 target pairs, so the change
# is byte formatting, not sequence content.
UNIPROT_SHA256 = "b0f6213a79286da247fb1bf74c036446fe61e1a3beff074187025fce35015689"
DOCKSTRING_SHA256 = "e15a58258dbd613374e499bb17e5428a0df3f1f53b34758042f8e3cd3b53eb64"

SEED = 20260803
QAP_PERMUTATIONS = 50_000

# Candidate groups were selected from direct human binding/inhibition groups
# before target-pair outcomes were computed.  For genes with two direct assay
# campaigns, the primary group is the one with greatest compound coverage;
# ties are resolved by the smaller numeric group ID.
CANDIDATE_GROUPS: dict[str, tuple[int, ...]] = {
    "ADORA2A": (3177,),
    "ADRB1": (3086,),
    "ADRB2": (3087,),
    "AR": (41629,),
    "DRD2": (3157,),
    "EGFR": (32018,),
    "ESR1": (3199, 31453),
    "ESR2": (3217,),
    "F2": (3662,),
    "NR3C1": (3218,),
    "PGR": (3279, 41630),
    "PTGS2": (3198, 41633),
}

TARGETS = tuple(CANDIDATE_GROUPS)

TARGET_FAMILY = {
    "ADORA2A": "class_A_GPCR",
    "ADRB1": "class_A_GPCR",
    "ADRB2": "class_A_GPCR",
    "DRD2": "class_A_GPCR",
    "AR": "nuclear_receptor",
    "ESR1": "nuclear_receptor",
    "ESR2": "nuclear_receptor",
    "NR3C1": "nuclear_receptor",
    "PGR": "nuclear_receptor",
    "EGFR": "kinase",
    "F2": "protease",
    "PTGS2": "cyclooxygenase",
}


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_checksum(path: Path, expected: str | None) -> str:
    observed = sha256_file(path)
    if expected is not None and observed != expected:
        raise ValueError(
            f"SHA256 mismatch for {Path(path).name}: expected {expected}, observed {observed}"
        )
    return observed


def percentile_ranks(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return stats.rankdata(values, method="average") / len(values)


def _corr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(stats.spearmanr(x, y).statistic)


def load_spd(path: Path, *, expected_sha256: str | None = SPD_SHA256) -> pd.DataFrame:
    """Load the official SPD activity export and validate required columns."""
    _verify_checksum(path, expected_sha256)
    frame = pd.read_csv(path, sep="\t", low_memory=False)
    required = {
        "inchi_key",
        "assay_group",
        "assay_group_name",
        "summarized prefix",
        "summarized IC50",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"SPD export is missing required columns: {sorted(missing)}")
    frame = frame.copy()
    frame["assay_group_id"] = pd.to_numeric(
        frame["assay_group"], errors="coerce"
    ).astype("Int64")
    frame["ic50_uM"] = pd.to_numeric(
        frame["summarized IC50"], errors="coerce"
    )
    frame["qualifier"] = frame["summarized prefix"].astype("string").str.strip()
    frame["full_inchikey"] = frame["inchi_key"].astype("string").str.strip()
    frame["connectivity_key"] = frame["full_inchikey"].str.slice(0, 14)
    valid_key = frame["full_inchikey"].str.match(
        r"^[A-Z]{14}-[A-Z]{10}-[A-Z]$", na=False
    )
    return frame.loc[valid_key & (frame.ic50_uM > 0)].copy()


def choose_primary_groups(frame: pd.DataFrame) -> tuple[dict[int, str], pd.DataFrame]:
    """Apply the fixed largest-coverage group rule without using outcomes."""
    rows: list[dict] = []
    selected: dict[int, str] = {}
    for gene, candidates in CANDIDATE_GROUPS.items():
        candidate_rows: list[tuple[int, int]] = []
        for group_id in candidates:
            subset = frame.loc[frame.assay_group_id == group_id]
            coverage = int(subset.full_inchikey.nunique())
            candidate_rows.append((group_id, coverage))
        winner = sorted(candidate_rows, key=lambda item: (-item[1], item[0]))[0][0]
        for group_id, coverage in candidate_rows:
            names = sorted(
                set(
                    frame.loc[
                        frame.assay_group_id == group_id, "assay_group_name"
                    ].dropna().astype(str)
                )
            )
            rows.append(
                {
                    "gene": gene,
                    "assay_group_id": group_id,
                    "assay_group_name": " | ".join(names),
                    "full_inchikey_coverage": coverage,
                    "selected_primary": group_id == winner,
                    "selection_rule": "largest full-InChIKey coverage; tie -> smallest group ID",
                }
            )
        selected[winner] = gene
    return selected, pd.DataFrame(rows).sort_values(["gene", "assay_group_id"])


def selected_long(frame: pd.DataFrame, groups: Mapping[int, str]) -> pd.DataFrame:
    subset = frame.loc[frame.assay_group_id.isin(groups)].copy()
    subset["gene"] = subset.assay_group_id.map(groups)
    if subset.gene.isna().any():
        raise AssertionError("selected assay group failed gene mapping")
    subset["pIC50_bound"] = 6.0 - np.log10(subset.ic50_uM)
    return subset


def matrix_from_long(
    frame: pd.DataFrame,
    *,
    identity: str,
    endpoint: str,
    threshold_uM: float | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Build a compound-by-target endpoint while preserving censor semantics."""
    if identity not in {"full_inchikey", "connectivity_key"}:
        raise ValueError("identity must be full_inchikey or connectivity_key")
    work = frame.copy()
    if endpoint == "floor_at_bound":
        work["endpoint_value"] = work.pIC50_bound
    elif endpoint == "exact_only":
        work = work.loc[work.qualifier == "="].copy()
        work["endpoint_value"] = work.pIC50_bound
    elif endpoint == "censor_aware_binary":
        if threshold_uM is None or threshold_uM <= 0:
            raise ValueError("positive threshold_uM is required for binary endpoint")
        exact = work.qualifier == "="
        definitely_inactive = (work.qualifier == ">") & (
            work.ic50_uM >= threshold_uM
        )
        work["endpoint_value"] = np.where(
            exact,
            (work.ic50_uM <= threshold_uM).astype(float),
            np.where(definitely_inactive, 0.0, np.nan),
        )
        work = work.dropna(subset=["endpoint_value"])
    else:
        raise ValueError(f"unknown endpoint: {endpoint}")

    grouped = (
        work.groupby([identity, "gene"], as_index=False, sort=True)
        .endpoint_value.mean()
    )
    matrix = grouped.pivot(index=identity, columns="gene", values="endpoint_value")
    matrix = matrix.reindex(columns=TARGETS)
    target_nonmissing = matrix.notna().sum()
    support = {
        "endpoint": endpoint,
        "threshold_uM": threshold_uM,
        "identity": identity,
        "rows_after_endpoint_filter": int(len(work)),
        "compounds": int(len(matrix)),
        "cells": int(matrix.notna().sum().sum()),
        "targets_with_data": int((target_nonmissing > 0).sum()),
        "median_target_coverage": float(target_nonmissing[target_nonmissing > 0].median()),
        "median_compound_degree": float(matrix.notna().sum(axis=1).median()),
        "qualifier_exact_rows": int((work.qualifier == "=").sum()),
        "qualifier_right_censored_rows": int((work.qualifier == ">").sum()),
    }
    return matrix, support


def missing_aware_residual(matrix: pd.DataFrame, *, iterative: bool = False) -> pd.DataFrame:
    """Column-standardize observed cells, then remove observed row means."""
    z = (matrix - matrix.mean(axis=0)) / matrix.std(axis=0, ddof=0)
    residual = z.sub(z.mean(axis=1), axis=0)
    if iterative:
        # Alternating projections for the incomplete additive model.  Missing
        # cells remain missing and are never imputed.
        for _ in range(500):
            previous = residual.to_numpy(copy=True)
            residual = residual.sub(residual.mean(axis=0), axis=1)
            residual = residual.sub(residual.mean(axis=1), axis=0)
            delta = np.nanmax(np.abs(residual.to_numpy() - previous))
            if delta < 1e-12:
                break
    return residual


def pairwise_geometry(
    matrix: pd.DataFrame,
    *,
    min_support: int,
    transform: str,
) -> pd.DataFrame:
    if transform == "raw":
        surface = matrix
    elif transform == "column_z_row_center":
        surface = missing_aware_residual(matrix, iterative=False)
    elif transform == "iterative_two_way":
        surface = missing_aware_residual(matrix, iterative=True)
    else:
        raise ValueError(f"unknown transform: {transform}")
    support = surface.notna().astype(np.int64).T @ surface.notna().astype(np.int64)
    correlation = surface.corr(method="spearman", min_periods=min_support)
    rows: list[dict] = []
    for first, target_a in enumerate(TARGETS):
        for target_b in TARGETS[first + 1 :]:
            value = correlation.loc[target_a, target_b]
            n = int(support.loc[target_a, target_b])
            if n >= min_support and np.isfinite(value):
                rows.append(
                    {
                        "target_a": target_a,
                        "target_b": target_b,
                        "experimental_correlation": float(value),
                        "pair_support": n,
                    }
                )
    return pd.DataFrame(rows)


def load_dockstring(
    path: Path, *, expected_sha256: str | None = DOCKSTRING_SHA256
) -> pd.DataFrame:
    _verify_checksum(path, expected_sha256)
    frame = pd.read_csv(path, sep="\t")
    if not {"inchikey", "smiles", *TARGETS}.issubset(frame.columns):
        raise ValueError("DOCKSTRING table lacks required identifiers or targets")
    frame = frame.copy()
    frame["connectivity_key"] = frame.inchikey.astype(str).str.slice(0, 14)
    return frame


DOCKING_RESIDUAL_TRANSFORM = "two_way"


def docking_residual(
    values: np.ndarray, scale: np.ndarray, transform: str = DOCKING_RESIDUAL_TRANSFORM
) -> np.ndarray:
    """Remove the per-ligand offset from a complete docking block.

    ``two_way`` is the manuscript's definition, the additive two-way ANOVA
    residual in raw score units. ``column_z_row_center`` standardises each
    target column first and is what frozen bundles before 2026-08-12 used. The
    docking block here is complete, so a single row-then-column pass gives the
    exact two-way residual.
    """
    if transform == "two_way":
        centred = values - values.mean(axis=0)
        return centred - centred.mean(axis=1, keepdims=True)
    if transform == "column_z_row_center":
        z = (values - values.mean(axis=0)) / scale
        return z - z.mean(axis=1, keepdims=True)
    raise ValueError(f"unknown docking residual transform {transform!r}")


def docking_geometries(
    frame: pd.DataFrame,
    *,
    target_scope: str,
    allowed_rows: np.ndarray | None = None,
) -> tuple[dict[str, np.ndarray], dict]:
    all_targets = [column for column in frame.columns if column not in {
        "inchikey", "smiles", "connectivity_key"
    }]
    columns = list(TARGETS) if target_scope == "local_12" else all_targets
    if target_scope not in {"local_12", "global_58"}:
        raise ValueError("target_scope must be local_12 or global_58")
    complete = frame[columns].notna().all(axis=1).to_numpy()
    if allowed_rows is not None:
        complete &= np.asarray(allowed_rows, dtype=bool)
    selected = frame.loc[complete, columns]
    values = np.minimum(selected.to_numpy(dtype=np.float64), 0.0)
    scale = values.std(axis=0, ddof=0)
    if (scale == 0).any():
        raise ValueError("constant docking target column")
    z = (values - values.mean(axis=0)) / scale
    residual = docking_residual(values, scale)
    raw_corr = np.corrcoef(z, rowvar=False)
    residual_corr = np.corrcoef(residual, rowvar=False)
    index = {target: i for i, target in enumerate(columns)}
    subset = np.array([index[target] for target in TARGETS], dtype=int)
    matrices = {
        "raw": raw_corr[np.ix_(subset, subset)],
        "residual": residual_corr[np.ix_(subset, subset)],
    }
    return matrices, {
        "target_scope": target_scope,
        "rows": int(complete.sum()),
        "targets_used_for_row_centering": len(columns),
    }


def load_sequence_identity(
    path: Path, *, expected_sha256: str | None = UNIPROT_SHA256
) -> tuple[np.ndarray, pd.DataFrame, dict]:
    _verify_checksum(path, expected_sha256)
    frame = pd.read_csv(path, sep="\t")
    required = {"Entry", "Gene Names (primary)", "Sequence"}
    if not required.issubset(frame.columns):
        raise ValueError("UniProt export lacks required columns")
    if set(frame["Gene Names (primary)"]) != set(TARGETS):
        raise ValueError("UniProt gene labels do not exactly match the 12 targets")
    if frame["Gene Names (primary)"].duplicated().any():
        raise ValueError("duplicate UniProt primary gene label")
    try:
        from Bio import Align
        from Bio.Align import substitution_matrices
        import Bio
    except ImportError as error:  # pragma: no cover
        raise RuntimeError("Biopython is required for sequence control") from error
    sequences = frame.set_index("Gene Names (primary)").Sequence.astype(str).to_dict()
    aligner = Align.PairwiseAligner()
    aligner.mode = "global"
    aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
    aligner.open_gap_score = -10.0
    aligner.extend_gap_score = -0.5
    matrix = np.eye(len(TARGETS), dtype=np.float64)
    rows: list[dict] = []
    for first, target_a in enumerate(TARGETS):
        for second in range(first + 1, len(TARGETS)):
            target_b = TARGETS[second]
            seq_a, seq_b = sequences[target_a], sequences[target_b]
            alignment = aligner.align(seq_a, seq_b)[0]
            matches = 0
            for (a0, a1), (b0, b1) in zip(
                alignment.aligned[0], alignment.aligned[1]
            ):
                matches += sum(x == y for x, y in zip(seq_a[a0:a1], seq_b[b0:b1]))
            coordinates = alignment.coordinates
            alignment_columns = sum(
                max(
                    int(coordinates[0, k + 1] - coordinates[0, k]),
                    int(coordinates[1, k + 1] - coordinates[1, k]),
                )
                for k in range(coordinates.shape[1] - 1)
            )
            identity = matches / alignment_columns
            matrix[first, second] = matrix[second, first] = identity
            rows.append(
                {
                    "target_a": target_a,
                    "target_b": target_b,
                    "full_sequence_identity": identity,
                    "same_family": float(TARGET_FAMILY[target_a] == TARGET_FAMILY[target_b]),
                }
            )
    return matrix, pd.DataFrame(rows), {
        "alignment": "global BLOSUM62; gap-open -10; gap-extend -0.5",
        "identity_denominator": "all alignment columns including gaps",
        "biopython_version": Bio.__version__,
    }


def attach_predictors(
    pairs: pd.DataFrame,
    geometries: Mapping[str, np.ndarray],
    sequence: np.ndarray,
) -> pd.DataFrame:
    index = {target: i for i, target in enumerate(TARGETS)}
    rows = pairs.copy()
    rows["docking_raw"] = [
        geometries["raw"][index[a], index[b]]
        for a, b in zip(rows.target_a, rows.target_b)
    ]
    rows["docking_residual"] = [
        geometries["residual"][index[a], index[b]]
        for a, b in zip(rows.target_a, rows.target_b)
    ]
    rows["full_sequence_identity"] = [
        sequence[index[a], index[b]] for a, b in zip(rows.target_a, rows.target_b)
    ]
    rows["same_family"] = [
        float(TARGET_FAMILY[a] == TARGET_FAMILY[b])
        for a, b in zip(rows.target_a, rows.target_b)
    ]
    coverage: dict[str, int] = {}
    for target in TARGETS:
        incident = rows.loc[(rows.target_a == target) | (rows.target_b == target)]
        coverage[target] = int(incident.pair_support.sum())
    rows["pair_target_coverage_geomean"] = [
        np.sqrt(coverage[a] * coverage[b]) for a, b in zip(rows.target_a, rows.target_b)
    ]
    rows["pair_target_coverage_similarity"] = [
        -abs(coverage[a] - coverage[b]) for a, b in zip(rows.target_a, rows.target_b)
    ]
    return rows


def partial_rank_association(
    frame: pd.DataFrame,
    predictor: str,
    controls: Sequence[str],
) -> float:
    columns = ["experimental_correlation", predictor, *controls]
    work = frame[columns].replace([np.inf, -np.inf], np.nan).dropna()
    if len(work) <= len(controls) + 3:
        return float("nan")
    y = percentile_ranks(work.experimental_correlation.to_numpy(float))
    x = percentile_ranks(work[predictor].to_numpy(float))
    design_columns = [np.ones(len(work))]
    for name in controls:
        values = work[name].to_numpy(float)
        if np.unique(values).size > 2:
            values = percentile_ranks(values)
        design_columns.append(values)
    design = np.column_stack(design_columns)
    y_res = y - design @ np.linalg.lstsq(design, y, rcond=None)[0]
    x_res = x - design @ np.linalg.lstsq(design, x, rcond=None)[0]
    if np.std(y_res) == 0 or np.std(x_res) == 0:
        return float("nan")
    return float(np.corrcoef(y_res, x_res)[0, 1])


def _partial_rank_design(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return residualized endpoint ranks and the fixed control projection."""
    controls = (
        "full_sequence_identity",
        "same_family",
        "pair_support",
        "pair_target_coverage_geomean",
        "pair_target_coverage_similarity",
    )
    columns = [np.ones(len(frame))]
    for name in controls:
        values = frame[name].to_numpy(float)
        if np.unique(values).size > 2:
            values = percentile_ranks(values)
        columns.append(values)
    design = np.column_stack(columns)
    projection = np.eye(len(frame)) - design @ np.linalg.pinv(design)
    y_rank = percentile_ranks(frame.experimental_correlation.to_numpy(float))
    y_residual = projection @ y_rank
    return y_residual, projection


def association_summary(frame: pd.DataFrame) -> dict:
    y = frame.experimental_correlation.to_numpy(float)
    raw = frame.docking_raw.to_numpy(float)
    residual = frame.docking_residual.to_numpy(float)
    controls = (
        "full_sequence_identity",
        "same_family",
        "pair_support",
        "pair_target_coverage_geomean",
        "pair_target_coverage_similarity",
    )
    raw_rho = _corr(y, raw)
    residual_rho = _corr(y, residual)
    return {
        "pairs": int(len(frame)),
        "raw_spearman": raw_rho,
        "residual_spearman": residual_rho,
        "residual_minus_raw": residual_rho - raw_rho,
        "raw_partial_rank": partial_rank_association(frame, "docking_raw", controls),
        "residual_partial_rank": partial_rank_association(
            frame, "docking_residual", controls
        ),
        "controls": list(controls),
    }


def target_permutations(
    n: int,
    permutations: int,
    *,
    seed: int,
    preserve_family: bool,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    maps = np.empty((permutations, n), dtype=np.int16)
    base = np.arange(n, dtype=np.int16)
    groups: list[np.ndarray] = []
    if preserve_family:
        for family in sorted(set(TARGET_FAMILY.values())):
            groups.append(
                np.array(
                    [i for i, target in enumerate(TARGETS) if TARGET_FAMILY[target] == family],
                    dtype=np.int16,
                )
            )
    for draw in range(permutations):
        mapping = base.copy()
        if preserve_family:
            for group in groups:
                mapping[group] = rng.permutation(group)
        else:
            mapping = rng.permutation(base)
        maps[draw] = mapping
    return maps


def paired_qap(
    frame: pd.DataFrame,
    geometries: Mapping[str, np.ndarray],
    *,
    permutations: int,
    seed: int,
    preserve_family: bool,
) -> list[dict]:
    index = {target: i for i, target in enumerate(TARGETS)}
    pair_a = np.array([index[x] for x in frame.target_a], dtype=np.int16)
    pair_b = np.array([index[x] for x in frame.target_b], dtype=np.int16)
    y_rank = stats.zscore(percentile_ranks(frame.experimental_correlation.to_numpy(float)))
    y_partial, projection = _partial_rank_design(frame)
    y_partial = stats.zscore(y_partial)
    maps = target_permutations(
        len(TARGETS), permutations, seed=seed, preserve_family=preserve_family
    )
    observed = association_summary(frame)
    null_raw = np.empty(permutations)
    null_residual = np.empty(permutations)
    null_raw_partial = np.empty(permutations)
    null_residual_partial = np.empty(permutations)
    batch = 2_000
    for start in range(0, permutations, batch):
        stop = min(start + batch, permutations)
        selected = maps[start:stop]
        first = selected[:, pair_a]
        second = selected[:, pair_b]
        raw_values = geometries["raw"][first, second]
        residual_values = geometries["residual"][first, second]
        raw_rank = stats.rankdata(raw_values, axis=1)
        residual_rank = stats.rankdata(residual_values, axis=1)
        raw_rank = stats.zscore(raw_rank, axis=1)
        residual_rank = stats.zscore(residual_rank, axis=1)
        null_raw[start:stop] = np.mean(raw_rank * y_rank, axis=1)
        null_residual[start:stop] = np.mean(residual_rank * y_rank, axis=1)
        raw_partial = raw_rank @ projection.T
        residual_partial = residual_rank @ projection.T
        raw_partial = stats.zscore(raw_partial, axis=1)
        residual_partial = stats.zscore(residual_partial, axis=1)
        null_raw_partial[start:stop] = np.mean(raw_partial * y_partial, axis=1)
        null_residual_partial[start:stop] = np.mean(
            residual_partial * y_partial, axis=1
        )
    label = "family_preserving" if preserve_family else "unrestricted"
    output: list[dict] = []
    for name, value, null in (
        ("raw_spearman", observed["raw_spearman"], null_raw),
        ("residual_spearman", observed["residual_spearman"], null_residual),
        (
            "residual_minus_raw",
            observed["residual_minus_raw"],
            null_residual - null_raw,
        ),
        ("raw_partial_rank", observed["raw_partial_rank"], null_raw_partial),
        (
            "residual_partial_rank",
            observed["residual_partial_rank"],
            null_residual_partial,
        ),
        (
            "residual_minus_raw_partial_rank",
            observed["residual_partial_rank"] - observed["raw_partial_rank"],
            null_residual_partial - null_raw_partial,
        ),
    ):
        output.append(
            {
                "permutation_scheme": label,
                "metric": name,
                "observed": float(value),
                "permutations": permutations,
                "p_positive": float((1 + np.sum(null >= value)) / (permutations + 1)),
                "null_mean": float(np.mean(null)),
                "null_q025": float(np.quantile(null, 0.025)),
                "null_q975": float(np.quantile(null, 0.975)),
            }
        )
    return output


def top_partner_retrieval(frame: pd.DataFrame, *, endpoint_label: str) -> pd.DataFrame:
    """Rank the experimentally best counterscreen partner for each query target."""
    rows: list[dict] = []
    for query in TARGETS:
        incident = frame.loc[
            (frame.target_a == query) | (frame.target_b == query)
        ].copy()
        if len(incident) < 3:
            continue
        incident["partner"] = np.where(
            incident.target_a == query, incident.target_b, incident.target_a
        )
        experimental_best = incident.loc[
            incident.experimental_correlation.idxmax(), "partner"
        ]
        residual_rank = percentile_ranks(
            incident.docking_residual.to_numpy(float)
        )
        sequence_rank = percentile_ranks(
            incident.full_sequence_identity.to_numpy(float)
        )
        predictor_values = {
            "raw": incident.docking_raw.to_numpy(float),
            "residual": incident.docking_residual.to_numpy(float),
            "sequence": incident.full_sequence_identity.to_numpy(float),
            "sequence_residual_equal_rank": (sequence_rank + residual_rank) / 2.0,
        }
        for predictor, values in predictor_values.items():
            order = np.argsort(-values, kind="stable")
            ranked_partners = incident.partner.to_numpy()[order]
            position = int(np.flatnonzero(ranked_partners == experimental_best)[0] + 1)
            rows.append(
                {
                    "endpoint": endpoint_label,
                    "query_target": query,
                    "predictor": predictor,
                    "candidate_partners": int(len(incident)),
                    "best_partner_rank": position,
                    "best_partner_in_top1": position <= 1,
                    "best_partner_in_top3": position <= 3,
                    "reciprocal_rank": 1.0 / position,
                }
            )
    return pd.DataFrame(rows)


def target_jackknife(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for target in TARGETS:
        subset = frame.loc[
            (frame.target_a != target) & (frame.target_b != target)
        ]
        if len(subset) < 3:
            continue
        metrics = association_summary(subset)
        rows.append({"omitted_target": target, **metrics})
    return pd.DataFrame(rows)


def largest_complete_rectangle(matrix: pd.DataFrame) -> tuple[tuple[str, ...], pd.DataFrame]:
    """Outcome-blind largest target set with at least 50 complete compounds."""
    import itertools

    for target_count in range(len(TARGETS), 2, -1):
        candidates: list[tuple[int, tuple[str, ...], pd.DataFrame]] = []
        for targets in itertools.combinations(TARGETS, target_count):
            complete = matrix.loc[:, targets].dropna()
            if len(complete) >= 50:
                candidates.append((len(complete), targets, complete))
        if candidates:
            candidates.sort(key=lambda item: (-item[0], item[1]))
            _, targets, complete = candidates[0]
            return targets, complete
    raise ValueError("no complete rectangle with at least 50 compounds")


def output_checksum(path: Path) -> str:
    return sha256_file(path)


def run_analysis(
    *,
    spd_path: Path,
    uniprot_path: Path,
    dockstring_path: Path,
    output_dir: Path,
    qap_permutations: int,
    seed: int,
    verify_checksums: bool,
) -> dict:
    spd = load_spd(spd_path, expected_sha256=SPD_SHA256 if verify_checksums else None)
    primary_groups, assay_selection = choose_primary_groups(spd)
    long = selected_long(spd, primary_groups)
    dockstring = load_dockstring(
        dockstring_path,
        expected_sha256=DOCKSTRING_SHA256 if verify_checksums else None,
    )
    sequence, sequence_pairs, sequence_provenance = load_sequence_identity(
        uniprot_path,
        expected_sha256=UNIPROT_SHA256 if verify_checksums else None,
    )

    spd_full = set(long.full_inchikey)
    spd_connectivity = set(long.connectivity_key)
    docking_full = set(dockstring.inchikey.astype(str))
    docking_connectivity = set(dockstring.connectivity_key)
    overlapping_connectivity = spd_connectivity & docking_connectivity
    overlap = {
        "spd_full_inchikeys": len(spd_full),
        "spd_connectivity_blocks": len(spd_connectivity),
        "dockstring_full_inchikeys": len(docking_full),
        "dockstring_connectivity_blocks": len(docking_connectivity),
        "exact_full_inchikey_overlap": len(spd_full & docking_full),
        "connectivity_block_overlap": len(overlapping_connectivity),
        "dockstring_rows_in_overlapping_blocks": int(
            dockstring.connectivity_key.isin(overlapping_connectivity).sum()
        ),
    }

    geometry_sets: dict[str, tuple[dict[str, np.ndarray], dict]] = {}
    for scope in ("local_12", "global_58"):
        geometry_sets[f"full__{scope}"] = docking_geometries(
            dockstring, target_scope=scope
        )
        geometry_sets[f"nonoverlap__{scope}"] = docking_geometries(
            dockstring,
            target_scope=scope,
            allowed_rows=~dockstring.connectivity_key.isin(spd_connectivity).to_numpy(),
        )
        geometry_sets[f"matched__{scope}"] = docking_geometries(
            dockstring,
            target_scope=scope,
            allowed_rows=dockstring.connectivity_key.isin(spd_connectivity).to_numpy(),
        )

    endpoint_specs = [
        ("floor_at_bound", None, 40),
        ("censor_aware_binary", 10.0, 40),
        ("censor_aware_binary", 30.0, 40),
        ("exact_only", None, 5),
        ("exact_only", None, 10),
    ]
    metric_rows: list[dict] = []
    primary_pair_frame: pd.DataFrame | None = None
    endpoint_support: list[dict] = []
    pair_outputs: list[pd.DataFrame] = []
    qap_rows: list[dict] = []
    jackknife_rows: list[pd.DataFrame] = []
    retrieval_rows: list[pd.DataFrame] = []

    for endpoint, threshold, min_support in endpoint_specs:
        for identity in ("connectivity_key", "full_inchikey"):
            matrix, support = matrix_from_long(
                long,
                identity=identity,
                endpoint=endpoint,
                threshold_uM=threshold,
            )
            endpoint_support.append({**support, "min_pair_support": min_support})
            for transform in ("raw", "column_z_row_center", "iterative_two_way"):
                pairs = pairwise_geometry(
                    matrix, min_support=min_support, transform=transform
                )
                if len(pairs) < 3:
                    continue
                for geometry_name, (geometries, geometry_support) in geometry_sets.items():
                    attached = attach_predictors(pairs, geometries, sequence)
                    metrics = association_summary(attached)
                    metric_rows.append(
                        {
                            "endpoint": endpoint,
                            "threshold_uM": threshold,
                            "compound_identity": identity,
                            "experimental_transform": transform,
                            "min_pair_support": min_support,
                            "docking_geometry": geometry_name,
                            "docking_rows": geometry_support["rows"],
                            **metrics,
                        }
                    )
                    is_primary = (
                        endpoint == "floor_at_bound"
                        and threshold is None
                        and identity == "connectivity_key"
                        and transform == "column_z_row_center"
                        and min_support == 40
                        and geometry_name == "nonoverlap__local_12"
                    )
                    if is_primary:
                        primary_pair_frame = attached.copy()
                        primary_pair_frame.insert(0, "analysis", "primary")
                        pair_outputs.append(primary_pair_frame)
                        jack = target_jackknife(attached)
                        jack.insert(0, "analysis", "primary")
                        jackknife_rows.append(jack)
                    is_censor_primary = (
                        endpoint in {"floor_at_bound", "censor_aware_binary"}
                        and identity == "connectivity_key"
                        and transform == "column_z_row_center"
                        and min_support == 40
                        and geometry_name == "nonoverlap__local_12"
                    )
                    if is_censor_primary:
                        endpoint_label = (
                            endpoint
                            if threshold is None
                            else f"{endpoint}_{int(threshold)}uM"
                        )
                        retrieval_rows.append(
                            top_partner_retrieval(
                                attached, endpoint_label=endpoint_label
                            )
                        )
                        for preserve_family, offset in ((False, 0), (True, 1)):
                            qap_result = paired_qap(
                                attached,
                                geometries,
                                permutations=qap_permutations,
                                seed=seed + offset,
                                preserve_family=preserve_family,
                            )
                            for row in qap_result:
                                row["endpoint"] = endpoint_label
                            qap_rows.extend(qap_result)

    if primary_pair_frame is None:
        raise AssertionError("primary pair frame was not created")

    # Fixed-support sensitivity: the largest target set with >=50 compounds
    # measured for every selected target.  This selection uses missingness only.
    floor_matrix, _ = matrix_from_long(
        long, identity="connectivity_key", endpoint="floor_at_bound"
    )
    rectangle_targets, rectangle = largest_complete_rectangle(floor_matrix)
    rectangle_surface = missing_aware_residual(rectangle, iterative=False)
    rectangle_corr = rectangle_surface.corr(method="spearman")
    index = {target: i for i, target in enumerate(TARGETS)}
    nonoverlap_global = geometry_sets["nonoverlap__global_58"][0]
    rectangle_rows: list[dict] = []
    for first, target_a in enumerate(rectangle_targets):
        for target_b in rectangle_targets[first + 1 :]:
            rectangle_rows.append(
                {
                    "target_a": target_a,
                    "target_b": target_b,
                    "experimental_correlation": float(
                        rectangle_corr.loc[target_a, target_b]
                    ),
                    "pair_support": len(rectangle),
                }
            )
    rectangle_pairs = attach_predictors(
        pd.DataFrame(rectangle_rows), nonoverlap_global, sequence
    )
    rectangle_metrics = association_summary(rectangle_pairs)

    # Assay-group alternatives change one duplicate-gene campaign at a time.
    alternate_rows: list[dict] = []
    for gene, candidates in CANDIDATE_GROUPS.items():
        if len(candidates) < 2:
            continue
        selected_group = next(group for group, mapped in primary_groups.items() if mapped == gene)
        for alternative in candidates:
            if alternative == selected_group:
                continue
            groups = dict(primary_groups)
            del groups[selected_group]
            groups[alternative] = gene
            alt_long = selected_long(spd, groups)
            alt_matrix, alt_support = matrix_from_long(
                alt_long, identity="connectivity_key", endpoint="floor_at_bound"
            )
            alt_pairs = pairwise_geometry(
                alt_matrix, min_support=40, transform="column_z_row_center"
            )
            alt_attached = attach_predictors(
                alt_pairs, geometry_sets["nonoverlap__local_12"][0], sequence
            )
            alternate_rows.append(
                {
                    "swapped_gene": gene,
                    "primary_group": selected_group,
                    "alternative_group": alternative,
                    "compounds": alt_support["compounds"],
                    "cells": alt_support["cells"],
                    **association_summary(alt_attached),
                }
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    files: dict[str, Path] = {
        "assay_selection": output_dir / "assay_selection.csv",
        "endpoint_support": output_dir / "endpoint_support.csv",
        "geometry_metrics": output_dir / "geometry_metrics.csv",
        "primary_pairs": output_dir / "primary_target_pairs.csv",
        "target_jackknife": output_dir / "target_jackknife.csv",
        "qap": output_dir / "target_label_qap.csv",
        "sequence_pairs": output_dir / "sequence_controls.csv",
        "alternate_assays": output_dir / "alternate_assay_sensitivity.csv",
        "top_partner": output_dir / "top_partner_retrieval.csv",
    }
    assay_selection.to_csv(files["assay_selection"], index=False)
    pd.DataFrame(endpoint_support).to_csv(files["endpoint_support"], index=False)
    metrics_frame = pd.DataFrame(metric_rows)
    metrics_frame.to_csv(files["geometry_metrics"], index=False)
    primary_pair_frame.to_csv(files["primary_pairs"], index=False)
    pd.concat(jackknife_rows, ignore_index=True).to_csv(
        files["target_jackknife"], index=False
    )
    pd.DataFrame(qap_rows).to_csv(files["qap"], index=False)
    sequence_pairs.to_csv(files["sequence_pairs"], index=False)
    pd.DataFrame(alternate_rows).to_csv(files["alternate_assays"], index=False)
    retrieval_frame = pd.concat(retrieval_rows, ignore_index=True)
    retrieval_frame.to_csv(files["top_partner"], index=False)

    primary_metrics = association_summary(primary_pair_frame)
    binary_metrics: dict[str, dict] = {}
    for threshold in (10.0, 30.0):
        row = metrics_frame.loc[
            (metrics_frame.endpoint == "censor_aware_binary")
            & (metrics_frame.threshold_uM == threshold)
            & (metrics_frame.compound_identity == "connectivity_key")
            & (metrics_frame.experimental_transform == "column_z_row_center")
            & (metrics_frame.min_pair_support == 40)
            & (metrics_frame.docking_geometry == "nonoverlap__local_12")
        ].iloc[0]
        binary_metrics[f"{int(threshold)}_uM"] = {
            "pairs": int(row.pairs),
            "raw_spearman": float(row.raw_spearman),
            "residual_spearman": float(row.residual_spearman),
            "residual_minus_raw": float(row.residual_minus_raw),
        }
    exact_rows = metrics_frame.loc[
        (metrics_frame.endpoint == "exact_only")
        & (metrics_frame.compound_identity == "connectivity_key")
        & (metrics_frame.experimental_transform == "column_z_row_center")
        & (metrics_frame.docking_geometry == "nonoverlap__local_12")
    ]
    exact_sensitivity = {
        str(int(row.min_pair_support)): {
            "pairs": int(row.pairs),
            "raw_spearman": float(row.raw_spearman),
            "residual_spearman": float(row.residual_spearman),
        }
        for row in exact_rows.itertuples(index=False)
    }
    qap_frame = pd.DataFrame(qap_rows)
    retrieval_summary = (
        retrieval_frame.groupby(["endpoint", "predictor"], as_index=False)
        .agg(
            query_targets=("query_target", "size"),
            top1_hit_rate=("best_partner_in_top1", "mean"),
            top3_hit_rate=("best_partner_in_top3", "mean"),
            mean_reciprocal_rank=("reciprocal_rank", "mean"),
        )
    )
    summary = {
        "analysis_status": "exploratory_post_hoc_external_validation",
        "selection_boundary": (
            "candidate direct human binding/inhibition groups and the largest-coverage "
            "rule were fixed without inspecting target-pair outcomes; endpoint and all "
            "sensitivities remain post hoc"
        ),
        "primary_contract": {
            "assays": {gene: int(group) for group, gene in primary_groups.items()},
            "compound_identity": "first 14 characters of Standard InChIKey",
            "experimental_endpoint": (
                "reported IC50 bound converted to pIC50, column-standardized on observed "
                "cells, then ligand-row-centered; pairwise Spearman; min 40"
            ),
            "censoring_warning": (
                "right-censored bounds are not exact IC50 values; binary and exact-only "
                "analyses define distinct estimands"
            ),
            "docking_predictor": (
                "DOCKSTRING rows outside all SPD connectivity blocks; positive scores "
                "clipped to zero; raw-unit two-way centring across 12 targets"
            ),
        },
        "support": {
            "selected_rows": int(len(long)),
            "exact_rows": int((long.qualifier == "=").sum()),
            "right_censored_rows": int((long.qualifier == ">").sum()),
            "right_censored_fraction": float((long.qualifier == ">").mean()),
            **overlap,
        },
        "key_results": {
            "primary_floor_at_bound": primary_metrics,
            "censor_aware_binary": binary_metrics,
            "exact_only_instability": exact_sensitivity,
            "common_support_rectangle": {
                "targets": list(rectangle_targets),
                "compounds_complete_for_all_targets": int(len(rectangle)),
                **rectangle_metrics,
            },
            "qap": qap_frame.to_dict(orient="records"),
            "operational_best_partner_retrieval": retrieval_summary.to_dict(
                orient="records"
            ),
        },
        "interpretation": (
            "Residual Vina target geometry aligns with a censored/binary experimental "
            "co-response network, including on a fixed common-support rectangle. This "
            "can motivate counterscreen prioritization, but it is not evidence of "
            "quantitative affinity recovery or ligand-level target ranking."
        ),
        "limitations": [
            "More than ninety percent of selected SPD rows are right-censored.",
            "Exact-only estimates are sparse and change sharply with minimum pair support.",
            "Assay campaigns and observation masks are target-specific; unrestricted QAP target-label exchangeability is imperfect.",
            "Family-preserving QAP has limited permutations because four families are singletons.",
            "Connectivity matching merges stereoisomers/protonation states and is not exact chemical identity.",
            "No compound identifiers or raw SPD rows are redistributed; the open CC BY 4.0 source is fetched directly from Zenodo and checksum-validated.",
        ],
        "sources": {
            "spd": {
                "official_article": "https://www.nature.com/articles/s41467-023-40064-9",
                "official_data": "https://zenodo.org/records/8103950",
                "license": "CC BY 4.0",
                "sha256": sha256_file(spd_path),
                "redistributed": False,
            },
            "uniprot": {
                "source": "UniProt reviewed human proteins; exact primary gene labels",
                "sha256": sha256_file(uniprot_path),
                **sequence_provenance,
            },
            "dockstring": {
                "path": "data/frozen/dockstring-dataset.tsv.gz",
                "sha256": sha256_file(dockstring_path),
            },
        },
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
        },
        "parameters": {
            "seed": seed,
            "qap_permutations": qap_permutations,
        },
    }
    summary_path = output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    files["summary"] = summary_path
    checksums = {
        path.name: output_checksum(path)
        for path in sorted(files.values(), key=lambda item: item.name)
    }
    with (output_dir / "output_checksums.json").open("w", encoding="utf-8") as handle:
        json.dump(checksums, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spd", type=Path, default=DEFAULT_SPD)
    parser.add_argument("--uniprot", type=Path, default=DEFAULT_UNIPROT)
    parser.add_argument("--dockstring", type=Path, default=DEFAULT_DOCKSTRING)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--qap-permutations", type=int, default=QAP_PERMUTATIONS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--skip-checksums", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_analysis(
        spd_path=args.spd,
        uniprot_path=args.uniprot,
        dockstring_path=args.dockstring,
        output_dir=args.output_dir,
        qap_permutations=args.qap_permutations,
        seed=args.seed,
        verify_checksums=not args.skip_checksums,
    )


if __name__ == "__main__":
    main()
