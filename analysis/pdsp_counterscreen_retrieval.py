#!/usr/bin/env python3
"""Post-hoc PDSP test of residual Vina geometry for counterscreen selection.

The decision represented here is graph-level, not ligand-level: given one
target in the fixed Docking-44 panel, which other assayed targets should be
prioritized as counterscreens because their experimental affinity profiles are
likely to co-vary?  Predictors are calculated without using PDSP outcomes.

The primary endpoint is the Spearman correlation of exact human PDSP pKi
profiles for every target pair with at least ten jointly observed compounds.
The analysis is explicitly exploratory.  In particular, pairwise endpoints
are observed on different chemical supports; a PDSP-Certified-Data-only
sensitivity and fixed-common-support five-target blocks audit that limitation.

Raw PDSP rows are not redistributed.  The CLI consumes a user-downloaded CSV
and an official UniProt TSV snapshot and writes target-pair aggregates only.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import platform
from collections import OrderedDict
from pathlib import Path
from typing import Iterable, Sequence

import Bio
import numpy as np
import pandas as pd
import rdkit
import scipy
import sklearn
from Bio import Align
from rdkit import Chem, RDLogger
from scipy import stats
from sklearn.metrics import average_precision_score, roc_auc_score


PACKAGE = Path(__file__).resolve().parents[1]
DEFAULT_DOCKING = PACKAGE / "data" / "frozen" / "df_final_v4.csv.gz"
DEFAULT_OUTPUT = PACKAGE / "results" / "pdsp_counterscreen_retrieval"
DEFAULT_MIN_PAIR_SUPPORT = 10
DEFAULT_QAP_PERMUTATIONS = 20_000
DEFAULT_SEED = 20260803
PRIMARY_TOP_FRACTION = 0.10

PDSP_DOWNLOAD_URL = "https://pdsp.unc.edu/databases/kiDownload/download.php"
UNIPROT_REST_URL = "https://rest.uniprot.org/uniprotkb/search"

PDB_TO_GENE = OrderedDict(
    [
        ("1m2z", "NR3C1"), ("1pbq", "GRIN1"), ("1xoq", "PDE4D"),
        ("2rh1", "ADRB2"), ("2vt4", "ADRB1"), ("2ydo", "ADORA2A"),
        ("2z5x", "MAOA"), ("3b66", "AR"), ("3kk6", "PTGS1"),
        ("3ln1", "PTGS2"), ("3rze", "HRH1"), ("4djh", "OPRK1"),
        ("4ey7", "ACHE"), ("4iar", "HTR1B"), ("4mqs", "CHRM2"),
        ("4n6h", "OPRD1"), ("5cxv", "CHRM1"), ("5i71", "SLC6A4"),
        ("5tvn", "HTR2B"), ("5u09", "CNR1"), ("5va1", "KCNH2"),
        ("6cm4", "DRD2"), ("6kpf", "CNR2"), ("6kux", "ADRA2A"),
        ("6lqa", "SCN5A"), ("6pdj", "LCK"), ("6x3x", "GABRA1"),
        ("6y1z", "HTR3A"), ("7f8y", "CCKAR"), ("7kwe", "PDE3A"),
        ("7ljd", "DRD1"), ("7wc9", "HTR2A"), ("7xnk", "KCNQ1"),
        ("7ym8", "ADRA1A"), ("8e9y", "CHRM3"), ("8ef6", "OPRM1"),
        ("8fhs", "CACNA1C"), ("8pjk", "HTR1A"), ("8st0", "CHRNA4"),
        ("8wty", "SLC6A2"), ("8xvk", "EDNRA"), ("8yn3", "HRH2"),
        ("9eo4", "SLC6A3"), ("V1A", "AVPR1A"),
    ]
)

ALLOWED_HUMAN_SOURCES = frozenset(
    {"CLONED", "?CLONED", "RECOMBINANT", "CHO CELLS", "HEK293 CELLS",
     "C.H.O.", "SF9"}
)

CURATED_FAMILIES = {
    "serotonin_GPCR": ("HTR1A", "HTR1B", "HTR2A", "HTR2B"),
    "adrenergic_GPCR": ("ADRA1A", "ADRA2A", "ADRB1", "ADRB2"),
    "dopamine_GPCR": ("DRD1", "DRD2"),
    "histamine_GPCR": ("HRH1", "HRH2"),
    "muscarinic_GPCR": ("CHRM1", "CHRM2", "CHRM3"),
    "opioid_GPCR": ("OPRD1", "OPRK1", "OPRM1"),
    "cannabinoid_GPCR": ("CNR1", "CNR2"),
    "monoamine_transporter": ("SLC6A2", "SLC6A3", "SLC6A4"),
}

PREDICTORS = (
    "raw_docking",
    "residual_docking",
    "full_sequence_identity",
    "same_curated_family",
    "equal_rank_sequence_family",
    "equal_rank_sequence_residual",
    "equal_rank_sequence_family_residual",
)


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def standard_inchi_key(smiles: object) -> str | None:
    """Return a nonempty Standard InChIKey; reject blank/zero-atom records."""
    if pd.isna(smiles) or not str(smiles).strip():
        return None
    molecule = Chem.MolFromSmiles(str(smiles))
    if molecule is None or molecule.GetNumAtoms() == 0:
        return None
    key = Chem.MolToInchiKey(molecule)
    return key if key else None


def strict_pdsp_cells(path: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    """Apply the frozen exact-human Ki contract and aggregate by molecule/target."""
    RDLogger.DisableLog("rdApp.*")
    frame = pd.read_csv(path, encoding="latin1", low_memory=False)
    required = {
        "SMILES", "Unigene", "species", "source", "ki Note", "ki Val",
        "Reference",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"PDSP CSV is missing columns: {sorted(missing)}")

    target_genes = set(PDB_TO_GENE.values())
    selected = frame.loc[
        frame["species"].fillna("").astype(str).str.strip().str.upper().eq("HUMAN")
        & frame["source"].fillna("").astype(str).str.strip().str.upper().isin(
            ALLOWED_HUMAN_SOURCES
        )
        & frame["ki Note"].fillna("").astype(str).str.strip().isin(("", "="))
        & frame["Unigene"].fillna("").astype(str).str.strip().isin(target_genes)
    ].copy()
    selected["ki_nM"] = pd.to_numeric(selected["ki Val"], errors="coerce")
    selected = selected.loc[selected["ki_nM"].gt(0)].copy()
    selected["full_inchikey"] = [standard_inchi_key(value) for value in selected["SMILES"]]
    selected = selected.loc[selected["full_inchikey"].notna()].copy()
    selected["target"] = selected["Unigene"].astype(str).str.strip()
    selected["pKi"] = 9.0 - np.log10(selected["ki_nM"].to_numpy(dtype=float))

    def aggregate(subset: pd.DataFrame) -> pd.DataFrame:
        return (
            subset.groupby(["full_inchikey", "target"], as_index=False, sort=True)["pKi"]
            .median()
            .sort_values(["full_inchikey", "target"], kind="mergesort")
            .reset_index(drop=True)
        )

    cells = aggregate(selected)
    certified = aggregate(
        selected.loc[selected["Reference"].fillna("").astype(str).str.strip().eq(
            "PDSP Certified Data"
        )]
    )
    audit = {
        "downloaded_rows": int(len(frame)),
        "strict_source_rows_before_median_aggregation": int(len(selected)),
        "strict_compounds": int(cells["full_inchikey"].nunique()),
        "strict_targets": int(cells["target"].nunique()),
        "strict_compound_target_cells": int(len(cells)),
        "certified_compounds": int(certified["full_inchikey"].nunique()),
        "certified_targets": int(certified["target"].nunique()),
        "certified_compound_target_cells": int(len(certified)),
    }
    return cells, certified, audit


def censor_bound_pdsp_cells(
    path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, int]]:
    """Represent right-censored Ki rows at their reported pKi bound.

    This is not a censored-data estimator.  It is an intentionally simple
    sensitivity showing whether exact-only geometry survives when explicit
    non-binders are restored.  The binary matrices encode whether a cell has
    any exact measurement (1) rather than only a ``> Ki`` record (0).
    """
    RDLogger.DisableLog("rdApp.*")
    frame = pd.read_csv(path, encoding="latin1", low_memory=False)
    relation = frame["ki Note"].fillna("").astype(str).str.strip()
    selected = frame.loc[
        frame["species"].fillna("").astype(str).str.strip().str.upper().eq("HUMAN")
        & frame["source"].fillna("").astype(str).str.strip().str.upper().isin(
            ALLOWED_HUMAN_SOURCES
        )
        & relation.isin(("", "=", ">"))
        & frame["Unigene"].fillna("").astype(str).str.strip().isin(
            set(PDB_TO_GENE.values())
        )
    ].copy()
    selected["relation"] = selected["ki Note"].fillna("").astype(str).str.strip()
    selected["ki_nM"] = pd.to_numeric(selected["ki Val"], errors="coerce")
    selected = selected.loc[selected["ki_nM"].gt(0)].copy()
    selected["full_inchikey"] = [standard_inchi_key(value) for value in selected["SMILES"]]
    selected = selected.loc[selected["full_inchikey"].notna()].copy()
    selected["target"] = selected["Unigene"].astype(str).str.strip()
    selected["pKi_bound"] = 9.0 - np.log10(selected["ki_nM"].to_numpy(dtype=float))
    selected["quantified"] = selected["relation"].isin(("", "=")).astype(float)

    def aggregate(subset: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        grouped = subset.groupby(["full_inchikey", "target"], as_index=False, sort=True)
        bound = grouped["pKi_bound"].median().rename(columns={"pKi_bound": "pKi"})
        quantified = grouped["quantified"].max().rename(columns={"quantified": "pKi"})
        return bound, quantified

    bound, binary = aggregate(selected)
    certified_source = selected.loc[
        selected["Reference"].fillna("").astype(str).str.strip().eq("PDSP Certified Data")
    ]
    certified_bound, certified_binary = aggregate(certified_source)
    audit = {
        "source_rows": int(len(selected)),
        "right_censored_source_rows": int(selected["relation"].eq(">").sum()),
        "certified_source_rows": int(len(certified_source)),
        "certified_right_censored_source_rows": int(
            certified_source["relation"].eq(">").sum()
        ),
        "bound_compound_target_cells": int(len(bound)),
        "certified_bound_compound_target_cells": int(len(certified_bound)),
    }
    return bound, binary, certified_bound, certified_binary, audit


def cell_matrix(cells: pd.DataFrame) -> pd.DataFrame:
    return cells.pivot(index="full_inchikey", columns="target", values="pKi").sort_index(
        axis=0
    ).sort_index(axis=1)


def docking_geometries(path: Path) -> tuple[list[str], dict[str, np.ndarray]]:
    pdb_columns = list(PDB_TO_GENE)
    scores = pd.read_csv(path, usecols=pdb_columns)[pdb_columns].to_numpy(dtype=float)
    scores = np.minimum(scores, 0.0)
    means = np.nanmean(scores, axis=0)
    if not np.isfinite(means).all():
        raise ValueError("at least one Docking-44 target column is entirely missing")
    missing = np.where(np.isnan(scores))
    scores[missing] = means[missing[1]]
    scales = scores.std(axis=0, ddof=1)
    if np.any(scales <= 0):
        raise ValueError("Docking-44 contains a constant target column")
    column_z = (scores - scores.mean(axis=0)) / scales
    residual = column_z - column_z.mean(axis=1, keepdims=True)
    genes = [PDB_TO_GENE[pdb] for pdb in pdb_columns]
    raw_geometry = np.corrcoef(scores, rowvar=False)
    residual_geometry = np.corrcoef(residual, rowvar=False)
    for matrix in (raw_geometry, residual_geometry):
        if not np.isfinite(matrix).all():
            raise ValueError("non-finite Docking-44 target geometry")
    return genes, {"raw_docking": raw_geometry, "residual_docking": residual_geometry}


def load_sequences(path: Path, targets: Iterable[str]) -> dict[str, str]:
    frame = pd.read_csv(path, sep="\t")
    required = {"Gene Names", "Sequence"}
    if not required <= set(frame.columns):
        raise ValueError("UniProt TSV must contain Gene Names and Sequence")
    frame = frame.copy()
    frame["primary_gene"] = frame["Gene Names"].fillna("").astype(str).str.split().str[0]
    sequences: dict[str, str] = {}
    for target in sorted(set(targets)):
        matches = frame.loc[frame["primary_gene"].eq(target), "Sequence"].dropna().unique()
        if len(matches) != 1:
            raise ValueError(f"expected one primary-gene UniProt sequence for {target}")
        sequence = str(matches[0]).strip().upper()
        if not sequence or not set(sequence) <= set("ACDEFGHIKLMNPQRSTVWYUXO"):
            raise ValueError(f"unexpected UniProt sequence alphabet for {target}")
        sequences[target] = sequence
    return sequences


def sequence_identity_matrix(targets: Sequence[str], sequences: dict[str, str]) -> np.ndarray:
    aligner = Align.PairwiseAligner()
    aligner.mode = "global"
    aligner.substitution_matrix = Align.substitution_matrices.load("BLOSUM62")
    aligner.open_gap_score = -10.0
    aligner.extend_gap_score = -0.5
    matrix = np.eye(len(targets), dtype=float)
    for first, target_a in enumerate(targets):
        for second in range(first + 1, len(targets)):
            target_b = targets[second]
            counts = aligner.align(sequences[target_a], sequences[target_b])[0].counts()
            denominator = counts.identities + counts.mismatches + counts.gaps
            value = float(counts.identities / denominator)
            matrix[first, second] = matrix[second, first] = value
    return matrix


def family_labels(targets: Sequence[str]) -> dict[str, str]:
    mapping = {gene: family for family, genes in CURATED_FAMILIES.items() for gene in genes}
    return {target: mapping.get(target, f"singleton_{target}") for target in targets}


def family_matrix(targets: Sequence[str]) -> np.ndarray:
    labels = family_labels(targets)
    return np.asarray(
        [[float(labels[left] == labels[right]) for right in targets] for left in targets],
        dtype=float,
    )


def align_docking_matrix(
    genes: Sequence[str], matrix: np.ndarray, targets: Sequence[str]
) -> np.ndarray:
    index = {gene: position for position, gene in enumerate(genes)}
    missing = set(targets) - set(index)
    if missing:
        raise ValueError(f"PDSP targets missing from Docking-44: {sorted(missing)}")
    order = [index[target] for target in targets]
    return matrix[np.ix_(order, order)]


def experimental_pair_frame(
    matrix: pd.DataFrame,
    predictor_targets: Sequence[str],
    predictors: dict[str, np.ndarray],
    min_support: int,
) -> pd.DataFrame:
    available = sorted(set(matrix.columns) & set(predictor_targets))
    target_index = {target: index for index, target in enumerate(predictor_targets)}
    rows: list[dict[str, object]] = []
    for first, target_a in enumerate(available):
        for target_b in available[first + 1 :]:
            common = matrix[[target_a, target_b]].dropna()
            if len(common) < min_support:
                continue
            if common[target_a].nunique() < 2 or common[target_b].nunique() < 2:
                continue
            endpoint = stats.spearmanr(common[target_a], common[target_b]).statistic
            if not np.isfinite(endpoint):
                continue
            ia, ib = target_index[target_a], target_index[target_b]
            rows.append(
                {
                    "target_a": target_a,
                    "target_b": target_b,
                    "pair_support": int(len(common)),
                    "experimental_spearman": float(endpoint),
                    **{name: float(value[ia, ib]) for name, value in predictors.items()},
                }
            )
    return pd.DataFrame.from_records(rows).sort_values(
        ["target_a", "target_b"], kind="mergesort"
    ).reset_index(drop=True)


def rank_values(values: Sequence[float]) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return stats.rankdata(values, method="average") / len(values)


def add_fixed_fusions(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    sequence = rank_values(result["full_sequence_identity"])
    family = rank_values(result["same_curated_family"])
    residual = rank_values(result["residual_docking"])
    result["equal_rank_sequence_family"] = (sequence + family) / 2.0
    result["equal_rank_sequence_residual"] = (sequence + residual) / 2.0
    result["equal_rank_sequence_family_residual"] = (
        sequence + family + residual
    ) / 3.0
    return result


def top_fraction_labels(endpoint: Sequence[float], fraction: float) -> np.ndarray:
    values = np.asarray(endpoint, dtype=float)
    if not 0 < fraction < 1:
        raise ValueError("top fraction must be between zero and one")
    count = max(1, int(math.ceil(fraction * len(values))))
    order = np.argsort(-values, kind="mergesort")
    labels = np.zeros(len(values), dtype=bool)
    labels[order[:count]] = True
    return labels


def expected_topk_inclusion(scores: np.ndarray, item: int, k: int) -> float:
    """Fractional inclusion probability under uniform ordering of score ties."""
    score = float(scores[item])
    above = int(np.sum(scores > score))
    tied = int(np.sum(scores == score))
    if above >= k:
        return 0.0
    if above + tied <= k:
        return 1.0
    return float((k - above) / tied)


def expected_reciprocal_rank(scores: np.ndarray, item: int) -> float:
    score = float(scores[item])
    above = int(np.sum(scores > score))
    tied = int(np.sum(scores == score))
    return float(np.mean([1.0 / rank for rank in range(above + 1, above + tied + 1)]))


def global_metrics(frame: pd.DataFrame, predictor: str, fraction: float) -> dict[str, float | int]:
    endpoint = frame["experimental_spearman"].to_numpy(dtype=float)
    scores = frame[predictor].to_numpy(dtype=float)
    labels = top_fraction_labels(endpoint, fraction)
    count = int(labels.sum())
    order = np.argsort(-scores, kind="mergesort")
    return {
        "target_pairs": int(len(frame)),
        "positive_pairs": count,
        "positive_fraction": float(labels.mean()),
        "continuous_spearman": float(stats.spearmanr(endpoint, scores).statistic),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "average_precision": float(average_precision_score(labels, scores)),
        "precision_at_positive_count": float(labels[order[:count]].mean()),
    }


def query_groups(frame: pd.DataFrame, min_candidates: int = 3) -> list[np.ndarray]:
    targets = sorted(set(frame["target_a"]) | set(frame["target_b"]))
    return [
        np.flatnonzero(
            frame["target_a"].eq(target).to_numpy()
            | frame["target_b"].eq(target).to_numpy()
        )
        for target in targets
        if int(frame["target_a"].eq(target).sum() + frame["target_b"].eq(target).sum())
        >= min_candidates
    ]


def targetwise_metrics(frame: pd.DataFrame, predictor: str) -> dict[str, float | int]:
    endpoint = frame["experimental_spearman"].to_numpy(dtype=float)
    scores = frame[predictor].to_numpy(dtype=float)
    mrr: list[float] = []
    best_at_three: list[float] = []
    top_two_at_three: list[float] = []
    rank_correlations: list[float] = []
    for group in query_groups(frame):
        local_endpoint = endpoint[group]
        local_scores = scores[group]
        true_order = np.argsort(-local_endpoint, kind="mergesort")
        mrr.append(expected_reciprocal_rank(local_scores, int(true_order[0])))
        best_at_three.append(expected_topk_inclusion(local_scores, int(true_order[0]), 3))
        top_two_at_three.append(
            float(
                np.mean(
                    [expected_topk_inclusion(local_scores, int(item), 3) for item in true_order[:2]]
                )
            )
        )
        if np.unique(local_endpoint).size > 1 and np.unique(local_scores).size > 1:
            rho = stats.spearmanr(local_endpoint, local_scores).statistic
            rank_correlations.append(float(rho))
    return {
        "eligible_query_targets": int(len(mrr)),
        "mean_reciprocal_rank_of_best_partner": float(np.mean(mrr)),
        "best_partner_recall_at_3": float(np.mean(best_at_three)),
        "top2_partner_recall_at_3": float(np.mean(top_two_at_three)),
        "macro_partner_rank_spearman": float(np.mean(rank_correlations)),
        "query_targets_with_finite_rank_spearman": int(len(rank_correlations)),
    }


def observed_metric_tables(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    global_rows: list[dict[str, object]] = []
    for fraction in (0.10, 0.20):
        for predictor in PREDICTORS:
            global_rows.append(
                {"top_fraction": fraction, "predictor": predictor,
                 **global_metrics(frame, predictor, fraction)}
            )
    query_rows = [
        {"predictor": predictor, **targetwise_metrics(frame, predictor)}
        for predictor in PREDICTORS
    ]
    return pd.DataFrame(global_rows), pd.DataFrame(query_rows)


def fast_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    ranks = stats.rankdata(scores, method="average")
    positives = int(labels.sum())
    negatives = len(labels) - positives
    return float(
        (ranks[labels].sum() - positives * (positives + 1) / 2.0)
        / (positives * negatives)
    )


def qap_key_metrics(frame: pd.DataFrame, scores: np.ndarray) -> dict[str, float]:
    endpoint = frame["experimental_spearman"].to_numpy(dtype=float)
    labels = top_fraction_labels(endpoint, PRIMARY_TOP_FRACTION)
    proxy = frame.copy()
    proxy["_candidate"] = scores
    return {
        "continuous_spearman": float(stats.spearmanr(endpoint, scores).statistic),
        "roc_auc_top10": fast_auc(labels, scores),
        "best_partner_recall_at_3": float(
            targetwise_metrics(proxy, "_candidate")["best_partner_recall_at_3"]
        ),
    }


def family_preserving_order(
    rng: np.random.Generator, targets: Sequence[str], labels: dict[str, str]
) -> np.ndarray:
    order = np.arange(len(targets))
    for family in sorted(set(labels.values())):
        members = np.asarray(
            [index for index, target in enumerate(targets) if labels[target] == family],
            dtype=int,
        )
        if len(members) > 1:
            order[members] = rng.permutation(members)
    return order


def qap_paired_gains(
    frame: pd.DataFrame,
    targets: Sequence[str],
    predictor_matrices: dict[str, np.ndarray],
    permutations: int,
    seed: int,
) -> pd.DataFrame:
    """Joint raw/residual QAP plus residual-label QAP for fixed fusion gain."""
    target_index = {target: index for index, target in enumerate(targets)}
    first = np.asarray([target_index[value] for value in frame["target_a"]], dtype=int)
    second = np.asarray([target_index[value] for value in frame["target_b"]], dtype=int)
    sequence = frame["full_sequence_identity"].to_numpy(dtype=float)
    family = frame["same_curated_family"].to_numpy(dtype=float)
    endpoint = frame["experimental_spearman"].to_numpy(dtype=float)
    endpoint_ranks = stats.rankdata(endpoint, method="average").astype(float)
    endpoint_ranks -= endpoint_ranks.mean()
    endpoint_rank_norm = float(np.linalg.norm(endpoint_ranks))
    top_labels = top_fraction_labels(endpoint, PRIMARY_TOP_FRACTION)
    groups = query_groups(frame)
    best_global_indices = [int(group[np.argmax(endpoint[group])]) for group in groups]

    def fast_metrics(scores: np.ndarray) -> dict[str, float]:
        score_ranks = stats.rankdata(scores, method="average").astype(float)
        score_ranks -= score_ranks.mean()
        score_norm = float(np.linalg.norm(score_ranks))
        best_at_three = []
        for group, best_global in zip(groups, best_global_indices):
            local_item = int(np.flatnonzero(group == best_global)[0])
            best_at_three.append(
                expected_topk_inclusion(scores[group], local_item, 3)
            )
        return {
            "continuous_spearman": float(
                np.dot(score_ranks, endpoint_ranks)
                / (score_norm * endpoint_rank_norm)
            ),
            "roc_auc_top10": fast_auc(top_labels, scores),
            "best_partner_recall_at_3": float(np.mean(best_at_three)),
        }

    base_fusion = (
        rank_values(sequence) + rank_values(family)
    ) / 2.0

    observed_raw = frame["raw_docking"].to_numpy(dtype=float)
    observed_residual = frame["residual_docking"].to_numpy(dtype=float)
    observed_fusion = (
        rank_values(sequence) + rank_values(family) + rank_values(observed_residual)
    ) / 3.0
    raw_metrics = fast_metrics(observed_raw)
    residual_metrics = fast_metrics(observed_residual)
    base_metrics = fast_metrics(base_fusion)
    fusion_metrics = fast_metrics(observed_fusion)

    observed = {
        ("residual_minus_raw_joint_label", metric): residual_metrics[metric] - raw_metrics[metric]
        for metric in raw_metrics
    }
    observed.update(
        {
            ("residual_increment_to_sequence_family", metric): fusion_metrics[metric]
            - base_metrics[metric]
            for metric in base_metrics
        }
    )
    exceed = {
        scheme: {key: 0 for key in observed}
        for scheme in ("all_target_labels", "within_curated_family")
    }
    family = family_labels(targets)
    rng = np.random.default_rng(seed)
    raw_matrix = predictor_matrices["raw_docking"]
    residual_matrix = predictor_matrices["residual_docking"]

    for _ in range(permutations):
        for scheme in exceed:
            order = (
                rng.permutation(len(targets))
                if scheme == "all_target_labels"
                else family_preserving_order(rng, targets, family)
            )
            raw_values = raw_matrix[np.ix_(order, order)][first, second]
            residual_values = residual_matrix[np.ix_(order, order)][first, second]
            candidate_fusion = (
                rank_values(sequence) + rank_values(frame["same_curated_family"])
                + rank_values(residual_values)
            ) / 3.0
            raw_null = fast_metrics(raw_values)
            residual_null = fast_metrics(residual_values)
            fusion_null = fast_metrics(candidate_fusion)
            for metric in raw_metrics:
                key = ("residual_minus_raw_joint_label", metric)
                if residual_null[metric] - raw_null[metric] >= observed[key]:
                    exceed[scheme][key] += 1
                key = ("residual_increment_to_sequence_family", metric)
                if fusion_null[metric] - base_metrics[metric] >= observed[key]:
                    exceed[scheme][key] += 1

    rows: list[dict[str, object]] = []
    for scheme, counts in exceed.items():
        for (contrast, metric), value in observed.items():
            rows.append(
                {
                    "permutation_scheme": scheme,
                    "contrast": contrast,
                    "metric": metric,
                    "observed_gain": value,
                    "permutations": int(permutations),
                    "one_sided_qap_p": float((counts[(contrast, metric)] + 1) / (permutations + 1)),
                }
            )
    return pd.DataFrame(rows)


def target_jackknife(frame: pd.DataFrame) -> pd.DataFrame:
    endpoint = frame["experimental_spearman"].to_numpy(dtype=float)
    primary_labels = top_fraction_labels(endpoint, PRIMARY_TOP_FRACTION)
    rows: list[dict[str, object]] = []
    targets = sorted(set(frame["target_a"]) | set(frame["target_b"]))
    for target in targets:
        keep = ~(
            frame["target_a"].eq(target).to_numpy()
            | frame["target_b"].eq(target).to_numpy()
        )
        subset = frame.loc[keep].reset_index(drop=True)
        labels = primary_labels[keep]
        if labels.sum() < 2 or (~labels).sum() < 2:
            continue
        for predictor in (
            "raw_docking", "residual_docking", "equal_rank_sequence_family",
            "equal_rank_sequence_family_residual",
        ):
            scores = subset[predictor].to_numpy(dtype=float)
            rows.append(
                {
                    "omitted_target": target,
                    "predictor": predictor,
                    "remaining_pairs": int(len(subset)),
                    "remaining_fixed_top10_positive_pairs": int(labels.sum()),
                    "continuous_spearman": float(
                        stats.spearmanr(subset["experimental_spearman"], scores).statistic
                    ),
                    "roc_auc_fixed_primary_top10": float(roc_auc_score(labels, scores)),
                    **targetwise_metrics(subset, predictor),
                }
            )
    result = pd.DataFrame(rows)
    pivots = []
    for metric in (
        "continuous_spearman", "roc_auc_fixed_primary_top10",
        "best_partner_recall_at_3", "mean_reciprocal_rank_of_best_partner",
    ):
        pivot = result.pivot(index="omitted_target", columns="predictor", values=metric)
        pivot[f"residual_minus_raw__{metric}"] = (
            pivot["residual_docking"] - pivot["raw_docking"]
        )
        pivot[f"fusion_increment__{metric}"] = (
            pivot["equal_rank_sequence_family_residual"]
            - pivot["equal_rank_sequence_family"]
        )
        pivots.append(pivot.filter(like=f"__{metric}").reset_index())
    merged = pivots[0]
    for values in pivots[1:]:
        merged = merged.merge(values, on="omitted_target", validate="one_to_one")
    return merged


def five_target_common_support_blocks(
    experimental: pd.DataFrame,
    targets: Sequence[str],
    predictors: dict[str, np.ndarray],
    min_complete_compounds: int = 20,
) -> pd.DataFrame:
    """Audit retrieval on every five-target block sharing identical compounds."""
    target_index = {target: index for index, target in enumerate(targets)}
    eligible = [target for target in targets if experimental[target].notna().sum() >= 20]
    rows: list[dict[str, object]] = []
    for subset in itertools.combinations(eligible, 5):
        complete = experimental[list(subset)].dropna()
        if len(complete) < min_complete_compounds:
            continue
        pair_rows = []
        for first, target_a in enumerate(subset):
            for target_b in subset[first + 1 :]:
                ia, ib = target_index[target_a], target_index[target_b]
                pair_rows.append(
                    {
                        "target_a": target_a,
                        "target_b": target_b,
                        "pair_support": len(complete),
                        "experimental_spearman": float(
                            stats.spearmanr(complete[target_a], complete[target_b]).statistic
                        ),
                        **{name: float(matrix[ia, ib]) for name, matrix in predictors.items()},
                    }
                )
        pair_frame = add_fixed_fusions(pd.DataFrame(pair_rows))
        for predictor in (
            "raw_docking", "residual_docking", "full_sequence_identity",
            "equal_rank_sequence_family_residual",
        ):
            metrics = global_metrics(pair_frame, predictor, 0.20)
            rows.append(
                {
                    "block_targets": "|".join(subset),
                    "complete_compounds": int(len(complete)),
                    "predictor": predictor,
                    **metrics,
                }
            )
    return pd.DataFrame(rows)


def targetwise_predictions(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for target in sorted(set(frame["target_a"]) | set(frame["target_b"])):
        subset = frame.loc[frame["target_a"].eq(target) | frame["target_b"].eq(target)].copy()
        if len(subset) < 3:
            continue
        subset["partner"] = np.where(
            subset["target_a"].eq(target), subset["target_b"], subset["target_a"]
        )
        true = subset.sort_values(
            ["experimental_spearman", "partner"], ascending=[False, True], kind="mergesort"
        ).iloc[0]
        row: dict[str, object] = {
            "query_target": target,
            "candidate_targets": int(len(subset)),
            "experimental_best_partner": str(true["partner"]),
            "experimental_best_spearman": float(true["experimental_spearman"]),
        }
        for predictor in PREDICTORS:
            predicted = subset.sort_values(
                [predictor, "partner"], ascending=[False, True], kind="mergesort"
            ).iloc[0]
            row[f"{predictor}__top_partner"] = str(predicted["partner"])
            ordered = subset.sort_values(
                [predictor, "partner"], ascending=[False, True], kind="mergesort"
            )["partner"].tolist()
            row[f"{predictor}__rank_of_experimental_best"] = int(
                ordered.index(str(true["partner"])) + 1
            )
        rows.append(row)
    return pd.DataFrame(rows)


def write_readme(output: Path, summary: dict) -> None:
    primary = summary["primary_decision_metrics"]
    text = f"""# PDSP counterscreen-retrieval audit

This post-hoc analysis asks a graph-level question: can Docking-44 target
geometry prioritize experimental co-affinity partners to include in a
counterscreen panel? It does **not** test ligand-wise target prediction.

## Frozen primary contract

- HUMAN PDSP Ki rows from the allowed recombinant/cloned source vocabulary.
- Exact relations only; positive Ki; valid nonempty molecular structure.
- Median pKi by full Standard InChIKey and target.
- Experimental edge: pairwise Spearman pKi, at least 10 common compounds.
- Predictors use no PDSP values. Fixed fusions average edge percentile ranks.
- Top-edge labels are the upper 10% of experimental edges.

Primary support: {summary['pdsp_audit']['strict_compounds']} compounds,
{summary['pdsp_audit']['strict_compound_target_cells']} cells,
{summary['primary_target_pairs']} target pairs, and
{summary['primary_targets']} targets.

Residual docking AUROC is {primary['residual_top10_auc']:.3f} versus
{primary['raw_top10_auc']:.3f} for raw docking. For choosing three
counterscreens per query target, residual geometry recovers the experimental
best partner with macro recall {primary['residual_best_partner_at_3']:.3f}
versus {primary['raw_best_partner_at_3']:.3f}. The fixed
sequence+family+residual fusion reaches
{primary['fusion_best_partner_at_3']:.3f}; its no-docking sequence+family
baseline reaches {primary['base_best_partner_at_3']:.3f}.

## Interpretation boundary and decision

**NO-GO as a general counterscreen selector.** Although the exact-only graph
passes aligned-target QAP and every target deletion, the operational advantage
does not survive restoration of explicit right-censored non-binders. Pair
endpoints in the pooled graph also use different compounds and publication
campaigns. The defensible positive result is narrower: residual geometry
retrieves affinity relationships *conditional on both targets having
quantifiable Ki values*. That conditional graph is a mechanistic hypothesis,
not yet a prospective counterscreen rule.

Raw PDSP data are not included because no redistribution license was located.
Download from `{PDSP_DOWNLOAD_URL}` and verify the checksum recorded in
`summary.json`.
"""
    (output / "README.md").write_text(text, encoding="utf-8")


def run(args: argparse.Namespace) -> dict:
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    cells, certified_cells, audit = strict_pdsp_cells(Path(args.pdsp_csv))
    (
        bound_cells,
        binary_cells,
        certified_bound_cells,
        certified_binary_cells,
        censor_audit,
    ) = censor_bound_pdsp_cells(Path(args.pdsp_csv))
    experimental = cell_matrix(cells)
    certified = cell_matrix(certified_cells)

    docking_genes, docking = docking_geometries(Path(args.docking_csv))
    overlap_targets = sorted(set(experimental.columns) & set(docking_genes))
    aligned_overlap = {
        name: align_docking_matrix(docking_genes, matrix, overlap_targets)
        for name, matrix in docking.items()
    }
    preliminary_pairs = experimental_pair_frame(
        experimental, overlap_targets, aligned_overlap, args.min_pair_support
    )
    primary_targets = sorted(
        set(preliminary_pairs["target_a"]) | set(preliminary_pairs["target_b"])
    )
    overlap_index = {target: index for index, target in enumerate(overlap_targets)}
    primary_order = [overlap_index[target] for target in primary_targets]
    aligned = {
        name: matrix[np.ix_(primary_order, primary_order)]
        for name, matrix in aligned_overlap.items()
    }
    sequences = load_sequences(Path(args.uniprot_tsv), primary_targets)
    aligned["full_sequence_identity"] = sequence_identity_matrix(
        primary_targets, sequences
    )
    aligned["same_curated_family"] = family_matrix(primary_targets)

    primary = add_fixed_fusions(
        experimental_pair_frame(
            experimental, primary_targets, aligned, args.min_pair_support
        )
    )
    global_table, query_table = observed_metric_tables(primary)
    qap = qap_paired_gains(
        primary,
        primary_targets,
        aligned,
        args.qap_permutations,
        args.seed,
    )
    jackknife = target_jackknife(primary)

    certified_pairs = add_fixed_fusions(
        experimental_pair_frame(
            certified, primary_targets, aligned, args.min_pair_support
        )
    )
    certified_global, certified_query = observed_metric_tables(certified_pairs)
    certified_global.insert(0, "sensitivity", "PDSP_Certified_Data_exact_only")
    certified_query.insert(0, "sensitivity", "PDSP_Certified_Data_exact_only")

    censor_frames: list[pd.DataFrame] = []
    censor_query_frames: list[pd.DataFrame] = []
    for sensitivity, values in (
        ("all_allowed_rows_right_censored_at_reported_bound", bound_cells),
        ("all_allowed_rows_quantified_vs_right_censored_binary", binary_cells),
        ("certified_right_censored_at_reported_bound", certified_bound_cells),
        ("certified_quantified_vs_right_censored_binary", certified_binary_cells),
    ):
        endpoint = cell_matrix(values)
        pair_frame = add_fixed_fusions(
            experimental_pair_frame(
                endpoint, primary_targets, aligned, args.min_pair_support
            )
        )
        table, query = observed_metric_tables(pair_frame)
        table.insert(0, "sensitivity", sensitivity)
        query.insert(0, "sensitivity", sensitivity)
        censor_frames.append(table)
        censor_query_frames.append(query)
    censor_sensitivity = pd.concat(censor_frames, ignore_index=True)
    censor_query_sensitivity = pd.concat(censor_query_frames, ignore_index=True)

    dense_blocks = five_target_common_support_blocks(
        experimental, primary_targets, aligned, min_complete_compounds=20
    )
    predictions = targetwise_predictions(primary)

    primary.to_csv(output / "target_pairs.csv", index=False)
    global_table.to_csv(output / "global_retrieval_metrics.csv", index=False)
    query_table.to_csv(output / "targetwise_retrieval_metrics.csv", index=False)
    qap.to_csv(output / "paired_qap.csv", index=False)
    jackknife.to_csv(output / "target_jackknife.csv", index=False)
    certified_global.to_csv(output / "certified_global_metrics.csv", index=False)
    certified_query.to_csv(output / "certified_targetwise_metrics.csv", index=False)
    censor_sensitivity.to_csv(output / "censor_sensitivity_metrics.csv", index=False)
    censor_query_sensitivity.to_csv(
        output / "censor_sensitivity_targetwise_metrics.csv", index=False
    )
    dense_blocks.to_csv(output / "common_support_blocks.csv", index=False)
    predictions.to_csv(output / "targetwise_predictions.csv", index=False)

    def metric(table: pd.DataFrame, predictor: str, column: str, fraction: float | None = None) -> float:
        selected = table.loc[table["predictor"].eq(predictor)]
        if fraction is not None:
            selected = selected.loc[np.isclose(selected["top_fraction"], fraction)]
        if len(selected) != 1:
            raise ValueError(f"metric lookup was not unique for {predictor}/{column}")
        return float(selected.iloc[0][column])

    raw_auc = metric(global_table, "raw_docking", "roc_auc", 0.10)
    residual_auc = metric(global_table, "residual_docking", "roc_auc", 0.10)
    raw_best3 = metric(query_table, "raw_docking", "best_partner_recall_at_3")
    residual_best3 = metric(query_table, "residual_docking", "best_partner_recall_at_3")
    base_best3 = metric(query_table, "equal_rank_sequence_family", "best_partner_recall_at_3")
    fusion_best3 = metric(
        query_table, "equal_rank_sequence_family_residual", "best_partner_recall_at_3"
    )

    dense_pivot = dense_blocks.pivot(
        index="block_targets", columns="predictor", values="roc_auc"
    )
    dense_delta = dense_pivot["residual_docking"] - dense_pivot["raw_docking"]
    certified_residual_rho = metric(
        certified_global.loc[np.isclose(certified_global["top_fraction"], 0.10)],
        "residual_docking", "continuous_spearman"
    )
    certified_raw_rho = metric(
        certified_global.loc[np.isclose(certified_global["top_fraction"], 0.10)],
        "raw_docking", "continuous_spearman"
    )

    def censor_query_metric(sensitivity: str, predictor: str, column: str) -> float:
        selected = censor_query_sensitivity.loc[
            censor_query_sensitivity["sensitivity"].eq(sensitivity)
            & censor_query_sensitivity["predictor"].eq(predictor)
        ]
        if len(selected) != 1:
            raise ValueError("censor query metric lookup was not unique")
        return float(selected.iloc[0][column])

    all_bound_name = "all_allowed_rows_right_censored_at_reported_bound"
    all_binary_name = "all_allowed_rows_quantified_vs_right_censored_binary"

    summary = {
        "analysis_status": "post_hoc_exploratory",
        "decision": "prioritize three target counterscreens for a query target",
        "go_no_go_verdict": "NO_GO_GENERAL_COUNTERSCREEN",
        "go_no_go_rule": (
            "GO only if residual-vs-raw gain survives aligned-target QAP, most target "
            "deletions, censor-aware endpoints, Certified-Data-only, and a majority "
            "of identical-support blocks"
        ),
        "pdsp_audit": audit,
        "censor_audit": censor_audit,
        "primary_target_pairs": int(len(primary)),
        "primary_targets": int(len(set(primary["target_a"]) | set(primary["target_b"]))),
        "minimum_pair_support": int(args.min_pair_support),
        "primary_decision_metrics": {
            "raw_top10_auc": raw_auc,
            "residual_top10_auc": residual_auc,
            "residual_minus_raw_top10_auc": residual_auc - raw_auc,
            "raw_best_partner_at_3": raw_best3,
            "residual_best_partner_at_3": residual_best3,
            "residual_minus_raw_best_partner_at_3": residual_best3 - raw_best3,
            "base_best_partner_at_3": base_best3,
            "fusion_best_partner_at_3": fusion_best3,
            "fusion_increment_best_partner_at_3": fusion_best3 - base_best3,
        },
        "certified_data_sensitivity": {
            "target_pairs": int(len(certified_pairs)),
            "targets": int(len(set(certified_pairs["target_a"]) | set(certified_pairs["target_b"]))),
            "raw_continuous_spearman": certified_raw_rho,
            "residual_continuous_spearman": certified_residual_rho,
            "residual_minus_raw": certified_residual_rho - certified_raw_rho,
        },
        "censor_aware_decision_sensitivity": {
            "reported_bound_best_partner_at_3": {
                predictor: censor_query_metric(
                    all_bound_name, predictor, "best_partner_recall_at_3"
                )
                for predictor in (
                    "raw_docking", "residual_docking", "full_sequence_identity",
                    "equal_rank_sequence_family",
                    "equal_rank_sequence_family_residual",
                )
            },
            "quantified_vs_censored_binary_best_partner_at_3": {
                predictor: censor_query_metric(
                    all_binary_name, predictor, "best_partner_recall_at_3"
                )
                for predictor in (
                    "raw_docking", "residual_docking", "full_sequence_identity",
                    "equal_rank_sequence_family",
                    "equal_rank_sequence_family_residual",
                )
            },
        },
        "identical_chemical_support_blocks": {
            "five_target_blocks_with_at_least_20_complete_compounds": int(len(dense_pivot)),
            "median_complete_compounds": float(
                dense_blocks.drop_duplicates("block_targets")["complete_compounds"].median()
            ),
            "median_residual_minus_raw_auc_top20": float(dense_delta.median()),
            "fraction_positive_residual_minus_raw_auc_top20": float((dense_delta > 0).mean()),
        },
        "qap_permutations_per_scheme": int(args.qap_permutations),
        "seed": int(args.seed),
        "inputs": {
            "pdsp": {"basename": Path(args.pdsp_csv).name, "sha256": sha256_file(Path(args.pdsp_csv)),
                     "url": PDSP_DOWNLOAD_URL},
            "uniprot": {"basename": Path(args.uniprot_tsv).name,
                        "sha256": sha256_file(Path(args.uniprot_tsv)), "url": UNIPROT_REST_URL},
            "docking44": {"path": "data/frozen/df_final_v4.csv.gz",
                          "sha256": sha256_file(Path(args.docking_csv))},
        },
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__, "rdkit": rdkit.__version__, "biopython": Bio.__version__,
        },
        "limitations": [
            "pairwise PDSP endpoints use nonidentical compound and publication supports",
            "exact-only correlations condition on a quantifiable Ki and are activity/MNAR sensitive",
            "target labels and families are a fixed, nonrandom Docking-44 panel",
            "all predictor combinations and evaluations are post hoc",
            "top-edge retrieval does not establish prospective ligand-level target retrieval",
        ],
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_readme(output, summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdsp-csv", type=Path, required=True)
    parser.add_argument("--uniprot-tsv", type=Path, required=True)
    parser.add_argument("--docking-csv", type=Path, default=DEFAULT_DOCKING)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--min-pair-support", type=int, default=DEFAULT_MIN_PAIR_SUPPORT)
    parser.add_argument("--qap-permutations", type=int, default=DEFAULT_QAP_PERMUTATIONS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
