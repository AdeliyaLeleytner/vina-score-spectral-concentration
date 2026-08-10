#!/usr/bin/env python3
"""Dense DAVIS x DOCKSTRING target-preference benchmark.

The benchmark matches the 72 DAVIS inhibitors to the released DOCKSTRING
structures and evaluates the 21 kinase targets shared by the two resources.
The primary molecular identity is the full Standard InChIKey recomputed from
the released SMILES on both sides.  DOCKSTRING profiles that represent the
same full key are collapsed by the cell-wise median *before* any score
transformation.

Score offsets and scales are learned from the complete DOCKSTRING reference
surface after excluding every row whose recomputed Standard InChI connectivity
block occurs in DAVIS.  Thus neither an evaluated representation nor a
stereochemical/protomer variant of it contributes to the fitted transformation.

The public entry point :func:`dense_davis_dockstring_benchmark` returns a
JSON-serialisable report, including record-oriented tables.  The command-line
interface additionally writes the report and each table to disk.
"""

from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.ML.Cluster import Butina
from scipy import stats

RDLogger.DisableLog("rdApp.*")


PACKAGE = Path(__file__).resolve().parents[1]
DEFAULT_DOCKSTRING = PACKAGE / "data" / "frozen" / "dockstring-dataset.tsv.gz"
DEFAULT_DAVIS = PACKAGE / "data" / "frozen" / "davis_complete.tab.gz"

# Ordered as the released DOCKSTRING target columns.  Values are the DAVIS
# protein identifiers; aliases are explicit rather than inferred from strings.
TARGET_MAP: OrderedDict[str, str] = OrderedDict(
    [
        ("ABL1", "abl1"),
        ("AKT1", "akt1"),
        ("AKT2", "akt2"),
        ("CDK2", "cdk2"),
        ("CSF1R", "csf1r"),
        ("EGFR", "egfr"),
        ("FGFR1", "fgfr1"),
        ("IGF1R", "igf1r"),
        ("JAK2", "jak2_jh1domain_catalytic"),
        ("KDR", "vegfr2"),
        ("KIT", "kit"),
        ("LCK", "lck"),
        ("MAP2K1", "mek1"),
        ("MAPK1", "erk2"),
        ("MAPK14", "p38_alpha"),
        ("MAPKAPK2", "mapkapk2"),
        ("MET", "met"),
        ("PLK1", "plk1"),
        ("PTK2", "fak"),
        ("ROCK1", "rock1"),
        ("SRC", "src"),
    ]
)

CORE_REPRESENTATIONS = (
    "absolute_vina",
    "target_centered_unscaled",
    "column_standardized",
    "target_centered_residual_scaled",
    "two_way_residual",
)
BASELINE_REPRESENTATIONS = ("docking_target_prior",)
PAIRWISE_CONTRASTS = (
    ("two_way_residual", "absolute_vina"),
    ("two_way_residual", "column_standardized"),
    ("two_way_residual", "target_centered_residual_scaled"),
    ("target_centered_residual_scaled", "target_centered_unscaled"),
    ("target_centered_unscaled", "absolute_vina"),
    ("column_standardized", "absolute_vina"),
    # Ligand-independent calibration baseline: every ligand receives the same
    # target ordering learned from the leakage-free DOCKSTRING reference pool.
    ("absolute_vina", "docking_target_prior"),
)

OPERATIONAL_PATH_STEPS: OrderedDict[str, str] = OrderedDict(
    [
        (
            "absolute_vina_over_ligand_independent_target_prior",
            "absolute_vina_minus_docking_target_prior",
        ),
        (
            "remove_reference_target_offsets",
            "target_centered_unscaled_minus_absolute_vina",
        ),
        (
            "apply_residual_target_scales_before_row_correction",
            "target_centered_residual_scaled_minus_target_centered_unscaled",
        ),
        (
            "add_row_correction_under_residual_target_scales",
            "two_way_residual_minus_target_centered_residual_scaled",
        ),
        (
            "net_scaled_two_way_residual_over_absolute_vina",
            "two_way_residual_minus_absolute_vina",
        ),
    ]
)
MATCH_VARIANTS = OrderedDict(
    [
        ("standard_full_inchikey", "standard_inchikey"),
        ("canonical_isomeric_graph", "canonical_isomeric_smiles"),
        ("canonical_nonisomeric_graph", "canonical_nonisomeric_smiles"),
        ("standard_inchi_connectivity", "connectivity_block"),
    ]
)


def _finite_float(value: float) -> float | None:
    value = float(value)
    return value if np.isfinite(value) else None


def _quantile(values: np.ndarray, probability: float) -> float | None:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    return _finite_float(np.quantile(finite, probability)) if len(finite) else None


def _distribution(values: Iterable[float]) -> dict:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if not len(array):
        return {
            "n": 0,
            "mean": None,
            "median": None,
            "standard_deviation": None,
            "interval_95": [None, None],
            "interval_90": [None, None],
            "minimum": None,
            "maximum": None,
        }
    return {
        "n": int(len(array)),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "standard_deviation": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        "interval_95": [_quantile(array, 0.025), _quantile(array, 0.975)],
        "interval_90": [_quantile(array, 0.05), _quantile(array, 0.95)],
        "minimum": float(array.min()),
        "maximum": float(array.max()),
    }


def _molecule_identity(smiles: str) -> tuple[str, str, str, str, float, int]:
    molecule = Chem.MolFromSmiles(str(smiles))
    if molecule is None:
        raise ValueError(f"RDKit could not parse SMILES: {smiles!r}")
    key = Chem.MolToInchiKey(molecule)
    if not key or len(key) < 14:
        raise ValueError(f"RDKit could not generate a Standard InChIKey: {smiles!r}")
    return (
        key,
        key[:14],
        Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True),
        Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=False),
        float(Descriptors.MolWt(molecule)),
        int(molecule.GetNumHeavyAtoms()),
    )


def _identity_table(
    frame: pd.DataFrame,
    smiles_column: str,
    *,
    scan: str,
    davis_connectivity_blocks: set[str] | None = None,
) -> pd.DataFrame:
    """Recompute structure identities, optionally using a documented fast prefilter.

    ``scan='full'`` is the primary and recomputes identities for every row.  The
    prefilter mode exists only for development/smoke tests and is labelled as such
    in the report; it must not be used for submission results.
    """
    if scan not in {"full", "reported_connectivity_prefilter"}:
        raise ValueError(f"Unsupported identity scan: {scan}")
    work = frame.copy()
    identity_index = work.index
    if scan == "reported_connectivity_prefilter":
        if davis_connectivity_blocks is None or "inchikey" not in work:
            raise ValueError("The prefilter requires DAVIS blocks and reported InChIKeys")
        identity_index = work.index[
            work["inchikey"].astype(str).str[:14].isin(davis_connectivity_blocks)
        ]
    identities = [
        _molecule_identity(value)
        for value in work.loc[identity_index, smiles_column].astype(str)
    ]
    identity_frame = pd.DataFrame(
        identities,
        columns=[
            "standard_inchikey",
            "connectivity_block",
            "canonical_isomeric_smiles",
            "canonical_nonisomeric_smiles",
            "molecular_weight",
            "heavy_atom_count",
        ],
        index=identity_index,
    )
    return work.join(identity_frame)


def _load_inputs(
    dockstring_path: Path,
    davis_path: Path,
    identity_scan: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    target_columns = list(TARGET_MAP)
    # Match the manuscript's frozen 260,060-ligand support: a row is retained
    # only when all 58 released target scores are observed, not merely the 21
    # kinase columns used below.
    dockstring_all = pd.read_csv(dockstring_path, sep="\t")
    score_columns = [
        column for column in dockstring_all.columns if column not in {"inchikey", "smiles"}
    ]
    complete = ~dockstring_all[score_columns].isna().any(axis=1)
    dockstring = dockstring_all.loc[
        complete, ["inchikey", "smiles", *target_columns]
    ].copy()
    if dockstring[target_columns].isna().any().any() or len(dockstring) != 260060:
        raise ValueError(
            "Expected the complete 260,060-row DOCKSTRING release support; "
            f"observed {len(dockstring):,} rows"
        )
    # Preserve the zero-based row position in the released 260,155-row table,
    # even though 95 incomplete rows are removed from the analysis support.
    dockstring.insert(0, "dockstring_row", dockstring.index.to_numpy(dtype=int))

    davis = pd.read_csv(
        davis_path,
        sep="\t",
        usecols=["drug_name", "protein", "affinity", "compound_iso_smiles", "y"],
    )
    if davis.drug_name.nunique() != 72 or davis.protein.nunique() != 442:
        raise ValueError(
            "Expected the frozen dense DAVIS 72 x 442 release, observed "
            f"{davis.drug_name.nunique()} x {davis.protein.nunique()}"
        )
    molecule_rows = davis[["drug_name", "compound_iso_smiles"]].drop_duplicates()
    if molecule_rows.drug_name.duplicated().any():
        raise ValueError("A DAVIS drug name maps to more than one released SMILES")
    davis_identity = _identity_table(
        molecule_rows,
        "compound_iso_smiles",
        scan="full",
    )
    davis_blocks = set(davis_identity.connectivity_block)
    dockstring_identity = _identity_table(
        dockstring,
        "smiles",
        scan=identity_scan,
        davis_connectivity_blocks=davis_blocks,
    )

    selected = davis[davis.protein.isin(TARGET_MAP.values())].copy()
    if selected.duplicated(["drug_name", "protein"]).any():
        raise ValueError("The frozen DAVIS table has duplicate drug-target records")
    experiment = selected.pivot(index="drug_name", columns="protein", values="y")
    experiment = experiment.reindex(
        index=davis_identity.drug_name,
        columns=list(TARGET_MAP.values()),
    )
    if experiment.shape != (72, 21) or experiment.isna().any().any():
        raise ValueError(
            f"Expected a complete DAVIS 72 x 21 block, observed {experiment.shape} "
            f"with {int(experiment.isna().sum().sum())} missing cells"
        )
    experiment.columns = list(TARGET_MAP)
    affinity = selected.pivot(index="drug_name", columns="protein", values="affinity")
    affinity = affinity.reindex(
        index=davis_identity.drug_name,
        columns=list(TARGET_MAP.values()),
    )
    affinity.columns = list(TARGET_MAP)
    if affinity.shape != experiment.shape or affinity.isna().any().any():
        raise ValueError("The frozen DAVIS affinity block is not complete")
    censored = affinity.eq(10000.0)
    # Do not infer censoring from rounded pKd.  The flag comes from the released
    # raw affinity cap, while these assertions ensure the transformed endpoint
    # remains exactly consistent with that cap.
    if not np.allclose(
        experiment.to_numpy(dtype=float)[censored.to_numpy(dtype=bool)],
        5.0,
        rtol=0,
        atol=1e-12,
    ):
        raise ValueError("A raw DAVIS 10,000 nM censoring value is not pKd=5")
    if np.any(
        experiment.to_numpy(dtype=float)[~censored.to_numpy(dtype=bool)]
        <= 5.0 + 1e-12
    ):
        raise ValueError("A non-censored DAVIS affinity is at or below the pKd floor")
    return dockstring_identity, davis_identity, experiment, censored


def _aggregate_rows(
    frame: pd.DataFrame,
    identity_column: str,
    target_columns: list[str],
    method: str,
) -> pd.DataFrame:
    grouped = frame.groupby(identity_column, sort=True)[target_columns]
    if method == "median":
        return grouped.median()
    if method == "mean":
        return grouped.mean()
    if method == "first":
        return grouped.first()
    raise ValueError(f"Unsupported duplicate aggregation: {method}")


def _matched_surface(
    dockstring: pd.DataFrame,
    davis_identity: pd.DataFrame,
    experiment: pd.DataFrame,
    censored: pd.DataFrame,
    identity_column: str,
    aggregation: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    target_columns = list(TARGET_MAP)
    davis = davis_identity.set_index("drug_name")
    common = sorted(set(davis[identity_column]) & set(dockstring[identity_column]))
    if not common:
        raise ValueError(f"No overlap under {identity_column}")

    # There are no duplicates in the frozen DAVIS structure support today.  The
    # groupby makes the sensitivity well-defined if that changes.
    exp_with_identity = experiment.join(davis[[identity_column]])
    experimental = exp_with_identity[
        exp_with_identity[identity_column].isin(common)
    ].groupby(identity_column, sort=True)[target_columns].median()
    censor_with_identity = censored.join(davis[[identity_column]])
    matched_censored = censor_with_identity[
        censor_with_identity[identity_column].isin(common)
    ].groupby(identity_column, sort=True)[target_columns].max().astype(bool)
    docking = _aggregate_rows(
        dockstring[dockstring[identity_column].isin(common)],
        identity_column,
        target_columns,
        aggregation,
    ).reindex(experimental.index)
    smiles = (
        davis[davis[identity_column].isin(common)]
        .groupby(identity_column, sort=True)["compound_iso_smiles"]
        .first()
        .reindex(experimental.index)
        .to_frame("smiles")
    )
    matched_censored = matched_censored.reindex(experimental.index)
    if (
        docking.isna().any().any()
        or experimental.isna().any().any()
        or matched_censored.isna().any().any()
    ):
        raise ValueError("A matched DAVIS x DOCKSTRING surface is not dense")
    return docking, experimental, matched_censored, smiles


def _score_representations(
    reference: np.ndarray,
    evaluation: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict]:
    reference = np.asarray(reference, dtype=float)
    evaluation = np.asarray(evaluation, dtype=float)
    target_mean = reference.mean(axis=0)
    target_sd = reference.std(axis=0, ddof=1)
    grand_mean = float(reference.mean())
    reference_residual = (
        reference
        - target_mean[None, :]
        - reference.mean(axis=1, keepdims=True)
        + grand_mean
    )
    residual_sd = reference_residual.std(axis=0, ddof=1)
    evaluation_residual = (
        evaluation
        - target_mean[None, :]
        - evaluation.mean(axis=1, keepdims=True)
        + grand_mean
    )
    if (target_sd <= 0).any() or (residual_sd <= 0).any():
        raise ValueError("A reference target has zero score variance")
    representations = {
        "absolute_vina": evaluation,
        "target_centered_unscaled": evaluation - target_mean[None, :],
        "two_way_centered_unscaled": evaluation_residual,
        "column_standardized": (
            evaluation - target_mean[None, :]
        ) / target_sd[None, :],
        "target_centered_residual_scaled": (
            evaluation - target_mean[None, :]
        ) / residual_sd[None, :],
        "two_way_residual": evaluation_residual / residual_sd[None, :],
        "docking_target_prior": np.tile(target_mean, (len(evaluation), 1)),
    }
    fit = {
        "reference_ligands": int(len(reference)),
        "targets": int(reference.shape[1]),
        "target_mean": [float(value) for value in target_mean],
        "target_standard_deviation": [float(value) for value in target_sd],
        "residual_target_standard_deviation": [float(value) for value in residual_sd],
        "grand_mean": grand_mean,
    }
    return representations, fit


def _pairwise_metrics(
    scores: np.ndarray,
    experiment: np.ndarray,
    censored: np.ndarray,
    margin: float,
    stratum: str,
) -> tuple[dict, np.ndarray]:
    """Equal-ligand target-preference concordance on a dense panel."""
    scores = np.asarray(scores, dtype=float)
    experiment = np.asarray(experiment, dtype=float)
    censored = np.asarray(censored, dtype=bool)
    if scores.shape != experiment.shape or censored.shape != experiment.shape:
        raise ValueError("Score, experiment, and censor matrices must have identical shape")
    if stratum not in {"all_informative", "both_uncensored", "floor_vs_uncensored"}:
        raise ValueError(f"Unknown pair stratum: {stratum}")

    first, second = np.triu_indices(experiment.shape[1], k=1)
    experimental_difference = experiment[:, first] - experiment[:, second]
    score_difference = scores[:, first] - scores[:, second]
    first_censored = censored[:, first]
    second_censored = censored[:, second]
    both_censored = first_censored & second_censored
    if np.any(
        both_censored & (np.abs(experimental_difference) > 1e-12)
    ):
        raise ValueError("Two explicitly censored DAVIS cells have unequal pKd values")
    excluded_both_censored = int(both_censored.sum())
    excluded_by_margin = (
        ~both_censored & (np.abs(experimental_difference) <= margin)
    )
    excluded_margin = int(excluded_by_margin.sum())
    eligible = ~both_censored & ~excluded_by_margin
    if stratum == "both_uncensored":
        stratum_mask = ~first_censored & ~second_censored
    elif stratum == "floor_vs_uncensored":
        stratum_mask = first_censored ^ second_censored
    else:
        stratum_mask = np.ones_like(eligible, dtype=bool)
    excluded_stratum = int((eligible & ~stratum_mask).sum())
    eligible &= stratum_mask

    score_tie = score_difference == 0
    correct = np.where(
        score_tie,
        0.5,
        (np.sign(experimental_difference) == -np.sign(score_difference)).astype(float),
    )
    correct_by_ligand = np.where(eligible, correct, 0.0).sum(axis=1)
    pairs_by_ligand = eligible.sum(axis=1)
    per_ligand = np.divide(
        correct_by_ligand,
        pairs_by_ligand,
        out=np.full(len(experiment), np.nan),
        where=pairs_by_ligand > 0,
    )
    total_correct = float(correct_by_ligand.sum())
    total_pairs = int(pairs_by_ligand.sum())
    score_ties = int((eligible & score_tie).sum())

    top1 = []
    spearman = []
    if stratum == "all_informative":
        for score_row, experimental_row, censor_row in zip(
            scores, experiment, censored
        ):
            best = np.flatnonzero(
                np.isclose(experimental_row, experimental_row.max(), rtol=0, atol=1e-12)
            )
            # If every target is censored, DAVIS provides no top-target label.
            top1.append(
                np.nan
                if censor_row.all()
                else float(int(np.argmin(score_row)) in set(best.tolist()))
            )
            if (
                np.ptp(score_row) <= 1e-12
                or np.ptp(experimental_row) <= 1e-12
            ):
                correlation = np.nan
            else:
                correlation = stats.spearmanr(-score_row, experimental_row).statistic
            spearman.append(float(correlation) if np.isfinite(correlation) else np.nan)
    finite = np.isfinite(per_ligand)
    summary = {
        "mean_per_ligand_pairwise_concordance": (
            float(np.nanmean(per_ligand)) if finite.any() else None
        ),
        "pair_weighted_concordance": (
            float(total_correct / total_pairs) if total_pairs else None
        ),
        "evaluated_ligands": int(finite.sum()),
        "evaluated_pairs": int(total_pairs),
        "excluded_by_experimental_margin": int(excluded_margin),
        "excluded_both_censored_pairs": int(excluded_both_censored),
        "excluded_by_stratum": int(excluded_stratum),
        "predicted_score_ties_half_credit": int(score_ties),
        "chance_concordance": 0.5,
        "difference_from_chance_0_5": (
            float(np.nanmean(per_ligand) - 0.5) if finite.any() else None
        ),
        "mean_within_ligand_spearman": (
            _finite_float(np.nanmean(spearman)) if spearman else None
        ),
        "spearman_evaluated_ligands": (
            int(np.isfinite(spearman).sum()) if spearman else 0
        ),
        "top1_accuracy_allowing_experimental_ties": (
            _finite_float(np.nanmean(top1))
            if top1 and np.isfinite(top1).any()
            else None
        ),
        "top1_evaluated_ligands": int(np.isfinite(top1).sum()) if top1 else 0,
    }
    return summary, per_ligand


def _hit_detection_auc(
    scores: np.ndarray,
    censored: np.ndarray,
) -> tuple[dict, np.ndarray]:
    """Per-ligand ROC AUC for detecting uncensored DAVIS affinities.

    An uncensored affinity (raw Kd < 10,000 nM) is the positive class and a
    more negative docking score is the positive prediction.  With no score
    ties, this is algebraically the same per-ligand quantity as concordance on
    the floor-versus-uncensored stratum; the separate name makes the operational
    binary retrieval task explicit.
    """
    scores = np.asarray(scores, dtype=float)
    censored = np.asarray(censored, dtype=bool)
    if scores.shape != censored.shape:
        raise ValueError("Score and censor matrices must have identical shape")
    records = np.full(len(scores), np.nan)
    positive_negative_pairs = 0
    for ligand, (score_row, censor_row) in enumerate(zip(scores, censored)):
        positive = ~censor_row
        n_positive = int(positive.sum())
        n_negative = int(censor_row.sum())
        if not n_positive or not n_negative:
            continue
        ranks = stats.rankdata(-score_row, method="average")
        records[ligand] = (
            ranks[positive].sum() - n_positive * (n_positive + 1) / 2
        ) / (n_positive * n_negative)
        positive_negative_pairs += n_positive * n_negative
    finite = np.isfinite(records)
    return (
        {
            "mean_per_ligand_auc": (
                float(np.nanmean(records)) if finite.any() else None
            ),
            "evaluated_ligands": int(finite.sum()),
            "positive_negative_pairs": int(positive_negative_pairs),
            "chance_auc": 0.5,
            "difference_from_chance_0_5": (
                float(np.nanmean(records) - 0.5) if finite.any() else None
            ),
            "positive_class": "raw DAVIS affinity < 10,000 nM",
            "negative_class": "raw DAVIS affinity = 10,000 nM censoring cap",
        },
        records,
    )


def _murcko_labels(smiles: pd.Series) -> np.ndarray:
    labels: list[str] = []
    for index, value in enumerate(smiles.astype(str)):
        molecule = Chem.MolFromSmiles(value)
        if molecule is None:
            labels.append(f"INVALID:{index}")
            continue
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=molecule)
        if scaffold:
            labels.append(scaffold)
        else:
            connectivity = Chem.MolToInchiKey(molecule)[:14]
            labels.append(f"ACYCLIC:{connectivity}")
    return np.asarray(labels, dtype=object)


def _butina_labels(
    smiles: pd.Series,
    radius: int = 2,
    fp_size: int = 2048,
    similarity_threshold: float = 0.65,
) -> np.ndarray:
    molecules = [Chem.MolFromSmiles(str(value)) for value in smiles]
    if any(molecule is None for molecule in molecules):
        raise ValueError("Cannot form Butina clusters because an evaluation SMILES is invalid")
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=radius,
        fpSize=fp_size,
    )
    fingerprints = [generator.GetFingerprint(molecule) for molecule in molecules]
    distances: list[float] = []
    for row in range(1, len(fingerprints)):
        similarities = DataStructs.BulkTanimotoSimilarity(
            fingerprints[row], fingerprints[:row]
        )
        distances.extend(1.0 - value for value in similarities)
    clusters = Butina.ClusterData(
        distances,
        len(fingerprints),
        1.0 - similarity_threshold,
        isDistData=True,
        reordering=True,
    )
    labels = np.full(len(fingerprints), -1, dtype=int)
    for cluster_id, members in enumerate(clusters):
        labels[list(members)] = cluster_id
    if (labels < 0).any():
        raise RuntimeError("Butina failed to assign every molecule")
    return labels


def _bootstrap_mean(
    values: np.ndarray,
    repeats: int,
    rng: np.random.Generator,
    cluster_labels: np.ndarray | None = None,
) -> dict:
    values = np.asarray(values, dtype=float)
    if cluster_labels is None:
        sampled = rng.integers(0, len(values), size=(repeats, len(values)))
        with np.errstate(invalid="ignore"):
            records = np.nanmean(values[sampled], axis=1)
        cluster_count = None
    else:
        labels = np.asarray(cluster_labels)
        _, inverse = np.unique(labels, return_inverse=True)
        cluster_count = int(inverse.max() + 1)
        finite = np.isfinite(values)
        cluster_sum = np.bincount(
            inverse[finite], weights=values[finite], minlength=cluster_count
        )
        cluster_n = np.bincount(inverse[finite], minlength=cluster_count)
        sampled = rng.integers(
            0, cluster_count, size=(repeats, cluster_count)
        )
        numerator = cluster_sum[sampled].sum(axis=1)
        denominator = cluster_n[sampled].sum(axis=1)
        records = np.divide(
            numerator,
            denominator,
            out=np.full(repeats, np.nan),
            where=denominator > 0,
        )
    return {
        "plugin_mean": _finite_float(np.nanmean(values)),
        "n_clusters": cluster_count,
        **_distribution(records),
    }


def _uncertainty(
    values: np.ndarray,
    murcko: np.ndarray,
    butina: np.ndarray,
    repeats: int,
    seed: int,
) -> dict:
    return {
        "ligand_bootstrap": _bootstrap_mean(
            values, repeats, np.random.default_rng(seed)
        ),
        "murcko_cluster_bootstrap": _bootstrap_mean(
            values,
            repeats,
            np.random.default_rng(seed + 1),
            cluster_labels=murcko,
        ),
        "butina_cluster_bootstrap": _bootstrap_mean(
            values,
            repeats,
            np.random.default_rng(seed + 2),
            cluster_labels=butina,
        ),
    }


def _panel_analysis(
    reference: pd.DataFrame,
    docking: pd.DataFrame,
    experiment: pd.DataFrame,
    censored: pd.DataFrame,
    smiles: pd.Series,
    target_columns: list[str],
    bootstrap_repeats: int,
    seed: int,
    include_uncertainty: bool,
) -> tuple[dict, list[dict], dict[str, dict[str, dict[str, np.ndarray]]]]:
    representations, fit = _score_representations(
        reference[target_columns].to_numpy(dtype=float),
        docking[target_columns].to_numpy(dtype=float),
    )
    murcko = _murcko_labels(smiles)
    butina = _butina_labels(smiles)
    estimates: dict[str, dict] = {}
    metric_rows: list[dict] = []
    vectors: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    reported_representations = (*CORE_REPRESENTATIONS, *BASELINE_REPRESENTATIONS)
    for representation in reported_representations:
        estimates[representation] = {}
        vectors[representation] = {}
        for margin in (0.0, 0.5, 1.0):
            margin_key = f"{margin:.1f}"
            estimates[representation][margin_key] = {}
            vectors[representation][margin_key] = {}
            for stratum in (
                "all_informative",
                "both_uncensored",
                "floor_vs_uncensored",
            ):
                summary, per_ligand = _pairwise_metrics(
                    representations[representation],
                    experiment[target_columns].to_numpy(dtype=float),
                    censored[target_columns].to_numpy(dtype=bool),
                    margin,
                    stratum,
                )
                record = dict(summary)
                if include_uncertainty:
                    record["uncertainty"] = _uncertainty(
                        per_ligand,
                        murcko,
                        butina,
                        bootstrap_repeats,
                        seed
                        + 1000 * list(reported_representations).index(representation)
                        + 100 * int(margin * 10)
                        + 10
                        * (
                            "all_informative",
                            "both_uncensored",
                            "floor_vs_uncensored",
                        ).index(stratum),
                    )
                estimates[representation][margin_key][stratum] = record
                vectors[representation][margin_key][stratum] = per_ligand
                row = {
                    "record_type": "representation_estimate",
                    "representation": representation,
                    "experimental_margin_pkd": margin,
                    "pair_stratum": stratum,
                    **{key: value for key, value in summary.items()},
                }
                for method, bootstrap in record.get("uncertainty", {}).items():
                    row[f"{method}_interval_95_low"] = bootstrap["interval_95"][0]
                    row[f"{method}_interval_95_high"] = bootstrap["interval_95"][1]
                metric_rows.append(row)

    contrasts: dict[str, dict] = {}
    for contrast_index, (first, second) in enumerate(PAIRWISE_CONTRASTS):
        contrast_name = f"{first}_minus_{second}"
        contrasts[contrast_name] = {}
        for margin in (0.0, 0.5, 1.0):
            margin_key = f"{margin:.1f}"
            contrasts[contrast_name][margin_key] = {}
            for stratum_index, stratum in enumerate(
                ("all_informative", "both_uncensored", "floor_vs_uncensored")
            ):
                difference = (
                    vectors[first][margin_key][stratum]
                    - vectors[second][margin_key][stratum]
                )
                report = {
                    "plugin_mean_difference": _finite_float(np.nanmean(difference))
                }
                if include_uncertainty:
                    report["uncertainty"] = _uncertainty(
                        difference,
                        murcko,
                        butina,
                        bootstrap_repeats,
                        seed + 10000 + 1000 * contrast_index + 100 * int(margin * 10)
                        + 10 * stratum_index,
                    )
                contrasts[contrast_name][margin_key][stratum] = report
                row = {
                    "record_type": "paired_difference",
                    "representation": contrast_name,
                    "experimental_margin_pkd": margin,
                    "pair_stratum": stratum,
                    "mean_per_ligand_pairwise_concordance": report[
                        "plugin_mean_difference"
                    ],
                }
                for method, bootstrap in report.get("uncertainty", {}).items():
                    row[f"{method}_interval_95_low"] = bootstrap["interval_95"][0]
                    row[f"{method}_interval_95_high"] = bootstrap["interval_95"][1]
                metric_rows.append(row)

    # This exact invariant identifies the operationally consequential scaling step.
    centered_metric, _ = _pairwise_metrics(
        representations["target_centered_unscaled"],
        experiment[target_columns].to_numpy(dtype=float),
        censored[target_columns].to_numpy(dtype=bool),
        0.0,
        "all_informative",
    )
    two_way_metric, _ = _pairwise_metrics(
        representations["two_way_centered_unscaled"],
        experiment[target_columns].to_numpy(dtype=float),
        censored[target_columns].to_numpy(dtype=bool),
        0.0,
        "all_informative",
    )
    invariant_difference = (
        centered_metric["mean_per_ligand_pairwise_concordance"]
        - two_way_metric["mean_per_ligand_pairwise_concordance"]
    )
    hit_detection = {}
    for offset, representation in enumerate(reported_representations):
        auc_summary, auc_vector = _hit_detection_auc(
            representations[representation],
            censored[target_columns].to_numpy(dtype=bool),
        )
        auc_summary["uncertainty"] = _uncertainty(
            auc_vector,
            murcko,
            butina,
            bootstrap_repeats,
            seed + 50000 + 10 * offset,
        )
        floor_summary = estimates[representation]["0.0"]["floor_vs_uncensored"]
        auc_summary["identity_check_against_floor_vs_uncensored_concordance"] = float(
            auc_summary["mean_per_ligand_auc"]
            - floor_summary["mean_per_ligand_pairwise_concordance"]
        )
        hit_detection[representation] = auc_summary
        auc_row = {
            "record_type": "hit_detection_auc",
            "representation": representation,
            "experimental_margin_pkd": 0.0,
            "pair_stratum": "floor_vs_uncensored",
            "mean_per_ligand_pairwise_concordance": auc_summary[
                "mean_per_ligand_auc"
            ],
            "evaluated_ligands": auc_summary["evaluated_ligands"],
            "evaluated_pairs": auc_summary["positive_negative_pairs"],
        }
        for method, bootstrap in auc_summary["uncertainty"].items():
            auc_row[f"{method}_interval_95_low"] = bootstrap["interval_95"][0]
            auc_row[f"{method}_interval_95_high"] = bootstrap["interval_95"][1]
        metric_rows.append(auc_row)

    binder_binder_vs_chance = {
        representation: {
            "concordance": estimates[representation]["0.0"]["both_uncensored"][
                "mean_per_ligand_pairwise_concordance"
            ],
            "difference_from_chance_0_5": estimates[representation]["0.0"][
                "both_uncensored"
            ]["difference_from_chance_0_5"],
            "uncertainty": estimates[representation]["0.0"]["both_uncensored"].get(
                "uncertainty"
            ),
        }
        for representation in reported_representations
    }
    primary_margin_key = "0.0"
    primary_stratum = "all_informative"
    target_prior_primary = estimates["docking_target_prior"][primary_margin_key][
        primary_stratum
    ]
    return (
        {
            "targets": target_columns,
            "fit": fit,
            "representations": estimates,
            "paired_comparisons": contrasts,
            "docking_target_prior_baseline": {
                "definition": (
                    "ligand-independent target ordering obtained by assigning every "
                    "evaluation ligand the target means fitted on the leakage-free "
                    "DOCKSTRING reference pool"
                ),
                "primary_margin_pkd": 0.0,
                "primary_pair_stratum": primary_stratum,
                "mean_per_ligand_pairwise_concordance": target_prior_primary[
                    "mean_per_ligand_pairwise_concordance"
                ],
                "mean_within_ligand_spearman": target_prior_primary[
                    "mean_within_ligand_spearman"
                ],
                "top1_accuracy_allowing_experimental_ties": target_prior_primary[
                    "top1_accuracy_allowing_experimental_ties"
                ],
                "evaluated_ligands": target_prior_primary["evaluated_ligands"],
                "evaluated_pairs": target_prior_primary["evaluated_pairs"],
                "uncertainty": target_prior_primary.get("uncertainty"),
            },
            "operational_path_steps": {
                "status": (
                    "exploratory post-hoc decomposition; intervals are paired but "
                    "unadjusted for multiplicity, and the sequential steps are not "
                    "independent tests"
                ),
                "contrast_orientation": (
                    "paired mean within-ligand concordance of the first "
                    "representation minus the second"
                ),
                "primary_margin_pkd": 0.0,
                "primary_pair_stratum": primary_stratum,
                "contrast_keys": dict(OPERATIONAL_PATH_STEPS),
                "primary_results": {
                    step: contrasts[contrast_key][primary_margin_key][primary_stratum]
                    for step, contrast_key in OPERATIONAL_PATH_STEPS.items()
                },
                "full_results_location": (
                    "primary.paired_comparisons contains every margin and each of "
                    "all_informative, both_uncensored, and floor_vs_uncensored strata"
                ),
            },
            "co_primary_hit_detection_auc": hit_detection,
            "pair_strata_status": {
                "all_informative": (
                    "primary exact-non-tie observed-pair concordance; pairs in which "
                    "both DAVIS cells are floor-censored are excluded"
                ),
                "both_uncensored": (
                    "exploratory outcome-conditioned stratum; uncertainty intervals "
                    "are unadjusted for multiplicity"
                ),
                "floor_vs_uncensored": (
                    "exploratory outcome-conditioned hit-detection stratum; uncertainty "
                    "intervals are unadjusted for multiplicity"
                ),
            },
            "co_primary_hit_detection_auc_status": (
                "retained schema name for compatibility; this outcome-conditioned "
                "diagnostic is exploratory and its intervals are unadjusted for "
                "multiplicity"
            ),
            "co_primary_hit_detection_auc_paired_comparisons": {
                name: values["0.0"]["floor_vs_uncensored"]
                for name, values in contrasts.items()
            },
            "exploratory_binder_binder_concordance_vs_chance": binder_binder_vs_chance,
            "binder_binder_status": (
                "exploratory and outcome-conditioned because membership requires both "
                "target affinities to be uncensored in DAVIS; intervals are unadjusted "
                "for multiplicity"
            ),
            "chemical_clusters": {
                "murcko_clusters": int(len(pd.unique(murcko))),
                "butina_clusters": int(len(pd.unique(butina))),
                "butina_definition": (
                    "RDKit Morgan fingerprints, radius 2, 2048 bits; Butina clustering "
                    "at Tanimoto similarity >= 0.65 (distance <= 0.35)"
                ),
            },
            "unscaled_row_centering_ranking_invariant": {
                "target_centered_concordance": centered_metric[
                    "mean_per_ligand_pairwise_concordance"
                ],
                "two_way_centered_concordance": two_way_metric[
                    "mean_per_ligand_pairwise_concordance"
                ],
                "difference": float(invariant_difference),
            },
        },
        metric_rows,
        vectors,
    )


def _target_quality_table(
    experiment: pd.DataFrame,
    censored: pd.DataFrame,
) -> pd.DataFrame:
    records = []
    for target in experiment.columns:
        values = experiment[target].to_numpy(dtype=float)
        uncensored = ~censored[target].to_numpy(dtype=bool)
        active = values[uncensored]
        records.append(
            {
                "target": target,
                "davis_protein": TARGET_MAP[target],
                "n_ligands": int(len(values)),
                "uncensored_cells": int(uncensored.sum()),
                "floor_cells": int((~uncensored).sum()),
                "floor_fraction": float((~uncensored).mean()),
                "unique_pkd_values": int(len(np.unique(values))),
                "pkd_minimum": float(values.min()),
                "pkd_maximum": float(values.max()),
                "pkd_range": float(values.max() - values.min()),
                "uncensored_pkd_standard_deviation": (
                    float(active.std(ddof=1)) if len(active) > 1 else None
                ),
            }
        )
    return pd.DataFrame(records)


def _target_quality_sensitivities(
    reference: pd.DataFrame,
    docking: pd.DataFrame,
    experiment: pd.DataFrame,
    censored: pd.DataFrame,
    target_quality: pd.DataFrame,
) -> tuple[dict, list[dict]]:
    filters: OrderedDict[str, list[str]] = OrderedDict()
    filters["all_21_targets"] = list(experiment.columns)
    for threshold in (5, 10, 15, 20):
        filters[f"at_least_{threshold}_uncensored_ligands"] = target_quality.loc[
            target_quality.uncensored_cells >= threshold, "target"
        ].tolist()
    report: dict[str, dict] = {}
    records: list[dict] = []
    for label, targets in filters.items():
        if len(targets) < 3:
            continue
        representations, _ = _score_representations(
            reference[targets].to_numpy(dtype=float),
            docking[targets].to_numpy(dtype=float),
        )
        values = experiment[targets].to_numpy(dtype=float)
        metrics: dict[str, float | None] = {}
        vectors: dict[str, np.ndarray] = {}
        for representation in CORE_REPRESENTATIONS:
            summary, vector = _pairwise_metrics(
                representations[representation],
                values,
                censored[targets].to_numpy(dtype=bool),
                0.0,
                "all_informative",
            )
            metrics[representation] = summary["mean_per_ligand_pairwise_concordance"]
            vectors[representation] = vector
        difference = vectors["two_way_residual"] - vectors["absolute_vina"]
        report[label] = {
            "n_targets": int(len(targets)),
            "targets": targets,
            "representations": metrics,
            "two_way_residual_minus_absolute_vina": _finite_float(
                np.nanmean(difference)
            ),
        }
        records.append({"filter": label, **report[label]})
    return report, records


def _target_jackknife(
    reference: pd.DataFrame,
    docking: pd.DataFrame,
    experiment: pd.DataFrame,
    censored: pd.DataFrame,
) -> tuple[dict, list[dict]]:
    full_representations, _ = _score_representations(
        reference.to_numpy(dtype=float), docking.to_numpy(dtype=float)
    )
    strata = ("all_informative", "both_uncensored", "floor_vs_uncensored")
    full_metrics: dict[str, dict[str, float | None]] = {}
    full_differences: dict[str, float] = {}
    for stratum in strata:
        full_vectors = {}
        full_metrics[stratum] = {}
        for representation in CORE_REPRESENTATIONS:
            summary, vector = _pairwise_metrics(
                full_representations[representation],
                experiment.to_numpy(dtype=float),
                censored.to_numpy(dtype=bool),
                0.0,
                stratum,
            )
            full_metrics[stratum][representation] = summary[
                "mean_per_ligand_pairwise_concordance"
            ]
            full_vectors[representation] = vector
        full_differences[stratum] = float(
            np.nanmean(
                full_vectors["two_way_residual"] - full_vectors["absolute_vina"]
            )
        )
    records: list[dict] = []
    for target in experiment.columns:
        keep = [column for column in experiment.columns if column != target]
        representations, _ = _score_representations(
            reference[keep].to_numpy(dtype=float),
            docking[keep].to_numpy(dtype=float),
        )
        record: dict[str, float | str | None] = {"deleted_target": target}
        for stratum in strata:
            vectors = {}
            for representation in CORE_REPRESENTATIONS:
                summary, vector = _pairwise_metrics(
                    representations[representation],
                    experiment[keep].to_numpy(dtype=float),
                    censored[keep].to_numpy(dtype=bool),
                    0.0,
                    stratum,
                )
                record[f"{stratum}__{representation}"] = summary[
                    "mean_per_ligand_pairwise_concordance"
                ]
                vectors[representation] = vector
            record[
                f"{stratum}__two_way_residual_minus_absolute_vina"
            ] = float(
                np.nanmean(
                    vectors["two_way_residual"] - vectors["absolute_vina"]
                )
            )
        records.append(record)

    strata_reports = {}
    for stratum in strata:
        field = f"{stratum}__two_way_residual_minus_absolute_vina"
        values = np.asarray([record[field] for record in records], dtype=float)
        p = len(values)
        leave_one_out_mean = float(values.mean())
        full_difference = full_differences[stratum]
        jackknife_estimate = float(
            p * full_difference - (p - 1) * leave_one_out_mean
        )
        standard_error = float(
            np.sqrt((p - 1) / p * np.square(values - leave_one_out_mean).sum())
        )
        strata_reports[stratum] = {
            "full_panel": {
                **full_metrics[stratum],
                "two_way_residual_minus_absolute_vina": full_difference,
            },
            "leave_one_target_out_minimum": float(values.min()),
            "leave_one_target_out_median": float(np.median(values)),
            "leave_one_target_out_maximum": float(values.max()),
            "jackknife_bias_corrected_difference": jackknife_estimate,
            "jackknife_standard_error": standard_error,
            "jackknife_bias_corrected_normal_95_interval": [
                jackknife_estimate - 1.96 * standard_error,
                jackknife_estimate + 1.96 * standard_error,
            ],
            "full_estimate_plus_minus_1_96_jackknife_se": [
                full_difference - 1.96 * standard_error,
                full_difference + 1.96 * standard_error,
            ],
            "most_influential_deleted_target": records[
                int(np.argmax(np.abs(values - full_difference)))
            ]["deleted_target"],
        }
    report = {
        "by_pair_stratum": strata_reports,
        "interpretation": (
            "delete-one-target composition sensitivity with all offsets, row means, and "
            "target-specific scales refitted on each 20-target reference panel"
        ),
    }
    return report, records


def _variant_point_estimate(
    reference: pd.DataFrame,
    docking: pd.DataFrame,
    experiment: pd.DataFrame,
    censored: pd.DataFrame,
) -> dict:
    representations, _ = _score_representations(
        reference.to_numpy(dtype=float), docking.to_numpy(dtype=float)
    )
    metrics = {}
    vectors = {}
    for representation in (*CORE_REPRESENTATIONS, *BASELINE_REPRESENTATIONS):
        summary, vector = _pairwise_metrics(
            representations[representation],
            experiment.to_numpy(dtype=float),
            censored.to_numpy(dtype=bool),
            0.0,
            "all_informative",
        )
        metrics[representation] = {
            key: value
            for key, value in summary.items()
            if key
            in {
                "mean_per_ligand_pairwise_concordance",
                "pair_weighted_concordance",
                "mean_within_ligand_spearman",
                "top1_accuracy_allowing_experimental_ties",
                "evaluated_pairs",
            }
        }
        vectors[representation] = vector
    metrics["paired_comparisons"] = {
        f"{first}_minus_{second}": _finite_float(
            np.nanmean(vectors[first] - vectors[second])
        )
        for first, second in PAIRWISE_CONTRASTS
    }
    return metrics


def _synthetic_recensoring_sweep(
    reference: pd.DataFrame,
    docking: pd.DataFrame,
    experiment: pd.DataFrame,
    original_censored: pd.DataFrame,
) -> tuple[dict, list[dict]]:
    """Stress-test conclusions after imposing progressively higher pKd floors."""
    representations, _ = _score_representations(
        reference.to_numpy(dtype=float), docking.to_numpy(dtype=float)
    )
    values = experiment.to_numpy(dtype=float)
    records: list[dict] = []
    report = {}
    for floor in (5.0, 5.5, 6.0, 6.5, 7.0):
        recensored = values <= floor + 1e-12
        recensored_values = np.maximum(values, floor)
        if floor == 5.0 and not np.array_equal(
            recensored, original_censored.to_numpy(dtype=bool)
        ):
            raise ValueError(
                "Synthetic pKd=5 censoring does not reproduce the raw-affinity flag"
            )
        floor_report: dict[str, dict] = {}
        stratum_differences: dict[str, float] = {}
        for representation in (*CORE_REPRESENTATIONS, *BASELINE_REPRESENTATIONS):
            floor_report[representation] = {}
            for stratum in (
                "all_informative",
                "both_uncensored",
                "floor_vs_uncensored",
            ):
                summary, _ = _pairwise_metrics(
                    representations[representation],
                    recensored_values,
                    recensored,
                    0.0,
                    stratum,
                )
                floor_report[representation][stratum] = summary
                records.append(
                    {
                        "synthetic_pkd_floor": floor,
                        "representation": representation,
                        "pair_stratum": stratum,
                        **summary,
                    }
                )
        for stratum in (
            "all_informative",
            "both_uncensored",
            "floor_vs_uncensored",
        ):
            stratum_differences[stratum] = float(
                floor_report["two_way_residual"][stratum][
                    "mean_per_ligand_pairwise_concordance"
                ]
                - floor_report["absolute_vina"][stratum][
                    "mean_per_ligand_pairwise_concordance"
                ]
            )
        floor_report["two_way_residual_minus_absolute_vina_by_stratum"] = (
            stratum_differences
        )
        floor_report["stratum_by_representation_interaction"] = {
            "residual_advantage_binder_binder_minus_hit_detection": float(
                stratum_differences["both_uncensored"]
                - stratum_differences["floor_vs_uncensored"]
            ),
            "interpretation": (
                "difference in residual-minus-absolute concordance between pairs with "
                "two affinities above the synthetic floor and pairs crossing that floor"
            ),
        }
        report[f"{floor:.1f}"] = floor_report
    return report, records


def _size_matched_reference_sensitivity(
    dockstring: pd.DataFrame,
    evaluation_related: pd.Series,
    davis_identity: pd.DataFrame,
    primary_docking: pd.DataFrame,
    primary_experiment: pd.DataFrame,
    primary_censored: pd.DataFrame,
    neighbour_counts: tuple[int, ...] = (50, 100, 250),
) -> dict:
    """Refit score calibration on an MW/heavy-atom matched reference support."""
    descriptor_columns = ["molecular_weight", "heavy_atom_count"]
    pool = dockstring.loc[~evaluation_related].copy()
    if pool[descriptor_columns].isna().any().any():
        return {
            "status": "not_run_in_nonexhaustive_identity_smoke_mode",
            "release_result_available": False,
        }
    davis = davis_identity.set_index("standard_inchikey")
    evaluation_descriptors = davis.reindex(primary_docking.index)[descriptor_columns]
    if evaluation_descriptors.isna().any().any():
        raise ValueError("A primary DAVIS ligand lacks size descriptors")
    pool_values = pool[descriptor_columns].to_numpy(dtype=float)
    evaluation_values = evaluation_descriptors.to_numpy(dtype=float)
    location = pool_values.mean(axis=0)
    scale = pool_values.std(axis=0, ddof=1)
    pool_standardized = (pool_values - location) / scale
    evaluation_standardized = (evaluation_values - location) / scale
    maximum_neighbours = max(neighbour_counts)
    ordered_neighbours: list[np.ndarray] = []
    for anchor in evaluation_standardized:
        distance_squared = np.square(pool_standardized - anchor).sum(axis=1)
        nearest = np.argpartition(
            distance_squared, maximum_neighbours - 1
        )[:maximum_neighbours]
        nearest = nearest[np.argsort(distance_squared[nearest], kind="stable")]
        ordered_neighbours.append(nearest)

    def standardized_mean_difference(values: np.ndarray) -> list[float]:
        return [
            float(value)
            for value in ((values.mean(axis=0) - evaluation_values.mean(axis=0)) / scale)
        ]

    sensitivity = {}
    for neighbours in neighbour_counts:
        selected_positions = np.concatenate(
            [values[:neighbours] for values in ordered_neighbours]
        )
        matched = pool.iloc[selected_positions]
        matched_reference = matched[list(TARGET_MAP)].clip(upper=0.0)
        estimates = _variant_point_estimate(
            matched_reference,
            primary_docking.clip(upper=0.0),
            primary_experiment,
            primary_censored,
        )
        sensitivity[str(neighbours)] = {
            "neighbours_per_evaluation_ligand": int(neighbours),
            "matched_reference_rows_with_reuse": int(len(matched)),
            "matched_reference_unique_rows": int(matched.dockstring_row.nunique()),
            "standardized_mean_difference_matched_reference_minus_evaluation": (
                standardized_mean_difference(
                    matched[descriptor_columns].to_numpy(dtype=float)
                )
            ),
            "representations": estimates,
        }
    primary = sensitivity["100"]
    return {
        "status": "complete",
        "release_result_available": True,
        "matching_variables": descriptor_columns,
        "primary_neighbours_per_evaluation_ligand": 100,
        "sampling": (
            "100 nearest complete-reference profiles per evaluation ligand in z-scored "
            "molecular-weight/heavy-atom space; reference rows may be reused across anchors"
        ),
        "matched_reference_rows_with_reuse": primary[
            "matched_reference_rows_with_reuse"
        ],
        "matched_reference_unique_rows": primary["matched_reference_unique_rows"],
        "standardized_mean_difference_full_reference_minus_evaluation": (
            standardized_mean_difference(pool_values)
        ),
        "standardized_mean_difference_matched_reference_minus_evaluation": (
            primary["standardized_mean_difference_matched_reference_minus_evaluation"]
        ),
        "representations": primary["representations"],
        "neighbour_count_sensitivity": sensitivity,
    }


def dense_davis_dockstring_benchmark(
    dockstring_path: str | Path = DEFAULT_DOCKSTRING,
    davis_path: str | Path = DEFAULT_DAVIS,
    *,
    bootstrap_repeats: int = 5000,
    seed: int = 0,
    identity_scan: str = "full",
) -> dict:
    """Build the dense 21-target benchmark and return JSON-ready records.

    Parameters
    ----------
    identity_scan:
        ``"full"`` (required for release results) recomputes Standard InChIKeys
        and canonical graphs for all 260,060 DOCKSTRING SMILES.  The faster
        ``"reported_connectivity_prefilter"`` is only a smoke-test mode.
    """
    dockstring_path = Path(dockstring_path)
    davis_path = Path(davis_path)
    dockstring, davis_identity, davis_experiment, davis_censored = _load_inputs(
        dockstring_path, davis_path, identity_scan
    )
    targets = list(TARGET_MAP)

    # Conservative common reference: remove all released DOCKSTRING rows whose
    # recomputed connectivity block occurs anywhere in the 72-drug DAVIS panel.
    davis_blocks = set(davis_identity.connectivity_block)
    evaluation_related = dockstring.connectivity_block.isin(davis_blocks)
    reference_clipped = dockstring.loc[~evaluation_related, targets].clip(upper=0.0)
    reference_unclipped = dockstring.loc[~evaluation_related, targets]

    (
        primary_docking,
        primary_experiment,
        primary_censored,
        primary_smiles,
    ) = _matched_surface(
        dockstring,
        davis_identity,
        davis_experiment,
        davis_censored,
        "standard_inchikey",
        "median",
    )
    primary_docking_clipped = primary_docking.clip(upper=0.0)
    primary, primary_rows, _ = _panel_analysis(
        reference_clipped,
        primary_docking_clipped,
        primary_experiment,
        primary_censored,
        primary_smiles.smiles,
        targets,
        bootstrap_repeats,
        seed,
        True,
    )

    target_quality = _target_quality_table(primary_experiment, primary_censored)
    quality_report, quality_records = _target_quality_sensitivities(
        reference_clipped,
        primary_docking_clipped,
        primary_experiment,
        primary_censored,
        target_quality,
    )
    jackknife_report, jackknife_records = _target_jackknife(
        reference_clipped,
        primary_docking_clipped,
        primary_experiment,
        primary_censored,
    )
    recensoring_report, recensoring_records = _synthetic_recensoring_sweep(
        reference_clipped,
        primary_docking_clipped,
        primary_experiment,
        primary_censored,
    )
    size_matched_calibration = _size_matched_reference_sensitivity(
        dockstring,
        evaluation_related,
        davis_identity,
        primary_docking,
        primary_experiment,
        primary_censored,
    )

    match_sensitivities = {}
    match_records = []
    for label, identity_column in MATCH_VARIANTS.items():
        docking, experiment, censored, _ = _matched_surface(
            dockstring,
            davis_identity,
            davis_experiment,
            davis_censored,
            identity_column,
            "median",
        )
        estimates = _variant_point_estimate(
            reference_clipped, docking.clip(upper=0.0), experiment, censored
        )
        matched_rows = dockstring[identity_column].isin(docking.index)
        record = {
            "match_variant": label,
            "identity_column": identity_column,
            "matched_ligands_after_identity_collapse": int(len(docking)),
            "matched_dockstring_rows_before_identity_collapse": int(matched_rows.sum()),
            "representations": estimates,
        }
        match_sensitivities[label] = record
        match_records.append(record)

    duplicate_sensitivity = {}
    duplicate_records = []
    for aggregation in ("median", "mean", "first"):
        docking, experiment, censored, _ = _matched_surface(
            dockstring,
            davis_identity,
            davis_experiment,
            davis_censored,
            "standard_inchikey",
            aggregation,
        )
        estimates = _variant_point_estimate(
            reference_clipped, docking.clip(upper=0.0), experiment, censored
        )
        duplicate_sensitivity[aggregation] = estimates
        duplicate_records.append(
            {"duplicate_aggregation": aggregation, "representations": estimates}
        )

    unclipped = _variant_point_estimate(
        reference_unclipped, primary_docking, primary_experiment, primary_censored
    )

    primary_identity_values = set(primary_docking.index)
    mapping_records = []
    davis_by_key = davis_identity.set_index("standard_inchikey")
    for key in sorted(primary_identity_values):
        ds_rows = dockstring[dockstring.standard_inchikey.eq(key)]
        davis_row = davis_by_key.loc[key]
        if isinstance(davis_row, pd.DataFrame):
            davis_row = davis_row.iloc[0]
        score_range = (
            ds_rows[targets].max(axis=0) - ds_rows[targets].min(axis=0)
        ).max()
        mapping_records.append(
            {
                "drug_name": str(davis_row.drug_name),
                "standard_inchikey": key,
                "connectivity_block": key[:14],
                "davis_smiles": str(davis_row.compound_iso_smiles),
                "dockstring_rows": int(len(ds_rows)),
                "dockstring_row_indices": ";".join(
                    str(value) for value in ds_rows.dockstring_row.tolist()
                ),
                "dockstring_reported_inchikeys": ";".join(
                    ds_rows.inchikey.astype(str).tolist()
                ),
                "maximum_target_score_range_across_duplicate_rows": float(score_range),
            }
        )

    report = {
        "benchmark": "dense_DAVIS_x_DOCKSTRING_21_target_observed_pair_concordance",
        "identity_scan": identity_scan,
        "release_eligible_identity_scan": bool(identity_scan == "full"),
        "analysis_parameters": {
            "seed": int(seed),
            "bootstrap_repeats": int(bootstrap_repeats),
            "identity_scan": identity_scan,
            "positive_score_handling": "clip values above zero to zero",
            "acyclic_cluster_rule": "recomputed Standard-InChI connectivity block",
        },
        "source_support": {
            "davis_ligands": int(davis_identity.drug_name.nunique()),
            "davis_targets_total": 442,
            "shared_targets": int(len(targets)),
            "shared_target_map": dict(TARGET_MAP),
            "dockstring_rows": int(len(dockstring)),
            "reference_rows_after_excluding_all_DAVIS_connectivity_blocks": int(
                len(reference_clipped)
            ),
            "dockstring_rows_excluded_from_reference": int(evaluation_related.sum()),
            "primary_full_key_ligands": int(len(primary_docking)),
            "primary_full_key_dockstring_rows_before_median_collapse": int(
                dockstring.standard_inchikey.isin(primary_identity_values).sum()
            ),
            "primary_experimental_cells": int(primary_experiment.size),
            "primary_floor_cells_pkd_5": int(
                primary_censored.to_numpy(dtype=bool).sum()
            ),
            "primary_uncensored_cells": int(
                (~primary_censored.to_numpy(dtype=bool)).sum()
            ),
        },
        "primary_estimand": (
            "equal-ligand mean of within-ligand pairwise concordance across the dense "
            "59 x 21 full-Standard-InChIKey support; lower DOCKSTRING scores predict "
            "higher DAVIS pKd; exact pKd ties and every pair in which both cells are "
            "floor-censored are excluded; score ties receive half credit"
        ),
        "primary_match": (
            "full Standard InChIKey recomputed with the pinned RDKit version from both "
            "released SMILES fields; duplicate raw DOCKSTRING profiles collapsed by median"
        ),
        "reference_fit_boundary": (
            "target offsets, raw target scales, residual target scales, and grand mean are "
            "fit on the complete DOCKSTRING 21-target surface after excluding every row "
            "whose recomputed Standard InChI connectivity block occurs in any DAVIS ligand"
        ),
        "censoring": (
            "The censor flag is defined explicitly as released raw affinity = 10,000 nM, "
            "not inferred from floating-point pKd. Assertions require every flagged cell "
            "to equal pKd=5, every unflagged cell to exceed 5, and two flagged cells never "
            "to enter an informative pair. Results are stratified as floor-versus-"
            "uncensored and both-uncensored pairs."
        ),
        "primary": primary,
        "target_quality_sensitivities": quality_report,
        "target_jackknife": jackknife_report,
        "synthetic_recensoring_sweep": recensoring_report,
        "molecular_size_matched_reference_calibration": size_matched_calibration,
        "molecular_identity_sensitivities": match_sensitivities,
        "duplicate_raw_profile_sensitivities": duplicate_sensitivity,
        "unclipped_score_sensitivity": unclipped,
        "boundary": (
            "This is a dense observed-pair concordance benchmark on 21 kinases already "
            "present in DOCKSTRING, not retrieval of unmeasured targets and not a benchmark "
            "of docking pose accuracy. The target jackknife describes composition sensitivity "
            "conditional on these 21 mapped targets."
        ),
        "tables": {
            "primary_metrics": primary_rows,
            "target_quality": target_quality.to_dict(orient="records"),
            "target_quality_sensitivities": quality_records,
            "target_jackknife": jackknife_records,
            "synthetic_recensoring_sweep": recensoring_records,
            "molecular_identity_sensitivities": match_records,
            "duplicate_profile_sensitivities": duplicate_records,
            "primary_molecule_mapping": mapping_records,
        },
    }
    # Enforce strict JSON serialisability (no numpy scalar and no non-finite float).
    json.dumps(report, allow_nan=False)
    return report


def write_outputs(report: dict, output_directory: str | Path) -> None:
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    (output / "dense_davis_benchmark.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    for name, records in report["tables"].items():
        frame = pd.json_normalize(records, sep=".")
        frame.to_csv(output / f"dense_davis_{name}.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dockstring", type=Path, default=DEFAULT_DOCKSTRING)
    parser.add_argument("--davis", type=Path, default=DEFAULT_DAVIS)
    parser.add_argument("--output-dir", type=Path, default=PACKAGE / "results")
    parser.add_argument("--bootstrap-repeats", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--identity-scan",
        choices=["full", "reported_connectivity_prefilter"],
        default="full",
    )
    arguments = parser.parse_args()
    report = dense_davis_dockstring_benchmark(
        arguments.dockstring,
        arguments.davis,
        bootstrap_repeats=arguments.bootstrap_repeats,
        seed=arguments.seed,
        identity_scan=arguments.identity_scan,
    )
    write_outputs(report, arguments.output_dir)
    primary = report["primary"]["representations"]
    print(
        json.dumps(
            {
                "ligands": report["source_support"]["primary_full_key_ligands"],
                "targets": report["source_support"]["shared_targets"],
                "absolute_vina": primary["absolute_vina"]["0.0"][
                    "all_informative"
                ]["mean_per_ligand_pairwise_concordance"],
                "column_standardized": primary["column_standardized"]["0.0"][
                    "all_informative"
                ]["mean_per_ligand_pairwise_concordance"],
                "two_way_residual": primary["two_way_residual"]["0.0"][
                    "all_informative"
                ]["mean_per_ligand_pairwise_concordance"],
                "release_eligible_identity_scan": report[
                    "release_eligible_identity_scan"
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
