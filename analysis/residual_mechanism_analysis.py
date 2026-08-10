#!/usr/bin/env python3
"""Mechanistic audit of the residual spectra in the two large Vina matrices.

This analysis keeps three claims separate:

1. an exact algebraic account of the PR change in terms of pairwise target
   correlations;
2. out-of-fold evidence that ligand physicochemical descriptors account for a
   reproducible part of the two-way-residual structure; and
3. an explicitly exploratory target-level association with pocket volume.

All transformations used to predict held-out residual scores are fitted within
the corresponding GroupKFold training split.  Docking-44 folds hold out frozen
Butina clusters.  DOCKSTRING folds hold out Bemis--Murcko scaffolds, with each
acyclic molecule treated as a singleton group.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, Descriptors, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from scipy import stats
from scipy.optimize import linear_sum_assignment
from sklearn.model_selection import GroupKFold


RDLogger.DisableLog("rdApp.*")

PACKAGE = Path(__file__).resolve().parents[1]
FROZEN = PACKAGE / "data" / "frozen"
DEFAULT_OUTPUT = PACKAGE / "results" / "residual_mechanism"

DOCK44 = [
    "1m2z", "1pbq", "1xoq", "2rh1", "2vt4", "2ydo", "2z5x", "3b66",
    "3kk6", "3ln1", "3rze", "4djh", "4ey7", "4iar", "4mqs", "4n6h",
    "5cxv", "5i71", "5tvn", "5u09", "5va1", "6cm4", "6kpf", "6kux",
    "6lqa", "6pdj", "6x3x", "6y1z", "7f8y", "7kwe", "7ljd", "7wc9",
    "7xnk", "7ym8", "8e9y", "8ef6", "8fhs", "8pjk", "8st0", "8wty",
    "8xvk", "8yn3", "9eo4", "V1A",
]

DESCRIPTOR_NAMES = [
    "heavy_atoms",
    "molecular_weight",
    "labute_asa",
    "tpsa",
    "clogp",
    "rotatable_bonds",
    "ring_count",
]


# Nine target identities that can be matched across the independently sourced
# panels.  The PDB-to-UniProt assignments were audited against PDBe/SIFTS v2.
# DOCKSTRING target symbols denote the corresponding reviewed human proteins;
# 2VT4 and 3LN1 use non-human orthologous receptor structures.
CROSS_PANEL_TARGETS = [
    {
        "docking44_target": "1m2z",
        "dockstring_target": "NR3C1",
        "human_uniprot_accession": "P04150",
        "docking44_structure_uniprot_accession": "P04150",
        "docking44_structure_species": "Homo sapiens",
        "same_receptor_structure_in_dockstring": False,
    },
    {
        "docking44_target": "2rh1",
        "dockstring_target": "ADRB2",
        "human_uniprot_accession": "P07550",
        "docking44_structure_uniprot_accession": "P07550",
        "docking44_structure_species": "Homo sapiens",
        "same_receptor_structure_in_dockstring": False,
    },
    {
        "docking44_target": "2vt4",
        "dockstring_target": "ADRB1",
        "human_uniprot_accession": "P08588",
        "docking44_structure_uniprot_accession": "P07700",
        "docking44_structure_species": "Meleagris gallopavo",
        "same_receptor_structure_in_dockstring": True,
    },
    {
        "docking44_target": "2ydo",
        "dockstring_target": "ADORA2A",
        "human_uniprot_accession": "P29274",
        "docking44_structure_uniprot_accession": "P29274",
        "docking44_structure_species": "Homo sapiens",
        "same_receptor_structure_in_dockstring": False,
    },
    {
        "docking44_target": "3b66",
        "dockstring_target": "AR",
        "human_uniprot_accession": "P10275",
        "docking44_structure_uniprot_accession": "P10275",
        "docking44_structure_species": "Homo sapiens",
        "same_receptor_structure_in_dockstring": False,
    },
    {
        "docking44_target": "3ln1",
        "dockstring_target": "PTGS2",
        "human_uniprot_accession": "P35354",
        "docking44_structure_uniprot_accession": "Q05769",
        "docking44_structure_species": "Mus musculus",
        "same_receptor_structure_in_dockstring": True,
    },
    {
        "docking44_target": "4ey7",
        "dockstring_target": "ACHE",
        "human_uniprot_accession": "P22303",
        "docking44_structure_uniprot_accession": "P22303",
        "docking44_structure_species": "Homo sapiens",
        "same_receptor_structure_in_dockstring": False,
    },
    {
        "docking44_target": "6cm4",
        "dockstring_target": "DRD2",
        "human_uniprot_accession": "P14416",
        "docking44_structure_uniprot_accession": "P14416",
        "docking44_structure_species": "Homo sapiens",
        "same_receptor_structure_in_dockstring": True,
    },
    {
        "docking44_target": "6pdj",
        "dockstring_target": "LCK",
        "human_uniprot_accession": "P06239",
        "docking44_structure_uniprot_accession": "P06239",
        "docking44_structure_species": "Homo sapiens",
        "same_receptor_structure_in_dockstring": False,
    },
]


@dataclass(frozen=True)
class AnalysisConfig:
    dockstring_sample_size: int = 15_000
    dockstring_sample_seed: int = 71
    folds: int = 5
    qap_permutations: int = 10_000
    qap_seed: int = 202_608_02
    pocket_permutations: int = 100_000
    pocket_bootstrap_replicates: int = 5_000
    pocket_seed: int = 202_608_03


@dataclass
class DatasetBundle:
    name: str
    spectral_matrix: np.ndarray
    matrix: np.ndarray
    targets: list[str]
    families: list[str]
    smiles: list[str]
    groups: np.ndarray
    descriptor_matrix: np.ndarray
    support_rule: str
    input_rows: int
    missing_cells_before_preprocessing: int
    positive_cells_clipped: int
    group_definition: str
    full_inchikeys: np.ndarray
    connectivity_blocks: np.ndarray
    nonempty_scaffolds: np.ndarray


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_ready(value: Any) -> Any:
    """Convert NumPy containers and scalars to strict JSON-compatible values."""
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return [json_ready(item) for item in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        if not np.isfinite(value):
            raise ValueError("non-finite value cannot be written to strict JSON")
        return value
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(json_ready(payload), indent=2, sort_keys=True, allow_nan=False) + "\n"
    )


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, float_format="%.17g", na_rep="")


def two_way_center(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2 or min(matrix.shape) < 2:
        raise ValueError("matrix must be two-dimensional with at least two rows and columns")
    if not np.isfinite(matrix).all():
        raise ValueError("matrix contains non-finite values")
    return (
        matrix
        - matrix.mean(axis=0, keepdims=True)
        - matrix.mean(axis=1, keepdims=True)
        + matrix.mean()
    )


def orthogonal_two_way_anova(matrix: np.ndarray) -> dict[str, Any]:
    """Exact Frobenius decomposition into ligand, target, and interaction terms."""
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2 or min(matrix.shape) < 2:
        raise ValueError("matrix must be two-dimensional with at least two rows and columns")
    grand_mean = float(matrix.mean())
    grand_centered = matrix - grand_mean
    ligand_vector = matrix.mean(axis=1, keepdims=True) - grand_mean
    target_vector = matrix.mean(axis=0, keepdims=True) - grand_mean
    ligand_component = np.broadcast_to(ligand_vector, matrix.shape)
    target_component = np.broadcast_to(target_vector, matrix.shape)
    interaction_component = two_way_center(matrix)
    components = {
        "ligand_main_effect": ligand_component,
        "target_main_effect": target_component,
        "ligand_target_interaction": interaction_component,
    }
    total_ss = float(np.square(grand_centered).sum())
    if total_ss <= 0:
        raise ValueError("grand-centered matrix has zero sum of squares")
    sums_of_squares = {
        name: float(np.square(component).sum())
        for name, component in components.items()
    }
    fractions = {
        name: value / total_ss for name, value in sums_of_squares.items()
    }
    names = list(components)
    inner_products = {
        f"{names[first]}__{names[second]}": float(
            np.sum(components[names[first]] * components[names[second]])
        )
        for first in range(len(names))
        for second in range(first + 1, len(names))
    }
    reconstructed = sum(components.values())
    main_effect_ss = (
        sums_of_squares["ligand_main_effect"]
        + sums_of_squares["target_main_effect"]
    )
    ligand_plus_interaction = (
        sums_of_squares["ligand_main_effect"]
        + sums_of_squares["ligand_target_interaction"]
    )
    return {
        "total_grand_centered_sum_squares": total_ss,
        "component_sum_squares": sums_of_squares,
        "component_fractions": fractions,
        "combined_ligand_and_target_main_effect_fraction": main_effect_ss / total_ss,
        "ligand_fraction_conditional_on_column_centering": (
            sums_of_squares["ligand_main_effect"] / ligand_plus_interaction
        ),
        "interaction_fraction_conditional_on_column_centering": (
            sums_of_squares["ligand_target_interaction"] / ligand_plus_interaction
        ),
        "fraction_closure_error": float(abs(sum(fractions.values()) - 1.0)),
        "maximum_absolute_pairwise_inner_product_fraction": float(
            max(abs(value) for value in inner_products.values()) / total_ss
        ),
        "maximum_absolute_reconstruction_error": float(
            np.max(np.abs(grand_centered - reconstructed))
        ),
        "pairwise_component_inner_products": inner_products,
    }


def standardize_columns(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    matrix = np.asarray(matrix, dtype=np.float64)
    means = matrix.mean(axis=0)
    standard_deviations = matrix.std(axis=0, ddof=1)
    if np.any(standard_deviations <= 1e-12):
        raise ValueError("matrix contains a target column with negligible variance")
    return (matrix - means) / standard_deviations, means, standard_deviations


def correlation_matrix(matrix: np.ndarray) -> np.ndarray:
    standardized, _, _ = standardize_columns(matrix)
    correlation = standardized.T @ standardized / (len(standardized) - 1)
    correlation = (correlation + correlation.T) / 2.0
    np.fill_diagonal(correlation, 1.0)
    return correlation


def participation_ratio_from_psd(matrix: np.ndarray) -> float:
    matrix = np.asarray(matrix, dtype=np.float64)
    trace = float(np.trace(matrix))
    denominator = float(np.square(matrix).sum())
    if denominator <= 0:
        raise ValueError("PSD matrix has zero squared Frobenius norm")
    return trace * trace / denominator


def correlation_pr(matrix: np.ndarray) -> float:
    return participation_ratio_from_psd(correlation_matrix(matrix))


def covariance_pr(matrix: np.ndarray) -> float:
    centered = np.asarray(matrix, dtype=np.float64) - np.mean(matrix, axis=0)
    covariance = centered.T @ centered / (len(centered) - 1)
    return participation_ratio_from_psd(covariance)


def correlation_moments(matrix: np.ndarray) -> dict[str, float]:
    correlation = correlation_matrix(matrix)
    target_count = len(correlation)
    upper = correlation[np.triu_indices(target_count, 1)]
    pair_mean = float(upper.mean())
    pair_mean_squared = float(np.square(upper).mean())
    pair_variance = float(np.square(upper - pair_mean).mean())
    pr_matrix = participation_ratio_from_psd(correlation)
    pr_identity = target_count / (
        1.0 + (target_count - 1.0) * pair_mean_squared
    )
    return {
        "n_targets": int(target_count),
        "mean_offdiagonal_correlation": pair_mean,
        "mean_squared_offdiagonal_correlation": pair_mean_squared,
        "pairwise_correlation_variance": pair_variance,
        "mean_square_plus_variance": pair_mean * pair_mean + pair_variance,
        "participation_ratio": pr_matrix,
        "participation_ratio_from_correlation_moment_identity": float(pr_identity),
        "maximum_absolute_moment_identity_error": float(
            abs(pair_mean_squared - (pair_mean * pair_mean + pair_variance))
        ),
        "maximum_absolute_pr_identity_error": float(abs(pr_matrix - pr_identity)),
    }


def uniform_axis_diagnostics(matrix: np.ndarray) -> dict[str, float | int]:
    correlation = correlation_matrix(matrix)
    eigenvalues, eigenvectors = np.linalg.eigh(correlation)
    leading_eigenvalue = float(eigenvalues[-1])
    leading = eigenvectors[:, -1]
    uniform = np.ones(len(leading), dtype=np.float64) / np.sqrt(len(leading))
    if float(leading @ uniform) < 0:
        leading = -leading
    cosine = float(leading @ uniform)
    rayleigh = float(uniform @ correlation @ uniform)
    upper = correlation[np.triu_indices(len(correlation), 1)]
    rayleigh_identity = 1.0 + (len(correlation) - 1.0) * float(upper.mean())
    return {
        "n_targets": int(len(correlation)),
        "leading_eigenvalue": leading_eigenvalue,
        "pc1_variance_fraction": leading_eigenvalue / len(correlation),
        "cosine_pc1_with_uniform_vector": cosine,
        "squared_cosine_pc1_with_uniform_vector": cosine * cosine,
        "angle_from_uniform_degrees": float(np.degrees(np.arccos(np.clip(cosine, -1, 1)))),
        "uniform_vector_rayleigh_quotient": rayleigh,
        "uniform_vector_variance_fraction": rayleigh / len(correlation),
        "uniform_rayleigh_as_fraction_of_pc1_eigenvalue": rayleigh / leading_eigenvalue,
        "uniform_rayleigh_identity_error": float(abs(rayleigh - rayleigh_identity)),
        "positive_pc1_loadings": int(np.sum(leading > 0)),
        "negative_pc1_loadings": int(np.sum(leading < 0)),
    }


def molecular_descriptors(smiles: Iterable[str]) -> np.ndarray:
    rows: list[list[float]] = []
    for index, value in enumerate(smiles):
        molecule = Chem.MolFromSmiles(str(value))
        if molecule is None:
            raise ValueError(f"invalid SMILES at row {index}")
        rows.append([
            float(molecule.GetNumHeavyAtoms()),
            float(Descriptors.MolWt(molecule)),
            float(rdMolDescriptors.CalcLabuteASA(molecule)),
            float(Descriptors.TPSA(molecule)),
            float(Crippen.MolLogP(molecule)),
            float(Descriptors.NumRotatableBonds(molecule)),
            float(rdMolDescriptors.CalcNumRings(molecule)),
        ])
    result = np.asarray(rows, dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError("molecular descriptor matrix contains non-finite values")
    return result


def scaffold_keys(smiles: Iterable[str]) -> np.ndarray:
    """RDKit Bemis--Murcko keys; each acyclic molecule is a singleton group."""
    keys: list[str] = []
    for index, value in enumerate(smiles):
        molecule = Chem.MolFromSmiles(str(value))
        if molecule is None:
            raise ValueError(f"invalid SMILES at row {index}")
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=molecule)
        keys.append(scaffold if scaffold else f"ACYCLIC_SINGLETON:{index}")
    return np.asarray(keys, dtype=object)


def chemical_identity_keys(
    smiles: Iterable[str],
    provided_inchikeys: Iterable[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return full InChIKeys, connectivity blocks, and nonempty Murcko keys."""
    smiles_values = [str(value) for value in smiles]
    if provided_inchikeys is None:
        supplied: list[str] | None = None
    else:
        supplied = [str(value) for value in provided_inchikeys]
        if len(supplied) != len(smiles_values):
            raise ValueError("provided InChIKeys and SMILES must have the same length")

    full_keys: list[str] = []
    connectivity: list[str] = []
    scaffolds: list[str] = []
    for index, value in enumerate(smiles_values):
        molecule = Chem.MolFromSmiles(value)
        if molecule is None:
            raise ValueError(f"invalid SMILES at row {index}")
        key = supplied[index] if supplied is not None else Chem.MolToInchiKey(molecule)
        if not key or len(key) < 14:
            raise ValueError(f"invalid InChIKey at row {index}")
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=molecule)
        full_keys.append(key)
        connectivity.append(key[:14])
        scaffolds.append(scaffold)
    return (
        np.asarray(full_keys, dtype=object),
        np.asarray(connectivity, dtype=object),
        np.asarray(scaffolds, dtype=object),
    )


def _target_families(dataset: str, targets: list[str]) -> list[str]:
    manifest = pd.read_csv(PACKAGE / "data" / "target_families.csv")
    selected = manifest.loc[manifest.dataset.eq(dataset)].set_index("target")
    missing = [target for target in targets if target not in selected.index]
    if missing:
        raise ValueError(f"missing target-family labels for {dataset}: {missing}")
    return selected.loc[targets, "family"].astype(str).tolist()


def load_docking44() -> DatasetBundle:
    path = FROZEN / "df_final_v4.csv.gz"
    columns = DOCK44 + [
        "Butina_clusters", "Canonical SMILES", "Cleaned SMILES",
    ]
    frame = pd.read_csv(path, usecols=columns)
    numeric = frame[DOCK44].apply(pd.to_numeric, errors="coerce")
    missing = int(numeric.isna().to_numpy().sum())
    positive = int((numeric.to_numpy(dtype=float) > 0).sum())
    clipped = numeric.clip(upper=0)
    matrix = clipped.fillna(clipped.mean()).to_numpy(dtype=np.float64)
    smiles = frame["Cleaned SMILES"].fillna(frame["Canonical SMILES"]).astype(str)
    if frame["Butina_clusters"].isna().any():
        raise ValueError("Docking-44 contains missing Butina cluster labels")
    descriptors = molecular_descriptors(smiles)
    full_keys, connectivity, nonempty_scaffolds = chemical_identity_keys(smiles)
    return DatasetBundle(
        name="Docking-44",
        spectral_matrix=matrix,
        matrix=matrix,
        targets=list(DOCK44),
        families=_target_families("Docking-44", DOCK44),
        smiles=smiles.tolist(),
        groups=frame["Butina_clusters"].astype(str).to_numpy(),
        descriptor_matrix=descriptors,
        support_rule="all 12,651 analyzable ligands",
        input_rows=int(len(frame)),
        missing_cells_before_preprocessing=missing,
        positive_cells_clipped=positive,
        group_definition="frozen source Butina clusters",
        full_inchikeys=full_keys,
        connectivity_blocks=connectivity,
        nonempty_scaffolds=nonempty_scaffolds,
    )


def load_dockstring(config: AnalysisConfig) -> DatasetBundle:
    path = FROZEN / "dockstring-dataset.tsv.gz"
    frame = pd.read_csv(path, sep="\t")
    targets = [column for column in frame.columns if column not in {"inchikey", "smiles"}]
    numeric = frame[targets].apply(pd.to_numeric, errors="coerce")
    missing = int(numeric.isna().to_numpy().sum())
    complete = ~numeric.isna().any(axis=1)
    complete_numeric = numeric.loc[complete].to_numpy(dtype=np.float64)
    positive = int((complete_numeric > 0).sum())
    complete_numeric = np.minimum(complete_numeric, 0.0)
    complete_smiles = frame.loc[complete, "smiles"].reset_index(drop=True).astype(str)
    complete_inchikeys = (
        frame.loc[complete, "inchikey"].reset_index(drop=True).astype(str)
    )
    if config.dockstring_sample_size > len(complete_numeric):
        raise ValueError("requested DOCKSTRING support exceeds complete rows")
    rng = np.random.default_rng(config.dockstring_sample_seed)
    support = np.sort(
        rng.choice(
            len(complete_numeric), config.dockstring_sample_size, replace=False
        )
    )
    matrix = complete_numeric[support]
    smiles = complete_smiles.iloc[support].reset_index(drop=True)
    inchikeys = complete_inchikeys.iloc[support].reset_index(drop=True)
    descriptors = molecular_descriptors(smiles)
    groups = scaffold_keys(smiles)
    full_keys, connectivity, nonempty_scaffolds = chemical_identity_keys(
        smiles, inchikeys
    )
    return DatasetBundle(
        name="DOCKSTRING-58",
        spectral_matrix=complete_numeric,
        matrix=matrix,
        targets=targets,
        families=_target_families("DOCKSTRING-58", targets),
        smiles=smiles.tolist(),
        groups=groups,
        descriptor_matrix=descriptors,
        support_rule=(
            f"sorted simple random sample without replacement of "
            f"{config.dockstring_sample_size:,} complete ligands; "
            f"NumPy seed {config.dockstring_sample_seed}"
        ),
        input_rows=int(len(frame)),
        missing_cells_before_preprocessing=missing,
        positive_cells_clipped=positive,
        group_definition=(
            "RDKit Bemis-Murcko scaffolds; every acyclic molecule is a singleton"
        ),
        full_inchikeys=full_keys,
        connectivity_blocks=connectivity,
        nonempty_scaffolds=nonempty_scaffolds,
    )


def fold_local_residual_transform(
    train: np.ndarray,
    test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Fit target offsets and residual target scales on train, apply to test."""
    target_offsets = train.mean(axis=0)
    train_target_centered = train - target_offsets
    train_residual = train_target_centered - train_target_centered.mean(
        axis=1, keepdims=True
    )
    residual_scales = train_residual.std(axis=0, ddof=1)
    if np.any(residual_scales <= 1e-12):
        raise ValueError("training fold has a residual target with negligible variance")
    test_target_centered = test - target_offsets
    test_residual = test_target_centered - test_target_centered.mean(
        axis=1, keepdims=True
    )
    return (
        train_residual / residual_scales,
        test_residual / residual_scales,
        {
            "minimum_training_residual_target_sd": float(residual_scales.min()),
            "maximum_training_residual_target_sd": float(residual_scales.max()),
        },
    )


def grouped_descriptor_decomposition(
    matrix: np.ndarray,
    descriptors: np.ndarray,
    groups: np.ndarray,
    n_splits: int,
) -> dict[str, Any]:
    """Strict fold-local, group-held-out linear descriptor decomposition."""
    matrix = np.asarray(matrix, dtype=np.float64)
    descriptors = np.asarray(descriptors, dtype=np.float64)
    groups = np.asarray(groups, dtype=object)
    if len(matrix) != len(descriptors) or len(matrix) != len(groups):
        raise ValueError("matrix, descriptors and groups must have the same rows")
    if len(np.unique(groups)) < n_splits:
        raise ValueError("fewer chemical groups than requested folds")

    oof_surface = np.full_like(matrix, np.nan, dtype=np.float64)
    oof_prediction = np.full_like(matrix, np.nan, dtype=np.float64)
    test_counts = np.zeros(len(matrix), dtype=np.int64)
    fold_rows: list[dict[str, Any]] = []
    splitter = GroupKFold(n_splits=n_splits)

    for fold, (train_index, test_index) in enumerate(
        splitter.split(descriptors, groups=groups), start=1
    ):
        train_groups = set(groups[train_index].tolist())
        test_groups = set(groups[test_index].tolist())
        overlap = train_groups.intersection(test_groups)
        if overlap:
            raise AssertionError("chemical group leakage between train and test fold")

        train_surface, test_surface, scale_record = fold_local_residual_transform(
            matrix[train_index], matrix[test_index]
        )
        descriptor_mean = descriptors[train_index].mean(axis=0)
        descriptor_sd = descriptors[train_index].std(axis=0, ddof=1)
        if np.any(descriptor_sd <= 1e-12):
            raise ValueError("training fold has a constant descriptor")
        train_descriptor = (
            descriptors[train_index] - descriptor_mean
        ) / descriptor_sd
        test_descriptor = (
            descriptors[test_index] - descriptor_mean
        ) / descriptor_sd
        train_design = np.column_stack([
            np.ones(len(train_index), dtype=np.float64), train_descriptor,
        ])
        test_design = np.column_stack([
            np.ones(len(test_index), dtype=np.float64), test_descriptor,
        ])
        coefficients = np.linalg.lstsq(
            train_design, train_surface, rcond=None
        )[0]
        predictions = test_design @ coefficients
        oof_surface[test_index] = test_surface
        oof_prediction[test_index] = predictions
        test_counts[test_index] += 1
        fold_rows.append({
            "fold": fold,
            "train_ligands": int(len(train_index)),
            "test_ligands": int(len(test_index)),
            "train_groups": int(len(train_groups)),
            "test_groups": int(len(test_groups)),
            "group_overlap_count": int(len(overlap)),
            "minimum_training_descriptor_sd": float(descriptor_sd.min()),
            "maximum_training_descriptor_sd": float(descriptor_sd.max()),
            **scale_record,
        })

    if not np.all(test_counts == 1):
        raise AssertionError("every ligand must occur in exactly one test fold")
    if not np.isfinite(oof_surface).all() or not np.isfinite(oof_prediction).all():
        raise AssertionError("out-of-fold matrices were not filled completely")

    oof_error = oof_surface - oof_prediction
    target_sst = np.square(oof_surface - oof_surface.mean(axis=0)).sum(axis=0)
    target_sse = np.square(oof_error).sum(axis=0)
    target_r2 = 1.0 - target_sse / target_sst

    standardized_surface, surface_mean, surface_sd = standardize_columns(oof_surface)
    correlation = standardized_surface.T @ standardized_surface / (
        len(standardized_surface) - 1
    )
    eigenvalues, eigenvectors = np.linalg.eigh(correlation)
    pc1_loading = eigenvectors[:, -1]
    observed_pc1_score = standardized_surface @ pc1_loading
    standardized_prediction = (
        oof_prediction - surface_mean
    ) / surface_sd
    predicted_pc1_score = standardized_prediction @ pc1_loading
    pc1_sst = float(
        np.square(observed_pc1_score - observed_pc1_score.mean()).sum()
    )
    pc1_sse = float(np.square(observed_pc1_score - predicted_pc1_score).sum())
    pc1_r2 = 1.0 - pc1_sse / pc1_sst

    base_moments = correlation_moments(oof_surface)
    error_moments = correlation_moments(oof_error)
    base_pr = base_moments["participation_ratio"]
    error_pr = error_moments["participation_ratio"]
    base_mean_r2 = base_moments["mean_squared_offdiagonal_correlation"]
    error_mean_r2 = error_moments["mean_squared_offdiagonal_correlation"]
    return {
        "oof_surface": oof_surface,
        "oof_prediction": oof_prediction,
        "oof_error": oof_error,
        "target_r2": target_r2,
        "fold_rows": fold_rows,
        "metrics": {
            "folds": int(n_splits),
            "chemical_groups": int(len(np.unique(groups))),
            "strict_fold_local_target_offsets": True,
            "strict_fold_local_residual_target_scales": True,
            "strict_fold_local_descriptor_scaling": True,
            "strict_fold_local_regression_coefficients": True,
            "mean_out_of_fold_target_r2": float(target_r2.mean()),
            "median_out_of_fold_target_r2": float(np.median(target_r2)),
            "minimum_out_of_fold_target_r2": float(target_r2.min()),
            "maximum_out_of_fold_target_r2": float(target_r2.max()),
            "out_of_fold_residual_pc1_r2": float(pc1_r2),
            "out_of_fold_pr_before_descriptor_removal": float(base_pr),
            "out_of_fold_pr_after_descriptor_removal": float(error_pr),
            "out_of_fold_pr_increase": float(error_pr - base_pr),
            "out_of_fold_pr_ratio": float(error_pr / base_pr),
            "out_of_fold_mean_squared_target_correlation_before_descriptor_removal": (
                float(base_mean_r2)
            ),
            "out_of_fold_mean_squared_target_correlation_after_descriptor_removal": (
                float(error_mean_r2)
            ),
            "out_of_fold_absolute_mean_squared_target_correlation_reduction": float(
                base_mean_r2 - error_mean_r2
            ),
            "out_of_fold_relative_mean_squared_target_correlation_reduction": float(
                (base_mean_r2 - error_mean_r2) / base_mean_r2
            ),
            "descriptor_removal_interpretation": (
                "Removing the group-held-out, fold-locally predicted descriptor "
                "component reduces mean squared target correlation; this is a "
                "predictive decomposition of score structure, not a causal attribution"
            ),
            "pc1_definition_scope": (
                "PC1 loadings are descriptive loadings of the pooled out-of-fold "
                "surface; every ligand-level descriptor prediction is group-held-out"
            ),
        },
    }


def signed_family_statistic(
    correlation: np.ndarray,
    families: np.ndarray,
) -> dict[str, float | int]:
    first, second = np.triu_indices(len(families), 1)
    within = families[first] == families[second]
    values = correlation[first, second]
    if not np.any(within) or np.all(within):
        raise ValueError("family labels do not define both within- and between-family pairs")
    within_mean = float(values[within].mean())
    between_mean = float(values[~within].mean())
    return {
        "within_family_pairs": int(within.sum()),
        "between_family_pairs": int((~within).sum()),
        "mean_signed_within_family_correlation": within_mean,
        "mean_signed_between_family_correlation": between_mean,
        "signed_within_minus_between": within_mean - between_mean,
    }


def family_qap(
    before: np.ndarray,
    after: np.ndarray,
    families: list[str],
    permutations: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Target-label QAP for signed within- versus between-family correlation."""
    before_correlation = correlation_matrix(before)
    after_correlation = correlation_matrix(after)
    family_array = np.asarray(families, dtype=object)
    observed_before = signed_family_statistic(before_correlation, family_array)
    observed_after = signed_family_statistic(after_correlation, family_array)
    observed = np.asarray([
        observed_before["signed_within_minus_between"],
        observed_after["signed_within_minus_between"],
        observed_after["signed_within_minus_between"]
        - observed_before["signed_within_minus_between"],
    ], dtype=np.float64)

    null = np.empty((permutations, 3), dtype=np.float64)
    rng = np.random.default_rng(seed)
    for repetition in range(permutations):
        permuted = rng.permutation(family_array)
        before_stat = signed_family_statistic(before_correlation, permuted)[
            "signed_within_minus_between"
        ]
        after_stat = signed_family_statistic(after_correlation, permuted)[
            "signed_within_minus_between"
        ]
        null[repetition] = [
            before_stat, after_stat, after_stat - before_stat,
        ]

    labels = [
        "before_descriptor_removal",
        "after_descriptor_removal",
        "after_minus_before",
    ]
    observed_records = [observed_before, observed_after, {
        "within_family_pairs": observed_before["within_family_pairs"],
        "between_family_pairs": observed_before["between_family_pairs"],
        "mean_signed_within_family_correlation": np.nan,
        "mean_signed_between_family_correlation": np.nan,
        "signed_within_minus_between": observed[2],
    }]
    rows: list[dict[str, Any]] = []
    for index, label in enumerate(labels):
        values = null[:, index]
        record = observed_records[index]
        rows.append({
            "surface_or_contrast": label,
            **record,
            "target_label_permutations": int(permutations),
            "qap_seed": int(seed),
            "qap_null_mean": float(values.mean()),
            "qap_null_q025": float(np.quantile(values, 0.025)),
            "qap_null_q975": float(np.quantile(values, 0.975)),
            "qap_one_sided_p_for_positive_statistic": float(
                (1 + np.sum(values >= observed[index])) / (permutations + 1)
            ),
            "qap_two_sided_p": float(
                (1 + np.sum(np.abs(values) >= abs(observed[index])))
                / (permutations + 1)
            ),
            "scope": (
                "fixed target panel; broad family labels permuted over target labels; "
                "signed correlations"
            ),
        })
    return rows


def target_descriptor_slopes(
    bundle: DatasetBundle,
    ligand_mask: np.ndarray | None = None,
    target_scaled: bool = False,
) -> np.ndarray:
    """Univariate target slopes for seven standardized ligand descriptors.

    Two-way centering is always performed on the complete target panel before
    any cross-panel target subset is selected.  If requested, each residual
    target is then standardized to unit sample standard deviation.
    """
    if ligand_mask is None:
        ligand_mask = np.ones(len(bundle.matrix), dtype=bool)
    ligand_mask = np.asarray(ligand_mask, dtype=bool)
    if ligand_mask.shape != (len(bundle.matrix),):
        raise ValueError("ligand mask has the wrong shape")
    if ligand_mask.sum() < 10:
        raise ValueError("too few ligands for target descriptor slopes")
    matrix = bundle.matrix[ligand_mask]
    descriptors = bundle.descriptor_matrix[ligand_mask]
    residual = two_way_center(matrix)
    if target_scaled:
        residual, _, _ = standardize_columns(residual)
    descriptor_sd = descriptors.std(axis=0, ddof=1)
    if np.any(descriptor_sd <= 1e-12):
        raise ValueError("constant molecular descriptor on selected support")
    standardized = (descriptors - descriptors.mean(axis=0)) / descriptor_sd
    return standardized.T @ residual / (len(standardized) - 1)


def exact_descriptor_omnibus_permutation(
    first: np.ndarray,
    second: np.ndarray,
    permutation_cache: dict[int, np.ndarray] | None = None,
) -> dict[str, Any]:
    """Exact target-label test of mean descriptor-wise Spearman correlation."""
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    if first.shape != second.shape or first.ndim != 2:
        raise ValueError("slope matrices must have the same target-by-descriptor shape")
    n_targets, n_descriptors = first.shape
    if n_descriptors != len(DESCRIPTOR_NAMES):
        raise ValueError("omnibus test requires the prespecified seven descriptors")
    if not 3 <= n_targets <= 9:
        raise ValueError("exact enumeration is restricted to 3--9 matched targets")

    first_rank = np.apply_along_axis(stats.rankdata, 0, first).astype(np.float64)
    second_rank = np.apply_along_axis(stats.rankdata, 0, second).astype(np.float64)
    first_rank -= first_rank.mean(axis=0)
    second_rank -= second_rank.mean(axis=0)
    denominator = np.sqrt(
        np.square(first_rank).sum(axis=0) * np.square(second_rank).sum(axis=0)
    )
    if np.any(denominator <= 0):
        raise ValueError("a descriptor has constant target slopes")
    descriptor_rho = (first_rank * second_rank).sum(axis=0) / denominator
    observed = float(descriptor_rho.mean())

    cache = permutation_cache if permutation_cache is not None else {}
    if n_targets not in cache:
        flat = np.fromiter(
            (
                index
                for permutation in itertools.permutations(range(n_targets))
                for index in permutation
            ),
            dtype=np.int16,
            count=int(math.factorial(n_targets) * n_targets),
        )
        cache[n_targets] = flat.reshape(-1, n_targets)
    permutations = cache[n_targets]
    null = np.empty(len(permutations), dtype=np.float64)
    chunk_size = 25_000
    for start in range(0, len(permutations), chunk_size):
        stop = min(start + chunk_size, len(permutations))
        permuted_rank = second_rank[permutations[start:stop]]
        descriptor_null = (
            (first_rank[None, :, :] * permuted_rank).sum(axis=1) / denominator
        )
        null[start:stop] = descriptor_null.mean(axis=1)
    exceedances = int(np.sum(np.abs(null) >= abs(observed) - 1e-15))
    return {
        "descriptor_spearman_rho": {
            name: float(value)
            for name, value in zip(DESCRIPTOR_NAMES, descriptor_rho)
        },
        "omnibus_mean_descriptor_spearman_rho": observed,
        "exact_target_label_permutation_two_sided_p": float(
            exceedances / len(permutations)
        ),
        "exact_target_label_permutations": int(len(permutations)),
        "exact_null_exceedances": exceedances,
        "exact_null_mean": float(null.mean()),
        "exact_null_q025": float(np.quantile(null, 0.025)),
        "exact_null_q975": float(np.quantile(null, 0.975)),
    }


def rank_fingerprint_hungarian_retrieval(
    first: np.ndarray,
    second: np.ndarray,
    permutation_cache: dict[int, np.ndarray] | None = None,
    distance_metric: str = "rank_squared_euclidean",
) -> dict[str, Any]:
    """One-to-one target retrieval under a stated fingerprint distance."""
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    if first.shape != second.shape or first.ndim != 2:
        raise ValueError("fingerprints must have equal target-by-feature shape")
    n_targets, n_features = first.shape
    if not 3 <= n_targets <= 9 or n_features < 2:
        raise ValueError("retrieval expects 3--9 targets and at least two features")
    first_rank = np.apply_along_axis(stats.rankdata, 0, first).astype(np.float64)
    second_rank = np.apply_along_axis(stats.rankdata, 0, second).astype(np.float64)
    distance_definitions = {
        "rank_squared_euclidean": (
            "mean squared difference of within-panel target ranks across the "
            "included descriptor-response slopes"
        ),
        "rank_manhattan": (
            "mean absolute difference of within-panel target ranks across the "
            "included descriptor-response slopes"
        ),
        "descriptor_zscore_squared_euclidean": (
            "mean squared difference after each descriptor slope is standardized "
            "over targets within each panel"
        ),
        "descriptor_zscore_cosine": (
            "one minus cosine similarity after each descriptor slope is standardized "
            "over targets within each panel"
        ),
    }
    if distance_metric == "rank_squared_euclidean":
        cost = np.square(
            first_rank[:, None, :] - second_rank[None, :, :]
        ).mean(axis=2)
    elif distance_metric == "rank_manhattan":
        cost = np.abs(
            first_rank[:, None, :] - second_rank[None, :, :]
        ).mean(axis=2)
    elif distance_metric in {
        "descriptor_zscore_squared_euclidean", "descriptor_zscore_cosine",
    }:
        first_z = (first - first.mean(axis=0)) / first.std(axis=0, ddof=1)
        second_z = (second - second.mean(axis=0)) / second.std(axis=0, ddof=1)
        if distance_metric == "descriptor_zscore_squared_euclidean":
            cost = np.square(
                first_z[:, None, :] - second_z[None, :, :]
            ).mean(axis=2)
        else:
            first_norm = np.linalg.norm(first_z, axis=1, keepdims=True)
            second_norm = np.linalg.norm(second_z, axis=1, keepdims=True)
            if np.any(first_norm <= 1e-12) or np.any(second_norm <= 1e-12):
                raise ValueError("a standardized target fingerprint has zero norm")
            cost = 1.0 - (first_z / first_norm) @ (second_z / second_norm).T
    else:
        raise ValueError(f"unknown fingerprint distance: {distance_metric}")
    row_indices, assigned_second = linear_sum_assignment(cost)
    if not np.array_equal(row_indices, np.arange(n_targets)):
        raise AssertionError("Hungarian assignment did not return every first-panel row")
    correct = int(np.sum(assigned_second == np.arange(n_targets)))

    cache = permutation_cache if permutation_cache is not None else {}
    if n_targets not in cache:
        flat = np.fromiter(
            (
                index
                for permutation in itertools.permutations(range(n_targets))
                for index in permutation
            ),
            dtype=np.int16,
            count=int(math.factorial(n_targets) * n_targets),
        )
        cache[n_targets] = flat.reshape(-1, n_targets)
    permutations = cache[n_targets]
    null_correct = np.sum(
        assigned_second[None, :] == permutations, axis=1
    )
    assignment_costs = cost[np.arange(n_targets)[None, :], permutations].sum(axis=1)
    optimum = float(cost[row_indices, assigned_second].sum())
    optimal_assignments = np.isclose(assignment_costs, optimum, rtol=0, atol=1e-12)
    higher = assignment_costs[assignment_costs > optimum + 1e-12]
    return {
        "n_targets": int(n_targets),
        "n_descriptor_features": int(n_features),
        "correct_target_identities": correct,
        "correct_fraction": float(correct / n_targets),
        "assigned_second_indices": assigned_second.astype(int),
        "assigned_pair_costs": cost[row_indices, assigned_second],
        "total_assignment_cost": optimum,
        "number_of_cost_optimal_assignments": int(optimal_assignments.sum()),
        "second_best_total_assignment_cost": float(higher.min()),
        "exact_target_label_p_for_at_least_observed_correct": float(
            np.mean(null_correct >= correct)
        ),
        "exact_target_label_permutations": int(len(permutations)),
        "distance_metric": distance_metric,
        "distance_definition": distance_definitions[distance_metric],
    }


def cross_panel_descriptor_fingerprint(
    docking: DatasetBundle,
    dockstring: DatasetBundle,
) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
    """Replicate target-specific ligand-property responses across the panels."""
    if docking.name != "Docking-44" or dockstring.name != "DOCKSTRING-58":
        raise ValueError("cross-panel fingerprint expects Docking-44 and DOCKSTRING-58")

    mapping = pd.DataFrame(CROSS_PANEL_TARGETS)
    mapping["docking44_structure_is_human"] = mapping[
        "docking44_structure_species"
    ].eq("Homo sapiens")
    mapping["mapping_basis"] = (
        "PDB-to-UniProt assignment audited with PDBe/SIFTS v2; DOCKSTRING gene "
        "symbol mapped to the reviewed human UniProt entry"
    )
    mapping["target_selection_used_score_outcomes"] = False

    shared_full = set(docking.full_inchikeys).intersection(dockstring.full_inchikeys)
    shared_connectivity = set(docking.connectivity_blocks).intersection(
        dockstring.connectivity_blocks
    )
    docking_nonempty = set(docking.nonempty_scaffolds) - {""}
    dockstring_nonempty = set(dockstring.nonempty_scaffolds) - {""}
    shared_scaffolds = docking_nonempty.intersection(dockstring_nonempty)

    primary_docking = np.ones(len(docking.matrix), dtype=bool)
    primary_dockstring = np.ones(len(dockstring.matrix), dtype=bool)
    connectivity_docking = ~np.isin(
        docking.connectivity_blocks, list(shared_connectivity)
    )
    connectivity_dockstring = ~np.isin(
        dockstring.connectivity_blocks, list(shared_connectivity)
    )
    scaffold_docking = (
        (docking.nonempty_scaffolds != "")
        & ~np.isin(docking.nonempty_scaffolds, list(shared_scaffolds))
    )
    scaffold_dockstring = (
        (dockstring.nonempty_scaffolds != "")
        & ~np.isin(dockstring.nonempty_scaffolds, list(shared_scaffolds))
    )
    support_definitions = {
        "primary_seeded_support": (primary_docking, primary_dockstring),
        "connectivity_disjoint": (connectivity_docking, connectivity_dockstring),
        "strict_nonempty_murcko_scaffold_disjoint": (
            scaffold_docking, scaffold_dockstring,
        ),
    }
    target_definitions = {
        "all_9_mapped_targets": np.ones(len(mapping), dtype=bool),
        "human_structure_only_7": mapping[
            "docking44_structure_is_human"
        ].to_numpy(dtype=bool),
        "different_receptor_structure_only_6": ~mapping[
            "same_receptor_structure_in_dockstring"
        ].to_numpy(dtype=bool),
    }
    analysis_definitions = [
        ("primary_seeded_support", "all_9_mapped_targets"),
        ("connectivity_disjoint", "all_9_mapped_targets"),
        ("strict_nonempty_murcko_scaffold_disjoint", "all_9_mapped_targets"),
        ("primary_seeded_support", "human_structure_only_7"),
        ("primary_seeded_support", "different_receptor_structure_only_6"),
    ]
    docking_indices = np.asarray(
        [docking.targets.index(value) for value in mapping["docking44_target"]],
        dtype=int,
    )
    dockstring_indices = np.asarray(
        [dockstring.targets.index(value) for value in mapping["dockstring_target"]],
        dtype=int,
    )
    permutation_cache: dict[int, np.ndarray] = {}
    fingerprint_rows: list[dict[str, Any]] = []
    concordance_rows: list[dict[str, Any]] = []
    nested: dict[str, Any] = {}
    slope_cache: dict[tuple[str, bool, str], np.ndarray] = {}

    for support_name, target_name in analysis_definitions:
        docking_mask, dockstring_mask = support_definitions[support_name]
        target_mask = target_definitions[target_name]
        analysis_key = f"{support_name}__{target_name}"
        nested[analysis_key] = {}
        for target_scaled in [False, True]:
            scaling_name = (
                "target_standardized_two_way_residual"
                if target_scaled else "unscaled_two_way_residual"
            )
            first_key = (support_name, target_scaled, docking.name)
            second_key = (support_name, target_scaled, dockstring.name)
            if first_key not in slope_cache:
                slope_cache[first_key] = target_descriptor_slopes(
                    docking, docking_mask, target_scaled
                )
                slope_cache[second_key] = target_descriptor_slopes(
                    dockstring, dockstring_mask, target_scaled
                )
            first = slope_cache[first_key][:, docking_indices].T[target_mask]
            second = slope_cache[second_key][:, dockstring_indices].T[target_mask]
            result = exact_descriptor_omnibus_permutation(
                first, second, permutation_cache
            )
            nested[analysis_key][scaling_name] = result
            selected_mapping = mapping.loc[target_mask].reset_index(drop=True)
            for target_index, target_row in selected_mapping.iterrows():
                for descriptor_index, descriptor_name in enumerate(DESCRIPTOR_NAMES):
                    fingerprint_rows.append({
                        "chemical_support_sensitivity": support_name,
                        "target_sensitivity": target_name,
                        "residual_scaling": scaling_name,
                        "n_docking44_ligands": int(docking_mask.sum()),
                        "n_dockstring_ligands": int(dockstring_mask.sum()),
                        "n_targets": int(target_mask.sum()),
                        "docking44_target": target_row["docking44_target"],
                        "dockstring_target": target_row["dockstring_target"],
                        "descriptor": descriptor_name,
                        "docking44_slope_per_descriptor_sd": float(
                            first[target_index, descriptor_index]
                        ),
                        "dockstring_slope_per_descriptor_sd": float(
                            second[target_index, descriptor_index]
                        ),
                        "docking44_target_slope_rank": float(
                            stats.rankdata(first[:, descriptor_index])[target_index]
                        ),
                        "dockstring_target_slope_rank": float(
                            stats.rankdata(second[:, descriptor_index])[target_index]
                        ),
                    })
            for descriptor_name in DESCRIPTOR_NAMES:
                concordance_rows.append({
                    "chemical_support_sensitivity": support_name,
                    "target_sensitivity": target_name,
                    "residual_scaling": scaling_name,
                    "n_docking44_ligands": int(docking_mask.sum()),
                    "n_dockstring_ligands": int(dockstring_mask.sum()),
                    "n_targets": int(target_mask.sum()),
                    "statistic": descriptor_name,
                    "spearman_rho": result["descriptor_spearman_rho"][descriptor_name],
                    "exact_target_label_permutation_two_sided_p": np.nan,
                    "exact_target_label_permutations": np.nan,
                })
            concordance_rows.append({
                "chemical_support_sensitivity": support_name,
                "target_sensitivity": target_name,
                "residual_scaling": scaling_name,
                "n_docking44_ligands": int(docking_mask.sum()),
                "n_dockstring_ligands": int(dockstring_mask.sum()),
                "n_targets": int(target_mask.sum()),
                "statistic": "prespecified_mean_of_7_descriptor_spearman_rho",
                "spearman_rho": result[
                    "omnibus_mean_descriptor_spearman_rho"
                ],
                "exact_target_label_permutation_two_sided_p": result[
                    "exact_target_label_permutation_two_sided_p"
                ],
                "exact_target_label_permutations": result[
                    "exact_target_label_permutations"
                ],
            })

    primary_first = slope_cache[
        ("primary_seeded_support", False, docking.name)
    ][:, docking_indices].T
    primary_second = slope_cache[
        ("primary_seeded_support", False, dockstring.name)
    ][:, dockstring_indices].T
    primary_retrieval = rank_fingerprint_hungarian_retrieval(
        primary_first, primary_second, permutation_cache
    )
    retrieval_rows: list[dict[str, Any]] = []
    retrieval_nested: dict[str, Any] = {}
    retrieval_definitions = [
        ("primary_seeded_support", False),
        ("primary_seeded_support", True),
        ("connectivity_disjoint", False),
        ("strict_nonempty_murcko_scaffold_disjoint", False),
    ]
    for support_name, target_scaled in retrieval_definitions:
        scaling_name = (
            "target_standardized_two_way_residual"
            if target_scaled else "unscaled_two_way_residual"
        )
        first = slope_cache[(support_name, target_scaled, docking.name)][
            :, docking_indices
        ].T
        second = slope_cache[(support_name, target_scaled, dockstring.name)][
            :, dockstring_indices
        ].T
        result = rank_fingerprint_hungarian_retrieval(
            first, second, permutation_cache
        )
        key = f"{support_name}__{scaling_name}__all_7_descriptors"
        retrieval_nested[key] = result
        retrieval_rows.append({
            "chemical_support_sensitivity": support_name,
            "residual_scaling": scaling_name,
            "descriptor_sensitivity": "all_7_descriptors",
            "distance_metric": result["distance_metric"],
            "n_descriptor_features": 7,
            "correct_target_identities": result["correct_target_identities"],
            "correct_fraction": result["correct_fraction"],
            "total_assignment_cost": result["total_assignment_cost"],
            "number_of_cost_optimal_assignments": result[
                "number_of_cost_optimal_assignments"
            ],
            "second_best_total_assignment_cost": result[
                "second_best_total_assignment_cost"
            ],
            "exact_target_label_p_for_at_least_observed_correct": result[
                "exact_target_label_p_for_at_least_observed_correct"
            ],
            "exact_target_label_permutations": result[
                "exact_target_label_permutations"
            ],
        })

    for alternative_metric in [
        "rank_manhattan",
        "descriptor_zscore_squared_euclidean",
        "descriptor_zscore_cosine",
    ]:
        result = rank_fingerprint_hungarian_retrieval(
            primary_first,
            primary_second,
            permutation_cache,
            distance_metric=alternative_metric,
        )
        key = (
            "primary_seeded_support__unscaled_two_way_residual__all_7_descriptors__"
            f"{alternative_metric}"
        )
        retrieval_nested[key] = result
        retrieval_rows.append({
            "chemical_support_sensitivity": "primary_seeded_support",
            "residual_scaling": "unscaled_two_way_residual",
            "descriptor_sensitivity": "all_7_descriptors",
            "distance_metric": result["distance_metric"],
            "n_descriptor_features": 7,
            "correct_target_identities": result["correct_target_identities"],
            "correct_fraction": result["correct_fraction"],
            "total_assignment_cost": result["total_assignment_cost"],
            "number_of_cost_optimal_assignments": result[
                "number_of_cost_optimal_assignments"
            ],
            "second_best_total_assignment_cost": result[
                "second_best_total_assignment_cost"
            ],
            "exact_target_label_p_for_at_least_observed_correct": result[
                "exact_target_label_p_for_at_least_observed_correct"
            ],
            "exact_target_label_permutations": result[
                "exact_target_label_permutations"
            ],
        })

    for excluded_index, excluded_name in enumerate(DESCRIPTOR_NAMES):
        retain_descriptors = np.arange(len(DESCRIPTOR_NAMES)) != excluded_index
        result = rank_fingerprint_hungarian_retrieval(
            primary_first[:, retain_descriptors],
            primary_second[:, retain_descriptors],
            permutation_cache,
        )
        key = f"primary_seeded_support__unscaled_two_way_residual__leave_out_{excluded_name}"
        retrieval_nested[key] = result
        retrieval_rows.append({
            "chemical_support_sensitivity": "primary_seeded_support",
            "residual_scaling": "unscaled_two_way_residual",
            "descriptor_sensitivity": f"leave_out_{excluded_name}",
            "distance_metric": result["distance_metric"],
            "n_descriptor_features": 6,
            "correct_target_identities": result["correct_target_identities"],
            "correct_fraction": result["correct_fraction"],
            "total_assignment_cost": result["total_assignment_cost"],
            "number_of_cost_optimal_assignments": result[
                "number_of_cost_optimal_assignments"
            ],
            "second_best_total_assignment_cost": result[
                "second_best_total_assignment_cost"
            ],
            "exact_target_label_p_for_at_least_observed_correct": result[
                "exact_target_label_p_for_at_least_observed_correct"
            ],
            "exact_target_label_permutations": result[
                "exact_target_label_permutations"
            ],
        })

    assignment_rows: list[dict[str, Any]] = []
    assigned_indices = primary_retrieval["assigned_second_indices"]
    assigned_costs = primary_retrieval["assigned_pair_costs"]
    for first_index, second_index in enumerate(assigned_indices):
        assignment_rows.append({
            "docking44_target": mapping.loc[first_index, "docking44_target"],
            "true_dockstring_target": mapping.loc[first_index, "dockstring_target"],
            "assigned_dockstring_target": mapping.loc[
                int(second_index), "dockstring_target"
            ],
            "correct_target_identity": bool(first_index == second_index),
            "assigned_pair_mean_squared_rank_distance": float(
                assigned_costs[first_index]
            ),
            "chemical_support_sensitivity": "primary_seeded_support",
            "residual_scaling": "unscaled_two_way_residual",
            "descriptor_set": "all_7_descriptors",
        })

    leave_one_target: dict[str, float] = {}
    for omitted in range(len(mapping)):
        retain = np.arange(len(mapping)) != omitted
        leave_one_target[str(mapping.loc[omitted, "dockstring_target"])] = (
            exact_descriptor_omnibus_permutation(
                primary_first[retain], primary_second[retain], permutation_cache
            )["omnibus_mean_descriptor_spearman_rho"]
        )

    support_rows = []
    for support_name, (first_mask, second_mask) in support_definitions.items():
        support_rows.append({
            "chemical_support_sensitivity": support_name,
            "docking44_ligands_retained": int(first_mask.sum()),
            "dockstring_ligands_retained": int(second_mask.sum()),
            "docking44_ligands_removed": int((~first_mask).sum()),
            "dockstring_ligands_removed": int((~second_mask).sum()),
            "full_inchikey_overlap_count_on_primary_support": int(len(shared_full)),
            "shared_connectivity_block_count_on_primary_support": int(
                len(shared_connectivity)
            ),
            "shared_nonempty_murcko_scaffold_count_on_primary_support": int(
                len(shared_scaffolds)
            ),
        })
    primary_result = nested[
        "primary_seeded_support__all_9_mapped_targets"
    ]["unscaled_two_way_residual"]
    summary = {
        "analysis_status": "exploratory_replication_with_prespecified_omnibus",
        "primary": primary_result,
        "sensitivities": nested,
        "leave_one_target_primary_omnibus": leave_one_target,
        "leave_one_target_primary_omnibus_range": [
            float(min(leave_one_target.values())),
            float(max(leave_one_target.values())),
        ],
        "exploratory_target_identity_retrieval": {
            "primary": primary_retrieval,
            "sensitivities": retrieval_nested,
            "analysis_status": "exploratory_post_hoc_operational_consequence",
            "selection_caveat": (
                "The retrieval task was formulated after observing cross-panel "
                "descriptor-slope concordance. Its exact conditional permutation "
                "probability is descriptive and requires external confirmation."
            ),
            "metric_sensitivity_caveat": (
                "The primary rank-squared-Euclidean rule is aligned with the rank-"
                "based omnibus statistic and was not tuned. Alternative reasonable "
                "fingerprint distances are reported because recovery is partly "
                "metric-dependent."
            ),
        },
        "mapping_count": int(len(mapping)),
        "human_structure_mapping_count": int(
            mapping["docking44_structure_is_human"].sum()
        ),
        "different_receptor_structure_mapping_count": int(
            (~mapping["same_receptor_structure_in_dockstring"]).sum()
        ),
        "chemical_overlap_audit": {
            "full_inchikey_overlap_count": int(len(shared_full)),
            "shared_connectivity_block_count": int(len(shared_connectivity)),
            "shared_nonempty_murcko_scaffold_count": int(len(shared_scaffolds)),
            "scope": (
                "Docking-44 full 12,651-ligand support versus the fixed seeded "
                "15,000-ligand DOCKSTRING descriptor support; these are not overlap "
                "counts against the full 260,060-ligand DOCKSTRING matrix"
            ),
        },
        "interpretation": (
            "The statistic tests whether independently generated panels reproduce "
            "the ordering of target-specific score responses to a prespecified set "
            "of seven ligand descriptors after the common ligand effect is removed."
        ),
        "claim_boundary": (
            "Concordant score-response fingerprints establish reproducible docking-"
            "surface structure, not binding accuracy, causal scoring-function "
            "components, or biological specificity."
        ),
        "selection_and_multiplicity_caveat": (
            "The cross-panel fingerprint hypothesis and target mapping were assembled "
            "during exploratory revision. Exact target-label probabilities condition "
            "on this mapping and the seven-descriptor set and are not confirmatory."
        ),
        "omnibus_dependence_handling": (
            "A single target-label permutation is applied jointly to all seven "
            "descriptors, preserving their cross-descriptor dependence."
        ),
    }
    return summary, {
        "cross_panel_target_mapping.csv": mapping,
        "cross_panel_support_audit.csv": pd.DataFrame(support_rows),
        "cross_panel_descriptor_fingerprint.csv": pd.DataFrame(fingerprint_rows),
        "cross_panel_descriptor_concordance.csv": pd.DataFrame(concordance_rows),
        "cross_panel_target_identity_assignment.csv": pd.DataFrame(assignment_rows),
        "cross_panel_target_identity_retrieval_sensitivity.csv": pd.DataFrame(
            retrieval_rows
        ),
    }


def dockstring_box_size_control(
    bundle: DatasetBundle,
) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
    """Test whether the nearly invariant official box sizes explain target slopes."""
    if bundle.name != "DOCKSTRING-58":
        raise ValueError("box-size control is defined only for DOCKSTRING-58")
    dimensions = pd.DataFrame({
        "target": bundle.targets,
        "size_x_angstrom": 30.0,
        "size_y_angstrom": 30.0,
        "size_z_angstrom": [32.0 if value == "DRD2" else 30.0 for value in bundle.targets],
    })
    dimensions["box_volume_angstrom3"] = (
        dimensions.size_x_angstrom
        * dimensions.size_y_angstrom
        * dimensions.size_z_angstrom
    )
    dimensions["source_package"] = "dockstring 0.3.4"
    dimensions["source_resource"] = dimensions.target.map(
        lambda value: f"dockstring/resources/targets/{value}_conf.txt"
    )
    dimensions["source_repository"] = "https://github.com/dockstring/dockstring"
    dimensions["metadata_selection_used_score_outcomes"] = False

    statistics_rows: list[dict[str, Any]] = []
    slope_columns: dict[str, np.ndarray] = {}
    volumes = dimensions.box_volume_angstrom3.to_numpy(dtype=np.float64)
    low_volume = float(volumes.min())
    high_volume = float(volumes.max())
    if np.sum(volumes == high_volume) != 1 or np.sum(volumes == low_volume) != len(volumes) - 1:
        raise AssertionError("expected 57 identical boxes and one DRD2 z-extension")
    for target_scaled in [False, True]:
        scaling_name = (
            "target_standardized_two_way_residual"
            if target_scaled else "unscaled_two_way_residual"
        )
        slopes = target_descriptor_slopes(bundle, target_scaled=target_scaled)
        for descriptor_index, descriptor_name in enumerate(DESCRIPTOR_NAMES):
            target_slopes = slopes[descriptor_index]
            observed = float(stats.spearmanr(volumes, target_slopes).statistic)
            exact_null = np.empty(len(volumes), dtype=np.float64)
            for high_index in range(len(volumes)):
                permuted_volume = np.full(len(volumes), low_volume)
                permuted_volume[high_index] = high_volume
                exact_null[high_index] = float(
                    stats.spearmanr(permuted_volume, target_slopes).statistic
                )
            exact_p = float(np.mean(np.abs(exact_null) >= abs(observed) - 1e-15))
            statistics_rows.append({
                "residual_scaling": scaling_name,
                "descriptor": descriptor_name,
                "spearman_rho_box_volume_vs_target_descriptor_slope": observed,
                "exact_two_sided_p": exact_p,
                "distinct_target_label_placements": int(len(exact_null)),
                "null_q025": float(np.quantile(exact_null, 0.025)),
                "null_q975": float(np.quantile(exact_null, 0.975)),
            })
            slope_columns[f"{scaling_name}__{descriptor_name}_slope"] = target_slopes
    for name, values in slope_columns.items():
        dimensions[name] = values
    statistics_frame = pd.DataFrame(statistics_rows)
    summary = {
        "official_configuration_pattern": (
            "57 targets use 30 x 30 x 30 A boxes; DRD2 uses 30 x 30 x 32 A"
        ),
        "unique_box_volumes": sorted(
            float(value) for value in dimensions.box_volume_angstrom3.unique()
        ),
        "configuration_source": "dockstring 0.3.4 packaged target configurations",
        "statistics": statistics_rows,
        "interpretation": (
            "Box volume is nearly invariant by design and therefore cannot explain "
            "the broad cross-target descriptor-response fingerprint. The single-"
            "exception test is included only as an artifact control and has very "
            "limited information about geometry."
        ),
    }
    return summary, {
        "dockstring_box_size_control.csv": dimensions,
        "dockstring_box_size_control_statistics.csv": statistics_frame,
    }


def _permutation_correlation_p(
    x: np.ndarray,
    y: np.ndarray,
    permutations: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    x_rank = stats.rankdata(x).astype(np.float64)
    y_rank = stats.rankdata(y).astype(np.float64)
    x_rank -= x_rank.mean()
    y_rank -= y_rank.mean()
    denominator = float(np.linalg.norm(x_rank) * np.linalg.norm(y_rank))
    observed = float(x_rank @ y_rank / denominator)
    exceedances = 0
    for _ in range(permutations):
        permuted = rng.permutation(y_rank)
        statistic = float(x_rank @ permuted / denominator)
        exceedances += int(abs(statistic) >= abs(observed))
    return observed, float((1 + exceedances) / (permutations + 1))


def exploratory_pocket_volume_analysis(
    bundle: DatasetBundle,
    missing_cells_by_target: np.ndarray,
    config: AnalysisConfig,
) -> tuple[dict[str, Any], pd.DataFrame]:
    if bundle.name != "Docking-44":
        raise ValueError("pocket-volume analysis is currently defined only for Docking-44")
    descriptor = pd.DataFrame(
        bundle.descriptor_matrix, columns=DESCRIPTOR_NAMES
    )
    molecular_weight_100 = descriptor["molecular_weight"].to_numpy() / 100.0
    rotatable_bonds = descriptor["rotatable_bonds"].to_numpy()
    centered_weight = molecular_weight_100 - molecular_weight_100.mean()
    raw_centered = bundle.matrix - bundle.matrix.mean(axis=0)
    raw_slopes = centered_weight @ raw_centered / float(centered_weight @ centered_weight)
    residual = two_way_center(bundle.matrix)
    residual_slopes = centered_weight @ residual / float(
        centered_weight @ centered_weight
    )
    identity_slopes = raw_slopes - raw_slopes.mean()
    slope_identity_error = float(np.max(np.abs(residual_slopes - identity_slopes)))

    conditional_design = np.column_stack([
        np.ones(len(bundle.matrix)), molecular_weight_100, rotatable_bonds,
    ])
    conditional_coefficients = np.linalg.lstsq(
        conditional_design, bundle.matrix, rcond=None
    )[0]
    pocket = pd.read_csv(
        FROZEN / "toxicodynamics" / "data" / "44pockets_analysis.csv"
    ).rename(columns={"PDB": "target"})
    target_frame = pd.DataFrame({
        "target": bundle.targets,
        "family": bundle.families,
        "raw_molecular_weight_slope_per_100_da": raw_slopes,
        "two_way_residual_molecular_weight_slope_per_100_da": residual_slopes,
        "conditional_molecular_weight_slope_per_100_da": conditional_coefficients[1],
        "conditional_rotatable_bond_slope": conditional_coefficients[2],
        "imputed_cells": missing_cells_by_target.astype(int),
        "imputed_fraction": missing_cells_by_target / len(bundle.matrix),
    }).merge(pocket, on="target", how="left", validate="one_to_one")
    complete = target_frame.dropna(
        subset=["volume", "two_way_residual_molecular_weight_slope_per_100_da"]
    ).reset_index(drop=True)
    x = complete["volume"].to_numpy(dtype=np.float64)
    y = complete[
        "two_way_residual_molecular_weight_slope_per_100_da"
    ].to_numpy(dtype=np.float64)
    rng = np.random.default_rng(config.pocket_seed)
    rho, permutation_p = _permutation_correlation_p(
        x, y, config.pocket_permutations, rng
    )

    bootstrap = np.empty(config.pocket_bootstrap_replicates, dtype=np.float64)
    for repetition in range(config.pocket_bootstrap_replicates):
        indices = rng.integers(0, len(complete), len(complete))
        statistic = stats.spearmanr(x[indices], y[indices]).statistic
        bootstrap[repetition] = float(statistic)
    bootstrap = bootstrap[np.isfinite(bootstrap)]

    leave_one_target = [
        float(stats.spearmanr(np.delete(x, index), np.delete(y, index)).statistic)
        for index in range(len(complete))
    ]
    leave_one_family: list[float] = []
    complete_families = complete["family"].astype(str).to_numpy()
    for family in sorted(set(complete_families)):
        retain = complete_families != family
        leave_one_family.append(
            float(stats.spearmanr(x[retain], y[retain]).statistic)
        )

    x_rank = stats.rankdata(x).astype(float)
    y_rank = stats.rankdata(y).astype(float)
    for family in sorted(set(complete_families)):
        members = complete_families == family
        x_rank[members] -= x_rank[members].mean()
        y_rank[members] -= y_rank[members].mean()
    family_adjusted = float(np.corrcoef(x_rank, y_rank)[0, 1])
    within_family_exceedances = 0
    family_permutations = min(config.pocket_permutations, 10_000)
    for _ in range(family_permutations):
        permuted = y_rank.copy()
        for family in sorted(set(complete_families)):
            members = np.flatnonzero(complete_families == family)
            permuted[members] = rng.permutation(permuted[members])
        statistic = float(np.corrcoef(x_rank, permuted)[0, 1])
        within_family_exceedances += int(abs(statistic) >= abs(family_adjusted))

    conditional_y = complete[
        "conditional_molecular_weight_slope_per_100_da"
    ].to_numpy(dtype=np.float64)
    conditional_rho = float(stats.spearmanr(x, conditional_y).statistic)
    summary = {
        "analysis_status": "exploratory_post_hoc",
        "n_targets_with_complete_pocket_volume": int(len(complete)),
        "pocket_volume_vs_residual_mw_slope_spearman_rho": rho,
        "target_label_permutation_two_sided_p": permutation_p,
        "target_label_permutations": int(config.pocket_permutations),
        "target_bootstrap_replicates_requested": int(
            config.pocket_bootstrap_replicates
        ),
        "target_bootstrap_replicates_finite": int(len(bootstrap)),
        "target_bootstrap_95_interval": [
            float(np.quantile(bootstrap, 0.025)),
            float(np.quantile(bootstrap, 0.975)),
        ],
        "leave_one_target_rho_range": [
            float(min(leave_one_target)), float(max(leave_one_target)),
        ],
        "leave_one_family_rho_range": [
            float(min(leave_one_family)), float(max(leave_one_family)),
        ],
        "family_rank_residual_correlation": family_adjusted,
        "within_family_permutation_two_sided_p": float(
            (1 + within_family_exceedances) / (family_permutations + 1)
        ),
        "within_family_permutations": int(family_permutations),
        "pocket_volume_vs_conditional_mw_slope_spearman_rho": conditional_rho,
        "maximum_absolute_two_way_slope_identity_error": slope_identity_error,
        "directional_interpretation": (
            "Vina scores are negative-favorable; a negative association is consistent "
            "with larger pockets assigning a more favorable marginal size slope"
        ),
        "selection_and_multiplicity_caveat": (
            "This target-level association was selected after inspection of the "
            "descriptor--residual-mode alignment. P values are descriptive and are "
            "not adjusted for the broader exploratory hypothesis search."
        ),
        "provenance_limitation": (
            "The frozen pocket table is attributed to descriptors derived from wwPDB "
            "structures, but the release does not contain an executable, versioned "
            "workflow defining the pockets and generating volume/area/depth fields. "
            "This analysis is therefore not an independently reproducible structural "
            "validation and must remain exploratory until the descriptors are rebuilt "
            "with documented software and parameters."
        ),
        "replication_boundary": (
            "No harmonized DOCKSTRING pocket-volume table is included; independent "
            "cross-panel replication is still required."
        ),
    }
    return summary, target_frame


def analyze_dataset(
    bundle: DatasetBundle,
    config: AnalysisConfig,
) -> dict[str, Any]:
    # Exact surface summaries use every available ligand.  The potentially
    # expensive descriptor analysis uses the separately frozen support in
    # ``bundle.matrix`` (all rows for Docking-44; seeded 15k for DOCKSTRING).
    raw = bundle.spectral_matrix
    residual = two_way_center(raw)
    raw_moments = correlation_moments(raw)
    residual_moments = correlation_moments(residual)
    common_mean_square_reduction = (
        raw_moments["mean_offdiagonal_correlation"] ** 2
        - residual_moments["mean_offdiagonal_correlation"] ** 2
    )
    dispersion_reduction = (
        raw_moments["pairwise_correlation_variance"]
        - residual_moments["pairwise_correlation_variance"]
    )
    total_mean_square_reduction = (
        raw_moments["mean_squared_offdiagonal_correlation"]
        - residual_moments["mean_squared_offdiagonal_correlation"]
    )
    if abs(total_mean_square_reduction) <= 1e-15:
        raise ValueError("raw and residual mean-squared correlations are indistinguishable")

    descriptor = grouped_descriptor_decomposition(
        bundle.matrix, bundle.descriptor_matrix, bundle.groups, config.folds
    )
    qap_rows = family_qap(
        descriptor["oof_surface"],
        descriptor["oof_error"],
        bundle.families,
        config.qap_permutations,
        config.qap_seed + (0 if bundle.name == "Docking-44" else 1),
    )
    full_residual_sd = residual.std(axis=0, ddof=1)
    anova = orthogonal_two_way_anova(raw)
    column_standardized, _, _ = standardize_columns(raw)
    standardized_anova = orthogonal_two_way_anova(column_standardized)
    return {
        "raw_moments": raw_moments,
        "residual_moments": residual_moments,
        "transition": {
            "pr_expansion_ratio": float(
                residual_moments["participation_ratio"]
                / raw_moments["participation_ratio"]
            ),
            "pr_increase": float(
                residual_moments["participation_ratio"]
                - raw_moments["participation_ratio"]
            ),
            "mean_squared_correlation_reduction": total_mean_square_reduction,
            "common_signed_mean_square_reduction": common_mean_square_reduction,
            "pairwise_correlation_dispersion_reduction": dispersion_reduction,
            "fraction_of_net_mean_square_reduction_from_common_signed_mean": float(
                common_mean_square_reduction / total_mean_square_reduction
            ),
            "fraction_of_net_mean_square_reduction_from_pair_dispersion": float(
                dispersion_reduction / total_mean_square_reduction
            ),
            "decomposition_closure_error": float(
                abs(
                    total_mean_square_reduction
                    - common_mean_square_reduction
                    - dispersion_reduction
                )
            ),
        },
        "uniform_axis": uniform_axis_diagnostics(raw),
        "variance_reweighting": {
            "raw_covariance_pr": covariance_pr(raw),
            "residual_covariance_pr": covariance_pr(residual),
            "raw_correlation_pr": raw_moments["participation_ratio"],
            "residual_correlation_pr": residual_moments["participation_ratio"],
            "covariance_pr_expansion_ratio": float(
                covariance_pr(residual) / covariance_pr(raw)
            ),
            "correlation_pr_expansion_ratio": float(
                residual_moments["participation_ratio"]
                / raw_moments["participation_ratio"]
            ),
            "residual_target_sd_coefficient_of_variation": float(
                full_residual_sd.std(ddof=1) / full_residual_sd.mean()
            ),
            "residual_correlation_pr_over_covariance_pr": float(
                residual_moments["participation_ratio"] / covariance_pr(residual)
            ),
        },
        "orthogonal_two_way_anova": anova,
        "column_standardized_orthogonal_two_way_anova": standardized_anova,
        "descriptor": descriptor,
        "qap_rows": qap_rows,
    }


def run_analysis(config: AnalysisConfig, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    bundles = [load_docking44(), load_dockstring(config)]
    analyses = {bundle.name: analyze_dataset(bundle, config) for bundle in bundles}

    moment_rows: list[dict[str, Any]] = []
    uniform_rows: list[dict[str, Any]] = []
    descriptor_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    target_r2_rows: list[dict[str, Any]] = []
    qap_rows: list[dict[str, Any]] = []
    variance_rows: list[dict[str, Any]] = []
    anova_rows: list[dict[str, Any]] = []
    dataset_rows: list[dict[str, Any]] = []

    for bundle in bundles:
        analysis = analyses[bundle.name]
        for surface in ["raw", "residual"]:
            moment_rows.append({
                "dataset": bundle.name,
                "surface": surface,
                **analysis[f"{surface}_moments"],
            })
        uniform_rows.append({"dataset": bundle.name, **analysis["uniform_axis"]})
        descriptor_rows.append({
            "dataset": bundle.name,
            "n_ligands": int(len(bundle.matrix)),
            "n_targets": int(len(bundle.targets)),
            "support_rule": bundle.support_rule,
            "group_definition": bundle.group_definition,
            **analysis["descriptor"]["metrics"],
        })
        for row in analysis["descriptor"]["fold_rows"]:
            fold_rows.append({"dataset": bundle.name, **row})
        for target, family, r2 in zip(
            bundle.targets, bundle.families, analysis["descriptor"]["target_r2"]
        ):
            target_r2_rows.append({
                "dataset": bundle.name,
                "target": target,
                "family": family,
                "out_of_fold_descriptor_r2": float(r2),
            })
        for row in analysis["qap_rows"]:
            qap_rows.append({"dataset": bundle.name, **row})
        variance_rows.append({
            "dataset": bundle.name, **analysis["variance_reweighting"]
        })
        for surface, anova in [
            ("grand_centered_score_surface", analysis["orthogonal_two_way_anova"]),
            (
                "column_standardized_score_surface",
                analysis["column_standardized_orthogonal_two_way_anova"],
            ),
        ]:
            for component, component_ss in anova["component_sum_squares"].items():
                anova_rows.append({
                    "dataset": bundle.name,
                    "surface": surface,
                    "component": component,
                    "sum_squares": component_ss,
                    "fraction_of_grand_centered_sum_squares": anova[
                        "component_fractions"
                    ][component],
                    "fraction_closure_error": anova["fraction_closure_error"],
                    "maximum_absolute_pairwise_inner_product_fraction": anova[
                        "maximum_absolute_pairwise_inner_product_fraction"
                    ],
                    "maximum_absolute_reconstruction_error": anova[
                        "maximum_absolute_reconstruction_error"
                    ],
                })
        dataset_rows.append({
            "dataset": bundle.name,
            "n_ligands": int(len(bundle.spectral_matrix)),
            "descriptor_support_ligands": int(len(bundle.matrix)),
            "n_targets": int(len(bundle.targets)),
            "chemical_groups": int(len(np.unique(bundle.groups))),
            "input_rows": bundle.input_rows,
            "missing_cells_before_preprocessing": (
                bundle.missing_cells_before_preprocessing
            ),
            "positive_cells_clipped": bundle.positive_cells_clipped,
            "support_rule": bundle.support_rule,
            "raw_pr": analysis["raw_moments"]["participation_ratio"],
            "residual_pr": analysis["residual_moments"]["participation_ratio"],
            "pr_expansion_ratio": analysis["transition"]["pr_expansion_ratio"],
            "ligand_main_effect_fraction_of_grand_centered_ss": analysis[
                "orthogonal_two_way_anova"
            ]["component_fractions"]["ligand_main_effect"],
            "target_main_effect_fraction_of_grand_centered_ss": analysis[
                "orthogonal_two_way_anova"
            ]["component_fractions"]["target_main_effect"],
            "interaction_fraction_of_grand_centered_ss": analysis[
                "orthogonal_two_way_anova"
            ]["component_fractions"]["ligand_target_interaction"],
            "combined_ligand_and_target_main_effect_fraction_of_grand_centered_ss": (
                analysis["orthogonal_two_way_anova"][
                    "combined_ligand_and_target_main_effect_fraction"
                ]
            ),
            "column_standardized_ligand_main_effect_fraction": analysis[
                "column_standardized_orthogonal_two_way_anova"
            ]["component_fractions"]["ligand_main_effect"],
            "column_standardized_interaction_fraction": analysis[
                "column_standardized_orthogonal_two_way_anova"
            ]["component_fractions"]["ligand_target_interaction"],
            "oof_pr_after_descriptor_removal": analysis["descriptor"]["metrics"][
                "out_of_fold_pr_after_descriptor_removal"
            ],
            "oof_residual_pc1_descriptor_r2": analysis["descriptor"]["metrics"][
                "out_of_fold_residual_pc1_r2"
            ],
        })

    docking_missing = (
        pd.read_csv(FROZEN / "df_final_v4.csv.gz", usecols=DOCK44)
        .apply(pd.to_numeric, errors="coerce")
        .isna()
        .sum()
        .reindex(DOCK44)
        .to_numpy(dtype=np.int64)
    )
    pocket_summary, pocket_rows = exploratory_pocket_volume_analysis(
        bundles[0], docking_missing, config
    )
    cross_panel_summary, cross_panel_frames = cross_panel_descriptor_fingerprint(
        bundles[0], bundles[1]
    )
    box_summary, box_frames = dockstring_box_size_control(bundles[1])

    frames = {
        "dataset_metrics.csv": pd.DataFrame(dataset_rows),
        "correlation_moment_decomposition.csv": pd.DataFrame(moment_rows),
        "uniform_axis_diagnostics.csv": pd.DataFrame(uniform_rows),
        "descriptor_decomposition.csv": pd.DataFrame(descriptor_rows),
        "fold_diagnostics.csv": pd.DataFrame(fold_rows),
        "target_descriptor_r2.csv": pd.DataFrame(target_r2_rows),
        "family_qap.csv": pd.DataFrame(qap_rows),
        "variance_reweighting.csv": pd.DataFrame(variance_rows),
        "orthogonal_two_way_anova.csv": pd.DataFrame(anova_rows),
        "pocket_volume_exploratory.csv": pocket_rows,
        **cross_panel_frames,
        **box_frames,
    }
    for filename, frame in frames.items():
        write_csv(frame, output_dir / filename)

    input_paths = {
        "docking44": FROZEN / "df_final_v4.csv.gz",
        "dockstring58": FROZEN / "dockstring-dataset.tsv.gz",
        "target_families": PACKAGE / "data" / "target_families.csv",
        "pocket_descriptors": (
            FROZEN / "toxicodynamics" / "data" / "44pockets_analysis.csv"
        ),
    }
    summary = {
        "analysis": "residual_mechanism_analysis",
        "analysis_version": "1.2.0",
        "claim_boundary": (
            "The descriptor analysis characterizes reproducible score structure. "
            "It does not validate docking accuracy or biological specificity."
        ),
        "config": asdict(config),
        "descriptor_names": DESCRIPTOR_NAMES,
        "datasets": {
            bundle.name: {
                "n_ligands": int(len(bundle.spectral_matrix)),
                "descriptor_support_ligands": int(len(bundle.matrix)),
                "n_targets": int(len(bundle.targets)),
                "support_rule": bundle.support_rule,
                "group_definition": bundle.group_definition,
                "raw_correlation_moments": analyses[bundle.name]["raw_moments"],
                "residual_correlation_moments": analyses[bundle.name][
                    "residual_moments"
                ],
                "raw_to_residual_transition": analyses[bundle.name]["transition"],
                "uniform_axis": analyses[bundle.name]["uniform_axis"],
                "descriptor_decomposition": analyses[bundle.name]["descriptor"][
                    "metrics"
                ],
                "variance_reweighting": analyses[bundle.name][
                    "variance_reweighting"
                ],
                "orthogonal_two_way_anova": analyses[bundle.name][
                    "orthogonal_two_way_anova"
                ],
                "column_standardized_orthogonal_two_way_anova": analyses[
                    bundle.name
                ]["column_standardized_orthogonal_two_way_anova"],
            }
            for bundle in bundles
        },
        "exploratory_pocket_volume": pocket_summary,
        "cross_panel_descriptor_fingerprint": cross_panel_summary,
        "dockstring_box_size_control": box_summary,
        "input_contracts": {
            name: {
                "path": str(path.relative_to(PACKAGE)),
                "sha256": sha256_file(path),
            }
            for name, path in input_paths.items()
        },
        "output_contracts": {
            filename: {
                "path": str((output_dir / filename).relative_to(PACKAGE)),
                "sha256": sha256_file(output_dir / filename),
                "rows": int(len(frame)),
            }
            for filename, frame in frames.items()
        },
    }
    write_json(output_dir / "analysis_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dockstring-sample-size", type=int, default=15_000)
    parser.add_argument("--dockstring-sample-seed", type=int, default=71)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--qap-permutations", type=int, default=10_000)
    parser.add_argument("--qap-seed", type=int, default=202_608_02)
    parser.add_argument("--pocket-permutations", type=int, default=100_000)
    parser.add_argument("--pocket-bootstrap-replicates", type=int, default=5_000)
    parser.add_argument("--pocket-seed", type=int, default=202_608_03)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = AnalysisConfig(
        dockstring_sample_size=args.dockstring_sample_size,
        dockstring_sample_seed=args.dockstring_sample_seed,
        folds=args.folds,
        qap_permutations=args.qap_permutations,
        qap_seed=args.qap_seed,
        pocket_permutations=args.pocket_permutations,
        pocket_bootstrap_replicates=args.pocket_bootstrap_replicates,
        pocket_seed=args.pocket_seed,
    )
    summary = run_analysis(config, args.output_dir)
    compact = {
        dataset: {
            "raw_pr": record["raw_correlation_moments"]["participation_ratio"],
            "residual_pr": record["residual_correlation_moments"][
                "participation_ratio"
            ],
            "oof_error_pr": record["descriptor_decomposition"][
                "out_of_fold_pr_after_descriptor_removal"
            ],
            "oof_pc1_r2": record["descriptor_decomposition"][
                "out_of_fold_residual_pc1_r2"
            ],
        }
        for dataset, record in summary["datasets"].items()
    }
    print(json.dumps(compact, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
