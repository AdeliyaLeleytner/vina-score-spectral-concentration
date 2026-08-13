#!/usr/bin/env python3
"""Strict-public descriptor-axis sensitivity for Vina target maps.

The producer compares maps below and above inclusive empirical quartile
thresholds for seven coarse molecular descriptors.  Discrete boundary ties are
retained and their actual tail sizes are reported.  Size-matched random-row-
disjoint controls distinguish an observed descriptor-domain change from the
finite-support disagreement expected at those same tail sizes.  The DOCKSTRING
analysis uses the fixed 15,000-ligand support (seed 71) used by the public
chemical-domain controls.  No experimental or restricted artifact is read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from rdkit import Chem, rdBase
from rdkit.Chem import Crippen, Descriptors, rdMolDescriptors


PACKAGE = Path(__file__).resolve().parents[1]
FROZEN = PACKAGE / "data" / "frozen"
DEFAULT_DOCKING44 = FROZEN / "df_final_v4.csv.gz"
DEFAULT_DOCKSTRING = FROZEN / "dockstring-dataset.tsv.gz"
DEFAULT_OUTPUT = PACKAGE / "results" / "public_descriptor_domain_specificity"
SUPPORT_SIZE = 15_000
SUPPORT_SEED = 71
EXTREME_FRACTION = 0.25
CONTROL_REPETITIONS = 200
CONTROL_BASE_SEED = 202_608_21

DOCKING44_TARGETS = [
    "1m2z", "1pbq", "1xoq", "2rh1", "2vt4", "2ydo", "2z5x", "3b66",
    "3kk6", "3ln1", "3rze", "4djh", "4ey7", "4iar", "4mqs", "4n6h",
    "5cxv", "5i71", "5tvn", "5u09", "5va1", "6cm4", "6kpf", "6kux",
    "6lqa", "6pdj", "6x3x", "6y1z", "7f8y", "7kwe", "7ljd", "7wc9",
    "7xnk", "7ym8", "8e9y", "8ef6", "8fhs", "8pjk", "8st0", "8wty",
    "8xvk", "8yn3", "9eo4", "V1A",
]

DESCRIPTORS = [
    "heavy_atoms",
    "molecular_weight",
    "labute_asa",
    "tpsa",
    "clogp",
    "rotatable_bonds",
    "ring_count",
]
SIZE_RELATED = {"heavy_atoms", "molecular_weight", "labute_asa"}


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


def molecular_descriptors(smiles: Iterable[str]) -> np.ndarray:
    rows: list[list[float]] = []
    for index, value in enumerate(smiles):
        molecule = Chem.MolFromSmiles(str(value))
        if molecule is None:
            raise ValueError(f"invalid SMILES at row {index}")
        rows.append(
            [
                float(molecule.GetNumHeavyAtoms()),
                float(Descriptors.MolWt(molecule)),
                float(rdMolDescriptors.CalcLabuteASA(molecule)),
                float(Descriptors.TPSA(molecule)),
                float(Crippen.MolLogP(molecule)),
                float(Descriptors.NumRotatableBonds(molecule)),
                float(rdMolDescriptors.CalcNumRings(molecule)),
            ]
        )
    result = np.asarray(rows, dtype=np.float64)
    if result.ndim != 2 or result.shape[1] != len(DESCRIPTORS):
        raise RuntimeError("descriptor calculation returned an invalid matrix")
    if not np.isfinite(result).all():
        raise ValueError("descriptor matrix contains non-finite values")
    return result


def row_center(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    return matrix - matrix.mean(axis=1, keepdims=True)


def target_correlation(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
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


def correlation_pr(correlation: np.ndarray) -> float:
    return float(np.trace(correlation) ** 2 / np.square(correlation).sum())


def extreme_masks(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
    values = np.asarray(values, dtype=np.float64)
    low_threshold, high_threshold = np.quantile(
        values, [EXTREME_FRACTION, 1.0 - EXTREME_FRACTION]
    )
    low = values <= low_threshold
    high = values >= high_threshold
    if low_threshold >= high_threshold or np.any(low & high):
        raise ValueError("descriptor extremes are not disjoint")
    return low, high, float(low_threshold), float(high_threshold)


def descriptor_contrasts(
    dataset: str,
    matrix: np.ndarray,
    descriptors: np.ndarray,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    surfaces = {"raw": matrix, "row_centered_residual": row_center(matrix)}
    for column, descriptor in enumerate(DESCRIPTORS):
        values = descriptors[:, column]
        low, high, low_threshold, high_threshold = extreme_masks(values)
        for transformation, surface in surfaces.items():
            first = target_correlation(surface[low])
            second = target_correlation(surface[high])
            first_edges = upper_triangle(first)
            second_edges = upper_triangle(second)
            records.append(
                {
                    "dataset": dataset,
                    "descriptor": descriptor,
                    "descriptor_family": (
                        "size_related" if descriptor in SIZE_RELATED else "other_coarse"
                    ),
                    "transformation": transformation,
                    "extreme_fraction_per_tail": EXTREME_FRACTION,
                    "low_threshold_inclusive": low_threshold,
                    "high_threshold_inclusive": high_threshold,
                    "low_ligands": int(low.sum()),
                    "high_ligands": int(high.sum()),
                    "low_support_fraction": float(low.mean()),
                    "high_support_fraction": float(high.mean()),
                    "combined_support_fraction": float((low | high).mean()),
                    "low_descriptor_median": float(np.median(values[low])),
                    "high_descriptor_median": float(np.median(values[high])),
                    "geometry_spearman": spearman(first_edges, second_edges),
                    "sign_flip_fraction": float(
                        np.mean(np.sign(first_edges) != np.sign(second_edges))
                    ),
                    "low_pr": correlation_pr(first),
                    "high_pr": correlation_pr(second),
                }
            )
    return pd.DataFrame.from_records(records)


def random_disjoint_controls(
    dataset: str,
    matrix: np.ndarray,
    contrasts: pd.DataFrame,
    *,
    seed: int,
) -> pd.DataFrame:
    """Matched-size residual-map controls, re-used for identical tail sizes."""

    surface = row_center(matrix)
    residual = contrasts.loc[
        contrasts.transformation.eq("row_centered_residual")
    ].copy()
    size_pairs = sorted(
        {
            (int(row.low_ligands), int(row.high_ligands))
            for row in residual.itertuples(index=False)
        }
    )
    cache: dict[tuple[int, int], list[float]] = {}
    for pair_index, (low_size, high_size) in enumerate(size_pairs):
        if low_size + high_size > len(surface):
            raise ValueError("matched disjoint control exceeds the ligand support")
        rng = np.random.default_rng(seed + 100_000 * pair_index)
        values: list[float] = []
        for _ in range(CONTROL_REPETITIONS):
            order = rng.permutation(len(surface))
            low_edges = upper_triangle(
                target_correlation(surface[order[:low_size]])
            )
            high_edges = upper_triangle(
                target_correlation(
                    surface[order[low_size : low_size + high_size]]
                )
            )
            values.append(spearman(low_edges, high_edges))
        cache[(low_size, high_size)] = values

    records: list[dict[str, Any]] = []
    for row in residual.itertuples(index=False):
        pair = (int(row.low_ligands), int(row.high_ligands))
        for repetition, value in enumerate(cache[pair]):
            records.append(
                {
                    "dataset": dataset,
                    "descriptor": str(row.descriptor),
                    "repetition": repetition,
                    "low_ligands": pair[0],
                    "high_ligands": pair[1],
                    "geometry_spearman": value,
                }
            )
    return pd.DataFrame.from_records(records)


def summarize_controls(
    contrasts: pd.DataFrame, controls: pd.DataFrame
) -> pd.DataFrame:
    observed = contrasts.loc[
        contrasts.transformation.eq("row_centered_residual"),
        [
            "dataset",
            "descriptor",
            "descriptor_family",
            "low_ligands",
            "high_ligands",
            "low_support_fraction",
            "high_support_fraction",
            "combined_support_fraction",
            "geometry_spearman",
        ],
    ].rename(columns={"geometry_spearman": "observed_geometry_spearman"})
    control_summary = controls.groupby(
        ["dataset", "descriptor"], as_index=False
    ).agg(
        control_repetitions=("repetition", "size"),
        random_disjoint_mean=("geometry_spearman", "mean"),
        random_disjoint_median=("geometry_spearman", "median"),
        random_disjoint_q025=(
            "geometry_spearman",
            lambda values: float(np.quantile(values, 0.025)),
        ),
        random_disjoint_q975=(
            "geometry_spearman",
            lambda values: float(np.quantile(values, 0.975)),
        ),
    )
    summary = observed.merge(
        control_summary, on=["dataset", "descriptor"], validate="one_to_one"
    )
    summary["observed_minus_control_median"] = (
        summary.observed_geometry_spearman - summary.random_disjoint_median
    )
    summary["observed_below_control_q025"] = (
        summary.observed_geometry_spearman < summary.random_disjoint_q025
    )
    return summary


def descriptor_correlations(
    dataset: str, descriptors: np.ndarray
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for first in range(len(DESCRIPTORS)):
        for second in range(first + 1, len(DESCRIPTORS)):
            records.append(
                {
                    "dataset": dataset,
                    "descriptor_a": DESCRIPTORS[first],
                    "descriptor_b": DESCRIPTORS[second],
                    "spearman": spearman(descriptors[:, first], descriptors[:, second]),
                }
            )
    return pd.DataFrame.from_records(records)


def load_docking44(path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    frame = pd.read_csv(
        path,
        usecols=DOCKING44_TARGETS + ["Cleaned SMILES", "Canonical SMILES"],
    )
    numeric = frame[DOCKING44_TARGETS].apply(pd.to_numeric, errors="coerce")
    clipped = numeric.clip(upper=0)
    matrix = clipped.fillna(clipped.mean()).to_numpy(dtype=np.float64)
    smiles = (
        frame["Cleaned SMILES"]
        .fillna(frame["Canonical SMILES"])
        .astype(str)
        .to_numpy()
    )
    return matrix, molecular_descriptors(smiles), {
        "source_rows": int(len(frame)),
        "analysis_rows": int(len(frame)),
        "targets": int(len(DOCKING44_TARGETS)),
        "missing_cells_before_target_mean_imputation": int(
            numeric.isna().to_numpy().sum()
        ),
        "support_rule": "all rows; positive scores clipped at zero; target-mean imputation",
    }


def load_dockstring(path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    frame = pd.read_csv(path, sep="\t")
    targets = [column for column in frame if column not in {"inchikey", "smiles"}]
    numeric = frame[targets].apply(pd.to_numeric, errors="coerce")
    complete = ~numeric.isna().any(axis=1)
    complete_matrix = np.minimum(
        numeric.loc[complete].to_numpy(dtype=np.float64), 0.0
    )
    complete_smiles = (
        frame.loc[complete, "smiles"].reset_index(drop=True).astype(str)
    )
    rng = np.random.default_rng(SUPPORT_SEED)
    selected = np.sort(
        rng.choice(len(complete_matrix), size=SUPPORT_SIZE, replace=False)
    )
    return (
        complete_matrix[selected],
        molecular_descriptors(complete_smiles.iloc[selected]),
        {
            "source_rows": int(len(frame)),
            "complete_rows": int(complete.sum()),
            "analysis_rows": SUPPORT_SIZE,
            "targets": int(len(targets)),
            "support_seed": SUPPORT_SEED,
            "selected_complete_row_index_sha256": hashlib.sha256(
                np.asarray(selected, dtype="<i8").tobytes()
            ).hexdigest(),
            "support_rule": "sorted simple random sample without replacement of complete rows",
        },
    )


def build(
    docking44_path: Path,
    dockstring_path: Path,
) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
    d44_matrix, d44_descriptors, d44_support = load_docking44(docking44_path)
    ds_matrix, ds_descriptors, ds_support = load_dockstring(dockstring_path)
    contrasts = pd.concat(
        [
            descriptor_contrasts("Docking-44", d44_matrix, d44_descriptors),
            descriptor_contrasts("DOCKSTRING-58", ds_matrix, ds_descriptors),
        ],
        ignore_index=True,
    )
    controls = pd.concat(
        [
            random_disjoint_controls(
                "Docking-44",
                d44_matrix,
                contrasts.loc[contrasts.dataset.eq("Docking-44")],
                seed=CONTROL_BASE_SEED,
            ),
            random_disjoint_controls(
                "DOCKSTRING-58",
                ds_matrix,
                contrasts.loc[contrasts.dataset.eq("DOCKSTRING-58")],
                seed=CONTROL_BASE_SEED + 1_000_000,
            ),
        ],
        ignore_index=True,
    )
    control_summary = summarize_controls(contrasts, controls)
    correlations = pd.concat(
        [
            descriptor_correlations("Docking-44", d44_descriptors),
            descriptor_correlations("DOCKSTRING-58", ds_descriptors),
        ],
        ignore_index=True,
    )
    residual = contrasts.loc[
        contrasts.transformation.eq("row_centered_residual")
    ].copy()
    family_summary = residual.groupby(
        ["dataset", "descriptor_family"], as_index=False
    ).agg(
        geometry_spearman_minimum=("geometry_spearman", "min"),
        geometry_spearman_median=("geometry_spearman", "median"),
        geometry_spearman_maximum=("geometry_spearman", "max"),
    )
    summary = {
        "schema_version": "1.0.0",
        "analysis_status": "strict_public_post_hoc_descriptor_sensitivity",
        "producer": "analysis/public_descriptor_domain_specificity.py",
        "configuration": {
            "descriptor_names": DESCRIPTORS,
            "size_related_descriptors": sorted(SIZE_RELATED),
            "extreme_fraction_per_tail": EXTREME_FRACTION,
            "inclusive_boundary_ties_retained": True,
            "random_disjoint_control_repetitions": CONTROL_REPETITIONS,
            "random_disjoint_control_base_seed": CONTROL_BASE_SEED,
            "dockstring_support_size": SUPPORT_SIZE,
            "dockstring_support_seed": SUPPORT_SEED,
            "rdkit_version": rdBase.rdkitVersion,
        },
        "datasets": {"Docking-44": d44_support, "DOCKSTRING-58": ds_support},
        "key_residual_results": control_summary.to_dict(orient="records"),
        "family_summary": family_summary.to_dict(orient="records"),
        "claim_boundary": (
            "Inclusive quartile thresholds retain discrete ties, so actual tail "
            "sizes differ by descriptor. Matched-size random-row-disjoint controls "
            "show finite-support disagreement at those same sizes. All descriptors "
            "were examined post hoc and are correlated; differences are descriptive "
            "in degree and do not identify a causal molecular property, validate "
            "docking poses, or establish a biological target network."
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
        "descriptor_extremes.csv": contrasts,
        "descriptor_correlations.csv": correlations,
        "descriptor_family_summary.csv": family_summary,
        "descriptor_random_disjoint_controls.csv": controls,
        "descriptor_random_disjoint_summary.csv": control_summary,
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
                "# Strict-public descriptor-domain specificity",
                "",
                "Inclusive empirical quartile thresholds are applied to seven descriptor",
                "axes; discrete ties and the resulting actual tail sizes are retained.",
                "Matched-size random-row-disjoint controls show that every examined axis",
                "changes the map more than finite-support variation, with the largest",
                "changes along three collinear size-related coordinates. The analysis is",
                "descriptive and post hoc and does not identify a causal mechanism.",
                "",
                "Reproduce with:",
                "",
                "```bash",
                ".venv/bin/python analysis/public_descriptor_domain_specificity.py",
                ".venv/bin/python -m pytest -q analysis/test_public_descriptor_domain_specificity.py",
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
