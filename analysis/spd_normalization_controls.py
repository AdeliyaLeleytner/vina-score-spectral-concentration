#!/usr/bin/env python3
"""Hostile normalization controls for the frozen SPD target-geometry result."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

try:
    from . import spd_external_validation as spd
except ImportError:  # pragma: no cover
    import spd_external_validation as spd  # type: ignore


DEFAULT_OUTPUT = spd.PACKAGE / "results" / "spd_normalization_controls"
DEFAULT_PERMUTATIONS = 10_000


def representation_geometries(
    scores: np.ndarray, heavy_atoms: np.ndarray
) -> dict[str, np.ndarray]:
    """Return five explicitly defined target-correlation geometries.

    Ligand efficiency is Vina kcal/mol divided by RDKit heavy-atom count.  It
    is a row-wise size normalization, not a thermodynamic efficiency estimate.
    """
    scores = np.asarray(scores, dtype=np.float64)
    heavy_atoms = np.asarray(heavy_atoms, dtype=np.float64)
    if scores.ndim != 2 or heavy_atoms.shape != (len(scores),):
        raise ValueError("score/heavy-atom shapes do not align")
    if not np.isfinite(scores).all() or (heavy_atoms <= 0).any():
        raise ValueError("scores must be finite and heavy-atom counts positive")

    def z_columns(values: np.ndarray) -> np.ndarray:
        scale = values.std(axis=0, ddof=0)
        if (scale == 0).any():
            raise ValueError("constant target column")
        return (values - values.mean(axis=0)) / scale

    raw_z = z_columns(scores)
    centered = raw_z - raw_z.mean(axis=1, keepdims=True)
    ordinal = stats.rankdata(scores, axis=1, method="average")
    ordinal_z = z_columns(ordinal)
    efficiency = scores / heavy_atoms[:, None]
    efficiency_z = z_columns(efficiency)
    efficiency_centered = efficiency_z - efficiency_z.mean(axis=1, keepdims=True)
    surfaces = {
        "raw": raw_z,
        "standard_centered": centered,
        "within_ligand_ordinal": ordinal_z,
        "ligand_efficiency_raw": efficiency_z,
        "ligand_efficiency_centered": efficiency_centered,
    }
    return {name: np.corrcoef(surface, rowvar=False) for name, surface in surfaces.items()}


def heavy_atom_counts(smiles: pd.Series) -> np.ndarray:
    try:
        from rdkit import Chem, RDLogger
    except ImportError as error:  # pragma: no cover
        raise RuntimeError("RDKit is required for ligand-efficiency control") from error
    RDLogger.DisableLog("rdApp.*")
    counts = np.empty(len(smiles), dtype=np.int32)
    for index, value in enumerate(smiles.astype(str)):
        molecule = Chem.MolFromSmiles(value)
        if molecule is None:
            raise ValueError(f"invalid DOCKSTRING SMILES at row {index}")
        counts[index] = molecule.GetNumHeavyAtoms()
    if (counts <= 0).any():
        raise ValueError("zero-heavy-atom molecule")
    return counts


def pair_vectors(
    pairs: pd.DataFrame, geometries: dict[str, np.ndarray]
) -> pd.DataFrame:
    index = {target: position for position, target in enumerate(spd.TARGETS)}
    result = pairs.copy()
    for name, matrix in geometries.items():
        result[name] = [
            matrix[index[a], index[b]]
            for a, b in zip(result.target_a, result.target_b)
        ]
    return result


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    return float(stats.spearmanr(x, y).statistic)


def qap_all(
    pairs: pd.DataFrame,
    geometries: dict[str, np.ndarray],
    *,
    permutations: int,
    seed: int,
) -> pd.DataFrame:
    index = {target: position for position, target in enumerate(spd.TARGETS)}
    first = np.array([index[x] for x in pairs.target_a], dtype=np.int16)
    second = np.array([index[x] for x in pairs.target_b], dtype=np.int16)
    y_rank = stats.zscore(stats.rankdata(pairs.experimental_correlation.to_numpy(float)))
    rng = np.random.default_rng(seed)
    maps = np.stack([rng.permutation(len(spd.TARGETS)) for _ in range(permutations)])
    mapped_first, mapped_second = maps[:, first], maps[:, second]
    observed: dict[str, float] = {}
    nulls: dict[str, np.ndarray] = {}
    for name, matrix in geometries.items():
        observed[name] = spearman(
            pairs.experimental_correlation.to_numpy(float), pairs[name].to_numpy(float)
        )
        values = matrix[mapped_first, mapped_second]
        ranked = stats.zscore(stats.rankdata(values, axis=1), axis=1)
        nulls[name] = np.mean(ranked * y_rank, axis=1)
    reference = "standard_centered"
    rows: list[dict] = []
    for name in geometries:
        null = nulls[name]
        rows.append(
            {
                "representation": name,
                "contrast": "alignment",
                "observed": observed[name],
                "p_positive": (1 + np.sum(null >= observed[name])) / (permutations + 1),
                "permutations": permutations,
            }
        )
        if name != reference:
            delta = observed[name] - observed[reference]
            null_delta = null - nulls[reference]
            rows.append(
                {
                    "representation": name,
                    "contrast": "minus_standard_centered",
                    "observed": delta,
                    "p_positive": (1 + np.sum(null_delta >= delta))
                    / (permutations + 1),
                    "permutations": permutations,
                }
            )
    return pd.DataFrame(rows)


def run(
    *,
    spd_path: Path,
    dockstring_path: Path,
    output_dir: Path,
    permutations: int,
    seed: int,
) -> dict:
    activity = spd.load_spd(spd_path)
    groups, _ = spd.choose_primary_groups(activity)
    long = spd.selected_long(activity, groups)
    dockstring = spd.load_dockstring(dockstring_path)
    target_complete = dockstring.loc[:, spd.TARGETS].notna().all(axis=1)
    nonoverlap = ~dockstring.connectivity_key.isin(set(long.connectivity_key))
    selected = dockstring.loc[target_complete & nonoverlap].copy()
    # Explicit target labels prevent pandas usecols/file-order errors.
    scores = np.minimum(selected.loc[:, spd.TARGETS].to_numpy(float), 0.0)
    heavy_atoms = heavy_atom_counts(selected.smiles)
    geometries = representation_geometries(scores, heavy_atoms)

    specs = [
        ("floor_at_bound", None),
        ("censor_aware_binary", 10.0),
        ("censor_aware_binary", 30.0),
    ]
    metrics: list[dict] = []
    qaps: list[pd.DataFrame] = []
    jackknives: list[dict] = []
    for endpoint, threshold in specs:
        matrix, support = spd.matrix_from_long(
            long,
            identity="connectivity_key",
            endpoint=endpoint,
            threshold_uM=threshold,
        )
        pairs = spd.pairwise_geometry(
            matrix, min_support=40, transform="column_z_row_center"
        )
        paired = pair_vectors(pairs, geometries)
        label = endpoint if threshold is None else f"{endpoint}_{int(threshold)}uM"
        y = paired.experimental_correlation.to_numpy(float)
        representation_rho = {
            name: spearman(y, paired[name].to_numpy(float)) for name in geometries
        }
        for name, rho in representation_rho.items():
            metrics.append(
                {
                    "endpoint": label,
                    "representation": name,
                    "pairs": len(paired),
                    "spearman": rho,
                    "minus_standard_centered": rho
                    - representation_rho["standard_centered"],
                }
            )
        qap = qap_all(
            paired, geometries, permutations=permutations, seed=seed
        )
        qap.insert(0, "endpoint", label)
        qaps.append(qap)
        for omitted in spd.TARGETS:
            subset = paired.loc[
                (paired.target_a != omitted) & (paired.target_b != omitted)
            ]
            y_jack = subset.experimental_correlation.to_numpy(float)
            reference = spearman(
                y_jack, subset.standard_centered.to_numpy(float)
            )
            for name in geometries:
                rho = spearman(y_jack, subset[name].to_numpy(float))
                jackknives.append(
                    {
                        "endpoint": label,
                        "omitted_target": omitted,
                        "representation": name,
                        "pairs": len(subset),
                        "spearman": rho,
                        "minus_standard_centered": rho - reference,
                    }
                )

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_frame = pd.DataFrame(metrics)
    qap_frame = pd.concat(qaps, ignore_index=True)
    jackknife_frame = pd.DataFrame(jackknives)
    metrics_frame.to_csv(output_dir / "metrics.csv", index=False)
    qap_frame.to_csv(output_dir / "target_label_qap.csv", index=False)
    jackknife_frame.to_csv(output_dir / "target_jackknife.csv", index=False)
    pivot = metrics_frame.pivot(
        index="representation", columns="endpoint", values="spearman"
    )
    summary = {
        "analysis_status": "exploratory_post_hoc_hostile_control",
        "docking_rows": int(len(selected)),
        "spd_connectivity_blocks_excluded": int(long.connectivity_key.nunique()),
        "ligand_efficiency_definition": (
            "clipped Vina kcal/mol divided by RDKit heavy-atom count; column "
            "standardized before target correlation"
        ),
        "results": {
            representation: {
                endpoint: float(value)
                for endpoint, value in row.dropna().items()
            }
            for representation, row in pivot.iterrows()
        },
        "qap": qap_frame.to_dict(orient="records"),
        "interpretation_boundary": (
            "If ordinal or ligand-efficiency geometry matches or exceeds standard "
            "centering, the positive SPD alignment supports within-ligand normalization "
            "generally, not a unique advantage of two-way centering."
        ),
        "parameters": {"permutations": permutations, "seed": seed},
        "sources": {
            "spd_sha256": spd.sha256_file(spd_path),
            "dockstring_sha256": spd.sha256_file(dockstring_path),
        },
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spd", type=Path, default=spd.DEFAULT_SPD)
    parser.add_argument("--dockstring", type=Path, default=spd.DEFAULT_DOCKSTRING)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--permutations", type=int, default=DEFAULT_PERMUTATIONS)
    parser.add_argument("--seed", type=int, default=spd.SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(
        spd_path=args.spd,
        dockstring_path=args.dockstring,
        output_dir=args.output_dir,
        permutations=args.permutations,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
