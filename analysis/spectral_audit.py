#!/usr/bin/env python3
"""Run the manuscript's reusable spectral-audit protocol on a score matrix.

The input is a rectangular CSV or TSV table with one row per ligand and one numeric column
per target. Identifier and SMILES columns may be retained in the same table and excluded by
name. The output is a machine-readable JSON report; no manuscript-specific frozen inputs are
loaded.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from build_evidence import (
    additive_main_effect_null,
    centering_ladder,
    empirical_residual_permutation_null,
    parallel_analysis,
    scaffold_keys,
    surface_bootstrap,
    target_jackknife,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit column-standardized and two-way-centered spectra, uncertainty, and "
            "transformation-matched nulls for a dense ligand-by-target score matrix."
        )
    )
    parser.add_argument("input", type=Path, help="CSV or TSV score table")
    parser.add_argument("--output", type=Path, required=True, help="JSON report path")
    parser.add_argument("--delimiter", choices=["csv", "tsv"], help="infer from suffix by default")
    parser.add_argument("--id-column", help="non-score identifier column")
    parser.add_argument("--smiles-column", help="SMILES column for Murcko-cluster bootstrap")
    parser.add_argument(
        "--score-columns",
        help="comma-separated score columns; default is every column except ID and SMILES",
    )
    parser.add_argument(
        "--missing",
        choices=["column-mean", "error"],
        default="column-mean",
        help="missing-score policy",
    )
    parser.add_argument(
        "--clip-positive-to-zero",
        action="store_true",
        help="apply only when positive values are known source-censored docking failures",
    )
    parser.add_argument("--sample-size", type=int, default=12000)
    parser.add_argument("--permutations", type=int, default=500)
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def load_matrix(args: argparse.Namespace) -> tuple[pd.DataFrame, np.ndarray | None]:
    separator = "\t" if args.delimiter == "tsv" or args.input.suffix.lower() == ".tsv" else ","
    frame = pd.read_csv(args.input, sep=separator)
    excluded = {value for value in [args.id_column, args.smiles_column] if value}
    columns = (
        [value.strip() for value in args.score_columns.split(",") if value.strip()]
        if args.score_columns
        else [column for column in frame.columns if column not in excluded]
    )
    if len(columns) < 2:
        raise ValueError("at least two target score columns are required")
    missing_columns = sorted(set(columns) - set(frame.columns))
    if missing_columns:
        raise ValueError(f"score columns not found: {missing_columns}")
    scores = frame[columns].apply(pd.to_numeric, errors="coerce")
    if args.clip_positive_to_zero:
        scores = scores.clip(upper=0)
    missing_count = int(scores.isna().sum().sum())
    if missing_count and args.missing == "error":
        raise ValueError(f"matrix contains {missing_count} non-numeric or missing score cells")
    if missing_count:
        scores = scores.fillna(scores.mean(axis=0))
    if scores.isna().any().any():
        raise ValueError("at least one target column has no finite value")
    clusters = None
    if args.smiles_column:
        if args.smiles_column not in frame:
            raise ValueError(f"SMILES column not found: {args.smiles_column}")
        clusters = scaffold_keys(frame[args.smiles_column])
    return scores, clusters


def main() -> None:
    args = parse_args()
    scores, clusters = load_matrix(args)
    matrix = scores.to_numpy(dtype=np.float64)
    common = {
        "sample_size": args.sample_size,
        "repeats": args.permutations,
        "seed": args.seed,
    }
    report = {
        "input": {
            "path": str(args.input),
            "n_ligands": int(len(scores)),
            "n_targets": int(scores.shape[1]),
            "target_columns": list(scores.columns),
            "missing_policy": args.missing,
            "clip_positive_to_zero": bool(args.clip_positive_to_zero),
            "scaffold_clusters": int(len(np.unique(clusters))) if clusters is not None else None,
        },
        "centering_ladder": centering_ladder(matrix),
        "paired_ligand_or_cluster_bootstrap": surface_bootstrap(
            matrix,
            repeats=args.bootstrap,
            seed=args.seed + 10,
            cluster_labels=clusters,
        ),
        "target_jackknife": target_jackknife(
            matrix,
            list(scores.columns),
            sample_size=args.sample_size,
            seed=args.seed + 20,
        ),
        "parallel_analysis": parallel_analysis(
            matrix,
            sample_size=args.sample_size,
            repeats=args.permutations,
            series=5 if args.permutations % 5 == 0 else 1,
            seed=args.seed + 30,
        ),
        "additive_gaussian_null": additive_main_effect_null(
            matrix, **{**common, "seed": args.seed + 40}
        ),
        "empirical_residual_permutation_null": empirical_residual_permutation_null(
            matrix, **{**common, "seed": args.seed + 40}
        ),
        "interpretation_boundary": (
            "effective dimension diagnoses variance concentration; it does not validate "
            "affinity, selectivity, or within-ligand target ranking"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
