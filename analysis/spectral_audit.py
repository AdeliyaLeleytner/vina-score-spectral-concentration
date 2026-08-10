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
from scipy import stats

from build_evidence import (
    additive_main_effect_null,
    centering_ladder,
    empirical_residual_permutation_null,
    parallel_analysis,
    row_norm_preserving_residual_null,
    scaffold_keys,
    surface_bootstrap,
    target_correlation_matrix,
    target_jackknife,
    two_way_center,
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
    parser.add_argument(
        "--plot-prefix",
        type=Path,
        help="optional prefix for publication-size PDF and PNG diagnostic plots",
    )
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
    parser.add_argument(
        "--recovery-sizes",
        default="50,100,200,500",
        help=(
            "comma-separated same-support ligand counts for the map-recovery curve; "
            "values larger than the input are skipped"
        ),
    )
    parser.add_argument(
        "--recovery-repeats",
        type=int,
        default=100,
        help="simple-random subsamples per recovery-curve point",
    )
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
    if len(columns) < 3:
        raise ValueError(
            "at least three target score columns are required because map recovery "
            "needs more than one target-pair edge"
        )
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
    if not np.isfinite(scores.to_numpy(dtype=np.float64)).all():
        raise ValueError("score columns must contain only finite values after imputation")
    clusters = None
    if args.smiles_column:
        if args.smiles_column not in frame:
            raise ValueError(f"SMILES column not found: {args.smiles_column}")
        clusters = scaffold_keys(frame[args.smiles_column])
    return scores, clusters


def make_diagnostic_plot(
    matrix: np.ndarray,
    target_names: list[str],
    prefix: Path,
) -> None:
    """Write a compact spectrum, correlation-distribution and heatmap audit."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    raw_corr = target_correlation_matrix(matrix)
    residual_corr = target_correlation_matrix(two_way_center(matrix))
    raw_values = raw_corr[np.triu_indices(len(raw_corr), 1)]
    residual_values = residual_corr[np.triu_indices(len(residual_corr), 1)]
    raw_eigenvalues = np.linalg.eigvalsh(raw_corr)[::-1] / len(raw_corr)
    residual_eigenvalues = np.linalg.eigvalsh(residual_corr)[::-1] / len(residual_corr)
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.2), constrained_layout=True)
    for values, label, color in [
        (raw_eigenvalues, "column-standardized", "#2C6FAD"),
        (residual_eigenvalues, "residual", "#E07A3E"),
    ]:
        axes[0, 0].plot(np.arange(1, len(values) + 1), np.clip(values, 1e-8, None),
                        marker="o", ms=2.5, lw=1, label=label, color=color)
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_xlabel("eigenvalue rank")
    axes[0, 0].set_ylabel("variance fraction (log scale)")
    axes[0, 0].legend(frameon=False)

    bins = np.linspace(-1, 1, 42)
    axes[0, 1].hist(raw_values, bins=bins, density=True, histtype="step",
                    color="#2C6FAD", label="column-standardized")
    axes[0, 1].hist(residual_values, bins=bins, density=True, histtype="stepfilled",
                    alpha=0.25, color="#3E8E8A", label="residual")
    axes[0, 1].set_xlabel("off-diagonal target correlation")
    axes[0, 1].set_ylabel("density")
    axes[0, 1].legend(frameon=False)

    for ax, corr, title in [
        (axes[1, 0], raw_corr, "column-standardized correlations"),
        (axes[1, 1], residual_corr, "residual correlations"),
    ]:
        artist = ax.imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r", interpolation="nearest")
        tick_count = min(8, len(target_names))
        ticks = np.unique(np.linspace(0, len(target_names) - 1, tick_count).round().astype(int))
        ax.set_xticks(ticks, np.asarray(target_names)[ticks], rotation=90, fontsize=6)
        ax.set_yticks(ticks, np.asarray(target_names)[ticks], fontsize=6)
        ax.set_title(title)
        fig.colorbar(artist, ax=ax, fraction=0.046, pad=0.03)
    for label, ax in zip("abcd", axes.flat):
        ax.text(-0.12, 1.04, label, transform=ax.transAxes, fontweight="bold")
    prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(prefix.with_suffix(".pdf"))
    fig.savefig(prefix.with_suffix(".png"), dpi=300)
    plt.close(fig)


def map_recovery_curve(
    matrix: np.ndarray,
    sample_sizes: list[int],
    *,
    repeats: int,
    seed: int,
) -> dict[str, object]:
    """Estimate support-specific recovery of the full residual target-correlation map.

    The reference and every probe are computed from the same input library.  This is a
    convergence diagnostic for that support, target panel and preprocessing pipeline; it is
    not a promise that the same ligand count transports to a shifted chemical domain.
    """

    if repeats < 1:
        raise ValueError("recovery repeats must be positive")
    n_ligands, n_targets = matrix.shape
    if n_ligands < 3:
        raise ValueError("map recovery requires at least three ligand rows")
    if n_targets < 3:
        raise ValueError("map recovery requires at least three target columns")
    full_corr = target_correlation_matrix(two_way_center(matrix))
    if full_corr.shape != (n_targets, n_targets):
        raise ValueError(
            "map recovery requires every target to retain non-zero variance after "
            "within-ligand centering"
        )
    triangle = np.triu_indices(n_targets, 1)
    full_edges = full_corr[triangle]
    rng = np.random.default_rng(seed)
    points: dict[str, dict[str, float | int]] = {}
    for size in sorted(set(sample_sizes)):
        if size < 3 or size > n_ligands:
            continue
        agreements = np.empty(repeats, dtype=np.float64)
        for repeat in range(repeats):
            rows = rng.choice(n_ligands, size=size, replace=False)
            probe_corr = target_correlation_matrix(two_way_center(matrix[rows]))
            if probe_corr.shape != full_corr.shape:
                raise ValueError(
                    "a recovery subsample produced a zero-variance target; increase "
                    "the requested ligand count or inspect the input columns"
                )
            agreement = float(
                stats.spearmanr(probe_corr[triangle], full_edges).statistic
            )
            if not np.isfinite(agreement):
                raise ValueError(
                    "map recovery is undefined because the target-pair correlation "
                    "vectors are constant"
                )
            agreements[repeat] = agreement
        points[str(size)] = {
            "ligands": int(size),
            "repeats": int(repeats),
            "mean_spearman": float(np.mean(agreements)),
            "q025_spearman": float(np.quantile(agreements, 0.025)),
            "q975_spearman": float(np.quantile(agreements, 0.975)),
        }
    if not points:
        raise ValueError(
            "no valid recovery size remains; request at least one size between 3 "
            f"and the input ligand count ({n_ligands})"
        )
    return {
        "points": points,
        "reference_ligands": int(n_ligands),
        "targets": int(n_targets),
        "seed": int(seed),
        "interpretation_boundary": (
            "same-support convergence diagnostic; choose a ligand count from this curve "
            "for the present matrix; 200 ligands is not a universal threshold"
        ),
    }


def main() -> None:
    args = parse_args()
    recovery_sizes = [
        int(value.strip())
        for value in args.recovery_sizes.split(",")
        if value.strip()
    ]
    if not recovery_sizes:
        raise ValueError("at least one recovery size is required")
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
        "same_support_map_recovery": map_recovery_curve(
            matrix,
            recovery_sizes,
            repeats=args.recovery_repeats,
            seed=args.seed + 25,
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
        "row_norm_preserving_residual_null": row_norm_preserving_residual_null(
            matrix, **{**common, "seed": args.seed + 40}
        ),
        "interpretation_boundary": (
            "effective dimension diagnoses variance concentration; it does not validate "
            "affinity, selectivity, or within-ligand target ranking"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    if args.plot_prefix:
        make_diagnostic_plot(matrix, list(scores.columns), args.plot_prefix)
        print(
            f"Wrote {args.plot_prefix.with_suffix('.pdf')} and "
            f"{args.plot_prefix.with_suffix('.png')}"
        )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
