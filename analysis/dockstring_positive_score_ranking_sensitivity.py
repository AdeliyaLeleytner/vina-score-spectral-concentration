#!/usr/bin/env python3
"""Ligand-wise ranking sensitivity to DOCKSTRING positive-score clipping.

The released DOCKSTRING matrix clips the small number of positive Vina scores
to zero.  This script holds the ChEMBL support, target set, external fitting
contract, ligand weighting and tie handling fixed, and compares the primary
docking-derived representations with the original positive values retained.

The result is a preprocessing sensitivity for observed-pair concordance.  It is
not evidence that positive Vina energies are calibrated physical affinities.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:  # Direct script execution and package-style import are both supported.
    from . import build_evidence as evidence
    from . import dense_davis_benchmark as dense_davis
    from . import dockstring_chembl_ranking_benchmark as benchmark
except ImportError:  # pragma: no cover - direct CLI execution.
    import build_evidence as evidence  # type: ignore
    import dense_davis_benchmark as dense_davis  # type: ignore
    import dockstring_chembl_ranking_benchmark as benchmark  # type: ignore


PACKAGE = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PACKAGE / "results" / "dockstring_positive_score_ranking_sensitivity"
DEFAULT_SEED = 202_608_10
DEFAULT_BOOTSTRAPS = 2_000
DOCKING_REPRESENTATIONS = (
    "absolute_vina",
    "column_standardized",
    "two_way_residual",
    "target_centered_unscaled",
    "two_way_centered_unscaled",
    "target_centered_residual_scaled",
)


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
        + "\n"
    )


def load_reference_variants(
    source_path: Path = benchmark.DOCKSTRING_INPUT,
    identity_path: Path = benchmark.IDENTITY_INPUT,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, dict[str, Any]]:
    """Return aligned unclipped and clipped full-InChIKey references."""
    source = pd.read_csv(source_path, sep="\t")
    score_columns = [
        column for column in source.columns if column not in {"inchikey", "smiles"}
    ]
    numeric = source[score_columns].apply(pd.to_numeric, errors="coerce")
    complete_mask = numeric.notna().all(axis=1)
    complete = source.loc[complete_mask].copy()
    complete_scores = numeric.loc[complete_mask].copy()
    if len(source) != benchmark.EXPECTED_DOCKSTRING_ROWS:
        raise ValueError("DOCKSTRING source row count changed")
    if len(complete) != benchmark.EXPECTED_COMPLETE_ROWS:
        raise ValueError("DOCKSTRING complete-row count changed")

    identity = pd.read_csv(
        identity_path, usecols=["source_row_index", "raw_inchikey"]
    )
    if not np.array_equal(
        identity.source_row_index.to_numpy(dtype=int),
        complete.index.to_numpy(dtype=int),
    ):
        raise ValueError("identity contract is not aligned to complete rows")
    if identity.raw_inchikey.isna().any():
        raise ValueError("identity contract contains a null full InChIKey")

    complete = complete.reset_index(drop=True)
    complete_scores = complete_scores.reset_index(drop=True)
    complete["full_inchikey"] = identity.raw_inchikey.to_numpy()
    complete_scores.index = pd.Index(complete.full_inchikey, name="inchikey")
    unclipped = complete_scores.groupby(level=0, sort=True).median()
    clipped = complete_scores.clip(upper=0.0).groupby(level=0, sort=True).median()
    smiles = (
        complete.groupby("full_inchikey", sort=True).smiles.first().reindex(unclipped.index)
    )
    if not unclipped.index.equals(clipped.index) or not unclipped.index.equals(smiles.index):
        raise ValueError("reference variants are not aligned")
    if unclipped.isna().to_numpy().any() or clipped.isna().to_numpy().any():
        raise ValueError("reference variant contains a missing value")
    report = {
        "released_rows": int(len(source)),
        "complete_rows": int(len(complete)),
        "unique_full_inchikeys": int(len(unclipped)),
        "complete_source_positive_cells": int(
            (complete_scores.to_numpy(dtype=np.float64) > 0).sum()
        ),
        "complete_source_positive_fraction": float(
            (complete_scores.to_numpy(dtype=np.float64) > 0).mean()
        ),
        "aggregated_reference_positive_cells": int(
            (unclipped.to_numpy(dtype=np.float64) > 0).sum()
        ),
        "aggregated_reference_positive_fraction": float(
            (unclipped.to_numpy(dtype=np.float64) > 0).mean()
        ),
        "aggregation": "median over complete rows sharing a standard full InChIKey",
        "clipping": "elementwise minimum(score, 0.0) before aggregation",
    }
    return unclipped, clipped, smiles, report


def informative_pair_order_change(
    clipped_scores: np.ndarray,
    unclipped_scores: np.ndarray,
    experiment: np.ndarray,
) -> dict[str, Any]:
    """Score-order changes on the exact informative experimental target pairs."""
    pair_arrays = evidence.preference_pair_arrays(experiment)
    ligand, first, second, truth = pair_arrays
    informative = truth != 0
    ligand = ligand[informative]
    first = first[informative]
    second = second[informative]
    clipped_difference = clipped_scores[ligand, first] - clipped_scores[ligand, second]
    unclipped_difference = (
        unclipped_scores[ligand, first] - unclipped_scores[ligand, second]
    )
    clipped_sign = np.sign(clipped_difference)
    unclipped_sign = np.sign(unclipped_difference)
    changed = clipped_sign != unclipped_sign
    strict_reversal = (clipped_sign * unclipped_sign) < 0
    return {
        "informative_experimental_pairs": int(len(ligand)),
        "score_order_changed_pairs": int(changed.sum()),
        "score_order_changed_fraction": float(changed.mean()),
        "strict_score_order_reversal_pairs": int(strict_reversal.sum()),
        "strict_score_order_reversal_fraction": float(strict_reversal.mean()),
        "clipped_score_ties": int((clipped_sign == 0).sum()),
        "unclipped_score_ties": int((unclipped_sign == 0).sum()),
    }


def conservative_cluster_interval(
    report: dict[str, Any], level: str = "interval_95"
) -> list[float]:
    intervals = [
        report[source][level]
        for source in ("scaffold_cluster_bootstrap", "butina_cluster_bootstrap")
    ]
    return [
        float(min(interval[0] for interval in intervals)),
        float(max(interval[1] for interval in intervals)),
    ]


def evaluate(
    *,
    activities_path: Path,
    bootstraps: int,
    seed: int,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    gene_to_chembl, chembl_to_gene = benchmark.gene_target_contract()
    unclipped_reference, clipped_reference, smiles, reference_report = (
        load_reference_variants()
    )
    # Byte-level behavior must match the current primary loader before changing
    # the one preprocessing decision under test.
    primary_reference, primary_smiles, _ = benchmark.load_docking_reference()
    if not clipped_reference.index.equals(primary_reference.index):
        raise ValueError("clipped sensitivity reference changed row identity")
    if not clipped_reference.columns.equals(primary_reference.columns):
        raise ValueError("clipped sensitivity reference changed target identity")
    if not np.array_equal(
        clipped_reference.to_numpy(dtype=np.float64),
        primary_reference.to_numpy(dtype=np.float64),
    ):
        raise ValueError("clipped sensitivity reference does not reproduce primary input")
    if not smiles.equals(primary_smiles):
        raise ValueError("clipped sensitivity SMILES do not reproduce primary input")

    target_columns = [
        gene for gene in sorted(gene_to_chembl) if gene in clipped_reference.columns
    ]
    activities, activity_report = benchmark.load_activities(
        chembl_to_gene, activity_path=activities_path
    )
    experimental_reference, experiment, support_report = benchmark.build_support(
        activities, clipped_reference, target_columns
    )
    evaluation_smiles = smiles.reindex(experiment.index)
    if evaluation_smiles.isna().any():
        raise ValueError("an evaluation ligand has no DOCKSTRING SMILES")
    murcko = evidence.scaffold_keys(evaluation_smiles)
    butina = dense_davis._butina_labels(evaluation_smiles)

    score_sets: dict[str, dict[str, np.ndarray]] = {}
    fit_contracts: dict[str, Any] = {}
    for variant, reference in (
        ("positive_scores_clipped", clipped_reference),
        ("positive_scores_retained", unclipped_reference),
    ):
        score_sets[variant], fit_contracts[variant] = benchmark.score_representations(
            reference, experimental_reference, experiment, murcko
        )

    experiment_values = experiment.to_numpy(dtype=np.float64)
    representation_rows: list[dict[str, Any]] = []
    pair_order_rows: list[dict[str, Any]] = []
    paired_reports: dict[str, Any] = {}
    for offset, representation in enumerate(DOCKING_REPRESENTATIONS):
        clipped_metric = evidence.preference_metrics(
            score_sets["positive_scores_clipped"][representation], experiment_values
        )
        unclipped_metric = evidence.preference_metrics(
            score_sets["positive_scores_retained"][representation], experiment_values
        )
        comparison = evidence.paired_preference_comparison(
            unclipped_metric["per_ligand_pairwise_accuracy"],
            clipped_metric["per_ligand_pairwise_accuracy"],
            bootstraps,
            seed + 100 * offset,
            murcko,
            butina_labels=butina,
        )
        interval = conservative_cluster_interval(comparison)
        paired_reports[representation] = {
            **comparison,
            "conservative_cluster_bootstrap_interval_95": interval,
            "direction": "positive_scores_retained_minus_clipped",
        }
        representation_rows.append(
            {
                "representation": representation,
                "clipped_mean_per_ligand_pairwise_accuracy": clipped_metric[
                    "mean_per_ligand_pairwise_accuracy"
                ],
                "unclipped_mean_per_ligand_pairwise_accuracy": unclipped_metric[
                    "mean_per_ligand_pairwise_accuracy"
                ],
                "unclipped_minus_clipped_accuracy": comparison[
                    "plugin_mean_difference"
                ],
                "conservative_cluster_bootstrap_95_low": interval[0],
                "conservative_cluster_bootstrap_95_high": interval[1],
                "clipped_predicted_score_ties": clipped_metric[
                    "predicted_score_ties_half_credit"
                ],
                "unclipped_predicted_score_ties": unclipped_metric[
                    "predicted_score_ties_half_credit"
                ],
                "evaluated_ligands": clipped_metric["evaluated_ligands"],
                "evaluated_pairs": clipped_metric["evaluated_pairs"],
            }
        )
        pair_order_rows.append(
            {
                "representation": representation,
                **informative_pair_order_change(
                    score_sets["positive_scores_clipped"][representation],
                    score_sets["positive_scores_retained"][representation],
                    experiment_values,
                ),
            }
        )

    released_path = benchmark.DEFAULT_OUTPUT / "summary.json"
    released_check: dict[str, Any] | None = None
    if released_path.exists():
        released = json.loads(released_path.read_text())
        differences = {
            representation: float(
                next(
                    row["clipped_mean_per_ligand_pairwise_accuracy"]
                    for row in representation_rows
                    if row["representation"] == representation
                )
                - released["mean_per_ligand_pairwise_accuracy"][representation]
            )
            for representation in DOCKING_REPRESENTATIONS
        }
        maximum = max(abs(value) for value in differences.values())
        if maximum > 1e-12:
            raise ValueError("clipped rerun does not reproduce the released ranking metrics")
        released_check = {
            "path": str(released_path.relative_to(PACKAGE)),
            "maximum_absolute_accuracy_difference": maximum,
            "differences": differences,
        }

    evaluated_unclipped = unclipped_reference.loc[
        experiment.index, experiment.columns
    ].to_numpy(dtype=np.float64)
    positive_by_ligand = (evaluated_unclipped > 0).sum(axis=1)
    summary = {
        "schema_version": "1.0.0",
        "analysis": "dockstring_positive_score_ligand_wise_ranking_sensitivity",
        "analysis_status": "complete_exploratory_preprocessing_sensitivity",
        "seed": int(seed),
        "bootstraps": int(bootstraps),
        "support": {
            "retained_ligands": int(len(experiment)),
            "targets": int(experiment.shape[1]),
            "observed_experimental_cells": int(experiment.notna().sum().sum()),
            "murcko_clusters": int(len(np.unique(murcko))),
            "butina_clusters": int(len(np.unique(butina))),
        },
        "reference": reference_report,
        "evaluation_positive_scores": {
            "positive_cells_among_evaluated_ligand_by_target_scores": int(
                (evaluated_unclipped > 0).sum()
            ),
            "positive_fraction_among_evaluated_ligand_by_target_scores": float(
                (evaluated_unclipped > 0).mean()
            ),
            "ligands_with_at_least_one_positive_evaluated_target_score": int(
                (positive_by_ligand > 0).sum()
            ),
            "fraction_ligands_with_at_least_one_positive_evaluated_target_score": float(
                (positive_by_ligand > 0).mean()
            ),
        },
        "activity_contract": activity_report,
        "support_contract": support_report,
        "fit_contracts": fit_contracts,
        "paired_accuracy_reports": paired_reports,
        "released_clipped_result_reproduction": released_check,
        "claim_boundary": (
            "This is a paired sensitivity to one released preprocessing choice on the "
            "fixed sparse observed-pair ChEMBL benchmark. It does not validate positive "
            "Vina scores as affinities, resolve unmeasured targets, or generalize to "
            "other docking engines or chemical supports."
        ),
    }
    return (
        summary,
        pd.DataFrame.from_records(representation_rows),
        pd.DataFrame.from_records(pair_order_rows),
    )


def readme(summary: dict[str, Any], representations: pd.DataFrame) -> str:
    positive = summary["evaluation_positive_scores"]
    lines = [
        "# Positive-score clipping sensitivity for ligand-wise ranking",
        "",
        "This paired preprocessing sensitivity retains the original positive",
        "DOCKSTRING scores while holding the ChEMBL cohort, target panel, external",
        "training support, target transformations, experimental ties and equal-ligand",
        "weighting fixed.",
        "",
        f"Only {positive['positive_cells_among_evaluated_ligand_by_target_scores']:,} "
        "evaluated ligand-by-target score cells were positive "
        f"({positive['positive_fraction_among_evaluated_ligand_by_target_scores']:.6%}).",
        "",
        "| Representation | Clipped accuracy | Retained-positive accuracy | Difference | Conservative cluster-bootstrap 95% interval |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in representations.itertuples(index=False):
        lines.append(
            f"| {row.representation} | "
            f"{row.clipped_mean_per_ligand_pairwise_accuracy:.4f} | "
            f"{row.unclipped_mean_per_ligand_pairwise_accuracy:.4f} | "
            f"{row.unclipped_minus_clipped_accuracy:+.4f} | "
            f"[{row.conservative_cluster_bootstrap_95_low:+.4f}, "
            f"{row.conservative_cluster_bootstrap_95_high:+.4f}] |"
        )
    lines.extend(
        [
            "",
            "The difference is retained-positive minus clipped. Intervals are the union",
            "of the Murcko- and Butina-cluster bootstrap intervals and remain conditional",
            "on the fixed 31-target panel.",
            "",
            "This result does not imply that positive Vina energies are calibrated or",
            "physically meaningful; it only tests whether clipping drives the reported",
            "observed-pair ranking result.",
            "",
            "## Reproduction",
            "",
            "```bash",
            ".venv/bin/python analysis/dockstring_positive_score_ranking_sensitivity.py \\",
            "  --bootstraps 2000 \\",
            "  --output-dir results/dockstring_positive_score_ranking_sensitivity",
            ".venv/bin/python -m pytest -q \\",
            "  analysis/test_dockstring_positive_score_ranking_sensitivity.py",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--activities", type=Path, default=benchmark.ACTIVITY_INPUT)
    parser.add_argument("--bootstraps", type=int, default=DEFAULT_BOOTSTRAPS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.bootstraps < 100:
        raise ValueError("use at least 100 bootstrap draws")
    summary, representations, pair_orders = evaluate(
        activities_path=args.activities,
        bootstraps=args.bootstraps,
        seed=args.seed,
    )
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "summary.json", summary)
    representations.to_csv(
        output / "representation_sensitivity.csv", index=False, float_format="%.17g"
    )
    pair_orders.to_csv(
        output / "pair_order_sensitivity.csv", index=False, float_format="%.17g"
    )
    (output / "README.md").write_text(readme(summary, representations))
    write_json(
        output / "output_checksums.json",
        {
            "algorithm": "sha256",
            "files": {
                filename: sha256_file(output / filename)
                for filename in (
                    "README.md",
                    "pair_order_sensitivity.csv",
                    "representation_sensitivity.csv",
                    "summary.json",
                )
            },
        },
    )
    print(representations.to_string(index=False))
    print(pair_orders.to_string(index=False))


if __name__ == "__main__":
    main()
