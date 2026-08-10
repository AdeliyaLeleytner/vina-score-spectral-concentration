#!/usr/bin/env python3
"""Outcome-blind sentinel-target selection for dense experimental kinase panels.

Sentinel kinases are selected without using an experimental endpoint: by DEIM
sensor placement on leading raw or residual DOCKSTRING correlation modes, or by
greedy receptor-sequence diversity.  DEIM applies pivoted QR to the leading
eigenvector matrix, whose target leverage scores break the arbitrary equal-norm
first pivot encountered when QR is applied directly to standardized columns.
For each dense PKIS2, PKIS1, and DAVIS panel, scaffold-held-out regression then
asks whether experimental measurements on only those sentinels, plus seven
ligand descriptors, reconstruct the remaining experimental target profile.

Fixed random target panels provide the primary target-selection null.  Target
standardization, descriptor scaling, and regression fitting are fold-local.
This is a science-only exploratory artifact and does not edit manuscript files.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.linalg import qr
from sklearn.model_selection import GroupKFold

import biological_core_modes as core
import dense_davis_benchmark as davis
from experimental_chemical_context_geometry import TARGETS, load_experimental_panels
from klifs_pocket_control import matrix_from_pairs
from residual_mechanism_analysis import molecular_descriptors
from sentinel_target_compression import (
    PRIMARY_METRICS,
    _evaluate_feature_set,
    _standardize_train_test,
)


PACKAGE = Path(__file__).resolve().parents[1]
DEFAULT_PAIRS = PACKAGE / "results" / "klifs_pocket_control" / "target_pairs.csv"
DEFAULT_PKIS1 = Path("/tmp/pkis1_supplement.zip")
DEFAULT_KIRHUB = Path("/private/tmp/kirhub_supp_tables.xlsx")
DEFAULT_OUTPUT = PACKAGE / "results" / "experimental_sentinel_panel"
DEFAULT_K = (3, 5, 10)
DEFAULT_RANDOM_REPEATS = 2_000
DEFAULT_SEED = 202_608_19
DEFAULT_SUPPORT_REPEATS = 25


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float) and not np.isfinite(value):
        raise ValueError("non-finite value cannot be written to strict JSON")
    return value


def sequence_diverse_order(identity: np.ndarray) -> np.ndarray:
    """Deterministic farthest-point traversal of receptor-sequence distance."""
    identity = np.asarray(identity, dtype=np.float64)
    if identity.ndim != 2 or identity.shape[0] != identity.shape[1]:
        raise ValueError("sequence identity must be square")
    if not np.isfinite(identity).all() or np.any(identity < 0) or np.any(identity > 1):
        raise ValueError("sequence identity must be finite and lie in [0, 1]")
    distance = 1.0 - identity
    np.fill_diagonal(distance, 0.0)
    off_diagonal_mean = distance.sum(axis=1) / (len(distance) - 1)
    selected = [int(np.argmax(off_diagonal_mean))]
    available = set(range(len(distance))) - set(selected)
    while available:
        candidates = np.asarray(sorted(available), dtype=int)
        minimum_distance = np.min(distance[np.ix_(candidates, selected)], axis=1)
        best = int(candidates[int(np.argmax(minimum_distance))])
        selected.append(best)
        available.remove(best)
    return np.asarray(selected, dtype=int)


def deim_selection(correlation: np.ndarray, k: int) -> np.ndarray:
    """Select k target sensors from the leading k correlation eigenvectors."""
    correlation = np.asarray(correlation, dtype=np.float64)
    if correlation.ndim != 2 or correlation.shape[0] != correlation.shape[1]:
        raise ValueError("DEIM requires a square correlation matrix")
    if not 1 <= k < len(correlation):
        raise ValueError("DEIM k must leave at least one target unselected")
    eigenvalues, eigenvectors = np.linalg.eigh(
        (correlation + correlation.T) / 2.0
    )
    leading = eigenvectors[:, np.argsort(eigenvalues)[::-1][:k]]
    _, _, pivots = qr(
        leading.T, mode="economic", pivoting=True, check_finite=False
    )
    return np.asarray(pivots[:k], dtype=int)


def _jaccard(first: np.ndarray, second: np.ndarray) -> float:
    a, b = set(map(int, first)), set(map(int, second))
    return float(len(a & b) / len(a | b))


def select_sentinel_panels(
    pair_path: Path,
    pkis1_zip: Path,
    kirhub_workbook: Path,
    k_values: tuple[int, ...],
    support_repeats: int,
    support_sample_size: int,
    seed: int,
) -> tuple[
    dict[tuple[str, int], np.ndarray], pd.DataFrame, dict[str, Any]
]:
    reference, conservative, reference_groups, _, provenance = core.load_inputs(
        pkis1_zip, kirhub_workbook
    )
    raw_correlation = np.corrcoef(reference, rowvar=False)
    residual_correlation = core.residual_correlation(reference)
    conservative_residual = core.residual_correlation(conservative)
    pairs = pd.read_csv(pair_path)
    sequence = matrix_from_pairs(pairs, "receptor_domain_sequence_identity")
    sequence_order = sequence_diverse_order(sequence)
    panels: dict[tuple[str, int], np.ndarray] = {}
    stability_rows: list[dict[str, object]] = []
    scaffold_folds = core.scaffold_fold_ids(reference_groups, 5, seed + 1)
    rng = np.random.default_rng(seed + 2)
    for k in k_values:
        panels[("raw_vina_deim", k)] = deim_selection(raw_correlation, k)
        panels[("residual_vina_deim", k)] = deim_selection(
            residual_correlation, k
        )
        panels[("sequence_diverse", k)] = sequence_order[:k]
        panels[("conservative_residual_vina_deim", k)] = deim_selection(
            conservative_residual, k
        )
        primary = panels[("residual_vina_deim", k)]
        conservative_panel = panels[("conservative_residual_vina_deim", k)]
        stability_rows.append(
            {
                "k": k,
                "support_type": "exact_vs_exact_plus_same_cyclic_murcko_exclusion",
                "support_id": "conservative",
                "jaccard_to_primary": _jaccard(primary, conservative_panel),
                "selected_targets": ";".join(
                    TARGETS[index] for index in conservative_panel
                ),
            }
        )
        for fold in range(5):
            selected = deim_selection(
                core.residual_correlation(reference[scaffold_folds != fold]), k
            )
            stability_rows.append(
                {
                    "k": k,
                    "support_type": "leave_one_scaffold_fold_out",
                    "support_id": str(fold),
                    "jaccard_to_primary": _jaccard(primary, selected),
                    "selected_targets": ";".join(TARGETS[index] for index in selected),
                }
            )
        for repetition in range(support_repeats):
            support = np.sort(
                rng.choice(len(reference), size=support_sample_size, replace=False)
            )
            selected = deim_selection(
                core.residual_correlation(reference[support]), k
            )
            stability_rows.append(
                {
                    "k": k,
                    "support_type": "random_ligand_support",
                    "support_id": str(repetition),
                    "jaccard_to_primary": _jaccard(primary, selected),
                    "selected_targets": ";".join(TARGETS[index] for index in selected),
                }
            )
    metadata = {
        "dockstring_support": provenance,
        "fixed_targets": list(TARGETS),
        "target_selection_uses_experimental_outcomes": False,
        "primary_target_selection": (
            "DEIM/pivoted-QR sensor placement on the leading k eigenvectors of "
            "the residual target correlation from all exact-connectivity-excluded "
            "complete DOCKSTRING ligands"
        ),
        "raw_vina_selection": (
            "representation-matched DEIM on the raw Vina target correlation"
        ),
        "sequence_selection": (
            "deterministic greedy farthest-point traversal of one minus "
            "receptor-domain sequence identity"
        ),
        "support_sensitivity": (
            f"{support_repeats} independent random supports of {support_sample_size:,} "
            "ligands, five leave-one-scaffold-fold-out supports, and conservative "
            "same-cyclic-Murcko exclusion"
        ),
    }
    return panels, pd.DataFrame.from_records(stability_rows), metadata


def evaluate_panel(
    panel_name: str,
    matrix: np.ndarray,
    smiles: pd.Series,
    sentinel_panels: dict[tuple[str, int], np.ndarray],
    random_panels: dict[tuple[int, int], np.ndarray],
    k_values: tuple[int, ...],
    folds: int,
) -> pd.DataFrame:
    matrix = np.asarray(matrix, dtype=np.float64)
    descriptors = molecular_descriptors(smiles.astype(str))
    groups = davis._murcko_labels(smiles.reset_index(drop=True))
    splitter = GroupKFold(n_splits=folds)
    rows: list[dict[str, object]] = []
    for fold, (train_index, test_index) in enumerate(
        splitter.split(matrix, groups=groups), start=1
    ):
        train, test = _standardize_train_test(
            matrix[train_index], matrix[test_index]
        )
        for k in k_values:
            panels = {
                method: targets
                for (method, panel_k), targets in sentinel_panels.items()
                if panel_k == k
            }
            panels.update(
                {
                    f"random_{repeat}": sentinels
                    for (random_k, repeat), sentinels in random_panels.items()
                    if random_k == k
                }
            )
            for method, sentinels in panels.items():
                train_features = np.column_stack(
                    [train[:, sentinels], descriptors[train_index]]
                )
                test_features = np.column_stack(
                    [test[:, sentinels], descriptors[test_index]]
                )
                # A fixed modest ridge is used because DAVIS has only 72 ligands.
                # _evaluate_feature_set calls the shared fixed-ridge implementation;
                # feature and outcome target scaling are both fold-local.
                metrics = _evaluate_feature_set(
                    train,
                    test,
                    train_features,
                    test_features,
                    sentinels,
                    alpha=1.0,
                )
                is_random = method.startswith("random_")
                common = {
                        "panel": panel_name,
                        "fold": fold,
                        "k": k,
                        "random_repeat": int(method.split("_")[1]) if is_random else -1,
                        "train_ligands": int(len(train_index)),
                        "test_ligands": int(len(test_index)),
                        "train_murcko_groups": int(len(set(groups[train_index]))),
                        "test_murcko_groups": int(len(set(groups[test_index]))),
                }
                rows.append(
                    {
                        **common,
                        "method": "random" if is_random else method,
                        **metrics,
                    }
                )
                if method == "residual_vina_deim":
                    descriptor_metrics = _evaluate_feature_set(
                        train,
                        test,
                        descriptors[train_index],
                        descriptors[test_index],
                        sentinels,
                        alpha=1.0,
                    )
                    rows.append(
                        {
                            **common,
                            "method": "residual_panel_descriptors_only",
                            **descriptor_metrics,
                        }
                    )
    return pd.DataFrame.from_records(rows)


def summarize_methods(records: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        column
        for column in records.columns
        if column.endswith(("_r2", "_rmse", "_spearman", "_overlap", "_accuracy"))
    ]
    deterministic = records[records.random_repeat < 0]
    deterministic_summary = (
        deterministic.groupby(["panel", "k", "method"], as_index=False)[metrics]
        .mean()
    )
    random = (
        records[records.random_repeat >= 0]
        .groupby(["panel", "k", "method", "random_repeat"], as_index=False)[metrics]
        .mean()
        .groupby(["panel", "k", "method"], as_index=False)[metrics]
        .mean()
    )
    return pd.concat([deterministic_summary, random], ignore_index=True).sort_values(
        ["panel", "k", "method"]
    )


def random_comparisons(records: pd.DataFrame) -> pd.DataFrame:
    deterministic_methods = (
        "raw_vina_deim",
        "residual_vina_deim",
        "sequence_diverse",
        "conservative_residual_vina_deim",
    )
    rows: list[dict[str, object]] = []
    for panel_scope in [*sorted(records.panel.unique()), "three_panel_mean"]:
        scoped = records if panel_scope == "three_panel_mean" else records[records.panel == panel_scope]
        for k in sorted(scoped.k.unique()):
            selected = scoped[scoped.k == k]
            random = (
                selected[selected.method == "random"]
                .groupby(["panel", "random_repeat"], as_index=False)[list(PRIMARY_METRICS)]
                .mean()
                .groupby("random_repeat")[list(PRIMARY_METRICS)]
                .mean()
            )
            for method in deterministic_methods:
                deterministic = (
                    selected[selected.method == method]
                    .groupby("panel")[list(PRIMARY_METRICS)]
                    .mean()
                    .mean(axis=0)
                )
                for metric in PRIMARY_METRICS:
                    random_values = random[metric].to_numpy(dtype=np.float64)
                    observed = float(deterministic[metric])
                    deltas = observed - random_values
                    rows.append(
                        {
                            "panel_scope": panel_scope,
                            "k": int(k),
                            "method": method,
                            "metric": metric,
                            "observed": observed,
                            "random_mean": float(random_values.mean()),
                            "observed_minus_random_mean": float(deltas.mean()),
                            "delta_2.5_percentile": float(np.quantile(deltas, 0.025)),
                            "delta_97.5_percentile": float(np.quantile(deltas, 0.975)),
                            "one_sided_target_panel_p": float(
                                (1 + np.sum(deltas <= 0)) / (1 + len(deltas))
                            ),
                            "random_panels": int(len(deltas)),
                        }
                    )
    return pd.DataFrame.from_records(rows)


def paired_method_contrasts(
    records: pd.DataFrame, k: int = 10
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Scaffold-fold-paired contrasts for residual DEIM versus fixed controls."""
    selected = records[(records.random_repeat < 0) & (records.k == k)]
    detail: list[dict[str, object]] = []
    comparisons = {
        "residual_minus_raw": ("residual_vina_deim", "raw_vina_deim"),
        "residual_minus_sequence": ("residual_vina_deim", "sequence_diverse"),
        "sentinel_measurements_plus_descriptors_minus_descriptors_only": (
            "residual_vina_deim",
            "residual_panel_descriptors_only",
        ),
    }
    for panel in sorted(selected.panel.unique()):
        panel_frame = selected[selected.panel == panel]
        for label, (first, second) in comparisons.items():
            left = panel_frame[panel_frame.method == first].set_index("fold")
            right = panel_frame[panel_frame.method == second].set_index("fold")
            if not left.index.equals(right.index):
                raise ValueError("deterministic methods do not share scaffold folds")
            for metric in PRIMARY_METRICS:
                differences = left[metric] - right[metric]
                for fold, value in differences.items():
                    detail.append(
                        {
                            "panel": panel,
                            "k": k,
                            "contrast": label,
                            "metric": metric,
                            "fold": int(fold),
                            "difference": float(value),
                        }
                    )
    detail_frame = pd.DataFrame.from_records(detail)
    summary: list[dict[str, object]] = []
    for keys, group in detail_frame.groupby(
        ["panel", "k", "contrast", "metric"], sort=True
    ):
        differences = group.difference.to_numpy(dtype=np.float64)
        observed = float(differences.mean())
        signs = np.asarray(
            np.meshgrid(*([[-1.0, 1.0]] * len(differences)))
        ).T.reshape(-1, len(differences))
        null = np.mean(signs * differences[None, :], axis=1)
        summary.append(
            {
                **dict(
                    zip(
                        ["panel", "k", "contrast", "metric"], keys, strict=True
                    )
                ),
                "mean_fold_paired_difference": observed,
                "minimum_fold_difference": float(differences.min()),
                "maximum_fold_difference": float(differences.max()),
                "positive_folds": int(np.sum(differences > 0)),
                "folds": int(len(differences)),
                "exact_one_sided_sign_flip_p": float(np.mean(null >= observed)),
            }
        )
    return detail_frame, pd.DataFrame.from_records(summary)


def selected_target_frame(
    panels: dict[tuple[str, int], np.ndarray]
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (method, k), targets in sorted(panels.items()):
        for rank, index in enumerate(targets, start=1):
            rows.append(
                {
                    "method": method,
                    "k": k,
                    "selection_rank": rank,
                    "target": TARGETS[int(index)],
                }
            )
    return pd.DataFrame.from_records(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair-path", type=Path, default=DEFAULT_PAIRS)
    parser.add_argument("--pkis1-zip", type=Path, default=DEFAULT_PKIS1)
    parser.add_argument("--kirhub-workbook", type=Path, default=DEFAULT_KIRHUB)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--k", default=",".join(map(str, DEFAULT_K)))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--random-repeats", type=int, default=DEFAULT_RANDOM_REPEATS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--support-repeats", type=int, default=DEFAULT_SUPPORT_REPEATS)
    parser.add_argument("--support-sample-size", type=int, default=15_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    k_values = tuple(int(value) for value in args.k.split(",") if value)
    if not k_values or min(k_values) < 1 or max(k_values) >= len(TARGETS):
        raise ValueError("k must be nonempty and leave at least one target held out")
    sentinel_panels, support_stability, selection_metadata = select_sentinel_panels(
        args.pair_path,
        args.pkis1_zip,
        args.kirhub_workbook,
        k_values,
        args.support_repeats,
        args.support_sample_size,
        args.seed,
    )
    rng = np.random.default_rng(args.seed)
    random_panels = {
        (k, repeat): np.sort(rng.choice(len(TARGETS), size=k, replace=False))
        for k in k_values
        for repeat in range(args.random_repeats)
    }
    panels = load_experimental_panels(args.pkis1_zip)
    records = pd.concat(
        [
            evaluate_panel(
                panel,
                np.asarray(values["matrix"], dtype=np.float64),
                pd.Series(values["smiles"]),
                sentinel_panels,
                random_panels,
                k_values,
                args.folds,
            )
            for panel, values in panels.items()
        ],
        ignore_index=True,
    )
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    records.to_csv(output / "panel_fold_metrics.csv", index=False)
    summary = summarize_methods(records)
    summary.to_csv(output / "method_summary.csv", index=False)
    comparisons = random_comparisons(records)
    comparisons.to_csv(output / "random_target_panel_comparisons.csv", index=False)
    selected_target_frame(sentinel_panels).to_csv(
        output / "selected_targets.csv", index=False
    )
    support_stability.to_csv(output / "selection_support_stability.csv", index=False)
    paired_detail, paired_summary = paired_method_contrasts(records, k=10)
    paired_detail.to_csv(output / "paired_method_contrasts.csv", index=False)
    paired_summary.to_csv(output / "paired_method_contrast_summary.csv", index=False)
    metadata = {
        "analysis_status": "post_hoc_exploratory_science_only",
        "experimental_panels": {
            panel: {
                "ligands": int(len(values["matrix"])),
                "targets": int(np.asarray(values["matrix"]).shape[1]),
            }
            for panel, values in panels.items()
        },
        "k_values": list(k_values),
        "random_target_panels": args.random_repeats,
        "random_seed": args.seed,
        "support_repeats": args.support_repeats,
        "support_sample_size": args.support_sample_size,
        "chemical_validation": (
            "five-fold GroupKFold by RDKit Bemis-Murcko scaffold; the existing "
            "helper assigns acyclic molecules to singleton groups"
        ),
        "model": (
            "fold-local target z-scoring; seven physicochemical descriptors plus "
            "observed sentinel experimental values; fixed ridge alpha=1.0 after "
            "fold-local predictor standardization"
        ),
        "claim_boundary": (
            "reconstruction of held-out targets within three existing dense kinase "
            "assay panels; not prospective assay-budget validation and not retrieval "
            "of targets outside the fixed 20-kinase panel"
        ),
        "metric_boundary": (
            "full-target R2 is the primary fair panel-compression metric because every "
            "k-target design is evaluated on the same 20 targets, with measured "
            "sentinels inserted exactly. Held-out-target R2 changes its target support "
            "when the sentinel set changes and is therefore reported only as a "
            "target-set-dependent diagnostic. Rank and top-5 metrics test a distinct "
            "within-ligand objective and need not improve with squared-error recovery."
        ),
        **selection_metadata,
    }
    (output / "summary.json").write_text(
        json.dumps(json_ready(metadata), indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(
        comparisons[
            (comparisons.panel_scope == "three_panel_mean")
            & comparisons.metric.isin(
                ["full_target_r2", "full_mean_row_spearman", "full_top5_overlap"]
            )
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
