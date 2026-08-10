#!/usr/bin/env python3
"""Can a small ligand panel recover the leading residual target subspace?

This science-only analysis joins two previously separate exploratory results:

1. a small, outcome-blind random ligand panel estimates the residual target
   correlation geometry of a large DOCKSTRING surface; and
2. the leading residual modes may be enriched for experimentally reproducible
   target co-selectivity.

The mode count is selected from docking scores alone as the smallest number of
eigenvalues explaining at least 50% of residual-correlation trace.  Because a
rank-k spectral reconstruction is not a correlation matrix unless its diagonal
is renormalized, the analysis freezes both the raw spectral kernel and its
unit-diagonal version.  Experimental labels never enter mode selection.

The endpoint has already been inspected elsewhere in this project.  Its use
here is therefore post-hoc process validation, not independent biological
confirmation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from biological_core_modes import (
    docking_only_mode_count,
    subspace_stability,
    truncated_geometry,
)
from calibration_panel_recovery import (
    DEFAULT_DOCKSTRING,
    DEFAULT_ENDPOINT,
    experimental_labels,
    geometry,
    load_dockstring_matrix,
    upper,
)
from replicated_pair_retrieval import retrieval_metrics


PACKAGE = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PACKAGE / "results" / "calibration_core_integration"
DEFAULT_SIZES = (100, 200, 500)
DEFAULT_REPETITIONS = 200
DEFAULT_SEED = 20260813
DEFAULT_TRACE_THRESHOLD = 0.50


def unit_diagonal(kernel: np.ndarray) -> np.ndarray:
    """Convert a positive-semidefinite similarity kernel to unit diagonal."""
    kernel = np.asarray(kernel, dtype=np.float64)
    if kernel.ndim != 2 or kernel.shape[0] != kernel.shape[1]:
        raise ValueError("kernel must be square")
    if not np.isfinite(kernel).all() or not np.allclose(
        kernel, kernel.T, atol=1e-10
    ):
        raise ValueError("kernel must be finite and symmetric")
    diagonal = np.diag(kernel)
    if np.any(diagonal <= 1e-14):
        raise ValueError("kernel diagonal must be positive")
    scale = np.sqrt(diagonal)
    correlation = kernel / np.outer(scale, scale)
    correlation = np.clip((correlation + correlation.T) / 2.0, -1.0, 1.0)
    np.fill_diagonal(correlation, 1.0)
    return correlation


def endpoint_metrics(labels: np.ndarray, similarity: np.ndarray) -> dict[str, float]:
    values = upper(similarity)
    metrics = retrieval_metrics(np.asarray(labels, dtype=bool), values)
    return {
        "roc_auc": float(metrics["roc_auc"]),
        "average_precision": float(metrics["average_precision"]),
    }


def evaluate(
    matrix: np.ndarray,
    common_indices: np.ndarray,
    labels: np.ndarray,
    sizes: tuple[int, ...],
    repetitions: int,
    seed: int,
    trace_threshold: float,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Evaluate empirical and OAS pilot estimates against the full support."""
    matrix = np.asarray(matrix, dtype=np.float64)
    common = matrix[:, np.asarray(common_indices, dtype=int)]
    full = geometry(common, shrink=False)
    selection = docking_only_mode_count(full, trace_threshold)
    locked_k = int(selection["selected_k"])
    full_kernel = truncated_geometry(full, locked_k)
    full_unit = unit_diagonal(full_kernel)
    full_metrics = {
        "full_residual_correlation": endpoint_metrics(labels, full),
        "leading_spectral_kernel": endpoint_metrics(labels, full_kernel),
        "leading_unit_diagonal_correlation": endpoint_metrics(labels, full_unit),
    }

    rng = np.random.default_rng(seed)
    records: list[dict[str, object]] = []
    for size in sizes:
        if size < 3 or size >= len(common):
            raise ValueError(f"invalid calibration size {size}")
        for repetition in range(repetitions):
            selected_rows = rng.choice(len(common), size=size, replace=False)
            sample = common[selected_rows]
            for estimator, shrink in (("empirical", False), ("OAS", True)):
                sample_geometry = geometry(sample, shrink=shrink)
                sample_selection = docking_only_mode_count(
                    sample_geometry, trace_threshold
                )
                selected_k = int(sample_selection["selected_k"])
                selected_kernel = truncated_geometry(sample_geometry, selected_k)
                selected_unit = unit_diagonal(selected_kernel)
                locked_kernel = truncated_geometry(sample_geometry, locked_k)
                locked_unit = unit_diagonal(locked_kernel)
                stability = subspace_stability(full, sample_geometry, locked_k)
                sample_full_metrics = endpoint_metrics(labels, sample_geometry)
                selected_kernel_metrics = endpoint_metrics(labels, selected_kernel)
                selected_unit_metrics = endpoint_metrics(labels, selected_unit)
                locked_kernel_metrics = endpoint_metrics(labels, locked_kernel)
                locked_unit_metrics = endpoint_metrics(labels, locked_unit)
                records.append(
                    {
                        "calibration_ligands": int(size),
                        "repetition": int(repetition),
                        "covariance_estimator": estimator,
                        "selected_k": selected_k,
                        "selected_cumulative_trace_fraction": float(
                            sample_selection[
                                "selected_cumulative_variance_fraction"
                            ]
                        ),
                        "geometry_spearman_to_full": float(
                            spearmanr(upper(sample_geometry), upper(full)).statistic
                        ),
                        "locked_k": locked_k,
                        **stability,
                        "sample_full_roc_auc": sample_full_metrics["roc_auc"],
                        "sample_full_average_precision": sample_full_metrics[
                            "average_precision"
                        ],
                        "selected_kernel_roc_auc": selected_kernel_metrics[
                            "roc_auc"
                        ],
                        "selected_kernel_average_precision": (
                            selected_kernel_metrics["average_precision"]
                        ),
                        "selected_unit_diagonal_roc_auc": selected_unit_metrics[
                            "roc_auc"
                        ],
                        "selected_unit_diagonal_average_precision": (
                            selected_unit_metrics["average_precision"]
                        ),
                        "locked_kernel_roc_auc": locked_kernel_metrics["roc_auc"],
                        "locked_kernel_average_precision": locked_kernel_metrics[
                            "average_precision"
                        ],
                        "locked_unit_diagonal_roc_auc": locked_unit_metrics[
                            "roc_auc"
                        ],
                        "locked_unit_diagonal_average_precision": (
                            locked_unit_metrics["average_precision"]
                        ),
                    }
                )

    metadata: dict[str, object] = {
        "analysis_status": "science_only_post_hoc_process_validation",
        "sampling": "simple random ligands without replacement; outcome-blind",
        "full_ligands": int(len(common)),
        "targets": int(common.shape[1]),
        "trace_threshold": float(trace_threshold),
        "full_selected_k": locked_k,
        "full_selected_cumulative_trace_fraction": float(
            selection["selected_cumulative_variance_fraction"]
        ),
        "full_endpoint_metrics": full_metrics,
        "representations": {
            "full_residual_correlation": (
                "ordinary unit-diagonal residual target correlation"
            ),
            "leading_spectral_kernel": (
                "rank-k PSD reconstruction; diagonal contains retained communality "
                "and is not one"
            ),
            "leading_unit_diagonal_correlation": (
                "same rank-k reconstruction renormalized to unit diagonal"
            ),
        },
        "inference_boundary": (
            "The endpoint was inspected in prior project analyses. The numerical "
            "selection rule is outcome-blind, but endpoint performance is post hoc "
            "and cannot serve as independent confirmation."
        ),
        "seed": int(seed),
        "repetitions_per_size": int(repetitions),
    }
    return pd.DataFrame.from_records(records), metadata


def summarize(records: pd.DataFrame) -> pd.DataFrame:
    identifiers = {
        "calibration_ligands",
        "repetition",
        "covariance_estimator",
    }
    numeric = [column for column in records if column not in identifiers]
    rows: list[dict[str, object]] = []
    for (size, estimator), group in records.groupby(
        ["calibration_ligands", "covariance_estimator"], sort=True
    ):
        row: dict[str, object] = {
            "calibration_ligands": int(size),
            "covariance_estimator": str(estimator),
            "repetitions": int(len(group)),
        }
        counts = group.selected_k.value_counts().sort_index()
        row["selected_k_counts"] = ";".join(
            f"{int(k)}:{int(count)}" for k, count in counts.items()
        )
        row["fraction_selecting_full_k"] = float(
            np.mean(group.selected_k == group.locked_k)
        )
        for metric in numeric:
            values = group[metric].to_numpy(dtype=float)
            row[f"{metric}_mean"] = float(np.mean(values))
            row[f"{metric}_q025"] = float(np.quantile(values, 0.025))
            row[f"{metric}_q975"] = float(np.quantile(values, 0.975))
        rows.append(row)
    return pd.DataFrame.from_records(rows)


def write_readme(output: Path, metadata: dict[str, object], summary: pd.DataFrame) -> None:
    empirical = summary.loc[summary.covariance_estimator.eq("empirical")].set_index(
        "calibration_ligands"
    )
    row_500 = empirical.loc[500] if 500 in empirical.index else None
    if row_500 is None:
        headline = "The requested run did not include 500 calibration ligands."
    else:
        headline = (
            f"With 500 calibration ligands, the full k={metadata['full_selected_k']} "
            "rule was recovered in "
            f"{100 * row_500['fraction_selecting_full_k']:.1f}% of empirical "
            "replicates and mean fixed-k subspace overlap was "
            f"{row_500['mean_squared_canonical_correlation_mean']:.3f}."
        )
    full = metadata["full_endpoint_metrics"]
    text = f"""# Calibration-panel recovery of the leading residual target subspace

This science-only artifact asks whether a small, outcome-blind random ligand
panel can recover the leading residual target modes of the complete
DOCKSTRING 20-kinase surface.

## Main process result

{headline}

The full surface selected k={metadata['full_selected_k']} at a fixed
{metadata['trace_threshold']:.0%}-trace rule.  On the previously frozen
experimental target-pair endpoint, full-support AUROC/AP were
{full['full_residual_correlation']['roc_auc']:.3f}/
{full['full_residual_correlation']['average_precision']:.3f} for the complete
residual correlation, {full['leading_spectral_kernel']['roc_auc']:.3f}/
{full['leading_spectral_kernel']['average_precision']:.3f} for the raw leading
spectral kernel, and
{full['leading_unit_diagonal_correlation']['roc_auc']:.3f}/
{full['leading_unit_diagonal_correlation']['average_precision']:.3f} after
unit-diagonal renormalization.

The raw rank-k reconstruction is a **spectral similarity kernel**, not a
correlation matrix: its diagonal is the communality retained in the leading
modes.  Both it and its unit-diagonal version are reported because pair
retrieval can depend on that weighting.

## Interpretation boundary

This is an internal process/recovery experiment.  It shows whether a small
panel can recover a full-docking target subspace under the same source-library
distribution.  The experimental endpoint was already inspected elsewhere in
the project, so its reuse is post hoc and is not independent biological
confirmation.  The analysis says nothing about individual ligand profiles,
pose quality, or a chemically shifted deployment library.

## Reproduction

```bash
.venv/bin/python analysis/calibration_core_integration.py \\
  --sizes 100,200,500 \\
  --repetitions {metadata['repetitions_per_size']} \\
  --output-dir results/calibration_core_integration

.venv/bin/python -m pytest -q \\
  analysis/test_calibration_core_integration.py
```
"""
    (output / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dockstring", type=Path, default=DEFAULT_DOCKSTRING)
    parser.add_argument("--endpoint", type=Path, default=DEFAULT_ENDPOINT)
    parser.add_argument("--sizes", default=",".join(map(str, DEFAULT_SIZES)))
    parser.add_argument("--repetitions", type=int, default=DEFAULT_REPETITIONS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--trace-threshold", type=float, default=DEFAULT_TRACE_THRESHOLD
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    sizes = tuple(int(value) for value in args.sizes.split(",") if value)
    matrix, targets = load_dockstring_matrix(args.dockstring)
    common_indices, labels = experimental_labels(targets, args.endpoint)
    records, metadata = evaluate(
        matrix,
        common_indices,
        labels,
        sizes,
        args.repetitions,
        args.seed,
        args.trace_threshold,
    )
    summary = summarize(records)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records.to_csv(
        args.output_dir / "replicate_metrics.csv", index=False, float_format="%.15g"
    )
    summary.to_csv(
        args.output_dir / "summary.csv", index=False, float_format="%.15g"
    )
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_readme(args.output_dir, metadata, summary)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
