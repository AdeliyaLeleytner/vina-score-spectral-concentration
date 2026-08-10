#!/usr/bin/env python3
"""Strict fixed-common-20 sensitivity to the ligand row-centering panel.

All four experimental panels are restricted to the same ordered 20 kinase
targets.  The docking reference, target columns, and experimental endpoint are
held fixed while the ligand row mean is estimated either from those 20 kinase
columns or from all 58 released DOCKSTRING columns.  This isolates sensitivity
to the centering panel without mixing the earlier 21-target DAVIS/PKIS2 and
20-target PKIS1/KiRHub supports.
"""

from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd

try:  # Support direct CLI execution and package-style imports.
    from . import centering_panel_sensitivity as centering
    from . import kirhub_external_validation as kirhub
    from . import replicated_pair_retrieval as retrieval
    from . import residual_target_geometry_validation as geometry
except ImportError:  # pragma: no cover - direct CLI execution.
    import centering_panel_sensitivity as centering  # type: ignore
    import kirhub_external_validation as kirhub  # type: ignore
    import replicated_pair_retrieval as retrieval  # type: ignore
    import residual_target_geometry_validation as geometry  # type: ignore


PACKAGE = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PACKAGE / "results" / "fixed20_centering_panel_sensitivity"
DEFAULT_PKIS1_ZIP = centering.DEFAULT_PKIS1_ZIP
DEFAULT_KIRHUB_WORKBOOK = centering.DEFAULT_KIRHUB_WORKBOOK
DEFAULT_SEED = 20260806
TARGETS20 = tuple(kirhub.TARGETS)

if TARGETS20 != tuple(centering.TARGETS_20):
    raise RuntimeError("the two locked definitions of the common 20 targets diverged")


def _validate_experiments(
    experiments: dict[str, np.ndarray],
) -> OrderedDict[str, np.ndarray]:
    expected = ("DAVIS", "PKIS2", "PKIS1", "KiRHub")
    if tuple(experiments) != expected:
        raise ValueError(f"experimental panels must be ordered as {expected}")
    validated: OrderedDict[str, np.ndarray] = OrderedDict()
    for panel, matrix in experiments.items():
        values = np.asarray(matrix, dtype=np.float64)
        if values.ndim != 2 or values.shape[0] < 3 or values.shape[1] != 20:
            raise ValueError(f"{panel} is not a dense ligand-by-20-target matrix")
        if not np.isfinite(values).all():
            raise ValueError(f"{panel} contains non-finite experimental values")
        validated[panel] = values
    return validated


def evaluate_fixed20(
    reference: np.ndarray,
    score_columns: tuple[str, ...],
    experiments: dict[str, np.ndarray],
    *,
    targets: tuple[str, ...] = TARGETS20,
    permutations: int,
    seed: int,
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    """Compare local-20 and full-58 row centering on one fixed target support."""
    values = np.asarray(reference, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != len(score_columns):
        raise ValueError("reference matrix and released score-column order do not align")
    if len(score_columns) < len(targets) or len(set(score_columns)) != len(score_columns):
        raise ValueError("score columns must be unique and include non-target context")
    if len(targets) != 20 or len(set(targets)) != 20:
        raise ValueError("this analysis requires exactly 20 unique targets")
    if not np.isfinite(values).all():
        raise ValueError("reference matrix contains non-finite scores")
    if permutations < 1:
        raise ValueError("QAP permutations must be positive")
    missing = [target for target in targets if target not in score_columns]
    if missing:
        raise ValueError(f"reference score columns are missing targets: {missing}")
    panels = _validate_experiments(experiments)

    score_index = {target: position for position, target in enumerate(score_columns)}
    target_indices = tuple(score_index[target] for target in targets)
    local_geometry, local_identity_error = centering.centered_geometry(
        values, target_indices, center_over_all_columns=False
    )
    full_geometry, full_identity_error = centering.centered_geometry(
        values, target_indices, center_over_all_columns=True
    )

    agreement = {
        "targets": len(targets),
        "target_pairs": len(targets) * (len(targets) - 1) // 2,
        "local_20_correlation_PR": retrieval.correlation_pr(local_geometry),
        "all_columns_then_extract_correlation_PR": retrieval.correlation_pr(
            full_geometry
        ),
        "spearman_of_target_pair_correlations": geometry.geometry_concordance(
            local_geometry, full_geometry
        ),
        "mean_absolute_correlation_difference": float(
            np.mean(np.abs(local_geometry - full_geometry))
        ),
        "maximum_absolute_correlation_difference": float(
            np.max(np.abs(local_geometry - full_geometry))
        ),
        "row_only_vs_two_way_max_abs_difference_local_20": local_identity_error,
        "row_only_vs_two_way_max_abs_difference_all_columns": full_identity_error,
    }

    rows: list[dict[str, object]] = []
    panel_results: OrderedDict[str, dict[str, object]] = OrderedDict()
    experimental_geometries: OrderedDict[str, np.ndarray] = OrderedDict()
    for panel_index, (panel, experiment) in enumerate(panels.items()):
        experimental_geometry = geometry.geometry_correlation(
            experiment, "center_then_correlation"
        )
        experimental_geometries[panel] = experimental_geometry
        local_qap = geometry.qap_test(
            local_geometry,
            experimental_geometry,
            permutations,
            seed + panel_index * 1_000_000,
        )
        full_qap = geometry.qap_test(
            full_geometry,
            experimental_geometry,
            permutations,
            seed + panel_index * 1_000_000 + 250_000,
        )
        paired_qap = geometry.fixed_experimental_geometry_paired_qap(
            local_geometry,
            full_geometry,
            experimental_geometry,
            permutations,
            seed + panel_index * 1_000_000 + 500_000,
        )
        record = {
            "experimental_ligands": int(len(experiment)),
            "targets": len(targets),
            "target_pairs": len(targets) * (len(targets) - 1) // 2,
            "experimental_centered_correlation_PR": retrieval.correlation_pr(
                experimental_geometry
            ),
            "local_20_centered_concordance_spearman": local_qap[
                "observed_spearman"
            ],
            "local_20_centered_qap_p_positive": local_qap[
                "one_sided_p_positive"
            ],
            "all_columns_centered_then_extract_concordance_spearman": full_qap[
                "observed_spearman"
            ],
            "all_columns_centered_then_extract_qap_p_positive": full_qap[
                "one_sided_p_positive"
            ],
            "all_columns_minus_local_20_concordance": paired_qap[
                "centered_minus_raw_docking"
            ],
            "paired_qap_p_positive_delta": paired_qap[
                "one_sided_p_positive_delta"
            ],
            "paired_delta_null_interval_95": paired_qap["null_interval_95"],
            "paired_delta_null_median": paired_qap["null_median"],
            "seeds": {
                "local_qap": local_qap["seed"],
                "all_columns_qap": full_qap["seed"],
                "paired_qap": paired_qap["seed"],
            },
        }
        panel_results[panel] = record
        rows.append(
            {
                "panel": panel,
                "experimental_ligands": record["experimental_ligands"],
                "targets": record["targets"],
                "target_pairs": record["target_pairs"],
                "experimental_centered_correlation_PR": record[
                    "experimental_centered_correlation_PR"
                ],
                "local_20_centered_concordance_spearman": record[
                    "local_20_centered_concordance_spearman"
                ],
                "all_58_centered_then_extract_concordance_spearman": record[
                    "all_columns_centered_then_extract_concordance_spearman"
                ],
                "all_58_minus_local_20_concordance": record[
                    "all_columns_minus_local_20_concordance"
                ],
                "paired_qap_p_positive_delta": record[
                    "paired_qap_p_positive_delta"
                ],
            }
        )

    summary = {
        "analysis": "strict fixed-common-20 row-centering-panel sensitivity",
        "status": "exploratory_post_hoc_sensitivity",
        "question": (
            "With the target identities and centered experimental endpoint fixed, "
            "does docking target-pair geometry depend on estimating each ligand's "
            "row mean from the 20 validation kinases or all released score columns?"
        ),
        "targets": list(targets),
        "target_pairs": len(targets) * (len(targets) - 1) // 2,
        "centering_contracts": {
            "local_20": (
                "extract the same ordered 20 targets for every panel, then remove "
                "each reference ligand's mean over those 20 scores"
            ),
            "all_columns_then_extract": (
                "remove each reference ligand's mean over all released score "
                "columns, then extract the same ordered 20 targets"
            ),
        },
        "docking_geometry_agreement": agreement,
        "experimental_panel_results": panel_results,
        "configuration": {
            "qap_permutations": int(permutations),
            "seed": int(seed),
            "permutation_unit": "complete target labels",
            "fixed_endpoint": "two-way-centered experimental target correlation",
        },
        "claim_boundary": (
            "This post hoc sensitivity changes only the ligand row-centering panel. "
            "It does not estimate transfer to new target identities, and QAP "
            "inference is conditional on this fixed 20-target panel."
        ),
    }
    pair_rows: list[dict[str, object]] = []
    for first_index, first_target in enumerate(targets):
        for second_index in range(first_index + 1, len(targets)):
            second_target = targets[second_index]
            pair_rows.append(
                {
                    "target_a": first_target,
                    "target_b": second_target,
                    "docking_local_20_centered_correlation": local_geometry[
                        first_index, second_index
                    ],
                    "docking_all_58_centered_then_extract_correlation": full_geometry[
                        first_index, second_index
                    ],
                    **{
                        f"{panel}_experimental_centered_correlation": matrix[
                            first_index, second_index
                        ]
                        for panel, matrix in experimental_geometries.items()
                    },
                }
            )
    return (
        summary,
        pd.DataFrame.from_records(rows),
        pd.DataFrame.from_records(pair_rows),
    )


def run(
    *,
    pkis1_zip: Path,
    kirhub_workbook: Path,
    permutations: int,
    seed: int,
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    reference, score_columns, support, payload = centering.load_reference(pkis1_zip)
    sources = payload["_sources"].copy()
    experiments: OrderedDict[str, np.ndarray] = OrderedDict(
        kirhub.load_old_experimental_panels(pkis1_zip)
    )
    kirhub_frame = kirhub.load_kirhub(kirhub_workbook)
    experiments["KiRHub"] = kirhub_frame[list(TARGETS20)].to_numpy(
        dtype=np.float64
    )
    sources["kirhub_sha256"] = geometry.sha256_file(kirhub_workbook)
    summary, table, pair_table = evaluate_fixed20(
        reference,
        score_columns,
        experiments,
        permutations=permutations,
        seed=seed,
    )
    summary["support"] = support
    summary["sources"] = sources
    return summary, table, pair_table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pkis1-zip", type=Path, default=DEFAULT_PKIS1_ZIP)
    parser.add_argument(
        "--kirhub-workbook", type=Path, default=DEFAULT_KIRHUB_WORKBOOK
    )
    parser.add_argument("--permutations", type=int, default=49_999)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    summary, table, pair_table = run(
        pkis1_zip=args.pkis1_zip,
        kirhub_workbook=args.kirhub_workbook,
        permutations=args.permutations,
        seed=args.seed,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    table.to_csv(args.output / "panel_concordance.csv", index=False)
    pair_table.to_csv(args.output / "target_pair_geometry.csv", index=False)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
