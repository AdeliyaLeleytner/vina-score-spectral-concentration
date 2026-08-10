#!/usr/bin/env python3
"""Sensitivity of DOCKSTRING target geometry to the row-centering panel.

The target-geometry analyses originally remove each ligand's mean over the
kinase targets used by the corresponding experimental validation panel (21 for
DAVIS and PKIS2; 20 for PKIS1, KiRHub, and the locked replicated-pair endpoint).
This focused sensitivity instead removes the ligand mean over all 58 released
DOCKSTRING targets and only then extracts the kinase columns.

The analysis preserves the primary DOCKSTRING contracts: the complete
260,060-row support, clipping positive scores to zero, exact Standard-InChI
connectivity exclusion for DAVIS/PKIS2/PKIS1 ligands, and the released target
order.  KiRHub compounds cannot be excluded because its supplement provides
names but no structures.  The 15-pair endpoint is loaded unchanged from the
frozen replicated-pair artifact.

Pearson target correlations after row-only and two-way centering must be
identical: two-way centering differs from row-only centering only by a constant
translation of each target column.  The implementation verifies that identity
numerically for every reported docking geometry.
"""

from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd

try:  # Support direct CLI execution and package-style imports.
    from . import dense_davis_benchmark as davis
    from . import kirhub_external_validation as kirhub
    from . import replicated_pair_retrieval as retrieval
    from . import residual_target_geometry_validation as geometry
except ImportError:  # pragma: no cover - direct CLI execution.
    import dense_davis_benchmark as davis  # type: ignore
    import kirhub_external_validation as kirhub  # type: ignore
    import replicated_pair_retrieval as retrieval  # type: ignore
    import residual_target_geometry_validation as geometry  # type: ignore


PACKAGE = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PACKAGE / "results" / "centering_panel_sensitivity"
DEFAULT_PKIS1_ZIP = Path("/tmp/pkis1_supplement.zip")
DEFAULT_KIRHUB_WORKBOOK = Path("/tmp/kirhub_supp_tables.xlsx")
DEFAULT_LOCKED_PAIRS = (
    PACKAGE / "results" / "replicated_pair_retrieval" / "target_pairs.csv"
)
DEFAULT_SEED = 20260831

TARGETS_21 = tuple(davis.TARGET_MAP)
TARGETS_20 = tuple(geometry.PKIS1_TARGET_MAP)
if tuple(target for target in TARGETS_21 if target != "PTK2") != TARGETS_20:
    raise RuntimeError("the ordered 20- and 21-target kinase panels no longer align")


def row_center(matrix: np.ndarray) -> np.ndarray:
    """Remove the unweighted per-row mean from a finite dense matrix."""
    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("row centering requires a finite two-dimensional matrix")
    if values.shape[1] < 2:
        raise ValueError("row centering requires at least two target columns")
    return values - values.mean(axis=1, keepdims=True)


def centered_geometry(
    full_matrix: np.ndarray,
    target_indices: tuple[int, ...],
    *,
    center_over_all_columns: bool,
) -> tuple[np.ndarray, float]:
    """Return target correlation and row-vs-two-way numerical discrepancy.

    When ``center_over_all_columns`` is false, kinase targets are extracted
    before row centering.  Otherwise the row mean is estimated over the complete
    target panel and the requested columns are extracted after centering.
    """
    matrix = np.asarray(full_matrix, dtype=np.float64)
    indices = np.asarray(target_indices, dtype=int)
    if matrix.ndim != 2 or not np.isfinite(matrix).all():
        raise ValueError("centering-panel input must be a finite 2-D matrix")
    if len(indices) < 2 or len(set(indices.tolist())) != len(indices):
        raise ValueError("target indices must be unique and contain at least two columns")
    if np.any(indices < 0) or np.any(indices >= matrix.shape[1]):
        raise ValueError("a target index lies outside the full panel")

    if center_over_all_columns:
        row_only = row_center(matrix)[:, indices]
        two_way = geometry.two_way_center(matrix)[:, indices]
    else:
        selected = matrix[:, indices]
        row_only = row_center(selected)
        two_way = geometry.two_way_center(selected)
    row_correlation = geometry.target_correlation(row_only)
    two_way_correlation = geometry.target_correlation(two_way)
    discrepancy = float(np.max(np.abs(row_correlation - two_way_correlation)))
    return row_correlation, discrepancy


def _score_columns(path: Path) -> tuple[str, ...]:
    columns = tuple(pd.read_csv(path, sep="\t", nrows=0).columns)
    if len(columns) != 60 or columns[:2] != ("inchikey", "smiles"):
        raise ValueError("the DOCKSTRING release no longer has 58 ordered score columns")
    scores = columns[2:]
    if len(scores) != 58 or len(set(scores)) != 58:
        raise ValueError("the DOCKSTRING score-column order is invalid")
    return scores


def load_reference(
    pkis1_zip: Path,
) -> tuple[np.ndarray, tuple[str, ...], dict, dict[str, np.ndarray]]:
    """Load the clipped, chemically de-leaked complete 58-target reference."""
    dockstring, davis_identity, davis_experiment, _ = davis._load_inputs(  # noqa: SLF001
        davis.DEFAULT_DOCKSTRING,
        davis.DEFAULT_DAVIS,
        identity_scan="full",
    )
    pkis2_frame = geometry.load_pkis2_full()
    pkis1_frame = geometry.load_pkis1_full(pkis1_zip)
    excluded_blocks = set(davis_identity.connectivity_block.dropna())
    excluded_blocks |= set(pkis2_frame.connectivity_block.dropna())
    excluded_blocks |= set(pkis1_frame.connectivity_block.dropna())
    keep = ~dockstring.connectivity_block.isin(excluded_blocks)

    score_columns = _score_columns(davis.DEFAULT_DOCKSTRING)
    released = pd.read_csv(
        davis.DEFAULT_DOCKSTRING,
        sep="\t",
        usecols=list(score_columns),
        dtype={target: np.float64 for target in score_columns},
    )
    selected_rows = dockstring.dockstring_row.to_numpy(dtype=int)
    complete_scores = released.iloc[selected_rows].to_numpy(dtype=np.float64)
    if complete_scores.shape != (260_060, 58) or not np.isfinite(complete_scores).all():
        raise ValueError(
            "the reconstructed complete DOCKSTRING surface is not 260,060 x 58"
        )
    reference = np.minimum(complete_scores[keep.to_numpy(dtype=bool)], 0.0)
    if len(reference) < 259_000:
        raise ValueError("unexpectedly many DOCKSTRING rows were excluded")

    experiments = {
        "DAVIS": davis_experiment[list(TARGETS_21)].to_numpy(dtype=np.float64),
        "PKIS2": pkis2_frame[list(TARGETS_21)].to_numpy(dtype=np.float64),
        "PKIS1": pkis1_frame[list(TARGETS_20)].to_numpy(dtype=np.float64),
    }
    support = {
        "dockstring_complete_rows_before_exclusion": int(len(complete_scores)),
        "dockstring_reference_rows": int(len(reference)),
        "dockstring_rows_excluded": int((~keep).sum()),
        "excluded_connectivity_blocks": int(len(excluded_blocks)),
        "all_target_count": int(len(score_columns)),
        "all_target_order": list(score_columns),
        "positive_score_handling": "clip to zero before either centering contract",
        "reference_exclusion": (
            "union of DAVIS, PKIS2, and PKIS1 Standard-InChI connectivity blocks"
        ),
    }
    sources = {
        "dockstring_sha256": geometry.sha256_file(davis.DEFAULT_DOCKSTRING),
        "davis_sha256": geometry.sha256_file(davis.DEFAULT_DAVIS),
        "pkis2_sha256": geometry.sha256_file(geometry.pkis2.DEFAULT_PKIS2),
        "pkis1_sha256": geometry.sha256_file(pkis1_zip),
    }
    return reference, score_columns, support, {**experiments, "_sources": sources}


def _paired_concordance_qap(
    kinase_panel_geometry: np.ndarray,
    full_panel_geometry: np.ndarray,
    experimental_geometry: np.ndarray,
    permutations: int,
    seed: int,
) -> dict:
    result = geometry.fixed_experimental_geometry_paired_qap(
        kinase_panel_geometry,
        full_panel_geometry,
        experimental_geometry,
        permutations,
        seed,
    )
    return {
        "kinase_panel_centered_spearman": result["raw_docking_concordance"],
        "all_58_centered_then_extracted_spearman": result[
            "centered_docking_concordance"
        ],
        "all_58_minus_kinase_panel": result["centered_minus_raw_docking"],
        "paired_target_label_qap_p_positive_delta": result[
            "one_sided_p_positive_delta"
        ],
        "paired_delta_null_interval_95": result["null_interval_95"],
        "permutations": int(permutations),
        "seed": int(seed),
    }


def load_locked_labels(path: Path) -> np.ndarray:
    frame = pd.read_csv(path)
    required = {"target_a", "target_b", "primary_replicated_positive"}
    if not required.issubset(frame):
        raise ValueError("locked target-pair artifact lacks required columns")
    pair_to_label = {
        tuple(sorted((str(row.target_a), str(row.target_b)))): bool(
            row.primary_replicated_positive
        )
        for row in frame.itertuples(index=False)
    }
    if len(pair_to_label) != 190:
        raise ValueError("locked artifact does not contain exactly 190 unique pairs")
    labels = []
    for first_index, first in enumerate(TARGETS_20):
        for second in TARGETS_20[first_index + 1 :]:
            key = tuple(sorted((first, second)))
            if key not in pair_to_label:
                raise ValueError(f"locked artifact is missing target pair {key}")
            labels.append(pair_to_label[key])
    out = np.asarray(labels, dtype=bool)
    if int(out.sum()) != 15:
        raise ValueError("locked endpoint no longer contains exactly 15 positives")
    return out


def _relabel_retrieval_qap(result: dict) -> dict:
    return {
        "permutations": result["permutations"],
        "seed": result["seed"],
        "metrics": {
            metric: {
                "kinase_panel_centered": values["raw_docking"],
                "all_58_centered_then_extracted": values["centered_docking"],
                "all_58_minus_kinase_panel": values["centered_minus_raw"],
                "paired_target_label_qap_p_positive_delta": values[
                    "paired_target_label_qap_p_positive_gain"
                ],
                "paired_delta_null_interval_95": values[
                    "paired_delta_null_interval_95"
                ],
            }
            for metric, values in result["metrics"].items()
        },
    }


def projection_matched_panel_delta_null(
    reference: np.ndarray,
    score_columns: tuple[str, ...],
    experiments: dict[str, np.ndarray],
    locked_labels: np.ndarray,
    *,
    reference_size: int,
    repeats: int,
    seed: int,
) -> dict:
    """Independent-column null followed by both row-centering projections.

    This is a sensitivity for the *difference* between centering panels.  It
    preserves each target's empirical score marginal on a fixed reference
    subsample, destroys ligand-wise target coordination, and then applies both
    projections to the same permuted matrix in every draw.
    """
    if reference_size < 100 or reference_size > len(reference):
        raise ValueError("invalid projection-null reference size")
    if repeats < 1:
        raise ValueError("projection-null repeats must be positive")
    index = {target: position for position, target in enumerate(score_columns)}
    targets_by_panel = {
        "DAVIS": TARGETS_21,
        "PKIS2": TARGETS_21,
        "PKIS1": TARGETS_20,
        "KiRHub": TARGETS_20,
    }
    indices_by_count = {
        21: tuple(index[target] for target in TARGETS_21),
        20: tuple(index[target] for target in TARGETS_20),
    }
    experimental_geometry = {
        panel: geometry.geometry_correlation(matrix, "center_then_correlation")
        for panel, matrix in experiments.items()
    }
    rng = np.random.default_rng(seed)
    rows = np.sort(rng.choice(len(reference), reference_size, replace=False))
    base = np.asarray(reference[rows], dtype=np.float64)
    def projected_geometries(matrix: np.ndarray) -> dict[int, tuple[np.ndarray, np.ndarray]]:
        full_row_centered = row_center(matrix)
        result: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for target_count, target_indices in indices_by_count.items():
            selected = matrix[:, target_indices]
            kinase = geometry.target_correlation(row_center(selected))
            all_targets = geometry.target_correlation(
                full_row_centered[:, target_indices]
            )
            result[target_count] = (kinase, all_targets)
        return result

    observed_geometries = projected_geometries(base)
    observed: dict[str, float] = {}
    for panel, targets in targets_by_panel.items():
        kinase, all_targets = observed_geometries[len(targets)]
        observed[panel] = geometry.geometry_concordance(
            all_targets, experimental_geometry[panel]
        ) - geometry.geometry_concordance(kinase, experimental_geometry[panel])

    observed_locked: dict[str, float] = {}
    locked_kinase, locked_all = observed_geometries[20]
    for metric in ("roc_auc", "average_precision"):
        observed_locked[metric] = float(
            retrieval.retrieval_metrics(
                locked_labels, geometry.upper_triangle(locked_all)
            )[metric]
            - retrieval.retrieval_metrics(
                locked_labels, geometry.upper_triangle(locked_kinase)
            )[metric]
        )

    null = {
        panel: np.empty(repeats, dtype=np.float64) for panel in targets_by_panel
    }
    locked_null = {
        metric: np.empty(repeats, dtype=np.float64)
        for metric in ("roc_auc", "average_precision")
    }
    work = np.empty_like(base)
    for repetition in range(repeats):
        for column in range(base.shape[1]):
            work[:, column] = base[rng.permutation(reference_size), column]
        projected = projected_geometries(work)
        for panel, targets in targets_by_panel.items():
            kinase, all_targets = projected[len(targets)]
            null[panel][repetition] = geometry.geometry_concordance(
                all_targets, experimental_geometry[panel]
            ) - geometry.geometry_concordance(
                kinase, experimental_geometry[panel]
            )
        locked_kinase, locked_all = projected[20]
        for metric in locked_null:
            locked_null[metric][repetition] = (
                retrieval.retrieval_metrics(
                    locked_labels, geometry.upper_triangle(locked_all)
                )[metric]
                - retrieval.retrieval_metrics(
                    locked_labels, geometry.upper_triangle(locked_kinase)
                )[metric]
            )
    return {
        "null": (
            "independent within-target ligand permutations on one fixed reference "
            "support, followed by both centering-panel projections"
        ),
        "reference_ligands": int(reference_size),
        "repeats": int(repeats),
        "seed": int(seed),
        "panels": {
            panel: {
                "matched_support_observed_all58_minus_kinase": float(observed[panel]),
                "two_sided_p_absolute_delta": float(
                    (1 + np.sum(np.abs(null[panel]) >= abs(observed[panel])))
                    / (repeats + 1)
                ),
                "one_sided_p_positive_delta": float(
                    (1 + np.sum(null[panel] >= observed[panel])) / (repeats + 1)
                ),
                "null_median": float(np.median(null[panel])),
                "null_interval_95": [
                    float(np.quantile(null[panel], 0.025)),
                    float(np.quantile(null[panel], 0.975)),
                ],
            }
            for panel in targets_by_panel
        },
        "locked_15_pair_retrieval": {
            metric: {
                "matched_support_observed_all58_minus_kinase": float(
                    observed_locked[metric]
                ),
                "two_sided_p_absolute_delta": float(
                    (
                        1
                        + np.sum(
                            np.abs(locked_null[metric])
                            >= abs(observed_locked[metric])
                        )
                    )
                    / (repeats + 1)
                ),
                "one_sided_p_positive_delta": float(
                    (
                        1
                        + np.sum(locked_null[metric] >= observed_locked[metric])
                    )
                    / (repeats + 1)
                ),
                "null_median": float(np.median(locked_null[metric])),
                "null_interval_95": [
                    float(np.quantile(locked_null[metric], 0.025)),
                    float(np.quantile(locked_null[metric], 0.975)),
                ],
            }
            for metric in locked_null
        },
    }


def run_analysis(
    *,
    pkis1_zip: Path,
    kirhub_workbook: Path,
    locked_pairs: Path,
    qap_permutations: int,
    projection_null_repeats: int,
    projection_reference_size: int,
    seed: int,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    if qap_permutations < 1:
        raise ValueError("qap_permutations must be positive")
    reference, score_columns, support, payload = load_reference(pkis1_zip)
    sources = payload.pop("_sources")
    experiments = payload
    kirhub_frame = kirhub.load_kirhub(kirhub_workbook)
    experiments["KiRHub"] = kirhub_frame[list(TARGETS_20)].to_numpy(
        dtype=np.float64
    )
    sources["kirhub_sha256"] = geometry.sha256_file(kirhub_workbook)

    score_index = {target: position for position, target in enumerate(score_columns)}
    target_specs: OrderedDict[str, tuple[str, ...]] = OrderedDict(
        [
            ("DAVIS", TARGETS_21),
            ("PKIS2", TARGETS_21),
            ("PKIS1", TARGETS_20),
            ("KiRHub", TARGETS_20),
        ]
    )
    continuous_rows: list[dict] = []
    paired_qap: dict[str, dict] = {}
    docking_geometries: dict[int, dict[str, np.ndarray]] = {}
    docking_geometry_agreement: dict[str, dict] = {}
    invariance: dict[str, float] = {}
    for target_count, targets in ((21, TARGETS_21), (20, TARGETS_20)):
        target_indices = tuple(score_index[target] for target in targets)
        kinase, kinase_error = centered_geometry(
            reference, target_indices, center_over_all_columns=False
        )
        all_targets, all_error = centered_geometry(
            reference, target_indices, center_over_all_columns=True
        )
        docking_geometries[target_count] = {
            "kinase_panel": kinase,
            "all_58_then_extract": all_targets,
        }
        docking_geometry_agreement[f"{target_count}_target_block"] = {
            "spearman_of_target_pair_correlations": geometry.geometry_concordance(
                kinase, all_targets
            ),
            "mean_absolute_correlation_difference": float(
                np.mean(np.abs(kinase - all_targets))
            ),
            "maximum_absolute_correlation_difference": float(
                np.max(np.abs(kinase - all_targets))
            ),
            "kinase_panel_correlation_PR": retrieval.correlation_pr(kinase),
            "all_58_then_extract_correlation_PR": retrieval.correlation_pr(
                all_targets
            ),
        }
        invariance[f"kinase_{target_count}_max_abs_correlation_difference"] = (
            kinase_error
        )
        invariance[f"all58_extract_{target_count}_max_abs_correlation_difference"] = (
            all_error
        )

    for panel_index, (panel, targets) in enumerate(target_specs.items()):
        target_count = len(targets)
        experiment_geometry = geometry.geometry_correlation(
            experiments[panel], "center_then_correlation"
        )
        kinase = docking_geometries[target_count]["kinase_panel"]
        all_targets = docking_geometries[target_count]["all_58_then_extract"]
        comparison = _paired_concordance_qap(
            kinase,
            all_targets,
            experiment_geometry,
            qap_permutations,
            seed + panel_index * 100_000,
        )
        paired_qap[panel] = comparison
        for contract, predictor in (
            ("kinase_validation_panel", kinase),
            ("all_58_then_extract_kinases", all_targets),
        ):
            qap = geometry.qap_test(
                predictor,
                experiment_geometry,
                qap_permutations,
                seed + 500_000 + panel_index * 200_000 + (contract.startswith("all")) * 100_000,
            )
            continuous_rows.append(
                {
                    "experimental_panel": panel,
                    "experimental_ligands": int(len(experiments[panel])),
                    "kinase_targets": int(target_count),
                    "docking_row_centering_panel": contract,
                    "concordance_spearman": qap["observed_spearman"],
                    "target_label_qap_p_positive": qap["one_sided_p_positive"],
                    "qap_permutations": int(qap_permutations),
                }
            )

    labels = load_locked_labels(locked_pairs)
    raw_20 = geometry.target_correlation(
        reference[:, [score_index[target] for target in TARGETS_20]]
    )
    kinase_20 = docking_geometries[20]["kinase_panel"]
    all_20 = docking_geometries[20]["all_58_then_extract"]
    locked_rows = []
    for representation, predictor in (
        ("raw_unprojected", raw_20),
        ("kinase_validation_panel_centered", kinase_20),
        ("all_58_centered_then_extract_kinases", all_20),
    ):
        metrics = retrieval.retrieval_metrics(
            labels, geometry.upper_triangle(predictor)
        )
        locked_rows.append(
            {
                "representation": representation,
                "positive_pairs": int(labels.sum()),
                "total_pairs": int(len(labels)),
                **metrics,
            }
        )
    locked_qap = _relabel_retrieval_qap(
        retrieval.fixed_endpoint_qap(
            labels,
            kinase_20,
            all_20,
            qap_permutations,
            seed + 1_500_000,
        )
    )
    raw_to_all_qap = retrieval.fixed_endpoint_qap(
        labels,
        raw_20,
        all_20,
        qap_permutations,
        seed + 1_600_000,
    )

    projection_null = projection_matched_panel_delta_null(
        reference,
        score_columns,
        experiments,
        labels,
        reference_size=projection_reference_size,
        repeats=projection_null_repeats,
        seed=seed + 2_000_000,
    )

    continuous_frame = pd.DataFrame(continuous_rows)
    locked_frame = pd.DataFrame(locked_rows)
    report = {
        "analysis": "DOCKSTRING row-centering-panel sensitivity",
        "status": "focused sensitivity of the fixed target-geometry endpoints",
        "question": (
            "Does residual kinase target geometry depend on estimating each ligand's "
            "row mean only from validation kinases rather than all 58 released targets?"
        ),
        "centering_contracts": {
            "kinase_validation_panel": (
                "extract the ordered 21-target DAVIS/PKIS2 or 20-target PKIS1/KiRHub "
                "block, then remove each ligand's mean over that block"
            ),
            "all_58_then_extract_kinases": (
                "remove each ligand's mean over all 58 released DOCKSTRING targets, "
                "then extract the identical ordered kinase block"
            ),
        },
        "continuous_geometry": {
            panel: paired_qap[panel] for panel in target_specs
        },
        "docking_geometry_agreement_between_centering_panels": (
            docking_geometry_agreement
        ),
        "locked_15_pair_retrieval": {
            "endpoint": (
                "unchanged 15 of 190 pairs in the centered upper 10% in at least "
                "two of DAVIS, PKIS2, and PKIS1"
            ),
            "point_estimates": locked_frame.to_dict(orient="records"),
            "all_58_vs_kinase_panel_paired_qap": locked_qap,
            "all_58_vs_raw_paired_qap": raw_to_all_qap,
        },
        "projection_matched_panel_delta_null": projection_null,
        "row_only_vs_two_way_correlation_identity": {
            "algebra": (
                "two-way-centered column j equals row-centered column j plus a "
                "column-specific constant; Pearson target correlations are invariant "
                "to these translations"
            ),
            "maximum_absolute_differences": invariance,
            "all_below_1e_minus_12": bool(max(invariance.values()) < 1e-12),
        },
        "support": support,
        "sources": sources,
        "configuration": {
            "qap_permutations": int(qap_permutations),
            "projection_null_repeats": int(projection_null_repeats),
            "projection_reference_size": int(projection_reference_size),
            "seed": int(seed),
            "locked_pairs_sha256": geometry.sha256_file(locked_pairs),
            "kirhub_identity_exclusion_boundary": (
                "KiRHub supplies compound names but no structures, so KiRHub-specific "
                "connectivity exclusion is unavailable"
            ),
        },
    }
    return report, continuous_frame, locked_frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pkis1-zip", type=Path, default=DEFAULT_PKIS1_ZIP)
    parser.add_argument(
        "--kirhub-workbook", type=Path, default=DEFAULT_KIRHUB_WORKBOOK
    )
    parser.add_argument("--locked-pairs", type=Path, default=DEFAULT_LOCKED_PAIRS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--qap-permutations", type=int, default=20_000)
    parser.add_argument("--projection-null-repeats", type=int, default=1_000)
    parser.add_argument("--projection-reference-size", type=int, default=15_000)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report, continuous, locked = run_analysis(
        pkis1_zip=args.pkis1_zip,
        kirhub_workbook=args.kirhub_workbook,
        locked_pairs=args.locked_pairs,
        qap_permutations=args.qap_permutations,
        projection_null_repeats=args.projection_null_repeats,
        projection_reference_size=args.projection_reference_size,
        seed=args.seed,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "summary.json").open("w") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    continuous.to_csv(args.output_dir / "continuous_geometry.csv", index=False)
    locked.to_csv(args.output_dir / "locked_pair_retrieval.csv", index=False)
    print(
        json.dumps(
            {
                "continuous_geometry": report["continuous_geometry"],
                "locked_pair_retrieval": report["locked_15_pair_retrieval"][
                    "point_estimates"
                ],
                "row_two_way_identity": report[
                    "row_only_vs_two_way_correlation_identity"
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
