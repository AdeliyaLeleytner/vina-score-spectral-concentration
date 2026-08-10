#!/usr/bin/env python3
"""Fixed-20-target decomposition of docking/experimental geometry concordance."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from . import dense_davis_benchmark as davis
    from . import kirhub_external_validation as kirhub
    from . import residual_target_geometry_validation as geometry
except ImportError:  # pragma: no cover
    import dense_davis_benchmark as davis  # type: ignore
    import kirhub_external_validation as kirhub  # type: ignore
    import residual_target_geometry_validation as geometry  # type: ignore


PACKAGE = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PACKAGE / "results/fixed20_estimand_decomposition"
RAW_EXPERIMENT_QAP_SEED_OFFSET = 20_000_000
MATCHED_NULL_SUPPORT_SEED_OFFSET = 30_000_000
MATCHED_NULL_PERMUTATION_SEED_OFFSET = 31_000_000
DEFAULT_IDENTITY_CONTRACT = (
    PACKAGE / "data/frozen/dockstring_identity_contract_2026-08-03.csv.gz"
)


def load_docking_reference(
    pkis1_zip: Path,
    identity_contract: Path = DEFAULT_IDENTITY_CONTRACT,
) -> tuple[np.ndarray, dict[str, object]]:
    """Load the de-leaked fixed-20 reference using the frozen identity contract.

    This recreates :func:`kirhub.load_docking_geometries` without recomputing
    Standard InChIKeys for all 260,060 complete DOCKSTRING rows.  The frozen
    contract was generated with the same raw-molecule identity definition and
    records the released row index, so row alignment is checked before use.
    """
    source = pd.read_csv(davis.DEFAULT_DOCKSTRING, sep="\t")
    score_columns = [
        column for column in source.columns if column not in {"inchikey", "smiles"}
    ]
    complete = ~source[score_columns].isna().any(axis=1)
    complete_source_rows = source.index[complete].to_numpy(dtype=np.int64)
    selected_scores = source.loc[complete, list(kirhub.TARGETS)].to_numpy(
        dtype=np.float64
    )
    contract = pd.read_csv(
        identity_contract,
        usecols=[
            "source_row_index",
            "complete_support_row_index",
            "raw_connectivity",
        ],
    )
    if len(contract) != len(selected_scores) or len(contract) != 260_060:
        raise ValueError("frozen DOCKSTRING identity contract has unexpected support")
    if not np.array_equal(
        contract.source_row_index.to_numpy(dtype=np.int64), complete_source_rows
    ):
        raise ValueError("identity contract does not align to released DOCKSTRING rows")
    if not np.array_equal(
        contract.complete_support_row_index.to_numpy(dtype=np.int64),
        np.arange(len(contract), dtype=np.int64),
    ):
        raise ValueError("identity contract complete-support indices are not contiguous")

    davis_source = pd.read_csv(
        davis.DEFAULT_DAVIS,
        sep="\t",
        usecols=["drug_name", "compound_iso_smiles"],
    ).drop_duplicates()
    davis_identity = davis._identity_table(  # noqa: SLF001
        davis_source,
        "compound_iso_smiles",
        scan="full",
    )
    pkis2_frame = geometry.load_pkis2_full()
    pkis1_frame = geometry.load_pkis1_full(pkis1_zip)
    excluded_blocks = set(davis_identity.connectivity_block.dropna())
    excluded_blocks |= set(pkis2_frame.connectivity_block.dropna())
    excluded_blocks |= set(pkis1_frame.connectivity_block.dropna())
    keep = ~contract.raw_connectivity.isin(excluded_blocks).to_numpy(dtype=bool)
    reference = np.minimum(selected_scores[keep], 0.0)
    if reference.shape != (259_579, 20):
        raise ValueError(f"unexpected fixed-20 reference shape: {reference.shape}")
    return reference, {
        "complete_dockstring_rows_before_exclusion": int(len(selected_scores)),
        "reference_rows_after_old_panel_connectivity_exclusion": int(len(reference)),
        "excluded_connectivity_blocks": int(len(excluded_blocks)),
        "positive_scores_clipped_to_zero_primary": True,
        "identity_contract": str(identity_contract.relative_to(PACKAGE)),
        "identity_contract_sha256": geometry.sha256_file(identity_contract),
        "identity_definition": (
            "first 14 characters of the Standard InChIKey generated from the raw "
            "parsed molecule, without FragmentParent or Uncharger"
        ),
        "kirhub_rows_excluded_from_reference": None,
        "kirhub_exclusion_boundary": (
            "The KiRHub supplement provides names but no structures; its remaining "
            "compounds could not be excluded by molecular identity."
        ),
    }


def _correlation_from_covariance(covariance: np.ndarray) -> np.ndarray:
    covariance = np.asarray(covariance, dtype=np.float64)
    scale = np.sqrt(np.diag(covariance))
    if np.any(scale <= 1e-14):
        raise ValueError("null draw contains a constant target")
    correlation = covariance / np.outer(scale, scale)
    correlation = np.clip(correlation, -1.0, 1.0)
    np.fill_diagonal(correlation, 1.0)
    return correlation


def paired_docking_geometries(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return raw and row-centred target correlations from one covariance.

    For a column-centred score matrix with covariance ``S``, row centring over
    targets changes that covariance to ``H S H``, where
    ``H = I - 11'/P``.  This is algebraically identical to materialising the
    two-way-centred ligand-by-target matrix and is materially faster inside the
    permutation loop.
    """
    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 3 or values.shape[1] < 2:
        raise ValueError("paired docking geometries require a dense 2-D matrix")
    if not np.isfinite(values).all():
        raise ValueError("paired docking geometries require finite values")
    centered = values - values.mean(axis=0, keepdims=True)
    covariance = centered.T @ centered / (len(values) - 1)
    target_count = values.shape[1]
    projection = np.eye(target_count) - np.ones(
        (target_count, target_count), dtype=np.float64
    ) / target_count
    residual_covariance = projection @ covariance @ projection
    return (
        _correlation_from_covariance(covariance),
        _correlation_from_covariance(residual_covariance),
    )


def transformation_matched_independent_column_null(
    reference: np.ndarray,
    experiments: dict[str, np.ndarray],
    *,
    reference_size: int,
    repeats: int,
    support_seed: int,
    permutation_seed: int,
) -> tuple[dict[str, object], pd.DataFrame]:
    """Test docking-side increments against their mechanical centring baseline.

    A single fixed reference subsample is selected without replacement.  Within
    each null draw, ligand identities are independently permuted inside every
    docking-target column, preserving its exact empirical marginal while
    destroying ligand-wise target coordination.  Raw and row-centred docking
    geometries are then computed from that *same* permuted surface and compared
    with each unchanged experimental endpoint.  Consequently, the null contrast
    is paired and contains the geometry induced mechanically by row centring.
    """
    values = np.asarray(reference, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 20 or not np.isfinite(values).all():
        raise ValueError("matched null requires a finite ligand-by-20-target reference")
    if reference_size < 100 or reference_size > len(values):
        raise ValueError("invalid matched-null reference size")
    if repeats < 1:
        raise ValueError("matched-null repeats must be positive")
    endpoints: dict[str, dict[str, np.ndarray]] = {}
    for panel, experiment in experiments.items():
        matrix = np.asarray(experiment, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[1] != 20 or not np.isfinite(matrix).all():
            raise ValueError(f"{panel} is not a finite ligand-by-20-target panel")
        endpoints[panel] = {
            "raw_experimental_target_geometry": geometry.target_correlation(matrix),
            "two_way_centered_experimental_target_geometry": (
                geometry.geometry_correlation(matrix, "center_then_correlation")
            ),
        }

    full_raw, full_centered = paired_docking_geometries(values)
    support_rng = np.random.default_rng(support_seed)
    support_rows = np.sort(
        support_rng.choice(len(values), reference_size, replace=False)
    )
    base = np.asarray(values[support_rows], dtype=np.float64)
    observed_raw, observed_centered = paired_docking_geometries(base)

    panel_summary: dict[str, dict[str, object]] = {}
    for panel, endpoint_geometries in endpoints.items():
        panel_summary[panel] = {}
        for endpoint_name, endpoint in endpoint_geometries.items():
            full_raw_concordance = geometry.geometry_concordance(full_raw, endpoint)
            full_centered_concordance = geometry.geometry_concordance(
                full_centered, endpoint
            )
            support_raw_concordance = geometry.geometry_concordance(
                observed_raw, endpoint
            )
            support_centered_concordance = geometry.geometry_concordance(
                observed_centered, endpoint
            )
            panel_summary[panel][endpoint_name] = {
                "full_reference_observed_raw_docking_concordance": (
                    full_raw_concordance
                ),
                "full_reference_observed_centered_docking_concordance": (
                    full_centered_concordance
                ),
                "full_reference_observed_centered_minus_raw_docking": (
                    full_centered_concordance - full_raw_concordance
                ),
                "matched_support_observed_raw_docking_concordance": (
                    support_raw_concordance
                ),
                "matched_support_observed_centered_docking_concordance": (
                    support_centered_concordance
                ),
                "matched_support_observed_centered_minus_raw_docking": (
                    support_centered_concordance - support_raw_concordance
                ),
            }

    rng = np.random.default_rng(permutation_seed)
    work = np.empty_like(base)
    rows: list[dict[str, object]] = []
    for repetition in range(repeats):
        for column in range(base.shape[1]):
            work[:, column] = base[rng.permutation(reference_size), column]
        null_raw, null_centered = paired_docking_geometries(work)
        for panel, endpoint_geometries in endpoints.items():
            for endpoint_name, endpoint in endpoint_geometries.items():
                raw_concordance = geometry.geometry_concordance(null_raw, endpoint)
                centered_concordance = geometry.geometry_concordance(
                    null_centered, endpoint
                )
                rows.append(
                    {
                        "null_repetition": repetition,
                        "panel": panel,
                        "fixed_experimental_endpoint": endpoint_name,
                        "raw_docking_concordance": raw_concordance,
                        "centered_docking_concordance": centered_concordance,
                        "centered_minus_raw_docking": (
                            centered_concordance - raw_concordance
                        ),
                    }
                )
    frame = pd.DataFrame.from_records(rows)
    for panel, endpoint_records in panel_summary.items():
        for endpoint_name, record in endpoint_records.items():
            subset = frame.loc[
                frame.panel.eq(panel)
                & frame.fixed_experimental_endpoint.eq(endpoint_name)
            ]
            null_raw = subset.raw_docking_concordance.to_numpy(dtype=np.float64)
            null_centered = subset.centered_docking_concordance.to_numpy(
                dtype=np.float64
            )
            null_delta = subset.centered_minus_raw_docking.to_numpy(dtype=np.float64)
            observed_raw = record[
                "matched_support_observed_raw_docking_concordance"
            ]
            observed_centered = record[
                "matched_support_observed_centered_docking_concordance"
            ]
            observed_delta = record[
                "matched_support_observed_centered_minus_raw_docking"
            ]
            record.update(
                {
                    "null_raw_mean": float(null_raw.mean()),
                    "null_centered_mean": float(null_centered.mean()),
                    "null_centered_minus_raw_mean": float(null_delta.mean()),
                    "null_centered_minus_raw_median": float(np.median(null_delta)),
                    "null_centered_minus_raw_interval_95": [
                        float(np.quantile(null_delta, 0.025)),
                        float(np.quantile(null_delta, 0.975)),
                    ],
                    "p_observed_raw_at_least_as_large": float(
                        (1 + np.sum(null_raw >= observed_raw)) / (repeats + 1)
                    ),
                    "p_observed_centered_at_least_as_large": float(
                        (1 + np.sum(null_centered >= observed_centered))
                        / (repeats + 1)
                    ),
                    "p_observed_gain_at_least_as_large": float(
                        (1 + np.sum(null_delta >= observed_delta)) / (repeats + 1)
                    ),
                }
            )

    return {
        "null": (
            "independently permute ligand identities within each docking-target "
            "column on one fixed reference support; preserve target marginals, "
            "variances, ties and zero clipping; derive raw and row-centred docking "
            "geometries from the identical permuted matrix while holding each "
            "experimental geometry fixed"
        ),
        "reference_ligands_full": int(len(values)),
        "reference_ligands_fixed_subsample": int(reference_size),
        "support_sampling": "simple random sample without replacement, selected once",
        "support_seed": int(support_seed),
        "support_row_index_sha256": hashlib.sha256(
            support_rows.astype("<i8", copy=False).tobytes()
        ).hexdigest(),
        "permutation_seed": int(permutation_seed),
        "repeats": int(repeats),
        "panels": panel_summary,
        "inference_boundary": (
            "Monte Carlo p-values compare each matched-support observed increment "
            "with paired null increments on that same fixed chemical support. "
            "Full-reference increments are descriptive and are not compared with "
            "the subsampled null distribution. Target-label QAP and this "
            "transformation-matched null test different hypotheses."
        ),
    }, frame


def decompose(
    raw_docking: np.ndarray,
    centered_docking: np.ndarray,
    experiment: np.ndarray,
    *,
    permutations: int,
    seed: int,
) -> dict[str, object]:
    raw_experiment = geometry.target_correlation(experiment)
    centered_experiment = geometry.geometry_correlation(
        experiment, "center_then_correlation"
    )
    rr = geometry.geometry_concordance(raw_docking, raw_experiment)
    rc = geometry.geometry_concordance(raw_docking, centered_experiment)
    cc = geometry.geometry_concordance(centered_docking, centered_experiment)
    cr = geometry.geometry_concordance(centered_docking, raw_experiment)
    docking_delta = geometry.fixed_experimental_geometry_paired_qap(
        raw_docking,
        centered_docking,
        centered_experiment,
        permutations,
        seed,
    )
    raw_experiment_docking_delta = geometry.fixed_experimental_geometry_paired_qap(
        raw_docking,
        centered_docking,
        raw_experiment,
        permutations,
        seed + RAW_EXPERIMENT_QAP_SEED_OFFSET,
    )
    raw_experiment_docking_delta["fixed_experimental_endpoint"] = (
        "raw experimental target geometry"
    )
    transition = geometry.matched_estimand_transition_qap(
        raw_docking,
        raw_experiment,
        centered_docking,
        centered_experiment,
        permutations,
        seed + 10_000_000,
    )
    total = cc - rr
    endpoint_increment = rc - rr
    predictor_increment = cc - rc
    predictor_first_increment = cr - rr
    endpoint_second_increment = cc - cr
    interaction = cc - cr - rc + rr
    shapley_endpoint = 0.5 * (endpoint_increment + endpoint_second_increment)
    shapley_docking = 0.5 * (predictor_first_increment + predictor_increment)
    return {
        "docking_raw__experimental_raw": rr,
        "docking_raw__experimental_centered": rc,
        "docking_centered__experimental_raw": cr,
        "docking_centered__experimental_centered": cc,
        "experimental_estimand_increment_with_raw_docking": endpoint_increment,
        "docking_transform_increment_with_centered_experiment": predictor_increment,
        "docking_transform_increment_with_raw_experiment": predictor_first_increment,
        "experimental_estimand_increment_with_centered_docking": endpoint_second_increment,
        "two_factor_interaction_on_spearman_scale": interaction,
        "path_averaged_descriptive_attribution": {
            "experimental_estimand": shapley_endpoint,
            "docking_transform": shapley_docking,
            "experimental_fraction_of_total": (
                shapley_endpoint / total if total != 0 else None
            ),
            "docking_fraction_of_total": (
                shapley_docking / total if total != 0 else None
            ),
            "boundary": (
                "Shapley-style averages of the two transformation orders on the "
                "Spearman scale; descriptive, not a causal or variance decomposition."
            ),
        },
        "matched_estimand_total_change": total,
        "fraction_of_matched_change_before_docking_centering": (
            endpoint_increment / total if total != 0 else None
        ),
        "sequential_fraction_boundary": (
            "This fraction follows the experimental-first path and is order-dependent; "
            "the complete 2 x 2 table and path-averaged attribution must accompany it."
        ),
        "paired_qap_for_docking_increment": docking_delta,
        "paired_qap_for_docking_increment_with_raw_experiment": (
            raw_experiment_docking_delta
        ),
        "paired_qap_for_joint_matched_transition": transition,
    }


def run(
    *,
    pkis1_zip: Path,
    kirhub_workbook: Path,
    permutations: int,
    matched_null_repeats: int,
    matched_null_reference_size: int,
    seed: int,
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    experiments = kirhub.load_old_experimental_panels(pkis1_zip)
    kirhub_frame = kirhub.load_kirhub(kirhub_workbook)
    experiments["KiRHub"] = kirhub_frame[list(kirhub.TARGETS)].to_numpy(
        dtype=np.float64
    )
    docking_reference, docking_support = load_docking_reference(pkis1_zip)
    raw_docking, centered_docking = paired_docking_geometries(docking_reference)
    if raw_docking.shape != (20, 20) or centered_docking.shape != (20, 20):
        raise ValueError("expected fixed 20-target docking geometries")

    panels: dict[str, object] = {}
    rows: list[dict[str, object]] = []
    for panel_index, (panel, experiment) in enumerate(experiments.items()):
        result = decompose(
            raw_docking,
            centered_docking,
            experiment,
            permutations=permutations,
            seed=seed + panel_index * 1_000_000,
        )
        panels[panel] = result
        rows.append(
            {
                "panel": panel,
                "experimental_ligands": int(len(experiment)),
                "targets": 20,
                "raw_docking_raw_experiment": result[
                    "docking_raw__experimental_raw"
                ],
                "raw_docking_centered_experiment": result[
                    "docking_raw__experimental_centered"
                ],
                "centered_docking_centered_experiment": result[
                    "docking_centered__experimental_centered"
                ],
                "experimental_estimand_increment": result[
                    "experimental_estimand_increment_with_raw_docking"
                ],
                "docking_transform_increment": result[
                    "docking_transform_increment_with_centered_experiment"
                ],
                "docking_transform_increment_with_raw_experiment": result[
                    "docking_transform_increment_with_raw_experiment"
                ],
                "experimental_estimand_increment_with_centered_docking": result[
                    "experimental_estimand_increment_with_centered_docking"
                ],
                "two_factor_interaction": result[
                    "two_factor_interaction_on_spearman_scale"
                ],
                "path_averaged_experimental_fraction": result[
                    "path_averaged_descriptive_attribution"
                ]["experimental_fraction_of_total"],
                "fraction_before_docking_centering": result[
                    "fraction_of_matched_change_before_docking_centering"
                ],
                "paired_qap_docking_increment_p_positive": result[
                    "paired_qap_for_docking_increment"
                ]["one_sided_p_positive_delta"],
                "paired_qap_docking_increment_with_raw_experiment_p_positive": result[
                    "paired_qap_for_docking_increment_with_raw_experiment"
                ]["one_sided_p_positive_delta"],
                "paired_qap_docking_increment_with_raw_experiment_null_lower_95": result[
                    "paired_qap_for_docking_increment_with_raw_experiment"
                ]["null_interval_95"][0],
                "paired_qap_docking_increment_with_raw_experiment_null_upper_95": result[
                    "paired_qap_for_docking_increment_with_raw_experiment"
                ]["null_interval_95"][1],
            }
        )
    matched_null, matched_null_frame = transformation_matched_independent_column_null(
        docking_reference,
        experiments,
        reference_size=matched_null_reference_size,
        repeats=matched_null_repeats,
        support_seed=seed + MATCHED_NULL_SUPPORT_SEED_OFFSET,
        permutation_seed=seed + MATCHED_NULL_PERMUTATION_SEED_OFFSET,
    )
    table = pd.DataFrame.from_records(rows)
    for row_index, panel in enumerate(table.panel):
        null_panel = matched_null["panels"][panel]
        for endpoint_prefix, endpoint_name in (
            ("raw_experiment", "raw_experimental_target_geometry"),
            (
                "centered_experiment",
                "two_way_centered_experimental_target_geometry",
            ),
        ):
            record = null_panel[endpoint_name]
            table.loc[
                row_index,
                f"matched_support_docking_increment_with_{endpoint_prefix}",
            ] = record["matched_support_observed_centered_minus_raw_docking"]
            table.loc[
                row_index,
                f"matched_null_mean_docking_increment_with_{endpoint_prefix}",
            ] = record["null_centered_minus_raw_mean"]
            table.loc[
                row_index,
                f"matched_null_p_positive_docking_increment_with_{endpoint_prefix}",
            ] = record["p_observed_gain_at_least_as_large"]
    summary = {
        "analysis": "fixed-20-target docking-by-experimental estimand decomposition",
        "status": "exploratory_post_hoc",
        "targets": list(kirhub.TARGETS),
        "target_pairs": 190,
        "docking_reference": docking_support,
        "panels": panels,
        "transformation_matched_independent_column_null": matched_null,
        "configuration": {
            "permutations": permutations,
            "seed": seed,
            "permutation_unit": "complete target labels",
            "matched_null_repeats": int(matched_null_repeats),
            "matched_null_reference_size": int(matched_null_reference_size),
            "matched_null_support_seed": int(
                seed + MATCHED_NULL_SUPPORT_SEED_OFFSET
            ),
            "matched_null_permutation_seed": int(
                seed + MATCHED_NULL_PERMUTATION_SEED_OFFSET
            ),
        },
        "claim_boundary": (
            "A raw/raw versus centered/centered transition changes predictor and "
            "endpoint simultaneously. Only raw-docking versus centered-docking "
            "comparisons against the same fixed experimental geometry isolate the "
            "docking transform on this panel; both raw and centered experimental "
            "geometries are reported. The sequential percentage is "
            "order-dependent; all four cells and the path-averaged descriptive "
            "attribution are reported. Target-label QAP asks whether the observed "
            "docking increment is tied to target identities, whereas the paired "
            "independent-column null asks whether it exceeds the increment induced "
            "mechanically by applying row centring to marginal-matched independent "
            "docking columns."
        ),
    }
    return summary, table, matched_null_frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pkis1-zip", type=Path, default=Path("/tmp/pkis1_supplement.zip"))
    parser.add_argument(
        "--kirhub-workbook", type=Path, default=Path("/tmp/kirhub_supp_tables.xlsx")
    )
    parser.add_argument("--permutations", type=int, default=49_999)
    parser.add_argument("--matched-null-repeats", type=int, default=2_000)
    parser.add_argument("--matched-null-reference-size", type=int, default=15_000)
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    summary, table, matched_null_frame = run(
        pkis1_zip=args.pkis1_zip,
        kirhub_workbook=args.kirhub_workbook,
        permutations=args.permutations,
        matched_null_repeats=args.matched_null_repeats,
        matched_null_reference_size=args.matched_null_reference_size,
        seed=args.seed,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    table.to_csv(args.output / "decomposition.csv", index=False)
    matched_null_frame.to_csv(
        args.output / "transformation_matched_null.csv", index=False
    )
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
