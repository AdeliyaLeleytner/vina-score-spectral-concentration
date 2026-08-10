#!/usr/bin/env python3
"""Strict within-KLIFS-group target-label QAP sensitivity.

This focused, exploratory analysis keeps the fixed 20-kinase support and the
previously released centered-Vina target geometry unchanged.  It replaces the
unrestricted target-label null with complete node relabellings that are allowed
only within KLIFS kinase groups.  Experimental geometries and structural
control matrices remain fixed.

The restricted permutation is a conditional sensitivity, not proof that the
targets are exchangeable within KLIFS groups.  The output therefore reports the
exchangeability blocks, the pair positions that can move, and diagnostic nulls
showing how strongly the result is driven by the 11-target TK block.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import platform
from collections import OrderedDict
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import scipy
from scipy import stats

try:  # Package imports and direct script execution are both supported.
    from . import kirhub_external_validation as kirhub
    from . import klifs_pocket_control as klifs
except ImportError:  # pragma: no cover - direct CLI execution.
    import kirhub_external_validation as kirhub  # type: ignore
    import klifs_pocket_control as klifs  # type: ignore


PACKAGE = Path(__file__).resolve().parents[1]
DEFAULT_TARGET_ANNOTATIONS = (
    PACKAGE / "results" / "klifs_pocket_control" / "target_annotations.csv"
)
DEFAULT_TARGET_PAIRS = (
    PACKAGE / "results" / "klifs_pocket_control" / "target_pairs.csv"
)
DEFAULT_OUTPUT = PACKAGE / "results" / "strict_klifs_group_qap"
DEFAULT_SEED = 20260805
DEFAULT_MONTE_CARLO_PERMUTATIONS = 100_000

TARGET_ANNOTATIONS_SHA256 = (
    "005d118db9510f59355330afeda0280f40041c8d53bbeb41a41eb3c6aaf4847a"
)
TARGET_PAIRS_SHA256 = (
    "08238b207ac4f9f991193db0d0e03063403727f98aa1230cbfe636eeb6501a23"
)
TARGETS = tuple(klifs.TARGETS)

REQUIRED_PAIR_COLUMNS = {
    "target_a",
    "target_b",
    "mean_experimental_centered_pair_percentile",
    "centered_docking_pair_percentile",
    "receptor_domain_sequence_identity",
    "klifs_pocket_identity",
}
REQUIRED_ANNOTATION_COLUMNS = {
    "target",
    "klifs_group",
    "klifs_family",
}


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def release_provenance_path(path: Path) -> str:
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(PACKAGE.resolve()).as_posix()
    except ValueError as error:
        raise ValueError("release inputs must resolve inside the package") from error


def load_annotations(
    path: Path,
    *,
    expected_sha256: str | None = TARGET_ANNOTATIONS_SHA256,
) -> pd.DataFrame:
    path = Path(path)
    observed = sha256_file(path)
    if expected_sha256 is not None and observed != expected_sha256:
        raise ValueError(
            "target-annotation SHA256 mismatch: "
            f"expected {expected_sha256}, observed {observed}"
        )
    frame = pd.read_csv(path)
    if not REQUIRED_ANNOTATION_COLUMNS.issubset(frame.columns):
        raise ValueError("target annotations lack KLIFS group/family columns")
    if frame.target.duplicated().any() or set(frame.target) != set(TARGETS):
        raise ValueError("target annotations do not contain the fixed targets once")
    frame = frame.set_index("target").loc[list(TARGETS)].reset_index()
    if frame[["klifs_group", "klifs_family"]].isna().any().any():
        raise ValueError("KLIFS group/family annotations contain missing values")
    return frame


def load_pairs(
    path: Path,
    *,
    expected_sha256: str | None = TARGET_PAIRS_SHA256,
) -> pd.DataFrame:
    path = Path(path)
    observed = sha256_file(path)
    if expected_sha256 is not None and observed != expected_sha256:
        raise ValueError(
            "target-pair SHA256 mismatch: "
            f"expected {expected_sha256}, observed {observed}"
        )
    frame = pd.read_csv(path)
    if not REQUIRED_PAIR_COLUMNS.issubset(frame.columns):
        raise ValueError("target-pair artifact lacks required columns")
    expected_pairs = len(TARGETS) * (len(TARGETS) - 1) // 2
    if len(frame) != expected_pairs:
        raise ValueError(f"expected {expected_pairs} pairs, observed {len(frame)}")
    observed_pairs = {
        tuple(sorted((row.target_a, row.target_b)))
        for row in frame.itertuples(index=False)
    }
    expected_pair_set = {
        tuple(sorted((first, second)))
        for first, second in itertools.combinations(TARGETS, 2)
    }
    if observed_pairs != expected_pair_set:
        raise ValueError("target-pair artifact is not the complete fixed panel")
    if frame[list(REQUIRED_PAIR_COLUMNS - {"target_a", "target_b"})].isna().any().any():
        raise ValueError("target-pair artifact contains missing numeric values")
    return frame


def ordered_blocks(labels: Sequence[str]) -> OrderedDict[str, np.ndarray]:
    blocks: OrderedDict[str, list[int]] = OrderedDict()
    for index, label in enumerate(labels):
        blocks.setdefault(str(label), []).append(index)
    return OrderedDict(
        (label, np.asarray(indices, dtype=np.int16))
        for label, indices in blocks.items()
    )


def pair_index_matrix(target_count: int) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    triangle = np.triu_indices(target_count, k=1)
    index = np.full((target_count, target_count), -1, dtype=np.int32)
    for pair_index, (first, second) in enumerate(zip(*triangle)):
        index[first, second] = index[second, first] = pair_index
    return index, triangle


def orders_to_pair_maps(orders: np.ndarray) -> np.ndarray:
    orders = np.asarray(orders)
    if orders.ndim != 2:
        raise ValueError("target orders must be a two-dimensional array")
    target_count = orders.shape[1]
    expected = np.arange(target_count)
    if any(not np.array_equal(np.sort(order), expected) for order in orders):
        raise ValueError("every target order must be a complete bijection")
    pair_index, triangle = pair_index_matrix(target_count)
    return pair_index[orders[:, triangle[0]], orders[:, triangle[1]]]


def restricted_target_orders(
    block_labels: Sequence[str],
    permutations: int,
    seed: int,
    *,
    active_blocks: Iterable[str] | None = None,
) -> np.ndarray:
    """Sample complete target relabellings within specified blocks only."""
    if permutations < 1:
        raise ValueError("at least one permutation is required")
    labels = np.asarray(block_labels, dtype=object)
    blocks = ordered_blocks(labels)
    active = set(blocks) if active_blocks is None else set(active_blocks)
    unknown = active - set(blocks)
    if unknown:
        raise ValueError(f"unknown exchangeability blocks: {sorted(unknown)}")
    rng = np.random.default_rng(seed)
    base = np.arange(len(labels), dtype=np.int16)
    orders = np.tile(base, (permutations, 1))
    for repetition in range(permutations):
        for label, members in blocks.items():
            if label in active and len(members) > 1:
                orders[repetition, members] = rng.permutation(members)
    if not np.all(labels[orders] == labels[np.newaxis, :]):
        raise RuntimeError("a restricted order moved a target across KLIFS groups")
    return orders


def exact_restricted_target_orders(
    block_labels: Sequence[str],
    *,
    active_blocks: Iterable[str],
) -> np.ndarray:
    """Enumerate every relabelling for small selected exchangeability blocks."""
    labels = np.asarray(block_labels, dtype=object)
    blocks = ordered_blocks(labels)
    active = tuple(active_blocks)
    unknown = set(active) - set(blocks)
    if unknown:
        raise ValueError(f"unknown exchangeability blocks: {sorted(unknown)}")
    choices = [tuple(itertools.permutations(blocks[label])) for label in active]
    total = math.prod(len(options) for options in choices)
    if total > 100_000:
        raise ValueError("exact enumeration is limited to 100,000 orders")
    base = np.arange(len(labels), dtype=np.int16)
    rows: list[np.ndarray] = []
    for selected in itertools.product(*choices):
        order = base.copy()
        for label, replacement in zip(active, selected):
            order[blocks[label]] = replacement
        rows.append(order)
    orders = np.vstack(rows)
    if not np.all(labels[orders] == labels[np.newaxis, :]):
        raise RuntimeError("an exact restricted order crossed KLIFS groups")
    return orders


def _rank(values: np.ndarray) -> np.ndarray:
    return stats.rankdata(np.asarray(values, dtype=np.float64), method="average")


def restricted_partial_rank_qap(
    predictor: np.ndarray,
    endpoint: np.ndarray,
    controls: Sequence[np.ndarray],
    permutation_maps: np.ndarray,
    *,
    exact_enumeration: bool,
    chunk_size: int = 5_000,
) -> dict:
    """Partial Spearman QAP under precomputed complete-node relabellings."""
    predictor = np.asarray(predictor, dtype=np.float64)
    endpoint = np.asarray(endpoint, dtype=np.float64)
    target_count = len(predictor)
    matrices = (endpoint, *controls)
    if predictor.shape != (target_count, target_count) or any(
        np.asarray(matrix).shape != predictor.shape for matrix in matrices
    ):
        raise ValueError("all matrices must be aligned and square")
    triangle = np.triu_indices(target_count, k=1)
    pair_count = len(triangle[0])
    if permutation_maps.ndim != 2 or permutation_maps.shape[1] != pair_count:
        raise ValueError("permutation maps do not match the target panel")

    predictor_rank = _rank(predictor[triangle])
    endpoint_rank = _rank(endpoint[triangle])
    design = np.column_stack(
        [
            np.ones(pair_count, dtype=np.float64),
            *[_rank(np.asarray(control)[triangle]) for control in controls],
        ]
    )
    projection = np.linalg.pinv(design.T @ design) @ design.T
    endpoint_residual = endpoint_rank - design @ (projection @ endpoint_rank)
    predictor_residual = predictor_rank - design @ (projection @ predictor_rank)
    endpoint_ss = float(endpoint_residual @ endpoint_residual)
    predictor_ss = float(predictor_residual @ predictor_residual)
    if endpoint_ss <= 0 or predictor_ss <= 0:
        raise ValueError("partial QAP has zero residual variance")
    observed = float(
        predictor_residual @ endpoint_residual
        / np.sqrt(predictor_ss * endpoint_ss)
    )

    null = np.empty(len(permutation_maps), dtype=np.float64)
    for start in range(0, len(permutation_maps), chunk_size):
        selected = permutation_maps[start : start + chunk_size]
        permuted = predictor_rank[selected]
        residual = permuted - (design @ (projection @ permuted.T)).T
        denominator = np.sqrt(np.sum(residual**2, axis=1) * endpoint_ss)
        null[start : start + len(selected)] = (
            residual @ endpoint_residual
        ) / denominator

    exceedances = int(np.sum(null >= observed))
    null_median = float(np.median(null))
    if exact_enumeration:
        p_positive = exceedances / len(null)
        mc_standard_error = 0.0
    else:
        p_positive = (1 + exceedances) / (len(null) + 1)
        mc_standard_error = math.sqrt(
            p_positive * (1 - p_positive) / (len(null) + 1)
        )
    one_sided_critical = float(np.quantile(null, 0.95))
    two_sided_deviation_bound = float(
        np.quantile(np.abs(null - null_median), 0.95)
    )
    return {
        "partial_spearman": observed,
        "restricted_qap_p_positive": float(p_positive),
        "null_mean": float(np.mean(null)),
        "null_median": null_median,
        "null_q05": float(np.quantile(null, 0.05)),
        "null_q95": one_sided_critical,
        "null_q025": float(np.quantile(null, 0.025)),
        "null_q975": float(np.quantile(null, 0.975)),
        "empirical_one_sided_alpha_0_05_critical_partial_spearman": (
            one_sided_critical
        ),
        "observed_minus_one_sided_alpha_0_05_critical": float(
            observed - one_sided_critical
        ),
        "additional_partial_spearman_needed_to_cross_one_sided_critical": float(
            max(0.0, one_sided_critical - observed)
        ),
        "observed_minus_null_median": float(observed - null_median),
        "empirical_two_sided_alpha_0_05_null_deviation_bound": (
            two_sided_deviation_bound
        ),
        "exceedances": exceedances,
        "permutations": int(len(null)),
        "exact_enumeration": bool(exact_enumeration),
        "monte_carlo_standard_error": float(mc_standard_error),
    }


def matrix_from_pairs(frame: pd.DataFrame, column: str) -> np.ndarray:
    return klifs.matrix_from_pairs(frame, column, TARGETS)


def exchangeability_diagnostics(
    annotations: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    groups = annotations.klifs_group.astype(str).to_numpy()
    families = annotations.klifs_family.astype(str).to_numpy()
    group_blocks = ordered_blocks(groups)
    family_blocks = ordered_blocks(families)

    block_rows: list[dict] = []
    for label, members in group_blocks.items():
        block_rows.append(
            {
                "klifs_group": label,
                "targets": ";".join(annotations.target.iloc[members]),
                "target_count": len(members),
                "movable": len(members) > 1,
                "within_group_pair_positions": math.comb(len(members), 2),
                "within_block_label_permutations": math.factorial(len(members)),
            }
        )
    block_frame = pd.DataFrame(block_rows)

    stratum_rows: list[dict] = []
    items = list(group_blocks.items())
    for first_index, (group_a, members_a) in enumerate(items):
        for second_index in range(first_index, len(items)):
            group_b, members_b = items[second_index]
            if first_index == second_index:
                pair_positions = math.comb(len(members_a), 2)
            else:
                pair_positions = len(members_a) * len(members_b)
            if pair_positions == 0:
                continue
            stratum_rows.append(
                {
                    "klifs_group_a": group_a,
                    "klifs_group_b": group_b,
                    "pair_positions": pair_positions,
                    "candidate_pair_values_per_position": pair_positions,
                    "pair_positions_can_change": (
                        pair_positions > 1
                        if first_index == second_index
                        else len(members_a) > 1 or len(members_b) > 1
                    ),
                }
            )
    stratum_frame = pd.DataFrame(stratum_rows)

    group_total = math.prod(
        math.factorial(len(members)) for members in group_blocks.values()
    )
    family_total = math.prod(
        math.factorial(len(members)) for members in family_blocks.values()
    )
    movable_targets = int(sum(len(m) for m in group_blocks.values() if len(m) > 1))
    total_pairs = len(TARGETS) * (len(TARGETS) - 1) // 2
    fixed_pair_positions = int(
        stratum_frame.loc[
            ~stratum_frame.pair_positions_can_change, "pair_positions"
        ].sum()
    )
    tk_members = group_blocks.get("TK", np.asarray([], dtype=np.int16))
    tk_involving_pairs = (
        math.comb(len(tk_members), 2)
        + len(tk_members) * (len(TARGETS) - len(tk_members))
    )
    diagnostics = {
        "exact_KLIFS_group_relabellings": int(group_total),
        "movable_targets": movable_targets,
        "movable_target_blocks": int(
            sum(len(members) > 1 for members in group_blocks.values())
        ),
        "movable_target_block_sizes": [
            int(len(members))
            for members in group_blocks.values()
            if len(members) > 1
        ],
        "fixed_singleton_targets": int(len(TARGETS) - movable_targets),
        "movable_pair_positions": int(total_pairs - fixed_pair_positions),
        "fixed_singleton_to_singleton_pair_positions": fixed_pair_positions,
        "TK_involving_pair_positions": int(tk_involving_pairs),
        "TK_involving_pair_fraction": float(tk_involving_pairs / total_pairs),
        "exact_KLIFS_family_relabellings": int(family_total),
        "minimum_exact_one_sided_family_p": float(1 / family_total),
        "non_singleton_families": {
            label: annotations.target.iloc[members].tolist()
            for label, members in family_blocks.items()
            if len(members) > 1
        },
        "effective_exchangeable_units": {
            "randomization_unit": "complete target labels within KLIFS-group blocks",
            "movable_target_labels": movable_targets,
            "movable_blocks": int(
                sum(len(members) > 1 for members in group_blocks.values())
            ),
            "block_size_vector": [
                int(len(members))
                for members in group_blocks.values()
                if len(members) > 1
            ],
            "fixed_singletons": int(len(TARGETS) - movable_targets),
            "scalar_effective_sample_size": None,
            "reason_no_scalar_effective_sample_size": (
                "A block-restricted node-label randomization has no defensible "
                "conversion to an equivalent number of independent target-pair "
                "observations: all 190 edges are functions of 20 nodes, relabellings "
                "are coupled within blocks, and the 11-target TK block dominates the "
                "accessible null. Resolution is therefore reported empirically from "
                "the randomization distribution instead."
            ),
        },
    }
    return block_frame, stratum_frame, diagnostics


def run_analysis(
    *,
    target_annotations: Path,
    target_pairs: Path,
    kirhub_workbook: Path,
    output_dir: Path,
    permutations: int,
    seed: int,
) -> dict:
    annotations = load_annotations(target_annotations)
    pairs = load_pairs(target_pairs)
    groups = annotations.klifs_group.astype(str).to_numpy()
    block_frame, stratum_frame, diagnostics = exchangeability_diagnostics(
        annotations
    )

    matrices = {
        "centered_Vina": matrix_from_pairs(
            pairs, "centered_docking_pair_percentile"
        ),
        "old_three_panel_mean": matrix_from_pairs(
            pairs, "mean_experimental_centered_pair_percentile"
        ),
        "receptor_domain_sequence_identity": matrix_from_pairs(
            pairs, "receptor_domain_sequence_identity"
        ),
        "klifs_pocket_identity": matrix_from_pairs(
            pairs, "klifs_pocket_identity"
        ),
    }
    kirhub_frame = kirhub.load_kirhub(kirhub_workbook)
    matrices["KiRHub"] = kirhub.correlation_geometry(
        kirhub_frame[list(TARGETS)].to_numpy(dtype=np.float64),
        "two_way_center",
    )

    full_orders = restricted_target_orders(groups, permutations, seed)
    tk_orders = restricted_target_orders(
        groups,
        permutations,
        seed + 1,
        active_blocks=("TK",),
    )
    small_orders = exact_restricted_target_orders(
        groups,
        active_blocks=("AGC", "CMGC"),
    )
    schemes = {
        "all_non_singleton_KLIFS_groups": (
            full_orders,
            False,
            "primary strict group-restricted null",
        ),
        "TK_only": (
            tk_orders,
            False,
            "diagnostic: only the 11-target TK block is relabelled",
        ),
        "AGC_plus_CMGC_only": (
            small_orders,
            True,
            "diagnostic exact enumeration of the two three-target blocks",
        ),
    }
    control_sets = {
        "none": [],
        "sequence_plus_KLIFS_pocket": [
            matrices["receptor_domain_sequence_identity"],
            matrices["klifs_pocket_identity"],
        ],
    }

    result_rows: list[dict] = []
    scheme_diagnostics: dict[str, dict] = {}
    for scheme_index, (scheme_name, (orders, exact, description)) in enumerate(
        schemes.items()
    ):
        maps = orders_to_pair_maps(orders)
        scheme_diagnostics[scheme_name] = {
            "description": description,
            "permutations": int(len(orders)),
            "exact_enumeration": exact,
            "unique_sampled_target_orders": int(len(np.unique(orders, axis=0))),
            "identity_orders": int(
                np.sum(np.all(orders == np.arange(len(TARGETS)), axis=1))
            ),
            "seed": None if exact else seed + int(scheme_index > 0),
        }
        for endpoint_name in ("old_three_panel_mean", "KiRHub"):
            for control_name, controls in control_sets.items():
                result = restricted_partial_rank_qap(
                    matrices["centered_Vina"],
                    matrices[endpoint_name],
                    controls,
                    maps,
                    exact_enumeration=exact,
                )
                result_rows.append(
                    {
                        "analysis_status": "exploratory_conditional_sensitivity",
                        "predictor": "centered_Vina_target_geometry",
                        "endpoint": endpoint_name,
                        "controls": control_name,
                        "permutation_scheme": scheme_name,
                        **result,
                        "seed": "exact" if exact else scheme_diagnostics[scheme_name]["seed"],
                    }
                )
    results = pd.DataFrame(result_rows)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "exchangeability_blocks.csv": block_frame,
        "pair_strata.csv": stratum_frame,
        "restricted_partial_qap.csv": results,
    }
    for filename, frame in outputs.items():
        frame.to_csv(output_dir / filename, index=False)

    def result_record(endpoint: str, controls: str, scheme: str) -> dict:
        row = results[
            (results.endpoint == endpoint)
            & (results.controls == controls)
            & (results.permutation_scheme == scheme)
        ].iloc[0]
        return {
            "partial_spearman": float(row.partial_spearman),
            "restricted_qap_p_positive": float(row.restricted_qap_p_positive),
            "null_interval_95": [float(row.null_q025), float(row.null_q975)],
            "one_sided_null_interval_90": [
                float(row.null_q05),
                float(row.null_q95),
            ],
            "null_median": float(row.null_median),
            "empirical_one_sided_alpha_0_05_critical_partial_spearman": float(
                row.empirical_one_sided_alpha_0_05_critical_partial_spearman
            ),
            "observed_minus_one_sided_alpha_0_05_critical": float(
                row.observed_minus_one_sided_alpha_0_05_critical
            ),
            "additional_partial_spearman_needed_to_cross_one_sided_critical": float(
                row.additional_partial_spearman_needed_to_cross_one_sided_critical
            ),
            "observed_minus_null_median": float(row.observed_minus_null_median),
            "empirical_two_sided_alpha_0_05_null_deviation_bound": float(
                row.empirical_two_sided_alpha_0_05_null_deviation_bound
            ),
            "monte_carlo_standard_error": float(row.monte_carlo_standard_error),
        }

    primary_scheme = "all_non_singleton_KLIFS_groups"
    summary = {
        "analysis": "strict within-KLIFS-group target-label QAP sensitivity",
        "analysis_status": "exploratory_conditional_sensitivity",
        "fixed_target_panel": list(TARGETS),
        "estimand": (
            "Spearman or rank-linear partial association between the fixed "
            "centered-Vina target-pair geometry and an experimental target-pair "
            "geometry, conditional on complete predictor-node relabellings "
            "within KLIFS groups"
        ),
        "permutation_rule": (
            "Every draw is a complete bijection of all 20 predictor target "
            "labels. Labels may exchange only with targets in the same KLIFS "
            "group; the experimental endpoint and structural controls remain fixed."
        ),
        "validity_boundary": (
            "This is a conditional randomization sensitivity under within-group "
            "target exchangeability, not a proof of exchangeability or a causal "
            "test. Unequal group sizes make the null predominantly a TK-block "
            "test. Rank-linear sequence and pocket adjustment reduces measured "
            "continuous structural similarity but cannot repair violations of "
            "the exchangeability assumption."
        ),
        "small_group_boundary": (
            "AGC and CMGC each contain only three targets (six label orders each), "
            "and three groups are singletons. An exact AGC+CMGC-only null has just "
            "36 orders. A stricter within-KLIFS-family null has only 16 total "
            "orders and cannot attain a one-sided p below 0.0625, so it is not "
            "used for significance claims."
        ),
        "resolution_boundary": (
            "The empirical critical values are randomization-resolution diagnostics "
            "for the observed fixed matrices, not power-based minimum detectable "
            "effects and not confidence intervals for a target population. A single "
            "effective target count is intentionally not reported because node-label "
            "permutations couple all incident edges and are restricted to blocks of "
            "sizes 11, 3 and 3, with three fixed singleton targets. The 95th "
            "percentile is calibrated to the primary one-sided positive alternative; "
            "the absolute-deviation bound around the null median is descriptive and "
            "is not reported as a two-sided p-value."
        ),
        "exchangeability_diagnostics": diagnostics,
        "permutation_schemes": scheme_diagnostics,
        "primary_results": {
            "old_three_panel_mean": {
                "unadjusted": result_record(
                    "old_three_panel_mean", "none", primary_scheme
                ),
                "sequence_plus_KLIFS_pocket_adjusted": result_record(
                    "old_three_panel_mean",
                    "sequence_plus_KLIFS_pocket",
                    primary_scheme,
                ),
            },
            "KiRHub": {
                "unadjusted": result_record("KiRHub", "none", primary_scheme),
                "sequence_plus_KLIFS_pocket_adjusted": result_record(
                    "KiRHub", "sequence_plus_KLIFS_pocket", primary_scheme
                ),
            },
        },
        "claim_boundary": (
            "A small conditional p-value would show that the observed continuous "
            "association is unusual under coarse-group-preserving relabellings; "
            "it would not show independence from all kinase biology, superiority "
            "to sequence or pocket baselines, pose correctness, or validity beyond "
            "this fixed 20-target panel."
        ),
        "configuration": {
            "monte_carlo_permutations": int(permutations),
            "seed": int(seed),
            "alternative": "positive association, one-sided",
            "monte_carlo_p_correction": "plus one",
            "exact_p_rule": "exceedance fraction over the complete enumeration",
        },
        "sources": {
            "target_annotations": {
                "path": release_provenance_path(target_annotations),
                "sha256": sha256_file(target_annotations),
            },
            "target_pairs": {
                "path": release_provenance_path(target_pairs),
                "sha256": sha256_file(target_pairs),
            },
            "KiRHub": {
                "doi": kirhub.KIRHUB_DOI,
                "url": kirhub.KIRHUB_URL,
                "sha256": sha256_file(kirhub_workbook),
                "source_activity_rows_redistributed": False,
                "derived_pair_values_redistributed": False,
            },
        },
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "script_sha256": sha256_file(Path(__file__)),
        },
        "output_files": {
            filename: {
                "rows": int(len(frame)),
                "sha256": sha256_file(output_dir / filename),
            }
            for filename, frame in outputs.items()
        },
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target-annotations", type=Path, default=DEFAULT_TARGET_ANNOTATIONS
    )
    parser.add_argument("--target-pairs", type=Path, default=DEFAULT_TARGET_PAIRS)
    parser.add_argument("--kirhub-workbook", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--permutations", type=int, default=DEFAULT_MONTE_CARLO_PERMUTATIONS
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_analysis(
        target_annotations=args.target_annotations,
        target_pairs=args.target_pairs,
        kirhub_workbook=args.kirhub_workbook,
        output_dir=args.output_dir,
        permutations=args.permutations,
        seed=args.seed,
    )
    old = summary["primary_results"]["old_three_panel_mean"]
    external = summary["primary_results"]["KiRHub"]
    print(
        "Strict within-KLIFS-group QAP complete: "
        f"old-panel rho={old['unadjusted']['partial_spearman']:.3f}, "
        f"p={old['unadjusted']['restricted_qap_p_positive']:.4g}; "
        f"KiRHub rho={external['unadjusted']['partial_spearman']:.3f}, "
        f"p={external['unadjusted']['restricted_qap_p_positive']:.4g}."
    )


if __name__ == "__main__":
    main()
