#!/usr/bin/env python3
"""Locked external validation with the 2026 KiRHub wild-type kinase panel.

This analysis was created after the three-panel DAVIS/PKIS2/PKIS1 endpoint had
already been defined.  Its primary question is therefore deliberately locked:
does the independently sourced KiRHub target-correlation geometry recover the
target pairs that were in the centered upper 10% in at least two of those three
older panels?  No KiRHub result is used to retune that endpoint.

Secondary analyses ask whether DOCKSTRING raw or two-way-centered geometry
retrieves KiRHub-only and four-panel consensus pairs.  These are explicitly
exploratory.  In particular, the KiRHub-only endpoint is a useful negative
boundary when centered docking does not outperform raw docking.

The source workbook is read from a user-supplied path, checksum-validated, and
never copied or exported.  Only aggregate statistics are written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
import warnings
from collections import OrderedDict
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats

try:  # Direct script execution and package-style import are both supported.
    from . import dense_davis_benchmark as davis
    from . import replicated_pair_retrieval as retrieval
    from . import residual_target_geometry_validation as geometry
except ImportError:  # pragma: no cover - exercised by direct CLI execution.
    import dense_davis_benchmark as davis  # type: ignore
    import replicated_pair_retrieval as retrieval  # type: ignore
    import residual_target_geometry_validation as geometry  # type: ignore


PACKAGE = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PACKAGE / "results" / "kirhub_external_validation"
DEFAULT_SEQUENCE_PAIRS = (
    PACKAGE / "results" / "replicated_pair_retrieval" / "target_pairs.csv"
)
DEFAULT_SEQUENCE_SUMMARY = (
    PACKAGE / "results" / "replicated_pair_retrieval" / "summary.json"
)
DEFAULT_SEED = 20260804

KIRHUB_DOI = "10.1038/s41587-026-03090-8"
KIRHUB_URL = (
    "https://media.springernature.com/original/springer-static/esm/"
    "art%3A10.1038%2Fs41587-026-03090-8/"
    "MediaObjects/41587_2026_3090_MOESM4_ESM.xlsx"
)
KIRHUB_SHA256 = "d9eef358396b193834b0c4d48ccd8cadb43697a6458cdcb50415af5a7e4e0b03"
KIRHUB_SHEET = "Table S4"
KIRHUB_HEADER_ROW_ZERO_BASED = 8
KIRHUB_ARTICLE_LICENSE = (
    "Creative Commons Attribution-NonCommercial-NoDerivatives 4.0"
)
KIRHUB_ARTICLE_LICENSE_URL = "https://creativecommons.org/licenses/by-nc-nd/4.0/"
KIRHUB_PORTAL_URL = "https://kirhub.fredhutch.org/"
KIRHUB_PORTAL_RIGHTS_NOTICE = (
    "Kinase inhibition data are the property of Reaction Biology Corporation; "
    "proper acknowledgement is required for download, use or publication."
)
# Compatibility aliases for exploratory modules that import these names.  The
# value is deliberately scoped to the article rather than asserted for the data.
KIRHUB_LICENSE = f"article license: {KIRHUB_ARTICLE_LICENSE}"
KIRHUB_LICENSE_URL = KIRHUB_ARTICLE_LICENSE_URL

TARGET_MAP: OrderedDict[str, str] = OrderedDict(
    [
        ("ABL1", "ABL1"),
        ("AKT1", "AKT1"),
        ("AKT2", "AKT2"),
        ("CDK2", "CDK2_CYCLIN_A"),
        ("CSF1R", "FMS"),
        ("EGFR", "EGFR"),
        ("FGFR1", "FGFR1"),
        ("IGF1R", "IGF1R"),
        ("JAK2", "JAK2"),
        ("KDR", "KDR_VEGFR2"),
        ("KIT", "C_KIT"),
        ("LCK", "LCK"),
        ("MAP2K1", "MEK1"),
        ("MAPK1", "ERK2_MAPK1"),
        ("MAPK14", "P38A_MAPK14"),
        ("MAPKAPK2", "MAPKAPK2"),
        ("MET", "C_MET"),
        ("PLK1", "PLK1"),
        ("ROCK1", "ROCK1"),
        ("SRC", "C_SRC"),
    ]
)

TARGETS = tuple(TARGET_MAP)
if TARGETS != tuple(geometry.PKIS1_TARGET_MAP):
    raise RuntimeError("KiRHub targets no longer align with the locked 20-target endpoint")

PRIMARY_FRACTION = 0.10
PRIMARY_MINIMUM_OLD_PANELS = 2
FOUR_PANEL_MINIMUM = 2
THRESHOLD_GRID = (0.05, 0.075, 0.10, 0.125, 0.15, 0.20, 0.25)

TRANSFORMS: OrderedDict[str, str] = OrderedDict(
    [
        ("raw", "no explicit preprocessing before target correlation"),
        (
            "column_center_only",
            "remove target offsets only; correlation is mathematically invariant",
        ),
        (
            "row_center_only",
            "remove the per-ligand mean; target offsets are left to correlation",
        ),
        (
            "two_way_center",
            "remove ligand and target additive effects",
        ),
        (
            "z_before_center",
            "column z-score, then remove ligand and target additive effects",
        ),
        (
            "rank_normal_before_center",
            "column inverse-normal ranks, then remove additive effects",
        ),
    ]
)


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_drug_name(value: object) -> str:
    """Normalize a drug name for an exact, punctuation-insensitive comparison."""
    normalized = unicodedata.normalize("NFKC", str(value)).casefold()
    return " ".join(re.sub(r"[^a-z0-9]+", " ", normalized).split())


def transformed_surface(matrix: np.ndarray, transform: str) -> np.ndarray:
    x = np.asarray(matrix, dtype=np.float64)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError("transform requires a finite two-dimensional matrix")
    if transform == "raw":
        return x
    if transform == "column_center_only":
        return x - x.mean(axis=0, keepdims=True)
    if transform == "row_center_only":
        return x - x.mean(axis=1, keepdims=True)
    if transform == "two_way_center":
        return geometry.two_way_center(x)
    if transform == "z_before_center":
        return geometry.two_way_center(geometry.column_zscore(x))
    if transform == "rank_normal_before_center":
        return geometry.two_way_center(geometry.column_rank_normal_scores(x))
    raise ValueError(f"unknown transform: {transform}")


def correlation_geometry(matrix: np.ndarray, transform: str) -> np.ndarray:
    return geometry.target_correlation(transformed_surface(matrix, transform))


def load_kirhub(
    workbook: Path,
    *,
    cdk2_construct: str = "cyclin_A",
) -> pd.DataFrame:
    """Load the dense 92 x 20 KiRHub residual-activity block.

    The returned table contains compound names only to support an exact-name
    independence audit.  No row-level values are written to the results package.
    """
    path = Path(workbook)
    observed_sha = sha256_file(path)
    if observed_sha != KIRHUB_SHA256:
        raise ValueError(
            "KiRHub workbook SHA256 mismatch: "
            f"expected {KIRHUB_SHA256}, observed {observed_sha}"
        )
    if cdk2_construct not in {"cyclin_A", "cyclin_E"}:
        raise ValueError("CDK2 construct must be cyclin_A or cyclin_E")
    target_map = TARGET_MAP.copy()
    target_map["CDK2"] = (
        "CDK2_CYCLIN_A" if cdk2_construct == "cyclin_A" else "CDK2_CYCLIN_E"
    )
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Unknown extension is not supported and will be removed",
            category=UserWarning,
        )
        source = pd.read_excel(
            path,
            sheet_name=KIRHUB_SHEET,
            header=KIRHUB_HEADER_ROW_ZERO_BASED,
        )
    if source.shape != (92, 410):
        raise ValueError(
            f"expected KiRHub Table S4 shape 92 x 410, observed {source.shape}"
        )
    required = ["Compound", *target_map.values()]
    missing = [column for column in required if column not in source]
    if missing:
        raise ValueError(f"KiRHub Table S4 is missing columns: {missing}")
    if source.Compound.isna().any() or source.Compound.duplicated().any():
        raise ValueError("KiRHub compound names are missing or duplicated")
    values = source[list(target_map.values())].apply(pd.to_numeric, errors="coerce")
    values.columns = list(target_map)
    if values.isna().any().any():
        raise ValueError("the selected KiRHub 92 x 20 block is not dense")
    if ((values < 0) | (values > 100)).any().any():
        raise ValueError("KiRHub residual activity lies outside [0, 100]")
    out = values.copy()
    out.insert(0, "Compound", source.Compound.astype(str).to_numpy())
    out.attrs["cdk2_construct"] = cdk2_construct
    return out


def load_old_experimental_panels(pkis1_zip: Path) -> dict[str, np.ndarray]:
    davis_experiment = retrieval._load_davis_experiment()  # noqa: SLF001
    pkis2_frame = geometry.load_pkis2_full()
    pkis1_frame = geometry.load_pkis1_full(pkis1_zip)
    return {
        "DAVIS": davis_experiment[list(TARGETS)].to_numpy(dtype=np.float64),
        "PKIS2": pkis2_frame[list(TARGETS)].to_numpy(dtype=np.float64),
        "PKIS1": pkis1_frame[list(TARGETS)].to_numpy(dtype=np.float64),
    }


def load_docking_geometries(pkis1_zip: Path) -> tuple[dict[str, np.ndarray], dict]:
    """Recreate the chemically de-leaked 20-target DOCKSTRING reference."""
    dockstring, davis_identity, _, _ = davis._load_inputs(  # noqa: SLF001
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
    unclipped = dockstring.loc[keep, list(TARGETS)].to_numpy(dtype=np.float64)
    clipped = np.minimum(unclipped, 0.0)
    if len(clipped) < 259_000:
        raise ValueError("unexpectedly many DOCKSTRING rows were excluded")
    geometries = {
        name: correlation_geometry(clipped, name) for name in TRANSFORMS
    }
    geometries["raw_unclipped"] = correlation_geometry(unclipped, "raw")
    geometries["two_way_center_unclipped"] = correlation_geometry(
        unclipped, "two_way_center"
    )
    return geometries, {
        "complete_dockstring_rows_before_exclusion": int(len(dockstring)),
        "reference_rows_after_old_panel_connectivity_exclusion": int(len(clipped)),
        "excluded_connectivity_blocks": int(len(excluded_blocks)),
        "positive_scores_clipped_to_zero_primary": True,
        "kirhub_rows_excluded_from_reference": None,
        "kirhub_exclusion_boundary": (
            "The KiRHub supplement provides names but no structures; its remaining "
            "compounds could not be excluded by molecular identity."
        ),
    }


def mean_rank_geometry(panel_ranks: dict[str, np.ndarray], targets: int) -> np.ndarray:
    expected = targets * (targets - 1) // 2
    if not panel_ranks or any(len(values) != expected for values in panel_ranks.values()):
        raise ValueError("panel ranks do not match the target-pair support")
    values = np.mean(np.vstack(list(panel_ranks.values())), axis=0)
    out = np.eye(targets, dtype=np.float64)
    tri = np.triu_indices(targets, k=1)
    out[tri] = values
    out[(tri[1], tri[0])] = values
    return out


def load_sequence_identity(path: Path) -> np.ndarray:
    frame = pd.read_csv(path)
    required = {"target_a", "target_b", "receptor_domain_sequence_identity"}
    if not required.issubset(frame):
        raise ValueError("sequence-pair artifact lacks required columns")
    expected = len(TARGETS) * (len(TARGETS) - 1) // 2
    if len(frame) != expected:
        raise ValueError("sequence-pair artifact does not contain all 190 pairs")
    index = {target: position for position, target in enumerate(TARGETS)}
    matrix = np.eye(len(TARGETS), dtype=np.float64)
    seen: set[tuple[int, int]] = set()
    for row in frame.itertuples(index=False):
        if row.target_a not in index or row.target_b not in index:
            raise ValueError("sequence-pair artifact has an unexpected target")
        first, second = sorted((index[row.target_a], index[row.target_b]))
        if first == second or (first, second) in seen:
            raise ValueError("sequence-pair artifact has a duplicate/diagonal pair")
        seen.add((first, second))
        value = float(row.receptor_domain_sequence_identity)
        if not 0 <= value <= 1:
            raise ValueError("sequence identity lies outside [0,1]")
        matrix[first, second] = matrix[second, first] = value
    if len(seen) != expected:
        raise ValueError("sequence-pair artifact is incomplete")
    return matrix


def _linear_residual(values: np.ndarray, covariate: np.ndarray) -> np.ndarray:
    design = np.column_stack([np.ones(len(covariate)), covariate])
    coefficients = np.linalg.lstsq(design, values, rcond=None)[0]
    return values - design @ coefficients


def partial_geometry_qap(
    predictor: np.ndarray,
    endpoint: np.ndarray,
    sequence_identity: np.ndarray,
    permutations: int,
    seed: int,
) -> dict:
    """Partial Spearman QAP, permuting whole target labels of the predictor."""
    p = len(predictor)
    if any(matrix.shape != (p, p) for matrix in (endpoint, sequence_identity)):
        raise ValueError("partial QAP matrices must be aligned and square")
    tri = np.triu_indices(p, k=1)
    endpoint_rank = retrieval.percentile_ranks(endpoint[tri])
    sequence_rank = retrieval.percentile_ranks(sequence_identity[tri])
    endpoint_residual = _linear_residual(endpoint_rank, sequence_rank)

    def statistic(matrix: np.ndarray) -> float:
        predictor_rank = retrieval.percentile_ranks(matrix[tri])
        predictor_residual = _linear_residual(predictor_rank, sequence_rank)
        return float(np.corrcoef(predictor_residual, endpoint_residual)[0, 1])

    observed = statistic(predictor)
    rng = np.random.default_rng(seed)
    null = np.empty(permutations, dtype=np.float64)
    for repetition in range(permutations):
        order = rng.permutation(p)
        null[repetition] = statistic(predictor[np.ix_(order, order)])
    return {
        "partial_spearman": observed,
        "target_label_qap_permutations": int(permutations),
        "target_label_qap_p_positive": float(
            (1 + np.sum(null >= observed)) / (permutations + 1)
        ),
        "null_interval_95": [
            float(np.quantile(null, 0.025)),
            float(np.quantile(null, 0.975)),
        ],
        "seed": int(seed),
    }


def subset_pair_vector(values: np.ndarray, keep_targets: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    full_p = len(keep_targets)
    expected = full_p * (full_p - 1) // 2
    if len(values) != expected:
        raise ValueError("pair vector and target mask do not align")
    full_pairs = list(zip(*np.triu_indices(full_p, k=1)))
    keep = np.asarray([keep_targets[i] and keep_targets[j] for i, j in full_pairs])
    return values[keep]


def target_delete_one(
    old_labels: np.ndarray,
    four_panel_labels: np.ndarray,
    old_mean_geometry: np.ndarray,
    kirhub_raw: np.ndarray,
    kirhub_centered: np.ndarray,
    docking_raw: np.ndarray,
    docking_centered: np.ndarray,
) -> pd.DataFrame:
    rows: list[dict] = []
    for omitted, target in enumerate(TARGETS):
        keep = np.ones(len(TARGETS), dtype=bool)
        keep[omitted] = False
        old_subset = subset_pair_vector(old_labels, keep)
        four_subset = subset_pair_vector(four_panel_labels, keep)
        matrices = {
            "old_mean": old_mean_geometry[np.ix_(keep, keep)],
            "kirhub_raw": kirhub_raw[np.ix_(keep, keep)],
            "kirhub_centered": kirhub_centered[np.ix_(keep, keep)],
            "dock_raw": docking_raw[np.ix_(keep, keep)],
            "dock_centered": docking_centered[np.ix_(keep, keep)],
        }
        kir_raw = retrieval.retrieval_metrics(
            old_subset, geometry.upper_triangle(matrices["kirhub_raw"])
        )
        kir_center = retrieval.retrieval_metrics(
            old_subset, geometry.upper_triangle(matrices["kirhub_centered"])
        )
        dock_raw = retrieval.retrieval_metrics(
            four_subset, geometry.upper_triangle(matrices["dock_raw"])
        )
        dock_center = retrieval.retrieval_metrics(
            four_subset, geometry.upper_triangle(matrices["dock_centered"])
        )
        rows.append(
            {
                "omitted_target": target,
                "old_endpoint_positive_pairs": int(old_subset.sum()),
                "kirhub_centered_vs_old_mean_geometry_spearman": (
                    geometry.geometry_concordance(
                        matrices["kirhub_centered"], matrices["old_mean"]
                    )
                ),
                "kirhub_old_endpoint_raw_roc_auc": kir_raw["roc_auc"],
                "kirhub_old_endpoint_centered_roc_auc": kir_center["roc_auc"],
                "kirhub_old_endpoint_centered_minus_raw_roc_auc": (
                    kir_center["roc_auc"] - kir_raw["roc_auc"]
                ),
                "kirhub_old_endpoint_raw_average_precision": kir_raw[
                    "average_precision"
                ],
                "kirhub_old_endpoint_centered_average_precision": kir_center[
                    "average_precision"
                ],
                "kirhub_old_endpoint_centered_minus_raw_average_precision": (
                    kir_center["average_precision"] - kir_raw["average_precision"]
                ),
                "four_panel_positive_pairs": int(four_subset.sum()),
                "four_panel_raw_docking_roc_auc": dock_raw["roc_auc"],
                "four_panel_centered_docking_roc_auc": dock_center["roc_auc"],
                "four_panel_centered_minus_raw_docking_roc_auc": (
                    dock_center["roc_auc"] - dock_raw["roc_auc"]
                ),
                "four_panel_raw_docking_average_precision": dock_raw[
                    "average_precision"
                ],
                "four_panel_centered_docking_average_precision": dock_center[
                    "average_precision"
                ],
                "four_panel_centered_minus_raw_docking_average_precision": (
                    dock_center["average_precision"] - dock_raw["average_precision"]
                ),
            }
        )
    return pd.DataFrame(rows)


def metric_record(labels: np.ndarray, raw: np.ndarray, centered: np.ndarray) -> dict:
    return retrieval.endpoint_metrics(labels, raw, centered)


def relabel_experimental_geometry_metrics(record: dict) -> dict:
    """Remove docking-specific labels from a KiRHub experimental predictor."""
    return {
        "positive_pairs": record["positive_pairs"],
        "total_pairs": record["total_pairs"],
        "prevalence": record["prevalence"],
        "raw_KiRHub_geometry": record["raw_docking"],
        "centered_KiRHub_geometry": record["centered_docking"],
        "centered_minus_raw_KiRHub_geometry": record["centered_minus_raw"],
    }


def relabel_experimental_geometry_qap(record: dict) -> dict:
    """Relabel fixed-endpoint QAP output when KiRHub geometry is the predictor."""
    metrics: dict[str, dict] = {}
    for metric, values in record["metrics"].items():
        metrics[metric] = {
            "raw_KiRHub_geometry": values["raw_docking"],
            "raw_KiRHub_geometry_target_label_qap_p": values[
                "raw_target_label_qap_p"
            ],
            "centered_KiRHub_geometry": values["centered_docking"],
            "centered_KiRHub_geometry_target_label_qap_p": values[
                "centered_target_label_qap_p"
            ],
            "centered_minus_raw_KiRHub_geometry": values["centered_minus_raw"],
            "paired_target_label_qap_p_positive_KiRHub_geometry_gain": values[
                "paired_target_label_qap_p_positive_gain"
            ],
            "paired_delta_null_interval_95": values[
                "paired_delta_null_interval_95"
            ],
        }
    return {
        "permutations": record["permutations"],
        "seed": record["seed"],
        "metrics": metrics,
    }


def old_endpoint_validation(
    matrix: np.ndarray,
    old_labels: np.ndarray,
    old_mean_geometry: np.ndarray,
) -> dict:
    raw = correlation_geometry(matrix, "raw")
    centered = correlation_geometry(matrix, "two_way_center")
    return {
        "geometry_spearman_to_old_three_panel_mean": geometry.geometry_concordance(
            centered, old_mean_geometry
        ),
        "retrieval": relabel_experimental_geometry_metrics(
            metric_record(old_labels, raw, centered)
        ),
    }


def qap_geometry_rows(
    kirhub_geometry: np.ndarray,
    old_panel_geometries: dict[str, np.ndarray],
    old_mean_geometry: np.ndarray,
    sequence_identity: np.ndarray,
    permutations: int,
    seed: int,
) -> tuple[pd.DataFrame, dict]:
    rows: list[dict] = []
    partial: dict[str, dict] = {}
    endpoints = {**old_panel_geometries, "old_three_panel_mean_rank": old_mean_geometry}
    for index, (name, endpoint) in enumerate(endpoints.items()):
        qap = geometry.qap_test(
            kirhub_geometry,
            endpoint,
            permutations,
            seed + index * 10_000,
        )
        conditioned = partial_geometry_qap(
            kirhub_geometry,
            endpoint,
            sequence_identity,
            permutations,
            seed + 100_000 + index * 10_000,
        )
        rows.append(
            {
                "endpoint": name,
                "spearman": qap["observed_spearman"],
                "target_label_qap_p_positive": qap["one_sided_p_positive"],
                "partial_spearman_controlling_receptor_sequence_identity": (
                    conditioned["partial_spearman"]
                ),
                "partial_target_label_qap_p_positive": conditioned[
                    "target_label_qap_p_positive"
                ],
                "permutations": permutations,
            }
        )
        partial[name] = conditioned
    return pd.DataFrame(rows), partial


def transform_sensitivity(
    kirhub_matrix: np.ndarray,
    old_labels: np.ndarray,
    old_mean_geometry: np.ndarray,
    kirhub_specific_labels: np.ndarray,
    four_panel_labels: np.ndarray,
    docking_geometries: dict[str, np.ndarray],
) -> pd.DataFrame:
    rows: list[dict] = []
    for name, description in TRANSFORMS.items():
        kir = correlation_geometry(kirhub_matrix, name)
        old_metrics = retrieval.retrieval_metrics(
            old_labels, geometry.upper_triangle(kir)
        )
        rows.append(
            {
                "sensitivity_family": "KiRHub_experimental_transform",
                "transform": name,
                "description": description,
                "old_endpoint_roc_auc": old_metrics["roc_auc"],
                "old_endpoint_average_precision": old_metrics["average_precision"],
                "old_mean_geometry_spearman": geometry.geometry_concordance(
                    kir, old_mean_geometry
                ),
            }
        )
    for name, dock in docking_geometries.items():
        kirhub_only = retrieval.retrieval_metrics(
            kirhub_specific_labels, geometry.upper_triangle(dock)
        )
        four_panel = retrieval.retrieval_metrics(
            four_panel_labels, geometry.upper_triangle(dock)
        )
        rows.append(
            {
                "sensitivity_family": "DOCKSTRING_predictor_transform",
                "transform": name,
                "description": TRANSFORMS.get(
                    name, "unclipped-score sensitivity using the named transform"
                ),
                "kirhub_specific_roc_auc": kirhub_only["roc_auc"],
                "kirhub_specific_average_precision": kirhub_only[
                    "average_precision"
                ],
                "four_panel_roc_auc": four_panel["roc_auc"],
                "four_panel_average_precision": four_panel["average_precision"],
            }
        )
    return pd.DataFrame(rows)


def threshold_sensitivity(
    old_panel_ranks: dict[str, np.ndarray],
    kirhub_rank: np.ndarray,
    raw_docking: np.ndarray,
    centered_docking: np.ndarray,
) -> pd.DataFrame:
    rows: list[dict] = []
    four_ranks = {**old_panel_ranks, "KiRHub": kirhub_rank}
    for fraction in THRESHOLD_GRID:
        for minimum_panels in (2, 3):
            labels, _ = retrieval.replicated_labels(
                four_ranks, fraction, minimum_panels=minimum_panels
            )
            if labels.sum() < 1 or (~labels).sum() < 1:
                continue
            metrics = metric_record(labels, raw_docking, centered_docking)
            rows.append(
                {
                    "upper_tail_fraction": fraction,
                    "minimum_panels_of_four": minimum_panels,
                    "positive_pairs": metrics["positive_pairs"],
                    "raw_roc_auc": metrics["raw_docking"]["roc_auc"],
                    "centered_roc_auc": metrics["centered_docking"]["roc_auc"],
                    "centered_minus_raw_roc_auc": metrics["centered_minus_raw"][
                        "roc_auc"
                    ],
                    "raw_average_precision": metrics["raw_docking"][
                        "average_precision"
                    ],
                    "centered_average_precision": metrics["centered_docking"][
                        "average_precision"
                    ],
                    "centered_minus_raw_average_precision": metrics[
                        "centered_minus_raw"
                    ]["average_precision"],
                    "inferential_status": "descriptive_post_hoc_threshold_sensitivity",
                }
            )
    return pd.DataFrame(rows)


def pairwise_panel_alignment(
    panel_geometries: dict[str, np.ndarray],
) -> tuple[pd.DataFrame, float]:
    """All pairwise panel-geometry Spearman correlations and their unweighted mean."""
    names = tuple(panel_geometries)
    if len(names) < 2:
        raise ValueError("at least two panels are required")
    shape = panel_geometries[names[0]].shape
    if any(matrix.shape != shape for matrix in panel_geometries.values()):
        raise ValueError("panel geometries must have aligned target supports")
    rows: list[dict] = []
    values: list[float] = []
    for first_index, first in enumerate(names):
        for second in names[first_index + 1 :]:
            value = geometry.geometry_concordance(
                panel_geometries[first], panel_geometries[second]
            )
            rows.append({"panel_a": first, "panel_b": second, "spearman": value})
            values.append(value)
    return pd.DataFrame(rows), float(np.mean(values))


def cross_panel_transform_sensitivity(
    panel_matrices: dict[str, np.ndarray],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cross-panel target-geometry reproducibility and PR across transformations."""
    alignment_rows: list[dict] = []
    pr_rows: list[dict] = []
    for transform, description in TRANSFORMS.items():
        geometries = {
            panel: correlation_geometry(matrix, transform)
            for panel, matrix in panel_matrices.items()
        }
        pairs, mean_value = pairwise_panel_alignment(geometries)
        for row in pairs.itertuples(index=False):
            alignment_rows.append(
                {
                    "transform": transform,
                    "description": description,
                    "panel_a": row.panel_a,
                    "panel_b": row.panel_b,
                    "spearman": row.spearman,
                    "mean_across_six_panel_pairs": mean_value,
                }
            )
        for panel, correlation in geometries.items():
            pr_rows.append(
                {
                    "transform": transform,
                    "description": description,
                    "panel": panel,
                    "ligands": int(len(panel_matrices[panel])),
                    "targets": int(panel_matrices[panel].shape[1]),
                    "correlation_PR_dimension": retrieval.correlation_pr(correlation),
                }
            )
    return pd.DataFrame(alignment_rows), pd.DataFrame(pr_rows)


def _rank_matrix(matrix: np.ndarray) -> np.ndarray:
    x = np.asarray(matrix, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] != x.shape[1]:
        raise ValueError("rank matrix requires a square matrix")
    tri = np.triu_indices(len(x), k=1)
    ranks = stats.rankdata(x[tri], method="average").astype(np.float64)
    ranks -= ranks.mean()
    out = np.zeros_like(x)
    out[tri] = ranks
    out[(tri[1], tri[0])] = ranks
    return out


def _rank_matrix_concordance(first: np.ndarray, second: np.ndarray) -> float:
    tri = np.triu_indices(len(first), k=1)
    x = first[tri]
    y = second[tri]
    return float(np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y)))


def _mean_ranked_panel_alignment(panel_rank_matrices: dict[str, np.ndarray]) -> float:
    names = tuple(panel_rank_matrices)
    values: list[float] = []
    for first_index, first in enumerate(names):
        for second in names[first_index + 1 :]:
            values.append(
                _rank_matrix_concordance(
                    panel_rank_matrices[first], panel_rank_matrices[second]
                )
            )
    return float(np.mean(values))


def paired_global_network_qap(
    raw_geometries: dict[str, np.ndarray],
    centered_geometries: dict[str, np.ndarray],
    permutations: int,
    seed: int,
) -> dict:
    """Paired global QAP for mean cross-panel alignment.

    The first panel's labels are fixed to define the coordinate system.  In every
    null draw, each of the other panels receives an independent whole-target
    permutation.  The same panel-specific permutations are applied to that
    panel's raw and centered geometries, preserving pairing of the two estimands.
    """
    names = tuple(raw_geometries)
    if names != tuple(centered_geometries) or len(names) < 3:
        raise ValueError("raw and centered panel dictionaries must align")
    p = len(raw_geometries[names[0]])
    if any(matrix.shape != (p, p) for matrix in (*raw_geometries.values(), *centered_geometries.values())):
        raise ValueError("global QAP geometries must be aligned square matrices")
    raw_ranks = {name: _rank_matrix(raw_geometries[name]) for name in names}
    centered_ranks = {
        name: _rank_matrix(centered_geometries[name]) for name in names
    }
    observed_raw = _mean_ranked_panel_alignment(raw_ranks)
    observed_centered = _mean_ranked_panel_alignment(centered_ranks)
    observed_delta = observed_centered - observed_raw
    null_raw = np.empty(permutations, dtype=np.float64)
    null_centered = np.empty(permutations, dtype=np.float64)
    rng = np.random.default_rng(seed)
    identity = np.arange(p)
    for repetition in range(permutations):
        orders = {names[0]: identity}
        orders.update({name: rng.permutation(p) for name in names[1:]})
        permuted_raw = {
            name: raw_ranks[name][np.ix_(orders[name], orders[name])]
            for name in names
        }
        permuted_centered = {
            name: centered_ranks[name][np.ix_(orders[name], orders[name])]
            for name in names
        }
        null_raw[repetition] = _mean_ranked_panel_alignment(permuted_raw)
        null_centered[repetition] = _mean_ranked_panel_alignment(permuted_centered)
    null_delta = null_centered - null_raw
    return {
        "panels": list(names),
        "panel_pairs": int(len(names) * (len(names) - 1) // 2),
        "raw_mean_pairwise_spearman": observed_raw,
        "centered_mean_pairwise_spearman": observed_centered,
        "centered_minus_raw": observed_delta,
        "raw_global_network_qap_p_positive": float(
            (1 + np.sum(null_raw >= observed_raw)) / (permutations + 1)
        ),
        "centered_global_network_qap_p_positive": float(
            (1 + np.sum(null_centered >= observed_centered)) / (permutations + 1)
        ),
        "paired_global_network_qap_p_positive_gain": float(
            (1 + np.sum(null_delta >= observed_delta)) / (permutations + 1)
        ),
        "paired_delta_null_interval_95": [
            float(np.quantile(null_delta, 0.025)),
            float(np.quantile(null_delta, 0.975)),
        ],
        "permutations": int(permutations),
        "seed": int(seed),
        "null": (
            "DAVIS labels fixed; independent whole-target permutations for PKIS2, "
            "PKIS1, and KiRHub; identical panel permutations used for raw and "
            "centered geometries within each draw"
        ),
    }


def cross_panel_target_jackknife(
    panel_matrices: dict[str, np.ndarray],
) -> pd.DataFrame:
    rows: list[dict] = []
    for omitted, target in enumerate(TARGETS):
        keep = np.ones(len(TARGETS), dtype=bool)
        keep[omitted] = False
        raw = {
            name: correlation_geometry(matrix[:, keep], "raw")
            for name, matrix in panel_matrices.items()
        }
        centered = {
            name: correlation_geometry(matrix[:, keep], "two_way_center")
            for name, matrix in panel_matrices.items()
        }
        raw_pairs, raw_mean = pairwise_panel_alignment(raw)
        centered_pairs, centered_mean = pairwise_panel_alignment(centered)
        merged = raw_pairs.merge(
            centered_pairs,
            on=["panel_a", "panel_b"],
            suffixes=("_raw", "_centered"),
            validate="1:1",
        )
        rows.append(
            {
                "omitted_target": target,
                "raw_mean_pairwise_spearman": raw_mean,
                "centered_mean_pairwise_spearman": centered_mean,
                "centered_minus_raw_mean_pairwise_spearman": centered_mean - raw_mean,
                "panel_pairs_improved": int(
                    (merged.spearman_centered > merged.spearman_raw).sum()
                ),
                "panel_pairs_total": int(len(merged)),
            }
        )
    return pd.DataFrame(rows)


def cross_panel_ligand_bootstrap(
    panel_matrices: dict[str, np.ndarray],
    repeats: int,
    seed: int,
) -> tuple[dict, pd.DataFrame]:
    """Independent ligand bootstrap within each panel, paired across estimands."""
    if repeats < 1:
        raise ValueError("bootstrap repeats must be positive")
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    attempts = 0
    maximum_attempts = max(1000, repeats * 100)
    while len(rows) < repeats and attempts < maximum_attempts:
        attempts += 1
        samples = {
            name: matrix[
                rng.integers(0, len(matrix), size=len(matrix), endpoint=False)
            ]
            for name, matrix in panel_matrices.items()
        }
        try:
            raw = {
                name: correlation_geometry(matrix, "raw")
                for name, matrix in samples.items()
            }
            centered = {
                name: correlation_geometry(matrix, "two_way_center")
                for name, matrix in samples.items()
            }
        except ValueError as error:
            # Sparse DAVIS uncensored support can occasionally yield a constant
            # target after ligand resampling; such a draw has no correlation
            # geometry and is rejected rather than imputed or jittered.
            if "constant target" not in str(error):
                raise
            continue
        _, raw_mean = pairwise_panel_alignment(raw)
        _, centered_mean = pairwise_panel_alignment(centered)
        rows.append(
            {
                "bootstrap_repetition": len(rows),
                "raw_mean_pairwise_spearman": raw_mean,
                "centered_mean_pairwise_spearman": centered_mean,
                "centered_minus_raw": centered_mean - raw_mean,
            }
        )
    if len(rows) != repeats:
        raise RuntimeError(
            f"only {len(rows)} valid bootstrap draws after {attempts} attempts"
        )
    frame = pd.DataFrame(rows)
    delta = frame.centered_minus_raw.to_numpy(dtype=np.float64)
    summary = {
        "resampling_unit": (
            "ligands independently within each panel; the same sampled rows are "
            "used for raw and centered estimands in each panel"
        ),
        "repeats": int(repeats),
        "attempts": int(attempts),
        "rejected_constant_target_draws": int(attempts - repeats),
        "seed": int(seed),
        "centered_minus_raw_mean": float(delta.mean()),
        "centered_minus_raw_interval_95": [
            float(np.quantile(delta, 0.025)),
            float(np.quantile(delta, 0.975)),
        ],
        "fraction_positive": float(np.mean(delta > 0)),
        "cluster_boundary": (
            "KiRHub supplies no molecular structures, so a chemically clustered "
            "four-panel bootstrap was not possible; this is a ligand-level, not "
            "scaffold-level, uncertainty interval."
        ),
    }
    return summary, frame


def transformation_matched_independent_column_null(
    panel_matrices: dict[str, np.ndarray],
    locked_labels: np.ndarray,
    repeats: int,
    seed: int,
) -> tuple[dict, pd.DataFrame]:
    """Null preserving every target marginal, variance, tie and ceiling pattern.

    Ligand identities are independently permuted within every target column of
    every panel.  The same permuted panel is then evaluated raw and after two-way
    centering.  This exposes any cross-panel geometry or apparent centering gain
    created mechanically by stable target-specific marginal distributions.
    """
    if repeats < 1:
        raise ValueError("matched-null repeats must be positive")
    observed_raw_geometries = {
        name: correlation_geometry(matrix, "raw")
        for name, matrix in panel_matrices.items()
    }
    observed_centered_geometries = {
        name: correlation_geometry(matrix, "two_way_center")
        for name, matrix in panel_matrices.items()
    }
    _, observed_raw = pairwise_panel_alignment(observed_raw_geometries)
    _, observed_centered = pairwise_panel_alignment(observed_centered_geometries)
    observed_delta = observed_centered - observed_raw
    if "KiRHub" not in panel_matrices:
        raise ValueError("matched null requires a KiRHub panel")
    observed_locked = metric_record(
        locked_labels,
        observed_raw_geometries["KiRHub"],
        observed_centered_geometries["KiRHub"],
    )
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    for repetition in range(repeats):
        permuted: dict[str, np.ndarray] = {}
        for name, matrix in panel_matrices.items():
            work = np.empty_like(matrix, dtype=np.float64)
            for column in range(matrix.shape[1]):
                work[:, column] = matrix[rng.permutation(len(matrix)), column]
            permuted[name] = work
        raw_geometries = {
            name: correlation_geometry(matrix, "raw")
            for name, matrix in permuted.items()
        }
        centered_geometries = {
            name: correlation_geometry(matrix, "two_way_center")
            for name, matrix in permuted.items()
        }
        _, raw_mean = pairwise_panel_alignment(raw_geometries)
        _, centered_mean = pairwise_panel_alignment(centered_geometries)
        locked = metric_record(
            locked_labels,
            raw_geometries["KiRHub"],
            centered_geometries["KiRHub"],
        )
        rows.append(
            {
                "null_repetition": repetition,
                "raw_mean_pairwise_spearman": raw_mean,
                "centered_mean_pairwise_spearman": centered_mean,
                "centered_minus_raw": centered_mean - raw_mean,
                "locked_endpoint_raw_roc_auc": locked["raw_docking"]["roc_auc"],
                "locked_endpoint_centered_roc_auc": locked["centered_docking"][
                    "roc_auc"
                ],
                "locked_endpoint_centered_minus_raw_roc_auc": locked[
                    "centered_minus_raw"
                ]["roc_auc"],
                "locked_endpoint_raw_average_precision": locked["raw_docking"][
                    "average_precision"
                ],
                "locked_endpoint_centered_average_precision": locked[
                    "centered_docking"
                ]["average_precision"],
                "locked_endpoint_centered_minus_raw_average_precision": locked[
                    "centered_minus_raw"
                ]["average_precision"],
            }
        )
    frame = pd.DataFrame(rows)
    null_raw = frame.raw_mean_pairwise_spearman.to_numpy(dtype=np.float64)
    null_centered = frame.centered_mean_pairwise_spearman.to_numpy(dtype=np.float64)
    null_delta = frame.centered_minus_raw.to_numpy(dtype=np.float64)
    locked_null_columns = {
        metric: {
            estimand: frame[
                f"locked_endpoint_{estimand}_{metric}"
            ].to_numpy(dtype=np.float64)
            for estimand in (
                "raw",
                "centered",
                "centered_minus_raw",
            )
        }
        for metric in ("roc_auc", "average_precision")
    }
    summary = {
        "null": (
            "independently permute ligand identities within every target column "
            "of every experimental panel; preserve target marginals, variances, "
            "ties and ceiling patterns; apply raw and two-way-centered estimands "
            "to the identical permuted matrices"
        ),
        "repeats": int(repeats),
        "seed": int(seed),
        "observed_raw_mean_pairwise_spearman": observed_raw,
        "observed_centered_mean_pairwise_spearman": observed_centered,
        "observed_centered_minus_raw": observed_delta,
        "null_raw_mean": float(null_raw.mean()),
        "null_centered_mean": float(null_centered.mean()),
        "null_centered_minus_raw_mean": float(null_delta.mean()),
        "null_raw_interval_95": [
            float(np.quantile(null_raw, 0.025)),
            float(np.quantile(null_raw, 0.975)),
        ],
        "null_centered_interval_95": [
            float(np.quantile(null_centered, 0.025)),
            float(np.quantile(null_centered, 0.975)),
        ],
        "null_centered_minus_raw_interval_95": [
            float(np.quantile(null_delta, 0.025)),
            float(np.quantile(null_delta, 0.975)),
        ],
        "p_observed_centered_at_least_as_large": float(
            (1 + np.sum(null_centered >= observed_centered)) / (repeats + 1)
        ),
        "p_observed_gain_at_least_as_large": float(
            (1 + np.sum(null_delta >= observed_delta)) / (repeats + 1)
        ),
        "locked_old_endpoint_validation": {
            metric: {
                "observed_raw": observed_locked["raw_docking"][metric],
                "observed_centered": observed_locked["centered_docking"][metric],
                "observed_centered_minus_raw": observed_locked[
                    "centered_minus_raw"
                ][metric],
                "null_raw_mean": float(
                    locked_null_columns[metric]["raw"].mean()
                ),
                "null_centered_mean": float(
                    locked_null_columns[metric]["centered"].mean()
                ),
                "null_centered_minus_raw_mean": float(
                    locked_null_columns[metric]["centered_minus_raw"].mean()
                ),
                "p_observed_centered_at_least_as_large": float(
                    (
                        1
                        + np.sum(
                            locked_null_columns[metric]["centered"]
                            >= observed_locked["centered_docking"][metric]
                        )
                    )
                    / (repeats + 1)
                ),
                "p_observed_gain_at_least_as_large": float(
                    (
                        1
                        + np.sum(
                            locked_null_columns[metric]["centered_minus_raw"]
                            >= observed_locked["centered_minus_raw"][metric]
                        )
                    )
                    / (repeats + 1)
                ),
                "null_centered_interval_95": [
                    float(
                        np.quantile(
                            locked_null_columns[metric]["centered"], 0.025
                        )
                    ),
                    float(
                        np.quantile(
                            locked_null_columns[metric]["centered"], 0.975
                        )
                    ),
                ],
                "null_gain_interval_95": [
                    float(
                        np.quantile(
                            locked_null_columns[metric]["centered_minus_raw"],
                            0.025,
                        )
                    ),
                    float(
                        np.quantile(
                            locked_null_columns[metric]["centered_minus_raw"],
                            0.975,
                        )
                    ),
                ],
            }
            for metric in ("roc_auc", "average_precision")
        },
        "gain_boundary": (
            "A significant centered predictor can coexist with a centered-minus-raw "
            "gain that is not larger than the mechanical gain induced by row "
            "centering target-marginal-preserving independent columns."
        ),
    }
    return summary, frame


def _range(values: Iterable[float]) -> list[float]:
    array = np.asarray(list(values), dtype=np.float64)
    return [float(array.min()), float(array.max())]


def run_analysis(
    *,
    kirhub_workbook: Path,
    pkis1_zip: Path,
    qap_permutations: int,
    bootstrap_repeats: int,
    matched_null_repeats: int,
    seed: int,
    sequence_pairs: Path,
    sequence_summary: Path,
) -> tuple[dict, dict[str, pd.DataFrame]]:
    if qap_permutations < 99:
        raise ValueError("use at least 99 target-label permutations")

    kirhub = load_kirhub(kirhub_workbook, cdk2_construct="cyclin_A")
    kirhub_cyclin_e = load_kirhub(kirhub_workbook, cdk2_construct="cyclin_E")
    old_panels = load_old_experimental_panels(pkis1_zip)
    old_panel_ranks = retrieval.panel_rank_vectors(
        old_panels, "center_then_correlation"
    )
    old_labels, old_panel_labels = retrieval.replicated_labels(
        old_panel_ranks,
        PRIMARY_FRACTION,
        minimum_panels=PRIMARY_MINIMUM_OLD_PANELS,
    )
    old_panel_geometries = {
        name: geometry.geometry_correlation(matrix, "center_then_correlation")
        for name, matrix in old_panels.items()
    }
    old_mean = mean_rank_geometry(old_panel_ranks, len(TARGETS))

    experimental_panel_matrices = {
        **old_panels,
        "KiRHub": kirhub[list(TARGETS)].to_numpy(dtype=np.float64),
    }
    experimental_alignment_frame, experimental_pr_frame = (
        cross_panel_transform_sensitivity(experimental_panel_matrices)
    )
    raw_experimental_geometries = {
        name: correlation_geometry(matrix, "raw")
        for name, matrix in experimental_panel_matrices.items()
    }
    centered_experimental_geometries = {
        name: correlation_geometry(matrix, "two_way_center")
        for name, matrix in experimental_panel_matrices.items()
    }
    global_network_qap = paired_global_network_qap(
        raw_experimental_geometries,
        centered_experimental_geometries,
        qap_permutations,
        seed + 50_000,
    )
    experimental_jackknife_frame = cross_panel_target_jackknife(
        experimental_panel_matrices
    )
    experimental_bootstrap, experimental_bootstrap_frame = (
        cross_panel_ligand_bootstrap(
            experimental_panel_matrices,
            bootstrap_repeats,
            seed + 75_000,
        )
    )
    matched_null, matched_null_frame = (
        transformation_matched_independent_column_null(
            experimental_panel_matrices,
            old_labels,
            matched_null_repeats,
            seed + 90_000,
        )
    )

    sequence_identity = load_sequence_identity(sequence_pairs)
    with Path(sequence_summary).open() as handle:
        sequence_report = json.load(handle)
    sequence_provenance = sequence_report.get(
        "receptor_sequence_identity_control", {}
    ).get("provenance")
    if not sequence_provenance:
        raise ValueError("sequence-summary artifact lacks provenance")

    primary_matrix = kirhub[list(TARGETS)].to_numpy(dtype=np.float64)
    kir_raw = correlation_geometry(primary_matrix, "raw")
    kir_centered = correlation_geometry(primary_matrix, "two_way_center")
    kir_rank = retrieval.percentile_ranks(geometry.upper_triangle(kir_centered))
    locked_validation = metric_record(old_labels, kir_raw, kir_centered)
    locked_validation_output = relabel_experimental_geometry_metrics(
        locked_validation
    )
    locked_qap = retrieval.fixed_endpoint_qap(
        old_labels,
        kir_raw,
        kir_centered,
        qap_permutations,
        seed,
    )
    locked_qap_output = relabel_experimental_geometry_qap(locked_qap)
    locked_holm_centered = retrieval.holm_adjust(
        {
            metric: record["centered_target_label_qap_p"]
            for metric, record in locked_qap["metrics"].items()
        }
    )
    locked_holm_gain = retrieval.holm_adjust(
        {
            metric: record["paired_target_label_qap_p_positive_gain"]
            for metric, record in locked_qap["metrics"].items()
        }
    )

    geometry_qap_frame, partial_controls = qap_geometry_rows(
        kir_centered,
        old_panel_geometries,
        old_mean,
        sequence_identity,
        qap_permutations,
        seed + 200_000,
    )

    # Name-only overlap audit.  DAVIS is the only old panel exposing comparable
    # clinical drug names; PKIS1/PKIS2 use project identifiers in these releases.
    davis_names = pd.read_csv(
        davis.DEFAULT_DAVIS,
        sep="\t",
        usecols=["drug_name"],
    ).drug_name.drop_duplicates()
    davis_normalized = set(davis_names.map(normalize_drug_name))
    overlap_mask = kirhub.Compound.map(normalize_drug_name).isin(davis_normalized)
    overlap_names = sorted(kirhub.loc[overlap_mask, "Compound"].astype(str))
    without_name_overlap = primary_matrix[~overlap_mask.to_numpy()]
    overlap_validation = old_endpoint_validation(
        without_name_overlap, old_labels, old_mean
    )
    overlap_raw = correlation_geometry(without_name_overlap, "raw")
    overlap_centered = correlation_geometry(without_name_overlap, "two_way_center")
    overlap_qap = retrieval.fixed_endpoint_qap(
        old_labels,
        overlap_raw,
        overlap_centered,
        qap_permutations,
        seed + 300_000,
    )
    overlap_qap_output = relabel_experimental_geometry_qap(overlap_qap)
    overlap_partial = partial_geometry_qap(
        overlap_centered,
        old_mean,
        sequence_identity,
        qap_permutations,
        seed + 400_000,
    )
    overlap_panel_matrices = {
        **old_panels,
        "KiRHub": without_name_overlap,
    }
    _, overlap_raw_alignment = pairwise_panel_alignment(
        {
            name: correlation_geometry(matrix, "raw")
            for name, matrix in overlap_panel_matrices.items()
        }
    )
    _, overlap_centered_alignment = pairwise_panel_alignment(
        {
            name: correlation_geometry(matrix, "two_way_center")
            for name, matrix in overlap_panel_matrices.items()
        }
    )

    docking_geometries, docking_provenance = load_docking_geometries(pkis1_zip)
    docking_raw = docking_geometries["raw"]
    docking_centered = docking_geometries["two_way_center"]

    kirhub_specific_labels = retrieval.upper_tail_labels(
        kir_rank, PRIMARY_FRACTION
    )
    kirhub_specific = metric_record(
        kirhub_specific_labels, docking_raw, docking_centered
    )
    kirhub_specific_qap = retrieval.fixed_endpoint_qap(
        kirhub_specific_labels,
        docking_raw,
        docking_centered,
        qap_permutations,
        seed + 500_000,
    )

    four_panel_ranks = {**old_panel_ranks, "KiRHub": kir_rank}
    four_labels, four_panel_individual_labels = retrieval.replicated_labels(
        four_panel_ranks,
        PRIMARY_FRACTION,
        minimum_panels=FOUR_PANEL_MINIMUM,
    )
    four_panel = metric_record(four_labels, docking_raw, docking_centered)
    four_panel_qap = retrieval.fixed_endpoint_qap(
        four_labels,
        docking_raw,
        docking_centered,
        qap_permutations,
        seed + 600_000,
    )
    four_panel_holm_gain = retrieval.holm_adjust(
        {
            metric: record["paired_target_label_qap_p_positive_gain"]
            for metric, record in four_panel_qap["metrics"].items()
        }
    )

    transform_frame = transform_sensitivity(
        primary_matrix,
        old_labels,
        old_mean,
        kirhub_specific_labels,
        four_labels,
        docking_geometries,
    )
    threshold_frame = threshold_sensitivity(
        old_panel_ranks,
        kir_rank,
        docking_raw,
        docking_centered,
    )
    jackknife_frame = target_delete_one(
        old_labels,
        four_labels,
        old_mean,
        kir_raw,
        kir_centered,
        docking_raw,
        docking_centered,
    )

    construct_rows: list[dict] = []
    for construct, frame in (
        ("CDK2_CYCLIN_A", kirhub),
        ("CDK2_CYCLIN_E", kirhub_cyclin_e),
    ):
        matrix = frame[list(TARGETS)].to_numpy(dtype=np.float64)
        centered = correlation_geometry(matrix, "two_way_center")
        rank = retrieval.percentile_ranks(geometry.upper_triangle(centered))
        construct_panels = {
            **old_panels,
            "KiRHub": matrix,
        }
        _, construct_raw_alignment = pairwise_panel_alignment(
            {
                name: correlation_geometry(values, "raw")
                for name, values in construct_panels.items()
            }
        )
        _, construct_centered_alignment = pairwise_panel_alignment(
            {
                name: correlation_geometry(values, "two_way_center")
                for name, values in construct_panels.items()
            }
        )
        validation = retrieval.retrieval_metrics(old_labels, geometry.upper_triangle(centered))
        four_construct_labels, _ = retrieval.replicated_labels(
            {**old_panel_ranks, "KiRHub": rank},
            PRIMARY_FRACTION,
            minimum_panels=FOUR_PANEL_MINIMUM,
        )
        four_metrics = metric_record(
            four_construct_labels, docking_raw, docking_centered
        )
        construct_rows.append(
            {
                "cdk2_construct": construct,
                "old_mean_geometry_spearman": geometry.geometry_concordance(
                    centered, old_mean
                ),
                "old_endpoint_roc_auc": validation["roc_auc"],
                "old_endpoint_average_precision": validation["average_precision"],
                "four_experimental_panels_raw_mean_pairwise_spearman": (
                    construct_raw_alignment
                ),
                "four_experimental_panels_centered_mean_pairwise_spearman": (
                    construct_centered_alignment
                ),
                "four_experimental_panels_centered_minus_raw": (
                    construct_centered_alignment - construct_raw_alignment
                ),
                "four_panel_positive_pairs": int(four_construct_labels.sum()),
                "four_panel_raw_docking_roc_auc": four_metrics["raw_docking"][
                    "roc_auc"
                ],
                "four_panel_centered_docking_roc_auc": four_metrics[
                    "centered_docking"
                ]["roc_auc"],
                "four_panel_centered_minus_raw_docking_roc_auc": four_metrics[
                    "centered_minus_raw"
                ]["roc_auc"],
                "four_panel_raw_docking_average_precision": four_metrics[
                    "raw_docking"
                ]["average_precision"],
                "four_panel_centered_docking_average_precision": four_metrics[
                    "centered_docking"
                ]["average_precision"],
                "four_panel_centered_minus_raw_docking_average_precision": four_metrics[
                    "centered_minus_raw"
                ]["average_precision"],
            }
        )
    construct_frame = pd.DataFrame(construct_rows)

    ceiling_fraction = (primary_matrix == 100.0).mean(axis=0)
    floor_fraction = (primary_matrix == 0.0).mean(axis=0)
    target_quality_frame = pd.DataFrame(
        {
            "target": TARGETS,
            "kirhub_column": list(TARGET_MAP.values()),
            "exact_100_ceiling_cells": (primary_matrix == 100.0).sum(axis=0),
            "exact_100_ceiling_fraction": ceiling_fraction,
            "exact_0_floor_cells": (primary_matrix == 0.0).sum(axis=0),
            "exact_0_floor_fraction": floor_fraction,
            "unique_values": [
                int(np.unique(primary_matrix[:, column]).size)
                for column in range(primary_matrix.shape[1])
            ],
        }
    )

    overlap_frame = pd.DataFrame(
        [
            {
                "support": "all_92_KiRHub_drugs",
                "drugs": len(primary_matrix),
                "removed_exact_normalized_DAVIS_names": 0,
                "old_mean_geometry_spearman": geometry.geometry_concordance(
                    kir_centered, old_mean
                ),
                "old_endpoint_roc_auc": locked_validation["centered_docking"][
                    "roc_auc"
                ],
                "old_endpoint_average_precision": locked_validation[
                    "centered_docking"
                ]["average_precision"],
                "four_experimental_panels_raw_mean_pairwise_spearman": (
                    global_network_qap["raw_mean_pairwise_spearman"]
                ),
                "four_experimental_panels_centered_mean_pairwise_spearman": (
                    global_network_qap["centered_mean_pairwise_spearman"]
                ),
            },
            {
                "support": "exclude_11_exact_normalized_DAVIS_names",
                "drugs": len(without_name_overlap),
                "removed_exact_normalized_DAVIS_names": int(overlap_mask.sum()),
                "old_mean_geometry_spearman": overlap_validation[
                    "geometry_spearman_to_old_three_panel_mean"
                ],
                "old_endpoint_roc_auc": overlap_validation["retrieval"][
                    "centered_KiRHub_geometry"
                ]["roc_auc"],
                "old_endpoint_average_precision": overlap_validation["retrieval"][
                    "centered_KiRHub_geometry"
                ]["average_precision"],
                "four_experimental_panels_raw_mean_pairwise_spearman": (
                    overlap_raw_alignment
                ),
                "four_experimental_panels_centered_mean_pairwise_spearman": (
                    overlap_centered_alignment
                ),
            },
        ]
    )

    summary = {
        "analysis_status": (
            "exploratory_external_validation_integrated_with_explicit_"
            "nonpreregistered_status"
        ),
        "locked_primary_endpoint": {
            "definition": (
                "target pair in the centered upper 10% in at least two of DAVIS, "
                "PKIS2, and PKIS1; fixed before KiRHub was inspected"
            ),
            "positive_pairs": int(old_labels.sum()),
            "total_pairs": int(len(old_labels)),
            "panel_specific_positive_pairs": {
                name: int(labels.sum()) for name, labels in old_panel_labels.items()
            },
            "KiRHub_validation": locked_validation_output,
            "target_label_qap": locked_qap_output,
            "holm_adjusted_centered_predictor_p": locked_holm_centered,
            "holm_adjusted_centered_minus_raw_gain_p": locked_holm_gain,
            "geometry_concordance": {
                row.endpoint: {
                    "spearman": row.spearman,
                    "target_label_qap_p_positive": row.target_label_qap_p_positive,
                    "partial_spearman_controlling_receptor_sequence_identity": (
                        row.partial_spearman_controlling_receptor_sequence_identity
                    ),
                    "partial_target_label_qap_p_positive": (
                        row.partial_target_label_qap_p_positive
                    ),
                }
                for row in geometry_qap_frame.itertuples(index=False)
            },
        },
        "experimental_cross_panel_agreement": {
            "status": (
                "absolute_centered_alignment_positive_incremental_gain_not_"
                "supported_by_matched_null"
            ),
            "raw_mean_pairwise_spearman": global_network_qap[
                "raw_mean_pairwise_spearman"
            ],
            "centered_mean_pairwise_spearman": global_network_qap[
                "centered_mean_pairwise_spearman"
            ],
            "centered_minus_raw": global_network_qap["centered_minus_raw"],
            "panel_pairs_improved": int(
                (
                    experimental_alignment_frame.query("transform == 'two_way_center'")
                    .set_index(["panel_a", "panel_b"])
                    .spearman
                    > experimental_alignment_frame.query("transform == 'raw'")
                    .set_index(["panel_a", "panel_b"])
                    .spearman
                ).sum()
            ),
            "panel_pairs_total": 6,
            "paired_global_network_qap": global_network_qap,
            "ligand_bootstrap": experimental_bootstrap,
            "transformation_matched_independent_column_null": matched_null,
            "target_jackknife_centered_minus_raw_range": _range(
                experimental_jackknife_frame[
                    "centered_minus_raw_mean_pairwise_spearman"
                ]
            ),
            "interpretation": (
                "Across the four separately published experimental panels, the centered target "
                "co-response geometry is strongly aligned beyond a target-marginal-"
                "preserving null. The observed centered-minus-raw increase itself is "
                "not larger than the null's mechanical centering gain, so it cannot "
                "be claimed as independent evidence that centering improves biological "
                "transferability."
            ),
        },
        "chemical_independence_audit": {
            "comparison": "exact normalized drug name against the DAVIS 72-drug panel",
            "overlap_count": int(overlap_mask.sum()),
            "overlap_names": overlap_names,
            "no_structure_claim": (
                "KiRHub Table S4/Table S1 do not provide structures. PKIS1/PKIS2 "
                "release identifiers are not comparable drug-name namespaces, so "
                "full InChIKey/connectivity/scaffold overlap with all three panels "
                "was not asserted."
            ),
            "after_excluding_overlap": overlap_validation,
            "after_excluding_overlap_target_label_qap": overlap_qap_output,
            "after_excluding_overlap_partial_geometry_qap": overlap_partial,
        },
        "KiRHub_specific_top10_boundary": {
            "status": "exploratory_nonreplication_of_centered_minus_raw_gain",
            "definition": "upper 10% of centered KiRHub target-pair correlations",
            "metrics": kirhub_specific,
            "target_label_qap": kirhub_specific_qap,
            "interpretation": (
                "Centered DOCKSTRING does not outperform raw DOCKSTRING on this "
                "KiRHub-only endpoint; this limits any universal gain claim."
            ),
        },
        "exploratory_four_panel_consensus": {
            "definition": (
                "upper 10% in at least two of DAVIS, PKIS2, PKIS1, and KiRHub"
            ),
            "positive_pairs": int(four_labels.sum()),
            "panel_specific_positive_pairs": {
                name: int(labels.sum())
                for name, labels in four_panel_individual_labels.items()
            },
            "metrics": four_panel,
            "target_label_qap": four_panel_qap,
            "holm_adjusted_gain_p": four_panel_holm_gain,
            "threshold_warning": (
                "The 10% and at-least-two-of-four rule was not a locked external "
                "endpoint; threshold-family rows are descriptive and no threshold "
                "was selected by its docking performance."
            ),
        },
        "KiRHub_measurement_audit": {
            "drugs": int(len(primary_matrix)),
            "targets": int(primary_matrix.shape[1]),
            "cells": int(primary_matrix.size),
            "missing_cells": 0,
            "exact_100_ceiling_cells": int((primary_matrix == 100.0).sum()),
            "exact_100_ceiling_fraction": float((primary_matrix == 100.0).mean()),
            "targetwise_exact_100_fraction_range": _range(ceiling_fraction),
            "targetwise_exact_100_fraction_median": float(np.median(ceiling_fraction)),
            "exact_0_floor_cells": int((primary_matrix == 0.0).sum()),
            "exact_0_floor_fraction": float((primary_matrix == 0.0).mean()),
            "measurement_boundary": (
                "Residual activity was measured at a single 1-uM drug concentration "
                "and Km ATP; exact 100 values create substantial ceiling ties."
            ),
        },
        "KiRHub_PR_boundary": {
            "raw_correlation_PR_dimension": float(
                experimental_pr_frame.query(
                    "panel == 'KiRHub' and transform == 'raw'"
                ).correlation_PR_dimension.iloc[0]
            ),
            "two_way_centered_correlation_PR_dimension": float(
                experimental_pr_frame.query(
                    "panel == 'KiRHub' and transform == 'two_way_center'"
                ).correlation_PR_dimension.iloc[0]
            ),
            "interpretation": (
                "KiRHub cross-panel agreement rises even though its PR dimension "
                "falls. Two-way centering changes the scientific estimand; it is not "
                "a generic rank-inflation or dimension-maximization operation."
            ),
        },
        "target_jackknife_ranges": {
            column: _range(jackknife_frame[column])
            for column in (
                "kirhub_centered_vs_old_mean_geometry_spearman",
                "kirhub_old_endpoint_centered_roc_auc",
                "kirhub_old_endpoint_centered_average_precision",
                "four_panel_centered_minus_raw_docking_roc_auc",
                "four_panel_centered_minus_raw_docking_average_precision",
            )
        },
        "inference": {
            "target_label_qap_permutations": qap_permutations,
            "seed": seed,
            "qap_unit": (
                "whole target labels are permuted jointly; the 190 target pairs are "
                "never treated as independent"
            ),
            "p_value_resolution": 1.0 / (qap_permutations + 1),
            "confirmatory_status": (
                "KiRHub is temporally external to the locked three-panel endpoint, "
                "but this analysis remains exploratory because it was not prospectively registered."
            ),
        },
        "provenance": {
            "KiRHub": {
                "paper_doi": KIRHUB_DOI,
                "official_supplement_url": KIRHUB_URL,
                "workbook_sha256": KIRHUB_SHA256,
                "sheet": KIRHUB_SHEET,
                "header_row_one_based": KIRHUB_HEADER_ROW_ZERO_BASED + 1,
                "article_license": KIRHUB_ARTICLE_LICENSE,
                "article_license_url": KIRHUB_ARTICLE_LICENSE_URL,
                "portal_url": KIRHUB_PORTAL_URL,
                "portal_rights_notice": KIRHUB_PORTAL_RIGHTS_NOTICE,
                "redistribution": (
                    "source workbook intentionally omitted; only aggregate analytical "
                    "summaries are written, not compound-by-target activity rows or "
                    "per-compound profiles. The 190-edge target-pair geometry is "
                    "released separately as an aggregate statistic: 190 correlation "
                    "coefficients leave a 1,630-dimensional family of 92-by-20 "
                    "activity blocks indistinguishable"
                ),
                "data_rights_boundary": (
                    "The article is CC BY-NC-ND 4.0, but this analysis does not "
                    "interpret that article license as a separate license for the "
                    "underlying inhibition data. The official KIRHub portal identifies "
                    "those data as the property of Reaction Biology Corporation and "
                    "requires proper acknowledgement. The workbook is fetched only "
                    "from the publisher and is not vendored; downstream users must "
                    "independently confirm that their intended reuse is permitted."
                ),
                "structure_and_portal_boundary": (
                    "Tables S1 and S4 and the interactive portal do not provide a "
                    "frozen, machine-readable structure mapping for these 92 names. "
                    "Chemical independence was therefore audited only by exact "
                    "normalized names against DAVIS, not by InChIKey or scaffold."
                ),
            },
            "target_map": dict(TARGET_MAP),
            "cdk2_primary": "CDK2_CYCLIN_A",
            "cdk2_sensitivity": "CDK2_CYCLIN_E",
            "DOCKSTRING_reference": docking_provenance,
            "receptor_sequence_identity": sequence_provenance,
        },
        "strongest_honest_claim": (
            "Centered target co-response geometry shows strong agreement across four "
            "separately published kinase panels beyond target-marginal and receptor-sequence "
            "controls, and KiRHub recovers the pre-existing three-panel co-selective-"
            "pair endpoint after exact-name DAVIS overlaps are removed. The incremental "
            "centered-minus-raw gain is not beyond the transformation-matched null, and "
            "KiRHub does not validate a universal centered-over-raw docking advantage."
        ),
    }

    frames = {
        "geometry_qap.csv": geometry_qap_frame,
        "target_jackknife.csv": jackknife_frame,
        "transform_sensitivity.csv": transform_frame,
        "threshold_sensitivity.csv": threshold_frame,
        "cdk2_construct_sensitivity.csv": construct_frame,
        "drug_name_overlap_sensitivity.csv": overlap_frame,
        "target_ceiling_audit.csv": target_quality_frame,
        "experimental_cross_panel_alignment.csv": experimental_alignment_frame,
        "experimental_panel_pr.csv": experimental_pr_frame,
        "experimental_cross_panel_target_jackknife.csv": (
            experimental_jackknife_frame
        ),
        "experimental_cross_panel_ligand_bootstrap.csv": (
            experimental_bootstrap_frame
        ),
        "experimental_cross_panel_matched_null.csv": matched_null_frame,
    }
    return summary, frames


def write_outputs(
    output: Path,
    summary: dict,
    frames: dict[str, pd.DataFrame],
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    with (output / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    for name, frame in frames.items():
        frame.to_csv(output / name, index=False)
    readme = f"""# KiRHub 2026 external validation

This directory contains aggregate, exploratory validation results generated by
`analysis/kirhub_external_validation.py`.  KiRHub was used as a fixed/no-retuning
check of the pre-existing DAVIS/PKIS2/PKIS1 co-selective target-pair endpoint.
The endpoint preceded KiRHub inspection, but the ordering was not preregistered.

The source workbook is **not redistributed**.  Obtain it from the official
supplement ({KIRHUB_URL}) for DOI `{KIRHUB_DOI}` and verify SHA256
`{KIRHUB_SHA256}`.  The article is published under
{KIRHUB_ARTICLE_LICENSE} ({KIRHUB_ARTICLE_LICENSE_URL}); this package does not
assert that the article license separately licenses the underlying inhibition
data.  The official KIRHub portal ({KIRHUB_PORTAL_URL}) states: "{KIRHUB_PORTAL_RIGHTS_NOTICE}"
These files contain aggregate computational summaries, not source activity rows
or per-compound profiles.  The 190-edge target-pair geometry is released
elsewhere in the package as an aggregate statistic; see DATA_LICENSES.md.

Tables S1 and S4 and the interactive portal do not provide a frozen,
machine-readable chemical-structure mapping for the 92 names.  Consequently,
the independence sensitivity removes 11 exact normalized-name overlaps with
DAVIS but makes no InChIKey-, connectivity-, or scaffold-independence claim for
PKIS1/PKIS2.  Downstream users must independently confirm that their intended
reuse is allowed by the stated data rights.

Primary interpretation: centered experimental target geometry shows strong
cross-panel agreement beyond the target-marginal-preserving null, and KiRHub
recovers the previously fixed three-panel endpoint.  Target-label relabelling
supports the incremental centered-minus-raw retrieval gain, whereas an independent-
column transformation-matched null does not; the absolute centered predictor exceeds
both nulls.  The KiRHub-only docking
comparison is a negative boundary: centered DOCKSTRING does not beat raw
DOCKSTRING.  The four-panel consensus is secondary and threshold-dependent.

Reproduce with:

```bash
python analysis/fetch_kirhub_supplement.py /tmp/kirhub_supp_tables.xlsx
python analysis/kirhub_external_validation.py \\
  --kirhub-workbook /tmp/kirhub_supp_tables.xlsx \\
  --pkis1-zip /tmp/pkis1_supplement.zip
```

No row-level KiRHub activities or per-drug profiles are included.  Reaction
Biology Corporation should be acknowledged in any download, use or publication
of the underlying inhibition data.
"""
    (output / "README.md").write_text(readme)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kirhub-workbook", type=Path, required=True)
    parser.add_argument("--pkis1-zip", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--qap-permutations", type=int, default=49_999)
    parser.add_argument("--bootstrap-repeats", type=int, default=2_000)
    parser.add_argument("--matched-null-repeats", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--sequence-pairs", type=Path, default=DEFAULT_SEQUENCE_PAIRS)
    parser.add_argument(
        "--sequence-summary", type=Path, default=DEFAULT_SEQUENCE_SUMMARY
    )
    args = parser.parse_args()
    summary, frames = run_analysis(
        kirhub_workbook=args.kirhub_workbook,
        pkis1_zip=args.pkis1_zip,
        qap_permutations=args.qap_permutations,
        bootstrap_repeats=args.bootstrap_repeats,
        matched_null_repeats=args.matched_null_repeats,
        seed=args.seed,
        sequence_pairs=args.sequence_pairs,
        sequence_summary=args.sequence_summary,
    )
    write_outputs(args.output, summary, frames)
    print(f"Wrote KiRHub external-validation artifacts to {args.output}")


if __name__ == "__main__":
    main()
