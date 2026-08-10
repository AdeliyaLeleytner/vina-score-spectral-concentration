#!/usr/bin/env python3
"""Post-hoc KLIFS controls for the fixed kinase target-pair endpoint.

This science-only analysis does not redefine the existing DAVIS/PKIS2/PKIS1
endpoint.  It asks whether the previously released centered DOCKSTRING target
geometry remains associated with experimental target-pair geometry after rank
control for KLIFS kinase-pocket identity and coarse kinase annotations.

The KLIFS questions were formulated after inspecting the docking/experimental
results.  Every output therefore labels them exploratory/post hoc.  Target-label
QAP permutes the complete docking predictor network and leaves the endpoint and
structural controls fixed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import scipy
from scipy import stats

try:  # Package imports and direct script execution are both supported.
    from . import kirhub_external_validation as kirhub
    from . import replicated_pair_retrieval as retrieval
except ImportError:  # pragma: no cover - direct CLI execution.
    import kirhub_external_validation as kirhub  # type: ignore
    import replicated_pair_retrieval as retrieval  # type: ignore


PACKAGE = Path(__file__).resolve().parents[1]
DEFAULT_TARGET_PAIRS = (
    PACKAGE / "results" / "replicated_pair_retrieval" / "target_pairs.csv"
)
DEFAULT_OUTPUT = PACKAGE / "results" / "klifs_pocket_control"
DEFAULT_SEED = 20260803
DEFAULT_QAP_PERMUTATIONS = 50_000

KLIFS_URL = "https://klifs.net/api_v2/kinase_information?species=HUMAN"
KLIFS_SNAPSHOT_DATE = "2026-08-03"
KLIFS_SHA256 = "041a159e662a27696554867acbe1ac7fb35d36f71483d42ab4ac90e219dbef8f"
TARGET_PAIRS_SHA256 = (
    "f2e372ee45b38648cea8a23d485d7ed519c769e5146eef99b2e4177503466fc3"
)
POCKET_LENGTH = 85
TARGETS = tuple(kirhub.TARGETS)

PAIR_COLUMNS = (
    "target_a",
    "target_b",
    "primary_replicated_positive",
    "number_of_panels_in_top_10_percent",
    "DAVIS_centered_pair_percentile",
    "PKIS2_centered_pair_percentile",
    "PKIS1_centered_pair_percentile",
    "mean_experimental_centered_pair_percentile",
    "raw_docking_pair_percentile",
    "centered_docking_pair_percentile",
    "receptor_domain_sequence_identity",
)


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def release_provenance_path(path: Path) -> str:
    """Return a package-relative path without leaking a local checkout root."""
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(PACKAGE.resolve()).as_posix()
    except ValueError as error:
        raise ValueError(
            "release provenance paths must resolve inside the package"
        ) from error


def load_pair_artifact(
    path: Path,
    *,
    expected_sha256: str | None = TARGET_PAIRS_SHA256,
) -> pd.DataFrame:
    """Load and validate the already fixed 20-target pair artifact."""
    path = Path(path)
    observed_sha256 = sha256_file(path)
    if expected_sha256 is not None and observed_sha256 != expected_sha256:
        raise ValueError(
            "target-pair artifact SHA256 mismatch: "
            f"expected {expected_sha256}, observed {observed_sha256}"
        )
    frame = pd.read_csv(path)
    if tuple(frame.columns) != PAIR_COLUMNS:
        raise ValueError("target-pair artifact columns no longer match the contract")
    expected_pairs = len(TARGETS) * (len(TARGETS) - 1) // 2
    if len(frame) != expected_pairs:
        raise ValueError(
            f"expected {expected_pairs} target pairs, observed {len(frame)}"
        )
    target_set = set(frame.target_a) | set(frame.target_b)
    if target_set != set(TARGETS):
        raise ValueError("target-pair artifact does not contain the fixed target set")
    seen: set[tuple[str, str]] = set()
    for row in frame.itertuples(index=False):
        if row.target_a == row.target_b:
            raise ValueError("self-pairs are not allowed")
        pair = tuple(sorted((row.target_a, row.target_b)))
        if pair in seen:
            raise ValueError(f"duplicate target pair: {pair}")
        seen.add(pair)
    if frame[list(PAIR_COLUMNS[4:])].isna().any().any():
        raise ValueError("target-pair artifact contains missing numeric values")
    return frame


def load_klifs_annotations(
    path: Path,
    *,
    expected_sha256: str | None = KLIFS_SHA256,
) -> tuple[dict[str, dict], pd.DataFrame]:
    """Select one exact human KLIFS kinase record per fixed target."""
    path = Path(path)
    observed_sha256 = sha256_file(path)
    if expected_sha256 is not None and observed_sha256 != expected_sha256:
        raise ValueError(
            "KLIFS snapshot SHA256 mismatch: "
            f"expected {expected_sha256}, observed {observed_sha256}"
        )
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError("KLIFS kinase_information response must be a JSON list")

    selected: dict[str, dict] = {}
    rows: list[dict] = []
    for target in TARGETS:
        matches = [
            record
            for record in payload
            if str(record.get("species", "")).casefold() == "human"
            and str(record.get("gene_name", "")).upper() == target
        ]
        if len(matches) != 1:
            raise ValueError(
                f"expected exactly one human KLIFS record for {target}, "
                f"observed {len(matches)}"
            )
        record = matches[0]
        pocket = str(record.get("pocket", ""))
        if len(pocket) != POCKET_LENGTH:
            raise ValueError(
                f"KLIFS pocket for {target} has length {len(pocket)}, "
                f"expected {POCKET_LENGTH}"
            )
        if not set(pocket) <= set("ACDEFGHIKLMNPQRSTVWY-"):
            raise ValueError(f"KLIFS pocket for {target} has unexpected symbols")
        selected[target] = record
        rows.append(
            {
                "target": target,
                "klifs_kinase_id": int(record["kinase_ID"]),
                "klifs_name": str(record["name"]),
                "uniprot": str(record["uniprot"]),
                "klifs_group": str(record["group"]),
                "klifs_family": str(record["family"]),
                "klifs_subfamily": str(record["subfamily"]),
                "pocket_length": len(pocket),
                "pocket_sequence_sha256": hashlib.sha256(
                    pocket.encode("ascii")
                ).hexdigest(),
            }
        )
    return selected, pd.DataFrame(rows)


def structural_matrices(
    records: dict[str, dict],
    targets: Sequence[str] = TARGETS,
) -> dict[str, np.ndarray]:
    """Create aligned pocket-identity and coarse annotation matrices."""
    p = len(targets)
    pocket = np.eye(p, dtype=np.float64)
    same_family = np.eye(p, dtype=np.float64)
    same_group = np.eye(p, dtype=np.float64)
    same_subfamily = np.eye(p, dtype=np.float64)
    for first, target_a in enumerate(targets):
        for second in range(first + 1, p):
            target_b = targets[second]
            left = records[target_a]
            right = records[target_b]
            pocket_a = str(left["pocket"])
            pocket_b = str(right["pocket"])
            identity = sum(
                residue_a == residue_b
                for residue_a, residue_b in zip(pocket_a, pocket_b)
            ) / POCKET_LENGTH
            family = float(str(left["family"]) == str(right["family"]))
            group = float(str(left["group"]) == str(right["group"]))
            left_subfamily = str(left["subfamily"])
            right_subfamily = str(right["subfamily"])
            subfamily = float(
                bool(left_subfamily)
                and left_subfamily == right_subfamily
            )
            for matrix, value in (
                (pocket, identity),
                (same_family, family),
                (same_group, group),
                (same_subfamily, subfamily),
            ):
                matrix[first, second] = matrix[second, first] = value
    return {
        "klifs_pocket_identity": pocket,
        "same_klifs_family": same_family,
        "same_klifs_group": same_group,
        "same_nonempty_klifs_subfamily": same_subfamily,
    }


def matrix_from_pairs(
    frame: pd.DataFrame,
    column: str,
    targets: Sequence[str] = TARGETS,
) -> np.ndarray:
    target_index = {target: index for index, target in enumerate(targets)}
    matrix = np.eye(len(targets), dtype=np.float64)
    for row in frame.itertuples(index=False):
        first = target_index[row.target_a]
        second = target_index[row.target_b]
        value = float(getattr(row, column))
        matrix[first, second] = matrix[second, first] = value
    if not np.isfinite(matrix).all():
        raise ValueError(f"matrix reconstructed from {column} is not finite")
    return matrix


def label_matrix_from_pairs(
    frame: pd.DataFrame,
    targets: Sequence[str] = TARGETS,
) -> np.ndarray:
    target_index = {target: index for index, target in enumerate(targets)}
    matrix = np.zeros((len(targets), len(targets)), dtype=bool)
    for row in frame.itertuples(index=False):
        first = target_index[row.target_a]
        second = target_index[row.target_b]
        matrix[first, second] = matrix[second, first] = bool(
            row.primary_replicated_positive
        )
    return matrix


def target_label_pair_maps(
    target_count: int,
    permutations: int,
    seed: int,
) -> np.ndarray:
    """Map each relabelled upper triangle back to the original pair vector."""
    if target_count < 3 or permutations < 1:
        raise ValueError("QAP requires at least three targets and one permutation")
    triangle = np.triu_indices(target_count, k=1)
    pair_count = len(triangle[0])
    pair_index = np.full((target_count, target_count), -1, dtype=np.int32)
    for index, (first, second) in enumerate(zip(*triangle)):
        pair_index[first, second] = pair_index[second, first] = index
    maps = np.empty((permutations, pair_count), dtype=np.int32)
    rng = np.random.default_rng(seed)
    for repetition in range(permutations):
        order = rng.permutation(target_count)
        maps[repetition] = pair_index[np.ix_(order, order)][triangle]
    return maps


def _rank(values: np.ndarray) -> np.ndarray:
    return stats.rankdata(np.asarray(values, dtype=np.float64), method="average")


def partial_rank_qap(
    predictor: np.ndarray,
    endpoint: np.ndarray,
    controls: Sequence[np.ndarray],
    permutation_maps: np.ndarray,
    *,
    chunk_size: int = 5_000,
) -> dict:
    """Partial Spearman QAP with fixed dyadic control matrices."""
    p = len(predictor)
    matrices = (endpoint, *controls)
    if predictor.shape != (p, p) or any(matrix.shape != (p, p) for matrix in matrices):
        raise ValueError("all QAP matrices must be aligned and square")
    triangle = np.triu_indices(p, k=1)
    pair_count = len(triangle[0])
    if permutation_maps.shape[1] != pair_count:
        raise ValueError("permutation maps do not match the target count")

    predictor_rank = _rank(predictor[triangle])
    endpoint_rank = _rank(endpoint[triangle])
    design = np.column_stack(
        [
            np.ones(pair_count, dtype=np.float64),
            *[_rank(control[triangle]) for control in controls],
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
        (predictor_residual @ endpoint_residual)
        / np.sqrt(predictor_ss * endpoint_ss)
    )

    permutations = len(permutation_maps)
    null = np.empty(permutations, dtype=np.float64)
    for start in range(0, permutations, chunk_size):
        selected = permutation_maps[start : start + chunk_size]
        permuted = predictor_rank[selected]
        residual = permuted - (design @ (projection @ permuted.T)).T
        numerator = residual @ endpoint_residual
        denominator = np.sqrt(np.sum(residual**2, axis=1) * endpoint_ss)
        null[start : start + len(selected)] = numerator / denominator
    return {
        "partial_spearman": observed,
        "target_label_qap_p_positive": float(
            (1 + np.sum(null >= observed)) / (permutations + 1)
        ),
        "null_q025": float(np.quantile(null, 0.025)),
        "null_q975": float(np.quantile(null, 0.975)),
        "permutations": int(permutations),
    }


def _row_retrieval_metrics(
    scores: np.ndarray,
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized equivalent of retrieval_metrics for many score rows."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=bool)
    positives = int(labels.sum())
    negatives = int(len(labels) - positives)
    if scores.ndim != 2 or scores.shape[1] != len(labels):
        raise ValueError("score rows and labels do not align")
    if positives < 1 or negatives < 1:
        raise ValueError("retrieval requires both endpoint classes")
    ranks = stats.rankdata(scores, axis=1, method="average")
    auc = (
        ranks[:, labels].sum(axis=1) - positives * (positives + 1) / 2
    ) / (positives * negatives)
    order = np.argsort(scores, axis=1, kind="stable")[:, ::-1]
    ordered = labels[order]
    precision = np.cumsum(ordered, axis=1) / (np.arange(len(labels)) + 1)
    average_precision = np.sum(ordered * precision, axis=1) / positives
    return auc.astype(np.float64), average_precision.astype(np.float64)


def endpoint_retrieval_qap(
    labels: np.ndarray,
    predictor: np.ndarray,
    permutation_maps: np.ndarray,
    *,
    pair_mask: np.ndarray | None = None,
    chunk_size: int = 5_000,
) -> dict:
    """One-predictor fixed-endpoint QAP, optionally on a fixed pair stratum."""
    p = len(predictor)
    triangle = np.triu_indices(p, k=1)
    endpoint = np.asarray(labels[triangle], dtype=bool)
    scores = np.asarray(predictor[triangle], dtype=np.float64)
    if pair_mask is None:
        pair_mask = np.ones(len(scores), dtype=bool)
    pair_mask = np.asarray(pair_mask, dtype=bool)
    endpoint = endpoint[pair_mask]
    observed = retrieval.retrieval_metrics(endpoint, scores[pair_mask])
    permutations = len(permutation_maps)
    null_auc = np.empty(permutations, dtype=np.float64)
    null_ap = np.empty(permutations, dtype=np.float64)
    for start in range(0, permutations, chunk_size):
        selected = permutation_maps[start : start + chunk_size]
        permuted_scores = scores[selected][:, pair_mask]
        auc, average_precision = _row_retrieval_metrics(
            permuted_scores, endpoint
        )
        null_auc[start : start + len(selected)] = auc
        null_ap[start : start + len(selected)] = average_precision
    return {
        "pairs": int(pair_mask.sum()),
        "positive_pairs": int(endpoint.sum()),
        "roc_auc": float(observed["roc_auc"]),
        "roc_auc_qap_p_positive": float(
            (1 + np.sum(null_auc >= observed["roc_auc"]))
            / (permutations + 1)
        ),
        "average_precision": float(observed["average_precision"]),
        "average_precision_qap_p_positive": float(
            (1 + np.sum(null_ap >= observed["average_precision"]))
            / (permutations + 1)
        ),
    }


def paired_endpoint_retrieval_qap(
    labels: np.ndarray,
    raw_predictor: np.ndarray,
    centered_predictor: np.ndarray,
    permutation_maps: np.ndarray,
    *,
    pair_mask: np.ndarray,
    chunk_size: int = 5_000,
) -> dict:
    """Paired raw-versus-centered QAP on one fixed pair stratum."""
    p = len(raw_predictor)
    triangle = np.triu_indices(p, k=1)
    pair_mask = np.asarray(pair_mask, dtype=bool)
    endpoint = np.asarray(labels[triangle], dtype=bool)[pair_mask]
    raw_scores = np.asarray(raw_predictor[triangle], dtype=np.float64)
    centered_scores = np.asarray(centered_predictor[triangle], dtype=np.float64)
    raw_observed = retrieval.retrieval_metrics(endpoint, raw_scores[pair_mask])
    centered_observed = retrieval.retrieval_metrics(
        endpoint, centered_scores[pair_mask]
    )
    permutations = len(permutation_maps)
    null = {
        name: {
            "roc_auc": np.empty(permutations, dtype=np.float64),
            "average_precision": np.empty(permutations, dtype=np.float64),
        }
        for name in ("raw", "centered")
    }
    for start in range(0, permutations, chunk_size):
        selected = permutation_maps[start : start + chunk_size]
        for name, scores in (
            ("raw", raw_scores),
            ("centered", centered_scores),
        ):
            auc, average_precision = _row_retrieval_metrics(
                scores[selected][:, pair_mask], endpoint
            )
            null[name]["roc_auc"][start : start + len(selected)] = auc
            null[name]["average_precision"][start : start + len(selected)] = (
                average_precision
            )
    output: dict[str, object] = {
        "pairs": int(pair_mask.sum()),
        "positive_pairs": int(endpoint.sum()),
        "raw_roc_auc": float(raw_observed["roc_auc"]),
        "centered_roc_auc": float(centered_observed["roc_auc"]),
        "raw_average_precision": float(raw_observed["average_precision"]),
        "centered_average_precision": float(
            centered_observed["average_precision"]
        ),
    }
    for metric in ("roc_auc", "average_precision"):
        observed_delta = centered_observed[metric] - raw_observed[metric]
        null_delta = null["centered"][metric] - null["raw"][metric]
        output[f"centered_minus_raw_{metric}"] = float(observed_delta)
        output[f"centered_{metric}_qap_p_positive"] = float(
            (1 + np.sum(null["centered"][metric] >= centered_observed[metric]))
            / (permutations + 1)
        )
        output[f"paired_gain_{metric}_qap_p_positive"] = float(
            (1 + np.sum(null_delta >= observed_delta)) / (permutations + 1)
        )
        output[f"paired_delta_{metric}_null_q025"] = float(
            np.quantile(null_delta, 0.025)
        )
        output[f"paired_delta_{metric}_null_q975"] = float(
            np.quantile(null_delta, 0.975)
        )
    return output


def pair_annotation_frame(
    pair_frame: pd.DataFrame,
    structural: dict[str, np.ndarray],
    targets: Sequence[str] = TARGETS,
) -> pd.DataFrame:
    target_index = {target: index for index, target in enumerate(targets)}
    by_pair = {
        tuple(sorted((row.target_a, row.target_b))): row
        for row in pair_frame.itertuples(index=False)
    }
    rows: list[dict] = []
    for first, target_a in enumerate(targets):
        for second in range(first + 1, len(targets)):
            target_b = targets[second]
            source = by_pair[tuple(sorted((target_a, target_b)))]
            rows.append(
                {
                    "target_a": target_a,
                    "target_b": target_b,
                    "primary_replicated_positive": bool(
                        source.primary_replicated_positive
                    ),
                    "number_of_panels_in_top_10_percent": int(
                        source.number_of_panels_in_top_10_percent
                    ),
                    "DAVIS_centered_pair_percentile": float(
                        source.DAVIS_centered_pair_percentile
                    ),
                    "PKIS2_centered_pair_percentile": float(
                        source.PKIS2_centered_pair_percentile
                    ),
                    "PKIS1_centered_pair_percentile": float(
                        source.PKIS1_centered_pair_percentile
                    ),
                    "mean_experimental_centered_pair_percentile": float(
                        source.mean_experimental_centered_pair_percentile
                    ),
                    "raw_docking_pair_percentile": float(
                        source.raw_docking_pair_percentile
                    ),
                    "centered_docking_pair_percentile": float(
                        source.centered_docking_pair_percentile
                    ),
                    "receptor_domain_sequence_identity": float(
                        source.receptor_domain_sequence_identity
                    ),
                    "klifs_pocket_identity": float(
                        structural["klifs_pocket_identity"][first, second]
                    ),
                    "same_klifs_group": bool(
                        structural["same_klifs_group"][first, second]
                    ),
                    "same_klifs_family": bool(
                        structural["same_klifs_family"][first, second]
                    ),
                    "same_nonempty_klifs_subfamily": bool(
                        structural["same_nonempty_klifs_subfamily"][
                            first, second
                        ]
                    ),
                }
            )
    frame = pd.DataFrame(rows)
    if set(frame.target_a) | set(frame.target_b) != set(target_index):
        raise RuntimeError("pair annotation frame lost targets")
    return frame


def _control_label(names: Iterable[str]) -> str:
    names = tuple(names)
    return "+".join(names) if names else "none"


def run_analysis(
    *,
    target_pairs: Path,
    klifs_json: Path,
    kirhub_workbook: Path,
    output_dir: Path,
    permutations: int,
    seed: int,
) -> dict:
    pair_frame = load_pair_artifact(target_pairs)
    records, target_annotations = load_klifs_annotations(klifs_json)
    structural = structural_matrices(records)
    pair_annotations = pair_annotation_frame(pair_frame, structural)

    matrices = {
        "DAVIS": matrix_from_pairs(
            pair_frame, "DAVIS_centered_pair_percentile"
        ),
        "PKIS2": matrix_from_pairs(
            pair_frame, "PKIS2_centered_pair_percentile"
        ),
        "PKIS1": matrix_from_pairs(
            pair_frame, "PKIS1_centered_pair_percentile"
        ),
        "old_three_panel_mean": matrix_from_pairs(
            pair_frame, "mean_experimental_centered_pair_percentile"
        ),
        "raw_Vina": matrix_from_pairs(
            pair_frame, "raw_docking_pair_percentile"
        ),
        "centered_Vina": matrix_from_pairs(
            pair_frame, "centered_docking_pair_percentile"
        ),
        "receptor_domain_sequence_identity": matrix_from_pairs(
            pair_frame, "receptor_domain_sequence_identity"
        ),
        **structural,
    }
    kirhub_frame = kirhub.load_kirhub(kirhub_workbook)
    kirhub_matrix = kirhub_frame[list(TARGETS)].to_numpy(dtype=np.float64)
    matrices["KiRHub"] = kirhub.correlation_geometry(
        kirhub_matrix, "two_way_center"
    )
    labels = label_matrix_from_pairs(pair_frame)
    permutation_maps = target_label_pair_maps(
        len(TARGETS), permutations, seed
    )

    control_sets = {
        "klifs_pocket_identity": ["klifs_pocket_identity"],
        "receptor_domain_sequence_identity": [
            "receptor_domain_sequence_identity"
        ],
        "sequence_plus_pocket": [
            "receptor_domain_sequence_identity",
            "klifs_pocket_identity",
        ],
        "sequence_pocket_family_group": [
            "receptor_domain_sequence_identity",
            "klifs_pocket_identity",
            "same_klifs_family",
            "same_klifs_group",
        ],
    }
    endpoint_names = (
        "DAVIS",
        "PKIS2",
        "PKIS1",
        "old_three_panel_mean",
        "KiRHub",
    )
    partial_rows: list[dict] = []
    for endpoint_name in endpoint_names:
        for control_name, control_matrix_names in control_sets.items():
            result = partial_rank_qap(
                matrices["centered_Vina"],
                matrices[endpoint_name],
                [matrices[name] for name in control_matrix_names],
                permutation_maps,
            )
            partial_rows.append(
                {
                    "analysis_status": "exploratory_post_hoc",
                    "predictor": "centered_Vina_target_geometry",
                    "endpoint": endpoint_name,
                    "controls": control_name,
                    **result,
                    "seed": seed,
                }
            )
    partial_frame = pd.DataFrame(partial_rows)

    annotation_rows: list[dict] = []
    for annotation_name in (
        "klifs_pocket_identity",
        "same_klifs_family",
        "same_klifs_group",
    ):
        result = partial_rank_qap(
            matrices["centered_Vina"],
            matrices[annotation_name],
            [],
            permutation_maps,
        )
        annotation_rows.append(
            {
                "analysis_status": "exploratory_post_hoc",
                "predictor": "centered_Vina_target_geometry",
                "annotation_endpoint": annotation_name,
                "controls": _control_label(()),
                **result,
                "seed": seed,
            }
        )
    annotation_frame = pd.DataFrame(annotation_rows)

    retrieval_predictors = {
        "receptor_domain_sequence_identity": matrices[
            "receptor_domain_sequence_identity"
        ],
        "klifs_pocket_identity": matrices["klifs_pocket_identity"],
        "same_klifs_group": matrices["same_klifs_group"],
        "same_klifs_family": matrices["same_klifs_family"],
        "raw_Vina_target_geometry": matrices["raw_Vina"],
        "centered_Vina_target_geometry": matrices["centered_Vina"],
    }
    retrieval_rows: list[dict] = []
    for predictor_name, predictor in retrieval_predictors.items():
        retrieval_rows.append(
            {
                "analysis_status": "exploratory_post_hoc_structural_control",
                "predictor": predictor_name,
                **endpoint_retrieval_qap(
                    labels, predictor, permutation_maps
                ),
                "seed": seed,
                "permutations": permutations,
            }
        )
    retrieval_frame = pd.DataFrame(retrieval_rows)

    triangle = np.triu_indices(len(TARGETS), k=1)
    strata = {
        "different_KLIFS_family": (
            matrices["same_klifs_family"][triangle] == 0
        ),
        "same_KLIFS_group_different_family": (
            (matrices["same_klifs_group"][triangle] == 1)
            & (matrices["same_klifs_family"][triangle] == 0)
        ),
        "KLIFS_pocket_identity_below_0.50": (
            matrices["klifs_pocket_identity"][triangle] < 0.50
        ),
    }
    stratum_rows: list[dict] = []
    for stratum_name, pair_mask in strata.items():
        stratum_rows.append(
            {
                "analysis_status": "exploratory_post_hoc_stratum",
                "stratum": stratum_name,
                "selection_boundary": (
                    "strata were formulated after inspection; nominal QAP "
                    "probabilities are descriptive and unadjusted"
                ),
                **paired_endpoint_retrieval_qap(
                    labels,
                    matrices["raw_Vina"],
                    matrices["centered_Vina"],
                    permutation_maps,
                    pair_mask=pair_mask,
                ),
                "seed": seed,
                "permutations": permutations,
            }
        )
    stratum_frame = pd.DataFrame(stratum_rows)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "target_annotations.csv": target_annotations,
        "target_pairs.csv": pair_annotations,
        "partial_geometry_qap.csv": partial_frame,
        "annotation_associations.csv": annotation_frame,
        "locked_endpoint_retrieval.csv": retrieval_frame,
        "stratified_retrieval.csv": stratum_frame,
    }
    for filename, frame in outputs.items():
        frame.to_csv(output_dir / filename, index=False)

    def partial_record(endpoint: str, controls: str) -> dict:
        row = partial_frame[
            (partial_frame.endpoint == endpoint)
            & (partial_frame.controls == controls)
        ].iloc[0]
        return {
            "partial_spearman": float(row.partial_spearman),
            "target_label_qap_p_positive": float(
                row.target_label_qap_p_positive
            ),
        }

    different_family = stratum_frame[
        stratum_frame.stratum == "different_KLIFS_family"
    ].iloc[0]
    summary = {
        "analysis": "KLIFS pocket and kinase-annotation controls",
        "analysis_status": "exploratory_post_hoc",
        "preexisting_locked_endpoint": (
            "upper 10% centered target-correlation pair in at least two of "
            "DAVIS, PKIS2, and PKIS1; fixed before the KLIFS analysis"
        ),
        "primary_estimand": (
            "partial Spearman association between centered Vina and "
            "experimental target-pair geometries after rank-linear structural "
            "control; inference permutes complete Vina target labels"
        ),
        "claim_boundary": (
            "The results support transferable chemistry-induced target "
            "co-response geometry beyond the tested sequence and KLIFS controls. "
            "They do not establish pose accuracy, causal pocket determinants, "
            "superiority to sequence as a standalone baseline, or a universal "
            "benefit of centering. PR magnitude is not evaluated as a quality score."
        ),
        "selection_boundary": (
            "This is post hoc: KLIFS controls and all structural strata were "
            "formulated after inspection of the docking/experimental geometry "
            "results. Nominal probabilities are descriptive and are not adjusted "
            "for this broader exploratory search."
        ),
        "key_results": {
            "old_three_panel_mean_controlling_KLIFS_pocket": partial_record(
                "old_three_panel_mean", "klifs_pocket_identity"
            ),
            "old_three_panel_mean_controlling_all_structural_labels": (
                partial_record(
                    "old_three_panel_mean",
                    "sequence_pocket_family_group",
                )
            ),
            "independent_KiRHub_controlling_KLIFS_pocket": partial_record(
                "KiRHub", "klifs_pocket_identity"
            ),
            "independent_KiRHub_controlling_all_structural_labels": (
                partial_record("KiRHub", "sequence_pocket_family_group")
            ),
            "different_family_stratum": {
                "pairs": int(different_family.pairs),
                "positive_pairs": int(different_family.positive_pairs),
                "raw_roc_auc": float(different_family.raw_roc_auc),
                "centered_roc_auc": float(
                    different_family.centered_roc_auc
                ),
                "paired_gain_roc_auc_qap_p_positive": float(
                    different_family.paired_gain_roc_auc_qap_p_positive
                ),
                "raw_average_precision": float(
                    different_family.raw_average_precision
                ),
                "centered_average_precision": float(
                    different_family.centered_average_precision
                ),
                "paired_gain_average_precision_qap_p_positive": float(
                    different_family.paired_gain_average_precision_qap_p_positive
                ),
                "analysis_status": "exploratory_post_hoc_stratum",
            },
        },
        "support": {
            "targets": len(TARGETS),
            "target_pairs": len(pair_frame),
            "locked_positive_pairs": int(
                pair_frame.primary_replicated_positive.sum()
            ),
            "targets_in_order": list(TARGETS),
            "KLIFS_same_family_pairs": int(
                matrices["same_klifs_family"][triangle].sum()
            ),
            "KLIFS_same_group_pairs": int(
                matrices["same_klifs_group"][triangle].sum()
            ),
        },
        "configuration": {
            "target_label_qap_permutations": permutations,
            "seed": seed,
            "pocket_identity_denominator": POCKET_LENGTH,
            "pocket_gap_handling": (
                "aligned characters are compared directly; identical gaps count "
                "as matches"
            ),
            "partial_control": (
                "ordinary least squares residualization of pairwise ranks with "
                "an intercept"
            ),
        },
        "sources": {
            "fixed_target_pairs": {
                "path": release_provenance_path(target_pairs),
                "sha256": sha256_file(target_pairs),
            },
            "KLIFS_kinase_information": {
                "url": KLIFS_URL,
                "snapshot_date": KLIFS_SNAPSHOT_DATE,
                "sha256": sha256_file(klifs_json),
                "selected_fields": [
                    "kinase_ID",
                    "name",
                    "gene_name",
                    "family",
                    "group",
                    "subfamily",
                    "species",
                    "uniprot",
                    "pocket",
                ],
                "source_data_redistributed": False,
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
        "--target-pairs",
        type=Path,
        default=DEFAULT_TARGET_PAIRS,
        help="checksum-validated fixed pair artifact",
    )
    parser.add_argument(
        "--klifs-json",
        type=Path,
        required=True,
        help="KLIFS kinase_information JSON snapshot",
    )
    parser.add_argument(
        "--kirhub-workbook",
        type=Path,
        required=True,
        help="checksum-validated KiRHub Table S4 workbook",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT
    )
    parser.add_argument(
        "--qap-permutations",
        type=int,
        default=DEFAULT_QAP_PERMUTATIONS,
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_analysis(
        target_pairs=args.target_pairs,
        klifs_json=args.klifs_json,
        kirhub_workbook=args.kirhub_workbook,
        output_dir=args.output_dir,
        permutations=args.qap_permutations,
        seed=args.seed,
    )
    key = summary["key_results"]
    old = key["old_three_panel_mean_controlling_KLIFS_pocket"]
    external = key["independent_KiRHub_controlling_KLIFS_pocket"]
    print(
        "KLIFS pocket control complete: "
        f"old-three-panel partial rho={old['partial_spearman']:.3f}, "
        f"p={old['target_label_qap_p_positive']:.4g}; "
        f"KiRHub partial rho={external['partial_spearman']:.3f}, "
        f"p={external['target_label_qap_p_positive']:.4g}."
    )


if __name__ == "__main__":
    main()
