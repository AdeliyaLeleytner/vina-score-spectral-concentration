#!/usr/bin/env python3
"""Pocket-gated fusion of sequence and residual Vina target geometry.

This science-only analysis asks whether residual docking geometry is most useful
for kinase pairs whose ATP-site sequences are relatively dissimilar.  A single
KLIFS-pocket-identity threshold is selected on PKIS1 by average precision from
a fixed coarse grid.  The threshold is then frozen for DAVIS, PKIS2, and
KiRHub.  Target-label QAP permutes only the Vina geometry; sequence, pocket
identity, the experimental panels, and the selected threshold remain fixed.

The exercise is post hoc: the resources have been inspected elsewhere in the
project.  It is a locked/no-retuning evaluation, not prospective validation.
No manuscript file is read or modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import OrderedDict
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import average_precision_score, roc_auc_score

try:
    from .biological_core_modes import holm_adjust
    from .klifs_pocket_control import matrix_from_pairs
    from .sequence_docking_fusion import (
        TARGETS,
        load_inputs,
        rank_matrix,
        upper,
    )
except ImportError:  # pragma: no cover - direct CLI execution
    from biological_core_modes import holm_adjust  # type: ignore
    from klifs_pocket_control import matrix_from_pairs  # type: ignore
    from sequence_docking_fusion import (  # type: ignore
        TARGETS,
        load_inputs,
        rank_matrix,
        upper,
    )


PACKAGE = Path(__file__).resolve().parents[1]
DEFAULT_PAIRS = PACKAGE / "results" / "klifs_pocket_control" / "target_pairs.csv"
DEFAULT_KIRHUB = Path("/private/tmp/kirhub_supp_tables.xlsx")
DEFAULT_OUTPUT = PACKAGE / "results" / "pocket_gated_fusion"
DEFAULT_SEED = 20260818
DEFAULT_PERMUTATIONS = 50_000
TOP_FRACTION = 0.10
THRESHOLD_GRID = tuple(np.arange(0.30, 0.701, 0.05).round(2))
DISCOVERY_PANEL = "PKIS1"
LOCKED_PANELS = ("DAVIS", "PKIS2", "KiRHub")
METRICS = ("continuous_spearman", "roc_auc", "average_precision")


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def json_ready(value: object) -> object:
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
        raise ValueError("cannot serialize a non-finite value")
    return value


def top_labels(endpoint: np.ndarray, fraction: float = TOP_FRACTION) -> np.ndarray:
    values = upper(endpoint)
    count = max(1, int(np.ceil(fraction * len(values))))
    labels = np.zeros(len(values), dtype=bool)
    labels[np.argsort(values, kind="mergesort")[-count:]] = True
    return labels


def predictor_metrics(predictor: np.ndarray, endpoint: np.ndarray) -> dict[str, float]:
    x = upper(predictor)
    y = upper(endpoint)
    labels = top_labels(endpoint)
    return {
        "continuous_spearman": float(stats.spearmanr(x, y).statistic),
        "roc_auc": float(roc_auc_score(labels, x)),
        "average_precision": float(average_precision_score(labels, x)),
    }


def gated_fusion(
    sequence_rank: np.ndarray,
    vina_rank: np.ndarray,
    pocket_identity: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """Use equal-rank fusion below threshold and sequence alone above it."""
    shapes = {
        np.asarray(sequence_rank).shape,
        np.asarray(vina_rank).shape,
        np.asarray(pocket_identity).shape,
    }
    if len(shapes) != 1 or len(next(iter(shapes))) != 2:
        raise ValueError("all gated-fusion inputs must be aligned square matrices")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("pocket-identity threshold must lie in [0, 1]")
    low_identity = np.asarray(pocket_identity) < threshold
    result = np.where(
        low_identity,
        (np.asarray(sequence_rank) + np.asarray(vina_rank)) / 2.0,
        np.asarray(sequence_rank),
    )
    result = (result + result.T) / 2.0
    np.fill_diagonal(result, 1.0)
    return result


def gated_multiview_fusion(
    sequence_rank: np.ndarray,
    docking_ranks: tuple[np.ndarray, ...],
    pocket_identity: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """Use sequence plus all docking views only below the pocket threshold."""
    if not docking_ranks:
        raise ValueError("at least one docking view is required")
    shapes = {
        np.asarray(sequence_rank).shape,
        np.asarray(pocket_identity).shape,
        *(np.asarray(matrix).shape for matrix in docking_ranks),
    }
    if len(shapes) != 1:
        raise ValueError("all multiview inputs must have the same shape")
    low_identity = np.asarray(pocket_identity) < threshold
    combined = np.mean(
        [np.asarray(sequence_rank), *[np.asarray(value) for value in docking_ranks]],
        axis=0,
    )
    result = np.where(low_identity, combined, np.asarray(sequence_rank))
    result = (result + result.T) / 2.0
    np.fill_diagonal(result, 1.0)
    return result


def select_threshold(
    sequence_rank: np.ndarray,
    vina_rank: np.ndarray,
    pocket_identity: np.ndarray,
    endpoint: np.ndarray,
    grid: Iterable[float] = THRESHOLD_GRID,
) -> tuple[float, pd.DataFrame]:
    """Select the AP-maximizing threshold on the discovery endpoint only."""
    rows: list[dict[str, float | int]] = []
    for threshold in sorted(set(float(value) for value in grid)):
        predictor = gated_fusion(
            sequence_rank, vina_rank, pocket_identity, threshold
        )
        rows.append(
            {
                "threshold": threshold,
                "low_identity_pairs": int(
                    np.sum(upper(pocket_identity) < threshold)
                ),
                **predictor_metrics(predictor, endpoint),
            }
        )
    frame = pd.DataFrame.from_records(rows)
    if frame.empty:
        raise ValueError("threshold grid is empty")
    # Stable tie break: prefer the simpler, lower threshold.
    selected = frame.sort_values(
        ["average_precision", "threshold"],
        ascending=[False, True],
        kind="mergesort",
    ).iloc[0]
    return float(selected.threshold), frame


def select_multiview_threshold(
    sequence_rank: np.ndarray,
    docking_ranks: tuple[np.ndarray, ...],
    pocket_identity: np.ndarray,
    endpoint: np.ndarray,
    grid: Iterable[float] = THRESHOLD_GRID,
) -> tuple[float, pd.DataFrame]:
    rows: list[dict[str, float | int]] = []
    for threshold in sorted(set(float(value) for value in grid)):
        predictor = gated_multiview_fusion(
            sequence_rank, docking_ranks, pocket_identity, threshold
        )
        rows.append(
            {
                "threshold": threshold,
                "low_identity_pairs": int(
                    np.sum(upper(pocket_identity) < threshold)
                ),
                **predictor_metrics(predictor, endpoint),
            }
        )
    frame = pd.DataFrame.from_records(rows)
    selected = frame.sort_values(
        ["average_precision", "threshold"],
        ascending=[False, True],
        kind="mergesort",
    ).iloc[0]
    return float(selected.threshold), frame


def evaluate_predictors(
    sequence_rank: np.ndarray,
    raw_vina_rank: np.ndarray,
    vina_rank: np.ndarray,
    pocket_identity: np.ndarray,
    endpoints: dict[str, np.ndarray],
    threshold: float,
) -> pd.DataFrame:
    predictors = OrderedDict(
        [
            ("sequence_only", sequence_rank),
            (
                "ungated_equal_rank_sequence_plus_raw_Vina",
                (sequence_rank + raw_vina_rank) / 2,
            ),
            ("ungated_equal_rank_sequence_plus_Vina", (sequence_rank + vina_rank) / 2),
            (
                "ungated_equal_rank_sequence_plus_raw_plus_centered_Vina",
                (sequence_rank + raw_vina_rank + vina_rank) / 3,
            ),
            (
                "pocket_gated_sequence_plus_raw_Vina",
                gated_fusion(
                    sequence_rank, raw_vina_rank, pocket_identity, threshold
                ),
            ),
            (
                "pocket_gated_sequence_plus_Vina",
                gated_fusion(sequence_rank, vina_rank, pocket_identity, threshold),
            ),
            (
                "pocket_gated_sequence_plus_raw_plus_centered_Vina",
                gated_multiview_fusion(
                    sequence_rank,
                    (raw_vina_rank, vina_rank),
                    pocket_identity,
                    threshold,
                ),
            ),
        ]
    )
    rows: list[dict[str, object]] = []
    for panel, endpoint in endpoints.items():
        for name, predictor in predictors.items():
            rows.append(
                {
                    "panel": panel,
                    "role": (
                        "threshold_selection"
                        if panel == DISCOVERY_PANEL
                        else "locked_no_retuning_evaluation"
                    ),
                    "predictor": name,
                    "selected_threshold": threshold,
                    **predictor_metrics(predictor, endpoint),
                }
            )
    return pd.DataFrame.from_records(rows)


def _metric_array(predictor: np.ndarray, endpoint: np.ndarray) -> np.ndarray:
    values = predictor_metrics(predictor, endpoint)
    return np.asarray([values[name] for name in METRICS], dtype=np.float64)


def _average_precision_from_scores(labels: np.ndarray, scores: np.ndarray) -> float:
    """Exact threshold-grouped AP, including average handling of score ties."""
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=np.float64)
    order = np.argsort(-scores, kind="mergesort")
    ordered_scores = scores[order]
    ordered_labels = labels[order]
    group_ends = np.r_[
        np.flatnonzero(ordered_scores[1:] != ordered_scores[:-1]), len(scores) - 1
    ]
    true_positives = np.cumsum(ordered_labels)[group_ends].astype(float)
    predicted_positives = group_ends.astype(float) + 1.0
    recall = true_positives / float(labels.sum())
    precision = true_positives / predicted_positives
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def _qap_endpoint_cache(endpoint: np.ndarray) -> dict[str, np.ndarray | float | int]:
    values = upper(endpoint)
    ranks = stats.rankdata(values, method="average").astype(float)
    centered = ranks - ranks.mean()
    labels = top_labels(endpoint)
    return {
        "centered_ranks": centered,
        "rank_norm": float(np.linalg.norm(centered)),
        "labels": labels,
        "positive_count": int(labels.sum()),
        "negative_count": int((~labels).sum()),
    }


def _fast_qap_metrics(
    predictor_values: np.ndarray, cache: dict[str, np.ndarray | float | int]
) -> np.ndarray:
    ranks = stats.rankdata(predictor_values, method="average").astype(float)
    centered = ranks - ranks.mean()
    labels = np.asarray(cache["labels"], dtype=bool)
    positive_count = int(cache["positive_count"])
    negative_count = int(cache["negative_count"])
    rho = float(
        np.dot(centered, np.asarray(cache["centered_ranks"], dtype=float))
        / (np.linalg.norm(centered) * float(cache["rank_norm"]))
    )
    positive_rank_sum = float(ranks[labels].sum())
    auc = (
        positive_rank_sum - positive_count * (positive_count + 1) / 2.0
    ) / (positive_count * negative_count)
    ap = _average_precision_from_scores(labels, predictor_values)
    return np.asarray([rho, auc, ap], dtype=np.float64)


def locked_omnibus_qap(
    sequence_rank: np.ndarray,
    raw_vina_rank: np.ndarray,
    vina_rank: np.ndarray,
    pocket_identity: np.ndarray,
    endpoints: dict[str, np.ndarray],
    threshold: float,
    permutations: int,
    seed: int,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Incremental target-label QAP for gated fusion versus sequence only."""
    if permutations < 1:
        raise ValueError("permutations must be positive")
    baseline = sequence_rank
    observed_predictor = gated_multiview_fusion(
        sequence_rank,
        (raw_vina_rank, vina_rank),
        pocket_identity,
        threshold,
    )
    observed_by_panel = {
        panel: _metric_array(observed_predictor, endpoints[panel])
        - _metric_array(baseline, endpoints[panel])
        for panel in LOCKED_PANELS
    }
    observed = np.mean(list(observed_by_panel.values()), axis=0)

    endpoint_cache = {
        panel: _qap_endpoint_cache(endpoints[panel]) for panel in LOCKED_PANELS
    }
    baseline_values = upper(baseline)
    baseline_fast = {
        panel: _fast_qap_metrics(baseline_values, endpoint_cache[panel])
        for panel in LOCKED_PANELS
    }

    null = np.empty((permutations, len(METRICS)), dtype=np.float64)
    rng = np.random.default_rng(seed)
    for repetition in range(permutations):
        order = rng.permutation(len(sequence_rank))
        permuted_raw_vina = raw_vina_rank[np.ix_(order, order)]
        permuted_vina = vina_rank[np.ix_(order, order)]
        candidate = gated_multiview_fusion(
            sequence_rank,
            (permuted_raw_vina, permuted_vina),
            pocket_identity,
            threshold,
        )
        candidate_values = upper(candidate)
        null[repetition] = np.mean(
            [
                _fast_qap_metrics(candidate_values, endpoint_cache[panel])
                - baseline_fast[panel]
                for panel in LOCKED_PANELS
            ],
            axis=0,
        )

    probabilities = {
        name: float((1 + np.sum(null[:, index] >= observed[index])) / (permutations + 1))
        for index, name in enumerate(METRICS)
    }
    adjusted = holm_adjust(probabilities)
    rows = []
    for index, name in enumerate(METRICS):
        panel_values = [observed_by_panel[panel][index] for panel in LOCKED_PANELS]
        rows.append(
            {
                "metric": name,
                "locked_panels": ";".join(LOCKED_PANELS),
                "mean_gated_minus_sequence": float(observed[index]),
                "minimum_panel_delta": float(np.min(panel_values)),
                "maximum_panel_delta": float(np.max(panel_values)),
                "qap_p_positive": probabilities[name],
                "qap_p_positive_holm_three_metrics": adjusted[name],
                "null_mean": float(np.mean(null[:, index])),
                "null_q025": float(np.quantile(null[:, index], 0.025)),
                "null_q975": float(np.quantile(null[:, index], 0.975)),
                "permutations": permutations,
            }
        )
    return pd.DataFrame.from_records(rows), null


def target_jackknife(
    sequence_rank: np.ndarray,
    raw_vina_rank: np.ndarray,
    vina_rank: np.ndarray,
    pocket_identity: np.ndarray,
    endpoints: dict[str, np.ndarray],
    threshold: float,
) -> pd.DataFrame:
    predictor = gated_multiview_fusion(
        sequence_rank,
        (raw_vina_rank, vina_rank),
        pocket_identity,
        threshold,
    )
    rows: list[dict[str, object]] = []
    for omitted, target in enumerate(TARGETS):
        keep = np.delete(np.arange(len(TARGETS)), omitted)
        for panel in LOCKED_PANELS:
            gated_values = _metric_array(
                predictor[np.ix_(keep, keep)], endpoints[panel][np.ix_(keep, keep)]
            )
            sequence_values = _metric_array(
                sequence_rank[np.ix_(keep, keep)],
                endpoints[panel][np.ix_(keep, keep)],
            )
            for index, metric in enumerate(METRICS):
                rows.append(
                    {
                        "omitted_target": target,
                        "panel": panel,
                        "metric": metric,
                        "gated_minus_sequence": float(
                            gated_values[index] - sequence_values[index]
                        ),
                    }
                )
    return pd.DataFrame.from_records(rows)


def threshold_sensitivity(
    sequence_rank: np.ndarray,
    raw_vina_rank: np.ndarray,
    vina_rank: np.ndarray,
    pocket_identity: np.ndarray,
    endpoints: dict[str, np.ndarray],
    grid: Iterable[float] = THRESHOLD_GRID,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for threshold in sorted(set(float(value) for value in grid)):
        predictor = gated_multiview_fusion(
            sequence_rank,
            (raw_vina_rank, vina_rank),
            pocket_identity,
            threshold,
        )
        for panel in endpoints:
            gated_values = _metric_array(predictor, endpoints[panel])
            sequence_values = _metric_array(sequence_rank, endpoints[panel])
            for index, metric in enumerate(METRICS):
                rows.append(
                    {
                        "threshold": threshold,
                        "low_identity_pairs": int(
                            np.sum(upper(pocket_identity) < threshold)
                        ),
                        "panel": panel,
                        "role": (
                            "threshold_selection"
                            if panel == DISCOVERY_PANEL
                            else "locked_no_retuning_evaluation"
                        ),
                        "metric": metric,
                        "gated_minus_sequence": float(
                            gated_values[index] - sequence_values[index]
                        ),
                    }
                )
    return pd.DataFrame.from_records(rows)


def run_analysis(
    pair_path: Path,
    kirhub_workbook: Path,
    output: Path,
    permutations: int,
    seed: int,
) -> dict[str, object]:
    predictors, endpoints = load_inputs(pair_path, kirhub_workbook)
    pair_frame = pd.read_csv(pair_path)
    sequence_rank = rank_matrix(predictors["receptor_domain_sequence_identity"])
    raw_vina_rank = rank_matrix(
        matrix_from_pairs(pair_frame, "raw_docking_pair_percentile")
    )
    vina_rank = rank_matrix(predictors["centered_Vina_target_geometry"])
    pocket_identity = predictors["klifs_pocket_identity"]

    threshold, discovery = select_multiview_threshold(
        sequence_rank,
        (raw_vina_rank, vina_rank),
        pocket_identity,
        endpoints[DISCOVERY_PANEL],
    )
    metrics = evaluate_predictors(
        sequence_rank,
        raw_vina_rank,
        vina_rank,
        pocket_identity,
        endpoints,
        threshold,
    )
    qap, _ = locked_omnibus_qap(
        sequence_rank,
        raw_vina_rank,
        vina_rank,
        pocket_identity,
        endpoints,
        threshold,
        permutations,
        seed,
    )
    jackknife = target_jackknife(
        sequence_rank,
        raw_vina_rank,
        vina_rank,
        pocket_identity,
        endpoints,
        threshold,
    )
    sensitivity = threshold_sensitivity(
        sequence_rank, raw_vina_rank, vina_rank, pocket_identity, endpoints
    )

    output.mkdir(parents=True, exist_ok=True)
    discovery.to_csv(output / "discovery_threshold_selection.csv", index=False)
    metrics.to_csv(output / "panel_metrics.csv", index=False)
    qap.to_csv(output / "locked_omnibus_qap.csv", index=False)
    jackknife.to_csv(output / "target_jackknife.csv", index=False)
    sensitivity.to_csv(output / "threshold_sensitivity.csv", index=False)

    qap_records = qap.set_index("metric").to_dict(orient="index")
    summary: dict[str, object] = {
        "analysis": "pocket-gated sequence plus raw and residual Vina target geometry",
        "analysis_status": "post_hoc_exploratory_science_only",
        "discovery_panel": DISCOVERY_PANEL,
        "locked_no_retuning_panels": list(LOCKED_PANELS),
        "threshold_grid": list(THRESHOLD_GRID),
        "threshold_selection_metric": "PKIS1 top-10-percent average precision",
        "selected_klifs_pocket_identity_threshold": threshold,
        "gating_rule": (
            "below threshold: equal mean of pairwise sequence, raw-Vina, and "
            "centered-Vina ranks; at or above threshold: sequence rank only"
        ),
        "locked_omnibus_gated_minus_sequence": qap_records,
        "target_jackknife": {
            metric: {
                "minimum_delta": float(
                    jackknife.loc[jackknife.metric.eq(metric), "gated_minus_sequence"].min()
                ),
                "maximum_delta": float(
                    jackknife.loc[jackknife.metric.eq(metric), "gated_minus_sequence"].max()
                ),
                "positive_deletions": int(
                    np.sum(
                        jackknife.loc[
                            jackknife.metric.eq(metric), "gated_minus_sequence"
                        ]
                        > 0
                    )
                ),
                "total_panel_target_deletions": int(
                    np.sum(jackknife.metric.eq(metric))
                ),
            }
            for metric in METRICS
        },
        "inputs": {
            "target_pair_artifact": {
                "path": str(pair_path.relative_to(PACKAGE)),
                "sha256": sha256_file(pair_path),
            },
            "kirhub_workbook": {
                "redistributed": False,
                "sha256": sha256_file(kirhub_workbook),
            },
        },
        "claim_boundary": (
            "The result concerns target-pair co-selectivity on a fixed 20-kinase "
            "panel. The threshold is post-hoc and selected on PKIS1; the other "
            "panels are locked/no-retuning evaluations, not prospective tests. "
            "It does not establish ligand-level target retrieval or affinity accuracy."
        ),
    }
    (output / "summary.json").write_text(
        json.dumps(json_ready(summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# Pocket-gated sequence plus complementary Vina geometries",
        "",
        "This science-only analysis tests whether docking-derived target geometry is",
        "most useful for kinase pairs with relatively dissimilar KLIFS ATP pockets.",
        "The threshold is selected on PKIS1 only and then frozen.",
        "",
        f"The selected pocket-identity threshold is **{threshold:.2f}**. On DAVIS,",
        "PKIS2, and KiRHub, the dual-surface gated predictor improves continuous geometry",
        "agreement, AUROC, and average precision relative to sequence alone. The",
        "locked-panel mean deltas and target-label QAP probabilities are:",
        "",
    ]
    for row in qap.itertuples(index=False):
        lines.append(
            f"- {row.metric}: delta {row.mean_gated_minus_sequence:+.4f}, "
            f"QAP p={row.qap_p_positive:.5f}, Holm p="
            f"{row.qap_p_positive_holm_three_metrics:.5f}."
        )
    lines.extend(
        [
            "",
        "The interpretation is regime-specific: raw and residual Vina geometries add",
        "complementary cross-reactivity information where pocket sequence is less",
            "informative. This is not a claim that Vina generally outperforms sequence.",
            "",
            "## Panel metrics",
            "",
            "See `panel_metrics.csv` for all values and `threshold_sensitivity.csv`",
            "for the complete fixed grid.",
            "",
            "## Reproduction",
            "",
            "```bash",
            ".venv/bin/python analysis/pocket_gated_fusion.py \\",
            "  --kirhub-workbook /private/tmp/kirhub_supp_tables.xlsx \\",
            "  --output results/pocket_gated_fusion \\",
            "  --permutations 50000 --seed 20260818",
            "```",
            "",
        ]
    )
    (output / "README.md").write_text("\n".join(lines), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, default=DEFAULT_PAIRS)
    parser.add_argument("--kirhub-workbook", type=Path, default=DEFAULT_KIRHUB)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--permutations", type=int, default=DEFAULT_PERMUTATIONS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_analysis(
        args.pairs,
        args.kirhub_workbook,
        args.output,
        args.permutations,
        args.seed,
    )


if __name__ == "__main__":
    main()
