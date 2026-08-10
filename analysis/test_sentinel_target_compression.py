from __future__ import annotations

import numpy as np
import pandas as pd

from sentinel_target_compression import (
    PRIMARY_METRICS,
    _metrics,
    _sample_family_diverse_targets,
    paired_qr_comparisons,
    selection_stability,
)


def test_family_diverse_sampler_maximizes_coverage() -> None:
    families = np.asarray(["A", "A", "B", "B", "C", "C"], dtype=object)
    rng = np.random.default_rng(7)
    for _ in range(20):
        panel = _sample_family_diverse_targets(families, 3, rng)
        assert len(panel) == 3
        assert len(set(families[panel])) == 3
    panel = _sample_family_diverse_targets(families, 5, rng)
    assert len(panel) == 5
    assert len(set(families[panel])) == 3


def test_full_metrics_credit_observed_sentinels() -> None:
    true = np.asarray(
        [[-3.0, -2.0, -1.0], [-1.0, -3.0, -2.0], [-2.0, -1.0, -3.0]]
    )
    prediction = true.copy()
    prediction[:, 1:] += 0.5
    result = _metrics(true, prediction, np.asarray([1, 2]))
    assert result["full_target_r2"] > result["held_out_target_r2"]
    assert 0.0 <= result["full_top5_overlap"] <= 1.0


def test_selection_stability_counts_consensus() -> None:
    records = []
    panels = {1: ["A", "B"], 2: ["A", "C"], 3: ["A", "B"]}
    for fold, targets in panels.items():
        for rank, target in enumerate(targets, start=1):
            records.append(
                {
                    "fold": fold,
                    "selection_strategy": "raw_qr",
                    "selection_rank": rank,
                    "target": target,
                }
            )
    result = selection_stability(pd.DataFrame(records), (2,), folds=3).iloc[0]
    assert result["targets_selected_in_all_folds"] == 1
    assert result["union_targets_across_folds"] == 3
    assert np.isclose(result["mean_pairwise_fold_jaccard"], 5.0 / 9.0)


def test_paired_comparison_uses_repeat_level_fold_means() -> None:
    records = []
    qr_methods = (
        "raw_qr_sentinels_plus_descriptors",
        "residual_qr_sentinels_plus_descriptors",
    )
    random_methods = (
        "random_sentinels_plus_descriptors",
        "family_diverse_random_sentinels_plus_descriptors",
    )
    for fold in (1, 2):
        for method in qr_methods:
            row = {"fold": fold, "k": 5, "method": method, "random_repeat": -1}
            row.update({metric: 0.8 for metric in PRIMARY_METRICS})
            records.append(row)
        for repeat, value in ((0, 0.7), (1, 0.9)):
            for method in random_methods:
                row = {
                    "fold": fold,
                    "k": 5,
                    "method": method,
                    "random_repeat": repeat,
                }
                row.update({metric: value for metric in PRIMARY_METRICS})
                records.append(row)
    detail, summary = paired_qr_comparisons(pd.DataFrame(records))
    assert len(detail) == 2 * 2 * 2 * len(PRIMARY_METRICS)
    selected = summary[
        (summary["qr_method"] == qr_methods[0])
        & (summary["random_method"] == random_methods[0])
        & (summary["metric"] == "full_target_r2")
    ].iloc[0]
    assert np.isclose(selected["mean_qr_minus_random"], 0.0)
    assert np.isclose(selected["one_sided_monte_carlo_p"], 2.0 / 3.0)
