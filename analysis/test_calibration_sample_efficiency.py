from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parent))

import calibration_sample_efficiency as audit


def test_cluster_round_robin_is_complete_and_diverse_first() -> None:
    labels = np.asarray([0, 0, 0, 1, 1, 2, 3, 3])
    order = audit.cluster_round_robin_order(
        labels, np.random.default_rng(101)
    )
    assert sorted(order.tolist()) == list(range(len(labels)))
    first_round = labels[order[: len(np.unique(labels))]]
    assert len(np.unique(first_round)) == len(np.unique(labels))


def test_fit_from_calibration_uses_aligned_models() -> None:
    rng = np.random.default_rng(202)
    values = rng.normal(size=(30, 5))
    base = rng.normal(size=(10, 2))
    tri = np.triu_indices(5, k=1)
    fitted = audit.fit_from_calibration(values, base, tri, activity_threshold=0.0)
    assert fitted["endpoint"].shape == (10,)
    assert fitted["target_degree"].shape == (5,)
    assert fitted["baseline_prediction"].shape == (10,)
    assert fitted["augmented_prediction"].shape == (10,)
    assert np.isfinite(fitted["augmented_prediction"]).all()


def test_strict_decision_requires_both_tail_and_all_panel_fraction() -> None:
    rows = []
    for size in (40, 80):
        for metric in audit.PRIMARY_METRICS:
            rows.append(
                {
                    "sampling_scheme": "Butina_cluster_round_robin",
                    "sample_size": size,
                    "metric": metric,
                    "mean_locked_delta_q025": 0.01 if size == 80 else -0.01,
                    "fraction_positive_on_every_locked_panel": (
                        0.96 if size == 80 else 0.99
                    ),
                }
            )
    decision, smallest = audit._strict_small_panel_decision(
        pd.DataFrame.from_records(rows)
    )
    assert decision
    assert smallest == 80


def test_released_summary_obeys_declared_rule() -> None:
    path = audit.DEFAULT_OUTPUT / "summary.json"
    if not path.exists():
        return
    summary = json.loads(path.read_text())
    for source in summary["inputs"].values():
        assert not Path(source["path"]).is_absolute()
    assert summary["decision"] == "GO_small_diverse_calibration_panel"
    assert summary["smallest_sample_size_passing_rule"] <= 80
    repeated = summary["practical_80_repeated_diversity_sampling"]
    qap = summary["deterministic_MaxMin_80"]["target_label_QAP"]
    jackknife = summary["deterministic_MaxMin_80"]["target_jackknife"]
    for metric in audit.PRIMARY_METRICS:
        assert repeated[metric]["q025_mean_locked_delta"] > 0
        assert repeated[metric]["fraction_positive_on_every_locked_panel"] >= 0.95
        assert qap[metric]["minimum_locked_panel_delta"] > 0
        assert qap[metric]["holm_p"] <= 0.05
        assert jackknife[metric]["positive_panel_target_deletions"] == 60
