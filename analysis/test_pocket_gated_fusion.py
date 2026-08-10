from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score

import pocket_gated_fusion as gated


def test_gated_fusion_uses_vina_only_below_threshold() -> None:
    sequence = np.asarray(
        [[1.0, 2.0, 3.0], [2.0, 1.0, 4.0], [3.0, 4.0, 1.0]]
    )
    vina = np.asarray(
        [[1.0, 8.0, 9.0], [8.0, 1.0, 10.0], [9.0, 10.0, 1.0]]
    )
    pocket = np.asarray(
        [[1.0, 0.2, 0.8], [0.2, 1.0, 0.3], [0.8, 0.3, 1.0]]
    )
    result = gated.gated_fusion(sequence, vina, pocket, 0.5)
    assert result[0, 1] == 5.0
    assert result[1, 2] == 7.0
    assert result[0, 2] == 3.0
    assert np.allclose(result, result.T)


def test_multiview_gate_averages_all_views_only_below_threshold() -> None:
    sequence = np.asarray([[1.0, 3.0], [3.0, 1.0]])
    raw = np.asarray([[1.0, 6.0], [6.0, 1.0]])
    centered = np.asarray([[1.0, 9.0], [9.0, 1.0]])
    low_pocket = np.asarray([[1.0, 0.2], [0.2, 1.0]])
    high_pocket = np.asarray([[1.0, 0.8], [0.8, 1.0]])
    low = gated.gated_multiview_fusion(
        sequence, (raw, centered), low_pocket, 0.5
    )
    high = gated.gated_multiview_fusion(
        sequence, (raw, centered), high_pocket, 0.5
    )
    assert low[0, 1] == 6.0
    assert high[0, 1] == 3.0


def test_threshold_selection_uses_average_precision() -> None:
    rng = np.random.default_rng(4)
    sequence = gated.rank_matrix(np.corrcoef(rng.normal(size=(20, 6))))
    vina = gated.rank_matrix(np.corrcoef(rng.normal(size=(20, 6))))
    pocket = np.corrcoef(rng.normal(size=(20, 6)))
    endpoint = np.corrcoef(rng.normal(size=(20, 6)))
    threshold, frame = gated.select_threshold(
        sequence, vina, pocket, endpoint, grid=(0.2, 0.4, 0.6)
    )
    expected = frame.sort_values(
        ["average_precision", "threshold"],
        ascending=[False, True],
        kind="mergesort",
    ).iloc[0].threshold
    assert threshold == expected


def test_fast_average_precision_matches_sklearn_with_ties() -> None:
    labels = np.asarray([True, False, True, False, False, True])
    scores = np.asarray([0.8, 0.8, 0.4, 0.2, 0.2, 0.1])
    observed = gated._average_precision_from_scores(labels, scores)
    expected = average_precision_score(labels, scores)
    assert np.isclose(observed, expected)


def test_frozen_science_artifact() -> None:
    package = Path(__file__).resolve().parents[1]
    summary_path = package / "results" / "pocket_gated_fusion" / "summary.json"
    if not summary_path.exists():
        return
    import json

    summary = json.loads(summary_path.read_text())
    assert summary["selected_klifs_pocket_identity_threshold"] == 0.55
    omnibus = summary["locked_omnibus_gated_minus_sequence"]
    assert omnibus["continuous_spearman"]["mean_gated_minus_sequence"] > 0.10
    assert omnibus["roc_auc"]["mean_gated_minus_sequence"] > 0.12
    assert omnibus["average_precision"]["mean_gated_minus_sequence"] > 0.07
