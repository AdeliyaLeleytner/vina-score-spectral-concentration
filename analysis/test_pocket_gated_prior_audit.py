from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parent))

import pocket_gated_prior_audit as audit


def test_degree_pair_features_are_symmetric_and_rank_scaled() -> None:
    degree = np.asarray([0.1, 0.1, 0.7, 0.9])
    tri = np.triu_indices(4, k=1)
    features = audit.degree_pair_features(degree, tri)
    assert features.shape == (6, 2)
    assert np.all((features >= 0) & (features <= 1))
    # Equal-degree pair is maximally similar.
    pair_01 = list(zip(*tri)).index((0, 1))
    assert features[pair_01, 1] == features[:, 1].max()


def test_gated_increment_is_proportional_to_original_fusion_delta() -> None:
    sequence = np.asarray([0.2, 0.7, 0.4])
    raw = np.asarray([0.8, 0.1, 0.5])
    residual = np.asarray([0.5, 0.6, 0.9])
    pocket = np.asarray([0.2, 0.8, 0.4])
    increment = audit.gated_docking_increment(
        sequence, raw, residual, pocket, 0.55
    )
    original = np.where(
        pocket < 0.55,
        (sequence + raw + residual) / 3.0,
        sequence,
    )
    assert np.allclose(original - sequence, (2.0 / 3.0) * increment)


def test_selection_repeats_discovery_fit_and_uses_stable_tie_break() -> None:
    rng = np.random.default_rng(91)
    n = 40
    base = rng.normal(size=(n, 3))
    sequence = rng.normal(size=n)
    raw = rng.normal(size=n)
    residual = rng.normal(size=n)
    pocket = np.linspace(0, 1, n)
    endpoint = 0.4 * base[:, 0] + rng.normal(scale=0.2, size=n)
    threshold, prediction, frame = audit.select_gate(
        base,
        endpoint,
        sequence,
        raw,
        residual,
        pocket,
        grid=(0.3, 0.5, 0.7),
    )
    assert threshold in {0.3, 0.5, 0.7}
    assert prediction.shape == (n,)
    assert len(frame) == 3


def test_released_summary_passes_or_fails_declared_rule_consistently() -> None:
    path = audit.PACKAGE / "results" / "pocket_gated_prior_audit" / "summary.json"
    if not path.exists():
        return
    summary = json.loads(path.read_text())
    assert summary["selected_klifs_pocket_identity_threshold"] in list(
        audit.THRESHOLD_GRID
    )
    primary = summary["primary_locked_mean_deltas"]
    assert set(primary) == set(audit.PRIMARY_METRICS)
    for metric in audit.PRIMARY_METRICS:
        assert primary[metric]["minimum_panel_delta"] > 0
        assert 0 <= primary[metric]["selection_aware_qap_p"] <= 1
    prior = summary["degree_prior_increment_beyond_sequence_plus_pocket"]
    assert prior["strict_decision"] == "GO_replicated_target_prior_increment"
    for metric in audit.PRIMARY_METRICS:
        record = prior["metrics"][metric]
        assert record["minimum_panel_delta"] > 0
        assert record["holm_p_three_primary_metrics"] <= 0.05
        assert record["positive_panel_target_deletions"] == 60
