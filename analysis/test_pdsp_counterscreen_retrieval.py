from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score


ANALYSIS = Path(__file__).resolve().parent
if str(ANALYSIS) not in sys.path:
    sys.path.insert(0, str(ANALYSIS))

import pdsp_counterscreen_retrieval as audit


def test_standard_inchi_key_rejects_missing_and_zero_atom() -> None:
    assert audit.standard_inchi_key(None) is None
    assert audit.standard_inchi_key("") is None
    key = audit.standard_inchi_key("CCO")
    assert key is not None and len(key) == 27


def test_top_fraction_labels_has_frozen_ceiling_count() -> None:
    labels = audit.top_fraction_labels(np.arange(21), 0.10)
    assert labels.sum() == 3
    assert labels[-3:].all()


def test_fractional_tie_metrics() -> None:
    scores = np.asarray([3.0, 2.0, 2.0, 1.0])
    assert audit.expected_topk_inclusion(scores, 1, 2) == pytest.approx(0.5)
    assert audit.expected_reciprocal_rank(scores, 1) == pytest.approx((1 / 2 + 1 / 3) / 2)


def test_fast_auc_matches_sklearn_with_ties() -> None:
    labels = np.asarray([True, False, True, False, False])
    scores = np.asarray([1.0, 0.0, 0.5, 0.5, -1.0])
    assert audit.fast_auc(labels, scores) == pytest.approx(roc_auc_score(labels, scores))


def test_fixed_fusion_does_not_use_endpoint() -> None:
    frame = pd.DataFrame(
        {
            "experimental_spearman": [0.9, 0.1, -0.2],
            "full_sequence_identity": [0.2, 0.8, 0.4],
            "same_curated_family": [0.0, 1.0, 0.0],
            "residual_docking": [0.3, -0.2, 0.7],
        }
    )
    first = audit.add_fixed_fusions(frame)
    changed = frame.copy()
    changed["experimental_spearman"] = [-4.0, 12.0, 2.0]
    second = audit.add_fixed_fusions(changed)
    for column in (
        "equal_rank_sequence_family",
        "equal_rank_sequence_residual",
        "equal_rank_sequence_family_residual",
    ):
        np.testing.assert_allclose(first[column], second[column])


def test_family_preserving_order_never_crosses_family() -> None:
    targets = ["HTR1A", "HTR1B", "DRD1", "DRD2", "HTR3A"]
    labels = audit.family_labels(targets)
    order = audit.family_preserving_order(np.random.default_rng(7), targets, labels)
    for destination, source in enumerate(order):
        assert labels[targets[destination]] == labels[targets[source]]
