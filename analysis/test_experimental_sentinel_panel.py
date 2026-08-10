from __future__ import annotations

import numpy as np

from experimental_sentinel_panel import deim_selection, sequence_diverse_order


def test_deim_selection_uses_leading_mode_leverage() -> None:
    # Target zero carries the dominant eigenmode; the second sensor is selected
    # from the independent target-two direction.
    correlation = np.asarray(
        [
            [1.0, 0.9, 0.0],
            [0.9, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    selected = deim_selection(correlation, 2)
    assert 2 in selected
    assert len(set(selected.tolist())) == 2


def test_sequence_diverse_order_is_complete_and_deterministic() -> None:
    identity = np.asarray(
        [
            [1.0, 0.9, 0.2, 0.1],
            [0.9, 1.0, 0.3, 0.2],
            [0.2, 0.3, 1.0, 0.8],
            [0.1, 0.2, 0.8, 1.0],
        ]
    )
    first = sequence_diverse_order(identity)
    second = sequence_diverse_order(identity)
    assert np.array_equal(first, second)
    assert sorted(first.tolist()) == [0, 1, 2, 3]
    # The first two entries must span the two clearly separated clusters.
    assert (first[0] < 2) != (first[1] < 2)


def test_sequence_diverse_order_rejects_invalid_similarity() -> None:
    invalid = np.asarray([[1.0, 1.1], [1.1, 1.0]])
    try:
        sequence_diverse_order(invalid)
    except ValueError as error:
        assert "[0, 1]" in str(error)
    else:  # pragma: no cover
        raise AssertionError("invalid sequence identity was accepted")
