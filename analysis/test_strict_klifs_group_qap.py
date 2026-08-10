from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

try:
    from . import strict_klifs_group_qap as strict
except ImportError:  # pragma: no cover
    import strict_klifs_group_qap as strict  # type: ignore


def test_restricted_orders_are_complete_within_block_bijections() -> None:
    labels = np.asarray(["A", "A", "A", "B", "B", "C"], dtype=object)
    first = strict.restricted_target_orders(labels, 257, seed=19)
    second = strict.restricted_target_orders(labels, 257, seed=19)
    assert np.array_equal(first, second)
    assert first.shape == (257, 6)
    for order in first:
        assert np.array_equal(np.sort(order), np.arange(6))
        assert np.array_equal(labels[order], labels)
        assert order[5] == 5


def test_exact_small_block_enumeration_and_pair_maps() -> None:
    labels = np.asarray(["A", "A", "A", "B", "B", "B", "C"], dtype=object)
    orders = strict.exact_restricted_target_orders(
        labels, active_blocks=("A", "B")
    )
    assert orders.shape == (36, 7)
    assert len(np.unique(orders, axis=0)) == 36
    assert np.all(orders[:, 6] == 6)
    maps = strict.orders_to_pair_maps(orders)
    assert maps.shape == (36, 21)


def test_restricted_partial_qap_exact_p_uses_full_enumeration() -> None:
    labels = np.asarray(["A", "A", "A", "B", "B", "B"], dtype=object)
    orders = strict.exact_restricted_target_orders(
        labels, active_blocks=("A", "B")
    )
    maps = strict.orders_to_pair_maps(orders)
    predictor = np.asarray(
        [
            [1.0, 0.9, 0.8, 0.1, 0.2, 0.3],
            [0.9, 1.0, 0.7, 0.2, 0.4, 0.1],
            [0.8, 0.7, 1.0, 0.3, 0.1, 0.4],
            [0.1, 0.2, 0.3, 1.0, 0.9, 0.8],
            [0.2, 0.4, 0.1, 0.9, 1.0, 0.7],
            [0.3, 0.1, 0.4, 0.8, 0.7, 1.0],
        ]
    )
    endpoint = predictor.copy()
    result = strict.restricted_partial_rank_qap(
        predictor,
        endpoint,
        [],
        maps,
        exact_enumeration=True,
    )
    assert result["partial_spearman"] == pytest.approx(1.0)
    assert result["restricted_qap_p_positive"] == pytest.approx(
        result["exceedances"] / 36
    )
    assert result["monte_carlo_standard_error"] == 0
    assert result["null_q05"] <= result["null_q95"]
    assert (
        result["additional_partial_spearman_needed_to_cross_one_sided_critical"]
        == 0.0
    )


def test_release_exchangeability_contract() -> None:
    annotations = strict.load_annotations(strict.DEFAULT_TARGET_ANNOTATIONS)
    blocks, strata, diagnostics = strict.exchangeability_diagnostics(annotations)
    sizes = dict(zip(blocks.klifs_group, blocks.target_count))
    assert sizes == {"TK": 11, "AGC": 3, "CMGC": 3, "STE": 1, "CAMK": 1, "Other": 1}
    assert diagnostics["exact_KLIFS_group_relabellings"] == 1_437_004_800
    assert diagnostics["movable_targets"] == 17
    assert diagnostics["movable_target_blocks"] == 3
    assert diagnostics["movable_target_block_sizes"] == [11, 3, 3]
    assert diagnostics["fixed_singleton_targets"] == 3
    assert diagnostics["movable_pair_positions"] == 187
    assert diagnostics["fixed_singleton_to_singleton_pair_positions"] == 3
    assert diagnostics["TK_involving_pair_positions"] == 154
    assert diagnostics["exact_KLIFS_family_relabellings"] == 16
    assert diagnostics["minimum_exact_one_sided_family_p"] == pytest.approx(0.0625)
    effective = diagnostics["effective_exchangeable_units"]
    assert effective["scalar_effective_sample_size"] is None
    assert effective["block_size_vector"] == [11, 3, 3]
    assert int(strata.pair_positions.sum()) == 190


def test_release_summary_preserves_conditional_boundary() -> None:
    path = strict.DEFAULT_OUTPUT / "summary.json"
    if not path.exists():
        pytest.skip("release result has not yet been generated")
    with path.open() as handle:
        summary = json.load(handle)
    assert summary["analysis_status"] == "exploratory_conditional_sensitivity"
    assert summary["exchangeability_diagnostics"]["movable_targets"] == 17
    assert "exchangeability" in summary["validity_boundary"]
    assert "0.0625" in summary["small_group_boundary"]
    assert "not power-based" in summary["resolution_boundary"]
    for endpoint in ("old_three_panel_mean", "KiRHub"):
        record = summary["primary_results"][endpoint]["unadjusted"]
        assert -1 <= record["partial_spearman"] <= 1
        assert 0 < record["restricted_qap_p_positive"] <= 1
        assert record["one_sided_null_interval_90"][0] <= record[
            "one_sided_null_interval_90"
        ][1]
        assert record[
            "additional_partial_spearman_needed_to_cross_one_sided_critical"
        ] >= 0
    for source in ("target_annotations", "target_pairs"):
        source_path = summary["sources"][source]["path"]
        assert not Path(source_path).is_absolute()
        assert str(strict.PACKAGE) not in json.dumps(summary)
