from __future__ import annotations

import numpy as np

from sequence_docking_fusion import (
    continuous_spearman,
    equal_rank_fusion,
    incremental_qap,
    predictor_metrics,
    rank_matrix,
    target_jackknife,
)


def symmetric(values: np.ndarray, size: int) -> np.ndarray:
    tri = np.triu_indices(size, k=1)
    result = np.eye(size)
    result[tri] = values
    result[(tri[1], tri[0])] = values
    return result


def test_equal_rank_fusion_is_symmetric_and_outcome_free() -> None:
    first = symmetric(np.arange(15, dtype=float), 6)
    second = symmetric(np.arange(15, dtype=float)[::-1], 6)
    fused = equal_rank_fusion(first, second)
    assert np.allclose(fused, fused.T)
    assert np.allclose(np.diag(fused), 1.0)
    assert np.array_equal(rank_matrix(first), rank_matrix(first.copy()))


def test_predictor_metrics_are_finite() -> None:
    first = symmetric(np.arange(15, dtype=float), 6)
    endpoint = symmetric(np.asarray([0, 2, 1, 4, 3, 5, 8, 7, 6, 9, 11, 10, 12, 14, 13]), 6)
    metrics = predictor_metrics(first, endpoint, 0.20)
    assert metrics["target_pairs"] == 15
    assert metrics["positive_pairs"] == 3
    assert np.isfinite(metrics["continuous_spearman"])


def test_incremental_qap_is_deterministic() -> None:
    rng = np.random.default_rng(4)
    sequence = symmetric(rng.normal(size=15), 6)
    vina = symmetric(rng.normal(size=15), 6)
    endpoint = symmetric(rng.normal(size=15), 6)
    first, first_omnibus = incremental_qap(
        sequence, vina, {"DAVIS": endpoint, "PKIS2": endpoint, "KiRHub": endpoint}, 99, 8
    )
    second, second_omnibus = incremental_qap(
        sequence, vina, {"DAVIS": endpoint, "PKIS2": endpoint, "KiRHub": endpoint}, 99, 8
    )
    assert first.equals(second)
    assert first_omnibus.equals(second_omnibus)


def test_target_jackknife_has_one_record_per_target_and_panel(monkeypatch) -> None:
    import sequence_docking_fusion as module

    targets = tuple(f"T{index}" for index in range(6))
    monkeypatch.setattr(module, "TARGETS", targets)
    rng = np.random.default_rng(5)
    sequence = symmetric(rng.normal(size=15), 6)
    vina = symmetric(rng.normal(size=15), 6)
    endpoint = symmetric(rng.normal(size=15), 6)
    frame = target_jackknife(sequence, vina, {"DAVIS": endpoint})
    assert len(frame) == 6
    assert frame.remaining_targets.eq(5).all()
    assert np.isfinite(frame.fusion_minus_sequence_continuous_spearman).all()
    assert np.isfinite(continuous_spearman(sequence, endpoint))
