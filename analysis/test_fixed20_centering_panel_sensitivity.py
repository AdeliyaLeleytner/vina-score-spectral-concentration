from __future__ import annotations

from collections import OrderedDict
import json

import numpy as np
import pytest

import fixed20_centering_panel_sensitivity as sensitivity


def _synthetic_experiments(rng: np.random.Generator) -> OrderedDict[str, np.ndarray]:
    return OrderedDict(
        (panel, rng.normal(size=(40 + index, 20)))
        for index, panel in enumerate(("DAVIS", "PKIS2", "PKIS1", "KiRHub"))
    )


def test_strict_support_and_delta_identity() -> None:
    rng = np.random.default_rng(14)
    score_columns = tuple(sensitivity.TARGETS20) + tuple(
        f"context_{index}" for index in range(5)
    )
    reference = rng.normal(size=(180, len(score_columns)))
    summary, table, pairs = sensitivity.evaluate_fixed20(
        reference,
        score_columns,
        _synthetic_experiments(rng),
        permutations=9,
        seed=7,
    )
    assert summary["targets"] == list(sensitivity.TARGETS20)
    assert summary["target_pairs"] == 190
    assert list(table.panel) == ["DAVIS", "PKIS2", "PKIS1", "KiRHub"]
    assert set(table.targets) == {20}
    assert set(table.target_pairs) == {190}
    assert len(pairs) == 190
    assert len(pairs[["target_a", "target_b"]].drop_duplicates()) == 190
    assert {
        "DAVIS_experimental_centered_correlation",
        "PKIS2_experimental_centered_correlation",
        "PKIS1_experimental_centered_correlation",
        "KiRHub_experimental_centered_correlation",
    }.issubset(pairs)
    for row in table.itertuples(index=False):
        observed_delta = (
            row.all_58_centered_then_extract_concordance_spearman
            - row.local_20_centered_concordance_spearman
        )
        assert abs(observed_delta - row.all_58_minus_local_20_concordance) < 1e-12


def test_identical_centering_panels_give_identical_geometry() -> None:
    rng = np.random.default_rng(27)
    reference = rng.normal(size=(150, 20))
    summary, table, pairs = sensitivity.evaluate_fixed20(
        reference,
        tuple(sensitivity.TARGETS20),
        _synthetic_experiments(rng),
        permutations=7,
        seed=11,
    )
    agreement = summary["docking_geometry_agreement"]
    assert agreement["spearman_of_target_pair_correlations"] == pytest.approx(1.0)
    assert agreement["maximum_absolute_correlation_difference"] < 1e-12
    assert np.max(np.abs(table.all_58_minus_local_20_concordance)) < 1e-12
    assert len(pairs) == 190


def test_rejects_mixed_or_misaligned_support() -> None:
    rng = np.random.default_rng(33)
    reference = rng.normal(size=(100, 22))
    score_columns = tuple(sensitivity.TARGETS20) + ("extra_a", "extra_b")
    experiments = _synthetic_experiments(rng)
    experiments["DAVIS"] = rng.normal(size=(40, 21))
    with pytest.raises(ValueError, match="ligand-by-20-target"):
        sensitivity.evaluate_fixed20(
            reference,
            score_columns,
            experiments,
            permutations=3,
            seed=1,
        )


def test_frozen_output_contract_if_present() -> None:
    path = sensitivity.DEFAULT_OUTPUT / "summary.json"
    if not path.exists():
        pytest.skip("strict fixed-20 centering-panel output has not been generated")
    with path.open() as handle:
        summary = json.load(handle)
    assert summary["targets"] == list(sensitivity.TARGETS20)
    assert summary["target_pairs"] == 190
    assert set(summary["experimental_panel_results"]) == {
        "DAVIS",
        "PKIS2",
        "PKIS1",
        "KiRHub",
    }
    assert {
        result["targets"] for result in summary["experimental_panel_results"].values()
    } == {20}
    assert summary["support"]["all_target_count"] == 58
