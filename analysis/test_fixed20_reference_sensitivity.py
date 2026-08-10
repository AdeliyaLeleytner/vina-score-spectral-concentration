from __future__ import annotations

import json
from math import comb

import numpy as np
import pandas as pd
import pytest

import fixed20_reference_sensitivity as sensitivity


METRICS = (
    "raw_docking__raw_experiment",
    "raw_docking__centered_experiment",
    "centered_docking__raw_experiment",
    "centered_docking__centered_experiment",
)
PANELS = ("DAVIS", "PKIS2", "PKIS1", "KiRHub")


def test_summarize_uses_sample_standard_deviation() -> None:
    values = [1.0, 2.0, 4.0]
    observed = sensitivity.summarize(values)
    assert observed == {
        "n": 3,
        "mean": pytest.approx(np.mean(values)),
        "minimum": 1.0,
        "maximum": 4.0,
        "standard_deviation": pytest.approx(np.std(values, ddof=1)),
    }
    assert sensitivity.summarize([2.5]) == {
        "n": 1,
        "mean": 2.5,
        "minimum": 2.5,
        "maximum": 2.5,
        "standard_deviation": 0.0,
    }


def test_concordances_recover_identical_raw_and_centered_geometries() -> None:
    rng = np.random.default_rng(42)
    reference = rng.normal(size=(250, 8))
    experiment = reference.copy()
    reference_before = reference.copy()
    experiment_before = experiment.copy()

    observed = sensitivity.concordances(reference, experiment)

    assert set(observed) == set(METRICS)
    assert observed["raw_docking__raw_experiment"] == pytest.approx(1.0)
    assert observed["centered_docking__centered_experiment"] == pytest.approx(1.0)
    assert all(np.isfinite(value) and -1.0 <= value <= 1.0 for value in observed.values())
    np.testing.assert_array_equal(reference, reference_before)
    np.testing.assert_array_equal(experiment, experiment_before)


def test_frozen_summary_contract_and_support_table_agree() -> None:
    summary_path = sensitivity.DEFAULT_OUTPUT / "summary.json"
    support_path = sensitivity.DEFAULT_OUTPUT / "support_concordance.csv"
    assert summary_path.exists(), "frozen fixed-20 sensitivity summary is missing"
    assert support_path.exists(), "frozen fixed-20 support table is missing"

    summary = json.loads(summary_path.read_text())
    support = pd.read_csv(support_path)

    assert summary["analysis"] == "strict common-20 reference-support sensitivity"
    assert summary["status"] == "exploratory_post_hoc"
    assert summary["targets"] == list(sensitivity.kirhub.TARGETS)
    assert len(summary["targets"]) == 20
    assert len(set(summary["targets"])) == 20
    assert summary["target_pairs"] == comb(20, 2) == 190

    reference = summary["reference"]
    assert reference == {
        "rows_after_connectivity_exclusion": 259_579,
        "excluded_connectivity_blocks": 1_050,
        "positive_scores_clipped_to_zero": True,
    }

    primary = summary["primary_full_reference"]
    assert tuple(primary) == PANELS
    assert all(set(primary[panel]) == set(METRICS) for panel in PANELS)
    assert all(
        np.isfinite(primary[panel][metric])
        and -1.0 <= primary[panel][metric] <= 1.0
        for panel in PANELS
        for metric in METRICS
    )
    expected_centered = {
        "DAVIS": 0.27979984866179414,
        "PKIS2": 0.287170806597646,
        "PKIS1": 0.1736426581287425,
        "KiRHub": 0.32158843881081417,
    }
    for panel, expected in expected_centered.items():
        assert primary[panel]["centered_docking__centered_experiment"] == pytest.approx(
            expected, abs=1e-12
        )

    scaffold = summary["same_cyclic_murcko_exclusion"]
    assert scaffold["experimental_panels_with_structures"] == [
        "DAVIS",
        "PKIS2",
        "PKIS1",
    ]
    assert scaffold["experimental_cyclic_scaffolds"] == 575
    assert scaffold["reference_rows_excluded"] == 3_039
    assert scaffold["reference_rows_remaining"] == 256_540
    assert (
        scaffold["reference_rows_excluded"] + scaffold["reference_rows_remaining"]
        == reference["rows_after_connectivity_exclusion"]
    )
    assert set(scaffold["panels"]) == {"DAVIS", "PKIS2", "PKIS1"}
    assert "unavailable" in scaffold["KiRHub_boundary"]
    for panel, values in scaffold["panels"].items():
        assert values["centered_delta_from_primary"] == pytest.approx(
            values["centered_docking__centered_experiment"]
            - primary[panel]["centered_docking__centered_experiment"],
            abs=1e-15,
        )
        assert values["raw_delta_from_primary"] == pytest.approx(
            values["raw_docking__raw_experiment"]
            - primary[panel]["raw_docking__raw_experiment"],
            abs=1e-15,
        )

    random_supports = summary["random_reference_supports"]
    assert random_supports["support_size"] == 15_000
    assert tuple(random_supports["seeds"]) == sensitivity.DEFAULT_SEEDS
    assert set(random_supports["panels"]) == set(PANELS)
    assert len(support) == len(PANELS) * len(sensitivity.DEFAULT_SEEDS)
    assert set(support["panel"]) == set(PANELS)
    assert set(support["seed"]) == set(sensitivity.DEFAULT_SEEDS)
    assert set(support["reference_ligands"]) == {15_000}
    assert not support.duplicated(["panel", "seed"]).any()

    for panel in PANELS:
        rows = support.loc[support["panel"] == panel]
        assert len(rows) == len(sensitivity.DEFAULT_SEEDS)
        for metric in METRICS:
            frozen = random_supports["panels"][panel][metric]
            recomputed = sensitivity.summarize(rows[metric].tolist())
            assert frozen["n"] == recomputed["n"] == 5
            for key in ("mean", "minimum", "maximum", "standard_deviation"):
                assert frozen[key] == pytest.approx(recomputed[key], abs=1e-15)
            assert frozen["minimum"] <= frozen["mean"] <= frozen["maximum"]
            assert frozen["standard_deviation"] >= 0.0

    assert "same 20 targets" in summary["claim_boundary"]
    assert "not biological validity" in summary["claim_boundary"]
