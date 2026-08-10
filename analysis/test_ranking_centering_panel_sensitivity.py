from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

import ranking_centering_panel_sensitivity as sensitivity


def _reference(seed: int = 9) -> tuple[pd.DataFrame, pd.Index, list[str]]:
    rng = np.random.default_rng(seed)
    columns = ["a", "b", "c", "context_1", "context_2"]
    index = pd.Index([f"train_{i}" for i in range(30)] + ["eval_1", "eval_2"])
    values = rng.normal(size=(len(index), len(columns)))
    values[-1, -1] = np.nan
    return pd.DataFrame(values, index=index, columns=columns), index[-2:], columns[:3]


def test_all_target_arm_matches_manual_external_contract() -> None:
    reference, evaluation_ids, selected = _reference()
    scores, metadata, rows = sensitivity.externally_fixed_row_panel_scores(
        reference, evaluation_ids, selected
    )

    train_raw = reference.drop(index=evaluation_ids)
    means = train_raw.mean(axis=0)
    train = train_raw.fillna(means)
    test = reference.loc[evaluation_ids].fillna(means)
    target_mean = train.mean(axis=0)
    grand = float(train.to_numpy().mean())
    train_residual = (
        train
        - target_mean
        - train.mean(axis=1).to_numpy()[:, None]
        + grand
    )
    scale = train_residual.std(axis=0, ddof=1)
    expected = (
        test[selected]
        - target_mean[selected]
        - test.mean(axis=1).to_numpy()[:, None]
        + grand
    ) / scale[selected]

    np.testing.assert_allclose(scores["all_44_row_mean"], expected.to_numpy())
    assert metadata["docking_training_ligands"] == 30
    assert metadata["all_target_count"] == 5
    assert metadata["local_target_count"] == 3
    assert metadata["targets_excluded_from_local_row_mean"] == [
        "context_1",
        "context_2",
    ]
    assert list(rows.inchikey) == list(evaluation_ids)


def test_only_row_mean_changes_before_fixed_target_scaling() -> None:
    reference, evaluation_ids, selected = _reference(seed=18)
    scores, _, rows = sensitivity.externally_fixed_row_panel_scores(
        reference, evaluation_ids, selected
    )

    train_raw = reference.drop(index=evaluation_ids)
    train = train_raw.fillna(train_raw.mean(axis=0))
    target_mean = train.mean(axis=0)
    grand = float(train.to_numpy().mean())
    train_residual = (
        train
        - target_mean
        - train.mean(axis=1).to_numpy()[:, None]
        + grand
    )
    scale = train_residual.std(axis=0, ddof=1)[selected].to_numpy()
    expected_delta = (
        -rows.local_minus_all_row_mean.to_numpy()[:, None] / scale[None, :]
    )
    observed_delta = (
        scores["local_evaluation_target_row_mean"] - scores["all_44_row_mean"]
    )
    np.testing.assert_allclose(observed_delta, expected_delta, atol=1e-14)


def test_identical_row_mean_panels_are_exactly_equivalent() -> None:
    reference, evaluation_ids, _ = _reference(seed=27)
    selected = list(reference.columns)
    scores, metadata, rows = sensitivity.externally_fixed_row_panel_scores(
        reference, evaluation_ids, selected
    )
    np.testing.assert_allclose(
        scores["all_44_row_mean"],
        scores["local_evaluation_target_row_mean"],
        atol=1e-14,
    )
    np.testing.assert_allclose(rows.local_minus_all_row_mean, 0.0, atol=1e-14)
    assert metadata["targets_excluded_from_local_row_mean"] == []


def test_rejects_unknown_or_duplicated_targets() -> None:
    reference, evaluation_ids, selected = _reference(seed=33)
    with pytest.raises(ValueError, match="not a subset"):
        sensitivity.externally_fixed_row_panel_scores(
            reference, evaluation_ids, selected + ["missing"]
        )
    with pytest.raises(ValueError, match="unique"):
        sensitivity.externally_fixed_row_panel_scores(
            reference, evaluation_ids, ["a", "a"]
        )


def test_frozen_output_contract_if_present() -> None:
    path = sensitivity.DEFAULT_OUTPUT / "summary.json"
    if not path.exists():
        pytest.skip("ranking centering-panel sensitivity has not been generated")
    with path.open() as handle:
        summary = json.load(handle)

    assert summary["support"]["ligands"] == 137
    assert summary["support"]["targets"] == 38
    assert summary["support"]["observed_experimental_cells"] == 691
    assert summary["support"]["eligible_nontied_within_ligand_target_pairs"] == 2522
    assert summary["fit_contract"]["targets_excluded_from_local_row_mean"] == [
        "1pbq",
        "2vt4",
        "6x3x",
        "7xnk",
        "8st0",
        "V1A",
    ]
    ranking = summary["ranking"]
    assert ranking["all_44_row_mean"][
        "mean_per_ligand_pairwise_accuracy"
    ] == pytest.approx(0.5639355480794093)
    assert ranking["local_38_row_mean"][
        "mean_per_ligand_pairwise_accuracy"
    ] == pytest.approx(0.5669823646682927)
    assert ranking["local_38_minus_all_44"][
        "plugin_mean_difference"
    ] == pytest.approx(0.0030468165888834346)
    assert ranking["changed_pair_predictions"] == 31
    assert summary["contract_audit"][
        "maximum_absolute_difference_from_released_all_44_two_way_residual"
    ] < 1e-12
