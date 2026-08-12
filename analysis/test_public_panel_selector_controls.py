from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from . import public_panel_selector_controls as selector
except ImportError:  # pragma: no cover
    import public_panel_selector_controls as selector  # type: ignore


def synthetic_scores(seed: int = 9) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    target_slopes = np.linspace(0.6, 1.8, 9)
    target_offsets = np.linspace(-1.0, 1.0, 9)
    interactions = rng.normal(size=(140, 3)) @ rng.normal(size=(3, 9))
    raw = (
        rng.normal(size=(140, 1)) * target_slopes[None, :]
        + target_offsets[None, :]
        + 0.35 * interactions
    )
    return raw[:60], raw[60:]


def test_raw_row_centering_precedes_residual_scaling() -> None:
    pilot, evaluation = synthetic_scores()
    transform = selector.pilot_transform(pilot, evaluation)
    expected_pilot_raw = pilot - pilot.mean(axis=1, keepdims=True)
    expected_eval_raw = evaluation - evaluation.mean(axis=1, keepdims=True)
    expected_means = expected_pilot_raw.mean(axis=0)
    expected_sds = expected_pilot_raw.std(axis=0, ddof=1)
    assert np.allclose(transform.pilot_residual_z, (expected_pilot_raw - expected_means) / expected_sds)
    assert np.allclose(transform.evaluation_residual_z, (expected_eval_raw - expected_means) / expected_sds)

    standardized_first = transform.pilot_raw_z - transform.pilot_raw_z.mean(
        axis=1, keepdims=True
    )
    assert not np.allclose(transform.pilot_residual_z, standardized_first)


def test_held_out_changes_cannot_change_pilot_selectors() -> None:
    pilot, evaluation = synthetic_scores()
    first = selector.pilot_transform(pilot, evaluation)
    perturbed = evaluation.copy()
    perturbed[:, 0] += np.linspace(-100.0, 100.0, len(perturbed))
    second = selector.pilot_transform(pilot, perturbed)
    assert np.array_equal(
        selector.residual_map_medoids(first, 4),
        selector.residual_map_medoids(second, 4),
    )
    assert np.array_equal(
        selector.pivoted_qr_panel(first.pilot_raw_z, 4),
        selector.pivoted_qr_panel(second.pilot_raw_z, 4),
    )
    assert np.allclose(first.pilot_raw_z, second.pilot_raw_z)
    assert np.allclose(first.pilot_residual_z, second.pilot_residual_z)


def test_qr_pivots_target_columns_and_is_nested() -> None:
    pilot, _ = synthetic_scores()
    transform = selector.pilot_transform(pilot, pilot[:25])
    panel4 = selector.pivoted_qr_panel(transform.pilot_raw_z, 4)
    panel6 = selector.pivoted_qr_panel(transform.pilot_raw_z, 6)
    assert len(panel4) == 4
    assert len(panel6) == 6
    assert set(panel4) <= set(panel6)
    assert panel6.min() >= 0 and panel6.max() < pilot.shape[1]


def test_random_panels_are_fixed_unique_and_sized() -> None:
    first = selector.fixed_random_panels(12, 4, 50, np.random.default_rng(123))
    second = selector.fixed_random_panels(12, 4, 50, np.random.default_rng(123))
    first_tuples = [tuple(panel) for panel in first]
    assert first_tuples == [tuple(panel) for panel in second]
    assert len(set(first_tuples)) == 50
    assert all(len(panel) == 4 for panel in first)


def test_ridge_uses_selected_inputs_and_returns_finite_prediction() -> None:
    pilot, evaluation = synthetic_scores()
    transform = selector.pilot_transform(pilot, evaluation)
    panel = np.array([0, 2, 5, 7])
    prediction, alpha, gcv = selector.ridge_gcv_predict(
        transform.pilot_raw_z[:, panel],
        transform.pilot_raw_z,
        transform.evaluation_raw_z[:, panel],
    )
    assert prediction.shape == transform.evaluation_raw_z.shape
    assert alpha in selector.RIDGE_ALPHAS
    assert np.isfinite(gcv)
    assert np.isfinite(prediction).all()


def test_prediction_policy_copies_selected_raw_scores_then_centers() -> None:
    pilot, evaluation = synthetic_scores()
    transform = selector.pilot_transform(pilot, evaluation)
    panel = np.array([0, 2, 5, 7])
    records = selector.prediction_records(
        transform=transform,
        panel=panel,
        target_names=tuple(f"T{i}" for i in range(pilot.shape[1])),
        selector="test",
        predictor="multivariate_ridge_gcv",
        random_draw=-1,
        base={"dataset": "toy", "fold": 1, "pilot_ligands": len(pilot), "evaluation_ligands": len(evaluation), "panel_k": len(panel)},
    )
    assert {record["surface"] for record in records} == {
        "raw",
        "raw_row_centered_residual",
    }
    assert all(np.isfinite(record[selector.PRIMARY_METRIC]) for record in records)


def test_selected_targets_do_not_enter_ridge_outcome_or_gcv(monkeypatch) -> None:
    pilot, evaluation = synthetic_scores()
    transform = selector.pilot_transform(pilot, evaluation)
    panel = np.array([0, 2, 5, 7])
    captured: dict[str, tuple[int, int]] = {}

    def fake_ridge(x_pilot, y_pilot, x_evaluation, alphas=selector.RIDGE_ALPHAS):
        captured["x"] = x_pilot.shape
        captured["y"] = y_pilot.shape
        return np.zeros((len(x_evaluation), y_pilot.shape[1])), 1.0, 1.0

    monkeypatch.setattr(selector, "ridge_gcv_predict", fake_ridge)
    selector.prediction_records(
        transform=transform,
        panel=panel,
        target_names=tuple(f"T{i}" for i in range(pilot.shape[1])),
        selector="test",
        predictor="multivariate_ridge_gcv",
        random_draw=-1,
        base={"dataset": "toy", "fold": 1, "pilot_ligands": len(pilot), "evaluation_ligands": len(evaluation), "panel_k": len(panel)},
    )
    assert captured["x"] == (len(pilot), len(panel))
    assert captured["y"] == (len(pilot), pilot.shape[1] - len(panel))


def test_exact_medoids_returns_valid_panel() -> None:
    pilot, evaluation = synthetic_scores()
    transform = selector.pilot_transform(pilot, evaluation)
    panel = selector.residual_map_medoids(transform, 4)
    assert len(panel) == len(np.unique(panel)) == 4
    assert panel.min() >= 0 and panel.max() < pilot.shape[1]


def test_existing_artifact_is_fail_closed_if_present() -> None:
    output = selector.DEFAULT_OUTPUT
    if not output.exists():
        return
    expected = {
        "panel_metrics.csv.gz",
        "split_diagnostics.csv",
        "panel_selections.csv",
        "paired_baseline_contrasts.csv",
        "paired_selector_contrasts.csv",
        "selector_metric_summary.csv",
        "baseline_contrast_summary.csv",
        "selector_pair_summary.csv",
        "summary.json",
        "README.md",
        "checksums.sha256",
    }
    assert expected <= {path.name for path in output.iterdir()}
    checksums = {}
    for line in (output / "checksums.sha256").read_text().splitlines():
        digest, filename = line.split("  ", maxsplit=1)
        checksums[filename] = digest
    for filename in expected - {"checksums.sha256"}:
        digest = hashlib.sha256((output / filename).read_bytes()).hexdigest()
        assert checksums[filename] == digest
    metrics = pd.read_csv(output / "panel_metrics.csv.gz")
    splits = pd.read_csv(output / "split_diagnostics.csv")
    assert splits.pilot_evaluation_group_overlap.eq(0).all()
    assert splits.pilot_missing_cells.eq(0).all()
    assert splits.evaluation_missing_cells.eq(0).all()
    random = metrics.loc[
        metrics.selector.eq("fixed_random_panel") & metrics.surface.eq("raw")
    ]
    counts = random.groupby(
        ["dataset", "fold", "pilot_ligands", "panel_k"]
    ).size()
    assert counts.min() >= 50
