from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd

try:
    from . import public_panel_selector_controls as selector
    from . import public_panel_selector_robustness as robustness
except ImportError:  # pragma: no cover
    import public_panel_selector_controls as selector  # type: ignore
    import public_panel_selector_robustness as robustness  # type: ignore


def synthetic_transform(seed: int = 87) -> selector.PilotTransform:
    rng = np.random.default_rng(seed)
    latent_pilot = rng.normal(size=(160, 3))
    latent_eval = rng.normal(size=(90, 3))
    weights = rng.normal(size=(3, 10))
    pilot = latent_pilot @ weights + 0.05 * rng.normal(size=(160, 10))
    evaluation = latent_eval @ weights + 0.05 * rng.normal(size=(90, 10))
    return selector.pilot_transform(pilot, evaluation)


def test_residual_derivation_copies_raw_selected_then_centers() -> None:
    transform = synthetic_transform()
    selected = np.array([0, 2, 5, 8])
    omitted = np.setdiff1d(np.arange(10), selected)
    prediction = np.zeros((len(transform.evaluation_raw_z), len(omitted)))
    observed = robustness.derive_residual_prediction(
        transform, selected, omitted, prediction
    )
    full_z = np.empty_like(transform.evaluation_raw_z)
    full_z[:, selected] = transform.evaluation_raw_z[:, selected]
    full_z[:, omitted] = prediction
    full_raw = full_z * transform.raw_sds + transform.raw_means
    expected = (
        full_raw - full_raw.mean(axis=1, keepdims=True) - transform.residual_means
    ) / transform.residual_sds
    np.testing.assert_allclose(observed, expected[:, omitted])


def test_common_omitted_uses_exact_same_targets_and_rmse_direction() -> None:
    rows = []
    for panel_selector, draw, targets, delta in (
        (robustness.DESIGNED_SELECTOR, -1, (2, 3, 4, 5), 0.0),
        (robustness.RANDOM_SELECTOR, 0, (0, 1, 4, 5), 0.2),
        (robustness.RANDOM_SELECTOR, 1, (0, 2, 3, 5), 0.1),
        (robustness.RANDOM_SELECTOR, 2, (1, 2, 3, 4), 0.3),
    ):
        for target in targets:
            rows.append(
                {
                    "dataset": "toy",
                    "fold": 1,
                    "pilot_ligands": 500,
                    "surface": "raw",
                    "training_outcome": "raw_outcome",
                    "panel_selector": panel_selector,
                    "random_panel_draw": draw,
                    "panel_indices": "0|1",
                    "selected_targets": 2,
                    "omitted_targets": 4,
                    "target_index": target,
                    "r2": 0.8 - delta,
                    "rmse": 0.2 + delta,
                    "pearson": 0.9 - delta,
                    "spearman": 0.85 - delta,
                    "equal_budget_lower_5pct_overlap": 0.7 - delta,
                }
            )
    contrasts = robustness.common_omitted_contrasts(pd.DataFrame(rows))
    assert len(contrasts) == 3
    assert contrasts.common_omitted_targets.min() == 2
    assert (contrasts.designed_benefit_mean_r2 > 0).all()
    assert (contrasts.designed_benefit_mean_rmse > 0).all()


def test_direct_residual_contrast_is_paired_on_one_panel() -> None:
    rows = []
    for training, predictor, r2, rmse in (
        ("raw_then_derived_residual", "multivariate_ridge_gcv", 0.2, 0.9),
        ("raw_then_derived_residual", "selected_target_pc1", 0.1, 1.0),
        ("direct_residual_outcome", "multivariate_ridge_gcv", 0.3, 0.8),
        ("direct_residual_outcome", "selected_target_pc1", 0.15, 0.95),
    ):
        rows.append(
            {
                "dataset": "toy",
                "fold": 1,
                "panel_selector": robustness.DESIGNED_SELECTOR,
                "surface": "raw_row_centered_residual",
                "training_outcome": training,
                "predictor": predictor,
                "variance_weighted_r2": r2,
                "pooled_rmse": rmse,
                "median_target_pearson": r2,
                "median_ligand_profile_pearson": r2,
                "mean_ligand_profile_pearson": r2,
            }
        )
    result = robustness.direct_residual_contrasts(pd.DataFrame(rows)).iloc[0]
    assert np.isclose(
        result.direct_minus_derived_ridge_benefit_variance_weighted_r2, 0.1
    )
    assert np.isclose(result.direct_minus_derived_ridge_benefit_pooled_rmse, 0.1)
    assert np.isclose(
        result.direct_ridge_minus_pc1_benefit_variance_weighted_r2, 0.15
    )


def test_released_artifact_checksums_and_design_if_present() -> None:
    output = robustness.DEFAULT_OUTPUT
    if not output.exists():
        return
    summary = json.loads((output / "summary.json").read_text())
    assert summary["analysis_status"] == "strict_public_k8_selector_robustness_control"
    assert summary["parameters"]["panel_k"] == 8
    checksums = {}
    for line in (output / "checksums.sha256").read_text().splitlines():
        digest, filename = line.split("  ", maxsplit=1)
        checksums[filename] = digest
    for filename, expected in checksums.items():
        assert hashlib.sha256((output / filename).read_bytes()).hexdigest() == expected
    pairs = pd.read_csv(output / "common_omitted_pair_contrasts.csv")
    direct = pd.read_csv(output / "direct_residual_contrasts.csv")
    splits = pd.read_csv(output / "split_diagnostics.csv")
    assert len(pairs) == 2 * 5 * 2 * 3
    assert len(direct) == 2 * 5
    assert splits.pilot_evaluation_row_overlap.eq(0).all()
    assert splits.pilot_evaluation_group_overlap.eq(0).all()
    assert pairs.common_omitted_targets.min() >= 44 - 16
