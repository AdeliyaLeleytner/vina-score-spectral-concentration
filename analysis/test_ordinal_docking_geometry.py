from __future__ import annotations

import numpy as np
import pandas as pd

try:
    from . import ordinal_docking_geometry as ordinal
except ImportError:  # pragma: no cover
    import ordinal_docking_geometry as ordinal  # type: ignore


def _synthetic_scores(seed: int = 3) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    scores = rng.normal(size=(200, 6))
    scores += rng.normal(size=(200, 1)) * 2.0
    heavy = rng.integers(8, 45, size=len(scores)).astype(float)
    return scores, heavy


def test_raw_column_z_geometry_equals_raw_correlation() -> None:
    scores, heavy = _synthetic_scores()
    surfaces = ordinal.transformed_surfaces(scores, heavy)
    observed = np.corrcoef(surfaces["raw_column_z"], rowvar=False)
    expected = np.corrcoef(scores, rowvar=False)
    np.testing.assert_allclose(observed, expected, atol=1e-12)


def test_ordinal_geometry_is_invariant_to_positive_row_affine_changes() -> None:
    scores, heavy = _synthetic_scores()
    scale = np.linspace(0.5, 3.0, len(scores))[:, None]
    offset = np.linspace(-10.0, 10.0, len(scores))[:, None]
    changed = scores * scale + offset
    original = ordinal.representation_geometries(scores, heavy)[
        "within_ligand_ordinal"
    ]
    transformed = ordinal.representation_geometries(changed, heavy)[
        "within_ligand_ordinal"
    ]
    np.testing.assert_allclose(original, transformed, atol=1e-12)


def test_within_row_ranks_use_average_ties() -> None:
    values = np.array([[1.0, 1.0, 3.0], [3.0, 2.0, 1.0]])
    observed = ordinal.within_row_ranks(values)
    expected = np.array([[1.5, 1.5, 3.0], [3.0, 2.0, 1.0]])
    np.testing.assert_allclose(observed, expected)


def test_paired_qap_identical_predictors_has_zero_gain() -> None:
    scores, heavy = _synthetic_scores()
    predictor = ordinal.representation_geometries(scores, heavy)[
        "standard_centered"
    ]
    endpoints = {"panel": predictor.copy()}
    labels = np.zeros(15, dtype=bool)
    labels[[0, 4, 10]] = True
    result = ordinal.paired_target_label_qap(
        predictor, predictor, endpoints, labels, permutations=49, seed=7
    )
    assert np.allclose(result.ordinal_minus_standard_centered, 0.0)
    assert np.allclose(result.one_sided_p_positive_gain, 1.0)
    assert np.allclose(result.one_sided_p_negative_gain, 1.0)
    assert np.allclose(result.two_sided_p_difference, 1.0)


def test_target_delete_one_returns_all_endpoints() -> None:
    scores, heavy = _synthetic_scores()
    predictors = ordinal.representation_geometries(scores, heavy)
    endpoints = {"A": predictors["standard_centered"].copy()}
    labels = np.zeros(15, dtype=bool)
    labels[[0, 4, 10]] = True
    frame = ordinal.target_delete_one(
        tuple(f"T{i}" for i in range(6)), predictors, endpoints, labels
    )
    assert len(frame) == 6 * 4
    assert set(frame.metric) == {
        "spearman_geometry_concordance",
        "mean_spearman_geometry_concordance",
        "roc_auc",
        "average_precision",
    }
    assert set(predictors).issubset(frame.columns)


def test_classification_requires_cross_endpoint_and_jackknife_gates() -> None:
    rows = []
    for panel in ("DAVIS", "PKIS2", "PKIS1", "KiRHub"):
        rows.extend(
            [
                {
                    "endpoint": panel,
                    "metric": "spearman_geometry_concordance",
                    "representation": "standard_centered",
                    "estimate": 0.1,
                },
                {
                    "endpoint": panel,
                    "metric": "spearman_geometry_concordance",
                    "representation": "within_ligand_ordinal",
                    "estimate": 0.2,
                },
            ]
        )
    qap = pd.DataFrame(
        [
            {
                "endpoint": "equal_weight_mean_of_four_panels",
                "metric": "mean_spearman_geometry_concordance",
                "ordinal_minus_standard_centered": 0.1,
                "one_sided_p_positive_gain": 0.01,
            },
            {
                "endpoint": "replicated_upper_10pct_in_at_least_two_old_panels",
                "metric": "roc_auc",
                "ordinal_minus_standard_centered": 0.1,
                "one_sided_p_positive_gain": 0.01,
            },
        ]
    )
    jackknife = pd.DataFrame(
        {
            "endpoint": ["equal_weight_mean_of_four_panels"] * 4,
            "ordinal_minus_standard_centered": [0.01, 0.02, 0.03, 0.04],
        }
    )
    verdict, details = ordinal.classify_result(pd.DataFrame(rows), qap, jackknife)
    assert verdict.startswith("GO:")
    assert all(details["criteria"].values())
