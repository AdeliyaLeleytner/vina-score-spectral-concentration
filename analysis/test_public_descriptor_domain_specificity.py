from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import public_descriptor_domain_specificity as mod


PACKAGE = Path(__file__).resolve().parents[1]
RESULTS = PACKAGE / "results" / "public_descriptor_domain_specificity"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_extreme_masks_are_disjoint_and_inclusive() -> None:
    values = np.arange(20, dtype=float)
    low, high, low_threshold, high_threshold = mod.extreme_masks(values)
    assert not np.any(low & high)
    assert np.all(values[low] <= low_threshold)
    assert np.all(values[high] >= high_threshold)
    assert low.sum() == high.sum() == 5

    tied = np.repeat(np.arange(4, dtype=float), [4, 6, 6, 4])
    low, high, _, _ = mod.extreme_masks(tied)
    assert not np.any(low & high)
    assert low.sum() > len(tied) * mod.EXTREME_FRACTION
    assert high.sum() > len(tied) * mod.EXTREME_FRACTION


def test_descriptor_contrast_detects_support_specific_geometry() -> None:
    rng = np.random.default_rng(17)
    descriptors = rng.normal(size=(200, len(mod.DESCRIPTORS)))
    descriptors[:, 1] = np.linspace(-2, 2, 200)
    matrix = rng.normal(size=(200, 5))
    low = descriptors[:, 1] < 0
    matrix[low, 1] = matrix[low, 0] + rng.normal(scale=0.05, size=low.sum())
    matrix[~low, 1] = -matrix[~low, 0] + rng.normal(
        scale=0.05, size=(~low).sum()
    )
    frame = mod.descriptor_contrasts("synthetic", matrix, descriptors)
    row = frame.loc[
        frame.descriptor.eq("molecular_weight")
        & frame.transformation.eq("row_centered_residual")
    ].iloc[0]
    assert row.geometry_spearman < 0.8


def test_production_descriptor_values_match_frozen_contract() -> None:
    frame = pd.read_csv(RESULTS / "descriptor_extremes.csv")
    residual = frame.loc[frame.transformation.eq("row_centered_residual")]
    lookup = residual.set_index(["dataset", "descriptor"]).geometry_spearman
    assert np.isclose(lookup["Docking-44", "molecular_weight"], 0.20496480945389367)
    assert np.isclose(lookup["DOCKSTRING-58", "molecular_weight"], 0.3505978420709316)
    assert np.isclose(lookup["Docking-44", "tpsa"], 0.75305974173389)
    assert np.isclose(lookup["DOCKSTRING-58", "tpsa"], 0.7871035949644485)
    assert residual.groupby("dataset").size().to_dict() == {
        "DOCKSTRING-58": 7,
        "Docking-44": 7,
    }


def test_matched_random_disjoint_controls_cover_every_descriptor() -> None:
    summary = pd.read_csv(RESULTS / "descriptor_random_disjoint_summary.csv")
    assert len(summary) == 2 * len(mod.DESCRIPTORS)
    assert (summary.control_repetitions == mod.CONTROL_REPETITIONS).all()
    assert summary.observed_below_control_q025.all()
    assert (summary.combined_support_fraction <= 1.0).all()
    dockstring_ring = summary.loc[
        summary.dataset.eq("DOCKSTRING-58")
        & summary.descriptor.eq("ring_count")
    ].iloc[0]
    assert np.isclose(dockstring_ring.combined_support_fraction, 1.0)


def test_random_control_generator_matches_tail_sizes_and_is_finite() -> None:
    rng = np.random.default_rng(81)
    matrix = rng.normal(size=(120, 6))
    descriptors = rng.normal(size=(120, len(mod.DESCRIPTORS)))
    contrasts = mod.descriptor_contrasts("synthetic", matrix, descriptors)
    controls = mod.random_disjoint_controls(
        "synthetic", matrix, contrasts, seed=91
    )
    assert len(controls) == len(mod.DESCRIPTORS) * mod.CONTROL_REPETITIONS
    assert np.isfinite(controls.geometry_spearman).all()
    sizes = contrasts.loc[
        contrasts.transformation.eq("row_centered_residual"),
        ["descriptor", "low_ligands", "high_ligands"],
    ]
    observed = controls.groupby("descriptor")[
        ["low_ligands", "high_ligands"]
    ].first().reset_index()
    pd.testing.assert_frame_equal(
        sizes.sort_values("descriptor").reset_index(drop=True),
        observed.sort_values("descriptor").reset_index(drop=True),
    )


def test_production_support_and_checksums_are_authenticated() -> None:
    summary = json.loads((RESULTS / "summary.json").read_text())
    assert summary["analysis_status"] == "strict_public_post_hoc_descriptor_sensitivity"
    assert summary["datasets"]["DOCKSTRING-58"]["analysis_rows"] == 15_000
    assert summary["datasets"]["DOCKSTRING-58"]["support_seed"] == 71
    assert summary["configuration"]["inclusive_boundary_ties_retained"]
    assert summary["configuration"]["random_disjoint_control_repetitions"] == 200
    assert (
        summary["datasets"]["DOCKSTRING-58"]["selected_complete_row_index_sha256"]
        == "8afe6ccc883bd5d376847ddc4bfe0fff3b8cba1a518e4d8941f1e50a9356b11d"
    )
    manifest = json.loads((RESULTS / "output_checksums.json").read_text())
    for filename, expected in manifest["files"].items():
        assert _sha256(RESULTS / filename) == expected


def test_producer_has_no_repository_local_import() -> None:
    source = (PACKAGE / "analysis" / "public_descriptor_domain_specificity.py").read_text()
    assert "from residual_" not in source
    assert "from replicated_" not in source
    assert "from public_" not in source
    assert "PKIS1" not in source
    assert "KiRHub" not in source
