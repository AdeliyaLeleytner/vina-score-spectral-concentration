from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import public_group_disjoint_panel_reconstruction as mod


PACKAGE = Path(__file__).resolve().parents[1]
RESULTS = PACKAGE / "results" / "public_group_disjoint_panel_reconstruction"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_fixed_group_splits_have_no_group_or_row_leakage() -> None:
    matrix = np.zeros((100, 9))
    groups = np.repeat(np.arange(20), 5).astype(object)
    splits = mod.fixed_splits(matrix, groups, seed=7)
    assert len(splits) == 2 * mod.groups_mod.FOLDS
    for split in splits:
        assert np.intersect1d(split["pool_index"], split["held_index"]).size == 0
        if split["holdout_design"] == "chemical_group":
            assert split["group_overlap"] == 0


def test_synthetic_reconstruction_is_pilot_only_and_deployable(monkeypatch) -> None:
    rng = np.random.default_rng(13)
    latent = rng.normal(size=(250, 4))
    matrix = latent @ rng.normal(size=(4, 10)) + 0.2 * rng.normal(size=(250, 10))
    groups = np.repeat(np.arange(50), 5).astype(object)
    monkeypatch.setattr(mod.groups_mod, "CALIBRATION_SIZES", (20,))
    metrics, target_metrics, selections, folds = mod.analyze_dataset(
        "synthetic",
        matrix,
        groups,
        tuple(f"T{index}" for index in range(10)),
        split_seed=17,
        sampling_seed=19,
        repetitions_per_fold=1,
        panel_k=3,
    )
    assert len(metrics) == 2 * mod.groups_mod.FOLDS * 8
    assert len(selections) == 2 * mod.groups_mod.FOLDS * 3
    assert len(folds) == 2 * mod.groups_mod.FOLDS
    assert metrics.pilot_held_row_overlap.max() == 0
    assert metrics.pilot_only_preprocessing.all()
    primary = metrics.loc[
        metrics.truth_surface.eq(mod.PRIMARY_TRUTH)
        & metrics.outcome_basis.eq(mod.PRIMARY_BASIS)
        & metrics.predictive_model.eq("multivariate_ridge")
    ]
    assert len(primary) == 2 * mod.groups_mod.FOLDS
    assert np.isfinite(primary.variance_weighted_r2).all()
    assert np.isfinite(primary.pooled_calibration_standardized_rmse).all()
    residual = metrics.loc[
        metrics.truth_surface.eq("raw_row_centered_residual_outcome")
    ]
    tail_columns = [column for column in metrics if "lower_5pct" in column]
    assert tail_columns
    assert residual[tail_columns].isna().all().all()
    assert not target_metrics.empty


def test_released_artifact_contract_and_checksums() -> None:
    summary_path = RESULTS / "summary.json"
    if not summary_path.exists():
        return
    summary = json.loads(summary_path.read_text())
    # This frozen v1 artifact is retained for historical reconstruction only.
    # The current producer emits schema v2 after migration to SplitTransform and
    # adds a declared selected-target-PC1 comparator. It is not used by v5.
    assert summary["schema_version"] == "1.0.0"
    assert summary["analysis_status"] == (
        "strict_public_group_disjoint_panel_reconstruction_sensitivity"
    )
    assert summary["configuration"]["repetitions_per_fold"] == 5
    assert summary["configuration"]["replicates_per_dataset_design_size"] == 25
    assert summary["configuration"]["panel_k"] == 8
    metrics = pd.read_csv(RESULTS / "reconstruction_metrics.csv")
    assert len(metrics) == 2 * 2 * 5 * 2 * 5 * 4
    primary = metrics.loc[
        metrics.truth_surface.eq(mod.PRIMARY_TRUTH)
        & metrics.outcome_basis.eq(mod.PRIMARY_BASIS)
    ]
    assert len(primary) == 2 * 2 * 5 * 2 * 5
    assert primary.pilot_held_row_overlap.max() == 0
    lookup = primary.groupby(
        ["dataset", "holdout_design", "pilot_ligands"], as_index=True
    ).variance_weighted_r2.mean()
    expected = {
        ("Docking-44", "chemical_group", 200): 0.7688314760023931,
        ("Docking-44", "chemical_group", 500): 0.7772635407241074,
        ("DOCKSTRING-58", "chemical_group", 200): 0.6936758201687054,
        ("DOCKSTRING-58", "chemical_group", 500): 0.7036858440718936,
    }
    for key, value in expected.items():
        assert np.isclose(lookup.loc[key], value)
    residual = metrics.loc[metrics.truth_surface.eq("full_row_residual_outcome")]
    tail_columns = [column for column in metrics if "lower_5pct" in column]
    assert residual[tail_columns].isna().all().all()
    folds = pd.read_csv(RESULTS / "fold_contract.csv")
    group_folds = folds.loc[folds.holdout_design.eq("chemical_group")]
    assert group_folds.group_overlap.max() == 0
    manifest = json.loads((RESULTS / "output_checksums.json").read_text())
    for filename, expected in manifest["files"].items():
        assert sha256(RESULTS / filename) == expected


def test_source_has_no_restricted_private_inputs() -> None:
    source = Path(mod.__file__).read_text(encoding="utf-8")
    prohibited = ("PKIS1", "KiRHub", "ChEMBL", "residual_private", "build_evidence")
    assert not any(token in source for token in prohibited)
