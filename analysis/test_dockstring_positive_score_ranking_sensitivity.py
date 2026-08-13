from __future__ import annotations

import json

import numpy as np

import dockstring_positive_score_ranking_sensitivity as sensitivity


def test_informative_pair_order_change_distinguishes_ties_and_reversals() -> None:
    experiment = np.asarray([[8.0, 7.0, 6.0], [5.0, 7.0, np.nan]])
    clipped = np.asarray([[0.0, 0.0, -1.0], [-1.0, -2.0, -3.0]])
    retained = np.asarray([[2.0, 1.0, -1.0], [-2.0, -1.0, -3.0]])
    report = sensitivity.informative_pair_order_change(
        clipped, retained, experiment
    )
    # Three informative pairs for ligand 1 and one for ligand 2.
    assert report["informative_experimental_pairs"] == 4
    # Ligand 1 A--B changes from a clipped tie to a strict order; ligand 2 A--B
    # reverses sign.
    assert report["score_order_changed_pairs"] == 2
    assert report["strict_score_order_reversal_pairs"] == 1
    assert report["clipped_score_ties"] == 1
    assert report["unclipped_score_ties"] == 0


def test_conservative_cluster_interval_takes_union() -> None:
    report = {
        "scaffold_cluster_bootstrap": {"interval_95": [-0.02, 0.03]},
        "butina_cluster_bootstrap": {"interval_95": [-0.01, 0.04]},
    }
    assert sensitivity.conservative_cluster_interval(report) == [-0.02, 0.04]


def test_released_positive_score_artifact_has_fixed_support() -> None:
    path = sensitivity.DEFAULT_OUTPUT / "summary.json"
    if not path.exists():
        return
    summary = json.loads(path.read_text())
    assert summary["schema_version"] == "1.0.0"
    assert summary["analysis_status"] == (
        "complete_exploratory_preprocessing_sensitivity"
    )
    assert summary["support"]["retained_ligands"] == 6557
    assert summary["support"]["targets"] == 31
    assert summary["reference"]["complete_source_positive_cells"] == 5393
    assert summary["released_clipped_result_reproduction"][
        "maximum_absolute_accuracy_difference"
    ] < 1e-12
    assert "does not validate positive Vina scores" in summary["claim_boundary"]
