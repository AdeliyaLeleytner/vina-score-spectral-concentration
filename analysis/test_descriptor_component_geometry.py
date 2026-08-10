"""Contract tests for the descriptor-mediated target-pair geometry artifact.

These read the released artifact rather than re-running the decomposition, which needs the
full DOCKSTRING surface and several minutes of RDKit work.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


PACKAGE = Path(__file__).resolve().parents[1]
RESULTS = PACKAGE / "results" / "descriptor_component_geometry"
PANELS = ("DAVIS", "PKIS2", "PKIS1", "KiRHub")


def load_summary() -> dict:
    return json.loads((RESULTS / "summary.json").read_text())


def test_support_and_panel_contract() -> None:
    summary = load_summary()
    assert summary["target_pairs"] == 190
    assert len(summary["targets"]) == 20
    assert tuple(summary["experimental_panel_results"]) == PANELS
    assert summary["decomposition_metrics"]["folds"] == 5
    assert summary["support"]["dockstring_reference_rows"] > 259_000
    assert summary["status"] == "exploratory_post_hoc_science_only"
    assert "does not show" in summary["claim_boundary"] or "not" in summary["claim_boundary"]


def test_observed_surface_reproduces_the_released_docking_geometry() -> None:
    """The out-of-fold observed residual must recover the published fixed-20 geometry.

    Correlation is invariant to per-column affine rescaling, so fold-local offsets and
    scales cannot change it; any real disagreement means the ligand support or the target
    order drifted away from the fixed-panel analysis.
    """
    summary = load_summary()
    assert summary["released_geometry_reconstruction_spearman"] > 0.99


def test_descriptor_component_carries_most_of_the_agreement() -> None:
    summary = load_summary()
    for panel in PANELS:
        record = summary["experimental_panel_results"][panel]
        observed = record["observed_residual"]["spearman"]
        component = record["descriptor_component"]["spearman"]
        removed = record["descriptor_removed"]["spearman"]
        assert observed > 0, panel
        assert component > 0, panel
        assert removed > 0, panel
        # The reported mediated share is only meaningful while both terms are positive.
        share = record["descriptor_mediated_fraction_of_observed_agreement"]
        assert share == component / observed
        assert 0.5 < share < 1.0, (panel, share)


def test_edge_table_matches_the_summary_ranges() -> None:
    frame = pd.read_csv(RESULTS / "target_pair_geometry.csv")
    assert len(frame) == 190
    assert frame.target_a.ne(frame.target_b).all()
    for panel in PANELS:
        assert f"{panel}_experimental_centered_correlation" in frame.columns
    for name in ("observed_residual", "descriptor_component", "descriptor_removed"):
        column = f"docking_{name}_correlation"
        assert column in frame.columns
        assert frame[column].between(-1.0, 1.0).all()
