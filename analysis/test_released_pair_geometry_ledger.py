"""The released 190-edge ledger must regenerate the manuscript's fixed-panel numbers.

This is the reproducibility promise made in Availability of data and materials: a reader
holding only the released aggregate can recompute the predictor-by-endpoint factorial, the
cross-panel agreement and the locked-endpoint retrieval metrics.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import average_precision_score, roc_auc_score


PACKAGE = Path(__file__).resolve().parents[1]
RESULTS = PACKAGE / "results" / "released_pair_geometry_ledger"
LEDGER = PACKAGE / "results" / "manuscript_evidence.json"
PANELS = ("DAVIS", "PKIS2", "PKIS1", "KiRHub")


def load_edges() -> pd.DataFrame:
    return pd.read_csv(RESULTS / "target_pair_geometry.csv")


def test_shape_and_locked_endpoint() -> None:
    edges = load_edges()
    assert len(edges) == 190
    assert int(edges.locked_endpoint_positive.sum()) == 15
    for panel in PANELS:
        assert f"{panel}_experimental_raw_correlation" in edges.columns
        assert f"{panel}_experimental_centered_correlation" in edges.columns


def test_reproduces_the_factorial_centred_cells() -> None:
    edges = load_edges()
    manuscript = json.loads(LEDGER.read_text())
    panels = manuscript["fixed20_estimand_decomposition"]["panels"]
    for panel in PANELS:
        observed = stats.spearmanr(
            edges["docking_local_20_centered_correlation"],
            edges[f"{panel}_experimental_centered_correlation"],
        ).statistic
        expected = panels[panel]["docking_centered__experimental_centered"]
        assert abs(observed - expected) < 5e-4, (panel, observed, expected)


def test_reproduces_cross_panel_agreement() -> None:
    edges = load_edges()
    manuscript = json.loads(LEDGER.read_text())
    cross = manuscript["experimental_cross_panel"]
    for suffix, key in (
        ("centered", "centered_mean_pairwise_spearman"),
        ("raw", "raw_mean_pairwise_spearman"),
    ):
        columns = [f"{panel}_experimental_{suffix}_correlation" for panel in PANELS]
        values = [
            stats.spearmanr(edges[first], edges[second]).statistic
            for first, second in itertools.combinations(columns, 2)
        ]
        assert abs(float(np.mean(values)) - cross[key]) < 5e-4, suffix


def test_reproduces_locked_endpoint_retrieval() -> None:
    edges = load_edges()
    manuscript = json.loads(LEDGER.read_text())
    locked = manuscript["kirhub_locked_endpoint"]["target_label_qap"]["metrics"]
    labels = edges.locked_endpoint_positive.astype(bool)
    for suffix, key in (("centered", "centered"), ("raw", "raw")):
        scores = edges[f"KiRHub_experimental_{suffix}_correlation"]
        assert (
            abs(
                roc_auc_score(labels, scores)
                - locked["roc_auc"][f"{key}_KiRHub_geometry"]
            )
            < 5e-4
        ), suffix
        assert (
            abs(
                average_precision_score(labels, scores)
                - locked["average_precision"][f"{key}_KiRHub_geometry"]
            )
            < 5e-4
        ), suffix
