#!/usr/bin/env python3
"""Tests for the aggregate-only residual/MIE boundary audit."""

from __future__ import annotations

import json
import re
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

from residual_mie_boundary_audit import (  # noqa: E402
    DEFAULT_ACTIVITIES,
    DEFAULT_SCORES,
    DOCK44,
    average_rank_matrix,
    bootstrap_distribution,
    bootstrap_draw_indices,
    contribution_arrays,
    derive_strict_active_pairs,
    metric_value_matrices,
    average_rank_matrix,
    preprocess_score_representations,
    rank_score_representations,
    switch_edges_in_place,
)


RESULTS = ROOT / "results" / "residual_mie_boundary_audit"


class ResidualMIEBoundaryUnitTests(unittest.TestCase):
    def test_row_centering_cannot_change_within_row_ranks(self) -> None:
        rng = np.random.default_rng(17)
        scores = rng.normal(size=(120, 11))
        column_z = (scores - scores.mean(axis=0)) / scores.std(axis=0, ddof=1)
        residual = column_z - column_z.mean(axis=1, keepdims=True)
        np.testing.assert_array_equal(
            average_rank_matrix(column_z), average_rank_matrix(residual)
        )

    def test_average_ranking_is_tie_aware(self) -> None:
        scores = np.array([[1.0, 1.0, 3.0, 2.0]])
        np.testing.assert_array_equal(
            average_rank_matrix(scores), np.array([[1.5, 1.5, 4.0, 3.0]])
        )

    def test_switch_chain_preserves_binary_row_and_column_margins(self) -> None:
        incidence = np.array(
            [
                [1, 1, 0, 0, 0],
                [0, 1, 1, 0, 0],
                [0, 0, 1, 1, 0],
                [0, 0, 0, 1, 1],
                [1, 0, 0, 0, 1],
                [1, 0, 1, 0, 0],
            ],
            dtype=bool,
        )
        row_before = incidence.sum(axis=1).copy()
        column_before = incidence.sum(axis=0).copy()
        compounds, targets = np.where(incidence)
        switch_edges_in_place(
            incidence,
            compounds.astype(int),
            targets.astype(int),
            accepted_swaps=500,
            rng=np.random.default_rng(19),
        )
        np.testing.assert_array_equal(incidence.sum(axis=1), row_before)
        np.testing.assert_array_equal(incidence.sum(axis=0), column_before)
        self.assertEqual(int(incidence.sum()), len(targets))
        self.assertEqual(incidence.dtype, np.bool_)

    def test_shared_draws_make_identical_representation_contrast_zero(self) -> None:
        incidence = np.array(
            [[1, 0, 1], [0, 1, 0], [1, 1, 0], [0, 0, 1]], dtype=bool
        )
        values = metric_value_matrices(
            average_rank_matrix(
                np.array(
                    [
                        [-2.0, -1.0, 0.0],
                        [-1.0, -3.0, -2.0],
                        [0.0, -1.0, -2.0],
                        [-3.0, -2.0, -1.0],
                    ]
                )
            )
        )["mean_rank_percentile"]
        sufficient = contribution_arrays(values, incidence)
        draws = bootstrap_draw_indices(4, 250, np.random.default_rng(23))
        first = bootstrap_distribution(sufficient, draws, "compound_balanced")
        second = bootstrap_distribution(sufficient, draws, "compound_balanced")
        np.testing.assert_array_equal(first - second, np.zeros_like(first))

    def test_frozen_inputs_reconstruct_the_strict_500_pair_benchmark(self) -> None:
        frame = pd.read_csv(
            DEFAULT_SCORES,
            usecols=["ligand_id", "Cleaned SMILES", "Butina_clusters", *DOCK44],
        )
        pairs = derive_strict_active_pairs(frame, DEFAULT_ACTIVITIES, 6.0)
        self.assertEqual(len(pairs), 500)
        self.assertEqual(pairs["ligand_id"].nunique(), 190)
        self.assertEqual(pairs["pdb"].nunique(), 39)

    def test_frozen_preprocessing_produces_exact_rank_identity(self) -> None:
        frame = pd.read_csv(
            DEFAULT_SCORES,
            usecols=["ligand_id", "Cleaned SMILES", "Butina_clusters", *DOCK44],
        )
        representations, counts = preprocess_score_representations(frame)
        selected_rows = np.arange(len(frame), dtype=int)
        ranks, diagnostics = rank_score_representations(
            representations, selected_rows
        )
        # Under the manuscript's definition the residual preserves the ordering of
        # column-centred scores, not of column-z scores; the two differ whenever
        # target columns have unequal spread.
        absolute = representations["absolute"]
        column_centred = absolute - absolute.mean(axis=0)
        expected = average_rank_matrix(
            column_centred - column_centred.mean(axis=1, keepdims=True)
        )
        np.testing.assert_array_equal(expected, ranks["two_way_residual"])
        self.assertGreater(diagnostics["maximum_absolute_rank_difference"], 0.0)
        self.assertFalse(diagnostics["all_rankings_identical"])
        self.assertLessEqual(
            diagnostics["maximum_absolute_score_reconstruction_error"], 1e-12
        )
        self.assertGreater(counts["missing_cells_before_imputation"], 0)


@unittest.skipUnless(
    (RESULTS / "analysis_summary.json").exists(),
    "run residual_mie_boundary_audit.py to create frozen outputs",
)
class ResidualMIEBoundaryFrozenOutputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.summary = json.loads(
            (RESULTS / "analysis_summary.json").read_text(encoding="utf-8")
        )
        cls.observed = pd.read_csv(RESULTS / "observed_metrics.csv")
        cls.contrasts = pd.read_csv(RESULTS / "paired_contrasts.csv")
        cls.nulls = pd.read_csv(RESULTS / "null_summary.csv")

    def test_frozen_counts_and_rank_identity(self) -> None:
        counts = self.summary["benchmark_counts"]
        self.assertEqual(counts["active_pairs"], 500)
        self.assertEqual(counts["compounds"], 190)
        self.assertEqual(counts["represented_targets"], 39)
        self.assertEqual(counts["ranking_panel_targets"], 44)
        self.assertFalse(self.summary["rank_identity"]["all_rankings_identical"])
        self.assertGreater(
            self.summary["rank_identity"]["maximum_absolute_rank_difference"], 0.0
        )
        self.assertIn(
            "column-centred ranking", self.summary["rank_identity"]["statement"]
        )
        readme = (RESULTS / "README.md").read_text(encoding="utf-8")
        self.assertIn("column-centred score order", readme)
        self.assertNotIn("exactly identical to\ncolumn-z", readme)

    def test_expected_absolute_and_column_z_pair_weighted_estimates(self) -> None:
        def estimate(representation: str, metric: str) -> float:
            row = self.observed[
                (self.observed["representation"] == representation)
                & (self.observed["weighting"] == "pair_weighted")
                & (self.observed["metric"] == metric)
            ].iloc[0]
            return float(row["estimate"])

        self.assertAlmostEqual(
            estimate("absolute", "mean_rank_percentile"),
            0.4181627906976745,
            places=12,
        )
        self.assertAlmostEqual(
            estimate("column_z", "mean_rank_percentile"),
            0.45227906976744187,
            places=12,
        )
        self.assertNotAlmostEqual(
            estimate("two_way_residual", "mean_rank_percentile"),
            estimate("column_z", "mean_rank_percentile"),
            places=6,
        )

    def test_null_and_bootstrap_tables_are_complete(self) -> None:
        self.assertEqual(len(self.observed), 3 * 2 * 4)
        self.assertEqual(len(self.contrasts), 3 * 2 * 4)
        self.assertEqual(len(self.nulls), 2 * 3 * 2 * 4)
        self.assertTrue(
            set(self.nulls["null_model"])
            == {"target_frequency_multiset", "fixed_margin_bipartite_switch"}
        )

    def test_aggregate_outputs_contain_no_raw_compound_identifiers(self) -> None:
        forbidden_columns = {
            "ligand_id",
            "inchikey",
            "smiles",
            "canonical_smiles",
            "cleaned_smiles",
            "pubchem_cid",
        }
        for filename in (
            "observed_metrics.csv",
            "paired_contrasts.csv",
            "null_summary.csv",
            "input_manifest.csv",
        ):
            frame = pd.read_csv(RESULTS / filename)
            self.assertTrue(forbidden_columns.isdisjoint(map(str.lower, frame.columns)))

        inchikey_pattern = re.compile(r"\b[A-Z]{14}-[A-Z]{10}-[A-Z]\b")
        for path in RESULTS.iterdir():
            if path.suffix.lower() not in {".csv", ".json", ".md"}:
                continue
            text = path.read_text(encoding="utf-8")
            self.assertIsNone(inchikey_pattern.search(text), msg=path.name)


if __name__ == "__main__":
    unittest.main()
