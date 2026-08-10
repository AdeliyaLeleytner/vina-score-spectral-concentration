#!/usr/bin/env python3
"""Focused tests for the Docking-44 Vina-numerator audit."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

from docking44_vina_numerator_audit import (  # noqa: E402
    DOCK44,
    NUMERATOR_TARGETS,
    additive_factor_toy,
    apply_scalar_support,
    correlation_pr,
    leave_one_target_out_reconstruction,
    mean_impute_columns,
    permute_within_groups,
    surface_metrics,
    two_way_center,
    validate_target_contract,
)


RESULTS = ROOT / "results" / "docking44_vina_numerator"


class Docking44VinaNumeratorUnitTests(unittest.TestCase):
    def test_exact_target_contract_excludes_v1a_and_never_substitutes_9uwl(self) -> None:
        self.assertEqual(len(DOCK44), 44)
        self.assertEqual(len(NUMERATOR_TARGETS), 43)
        self.assertNotIn("V1A", NUMERATOR_TARGETS)
        self.assertNotIn("9uwl", NUMERATOR_TARGETS)

        with tempfile.TemporaryDirectory() as directory:
            residue_dir = Path(directory)
            for target in NUMERATOR_TARGETS:
                (residue_dir / f"{target}_residue_matrix.csv").touch()
            # Presence of the tempting alternative key must not change support.
            (residue_dir / "9uwl_residue_matrix.csv").touch()
            paths = validate_target_contract(residue_dir)
        self.assertEqual(len(paths), 43)
        self.assertTrue(all("9uwl" not in path.name for path in paths))

    def test_scalar_mask_and_zero_censoring_are_preserved(self) -> None:
        numerator = np.array([
            [-3.0, 90.0, -2.0],
            [-5.0, -7.0, 4.0],
        ])
        scores = np.array([
            [-2.0, np.nan, 0.0],
            [-4.0, -6.0, -3.0],
        ])
        matched = apply_scalar_support(
            numerator, scores, clip_upper_zero=True
        )
        self.assertTrue(np.isnan(matched[0, 1]))
        self.assertEqual(matched[0, 2], 0.0)
        self.assertEqual(matched[1, 2], 0.0)
        self.assertEqual(matched[1, 0], -5.0)

    def test_column_mean_imputation_does_not_use_masked_upstream_value(self) -> None:
        matrix = np.array([
            [-2.0, np.nan],
            [-4.0, -8.0],
            [-6.0, -4.0],
        ])
        completed = mean_impute_columns(matrix)
        self.assertEqual(completed[0, 1], -6.0)

    def test_constant_row_factor_leaves_both_pr_values_unchanged(self) -> None:
        rng = np.random.default_rng(11)
        matrix = rng.normal(size=(500, 12))
        before = surface_metrics(matrix)
        after = surface_metrics(matrix / 1.37)
        self.assertAlmostEqual(before["raw_pr"], after["raw_pr"], places=12)
        self.assertAlmostEqual(
            before["residual_pr"], after["residual_pr"], places=12
        )

    def test_row_scaling_of_pure_additive_numerator_creates_one_mode(self) -> None:
        rng = np.random.default_rng(13)
        n, p = 300, 9
        row_effect = rng.normal(scale=2.0, size=n)
        target_effect = np.linspace(-1.2, 1.4, p)
        additive = 3.0 + row_effect[:, None] + target_effect[None, :]
        factor = 1.0 + np.linspace(0.0, 0.8, n)
        record = additive_factor_toy(additive, factor)
        self.assertLess(
            record["maximum_absolute_centered_value_before_factor"], 1e-12
        )
        self.assertEqual(record["centered_numerical_rank_after_factor"], 1)
        self.assertAlmostEqual(
            record["centered_correlation_pr_after_factor"], 1.0, places=12
        )

    def test_two_way_center_and_pr_contract(self) -> None:
        rng = np.random.default_rng(17)
        matrix = rng.normal(size=(250, 8))
        residual = two_way_center(matrix)
        self.assertLess(np.max(np.abs(residual.mean(axis=0))), 1e-14)
        self.assertLess(np.max(np.abs(residual.mean(axis=1))), 1e-14)
        self.assertGreater(correlation_pr(residual)["participation_ratio"], 1.0)

    def test_grouped_factor_permutation_preserves_group_membership(self) -> None:
        values = np.array([1.0, 2.0, 3.0, 10.0, 20.0])
        groups = np.array([0, 0, 0, 1, 1])
        permuted = permute_within_groups(
            values, groups, np.random.default_rng(31)
        )
        self.assertEqual(set(permuted[:3]), {1.0, 2.0, 3.0})
        self.assertEqual(set(permuted[3:]), {10.0, 20.0})

    def test_leave_one_target_out_factor_reconstructs_shared_row_scaling(self) -> None:
        rng = np.random.default_rng(37)
        numerator = -np.exp(rng.normal(size=(80, len(NUMERATOR_TARGETS))))
        factor = rng.uniform(1.0, 2.0, size=80)
        scores = numerator / factor[:, None]
        rows = leave_one_target_out_reconstruction(numerator, scores)
        self.assertTrue(
            np.allclose(rows.pearson_leave_one_target_out, 1.0, atol=1e-12)
        )
        self.assertLess(
            rows.mean_absolute_error_leave_one_target_out_kcal_mol.max(), 1e-12
        )


@unittest.skipUnless(
    (RESULTS / "analysis_summary.json").exists(),
    "run docking44_vina_numerator_audit.py to create frozen outputs",
)
class Docking44VinaNumeratorFrozenOutputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.summary = json.loads((RESULTS / "analysis_summary.json").read_text())
        cls.surfaces = pd.read_csv(RESULTS / "surface_metrics.csv")

    def primary_surface(self, name: str) -> pd.Series:
        rows = self.surfaces[
            self.surfaces.support.eq("43_target_mean_imputed_primary")
            & self.surfaces.surface.eq(name)
        ]
        self.assertEqual(len(rows), 1)
        return rows.iloc[0]

    def test_frozen_exact_overlap_and_headline_values(self) -> None:
        support = self.summary["support"]
        self.assertEqual(support["exact_numerator_overlap_targets"], 43)
        self.assertEqual(support["excluded_target"], "V1A")
        self.assertNotIn("9uwl", support["included_targets"])

        score = self.primary_surface("normalized_scalar_score")
        numerator = self.primary_surface("unnormalized_numerator_matched_mask")
        inferred = self.primary_surface("numerator_divided_by_inferred_q")
        descriptor = self.primary_surface(
            "numerator_divided_by_rotatable_bond_q"
        )
        self.assertAlmostEqual(score.raw_pr, 1.838796591634674, places=11)
        self.assertAlmostEqual(score.residual_pr, 9.31427197645104, places=10)
        self.assertAlmostEqual(numerator.raw_pr, 1.70007585586861, places=10)
        self.assertAlmostEqual(numerator.residual_pr, 7.12942851949878, places=9)
        self.assertAlmostEqual(inferred.residual_pr, 9.28305210140204, places=9)
        self.assertAlmostEqual(descriptor.residual_pr, 8.87043414364597, places=9)

    def test_torsion_amplifies_but_does_not_create_residual_dimension(self) -> None:
        headline = self.summary["headline"]
        self.assertGreater(headline["numerator_residual_pr"], 7.0)
        self.assertGreater(
            headline["residual_pr_amplification_by_inferred_q"], 2.0
        )
        self.assertLess(
            headline["residual_pr_amplification_by_inferred_q"], 2.3
        )
        self.assertAlmostEqual(
            headline["rotatable_bond_q_score_residual_pr"],
            8.87043414364597,
            places=9,
        )
        self.assertGreater(
            headline[
                "descriptive_fraction_of_inferred_q_pr_increase_due_to_amplification"
            ],
            0.25,
        )
        self.assertLess(
            headline[
                "descriptive_fraction_of_inferred_q_pr_increase_due_to_amplification"
            ],
            0.30,
        )
        self.assertGreater(
            self.summary["factor_diagnostics"][
                "q_vs_rotatable_bonds_pearson"
            ],
            0.96,
        )

    def test_shuffle_and_reconstruction_controls(self) -> None:
        shuffle = self.summary["q_shuffle_sensitivity"]
        self.assertEqual(shuffle["repeats"], 200)
        self.assertGreater(
            shuffle["observed_minus_shuffle_median_residual_pr"], 2.1
        )
        self.assertLess(
            shuffle[
                "observed_minus_exact_heavy_atom_shuffle_median_residual_pr"
            ],
            0.0,
        )
        self.assertGreater(
            shuffle["within_exact_heavy_atom_count"]["residual_pr"]["median"],
            shuffle["observed_inferred_q_residual_pr"],
        )
        reconstruction = self.summary["target_reconstruction"]
        self.assertGreater(reconstruction["median_target_pearson"], 0.999)
        self.assertLess(reconstruction["median_target_mae_kcal_mol"], 0.03)
        self.assertGreater(
            reconstruction["leave_one_target_out_median_pearson"], 0.999
        )
        self.assertLess(
            reconstruction["leave_one_target_out_median_mae_kcal_mol"], 0.03
        )

    def test_additive_toy_cannot_generate_nine_dimensions(self) -> None:
        toy = self.summary["additive_factor_toy"]
        self.assertEqual(toy["centered_numerical_rank_after_factor"], 1)
        self.assertAlmostEqual(
            toy["centered_correlation_pr_after_factor"], 1.0, places=10
        )


if __name__ == "__main__":
    unittest.main()
