#!/usr/bin/env python3
"""Tests for the release-grade residual-mechanism analysis."""

from __future__ import annotations

import hashlib
import json
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

from residual_mechanism_analysis import (  # noqa: E402
    correlation_moments,
    exact_descriptor_omnibus_permutation,
    family_qap,
    grouped_descriptor_decomposition,
    orthogonal_two_way_anova,
    rank_fingerprint_hungarian_retrieval,
    two_way_center,
    uniform_axis_diagnostics,
)


RESULTS = ROOT / "results" / "residual_mechanism"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class ResidualMechanismUnitTests(unittest.TestCase):
    def test_pr_correlation_moment_identity_is_exact(self) -> None:
        rng = np.random.default_rng(17)
        matrix = rng.normal(size=(500, 9))
        record = correlation_moments(matrix)
        self.assertLess(record["maximum_absolute_moment_identity_error"], 1e-14)
        self.assertLess(record["maximum_absolute_pr_identity_error"], 1e-12)

    def test_uniform_axis_diagnostic_recovers_common_factor(self) -> None:
        rng = np.random.default_rng(19)
        common = rng.normal(size=(2000, 1))
        matrix = common + 0.15 * rng.normal(size=(2000, 12))
        record = uniform_axis_diagnostics(matrix)
        self.assertGreater(record["cosine_pc1_with_uniform_vector"], 0.999)
        self.assertGreater(
            record["uniform_rayleigh_as_fraction_of_pc1_eigenvalue"], 0.999
        )
        self.assertLess(record["uniform_rayleigh_identity_error"], 1e-12)

    def test_two_way_anova_is_orthogonal_and_closes(self) -> None:
        rng = np.random.default_rng(21)
        matrix = (
            rng.normal(scale=3.0, size=(80, 1))
            + rng.normal(scale=2.0, size=(1, 11))
            + rng.normal(size=(80, 11))
        )
        record = orthogonal_two_way_anova(matrix)
        self.assertLess(record["fraction_closure_error"], 1e-14)
        self.assertLess(
            record["maximum_absolute_pairwise_inner_product_fraction"], 1e-14
        )
        self.assertLess(record["maximum_absolute_reconstruction_error"], 1e-13)
        self.assertAlmostEqual(
            sum(record["component_fractions"].values()), 1.0, places=14
        )

    def test_exact_descriptor_omnibus_recovers_aligned_target_order(self) -> None:
        rng = np.random.default_rng(22)
        first = rng.normal(size=(7, 7))
        second = first + 0.01 * rng.normal(size=(7, 7))
        record = exact_descriptor_omnibus_permutation(first, second)
        self.assertGreater(record["omnibus_mean_descriptor_spearman_rho"], 0.99)
        self.assertEqual(record["exact_target_label_permutations"], 5040)
        self.assertLess(
            record["exact_target_label_permutation_two_sided_p"], 0.01
        )

    def test_rank_fingerprint_retrieval_recovers_one_to_one_mapping(self) -> None:
        rng = np.random.default_rng(24)
        first = rng.normal(size=(8, 7))
        second = first + 0.01 * rng.normal(size=(8, 7))
        record = rank_fingerprint_hungarian_retrieval(first, second)
        self.assertEqual(record["correct_target_identities"], 8)
        self.assertEqual(record["number_of_cost_optimal_assignments"], 1)
        self.assertLess(
            record["exact_target_label_p_for_at_least_observed_correct"], 0.001
        )

    def test_grouped_decomposition_is_group_held_out_and_fold_local(self) -> None:
        rng = np.random.default_rng(23)
        groups = np.repeat(np.arange(30), 5)
        descriptors = rng.normal(size=(len(groups), 3))
        slopes = rng.normal(size=(3, 8))
        slopes -= slopes.mean(axis=1, keepdims=True)
        common = rng.normal(scale=4.0, size=(len(groups), 1))
        matrix = (
            common
            + descriptors @ slopes
            + 0.35 * rng.normal(size=(len(groups), 8))
        )
        result = grouped_descriptor_decomposition(
            matrix, descriptors, groups.astype(str), n_splits=5
        )
        self.assertTrue(
            all(row["group_overlap_count"] == 0 for row in result["fold_rows"])
        )
        self.assertTrue(
            result["metrics"]["strict_fold_local_target_offsets"]
        )
        self.assertTrue(
            result["metrics"]["strict_fold_local_descriptor_scaling"]
        )
        self.assertGreater(result["metrics"]["mean_out_of_fold_target_r2"], 0.75)
        self.assertGreater(
            result["metrics"]["out_of_fold_pr_after_descriptor_removal"],
            result["metrics"]["out_of_fold_pr_before_descriptor_removal"],
        )

    def test_signed_family_qap_detects_family_aligned_latent_factors(self) -> None:
        rng = np.random.default_rng(29)
        n = 1200
        families = [label for label in ["A", "B", "C", "D"] for _ in range(3)]
        latent = rng.normal(size=(n, 4))
        before = np.column_stack([
            latent[:, family] + 0.2 * rng.normal(size=n)
            for family in range(4)
            for _ in range(3)
        ])
        after = two_way_center(before)
        rows = family_qap(before, after, families, permutations=999, seed=31)
        first = rows[0]
        self.assertGreater(first["signed_within_minus_between"], 0.8)
        self.assertLessEqual(first["qap_one_sided_p_for_positive_statistic"], 0.01)


@unittest.skipUnless(
    (RESULTS / "analysis_summary.json").exists(),
    "run residual_mechanism_analysis.py to create frozen outputs",
)
class ResidualMechanismFrozenOutputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.summary = json.loads((RESULTS / "analysis_summary.json").read_text())
        cls.dataset_metrics = pd.read_csv(RESULTS / "dataset_metrics.csv")
        cls.fold_diagnostics = pd.read_csv(RESULTS / "fold_diagnostics.csv")

    def test_large_panel_pr_values_and_descriptor_signal(self) -> None:
        docking = self.summary["datasets"]["Docking-44"]
        dockstring = self.summary["datasets"]["DOCKSTRING-58"]
        self.assertAlmostEqual(
            docking["raw_correlation_moments"]["participation_ratio"],
            1.834045574797235,
            places=10,
        )
        self.assertAlmostEqual(
            docking["residual_correlation_moments"]["participation_ratio"],
            9.30291194271645,
            places=9,
        )
        self.assertGreater(
            docking["descriptor_decomposition"]["out_of_fold_residual_pc1_r2"],
            0.55,
        )
        self.assertGreater(
            dockstring["descriptor_decomposition"]["out_of_fold_residual_pc1_r2"],
            0.65,
        )
        self.assertGreater(
            docking["descriptor_decomposition"][
                "out_of_fold_pr_after_descriptor_removal"
            ],
            docking["descriptor_decomposition"][
                "out_of_fold_pr_before_descriptor_removal"
            ],
        )
        self.assertGreater(
            dockstring["descriptor_decomposition"][
                "out_of_fold_pr_after_descriptor_removal"
            ],
            dockstring["descriptor_decomposition"][
                "out_of_fold_pr_before_descriptor_removal"
            ],
        )
        self.assertGreater(
            docking["descriptor_decomposition"][
                "out_of_fold_relative_mean_squared_target_correlation_reduction"
            ],
            0.4,
        )

    def test_anova_labels_and_cross_panel_replication(self) -> None:
        docking = self.summary["datasets"]["Docking-44"]
        dockstring = self.summary["datasets"]["DOCKSTRING-58"]
        docking_fractions = docking["orthogonal_two_way_anova"][
            "component_fractions"
        ]
        dockstring_fractions = dockstring["orthogonal_two_way_anova"][
            "component_fractions"
        ]
        self.assertAlmostEqual(
            docking_fractions["ligand_main_effect"], 0.5381825, places=6
        )
        self.assertAlmostEqual(
            docking_fractions["target_main_effect"], 0.1943530, places=6
        )
        self.assertAlmostEqual(
            dockstring_fractions["ligand_target_interaction"], 0.3116942, places=6
        )
        primary = self.summary["cross_panel_descriptor_fingerprint"]["primary"]
        self.assertAlmostEqual(
            primary["omnibus_mean_descriptor_spearman_rho"],
            0.8166666666666665,
            places=12,
        )
        self.assertLess(
            primary["exact_target_label_permutation_two_sided_p"], 0.003
        )
        retrieval = self.summary["cross_panel_descriptor_fingerprint"][
            "exploratory_target_identity_retrieval"
        ]["primary"]
        self.assertEqual(retrieval["correct_target_identities"], 5)
        self.assertEqual(retrieval["number_of_cost_optimal_assignments"], 1)
        self.assertLess(
            retrieval["exact_target_label_p_for_at_least_observed_correct"],
            0.004,
        )

    def test_every_fold_has_zero_group_overlap(self) -> None:
        self.assertEqual(len(self.fold_diagnostics), 10)
        self.assertTrue((self.fold_diagnostics.group_overlap_count == 0).all())

    def test_pocket_result_is_explicitly_exploratory(self) -> None:
        pocket = self.summary["exploratory_pocket_volume"]
        self.assertEqual(pocket["analysis_status"], "exploratory_post_hoc")
        self.assertIn("does not contain an executable", pocket["provenance_limitation"])
        self.assertLess(
            pocket["pocket_volume_vs_residual_mw_slope_spearman_rho"], -0.5
        )

    def test_output_hash_contracts(self) -> None:
        for contract in self.summary["output_contracts"].values():
            path = ROOT / contract["path"]
            self.assertTrue(path.is_file())
            self.assertEqual(sha256_file(path), contract["sha256"])


if __name__ == "__main__":
    unittest.main()
