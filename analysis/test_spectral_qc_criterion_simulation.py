#!/usr/bin/env python3
"""Invariants for the spectral-QC criterion falsification study."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from spectral_qc_criterion_simulation import (
    SimulationConfig,
    apply_rules,
    correlation_pr,
    median_pairwise_seed_spearman,
    preference_accuracy,
    run_simulation,
    spectral_metrics,
    two_way_center,
)


class SpectralQCCriterionSimulationTests(unittest.TestCase):
    def test_ligand_effect_plus_independent_noise_falsifies_expansion_as_accuracy(self) -> None:
        rng = np.random.default_rng(71)
        n_ligands, n_targets = 3000, 12
        matrix = rng.normal(0, 7, size=(n_ligands, 1)) + rng.normal(
            size=(n_ligands, n_targets)
        )
        unrelated_truth = rng.normal(size=(n_ligands, n_targets))
        metrics = spectral_metrics(matrix)
        self.assertGreater(metrics["expansion_log2"], 2.5)
        self.assertAlmostEqual(
            preference_accuracy(matrix, unrelated_truth), 0.5, delta=0.02
        )

    def test_target_label_permutation_preserves_observable_spectral_metrics(self) -> None:
        rng = np.random.default_rng(113)
        truth = rng.normal(size=(800, 9))
        row_effect = rng.normal(0, 5, size=(800, 1))
        seed_surfaces = np.asarray(
            [row_effect + truth + rng.normal(0, 0.1, truth.shape) for _ in range(3)]
        )
        permutation = np.roll(np.arange(9), 1)
        shuffled = seed_surfaces[:, :, permutation]
        first = spectral_metrics(np.median(seed_surfaces, axis=0))
        second = spectral_metrics(np.median(shuffled, axis=0))
        for metric in ["raw_pr", "residual_pr", "expansion_log2"]:
            self.assertAlmostEqual(first[metric], second[metric], places=12)
        first_stability = median_pairwise_seed_spearman(seed_surfaces)
        second_stability = median_pairwise_seed_spearman(shuffled)
        for metric in first_stability:
            self.assertAlmostEqual(
                first_stability[metric], second_stability[metric], places=12
            )
        self.assertGreater(
            preference_accuracy(np.median(seed_surfaces, axis=0), truth), 0.95
        )
        self.assertLess(
            preference_accuracy(np.median(shuffled, axis=0), truth), 0.58
        )

    def test_raw_seed_stability_can_be_inflated_by_ligand_main_effect(self) -> None:
        rng = np.random.default_rng(191)
        row_effect = rng.normal(0, 8, size=(500, 1))
        surfaces = np.asarray(
            [row_effect + rng.normal(size=(500, 10)) for _ in range(3)]
        )
        stability = median_pairwise_seed_spearman(surfaces)
        self.assertGreater(stability["raw_seed_spearman"], 0.90)
        self.assertLess(abs(stability["residual_seed_spearman"]), 0.05)

    def test_primary_gate_requires_both_reproducibility_and_structure(self) -> None:
        base = {
            "expansion_log2": 3.0,
            "residual_seed_spearman": 0.85,
            "residual_pr_null_ratio": 0.60,
            "residual_pr_null_lower_tail_p": 0.025,
        }
        self.assertTrue(apply_rules(base)["coherence_gate_pass"])
        unstable = {**base, "residual_seed_spearman": 0.1}
        self.assertFalse(apply_rules(unstable)["coherence_gate_pass"])
        independent = {**base, "residual_pr_null_ratio": 0.98}
        self.assertFalse(apply_rules(independent)["coherence_gate_pass"])

    def test_correlation_pr_respects_row_centered_rank_ceiling(self) -> None:
        rng = np.random.default_rng(17)
        residual = two_way_center(rng.normal(size=(2000, 11)))
        self.assertLessEqual(correlation_pr(residual), 10.0 + 1e-9)

    def test_small_run_emits_machine_readable_falsification_package(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            config = SimulationConfig(
                n_ligands=80,
                n_targets=8,
                n_target_families=2,
                replicates=3,
                null_draws=19,
                seed=31,
            )
            manifest = run_simulation(config, output)
            replicates = pd.read_csv(output / "replicate_metrics.csv")
            self.assertEqual(len(replicates), 3 * 6)
            self.assertEqual(replicates["regime"].nunique(), 6)
            self.assertTrue((output / "regime_summary.csv").exists())
            falsification = json.loads(
                (output / "falsification_summary.json").read_text()
            )
            self.assertIn("target_label_permutation_invariance", falsification["counterexamples"])
            self.assertIn("outputs", manifest)


if __name__ == "__main__":
    unittest.main()
