#!/usr/bin/env python3
"""Fast invariants for the dense DAVIS benchmark helpers."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from dense_davis_benchmark import (
    PAIRWISE_CONTRASTS,
    TARGET_MAP,
    _hit_detection_auc,
    _panel_analysis,
    _pairwise_metrics,
    _score_representations,
)


class DenseDavisBenchmarkTests(unittest.TestCase):
    def setUp(self) -> None:
        self.reference = np.asarray(
            [
                [-7.0, -8.0, -6.0],
                [-9.0, -7.0, -8.0],
                [-6.0, -6.5, -7.5],
                [-8.0, -9.0, -7.0],
            ]
        )
        self.evaluation = np.asarray(
            [[-8.0, -7.0, -6.0], [-6.0, -8.0, -7.0]]
        )
        self.experiment = np.asarray([[7.0, 6.0, 5.0], [5.0, 8.0, 6.0]])
        self.censored = self.experiment == 5.0

    def test_fixed_target_map_has_21_unique_targets(self) -> None:
        self.assertEqual(len(TARGET_MAP), 21)
        self.assertEqual(len(set(TARGET_MAP.values())), 21)

    def test_unscaled_row_centering_preserves_rankings(self) -> None:
        representations, _ = _score_representations(
            self.reference, self.evaluation
        )
        first, _ = _pairwise_metrics(
            representations["target_centered_unscaled"],
            self.experiment,
            self.censored,
            0.0,
            "all_informative",
        )
        second, _ = _pairwise_metrics(
            representations["two_way_centered_unscaled"],
            self.experiment,
            self.censored,
            0.0,
            "all_informative",
        )
        self.assertEqual(
            first["mean_per_ligand_pairwise_concordance"],
            second["mean_per_ligand_pairwise_concordance"],
        )

    def test_hit_auc_equals_floor_crossing_concordance(self) -> None:
        auc, auc_vector = _hit_detection_auc(self.evaluation, self.censored)
        concordance, concordance_vector = _pairwise_metrics(
            self.evaluation,
            self.experiment,
            self.censored,
            0.0,
            "floor_vs_uncensored",
        )
        np.testing.assert_allclose(auc_vector, concordance_vector)
        self.assertEqual(
            auc["mean_per_ligand_auc"],
            concordance["mean_per_ligand_pairwise_concordance"],
        )

    def test_two_censored_cells_never_enter(self) -> None:
        experiment = np.asarray([[5.0, 5.0, 7.0]])
        censored = np.asarray([[True, True, False]])
        summary, _ = _pairwise_metrics(
            np.asarray([[-8.0, -7.0, -6.0]]),
            experiment,
            censored,
            0.0,
            "all_informative",
        )
        self.assertEqual(summary["excluded_both_censored_pairs"], 1)
        self.assertEqual(summary["evaluated_pairs"], 2)

    def test_target_prior_and_offset_step_have_paired_cluster_uncertainty(self) -> None:
        self.assertIn(
            ("absolute_vina", "docking_target_prior"), PAIRWISE_CONTRASTS
        )
        targets = ["T1", "T2", "T3"]
        reference = pd.DataFrame(self.reference, columns=targets)
        docking = pd.DataFrame(self.evaluation, columns=targets)
        local_experiment = np.asarray([[9.0, 7.0, 5.0], [5.0, 9.0, 7.0]])
        experiment = pd.DataFrame(local_experiment, columns=targets)
        censored = pd.DataFrame(local_experiment == 5.0, columns=targets)
        report, _, _ = _panel_analysis(
            reference,
            docking,
            experiment,
            censored,
            pd.Series(["CCO", "c1ccccc1"]),
            targets,
            bootstrap_repeats=8,
            seed=13,
            include_uncertainty=True,
        )
        prior = report["docking_target_prior_baseline"]
        self.assertIsNotNone(prior["mean_per_ligand_pairwise_concordance"])
        self.assertIsNotNone(prior["mean_within_ligand_spearman"])
        self.assertIsNotNone(prior["top1_accuracy_allowing_experimental_ties"])
        for contrast in (
            "absolute_vina_minus_docking_target_prior",
            "target_centered_unscaled_minus_absolute_vina",
        ):
            by_stratum = report["paired_comparisons"][contrast]["0.0"]
            self.assertEqual(
                set(by_stratum),
                {"all_informative", "both_uncensored", "floor_vs_uncensored"},
            )
            uncertainty = by_stratum["all_informative"]["uncertainty"]
            self.assertEqual(
                set(uncertainty),
                {
                    "ligand_bootstrap",
                    "murcko_cluster_bootstrap",
                    "butina_cluster_bootstrap",
                },
            )
        self.assertIn("exploratory", report["operational_path_steps"]["status"])
        self.assertIn(
            "outcome-conditioned",
            report["pair_strata_status"]["both_uncensored"],
        )


if __name__ == "__main__":
    unittest.main()
