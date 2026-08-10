#!/usr/bin/env python3
"""Fast invariants for the dense PKIS2 benchmark helpers."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from dense_pkis2_benchmark import (
    ACTIVITY_THRESHOLD,
    MARGINS,
    PAIRWISE_CONTRASTS,
    PRIMARY_MARGIN,
    TARGET_MAP,
    _aggregate_rows,
    _binary_activity_metrics,
    _panel_analysis,
    _pairwise_concordance,
    _score_representations,
    _size_matched_reference_sensitivity,
)


class DensePkis2BenchmarkTests(unittest.TestCase):
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
        self.experiment = np.asarray(
            [[90.0, 70.0, 40.0], [40.0, 90.0, 60.0]]
        )

    def test_fixed_target_map_has_21_unique_targets_and_constructs(self) -> None:
        self.assertEqual(len(TARGET_MAP), 21)
        self.assertEqual(len(set(TARGET_MAP)), 21)
        self.assertEqual(len(set(TARGET_MAP.values())), 21)
        self.assertEqual(TARGET_MAP["ABL1"], "ABL1-nonphosphorylated")
        self.assertEqual(TARGET_MAP["KDR"], "VEGFR2")

    def test_margin_is_strictly_greater_than_requested_difference(self) -> None:
        summary, values, counts = _pairwise_concordance(
            np.asarray([[-9.0, -8.0, -7.0]]),
            np.asarray([[70.0, 60.0, 49.0]]),
            margin=10.0,
        )
        # The 70-vs-60 difference is exactly 10 and is excluded; the other two
        # target pairs exceed 10 percentage points.
        self.assertEqual(summary["evaluated_pairs"], 2)
        self.assertEqual(counts.tolist(), [2])
        self.assertEqual(values.tolist(), [1.0])

    def test_score_ties_receive_half_credit(self) -> None:
        summary, values, _ = _pairwise_concordance(
            np.asarray([[-8.0, -8.0]]),
            np.asarray([[80.0, 20.0]]),
            margin=10.0,
        )
        self.assertEqual(summary["predicted_score_ties_half_credit"], 1)
        self.assertEqual(values.tolist(), [0.5])

    def test_unscaled_row_centering_preserves_within_ligand_rankings(self) -> None:
        representations, _ = _score_representations(
            self.reference, self.evaluation
        )
        centered, _, _ = _pairwise_concordance(
            representations["target_centered_unscaled"],
            self.experiment,
            margin=0.0,
        )
        two_way, _, _ = _pairwise_concordance(
            representations["two_way_centered_unscaled"],
            self.experiment,
            margin=0.0,
        )
        self.assertEqual(
            centered["mean_per_ligand_pairwise_concordance"],
            two_way["mean_per_ligand_pairwise_concordance"],
        )

    def test_binary_auc_equals_active_inactive_pairwise_concordance(self) -> None:
        auc_summary, auc, average_precision = _binary_activity_metrics(
            self.evaluation, self.experiment
        )
        pair_summary, pair_values, _ = _pairwise_concordance(
            self.evaluation,
            self.experiment,
            margin=0.0,
            pair_stratum="active_vs_inactive",
        )
        np.testing.assert_allclose(auc, pair_values, atol=1e-15)
        self.assertEqual(
            auc_summary["mean_per_ligand_roc_auc"],
            pair_summary["mean_per_ligand_pairwise_concordance"],
        )
        self.assertTrue(np.isfinite(average_precision).all())

    def test_binary_metrics_skip_single_class_ligands(self) -> None:
        summary, auc, average_precision = _binary_activity_metrics(
            np.asarray([[-8.0, -7.0, -6.0]]),
            np.asarray(
                [[ACTIVITY_THRESHOLD, ACTIVITY_THRESHOLD + 5.0, 100.0]]
            ),
        )
        self.assertEqual(summary["evaluated_ligands_with_both_classes"], 0)
        self.assertTrue(np.isnan(auc[0]))
        self.assertTrue(np.isnan(average_precision[0]))

    def test_both_active_stratum_is_outcome_conditioned(self) -> None:
        summary, _, counts = _pairwise_concordance(
            np.asarray([[-9.0, -8.0, -7.0, -6.0]]),
            np.asarray([[90.0, 80.0, 70.0, 60.0]]),
            margin=0.0,
            pair_stratum="both_active",
        )
        self.assertEqual(summary["evaluated_pairs"], 3)
        self.assertEqual(counts.tolist(), [3])

    def test_both_active_primary_margin_is_strictly_greater_than_ten(self) -> None:
        summary, _, counts = _pairwise_concordance(
            np.asarray([[-9.0, -8.0, -7.0, -6.0]]),
            np.asarray([[90.0, 80.0, 70.0, 60.0]]),
            margin=MARGINS[PRIMARY_MARGIN],
            pair_stratum="both_active",
        )
        # The 90-vs-80 and 80-vs-70 differences are exactly 10 percentage
        # points and are excluded; only the 90-vs-70 comparison remains.
        self.assertEqual(summary["evaluated_pairs"], 1)
        self.assertEqual(counts.tolist(), [1])

    def test_duplicate_aggregation_is_cellwise(self) -> None:
        frame = pd.DataFrame(
            {
                "identity": ["A", "A", "B"],
                "T1": [10.0, 30.0, 50.0],
                "T2": [20.0, 60.0, 70.0],
            }
        )
        median = _aggregate_rows(frame, "identity", ["T1", "T2"], "median")
        self.assertEqual(median.loc["A", "T1"], 20.0)
        self.assertEqual(median.loc["A", "T2"], 40.0)

    def test_target_prior_offset_and_both_active_paired_uncertainty(self) -> None:
        self.assertIn(
            ("absolute_vina", "docking_target_prior"), PAIRWISE_CONTRASTS
        )
        targets = ["T1", "T2", "T3"]
        reference = pd.DataFrame(self.reference, columns=targets)
        docking = pd.DataFrame(self.evaluation, columns=targets)
        local_experiment = np.asarray(
            [[90.0, 75.0, 40.0], [70.0, 90.0, 50.0]]
        )
        experiment = pd.DataFrame(local_experiment, columns=targets)
        experiment.index = ["A", "B"]
        docking.index = experiment.index
        report, tables = _panel_analysis(
            reference,
            docking,
            experiment,
            pd.Series(["CCO", "c1ccccc1"], index=experiment.index),
            bootstrap_repeats=8,
            seed=17,
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
            primary = report["paired_comparisons"][contrast][
                "absolute_difference_gt_10"
            ]
            both_active = report[
                "exploratory_both_active_paired_comparisons"
            ][contrast]
            exact_non_tie_both_active = report[
                "secondary_exact_non_tie_both_active_paired_comparisons"
            ][contrast]
            for record in (primary, both_active):
                self.assertEqual(
                    set(record["uncertainty"]),
                    {
                        "ligand_bootstrap",
                        "murcko_cluster_bootstrap",
                        "butina_cluster_bootstrap",
                    },
                )
            self.assertEqual(both_active["pair_stratum"], "both_active")
            self.assertEqual(
                both_active["experimental_margin_name"], PRIMARY_MARGIN
            )
            self.assertEqual(
                both_active["experimental_margin_percentage_points"], 10.0
            )
            self.assertIn("outcome-conditioned", both_active["status"])
            self.assertEqual(
                exact_non_tie_both_active["experimental_margin_name"],
                "exact_non_ties",
            )
            self.assertEqual(
                exact_non_tie_both_active[
                    "experimental_margin_percentage_points"
                ],
                0.0,
            )
        self.assertEqual(
            len(tables["exploratory_both_active_paired_comparisons"]),
            len(PAIRWISE_CONTRASTS),
        )
        self.assertEqual(
            len(
                tables[
                    "secondary_exact_non_tie_both_active_paired_comparisons"
                ]
            ),
            len(PAIRWISE_CONTRASTS),
        )
        for record in report[
            "exploratory_both_active_pairwise_concordance"
        ].values():
            self.assertEqual(record["experimental_margin_name"], PRIMARY_MARGIN)
        for record in report[
            "secondary_exact_non_tie_both_active_pairwise_concordance"
        ].values():
            self.assertEqual(record["experimental_margin_name"], "exact_non_ties")
        self.assertIn("unadjusted", report["both_active_status"])
        self.assertIn(
            "exact-non-tie", report["secondary_exact_non_tie_both_active_status"]
        )
        self.assertIn("exploratory", report["operational_path_steps"]["status"])

    def test_size_matched_reference_reports_each_requested_support(self) -> None:
        rng = np.random.default_rng(7)
        targets = list(TARGET_MAP)
        dockstring = pd.DataFrame(
            rng.normal(-7.0, 1.0, size=(12, len(targets))),
            columns=targets,
        )
        dockstring.insert(0, "dockstring_row", np.arange(len(dockstring)))
        dockstring["molecular_weight"] = np.linspace(250.0, 500.0, len(dockstring))
        dockstring["heavy_atom_count"] = np.arange(20, 32)
        pkis2 = pd.DataFrame(
            {
                "standard_inchikey": ["A", "B"],
                "molecular_weight": [300.0, 450.0],
                "heavy_atom_count": [22, 29],
            }
        )
        primary_docking = pd.DataFrame(
            rng.normal(-7.0, 1.0, size=(2, len(targets))),
            index=["A", "B"],
            columns=targets,
        )
        primary_experiment = pd.DataFrame(
            rng.uniform(0.0, 100.0, size=(2, len(targets))),
            index=["A", "B"],
            columns=targets,
        )
        report, records = _size_matched_reference_sensitivity(
            dockstring,
            pd.Series(False, index=dockstring.index),
            pkis2,
            primary_docking,
            primary_experiment,
            neighbours_per_evaluation_ligand=(3, 5),
        )
        self.assertEqual(report["status"], "complete")
        self.assertEqual(len(records), 2)
        self.assertEqual(
            report["by_neighbour_count"]["3"][
                "matched_reference_rows_with_reuse"
            ],
            6,
        )
        self.assertEqual(
            len(report["standardized_mean_difference_full_reference_minus_evaluation"]),
            2,
        )


if __name__ == "__main__":
    unittest.main()
