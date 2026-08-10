#!/usr/bin/env python3
"""Tests for the non-Vina scoring-function transport analysis."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

from nonvina_scorer_transport import (  # noqa: E402
    BOOTSTRAP_METRICS,
    PC1_DOMINANCE_THRESHOLD,
    PC1_UNIFORM_COSINE_THRESHOLD,
    RAW_PR_FRACTION_THRESHOLD,
    SCORER_FAMILY,
    TARGETS,
    butina_clusters,
    conservative_interval,
    evaluate_prediction,
    transport_metrics,
)
import dockstring_vina_term_decomposition as terms  # noqa: E402


RESULTS = ROOT / "results" / "nonvina_scorer_transport"
SUMMARY_PATH = RESULTS / "summary.json"
METRICS_PATH = RESULTS / "scorer_spectral_metrics.csv"


class TransportMetricUnitTests(unittest.TestCase):
    def test_shared_axis_matrix_collapses_the_spectrum(self) -> None:
        """A single ligand-wide factor gives PR ~ 1, PC1 ~ 1 and a uniform loading."""
        rng = np.random.default_rng(3)
        factor = rng.normal(size=(500, 1))
        matrix = factor @ np.ones((1, 9)) + 0.05 * rng.normal(size=(500, 9))
        values = transport_metrics(matrix)
        self.assertGreater(values["raw_pc1_fraction"], 0.98)
        self.assertLess(values["raw_participation_ratio"], 1.1)
        self.assertGreater(values["pc1_uniform_cosine"], 0.999)
        self.assertGreater(values["pc1_min_loading"], 0.0)
        # Removing each ligand's own mean deletes the shared factor, so the
        # participation ratio must rise back towards the target count.
        self.assertGreater(values["row_centered_participation_ratio"], 7.0)
        self.assertGreater(values["participation_ratio_increase"], 6.0)

    def test_independent_targets_give_full_rank_spectrum(self) -> None:
        rng = np.random.default_rng(5)
        matrix = rng.normal(size=(4000, 9))
        values = transport_metrics(matrix)
        self.assertAlmostEqual(values["raw_participation_ratio"], 9.0, delta=0.2)
        self.assertAlmostEqual(values["raw_pc1_fraction"], 1 / 9, delta=0.03)
        self.assertLess(values["raw_mean_squared_offdiagonal_correlation"], 0.01)

    def test_within_ligand_centring_equals_two_way_centring(self) -> None:
        """The reported ``row_centered_*`` metrics are the package's residual metrics.

        Column correlation removes column means, so subtracting only the ligand
        mean and subtracting both main effects give the same correlation matrix.
        """
        rng = np.random.default_rng(13)
        matrix = rng.normal(size=(300, 9)) + rng.normal(size=(300, 1)) * 2.0
        values = transport_metrics(matrix)
        two_way = terms.spectral_summary(matrix)
        self.assertAlmostEqual(
            values["row_centered_participation_ratio"], two_way["residual_pr"], places=9
        )
        self.assertAlmostEqual(
            values["row_centered_pc1_fraction"], two_way["residual_pc1_fraction"], places=9
        )
        self.assertAlmostEqual(values["raw_participation_ratio"], two_way["raw_pr"], places=9)

    def test_metrics_are_invariant_to_scorer_polarity_and_units(self) -> None:
        """Vina reports kcal/mol and RF-Score reports pKd; the metrics must not care."""
        rng = np.random.default_rng(17)
        matrix = rng.normal(size=(400, 9)) + rng.normal(size=(400, 1))
        baseline = transport_metrics(matrix)
        scaled = transport_metrics(-3.7 * matrix + 11.0)
        for key, value in baseline.items():
            if key in {"pc1_min_loading", "pc1_max_loading"}:
                continue
            self.assertAlmostEqual(value, scaled[key], places=9, msg=key)

    def test_conservative_interval_takes_the_union(self) -> None:
        self.assertEqual(conservative_interval([0.2, 0.6], [0.1, 0.5]), [0.1, 0.6])
        self.assertEqual(conservative_interval([-1.0, 1.0], [0.0, 0.2]), [-1.0, 1.0])

    def test_prediction_criteria_are_applied_as_stated(self) -> None:
        observed = {
            "raw_pc1_fraction": 0.60,
            "pc1_uniform_cosine": 0.97,
            "pc1_min_loading": 0.10,
            "raw_participation_ratio": 2.5,
        }
        intervals = {"participation_ratio_increase": [0.4, 3.0]}
        self.assertTrue(evaluate_prediction(observed, intervals, 9)["prediction_holds"])

        weak_pc1 = dict(observed, raw_pc1_fraction=0.40)
        self.assertFalse(evaluate_prediction(weak_pc1, intervals, 9)["pc1_dominant"])

        signed = dict(observed, pc1_min_loading=-0.05)
        self.assertFalse(evaluate_prediction(signed, intervals, 9)["pc1_near_uniform"])

        flat = {"participation_ratio_increase": [-0.1, 3.0]}
        self.assertFalse(
            evaluate_prediction(observed, flat, 9)[
                "participation_ratio_rises_under_within_ligand_centring"
            ]
        )

    def test_butina_clusters_are_deterministic_and_cover_every_ligand(self) -> None:
        smiles = [
            "c1ccccc1",
            "c1ccccc1C",
            "c1ccccc1CC",
            "CCCCCCCC",
            "CCCCCCCCC",
            "C1CCNCC1",
            "O=C(O)c1ccccc1",
            "O=C(O)c1ccccc1O",
        ]
        first = butina_clusters(smiles)
        second = butina_clusters(smiles)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(len(first), len(smiles))
        self.assertEqual(set(first), set(range(first.max() + 1)))


@unittest.skipUnless(
    SUMMARY_PATH.is_file() and METRICS_PATH.is_file(),
    "run analysis/nonvina_scorer_transport.py first",
)
class TransportArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.summary = json.loads(SUMMARY_PATH.read_text())
        cls.metrics = pd.read_csv(METRICS_PATH)

    def test_summary_carries_the_package_required_reporting_fields(self) -> None:
        for key in (
            "analysis",
            "question",
            "falsifiable_prediction",
            "support",
            "seeds",
            "status",
            "claim_boundary",
        ):
            self.assertIn(key, self.summary)
        support = self.summary["support"]
        self.assertEqual(support["n_targets"], len(TARGETS))
        self.assertEqual(support["targets"], list(TARGETS))
        self.assertEqual(support["n_cells"], support["n_ligands"] * support["n_targets"])
        self.assertGreaterEqual(support["n_ligands"], 512)
        self.assertGreaterEqual(support["n_non_vina_scorers"], 1)
        self.assertTrue(self.summary["claim_boundary"].strip())
        self.assertIn("re-docking", self.summary["claim_boundary"])
        boundary = self.summary["reproducibility_boundary"]
        self.assertFalse(boundary["independent_redocking_per_scorer"])
        self.assertIn("Python 3.9", boundary["source_restricted_oddt_stage"]["environment"])
        self.assertFalse(boundary["source_restricted_oddt_stage"]["redistributed"])
        self.assertEqual(self.summary["seeds"]["dockstring_primary_support_seed"], 71)

    def test_every_scorer_is_measured_on_the_identical_block(self) -> None:
        scorers = self.summary["support"]["scorers"]
        self.assertEqual(sorted(self.metrics.scorer), sorted(scorers))
        self.assertEqual(set(self.metrics.n_ligands), {self.summary["support"]["n_ligands"]})
        self.assertEqual(set(self.metrics.n_targets), {len(TARGETS)})
        self.assertIn("released_vina", scorers)
        for scorer in scorers:
            self.assertIn(scorer, SCORER_FAMILY)

    def test_reported_metrics_are_finite_and_within_their_admissible_range(self) -> None:
        n_targets = len(TARGETS)
        for _, row in self.metrics.iterrows():
            self.assertTrue(np.isfinite(row.raw_pc1_fraction))
            self.assertGreater(row.raw_pc1_fraction, 1.0 / n_targets - 1e-9)
            self.assertLessEqual(row.raw_pc1_fraction, 1.0 + 1e-9)
            self.assertGreaterEqual(row.raw_participation_ratio, 1.0 - 1e-9)
            self.assertLessEqual(row.raw_participation_ratio, n_targets + 1e-9)
            self.assertLessEqual(row.row_centered_participation_ratio, n_targets + 1e-9)
            self.assertGreaterEqual(row.raw_mean_squared_offdiagonal_correlation, 0.0)
            self.assertLessEqual(abs(row.pc1_uniform_cosine), 1.0 + 1e-9)
            for metric in BOOTSTRAP_METRICS:
                low = row[f"{metric}_ci_low"]
                high = row[f"{metric}_ci_high"]
                self.assertTrue(np.isfinite(low) and np.isfinite(high), metric)
                self.assertLessEqual(low, high, metric)

    def test_released_vina_reference_reproduces_the_frozen_low_rank_regime(self) -> None:
        """The reference column must land where the frozen 512-ligand block did."""
        row = self.metrics.set_index("scorer").loc["released_vina"]
        self.assertLess(row.raw_participation_ratio, RAW_PR_FRACTION_THRESHOLD * len(TARGETS))
        self.assertGreater(row.raw_pc1_fraction, PC1_DOMINANCE_THRESHOLD)
        self.assertGreater(row.pc1_uniform_cosine, PC1_UNIFORM_COSINE_THRESHOLD)
        self.assertGreater(row.row_centered_participation_ratio, row.raw_participation_ratio)
        frozen = pd.read_csv(ROOT / "results" / "dockstring_vina_terms" / "term_spectral_metrics.csv")
        frozen_released = frozen.set_index("component").loc["released_score"]
        self.assertAlmostEqual(row.raw_participation_ratio, frozen_released.raw_pr, delta=0.75)
        self.assertAlmostEqual(row.raw_pc1_fraction, frozen_released.raw_pc1_fraction, delta=0.10)

    def test_smina_vina_control_reproduces_the_released_score(self) -> None:
        """Guards the order-based alignment of smina output to ligand keys."""
        check = self.summary["smina_vina_reproduction_check"]
        if check.get("status") == "not_run":
            self.skipTest("smina stage was not run")
        self.assertGreater(check["minimum_pearson_r"], 0.95)
        self.assertEqual(sorted(check["per_target_pearson_r"]), sorted(TARGETS))

    def test_prediction_verdicts_follow_from_the_recorded_numbers(self) -> None:
        """Every published verdict is recomputable from the published metrics."""
        indexed = self.metrics.set_index("scorer")
        n_targets = len(TARGETS)
        for scorer, verdict in self.summary["prediction_outcome"].items():
            row = indexed.loc[scorer]
            self.assertEqual(
                verdict["pc1_dominant"],
                bool(row.raw_pc1_fraction >= PC1_DOMINANCE_THRESHOLD),
                scorer,
            )
            self.assertEqual(
                verdict["pc1_near_uniform"],
                bool(
                    row.pc1_uniform_cosine >= PC1_UNIFORM_COSINE_THRESHOLD
                    and row.pc1_min_loading > 0.0
                ),
                scorer,
            )
            self.assertEqual(
                verdict["raw_participation_ratio_low"],
                bool(row.raw_participation_ratio <= RAW_PR_FRACTION_THRESHOLD * n_targets),
                scorer,
            )
            self.assertEqual(
                verdict["participation_ratio_rises_under_within_ligand_centring"],
                bool(row.participation_ratio_increase_ci_low > 0.0),
                scorer,
            )
            self.assertEqual(verdict["prediction_holds"], bool(row.prediction_holds), scorer)

        summary_holds = set(self.summary["prediction_summary"]["non_vina_scorers_supporting_the_prediction"])
        summary_fails = set(self.summary["prediction_summary"]["non_vina_scorers_falsifying_the_prediction"])
        non_vina = {
            scorer
            for scorer in self.summary["support"]["scorers"]
            if SCORER_FAMILY[scorer] != "vina_family"
        }
        self.assertEqual(summary_holds | summary_fails, non_vina)
        self.assertFalse(summary_holds & summary_fails)

    def test_manuscript_vinardo_direction_prediction_is_adjudicated(self) -> None:
        """The Discussion states a Vinardo direction prediction; it must be judged here."""
        check = self.summary["manuscript_vinardo_direction_prediction"]
        if check["status"] != "tested":
            self.skipTest("Vinardo column absent")
        self.assertIn(
            check["verdict"],
            {"supported", "refuted_the_predicted_direction", "indeterminate"},
        )
        indexed = self.metrics.set_index("scorer")
        observed_pr_delta = (
            indexed.loc["vinardo"].raw_participation_ratio
            - indexed.loc["released_vina"].raw_participation_ratio
        )
        self.assertAlmostEqual(
            check["observed_raw_participation_ratio_vinardo_minus_vina"],
            observed_pr_delta,
            places=9,
        )
        low, high = check["raw_participation_ratio_delta_ci95"]
        self.assertLessEqual(low, high)
        # A verdict of "supported" or "refuted" must be backed by an interval that
        # excludes zero; otherwise the verdict must be "indeterminate".
        if check["verdict"] == "indeterminate":
            self.assertFalse(
                check["vinardo_raw_participation_ratio_is_higher_as_predicted"]
                and check["vinardo_raw_pc1_fraction_is_smaller_as_predicted"]
            )
        if check["vinardo_raw_participation_ratio_is_lower_contradicting_prediction"]:
            self.assertLess(high, 0.0)
        if check["vinardo_raw_participation_ratio_is_higher_as_predicted"]:
            self.assertGreater(low, 0.0)

    def test_bootstrap_contract_matches_the_package(self) -> None:
        bootstrap = self.summary["bootstrap"]
        self.assertEqual(bootstrap["replicates"], 5000)
        self.assertGreater(bootstrap["murcko_scaffold_clusters"], 1)
        self.assertGreater(bootstrap["butina_clusters"], 1)
        self.assertIn("Murcko", self.summary["interval_rule"])
        self.assertIn("Butina", self.summary["interval_rule"])
        for scheme in ("murcko", "butina"):
            for scorer in self.summary["support"]["scorers"]:
                self.assertIn(scorer, bootstrap[scheme])

    def test_both_interpreters_are_recorded(self) -> None:
        """The ODDT stage runs on a different Python than the analysis stage."""
        toolchain = self.summary["toolchain"]
        self.assertIn("analysis_interpreter", toolchain)
        oddt_stage = toolchain["oddt_stage"]
        if oddt_stage.get("status") == "skipped":
            self.skipTest("ODDT stage was not run")
        self.assertIn("interpreter", oddt_stage)
        self.assertNotEqual(
            oddt_stage["interpreter"].split()[0],
            toolchain["analysis_interpreter"].split()[0],
        )
        self.assertIn("oddt_version", oddt_stage)
        for scorer, entry in oddt_stage["model_provenance"].items():
            self.assertIn("model_pickle_sha256", entry)
            if entry.get("core_set_pearson_r") is not None:
                # The offline training path must reproduce the published PDBbind
                # core-set performance of each model.
                self.assertGreater(entry["core_set_pearson_r"], 0.70, scorer)
                self.assertLess(entry["core_set_pearson_r"], 0.95, scorer)

    def test_release_summary_contains_no_machine_specific_absolute_paths(self) -> None:
        serialized = json.dumps(self.summary)
        self.assertNotIn("/Users/", serialized)
        self.assertNotIn("/private/tmp/", serialized)
        self.assertIn("external-not-redistributed", serialized)


if __name__ == "__main__":
    unittest.main()
