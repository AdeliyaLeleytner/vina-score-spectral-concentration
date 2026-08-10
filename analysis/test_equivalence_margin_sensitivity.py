from __future__ import annotations

import json
import unittest
from pathlib import Path

import equivalence_margin_sensitivity as sensitivity
import make_reported_results as reported


PACKAGE = Path(__file__).resolve().parents[1]
RESULTS = PACKAGE / "results"
ARTIFACT = RESULTS / "equivalence_margin_sensitivity" / "summary.json"

PRIMARY = ("DAVIS", "PKIS2", "DOCKSTRING-ChEMBL")


def rebuild() -> dict:
    return sensitivity.build(
        json.loads((RESULTS / "dense_davis_benchmark.json").read_text()),
        json.loads((RESULTS / "dense_pkis2_benchmark.json").read_text()),
        json.loads(
            (RESULTS / "dockstring_chembl_ranking" / "summary.json").read_text()
        ),
    )


class EquivalenceMarginSensitivityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.result = rebuild()
        cls.artifact = json.loads(ARTIFACT.read_text())

    def test_margin_grid_is_the_requested_010_to_060_by_005(self) -> None:
        grid = self.result["margin_grid"]
        self.assertEqual(
            grid["labels"],
            [
                "0.010",
                "0.015",
                "0.020",
                "0.025",
                "0.030",
                "0.035",
                "0.040",
                "0.045",
                "0.050",
                "0.055",
                "0.060",
            ],
        )
        self.assertEqual(len(grid["values"]), 11)
        for benchmark in PRIMARY:
            decisions = self.result["benchmarks"][benchmark]["margin_decisions_90"]
            self.assertEqual(list(decisions), grid["labels"])

    def test_decisions_are_monotone_and_agree_with_the_interval(self) -> None:
        """A TOST decision can only turn on as the margin widens, never off again."""
        for benchmark in PRIMARY:
            block = self.result["benchmarks"][benchmark]
            reach = max(
                abs(value) for value in block["conservative_union_interval_90"]
            )
            self.assertAlmostEqual(block["interval_90_half_width"], reach, places=15)
            seen_true = False
            for label, passes in block["margin_decisions_90"].items():
                margin = float(label)
                self.assertEqual(passes, reach < margin, msg=f"{benchmark} {label}")
                if passes:
                    seen_true = True
                else:
                    self.assertFalse(seen_true, msg=f"{benchmark} not monotone at {label}")

    def test_union_is_conservative_relative_to_each_cluster_scheme(self) -> None:
        for benchmark in PRIMARY:
            block = self.result["benchmarks"][benchmark]
            union = block["conservative_union_interval_90"]
            schemes = block["cluster_bootstrap_intervals_90"]
            self.assertEqual(len(schemes), 2)
            for name, interval in schemes.items():
                self.assertLessEqual(union[0], interval[0], msg=f"{benchmark} {name}")
                self.assertGreaterEqual(union[1], interval[1], msg=f"{benchmark} {name}")
            self.assertGreaterEqual(union[0], -1.0)
            self.assertLessEqual(union[1], 1.0)

    def test_broad_union_reproduces_the_frozen_headline_interval(self) -> None:
        """Recomputing the union must not disagree with what the benchmark published."""
        block = self.result["benchmarks"]["DOCKSTRING-ChEMBL"]
        self.assertTrue(block["matches_frozen_headline_interval_90"])
        self.assertEqual(
            block["conservative_union_interval_90"],
            block["frozen_headline_interval_90"],
        )

    def test_smallest_supported_margins_are_the_package_helper_values(self) -> None:
        for benchmark in PRIMARY:
            block = self.result["benchmarks"][benchmark]
            self.assertEqual(
                block["smallest_supported_symmetric_margin_90_rounded_up_3dp"],
                reported.smallest_equivalence_margin(
                    block["conservative_union_interval_90"]
                ),
            )
            grid_margin = block["smallest_grid_margin_passing"]
            self.assertIsNotNone(grid_margin, msg=benchmark)
            self.assertGreater(
                grid_margin, block["smallest_supported_symmetric_margin_90_exact"]
            )

    def test_pkis2_rounding_and_grid_conventions_are_not_conflated(self) -> None:
        """PKIS2 has an exact reach of 0.015512: report 0.016, grid point 0.020."""
        block = self.result["benchmarks"]["PKIS2"]
        self.assertAlmostEqual(
            block["smallest_supported_symmetric_margin_90_exact"],
            0.015512213214719999,
            places=15,
        )
        self.assertEqual(
            block["smallest_supported_symmetric_margin_90_rounded_up_3dp"], 0.016
        )
        self.assertEqual(block["smallest_grid_margin_passing"], 0.020)
        self.assertEqual(
            reported.smallest_grid_margin(block["conservative_union_interval_90"]),
            0.020,
        )
        # Strict TOST containment means an interval that reaches the grid boundary does
        # not pass at that same boundary.
        self.assertEqual(reported.smallest_grid_margin([-0.030, 0.010]), 0.035)

    def test_headline_decisions_are_the_expected_frozen_ones(self) -> None:
        """DAVIS needs 0.030; PKIS2 and the broad benchmark pass from 0.020."""
        summary = self.result["cross_benchmark_summary"]
        self.assertEqual(summary["smallest_grid_margin_passing"]["DAVIS"], 0.030)
        self.assertEqual(summary["smallest_grid_margin_passing"]["PKIS2"], 0.020)
        self.assertEqual(
            summary["smallest_grid_margin_passing"]["DOCKSTRING-ChEMBL"], 0.020
        )
        self.assertEqual(summary["smallest_grid_margin_passing_on_all_three"], 0.030)
        self.assertFalse(
            self.result["benchmarks"]["DAVIS"]["margin_decisions_90"]["0.020"]
        )

    def test_binder_only_strata_do_not_pass_any_margin(self) -> None:
        """The equivalence reading is confined to the primary strata; say so honestly."""
        for label in ("DAVIS_both_uncensored", "PKIS2_both_active"):
            block = self.result["secondary_outcome_conditioned_strata"][label]
            self.assertIsNone(block["smallest_grid_margin_passing"], msg=label)
            self.assertTrue(block["excludes_zero_90"], msg=label)

    def test_required_output_contract_fields_are_present(self) -> None:
        for key in (
            "analysis_question",
            "support",
            "seeds",
            "status",
            "claim_boundary",
        ):
            self.assertIn(key, self.result)
        for benchmark in PRIMARY:
            support = self.result["support"][benchmark]
            self.assertGreater(support["evaluated_ligands"], 0)
            self.assertGreater(support["targets"], 0)
            self.assertGreater(support["evaluated_within_ligand_target_pairs"], 0)
            self.assertEqual(support["bootstrap_replicates_per_scheme"], 5000)
        self.assertIn("post hoc", self.result["prespecification"])
        self.assertIn("withdrawn", self.result["prespecification"])
        self.assertIn("not a confirmatory", self.result["claim_boundary"])

    def test_registered_artifact_matches_a_fresh_build(self) -> None:
        self.assertEqual(
            json.dumps(self.artifact, indent=2, sort_keys=True),
            json.dumps(self.result, indent=2, sort_keys=True),
        )

    def test_no_wall_clock_or_timing_fields_leak_into_the_artifact(self) -> None:
        text = json.dumps(self.artifact)
        for banned in ("runtime_seconds", "wall_seconds", "elapsed", "timestamp"):
            self.assertNotIn(banned, text)


if __name__ == "__main__":
    unittest.main()
