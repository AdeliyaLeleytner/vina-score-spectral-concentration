from __future__ import annotations

import json
import unittest
from pathlib import Path

import vina_term_attribution_report as report


PACKAGE = Path(__file__).resolve().parents[1]
ARTIFACT = PACKAGE / "results" / "vina_term_attribution_report" / "summary.json"
TERMS_DIR = PACKAGE / "results" / "dockstring_vina_terms"
TRANSPORT = PACKAGE / "results" / "nonvina_scorer_transport" / "summary.json"


def rebuild() -> dict:
    summary_path = TERMS_DIR / "analysis_summary.json"
    return report.build(
        json.loads(summary_path.read_text()),
        report.read_csv_rows(TERMS_DIR / "term_spectral_metrics.csv"),
        report.read_csv_rows(TERMS_DIR / "fixed_pose_term_ablations.csv"),
        report.read_csv_rows(TERMS_DIR / "term_attribution.csv"),
        json.loads(TRANSPORT.read_text()),
        {},
    )


class VinaTermAttributionReportTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.result = rebuild()
        cls.artifact = json.loads(ARTIFACT.read_text())

    def test_support_matches_the_methods_paragraph(self) -> None:
        """512 ligands x 9 targets = 4,608 cells, exactly as manuscript.tex describes."""
        support = self.result["support"]
        self.assertEqual(support["n_ligands"], 512)
        self.assertEqual(support["n_targets"], 9)
        self.assertEqual(support["n_cells"], 4608)
        self.assertEqual(support["n_ligands"] * support["n_targets"], support["n_cells"])

    def test_score_reproduction_licenses_the_attribution(self) -> None:
        """The roundtrip claim in Methods: 99.6% within 0.10 kcal/mol, Pearson 0.9996."""
        reproduction = self.result["score_reproduction"]
        self.assertEqual(reproduction["cells_reproduced_within_0.10_kcal_mol"], 4590)
        self.assertAlmostEqual(
            reproduction["fraction_cells_reproduced_within_0.10_kcal_mol"],
            4590 / 4608,
            places=12,
        )
        self.assertGreater(reproduction["fraction_cells_reproduced_within_0.10_kcal_mol"], 0.995)
        self.assertGreater(reproduction["rescored_vs_released_cell_pearson"], 0.999)

    def test_term_shares_are_a_complete_decomposition(self) -> None:
        """The five shared row-axis shares partition the total, so they can be tabled."""
        shares = self.result["concentration_spread"]["shared_row_axis_covariance_shares"]
        self.assertEqual(set(shares), set(report.TERM_ORDER))
        self.assertAlmostEqual(sum(shares.values()), 1.0, places=6)
        self.assertGreater(shares["gauss2"], 0.5)
        self.assertLess(shares["repulsion"], 0.0)

    def test_every_term_row_carries_a_bootstrap_interval_bracketing_its_point(self) -> None:
        for record in self.result["terms"]:
            low, high = record["shared_row_axis_covariance_share_ci95"]
            self.assertLess(low, high, msg=record["term"])
            self.assertLessEqual(low, record["shared_row_axis_covariance_share"])
            self.assertGreaterEqual(high, record["shared_row_axis_covariance_share"])

    def test_table_prints_one_row_per_component_with_all_columns(self) -> None:
        table = self.result["supplementary_table"]
        self.assertEqual(tuple(table["columns"]), report.TABLE_COLUMNS)
        self.assertEqual(len(table["column_headers"]), len(report.TABLE_COLUMNS))
        rows = table["rows"]
        self.assertEqual(len(rows), len(report.REFERENCE_ORDER) + len(report.TERM_ORDER))
        for row in rows:
            self.assertEqual(set(row), set(report.TABLE_COLUMNS))
            self.assertNotEqual(row["raw_pr"], "")
        term_rows = {row["component"]: row for row in rows if row["component"] in report.TERM_ORDER}
        self.assertEqual(set(term_rows), set(report.TERM_ORDER))
        self.assertEqual(term_rows["gauss2"]["shared_row_axis_covariance_share"], "0.715")
        self.assertEqual(term_rows["repulsion"]["raw_pr"], "6.90")

    def test_claim_assessment_separates_descriptive_from_causal(self) -> None:
        """The strong causal reading must not be asserted: Vinardo contradicts it."""
        assessment = self.result["claim_assessment"]
        self.assertTrue(assessment["descriptive_reading_supported"])
        self.assertFalse(assessment["causal_reading_supported"])
        cross = self.result["refit_scorer_cross_check"]
        self.assertLess(
            abs(cross["vinardo_minus_vina_raw_pr"]),
            cross["arithmetic_gauss2_ablation_raw_pr_shift"],
        )

    def test_required_output_contract_fields_are_present(self) -> None:
        for key in (
            "analysis_question",
            "support",
            "seeds",
            "status",
            "claim_boundary",
        ):
            self.assertIn(key, self.result)
        self.assertTrue(self.result["claim_boundary"].strip())

    def test_registered_artifact_matches_a_fresh_build(self) -> None:
        """Determinism: the on-disk JSON must be reproducible from the frozen inputs."""
        stored = dict(self.artifact)
        self.assertIn("provenance", stored)
        stored["provenance"] = {}
        self.assertEqual(
            json.dumps(stored, indent=2, sort_keys=True),
            json.dumps(self.result, indent=2, sort_keys=True),
        )

    def test_no_wall_clock_or_timing_fields_leak_into_the_artifact(self) -> None:
        text = json.dumps(self.artifact)
        for banned in ("runtime_seconds", "wall_seconds", "elapsed", "timestamp"):
            self.assertNotIn(banned, text)


if __name__ == "__main__":
    unittest.main()
