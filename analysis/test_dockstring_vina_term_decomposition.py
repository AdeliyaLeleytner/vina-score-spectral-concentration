#!/usr/bin/env python3
"""Tests for the fixed-pose DOCKSTRING Vina-term decomposition."""

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

from dockstring_vina_term_decomposition import (  # noqa: E402
    component_attribution,
    heavy_atom_slope_decomposition,
    parse_vina_score_only,
    quality_support_sensitivity,
    si_ready_term_summary,
    two_way_center,
)


RESULTS = ROOT / "results" / "dockstring_vina_terms"


class VinaTermUnitTests(unittest.TestCase):
    def test_score_only_parser_reads_affinity_and_all_terms(self) -> None:
        text = """
Affinity: -8.12500 (kcal/mol)
Intermolecular contributions to the terms, before weighting:
    gauss 1     : 12.5
    gauss 2     : 34.0
    repulsion   : 0.75
    hydrophobic : 4.25
    Hydrogen    : 1.50
"""
        parsed = parse_vina_score_only(text)
        self.assertEqual(parsed["affinity"], -8.125)
        self.assertEqual(parsed["gauss1"], 12.5)
        self.assertEqual(parsed["hydrogen"], 1.5)

    def test_two_way_center_removes_both_main_effects(self) -> None:
        rng = np.random.default_rng(7)
        matrix = rng.normal(size=(50, 9))
        residual = two_way_center(matrix)
        self.assertLess(np.abs(residual.mean(0)).max(), 1e-14)
        self.assertLess(np.abs(residual.mean(1)).max(), 1e-14)

    def test_linear_term_attributions_close_exactly(self) -> None:
        rng = np.random.default_rng(11)
        components = {
            name: rng.normal(size=(100, 9))
            for name in ("a", "b", "c", "d", "e")
        }
        total = sum(components.values())
        attribution, modes = component_attribution(total, components)
        for column in (
            "shared_row_axis_covariance_share",
            "raw_pc1_bilinear_share",
            "residual_frobenius_inner_product_share",
            "residual_pc1_bilinear_share",
        ):
            self.assertAlmostEqual(float(attribution[column].sum()), 1.0, places=12)
        mode_sums = modes.groupby("residual_mode").bilinear_share.sum()
        self.assertTrue(np.allclose(mode_sums, 1.0, atol=1e-12))

    def test_heavy_atom_projection_recovers_target_specific_slopes(self) -> None:
        heavy_atoms = np.arange(15, 75, dtype=float)
        centered = heavy_atoms - heavy_atoms.mean()
        slopes = np.array([-0.30, -0.21, -0.17, -0.10, -0.02, 0.03, 0.08, 0.15, 0.22])
        matrix = np.outer(centered, slopes) + np.arange(9)[None, :]
        descriptors = pd.DataFrame({"heavy_atoms": heavy_atoms})
        summary, targets = heavy_atom_slope_decomposition(
            {"synthetic": matrix}, descriptors
        )
        row = summary.iloc[0]
        self.assertAlmostEqual(row.raw_column_centered_r2, 1.0, places=12)
        self.assertAlmostEqual(row.two_way_residual_r2, 1.0, places=12)
        self.assertAlmostEqual(row.residual_pc1_score_r2, 1.0, places=12)
        self.assertTrue(
            np.allclose(
                targets.slope_kcal_per_mol_per_heavy_atom.to_numpy(),
                slopes,
                atol=1e-12,
            )
        )

    def test_quality_support_rules_exclude_only_defined_anomalies(self) -> None:
        rng = np.random.default_rng(13)
        total = rng.normal(size=(20, 9))
        released = total.copy()
        released[3, 2] += 0.2
        pre_total = total * 1.5 + 0.1 * rng.normal(size=total.shape)
        effective_tors = np.repeat(np.arange(20)[:, None], 9, axis=1).astype(float)
        effective_tors[5, 4] += 1.0
        components = {
            "a": total * 0.6,
            "b": total * 0.4,
        }
        _, _, rules = quality_support_sensitivity(
            total, released, pre_total, effective_tors, components
        )
        supports = rules["supports"]
        self.assertEqual(
            supports["target_invariant_effective_torsions"]["n_ligands"], 19
        )
        self.assertEqual(
            supports["released_scores_reproduced_within_0.10_kcal_mol"][
                "n_ligands"
            ],
            19,
        )
        self.assertEqual(supports["both_quality_rules"]["n_ligands"], 18)

    def test_si_term_table_preserves_fixed_pose_boundary(self) -> None:
        attribution = pd.DataFrame(
            {
                "term": list(("gauss1", "gauss2", "repulsion", "hydrophobic", "hydrogen")),
                "shared_row_axis_covariance_share": [0.1, 0.6, -0.1, 0.3, 0.1],
                "raw_pc1_bilinear_share": [0.1, 0.6, -0.1, 0.3, 0.1],
                "residual_frobenius_inner_product_share": [0.2] * 5,
                "residual_pc1_bilinear_share": [0.2] * 5,
            }
        )
        ablations = pd.DataFrame(
            {
                "removed_term": attribution.term,
                "raw_pr": np.arange(5, dtype=float) + 2,
                "residual_pr": np.arange(5, dtype=float) + 4,
            }
        )
        spectral = pd.DataFrame(
            {"component": ["rescored_post_total"], "raw_pr": [2.5], "residual_pr": [5.0]}
        )
        table = si_ready_term_summary(attribution, ablations, spectral)
        self.assertEqual(table.term.tolist(), attribution.term.tolist())
        self.assertTrue(table.analysis_boundary.str.contains("not an independent").all())
        self.assertAlmostEqual(
            table.loc[table.term.eq("gauss1"), "fixed_pose_term_removal_raw_pr_minus_full"].iloc[0],
            -0.5,
        )


@unittest.skipUnless(
    (RESULTS / "analysis_summary.json").exists(),
    "run dockstring_vina_term_decomposition.py to create frozen outputs",
)
class VinaTermFrozenOutputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.summary = json.loads((RESULTS / "analysis_summary.json").read_text())
        cls.spectral = pd.read_csv(RESULTS / "term_spectral_metrics.csv").set_index(
            "component"
        )

    def test_score_roundtrip_and_term_reconstruction(self) -> None:
        reproduction = self.summary["score_reproduction"]
        self.assertGreater(reproduction["rescored_vs_released_cell_pearson"], 0.999)
        self.assertLess(reproduction["rescored_minus_released_mae"], 0.03)
        self.assertLess(reproduction["term_sum_max_abs_reconstruction_error"], 1e-12)

    def test_rank_expansion_precedes_torsional_normalisation(self) -> None:
        pre = self.spectral.loc["zero_rot_pre_total"]
        post = self.spectral.loc["rescored_post_total"]
        self.assertGreater(pre.residual_pr - pre.raw_pr, 2.9)
        self.assertGreater(post.residual_pr - post.raw_pr, 2.1)
        self.assertLess(pre.raw_pr, post.raw_pr)

    def test_gauss2_dominates_both_axes_but_repulsion_opposes_shared_axis(self) -> None:
        attribution = self.summary["term_attribution"]
        self.assertGreater(
            attribution["gauss2"]["shared_row_axis_covariance_share"], 0.70
        )
        self.assertGreater(
            attribution["gauss2"]["residual_frobenius_inner_product_share"],
            0.50,
        )
        self.assertLess(
            attribution["repulsion"]["shared_row_axis_covariance_share"], -0.20
        )
        self.assertIn("fixed", self.summary["claim_boundary"])
        self.assertFalse(
            self.summary["reproducibility_boundary"][
                "independent_redocking_performed"
            ]
        )
        self.assertTrue((RESULTS / "si_vina_term_summary.csv").is_file())

    def test_repulsive_term_is_a_negative_control_for_rank_expansion(self) -> None:
        self.assertLess(self.spectral.loc["post_repulsion", "pr_increase"], 0)
        for term in ("post_gauss1", "post_gauss2", "post_hydrophobic", "post_hydrogen"):
            self.assertGreater(self.spectral.loc[term, "pr_increase"], 1.0)

    def test_size_slope_mechanism_and_quality_support_are_stable(self) -> None:
        size = self.summary["heavy_atom_slope_decomposition"]
        self.assertEqual(size["zero_rot_pre_total"]["negative_target_slopes"], 9)
        self.assertGreater(
            size["zero_rot_pre_total"]["residual_pc1_score_r2"], 0.65
        )
        support = self.summary["quality_support_sensitivity"]["supports"]
        self.assertGreaterEqual(
            support["released_scores_reproduced_within_0.10_kcal_mol"]["n_ligands"],
            490,
        )

    def test_cross_panel_fingerprint_localises_to_hydrophobic_term(self) -> None:
        supports = self.summary["cross_panel_term_fingerprint"]["supports"]
        all_targets = supports["all_9_mapped_targets"]["components"]
        self.assertGreater(
            all_targets["rescored_post_total"]["mean_descriptor_spearman_rho"],
            0.80,
        )
        self.assertLess(
            all_targets["rescored_post_total"]["exact_two_sided_p"], 0.005
        )
        self.assertGreater(
            all_targets["post_hydrophobic"]["mean_descriptor_spearman_rho"],
            0.70,
        )
        self.assertLess(
            all_targets["post_hydrophobic"][
                "max_abs_fwer_p_across_five_terms"
            ],
            0.03,
        )
        self.assertLess(
            all_targets["post_hydrogen"]["mean_descriptor_spearman_rho"], -0.65
        )
        different_structures = supports[
            "different_receptor_structure_only_6"
        ]["components"]
        self.assertGreater(
            different_structures["rescored_post_total"][
                "mean_descriptor_spearman_rho"
            ],
            0.80,
        )
        self.assertLess(
            different_structures["rescored_post_total"]["exact_two_sided_p"],
            0.02,
        )
        split_supports = self.summary["cross_panel_term_fingerprint"][
            "disjoint_ligand_support_sensitivity"
        ]
        for label in ("seeded_random_first_256", "seeded_random_second_256"):
            hydrophobic = split_supports[label]["supports"][
                "all_9_mapped_targets"
            ]["components"]["post_hydrophobic"]
            self.assertGreater(
                hydrophobic["mean_descriptor_spearman_rho"], 0.69
            )
            self.assertLess(
                hydrophobic["max_abs_fwer_p_across_five_terms"], 0.05
            )


if __name__ == "__main__":
    unittest.main()
