from __future__ import annotations

import unittest

from cross_panel_overlap_audit import analyze, PACKAGE


class CrossPanelOverlapAuditTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.result = analyze(
            PACKAGE / "data/frozen/df_final_v4.csv.gz",
            PACKAGE / "data/frozen/dockstring_identity_contract_2026-08-03.csv.gz",
            PACKAGE / "results/residual_mechanism/cross_panel_target_mapping.csv",
            PACKAGE / "data/target_families.csv",
        )

    def test_support(self) -> None:
        self.assertEqual(self.result["support"]["docking44_ligands"], 12651)
        self.assertEqual(self.result["support"]["dockstring_complete_ligands"], 260060)

    def test_target_mapping_is_frozen(self) -> None:
        target = self.result["target_overlap"]
        self.assertEqual(target["mapped_target_identities"], 9)
        self.assertEqual(target["same_receptor_structure_used"], 3)

    def test_overlap_counts_are_bounded(self) -> None:
        chemical = self.result["chemical_overlap"]
        self.assertGreaterEqual(chemical["shared_connectivity_blocks"], 0)
        self.assertLessEqual(
            chemical["docking44_rows_in_shared_connectivity_blocks"], 12651
        )
        self.assertLessEqual(
            chemical["dockstring_rows_in_shared_connectivity_blocks"], 260060
        )


if __name__ == "__main__":
    unittest.main()
