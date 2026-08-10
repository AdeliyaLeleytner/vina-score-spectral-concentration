from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


ANALYSIS = Path(__file__).resolve().parent
if str(ANALYSIS) not in sys.path:
    sys.path.insert(0, str(ANALYSIS))

import dense_davis_benchmark as davis
import experimental_panel_overlap as subject


def identity_frame(smiles: list[str]) -> pd.DataFrame:
    frame = pd.DataFrame({"smiles": smiles})
    return davis._identity_table(frame, "smiles", scan="full")


def test_structure_summary_distinguishes_acyclic_singletons() -> None:
    frame = identity_frame(["c1ccccc1", "Cc1ccccc1", "CCO", "CC"])
    counts, sets = subject.structure_summary(frame, "smiles")
    assert counts["ligand_rows"] == 4
    assert counts["unique_standard_inchikeys"] == 4
    assert counts["unique_connectivity_blocks"] == 4
    assert counts["unique_nonempty_murcko_scaffolds"] == 1
    assert counts["acyclic_ligand_rows"] == 2
    assert counts["murcko_clusters_with_acyclic_singletons"] == 3
    assert len(sets["nonempty_murcko_scaffold"]) == 1


def test_pairwise_overlap_levels_are_kept_distinct() -> None:
    first = identity_frame(["c1ccccc1", "CCO"])
    second = identity_frame(["Cc1ccccc1", "CCO", "CCC"])
    _, first_sets = subject.structure_summary(first, "smiles")
    _, second_sets = subject.structure_summary(second, "smiles")
    rows = subject.pairwise_overlap_rows({"first": first_sets, "second": second_sets})
    assert rows == [
        {
            "first_panel": "first",
            "second_panel": "second",
            "shared_standard_inchikeys": 1,
            "shared_connectivity_blocks": 1,
            "shared_nonempty_murcko_scaffolds": 1,
        }
    ]


def test_kirhub_boundary_is_read_from_frozen_summary() -> None:
    result = subject.load_kirhub_boundary(subject.DEFAULT_KIRHUB_SUMMARY)
    assert result["exact_normalized_name_overlap_with_DAVIS"] == 11
    assert result[
        "geometry_spearman_to_DAVIS_PKIS2_PKIS1_mean_after_removing_11_DAVIS_names"
    ] == 0.8165798649269518
    assert result["structures_available"] is False
