from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

try:
    from . import spd_external_validation as spd
except ImportError:  # pragma: no cover
    import spd_external_validation as spd  # type: ignore


def test_largest_coverage_group_selection_is_outcome_blind() -> None:
    rows = []
    for gene, candidates in spd.CANDIDATE_GROUPS.items():
        for offset, group in enumerate(candidates):
            coverage = 3 + offset
            for index in range(coverage):
                rows.append(
                    {
                        "assay_group_id": group,
                        "full_inchikey": f"{gene}_{group}_{index}",
                        "assay_group_name": f"{gene} direct",
                    }
                )
    frame = pd.DataFrame(rows)
    selected, audit = spd.choose_primary_groups(frame)
    for gene, candidates in spd.CANDIDATE_GROUPS.items():
        expected = candidates[-1] if len(candidates) > 1 else candidates[0]
        assert selected[expected] == gene
    assert audit.selected_primary.sum() == len(spd.TARGETS)


def test_censor_aware_binary_does_not_treat_unknown_bound_as_inactive() -> None:
    target = spd.TARGETS[0]
    frame = pd.DataFrame(
        {
            "full_inchikey": [
                "AAAAAAAAAAAAAA-BBBBBBBBBB-C",
                "CCCCCCCCCCCCCC-DDDDDDDDDD-E",
                "EEEEEEEEEEEEEE-FFFFFFFFFF-G",
            ],
            "connectivity_key": [
                "AAAAAAAAAAAAAA",
                "CCCCCCCCCCCCCC",
                "EEEEEEEEEEEEEE",
            ],
            "gene": [target] * 3,
            "qualifier": ["=", ">", ">"],
            "ic50_uM": [5.0, 30.0, 1.0],
            "pIC50_bound": [6 - np.log10(5), 6 - np.log10(30), 6.0],
        }
    )
    matrix, support = spd.matrix_from_long(
        frame,
        identity="connectivity_key",
        endpoint="censor_aware_binary",
        threshold_uM=10.0,
    )
    assert support["rows_after_endpoint_filter"] == 2
    assert matrix.loc["AAAAAAAAAAAAAA", target] == 1.0
    assert matrix.loc["CCCCCCCCCCCCCC", target] == 0.0
    assert "EEEEEEEEEEEEEE" not in matrix.index


def test_missing_aware_centering_preserves_missingness_and_zeroes_rows() -> None:
    frame = pd.DataFrame(
        [[1.0, 2.0, np.nan], [2.0, 4.0, 5.0], [4.0, np.nan, 7.0]],
        columns=list(spd.TARGETS[:3]),
    )
    residual = spd.missing_aware_residual(frame)
    assert np.array_equal(frame.isna(), residual.isna())
    assert np.allclose(residual.mean(axis=1).to_numpy(), 0.0, atol=1e-12)


def test_docking_geometry_uses_explicit_target_labels_not_file_order() -> None:
    rng = np.random.default_rng(17)
    values = rng.normal(size=(100, len(spd.TARGETS)))
    ordered = pd.DataFrame(values, columns=spd.TARGETS)
    ordered.insert(0, "smiles", ["C"] * len(ordered))
    ordered.insert(0, "inchikey", [f"KEY{i:011d}-AAAAAAAAAA-A" for i in range(100)])
    ordered["connectivity_key"] = ordered.inchikey.str[:14]
    shuffled_columns = [
        "inchikey",
        "smiles",
        *reversed(spd.TARGETS),
        "connectivity_key",
    ]
    shuffled = ordered.loc[:, shuffled_columns]
    first, _ = spd.docking_geometries(ordered, target_scope="local_12")
    second, _ = spd.docking_geometries(shuffled, target_scope="local_12")
    assert np.allclose(first["raw"], second["raw"])
    assert np.allclose(first["residual"], second["residual"])


def _synthetic_pair_frame() -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    rng = np.random.default_rng(9)
    raw = np.corrcoef(rng.normal(size=(len(spd.TARGETS), 100)))
    residual = np.corrcoef(rng.normal(size=(len(spd.TARGETS), 100)))
    rows = []
    for first, target_a in enumerate(spd.TARGETS):
        for second in range(first + 1, len(spd.TARGETS)):
            target_b = spd.TARGETS[second]
            rows.append(
                {
                    "target_a": target_a,
                    "target_b": target_b,
                    "experimental_correlation": rng.normal(),
                    "pair_support": 50 + first + second,
                    "docking_raw": raw[first, second],
                    "docking_residual": residual[first, second],
                    "full_sequence_identity": rng.uniform(),
                    "same_family": float(
                        spd.TARGET_FAMILY[target_a] == spd.TARGET_FAMILY[target_b]
                    ),
                    "pair_target_coverage_geomean": 100 + first + second,
                    "pair_target_coverage_similarity": -abs(first - second),
                }
            )
    return pd.DataFrame(rows), {"raw": raw, "residual": residual}


def test_paired_qap_is_deterministic() -> None:
    frame, geometries = _synthetic_pair_frame()
    first = spd.paired_qap(
        frame,
        geometries,
        permutations=99,
        seed=7,
        preserve_family=False,
    )
    second = spd.paired_qap(
        frame,
        geometries,
        permutations=99,
        seed=7,
        preserve_family=False,
    )
    assert first == second
    assert {row["metric"] for row in first} == {
        "raw_spearman",
        "residual_spearman",
        "residual_minus_raw",
        "raw_partial_rank",
        "residual_partial_rank",
        "residual_minus_raw_partial_rank",
    }


def test_released_summary_and_no_compound_identifiers() -> None:
    summary_path = spd.DEFAULT_OUTPUT / "summary.json"
    if not summary_path.is_file():
        pytest.skip("release outputs have not been built")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["support"]["spd_connectivity_blocks"] == 1925
    assert summary["support"]["connectivity_block_overlap"] == 932
    assert summary["key_results"]["primary_floor_at_bound"]["pairs"] == 59
    assert summary["key_results"]["primary_floor_at_bound"][
        "residual_spearman"
    ] == pytest.approx(0.33302162478083)
    prohibited = {"inchi_key", "inchikey", "connectivity_key", "smiles", "RowId"}
    for path in spd.DEFAULT_OUTPUT.glob("*.csv"):
        assert not (prohibited & set(pd.read_csv(path, nrows=0).columns)), path.name
    assert str(spd.PACKAGE) not in summary_path.read_text(encoding="utf-8")
