from __future__ import annotations

import pandas as pd

import reproduce_pdsp_derived_qap as reproduction


def test_released_aggregate_qap_reproduces_the_frozen_report_table() -> None:
    result = reproduction.reproduce()
    reproduction.verify(result)
    assert len(result) == 12
    assert set(result["permutation_scheme"]) == {
        "all_target_labels",
        "within_curated_family",
    }
    assert set(result["metric"]) == {
        "continuous_spearman",
        "roc_auc_top10",
        "best_partner_recall_at_3",
    }


def test_reproduction_uses_only_aggregate_pdsp_fields() -> None:
    pairs = pd.read_csv(reproduction.DEFAULT_PAIRS, nrows=1)
    prohibited = {"full_inchikey", "SMILES", "ki Val", "Reference"}
    assert not (prohibited & set(pairs.columns))
    exclusions = pd.read_csv(reproduction.DEFAULT_EXCLUSIONS)
    assert list(exclusions.columns) == ["ligand_id"]
    assert exclusions["ligand_id"].is_unique
    assert len(exclusions) > 0


def test_complete_case_qap_and_support_sweep_reproduce_from_aggregates() -> None:
    metrics, qap = reproduction.reproduce_complete_case()
    reproduction.verify_table(
        metrics, reproduction.DEFAULT_COMPLETE_METRICS, ("predictor",)
    )
    reproduction.verify_table(qap, reproduction.DEFAULT_COMPLETE_QAP, reproduction.KEYS)
    sweep = reproduction.reproduce_support_sweep()
    reproduction.verify_table(
        sweep,
        reproduction.DEFAULT_SUPPORT_SWEEP,
        ("minimum_pair_support",),
    )
    assert len(qap) == 12
    assert len(sweep) == 6
