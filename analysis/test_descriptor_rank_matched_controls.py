"""Contract tests for rank/dimension-matched descriptor controls."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator


PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE / "analysis"))
import descriptor_rank_matched_controls as controls  # noqa: E402

RESULTS = PACKAGE / "results" / "descriptor_rank_matched_controls"
PANELS = ("DAVIS", "PKIS2", "PKIS1", "KiRHub")
PHYSICAL = "physicochemical_7"
MORGAN = (
    "morgan_normalized_bit_rp_7_seed_a",
    "morgan_normalized_bit_rp_7_seed_b",
)
COUNT_MORGAN = "morgan_count_plus_size_rp_7_locked_seed_20260910"
HASH = "stable_hash_nuisance_7"
ORACLE = "outcome_rank7_svd_transported_oracle"


def load_summary() -> dict:
    return json.loads((RESULTS / "summary.json").read_text())


def test_count_morgan_feature_construction_is_deterministic_and_size_explicit() -> None:
    smiles = ["CCCC", "CCCO", "CC(C)C", "c1ccccc1O"]
    seeds = tuple(range(20260910, 20260930))
    first = controls.morgan_random_projection_features(
        smiles, (20260821, 20260822), seeds
    )
    second = controls.morgan_random_projection_features(
        smiles, (20260821, 20260822), seeds
    )
    normalized_a, normalized_b, ensemble, report = first
    assert normalized_a.shape == (len(smiles), 7)
    assert normalized_b.shape == (len(smiles), 7)
    assert ensemble.shape == (len(smiles), 20, 7)
    assert np.array_equal(normalized_a, second[0])
    assert np.array_equal(normalized_b, second[1])
    assert np.array_equal(ensemble, second[2])

    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    expected_occurrences = np.asarray(
        [
            sum(
                generator.GetCountFingerprint(Chem.MolFromSmiles(smiles_value))
                .GetNonzeroElements()
                .values()
            )
            for smiles_value in smiles
        ],
        dtype=np.float32,
    )
    assert np.array_equal(
        ensemble[:, :, 0], np.repeat(expected_occurrences[:, None], 20, axis=1)
    )
    assert not np.array_equal(ensemble[:, 0, 1:], ensemble[:, 1, 1:])
    assert report["count_preserving_primary_ensemble"]["members"] == 20


def test_row_permutation_nuisance_is_deterministic_bijective_and_checksummed() -> None:
    source = np.arange(84, dtype=np.float64).reshape(12, 7)
    source += np.arange(7)[None, :] ** 2
    standardized = controls.standardized_descriptor_matrix(source)
    first, permutation, checksums = controls.row_permuted_nuisance_features(
        standardized, 20261010
    )
    second = controls.row_permuted_nuisance_features(standardized, 20261010)
    different = controls.row_permuted_nuisance_features(standardized, 20261011)
    assert np.array_equal(first, second[0])
    assert np.array_equal(permutation, second[1])
    assert np.array_equal(np.sort(permutation), np.arange(len(source)))
    assert not np.array_equal(permutation, different[1])
    for column in range(7):
        assert np.array_equal(np.sort(first[:, column]), np.sort(standardized[:, column]))
    assert controls.array_sha256(standardized, "<f8") == (
        "123eee32f900580c3f51634774c1e652cfe897b46827e841a76882a63313de29"
    )
    assert checksums == {
        "permutation_index_sha256": (
            "c98951d00e838da717dea8960c1f82c56ba31e34881c21e419829e6bb1317692"
        ),
        "permuted_feature_matrix_sha256": (
            "7644b92985b1993be50e9fbf6cece4cce5160da2d23a28dd35b786cdee58bb13"
        ),
    }


def test_predictive_bases_are_dimension_matched_and_separately_named() -> None:
    summary = load_summary()
    assert summary["configuration"]["features_per_predictive_basis"] == 7
    ranks = summary["coefficient_effective_ranks_by_fold"]
    assert set(ranks) == {PHYSICAL, COUNT_MORGAN, *MORGAN, HASH}
    for basis, values in ranks.items():
        assert len(values) == 5, basis
        assert max(values) <= 7, basis
        assert min(values) >= 1, basis
    diagnostics = pd.read_csv(RESULTS / "feature_diagnostics.csv")
    assert set(diagnostics.basis) == set(ranks)
    assert (diagnostics.dimensions == 7).all()
    assert (diagnostics.matrix_rank == 7).all()
    construction = summary["feature_construction"]["morgan_random_projections"]
    normalized = construction["normalized_bit_secondary_controls"]
    assert normalized["seeds"][0] != normalized["seeds"][1]
    count = construction["count_preserving_primary_ensemble"]
    assert count["members"] >= 20
    assert "explicit total-occurrence-count coordinate" in count["definition"]
    assert "without ligand-wise normalization" in count["definition"]


def test_transported_oracle_is_not_misreported_as_a_predictive_control() -> None:
    summary = load_summary()
    oracle = summary["outcome_rank7_svd_transported_oracle"]
    assert "not an OOF ligand-feature predictor" in oracle["role"]
    assert "not a universal rank-7 capacity ceiling" in oracle["role"]
    assert summary["verification"][
        "transported_oracle_uses_test_outcomes_for_projection"
    ] is True
    assert len(oracle["folds"]) == 5
    for fold in oracle["folds"]:
        assert 0.0 < fold["rank7_training_energy_share"] <= 1.0
        assert 0.0 < fold["rank7_test_reconstruction_r2"] <= 1.0
    assert ORACLE in summary["panel_geometry_agreement"]


def test_estimand_is_never_called_an_explained_fraction() -> None:
    summary = load_summary()
    text = json.dumps(summary).lower()
    assert "not_an_explained_fraction" in text
    assert "not an explained fraction" in summary["estimands"][
        "predictive_score_structure"
    ]
    for basis, metrics in summary["predictive_metrics"].items():
        before = metrics[
            "mean_squared_offdiagonal_target_correlation_before_subtraction"
        ]
        after = metrics[
            "mean_squared_offdiagonal_target_correlation_after_subtraction"
        ]
        reduction = metrics[
            "relative_reduction_in_mean_squared_offdiagonal_target_correlation"
        ]
        assert np.isclose(reduction, (before - after) / before), basis


def test_every_arm_has_four_panel_geometry_and_paired_intervals() -> None:
    summary = load_summary()
    agreement = summary["panel_geometry_agreement"]
    assert set(agreement) == {
        "observed_residual",
        PHYSICAL,
        COUNT_MORGAN,
        *MORGAN,
        HASH,
        ORACLE,
    }
    for arm, values in agreement.items():
        assert set(values) == set(PANELS), arm
        assert all(np.isfinite(list(values.values()))), arm

    bootstrap = summary["paired_chemical_group_block_bootstrap"]
    assert "coarsened" in bootstrap["coarsening_limitation"]
    assert set(bootstrap["per_grouping"]) == {
        "murcko_scaffold",
        "murcko_generic_framework",
        "generic_murcko_or_acyclic_topology",
    }
    contrasts = bootstrap["physicochemical_minus_control"]
    assert set(contrasts) == {COUNT_MORGAN, *MORGAN, HASH, ORACLE}
    for control, panels in contrasts.items():
        for panel, record in panels.items():
            low, high = record[
                "conservative_union_sensitivity_interval_95"
            ]
            assert low < high, (control, panel)
            assert np.isfinite(record["point"]), (control, panel)


def test_hash_basis_behaves_as_a_negative_control() -> None:
    summary = load_summary()
    physical = summary["predictive_metrics"][PHYSICAL]
    nuisance = summary["predictive_metrics"][HASH]
    assert physical["mean_out_of_fold_target_r2"] > 0.05
    assert nuisance["mean_out_of_fold_target_r2"] < 0.01
    assert (
        nuisance["relative_reduction_in_mean_squared_offdiagonal_target_correlation"]
        < physical[
            "relative_reduction_in_mean_squared_offdiagonal_target_correlation"
        ]
    )


def test_nuisance_projection_ensemble_and_interpretation_are_synchronized() -> None:
    summary = load_summary()
    ensemble = summary["physicochemical_row_permutation_nuisance_ensemble"]
    assert ensemble["members"] == 20
    assert len(ensemble["seeds"]) == 20
    assert len(set(ensemble["seeds"])) == 20
    assert "not confidence intervals" in ensemble["interpretation"]
    assert ensemble["same_oof_observed_surface_maximum_absolute_difference"] == 0.0
    assert len(ensemble["source_standardized_feature_matrix_sha256"]) == 64
    assert len(ensemble["members_detail"]) == 20
    for member in ensemble["members_detail"]:
        assert member["permutation_is_bijection"] is True
        assert len(member["permutation_index_sha256"]) == 64
        assert len(member["permuted_feature_matrix_sha256"]) == 64
        assert len(member["coefficient_effective_ranks_by_fold"]) == 5
        assert set(member["panel_geometry_agreement"]) == set(PANELS)
        assert np.isfinite(member["mean_out_of_fold_target_r2"])
        assert np.isfinite(
            member[
                "relative_reduction_in_mean_squared_offdiagonal_target_correlation"
            ]
        )
    distributions = ensemble["algorithmic_seed_distributions"]
    assert "interval_95" not in json.dumps(distributions)
    assert "not_an_uncertainty_interval" in json.dumps(distributions)

    frame = pd.read_csv(
        RESULTS / "physicochemical_row_permutation_nuisance_ensemble.csv"
    )
    assert len(frame) == 20
    assert frame.seed.tolist() == ensemble["seeds"]
    assert frame.permutation_index_sha256.tolist() == [
        member["permutation_index_sha256"] for member in ensemble["members_detail"]
    ]

    negative = summary["negative_control_interpretation"]
    hash_metrics = summary["predictive_metrics"][HASH]
    assert negative["stable_hash_mean_out_of_fold_target_r2"] == hash_metrics[
        "mean_out_of_fold_target_r2"
    ]
    assert negative[
        "stable_hash_relative_reduction_in_mean_squared_offdiagonal_target_correlation"
    ] == hash_metrics[
        "relative_reduction_in_mean_squared_offdiagonal_target_correlation"
    ]
    assert negative["stable_hash_panel_geometry_agreement"] == summary[
        "panel_geometry_agreement"
    ][HASH]
    assert "falsifies" in negative["conclusion"]
    refreshed = controls.refresh_interpretive_fields(
        json.loads(json.dumps(summary))
    )
    assert refreshed["negative_control_interpretation"] == negative


def test_count_morgan_ensemble_is_multiseed_count_preserving_and_descriptive() -> None:
    summary = load_summary()
    ensemble = summary["morgan_count_plus_size_rp_7_seed_ensemble"]
    assert ensemble["members"] >= 20
    assert len(ensemble["seeds"]) == ensemble["members"]
    assert ensemble["seeds"][0] == ensemble["locked_representative_seed"]
    assert "not selected by performance" in ensemble["representative_selection"]
    assert "not a confidence interval" in ensemble["interpretation"]
    assert len(ensemble["members_detail"]) == ensemble["members"]
    for member in ensemble["members_detail"]:
        assert len(member["coefficient_effective_ranks_by_fold"]) == 5
        assert set(member["panel_geometry_agreement"]) == set(PANELS)
    distributions = ensemble["algorithmic_distributions"]
    operational = distributions[
        "relative_reduction_in_mean_squared_offdiagonal_target_correlation"
    ]
    seed_distribution = operational["count_morgan_seed_distribution"]
    assert "interval_95" not in seed_distribution
    assert "not_an_uncertainty_interval" in seed_distribution
    assert operational["count_morgan_seed_minimum"] < (
        operational["count_morgan_seed_maximum"]
    )
    for panel in PANELS:
        record = distributions["panel_geometry_agreement"][panel]
        assert record["count_morgan_seed_minimum"] < record[
            "count_morgan_seed_maximum"
        ]
    reproduction = ensemble["locked_representative_reproduction"]
    assert reproduction["maximum_absolute_predictive_metric_difference"] < 1e-9
    assert reproduction["maximum_absolute_panel_geometry_difference"] < 1e-9

    metric_frame = pd.read_csv(RESULTS / "morgan_count_rp_ensemble_metrics.csv")
    geometry_frame = pd.read_csv(
        RESULTS / "morgan_count_rp_ensemble_panel_geometry.csv"
    )
    association_frame = pd.read_csv(
        RESULTS / "morgan_count_rp_ensemble_partial_associations.csv"
    )
    assert len(metric_frame) == ensemble["members"]
    assert len(geometry_frame) == ensemble["members"] * len(PANELS)
    assert len(association_frame) == ensemble["members"] * len(PANELS)


def test_operational_reduction_has_paired_conditional_intervals() -> None:
    summary = load_summary()
    bootstrap = summary["operational_reduction_paired_block_bootstrap"]
    assert "same draw weights" in bootstrap["method"]
    assert "not refitted" in bootstrap["conditioning"]
    assert "coarsened" in bootstrap["coarsening_limitation"]
    contrasts = bootstrap[
        "physicochemical_minus_control_conservative_union"
    ]
    assert set(contrasts) == {COUNT_MORGAN, *MORGAN, HASH, ORACLE}
    for record in contrasts.values():
        low, high = record["conservative_union_sensitivity_interval_95"]
        assert low < high
        assert np.isfinite(record["point"])
    frame = pd.read_csv(
        RESULTS / "paired_operational_reduction_differences.csv"
    )
    assert len(frame) == 3 * 5
    assert (
        frame.conditional_coarsened_sensitivity_low95
        < frame.conditional_coarsened_sensitivity_high95
    ).all()


def test_component_remainder_dependence_and_partial_associations_are_retained() -> None:
    summary = load_summary()
    dependence = summary["component_remainder_and_partial_associations"]
    assert set(dependence) == {PHYSICAL, COUNT_MORGAN, *MORGAN, HASH}
    for basis, record in dependence.items():
        assert np.isfinite(record["component_remainder_spearman"]), basis
        assert np.isfinite(record["component_remainder_pearson"]), basis
        assert set(record["panels"]) == set(PANELS)
        for panel_record in record["panels"].values():
            assert np.isfinite(list(panel_record.values())).all()


def test_acyclic_topology_holdout_is_a_separate_predictive_refit() -> None:
    summary = load_summary()
    report = summary["generic_murcko_acyclic_topology_grouping"]
    assert report["acyclic_ligands"] > 0
    assert report["acyclic_groups"] < report["acyclic_ligands"]
    sensitivity = summary[
        "generic_murcko_acyclic_topology_predictive_sensitivity"
    ]
    assert "single deterministic GroupKFold" in sensitivity["uncertainty_boundary"]
    assert set(sensitivity["predictive_metrics"]) == {
        PHYSICAL,
        COUNT_MORGAN,
        *MORGAN,
        HASH,
    }
    assert set(sensitivity["panel_geometry_agreement"]) == {
        "observed_residual",
        PHYSICAL,
        COUNT_MORGAN,
        *MORGAN,
        HASH,
    }
    assert len(sensitivity["folds"]) == 5
    assert (
        summary["verification"][
            "physical_prediction_reproduction_max_abs_difference"
        ]
        < 1e-9
    )


def test_reporting_contract_and_no_raw_ligand_rows() -> None:
    summary = load_summary()
    assert summary["status"] == "exploratory_post_hoc_reviewer_requested_controls"
    assert summary["support"]["ligands"] > 259_000
    assert summary["support"]["targets"] == 20
    assert summary["support"]["cells"] == summary["support"]["ligands"] * 20
    assert "do NOT" in summary["claim_boundary"]
    assert "not a share" in summary["claim_boundary"]
    assert not (RESULTS / "ligand_features.csv").exists()
    differences = pd.read_csv(RESULTS / "paired_geometry_differences.csv")
    assert len(differences) == 3 * 5 * len(PANELS)
    assert differences.physicochemical_minus_control_point.notna().all()
