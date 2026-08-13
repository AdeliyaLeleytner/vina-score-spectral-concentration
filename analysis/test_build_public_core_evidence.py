from __future__ import annotations

import ast
import json
from pathlib import Path

import pandas as pd
import pytest

import build_public_core_evidence as public_core


FORBIDDEN_DEPENDENCY_MARKERS = (
    "pkis1",
    "kirhub",
    "klifs",
    "spd_",
    "chembl",
    "fixed_pose",
    "fixed-pose",
    "negative_results_paper",
    "manuscript_evidence",
    "evidence_summary",
    "released_pair_geometry",
)


def test_input_gateway_is_exact_and_fails_closed() -> None:
    registry = public_core.InputRegistry()
    with pytest.raises(PermissionError):
        registry.path("historical_aggregate")
    declared_paths = [spec.path.lower() for spec in public_core.INPUT_SPECS.values()]
    assert len(declared_paths) == len(set(declared_paths))
    joined = "\n".join(declared_paths)
    assert all(marker not in joined for marker in FORBIDDEN_DEPENDENCY_MARKERS)


def test_builder_imports_only_stdlib_numpy_and_pandas() -> None:
    source_path = Path(public_core.__file__)
    tree = ast.parse(source_path.read_text())
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert imported <= {
        "__future__",
        "argparse",
        "dataclasses",
        "hashlib",
        "json",
        "math",
        "numpy",
        "pandas",
        "pathlib",
        "platform",
        "typing",
        "warnings",
    }


def test_target_panel_selector_controls_import_is_exact_and_locked() -> None:
    registry = public_core.InputRegistry()
    evidence = public_core.target_panel_selector_controls_evidence(registry)

    assert evidence["analysis_status"] == (
        "strict_public_k8_selector_robustness_control"
    )
    assert evidence["selector_control_configuration"][
        "random_panels_per_split_and_k"
    ] == 50
    assert evidence["configuration"]["random_panel_draws"] == [0, 1, 2]
    assert evidence["selector_bundle_table_rows"]["panel_metrics.csv.gz"] == 5420
    assert evidence["robustness_bundle_table_rows"]["primary_target_metrics.csv"] == 3440

    results = {
        row["dataset"]: row for row in evidence["k8_omitted_target_results"]
    }
    assert set(results) == {"Docking-44", "DOCKSTRING-58"}
    docking = results["Docking-44"]
    dockstring = results["DOCKSTRING-58"]
    assert docking["raw_ridge_variance_weighted_r2_median"] == pytest.approx(
        0.9004739309249187
    )
    assert docking[
        "derived_residual_ridge_variance_weighted_r2_median"
    ] == pytest.approx(0.2779385175733468)
    assert docking[
        "direct_residual_ridge_variance_weighted_r2_median"
    ] == pytest.approx(0.2779535120813503)
    assert docking[
        "derived_residual_pc1_variance_weighted_r2_median"
    ] == pytest.approx(0.1803243099681972)
    assert docking[
        "direct_minus_derived_ridge_variance_weighted_r2_median"
    ] == pytest.approx(1.4994508003463023e-05)
    assert docking[
        "direct_minus_derived_ridge_variance_weighted_r2_min"
    ] < 0 < docking["direct_minus_derived_ridge_variance_weighted_r2_max"]

    assert dockstring["raw_ridge_variance_weighted_r2_median"] == pytest.approx(
        0.6979557280255279
    )
    assert dockstring[
        "derived_residual_ridge_variance_weighted_r2_median"
    ] == pytest.approx(0.2192108741945733)
    assert dockstring[
        "direct_residual_ridge_variance_weighted_r2_median"
    ] == pytest.approx(0.2192102405067755)
    assert dockstring[
        "derived_residual_pc1_variance_weighted_r2_median"
    ] == pytest.approx(0.0654109942139661)
    assert dockstring[
        "direct_minus_derived_ridge_variance_weighted_r2_median"
    ] == pytest.approx(-6.336877977908273e-07)
    assert dockstring[
        "direct_minus_derived_ridge_variance_weighted_r2_min"
    ] < 0 < dockstring["direct_minus_derived_ridge_variance_weighted_r2_max"]

    common = {
        (row["dataset"], row["surface"]): row
        for row in evidence["common_omitted_designed_vs_random"]
    }
    assert len(common) == 4
    assert common[("Docking-44", "raw_row_centered_residual")][
        "designed_benefit_mean_r2_random_median_median"
    ] == pytest.approx(0.0359384743850306)
    assert common[("DOCKSTRING-58", "raw_row_centered_residual")][
        "designed_benefit_mean_r2_random_median_median"
    ] == pytest.approx(0.058433368535526)
    assert all(row["random_panels_median"] == 3 for row in common.values())

    expected_opened = {
        key
        for key in public_core.INPUT_SPECS
        if key.startswith("selector_controls_")
        or key.startswith("selector_robustness_")
    } | {"scaffold_holdout_producer", "revision_loader_helper"}
    assert registry.opened_keys == expected_opened


def test_hotspot_panel_validation_import_is_exact_and_locked() -> None:
    registry = public_core.InputRegistry()
    evidence = public_core.hotspot_panel_validation_evidence(registry)
    assert evidence["analysis_status"] == "post_hoc_external_extension"
    assert evidence["experimental_support"]["Anastassiadis"] == {
        "ligands": 176,
        "targets": 21,
    }
    assert evidence["broad_vina_contract"]["primary_rows"] == 259_641
    assert evidence["hotspot_reliability"]["median"] == pytest.approx(
        0.7938631585426297
    )
    primary = evidence["panel_transfer"]["primary_inference"]
    assert primary["conservative_tie_cells_with_positive_improvement"] == 14
    assert primary["cells_averaged"] == 15
    assert primary[
        "conservative_mean_coverage_loss_improvement_over_ties"
    ] == pytest.approx(0.06068713903344698)
    joint = evidence["panel_transfer"]["joint_k_averaged_inference"]
    assert joint["normalized_trapezoid_auc_over_k"] == pytest.approx(
        0.061551311281194517
    )
    assert joint[
        "max_over_three_objectives_fwer_p_for_this_objective"
    ] == pytest.approx(0.006899655017249137)
    rank = evidence["panel_transfer"]["column_rank_inference"]
    assert rank["conservative_tie_cells_with_positive_improvement"] == 14
    assert rank[
        "conservative_mean_coverage_loss_improvement_over_ties"
    ] == pytest.approx(0.0383672539228506)
    cluster = evidence["chemical_cluster_multiplier"]["statistics"][
        "conservative_best_raw_minus_worst_residual_normalized_auc_over_k"
    ]
    assert cluster["positive_fraction"] == 1.0
    assert cluster["median"] == pytest.approx(0.06283173547231383)
    target_loo = evidence["target_leave_one_out_influence"]["panel_transfer"]
    assert target_loo["joint_normalized_auc_positive_omissions"] == 21
    assert target_loo["joint_normalized_auc_min"] == pytest.approx(
        0.029356209793230906
    )
    assert target_loo["joint_normalized_auc_max"] == pytest.approx(
        0.06223916889773784
    )
    assert len(evidence["verified_output_sha256"]) == 17


def test_sha256_text_bundle_manifest_fails_closed(tmp_path: Path) -> None:
    artifact_keys = ("selector_controls_summary", "selector_controls_readme")
    for key, content in zip(artifact_keys, ("{}\n", "contract\n"), strict=True):
        path = tmp_path / public_core.INPUT_SPECS[key].path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    checksum_path = tmp_path / public_core.INPUT_SPECS[
        "selector_controls_checksums"
    ].path
    checksum_path.write_text(
        "\n".join(
            f"{public_core.sha256_file(tmp_path / public_core.INPUT_SPECS[key].path)}  "
            f"{Path(public_core.INPUT_SPECS[key].path).name}"
            for key in artifact_keys
        )
        + "\n"
    )
    registry = public_core.InputRegistry(package=tmp_path)
    verified = public_core.verify_sha256_text_bundle(
        registry,
        checksum_key="selector_controls_checksums",
        artifact_keys=artifact_keys,
    )
    assert set(verified) == {"summary.json", "README.md"}

    checksum_path.write_text(checksum_path.read_text() + f"{'0' * 64}  extra.csv\n")
    with pytest.raises(ValueError, match="membership changed"):
        public_core.verify_sha256_text_bundle(
            public_core.InputRegistry(package=tmp_path),
            checksum_key="selector_controls_checksums",
            artifact_keys=artifact_keys,
        )


def test_released_public_core_has_locked_headlines_and_boundaries() -> None:
    ledger = json.loads(public_core.DEFAULT_OUTPUT.read_text())
    assert ledger["schema_version"] == "4.0.0"
    assert ledger["evidence_scope"] == "strict_public_core"
    assert ledger["hotspot_panel_validation"]["experimental_support"][
        "Anastassiadis"
    ]["ligands"] == 176

    docking = ledger["large_vina_panels"]["Docking-44"]
    dockstring = ledger["large_vina_panels"]["DOCKSTRING-58"]
    assert docking["support"]["analysis_rows"] == 12_651
    assert docking["support"]["targets"] == 44
    assert dockstring["support"]["analysis_rows"] == 260_060
    assert dockstring["support"]["targets"] == 58
    assert round(
        docking["column_standardized_surface"]["participation_ratio_dimension"], 3
    ) == 1.834
    assert round(
        docking["two_way_centered_residual_surface"][
            "participation_ratio_dimension"
        ],
        3,
    ) == 9.303
    assert round(
        dockstring["column_standardized_surface"]["participation_ratio_dimension"],
        3,
    ) == 2.266
    assert round(
        dockstring["two_way_centered_residual_surface"][
            "participation_ratio_dimension"
        ],
        3,
    ) == 18.206
    for panel in (docking, dockstring):
        for surface in (
            "column_standardized_surface",
            "two_way_centered_residual_surface",
        ):
            record = panel[surface]
            assert abs(
                record["participation_ratio_dimension"]
                - record["pr_identity_from_mean_squared_correlation"]
            ) < 1e-10

    assert docking["shared_axis"]["positive_loadings"] >= 43
    assert docking["shared_axis"]["cosine_similarity_with_uniform_axis"] > 0.95
    assert dockstring["shared_axis"]["cosine_similarity_with_uniform_axis"] > 0.95

    assert round(
        docking["covariance_sensitivity"]["unscaled_surface"][
            "participation_ratio_dimension"
        ],
        3,
    ) == 1.915
    assert round(
        docking["covariance_sensitivity"]["unscaled_two_way_centered_residual"][
            "participation_ratio_dimension"
        ],
        3,
    ) == 6.911
    assert round(
        dockstring["covariance_sensitivity"]["unscaled_surface"][
            "participation_ratio_dimension"
        ],
        3,
    ) == 2.474
    assert round(
        dockstring["covariance_sensitivity"][
            "unscaled_two_way_centered_residual"
        ]["participation_ratio_dimension"],
        3,
    ) == 8.235

    missing = docking["missing_data_sensitivity"]
    assert missing["source_missing_cells"] == 8_159
    assert missing["rows_with_any_missing"] == 3_354
    assert missing["most_incomplete_target"] == "4mqs"
    assert missing["most_incomplete_target_missing_cells"] == 3_352
    assert round(
        missing["variants"]["target_median_imputation"]["residual_pr"],
        3,
    ) == 9.528
    assert round(
        missing["variants"]["complete_case"]["residual_pr"],
        3,
    ) == 13.970
    assert round(
        missing["variants"]["observed_cell_wls_zero_residual_completion"][
            "residual_pr"
        ],
        3,
    ) == 9.663
    equal_n = missing["equal_n_random_support_control"]
    assert equal_n["repetitions"] == 1_000
    assert equal_n["q975"] < missing["variants"]["complete_case"]["residual_pr"]
    chemistry = {
        row["descriptor"]: row for row in missing["chemistry_effect_sizes"]
    }
    assert chemistry["molecular_weight"][
        "standardized_mean_difference_excluded_minus_complete"
    ] > 1.8
    assert chemistry["heavy_atoms"][
        "standardized_mean_difference_excluded_minus_complete"
    ] > 2.0


def test_public_core_registers_shift_recovery_and_experimental_repeatability() -> None:
    ledger = json.loads(public_core.DEFAULT_OUTPUT.read_text())
    revision = ledger["chemical_support_and_recovery"]
    for dataset, expected in {
        "Docking-44": (0.940, 0.973),
        "DOCKSTRING-58": (0.920, 0.965),
    }.items():
        trend = revision["support_dependence"][dataset]["ten_bin_continuum"][
            "row_centered_residual"
        ]
        assert trend["mw_bins"] == 10
        assert trend["bin_pairs"] == 45
        assert trend["spearman_mw_separation_vs_map_dissimilarity"] > 0.90
        assert round(
            revision["pilot_recovery"][dataset]["200"]["map_edges"][
                "geometry_spearman_mean"
            ],
            3,
        ) == expected[0]
        assert round(
            revision["pilot_recovery"][dataset]["500"]["map_edges"][
                "geometry_spearman_mean"
            ],
            3,
        ) == expected[1]
        for size in ("200", "500"):
            recovery = revision["pilot_recovery"][dataset][size]
            assert recovery["map_edges"]["reference_scope"] == (
                "row_disjoint_complement_map"
            )
            decision = recovery["eight_target_panel_decision"]
            assert decision["reference_scope"] == "row_disjoint_complement_map"
            assert (
                decision[
                    "sample_vs_fixed_full_source_panel_medoid_overlap_mean"
                ]
                > 0
            )

        for transform in ("raw", "row_centered_residual"):
            threshold = revision["support_dependence"][dataset]["threshold_sweep"][
                transform
            ]
            assert threshold["observed_below_control_q025_at_every_threshold"]
            assert threshold["maximum_observed_minus_control_mean"] < 0

    experimental = ledger["experimental_map_boundary"]
    cross_panel = next(
        row
        for row in experimental["cross_panel_experimental_geometry"]
        if row["transform"] == "two_way_centered"
    )
    assert round(cross_panel["edge_spearman"], 3) == 0.848
    assert cross_panel["target_label_qap_p_two_sided"] == 0.0001
    assert round(cross_panel["top_decile_precision"], 3) == 0.571
    for panel_name, informative_ligands, geometry in (
        ("DAVIS", 56, 0.30656542260433506),
        ("PKIS2", 154, 0.3260825480331591),
    ):
        panel = experimental["panels"][panel_name]
        assert panel["primary_same_support_informative_ligands"] == informative_ligands
        assert panel["released_map_reliability"]["ligand"][
            "split_map_spearman"
        ]["median"] > 0.75
        assert abs(
            panel["primary_same_support_geometry"]["two_way_centered"][
                "same_support_empirical_spearman"
            ]
            - geometry
        ) < 1e-12
        for unit in ("ligand", "butina_cluster"):
            bootstrap = panel["primary_same_support_bootstrap"][unit]
            assert bootstrap["same_support_spearman"]["interval_95"][0] > 0
            repeatability = panel["primary_same_support_split_half"][unit]
            assert 0.25 < repeatability[
                "median_cross_half_spearman_divided_by_experimental_self_repeatability"
            ] < 0.40
            assert repeatability["same_minus_cross_half"]["interval_95"][0] < 0
            assert repeatability["same_minus_cross_half"]["interval_95"][1] > 0
    serialized = json.dumps(experimental).lower()
    assert "davis" in serialized and "pkis2" in serialized

    overlap = ledger["cross_panel_overlap"]
    assert overlap["chemical_overlap"]["shared_full_standard_inchikeys"] == 582
    assert overlap["chemical_overlap"]["shared_connectivity_blocks"] == 804
    assert overlap["target_overlap"]["mapped_target_identities"] == 9
    assert overlap["target_overlap"]["same_receptor_structure_used"] == 3

    chemical_domain = ledger["chemical_domain_controls"]
    assert chemical_domain["configuration"]["dockstring_support_seeds"] == [
        71,
        72,
        73,
        74,
        75,
        20260809,
    ]
    boundary = {
        (row["dataset"], row["transformation"]): row
        for row in chemical_domain["key_boundary_results"]
    }
    assert round(
        boundary[("Docking-44", "row_centered_residual")][
            "observed_low_high_geometry_spearman"
        ],
        3,
    ) == 0.205
    assert round(
        boundary[("DOCKSTRING-58", "row_centered_residual")][
            "observed_low_high_geometry_spearman"
        ],
        3,
    ) == 0.351
    for row in boundary.values():
        assert row["observed_below_global_group_disjoint_q025"]
        assert row["observed_below_both_within_band_group_disjoint_q025"]

    descriptors = ledger["descriptor_domain_specificity"]
    descriptor_rows = {
        (row["dataset"], row["descriptor"]): row
        for row in descriptors["key_residual_results"]
    }
    assert len(descriptor_rows) == 14
    assert round(
        descriptor_rows[("Docking-44", "molecular_weight")][
            "observed_geometry_spearman"
        ],
        3,
    ) == 0.205
    assert round(
        descriptor_rows[("DOCKSTRING-58", "molecular_weight")][
            "observed_geometry_spearman"
        ],
        3,
    ) == 0.351
    controls = descriptors["random_disjoint_summary"]
    assert len(controls) == 14
    assert all(row["control_repetitions"] == 200 for row in controls)
    assert all(row["observed_below_control_q025"] for row in controls)
    ring = next(
        row
        for row in controls
        if row["dataset"] == "DOCKSTRING-58" and row["descriptor"] == "ring_count"
    )
    assert ring["combined_support_fraction"] == 1.0

    heldout = ledger["scaffold_holdout_recovery"]
    heldout_rows = {
        (row["dataset"], row["calibration_ligands"]): row
        for row in heldout["key_results"]
    }
    assert len(heldout_rows) == 4
    for dataset, expected in {
        "Docking-44": (0.925, 0.958),
        "DOCKSTRING-58": (0.914, 0.958),
    }.items():
        assert round(heldout_rows[(dataset, 200)]["geometry_spearman_mean"], 3) == expected[0]
        assert round(heldout_rows[(dataset, 500)]["geometry_spearman_mean"], 3) == expected[1]
        assert heldout_rows[(dataset, 200)]["maximum_group_overlap"] == 0
        assert heldout_rows[(dataset, 500)]["maximum_group_overlap"] == 0
    assert len(heldout["matched_random_row_results"]) == 4
    assert len(heldout["paired_design_contrasts"]) == 4
    assert all(
        row["q025_difference"] < 0 < row["q975_difference"]
        for row in heldout["paired_design_contrasts"]
    )

    nulls = ledger["residual_nulls"]["datasets"]
    assert set(nulls) == {"docking44", "dockstring58"}
    for dataset, observed_upper, conditioned_median, explained_share in (
        ("docking44", 10.0, 34.875, 0.238),
        ("dockstring58", 20.0, 47.039, 0.251),
    ):
        record = nulls[dataset]
        assert record["observed_residual_pr"] < observed_upper
        assert set(record["nulls"]) == {
            "additive_gaussian",
            "empirical_residual_column_permutation",
            "row_norm_preserving_random_direction",
            "molecular_weight_conditional_permutation",
        }
        for name in (
            "additive_gaussian",
            "empirical_residual_column_permutation",
            "row_norm_preserving_random_direction",
        ):
            null = record["nulls"][name]
            assert null["null_residual_pr_interval_95"][0] > 40
            assert null["empirical_lower_tail_probability"] < 0.01
        conditional = record["nulls"][
            "molecular_weight_conditional_permutation"
        ]["mw_10_bins"]
        assert round(
            conditional["mw_conditioned_residual_pr"]["median"], 3
        ) == conditioned_median
        assert round(
            conditional["comparisons"]["molecular_weight_quantile"][
                "explained_share_of_observed_pr_gap_relative_to_one_bin"
            ],
            3,
        ) == explained_share
        assert conditional["observed_residual_pr"] < conditional[
            "mw_conditioned_residual_pr"
        ]["interval_95"][0]

    reusable = ledger["reusable_target_map_audit"]
    assert reusable["command"] == "python analysis/target_map_audit.py --help"
    assert set(reusable["files"]) == {
        "target_map_audit_tool",
        "target_map_audit_test",
        "target_map_audit_fixture",
        "target_map_audit_documentation",
    }
    assert "row-centering preserves within-row target ranks" in reusable[
        "synthetic_invariants"
    ]


def test_manifest_covers_exact_inputs_code_and_evidence_with_valid_hashes() -> None:
    manifest = pd.read_csv(public_core.DEFAULT_MANIFEST)
    expected_inputs = {
        spec.path
        for spec in public_core.INPUT_SPECS.values()
        if spec.kind == "input"
    }
    observed_inputs = set(manifest.loc[manifest.kind.eq("input"), "path"])
    assert observed_inputs == expected_inputs
    assert set(manifest.kind) == {"input", "code", "output"}
    expected_code = {
        spec.path for spec in public_core.INPUT_SPECS.values() if spec.kind == "code"
    } | {str(Path(public_core.__file__).relative_to(public_core.PACKAGE))}
    assert set(manifest.loc[manifest.kind.eq("code"), "path"]) == expected_code
    assert set(manifest.loc[manifest.kind.eq("output"), "path"]) == {
        str(public_core.DEFAULT_OUTPUT.relative_to(public_core.PACKAGE))
    }
    for row in manifest.itertuples(index=False):
        path = public_core.PACKAGE / row.path
        assert path.is_file()
        assert path.stat().st_size == row.bytes
        assert public_core.sha256_file(path) == row.sha256

    digest, filename = public_core.DEFAULT_MANIFEST_DIGEST.read_text().strip().split()
    assert filename == public_core.DEFAULT_MANIFEST.name
    assert digest == public_core.sha256_file(public_core.DEFAULT_MANIFEST)

    canonical = "LicenseRef-NCBI-PubChem-molecular-data-policy"
    for key in ("hotspot_identity", "hotspot_identity_provenance"):
        assert public_core.INPUT_SPECS[key].license == canonical
        row = manifest.loc[manifest.path.eq(public_core.INPUT_SPECS[key].path)].iloc[0]
        assert row.license == canonical
        assert "CC0" not in row.license


def test_released_ledger_does_not_serialize_excluded_dependency_markers() -> None:
    serialized = public_core.DEFAULT_OUTPUT.read_text().lower()
    manifest = public_core.DEFAULT_MANIFEST.read_text().lower()
    assert all(
        marker not in serialized and marker not in manifest
        for marker in FORBIDDEN_DEPENDENCY_MARKERS
    )
    json.loads(serialized)
