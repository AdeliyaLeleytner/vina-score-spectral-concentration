#!/usr/bin/env python3
"""Build the fail-closed, public-only evidence ledger for manuscript v4.

This builder deliberately has a much narrower dependency surface than the
historical manuscript ledger.  It recomputes the large-panel spectral
diagnostics from the redistributed matrices and imports only the explicitly
listed public revision tables.  Every input read by the builder is checked
against an exact allowlist and recorded by byte size and SHA-256 digest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PACKAGE = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PACKAGE / "results" / "public_core_evidence.json"
DEFAULT_MANIFEST = PACKAGE / "results" / "public_core_source_manifest.csv"
DEFAULT_MANIFEST_DIGEST = PACKAGE / "results" / "public_core_source_manifest.sha256"

DOCKING44_TARGETS = (
    "1m2z", "1pbq", "1xoq", "2rh1", "2vt4", "2ydo", "2z5x", "3b66",
    "3kk6", "3ln1", "3rze", "4djh", "4ey7", "4iar", "4mqs", "4n6h",
    "5cxv", "5i71", "5tvn", "5u09", "5va1", "6cm4", "6kpf", "6kux",
    "6lqa", "6pdj", "6x3x", "6y1z", "7f8y", "7kwe", "7ljd", "7wc9",
    "7xnk", "7ym8", "8e9y", "8ef6", "8fhs", "8pjk", "8st0", "8wty",
    "8xvk", "8yn3", "9eo4", "V1A",
)

PKIS2_TARGET_MAP = {
    "ABL1": "ABL1-nonphosphorylated",
    "AKT1": "AKT1",
    "AKT2": "AKT2",
    "CDK2": "CDK2",
    "CSF1R": "CSF1R",
    "EGFR": "EGFR",
    "FGFR1": "FGFR1",
    "IGF1R": "IGF1R",
    "JAK2": "JAK2(JH1domain-catalytic)",
    "KDR": "VEGFR2",
    "KIT": "KIT",
    "LCK": "LCK",
    "MAP2K1": "MEK1",
    "MAPK1": "ERK2",
    "MAPK14": "p38-alpha",
    "MAPKAPK2": "MAPKAPK2",
    "MET": "MET",
    "PLK1": "PLK1",
    "PTK2": "FAK",
    "ROCK1": "ROCK1",
    "SRC": "SRC",
}


@dataclass(frozen=True)
class InputSpec:
    path: str
    role: str
    level: str
    kind: str = "input"
    license: str = "derived from redistributed public inputs"


INPUT_SPECS = {
    "docking44": InputSpec(
        "data/frozen/df_final_v4.csv.gz",
        "Docking-44 row-level matrix",
        "row_level",
        license="CC BY 4.0",
    ),
    "dockstring": InputSpec(
        "data/frozen/dockstring-dataset.tsv.gz",
        "DOCKSTRING row-level matrix",
        "row_level",
        license="Apache-2.0",
    ),
    "davis": InputSpec(
        "data/frozen/davis_complete.tab.gz",
        "DAVIS-Complete row-level affinity table",
        "row_level",
        license="CC0-1.0",
    ),
    "pkis2": InputSpec(
        "data/frozen/pkis2_s4.xlsx",
        "PKIS2 public source workbook",
        "row_level",
        license="CC BY 4.0",
    ),
    "dockstring_identity_contract": InputSpec(
        "data/frozen/dockstring_identity_contract_2026-08-03.csv.gz",
        "public DOCKSTRING molecular-identity contract",
        "row_level_derived_public",
        license="Apache-2.0",
    ),
    "dockstring_identity_provenance": InputSpec(
        "data/frozen/dockstring_identity_contract_2026-08-03.provenance.json",
        "provenance for the public DOCKSTRING molecular-identity contract",
        "derived_public",
        license="Apache-2.0",
    ),
    "target_families": InputSpec(
        "data/target_families.csv",
        "public broad target-family labels",
        "curated_public",
        license="CC BY 4.0",
    ),
    "cross_panel_target_mapping": InputSpec(
        "results/residual_mechanism/cross_panel_target_mapping.csv",
        "audited cross-panel target-identity mapping",
        "curated_public",
        license="CC BY 4.0",
    ),
    "cross_panel_overlap": InputSpec(
        "results/cross_panel_overlap_audit/summary.json",
        "cross-panel chemical and target overlap audit",
        "derived_public",
    ),
    "cross_panel_overlap_producer": InputSpec(
        "analysis/cross_panel_overlap_audit.py",
        "producer for the cross-panel overlap audit",
        "producer_code",
        "code",
    ),
    "cross_panel_overlap_test": InputSpec(
        "analysis/test_cross_panel_overlap_audit.py",
        "tests for the cross-panel overlap audit",
        "producer_test",
        "code",
    ),
    "revision_metadata": InputSpec(
        "results/revision_map_decision_sensitivities/metadata.json",
        "support-dependence and pilot-recovery analysis contract",
        "derived_public",
    ),
    "mw_threshold_summary": InputSpec(
        "results/revision_map_decision_sensitivities/mw_threshold_sweep_summary.csv",
        "molecular-weight threshold sweep summary",
        "derived_public",
    ),
    "mw_continuous_trends": InputSpec(
        "results/revision_map_decision_sensitivities/mw_decile_continuous_trends.csv",
        "molecular-weight continuum summary",
        "derived_public",
    ),
    "recovery_edge_summary": InputSpec(
        "results/revision_map_decision_sensitivities/recovery_edge_summary.csv",
        "pilot edge-recovery summary",
        "derived_public",
    ),
    "recovery_decision_summary": InputSpec(
        "results/revision_map_decision_sensitivities/recovery_decision_summary.csv",
        "pilot panel-compression decision summary",
        "derived_public",
    ),
    "revision_checksums": InputSpec(
        "results/revision_map_decision_sensitivities/output_checksums.json",
        "checksum manifest for support-dependence and pilot-recovery outputs",
        "derived_public",
    ),
    "revision_producer": InputSpec(
        "analysis/revision_map_decision_sensitivities.py",
        "producer for support-dependence and pilot-recovery outputs",
        "producer_code",
        "code",
    ),
    "revision_test": InputSpec(
        "analysis/test_revision_map_decision_sensitivities.py",
        "tests for support-dependence and pilot-recovery outputs",
        "producer_test",
        "code",
    ),
    "revision_loader_helper": InputSpec(
        "analysis/residual_mechanism_analysis.py",
        "public row-level Vina loaders imported by the revision producer",
        "supporting_code",
        "code",
    ),
    "revision_loader_helper_test": InputSpec(
        "analysis/test_residual_mechanism_analysis.py",
        "tests for the public row-level Vina loaders",
        "supporting_test",
        "code",
    ),
    "experimental_reliability": InputSpec(
        "results/experimental_map_reliability/map_reliability.csv",
        "experimental-map reliability summary",
        "derived_public",
    ),
    "experimental_summary": InputSpec(
        "results/experimental_map_reliability/summary.json",
        "public experimental reliability analysis contract",
        "derived_public",
    ),
    "experimental_cross_panel_geometry": InputSpec(
        "results/experimental_map_reliability/cross_panel_geometry.csv",
        "cross-panel DAVIS--PKIS2 experimental-map geometry",
        "derived_public",
    ),
    "experimental_edge_stability": InputSpec(
        "results/experimental_map_reliability/edge_stability.csv",
        "experimental-map edge-stability summary",
        "derived_public",
    ),
    "same_support_geometry": InputSpec(
        "results/experimental_map_reliability/same_support_geometry.csv",
        "same-ligand-support docking--experiment geometry summary",
        "derived_public",
    ),
    "same_support_bootstrap": InputSpec(
        "results/experimental_map_reliability/same_support_bootstrap.csv",
        "same-support geometry bootstrap summary",
        "derived_public",
    ),
    "same_support_split_half": InputSpec(
        "results/experimental_map_reliability/same_support_split_half.csv",
        "same-support experimental split-half reliability",
        "derived_public",
    ),
    "experimental_checksums": InputSpec(
        "results/experimental_map_reliability/output_checksums.json",
        "checksum manifest for experimental reliability outputs",
        "derived_public",
    ),
    "experimental_producer": InputSpec(
        "analysis/experimental_map_reliability.py",
        "producer for experimental reliability and same-support outputs",
        "producer_code",
        "code",
    ),
    "experimental_test": InputSpec(
        "analysis/test_experimental_map_reliability.py",
        "tests for experimental reliability and same-support outputs",
        "producer_test",
        "code",
    ),
    "davis_loader": InputSpec(
        "analysis/dense_davis_benchmark.py",
        "public DAVIS loader imported by the experimental producer",
        "supporting_code",
        "code",
    ),
    "davis_loader_test": InputSpec(
        "analysis/test_dense_davis_benchmark.py",
        "tests for the public DAVIS loader",
        "supporting_test",
        "code",
    ),
    "pkis2_loader": InputSpec(
        "analysis/dense_pkis2_benchmark.py",
        "public PKIS2 loader imported by the experimental producer",
        "supporting_code",
        "code",
    ),
    "pkis2_loader_test": InputSpec(
        "analysis/test_dense_pkis2_benchmark.py",
        "tests for the public PKIS2 loader",
        "supporting_test",
        "code",
    ),
    "residual_null_summary": InputSpec(
        "results/public_residual_null_audit/summary.json",
        "standalone residual-null audit including molecular-weight conditioning",
        "derived_public",
    ),
    "residual_null_conditional_repeats": InputSpec(
        "results/public_residual_null_audit/mw_conditional_null_repeats.csv",
        "molecular-weight-conditioned residual-null replicate table",
        "derived_public",
    ),
    "residual_null_bin_contract": InputSpec(
        "results/public_residual_null_audit/mw_conditional_bin_contract.csv",
        "molecular-weight bin membership and support contract",
        "derived_public",
    ),
    "residual_null_readme": InputSpec(
        "results/public_residual_null_audit/README.md",
        "interpretation and reproduction contract for residual nulls",
        "derived_public",
    ),
    "residual_null_checksums": InputSpec(
        "results/public_residual_null_audit/output_checksums.json",
        "checksum manifest for the standalone residual-null audit",
        "derived_public",
    ),
    "residual_null_producer": InputSpec(
        "analysis/public_residual_null_audit.py",
        "producer for the standalone residual-null audit",
        "producer_code",
        "code",
    ),
    "residual_null_test": InputSpec(
        "analysis/test_public_residual_null_audit.py",
        "tests for the standalone residual-null audit",
        "producer_test",
        "code",
    ),
    "missing_data_summary": InputSpec(
        "results/public_docking44_missing_data_sensitivity/summary.json",
        "standalone Docking-44 missing-data sensitivity contract",
        "derived_public",
    ),
    "missing_data_by_target": InputSpec(
        "results/public_docking44_missing_data_sensitivity/missingness_by_target.csv",
        "target-wise Docking-44 missingness table",
        "derived_public",
    ),
    "missing_data_spectral": InputSpec(
        "results/public_docking44_missing_data_sensitivity/spectral_sensitivity.csv",
        "Docking-44 spectral missing-data sensitivity table",
        "derived_public",
    ),
    "missing_data_equal_n_controls": InputSpec(
        "results/public_docking44_missing_data_sensitivity/equal_n_random_controls.csv",
        "equal-size random-support control replicates",
        "derived_public",
    ),
    "missing_data_chemistry_effects": InputSpec(
        "results/public_docking44_missing_data_sensitivity/chemistry_effect_sizes.csv",
        "chemical-support effect-size table",
        "derived_public",
    ),
    "missing_data_chemistry_groups": InputSpec(
        "results/public_docking44_missing_data_sensitivity/chemistry_group_summary.csv",
        "complete- versus incomplete-row chemistry summaries",
        "derived_public",
    ),
    "missing_data_readme": InputSpec(
        "results/public_docking44_missing_data_sensitivity/README.md",
        "interpretation contract for the missing-data bundle",
        "derived_public",
    ),
    "missing_data_checksums": InputSpec(
        "results/public_docking44_missing_data_sensitivity/output_checksums.json",
        "checksum manifest for the missing-data bundle",
        "derived_public",
    ),
    "missing_data_producer": InputSpec(
        "analysis/public_docking44_missing_data_sensitivity.py",
        "standalone producer for Docking-44 missing-data sensitivity",
        "producer_code",
        "code",
    ),
    "missing_data_test": InputSpec(
        "analysis/test_public_docking44_missing_data_sensitivity.py",
        "tests for Docking-44 missing-data sensitivity",
        "producer_test",
        "code",
    ),
    "chemical_domain_summary": InputSpec(
        "results/public_chemical_domain_controls/summary.json",
        "standalone public chemical-domain control contract",
        "derived_public",
    ),
    "chemical_domain_observed": InputSpec(
        "results/public_chemical_domain_controls/observed_low_high_mw_contrasts.csv",
        "observed low- versus high-MW map contrasts",
        "derived_public",
    ),
    "chemical_domain_seed_contrasts": InputSpec(
        "results/public_chemical_domain_controls/dockstring_support_seed_contrasts.csv",
        "DOCKSTRING support-seed map contrasts",
        "derived_public",
    ),
    "chemical_domain_seed_summary": InputSpec(
        "results/public_chemical_domain_controls/dockstring_support_seed_summary.csv",
        "DOCKSTRING support-seed sensitivity summary",
        "derived_public",
    ),
    "chemical_domain_global_controls": InputSpec(
        "results/public_chemical_domain_controls/mw_matched_group_disjoint_controls.csv",
        "MW-matched chemical-group-disjoint control replicates",
        "derived_public",
    ),
    "chemical_domain_global_summary": InputSpec(
        "results/public_chemical_domain_controls/mw_matched_group_disjoint_summary.csv",
        "MW-matched chemical-group-disjoint control summary",
        "derived_public",
    ),
    "chemical_domain_within_controls": InputSpec(
        "results/public_chemical_domain_controls/within_band_reproducibility_controls.csv",
        "within-MW-band reproducibility control replicates",
        "derived_public",
    ),
    "chemical_domain_within_summary": InputSpec(
        "results/public_chemical_domain_controls/within_band_reproducibility_summary.csv",
        "within-MW-band reproducibility summary",
        "derived_public",
    ),
    "chemical_domain_boundary_table": InputSpec(
        "results/public_chemical_domain_controls/causal_boundary_summary.csv",
        "compact chemical-domain causal-boundary table",
        "derived_public",
    ),
    "chemical_domain_readme": InputSpec(
        "results/public_chemical_domain_controls/README.md",
        "interpretation contract for chemical-domain controls",
        "derived_public",
    ),
    "chemical_domain_checksums": InputSpec(
        "results/public_chemical_domain_controls/output_checksums.json",
        "checksum manifest for chemical-domain controls",
        "derived_public",
    ),
    "chemical_domain_producer": InputSpec(
        "analysis/public_chemical_domain_controls.py",
        "standalone producer for chemical-domain controls",
        "producer_code",
        "code",
    ),
    "chemical_domain_test": InputSpec(
        "analysis/test_public_chemical_domain_controls.py",
        "tests for chemical-domain controls",
        "producer_test",
        "code",
    ),
    "descriptor_domain_summary": InputSpec(
        "results/public_descriptor_domain_specificity/summary.json",
        "strict-public descriptor-domain analysis contract",
        "derived_public",
    ),
    "descriptor_domain_extremes": InputSpec(
        "results/public_descriptor_domain_specificity/descriptor_extremes.csv",
        "inclusive-threshold descriptor-domain target-map contrasts",
        "derived_public",
    ),
    "descriptor_domain_correlations": InputSpec(
        "results/public_descriptor_domain_specificity/descriptor_correlations.csv",
        "descriptor correlation table for domain-axis interpretation",
        "derived_public",
    ),
    "descriptor_domain_family_summary": InputSpec(
        "results/public_descriptor_domain_specificity/descriptor_family_summary.csv",
        "size-related versus other descriptor-axis summary",
        "derived_public",
    ),
    "descriptor_domain_random_controls": InputSpec(
        "results/public_descriptor_domain_specificity/descriptor_random_disjoint_controls.csv",
        "matched-size random-row-disjoint descriptor controls",
        "derived_public",
    ),
    "descriptor_domain_random_summary": InputSpec(
        "results/public_descriptor_domain_specificity/descriptor_random_disjoint_summary.csv",
        "matched-size descriptor-control summary",
        "derived_public",
    ),
    "descriptor_domain_readme": InputSpec(
        "results/public_descriptor_domain_specificity/README.md",
        "interpretation contract for descriptor-domain sensitivity",
        "derived_public",
    ),
    "descriptor_domain_checksums": InputSpec(
        "results/public_descriptor_domain_specificity/output_checksums.json",
        "checksum manifest for descriptor-domain sensitivity",
        "derived_public",
    ),
    "descriptor_domain_producer": InputSpec(
        "analysis/public_descriptor_domain_specificity.py",
        "strict-public producer for descriptor-domain sensitivity",
        "producer_code",
        "code",
    ),
    "descriptor_domain_test": InputSpec(
        "analysis/test_public_descriptor_domain_specificity.py",
        "tests for descriptor-domain sensitivity",
        "producer_test",
        "code",
    ),
    "scaffold_holdout_summary": InputSpec(
        "results/public_scaffold_holdout_recovery/summary.json",
        "strict-public chemical-group-held-out recovery contract",
        "derived_public",
    ),
    "scaffold_holdout_metrics": InputSpec(
        "results/public_scaffold_holdout_recovery/scaffold_holdout_metrics.csv",
        "chemical-group-held-out recovery replicates",
        "derived_public",
    ),
    "scaffold_holdout_table": InputSpec(
        "results/public_scaffold_holdout_recovery/scaffold_holdout_summary.csv",
        "chemical-group-held-out recovery summary",
        "derived_public",
    ),
    "scaffold_holdout_random_summary": InputSpec(
        "results/public_scaffold_holdout_recovery/matched_random_row_holdout_summary.csv",
        "matched random-row-holdout recovery summary",
        "derived_public",
    ),
    "scaffold_holdout_design_contrasts": InputSpec(
        "results/public_scaffold_holdout_recovery/chemical_group_minus_random_summary.csv",
        "paired chemical-group-minus-random holdout contrasts",
        "derived_public",
    ),
    "scaffold_holdout_fold_diagnostics": InputSpec(
        "results/public_scaffold_holdout_recovery/chemical_group_fold_diagnostics.csv",
        "fold-level chemical-group and missing-data diagnostics",
        "derived_public",
    ),
    "scaffold_holdout_by_fold": InputSpec(
        "results/public_scaffold_holdout_recovery/scaffold_holdout_by_fold_summary.csv",
        "fold-stratified chemical-group recovery summaries",
        "derived_public",
    ),
    "scaffold_holdout_similarity": InputSpec(
        "results/public_scaffold_holdout_recovery/heldout_maximum_morgan_tanimoto.csv",
        "exact held-out-to-calibration maximum Morgan similarity records",
        "derived_public",
    ),
    "scaffold_holdout_similarity_summary": InputSpec(
        "results/public_scaffold_holdout_recovery/heldout_maximum_morgan_tanimoto_summary.csv",
        "fold-level exact maximum Morgan similarity summaries",
        "derived_public",
    ),
    "scaffold_holdout_readme": InputSpec(
        "results/public_scaffold_holdout_recovery/README.md",
        "interpretation contract for chemical-group-held-out recovery",
        "derived_public",
    ),
    "scaffold_holdout_checksums": InputSpec(
        "results/public_scaffold_holdout_recovery/output_checksums.json",
        "checksum manifest for chemical-group-held-out recovery",
        "derived_public",
    ),
    "scaffold_holdout_producer": InputSpec(
        "analysis/public_scaffold_holdout_recovery.py",
        "strict-public producer for chemical-group-held-out recovery",
        "producer_code",
        "code",
    ),
    "scaffold_holdout_test": InputSpec(
        "analysis/test_public_scaffold_holdout_recovery.py",
        "tests for chemical-group-held-out recovery",
        "producer_test",
        "code",
    ),
    "selector_controls_summary": InputSpec(
        "results/public_panel_selector_controls/summary.json",
        "strict-public target-panel selector and size control contract",
        "derived_public",
    ),
    "selector_controls_metrics": InputSpec(
        "results/public_panel_selector_controls/panel_metrics.csv.gz",
        "fold-level target-panel selector reconstruction metrics",
        "derived_public",
    ),
    "selector_controls_splits": InputSpec(
        "results/public_panel_selector_controls/split_diagnostics.csv",
        "chemical-group-disjoint selector split diagnostics",
        "derived_public",
    ),
    "selector_controls_selections": InputSpec(
        "results/public_panel_selector_controls/panel_selections.csv",
        "selected targets for deterministic panel selectors",
        "derived_public",
    ),
    "selector_controls_baseline_contrasts": InputSpec(
        "results/public_panel_selector_controls/paired_baseline_contrasts.csv",
        "fold-paired selector contrasts against random, PC1 and descriptor controls",
        "derived_public",
    ),
    "selector_controls_pair_contrasts": InputSpec(
        "results/public_panel_selector_controls/paired_selector_contrasts.csv",
        "fold-paired exact-medoid and pivoted-QR selector contrasts",
        "derived_public",
    ),
    "selector_controls_metric_summary": InputSpec(
        "results/public_panel_selector_controls/selector_metric_summary.csv",
        "five-fold selector metric summaries",
        "derived_public",
    ),
    "selector_controls_baseline_summary": InputSpec(
        "results/public_panel_selector_controls/baseline_contrast_summary.csv",
        "five-fold selector baseline contrast summaries",
        "derived_public",
    ),
    "selector_controls_pair_summary": InputSpec(
        "results/public_panel_selector_controls/selector_pair_summary.csv",
        "five-fold selector-pair summaries",
        "derived_public",
    ),
    "selector_controls_readme": InputSpec(
        "results/public_panel_selector_controls/README.md",
        "interpretation contract for selector controls",
        "derived_public",
    ),
    "selector_controls_checksums": InputSpec(
        "results/public_panel_selector_controls/checksums.sha256",
        "SHA-256 manifest for selector controls",
        "derived_public",
    ),
    "selector_controls_producer": InputSpec(
        "analysis/public_panel_selector_controls.py",
        "strict-public producer for panel selector and size controls",
        "producer_code",
        "code",
    ),
    "selector_controls_test": InputSpec(
        "analysis/test_public_panel_selector_controls.py",
        "tests for panel selector and size controls",
        "producer_test",
        "code",
    ),
    "selector_robustness_summary": InputSpec(
        "results/public_panel_selector_robustness/summary.json",
        "strict-public k=8 common-omitted and direct-residual contract",
        "derived_public",
    ),
    "selector_robustness_metrics": InputSpec(
        "results/public_panel_selector_robustness/panel_metrics.csv",
        "k=8 selector robustness fold metrics",
        "derived_public",
    ),
    "selector_robustness_target_metrics": InputSpec(
        "results/public_panel_selector_robustness/primary_target_metrics.csv",
        "target-level primary-model selector robustness metrics",
        "derived_public",
    ),
    "selector_robustness_splits": InputSpec(
        "results/public_panel_selector_robustness/split_diagnostics.csv",
        "reproduced chemical-group fold diagnostics",
        "derived_public",
    ),
    "selector_robustness_common_pairs": InputSpec(
        "results/public_panel_selector_robustness/common_omitted_pair_contrasts.csv",
        "designed-versus-random contrasts on common omitted targets",
        "derived_public",
    ),
    "selector_robustness_common_splits": InputSpec(
        "results/public_panel_selector_robustness/common_omitted_split_contrasts.csv",
        "random-panel-aggregated common-omitted split contrasts",
        "derived_public",
    ),
    "selector_robustness_common_summary": InputSpec(
        "results/public_panel_selector_robustness/common_omitted_summary.csv",
        "five-fold common-omitted selector summaries",
        "derived_public",
    ),
    "selector_robustness_direct_contrasts": InputSpec(
        "results/public_panel_selector_robustness/direct_residual_contrasts.csv",
        "paired direct-versus-derived residual fold contrasts",
        "derived_public",
    ),
    "selector_robustness_direct_summary": InputSpec(
        "results/public_panel_selector_robustness/direct_residual_summary.csv",
        "five-fold direct-residual sensitivity summaries",
        "derived_public",
    ),
    "selector_robustness_readme": InputSpec(
        "results/public_panel_selector_robustness/README.md",
        "interpretation contract for k=8 selector robustness",
        "derived_public",
    ),
    "selector_robustness_checksums": InputSpec(
        "results/public_panel_selector_robustness/checksums.sha256",
        "SHA-256 manifest for k=8 selector robustness",
        "derived_public",
    ),
    "selector_robustness_producer": InputSpec(
        "analysis/public_panel_selector_robustness.py",
        "strict-public producer for k=8 selector robustness",
        "producer_code",
        "code",
    ),
    "selector_robustness_test": InputSpec(
        "analysis/test_public_panel_selector_robustness.py",
        "tests for k=8 selector robustness",
        "producer_test",
        "code",
    ),
    "hotspot_identity": InputSpec(
        "data/frozen/anastassiadis2011_pubchem_identity_2026-08-10.csv",
        "audited PubChem identity crosswalk for the HotSpot compounds",
        "row_level_derived_public",
        license="PubChem public data; derived crosswalk CC0-1.0",
    ),
    "hotspot_identity_provenance": InputSpec(
        "data/frozen/anastassiadis2011_pubchem_identity_2026-08-10.provenance.json",
        "provenance for the HotSpot identity crosswalk",
        "derived_public",
        license="CC0-1.0",
    ),
    "hotspot_kinase_fasta": InputSpec(
        "data/frozen/dockstring_kinase_receptors.fasta",
        "fixed-21 DOCKSTRING receptor sequences",
        "derived_public",
        license="Apache-2.0",
    ),
    "hotspot_kinase_manifest": InputSpec(
        "data/frozen/dockstring_kinase_receptors_manifest.csv",
        "provenance manifest for the fixed-21 receptor sequences",
        "derived_public",
        license="Apache-2.0",
    ),
    "hotspot_summary": InputSpec(
        "results/public_anastassiadis_panel_validation/summary.json",
        "HotSpot fixed-21 panel validation contract",
        "derived_public",
    ),
    "hotspot_excluded_ligands": InputSpec(
        "results/public_anastassiadis_panel_validation/excluded_ligands.csv",
        "HotSpot missing-cell exclusions",
        "derived_public",
    ),
    "hotspot_cluster_multiplier": InputSpec(
        "results/public_anastassiadis_panel_validation/chemical_cluster_multiplier.csv",
        "HotSpot chemical-cluster multiplier sensitivity",
        "derived_public",
    ),
    "hotspot_identity_overlap": InputSpec(
        "results/public_anastassiadis_panel_validation/identity_overlap_summary.csv",
        "HotSpot identity and overlap summary",
        "derived_public",
    ),
    "hotspot_joint_qap": InputSpec(
        "results/public_anastassiadis_panel_validation/joint_cross_assay_qap.csv",
        "shared-target-label panel-transfer inference",
        "derived_public",
    ),
    "hotspot_ligand_bootstrap": InputSpec(
        "results/public_anastassiadis_panel_validation/ligand_bootstrap_descriptive.csv",
        "HotSpot ligand-bootstrap map sensitivity",
        "derived_public",
    ),
    "hotspot_map_concordance": InputSpec(
        "results/public_anastassiadis_panel_validation/map_concordance.csv",
        "HotSpot cross-assay target-map concordance",
        "derived_public",
    ),
    "hotspot_mapping_sensitivities": InputSpec(
        "results/public_anastassiadis_panel_validation/mapping_sensitivities.csv",
        "HotSpot mapping sensitivity grid",
        "derived_public",
    ),
    "hotspot_mapping_summary": InputSpec(
        "results/public_anastassiadis_panel_validation/mapping_sensitivity_summary.csv",
        "HotSpot mapping sensitivity summaries",
        "derived_public",
    ),
    "hotspot_panel_selections": InputSpec(
        "results/public_anastassiadis_panel_validation/optimal_panel_selections.csv",
        "exact fixed-21 panel selections and numerical tie sets",
        "derived_public",
    ),
    "hotspot_deleak_sensitivity": InputSpec(
        "results/public_anastassiadis_panel_validation/reference_deleak_sensitivity.csv",
        "HotSpot molecular-overlap exclusion sensitivity",
        "derived_public",
    ),
    "hotspot_panel_transfer": InputSpec(
        "results/public_anastassiadis_panel_validation/same_endpoint_panel_transfer.csv",
        "raw-versus-residual panel transfer results",
        "derived_public",
    ),
    "hotspot_split_half": InputSpec(
        "results/public_anastassiadis_panel_validation/split_half_reliability.csv",
        "HotSpot split-half map reliability",
        "derived_public",
    ),
    "hotspot_target_loo_maps": InputSpec(
        "results/public_anastassiadis_panel_validation/target_leave_one_out_map_concordance.csv",
        "fixed-panel leave-one-target-out assay-map concordance",
        "derived_public",
    ),
    "hotspot_target_loo_transfer": InputSpec(
        "results/public_anastassiadis_panel_validation/target_leave_one_out_panel_transfer.csv",
        "fixed-panel leave-one-target-out panel-transfer sensitivity",
        "derived_public",
    ),
    "hotspot_target_mapping": InputSpec(
        "results/public_anastassiadis_panel_validation/target_mapping.csv",
        "HotSpot-to-fixed-21 target mapping",
        "derived_public",
    ),
    "hotspot_readme": InputSpec(
        "results/public_anastassiadis_panel_validation/README.md",
        "HotSpot validation interpretation contract",
        "derived_public",
    ),
    "hotspot_checksums": InputSpec(
        "results/public_anastassiadis_panel_validation/bundle_checksums.sha256",
        "SHA-256 manifest for the HotSpot validation bundle",
        "derived_public",
    ),
    "hotspot_producer": InputSpec(
        "analysis/public_anastassiadis_panel_validation.py",
        "strict-public HotSpot panel-validation producer",
        "producer_code",
        "code",
    ),
    "hotspot_test": InputSpec(
        "analysis/test_public_anastassiadis_panel_validation.py",
        "tests for HotSpot panel validation",
        "producer_test",
        "code",
    ),
    "hotspot_identity_producer": InputSpec(
        "analysis/public_anastassiadis_identity_audit.py",
        "producer for the HotSpot PubChem identity crosswalk",
        "producer_code",
        "code",
    ),
    "hotspot_identity_test": InputSpec(
        "analysis/test_public_anastassiadis_identity_audit.py",
        "tests for the HotSpot identity crosswalk",
        "producer_test",
        "code",
    ),
    "hotspot_fetcher": InputSpec(
        "analysis/fetch_anastassiadis_supplement.py",
        "checksum-gated fetcher for the non-redistributed official workbook",
        "producer_code",
        "code",
    ),
    "target_map_audit_tool": InputSpec(
        "analysis/target_map_audit.py",
        "generic strict-public target-map audit CLI",
        "research_tool",
        "code",
    ),
    "target_map_audit_test": InputSpec(
        "analysis/test_target_map_audit.py",
        "tests for the generic target-map audit CLI",
        "research_tool_test",
        "code",
    ),
    "target_map_audit_fixture": InputSpec(
        "analysis/fixtures/target_map_audit_synthetic.csv",
        "synthetic support-shift fixture for the generic audit",
        "synthetic_fixture",
        license="CC0-1.0",
    ),
    "target_map_audit_documentation": InputSpec(
        "analysis/TARGET_MAP_AUDIT.md",
        "usage and interpretation contract for the generic audit",
        "research_tool_documentation",
        "code",
    ),
    "public_core_builder_test": InputSpec(
        "analysis/test_build_public_core_evidence.py",
        "tests for the strict public-core evidence builder",
        "builder_test",
        "code",
    ),
    "v4_submission_consistency_test": InputSpec(
        "analysis/test_v4_submission_consistency.py",
        "submission-facing consistency tests for the strict-public v4 package",
        "report_test",
        "code",
    ),
}

PINNED_SHA256 = {
    "docking44": "25436e49b9fc80421d8499ef289ed8ef00b4ab35b904c7d930fc377f493e1d51",
    "dockstring": "e15a58258dbd613374e499bb17e5428a0df3f1f53b34758042f8e3cd3b53eb64",
    "davis": "e5614549ec970bf24bd58d04d8a74feb8f8a53806410eaa8d84d24e96a4ff827",
    "pkis2": "48ead22a1f860cd0d5096fa87d5acd329f722fe8d65e693bb0be682a333e2a2c",
    "dockstring_identity_contract": "5c5d06635cb276f543f9744487d6b8d4015432087e5f87b7875628973b8f32b3",
    "dockstring_identity_provenance": "a448285ffe5b9ced1f82179ae59e29fe1f74b6a5050ac55f1173fb1106963459",
    "target_families": "bc313ba0d28d7de7981a67c0088d44cb7cdd86c09e7b319bfb2e609040e3c9a6",
    "cross_panel_target_mapping": "ccaa1e43369246a333462712ad869d3d52111e3f1bb6a7c18f340ed4903e69d3",
    "hotspot_identity": "a3fb9469484ca94f34c51497ab28e447a2a0d9cce0a42829c927726148b767b2",
    "hotspot_identity_provenance": "08c11247309b186a52f593639a39e5fb89ec6583faabd108366d0587adec1937",
    "hotspot_kinase_fasta": "c80e13dcb7e00dd297c85198b7afccd81d582950b6c9449b760a1c618322b244",
    "hotspot_kinase_manifest": "6e144d60256ad8f1c62daa78b778ee67bfd95ca6ed281a62d1a8f3ea6d99cda0",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("non-finite value cannot be serialized in public evidence")
    return value


class InputRegistry:
    """Exact-path input gateway; reads outside ``INPUT_SPECS`` fail closed."""

    def __init__(self, package: Path = PACKAGE) -> None:
        self.package = package.resolve()
        self._allowed = {key: self.package / spec.path for key, spec in INPUT_SPECS.items()}
        self._opened: dict[str, dict[str, Any]] = {}

    def path(self, key: str) -> Path:
        if key not in self._allowed:
            raise PermissionError(f"input key is outside the public-core contract: {key}")
        declared = self._allowed[key]
        if not declared.is_file():
            raise FileNotFoundError(f"required public-core input is missing: {declared}")
        cursor = declared
        while cursor != self.package:
            if cursor.is_symlink():
                raise PermissionError(f"symlinks are not allowed in public-core inputs: {cursor}")
            cursor = cursor.parent
        path = declared.resolve()
        expected = (self.package / INPUT_SPECS[key].path).resolve()
        if path != expected or self.package not in path.parents:
            raise PermissionError(f"input escaped the package allowlist: {path}")
        digest = sha256_file(path)
        pinned = PINNED_SHA256.get(key)
        if pinned is not None and digest != pinned:
            raise ValueError(
                f"pinned public input checksum mismatch for {INPUT_SPECS[key].path}: "
                f"expected {pinned}, observed {digest}"
            )
        self._opened[key] = {
            "path": INPUT_SPECS[key].path,
            "kind": INPUT_SPECS[key].kind,
            "level": INPUT_SPECS[key].level,
            "role": INPUT_SPECS[key].role,
            "license": (
                "MIT" if INPUT_SPECS[key].kind == "code" else INPUT_SPECS[key].license
            ),
            "bytes": path.stat().st_size,
            "sha256": digest,
        }
        return path

    def read_csv(self, key: str, **kwargs: Any) -> pd.DataFrame:
        return pd.read_csv(self.path(key), **kwargs)

    def read_json(self, key: str) -> dict[str, Any]:
        return json.loads(self.path(key).read_text())

    def read_excel(self, key: str, **kwargs: Any) -> pd.DataFrame:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Unknown extension is not supported and will be removed",
                category=UserWarning,
                module=r"openpyxl\.worksheet\._reader",
            )
            return pd.read_excel(self.path(key), **kwargs)

    def manifest_rows(self) -> list[dict[str, Any]]:
        return [self._opened[key] for key in sorted(self._opened)]

    @property
    def opened_keys(self) -> set[str]:
        return set(self._opened)

    def opened_record(self, key: str) -> dict[str, Any]:
        if key not in self._opened:
            self.path(key)
        return dict(self._opened[key])


def verify_bundle_checksums(
    registry: InputRegistry,
    *,
    checksum_key: str,
    artifact_keys: tuple[str, ...],
) -> dict[str, str]:
    manifest = registry.read_json(checksum_key)
    if manifest.get("algorithm") == "sha256" and isinstance(
        manifest.get("files"), dict
    ):
        file_entries = manifest["files"]
    elif "algorithm" not in manifest and all(
        isinstance(value, dict) and "sha256" in value
        for value in manifest.values()
    ):
        # The experimental producer predates the wrapped checksum schema and
        # emits a flat filename -> {bytes, sha256} contract.
        file_entries = manifest
    else:
        raise ValueError(f"invalid output-checksum contract: {checksum_key}")
    verified: dict[str, str] = {}
    checksum_parent = Path(INPUT_SPECS[checksum_key].path).parent
    for artifact_key in artifact_keys:
        path = registry.path(artifact_key)
        relative = Path(INPUT_SPECS[artifact_key].path)
        if relative.parent != checksum_parent:
            raise ValueError(f"artifact is outside checksum bundle: {artifact_key}")
        expected_entry = file_entries.get(relative.name)
        if isinstance(expected_entry, dict):
            expected = expected_entry.get("sha256")
            expected_bytes = expected_entry.get("bytes")
            if expected_bytes is not None and int(expected_bytes) != path.stat().st_size:
                raise ValueError(
                    f"derived artifact byte count mismatch for {relative}: "
                    f"expected {expected_bytes}, observed {path.stat().st_size}"
                )
        else:
            expected = expected_entry
        observed = sha256_file(path)
        if expected != observed:
            raise ValueError(
                f"derived artifact checksum mismatch for {relative}: "
                f"expected {expected}, observed {observed}"
            )
        verified[relative.name] = observed
    return verified


def verify_sha256_text_bundle(
    registry: InputRegistry,
    *,
    checksum_key: str,
    artifact_keys: tuple[str, ...],
) -> dict[str, str]:
    """Verify an exact ``sha256sum``-style bundle manifest.

    Unlike :func:`verify_bundle_checksums`, this parser accepts only canonical
    lower-case SHA-256 records of the form ``<digest><two spaces><basename>``.
    The manifest must enumerate exactly the registered artifacts: neither an
    undeclared file nor a missing registered file is tolerated.
    """

    checksum_path = registry.path(checksum_key)
    checksum_parent = Path(INPUT_SPECS[checksum_key].path).parent
    expected_names: dict[str, str] = {}
    for artifact_key in artifact_keys:
        relative = Path(INPUT_SPECS[artifact_key].path)
        if relative.parent != checksum_parent or relative.name != relative.as_posix().split("/")[-1]:
            raise ValueError(f"artifact is outside checksum bundle: {artifact_key}")
        if relative.name in expected_names:
            raise ValueError(f"duplicate registered bundle basename: {relative.name}")
        expected_names[relative.name] = artifact_key

    declared: dict[str, str] = {}
    lines = checksum_path.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise ValueError(f"empty SHA-256 manifest: {checksum_key}")
    for line in lines:
        parts = line.split("  ")
        if len(parts) != 2:
            raise ValueError(f"non-canonical SHA-256 manifest line: {line!r}")
        digest, filename = parts
        if (
            len(digest) != 64
            or digest != digest.lower()
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError(f"invalid SHA-256 digest in {checksum_key}: {digest!r}")
        if not filename or Path(filename).name != filename or filename in {".", ".."}:
            raise ValueError(f"invalid bundle filename in {checksum_key}: {filename!r}")
        if filename in declared:
            raise ValueError(f"duplicate bundle filename in {checksum_key}: {filename}")
        declared[filename] = digest

    if set(declared) != set(expected_names):
        missing = sorted(set(expected_names) - set(declared))
        extra = sorted(set(declared) - set(expected_names))
        raise ValueError(
            f"SHA-256 bundle membership changed for {checksum_key}; "
            f"missing={missing}, extra={extra}"
        )

    verified: dict[str, str] = {}
    for filename in sorted(expected_names):
        path = registry.path(expected_names[filename])
        observed = sha256_file(path)
        if observed != declared[filename]:
            raise ValueError(
                f"derived artifact checksum mismatch for {filename}: "
                f"expected {declared[filename]}, observed {observed}"
            )
        verified[filename] = observed
    return verified


def two_way_center(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    return (
        matrix
        - matrix.mean(axis=0, keepdims=True)
        - matrix.mean(axis=1, keepdims=True)
        + matrix.mean()
    )


def correlation_matrix(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2 or min(matrix.shape) < 2 or not np.isfinite(matrix).all():
        raise ValueError("correlation input must be a finite two-dimensional matrix")
    result = np.corrcoef(matrix, rowvar=False)
    result = np.clip((result + result.T) / 2.0, -1.0, 1.0)
    np.fill_diagonal(result, 1.0)
    return result


def pearson(first: np.ndarray, second: np.ndarray) -> float:
    return float(np.corrcoef(first, second)[0, 1])


def spectrum_record(matrix: np.ndarray) -> dict[str, Any]:
    correlation = correlation_matrix(matrix)
    eigenvalues = np.maximum(np.linalg.eigvalsh(correlation), 0.0)[::-1]
    probabilities = eigenvalues / eigenvalues.sum()
    positive = probabilities > 0
    upper = correlation[np.triu_indices(len(correlation), k=1)]
    cumulative = np.cumsum(probabilities)
    return {
        "participation_ratio_dimension": float(
            eigenvalues.sum() ** 2 / np.square(eigenvalues).sum()
        ),
        "entropy_effective_dimension": float(
            np.exp(-np.sum(probabilities[positive] * np.log(probabilities[positive])))
        ),
        "pc1_variance_fraction": float(probabilities[0]),
        "components_for_90_percent": int(np.searchsorted(cumulative, 0.90) + 1),
        "mean_target_correlation": float(upper.mean()),
        "mean_absolute_target_correlation": float(np.abs(upper).mean()),
        "mean_squared_target_correlation": float(np.square(upper).mean()),
        "negative_edge_fraction": float(np.mean(upper < 0)),
        "pr_identity_from_mean_squared_correlation": float(
            len(correlation) / (1.0 + (len(correlation) - 1) * np.square(upper).mean())
        ),
    }


def covariance_spectrum_record(matrix: np.ndarray) -> dict[str, Any]:
    matrix = np.asarray(matrix, dtype=np.float64)
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    covariance = centered.T @ centered / (len(centered) - 1)
    eigenvalues = np.maximum(np.linalg.eigvalsh(covariance), 0.0)[::-1]
    probabilities = eigenvalues / eigenvalues.sum()
    positive = probabilities > 0
    target_sd = np.sqrt(np.diag(covariance))
    return {
        "participation_ratio_dimension": float(
            eigenvalues.sum() ** 2 / np.square(eigenvalues).sum()
        ),
        "entropy_effective_dimension": float(
            np.exp(-np.sum(probabilities[positive] * np.log(probabilities[positive])))
        ),
        "pc1_variance_fraction": float(probabilities[0]),
        "components_for_90_percent": int(
            np.searchsorted(np.cumsum(probabilities), 0.90) + 1
        ),
        "target_standard_deviation_cv": float(target_sd.std(ddof=0) / target_sd.mean()),
    }


def common_axis_record(matrix: np.ndarray) -> dict[str, Any]:
    matrix = np.asarray(matrix, dtype=np.float64)
    standardized = (matrix - matrix.mean(axis=0)) / matrix.std(axis=0, ddof=1)
    correlation = correlation_matrix(matrix)
    eigenvalues, eigenvectors = np.linalg.eigh(correlation)
    loading = eigenvectors[:, -1]
    if loading.sum() < 0:
        loading *= -1
    uniform = np.ones_like(loading) / np.sqrt(len(loading))
    scores = standardized @ loading
    return {
        "positive_loadings": int(np.sum(loading > 0)),
        "targets": int(len(loading)),
        "cosine_similarity_with_uniform_axis": float(np.dot(loading, uniform)),
        "minimum_loading": float(loading.min()),
        "maximum_loading": float(loading.max()),
        "pc1_score_correlation_with_absolute_score_row_mean": pearson(
            scores, matrix.mean(axis=1)
        ),
        "pc1_score_correlation_with_column_standardized_row_mean": pearson(
            scores, standardized.mean(axis=1)
        ),
        "interpretation": (
            "The two row-mean correlations are reported separately so the nearly "
            "algebraic relationship to the column-standardized row mean is explicit."
        ),
    }


def large_panel_record(
    matrix: np.ndarray,
    *,
    target_order: list[str],
    input_rows: int,
    missing_cells: int,
    rows_removed_for_missingness: int,
    positive_cells: int,
    preprocessing: str,
) -> dict[str, Any]:
    if matrix.shape[1] != len(target_order):
        raise ValueError("target order does not match matrix width")
    residual = two_way_center(matrix)
    return {
        "support": {
            "source_rows": int(input_rows),
            "analysis_rows": int(matrix.shape[0]),
            "targets": int(matrix.shape[1]),
            "source_missing_cells": int(missing_cells),
            "rows_removed_for_missingness": int(rows_removed_for_missingness),
            "strictly_positive_source_cells": int(positive_cells),
            "target_order": target_order,
        },
        "preprocessing": preprocessing,
        "column_standardized_surface": spectrum_record(matrix),
        "two_way_centered_residual_surface": spectrum_record(residual),
        "covariance_sensitivity": {
            "unscaled_surface": covariance_spectrum_record(matrix),
            "unscaled_two_way_centered_residual": covariance_spectrum_record(residual),
            "primary_estimand": (
                "The correlation spectrum is primary because the audit concerns target "
                "dependence after putting target columns on a common variance scale; "
                "covariance results show the dependence on that choice."
            ),
        },
        "shared_axis": common_axis_record(matrix),
    }


def docking44_missing_data_sensitivity(clipped: pd.DataFrame) -> dict[str, Any]:
    if list(clipped.columns) != list(DOCKING44_TARGETS):
        raise ValueError("Docking-44 missingness audit received the wrong target order")
    missing_per_target = clipped.isna().sum().sort_values(ascending=False)
    complete_mask = ~clipped.isna().any(axis=1)
    variants = {
        "target_mean_imputation": clipped.fillna(clipped.mean()),
        "target_median_imputation": clipped.fillna(clipped.median()),
        "complete_case": clipped.loc[complete_mask],
        "mean_imputed_restricted_to_complete_rows": clipped.fillna(clipped.mean()).loc[
            complete_mask
        ],
    }
    records: dict[str, Any] = {}
    for name, frame in variants.items():
        matrix = frame.to_numpy(dtype=np.float64)
        records[name] = {
            "ligands": int(len(matrix)),
            "raw_participation_ratio_dimension": spectrum_record(matrix)[
                "participation_ratio_dimension"
            ],
            "residual_participation_ratio_dimension": spectrum_record(
                two_way_center(matrix)
            )["participation_ratio_dimension"],
        }
    return {
        "source_missing_cells": int(clipped.isna().to_numpy().sum()),
        "rows_with_any_missing": int((~complete_mask).sum()),
        "most_incomplete_target": str(missing_per_target.index[0]),
        "most_incomplete_target_missing_cells": int(missing_per_target.iloc[0]),
        "variants": records,
        "interpretation": (
            "Complete-case deletion changes chemical support and is therefore not a "
            "like-for-like imputation comparison; the restricted mean-imputed row is "
            "included to make that identity explicit."
        ),
    }


def load_large_panels(registry: InputRegistry) -> dict[str, Any]:
    dock = registry.read_csv("docking44", usecols=list(DOCKING44_TARGETS))
    dock = dock.apply(pd.to_numeric, errors="coerce")
    dock_missing = int(dock.isna().to_numpy().sum())
    dock_positive = int((dock.to_numpy(dtype=np.float64) > 0).sum())
    dock = dock.clip(upper=0.0)
    dock_missing_sensitivity = docking44_missing_data_sensitivity(dock)
    dock_matrix = dock.fillna(dock.mean()).to_numpy(dtype=np.float64)
    if dock_matrix.shape != (12_651, 44):
        raise ValueError(f"unexpected Docking-44 shape: {dock_matrix.shape}")

    dockstring = registry.read_csv("dockstring", sep="\t")
    ds_targets = [
        column for column in dockstring.columns if column not in {"inchikey", "smiles"}
    ]
    ds_numeric = dockstring[ds_targets].apply(pd.to_numeric, errors="coerce")
    ds_missing = int(ds_numeric.isna().to_numpy().sum())
    complete = ~ds_numeric.isna().any(axis=1)
    ds_matrix = ds_numeric.loc[complete].to_numpy(dtype=np.float64)
    ds_positive = int((ds_matrix > 0).sum())
    unclipped_sensitivity = {
        "column_standardized_surface": spectrum_record(ds_matrix),
        "two_way_centered_residual_surface": spectrum_record(two_way_center(ds_matrix)),
    }
    np.minimum(ds_matrix, 0.0, out=ds_matrix)
    if ds_matrix.shape != (260_060, 58):
        raise ValueError(f"unexpected complete DOCKSTRING shape: {ds_matrix.shape}")

    primary = large_panel_record(
        ds_matrix,
        target_order=ds_targets,
        input_rows=len(dockstring),
        missing_cells=ds_missing,
        rows_removed_for_missingness=int((~complete).sum()),
        positive_cells=ds_positive,
        preprocessing=(
            "complete rows; strictly positive scores clipped at zero following the "
            "released analysis contract; exact count and fraction are reported below"
        ),
    )
    primary["positive_score_clipping_sensitivity"] = {
        "positive_cells": ds_positive,
        "positive_fraction_of_complete_cells": float(ds_positive / ds_matrix.size),
        "positive_scores_retained": unclipped_sensitivity,
        "positive_scores_clipped": {
            "column_standardized_surface": primary["column_standardized_surface"],
            "two_way_centered_residual_surface": primary[
                "two_way_centered_residual_surface"
            ],
        },
        "claim_boundary": (
            "This is a matrix-geometry preprocessing sensitivity, not evidence that "
            "positive Vina energies are calibrated physical affinities."
        ),
    }

    docking_record = large_panel_record(
            dock_matrix,
            target_order=list(DOCKING44_TARGETS),
            input_rows=len(dock),
            missing_cells=dock_missing,
            rows_removed_for_missingness=0,
            positive_cells=dock_positive,
            preprocessing=(
                "scores clipped at zero; missing scores imputed by the corresponding "
                "target mean; correlations standardize target columns"
            ),
        )
    docking_record["missing_data_sensitivity"] = dock_missing_sensitivity
    return {
        "Docking-44": docking_record,
        "DOCKSTRING-58": primary,
    }


def one_row(frame: pd.DataFrame, **selectors: Any) -> dict[str, Any]:
    selected = frame
    for column, value in selectors.items():
        selected = selected.loc[selected[column].eq(value)]
    if len(selected) != 1:
        raise ValueError(f"expected one row for {selectors}, observed {len(selected)}")
    return json_ready(selected.iloc[0].to_dict())


def one_record(records: list[dict[str, Any]], **selectors: Any) -> dict[str, Any]:
    selected = [
        record
        for record in records
        if all(record.get(key) == value for key, value in selectors.items())
    ]
    if len(selected) != 1:
        raise ValueError(f"expected one record for {selectors}, observed {len(selected)}")
    return selected[0]


def compact_distribution(record: dict[str, Any]) -> dict[str, Any]:
    return {
        key: record[key]
        for key in ("n", "mean", "median", "interval_95")
        if key in record
    }


def revision_evidence(registry: InputRegistry) -> dict[str, Any]:
    registry.path("revision_producer")
    registry.path("revision_test")
    registry.path("revision_loader_helper")
    registry.path("revision_loader_helper_test")
    verified = verify_bundle_checksums(
        registry,
        checksum_key="revision_checksums",
        artifact_keys=(
            "revision_metadata",
            "mw_threshold_summary",
            "mw_continuous_trends",
            "recovery_edge_summary",
            "recovery_decision_summary",
        ),
    )
    metadata = registry.read_json("revision_metadata")
    threshold = registry.read_csv("mw_threshold_summary")
    trends = registry.read_csv("mw_continuous_trends")
    edges = registry.read_csv("recovery_edge_summary")
    decisions = registry.read_csv("recovery_decision_summary")
    expected = {"Docking-44", "DOCKSTRING-58"}
    for frame in (threshold, trends, edges, decisions):
        if set(frame.dataset) != expected:
            raise ValueError("revision table has an unexpected dataset set")

    support_dependence: dict[str, Any] = {}
    pilot_recovery: dict[str, Any] = {}
    for dataset in sorted(expected):
        support_dependence[dataset] = {
            "threshold_sweep": {
                transform: one_row(
                    threshold, dataset=dataset, transformation=transform
                )
                for transform in ("raw", "row_centered_residual")
            },
            "ten_bin_continuum": {
                transform: one_row(trends, dataset=dataset, transformation=transform)
                for transform in ("raw", "row_centered_residual")
            },
        }
        pilot_recovery[dataset] = {}
        for size in (200, 500):
            edge = one_row(
                edges,
                dataset=dataset,
                calibration_ligands=size,
                reference_scope="row_disjoint_complement_map",
            )
            decision = one_row(
                decisions,
                dataset=dataset,
                calibration_ligands=size,
                panel_targets_k=8,
                reference_scope="row_disjoint_complement_map",
            )
            pilot_recovery[dataset][str(size)] = {
                "map_edges": {
                    key: edge[key]
                    for key in (
                        "repetitions",
                        "reference_scope",
                        "geometry_spearman_mean",
                        "geometry_spearman_q025",
                        "geometry_spearman_q975",
                        "edge_sign_agreement_mean",
                        "edge_sign_agreement_q025",
                        "edge_sign_agreement_q975",
                        "top_positive_10_precision_mean",
                        "top_positive_10_precision_q025",
                        "top_positive_10_precision_q975",
                        "top_negative_10_precision_mean",
                        "top_negative_10_precision_q025",
                        "top_negative_10_precision_q975",
                    )
                },
                "eight_target_panel_decision": {
                    key: decision[key]
                    for key in (
                        "repetitions",
                        "reference_scope",
                        "cluster_adjusted_rand_index_mean",
                        "cluster_adjusted_rand_index_q025",
                        "cluster_adjusted_rand_index_q975",
                        "sample_vs_fixed_full_source_panel_medoid_overlap_mean",
                        "sample_vs_fixed_full_source_panel_medoid_overlap_q025",
                        "sample_vs_fixed_full_source_panel_medoid_overlap_q975",
                        "sample_minus_fixed_full_source_panel_mean_nearest_distance_on_reference_map_mean",
                        "sample_minus_fixed_full_source_panel_mean_nearest_distance_on_reference_map_q025",
                        "sample_minus_fixed_full_source_panel_mean_nearest_distance_on_reference_map_q975",
                        "fraction_random_panels_with_mean_coverage_no_worse_than_sample_mean",
                        "fraction_random_panels_with_mean_coverage_no_worse_than_sample_q025",
                        "fraction_random_panels_with_mean_coverage_no_worse_than_sample_q975",
                    )
                },
            }
    return {
        "analysis_status": metadata["analysis_status"],
        "molecular_weight_chronology": metadata["molecular_weight_support"][
            "chronology"
        ],
        "support_dependence": support_dependence,
        "pilot_recovery": pilot_recovery,
        "verified_output_sha256": verified,
        "claim_boundary": metadata["claim_boundary"],
    }


def experimental_source_audits(registry: InputRegistry) -> dict[str, Any]:
    davis = registry.read_csv("davis", sep="\t")
    required = {
        "drug_name", "protein", "affinity", "compound_iso_smiles", "target_sequence", "y"
    }
    if set(davis.columns) != required or davis.isna().any().any():
        raise ValueError("DAVIS-Complete schema or completeness changed")
    if davis.shape != (31_824, 6) or davis.duplicated(["drug_name", "protein"]).any():
        raise ValueError("DAVIS-Complete rectangular support changed")

    pkis2 = registry.read_excel("pkis2", sheet_name="Table 4 - PKIS2 %Inh")
    metadata_columns = ["Regno", "Compound", "Chemotype", "Smiles"]
    pkis2 = pkis2.dropna(subset=metadata_columns).copy()
    if pkis2.shape[0] != 645 or pkis2.shape[1] != 413:
        raise ValueError("PKIS2 public workbook support changed")
    selected = pkis2[list(PKIS2_TARGET_MAP.values())].apply(
        pd.to_numeric, errors="coerce"
    )
    if selected.isna().any().any() or ((selected < 0) | (selected > 100)).any().any():
        raise ValueError("PKIS2 public 645 x 21 block is no longer dense and bounded")
    return {
        "DAVIS": {
            "rows": int(len(davis)),
            "ligands": int(davis.drug_name.nunique()),
            "targets": int(davis.protein.nunique()),
            "complete_ligand_target_grid": bool(
                len(davis) == davis.drug_name.nunique() * davis.protein.nunique()
            ),
        },
        "PKIS2": {
            "populated_compound_rows": int(len(pkis2)),
            "all_assay_columns": 406,
            "public_core_targets": len(PKIS2_TARGET_MAP),
            "public_core_cells": int(selected.size),
            "public_core_missing_cells": int(selected.isna().to_numpy().sum()),
        },
    }


def cross_panel_overlap_evidence(registry: InputRegistry) -> dict[str, Any]:
    registry.path("cross_panel_overlap_producer")
    registry.path("cross_panel_overlap_test")
    identity_path = registry.path("dockstring_identity_contract")
    registry.path("cross_panel_target_mapping")
    registry.path("target_families")
    provenance = registry.read_json("dockstring_identity_provenance")
    summary = registry.read_json("cross_panel_overlap")

    if provenance["source"]["sha256"] != registry.opened_record("dockstring")["sha256"]:
        raise ValueError("DOCKSTRING identity provenance does not match the row-level input")
    if provenance["output"]["sha256"] != sha256_file(identity_path):
        raise ValueError("DOCKSTRING identity-contract checksum changed")
    source_contract = {
        "docking44": "docking44",
        "dockstring_identity_contract": "dockstring_identity_contract",
        "target_mapping": "cross_panel_target_mapping",
        "target_families": "target_families",
    }
    for summary_key, registry_key in source_contract.items():
        source = summary["sources"][summary_key]
        expected_path = INPUT_SPECS[registry_key].path
        expected_hash = registry.opened_record(registry_key)["sha256"]
        if source["path"] != expected_path or source["sha256"] != expected_hash:
            raise ValueError(f"cross-panel overlap source drifted: {summary_key}")
    if summary["chemical_overlap"]["shared_full_standard_inchikeys"] != 582:
        raise ValueError("cross-panel full-InChIKey overlap changed")
    if summary["chemical_overlap"]["shared_connectivity_blocks"] != 804:
        raise ValueError("cross-panel connectivity overlap changed")
    if summary["target_overlap"]["mapped_target_identities"] != 9:
        raise ValueError("cross-panel target-identity mapping changed")
    if summary["target_overlap"]["same_receptor_structure_used"] != 3:
        raise ValueError("cross-panel receptor-structure overlap changed")
    # Paths and hashes were validated above; retain the scientific result only.
    return {
        "support": summary["support"],
        "chemical_overlap": summary["chemical_overlap"],
        "target_overlap": summary["target_overlap"],
        "interpretation": summary["interpretation"],
    }


def experimental_map_evidence(registry: InputRegistry) -> dict[str, Any]:
    registry.path("experimental_producer")
    registry.path("experimental_test")
    registry.path("davis_loader")
    registry.path("davis_loader_test")
    registry.path("pkis2_loader")
    registry.path("pkis2_loader_test")
    verified = verify_bundle_checksums(
        registry,
        checksum_key="experimental_checksums",
        artifact_keys=(
            "experimental_summary",
            "experimental_cross_panel_geometry",
            "experimental_reliability",
            "experimental_edge_stability",
            "same_support_geometry",
            "same_support_bootstrap",
            "same_support_split_half",
        ),
    )
    summary = registry.read_json("experimental_summary")
    if summary.get("secondary_input_sha256", {}) != {}:
        raise ValueError("experimental public-core bundle contains secondary inputs")
    primary_hashes = set(summary.get("primary_input_sha256", {}).values())
    required_hashes = {
        PINNED_SHA256[key]
        for key in ("dockstring", "dockstring_identity_contract", "davis", "pkis2")
    }
    if primary_hashes != required_hashes:
        raise ValueError("experimental public-core bundle has an unexpected input set")

    panels: dict[str, Any] = {}
    for panel_name in ("DAVIS", "PKIS2"):
        panel = summary["panels"][panel_name]
        released_reliability: dict[str, Any] = {}
        released_edge_stability: dict[str, Any] = {}
        primary_bootstrap: dict[str, Any] = {}
        primary_split_half: dict[str, Any] = {}
        for unit in ("ligand", "butina_cluster"):
            reliability = one_record(
                panel["map_reliability"],
                support="released_full_panel",
                transform="two_way_centered",
                estimator="empirical",
                split_unit=unit,
            )
            released_reliability[unit] = {
                "ligands": reliability["ligands"],
                "valid_repeats": reliability["valid_repeats"],
                "split_map_spearman": compact_distribution(
                    reliability["split_map_spearman"]
                ),
                "edge_sign_agreement_fraction": compact_distribution(
                    reliability["edge_sign_agreement_fraction"]
                ),
                "top_decile_precision_between_halves": compact_distribution(
                    reliability["top_decile_precision_between_halves"]
                ),
                "mean_half_to_full_map_spearman": compact_distribution(
                    reliability["mean_half_to_full_map_spearman"]
                ),
            }
            stability = one_record(
                panel["edge_stability"],
                support="released_full_panel",
                transform="two_way_centered",
                bootstrap_unit=unit,
            )
            released_edge_stability[unit] = {
                "ligands": stability["ligands"],
                "valid_repeats": stability["valid_repeats"],
                "top_decile_edges": stability["top_decile_edges"],
                "edges_with_sign_match_at_least_0_90": stability[
                    "edges_with_sign_match_at_least_0_90"
                ],
                "full_top_decile_edges_with_inclusion_at_least_0_80": stability[
                    "full_top_decile_edges_with_inclusion_at_least_0_80"
                ],
                "bootstrap_map_to_full_spearman": compact_distribution(
                    stability["bootstrap_map_to_full_spearman"]
                ),
                "bootstrap_top_decile_precision_against_full": compact_distribution(
                    stability["bootstrap_top_decile_precision_against_full"]
                ),
            }
            bootstrap = one_record(
                panel["same_support_bootstrap"],
                support="informative_exact_matches_primary",
                transform="two_way_centered",
                bootstrap_unit=unit,
            )
            primary_bootstrap[unit] = {
                "ligands": bootstrap["ligands"],
                "valid_repeats": bootstrap["valid_repeats"],
                "point_estimate": bootstrap["point_estimate"],
                "same_support_spearman": compact_distribution(
                    bootstrap["same_support_spearman"]
                ),
                "same_minus_broad": compact_distribution(
                    bootstrap["same_minus_broad"]
                ),
            }
            split_half = one_record(
                panel["same_support_reliability"],
                support="informative_exact_matches_primary",
                transform="two_way_centered",
                split_unit=unit,
            )
            primary_split_half[unit] = {
                "ligands": split_half["ligands"],
                "valid_repeats": split_half["valid_repeats"],
                "plugin_same_support_spearman": split_half[
                    "plugin_same_support_spearman"
                ],
                "docking_split_map_spearman": compact_distribution(
                    split_half["docking_split_map_spearman"]
                ),
                "experimental_split_map_spearman": compact_distribution(
                    split_half["experimental_split_map_spearman"]
                ),
                "cross_half_cross_modal_spearman": compact_distribution(
                    split_half["cross_half_cross_modal_spearman"]
                ),
                "same_minus_cross_half": compact_distribution(
                    split_half["same_minus_cross_half"]
                ),
                "median_cross_half_spearman_divided_by_experimental_self_repeatability": split_half[
                    "median_cross_half_spearman_divided_by_experimental_self_repeatability"
                ],
            }
        primary_geometry = {
            transform: one_record(
                panel["same_support_geometry"],
                support="informative_exact_matches_primary",
                transform=transform,
            )
            for transform in ("raw", "two_way_centered")
        }
        panels[panel_name] = {
            "released_panel_ligands": panel["released_panel_ligands"],
            "released_panel_informative_profiles": panel[
                "released_panel_informative_profiles"
            ],
            "exact_matches": panel["exact_matches"],
            "primary_same_support_informative_ligands": panel[
                "primary_same_support_informative_ligands"
            ],
            "released_map_reliability": released_reliability,
            "released_edge_stability": released_edge_stability,
            "primary_same_support_geometry": primary_geometry,
            "primary_same_support_bootstrap": primary_bootstrap,
            "primary_same_support_split_half": primary_split_half,
        }
    cross_panel_geometry = [
        {
            key: record[key]
            for key in (
                "panel_a",
                "panel_b",
                "transform",
                "support_a",
                "support_b",
                "targets",
                "target_pairs",
                "edge_spearman",
                "edge_sign_agreement_fraction",
                "edge_sign_agreement_null_median",
                "top_decile_precision",
                "top_decile_precision_null_median",
                "target_label_qap_p_two_sided",
                "permutations",
            )
        }
        for record in summary["cross_panel_experimental_geometry"]
    ]
    return {
        "primary_estimand": summary["primary_estimand"],
        "cross_panel_experimental_geometry": cross_panel_geometry,
        "cross_panel_role": (
            "External experimental-map reproducibility benchmark across two assay "
            "panels with different chemical supports and measurement technologies."
        ),
        "panels": panels,
        "verified_output_sha256": verified,
        "interpretation_boundary": (
            "Reliability limits how strongly docking--experiment map agreement can be "
            "interpreted. Same-support geometry is descriptive of exact matched ligands "
            "and does not establish target retrieval on unmeasured proteins."
        ),
    }


def residual_null_evidence(registry: InputRegistry) -> dict[str, Any]:
    registry.path("residual_null_producer")
    registry.path("residual_null_test")
    verified = verify_bundle_checksums(
        registry,
        checksum_key="residual_null_checksums",
        artifact_keys=(
            "residual_null_summary",
            "residual_null_conditional_repeats",
            "residual_null_bin_contract",
            "residual_null_readme",
        ),
    )
    payload = registry.read_json("residual_null_summary")
    conditional_repeats = registry.read_csv("residual_null_conditional_repeats")
    bin_contract = registry.read_csv("residual_null_bin_contract")
    expected_inputs = {
        "docking44": (INPUT_SPECS["docking44"].path, PINNED_SHA256["docking44"]),
        "dockstring58": (
            INPUT_SPECS["dockstring"].path,
            PINNED_SHA256["dockstring"],
        ),
    }
    if set(payload.get("inputs", {})) != set(expected_inputs):
        raise ValueError("standalone residual-null audit has an unexpected input set")
    for key, (path, digest) in expected_inputs.items():
        record = payload["inputs"][key]
        if record.get("path") != path or record.get("sha256") != digest:
            raise ValueError(f"standalone residual-null input drifted: {key}")

    compact: dict[str, Any] = {}
    expected_nulls = {
        "additive_gaussian",
        "empirical_residual_column_permutation",
        "row_norm_preserving_random_direction",
        "molecular_weight_conditional_permutation",
    }
    for dataset in ("docking44", "dockstring58"):
        record = payload["datasets"][dataset]
        if set(record["nulls"]) != expected_nulls:
            raise ValueError(f"standalone residual-null families drifted: {dataset}")
        conventional = {
            key: value
            for key, value in record["nulls"].items()
            if key != "molecular_weight_conditional_permutation"
        }
        conditional = record["nulls"]["molecular_weight_conditional_permutation"]
        expected_conditional = {f"mw_{bins}_bins" for bins in (1, 5, 10, 20)}
        if set(conditional) != expected_conditional:
            raise ValueError(f"molecular-weight null ladder drifted: {dataset}")
        support_hashes = {
            null["support_selection"]["support_index_sha256"]
            for null in (*conventional.values(), *conditional.values())
        }
        if len(support_hashes) != 1:
            raise ValueError(f"residual nulls do not share one support: {dataset}")
        compact_nulls: dict[str, Any] = {}
        for name, null in conventional.items():
            compact_nulls[name] = {
                "null_residual_pr_median": null["null_residual"]["median"],
                "null_residual_pr_interval_95": null["null_residual"]["interval_95"],
                "empirical_lower_tail_probability": null[
                    "empirical_lower_tail_p_for_residual_pr"
                ],
                "model": null["model"],
            }
            for key in ("preserves_at_permutation_step", "preserves_exactly"):
                if key in null:
                    compact_nulls[name][key] = null[key]
            if "does_not_preserve" in null:
                compact_nulls[name]["does_not_preserve"] = null[
                    "does_not_preserve"
                ]
        compact_conditional: dict[str, Any] = {}
        for name, null in conditional.items():
            distributions = null["null_distributions_by_bin_design"]
            compact_conditional[name] = {
                "requested_mw_bins": null["molecular_weight"]["requested_bins"],
                "actual_mw_bins": null["molecular_weight"]["actual_bins"],
                "mw_bin_sizes": null["molecular_weight"]["bin_sizes"],
                "observed_residual_pr": null["observed"]["participation_ratio"],
                "mw_conditioned_residual_pr": distributions[
                    "molecular_weight_quantile"
                ]["participation_ratio"],
                "mw_conditioned_map_agreement": distributions[
                    "molecular_weight_quantile"
                ]["spearman_to_observed_map"],
                "mw_conditioned_sign_agreement": distributions[
                    "molecular_weight_quantile"
                ]["edge_sign_agreement_to_observed_map"],
                "comparisons": null["comparisons"],
                "empirical_lower_tail_probability": null[
                    "empirical_lower_tail_p_for_residual_pr"
                ],
                "model": null["model"],
                "preserves_at_permutation_step": null[
                    "preserves_at_permutation_step"
                ],
                "does_not_preserve": null["does_not_preserve"],
            }
            if "size_matched_random_partition" in distributions:
                compact_conditional[name]["size_matched_random_partition_pr"] = (
                    distributions["size_matched_random_partition"][
                        "participation_ratio"
                    ]
                )
        compact_nulls["molecular_weight_conditional_permutation"] = (
            compact_conditional
        )
        compact[dataset] = {
            "full_processed_shape": record["full_processed_shape"],
            "support_selection": record["support_selection"],
            "observed_residual_pr": record["observed_residual_pr"],
            "nulls": compact_nulls,
        }
    expected_repeat_rows = 2 * 500 * (1 + 2 + 2 + 2)
    if len(conditional_repeats) != expected_repeat_rows:
        raise ValueError("molecular-weight null replicate coverage changed")
    if set(conditional_repeats.dataset) != {"docking44", "dockstring58"}:
        raise ValueError("molecular-weight null dataset coverage changed")
    if set(conditional_repeats.requested_mw_bins) != {1, 5, 10, 20}:
        raise ValueError("molecular-weight null bin-count coverage changed")
    if len(bin_contract) != 72:
        raise ValueError("molecular-weight bin contract coverage changed")
    return {
        "configuration": payload["configuration"],
        "datasets": compact,
        "conditional_null_replicates": int(len(conditional_repeats)),
        "conditional_bin_contract_rows": int(len(bin_contract)),
        "verified_output_sha256": verified,
        "claim_boundary": payload["claim_boundary"],
    }


def docking44_missing_data_evidence(registry: InputRegistry) -> dict[str, Any]:
    registry.path("missing_data_producer")
    registry.path("missing_data_test")
    verified = verify_bundle_checksums(
        registry,
        checksum_key="missing_data_checksums",
        artifact_keys=(
            "missing_data_summary",
            "missing_data_by_target",
            "missing_data_spectral",
            "missing_data_equal_n_controls",
            "missing_data_chemistry_effects",
            "missing_data_chemistry_groups",
            "missing_data_readme",
        ),
    )
    summary = registry.read_json("missing_data_summary")
    missingness = registry.read_csv("missing_data_by_target")
    chemistry_effects = registry.read_csv("missing_data_chemistry_effects")
    chemistry_groups = registry.read_csv("missing_data_chemistry_groups")
    if summary.get("schema_version") != "1.0.0":
        raise ValueError("Docking-44 missing-data schema changed")
    if summary.get("input") != {
        "path": INPUT_SPECS["docking44"].path,
        "sha256": PINNED_SHA256["docking44"],
    }:
        raise ValueError("Docking-44 missing-data input contract drifted")
    matrix = summary["matrix"]
    if (matrix["rows"], matrix["targets"], matrix["missing_cells"]) != (
        12_651,
        44,
        8_159,
    ):
        raise ValueError("Docking-44 missing-data support changed")
    if set(missingness.target) != set(DOCKING44_TARGETS):
        raise ValueError("Docking-44 target-wise missingness table changed")
    most_incomplete = missingness.sort_values(
        ["missing_cells", "target"], ascending=[False, True]
    ).iloc[0]
    spectral = summary["spectral_sensitivity"]
    aliases = {
        "target_mean_imputation": "target_mean_imputation_full_support",
        "target_median_imputation": "target_median_imputation_full_support",
        "complete_case": "complete_rows_observed_scores",
        "mean_imputed_restricted_to_complete_rows": (
            "target_mean_imputed_restricted_to_complete_rows"
        ),
        "observed_cell_wls_zero_residual_completion": (
            "observed_cell_wls_zero_residual_completion"
        ),
    }
    variants = {alias: spectral[source] for alias, source in aliases.items()}
    if abs(
        variants["complete_case"]["residual_pr"]
        - variants["mean_imputed_restricted_to_complete_rows"]["residual_pr"]
    ) > 1e-12:
        raise ValueError("complete-row identity check failed")
    return {
        "source_missing_cells": matrix["missing_cells"],
        "missing_cell_fraction": matrix["missing_cell_fraction"],
        "rows_with_any_missing": matrix["rows_with_any_missing_score"],
        "complete_rows": matrix["complete_rows"],
        "most_incomplete_target": str(most_incomplete.target),
        "most_incomplete_target_missing_cells": int(most_incomplete.missing_cells),
        "variants": variants,
        "observed_cell_wls_diagnostics": summary["observed_cell_wls_diagnostics"],
        "complete_row_identity_check": summary["complete_row_identity_check"],
        "equal_n_random_support_control": summary[
            "equal_n_random_support_control"
        ],
        "chemistry_effect_size_direction": summary["chemistry"][
            "effect_size_direction"
        ],
        "chemistry_effect_sizes": json_ready(chemistry_effects.to_dict("records")),
        "chemistry_group_summaries": json_ready(chemistry_groups.to_dict("records")),
        "target_wise_missingness": json_ready(missingness.to_dict("records")),
        "verified_output_sha256": verified,
        "claim_boundary": summary["claim_boundary"],
    }


def chemical_domain_control_evidence(registry: InputRegistry) -> dict[str, Any]:
    registry.path("chemical_domain_producer")
    registry.path("chemical_domain_test")
    verified = verify_bundle_checksums(
        registry,
        checksum_key="chemical_domain_checksums",
        artifact_keys=(
            "chemical_domain_summary",
            "chemical_domain_observed",
            "chemical_domain_seed_contrasts",
            "chemical_domain_seed_summary",
            "chemical_domain_global_controls",
            "chemical_domain_global_summary",
            "chemical_domain_within_controls",
            "chemical_domain_within_summary",
            "chemical_domain_boundary_table",
            "chemical_domain_readme",
        ),
    )
    summary = registry.read_json("chemical_domain_summary")
    if summary.get("schema_version") != "1.0.0":
        raise ValueError("chemical-domain control schema changed")
    expected_inputs = {
        "docking44": {
            "path": INPUT_SPECS["docking44"].path,
            "sha256": PINNED_SHA256["docking44"],
        },
        "dockstring": {
            "path": INPUT_SPECS["dockstring"].path,
            "sha256": PINNED_SHA256["dockstring"],
        },
    }
    if summary.get("inputs") != expected_inputs:
        raise ValueError("chemical-domain control inputs drifted")
    key_results = summary["key_boundary_results"]
    expected_pairs = {
        (dataset, transform)
        for dataset in ("Docking-44", "DOCKSTRING-58")
        for transform in ("raw", "row_centered_residual")
    }
    observed_pairs = {
        (record["dataset"], record["transformation"]) for record in key_results
    }
    if observed_pairs != expected_pairs:
        raise ValueError("chemical-domain key-result coverage changed")
    if not all(
        record["observed_below_global_group_disjoint_q025"]
        and record["observed_below_both_within_band_group_disjoint_q025"]
        for record in key_results
    ):
        raise ValueError("chemical-domain separation from controls changed")
    return {
        "analysis_status": summary["analysis_status"],
        "configuration": summary["configuration"],
        "datasets": summary["datasets"],
        "control_estimands": summary["control_estimands"],
        "dockstring_seed_summary": summary["dockstring_seed_summary"],
        "key_boundary_results": key_results,
        "support_fingerprints": summary["support_fingerprints"],
        "verified_output_sha256": verified,
        "claim_boundary": summary["claim_boundary"],
    }


def descriptor_domain_evidence(registry: InputRegistry) -> dict[str, Any]:
    """Import the strict-public, post-hoc descriptor-domain sensitivity."""

    registry.path("descriptor_domain_producer")
    registry.path("descriptor_domain_test")
    verified = verify_bundle_checksums(
        registry,
        checksum_key="descriptor_domain_checksums",
        artifact_keys=(
            "descriptor_domain_summary",
            "descriptor_domain_extremes",
            "descriptor_domain_correlations",
            "descriptor_domain_family_summary",
            "descriptor_domain_random_controls",
            "descriptor_domain_random_summary",
            "descriptor_domain_readme",
        ),
    )
    summary = registry.read_json("descriptor_domain_summary")
    extremes = registry.read_csv("descriptor_domain_extremes")
    correlations = registry.read_csv("descriptor_domain_correlations")
    family_summary = registry.read_csv("descriptor_domain_family_summary")
    random_controls = registry.read_csv("descriptor_domain_random_controls")
    random_summary = registry.read_csv("descriptor_domain_random_summary")
    if summary.get("schema_version") != "1.0.0":
        raise ValueError("descriptor-domain schema changed")
    expected_inputs = {
        "docking44": {
            "path": INPUT_SPECS["docking44"].path,
            "sha256": PINNED_SHA256["docking44"],
        },
        "dockstring": {
            "path": INPUT_SPECS["dockstring"].path,
            "sha256": PINNED_SHA256["dockstring"],
        },
    }
    if summary.get("inputs") != expected_inputs:
        raise ValueError("descriptor-domain public inputs drifted")

    datasets = {"Docking-44", "DOCKSTRING-58"}
    descriptors = set(summary["configuration"]["descriptor_names"])
    expected_extremes = {
        (dataset, descriptor, transform)
        for dataset in datasets
        for descriptor in descriptors
        for transform in ("raw", "row_centered_residual")
    }
    observed_extremes = set(
        zip(extremes.dataset, extremes.descriptor, extremes.transformation)
    )
    if observed_extremes != expected_extremes or len(extremes) != 28:
        raise ValueError("descriptor-domain contrast coverage changed")
    if len(correlations) != 42 or set(correlations.dataset) != datasets:
        raise ValueError("descriptor correlation coverage changed")
    expected_families = {
        (dataset, family)
        for dataset in datasets
        for family in ("size_related", "other_coarse")
    }
    if set(zip(family_summary.dataset, family_summary.descriptor_family)) != expected_families:
        raise ValueError("descriptor-family summary coverage changed")

    key_results = summary["key_residual_results"]
    if len(key_results) != 14 or len(random_summary) != 14:
        raise ValueError("descriptor-domain residual result coverage changed")
    expected_descriptor_pairs = {
        (dataset, descriptor)
        for dataset in datasets
        for descriptor in descriptors
    }
    if {
        (record["dataset"], record["descriptor"]) for record in key_results
    } != expected_descriptor_pairs:
        raise ValueError("descriptor-domain key-result identities changed")
    if set(zip(random_summary.dataset, random_summary.descriptor)) != expected_descriptor_pairs:
        raise ValueError("descriptor-control summary identities changed")
    if len(random_controls) != 14 * int(
        summary["configuration"]["random_disjoint_control_repetitions"]
    ):
        raise ValueError("descriptor-control replicate coverage changed")
    if random_summary.observed_below_control_q025.isna().any():
        raise ValueError("descriptor-control diagnostic contains missing values")

    return {
        "analysis_status": summary["analysis_status"],
        "configuration": summary["configuration"],
        "datasets": summary["datasets"],
        "key_residual_results": key_results,
        "family_summary": json_ready(family_summary.to_dict("records")),
        "descriptor_correlations": json_ready(correlations.to_dict("records")),
        "random_disjoint_summary": json_ready(random_summary.to_dict("records")),
        "verified_output_sha256": verified,
        "claim_boundary": summary["claim_boundary"],
    }


def scaffold_holdout_evidence(registry: InputRegistry) -> dict[str, Any]:
    """Import the strict-public chemical-group-held-out recovery sensitivity."""

    registry.path("scaffold_holdout_producer")
    registry.path("scaffold_holdout_test")
    verified = verify_bundle_checksums(
        registry,
        checksum_key="scaffold_holdout_checksums",
        artifact_keys=(
            "scaffold_holdout_summary",
            "scaffold_holdout_metrics",
            "scaffold_holdout_table",
            "scaffold_holdout_random_summary",
            "scaffold_holdout_design_contrasts",
            "scaffold_holdout_fold_diagnostics",
            "scaffold_holdout_by_fold",
            "scaffold_holdout_similarity",
            "scaffold_holdout_similarity_summary",
            "scaffold_holdout_readme",
        ),
    )
    summary = registry.read_json("scaffold_holdout_summary")
    metrics = registry.read_csv("scaffold_holdout_metrics")
    table = registry.read_csv("scaffold_holdout_table")
    random_table = registry.read_csv("scaffold_holdout_random_summary")
    design_contrasts = registry.read_csv("scaffold_holdout_design_contrasts")
    fold_diagnostics = registry.read_csv("scaffold_holdout_fold_diagnostics")
    by_fold = registry.read_csv("scaffold_holdout_by_fold")
    similarity = registry.read_csv("scaffold_holdout_similarity")
    similarity_summary = registry.read_csv("scaffold_holdout_similarity_summary")
    if summary.get("schema_version") != "1.0.0":
        raise ValueError("scaffold-holdout schema changed")
    expected_inputs = {
        "docking44": {
            "path": INPUT_SPECS["docking44"].path,
            "sha256": PINNED_SHA256["docking44"],
        },
        "dockstring": {
            "path": INPUT_SPECS["dockstring"].path,
            "sha256": PINNED_SHA256["dockstring"],
        },
    }
    if summary.get("inputs") != expected_inputs:
        raise ValueError("scaffold-holdout public inputs drifted")
    expected_pairs = {
        (dataset, size)
        for dataset in ("Docking-44", "DOCKSTRING-58")
        for size in (200, 500)
    }
    if set(zip(table.dataset, table.calibration_ligands)) != expected_pairs:
        raise ValueError("scaffold-holdout summary coverage changed")
    if len(metrics) != 1_000 or set(zip(metrics.dataset, metrics.calibration_ligands)) != expected_pairs:
        raise ValueError("scaffold-holdout replicate coverage changed")
    group_metrics = metrics.loc[metrics.holdout_design.eq("chemical_group")]
    if len(group_metrics) != 500 or int(group_metrics.chemical_group_overlap.max()) != 0:
        raise ValueError("scaffold-holdout chemical groups overlap")
    if not (table.repetitions.eq(125).all() and table.maximum_group_overlap.eq(0).all()):
        raise ValueError("scaffold-holdout replication contract changed")
    if set(zip(random_table.dataset, random_table.calibration_ligands)) != expected_pairs:
        raise ValueError("matched random-row summary coverage changed")
    if set(zip(design_contrasts.dataset, design_contrasts.calibration_ligands)) != expected_pairs:
        raise ValueError("holdout-design contrast coverage changed")
    if not design_contrasts.paired_repetitions.eq(125).all():
        raise ValueError("holdout-design paired replication contract changed")
    if len(fold_diagnostics) != 20 or len(by_fold) != 40:
        raise ValueError("chemical-group fold diagnostic coverage changed")
    if len(similarity_summary) != 20 or similarity.empty:
        raise ValueError("held-out nearest-neighbour audit coverage changed")
    chemical_similarity = similarity.loc[
        similarity.holdout_design.eq("chemical_group")
    ]
    if len(chemical_similarity) != 12_651 + 15_000:
        raise ValueError("chemical-group held-out similarity support changed")
    if not set(similarity_summary.columns).issuperset(
        {
            "maximum",
            "median",
            "fraction_at_least_0.7",
            "fraction_at_least_0.9",
            "fraction_tanimoto_equal_one",
        }
    ):
        raise ValueError("held-out similarity summaries lost declared diagnostics")

    return {
        "analysis_status": summary["analysis_status"],
        "configuration": summary["configuration"],
        "datasets": summary["datasets"],
        "key_results": json_ready(table.to_dict("records")),
        "matched_random_row_results": json_ready(random_table.to_dict("records")),
        "paired_design_contrasts": json_ready(design_contrasts.to_dict("records")),
        "fold_diagnostics": json_ready(fold_diagnostics.to_dict("records")),
        "recovery_by_fold": json_ready(by_fold.to_dict("records")),
        "nearest_neighbour_similarity_results": json_ready(
            similarity_summary.to_dict("records")
        ),
        "nearest_neighbour_ligands_audited": int(len(similarity)),
        "verified_output_sha256": verified,
        "claim_boundary": summary["claim_boundary"],
    }


def group_disjoint_panel_reconstruction_evidence(
    registry: InputRegistry,
) -> dict[str, Any]:
    """Import the frozen chemical-group-disjoint k=8 reconstruction audit."""

    registry.path("group_reconstruction_producer")
    registry.path("group_reconstruction_test")
    panel_utility = registry.opened_record("group_reconstruction_panel_utility")
    scaffold_producer = registry.opened_record("scaffold_holdout_producer")
    verified = verify_bundle_checksums(
        registry,
        checksum_key="group_reconstruction_checksums",
        artifact_keys=(
            "group_reconstruction_summary",
            "group_reconstruction_metrics",
            "group_reconstruction_target_metrics",
            "group_reconstruction_table",
            "group_reconstruction_contrasts",
            "group_reconstruction_folds",
            "group_reconstruction_selections",
            "group_reconstruction_readme",
        ),
    )
    summary = registry.read_json("group_reconstruction_summary")
    metrics = registry.read_csv("group_reconstruction_metrics")
    target_metrics = registry.read_csv("group_reconstruction_target_metrics")
    table = registry.read_csv("group_reconstruction_table")
    contrasts = registry.read_csv("group_reconstruction_contrasts")
    folds = registry.read_csv("group_reconstruction_folds")
    selections = registry.read_csv("group_reconstruction_selections")

    expected_summary_keys = {
        "analysis_status",
        "claim_boundary",
        "configuration",
        "datasets",
        "dependency_sources",
        "inputs",
        "primary_results",
        "producer",
        "schema_version",
    }
    if set(summary) != expected_summary_keys or summary.get("schema_version") != "1.0.0":
        raise ValueError("group-disjoint reconstruction summary schema changed")
    if summary.get("analysis_status") != (
        "strict_public_group_disjoint_panel_reconstruction_sensitivity"
    ):
        raise ValueError("group-disjoint reconstruction analysis status changed")
    if summary.get("producer") != INPUT_SPECS["group_reconstruction_producer"].path:
        raise ValueError("group-disjoint reconstruction producer path changed")
    expected_configuration = {
        "base_seed": 20260826,
        "folds": 5,
        "panel_k": 8,
        "panel_objective": "exact k-medoids on pilot residual distance 1-r^2",
        "pilot_sizes": [200, 500],
        "primary_deployment_contract": (
            "pilot-only target-mean imputation and scaling; eight selected raw "
            "standardized scores are the only held-out predictors; selected raw "
            "scores are copied into the full-profile policy output"
        ),
        "primary_outcome_basis": "split_imputed_all_rows",
        "primary_truth": "calibration_standardized_raw",
        "repetitions_per_fold": 5,
        "replicates_per_dataset_design_size": 25,
    }
    if summary.get("configuration") != expected_configuration:
        raise ValueError("group-disjoint reconstruction configuration changed")
    expected_inputs = {
        "docking44": {
            "path": INPUT_SPECS["docking44"].path,
            "sha256": PINNED_SHA256["docking44"],
        },
        "dockstring58": {
            "path": INPUT_SPECS["dockstring"].path,
            "sha256": PINNED_SHA256["dockstring"],
        },
    }
    if summary.get("inputs") != expected_inputs:
        raise ValueError("group-disjoint reconstruction public inputs drifted")
    expected_dependencies = {
        "public_panel_domain_utility.py": panel_utility["sha256"],
        "public_scaffold_holdout_recovery.py": scaffold_producer["sha256"],
    }
    if summary.get("dependency_sources") != expected_dependencies:
        raise ValueError(
            "group-disjoint reconstruction dependency source checksum drifted"
        )

    metric_columns = tuple(
        """dataset holdout_design fold pilot_ligands held_out_ligands repetition
        panel_method panel_k panel_indices panel_targets pilot_held_row_overlap
        pilot_only_preprocessing pilot_imputed_cells held_imputed_cells
        maximum_absolute_pilot_mean minimum_pilot_target_sd variance_weighted_r2
        mean_target_r2 median_target_r2 pooled_calibration_standardized_rmse
        mean_target_pearson median_target_pearson mean_target_spearman
        fraction_targets_positive_r2 median_ligand_profile_pearson
        mean_ligand_profile_pearson median_ligand_profile_spearman
        mean_ligand_profile_spearman full_profile_policy_variance_weighted_r2
        full_profile_policy_mean_target_r2 full_profile_policy_median_target_r2
        full_profile_policy_pooled_calibration_standardized_rmse
        full_profile_policy_mean_target_pearson
        full_profile_policy_median_target_pearson
        full_profile_policy_mean_target_spearman
        full_profile_policy_fraction_targets_positive_r2
        full_profile_policy_median_ligand_profile_pearson
        full_profile_policy_mean_ligand_profile_pearson
        full_profile_policy_median_ligand_profile_spearman
        full_profile_policy_mean_ligand_profile_spearman
        median_target_calibration_threshold_lower_5pct_precision
        median_target_calibration_threshold_lower_5pct_recall
        micro_calibration_threshold_lower_5pct_precision
        micro_calibration_threshold_lower_5pct_recall
        median_target_equal_budget_lower_5pct_overlap
        mean_target_equal_budget_lower_5pct_overlap
        fraction_targets_with_no_calibration_threshold_evaluation_members
        truth_surface outcome_basis ridge_alpha selected_targets omitted_targets
        calibration_rows_scored evaluation_rows_scored
        zero_baseline_variance_weighted_r2
        zero_baseline_pooled_calibration_standardized_rmse""".split()
    )
    target_metric_columns = tuple(
        """dataset holdout_design fold pilot_ligands held_out_ligands repetition
        panel_method panel_k panel_indices panel_targets pilot_held_row_overlap
        pilot_only_preprocessing pilot_imputed_cells held_imputed_cells
        maximum_absolute_pilot_mean minimum_pilot_target_sd truth_surface
        outcome_basis target r2 calibration_standardized_rmse pearson spearman
        calibration_threshold_lower_5pct
        calibration_threshold_lower_5pct_true_count
        calibration_threshold_lower_5pct_predicted_count
        calibration_threshold_lower_5pct_true_positive
        calibration_threshold_lower_5pct_precision
        calibration_threshold_lower_5pct_recall equal_budget_lower_5pct_count
        equal_budget_lower_5pct_overlap""".split()
    )
    summary_columns = tuple(
        """dataset holdout_design pilot_ligands truth_surface outcome_basis
        replicates variance_weighted_r2_mean variance_weighted_r2_median
        variance_weighted_r2_q025 variance_weighted_r2_q975
        pooled_calibration_standardized_rmse_mean
        pooled_calibration_standardized_rmse_median
        pooled_calibration_standardized_rmse_q025
        pooled_calibration_standardized_rmse_q975 median_target_pearson_mean
        median_target_pearson_median median_target_pearson_q025
        median_target_pearson_q975 median_ligand_profile_pearson_mean
        median_ligand_profile_pearson_median median_ligand_profile_pearson_q025
        median_ligand_profile_pearson_q975 median_ligand_profile_spearman_mean
        median_ligand_profile_spearman_median
        median_ligand_profile_spearman_q025
        median_ligand_profile_spearman_q975
        full_profile_policy_variance_weighted_r2_mean
        full_profile_policy_variance_weighted_r2_median
        full_profile_policy_variance_weighted_r2_q025
        full_profile_policy_variance_weighted_r2_q975
        full_profile_policy_pooled_calibration_standardized_rmse_mean
        full_profile_policy_pooled_calibration_standardized_rmse_median
        full_profile_policy_pooled_calibration_standardized_rmse_q025
        full_profile_policy_pooled_calibration_standardized_rmse_q975
        full_profile_policy_median_ligand_profile_pearson_mean
        full_profile_policy_median_ligand_profile_pearson_median
        full_profile_policy_median_ligand_profile_pearson_q025
        full_profile_policy_median_ligand_profile_pearson_q975
        full_profile_policy_median_ligand_profile_spearman_mean
        full_profile_policy_median_ligand_profile_spearman_median
        full_profile_policy_median_ligand_profile_spearman_q025
        full_profile_policy_median_ligand_profile_spearman_q975
        median_target_calibration_threshold_lower_5pct_precision_mean
        median_target_calibration_threshold_lower_5pct_precision_median
        median_target_calibration_threshold_lower_5pct_precision_q025
        median_target_calibration_threshold_lower_5pct_precision_q975
        median_target_calibration_threshold_lower_5pct_recall_mean
        median_target_calibration_threshold_lower_5pct_recall_median
        median_target_calibration_threshold_lower_5pct_recall_q025
        median_target_calibration_threshold_lower_5pct_recall_q975
        median_target_equal_budget_lower_5pct_overlap_mean
        median_target_equal_budget_lower_5pct_overlap_median
        median_target_equal_budget_lower_5pct_overlap_q025
        median_target_equal_budget_lower_5pct_overlap_q975""".split()
    )
    contrast_columns = tuple(
        """dataset pilot_ligands truth_surface outcome_basis metric
        descriptive_replicates group_minus_random_mean group_minus_random_median
        group_minus_random_q025 group_minus_random_q975""".split()
    )
    fold_columns = tuple(
        """dataset holdout_design fold pool_ligands held_ligands pool_groups
        held_groups group_overlap pool_held_row_overlap pool_missing_cells
        held_missing_cells""".split()
    )
    selection_columns = tuple(
        """dataset holdout_design fold pilot_ligands held_out_ligands repetition
        panel_method panel_k panel_indices panel_targets pilot_held_row_overlap
        pilot_only_preprocessing pilot_imputed_cells held_imputed_cells
        maximum_absolute_pilot_mean minimum_pilot_target_sd target_index
        target""".split()
    )
    for name, frame, expected in (
        ("replicate metrics", metrics, metric_columns),
        ("target metrics", target_metrics, target_metric_columns),
        ("summary table", table, summary_columns),
        ("descriptive contrasts", contrasts, contrast_columns),
        ("fold contract", folds, fold_columns),
        ("panel selections", selections, selection_columns),
    ):
        if tuple(frame.columns) != expected:
            raise ValueError(f"group-disjoint reconstruction {name} schema changed")
        if frame.duplicated().any():
            raise ValueError(f"group-disjoint reconstruction {name} has duplicate rows")

    datasets = {"Docking-44", "DOCKSTRING-58"}
    designs = {"chemical_group", "matched_random_row"}
    truths = {"calibration_standardized_raw", "full_row_residual_outcome"}
    outcomes = {"split_imputed_all_rows", "originally_complete_outcome_rows"}
    expected_metric_grid = {
        (dataset, design, fold, pilot, repetition, truth, outcome)
        for dataset in datasets
        for design in designs
        for fold in range(1, 6)
        for pilot in (200, 500)
        for repetition in range(5)
        for truth in truths
        for outcome in outcomes
    }
    metric_grid = set(
        metrics[
            [
                "dataset",
                "holdout_design",
                "fold",
                "pilot_ligands",
                "repetition",
                "truth_surface",
                "outcome_basis",
            ]
        ].itertuples(index=False, name=None)
    )
    if len(metrics) != 800 or metric_grid != expected_metric_grid:
        raise ValueError("group-disjoint reconstruction replicate grid changed")
    if not (
        metrics["panel_method"].eq("pilot_exact_residual_r2").all()
        and metrics["panel_k"].eq(8).all()
        and metrics["pilot_held_row_overlap"].eq(0).all()
        and metrics["pilot_only_preprocessing"].eq(True).all()
        and metrics["selected_targets"].eq(8).all()
    ):
        raise ValueError("group-disjoint reconstruction deployment contract changed")

    lower_tail_metrics = (
        "median_target_calibration_threshold_lower_5pct_precision",
        "median_target_calibration_threshold_lower_5pct_recall",
        "micro_calibration_threshold_lower_5pct_precision",
        "micro_calibration_threshold_lower_5pct_recall",
        "median_target_equal_budget_lower_5pct_overlap",
        "mean_target_equal_budget_lower_5pct_overlap",
    )
    raw_rows = metrics["truth_surface"].eq("calibration_standardized_raw")
    if metrics.loc[raw_rows, list(lower_tail_metrics)].isna().any().any() or not metrics.loc[
        ~raw_rows, list(lower_tail_metrics)
    ].isna().all().all():
        raise ValueError("residual lower-tail omission contract changed")
    if metrics.drop(columns=list(lower_tail_metrics)).isna().any().any():
        raise ValueError("group-disjoint replicate metrics gained unexpected missingness")

    if len(target_metrics) != 34_400:
        raise ValueError("group-disjoint target-metric cardinality changed")
    target_lower_tail = (
        "calibration_threshold_lower_5pct",
        "calibration_threshold_lower_5pct_true_count",
        "calibration_threshold_lower_5pct_predicted_count",
        "calibration_threshold_lower_5pct_true_positive",
        "calibration_threshold_lower_5pct_precision",
        "calibration_threshold_lower_5pct_recall",
        "equal_budget_lower_5pct_count",
        "equal_budget_lower_5pct_overlap",
    )
    raw_target_rows = target_metrics["truth_surface"].eq(
        "calibration_standardized_raw"
    )
    if target_metrics.loc[raw_target_rows, list(target_lower_tail)].isna().any().any() or not (
        target_metrics.loc[~raw_target_rows, list(target_lower_tail)].isna().all().all()
    ):
        raise ValueError("target-level residual lower-tail omission contract changed")
    if target_metrics.drop(columns=list(target_lower_tail)).isna().any().any():
        raise ValueError("target-level reconstruction metrics gained missingness")
    target_group_sizes = target_metrics.groupby(
        [
            "dataset",
            "holdout_design",
            "fold",
            "pilot_ligands",
            "repetition",
            "truth_surface",
            "outcome_basis",
        ],
        sort=False,
    ).size()
    expected_target_sizes = target_group_sizes.index.get_level_values("dataset").map(
        {"Docking-44": 36, "DOCKSTRING-58": 50}
    )
    if not np.array_equal(target_group_sizes.to_numpy(), expected_target_sizes.to_numpy()):
        raise ValueError("omitted-target coverage changed")

    expected_summary_grid = {
        (dataset, design, pilot, truth, outcome)
        for dataset in datasets
        for design in designs
        for pilot in (200, 500)
        for truth in truths
        for outcome in outcomes
    }
    observed_summary_grid = set(
        table[
            [
                "dataset",
                "holdout_design",
                "pilot_ligands",
                "truth_surface",
                "outcome_basis",
            ]
        ].itertuples(index=False, name=None)
    )
    if len(table) != 32 or observed_summary_grid != expected_summary_grid:
        raise ValueError("group-disjoint reconstruction summary coverage changed")
    if not table["replicates"].eq(25).all():
        raise ValueError("group-disjoint reconstruction summary replication changed")
    summary_lower_tail = tuple(
        column
        for column in summary_columns
        if column.startswith("median_target_calibration_threshold_lower_5pct_")
        or column.startswith("median_target_equal_budget_lower_5pct_overlap_")
    )
    raw_summary_rows = table["truth_surface"].eq("calibration_standardized_raw")
    if table.loc[raw_summary_rows, list(summary_lower_tail)].isna().any().any() or not table.loc[
        ~raw_summary_rows, list(summary_lower_tail)
    ].isna().all().all():
        raise ValueError("summary residual lower-tail omission contract changed")

    if len(contrasts) != 168 or not contrasts["descriptive_replicates"].eq(25).all():
        raise ValueError("group-minus-random descriptive contrast coverage changed")
    if contrasts.isna().any().any():
        raise ValueError("group-minus-random descriptive contrasts gained missingness")
    if len(folds) != 20 or not folds["pool_held_row_overlap"].eq(0).all():
        raise ValueError("group-disjoint fold contract changed")
    chemical_folds = folds["holdout_design"].eq("chemical_group")
    if not folds.loc[chemical_folds, "group_overlap"].eq(0).all():
        raise ValueError("chemical-group folds overlap")
    if len(selections) != 1_600:
        raise ValueError("group-disjoint panel selection cardinality changed")
    selection_sizes = selections.groupby(
        ["dataset", "holdout_design", "fold", "pilot_ligands", "repetition"],
        sort=False,
    ).size()
    if not selection_sizes.eq(8).all() or selections.duplicated(
        ["dataset", "holdout_design", "fold", "pilot_ligands", "repetition", "target"]
    ).any():
        raise ValueError("group-disjoint panel selections are not unique k=8 panels")
    if not (
        selections["panel_method"].eq("pilot_exact_residual_r2").all()
        and selections["panel_k"].eq(8).all()
        and selections["pilot_held_row_overlap"].eq(0).all()
        and selections["pilot_only_preprocessing"].eq(True).all()
    ):
        raise ValueError("group-disjoint panel-selection contract changed")

    primary_table = table.loc[
        table["truth_surface"].eq("calibration_standardized_raw")
        & table["outcome_basis"].eq("split_imputed_all_rows")
    ].sort_values(["dataset", "holdout_design", "pilot_ligands"])
    primary_payload = sorted(
        summary["primary_results"],
        key=lambda row: (row["dataset"], row["holdout_design"], row["pilot_ligands"]),
    )
    primary_rows = primary_table.to_dict("records")
    if len(primary_payload) != 8 or len(primary_rows) != 8:
        raise ValueError("group-disjoint primary-result coverage changed")
    for csv_row, json_row in zip(primary_rows, primary_payload, strict=True):
        if set(csv_row) != set(json_row):
            raise ValueError("group-disjoint primary-result schema changed")
        for key, csv_value in csv_row.items():
            json_value = json_row[key]
            if isinstance(csv_value, (float, np.floating)):
                if abs(float(csv_value) - float(json_value)) > 1e-12:
                    raise ValueError("group-disjoint primary results disagree")
            elif csv_value != json_value:
                raise ValueError("group-disjoint primary results disagree")

    if set(summary.get("datasets", {})) != datasets:
        raise ValueError("group-disjoint dataset contract changed")
    for dataset, rows, targets in (
        ("Docking-44", 12_651, 44),
        ("DOCKSTRING-58", 15_000, 58),
    ):
        record = summary["datasets"][dataset]
        if record.get("analysis_rows") != rows or record.get("targets") != targets:
            raise ValueError(f"group-disjoint support changed: {dataset}")

    return {
        "analysis_status": summary["analysis_status"],
        "configuration": summary["configuration"],
        "datasets": summary["datasets"],
        "primary_results": summary["primary_results"],
        "descriptive_group_minus_random": json_ready(contrasts.to_dict("records")),
        "fold_contract": json_ready(folds.to_dict("records")),
        "replicate_metric_rows": int(len(metrics)),
        "target_metric_rows": int(len(target_metrics)),
        "panel_selection_rows": int(len(selections)),
        "dependency_sources": summary["dependency_sources"],
        "verified_output_sha256": verified,
        "claim_boundary": summary["claim_boundary"],
    }


def target_panel_selector_controls_evidence(
    registry: InputRegistry,
) -> dict[str, Any]:
    """Import the frozen selector controls and their no-copy k=8 audit."""

    selector_producer = registry.opened_record("selector_controls_producer")
    selector_test = registry.opened_record("selector_controls_test")
    robustness_producer = registry.opened_record("selector_robustness_producer")
    robustness_test = registry.opened_record("selector_robustness_test")
    dependency_sources = {
        "public_scaffold_holdout_recovery.py": registry.opened_record(
            "scaffold_holdout_producer"
        )["sha256"],
        "residual_mechanism_analysis.py": registry.opened_record(
            "revision_loader_helper"
        )["sha256"],
    }

    selector_artifact_keys = (
        "selector_controls_summary",
        "selector_controls_metrics",
        "selector_controls_splits",
        "selector_controls_selections",
        "selector_controls_baseline_contrasts",
        "selector_controls_pair_contrasts",
        "selector_controls_metric_summary",
        "selector_controls_baseline_summary",
        "selector_controls_pair_summary",
        "selector_controls_readme",
    )
    robustness_artifact_keys = (
        "selector_robustness_summary",
        "selector_robustness_metrics",
        "selector_robustness_target_metrics",
        "selector_robustness_splits",
        "selector_robustness_common_pairs",
        "selector_robustness_common_splits",
        "selector_robustness_common_summary",
        "selector_robustness_direct_contrasts",
        "selector_robustness_direct_summary",
        "selector_robustness_readme",
    )
    selector_verified = verify_sha256_text_bundle(
        registry,
        checksum_key="selector_controls_checksums",
        artifact_keys=selector_artifact_keys,
    )
    robustness_verified = verify_sha256_text_bundle(
        registry,
        checksum_key="selector_robustness_checksums",
        artifact_keys=robustness_artifact_keys,
    )

    selector_summary = registry.read_json("selector_controls_summary")
    selector_metrics = registry.read_csv("selector_controls_metrics")
    selector_splits = registry.read_csv("selector_controls_splits")
    selector_selections = registry.read_csv("selector_controls_selections")
    selector_baseline_contrasts = registry.read_csv(
        "selector_controls_baseline_contrasts"
    )
    selector_pair_contrasts = registry.read_csv("selector_controls_pair_contrasts")
    selector_metric_summary = registry.read_csv("selector_controls_metric_summary")
    selector_baseline_summary = registry.read_csv(
        "selector_controls_baseline_summary"
    )
    selector_pair_summary = registry.read_csv("selector_controls_pair_summary")

    robustness_summary = registry.read_json("selector_robustness_summary")
    robustness_metrics = registry.read_csv("selector_robustness_metrics")
    robustness_target_metrics = registry.read_csv(
        "selector_robustness_target_metrics"
    )
    robustness_splits = registry.read_csv("selector_robustness_splits")
    common_pairs = registry.read_csv("selector_robustness_common_pairs")
    common_splits = registry.read_csv("selector_robustness_common_splits")
    common_summary = registry.read_csv("selector_robustness_common_summary")
    direct_contrasts = registry.read_csv("selector_robustness_direct_contrasts")
    direct_summary = registry.read_csv("selector_robustness_direct_summary")

    expected_selector_parameters = {
        "descriptor_names": [
            "heavy_atoms", "molecular_weight", "labute_asa", "tpsa",
            "clogp", "rotatable_bonds", "ring_count",
        ],
        "folds": 5,
        "max_folds_diagnostic": None,
        "panel_sizes": [4, 6, 8, 10, 12],
        "pilot_sizes": [500],
        "primary_metric": "full_profile_variance_weighted_r2",
        "random_panels_per_split_and_k": 50,
        "ridge_alphas": [0.01, 0.1, 1.0, 10.0, 100.0],
        "seed": 20260828,
    }
    expected_selector_rows = {
        "baseline_contrast_summary.csv": 40,
        "paired_baseline_contrasts.csv": 200,
        "paired_selector_contrasts.csv": 100,
        "panel_metrics.csv.gz": 5420,
        "panel_selections.csv": 800,
        "selector_metric_summary.csv": 104,
        "selector_pair_summary.csv": 20,
        "split_diagnostics.csv": 10,
    }
    if set(selector_summary) != {
        "analysis_status", "contracts", "dataset_metadata", "input_sha256",
        "key_results", "limitations", "parameters", "producer",
        "producer_sha256", "table_rows",
    }:
        raise ValueError("selector-control summary schema changed")
    if selector_summary.get("analysis_status") != (
        "strict_public_selector_and_panel_size_control"
    ):
        raise ValueError("selector-control analysis status changed")
    if selector_summary.get("parameters") != expected_selector_parameters:
        raise ValueError("selector-control parameters changed")
    if selector_summary.get("table_rows") != expected_selector_rows:
        raise ValueError("selector-control table row contract changed")
    if selector_summary.get("input_sha256") != {
        INPUT_SPECS["docking44"].path: PINNED_SHA256["docking44"],
        INPUT_SPECS["dockstring"].path: PINNED_SHA256["dockstring"],
    }:
        raise ValueError("selector-control public inputs changed")
    if (
        selector_summary.get("producer") != selector_producer["path"]
        or selector_summary.get("producer_sha256") != selector_producer["sha256"]
    ):
        raise ValueError("selector-control producer changed")
    if set(selector_summary.get("key_results", {})) != {
        "designed_panel_baselines", "selector_pair"
    }:
        raise ValueError("selector-control key-result schema changed")

    expected_robustness_parameters = {
        "folds": 5,
        "panel_k": 8,
        "pilot_ligands": 500,
        "random_panel_draws": [0, 1, 2],
        "ridge_alphas": [0.01, 0.1, 1.0, 10.0, 100.0],
        "source_seed": 20260828,
    }
    expected_robustness_rows = {
        "common_omitted_pair_contrasts.csv": 60,
        "common_omitted_split_contrasts.csv": 20,
        "common_omitted_summary.csv": 4,
        "direct_residual_contrasts.csv": 10,
        "direct_residual_summary.csv": 2,
        "panel_metrics.csv": 180,
        "primary_target_metrics.csv": 3440,
        "split_diagnostics.csv": 10,
    }
    if set(robustness_summary) != {
        "analysis_status", "contracts", "dataset_input_sha256", "limitations",
        "parameters", "producer", "producer_sha256", "source_artifact",
        "source_artifact_sha256", "table_rows", "test_sha256",
    }:
        raise ValueError("selector-robustness summary schema changed")
    if robustness_summary.get("analysis_status") != (
        "strict_public_k8_selector_robustness_control"
    ):
        raise ValueError("selector-robustness analysis status changed")
    if robustness_summary.get("parameters") != expected_robustness_parameters:
        raise ValueError("selector-robustness parameters changed")
    if robustness_summary.get("table_rows") != expected_robustness_rows:
        raise ValueError("selector-robustness table row contract changed")
    if robustness_summary.get("dataset_input_sha256") != {
        "Docking-44": PINNED_SHA256["docking44"],
        "DOCKSTRING-58": PINNED_SHA256["dockstring"],
    }:
        raise ValueError("selector-robustness public inputs changed")
    if (
        robustness_summary.get("producer") != robustness_producer["path"]
        or robustness_summary.get("producer_sha256")
        != robustness_producer["sha256"]
        or robustness_summary.get("test_sha256") != robustness_test["sha256"]
    ):
        raise ValueError("selector-robustness producer or test changed")
    if robustness_summary.get("source_artifact") != (
        "results/public_panel_selector_controls"
    ):
        raise ValueError("selector-robustness source artifact changed")
    expected_source_hashes = {
        "checksums.sha256": registry.opened_record("selector_controls_checksums")[
            "sha256"
        ],
        "panel_metrics.csv.gz": selector_verified["panel_metrics.csv.gz"],
        "split_diagnostics.csv": selector_verified["split_diagnostics.csv"],
        "summary.json": selector_verified["summary.json"],
    }
    if robustness_summary.get("source_artifact_sha256") != expected_source_hashes:
        raise ValueError("selector-robustness source-artifact checksums changed")

    selector_metric_columns = tuple(
        """dataset fold pilot_ligands evaluation_ligands pilot_complete_rows
        evaluation_complete_rows pilot_evaluation_row_overlap
        pilot_evaluation_group_overlap panel_k selector predictor
        random_panel_draw surface panel_indices panel_targets selected_targets
        omitted_targets ridge_alpha pilot_gcv full_profile_variance_weighted_r2
        full_profile_pooled_rmse full_profile_median_target_pearson
        full_profile_median_ligand_profile_pearson
        full_profile_mean_ligand_profile_pearson
        omitted_only_variance_weighted_r2 omitted_only_pooled_rmse
        omitted_only_median_target_pearson
        omitted_only_median_ligand_profile_pearson
        omitted_only_mean_ligand_profile_pearson""".split()
    )
    selector_split_columns = tuple(
        """dataset fold pilot_ligands evaluation_ligands pilot_complete_rows
        evaluation_complete_rows pilot_evaluation_row_overlap
        pilot_evaluation_group_overlap training_pool_rows
        complete_training_pool_rows held_fold_rows_before_complete_filter
        pilot_groups evaluation_groups pilot_missing_cells
        evaluation_missing_cells pilot_row_index_sha256
        evaluation_row_index_sha256""".split()
    )
    selector_selection_columns = tuple(
        """dataset fold pilot_ligands evaluation_ligands pilot_complete_rows
        evaluation_complete_rows pilot_evaluation_row_overlap
        pilot_evaluation_group_overlap panel_k selector target_index target""".split()
    )
    selector_baseline_contrast_columns = tuple(
        """dataset fold pilot_ligands panel_k surface selector metric
        higher_is_better designed_value random_median
        designed_minus_random_median fraction_random_panels_no_worse
        selected_panel_pc1_value ridge_minus_selected_pc1
        descriptor_only_value ridge_minus_descriptor_only random_panels""".split()
    )
    selector_pair_contrast_columns = tuple(
        """dataset fold pilot_ligands panel_k surface comparison metric
        higher_is_better pivoted_qr_value residual_kmedoids_value
        pivoted_qr_minus_residual_kmedoids""".split()
    )
    selector_metric_summary_columns = tuple(
        "dataset pilot_ligands panel_k selector predictor surface records folds".split()
        + [
            f"{metric}_{statistic}"
            for metric in (
                "full_profile_variance_weighted_r2",
                "full_profile_pooled_rmse",
                "full_profile_median_target_pearson",
                "full_profile_median_ligand_profile_pearson",
                "omitted_only_variance_weighted_r2",
                "omitted_only_pooled_rmse",
                "omitted_only_median_target_pearson",
                "omitted_only_median_ligand_profile_pearson",
            )
            for statistic in ("median", "minimum", "maximum")
        ]
    )
    selector_baseline_summary_columns = tuple(
        """dataset pilot_ligands panel_k surface selector folds
        designed_minus_random_median_median
        designed_minus_random_median_minimum
        designed_minus_random_median_maximum
        fraction_random_panels_no_worse_median
        fraction_random_panels_no_worse_minimum
        fraction_random_panels_no_worse_maximum
        ridge_minus_selected_pc1_median ridge_minus_selected_pc1_minimum
        ridge_minus_selected_pc1_maximum ridge_minus_descriptor_only_median
        ridge_minus_descriptor_only_minimum ridge_minus_descriptor_only_maximum""".split()
    )
    selector_pair_summary_columns = tuple(
        """dataset pilot_ligands panel_k surface comparison folds
        pivoted_qr_minus_residual_kmedoids_median
        pivoted_qr_minus_residual_kmedoids_minimum
        pivoted_qr_minus_residual_kmedoids_maximum""".split()
    )
    robustness_metric_columns = tuple(
        """dataset fold pilot_ligands panel_k panel_selector random_panel_draw
        panel_indices selected_targets omitted_targets training_outcome surface
        predictor ridge_alpha pilot_gcv variance_weighted_r2 pooled_rmse
        median_target_pearson median_ligand_profile_pearson
        mean_ligand_profile_pearson""".split()
    )
    robustness_target_columns = tuple(
        """dataset fold pilot_ligands panel_k panel_selector random_panel_draw
        panel_indices selected_targets omitted_targets training_outcome surface
        predictor target_index target r2 rmse pearson spearman
        equal_budget_lower_5pct_overlap calibration_lower_5pct_threshold
        calibration_threshold_true_count calibration_threshold_predicted_count
        calibration_threshold_true_positive""".split()
    )
    robustness_split_columns = tuple(
        """dataset fold pilot_ligands evaluation_ligands pilot_row_index_sha256
        evaluation_row_index_sha256 pilot_evaluation_row_overlap
        pilot_evaluation_group_overlap""".split()
    )
    common_metric_names = (
        "r2", "rmse", "pearson", "spearman", "equal_budget_lower_5pct_overlap",
    )
    common_pair_columns = tuple(
        """dataset fold pilot_ligands surface training_outcome random_panel_draw
        common_omitted_targets total_targets designed_panel_indices
        random_panel_indices""".split()
        + [
            f"{prefix}_{aggregation}_{metric}"
            for metric in common_metric_names
            for aggregation in ("mean", "median")
            for prefix in (
                "designed", "random", "designed_minus_random", "designed_benefit",
            )
        ]
    )
    common_split_columns = tuple(
        """dataset fold pilot_ligands surface training_outcome random_panels
        common_omitted_targets_min common_omitted_targets_max""".split()
        + [
            f"designed_benefit_{aggregation}_{metric}_{summary}"
            for metric in common_metric_names
            for aggregation in ("mean", "median")
            for summary in ("random_median", "fraction_positive")
        ]
    )
    common_summary_columns = tuple(
        """dataset surface training_outcome folds pilot_ligands_median
        pilot_ligands_min pilot_ligands_max random_panels_median
        random_panels_min random_panels_max common_omitted_targets_min_median
        common_omitted_targets_min_min common_omitted_targets_min_max
        common_omitted_targets_max_median common_omitted_targets_max_min
        common_omitted_targets_max_max""".split()
        + [
            f"designed_benefit_{aggregation}_{metric}_{summary}_{statistic}"
            for metric in common_metric_names
            for aggregation in ("mean", "median")
            for summary in ("random_median", "fraction_positive")
            for statistic in ("median", "min", "max")
        ]
    )
    direct_metric_names = (
        "variance_weighted_r2", "pooled_rmse", "median_target_pearson",
        "median_ligand_profile_pearson", "mean_ligand_profile_pearson",
    )
    direct_terms = (
        "derived_ridge", "direct_ridge", "derived_pc1", "direct_pc1",
        "direct_minus_derived_ridge_benefit",
        "direct_ridge_minus_pc1_benefit", "derived_ridge_minus_pc1_benefit",
    )
    direct_contrast_columns = tuple(
        ["dataset", "fold"]
        + [f"{term}_{metric}" for metric in direct_metric_names for term in direct_terms]
    )
    direct_summary_columns = tuple(
        ["dataset", "folds"]
        + [
            f"{term}_{metric}_{statistic}"
            for metric in direct_metric_names
            for term in direct_terms
            for statistic in ("median", "min", "max")
        ]
    )
    tables = (
        ("selector metrics", selector_metrics, selector_metric_columns, 5420),
        ("selector splits", selector_splits, selector_split_columns, 10),
        ("selector selections", selector_selections, selector_selection_columns, 800),
        ("selector baseline contrasts", selector_baseline_contrasts, selector_baseline_contrast_columns, 200),
        ("selector pair contrasts", selector_pair_contrasts, selector_pair_contrast_columns, 100),
        ("selector metric summary", selector_metric_summary, selector_metric_summary_columns, 104),
        ("selector baseline summary", selector_baseline_summary, selector_baseline_summary_columns, 40),
        ("selector pair summary", selector_pair_summary, selector_pair_summary_columns, 20),
        ("robustness metrics", robustness_metrics, robustness_metric_columns, 180),
        ("robustness target metrics", robustness_target_metrics, robustness_target_columns, 3440),
        ("robustness splits", robustness_splits, robustness_split_columns, 10),
        ("common-omitted pairs", common_pairs, common_pair_columns, 60),
        ("common-omitted splits", common_splits, common_split_columns, 20),
        ("common-omitted summary", common_summary, common_summary_columns, 4),
        ("direct-residual contrasts", direct_contrasts, direct_contrast_columns, 10),
        ("direct-residual summary", direct_summary, direct_summary_columns, 2),
    )
    for name, frame, expected_columns, expected_rows in tables:
        if tuple(frame.columns) != expected_columns:
            raise ValueError(f"{name} schema changed")
        if len(frame) != expected_rows or frame.duplicated().any():
            raise ValueError(f"{name} row contract changed")
    selector_split_projection = selector_splits.loc[
        :, list(robustness_split_columns)
    ]
    if not selector_split_projection.equals(robustness_splits):
        raise ValueError("selector and robustness split contracts disagree")

    designed_selector = "pilot_residual_r2_exact_kmedoids"
    k8_results: list[dict[str, Any]] = []
    for dataset in ("Docking-44", "DOCKSTRING-58"):
        source = selector_metrics.loc[
            selector_metrics.dataset.eq(dataset)
            & selector_metrics.pilot_ligands.eq(500)
            & selector_metrics.panel_k.eq(8)
            & selector_metrics.selector.eq(designed_selector)
        ]
        robust = robustness_metrics.loc[
            robustness_metrics.dataset.eq(dataset)
            & robustness_metrics.panel_selector.eq(designed_selector)
            & robustness_metrics.random_panel_draw.eq(-1)
        ]
        if len(source) != 20 or len(robust) != 30:
            raise ValueError(f"k=8 designed-panel grid changed for {dataset}")
        selectors = (
            ("raw_ridge", "raw_outcome", "raw", "multivariate_ridge_gcv"),
            ("raw_pc1", "raw_outcome", "raw", "selected_target_pc1"),
            ("derived_residual_ridge", "raw_then_derived_residual", "raw_row_centered_residual", "multivariate_ridge_gcv"),
            ("derived_residual_pc1", "raw_then_derived_residual", "raw_row_centered_residual", "selected_target_pc1"),
            ("direct_residual_ridge", "direct_residual_outcome", "raw_row_centered_residual", "multivariate_ridge_gcv"),
            ("direct_residual_pc1", "direct_residual_outcome", "raw_row_centered_residual", "selected_target_pc1"),
        )
        record: dict[str, Any] = {"dataset": dataset, "folds": 5, "panel_k": 8}
        for label, outcome, surface, predictor in selectors:
            rows = robust.loc[
                robust.training_outcome.eq(outcome)
                & robust.surface.eq(surface)
                & robust.predictor.eq(predictor)
            ]
            if len(rows) != 5 or set(rows.fold) != {1, 2, 3, 4, 5}:
                raise ValueError(f"k=8 {label} fold grid changed for {dataset}")
            for statistic, value in (
                ("median", rows.variance_weighted_r2.median()),
                ("minimum", rows.variance_weighted_r2.min()),
                ("maximum", rows.variance_weighted_r2.max()),
            ):
                record[f"{label}_variance_weighted_r2_{statistic}"] = float(value)
            if outcome != "direct_residual_outcome":
                source_rows = source.loc[
                    source.surface.eq(surface) & source.predictor.eq(predictor)
                ].sort_values("fold")
                robust_rows = rows.sort_values("fold")
                if len(source_rows) != 5 or not np.allclose(
                    source_rows.omitted_only_variance_weighted_r2.to_numpy(),
                    robust_rows.variance_weighted_r2.to_numpy(),
                    rtol=0.0,
                    atol=1e-12,
                ):
                    raise ValueError(
                        f"selector source and robustness k=8 metrics disagree for {dataset}"
                    )
        direct = one_row(direct_summary, dataset=dataset)
        summary_to_record = {
            "derived_ridge": "derived_residual_ridge",
            "direct_ridge": "direct_residual_ridge",
            "derived_pc1": "derived_residual_pc1",
            "direct_pc1": "direct_residual_pc1",
        }
        for summary_prefix, record_prefix in summary_to_record.items():
            summary_key = f"{summary_prefix}_variance_weighted_r2_median"
            record_key = f"{record_prefix}_variance_weighted_r2_median"
            if abs(float(direct[summary_key]) - float(record[record_key])) > 1e-12:
                raise ValueError(
                    f"direct-residual summary disagrees for {dataset}: {summary_key}"
                )
        for statistic in ("median", "min", "max"):
            record[
                f"direct_minus_derived_ridge_variance_weighted_r2_{statistic}"
            ] = float(
                direct[
                    f"direct_minus_derived_ridge_benefit_variance_weighted_r2_{statistic}"
                ]
            )
            record[
                f"derived_ridge_minus_pc1_variance_weighted_r2_{statistic}"
            ] = float(
                direct[
                    f"derived_ridge_minus_pc1_benefit_variance_weighted_r2_{statistic}"
                ]
            )
        k8_results.append(record)

    common_records = json_ready(common_summary.to_dict("records"))
    if {
        (row["dataset"], row["surface"], row["training_outcome"])
        for row in common_records
    } != {
        (dataset, surface, outcome)
        for dataset in ("Docking-44", "DOCKSTRING-58")
        for surface, outcome in (
            ("raw", "raw_outcome"),
            ("raw_row_centered_residual", "raw_then_derived_residual"),
        )
    }:
        raise ValueError("common-omitted summary grid changed")
    common_fields = (
        "dataset", "surface", "training_outcome", "folds",
        "random_panels_median", "common_omitted_targets_min_median",
        "common_omitted_targets_max_median",
        "designed_benefit_mean_r2_random_median_median",
        "designed_benefit_mean_r2_random_median_min",
        "designed_benefit_mean_r2_random_median_max",
        "designed_benefit_mean_pearson_random_median_median",
        "designed_benefit_mean_spearman_random_median_median",
        "designed_benefit_mean_equal_budget_lower_5pct_overlap_random_median_median",
    )
    compact_common = [
        {key: row[key] for key in common_fields} for row in common_records
    ]
    return {
        "analysis_status": robustness_summary["analysis_status"],
        "configuration": robustness_summary["parameters"],
        "selector_control_configuration": selector_summary["parameters"],
        "contracts": {
            "selector_controls": selector_summary["contracts"],
            "k8_robustness": robustness_summary["contracts"],
        },
        "k8_omitted_target_results": json_ready(k8_results),
        "common_omitted_designed_vs_random": compact_common,
        "selector_bundle_table_rows": selector_summary["table_rows"],
        "robustness_bundle_table_rows": robustness_summary["table_rows"],
        "verified_output_sha256": {
            "selector_controls": selector_verified,
            "selector_robustness": robustness_verified,
        },
        "dependency_sources": {
            **dependency_sources,
            Path(selector_producer["path"]).name: selector_producer["sha256"],
            Path(selector_test["path"]).name: selector_test["sha256"],
            Path(robustness_producer["path"]).name: robustness_producer["sha256"],
            Path(robustness_test["path"]).name: robustness_test["sha256"],
        },
        "limitations": robustness_summary["limitations"],
    }


def hotspot_panel_validation_evidence(registry: InputRegistry) -> dict[str, Any]:
    """Import the checksum-locked fixed-21 HotSpot and panel-transfer audit."""

    code_keys = (
        "hotspot_producer", "hotspot_test", "hotspot_identity_producer",
        "hotspot_identity_test", "hotspot_fetcher",
    )
    code = {key: registry.opened_record(key) for key in code_keys}
    identity = registry.opened_record("hotspot_identity")
    identity_provenance = registry.opened_record("hotspot_identity_provenance")
    kinase_fasta = registry.opened_record("hotspot_kinase_fasta")
    kinase_manifest = registry.opened_record("hotspot_kinase_manifest")
    artifact_keys = (
        "hotspot_readme", "hotspot_excluded_ligands",
        "hotspot_cluster_multiplier", "hotspot_identity_overlap",
        "hotspot_joint_qap", "hotspot_ligand_bootstrap",
        "hotspot_map_concordance", "hotspot_mapping_sensitivities",
        "hotspot_mapping_summary", "hotspot_panel_selections",
        "hotspot_deleak_sensitivity", "hotspot_panel_transfer",
        "hotspot_split_half", "hotspot_summary", "hotspot_target_loo_maps",
        "hotspot_target_loo_transfer", "hotspot_target_mapping",
    )
    verified = verify_sha256_text_bundle(
        registry,
        checksum_key="hotspot_checksums",
        artifact_keys=artifact_keys,
    )
    summary = registry.read_json("hotspot_summary")
    tables = {
        "excluded_ligands": registry.read_csv("hotspot_excluded_ligands"),
        "cluster_multiplier": registry.read_csv("hotspot_cluster_multiplier"),
        "identity_overlap": registry.read_csv("hotspot_identity_overlap"),
        "joint_qap": registry.read_csv("hotspot_joint_qap"),
        "ligand_bootstrap": registry.read_csv("hotspot_ligand_bootstrap"),
        "map_concordance": registry.read_csv("hotspot_map_concordance"),
        "mapping_sensitivities": registry.read_csv("hotspot_mapping_sensitivities"),
        "mapping_summary": registry.read_csv("hotspot_mapping_summary"),
        "panel_selections": registry.read_csv("hotspot_panel_selections"),
        "deleak_sensitivity": registry.read_csv("hotspot_deleak_sensitivity"),
        "panel_transfer": registry.read_csv("hotspot_panel_transfer"),
        "split_half": registry.read_csv("hotspot_split_half"),
        "target_loo_maps": registry.read_csv("hotspot_target_loo_maps"),
        "target_loo_transfer": registry.read_csv("hotspot_target_loo_transfer"),
        "target_mapping": registry.read_csv("hotspot_target_mapping"),
    }
    expected_rows = {
        "excluded_ligands": 6, "cluster_multiplier": 2000,
        "identity_overlap": 3, "joint_qap": 36,
        "ligand_bootstrap": 2000, "map_concordance": 6,
        "mapping_sensitivities": 180, "mapping_summary": 36,
        "panel_selections": 544, "deleak_sensitivity": 21,
        "panel_transfer": 90, "split_half": 2000,
        "target_loo_maps": 63, "target_loo_transfer": 315,
        "target_mapping": 21,
    }
    for name, expected in expected_rows.items():
        frame = tables[name]
        if len(frame) != expected or frame.duplicated().any():
            raise ValueError(f"HotSpot {name} row contract changed")

    if set(summary) != {
        "analysis", "broad_vina_contract", "chemical_cluster_multiplier",
        "claim_boundary", "endpoint_contract", "experimental_support",
        "identity_contract", "inputs", "joint_cross_assay_inference",
        "joint_primary_target_label_qap", "primary_mapping_sensitivity_summary",
        "resampling", "same_endpoint_inference", "schema_version",
        "source_contract", "status", "target_leave_one_out_influence",
        "targets", "verdict",
    }:
        raise ValueError("HotSpot summary schema changed")
    if summary.get("schema_version") != "1.0.0" or summary.get("status") != (
        "post_hoc_external_extension"
    ):
        raise ValueError("HotSpot analysis status changed")
    if summary["endpoint_contract"]["primary_support"] != "176 complete ligands x 21 targets":
        raise ValueError("HotSpot primary support changed")
    if summary["experimental_support"] != {
        "Anastassiadis": {"ligands": 176, "targets": 21},
        "DAVIS": {"ligands": 67, "targets": 21},
        "PKIS2": {"ligands": 644, "targets": 21},
    }:
        raise ValueError("HotSpot experimental support changed")
    source = summary["source_contract"]
    if (
        source.get("sha256") != "cd756bf2b6ad541a1781508c563caf0da6da876dfb71f2546fbff02e13d98684"
        or source.get("doi") != "10.1038/nbt.2017"
        or source.get("redistributed") is not False
        or source.get("office_conversion_used") is not False
    ):
        raise ValueError("HotSpot official-source contract changed")
    if summary["identity_contract"]["identity_csv_sha256"] != identity["sha256"]:
        raise ValueError("HotSpot identity crosswalk changed")
    if summary["identity_contract"]["identity_provenance_sha256"] != (
        identity_provenance["sha256"]
    ):
        raise ValueError("HotSpot identity provenance changed")
    expected_inputs = {
        INPUT_SPECS[key].path: registry.opened_record(key)["sha256"]
        for key in (
            "hotspot_identity", "hotspot_identity_provenance", "davis", "dockstring",
            "dockstring_identity_contract", "hotspot_kinase_fasta",
            "hotspot_kinase_manifest", "pkis2",
        )
    }
    if summary.get("inputs") != expected_inputs:
        raise ValueError("HotSpot registered inputs changed")
    broad = summary["broad_vina_contract"]
    if (
        broad.get("primary_rows") != 259641
        or broad.get("locked_DAVIS_PKIS2_only_rows") != 259806
        or broad.get("additional_rows_removed_after_DAVIS_PKIS2_exclusion") != 165
    ):
        raise ValueError("HotSpot triple-deleak contract changed")

    transfer = tables["panel_transfer"]
    primary_grid = transfer.loc[
        transfer.experimental_endpoint.eq("metric_row_centered_residual")
        & transfer.objective.eq("signed_1_minus_r")
    ]
    if len(primary_grid) != 15 or set(primary_grid.targets_selected_k) != {
        4, 6, 8, 10, 12
    }:
        raise ValueError("HotSpot primary panel-transfer grid changed")
    primary_inference = one_record(
        summary["same_endpoint_inference"]["inference"],
        experimental_endpoint="metric_row_centered_residual",
        objective="signed_1_minus_r",
    )
    rank_inference = one_record(
        summary["same_endpoint_inference"]["inference"],
        experimental_endpoint="column_rank_then_row_centered",
        objective="signed_1_minus_r",
    )
    if (
        int(primary_inference["conservative_tie_cells_with_positive_improvement"]) != 14
        or abs(float(primary_inference["conservative_mean_coverage_loss_improvement_over_ties"]) - 0.06068713903344698) > 1e-12
        or abs(float(primary_inference["max_over_three_objectives_p_conservative_tie_improvement"]) - 0.005699715014249288) > 1e-15
    ):
        raise ValueError("HotSpot primary panel-transfer result changed")
    if (
        int(rank_inference["conservative_tie_cells_with_positive_improvement"]) != 14
        or abs(float(rank_inference["conservative_mean_coverage_loss_improvement_over_ties"]) - 0.0383672539228506) > 1e-12
        or abs(float(rank_inference["max_over_three_objectives_p_conservative_tie_improvement"]) - 0.005649717514124294) > 1e-15
    ):
        raise ValueError("column-rank panel-transfer sensitivity changed")
    joint = one_record(
        summary["joint_primary_target_label_qap"],
        scope="joint_mean_DAVIS_PKIS2_Anastassiadis",
        objective="signed_1_minus_r",
        statistic="conservative_best_raw_minus_worst_residual_loss",
    )
    if (
        abs(float(joint["normalized_trapezoid_auc_over_k"]) - 0.061551311281194517) > 1e-12
        or abs(float(joint["max_over_three_objectives_fwer_p_for_this_objective"]) - 0.006899655017249137) > 1e-15
    ):
        raise ValueError("joint three-assay panel-transfer result changed")

    target_loo = summary["target_leave_one_out_influence"]
    target_loo_transfer = target_loo["panel_transfer"]
    if (
        int(target_loo_transfer["omitted_targets"]) != 21
        or int(target_loo_transfer["joint_normalized_auc_positive_omissions"]) != 21
        or int(target_loo_transfer["positive_cells_min"]) != 11
        or int(target_loo_transfer["positive_cells_max"]) != 15
        or abs(float(target_loo_transfer["joint_normalized_auc_min"]) - 0.029356209793230906) > 1e-12
        or abs(float(target_loo_transfer["joint_normalized_auc_max"]) - 0.06223916889773784) > 1e-12
    ):
        raise ValueError("leave-one-target-out panel-transfer result changed")
    if not all(
        record["residual_exceeds_raw_for_every_omission"]
        for record in target_loo["cross_assay_map_concordance"]
    ):
        raise ValueError("leave-one-target-out cross-assay map result changed")

    map_rows = tables["map_concordance"]
    compact_maps = json_ready(map_rows.to_dict("records"))
    for panel, raw_expected, residual_expected in (
        ("DAVIS", 0.689729277955768, 0.76506626324536),
        ("PKIS2", 0.52640066344232, 0.751496794549925),
    ):
        raw = one_row(
            map_rows,
            comparison_panel=panel,
            anastassiadis_transformation="raw_correlation",
        )
        residual = one_row(
            map_rows,
            comparison_panel=panel,
            anastassiadis_transformation="row_centered_correlation",
        )
        if (
            abs(float(raw["target_pair_spearman"]) - raw_expected) > 1e-12
            or abs(float(residual["target_pair_spearman"]) - residual_expected) > 1e-12
        ):
            raise ValueError(f"HotSpot {panel} map concordance changed")

    return {
        "analysis_status": summary["status"],
        "source_contract": source,
        "identity_contract": summary["identity_contract"],
        "experimental_support": summary["experimental_support"],
        "broad_vina_contract": broad,
        "hotspot_reliability": summary["resampling"]["split_half_cross_map_spearman"],
        "map_concordance": compact_maps,
        "panel_transfer": {
            "primary_inference": primary_inference,
            "column_rank_inference": rank_inference,
            "joint_k_averaged_inference": joint,
            "k_grid": summary["same_endpoint_inference"]["k_grid"],
            "sequence_better_or_equal_cells": int(
                np.sum(
                    primary_grid.sequence_minus_residual_selected_coverage_loss_improvement
                    <= 0
                )
            ),
            "cells": int(len(primary_grid)),
        },
        "target_leave_one_out_influence": target_loo,
        "chemical_cluster_multiplier": summary["chemical_cluster_multiplier"],
        "verified_output_sha256": verified,
        "dependency_sources": {
            **{Path(record["path"]).name: record["sha256"] for record in code.values()},
            Path(identity["path"]).name: identity["sha256"],
            Path(identity_provenance["path"]).name: identity_provenance["sha256"],
            Path(kinase_fasta["path"]).name: kinase_fasta["sha256"],
            Path(kinase_manifest["path"]).name: kinase_manifest["sha256"],
        },
        "claim_boundary": summary["claim_boundary"],
    }


def reusable_target_map_audit_evidence(registry: InputRegistry) -> dict[str, Any]:
    keys = (
        "target_map_audit_tool",
        "target_map_audit_test",
        "target_map_audit_fixture",
        "target_map_audit_documentation",
    )
    records = {key: registry.opened_record(key) for key in keys}
    return {
        "command": "python analysis/target_map_audit.py --help",
        "files": {
            key: {
                "path": record["path"],
                "bytes": record["bytes"],
                "sha256": record["sha256"],
            }
            for key, record in records.items()
        },
        "capabilities": [
            "explicit or regular-expression target selection",
            "fail-closed numeric and missing-data contracts",
            "raw and two-way-centered correlation spectra and signed edge maps",
            "quantile-domain support comparisons",
            "row-disjoint pilot map recovery",
            "deterministic 1-|r| medoid panel compression",
            "per-run SHA-256 artifact manifest",
        ],
        "synthetic_invariants": [
            "row-centering preserves within-row target ranks",
            "a controlled ligand-support shift changes the target map",
        ],
        "interpretation_boundary": (
            "The tool audits score-map geometry. It does not validate affinity "
            "accuracy, biological mechanism, or target retrieval."
        ),
    }


def build(registry: InputRegistry | None = None) -> dict[str, Any]:
    registry = registry or InputRegistry()
    registry.path("public_core_builder_test")
    registry.path("v4_submission_consistency_test")
    large_panels = load_large_panels(registry)
    missing_data = docking44_missing_data_evidence(registry)
    direct_missing_pr = large_panels["Docking-44"][
        "two_way_centered_residual_surface"
    ]["participation_ratio_dimension"]
    artifact_missing_pr = missing_data["variants"]["target_mean_imputation"][
        "residual_pr"
    ]
    if abs(direct_missing_pr - artifact_missing_pr) > 1e-12:
        raise ValueError("direct and standalone Docking-44 residual PR disagree")
    large_panels["Docking-44"]["missing_data_sensitivity"] = missing_data
    chemical_domain = chemical_domain_control_evidence(registry)
    descriptor_domain = descriptor_domain_evidence(registry)
    scaffold_holdout = scaffold_holdout_evidence(registry)
    target_panel_selectors = target_panel_selector_controls_evidence(registry)
    hotspot_validation = hotspot_panel_validation_evidence(registry)
    revision = revision_evidence(registry)
    source_audits = experimental_source_audits(registry)
    experimental = experimental_map_evidence(registry)
    overlap = cross_panel_overlap_evidence(registry)
    residual_nulls = residual_null_evidence(registry)
    reusable_audit = reusable_target_map_audit_evidence(registry)
    if registry.opened_keys != set(INPUT_SPECS):
        missing = sorted(set(INPUT_SPECS) - registry.opened_keys)
        extra = sorted(registry.opened_keys - set(INPUT_SPECS))
        raise RuntimeError(f"public-core input contract mismatch; missing={missing}, extra={extra}")
    return {
        "schema_version": "4.0.0",
        "generated_by": "analysis/build_public_core_evidence.py",
        "analysis_status": "submission_public_core",
        "evidence_scope": "strict_public_core",
        "central_question": (
            "What target-relative structure remains after removing shared ligand effects "
            "from Vina matrices, how compressible is it, and does it inform fixed-target "
            "experimental panel coverage?"
        ),
        "scope_contract": {
            "row_level_sources": [
                "Docking-44", "DOCKSTRING-58", "DAVIS-Complete", "PKIS2",
                "Anastassiadis/HotSpot"
            ],
            "derived_sources": [
                "threshold and continuous molecular-weight support analyses",
                "within-source pilot recovery",
                "target-panel selector, panel-size and k=8 omitted-target controls",
                "three-panel experimental-map reliability and fixed-target panel transfer",
                "public same-support Vina--experiment geometry",
            ],
            "policy": (
                "Every central claim is reconstructable from redistributed public inputs "
                "without registration, private files, or a historical aggregate ledger."
            ),
        },
        "large_vina_panels": large_panels,
        "chemical_support_and_recovery": revision,
        "chemical_domain_controls": chemical_domain,
        "descriptor_domain_specificity": descriptor_domain,
        "scaffold_holdout_recovery": scaffold_holdout,
        "target_panel_selector_controls": target_panel_selectors,
        "hotspot_panel_validation": hotspot_validation,
        "experimental_source_audits": source_audits,
        "experimental_map_boundary": experimental,
        "cross_panel_overlap": overlap,
        "residual_nulls": residual_nulls,
        "reusable_target_map_audit": reusable_audit,
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(json_ready(payload), indent=2, sort_keys=True, allow_nan=False) + "\n"
    )


def write_outputs(
    ledger: dict[str, Any],
    registry: InputRegistry,
    output: Path = DEFAULT_OUTPUT,
    manifest: Path = DEFAULT_MANIFEST,
    manifest_digest: Path = DEFAULT_MANIFEST_DIGEST,
) -> None:
    output = output.resolve()
    manifest = manifest.resolve()
    manifest_digest = manifest_digest.resolve()
    for path in (output, manifest, manifest_digest):
        if PACKAGE.resolve() not in path.parents:
            raise PermissionError(f"public-core output escaped the package: {path}")
    write_json(output, ledger)
    rows = registry.manifest_rows()
    rows.extend(
        [
            {
                "path": str(Path(__file__).resolve().relative_to(PACKAGE)),
                "kind": "code",
                "level": "builder",
                "role": "public-core evidence builder",
                "license": "MIT",
                "bytes": Path(__file__).stat().st_size,
                "sha256": sha256_file(Path(__file__)),
            },
            {
                "path": str(output.relative_to(PACKAGE)),
                "kind": "output",
                "level": "evidence",
                "role": "strict public-core evidence ledger",
                "license": "inherits the licenses of the cited source fields",
                "bytes": output.stat().st_size,
                "sha256": sha256_file(output),
            },
        ]
    )
    frame = pd.DataFrame(rows).sort_values(["kind", "path"]).reset_index(drop=True)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(manifest, index=False, lineterminator="\n")
    manifest_digest.write_text(f"{sha256_file(manifest)}  {manifest.name}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--manifest-digest", type=Path, default=DEFAULT_MANIFEST_DIGEST)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    registry = InputRegistry()
    ledger = build(registry)
    write_outputs(ledger, registry, args.output, args.manifest, args.manifest_digest)
    print(
        json.dumps(
            {
                "evidence": str(args.output),
                "manifest": str(args.manifest),
                "opened_inputs": len(registry.opened_keys),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
