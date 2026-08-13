#!/usr/bin/env python3
"""Post-hoc external validation on the public Anastassiadis kinase panel.

The producer reads Supplementary Table 3 from Anastassiadis et al. (2011)
directly from the publisher's legacy XLS file.  It never vendors the source.
The byte checksum, workbook dimensions, labels, global missingness and the two
selected-block missing cells are fail-closed.

The primary experimental endpoint is 100 minus percent remaining activity,
without clipping.  CDK2 is the per-compound mean of the cyclin-A and cyclin-E
assay contexts.  Removing the only two compounds with a missing selected cell
gives a dense 176 x 21 block.  Correlation maps are invariant to reversing the
global endpoint direction; this is checked numerically.

Raw- and fixed-21-row-centred Vina selectors are inherited from the locked
public panel-transfer contract.  Both are evaluated on the same experimental
map.  Exact C(21,k) enumeration propagates every optimum within an absolute
1e-12 tolerance and reports lexicographic, tie-average, and conservative
best-raw-minus-worst-residual contrasts.  Target-label permutation, rather
than a ligand bootstrap, is the inferential resampling unit.  Ligand split-half
and ordinary-bootstrap results are explicitly descriptive; no chemical
structures are present in the publisher workbook, so they are not
chemical-cluster-aware.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd


ANALYSIS_DIR = Path(__file__).resolve().parent
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

import experimental_map_reliability as reliability  # noqa: E402
import public_anastassiadis_identity_audit as identity_audit  # noqa: E402
import public_experimental_boundary_controls as boundary  # noqa: E402
import public_vina_panel_calibration_transfer as transfer  # noqa: E402


PACKAGE = ANALYSIS_DIR.parent
DEFAULT_DOCKSTRING = reliability.DEFAULT_DOCKSTRING
DEFAULT_OUTPUT = PACKAGE / "results" / "public_anastassiadis_panel_validation"
DEFAULT_IDENTITY = identity_audit.DEFAULT_IDENTITY
DEFAULT_IDENTITY_PROVENANCE = identity_audit.DEFAULT_PROVENANCE
SOURCE_DOI = "10.1038/nbt.2017"
SOURCE_URL = (
    "https://media.springernature.com/original/springer-static/esm/"
    "art%3A10.1038%2Fnbt.2017/MediaObjects/41587_2011_BFnbt2017_MOESM23_ESM.xls"
)
SOURCE_SHA256 = "cd756bf2b6ad541a1781508c563caf0da6da876dfb71f2546fbff02e13d98684"
SOURCE_SHEET = "Sheet1"
SOURCE_SHAPE = (303, 179)
SOURCE_BODY_SHAPE = (300, 178)
SOURCE_MISSING_CELLS = 566
TARGETS = tuple(reliability.TARGETS)
K_GRID = tuple(transfer.K_GRID)
OBJECTIVES = tuple(transfer.OBJECTIVES)
PRIMARY_CDK2_CONTEXT = "mean_cyclin_A_and_E"
DEFAULT_PERMUTATIONS = 20_000
DEFAULT_RESAMPLES = 2_000
DEFAULT_SEED = 202_610_08
EXPECTED_COMPLETE_ACTIVITY_SHA256 = (
    "b73a9d4af993bd28ec29b0a5fc44542020f31ce4079c8636f477f808f7bf747e"
)
EXPECTED_REMAINING_178_SHA256 = (
    "8c0aaae2b2fa5cb40084240a63ae3555b64727f95aa2f692ca4dd1447fec722f"
)
EXPECTED_CDK2_MEAN_SHA256 = (
    "3912762da0bda4e5dfb8e1c612f2430d18c83ae131ab0fc3ff09236b05111fc0"
)
EXPECTED_MISSING = (
    ("SB 202474", "AKT1"),
    ("VEGF Receptor 2 Kinase Inhibitor II", "MAPK14"),
)

TARGET_ALIASES: OrderedDict[str, tuple[str, ...]] = OrderedDict(
    [
        ("ABL1", ("ABL1",)),
        ("AKT1", ("AKT1",)),
        ("AKT2", ("AKT2",)),
        ("CDK2", ("CDK2/cyclin A", "CDK2/cyclin E")),
        ("CSF1R", ("FMS",)),
        ("EGFR", ("EGFR",)),
        ("FGFR1", ("FGFR1",)),
        ("IGF1R", ("IGF1R",)),
        ("JAK2", ("JAK2",)),
        ("KDR", ("KDR/VEGFR2",)),
        ("KIT", ("c-Kit",)),
        ("LCK", ("LCK",)),
        ("MAP2K1", ("MEK1",)),
        ("MAPK1", ("ERK2 MAPK1",)),
        ("MAPK14", ("P38a/MAPK14",)),
        ("MAPKAPK2", ("MAPKAPK2",)),
        ("MET", ("c-MET",)),
        ("PLK1", ("PLK1",)),
        ("PTK2", ("FAK/PTK2",)),
        ("ROCK1", ("ROCK1",)),
        ("SRC", ("c-SRC",)),
    ]
)
if tuple(TARGET_ALIASES) != TARGETS:
    raise RuntimeError("Anastassiadis and public-panel target orders diverged")

OUTPUT_FILES = (
    "README.md",
    "excluded_ligands.csv",
    "chemical_cluster_multiplier.csv",
    "identity_overlap_summary.csv",
    "joint_cross_assay_qap.csv",
    "ligand_bootstrap_descriptive.csv",
    "map_concordance.csv",
    "mapping_sensitivities.csv",
    "mapping_sensitivity_summary.csv",
    "optimal_panel_selections.csv",
    "reference_deleak_sensitivity.csv",
    "same_endpoint_panel_transfer.csv",
    "split_half_reliability.csv",
    "summary.json",
    "target_leave_one_out_map_concordance.csv",
    "target_leave_one_out_panel_transfer.csv",
    "target_mapping.csv",
)


@dataclass(frozen=True)
class ParsedSource:
    compounds: tuple[str, ...]
    cas_numbers: tuple[str, ...]
    remaining_by_cdk2_context: dict[str, np.ndarray]
    missing_records: pd.DataFrame
    source_metadata: dict[str, Any]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(matrix: np.ndarray) -> str:
    """Platform-stable hash of C-contiguous little-endian float64 values."""
    values = np.ascontiguousarray(np.asarray(matrix, dtype="<f8"))
    return hashlib.sha256(values.tobytes(order="C")).hexdigest()


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, float_format="%.15g")
    temporary.replace(path)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(json_ready(payload), indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _stripped(values: pd.Series) -> tuple[str, ...]:
    return tuple(values.astype(str).str.strip().tolist())


def parse_source_frame(frame: pd.DataFrame) -> ParsedSource:
    """Validate the exact worksheet and extract three declared CDK2 variants."""
    if frame.shape != SOURCE_SHAPE:
        raise ValueError(
            f"expected Anastassiadis worksheet shape {SOURCE_SHAPE}, observed {frame.shape}"
        )
    note = str(frame.iloc[0, 0]).strip()
    expected_note = (
        "Anastassiadis et al. NBT (2011) Supplementary Table 3: Complete "
        "pairwise kinase-compound activity dataset."
    )
    if not note.startswith(expected_note):
        raise ValueError("Anastassiadis worksheet title changed")
    if str(frame.iloc[1, 0]).strip() != "compound name:":
        raise ValueError("compound-name header changed")
    if str(frame.iloc[2, 0]).strip() != "compound CAS#:":
        raise ValueError("compound-CAS header changed")

    compounds = _stripped(frame.iloc[1, 1:])
    cas_numbers = _stripped(frame.iloc[2, 1:])
    source_targets = _stripped(frame.iloc[3:, 0])
    if not (
        len(compounds)
        == len(set(compounds))
        == len(cas_numbers)
        == len(set(cas_numbers))
        == SOURCE_BODY_SHAPE[1]
    ):
        raise ValueError("compound names or CAS numbers are not 178 unique values")
    if any(not re.fullmatch(r"\d{2,7}-\d{2}-\d", value) for value in cas_numbers):
        raise ValueError("a compound CAS number has an unexpected format")
    if len(source_targets) != SOURCE_BODY_SHAPE[0] or len(set(source_targets)) != 300:
        raise ValueError("source kinase names are not 300 unique values")

    raw_body = frame.iloc[3:, 1:]
    numeric = raw_body.apply(pd.to_numeric, errors="coerce")
    if numeric.shape != SOURCE_BODY_SHAPE:
        raise ValueError("pairwise activity body dimensions changed")
    if int(raw_body.isna().sum().sum()) != SOURCE_MISSING_CELLS:
        raise ValueError("global source missing-cell count changed")
    if int(numeric.isna().sum().sum()) != SOURCE_MISSING_CELLS:
        raise ValueError("a populated activity cell is no longer numeric")
    numeric.index = list(source_targets)
    numeric.columns = list(compounds)
    required_rows = [alias for aliases in TARGET_ALIASES.values() for alias in aliases]
    absent = [target for target in required_rows if target not in numeric.index]
    if absent:
        raise ValueError(f"required Anastassiadis kinase rows are absent: {absent}")

    contexts = OrderedDict(
        [
            ("mean_cyclin_A_and_E", ("CDK2/cyclin A", "CDK2/cyclin E")),
            ("cyclin_A_only", ("CDK2/cyclin A",)),
            ("cyclin_E_only", ("CDK2/cyclin E",)),
        ]
    )
    remaining_by_context: dict[str, np.ndarray] = {}
    missing_records: list[dict[str, Any]] = []
    for context, cdk_rows in contexts.items():
        columns: list[np.ndarray] = []
        for target in TARGETS:
            aliases = cdk_rows if target == "CDK2" else TARGET_ALIASES[target]
            # skipna=False ensures an assay-context missing value cannot be hidden.
            values = numeric.loc[list(aliases)].mean(axis=0, skipna=False)
            columns.append(values.to_numpy(dtype=np.float64))
        remaining = np.ascontiguousarray(np.column_stack(columns), dtype="<f8")
        observed_missing = tuple(
            (compounds[row], TARGETS[column])
            for row, column in np.argwhere(~np.isfinite(remaining))
        )
        if observed_missing != EXPECTED_MISSING:
            raise ValueError(
                f"{context}: selected-block missing cells changed: {observed_missing}"
            )
        remaining_by_context[context] = remaining
        for row, column in np.argwhere(~np.isfinite(remaining)):
            missing_records.append(
                {
                    "cdk2_context": context,
                    "source_compound_column_1_based": int(row + 2),
                    "compound": compounds[row],
                    "cas_number": cas_numbers[row],
                    "canonical_target": TARGETS[column],
                    "source_target_alias": TARGET_ALIASES[TARGETS[column]][0],
                    "exclusion_rule": "remove ligand from dense complete-case analysis",
                }
            )

    primary = remaining_by_context[PRIMARY_CDK2_CONTEXT]
    if array_sha256(primary) != EXPECTED_REMAINING_178_SHA256:
        raise ValueError("raw-XLS 178 x 21 remaining-activity matrix hash changed")
    if array_sha256(primary[:, TARGETS.index("CDK2")]) != EXPECTED_CDK2_MEAN_SHA256:
        raise ValueError("raw-XLS mean CDK2 vector hash changed")
    complete = np.ascontiguousarray((100.0 - primary)[np.isfinite(primary).all(axis=1)])
    if complete.shape != (176, 21):
        raise ValueError(f"expected complete 176 x 21 block, observed {complete.shape}")
    if array_sha256(complete) != EXPECTED_COMPLETE_ACTIVITY_SHA256:
        raise ValueError("raw-XLS complete activity matrix hash changed")

    return ParsedSource(
        compounds=compounds,
        cas_numbers=cas_numbers,
        remaining_by_cdk2_context=remaining_by_context,
        missing_records=pd.DataFrame.from_records(missing_records),
        source_metadata={
            "worksheet_shape": list(SOURCE_SHAPE),
            "pairwise_body_shape": list(SOURCE_BODY_SHAPE),
            "pairwise_body_observed_cells": int(np.isfinite(numeric).sum().sum()),
            "pairwise_body_missing_cells": SOURCE_MISSING_CELLS,
            "compound_names": len(compounds),
            "unique_cas_numbers": len(set(cas_numbers)),
            "source_kinase_rows": len(source_targets),
            "parser": "pandas read_excel engine=xlrd; direct legacy XLS/BIFF parse",
            "office_conversion_used": False,
        },
    )


def load_source(path: Path) -> ParsedSource:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Anastassiadis source XLS does not exist: {source}")
    observed_sha = sha256_file(source)
    if observed_sha != SOURCE_SHA256:
        raise ValueError(
            "Anastassiadis source SHA-256 mismatch: "
            f"expected {SOURCE_SHA256}, observed {observed_sha}"
        )
    workbook = pd.ExcelFile(source, engine="xlrd")
    if workbook.sheet_names != [SOURCE_SHEET]:
        raise ValueError(f"expected sole worksheet {SOURCE_SHEET!r}")
    frame = pd.read_excel(source, sheet_name=SOURCE_SHEET, header=None, engine="xlrd")
    return parse_source_frame(frame)


def activity_matrix(
    source: ParsedSource,
    *,
    cdk2_context: str,
    support: str,
    clipping: str,
) -> np.ndarray:
    if cdk2_context not in source.remaining_by_cdk2_context:
        raise ValueError(f"unknown CDK2 context: {cdk2_context}")
    remaining = source.remaining_by_cdk2_context[cdk2_context]
    activity = 100.0 - remaining
    if clipping == "clip_activity_to_0_100":
        activity = np.clip(activity, 0.0, 100.0)
    elif clipping != "unclipped_activity":
        raise ValueError(f"unknown clipping rule: {clipping}")
    if support == "complete_176":
        activity = activity[np.isfinite(activity).all(axis=1)]
    elif support == "target_median_imputed_178":
        activity = activity.copy()
        for column in range(activity.shape[1]):
            missing = ~np.isfinite(activity[:, column])
            if missing.any():
                activity[missing, column] = np.nanmedian(activity[:, column])
    else:
        raise ValueError(f"unknown support rule: {support}")
    values = np.ascontiguousarray(activity, dtype=np.float64)
    expected_rows = 176 if support == "complete_176" else 178
    if values.shape != (expected_rows, 21) or not np.isfinite(values).all():
        raise ValueError("activity variant is not the expected dense matrix")
    return values


def load_identity_contract(
    source: ParsedSource,
    identity_path: Path,
    provenance_path: Path,
) -> tuple[pd.DataFrame, set[str], np.ndarray, dict[str, Any]]:
    """Align the frozen PubChem crosswalk and build conservative de-leak keys.

    Resolved compounds contribute their audited RDKit connectivity block.
    Unresolved compounds contribute the union of every plausible PubChem
    candidate block.  No unresolved evaluation compound may silently remain in
    the Vina calibration pool.
    """
    identity_path = Path(identity_path)
    provenance_path = Path(provenance_path)
    if not identity_path.is_file() or not provenance_path.is_file():
        raise FileNotFoundError(
            "the frozen Anastassiadis PubChem identity CSV and provenance JSON "
            "are required for conservative three-panel de-leakage"
        )
    identity = identity_audit.validate_frozen_identity(pd.read_csv(identity_path))
    normalized_source_names = tuple(
        identity_audit.normalize_label(value) for value in source.compounds
    )
    if tuple(identity.workbook_compound_name.astype(str)) != normalized_source_names:
        raise ValueError("identity compound names do not align to the source workbook")
    if tuple(identity.workbook_cas.astype(str)) != source.cas_numbers:
        raise ValueError("identity CAS numbers do not align to the source workbook")
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if provenance.get("official_workbook_sha256") != SOURCE_SHA256:
        raise ValueError("identity provenance does not point to the frozen source XLS")

    complete = identity.loc[identity.complete_fixed21].copy().reset_index(drop=True)
    if len(complete) != 176 or int(complete.structure_usable.sum()) != 155:
        raise ValueError("expected 176 identity rows with 155 usable structures")
    de_leakage_blocks: set[str] = set()
    for value in complete.de_leakage_connectivity_blocks.fillna("").astype(str):
        blocks = [item.strip() for item in value.split(";") if item.strip()]
        if not blocks or any(len(item) != 14 for item in blocks):
            raise ValueError("an identity row lacks valid conservative de-leakage blocks")
        de_leakage_blocks.update(blocks)

    usable, cluster_metadata = identity_audit.chemical_group_assignments(identity)
    cluster_lookup = dict(
        zip(
            usable.workbook_compound_index_1based.astype(int),
            usable.butina_r2_2048_tanimoto50_cluster.astype(str),
        )
    )
    cluster_labels = np.asarray(
        [
            cluster_lookup.get(
                int(row.workbook_compound_index_1based),
                f"UNRESOLVED_{int(row.workbook_compound_index_1based):03d}",
            )
            for row in complete.itertuples(index=False)
        ],
        dtype=object,
    )
    if len(cluster_labels) != 176 or len(np.unique(cluster_labels)) != 149:
        raise ValueError("expected 149 whole-panel clusters including 21 singletons")
    metadata = {
        "identity_csv_sha256": sha256_file(identity_path),
        "identity_provenance_sha256": sha256_file(provenance_path),
        "complete_fixed21_records": len(complete),
        "usable_structures": int(complete.structure_usable.sum()),
        "unresolved_records_as_singleton_clusters": int(
            (~complete.structure_usable).sum()
        ),
        "resolved_butina_clusters_on_complete_support": int(
            len(set(cluster_labels) - {x for x in cluster_labels if str(x).startswith("UNRESOLVED_")})
        ),
        "whole_panel_clusters": int(len(np.unique(cluster_labels))),
        "conservative_de_leakage_connectivity_blocks": len(de_leakage_blocks),
        "de_leakage_rule": (
            "resolved structures contribute audited RDKit connectivity; each "
            "unresolved record contributes the union of all plausible PubChem "
            "candidate connectivities"
        ),
        "cluster_contract": cluster_metadata["butina_contract"],
    }
    return complete, de_leakage_blocks, cluster_labels, metadata


def build_vina_references(
    dockstring: pd.DataFrame,
    public_panels: dict[str, pd.DataFrame],
    hotspot_connectivities: set[str],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Primary three-panel and locked two-panel de-leaked Vina references."""
    locked, locked_metadata = reliability.broad_reference(dockstring, public_panels)
    existing_excluded: set[str] = set()
    for panel in public_panels.values():
        existing_excluded.update(panel.connectivity_block.dropna().astype(str))
    connectivity = dockstring.connectivity_block.astype(str)
    locked_keep = ~connectivity.isin(existing_excluded)
    hotspot_match = connectivity.isin(hotspot_connectivities)
    primary_keep = locked_keep & ~hotspot_match
    primary = np.minimum(
        dockstring.loc[primary_keep, list(TARGETS)].to_numpy(dtype=np.float64),
        0.0,
    )
    if len(locked) != int(locked_keep.sum()) or primary.shape[1] != len(TARGETS):
        raise RuntimeError("Vina de-leakage support did not align")
    if len(primary) >= len(locked) or int((locked_keep & hotspot_match).sum()) < 1:
        raise ValueError("HotSpot conservative connectivity exclusion removed no rows")
    metadata = {
        "primary_definition": (
            "complete DOCKSTRING support after excluding the union of DAVIS, "
            "PKIS2, and conservative HotSpot connectivity blocks"
        ),
        "primary_rows": int(len(primary)),
        "primary_matrix_sha256": array_sha256(primary),
        "locked_DAVIS_PKIS2_only_rows": int(len(locked)),
        "locked_DAVIS_PKIS2_only_matrix_sha256": array_sha256(locked),
        "locked_contract_metadata": locked_metadata,
        "hotspot_connectivity_union_size": int(len(hotspot_connectivities)),
        "hotspot_matching_rows_in_full_complete_support": int(hotspot_match.sum()),
        "additional_rows_removed_after_DAVIS_PKIS2_exclusion": int(
            (locked_keep & hotspot_match).sum()
        ),
        "total_rows_removed_from_complete_support": int((~primary_keep).sum()),
        "positive_scores_clipped_to_zero": True,
        "centering_panel": (
            "the fixed 21 columns only; each ligand is row-centered within those "
            "21 targets before target correlation"
        ),
    }
    return primary, locked, metadata


def identity_overlap_summary(
    identity: pd.DataFrame,
    dockstring: pd.DataFrame,
    public_panels: dict[str, pd.DataFrame],
    hotspot_connectivities: set[str],
) -> pd.DataFrame:
    usable = identity.loc[identity.structure_usable]
    full_keys = set(usable.rdkit_standard_inchikey.dropna().astype(str))
    connectivities = set(usable.rdkit_connectivity.dropna().astype(str))
    references: OrderedDict[str, pd.DataFrame] = OrderedDict(
        [("DOCKSTRING", dockstring), *public_panels.items()]
    )
    records: list[dict[str, Any]] = []
    for panel_name, frame in references.items():
        reference_full = set(frame.standard_inchikey.dropna().astype(str))
        reference_connectivity = set(frame.connectivity_block.dropna().astype(str))
        conservative_record_overlap = 0
        for value in identity.de_leakage_connectivity_blocks.fillna("").astype(str):
            record_blocks = {item for item in value.split(";") if item}
            conservative_record_overlap += bool(record_blocks & reference_connectivity)
        records.append(
            {
                "reference_panel": panel_name,
                "hotspot_complete_records": len(identity),
                "hotspot_usable_structures": len(usable),
                "hotspot_unresolved_records": int((~identity.structure_usable).sum()),
                "exact_full_standard_inchikey_overlap_records": int(
                    usable.rdkit_standard_inchikey.astype(str).isin(reference_full).sum()
                ),
                "connectivity_overlap_records": int(
                    usable.rdkit_connectivity.astype(str).isin(reference_connectivity).sum()
                ),
                "conservative_deleak_overlap_records": int(
                    conservative_record_overlap
                ),
                "conservative_deleak_union_blocks": len(hotspot_connectivities),
                "conservative_deleak_union_blocks_present_in_reference": len(
                    hotspot_connectivities & reference_connectivity
                ),
            }
        )
    return pd.DataFrame.from_records(records)


def _experimental_map(matrix: np.ndarray, transformation: str) -> np.ndarray:
    if transformation == "row_centered_correlation":
        return boundary.target_map(matrix)
    if transformation == "raw_correlation":
        return reliability.target_correlation(matrix)
    if transformation == "column_rank_then_row_centered":
        return boundary.column_rank_target_map(matrix)
    raise ValueError(f"unknown experimental transformation: {transformation}")


def _selector_ties(
    broad_vina: np.ndarray,
) -> dict[tuple[str, int, str], np.ndarray]:
    maps = {
        "raw_vina": reliability.target_correlation(broad_vina),
        "row_centered_residual_vina": boundary.target_map(broad_vina),
    }
    ties: dict[tuple[str, int, str], np.ndarray] = {}
    for k in K_GRID:
        panels = boundary.all_panels(n_targets=len(TARGETS), k=k)
        if len(panels) != math.comb(len(TARGETS), k):
            raise RuntimeError("panel enumeration is incomplete")
        for objective in OBJECTIVES:
            for selector, correlation in maps.items():
                distribution = boundary.coverage_distribution(
                    boundary.distance_from_map(correlation, objective), panels
                )
                optimal = transfer.numerical_optimal_indices(distribution)
                ties[(objective, k, selector)] = panels[optimal].copy()
    return ties


def _losses_for_panels(distance: np.ndarray, panels: np.ndarray) -> np.ndarray:
    return np.asarray(
        [transfer.coverage_value(distance, panel) for panel in panels],
        dtype=np.float64,
    )


def _selector_record(
    distance: np.ndarray,
    panels: np.ndarray,
    raw_ties: np.ndarray,
    residual_ties: np.ndarray,
) -> dict[str, Any]:
    distribution = boundary.coverage_distribution(distance, panels)
    sorted_distribution = np.sort(distribution)
    raw = _losses_for_panels(distance, raw_ties)
    residual = _losses_for_panels(distance, residual_ties)
    raw_lex = float(raw[0])
    residual_lex = float(residual[0])
    return {
        "all_possible_panels": int(len(panels)),
        "experimental_oracle_mean_nearest_distance": float(distribution.min()),
        "raw_vina_lex_mean_nearest_distance": raw_lex,
        "residual_vina_lex_mean_nearest_distance": residual_lex,
        "lex_raw_minus_residual_loss": raw_lex - residual_lex,
        "tie_average_raw_minus_residual_loss": float(raw.mean() - residual.mean()),
        "conservative_best_raw_minus_worst_residual_loss": float(
            raw.min() - residual.max()
        ),
        "raw_optimal_tie_count": int(len(raw)),
        "residual_optimal_tie_count": int(len(residual)),
        "raw_lex_exact_fraction_all_panels_no_worse": transfer.exact_percentile(
            sorted_distribution, raw_lex
        ),
        "residual_lex_exact_fraction_all_panels_no_worse": transfer.exact_percentile(
            sorted_distribution, residual_lex
        ),
    }


def mapping_sensitivity(
    source: ParsedSource,
    broad_vina: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Mapping/range/support sensitivity on the primary signed objective."""
    ties = _selector_ties(broad_vina)
    records: list[dict[str, Any]] = []
    for cdk2_context in source.remaining_by_cdk2_context:
        for support in ("complete_176", "target_median_imputed_178"):
            for clipping in ("unclipped_activity", "clip_activity_to_0_100"):
                matrix = activity_matrix(
                    source,
                    cdk2_context=cdk2_context,
                    support=support,
                    clipping=clipping,
                )
                for transformation in (
                    "row_centered_correlation",
                    "raw_correlation",
                    "column_rank_then_row_centered",
                ):
                    correlation = _experimental_map(matrix, transformation)
                    distance = boundary.distance_from_map(
                        correlation, "signed_1_minus_r"
                    )
                    for k in K_GRID:
                        panels = boundary.all_panels(n_targets=len(TARGETS), k=k)
                        records.append(
                            {
                                "cdk2_context": cdk2_context,
                                "ligand_support": support,
                                "range_handling": clipping,
                                "experimental_transformation": transformation,
                                "objective": "signed_1_minus_r",
                                "experimental_ligands": len(matrix),
                                "targets_selected_k": k,
                                **_selector_record(
                                    distance,
                                    panels,
                                    ties[("signed_1_minus_r", k, "raw_vina")],
                                    ties[
                                        (
                                            "signed_1_minus_r",
                                            k,
                                            "row_centered_residual_vina",
                                        )
                                    ],
                                ),
                            }
                        )
    frame = pd.DataFrame.from_records(records)
    keys = [
        "cdk2_context",
        "ligand_support",
        "range_handling",
        "experimental_transformation",
        "objective",
    ]
    summaries: list[dict[str, Any]] = []
    for key, members in frame.groupby(keys, sort=False):
        ordered = members.sort_values("targets_selected_k")
        summaries.append(
            {
                **dict(zip(keys, key)),
                "k_grid": ";".join(map(str, K_GRID)),
                "mean_lex_raw_minus_residual_loss_over_k": float(
                    ordered.lex_raw_minus_residual_loss.mean()
                ),
                "mean_tie_average_raw_minus_residual_loss_over_k": float(
                    ordered.tie_average_raw_minus_residual_loss.mean()
                ),
                "mean_conservative_best_raw_minus_worst_residual_loss_over_k": float(
                    ordered.conservative_best_raw_minus_worst_residual_loss.mean()
                ),
                "all_k_lex_positive": bool(
                    (ordered.lex_raw_minus_residual_loss > 0.0).all()
                ),
                "all_k_conservative_positive": bool(
                    (
                        ordered.conservative_best_raw_minus_worst_residual_loss
                        > 0.0
                    ).all()
                ),
            }
        )
    return frame, pd.DataFrame.from_records(summaries)


def map_concordance(
    anastassiadis: np.ndarray,
    experimental_matrices: dict[str, np.ndarray],
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    transformations: OrderedDict[str, Callable[[np.ndarray], np.ndarray]] = OrderedDict(
        [
            ("row_centered_correlation", boundary.target_map),
            ("raw_correlation", reliability.target_correlation),
            ("column_rank_then_row_centered", boundary.column_rank_target_map),
        ]
    )
    for transformation, function in transformations.items():
        anast_map = function(anastassiadis)
        for panel_name in ("DAVIS", "PKIS2"):
            other = function(experimental_matrices[panel_name])
            records.append(
                {
                    "anastassiadis_transformation": transformation,
                    "comparison_panel": panel_name,
                    "targets": len(TARGETS),
                    "target_pairs": len(TARGETS) * (len(TARGETS) - 1) // 2,
                    "target_pair_spearman": boundary.map_spearman(anast_map, other),
                    "target_pair_pearson": float(
                        np.corrcoef(boundary.upper(anast_map), boundary.upper(other))[0, 1]
                    ),
                }
            )
    return pd.DataFrame.from_records(records)


def resampling_diagnostics(
    matrix: np.ndarray,
    experimental_matrices: dict[str, np.ndarray],
    *,
    repeats: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if repeats < 10:
        raise ValueError("resampling diagnostics require at least 10 repeats")
    values = np.asarray(matrix, dtype=np.float64)
    full_map = boundary.target_map(values)
    comparison_maps = {
        name: boundary.target_map(experimental_matrices[name])
        for name in ("DAVIS", "PKIS2")
    }
    split_records: list[dict[str, Any]] = []
    bootstrap_records: list[dict[str, Any]] = []
    rng = np.random.default_rng(seed)
    half = len(values) // 2
    for repetition in range(repeats):
        order = rng.permutation(len(values))
        first = boundary.target_map(values[order[:half]])
        second = boundary.target_map(values[order[half : 2 * half]])
        split_records.append(
            {
                "repetition": repetition,
                "ligands_per_half": half,
                "cross_half_target_map_spearman": boundary.map_spearman(first, second),
                "first_half_to_full_map_spearman": boundary.map_spearman(first, full_map),
                "second_half_to_full_map_spearman": boundary.map_spearman(second, full_map),
            }
        )
        selected = rng.integers(0, len(values), size=len(values))
        boot = boundary.target_map(values[selected])
        bootstrap_records.append(
            {
                "repetition": repetition,
                "sampled_ligands_with_replacement": len(values),
                "bootstrap_to_full_map_spearman": boundary.map_spearman(boot, full_map),
                "bootstrap_to_DAVIS_map_spearman": boundary.map_spearman(
                    boot, comparison_maps["DAVIS"]
                ),
                "bootstrap_to_PKIS2_map_spearman": boundary.map_spearman(
                    boot, comparison_maps["PKIS2"]
                ),
            }
        )
    split = pd.DataFrame.from_records(split_records)
    bootstrap = pd.DataFrame.from_records(bootstrap_records)
    metadata = {
        "repeats": repeats,
        "seed": seed,
        "split_unit": "individual ligand rows; disjoint equal halves",
        "bootstrap_unit": "individual ligand rows sampled with replacement",
        "inferential_role": (
            "descriptive map reliability only; neither resampling scheme is "
            "chemical-cluster-aware and neither is used for target-panel inference"
        ),
    }
    return split, bootstrap, metadata


def cluster_multiplier_sensitivity(
    matrix: np.ndarray,
    cluster_labels: np.ndarray,
    broad_vina: np.ndarray,
    *,
    repeats: int,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Positive Exp(1) Butina-cluster multiplier for the primary signed map."""
    values = np.asarray(matrix, dtype=np.float64)
    labels = np.asarray(cluster_labels, dtype=object)
    if values.shape != (176, 21) or labels.shape != (176,):
        raise ValueError("cluster multiplier requires the aligned fixed 176 x 21 block")
    unique, inverse = np.unique(labels, return_inverse=True)
    if len(unique) != 149 or sum(str(value).startswith("UNRESOLVED_") for value in unique) != 21:
        raise ValueError("cluster labels do not contain 128 resolved plus 21 singleton groups")
    ties = _selector_ties(broad_vina)
    rng = np.random.default_rng(seed)
    weights_over_k = _auc_weights()
    records: list[dict[str, Any]] = []
    for repetition in range(repeats):
        cluster_weights = rng.exponential(scale=1.0, size=len(unique))
        row_weights = cluster_weights[inverse]
        correlation = boundary.weighted_target_map(
            values, row_weights, row_centered=True
        )
        distance = boundary.distance_from_map(correlation, "signed_1_minus_r")
        contrasts = np.empty((len(K_GRID), 3), dtype=np.float64)
        for k_index, k in enumerate(K_GRID):
            raw = _losses_for_panels(
                distance, ties[("signed_1_minus_r", k, "raw_vina")]
            )
            residual = _losses_for_panels(
                distance,
                ties[
                    (
                        "signed_1_minus_r",
                        k,
                        "row_centered_residual_vina",
                    )
                ],
            )
            contrasts[k_index] = (
                raw[0] - residual[0],
                raw.mean() - residual.mean(),
                raw.min() - residual.max(),
            )
        auc = weights_over_k @ contrasts
        records.append(
            {
                "repetition": repetition,
                "source_ligands": len(values),
                "source_clusters": len(unique),
                "unresolved_singleton_clusters": 21,
                "effective_row_sample_size": float(
                    np.square(row_weights.sum()) / np.square(row_weights).sum()
                ),
                "lex_raw_minus_residual_normalized_auc_over_k": float(auc[0]),
                "tie_average_raw_minus_residual_normalized_auc_over_k": float(auc[1]),
                "conservative_best_raw_minus_worst_residual_normalized_auc_over_k": float(
                    auc[2]
                ),
            }
        )
    frame = pd.DataFrame.from_records(records)
    metrics = [
        "lex_raw_minus_residual_normalized_auc_over_k",
        "tie_average_raw_minus_residual_normalized_auc_over_k",
        "conservative_best_raw_minus_worst_residual_normalized_auc_over_k",
    ]
    metadata = {
        "repeats": repeats,
        "seed": seed,
        "resampling_scheme": (
            "independent positive Exp(1) weight per Morgan-radius-2/2048-bit "
            "Butina Tanimoto-0.5 cluster; every member inherits its cluster weight"
        ),
        "resolved_clusters": 128,
        "unresolved_singleton_clusters": 21,
        "whole_panel_clusters": 149,
        "experimental_endpoint": "unclipped activity, metric row-centered correlation",
        "objective": "signed_1_minus_r",
        "inference_boundary": (
            "descriptive chemical-support sensitivity conditional on the fixed "
            "library; target-panel inference remains the target-label permutation"
        ),
        "statistics": {
            metric: {
                **_describe(frame[metric]),
                "positive_fraction": float(np.mean(frame[metric] > 0.0)),
            }
            for metric in metrics
        },
    }
    return frame, metadata


def _auc_weights() -> np.ndarray:
    grid = np.asarray(K_GRID, dtype=np.float64)
    weights = np.zeros(len(grid), dtype=np.float64)
    intervals = np.diff(grid)
    weights[:-1] += intervals / 2.0
    weights[1:] += intervals / 2.0
    return weights / (grid[-1] - grid[0])


def joint_cross_assay_qap(
    broad_vina: np.ndarray,
    experimental_matrices: OrderedDict[str, np.ndarray],
    *,
    permutations: int,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Shared-label target QAP for per-panel and joint cross-assay AUCs."""
    if permutations < 100:
        raise ValueError("joint target-label QAP requires at least 100 permutations")
    ties = _selector_ties(broad_vina)
    panels_by_k = {
        k: boundary.all_panels(n_targets=len(TARGETS), k=k) for k in K_GRID
    }
    distances = {
        (panel_name, objective): boundary.distance_from_map(
            boundary.target_map(matrix), objective
        )
        for panel_name, matrix in experimental_matrices.items()
        for objective in OBJECTIVES
    }
    statistic_names = (
        "lex_raw_minus_residual_loss",
        "tie_average_raw_minus_residual_loss",
        "conservative_best_raw_minus_worst_residual_loss",
    )
    panel_names = tuple(experimental_matrices)
    observed = np.empty(
        (len(panel_names), len(OBJECTIVES), len(K_GRID), len(statistic_names)),
        dtype=np.float64,
    )

    def contrasts(
        distance: np.ndarray, raw_panels: np.ndarray, residual_panels: np.ndarray
    ) -> np.ndarray:
        raw = _losses_for_panels(distance, raw_panels)
        residual = _losses_for_panels(distance, residual_panels)
        return np.asarray(
            [
                raw[0] - residual[0],
                raw.mean() - residual.mean(),
                raw.min() - residual.max(),
            ],
            dtype=np.float64,
        )

    for panel_index, panel_name in enumerate(panel_names):
        for objective_index, objective in enumerate(OBJECTIVES):
            distance = distances[(panel_name, objective)]
            for k_index, k in enumerate(K_GRID):
                observed[panel_index, objective_index, k_index] = contrasts(
                    distance,
                    ties[(objective, k, "raw_vina")],
                    ties[(objective, k, "row_centered_residual_vina")],
                )

    orders = np.empty((permutations, len(TARGETS)), dtype=np.int16)
    rng = np.random.default_rng(seed)
    for repetition in range(permutations):
        orders[repetition] = rng.permutation(len(TARGETS))
    null = np.empty((permutations, *observed.shape), dtype=np.float64)
    for repetition, order in enumerate(orders):
        for panel_index, panel_name in enumerate(panel_names):
            for objective_index, objective in enumerate(OBJECTIVES):
                distance = distances[(panel_name, objective)]
                for k_index, k in enumerate(K_GRID):
                    raw = np.sort(
                        order[ties[(objective, k, "raw_vina")]], axis=1
                    )
                    residual = np.sort(
                        order[
                            ties[
                                (
                                    objective,
                                    k,
                                    "row_centered_residual_vina",
                                )
                            ]
                        ],
                        axis=1,
                    )
                    null[repetition, panel_index, objective_index, k_index] = (
                        contrasts(distance, raw, residual)
                    )

    weights = _auc_weights()
    observed_auc = np.tensordot(observed, weights, axes=([2], [0]))
    # observed_auc: panel x objective x statistic
    null_auc = np.tensordot(null, weights, axes=([3], [0]))
    # null_auc: permutation x panel x objective x statistic
    joint_observed = observed_auc.mean(axis=0)
    joint_null = null_auc.mean(axis=1)
    records: list[dict[str, Any]] = []

    def p_value(samples: np.ndarray, value: float) -> float:
        return float((1 + np.sum(samples >= value)) / (len(samples) + 1))

    for statistic_index, statistic in enumerate(statistic_names):
        for panel_index, panel_name in enumerate(panel_names):
            family_null = null_auc[:, panel_index, :, statistic_index].max(axis=1)
            family_observed = observed_auc[panel_index, :, statistic_index]
            omnibus = p_value(family_null, float(family_observed.max()))
            for objective_index, objective in enumerate(OBJECTIVES):
                value = float(observed_auc[panel_index, objective_index, statistic_index])
                records.append(
                    {
                        "scope": "per_panel",
                        "evaluation_panel": panel_name,
                        "objective": objective,
                        "statistic": statistic,
                        "normalized_trapezoid_auc_over_k": value,
                        "target_label_p_one_sided": p_value(
                            null_auc[:, panel_index, objective_index, statistic_index],
                            value,
                        ),
                        "max_over_three_objectives_fwer_p_for_this_objective": p_value(
                            family_null, value
                        ),
                        "omnibus_max_observed_over_three_objectives_p": omnibus,
                    }
                )
        family_null = joint_null[:, :, statistic_index].max(axis=1)
        family_observed = joint_observed[:, statistic_index]
        omnibus = p_value(family_null, float(family_observed.max()))
        for objective_index, objective in enumerate(OBJECTIVES):
            value = float(joint_observed[objective_index, statistic_index])
            records.append(
                {
                    "scope": "joint_mean_DAVIS_PKIS2_Anastassiadis",
                    "evaluation_panel": "DAVIS;PKIS2;Anastassiadis",
                    "objective": objective,
                    "statistic": statistic,
                    "normalized_trapezoid_auc_over_k": value,
                    "target_label_p_one_sided": p_value(
                        joint_null[:, objective_index, statistic_index], value
                    ),
                    "max_over_three_objectives_fwer_p_for_this_objective": p_value(
                        family_null, value
                    ),
                    "omnibus_max_observed_over_three_objectives_p": omnibus,
                }
            )
    metadata = {
        "permutations": permutations,
        "seed": seed,
        "permutation_orders_sha256": hashlib.sha256(orders.tobytes()).hexdigest(),
        "permutation_unit": "complete target labels on the fixed ordered 21-target panel",
        "shared_permutation": (
            "one target-label permutation is shared across DAVIS, PKIS2, "
            "Anastassiadis, every k, all objectives, and all tie statistics"
        ),
        "joint_statistic": (
            "mean across the three assay panels of the normalized trapezoid AUC "
            "over k=4,6,8,10,12"
        ),
        "multiplicity": (
            "one-sided maximum-statistic adjustment over the three objective "
            "families, reported separately for lexicographic, tie-average, and "
            "conservative best-raw-minus-worst-residual statistics"
        ),
    }
    return pd.DataFrame.from_records(records), metadata


def reference_deleak_observed_summary(
    references: OrderedDict[str, np.ndarray],
    experimental_matrices: OrderedDict[str, np.ndarray],
) -> pd.DataFrame:
    """Compact observed signed comparison of primary and locked references."""
    weights = _auc_weights()
    records: list[dict[str, Any]] = []
    for reference_name, broad_vina in references.items():
        ties = _selector_ties(broad_vina)
        panel_values = np.empty((len(experimental_matrices), len(K_GRID), 3))
        for panel_index, (panel_name, matrix) in enumerate(
            experimental_matrices.items()
        ):
            distance = boundary.distance_from_map(
                boundary.target_map(matrix), "signed_1_minus_r"
            )
            for k_index, k in enumerate(K_GRID):
                raw = _losses_for_panels(
                    distance, ties[("signed_1_minus_r", k, "raw_vina")]
                )
                residual = _losses_for_panels(
                    distance,
                    ties[
                        (
                            "signed_1_minus_r",
                            k,
                            "row_centered_residual_vina",
                        )
                    ],
                )
                panel_values[panel_index, k_index] = (
                    raw[0] - residual[0],
                    raw.mean() - residual.mean(),
                    raw.min() - residual.max(),
                )
            for statistic_index, statistic in enumerate(
                ("lex", "tie_average", "conservative")
            ):
                records.append(
                    {
                        "reference_contract": reference_name,
                        "aggregation_scope": f"{panel_name}_mean_over_k",
                        "statistic": statistic,
                        "value": float(panel_values[panel_index, :, statistic_index].mean()),
                    }
                )
                records.append(
                    {
                        "reference_contract": reference_name,
                        "aggregation_scope": f"{panel_name}_normalized_auc_over_k",
                        "statistic": statistic,
                        "value": float(weights @ panel_values[panel_index, :, statistic_index]),
                    }
                )
        joint_auc = (np.tensordot(panel_values, weights, axes=([1], [0]))).mean(
            axis=0
        )
        for statistic_index, statistic in enumerate(
            ("lex", "tie_average", "conservative")
        ):
            records.append(
                {
                    "reference_contract": reference_name,
                    "aggregation_scope": "joint_three_panel_mean_normalized_auc_over_k",
                    "statistic": statistic,
                    "value": float(joint_auc[statistic_index]),
                }
            )
    frame = pd.DataFrame.from_records(records)
    pivot = frame.pivot(
        index=["aggregation_scope", "statistic"],
        columns="reference_contract",
        values="value",
    ).reset_index()
    pivot["primary_minus_locked"] = (
        pivot["primary_triple_deleaked"] - pivot["locked_DAVIS_PKIS2_only"]
    )
    return pivot


def target_mapping_frame() -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for target, aliases in TARGET_ALIASES.items():
        records.append(
            {
                "canonical_dockstring_target": target,
                "source_aliases": ";".join(aliases),
                "primary_aggregation": (
                    "per-compound arithmetic mean of cyclin-A and cyclin-E rows"
                    if target == "CDK2"
                    else "single source row"
                ),
                "sensitivity": (
                    "cyclin-A-only and cyclin-E-only"
                    if target == "CDK2"
                    else "none"
                ),
            }
        )
    return pd.DataFrame.from_records(records)


def _describe(values: Iterable[float]) -> dict[str, float | int]:
    array = np.asarray(list(values), dtype=np.float64)
    return {
        "n": int(len(array)),
        "median": float(np.median(array)),
        "mean": float(np.mean(array)),
        "q025": float(np.quantile(array, 0.025)),
        "q975": float(np.quantile(array, 0.975)),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
    }


def target_leave_one_out_influence(
    broad_vina: np.ndarray,
    experiments: OrderedDict[str, np.ndarray],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Re-select panels and re-estimate assay maps after omitting each target.

    This is an influence analysis conditional on the observed target panel, not
    a target-superpopulation interval. Both Vina panels are reselected on each
    reduced 20-target map rather than merely deleting evaluation edges.
    """

    objective = "signed_1_minus_r"
    weights = _auc_weights()
    panel_records: list[dict[str, Any]] = []
    map_records: list[dict[str, Any]] = []
    panel_summaries: list[dict[str, Any]] = []
    names = list(experiments)

    for omitted_index, omitted_target in enumerate(TARGETS):
        kept = np.asarray(
            [index for index in range(len(TARGETS)) if index != omitted_index],
            dtype=np.int16,
        )
        reduced_vina = broad_vina[:, kept]
        raw_vina_map = reliability.target_correlation(reduced_vina)
        residual_vina_map = boundary.target_map(reduced_vina)
        experimental_raw_maps = {
            name: reliability.target_correlation(matrix[:, kept])
            for name, matrix in experiments.items()
        }
        experimental_residual_maps = {
            name: boundary.target_map(matrix[:, kept])
            for name, matrix in experiments.items()
        }

        for first_index in range(len(names)):
            for second_index in range(first_index + 1, len(names)):
                first = names[first_index]
                second = names[second_index]
                raw_agreement = reliability.map_spearman(
                    experimental_raw_maps[first], experimental_raw_maps[second]
                )
                residual_agreement = reliability.map_spearman(
                    experimental_residual_maps[first],
                    experimental_residual_maps[second],
                )
                map_records.append(
                    {
                        "omitted_target": omitted_target,
                        "panel_a": first,
                        "panel_b": second,
                        "remaining_targets": len(kept),
                        "raw_map_spearman": raw_agreement,
                        "residual_map_spearman": residual_agreement,
                        "residual_minus_raw_map_spearman": (
                            residual_agreement - raw_agreement
                        ),
                    }
                )

        target_records: list[dict[str, Any]] = []
        for k in K_GRID:
            panels = boundary.all_panels(n_targets=len(kept), k=k)
            raw_distribution = boundary.coverage_distribution(
                boundary.distance_from_map(raw_vina_map, objective), panels
            )
            residual_distribution = boundary.coverage_distribution(
                boundary.distance_from_map(residual_vina_map, objective), panels
            )
            raw_ties = panels[transfer.numerical_optimal_indices(raw_distribution)]
            residual_ties = panels[
                transfer.numerical_optimal_indices(residual_distribution)
            ]
            for assay_name in names:
                distance = boundary.distance_from_map(
                    experimental_residual_maps[assay_name], objective
                )
                raw_losses = _losses_for_panels(distance, raw_ties)
                residual_losses = _losses_for_panels(distance, residual_ties)
                contrast = float(raw_losses.min() - residual_losses.max())
                record = {
                    "omitted_target": omitted_target,
                    "evaluation_panel": assay_name,
                    "remaining_targets": len(kept),
                    "targets_selected_k": k,
                    "objective": objective,
                    "raw_optimal_tie_count": len(raw_ties),
                    "residual_optimal_tie_count": len(residual_ties),
                    "conservative_best_raw_minus_worst_residual_loss": contrast,
                    "positive_favours_residual": contrast > 0.0,
                }
                target_records.append(record)
                panel_records.append(record)

        assay_aucs = []
        for assay_name in names:
            values = np.asarray(
                [
                    record["conservative_best_raw_minus_worst_residual_loss"]
                    for record in target_records
                    if record["evaluation_panel"] == assay_name
                ],
                dtype=np.float64,
            )
            if len(values) != len(K_GRID):
                raise RuntimeError("leave-one-target-out k grid is incomplete")
            assay_aucs.append(float(values @ weights))
        contrasts = np.asarray(
            [
                record["conservative_best_raw_minus_worst_residual_loss"]
                for record in target_records
            ],
            dtype=np.float64,
        )
        panel_summaries.append(
            {
                "omitted_target": omitted_target,
                "joint_normalized_auc": float(np.mean(assay_aucs)),
                "positive_cells": int(np.sum(contrasts > 0.0)),
                "total_cells": int(len(contrasts)),
                "minimum_cell_contrast": float(contrasts.min()),
                "maximum_cell_contrast": float(contrasts.max()),
            }
        )

    panel_frame = pd.DataFrame.from_records(panel_records)
    map_frame = pd.DataFrame.from_records(map_records)
    panel_summary = pd.DataFrame.from_records(panel_summaries)
    pair_summary = []
    for (panel_a, panel_b), group in map_frame.groupby(
        ["panel_a", "panel_b"], sort=False
    ):
        pair_summary.append(
            {
                "panel_a": panel_a,
                "panel_b": panel_b,
                "omitted_targets": int(len(group)),
                "raw_map_spearman_min": float(group.raw_map_spearman.min()),
                "raw_map_spearman_max": float(group.raw_map_spearman.max()),
                "residual_map_spearman_min": float(
                    group.residual_map_spearman.min()
                ),
                "residual_map_spearman_max": float(
                    group.residual_map_spearman.max()
                ),
                "residual_exceeds_raw_for_every_omission": bool(
                    (group.residual_minus_raw_map_spearman > 0.0).all()
                ),
            }
        )
    metadata = {
        "role": (
            "fixed-panel target influence sensitivity; not a confidence interval "
            "for a target superpopulation"
        ),
        "selection_contract": (
            "for each omitted kinase, raw and residual panels are reselected by "
            "complete enumeration on the remaining 20-target Vina maps"
        ),
        "objective": objective,
        "k_grid": list(K_GRID),
        "panel_transfer": {
            "omitted_targets": int(len(panel_summary)),
            "joint_normalized_auc_min": float(
                panel_summary.joint_normalized_auc.min()
            ),
            "joint_normalized_auc_max": float(
                panel_summary.joint_normalized_auc.max()
            ),
            "joint_normalized_auc_positive_omissions": int(
                (panel_summary.joint_normalized_auc > 0.0).sum()
            ),
            "positive_cells_min": int(panel_summary.positive_cells.min()),
            "positive_cells_max": int(panel_summary.positive_cells.max()),
            "panel_summary": panel_summary.to_dict(orient="records"),
        },
        "cross_assay_map_concordance": pair_summary,
    }
    return panel_frame, map_frame, metadata


def run(
    *,
    source_path: Path,
    identity_path: Path,
    identity_provenance_path: Path,
    permutations: int,
    resamples: int,
    seed: int,
) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
    source = load_source(source_path)
    identity, hotspot_connectivities, cluster_labels, identity_metadata = (
        load_identity_contract(
            source,
            identity_path=identity_path,
            provenance_path=identity_provenance_path,
        )
    )
    primary = activity_matrix(
        source,
        cdk2_context=PRIMARY_CDK2_CONTEXT,
        support="complete_176",
        clipping="unclipped_activity",
    )
    dockstring = reliability.load_dockstring_support()
    public_panels = reliability.load_public_panels()
    broad_vina, locked_broad_vina, broad_metadata = build_vina_references(
        dockstring,
        public_panels,
        hotspot_connectivities,
    )

    experiments: OrderedDict[str, np.ndarray] = OrderedDict()
    for panel_name, panel in public_panels.items():
        values = panel[list(TARGETS)].to_numpy(dtype=np.float64)
        values = values[reliability.informative_rows(values)]
        experiments[panel_name] = values
    experiments["Anastassiadis"] = primary

    same_endpoint, selections, same_endpoint_metadata = (
        transfer.same_endpoint_centering_comparison(
            broad_vina,
            experiments,
            permutations=permutations,
            seed=seed,
        )
    )
    # The reused producer originally described its DAVIS/PKIS2 application.
    # Replace only the prose fields whose panel count changes in this extension.
    same_endpoint_metadata["permutation_contract"] = (
        "one common permutation of the 21 target labels is applied to every "
        "fixed raw- and residual-Vina selected set across DAVIS, PKIS2, and "
        "Anastassiadis, all k, and all three objectives in a replicate; "
        "experimental maps and exact all-panel distributions remain fixed"
    )
    same_endpoint_metadata["primary_statistic"] = (
        "signed-1-r conservative best-raw-optimum minus worst-residual-optimum "
        "coverage-loss improvement, averaged over the prespecified k grid and "
        "DAVIS, PKIS2, and Anastassiadis; numerical Vina optima use the declared "
        "absolute tolerance"
    )
    sensitivity, sensitivity_summary = mapping_sensitivity(source, broad_vina)
    concordance = map_concordance(primary, experiments)
    split_half, bootstrap, resampling_metadata = resampling_diagnostics(
        primary,
        experiments,
        repeats=resamples,
        seed=seed + 1_000_000,
    )
    joint_qap, joint_metadata = joint_cross_assay_qap(
        broad_vina,
        experiments,
        permutations=permutations,
        seed=seed,
    )
    cluster_multiplier, cluster_metadata = cluster_multiplier_sensitivity(
        primary,
        cluster_labels,
        broad_vina,
        repeats=resamples,
        seed=seed + 2_000_000,
    )
    target_loo_panel, target_loo_maps, target_loo_metadata = (
        target_leave_one_out_influence(broad_vina, experiments)
    )
    reference_sensitivity = reference_deleak_observed_summary(
        OrderedDict(
            [
                ("primary_triple_deleaked", broad_vina),
                ("locked_DAVIS_PKIS2_only", locked_broad_vina),
            ]
        ),
        experiments,
    )
    overlap = identity_overlap_summary(
        identity,
        dockstring,
        public_panels,
        hotspot_connectivities,
    )

    primary_map = boundary.target_map(primary)
    reversed_map = boundary.target_map(-primary)
    direction_error = float(np.max(np.abs(primary_map - reversed_map)))
    if direction_error > 1e-12:
        raise RuntimeError("experimental target geometry is not direction invariant")
    primary_rows = sensitivity_summary.loc[
        sensitivity_summary.cdk2_context.eq(PRIMARY_CDK2_CONTEXT)
        & sensitivity_summary.ligand_support.eq("complete_176")
        & sensitivity_summary.range_handling.eq("unclipped_activity")
        & sensitivity_summary.experimental_transformation.eq(
            "row_centered_correlation"
        )
    ]
    if len(primary_rows) != 1:
        raise RuntimeError("primary mapping-sensitivity row is not unique")
    primary_sensitivity = primary_rows.iloc[0].to_dict()
    joint_primary = joint_qap.loc[
        joint_qap.scope.eq("joint_mean_DAVIS_PKIS2_Anastassiadis")
        & joint_qap.objective.eq("signed_1_minus_r")
    ]

    input_paths = OrderedDict(
        [
            ("data/frozen/dockstring-dataset.tsv.gz", reliability.DEFAULT_DOCKSTRING),
            ("data/frozen/davis_complete.tab.gz", reliability.DEFAULT_DAVIS),
            ("data/frozen/pkis2_s4.xlsx", reliability.DEFAULT_PKIS2),
            (
                "data/frozen/dockstring_identity_contract_2026-08-03.csv.gz",
                reliability.DEFAULT_IDENTITY_CONTRACT,
            ),
            (
                "data/frozen/dockstring_kinase_receptors.fasta",
                transfer.SEQUENCE_FASTA,
            ),
            (
                "data/frozen/dockstring_kinase_receptors_manifest.csv",
                transfer.SEQUENCE_MANIFEST,
            ),
        ]
    )
    summary = {
        "schema_version": "1.0.0",
        "analysis": "public Anastassiadis fixed-21 panel validation",
        "status": "post_hoc_external_extension",
        "verdict": (
            "conditional_go_for_signed_row_centered_target_geometry; no_go_for_"
            "representation_invariant_or_unseen_target_inference"
        ),
        "source_contract": {
            "doi": SOURCE_DOI,
            "official_url": SOURCE_URL,
            "sha256": SOURCE_SHA256,
            "source_filename": Path(source_path).name,
            "redistributed": False,
            "access_and_license_boundary": (
                "The official supplement is publicly downloadable, but no explicit "
                "permissive dataset redistribution license was identified. The XLS "
                "is therefore checksum-fetched/read locally and is not included."
            ),
            **source.source_metadata,
        },
        "endpoint_contract": {
            "source_measurement": (
                "average of two replicates, percent remaining kinase activity "
                "relative to solvent control"
            ),
            "primary_transform": "100 - percent remaining activity; no clipping",
            "primary_cdk2": (
                "per-compound arithmetic mean of CDK2/cyclin A and CDK2/cyclin E"
            ),
            "primary_support": "176 complete ligands x 21 targets",
            "excluded_ligands": list(EXPECTED_MISSING),
            "activity_range": [float(primary.min()), float(primary.max())],
            "direction_invariance_max_abs_map_difference": direction_error,
            "complete_activity_matrix_sha256": array_sha256(primary),
            "remaining_178x21_matrix_sha256": array_sha256(
                source.remaining_by_cdk2_context[PRIMARY_CDK2_CONTEXT]
            ),
            "cdk2_mean_178_vector_sha256": array_sha256(
                source.remaining_by_cdk2_context[PRIMARY_CDK2_CONTEXT][
                    :, TARGETS.index("CDK2")
                ]
            ),
            "numeric_hash_serialization": (
                "C-contiguous little-endian IEEE-754 float64 bytes"
            ),
        },
        "targets": list(TARGETS),
        "broad_vina_contract": broad_metadata,
        "identity_contract": identity_metadata,
        "experimental_support": {
            name: {"ligands": int(len(matrix)), "targets": int(matrix.shape[1])}
            for name, matrix in experiments.items()
        },
        "primary_mapping_sensitivity_summary": json_ready(primary_sensitivity),
        "joint_primary_target_label_qap": joint_primary.to_dict(orient="records"),
        "same_endpoint_inference": same_endpoint_metadata,
        "joint_cross_assay_inference": joint_metadata,
        "chemical_cluster_multiplier": cluster_metadata,
        "target_leave_one_out_influence": target_loo_metadata,
        "resampling": {
            **resampling_metadata,
            "split_half_cross_map_spearman": _describe(
                split_half.cross_half_target_map_spearman
            ),
            "ordinary_bootstrap_to_full_map_spearman": _describe(
                bootstrap.bootstrap_to_full_map_spearman
            ),
        },
        "inputs": {
            **{name: sha256_file(path) for name, path in input_paths.items()},
            "data/frozen/anastassiadis2011_pubchem_identity_2026-08-10.csv": sha256_file(
                identity_path
            ),
            "data/frozen/anastassiadis2011_pubchem_identity_2026-08-10.provenance.json": sha256_file(
                identity_provenance_path
            ),
        },
        "claim_boundary": (
            "This is a post-hoc extension that reuses a locked fixed-21 contract. "
            "It supports the signed row-centered target-map/panel result across a "
            "different functional assay, but absolute and squared objectives are "
            "mixed and the rank sensitivity is weaker. PubChem identity resolution "
            "supports a secondary cluster multiplier with unresolved compounds as "
            "singletons; the ordinary ligand bootstrap remains non-cluster-aware. "
            "Target-label QAP is conditional on these same "
            "21 kinase identities and does not establish transfer to unseen targets."
        ),
    }
    tables = {
        "excluded_ligands.csv": source.missing_records,
        "chemical_cluster_multiplier.csv": cluster_multiplier,
        "identity_overlap_summary.csv": overlap,
        "joint_cross_assay_qap.csv": joint_qap,
        "ligand_bootstrap_descriptive.csv": bootstrap,
        "map_concordance.csv": concordance,
        "mapping_sensitivities.csv": sensitivity,
        "mapping_sensitivity_summary.csv": sensitivity_summary,
        "optimal_panel_selections.csv": selections,
        "reference_deleak_sensitivity.csv": reference_sensitivity,
        "same_endpoint_panel_transfer.csv": same_endpoint,
        "split_half_reliability.csv": split_half,
        "target_leave_one_out_map_concordance.csv": target_loo_maps,
        "target_leave_one_out_panel_transfer.csv": target_loo_panel,
        "target_mapping.csv": target_mapping_frame(),
    }
    return summary, tables


def readme_text(summary: dict[str, Any]) -> str:
    primary = summary["primary_mapping_sensitivity_summary"]
    return f"""# Public Anastassiadis panel validation

This post-hoc external extension reads the official Anastassiadis et al. 2011
Supplementary Table 3 by local `--source` path after an exact SHA-256 check. The
source XLS is **not redistributed** in this result bundle.

The primary block is 176 ligands by the fixed 21 kinases. Its endpoint is
`100 - percent remaining activity`, without clipping. CDK2 is the arithmetic
mean of its cyclin-A and cyclin-E assay rows. The only removed compounds are
`SB 202474` (missing AKT1) and `VEGF Receptor 2 Kinase Inhibitor II` (missing
MAPK14). `target_mapping.csv` and `excluded_ligands.csv` make those decisions
machine-readable.

The primary Vina reference is the conservatively triple-de-leaked
{summary['broad_vina_contract']['primary_rows']:,}-row
support. Raw and fixed-21-row-centred Vina selectors are exact C(21,k) optima at
k={list(K_GRID)}. Both selectors are evaluated on the same experimental map.
For the primary signed 1-r endpoint, the mean lexicographic raw-minus-residual
loss over k is {primary['mean_lex_raw_minus_residual_loss_over_k']:.4f}, and the
mean conservative best-raw-minus-worst-residual contrast is
{primary['mean_conservative_best_raw_minus_worst_residual_loss_over_k']:.4f}.
Positive values favour the residual selector.

`same_endpoint_panel_transfer.csv` reports exact oracle loss, exact panel
percentile, sequence-identity baseline and every tie sensitivity. The inferential
unit is the complete target label: 20,000 shared target-label permutations are
used with max-over-three-objective adjustment. `joint_cross_assay_qap.csv`
reports per-panel and joint DAVIS/PKIS2/Anastassiadis normalized AUCs over k.

`mapping_sensitivities.csv` crosses CDK2 mean/A-only/E-only, complete-case versus
target-median imputation, unclipped versus [0,100]-clipped activity, and raw,
row-centred and column-rank-then-row-centred experimental maps. Absolute and
squared objectives are sensitivities, not confirmatory substitutes.

`target_leave_one_out_panel_transfer.csv` repeats exact panel selection after
removing each kinase in turn; `target_leave_one_out_map_concordance.csv` repeats
the three cross-assay comparisons on the corresponding 20-target maps. These
are fixed-panel influence checks, not target-superpopulation intervals.

Ligand split halves and the ordinary ligand bootstrap are descriptive map
reliability diagnostics only. They are not chemical-cluster-aware because the
publisher workbook supplies compound names and CAS numbers but no structures.
The separate PubChem identity crosswalk enables a secondary cluster multiplier;
unresolved records are singleton clusters. This analysis is conditional on the
fixed 21 kinase identities and does not show transfer to unseen targets.
"""


def write_outputs(
    output: Path,
    summary: dict[str, Any],
    tables: dict[str, pd.DataFrame],
) -> dict[str, str]:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    expected_tables = set(OUTPUT_FILES) - {"README.md", "summary.json"}
    if set(tables) != expected_tables:
        raise ValueError("output table set does not match the declared bundle")
    for filename, frame in tables.items():
        write_csv(output / filename, frame)
    (output / "README.md").write_text(readme_text(summary), encoding="utf-8")
    write_json(output / "summary.json", summary)
    checksums = {
        filename: sha256_file(output / filename) for filename in OUTPUT_FILES
    }
    manifest = "".join(
        f"{digest}  {filename}\n" for filename, digest in sorted(checksums.items())
    )
    (output / "bundle_checksums.sha256").write_text(manifest, encoding="utf-8")
    return checksums


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="local official MOESM23 legacy XLS; exact SHA-256 is required",
    )
    parser.add_argument("--permutations", type=int, default=DEFAULT_PERMUTATIONS)
    parser.add_argument("--resamples", type=int, default=DEFAULT_RESAMPLES)
    parser.add_argument("--identity", type=Path, default=DEFAULT_IDENTITY)
    parser.add_argument(
        "--identity-provenance",
        type=Path,
        default=DEFAULT_IDENTITY_PROVENANCE,
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    summary, tables = run(
        source_path=args.source,
        identity_path=args.identity,
        identity_provenance_path=args.identity_provenance,
        permutations=args.permutations,
        resamples=args.resamples,
        seed=args.seed,
    )
    checksums = write_outputs(args.output, summary, tables)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "status": summary["status"],
                "verdict": summary["verdict"],
                "bundle_files": len(checksums),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
