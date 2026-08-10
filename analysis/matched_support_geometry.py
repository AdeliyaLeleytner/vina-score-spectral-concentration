#!/usr/bin/env python3
"""Same-compound (matched-support) docking/experiment target-pair geometry.

Reviewer objection answered here
--------------------------------
The manuscript's external validation compares a docking target-pair geometry
estimated on the broad DOCKSTRING library against an experimental target-pair
geometry estimated on the DAVIS / PKIS2 / PKIS1 / KiRHub screening compounds.
Those are different chemical supports, and the manuscript's own strongest
negative result is that this geometry is chemical-support dependent (low- versus
high-molecular-weight maps agree at only 0.205--0.351).  A reviewer therefore
reads the reported 0.174--0.322 docking/experiment agreement as a *cross-support
association* rather than a validation.

DOCKSTRING and the two dense panels share exactly matched compounds: 59 DAVIS
ligands and 154 PKIS2 ligands match DOCKSTRING rows on the full Standard
InChIKey (frozen in ``results/dense_{davis,pkis2}_primary_molecule_mapping.csv``).
For those compounds both a docking profile and an experimental profile exist, so
the two geometries can be computed on *the same molecules*.

Design
------
For each panel the experimental endpoint is held fixed at the two-way-centred
experimental geometry of the matched compounds, and only the docking support is
varied:

``matched_support``             docking geometry from the same matched compounds;
``broad_support``               docking geometry from the de-leaked DOCKSTRING
                                reference (the manuscript's estimator);
``propensity_matched_support``  docking geometry from a k-nearest-neighbour
                                molecular-weight / cLogP / TPSA matched draw of
                                the broad pool (middle option, with standardized
                                mean differences before and after matching);
``size_matched_random_support`` random draws of the same size from the broad
                                pool, separating "N" from "chemical support".

A ``manuscript_reference`` arm reproduces the published broad-docking versus
full-panel-experimental value so the new numbers sit on the published scale.

Because N is 59 and 154, the script also reports (a) 5,000-replicate paired
Bayesian bootstrap intervals resampling ligands, Murcko scaffolds and Butina
clusters, with a conservative union of the two chemical-cluster intervals,
(b) OAS covariance shrinkage of the small-sample target correlation matrices,
and (c) split-half and subsample reliabilities that quantify how far sampling
noise at these N attenuates the raw estimate.

Outputs are byte-deterministic: no timings, no timestamps, no absolute paths.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

try:  # Package-style import and direct execution are both supported.
    from . import build_evidence as evidence
    from . import calibration_panel_recovery as calibration
    from . import dense_davis_benchmark as davis
    from . import dense_pkis2_benchmark as pkis2
    from . import residual_target_geometry_validation as geometry
except ImportError:  # pragma: no cover - direct execution path
    import build_evidence as evidence  # type: ignore
    import calibration_panel_recovery as calibration  # type: ignore
    import dense_davis_benchmark as davis  # type: ignore
    import dense_pkis2_benchmark as pkis2  # type: ignore
    import residual_target_geometry_validation as geometry  # type: ignore


PACKAGE = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PACKAGE / "results" / "matched_support_geometry"
DEFAULT_IDENTITY_CONTRACT = (
    PACKAGE / "data/frozen/dockstring_identity_contract_2026-08-03.csv.gz"
)
DEFAULT_PUBLISHED_DECOMPOSITION = (
    PACKAGE / "results/fixed20_estimand_decomposition/decomposition.csv"
)
DAVIS_MAPPING = PACKAGE / "results/dense_davis_primary_molecule_mapping.csv"
PKIS2_MAPPING = PACKAGE / "results/dense_pkis2_primary_molecule_mapping.csv"

DEFAULT_SEED = 20260806
DEFAULT_BOOTSTRAPS = 5_000
DEFAULT_QAP_PERMUTATIONS = 49_999
DEFAULT_RELIABILITY_REPEATS = 500
DEFAULT_RANDOM_SUPPORT_REPEATS = 200

# Seed offsets keep every stochastic component independent and reproducible.
PANEL_SEED_STRIDE = 1_000_000
QAP_SEED_OFFSET = 0
BOOTSTRAP_SEED_OFFSET = 100_000
SPLIT_HALF_SEED_OFFSET = 200_000
SUBSAMPLE_SEED_OFFSET = 300_000
RANDOM_SUPPORT_SEED_OFFSET = 400_000
SENSITIVITY_SEED_OFFSET = 500_000

# The locked 20-kinase panel of the manuscript's fixed-20 endpoint (190 pairs).
TARGETS = tuple(geometry.PKIS1_TARGET_MAP)
# The dense benchmarks additionally share PTK2/FAK; the 21-kinase panel (210
# pairs) is reported as a sensitivity because it is not the published endpoint.
SENSITIVITY_TARGETS = tuple(davis.TARGET_MAP)
if tuple(pkis2.TARGET_MAP) != SENSITIVITY_TARGETS:
    raise RuntimeError("the dense DAVIS and PKIS2 target maps no longer agree")
if not set(TARGETS).issubset(SENSITIVITY_TARGETS):
    raise RuntimeError("the fixed-20 endpoint is no longer a subset of the dense panel")

MATCHING_DESCRIPTORS = ("molecular_weight", "clogp", "tpsa")
NEIGHBOUR_COUNTS = (20, 50, 100)
PRIMARY_NEIGHBOURS = 100
PRIMARY_TRANSFORM = "center_then_correlation"
BOOTSTRAP_UNITS = ("ligand", "murcko_scaffold", "butina_cluster")
OUTPUT_FILES = (
    "README.md",
    "arms.csv",
    "matched_ligands.csv",
    "propensity_matching.csv",
    "sampling_noise.csv",
    "summary.json",
    "support_contrast_bootstrap.csv",
)


def json_ready(value: Any) -> Any:
    """Recursively convert numpy scalars/arrays so json.dumps stays deterministic."""
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return [json_ready(item) for item in value.tolist()]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def pair_count(targets: Sequence[str]) -> int:
    return len(targets) * (len(targets) - 1) // 2


# --------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------


def load_dockstring_support(
    dockstring_path: Path = davis.DEFAULT_DOCKSTRING,
    identity_contract: Path = DEFAULT_IDENTITY_CONTRACT,
) -> pd.DataFrame:
    """Complete 260,060-row DOCKSTRING support with the frozen identity contract.

    Row identity comes from the frozen contract rather than a fresh 260k-molecule
    InChIKey scan; the contract records the released row index, so alignment is
    verified before use.  ``raw_inchikey`` is the same identity definition the
    dense benchmarks use for their exact full-InChIKey matches.
    """
    source = pd.read_csv(dockstring_path, sep="\t")
    score_columns = [
        column for column in source.columns if column not in {"inchikey", "smiles"}
    ]
    complete = ~source[score_columns].isna().any(axis=1)
    frame = source.loc[complete, ["inchikey", "smiles", *SENSITIVITY_TARGETS]].copy()
    frame.insert(0, "source_row_index", frame.index.to_numpy(dtype=np.int64))
    frame = frame.reset_index(drop=True)
    if len(frame) != 260_060 or frame[list(SENSITIVITY_TARGETS)].isna().any().any():
        raise ValueError(
            "expected the complete 260,060-row DOCKSTRING support, observed "
            f"{len(frame):,} rows"
        )
    contract = pd.read_csv(
        identity_contract,
        usecols=[
            "source_row_index",
            "complete_support_row_index",
            "raw_inchikey",
            "raw_connectivity",
        ],
    )
    if len(contract) != len(frame):
        raise ValueError("frozen DOCKSTRING identity contract has unexpected support")
    if not np.array_equal(
        contract.source_row_index.to_numpy(dtype=np.int64),
        frame.source_row_index.to_numpy(dtype=np.int64),
    ):
        raise ValueError("identity contract does not align to released DOCKSTRING rows")
    if not np.array_equal(
        contract.complete_support_row_index.to_numpy(dtype=np.int64),
        np.arange(len(contract), dtype=np.int64),
    ):
        raise ValueError("identity contract complete-support indices are not contiguous")
    frame["standard_inchikey"] = contract.raw_inchikey.to_numpy(dtype=object)
    frame["connectivity_block"] = contract.raw_connectivity.to_numpy(dtype=object)
    return frame


def load_davis_panel() -> pd.DataFrame:
    """Dense DAVIS 72 x 21 pKd block with exact molecular identity."""
    raw = pd.read_csv(
        davis.DEFAULT_DAVIS,
        sep="\t",
        usecols=["drug_name", "protein", "compound_iso_smiles", "y"],
    )
    molecules = raw[["drug_name", "compound_iso_smiles"]].drop_duplicates()
    if molecules.drug_name.duplicated().any():
        raise ValueError("a DAVIS drug name maps to more than one released SMILES")
    selected = raw[raw.protein.isin(davis.TARGET_MAP.values())]
    matrix = selected.pivot(index="drug_name", columns="protein", values="y").reindex(
        index=molecules.drug_name,
        columns=[davis.TARGET_MAP[target] for target in SENSITIVITY_TARGETS],
    )
    matrix.columns = list(SENSITIVITY_TARGETS)
    if matrix.shape != (72, 21) or matrix.isna().any().any():
        raise ValueError(f"expected a dense DAVIS 72 x 21 block, observed {matrix.shape}")
    identity = davis._identity_table(  # noqa: SLF001
        molecules, "compound_iso_smiles", scan="full"
    )
    panel = pd.DataFrame(
        matrix.to_numpy(dtype=np.float64), columns=list(SENSITIVITY_TARGETS)
    )
    panel["standard_inchikey"] = identity.standard_inchikey.to_numpy(dtype=object)
    panel["connectivity_block"] = identity.connectivity_block.to_numpy(dtype=object)
    panel["smiles"] = identity.compound_iso_smiles.to_numpy(dtype=object)
    return panel


def load_pkis2_panel() -> pd.DataFrame:
    """Dense PKIS2 645 x 21 percent-inhibition block with exact molecular identity.

    ``geometry.load_pkis2_full`` already validates the workbook, recomputes exact
    molecular identities and renames the 21 assay columns to the DOCKSTRING
    target names, so this only reshapes its output.
    """
    frame = geometry.load_pkis2_full()
    panel = frame[list(SENSITIVITY_TARGETS)].reset_index(drop=True).astype(np.float64)
    panel["standard_inchikey"] = frame.standard_inchikey.to_numpy(dtype=object)
    panel["connectivity_block"] = frame.connectivity_block.to_numpy(dtype=object)
    panel["smiles"] = frame.Smiles.to_numpy(dtype=object)
    if len(panel) != 645 or panel[list(SENSITIVITY_TARGETS)].isna().any().any():
        raise ValueError(f"expected a dense PKIS2 645 x 21 block, observed {panel.shape}")
    return panel


def match_panel(
    panel: pd.DataFrame, dockstring: pd.DataFrame, targets: Sequence[str]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.DataFrame]:
    """Exact full-Standard-InChIKey intersection of a panel with DOCKSTRING.

    Duplicate handling follows the frozen dense benchmarks: experimental and
    docking rows sharing one key are aggregated by the median.
    """
    columns = list(targets)
    common = sorted(
        set(panel.standard_inchikey.dropna()) & set(dockstring.standard_inchikey.dropna())
    )
    if not common:
        raise ValueError("no exact molecular overlap between the panel and DOCKSTRING")
    panel_rows = panel[panel.standard_inchikey.isin(common)]
    dock_rows = dockstring[dockstring.standard_inchikey.isin(common)]
    experimental = panel_rows.groupby("standard_inchikey", sort=True)[columns].median()
    docking = (
        dock_rows.groupby("standard_inchikey", sort=True)[columns]
        .median()
        .reindex(experimental.index)
    )
    smiles = (
        panel_rows.groupby("standard_inchikey", sort=True)["smiles"]
        .first()
        .reindex(experimental.index)
    )
    provenance = pd.DataFrame(
        {
            "standard_inchikey": experimental.index.to_numpy(dtype=object),
            "panel_rows": panel_rows.groupby("standard_inchikey", sort=True)
            .size()
            .reindex(experimental.index)
            .to_numpy(dtype=np.int64),
            "dockstring_rows": dock_rows.groupby("standard_inchikey", sort=True)
            .size()
            .reindex(experimental.index)
            .to_numpy(dtype=np.int64),
            "dockstring_source_row_indices": dock_rows.groupby(
                "standard_inchikey", sort=True
            )["source_row_index"]
            .apply(lambda values: ";".join(str(int(value)) for value in sorted(values)))
            .reindex(experimental.index)
            .to_numpy(dtype=object),
        }
    )
    if docking.isna().any().any() or experimental.isna().any().any():
        raise ValueError("a matched panel x DOCKSTRING surface is not dense")
    return docking, experimental, smiles, provenance


def verify_released_mapping(matched_keys: Sequence[str], mapping_path: Path) -> dict:
    """Confirm the recomputed match equals the frozen dense-benchmark mapping."""
    released = pd.read_csv(mapping_path)
    released_keys = sorted(set(released.standard_inchikey.astype(str)))
    if released_keys != sorted(str(key) for key in matched_keys):
        raise ValueError(
            f"recomputed exact match disagrees with the frozen mapping {mapping_path.name}"
        )
    return {
        "frozen_mapping": str(mapping_path.relative_to(PACKAGE)),
        "frozen_mapping_sha256": geometry.sha256_file(mapping_path),
        "frozen_mapping_ligands": int(len(released_keys)),
        "recomputed_match_equals_frozen_mapping": True,
    }


# --------------------------------------------------------------------------
# Geometry arms
# --------------------------------------------------------------------------


def clipped(matrix: np.ndarray) -> np.ndarray:
    """Primary preprocessing of released Vina scores: positive scores clipped to 0."""
    return np.minimum(np.asarray(matrix, dtype=np.float64), 0.0)


def arm_record(
    panel: str,
    arm: str,
    docking_support: str,
    docking_rows: int,
    transform: str,
    endpoint: str,
    endpoint_rows: int,
    docking_geometry: np.ndarray,
    experimental_geometry: np.ndarray,
    permutations: int,
    seed: int,
    targets: Sequence[str] = TARGETS,
) -> dict:
    qap = geometry.qap_test(docking_geometry, experimental_geometry, permutations, seed)
    return {
        "panel": panel,
        "arm": arm,
        "docking_support": docking_support,
        "docking_support_ligands": int(docking_rows),
        "docking_transform": transform,
        "experimental_endpoint": endpoint,
        "experimental_endpoint_ligands": int(endpoint_rows),
        "targets": len(targets),
        "target_pairs": pair_count(targets),
        "spearman": float(qap["observed_spearman"]),
        "target_label_qap_p_positive": float(qap["one_sided_p_positive"]),
        "target_label_qap_null_lower_95": float(qap["null_interval_95"][0]),
        "target_label_qap_null_upper_95": float(qap["null_interval_95"][1]),
        "target_label_qap_permutations": int(qap["permutations"]),
        "seed": int(seed),
    }


def paired_support_qap(
    broad_geometry: np.ndarray,
    matched_geometry: np.ndarray,
    experimental_geometry: np.ndarray,
    permutations: int,
    seed: int,
) -> dict:
    """Paired target-label QAP for matched-minus-broad docking support.

    Reuses the package's fixed-endpoint paired QAP: the experimental geometry is
    never permuted and the same target-label permutation is applied to both
    docking geometries, so the contrast isolates the change of chemical support.
    """
    result = geometry.fixed_experimental_geometry_paired_qap(
        broad_geometry, matched_geometry, experimental_geometry, permutations, seed
    )
    return {
        "fixed_experimental_endpoint": result["fixed_experimental_endpoint"],
        "broad_support_concordance": result["raw_docking_concordance"],
        "matched_support_concordance": result["centered_docking_concordance"],
        "matched_minus_broad_support": result["centered_minus_raw_docking"],
        "permutations": result["permutations"],
        "one_sided_p_positive_matched_minus_broad": result["one_sided_p_positive_delta"],
        "null_interval_95": result["null_interval_95"],
        "null_median": result["null_median"],
        "seed": result["seed"],
    }


# --------------------------------------------------------------------------
# Uncertainty at small N
# --------------------------------------------------------------------------


def paired_support_bootstrap(
    docking_matrix: np.ndarray,
    experimental_matrix: np.ndarray,
    broad_geometry: np.ndarray,
    repeats: int,
    seed: int,
    cluster_labels: np.ndarray | None,
    unit: str,
    point_estimates: dict[str, float],
) -> dict:
    """Bayesian bootstrap of the matched ligands, recomputing both geometries.

    One Exp(1) weight is drawn per ligand (or per chemical cluster and inherited
    by its members) and applied to the matched docking matrix and the matched
    experimental matrix simultaneously, because they are the same molecules.  The
    broad-support docking geometry is held fixed: it is estimated from ~2.6e5
    ligands and carries negligible sampling error on this scale.

    Exp(1) weights lower the effective ligand count of every replicate, which
    attenuates a correlation estimated from the resampled matrix.  The matched
    docking geometry is re-estimated in each replicate whereas the broad one is
    not, so this attenuation is asymmetric and pushes the resampled
    matched-minus-broad contrast below its point estimate.  Both the point
    estimate and the resampled centre are therefore reported, and the interval is
    read as a width rather than as a relocated estimate.
    """
    if repeats < 1:
        raise ValueError("bootstrap requires at least one repeat")
    x_dock = np.asarray(docking_matrix, dtype=np.float64)
    x_exp = np.asarray(experimental_matrix, dtype=np.float64)
    if x_dock.shape != x_exp.shape:
        raise ValueError("matched docking and experimental matrices must align")
    rng = np.random.default_rng(seed)
    if cluster_labels is None:
        cluster_count = None
        cluster_index = None
    else:
        labels = np.asarray(cluster_labels)
        if labels.shape != (len(x_dock),):
            raise ValueError("cluster labels do not match the matched ligands")
        _, cluster_index = np.unique(labels, return_inverse=True)
        cluster_count = int(cluster_index.max() + 1)
    matched = np.empty(repeats, dtype=np.float64)
    broad = np.empty(repeats, dtype=np.float64)
    for repetition in range(repeats):
        if cluster_index is None:
            weights = rng.exponential(size=len(x_dock))
        else:
            weights = rng.exponential(size=cluster_count)[cluster_index]
        experimental_geometry = geometry.weighted_target_correlation(
            geometry.weighted_two_way_center(x_exp, weights), weights
        )
        docking_geometry = geometry.weighted_target_correlation(
            geometry.weighted_two_way_center(x_dock, weights), weights
        )
        matched[repetition] = geometry.geometry_concordance(
            docking_geometry, experimental_geometry
        )
        broad[repetition] = geometry.geometry_concordance(
            broad_geometry, experimental_geometry
        )
    delta = matched - broad
    described = {
        "matched_support_spearman": geometry.describe(matched),
        "broad_support_spearman": geometry.describe(broad),
        "matched_minus_broad_support": geometry.describe(delta),
    }
    expected = set(described)
    if set(point_estimates) != expected:
        raise ValueError("point estimates do not cover the bootstrapped statistics")
    return {
        "method": "paired Bayesian bootstrap with iid Exp(1) weights",
        "resampling_unit": unit,
        "clusters": cluster_count,
        "repeats": int(repeats),
        "seed": int(seed),
        **described,
        "point_estimate": {
            name: float(point_estimates[name]) for name in sorted(point_estimates)
        },
        "resampled_median_minus_point_estimate": {
            name: float(described[name]["median"] - point_estimates[name])
            for name in sorted(point_estimates)
        },
        "fraction_matched_above_broad": float(np.mean(delta > 0.0)),
        "resampling_bias_note": (
            "Exp(1) weights reduce each replicate's effective ligand count and the "
            "matched docking geometry is re-estimated in every replicate while the "
            "broad one is fixed, so the resampled contrast is displaced downward "
            "relative to its point estimate; read the interval as a width."
        ),
    }


def union_interval(first: Sequence[float], second: Sequence[float]) -> list[float]:
    """Conservative union of two 95% intervals, as used elsewhere in the package."""
    return [float(min(first[0], second[0])), float(max(first[1], second[1]))]


def split_half_reliability(
    matrix: np.ndarray, repeats: int, seed: int, transform: str = PRIMARY_TRANSFORM
) -> dict:
    """Spearman-Brown corrected split-half reliability of a target-pair geometry."""
    x = np.asarray(matrix, dtype=np.float64)
    if len(x) < 8:
        raise ValueError("split-half reliability requires at least eight ligands")
    rng = np.random.default_rng(seed)
    half = np.empty(repeats, dtype=np.float64)
    corrected = np.empty(repeats, dtype=np.float64)
    half_size = len(x) // 2
    for repetition in range(repeats):
        order = rng.permutation(len(x))
        first = geometry.geometry_correlation(x[order[:half_size]], transform)
        second = geometry.geometry_correlation(
            x[order[half_size : 2 * half_size]], transform
        )
        value = geometry.geometry_concordance(first, second)
        half[repetition] = value
        denominator = 1.0 + value
        corrected[repetition] = (
            float(np.clip(2.0 * value / denominator, -1.0, 1.0))
            if denominator > 1e-12
            else -1.0
        )
    return {
        "method": "random split-half of the ligand support, Spearman-Brown corrected",
        "repeats": int(repeats),
        "half_size": int(half_size),
        "seed": int(seed),
        "half_split_spearman": geometry.describe(half),
        "spearman_brown_reliability": geometry.describe(corrected),
    }


def subsample_reliability(
    reference_matrix: np.ndarray,
    reference_geometry: np.ndarray,
    subsample_size: int,
    repeats: int,
    seed: int,
    transform: str = PRIMARY_TRANSFORM,
) -> dict:
    """Agreement of an n-ligand docking geometry with the full broad-support one."""
    x = np.asarray(reference_matrix, dtype=np.float64)
    if subsample_size < 8 or subsample_size > len(x):
        raise ValueError("invalid subsample size for reference reliability")
    rng = np.random.default_rng(seed)
    values = np.empty(repeats, dtype=np.float64)
    for repetition in range(repeats):
        rows = rng.choice(len(x), size=subsample_size, replace=False)
        values[repetition] = geometry.geometry_concordance(
            geometry.geometry_correlation(x[rows], transform), reference_geometry
        )
    return {
        "method": (
            "random ligand subsamples of the broad reference scored against the "
            "full broad-support geometry"
        ),
        "subsample_size": int(subsample_size),
        "repeats": int(repeats),
        "seed": int(seed),
        "spearman_against_full_reference": geometry.describe(values),
    }


def attenuation_correction(
    observed: float, docking_reliability: float, experimental_reliability: float
) -> dict:
    """Classical disattenuation of an edge-level Spearman correlation.

    Approximate: the correction is exact for Pearson correlations of parallel
    measurements, and the 190 target-pair edges are neither independent nor
    normally distributed.  It is reported as an order-of-magnitude statement
    about how far sampling noise at N = 59 / 154 deflates the raw estimate, not
    as a corrected estimand.
    """
    product = float(docking_reliability) * float(experimental_reliability)
    if product <= 0.0:
        return {
            "status": "not_defined_for_nonpositive_reliability_product",
            "observed_spearman": float(observed),
            "docking_geometry_reliability": float(docking_reliability),
            "experimental_geometry_reliability": float(experimental_reliability),
        }
    corrected = float(observed) / float(np.sqrt(product))
    return {
        "status": "approximate",
        "observed_spearman": float(observed),
        "docking_geometry_reliability": float(docking_reliability),
        "experimental_geometry_reliability": float(experimental_reliability),
        "attenuation_factor_sqrt_reliability_product": float(np.sqrt(product)),
        "disattenuated_spearman": float(np.clip(corrected, -1.0, 1.0)),
        "disattenuated_spearman_uncapped": corrected,
    }


def shrunk_geometry(matrix: np.ndarray) -> np.ndarray:
    """OAS-shrunk target correlation of the two-way-centred surface."""
    return calibration.geometry(np.asarray(matrix, dtype=np.float64), shrink=True)


# --------------------------------------------------------------------------
# Propensity-matched broad support
# --------------------------------------------------------------------------


def descriptor_matrix(smiles: pd.Series) -> np.ndarray:
    frame = evidence.molecular_descriptor_frame(pd.Series(smiles).reset_index(drop=True))
    values = frame[list(MATCHING_DESCRIPTORS)]
    if values.isna().any().any():
        raise ValueError("a molecule in the matching support could not be described")
    return values.to_numpy(dtype=np.float64)


def propensity_matched_positions(
    pool_descriptors: np.ndarray,
    anchor_descriptors: np.ndarray,
    neighbour_counts: Sequence[int],
) -> tuple[dict[int, np.ndarray], np.ndarray, np.ndarray]:
    """k-nearest-neighbour matching in z-scored descriptor space.

    Distances are standardized by the pool's location and scale, exactly as the
    dense benchmarks' size-matched reference sensitivity does.  Pool rows may be
    reused across anchors, which is what makes the matched draw reproduce the
    anchor distribution rather than merely intersect it.
    """
    location = pool_descriptors.mean(axis=0)
    scale = pool_descriptors.std(axis=0, ddof=1)
    if np.any(scale <= 0) or not np.isfinite(scale).all():
        raise ValueError("a matching descriptor has zero or invalid pool scale")
    pool_standardized = (pool_descriptors - location) / scale
    anchor_standardized = (anchor_descriptors - location) / scale
    maximum = max(neighbour_counts)
    ordered: list[np.ndarray] = []
    for anchor in anchor_standardized:
        distance = np.square(pool_standardized - anchor).sum(axis=1)
        nearest = np.argpartition(distance, maximum - 1)[:maximum]
        nearest = nearest[np.argsort(distance[nearest], kind="stable")]
        ordered.append(nearest)
    selections = {
        count: np.concatenate([values[:count] for values in ordered])
        for count in sorted(neighbour_counts)
    }
    return selections, location, scale


def standardized_mean_differences(
    support_descriptors: np.ndarray,
    anchor_descriptors: np.ndarray,
    scale: np.ndarray,
) -> dict[str, float]:
    difference = (
        support_descriptors.mean(axis=0) - anchor_descriptors.mean(axis=0)
    ) / scale
    return {name: float(value) for name, value in zip(MATCHING_DESCRIPTORS, difference)}


def random_support_arm(
    pool_matrix: np.ndarray,
    experimental_geometry: np.ndarray,
    support_size: int,
    repeats: int,
    seed: int,
) -> dict:
    """Random broad draws of the propensity arm's size, isolating the N effect."""
    rng = np.random.default_rng(seed)
    values = np.empty(repeats, dtype=np.float64)
    for repetition in range(repeats):
        rows = rng.integers(0, len(pool_matrix), size=support_size)
        values[repetition] = geometry.geometry_concordance(
            geometry.geometry_correlation(pool_matrix[rows], PRIMARY_TRANSFORM),
            experimental_geometry,
        )
    return {
        "method": (
            "random draws with replacement from the broad pool, matched to the "
            "propensity arm's row count including its reuse"
        ),
        "support_size_with_reuse": int(support_size),
        "repeats": int(repeats),
        "seed": int(seed),
        "spearman": geometry.describe(values),
    }


# --------------------------------------------------------------------------
# Panel analysis
# --------------------------------------------------------------------------


def analyse_panel(
    panel_name: str,
    panel: pd.DataFrame,
    dockstring: pd.DataFrame,
    broad_matrix: np.ndarray,
    broad_geometries: dict[str, np.ndarray],
    broad_sensitivity_matrix: np.ndarray,
    broad_sensitivity_geometry: np.ndarray,
    pool_descriptors: np.ndarray,
    mapping_path: Path,
    published_broad_vs_full_panel: float,
    bootstraps: int,
    permutations: int,
    reliability_repeats: int,
    random_support_repeats: int,
    seed: int,
) -> tuple[dict, list[dict], list[dict], list[dict], list[dict], pd.DataFrame]:
    docking, experimental, smiles, provenance = match_panel(panel, dockstring, TARGETS)
    mapping_check = verify_released_mapping(list(experimental.index), mapping_path)

    matched_docking = clipped(docking.to_numpy(dtype=np.float64))
    matched_experimental = experimental.to_numpy(dtype=np.float64)
    full_experimental = panel[list(TARGETS)].to_numpy(dtype=np.float64)

    endpoint = geometry.geometry_correlation(matched_experimental, PRIMARY_TRANSFORM)
    full_endpoint = geometry.geometry_correlation(full_experimental, PRIMARY_TRANSFORM)
    matched_geometries = {
        PRIMARY_TRANSFORM: geometry.geometry_correlation(
            matched_docking, PRIMARY_TRANSFORM
        ),
        "raw": geometry.geometry_correlation(matched_docking, "raw"),
    }

    arms: list[dict] = []
    arm_seed = seed + QAP_SEED_OFFSET
    arm_specifications = (
        (
            "manuscript_reference",
            "broad_dockstring_deleaked",
            len(broad_matrix),
            PRIMARY_TRANSFORM,
            "centred_experimental_geometry_full_panel",
            len(full_experimental),
            broad_geometries[PRIMARY_TRANSFORM],
            full_endpoint,
        ),
        (
            "matched_support",
            "exact_inchikey_matched_compounds",
            len(matched_docking),
            PRIMARY_TRANSFORM,
            "centred_experimental_geometry_matched_compounds",
            len(matched_experimental),
            matched_geometries[PRIMARY_TRANSFORM],
            endpoint,
        ),
        (
            "broad_support",
            "broad_dockstring_deleaked",
            len(broad_matrix),
            PRIMARY_TRANSFORM,
            "centred_experimental_geometry_matched_compounds",
            len(matched_experimental),
            broad_geometries[PRIMARY_TRANSFORM],
            endpoint,
        ),
        (
            "matched_support_raw_docking",
            "exact_inchikey_matched_compounds",
            len(matched_docking),
            "raw",
            "centred_experimental_geometry_matched_compounds",
            len(matched_experimental),
            matched_geometries["raw"],
            endpoint,
        ),
        (
            "broad_support_raw_docking",
            "broad_dockstring_deleaked",
            len(broad_matrix),
            "raw",
            "centred_experimental_geometry_matched_compounds",
            len(matched_experimental),
            broad_geometries["raw"],
            endpoint,
        ),
    )
    for offset, specification in enumerate(arm_specifications):
        arms.append(arm_record(panel_name, *specification, permutations, arm_seed + offset))
    by_arm = {row["arm"]: row for row in arms}

    paired = paired_support_qap(
        broad_geometries[PRIMARY_TRANSFORM],
        matched_geometries[PRIMARY_TRANSFORM],
        endpoint,
        permutations,
        arm_seed + len(arm_specifications),
    )

    # ---- uncertainty at small N -----------------------------------------
    matched_point = geometry.geometry_concordance(
        matched_geometries[PRIMARY_TRANSFORM], endpoint
    )
    broad_point = geometry.geometry_concordance(
        broad_geometries[PRIMARY_TRANSFORM], endpoint
    )
    point_estimates = {
        "matched_support_spearman": float(matched_point),
        "broad_support_spearman": float(broad_point),
        "matched_minus_broad_support": float(matched_point - broad_point),
    }
    murcko = davis._murcko_labels(pd.Series(smiles).reset_index(drop=True))  # noqa: SLF001
    butina = davis._butina_labels(pd.Series(smiles).reset_index(drop=True))  # noqa: SLF001
    bootstrap_records: dict[str, dict] = {}
    for offset, (unit, labels) in enumerate(
        (("ligand", None), ("murcko_scaffold", murcko), ("butina_cluster", butina))
    ):
        bootstrap_records[unit] = paired_support_bootstrap(
            matched_docking,
            matched_experimental,
            broad_geometries[PRIMARY_TRANSFORM],
            bootstraps,
            seed + BOOTSTRAP_SEED_OFFSET + offset,
            labels,
            unit,
            point_estimates,
        )
    statistics = (
        "matched_support_spearman",
        "broad_support_spearman",
        "matched_minus_broad_support",
    )
    conservative = {
        statistic: union_interval(
            bootstrap_records["murcko_scaffold"][statistic]["interval_95"],
            bootstrap_records["butina_cluster"][statistic]["interval_95"],
        )
        for statistic in statistics
    }

    shrunk = {
        "estimator": "Oracle Approximating Shrinkage (OAS) on the two-way-centred surface",
        "matched_support_spearman_empirical": float(matched_point),
        "broad_support_spearman_empirical": float(broad_point),
        "matched_support_spearman_oas": float(
            geometry.geometry_concordance(
                shrunk_geometry(matched_docking), shrunk_geometry(matched_experimental)
            )
        ),
        "broad_support_spearman_oas": float(
            geometry.geometry_concordance(
                broad_geometries["oas"], shrunk_geometry(matched_experimental)
            )
        ),
    }
    shrunk["matched_support_oas_minus_empirical"] = (
        shrunk["matched_support_spearman_oas"] - shrunk["matched_support_spearman_empirical"]
    )
    shrunk["broad_support_oas_minus_empirical"] = (
        shrunk["broad_support_spearman_oas"] - shrunk["broad_support_spearman_empirical"]
    )
    shrunk["interpretation"] = (
        "OAS shrinks every off-diagonal entry by a nearly common factor, so it is "
        "close to rank preserving on the 190 edges and moves this Spearman "
        "estimate very little; the reliability analysis, not the shrinkage, is "
        "what quantifies sampling noise at these N."
    )

    docking_split = split_half_reliability(
        matched_docking, reliability_repeats, seed + SPLIT_HALF_SEED_OFFSET
    )
    experimental_split = split_half_reliability(
        matched_experimental, reliability_repeats, seed + SPLIT_HALF_SEED_OFFSET + 1
    )
    reference_subsample = subsample_reliability(
        broad_matrix,
        broad_geometries[PRIMARY_TRANSFORM],
        len(matched_docking),
        reliability_repeats,
        seed + SUBSAMPLE_SEED_OFFSET,
    )
    disattenuated = attenuation_correction(
        matched_point,
        docking_split["spearman_brown_reliability"]["median"],
        experimental_split["spearman_brown_reliability"]["median"],
    )

    # ---- propensity-matched broad support --------------------------------
    anchor_descriptors = descriptor_matrix(pd.Series(smiles))
    selections, _, scale = propensity_matched_positions(
        pool_descriptors, anchor_descriptors, NEIGHBOUR_COUNTS
    )
    propensity_rows: list[dict] = []
    propensity_arms: dict[str, dict] = {}
    pool_smd = standardized_mean_differences(pool_descriptors, anchor_descriptors, scale)
    for count in sorted(selections):
        positions = selections[count]
        matched_pool_descriptors = pool_descriptors[positions]
        matched_smd = standardized_mean_differences(
            matched_pool_descriptors, anchor_descriptors, scale
        )
        value = float(
            geometry.geometry_concordance(
                geometry.geometry_correlation(broad_matrix[positions], PRIMARY_TRANSFORM),
                endpoint,
            )
        )
        propensity_arms[str(count)] = {
            "neighbours_per_matched_ligand": int(count),
            "support_rows_with_reuse": int(len(positions)),
            "support_unique_rows": int(len(np.unique(positions))),
            "spearman": value,
            "standardized_mean_difference_before": dict(sorted(pool_smd.items())),
            "standardized_mean_difference_after": dict(sorted(matched_smd.items())),
            "maximum_absolute_standardized_mean_difference_before": float(
                max(abs(item) for item in pool_smd.values())
            ),
            "maximum_absolute_standardized_mean_difference_after": float(
                max(abs(item) for item in matched_smd.values())
            ),
        }
        for index, descriptor in enumerate(MATCHING_DESCRIPTORS):
            propensity_rows.append(
                {
                    "panel": panel_name,
                    "neighbours_per_matched_ligand": int(count),
                    "descriptor": descriptor,
                    "matched_ligand_mean": float(anchor_descriptors[:, index].mean()),
                    "broad_pool_mean": float(pool_descriptors[:, index].mean()),
                    "propensity_support_mean": float(
                        matched_pool_descriptors[:, index].mean()
                    ),
                    "broad_pool_sd": float(scale[index]),
                    "standardized_mean_difference_before": pool_smd[descriptor],
                    "standardized_mean_difference_after": matched_smd[descriptor],
                    "spearman_against_matched_endpoint": value,
                }
            )

    primary_positions = selections[PRIMARY_NEIGHBOURS]
    arms.append(
        arm_record(
            panel_name,
            "propensity_matched_support",
            f"broad_dockstring_knn_matched_k{PRIMARY_NEIGHBOURS}",
            len(primary_positions),
            PRIMARY_TRANSFORM,
            "centred_experimental_geometry_matched_compounds",
            len(matched_experimental),
            geometry.geometry_correlation(
                broad_matrix[primary_positions], PRIMARY_TRANSFORM
            ),
            endpoint,
            permutations,
            arm_seed + len(arm_specifications) + 1,
        )
    )
    by_arm["propensity_matched_support"] = arms[-1]
    random_support = random_support_arm(
        broad_matrix,
        endpoint,
        len(primary_positions),
        random_support_repeats,
        seed + RANDOM_SUPPORT_SEED_OFFSET,
    )

    # ---- 21-kinase sensitivity -------------------------------------------
    sensitivity_docking, sensitivity_experimental, _, _ = match_panel(
        panel, dockstring, SENSITIVITY_TARGETS
    )
    sensitivity_endpoint = geometry.geometry_correlation(
        sensitivity_experimental.to_numpy(dtype=np.float64), PRIMARY_TRANSFORM
    )
    sensitivity_matched_geometry = geometry.geometry_correlation(
        clipped(sensitivity_docking.to_numpy(dtype=np.float64)), PRIMARY_TRANSFORM
    )
    sensitivity_arms = [
        arm_record(
            panel_name,
            "matched_support_21_kinase_sensitivity",
            "exact_inchikey_matched_compounds",
            len(sensitivity_docking),
            PRIMARY_TRANSFORM,
            "centred_experimental_geometry_matched_compounds_21_kinases",
            len(sensitivity_experimental),
            sensitivity_matched_geometry,
            sensitivity_endpoint,
            permutations,
            seed + SENSITIVITY_SEED_OFFSET,
            SENSITIVITY_TARGETS,
        ),
        arm_record(
            panel_name,
            "broad_support_21_kinase_sensitivity",
            "broad_dockstring_deleaked",
            len(broad_sensitivity_matrix),
            PRIMARY_TRANSFORM,
            "centred_experimental_geometry_matched_compounds_21_kinases",
            len(sensitivity_experimental),
            broad_sensitivity_geometry,
            sensitivity_endpoint,
            permutations,
            seed + SENSITIVITY_SEED_OFFSET + 1,
            SENSITIVITY_TARGETS,
        ),
    ]
    arms.extend(sensitivity_arms)

    # ---- assembled panel record ------------------------------------------
    record = {
        "panel": panel_name,
        "panel_ligands": int(len(panel)),
        "matched_ligands": int(len(matched_docking)),
        "targets": len(TARGETS),
        "target_pairs": pair_count(TARGETS),
        "molecular_identity": (
            "exact full Standard InChIKey of the raw parsed molecule; duplicates "
            "aggregated by the median on both sides"
        ),
        "frozen_mapping_check": mapping_check,
        "published_broad_support_versus_full_panel_endpoint": float(
            published_broad_vs_full_panel
        ),
        "recomputed_broad_support_versus_full_panel_endpoint": float(
            by_arm["manuscript_reference"]["spearman"]
        ),
        "matched_support_spearman": float(matched_point),
        "broad_support_spearman": float(broad_point),
        "matched_minus_broad_support": float(matched_point - broad_point),
        "matched_support_raw_docking_spearman": float(
            by_arm["matched_support_raw_docking"]["spearman"]
        ),
        "broad_support_raw_docking_spearman": float(
            by_arm["broad_support_raw_docking"]["spearman"]
        ),
        "paired_target_label_qap_matched_minus_broad": paired,
        "bootstrap": bootstrap_records,
        "conservative_union_interval_95_murcko_and_butina": conservative,
        "covariance_shrinkage": shrunk,
        "sampling_noise": {
            "matched_docking_geometry_split_half": docking_split,
            "matched_experimental_geometry_split_half": experimental_split,
            "broad_reference_subsample_at_matched_n": reference_subsample,
            "attenuation_correction_of_matched_support_spearman": disattenuated,
        },
        "propensity_matched_support": {
            "matching_variables": list(MATCHING_DESCRIPTORS),
            "primary_neighbours_per_matched_ligand": PRIMARY_NEIGHBOURS,
            "neighbour_count_sensitivity": propensity_arms,
            "size_matched_random_support_control": random_support,
        },
        "twenty_one_kinase_sensitivity": {
            "targets": len(SENSITIVITY_TARGETS),
            "target_pairs": pair_count(SENSITIVITY_TARGETS),
            "matched_support_spearman": float(sensitivity_arms[0]["spearman"]),
            "broad_support_spearman": float(sensitivity_arms[1]["spearman"]),
            "matched_minus_broad_support": float(
                sensitivity_arms[0]["spearman"] - sensitivity_arms[1]["spearman"]
            ),
        },
    }

    bootstrap_rows = []
    for unit in BOOTSTRAP_UNITS:
        entry = bootstrap_records[unit]
        for statistic in statistics:
            values = entry[statistic]
            bootstrap_rows.append(
                {
                    "panel": panel_name,
                    "resampling_unit": unit,
                    "clusters": entry["clusters"],
                    "statistic": statistic,
                    "repeats": entry["repeats"],
                    "point_estimate": entry["point_estimate"][statistic],
                    "resampled_median_minus_point_estimate": entry[
                        "resampled_median_minus_point_estimate"
                    ][statistic],
                    "mean": values["mean"],
                    "median": values["median"],
                    "standard_deviation": values["standard_deviation"],
                    "lower_95": values["interval_95"][0],
                    "upper_95": values["interval_95"][1],
                    "fraction_matched_above_broad": entry["fraction_matched_above_broad"],
                    "seed": entry["seed"],
                }
            )

    def noise_row(quantity: str, entry: dict, key: str, n: int) -> dict:
        values = entry[key]
        return {
            "panel": panel_name,
            "quantity": quantity,
            "method": entry["method"],
            "n": int(n),
            "repeats": entry["repeats"],
            "mean": values["mean"],
            "median": values["median"],
            "lower_95": values["interval_95"][0],
            "upper_95": values["interval_95"][1],
            "seed": entry["seed"],
        }

    noise_rows = [
        noise_row(
            "matched_docking_geometry_split_half_reliability",
            docking_split,
            "spearman_brown_reliability",
            len(matched_docking),
        ),
        noise_row(
            "matched_experimental_geometry_split_half_reliability",
            experimental_split,
            "spearman_brown_reliability",
            len(matched_experimental),
        ),
        noise_row(
            "broad_reference_subsample_agreement_at_matched_n",
            reference_subsample,
            "spearman_against_full_reference",
            reference_subsample["subsample_size"],
        ),
        noise_row(
            "size_matched_random_broad_support_spearman",
            random_support,
            "spearman",
            random_support["support_size_with_reuse"],
        ),
    ]

    ligand_table = provenance.copy()
    ligand_table.insert(0, "panel", panel_name)
    for index, descriptor in enumerate(MATCHING_DESCRIPTORS):
        ligand_table[descriptor] = anchor_descriptors[:, index]
    ligand_table["murcko_scaffold_cluster"] = murcko
    ligand_table["butina_cluster"] = butina

    return record, arms, bootstrap_rows, propensity_rows, noise_rows, ligand_table


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------


def write_readme(output_dir: Path, summary: dict) -> None:
    panels = summary["panels"]
    lines = [
        "# Matched-support docking/experiment target-pair geometry",
        "",
        f"**Question.** {summary['question']}",
        "",
        f"Status: `{summary['status']}`. Primary endpoint: "
        f"{summary['support']['targets']} kinases, "
        f"{summary['support']['target_pairs']} target pairs.",
        "",
        "## Headline",
        "",
        "| panel | matched ligands | matched support | broad support | matched - broad "
        "| conservative 95% union (matched - broad) |",
        "|---|---|---|---|---|---|",
    ]
    for name in sorted(panels):
        record = panels[name]
        interval = record["conservative_union_interval_95_murcko_and_butina"][
            "matched_minus_broad_support"
        ]
        lines.append(
            f"| {name} | {record['matched_ligands']} | "
            f"{record['matched_support_spearman']:.4f} | "
            f"{record['broad_support_spearman']:.4f} | "
            f"{record['matched_minus_broad_support']:+.4f} | "
            f"[{interval[0]:+.4f}, {interval[1]:+.4f}] |"
        )
    lines += [
        "",
        f"Verdict: {summary['headline']['verdict']}",
        "",
        "The experimental endpoint is held fixed at the two-way-centred experimental",
        "geometry of the matched compounds; only the docking support changes between",
        "the two arms.",
        "",
        "## Files",
        "",
        "- `summary.json` - full record, seeds, claim boundary.",
        "- `arms.csv` - every geometry arm with its target-label QAP.",
        "- `support_contrast_bootstrap.csv` - paired Bayesian bootstrap by ligand,",
        "  Murcko scaffold and Butina cluster.",
        "- `propensity_matching.csv` - standardized mean differences before and after",
        "  molecular-weight / cLogP / TPSA matching of the broad support.",
        "- `sampling_noise.csv` - split-half and subsample reliabilities at these N.",
        "- `matched_ligands.csv` - the matched compounds, their DOCKSTRING rows,",
        "  matching descriptors and chemical-cluster labels.",
        "",
        "## Claim boundary",
        "",
        summary["claim_boundary"],
        "",
    ]
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_analysis(
    output_dir: Path = DEFAULT_OUTPUT,
    bootstraps: int = DEFAULT_BOOTSTRAPS,
    permutations: int = DEFAULT_QAP_PERMUTATIONS,
    reliability_repeats: int = DEFAULT_RELIABILITY_REPEATS,
    random_support_repeats: int = DEFAULT_RANDOM_SUPPORT_REPEATS,
    seed: int = DEFAULT_SEED,
) -> dict:
    if min(bootstraps, permutations, reliability_repeats, random_support_repeats) < 1:
        raise ValueError("all repetition counts must be positive")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dockstring = load_dockstring_support()
    panels = {
        "DAVIS": (load_davis_panel(), DAVIS_MAPPING),
        "PKIS2": (load_pkis2_panel(), PKIS2_MAPPING),
    }

    # De-leak the broad reference against both dense panels, matching the
    # manuscript's chemical-identity exclusion rule.
    excluded_blocks: set[str] = set()
    for panel, _ in panels.values():
        excluded_blocks |= set(panel.connectivity_block.dropna().astype(str))
    keep = ~dockstring.connectivity_block.astype(str).isin(excluded_blocks).to_numpy(bool)
    broad_pool = dockstring.loc[keep].reset_index(drop=True)
    broad_matrix = clipped(broad_pool[list(TARGETS)].to_numpy(dtype=np.float64))
    broad_sensitivity_matrix = clipped(
        broad_pool[list(SENSITIVITY_TARGETS)].to_numpy(dtype=np.float64)
    )
    if len(broad_matrix) < 259_000:
        raise ValueError("unexpectedly many DOCKSTRING rows were excluded")
    broad_geometries = {
        PRIMARY_TRANSFORM: geometry.geometry_correlation(broad_matrix, PRIMARY_TRANSFORM),
        "raw": geometry.geometry_correlation(broad_matrix, "raw"),
        "oas": shrunk_geometry(broad_matrix),
    }
    broad_sensitivity_geometry = geometry.geometry_correlation(
        broad_sensitivity_matrix, PRIMARY_TRANSFORM
    )
    pool_descriptors = descriptor_matrix(broad_pool.smiles)

    published = pd.read_csv(DEFAULT_PUBLISHED_DECOMPOSITION).set_index("panel")

    panel_records: dict[str, dict] = {}
    arm_rows: list[dict] = []
    bootstrap_rows: list[dict] = []
    propensity_rows: list[dict] = []
    noise_rows: list[dict] = []
    ligand_tables: list[pd.DataFrame] = []
    for offset, name in enumerate(sorted(panels)):
        panel, mapping_path = panels[name]
        (
            record,
            arms,
            panel_bootstraps,
            panel_propensity,
            panel_noise,
            ligand_table,
        ) = analyse_panel(
            name,
            panel,
            dockstring,
            broad_matrix,
            broad_geometries,
            broad_sensitivity_matrix,
            broad_sensitivity_geometry,
            pool_descriptors,
            mapping_path,
            float(published.loc[name, "centered_docking_centered_experiment"]),
            bootstraps,
            permutations,
            reliability_repeats,
            random_support_repeats,
            seed + offset * PANEL_SEED_STRIDE,
        )
        panel_records[name] = record
        arm_rows.extend(arms)
        bootstrap_rows.extend(panel_bootstraps)
        propensity_rows.extend(panel_propensity)
        noise_rows.extend(panel_noise)
        ligand_tables.append(ligand_table)

    pd.DataFrame(arm_rows).to_csv(output_dir / "arms.csv", index=False)
    pd.DataFrame(bootstrap_rows).to_csv(
        output_dir / "support_contrast_bootstrap.csv", index=False
    )
    pd.DataFrame(propensity_rows).to_csv(
        output_dir / "propensity_matching.csv", index=False
    )
    pd.DataFrame(noise_rows).to_csv(output_dir / "sampling_noise.csv", index=False)
    pd.concat(ligand_tables, ignore_index=True).to_csv(
        output_dir / "matched_ligands.csv", index=False
    )

    deltas = {
        name: record["matched_minus_broad_support"]
        for name, record in panel_records.items()
    }
    resolved = {}
    for name, record in panel_records.items():
        lower, upper = record["conservative_union_interval_95_murcko_and_butina"][
            "matched_minus_broad_support"
        ]
        resolved[name] = bool(lower > 0.0 or upper < 0.0)
    summary = {
        "analysis": "matched_support_geometry",
        "question": (
            "When the docking target-pair geometry and the experimental "
            "target-pair geometry are computed on exactly the same compounds, is "
            "their agreement higher, lower or indistinguishable from the "
            "cross-support agreement the manuscript reports?"
        ),
        "status": "exploratory_post_hoc_reviewer_requested",
        "support": {
            "targets": len(TARGETS),
            "target_names": list(TARGETS),
            "target_pairs": pair_count(TARGETS),
            "sensitivity_targets": len(SENSITIVITY_TARGETS),
            "sensitivity_target_pairs": pair_count(SENSITIVITY_TARGETS),
            "broad_reference_ligands": int(len(broad_matrix)),
            "broad_reference_definition": (
                "complete 260,060-row DOCKSTRING support minus every row whose raw "
                "InChIKey connectivity block occurs in DAVIS or PKIS2"
            ),
            "panels": {
                name: {
                    "panel_ligands": record["panel_ligands"],
                    "matched_ligands": record["matched_ligands"],
                }
                for name, record in sorted(panel_records.items())
            },
        },
        "seeds": {
            "base_seed": int(seed),
            "panel_seed_stride": PANEL_SEED_STRIDE,
            "qap_seed_offset": QAP_SEED_OFFSET,
            "bootstrap_seed_offset": BOOTSTRAP_SEED_OFFSET,
            "split_half_seed_offset": SPLIT_HALF_SEED_OFFSET,
            "subsample_seed_offset": SUBSAMPLE_SEED_OFFSET,
            "random_support_seed_offset": RANDOM_SUPPORT_SEED_OFFSET,
            "sensitivity_seed_offset": SENSITIVITY_SEED_OFFSET,
        },
        "configuration": {
            "bootstrap_replicates": int(bootstraps),
            "bootstrap_units": list(BOOTSTRAP_UNITS),
            "target_label_qap_permutations": int(permutations),
            "reliability_repeats": int(reliability_repeats),
            "random_support_repeats": int(random_support_repeats),
            "primary_transform": PRIMARY_TRANSFORM,
            "score_preprocessing": "positive Vina scores clipped to zero",
            "matching_variables": list(MATCHING_DESCRIPTORS),
            "neighbour_counts": list(NEIGHBOUR_COUNTS),
            "primary_neighbours": PRIMARY_NEIGHBOURS,
            "published_reference_table": str(
                DEFAULT_PUBLISHED_DECOMPOSITION.relative_to(PACKAGE)
            ),
        },
        "headline": {
            "matched_minus_broad_support": dict(sorted(deltas.items())),
            "resolved_by_conservative_cluster_bootstrap_union": dict(sorted(resolved.items())),
            "verdict": (
                "indistinguishable: no panel's conservative chemical-cluster "
                "bootstrap interval for matched-minus-broad support excludes zero"
                if not any(resolved.values())
                else "at least one panel resolves a matched-versus-broad difference"
            ),
        },
        "panels": panel_records,
        "claim_boundary": (
            "This does NOT show that the docking geometry is target-resolved, that "
            "it is mechanistically informative, or that it ranks compounds. It "
            "compares two estimators of the same docking target-pair geometry "
            "against one fixed experimental endpoint and asks only whether "
            "restricting the docking support to exactly the compounds that were "
            "screened changes the agreement. The matched arms have 59 and 154 "
            "ligands, so their intervals are wide and a real difference the size of "
            "the manuscript's own low- versus high-molecular-weight contrast is not "
            "excluded. Agreement of two target-pair maps is a network-level "
            "statement and implies nothing about ligand-level retrieval. Nothing "
            "here addresses the separate objection that the descriptor surface fits "
            "per-target regression coefficients and is therefore not target-blind. "
            "PKIS1 and KiRHub are not analysed: KiRHub publishes no structures and "
            "PKIS1 requires an external supplement that is not a frozen input of "
            "this package."
        ),
        "limitations": [
            "Matched supports of 59 and 154 ligands give wide intervals; absence of a resolved difference is not evidence of equality.",
            "The broad arm excludes the panel compounds by connectivity block while the matched arm is exactly those compounds, so the contrast mixes chemical support with in-sample versus out-of-sample support.",
            "The matched experimental endpoint is estimated from the same small support in both arms; the bootstrap is paired for that reason and its interval is not a target-population interval.",
            "The Bayesian bootstrap re-estimates the matched docking geometry in every replicate but holds the broad geometry fixed, so the resampled contrast is displaced below its point estimate; the reported displacement makes that asymmetry explicit and the interval should be read as a width.",
            "OAS shrinkage is close to rank preserving on the 190 edges, so it is a weak instrument for this question; the reliability analysis carries the sampling-noise statement instead.",
            "The attenuation correction is approximate: the 190 edges are neither independent nor normal.",
            "Propensity matching balances molecular weight, cLogP and TPSA only; unmeasured chemotype differences remain.",
            "Every arm here is post hoc and none is multiplicity adjusted.",
        ],
        "source_sha256": {
            "DOCKSTRING": geometry.sha256_file(davis.DEFAULT_DOCKSTRING),
            "DAVIS": geometry.sha256_file(davis.DEFAULT_DAVIS),
            "PKIS2": geometry.sha256_file(pkis2.DEFAULT_PKIS2),
            "dockstring_identity_contract": geometry.sha256_file(DEFAULT_IDENTITY_CONTRACT),
            "published_decomposition": geometry.sha256_file(DEFAULT_PUBLISHED_DECOMPOSITION),
            "dense_davis_primary_molecule_mapping": geometry.sha256_file(DAVIS_MAPPING),
            "dense_pkis2_primary_molecule_mapping": geometry.sha256_file(PKIS2_MAPPING),
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(json_ready(summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_readme(output_dir, summary)
    checksums = {
        name: geometry.sha256_file(output_dir / name) for name in sorted(OUTPUT_FILES)
    }
    (output_dir / "output_checksums.json").write_text(
        json.dumps(checksums, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstraps", type=int, default=DEFAULT_BOOTSTRAPS)
    parser.add_argument("--qap-permutations", type=int, default=DEFAULT_QAP_PERMUTATIONS)
    parser.add_argument(
        "--reliability-repeats", type=int, default=DEFAULT_RELIABILITY_REPEATS
    )
    parser.add_argument(
        "--random-support-repeats", type=int, default=DEFAULT_RANDOM_SUPPORT_REPEATS
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_analysis(
        args.output_dir,
        args.bootstraps,
        args.qap_permutations,
        args.reliability_repeats,
        args.random_support_repeats,
        args.seed,
    )
    print(json.dumps(json_ready(summary["headline"]), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
