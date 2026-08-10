#!/usr/bin/env python3
"""Sampling-uncertainty intervals for the docking-side increment against a fixed
centred experimental endpoint.

The manuscript reports the docking-side increment

    Delta_D(panel) = rho( G_ctr(D), G_ctr(E_panel) ) - rho( G_raw(D), G_ctr(E_panel) )

for four kinase panels (0.016, 0.044, 0.001, 0.045) together with one-sided paired
target-label QAP probabilities (0.335, 0.146, 0.462, 0.126) and the 2.5/97.5 quantiles
of the QAP null.  A QAP null interval is not a confidence interval.  QAP asks whether
the alignment between two *target-labelled* networks exceeds what target relabelling
produces; it holds both compound libraries fixed and therefore says nothing about the
uncertainty contributed by the finite DOCKSTRING reference or by the finite experimental
panels.  Reported alone, a null interval cannot separate

    (a) "no target-label evidence",
    (b) "large sampling uncertainty", and
    (c) "genuinely tiny effect".

This analysis supplies the missing quantity: paired nonparametric bootstrap confidence
intervals for the same Delta_D, on the same fixed-20-kinase estimand, holding every
other contract of ``fixed20_estimand_decomposition`` unchanged.

Three resampling schemes are run, each with 5,000 replicates:

``experimental_only``
    Resample the panel's compounds (whole chemical clusters where structures exist) and
    recompute that panel's centred experimental geometry.  The raw and centred docking
    geometries stay at their full-reference values, so the difference stays paired.

``docking_only``
    Resample the DOCKSTRING reference by chemical group (cyclic Bemis--Murcko scaffold;
    acyclic molecules keyed by InChI connectivity block) and recompute the raw *and* the
    within-ligand-centred docking geometry from the identical resampled support, so the
    two arms of the difference always describe the same molecules.  Every experimental
    geometry stays fixed.  A molecule-level resampling of the same reference is reported
    alongside it as the less conservative comparison.

``combined``
    Both supports resampled independently within the same replicate.

Intervals are 2.5/97.5 and 5/95 percentiles of the replicate distribution, following the
package convention in ``dense_davis_benchmark._distribution``.  Where both a Murcko and
a Butina-0.65 cluster bootstrap exist, the reported interval is their conservative union,
following ``dockstring_chembl_ranking_benchmark.conservative_union``.

No manuscript file, manifest-listed input or released result is modified; the released
190-edge ledger and the frozen decomposition summary are read only to verify that this
script reproduces the numbers it claims to bound.
"""

from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import centering_panel_sensitivity as centering
import dense_davis_benchmark as davis
import experimental_chemical_context_geometry as expchem
import fixed20_estimand_decomposition as fixed20
import kirhub_external_validation as kirhub
import replicated_pair_retrieval as retrieval
import residual_target_geometry_validation as geometry


PACKAGE = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PACKAGE / "results" / "increment_bootstrap_intervals"
DEFAULT_PKIS1_ZIP = centering.DEFAULT_PKIS1_ZIP
DEFAULT_KIRHUB_WORKBOOK = centering.DEFAULT_KIRHUB_WORKBOOK
FROZEN_DECOMPOSITION = (
    PACKAGE / "results" / "fixed20_estimand_decomposition" / "summary.json"
)
RELEASED_LEDGER = (
    PACKAGE / "results" / "released_pair_geometry_ledger" / "target_pair_geometry.csv"
)

TARGETS = tuple(kirhub.TARGETS)
PANELS = ("DAVIS", "PKIS2", "PKIS1", "KiRHub")
DEFAULT_REPEATS = 5_000
DEFAULT_SEED = 20260806
BUTINA_SIMILARITY = 0.65
BUTINA_GROUPING = f"butina_{BUTINA_SIMILARITY}"
CLUSTER_GROUPINGS = ("murcko", BUTINA_GROUPING)

# Seed offsets.  Every stream is derived deterministically from one base seed and every
# derived seed is written into the summary.
DOCKING_SCAFFOLD_SEED_OFFSET = 1_000_000
DOCKING_MOLECULE_SEED_OFFSET = 2_000_000
EXPERIMENTAL_SEED_OFFSET = 3_000_000
EXPERIMENTAL_PANEL_STRIDE = 10_000
EXPERIMENTAL_GROUPING_STRIDE = 100

# Reporting threshold for calling a bootstrap interval "negligible" on the Spearman
# scale.  It is a post-hoc reporting convention, not a prespecified hypothesis: it is
# roughly one quarter of the experimental-side increments the same design produces
# (0.21-0.26).  The interval and the smallest supported symmetric margin are always
# reported so a reader can substitute another margin.
NEGLIGIBLE_MARGIN = 0.05
CONSISTENCY_TOLERANCE = 1e-10


def rank_unit_vector(edges: np.ndarray) -> np.ndarray:
    """Mean-centred, unit-norm average-rank vector.

    The dot product of two such vectors is exactly the Spearman correlation that
    ``residual_target_geometry_validation.geometry_concordance`` returns, so the
    bootstrap inner loop reproduces the released statistic without calling scipy
    tens of thousands of times.  The identity is verified numerically in
    :func:`consistency_checks`.
    """
    values = np.asarray(edges, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("rank vectors require a finite one-dimensional edge vector")
    ranks = stats.rankdata(values, method="average").astype(np.float64)
    ranks -= ranks.mean()
    norm = float(np.linalg.norm(ranks))
    if norm <= 0:
        raise ValueError("edge vector is constant; Spearman concordance is undefined")
    return ranks / norm


def geometry_rank_unit(matrix: np.ndarray) -> np.ndarray:
    return rank_unit_vector(geometry.upper_triangle(matrix))


def concordance(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Row-wise Spearman concordance between stacked rank-unit vectors.

    ``np.einsum(..., optimize=False)`` is used instead of a BLAS product because BLAS
    reductions differ bitwise across thread counts and the registered summary has to be
    byte-deterministic.
    """
    left = np.atleast_2d(np.asarray(first, dtype=np.float64))
    right = np.atleast_2d(np.asarray(second, dtype=np.float64))
    if left.shape[1] != right.shape[1]:
        raise ValueError("rank-unit vectors have inconsistent edge counts")
    if left.shape[0] == right.shape[0]:
        return np.einsum("re,re->r", left, right, optimize=False)
    if left.shape[0] == 1:
        return np.einsum("e,re->r", left[0], right, optimize=False)
    if right.shape[0] == 1:
        return np.einsum("re,e->r", left, right[0], optimize=False)
    raise ValueError("cannot broadcast the two rank-unit stacks")


def geometries_from_moments(
    count: float, sums: np.ndarray, second_moment: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Raw and within-ligand-centred target correlations from weighted moments.

    ``count``, ``sums`` and ``second_moment`` are the resampled support's total weight,
    weighted column sums and weighted second-moment matrix.  Both geometries come from
    the same moments, which is what keeps the bootstrap difference paired.  The row
    projection ``H C H`` is the identity documented in
    ``fixed20_estimand_decomposition.paired_docking_geometries``.
    """
    total = float(count)
    if total <= 2:
        raise ValueError("a bootstrap replicate has fewer than three effective ligands")
    sums = np.asarray(sums, dtype=np.float64)
    second_moment = np.asarray(second_moment, dtype=np.float64)
    target_count = len(sums)
    if second_moment.shape != (target_count, target_count):
        raise ValueError("moment shapes are inconsistent")
    covariance = (second_moment - np.outer(sums, sums) / total) / (total - 1.0)
    covariance = (covariance + covariance.T) / 2.0
    projection = np.eye(target_count) - np.ones(
        (target_count, target_count), dtype=np.float64
    ) / target_count
    return (
        fixed20._correlation_from_covariance(covariance),  # noqa: SLF001
        fixed20._correlation_from_covariance(  # noqa: SLF001
            projection @ covariance @ projection
        ),
    )


def group_moment_table(
    matrix: np.ndarray, codes: np.ndarray, group_count: int
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    """Per-group sufficient statistics ``[n, column sums, upper second moments]``.

    A cluster bootstrap over 100,000 chemical groups and 260,000 ligands is tractable
    only because the target geometry depends on the resampled support through these
    231 numbers per group and nothing else.
    """
    values = np.asarray(matrix, dtype=np.float64)
    codes = np.asarray(codes, dtype=np.int64)
    if values.ndim != 2 or len(values) != len(codes):
        raise ValueError("moment table requires aligned matrix and group codes")
    target_count = values.shape[1]
    upper = np.triu_indices(target_count)
    order = np.argsort(codes, kind="stable")
    ordered_values = values[order]
    ordered_codes = codes[order]
    if not np.array_equal(np.unique(ordered_codes), np.arange(group_count)):
        raise ValueError("group codes must be a contiguous 0..G-1 labelling")
    starts = np.searchsorted(ordered_codes, np.arange(group_count))
    table = np.empty((group_count, 1 + target_count + len(upper[0])), dtype=np.float64)
    table[:, 0] = np.bincount(ordered_codes, minlength=group_count).astype(np.float64)
    table[:, 1 : 1 + target_count] = np.add.reduceat(ordered_values, starts, axis=0)
    for position, (first, second) in enumerate(zip(*upper)):
        product = ordered_values[:, first] * ordered_values[:, second]
        table[:, 1 + target_count + position] = np.add.reduceat(product, starts)
    return table, upper


def expand_moments(
    row: np.ndarray, target_count: int, upper: tuple[np.ndarray, np.ndarray]
) -> tuple[float, np.ndarray, np.ndarray]:
    count = float(row[0])
    sums = np.asarray(row[1 : 1 + target_count], dtype=np.float64)
    second = np.zeros((target_count, target_count), dtype=np.float64)
    second[upper] = row[1 + target_count :]
    return count, sums, second + second.T - np.diag(np.diag(second))


def bootstrap_docking_rank_vectors(
    table: np.ndarray,
    upper: tuple[np.ndarray, np.ndarray],
    target_count: int,
    repeats: int,
    seed: int,
    chunk: int = 250,
) -> tuple[np.ndarray, np.ndarray]:
    """Resample whole chemical groups and return per-replicate edge rank vectors.

    Each replicate draws ``G`` groups with replacement from the ``G`` observed groups,
    matching ``dense_davis_benchmark._bootstrap_mean``'s cluster contract.  Replicate
    ``r`` uses its own generator seeded with ``[seed, r]``, so the draw does not depend
    on how replicates are chunked for the accumulation.
    """
    if repeats < 1 or chunk < 1:
        raise ValueError("bootstrap repeats and chunk size must be positive")
    group_count = len(table)
    edge_count = target_count * (target_count - 1) // 2
    raw_vectors = np.empty((repeats, edge_count), dtype=np.float64)
    centered_vectors = np.empty((repeats, edge_count), dtype=np.float64)
    multiplicities = np.empty((chunk, group_count), dtype=np.float64)
    for start in range(0, repeats, chunk):
        size = min(chunk, repeats - start)
        for offset in range(size):
            generator = np.random.default_rng([seed, start + offset])
            draws = generator.integers(0, group_count, size=group_count)
            multiplicities[offset] = np.bincount(draws, minlength=group_count)
        moments = np.einsum(
            "rg,gk->rk", multiplicities[:size], table, optimize=False
        )
        for offset in range(size):
            count, sums, second = expand_moments(moments[offset], target_count, upper)
            raw, centered = geometries_from_moments(count, sums, second)
            raw_vectors[start + offset] = geometry_rank_unit(raw)
            centered_vectors[start + offset] = geometry_rank_unit(centered)
    return raw_vectors, centered_vectors


def bootstrap_experimental_rank_vectors(
    matrix: np.ndarray,
    cluster_labels: np.ndarray | None,
    repeats: int,
    seed: int,
) -> tuple[np.ndarray, int]:
    """Resample compounds (or whole chemical clusters) and recompute centred geometry.

    Row centring is per-compound over the 20 targets and therefore does not depend on
    which compounds are drawn; the Pearson target correlation is invariant to the column
    shifts that distinguish row-only from two-way centring, so each replicate geometry
    is exactly the resampled panel's ``center_then_correlation`` geometry.  The identity
    is verified numerically in :func:`consistency_checks`.
    """
    values = np.asarray(matrix, dtype=np.float64)
    row_centered = expchem.row_center(values)
    if cluster_labels is None:
        inverse = np.arange(len(values))
        cluster_count = len(values)
    else:
        labels = np.asarray(cluster_labels)
        if len(labels) != len(values):
            raise ValueError("cluster labels do not align with the panel matrix")
        _, inverse = np.unique(labels, return_inverse=True)
        cluster_count = int(inverse.max() + 1)
    edge_count = values.shape[1] * (values.shape[1] - 1) // 2
    vectors = np.empty((repeats, edge_count), dtype=np.float64)
    for replicate in range(repeats):
        generator = np.random.default_rng([seed, replicate])
        draws = generator.integers(0, cluster_count, size=cluster_count)
        multiplicity = np.bincount(draws, minlength=cluster_count).astype(np.float64)
        vectors[replicate] = geometry_rank_unit(
            expchem.target_correlation(row_centered, multiplicity[inverse])
        )
    return vectors, cluster_count


def describe(values: np.ndarray) -> dict[str, object]:
    """Replicate summary using the package's interval convention."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not np.isfinite(array).all():
        raise ValueError("bootstrap replicates must be a finite one-dimensional vector")
    return {
        "replicates": int(len(array)),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "standard_deviation": float(array.std(ddof=1)),
        "interval_95": [
            float(np.quantile(array, 0.025)),
            float(np.quantile(array, 0.975)),
        ],
        "interval_90": [
            float(np.quantile(array, 0.05)),
            float(np.quantile(array, 0.95)),
        ],
        "minimum": float(array.min()),
        "maximum": float(array.max()),
        "fraction_of_replicates_at_or_below_zero": float(np.mean(array <= 0.0)),
    }


def basic_interval(observed: float, percentile: list[float]) -> list[float]:
    """Reverse-percentile ("basic") interval from a percentile interval.

    The percentile interval is not bias corrected and several of these bootstrap
    distributions are visibly shifted away from the plug-in estimate.  The basic
    interval reflects the shift the other way, so reporting both shows whether a
    verdict survives the choice of interval construction.
    """
    if len(percentile) != 2 or percentile[0] > percentile[1]:
        raise ValueError("a percentile interval must be an ordered pair")
    return [
        float(2.0 * observed - percentile[1]),
        float(2.0 * observed - percentile[0]),
    ]


def conservative_union(intervals: list[list[float]]) -> list[float]:
    """Union of several cluster-bootstrap intervals at one level."""
    if not intervals:
        raise ValueError("a conservative union needs at least one interval")
    return [
        float(min(interval[0] for interval in intervals)),
        float(max(interval[1] for interval in intervals)),
    ]


def verdict_for(interval: list[float]) -> tuple[str, str]:
    """Map an interval onto the three explanations the reviewer asked us to separate."""
    low, high = float(interval[0]), float(interval[1])
    margin = max(abs(low), abs(high))
    if low > 0.0:
        return (
            "resolved_positive_increment",
            "the interval excludes zero from above, so the increment is resolved once "
            "compound sampling is accounted for",
        )
    if high < 0.0:
        return (
            "resolved_negative_increment",
            "the interval excludes zero from below, so the increment is resolved with "
            "the opposite sign",
        )
    if margin <= NEGLIGIBLE_MARGIN:
        return (
            "consistent_with_a_negligible_increment",
            "zero lies inside the interval and the interval also excludes every "
            f"increment larger than {NEGLIGIBLE_MARGIN:g} Spearman units, so the data "
            "support explanation (c), a genuinely tiny effect, rather than mere absence "
            "of evidence",
        )
    return (
        "dominated_by_sampling_uncertainty",
        "zero lies inside the interval and the interval also admits increments larger "
        f"than {NEGLIGIBLE_MARGIN:g} Spearman units, so the data support explanation "
        "(b), large sampling uncertainty, and cannot distinguish a null effect from a "
        "scientifically appreciable one",
    )


def docking_chemical_groups(pkis1_zip: Path) -> tuple[np.ndarray, dict[str, object]]:
    """Chemical-group key for every row of the fixed-20 DOCKSTRING reference.

    Keys and the retained-row mask are rebuilt from the frozen identity contract with
    the same rule ``biological_core_modes`` uses (``MURCKO:<cyclic scaffold>``, or
    ``ACYCLIC:<connectivity block>`` when a molecule has no ring system) and with the
    same DAVIS/PKIS2/PKIS1 connectivity exclusion that
    ``fixed20_estimand_decomposition.load_docking_reference`` applies.
    """
    contract = pd.read_csv(
        fixed20.DEFAULT_IDENTITY_CONTRACT,
        usecols=["raw_connectivity", "raw_murcko_scaffold"],
    )
    davis_source = pd.read_csv(
        davis.DEFAULT_DAVIS,
        sep="\t",
        usecols=["drug_name", "compound_iso_smiles"],
    ).drop_duplicates()
    davis_identity = davis._identity_table(  # noqa: SLF001
        davis_source, "compound_iso_smiles", scan="full"
    )
    excluded_blocks = set(davis_identity.connectivity_block.dropna())
    excluded_blocks |= set(geometry.load_pkis2_full().connectivity_block.dropna())
    excluded_blocks |= set(
        geometry.load_pkis1_full(pkis1_zip).connectivity_block.dropna()
    )
    keep = ~contract.raw_connectivity.isin(excluded_blocks).to_numpy(dtype=bool)
    scaffold = np.asarray(
        contract.raw_murcko_scaffold.fillna("").astype(str).to_numpy()[keep], dtype=str
    )
    connectivity = np.asarray(
        contract.raw_connectivity.astype(str).to_numpy()[keep], dtype=str
    )
    groups = np.where(
        scaffold == "",
        np.char.add("ACYCLIC:", connectivity),
        np.char.add("MURCKO:", scaffold),
    )
    unique = np.unique(groups)
    return groups, {
        "reference_rows": int(keep.sum()),
        "chemical_groups": int(len(unique)),
        "cyclic_murcko_groups": int(np.sum(np.char.startswith(unique, "MURCKO:"))),
        "acyclic_connectivity_groups": int(
            np.sum(np.char.startswith(unique, "ACYCLIC:"))
        ),
        "group_definition": (
            "MURCKO:<canonical cyclic Bemis-Murcko scaffold> when the molecule has a "
            "ring system, otherwise ACYCLIC:<14-character Standard InChI connectivity "
            "block>; identical to the grouping used by biological_core_modes"
        ),
        "butina_boundary": (
            "Morgan/Butina clustering is quadratic in 259,579 ligands and was not run "
            "on the docking reference, so the docking side carries a Murcko/acyclic "
            "cluster bootstrap only and no Murcko-versus-Butina conservative union"
        ),
    }


def experimental_panels(
    pkis1_zip: Path, kirhub_workbook: Path
) -> tuple[OrderedDict[str, np.ndarray], dict[str, pd.Series | None]]:
    """The four fixed-20 experimental panels and their structures where released."""
    matrices = kirhub.load_old_experimental_panels(pkis1_zip)
    kirhub_frame = kirhub.load_kirhub(kirhub_workbook)
    davis_experiment = retrieval._load_davis_experiment()  # noqa: SLF001
    davis_structures = pd.read_csv(
        davis.DEFAULT_DAVIS,
        sep="\t",
        usecols=["drug_name", "compound_iso_smiles"],
    ).drop_duplicates()
    if davis_structures.drug_name.duplicated().any():
        raise ValueError("DAVIS drug names do not map one-to-one to SMILES")
    if not np.array_equal(
        davis_experiment[list(TARGETS)].to_numpy(dtype=np.float64), matrices["DAVIS"]
    ):
        raise ValueError("DAVIS structure alignment does not reproduce the loaded panel")
    davis_smiles = (
        davis_structures.set_index("drug_name")
        .loc[davis_experiment.index, "compound_iso_smiles"]
        .reset_index(drop=True)
    )
    ordered: OrderedDict[str, np.ndarray] = OrderedDict(
        [
            ("DAVIS", matrices["DAVIS"]),
            ("PKIS2", matrices["PKIS2"]),
            ("PKIS1", matrices["PKIS1"]),
            ("KiRHub", kirhub_frame[list(TARGETS)].to_numpy(dtype=np.float64)),
        ]
    )
    if tuple(ordered) != PANELS:
        raise ValueError(f"experimental panels must be ordered as {PANELS}")
    smiles: dict[str, pd.Series | None] = {
        "DAVIS": davis_smiles,
        "PKIS2": geometry.load_pkis2_full()["Smiles"].reset_index(drop=True),
        "PKIS1": geometry.load_pkis1_full(pkis1_zip)["SMILES"].reset_index(drop=True),
        # The KiRHub supplement releases compound names but no structures, so no
        # chemical clustering of that panel is possible inside this package.
        "KiRHub": None,
    }
    for panel, matrix in ordered.items():
        if matrix.shape[1] != len(TARGETS) or not np.isfinite(matrix).all():
            raise ValueError(f"{panel} is not a finite ligand-by-20-target panel")
        series = smiles[panel]
        if series is not None and len(series) != len(matrix):
            raise ValueError(f"{panel} structures do not align with its activity block")
    return ordered, smiles


def consistency_checks(
    reference: np.ndarray,
    observed_raw: np.ndarray,
    observed_centered: np.ndarray,
    observed_raw_unit: np.ndarray,
    observed_centered_unit: np.ndarray,
    panels: OrderedDict[str, np.ndarray],
    observed_delta: dict[str, float],
) -> dict[str, object]:
    """Verify that this script reproduces the released objects it claims to bound."""
    ledger = pd.read_csv(RELEASED_LEDGER)
    differences: dict[str, float] = {
        "released_ledger_raw_docking": float(
            np.max(
                np.abs(
                    geometry.upper_triangle(observed_raw)
                    - ledger.docking_raw_correlation.to_numpy(dtype=np.float64)
                )
            )
        ),
        "released_ledger_centered_docking": float(
            np.max(
                np.abs(
                    geometry.upper_triangle(observed_centered)
                    - ledger.docking_local_20_centered_correlation.to_numpy(
                        dtype=np.float64
                    )
                )
            )
        ),
    }
    frozen = json.loads(FROZEN_DECOMPOSITION.read_text())
    frozen_delta = {
        panel: float(
            frozen["panels"][panel][
                "docking_transform_increment_with_centered_experiment"
            ]
        )
        for panel in panels
    }
    frozen_qap = {
        panel: float(
            frozen["panels"][panel]["paired_qap_for_docking_increment"][
                "one_sided_p_positive_delta"
            ]
        )
        for panel in panels
    }
    differences["frozen_fixed20_increment"] = float(
        max(abs(observed_delta[panel] - frozen_delta[panel]) for panel in panels)
    )
    scipy_gap = 0.0
    moment_gap = 0.0
    weighted_gap = 0.0
    for matrix in panels.values():
        endpoint = geometry.geometry_correlation(matrix, "center_then_correlation")
        endpoint_unit = geometry_rank_unit(endpoint)
        for docking_geometry, docking_unit in (
            (observed_raw, observed_raw_unit),
            (observed_centered, observed_centered_unit),
        ):
            scipy_gap = max(
                scipy_gap,
                abs(
                    float(concordance(docking_unit, endpoint_unit)[0])
                    - geometry.geometry_concordance(docking_geometry, endpoint)
                ),
            )
        weighted_gap = max(
            weighted_gap,
            float(
                np.max(
                    np.abs(
                        expchem.target_correlation(
                            expchem.row_center(matrix),
                            np.ones(len(matrix), dtype=np.float64),
                        )
                        - endpoint
                    )
                )
            ),
        )
    moment_raw, moment_centered = geometries_from_moments(
        float(len(reference)), reference.sum(axis=0), reference.T @ reference
    )
    moment_gap = float(
        max(
            np.max(np.abs(moment_raw - observed_raw)),
            np.max(np.abs(moment_centered - observed_centered)),
        )
    )
    differences["rank_dot_versus_scipy_spearman"] = float(scipy_gap)
    differences["moment_reconstruction"] = float(moment_gap)
    differences["unit_weight_experimental_geometry"] = float(weighted_gap)
    worst = max(differences.values())
    if worst >= CONSISTENCY_TOLERANCE:
        raise RuntimeError(f"consistency checks failed: {differences}")
    return {
        "max_absolute_differences": differences,
        "tolerance": CONSISTENCY_TOLERANCE,
        "all_within_tolerance": True,
        "frozen_fixed20_increments": frozen_delta,
        "frozen_paired_target_label_qap_p_positive": frozen_qap,
        "verified_against": (
            "results/released_pair_geometry_ledger/target_pair_geometry.csv and "
            "results/fixed20_estimand_decomposition/summary.json, both read-only"
        ),
    }


def run(
    *,
    pkis1_zip: Path,
    kirhub_workbook: Path,
    repeats: int,
    seed: int,
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if repeats < 100:
        raise ValueError("the released bootstrap contract requires >=100 replicates")
    reference, docking_support = fixed20.load_docking_reference(pkis1_zip)
    groups, group_support = docking_chemical_groups(pkis1_zip)
    if len(groups) != len(reference):
        raise ValueError("chemical groups do not align with the docking reference")
    target_count = int(reference.shape[1])

    observed_raw, observed_centered = fixed20.paired_docking_geometries(reference)
    observed_raw_unit = geometry_rank_unit(observed_raw)
    observed_centered_unit = geometry_rank_unit(observed_centered)

    panels, panel_smiles = experimental_panels(pkis1_zip, kirhub_workbook)
    observed_panel_unit: dict[str, np.ndarray] = {}
    observed_delta: dict[str, float] = {}
    for panel, matrix in panels.items():
        endpoint_unit = geometry_rank_unit(
            geometry.geometry_correlation(matrix, "center_then_correlation")
        )
        observed_panel_unit[panel] = endpoint_unit
        observed_delta[panel] = float(
            concordance(observed_centered_unit, endpoint_unit)[0]
            - concordance(observed_raw_unit, endpoint_unit)[0]
        )
    checks = consistency_checks(
        reference,
        observed_raw,
        observed_centered,
        observed_raw_unit,
        observed_centered_unit,
        panels,
        observed_delta,
    )

    # ---- docking-side resampling -------------------------------------------------
    column_centered = reference - reference.mean(axis=0, keepdims=True)
    _, group_codes = np.unique(groups, return_inverse=True)
    group_count = int(group_codes.max() + 1)
    scaffold_table, upper = group_moment_table(
        column_centered, group_codes, group_count
    )
    scaffold_seed = seed + DOCKING_SCAFFOLD_SEED_OFFSET
    scaffold_raw_units, scaffold_centered_units = bootstrap_docking_rank_vectors(
        scaffold_table, upper, target_count, repeats, scaffold_seed
    )
    del scaffold_table
    molecule_table, molecule_upper = group_moment_table(
        column_centered,
        np.arange(len(column_centered), dtype=np.int64),
        len(column_centered),
    )
    molecule_seed = seed + DOCKING_MOLECULE_SEED_OFFSET
    molecule_raw_units, molecule_centered_units = bootstrap_docking_rank_vectors(
        molecule_table, molecule_upper, target_count, repeats, molecule_seed, chunk=100
    )
    del molecule_table

    # ---- experimental-side resampling --------------------------------------------
    experimental_units: "OrderedDict[tuple[str, str], np.ndarray]" = OrderedDict()
    experimental_clusters: dict[tuple[str, str], int] = {}
    experimental_seeds: dict[str, int] = {}
    for panel_index, panel in enumerate(PANELS):
        smiles = panel_smiles[panel]
        groupings: list[tuple[str, np.ndarray | None]] = [("compound", None)]
        if smiles is not None:
            groupings.append(("murcko", davis._murcko_labels(smiles)))  # noqa: SLF001
            groupings.append(
                (
                    BUTINA_GROUPING,
                    davis._butina_labels(  # noqa: SLF001
                        smiles, similarity_threshold=BUTINA_SIMILARITY
                    ),
                )
            )
        for grouping_index, (grouping, labels) in enumerate(groupings):
            stream_seed = (
                seed
                + EXPERIMENTAL_SEED_OFFSET
                + panel_index * EXPERIMENTAL_PANEL_STRIDE
                + grouping_index * EXPERIMENTAL_GROUPING_STRIDE
            )
            experimental_seeds[f"{panel}:{grouping}"] = int(stream_seed)
            vectors, clusters = bootstrap_experimental_rank_vectors(
                panels[panel], labels, repeats, stream_seed
            )
            experimental_units[(panel, grouping)] = vectors
            experimental_clusters[(panel, grouping)] = clusters

    # ---- assemble the three schemes ----------------------------------------------
    interval_rows: list[dict[str, object]] = []
    replicate_frames: list[pd.DataFrame] = []

    def register(
        panel: str,
        scheme: str,
        unit: str,
        resampled_units: int,
        stream_seed: int,
        deltas: np.ndarray,
    ) -> None:
        summary = describe(deltas)
        observed = observed_delta[panel]
        basic_95 = basic_interval(observed, summary["interval_95"])
        basic_90 = basic_interval(observed, summary["interval_90"])
        interval_rows.append(
            {
                "panel": panel,
                "resampling_scheme": scheme,
                "resampling_unit": unit,
                "resampled_units": int(resampled_units),
                "replicates": summary["replicates"],
                "seed": int(stream_seed),
                "observed_delta_d": observed,
                "bootstrap_mean": summary["mean"],
                "bootstrap_median": summary["median"],
                "bootstrap_sd": summary["standard_deviation"],
                "bootstrap_bias": float(summary["mean"] - observed),
                "interval_95_low": summary["interval_95"][0],
                "interval_95_high": summary["interval_95"][1],
                "interval_90_low": summary["interval_90"][0],
                "interval_90_high": summary["interval_90"][1],
                "basic_interval_95_low": basic_95[0],
                "basic_interval_95_high": basic_95[1],
                "basic_interval_90_low": basic_90[0],
                "basic_interval_90_high": basic_90[1],
                "minimum": summary["minimum"],
                "maximum": summary["maximum"],
                "fraction_of_replicates_at_or_below_zero": summary[
                    "fraction_of_replicates_at_or_below_zero"
                ],
            }
        )
        replicate_frames.append(
            pd.DataFrame(
                {
                    "panel": panel,
                    "resampling_scheme": scheme,
                    "resampling_unit": unit,
                    "replicate": np.arange(len(deltas), dtype=np.int64),
                    "delta_d": np.asarray(deltas, dtype=np.float64),
                }
            )
        )

    for panel in PANELS:
        fixed_endpoint = observed_panel_unit[panel]
        register(
            panel,
            "docking_only",
            "dockstring_chemical_group",
            group_count,
            scaffold_seed,
            concordance(scaffold_centered_units, fixed_endpoint)
            - concordance(scaffold_raw_units, fixed_endpoint),
        )
        register(
            panel,
            "docking_only",
            "dockstring_molecule",
            len(reference),
            molecule_seed,
            concordance(molecule_centered_units, fixed_endpoint)
            - concordance(molecule_raw_units, fixed_endpoint),
        )
        for (candidate, grouping), vectors in experimental_units.items():
            if candidate != panel:
                continue
            stream_seed = experimental_seeds[f"{panel}:{grouping}"]
            register(
                panel,
                "experimental_only",
                grouping,
                experimental_clusters[(panel, grouping)],
                stream_seed,
                concordance(vectors, observed_centered_unit)
                - concordance(vectors, observed_raw_unit),
            )
            register(
                panel,
                "combined",
                grouping,
                experimental_clusters[(panel, grouping)],
                stream_seed,
                concordance(scaffold_centered_units, vectors)
                - concordance(scaffold_raw_units, vectors),
            )

    interval_frame = pd.DataFrame.from_records(interval_rows)
    replicate_frame = pd.concat(replicate_frames, ignore_index=True)

    # ---- conservative unions and verdicts ----------------------------------------
    conservative_rows: list[dict[str, object]] = []
    for panel in PANELS:
        available = [
            grouping
            for grouping in CLUSTER_GROUPINGS
            if (panel, grouping) in experimental_units
        ]
        cluster_available = bool(available)
        sources = available if cluster_available else ["compound"]
        for scheme in ("experimental_only", "combined"):
            selected = interval_frame[
                interval_frame.panel.eq(panel)
                & interval_frame.resampling_scheme.eq(scheme)
                & interval_frame.resampling_unit.isin(sources)
            ]
            if len(selected) != len(sources):
                raise RuntimeError(f"missing bootstrap rows for {panel}/{scheme}")
            union_95 = conservative_union(
                [
                    [row.interval_95_low, row.interval_95_high]
                    for row in selected.itertuples(index=False)
                ]
            )
            union_90 = conservative_union(
                [
                    [row.interval_90_low, row.interval_90_high]
                    for row in selected.itertuples(index=False)
                ]
            )
            basic_95 = conservative_union(
                [
                    [row.basic_interval_95_low, row.basic_interval_95_high]
                    for row in selected.itertuples(index=False)
                ]
            )
            verdict, reading = verdict_for(union_95)
            basic_verdict, _ = verdict_for(basic_95)
            conservative_rows.append(
                {
                    "panel": panel,
                    "resampling_scheme": scheme,
                    "bootstrap_sources": "|".join(sources),
                    "chemical_cluster_bootstrap_available": cluster_available,
                    "observed_delta_d": observed_delta[panel],
                    "conservative_interval_95_low": union_95[0],
                    "conservative_interval_95_high": union_95[1],
                    "conservative_interval_95_width": float(union_95[1] - union_95[0]),
                    "conservative_interval_90_low": union_90[0],
                    "conservative_interval_90_high": union_90[1],
                    "conservative_basic_interval_95_low": basic_95[0],
                    "conservative_basic_interval_95_high": basic_95[1],
                    "smallest_supported_symmetric_margin_95": float(
                        max(abs(union_95[0]), abs(union_95[1]))
                    ),
                    "interval_contains_zero": bool(union_95[0] <= 0.0 <= union_95[1]),
                    "verdict": verdict,
                    "basic_interval_verdict": basic_verdict,
                    "verdict_robust_to_interval_construction": bool(
                        verdict == basic_verdict
                    ),
                    "reading": reading,
                }
            )
    conservative_frame = pd.DataFrame.from_records(conservative_rows)

    combined = conservative_frame[
        conservative_frame.resampling_scheme.eq("combined")
    ].set_index("panel")
    docking_only = interval_frame[
        interval_frame.resampling_scheme.eq("docking_only")
        & interval_frame.resampling_unit.eq("dockstring_chemical_group")
    ].set_index("panel")
    panel_reports: dict[str, dict[str, object]] = {}
    for panel in PANELS:
        row = combined.loc[panel]
        docking_row = docking_only.loc[panel]
        panel_reports[panel] = {
            "experimental_ligands": int(len(panels[panel])),
            "observed_delta_d": observed_delta[panel],
            "published_paired_target_label_qap_p_positive": checks[
                "frozen_paired_target_label_qap_p_positive"
            ][panel],
            "docking_chemical_group_interval_95": [
                float(docking_row.interval_95_low),
                float(docking_row.interval_95_high),
            ],
            "combined_conservative_interval_95": [
                float(row.conservative_interval_95_low),
                float(row.conservative_interval_95_high),
            ],
            "combined_conservative_interval_95_width": float(
                row.conservative_interval_95_width
            ),
            "combined_conservative_basic_interval_95": [
                float(row.conservative_basic_interval_95_low),
                float(row.conservative_basic_interval_95_high),
            ],
            "smallest_supported_symmetric_margin_95": float(
                row.smallest_supported_symmetric_margin_95
            ),
            "chemical_cluster_bootstrap_available": bool(
                row.chemical_cluster_bootstrap_available
            ),
            "bootstrap_sources": str(row.bootstrap_sources),
            "verdict": str(row.verdict),
            "basic_interval_verdict": str(row.basic_interval_verdict),
            "verdict_robust_to_interval_construction": bool(
                row.verdict_robust_to_interval_construction
            ),
            "reading": str(row.reading),
        }

    resolved = [panel for panel in PANELS if panel_reports[panel]["verdict"].startswith("resolved")]
    negligible = [
        panel
        for panel in PANELS
        if panel_reports[panel]["verdict"] == "consistent_with_a_negligible_increment"
    ]
    uncertain = [
        panel
        for panel in PANELS
        if panel_reports[panel]["verdict"] == "dominated_by_sampling_uncertainty"
    ]
    fragile = [
        panel
        for panel in PANELS
        if not panel_reports[panel]["verdict_robust_to_interval_construction"]
    ]
    conclusion = (
        "Compound resampling separates the four panels that the target-label QAP could "
        "not. "
        + (
            "The 95% interval excludes zero for "
            f"{', '.join(resolved)}, so the increment there is reproducible across "
            "compound resamples rather than absent. "
            if resolved
            else ""
        )
        + (
            "The interval contains zero and excludes every increment above "
            f"{NEGLIGIBLE_MARGIN:g} for {', '.join(negligible)}, which supports a "
            "genuinely tiny effect. "
            if negligible
            else ""
        )
        + (
            "The interval contains zero and still admits increments above "
            f"{NEGLIGIBLE_MARGIN:g} for {', '.join(uncertain)}, which remains "
            "uninformative about effect size. "
            if uncertain
            else ""
        )
        + (
            "The verdict changes under a reverse-percentile interval for "
            f"{', '.join(fragile)} and should be read as borderline. "
            if fragile
            else "Every verdict is unchanged under a reverse-percentile interval. "
        )
        + "An interval excluding zero is not evidence of target specificity: the "
        "paired target-label QAP and the transformation-matched null remain "
        "unresolved for all four panels, and those are the tests that address "
        "target identity and the mechanical centring baseline."
    )
    summary: dict[str, object] = {
        "analysis": (
            "paired bootstrap confidence intervals for the docking-side increment "
            "against the fixed centred experimental endpoint"
        ),
        "strongest_honest_conclusion": conclusion,
        "verdicts": {
            "resolved_nonzero": resolved,
            "consistent_with_a_negligible_increment": negligible,
            "dominated_by_sampling_uncertainty": uncertain,
            "verdict_not_robust_to_interval_construction": fragile,
        },
        "question": (
            "The manuscript resolves the docking-side increments Delta_D only against a "
            "paired target-label QAP null. A QAP null interval is not a confidence "
            "interval and holds both compound libraries fixed. Once the finite "
            "DOCKSTRING reference and the finite experimental panels are resampled, how "
            "wide is Delta_D, and does that width distinguish (a) no target-label "
            "evidence, (b) large sampling uncertainty, and (c) a genuinely tiny effect?"
        ),
        "status": "post_hoc_uncertainty_companion_to_the_reported_increments",
        "estimand": (
            "Delta_D = Spearman(edges of the within-ligand-centred docking target "
            "correlation, edges of the two-way-centred experimental target "
            "correlation) minus Spearman(edges of the raw docking target correlation, "
            "the same experimental edges), over the 190 pairs of the fixed 20-kinase "
            "panel"
        ),
        "support": {
            "targets": list(TARGETS),
            "target_pairs": int(target_count * (target_count - 1) // 2),
            "docking_reference_ligands": int(len(reference)),
            "docking_chemical_groups": int(group_count),
            "experimental_panel_ligands": {
                panel: int(len(matrix)) for panel, matrix in panels.items()
            },
            "experimental_chemical_clusters": {
                f"{panel}:{grouping}": int(count)
                for (panel, grouping), count in sorted(experimental_clusters.items())
            },
        },
        "configuration": {
            "bootstrap_replicates": int(repeats),
            "base_seed": int(seed),
            "docking_chemical_group_seed": int(scaffold_seed),
            "docking_molecule_seed": int(molecule_seed),
            "experimental_seeds": {
                key: int(value) for key, value in sorted(experimental_seeds.items())
            },
            "seed_derivation": (
                "replicate r of a stream with seed s uses "
                "numpy.random.default_rng([s, r]), so every replicate is independent of "
                "the chunking used to accumulate it"
            ),
            "resampling_contract": (
                "clusters are drawn with replacement, as many clusters as were "
                "observed, following dense_davis_benchmark._bootstrap_mean"
            ),
            "interval_convention": (
                "2.5/97.5 and 5/95 replicate percentiles, following "
                "dense_davis_benchmark._distribution; the reported interval is the "
                "conservative union of the Murcko and Butina-0.65 intervals, following "
                "dockstring_chembl_ranking_benchmark.conservative_union; a "
                "reverse-percentile (basic) interval is reported alongside because "
                "the percentile interval is not bias corrected and some replicate "
                "distributions are shifted away from the plug-in estimate"
            ),
            "negligible_margin": NEGLIGIBLE_MARGIN,
            "negligible_margin_boundary": (
                "0.05 Spearman units is a post-hoc reporting threshold, roughly a "
                "quarter of the experimental-side increments this same design "
                "produces; the interval and the smallest supported symmetric margin "
                "are reported so a reader can substitute another margin"
            ),
            "accumulation_boundary": (
                "group moments and every concordance are accumulated with "
                "numpy.einsum(optimize=False) because BLAS reductions differ bitwise "
                "across thread counts and this summary must be byte-deterministic"
            ),
        },
        "docking_reference": docking_support,
        "docking_chemical_grouping": group_support,
        "panels": panel_reports,
        "consistency_checks": checks,
        "kirhub_boundary": (
            "The KiRHub supplement releases compound names but no structures, so that "
            "panel can only be resampled compound by compound. If KiRHub contains "
            "analogue series its interval is narrower than a chemical-cluster interval "
            "would be, making the KiRHub row the least conservative of the four."
        ),
        "claim_boundary": (
            "These are nonparametric bootstrap intervals for one fixed estimand on one "
            "fixed 20-kinase panel, and they quantify uncertainty from the finite "
            "compound libraries only. An interval excluding zero establishes that the "
            "increment reproduces across compound resamples with the twenty target "
            "labels held fixed. It does NOT establish that the increment is tied to "
            "target identity: that is the paired target-label QAP, which remains "
            "unresolved for every panel. It does NOT establish that the increment "
            "exceeds the alignment that within-ligand centring induces mechanically on "
            "marginal-matched independent docking columns: that is the "
            "transformation-matched null, which also remains unresolved for every "
            "panel. The three tests answer different questions and a bootstrap "
            "interval cannot substitute for either null. These intervals also say "
            "nothing about the separate question of whether the residual surface is "
            "target-blind, because Delta_D is computed from docking scores and "
            "activities and involves no descriptor regression. The 190 edges of a "
            "20-target panel are not independent, no interval is adjusted for the four "
            "panels sharing those same 190 pairs, and percentile intervals are not "
            "bias corrected, which is why the reverse-percentile interval is reported "
            "beside every conservative union."
        ),
    }
    return summary, interval_frame, conservative_frame, replicate_frame


def write_outputs(
    output: Path,
    summary: dict[str, object],
    intervals: pd.DataFrame,
    conservative: pd.DataFrame,
    replicates: pd.DataFrame,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    intervals.sort_values(
        ["panel", "resampling_scheme", "resampling_unit"], kind="mergesort"
    ).to_csv(output / "panel_intervals.csv", index=False, float_format="%.15g")
    conservative.sort_values(
        ["panel", "resampling_scheme"], kind="mergesort"
    ).to_csv(output / "conservative_intervals.csv", index=False, float_format="%.15g")
    replicates.to_csv(
        output / "bootstrap_replicates.csv", index=False, float_format="%.15g"
    )
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pkis1-zip", type=Path, default=DEFAULT_PKIS1_ZIP)
    parser.add_argument(
        "--kirhub-workbook", type=Path, default=DEFAULT_KIRHUB_WORKBOOK
    )
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary, intervals, conservative, replicates = run(
        pkis1_zip=args.pkis1_zip,
        kirhub_workbook=args.kirhub_workbook,
        repeats=args.repeats,
        seed=args.seed,
    )
    write_outputs(args.output, summary, intervals, conservative, replicates)
    print(
        conservative[
            [
                "panel",
                "resampling_scheme",
                "observed_delta_d",
                "conservative_interval_95_low",
                "conservative_interval_95_high",
                "verdict",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
