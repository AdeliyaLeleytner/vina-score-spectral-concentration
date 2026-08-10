#!/usr/bin/env python3
"""Guard against a descriptor "mediated share" ratio with a decomposition that carries error.

The absolute agreements from ``descriptor_component_geometry`` can be divided to form
``rho(descriptor-component geometry, E) / rho(observed residual geometry, E)``.  The current
manuscript correctly says that this is *not* a mediation fraction; this analysis makes that
boundary machine-testable.  The ratio is not additive, its numerator and denominator are
correlated, and it explodes whenever the denominator is small.  On the identical fixed 190
edges of the common-20 kinase panel and the identical de-leaked DOCKSTRING support, this
analysis instead computes:

1. the three *absolute* agreements -- observed residual geometry, descriptor-predicted
   component, descriptor-removed remainder -- each with a conditional coarsened
   chemical-group sensitivity interval obtained by paired reweighting of the fixed OOF
   surfaces; this is explicitly not a full-refit or chemical-space confidence interval;
2. a proper edge-level decomposition: the experimental edge vector regressed jointly on the
   descriptor-component and remainder edge vectors, with standardized partial coefficients,
   MRQAP target-label permutation probabilities (Y-permutation and Dekker
   double-semi-partialling), partial Spearman correlations, and the correlation *between*
   the two component edge vectors, which is not zero;
3. an orthogonalized variant: each component residualized against the other, reporting the
   incremental agreement and the semipartial R-squared each one contributes.

It also records the diagnostic that motivates the revision.  The decomposition fits
*separate regression coefficients per target*: the descriptor component surface is
``X B`` with ``X`` the 7 ligand descriptors and ``B`` a 7-by-20 matrix of fitted per-target
slopes.  Its 190-edge geometry is therefore a deterministic function of 140 fitted numbers
that are indexed by target, and its target correlation matrix has rank at most 7 by
construction.  A surface built this way is *descriptor-driven*; it is not target-blind, and
this package should not describe it as such.

Outputs are deterministic within the configured numerical environment: the default
fixed-OOF mode uses one seeded block-weight stream per grouping, the optional full-refit
mode seeds every replicate from its own index, BLAS is pinned to one thread so results do
not depend on worker count, and no timing or wall-clock value is written.
"""

from __future__ import annotations

import os

# Pin BLAS before NumPy is imported anywhere in this process.  Bootstrap replicates are
# distributed over worker processes; single-threaded BLAS makes the arithmetic in each
# replicate independent of how many workers run, which is what keeps the artifact
# byte-identical across reruns and across machines with different core counts.
for _thread_variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_thread_variable] = "1"

import argparse  # noqa: E402
import multiprocessing as mp  # noqa: E402
import tempfile  # noqa: E402
from collections import OrderedDict  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from rdkit import Chem, RDLogger  # noqa: E402
from rdkit.Chem.Scaffolds import MurckoScaffold  # noqa: E402
from scipy import stats  # noqa: E402

try:  # Support direct CLI execution and package-style imports.
    from . import descriptor_component_geometry as component
    from . import residual_mechanism_analysis as mechanism
    from . import residual_target_geometry_validation as geometry
except ImportError:  # pragma: no cover - direct CLI execution.
    import descriptor_component_geometry as component  # type: ignore
    import residual_mechanism_analysis as mechanism  # type: ignore
    import residual_target_geometry_validation as geometry  # type: ignore


PACKAGE = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PACKAGE / "results" / "descriptor_geometry_decomposition"
DEFAULT_SEED = 20260806
DEFAULT_FOLDS = component.DEFAULT_FOLDS
DEFAULT_BOOTSTRAP_REPEATS = 5_000
DEFAULT_BOOTSTRAP_BLOCKS = 512
DEFAULT_BOOTSTRAP_MODE = "fixed_oof_coarsened"
DEFAULT_PERMUTATIONS = 49_999
BOOTSTRAP_CHUNK = 100
TARGETS20 = component.TARGETS20
PANELS = component.PANELS
SURFACES = ("observed_residual", "descriptor_component", "descriptor_removed")

# Two chemical groupings, unioned conservatively.  The package's usual second grouping is
# Butina at Tanimoto 0.65, but ``dense_davis_benchmark._butina_labels`` materializes every
# pairwise Tanimoto distance, which is 3.4e10 pairs on this 259,579-ligand support and is
# not computable.  The coarser Bemis--Murcko *generic framework* (all atoms carbon, all
# bonds single) is used instead: it strictly coarsens the Murcko grouping, so it plays the
# same conservative role of widening the interval when scaffold classes co-vary.
GROUPINGS = ("murcko_scaffold", "murcko_generic_framework")

# One statistic block per replicate.  Order is fixed so the replicate matrix is stable.
GLOBAL_STATISTICS = (
    "component_remainder_spearman",
    "component_remainder_pearson",
)
PANEL_STATISTICS = (
    "agreement_observed_residual",
    "agreement_descriptor_component",
    "agreement_descriptor_removed",
    "beta_descriptor_component",
    "beta_descriptor_removed",
    "partial_spearman_descriptor_component",
    "partial_spearman_descriptor_removed",
    "joint_r2",
    "incremental_spearman_descriptor_component",
    "incremental_spearman_descriptor_removed",
    "semipartial_r2_descriptor_component",
    "semipartial_r2_descriptor_removed",
    "retired_mediated_share_ratio",
)


def statistic_names() -> list[str]:
    names = list(GLOBAL_STATISTICS)
    for panel in PANELS:
        names.extend(f"{panel}::{statistic}" for statistic in PANEL_STATISTICS)
    return names


# --------------------------------------------------------------------------------------
# Edge-level statistics.  Every one of these operates on 190-vectors of the locked
# upper-triangle order, so ``geometry.upper_triangle`` and ``component._square_from_edges``
# are the only reshaping primitives used.
# --------------------------------------------------------------------------------------


def rank_z(values: np.ndarray) -> np.ndarray:
    """Average ranks, centred and scaled to unit sample standard deviation."""
    ranks = stats.rankdata(np.asarray(values, dtype=np.float64), method="average")
    scale = ranks.std(ddof=1)
    if scale <= 0:
        raise ValueError("edge vector is constant and cannot be rank-standardized")
    return (ranks - ranks.mean()) / scale


def joint_edge_regression(
    experimental: np.ndarray, first: np.ndarray, second: np.ndarray
) -> dict[str, float]:
    """Rank-scale joint regression of one experimental edge vector on two predictors.

    Because all three vectors are rank-standardized, the two slopes are standardized
    partial coefficients and ``r2`` is the fraction of rank variance in the experimental
    edge vector explained jointly.  The single-predictor fits give the semipartial
    (incremental) R-squared of each component over the other.
    """
    y = rank_z(experimental)
    a = rank_z(first)
    b = rank_z(second)
    n = len(y)
    design = np.column_stack([np.ones(n), a, b])
    coefficients = np.linalg.lstsq(design, y, rcond=None)[0]
    residual = y - design @ coefficients
    total = float(np.square(y - y.mean()).sum())
    r2_joint = 1.0 - float(np.square(residual).sum()) / total

    def _single(predictor: np.ndarray) -> float:
        matrix = np.column_stack([np.ones(n), predictor])
        beta = np.linalg.lstsq(matrix, y, rcond=None)[0]
        return 1.0 - float(np.square(y - matrix @ beta).sum()) / total

    r2_first_only = _single(a)
    r2_second_only = _single(b)
    return {
        "beta_first": float(coefficients[1]),
        "beta_second": float(coefficients[2]),
        "r2_joint": float(r2_joint),
        "r2_first_only": float(r2_first_only),
        "r2_second_only": float(r2_second_only),
        "semipartial_r2_first": float(r2_joint - r2_second_only),
        "semipartial_r2_second": float(r2_joint - r2_first_only),
    }


def partial_spearman(
    experimental: np.ndarray, focal: np.ndarray, control: np.ndarray
) -> float:
    """Spearman correlation of ``experimental`` with ``focal``, controlling ``control``."""
    r_ef = stats.spearmanr(experimental, focal).statistic
    r_ec = stats.spearmanr(experimental, control).statistic
    r_fc = stats.spearmanr(focal, control).statistic
    denominator = np.sqrt((1.0 - r_fc**2) * (1.0 - r_ec**2))
    if denominator <= 0:
        raise ValueError("partial Spearman has a degenerate denominator")
    return float((r_ef - r_ec * r_fc) / denominator)


def orthogonalize(focal: np.ndarray, control: np.ndarray) -> np.ndarray:
    """Rank-scale residual of ``focal`` after removing its projection on ``control``."""
    a = rank_z(focal)
    b = rank_z(control)
    slope = float(np.dot(a, b) / np.dot(b, b))
    return a - slope * b


def replicate_statistics(
    geometries: "OrderedDict[str, np.ndarray]",
    experimental_edges: "OrderedDict[str, np.ndarray]",
) -> np.ndarray:
    """The fixed-order statistic vector recorded for one bootstrap replicate."""
    observed = geometries["observed_residual"]
    predicted = geometries["descriptor_component"]
    remainder = geometries["descriptor_removed"]
    values = [
        float(stats.spearmanr(predicted, remainder).statistic),
        float(stats.pearsonr(predicted, remainder).statistic),
    ]
    for panel in PANELS:
        edges = experimental_edges[panel]
        agreement_observed = float(stats.spearmanr(observed, edges).statistic)
        agreement_component = float(stats.spearmanr(predicted, edges).statistic)
        agreement_remainder = float(stats.spearmanr(remainder, edges).statistic)
        fit = joint_edge_regression(edges, predicted, remainder)
        incremental_component = float(
            stats.spearmanr(edges, orthogonalize(predicted, remainder)).statistic
        )
        incremental_remainder = float(
            stats.spearmanr(edges, orthogonalize(remainder, predicted)).statistic
        )
        values.extend([
            agreement_observed,
            agreement_component,
            agreement_remainder,
            fit["beta_first"],
            fit["beta_second"],
            partial_spearman(edges, predicted, remainder),
            partial_spearman(edges, remainder, predicted),
            fit["r2_joint"],
            incremental_component,
            incremental_remainder,
            fit["semipartial_r2_first"],
            fit["semipartial_r2_second"],
            (
                agreement_component / agreement_observed
                if agreement_observed > 0
                else np.nan
            ),
        ])
    return np.asarray(values, dtype=np.float64)


# --------------------------------------------------------------------------------------
# MRQAP.  The permutation unit is the complete target label, matching
# ``residual_target_geometry_validation.qap_test``.
# --------------------------------------------------------------------------------------


def _permute_edges(edges: np.ndarray, order: np.ndarray) -> np.ndarray:
    matrix = component._square_from_edges(edges)  # noqa: SLF001
    return geometry.upper_triangle(matrix[np.ix_(order, order)])


def mrqap(
    experimental: np.ndarray,
    predicted: np.ndarray,
    remainder: np.ndarray,
    permutations: int,
    seed: int,
) -> dict[str, object]:
    """Y-permutation and Dekker double-semi-partialling MRQAP for two edge predictors.

    Y-permutation relabels the experimental geometry, which is the exact analogue of the
    package's single-predictor QAP.  Dekker's double-semi-partialling permutes the focal
    predictor *after* residualizing it on the other predictor, which is the standard
    correction for the two predictors not being orthogonal -- and here they are not
    (their edge vectors correlate at Spearman ~0.34).
    """
    if permutations < 1:
        raise ValueError("MRQAP requires at least one permutation")
    p = len(TARGETS20)
    y = rank_z(experimental)
    a = rank_z(predicted)
    b = rank_z(remainder)
    n = len(y)
    design = np.column_stack([np.ones(n), a, b])
    observed = np.linalg.lstsq(design, y, rcond=None)[0]
    total = float(np.square(y - y.mean()).sum())
    observed_r2 = 1.0 - float(
        np.square(y - design @ observed).sum()
    ) / total

    rng = np.random.default_rng(seed)
    null_beta_first = np.empty(permutations, dtype=np.float64)
    null_beta_second = np.empty(permutations, dtype=np.float64)
    null_r2 = np.empty(permutations, dtype=np.float64)
    for repetition in range(permutations):
        order = rng.permutation(p)
        permuted = rank_z(_permute_edges(experimental, order))
        coefficients = np.linalg.lstsq(design, permuted, rcond=None)[0]
        null_beta_first[repetition] = coefficients[1]
        null_beta_second[repetition] = coefficients[2]
        null_r2[repetition] = 1.0 - float(
            np.square(permuted - design @ coefficients).sum()
        ) / total

    dekker = {}
    for label, focal, control, index in (
        ("descriptor_component", a, b, 1),
        ("descriptor_removed", b, a, 2),
    ):
        slope = float(np.dot(focal, control) / np.dot(control, control))
        residual = focal - slope * control
        dekker_rng = np.random.default_rng(seed + 7_000_000 + index)
        null = np.empty(permutations, dtype=np.float64)
        for repetition in range(permutations):
            order = dekker_rng.permutation(p)
            permuted_residual = _permute_edges(residual, order)
            if index == 1:
                matrix = np.column_stack([np.ones(n), permuted_residual, control])
            else:
                matrix = np.column_stack([np.ones(n), control, permuted_residual])
            coefficients = np.linalg.lstsq(matrix, y, rcond=None)[0]
            null[repetition] = coefficients[index]
        dekker[label] = {
            "one_sided_p_positive": float(
                (1 + int(np.sum(null >= observed[index]))) / (permutations + 1)
            ),
            "two_sided_p": float(
                (1 + int(np.sum(np.abs(null) >= abs(observed[index]))))
                / (permutations + 1)
            ),
            "null_interval_95": [
                float(np.quantile(null, 0.025)),
                float(np.quantile(null, 0.975)),
            ],
            "seed": int(seed + 7_000_000 + index),
        }

    return {
        "standardized_partial_coefficients": {
            "descriptor_component": float(observed[1]),
            "descriptor_removed": float(observed[2]),
        },
        "joint_rank_r2": float(observed_r2),
        "y_permutation": {
            "descriptor_component": {
                "one_sided_p_positive": float(
                    (1 + int(np.sum(null_beta_first >= observed[1])))
                    / (permutations + 1)
                ),
                "two_sided_p": float(
                    (1 + int(np.sum(np.abs(null_beta_first) >= abs(observed[1]))))
                    / (permutations + 1)
                ),
                "null_interval_95": [
                    float(np.quantile(null_beta_first, 0.025)),
                    float(np.quantile(null_beta_first, 0.975)),
                ],
            },
            "descriptor_removed": {
                "one_sided_p_positive": float(
                    (1 + int(np.sum(null_beta_second >= observed[2])))
                    / (permutations + 1)
                ),
                "two_sided_p": float(
                    (1 + int(np.sum(np.abs(null_beta_second) >= abs(observed[2]))))
                    / (permutations + 1)
                ),
                "null_interval_95": [
                    float(np.quantile(null_beta_second, 0.025)),
                    float(np.quantile(null_beta_second, 0.975)),
                ],
            },
            "omnibus_joint_r2": {
                "one_sided_p_positive": float(
                    (1 + int(np.sum(null_r2 >= observed_r2))) / (permutations + 1)
                ),
                "null_interval_95": [
                    float(np.quantile(null_r2, 0.025)),
                    float(np.quantile(null_r2, 0.975)),
                ],
            },
            "seed": int(seed),
        },
        "dekker_double_semi_partialling": dekker,
        "permutations": int(permutations),
        "permutation_unit": "complete target labels",
    }


# --------------------------------------------------------------------------------------
# Chemical groupings and the bootstrap.
# --------------------------------------------------------------------------------------


def generic_framework_keys(smiles: np.ndarray) -> np.ndarray:
    """Bemis--Murcko generic frameworks; acyclic molecules stay singletons.

    ``mechanism.scaffold_keys`` supplies the Murcko grouping; this coarsens it by mapping
    every scaffold atom to carbon and every bond to a single bond, which merges scaffold
    classes that differ only in heteroatom placement.
    """
    RDLogger.DisableLog("rdApp.*")
    keys: list[str] = []
    for index, value in enumerate(smiles):
        molecule = Chem.MolFromSmiles(str(value))
        if molecule is None:
            raise ValueError(f"invalid SMILES at row {index}")
        scaffold = MurckoScaffold.GetScaffoldForMol(molecule)
        framework = ""
        if scaffold is not None and scaffold.GetNumAtoms():
            try:
                framework = Chem.MolToSmiles(
                    MurckoScaffold.MakeScaffoldGeneric(scaffold)
                )
            except Exception:  # pragma: no cover - defensive, never hit on this support
                framework = ""
        keys.append(framework if framework else f"ACYCLIC_SINGLETON:{index}")
    return np.asarray(keys, dtype=object)


class _GroupIndex:
    """Contiguous membership index so a group bootstrap is an O(n) gather."""

    def __init__(self, labels: np.ndarray) -> None:
        codes, inverse = np.unique(np.asarray(labels, dtype=object), return_inverse=True)
        order = np.argsort(inverse, kind="stable")
        sorted_inverse = inverse[order]
        positions = np.arange(len(codes))
        self.order = order
        self.starts = np.searchsorted(sorted_inverse, positions)
        self.ends = np.searchsorted(sorted_inverse, positions, side="right")
        self.sizes = self.ends - self.starts
        self.count = int(len(codes))

    def draw(self, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        picks = rng.integers(0, self.count, size=self.count)
        rows = np.concatenate(
            [self.order[self.starts[pick] : self.ends[pick]] for pick in picks]
        )
        labels = np.repeat(np.arange(self.count), self.sizes[picks]).astype(object)
        return rows, labels


_WORKER: dict[str, object] = {}


def _worker_initializer(cache_directory: str, folds: int) -> None:
    cache = Path(cache_directory)
    block = np.load(cache / "block.npy", mmap_mode="r")
    descriptors = np.load(cache / "descriptors.npy", mmap_mode="r")
    experimental = np.load(cache / "experimental.npy")
    _WORKER["block"] = block
    _WORKER["descriptors"] = descriptors
    _WORKER["experimental"] = OrderedDict(
        (panel, experimental[index]) for index, panel in enumerate(PANELS)
    )
    _WORKER["folds"] = int(folds)
    _WORKER["indices"] = {
        grouping: _GroupIndex(
            np.load(cache / f"groups_{grouping}.npy", allow_pickle=False)
        )
        for grouping in GROUPINGS
    }


def _worker_chunk(task: tuple[str, int, int, int]) -> tuple[int, np.ndarray]:
    grouping, start, stop, base_seed = task
    index = _WORKER["indices"][grouping]  # type: ignore[index]
    block = _WORKER["block"]
    descriptors = _WORKER["descriptors"]
    experimental = _WORKER["experimental"]
    folds = _WORKER["folds"]
    width = len(statistic_names())
    output = np.full((stop - start, width), np.nan, dtype=np.float64)
    for position, replicate in enumerate(range(start, stop)):
        rng = np.random.default_rng(base_seed + replicate)
        rows, labels = index.draw(rng)
        try:
            decomposition = mechanism.grouped_descriptor_decomposition(
                np.asarray(block[rows]), np.asarray(descriptors[rows]), labels, folds
            )
            geometries = OrderedDict(
                (
                    name,
                    geometry.upper_triangle(geometry.target_correlation(surface)),
                )
                for name, surface in (
                    ("observed_residual", decomposition["oof_surface"]),
                    ("descriptor_component", decomposition["oof_prediction"]),
                    ("descriptor_removed", decomposition["oof_error"]),
                )
            )
            output[position] = replicate_statistics(geometries, experimental)
        except Exception:  # pragma: no cover - degenerate resample, recorded as a failure
            continue
    return start, output


def run_bootstrap(
    cache_directory: Path,
    grouping: str,
    repeats: int,
    base_seed: int,
    folds: int,
    workers: int,
) -> np.ndarray:
    """Replicate-by-statistic matrix; row ``i`` is always seeded ``base_seed + i``."""
    bounds = np.linspace(0, repeats, workers + 1).astype(int)
    tasks = [
        (grouping, int(bounds[i]), int(bounds[i + 1]), base_seed)
        for i in range(workers)
        if bounds[i + 1] > bounds[i]
    ]
    records = np.full((repeats, len(statistic_names())), np.nan, dtype=np.float64)
    if workers == 1:
        _worker_initializer(str(cache_directory), folds)
        for task in tasks:
            start, block = _worker_chunk(task)
            records[start : start + len(block)] = block
        return records
    context = mp.get_context("spawn")
    with context.Pool(
        processes=len(tasks),
        initializer=_worker_initializer,
        initargs=(str(cache_directory), folds),
    ) as pool:
        for start, block in pool.imap_unordered(_worker_chunk, tasks):
            records[start : start + len(block)] = block
    return records


def balanced_group_blocks(
    labels: np.ndarray, requested_blocks: int, seed: int
) -> tuple[np.ndarray, dict[str, object]]:
    """Pack intact chemical groups into reproducible ligand-balanced blocks."""
    _, inverse, sizes = np.unique(
        np.asarray(labels, dtype=object), return_inverse=True, return_counts=True
    )
    blocks = min(int(requested_blocks), len(sizes))
    rng = np.random.default_rng(seed)
    order = np.lexsort((rng.random(len(sizes)), -sizes))
    loads = np.zeros(blocks, dtype=np.int64)
    assignment = np.empty(len(sizes), dtype=np.int32)
    for group in order:
        destination = int(np.argmin(loads))
        assignment[group] = destination
        loads[destination] += sizes[group]
    return assignment[inverse], {
        "chemical_groups": int(len(sizes)),
        "blocks": int(blocks),
        "minimum_block_ligands": int(loads.min()),
        "median_block_ligands": float(np.median(loads)),
        "maximum_block_ligands": int(loads.max()),
        "block_assignment_seed": int(seed),
    }


def block_raw_moments(
    surface: np.ndarray, row_blocks: np.ndarray, blocks: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    triangle = np.triu_indices(surface.shape[1])
    products = np.column_stack(
        [
            np.bincount(
                row_blocks,
                weights=surface[:, first] * surface[:, second],
                minlength=blocks,
            )
            for first, second in zip(*triangle)
        ]
    )
    sums = np.column_stack(
        [
            np.bincount(row_blocks, weights=surface[:, column], minlength=blocks)
            for column in range(surface.shape[1])
        ]
    )
    counts = np.bincount(row_blocks, minlength=blocks).astype(np.float64)
    return products, sums, counts


def correlation_edges_from_moments(
    weighted_products: np.ndarray,
    weighted_sums: np.ndarray,
    weighted_count: float,
    targets: int,
) -> np.ndarray:
    moment_triangle = np.triu_indices(targets)
    cross = np.zeros((targets, targets), dtype=np.float64)
    cross[moment_triangle] = weighted_products
    cross[(moment_triangle[1], moment_triangle[0])] = weighted_products
    covariance = cross - np.outer(weighted_sums, weighted_sums) / weighted_count
    variances = np.diag(covariance)
    if np.any(variances <= 0):
        raise ValueError("non-positive weighted target variance")
    correlation = covariance / np.sqrt(np.outer(variances, variances))
    return correlation[np.triu_indices(targets, k=1)]


def run_fixed_oof_coarsened_bootstrap(
    surfaces: "OrderedDict[str, np.ndarray]",
    groupings: "OrderedDict[str, np.ndarray]",
    experimental: "OrderedDict[str, np.ndarray]",
    repeats: int,
    requested_blocks: int,
    seed: int,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    """Paired Exp(1) block reweighting of fixed OOF surfaces."""
    width = len(statistic_names())
    records: dict[str, np.ndarray] = {}
    reports: dict[str, object] = OrderedDict()
    for grouping_index, (grouping, labels) in enumerate(groupings.items()):
        assignment_seed = seed + 1_000 + grouping_index * 500_000
        draw_seed = assignment_seed + 100_000
        row_blocks, block_report = balanced_group_blocks(
            labels, requested_blocks, assignment_seed
        )
        blocks = int(block_report["blocks"])
        moments = OrderedDict(
            (
                name,
                block_raw_moments(surface, row_blocks, blocks),
            )
            for name, surface in surfaces.items()
        )
        output = np.full((repeats, width), np.nan, dtype=np.float64)
        rng = np.random.default_rng(draw_seed)
        position = 0
        while position < repeats:
            size = min(BOOTSTRAP_CHUNK, repeats - position)
            weights = rng.exponential(size=(size, blocks))
            edge_blocks: dict[str, np.ndarray] = {}
            for name, (products, sums, counts) in moments.items():
                weighted_products = weights @ products
                weighted_sums = weights @ sums
                weighted_counts = weights @ counts
                edges = np.empty((size, len(TARGETS20) * (len(TARGETS20) - 1) // 2))
                for offset in range(size):
                    edges[offset] = correlation_edges_from_moments(
                        weighted_products[offset],
                        weighted_sums[offset],
                        float(weighted_counts[offset]),
                        len(TARGETS20),
                    )
                edge_blocks[name] = edges
            for offset in range(size):
                geometries = OrderedDict(
                    (name, edge_blocks[name][offset]) for name in SURFACES
                )
                output[position + offset] = replicate_statistics(
                    geometries, experimental
                )
            position += size
        records[grouping] = output
        reports[grouping] = {
            **block_report,
            "repeats": int(repeats),
            "draw_seed": int(draw_seed),
        }
    return records, reports


def summarize_bootstrap(records: dict[str, np.ndarray]) -> dict[str, object]:
    """Per-grouping sensitivity ranges plus their conservative union."""
    names = statistic_names()
    summary: "OrderedDict[str, dict[str, object]]" = OrderedDict()
    for position, name in enumerate(names):
        entry: dict[str, object] = {}
        lows: list[float] = []
        highs: list[float] = []
        for grouping in GROUPINGS:
            column = records[grouping][:, position]
            finite = column[np.isfinite(column)]
            if not len(finite):
                raise ValueError(f"every bootstrap replicate failed for {name}")
            raw = geometry.describe(finite)
            described = {
                key: value
                for key, value in raw.items()
                if key not in {"interval_90", "interval_95"}
            }
            described["conditional_coarsened_sensitivity_interval_90"] = raw[
                "interval_90"
            ]
            described["conditional_coarsened_sensitivity_interval_95"] = raw[
                "interval_95"
            ]
            described["not_a_confidence_interval"] = True
            described["failed_replicates"] = int(len(column) - len(finite))
            entry[grouping] = described
            lows.append(described["conditional_coarsened_sensitivity_interval_95"][0])
            highs.append(described["conditional_coarsened_sensitivity_interval_95"][1])
        entry["conservative_union_sensitivity_interval_95"] = [
            min(lows),
            max(highs),
        ]
        summary[name] = entry
    return summary


# --------------------------------------------------------------------------------------
# The diagnostic that motivates the revision: the descriptor component is not target-blind.
# --------------------------------------------------------------------------------------


def per_target_slope_diagnostic(
    block: np.ndarray,
    descriptors: np.ndarray,
    component_geometry: np.ndarray,
    observed_surface: np.ndarray,
) -> dict[str, object]:
    """Show that the descriptor component's geometry is made of per-target fitted slopes.

    Inside every fold the decomposition regresses each target column separately, so the
    predicted surface is ``X B`` with ``B`` a fitted 7-by-20 slope matrix (plus per-target
    intercepts).  Two consequences are recorded here.

    First, the pooled out-of-fold component still lives, fold by fold, in the eight-
    dimensional column space of the descriptors, so almost all of the trace of its target
    correlation matrix sits in about as many eigenvalues as there are descriptors.

    Second -- and this is the point the reviewer raises -- a genuinely target-blind
    descriptor model would use one common slope vector for all 20 targets.  That surface
    has 20 identical columns, its target correlation matrix is all ones, and its 190-edge
    geometry is constant, so it has no agreement with any experimental geometry at all.
    Every edge of the descriptor component therefore comes from between-target differences
    in the fitted slopes.  The descriptor model does see the protein: through its labels.
    """
    eigenvalues = np.linalg.eigvalsh(component_geometry)[::-1]
    trace = float(eigenvalues.sum())
    descriptor_count = int(descriptors.shape[1])

    standardized = (descriptors - descriptors.mean(axis=0)) / descriptors.std(
        axis=0, ddof=1
    )
    design = np.column_stack([np.ones(len(standardized)), standardized])
    pooled = np.linalg.lstsq(design, observed_surface, rcond=None)[0]
    slopes = pooled[1:]  # descriptors by targets
    common = slopes.mean(axis=1)
    blind_surface = standardized @ common[:, None] * np.ones((1, len(TARGETS20)))
    blind_correlation = np.corrcoef(blind_surface, rowvar=False)
    blind_edges = geometry.upper_triangle(blind_correlation)

    return {
        "descriptor_component_is_built_from_per_target_fitted_slopes": True,
        "fitted_coefficients_per_fold": (descriptor_count + 1) * len(TARGETS20),
        "descriptor_component_geometry_eigenvalues": eigenvalues.tolist(),
        "descriptor_component_geometry_participation_ratio": float(
            mechanism.participation_ratio_from_psd(component_geometry)
        ),
        "trace_fraction_in_leading_seven_eigenvalues": float(
            eigenvalues[:descriptor_count].sum() / trace
        ),
        "eigenvalues_above_one_percent_of_leading": int(
            np.sum(eigenvalues > 0.01 * float(eigenvalues[0]))
        ),
        "pooled_descriptive_target_slopes": {
            name: {
                "minimum_across_targets": float(slopes[index].min()),
                "median_across_targets": float(np.median(slopes[index])),
                "maximum_across_targets": float(slopes[index].max()),
                "between_target_standard_deviation": float(slopes[index].std(ddof=1)),
                "sign_changes_across_targets": bool(
                    slopes[index].min() < 0 < slopes[index].max()
                ),
            }
            for index, name in enumerate(mechanism.DESCRIPTOR_NAMES)
        },
        "target_blind_common_slope_control": {
            "definition": (
                "the same seven descriptors with one slope vector shared by all 20 "
                "targets, i.e. a descriptor model that truly never sees the protein"
            ),
            "minimum_offdiagonal_correlation": float(blind_edges.min()),
            "edge_standard_deviation": float(blind_edges.std(ddof=1)),
            "agreement_with_experiment_is_defined": bool(blind_edges.std(ddof=1) > 1e-9),
            "consequence": (
                "a target-blind descriptor surface has 20 identical columns, a constant "
                "190-edge geometry and therefore no target-pair agreement with any "
                "experimental panel; all of the descriptor component's agreement comes "
                "from between-target differences in the fitted slopes"
            ),
        },
        "interpretation": (
            "The descriptor component surface is X B with X the seven ligand descriptors "
            "and B a fitted matrix of per-target slopes and intercepts. Its 190-edge "
            "geometry is a deterministic function of those fitted per-target "
            "coefficients: target identity enters the model through the labels attached "
            "to the fitted slopes. The surface is descriptor-driven and protein-"
            "structure-free, but it is not target-blind, and this package should not "
            "describe it as target-blind."
        ),
    }


# --------------------------------------------------------------------------------------
# Point estimates.
# --------------------------------------------------------------------------------------


def point_estimates(
    block: np.ndarray,
    descriptors: np.ndarray,
    groups: np.ndarray,
    published: pd.DataFrame,
    published_docking: np.ndarray,
    folds: int,
    permutations: int,
    seed: int,
) -> dict[str, object]:
    decomposition = mechanism.grouped_descriptor_decomposition(
        block, descriptors, groups, folds
    )
    surfaces = OrderedDict(
        [
            ("observed_residual", decomposition["oof_surface"]),
            ("descriptor_component", decomposition["oof_prediction"]),
            ("descriptor_removed", decomposition["oof_error"]),
        ]
    )
    matrices = OrderedDict(
        (name, geometry.target_correlation(surface)) for name, surface in surfaces.items()
    )
    edges = OrderedDict(
        (name, geometry.upper_triangle(matrix)) for name, matrix in matrices.items()
    )
    reconstruction = geometry.geometry_concordance(
        matrices["observed_residual"], component._square_from_edges(published_docking)  # noqa: SLF001
    )
    if reconstruction < 0.99:
        raise ValueError(
            "the reconstructed fixed-20 docking geometry does not match the released "
            f"table (Spearman {reconstruction:.4f}); the support or target order drifted"
        )

    experimental_edges = OrderedDict(
        (
            panel,
            published[f"{panel}_experimental_centered_correlation"].to_numpy(
                dtype=np.float64
            ),
        )
        for panel in PANELS
    )

    absolute_rows: list[dict[str, object]] = []
    joint_rows: list[dict[str, object]] = []
    orthogonal_rows: list[dict[str, object]] = []
    panel_results: "OrderedDict[str, dict[str, object]]" = OrderedDict()
    for panel_index, panel in enumerate(PANELS):
        experimental_matrix = component._square_from_edges(  # noqa: SLF001
            experimental_edges[panel]
        )
        record: dict[str, object] = {"absolute_agreement": OrderedDict()}
        for surface_index, name in enumerate(SURFACES):
            test = geometry.qap_test(
                matrices[name],
                experimental_matrix,
                permutations,
                seed + panel_index * 1_000_000 + surface_index * 10_000,
            )
            jackknife, _ = geometry.target_jackknife(matrices[name], experimental_matrix)
            record["absolute_agreement"][name] = {
                "spearman": test["observed_spearman"],
                "target_label_qap_p_positive": test["one_sided_p_positive"],
                "qap_null_interval_95": test["null_interval_95"],
                "leave_one_target_out": jackknife,
                "seed": test["seed"],
            }
            absolute_rows.append(
                {
                    "panel": panel,
                    "surface": name,
                    "spearman": test["observed_spearman"],
                    "target_label_qap_p_positive": test["one_sided_p_positive"],
                    "leave_one_target_out_minimum": jackknife["minimum"],
                    "leave_one_target_out_median": jackknife["median"],
                    "leave_one_target_out_maximum": jackknife["maximum"],
                }
            )

        joint = mrqap(
            experimental_edges[panel],
            edges["descriptor_component"],
            edges["descriptor_removed"],
            permutations,
            seed + 100 + panel_index,
        )
        fit = joint_edge_regression(
            experimental_edges[panel],
            edges["descriptor_component"],
            edges["descriptor_removed"],
        )
        joint["partial_spearman"] = {
            "descriptor_component": partial_spearman(
                experimental_edges[panel],
                edges["descriptor_component"],
                edges["descriptor_removed"],
            ),
            "descriptor_removed": partial_spearman(
                experimental_edges[panel],
                edges["descriptor_removed"],
                edges["descriptor_component"],
            ),
        }
        joint["single_predictor_rank_r2"] = {
            "descriptor_component": fit["r2_first_only"],
            "descriptor_removed": fit["r2_second_only"],
        }
        joint["semipartial_rank_r2"] = {
            "descriptor_component": fit["semipartial_r2_first"],
            "descriptor_removed": fit["semipartial_r2_second"],
        }
        record["edge_level_joint_decomposition"] = joint
        joint_rows.append(
            {
                "panel": panel,
                "beta_descriptor_component": joint[
                    "standardized_partial_coefficients"
                ]["descriptor_component"],
                "beta_descriptor_removed": joint["standardized_partial_coefficients"][
                    "descriptor_removed"
                ],
                "y_permutation_p_component": joint["y_permutation"][
                    "descriptor_component"
                ]["one_sided_p_positive"],
                "y_permutation_p_removed": joint["y_permutation"]["descriptor_removed"][
                    "one_sided_p_positive"
                ],
                "dekker_p_component": joint["dekker_double_semi_partialling"][
                    "descriptor_component"
                ]["one_sided_p_positive"],
                "dekker_p_removed": joint["dekker_double_semi_partialling"][
                    "descriptor_removed"
                ]["one_sided_p_positive"],
                "partial_spearman_component": joint["partial_spearman"][
                    "descriptor_component"
                ],
                "partial_spearman_removed": joint["partial_spearman"][
                    "descriptor_removed"
                ],
                "joint_rank_r2": joint["joint_rank_r2"],
                "semipartial_rank_r2_component": fit["semipartial_r2_first"],
                "semipartial_rank_r2_removed": fit["semipartial_r2_second"],
            }
        )

        orthogonal: dict[str, object] = {}
        for side, (focal, control) in enumerate(
            (
                ("descriptor_component", "descriptor_removed"),
                ("descriptor_removed", "descriptor_component"),
            )
        ):
            residual = orthogonalize(edges[focal], edges[control])
            incremental = float(
                stats.spearmanr(experimental_edges[panel], residual).statistic
            )
            residual_test = geometry.qap_test(
                component._square_from_edges(residual),  # noqa: SLF001
                experimental_matrix,
                permutations,
                seed + 200_000 + panel_index * 100 + side,
            )
            semipartial = (
                fit["semipartial_r2_first"] if side == 0 else fit["semipartial_r2_second"]
            )
            orthogonal[focal] = {
                "residualized_against": control,
                "incremental_spearman": incremental,
                "target_label_qap_p_positive": residual_test["one_sided_p_positive"],
                "semipartial_rank_r2": semipartial,
                "seed": residual_test["seed"],
            }
            orthogonal_rows.append(
                {
                    "panel": panel,
                    "component": focal,
                    "residualized_against": control,
                    "unadjusted_spearman": float(
                        stats.spearmanr(
                            experimental_edges[panel], edges[focal]
                        ).statistic
                    ),
                    "incremental_spearman": incremental,
                    "target_label_qap_p_positive": residual_test[
                        "one_sided_p_positive"
                    ],
                    "semipartial_rank_r2": semipartial,
                }
            )
        record["orthogonalized_components"] = orthogonal

        observed = record["absolute_agreement"]["observed_residual"]["spearman"]
        predicted = record["absolute_agreement"]["descriptor_component"]["spearman"]
        record["retired_mediated_share_ratio"] = {
            "value": float(predicted / observed),
            "why_retired": (
                "a ratio of two Spearman correlations is not a mediation fraction: it is "
                "not additive, its numerator and denominator are correlated, and it is "
                "undefined or explosive as the denominator approaches zero"
            ),
        }
        panel_results[panel] = record

    component_remainder = {
        "spearman": float(
            stats.spearmanr(
                edges["descriptor_component"], edges["descriptor_removed"]
            ).statistic
        ),
        "pearson": float(
            stats.pearsonr(
                edges["descriptor_component"], edges["descriptor_removed"]
            ).statistic
        ),
        "note": (
            "the two components of the decomposition are not orthogonal at the edge level, "
            "so their separate agreements cannot be added or ratioed"
        ),
    }

    target_blindness = per_target_slope_diagnostic(
        block, descriptors, matrices["descriptor_component"], surfaces["observed_residual"]
    )

    return {
        "estimates": {
            "released_geometry_reconstruction_spearman": reconstruction,
            "decomposition_metrics": decomposition["metrics"],
            "experimental_panel_results": panel_results,
            "component_versus_remainder_edge_correlation": component_remainder,
            "per_target_slope_diagnostic": target_blindness,
        },
        "edges": edges,
        "surfaces": surfaces,
        "absolute_table": pd.DataFrame.from_records(absolute_rows),
        "joint_table": pd.DataFrame.from_records(joint_rows),
        "orthogonal_table": pd.DataFrame.from_records(orthogonal_rows),
    }


# --------------------------------------------------------------------------------------
# Driver.
# --------------------------------------------------------------------------------------


def run(
    *,
    pkis1_zip: Path,
    published_geometry: Path,
    folds: int,
    permutations: int,
    bootstrap_repeats: int,
    bootstrap_blocks: int,
    bootstrap_mode: str,
    seed: int,
    workers: int,
) -> dict[str, object]:
    reference, score_columns, smiles, support = component.load_deleaked_reference(
        pkis1_zip
    )
    published, published_docking = component.load_published_experimental_geometry(
        published_geometry
    )
    score_index = {target: position for position, target in enumerate(score_columns)}
    block = reference[:, [score_index[target] for target in TARGETS20]]
    descriptors = mechanism.molecular_descriptors(smiles)
    murcko = mechanism.scaffold_keys(smiles)
    if not np.isfinite(descriptors).all():
        raise AssertionError("validated descriptor matrix unexpectedly became non-finite")
    generic = generic_framework_keys(smiles)

    point = point_estimates(
        block,
        descriptors,
        murcko,
        published,
        published_docking,
        folds,
        permutations,
        seed,
    )
    estimates = point["estimates"]
    edges = point["edges"]
    bootstrap_reports: dict[str, object] = OrderedDict()
    if bootstrap_mode == "fixed_oof_coarsened":
        experimental = OrderedDict(
            (
                panel,
                published[
                    f"{panel}_experimental_centered_correlation"
                ].to_numpy(dtype=np.float64),
            )
            for panel in PANELS
        )
        records, bootstrap_reports = run_fixed_oof_coarsened_bootstrap(
            point["surfaces"],
            OrderedDict(
                [
                    ("murcko_scaffold", murcko),
                    ("murcko_generic_framework", generic),
                ]
            ),
            experimental,
            bootstrap_repeats,
            bootstrap_blocks,
            seed,
        )
    elif bootstrap_mode == "full_refit":
        with tempfile.TemporaryDirectory(
            prefix="descriptor_geometry_decomposition_"
        ) as tmp:
            cache = Path(tmp)
            np.save(cache / "block.npy", block)
            np.save(cache / "descriptors.npy", descriptors)
            np.save(
                cache / "experimental.npy",
                np.stack(
                    [
                        published[
                            f"{panel}_experimental_centered_correlation"
                        ].to_numpy(dtype=np.float64)
                        for panel in PANELS
                    ]
                ),
            )
            # Fixed-width Unicode keeps the temporary cache non-executable.  Object arrays
            # require pickle on load and are unnecessary for plain scaffold labels.
            np.save(
                cache / "groups_murcko_scaffold.npy",
                np.asarray(murcko, dtype=str),
            )
            np.save(
                cache / "groups_murcko_generic_framework.npy",
                np.asarray(generic, dtype=str),
            )
            records = {
                grouping: run_bootstrap(
                    cache,
                    grouping,
                    bootstrap_repeats,
                    seed + 1_000 + 500_000 * index,
                    folds,
                    workers,
                )
                for index, grouping in enumerate(GROUPINGS)
            }
    else:
        raise ValueError(f"unknown bootstrap mode: {bootstrap_mode}")

    bootstrap = summarize_bootstrap(records)

    summary: dict[str, object] = {
        "analysis": (
            "edge-level decomposition of docking-experiment target-pair agreement into a "
            "descriptor-predicted component and a descriptor-removed remainder, with "
            "conditional coarsened chemical-group sensitivity intervals and MRQAP "
            "probabilities"
        ),
        "question": (
            "On the fixed 190 edges of the common-20 kinase panel, how much of the "
            "agreement between centred Vina target-pair geometry and experimental kinase "
            "co-response is carried by the descriptor-predicted component, how much by "
            "the descriptor-removed remainder, and how sensitive are those fixed-OOF "
            "edge statistics to paired chemical-group reweighting?"
        ),
        "status": "exploratory_post_hoc_science_only",
        "replaces": {
            "retired_statistic": (
                "rho(descriptor component, experiment) / rho(observed residual, "
                "experiment), a tempting but prohibited 'mediated share' reading"
            ),
            "reason": (
                "The current manuscript already declines this reading. The ratio is not "
                "a mediation fraction: its two terms are Spearman "
                "correlations of two non-orthogonal edge vectors with the same "
                "experimental edge vector; the quantity is not additive, has no variance "
                "estimate, and diverges as the denominator approaches zero. Its own "
                "conditional chemical-group sensitivity interval, reported here under "
                "retired_mediated_share_ratio, shows the instability directly."
            ),
            "replacement": (
                "three absolute agreements with sensitivity intervals, plus standardized partial "
                "coefficients with MRQAP probabilities, partial Spearman correlations, "
                "and semipartial rank R-squared for each component"
            ),
        },
        "support": {
            **support,
            "ligands_after_descriptor_filter": int(len(block)),
            "targets": len(TARGETS20),
            "target_pairs": int(len(published)),
            "cells": int(len(block) * len(TARGETS20)),
            "experimental_panels": list(PANELS),
            "murcko_scaffold_groups": int(len(np.unique(murcko))),
            "murcko_generic_framework_groups": int(len(np.unique(generic))),
        },
        "targets": list(TARGETS20),
        "surfaces": dict(component.SURFACE_LABELS),
        "descriptors": list(mechanism.DESCRIPTOR_NAMES),
        **estimates,
        "chemical_group_bootstrap": {
            "resampled_unit": "chemical group of the DOCKSTRING ligand support",
            "groupings": list(GROUPINGS),
            "butina_note": (
                "The package's usual second grouping is Butina at Tanimoto 0.65, but the "
                "package helper materializes all pairwise distances, which is 3.4e10 "
                "pairs on this 259,579-ligand support and is not computable. The coarser "
                "Bemis-Murcko generic framework is substituted; it strictly coarsens the "
                "Murcko grouping and therefore plays the same conservative role."
            ),
            "what_is_reweighted": (
                "The three already group-held-out OOF surfaces are fixed. Intact chemical "
                "groups are packed into balanced blocks, paired iid Exp(1) weights are "
                "applied to all three surfaces, and their correlation geometries and every "
                "edge-level statistic are recomputed from weighted raw moments."
            ),
            "conditioning_and_coarsening_limitation": (
                "Predictive models are not refitted inside draws, and unrelated groups in "
                "one balanced block share a weight. These outputs are conditional "
                "coarsened sensitivity intervals, not individual-cluster, chemical-space, "
                "experimental-measurement or target-population confidence intervals."
            ),
            "what_is_held_fixed": (
                "The four experimental target-pair geometries are held at their published "
                "190-edge values and the 20-target panel is fixed. These are docking-side "
                "sensitivity intervals; they do not carry experimental measurement uncertainty and "
                "they do not carry uncertainty over which kinases are on the panel, which "
                "the reported leave-one-target-out range addresses separately."
            ),
            "replicates": int(bootstrap_repeats),
            "mode": bootstrap_mode,
            "requested_blocks": int(bootstrap_blocks),
            "block_reports": bootstrap_reports,
            "seeds": {
                grouping: int(seed + 1_000 + 500_000 * index)
                for index, grouping in enumerate(GROUPINGS)
            },
            "replicate_seed_rule": (
                "fixed_oof_coarsened uses one seeded Exp(1) block-weight stream per "
                "grouping; full_refit uses default_rng(grouping_seed + replicate_index)"
            ),
            "sensitivity_interval": "percentile, 2.5% and 97.5%",
            "union_rule": (
                "conservative union across groupings: the lowest lower bound and the "
                "highest upper bound"
            ),
            "statistics": bootstrap,
        },
        "configuration": {
            "folds": int(folds),
            "group_definition_for_holdout": (
                "RDKit Bemis-Murcko scaffolds; every acyclic molecule is a singleton"
            ),
            "permutations": int(permutations),
            "permutation_unit": "complete target labels",
            "seed": int(seed),
            "bootstrap_mode": bootstrap_mode,
            "bootstrap_blocks": int(bootstrap_blocks),
            "blas_threads": 1,
            "determinism": (
                "fixed_oof_coarsened uses a deterministic seeded block-weight stream; "
                "full_refit seeds every replicate from its own index. BLAS is pinned to "
                "one thread, so the artifact does not depend on worker count and no "
                "wall-clock value is written"
            ),
            "fixed_endpoint": "two-way-centered experimental target correlation",
            "experimental_geometry_source": str(
                published_geometry.resolve().relative_to(PACKAGE)
            ),
        },
        "claim_boundary": (
            "This does NOT show that the docking-experiment agreement is caused by ligand "
            "physicochemistry, and it does NOT show that the descriptor component is "
            "target-blind. The descriptor component is fitted with separate regression "
            "coefficients for every target, so its 190-edge geometry is a deterministic "
            "function of the fitted per-target slopes and target identity enters through "
            "them; the manuscript is not entitled to call this surface target-blind. It "
            "also does NOT show that the descriptor-removed remainder is mechanistic: the "
            "remainder is only what a seven-descriptor linear model failed to predict. "
            "The default paired bootstrap only reweights coarsened blocks of the fixed "
            "OOF DOCKSTRING surfaces and does not refit the descriptor model; the four "
            "experimental geometries and the 20-kinase panel are fixed, so the intervals "
            "are not experimental-measurement intervals and the MRQAP probabilities are "
            "conditional on this panel. The retired ratio is reported only to document "
            "its instability, not as a result."
        ),
    }
    return {
        "summary": summary,
        "absolute_table": point["absolute_table"],
        "joint_table": point["joint_table"],
        "orthogonal_table": point["orthogonal_table"],
        "bootstrap_records": records,
        "edges": edges,
        "published": published,
    }


def bootstrap_quantile_table(bootstrap: dict[str, object]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for name, entry in bootstrap.items():
        panel, _, statistic = name.partition("::")
        if not statistic:
            panel, statistic = "", name
        row: dict[str, object] = {"panel": panel, "statistic": statistic}
        for grouping in GROUPINGS:
            described = entry[grouping]  # type: ignore[index]
            row[f"{grouping}_median"] = described["median"]
            interval = described[
                "conditional_coarsened_sensitivity_interval_95"
            ]
            row[f"{grouping}_sensitivity_low95"] = interval[0]
            row[f"{grouping}_sensitivity_high95"] = interval[1]
            row[f"{grouping}_failed_replicates"] = described["failed_replicates"]
        union = entry["conservative_union_sensitivity_interval_95"]  # type: ignore[index]
        row["union_sensitivity_low95"] = union[0]
        row["union_sensitivity_high95"] = union[1]
        rows.append(row)
    return pd.DataFrame.from_records(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pkis1-zip", type=Path, default=component.DEFAULT_PKIS1_ZIP)
    parser.add_argument(
        "--published-geometry",
        type=Path,
        default=component.DEFAULT_PUBLISHED_GEOMETRY,
    )
    parser.add_argument("--folds", type=int, default=DEFAULT_FOLDS)
    parser.add_argument("--permutations", type=int, default=DEFAULT_PERMUTATIONS)
    parser.add_argument(
        "--bootstrap-repeats", type=int, default=DEFAULT_BOOTSTRAP_REPEATS
    )
    parser.add_argument("--bootstrap-blocks", type=int, default=DEFAULT_BOOTSTRAP_BLOCKS)
    parser.add_argument(
        "--bootstrap-mode",
        choices=("fixed_oof_coarsened", "full_refit"),
        default=DEFAULT_BOOTSTRAP_MODE,
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    result = run(
        pkis1_zip=args.pkis1_zip,
        published_geometry=args.published_geometry,
        folds=args.folds,
        permutations=args.permutations,
        bootstrap_repeats=args.bootstrap_repeats,
        bootstrap_blocks=args.bootstrap_blocks,
        bootstrap_mode=args.bootstrap_mode,
        seed=args.seed,
        workers=max(1, args.workers),
    )
    summary = result["summary"]
    args.output.mkdir(parents=True, exist_ok=True)
    mechanism.write_json(args.output / "summary.json", summary)
    mechanism.write_csv(
        result["absolute_table"], args.output / "panel_absolute_agreement.csv"
    )
    mechanism.write_csv(
        result["joint_table"], args.output / "edge_level_joint_regression.csv"
    )
    mechanism.write_csv(
        result["orthogonal_table"], args.output / "component_orthogonalization.csv"
    )
    mechanism.write_csv(
        bootstrap_quantile_table(summary["chemical_group_bootstrap"]["statistics"]),
        args.output / "bootstrap_intervals.csv",
    )
    edge_frame = pd.DataFrame(
        {
            "target_a": result["published"].target_a,
            "target_b": result["published"].target_b,
            **{
                f"docking_{name}_correlation": values
                for name, values in result["edges"].items()
            },
            **{
                f"{panel}_experimental_centered_correlation": result["published"][
                    f"{panel}_experimental_centered_correlation"
                ]
                for panel in PANELS
            },
        }
    )
    mechanism.write_csv(edge_frame, args.output / "target_pair_edges.csv")
    print(result["absolute_table"].to_string(index=False))
    print()
    print(result["joint_table"].to_string(index=False))


if __name__ == "__main__":
    main()
