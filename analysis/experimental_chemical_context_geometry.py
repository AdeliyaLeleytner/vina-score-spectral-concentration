#!/usr/bin/env python3
"""Chemical-domain dependence of dense experimental target geometry.

This science-only exploratory analysis asks whether the target--target
correlation network changes across ligand molecular-weight domains in three
dense kinase panels (PKIS2, PKIS1, and DAVIS).  Molecular weight and all
thresholds are explicitly post hoc.  The primary statistic is Spearman
agreement between the strict upper triangles of two-way-centered target
correlation matrices in the lowest and highest molecular-weight quartiles.

A random disjoint-support null quantifies finite-ligand sampling disagreement.
Murcko- and Butina-cluster bootstraps preserve chemical dependence, and a
target jackknife measures dependence on the fixed target panel.  A separate
matched-ligand analysis compares experimental and DOCKSTRING low-to-high-MW
network changes; it is descriptive/exploratory and is never substituted for
the primary within-experiment analysis.

No manuscript file is read or modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from scipy import stats

import dense_davis_benchmark as davis
import residual_target_geometry_validation as geometry


PACKAGE = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PACKAGE / "results" / "experimental_chemical_context_geometry"
DEFAULT_PKIS1_ZIP = Path("/tmp/pkis1_supplement.zip")
DEFAULT_RANDOM_REPETITIONS = 1_000
DEFAULT_BOOTSTRAPS = 1_000
DEFAULT_QAP_PERMUTATIONS = 10_000
DEFAULT_SEED = 202_608_15
EXTREME_FRACTION = 0.25
QUINTILES = 5

# PKIS1 fixes the exact common support: 20 targets, excluding PTK2/FAK.
TARGETS = tuple(geometry.PKIS1_TARGET_MAP)
if not set(TARGETS).issubset(davis.TARGET_MAP):
    raise RuntimeError("fixed PKIS1 targets are not contained in DAVIS/PKIS2")


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float) and not np.isfinite(value):
        raise ValueError("non-finite value cannot be serialized")
    return value


def molecular_weights(smiles: pd.Series) -> np.ndarray:
    values: list[float] = []
    for value in smiles.astype(str):
        molecule = Chem.MolFromSmiles(value)
        if molecule is None:
            raise ValueError("an experimental SMILES cannot be parsed")
        values.append(float(Descriptors.MolWt(molecule)))
    result = np.asarray(values, dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError("molecular weights contain a non-finite value")
    return result


def row_center(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2 or min(matrix.shape) < 2 or not np.isfinite(matrix).all():
        raise ValueError("row_center requires a finite two-dimensional matrix")
    return matrix - matrix.mean(axis=1, keepdims=True)


def target_correlation(
    matrix: np.ndarray, weights: np.ndarray | None = None
) -> np.ndarray:
    """Target correlation; integer weights are equivalent to repeated rows."""
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] < 3 or matrix.shape[1] < 2:
        raise ValueError("target correlation requires >=3 ligands and >=2 targets")
    if not np.isfinite(matrix).all():
        raise ValueError("target-correlation input contains non-finite values")
    if weights is None:
        centered = matrix - matrix.mean(axis=0, keepdims=True)
        covariance = centered.T @ centered
    else:
        weights = np.asarray(weights, dtype=np.float64)
        if weights.shape != (len(matrix),) or np.any(weights < 0):
            raise ValueError("weights must be nonnegative and align to rows")
        total = float(weights.sum())
        if total <= 2:
            raise ValueError("weighted target correlation has <=2 effective rows")
        weighted_sum = weights @ matrix
        covariance = matrix.T @ (matrix * weights[:, None])
        covariance -= np.outer(weighted_sum, weighted_sum) / total
    variance = np.diag(covariance)
    if np.any(variance <= 1e-14):
        raise ValueError("target correlation contains a constant target")
    scale = np.sqrt(variance)
    result = covariance / np.outer(scale, scale)
    result = np.clip((result + result.T) / 2.0, -1.0, 1.0)
    np.fill_diagonal(result, 1.0)
    return result


def residual_geometry(matrix: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    return target_correlation(row_center(matrix), weights)


def upper(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("upper requires a square matrix")
    return matrix[np.triu_indices(len(matrix), k=1)]


def pr_dimension(correlation: np.ndarray) -> float:
    correlation = np.asarray(correlation, dtype=np.float64)
    return float(np.trace(correlation) ** 2 / np.square(correlation).sum())


def geometry_comparison(first: np.ndarray, second: np.ndarray) -> dict[str, float]:
    a, b = upper(first), upper(second)
    rho = float(stats.spearmanr(a, b).statistic)
    if not np.isfinite(rho):
        raise ValueError("geometry agreement is not finite")
    top_count = max(1, int(np.ceil(0.10 * len(a))))
    top_a = np.zeros(len(a), dtype=bool)
    top_b = np.zeros(len(b), dtype=bool)
    top_a[np.argsort(a, kind="mergesort")[-top_count:]] = True
    top_b[np.argsort(b, kind="mergesort")[-top_count:]] = True
    return {
        "geometry_spearman": rho,
        "geometry_dissimilarity": 1.0 - rho,
        "sign_flip_fraction": float(np.mean(np.sign(a) != np.sign(b))),
        "top_positive_10pct_pair_jaccard": float(
            np.sum(top_a & top_b) / np.sum(top_a | top_b)
        ),
        "first_pr": pr_dimension(first),
        "second_pr": pr_dimension(second),
        "first_mean_absolute_correlation": float(np.mean(np.abs(a))),
        "second_mean_absolute_correlation": float(np.mean(np.abs(b))),
    }


def extreme_masks(mw: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
    mw = np.asarray(mw, dtype=np.float64)
    low_cut, high_cut = np.quantile(mw, [EXTREME_FRACTION, 1 - EXTREME_FRACTION])
    low, high = mw <= low_cut, mw >= high_cut
    if np.any(low & high) or min(low.sum(), high.sum()) < 3:
        raise ValueError("MW quartiles do not define valid disjoint supports")
    return low, high, float(low_cut), float(high_cut)


def quantile_ids(values: np.ndarray, bins: int = QUINTILES) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.float64)
    edges = np.quantile(values, np.linspace(0.0, 1.0, bins + 1))
    if np.any(np.diff(edges) <= 0):
        raise ValueError("tied quantiles prevent requested binning")
    identifiers = np.searchsorted(edges[1:-1], values, side="left")
    return identifiers.astype(np.int16), edges


def random_disjoint_null(
    matrix: np.ndarray,
    first_size: int,
    second_size: int,
    repetitions: int,
    seed: int,
) -> np.ndarray:
    if first_size + second_size > len(matrix):
        raise ValueError("requested random supports are not disjoint-feasible")
    rng = np.random.default_rng(seed)
    result = np.empty(repetitions, dtype=np.float64)
    for repetition in range(repetitions):
        order = rng.permutation(len(matrix))
        first = residual_geometry(matrix[order[:first_size]])
        second = residual_geometry(matrix[order[first_size:first_size + second_size]])
        result[repetition] = stats.spearmanr(upper(first), upper(second)).statistic
    return result


def _cluster_weights(
    labels: np.ndarray, mask: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    indices = np.flatnonzero(mask)
    groups, inverse = np.unique(np.asarray(labels)[indices], return_inverse=True)
    multiplicity = np.bincount(
        rng.integers(0, len(groups), size=len(groups)), minlength=len(groups)
    )
    weights = np.zeros(len(labels), dtype=np.float64)
    weights[indices] = multiplicity[inverse]
    return weights


def cluster_bootstrap(
    matrix: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
    labels: np.ndarray,
    repetitions: int,
    seed: int,
) -> np.ndarray:
    """Independently resample whole chemical groups within each MW domain."""
    rng = np.random.default_rng(seed)
    centered = row_center(matrix)
    result = np.empty(repetitions, dtype=np.float64)
    for repetition in range(repetitions):
        low_w = _cluster_weights(labels, low, rng)
        high_w = _cluster_weights(labels, high, rng)
        low_geometry = target_correlation(centered, low_w)
        high_geometry = target_correlation(centered, high_w)
        result[repetition] = stats.spearmanr(
            upper(low_geometry), upper(high_geometry)
        ).statistic
    return result


def load_experimental_panels(pkis1_zip: Path) -> OrderedDict[str, dict[str, Any]]:
    # DAVIS: reconstruct only the released dense experimental block, avoiding a
    # needless 260k-ligand identity scan.
    raw_davis = pd.read_csv(
        davis.DEFAULT_DAVIS,
        sep="\t",
        usecols=["drug_name", "protein", "compound_iso_smiles", "y"],
    )
    molecules = raw_davis[["drug_name", "compound_iso_smiles"]].drop_duplicates()
    if molecules["drug_name"].duplicated().any():
        raise ValueError("DAVIS drug names do not map one-to-one to SMILES")
    selected = raw_davis[raw_davis.protein.isin(davis.TARGET_MAP.values())]
    davis_matrix = selected.pivot(index="drug_name", columns="protein", values="y")
    davis_matrix = davis_matrix.reindex(
        index=molecules.drug_name,
        columns=[davis.TARGET_MAP[target] for target in TARGETS],
    )
    davis_matrix.columns = list(TARGETS)
    if davis_matrix.shape != (72, 20) or davis_matrix.isna().any().any():
        raise ValueError("expected a dense DAVIS 72 x 20 block")

    pkis2 = geometry.load_pkis2_full()
    pkis1 = geometry.load_pkis1_full(pkis1_zip)
    panels: OrderedDict[str, dict[str, Any]] = OrderedDict(
        [
            (
                "PKIS2",
                {
                    "matrix": pkis2[list(TARGETS)].to_numpy(float),
                    "smiles": pkis2["Smiles"].reset_index(drop=True),
                    "keys": pkis2["standard_inchikey"].to_numpy(object),
                    "row_labels": pkis2["standard_inchikey"].to_numpy(object),
                },
            ),
            (
                "PKIS1",
                {
                    "matrix": pkis1[list(TARGETS)].to_numpy(float),
                    "smiles": pkis1["SMILES"].reset_index(drop=True),
                    "keys": pkis1["standard_inchikey"].to_numpy(object),
                    "row_labels": pkis1["standard_inchikey"].to_numpy(object),
                },
            ),
            (
                "DAVIS",
                {
                    "matrix": davis_matrix.to_numpy(float),
                    "smiles": molecules["compound_iso_smiles"].reset_index(drop=True),
                    "keys": molecules["drug_name"].to_numpy(object),
                    "row_labels": molecules["drug_name"].to_numpy(object),
                },
            ),
        ]
    )
    return panels


def summarize_distribution(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    return {
        "mean": float(values.mean()),
        "sd": float(values.std(ddof=1)),
        "q025": float(np.quantile(values, 0.025)),
        "median": float(np.median(values)),
        "q975": float(np.quantile(values, 0.975)),
    }


def panel_analysis(
    panel: str,
    matrix: np.ndarray,
    smiles: pd.Series,
    random_repetitions: int,
    bootstraps: int,
    seed: int,
) -> tuple[dict[str, Any], list[dict], list[dict], list[dict], list[dict]]:
    mw = molecular_weights(smiles)
    low, high, low_cut, high_cut = extreme_masks(mw)
    low_geometry = residual_geometry(matrix[low])
    high_geometry = residual_geometry(matrix[high])
    observed = geometry_comparison(low_geometry, high_geometry)

    null = random_disjoint_null(
        matrix, int(low.sum()), int(high.sum()), random_repetitions, seed
    )
    null_p = float((1 + np.sum(null <= observed["geometry_spearman"])) / (len(null) + 1))

    murcko = davis._murcko_labels(smiles.reset_index(drop=True))
    butina = davis._butina_labels(smiles.reset_index(drop=True), similarity_threshold=0.65)
    bootstrap_rows: list[dict] = []
    bootstrap_summaries: dict[str, dict] = {}
    for offset, (method, labels) in enumerate((("murcko", murcko), ("butina_0.65", butina))):
        values = cluster_bootstrap(
            matrix, low, high, labels, bootstraps, seed + 100 + offset
        )
        bootstrap_summaries[method] = {
            "chemical_groups_total": int(len(np.unique(labels))),
            "low_groups": int(len(np.unique(labels[low]))),
            "high_groups": int(len(np.unique(labels[high]))),
            **summarize_distribution(values),
        }
        bootstrap_rows.extend(
            {
                "panel": panel,
                "grouping": method,
                "repetition": repetition,
                "geometry_spearman": float(value),
            }
            for repetition, value in enumerate(values)
        )

    null_rows = [
        {"panel": panel, "repetition": index, "geometry_spearman": float(value)}
        for index, value in enumerate(null)
    ]

    bin_ids, edges = quantile_ids(mw)
    bin_geometries = [residual_geometry(matrix[bin_ids == index]) for index in range(QUINTILES)]
    quintile_rows: list[dict] = []
    pair_rows: list[dict] = []
    medians = [float(np.median(mw[bin_ids == index])) for index in range(QUINTILES)]
    for index, corr in enumerate(bin_geometries):
        quintile_rows.append(
            {
                "panel": panel,
                "mw_quintile": index + 1,
                "ligands": int(np.sum(bin_ids == index)),
                "mw_lower_edge": float(edges[index]),
                "mw_upper_edge": float(edges[index + 1]),
                "mw_median": medians[index],
                "residual_pr": pr_dimension(corr),
                "mean_absolute_correlation": float(np.mean(np.abs(upper(corr)))),
            }
        )
    for first in range(QUINTILES):
        for second in range(first + 1, QUINTILES):
            comparison = geometry_comparison(bin_geometries[first], bin_geometries[second])
            pair_rows.append(
                {
                    "panel": panel,
                    "first_quintile": first + 1,
                    "second_quintile": second + 1,
                    "median_mw_separation": medians[second] - medians[first],
                    **comparison,
                }
            )
    separation_rho = float(
        stats.spearmanr(
            [row["median_mw_separation"] for row in pair_rows],
            [row["geometry_dissimilarity"] for row in pair_rows],
        ).statistic
    )

    jackknife_rows: list[dict] = []
    for removed in range(len(TARGETS)):
        keep = np.arange(len(TARGETS)) != removed
        comparison = geometry_comparison(
            residual_geometry(matrix[low][:, keep]),
            residual_geometry(matrix[high][:, keep]),
        )
        jackknife_rows.append(
            {"panel": panel, "removed_target": TARGETS[removed], **comparison}
        )

    summary = {
        "ligands": int(len(matrix)),
        "targets": int(matrix.shape[1]),
        "mw_post_hoc": True,
        "low_quartile_ligands": int(low.sum()),
        "high_quartile_ligands": int(high.sum()),
        "low_mw_cut": low_cut,
        "high_mw_cut": high_cut,
        "observed": observed,
        "random_disjoint_null": {
            "repetitions": random_repetitions,
            "lower_tail_monte_carlo_p": null_p,
            **summarize_distribution(null),
        },
        "chemical_group_bootstrap": bootstrap_summaries,
        "quintile_separation_dissimilarity_spearman": separation_rho,
        "target_jackknife_geometry_spearman_range": [
            float(min(row["geometry_spearman"] for row in jackknife_rows)),
            float(max(row["geometry_spearman"] for row in jackknife_rows)),
        ],
    }
    return summary, null_rows, bootstrap_rows, quintile_rows, pair_rows + jackknife_rows


def _parse_row_indices(value: object) -> list[int]:
    return [int(item) for item in str(value).split(";") if str(item).strip()]


def _dock_profiles_by_rows(
    dockstring: pd.DataFrame, row_index_strings: list[object]
) -> np.ndarray:
    rows: list[np.ndarray] = []
    for value in row_index_strings:
        indices = _parse_row_indices(value)
        if not indices:
            raise ValueError("a frozen mapping has no DOCKSTRING row index")
        rows.append(dockstring.iloc[indices][list(TARGETS)].to_numpy(float).mean(axis=0))
    return np.asarray(rows, dtype=np.float64)


def delta_geometry(matrix: np.ndarray, mw: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    low, high, _, _ = extreme_masks(mw)
    low_corr = residual_geometry(matrix[low])
    high_corr = residual_geometry(matrix[high])
    return high_corr - low_corr, low, high


def direction_qap(
    experimental_delta: np.ndarray,
    docking_delta: np.ndarray,
    permutations: int,
    seed: int,
) -> dict[str, float | int | list[float]]:
    observed = float(stats.spearmanr(upper(experimental_delta), upper(docking_delta)).statistic)
    rng = np.random.default_rng(seed)
    null = np.empty(permutations, dtype=np.float64)
    for repetition in range(permutations):
        order = rng.permutation(len(docking_delta))
        permuted = docking_delta[np.ix_(order, order)]
        null[repetition] = stats.spearmanr(upper(experimental_delta), upper(permuted)).statistic
    return {
        "direction_spearman": observed,
        "sign_agreement_fraction": float(
            np.mean(np.sign(upper(experimental_delta)) == np.sign(upper(docking_delta)))
        ),
        "target_label_qap_p_positive": float((1 + np.sum(null >= observed)) / (permutations + 1)),
        "qap_null_interval_95": [float(np.quantile(null, 0.025)), float(np.quantile(null, 0.975))],
        "qap_permutations": int(permutations),
    }


def holm_adjust(p_values: list[float]) -> list[float]:
    """Holm family-wise adjustment in the original input order."""
    values = np.asarray(p_values, dtype=np.float64)
    order = np.argsort(values)
    adjusted = np.empty(len(values), dtype=np.float64)
    running = 0.0
    for rank, index in enumerate(order):
        candidate = (len(values) - rank) * values[index]
        running = max(running, candidate)
        adjusted[index] = min(1.0, running)
    return adjusted.tolist()


def experimental_cross_panel_directions(
    panels: OrderedDict[str, dict[str, Any]],
    permutations: int,
    seed: int,
) -> list[dict[str, Any]]:
    deltas = {
        name: delta_geometry(
            np.asarray(panel["matrix"], float),
            molecular_weights(pd.Series(panel["smiles"])),
        )[0]
        for name, panel in panels.items()
    }
    comparisons = (("PKIS2", "PKIS1"), ("PKIS2", "DAVIS"), ("PKIS1", "DAVIS"))
    rows: list[dict[str, Any]] = []
    for offset, (first, second) in enumerate(comparisons):
        rows.append(
            {
                "first_panel": first,
                "second_panel": second,
                **direction_qap(
                    deltas[first], deltas[second], permutations, seed + offset
                ),
            }
        )
    adjusted = holm_adjust(
        [float(row["target_label_qap_p_positive"]) for row in rows]
    )
    for row, value in zip(rows, adjusted):
        row["holm_p_across_three_panel_pairs"] = float(value)
    return rows


def matched_dockstring_analysis(
    panels: OrderedDict[str, dict[str, Any]],
    qap_permutations: int,
    seed: int,
) -> list[dict[str, Any]]:
    mapping_specs = {
        "PKIS2": (
            PACKAGE / "results" / "dense_pkis2_primary_molecule_mapping.csv",
            "standard_inchikey",
            "standard_inchikey",
        ),
        "DAVIS": (
            PACKAGE / "results" / "dense_davis_primary_molecule_mapping.csv",
            "drug_name",
            "drug_name",
        ),
    }
    # Frozen mapping indices refer to the manuscript's 260,060-row support,
    # defined by completeness across all 58 score columns rather than only the
    # 20 columns used here.
    dockstring_all = pd.read_csv(davis.DEFAULT_DOCKSTRING, sep="\t")
    score_columns = [
        column for column in dockstring_all if column not in {"inchikey", "smiles"}
    ]
    complete = ~dockstring_all[score_columns].isna().any(axis=1)
    dockstring = dockstring_all.loc[complete, ["inchikey", "smiles", *TARGETS]].reset_index(drop=True)
    if len(dockstring) != 260060:
        raise ValueError("DOCKSTRING complete-row support changed")

    rows: list[dict[str, Any]] = []
    for offset, (panel_name, (mapping_path, mapping_key, panel_key)) in enumerate(mapping_specs.items()):
        mapping = pd.read_csv(mapping_path)
        panel = panels[panel_name]
        panel_labels = np.asarray(panel["row_labels"], object).astype(str)
        mapping = mapping[
            mapping[mapping_key].astype(str).isin(set(panel_labels))
        ].copy()
        mapping = mapping.sort_values(mapping_key).reset_index(drop=True)
        experimental_rows: list[np.ndarray] = []
        smiles_rows: list[str] = []
        for key in mapping[mapping_key].astype(str):
            positions = np.flatnonzero(panel_labels == key)
            if len(positions) < 1:
                raise RuntimeError("frozen mapping key disappeared from panel")
            # Match the frozen benchmark's duplicate handling rather than
            # selecting an arbitrary duplicate experimental row.
            experimental_rows.append(
                np.median(np.asarray(panel["matrix"], float)[positions], axis=0)
            )
            smiles_rows.append(str(pd.Series(panel["smiles"]).iloc[positions[0]]))
        experiment = np.asarray(experimental_rows, dtype=np.float64)
        smiles = pd.Series(smiles_rows)
        docking_unclipped = _dock_profiles_by_rows(
            dockstring, mapping.dockstring_row_indices.tolist()
        )
        mw = molecular_weights(smiles)
        experiment_delta, low, high = delta_geometry(experiment, mw)
        for representation_offset, (preprocessing, docking) in enumerate(
            (
                (
                    "positive_scores_clipped_to_zero_primary",
                    np.minimum(docking_unclipped, 0.0),
                ),
                ("released_scores_unclipped_sensitivity", docking_unclipped),
            )
        ):
            docking_delta, docking_low, docking_high = delta_geometry(docking, mw)
            if not np.array_equal(low, docking_low) or not np.array_equal(high, docking_high):
                raise RuntimeError("matched experimental and docking MW supports diverged")
            direction = direction_qap(
                experiment_delta,
                docking_delta,
                qap_permutations,
                seed + 10 * offset + representation_offset,
            )
            rows.append(
                {
                    "panel": panel_name,
                    "score_preprocessing": preprocessing,
                    "matched_ligands": int(len(experiment)),
                    "low_quartile_ligands": int(low.sum()),
                    "high_quartile_ligands": int(high.sum()),
                    "targets": len(TARGETS),
                    **direction,
                }
            )
    return rows


def write_readme(output_dir: Path, summary: dict[str, Any]) -> None:
    panel_lines = []
    for panel, values in summary["panels"].items():
        observed = values["observed"]
        null = values["random_disjoint_null"]
        panel_lines.append(
            f"- **{panel}:** low--high residual-geometry Spearman "
            f"{observed['geometry_spearman']:.3f}; random-disjoint mean "
            f"{null['mean']:.3f}, lower-tail Monte Carlo p={null['lower_tail_monte_carlo_p']:.4g}; "
            f"target-jackknife range {values['target_jackknife_geometry_spearman_range'][0]:.3f}--"
            f"{values['target_jackknife_geometry_spearman_range'][1]:.3f}."
        )
    matched_lines = []
    for row in summary["matched_dockstring_direction"]:
        if row["score_preprocessing"] != "positive_scores_clipped_to_zero_primary":
            continue
        matched_lines.append(
            f"- **{row['panel']} ({row['matched_ligands']} exact matched ligands):** "
            f"experimental-versus-DOCKSTRING low-to-high delta-network Spearman "
            f"{row['direction_spearman']:.3f}, target-label QAP p={row['target_label_qap_p_positive']:.4g}."
        )
    cross_panel_lines = []
    for row in summary["experimental_cross_panel_direction"]:
        cross_panel_lines.append(
            f"- **{row['first_panel']} versus {row['second_panel']}:** delta-network "
            f"Spearman {row['direction_spearman']:.3f}, QAP p={row['target_label_qap_p_positive']:.4g}, "
            f"Holm p={row['holm_p_across_three_panel_pairs']:.4g}."
        )
    readme = f"""# Experimental chemical-context geometry

This is a science-only, post-hoc analysis on the exact 20-target support shared
by the complete PKIS2 (645 ligands), PKIS1 (360 ligands), and DAVIS (72 ligands)
panels.  It tests whether two-way-centered experimental target-correlation
geometry is stable between the lowest and highest molecular-weight quartiles.
It does **not** test ligand-level target ranking and does not establish molecular
weight as a cause.

## Main result

{chr(10).join(panel_lines)}

The random-disjoint null uses two disjoint ligand samples with exactly the
observed quartile sizes.  Murcko and Morgan-radius-2, 2048-bit Butina clustering
(Tanimoto threshold 0.65) are reported as separate cluster-bootstrap
sensitivities.  A five-bin analysis asks whether target-network dissimilarity
increases with separation in median MW.  DAVIS is explicitly low power: its
quartiles contain only 18 ligands each.

## Does the direction replicate across experimental panels?

{chr(10).join(cross_panel_lines)}

The property of chemistry-dependent geometry replicates in the two larger
panels, but a single universal direction of MW-associated rewiring does not:
only PKIS2--DAVIS aligns after the three-pair exploratory correction.  The
panels have different compounds and assay modalities, so this is a target-label
comparison rather than an exact-ligand replication.

## Exact matched-ligand DOCKSTRING check

{chr(10).join(matched_lines)}

This matched check compares **directions of network change**, not the absolute
correlation networks.  It is exploratory and is not promoted when QAP does not
support positive alignment.  PKIS1 is absent because there is no pre-frozen,
audited exact-ligand DOCKSTRING mapping in this package; creating a new mapping
after inspecting outcomes would add another post-hoc degree of freedom.

## Reproduction

```bash
python analysis/experimental_chemical_context_geometry.py \\
  --pkis1-zip /tmp/pkis1_supplement.zip \\
  --output-dir results/experimental_chemical_context_geometry
```

The PKIS1 archive must match SHA-256 `{geometry.PKIS1_SHA256}`.  Raw source
files are never modified.  `summary.json` records parameters, source hashes,
and limitations; CSV files contain the null draws, cluster-bootstrap draws,
quintile summaries, target jackknife, and matched-support comparison.
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")


def run_analysis(
    output_dir: Path,
    pkis1_zip: Path,
    random_repetitions: int,
    bootstraps: int,
    qap_permutations: int,
    seed: int,
) -> dict[str, Any]:
    if min(random_repetitions, bootstraps, qap_permutations) < 1:
        raise ValueError("all repetition counts must be positive")
    panels = load_experimental_panels(pkis1_zip)
    output_dir.mkdir(parents=True, exist_ok=True)
    panel_summaries: dict[str, Any] = {}
    all_null: list[dict] = []
    all_bootstrap: list[dict] = []
    all_quintiles: list[dict] = []
    all_pair_and_jackknife: list[dict] = []
    for offset, (panel_name, panel) in enumerate(panels.items()):
        values = panel_analysis(
            panel_name,
            np.asarray(panel["matrix"], float),
            pd.Series(panel["smiles"]),
            random_repetitions,
            bootstraps,
            seed + 10_000 * offset,
        )
        panel_summaries[panel_name] = values[0]
        all_null.extend(values[1])
        all_bootstrap.extend(values[2])
        all_quintiles.extend(values[3])
        all_pair_and_jackknife.extend(values[4])

    # Split heterogeneous records generated above into explicit tables.
    pair_rows = [row for row in all_pair_and_jackknife if "first_quintile" in row]
    jackknife_rows = [row for row in all_pair_and_jackknife if "removed_target" in row]
    matched = matched_dockstring_analysis(panels, qap_permutations, seed + 50_000)
    cross_panel = experimental_cross_panel_directions(
        panels, qap_permutations, seed + 60_000
    )

    pd.DataFrame(all_null).to_csv(output_dir / "random_disjoint_null.csv", index=False)
    pd.DataFrame(all_bootstrap).to_csv(output_dir / "chemical_group_bootstrap.csv", index=False)
    pd.DataFrame(all_quintiles).to_csv(output_dir / "mw_quintile_metrics.csv", index=False)
    pd.DataFrame(pair_rows).to_csv(output_dir / "mw_quintile_pair_geometry.csv", index=False)
    pd.DataFrame(jackknife_rows).to_csv(output_dir / "target_jackknife.csv", index=False)
    pd.DataFrame(matched).to_csv(output_dir / "matched_dockstring_direction.csv", index=False)
    pd.DataFrame(cross_panel).to_csv(
        output_dir / "experimental_cross_panel_direction.csv", index=False
    )

    summary = {
        "analysis": "experimental_chemical_context_geometry",
        "status": "exploratory_post_hoc",
        "seed": seed,
        "fixed_target_support": list(TARGETS),
        "primary_stratifier": "RDKit molecular weight",
        "extreme_fraction": EXTREME_FRACTION,
        "random_disjoint_repetitions": random_repetitions,
        "chemical_group_bootstraps": bootstraps,
        "matched_direction_qap_permutations": qap_permutations,
        "panels": panel_summaries,
        "experimental_cross_panel_direction": cross_panel,
        "matched_dockstring_direction": matched,
        "source_sha256": {
            "PKIS2": sha256_file(geometry.pkis2.DEFAULT_PKIS2),
            "PKIS1": sha256_file(pkis1_zip),
            "DAVIS": sha256_file(davis.DEFAULT_DAVIS),
            "DOCKSTRING": sha256_file(davis.DEFAULT_DOCKSTRING),
            "PKIS2_mapping": sha256_file(PACKAGE / "results" / "dense_pkis2_primary_molecule_mapping.csv"),
            "DAVIS_mapping": sha256_file(PACKAGE / "results" / "dense_davis_primary_molecule_mapping.csv"),
        },
        "limitations": [
            "Molecular weight and this entire analysis are post hoc.",
            "The panels use different assay modalities and ligand libraries; cross-panel equality is not assumed.",
            "DAVIS has only 72 ligands and 18 ligands per extreme quartile.",
            "Intervals are chemical-support sensitivities conditional on each fixed 20-target panel.",
            "Cluster bootstrap independently resamples chemical groups within each MW extreme and is not a target-population interval.",
            "Matched DOCKSTRING comparisons reuse previously inspected resources and test network-change direction, not causal biology or ligand-level retrieval.",
            "A low experimental low--high geometry correlation is evidence of context dependence only relative to the finite-sample null, not evidence that MW is causal.",
        ],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(json_ready(summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_readme(output_dir, summary)

    checksums = {
        path.name: sha256_file(path)
        for path in sorted(output_dir.iterdir())
        if path.is_file() and path.name != "output_checksums.json"
    }
    (output_dir / "output_checksums.json").write_text(
        json.dumps(checksums, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--pkis1-zip", type=Path, default=DEFAULT_PKIS1_ZIP)
    parser.add_argument("--random-repetitions", type=int, default=DEFAULT_RANDOM_REPETITIONS)
    parser.add_argument("--bootstraps", type=int, default=DEFAULT_BOOTSTRAPS)
    parser.add_argument("--qap-permutations", type=int, default=DEFAULT_QAP_PERMUTATIONS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_analysis(
        args.output_dir,
        args.pkis1_zip,
        args.random_repetitions,
        args.bootstraps,
        args.qap_permutations,
        args.seed,
    )
    print(json.dumps(json_ready(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
