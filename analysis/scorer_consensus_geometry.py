#!/usr/bin/env python3
"""Exploratory Vina--Boltz consensus target-geometry analysis on DAVIS.

All scorer combinations are outcome-blind and use equal weights.  The analysis
tests whether target-pair structure shared across a classical docking scorer and
the two deployed Boltz-2 affinity heads is more concordant with the DAVIS
experimental target geometry than either scorer alone.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

from dense_davis_benchmark import TARGET_MAP


PACKAGE = Path(__file__).resolve().parents[1]
DEFAULT_DOCKSTRING = PACKAGE / "data/frozen/dockstring-dataset.tsv.gz"
DEFAULT_DAVIS = (
    PACKAGE
    / "data/frozen/project_clean/data/processed/davis_boltz2/davis_complete.tab.gz"
)
DEFAULT_MAPPING = PACKAGE / "data/frozen/davis_boltz_cid_mapping.csv"
DEFAULT_DENSE_MAPPING = PACKAGE / "results/dense_davis_primary_molecule_mapping.csv"
DEFAULT_BOLTZ = PACKAGE.parent / "project_clean/data/processed/davis_boltz2/affinity_out.npz"


def two_way_center(matrix: np.ndarray) -> np.ndarray:
    return (
        matrix
        - matrix.mean(axis=0, keepdims=True)
        - matrix.mean(axis=1, keepdims=True)
        + matrix.mean()
    )


def target_geometry(matrix: np.ndarray) -> np.ndarray:
    return np.corrcoef(two_way_center(np.asarray(matrix, dtype=float)), rowvar=False)


def correlation_pr(geometry: np.ndarray) -> float:
    eigenvalues = np.linalg.eigvalsh(geometry)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    return float(eigenvalues.sum() ** 2 / np.sum(eigenvalues**2))


def upper(matrix: np.ndarray) -> np.ndarray:
    return matrix[np.triu_indices(len(matrix), k=1)]


def percentile(values: np.ndarray) -> np.ndarray:
    return (rankdata(values, method="average") - 0.5) / len(values)


def rank_consensus(*geometries: np.ndarray) -> np.ndarray:
    p = len(geometries[0])
    tri = np.triu_indices(p, k=1)
    score = np.mean([percentile(matrix[tri]) for matrix in geometries], axis=0)
    output = np.eye(p, dtype=float)
    output[tri] = score
    output[(tri[1], tri[0])] = score
    return output


def top_fraction_labels(values: np.ndarray, fraction: float = 0.10) -> np.ndarray:
    count = int(np.ceil(len(values) * fraction))
    labels = np.zeros(len(values), dtype=bool)
    labels[np.argsort(values, kind="mergesort")[-count:]] = True
    return labels


def retrieval(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    ranks = rankdata(scores, method="average")
    positives = int(labels.sum())
    negatives = int(len(labels) - positives)
    auc = (
        float(ranks[labels].sum()) - positives * (positives + 1) / 2
    ) / (positives * negatives)
    order = np.argsort(scores, kind="mergesort")[::-1]
    ordered = labels[order].astype(int)
    precision = np.cumsum(ordered) / (np.arange(len(ordered)) + 1)
    ap = float(np.sum(ordered * precision) / positives)
    return float(auc), ap


def load_aligned(
    dockstring_path: Path,
    davis_path: Path,
    cid_mapping_path: Path,
    dense_mapping_path: Path,
    boltz_path: Path,
) -> tuple[dict[str, np.ndarray], list[str], list[str]]:
    dense_mapping = pd.read_csv(dense_mapping_path)
    cid_mapping = pd.read_csv(cid_mapping_path)
    mapping = dense_mapping.merge(
        cid_mapping[["drug_name", "pubchem_cid", "match_method"]],
        on="drug_name",
        how="left",
        validate="one_to_one",
    )
    if mapping.pubchem_cid.isna().any():
        raise ValueError("Boltz CID mapping is incomplete on the dense support")
    mapping["pubchem_cid"] = mapping.pubchem_cid.astype(str)
    ligand_order = mapping.drug_name.astype(str).tolist()
    targets = list(TARGET_MAP)
    protein_order = list(TARGET_MAP.values())

    docking = pd.read_csv(dockstring_path, sep="\t", usecols=targets)
    vina_rows: list[np.ndarray] = []
    for encoded_indices in mapping.dockstring_row_indices.astype(str):
        indices = [int(value) for value in encoded_indices.split(";")]
        vina_rows.append(
            np.median(docking.iloc[indices].to_numpy(dtype=float), axis=0)
        )
    vina = np.minimum(np.vstack(vina_rows), 0.0)

    davis = pd.read_csv(
        davis_path, sep="\t", usecols=["drug_name", "protein", "y"]
    )
    experiment = (
        davis[davis.drug_name.isin(ligand_order) & davis.protein.isin(protein_order)]
        .pivot(index="drug_name", columns="protein", values="y")
        .reindex(index=ligand_order, columns=protein_order)
    )
    if experiment.isna().any().any():
        raise ValueError("DAVIS matrix is incomplete after alignment")

    with np.load(boltz_path, allow_pickle=False) as archive:
        base = pd.DataFrame(
            {
                "pubchem_cid": archive["cids"].astype(str),
                "protein": archive["proteins"].astype(str),
                "aff": archive["aff"].astype(float),
                "aff1": archive["aff1"].astype(float),
                "aff2": archive["aff2"].astype(float),
            }
        )
    boltz: dict[str, np.ndarray] = {}
    for field in ("aff", "aff1", "aff2"):
        pivot = (
            base[
                base.pubchem_cid.isin(mapping.pubchem_cid)
                & base.protein.isin(protein_order)
            ]
            .pivot(index="pubchem_cid", columns="protein", values=field)
            .reindex(index=mapping.pubchem_cid, columns=protein_order)
        )
        if pivot.isna().any().any():
            raise ValueError(f"Boltz field {field} is incomplete after alignment")
        boltz[field] = pivot.to_numpy(dtype=float)

    matrices = {
        "Vina": vina,
        "Boltz_mean_head": boltz["aff"],
        "Boltz_head_1": boltz["aff1"],
        "Boltz_head_2": boltz["aff2"],
        "DAVIS_experiment": experiment.to_numpy(dtype=float),
    }
    return matrices, targets, mapping.match_method.astype(str).tolist()


def paired_qap(
    labels: np.ndarray,
    experiment_geometry: np.ndarray,
    reference: np.ndarray,
    candidate: np.ndarray,
    permutations: int,
    seed: int,
) -> dict[str, float]:
    tri = np.triu_indices(len(reference), k=1)
    reference_scores = reference[tri]
    candidate_scores = candidate[tri]
    observed_rho = float(
        spearmanr(candidate_scores, experiment_geometry[tri]).statistic
        - spearmanr(reference_scores, experiment_geometry[tri]).statistic
    )
    observed_auc = retrieval(labels, candidate_scores)[0] - retrieval(
        labels, reference_scores
    )[0]
    null_rho = np.empty(permutations, dtype=float)
    null_auc = np.empty(permutations, dtype=float)
    rng = np.random.default_rng(seed)
    for repetition in range(permutations):
        order = rng.permutation(len(reference))
        ref_scores = reference[np.ix_(order, order)][tri]
        cand_scores = candidate[np.ix_(order, order)][tri]
        null_rho[repetition] = float(
            spearmanr(cand_scores, experiment_geometry[tri]).statistic
            - spearmanr(ref_scores, experiment_geometry[tri]).statistic
        )
        null_auc[repetition] = retrieval(labels, cand_scores)[0] - retrieval(
            labels, ref_scores
        )[0]
    return {
        "delta_continuous_spearman": observed_rho,
        "paired_qap_p_positive_continuous": float(
            (1 + np.sum(null_rho >= observed_rho)) / (permutations + 1)
        ),
        "delta_top10_auc": observed_auc,
        "paired_qap_p_positive_top10_auc": float(
            (1 + np.sum(null_auc >= observed_auc)) / (permutations + 1)
        ),
    }


def run(
    dockstring: Path,
    davis: Path,
    cid_mapping: Path,
    dense_mapping: Path,
    boltz: Path,
    permutations: int,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    matrices, targets, match_methods = load_aligned(
        dockstring, davis, cid_mapping, dense_mapping, boltz
    )
    geometries = {name: target_geometry(matrix) for name, matrix in matrices.items()}
    experiment = geometries.pop("DAVIS_experiment")
    geometries["Boltz_head_rank_consensus"] = rank_consensus(
        geometries["Boltz_head_1"], geometries["Boltz_head_2"]
    )
    geometries["Vina_Boltz_equal_rank_consensus"] = rank_consensus(
        geometries["Vina"], geometries["Boltz_mean_head"]
    )
    geometries["Vina_two_Boltz_heads_equal_rank_consensus"] = rank_consensus(
        geometries["Vina"], geometries["Boltz_head_1"], geometries["Boltz_head_2"]
    )
    labels = top_fraction_labels(upper(experiment), 0.10)
    rows: list[dict[str, object]] = []
    for name, geometry in geometries.items():
        auc, ap = retrieval(labels, upper(geometry))
        rows.append(
            {
                "representation": name,
                "residual_correlation_pr": (
                    correlation_pr(geometry)
                    if "consensus" not in name
                    else float("nan")
                ),
                "continuous_experimental_geometry_spearman": float(
                    spearmanr(upper(geometry), upper(experiment)).statistic
                ),
                "experimental_top10_pair_roc_auc": auc,
                "experimental_top10_pair_average_precision": ap,
            }
        )
    comparisons = {
        "Boltz_head_consensus_minus_Boltz_mean_head": paired_qap(
            labels,
            experiment,
            geometries["Boltz_mean_head"],
            geometries["Boltz_head_rank_consensus"],
            permutations,
            seed,
        ),
        "Vina_Boltz_consensus_minus_Boltz_mean_head": paired_qap(
            labels,
            experiment,
            geometries["Boltz_mean_head"],
            geometries["Vina_Boltz_equal_rank_consensus"],
            permutations,
            seed + 1,
        ),
    }
    report: dict[str, object] = {
        "analysis_status": "post_hoc_exploratory_science_only",
        "ligands": int(next(iter(matrices.values())).shape[0]),
        "targets": targets,
        "target_count": len(targets),
        "positive_target_pairs": int(labels.sum()),
        "cid_match_method_counts_on_dense_support": pd.Series(match_methods)
        .value_counts()
        .to_dict(),
        "comparisons": comparisons,
        "claim_boundary": (
            "DAVIS is the evaluation surface and may overlap Boltz-2 training; "
            "this is not independent external validation. Geometry metrics do not "
            "establish ligand-level target retrieval."
        ),
    }
    return pd.DataFrame(rows), report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dockstring", type=Path, default=DEFAULT_DOCKSTRING)
    parser.add_argument("--davis", type=Path, default=DEFAULT_DAVIS)
    parser.add_argument("--cid-mapping", type=Path, default=DEFAULT_MAPPING)
    parser.add_argument("--dense-mapping", type=Path, default=DEFAULT_DENSE_MAPPING)
    parser.add_argument("--boltz", type=Path, default=DEFAULT_BOLTZ)
    parser.add_argument("--permutations", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=202_608_03)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PACKAGE / "results/scorer_consensus_geometry",
    )
    args = parser.parse_args()
    frame, report = run(
        args.dockstring,
        args.davis,
        args.cid_mapping,
        args.dense_mapping,
        args.boltz,
        args.permutations,
        args.seed,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_dir / "metrics.csv", index=False)
    (args.output_dir / "summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(frame.to_string(index=False))
    print(json.dumps(report["comparisons"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
