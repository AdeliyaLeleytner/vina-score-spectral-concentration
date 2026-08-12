#!/usr/bin/env python3
"""Protein-sequence controls for raw and row-centred DOCKSTRING target maps.

The analysis asks whether the protein-related signal seen in the 21-kinase
experimental overlap generalizes to the complete heterogeneous 58-target
DOCKSTRING panel. It deliberately reports both contexts: a result that holds
within the kinase subset need not hold as a raw-versus-residual contrast across
unrelated protein classes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ANALYSIS_DIR = Path(__file__).resolve().parent
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

import experimental_map_reliability as reliability  # noqa: E402
from fetch_dockstring_full_receptor_sequences import TARGETS  # noqa: E402


PACKAGE = ANALYSIS_DIR.parent
MATRIX = PACKAGE / "data" / "frozen" / "dockstring-dataset.tsv.gz"
FASTA = PACKAGE / "data" / "frozen" / "dockstring_58_receptors.fasta"
MANIFEST = PACKAGE / "data" / "frozen" / "dockstring_58_receptors_manifest.csv"
FAMILIES = PACKAGE / "data" / "target_families.csv"
DEFAULT_OUTPUT = PACKAGE / "results" / "public_sequence_geometry_controls"
KINASE21 = tuple(reliability.TARGETS)
QAP_PERMUTATIONS = 100_000
SEED = 202_609_17
OUTPUT_FILES = (
    "README.md",
    "edge_geometry.csv.gz",
    "family_edge_summary.csv",
    "qap_summary.csv",
    "summary.json",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def atomic_csv(
    path: Path,
    frame: pd.DataFrame,
    *,
    compression: str | dict[str, Any] | None = None,
) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(
        temporary,
        index=False,
        float_format="%.12g",
        compression=compression,
    )
    temporary.replace(path)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def load_sequences() -> tuple[dict[str, str], dict[str, Any]]:
    sequences: dict[str, str] = {}
    current: str | None = None
    for line in FASTA.read_text(encoding="utf-8").splitlines():
        if line.startswith(">"):
            current = line[1:].split()[0]
            if current in sequences:
                raise ValueError(f"duplicate FASTA target: {current}")
            sequences[current] = ""
        elif current is not None:
            sequences[current] += line.strip()
    if tuple(sequences) != TARGETS:
        raise ValueError("full receptor FASTA order does not match the matrix")
    manifest = pd.read_csv(MANIFEST)
    if tuple(manifest.target) != TARGETS:
        raise ValueError("full receptor manifest order does not match the matrix")
    for row in manifest.itertuples(index=False):
        sequence = sequences[row.target]
        if len(sequence) != int(row.sequence_length):
            raise ValueError(f"sequence length mismatch for {row.target}")
        observed = hashlib.sha256(sequence.encode("ascii")).hexdigest()
        if observed != row.sequence_sha256:
            raise ValueError(f"sequence checksum mismatch for {row.target}")
    return sequences, {
        "fasta_sha256": sha256_file(FASTA),
        "manifest_sha256": sha256_file(MANIFEST),
        "targets": len(sequences),
        "minimum_sequence_length": min(map(len, sequences.values())),
        "maximum_sequence_length": max(map(len, sequences.values())),
    }


def symmetric_sequence_identity(
    sequences: dict[str, str], target_order: tuple[str, ...]
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return order-invariant global BLOSUM62 identities including gaps."""
    try:
        import Bio
        from Bio import Align
        from Bio.Align import substitution_matrices
    except ImportError as error:  # pragma: no cover - dependency gate
        raise RuntimeError("Biopython is required for sequence controls") from error
    aligner = Align.PairwiseAligner()
    aligner.mode = "global"
    aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
    aligner.open_gap_score = -10.0
    aligner.extend_gap_score = -0.5

    def oriented_identity(first: str, second: str) -> float:
        alignment = aligner.align(first, second)[0]
        return float(alignment.counts().identities / alignment.length)

    identity = np.eye(len(target_order), dtype=np.float64)
    maximum_orientation_difference = 0.0
    for first in range(len(target_order)):
        for second in range(first):
            forward = oriented_identity(
                sequences[target_order[first]], sequences[target_order[second]]
            )
            reverse = oriented_identity(
                sequences[target_order[second]], sequences[target_order[first]]
            )
            maximum_orientation_difference = max(
                maximum_orientation_difference, abs(forward - reverse)
            )
            identity[first, second] = identity[second, first] = (
                forward + reverse
            ) / 2.0
    if not np.allclose(identity, identity.T, atol=0.0, rtol=0.0):
        raise RuntimeError("sequence identity matrix is not exactly symmetric")
    return identity, {
        "alignment": "global BLOSUM62; gap-open -10; gap-extend -0.5",
        "identity_denominator": "all alignment columns, including gaps",
        "tie_rule": (
            "mean identity of the first optimal alignment in both sequence "
            "orientations, making the result invariant to arbitrary target order"
        ),
        "maximum_pre_symmetrization_orientation_difference": float(
            maximum_orientation_difference
        ),
        "identity_matrix_sha256": hashlib.sha256(identity.tobytes()).hexdigest(),
        "biopython_version": Bio.__version__,
    }


def load_score_maps() -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    header = pd.read_csv(MATRIX, sep="\t", nrows=0).columns.tolist()
    if tuple(header[2:]) != TARGETS:
        raise ValueError("DOCKSTRING target columns changed")
    frame = pd.read_csv(MATRIX, sep="\t", usecols=list(TARGETS))
    complete = frame.dropna().to_numpy(dtype=np.float64)
    if complete.shape != (260_060, 58):
        raise ValueError("unexpected complete DOCKSTRING score support")
    positive = int(np.sum(complete > 0.0))
    clipped = np.minimum(complete, 0.0)
    raw = reliability.target_correlation(clipped)
    residual = reliability.target_correlation(
        clipped - clipped.mean(axis=1, keepdims=True)
    )
    return raw, residual, clipped, {
        "complete_ligands": len(clipped),
        "targets": clipped.shape[1],
        "positive_scores_clipped_to_zero": positive,
        "matrix_sha256": sha256_file(MATRIX),
    }


def upper(matrix: np.ndarray) -> np.ndarray:
    return np.asarray(matrix, dtype=np.float64)[np.triu_indices(len(matrix), 1)]


def pearson_edges(first: np.ndarray, second: np.ndarray) -> float:
    left = upper(first)
    right = upper(second)
    left = left - left.mean()
    right = right - right.mean()
    denominator = np.sqrt(float(left @ left) * float(right @ right))
    if denominator <= 1e-15:
        raise ValueError("edge correlation is degenerate")
    return float(left @ right / denominator)


def qap_context(
    raw: np.ndarray,
    residual: np.ndarray,
    sequence: np.ndarray,
    indices: np.ndarray,
    *,
    context: str,
    permutations: int,
    seed: int,
) -> dict[str, Any]:
    raw = raw[np.ix_(indices, indices)]
    residual = residual[np.ix_(indices, indices)]
    sequence = sequence[np.ix_(indices, indices)]
    observed_raw = pearson_edges(raw, sequence)
    observed_residual = pearson_edges(residual, sequence)
    observed_difference = observed_residual - observed_raw
    null_raw = np.empty(permutations, dtype=np.float64)
    null_residual = np.empty(permutations, dtype=np.float64)
    rng = np.random.default_rng(seed)
    for repetition in range(permutations):
        permutation = rng.permutation(len(indices))
        permuted = sequence[np.ix_(permutation, permutation)]
        null_raw[repetition] = pearson_edges(raw, permuted)
        null_residual[repetition] = pearson_edges(residual, permuted)
    null_difference = null_residual - null_raw
    return {
        "context": context,
        "targets": int(len(indices)),
        "edges": int(len(indices) * (len(indices) - 1) // 2),
        "raw_sequence_pearson": observed_raw,
        "residual_sequence_pearson": observed_residual,
        "residual_minus_raw_sequence_pearson": observed_difference,
        "raw_qap_p_one_sided": float(
            (1 + np.sum(null_raw >= observed_raw)) / (permutations + 1)
        ),
        "residual_qap_p_one_sided": float(
            (1 + np.sum(null_residual >= observed_residual)) / (permutations + 1)
        ),
        "residual_minus_raw_qap_p_one_sided": float(
            (1 + np.sum(null_difference >= observed_difference))
            / (permutations + 1)
        ),
        "raw_null_q025": float(np.quantile(null_raw, 0.025)),
        "raw_null_q975": float(np.quantile(null_raw, 0.975)),
        "residual_null_q025": float(np.quantile(null_residual, 0.025)),
        "residual_null_q975": float(np.quantile(null_residual, 0.975)),
        "difference_null_q025": float(np.quantile(null_difference, 0.025)),
        "difference_null_q975": float(np.quantile(null_difference, 0.975)),
        "permutations": permutations,
        "seed": seed,
    }


def edge_frame(
    raw: np.ndarray, residual: np.ndarray, sequence: np.ndarray
) -> tuple[pd.DataFrame, pd.DataFrame]:
    families = pd.read_csv(FAMILIES)
    families = families.loc[families.dataset.eq("DOCKSTRING-58")]
    family = dict(zip(families.target, families.family))
    if set(family) != set(TARGETS):
        raise ValueError("DOCKSTRING family table does not match the target panel")
    records: list[dict[str, Any]] = []
    for first in range(len(TARGETS)):
        for second in range(first):
            first_family = family[TARGETS[first]]
            second_family = family[TARGETS[second]]
            records.append(
                {
                    "target_a": TARGETS[second],
                    "target_b": TARGETS[first],
                    "family_a": first_family,
                    "family_b": second_family,
                    "same_family": first_family == second_family,
                    "sequence_identity": sequence[first, second],
                    "raw_score_correlation": raw[first, second],
                    "residual_score_correlation": residual[first, second],
                }
            )
    edges = pd.DataFrame.from_records(records)
    summaries: list[dict[str, Any]] = []
    groups: list[tuple[str, pd.DataFrame]] = [
        ("all_58", edges),
        ("all_same_family_edges", edges.loc[edges.same_family]),
        ("all_between_family_edges", edges.loc[~edges.same_family]),
    ]
    for name, members in edges.loc[edges.same_family].groupby("family_a", sort=True):
        if len(members) >= 3:
            groups.append((f"within_{name}", members))
    for name, members in groups:
        summaries.append(
            {
                "edge_subset": name,
                "edges": int(len(members)),
                "raw_sequence_pearson": float(
                    np.corrcoef(members.raw_score_correlation, members.sequence_identity)[
                        0, 1
                    ]
                ),
                "residual_sequence_pearson": float(
                    np.corrcoef(
                        members.residual_score_correlation, members.sequence_identity
                    )[0, 1]
                ),
                "raw_sequence_spearman": float(
                    spearmanr(
                        members.raw_score_correlation, members.sequence_identity
                    ).statistic
                ),
                "residual_sequence_spearman": float(
                    spearmanr(
                        members.residual_score_correlation, members.sequence_identity
                    ).statistic
                ),
            }
        )
    return edges, pd.DataFrame.from_records(summaries)


def analyse(permutations: int, seed: int) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
    if permutations < 1_000:
        raise ValueError("production QAP requires at least 1,000 permutations")
    sequences, sequence_source = load_sequences()
    identity, alignment = symmetric_sequence_identity(sequences, TARGETS)
    raw, residual, clipped, matrix = load_score_maps()
    kinase_indices = np.asarray([TARGETS.index(target) for target in KINASE21])
    kinase_raw = reliability.target_correlation(clipped[:, kinase_indices])
    embedded_kinase_raw = raw[np.ix_(kinase_indices, kinase_indices)]
    if not np.allclose(kinase_raw, embedded_kinase_raw, atol=2e-15, rtol=0.0):
        raise RuntimeError("raw kinase correlations changed under target subsetting")
    kinase_scores = clipped[:, kinase_indices]
    kinase_residual = reliability.target_correlation(
        kinase_scores - kinase_scores.mean(axis=1, keepdims=True)
    )
    qap = [
        qap_context(
            raw,
            residual,
            identity,
            np.arange(len(TARGETS)),
            context="all_58_targets_centered_within_58",
            permutations=permutations,
            seed=seed,
        ),
        qap_context(
            raw,
            residual,
            identity,
            kinase_indices,
            context="common_21_kinases_embedded_in_58_target_centering",
            permutations=permutations,
            seed=seed + 1_000_000,
        ),
        qap_context(
            kinase_raw,
            kinase_residual,
            identity[np.ix_(kinase_indices, kinase_indices)],
            np.arange(len(kinase_indices)),
            context="common_21_kinases_centered_within_21",
            permutations=permutations,
            seed=seed + 2_000_000,
        ),
    ]
    edges, family_summary = edge_frame(raw, residual, identity)
    summary = {
        "analysis_status": "strict_public_descriptive_sequence_geometry_control",
        "inputs": {
            "matrix": matrix,
            "sequence_source": sequence_source,
            "alignment": alignment,
        },
        "qap": qap,
        "interpretation": (
            "Across all 58 heterogeneous targets both raw and residual score maps "
            "show modest protein-sequence alignment, and their difference is tested "
            "rather than assumed. The 21-kinase subset is reported twice because a "
            "row-centred residual is defined relative to its target panel: once after "
            "centring within all 58 targets and once after restricting and centring "
            "within the 21 kinases. These are map associations, not pose validation "
            "or evidence that docking outperforms sequence-based panel design."
        ),
        "claim_boundary": (
            "Target-label QAP preserves each map but does not sample proteins, "
            "structures or scoring engines. Sequence identity is a cheap baseline; "
            "the analysis cannot assign causality or establish biological function."
        ),
        "producer": "analysis/public_sequence_geometry_controls.py",
    }
    return summary, {
        "edge_geometry.csv.gz": edges,
        "family_edge_summary.csv": family_summary,
        "qap_summary.csv": pd.DataFrame.from_records(qap),
    }


def write_bundle(
    output: Path, summary: dict[str, Any], frames: dict[str, pd.DataFrame]
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for name, frame in frames.items():
        atomic_csv(
            output / name,
            frame,
            compression={"method": "gzip", "mtime": 0}
            if name.endswith(".gz")
            else None,
        )
    readme = """# Public sequence-geometry controls

This artifact compares raw and raw-row-centred DOCKSTRING target-correlation
maps with global sequence identity for all 58 receptors and for the common
21-kinase subset. Because row centring depends on the target panel, the kinase
analysis is reported both after centring within all 58 targets and after
restriction followed by centring within the 21 kinases. Shared target-label QAP
tests raw, residual and their paired difference. Family-specific edge summaries
are descriptive only.

Reproduce with:

```bash
.venv/bin/python analysis/public_sequence_geometry_controls.py --overwrite
.venv/bin/pytest -q analysis/test_public_sequence_geometry_controls.py
```
"""
    (output / "README.md").write_text(readme, encoding="utf-8")
    atomic_json(output / "summary.json", summary)
    checksums = {
        name: sha256_file(output / name)
        for name in OUTPUT_FILES
        if (output / name).exists()
    }
    atomic_json(output / "output_checksums.json", checksums)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--permutations", type=int, default=QAP_PERMUTATIONS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists() and any(args.output.iterdir()) and not args.overwrite:
        raise FileExistsError(f"output exists: {args.output}; pass --overwrite")
    summary, frames = analyse(args.permutations, args.seed)
    write_bundle(args.output, summary, frames)


if __name__ == "__main__":
    main()
