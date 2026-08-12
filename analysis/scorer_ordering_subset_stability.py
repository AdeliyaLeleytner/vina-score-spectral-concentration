#!/usr/bin/env python3
"""How far does the feature-class ordering of the shared component survive target subsetting?

The fixed-pose transport analysis reports, on the full nine-target block, that the shared
component is largest for atom-pair-count scorers, smaller for empirical term sums, smaller
again for a learned contact network, and smallest for an interaction fingerprint. That
ordering is a statement about nine targets. This script asks how often it still holds when
the panel is cut down, which bounds how much the ordering can be generalised.

The ordering is treated as holding on a target subset when

    min(PC1 fraction over RF-Score v1/v2/v3)
        > max(PC1 fraction over released Vina, smina, Vinardo)
        > PC1 fraction of NNScore
        > PC1 fraction of PLECscore

evaluated on the target-correlation matrix of that subset alone. Every subset of size three
through nine is enumerated exhaustively, so there is no sampling and no seed.

Input:  results/nonvina_scorer_transport/fixed_pose_scores.csv.gz  (frozen, checksum recorded
        in that directory's output_checksums.json)
Output: results/scorer_ordering_subset_stability/{summary.json,subset_counts.csv}

This is a descriptive sensitivity on one fixed nine-target block of retained Vina poses. It
does not establish the ordering for unseen targets, unseen scoring functions, or poses that
another function would have selected in its own search.
"""

from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

PACKAGE = Path(__file__).resolve().parents[1]
SOURCE = PACKAGE / "results" / "nonvina_scorer_transport" / "fixed_pose_scores.csv.gz"
OUT = PACKAGE / "results" / "scorer_ordering_subset_stability"

CLASSES = {
    "atom_pair_counts": ("rfscore_v1", "rfscore_v2", "rfscore_v3"),
    "empirical_terms": ("released_vina", "smina_vina", "vinardo"),
    "learned_contact_net": ("nnscore",),
    "interaction_fingerprint": ("plecscore_linear",),
}
# Polarity: the released Vina family reports more negative as better, the machine-learned
# scorers report larger as better. The transport analysis fixes this by sign; the leading
# eigenvalue fraction is invariant to a global sign flip, so no polarity term is needed here.


def pc1_fraction(block: np.ndarray) -> float:
    """Fraction of target-correlation variance on the leading component."""
    correlation = np.corrcoef(block, rowvar=False)
    eigenvalues = np.linalg.eigvalsh(correlation)
    return float(eigenvalues.max() / eigenvalues.sum())


def main() -> None:
    frame = pd.read_csv(SOURCE)
    wide = frame.pivot_table(index=["scorer", "key"], columns="target", values="score")
    targets = tuple(wide.columns)
    scorers = {name: wide.xs(name, level="scorer").loc[:, list(targets)].to_numpy(float)
               for name in frame["scorer"].unique()}

    rows = []
    for size in range(3, len(targets) + 1):
        held = 0
        total = 0
        for subset in itertools.combinations(range(len(targets)), size):
            index = list(subset)
            fractions = {name: pc1_fraction(matrix[:, index]) for name, matrix in scorers.items()}
            counts = min(fractions[s] for s in CLASSES["atom_pair_counts"])
            terms = max(fractions[s] for s in CLASSES["empirical_terms"])
            net = fractions["nnscore"]
            fingerprint = fractions["plecscore_linear"]
            total += 1
            if counts > terms > net > fingerprint:
                held += 1
        rows.append({"target_subset_size": size, "subsets": total, "ordering_holds": held,
                     "fraction": held / total})
        print(f"  size {size}: {held}/{total}")

    counts = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    counts.to_csv(OUT / "subset_counts.csv", index=False)

    digest = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    summary = {
        "analysis": "feature-class ordering of the shared component under target subsetting",
        "analysis_status": "descriptive_post_hoc_sensitivity",
        "ordering_tested": ("min(atom-pair counts) > max(empirical terms) > learned contact net "
                            "> interaction fingerprint, on PC1 variance fraction"),
        "enumeration": "exhaustive over all target subsets of each size; no sampling, no seed",
        "input": {"path": str(SOURCE.relative_to(PACKAGE)), "sha256": digest,
                  "ligands": int(wide.xs(list(scorers)[0], level="scorer").shape[0]),
                  "targets": len(targets), "scorers": len(scorers)},
        "by_size": {int(r["target_subset_size"]):
                    {"subsets": int(r["subsets"]), "ordering_holds": int(r["ordering_holds"]),
                     "fraction": float(r["fraction"])} for r in rows},
        "claim_boundary": (
            "A descriptive sensitivity on one fixed nine-target block of retained AutoDock Vina "
            "poses. It bounds how far the full-panel ordering can be read as general; it does not "
            "establish the ordering for unseen targets or unseen scoring functions, and it does "
            "not simulate poses another function would select in its own search."),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("wrote", OUT / "summary.json")


if __name__ == "__main__":
    main()
