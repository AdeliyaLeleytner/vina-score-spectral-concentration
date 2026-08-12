#!/usr/bin/env python3
"""Cluster-bootstrap interval for the minimum PC1 loading of each scoring output.

The fixed-pose transport analysis evaluates four criteria, and the second of them asks
for a leading loading vector close to the uniform target direction *with every loading
positive*. Its producer records a bootstrap interval for the cosine but not for the
minimum loading, so the positivity half of that criterion has a point estimate and no
uncertainty attached. RF-Score v2's minimum loading is 0.0469, near enough to zero that
the question is not rhetorical.

This script fills the gap without rerunning the rescoring stage. It reads the frozen
per-cell scores, rebuilds the eight ligand-by-target blocks, and reuses the producer's
own metric and resampling functions so the definitions, the clustering recipe and the
seeds match the released table exactly. It needs neither the DOCKSTRING pose archives,
nor smina, nor the Python 3.9 ODDT environment.

Input
    results/nonvina_scorer_transport/fixed_pose_scores.csv.gz  (frozen)
    data/frozen/dockstring-dataset.tsv.gz                      (frozen, for SMILES)
Output
    results/nonvina_scorer_min_loading_interval/summary.json

The interval is the conservative union of a Bemis-Murcko-scaffold and a Butina-cluster
bootstrap, which is the rule the released table already uses for its other columns.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE / "analysis"))

import dockstring_vina_term_decomposition as terms  # noqa: E402
import nonvina_scorer_transport as transport  # noqa: E402

SCORES = PACKAGE / "results" / "nonvina_scorer_transport" / "fixed_pose_scores.csv.gz"
DATASET = PACKAGE / "data" / "frozen" / "dockstring-dataset.tsv.gz"
OUT = PACKAGE / "results" / "nonvina_scorer_min_loading_interval"
METRIC = "pc1_min_loading"


def main() -> None:
    scores = pd.read_csv(SCORES)
    wide = scores.pivot_table(index=["scorer", "key"], columns="target", values="score")
    targets = tuple(wide.columns)
    scorers = tuple(scores["scorer"].unique())

    # One shared ligand order for every scorer, so a resampled cluster selects the same
    # ligands in all eight blocks and the contrast stays paired.
    keys = tuple(wide.xs(scorers[0], level="scorer").index)
    matrices = {name: wide.xs(name, level="scorer").loc[list(keys), list(targets)].to_numpy(float)
                for name in scorers}

    dataset = pd.read_csv(DATASET, sep="\t", usecols=["inchikey", "smiles"])
    smiles_by_key = dict(zip(dataset["inchikey"], dataset["smiles"]))
    missing = [k for k in keys if k not in smiles_by_key]
    if missing:
        raise SystemExit(f"{len(missing)} ligand keys have no SMILES in the frozen dataset")
    # molecule_descriptors sorts by selection_rank; number the rows in the same order as
    # the score matrices so the returned group labels stay row-aligned.
    block = pd.DataFrame({"key": list(keys), "smiles": [smiles_by_key[k] for k in keys],
                          "selection_rank": range(len(keys))})

    # cluster_bootstrap reads this module global at call time, so extending it adds the
    # minimum loading without touching the producer. The other metrics come along, which
    # gives a free check that this run reproduces the published intervals.
    transport.BOOTSTRAP_METRICS = tuple(transport.BOOTSTRAP_METRICS) + (METRIC,)

    _, murcko_groups = terms.molecule_descriptors(block)
    butina_labels = transport.butina_clusters(block.smiles.astype(str).tolist())

    murcko = transport.cluster_bootstrap(
        matrices, murcko_groups, transport.BOOTSTRAP_REPLICATES, transport.BOOTSTRAP_SEED)
    butina = transport.cluster_bootstrap(
        matrices, butina_labels, transport.BOOTSTRAP_REPLICATES, transport.BOOTSTRAP_SEED + 1)

    observed = {name: transport.transport_metrics(matrix)[METRIC]
                for name, matrix in matrices.items()}

    rows = []
    for name in scorers:
        a = murcko["metrics"][name][METRIC]["interval95"]
        b = butina["metrics"][name][METRIC]["interval95"]
        low, high = min(a[0], b[0]), max(a[1], b[1])
        rows.append({
            "scorer": name,
            "pc1_min_loading": observed[name],
            "murcko_interval95_low": a[0], "murcko_interval95_high": a[1],
            "butina_interval95_low": b[0], "butina_interval95_high": b[1],
            "union_interval95_low": low, "union_interval95_high": high,
            "interval_excludes_zero": bool(low > 0.0),
        })
        print(f"  {name:20s} {observed[name]:7.4f}  union [{low:.4f}, {high:.4f}]"
              f"  {'positive' if low > 0 else 'CROSSES ZERO'}")

    # Cross-check: the recomputed cosine intervals must match the released table.
    published = pd.read_csv(
        PACKAGE / "results" / "nonvina_scorer_transport" / "scorer_spectral_metrics.csv"
    ).set_index("scorer")
    drift = []
    for name in scorers:
        a = murcko["metrics"][name]["pc1_uniform_cosine"]["interval95"]
        b = butina["metrics"][name]["pc1_uniform_cosine"]["interval95"]
        low, high = min(a[0], b[0]), max(a[1], b[1])
        drift.append(max(abs(low - published.loc[name, "pc1_uniform_cosine_ci_low"]),
                         abs(high - published.loc[name, "pc1_uniform_cosine_ci_high"])))
    reproduction_error = float(max(drift))
    print(f"  cosine-interval reproduction error: {reproduction_error:.2e}")

    frame = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT / "min_loading_intervals.csv", index=False)

    summary = {
        "analysis": "cluster-bootstrap interval for the minimum PC1 loading",
        "analysis_status": "descriptive_post_hoc_sensitivity",
        "why": ("The released transport table records pc1_min_loading as a point value only, "
                "so the all-loadings-positive half of the second criterion carried no "
                "uncertainty. This recomputes it from the frozen scores."),
        "interval_rule": ("conservative union of the Bemis-Murcko-scaffold and Butina-cluster "
                          "95% bootstrap intervals, matching the released table's other columns"),
        "replicates_per_scheme": int(transport.BOOTSTRAP_REPLICATES),
        "seeds": {"murcko": int(transport.BOOTSTRAP_SEED),
                  "butina": int(transport.BOOTSTRAP_SEED + 1)},
        "support": {"ligands": len(keys), "targets": len(targets), "scorers": len(scorers),
                    "murcko_groups": int(len(set(murcko_groups))),
                    "butina_clusters": int(len(set(butina_labels)))},
        "inputs": {
            "fixed_pose_scores": {"path": str(SCORES.relative_to(PACKAGE)),
                                  "sha256": hashlib.sha256(SCORES.read_bytes()).hexdigest()},
            "dockstring_dataset": {"path": str(DATASET.relative_to(PACKAGE)),
                                   "sha256": hashlib.sha256(DATASET.read_bytes()).hexdigest()},
        },
        "by_scorer": {r["scorer"]: {k: v for k, v in r.items() if k != "scorer"} for r in rows},
        "scorers_whose_interval_crosses_zero": [r["scorer"] for r in rows
                                                if not r["interval_excludes_zero"]],
        "released_cosine_interval_reproduction_max_abs_error": reproduction_error,
        "claim_boundary": ("A ligand-side resampling on one fixed nine-target block of retained "
                           "AutoDock Vina poses. It quantifies uncertainty in the minimum loading "
                           "under chemical resampling; it says nothing about unseen targets or "
                           "poses another scoring function would select in its own search."),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("wrote", OUT / "summary.json")


if __name__ == "__main__":
    main()
