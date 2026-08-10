#!/usr/bin/env python3
"""Put the three released residual nulls on one deterministic ligand support.

The additive-Gaussian and empirical-column-permutation blocks in the frozen legacy
ledger already used seeds 60/61 and therefore the same sampled rows.  The historical
row-norm block used seeds 62/63, which made its displayed observed PR differ slightly.
This bounded refresh fingerprints the common support, verifies the two existing blocks
against it, recomputes only the row-norm null on that support, and atomically updates
``results/evidence_summary.json``.  The expensive unchanged null draws are not rerun.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from build_evidence import (
    DOCK44,
    fixed_ligand_support_indices,
    rank_summary,
    row_norm_preserving_residual_null,
    source_path,
    two_way_center,
)


PACKAGE = Path(__file__).resolve().parents[1]
EVIDENCE = PACKAGE / "results" / "evidence_summary.json"
SAMPLE_SIZE = 12_000
REPEATS = 500


def load_matrices() -> dict[str, np.ndarray]:
    docking_frame = pd.read_csv(source_path("df_final_v4.csv"), usecols=DOCK44)
    docking = docking_frame[DOCK44].apply(pd.to_numeric, errors="coerce").clip(upper=0)
    docking = docking.fillna(docking.mean()).to_numpy(dtype=np.float64)

    dockstring_frame = pd.read_csv(source_path("dockstring-dataset.tsv"), sep="\t")
    columns = [
        column
        for column in dockstring_frame.columns
        if column not in {"inchikey", "smiles"}
    ]
    numeric = dockstring_frame[columns].apply(pd.to_numeric, errors="coerce")
    numeric = numeric.loc[~numeric.isna().any(axis=1)]
    dockstring = np.minimum(numeric.to_numpy(dtype=np.float64), 0.0)
    return {"docking44": docking, "dockstring58": dockstring}


def observed_residual_on_support(
    matrix: np.ndarray, seed: int
) -> tuple[float, dict]:
    support, _, metadata = fixed_ligand_support_indices(
        len(matrix), SAMPLE_SIZE, seed
    )
    observed = rank_summary(two_way_center(matrix[support]))["participation_ratio"]
    return float(observed), metadata


def refresh() -> dict:
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    matrices = load_matrices()
    for dataset, seed in (("docking44", 60), ("dockstring58", 61)):
        matrix = matrices[dataset]
        observed, metadata = observed_residual_on_support(matrix, seed)

        additive = evidence["additive_main_effect_null"][dataset]
        empirical = evidence["empirical_residual_permutation_null"][dataset]
        if not np.isclose(additive["observed"]["residual"], observed, atol=1e-12):
            raise ValueError(f"{dataset}: additive-null support cannot be reconciled")
        if not np.isclose(empirical["observed_residual"], observed, atol=1e-12):
            raise ValueError(f"{dataset}: empirical-null support cannot be reconciled")
        if additive["sample_n"] != SAMPLE_SIZE or empirical["sample_n"] != SAMPLE_SIZE:
            raise ValueError(f"{dataset}: released residual-null sample size drifted")
        additive["support_selection"] = metadata
        empirical["support_selection"] = metadata

        refreshed_row_norm = row_norm_preserving_residual_null(
            matrix,
            sample_size=SAMPLE_SIZE,
            repeats=REPEATS,
            seed=seed,
        )
        if not np.isclose(
            refreshed_row_norm["observed_residual"], observed, atol=1e-12
        ):
            raise ValueError(f"{dataset}: refreshed row-norm support mismatch")
        evidence["row_norm_preserving_residual_null"][dataset] = refreshed_row_norm

    temporary = EVIDENCE.with_suffix(EVIDENCE.suffix + ".tmp")
    temporary.write_text(
        json.dumps(evidence, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    os.replace(temporary, EVIDENCE)
    return evidence


def main() -> None:
    evidence = refresh()
    for dataset in ("docking44", "dockstring58"):
        row_norm = evidence["row_norm_preserving_residual_null"][dataset]
        print(
            f"{dataset}: observed={row_norm['observed_residual']:.6f}; "
            f"row-norm null median={row_norm['null_residual']['median']:.6f}; "
            f"support={row_norm['support_selection']['support_index_sha256']}"
        )
    print(f"Updated {EVIDENCE.relative_to(PACKAGE)}")


if __name__ == "__main__":
    main()
