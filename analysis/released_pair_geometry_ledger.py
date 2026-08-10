#!/usr/bin/env python3
"""Emit the complete 190-edge target-pair ledger for the fixed common-20 kinase panel.

Every fixed-panel number in the manuscript -- the whole predictor-by-endpoint factorial,
the cross-panel agreement bars, the locked 15-pair retrieval metrics and the structural
baselines -- is a function of one object: the upper triangle of a 20-by-20 correlation
matrix, in the raw and the within-ligand-centred parameterisation, for four experimental
panels and for the docking reference.  This script writes that object.

Why it is releasable.  Each column is a single correlation coefficient computed *across*
compounds, so the table is an author-generated aggregate statistic (CC BY 4.0), not a
redistribution of any source panel's activities.  For the most restricted panel, KiRHub,
the underlying block is 92 compounds by 20 kinases (1,840 inhibition values) and what is
released for it is 190 correlations.  A correlation matrix is invariant to affine
rescaling of every kinase column and to any orthogonal transformation of the compound
axis, so the set of 92-by-20 blocks reproducing a given edge vector is a Stiefel manifold
of dimension 20*92 - 20*21/2 = 1,630.  No compound identity, no per-compound profile and
no individual measurement is recoverable from it.

Running this script needs the source panels (PKIS1 publisher archive and the KiRHub
supplementary workbook, both fetched by their documented retrieval contracts).  The
report-layer build does not run it; the released CSV it produces is what readers verify
against.
"""

from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd

try:  # Support direct CLI execution and package-style imports.
    from . import centering_panel_sensitivity as centering
    from . import kirhub_external_validation as kirhub
    from . import residual_target_geometry_validation as geometry
except ImportError:  # pragma: no cover - direct CLI execution.
    import centering_panel_sensitivity as centering  # type: ignore
    import kirhub_external_validation as kirhub  # type: ignore
    import residual_target_geometry_validation as geometry  # type: ignore


PACKAGE = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PACKAGE / "results" / "released_pair_geometry_ledger"
DEFAULT_LOCKED_PAIRS = PACKAGE / "results" / "klifs_pocket_control" / "target_pairs.csv"
DEFAULT_PKIS1_ZIP = centering.DEFAULT_PKIS1_ZIP
DEFAULT_KIRHUB_WORKBOOK = centering.DEFAULT_KIRHUB_WORKBOOK
TARGETS20 = tuple(kirhub.TARGETS)
PANELS = ("DAVIS", "PKIS2", "PKIS1", "KiRHub")


def _edges(matrix: np.ndarray) -> np.ndarray:
    return geometry.upper_triangle(matrix)


def build_ledger(
    *,
    pkis1_zip: Path,
    kirhub_workbook: Path,
    locked_pairs: Path,
) -> tuple[pd.DataFrame, dict[str, object]]:
    reference, score_columns, support, _ = centering.load_reference(pkis1_zip)
    score_index = {target: position for position, target in enumerate(score_columns)}
    target_indices = tuple(score_index[target] for target in TARGETS20)

    local_geometry, _ = centering.centered_geometry(
        reference, target_indices, center_over_all_columns=False
    )
    full_geometry, _ = centering.centered_geometry(
        reference, target_indices, center_over_all_columns=True
    )
    raw_docking_geometry = geometry.target_correlation(reference[:, list(target_indices)])

    experiments: OrderedDict[str, np.ndarray] = OrderedDict(
        kirhub.load_old_experimental_panels(pkis1_zip)
    )
    experiments["KiRHub"] = kirhub.load_kirhub(kirhub_workbook)[
        list(TARGETS20)
    ].to_numpy(dtype=np.float64)
    if tuple(experiments) != PANELS:
        raise ValueError(f"experimental panels must be ordered as {PANELS}")

    columns: OrderedDict[str, np.ndarray] = OrderedDict()
    columns["docking_raw_correlation"] = _edges(raw_docking_geometry)
    columns["docking_local_20_centered_correlation"] = _edges(local_geometry)
    columns["docking_all_58_centered_then_extract_correlation"] = _edges(full_geometry)
    panel_ligands: dict[str, int] = {}
    for panel, matrix in experiments.items():
        panel_ligands[panel] = int(len(matrix))
        columns[f"{panel}_experimental_raw_correlation"] = _edges(
            geometry.geometry_correlation(matrix, "raw")
        )
        columns[f"{panel}_experimental_centered_correlation"] = _edges(
            geometry.geometry_correlation(matrix, "center_then_correlation")
        )

    pairs = [
        (first, second)
        for index, first in enumerate(TARGETS20)
        for second in TARGETS20[index + 1 :]
    ]
    frame = pd.DataFrame(
        {"target_a": [a for a, _ in pairs], "target_b": [b for _, b in pairs], **columns}
    )

    locked = pd.read_csv(locked_pairs)
    label_column = next(
        (
            name
            for name in ("primary_replicated_positive", "replicated_endpoint_positive")
            if name in locked.columns
        ),
        None,
    )
    if label_column is None:
        raise ValueError(
            f"{locked_pairs} carries no locked-endpoint label column; "
            f"observed {list(locked.columns)}"
        )
    merged = frame.merge(
        locked[["target_a", "target_b", label_column]],
        on=["target_a", "target_b"],
        how="left",
        validate="one_to_one",
    ).rename(columns={label_column: "locked_endpoint_positive"})
    if merged.locked_endpoint_positive.isna().any() or len(merged) != 190:
        raise ValueError("the locked 15-of-190 endpoint did not merge onto every edge")

    summary = {
        "artifact": (
            "complete raw and within-ligand-centred target-pair geometry for the fixed "
            "common-20 kinase panel"
        ),
        "targets": list(TARGETS20),
        "target_pairs": int(len(merged)),
        "experimental_panel_ligands": panel_ligands,
        "locked_endpoint_positives": int(merged.locked_endpoint_positive.sum()),
        "docking_support": support,
        "rights_boundary": (
            "Each column is one correlation coefficient computed across compounds. The "
            "table is an author-generated aggregate under CC BY 4.0 and does not "
            "redistribute any source panel's compound-by-target activities. For the "
            "92-compound KiRHub block the 190 released edges leave a 1,630-dimensional "
            "family of compound-level matrices indistinguishable, so no per-compound "
            "profile or individual measurement is recoverable."
        ),
        "reproduces": (
            "the complete predictor-by-endpoint factorial (main-text Table 1), the "
            "cross-panel agreement summary, the locked 15-pair retrieval metrics and "
            "every fixed-panel Spearman reported in the main text"
        ),
        "sources": {
            "pkis1_sha256": geometry.sha256_file(pkis1_zip),
            "kirhub_sha256": geometry.sha256_file(kirhub_workbook),
            "locked_pairs": str(locked_pairs.relative_to(PACKAGE)),
        },
    }
    return merged, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pkis1-zip", type=Path, default=DEFAULT_PKIS1_ZIP)
    parser.add_argument("--kirhub-workbook", type=Path, default=DEFAULT_KIRHUB_WORKBOOK)
    parser.add_argument("--locked-pairs", type=Path, default=DEFAULT_LOCKED_PAIRS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    frame, summary = build_ledger(
        pkis1_zip=args.pkis1_zip,
        kirhub_workbook=args.kirhub_workbook,
        locked_pairs=args.locked_pairs,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output / "target_pair_geometry.csv", index=False)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Wrote {len(frame)} edges and {len(frame.columns)} columns")


if __name__ == "__main__":
    main()
