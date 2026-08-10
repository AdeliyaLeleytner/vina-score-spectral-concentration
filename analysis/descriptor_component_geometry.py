#!/usr/bin/env python3
"""Does docking--experiment target-pair agreement run through ligand physicochemistry?

Two facts are established separately elsewhere in this package: roughly half of the
target correlation left in a within-ligand-centred Vina matrix is predictable from seven
ordinary ligand descriptors under chemical-group holdout, and the centred docking
geometry agrees with experimental kinase co-response at Spearman 0.17--0.32 over the
fixed 190 target pairs.  This analysis joins them.

The out-of-fold descriptor-predicted component of the residual surface is itself a dense
ligand-by-target matrix.  We form its target-pair geometry, and the geometry of the
descriptor-removed remainder, and score both against the same four experimental panels
over the same 190 edges.  The split is informative in every direction:

* if the descriptor component alone reaches the observed 0.17--0.32, the agreement is
  carried by ligand size and lipophilicity rather than by anything protein-specific;
* if it reaches zero, the agreement is carried by the part of the residual that ordinary
  physicochemistry cannot predict;
* anything between is a quantitative decomposition of the agreement.

The docking support, target order and experimental endpoint are identical to
``fixed20_centering_panel_sensitivity``, so the numbers are directly comparable with the
manuscript's fixed-20 estimand.  The published 190-edge experimental geometries are read
from that analysis's released table rather than recomputed, which keeps this script free
of the source-restricted KiRHub workbook.
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
    from . import dense_davis_benchmark as davis
    from . import kirhub_external_validation as kirhub
    from . import residual_mechanism_analysis as mechanism
    from . import residual_target_geometry_validation as geometry
except ImportError:  # pragma: no cover - direct CLI execution.
    import centering_panel_sensitivity as centering  # type: ignore
    import dense_davis_benchmark as davis  # type: ignore
    import kirhub_external_validation as kirhub  # type: ignore
    import residual_mechanism_analysis as mechanism  # type: ignore
    import residual_target_geometry_validation as geometry  # type: ignore


PACKAGE = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PACKAGE / "results" / "descriptor_component_geometry"
DEFAULT_PUBLISHED_GEOMETRY = (
    PACKAGE / "results" / "fixed20_centering_panel_sensitivity" / "target_pair_geometry.csv"
)
DEFAULT_PKIS1_ZIP = centering.DEFAULT_PKIS1_ZIP
DEFAULT_SEED = 20260812
DEFAULT_FOLDS = 5
TARGETS20 = tuple(kirhub.TARGETS)
PANELS = ("DAVIS", "PKIS2", "PKIS1", "KiRHub")

# Named exactly as in the manuscript's Results and in residual_mechanism_analysis.
SURFACE_LABELS = OrderedDict(
    [
        ("observed_residual", "out-of-fold within-ligand residual surface"),
        (
            "descriptor_component",
            "out-of-fold prediction of that surface from seven ligand descriptors",
        ),
        (
            "descriptor_removed",
            "the same surface after subtracting the out-of-fold descriptor prediction",
        ),
    ]
)


def load_deleaked_reference(
    pkis1_zip: Path,
) -> tuple[np.ndarray, tuple[str, ...], np.ndarray, dict]:
    """Rebuild the fixed-20 docking reference and carry its aligned SMILES.

    This mirrors ``centering_panel_sensitivity.load_reference`` exactly -- the complete
    260,060-row DOCKSTRING support, minus every connectivity block that occurs in DAVIS,
    PKIS2 or PKIS1, with positive scores clipped to zero -- and additionally returns the
    SMILES column, which the descriptor decomposition needs and the published loader does
    not expose.  ``verify_against_published_geometry`` checks the reconstruction.
    """
    dockstring, davis_identity, _, _ = davis._load_inputs(  # noqa: SLF001
        davis.DEFAULT_DOCKSTRING,
        davis.DEFAULT_DAVIS,
        identity_scan="full",
    )
    pkis2_frame = geometry.load_pkis2_full()
    pkis1_frame = geometry.load_pkis1_full(pkis1_zip)
    excluded_blocks = set(davis_identity.connectivity_block.dropna())
    excluded_blocks |= set(pkis2_frame.connectivity_block.dropna())
    excluded_blocks |= set(pkis1_frame.connectivity_block.dropna())
    keep = ~dockstring.connectivity_block.isin(excluded_blocks)

    score_columns = centering._score_columns(davis.DEFAULT_DOCKSTRING)  # noqa: SLF001
    released = pd.read_csv(
        davis.DEFAULT_DOCKSTRING,
        sep="\t",
        usecols=list(score_columns),
        dtype={target: np.float64 for target in score_columns},
    )
    selected_rows = dockstring.dockstring_row.to_numpy(dtype=int)
    complete_scores = released.iloc[selected_rows].to_numpy(dtype=np.float64)
    if complete_scores.shape != (260_060, 58) or not np.isfinite(complete_scores).all():
        raise ValueError("the reconstructed complete DOCKSTRING surface is not 260,060 x 58")
    keep_mask = keep.to_numpy(dtype=bool)
    reference = np.minimum(complete_scores[keep_mask], 0.0)
    smiles = dockstring.loc[keep_mask, "smiles"].to_numpy(dtype=object)
    if len(reference) != len(smiles) or len(reference) < 259_000:
        raise ValueError("the de-leaked reference and its SMILES column are not aligned")
    support = {
        "dockstring_complete_rows_before_exclusion": int(len(complete_scores)),
        "dockstring_reference_rows": int(len(reference)),
        "dockstring_rows_excluded": int((~keep_mask).sum()),
        "excluded_connectivity_blocks": int(len(excluded_blocks)),
        "positive_score_handling": "clip to zero before centering, as in the primary analysis",
        "target_columns": list(score_columns),
    }
    return reference, tuple(score_columns), smiles, support


def load_published_experimental_geometry(path: Path) -> tuple[pd.DataFrame, np.ndarray]:
    """Read the released 190-edge table and confirm its pair order."""
    frame = pd.read_csv(path)
    expected_pairs = [
        (first, second)
        for index, first in enumerate(TARGETS20)
        for second in TARGETS20[index + 1 :]
    ]
    observed_pairs = list(zip(frame.target_a, frame.target_b))
    if observed_pairs != expected_pairs:
        raise ValueError(
            "the released target-pair table is not in the locked upper-triangle order"
        )
    missing = [
        f"{panel}_experimental_centered_correlation"
        for panel in PANELS
        if f"{panel}_experimental_centered_correlation" not in frame.columns
    ]
    if missing:
        raise ValueError(f"the released target-pair table is missing columns: {missing}")
    published_docking = frame["docking_local_20_centered_correlation"].to_numpy(
        dtype=np.float64
    )
    return frame, published_docking


def _square_from_edges(edges: np.ndarray) -> np.ndarray:
    """Rebuild a 20x20 symmetric correlation matrix from its 190 upper-triangle edges."""
    matrix = np.eye(len(TARGETS20), dtype=np.float64)
    rows, columns = np.triu_indices(len(TARGETS20), k=1)
    matrix[rows, columns] = edges
    matrix[columns, rows] = edges
    return matrix


def evaluate(
    reference: np.ndarray,
    score_columns: tuple[str, ...],
    smiles: np.ndarray,
    published: pd.DataFrame,
    published_docking: np.ndarray,
    *,
    folds: int,
    permutations: int,
    seed: int,
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    score_index = {target: position for position, target in enumerate(score_columns)}
    target_indices = [score_index[target] for target in TARGETS20]
    block = reference[:, target_indices]

    descriptors = mechanism.molecular_descriptors(smiles)
    groups = mechanism.scaffold_keys(smiles)
    finite = np.isfinite(descriptors).all(axis=1)
    if not finite.all():
        block = block[finite]
        descriptors = descriptors[finite]
        groups = groups[finite]

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
    geometries = OrderedDict(
        (name, geometry.target_correlation(surface)) for name, surface in surfaces.items()
    )

    # Self-check: the out-of-fold observed residual must reproduce the released fixed-20
    # docking geometry.  Fold-local offsets and scales cannot change a correlation matrix,
    # so any real disagreement means the support or the target order drifted.
    reconstruction = geometry.geometry_concordance(
        geometries["observed_residual"], _square_from_edges(published_docking)
    )
    if reconstruction < 0.99:
        raise ValueError(
            "the reconstructed fixed-20 docking geometry does not match the released "
            f"table (Spearman {reconstruction:.4f}); the support or target order drifted"
        )

    rows: list[dict[str, object]] = []
    panel_results: OrderedDict[str, dict[str, object]] = OrderedDict()
    for panel_index, panel in enumerate(PANELS):
        experimental_edges = published[
            f"{panel}_experimental_centered_correlation"
        ].to_numpy(dtype=np.float64)
        experimental_geometry = _square_from_edges(experimental_edges)
        panel_record: dict[str, object] = {}
        for surface_index, (name, docking_geometry) in enumerate(geometries.items()):
            test = geometry.qap_test(
                docking_geometry,
                experimental_geometry,
                permutations,
                seed + panel_index * 1_000_000 + surface_index * 10_000,
            )
            panel_record[name] = {
                "spearman": test["observed_spearman"],
                "target_label_qap_p_positive": test["one_sided_p_positive"],
                "seed": test["seed"],
            }
            rows.append(
                {
                    "panel": panel,
                    "docking_surface": name,
                    "spearman": test["observed_spearman"],
                    "target_label_qap_p_positive": test["one_sided_p_positive"],
                }
            )
        observed = panel_record["observed_residual"]["spearman"]
        component = panel_record["descriptor_component"]["spearman"]
        removed = panel_record["descriptor_removed"]["spearman"]
        panel_record["descriptor_mediated_fraction_of_observed_agreement"] = (
            float(component / observed) if observed > 0 else None
        )
        panel_record["descriptor_removed_fraction_of_observed_agreement"] = (
            float(removed / observed) if observed > 0 else None
        )
        panel_results[panel] = panel_record

    component_values = [
        panel_results[panel]["descriptor_component"]["spearman"] for panel in PANELS
    ]
    removed_values = [
        panel_results[panel]["descriptor_removed"]["spearman"] for panel in PANELS
    ]
    observed_values = [
        panel_results[panel]["observed_residual"]["spearman"] for panel in PANELS
    ]

    pair_rows = [
        {
            "target_a": published.target_a.iloc[edge],
            "target_b": published.target_b.iloc[edge],
            **{
                f"docking_{name}_correlation": geometry.upper_triangle(matrix)[edge]
                for name, matrix in geometries.items()
            },
            **{
                f"{panel}_experimental_centered_correlation": published[
                    f"{panel}_experimental_centered_correlation"
                ].iloc[edge]
                for panel in PANELS
            },
        }
        for edge in range(len(published))
    ]

    summary = {
        "analysis": (
            "descriptor-mediated versus descriptor-independent target-pair agreement "
            "on the fixed common-20 kinase panel"
        ),
        "status": "exploratory_post_hoc_science_only",
        "question": (
            "Is the agreement between centred Vina target-pair geometry and experimental "
            "kinase co-response carried by the part of the residual surface that seven "
            "ordinary ligand descriptors can predict out of fold, or by the part they "
            "cannot?"
        ),
        "targets": list(TARGETS20),
        "target_pairs": len(published),
        "surfaces": dict(SURFACE_LABELS),
        "descriptors": list(mechanism.DESCRIPTOR_NAMES),
        "decomposition_metrics": decomposition["metrics"],
        "released_geometry_reconstruction_spearman": reconstruction,
        "experimental_panel_results": panel_results,
        "range_across_panels": {
            "observed_residual": [min(observed_values), max(observed_values)],
            "descriptor_component": [min(component_values), max(component_values)],
            "descriptor_removed": [min(removed_values), max(removed_values)],
        },
        "configuration": {
            "folds": int(folds),
            "group_definition": (
                "RDKit Bemis-Murcko scaffolds; every acyclic molecule is a singleton"
            ),
            "qap_permutations": int(permutations),
            "seed": int(seed),
            "permutation_unit": "complete target labels",
            "fixed_endpoint": "two-way-centered experimental target correlation",
            "experimental_geometry_source": str(
                DEFAULT_PUBLISHED_GEOMETRY.relative_to(PACKAGE)
            ),
        },
        "claim_boundary": (
            "This is a post hoc predictive decomposition on one fixed 20-kinase panel and "
            "one ligand support. A descriptor component that agrees with experiment does "
            "not show that experimental co-response is caused by ligand size; it shows "
            "that the docking geometry's agreement with experiment is reproduced by a "
            "surface built only from ligand physicochemistry and target-heterogeneous "
            "responses to it. QAP probabilities are conditional on this fixed panel."
        ),
    }
    return summary, pd.DataFrame.from_records(rows), pd.DataFrame.from_records(pair_rows)


def run(
    *,
    pkis1_zip: Path,
    published_geometry: Path,
    folds: int,
    permutations: int,
    seed: int,
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    reference, score_columns, smiles, support = load_deleaked_reference(pkis1_zip)
    published, published_docking = load_published_experimental_geometry(published_geometry)
    summary, table, pair_table = evaluate(
        reference,
        score_columns,
        smiles,
        published,
        published_docking,
        folds=folds,
        permutations=permutations,
        seed=seed,
    )
    summary["support"] = support
    return summary, table, pair_table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pkis1-zip", type=Path, default=DEFAULT_PKIS1_ZIP)
    parser.add_argument(
        "--published-geometry", type=Path, default=DEFAULT_PUBLISHED_GEOMETRY
    )
    parser.add_argument("--folds", type=int, default=DEFAULT_FOLDS)
    parser.add_argument("--permutations", type=int, default=49_999)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    summary, table, pair_table = run(
        pkis1_zip=args.pkis1_zip,
        published_geometry=args.published_geometry,
        folds=args.folds,
        permutations=args.permutations,
        seed=args.seed,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(
        json.dumps(mechanism.json_ready(summary), indent=2) + "\n"
    )
    table.to_csv(args.output / "panel_concordance.csv", index=False)
    pair_table.to_csv(args.output / "target_pair_geometry.csv", index=False)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
