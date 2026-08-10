#!/usr/bin/env python3
"""Strict common-20 chemical-exclusion and reference-support sensitivities."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from . import dense_davis_benchmark as davis
    from . import kirhub_external_validation as kirhub
    from . import residual_target_geometry_validation as geometry
except ImportError:  # pragma: no cover
    import dense_davis_benchmark as davis  # type: ignore
    import kirhub_external_validation as kirhub  # type: ignore
    import residual_target_geometry_validation as geometry  # type: ignore


PACKAGE = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PACKAGE / "results/fixed20_reference_sensitivity"
DEFAULT_SEEDS = (11, 29, 47, 71, 97)


def summarize(values: list[float]) -> dict[str, float | int]:
    data = np.asarray(values, dtype=np.float64)
    return {
        "n": int(len(data)),
        "mean": float(data.mean()),
        "minimum": float(data.min()),
        "maximum": float(data.max()),
        "standard_deviation": float(data.std(ddof=1)) if len(data) > 1 else 0.0,
    }


def concordances(reference: np.ndarray, experiment: np.ndarray) -> dict[str, float]:
    raw_docking = kirhub.correlation_geometry(reference, "raw")
    centered_docking = kirhub.correlation_geometry(reference, "two_way_center")
    raw_experiment = geometry.target_correlation(experiment)
    centered_experiment = geometry.geometry_correlation(
        experiment, "center_then_correlation"
    )
    return {
        "raw_docking__raw_experiment": geometry.geometry_concordance(
            raw_docking, raw_experiment
        ),
        "raw_docking__centered_experiment": geometry.geometry_concordance(
            raw_docking, centered_experiment
        ),
        "centered_docking__raw_experiment": geometry.geometry_concordance(
            centered_docking, raw_experiment
        ),
        "centered_docking__centered_experiment": geometry.geometry_concordance(
            centered_docking, centered_experiment
        ),
    }


def run(
    *,
    pkis1_zip: Path,
    kirhub_workbook: Path,
    support_size: int,
    support_seeds: tuple[int, ...],
) -> tuple[dict, pd.DataFrame]:
    dockstring, davis_identity, davis_experiment, _ = davis._load_inputs(  # noqa: SLF001
        davis.DEFAULT_DOCKSTRING,
        davis.DEFAULT_DAVIS,
        identity_scan="full",
    )
    pkis2 = geometry.load_pkis2_full()
    pkis1 = geometry.load_pkis1_full(pkis1_zip)
    kirhub_frame = kirhub.load_kirhub(kirhub_workbook)

    excluded_blocks = set(davis_identity.connectivity_block.dropna())
    excluded_blocks |= set(pkis2.connectivity_block.dropna())
    excluded_blocks |= set(pkis1.connectivity_block.dropna())
    keep = ~dockstring.connectivity_block.isin(excluded_blocks)
    reference_frame = dockstring.loc[keep].copy()
    reference = np.minimum(
        reference_frame[list(kirhub.TARGETS)].to_numpy(dtype=np.float64), 0.0
    )
    if reference.shape != (259_579, 20):
        raise ValueError(f"unexpected fixed-20 reference shape: {reference.shape}")

    experiments = {
        "DAVIS": davis_experiment[list(kirhub.TARGETS)].to_numpy(dtype=np.float64),
        "PKIS2": pkis2[list(kirhub.TARGETS)].to_numpy(dtype=np.float64),
        "PKIS1": pkis1[list(kirhub.TARGETS)].to_numpy(dtype=np.float64),
        "KiRHub": kirhub_frame[list(kirhub.TARGETS)].to_numpy(dtype=np.float64),
    }
    primary = {
        panel: concordances(reference, experiment)
        for panel, experiment in experiments.items()
    }

    davis_smiles = (
        davis_identity.set_index("drug_name")
        .loc[davis_experiment.index, "compound_iso_smiles"]
        .reset_index(drop=True)
    )
    experimental_scaffolds = {
        key
        for key in np.concatenate(
            [
                geometry.murcko_scaffold_keys(davis_smiles),
                geometry.murcko_scaffold_keys(pkis2["Smiles"].reset_index(drop=True)),
                geometry.murcko_scaffold_keys(pkis1["SMILES"].reset_index(drop=True)),
            ]
        )
        if key
    }
    reference_scaffolds = geometry.murcko_scaffold_keys(reference_frame["smiles"])
    scaffold_keep = ~np.isin(reference_scaffolds, list(experimental_scaffolds))
    scaffold_reference = reference[scaffold_keep]
    scaffold_results: dict[str, dict] = {}
    for panel in ("DAVIS", "PKIS2", "PKIS1"):
        values = concordances(scaffold_reference, experiments[panel])
        values["centered_delta_from_primary"] = (
            values["centered_docking__centered_experiment"]
            - primary[panel]["centered_docking__centered_experiment"]
        )
        values["raw_delta_from_primary"] = (
            values["raw_docking__raw_experiment"]
            - primary[panel]["raw_docking__raw_experiment"]
        )
        scaffold_results[panel] = values

    if support_size < 100 or support_size > len(reference):
        raise ValueError("invalid reference support size")
    rows: list[dict] = []
    for seed in support_seeds:
        rng = np.random.default_rng(seed)
        indices = np.sort(rng.choice(len(reference), support_size, replace=False))
        sampled = reference[indices]
        for panel, experiment in experiments.items():
            values = concordances(sampled, experiment)
            rows.append(
                {
                    "panel": panel,
                    "reference_ligands": support_size,
                    "seed": seed,
                    **values,
                }
            )
    frame = pd.DataFrame.from_records(rows)
    support_summary: dict[str, dict] = {}
    for panel, group in frame.groupby("panel", sort=False):
        support_summary[panel] = {
            metric: summarize(group[metric].tolist())
            for metric in (
                "raw_docking__raw_experiment",
                "raw_docking__centered_experiment",
                "centered_docking__raw_experiment",
                "centered_docking__centered_experiment",
            )
        }

    summary = {
        "analysis": "strict common-20 reference-support sensitivity",
        "status": "exploratory_post_hoc",
        "targets": list(kirhub.TARGETS),
        "target_pairs": 190,
        "reference": {
            "rows_after_connectivity_exclusion": int(len(reference)),
            "excluded_connectivity_blocks": int(len(excluded_blocks)),
            "positive_scores_clipped_to_zero": True,
        },
        "primary_full_reference": primary,
        "same_cyclic_murcko_exclusion": {
            "experimental_panels_with_structures": ["DAVIS", "PKIS2", "PKIS1"],
            "experimental_cyclic_scaffolds": int(len(experimental_scaffolds)),
            "reference_rows_excluded": int((~scaffold_keep).sum()),
            "reference_rows_remaining": int(scaffold_keep.sum()),
            "panels": scaffold_results,
            "KiRHub_boundary": (
                "KiRHub compound structures were unavailable, so no KiRHub-specific "
                "scaffold exclusion was possible."
            ),
        },
        "random_reference_supports": {
            "support_size": int(support_size),
            "seeds": list(support_seeds),
            "panels": support_summary,
        },
        "claim_boundary": (
            "These checks use the same 20 targets and the same local-20 centering "
            "contract as the primary factorial comparison. They assess sensitivity "
            "to reference-library composition, not biological validity."
        ),
    }
    return summary, frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pkis1-zip", type=Path, default=Path("/tmp/pkis1_supplement.zip"))
    parser.add_argument(
        "--kirhub-workbook", type=Path, default=Path("/tmp/kirhub_supp_tables.xlsx")
    )
    parser.add_argument("--support-size", type=int, default=15_000)
    parser.add_argument(
        "--support-seeds",
        default=",".join(str(value) for value in DEFAULT_SEEDS),
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    seeds = tuple(int(value) for value in args.support_seeds.split(",") if value)
    summary, frame = run(
        pkis1_zip=args.pkis1_zip,
        kirhub_workbook=args.kirhub_workbook,
        support_size=args.support_size,
        support_seeds=seeds,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    frame.to_csv(args.output / "support_concordance.csv", index=False)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
