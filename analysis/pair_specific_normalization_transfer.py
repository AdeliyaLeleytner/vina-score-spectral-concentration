#!/usr/bin/env python3
"""Transfer target-pair-specific Vina normalization rules across assay panels.

The analysis asks whether the weak average effect of target-specific scaling is
a mixture of reproducible target-pair regimes.  For each target pair, a
discovery panel chooses either absolute Vina or the frozen two-way-residual
representation using only discovery experimental outcomes.  The resulting
lookup table is then applied, without retuning, to future compounds in a
different experimental panel containing the same fixed targets.

PKIS2 and DAVIS exact-full-InChIKey matched panels are primary.  PKIS1 is a
connectivity-block sensitivity because no exact full-key overlap exists.  The
pairwise rule need not be transitive and is therefore evaluated only as a
pairwise preference/counter-screen classifier, never as a global target rank.
No manuscript file is read or modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

import dense_davis_benchmark as davis
import dense_pkis2_benchmark as pkis2_benchmark
import residual_target_geometry_validation as geometry


PACKAGE = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PACKAGE / "results" / "pair_specific_normalization_transfer"
DEFAULT_PKIS1_ZIP = Path("/tmp/pkis1_supplement.zip")
DEFAULT_QAP_PERMUTATIONS = 50_000
DEFAULT_BOOTSTRAPS = 5_000
DEFAULT_SEED = 202_608_21

ALL_TARGETS = tuple(davis.TARGET_MAP)
TARGETS = tuple(geometry.PKIS1_TARGET_MAP)
if tuple(pkis2_benchmark.TARGET_MAP) != ALL_TARGETS:
    raise RuntimeError("DAVIS and PKIS2 target order diverged")
if not set(TARGETS).issubset(ALL_TARGETS) or len(TARGETS) != 20:
    raise RuntimeError("expected the frozen common 20-target kinase support")
TARGET_INDICES = np.asarray([ALL_TARGETS.index(target) for target in TARGETS])
TRI = np.triu_indices(len(TARGETS), k=1)

PANEL_SPECS = OrderedDict(
    [
        (
            "PKIS2",
            {
                "match": "exact_full_standard_InChIKey",
                "margin": 10.0,
                "mapping": PACKAGE / "results" / "dense_pkis2_primary_molecule_mapping.csv",
                "mapping_key": "standard_inchikey",
                "fit": PACKAGE / "results" / "dense_pkis2_benchmark.json",
            },
        ),
        (
            "DAVIS",
            {
                "match": "exact_full_standard_InChIKey",
                "margin": 0.5,
                "mapping": PACKAGE / "results" / "dense_davis_primary_molecule_mapping.csv",
                "mapping_key": "drug_name",
                "fit": PACKAGE / "results" / "dense_davis_benchmark.json",
            },
        ),
        (
            "PKIS1",
            {
                "match": "standard_InChI_connectivity_block_sensitivity",
                "margin": 10.0,
                "mapping": None,
                "mapping_key": "connectivity_block",
                "fit": None,
            },
        ),
    ]
)

PRIMARY_TRANSFERS = (("PKIS2", "DAVIS"), ("DAVIS", "PKIS2"))
SENSITIVITY_TRANSFERS = (
    ("PKIS2", "PKIS1"),
    ("DAVIS", "PKIS1"),
    ("PKIS1", "PKIS2"),
    ("PKIS1", "DAVIS"),
)
SUPPORT_GRID = (3, 5, 10, 15, 20)
PRIMARY_MINIMUM_PAIR_SUPPORT = 10
ALPHA_GRID = tuple(np.linspace(0.0, 1.0, 21))


@dataclass(frozen=True)
class Panel:
    name: str
    experiment: np.ndarray
    docking: np.ndarray
    absolute: np.ndarray
    residual: np.ndarray
    smiles: tuple[str, ...]
    dockstring_rows: tuple[tuple[int, ...], ...]
    margin: float
    match: str
    pooled_absolute_sd: float


@dataclass(frozen=True)
class PairCells:
    eligible: np.ndarray
    absolute_correct: np.ndarray
    residual_correct: np.ndarray
    support: np.ndarray
    absolute_accuracy: np.ndarray
    residual_accuracy: np.ndarray


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


def _parse_row_indices(value: object) -> tuple[int, ...]:
    result = tuple(int(item) for item in str(value).split(";") if item.strip())
    if not result:
        raise ValueError("a frozen mapping row has no DOCKSTRING index")
    return result


def _complete_dockstring() -> pd.DataFrame:
    frame = pd.read_csv(davis.DEFAULT_DOCKSTRING, sep="\t")
    score_columns = [
        column for column in frame if column not in {"inchikey", "smiles"}
    ]
    complete = ~frame[score_columns].isna().any(axis=1)
    result = frame.loc[complete, ["inchikey", "smiles", *ALL_TARGETS]].reset_index(
        drop=True
    )
    if len(result) != 260_060 or result[list(ALL_TARGETS)].isna().any().any():
        raise ValueError("the frozen complete DOCKSTRING support changed")
    return result


def _davis_experiment() -> tuple[pd.DataFrame, pd.Series]:
    raw = pd.read_csv(
        davis.DEFAULT_DAVIS,
        sep="\t",
        usecols=["drug_name", "protein", "compound_iso_smiles", "y"],
    )
    molecules = raw[["drug_name", "compound_iso_smiles"]].drop_duplicates()
    if molecules.drug_name.duplicated().any():
        raise ValueError("a DAVIS drug name maps to multiple structures")
    selected = raw[raw.protein.isin(davis.TARGET_MAP.values())]
    matrix = selected.pivot(index="drug_name", columns="protein", values="y")
    matrix = matrix.reindex(
        index=molecules.drug_name,
        columns=[davis.TARGET_MAP[target] for target in ALL_TARGETS],
    )
    matrix.columns = list(ALL_TARGETS)
    if matrix.shape != (72, 21) or matrix.isna().any().any():
        raise ValueError("expected the frozen dense DAVIS 72 x 21 block")
    smiles = molecules.set_index("drug_name").compound_iso_smiles.reindex(matrix.index)
    return matrix, smiles


def _pkis2_experiment() -> tuple[pd.DataFrame, pd.Series]:
    frame = geometry.load_pkis2_full()
    matrix = frame.set_index("standard_inchikey")[list(ALL_TARGETS)]
    smiles = frame.set_index("standard_inchikey").Smiles
    return matrix, smiles


def _frozen_fit(path: Path) -> dict[str, Any]:
    with Path(path).open() as handle:
        fit = json.load(handle)["primary"]["fit"]
    if int(fit["targets"]) != len(ALL_TARGETS):
        raise ValueError("a frozen benchmark fit has the wrong target count")
    return fit


def _fit_from_reference(reference: np.ndarray) -> dict[str, Any]:
    x = np.asarray(reference, dtype=np.float64)
    target_mean = x.mean(axis=0)
    target_sd = x.std(axis=0, ddof=1)
    grand_mean = float(x.mean())
    residual = (
        x
        - target_mean[None, :]
        - x.mean(axis=1, keepdims=True)
        + grand_mean
    )
    residual_sd = residual.std(axis=0, ddof=1)
    if np.any(target_sd <= 0) or np.any(residual_sd <= 0):
        raise ValueError("reference fit contains a constant target")
    return {
        "reference_ligands": int(len(x)),
        "targets": int(x.shape[1]),
        "target_mean": target_mean.tolist(),
        "target_standard_deviation": target_sd.tolist(),
        "residual_target_standard_deviation": residual_sd.tolist(),
        "grand_mean": grand_mean,
    }


def representations(
    evaluation: np.ndarray, fit: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return absolute and frozen residual scores on the common 20 targets."""
    x = np.asarray(evaluation, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != len(ALL_TARGETS):
        raise ValueError("evaluation must align to the frozen 21-target order")
    mean = np.asarray(fit["target_mean"], dtype=np.float64)
    target_sd = np.asarray(fit["target_standard_deviation"], dtype=np.float64)
    residual_sd = np.asarray(
        fit["residual_target_standard_deviation"], dtype=np.float64
    )
    grand = float(fit["grand_mean"])
    residual = x - mean[None, :] - x.mean(axis=1, keepdims=True) + grand
    residual = residual / residual_sd[None, :]
    # Exact law-of-total-variance scaling up to the tiny n/(n-1) convention.
    pooled_absolute_sd = float(
        np.sqrt(np.mean(np.square(target_sd) + np.square(mean - grand)))
    )
    if pooled_absolute_sd <= 0 or not np.isfinite(pooled_absolute_sd):
        raise ValueError("invalid pooled absolute-score scale")
    return (
        x[:, TARGET_INDICES],
        residual[:, TARGET_INDICES],
        pooled_absolute_sd,
    )


def _mapped_exact_panel(
    name: str,
    dockstring: pd.DataFrame,
    experiment: pd.DataFrame,
    smiles: pd.Series,
    mapping_path: Path,
    mapping_key: str,
    fit: dict[str, Any],
) -> Panel:
    mapping = pd.read_csv(mapping_path).sort_values(mapping_key).reset_index(drop=True)
    labels = experiment.index.astype(str)
    mapping = mapping[mapping[mapping_key].astype(str).isin(set(labels))].copy()
    exp_rows: list[np.ndarray] = []
    docking_rows: list[np.ndarray] = []
    smile_rows: list[str] = []
    source_rows: list[tuple[int, ...]] = []
    for row in mapping.itertuples(index=False):
        key = str(getattr(row, mapping_key))
        selected_experiment = experiment.loc[[key], list(ALL_TARGETS)]
        exp_rows.append(np.median(selected_experiment.to_numpy(float), axis=0))
        indices = _parse_row_indices(row.dockstring_row_indices)
        docking_rows.append(
            np.median(
                dockstring.iloc[list(indices)][list(ALL_TARGETS)].to_numpy(float),
                axis=0,
            )
        )
        selected_smiles = smiles.loc[key]
        if isinstance(selected_smiles, pd.Series):
            selected_smiles = selected_smiles.iloc[0]
        smile_rows.append(str(selected_smiles))
        source_rows.append(indices)
    experimental = np.asarray(exp_rows, dtype=np.float64)[:, TARGET_INDICES]
    docking = np.minimum(np.asarray(docking_rows, dtype=np.float64), 0.0)
    absolute, residual, pooled_sd = representations(docking, fit)
    if not np.isfinite(experimental).all() or len(experimental) < 20:
        raise ValueError(f"{name} exact matched panel is invalid")
    return Panel(
        name=name,
        experiment=experimental,
        docking=docking[:, TARGET_INDICES],
        absolute=absolute,
        residual=residual,
        smiles=tuple(smile_rows),
        dockstring_rows=tuple(source_rows),
        margin=float(PANEL_SPECS[name]["margin"]),
        match=str(PANEL_SPECS[name]["match"]),
        pooled_absolute_sd=pooled_sd,
    )


def _pkis1_panel(dockstring: pd.DataFrame, pkis1_zip: Path) -> Panel:
    frame = geometry.load_pkis1_full(pkis1_zip).copy()
    frame["connectivity_block"] = frame.standard_inchikey.astype(str).str[:14]
    ds = dockstring.copy()
    ds["connectivity_block"] = ds.inchikey.astype(str).str[:14]
    common = sorted(set(frame.connectivity_block) & set(ds.connectivity_block))
    if len(common) < 100:
        raise ValueError("unexpectedly small PKIS1 connectivity overlap")
    experimental = (
        frame[frame.connectivity_block.isin(common)]
        .groupby("connectivity_block", sort=True)[list(TARGETS)]
        .median()
    )
    docking = (
        ds[ds.connectivity_block.isin(common)]
        .groupby("connectivity_block", sort=True)[list(ALL_TARGETS)]
        .median()
        .reindex(experimental.index)
        .clip(upper=0.0)
    )
    smiles = (
        frame[frame.connectivity_block.isin(common)]
        .groupby("connectivity_block", sort=True).SMILES.first()
        .reindex(experimental.index)
    )
    row_lookup = (
        ds.reset_index()
        .groupby("connectivity_block", sort=True)["index"]
        .apply(lambda values: tuple(int(value) for value in values))
        .reindex(experimental.index)
    )
    reference = ds.loc[
        ~ds.connectivity_block.isin(common), list(ALL_TARGETS)
    ].to_numpy(float)
    reference = np.minimum(reference, 0.0)
    fit = _fit_from_reference(reference)
    absolute, residual, pooled_sd = representations(docking.to_numpy(float), fit)
    return Panel(
        name="PKIS1",
        experiment=experimental.to_numpy(float),
        docking=docking.to_numpy(float)[:, TARGET_INDICES],
        absolute=absolute,
        residual=residual,
        smiles=tuple(smiles.astype(str)),
        dockstring_rows=tuple(row_lookup),
        margin=float(PANEL_SPECS["PKIS1"]["margin"]),
        match=str(PANEL_SPECS["PKIS1"]["match"]),
        pooled_absolute_sd=pooled_sd,
    )


def load_panels(pkis1_zip: Path) -> OrderedDict[str, Panel]:
    dockstring = _complete_dockstring()
    pkis2_experiment, pkis2_smiles = _pkis2_experiment()
    davis_experiment, davis_smiles = _davis_experiment()
    panels: OrderedDict[str, Panel] = OrderedDict()
    panels["PKIS2"] = _mapped_exact_panel(
        "PKIS2",
        dockstring,
        pkis2_experiment,
        pkis2_smiles,
        Path(PANEL_SPECS["PKIS2"]["mapping"]),
        str(PANEL_SPECS["PKIS2"]["mapping_key"]),
        _frozen_fit(Path(PANEL_SPECS["PKIS2"]["fit"])),
    )
    panels["DAVIS"] = _mapped_exact_panel(
        "DAVIS",
        dockstring,
        davis_experiment,
        davis_smiles,
        Path(PANEL_SPECS["DAVIS"]["mapping"]),
        str(PANEL_SPECS["DAVIS"]["mapping_key"]),
        _frozen_fit(Path(PANEL_SPECS["DAVIS"]["fit"])),
    )
    panels["PKIS1"] = _pkis1_panel(dockstring, pkis1_zip)
    return panels


def pair_cell_arrays(
    scores: np.ndarray, experiment: np.ndarray, margin: float
) -> tuple[np.ndarray, np.ndarray]:
    """Per-ligand/pair eligibility and correctness; lower score is preferred."""
    scores = np.asarray(scores, dtype=np.float64)
    experiment = np.asarray(experiment, dtype=np.float64)
    if scores.shape != experiment.shape or scores.shape[1] != len(TARGETS):
        raise ValueError("score and experiment surfaces must align to 20 targets")
    experimental_difference = experiment[:, TRI[0]] - experiment[:, TRI[1]]
    score_difference = scores[:, TRI[0]] - scores[:, TRI[1]]
    eligible = np.abs(experimental_difference) > float(margin) + 1e-12
    tied = np.isclose(score_difference, 0.0, rtol=0.0, atol=1e-12)
    correct = np.where(
        tied,
        0.5,
        (
            np.sign(experimental_difference) == -np.sign(score_difference)
        ).astype(float),
    )
    return eligible, correct


def pair_cells(
    absolute: np.ndarray,
    residual: np.ndarray,
    experiment: np.ndarray,
    margin: float,
    weights: np.ndarray | None = None,
) -> PairCells:
    eligible, absolute_correct = pair_cell_arrays(absolute, experiment, margin)
    residual_eligible, residual_correct = pair_cell_arrays(
        residual, experiment, margin
    )
    if not np.array_equal(eligible, residual_eligible):
        raise AssertionError("eligibility depends on score representation")
    if weights is None:
        support = eligible.sum(axis=0, dtype=np.float64)
        absolute_sum = (eligible * absolute_correct).sum(axis=0)
        residual_sum = (eligible * residual_correct).sum(axis=0)
    else:
        weights = np.asarray(weights, dtype=np.float64)
        if weights.shape != (len(experiment),) or np.any(weights < 0):
            raise ValueError("weights must be nonnegative and align to ligands")
        support = weights @ eligible
        absolute_sum = weights @ (eligible * absolute_correct)
        residual_sum = weights @ (eligible * residual_correct)
    absolute_accuracy = np.divide(
        absolute_sum,
        support,
        out=np.full(len(support), np.nan),
        where=support > 0,
    )
    residual_accuracy = np.divide(
        residual_sum,
        support,
        out=np.full(len(support), np.nan),
        where=support > 0,
    )
    return PairCells(
        eligible=eligible,
        absolute_correct=absolute_correct,
        residual_correct=residual_correct,
        support=support,
        absolute_accuracy=absolute_accuracy,
        residual_accuracy=residual_accuracy,
    )


def mixed_scores(panel: Panel, alpha: float) -> np.ndarray:
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("mixture alpha must lie in [0, 1]")
    absolute_scaled = panel.absolute / panel.pooled_absolute_sd
    return (1.0 - alpha) * absolute_scaled + alpha * panel.residual


def aggregate_accuracy(
    accuracy: np.ndarray,
    support: np.ndarray,
    mask: np.ndarray,
    *,
    pair_weighted: bool,
) -> float:
    accuracy = np.asarray(accuracy, dtype=float)
    support = np.asarray(support, dtype=float)
    mask = np.asarray(mask, dtype=bool) & np.isfinite(accuracy) & (support > 0)
    if not mask.any():
        return float("nan")
    if pair_weighted:
        return float(np.sum(accuracy[mask] * support[mask]) / np.sum(support[mask]))
    return float(np.mean(accuracy[mask]))


def select_global_alpha(
    panel: Panel,
    minimum_support: int,
    alpha_grid: tuple[float, ...] = ALPHA_GRID,
    weights: np.ndarray | None = None,
    pair_mask: np.ndarray | None = None,
) -> tuple[float, pd.DataFrame]:
    base = pair_cells(
        panel.absolute,
        panel.residual,
        panel.experiment,
        panel.margin,
        weights,
    )
    mask = base.support >= minimum_support
    if pair_mask is not None:
        pair_mask = np.asarray(pair_mask, dtype=bool)
        if pair_mask.shape != mask.shape:
            raise ValueError("global-alpha pair mask has the wrong shape")
        mask &= pair_mask
    rows: list[dict[str, float]] = []
    for alpha in alpha_grid:
        mix = mixed_scores(panel, float(alpha))
        cells = pair_cells(
            mix, mix, panel.experiment, panel.margin, weights
        )
        rows.append(
            {
                "alpha": float(alpha),
                "discovery_pair_weighted_accuracy": aggregate_accuracy(
                    cells.absolute_accuracy,
                    cells.support,
                    mask,
                    pair_weighted=True,
                ),
                "discovery_equal_pair_accuracy": aggregate_accuracy(
                    cells.absolute_accuracy,
                    cells.support,
                    mask,
                    pair_weighted=False,
                ),
            }
        )
    frame = pd.DataFrame.from_records(rows)
    selected = frame.sort_values(
        ["discovery_pair_weighted_accuracy", "alpha"],
        ascending=[False, True],
        kind="mergesort",
    ).iloc[0]
    return float(selected.alpha), frame


def transfer_metrics(
    discovery: Panel,
    validation: Panel,
    minimum_support: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray | float]]:
    discovery_cells = pair_cells(
        discovery.absolute,
        discovery.residual,
        discovery.experiment,
        discovery.margin,
    )
    validation_cells = pair_cells(
        validation.absolute,
        validation.residual,
        validation.experiment,
        validation.margin,
    )
    mask = (
        (discovery_cells.support >= minimum_support)
        & (validation_cells.support >= minimum_support)
    )
    choice = (
        discovery_cells.residual_accuracy
        > discovery_cells.absolute_accuracy
    )
    rule_accuracy = np.where(
        choice,
        validation_cells.residual_accuracy,
        validation_cells.absolute_accuracy,
    )
    alpha, alpha_frame = select_global_alpha(discovery, minimum_support)
    validation_mix = mixed_scores(validation, alpha)
    mix_cells = pair_cells(
        validation_mix,
        validation_mix,
        validation.experiment,
        validation.margin,
    )
    discovery_global_choice = aggregate_accuracy(
        discovery_cells.residual_accuracy,
        discovery_cells.support,
        discovery_cells.support >= minimum_support,
        pair_weighted=True,
    ) > aggregate_accuracy(
        discovery_cells.absolute_accuracy,
        discovery_cells.support,
        discovery_cells.support >= minimum_support,
        pair_weighted=True,
    )
    selected_single = (
        validation_cells.residual_accuracy
        if discovery_global_choice
        else validation_cells.absolute_accuracy
    )
    # Ligand-independent calibration baseline.  It transfers only the discovery
    # panel's target-wise experimental means and therefore reveals apparent
    # gains caused by stable target marginals rather than compound selectivity.
    target_mean_prior = -discovery.experiment.mean(axis=0)
    prior_scores = np.tile(target_mean_prior, (len(validation.experiment), 1))
    prior_cells = pair_cells(
        prior_scores,
        prior_scores,
        validation.experiment,
        validation.margin,
    )
    metrics: dict[str, Any] = {
        "discovery": discovery.name,
        "validation": validation.name,
        "role": (
            "primary_exact_full_key_transfer"
            if (discovery.name, validation.name) in PRIMARY_TRANSFERS
            else "connectivity_match_sensitivity"
        ),
        "minimum_pair_support_in_each_panel": int(minimum_support),
        "eligible_target_pairs": int(mask.sum()),
        "discovery_ligands": int(len(discovery.experiment)),
        "validation_ligands": int(len(validation.experiment)),
        "discovery_match": discovery.match,
        "validation_match": validation.match,
        "pairs_selecting_residual": int(np.sum(choice & mask)),
        "fraction_pairs_selecting_residual": float(np.mean(choice[mask])),
        "selected_global_alpha": alpha,
        "globally_selected_single_representation": (
            "residual" if discovery_global_choice else "absolute"
        ),
    }
    representations = {
        "absolute": validation_cells.absolute_accuracy,
        "residual": validation_cells.residual_accuracy,
        "globally_selected_single": selected_single,
        "globally_selected_alpha_mixture": mix_cells.absolute_accuracy,
        "discovery_experimental_target_mean_prior": prior_cells.absolute_accuracy,
        "transferred_pair_rule": rule_accuracy,
    }
    for weighting, pair_weighted in (
        ("pair_weighted", True), ("equal_target_pair", False)
    ):
        for name, values in representations.items():
            metrics[f"{weighting}_{name}_accuracy"] = aggregate_accuracy(
                values,
                validation_cells.support,
                mask,
                pair_weighted=pair_weighted,
            )
        metrics[f"{weighting}_rule_minus_absolute"] = (
            metrics[f"{weighting}_transferred_pair_rule_accuracy"]
            - metrics[f"{weighting}_absolute_accuracy"]
        )
        metrics[f"{weighting}_rule_minus_residual"] = (
            metrics[f"{weighting}_transferred_pair_rule_accuracy"]
            - metrics[f"{weighting}_residual_accuracy"]
        )
        metrics[f"{weighting}_rule_minus_global_alpha_mixture"] = (
            metrics[f"{weighting}_transferred_pair_rule_accuracy"]
            - metrics[f"{weighting}_globally_selected_alpha_mixture_accuracy"]
        )
        metrics[f"{weighting}_rule_minus_target_mean_prior"] = (
            metrics[f"{weighting}_transferred_pair_rule_accuracy"]
            - metrics[
                f"{weighting}_discovery_experimental_target_mean_prior_accuracy"
            ]
        )
        metrics[f"{weighting}_rule_minus_validation_oracle_best_single"] = (
            metrics[f"{weighting}_transferred_pair_rule_accuracy"]
            - max(
                metrics[f"{weighting}_absolute_accuracy"],
                metrics[f"{weighting}_residual_accuracy"],
            )
        )
    finite = mask & np.isfinite(discovery_cells.residual_accuracy) & np.isfinite(
        validation_cells.residual_accuracy
    )
    discovery_delta = (
        discovery_cells.residual_accuracy - discovery_cells.absolute_accuracy
    )
    validation_delta = (
        validation_cells.residual_accuracy - validation_cells.absolute_accuracy
    )
    metrics["normalization_gain_spearman"] = float(
        stats.spearmanr(discovery_delta[finite], validation_delta[finite]).statistic
    )
    metrics["normalization_gain_sign_agreement"] = float(
        np.mean(np.sign(discovery_delta[finite]) == np.sign(validation_delta[finite]))
    )
    prior_reversal, prior_routine, prior_tied = prior_stratum_masks(
        prior_cells.absolute_correct,
        validation_cells.eligible,
        mask,
    )
    rule_cell_correct = np.where(
        choice[None, :],
        validation_cells.residual_correct,
        validation_cells.absolute_correct,
    )
    metrics["prior_tied_cells"] = int(prior_tied.sum())
    for stratum, cell_mask in (
        ("prior_reversal", prior_reversal),
        ("prior_routine", prior_routine),
    ):
        metrics[f"{stratum}_cells"] = int(cell_mask.sum())
        for name, correct in (
            ("absolute", validation_cells.absolute_correct),
            ("residual", validation_cells.residual_correct),
            ("transferred_pair_rule", rule_cell_correct),
        ):
            metrics[f"{stratum}_{name}_accuracy"] = float(
                np.sum(correct * cell_mask) / np.sum(cell_mask)
            )
    cache: dict[str, np.ndarray | float] = {
        "choice": choice,
        "discovery_support": discovery_cells.support,
        "validation_support": validation_cells.support,
        "validation_absolute_accuracy": validation_cells.absolute_accuracy,
        "validation_residual_accuracy": validation_cells.residual_accuracy,
        "validation_mix_accuracy": mix_cells.absolute_accuracy,
        "minimum_support": float(minimum_support),
        "alpha": alpha,
        "mask": mask,
        "alpha_frame": alpha_frame,
    }
    return metrics, cache


def prior_stratum_masks(
    prior_correct_values: np.ndarray,
    eligible: np.ndarray,
    pair_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return prior-wrong, prior-correct, and prior-tied analysis masks.

    Pairwise correctness uses 0, 0.5, and 1.  Casting that array to bool would
    incorrectly classify ties as correct.  Every stratum is also restricted to
    the target-pair support mask used by the primary transfer estimand.
    """
    prior_correct_values = np.asarray(prior_correct_values, dtype=float)
    eligible = np.asarray(eligible, dtype=bool)
    pair_mask = np.asarray(pair_mask, dtype=bool)
    if prior_correct_values.shape != eligible.shape:
        raise ValueError("prior correctness and eligibility shapes differ")
    if pair_mask.shape != (eligible.shape[1],):
        raise ValueError("pair mask has the wrong shape")
    supported = eligible & pair_mask[None, :]
    decisive_correct = np.isclose(
        prior_correct_values, 1.0, rtol=0.0, atol=1e-12
    )
    decisive_wrong = np.isclose(
        prior_correct_values, 0.0, rtol=0.0, atol=1e-12
    )
    reversal = supported & decisive_wrong
    routine = supported & decisive_correct
    tied = supported & ~(decisive_wrong | decisive_correct)
    return reversal, routine, tied


def _vector_to_symmetric(values: np.ndarray, diagonal: float = 0.0) -> np.ndarray:
    values = np.asarray(values)
    if values.shape != (len(TRI[0]),):
        raise ValueError("pair vector has the wrong length")
    result = np.full((len(TARGETS), len(TARGETS)), diagonal, dtype=values.dtype)
    result[TRI] = values
    result[(TRI[1], TRI[0])] = values
    return result


def qap_transfer_null(
    caches: dict[tuple[str, str], dict[str, np.ndarray | float]],
    transfers: tuple[tuple[str, str], ...],
    permutations: int,
    seed: int,
) -> tuple[pd.DataFrame, np.ndarray, dict[str, Any]]:
    """Jointly permute learned pair identities; validation outcomes stay fixed."""
    rng = np.random.default_rng(seed)
    null = np.empty((permutations, len(transfers)), dtype=np.float64)
    observed: list[float] = []
    matrices: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    for transfer in transfers:
        cache = caches[transfer]
        choice = np.asarray(cache["choice"], dtype=bool)
        discovery_support = np.asarray(cache["discovery_support"], dtype=float)
        validation_support = np.asarray(cache["validation_support"], dtype=float)
        absolute = np.asarray(cache["validation_absolute_accuracy"], dtype=float)
        residual = np.asarray(cache["validation_residual_accuracy"], dtype=float)
        mixture = np.asarray(cache["validation_mix_accuracy"], dtype=float)
        minimum = int(float(cache["minimum_support"]))
        mask = (discovery_support >= minimum) & (validation_support >= minimum)
        rule = np.where(choice, residual, absolute)
        observed.append(
            aggregate_accuracy(rule, validation_support, mask, pair_weighted=True)
            - aggregate_accuracy(
                mixture, validation_support, mask, pair_weighted=True
            )
        )
        matrices[transfer] = {
            "choice": _vector_to_symmetric(choice),
            "discovery_support": _vector_to_symmetric(discovery_support),
            "validation_support": validation_support,
            "absolute": absolute,
            "residual": residual,
            "mixture": mixture,
        }
    for repetition in range(permutations):
        order = rng.permutation(len(TARGETS))
        for index, transfer in enumerate(transfers):
            item = matrices[transfer]
            permuted_choice = item["choice"][np.ix_(order, order)][TRI].astype(bool)
            permuted_support = item["discovery_support"][np.ix_(order, order)][TRI]
            minimum = int(float(caches[transfer]["minimum_support"]))
            support = item["validation_support"]
            mask = (permuted_support >= minimum) & (support >= minimum)
            rule = np.where(permuted_choice, item["residual"], item["absolute"])
            null[repetition, index] = (
                aggregate_accuracy(rule, support, mask, pair_weighted=True)
                - aggregate_accuracy(
                    item["mixture"], support, mask, pair_weighted=True
                )
            )
    rows: list[dict[str, Any]] = []
    observed_array = np.asarray(observed)
    for index, transfer in enumerate(transfers):
        rows.append(
            {
                "discovery": transfer[0],
                "validation": transfer[1],
                "observed_rule_minus_global_alpha_mixture": observed[index],
                "qap_p_one_sided": float(
                    (1 + np.sum(null[:, index] >= observed[index]))
                    / (permutations + 1)
                ),
                "null_mean": float(np.mean(null[:, index])),
                "null_q025": float(np.quantile(null[:, index], 0.025)),
                "null_q975": float(np.quantile(null[:, index], 0.975)),
                "permutations": int(permutations),
            }
        )
    null_mean = null.mean(axis=1)
    observed_mean = float(observed_array.mean())
    omnibus = {
        "transfers": [list(item) for item in transfers],
        "observed_mean_rule_minus_global_alpha_mixture": observed_mean,
        "joint_target_label_qap_p_one_sided": float(
            (1 + np.sum(null_mean >= observed_mean)) / (permutations + 1)
        ),
        "null_mean": float(np.mean(null_mean)),
        "null_interval_95": [
            float(np.quantile(null_mean, 0.025)),
            float(np.quantile(null_mean, 0.975)),
        ],
        "permutations": int(permutations),
        "seed": int(seed),
    }
    return pd.DataFrame.from_records(rows), null, omnibus


def target_jackknife(
    panels: dict[str, Panel],
    metrics: dict[tuple[str, str], dict[str, Any]],
    caches: dict[tuple[str, str], dict[str, np.ndarray | float]],
    transfers: tuple[tuple[str, str], ...],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    omnibus_values: list[float] = []
    alpha_cache: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    for transfer in transfers:
        discovery = panels[transfer[0]]
        validation = panels[transfer[1]]
        discovery_support = np.asarray(
            caches[transfer]["discovery_support"], dtype=float
        )
        discovery_mix = []
        validation_mix = []
        for alpha in ALPHA_GRID:
            discovery_scores = mixed_scores(discovery, alpha)
            discovery_mix.append(
                pair_cells(
                    discovery_scores,
                    discovery_scores,
                    discovery.experiment,
                    discovery.margin,
                ).absolute_accuracy
            )
            validation_scores = mixed_scores(validation, alpha)
            validation_mix.append(
                pair_cells(
                    validation_scores,
                    validation_scores,
                    validation.experiment,
                    validation.margin,
                ).absolute_accuracy
            )
        alpha_cache[transfer] = {
            "discovery_support": discovery_support,
            "discovery_mix_accuracy": np.asarray(discovery_mix),
            "validation_mix_accuracy": np.asarray(validation_mix),
        }
    for removed in range(len(TARGETS)):
        keep_target = np.arange(len(TARGETS)) != removed
        keep_pair = keep_target[TRI[0]] & keep_target[TRI[1]]
        transfer_gains: list[float] = []
        for transfer in transfers:
            cache = caches[transfer]
            choice = np.asarray(cache["choice"], dtype=bool)
            discovery_support = np.asarray(cache["discovery_support"], dtype=float)
            validation_support = np.asarray(cache["validation_support"], dtype=float)
            minimum = int(float(cache["minimum_support"]))
            mask = (
                keep_pair
                & (discovery_support >= minimum)
                & (validation_support >= minimum)
            )
            alpha_data = alpha_cache[transfer]
            discovery_alpha_mask = (
                keep_pair
                & (np.asarray(alpha_data["discovery_support"]) >= minimum)
            )
            alpha_scores = [
                aggregate_accuracy(
                    values,
                    np.asarray(alpha_data["discovery_support"]),
                    discovery_alpha_mask,
                    pair_weighted=True,
                )
                for values in np.asarray(alpha_data["discovery_mix_accuracy"])
            ]
            selected_alpha_index = int(np.flatnonzero(
                np.asarray(alpha_scores) == np.max(alpha_scores)
            )[0])
            rule = np.where(
                choice,
                np.asarray(cache["validation_residual_accuracy"], dtype=float),
                np.asarray(cache["validation_absolute_accuracy"], dtype=float),
            )
            gain = aggregate_accuracy(
                rule, validation_support, mask, pair_weighted=True
            ) - aggregate_accuracy(
                np.asarray(alpha_data["validation_mix_accuracy"])[selected_alpha_index],
                validation_support,
                mask,
                pair_weighted=True,
            )
            transfer_gains.append(gain)
            rows.append(
                {
                    "removed_target": TARGETS[removed],
                    "discovery": transfer[0],
                    "validation": transfer[1],
                    "rule_minus_global_alpha_mixture": gain,
                }
            )
        mean_gain = float(np.mean(transfer_gains))
        omnibus_values.append(mean_gain)
        rows.append(
            {
                "removed_target": TARGETS[removed],
                "discovery": "locked_exact_mean",
                "validation": "locked_exact_mean",
                "rule_minus_global_alpha_mixture": mean_gain,
            }
        )
    values = np.asarray(omnibus_values)
    full = float(
        np.mean(
            [
                metrics[transfer][
                    "pair_weighted_rule_minus_global_alpha_mixture"
                ]
                for transfer in transfers
            ]
        )
    )
    pseudo = len(TARGETS) * full - (len(TARGETS) - 1) * values
    estimate = float(np.mean(pseudo))
    standard_error = float(
        np.sqrt(np.sum(np.square(pseudo - estimate)) / (len(TARGETS) * (len(TARGETS) - 1)))
    )
    summary = {
        "full_locked_mean_gain": full,
        "leave_one_target_out_minimum": float(values.min()),
        "leave_one_target_out_maximum": float(values.max()),
        "positive_deletions": int(np.sum(values > 0)),
        "deletions": int(len(values)),
        "jackknife_pseudovalue_estimate": estimate,
        "jackknife_standard_error": standard_error,
        "normal_approximation_interval_95": [
            estimate - 1.96 * standard_error,
            estimate + 1.96 * standard_error,
        ],
    }
    return pd.DataFrame.from_records(rows), summary


def _cluster_weights(labels: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    groups, inverse = np.unique(labels, return_inverse=True)
    multiplicity = np.bincount(
        rng.integers(0, len(groups), size=len(groups)), minlength=len(groups)
    )
    return multiplicity[inverse].astype(float)


def cluster_bootstrap(
    panels: dict[str, Panel],
    caches: dict[tuple[str, str], dict[str, np.ndarray | float]],
    transfers: tuple[tuple[str, str], ...],
    repetitions: int,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    labels = {
        name: davis._butina_labels(  # noqa: SLF001
            pd.Series(panel.smiles), similarity_threshold=0.65
        )
        for name, panel in panels.items()
    }
    # Precompute every cell-level decision once.  Alpha selection and validation
    # mixture evaluation can then be reduced to weighted per-ligand totals,
    # while only the two endpoint representations require pairwise aggregates.
    precomputed: dict[str, dict[str, Any]] = {}
    for name, panel in panels.items():
        eligible, absolute_correct = pair_cell_arrays(
            panel.absolute, panel.experiment, panel.margin
        )
        _, residual_correct = pair_cell_arrays(
            panel.residual, panel.experiment, panel.margin
        )
        mix_correct: list[np.ndarray] = []
        for alpha in ALPHA_GRID:
            _, correct = pair_cell_arrays(
                mixed_scores(panel, alpha), panel.experiment, panel.margin
            )
            mix_correct.append(correct)
        precomputed[name] = {
            "eligible": eligible.astype(float),
            "absolute_eligible_correct": eligible * absolute_correct,
            "residual_eligible_correct": eligible * residual_correct,
            "mix_eligible_correct": np.asarray(
                [eligible * correct for correct in mix_correct], dtype=float
            ),
        }
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    omnibus = np.empty(repetitions, dtype=float)
    for repetition in range(repetitions):
        gains: list[float] = []
        panel_weights = {
            name: _cluster_weights(group_labels, rng)
            for name, group_labels in labels.items()
        }
        for transfer in transfers:
            discovery = panels[transfer[0]]
            validation = panels[transfer[1]]
            cache = caches[transfer]
            discovery_weights = panel_weights[transfer[0]]
            validation_weights = panel_weights[transfer[1]]
            discovery_data = precomputed[transfer[0]]
            validation_data = precomputed[transfer[1]]
            discovery_support = discovery_weights @ discovery_data["eligible"]
            discovery_absolute_sum = (
                discovery_weights @ discovery_data["absolute_eligible_correct"]
            )
            discovery_residual_sum = (
                discovery_weights @ discovery_data["residual_eligible_correct"]
            )
            discovery_absolute_accuracy = np.divide(
                discovery_absolute_sum,
                discovery_support,
                out=np.full(len(discovery_support), np.nan),
                where=discovery_support > 0,
            )
            discovery_residual_accuracy = np.divide(
                discovery_residual_sum,
                discovery_support,
                out=np.full(len(discovery_support), np.nan),
                where=discovery_support > 0,
            )
            validation_support = validation_weights @ validation_data["eligible"]
            validation_absolute_sum = (
                validation_weights @ validation_data["absolute_eligible_correct"]
            )
            validation_residual_sum = (
                validation_weights @ validation_data["residual_eligible_correct"]
            )
            original_mask = np.asarray(cache["mask"], dtype=bool)
            discovery_denominator_by_ligand = (
                np.asarray(discovery_data["eligible"])[:, original_mask].sum(axis=1)
            )
            discovery_mix_numerator_by_alpha_ligand = np.asarray(
                discovery_data["mix_eligible_correct"]
            )[:, :, original_mask].sum(axis=2)
            denominator = float(
                discovery_weights @ discovery_denominator_by_ligand
            )
            alpha_accuracies = (
                discovery_mix_numerator_by_alpha_ligand @ discovery_weights
            ) / denominator
            selected_alpha_index = int(np.flatnonzero(
                alpha_accuracies == np.max(alpha_accuracies)
            )[0])
            mask = (
                original_mask
                & (discovery_support > 0)
                & (validation_support > 0)
            )
            choice = discovery_residual_accuracy > discovery_absolute_accuracy
            rule_correct_sum = np.where(
                choice, validation_residual_sum, validation_absolute_sum
            )
            rule_accuracy = float(
                np.sum(rule_correct_sum[mask]) / np.sum(validation_support[mask])
            )
            validation_mix_correct = np.asarray(
                validation_data["mix_eligible_correct"]
            )[selected_alpha_index]
            validation_mix_accuracy = float(
                validation_weights
                @ validation_mix_correct[:, mask].sum(axis=1)
                / np.sum(validation_support[mask])
            )
            gain = rule_accuracy - validation_mix_accuracy
            gains.append(gain)
            rows.append(
                {
                    "repetition": repetition,
                    "discovery": transfer[0],
                    "validation": transfer[1],
                    "rule_minus_global_alpha_mixture": gain,
                }
            )
        omnibus[repetition] = float(np.mean(gains))
        rows.append(
            {
                "repetition": repetition,
                "discovery": "locked_exact_mean",
                "validation": "locked_exact_mean",
                "rule_minus_global_alpha_mixture": omnibus[repetition],
            }
        )
    summary = {
        "repetitions": int(repetitions),
        "seed": int(seed),
        "grouping": "RDKit Morgan radius 2, 2048 bits; Butina Tanimoto >=0.65",
        "resampling_scope": (
            "whole chemotype clusters in both discovery and validation; "
            "pair rule and global alpha are refit in every draw"
        ),
        "validation_clusters": {
            name: int(len(np.unique(labels[name]))) for name in {item[1] for item in transfers}
        },
        "locked_mean": float(np.mean(omnibus)),
        "locked_median": float(np.median(omnibus)),
        "locked_interval_95": [
            float(np.quantile(omnibus, 0.025)),
            float(np.quantile(omnibus, 0.975)),
        ],
        "positive_fraction": float(np.mean(omnibus > 0)),
    }
    return pd.DataFrame.from_records(rows), summary


def shared_ligand_exclusion(
    panels: OrderedDict[str, Panel], minimum_support: int
) -> pd.DataFrame:
    first, second = panels["PKIS2"], panels["DAVIS"]
    first_rows = {row for group in first.dockstring_rows for row in group}
    second_rows = {row for group in second.dockstring_rows for row in group}
    shared = first_rows & second_rows
    rows: list[dict[str, Any]] = []
    filtered: dict[str, Panel] = {}
    for panel in (first, second):
        keep = np.asarray(
            [not bool(set(group) & shared) for group in panel.dockstring_rows]
        )
        filtered[panel.name] = Panel(
            name=panel.name,
            experiment=panel.experiment[keep],
            docking=panel.docking[keep],
            absolute=panel.absolute[keep],
            residual=panel.residual[keep],
            smiles=tuple(np.asarray(panel.smiles, object)[keep].astype(str)),
            dockstring_rows=tuple(
                group for group, retained in zip(panel.dockstring_rows, keep) if retained
            ),
            margin=panel.margin,
            match=panel.match,
            pooled_absolute_sd=panel.pooled_absolute_sd,
        )
    for discovery, validation in PRIMARY_TRANSFERS:
        result, _ = transfer_metrics(
            filtered[discovery], filtered[validation], minimum_support
        )
        rows.append(
            {
                "discovery": discovery,
                "validation": validation,
                "shared_dockstring_rows": int(len(shared)),
                "discovery_ligands_after_exclusion": int(
                    len(filtered[discovery].experiment)
                ),
                "validation_ligands_after_exclusion": int(
                    len(filtered[validation].experiment)
                ),
                "rule_minus_absolute": result["pair_weighted_rule_minus_absolute"],
                "rule_minus_residual": result["pair_weighted_rule_minus_residual"],
                "rule_minus_global_alpha_mixture": result[
                    "pair_weighted_rule_minus_global_alpha_mixture"
                ],
                "normalization_gain_spearman": result[
                    "normalization_gain_spearman"
                ],
            }
        )
    return pd.DataFrame.from_records(rows)


def pair_metric_frame(panels: OrderedDict[str, Panel]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for name, panel in panels.items():
        cells = pair_cells(
            panel.absolute, panel.residual, panel.experiment, panel.margin
        )
        for index, (first, second) in enumerate(zip(*TRI)):
            rows.append(
                {
                    "panel": name,
                    "match": panel.match,
                    "target_a": TARGETS[first],
                    "target_b": TARGETS[second],
                    "informative_ligands": int(cells.support[index]),
                    "absolute_accuracy": float(cells.absolute_accuracy[index]),
                    "residual_accuracy": float(cells.residual_accuracy[index]),
                    "residual_minus_absolute": float(
                        cells.residual_accuracy[index]
                        - cells.absolute_accuracy[index]
                    ),
                }
            )
    return pd.DataFrame.from_records(rows)


def write_readme(output_dir: Path, summary: dict[str, Any]) -> None:
    primary = summary["primary_transfers"]
    lines = []
    for item in primary:
        lines.append(
            f"- **{item['discovery']} → {item['validation']}:** pair-rule "
            f"{item['pair_weighted_transferred_pair_rule_accuracy']:.3f}; "
            f"absolute {item['pair_weighted_absolute_accuracy']:.3f}; residual "
            f"{item['pair_weighted_residual_accuracy']:.3f}; target-mean prior "
            f"{item['pair_weighted_discovery_experimental_target_mean_prior_accuracy']:.3f}."
        )
    qap = summary["primary_joint_qap"]
    bootstrap = summary["primary_cluster_bootstrap"]
    jackknife = summary["primary_target_jackknife"]
    text = f"""# Target-pair normalization transfer: an operational no-go

This science-only analysis tests whether the average effect of target-specific
Vina scaling hides reproducible target-pair regimes.  A discovery panel learns,
for each fixed target pair, whether absolute or two-way-residual Vina gives
higher experimental pairwise-preference concordance.  That lookup table is
then frozen and applied to different compounds in another assay panel.

## Primary exact-full-key result and decisive baseline

{chr(10).join(lines)}

The normalization-choice pattern is statistically aligned: its locked mean
gain over the discovery-selected coherent global Vina mixture is
**{qap['observed_mean_rule_minus_global_alpha_mixture']:+.4f}**.  A joint
target-label QAP gives **p={qap['joint_target_label_qap_p_one_sided']:.5f}**.
The bootstrap independently resamples whole chemotype clusters in discovery
and validation and refits the rule and global alpha in every draw; its interval is
**[{bootstrap['locked_interval_95'][0]:+.4f}, {bootstrap['locked_interval_95'][1]:+.4f}]**.
The leave-one-target-out mean gain is positive in
**{jackknife['positive_deletions']}/{jackknife['deletions']}** deletions, with
range **[{jackknife['leave_one_target_out_minimum']:+.4f},
{jackknife['leave_one_target_out_maximum']:+.4f}]**.

That statistical pattern does **not** supply an operational advance.  A
ligand-independent baseline transferring only the discovery panel's target
means reaches **{primary[0]['pair_weighted_discovery_experimental_target_mean_prior_accuracy']:.3f}**
for PKIS2 → DAVIS and **{primary[1]['pair_weighted_discovery_experimental_target_mean_prior_accuracy']:.3f}**
for DAVIS → PKIS2, far above the pair rule.  On cells where this target-mean
prior is wrong, the transferred pair rule reaches only
**{primary[0]['prior_reversal_transferred_pair_rule_accuracy']:.3f}** and
**{primary[1]['prior_reversal_transferred_pair_rule_accuracy']:.3f}**, versus
absolute Vina **{primary[0]['prior_reversal_absolute_accuracy']:.3f}** and
**{primary[1]['prior_reversal_absolute_accuracy']:.3f}**.  Thus the apparent
transfer mainly reinforces stable target marginals rather than recovering the
ligand-specific exceptions that matter for selectivity.

PKIS2 and DAVIS use the already frozen exact-full-Standard-InChIKey mappings.
Only two complete-support DOCKSTRING rows overlap the two panels; excluding
every profile touching either row is reported separately.  PKIS1 is not part of
the primary claim because it has no exact-full-key overlap and is included only
as a 220-connectivity-block sensitivity.

## Decision and interpretation boundary

**Operational decision: NO-GO.**  This result must not be promoted as an
improved target-ranking or counter-screen method.  Its defensible use is a
mechanistic diagnosis: the effect of target scaling is reproducibly
pair-specific, but much of the transfer reflects stable target-level assay or
library marginals.

The transferred object is a lookup table for **the same fixed target pairs**.
It learns pair identity from discovery experimental outcomes but never uses a
validation ligand outcome.  Pairwise choices can form cycles, so the rule does
**not** define a coherent global target ranking.  More importantly, it fails
the target-mean baseline and the prior-reversal stress test.

All analyses are post hoc/exploratory because the resources had been inspected
elsewhere in the project.  Target-label QAP tests alignment rather than
prospective confirmation; the target jackknife remains conditional on this
finite 20-kinase family.

## Reproduction

```bash
.venv/bin/python analysis/pair_specific_normalization_transfer.py \\
  --pkis1-zip /tmp/pkis1_supplement.zip \\
  --output results/pair_specific_normalization_transfer \\
  --qap-permutations {summary['qap_permutations']} \\
  --bootstraps {summary['chemotype_bootstraps']} --seed {summary['seed']}
```
"""
    (output_dir / "README.md").write_text(text, encoding="utf-8")


def run_analysis(
    pkis1_zip: Path,
    output_dir: Path,
    qap_permutations: int,
    bootstraps: int,
    seed: int,
) -> dict[str, Any]:
    if qap_permutations < 1 or bootstraps < 1:
        raise ValueError("permutation and bootstrap counts must be positive")
    panels = load_panels(pkis1_zip)
    pair_metrics = pair_metric_frame(panels)
    transfer_rows: list[dict[str, Any]] = []
    support_rows: list[dict[str, Any]] = []
    caches: dict[tuple[str, str], dict[str, np.ndarray | float]] = {}
    primary_metrics: dict[tuple[str, str], dict[str, Any]] = {}
    alpha_rows: list[pd.DataFrame] = []
    for minimum in SUPPORT_GRID:
        for transfer in (*PRIMARY_TRANSFERS, *SENSITIVITY_TRANSFERS):
            metrics, cache = transfer_metrics(
                panels[transfer[0]], panels[transfer[1]], minimum
            )
            support_rows.append(metrics)
            if minimum == PRIMARY_MINIMUM_PAIR_SUPPORT:
                transfer_rows.append(metrics)
                caches[transfer] = cache
                if transfer in PRIMARY_TRANSFERS:
                    primary_metrics[transfer] = metrics
                alpha_frame = pd.DataFrame(cache["alpha_frame"]).copy()
                alpha_frame.insert(0, "validation", transfer[1])
                alpha_frame.insert(0, "discovery", transfer[0])
                alpha_rows.append(alpha_frame)
    qap_frame, qap_null, qap_summary = qap_transfer_null(
        caches,
        PRIMARY_TRANSFERS,
        qap_permutations,
        seed,
    )
    jackknife_frame, jackknife_summary = target_jackknife(
        panels, primary_metrics, caches, PRIMARY_TRANSFERS
    )
    bootstrap_frame, bootstrap_summary = cluster_bootstrap(
        panels, caches, PRIMARY_TRANSFERS, bootstraps, seed + 100_000
    )
    shared = shared_ligand_exclusion(panels, PRIMARY_MINIMUM_PAIR_SUPPORT)

    output_dir.mkdir(parents=True, exist_ok=True)
    pair_metrics.to_csv(output_dir / "panel_pair_metrics.csv", index=False)
    pd.DataFrame.from_records(transfer_rows).to_csv(
        output_dir / "ordered_transfer_metrics.csv", index=False
    )
    pd.DataFrame.from_records(support_rows).to_csv(
        output_dir / "minimum_support_sensitivity.csv", index=False
    )
    pd.concat(alpha_rows, ignore_index=True).to_csv(
        output_dir / "global_alpha_selection.csv", index=False
    )
    qap_frame.to_csv(output_dir / "target_label_qap_summary.csv", index=False)
    pd.DataFrame(
        {
            "repetition": np.arange(qap_permutations),
            **{
                f"{first}_to_{second}": qap_null[:, index]
                for index, (first, second) in enumerate(PRIMARY_TRANSFERS)
            },
            "locked_exact_mean": qap_null.mean(axis=1),
        }
    ).to_csv(output_dir / "target_label_qap_draws.csv.gz", index=False)
    jackknife_frame.to_csv(output_dir / "target_jackknife.csv", index=False)
    bootstrap_frame.to_csv(
        output_dir / "chemotype_bootstrap_draws.csv.gz", index=False
    )
    shared.to_csv(output_dir / "shared_ligand_exclusion.csv", index=False)

    summary: dict[str, Any] = {
        "analysis": "pair_specific_normalization_transfer",
        "status": "post_hoc_exploratory",
        "operational_decision": "NO_GO",
        "decision_reason": (
            "the transferred pair rule is dominated by a ligand-independent "
            "discovery experimental target-mean prior and degrades on prior-reversal cells"
        ),
        "fixed_targets": list(TARGETS),
        "primary_minimum_pair_support": PRIMARY_MINIMUM_PAIR_SUPPORT,
        "support_grid": list(SUPPORT_GRID),
        "alpha_grid": list(ALPHA_GRID),
        "primary_transfers": [primary_metrics[item] for item in PRIMARY_TRANSFERS],
        "connectivity_sensitivity_transfers": [
            next(
                row
                for row in transfer_rows
                if row["discovery"] == item[0] and row["validation"] == item[1]
            )
            for item in SENSITIVITY_TRANSFERS
        ],
        "primary_joint_qap": qap_summary,
        "primary_cluster_bootstrap": bootstrap_summary,
        "primary_target_jackknife": jackknife_summary,
        "shared_ligand_exclusion": shared.to_dict(orient="records"),
        "panels": {
            name: {
                "ligands": int(len(panel.experiment)),
                "targets": int(panel.experiment.shape[1]),
                "match": panel.match,
                "experimental_margin": panel.margin,
                "butina_clusters": int(
                    len(
                        np.unique(
                            davis._butina_labels(  # noqa: SLF001
                                pd.Series(panel.smiles), similarity_threshold=0.65
                            )
                        )
                    )
                ),
            }
            for name, panel in panels.items()
        },
        "claim_boundary": {
            "learned_object": "target-pair identity lookup on the fixed 20-target panel",
            "validation_outcomes_used_in_rule_selection": False,
            "global_ranking_claim": False,
            "reason": "pair-specific representation choices need not be transitive",
            "prospective_confirmation": False,
            "operational_improvement_claim": False,
        },
        "source_sha256": {
            "dockstring": sha256_file(davis.DEFAULT_DOCKSTRING),
            "davis": sha256_file(davis.DEFAULT_DAVIS),
            "pkis2": sha256_file(pkis2_benchmark.DEFAULT_PKIS2),
            "pkis1_zip": sha256_file(pkis1_zip),
            "pkis2_mapping": sha256_file(Path(PANEL_SPECS["PKIS2"]["mapping"])),
            "davis_mapping": sha256_file(Path(PANEL_SPECS["DAVIS"]["mapping"])),
        },
        "qap_permutations": int(qap_permutations),
        "chemotype_bootstraps": int(bootstraps),
        "seed": int(seed),
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(json_ready(summary), handle, indent=2, sort_keys=True)
        handle.write("\n")
    write_readme(output_dir, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pkis1-zip", type=Path, default=DEFAULT_PKIS1_ZIP)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--qap-permutations", type=int, default=DEFAULT_QAP_PERMUTATIONS
    )
    parser.add_argument("--bootstraps", type=int, default=DEFAULT_BOOTSTRAPS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    summary = run_analysis(
        args.pkis1_zip,
        args.output,
        args.qap_permutations,
        args.bootstraps,
        args.seed,
    )
    primary = summary["primary_joint_qap"]
    print(
        "locked mean rule-minus-global-mixture "
        f"{primary['observed_mean_rule_minus_global_alpha_mixture']:+.4f}; "
        f"QAP p={primary['joint_target_label_qap_p_one_sided']:.5f}"
    )


if __name__ == "__main__":
    main()
