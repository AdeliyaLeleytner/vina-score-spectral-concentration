#!/usr/bin/env python3
"""Falsification study for spectral expansion as a docking-matrix QC criterion.

The simulation deliberately separates three questions that are otherwise easy to
conflate:

1. Does removal of a ligand main effect expand the correlation spectrum?
2. Is the residual surface non-independent and reproducible across docking seeds?
3. Does the residual surface recover the (simulated) ligand--target truth?

Only the third question is accuracy.  The first two are observable without an
experimental reference and are therefore candidates for matrix-level QC.  The
simulated truth is used solely to expose false positives and false negatives of
those observable diagnostics; it is never used to set a decision threshold.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


PRIMARY_RULE = {
    "name": "reproducible_correlated_residual_gate",
    "minimum_median_pairwise_residual_seed_spearman": 0.70,
    "maximum_residual_pr_over_row_norm_null_median": 0.80,
    "maximum_empirical_lower_tail_p": 0.05,
    "logic": "all three conditions must hold",
    "scope": (
        "A matrix-level coherence screen, not an affinity-accuracy, target-retrieval, "
        "pose-validity, receptor-quality, or box-validity criterion."
    ),
    "threshold_provenance": (
        "Fixed before inspecting simulation outcomes: rho>=0.70 requires substantial "
        "seed reproducibility; a null ratio<=0.80 requires at least a 20% PR deficit; "
        "and p<=0.05 requires lower-tail separation from the row-norm null."
    ),
}

EXPANSION_ONLY_RULE = {
    "name": "dimension_doubling_after_two_way_centering",
    "minimum_expansion_log2": 1.0,
    "logic": "E=log2(PR_residual/PR_raw) >= 1",
    "scope": "A deliberately weak comparator, not a recommended rule.",
}

REGIME_METADATA: dict[str, dict[str, Any]] = {
    "good_correlated_interactions": {
        "truth_informative": True,
        "description": (
            "A strong ligand main effect plus reproducible, family-correlated "
            "ligand--target interactions and small seed noise."
        ),
    },
    "generic_ligand_effect_independent_noise": {
        "truth_informative": False,
        "description": (
            "A strong ligand main effect plus a reproducible but target-independent "
            "cell artifact; this is the canonical expansion-without-information case."
        ),
    },
    "target_correlated_artifact": {
        "truth_informative": False,
        "description": (
            "A reproducible low-dimensional target-correlated artifact independent of "
            "the ligand--target truth."
        ),
    },
    "shuffled_box_labels": {
        "truth_informative": False,
        "description": (
            "The complete good score surface, including seed noise, is globally "
            "permuted across target labels, mimicking a receptor/box-label mismatch."
        ),
    },
    "seed_unstable_noise": {
        "truth_informative": False,
        "description": (
            "A strong ligand main effect with independently regenerated cell noise for "
            "every seed and no reproducible residual surface."
        ),
    },
    "good_high_dimensional_interactions": {
        "truth_informative": True,
        "description": (
            "Accurate, reproducible target-specific interactions with approximately "
            "independent target directions; a deliberate false-negative challenge for "
            "any criterion that requires residual spectral concentration."
        ),
    },
}


@dataclass(frozen=True)
class SimulationConfig:
    n_ligands: int = 480
    n_targets: int = 20
    n_seeds: int = 3
    n_target_families: int = 4
    replicates: int = 160
    null_draws: int = 39
    seed: int = 20260802
    ligand_main_sd: float = 4.5
    target_offset_sd: float = 0.12
    stable_seed_noise_sd: float = 0.25
    unstable_seed_noise_sd: float = 1.25

    def validate(self) -> None:
        if self.n_ligands < 30:
            raise ValueError("n_ligands must be at least 30")
        if self.n_targets < 6:
            raise ValueError("n_targets must be at least 6")
        if self.n_seeds < 2:
            raise ValueError("n_seeds must be at least 2")
        if not 2 <= self.n_target_families <= self.n_targets:
            raise ValueError("n_target_families must lie between 2 and n_targets")
        if self.replicates < 1:
            raise ValueError("replicates must be positive")
        if self.null_draws < 19:
            raise ValueError(
                "null_draws must be >=19 so a finite-sample empirical p-value can reach 0.05"
            )


def two_way_center(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    return (
        matrix
        - matrix.mean(axis=0, keepdims=True)
        - matrix.mean(axis=1, keepdims=True)
        + matrix.mean()
    )


def _unit_sd(matrix: np.ndarray) -> np.ndarray:
    centered = two_way_center(matrix)
    scale = float(centered.std(ddof=1))
    if not np.isfinite(scale) or scale <= 1e-12:
        raise ValueError("cannot scale a constant simulated surface")
    return centered / scale


def correlation_pr(matrix: np.ndarray) -> float:
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] < 3 or matrix.shape[1] < 2:
        raise ValueError("matrix must have >=3 rows and >=2 columns")
    if not np.isfinite(matrix).all():
        raise ValueError("matrix has non-finite values")
    standard_deviation = matrix.std(axis=0, ddof=1)
    if np.any(standard_deviation <= 1e-12):
        raise ValueError("matrix has a constant target column")
    standardized = (
        matrix - matrix.mean(axis=0, keepdims=True)
    ) / standard_deviation
    correlation = (standardized.T @ standardized) / (len(standardized) - 1)
    correlation = (correlation + correlation.T) / 2
    np.fill_diagonal(correlation, 1.0)
    return float(np.square(np.trace(correlation)) / np.square(correlation).sum())


def spectral_metrics(matrix: np.ndarray) -> dict[str, float]:
    raw_pr = correlation_pr(matrix)
    residual_pr = correlation_pr(two_way_center(matrix))
    return {
        "raw_pr": raw_pr,
        "residual_pr": residual_pr,
        "expansion_log2": float(np.log2(residual_pr / raw_pr)),
    }


def median_pairwise_seed_spearman(seed_surfaces: np.ndarray) -> dict[str, float]:
    """Return raw and residual seed stability without pseudoreplicating cells."""
    raw_values: list[float] = []
    residual_values: list[float] = []
    for first in range(len(seed_surfaces)):
        for second in range(first + 1, len(seed_surfaces)):
            raw_values.append(
                float(
                    spearmanr(
                        seed_surfaces[first].ravel(),
                        seed_surfaces[second].ravel(),
                    ).statistic
                )
            )
            residual_values.append(
                float(
                    spearmanr(
                        two_way_center(seed_surfaces[first]).ravel(),
                        two_way_center(seed_surfaces[second]).ravel(),
                    ).statistic
                )
            )
    return {
        "raw_seed_spearman": float(np.median(raw_values)),
        "residual_seed_spearman": float(np.median(residual_values)),
    }


def row_norm_null(
    residual: np.ndarray,
    common_random_directions: np.ndarray,
) -> np.ndarray:
    """Row-norm-preserving null PR values in the target-zero-sum subspace."""
    residual = two_way_center(residual)
    observed_norms = np.linalg.norm(residual, axis=1)
    values: list[float] = []
    for draw in common_random_directions:
        directions = draw - draw.mean(axis=1, keepdims=True)
        generated_norms = np.linalg.norm(directions, axis=1)
        simulated = directions * np.divide(
            observed_norms,
            generated_norms,
            out=np.zeros_like(observed_norms),
            where=generated_norms > 0,
        )[:, None]
        values.append(correlation_pr(two_way_center(simulated)))
    return np.asarray(values, dtype=np.float64)


def preference_accuracy(score: np.ndarray, truth: np.ndarray) -> float:
    """Within-ligand pairwise target-preference concordance; higher is better."""
    upper = np.triu_indices(score.shape[1], 1)
    score_difference = score[:, upper[0]] - score[:, upper[1]]
    truth_difference = truth[:, upper[0]] - truth[:, upper[1]]
    valid = (score_difference != 0) & (truth_difference != 0)
    return float(
        np.mean(np.sign(score_difference[valid]) == np.sign(truth_difference[valid]))
    )


def residual_truth_spearman(score: np.ndarray, truth: np.ndarray) -> float:
    return float(
        spearmanr(two_way_center(score).ravel(), two_way_center(truth).ravel()).statistic
    )


def _structured_surface(
    rng: np.random.Generator,
    n_ligands: int,
    n_targets: int,
    n_families: int,
) -> np.ndarray:
    family = np.arange(n_targets) % n_families
    loadings = np.eye(n_families, dtype=float)[family]
    loadings += rng.normal(0, 0.18, size=loadings.shape)
    ligand_factors = rng.normal(size=(n_ligands, n_families))
    idiosyncratic = rng.normal(size=(n_ligands, n_targets))
    surface = ligand_factors @ loadings.T + 0.30 * idiosyncratic
    return _unit_sd(surface)


def _derangement(rng: np.random.Generator, size: int) -> np.ndarray:
    original = np.arange(size)
    for _ in range(1000):
        candidate = rng.permutation(size)
        if np.all(candidate != original):
            return candidate
    # A cyclic shift is always a derangement for size > 1.
    return np.roll(original, 1)


def simulate_regimes(
    config: SimulationConfig,
    rng: np.random.Generator,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray]:
    """Create paired regimes from the same latent ligand and target support."""
    n, p, s = config.n_ligands, config.n_targets, config.n_seeds
    correlated_truth = _structured_surface(
        rng, n, p, config.n_target_families
    )
    independent_truth = _unit_sd(rng.normal(size=(n, p)))
    correlated_artifact = _structured_surface(
        rng, n, p, config.n_target_families
    )
    fixed_independent_artifact = _unit_sd(rng.normal(size=(n, p)))
    ligand_main = rng.normal(0, config.ligand_main_sd, size=(n, 1))
    target_offset = rng.normal(0, config.target_offset_sd, size=(1, p))

    def stable_seeds(base_residual: np.ndarray) -> np.ndarray:
        noise = rng.normal(
            0, config.stable_seed_noise_sd, size=(s, n, p)
        )
        return ligand_main[None, :, :] + target_offset[None, :, :] + base_residual + noise

    good = stable_seeds(correlated_truth)
    permutation = _derangement(rng, p)
    regimes = {
        "good_correlated_interactions": good,
        "generic_ligand_effect_independent_noise": stable_seeds(
            fixed_independent_artifact
        ),
        "target_correlated_artifact": stable_seeds(correlated_artifact),
        # Permuting every seed identically makes the spectral invariance exact.
        "shuffled_box_labels": good[:, :, permutation],
        "seed_unstable_noise": (
            ligand_main[None, :, :]
            + target_offset[None, :, :]
            + rng.normal(0, config.unstable_seed_noise_sd, size=(s, n, p))
        ),
        "good_high_dimensional_interactions": stable_seeds(independent_truth),
    }
    truths = {
        "good_correlated_interactions": correlated_truth,
        "generic_ligand_effect_independent_noise": correlated_truth,
        "target_correlated_artifact": correlated_truth,
        "shuffled_box_labels": correlated_truth,
        "seed_unstable_noise": correlated_truth,
        "good_high_dimensional_interactions": independent_truth,
    }
    return regimes, truths, permutation


def apply_rules(metrics: dict[str, float]) -> dict[str, float | bool]:
    expansion_pass = (
        metrics["expansion_log2"]
        >= EXPANSION_ONLY_RULE["minimum_expansion_log2"]
    )
    reproducible = (
        metrics["residual_seed_spearman"]
        >= PRIMARY_RULE["minimum_median_pairwise_residual_seed_spearman"]
    )
    nonindependent = (
        metrics["residual_pr_null_ratio"]
        <= PRIMARY_RULE["maximum_residual_pr_over_row_norm_null_median"]
        and metrics["residual_pr_null_lower_tail_p"]
        <= PRIMARY_RULE["maximum_empirical_lower_tail_p"]
    )
    coherence_pass = reproducible and nonindependent
    structure_strength = max(
        0.0, -float(np.log2(metrics["residual_pr_null_ratio"]))
    )
    composite_score = (
        float(np.clip(metrics["residual_seed_spearman"], 0.0, 1.0))
        * structure_strength
    )
    return {
        "expansion_only_pass": bool(expansion_pass),
        "reproducibility_axis_pass": bool(reproducible),
        "residual_structure_axis_pass": bool(nonindependent),
        "coherence_gate_pass": bool(coherence_pass),
        "coherence_composite_score": float(composite_score),
    }


def evaluate_regime(
    seed_surfaces: np.ndarray,
    truth: np.ndarray,
    common_random_directions: np.ndarray,
) -> dict[str, float | bool]:
    aggregate = np.median(seed_surfaces, axis=0)
    metrics: dict[str, float] = {
        **spectral_metrics(aggregate),
        **median_pairwise_seed_spearman(seed_surfaces),
        "preference_accuracy": preference_accuracy(aggregate, truth),
        "residual_truth_spearman": residual_truth_spearman(aggregate, truth),
    }
    null_values = row_norm_null(two_way_center(aggregate), common_random_directions)
    metrics.update(
        {
            "residual_pr_null_median": float(np.median(null_values)),
            "residual_pr_null_ratio": float(
                metrics["residual_pr"] / np.median(null_values)
            ),
            "residual_pr_null_lower_tail_p": float(
                (1 + np.sum(null_values <= metrics["residual_pr"]))
                / (len(null_values) + 1)
            ),
        }
    )
    return {**metrics, **apply_rules(metrics)}


SUMMARY_METRICS = [
    "raw_pr",
    "residual_pr",
    "expansion_log2",
    "residual_pr_null_ratio",
    "residual_pr_null_lower_tail_p",
    "raw_seed_spearman",
    "residual_seed_spearman",
    "coherence_composite_score",
    "preference_accuracy",
    "residual_truth_spearman",
]


def summarize_replicates(replicates: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for regime, frame in replicates.groupby("regime", sort=False):
        record: dict[str, Any] = {
            "regime": regime,
            "truth_informative": REGIME_METADATA[regime]["truth_informative"],
            "n_replicates": int(len(frame)),
        }
        for metric in SUMMARY_METRICS:
            values = frame[metric].to_numpy(dtype=float)
            record[f"{metric}_median"] = float(np.median(values))
            record[f"{metric}_q025"] = float(np.quantile(values, 0.025))
            record[f"{metric}_q975"] = float(np.quantile(values, 0.975))
        for metric in [
            "expansion_only_pass",
            "reproducibility_axis_pass",
            "residual_structure_axis_pass",
            "coherence_gate_pass",
        ]:
            record[f"{metric}_rate"] = float(frame[metric].mean())
        records.append(record)
    return pd.DataFrame(records)


def threshold_sensitivity(replicates: pd.DataFrame) -> pd.DataFrame:
    """Report a fixed grid; it is descriptive and never selects the primary cutoffs."""
    records: list[dict[str, Any]] = []
    for rho_threshold in [0.60, 0.70, 0.80]:
        for ratio_threshold in [0.70, 0.80, 0.90]:
            passes = (
                replicates["residual_seed_spearman"].ge(rho_threshold)
                & replicates["residual_pr_null_ratio"].le(ratio_threshold)
                & replicates["residual_pr_null_lower_tail_p"].le(0.05)
            )
            for regime, index in replicates.groupby("regime", sort=False).groups.items():
                records.append(
                    {
                        "minimum_residual_seed_spearman": rho_threshold,
                        "maximum_residual_pr_null_ratio": ratio_threshold,
                        "maximum_null_lower_tail_p": 0.05,
                        "regime": regime,
                        "pass_rate": float(passes.loc[index].mean()),
                        "is_primary_rule": bool(
                            rho_threshold
                            == PRIMARY_RULE[
                                "minimum_median_pairwise_residual_seed_spearman"
                            ]
                            and ratio_threshold
                            == PRIMARY_RULE[
                                "maximum_residual_pr_over_row_norm_null_median"
                            ]
                        ),
                    }
                )
    return pd.DataFrame(records)


def falsification_summary(replicates: pd.DataFrame) -> dict[str, Any]:
    indexed = replicates.set_index(["replicate", "regime"])
    good = indexed.xs("good_correlated_interactions", level="regime")
    shuffled = indexed.xs("shuffled_box_labels", level="regime")
    generic = indexed.xs(
        "generic_ligand_effect_independent_noise", level="regime"
    )
    artifact = indexed.xs("target_correlated_artifact", level="regime")
    highdim = indexed.xs("good_high_dimensional_interactions", level="regime")
    invariant_metrics = [
        "raw_pr",
        "residual_pr",
        "expansion_log2",
        "raw_seed_spearman",
        "residual_seed_spearman",
        "residual_pr_null_ratio",
        "coherence_composite_score",
    ]

    def binary_performance(column: str) -> dict[str, float]:
        truth = replicates["truth_informative"].astype(bool)
        predicted = replicates[column].astype(bool)
        sensitivity = float(predicted.loc[truth].mean())
        false_positive_rate = float(predicted.loc[~truth].mean())
        return {
            "sensitivity_across_the_two_informative_regimes": sensitivity,
            "specificity_across_the_four_noninformative_regimes": 1.0
            - false_positive_rate,
            "false_positive_rate": false_positive_rate,
            "balanced_accuracy": float(
                (sensitivity + 1.0 - false_positive_rate) / 2
            ),
            "warning": (
                "These rates summarize deliberately selected falsification regimes, "
                "not a prevalence-weighted estimate for real docking campaigns."
            ),
        }

    return {
        "claim_boundary": (
            "No statistic computed only from a score matrix and replicate seeds can "
            "distinguish the good surface from an identical surface assigned to the "
            "wrong target labels. The proposed gate can establish reproducible, "
            "non-independent residual structure, not docking accuracy."
        ),
        "counterexamples": {
            "expansion_without_information": {
                "regime": "generic_ligand_effect_independent_noise",
                "fraction_with_E_at_least_one": float(
                    generic["expansion_only_pass"].mean()
                ),
                "median_preference_accuracy": float(
                    generic["preference_accuracy"].median()
                ),
                "fraction_E_exceeding_good_correlated": float(
                    (generic["expansion_log2"] > good["expansion_log2"]).mean()
                ),
            },
            "reproducible_correlated_artifact": {
                "regime": "target_correlated_artifact",
                "coherence_gate_pass_rate": float(
                    artifact["coherence_gate_pass"].mean()
                ),
                "median_preference_accuracy": float(
                    artifact["preference_accuracy"].median()
                ),
            },
            "target_label_permutation_invariance": {
                "regime": "shuffled_box_labels",
                "maximum_absolute_observable_metric_difference_from_good": {
                    metric: float(np.max(np.abs(good[metric] - shuffled[metric])))
                    for metric in invariant_metrics
                },
                "median_good_preference_accuracy": float(
                    good["preference_accuracy"].median()
                ),
                "median_shuffled_preference_accuracy": float(
                    shuffled["preference_accuracy"].median()
                ),
                "coherence_gate_pass_rate": float(
                    shuffled["coherence_gate_pass"].mean()
                ),
            },
            "accurate_high_dimensional_false_negative": {
                "regime": "good_high_dimensional_interactions",
                "coherence_gate_pass_rate": float(
                    highdim["coherence_gate_pass"].mean()
                ),
                "median_preference_accuracy": float(
                    highdim["preference_accuracy"].median()
                ),
                "median_residual_pr_null_ratio": float(
                    highdim["residual_pr_null_ratio"].median()
                ),
            },
        },
        "rule_comparison_against_simulated_truth": {
            "expansion_only": binary_performance("expansion_only_pass"),
            "two_axis_coherence_gate": binary_performance("coherence_gate_pass"),
        },
        "recommended_interpretation": {
            "E": "Magnitude of ligand-main-effect removal relative to residual dimension.",
            "residual_pr_null_ratio": (
                "Evidence for correlated residual directions relative to a row-norm null; "
                "lower is more structured, not necessarily more correct."
            ),
            "residual_seed_spearman": (
                "Technical reproducibility of the residual score field; raw seed "
                "stability can be inflated by the ligand main effect."
            ),
            "coherence_gate": (
                "Use only to flag reproducible correlated residual structure. Require "
                "external pose/site controls or affinity/preference truth for quality."
            ),
            "external_evidence_required_for_accuracy": [
                "box containment and intended-site identity",
                "cognate-pose recovery or pocket-occupancy controls",
                "receptor/box label integrity",
                "experimental affinity or target-preference concordance",
            ],
        },
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_simulation(config: SimulationConfig, output_dir: Path) -> dict[str, Any]:
    config.validate()
    output_dir.mkdir(parents=True, exist_ok=True)
    master = np.random.SeedSequence(config.seed)
    replicate_sequences = master.spawn(config.replicates)
    records: list[dict[str, Any]] = []
    permutations: list[dict[str, Any]] = []
    for replicate, sequence in enumerate(replicate_sequences):
        rng = np.random.default_rng(sequence)
        regimes, truths, permutation = simulate_regimes(config, rng)
        common_random_directions = rng.normal(
            size=(config.null_draws, config.n_ligands, config.n_targets)
        )
        permutations.append(
            {
                "replicate": replicate,
                "target_permutation": " ".join(map(str, permutation.tolist())),
                "fixed_points": int(np.sum(permutation == np.arange(config.n_targets))),
            }
        )
        for regime in REGIME_METADATA:
            metrics = evaluate_regime(
                regimes[regime], truths[regime], common_random_directions
            )
            records.append(
                {
                    "replicate": replicate,
                    "regime": regime,
                    "truth_informative": REGIME_METADATA[regime]["truth_informative"],
                    **metrics,
                }
            )

    replicate_frame = pd.DataFrame(records)
    summary = summarize_replicates(replicate_frame)
    sensitivity = threshold_sensitivity(replicate_frame)
    falsification = falsification_summary(replicate_frame)

    paths = {
        "replicates": output_dir / "replicate_metrics.csv",
        "summary": output_dir / "regime_summary.csv",
        "threshold_sensitivity": output_dir / "threshold_sensitivity.csv",
        "permutations": output_dir / "shuffled_target_permutations.csv",
        "rules": output_dir / "predeclared_rules.json",
        "falsification": output_dir / "falsification_summary.json",
    }
    replicate_frame.to_csv(paths["replicates"], index=False)
    summary.to_csv(paths["summary"], index=False)
    sensitivity.to_csv(paths["threshold_sensitivity"], index=False)
    pd.DataFrame(permutations).to_csv(paths["permutations"], index=False)
    paths["rules"].write_text(
        json.dumps(
            {
                "primary": PRIMARY_RULE,
                "expansion_only_comparator": EXPANSION_ONLY_RULE,
                "threshold_sensitivity_grid_is_not_used_for_selection": True,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    paths["falsification"].write_text(
        json.dumps(falsification, indent=2, sort_keys=True) + "\n"
    )
    manifest = {
        "analysis": "spectral_qc_criterion_falsification_simulation",
        "config": asdict(config),
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "regimes": REGIME_METADATA,
        "paired_design": (
            "Every replicate shares ligand effects, target support, and latent truth; "
            "the shuffled-label regime is an exact column permutation of the good regime."
        ),
        "outputs": {
            key: {"path": path.name, "sha256": _sha256(path)}
            for key, path in paths.items()
        },
        "interpretation_boundary": falsification["claim_boundary"],
    }
    manifest_path = output_dir / "simulation_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/spectral_qc_criterion_simulation"),
    )
    parser.add_argument("--replicates", type=int, default=160)
    parser.add_argument("--n-ligands", type=int, default=480)
    parser.add_argument("--n-targets", type=int, default=20)
    parser.add_argument("--n-seeds", type=int, default=3)
    parser.add_argument("--null-draws", type=int, default=39)
    parser.add_argument("--seed", type=int, default=20260802)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = SimulationConfig(
        n_ligands=args.n_ligands,
        n_targets=args.n_targets,
        n_seeds=args.n_seeds,
        replicates=args.replicates,
        null_draws=args.null_draws,
        seed=args.seed,
    )
    manifest = run_simulation(config, args.output_dir)
    print(
        f"Wrote {len(manifest['outputs']) + 1} files to {args.output_dir} "
        f"for {config.replicates} paired replicates"
    )


if __name__ == "__main__":
    main()
