#!/usr/bin/env python3
"""Does the shared ligand-wide docking axis survive a change of scoring function?

Every spectral result in this package is computed on AutoDock Vina scores, so a
reader can reasonably ask whether the low participation ratio and the dominant
near-uniform first principal component are properties of *empirical scoring
functions* or artefacts of Vina's particular size penalty and rotatable-bond
correction.  Vinardo, Glide and GOLD weight molecular size differently, and the
normalisation literature this manuscript cites reports that the same ligand-
efficiency correction helps one program and hurts another.

Falsifiable prediction tested here.  Hold the poses fixed -- reuse the released
DOCKSTRING top Vina pose for every ligand--target cell -- and rescore the
identical ligand-by-target block with scoring functions that do not share Vina's
functional form.  If the shared ligand axis is a general property of empirical
scoring functions, then every scorer should show (i) a dominant first principal
component of the target-correlation matrix, (ii) a PC1 loading vector close to
the uniform target direction, (iii) a raw participation ratio well below the
number of targets, and (iv) a participation ratio that *rises* once each
ligand's own mean across targets is removed.  If instead the pattern is a Vina
artefact, at least one non-Vina scorer should break it.

Scorers compared on one identical block
---------------------------------------
========================  ===========================================================
released_vina             AutoDock Vina 1.1.2, the released DOCKSTRING score (reference)
smina_vina                smina's re-implementation of the Vina function, same poses
                          (a same-program control that isolates "different program"
                          from "different scoring function")
vinardo                   Vinardo via smina: drops gauss2, reweights repulsion,
                          redefines the hydrophobic and hydrogen-bond terms
rfscore_v1/v2/v3          RF-Score random forests over element-pair close-contact
                          counts (v3 adds Vina terms as extra features)
nnscore                   NNScore 2.0, neural-network ensemble over BINANA descriptors
plecscore_linear          PLECscore, pretrained linear model on protein-ligand
                          extended-connectivity fingerprints
========================  ===========================================================

Interpreter split.  ODDT 0.7 (RF-Score, NNScore, PLECscore) predates Python 3.12
and lives only in the project's ``toxic_agent_conda_env`` (CPython 3.9).  That
stage therefore runs ``analysis/nonvina_scorer_oddt_worker.py`` as a subprocess
under the conda interpreter and writes a CSV; every spectral statistic in this
file is then computed with the release ``.venv`` interpreter.  Both interpreter
versions are recorded in the emitted summary.

What this cannot show.  These are rescorings of *retained Vina poses*.  A scorer
that would have selected a different pose during search is not simulated, so
agreement could differ in either direction in a full re-docking run.  Glide and
GOLD are commercial and are not tested.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from rdkit.ML.Cluster import Butina
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))

import dockstring_vina_term_decomposition as terms  # noqa: E402  (sibling module reuse)


RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "results" / "nonvina_scorer_transport"
# Gitignored in-package scratch: pose SDFs and ODDT model pickles are large and
# regenerable, so they must not enter the release tree.
DEFAULT_WORK_DIR = ROOT / "downloads" / "nonvina_scorer_transport"
DEFAULT_CONDA_PYTHON = Path.home() / "miniconda3" / "envs" / "toxic_agent_conda_env" / "bin" / "python"
DEFAULT_SMINA = Path.home() / "miniconda3" / "envs" / "smina" / "bin" / "smina"

# Reused verbatim from the sibling fixed-pose analysis so that the ligand block
# here is drawn from the same frozen 15,000-ligand DOCKSTRING primary support.
TARGETS = terms.TARGETS
PRIMARY_SUPPORT_SEED = terms.PRIMARY_SUPPORT_SEED       # 71
LIGAND_SUBSET_SEED = terms.TERM_SUPPORT_SEED            # 20260802
BOOTSTRAP_SEED = 20_260_805
DEFAULT_SAMPLE_SIZE = 2_000
BOOTSTRAP_REPLICATES = 5_000
BUTINA_FINGERPRINT_RADIUS = 2
BUTINA_FINGERPRINT_BITS = 2_048
BUTINA_DISTANCE_CUTOFF = 0.35  # Tanimoto similarity 0.65

ODDT_SCORERS = ("rfscore_v1", "rfscore_v2", "rfscore_v3", "nnscore", "plecscore_linear")
SMINA_SCORERS = ("smina_vina", "vinardo")
SMINA_SCORING_ARGUMENT = {"smina_vina": "vina", "vinardo": "vinardo"}

# Sign convention: every matrix is oriented so that a LARGER value means a
# STRONGER predicted interaction.  Vina-family scores are free energies in
# kcal/mol (more negative is better) and are negated; RF-Score, NNScore and
# PLECscore already predict pKd/pKi.  A global sign flip leaves the column
# correlation matrix -- and hence every participation ratio, PC1 fraction and
# mean correlation reported here -- unchanged; the convention only fixes the
# otherwise arbitrary sign of the reported PC1 loading vector.
SCORER_POLARITY = {
    "released_vina": -1.0,
    "smina_vina": -1.0,
    "vinardo": -1.0,
    "rfscore_v1": 1.0,
    "rfscore_v2": 1.0,
    "rfscore_v3": 1.0,
    "nnscore": 1.0,
    "plecscore_linear": 1.0,
}
SCORER_FAMILY = {
    "released_vina": "vina_family",
    "smina_vina": "vina_family",
    "vinardo": "non_vina_empirical",
    "rfscore_v1": "non_vina_machine_learned",
    "rfscore_v2": "non_vina_machine_learned",
    "rfscore_v3": "non_vina_machine_learned_with_vina_features",
    "nnscore": "non_vina_machine_learned",
    "plecscore_linear": "non_vina_machine_learned",
}

SCORER_NOTES = {
    "released_vina": (
        "AutoDock Vina 1.1.2 score released with DOCKSTRING; the reference column and the "
        "score whose pose every other column is evaluated on."
    ),
    "smina_vina": (
        "smina's re-implementation of the Vina function. Not an independent scoring "
        "function: it is here only as a same-program control, so that any Vinardo-versus-"
        "Vina difference cannot be attributed to the change of executable."
    ),
    "vinardo": (
        "Vinardo drops Vina's gauss2 term, reweights repulsion and redefines the "
        "hydrophobic and hydrogen-bond terms; it is a genuinely different empirical "
        "function evaluated by the same program as smina_vina."
    ),
    "rfscore_v1": "Random forest over 36 element-pair close-contact counts within 12 A.",
    "rfscore_v2": "RF-Score v1 contacts binned over multiple distance shells.",
    "rfscore_v3": (
        "RF-Score v1 contacts PLUS the AutoDock Vina intermolecular terms as extra "
        "features. It is therefore NOT Vina-independent and is reported as a graded "
        "intermediate between the Vina family and the contact-only models."
    ),
    "nnscore": "NNScore 2.0: ensemble of neural networks over 350 BINANA descriptors.",
    "plecscore_linear": (
        "PLECscore linear, applied from the pretrained coefficients shipped with ODDT. "
        "Those coefficients carry intercept_ = 0, so absolute PLECscore values are offset "
        "from the pKd scale; every statistic reported here is correlation-based and is "
        "invariant to a per-column constant shift."
    ),
}

SMINA_AFFINITY_PATTERN = re.compile(r"^Affinity:\s+([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)", re.MULTILINE)

# Prediction thresholds, fixed before any scorer was run.
PC1_DOMINANCE_THRESHOLD = 0.50          # PC1 explains at least half the spectrum
PC1_UNIFORM_COSINE_THRESHOLD = 0.95     # PC1 within ~18 degrees of the uniform direction
RAW_PR_FRACTION_THRESHOLD = 0.50        # raw PR below half the target count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, default=terms.DEFAULT_DATASET)
    parser.add_argument("--pose-dir", type=Path, default=terms.DEFAULT_POSE_DIR)
    parser.add_argument("--target-dir", type=Path, default=terms.DEFAULT_TARGET_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    parser.add_argument("--conda-python", type=Path, default=DEFAULT_CONDA_PYTHON)
    parser.add_argument("--smina", type=Path, default=DEFAULT_SMINA)
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    parser.add_argument("--bootstrap-replicates", type=int, default=BOOTSTRAP_REPLICATES)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument(
        "--oddt-scorers",
        default=",".join(ODDT_SCORERS),
        help="comma separated subset of the ODDT scorers; empty string disables the ODDT stage",
    )
    parser.add_argument("--skip-smina", action="store_true")
    parser.add_argument("--force", action="store_true", help="recompute cached pose and score stages")
    parser.add_argument(
        "--finalize-existing",
        action="store_true",
        help=(
            "sanitize and checksum the already frozen score/summary artifacts "
            "without recreating source-restricted scorer outputs"
        ),
    )
    return parser.parse_args()


# --------------------------------------------------------------------------- #
# Stage 1 -- fixed poses                                                       #
# --------------------------------------------------------------------------- #


def extract_target_poses(archive: Path, keys: set[str], destination: Path) -> dict[str, Any]:
    """Stream one DOCKSTRING pose archive and write the selected records in place.

    ``terms.iter_sdf_records`` and ``terms.KEY_PATTERN`` are reused so the record
    boundaries and the ligand key are parsed exactly as in the sibling fixed-pose
    term decomposition.
    """
    started = time.perf_counter()
    found: dict[str, str] = {}
    scanned = 0
    for record in terms.iter_sdf_records(archive):
        scanned += 1
        match = terms.KEY_PATTERN.search(record)
        if match is None:
            raise ValueError(f"pose archive record lacks key: {archive}")
        key = match.group(1)
        if key in keys:
            if key in found:
                raise ValueError(f"duplicate selected key in {archive}: {key}")
            found[key] = record
    destination.parent.mkdir(parents=True, exist_ok=True)
    return {
        "archive": str(archive),
        "records_scanned": scanned,
        "records_found": len(found),
        "seconds": float(time.perf_counter() - started),
        "records": found,
    }


def build_pose_block(
    args: argparse.Namespace, selected: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Write one SDF of fixed poses per target and return the rectangular block."""
    pose_dir = args.work_dir / "poses"
    manifest_path = args.work_dir / "pose_manifest.json"
    keys = set(selected.inchikey.astype(str))

    if not args.force and manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("requested_keys") == len(keys) and all(
            (pose_dir / f"{target}.sdf").is_file() for target in TARGETS
        ):
            retained = pd.Index(manifest["retained_keys"])
            return selected[selected.inchikey.isin(retained)].copy(), manifest

    pose_dir.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=min(args.workers, len(TARGETS))) as pool:
        futures = {
            target: pool.submit(
                extract_target_poses,
                terms.pose_archive_path(args.pose_dir, target),
                keys,
                pose_dir / f"{target}.sdf",
            )
            for target in TARGETS
        }
        results = {target: future.result() for target, future in futures.items()}

    retained = set(keys)
    for result in results.values():
        retained &= set(result["records"])
    order = [key for key in selected.inchikey.astype(str) if key in retained]

    per_target: dict[str, Any] = {}
    for target in TARGETS:
        result = results[target]
        destination = pose_dir / f"{target}.sdf"
        temporary = destination.with_suffix(".sdf.tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            for key in order:
                handle.write(result["records"][key])
        os.replace(temporary, destination)
        per_target[target] = {
            "archive": result["archive"],
            "archive_md5": terms.md5_file(Path(result["archive"])),
            "expected_archive_md5": terms.POSE_ARCHIVES[target][1],
            "records_scanned": result["records_scanned"],
            "records_found": result["records_found"],
            "extract_seconds": result["seconds"],
            "pose_sdf_sha256": terms.sha256_file(destination),
        }
        if per_target[target]["archive_md5"] != per_target[target]["expected_archive_md5"]:
            raise ValueError(f"pose archive MD5 mismatch for {target}")

    manifest = {
        "requested_keys": len(keys),
        "retained_keys": order,
        "n_retained": len(order),
        "per_target": per_target,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    block = selected[selected.inchikey.isin(set(order))].copy()
    return block, manifest


# --------------------------------------------------------------------------- #
# Stage 2 -- smina (Vina re-implementation and Vinardo) on the fixed poses      #
# --------------------------------------------------------------------------- #


def run_smina(args: argparse.Namespace, keys_by_target: dict[str, list[str]]) -> tuple[pd.DataFrame, dict[str, Any]]:
    csv_path = args.work_dir / "smina_scores.csv"
    json_path = args.work_dir / "smina_provenance.json"
    if not args.force and csv_path.is_file() and json_path.is_file():
        return pd.read_csv(csv_path), json.loads(json_path.read_text())

    version = subprocess.run(
        [str(args.smina), "--version"], text=True, capture_output=True, check=True
    ).stdout.strip()
    rows: list[dict[str, Any]] = []
    timings: dict[str, Any] = {}
    for scorer in SMINA_SCORERS:
        for target in TARGETS:
            receptor = terms.receptor_path(args.target_dir, target)
            poses = args.work_dir / "poses" / f"{target}.sdf"
            started = time.perf_counter()
            completed = subprocess.run(
                [
                    str(args.smina),
                    "--score_only",
                    "--scoring",
                    SMINA_SCORING_ARGUMENT[scorer],
                    "-r",
                    str(receptor),
                    "-l",
                    str(poses),
                    "--cpu",
                    "1",
                ],
                text=True,
                capture_output=True,
                env=terms.command_environment(),
                timeout=3_600,
            )
            if completed.returncode != 0:
                raise RuntimeError(f"smina failed on {target}/{scorer}: {completed.stderr[-2000:]}")
            affinities = [float(value) for value in SMINA_AFFINITY_PATTERN.findall(completed.stdout)]
            expected = keys_by_target[target]
            if len(affinities) != len(expected):
                raise ValueError(
                    f"smina returned {len(affinities)} affinities for {len(expected)} poses "
                    f"({target}/{scorer}); input-order alignment is unsafe"
                )
            for key, value in zip(expected, affinities):
                rows.append({"target": target, "key": key, "scorer": scorer, "score": value})
            timings[f"{scorer}/{target}"] = float(time.perf_counter() - started)

    frame = pd.DataFrame(rows).pivot_table(
        index=["target", "key"], columns="scorer", values="score"
    ).reset_index()
    frame.columns.name = None
    provenance = {
        "binary": str(args.smina),
        "binary_sha256": terms.sha256_file(args.smina),
        "version_string": version,
        "scoring_arguments": SMINA_SCORING_ARGUMENT,
        "alignment_rule": (
            "smina --score_only does not echo SDF data tags, so scores are aligned to "
            "the input SDF record order; the per-target count is asserted and the "
            "smina_vina column is cross-checked against the released DOCKSTRING score."
        ),
        "seconds_by_target": timings,
    }
    frame.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    return frame, provenance


# --------------------------------------------------------------------------- #
# Stage 3 -- ODDT machine-learned scorers under the conda interpreter           #
# --------------------------------------------------------------------------- #


def run_oddt(args: argparse.Namespace, scorers: list[str]) -> tuple[pd.DataFrame, dict[str, Any]]:
    csv_path = args.work_dir / "oddt_scores.csv"
    json_path = args.work_dir / "oddt_provenance.json"
    if not args.force and csv_path.is_file() and json_path.is_file():
        return pd.read_csv(csv_path), json.loads(json_path.read_text())
    if not args.conda_python.is_file():
        raise FileNotFoundError(f"ODDT interpreter absent: {args.conda_python}")

    worker = Path(__file__).resolve().parent / "nonvina_scorer_oddt_worker.py"
    command = [
        str(args.conda_python),
        str(worker),
        "--poses-dir",
        str(args.work_dir / "poses"),
        "--target-dir",
        str(args.target_dir),
        "--model-dir",
        str(args.work_dir / "models"),
        "--out-csv",
        str(csv_path),
        "--out-json",
        str(json_path),
        "--targets",
        ",".join(TARGETS),
        "--scorers",
        ",".join(scorers),
        "--workers",
        str(args.workers),
    ]
    started = time.perf_counter()
    completed = subprocess.run(command, text=True, capture_output=True, env=terms.command_environment())
    if completed.returncode != 0:
        raise RuntimeError(
            "ODDT worker failed:\n"
            + completed.stdout[-4000:]
            + "\n"
            + completed.stderr[-4000:]
        )
    provenance = json.loads(json_path.read_text())
    provenance["wall_seconds"] = float(time.perf_counter() - started)
    json_path.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    return pd.read_csv(csv_path), provenance


# --------------------------------------------------------------------------- #
# Spectral statistics                                                          #
# --------------------------------------------------------------------------- #


def transport_metrics(matrix: np.ndarray) -> dict[str, float]:
    """Spectral descriptors of one ligand-by-target score block.

    ``terms.standardize_columns`` and ``terms.correlation_pr`` are reused so the
    definitions match the package's existing spectral tables exactly.

    Note that removing each ligand's own mean across targets ("within-ligand
    centring") and two-way centring give the *same* column-correlation matrix,
    because correlation removes column means anyway.  ``row_centered_*`` here is
    therefore directly comparable to the ``residual_*`` columns of
    ``results/dockstring_vina_terms/term_spectral_metrics.csv``.
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    n_rows, n_targets = matrix.shape
    raw_z = terms.standardize_columns(matrix)
    raw_corr = raw_z.T @ raw_z / (n_rows - 1)
    row_centered = matrix - matrix.mean(1, keepdims=True)
    residual_z = terms.standardize_columns(row_centered)
    residual_corr = residual_z.T @ residual_z / (n_rows - 1)

    raw_eigenvalues, raw_vectors = np.linalg.eigh(raw_corr)
    raw_eigenvalues = raw_eigenvalues[::-1]
    raw_vectors = raw_vectors[:, ::-1]
    residual_eigenvalues = np.linalg.eigvalsh(residual_corr)[::-1]

    loading = raw_vectors[:, 0]
    if loading.sum() < 0:  # eigenvector sign is arbitrary; fix it to the positive orthant
        loading = -loading
    uniform = np.ones(n_targets) / np.sqrt(n_targets)
    upper = np.triu_indices(n_targets, 1)

    raw_pr = float(raw_eigenvalues.sum() ** 2 / np.square(raw_eigenvalues).sum())
    residual_pr = float(residual_eigenvalues.sum() ** 2 / np.square(residual_eigenvalues).sum())
    return {
        "raw_pc1_fraction": float(raw_eigenvalues[0] / raw_eigenvalues.sum()),
        "raw_participation_ratio": raw_pr,
        "row_centered_participation_ratio": residual_pr,
        "participation_ratio_increase": residual_pr - raw_pr,
        "raw_mean_squared_offdiagonal_correlation": float(np.square(raw_corr[upper]).mean()),
        "row_centered_mean_squared_offdiagonal_correlation": float(
            np.square(residual_corr[upper]).mean()
        ),
        "pc1_uniform_cosine": float(loading @ uniform),
        "pc1_min_loading": float(loading.min()),
        "pc1_max_loading": float(loading.max()),
        "raw_mean_correlation": float(raw_corr[upper].mean()),
        "row_centered_pc1_fraction": float(residual_eigenvalues[0] / residual_eigenvalues.sum()),
        "row_centered_mean_absolute_correlation": float(np.abs(residual_corr[upper]).mean()),
        "z_then_row_center_participation_ratio": terms.correlation_pr(
            raw_z - raw_z.mean(1, keepdims=True)
        ),
    }


BOOTSTRAP_METRICS = (
    "raw_pc1_fraction",
    "raw_participation_ratio",
    "row_centered_participation_ratio",
    "participation_ratio_increase",
    "raw_mean_squared_offdiagonal_correlation",
    "row_centered_mean_squared_offdiagonal_correlation",
    "pc1_uniform_cosine",
)


REFERENCE_SCORER = "released_vina"


def _interval(values: np.ndarray) -> dict[str, Any]:
    return {
        "median": float(np.median(values)),
        "interval95": [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))],
    }


def cluster_bootstrap(
    matrices: dict[str, np.ndarray],
    groups: np.ndarray,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    """Resample chemical clusters with replacement and recompute every metric.

    Also returns the *paired* contrast of each scorer against the released Vina
    reference within the same resampled ligand set; the two matrices share the
    ligand block, so the paired difference is far better determined than the
    difference of two marginal intervals.
    """
    unique = np.unique(groups)
    members = {group: np.flatnonzero(groups == group) for group in unique}
    rng = np.random.default_rng(seed)
    collected = {
        scorer: {metric: np.empty(replicates, dtype=float) for metric in BOOTSTRAP_METRICS}
        for scorer in matrices
    }
    contrasts = {
        scorer: {metric: np.empty(replicates, dtype=float) for metric in BOOTSTRAP_METRICS}
        for scorer in matrices
        if scorer != REFERENCE_SCORER
    }
    for replicate in range(replicates):
        sampled = rng.choice(unique, len(unique), replace=True)
        index = np.concatenate([members[group] for group in sampled])
        replicate_values = {
            scorer: transport_metrics(matrix[index]) for scorer, matrix in matrices.items()
        }
        reference_values = replicate_values.get(REFERENCE_SCORER)
        for scorer, values in replicate_values.items():
            for metric in BOOTSTRAP_METRICS:
                collected[scorer][metric][replicate] = values[metric]
                if reference_values is not None and scorer in contrasts:
                    contrasts[scorer][metric][replicate] = values[metric] - reference_values[metric]
    return {
        "n_clusters": int(len(unique)),
        "replicates": int(replicates),
        "seed": int(seed),
        "metrics": {
            scorer: {metric: _interval(values) for metric, values in per_metric.items()}
            for scorer, per_metric in collected.items()
        },
        "paired_contrast_vs_released_vina": {
            scorer: {metric: _interval(values) for metric, values in per_metric.items()}
            for scorer, per_metric in contrasts.items()
        },
    }


def conservative_interval(*intervals: list[float]) -> list[float]:
    """Union of several bootstrap intervals -- the package's conservative rule."""
    lows = [interval[0] for interval in intervals]
    highs = [interval[1] for interval in intervals]
    return [float(min(lows)), float(max(highs))]


def butina_clusters(smiles: list[str]) -> np.ndarray:
    """Butina clusters over Morgan(r=2, 2048-bit) Tanimoto distances.

    DOCKSTRING ships no precomputed Butina column (unlike the Docking-44 canonical
    table), so the clustering recipe and cutoff are stated explicitly here and
    the resulting interval is only ever used in a conservative union with the
    Bemis-Murcko-scaffold interval.
    """
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=BUTINA_FINGERPRINT_RADIUS, fpSize=BUTINA_FINGERPRINT_BITS
    )
    fingerprints = []
    for index, value in enumerate(smiles):
        molecule = Chem.MolFromSmiles(value)
        if molecule is None:
            raise ValueError(f"invalid SMILES at position {index}")
        fingerprints.append(generator.GetFingerprint(molecule))
    distances: list[float] = []
    for index in range(1, len(fingerprints)):
        similarities = DataStructs.BulkTanimotoSimilarity(fingerprints[index], fingerprints[:index])
        distances.extend(1.0 - value for value in similarities)
    clusters = Butina.ClusterData(
        distances, len(fingerprints), BUTINA_DISTANCE_CUTOFF, isDistData=True
    )
    labels = np.empty(len(fingerprints), dtype=np.int64)
    for cluster_index, cluster in enumerate(clusters):
        for member in cluster:
            labels[member] = cluster_index
    return labels


def evaluate_prediction(observed: dict[str, float], intervals: dict[str, list[float]], n_targets: int) -> dict[str, Any]:
    """Score one scorer against the four prespecified transport criteria."""
    dominant = observed["raw_pc1_fraction"] >= PC1_DOMINANCE_THRESHOLD
    near_uniform = (
        observed["pc1_uniform_cosine"] >= PC1_UNIFORM_COSINE_THRESHOLD
        and observed["pc1_min_loading"] > 0.0
    )
    low_raw_pr = observed["raw_participation_ratio"] <= RAW_PR_FRACTION_THRESHOLD * n_targets
    pr_rises = intervals["participation_ratio_increase"][0] > 0.0
    return {
        "pc1_dominant": bool(dominant),
        "pc1_near_uniform": bool(near_uniform),
        "raw_participation_ratio_low": bool(low_raw_pr),
        "participation_ratio_rises_under_within_ligand_centring": bool(pr_rises),
        "prediction_holds": bool(dominant and near_uniform and low_raw_pr and pr_rises),
        "criteria": {
            "pc1_dominant": f"raw_pc1_fraction >= {PC1_DOMINANCE_THRESHOLD}",
            "pc1_near_uniform": (
                f"cosine with the uniform target vector >= {PC1_UNIFORM_COSINE_THRESHOLD} "
                "and every PC1 loading strictly positive"
            ),
            "raw_participation_ratio_low": (
                f"raw participation ratio <= {RAW_PR_FRACTION_THRESHOLD} x number of targets "
                f"({RAW_PR_FRACTION_THRESHOLD * n_targets:g})"
            ),
            "participation_ratio_rises_under_within_ligand_centring": (
                "lower end of the conservative 95% cluster-bootstrap interval on the "
                "participation-ratio increase is strictly positive"
            ),
        },
    }


# --------------------------------------------------------------------------- #
# Assembly                                                                     #
# --------------------------------------------------------------------------- #


def assemble_matrices(
    block: pd.DataFrame,
    smina_frame: pd.DataFrame | None,
    oddt_frame: pd.DataFrame | None,
    oddt_scorers: list[str],
) -> tuple[dict[str, np.ndarray], list[str], list[str]]:
    order = block.inchikey.astype(str).tolist()
    matrices: dict[str, np.ndarray] = {}

    released = block.set_index(block.inchikey.astype(str))[list(TARGETS)].to_numpy(dtype=np.float64)
    matrices["released_vina"] = released

    frames: list[tuple[list[str], pd.DataFrame]] = []
    if smina_frame is not None:
        frames.append((list(SMINA_SCORERS), smina_frame))
    if oddt_frame is not None:
        frames.append((oddt_scorers, oddt_frame))
    for names, frame in frames:
        for name in names:
            pivot = frame.pivot(index="key", columns="target", values=name)
            pivot = pivot.reindex(index=order, columns=list(TARGETS))
            if pivot.isna().any().any():
                raise ValueError(f"non-rectangular score matrix for {name}")
            matrices[name] = pivot.to_numpy(dtype=np.float64)

    ordered = [name for name in SCORER_POLARITY if name in matrices]
    oriented = {name: matrices[name] * SCORER_POLARITY[name] for name in ordered}
    non_vina = [name for name in ordered if SCORER_FAMILY[name] != "vina_family"]
    return oriented, ordered, non_vina


def drop_wall_clock(value: object) -> object:
    """Strip every wall-clock timing from a summary payload, recursively.

    Timings vary between a cold and a warm run, and this summary is registered by
    SHA-256 in ``results/manuscript_source_manifest.csv``. Leaving them in would make a
    faithful rerun of the analysis fail the very first gate of the build.
    """
    if isinstance(value, dict):
        return {
            key: drop_wall_clock(item)
            for key, item in value.items()
            if not (key.endswith("_seconds") or key == "seconds")
        }
    if isinstance(value, list):
        return [drop_wall_clock(item) for item in value]
    return value


def _portable_release_path(value: str) -> str:
    path = Path(value)
    if not path.is_absolute():
        return value
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return f"external-not-redistributed/{path.name}"


def _sanitize_portable_paths(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _sanitize_portable_paths(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_sanitize_portable_paths(item) for item in value]
    if isinstance(value, str) and value.startswith("/"):
        return _portable_release_path(value)
    return value


def finalize_existing_release(output_dir: Path) -> dict[str, Any]:
    """Make the fixed-pose transport boundary portable and machine-readable."""

    output_dir = output_dir.resolve()
    summary_path = output_dir / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(summary_path)
    summary = _sanitize_portable_paths(json.loads(summary_path.read_text()))
    summary["question"] = (
        "Conditional on the identical Vina-selected DOCKSTRING poses, do the tested "
        "empirical and learned scorers retain a dominant shared ligand axis and a "
        "participation-ratio increase after within-ligand centring?"
    )
    summary["reproducibility_boundary"] = {
        "pose_selection": "top poses selected by AutoDock Vina and held fixed",
        "independent_redocking_per_scorer": False,
        "standard_release_stage": {
            "environment": "release Python environment pinned by requirements-lock.txt",
            "inputs": [
                "results/nonvina_scorer_transport/fixed_pose_scores.csv.gz",
                "results/nonvina_scorer_transport/scorer_spectral_metrics.csv",
            ],
            "scope": "verification and downstream reporting from frozen scores",
        },
        "source_restricted_smina_stage": {
            "environment": "external smina executable identified by SHA-256 and version",
            "redistributed": False,
            "requires": "external Figshare pose archives and Dockstring receptors",
        },
        "source_restricted_oddt_stage": {
            "environment": "Python 3.9 with ODDT 0.7; separate from the release Python environment",
            "redistributed": False,
            "requires": (
                "external Figshare pose archives, Dockstring receptors, PDBbind-derived "
                "ODDT training tables/models and analysis/nonvina_scorer_oddt_worker.py"
            ),
        },
        "generalization_limit": (
            "Agreement across these rescored columns is conditional on Vina pose "
            "selection and cannot establish transport to independent re-docking, "
            "commercial scorers, new targets, pose accuracy or affinity accuracy."
        ),
    }
    tracked = sorted(
        path
        for path in output_dir.iterdir()
        if path.is_file() and path.name not in {"summary.json", "output_checksums.json"}
    )
    summary["frozen_output_contract"] = {
        "algorithm": "sha256",
        "files_excluding_summary_and_checksum_manifest": {
            path.name: terms.sha256_file(path) for path in tracked
        },
    }
    terms.atomic_json(summary_path, drop_wall_clock(summary))
    return summary


def main() -> None:
    args = parse_args()
    if args.finalize_existing:
        summary = finalize_existing_release(args.output_dir)
        print(
            json.dumps(
                {
                    "status": summary["status"],
                    "reproducibility_boundary": summary[
                        "reproducibility_boundary"
                    ],
                },
                indent=2,
            )
        )
        return
    started = time.perf_counter()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    oddt_scorers = [value for value in args.oddt_scorers.split(",") if value]

    _, selected = terms.select_ligands(args.dataset, args.sample_size)
    block, pose_manifest = build_pose_block(args, selected)
    block = block.sort_values("selection_rank").reset_index(drop=True)
    keys_by_target = {target: block.inchikey.astype(str).tolist() for target in TARGETS}

    smina_frame = None
    smina_provenance: dict[str, Any] = {"status": "skipped"}
    if not args.skip_smina and args.smina.is_file():
        smina_frame, smina_provenance = run_smina(args, keys_by_target)
    elif not args.skip_smina:
        smina_provenance = {"status": "unavailable", "path": str(args.smina)}

    oddt_frame = None
    oddt_provenance: dict[str, Any] = {"status": "skipped"}
    if oddt_scorers:
        oddt_frame, oddt_provenance = run_oddt(args, oddt_scorers)

    matrices, ordered, non_vina = assemble_matrices(block, smina_frame, oddt_frame, oddt_scorers)

    descriptors, murcko_groups = terms.molecule_descriptors(block)
    butina_labels = butina_clusters(block.smiles.astype(str).tolist())

    observed = {name: transport_metrics(matrix) for name, matrix in matrices.items()}
    murcko_bootstrap = cluster_bootstrap(matrices, murcko_groups, args.bootstrap_replicates, BOOTSTRAP_SEED)
    butina_bootstrap = cluster_bootstrap(
        matrices, butina_labels, args.bootstrap_replicates, BOOTSTRAP_SEED + 1
    )

    n_targets = len(TARGETS)
    rows: list[dict[str, Any]] = []
    prediction: dict[str, Any] = {}
    for name in ordered:
        values = observed[name]
        conservative = {
            metric: conservative_interval(
                murcko_bootstrap["metrics"][name][metric]["interval95"],
                butina_bootstrap["metrics"][name][metric]["interval95"],
            )
            for metric in BOOTSTRAP_METRICS
        }
        prediction[name] = evaluate_prediction(values, conservative, n_targets)
        row: dict[str, Any] = {
            "scorer": name,
            "family": SCORER_FAMILY[name],
            "polarity_applied": SCORER_POLARITY[name],
            "n_ligands": int(len(block)),
            "n_targets": n_targets,
            "n_cells": int(len(block) * n_targets),
        }
        row.update(values)
        for metric in BOOTSTRAP_METRICS:
            row[f"{metric}_ci_low"] = conservative[metric][0]
            row[f"{metric}_ci_high"] = conservative[metric][1]
        for metric in ("raw_participation_ratio", "raw_pc1_fraction"):
            if name == REFERENCE_SCORER:
                row[f"{metric}_vs_released_vina"] = 0.0
                row[f"{metric}_vs_released_vina_ci_low"] = 0.0
                row[f"{metric}_vs_released_vina_ci_high"] = 0.0
                continue
            interval = conservative_interval(
                murcko_bootstrap["paired_contrast_vs_released_vina"][name][metric]["interval95"],
                butina_bootstrap["paired_contrast_vs_released_vina"][name][metric]["interval95"],
            )
            row[f"{metric}_vs_released_vina"] = (
                values[metric] - observed[REFERENCE_SCORER][metric]
            )
            row[f"{metric}_vs_released_vina_ci_low"] = interval[0]
            row[f"{metric}_vs_released_vina_ci_high"] = interval[1]
        row["prediction_holds"] = prediction[name]["prediction_holds"]
        rows.append(row)
    metrics_frame = pd.DataFrame(rows)

    # Is it the *same* axis?  Compare each scorer's ligand-wide mean, and its
    # relationship to molecular size, against the released Vina reference.
    heavy_atoms = descriptors["heavy_atoms"].to_numpy(dtype=float)
    reference_axis = matrices["released_vina"].mean(1)
    axis_agreement: dict[str, Any] = {}
    for name in ordered:
        axis = matrices[name].mean(1)
        axis_agreement[name] = {
            "spearman_with_released_vina_ligand_axis": float(
                stats.spearmanr(axis, reference_axis).statistic
            ),
            "spearman_ligand_axis_vs_heavy_atoms": float(
                stats.spearmanr(axis, heavy_atoms).statistic
            ),
            "spearman_residual_rms_vs_heavy_atoms": float(
                stats.spearmanr(
                    np.sqrt(np.mean(np.square(terms.two_way_center(matrices[name])), axis=1)),
                    heavy_atoms,
                ).statistic
            ),
        }

    # The manuscript's Discussion already states a falsifiable Vinardo prediction
    # (manuscript.tex, "Vinardo removes gauss2, reweights repulsion and redefines the
    # hydrophobic and hydrogen-bond terms, so on identical poses it should show a
    # higher raw PR and a smaller uniform PC1 fraction than Vina, while still showing
    # the centring-induced rise").  This block tests it directly.
    manuscript_check: dict[str, Any] = {"status": "not_testable_without_vinardo"}
    if "vinardo" in matrices:
        contrast = {
            metric: conservative_interval(
                murcko_bootstrap["paired_contrast_vs_released_vina"]["vinardo"][metric]["interval95"],
                butina_bootstrap["paired_contrast_vs_released_vina"]["vinardo"][metric]["interval95"],
            )
            for metric in ("raw_participation_ratio", "raw_pc1_fraction", "participation_ratio_increase")
        }
        pr_delta = observed["vinardo"]["raw_participation_ratio"] - observed[REFERENCE_SCORER]["raw_participation_ratio"]
        pc1_delta = observed["vinardo"]["raw_pc1_fraction"] - observed[REFERENCE_SCORER]["raw_pc1_fraction"]
        # A direction is "confirmed" only when the paired 95% interval lies wholly
        # on that side of zero, so neither confirmation nor refutation can rest on
        # a point estimate.
        higher_pr = contrast["raw_participation_ratio"][0] > 0.0
        lower_pr = contrast["raw_participation_ratio"][1] < 0.0
        smaller_pc1 = contrast["raw_pc1_fraction"][1] < 0.0
        larger_pc1 = contrast["raw_pc1_fraction"][0] > 0.0
        rise = prediction["vinardo"]["participation_ratio_rises_under_within_ligand_centring"]
        if higher_pr and smaller_pc1 and rise:
            verdict = "supported"
        elif lower_pr or larger_pc1:
            verdict = "refuted_the_predicted_direction"
        else:
            verdict = "indeterminate"
        manuscript_check = {
            "status": "tested",
            "source": "manuscript.tex, Discussion, transportability paragraph",
            "stated_prediction": (
                "Vinardo removes gauss2, reweights repulsion and redefines the hydrophobic "
                "and hydrogen-bond terms, so on identical poses it should show a higher raw "
                "PR and a smaller uniform PC1 fraction than Vina, while still showing the "
                "centring-induced rise."
            ),
            "observed_raw_participation_ratio_vinardo_minus_vina": float(pr_delta),
            "raw_participation_ratio_delta_ci95": contrast["raw_participation_ratio"],
            "observed_raw_pc1_fraction_vinardo_minus_vina": float(pc1_delta),
            "raw_pc1_fraction_delta_ci95": contrast["raw_pc1_fraction"],
            "vinardo_raw_participation_ratio_is_higher_as_predicted": bool(higher_pr),
            "vinardo_raw_participation_ratio_is_lower_contradicting_prediction": bool(lower_pr),
            "vinardo_raw_pc1_fraction_is_smaller_as_predicted": bool(smaller_pc1),
            "vinardo_raw_pc1_fraction_is_larger_contradicting_prediction": bool(larger_pc1),
            "vinardo_shows_the_centring_induced_rise": bool(rise),
            "verdict": verdict,
            "interpretation": (
                "The direction test uses the paired cluster-bootstrap contrast on the same "
                "ligand block, so it is not confounded by ligand sampling. A refutation "
                "concerns only the predicted DIRECTION of the Vina-to-Vinardo shift; it "
                "leaves the qualitative transport claim -- dominant near-uniform PC1, low raw "
                "PR rising under within-ligand centring -- to be judged on its own criteria."
            ),
        }

    released_raw = block[list(TARGETS)].to_numpy(dtype=np.float64)
    smina_vina_check: dict[str, Any] = {"status": "not_run"}
    if "smina_vina" in matrices:
        per_target = {
            target: float(
                stats.pearsonr(
                    released_raw[:, index], matrices["smina_vina"][:, index] * SCORER_POLARITY["smina_vina"]
                ).statistic
            )
            for index, target in enumerate(TARGETS)
        }
        smina_vina_check = {
            "purpose": (
                "smina re-implements the Vina function, so smina_vina rescored on the "
                "released pose must reproduce the released DOCKSTRING score; this also "
                "verifies that the order-based alignment of smina output to ligand keys "
                "is correct."
            ),
            "per_target_pearson_r": per_target,
            "minimum_pearson_r": float(min(per_target.values())),
        }
        if smina_vina_check["minimum_pearson_r"] < 0.95:
            raise ValueError("smina Vina rescoring disagrees with the released score; alignment unsafe")

    positive_released = int((released_raw > 0).sum())

    holds = [name for name in non_vina if prediction[name]["prediction_holds"]]
    fails = [name for name in non_vina if not prediction[name]["prediction_holds"]]

    summary: dict[str, Any] = {
        "analysis": "nonvina_scorer_transport",
        "analysis_version": "1.0.0",
        "analysis_status": "confirmatory_prespecified_prediction_on_fixed_poses",
        "status": "complete" if not fails else "complete_with_at_least_one_scorer_failing_the_prediction",
        "question": (
            "Is the dominant near-uniform first principal component, and the low raw "
            "participation ratio that rises under within-ligand centring, a property of "
            "empirical scoring functions in general, or an artefact of the AutoDock Vina "
            "scoring function specifically?"
        ),
        "falsifiable_prediction": (
            "On an identical ligand-by-target block of fixed DOCKSTRING poses, every "
            "non-Vina scorer shows raw PC1 fraction >= 0.50, a PC1 loading vector with "
            "cosine >= 0.95 to the uniform target direction and no negative loading, a raw "
            "participation ratio <= half the target count, and a participation-ratio "
            "increase under within-ligand centring whose conservative 95% cluster-bootstrap "
            "interval excludes zero. Any scorer failing any criterion falsifies the general "
            "form of the claim for that scorer."
        ),
        "support": {
            "n_ligands": int(len(block)),
            "n_targets": n_targets,
            "n_cells": int(len(block) * n_targets),
            "n_scorers": len(ordered),
            "n_non_vina_scorers": len(non_vina),
            "targets": list(TARGETS),
            "scorers": ordered,
            "non_vina_scorers": non_vina,
            "requested_sample_size": int(args.sample_size),
            "ligands_dropped_for_missing_pose": int(args.sample_size - len(block)),
            "ligand_source": (
                "the frozen 15,000-ligand DOCKSTRING primary support (seed 71) used "
                "throughout this package, sub-sampled with seed 20260802 exactly as in "
                "analysis/dockstring_vina_term_decomposition.py"
            ),
            "pose_source": (
                "released DOCKSTRING pose archives (Figshare DOI "
                f"{terms.FIGSHARE_DOI}); the top Vina pose per ligand-target cell, held "
                "fixed across all scorers"
            ),
            "released_scores_above_zero": positive_released,
            "clipping": (
                "no clip(upper=0) is applied: the comparison is across scorers on one "
                "block, and clipping only the Vina-family columns would be asymmetric"
            ),
        },
        "seeds": {
            "dockstring_primary_support_seed": int(PRIMARY_SUPPORT_SEED),
            "ligand_subset_seed": int(LIGAND_SUBSET_SEED),
            "murcko_cluster_bootstrap_seed": int(BOOTSTRAP_SEED),
            "butina_cluster_bootstrap_seed": int(BOOTSTRAP_SEED + 1),
            "oddt_internal_training_seed": 1,
        },
        "metric_definitions": {
            "raw_pc1_fraction": "largest eigenvalue of the target correlation matrix divided by its trace",
            "raw_participation_ratio": "(sum of eigenvalues)^2 / sum of squared eigenvalues of the raw target correlation matrix",
            "row_centered_participation_ratio": (
                "same, after subtracting each ligand's mean across targets; identical to "
                "the two-way-centred residual because correlation removes column means"
            ),
            "raw_mean_squared_offdiagonal_correlation": "mean of squared off-diagonal target correlations",
            "pc1_uniform_cosine": (
                "cosine between the raw PC1 loading vector (sign fixed so its sum is positive) "
                "and the uniform target vector 1/sqrt(p)"
            ),
        },
        "interval_rule": (
            "conservative union of the Bemis-Murcko-scaffold and Butina-cluster 95% "
            "bootstrap intervals, 5,000 replicates each, resampling whole clusters"
        ),
        "bootstrap": {
            "replicates": int(args.bootstrap_replicates),
            "murcko_scaffold_clusters": murcko_bootstrap["n_clusters"],
            "butina_clusters": butina_bootstrap["n_clusters"],
            "butina_recipe": (
                f"RDKit Butina on Morgan r={BUTINA_FINGERPRINT_RADIUS} "
                f"{BUTINA_FINGERPRINT_BITS}-bit Tanimoto distances, cutoff "
                f"{BUTINA_DISTANCE_CUTOFF} (similarity {1 - BUTINA_DISTANCE_CUTOFF:g}); "
                "DOCKSTRING carries no upstream Butina column, so this recipe is stated "
                "rather than inherited"
            ),
            "murcko": murcko_bootstrap["metrics"],
            "butina": butina_bootstrap["metrics"],
        },
        "observed": observed,
        "prediction_outcome": prediction,
        "prediction_summary": {
            "non_vina_scorers_supporting_the_prediction": holds,
            "non_vina_scorers_falsifying_the_prediction": fails,
            "fraction_non_vina_supporting": float(len(holds) / len(non_vina)) if non_vina else None,
            "non_vina_scorers_failing_each_criterion": {
                criterion: [
                    name for name in non_vina if not prediction[name][criterion]
                ]
                for criterion in (
                    "pc1_dominant",
                    "pc1_near_uniform",
                    "raw_participation_ratio_low",
                    "participation_ratio_rises_under_within_ligand_centring",
                )
            },
        },
        "scorer_notes": SCORER_NOTES,
        "manuscript_vinardo_direction_prediction": manuscript_check,
        "reproduction": {
            "command": (
                "analysis/nonvina_scorer_transport.py --sample-size 2000 --workers 6, run "
                "with the release .venv interpreter"
            ),
            "not_in_make_all": (
                "This analysis is deliberately outside the `make PYTHON=.venv/bin/python all` "
                "target because it needs three things the release bundle does not carry: the "
                "~600 MB DOCKSTRING pose archives, an ODDT 0.7 environment on Python 3.9, and "
                "a smina binary. Its outputs are versioned instead, exactly as for the other "
                "source-restricted upstream analyses."
            ),
            "external_requirements": {
                "pose_archives": f"Figshare DOI {terms.FIGSHARE_DOI}, MD5-verified at run time",
                "oddt_interpreter": str(args.conda_python),
                "smina_binary": str(args.smina),
                "receptors": "dockstring package resources/targets/*_target.pdbqt",
            },
        },
        "shared_axis_agreement": axis_agreement,
        "smina_vina_reproduction_check": smina_vina_check,
        "toolchain": {
            "analysis_interpreter": sys.version,
            "analysis_interpreter_executable": sys.executable,
            "numpy_version": np.__version__,
            "pandas_version": pd.__version__,
            "scipy_version": __import__("scipy").__version__,
            "rdkit_version": __import__("rdkit").__version__,
            "platform": platform.platform(),
            "oddt_stage": oddt_provenance,
            "smina_stage": smina_provenance,
            "pose_manifest": {
                target: {
                    key: value
                    for key, value in entry.items()
                    if key != "records"
                }
                for target, entry in pose_manifest["per_target"].items()
            },
        },
        "claim_boundary": (
            "This is a rescoring of retained AutoDock Vina poses, not an independent "
            "re-docking: each scorer is evaluated on the pose Vina selected, so a scoring "
            "function that would have chosen a different pose during search is not "
            "simulated, so agreement could differ in either direction in a full re-docking "
            "run. RF-Score, NNScore and PLECscore are trained on "
            "PDBbind crystal complexes and are being applied off-distribution to docked "
            "poses; RF-Score v3 additionally takes Vina terms as input features and is "
            "therefore not scoring-function-independent. The panel is nine structurally "
            "diverse DOCKSTRING targets, which speaks to the spectral claims and not to any "
            "ranking or experimental-recovery claim. No commercial scoring function (Glide, "
            "GOLD) was tested, and no result here shows that any scorer is accurate, that "
            "the shared axis is biologically meaningful, or that the residual carries "
            "target-resolved mechanism."
        ),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    terms.atomic_csv(args.output_dir / "scorer_spectral_metrics.csv", metrics_frame)
    # summary.json is checksum-registered in results/manuscript_source_manifest.csv, so it
    # must be byte-identical across reruns. Wall-clock timings are collected for operational
    # use but never serialised here: they would make every rerun fail `verify_inputs.py`.
    terms.atomic_json(args.output_dir / "summary.json", drop_wall_clock(summary))

    score_columns = [name for name in ordered if name != "released_vina"]
    if score_columns:
        long_rows = []
        for name in ordered:
            for row_index, key in enumerate(block.inchikey.astype(str)):
                for column_index, target in enumerate(TARGETS):
                    long_rows.append(
                        {
                            "scorer": name,
                            "key": key,
                            "target": target,
                            "score": matrices[name][row_index, column_index]
                            * SCORER_POLARITY[name],
                        }
                    )
        long_frame = pd.DataFrame(long_rows)
        long_frame.to_csv(
            args.output_dir / "fixed_pose_scores.csv.gz",
            index=False,
            compression={"method": "gzip", "mtime": 0},
        )

    summary = finalize_existing_release(args.output_dir)

    print(json.dumps({"status": summary["status"], "support": summary["support"], "prediction_summary": summary["prediction_summary"]}, indent=2))


if __name__ == "__main__":
    main()
