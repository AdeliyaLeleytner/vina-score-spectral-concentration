#!/usr/bin/env python3
"""Fixed-pose decomposition of a dense DOCKSTRING target panel.

The public DOCKSTRING pose archives contain the top Vina pose for every
ligand--target cell.  Vina 1.1.2 ``--score_only`` reports the five classical
intermolecular potentials before weighting.  This script reconstructs the
weighted, post-normalisation contribution of every potential on the *same*
pose, then asks which terms support the shared ligand axis and which support
the two-way-centred residual modes.

This is an attribution of scores at retained poses.  It is deliberately not a
causal re-docking ablation: changing a weight during search could select a
different pose.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import itertools
import json
import lzma
import math
import os
import re
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, Descriptors, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from scipy import stats


RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "results" / "dockstring_vina_terms"
DEFAULT_POSE_DIR = Path("/private/tmp")
DEFAULT_DATASET = ROOT / "data" / "frozen" / "dockstring-dataset.tsv.gz"
DEFAULT_CROSS_PANEL_FINGERPRINT = (
    ROOT / "results" / "residual_mechanism" / "cross_panel_descriptor_fingerprint.csv"
)
DEFAULT_VINA = (
    ROOT
    / ".venv/lib/python3.12/site-packages/dockstring/resources/bin/vina_mac_catalina"
)
DEFAULT_OBABEL = ROOT / "downloads/vina_qc_toolchain/openbabel-env/bin/obabel"
DEFAULT_TARGET_DIR = (
    ROOT / ".venv/lib/python3.12/site-packages/dockstring/resources/targets"
)

TARGETS = (
    "NR3C1",
    "ADRB2",
    "ADRB1",
    "ADORA2A",
    "AR",
    "PTGS2",
    "ACHE",
    "DRD2",
    "LCK",
)

FIGSHARE_DOI = "10.6084/m9.figshare.16511577.v1"
POSE_ARCHIVES = {
    "ACHE": ("35948087", "edc3b91759d2bfffb81960b157c3dac2"),
    "ADORA2A": ("35948093", "2f24eed4245e23b415a6acea757138f6"),
    "ADRB1": ("35948096", "914ac7c3b100e731f04746476a6bf4ec"),
    "ADRB2": ("35948099", "d93ab68c4bfd32c5c3c9359e7e96f239"),
    "AR": ("35948108", "6c782e184a2de5a762fa7385452d0c2c"),
    "DRD2": ("35948144", "96712bd62173e274367b2bb150b670ac"),
    "LCK": ("35948216", "49229ce457f578763dfe7c9b58b6462b"),
    "NR3C1": ("35948243", "d5dfaa591ded3a5cf66c87fb9996ef81"),
    "PTGS2": ("35948267", "c774ede532ce42643a2b65c39d456500"),
}

TERM_WEIGHTS = {
    "gauss1": -0.035579,
    "gauss2": -0.005156,
    "repulsion": 0.840245,
    "hydrophobic": -0.035069,
    "hydrogen": -0.587439,
}
ROT_WEIGHT = 0.05846
PRIMARY_SUPPORT_SIZE = 15_000
PRIMARY_SUPPORT_SEED = 71
TERM_SUPPORT_SEED = 202_608_02
DEFAULT_SAMPLE_SIZE = 512
NESTED_SUPPORTS = (64, 128, 256, 512)
CROSS_PANEL_DESCRIPTOR_NAMES = (
    "heavy_atoms",
    "molecular_weight",
    "labute_asa",
    "tpsa",
    "clogp",
    "rotatable_bonds",
    "ring_count",
)
DIFFERENT_RECEPTOR_STRUCTURE_TARGETS = (
    "NR3C1",
    "ADRB2",
    "ADORA2A",
    "AR",
    "ACHE",
    "LCK",
)

AFFINITY_PATTERN = re.compile(
    r"^Affinity:\s+([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)", re.MULTILINE
)
TERM_PATTERNS = {
    "gauss1": re.compile(r"^\s*gauss 1\s*:\s*([-+0-9.eE]+)", re.MULTILINE),
    "gauss2": re.compile(r"^\s*gauss 2\s*:\s*([-+0-9.eE]+)", re.MULTILINE),
    "repulsion": re.compile(r"^\s*repulsion\s*:\s*([-+0-9.eE]+)", re.MULTILINE),
    "hydrophobic": re.compile(r"^\s*hydrophobic\s*:\s*([-+0-9.eE]+)", re.MULTILINE),
    "hydrogen": re.compile(r"^\s*Hydrogen\s*:\s*([-+0-9.eE]+)", re.MULTILINE),
}
KEY_PATTERN = re.compile(r">\s*<key>[^\n]*\n\s*([^\s]+)", re.IGNORECASE)
TORSDOF_PATTERN = re.compile(r"^TORSDOF\s+(\d+)", re.MULTILINE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--pose-dir", type=Path, default=DEFAULT_POSE_DIR)
    parser.add_argument("--target-dir", type=Path, default=DEFAULT_TARGET_DIR)
    parser.add_argument("--vina", type=Path, default=DEFAULT_VINA)
    parser.add_argument("--obabel", type=Path, default=DEFAULT_OBABEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    parser.add_argument("--factor-permutations", type=int, default=1000)
    parser.add_argument("--score-only", action="store_true")
    parser.add_argument("--analysis-only", action="store_true")
    parser.add_argument(
        "--finalize-existing",
        action="store_true",
        help=(
            "build the SI-ready term table and release metadata from the already "
            "frozen outputs without requiring external pose archives or executables"
        ),
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def md5_file(path: Path) -> str:
    digest = hashlib.md5()  # noqa: S324 - required to verify upstream manifest
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def si_ready_term_summary(
    attribution: pd.DataFrame,
    ablations: pd.DataFrame,
    spectral: pd.DataFrame,
) -> pd.DataFrame:
    """Build the compact, fixed-pose term table intended for the SI.

    The linear inner-product shares add to one but are not non-negative variance
    fractions.  The ablations remove a term while holding the Vina-selected pose
    fixed; they therefore describe score-function attribution, not the result of
    a new docking search.
    """

    attribution = attribution.set_index("term")
    ablations = ablations.set_index("removed_term")
    spectral = spectral.set_index("component")
    baseline = spectral.loc["rescored_post_total"]
    rows: list[dict[str, Any]] = []
    for term, weight in TERM_WEIGHTS.items():
        term_attribution = attribution.loc[term]
        removal = ablations.loc[term]
        rows.append(
            {
                "term": term,
                "vina_1_1_2_weight": weight,
                "shared_row_axis_covariance_share": float(
                    term_attribution.shared_row_axis_covariance_share
                ),
                "raw_pc1_bilinear_share": float(
                    term_attribution.raw_pc1_bilinear_share
                ),
                "residual_frobenius_inner_product_share": float(
                    term_attribution.residual_frobenius_inner_product_share
                ),
                "residual_pc1_bilinear_share": float(
                    term_attribution.residual_pc1_bilinear_share
                ),
                "full_score_raw_pr": float(baseline.raw_pr),
                "fixed_pose_term_removal_raw_pr": float(removal.raw_pr),
                "fixed_pose_term_removal_raw_pr_minus_full": float(
                    removal.raw_pr - baseline.raw_pr
                ),
                "full_score_residual_pr": float(baseline.residual_pr),
                "fixed_pose_term_removal_residual_pr": float(removal.residual_pr),
                "fixed_pose_term_removal_residual_pr_minus_full": float(
                    removal.residual_pr - baseline.residual_pr
                ),
                "analysis_boundary": (
                    "linear score attribution and term removal on fixed "
                    "Vina-selected poses; not an independent re-docking ablation"
                ),
            }
        )
    return pd.DataFrame(rows)


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


def finalize_existing_release(output: Path) -> dict[str, Any]:
    """Finalize tracked release artifacts without source-restricted rescoring."""

    output = output.resolve()
    summary_path = output / "analysis_summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(summary_path)
    summary = json.loads(summary_path.read_text())
    attribution = pd.read_csv(output / "term_attribution.csv")
    ablations = pd.read_csv(output / "fixed_pose_term_ablations.csv")
    spectral = pd.read_csv(output / "term_spectral_metrics.csv")
    si_table = si_ready_term_summary(attribution, ablations, spectral)
    atomic_csv(output / "si_vina_term_summary.csv", si_table)

    summary = _sanitize_portable_paths(summary)
    summary["claim_boundary"] = (
        "The five-term decomposition is an exact linear attribution of Vina 1.1.2 "
        "scores reconstructed on 512 ligands and nine targets after SDF-to-PDBQT "
        "pose roundtripping. Term shares may be negative and are not independent "
        "variance fractions. Every removal keeps the Vina-selected pose fixed; no "
        "result estimates which pose would be selected if a term were removed during "
        "search, and no result establishes docking accuracy or biological mechanism."
    )
    summary["reproducibility_boundary"] = {
        "analysis_from_frozen_terms": (
            "The SI table and all downstream attribution summaries are reproducible "
            "from results/dockstring_vina_terms/cell_terms.csv.gz and "
            "selected_ligands.csv in the release environment."
        ),
        "source_restricted_upstream": (
            "Recreating cell_terms.csv.gz additionally requires the external Figshare "
            "DOCKSTRING top-pose archives, Dockstring receptor resources, the exact "
            "Vina 1.1.2 and Open Babel executables identified by hashes, and the "
            "documented SDF-to-PDBQT roundtrip. It is outside the standard build."
        ),
        "pose_selection": "top poses selected by Vina and held fixed",
        "independent_redocking_performed": False,
        "si_ready_table": "results/dockstring_vina_terms/si_vina_term_summary.csv",
    }
    summary["si_ready_term_table"] = si_table.set_index("term").to_dict("index")
    tracked = sorted(
        path
        for path in output.iterdir()
        if path.is_file()
        and path.name not in {"analysis_summary.json", "output_checksums.json"}
    )
    summary["frozen_output_contract"] = {
        "algorithm": "sha256",
        "files_excluding_summary_and_checksum_manifest": {
            path.name: sha256_file(path) for path in tracked
        },
    }
    atomic_json(summary_path, summary)
    return summary


def command_environment() -> dict[str, str]:
    environment = dict(os.environ)
    environment.update(
        {
            "LC_ALL": "C",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        }
    )
    return environment


def run_command(command: list[str], timeout: float = 120) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        text=True,
        capture_output=True,
        env=command_environment(),
        timeout=timeout,
    )


def parse_vina_score_only(text: str) -> dict[str, float]:
    affinity_match = AFFINITY_PATTERN.search(text)
    if affinity_match is None:
        raise ValueError("Vina score-only output lacks Affinity")
    output = {"affinity": float(affinity_match.group(1))}
    for term, pattern in TERM_PATTERNS.items():
        match = pattern.search(text)
        if match is None:
            raise ValueError(f"Vina score-only output lacks {term}")
        output[term] = float(match.group(1))
    if not all(math.isfinite(value) for value in output.values()):
        raise ValueError("Vina score-only output contains a non-finite value")
    return output


def iter_sdf_records(path: Path) -> Iterable[str]:
    with lzma.open(path, "rt", encoding="utf-8", errors="replace") as handle:
        buffer: list[str] = []
        for line in handle:
            buffer.append(line)
            if line.strip() == "$$$$":
                yield "".join(buffer)
                buffer.clear()
        if any(value.strip() for value in buffer):
            raise ValueError(f"unterminated SDF record in {path}")


def extract_selected_records(path: Path, keys: set[str]) -> dict[str, str]:
    selected: dict[str, str] = {}
    for record in iter_sdf_records(path):
        match = KEY_PATTERN.search(record)
        if match is None:
            raise ValueError(f"pose archive record lacks key: {path}")
        key = match.group(1)
        if key in keys:
            if key in selected:
                raise ValueError(f"duplicate selected key in {path}: {key}")
            selected[key] = record
    missing = keys - set(selected)
    if missing:
        raise ValueError(f"{path} lacks {len(missing)} selected keys")
    return selected


def select_ligands(dataset_path: Path, sample_size: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = pd.read_csv(dataset_path, sep="\t")
    numeric = frame[list(TARGETS)].apply(pd.to_numeric, errors="coerce")
    complete = ~numeric.isna().any(axis=1)
    complete_frame = frame.loc[complete, ["inchikey", "smiles", *TARGETS]].reset_index()
    if PRIMARY_SUPPORT_SIZE > len(complete_frame):
        raise ValueError("primary support exceeds complete DOCKSTRING rows")
    if not 1 <= sample_size <= PRIMARY_SUPPORT_SIZE:
        raise ValueError("sample size must lie within the 15,000-ligand primary support")
    primary_rng = np.random.default_rng(PRIMARY_SUPPORT_SEED)
    primary_positions = np.sort(
        primary_rng.choice(len(complete_frame), PRIMARY_SUPPORT_SIZE, replace=False)
    )
    term_rng = np.random.default_rng(TERM_SUPPORT_SEED)
    term_order = term_rng.choice(PRIMARY_SUPPORT_SIZE, sample_size, replace=False)
    chosen_positions = primary_positions[term_order]
    selected = complete_frame.iloc[chosen_positions].copy().reset_index(drop=True)
    selected.insert(0, "selection_rank", np.arange(1, sample_size + 1))
    if selected.inchikey.duplicated().any():
        raise ValueError("selected DOCKSTRING InChIKeys are not unique")
    return frame, selected


def receptor_path(target_dir: Path, target: str) -> Path:
    path = target_dir / f"{target}_target.pdbqt"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def pose_archive_path(pose_dir: Path, target: str) -> Path:
    candidates = [
        pose_dir / f"dockstring_{target}.sdf.xz",
        pose_dir / f"{target}.sdf.xz",
    ]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(candidates[0])


def score_cell(
    target: str,
    key: str,
    smiles: str,
    released_score: float,
    record: str,
    receptor: Path,
    vina: Path,
    obabel: Path,
) -> dict[str, Any]:
    started = time.perf_counter()
    base: dict[str, Any] = {
        "target": target,
        "inchikey": key,
        "smiles": smiles,
        "released_score": released_score,
        "status": "",
        "error": "",
    }
    try:
        with tempfile.TemporaryDirectory(prefix="dockstring_term_") as temporary:
            temporary_path = Path(temporary)
            sdf = temporary_path / "pose.sdf"
            pdbqt = temporary_path / "pose.pdbqt"
            sdf.write_text(record)
            conversion = run_command(
                [
                    str(obabel),
                    "-isdf",
                    str(sdf),
                    "-opdbqt",
                    "-O",
                    str(pdbqt),
                    "--partialcharge",
                    "gasteiger",
                    "-h",
                ]
            )
            if conversion.returncode or not pdbqt.is_file() or not pdbqt.stat().st_size:
                raise RuntimeError(
                    "Open Babel conversion failed: "
                    + (conversion.stderr or conversion.stdout).strip()[-1000:]
                )
            pdbqt_text = pdbqt.read_text(errors="replace")
            torsdof_match = TORSDOF_PATTERN.search(pdbqt_text)
            if torsdof_match is None:
                raise ValueError("converted PDBQT lacks TORSDOF")
            common = [
                str(vina),
                "--receptor",
                str(receptor),
                "--ligand",
                str(pdbqt),
                "--score_only",
                "--cpu",
                "1",
            ]
            default = run_command(common)
            if default.returncode:
                raise RuntimeError((default.stderr or default.stdout).strip()[-1000:])
            no_rot = run_command([*common, "--weight_rot", "0"])
            if no_rot.returncode:
                raise RuntimeError((no_rot.stderr or no_rot.stdout).strip()[-1000:])
            default_score = parse_vina_score_only(default.stdout)
            no_rot_score = parse_vina_score_only(no_rot.stdout)
            maximum_term_repeat_error = max(
                abs(default_score[term] - no_rot_score[term]) for term in TERM_WEIGHTS
            )
            weighted = {
                term: no_rot_score[term] * weight for term, weight in TERM_WEIGHTS.items()
            }
            weighted_sum = float(sum(weighted.values()))
            if abs(weighted_sum) < 1e-12 or abs(default_score["affinity"]) < 1e-12:
                raise ValueError("degenerate affinity prevents normalisation audit")
            display_correction = no_rot_score["affinity"] / weighted_sum
            normalisation_factor = no_rot_score["affinity"] / default_score["affinity"]
            effective_num_tors = (normalisation_factor - 1.0) / ROT_WEIGHT
            base.update(
                {
                    "status": "success",
                    "rescored_affinity": default_score["affinity"],
                    "zero_rot_affinity": no_rot_score["affinity"],
                    "normalisation_factor": normalisation_factor,
                    "effective_num_tors": effective_num_tors,
                    "pdbqt_torsdof": int(torsdof_match.group(1)),
                    "maximum_term_repeat_error": maximum_term_repeat_error,
                    "weighted_sum_minus_zero_rot_affinity": (
                        weighted_sum - no_rot_score["affinity"]
                    ),
                    "display_rounding_correction": display_correction,
                }
            )
            for term in TERM_WEIGHTS:
                base[f"unweighted_{term}"] = no_rot_score[term]
                base[f"weighted_pre_{term}"] = weighted[term] * display_correction
                base[f"weighted_post_{term}"] = (
                    weighted[term] * display_correction / normalisation_factor
                )
    except Exception as error:  # preserve a complete failure ledger
        base["status"] = "failed"
        base["error"] = f"{type(error).__name__}: {error}"
    base["wall_seconds"] = time.perf_counter() - started
    return base


def score_panel(args: argparse.Namespace, selected: pd.DataFrame) -> pd.DataFrame:
    output = args.output_dir.resolve()
    result_path = output / "cell_terms.csv.gz"
    existing = pd.read_csv(result_path) if result_path.exists() else pd.DataFrame()
    completed = (
        set(zip(existing.target, existing.inchikey)) if len(existing) else set()
    )
    rows = existing.to_dict("records") if len(existing) else []
    wanted_keys = set(selected.inchikey.astype(str))
    metadata = selected.set_index("inchikey")
    for target in TARGETS:
        archive = pose_archive_path(args.pose_dir.resolve(), target)
        expected_md5 = POSE_ARCHIVES[target][1]
        observed_md5 = md5_file(archive)
        if observed_md5 != expected_md5:
            raise ValueError(f"pose archive MD5 mismatch for {target}")
        records = extract_selected_records(archive, wanted_keys)
        receptor = receptor_path(args.target_dir.resolve(), target)
        tasks = []
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            for key, row in metadata.iterrows():
                if (target, key) in completed:
                    continue
                tasks.append(
                    executor.submit(
                        score_cell,
                        target,
                        str(key),
                        str(row.smiles),
                        float(row[target]),
                        records[str(key)],
                        receptor,
                        args.vina.resolve(),
                        args.obabel.resolve(),
                    )
                )
            for future in as_completed(tasks):
                rows.append(future.result())
        frame = pd.DataFrame(rows).sort_values(["target", "inchikey"]).reset_index(drop=True)
        output.mkdir(parents=True, exist_ok=True)
        temporary = result_path.with_suffix(".csv.gz.tmp")
        frame.to_csv(
            temporary,
            index=False,
            compression={"method": "gzip", "mtime": 0},
        )
        os.replace(temporary, result_path)
        counts = frame.loc[frame.target.eq(target), "status"].value_counts().to_dict()
        print(f"{target}: {counts}", flush=True)
    return pd.read_csv(result_path)


def two_way_center(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    return matrix - matrix.mean(0) - matrix.mean(1)[:, None] + matrix.mean()


def standardize_columns(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    scale = matrix.std(0, ddof=1)
    if np.any(scale <= 1e-12):
        raise ValueError("matrix contains a near-constant target")
    return (matrix - matrix.mean(0)) / scale


def correlation_pr(matrix: np.ndarray) -> float:
    z = standardize_columns(matrix)
    correlation = z.T @ z / (len(z) - 1)
    return float(np.trace(correlation) ** 2 / np.square(correlation).sum())


def spectral_summary(matrix: np.ndarray) -> dict[str, float]:
    raw_z = standardize_columns(matrix)
    raw_corr = raw_z.T @ raw_z / (len(raw_z) - 1)
    residual = two_way_center(matrix)
    residual_z = standardize_columns(residual)
    residual_corr = residual_z.T @ residual_z / (len(residual_z) - 1)
    raw_eigen = np.linalg.eigvalsh(raw_corr)[::-1]
    residual_eigen = np.linalg.eigvalsh(residual_corr)[::-1]
    upper = np.triu_indices(matrix.shape[1], 1)
    return {
        "raw_pr": float(raw_eigen.sum() ** 2 / np.square(raw_eigen).sum()),
        "residual_pr": float(
            residual_eigen.sum() ** 2 / np.square(residual_eigen).sum()
        ),
        "pr_increase": float(
            residual_eigen.sum() ** 2 / np.square(residual_eigen).sum()
            - raw_eigen.sum() ** 2 / np.square(raw_eigen).sum()
        ),
        "raw_pc1_fraction": float(raw_eigen[0] / raw_eigen.sum()),
        "residual_pc1_fraction": float(residual_eigen[0] / residual_eigen.sum()),
        "raw_mean_correlation": float(raw_corr[upper].mean()),
        "residual_mean_absolute_correlation": float(np.abs(residual_corr[upper]).mean()),
        "residual_fraction_grand_centered_ss": float(
            np.square(residual).sum() / np.square(matrix - matrix.mean()).sum()
        ),
        "z_then_row_center_pr": correlation_pr(raw_z - raw_z.mean(1)[:, None]),
    }


def pivot_matrix(frame: pd.DataFrame, value: str, selected: pd.DataFrame) -> np.ndarray:
    order = selected.sort_values("selection_rank").inchikey.astype(str).tolist()
    pivot = frame.pivot(index="inchikey", columns="target", values=value)
    pivot = pivot.reindex(index=order, columns=list(TARGETS))
    if pivot.isna().any().any():
        raise ValueError(f"non-rectangular term matrix: {value}")
    return pivot.to_numpy(dtype=np.float64)


def component_attribution(
    total: np.ndarray,
    components: dict[str, np.ndarray],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    total_scale = total.std(0, ddof=1)
    z_total = (total - total.mean(0)) / total_scale
    z_components = {
        name: (value - value.mean(0)) / total_scale for name, value in components.items()
    }
    u, singular, vt = np.linalg.svd(z_total, full_matrices=False)
    row_total = z_total.mean(1)
    row_variance = float(np.var(row_total, ddof=0))

    residual_total_raw = two_way_center(total)
    residual_scale = residual_total_raw.std(0, ddof=1)
    residual_total = residual_total_raw / residual_scale
    residual_components = {
        name: two_way_center(value) / residual_scale for name, value in components.items()
    }
    ru, rsingular, rvt = np.linalg.svd(residual_total, full_matrices=False)
    residual_norm = float(np.square(residual_total).sum())

    rows: list[dict[str, Any]] = []
    modes: list[dict[str, Any]] = []
    for name, value in components.items():
        row_component = z_components[name].mean(1)
        rows.append(
            {
                "term": name,
                "shared_row_axis_covariance_share": float(
                    np.cov(row_component, row_total, ddof=0)[0, 1] / row_variance
                ),
                "raw_pc1_bilinear_share": float(
                    u[:, 0] @ z_components[name] @ vt[0] / singular[0]
                ),
                "residual_frobenius_inner_product_share": float(
                    np.sum(residual_components[name] * residual_total) / residual_norm
                ),
                "residual_pc1_bilinear_share": float(
                    ru[:, 0] @ residual_components[name] @ rvt[0] / rsingular[0]
                ),
            }
        )
        for mode in range(min(5, total.shape[1] - 1)):
            modes.append(
                {
                    "term": name,
                    "residual_mode": mode + 1,
                    "mode_variance_fraction": float(
                        rsingular[mode] ** 2 / np.square(rsingular).sum()
                    ),
                    "bilinear_share": float(
                        ru[:, mode]
                        @ residual_components[name]
                        @ rvt[mode]
                        / rsingular[mode]
                    ),
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(modes)


def molecule_descriptors(selected: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    rows: list[dict[str, Any]] = []
    groups: list[str] = []
    for index, smiles in enumerate(selected.sort_values("selection_rank").smiles.astype(str)):
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            raise ValueError(f"invalid selected SMILES at {index}")
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=molecule)
        groups.append(scaffold if scaffold else f"acyclic:{index}")
        rows.append(
            {
                "heavy_atoms": molecule.GetNumHeavyAtoms(),
                "molecular_weight": Descriptors.MolWt(molecule),
                "labute_asa": rdMolDescriptors.CalcLabuteASA(molecule),
                "clogp": Crippen.MolLogP(molecule),
                "tpsa": Descriptors.TPSA(molecule),
                "hbd": rdMolDescriptors.CalcNumHBD(molecule),
                "hba": rdMolDescriptors.CalcNumHBA(molecule),
                "rotatable_bonds": Descriptors.NumRotatableBonds(molecule),
                "rdkit_rotatable_bonds": Descriptors.NumRotatableBonds(molecule),
                "ring_count": rdMolDescriptors.CalcNumRings(molecule),
            }
        )
    return pd.DataFrame(rows), np.asarray(groups, dtype=object)


def descriptor_correlations(
    matrices: dict[str, np.ndarray], descriptors: pd.DataFrame
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for component, matrix in matrices.items():
        row_mean = matrix.mean(1)
        residual_rms = np.sqrt(np.mean(np.square(two_way_center(matrix)), axis=1))
        for descriptor in descriptors:
            values = descriptors[descriptor].to_numpy(float)
            for statistic, response in (("row_mean", row_mean), ("residual_rms", residual_rms)):
                result = stats.spearmanr(values, response)
                rows.append(
                    {
                        "component": component,
                        "response": statistic,
                        "descriptor": descriptor,
                        "spearman_rho": float(result.statistic),
                        "p_value_descriptive_only": float(result.pvalue),
                    }
                )
    return pd.DataFrame(rows)


def heavy_atom_slope_decomposition(
    matrices: dict[str, np.ndarray], descriptors: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Describe common and target-specific molecular-size slopes.

    For every matrix, each target column is projected separately onto centered
    heavy-atom count.  Double centering of that projection is exactly the
    rank-one interaction ``(size_i - mean(size)) * (slope_j - mean(slope))``.
    The resulting residual R2 therefore quantifies one concrete mechanism by
    which a shared size axis can coexist with target-specific residual modes.
    These fits are descriptive rather than causal or externally predictive.
    """

    heavy_atoms = descriptors["heavy_atoms"].to_numpy(dtype=float)
    centered_size = heavy_atoms - heavy_atoms.mean()
    denominator = float(centered_size @ centered_size)
    if denominator <= 0:
        raise ValueError("heavy-atom count has zero variance")
    summaries: list[dict[str, Any]] = []
    target_rows: list[dict[str, Any]] = []
    for component, matrix in matrices.items():
        slopes = centered_size @ matrix / denominator
        prediction = np.outer(centered_size, slopes)
        column_centered = matrix - matrix.mean(0)
        residual = two_way_center(matrix)
        predicted_residual = two_way_center(prediction)
        raw_r2 = 1.0 - float(
            np.square(column_centered - prediction).sum()
            / np.square(column_centered).sum()
        )
        residual_r2 = 1.0 - float(
            np.square(residual - predicted_residual).sum()
            / np.square(residual).sum()
        )
        residual_z = standardize_columns(residual)
        u, singular, _ = np.linalg.svd(residual_z, full_matrices=False)
        pc1_scores = u[:, 0] * singular[0]
        pc1_slope = float(centered_size @ pc1_scores / denominator)
        predicted_pc1 = centered_size * pc1_slope
        pc1_r2 = 1.0 - float(
            np.square(pc1_scores - predicted_pc1).sum()
            / np.square(pc1_scores - pc1_scores.mean()).sum()
        )
        summaries.append(
            {
                "component": component,
                "descriptor": "heavy_atoms",
                "analysis_role": "descriptive_linear_projection",
                "raw_column_centered_r2": raw_r2,
                "two_way_residual_r2": residual_r2,
                "residual_pc1_score_r2": pc1_r2,
                "mean_target_slope_per_heavy_atom": float(slopes.mean()),
                "sd_target_slope_per_heavy_atom": float(slopes.std(ddof=1)),
                "minimum_target_slope_per_heavy_atom": float(slopes.min()),
                "maximum_target_slope_per_heavy_atom": float(slopes.max()),
                "negative_target_slopes": int((slopes < 0).sum()),
                "positive_target_slopes": int((slopes > 0).sum()),
            }
        )
        for target, slope in zip(TARGETS, slopes):
            target_rows.append(
                {
                    "component": component,
                    "target": target,
                    "slope_kcal_per_mol_per_heavy_atom": float(slope),
                    "slope_deviation_from_panel_mean": float(slope - slopes.mean()),
                }
            )
    return pd.DataFrame(summaries), pd.DataFrame(target_rows)


def quality_support_sensitivity(
    total: np.ndarray,
    released: np.ndarray,
    pre_total: np.ndarray,
    effective_tors: np.ndarray,
    post_terms: dict[str, np.ndarray],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Check that pose-roundtrip or graph anomalies do not drive conclusions."""

    graph_consistent = np.ptp(effective_tors, axis=1) <= 0.01
    score_reproduced = np.max(np.abs(total - released), axis=1) <= 0.10
    supports = {
        "all_selected_ligands": np.ones(len(total), dtype=bool),
        "target_invariant_effective_torsions": graph_consistent,
        "released_scores_reproduced_within_0.10_kcal_mol": score_reproduced,
        "both_quality_rules": graph_consistent & score_reproduced,
    }
    representations = {
        "released_score": released,
        "rescored_post_total": total,
        "zero_rot_pre_total": pre_total,
    }
    spectral_rows: list[dict[str, Any]] = []
    attribution_rows: list[dict[str, Any]] = []
    support_summary: dict[str, Any] = {}
    for support, mask in supports.items():
        support_summary[support] = {
            "n_ligands": int(mask.sum()),
            "excluded_ligands": int((~mask).sum()),
        }
        for representation, matrix in representations.items():
            spectral_rows.append(
                {
                    "support": support,
                    "n_ligands": int(mask.sum()),
                    "representation": representation,
                    **spectral_summary(matrix[mask]),
                }
            )
        attribution, _ = component_attribution(
            total[mask], {term: value[mask] for term, value in post_terms.items()}
        )
        attribution.insert(0, "n_ligands", int(mask.sum()))
        attribution.insert(0, "support", support)
        attribution_rows.extend(attribution.to_dict("records"))
    rules = {
        "effective_torsion_rule": (
            "within-ligand max minus min effective Vina torsion count across targets "
            "must be <=0.01"
        ),
        "score_roundtrip_rule": (
            "maximum absolute rescored-minus-released score across targets must be "
            "<=0.10 kcal/mol; released scores are displayed to 0.1 kcal/mol"
        ),
        "supports": support_summary,
    }
    return pd.DataFrame(spectral_rows), pd.DataFrame(attribution_rows), rules


def _normalised_rank_columns(matrix: np.ndarray) -> np.ndarray:
    ranks = np.column_stack(
        [stats.rankdata(matrix[:, column]) for column in range(matrix.shape[1])]
    ).astype(float)
    ranks -= ranks.mean(0)
    norms = np.sqrt(np.square(ranks).sum(0))
    if np.any(norms <= 0):
        raise ValueError("fingerprint contains a constant descriptor column")
    return ranks / norms


def cross_panel_term_fingerprint(
    matrices: dict[str, np.ndarray],
    descriptors: pd.DataFrame,
    reference_path: Path = DEFAULT_CROSS_PANEL_FINGERPRINT,
    ligand_support_label: str = "all_selected_ligands",
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Localise the replicated Docking-44/DOCKSTRING target fingerprint by term.

    The Docking-44 reference slopes come from the independently generated,
    full-support residual-mechanism analysis.  DOCKSTRING term slopes are
    estimated on the fixed 512-ligand pose subset.  Target labels are permuted
    exactly; max-|statistic| adjustment controls the five-term family.  This is
    an exploratory localisation of an already observed cross-panel signal.
    """

    if not reference_path.is_file():
        return (
            pd.DataFrame(),
            pd.DataFrame(),
            {
                "status": "not_run_missing_cross_panel_reference",
                "reference_path": str(reference_path),
            },
        )
    reference = pd.read_csv(reference_path)
    supports = {
        "all_9_mapped_targets": tuple(TARGETS),
        "different_receptor_structure_only_6": DIFFERENT_RECEPTOR_STRUCTURE_TARGETS,
    }
    summary_rows: list[dict[str, Any]] = []
    descriptor_rows: list[dict[str, Any]] = []
    json_supports: dict[str, Any] = {}
    for support_name, support_targets in supports.items():
        target_indices = [TARGETS.index(target) for target in support_targets]
        subset = reference.loc[
            reference.chemical_support_sensitivity.eq("primary_seeded_support")
            & reference.target_sensitivity.eq(support_name)
            & reference.residual_scaling.eq("unscaled_two_way_residual")
        ]
        expected = len(support_targets) * len(CROSS_PANEL_DESCRIPTOR_NAMES)
        if len(subset) != expected:
            raise ValueError(
                f"cross-panel reference {support_name} has {len(subset)} rows; "
                f"expected {expected}"
            )
        docking44 = np.empty(
            (len(support_targets), len(CROSS_PANEL_DESCRIPTOR_NAMES)), dtype=float
        )
        for target_index, target in enumerate(support_targets):
            for descriptor_index, descriptor in enumerate(
                CROSS_PANEL_DESCRIPTOR_NAMES
            ):
                values = subset.loc[
                    subset.dockstring_target.eq(target)
                    & subset.descriptor.eq(descriptor),
                    "docking44_slope_per_descriptor_sd",
                ]
                if len(values) != 1:
                    raise ValueError(
                        f"ambiguous Docking-44 reference for {target}/{descriptor}"
                    )
                docking44[target_index, descriptor_index] = float(values.iloc[0])
        docking44_ranks = _normalised_rank_columns(docking44)
        permutations = np.asarray(
            list(itertools.permutations(range(len(support_targets)))), dtype=np.int16
        )
        term_ranks: dict[str, np.ndarray] = {}
        observed: dict[str, float] = {}
        term_slopes: dict[str, np.ndarray] = {}
        for component, full_matrix in matrices.items():
            residual = two_way_center(full_matrix[:, target_indices])
            slopes = np.empty_like(docking44)
            for descriptor_index, descriptor in enumerate(
                CROSS_PANEL_DESCRIPTOR_NAMES
            ):
                values = descriptors[descriptor].to_numpy(dtype=float)
                values = (values - values.mean()) / values.std(ddof=1)
                slopes[:, descriptor_index] = values @ residual / (values @ values)
            ranks = _normalised_rank_columns(slopes)
            term_ranks[component] = ranks
            term_slopes[component] = slopes
            observed[component] = float(np.sum(ranks * docking44_ranks, axis=0).mean())
        null_chunks: list[np.ndarray] = []
        component_names = list(matrices)
        for start in range(0, len(permutations), 10_000):
            indices = permutations[start : start + 10_000]
            null_chunks.append(
                np.column_stack(
                    [
                        np.einsum(
                            "ijk,jk->ik", term_ranks[name][indices], docking44_ranks
                        ).mean(1)
                        for name in component_names
                    ]
                )
            )
        null = np.vstack(null_chunks)
        term_columns = [
            index
            for index, name in enumerate(component_names)
            if name != "rescored_post_total"
        ]
        family_maximum = np.abs(null[:, term_columns]).max(1)
        support_payload: dict[str, Any] = {}
        for component_index, component in enumerate(component_names):
            statistic = observed[component]
            unadjusted = float(
                np.mean(np.abs(null[:, component_index]) >= abs(statistic) - 1e-14)
            )
            adjusted = (
                None
                if component == "rescored_post_total"
                else float(np.mean(family_maximum >= abs(statistic) - 1e-14))
            )
            summary_rows.append(
                {
                    "target_support": support_name,
                    "ligand_support": ligand_support_label,
                    "n_targets": len(support_targets),
                    "n_ligands": len(descriptors),
                    "component": component,
                    "mean_descriptor_spearman_rho": statistic,
                    "exact_target_label_permutations": len(permutations),
                    "exact_two_sided_p": unadjusted,
                    "max_abs_fwer_p_across_five_terms": adjusted,
                    "analysis_role": "exploratory_cross_panel_term_localisation",
                }
            )
            support_payload[component] = {
                "mean_descriptor_spearman_rho": statistic,
                "exact_two_sided_p": unadjusted,
                "max_abs_fwer_p_across_five_terms": adjusted,
            }
            for descriptor_index, descriptor in enumerate(
                CROSS_PANEL_DESCRIPTOR_NAMES
            ):
                rho = float(
                    np.sum(
                        term_ranks[component][:, descriptor_index]
                        * docking44_ranks[:, descriptor_index]
                    )
                )
                for target_index, target in enumerate(support_targets):
                    descriptor_rows.append(
                        {
                            "target_support": support_name,
                            "ligand_support": ligand_support_label,
                            "component": component,
                            "descriptor": descriptor,
                            "descriptor_spearman_rho": rho,
                            "dockstring_target": target,
                            "docking44_slope_per_descriptor_sd": docking44[
                                target_index, descriptor_index
                            ],
                            "dockstring_term_slope_per_descriptor_sd": term_slopes[
                                component
                            ][target_index, descriptor_index],
                        }
                    )
        json_supports[support_name] = {
            "targets": list(support_targets),
            "exact_target_label_permutations": len(permutations),
            "components": support_payload,
        }
    metadata = {
        "status": "exploratory_cross_panel_term_localisation_complete",
        "ligand_support": ligand_support_label,
        "reference_path": str(reference_path.resolve()),
        "reference_sha256": sha256_file(reference_path),
        "descriptor_names": list(CROSS_PANEL_DESCRIPTOR_NAMES),
        "multiplicity_control": (
            "exact target-label max-absolute-statistic adjustment across the five "
            "post-normalisation Vina terms within each target support"
        ),
        "claim_boundary": (
            "Localises a previously observed target-response fingerprint; fixed-pose "
            "term attribution is not binding validation or a redocking ablation"
        ),
        "supports": json_supports,
    }
    return pd.DataFrame(summary_rows), pd.DataFrame(descriptor_rows), metadata


def nested_support_metrics(
    matrices: dict[str, np.ndarray], maximum: int
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    supports = sorted({value for value in NESTED_SUPPORTS if value <= maximum} | {maximum})
    for size in supports:
        for component, matrix in matrices.items():
            rows.append({"n_ligands": size, "component": component, **spectral_summary(matrix[:size])})
    return pd.DataFrame(rows)


def bootstrap_key_metrics(
    total: np.ndarray,
    pre_total: np.ndarray,
    terms: dict[str, np.ndarray],
    groups: np.ndarray,
    replicates: int,
) -> dict[str, Any]:
    unique = np.unique(groups)
    members = {group: np.flatnonzero(groups == group) for group in unique}
    rng = np.random.default_rng(202_608_03)
    metrics: dict[str, list[float]] = {
        "post_raw_pr": [],
        "post_residual_pr": [],
        "post_pr_increase": [],
        "pre_raw_pr": [],
        "pre_residual_pr": [],
    }
    for term in terms:
        metrics[f"{term}_shared_row_axis_share"] = []
        metrics[f"{term}_residual_ss_share"] = []
    for _ in range(replicates):
        sampled = rng.choice(unique, len(unique), replace=True)
        index = np.concatenate([members[group] for group in sampled])
        post = spectral_summary(total[index])
        pre = spectral_summary(pre_total[index])
        metrics["post_raw_pr"].append(post["raw_pr"])
        metrics["post_residual_pr"].append(post["residual_pr"])
        metrics["post_pr_increase"].append(post["pr_increase"])
        metrics["pre_raw_pr"].append(pre["raw_pr"])
        metrics["pre_residual_pr"].append(pre["residual_pr"])
        attribution, _ = component_attribution(
            total[index], {term: value[index] for term, value in terms.items()}
        )
        attribution = attribution.set_index("term")
        for term in terms:
            metrics[f"{term}_shared_row_axis_share"].append(
                attribution.loc[term, "shared_row_axis_covariance_share"]
            )
            metrics[f"{term}_residual_ss_share"].append(
                attribution.loc[term, "residual_frobenius_inner_product_share"]
            )
    output: dict[str, Any] = {}
    for name, values in metrics.items():
        array = np.asarray(values, dtype=float)
        output[name] = {
            "median": float(np.median(array)),
            "ci95": [float(np.quantile(array, 0.025)), float(np.quantile(array, 0.975))],
        }
    return {
        "replicates": replicates,
        "seed": 202_608_03,
        "resampling_unit": "RDKit Bemis-Murcko scaffold; each acyclic ligand is a singleton",
        "n_clusters": int(len(unique)),
        "metrics": output,
    }


def factor_permutation_test(
    pre_total: np.ndarray,
    factor: np.ndarray,
    permutations: int,
) -> dict[str, Any]:
    factor = np.asarray(factor, dtype=float)
    observed = pre_total / factor[:, None]
    observed_summary = spectral_summary(observed)
    rng = np.random.default_rng(202_608_04)
    raw: list[float] = []
    residual: list[float] = []
    for _ in range(permutations):
        permuted = pre_total / rng.permutation(factor)[:, None]
        summary = spectral_summary(permuted)
        raw.append(summary["raw_pr"])
        residual.append(summary["residual_pr"])
    return {
        "permutations": permutations,
        "seed": 202_608_04,
        "observed": observed_summary,
        "permuted_factor_raw_pr": {
            "median": float(np.median(raw)),
            "interval95": [float(np.quantile(raw, 0.025)), float(np.quantile(raw, 0.975))],
        },
        "permuted_factor_residual_pr": {
            "median": float(np.median(residual)),
            "interval95": [
                float(np.quantile(residual, 0.025)),
                float(np.quantile(residual, 0.975)),
            ],
        },
    }


def analyse(
    args: argparse.Namespace,
    selected: pd.DataFrame,
    cells: pd.DataFrame,
) -> dict[str, Any]:
    failed = cells.loc[~cells.status.eq("success")]
    if len(failed):
        raise RuntimeError(f"{len(failed)} term cells failed; inspect cell_terms.csv.gz")
    expected = len(selected) * len(TARGETS)
    if len(cells) != expected:
        raise ValueError(f"expected {expected} term cells, found {len(cells)}")

    total = pivot_matrix(cells, "rescored_affinity", selected)
    released = pivot_matrix(cells, "released_score", selected)
    pre_total = pivot_matrix(cells, "zero_rot_affinity", selected)
    factors = pivot_matrix(cells, "normalisation_factor", selected)
    effective_tors = pivot_matrix(cells, "effective_num_tors", selected)
    factor = np.median(factors, axis=1)
    post_terms = {
        term: pivot_matrix(cells, f"weighted_post_{term}", selected)
        for term in TERM_WEIGHTS
    }
    pre_terms = {
        term: pivot_matrix(cells, f"weighted_pre_{term}", selected)
        for term in TERM_WEIGHTS
    }
    matrices = {
        "released_score": released,
        "rescored_post_total": total,
        "zero_rot_pre_total": pre_total,
        **{f"post_{term}": value for term, value in post_terms.items()},
        **{f"pre_{term}": value for term, value in pre_terms.items()},
    }

    spectral_rows = [
        {"component": name, **spectral_summary(matrix)} for name, matrix in matrices.items()
    ]
    spectral = pd.DataFrame(spectral_rows)
    attribution, modes = component_attribution(total, post_terms)
    ablations = []
    for term, matrix in post_terms.items():
        ablations.append(
            {"removed_term": term, **spectral_summary(total - matrix)}
        )
    descriptors, groups = molecule_descriptors(selected)
    descriptor_table = descriptor_correlations(matrices, descriptors)
    size_slopes, target_size_slopes = heavy_atom_slope_decomposition(
        matrices, descriptors
    )
    fingerprint_matrices = {
        "rescored_post_total": total,
        **{f"post_{term}": value for term, value in post_terms.items()},
    }
    cross_panel_summary, cross_panel_by_descriptor, cross_panel_metadata = (
        cross_panel_term_fingerprint(fingerprint_matrices, descriptors)
    )
    roundtrip_quality_mask = np.max(np.abs(total - released), axis=1) <= 0.10
    quality_cross_summary, quality_cross_by_descriptor, quality_cross_metadata = (
        cross_panel_term_fingerprint(
            {
                component: matrix[roundtrip_quality_mask]
                for component, matrix in fingerprint_matrices.items()
            },
            descriptors.loc[roundtrip_quality_mask].reset_index(drop=True),
            ligand_support_label="released_scores_reproduced_within_0.10_kcal_mol",
        )
    )
    if len(quality_cross_summary):
        cross_panel_summary = pd.concat(
            [cross_panel_summary, quality_cross_summary], ignore_index=True
        )
        cross_panel_by_descriptor = pd.concat(
            [cross_panel_by_descriptor, quality_cross_by_descriptor],
            ignore_index=True,
        )
        cross_panel_metadata["score_roundtrip_quality_sensitivity"] = (
            quality_cross_metadata
        )
    disjoint_support_metadata: dict[str, Any] = {}
    for label, index in (
        ("seeded_random_first_256", np.arange(0, 256)),
        ("seeded_random_second_256", np.arange(256, 512)),
    ):
        if len(descriptors) < 512:
            continue
        split_summary, split_by_descriptor, split_metadata = (
            cross_panel_term_fingerprint(
                {
                    component: matrix[index]
                    for component, matrix in fingerprint_matrices.items()
                },
                descriptors.iloc[index].reset_index(drop=True),
                ligand_support_label=label,
            )
        )
        if len(split_summary):
            cross_panel_summary = pd.concat(
                [cross_panel_summary, split_summary], ignore_index=True
            )
            cross_panel_by_descriptor = pd.concat(
                [cross_panel_by_descriptor, split_by_descriptor], ignore_index=True
            )
            disjoint_support_metadata[label] = split_metadata
    if disjoint_support_metadata:
        cross_panel_metadata["disjoint_ligand_support_sensitivity"] = (
            disjoint_support_metadata
        )
    nested = nested_support_metrics(matrices, len(selected))
    bootstrap = bootstrap_key_metrics(
        total,
        pre_total,
        post_terms,
        groups,
        args.bootstrap_replicates,
    )
    factor_null = factor_permutation_test(pre_total, factor, args.factor_permutations)
    quality_spectral, quality_attribution, quality_rules = quality_support_sensitivity(
        total, released, pre_total, effective_tors, post_terms
    )

    output = args.output_dir.resolve()
    atomic_csv(output / "term_spectral_metrics.csv", spectral)
    atomic_csv(output / "term_attribution.csv", attribution)
    atomic_csv(output / "residual_mode_term_attribution.csv", modes)
    atomic_csv(output / "fixed_pose_term_ablations.csv", pd.DataFrame(ablations))
    atomic_csv(output / "descriptor_correlations.csv", descriptor_table)
    atomic_csv(output / "heavy_atom_slope_decomposition.csv", size_slopes)
    atomic_csv(output / "target_heavy_atom_slopes.csv", target_size_slopes)
    if len(cross_panel_summary):
        atomic_csv(output / "cross_panel_term_fingerprint.csv", cross_panel_summary)
        atomic_csv(
            output / "cross_panel_term_fingerprint_by_descriptor.csv",
            cross_panel_by_descriptor,
        )
    atomic_csv(output / "nested_support_metrics.csv", nested)
    atomic_csv(output / "quality_support_spectral_sensitivity.csv", quality_spectral)
    atomic_csv(
        output / "quality_support_term_attribution.csv", quality_attribution
    )

    released_difference = total - released
    term_reconstruction = total - sum(post_terms.values())
    torsdof = pivot_matrix(cells, "pdbqt_torsdof", selected)
    summary = {
        "status": "exploratory_fixed_pose_attribution_complete",
        "scope": {
            "interpretation": (
                "Attribution of retained-pose Vina scores; not a causal re-docking ablation"
            ),
            "targets": list(TARGETS),
            "n_targets": len(TARGETS),
            "n_ligands": len(selected),
            "n_cells": len(cells),
            "figshare_doi": FIGSHARE_DOI,
            "sample_rule": (
                "simple random subset (seed 20260802) of the frozen 15,000-ligand "
                "DOCKSTRING support selected with NumPy seed 71"
            ),
            "ligand_conversion": (
                "Published top-pose SDF converted to PDBQT with Open Babel 3.1.0, "
                "Gasteiger charges, and explicit hydrogens added to the archived "
                "prepared graph, without a second pH-dependent protonation step"
            ),
            "pose_roundtrip_limit": (
                "The original docked ligand PDBQT torsion trees were not published; "
                "the archived SDF-to-PDBQT roundtrip is audited against released scores, "
                "and conclusions are repeated on a <=0.10 kcal/mol reproduction support"
            ),
        },
        "score_reproduction": {
            "rescored_vs_released_cell_pearson": float(
                stats.pearsonr(total.ravel(), released.ravel()).statistic
            ),
            "rescored_minus_released_mean": float(released_difference.mean()),
            "rescored_minus_released_mae": float(np.abs(released_difference).mean()),
            "rescored_minus_released_max_abs": float(np.abs(released_difference).max()),
            "cells_reproduced_within_0.10_kcal_mol": int(
                (np.abs(released_difference) <= 0.10).sum()
            ),
            "fraction_cells_reproduced_within_0.10_kcal_mol": float(
                np.mean(np.abs(released_difference) <= 0.10)
            ),
            "ligands_reproduced_within_0.10_kcal_mol_on_all_targets": int(
                (np.max(np.abs(released_difference), axis=1) <= 0.10).sum()
            ),
            "term_sum_max_abs_reconstruction_error": float(
                np.abs(term_reconstruction).max()
            ),
            "maximum_repeated_unweighted_term_error": float(
                cells.maximum_term_repeat_error.max()
            ),
            "weighted_display_sum_minus_zero_rot_max_abs": float(
                cells.weighted_sum_minus_zero_rot_affinity.abs().max()
            ),
            "display_rounding_correction_max_abs_deviation_from_one": float(
                np.abs(cells.display_rounding_correction - 1.0).max()
            ),
            "term_rounding_method": (
                "The five direct Vina 1.1.2 terms are printed to finite precision; "
                "a common per-cell multiplicative correction makes their weighted "
                "sum equal the zero-rot affinity before torsional normalisation"
            ),
        },
        "torsional_normalisation": {
            "factor_targetwise_max_sd_within_ligand": float(factors.std(1).max()),
            "factor_range": [float(factor.min()), float(factor.max())],
            "effective_num_tors_range": [
                float(effective_tors.min()),
                float(effective_tors.max()),
            ],
            "effective_num_tors_targetwise_max_sd_within_ligand": float(
                effective_tors.std(1).max()
            ),
            "pdbqt_torsdof_range": [float(torsdof.min()), float(torsdof.max())],
            "factor_permutation_null": factor_null,
        },
        "spectral_metrics": spectral.set_index("component").to_dict("index"),
        "term_attribution": attribution.set_index("term").to_dict("index"),
        "fixed_pose_ablations": pd.DataFrame(ablations)
        .set_index("removed_term")
        .to_dict("index"),
        "heavy_atom_slope_decomposition": size_slopes.set_index("component").to_dict(
            "index"
        ),
        "cross_panel_term_fingerprint": cross_panel_metadata,
        "quality_support_sensitivity": quality_rules,
        "cluster_bootstrap": bootstrap,
        "toolchain": {
            "vina_path": str(args.vina.resolve()),
            "vina_sha256": sha256_file(args.vina.resolve()),
            "vina_version": run_command([str(args.vina.resolve()), "--version"]).stdout.strip(),
            "obabel_path": str(args.obabel.resolve()),
            "obabel_sha256": sha256_file(args.obabel.resolve()),
            "obabel_version": run_command([str(args.obabel.resolve()), "-V"]).stdout.strip(),
            "dataset_sha256": sha256_file(args.dataset.resolve()),
            "pose_archive_md5": {
                target: md5_file(pose_archive_path(args.pose_dir.resolve(), target))
                for target in TARGETS
            },
            "receptor_sha256": {
                target: sha256_file(receptor_path(args.target_dir.resolve(), target))
                for target in TARGETS
            },
        },
    }
    atomic_json(output / "analysis_summary.json", summary)
    return finalize_existing_release(output)


def main() -> None:
    args = parse_args()
    if args.finalize_existing:
        summary = finalize_existing_release(args.output_dir)
        print(
            json.dumps(
                {
                    "status": summary["status"],
                    "si_ready_table": summary["reproducibility_boundary"][
                        "si_ready_table"
                    ],
                },
                indent=2,
            )
        )
        return
    if args.workers < 1:
        raise ValueError("workers must be positive")
    if args.score_only and args.analysis_only:
        raise ValueError("use at most one of --score-only and --analysis-only")
    for path in (args.dataset, args.vina, args.obabel, args.target_dir):
        if not Path(path).exists():
            raise FileNotFoundError(path)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _, selected = select_ligands(args.dataset.resolve(), args.sample_size)
    atomic_csv(args.output_dir / "selected_ligands.csv", selected)
    if args.analysis_only:
        cells = pd.read_csv(args.output_dir / "cell_terms.csv.gz")
    else:
        cells = score_panel(args, selected)
    if args.score_only:
        return
    summary = analyse(args, selected, cells)
    print(json.dumps({
        "status": summary["status"],
        "n_cells": summary["scope"]["n_cells"],
        "rescored_post_total": summary["spectral_metrics"]["rescored_post_total"],
        "zero_rot_pre_total": summary["spectral_metrics"]["zero_rot_pre_total"],
        "term_attribution": summary["term_attribution"],
    }, indent=2))


if __name__ == "__main__":
    main()
