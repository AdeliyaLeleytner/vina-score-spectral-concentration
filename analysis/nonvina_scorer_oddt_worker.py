#!/usr/bin/env python3
"""ODDT rescoring worker for ``nonvina_scorer_transport``.

This module is deliberately *not* part of the ``.venv`` release interpreter.
ODDT 0.7 predates Python 3.12 and is only installed in the project's
``toxic_agent_conda_env`` (CPython 3.9).  ``analysis/nonvina_scorer_transport.py``
runs this file as a subprocess under that interpreter, and consumes the CSV it
writes.  Everything here is therefore written to Python 3.9 syntax.

The worker rescores fixed DOCKSTRING poses -- the released top Vina pose for a
ligand--target cell -- with machine-learned scoring functions that share no
functional form with AutoDock Vina:

* RF-Score v1/v2/v3 (random forest over close-contact counts; v3 adds Vina terms)
* NNScore 2.0 (neural-network ensemble over BINANA descriptors)
* PLECscore linear (protein-ligand extended connectivity fingerprint, pretrained)

RF-Score and NNScore are trained here, offline, from the PDBbind descriptor
tables bundled inside the ODDT wheel; PLECscore's pretrained linear coefficients
ship as JSON inside the same wheel.  No network access is required or attempted.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import multiprocessing
import os
import sys
import time
from os.path import dirname, isfile, join

import numpy as np


ODDT_SCORERS = ("rfscore_v1", "rfscore_v2", "rfscore_v3", "nnscore", "plecscore_linear")
PDBBIND_VERSION = 2016
# ODDT's own trainers call numpy.random.seed(1) before fitting; recorded here so
# the seed appears explicitly in the emitted provenance.
ODDT_INTERNAL_TRAINING_SEED = 1

_STATE = {}


def patch_sparse_vstack():
    """Make ODDT's sparse descriptor path work with modern SciPy.

    ODDT 0.7 calls ``scipy.sparse.vstack(map(...))``.  SciPy >= 1.11 coerces its
    argument with ``np.asarray(..., dtype=object)``, which turns a ``map`` object
    into a 0-d array and raises ``TypeError: iteration over a 0-d array``.  Only
    the sparse-fingerprint scorer (PLECscore) reaches this path.  Materialising
    the iterator restores the original behaviour and changes no numerics.
    """
    from scipy.sparse import vstack as scipy_vstack

    def vstack_materialised(blocks, format=None, dtype=None):
        return scipy_vstack(list(blocks), format=format, dtype=dtype)

    import oddt.scoring
    import oddt.scoring.descriptors

    oddt.scoring.descriptors.sparse_vstack = vstack_materialised
    oddt.scoring.sparse_vstack = vstack_materialised
    import scipy

    return {
        "reason": "scipy.sparse.vstack no longer accepts a map object",
        "scipy_version": scipy.__version__,
        "patched_symbols": ["oddt.scoring.sparse_vstack", "oddt.scoring.descriptors.sparse_vstack"],
        "numeric_effect": "none; the iterator is materialised into a list before stacking",
    }


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _functions_dir():
    """Directory holding ODDT's bundled PDBbind descriptor tables.

    ``oddt.scoring.functions`` re-exports the scorer *classes* under the same
    names as their modules, so ``oddt.scoring.functions.PLECscore.__file__``
    resolves to a class, not a module; go through the package instead.
    """
    import oddt.scoring.functions as package

    return dirname(package.__file__)


def _rfscore_desc_path(version):
    return join(_functions_dir(), "RFScore", "rfscore_descs_v%i.csv" % version)


def _nnscore_desc_path():
    return join(_functions_dir(), "NNScore", "nnscore_descs.csv")


def _plecscore_json_path(depth_protein=5, depth_ligand=1, size=65536):
    return join(
        _functions_dir(),
        "PLECscore",
        "plecscore_linear_p%i_l%i_s%i_pdbbind%i.json"
        % (depth_protein, depth_ligand, size, PDBBIND_VERSION),
    )


def _holdout_pearson(scorer_object, desc_path):
    """Recompute the PDBbind core-set Pearson r of a trained ODDT model.

    This is a provenance check: it must reproduce the published RF-Score /
    NNScore core-set numbers, otherwise the offline training path is broken.
    """
    from scipy.stats import pearsonr

    scorer_object._load_pdbbind_desc(desc_path, pdbbind_version=PDBBIND_VERSION)
    predicted = scorer_object.model.predict(scorer_object.test_descs)
    return {
        "pdbbind_version": PDBBIND_VERSION,
        "train_complexes": int(len(scorer_object.train_target)),
        "core_set_complexes": int(len(scorer_object.test_target)),
        "core_set_pearson_r": float(pearsonr(predicted, scorer_object.test_target)[0]),
    }


def prepare_models(model_dir, scorers):
    """Train (or reuse) every requested ODDT model and record its provenance."""
    from oddt.scoring.functions.NNScore import nnscore
    from oddt.scoring.functions.PLECscore import PLECscore
    from oddt.scoring.functions.RFScore import rfscore

    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)
    provenance = {}
    for name in scorers:
        started = time.time()
        pickle_path = join(model_dir, "%s.pickle" % name)
        trained_now = not isfile(pickle_path)
        if name.startswith("rfscore_v"):
            version = int(name.rsplit("_v", 1)[1])
            desc_path = _rfscore_desc_path(version)
            if trained_now:
                model = rfscore(version=version, n_jobs=1)
                model.train(sf_pickle=pickle_path, pdbbind_version=PDBBIND_VERSION)
            model = rfscore.load(pickle_path)
            entry = _holdout_pearson(model, desc_path)
            entry["training_table"] = os.path.basename(desc_path)
            entry["training_table_sha256"] = sha256_file(desc_path)
        elif name == "nnscore":
            desc_path = _nnscore_desc_path()
            if trained_now:
                model = nnscore(n_jobs=4)
                model.train(sf_pickle=pickle_path, pdbbind_version=PDBBIND_VERSION)
            model = nnscore.load(pickle_path)
            entry = _holdout_pearson(model, desc_path)
            entry["training_table"] = os.path.basename(desc_path)
            entry["training_table_sha256"] = sha256_file(desc_path)
        elif name == "plecscore_linear":
            json_path = _plecscore_json_path()
            if not isfile(json_path):
                raise RuntimeError("pretrained PLECscore JSON absent: %s" % json_path)
            if trained_now:
                model = PLECscore(version="linear")
                # With the shipped JSON present, train() loads the pretrained
                # coefficients instead of refitting; it never touches the network.
                model.train(sf_pickle=pickle_path, pdbbind_version=PDBBIND_VERSION)
            model = PLECscore.load(pickle_path)
            entry = {
                "pdbbind_version": PDBBIND_VERSION,
                "pretrained_coefficients": os.path.basename(json_path),
                "pretrained_coefficients_sha256": sha256_file(json_path),
                "core_set_pearson_r": None,
                "core_set_pearson_note": (
                    "PLECscore ships pretrained linear coefficients but not its "
                    "PDBbind descriptor table, so no offline hold-out check is possible."
                ),
            }
        else:
            raise ValueError("unknown ODDT scorer: %s" % name)
        entry["score_title"] = model.score_title
        entry["model_pickle_sha256"] = sha256_file(pickle_path)
        entry["trained_in_this_run"] = bool(trained_now)
        entry["prepare_seconds"] = float(time.time() - started)
        entry["oddt_internal_training_seed"] = ODDT_INTERNAL_TRAINING_SEED
        provenance[name] = entry
    return provenance


def _init_worker(model_dir, scorers, target_dir, poses_dir):
    patch_sparse_vstack()
    from oddt.scoring.functions.NNScore import nnscore
    from oddt.scoring.functions.PLECscore import PLECscore
    from oddt.scoring.functions.RFScore import rfscore

    loaders = {"nnscore": nnscore, "plecscore_linear": PLECscore}
    models = {}
    for name in scorers:
        loader = loaders.get(name, rfscore)
        model = loader.load(join(model_dir, "%s.pickle" % name))
        # Keep every pool worker single-threaded: nested joblib parallelism inside
        # a process pool oversubscribes the machine and slows scoring down.
        for attribute in ("n_jobs",):
            if hasattr(model, attribute):
                setattr(model, attribute, 1)
            if hasattr(model.model, attribute):
                try:
                    setattr(model.model, attribute, 1)
                except Exception:  # pragma: no cover - defensive only
                    pass
        models[name] = model
    _STATE["models"] = models
    _STATE["scorers"] = tuple(scorers)
    _STATE["target_dir"] = target_dir
    _STATE["poses_dir"] = poses_dir


def read_receptor(path):
    """Load a prepared DOCKSTRING receptor PDBQT with tolerant sanitisation.

    ODDT's ``readfile('pdbqt', ...)`` runs a full RDKit sanitisation and drops
    any keyword arguments, so it raises ``AtomValenceException`` on the DRD2
    receptor (proximity bonding gives atom 503 five bonds).  Reading with
    ``sanitize=False`` and then sanitising with the valence-property check
    disabled recovers ring perception, aromaticity and hybridisation for every
    receptor.  Verified to give bit-identical RF-Score v3, NNScore and PLECscore
    values to the default path on receptors where the default path works, and
    applied uniformly to all targets so no target is treated differently.
    """
    import oddt
    from rdkit import Chem

    with open(path) as handle:
        block = handle.read()
    molecule = oddt.toolkit.readstring("pdbqt", block, sanitize=False)
    Chem.SanitizeMol(molecule.Mol, sanitizeOps=Chem.SANITIZE_ALL ^ Chem.SANITIZE_PROPERTIES)
    molecule.protein = True
    return molecule


def _score_target(target):
    import oddt

    started = time.time()
    receptor = join(_STATE["target_dir"], "%s_target.pdbqt" % target)
    poses = join(_STATE["poses_dir"], "%s.sdf" % target)
    protein = read_receptor(receptor)
    ligands = list(oddt.toolkit.readfile("sdf", poses))
    keys = [molecule.data["key"] for molecule in ligands]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate pose keys in %s" % poses)
    per_scorer = {}
    timings = {}
    for name in _STATE["scorers"]:
        model = _STATE["models"][name]
        model.set_protein(protein)
        scorer_started = time.time()
        values = []
        for molecule in model.predict_ligands(ligands):
            values.append(
                (molecule.data["key"], float(molecule.data[model.score_title]))
            )
        if len(values) != len(keys):
            raise ValueError(
                "%s returned %i scores for %i poses on %s"
                % (name, len(values), len(keys), target)
            )
        per_scorer[name] = dict(values)
        timings[name] = float(time.time() - scorer_started)
    rows = []
    for key in keys:
        row = {"target": target, "key": key}
        for name in _STATE["scorers"]:
            row[name] = per_scorer[name][key]
        rows.append(row)
    return {
        "target": target,
        "rows": rows,
        "n_poses": len(keys),
        "protein_atoms": int(len(protein.atoms)),
        "receptor_sha256": sha256_file(receptor),
        "scorer_seconds": timings,
        "target_seconds": float(time.time() - started),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--poses-dir", required=True)
    parser.add_argument("--target-dir", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--targets", required=True, help="comma separated")
    parser.add_argument("--scorers", default=",".join(ODDT_SCORERS))
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()

    targets = [value for value in args.targets.split(",") if value]
    scorers = [value for value in args.scorers.split(",") if value]
    unknown = set(scorers) - set(ODDT_SCORERS)
    if unknown:
        raise SystemExit("unknown scorers: %s" % sorted(unknown))

    import oddt

    scipy_patch = patch_sparse_vstack()
    started = time.time()
    provenance = prepare_models(args.model_dir, scorers)
    prepare_seconds = time.time() - started

    started = time.time()
    context = multiprocessing.get_context("spawn")
    if args.workers > 1:
        pool = context.Pool(
            processes=min(args.workers, len(targets)),
            initializer=_init_worker,
            initargs=(args.model_dir, scorers, args.target_dir, args.poses_dir),
        )
        try:
            results = pool.map(_score_target, targets)
        finally:
            pool.close()
            pool.join()
    else:
        _init_worker(args.model_dir, scorers, args.target_dir, args.poses_dir)
        results = [_score_target(target) for target in targets]
    score_seconds = time.time() - started

    results = sorted(results, key=lambda item: item["target"])
    fieldnames = ["target", "key"] + scorers
    tmp_csv = args.out_csv + ".tmp"
    with open(tmp_csv, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            for row in result["rows"]:
                writer.writerow(row)
    os.replace(tmp_csv, args.out_csv)

    payload = {
        "interpreter": sys.version,
        "interpreter_executable": sys.executable,
        "oddt_version": oddt.__version__,
        "oddt_toolkit": oddt.toolkit.backend,
        "numpy_version": np.__version__,
        "scorers": scorers,
        "targets": targets,
        "workers": int(args.workers),
        "scipy_compatibility_patch": scipy_patch,
        "receptor_reader": (
            "oddt.toolkit.readstring('pdbqt', ..., sanitize=False) followed by "
            "Chem.SanitizeMol(sanitizeOps=SANITIZE_ALL ^ SANITIZE_PROPERTIES); applied "
            "uniformly to all targets because the DRD2 receptor fails the RDKit valence "
            "check, and verified bit-identical to the default reader where the default "
            "reader succeeds"
        ),
        "model_provenance": provenance,
        "prepare_seconds": float(prepare_seconds),
        "score_seconds": float(score_seconds),
        "per_target": {
            result["target"]: {
                "n_poses": result["n_poses"],
                "protein_atoms": result["protein_atoms"],
                "receptor_sha256": result["receptor_sha256"],
                "scorer_seconds": result["scorer_seconds"],
                "target_seconds": result["target_seconds"],
            }
            for result in results
        },
    }
    tmp_json = args.out_json + ".tmp"
    with open(tmp_json, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(tmp_json, args.out_json)


if __name__ == "__main__":
    main()
