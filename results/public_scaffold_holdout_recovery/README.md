# Strict-public chemical-group-held-out pilot recovery

Calibration maps are estimated on four chemically grouped folds and
compared with the residual Vina map on the fifth. Cyclic molecules use
non-isomeric Bemis--Murcko groups; acyclic molecules use canonical
non-isomeric full-connectivity identity, so duplicate acyclic structures
cannot cross folds.
Matched random-row holdouts use the same calibration and evaluation
sizes. Docking-44 imputation is fold-local. All intervals are repeated-
sample sensitivity ranges conditional on the fixed panel and source.
For every held-out ligand, the exact maximum radius-2, 2048-bit Morgan
Tanimoto similarity to the full calibration pool is reported. This
diagnoses remaining analogue proximity but is not a fingerprint-cluster-
disjoint split.
A Tanimoto value of one means Morgan-bit-vector identity only and is
not labelled as exact molecular identity.

```bash
.venv/bin/python analysis/public_scaffold_holdout_recovery.py
.venv/bin/python -m pytest -q analysis/test_public_scaffold_holdout_recovery.py
```
