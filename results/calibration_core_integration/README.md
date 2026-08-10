# Calibration-panel recovery of the leading residual target subspace

This science-only artifact asks whether a small, outcome-blind random ligand
panel can recover the leading residual target modes of the complete
DOCKSTRING 20-kinase surface.

## Main process result

With 500 calibration ligands, the full k=5 rule was recovered in 100.0% of empirical replicates and mean fixed-k subspace overlap was 0.877.

The full surface selected k=5 at a fixed
50%-trace rule.  On the previously frozen
experimental target-pair endpoint, full-support AUROC/AP were
0.710/
0.301 for the complete
residual correlation, 0.741/
0.316 for the raw leading
spectral kernel, and
0.747/
0.260 after
unit-diagonal renormalization.

The raw rank-k reconstruction is a **spectral similarity kernel**, not a
correlation matrix: its diagonal is the communality retained in the leading
modes.  Both it and its unit-diagonal version are reported because pair
retrieval can depend on that weighting.

## Interpretation boundary

This is an internal process/recovery experiment.  It shows whether a small
panel can recover a full-docking target subspace under the same source-library
distribution.  The experimental endpoint was already inspected elsewhere in
the project, so its reuse is post hoc and is not independent biological
confirmation.  The analysis says nothing about individual ligand profiles,
pose quality, or a chemically shifted deployment library.

## Reproduction

```bash
.venv/bin/python analysis/calibration_core_integration.py \
  --sizes 100,200,500 \
  --repetitions 200 \
  --output-dir results/calibration_core_integration

.venv/bin/python -m pytest -q \
  analysis/test_calibration_core_integration.py
```
