# Small calibration panels recover residual target geometry

This science-only exploratory artifact asks how many ligands must be docked to
estimate the two-way-centered target-correlation geometry of a much larger Vina
surface. It uses 100 simple-random panels at each size and writes aggregate
metrics only.

## Main result

At 200 calibration ligands, mean geometry concordance with the complete matrix
was 0.918 for DOCKSTRING-58 and 0.942 for Docking-44. At 500 ligands it was
0.965 and 0.976, respectively. The previously frozen 20-kinase experimental
co-selective-pair endpoint was also nearly preserved: mean AUROC was 0.701 at
200 ligands and 0.706 at 500 ligands, versus 0.710 from all 260,060 complete
DOCKSTRING ligands.

The result is not driven by reusing the same scaffolds on both sides. In a
five-fold scaffold GroupKFold sensitivity, 200 calibration ligands recovered
the geometry of held-out scaffolds with mean concordance 0.916 for DOCKSTRING
and 0.927 for Docking-44; at 500 ligands the corresponding values were 0.958
and 0.959.

Finite-sample sampling correlations bias ordinary PR downward. OAS shrinkage
removed most of this bias. At 200 ligands, mean OAS PR was 18.154 versus the
DOCKSTRING reference 18.206, and 9.670 versus the Docking-44 reference 9.303.
At 500 ligands the corresponding estimates were 18.114 and 9.468.

For DOCKSTRING, a 200-ligand pilot requires 11,600 score cells rather than
15,083,480, a 1,300-fold reduction. For Docking-44, it requires 8,800 rather
than 556,644 cells, a 63-fold reduction.

## Interpretation boundary

The result supports a low-cost pilot for estimating target-network geometry and
its leading modes on a representative source library. It does not reconstruct
individual ligand profiles, validate docking poses, or establish transfer to a
chemically shifted deployment library. Random sampling targets the source
library's chemical distribution; forcing one compound per scaffold did not
improve geometry recovery in the exploratory sensitivity and is not the primary
design.

## Reproduction

```bash
.venv/bin/python analysis/calibration_panel_recovery.py \
  --repetitions 100 \
  --sizes 50,100,200,500,1000,2000,5000 \
  --output-dir results/calibration_panel_recovery

.venv/bin/python -m pytest -q \
  analysis/test_calibration_panel_recovery.py
```

The machine-readable outputs are `replicate_metrics.csv`, `summary.csv`,
`scaffold_holdout_metrics.csv`, `scaffold_holdout_summary.csv`, and
`metadata.json`.
