# Residual-mode sentinel panels for experimental assay compression

## Verdict

**Qualified GO as an exploratory positive result for calibrated-value
reconstruction; NO-GO as a target-ranking result.**  A deterministic ten-target
panel selected by DEIM sensor placement on leading two-way-centered Vina modes
reconstructs a 20-kinase experimental panel more accurately than target panels
selected from the raw Vina modes or at random.  It does not resolve a gain in
within-ligand rank or top-5 target recovery.

The original direct column-pivoted-QR idea is not used: after column
standardization every target has equal norm, making the first pivot arbitrary
and the selected set support-sensitive.  DEIM instead pivots the leading
eigenvector matrix, so target leverage in the retained residual subspace defines
the sensor placement.

## Design

- Fixed targets: 20 kinases shared by DOCKSTRING, PKIS2, PKIS1, and DAVIS.
- Target selection is outcome-blind.  The primary panel uses the residual target
  correlation from 259,579 complete DOCKSTRING ligands after excluding exact
  experimental-ligand connectivity blocks.
- DEIM selects targets spanning the leading `k` raw or residual eigenmodes.
  Receptor-sequence maximin and 2,000 uniformly random target panels are controls.
- Experimental reconstruction uses five-fold Bemis--Murcko-group-held-out
  validation.  Only training-fold target means/scales and regression parameters
  are used.  Predictors are seven ligand descriptors plus experimental values at
  the selected sentinel targets; ridge `alpha=1.0` is fixed.
- The same-cyclic-Murcko exclusion and every leave-one-scaffold-fold-out
  DOCKSTRING support select the same primary ten targets.  Across 25 independent
  15,000-ligand supports, median Jaccard agreement with the full-support panel is
  0.818.

The primary ten-target residual-DEIM panel is:

`MAPKAPK2, AKT2, EGFR, PLK1, IGF1R, KIT, MAPK14, LCK, MAP2K1, CDK2`.

## Main result

Full 20-target cross-validated R2 (the ten measured sentinels are inserted and
the other ten are predicted) is:

| Panel | Residual DEIM | Raw DEIM | Random-panel mean |
|---|---:|---:|---:|
| DAVIS | 0.510 | 0.374 | 0.400 |
| PKIS1 | 0.770 | 0.706 | 0.679 |
| PKIS2 | 0.649 | 0.607 | 0.618 |
| Three-panel mean | **0.643** | 0.562 | 0.566 |

The residual-DEIM minus random mean is +0.077; only 21 of 2,000 random
target-panel draws reach or exceed it (one-sided Monte Carlo `p=0.011`).  The
residual-minus-raw point contrast is positive in every experimental panel.
This probability is descriptive and unadjusted across the explored `k=3,5,10`
budgets; `k=10` is the operational half-panel setting, not a prospectively
registered optimum.

This is not merely credit for measuring half the targets.  On the ten genuinely
unmeasured targets, residual-DEIM sentinel measurements plus descriptors give R2
of 0.223 (DAVIS), 0.534 (PKIS1), and 0.289 (PKIS2).  Descriptor-only predictions
on the exact same target supports give -0.140, 0.002, and 0.029, respectively.

Rank objectives do not show the same result: the three-panel mean row-Spearman
is 0.699 versus random 0.679 (`p=0.250`), and top-5 overlap is 0.632 versus 0.631
(`p=0.539`).  The positive finding is therefore calibrated response-surface
reconstruction, not improved target ranking.

## Practical interpretation

The result provides one concrete use for the residual spectrum.  After a dense
calibration set has been measured, leading residual docking modes can nominate a
small set of experimental sensor targets.  New chemical scaffolds can then be
measured on that reduced panel and the remaining calibrated target responses can
be reconstructed.  In this fixed example the assay count is halved from 20 to
10.  A prospective study would still be required before claiming real assay-cost
savings or transfer to unmeasured kinases.

## Reproduce

```bash
.venv/bin/python analysis/experimental_sentinel_panel.py \
  --random-repeats 2000 \
  --support-repeats 25
```

Key outputs are `selected_targets.csv`, `method_summary.csv`,
`random_target_panel_comparisons.csv`, `paired_method_contrast_summary.csv`, and
`selection_support_stability.csv`.
