# Positive-score clipping sensitivity for ligand-wise ranking

This paired preprocessing sensitivity retains the original positive
DOCKSTRING scores while holding the ChEMBL cohort, target panel, external
training support, target transformations, experimental ties and equal-ligand
weighting fixed.

Only 12 evaluated ligand-by-target score cells were positive (0.005904%).

| Representation | Clipped accuracy | Retained-positive accuracy | Difference | Conservative cluster-bootstrap 95% interval |
|---|---:|---:|---:|---:|
| absolute_vina | 0.5371 | 0.5371 | +0.0000 | [+0.0000, +0.0000] |
| column_standardized | 0.5304 | 0.5305 | +0.0001 | [-0.0007, +0.0009] |
| two_way_residual | 0.5294 | 0.5290 | -0.0004 | [-0.0014, +0.0004] |
| target_centered_unscaled | 0.5331 | 0.5331 | +0.0000 | [+0.0000, +0.0000] |
| two_way_centered_unscaled | 0.5331 | 0.5331 | +0.0000 | [+0.0000, +0.0000] |
| target_centered_residual_scaled | 0.5256 | 0.5251 | -0.0005 | [-0.0018, +0.0008] |

The difference is retained-positive minus clipped. Intervals are the union
of the Murcko- and Butina-cluster bootstrap intervals and remain conditional
on the fixed 31-target panel.

This result does not imply that positive Vina energies are calibrated or
physically meaningful; it only tests whether clipping drives the reported
observed-pair ranking result.

## Reproduction

```bash
.venv/bin/python analysis/dockstring_positive_score_ranking_sensitivity.py \
  --bootstraps 2000 \
  --output-dir results/dockstring_positive_score_ranking_sensitivity
.venv/bin/python -m pytest -q \
  analysis/test_dockstring_positive_score_ranking_sensitivity.py
```
