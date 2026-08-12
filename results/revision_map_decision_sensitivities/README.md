# Threshold-robust support shift and decision-level map recovery

This revision-only artifact is exploratory and outcome-blind. It does not
modify the manuscript or establish biological validity of a Vina map.

## Molecular-weight support continuum

- DOCKSTRING-58, raw: low/high-tail map agreement ranged from 0.134 to 0.598 across 7 thresholds (0.10--0.40 per tail); equal-N random-disjoint means ranged 0.991--0.998, and every observed value fell below its matched control 2.5th percentile: True.
- DOCKSTRING-58, row_centered_residual: low/high-tail map agreement ranged from 0.174 to 0.528 across 7 thresholds (0.10--0.40 per tail); equal-N random-disjoint means ranged 0.976--0.994, and every observed value fell below its matched control 2.5th percentile: True.
- Docking-44, raw: low/high-tail map agreement ranged from 0.175 to 0.351 across 7 thresholds (0.10--0.40 per tail); equal-N random-disjoint means ranged 0.988--0.997, and every observed value fell below its matched control 2.5th percentile: True.
- Docking-44, row_centered_residual: low/high-tail map agreement ranged from 0.201 to 0.259 across 7 thresholds (0.10--0.40 per tail); equal-N random-disjoint means ranged 0.978--0.995, and every observed value fell below its matched control 2.5th percentile: True.

Across ten non-overlapping MW bins:

- Docking-44, raw: Spearman(MW separation, map dissimilarity) = 0.814.
- Docking-44, row_centered_residual: Spearman(MW separation, map dissimilarity) = 0.945.
- DOCKSTRING-58, raw: Spearman(MW separation, map dissimilarity) = 0.925.
- DOCKSTRING-58, row_centered_residual: Spearman(MW separation, map dissimilarity) = 0.988.

These are descriptive pairwise trends. The 45 bin pairs reuse ten maps, so
they are not treated as independent inferential observations.

## Recovery of edges and decisions

Each size uses 100 simple-random representative panels. Selected
numbers below are replicate means; CSV files retain central 95% replicate
ranges.

- DOCKSTRING-58, n=200, row-disjoint complement: global rho 0.920; all-edge sign agreement 0.877; top-positive-10 precision 0.639; top-negative-10 precision 0.677.
- DOCKSTRING-58, n=500, row-disjoint complement: global rho 0.965; all-edge sign agreement 0.920; top-positive-10 precision 0.779; top-negative-10 precision 0.776.
- Docking-44, n=200, row-disjoint complement: global rho 0.940; all-edge sign agreement 0.896; top-positive-10 precision 0.550; top-negative-10 precision 0.480.
- Docking-44, n=500, row-disjoint complement: global rho 0.973; all-edge sign agreement 0.931; top-positive-10 precision 0.618; top-negative-10 precision 0.606.

For the fixed k=8 decision example:

- DOCKSTRING-58, n=200, row-disjoint complement: cluster ARI 0.595; overlap with the fixed full-source panel 3.65/8; sample-minus-fixed-full-source-panel coverage difference, with both evaluated on the complement, 0.0113; fraction of random panels with no worse mean coverage 0.001.
- DOCKSTRING-58, n=500, row-disjoint complement: cluster ARI 0.658; overlap with the fixed full-source panel 5.01/8; sample-minus-fixed-full-source-panel coverage difference, with both evaluated on the complement, 0.0056; fraction of random panels with no worse mean coverage 0.000.
- Docking-44, n=200, row-disjoint complement: cluster ARI 0.694; overlap with the fixed full-source panel 3.41/8; sample-minus-fixed-full-source-panel coverage difference, with both evaluated on the complement, 0.0153; fraction of random panels with no worse mean coverage 0.013.
- Docking-44, n=500, row-disjoint complement: cluster ARI 0.784; overlap with the fixed full-source panel 4.63/8; sample-minus-fixed-full-source-panel coverage difference, with both evaluated on the complement, 0.0083; fraction of random panels with no worse mean coverage 0.001.

The medoid exercise is a decision-level demonstration for compressing the
tested Vina target map. It does not show that the selected targets maximize
biological coverage or experimental selectivity. On the row-disjoint
complement, the comparator is the fixed full-source panel, not a separately
optimized complement panel; the displayed difference is therefore not
held-out optimality regret.

## Reproduction

```bash
.venv/bin/python analysis/revision_map_decision_sensitivities.py \
  --repetitions 100 --sizes 200,500 \
  --mw-null-repetitions 100 \
  --cluster-k 6,8,10 \
  --output-dir results/revision_map_decision_sensitivities
.venv/bin/python -m pytest -q analysis/test_revision_map_decision_sensitivities.py
```
