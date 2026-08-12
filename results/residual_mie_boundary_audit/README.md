# Residual/MIE boundary audit

This directory contains aggregate results from a frozen ChEMBL active-target
benchmark on Docking-44. The active set uses exact-relation Ki/Kd/IC50/EC50
records, median aggregation per compound-target cell, and pChEMBL >=
6.0. It contains
500 compound-target pairs, 190 compounds,
and 39 represented targets within the 44-target
ranking panel. No row-level compound, structure, target, or activity identifiers
are emitted here.

## Prespecified comparison

Lower mean rank percentile and higher top-k rates are favorable. With equal
weight per compound, absolute Vina gives mean rank percentile
0.4017 and top-5 rate
0.2361; column-z gives
0.4318 and
0.1704. The column-z minus absolute paired difference
is +0.0301 for rank percentile (Butina-cluster 95%
CI -0.0028 to
+0.0624) and
-0.0658 for top-5 rate (cluster 95% CI
-0.1243 to
-0.0103).

Against the fixed-margin target-frequency null, the one-sided compound-balanced
mean-rank probabilities are 0.0002 for
absolute Vina and 0.0002 for column-z.
The residual top-5 rate remains above its matched null
(0.1641 versus
0.1253; one-sided
p=0.0212). Thus target-specific
signal survives removal of the shared row axis, although target-wise
standardization lowers observed top-5 recovery relative to absolute scores.
These are weak aggregate retrieval results, not prospective target discovery.

## Exact boundary

Within each ligand, the two-way residual equals the column-centred score vector
minus one row-specific constant. Consequently its target order is exactly
identical to the column-centred score order. It need not equal the column-z order
(maximum rank difference =
33.0). The rise in
spectral effective dimension after row centering can reveal residual covariance,
but row centering itself cannot create a new within-ligand ranking signal.

## Files

- `observed_metrics.csv`: aggregate estimates and paired-unit bootstrap intervals.
- `paired_contrasts.csv`: representation differences using identical bootstrap draws.
- `null_summary.csv`: frequency-multiset and binary fixed-margin null summaries.
- `analysis_summary.json`: configuration, hashes, counts, assertions, and diagnostics.
- `input_manifest.csv`: frozen input paths, hashes, and row counts.

Run from the package root:

```bash
.venv/bin/python analysis/residual_mie_boundary_audit.py
.venv/bin/python -m unittest analysis/test_residual_mie_boundary_audit.py
```

Bootstrap intervals are descriptive for the frozen support. The cluster bootstrap
resamples the pre-existing frozen Butina labels; it does not re-fit the original
clustering. The fixed-margin null uses degree-preserving double-edge swaps and
therefore preserves every compound active-count, every target frequency, and
binary incidence.
