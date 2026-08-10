# Residual target-geometry validation

This directory contains an exploratory, manuscript-independent validation of the
target-correlation geometry exposed by two-way centering of the DOCKSTRING Vina
surface.  The primary experimental supports are the complete 72 x 21 DAVIS pKd,
645 x 21 PKIS2 percentage-inhibition, and 360 x 20 PKIS1 Nanosyn 1-uM panels.

The docking geometry is estimated from 259,579 complete DOCKSTRING ligands after
excluding every Standard-InChI connectivity block observed in any of the three
experimental panels.  Experimental ligands are not required to match DOCKSTRING.
The concordance statistic is Spearman correlation between the strict-upper-triangle
entries of docking and experimental target-correlation matrices.  Target-label QAP,
projection-matched independent-column nulls, Bayesian ligand/chemical-cluster
bootstraps, target jackknife, independent reference supports, censoring checks, and
same-Murcko-scaffold exclusion are reported in `summary.json` and the CSV files.
As a secondary operational endpoint, the analysis asks whether docking correlations
retrieve the top 10% of pairs in the centered experimental co-selectivity geometry;
raw and centered docking predict the same fixed experimental labels and are compared
with paired target-label QAP.

## Reproduction

Obtain and checksum-validate the original PKIS1 supplement (Elkins et al., DOI
10.1038/nbt.3374):

```bash
python analysis/fetch_pkis1_supplement.py /tmp/pkis1_supplement.zip
```

Run the release-scale analysis:

```bash
python analysis/residual_target_geometry_validation.py \
  --output-dir results/residual_target_geometry \
  --qap-permutations 20000 \
  --bootstrap-repeats 5000 \
  --projection-null-repeats 500 \
  --projection-reference-size 15000 \
  --reference-support-size 15000 \
  --reference-support-seeds 11,29,47,71,97 \
  --pkis1-zip /tmp/pkis1_supplement.zip
```

The source ZIP is not redistributed.  Its required SHA-256 is recorded by both the
fetcher and `summary.json`.
