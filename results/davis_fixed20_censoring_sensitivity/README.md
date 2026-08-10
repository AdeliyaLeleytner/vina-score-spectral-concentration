# Fixed-20 DAVIS censoring sensitivity

This directory contains the measurement-limit sensitivity used in the manuscript. The
DAVIS-Complete 72 × 20 block has 965/1,440 values at the released 10,000-nM Kd upper
bound, which becomes a pKd floor of 5.

`summary.json` reports:

- agreement between continuous-pKd and binary above-floor-status target geometry;
- centred-DOCKSTRING concordance with both DAVIS representations;
- target filters requiring at least 5, 10 or 15 values above the floor;
- a three-panel endpoint rebuilt after replacing continuous DAVIS correlation by binary
  above-floor-status correlation, evaluated against centred KiRHub geometry.

`target_quality_sensitivity.csv` gives the target-filter results. The KiRHub source workbook
is not redistributed; only aggregate results are released.

Reproduce from the checksum-verified source workbook with:

```bash
python analysis/davis_fixed20_censoring_sensitivity.py \
  --kirhub-workbook /tmp/kirhub_supp_tables.xlsx
```

These results support a coarse co-response or measurement-limit-status interpretation of
DAVIS, not a finely resolved quantitative-affinity surface.
