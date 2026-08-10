# Fixed-20 estimand decomposition and transformation-matched null

This directory contains the fixed-common-20 comparison of raw and locally
row-centred DOCKSTRING target-pair geometries against four experimental panels.
`summary.json` is the machine-readable result, `decomposition.csv` contains the
four observed 2 x 2 decompositions, and `transformation_matched_null.csv`
contains all Monte Carlo draws for the docking-side null.

## Transformation-matched null

One seeded sample of 15,000 ligands was drawn without replacement from the
259,579-ligand chemically de-leaked reference. Within every null repetition,
ligand identities were independently permuted inside each docking-target
column. This preserves every target marginal, variance, tie, and clipped zero,
while destroying ligand-wise target coordination. Raw and row-centred docking
geometries were derived from the same permuted matrix and compared with each
unchanged experimental geometry. The reported p-values therefore test whether
the observed docking-side increment on that same fixed support exceeds the
increment induced mechanically by row centring independent columns.

| Experimental panel | Fixed endpoint | Full-reference delta | Matched-support delta | Mean null delta | One-sided p |
| --- | --- | ---: | ---: | ---: | ---: |
| DAVIS | raw | 0.1684 | 0.1625 | 0.0226 | 0.0030 |
| PKIS2 | raw | 0.2486 | 0.2428 | -0.0667 | 0.0005 |
| PKIS1 | raw | 0.2180 | 0.2151 | -0.0200 | 0.0005 |
| KiRHub | raw | 0.2644 | 0.2610 | 0.0259 | 0.0005 |
| DAVIS | two-way-centred | 0.0164 | 0.0102 | 0.0261 | 0.6902 |
| PKIS2 | two-way-centred | 0.0437 | 0.0341 | 0.0050 | 0.1719 |
| PKIS1 | two-way-centred | 0.0009 | -0.0058 | -0.0047 | 0.5127 |
| KiRHub | two-way-centred | 0.0455 | 0.0411 | 0.0016 | 0.0730 |

The target-label paired QAP and this null answer different questions. The QAP
tests whether an observed increment is tied to target identities; the matched
null tests whether it exceeds the transformation-induced increment expected
after ligand-wise target coordination has been removed. Full-reference deltas
are descriptive and are not compared directly with the subsampled null.

## Reproduction

```bash
.venv/bin/python analysis/fixed20_estimand_decomposition.py \
  --pkis1-zip /tmp/pkis1_supplement.zip \
  --kirhub-workbook /tmp/kirhub_supp_tables.xlsx \
  --permutations 49999 \
  --matched-null-repeats 2000 \
  --matched-null-reference-size 15000
```

The frozen run used root seed `20260806`, support seed `50260806`, permutation
seed `51260806`, and support-row SHA256
`8f6eec2cd6f96b9e1fcfd6228ac64480aebb1106fa3d5f1f2c5bc65ff9de397d`.
