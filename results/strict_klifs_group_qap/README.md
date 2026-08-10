# Strict within-KLIFS-group target-label QAP

This directory is a focused, exploratory conditional sensitivity for the fixed
20-kinase target panel. It asks whether the continuous association between the
previously released centered-Vina target geometry and experimental target
geometry is stronger than expected when complete predictor target labels may be
reassigned **only within the same KLIFS kinase group**.

This is not a dyad shuffle. Every null draw is a complete 20-target bijection.
The experimental endpoint, receptor-sequence identity, and KLIFS pocket-identity
matrices remain fixed.

## Exchangeability blocks and attainable inference

The fixed panel has these KLIFS groups:

| Group | Targets | Size | Label orders |
|---|---|---:|---:|
| TK | ABL1, CSF1R, EGFR, FGFR1, IGF1R, JAK2, KDR, KIT, LCK, MET, SRC | 11 | 39,916,800 |
| AGC | AKT1, AKT2, ROCK1 | 3 | 6 |
| CMGC | CDK2, MAPK1, MAPK14 | 3 | 6 |
| STE | MAP2K1 | 1 | 1 |
| CAMK | MAPKAPK2 | 1 | 1 |
| Other | PLK1 | 1 | 1 |

The full restricted space therefore contains `11! x 3! x 3! =
1,437,004,800` target orders. Seventeen targets can move and three are fixed.
Of 190 pair positions, 187 can receive a different predictor value; only the
three singleton-to-singleton positions are invariant. However, 154/190
(`81.1%`) pair positions involve TK, so the test is predominantly a TK-block
test.

AGC and CMGC each provide only six orders. Their joint exact diagnostic has 36
orders and little inferential resolution. A still stricter within-KLIFS-family
scheme would have only four exchangeable target pairs (Akt, MAPK, PDGFR, Src),
or `2^4 = 16` orders; its smallest attainable exact one-sided probability is
`1/16 = 0.0625`. It therefore cannot support a conventional 0.05 significance
claim on this panel and was not used as a confirmatory test.

## Results

The primary null used 100,000 independently sampled group-restricted target
orders (seed `20260805`). Probabilities are one-sided for a positive association
and use the plus-one Monte Carlo correction.

| Experimental geometry | Statistic | Spearman/partial Spearman | Restricted p | Null median | Null 95% interval |
|---|---|---:|---:|---:|---:|
| DAVIS/PKIS2/PKIS1 mean | unadjusted | 0.2676 | 0.1670 | 0.1346 | -0.0185 to 0.4342 |
| DAVIS/PKIS2/PKIS1 mean | sequence + KLIFS pocket adjusted | 0.2247 | 0.1972 | 0.1030 | -0.0631 to 0.4149 |
| KiRHub | unadjusted | 0.3216 | 0.1173 | 0.1867 | 0.0345 to 0.4221 |
| KiRHub | sequence + KLIFS pocket adjusted | 0.2799 | 0.1659 | 0.1614 | -0.0190 to 0.4189 |

The TK-only Monte Carlo diagnostic gives similar or weaker probabilities
(`0.1691` and `0.1504` unadjusted for the old-panel mean and KiRHub), confirming
that the large TK block dominates the available relabelling information. Exact
AGC+CMGC-only probabilities are `0.4722` and `0.2222`, respectively, and should
not be overinterpreted because only 36 orders exist.

## Interpretation and decision

**NO-GO for the strong claim that centered-Vina target geometry is associated
with experimental co-response beyond KLIFS group membership on this panel.**
The observed associations remain positive, but neither is resolved against the
strict group-preserving null. The positive unrestricted QAP and continuous
sequence/pocket-adjusted results should therefore be described as evidence of
alignment that is not exhausted by the *tested continuous covariates*, while
coarse kinase-group structure remains a viable explanation.

The restricted test itself is valid only under the assumption that targets are
exchangeable within their KLIFS group under the null. That assumption is not
guaranteed: groups contain distinct families, pockets, constructs, and assay
contexts. Rank-linear adjustment for receptor sequence and KLIFS pocket
identity is a useful sensitivity but cannot repair an exchangeability violation.
Accordingly, these probabilities are conditional diagnostics, not causal or
panel-general inference.

## Reproduction

Fetch the checksum-validated KiRHub workbook outside the repository and run:

```bash
.venv/bin/python analysis/fetch_kirhub_supplement.py /tmp/kirhub_strict_qap.xlsx
.venv/bin/python analysis/strict_klifs_group_qap.py \
  --kirhub-workbook /tmp/kirhub_strict_qap.xlsx \
  --permutations 100000 \
  --seed 20260805
.venv/bin/python -m pytest -q analysis/test_strict_klifs_group_qap.py
```

The KiRHub source activity rows and derived pair values are not redistributed.
Only aggregate statistics are written.

## Files

- `summary.json`: machine-readable estimates, null definition, power and
  exchangeability boundaries, provenance, and checksums.
- `exchangeability_blocks.csv`: exact KLIFS groups and target membership.
- `pair_strata.csv`: number and mobility of pair positions for every unordered
  KLIFS-group pair.
- `restricted_partial_qap.csv`: primary and diagnostic continuous QAP results.
