# Ordinal within-ligand docking geometry

Status: **science-only exploratory test; manuscript unchanged**.

## Question

Does a docking fingerprint that retains only each ligand's ordering of 20 kinase
targets transfer more robustly than the standard column-z-scored, row-centered
Vina surface?

All four predictors use the same 259,579-ligand DOCKSTRING support after excluding
all DAVIS, PKIS2, and PKIS1 Standard-InChI connectivity blocks:

1. raw column-z scores;
2. column-z scores followed by row centering;
3. within-ligand target ranks followed by column z-scoring and row centering;
4. Vina score per heavy atom followed by column z-scoring and row centering.

Experimental endpoints were fixed before predictor comparison: continuous
two-way-centered target geometries from DAVIS, PKIS2, PKIS1, and KiRHub, plus the
previously defined replicated upper-tail target-pair endpoint. Target-label QAP
used the same permutation for the ordinal and standard-centered predictors.

## Result: no-go

Ordinal geometry was lower than standard centered geometry in every continuous
panel:

| Panel | Standard centered | Ordinal | Ordinal minus standard |
|---|---:|---:|---:|
| DAVIS | 0.2925 | 0.2526 | -0.0399 |
| PKIS2 | 0.3003 | 0.2462 | -0.0541 |
| PKIS1 | 0.1854 | 0.1419 | -0.0435 |
| KiRHub | 0.3361 | 0.3040 | -0.0321 |
| Equal-weight mean | 0.2786 | 0.2362 | -0.0424 |

The mean loss was present in all 20 target-delete-one analyses (range
-0.0640 to -0.0167). Its paired QAP lower-tail probability was 0.0280
(two-sided p=0.0560). PKIS2 alone showed a two-sided loss at p=0.0302.

Ordinal geometry had a small advantage only on the thresholded replicated
endpoint (AUROC 0.7303 versus 0.7181; difference +0.0122), but the paired QAP did
not support it (one-sided p=0.3207). The gain remained positive in 18/20 target
deletions for AUROC but only 11/20 for average precision. Heavy-atom efficiency
was also not stronger: its four-panel mean Spearman was 0.2031 and its replicated
AUROC was 0.7090.

**Conclusion:** the ordinal hypothesis is rejected as a stronger cross-panel
representation. The consistent continuous-panel loss instead indicates that
experimental target co-response uses information in relative score magnitudes
beyond within-ligand target ordering. The small binary-endpoint gain is
endpoint-specific and should not be promoted as a general result.

This result concerns target-pair geometry on a fixed 20-kinase panel. It is not a
test of ligand-level target ranking, unseen-target retrieval, affinity calibration,
or docking pose quality. KiRHub supplies names but not structures, so structural
chemical-overlap exclusion is exact for DAVIS/PKIS1/PKIS2 but unavailable for
KiRHub. The SPD panel overlaps this kinase panel only at EGFR and therefore cannot
test the same target-pair geometry.

## Reproduction

```bash
.venv/bin/python analysis/ordinal_docking_geometry.py \
  --pkis1-zip /path/to/pkis1_supplement.zip \
  --kirhub-workbook /path/to/kirhub_supp_tables.xlsx \
  --output-dir results/ordinal_docking_geometry \
  --qap-permutations 50000 \
  --seed 20260805

.venv/bin/python -m pytest -q \
  analysis/test_ordinal_docking_geometry.py \
  analysis/test_replicated_pair_retrieval.py \
  analysis/test_kirhub_external_validation.py
```

Aggregate outputs contain no compound identifiers. `output_checksums.json`
records the deterministic result-file checksums.
