# Docking-44 missing-data residual sensitivity

This strict-public artifact is generated directly from the frozen row-level
Docking-44 table. It does not read an evidence ledger or import a local analysis
builder.

## Result

- Target-mean full-support residual PR: 9.302912
- Target-median full-support residual PR: 9.527863
- Complete-row residual PR: 13.970195
- Observed-cell additive-WLS residual PR with zero residual completion: 9.662954
- Equal-N random-support residual PR median and central 95% range:
  9.293952 [9.128066, 9.445166].
- Complete-case minus random-support mean: 4.675052

The complete-row and mean-imputed-restricted surfaces are exactly identical.
Accordingly, their difference from the primary surface is caused by ligand-support
selection, not by imputed values surviving on those rows. Chemical differences are
reported as raw shifts, standardized mean differences, robust median/IQR shifts,
and Cliff's delta; no hypothesis tests are used.

The observed-cell additive sensitivity fits row and target effects only to observed
cells and completes unobserved residuals with zero. It is not an estimate of the
missing raw docking scores.

## Reproduce

```bash
.venv/bin/python analysis/public_docking44_missing_data_sensitivity.py \
  --random-controls 1000 \
  --output-dir results/public_docking44_missing_data_sensitivity
.venv/bin/python -m pytest -q \
  analysis/test_public_docking44_missing_data_sensitivity.py
```
