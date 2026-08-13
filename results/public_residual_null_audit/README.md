# Public residual-null audit

This standalone artifact is produced directly from the two frozen row-level Vina
score tables. It does not consume any historical evidence ledger. All nulls
for a dataset share one fingerprinted deterministic ligand support.

Reproduce with:

```bash
.venv/bin/python analysis/public_residual_null_audit.py \
  --sample-size 12000 --repeats 500 \
  --output-dir results/public_residual_null_audit
.venv/bin/python -m pytest -q analysis/test_public_residual_null_audit.py
```
