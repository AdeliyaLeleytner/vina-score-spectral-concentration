# Strict-public chemical-domain controls

This artifact is recomputed directly from the frozen row-level Docking-44
and DOCKSTRING score tables. It imports no local analysis builder and reads
no evidence ledger.

## Key comparison

- Docking-44, raw: observed low/high rho=0.227; global MW-matched group-disjoint median=0.996; within-low/high group-disjoint medians=0.989/0.989.
- Docking-44, row_centered_residual: observed low/high rho=0.205; global MW-matched group-disjoint median=0.992; within-low/high group-disjoint medians=0.980/0.979.
- DOCKSTRING-58, raw: observed low/high rho=0.409; global MW-matched group-disjoint median=0.998; within-low/high group-disjoint medians=0.991/0.980.
- DOCKSTRING-58, row_centered_residual: observed low/high rho=0.351; global MW-matched group-disjoint median=0.994; within-low/high group-disjoint medians=0.979/0.966.

Repeated-split ranges are composition-sensitivity ranges, not population
confidence intervals. MW was selected post hoc; the result diagnoses
chemical-domain dependence and does not identify a causal descriptor.

## Reproduce

```bash
.venv/bin/python analysis/public_chemical_domain_controls.py \
  --repetitions 200 \
  --output-dir results/public_chemical_domain_controls
.venv/bin/python -m pytest -q analysis/test_public_chemical_domain_controls.py
```
