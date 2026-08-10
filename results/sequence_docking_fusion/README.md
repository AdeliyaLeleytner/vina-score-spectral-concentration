# Sequence plus residual docking target geometry

This science-only artifact tests whether two-way-centered Vina target geometry
contains information about experimental kinase co-selectivity beyond receptor
sequence identity.

## Fixed predictor

The predictor is an outcome-blind 50:50 mean of pairwise percentile ranks from
receptor-domain sequence identity and centered Vina target geometry.  No weight
or threshold is fitted to an experimental panel.

The fusion's continuous target-geometry concordance exceeds both individual
components in all four examined panels.  Incremental QAP keeps sequence and the
experimental endpoint fixed and permutes only Vina target labels.  See
`omnibus_incremental_qap.csv` for the locked-panel mean contrast and
`panel_metrics.csv` for all point estimates.

For the continuous endpoint, fusion exceeds sequence after every one-target
deletion in each of DAVIS, PKIS2, and KiRHub.  PKIS1 is weaker and retains a
positive fusion-minus-sequence contrast after 13 of 20 deletions.

Continuous concordance is primary and upper-tail AUROC is secondary.  Average
precision is heterogeneous and is **not** a positive general claim.

## Boundary

This is post-hoc exploratory evidence on a fixed 20-kinase panel.  It concerns
target-pair geometry, not ligand-level target retrieval.  The fixed formula is
numerically outcome-blind, but the resources were inspected earlier in the
project and are not prospective confirmation.

## Reproduction

```bash
.venv/bin/python analysis/sequence_docking_fusion.py \
  --kirhub-workbook /tmp/kirhub_supp_tables.xlsx \
  --output results/sequence_docking_fusion \
  --permutations 50000 \
  --seed 20260814

.venv/bin/python -m pytest -q analysis/test_sequence_docking_fusion.py
```
