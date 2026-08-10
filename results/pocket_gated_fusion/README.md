# Pocket-gated sequence plus complementary Vina geometries

This science-only analysis tests whether docking-derived target geometry is
most useful for kinase pairs with relatively dissimilar KLIFS ATP pockets.
The threshold is selected on PKIS1 only and then frozen.

The selected pocket-identity threshold is **0.55**. On DAVIS,
PKIS2, and KiRHub, the dual-surface gated predictor improves continuous geometry
agreement, AUROC, and average precision relative to sequence alone. The
locked-panel mean deltas and target-label QAP probabilities are:

- continuous_spearman: delta +0.1102, QAP p=0.00848, Holm p=0.01300.
- roc_auc: delta +0.1287, QAP p=0.00650, Holm p=0.01300.
- average_precision: delta +0.0818, QAP p=0.00326, Holm p=0.00978.

The interpretation is regime-specific: raw and residual Vina geometries add
complementary cross-reactivity information where pocket sequence is less
informative. This is not a claim that Vina generally outperforms sequence.

## Panel metrics

See `panel_metrics.csv` for all values and `threshold_sensitivity.csv`
for the complete fixed grid.

## Reproduction

```bash
.venv/bin/python analysis/pocket_gated_fusion.py \
  --kirhub-workbook /private/tmp/kirhub_supp_tables.xlsx \
  --output results/pocket_gated_fusion \
  --permutations 50000 --seed 20260818
```
