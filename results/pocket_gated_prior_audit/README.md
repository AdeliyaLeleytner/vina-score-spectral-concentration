# Target-degree-prior audit of pocket-gated Vina fusion

**Strict decision: NO-GO as an independently resolved docking increment.**

This analysis tests the strongest simple rival explanation for the
pocket-gated fusion result: target pairs may look co-selective because
their targets merely have similar marginal hit rates. PKIS1 alone defines
the target-degree prior, fits all coefficients, and selects the gate; DAVIS,
PKIS2, and KiRHub remain locked/no-retuning evaluations.

The selected gate remains **0.55**. Relative to a
baseline already containing the PKIS1 target-degree prior, receptor-domain
sequence identity, and KLIFS pocket identity, the exact dual-surface gated
docking addition gives:

- continuous_spearman: mean delta +0.0213; minimum panel delta +0.0142; selection-aware QAP p=0.06362; Holm p=0.19086.
- top_10_roc_auc: mean delta +0.0254; minimum panel delta +0.0206; selection-aware QAP p=0.07320; Holm p=0.19086.
- top_10_average_precision: mean delta +0.0282; minimum panel delta +0.0149; selection-aware QAP p=0.07922; Holm p=0.19086.

All point estimates are positive in every locked panel and in all 60 panel-by-target deletions, but the full-procedure QAP does not resolve the increment after multiplicity correction. The directional effect is therefore suggestive, not a new headline result.

The alternative explanation is substantial: the two-feature PKIS1 degree
prior alone reaches external Spearman correlations from 0.323 to 0.426; adding sequence and pocket
identity raises them to 0.454--0.505.
This target-degree increment passes its target-label QAP and is the strong positive result of the audit.
See `degree_prior_increment_qap.csv` for the complete null comparison.
The degree-prior increment remains positive in all 60 panel-by-target
deletions for every primary metric. Alternate cutoffs and threshold-free
marginal summaries are reported in
`target_prior_definition_sensitivity.csv`; these are explicitly post hoc.

In the fixed-gate component decomposition, adding a separately weighted
residual-Vina gate after the raw-Vina gate changes the locked mean by
+0.0012 Spearman, -0.0015 AUROC, and +0.0047 AP. At least one locked panel is negative for each of these nested contrasts,
so the present audit does not isolate an incremental residual-surface gain
beyond raw Vina once the strong baseline is included.

The QAP is selection-aware: it jointly permutes raw and residual Vina
target labels and repeats PKIS1 gate selection and model fitting on every
permutation. See `selection_aware_qap.csv` for top-5%, top-10%, and top-15%
endpoints, `target_jackknife.csv` for delete-one-target results, and
`degree_threshold_sensitivity.csv` for alternate PKIS1 activity cutoffs.

## No-hard-gate test

Without imposing a hard gate, pairwise error improvement has Spearman rho -0.034 with KLIFS pocket identity (one-sided target-label QAP p=0.3285).
This is a separate falsification test of a monotonic low-identity trend; a
non-significant result must not be used to justify the hard threshold.

## Boundary

This post-hoc result concerns target-pair geometry on a fixed 20-kinase
panel. It does not establish ligand-level target retrieval or generalization
to unseen targets.

## Reproduction

```bash
.venv/bin/python analysis/pocket_gated_prior_audit.py \
  --pkis1-zip /private/tmp/pkis1_supplement.zip \
  --kirhub-workbook /private/tmp/kirhub_supp_tables.xlsx \
  --output results/pocket_gated_prior_audit \
  --permutations 50000 --seed 20260821
```
