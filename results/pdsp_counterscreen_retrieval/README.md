# PDSP counterscreen-retrieval audit

This post-hoc analysis asks a graph-level question: can Docking-44 target
geometry prioritize experimental co-affinity partners to include in a
counterscreen panel? It does **not** test ligand-wise target prediction.

## Frozen primary contract

- HUMAN PDSP Ki rows from the allowed recombinant/cloned source vocabulary.
- Exact relations only; positive Ki; valid nonempty molecular structure.
- Median pKi by full Standard InChIKey and target.
- Experimental edge: pairwise Spearman pKi, at least 10 common compounds.
- Predictors use no PDSP values. Fixed fusions average edge percentile ranks.
- Top-edge labels are the upper 10% of experimental edges.

Primary support: 481 compounds,
1560 cells,
148 target pairs, and
21 targets.

Residual docking AUROC is 0.681 versus
0.475 for raw docking. For choosing three
counterscreens per query target, residual geometry recovers the experimental
best partner with macro recall 0.611
versus 0.389. The fixed
sequence+family+residual fusion reaches
0.667; its no-docking sequence+family
baseline reaches 0.556.

## Interpretation boundary and decision

**NO-GO as a general counterscreen selector.** The exact-only, mean-imputed
comparison is encouraging, including after Docking-44 compounds with PDSP
connectivity overlap are removed. It is not robust enough for an operational
claim. The advantage weakens when explicit right-censored non-binders are
restored and largely disappears when the docking map is rebuilt on chemically
shifted complete-case support. Pair endpoints also use different compounds and
publication campaigns. The result is therefore an exploratory, conditional
association rather than a prospective counterscreen rule.

Raw PDSP data are not included because no redistribution license was located.
Download from `https://pdsp.unc.edu/databases/kiDownload/download.php` and verify the checksum recorded in
`summary.json`.
