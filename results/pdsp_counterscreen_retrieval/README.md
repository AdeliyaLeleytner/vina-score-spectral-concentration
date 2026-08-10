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

Residual docking AUROC is 0.700 versus
0.474 for raw docking. For choosing three
counterscreens per query target, residual geometry recovers the experimental
best partner with macro recall 0.667
versus 0.389. The fixed
sequence+family+residual fusion reaches
0.722; its no-docking sequence+family
baseline reaches 0.556.

## Interpretation boundary and decision

**NO-GO as a general counterscreen selector.** Although the exact-only graph
passes aligned-target QAP and every target deletion, the operational advantage
does not survive restoration of explicit right-censored non-binders. Pair
endpoints in the pooled graph also use different compounds and publication
campaigns. The defensible positive result is narrower: residual geometry
retrieves affinity relationships *conditional on both targets having
quantifiable Ki values*. That conditional graph is a mechanistic hypothesis,
not yet a prospective counterscreen rule.

Raw PDSP data are not included because no redistribution license was located.
Download from `https://pdsp.unc.edu/databases/kiDownload/download.php` and verify the checksum recorded in
`summary.json`.
