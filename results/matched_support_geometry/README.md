# Matched-support docking/experiment target-pair geometry

**Question.** When the docking target-pair geometry and the experimental target-pair geometry are computed on exactly the same compounds, is their agreement higher, lower or indistinguishable from the cross-support agreement the manuscript reports?

Status: `exploratory_post_hoc_reviewer_requested`. Primary endpoint: 20 kinases, 190 target pairs.

## Headline

| panel | matched ligands | matched support | broad support | matched - broad | conservative 95% union (matched - broad) |
|---|---|---|---|---|---|
| DAVIS | 59 | 0.3234 | 0.2977 | +0.0257 | [-0.1413, +0.1325] |
| PKIS2 | 154 | 0.3100 | 0.3251 | -0.0151 | [-0.1368, +0.0684] |

Verdict: indistinguishable: no panel's conservative chemical-cluster bootstrap interval for matched-minus-broad support excludes zero

The experimental endpoint is held fixed at the two-way-centred experimental
geometry of the matched compounds; only the docking support changes between
the two arms.

## Files

- `summary.json` - full record, seeds, claim boundary.
- `arms.csv` - every geometry arm with its target-label QAP.
- `support_contrast_bootstrap.csv` - paired Bayesian bootstrap by ligand,
  Murcko scaffold and Butina cluster.
- `propensity_matching.csv` - standardized mean differences before and after
  molecular-weight / cLogP / TPSA matching of the broad support.
- `sampling_noise.csv` - split-half and subsample reliabilities at these N.
- `matched_ligands.csv` - the matched compounds, their DOCKSTRING rows,
  matching descriptors and chemical-cluster labels.

## Claim boundary

This does NOT show that the docking geometry is target-resolved, that it is mechanistically informative, or that it ranks compounds. It compares two estimators of the same docking target-pair geometry against one fixed experimental endpoint and asks only whether restricting the docking support to exactly the compounds that were screened changes the agreement. The matched arms have 59 and 154 ligands, so their intervals are wide and a real difference the size of the manuscript's own low- versus high-molecular-weight contrast is not excluded. Agreement of two target-pair maps is a network-level statement and implies nothing about ligand-level retrieval. Nothing here addresses the separate objection that the descriptor surface fits per-target regression coefficients and is therefore not target-blind. PKIS1 and KiRHub are not analysed: KiRHub publishes no structures and PKIS1 requires an external supplement that is not a frozen input of this package.
