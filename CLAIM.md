# Claim map

## Scope

- Intended venue and type: *Journal of Cheminformatics*, Methodology article.
- Object of inference: a target-correlation map conditional on the scoring
  pipeline, receptor panel and ligand support.
- Status: exploratory, post hoc and finite-panel. The study was not
  preregistered and makes no target-superpopulation claim.

## Central claim

Raw docking target correlations can be dominated by a ligand-wide scoring axis.
The map obtained after subtracting each ligand's mean across targets retains structured geometry,
but that map depends on the ligand library and is experimentally informative
only on some panels.

The primary PDSP graph provides the clearest descriptive external contrast. On
target pairs supported by quantified $K_i$ measurements, the residual map is
closer to the experimental map than the raw map. The eligible full Docking-44
support is used only after ligands sharing a PDSP connectivity block have been
removed.

This contrast is post hoc and does not transfer unchanged across support
definitions. Restoring censored PDSP cells weakens it, and residual agreement
largely disappears on the chemically shifted complete-case Docking-44
support. Paired QAP probabilities are two-sided, with Holm and Bonferroni
correction within the 12-row primary QAP table. Adding residual docking to a
sequence-and-family baseline is retained as a sensitivity analysis, not as a
headline or statistically resolved result. Kinase, SPD and KiRHub results are
also mixed.

This supports residual target geometry as a diagnostic worth testing on a
declared ligand support. It does not support affinity prediction, universal
improvement beyond homology, unseen-target retrieval, docking-quality
assessment, a general counterscreen selector, or ranking targets for an
individual ligand.

## Evidence and boundaries

| Claim | Evidence | Required boundary |
| --- | --- | --- |
| The shared axis is not specific to one scoring output. | Eight outputs on identical fixed poses have leading-loading cosine 0.950--0.986 with the uniform target direction. | The outputs share Vina poses and some scorer lineages; this is not a comparison of independent docking pipelines. |
| Residual structure is not produced by centring alone. | Residual participation-ratio dimensions are 9.3/43 and 18.2/57; four matched-null medians are at least 34.9 and 47.0. | This rules out the tested mechanical explanations, not all nuisance structure or a biological mechanism. |
| The map depends on ligand support. | Low/high molecular-weight quartile map agreement is 0.205 and 0.351. | Molecular weight was chosen post hoc and covaries with chemistry; it is a domain marker, not an identified cause. |
| Some external panels favour residual geometry descriptively. | On the primary exact-value PDSP graph, the residual map is closer to experiment than the raw map after connectivity-overlap exclusion. | The comparison is post hoc and conditional on its target-pair and ligand support. |
| Censoring and docking support matter. | PDSP residual agreement weakens with censored cells and largely disappears on chemically shifted complete-case Docking-44 rows. | The complete-case analysis changes chemical support; it is not a pure imputation test. |
| Multiplicity limits the fusion result. | Paired QAP inference is two-sided and corrected by Holm and Bonferroni within the 12-row primary QAP table. | Sequence-and-family fusion is a sensitivity, not a significant or headline claim. |
| A small source-library sample can recover the map. | At 200 ligands, recovery is 0.942 and 0.918. | This is same-library geometry recovery, not a universal threshold or affinity prediction. |
| Ligand-wise ranking is a different task. | On the Docking-44 boundary audit, top-5 retrieval falls from 23.6% with absolute Vina to 16.4% with the residual map. | Never describe centring as a general compound-level target-ranking improvement. |

## Required wording

- Prefer “shared ligand-wide score axis”, “residual target map”, “declared
  ligand support” and “source-library recovery”.
- Call PDSP and other external comparisons exploratory and name their support.
- State that PDSP-overlapping Docking-44 connectivity blocks were excluded.
- Describe paired QAP inference as two-sided with both Holm and Bonferroni
  correction within the 12-row primary QAP table.
- Keep sequence-and-family fusion in sensitivity analysis; do not call it a
  significant or headline result.
- Call null ranges simulation ranges, not target-superpopulation confidence
  intervals.
- Say that kinase and SPD evidence is mixed and that KiRHub is a
  non-replication.
- Keep target-pair geometry separate from compound ranking, affinity accuracy,
  docking quality and biological mechanism.
- Unresolved contrasts do not establish equivalence or absence of an effect.

## Outside the submitted claim

Prospective screening performance, proteome-wide target prediction, a universal
200-ligand rule, model-family leaderboards, causal structural mechanisms, and any
claim that docking generally outperforms sequence-based selection.
