# Chemistry-support edge stability: NO-GO

This science-only artifact tests whether a residual docking target edge that is
stable across chemistry-disjoint ligand supports is more likely to reproduce as
experimental target co-selectivity.  The prespecified design uses
50 mutually scaffold-disjoint DOCKSTRING
supports of exactly 1995 ligands.  It also
includes row-disjoint random-support and uncentered raw-score controls.

## Result

**NO-GO.** In the prespecified 50 x 1,995 design, inverse between-support edge rank variability does not meet the locked-panel effect, conditional-QAP, and target-jackknife criteria for incremental co-selectivity information beyond mean docking strength and fixed controls.

The primary candidate adds inverse between-support percentile-rank variability
to a baseline containing mean residual docking edge strength, receptor-domain
sequence identity, KLIFS ATP-pocket identity, and two target-breadth pair
features estimated from an outcome-blind MaxMin selection of 80 PKIS1 compounds.
PKIS1 fits the fixed class-balanced ridge score. DAVIS, PKIS2, and KiRHub are
locked evaluations.  Conditional QAP relabels only the stability network while
holding edge strength, all controls, and endpoints fixed.

The GO rule was fixed before this run: locked-panel mean improvements of at
least +0.05 Spearman, +0.07 AUROC, and +0.04 average precision; no negative
locked-panel delta; one-sided conditional-QAP p < 0.05 for every metric; and a
positive locked mean after at least 18 of 20 target deletions for every metric.
An initial whole-group 50 x 2,000 allocator was rejected before inference when
an outcome-blind audit showed that it overselected dominant analogue series.
The released 50 x 1,995 size is the mathematical maximum for 99,755 groups with
one representative per group; the model, endpoints, statistics, and GO
thresholds were not changed in response to the result.

## Reproduce

```bash
python analysis/edge_stability_confidence.py \
  --pkis1-zip /tmp/pkis1_supplement.zip \
  --kirhub-workbook /private/tmp/kirhub_supp_tables.xlsx \
  --output results/edge_stability_confidence \
  --supports 50 \
  --support-size 1995 \
  --permutations 50000
```

The external PKIS1 and KiRHub inputs are checksum/provenance-validated by the
reused project loaders. Compound-level experimental values and KiRHub-derived
pair values are not written.
