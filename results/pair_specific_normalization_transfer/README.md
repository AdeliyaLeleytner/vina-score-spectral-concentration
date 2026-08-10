# Target-pair normalization transfer: an operational no-go

This science-only analysis tests whether the average effect of target-specific
Vina scaling hides reproducible target-pair regimes.  A discovery panel learns,
for each fixed target pair, whether absolute or two-way-residual Vina gives
higher experimental pairwise-preference concordance.  That lookup table is
then frozen and applied to different compounds in another assay panel.

## Primary exact-full-key result and decisive baseline

- **PKIS2 → DAVIS:** pair-rule 0.537; absolute 0.491; residual 0.499; target-mean prior 0.759.
- **DAVIS → PKIS2:** pair-rule 0.550; absolute 0.525; residual 0.529; target-mean prior 0.684.

The normalization-choice pattern is statistically aligned: its locked mean
gain over the discovery-selected coherent global Vina mixture is
**+0.0304**.  A joint
target-label QAP gives **p=0.00002**.
The bootstrap independently resamples whole chemotype clusters in discovery
and validation and refits the rule and global alpha in every draw; its interval is
**[+0.0197, +0.0415]**.
The leave-one-target-out mean gain is positive in
**20/20** deletions, with
range **[+0.0205,
+0.0404]**.

That statistical pattern does **not** supply an operational advance.  A
ligand-independent baseline transferring only the discovery panel's target
means reaches **0.759**
for PKIS2 → DAVIS and **0.684**
for DAVIS → PKIS2, far above the pair rule.  On cells where this target-mean
prior is wrong, the transferred pair rule reaches only
**0.463** and
**0.440**, versus
absolute Vina **0.524** and
**0.508**.  Thus the apparent
transfer mainly reinforces stable target marginals rather than recovering the
ligand-specific exceptions that matter for selectivity.

PKIS2 and DAVIS use the already frozen exact-full-Standard-InChIKey mappings.
Only two complete-support DOCKSTRING rows overlap the two panels; excluding
every profile touching either row is reported separately.  PKIS1 is not part of
the primary claim because it has no exact-full-key overlap and is included only
as a 220-connectivity-block sensitivity.

## Decision and interpretation boundary

**Operational decision: NO-GO.**  This result must not be promoted as an
improved target-ranking or counter-screen method.  Its defensible use is a
mechanistic diagnosis: the effect of target scaling is reproducibly
pair-specific, but much of the transfer reflects stable target-level assay or
library marginals.

The transferred object is a lookup table for **the same fixed target pairs**.
It learns pair identity from discovery experimental outcomes but never uses a
validation ligand outcome.  Pairwise choices can form cycles, so the rule does
**not** define a coherent global target ranking.  More importantly, it fails
the target-mean baseline and the prior-reversal stress test.

All analyses are post hoc/exploratory because the resources had been inspected
elsewhere in the project.  Target-label QAP tests alignment rather than
prospective confirmation; the target jackknife remains conditional on this
finite 20-kinase family.

## Reproduction

```bash
.venv/bin/python analysis/pair_specific_normalization_transfer.py \
  --pkis1-zip /tmp/pkis1_supplement.zip \
  --output results/pair_specific_normalization_transfer \
  --qap-permutations 50000 \
  --bootstraps 2000 --seed 20260821
```
