# Novartis SPD external target-geometry validation

Status: **exploratory and post hoc**. This directory is a frozen v5 manuscript
evidence bundle; it does not establish a general performance improvement.

## Question

Does the target--target geometry revealed by centering DOCKSTRING Vina scores
agree with co-response across a broad, non-kinase safety-pharmacology panel?
The operational interpretation is counterscreen prioritization: for a query
target, the geometry may suggest other targets whose responses vary similarly
across compounds. It is not an affinity predictor and does not validate
ligand-level target ranking.

## Fixed analysis contract

- Twelve human targets: ADORA2A, ADRB1, ADRB2, AR, DRD2, EGFR, ESR1, ESR2,
  F2, NR3C1, PGR, and PTGS2.
- Candidate direct binding/inhibition assay groups were fixed before inspecting
  target-pair outcomes. If two campaigns were available, the group with the
  greatest compound coverage was selected; ties go to the smaller group ID.
- Compounds are aggregated by the first 14 Standard InChIKey characters.
- The primary experimental surface converts released IC50 bounds in micromolar
  to pIC50, standardizes each observed target column, removes each observed
  ligand-row mean, and computes pairwise Spearman correlations with at least 40
  common compounds.
- The primary docking predictor removes all DOCKSTRING rows whose connectivity
  block occurs in SPD, clips positive scores to zero, standardizes the 12 target
  columns, removes each row mean, and correlates target columns.
- QAP permutes complete target labels while leaving the experimental endpoint,
  support mask, and controls fixed. A family-preserving version is also shown.

The floor-at-bound surface is not a quantitative affinity surface: 10,188 of
11,126 selected rows (91.6%) are right-censored. Therefore two censor-aware
binary endpoints and an exact-only diagnostic are mandatory, not optional.

## Main results

On 59 target pairs, with all 932 connectivity-overlapping chemical blocks
removed from the docking predictor:

| Experimental endpoint | Raw Vina rho | Residual Vina rho | Difference |
|---|---:|---:|---:|
| Released bound ranks | 0.149 | 0.372 | +0.223 |
| Censor-aware active at 10 uM | 0.348 | 0.388 | +0.040 |
| Censor-aware active at 30 uM | 0.180 | 0.451 | +0.271 |

Unrestricted 50,000-draw target-label QAP gives residual-network probabilities
of 0.0124, 0.0052, and 0.0058, respectively. The residual-minus-raw contrast
is resolved for released-bound ranks (0.0376) and the 30-uM binary endpoint
(0.0206), but not at 10 uM (0.3811). After rank control for full-sequence
identity, curated family, pair support, and target coverage, residual partial
rho is 0.2405 for the released-bound endpoint; its unrestricted QAP probability
is 0.0572. We therefore make no beyond-control claim for that endpoint.

The observation-mask control is favorable. The largest outcome-blind complete
rectangle contains 102 compounds measured on the same 11 targets (all except
EGFR). On its 55 pairs, raw rho is 0.055 and residual rho is 0.575; residual
partial rho is 0.371. Leave-one-target-out residual-minus-raw differences are
positive for all 12 targets.

The fixed 58-target DOCKSTRING residual representation is more concordant by
point estimate than centering only the 12 validation targets (rho 0.571 versus
0.372 on the primary endpoint), but this is reported as a representation sensitivity rather
than used to replace the more conservative local-panel primary result.

## What did not become a headline

- Exact-only measurements are too sparse: residual rho is 0.404 across 28
  pairs at minimum support 5 and -0.006 across only 16 pairs at support 10.
- Family-preserving QAP supports the residual network itself, but not a
  residual-minus-raw gain. Family structure explains part of the advantage.
- In the current per-query best-partner stress test, residual Vina does not
  improve binary top-3 recovery over raw Vina (5/12 versus 6/12 targets at
  10 uM; 6/12 versus 7/12 at 30 uM) and does not beat sequence. This does not
  justify a strong counterscreen-design claim.
- Replacing the selected ESR1, PGR, or PTGS2 campaign leaves residual rho
  positive, but the PGR and PTGS2 alternatives weaken support-adjusted effects.

## Reproduction

Retrieve and verify the two exact upstream snapshots with:

```bash
.venv/bin/python analysis/fetch_spd_source.py /tmp/spd_activity.txt
.venv/bin/python analysis/fetch_spd_uniprot.py /tmp/spd_uniprot.tsv
```

Both commands fail closed on a byte-count, schema or SHA-256 mismatch. The
report-layer release consumes the frozen aggregate outputs below; a clean
source-level replay remains a separate release gate.

The SPD input is not redistributed here.  The official Zenodo record is open
under CC BY 4.0 (metadata checked 2026-08-03). Download the activity export from
the [Novartis SPD Zenodo record](https://zenodo.org/records/8103950). Expected
SHA256: `7132723f85e746de2f8387d01dcde6ffff703c92561fda9751cbd6753e900240`.

The reviewed human UniProt snapshot contains exactly the 12 primary gene
labels. Expected SHA256:
`b0f6213a79286da247fb1bf74c036446fe61e1a3beff074187025fce35015689`.

```bash
.venv/bin/python analysis/spd_external_validation.py \
  --spd /tmp/spd_activity.txt \
  --uniprot /tmp/spd_uniprot.tsv \
  --dockstring data/frozen/dockstring-dataset.tsv.gz \
  --output-dir results/spd_external_validation \
  --qap-permutations 50000 \
  --seed 20260803

.venv/bin/python -m pytest -q analysis/test_spd_external_validation.py
```

The released CSVs contain target-level and aggregate results only. No SPD or
DOCKSTRING compound identifiers, structures, or row-level activities are
written.
