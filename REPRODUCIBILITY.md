# Reproducibility contract

## Deterministic report-layer build

```bash
make PYTHON=.venv/bin/python all
```

The dependency graph is:

1. `verify`: validate `data_manifest.csv` and
   `results/manuscript_source_manifest.csv` by byte size and SHA-256.
2. `test`: run focused fixed-20, overlap, centring-panel and evidence-ledger tests.
3. `evidence`: rebuild `results/manuscript_evidence.json` from the verified source-result
   artifacts.
4. `validate-evidence`: assert PR identities, residual zero-mode and null-model invariants.
5. `reported` and `verify-reported`: generate the LaTeX macro registry and fail on known
   numeric drift.
6. `tables`: regenerate all supplementary tables.
7. `main-figures` and `supplement-figures`: regenerate the five main and six supplementary
   figures used in the PDFs.
8. `manuscript` and `supplement`: compile both LaTeX documents twice.

The default target is intentionally a report-layer build. It does not imply that every
source-restricted third-party workbook is redistributed or that upstream docking is rerun.
Analysis contracts and retrieval boundaries are documented in `PROVENANCE.md`.
The secondary ODDT fixed-pose rescoring is likewise not recreated by the default container.
Its released outputs are checksum-verified; the source run requires ODDT 0.7 under a
separate Python 3.9 environment, external DOCKSTRING pose archives and smina.

The two publisher supplements needed for source-level experimental reruns can be fetched
without guessing URLs or accepted bytes:

```bash
make PYTHON=.venv/bin/python fetch-source-inputs
```

Both helpers download to the git-ignored `downloads/source_restricted/` directory and abort
unless the payload matches the recorded SHA-256. The report-layer build remains offline and
uses the released aggregate result bundles.

## Expected scientific checks

The large-matrix summaries must be:

```text
Docking-44:     PR 1.834 -> 9.303
DOCKSTRING-58:  PR 2.266 -> 18.206
```

The DOCKSTRING source has 260,155 rows, 339 missing cells in 95 rows and 5,413 positive
scores. The complete-row analysis retains 260,060 rows and 5,393 positive scores before
clipping. Target-mean and target-median imputation on all rows give raw/residual PR
2.267/18.214, confirming that the 95-row exclusion is negligible for the headline result.

For transformation-matched parallel analysis:

- Docking-44 rank-wise and simultaneous counts are 3→7;
- DOCKSTRING rank-wise counts are 4→11;
- DOCKSTRING simultaneous family-wise-error-controlled counts are 4→10.

Within each matrix, the additive Gaussian, empirical residual-permutation and
row-norm-preserving nulls use the same fingerprinted deterministic 12,000-ligand support.
The observed PR values on those supports are 9.295 and 18.158. Additive/empirical null
medians are approximately 42.40/42.40 and 56.48/56.48; row-norm-preserving medians are
42.63 and 56.61. Each observed residual PR is below all 500 matched null draws (`p=0.002`).

The strict five-fold chemical-group-held-out descriptor decomposition on the full analysis
supports must reproduce:

| Matrix | Analysis support | Chemical groups | residual PR before → after descriptor removal | reduction in mean squared target correlation | residual-PC1 out-of-fold R² |
| --- | ---: | ---: | ---: | ---: | ---: |
| Docking-44 | 12,651 | 8,150 | 9.294 → 14.746 | 46.9% | 0.586 |
| DOCKSTRING-58 | 15,000 | 11,905 | 17.923 → 28.466 | 53.6% | 0.687 |

All offsets, residual target scales, feature scales and regression coefficients are
fold-local, and chemical-group overlap between train and test is exactly zero in every fold.
This is a predictive decomposition, not causal attribution or evidence that the named
features uniquely identify the source of the correlation.
Repeated chemical-group partitions give full-matrix reduction ranges 0.462–0.470 for
Docking-44 and 0.527–0.546 across the fixed DOCKSTRING supports. These are design
sensitivities, not population confidence intervals.

On the fixed common-20 kinase support after excluding structural overlaps with DAVIS,
PKIS2 and PKIS1, the capacity-matched controls must give:

| Seven-dimensional basis | Mean out-of-fold target R² | Reduction in mean squared target correlation | Component-map agreement range across DAVIS, PKIS2, PKIS1 and KiRHub |
| --- | ---: | ---: | ---: |
| physicochemical descriptors | 0.1068 | 0.2336 | 0.097–0.288 |
| count-Morgan projection, locked seed | 0.0903 | 0.2377 | 0.139–0.359 |
| stable ligand-identity hash nuisance | -0.00004 | 0.00002 | 0.103–0.348 |

Across 20 deterministic count-Morgan projections, the correlation reduction spans
0.1760–0.2600 and 3/20 exceed the physicochemical value. Across 20 deterministic row
permutations of the standardized physicochemical basis, mean out-of-fold target R² remains
between -0.00005 and -0.00002 and the correlation reduction between -0.00007 and 0.00007,
while scale-free component-map agreement spans -0.072–0.564. These ranges are fixed-seed
algorithmic sensitivities, not confidence intervals. The required invariant is therefore
performance-plus-geometry: a fitted-component correlation map is uninterpretable without an
out-of-fold predictive-performance metric and a nuisance-basis control. Column correlation
removes the fitted component's scale; mean out-of-fold target R² quantifies held-out
predictive performance, not physical amplitude.

The exploratory low/high-molecular-weight residual-map agreements must be 0.205
[0.185, 0.219] for Docking-44 and 0.351 [0.326, 0.363] for DOCKSTRING-58. MW-matched
chemical-group-disjoint controls give 0.991 and 0.994. Strict descriptor adjustment gives
0.489 and 0.437, and the six seeded DOCKSTRING supports span 0.325–0.361.
Raw low/high-map agreement must be 0.227 [0.209, 0.246] and 0.409 [0.384, 0.432].
MW-stratified chemical-group-disjoint halves made independently within the low and high
bands must give residual-map means 0.980/0.978 for Docking-44 and 0.978/0.966 for
DOCKSTRING-58; corresponding raw means are 0.989/0.989 and 0.991/0.979. These repeated-
split ranges are composition sensitivities, not population confidence intervals.
Across post hoc extreme splits of all seven inspected descriptors, the three strongest map
shifts in each matrix must be Labute ASA, molecular weight and heavy-atom count: 0.200–0.207
in Docking-44 and 0.322–0.361 in DOCKSTRING-58. The other split-map agreements are
0.703–0.887 and 0.787–0.831, respectively. Low-to-high-MW residual PR moves in opposite
directions, 13.6246→10.1470 for Docking-44 and 19.5521→23.7997 for DOCKSTRING-58.

Same-distribution probe recovery must reproduce full-map Spearman agreement of 0.942 and
0.918 from 200 ligands for Docking-44 and DOCKSTRING-58. Chemical-group-held-out recovery
at the same panel size is 0.927 and 0.916. These values support remeasurement on a small
representative probe library on the observed supports; they do not license transport across
the low/high-MW split or establish 200 ligands as a universal accuracy threshold.

The two dense full-Standard-InChIKey-matched operational checks must reproduce:

| Panel | Matched/evaluated ligands | Targets | Evaluated pairs | Absolute Vina | Scaled two-way residual | Residual - absolute (chemical-cluster union) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| DAVIS | 59/56 | 21 | 6,065 | 0.539 | 0.530 | -0.009 [-0.030, 0.010] |
| PKIS2 | 154/154 | 21 | 16,417 | 0.525 | 0.523 | -0.003 [-0.018, 0.012] |

The ligand-invariant docking target priors must score 0.522 in DAVIS and 0.504 in PKIS2.
Absolute-minus-prior differences are +0.017 [-0.025, 0.059] and +0.021
[-0.006, 0.048], so both chemical-cluster intervals span zero.

The exploratory normalization-path decomposition must also retain opposite steps. The
target-offset changes are -0.013 [-0.033, 0.006] and -0.001 [-0.016, 0.015]. Dividing
target-centred scores by target-specific residual SD changes concordance by -0.0208
[-0.0366, -0.0058] in DAVIS and -0.0174 [-0.0281, -0.0065] in PKIS2. Adding the ligand-row
correction under the same scales changes it by +0.0246 [0.0109, 0.0391] and +0.0151
[0.0055, 0.0253], respectively. These path intervals are post hoc, correlated and
unadjusted for multiplicity. Delete-one-target normal intervals for the net contrast are
[-0.081, 0.061] and [-0.060, 0.053]; all statements remain conditional on the fixed,
non-random 21-target panels.

Exploratory outcome-conditioned checks must retain their sign pattern:

- DAVIS both-uncensored: absolute 0.459, scaled residual 0.503, difference +0.044
  [0.004, 0.089] on 1,757 within-ligand comparisons;
- DAVIS floor-versus-uncensored: 0.547 versus 0.534, difference -0.013
  [-0.037, 0.010] on 4,308 within-ligand comparisons;
- PKIS2 both-active with the primary $>10$-percentage-point margin: 0.517 versus 0.569,
  difference +0.052 [0.002, 0.109] on 256 within-ligand comparisons. Relaxing only the
  margin to exact non-ties is retained as a secondary sensitivity.

These strata are outcome-conditioned and unadjusted for multiplicity.

The strict common-20 docking-versus-experiment factorial must contain:

| Panel | raw/raw | raw/centred | centred/raw | centred/centred | full/matched-support docking increment against fixed raw experiment (`p_label`; `p_matched`) | full/matched-support docking increment against fixed centred experiment (`p_label`; `p_matched`) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| DAVIS | 0.050 | 0.263 | 0.219 | 0.280 | 0.168/0.162 (`.01512`; `.00300`) | 0.016/0.010 (`.335`; `.690`) |
| PKIS2 | -0.215 | 0.243 | 0.033 | 0.287 | 0.249/0.243 (`.00152`; `.00050`) | 0.044/0.034 (`.146`; `.172`) |
| PKIS1 | -0.159 | 0.173 | 0.059 | 0.174 | 0.218/0.215 (`.00536`; `.00050`) | 0.001/-0.006 (`.462`; `.513`) |
| KiRHub | -0.029 | 0.276 | 0.235 | 0.322 | 0.264/0.261 (`.00008`; `.00050`) | 0.045/0.041 (`.126`; `.073`) |

The four paired target-label probabilities for the docking increment against the fixed raw
endpoint all remain below 0.05 after Holm correction across those four tests (adjusted values
0.01512, 0.00456, 0.01072 and 0.00032 in table order).

The matched probabilities use one fixed score-blind 15,000-ligand support and 2,000
independent within-target ligand permutations. Observed increments and null increments are
compared on that same support; full-reference increments remain descriptive.

The mean experimental cross-panel concordance is 0.632 before and 0.763 after centring. The
global target-label network QAP gives `p=.00204` for the increment because it tests target
correspondence across panels. The transformation-matched independent-column null gives
`p=.946527` because it asks whether the increment exceeds the mechanical change induced by
centring. Its raw null mean is approximately -0.0002 rather than the observed 0.632, and
the absolute centred agreement exceeds the same null (`p=.000500`). The two nulls answer
different questions.

The fixed 15-of-190 endpoint, defined before KiRHub inspection, is recovered by centred
KiRHub experimental geometry with AUROC 0.963 and AP 0.694 (absolute target-label QAP
`p=.00002` for each). Paired target-label gain probabilities are 0.04928 and 0.02306
(Holm-adjusted 0.04928 and 0.04612), whereas transformation-matched centred-minus-raw
gain probabilities are 0.748 and 0.997. The absolute centred metrics also exceed the
transformation-matched null (`p=.000500` each). The reproducible claim is therefore the
centred KiRHub level; the centred-minus-raw gain is resolved by label QAP but not by the
mechanical transformation-matched null.

The fixed-20 DAVIS block has 965/1,440 values at the pKd-5 floor. Its continuous and
binary above-floor-status target geometries correlate at 0.885, so the continuous map
largely tracks whether measurements clear the reporting floor. Centred DOCKSTRING
correlates 0.292 with the binary geometry (`p=.00426`), and the association remains
positive after requiring 5, 10 or 15 above-floor values per retained target. The binary
DAVIS replacement preserves 13 of 15 fixed pairs; centred KiRHub retrieves the resulting
16-pair endpoint with AUROC 0.956 and AP 0.673.

The broad primary DOCKSTRING--ChEMBL support must retain 6,557 ligands with at least two
observed targets; 6,480 contribute a non-tied comparison. It contains 31 targets, 18,020
observed cells and 25,214 within-ligand comparisons. Mean concordances are 0.537 for
absolute Vina, 0.530 for column-standardized Vina and 0.529 for the scaled residual.
Target-centred and two-way-centred unscaled concordances are exactly identical at 0.533.
The scaffold-cross-fitted cohort outcome prior scores 0.604 and remains a benchmark-fitted,
non-deployable reference.

Residual minus absolute is -0.0077. The conservative union of Murcko- and Butina-cluster
95% intervals is [-0.0220, 0.0061], conditional on the fixed 31 target labels; the separate
delete-one-target jackknife approximation is [-0.0562, 0.0295]. The latter probes target
composition and is not a population confidence interval for all proteins.
Absolute Vina minus the ligand-invariant target-only prior fitted on the external DOCKSTRING
reference is +0.0134, with a
paired Murcko/Butina conservative interval of [-0.0113, 0.0380]. The broad benchmark does
not resolve ligand-specific ordering information beyond that target-only baseline.

The less assay-heterogeneous, endpoint-restricted human binding Ki/Kd arm must retain 1,056
ligands, of which 1,049 contribute
3,955 non-tied comparisons across 31 targets. Absolute and scaled-residual concordances are
0.559 and 0.520, with residual-minus-absolute -0.0396, chemical-cluster interval
[-0.0914, 0.0029] and target-jackknife approximation [-0.1332, 0.0291]. The combined arm
can still compare a Ki-derived cell with a Kd-derived cell and is not
homogeneous. Of all 3,955 pairs, 3,610 (91.3%) use the same single endpoint, 33 (0.8%) are
unambiguous Ki--Kd comparisons, and 312 (7.9%) contain at least one cell supported by both
endpoint types. Separate endpoint arms retain their heterogeneous sign pattern: Ki -0.0331
[-0.0687, -0.0054], Kd +0.0380
[-0.0081, 0.0801], IC50 -0.0062 [-0.0220, 0.0107] and EC50 +0.0132
[-0.0134, 0.0423]. These are post hoc endpoint sensitivities, not four confirmatory tests.

The original Docking-44--ChEMBL support (137 ligands, 38 targets, 691 cells and 2,522
comparisons) is a legacy sparse sensitivity only. Its mean concordances remain 0.566,
0.546 and 0.564 for absolute, column-standardized and scaled-residual Vina. The all-44 versus
local-38 row-mean analysis belongs only to this legacy sensitivity and must not be reported
as the broad primary benchmark.

## Integrity layers

- `data_manifest.csv` covers frozen matrix/activity inputs used within the archived scope.
- `results/manuscript_source_manifest.csv` freezes every result artifact consumed by the
  report layer, including the exact 15-pair endpoint table.
- `results/manuscript_evidence.json` records source hashes again alongside selected evidence
  and analysis boundaries.
- `results/evidence_macros.tex` is generated, never hand edited.
- `analysis/verify_reported_results.py` rejects known stale numerical fragments.
- The same guard enforces the journal `Scientific Contribution` abstract heading and word
  limit and synchronizes the manuscript title across the manuscript, Supplement, README and
  `CITATION.cff`.
- `analysis/test_released_pair_geometry_ledger.py` checks that the released 190-edge
  target-pair ledger reproduces the factorial's centred cells, the cross-panel agreement
  means and the locked-endpoint retrieval metrics, so the data-availability claim in the
  manuscript is machine-verified rather than asserted.

## Focused commands

```bash
make PYTHON=.venv/bin/python verify
make PYTHON=.venv/bin/python test
.venv/bin/python analysis/build_manuscript_evidence.py
.venv/bin/python analysis/make_reported_results.py
.venv/bin/python analysis/verify_reported_results.py
/Library/TeX/texbin/pdflatex -interaction=nonstopmode -halt-on-error manuscript.tex
```

Use the `PDFLATEX` make variable on systems where `pdflatex` is installed elsewhere.
Numerical results should agree in the pinned environment; PDF bytes need not match across
different TeX distributions even when text, figures, page count and page size do.
