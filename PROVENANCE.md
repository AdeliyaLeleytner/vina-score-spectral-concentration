# Data and computation provenance

## Reproducibility boundary

The default build regenerates the manuscript ledger, tables, figures and PDFs from two
checksum layers:

- `data_manifest.csv` records frozen matrix/activity inputs;
- `results/manuscript_source_manifest.csv` records every frozen result artifact consumed by
  the report layer.

`analysis/verify_inputs.py` checks byte size and SHA-256 before the ledger is rebuilt. This
release does not rerun receptor preparation, pose generation or Vina docking. Those stages
belong to the cited source data releases. Source-restricted assay workbooks and database
snapshots are not silently downloaded by `make all`.

## Large Vina matrices

### Docking-44

- Frozen matrix: `data/frozen/df_final_v4.csv.gz`.
- Source: Nikitin et al., *Pharmaceutics* 2025, DOI
  `10.3390/pharmaceutics17121573`.
- Analysis support: 12,651 ligands × 44 targets; Vina-GPU 2.0 scores.
- The upstream workflow replaced positive scores with zero before release. The frozen
  analyzable matrix therefore contains no value above zero and 111 cells exactly at zero;
  reapplying the upper-zero rule makes no further change. Its 8,159 missing cells (1.466%)
  are filled by the target mean for the primary analysis, followed by column
  standardization.
- Mean, median, observed-cell additive least-squares and complete-case sensitivities are
  separate estimands.

### DOCKSTRING-58

- Frozen matrix: `data/frozen/dockstring-dataset.tsv.gz`.
- Source: DOCKSTRING v1, DOI `10.6084/m9.figshare.16511577.v1`.
- License: Apache-2.0.
- Source support: 260,155 molecules × 58 targets. The source contains 339 missing scores
  (0.00225%) across 95 rows.
- Analysis support: the 260,060 rows complete across all 58 targets; removing 95 rows
  changes the molecular support by 0.0365%. Target-mean and target-median analyses on all
  260,155 rows give raw/residual PR 2.267/18.214, compared with 2.266/18.206 on complete
  rows.
- The source contains 5,413 positive scores, 5,393 of them in retained rows. These are
  clipped to zero in the primary analysis; retaining them is a reported sensitivity.

The local manifest contains the published score and identity matrices. Receptor structures,
target configurations and upstream docking commands are provided by the checksum-anchored
DOCKSTRING source release rather than duplicated here.

## Spectral and physicochemical analysis

The primary correlation-spectrum estimand gives each target unit variance. The covariance-
spectrum sensitivity retains target-specific variances. Two-way residuals remove fitted
additive ligand and target effects. Transformation-matched parallel, additive Gaussian,
empirical target-marginal and row-norm-preserving nulls are stored in
`results/evidence_summary.json` and selected into the manuscript ledger.

Physicochemical slopes use RDKit molecular weight, heavy-atom count and rotatable-bond
count. Docking-44 uses all 12,651 rows; DOCKSTRING uses a fixed score-blind 15,000-row sample
(seed 71). RDKit rotatable bonds are a proxy and are not represented as Vina's internal
torsion term.

The stricter predictive decomposition uses seven descriptors: heavy-atom count, molecular
weight, Labute ASA, TPSA, cLogP, RDKit rotatable-bond count and ring count. Five-fold
GroupKFold uses frozen Docking-44 Butina groups or DOCKSTRING Murcko groups, with acyclic
DOCKSTRING compounds treated as singletons. Every target offset, residual target scale,
descriptor scale and coefficient is fitted within the training fold. The pooled surface
therefore contains only out-of-fold ligand predictions.

Rank-matched controls use the DOCKSTRING reference after excluding structural overlaps with
DAVIS, PKIS2 and PKIS1, and the ordered common-20 kinase panel. KiRHub lacks released
structures and is covered by the documented name-overlap sensitivity instead. Every arm
receives the same five scaffold folds, fold-local target offsets and
scales, feature standardization and one separately fitted coefficient vector per target.
The seven-dimensional chemical control contains total Morgan-environment count plus six
random projections of unnormalised radius-2, 2,048-bin Morgan count fingerprints. Twenty
fixed projection seeds quantify algorithmic sensitivity. The nuisance controls are a
deterministic seven-coordinate BLAKE2b mapping of exact input SMILES and 20 fixed row
permutations of the standardized physicochemical feature cloud. The latter preserve feature
dimension and covariance but destroy ligand--feature assignment. Their seed ranges are not
sampling confidence intervals.

Mean out-of-fold target R² and reduction in mean squared target correlation quantify
held-out predictive performance. Correlation between a fitted component's target map and an external
map is reported only beside those performance measures: target-specific coefficient noise can
inherit a low-rank projection of training target covariance, and column correlation removes
the component's scale. Accordingly, component-map agreement alone is not interpreted as
feature-specific mechanism, mediation or biological orientation.

The exploratory molecular-size-domain analysis compares groups at the empirical 25th and
75th molecular-weight thresholds, retaining every boundary tie. It uses paired chemical-group bootstrap, simple-random finite-support controls
and MW-matched group-disjoint controls. Molecular weight was selected after preliminary
inspection. A post hoc extension repeats the split for the complete seven-descriptor family;
the strongest shifts in both matrices are the correlated Labute-ASA, molecular-weight and
heavy-atom-count splits, while the other descriptor splits are weaker. MW-domain residual
PR moves in opposite directions between Docking-44 and DOCKSTRING-58. These outputs describe
transfer across the observed source-library supports and do not identify a causal descriptor,
a monotone rank relation or a universal transport boundary.

The secondary fixed-pose cross-scorer outputs are recorded in the manuscript-source checksum
manifest. ODDT 0.7 predates the report-layer environment, so RF-Score, NNScore and PLECscore
rescoring ran under a separate Python 3.9 environment. The default Python 3.12 container and
`make all` verify those outputs but do not recreate them; a source rerun additionally needs
the external DOCKSTRING pose archives and smina. The criteria used for this sensitivity are
descriptive and weak, and a broadly positive score matrix can yield a near-uniform leading
vector without establishing a useful shared axis.

## Experimental target-pair panels

The strict external geometry analysis fixes 20 kinases and 190 target pairs across four
sources:

- DAVIS-Complete V3 quantitative kinase-affinity table, exact CC0 TSV export from Wu
  (2025), Harvard Dataverse DOI `10.7910/DVN/RTQGP1`; the affinity measurements originate
  from Davis et al. (2011), DOI `10.1038/nbt.1990`;
- PKIS2 single-dose KINOMEscan data from the official S4 workbook of Drewry et al. (2017),
  DOI `10.1371/journal.pone.0181585`;
- PKIS1 data from the publisher supplement of Elkins et al. (2016), DOI
  `10.1038/nbt.3374`;
- KiRHub residual-activity data from Saifudeen, Zhu, Liang et al. (2026), DOI
  `10.1038/s41587-026-03090-8`.

PKIS1 and KiRHub source archives are not redistributed. Checksum-validating retrieval
helpers are provided as `analysis/fetch_pkis1_supplement.py` and
`analysis/fetch_kirhub_supplement.py`; `make fetch-source-inputs` downloads both to the
git-ignored `downloads/source_restricted/` directory. The KiRHub article is published
under CC BY-NC-ND 4.0, but this package does not assert that the article license separately
licenses its underlying inhibition data. The official KIRHub portal states that the kinase
inhibition data are the property of Reaction Biology Corporation and require proper
acknowledgement for download, use or publication. The analysis reads a user-obtained,
checksum-verified workbook and writes aggregate analytical summaries; neither the workbook,
compound-by-target activity rows nor per-compound profiles are included. The 190-edge
target-pair geometry is included because each entry is an aggregate correlation coefficient
computed across compounds rather than a compound-level value or profile;
`DATA_LICENSES.md` records this release boundary alongside the checksums and source terms.

Experimental target-pair geometry is calculated from correlations before and after two-way
centring. The complete $2\times2$ predictor-by-endpoint table uses the same 20 targets. The
paired target-label QAP relabels whole target identities. Its complementary
transformation-matched null independently permutes ligand identities inside each docking
target column on one fixed 15,000-ligand reference support, derives raw and centred maps
from the same permuted matrix, and holds the experimental endpoint fixed. These nulls test
target correspondence and the mechanical effect of centring, respectively. The
15-pair binary endpoint is fixed by top-decile co-response in at least two of DAVIS, PKIS2
and PKIS1 before KiRHub is inspected. The exact 15 pairs are frozen in
`results/klifs_pocket_control/target_pairs.csv`. KiRHub was inspected only after this set
was fixed, and no KiRHub result was used to alter it; the ordering was not preregistered.

The DAVIS fixed-20 block has 965 of 1,440 cells (67.0%) at the released 10,000-nM Kd
upper bound, corresponding to a pKd floor of 5. The censoring sensitivity replaces the
continuous DAVIS surface with binary above-floor status, removes targets with little
above-floor support, and rebuilds the three-panel endpoint under the binary representation.

To prevent direct chemical reuse, 1,050 experimental Standard-InChI connectivity blocks are
excluded from the DOCKSTRING reference. A cyclic-Murcko exclusion and five fixed score-blind
15,000-row supports are strict common-20 sensitivities. KiRHub supplies names but not
structures, so KiRHub-specific identity/scaffold exclusion is unavailable.

## Protein controls

Receptor-domain sequence identities are calculated from the exact DOCKSTRING receptor
sequences. KLIFS pocket identities use aligned 85-residue kinase pockets and frozen group
labels. The KLIFS API response itself is not redistributed; its checksum, retrieval
contract and derived 190-pair table are retained. Unrestricted and within-KLIFS-group
target-label permutations are reported separately.

The Secondary Pharmacology Database sensitivity uses the open Novartis release at Zenodo
record `8103950`. The primary analysis contains 59 of 66 pairs among 12 mapped target groups
and is 91.6% censored. It is treated as a censoring boundary rather than quantitative
affinity validation.

## Dense matched operational panels

The dense ranking checks recompute pinned-RDKit Standard InChIKeys for DOCKSTRING and the
released DAVIS and PKIS2 structures. They retain 59 ligands by 21 targets in DAVIS and 154
by 21 in PKIS2. All DOCKSTRING rows sharing an evaluation connectivity block are removed
before fitting target means, scales or the ligand-invariant docking target prior.

DAVIS uses pKd ordering, excludes exact ties and pairs with two released pKd-5 floor values,
and reports both-uncensored and floor-versus-uncensored strata separately. PKIS2 uses
single-dose percentage inhibition/displacement with a 10-percentage-point primary margin;
its primary both-active sensitivity requires both values to be at least 65% and preserves
the aggregate difference margin of more than 10 percentage points; an exact-non-tie version
is retained as a secondary sensitivity. The strata are
outcome-conditioned, post hoc and unadjusted for multiplicity.

The operational path is absolute Vina, target-offset removal, residual-target-SD scaling and
then ligand-row correction under those scales. Path contrasts are paired but correlated and
unadjusted. Delete-one-target analyses refit every transformation on the remaining
20-target reference. All intervals remain conditional on the fixed 21-target panel.

## ChEMBL spectral control and observed-pair ranking

ChEMBL records are from release 34 and retain CC BY-SA 3.0.

- The matched spectral control contains 74 ligands × 6 targets and aggregates exact-relation
  Ki, Kd, IC50 and EC50 values by the median.
- The broad primary operational endpoint refetches the 31 human single-protein targets
  mapped to DOCKSTRING without a potency floor or assay-type restriction. It requires a
  non-null pChEMBL value, exact standard relation and endpoint Ki, Kd, IC50 or EC50. The
  source pull contains 212,254 records; endpoint filtering retains 196,889, and 196,743
  remain after removal of structures that cannot be parsed.
- Full Standard InChIKeys computed with the pinned RDKit version match aggregated ChEMBL
  cells to DOCKSTRING. The primary support retains 6,557 ligands with at least two observed
  targets; 6,480 contribute 25,214 non-tied comparisons over 18,020 observed cells. No
  unobserved activity is imputed. Experimental ties are excluded and predicted ties receive
  half credit.
- A less assay-heterogeneous, endpoint-restricted arm requires a human binding assay and
  endpoint Ki or Kd. Its comparisons can still pair a Ki-derived cell with a Kd-derived cell.
  Of 3,955 pairs, 3,610 (91.3%) use the same single endpoint, 33 (0.8%) are unambiguous
  Ki--Kd comparisons and 312 (7.9%) contain at least one cell aggregated from both endpoint
  types; 71 ligand--target cells pool Ki and Kd records.
  Separate Ki, Kd, IC50 and EC50 arms never mix endpoint types within a target pair and
  expose the opposite-sign Ki/Kd results and wider endpoint heterogeneity.
- Every broad-benchmark target mean, offset and scale is fitted after excluding the complete
  evaluation set from DOCKSTRING. Evaluation-ligand row means use all 58 docking targets.
  Target-only controls include a docking target-mean prior, an external ChEMBL target-mean
  prior and a cohort outcome prior refitted after excluding each ligand's complete
  Bemis--Murcko cluster; the last is outcome-informed and non-deployable.
- Paired uncertainty separately resamples ligands, intact cyclic Murcko scaffolds and intact
  Morgan-radius-2, 2,048-bit Butina clusters at Tanimoto similarity at least 0.65. The
  chemical interval is the union of the two cluster intervals. Delete-one-target
  jackknifing masks one experimental target at a time and reevaluates the externally fitted
  score representations; it is a fixed evaluation-panel composition sensitivity, not a
  confidence interval for all proteins.
- The original Docking-44 endpoint (137 ligands, 38 targets, 691 observed cells and 2,522
  non-tied comparisons) is retained only as a legacy sparse sensitivity. Its primary scaled
  residual uses the row mean over all 44 Docking-44 targets; the 38-target row-mean analysis
  belongs only to that sensitivity.

In every cluster bootstrap, a resampled cluster contributes all of its ligands. Broad
chemical intervals remain conditional on the fixed 31 target labels; legacy Docking-44
target-deletion ranges remain conditional on its fixed 38-target panel.

## Numerical authority

`results/manuscript_evidence.json` is the compact authority for the submitted manuscript.
It contains selected evidence, analysis boundaries and hashes of all source-result files.
`results/evidence_macros.tex` and `results/supplement_tables.tex` are generated from that
ledger and must not be edited manually. Software versions are pinned in
`requirements.lock`; individual randomized analyses record their own seeds.
