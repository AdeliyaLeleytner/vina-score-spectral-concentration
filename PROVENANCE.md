# Data and computation provenance

## v5 evidence boundary

The v5 report build starts from frozen score matrices and analysis result
bundles. It regenerates figures, checks manuscript-facing values and compiles the
article and Supplement. Source-level reanalysis is separate and may require
nonredistributed files listed in `external_sources.csv`. No default target reruns
receptor preparation, search-box selection, pose generation or docking.

The older strict-public v4 evidence builder and
`results/public_core_source_manifest.csv` remain available for historical
reconstruction. They do not define the full v5 evidence base.
`results/v5_report_source_manifest.csv` instead records the exact local code,
TeX and frozen inputs consumed by the v5 report build; it is a report-level, not
source-level, provenance contract.

## Docking matrices

### Docking-44

- Frozen matrix: `data/frozen/df_final_v4.csv.gz`.
- Source: Nikitin et al., *Pharmaceutics* (2025), DOI
  `10.3390/pharmaceutics17121573`.
- Licence: CC BY 4.0; redistributed with attribution.
- Analysis support: 12,651 ligands by the 44 targets in the published source
  panel. The carrier file has three additional appended PDB columns---8pm6
  (BSEP), 6itm (FXR) and 9fzj (PXR)---which are excluded by the source-panel
  identity rule before any score statistic is computed.
- The frozen matrix has 8,159 missing cells. The primary analysis uses
  target-mean imputation followed by target standardisation; median,
  observed-cell additive and complete-case variants are reported separately.
- The released upstream processing set positive values to zero. Reapplying that
  rule does not alter the frozen matrix.

### DOCKSTRING-58

- Frozen matrix: `data/frozen/dockstring-dataset.tsv.gz`.
- Source: DOCKSTRING v1, DOI `10.6084/m9.figshare.16511577.v1`.
- Licence: Apache-2.0.
- Source support: 260,155 molecules by 58 targets. The primary complete-case
  analysis retains 260,060 rows; missing-score and positive-score sensitivities
  are reported separately.

The release starts from these published score products. Receptor structures,
target configurations, pose archives and upstream docking commands remain with
the source studies, except where a fixed-pose analysis retrieves and verifies a
named DOCKSTRING archive.

## Target-map estimands

The primary correlation spectrum gives each target unit variance. The covariance
sensitivity retains target-specific variances. The residual matrix subtracts
fitted additive target and ligand effects; on complete rectangular matrices this
is equivalent to centring each target column and then subtracting each ligand's
mean in the original score units.

Four transformation-matched nulls pass through the same centring pipeline. They
preserve different marginal features while breaking target correspondence.
Their purpose is to test whether the residual spectrum can arise from the
transformation and preserved nuisance structure alone; they do not identify a
biological mechanism.

Molecular-weight and other descriptor splits diagnose support dependence.
Descriptor axes are correlated and were inspected post hoc. Chemical-group
controls use inherited Docking-44 groups or standardised-parent/scaffold blocks
for DOCKSTRING. These are fixed-library sensitivities, not transport guarantees.

Calibration recovery samples ligands from the same source library and compares
the resulting edge ranks with the full source-library map. Whole-group-held-out
variants keep chemical groups disjoint. Repeated-split ranges are conditional on
the observed targets and library.

## Fixed-pose scorer comparison

The eight-output comparison uses the same 2,000-ligand by 9-target block of
retained DOCKSTRING Vina poses. The source pose archive identifiers and MD5
values are embedded in `analysis/dockstring_vina_term_decomposition.py`; the
producer verifies them at runtime. ODDT training-table and model checksums,
per-target DOCKSTRING receptor checksums, and the smina binary checksum and
version are recorded in `results/nonvina_scorer_transport/summary.json`.

This design holds pose selection fixed. It tests whether the leading score axis
persists across scoring functions on identical poses, not whether independent
docking pipelines choose the same poses or have equal predictive accuracy.

## External pharmacology

### PDSP

The PDSP analysis uses a manually downloaded $K_i$ export whose SHA-256 is
recorded in `results/pdsp_counterscreen_retrieval/summary.json`. No redistribution
licence was located, so the row-level source is not shipped. Aggregate target-pair
tables contain no compound-level activity profiles and make the reported map
comparisons auditable. Sequence identities use checksum-recorded UniProt
responses. Continuous exact-value, bound-insertion, binary and certified-subset
endpoints expose the censoring boundary.

The exact UniProt response can be fetched without a manual query:

```bash
.venv/bin/python analysis/fetch_pdsp_uniprot.py \
  --output /path/to/pdsp_uniprot.tsv
```

The fetcher requests reviewed human records for the 27 frozen gene labels,
sorts them by accession and retains accession, gene names, sequence and length.
It writes only the checksum-matched response (SHA-256
`051690a9647c3af3ad71708d935485c73bdbffc07772b0ec8c6f484e9c1b408a`): 28
records plus the header. One returned record has primary symbol `ADRA1D` and an
`ADRA1A` synonym; the analysis uses primary gene symbols and therefore ignores
that extra record. `--print-url` prints the full encoded REST request.

The 12 paired-QAP rows comprise two contrasts, three metrics and two permutation
schemes. Holm and Bonferroni adjustments are computed across this declared
family. The analyses are post hoc and conditional on the observed target graph.

### Novartis Secondary Pharmacology Database

The source is the CC BY 4.0 Zenodo release at record `8103950`. The accepted file
checksum is recorded in `results/spd_external_validation/summary.json`.
Target-pair summaries are released; the row-level source is obtained from Zenodo.
The panel is highly censored, so bound-rank and threshold endpoints are reported
separately. Homology-adjusted results are endpoint-dependent.

### Kinase panels and KiRHub

The fixed kinase comparisons use DAVIS-Complete, PKIS2, PKIS1 and KiRHub on a
declared common target set. DAVIS and PKIS2 are redistributed under their source
licences. PKIS1 and KiRHub are fetched with checksum-gated helpers. KiRHub
compound-level data are not redistributed; Reaction Biology Corporation is
acknowledged as required. Released target-pair correlations are aggregate values
computed across compounds.

Sequence and KLIFS pocket identities are separate baselines. Target-label QAP
relabels complete target identities; within-family variants preserve the curated
homology grouping. The panels share targets and may share chemistry, so they are
not independent samples.

## Ligand-wise ranking boundary

ChEMBL release-34 records retain CC BY-SA 3.0. The v5 boundary audit uses the
checksum-identified extract in `data_manifest.csv`, exact standard relations and
the endpoint rules recorded in
`results/residual_mie_boundary_audit/analysis_summary.json`. It releases aggregate
metrics only. Experimental ties are excluded and predicted ties receive half
credit.

Target-map geometry and within-ligand target ranking are different estimands.
The ranking analysis is included to prevent improvements in target-pair geometry
from being misread as improved compound-level target ordering.
Delete-one-target jackknifing reevaluates the externally fitted
  score representations on each reduced target set; it does not refit the
transformations.

## Numerical authority and software

Frozen CSV and JSON result bundles are the numerical authorities for v5.
`analysis/test_v5_manuscript_numbers.py` and
`analysis/test_v5_supplement_numbers.py` bind manuscript statements to them.
Figure drivers read the same files. Values typed into TeX are not independent
authorities.

Python dependencies are pinned in `requirements.lock`; convenience
specifications are provided in `requirements.txt` and `environment.yml`.
Randomised analyses record their seeds and parameters within their result
bundles. See `REPRODUCIBILITY.md` for the report build and clean-archive release
procedure, and `DATA_LICENSES.md` for redistribution terms.
