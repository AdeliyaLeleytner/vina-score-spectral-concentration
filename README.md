# Target-correlation maps are library-conditional but recoverable from a few hundred ligands in two large Vina panels

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21874032.svg)](https://doi.org/10.5281/zenodo.21874032)

Reproducibility package for the *Journal of Cheminformatics* Research article:

> **Target-correlation maps are library-conditional but recoverable from a few hundred ligands in two large Vina panels**

The study separates three quantities that are often conflated in multi-target docking:

1. spectral redundancy of a ligand-by-target score matrix;
2. the target-correlation map estimated across a stated ligand library;
3. compound-level ordering of targets.

Participation-ratio (PR) effective dimension is used as a transparent dependence scale. For
a target correlation matrix it is exactly a transformation of mean squared inter-target
correlation, not a new rank estimator, docking-quality score or affinity-accuracy metric.

## Main findings

| Matrix | Ligands × targets | Uncentred correlation PR | Within-ligand-centred PR |
| --- | ---: | ---: | ---: |
| Docking-44 (Vina-GPU 2.0) | 12,651 × 44 | 1.834 | 9.303 |
| DOCKSTRING-58 (AutoDock Vina) | 260,060 × 58 | 2.266 | 18.206 |

The residual maps remain much more correlated than additive, empirical-marginal and
row-norm-preserving nulls. Centring reduces spectral concentration; it does not eliminate it
and is not a quality criterion.

The central practical result is support dependence. In exploratory, post hoc splits within
the two observed source libraries, low- and high-molecular-weight maps agree at only 0.205
and 0.351, whereas chemically disjoint controls with matched molecular-weight distributions
agree at 0.991 and 0.994. The same cross-band shift is already visible on the raw surfaces
(0.227 and 0.409). Conversely, MW-stratified chemical-group-disjoint halves made separately
inside each extreme band retain residual-map agreement of 0.966--0.980 and raw-map
agreement of 0.979--0.991. Thus row centring and restriction alone do not account for the
transport boundary. Across the seven inspected descriptor splits, the strongest shifts
in both matrices form a correlated molecular-size family (Labute ASA, molecular weight and
heavy-atom count); the other splits are weaker. Residual PR moves in opposite directions
across the MW split in the two matrices, so the result is neither a unique descriptor
mechanism nor a monotone quality relation. In descriptive, support-specific sampling
experiments, 200 same-distribution ligands recover the corresponding source-library maps at
0.942 and 0.918, and chemical-group-held-out comparisons remain 0.927 and 0.916. These
values do not define a universal 200-ligand accuracy threshold. A target-correlation map
should be re-estimated after molecular-size-domain shift and reported with its scoring
pipeline, receptor panel and ligand support.

Low-dimensional ligand features have non-zero held-out predictive performance out of chemical group, but the
basis is not unique. On the fixed 20-target support, seven physicochemical descriptors and a
locked seven-dimensional count-Morgan projection reduce mean squared target correlation by
0.234 and 0.238. Across 20 count-Morgan seeds the range is 0.176–0.260.

The rank-matched controls also expose a scale-free projection failure. A stable-hash basis
has mean out-of-fold target R² near zero and correlation reduction near zero, yet its fitted
component map agrees 0.103–0.348 with four experimental maps. Across 20 row-permuted
physicochemical bases, held-out predictive performance remains near zero while map agreement ranges
from −0.072 to 0.564. Target-wise coefficient noise can inherit a low-rank sketch of target
covariance, and column correlation removes the predicted columns' scale. Component-map agreement
therefore requires an out-of-fold predictive-performance metric and a nuisance-basis control; it cannot by
itself establish physicochemical mediation or biological orientation.

Four separately published kinase panels bound the external interpretation. Against fixed
centred experimental endpoints, docking-side centring increments are 0.001–0.045 and are
unresolved; against uncentred endpoints they are 0.168–0.264. Sequence and KLIFS pocket
similarity have comparable or higher retrieval point estimates for the fixed co-selective-
pair endpoint than Vina, and a
strict within-KLIFS-group analysis does not resolve an additional docking component. DAVIS
is treated as measurement-limited because 67.0% of its fixed block lies at the reporting
floor and its continuous target map largely tracks binary above-floor status.

For ligand-wise ranking, subtracting a ligand row mean is algebraically invariant to target
order. Unequal target scaling converts that row term into an exact rank-one correction and
must be validated as a new score rule. The broad DOCKSTRING–ChEMBL benchmark contains 6,480
informative ligands, 31 targets and 25,214 non-tied observed within-ligand comparisons: absolute, column-standardized
and scaled-residual Vina concordances are 0.537, 0.530 and 0.529. Residual-minus-absolute is
−0.008, with chemical-cluster 95% interval [−0.022, 0.006] and target-jackknife approximation
[−0.056, 0.030]. Absolute Vina exceeds the ligand-invariant target-only prior fitted on the
external DOCKSTRING reference by
only 0.013, with paired chemical-cluster interval [−0.011, 0.038]; the benchmark therefore
does not resolve ligand-specific ordering information beyond that target-only baseline.

A less assay-heterogeneous, endpoint-restricted human binding Ki/Kd arm contains 1,049
informative ligands and 3,955 comparisons; it can still compare a Ki-derived cell with a Kd-derived
cell (91.3% of pairs used the same single endpoint, 0.8% were unambiguous Ki–Kd comparisons,
and 7.9% involved at least one cell pooling both endpoint types):
absolute and residual scores are 0.559 and 0.520, with paired difference −0.040,
chemical-cluster interval [−0.091, 0.003] and target-jackknife approximation
[−0.133, 0.029]. Separate endpoints are heterogeneous: Ki is negative with a chemical
interval excluding zero, while the much smaller Kd arm reverses sign and remains unresolved.
No universal normalized representation is supported.

The reusable output is a decision-oriented audit: define the score, endpoint, ligand support
and target panel; report correlation and covariance spectra; test molecular-size-domain transport and other support shifts;
pair fitted-component maps with held-out predictive-performance and nuisance controls; compare the full
predictor-by-endpoint factorial; include protein-similarity baselines; and validate any
operational score with chemical- and target-aware uncertainty.

## Build the submission package

```bash
uv venv --python 3.12.11 .venv
uv pip install --python .venv/bin/python -r requirements.lock
make PYTHON=.venv/bin/python all
```

`make all` verifies frozen inputs and manuscript-facing result bundles by SHA-256, runs the
invariant tests, rebuilds the evidence ledger, regenerates registered values/tables/figures,
and compiles the article and Supplement twice.

The report-layer build does not silently download source-restricted assay files or rerun
receptor preparation, pose generation or docking. Retrieval commands and dataset-specific
analysis contracts are documented in `PROVENANCE.md` and `REPRODUCIBILITY.md`.
The secondary fixed-pose ODDT rescoring outputs are also checksum-verified rather than
recreated by the default container: their source run used ODDT 0.7 in a separate Python 3.9
environment, plus external pose archives and a smina executable.

Container build:

```bash
docker build -t vina-target-map-audit .
docker run --rm vina-target-map-audit
```

## Apply the spectral audit to another matrix

```bash
.venv/bin/python analysis/spectral_audit.py scores.csv \
  --score-columns target_A,target_B,target_C \
  --id-column ligand_id \
  --recovery-sizes 50,100,200,500 \
  --recovery-repeats 100 \
  --output audit.json \
  --plot-prefix audit_diagnostics
```

The generic command reports uncentred and within-ligand-centred spectra, correlation
distributions, heatmaps, target-composition sensitivity, matched nulls and a same-support
map-recovery curve. The curve chooses a matrix-specific probe size; the observed 200-ligand
result in this article is not a universal threshold. Experimental geometry, chemical-domain
transport and task-level ranking remain dataset-specific.

## Package map

- `manuscript.tex`, `manuscript.pdf`: article source and compiled manuscript.
- `supplementary_information.tex`, `supplementary_information.pdf`: Supplement.
- `analysis/build_manuscript_evidence.py`: machine-readable report ledger.
- `analysis/make_reported_results.py`: registered manuscript values.
- `analysis/make_manuscript_figures.py`: five main figures.
- `analysis/make_supplement_tables.py`: evidence-driven supplementary tables.
- `analysis/descriptor_correlation_reduction_uncertainty.py`: repeated chemical-group
  partitions and score-blind supports.
- `analysis/descriptor_rank_matched_controls.py`: physicochemical, count-Morgan, stable-hash
  and row-permuted nuisance controls.
- `analysis/dockstring_chembl_ranking_benchmark.py`: broad and endpoint-restricted ranking
  benchmarks.
- `analysis/spectral_audit.py`: reusable spectral-audit CLI.
- `results/manuscript_evidence.json`: evidence consumed by the paper.
- `data_manifest.csv`: frozen matrix metadata, rights, sizes and checksums.
- `results/manuscript_source_manifest.csv`: checksums for manuscript-facing artifacts.
- `PROVENANCE.md`, `DATA_LICENSES.md`, `REPRODUCIBILITY.md`: source, rights and build boundaries.

## Licensing and citation

Analysis code is MIT licensed. File-specific data rights are recorded in
`data_manifest.csv` and `DATA_LICENSES.md`. ChEMBL-derived data retain their source terms and
DOCKSTRING is distributed under Apache-2.0. The package does not redistribute the KiRHub
workbook or compound-level activity profiles; it releases only the aggregate target-pair
statistics used in the article under the documented boundary.

Citation metadata are in `CITATION.cff`. The immutable `v3.0.1` submission archive has DOI
[`10.5281/zenodo.21874032`](https://doi.org/10.5281/zenodo.21874032); all versions are indexed
by the concept DOI [`10.5281/zenodo.21733767`](https://doi.org/10.5281/zenodo.21733767).
