# A shared ligand-wide score axis obscures library-dependent target geometry in docking matrices

Reproducibility package for a *Journal of Cheminformatics* Methodology article.

Multi-target docking matrices are often read as maps of relationships between
targets. That interpretation can be misleading when some ligands score
favourably against nearly every receptor. This project asks what remains after
subtracting each ligand's mean score across targets, whether that residual
structure exceeds matched-null expectations, and where it agrees with external
pharmacology.

## Main result

The leading component is close to the uniform target direction in all eight
scoring outputs tested on the same fixed poses. After ligand-wise centring, the
two large docking matrices retain structured, lower-dimensional residual maps
that are more concentrated than four transformation-matched nulls.

The maps are not universal protein networks: low- and high-molecular-weight
subsets of the same libraries give weakly agreeing geometries. On the primary
PDSP graph, the residual map is descriptively closer to the exact-value
experimental map than the raw map. Docking-44 ligands that share a connectivity
block with PDSP are excluded before this comparison.

All external comparisons are post hoc and support-sensitive. PDSP residual agreement
weakens when censored PDSP cells are restored and largely disappears when Docking-44 is
restricted to its chemically shifted complete-case support. Paired target-label
QAP inference is two-sided, with Holm and Bonferroni correction across the
12-row primary QAP table. The sequence-and-family fusion is retained as a sensitivity,
not as a headline or statistically resolved result. Kinase, SPD and KiRHub
results are also mixed. We therefore claim a diagnostic worth testing on a
declared ligand support, not general superiority over homology or a
compound-level target-ranking rule.

About 200 randomly selected ligands recover each source-library map above
Spearman 0.91. This is a practical sampling result for the studied libraries,
not a universal sample-size guarantee.

The separate compound-ranking audit points in the opposite direction: centring
does not improve target ranking for an individual ligand. Endpoint mixing adds
another limitation, so this analysis is used only to define the boundary of the
target-map claim.

The frozen compound-ranking audit contains 6,480 informative ligands, 31
targets and 25,214 non-tied observed within-ligand comparisons. Of these,
91.3% of pairs used the same endpoint, 0.8% were unambiguous Ki–Kd
comparisons, and 7.9% involved at least one cell pooling both endpoint types.

## Rebuild the report

With Python 3.12 and a LaTeX installation:

```bash
uv venv --python 3.12.11 .venv
uv pip install --python .venv/bin/python -r requirements.lock
make PYTHON=.venv/bin/python all
```

The default target works from frozen result bundles. It regenerates the v5
figures and upload assets, checks manuscript and Supplement claims against those
bundles, and compiles both PDFs twice. It does not rerun docking or every
source-level analysis, create a Git tag, or mint a DOI.

Useful targets:

```bash
make PYTHON=.venv/bin/python figures-v5
make PYTHON=.venv/bin/python verify-v5-numbers
make PYTHON=.venv/bin/python manuscript-v5
make PYTHON=.venv/bin/python supplement-v5
make PYTHON=.venv/bin/python submission-figures
make PYTHON=.venv/bin/python release-check-v5
```

Source-level refresh commands and their external inputs are described in
[REPRODUCIBILITY.md](REPRODUCIBILITY.md). The older strict-public v4 builder is
retained as a historical, separately named workflow.

## Apply the audit to another matrix

```bash
.venv/bin/python analysis/target_map_audit.py scores.csv.gz \
  --output-dir results/my_target_map_audit \
  --target-regex '^target_' \
  --ligand-id compound_id \
  --domain-column molecular_weight \
  --domain-quantiles 4 \
  --pilot-sizes 100,200,500 \
  --pilot-repeats 200 \
  --panel-k 8 \
  --seed 20260810
```

The command writes preprocessing metadata, raw and residual spectra, signed
target-correlation edges and checksums. Domain, calibration-recovery and panel
outputs are optional. See
[analysis/TARGET_MAP_AUDIT.md](analysis/TARGET_MAP_AUDIT.md).

## Data boundary

Redistributable inputs and derived result tables are shipped with the release
candidate. PDSP, the Novartis Secondary Pharmacology Database, PKIS1, KiRHub,
KLIFS, UniProt, RCSB/PDBe metadata and the fixed-pose scorer resources must be
obtained from their named sources when a source-level analysis requires them.
Access routes, versions, integrity records, licences and fetchers are recorded
in `external_sources.csv`, `JOC_ACCESS_MATRIX.csv`, `data_manifest.csv` and
`DATA_LICENSES.md`. The access
matrix uses only `VERIFIED` and `UNKNOWN`; it exposes unresolved rights or replay
evidence instead of treating public downloadability as a licence.

The report build does not repeat receptor preparation, pose generation or
docking. Those upstream stages belong to the cited Docking-44 and DOCKSTRING
studies. Derived target-pair summaries are included so the reported external-map
comparisons can be audited without redistributing source activities. In
particular, `analysis/reproduce_pdsp_derived_qap.py` rebuilds the primary and
complete-case PDSP paired-QAP tables and the support-threshold sensitivity from
the released aggregate table and Docking-44 matrix.

## Package map

- `manuscript_v5.tex` / `manuscript_v5.pdf`: article source and built PDF.
- `supplementary_v5.tex` / `supplementary_v5.pdf`: Supplement source and PDF.
- `sections/` and `si_sections/`: modular manuscript sources.
- `figures/v5/`: generated main, Supplementary and graphical-abstract figures.
- `results/`: frozen analysis bundles used by the report.
- `analysis/test_v5_manuscript_numbers.py` and
  `analysis/test_v5_supplement_numbers.py`: report-facing consistency checks.
- `results/v5_report_source_manifest.csv`: exact frozen report inputs and code.
- `analysis/reproduce_pdsp_derived_qap.py`: aggregate-only reconstruction of the
  PDSP paired-QAP tables and support sensitivities.
- `analysis/target_map_audit.py`: reusable audit command-line tool.
- `CLAIM.md`, `PREFLIGHT.md` and `REPRODUCIBILITY.md`: claim, submission and
  reconstruction contracts.
- `PROVENANCE.md`, `DATA_LICENSES.md`, `data_manifest.csv` and
  `external_sources.csv`: provenance and licensing records.
- `JOC_ACCESS_MATRIX.csv`: source-by-source access, licence, integrity and replay
  evidence for editorial review.

## Licensing and release status

Analysis code is MIT licensed. Third-party data retain their upstream terms;
consult the manifests before redistribution. Author-generated documentation and
derived summaries are intended for CC BY 4.0 release unless an upstream term
applies.

`CITATION.cff` describes an unreleased `5.0.0-dev` package. A final archive DOI
will be inserted only after the exact release candidate has passed a clean
archive build and been deposited. No earlier DOI is claimed for v5.
