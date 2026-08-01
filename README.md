# Vina score spectral concentration

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21733767.svg)](https://doi.org/10.5281/zenodo.21733767)

Reproducibility package for the manuscript:

> Spectral concentration in Vina docking-score matrices: what two-way centering reveals and
> does not correct

The analysis distinguishes the column-standardized ligand-by-target score surface from its
two-way-centered residual surface. Participation-ratio (PR) effective dimension is used as a
continuous measure of spectral concentration, not as algebraic rank or a direct measure of
affinity accuracy or selectivity.

## Headline result

| Matrix | Ligands × targets | Column-standardized PR | Residual PR |
| --- | ---: | ---: | ---: |
| Docking-44 (Vina-GPU 2.0) | 12,651 × 44 | 1.834 | 9.303 |
| DOCKSTRING (AutoDock Vina) | 260,060 × 58 | 2.266 | 18.206 |

These claims are limited to the tested Vina systems. Two small RF-Score v1 blocks are
reported as sensitivities, not large independent replications.

The practical benchmark uses 74 matched ligands and six ChEMBL targets. Absolute Vina scores
reached pairwise target-preference accuracy 0.619, but a target-offset-only baseline reached
0.625 and ligand-identity shuffling reproduced the absolute-score result. Column-standardized
and two-way-residual scores reached 0.537 and 0.540; their paired scaffold-bootstrap
difference was 0.004 (95% interval -0.019--0.026). The release therefore treats two-way
centering as a diagnostic transformation, not an automatically beneficial ranking rule.

## Reproduce

The repository contains all frozen matrices and summaries needed for the reported analysis.
From a clean clone:

```bash
uv venv --python 3.12.11 .venv
uv pip install --python .venv/bin/python -r requirements.lock
make PYTHON=.venv/bin/python all
```

`make all` verifies 26 inputs against byte sizes and SHA-256 digests, rebuilds the evidence
ledger, regenerates every figure and supplementary table, and compiles both PDFs twice.
The 500-permutation and 5,000-replicate target-preference analyses are the longest steps.
Seed 0 initializes all stochastic procedures; derived support and replicate seeds are stored
in the evidence ledger.

Container reproduction is self-contained:

```bash
docker build -t vina-score-spectral-concentration .
docker run --rm vina-score-spectral-concentration
```

## Package map

- `manuscript.tex` and `manuscript.pdf`: article source and compiled manuscript.
- `supplementary_information.tex` and `.pdf`: supplementary source and compiled PDF.
- `analysis/build_evidence.py`: statistical analysis and machine-readable evidence ledger.
- `analysis/make_figures.py`: five main and six supplementary figures.
- `analysis/make_supplement_tables.py`: supplementary tables generated from the ledger.
- `results/evidence_summary.json`: authoritative numeric result ledger.
- `data/frozen/`: compressed frozen analysis inputs.
- `data_manifest.csv`: source, version, license, byte size, and SHA-256 for every input.
- `PROVENANCE.md`: boundary between reproducible analysis and upstream docking/rescoring.
- `REPRODUCIBILITY.md`: expected outputs and validation checks.
- `REVISION_MATRIX.md`: disposition of the pre-submission major-revision feedback.

## Scope of reproducibility

The release exactly reproduces the manuscript from frozen score/activity matrices. It does
not rerun receptor preparation, pose generation, Vina docking, or RF-Score rescoring; those
upstream stages are documented in `PROVENANCE.md` and the cited source releases. This is an
intentional boundary: the frozen matrices themselves are versioned and checksum-verified.

## Licenses and citation

Analysis code is MIT licensed. Data products have file-specific licenses in
`data_manifest.csv` and `DATA_LICENSES.md`; in particular, ChEMBL-derived files remain under
CC BY-SA 3.0 and DOCKSTRING remains under Apache-2.0. Cite the manuscript and the upstream
datasets when reusing the package. Citation metadata are provided in `CITATION.cff`.
Versioned archives are indexed under the Zenodo concept DOI
[`10.5281/zenodo.21733767`](https://doi.org/10.5281/zenodo.21733767).
