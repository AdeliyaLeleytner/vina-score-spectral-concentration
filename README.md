# Vina score spectral concentration

Reproducibility package for the manuscript:

> Vina docking-score matrices are strongly spectrally concentrated before, but not after,
> two-way centering

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
The 500-permutation analysis is the longest step. All stochastic procedures use seed 0.

Container reproduction is self-contained:

```bash
docker build -t vina-score-spectral-concentration .
docker run --rm vina-score-spectral-concentration
```

## Package map

- `manuscript.tex` and `manuscript.pdf`: article source and compiled manuscript.
- `supplementary_information.tex` and `.pdf`: supplementary source and compiled PDF.
- `analysis/build_evidence.py`: statistical analysis and machine-readable evidence ledger.
- `analysis/make_figures.py`: five main and three supplementary figures.
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
