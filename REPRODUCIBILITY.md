# Reproducibility checks

## Full build

```bash
make PYTHON=.venv/bin/python all
```

The dependency order is:

1. `verify`: check all 26 frozen inputs against `data_manifest.csv`.
2. `evidence`: rebuild `results/evidence_summary.json` and CSV summaries.
3. `tables`: regenerate `results/supplement_tables.tex`.
4. `figures`: regenerate five main and six supplementary PDF/PNG figures.
5. `manuscript` and `supplement`: compile both LaTeX documents twice.

## Expected headline values

The evidence build ends with:

```text
Headline PR: Docking-44 1.834 -> 9.303; DOCKSTRING-58 2.266 -> 18.206
```

The 500-permutation common-protocol analysis should report 3 versus 7 modes for Docking-44
and 4 versus 10 for DOCKSTRING-58 under both the descriptive rank-wise threshold and the
family-wise-error-controlled simultaneous envelope. Five independent 100-permutation series
should repeat those counts. The fitted additive null should have median residual PR
dimensions 42.40 and 56.48, with one-sided empirical probabilities 0.002.

The dense matched ChEMBL analysis should report raw and residual experimental-minus-docking
PR differences 2.006 and 0.483, with paired ligand-bootstrap intervals 1.174--2.673 and
-0.071--0.921. The expanded operational support should contain 137 ligands, 38 targets, 691
observed cells, 95 scaffold clusters and 2,522 non-tied pairs. Target-preference accuracies
should be 0.566 for absolute Vina, 0.546 for column-standardized Vina, 0.564 for two-way
residual Vina, 0.525 for the external docking target prior and 0.592 for the
scaffold-cluster-held-out cohort experimental prior. The paired residual-minus-column
difference should be 0.017 with a scaffold interval of -0.022--0.054; residual-minus-absolute
should be -0.002 with an interval of -0.065--0.062. Requiring at least three observed targets
should retain 93 ligands and give 0.587 for absolute Vina, 0.530 for residual Vina and 0.636
for the cohort prior.

Figure metadata and LaTeX builds use a fixed release timestamp (`SOURCE_DATE_EPOCH`), so
consecutive rebuilds in the same pinned environment produce byte-identical figure and PDF
artifacts.

## Focused checks

```bash
make PYTHON=.venv/bin/python verify
.venv/bin/python -m py_compile analysis/*.py
pdftotext manuscript.pdf - | grep -E '\[(PUBLIC REPOSITORY URL|ZENODO DOI)\]'
```

The last command must return no line in a submission or archived release. Versioned archives
are indexed under the stable Zenodo concept DOI `10.5281/zenodo.21733767`.

## Clean-clone guarantee

Analysis scripts resolve `data/frozen/` first. A fallback to the historical parent project
tree exists only for local development and is not used by a clean clone. The Docker build
copies this repository alone and is therefore an independent check of package completeness.
