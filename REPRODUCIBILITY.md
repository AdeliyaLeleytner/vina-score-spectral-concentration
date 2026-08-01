# Reproducibility checks

## Full build

```bash
make PYTHON=.venv/bin/python all
```

The dependency order is:

1. `verify`: check all 26 frozen inputs against `data_manifest.csv`.
2. `evidence`: rebuild `results/evidence_summary.json` and CSV summaries.
3. `tables`: regenerate `results/supplement_tables.tex`.
4. `figures`: regenerate five main and three supplementary PDF/PNG figures.
5. `manuscript` and `supplement`: compile both LaTeX documents twice.

## Expected headline values

The evidence build ends with:

```text
Headline PR: Docking-44 1.834 -> 9.303; DOCKSTRING-58 2.266 -> 18.206
```

The 500-permutation common-protocol analysis should report 3 versus 7 supra-null modes for
Docking-44 and 4 versus 10 for DOCKSTRING-58. Five independent 100-permutation series should
repeat those counts.

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
