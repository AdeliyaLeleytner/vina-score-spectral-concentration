# Reproducibility checks

## Full build

```bash
make PYTHON=.venv/bin/python all
```

The dependency order is:

1. `verify`: check all 26 frozen inputs against `data_manifest.csv`.
2. `evidence`: rebuild `results/evidence_summary.json` and CSV summaries.
3. `validate-evidence`: assert PR/correlation identities, the structural residual zero,
   row-norm-null separation and operational rank invariance.
4. `reported` and `verify-reported`: generate registered TeX values/table and fail on drift.
5. `tables`: regenerate `results/supplement_tables.tex`.
6. `figures`: regenerate five main and seven supplementary PDF/PNG figures.
7. `manuscript` and `supplement`: compile both LaTeX documents twice.

## Expected headline values

The evidence build ends with:

```text
Headline PR: Docking-44 1.834 -> 9.303; DOCKSTRING-58 2.266 -> 18.206
```

The 500-permutation common-protocol analysis should report 3 versus 7 modes for Docking-44
and 4 versus 10 for DOCKSTRING-58 under both the descriptive rank-wise threshold and the
family-wise-error-controlled simultaneous envelope. Five independent 100-permutation series
should repeat those counts. Both the fitted additive null and the empirical residual-
permutation null should have median residual PR dimensions 42.40 and 56.48, with one-sided
empirical probabilities 0.002. Across 200 one-ligand-per-Butina-cluster Docking-44 supports,
the observed residual PR median should be 9.43 and the matched empirical-null median 42.33.
The row-norm-preserving null should have medians 42.63 and 56.62, again with one-sided
probabilities 0.002. Residual mean absolute target correlations should be 0.246 and 0.155 on
the full matrices.

The dense matched ChEMBL analysis should report raw and residual experimental-minus-docking
PR differences 2.006 and 0.483, with paired ligand-bootstrap intervals 1.174--2.673 and
-0.071--0.921. The expanded operational support should contain 137 ligands, 38 targets, 691
observed cells, 95 Murcko clusters, 121 Butina clusters and 2,522 non-tied pairs.
Target-preference accuracies
should be 0.566 for absolute Vina, 0.546 for column-standardized Vina, 0.564 for two-way
residual Vina, 0.525 for the external docking target prior and 0.592 for the
scaffold-cluster-held-out cohort experimental prior. The paired residual-minus-column
difference should be 0.017 with a union of 95% cluster intervals of -0.022--0.055;
residual-minus-absolute should be -0.002 with a union of -0.065--0.063. Conservative 90%
cluster-interval unions should be -0.015--0.049 and -0.056--0.051. Post hoc equivalence
should fail at +/-0.02 for both contrasts and at +/-0.05 for residual minus absolute.

Unscaled target-centered and two-way-centered accuracy should be exactly identical at 0.556.
Target-centered scores divided by residual target scales should give 0.549. The fixed-support
same-endpoint analysis should retain 132 evaluable ligands and 4,330 endpoint-specific pair
instances, with absolute, column-standardized and residual accuracies 0.553, 0.538 and 0.555.
The human binding Ki/Kd block should contain 107 ligands, 30 targets and 2,224 pairs, with
accuracies 0.565, 0.554 and 0.579. Requiring at least three observed targets
should retain 93 ligands and give 0.587 for absolute Vina, 0.530 for residual Vina and 0.636
for the cohort prior.
The duplicate-collapsed same-endpoint analysis should retain 132 ligands and 2,460 unique
ligand--target-pair instances, with accuracies 0.567, 0.541 and 0.562. Broad-panel
leave-one-target-out residual-minus-absolute differences should range from -0.030 to 0.010.
Across one-ligand-per-Murcko-cluster supports, median identity-permutation probabilities
should be 0.134 for absolute and 0.284 for residual Vina.

Figure metadata and LaTeX builds use a fixed release timestamp (`SOURCE_DATE_EPOCH`), so
consecutive rebuilds in the same pinned environment produce byte-identical figure and PDF
artifacts.

A clean clone rebuilt with the reference macOS environment produces byte-identical JSON,
CSV, figure and PDF artifacts. The Docker image uses Debian's numerical and TeX libraries:
registered TeX values/tables and all operational CSVs are byte-identical. Two unrounded
spectral-summary CSVs can differ at the last floating-point bits (maximum observed numeric
difference `3.6e-15`), as can the unrounded JSON (maximum absolute difference in the v1.4.0
preflight: `2.2e-14`). Container PDFs have the same content, page count and page size but are not
expected to be byte-identical to PDFs produced by a different TeX distribution.

## Focused checks

```bash
make PYTHON=.venv/bin/python verify
.venv/bin/python -m py_compile analysis/*.py
.venv/bin/python analysis/validate_evidence.py
.venv/bin/python analysis/spectral_audit.py data/frozen/df_final_v4.csv.gz \
  --score-columns 1m2z,1pbq,1xoq --id-column ligand_id \
  --smiles-column 'Canonical SMILES' --permutations 5 --bootstrap 5 \
  --sample-size 1000 --output /tmp/spectral_audit_smoke.json \
  --plot-prefix /tmp/spectral_audit_smoke
.venv/bin/python analysis/verify_reported_results.py
pdftotext manuscript.pdf - | grep -E '\[(PUBLIC REPOSITORY URL|ZENODO DOI)\]'
```

The last command must return no line in a submission or archived release. Versioned archives
are indexed under the stable Zenodo concept DOI `10.5281/zenodo.21733767`.

## Clean-clone guarantee

Analysis scripts resolve `data/frozen/` first. A fallback to the historical parent project
tree exists only for local development and is not used by a clean clone. The Docker build
copies this repository alone and is therefore an independent check of package completeness.
