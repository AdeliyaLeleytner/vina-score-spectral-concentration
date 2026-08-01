# Journal of Cheminformatics submission preflight

Audit date: 2026-08-01. Article type: Research article.

## Passed locally

- Title states that centering reveals residual structure but is not itself a scoring correction;
  conclusions remain limited to the two tested large Vina matrices.
- Target uncertainty includes leave-one-out deletion plus uniform and family-stratified
  subsampling for both column-standardized and residual surfaces.
- Parallel analysis uses 500 permutations, a family-wise-error-controlled simultaneous
  envelope, and five independent 100-permutation series.
- A fitted additive null shows that the centered residuals remain much more concentrated
  than independent target-specific noise.
- Exact-relation median ChEMBL aggregation is primary; raw and residual paired contrasts,
  100 multiple imputations, human binding, and Ki/Kd blocks are reported and limited to their
  matched support.
- The operational target-preference benchmark includes leave-one-ligand-out estimation,
  target-offset-only and experimental-prior baselines, ligand-identity shuffles, scaffold
  bootstrap uncertainty, tie sensitivity, and a minimum detectable difference.
- The DTI rank–selectivity analysis is entirely Supplementary; all 20 arms are primary and
  the outcome-restricted 12-arm panel is a labelled sensitivity.
- All 26 frozen inputs are bundled and pass byte-size and SHA-256 verification.
- `make all` succeeds while forcing bundled inputs; headline PR values are
  `1.834 -> 9.303` and `2.266 -> 18.206`.
- Abstract: 272 plain-token words; Scientific Contribution: two sentences.
- Bibliography: 26 cited entries, with no missing keys after final compilation.
- Article: 25 A4 pages. Supplement: 11 A4 pages, with no blank or near-empty continuation page.
- Five main and six supplementary figures are regenerated as vector PDFs plus PNG previews;
  Fig. S5 point IDs map to Table S6.
- Two-pass LaTeX builds contain no unresolved references, citation warnings, package warnings,
  or overfull boxes. Greek rho survives PDF text extraction.
- Consecutive pinned-environment rebuilds produce byte-identical figure and compiled PDF
  artifacts through fixed release metadata and `SOURCE_DATE_EPOCH`.
- MIT code license, file-specific data licenses, provenance, pinned environment, Makefile,
  Dockerfile, and clean-clone instructions are present.
- AI systems are not authors or evidence sources; the Methods disclosure assigns scientific
  responsibility to the manuscript authors.
- The public GitHub repository and permanent Zenodo concept DOI `10.5281/zenodo.21733767`
  are present as numbered references and in citation metadata. The release target is v1.1.0.

## Blocking before journal submission

1. Obtain final coauthor confirmation of funding, contribution roles, author order,
   affiliations, ORCIDs, and the AI-use disclosure.

The GitHub repository is public at
<https://github.com/AdeliyaLeleytner/vina-score-spectral-concentration>; versioned archives
are indexed at <https://doi.org/10.5281/zenodo.21733767>.
