# Journal of Cheminformatics submission preflight

Audit date: 2026-08-01. Article type: Research article.

## Passed locally

- Title and conclusions are limited to the two tested large Vina matrices.
- Target uncertainty includes leave-one-out deletion plus uniform and family-stratified
  subsampling for both column-standardized and residual surfaces.
- Parallel analysis uses 500 permutations and five independent 100-permutation series.
- Exact-relation median ChEMBL aggregation is primary; human binding and Ki/Kd blocks are
  prominent and explicitly limited to their matched support.
- The DTI rank–selectivity analysis is entirely Supplementary; all 20 arms are primary and
  the outcome-restricted 12-arm panel is a labelled sensitivity.
- All 26 frozen inputs are bundled and pass byte-size and SHA-256 verification.
- `make all` succeeds while forcing bundled inputs; headline PR values are
  `1.834 -> 9.303` and `2.266 -> 18.206`.
- Abstract: 305 plain-token words; Scientific Contribution: two sentences.
- Bibliography: 24 cited entries, with no missing or unused keys after final compilation.
- Article: 22 A4 pages. Supplement: 9 A4 pages, with no blank table-continuation page.
- Five main and three supplementary figures are regenerated as vector PDFs plus PNG previews;
  Fig. S3 point IDs map to Table S5.
- Two-pass LaTeX builds contain no unresolved references, citation warnings, package warnings,
  or overfull boxes. Greek rho survives PDF text extraction.
- Consecutive pinned-environment rebuilds produce byte-identical figure and compiled PDF
  artifacts through fixed release metadata and `SOURCE_DATE_EPOCH`.
- MIT code license, file-specific data licenses, provenance, pinned environment, Makefile,
  Dockerfile, and clean-clone instructions are present.
- AI systems are not authors or evidence sources; the Methods disclosure assigns scientific
  responsibility to the manuscript authors.
- The public GitHub repository, tagged release, and permanent Zenodo concept DOI
  `10.5281/zenodo.21733767` are present in the manuscript and citation metadata.

## Blocking before journal submission

1. Obtain final coauthor confirmation of funding, contribution roles, author order,
   affiliations, ORCIDs, and the AI-use disclosure.

The GitHub repository is public at
<https://github.com/AdeliyaLeleytner/vina-score-spectral-concentration>; versioned archives
are indexed at <https://doi.org/10.5281/zenodo.21733767>.
