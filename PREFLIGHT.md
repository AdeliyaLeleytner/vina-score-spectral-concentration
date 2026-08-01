# Journal of Cheminformatics submission preflight

Audit date: 2026-08-01. Article type: Research article.

## Passed locally

- Title states that the target-ranking gain is unresolved; conclusions remain limited to the
  two tested large Vina matrices and do not confuse absence of significance with equivalence.
- Target uncertainty includes leave-one-out deletion plus uniform and family-stratified
  subsampling for both column-standardized and residual surfaces.
- Parallel analysis uses 500 permutations, a family-wise-error-controlled simultaneous
  envelope, and five independent 100-permutation series.
- Fitted additive and empirical residual-permutation nulls show that the centered residuals
  remain much more concentrated than independent target-specific noise. A one-ligand-per-
  Butina-cluster sensitivity addresses ordinary chemical redundancy.
- Exact-relation median ChEMBL aggregation is primary; raw and residual paired contrasts,
  100 multiple imputations, human binding, and Ki/Kd blocks are reported and limited to their
  matched support.
- The operational target-preference benchmark now covers 137 ligands, 38 targets, 691
  observed cells and 2,522 non-tied target pairs without activity imputation. Evaluation
  ligands are excluded from docking imputation/transformation references. Predicted ties
  receive half credit. Controls include external and scaffold-held-out cohort target
  priors, ligand-identity and within-ligand outcome permutations, independent Murcko and
  Butina cluster uncertainty, target deletion, experimental margins, assay restrictions,
  within-cell dispersion and minimum observed-target coverage. A fixed-support same-endpoint
  analysis and a prominent human binding Ki/Kd sensitivity reduce endpoint mixing.
- The ranking decomposition proves that unscaled row centering cannot change within-ligand
  target order; target-specific scaling is the operationally active component. Post hoc
  equivalence checks at +/-0.02 and +/-0.05 are reported as sensitivities, not preregistered
  endpoints.
- A generic `analysis/spectral_audit.py` CLI and a nine-step protocol in Table S9 make the
  recommended workflow reusable; the CLI passed a real three-target smoke test.
- The DTI rank–selectivity analysis is entirely Supplementary; all 20 arms are primary and
  the outcome-restricted 12-arm panel is a labelled sensitivity.
- All 26 frozen inputs are bundled and pass byte-size and SHA-256 verification.
- `make all` succeeds while forcing bundled inputs; headline PR values are
  `1.834 -> 9.303` and `2.266 -> 18.206`.
- A fresh local clone reproduces the evidence JSON, machine-readable tables, figures and PDFs
  byte for byte. A fresh Docker build/run completes `make all`; all table CSVs are identical
  and the largest unrounded cross-platform JSON difference is `1.6e-14`.
- Abstract: approximately 257 plain words; Scientific Contribution: two sentences.
- Bibliography: 30 cited entries, with exactly 30 defined keys and no missing keys.
- Article: 28 A4 pages. Supplement: 14 A4 pages, with no blank or near-empty continuation page.
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
  are present as numbered references and in citation metadata. The release target is v1.3.0;
  its version DOI is added to the GitHub release metadata after Zenodo mints it.

## Blocking before journal submission

1. Obtain final coauthor confirmation of funding, contribution roles, author order,
   affiliations, ORCIDs, and the AI-use disclosure.

The GitHub repository is public at
<https://github.com/AdeliyaLeleytner/vina-score-spectral-concentration>; versioned archives
are indexed at <https://doi.org/10.5281/zenodo.21733767>.
