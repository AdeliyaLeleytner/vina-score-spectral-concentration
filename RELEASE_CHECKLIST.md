# Pre-submission release checklist

The submission artifact is version `3.0.0` and is titled *Target-correlation maps are
library-conditional but recoverable from a few hundred ligands in two large Vina panels*.
It includes the broad 6,480-ligand/31-target DOCKSTRING--ChEMBL benchmark, its endpoint-
restricted human binding Ki/Kd arm,
rank-matched physicochemical and count-Morgan feature controls, and stable-hash and
row-permuted nuisance controls. The old 137-ligand/38-target Docking-44 benchmark is a legacy
sparse sensitivity only.

The release is not ready until the public tag, rendered repository, versioned archive and
clean-room build all correspond to this same artifact. Publishing is an outward-facing
author action; this checklist records the required synchronization.

## Do this before the cover letter claims a reproducibility package

1. **Push.** `git push origin main`. Confirm afterwards that `git log origin/main -1` matches
   the local HEAD and that the rendered README on GitHub shows the current title.
2. **Tag an immutable release** matching the submitted manuscript, e.g. `v3.0.0`, and confirm
   `CITATION.cff` (`version`, `date-released`) agrees with the tag.
3. **Mint a version-specific Zenodo DOI** for that tag. The manuscript's Availability section
   now tells readers to cite the version-specific archive rather than the concept record
   `10.5281/zenodo.21733767`. The reserved version DOI is
   `10.5281/zenodo.21865607`; publish the completed archive so it resolves publicly.
4. **Clean-room reproduce.** From a fresh clone of the tag, in the container:
   ```bash
   git clone --branch v3.0.0 <url> && cd vina-score-spectral-concentration
   docker build -t jcheminf-repro . && docker run --rm jcheminf-repro
   ```
   The build must reach `manuscript.pdf` and `supplementary_information.pdf` without
   consulting anything outside the clone.
5. **Confirm the numbers came from the clone**, not from a local cache: every figure, every
   supplementary table, and every value in the abstract must regenerate from
   `results/manuscript_evidence.json` as rebuilt inside the container.
6. **Re-read the README** against the final manuscript: title, headline results, package map,
   limitations, licences.
7. **Check the high-risk contracts** in the clean clone:
   - broad ranking: 6,480 evaluated ligands, 31 targets, 25,214 non-tied observed within-
     ligand comparisons; absolute/column/
     residual concordance 0.537/0.530/0.529;
   - endpoint-restricted human binding Ki/Kd: 1,049 evaluated ligands, 3,955 comparisons;
     absolute/residual
     0.559/0.520;
   - stable-hash and row-permuted controls: essentially zero out-of-fold predictive
     performance despite nonzero scale-free fitted-component map agreement.

## Known reproducibility boundaries

- **Source-restricted inputs.** PKIS1 and the KiRHub workbook are not redistributed; they are
  fetched by checksum-verified scripts from their publishers, and the report-layer build does
  not download them. The 190-edge target-pair ledger is released so that every fixed-panel
  number is recomputable, but a reader cannot rebuild the experimental geometry from
  compound-by-target measurements without obtaining those two sources themselves. The
  Before submission, the authors should verify that this boundary satisfies the journal's
  data policy and, if needed, ask the editor about the KiRHub and PKIS1 licence terms.
- **The ODDT rescoring stage** runs under a separate Python 3.9 interpreter (conda) rather
  than the pinned `.venv`. Both interpreters are recorded in
  `results/nonvina_scorer_transport/summary.json`, but a clean-room reproducer needs both.
- **Pose archives.** The non-Vina analysis consumes DOCKSTRING pose archives that are not
  redistributed with the package; their MD5s are verified in-run and recorded.

## Independent review status

Run fresh independent reviews only after the final PDFs and evidence ledger are rebuilt.
Record the reviewed commit and tag in the review notes; a review of an earlier title,
benchmark contract or descriptor interpretation does not validate version `3.0.0`.
