# v5 submission-release checklist

This checklist covers the exact package that will accompany the *Journal of
Cheminformatics* submission. It must be applied to a clean archive, not the
current dirty working tree.

## Scientific lock

- [ ] Title matches `manuscript_v5.tex`, `supplementary_v5.tex`, `README.md`,
  `PREFLIGHT.md` and `CITATION.cff`.
- [ ] Abstract is no more than 350 words and contains one concise `Scientific Contribution`
  paragraph.
- [ ] The primary full-support PDSP residual-versus-raw contrast is described as
  descriptive and post hoc.
- [ ] Every external analysis is identified as post hoc.
- [ ] PDSP-overlapping Docking-44 connectivity blocks are excluded before the
  target map is built.
- [ ] Censor-aware weakening and loss of residual agreement on the chemically shifted complete-case
  Docking-44 support remain explicit.
- [ ] Paired QAP inference is two-sided, with Holm and Bonferroni correction
  within the 12-row primary QAP table.
- [ ] Sequence-and-family fusion is not presented as significant or as a
  headline result.
- [ ] Kinase, SPD, KiRHub and ligand-wise ranking boundaries remain.
- [ ] Pocket volume is Supplementary and hypothesis-generating.
- [ ] No claim implies universal improvement beyond homology, causal mechanism,
  affinity prediction or compound-level target-ranking improvement.

## Files and build

- [ ] Review an explicit release file list; never use `git add .` in this tree.
- [ ] Ensure all v5 TeX, section, analysis, result, figure, test and metadata
  files are included in the candidate archive.
- [ ] Install from `requirements.lock` in a clean Python 3.12.11 environment.
- [ ] Run `make PYTHON=.venv/bin/python all` successfully.
- [ ] Run `make PYTHON=.venv/bin/python release-check-v5` and resolve every
  failure; retain the printed reason for every skip.
- [ ] Confirm both PDFs compile twice with no undefined citations/references or
  overfull boxes.
- [ ] After the report inputs are frozen, run
  `.venv/bin/python analysis/build_v5_report_manifest.py --write` and review the
  resulting diff.
- [ ] Verify `results/v5_report_source_manifest.csv` and its sidecar digest.
- [ ] Reconstruct both 12-row PDSP paired-QAP tables and the support-threshold
  sensitivity from the released aggregate table.
- [ ] Confirm the Supplement contents lists S1--S7 and all subsections.
- [ ] Confirm four main upload figures, Supplementary Figure S1 and the graphical
  abstract match the PDF captions.
- [ ] Confirm `submission/files/Manuscript.pdf` and
  `submission/files/Additional_file_1.pdf` match the final root PDFs byte for byte.
- [ ] Confirm the graphical abstract is 920×300 RGB, white-background and at
  most 150,000 bytes.

## Data and provenance

- [ ] Re-hash every redistributed input against `data_manifest.csv`.
- [ ] Check `external_sources.csv` against the exact source files used.
- [ ] Confirm aggregate external-map tables contain no compound-level activity
  profiles or restricted source rows.
- [ ] Confirm PDSP source access and checksum; obtain editorial acceptance of the
  nonredistributed-source route.
- [ ] Verify that the recorded PDSP connectivity-block exclusions match the
  docking support used by every reported PDSP map.
- [ ] Preserve DOCKSTRING pose/receptor checksums, ODDT training/model checksums
  and the smina executable version/checksum.
- [ ] Confirm all licences and required acknowledgements, including Reaction
  Biology Corporation for KiRHub.
- [ ] Keep v4 manifests explicitly historical; distinguish them from the v5
  report-source manifest.

## Visual and editorial inspection

- [ ] Read both PDFs from start to finish at final size.
- [ ] Inspect every figure label, legend, table, symbol, reference and page break.
- [ ] Search extracted text for placeholders, the old title, stale v4 claims,
  `Figure 5`, and a main-text pocket figure.
- [ ] Check author names, affiliations, ORCIDs, corresponding-author email,
  contributions, funding, competing interests and AI disclosure with all
  coauthors.
- [ ] Confirm the journal's current article type, abstract, data-availability,
  supplementary-file and graphical-abstract requirements.

## Archive and upload

- [ ] Record the candidate commit or archive checksum.
- [ ] Record SHA-256 hashes for both PDFs and every upload asset.
- [ ] Deposit the exact tested candidate and mint a version-specific DOI.
- [ ] Insert that DOI into the availability statement, README and metadata;
  rebuild and re-hash without changing scientific content.
- [ ] Obtain final coauthor approval of the exact PDF bytes.
- [ ] Upload the manuscript, four main figures, Additional file 1, Supplementary
  Figure S1 and graphical abstract with matching titles and descriptions.

Current state: **v5 submission candidate; no release tag or version-specific DOI
is yet claimed.**
