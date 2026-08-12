# Journal of Cheminformatics submission preflight: v5

Article type: **Research article**.

Exact title:

> **A shared ligand-wide score axis obscures library-dependent target geometry
> in docking matrices**

## Claim gate

The submitted claim is deliberately narrower than “docking beats sequence.” Raw
docking correlations contain a shared ligand-wide score axis. Ligand-wise
centring reveals a structured but library-dependent target map. The residual map
agrees better with external pharmacology on some diversified panels, with mixed
results elsewhere.

The primary PDSP result is a descriptive raw-versus-residual comparison on
target pairs supported by quantified $K_i$ measurements. It uses the eligible
full Docking-44 support after removing ligands that share a connectivity block
with PDSP. The residual map is closer to the experimental map on that support,
but residual agreement weakens when censored cells are restored and largely disappears on
the chemically shifted complete-case Docking-44 support.

All external comparisons are post hoc. Paired target-label QAP inference is
two-sided, with Holm and Bonferroni correction within each 12-row QAP table. The
sequence-and-family fusion remains a sensitivity analysis and must not be
presented as significant or as a headline result.

Required boundaries:

- PDSP is conditional on target pairs with quantified $K_i$ measurements.
- PDSP-overlapping Docking-44 connectivity blocks are excluded before map
  construction.
- Restoring censored PDSP values weakens the separation.
- Restricting Docking-44 to complete rows changes its chemical support and
  largely removes residual agreement; this is not a pure imputation comparison.
- Kinase and SPD evidence is mixed; KiRHub is a non-replication.
- The map depends on ligand support and is not an intrinsic protein network.
- Recovery from about 200 ligands is source-library recovery, not a universal
  threshold.
- Ligand-wise target ranking gets worse after centring in the boundary audit.
- The pocket-volume result and display are Supplement-only; the main text contains
  one limitations cross-reference and makes no mechanistic claim from it.
- Fixed-pose scorer transport does not compare independent docking pipelines.

## Report-format gate

- The abstract uses Background, Results, Conclusions and the required
  **Scientific Contribution** heading, contains no citations, and is at most 350 words.
- The abstract and figure captions carry only decision-relevant numbers; detailed
  uncertainty remains in Results, tables and frozen CSV files.
- Keywords, abbreviations and declarations are present.
- Main text contains four figures in citation order:
  `fig1_shared_axis`, `fig2_residual_structure`,
  `fig3_external_agreement`, `fig4_cost_and_core`.
- `fig_pocket_volume` is Supplementary Figure S1, not a main figure.
- The graphical abstract is 920 by 300 pixels, white-background RGB and no more
  than 150,000 bytes.
- Main and Supplement use double spacing and line numbers for review.
- All Supplement sections and subsections appear in its contents page.
- No generated figure is generative artwork; `promo/` is outside the submission.

## Evidence and availability gate

- `external_sources.csv`, `data_manifest.csv` and `DATA_LICENSES.md` agree on
  access, redistribution and licence boundaries.
- Aggregate target-pair tables contain no compound-level activity profiles.
- PDSP has no located redistribution licence; source-level reconstruction uses
  the named manual download and recorded SHA-256.
- The Novartis panel, PKIS1, KiRHub, Anastassiadis, KLIFS, UniProt, ODDT models
  and DOCKSTRING pose archives have named access routes and integrity checks.
- Reaction Biology Corporation is acknowledged for KiRHub data.
- The release does not claim to rerun receptor preparation, pose generation or
  docking.

## Build gate

```bash
uv venv --python 3.12.11 .venv
uv pip install --python .venv/bin/python -r requirements.lock
make PYTHON=.venv/bin/python all
```

The default target must regenerate and stage v5 figures, verify the report-source
manifest, run main-manuscript and Supplement checks, reconstruct the primary and
complete-case PDSP paired-QAP tables and support-threshold sensitivity from released
aggregates, compile each PDF twice, and reject undefined
references, undefined citations and overfull boxes. Source-level reanalysis is
separate and may require the external inputs above.

For the complete release gate, including the repository-wide suite and explicit
skip reasons, run `make PYTHON=.venv/bin/python release-check-v5`.

Before accepting a build:

1. Run the scoped v5 tests and the full repository suite; document any
   intentionally historical exclusion.
2. Inspect every page of both PDFs at final size.
3. Inspect the graphical abstract and all upload figures independently.
4. Search the built text for placeholders, stale v4 language, the old title and
   the removed main-text pocket reference.
5. Record SHA-256 hashes for the exact PDFs and upload assets.

## External actions required immediately before submission

1. Obtain all coauthors' approval of the exact manuscript, author order,
   affiliations, contributions, funding, competing interests and AI-use wording.
2. Assemble an explicit release file list; do not stage the current dirty tree
   wholesale.
3. Build and test that exact candidate from a clean archive or checkout.
4. Deposit the immutable v5 archive, insert its version-specific DOI into the
   availability statement and metadata, and rebuild once more.
5. Confirm that the journal accepts the source-plus-checksum route for PDSP under
   its current data-availability policy.
6. Upload four main figures, Supplementary Figure S1, Additional file 1 and the
   graphical abstract with matching captions.

Repository: <https://github.com/AdeliyaLeleytner/vina-score-spectral-concentration>

Current state: submission candidate under verification; no v5 tag or
version-specific DOI is claimed.
