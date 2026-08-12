# Reproducibility contract: v5 submission candidate

## Two reconstruction levels

The repository separates report reconstruction from source-level reanalysis.

1. **Report reconstruction** starts from frozen result bundles. It regenerates
   figures and upload assets, checks manuscript-facing values, and compiles the
   article and Supplement.
2. **Source-level reanalysis** regenerates a named result bundle from its row-level
   inputs. Several analyses require files that cannot be redistributed and must
   be obtained from the sources in `external_sources.csv`.

The default build performs the first level. It does not rerun receptor
preparation, pose generation, docking or every statistical simulation.

## Pinned environment and report build

```bash
uv venv --python 3.12.11 .venv
uv pip install --python .venv/bin/python -r requirements.lock
make PYTHON=.venv/bin/python all
```

`requirements.lock` is the normative Python package lock. `environment.yml` and
`requirements.txt` provide equivalent convenience specifications; the Dockerfile
is an independent packaging check.

The v5 report dependency graph is:

1. `figures-v5`: regenerate the four main figures, Supplementary Figure S1 and
   graphical abstract from frozen results.
2. `submission-figures`: stage journal-named upload files and validate the
   graphical-abstract dimensions and size.
3. `verify-v5-numbers`: verify the report-source manifest, run the manuscript and
   Supplement numerical contracts, and reconstruct the primary and
   complete-case PDSP paired-QAP tables and support-threshold sensitivity from
   released aggregate/public inputs.
4. `manuscript-v5` and `supplement-v5`: compile twice and reject undefined
   references, undefined citations and overfull boxes.
5. `submission-v5`: combine those steps; `make all` points here.

The build does not create a tag or archive DOI.

## What the report tests protect

The v5 tests read values from the frozen CSV and JSON files and check every
load-bearing manuscript site. They also protect:

- the exact title across TeX and citation metadata;
- an abstract of at most 350 words with a `Scientific Contribution` heading;
- four main figures and one Supplementary pocket figure;
- two-sided paired-QAP inference with Holm and Bonferroni correction within each
  12-row PDSP QAP table;
- PDSP connectivity-overlap exclusion, censor-aware weakening and the
  chemically shifted complete-case support boundary;
- the post-hoc status of every external comparison and the non-headline status
  of predictor fusion;
- the kinase and KiRHub non-generalisation statements;
- ligand-wise ranking deterioration after centring;
- complete Supplement contents navigation; and
- journal-compatible graphical-abstract output.

`results/v5_report_source_manifest.csv` and its sidecar digest record the exact
local sources, code and frozen inputs consumed by this report build. They do not
claim source-level regeneration of the scientific result bundles.

After an intentional change to a report input, refresh this manifest before the
release checks with:

```bash
.venv/bin/python analysis/build_v5_report_manifest.py --write
```

Review the diff before accepting the refreshed inventory.

Run the scoped checks alone with:

```bash
make PYTHON=.venv/bin/python verify-v5-numbers
```

Run the complete repository suite with:

```bash
.venv/bin/python -m pytest -q
```

Run the complete local release gate, including explicit skip reasons, with:

```bash
make PYTHON=.venv/bin/python release-check-v5
```

The included GitHub Actions workflow runs this same gate in a clean Linux checkout
and retains the compiled PDFs and staged artwork as workflow artifacts.

The current full suite reports 10 explicit skips: seven tests for the retired
descriptor-geometry exploratory artifact, one unavailable production artifact in
`test_public_panel_domain_utility.py` that the v5 report does not consume, and two
v4 metadata-consistency checks superseded by the v5 submission. A different skip
count or reason requires review; exclusions must not be silent.

## Source-level producers

Each result directory records its producer, parameters, seeds, input checksums
and output checksums in its README, summary or manifest. The main v5 producers
include:

- `analysis/nonvina_scorer_transport.py` for the fixed-pose scorer comparison;
- `analysis/public_residual_null_audit.py` for matched nulls;
- `analysis/public_chemical_domain_controls.py` and
  `analysis/public_descriptor_domain_specificity.py` for ligand-domain checks;
- `analysis/pdsp_counterscreen_retrieval.py` for PDSP comparisons;
- `analysis/spd_external_validation.py` and
  `analysis/spd_normalization_controls.py` for the Novartis panel;
- the fixed-20 kinase and KiRHub producers named in their result bundles;
- the calibration-recovery and biological-core-mode producers named in their
  result bundles; and
- `analysis/residual_mie_boundary_audit.py` for the ligand-wise ranking boundary.

Before rerunning a producer, inspect `--help`, its current result metadata and
`external_sources.csv`. Do not overwrite frozen outputs until the replacement has
passed its unit test and all dependent report tests. A source refresh is a new
scientific artifact and should receive new checksums.

The older strict-public v4 workflow remains available under the explicitly named
`public-*`, `manuscript-v4` and `supplement-v4` Make targets. Its
`results/public_core_source_manifest.csv` covers only that historical evidence
builder and is not the v5 report manifest.

## Data and licensing boundary

Redistributed inputs are listed in `data_manifest.csv`. Nonredistributed inputs,
including PDSP and the Novartis source table, are listed in
`external_sources.csv` with their access routes and integrity checks.
`DATA_LICENSES.md` explains which derived target-pair tables can be shared.

The package releases aggregate target-pair geometry, not compound-level PDSP,
Novartis or KiRHub activity profiles. Source-level preprocessing therefore
requires the original checksum-matched files, while report-level verification
uses the released aggregates and frozen results. The primary and complete-case
PDSP paired-QAP tables and support-threshold sensitivity are independently
rebuilt by `analysis/reproduce_pdsp_derived_qap.py` from `target_pairs.csv` and
the redistributed Docking-44 matrix.

For PDSP, Docking-44 ligands sharing a 14-character connectivity block with the
assay export are excluded before a docking map is constructed. The primary map
uses the remaining eligible support, where the residual map is descriptively
closer to the experimental map than the raw map. All external comparisons are
post hoc. The complete-case sensitivity is a chemically shifted support
analysis, not a pure test of missing-value handling. The paired QAP table uses
two-sided probabilities and reports both Holm and Bonferroni corrections across
its 12 tests; predictor fusion is not treated as a resolved headline result.

## Final clean-archive check

The current workspace is not itself a release artifact. Before submission:

1. review an explicit list of v5 files;
2. commit or archive only that list;
3. build in a clean directory with the pinned environment;
4. run the scoped and full tests;
5. inspect both PDFs and all upload figures;
6. record SHA-256 hashes; and
7. deposit that exact candidate and insert its version-specific DOI.

PDF bytes may differ across TeX distributions even when text, figures, page
count and page size agree. The archived environment, source files and hashes are
therefore part of the release record.
