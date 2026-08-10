# Data licenses

The repository `LICENSE` applies to author-written software and documentation. It does not
override upstream data licenses. `data_manifest.csv` is the machine-readable authority for
redistributed frozen inputs.

- Author-generated analysis summaries and target annotations are released under CC BY 4.0
  unless an upstream share-alike condition applies.
- ChEMBL release 34 records and derived row-level tables retain CC BY-SA 3.0.
- DOCKSTRING v1 retains its Apache-2.0 license and source DOI
  `10.6084/m9.figshare.16511577.v1`.
- wwPDB source structures are CC0; compact author-derived descriptors are CC BY 4.0.
- The DAVIS-Complete V3 table is the exact upstream Harvard Dataverse TSV export and is
  CC0 1.0; cite Wu (2025), DOI `10.7910/DVN/RTQGP1`, and the original DAVIS affinity
  study, DOI `10.1038/nbt.1990`.
- The official PKIS2 S4 table is redistributed under CC BY 4.0; cite Drewry et al. (2017),
  DOI `10.1371/journal.pone.0181585`.
- The PKIS1 publisher archive is not redistributed. It can be retrieved from the official
  publisher URL and checksum-validated with `analysis/fetch_pkis1_supplement.py` (or
  `make fetch-pkis1`). The accepted source file has SHA-256
  `1ffbe7fd0b4fc1ef72f2434a2b247d15c0f2b4d3c34668f112f4b4c9e4ead7dc`; cite
  Elkins et al. (2016), DOI `10.1038/nbt.3374`.
- The Saifudeen et al. KiRHub article is published under CC BY-NC-ND 4.0. We do not
  interpret that article license as a separate license for the underlying inhibition data:
  the official KIRHub portal states that those data are the property of Reaction Biology
  Corporation and require proper acknowledgement for download, use or publication. The
  source workbook is therefore not redistributed. Its accepted SHA-256 is
  `d9eef358396b193834b0c4d48ccd8cadb43697a6458cdcb50415af5a7e4e0b03`; the release contains
  aggregate analytical summaries, not the workbook, compound-by-target activity rows or
  per-compound profiles. Cite DOI
  `10.1038/s41587-026-03090-8`, consult `https://kirhub.fredhutch.org/` for the current data
  notice, and acknowledge Reaction Biology Corporation.

- The KLIFS API response is not redistributed. Derived pocket identities are released with
  response checksum and retrieval provenance; cite KLIFS and observe current database terms.
- The Novartis Secondary Pharmacology Database release at Zenodo record `8103950` is CC BY
  4.0. The row-level source is not duplicated here; the accepted file has SHA-256
  `7132723f85e746de2f8387d01dcde6ffff703c92561fda9751cbd6753e900240`.

Software assistants used during editing receive no authorship or data ownership.

## Released target-pair geometry and why it is not the source data

Every fixed-panel number in the manuscript is a function of one released object: the
190-element upper triangle of a 20-by-20 target correlation matrix, in the raw and the
within-ligand-centred parameterisation, for four experimental panels and the docking
reference. It is distributed as
`results/released_pair_geometry_ledger/target_pair_geometry.csv`, alongside the two
analysis-specific tables `results/fixed20_centering_panel_sensitivity/target_pair_geometry.csv`
and `results/ordinal_docking_geometry/target_pairs.csv`.

Each entry is a single correlation coefficient computed **across** compounds. The tables
contain no compound identifier, compound-level value or per-compound profile and are treated
here as author-generated aggregate statistics under the CC BY 4.0 clause above, not as a
redistribution of the source panel's activities. The target-pair edges are also the object
under test: releasing them makes every reported geometry comparison auditable without
releasing the underlying compound-by-target table.

The KiRHub source workbook, its compound-by-target activity rows and its per-compound
profiles remain unredistributed. Readers who need compound-level KiRHub data must obtain it
from the KIRHub portal under Reaction Biology Corporation's terms; `analysis/fetch_kirhub_supplement.py`
retrieves the exact checksum-verified bytes the analysis consumed.
