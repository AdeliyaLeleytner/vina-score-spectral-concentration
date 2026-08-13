# Data licenses

The repository `LICENSE` applies to author-written software and documentation.
It does not override upstream data licences.

Four inventories, in order of scope:

- `external_sources.csv` — the source inventory for the v5 manuscript:
  its role in the paper, whether it is redistributed, how to obtain it, its
  integrity record, licence and fetcher. Start here.
- `JOC_ACCESS_MATRIX.csv` — a conservative, source-by-source evidence ledger for
  the journal's reproducibility rule. `VERIFIED` means that the named property
  has direct evidence in the release; `UNKNOWN` identifies evidence or an
  editorial/rights-holder determination still needed. It is not a blanket
  policy-compliance claim. `analysis_producer_present_status` means that code
  consuming the external artifact is released; it does not imply that a fetcher,
  immutable response snapshot or clean replay exists.
- `data_manifest.csv` — machine-readable inventory of the registered shipped
  data artifacts. It is not a complete repository file listing. Its `v4_scope`
  column distinguishes the strict-public v4 inputs from retained historical
  material.
- `results/public_core_source_manifest.csv` — the exact files opened by the v4
  evidence builder. Narrower than the v5 evidence base; see the note at the end.

## Evidence inputs redistributed with the release

- Docking-44 (`data/frozen/df_final_v4.csv.gz`) is a frozen project export from
  Nikitin et al., *Pharmaceutics* (2025), DOI
  `10.3390/pharmaceutics17121573`, and is redistributed under CC BY 4.0 with
  source attribution. Its working-tree mode is `0644`, making it readable to
  third parties without changing its checksum-authenticated bytes.
- DOCKSTRING v1 retains Apache-2.0 and source DOI
  `10.6084/m9.figshare.16511577.v1`. The row-aligned identity contract and its
  provenance record are derived from that source and retain Apache-2.0.
- DAVIS-Complete V3 is the exact upstream Harvard Dataverse TSV export,
  recompressed without content changes, and is CC0 1.0. Cite Wu (2025), DOI
  `10.7910/DVN/RTQGP1`, and Davis et al. (2011), DOI `10.1038/nbt.1990`.
- The official PKIS2 S4 table is redistributed under CC BY 4.0. Cite Drewry et
  al. (2017), DOI `10.1371/journal.pone.0181585`.
## Evidence inputs that are not redistributed

These support external-agreement or scorer-sensitivity analyses. Access,
integrity and redistribution evidence is source-specific; unresolved properties
are marked `UNKNOWN` in `JOC_ACCESS_MATRIX.csv` rather than inferred from public
visibility.

- **PDSP $K_i$ database** — the primary exploratory raw-versus-residual map comparison and the
  Certified-subset sensitivity are computed from it. No redistribution licence was
  located. Download from `https://pdsp.unc.edu/databases/kiDownload/download.php`;
  the export consumed here has SHA-256
  `45c9a18ac30f1fad350d1dde186bc1f226c5a75d474ca50f50713852a5637ac6`, recorded
  with row counts in `results/pdsp_counterscreen_retrieval/summary.json`. Cite
  Roth et al. (2000). `analysis/fetch_pdsp_ki_database.py` retrieves the public,
  no-login CSV route and validates its byte count, header and SHA-256. This
  resolves retrieval mechanics but does not supply a redistribution licence.
- **Novartis Secondary Pharmacology Database** — the safety-panel partial
  agreements. The Zenodo record `8103950` release is CC BY 4.0. The row-level
  source is not duplicated here; `analysis/fetch_spd_source.py` retrieves the
  exact no-login file and checks the required header, byte count and SHA-256. The accepted file has SHA-256
  `7132723f85e746de2f8387d01dcde6ffff703c92561fda9751cbd6753e900240`. Cite
  Sutherland et al. (2023).
- **PKIS1 publisher archive** — third panel defining the frozen kinase endpoint,
  and the discovery panel for the exploratory five-mode compression. Not redistributed. Retrieve
  from the official publisher URL and checksum-validate with
  `analysis/fetch_pkis1_supplement.py`. Accepted SHA-256
  `1ffbe7fd0b4fc1ef72f2434a2b247d15c0f2b4d3c34668f112f4b4c9e4ead7dc`; cite
  Elkins et al. (2016), DOI `10.1038/nbt.3374`.
- **KiRHub** — the fourth kinase panel, on which the paper reports a
  non-replication. The Saifudeen et al. article is published under
  CC BY-NC-ND 4.0. We do not interpret that article licence as a separate licence
  for the underlying inhibition data: the official KIRHub portal states that
  those data are the property of Reaction Biology Corporation and require proper
  acknowledgement for download, use or publication. The source workbook is
  therefore not redistributed. Accepted SHA-256
  `d9eef358396b193834b0c4d48ccd8cadb43697a6458cdcb50415af5a7e4e0b03`; the release
  contains aggregate analytical summaries, not the workbook, compound-by-target
  activity rows or per-compound profiles. Cite DOI `10.1038/s41587-026-03090-8`,
  consult `https://kirhub.fredhutch.org/` for the current data notice, and
  acknowledge Reaction Biology Corporation.
- **KLIFS** — the 85-residue ATP-pocket identity baseline on the 190-pair kinase
  endpoint. The API response is not redistributed. Derived pocket identities are
  released with response checksum
  (`041a159e662a27696554867acbe1ac7fb35d36f71483d42ab4ac90e219dbef8f`) and
  retrieval provenance; cite KLIFS and observe current database terms.
- **UniProt** — receptor sequence-identity baselines. API responses, released as
  derived identities with response checksums recorded per analysis. The PDSP
  and SPD snapshots have checksum-gated fetchers
  (`051690a9…1b408a` for the PDSP mapping, `b0f6213a…15689` for the safety
  panel).
- **RCSB PDB and PDBe/SIFTS** — support the receptor-species sensitivity and
  the nine-pair cross-panel target mapping. The derived tables are shipped, but
  the exact upstream API responses and executable producers were not retained;
  the access matrix therefore leaves their source-byte, digest, checksum-gate
  and clean-replay states unknown. RCSB programmatic PDB data are CC0; a precise
  JoC-compatible licence determination for the consumed SIFTS mapping remains
  to be recorded.
- **ODDT scoring resources** — RF-Score v1--v3, NNScore 2.0 and PLECscore
  linear use ODDT 0.7 package resources derived from PDBbind 2016. Training-table,
  coefficient and fitted-model checksums are recorded in
  `results/nonvina_scorer_transport/summary.json`. They are not redistributed;
  source rescoring used the separately documented Python 3.9/ODDT 0.7 environment
  and remains subject to the ODDT-package and PDBbind terms.
- **DOCKSTRING pose archives** — the retained Vina poses used for the
  fixed-pose scorer comparison. The nine Figshare file identifiers and published
  MD5 values are embedded in `analysis/dockstring_vina_term_decomposition.py`;
  the analysis verifies each archive before use.
- **DOCKSTRING receptor PDBQTs** — the nine package receptors used for
  fixed-pose rescoring. Their SHA-256 values are recorded per target in the
  scorer summary and they retain the DOCKSTRING Apache-2.0 terms.
- **smina executable** — smina 2020.12.10, conda-forge build `b08c07c`, was
  used for the Vina and Vinardo rescoring rows. The binary SHA-256 and version
  string are recorded in the scorer summary; the executable is not redistributed.

The ChEMBL release-34 extract used for the ligand-wise ranking boundary is
redistributed under CC BY-SA 3.0 with its provenance record and SHA-256 in
`data_manifest.csv`.

## Historical release material not used for v5 conclusions

The retained strict-public v4 audit includes an Anastassiadis/HotSpot identity
workflow, but v5 does not use that panel or crosswalk for a scientific claim.
The source workbook is not redistributed; its official publisher copy is pinned
to SHA-256
`cd756bf2b6ad541a1781508c563caf0da6da876dfb71f2546fbff02e13d98684`.
The included identity/provenance crosswalk is a mixed-source table containing
workbook identifiers, PubChem molecular records and author-generated resolution
fields, but no HotSpot assay values. It follows the NCBI/PubChem molecular-data
policy; because NCBI cannot transfer possible third-party contributor rights,
the crosswalk is not labelled CC0. This historical material is registered by
the separate public-core manifest, not `external_sources.csv` or the JoC access
matrix.

Author-generated summaries and target annotations are released under CC BY 4.0
unless an upstream condition applies. Software assistants used during editing
receive no authorship or data ownership.

## ChEMBL-derived repository material

ChEMBL release 34 records and derived row-level tables retain CC BY-SA 3.0.
Only the checksum-identified target-ranking extract named above enters v5; other
ChEMBL tables are retained as historical material.

## Released target-pair geometry and why it is not the source data

The fixed 20-kinase comparisons are functions of one released object: the
190-element upper triangle of a 20-by-20 target correlation matrix, in the raw
and within-ligand-centred parameterisations, for four experimental panels and
the docking reference. It is distributed as
`results/released_pair_geometry_ledger/target_pair_geometry.csv`, alongside the two
analysis-specific tables `results/fixed20_centering_panel_sensitivity/target_pair_geometry.csv`
and `results/ordinal_docking_geometry/target_pairs.csv`.

Each entry is a single correlation coefficient computed **across** compounds. The tables
contain no compound identifier, compound-level value or per-compound profile and are treated
here as author-generated aggregate statistics under the CC BY 4.0 clause above, not as a
redistribution of the source panel's activities. The target-pair edges are also the object
under test: releasing them makes every reported geometry comparison auditable without
releasing the underlying compound-by-target table.

For PDSP and the Novartis panel, the release likewise includes aggregate
target-pair tables under the corresponding `results/` directories. These tables
contain no compound identifier or compound-level activity profile; source-level
preprocessing still requires the checksum-pinned upstream files.
`analysis/reproduce_pdsp_derived_qap.py` reconstructs and verifies the primary
and complete-case 12-row paired-QAP tables and the support-threshold
sensitivity from the aggregate PDSP table and redistributed Docking-44 matrix
alone.

The KiRHub source workbook, its compound-by-target activity rows and its per-compound
profiles remain unredistributed. Readers who need compound-level KiRHub data must obtain it
from the KIRHub portal under Reaction Biology Corporation's terms; `analysis/fetch_kirhub_supplement.py`
retrieves the exact checksum-verified bytes the analysis consumed.

## Note on manifest scope

`results/public_core_source_manifest.csv` and the `v4_scope` column of
`data_manifest.csv` describe the v4 strict-public evidence boundary, which
predates the external-agreement results that carry v5. They remain accurate for
what they cover and are not the authority for the v5 evidence base;
`external_sources.csv` is.
