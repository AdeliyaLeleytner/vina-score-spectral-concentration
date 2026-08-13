# Anastassiadis 2011 identity and chemical-dependence audit

This standalone audit maps all 178 compound name/CAS records from official
Supplementary Table 3 (DOI `10.1038/nbt.2017`) to PubChem. It exports **no HotSpot
assay values**. The fixed 21-target completeness flag was computed in memory:
176 compounds are complete and
the two excluded compound--target cells are listed in `summary.json`.

## Identity contract

CAS and workbook name were queried independently against PubChem PUG REST on
2026-08-10. Candidate lists, query URLs, the resolution status, and ambiguity
flags are retained in `identity_resolution_audit.csv`. A structure is usable only
when the resolution rule selects one CID and the PubChem InChIKey agrees exactly
with the Standard InChIKey recomputed from the returned PubChem SMILES by RDKit.
There are 157 usable structures and
21 ambiguity-flagged records.
For every unresolved record, `candidate_only_connectivity_blocks` contains the
union from all CAS/name candidate CIDs. `de_leakage_connectivity_blocks` is the
field to use for conservative calibration-pool exclusion: one selected block
for resolved records and the complete candidate union otherwise.

## Dependence contract

Usable structures were fingerprinted as Morgan radius 2, 2048 bits. Butina
clustering used Tanimoto distance <= 0.5 (`reordering=True`) in official workbook
order. This gives 130 clusters, including
114 singleton clusters. RDKit Bemis--Murcko
grouping gives 130 groups, including
114 singletons. Acyclic structures receive
identity-specific `ACYCLIC:<connectivity>` labels rather than sharing an empty
scaffold label. For whole-panel multiplier resampling, each of the
21 unresolved records receives an
identity-specific singleton, giving 151
groups across all 178 records.

## Cross-panel overlap

`cross_panel_overlap_summary.csv` reports exact full-Standard-InChIKey and
connectivity-block overlap with the complete 260,060-row DOCKSTRING, 72-compound
DAVIS, and 645-row PKIS2 supports. Connectivity-only hits are not called exact
chemical matches.

## Reuse boundary

NCBI states that it places no restrictions on use or distribution of molecular
database data, but also warns that submitters may retain third-party rights and
that NCBI cannot transfer those rights. PubChem likewise directs users to inspect
source-specific licensing. Accordingly, the frozen table is an identity/provenance
crosswalk with PubChem-derived structures, carries source attribution and policy
links, and contains no Anastassiadis assay measurements. It is not labelled CC0.

Primary policy pages:

* https://pubchem.ncbi.nlm.nih.gov/docs/downloads
* https://www.ncbi.nlm.nih.gov/home/about/policies/
