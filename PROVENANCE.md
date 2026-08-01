# Data and computation provenance

## Reproducibility boundary

`make all` starts from the frozen matrices in `data/frozen/` and reproduces all reported
spectral statistics, target and ligand sensitivities, empirical nulls, figures, tables, and
PDFs. Receptor preparation, pose generation, docking, and RF-Score feature generation are
upstream of this release and are not recomputed.

Every frozen input has a version, license, byte size, and SHA-256 digest in
`data_manifest.csv`. `analysis/verify_inputs.py` fails before analysis if any artifact is
missing or changed. Gzip compression changes storage bytes but not the tabular content read
by pandas.

## Primary docking matrices

### Docking-44

- Frozen file: `data/frozen/df_final_v4.csv.gz`.
- Source: Nikitin et al., *Pharmaceutics* 2025, 17, 1573,
  doi: `10.3390/pharmaceutics17121573`.
- Source project: `https://github.com/chemagents/ld50-antitargets`.
- Analysis support: 12,651 ligands and the 44 target columns named in
  `analysis/build_evidence.py`; Vina-GPU 2.0 source scores.
- Primary preprocessing: positive/source-censored scores clipped to zero, 1.466% missing
  entries filled by the target mean, then column standardization. The frozen matrix contains
  111 zero-censored cells (0.020%).
- Chemical clusters: the `Butina_clusters` column in the same file.

### DOCKSTRING-58

- Frozen file: `data/frozen/dockstring-dataset.tsv.gz`.
- Source: DOCKSTRING v1 Figshare dataset, doi:
  `10.6084/m9.figshare.16511577.v1` (Apache-2.0).
- Analysis support: 260,060 complete molecules and 58 targets.
- Primary preprocessing: positive scores clipped to zero, complete rows retained, then
  column standardization.
- Scaffold sensitivity: Bemis–Murcko scaffolds computed by RDKit on one seeded 15,000-row
  support; acyclic molecules are singleton clusters.

Broad family labels used only for stratified target subsampling are in
`data/target_families.csv`. They are intentionally coarse curation labels, not an ontology.

## Supporting scoring-function blocks

- `rescore_ml_rfscore_scores.csv`: 41-ligand × 14-target common block. RF-Score v1 was run
  with ODDT's PDBbind-2016 model on smina/Vina poses generated at exhaustiveness 8.
- `rescore_ml_rfscore_v2_scores.csv`: 35-ligand × 20-target common block. RF-Score v1 used
  poses generated with smina/Vinardo; original Vina columns provide a same-support reference.

The small sample sizes and one older learned scorer make these sensitivity analyses rather
than broad evidence about modern neural scoring functions.

## Experimental affinity controls

- ChEMBL activity records are from release 34 and retain CC BY-SA 3.0.
- The primary 74 × 6 control requires exact standard relation, full InChIKey, endpoints Ki,
  Kd, IC50, or EC50, and uses the median over repeated compound–target measurements.
- Human binding and Ki/Kd restrictions, missing-data estimators, fully observed blocks, and
  repeatability-calibrated noise are regenerated or loaded from the checksummed summaries
  listed in the manifest.
- The historical most-potent aggregation is retained only as a labelled sensitivity.

## DAVIS, Boltz-2, and exploratory DTI arms

`davis_complete.tab.gz` supplies only support counts for the supplementary precision-at-five
audit. Frozen rank and performance values for all 20 heterogeneous DTI scorer arms are in
`dti_leaderboard_summary.json`. These arms share data, representations, and architectures;
their correlations are descriptive and are not independent model-population inference.
The 12 above-chance arms are an explicitly outcome-restricted sensitivity.

## Randomness and software

All new stochastic analyses use NumPy seed 0. Versions are pinned in `requirements.lock`;
the reference environment uses Python 3.12.11. The generated
`results/evidence_summary.json` is the numeric authority for the manuscript.
