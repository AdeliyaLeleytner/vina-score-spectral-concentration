# Pre-submission major-revision disposition

Status date: 2026-08-01.

| Review point | Resolution | Location |
| --- | --- | --- |
| Claims broader than the evidence | Title, abstract, discussion, and conclusion now refer to the two tested Vina systems. “General-binding axis” was replaced by “shared docking-score axis”; RF-Score is explicitly a small sensitivity. | Manuscript throughout |
| Target-panel uncertainty missing | Added leave-one-target-out analyses for both surfaces, 200 uniform and family-stratified target subsets, matched-target-count comparison, and target-subsampling curves for residual surfaces. Added a Bemis–Murcko scaffold-cluster bootstrap for DOCKSTRING. | Fig. 1c,d; Fig. S2; Table S4; evidence ledger |
| Only 40 permutations | Replaced with 500 transformation-matched permutations on a common 12,000-row support and five independent 100-permutation stability series. Counts are 3→7 and 4→10 before→after centering and are identical across series. | Results, Methods, Supplementary note |
| Experimental affinity overgeneralized | Exact-relation median aggregation is primary; human binding and Ki/Kd blocks are prominent. The comparison is stated only for the observed matched ChEMBL surfaces. Legacy most-potent aggregation is a sensitivity. | Fig. 4; Results; Table S3 |
| DTI rank–selectivity section weakens the paper | Removed from the main narrative and figures. All 20 arms are the primary exploratory Supplement panel; the outcome-restricted 12-arm subset is a sensitivity. Pearson, Spearman, and Kendall results and high-rank chance counterexamples are reported. | Fig. S3; Tables S5–S7 |
| Novelty insufficiently positioned | Added SePreSA and the Fukunishi docking-matrix/PCA studies. Novelty is defined as large-panel spectral quantification, matched preprocessing, explicit estimand separation, and composition sensitivity—not PCA or two-way normalization itself. | Introduction and Discussion |
| Public reproducibility package required | Bundled all frozen inputs, 26-file checksum manifest, pinned requirements, Dockerfile, Makefile, provenance, file-specific data licenses, and exact reproduction commands. | Repository root |
| Terminology and PDF defects | “Raw” is replaced where needed by “column-standardized”; “rank” is qualified as PR effective dimension; the main text uses “residual structure.” Figures were regenerated with vector text, rho extracts correctly, Fig. S3 has point IDs, and the obsolete/blank continuation layout was removed. | Manuscript, figures, Supplement |

## Remaining submission handoff

- Insert the public GitHub URL after the repository exists.
- Archive the exact `v1.0.0` tag on Zenodo and insert its DOI.
- Obtain final coauthor confirmation of author order, contribution statement, funding,
  affiliation, and AI-use disclosure.
