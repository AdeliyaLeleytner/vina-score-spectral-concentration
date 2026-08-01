# Pre-submission major-revision disposition

Status date: 2026-08-01.

| Review point | Resolution | Location |
| --- | --- | --- |
| Title overstates loss of concentration | Replaced “before, but not after” with a title that distinguishes what centering reveals from what it does not correct. The Results and Discussion state that residual surfaces remain concentrated. | Title; Fig. 4; Discussion |
| Claims broader than the evidence | The paper refers to the two tested Vina systems. “General-binding axis” was replaced by “shared docking-score axis”; RF-Score is explicitly a small illustrative sensitivity. | Manuscript throughout; Fig. S3 |
| Target-panel uncertainty missing | Added leave-one-target-out analyses for both surfaces, 200 uniform and family-stratified target subsets, matched-target-count comparison, and residual target-subsampling curves. | Fig. 1c,d; Fig. S2; Table S4 |
| Chemical-support uncertainty too narrow | Docking-44 uses a paired Butina-cluster bootstrap. DOCKSTRING scaffold analysis now spans five independent 15,000-molecule supports with 100 molecule and 100 scaffold replicates per support. An unclipped positive-score sensitivity was added. | Fig. 4a; Fig. S6; Tables S2/S4; evidence ledger |
| Only 40 permutations and no multiplicity control | Replaced with 500 transformation-matched permutations, a studentized maximum-deviation simultaneous envelope controlling family-wise error over ranks, and five independent 100-permutation stability series. | Results; Methods; Supplementary note |
| Additive-null interpretation missing | Added a fitted additive null preserving ligand effects, target effects, and target-specific residual variances. Observed residual PR values remain far below null medians 42.40 and 56.48. | Fig. 4b; Results; Methods |
| Experimental comparison omitted residual space | The main text and figure now show both raw and residual docking–experiment contrasts. The paired difference narrows from 2.006 (1.174–2.673) to 0.483 (-0.071–0.921). Exact-relation median aggregation remains primary, with 100 multiple imputations. | Fig. 3; Results; Table S3 |
| Uncertainty of the centering contrast | Added paired raw-to-residual cluster-bootstrap distributions and ratios for Docking-44, multi-support DOCKSTRING contrasts, and paired ligand intervals for both matched ChEMBL contrasts. | Fig. 3b; Fig. 4a; evidence ledger |
| Practical consequence unclear | Added a leave-one-ligand-out target-preference benchmark with target-only, experimental-prior, and ligand-shuffle controls. It shows that higher residual dimension does not by itself improve target ranking and yields a concrete validation recommendation. | Fig. 5; Results; Discussion; Table S5 |
| DTI rank–selectivity section weakens the paper | Removed from the main narrative. All 20 arms are the primary exploratory Supplement panel; the outcome-restricted 12-arm subset is a sensitivity. | Fig. S5; Tables S6–S8 |
| Novelty insufficiently positioned | Added SePreSA and the Fukunishi docking-matrix/PCA studies. Novelty is defined as large-panel spectral quantification, matched preprocessing, explicit estimand separation, and composition sensitivity—not PCA or two-way normalization itself. | Introduction and Discussion |
| Public reproducibility package required | Bundled all frozen inputs, 26-file checksum manifest, pinned requirements, Dockerfile, Makefile, provenance, file-specific data licenses, and exact reproduction commands. | Repository root |
| Main story diluted by RF/DAVIS/Boltz | RF-Score/Docking-47 and DAVIS/Boltz figures were moved to the Supplement. The five main figures now form one spectral-to-operational argument. | Figs. 1–5; Figs. S3–S4 |
| PC1 loadings and row-mean definition absent | Added target-family loading plots. The text states that PC1 correlates with raw score-scale per-ligand means, not standardized row means. | Fig. 2c,d; Results |
| Complete-case and clipping sensitivity underplayed | The 1.320 complete-case value is discussed as a change of chemical support; DOCKSTRING clipping has an explicit unclipped sensitivity. | Results; Methods; Table S2 |
| Funding, links, and formatting | Funding language is finalized; GitHub and Zenodo are numbered references; main and supplementary sources use double spacing. | Declarations; References; LaTeX sources |
| Terminology and PDF defects | “Rank” is qualified as PR effective dimension; the main text uses “residual structure.” Figures use vector text, rho extracts correctly, all DTI points map to Table S6, and no blank continuation page remains. | Manuscript; figures; Supplement |

## Remaining submission handoff

- Obtain final coauthor confirmation of author order, contribution statement, funding,
  affiliation, and AI-use disclosure.

The public GitHub repository and Zenodo concept DOI `10.5281/zenodo.21733767` are now
included in the manuscript and release metadata.
