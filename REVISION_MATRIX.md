# Pre-submission major-revision disposition

Status date: 2026-08-01.

| Review point | Resolution | Location |
| --- | --- | --- |
| Title and negative operational claim too strong | Title now limits scope to two large Vina matrices and says “without a resolved target-ranking gain.” The Results state both that residual concentration remains and that the benchmark establishes neither gain nor equivalence. | Title; Abstract; Results; Discussion |
| Claims broader than the evidence | The paper refers to the two tested Vina systems. “General-binding axis” was replaced by “shared docking-score axis”; RF-Score is explicitly a small illustrative sensitivity. | Manuscript throughout; Fig. S3 |
| Target-panel uncertainty missing | Added leave-one-target-out analyses for both surfaces, 200 uniform and family-stratified target subsets, matched-target-count comparison, and residual target-subsampling curves. | Fig. 1c,d; Fig. S2; Table S4 |
| Chemical-support uncertainty too narrow | Docking-44 uses a paired Butina-cluster bootstrap. DOCKSTRING scaffold analysis now spans five independent 15,000-molecule supports with 100 molecule and 100 scaffold replicates per support. An unclipped positive-score sensitivity was added. | Fig. 4a; Fig. S6; Tables S2/S4; evidence ledger |
| Only 40 permutations and no multiplicity control | Replaced with 500 transformation-matched permutations, a studentized maximum-deviation simultaneous envelope controlling family-wise error over ranks, and five independent 100-permutation stability series. | Results; Methods; Supplementary note |
| Additive Gaussian null too idealized | Retained the fitted additive Gaussian null and added a nonparametric residual-permutation null that samples from each target's empirical residual values before transformation-matched re-centering. Both give medians 42.40 and 56.48. A one-ligand-per-Butina-cluster sensitivity gives observed/null medians 9.43/42.33. | Fig. 4b; Results; Methods; Table S4 |
| Experimental comparison omitted residual space | The main text and figure now show both raw and residual docking–experiment contrasts. The paired difference narrows from 2.006 (1.174–2.673) to 0.483 (-0.071–0.921). Exact-relation median aggregation remains primary, with 100 multiple imputations. | Fig. 3; Results; Table S3 |
| Uncertainty of the centering contrast | Added paired raw-to-residual cluster-bootstrap distributions and ratios for Docking-44, multi-support DOCKSTRING contrasts, and paired ligand intervals for both matched ChEMBL contrasts. | Fig. 3b; Fig. 4a; evidence ledger |
| Practical consequence unclear | The operational analysis now proves that unscaled row centering is exactly target-rank invariant and decomposes the observed ranking change into target-specific scale and ligand-offset-by-inverse-scale effects. It reports absolute, five transformed representations and three target priors; Murcko and Butina uncertainty; permutation nulls; coverage strata; fixed-support same-endpoint comparisons; and a human binding Ki/Kd subset. A reusable nine-step audit and generic CLI turn the result into a concrete workflow recommendation. | Eq. 6; Fig. 5; Results; Methods; Discussion; Tables S5/S9; `analysis/spectral_audit.py` |
| Absence of significance interpreted as equivalence | Added paired Murcko and Butina cluster intervals, conservative 90% interval unions, post hoc +/-0.02 and +/-0.05 equivalence sensitivities, and minimum detectable difference. Neither main contrast is equivalent at +/-0.02; residual versus absolute also fails +/-0.05. | Abstract; Results; Methods; Table S5; evidence ledger |
| Ki/Kd, IC50 and EC50 mixed | Added a fixed-support analysis that compares target pairs only within the same endpoint type, then averages endpoints equally within ligand. Human binding Ki/Kd is shown prominently as a cleaner sensitivity. Both remain precision limited. | Results; Methods; Table S5; machine-readable assay CSV |
| DTI rank–selectivity section weakens the paper | Removed from the main narrative. All 20 arms are the primary exploratory Supplement panel; the outcome-restricted 12-arm subset is a sensitivity. | Fig. S5; Tables S6–S8 |
| Novelty insufficiently positioned | Added SePreSA, Fukunishi, reverse-docking benchmarking, normalization, external reverse screening and AutoRevDock literature. Novelty is the reusable large-panel audit, explicit estimand/ranking decomposition and operational boundary—not PCA or two-way normalization itself. | Introduction; Discussion; Table S9 |
| Public reproducibility package required | Bundled all frozen inputs, 26-file checksum manifest, pinned requirements, Dockerfile, Makefile, provenance, file-specific data licenses, exact reproduction commands, generic audit CLI and machine-readable operational/protocol tables. | Repository root |
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
