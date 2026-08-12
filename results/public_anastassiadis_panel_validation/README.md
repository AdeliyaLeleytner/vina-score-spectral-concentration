# Public Anastassiadis panel validation

This post-hoc external extension reads the official Anastassiadis et al. 2011
Supplementary Table 3 by local `--source` path after an exact SHA-256 check. The
source XLS is **not redistributed** in this result bundle.

The primary block is 176 ligands by the fixed 21 kinases. Its endpoint is
`100 - percent remaining activity`, without clipping. CDK2 is the arithmetic
mean of its cyclin-A and cyclin-E assay rows. The only removed compounds are
`SB 202474` (missing AKT1) and `VEGF Receptor 2 Kinase Inhibitor II` (missing
MAPK14). `target_mapping.csv` and `excluded_ligands.csv` make those decisions
machine-readable.

The primary Vina reference is the conservatively triple-de-leaked
259,641-row
support. Raw and fixed-21-row-centred Vina selectors are exact C(21,k) optima at
k=[4, 6, 8, 10, 12]. Both selectors are evaluated on the same experimental map.
For the primary signed 1-r endpoint, the mean lexicographic raw-minus-residual
loss over k is 0.0961, and the
mean conservative best-raw-minus-worst-residual contrast is
0.0722.
Positive values favour the residual selector.

`same_endpoint_panel_transfer.csv` reports exact oracle loss, exact panel
percentile, sequence-identity baseline and every tie sensitivity. The inferential
unit is the complete target label: 20,000 shared target-label permutations are
used with max-over-three-objective adjustment. `joint_cross_assay_qap.csv`
reports per-panel and joint DAVIS/PKIS2/Anastassiadis normalized AUCs over k.

`mapping_sensitivities.csv` crosses CDK2 mean/A-only/E-only, complete-case versus
target-median imputation, unclipped versus [0,100]-clipped activity, and raw,
row-centred and column-rank-then-row-centred experimental maps. Absolute and
squared objectives are sensitivities, not confirmatory substitutes.

`target_leave_one_out_panel_transfer.csv` repeats exact panel selection after
removing each kinase in turn; `target_leave_one_out_map_concordance.csv` repeats
the three cross-assay comparisons on the corresponding 20-target maps. These
are fixed-panel influence checks, not target-superpopulation intervals.

Ligand split halves and the ordinary ligand bootstrap are descriptive map
reliability diagnostics only. They are not chemical-cluster-aware because the
publisher workbook supplies compound names and CAS numbers but no structures.
The separate PubChem identity crosswalk enables a secondary cluster multiplier;
unresolved records are singleton clusters. This analysis is conditional on the
fixed 21 kinase identities and does not show transfer to unseen targets.
