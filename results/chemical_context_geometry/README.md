# Chemical-context dependence of target geometry

This is a science-only, explicitly post-hoc analysis. Molecular weight was
chosen after preliminary inspection, so the reported probabilities are
calibration diagnostics, not confirmatory hypothesis tests.

## Primary result

- **DOCKSTRING-58:** low-versus-high MW threshold-group geometry Spearman changes from 0.409 on the raw surface to 0.351 after row centering; random disjoint residual supports mean = 0.990 (lower-tail Monte Carlo probability 0.0020). The correlation is 0.437 after strict chemical-group-held-out linear removal of seven ligand descriptors. MW-stratified halves with no shared chemical groups retain geometry agreement 0.994.
  Restriction-matched, chemical-group-disjoint splits reproduce the residual map within the low- and high-MW domains with mean Spearman 0.978 and 0.966, respectively; their repeated-split ranges are composition sensitivities, not population confidence intervals.
- **Docking-44:** low-versus-high MW threshold-group geometry Spearman changes from 0.227 on the raw surface to 0.205 after row centering; random disjoint residual supports mean = 0.991 (lower-tail Monte Carlo probability 0.0020). The correlation is 0.489 after strict chemical-group-held-out linear removal of seven ligand descriptors. MW-stratified halves with no shared chemical groups retain geometry agreement 0.991.
  Restriction-matched, chemical-group-disjoint splits reproduce the residual map within the low- and high-MW domains with mean Spearman 0.980 and 0.978, respectively; their repeated-split ranges are composition sensitivities, not population confidence intervals.

Across five MW bins, increasing separation in median MW tracks increasing
dissimilarity between residual target networks:

- DOCKSTRING-58: Spearman = 1.000.
- Docking-44: Spearman = 0.988.

Across six independently sampled 15,000-ligand DOCKSTRING supports, the
primary residual agreement ranges from 0.325 to 0.361.

On the reused 20-target experimental co-selectivity endpoint, the low-
MW quintile gives AUROC/AP 0.755/0.267, versus 0.659/0.176 in the high-MW quintile.
Over the same bins, residual PR rises from 19.54 to 24.04; thus higher effective dimension is not a monotone fidelity signal in
this descriptive within-dataset comparison.
This comparison is descriptive and was not prospectively specified.

## Interpretation boundary

The result supports a *conditional geometry* interpretation: the inferred
target network depends on which chemical domain probes the proteins. It does
not show that molecular weight is causal, that either domain is biologically
correct, or that target ranking for individual ligands improves. Correlated
size descriptors, scaffold composition, Vina's score form, and pocket fit can
all contribute. The primary descriptor-removal analysis uses one inherited
fixed 15,000-ligand DOCKSTRING support, supplemented by six support seeds;
Docking-44 provides cross-panel replication but differs in targets, receptors,
and chemical library.

## Reproduction

```bash
./.venv/bin/python analysis/chemical_context_geometry.py
./.venv/bin/python -m pytest -q analysis/test_chemical_context_geometry.py
```

Random disjoint repetitions: 500; chemical-group bootstrap repetitions: 300; MW-stratified group-disjoint repetitions: 200; seed: 20260809.
