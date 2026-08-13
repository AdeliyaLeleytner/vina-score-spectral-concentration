# k=8 target-panel robustness controls

This compact artifact adds two controls to `public_panel_selector_controls`:
designed-versus-random comparisons on the same common omitted targets, and a
secondary direct-residual training sensitivity.  It reuses the exact five
chemical-group folds, 500-ligand pilots, k=8 panels and first three fixed random
panels from the checksummed source artifact.

The common-omitted analysis contains no copied selected truths.  The direct
residual analysis uses selected raw scores as its only predictors and is not a
replacement for the primary raw-score deployment model.
