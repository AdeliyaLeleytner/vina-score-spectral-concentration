# Historical Vina self-reconstruction experiment — do not cite

This directory contains an early exploratory column-pivoted-QR experiment that
reconstructs Vina scores from a subset of Vina target columns.  It is **not a
validated positive result** and is superseded by
`results/experimental_sentinel_panel/`.

The direct-QR selector is unsuitable as a scientific target-panel rule after
column standardization: every target column has equal norm, so the first pivot
is arbitrary and the chosen set is support/numerics-sensitive.  The code now
contains fair random-plus-descriptor and family-diverse controls for audit
purposes, but the historical CSV files in this directory predate the final
experimental-DEIM analysis and must not be used for claims.

The replacement uses deterministic DEIM sensor placement on leading residual
eigenmodes, exact-ligand exclusion, scaffold-support sensitivity, 2,000 random
target panels, and scaffold-held-out reconstruction of PKIS2, PKIS1, and DAVIS
experimental values.
