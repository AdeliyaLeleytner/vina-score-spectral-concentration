# Experimental chemical-context geometry

This is a science-only, post-hoc analysis on the exact 20-target support shared
by the complete PKIS2 (645 ligands), PKIS1 (360 ligands), and DAVIS (72 ligands)
panels.  It tests whether two-way-centered experimental target-correlation
geometry is stable between the lowest and highest molecular-weight quartiles.
It does **not** test ligand-level target ranking and does not establish molecular
weight as a cause.

## Main result

- **PKIS2:** low--high residual-geometry Spearman 0.333; random-disjoint mean 0.845, lower-tail Monte Carlo p=0.000999; target-jackknife range 0.271--0.386.
- **PKIS1:** low--high residual-geometry Spearman 0.547; random-disjoint mean 0.833, lower-tail Monte Carlo p=0.000999; target-jackknife range 0.479--0.673.
- **DAVIS:** low--high residual-geometry Spearman 0.531; random-disjoint mean 0.619, lower-tail Monte Carlo p=0.1728; target-jackknife range 0.485--0.605.

The random-disjoint null uses two disjoint ligand samples with exactly the
observed quartile sizes.  Murcko and Morgan-radius-2, 2048-bit Butina clustering
(Tanimoto threshold 0.65) are reported as separate cluster-bootstrap
sensitivities.  A five-bin analysis asks whether target-network dissimilarity
increases with separation in median MW.  DAVIS is explicitly low power: its
quartiles contain only 18 ligands each.

## Does the direction replicate across experimental panels?

- **PKIS2 versus PKIS1:** delta-network Spearman 0.060, QAP p=0.1825, Holm p=0.365.
- **PKIS2 versus DAVIS:** delta-network Spearman 0.169, QAP p=0.0117, Holm p=0.0351.
- **PKIS1 versus DAVIS:** delta-network Spearman -0.001, QAP p=0.4902, Holm p=0.4902.

The property of chemistry-dependent geometry replicates in the two larger
panels, but a single universal direction of MW-associated rewiring does not:
only PKIS2--DAVIS aligns after the three-pair exploratory correction.  The
panels have different compounds and assay modalities, so this is a target-label
comparison rather than an exact-ligand replication.

## Exact matched-ligand DOCKSTRING check

- **PKIS2 (154 exact matched ligands):** experimental-versus-DOCKSTRING low-to-high delta-network Spearman 0.065, target-label QAP p=0.1789.
- **DAVIS (59 exact matched ligands):** experimental-versus-DOCKSTRING low-to-high delta-network Spearman -0.084, target-label QAP p=0.8819.

This matched check compares **directions of network change**, not the absolute
correlation networks.  It is exploratory and is not promoted when QAP does not
support positive alignment.  PKIS1 is absent because there is no pre-frozen,
audited exact-ligand DOCKSTRING mapping in this package; creating a new mapping
after inspecting outcomes would add another post-hoc degree of freedom.

## Reproduction

```bash
python analysis/experimental_chemical_context_geometry.py \
  --pkis1-zip /tmp/pkis1_supplement.zip \
  --output-dir results/experimental_chemical_context_geometry
```

The PKIS1 archive must match SHA-256 `1ffbe7fd0b4fc1ef72f2434a2b247d15c0f2b4d3c34668f112f4b4c9e4ead7dc`.  Raw source
files are never modified.  `summary.json` records parameters, source hashes,
and limitations; CSV files contain the null draws, cluster-bootstrap draws,
quintile summaries, target jackknife, and matched-support comparison.
