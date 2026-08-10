# Calibration sample-efficiency audit

## Outcome

**GO_small_diverse_calibration_panel**.  The smallest tested sample size satisfying the
declared all-metric robustness rule was **80 compounds**.

The experiment asks a practical question: how many compounds must be profiled
densely across the same 20 kinases before their marginal hit rates become a
useful prior for predicting which target pairs will be co-selected in other
chemical libraries?  Every fit uses only the sampled PKIS1 compounds.  DAVIS,
PKIS2, and KiRHub are locked evaluations.

For 1,000 Butina-diversified 80-compound panels, median mean locked-panel gains
over sequence plus KLIFS pocket identity were
+0.196 Spearman,
+0.122 top-10% AUROC, and
+0.114 top-10% AP.
Their 2.5th percentiles were respectively
+0.048,
+0.066, and
+0.052; the fractions
improving all three locked panels were
0.985,
0.994, and
0.991.

A structure-only deterministic Morgan-MaxMin 80-compound panel gave mean gains
of +0.263,
+0.150, and
+0.126.  Correct
target alignment was resolved by target-label QAP after Holm adjustment
(`practical_80_target_label_qap.csv`), and every metric remained positive in
all 60 panel-by-target jackknife cells.

## Operational meaning

For a fixed target panel, an **80-compound chemically diverse calibration
screen** can estimate target breadth well enough to improve predictions of
future target-pair co-selectivity over protein sequence and pocket similarity
alone.  This is an assay-panel calibration result: it can inform how much
up-front multi-target profiling is needed before predicting likely shared
off-targets or counterscreens in a new compound collection.

## Guardrails

- Compound selection is outcome-blind.  Uniform and Butina-round-robin samples
  use reproducible seeds; the practical panel is selected from structures alone
  by Morgan-r2/2048 MaxMin.
- Baseline and augmented coefficients are refit on exactly the same sampled
  PKIS1 endpoint.  No full-PKIS1 activity enters panel selection or fitting.
- The 50% inhibition degree is primary.  Thresholds 20/35/65/80% and continuous
  target means are sensitivities in
  `practical_80_prior_definition_sensitivity.csv`.
- This is post-hoc and conditional on 20 kinases.  It is not evidence for
  ligand-level target ranking, unseen targets, or other protein families.

## Files

- `summary.json`: machine-readable decision and headline estimates.
- `sampling_summary.csv`: repeated-sampling distributions by n and scheme.
- `repeated_subsample_metrics.csv`: all locked-panel deltas.
- `repeated_degree_recovery.csv` and `degree_recovery_summary.csv`: recovery of
  full-PKIS1 target degrees.
- `sampling_scheme_comparison.csv`: diversity-minus-random contrasts.
- `deterministic_MaxMin_manifest.csv`: exact structure-only nested panel order.
- `deterministic_MaxMin_metrics.csv`: locked metrics for each MaxMin prefix.
- `practical_80_target_label_qap.csv`: aligned-degree target-label null.
- `practical_80_target_jackknife.csv`: target-deletion robustness.
