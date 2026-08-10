# Kinobeads library information-diversity audit

Exploratory, fixed-collection analysis; no manuscript files were edited.

## Main result

On the primary common support (26 direct kinase targets), residual correlation PR was 15.70 for KCGS, 13.90 for exclusive PKIS, 13.39 for exclusive PKIS2, and 11.25 for Roche.

After a configuration null preserving every compound hit count and every target marginal, observed/null residual-PR ratios were 0.759, 0.653, 0.617, and 0.586, respectively. A value closer to one means less co-target redundancy than expected from sparsity and marginals alone.

Using every classifier-called direct kinase interaction rather than only submicromolar calls gave the same ordering on a larger 56-target common support: 27.32, 22.18, 23.35, and 17.33.

Among compounds with the same positive hit count and from different annotated chemotypes, exact target-profile collisions were 3.8x, 4.0x, and 7.1x more frequent than in KCGS after standardization to the KCGS hit-count distribution.

## Essential boundary

Inverse-probability matching of per-compound hit counts nearly removes the KCGS residual-correlation-PR difference versus PKIS and PKIS2. Thus raw PR alone is not a pure information-diversity score. The more defensible positive result is configuration-normalized redundancy and degree-conditioned profile uniqueness.

Chemotype-cluster bootstrap differences (KCGS minus comparator) were:

- KCGS_minus_PKIS_exclusive: median 1.96, 95% range [-1.47, 5.29].
- KCGS_minus_PKIS2_exclusive: median 1.97, 95% range [-0.75, 4.41].
- KCGS_minus_Roche: median 4.93, 95% range [1.85, 7.91].

The PKIS and PKIS2 cluster-bootstrap ranges include zero; this is reported rather than hidden.
Bootstrap draws are conditional on retaining nonzero variance in all 26 fixed target columns; rejected degenerate-draw counts are recorded in chemotype_bootstrap_summary.csv.

## Data semantics and provenance

All 1,183 compounds were profiled at 100 nM and 1 uM in the same Kinobeads workflow. A blank workbook cell is interpreted only as no classifier-called target in that assay. The primary binary matrix uses direct protein/lipid-kinase calls below 1,000 nM. Inputs are the open supplementary Tables S1 and S2 from Reinecke et al., Nature Chemical Biology (2024), DOI 10.1038/s41589-023-01459-3; exact SHA-256 values are recorded in summary.json.

## Reproduction

```bash
python analysis/kinobeads_library_information_diversity.py \
  --table1 /path/to/41589_2023_1459_MOESM2_ESM.xlsx \
  --table2 /path/to/41589_2023_1459_MOESM3_ESM.xlsx \
  --output-dir results/kinobeads_library_information_diversity
```
