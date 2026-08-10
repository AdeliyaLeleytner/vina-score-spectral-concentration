# Exploratory KLIFS pocket controls

This directory contains a science-only, post-hoc structural-control analysis for
the fixed 20-kinase target-pair endpoint released in
`results/replicated_pair_retrieval`. It does **not** redefine the endpoint or
select a new co-selectivity threshold. The predictor is the previously released
centered Vina target-correlation geometry.

## Estimand and inference

The primary estimand is the rank-linear partial association between centered
Vina target-pair geometry and experimental target-pair geometry after controlling
for one or more of:

- global identity of the exact DOCKSTRING receptor-domain sequences;
- identity across the aligned 85-residue KLIFS kinase pocket;
- same KLIFS family;
- same KLIFS group.

Target-label QAP permutes the complete Vina predictor network while leaving each
experimental endpoint and structural-control network fixed. The 50,000
permutations use seed `20260803`. Pairwise ranks are residualized by ordinary
least squares with an intercept. QAP probabilities are one-sided for a positive
association and use the plus-one correction.

The old-three-panel endpoint is the mean of the fixed DAVIS, PKIS2, and PKIS1
centered pair percentile ranks. KiRHub is an independently sourced experimental
geometry. No KiRHub activity rows or pair-level KiRHub values are redistributed.

## Main aggregate results

- Centered Vina versus the old-three-panel mean remains associated after KLIFS
  pocket control: partial Spearman `0.2630`, target-label QAP `p=0.00872`.
- It remains associated after joint control for receptor sequence, KLIFS pocket,
  family, and group: `0.2216`, `p=0.01458`.
- Against independent KiRHub geometry the corresponding estimates are `0.3153`,
  `p=0.00416`, and `0.2856`, `p=0.00614`.
- Centered Vina geometry is not itself resolved as a simple proxy for KLIFS pocket
  identity (`rho=0.0669`, `p=0.1746`), family (`rho=0.1123`, `p=0.0637`), or
  group (`rho=0.0307`, `p=0.2656`).

The structural strata were formulated after inspecting the main result and are
more strongly post hoc. After excluding all four same-KLIFS-family pairs, 186
pairs and 12 endpoint positives remain. Centered versus raw Vina has AUROC
`0.6624` versus `0.5441` (paired gain `p=0.04372`) and average precision `0.2182`
versus `0.1602` (`p=0.02330`). These nominal probabilities are descriptive and
unadjusted for the exploratory search.

## Interpretation boundary

The result supports a transferable, chemistry-induced target co-response
geometry that is not exhausted by the tested sequence and KLIFS annotations. It
does not establish pose accuracy, causal pocket determinants, superiority to
sequence identity as a standalone predictor, or a universal benefit of
centering. The magnitude or change of participation-ratio dimension is not
evaluated here as a docking-quality score.

KLIFS pocket identity is a sequence control, not a matched 3D comparison of the
exact receptor conformations used by DOCKSTRING. A confirmatory extension would
predeclare these controls and test matched structural-pocket similarity on a new
experimental panel.

## Sources and reproduction

KLIFS defines an aligned 85-residue kinase binding site and states that its data
are freely available/open. Cite the KLIFS database paper, DOI
`10.1093/nar/gkaa895`. The complete KLIFS response is not redistributed here.
The exact API snapshot used on 2026-08-03 has SHA256
`041a159e662a27696554867acbe1ac7fb35d36f71483d42ab4ac90e219dbef8f`:

```bash
curl -L --fail \
  'https://klifs.net/api_v2/kinase_information?species=HUMAN' \
  -o /tmp/klifs_human_20260803.json
```

Because KLIFS is a live database, a later response may have a different checksum.
The frozen analysis fails closed rather than silently accepting changed
annotations.

Obtain the checksum-validated KiRHub supplement using the existing fetch helper,
then run:

```bash
python analysis/fetch_kirhub_supplement.py /tmp/kirhub_supp_tables.xlsx
python analysis/klifs_pocket_control.py \
  --klifs-json /tmp/klifs_human_20260803.json \
  --kirhub-workbook /tmp/kirhub_supp_tables.xlsx \
  --output-dir results/klifs_pocket_control \
  --qap-permutations 50000 \
  --seed 20260803
python -m pytest -q analysis/test_klifs_pocket_control.py
```

`summary.json` records source checksums, software versions, output checksums,
estimands, seeds, and claim boundaries. The CSV files contain only target-level,
target-pair, or aggregate results.

## Output files

- `summary.json`: machine-readable claims, provenance, configuration, and key
  estimates.
- `target_annotations.csv`: selected KLIFS identifiers and annotation metadata;
  pocket sequences are represented only by SHA256 hashes.
- `target_pairs.csv`: fixed endpoint/docking pair quantities plus derived KLIFS
  pair annotations; no KiRHub pair values.
- `partial_geometry_qap.csv`: all endpoint-by-control partial-QAP results.
- `annotation_associations.csv`: centered Vina association with the structural
  annotations themselves.
- `locked_endpoint_retrieval.csv`: structural and docking baselines for the
  unchanged 15-pair endpoint.
- `stratified_retrieval.csv`: explicitly post-hoc paired raw-versus-centered
  sensitivities.
