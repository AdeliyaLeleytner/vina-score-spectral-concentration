# Target-map audit CLI

`target_map_audit.py` is a standalone audit for a dense ligand × target numeric
matrix. It reads no manuscript ledger and imports no repository-local analysis
module. The command is intended to make the manuscript's target-map checks
reusable on a new scoring method or ligand support.

## Minimal audit

```bash
python analysis/target_map_audit.py scores.csv \
  --output-dir results/my_target_map_audit \
  --target-regex '^score_' \
  --ligand-id compound_id
```

Targets can instead be supplied in an explicit, ordered list:

```bash
python analysis/target_map_audit.py scores.tsv.gz \
  --output-dir results/my_target_map_audit \
  --target-columns EGFR,ABL1,SRC,CDK2
```

By default, any missing or nonnumeric target cell stops the audit. Missing
values are accepted only with `--imputation target-mean` or
`--imputation target-median`; malformed nonnumeric values always stop it.

## Support and recovery analyses

```bash
python analysis/target_map_audit.py scores.csv.gz \
  --output-dir results/my_target_map_audit \
  --target-regex '^target_' \
  --domain-column molecular_weight \
  --domain-quantiles 4 \
  --pilot-sizes 100,200,500 \
  --pilot-repeats 200 \
  --top-edge-count 10 \
  --panel-k 8 \
  --seed 20260810
```

Pilot maps are compared with maps estimated from their row-disjoint
complements. The reported medoids minimize `1 - |r|` on the observed score map.
They are a compression of that map, not a biologically validated target panel.

## Outputs

Every run writes:

- `preprocessing_contract.json` — source digest, selected columns, missing-data
  policy, transformations, seed, and interpretation boundaries;
- `target_correlation_raw.csv` and
  `target_correlation_two_way_centered.csv` — square target maps;
- `spectra.json` — PR dimension, its exact mean-squared-correlation identity,
  entropy dimension, PC1 fraction, k90, and the full eigenspectrum;
- `signed_edges.csv` — reconstructable upper-triangle edge table;
- `output_checksums.json` — byte counts and SHA-256 hashes for every artifact.

Domain, recovery, and medoid outputs are added only when their corresponding
options are selected. The output directory must be absent or empty, preventing
an audit from silently mixing with files from an earlier configuration.

The synthetic example used by the tests is
`analysis/fixtures/target_map_audit_synthetic.csv`.
