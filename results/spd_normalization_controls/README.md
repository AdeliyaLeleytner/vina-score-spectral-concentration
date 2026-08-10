# SPD normalization controls

Status: **exploratory hostile control; no manuscript integration**.

This analysis tests whether the SPD result is unique to two-way centering or is
shared by simpler within-ligand normalizations. It uses the same three frozen
SPD endpoints and removes every DOCKSTRING row whose InChI connectivity block
occurs in SPD before constructing predictors.

Representations:

- `raw`: clipped Vina scores, target-column standardized;
- `standard_centered`: raw, then subtract each ligand's 12-target mean;
- `within_ligand_ordinal`: replace each ligand's 12 scores by within-row ranks,
  then target-column standardize;
- `ligand_efficiency_raw`: Vina kcal/mol divided by RDKit heavy-atom count,
  then target-column standardize;
- `ligand_efficiency_centered`: ligand efficiency followed by row centering.

| Representation | Bound ranks | Binary 10 uM | Binary 30 uM |
|---|---:|---:|---:|
| Raw | 0.149 | 0.348 | 0.180 |
| Standard centered | 0.333 | **0.395** | 0.409 |
| Within-ligand ordinal | 0.366 | 0.288 | 0.363 |
| Ligand efficiency raw | 0.285 | 0.293 | 0.264 |
| Ligand efficiency centered | **0.449** | 0.283 | **0.445** |

All four normalized alternatives show positive alignment on at least one
censor-aware endpoint, but none resolves a paired advantage over standard
centering under 10,000 target-label permutations. Ligand-efficiency centering
exceeds standard centering for bound ranks by +0.116 (paired QAP p=0.056) and
for binary 30 uM by +0.035 (p=0.327); its bound-rank advantage is positive in
all 12 target jackknives. Standard centering remains best at binary 10 uM.

**Verdict:** this is a no-go for claiming a unique benefit of two-way
centering. The positive result is broader and more useful: removing or
attenuating ligand-wide magnitude exposes target co-response geometry, while
the precise normalization should be treated as a workflow choice and audited
against censor-aware endpoints. The raw-score comparator is not the only
reasonable baseline.

Reproduce:

```bash
.venv/bin/python analysis/spd_normalization_controls.py \
  --spd /tmp/spd_activity.txt \
  --dockstring data/frozen/dockstring-dataset.tsv.gz \
  --output-dir results/spd_normalization_controls \
  --permutations 10000 \
  --seed 20260803

.venv/bin/python -m pytest -q analysis/test_spd_normalization_controls.py
```
