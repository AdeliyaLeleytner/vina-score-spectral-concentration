# Replicated co-selective target-pair retrieval

This exploratory analysis defines one fixed experimental endpoint on the 20 kinases
shared by DAVIS, PKIS2, and PKIS1. A pair is positive when its target correlation is
in the upper 10% after two-way centering in at least two of the three experimental
panels. Raw and centered DOCKSTRING geometries predict the same 15 positive pairs.
Only docking target labels are permuted in paired QAP inference.

The centered geometry has AUROC 0.710 and average precision 0.301, versus 0.597 and
0.236 for raw docking. Paired target-label QAP probabilities are 0.033 and 0.017;
Holm adjustment across the two primary metrics gives 0.033 and 0.033. The gain is
positive after deleting any one target and on five independent 15,000-ligand
DOCKSTRING supports. The declared 5--25% threshold-family omnibus is only borderline
(0.061 and 0.076), so the result is an upper-tail finding rather than evidence of a
universal improvement at every co-selectivity threshold.

On this same 20-target surface, correlation PR increases from 1.805 to 11.616.
The full centered geometry exceeds its leading-mode-only reconstruction by 0.071
AUROC and 0.127 average precision (paired QAP 0.043 and 0.0023). The complete
unselected k=1,...,20 curve is in `cumulative_mode_retrieval.csv`. Continuous
experimental-geometry concordance improves only slightly and not monotonically
beyond the leading mode; the clearest multi-mode consequence is upper-tail pair
retrieval.

Receptor-domain sequence identity is a deliberately strong control. It has AUROC
0.718 and average precision 0.426, so centered docking does not beat this standalone
baseline. Centered docking nevertheless retains a partial rank association after
linear rank control for sequence identity (partial r=0.167; target-QAP p=0.0188).
Within the fixed <40% identity stratum (183 pairs, 12 positives), centered docking
improves AUROC by 0.126 and average precision by 0.074 over raw docking (paired QAP
0.035 and 0.011). This supports a nonredundant signal but not superiority to sequence.

The endpoint concerns co-selective pair/panel-redundancy recovery. It is not a test
of ligand-level target retrieval, pose quality, or universal affinity accuracy, and
the analyses remain post hoc until replicated on a new experimental panel.

## Reproduction

Obtain the checksum-validated PKIS1 supplement as documented by
`analysis/fetch_pkis1_supplement.py`, and obtain the DOCKSTRING source tree containing
the released receptor PDBQT files. Then run:

```bash
python analysis/replicated_pair_retrieval.py \
  --pkis1-zip /tmp/pkis1_supplement.zip \
  --dockstring-source-root /path/to/dockstring \
  --output-dir results/replicated_pair_retrieval \
  --qap-permutations 50000 \
  --reference-support-size 15000 \
  --reference-support-seeds 11,29,47,71,97
```

`summary.json` records all source checksums, exact alignment parameters, seeds,
estimands, point estimates, null results, and claim boundaries.
