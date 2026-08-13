# Strict-public descriptor-domain specificity

Inclusive empirical quartile thresholds are applied to seven descriptor
axes; discrete ties and the resulting actual tail sizes are retained.
Matched-size random-row-disjoint controls show that every examined axis
changes the map more than finite-support variation, with the largest
changes along three collinear size-related coordinates. The analysis is
descriptive and post hoc and does not identify a causal mechanism.

Reproduce with:

```bash
.venv/bin/python analysis/public_descriptor_domain_specificity.py
.venv/bin/python -m pytest -q analysis/test_public_descriptor_domain_specificity.py
```
