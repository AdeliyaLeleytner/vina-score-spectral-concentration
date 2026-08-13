# Public sequence-geometry controls

This artifact compares raw and raw-row-centred DOCKSTRING target-correlation
maps with global sequence identity for all 58 receptors and for the common
21-kinase subset. Because row centring depends on the target panel, the kinase
analysis is reported both after centring within all 58 targets and after
restriction followed by centring within the 21 kinases. Shared target-label QAP
tests raw, residual and their paired difference. Family-specific edge summaries
are descriptive only.

Reproduce with:

```bash
.venv/bin/python analysis/public_sequence_geometry_controls.py --overwrite
.venv/bin/pytest -q analysis/test_public_sequence_geometry_controls.py
```
