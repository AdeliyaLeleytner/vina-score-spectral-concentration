# Vina panel calibration-size transfer

This strict-public analysis selects an exact residual-signed k=8 Vina target
panel from random calibration supports of 200, 500, 2000, 5000, 15000, 50000
ligands and evaluates the resulting decision on fixed DAVIS and PKIS2 maps.
Selection is outcome-blind: no experimental value is used until evaluation.

The same-endpoint control compares raw- and residual-Vina panel selectors over
k=[4, 6, 8, 10, 12] while evaluating both selectors on the same residual
experimental map. Shared target-label permutations and positive Exp(1)
chemical-cluster multipliers quantify target-label and chemical-support
uncertainty, respectively.

All 203,490 possible eight-target panels are enumerated.
Results are conditional on the fixed 21 targets and do not validate affinity or
compound ranking.

```bash
python analysis/public_vina_panel_calibration_transfer.py
pytest -q analysis/test_public_vina_panel_calibration_transfer.py
```
