# Public panel-selector and panel-size controls

This fail-closed artifact compares a residual-map exact k-medoids selector with
a reconstruction-aware pivoted-QR selector, 50 fixed random panels per split,
a selected-target PC1 predictor, and a seven-descriptor ridge baseline.  Five
chemical-group folds are evaluated at every reported panel size.  The primary
endpoint is a common all-target profile; omitted-target endpoints are secondary
because different panels omit different targets.

The pilot and held-out evaluation rows are chemically group-disjoint and fully
observed.  Only pilot rows determine preprocessing, selection and fitted models.
Residual profiles are formed by row-centring raw scores before pilot residual
column scaling.

Reproduce with:

```bash
.venv/bin/python analysis/public_panel_selector_controls.py --overwrite
.venv/bin/python -m pytest -q analysis/test_public_panel_selector_controls.py
```
