#!/usr/bin/env python3
"""Fail the build when registered manuscript numbers drift from the evidence ledger."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from make_reported_results import build


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def main() -> None:
    evidence = json.loads((RESULTS / "evidence_summary.json").read_text())
    expected_macros, expected_table, expected_registry = build(evidence)
    expected = {
        RESULTS / "evidence_macros.tex": expected_macros,
        RESULTS / "main_spectral_metrics_table.tex": expected_table,
        RESULTS / "reported_values.json": json.dumps(expected_registry, indent=2) + "\n",
    }
    errors: list[str] = []
    for path, content in expected.items():
        if not path.exists():
            errors.append(f"missing generated result artifact: {path.relative_to(ROOT)}")
        elif path.read_text() != content:
            errors.append(f"stale generated result artifact: {path.relative_to(ROOT)}")

    manuscript = (ROOT / "manuscript.tex").read_text()
    supplement = (ROOT / "supplementary_information.tex").read_text()
    if r"\input{results/evidence_macros.tex}" not in manuscript:
        errors.append("manuscript.tex does not load the generated evidence macro registry")
    stale_fragments = {
        "2.29 (simulation interval": "old calibrated-noise point estimate",
        "2.12--2.50": "old calibrated-noise interval",
        "0.531--0.656": "stale cohort-prior interval",
        "$-0.065$--0.062": "stale residual-minus-absolute upper bound",
        "$-0.022$--0.054": "stale residual-minus-column upper bound",
    }
    for fragment, label in stale_fragments.items():
        for filename, text in [("manuscript.tex", manuscript),
                               ("supplementary_information.tex", supplement)]:
            if fragment in text:
                errors.append(f"{filename} contains {label}: {fragment}")
    if errors:
        print("Reported-result verification failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        raise SystemExit(1)
    print("Reported-result registry matches the evidence ledger; no known stale values found")


if __name__ == "__main__":
    main()
