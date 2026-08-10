#!/usr/bin/env python3
"""Generate only the supplementary figures used by the JoC manuscript."""

from __future__ import annotations

from pathlib import Path

import make_figures as figures


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    figures.OUT.mkdir(parents=True, exist_ok=True)
    figures.use_paper_style()
    evidence = figures.load_json(ROOT / "results" / "evidence_summary.json")
    figures.fig_s1(evidence)
    figures.fig_s2(evidence)
    figures.fig_s7_residual_correlations(evidence)
    figures.fig_s8_spectral_estimand_sensitivities(evidence)
    figures.fig_s9_physicochemical_target_slopes(evidence)
    figures.fig_s6_dockstring_supports(evidence)
    print(f"Wrote six manuscript supplementary figures to {figures.OUT}")


if __name__ == "__main__":
    main()
