#!/usr/bin/env python3
"""Fail the build when registered manuscript numbers drift from the evidence ledger."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from make_reported_results import build


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
MANUSCRIPT_TITLE = (
    "Target-correlation maps are library-conditional but recoverable from a few "
    "hundred ligands in two large Vina panels"
)


def normalize_tex_text(value: str) -> str:
    """Collapse TeX/Markdown line wrapping for metadata-level comparisons."""

    value = value.replace("\\\\", " ")
    value = re.sub(r"\\(?:large|textbf)\s*", "", value)
    value = value.replace("{", "").replace("}", "")
    value = value.replace("**", "")
    return " ".join(value.split())


def main() -> None:
    manuscript_evidence = json.loads(
        (RESULTS / "manuscript_evidence.json").read_text()
    )
    broad_ranking = json.loads(
        (RESULTS / "dockstring_chembl_ranking/summary.json").read_text()
    )
    expected_macros, expected_table, expected_registry = build(manuscript_evidence)
    expected = {
        RESULTS / "evidence_macros.tex": expected_macros,
        RESULTS / "main_spectral_metrics_table.tex": expected_table,
        RESULTS / "reported_values.json": json.dumps(expected_registry, indent=2) + "\n",
    }
    errors: list[str] = []
    ranking_boundary = broad_ranking.get("claim_boundary", "")
    if "30 of DOCKSTRING's 58" in ranking_boundary or "20 kinases plus 10" in ranking_boundary:
        errors.append(
            "dockstring_chembl_ranking summary contains stale 30/58 or 20+10 support text"
        )
    if broad_ranking.get("support", {}).get("targets") != 31:
        errors.append("dockstring_chembl_ranking support is not the frozen 31-target panel")
    for path, content in expected.items():
        if not path.exists():
            errors.append(f"missing generated result artifact: {path.relative_to(ROOT)}")
        elif path.read_text() != content:
            errors.append(f"stale generated result artifact: {path.relative_to(ROOT)}")

    manuscript = (ROOT / "manuscript.tex").read_text()
    supplement = (ROOT / "supplementary_information.tex").read_text()
    supplement_table_path = RESULTS / "supplement_tables.tex"
    if not supplement_table_path.exists():
        errors.append("missing generated result artifact: results/supplement_tables.tex")
        supplement_tables = ""
    else:
        supplement_tables = supplement_table_path.read_text()
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
    macro_names = expected_registry["macros"]
    dense_interval_macro_pairs = (
        ("DenseDavisScalePenaltyLow", "DenseDavisScalePenaltyHigh"),
        ("DenseDavisBinderDeltaLow", "DenseDavisBinderDeltaHigh"),
        ("DensePkisTwoBothActiveDeltaLow", "DensePkisTwoBothActiveDeltaHigh"),
    )
    for low_name, high_name in dense_interval_macro_pairs:
        expected_interval = (
            f"{macro_names[low_name]}~--~{macro_names[high_name]}"
        )
        if expected_interval not in supplement_tables:
            errors.append(
                "supplement_tables.tex does not contain the registered dense "
                f"cluster-union interval {low_name}/{high_name}: {expected_interval}"
            )
    for filename, source in (
        ("manuscript.tex", manuscript),
        ("supplementary_information.tex", supplement),
    ):
        for name in macro_names:
            if f"\\\\{name}" in source:
                errors.append(
                    f"{filename} contains a doubled-slash evidence macro: {name}"
                )
        for pattern, label in (
            (r"\[PUBLIC REPOSITORY URL\]", "repository placeholder"),
            (r"\[ZENODO DOI\]", "Zenodo placeholder"),
            (r"\b(?:TODO|TBD)\b", "unfinished text marker"),
        ):
            if re.search(pattern, source, flags=re.IGNORECASE):
                errors.append(f"{filename} contains {label}")
    # The current Journal of Cheminformatics Research Article guideline requires
    # the exact abstract subheading "Scientific Contribution" and at most three
    # sentences (checked against the official Springer page on 2026-08-10).
    if r"\textbf{Scientific Contribution.}" not in manuscript:
        errors.append(
            "manuscript.tex lacks the exact required Scientific Contribution heading"
        )
    contribution_match = re.search(
        r"\\textbf\{Scientific Contribution\.\}(.*?)\\end\{abstract\}",
        manuscript,
        flags=re.DOTALL,
    )
    if contribution_match is not None:
        contribution = re.sub(
            r"\\[A-Za-z]+(?:\{[^}]*\})?", " ", contribution_match.group(1)
        )
        sentence_count = len(re.findall(r"[.!?](?=\s|$)", contribution))
        if sentence_count > 3:
            errors.append(
                "Contribution statement exceeds the three-sentence "
                f"journal limit ({sentence_count} sentences)"
            )

    abstract_match = re.search(
        r"\\begin\{abstract\}(.*?)\\end\{abstract\}", manuscript, flags=re.DOTALL
    )
    if abstract_match is None:
        errors.append("manuscript.tex lacks an abstract environment")
    else:
        abstract = abstract_match.group(1)
        # Treat every evidence macro as one printed value; remove formatting commands.
        abstract = re.sub(r"\\[A-Za-z]+\{\}", " value ", abstract)
        abstract = re.sub(r"\\(?:textbf|medskip|noindent)\b", " ", abstract)
        abstract = abstract.replace("{", " ").replace("}", " ")
        abstract_words = re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*", abstract)
        if len(abstract_words) > 350:
            errors.append(
                f"abstract exceeds the 350-word limit ({len(abstract_words)} words)"
            )

    metadata_sources = {
        "manuscript.tex": manuscript,
        "supplementary_information.tex": supplement,
        "README.md": (ROOT / "README.md").read_text(),
        "CITATION.cff": (ROOT / "CITATION.cff").read_text(),
    }
    for filename, source in metadata_sources.items():
        if MANUSCRIPT_TITLE not in normalize_tex_text(source):
            errors.append(f"{filename} is not synchronized to the canonical manuscript title")
    if errors:
        print("Reported-result verification failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        raise SystemExit(1)
    print("Reported-result registry matches the evidence ledger; no known stale values found")


if __name__ == "__main__":
    main()
