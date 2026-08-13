"""Submission-facing consistency checks for the strict-public v4 package."""

from __future__ import annotations

import json
import re

import pytest
import stat
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1]
TITLE = (
    "Two-way centering reveals distributed target-relative structure and "
    "limits score-surface compression in Vina docking matrices"
)
SUBMISSION_SOURCES = (
    "manuscript_v4_draft.tex",
    "supplementary_information_v4_draft.tex",
    "results/v4_methods_draft.tex",
    "results/v4_declarations_draft.tex",
    "references_public.tex",
)


def _read(relative: str) -> str:
    return (PACKAGE / relative).read_text(encoding="utf-8")


def _expanded_abstract() -> str:
    manuscript = _read("manuscript_v4_draft.tex")
    macros = _read("results/public_core_macros.tex")
    for name, value in re.findall(
        r"\\newcommand\{\\([A-Za-z]+)\}\{([^{}]*)\}", macros
    ):
        manuscript = manuscript.replace(f"\\{name}{{}}", value)
        manuscript = manuscript.replace(f"\\{name}", value)
    return manuscript.split(r"\begin{abstract}", 1)[1].split(
        r"\end{abstract}", 1
    )[0]


def _plain_words(tex: str) -> list[str]:
    text = re.sub(r"%.*", "", tex)
    text = re.sub(r"\\textbf\{([^{}]*)\}", r"\1", text)
    text = re.sub(r"\\(?:medskip|noindent)", " ", text)
    text = re.sub(r"\\[A-Za-z]+(?:\[[^\]]*\])?", " ", text)
    text = re.sub(r"[{}$\\]", " ", text)
    return re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*", text)


@pytest.mark.skip(reason="v5 superseded this submission on 2026-08-12: README.md, PREFLIGHT.md, CITATION.cff and CLAIM.md now describe v5, which analysis/test_v5_manuscript_numbers.py guards. Retained so the v4 drafts in the tree stay documented.")
def test_title_and_release_metadata_are_synchronized() -> None:
    # v5 superseded this submission on 2026-08-12; README and PREFLIGHT now carry the
    # v5 title, which analysis/test_v5_manuscript_numbers.py guards. These assertions
    # are scoped to the v4 documents that are still in the tree.
    for document in ("RELEASE_CHECKLIST.md",):
        assert TITLE in _read(document)
    manuscript = _read("manuscript_v4_draft.tex").replace("\n", " ")
    supplement = _read("supplementary_information_v4_draft.tex").replace(
        "\n", " "
    )
    assert TITLE in re.sub(r"\s+", " ", manuscript)
    assert TITLE in re.sub(r"\s+", " ", supplement)
    citation = _read("CITATION.cff")
    assert f'title: "{TITLE}"' in citation
    assert re.search(r'^version: "4\.0\.0(?:-dev)?"$', citation, re.MULTILINE)


def test_abstract_and_scientific_contribution_meet_limits() -> None:
    abstract = _expanded_abstract()
    assert len(_plain_words(abstract)) <= 350
    contribution = abstract.split(r"\textbf{Scientific Contribution.}", 1)[1]
    sentences = [part for part in re.split(r"[.!?]+", contribution) if part.strip()]
    assert len(sentences) <= 3


def test_submission_sources_have_no_provisional_archive_text() -> None:
    forbidden = (
        "PLACEHOLDER",
        "to be inserted",
        "[ZENODO DOI]",
        "[PUBLIC REPOSITORY URL]",
        "10.5281/zenodo.21874630",
    )
    combined = "\n".join(_read(path) for path in SUBMISSION_SOURCES)
    for marker in forbidden:
        assert marker not in combined


def test_every_manual_bibliography_item_is_cited() -> None:
    body = "\n".join(_read(path) for path in SUBMISSION_SOURCES[:-1])
    cited: set[str] = set()
    for group in re.findall(r"\\cite\{([^}]*)\}", body):
        cited.update(key.strip() for key in group.split(","))
    bibliography = set(re.findall(r"\\bibitem\{([^}]+)\}", _read("references_public.tex")))
    assert cited == bibliography


def test_every_submission_path_target_exists() -> None:
    combined = "\n".join(_read(path) for path in SUBMISSION_SOURCES)
    targets = set(re.findall(r"\\path\{([^}]+)\}", combined))
    missing = sorted(path for path in targets if not (PACKAGE / path).exists())
    assert not missing


@pytest.mark.skip(reason="v5 superseded this submission on 2026-08-12: README.md, PREFLIGHT.md, CITATION.cff and CLAIM.md now describe v5, which analysis/test_v5_manuscript_numbers.py guards. Retained so the v4 drafts in the tree stay documented.")
def test_release_claim_text_matches_evidence_ledger() -> None:
    ledger = json.loads(_read("results/public_core_evidence.json"))
    group = {
        (row["dataset"], row["calibration_ligands"]): row
        for row in ledger["scaffold_holdout_recovery"]["key_results"]
    }
    random = {
        (row["dataset"], row["calibration_ligands"]): row
        for row in ledger["scaffold_holdout_recovery"]["matched_random_row_results"]
    }
    group_values = (
        f"{group[('Docking-44', 200)]['geometry_spearman_mean']:.3f}/"
        f"{group[('Docking-44', 500)]['geometry_spearman_mean']:.3f} and\n"
        f"  {group[('DOCKSTRING-58', 200)]['geometry_spearman_mean']:.3f}/"
        f"{group[('DOCKSTRING-58', 500)]['geometry_spearman_mean']:.3f}"
    )
    random_values = (
        f"{random[('Docking-44', 200)]['geometry_spearman_mean']:.3f}/"
        f"{random[('Docking-44', 500)]['geometry_spearman_mean']:.3f} and\n"
        f"  {random[('DOCKSTRING-58', 200)]['geometry_spearman_mean']:.3f}/"
        f"{random[('DOCKSTRING-58', 500)]['geometry_spearman_mean']:.3f}"
    )
    reproducibility = _read("REPRODUCIBILITY.md")
    assert group_values in reproducibility
    assert random_values in reproducibility
    for document in ("README.md", "CLAIM.md"):
        text = _read(document)
        for value in ("0.925", "0.958", "0.914"):
            assert value in text


def test_docking44_input_is_archive_readable_without_byte_drift() -> None:
    path = PACKAGE / "data/frozen/df_final_v4.csv.gz"
    assert path.stat().st_mode & stat.S_IROTH
    manifest = _read("results/public_core_source_manifest.csv")
    assert "25436e49b9fc80421d8499ef289ed8ef00b4ab35b904c7d930fc377f493e1d51" in manifest
