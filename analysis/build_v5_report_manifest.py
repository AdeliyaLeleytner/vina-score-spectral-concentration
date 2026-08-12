#!/usr/bin/env python3
"""Build or verify the v5 *report-level* source manifest.

This manifest answers one deliberately narrow question: which local files are
consumed when the two v5 PDFs and their submission artwork are rebuilt from the
frozen result bundles?  It does **not** claim that the upstream scientific
analyses were rerun.  Source-level refreshes are a separate, more expensive
workflow.

The file list is discovered from three contracts rather than copied by hand:

* the recursive ``\\input`` graph of ``manuscript_v5.tex`` and
  ``supplementary_v5.tex``;
* path-valued reads in the v5 figure drivers, report tests and released-
  aggregate PDSP reconstruction;
* the small explicit build/environment contract required to run those steps.

Generated PDFs, PNGs, LaTeX auxiliaries and the manifest itself are never rows
in the source manifest.  Submission artwork is instead checked as a derived
product: the staged PDFs must be byte-identical to their regenerated figure
counterparts, and the journal-sized graphical abstract must be the exact resize
specified by the Makefile.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import io
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
MANIFEST = RESULTS / "v5_report_source_manifest.csv"
DIGEST = RESULTS / "v5_report_source_manifest.sha256"

TEX_ROOTS = (ROOT / "manuscript_v5.tex", ROOT / "supplementary_v5.tex")
PYTHON_CONSUMERS = (
    ROOT / "analysis" / "make_v5_figures.py",
    ROOT / "analysis" / "make_v5_graphical_abstract.py",
    ROOT / "analysis" / "reproduce_pdsp_derived_qap.py",
    ROOT / "analysis" / "test_reproduce_pdsp_derived_qap.py",
    ROOT / "analysis" / "test_v5_manuscript_numbers.py",
    ROOT / "analysis" / "test_v5_supplement_numbers.py",
)
# The public PDSP reconstruction imports the scientific functions below as a
# library.  It is therefore executable report code in this workflow, even
# though the same module can also perform the separate source-level analysis.
REPORT_SUPPORT_CODE = (ROOT / "analysis" / "pdsp_counterscreen_retrieval.py",)

# These dependencies are passed through function defaults/module attributes,
# rather than as literal path expressions at the eventual read_csv() calls.
# Keeping this short exception table beside the consumer inventory is clearer
# and safer than attempting general Python data-flow analysis.
EXPLICIT_CONSUMER_INPUTS = {
    ROOT / "analysis" / "reproduce_pdsp_derived_qap.py": (
        ROOT / "analysis" / "pdsp_counterscreen_retrieval.py",
        ROOT / "data" / "frozen" / "df_final_v4.csv.gz",
        ROOT / "results" / "pdsp_counterscreen_retrieval"
        / "excluded_docking44_ligand_ids.csv",
        ROOT / "results" / "pdsp_counterscreen_retrieval" / "target_pairs.csv",
        ROOT / "results" / "pdsp_counterscreen_retrieval" / "paired_qap.csv",
    ),
    ROOT / "analysis" / "test_reproduce_pdsp_derived_qap.py": (
        ROOT / "analysis" / "reproduce_pdsp_derived_qap.py",
        ROOT / "results" / "pdsp_counterscreen_retrieval"
        / "excluded_docking44_ligand_ids.csv",
        ROOT / "results" / "pdsp_counterscreen_retrieval" / "target_pairs.csv",
    ),
}
BUILD_CONTRACT = (
    ROOT / "Makefile",
    ROOT / ".github" / "workflows" / "v5-report.yml",
    ROOT / "environment.yml",
    ROOT / "requirements.lock",
    ROOT / "analysis" / "build_v5_report_manifest.py",
    ROOT / "analysis" / "test_v5_report_manifest.py",
)

# The old public-core manifest is itself read by the v5 Supplement test.  Its
# sidecar is included and checked as part of that frozen-result integrity
# contract even though pytest does not open the sidecar directly.
INTEGRITY_PAIRS = (
    (
        RESULTS / "public_core_source_manifest.csv",
        RESULTS / "public_core_source_manifest.sha256",
    ),
)

REPORT_SCOPE = "report_rebuild_from_frozen_results"
MANIFEST_FIELDS = ("path", "kind", "scope", "consumers", "bytes", "sha256")
FORBIDDEN_SOURCE_SUFFIXES = {
    ".aux",
    ".log",
    ".out",
    ".pdf",
    ".png",
    ".toc",
}

FIGURE_STAGE_MAP = {
    "Fig1.pdf": "fig1_shared_axis.pdf",
    "Fig2.pdf": "fig2_residual_structure.pdf",
    "Fig3.pdf": "fig3_external_agreement.pdf",
    "Fig4.pdf": "fig4_cost_and_core.pdf",
    "FigS1.pdf": "fig_pocket_volume.pdf",
    "GraphicalAbstract.pdf": "graphical_abstract.pdf",
}
GRAPHICAL_STAGE_NAME = "GraphicalAbstract_920x300.png"
EXPECTED_TEX_GRAPHICS = {
    "figures/v5/fig1_shared_axis.pdf",
    "figures/v5/fig2_residual_structure.pdf",
    "figures/v5/fig3_external_agreement.pdf",
    "figures/v5/fig4_cost_and_core.pdf",
    "figures/v5/fig_pocket_volume.pdf",
}

INPUT_RE = re.compile(r"\\input\{([^}]+)\}")
GRAPHICS_RE = re.compile(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}")


class ManifestError(RuntimeError):
    """A missing, unsafe or stale report input."""


@dataclass(frozen=True)
class ManifestRow:
    path: str
    kind: str
    scope: str
    consumers: str
    bytes: str
    sha256: str

    def as_dict(self) -> dict[str, str]:
        return {field: getattr(self, field) for field in MANIFEST_FIELDS}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_digest_pair(payload: Path, sidecar: Path) -> None:
    """Fail unless the first digest in *sidecar* authenticates *payload*."""
    payload = Path(payload)
    sidecar = Path(sidecar)
    payload_name = payload.as_posix()
    sidecar_name = sidecar.as_posix()
    if not payload.is_file():
        raise ManifestError(f"integrity payload is missing: {payload_name}")
    if not sidecar.is_file():
        raise ManifestError(f"integrity sidecar is missing: {sidecar_name}")
    try:
        recorded = sidecar.read_text(encoding="ascii").split()[0]
    except IndexError as error:
        raise ManifestError(f"integrity sidecar is empty: {sidecar_name}") from error
    actual = sha256(payload)
    if recorded != actual:
        raise ManifestError(
            f"integrity sidecar is stale for {payload_name}: "
            f"recorded {recorded}, actual {actual} ({sidecar_name})"
        )


def relative_file(path: Path) -> tuple[Path, str]:
    """Resolve *path* as a safe, existing regular file below the repository."""
    try:
        resolved = path.resolve(strict=True)
    except FileNotFoundError as error:
        raise ManifestError(f"required report input is missing: {path}") from error
    try:
        relative = resolved.relative_to(ROOT.resolve())
    except ValueError as error:
        raise ManifestError(f"report input escapes the repository: {resolved}") from error
    if not resolved.is_file():
        raise ManifestError(f"report input is not a regular file: {relative}")
    if resolved.suffix.lower() in FORBIDDEN_SOURCE_SUFFIXES:
        raise ManifestError(f"generated artifact cannot be a source-manifest row: {relative}")
    if resolved in {MANIFEST.resolve(), DIGEST.resolve()}:
        raise ManifestError(f"source manifest cannot contain itself: {relative}")
    return resolved, relative.as_posix()


def tex_inputs() -> tuple[set[Path], set[str]]:
    """Return the recursive TeX source graph and its generated figure references."""
    pending = list(TEX_ROOTS)
    seen: set[Path] = set()
    graphics: set[str] = set()
    while pending:
        source, _ = relative_file(pending.pop())
        if source in seen:
            continue
        seen.add(source)
        text = source.read_text(encoding="utf-8")
        for name in INPUT_RE.findall(text):
            child = ROOT / name
            if child.suffix == "":
                child = child.with_suffix(".tex")
            pending.append(child)
        graphics.update(GRAPHICS_RE.findall(text))

    if graphics != EXPECTED_TEX_GRAPHICS:
        missing = sorted(EXPECTED_TEX_GRAPHICS - graphics)
        unexpected = sorted(graphics - EXPECTED_TEX_GRAPHICS)
        raise ManifestError(
            f"v5 TeX artwork contract changed; missing={missing}, unexpected={unexpected}"
        )
    return seen, graphics


class PythonInputVisitor(ast.NodeVisitor):
    """Conservative extractor for the path idioms used by the v5 consumers."""

    def __init__(self) -> None:
        self.paths: set[Path] = set()

    @staticmethod
    def _path(node: ast.AST) -> Path | None:
        if isinstance(node, ast.Name):
            roots = {"ROOT": ROOT, "PACKAGE": ROOT, "RESULTS": RESULTS}
            return roots.get(node.id)
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
            left = PythonInputVisitor._path(node.left)
            if left is not None and isinstance(node.right, ast.Constant):
                if isinstance(node.right.value, str):
                    return left / node.right.value
        return None

    @staticmethod
    def _call_name(node: ast.AST) -> str | None:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            return node.attr
        return None

    def visit_Call(self, node: ast.Call) -> None:  # noqa: N802 (ast API)
        name = self._call_name(node.func)

        # The tests' rows()/csv_rows() helpers resolve relative names under
        # results/.  Restricting this to literal arguments makes the inventory
        # deterministic and reviewable.
        if name in {"rows", "csv_rows"} and node.args:
            arg = node.args[0]
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                self.paths.add(RESULTS / arg.value)

        # Direct readers in drivers and tests.
        if name in {"read", "read_csv"} and node.args:
            candidate = self._path(node.args[0])
            if candidate is not None:
                self.paths.add(candidate)

        # Path.read_text() and Path.open() readers.
        if isinstance(node.func, ast.Attribute) and name in {"read_text", "open"}:
            candidate = self._path(node.func.value)
            if candidate is not None:
                mode = "r"
                if node.args and isinstance(node.args[0], ast.Constant):
                    mode = str(node.args[0].value)
                for keyword in node.keywords:
                    if keyword.arg == "mode" and isinstance(keyword.value, ast.Constant):
                        mode = str(keyword.value.value)
                if not any(flag in mode for flag in "wax+"):
                    self.paths.add(candidate)

        # The two number tests deliberately read every section through globbed
        # path lists.  Expand those globs at manifest-build time.
        if isinstance(node.func, ast.Attribute) and name == "glob" and node.args:
            base = self._path(node.func.value)
            pattern = node.args[0]
            if base is not None and isinstance(pattern, ast.Constant):
                if isinstance(pattern.value, str):
                    self.paths.update(path for path in base.glob(pattern.value) if path.is_file())

        self.generic_visit(node)


def python_inputs(source: Path) -> set[Path]:
    resolved, relative = relative_file(source)
    try:
        tree = ast.parse(resolved.read_text(encoding="utf-8"), filename=relative)
    except SyntaxError as error:
        raise ManifestError(f"cannot parse Python consumer {relative}: {error}") from error
    visitor = PythonInputVisitor()
    visitor.visit(tree)
    return visitor.paths


def classify(path: Path) -> str:
    relative = path.relative_to(ROOT)
    if path in BUILD_CONTRACT:
        return "build_contract"
    if path in PYTHON_CONSUMERS or path in REPORT_SUPPORT_CODE:
        return "report_code"
    if relative.suffix == ".tex":
        return "tex_source"
    if relative.parts and relative.parts[0] in {"results", "data"}:
        return "frozen_input"
    return "report_metadata"


def build_rows() -> list[ManifestRow]:
    """Discover, hash and return the canonical sorted manifest rows."""
    consumers: dict[Path, set[str]] = defaultdict(set)

    tex_files, _ = tex_inputs()
    for path in tex_files:
        consumers[path].add("pdflatex:v5")

    for consumer in PYTHON_CONSUMERS:
        resolved_consumer, consumer_name = relative_file(consumer)
        consumers[resolved_consumer].add("python:source")
        for dependency in python_inputs(resolved_consumer):
            resolved, _ = relative_file(dependency)
            consumers[resolved].add(consumer_name)

    for consumer, dependencies in EXPLICIT_CONSUMER_INPUTS.items():
        _, consumer_name = relative_file(consumer)
        for dependency in dependencies:
            resolved, _ = relative_file(dependency)
            consumers[resolved].add(consumer_name)

    for path in BUILD_CONTRACT:
        resolved, _ = relative_file(path)
        consumers[resolved].add("v5:build-contract")

    for payload, sidecar in INTEGRITY_PAIRS:
        verify_digest_pair(payload, sidecar)
        for path in (payload, sidecar):
            resolved, _ = relative_file(path)
            consumers[resolved].add("v5:frozen-integrity")

    rows: list[ManifestRow] = []
    for path in sorted(consumers, key=lambda item: item.relative_to(ROOT).as_posix()):
        resolved, relative = relative_file(path)
        rows.append(
            ManifestRow(
                path=relative,
                kind=classify(resolved),
                scope=REPORT_SCOPE,
                consumers=";".join(sorted(consumers[path])),
                bytes=str(resolved.stat().st_size),
                sha256=sha256(resolved),
            )
        )
    return rows


def serialize(rows: Iterable[ManifestRow]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=MANIFEST_FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(row.as_dict() for row in rows)
    return stream.getvalue().encode("utf-8")


def write_manifest() -> list[ManifestRow]:
    rows = build_rows()
    payload = serialize(rows)
    MANIFEST.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    DIGEST.write_text(f"{digest}  {MANIFEST.name}\n", encoding="ascii")
    return rows


def parse_saved_manifest(path: Path) -> list[ManifestRow]:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != MANIFEST_FIELDS:
                raise ManifestError(
                    f"unexpected manifest header: {reader.fieldnames}; expected {MANIFEST_FIELDS}"
                )
            rows = [ManifestRow(**row) for row in reader]
    except FileNotFoundError as error:
        raise ManifestError(f"report manifest is missing: {path}") from error
    paths = [row.path for row in rows]
    if paths != sorted(paths) or len(paths) != len(set(paths)):
        raise ManifestError("manifest paths must be unique and sorted")
    return rows


def verify_staged_assets() -> None:
    """Verify derived submission artwork without adding outputs to the source manifest."""
    from PIL import Image

    source_dir = ROOT / "figures" / "v5"
    stage_dir = ROOT / "submission" / "figures"
    expected = set(FIGURE_STAGE_MAP) | {GRAPHICAL_STAGE_NAME}
    if not stage_dir.is_dir():
        raise ManifestError(f"staged artwork directory is missing: {stage_dir.relative_to(ROOT)}")
    actual = {path.name for path in stage_dir.iterdir() if path.is_file()}
    if actual != expected:
        raise ManifestError(
            f"staged artwork set changed; missing={sorted(expected-actual)}, "
            f"unexpected={sorted(actual-expected)}"
        )

    for staged_name, source_name in FIGURE_STAGE_MAP.items():
        staged = stage_dir / staged_name
        source = source_dir / source_name
        if not source.is_file() or not staged.is_file():
            raise ManifestError(f"missing generated/staged figure pair: {source_name}/{staged_name}")
        if sha256(staged) != sha256(source):
            raise ManifestError(f"staged artwork is stale: {staged.relative_to(ROOT)}")

    source_png = source_dir / "graphical_abstract.png"
    staged_png = stage_dir / GRAPHICAL_STAGE_NAME
    if not source_png.is_file():
        raise ManifestError(f"missing graphical-abstract source: {source_png.relative_to(ROOT)}")
    with Image.open(source_png) as opened:
        source = opened.convert("RGB")
    width, height = 920, 300
    scale = min(width / source.width, height / source.height)
    resized = source.resize((int(source.width * scale), int(source.height * scale)), Image.LANCZOS)
    canvas = Image.new("RGB", (width, height), source.getpixel((4, 4)))
    canvas.paste(resized, ((width - resized.width) // 2, (height - resized.height) // 2))
    expected_png = io.BytesIO()
    canvas.save(expected_png, "PNG", optimize=True)
    if hashlib.sha256(expected_png.getvalue()).hexdigest() != sha256(staged_png):
        raise ManifestError(f"journal-sized graphical abstract is stale: {staged_png.relative_to(ROOT)}")
    if staged_png.stat().st_size > 150_000:
        raise ManifestError("journal-sized graphical abstract exceeds 150,000 bytes")


def verify_manifest(
    manifest: Path = MANIFEST,
    digest_path: Path = DIGEST,
    *,
    check_staged_assets: bool = True,
) -> list[ManifestRow]:
    """Verify manifest digest, discovery set, byte counts and file hashes."""
    try:
        recorded_digest = digest_path.read_text(encoding="ascii").split()[0]
    except (FileNotFoundError, IndexError) as error:
        raise ManifestError(f"manifest digest is missing or empty: {digest_path}") from error
    try:
        actual_digest = sha256(manifest)
    except FileNotFoundError as error:
        raise ManifestError(f"report manifest is missing: {manifest}") from error
    if actual_digest != recorded_digest:
        raise ManifestError(
            f"manifest digest is stale: recorded {recorded_digest}, actual {actual_digest}"
        )

    saved = parse_saved_manifest(manifest)
    expected = build_rows()
    if saved != expected:
        saved_by_path = {row.path: row for row in saved}
        expected_by_path = {row.path: row for row in expected}
        missing = sorted(expected_by_path.keys() - saved_by_path.keys())
        extra = sorted(saved_by_path.keys() - expected_by_path.keys())
        stale = sorted(
            path
            for path in saved_by_path.keys() & expected_by_path.keys()
            if saved_by_path[path] != expected_by_path[path]
        )
        raise ManifestError(
            f"v5 report manifest is stale; missing={missing}, extra={extra}, stale={stale}"
        )
    if check_staged_assets:
        verify_staged_assets()
    return saved


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--write", action="store_true", help="write the canonical manifest")
    action.add_argument("--verify", action="store_true", help="verify the saved manifest (default)")
    parser.add_argument(
        "--skip-staged-assets",
        action="store_true",
        help="skip derived artwork checks (intended only while bootstrapping a clean clone)",
    )
    args = parser.parse_args(argv)

    try:
        if args.write:
            if not args.skip_staged_assets:
                verify_staged_assets()
            rows = write_manifest()
            verb = "wrote"
        else:
            rows = verify_manifest(check_staged_assets=not args.skip_staged_assets)
            verb = "verified"
    except ManifestError as error:
        print(f"v5 report manifest error: {error}", file=sys.stderr)
        return 1

    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        counts[row.kind] += 1
    detail = ", ".join(f"{kind}={counts[kind]}" for kind in sorted(counts))
    print(f"{verb} {len(rows)} v5 report-source rows ({detail})")
    print("scope: report rebuild from frozen inputs; upstream analyses were not rerun")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
