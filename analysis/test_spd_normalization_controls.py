from __future__ import annotations

import json

import numpy as np
import pytest

try:
    from . import spd_normalization_controls as controls
except ImportError:  # pragma: no cover
    import spd_normalization_controls as controls  # type: ignore


def test_representation_shapes_and_ligand_efficiency_definition() -> None:
    scores = np.array(
        [[-2.0, -4.0, -3.0], [-6.0, -3.0, -9.0], [-8.0, -4.0, -2.0], [-5.0, -7.0, -6.0]]
    )
    heavy = np.array([2, 3, 4, 5])
    observed = controls.representation_geometries(scores, heavy)
    assert set(observed) == {
        "raw",
        "standard_centered",
        "within_ligand_ordinal",
        "ligand_efficiency_raw",
        "ligand_efficiency_centered",
    }
    assert all(matrix.shape == (3, 3) for matrix in observed.values())
    assert all(np.allclose(np.diag(matrix), 1.0) for matrix in observed.values())


def test_within_ligand_ordinal_is_invariant_to_rowwise_positive_affine_change() -> None:
    rng = np.random.default_rng(3)
    scores = rng.normal(size=(30, 8))
    heavy = rng.integers(5, 30, size=30)
    transformed = scores * rng.uniform(0.5, 4.0, size=(30, 1)) + rng.normal(
        size=(30, 1)
    )
    first = controls.representation_geometries(scores, heavy)
    second = controls.representation_geometries(transformed, heavy)
    assert np.allclose(
        first["within_ligand_ordinal"], second["within_ligand_ordinal"]
    )


def test_release_summary() -> None:
    path = controls.DEFAULT_OUTPUT / "summary.json"
    if not path.is_file():
        pytest.skip("normalization controls have not been built")
    summary = json.loads(path.read_text(encoding="utf-8"))
    assert summary["docking_rows"] == 258770
    assert summary["results"]["standard_centered"]["floor_at_bound"] == pytest.approx(
        0.37229690239625957
    )
    assert "heavy-atom" in summary["ligand_efficiency_definition"]
