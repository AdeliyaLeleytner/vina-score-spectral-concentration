from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

import public_sequence_geometry_controls as sequence_controls


def test_symmetric_identity_is_invariant_to_target_order() -> None:
    sequences = {
        "first": "MKTAYIAKQRQISFVKSHFSRQ",
        "second": "MKTAYIAKQKQISFVKSYFSRQ",
    }
    forward, _ = sequence_controls.symmetric_sequence_identity(
        sequences, ("first", "second")
    )
    reverse, _ = sequence_controls.symmetric_sequence_identity(
        sequences, ("second", "first")
    )
    assert np.array_equal(forward, forward.T)
    assert np.array_equal(reverse, reverse.T)
    assert forward[0, 1] == reverse[0, 1]


def test_qap_uses_one_shared_permutation_for_raw_residual_difference() -> None:
    sequence = np.asarray(
        [
            [1.0, 0.8, 0.2, 0.1],
            [0.8, 1.0, 0.3, 0.2],
            [0.2, 0.3, 1.0, 0.7],
            [0.1, 0.2, 0.7, 1.0],
        ]
    )
    residual = sequence.copy()
    raw = 1.0 - sequence
    np.fill_diagonal(raw, 1.0)
    first = sequence_controls.qap_context(
        raw,
        residual,
        sequence,
        np.arange(4),
        context="fixture",
        permutations=200,
        seed=19,
    )
    second = sequence_controls.qap_context(
        raw,
        residual,
        sequence,
        np.arange(4),
        context="fixture",
        permutations=200,
        seed=19,
    )
    assert first == second
    assert first["residual_sequence_pearson"] == pytest.approx(1.0)
    assert first["residual_minus_raw_sequence_pearson"] > 0.9


def test_frozen_full_sequence_contract_matches_matrix() -> None:
    sequences, metadata = sequence_controls.load_sequences()
    assert tuple(sequences) == sequence_controls.TARGETS
    assert len(sequences) == 58
    assert metadata["targets"] == 58


def test_production_artifact_is_fail_closed_if_present() -> None:
    output = sequence_controls.DEFAULT_OUTPUT
    if not output.exists():
        pytest.skip("production sequence artifact has not been generated")
    checksums = json.loads((output / "output_checksums.json").read_text())
    for name in sequence_controls.OUTPUT_FILES:
        path = output / name
        assert path.exists()
        assert checksums[name] == hashlib.sha256(path.read_bytes()).hexdigest()
    summary = json.loads((output / "summary.json").read_text())
    contexts = {record["context"] for record in summary["qap"]}
    assert contexts == {
        "all_58_targets_centered_within_58",
        "common_21_kinases_embedded_in_58_target_centering",
        "common_21_kinases_centered_within_21",
    }
