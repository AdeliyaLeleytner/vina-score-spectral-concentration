from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

try:
    from . import klifs_pocket_control as klifs
except ImportError:  # pragma: no cover
    import klifs_pocket_control as klifs  # type: ignore


def _synthetic_klifs_payload() -> list[dict]:
    rows = []
    for index, target in enumerate(klifs.TARGETS):
        pocket = "A" * klifs.POCKET_LENGTH
        if index == 1:
            pocket = "C" + pocket[1:]
        rows.append(
            {
                "kinase_ID": index + 1,
                "name": target,
                "gene_name": target,
                "family": "family_a" if index < 2 else f"family_{index}",
                "group": "group_a" if index < 3 else f"group_{index}",
                "subfamily": "",
                "species": "Human",
                "uniprot": f"P{index:05d}",
                "pocket": pocket,
            }
        )
    return rows


def test_klifs_loader_and_pocket_identity(tmp_path) -> None:
    path = tmp_path / "klifs.json"
    path.write_text(json.dumps(_synthetic_klifs_payload()), encoding="utf-8")
    records, annotations = klifs.load_klifs_annotations(
        path, expected_sha256=None
    )
    assert list(records) == list(klifs.TARGETS)
    assert len(annotations) == len(klifs.TARGETS)
    matrices = klifs.structural_matrices(records)
    expected_identity = (klifs.POCKET_LENGTH - 1) / klifs.POCKET_LENGTH
    assert matrices["klifs_pocket_identity"][0, 1] == pytest.approx(
        expected_identity
    )
    assert matrices["same_klifs_family"][0, 1] == 1
    assert matrices["same_klifs_group"][0, 2] == 1
    assert matrices["same_nonempty_klifs_subfamily"][0, 1] == 0


def test_klifs_loader_rejects_bad_pocket_length(tmp_path) -> None:
    payload = _synthetic_klifs_payload()
    payload[0]["pocket"] = "A" * (klifs.POCKET_LENGTH - 1)
    path = tmp_path / "klifs.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="pocket"):
        klifs.load_klifs_annotations(path, expected_sha256=None)


def test_partial_rank_qap_is_seed_deterministic() -> None:
    rng = np.random.default_rng(7)
    target_count = 8
    predictor = np.corrcoef(rng.normal(size=(target_count, 200)))
    endpoint = np.corrcoef(rng.normal(size=(target_count, 200)))
    control = np.corrcoef(rng.normal(size=(target_count, 200)))
    maps_first = klifs.target_label_pair_maps(target_count, 199, seed=13)
    maps_second = klifs.target_label_pair_maps(target_count, 199, seed=13)
    assert np.array_equal(maps_first, maps_second)
    first = klifs.partial_rank_qap(
        predictor, endpoint, [control], maps_first
    )
    second = klifs.partial_rank_qap(
        predictor, endpoint, [control], maps_second
    )
    assert first == second
    assert -1 <= first["partial_spearman"] <= 1
    assert 0 < first["target_label_qap_p_positive"] <= 1


def test_vectorized_retrieval_matches_scalar_implementation() -> None:
    labels = np.array([True, False, False, True, False, False])
    scores = np.array(
        [
            [0.9, 0.1, 0.2, 0.8, 0.3, 0.4],
            [1.0, 1.0, 0.0, 1.0, 0.0, 0.0],
        ]
    )
    auc, average_precision = klifs._row_retrieval_metrics(scores, labels)
    for index, row in enumerate(scores):
        scalar = klifs.retrieval.retrieval_metrics(labels, row)
        assert auc[index] == pytest.approx(scalar["roc_auc"])
        assert average_precision[index] == pytest.approx(
            scalar["average_precision"]
        )


def test_release_provenance_path_is_relative_and_fails_closed(tmp_path) -> None:
    observed = klifs.release_provenance_path(klifs.DEFAULT_TARGET_PAIRS)
    assert observed == "results/replicated_pair_retrieval/target_pairs.csv"
    assert not Path(observed).is_absolute()
    external = tmp_path / "external.csv"
    external.write_text("x\n", encoding="utf-8")
    with pytest.raises(ValueError, match="inside the package"):
        klifs.release_provenance_path(external)


def test_release_summary_preserves_post_hoc_boundary() -> None:
    with (klifs.DEFAULT_OUTPUT / "summary.json").open() as handle:
        summary = json.load(handle)
    assert summary["analysis_status"] == "exploratory_post_hoc"
    assert summary["support"]["targets"] == 20
    assert summary["support"]["target_pairs"] == 190
    assert summary["support"]["locked_positive_pairs"] == 15
    old = summary["key_results"][
        "old_three_panel_mean_controlling_KLIFS_pocket"
    ]
    assert old["partial_spearman"] == pytest.approx(0.263039808962936)
    assert old["target_label_qap_p_positive"] < 0.02
    external = summary["key_results"][
        "independent_KiRHub_controlling_KLIFS_pocket"
    ]
    assert external["partial_spearman"] == pytest.approx(0.31525295499214584)
    assert external["target_label_qap_p_positive"] < 0.01
    assert "post hoc" in summary["selection_boundary"]
    source_path = summary["sources"]["fixed_target_pairs"]["path"]
    assert source_path == "results/replicated_pair_retrieval/target_pairs.csv"
    assert not Path(source_path).is_absolute()
    assert str(klifs.PACKAGE) not in json.dumps(summary)
