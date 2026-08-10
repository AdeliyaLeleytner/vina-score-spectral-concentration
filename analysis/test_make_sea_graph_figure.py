from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from analysis import make_sea_graph_figure as figure


def _inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return (
        pd.read_csv(figure.DEFAULT_SEA),
        pd.read_csv(figure.DEFAULT_PAIRS),
        pd.read_csv(figure.DEFAULT_ANNOTATIONS),
        pd.read_csv(figure.DEFAULT_PANEL_METRICS),
    )


def test_primary_edge_classification_matches_frozen_audit() -> None:
    sea, pairs, _, _ = _inputs()
    edges = figure.prepare_edges(sea, pairs)
    summary = figure.summarize(edges)

    assert summary["target_pair_count"] == 190
    assert summary["sea_only_top_edges"] == 15
    assert summary["residual_only_top_edges"] == 15
    assert summary["shared_top_edges"] == 4
    assert summary["experimental_positives_in_sea_top19"] == 6
    assert summary["experimental_positives_in_residual_top19"] == 5
    assert summary["shared_top_edge_pairs"] == [
        "ABL1--KIT",
        "AKT1--AKT2",
        "KDR--KIT",
        "LCK--SRC",
    ]


def test_grouped_layout_is_deterministic_and_complete() -> None:
    _, _, annotations, _ = _inputs()
    first = figure.grouped_circular_layout(annotations)
    second = figure.grouped_circular_layout(annotations.sample(frac=1, random_state=17))

    pd.testing.assert_frame_equal(first, second)
    assert len(first) == 20
    assert first.target.nunique() == 20
    assert set(first.klifs_group) == set(figure.GROUP_ORDER)


def test_publication_figure_has_four_panels() -> None:
    sea, pairs, annotations, metrics = _inputs()
    edges = figure.prepare_edges(sea, pairs)
    layout = figure.grouped_circular_layout(annotations)

    result = figure.make_figure(edges, layout, metrics)
    try:
        assert len(result.axes) == 4
        assert tuple(result.get_size_inches()) == (
            figure.FIGURE_WIDTH,
            figure.FIGURE_HEIGHT,
        )
    finally:
        plt.close(result)
