"""Layout features measured on a known shape, and the bundled centrelines."""
import numpy as np
import pandas as pd
import pytest

from apex.features.track_layout import CIRCUIT_IDS, add_layout_features, layout_features, layout_table

DEG_PER_M = 180 / (np.pi * 6_371_000)


def _stadium(straight=1000.0, radius=100.0, spacing=40.0):
    """Two straights joined by two half circles, as (lon, lat) points near the equator."""
    points = []
    for x in np.arange(0, straight, spacing):
        points.append((x, -radius))
    for a in np.arange(-np.pi / 2, np.pi / 2, spacing / radius):
        points.append((straight + radius * np.cos(a), radius * np.sin(a)))
    for x in np.arange(straight, 0, -spacing):
        points.append((x, radius))
    for a in np.arange(np.pi / 2, 3 * np.pi / 2, spacing / radius):
        points.append((radius * np.cos(a), radius * np.sin(a)))
    return [(x * DEG_PER_M, y * DEG_PER_M) for x, y in points]


def test_stadium_has_two_corners_and_its_straights():
    result = layout_features(_stadium())
    assert result["corners"] == 2 and result["slow_corners"] == 0
    assert result["length_km"] == pytest.approx((2000 + 2 * np.pi * 100) / 1000, rel=0.03)
    assert result["longest_straight_m"] == pytest.approx(1000, rel=0.15)
    assert result["turning_deg_per_km"] == pytest.approx(360 / result["length_km"], rel=0.05)


def test_bundled_centrelines_rank_monza_and_monaco_sensibly():
    table = layout_table().set_index("circuit")
    assert set(table.index) == set(CIRCUIT_IDS)
    assert table["corners_per_km"].idxmin() == "monza" and table["corners_per_km"].idxmax() == "monaco"
    assert table.loc["spa", "length_km"] == pytest.approx(7.004)


def test_layout_features_join_by_race_and_use_prior_races_only():
    layouts = pd.DataFrame({"circuit": ["twisty", "fast"], "corners_per_km": [4.0, 2.0],
                            "longest_straight_m": [300.0, 1000.0]})
    circuits = pd.DataFrame({"race_id": ["2025_1", "2025_2", "2025_3"], "circuit": ["twisty", "fast", "twisty"]})
    features = pd.DataFrame({
        "race_id": ["2025_1", "2025_1", "2025_2", "2025_2", "2025_3", "2025_3"],
        "team_canonical": ["A", "B"] * 3,
        "grid_log": np.log([1, 2, 1, 2, 1, 2]),
        "finish_position": [1, 2, 2, 1, 1, 2],
    })
    out = add_layout_features(features, circuits, layouts)
    assert out.loc[0, "grid_x_twisty"] == 0 and out.loc[1, "grid_x_twisty"] == pytest.approx(np.log(2))
    assert out.loc[1, "grid_x_straight"] == pytest.approx(-np.log(2))
    assert (out.loc[[0, 1], "team_layout_effect"] == 0).all()   # no earlier race
