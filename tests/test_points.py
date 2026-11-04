import pandas as pd
import pytest

from apex.models.points import aggregate_championship_points, derive_points, points_for_position


def test_grand_prix_and_sprint_points_have_no_fastest_lap_bonus():
    assert points_for_position(1) == 25
    assert points_for_position(1, event_type="sprint") == 8
    assert points_for_position(11) == 0


def test_shortened_race_and_disqualification():
    assert points_for_position(1, distance_fraction=0.2) == 0
    assert points_for_position(1, distance_fraction=0.4) == 12
    assert points_for_position(1, distance_fraction=0.6) == 18
    assert points_for_position(1, disqualified=True) == 0
    with pytest.raises(ValueError):
        points_for_position(1, event_type="qualifying")


def test_sprint_points_aggregate_to_constructors():
    rows = pd.DataFrame(
        [
            {"driver": "A", "team": "X", "finish_position": 1, "classified": True, "event_type": "race"},
            {"driver": "B", "team": "X", "finish_position": 2, "classified": True, "event_type": "sprint"},
            {"driver": "C", "team": "Y", "finish_position": 1, "classified": True, "event_type": "sprint"},
        ]
    )
    derived = derive_points(rows)
    drivers, teams = aggregate_championship_points(derived)
    assert drivers.set_index("driver").loc["A", "points"] == 25
    assert drivers.set_index("driver").loc["B", "points"] == 7
    assert teams.set_index("team").loc["X", "points"] == 32
