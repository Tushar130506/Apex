import pandas as pd
import pytest

from apex.models.championship import forecast_championship
from apex.models.replay import replay_snapshots


def test_championship_forecast_accumulates_driver_and_constructor_points():
    completed = pd.DataFrame(
        [
            {"race_id": "2026_1", "driver": "A", "team": "X", "points": 25},
            {"race_id": "2026_1", "driver": "B", "team": "Y", "points": 18},
        ]
    )
    future = pd.DataFrame(
        [
            {"race_id": "2026_2", "driver": "A", "team": "X", "score": 0.9},
            {"race_id": "2026_2", "driver": "B", "team": "Y", "score": 0.1},
            {"race_id": "2026_3", "driver": "A", "team": "X", "score": 0.8},
            {"race_id": "2026_3", "driver": "B", "team": "Y", "score": 0.2},
        ]
    )
    forecast = forecast_championship(completed, future, simulations=500, seed=2)
    assert forecast.driver.iloc[0]["driver"] == "A"
    assert forecast.driver.iloc[0]["win_probability"] > 0.8
    assert forecast.constructor.iloc[0]["team"] == "X"
    assert forecast.driver["win_probability"].sum() == pytest.approx(1.0)


def test_replay_snapshots_do_not_count_future_events():
    laps = pd.DataFrame(
        [
            {"race_id": "2026_1", "lap_no": 1, "driver": "A", "lap_time_s": 90.0, "compound": "MEDIUM", "tyre_age": 1, "is_pit_lap": False, "track_status": "1"},
            {"race_id": "2026_1", "lap_no": 2, "driver": "A", "lap_time_s": 91.0, "compound": "HARD", "tyre_age": 1, "is_pit_lap": True, "track_status": "4"},
        ]
    )
    snapshots = replay_snapshots(laps)
    assert snapshots.iloc[0]["pit_stops_observed"] == 0
    assert snapshots.iloc[0]["safety_car_laps_observed"] == 0
    assert snapshots.iloc[1]["pit_stops_observed"] == 1
    assert snapshots.iloc[1]["safety_car_laps_observed"] == 1


def test_championship_rejects_overlap():
    rows = pd.DataFrame([{"race_id": "2026_1", "driver": "A", "team": "X", "points": 25}])
    predictions = pd.DataFrame([{"race_id": "2026_1", "driver": "A", "team": "X", "score": 1.0}])
    with pytest.raises(ValueError, match="overlap"):
        forecast_championship(rows, predictions)
