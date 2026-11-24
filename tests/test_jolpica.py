"""Jolpica parsing and the weekend grid, with hand-written API payloads."""
import importlib.util
from pathlib import Path

import pandas as pd
import pytest

from apex.ingestion.jolpica import lap_seconds, lap_table, pit_stop_table, qualifying_table

ROOT = Path(__file__).resolve().parents[1]
RACE = {"season": "2026", "round": "22", "raceName": "Qatar Grand Prix"}


def _driver(code, driver_id):
    return {"driverId": driver_id, "code": code}


def test_lap_seconds():
    assert lap_seconds("1:33.893") == 93.893 and lap_seconds("59.1") == 59.1
    assert pd.isna(lap_seconds("")) and pd.isna(lap_seconds(None)) and pd.isna(lap_seconds("DNF"))


def test_qualifying_uses_best_session_and_canonical_teams():
    race = {**RACE, "QualifyingResults": [
        {"position": "1", "Driver": _driver("VER", "max_verstappen"), "Constructor": {"name": "Red Bull"},
         "Q1": "1:21.5", "Q2": "1:20.9", "Q3": "1:20.7"},
        {"position": "2", "Driver": _driver("LAW", "lawson"), "Constructor": {"name": "RB F1 Team"},
         "Q1": "1:21.0", "Q2": "1:20.8", "Q3": ""},
        {"position": "20", "Driver": _driver("BOT", "bottas"), "Constructor": {"name": "Cadillac F1 Team"},
         "Q1": ""},
    ]}
    table = qualifying_table(race)
    assert table["qualifying_best_lap_s"].tolist()[:2] == [80.7, 80.8] and pd.isna(table["qualifying_best_lap_s"].iloc[2])
    assert table["team"].tolist() == ["Red Bull Racing", "Racing Bulls", "Cadillac"]
    assert set(table["race_id"]) == {"2026_22"}


def test_laps_and_pit_stops_use_driver_codes():
    codes = {"max_verstappen": "VER", "norris": "NOR"}
    laps = lap_table({**RACE, "Laps": [
        {"number": "1", "Timings": [{"driverId": "max_verstappen", "position": "1", "time": "1:30.1"},
                                    {"driverId": "norris", "position": "2", "time": "1:30.6"}]},
    ]}, codes)
    assert laps["driver"].tolist() == ["VER", "NOR"] and laps["lap_time_s"].tolist() == [90.1, 90.6]
    pits = pit_stop_table({**RACE, "PitStops": [
        {"driverId": "norris", "lap": "18", "stop": "1", "duration": "22.4"}]}, codes)
    assert pits.iloc[0].to_dict() == {"race_id": "2026_22", "driver": "NOR", "lap_no": 18, "stop": 1, "pit_lane_s": 22.4}
    assert list(pit_stop_table({}, codes).columns) == ["race_id", "driver", "lap_no", "stop", "pit_lane_s"]


def _weekend():
    spec = importlib.util.spec_from_file_location("fetch_weekend", ROOT / "scripts" / "fetch_weekend.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_grid_penalties_move_cars_and_the_rest_close_up():
    entries = pd.DataFrame({"driver": ["ANT", "VER", "RUS", "NOR", "HAD"], "qualifying_position": [1, 2, 3, 4, 5]})
    grid = _weekend().starting_grid(entries, {"ANT": 5})
    assert dict(zip(entries["driver"], grid)) == {"ANT": 5, "VER": 1, "RUS": 2, "NOR": 3, "HAD": 4}
    assert _weekend().starting_grid(entries, {}).tolist() == [1, 2, 3, 4, 5]
    with pytest.raises(ValueError):
        _weekend().starting_grid(entries, {"XXX": 2})
    with pytest.raises(ValueError):
        _weekend().starting_grid(entries, {"ANT": 2, "VER": 2})
