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


def _three_way(points, races):
    from apex.models.championship import title_status
    return title_status(pd.Series(points, index=list("ABC")), races * 25)


def test_title_status_marks_clinched_contender_and_eliminated():
    status = _three_way([100, 60, 40], races=2)
    assert status["title_status"].tolist() == ["contender", "contender", "eliminated"]
    assert status["max_possible_points"].tolist() == [150, 110, 90]
    assert status["points_behind_leader"].tolist() == [0, 40, 60]
    assert _three_way([100, 49, 40], races=2)["title_status"].tolist() == ["clinched", "eliminated", "eliminated"]


def test_a_possible_draw_on_points_stays_open():
    # B can reach 100 exactly; countback could still decide for B.
    assert _three_way([100, 50, 0], races=2)["title_status"].tolist()[:2] == ["contender", "contender"]


def _season(points_a, points_b, races):
    completed = pd.DataFrame(
        [
            {"race_id": "2026_1", "driver": "A", "team": "X", "points": points_a},
            {"race_id": "2026_1", "driver": "B", "team": "Y", "points": points_b},
            {"race_id": "2026_1", "driver": "C", "team": "Z", "points": 0},
        ]
    )
    future = pd.DataFrame(
        [
            {"race_id": f"2026_{number}", "driver": driver, "team": team, "score": score}
            for number in range(2, 2 + races)
            for driver, team, score in (("A", "X", 0.98), ("B", "Y", 0.01), ("C", "Z", 0.01))
        ]
    )
    return forecast_championship(completed, future, simulations=400, seed=1)


def test_simulation_agrees_with_title_status_and_sentences_say_so():
    from apex.models.championship import describe_title_chances
    forecast = _season(60, 50, races=2)
    drivers = forecast.driver.set_index("driver")
    assert drivers.loc["C", "title_status"] == "eliminated" and drivers.loc["C", "win_probability"] == 0
    assert drivers.loc["B", "title_status"] == "contender"
    assert "can no longer win" in describe_title_chances(forecast, "c")
    assert "still mathematically open" in describe_title_chances(forecast, "B")
    assert "under" in describe_title_chances(forecast, "B") or drivers.loc["B", "win_probability"] > 0
    assert forecast.constructor.set_index("team").loc["Z", "max_possible_points"] == 2 * 43
    clinched = _season(200, 50, races=2)
    assert clinched.driver.set_index("driver").loc["A", "win_probability"] == 1
    assert "has won the title" in describe_title_chances(clinched, "A")
    with pytest.raises(ValueError):
        describe_title_chances(clinched, "ZZZ")


def test_unknown_remaining_sprints_keep_the_title_check_safe():
    completed = pd.DataFrame(
        [
            {"race_id": "2026_1", "driver": "A", "team": "X", "points": 100},
            {"race_id": "2026_1", "driver": "B", "team": "Y", "points": 48},
        ]
    )
    future = pd.DataFrame(
        [
            {"race_id": f"2026_{number}", "driver": driver, "team": team, "score": score}
            for number in (2, 3)
            for driver, team, score in (("A", "X", 0.9), ("B", "Y", 0.1))
        ]
    )
    known = forecast_championship(completed, future, simulations=50, seed=0)
    assumed = forecast_championship(completed, future, simulations=50, seed=0, sprints_for_title_status=2)
    assert known.driver.set_index("driver").loc["B", "title_status"] == "eliminated"
    assert assumed.driver.set_index("driver").loc["B", "title_status"] == "contender"
    assert assumed.driver.set_index("driver").loc["B", "max_possible_points"] == 48 + 50 + 16

    sentence = describe_title_chances_for(assumed, "B")
    assert "2 race(s) and 2 sprint(s), 2 of them assumed" in sentence and "at most 114" in sentence


def describe_title_chances_for(forecast, name):
    from apex.models.championship import describe_title_chances
    return describe_title_chances(forecast, name)
