"""Result materialization and the one-command rebuild, without any download."""
import importlib.util
from pathlib import Path

import pandas as pd
import pytest

from apex.ingestion.race_results import completed_rounds, result_rows, session_has_run
from apex.models.forecast import completed_sprints

ROOT = Path(__file__).resolve().parents[1]
NOW = pd.Timestamp("2026-11-17 12:00")


def _schedule() -> pd.DataFrame:
    def event(round_number, name, sessions, race_start):
        row = {"RoundNumber": round_number, "EventName": name, "EventDate": pd.Timestamp(race_start).normalize()}
        for number, session in enumerate(sessions, 1):
            row[f"Session{number}"] = session
            row[f"Session{number}DateUtc"] = pd.Timestamp(race_start) - pd.Timedelta(days=len(sessions) - number)
        return row

    normal = ["Practice 1", "Practice 2", "Practice 3", "Qualifying", "Race"]
    sprint = ["Practice 1", "Sprint Qualifying", "Sprint", "Qualifying", "Race"]
    return pd.DataFrame([
        event(0, "Pre-Season Testing", ["Practice 1", "Practice 2", "Practice 3"], "2026-02-20 15:00"),
        event(19, "Mexico City Grand Prix", normal, "2026-10-25 20:00"),
        event(20, "São Paulo Grand Prix", sprint, "2026-11-08 17:00"),
        event(21, "Las Vegas Grand Prix", normal, "2026-11-22 04:00"),
        event(22, "Qatar Grand Prix", sprint, "2026-11-29 16:00"),
    ])


def test_only_finished_rounds_are_listed():
    schedule = _schedule()
    assert completed_rounds(schedule, "race", NOW) == [
        (19, "Mexico City Grand Prix"), (20, "São Paulo Grand Prix"),
    ]
    assert completed_rounds(schedule, "sprint", NOW) == [(20, "São Paulo Grand Prix")]
    with pytest.raises(ValueError):
        completed_rounds(schedule, "qualifying", NOW)


def test_sprint_qualifying_is_not_a_sprint_and_running_sessions_wait():
    qatar = _schedule().iloc[4]
    after_sprint_qualifying = pd.Timestamp("2026-11-27 03:00")
    assert session_has_run(qatar, "Sprint Qualifying", after_sprint_qualifying)
    assert not session_has_run(qatar, "Sprint", after_sprint_qualifying)
    race_start = pd.Timestamp("2026-11-29 16:00")
    assert not session_has_run(qatar, "Race", race_start + pd.Timedelta(hours=2))
    assert session_has_run(qatar, "Race", race_start + pd.Timedelta(hours=6))


def _classification() -> pd.DataFrame:
    return pd.DataFrame({
        "Abbreviation": ["NOR", "VER", "HAD", "ALB"],
        "TeamName": ["McLaren", "Red Bull Racing", "Red Bull Racing", "Williams"],
        "Position": [1.0, 2.0, 3.0, 20.0],
        "ClassifiedPosition": ["1", "2", "3", "R"],
        "GridPosition": [1.0, 4.0, 3.0, 9.0],
        "Status": ["Finished", "Finished", "+1 Lap", "Accident"],
        "Points": [8.0, 7.0, 6.0, 0.0],
    })


def test_result_rows_score_races_and_sprints_on_their_own_scales():
    race = pd.DataFrame(result_rows(_classification(), 2026, 20, "São Paulo Grand Prix"))
    sprint = pd.DataFrame(result_rows(_classification(), 2026, 20, "São Paulo Grand Prix", "sprint"))
    assert race["points"].tolist() == [25.0, 18.0, 15.0, 0.0]
    assert sprint["points"].tolist() == [8.0, 7.0, 6.0, 0.0]
    assert set(sprint["event_type"]) == {"sprint"} and set(race["race_id"]) == {"2026_20"}
    assert race["classified"].tolist() == [True, True, True, False]


def test_completed_sprints_keep_finished_rounds_only():
    sprints = pd.concat([
        pd.DataFrame(result_rows(_classification(), 2026, 20, "São Paulo Grand Prix", "sprint")),
        pd.DataFrame(result_rows(_classification(), 2026, 22, "Qatar Grand Prix", "sprint")),
    ])
    rows = completed_sprints(sprints, {"2026_19", "2026_20"})
    assert set(rows["race_id"]) == {"2026_20"} and set(rows["event_type"]) == {"sprint"}
    assert rows["team"].notna().all()
    with pytest.raises(ValueError):
        completed_sprints(pd.concat([sprints, sprints]), {"2026_20"})


def _rebuild():
    spec = importlib.util.spec_from_file_location("rebuild_data", ROOT / "scripts" / "rebuild_data.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_rebuild_runs_every_step_in_order_on_existing_scripts():
    rebuild = _rebuild()
    commands = rebuild.build_commands(rebuild.parse_args(["--first-year", "2018", "--last-year", "2026"]))
    assert [name for name, _ in commands] == list(rebuild.STEPS)
    for _, command in commands:
        assert (ROOT / command[1]).exists()
    steps = dict(commands)
    assert "2018-2026" in steps["download"] and "--telemetry" not in steps["download"]
    assert steps["results"][2:11] == [str(year) for year in range(2018, 2027)]
    assert "2020" not in steps["sprints"] and "2021" in steps["sprints"]
    results_file = steps["results"][steps["results"].index("--out") + 1]
    assert steps["features"][steps["features"].index("--results") + 1] == results_file


def test_rebuild_can_resume_and_skips_sprints_before_they_existed():
    rebuild = _rebuild()
    resumed = rebuild.build_commands(rebuild.parse_args(["--from-step", "results", "--last-year", "2026"]))
    assert [name for name, _ in resumed] == ["results", "sprints", "features"]
    early = rebuild.build_commands(rebuild.parse_args(["--first-year", "2018", "--last-year", "2019"]))
    assert "sprints" not in [name for name, _ in early]
