"""Official race and sprint classifications from FastF1 sessions.

Kept free of any FastF1 import so it can be tested with plain tables.
"""
from __future__ import annotations

import re

import pandas as pd

from apex.models.points import points_for_position

SESSION_NAMES = {"race": "Race", "sprint": "Sprint"}


def session_has_run(event: pd.Series, session_name: str, now: pd.Timestamp | None = None) -> bool:
    """True when ``session_name`` is on this event's schedule and already over.

    The schedule lists the whole season, so sessions in the future must be
    skipped instead of being reported as download failures. A session counts
    as over six hours after its scheduled start.
    """
    now = pd.Timestamp.now(tz="UTC").tz_localize(None) if now is None else pd.Timestamp(now)
    for column in event.index:
        if not re.fullmatch(r"Session\d+", str(column)) or str(event[column]).strip() != session_name:
            continue
        start = pd.to_datetime(event.get(f"{column}DateUtc"), errors="coerce")
        if pd.isna(start):
            start = pd.to_datetime(event.get("EventDate"), errors="coerce")
        return bool(pd.notna(start) and start + pd.Timedelta(hours=6) <= now)
    return False


def completed_rounds(
    schedule: pd.DataFrame, event_type: str = "race", now: pd.Timestamp | None = None
) -> list[tuple[int, str]]:
    """``(round, event name)`` for every finished race, or finished sprint."""
    if event_type not in SESSION_NAMES:
        raise ValueError(f"event_type must be one of {sorted(SESSION_NAMES)}")
    name = SESSION_NAMES[event_type]
    return [
        (int(event["RoundNumber"]), str(event["EventName"]))
        for _, event in schedule.iterrows()
        if int(event["RoundNumber"]) > 0 and session_has_run(event, name, now)
    ]


def result_rows(
    results: pd.DataFrame, year: int, round_number: int, track: str, event_type: str = "race"
) -> list[dict]:
    """One row per driver from a FastF1 ``session.results`` table.

    Points are derived from the classified position on the current scale
    (sprints: 8-7-6-5-4-3-2-1). The 2021 sprints paid 3-2-1, so derived sprint
    points for that season are too high; ``source_points`` keeps what the
    timing source reported.
    """
    rows = []
    for _, row in results.iterrows():
        position = pd.to_numeric(row.get("Position"), errors="coerce")
        classified_position = pd.to_numeric(row.get("ClassifiedPosition"), errors="coerce")
        status = str(row.get("Status", ""))
        disqualified = "disqual" in status.lower()
        running = status.startswith(("Finished", "Lapped", "+"))
        classified = running or pd.notna(classified_position)
        points = points_for_position(
            position, event_type=event_type, classified=classified, disqualified=disqualified
        )
        rows.append({
            "race_id": f"{year}_{round_number}",
            "season": year,
            "track": track,
            "driver": str(row["Abbreviation"]),
            "finish_position": position,
            "grid_position": pd.to_numeric(row.get("GridPosition"), errors="coerce"),
            "classified": classified,
            "running_at_finish": running,
            "points": points,
            "derived_points": points,
            "source_points": pd.to_numeric(row.get("Points"), errors="coerce"),
            "event_type": event_type,
            "status": status,
            "team": str(row.get("TeamName", row.get("Team", "UNKNOWN"))),
        })
    return rows
