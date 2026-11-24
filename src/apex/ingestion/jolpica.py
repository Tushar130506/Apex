"""Official timing data from the Jolpica F1 API (the Ergast successor).

Needs no account, unlike FastF1's live-timing laps. It has classifications,
qualifying times, every race lap time and pit stops, but no tyre compounds,
sector times or telemetry.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request

import pandas as pd

from apex.features.form_features import canonical_team

BASE_URL = "https://api.jolpi.ca/ergast/f1"
PAGE = 100
USER_AGENT = "apex-f1-research"


def lap_seconds(value: object) -> float:
    """'1:33.893' or '93.893' to seconds; blank or missing to NaN."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return float("nan")
    text = str(value).strip()
    if not text:
        return float("nan")
    minutes, _, seconds = text.rpartition(":")
    try:
        return round(float(minutes or 0) * 60.0 + float(seconds), 3)
    except ValueError:
        return float("nan")


def fetch(path: str, *, offset: int = 0, limit: int = PAGE, retries: int = 4) -> dict:
    """One page of ``/ergast/f1/<path>.json``, retrying when rate-limited."""
    query = urllib.parse.urlencode({"limit": limit, "offset": offset})
    request = urllib.request.Request(
        f"{BASE_URL}/{path}.json?{query}", headers={"User-Agent": USER_AGENT}
    )
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return json.load(response)["MRData"]
        except urllib.error.HTTPError as error:
            if error.code not in (429, 500, 502, 503) or attempt == retries:
                raise
        except urllib.error.URLError:
            if attempt == retries:
                raise
        time.sleep(2.0 * 2**attempt)
    raise RuntimeError("unreachable")


def races(path: str) -> list[dict]:
    """Every race object for ``path``, merging the pages of long tables (laps)."""
    merged: dict[tuple[str, str], dict] = {}
    offset = 0
    while True:
        data = fetch(path, offset=offset)
        for race in data["RaceTable"]["Races"]:
            key = (race["season"], race["round"])
            if key not in merged:
                merged[key] = race
                continue
            for field, value in race.items():
                if isinstance(value, list):
                    merged[key].setdefault(field, []).extend(value)
        offset += int(data["limit"])
        if offset >= int(data["total"]):
            return list(merged.values())


def _race_id(race: dict) -> str:
    return f"{race['season']}_{race['round']}"


def qualifying_table(race: dict) -> pd.DataFrame:
    """One row per driver: qualifying position and best lap of Q1-Q3."""
    rows = []
    for row in race.get("QualifyingResults", []):
        best = min(
            (lap_seconds(row.get(part)) for part in ("Q1", "Q2", "Q3")),
            key=lambda value: (pd.isna(value), value),
        )
        rows.append({
            "race_id": _race_id(race),
            "track": race["raceName"],
            "driver": row["Driver"].get("code") or row["Driver"]["driverId"][:3].upper(),
            "team": row["Constructor"]["name"],
            "qualifying_position": int(row["position"]),
            "qualifying_best_lap_s": best,
        })
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame["team"] = canonical_team(frame["team"])
    return frame


def qualifying(season: int, round_number: int) -> pd.DataFrame:
    """Qualifying for one round; empty until the session is published."""
    found = races(f"{season}/{round_number}/qualifying")
    return qualifying_table(found[0]) if found else pd.DataFrame()


def _driver_codes(race: dict) -> dict[str, str]:
    """driverId -> three-letter code, from the race classification."""
    return {
        row["Driver"]["driverId"]: row["Driver"].get("code") or row["Driver"]["driverId"][:3].upper()
        for row in race.get("Results", [])
    }


def lap_table(laps_race: dict, codes: dict[str, str]) -> pd.DataFrame:
    """Every race lap: driver, lap number, position and time in seconds."""
    rows = [
        {
            "race_id": _race_id(laps_race),
            "driver": codes.get(timing["driverId"], timing["driverId"]),
            "lap_no": int(lap["number"]),
            "position": int(timing["position"]),
            "lap_time_s": lap_seconds(timing["time"]),
        }
        for lap in laps_race.get("Laps", [])
        for timing in lap["Timings"]
    ]
    return pd.DataFrame(rows)


def pit_stop_table(pits_race: dict, codes: dict[str, str]) -> pd.DataFrame:
    """Every pit stop: driver, in-lap, stop number and pit-lane time."""
    rows = [
        {
            "race_id": _race_id(pits_race),
            "driver": codes.get(stop["driverId"], stop["driverId"]),
            "lap_no": int(stop["lap"]),
            "stop": int(stop["stop"]),
            "pit_lane_s": lap_seconds(stop.get("duration")),
        }
        for stop in pits_race.get("PitStops", [])
    ]
    return pd.DataFrame(rows, columns=["race_id", "driver", "lap_no", "stop", "pit_lane_s"])


def race_laps(season: int, round_number: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Race lap times and pit stops for one round, with three-letter driver codes."""
    results = races(f"{season}/{round_number}/results")
    if not results:
        raise ValueError(f"no classification for {season} round {round_number}")
    codes = _driver_codes(results[0])
    laps = races(f"{season}/{round_number}/laps")
    pits = races(f"{season}/{round_number}/pitstops")
    lap_frame = lap_table(laps[0], codes) if laps else pd.DataFrame()
    pit_frame = pit_stop_table(pits[0], codes) if pits else pit_stop_table({}, codes)
    return lap_frame, pit_frame


__all__ = [
    "fetch",
    "lap_seconds",
    "lap_table",
    "pit_stop_table",
    "qualifying",
    "qualifying_table",
    "race_laps",
    "races",
]
