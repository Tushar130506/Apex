"""FastF1 historical loader -> APEX lap contract (source=fastf1_real). Requires `pip install fastf1`.

Real-world fuel mass is NOT in FastF1; fuel is an explicit *estimate* (linear burn) flagged 'estimated'.
"""
from __future__ import annotations
from pathlib import Path
import re
from typing import Any
import numpy as np
import pandas as pd

START_FUEL_KG = 105.0  # assumption: ~110 kg limit, small allowance; used only as an estimate


def _enable_cache(cache_dir: str) -> None:
    """FastF1 >=3.9 deprecates Cache.enable_cache in favour of Cache.configure; support both."""
    import warnings
    import fastf1
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    if hasattr(fastf1.Cache, "configure"):
        try:
            fastf1.Cache.configure(cache_dir=cache_dir)
            return
        except TypeError:
            pass
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        fastf1.Cache.enable_cache(cache_dir)


def load_session(year: int, event: str | int, session: str = "R", cache_dir: str = ".fastf1-cache",
                 total_laps: int | None = None) -> pd.DataFrame:
    import fastf1  # lazy import: optional dependency
    _enable_cache(cache_dir)
    ses = fastf1.get_session(year, event, session)
    ses.load(telemetry=False, weather=True, messages=False)
    return laps_to_contract(ses.laps, ses.weather_data, f"fastf1_{year}_{ses.event['RoundNumber']}_{session}",
                            str(ses.event["EventName"]), total_laps or int(ses.laps["LapNumber"].max()), year)


def laps_to_contract(laps: pd.DataFrame, weather: pd.DataFrame | None, session_id: str, track: str,
                     total_laps: int, season: int) -> pd.DataFrame:
    df = laps.copy()
    reasons = pd.Series(pd.NA, index=df.index, dtype="object")
    pit = df["PitInTime"].notna() | df["PitOutTime"].notna()
    reasons[df["LapTime"].isna()] = "no_time"
    reasons[pit & reasons.isna()] = "pit_lap"
    ts = df["TrackStatus"].astype(str)
    reasons[ts.str.contains(r"[4567]") & reasons.isna()] = "neutralised"   # SC/VSC/red flag codes
    if "IsAccurate" in df:
        reasons[(~df["IsAccurate"].fillna(False)) & reasons.isna()] = "not_accurate"
    tt = at = np.nan
    out = pd.DataFrame({
        "source": "fastf1_real", "season": int(season), "session_id": session_id, "track": track, "driver": df["Driver"],
        "lap_no": df["LapNumber"].astype(int), "lap_time_s": df["LapTime"].dt.total_seconds(),
        "valid": reasons.isna(), "exclusion_reason": reasons,
        "compound": df["Compound"].fillna("UNKNOWN"), "tyre_age": df["TyreLife"].fillna(0).astype(int),
        "stint_id": df["Stint"].fillna(0).astype(int),
        "fuel_kg": START_FUEL_KG * (1 - (df["LapNumber"] - 1) / total_laps), "fuel_provenance": "estimated",
        "track_temp": tt, "air_temp": at, "track_status": ts, "is_pit_lap": pit})
    if weather is not None and len(weather):
        w = weather.sort_values("Time")
        idx = np.searchsorted(w["Time"].values, df["LapStartTime"].values).clip(0, len(w) - 1)
        out["track_temp"] = w["TrackTemp"].values[idx]; out["air_temp"] = w["AirTemp"].values[idx]
    return out.reset_index(drop=True)


def load_season(year: int, session: str = "R", cache_dir: str = ".fastf1-cache", max_events: int | None = None,
                skip_errors: bool = True) -> pd.DataFrame:
    """Download every race of a season (volume comes from distinct sessions, not rows)."""
    import fastf1
    _enable_cache(cache_dir)
    sched = fastf1.get_event_schedule(year, include_testing=False)
    frames = []
    for rnd in sched["RoundNumber"].tolist()[:max_events]:
        try:
            frames.append(load_session(year, int(rnd), session, cache_dir))
        except Exception as e:  # noqa: BLE001 - one bad round must not kill the season
            if not skip_errors:
                raise
            print(f"skip {year} round {rnd}: {e}")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def download_cache(
    years: list[int],
    cache_dir: str = ".fastf1-cache",
    sessions: str = "all",
    telemetry: bool = False,
    include_testing: bool = False,
    skip_errors: bool = True,
) -> list[dict[str, Any]]:
    """Download FastF1 data for every scheduled session into ``cache_dir``.

    The operation is resumable because FastF1 serves existing cache entries
    without downloading them again. The returned records make skipped sessions
    visible to callers and scripts.
    """
    import fastf1

    if sessions not in {"all", "race-only"}:
        raise ValueError("sessions must be 'all' or 'race-only'")

    _enable_cache(cache_dir)
    records: list[dict[str, Any]] = []
    for year in years:
        schedule = fastf1.get_event_schedule(year, include_testing=include_testing)
        for _, event in schedule.iterrows():
            event_name = str(event["EventName"])
            event_sessions = ["R"]
            if sessions == "all":
                event_sessions = [
                    str(event[column])
                    for column in schedule.columns
                    if re.fullmatch(r"Session\d+", column)
                    and pd.notna(event[column])
                    and str(event[column]).strip()
                ]
                event_sessions = list(dict.fromkeys(event_sessions))

            for session_name in event_sessions:
                record: dict[str, Any] = {
                    "year": year,
                    "event": event_name,
                    "session": session_name,
                    "status": "downloaded",
                }
                try:
                    session = fastf1.get_session(year, event_name, session_name)
                    session.load(
                        telemetry=telemetry,
                        weather=True,
                        messages=True,
                    )
                except Exception as error:  # noqa: BLE001 - preserve per-session progress
                    record["status"] = "failed"
                    record["error"] = str(error)
                    if not skip_errors:
                        raise
                records.append(record)
                if record["status"] == "failed":
                    print(
                        f"skip {year} {event_name} {session_name}: "
                        f"{record['error']}"
                    )
                else:
                    print(f"cached {year} {event_name} {session_name}")
    return records