"""Leakage-safe historical race replay state construction."""
from __future__ import annotations

import pandas as pd


def replay_snapshots(laps: pd.DataFrame) -> pd.DataFrame:
    """Build one observable race snapshot per lap from already observed rows.

    Rows at lap ``t`` contain only data with ``lap_no <= t``. Pit-stop and
    safety-car counts are therefore usable for replay validation without
    reading future race events.
    """
    frame = laps.copy()
    if "race_id" not in frame and "session_id" in frame:
        frame["race_id"] = frame["session_id"]
    required = {"race_id", "lap_no", "driver", "lap_time_s", "compound", "tyre_age"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"laps missing columns: {sorted(missing)}")
    if "is_pit_lap" not in frame:
        frame["is_pit_lap"] = False
    if "track_status" not in frame:
        frame["track_status"] = ""
    frame["is_pit_lap"] = frame["is_pit_lap"].fillna(False).astype(bool)
    frame["track_status"] = frame["track_status"].fillna("").astype(str)
    frame["safety_car_lap"] = frame["track_status"].str.contains(r"[46]", regex=True)
    frame["neutralized_lap"] = frame["track_status"].str.contains(r"[4567]", regex=True)
    snapshots: list[dict] = []
    for (race_id, lap_no), observed in frame.groupby(["race_id", "lap_no"], sort=True):
        history = frame[(frame["race_id"] == race_id) & (frame["lap_no"] <= lap_no)]
        latest = (
            observed.sort_values(["lap_time_s", "driver"], na_position="last")
            .drop_duplicates("driver")
            .sort_values("lap_time_s", na_position="last")
        )
        snapshots.append(
            {
                "race_id": race_id,
                "lap_no": int(lap_no),
                "observed_drivers": int(latest["driver"].nunique()),
                "pit_stops_observed": int(history["is_pit_lap"].sum()),
                "safety_car_laps_observed": int(history["safety_car_lap"].sum()),
                "neutralized_laps_observed": int(history["neutralized_lap"].sum()),
                "current_compounds": latest.set_index("driver")["compound"].to_dict(),
                "current_tyre_age": latest.set_index("driver")["tyre_age"].to_dict(),
                "current_lap_time_s": latest.set_index("driver")["lap_time_s"].to_dict(),
            }
        )
    return pd.DataFrame(snapshots)
