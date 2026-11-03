"""Leakage-aware race-level features for winner prediction.

These features describe observed pace. They are not a winner target and should
only be used for pre-race prediction after the source sessions are verified as
available before the prediction timestamp.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from apex.models.degradation import stint_slopes


REQUIRED_COLUMNS = {
    "season",
    "session_id",
    "track",
    "driver",
    "lap_time_s",
    "valid",
    "compound",
    "tyre_age",
    "stint_id",
    "fuel_kg",
    "fuel_provenance",
    "is_pit_lap",
}


def build_race_features(
    laps: pd.DataFrame,
    fuel_s_per_kg: float = 0.032,
    min_valid_laps: int = 5,
    race_key: str = "race_id",
) -> pd.DataFrame:
    """Aggregate clean lap and stint information into one row per driver/group.

    ``race_key`` is preferred when present. Falling back to ``session_id`` is
    deliberate: it prevents accidental merging of practice and race sessions.
    The degradation estimate is descriptive and must not be used as a pre-race
    feature unless it comes from an earlier session.
    """
    missing = REQUIRED_COLUMNS - set(laps.columns)
    if missing:
        raise ValueError(f"laps missing required columns: {sorted(missing)}")
    if fuel_s_per_kg < 0:
        raise ValueError("fuel_s_per_kg must be non-negative")
    if min_valid_laps < 1:
        raise ValueError("min_valid_laps must be positive")

    key = race_key if race_key in laps.columns else "session_id"
    clean = laps[laps["valid"].fillna(False) & ~laps["is_pit_lap"].fillna(False)].copy()
    clean = clean.dropna(subset=["lap_time_s", "fuel_kg", "tyre_age"])
    group_columns = [key, "season", "track", "driver"]
    grouped = clean.groupby(group_columns, dropna=False, sort=False)
    result = grouped.agg(
        valid_laps=("lap_time_s", "size"),
        stints=("stint_id", "nunique"),
        median_lap_time_s=("lap_time_s", "median"),
        best_lap_time_s=("lap_time_s", "min"),
    ).reset_index()
    clean["fuel_corrected_lap_s"] = clean["lap_time_s"] - fuel_s_per_kg * clean["fuel_kg"]
    corrected = clean.groupby(group_columns, dropna=False)["fuel_corrected_lap_s"].median()
    result = result.merge(corrected.rename("fuel_corrected_median_lap_s"), on=group_columns, how="left")

    slopes = stint_slopes(clean, fuel_s_per_kg=fuel_s_per_kg)
    if slopes.empty:
        result["degradation_s_per_lap"] = np.nan
        result["degradation_stints"] = 0
    else:
        slope_group = slopes.groupby(["session_id", "driver"], as_index=False).agg(
            degradation_s_per_lap=("slope_s_per_lap", "median"),
            degradation_stints=("slope_s_per_lap", "size"),
        )
        if key == "session_id":
            result = result.merge(slope_group, on=["session_id", "driver"], how="left")
        else:
            session_map = clean[[key, "session_id", "driver"]].drop_duplicates()
            slope_group = session_map.merge(slope_group, on=["session_id", "driver"], how="inner")
            slope_group = slope_group.groupby([key, "driver"], as_index=False).agg(
                degradation_s_per_lap=("degradation_s_per_lap", "median"),
                degradation_stints=("degradation_stints", "sum"),
            )
            result = result.merge(slope_group, on=[key, "driver"], how="left")

    provenance = clean.groupby(group_columns, dropna=False)["fuel_provenance"].agg(
        estimated_fuel_fraction=lambda values: float((values == "estimated").mean())
    ).reset_index()
    result = result.merge(provenance, on=group_columns, how="left")
    result["fuel_sensitivity_s_per_kg"] = float(fuel_s_per_kg)
    result["feature_quality"] = np.where(result["valid_laps"] >= min_valid_laps, "usable", "sparse")
    return result.sort_values(group_columns).reset_index(drop=True)
