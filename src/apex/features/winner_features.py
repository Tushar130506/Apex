"""Pre-race driver/race features and official-result targets."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from apex.models.degradation import stint_slopes


SESSION_RE = re.compile(r"^fastf1_(?P<season>\d{4})_(?P<round>\d+)_(?P<session>.+)$")
PRE_RACE_SESSIONS = {"Practice 1", "Practice 2", "Practice 3", "Qualifying", "Sprint Qualifying", "Sprint Shootout"}
MODEL_COLUMNS = [
    "fuel_corrected_median_lap_s",
    "best_lap_time_s",
    "pace_rank",
    "pace_gap_to_best_s",
    "best_lap_rank",
    "best_lap_gap_to_best_s",
    "degradation_s_per_lap",
    "valid_laps",
    "session_count",
]


def add_session_identity(laps: pd.DataFrame) -> pd.DataFrame:
    """Add race and session labels parsed from the repository's FastF1 IDs."""
    parsed = laps["session_id"].astype(str).str.extract(SESSION_RE)
    if parsed.isna().any().any():
        bad = laps.loc[parsed.isna().any(axis=1), "session_id"].head(3).tolist()
        raise ValueError(f"unrecognised FastF1 session_id values: {bad}")
    result = laps.copy()
    result["season"] = parsed["season"].astype(int)
    result["race_id"] = parsed["season"] + "_" + parsed["round"]
    result["session_type"] = parsed["session"]
    return result


def build_pre_race_features(laps: pd.DataFrame, fuel_s_per_kg: float = 0.032) -> pd.DataFrame:
    """Aggregate only sessions available before a race starts."""
    if fuel_s_per_kg < 0:
        raise ValueError("fuel_s_per_kg must be non-negative")
    data = add_session_identity(laps)
    data = data[data["session_type"].isin(PRE_RACE_SESSIONS)].copy()
    clean = data[data["valid"].fillna(False) & ~data["is_pit_lap"].fillna(False)].dropna(
        subset=["lap_time_s", "fuel_kg", "tyre_age"]
    )
    group = ["race_id", "season", "track", "driver"]
    if clean.empty:
        raise ValueError("no valid pre-race laps available")
    clean["fuel_corrected_lap_s"] = clean["lap_time_s"] - fuel_s_per_kg * clean["fuel_kg"]
    result = clean.groupby(group, dropna=False).agg(
        fuel_corrected_median_lap_s=("fuel_corrected_lap_s", "median"),
        best_lap_time_s=("lap_time_s", "min"),
        valid_laps=("lap_time_s", "size"),
        session_count=("session_id", "nunique"),
    ).reset_index()
    session_counts = (
        clean.assign(
            practice_session=clean["session_type"].str.startswith("Practice"),
            qualifying_session=clean["session_type"].isin({"Qualifying", "Sprint Qualifying", "Sprint Shootout"}),
        )
        .groupby(group, dropna=False)
        .agg(
            practice_laps=("practice_session", "sum"),
            qualifying_laps=("qualifying_session", "sum"),
            practice_sessions=("session_type", lambda values: values[values.str.startswith("Practice")].nunique()),
            qualifying_sessions=("session_type", lambda values: values[values.isin({"Qualifying", "Sprint Qualifying", "Sprint Shootout"})].nunique()),
        )
        .reset_index()
    )
    result = result.merge(session_counts, on=group, how="left")
    slopes = stint_slopes(clean, fuel_s_per_kg=fuel_s_per_kg)
    if slopes.empty:
        result["degradation_s_per_lap"] = np.nan
    else:
        slope = slopes.merge(clean[["session_id", "race_id"]].drop_duplicates(), on="session_id")
        slope = slope.groupby(["race_id", "driver"], as_index=False)["slope_s_per_lap"].median()
        result = result.merge(slope.rename(columns={"slope_s_per_lap": "degradation_s_per_lap"}),
                              on=["race_id", "driver"], how="left")
    result["fuel_sensitivity_s_per_kg"] = float(fuel_s_per_kg)
    race_group = result.groupby("race_id")
    result["pace_rank"] = race_group["fuel_corrected_median_lap_s"].rank(method="min")
    result["pace_gap_to_best_s"] = (
        result["fuel_corrected_median_lap_s"]
        - race_group["fuel_corrected_median_lap_s"].transform("min")
    )
    result["best_lap_rank"] = race_group["best_lap_time_s"].rank(method="min")
    result["best_lap_gap_to_best_s"] = (
        result["best_lap_time_s"] - race_group["best_lap_time_s"].transform("min")
    )
    result["feature_quality"] = np.where(result["valid_laps"] >= 5, "usable", "sparse")
    return result.sort_values(["race_id", "driver"]).reset_index(drop=True)


def attach_race_results(features: pd.DataFrame, results: pd.DataFrame) -> pd.DataFrame:
    """Join official results and create the winner target."""
    required = {"race_id", "driver", "finish_position", "grid_position", "classified"}
    missing = required - set(results.columns)
    if missing:
        raise ValueError(f"results missing required columns: {sorted(missing)}")
    result_columns = ["race_id", "driver", "finish_position", "grid_position", "classified"]
    for optional in ("points", "status"):
        if optional in results.columns:
            result_columns.append(optional)
    joined = features.merge(
        results[result_columns],
        on=["race_id", "driver"],
        how="inner",
        validate="one_to_one",
    )
    joined["winner"] = (
        (joined["finish_position"] == 1)
        & joined["classified"].fillna(False)
    ).astype(int)
    return joined


def validate_race_targets(data: pd.DataFrame) -> None:
    """Ensure every race has exactly one classified winner."""
    counts = data.groupby("race_id")["winner"].sum()
    bad = counts[counts != 1]
    if not bad.empty:
        raise ValueError(f"races without exactly one classified winner: {bad.index.tolist()[:5]}")
