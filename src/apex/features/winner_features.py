"""Pre-race driver/race features and official-result targets."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from apex.models.degradation import stint_slopes
from apex.models.points import derive_points


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
    "grid_position",
    "grid_rank",
    "grid_gap_to_pole",
    "practice_median_lap_s",
    "practice_best_lap_s",
    "practice_pace_rank",
    "practice_gap_to_best_s",
    "qualifying_median_lap_s",
    "qualifying_best_lap_s",
    "qualifying_pace_rank",
    "qualifying_gap_to_best_s",
    "has_practice_data",
    "has_qualifying_data",
    "practice_track_temp",
    "practice_air_temp",
    "practice_wet_lap_fraction",
    "driver_recent_mean_finish",
    "driver_form_score",
    "driver_finish_rate",
    "driver_dnf_rate",
    "team_recent_mean_finish",
    "team_form_score",
    "team_finish_rate",
    "team_dnf_rate",
    "driver_track_mean_finish",
    "driver_track_start_count",
    "team_track_mean_finish",
    "team_track_start_count",
    "prior_track_safety_car_lap_rate",
    "prior_track_neutralized_lap_rate",
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
    session_pace = clean.assign(
        session_category=np.where(
            clean["session_type"].str.startswith("Practice"),
            "practice",
            np.where(
                clean["session_type"].isin({"Qualifying", "Sprint Qualifying", "Sprint Shootout"}),
                "qualifying",
                "other",
            ),
        )
    )
    session_pace = (
        session_pace[session_pace["session_category"] != "other"]
        .groupby(group + ["session_category"], dropna=False)
        .agg(
            session_median_lap_s=("fuel_corrected_lap_s", "median"),
            session_best_lap_s=("lap_time_s", "min"),
        )
        .reset_index()
    )
    if not session_pace.empty:
        session_pace = session_pace.pivot(
            index=group,
            columns="session_category",
            values=["session_median_lap_s", "session_best_lap_s"],
        )
        session_pace.columns = [
            f"{category}_{metric.removeprefix('session_')}"
            for metric, category in session_pace.columns
        ]
        result = result.merge(session_pace.reset_index(), on=group, how="left")
    for category in ("practice", "qualifying"):
        for metric in ("median_lap_s", "best_lap_s"):
            column = f"{category}_{metric}"
            if column not in result:
                result[column] = np.nan
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
    for category in ("practice", "qualifying"):
        median_column = f"{category}_median_lap_s"
        best_column = f"{category}_best_lap_s"
        result[f"{category}_pace_rank"] = result.groupby("race_id")[median_column].rank(method="min")
        result[f"{category}_gap_to_best_s"] = (
            result[median_column] - result.groupby("race_id")[median_column].transform("min")
        )
        result[f"has_{category}_data"] = result[median_column].notna().astype(int)
    practice = clean[clean["session_type"].str.startswith("Practice")].copy()
    wet_signal = practice["compound"].astype(str).str.upper().isin({"INTERMEDIATE", "WET"})
    for column in ("rainfall", "rainfall_mm", "is_wet", "wet"):
        if column in practice:
            values = practice[column]
            wet_signal = wet_signal | (
                values.astype(bool) if values.dtype == bool
                else pd.to_numeric(values, errors="coerce").fillna(0).gt(0)
            )
    practice_weather = (
        practice.assign(wet_lap=wet_signal)
        .groupby(group, dropna=False)
        .agg(
            practice_track_temp=("track_temp", "mean"),
            practice_air_temp=("air_temp", "mean"),
            practice_wet_lap_fraction=("wet_lap", "mean"),
        )
        .reset_index()
    )
    result = result.merge(practice_weather, on=group, how="left")
    result["prior_track_safety_car_lap_rate"] = np.nan
    result["prior_track_neutralized_lap_rate"] = np.nan
    result["feature_quality"] = np.where(result["valid_laps"] >= 5, "usable", "sparse")
    return result.sort_values(["race_id", "driver"]).reset_index(drop=True)


def build_race_safety_summary(laps: pd.DataFrame) -> pd.DataFrame:
    """Summarize race-session safety-car conditions into a compact table."""
    data = add_session_identity(laps)
    race_laps = data[data["session_type"].eq("Race")].copy()
    if race_laps.empty:
        return pd.DataFrame(
            columns=[
                "race_id",
                "track",
                "race_order",
                "safety_car_lap_rate",
                "neutralized_lap_rate",
            ]
        )
    race_laps["race_order"] = race_laps["season"] * 100 + race_laps["race_id"].str.extract(
        r"_(\d+)$"
    )[0].astype(int)
    statuses = race_laps["track_status"].astype(str)
    race_laps["neutralized_lap"] = statuses.str.contains(r"[4567]")
    race_laps["safety_car_lap"] = statuses.str.contains(r"[46]")
    return (
        race_laps.groupby(["race_id", "race_order", "track"], dropna=False)
        .agg(
            safety_car_lap_rate=("safety_car_lap", "mean"),
            neutralized_lap_rate=("neutralized_lap", "mean"),
        )
        .reset_index()
    )


def attach_prior_safety_features(
    features: pd.DataFrame, safety_summary: pd.DataFrame
) -> pd.DataFrame:
    """Attach prior same-track safety history without target-race leakage."""
    if safety_summary.empty:
        return features
    result = features.copy()
    result["_race_order"] = (
        result["season"] * 100
        + result["race_id"].str.extract(r"_(\d+)$")[0].astype(int)
    )
    values = []
    for _, row in result.iterrows():
        prior = safety_summary[
            (safety_summary["track"].eq(row["track"]))
            & (safety_summary["race_order"] < row["_race_order"])
        ]
        values.append(
            {
                "prior_track_safety_car_lap_rate": prior["safety_car_lap_rate"].mean(),
                "prior_track_neutralized_lap_rate": prior["neutralized_lap_rate"].mean(),
            }
        )
    result[["prior_track_safety_car_lap_rate", "prior_track_neutralized_lap_rate"]] = (
        pd.DataFrame(values, index=result.index)
    )
    return result.drop(columns="_race_order")


def attach_race_results(features: pd.DataFrame, results: pd.DataFrame) -> pd.DataFrame:
    """Join official results and create the winner target."""
    required = {"race_id", "driver", "finish_position", "grid_position", "classified"}
    missing = required - set(results.columns)
    if missing:
        raise ValueError(f"results missing required columns: {sorted(missing)}")
    results = derive_points(results)
    joined = attach_grid_features(features, results)
    result_columns = ["race_id", "driver", "finish_position", "classified"]
    for optional in ("points", "derived_points", "status", "team", "event_type"):
        if optional in results.columns:
            result_columns.append(optional)
    joined = joined.merge(
        results[result_columns],
        on=["race_id", "driver"],
        how="inner",
        validate="one_to_one",
    )
    joined["winner"] = (
        (joined["finish_position"] == 1)
        & joined["classified"].fillna(False)
    ).astype(int)
    joined = attach_historical_features(joined, results)
    return joined


def attach_historical_features(data: pd.DataFrame, results: pd.DataFrame) -> pd.DataFrame:
    """Attach only prior-race driver, team, and circuit history."""
    frame = data.copy()
    history = results.copy()
    if "season" not in history:
        history["season"] = history["race_id"].astype(str).str.extract(r"^(\d{4})$")[0]
        history["season"] = history["season"].fillna(
            history["race_id"].astype(str).str.extract(r"^(\d{4})_")[0]
        ).astype(int)
    if "track" not in history:
        track_map = frame[["race_id", "track"]].drop_duplicates("race_id")
        history = history.merge(track_map, on="race_id", how="left")
    if "team" not in history:
        history["team"] = "UNKNOWN"
    history["team"] = history["team"].fillna("UNKNOWN").astype(str)
    history["race_order"] = (
        history["race_id"].astype(str).str.extract(r"_(\d+)$")[0].astype(int)
        + history["season"].astype(int) * 100
    )
    history["classified_bool"] = history["classified"].fillna(False).astype(bool)
    history["dnf"] = (~history["classified_bool"]).astype(int)
    history["finish_value"] = pd.to_numeric(history["finish_position"], errors="coerce")
    history["finish_value"] = history["finish_value"].where(history["finish_value"].gt(0))
    history["form_value"] = 1.0 / history["finish_value"].clip(lower=1)
    current = frame.merge(
        history[["race_id", "driver", "team"]].drop_duplicates(["race_id", "driver"]),
        on=["race_id", "driver"],
        how="left",
        suffixes=("", "_result"),
    )
    if "team_result" in current:
        current["team"] = current["team"].fillna(current["team_result"])
    current["team"] = current["team"].fillna("UNKNOWN")
    current["race_order"] = (
        current["race_id"].astype(str).str.extract(r"^(\d{4})_")[0].astype(int) * 100
        + current["race_id"].astype(str).str.extract(r"_(\d+)$")[0].astype(int)
    )
    entity_specs = [("driver", "driver"), ("team", "team")]
    for entity, column in entity_specs:
        values = []
        for _, row in current.iterrows():
            prior = history[
                (history[column].eq(row[column]))
                & (history["race_order"] < row["race_order"])
            ].sort_values("race_order")
            recent_orders = prior["race_order"].drop_duplicates().tail(5)
            recent = prior[prior["race_order"].isin(recent_orders)]
            values.append({
                f"{entity}_recent_mean_finish": recent["finish_value"].mean(),
                f"{entity}_form_score": recent["form_value"].mean(),
                f"{entity}_finish_rate": (prior["classified_bool"].sum() + 1.0)
                / (len(prior) + 2.0),
                f"{entity}_dnf_rate": (prior["dnf"].sum() + 1.0)
                / (len(prior) + 2.0),
                f"{entity}_track_mean_finish": (
                    prior[prior["track"].eq(row["track"])]["finish_value"].mean()
                ),
                f"{entity}_track_start_count": float(
                    prior[prior["track"].eq(row["track"])]["finish_value"].count()
                ),
            })
        current = pd.concat(
            [current.reset_index(drop=True), pd.DataFrame(values)],
            axis=1,
        )
    return current.drop(columns=["team_result", "race_order"], errors="ignore")


def attach_grid_features(features: pd.DataFrame, results: pd.DataFrame) -> pd.DataFrame:
    """Attach pre-race grid information without importing race outcomes."""
    required = {"race_id", "driver", "grid_position"}
    missing = required - set(results.columns)
    if missing:
        raise ValueError(f"results missing required columns: {sorted(missing)}")
    grid = results[["race_id", "driver", "grid_position"]].copy()
    grid["grid_position"] = pd.to_numeric(grid["grid_position"], errors="coerce")
    grid["grid_position"] = grid["grid_position"].where(grid["grid_position"].gt(0))
    joined = features.drop(
        columns=["grid_position", "grid_rank", "grid_gap_to_pole"],
        errors="ignore",
    ).merge(grid, on=["race_id", "driver"], how="left", validate="one_to_one")
    grid_group = joined.groupby("race_id")["grid_position"]
    joined["grid_rank"] = grid_group.rank(method="min")
    joined["grid_gap_to_pole"] = joined["grid_position"] - grid_group.transform("min")
    return joined


def validate_race_targets(data: pd.DataFrame) -> None:
    """Ensure every race has exactly one classified winner."""
    counts = data.groupby("race_id")["winner"].sum()
    bad = counts[counts != 1]
    if not bad.empty:
        raise ValueError(f"races without exactly one classified winner: {bad.index.tolist()[:5]}")
