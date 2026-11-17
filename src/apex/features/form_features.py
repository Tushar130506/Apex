"""Race-relative and prior-race form features for the finish-order model.

Every feature here is either

* computed inside one race weekend from sessions that end before the race
  (qualifying, practice, the starting grid), or
* computed from races strictly earlier than the target race.

Target rows may have missing results (a race that has not happened yet); they
still receive all features, which is what makes genuine pre-race prediction
possible.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from apex.models.points import GRAND_PRIX_POINTS

# Constructor lineages: a renamed team keeps its factory, staff and car
# concept, so its history must follow it across the rename.
TEAM_LINEAGE = {
    "Toro Rosso": "Racing Bulls",
    "AlphaTauri": "Racing Bulls",
    "RB": "Racing Bulls",
    "Force India": "Aston Martin",
    "Racing Point": "Aston Martin",
    "Renault": "Alpine",
    "Sauber": "Audi",
    "Alfa Romeo Racing": "Audi",
    "Alfa Romeo": "Audi",
    "Kick Sauber": "Audi",
    # Names used by the results service when timing data is unavailable.
    "Red Bull": "Red Bull Racing",
    "RB F1 Team": "Racing Bulls",
    "Alpine F1 Team": "Alpine",
    "Cadillac F1 Team": "Cadillac",
}
MISSING_TEAM_LABELS = {"", "None", "nan", "NaN", "UNKNOWN", "<NA>"}
FINISHED_PREFIXES = ("Finished", "Lapped", "+")

# Features that need a completed qualifying session / known grid.
WEEKEND_COLUMNS = [
    "grid_filled",
    "grid_log",
    "front_row",
    "quali_gap_pct",
    "quali_rank",
    "best_lap_gap_pct",
    "practice_gap_pct",
    "practice_rank",
    "degradation_rel",
    "team_quali_gap_pct_now",
    "grid_vs_team_form",
    "practice_wet_lap_fraction",
    "prior_track_safety_car_lap_rate",
]
# Features available before the weekend starts (prior races only).
FORM_COLUMNS = [
    "driver_points_fast",
    "driver_points_slow",
    "driver_finish_slow",
    "driver_win_rate",
    "driver_podium_rate",
    "driver_grid_fast",
    "driver_quali_gap_slow",
    "driver_gain_slow",
    "driver_dnf_slow",
    "driver_vs_teammate_quali",
    "driver_starts",
    "driver_season_points_share",
    "driver_track_finish",
    "team_points_fast",
    "team_points_slow",
    "team_best_finish_fast",
    "team_quali_gap_fast",
    "team_quali_gap_slow",
    "team_win_rate",
    "team_dnf_slow",
    "team_season_points_share",
    "driver_points_fast_rank",
    "team_quali_gap_fast_rank",
]
ORDER_MODEL_COLUMNS = WEEKEND_COLUMNS + FORM_COLUMNS

FAST_HALFLIFE = 3.0
SLOW_HALFLIFE = 10.0
GAP_CAP_PCT = 4.0


def canonical_team(team: pd.Series) -> pd.Series:
    """Map team labels onto one name per constructor lineage."""
    cleaned = team.astype("object").where(team.notna(), None)
    cleaned = cleaned.map(lambda value: None if value is None else str(value).strip())
    cleaned = cleaned.where(~cleaned.isin(MISSING_TEAM_LABELS), None)
    return cleaned.map(lambda value: TEAM_LINEAGE.get(value, value) if value else None)


def race_order(race_id: pd.Series) -> pd.Series:
    """Chronological integer key from ``YYYY_round`` race identifiers."""
    parts = race_id.astype(str).str.extract(r"^(?P<season>\d{4})_(?P<round>\d+)$")
    if parts.isna().any().any():
        raise ValueError("race_id values must have the form YYYY_round")
    return parts["season"].astype(int) * 100 + parts["round"].astype(int)


def finished_from_status(status: pd.Series) -> pd.Series:
    """True when the car was running at the flag.

    The cached ``classified`` flag is true for every row, including
    retirements, so reliability has to be read from the status text instead.
    """
    text = status.astype("object").where(status.notna(), "").astype(str)
    return text.str.startswith(FINISHED_PREFIXES)


def _fill_missing_teams(frame: pd.DataFrame) -> pd.Series:
    """Resolve unlabeled rows from the driver's nearest labeled race."""
    team = canonical_team(frame["team"]) if "team" in frame else pd.Series(None, index=frame.index)
    ordered = frame.assign(_team=team).sort_values("_order")
    filled = ordered.groupby("driver")["_team"].transform(lambda values: values.ffill().bfill())
    return filled.reindex(frame.index).fillna("UNKNOWN")


def _prior_ewm(frame: pd.DataFrame, key: str, value: str, halflife: float) -> pd.Series:
    """Exponentially weighted mean of earlier rows only (target row excluded)."""
    ordered = frame.sort_values("_order")
    result = ordered.groupby(key, sort=False)[value].transform(
        lambda values: values.shift(1).ewm(halflife=halflife, min_periods=1, ignore_na=True).mean()
    )
    return result.reindex(frame.index)


def _prior_mean(frame: pd.DataFrame, keys: list[str], value: str) -> pd.Series:
    ordered = frame.sort_values("_order")
    result = ordered.groupby(keys, sort=False)[value].transform(
        lambda values: values.shift(1).expanding(min_periods=1).mean()
    )
    return result.reindex(frame.index)


def _prior_sum(frame: pd.DataFrame, keys: list[str], value: str) -> pd.Series:
    ordered = frame.sort_values("_order")
    result = ordered.groupby(keys, sort=False)[value].transform(
        lambda values: values.shift(1).fillna(0.0).cumsum()
    )
    return result.reindex(frame.index)


def _gap_pct(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(np.nan, index=frame.index)
    values = pd.to_numeric(frame[column], errors="coerce")
    best = values.groupby(frame["race_id"]).transform("min")
    return ((values / best - 1.0) * 100.0).clip(upper=GAP_CAP_PCT)


def build_order_features(table: pd.DataFrame) -> pd.DataFrame:
    """Add :data:`ORDER_MODEL_COLUMNS` to a driver/race table.

    ``table`` needs ``race_id``, ``driver``, ``team`` and ``grid_position``.
    Result columns (``finish_position``, ``status``) may be missing or NaN for
    races that have not been run; such rows contribute nothing to history.
    """
    required = {"race_id", "driver"}
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"table missing columns: {sorted(missing)}")
    if table.duplicated(["race_id", "driver"]).any():
        raise ValueError("table must have one row per race and driver")
    frame = table.copy().reset_index(drop=True)
    frame["_order"] = race_order(frame["race_id"])
    frame["season"] = frame["_order"] // 100
    frame["team_canonical"] = _fill_missing_teams(frame)
    for column in ("finish_position", "grid_position", "status", "track"):
        if column not in frame:
            frame[column] = np.nan
    race = frame.groupby("race_id")
    field_size = race["driver"].transform("size")

    # ---- race weekend (pre-race) -------------------------------------------
    grid = pd.to_numeric(frame["grid_position"], errors="coerce")
    grid = grid.where(grid.gt(0))
    # Pit-lane starts carry no grid slot: they start behind the whole field.
    frame["grid_filled"] = grid.fillna(field_size + 1.0)
    has_grid = grid.groupby(frame["race_id"]).transform("count").gt(0)
    frame.loc[~has_grid, "grid_filled"] = np.nan
    frame["grid_log"] = np.log(frame["grid_filled"])
    frame["front_row"] = frame["grid_filled"].le(2).astype(float).where(has_grid)
    frame["quali_gap_pct"] = _gap_pct(frame, "qualifying_best_lap_s")
    frame["quali_rank"] = frame.groupby("race_id")["quali_gap_pct"].rank(method="min")
    frame["best_lap_gap_pct"] = _gap_pct(frame, "best_lap_time_s")
    frame["practice_gap_pct"] = _gap_pct(frame, "practice_median_lap_s")
    frame["practice_rank"] = frame.groupby("race_id")["practice_gap_pct"].rank(method="min")
    if "degradation_s_per_lap" in frame:
        degradation = pd.to_numeric(frame["degradation_s_per_lap"], errors="coerce").clip(-0.5, 0.5)
        frame["degradation_rel"] = degradation - degradation.groupby(frame["race_id"]).transform("median")
    else:
        frame["degradation_rel"] = np.nan
    frame["team_quali_gap_pct_now"] = frame.groupby(["race_id", "team_canonical"])[
        "quali_gap_pct"
    ].transform("min")
    for column in ("practice_wet_lap_fraction", "prior_track_safety_car_lap_rate"):
        if column not in frame:
            frame[column] = np.nan

    # ---- outcomes of completed races (used only through shifted windows) ---
    finish = pd.to_numeric(frame["finish_position"], errors="coerce")
    finish = finish.where(finish.gt(0))
    completed = finish.notna()
    points_scale = np.zeros(64)
    points_scale[1 : len(GRAND_PRIX_POINTS) + 1] = GRAND_PRIX_POINTS
    frame["_finish"] = finish
    frame["_points"] = pd.Series(
        points_scale[finish.fillna(0).astype(int).clip(upper=63)], index=frame.index
    ).where(completed)
    frame["_win"] = finish.eq(1).astype(float).where(completed)
    frame["_podium"] = finish.le(3).astype(float).where(completed)
    frame["_dnf"] = (~finished_from_status(frame["status"])).astype(float).where(completed)
    frame["_grid"] = frame["grid_filled"].where(completed)
    frame["_gain"] = (frame["grid_filled"] - finish).where(completed)
    frame["_quali_gap"] = frame["quali_gap_pct"]
    frame["_started"] = completed.astype(float)

    # ---- driver form -------------------------------------------------------
    frame["driver_points_fast"] = _prior_ewm(frame, "driver", "_points", FAST_HALFLIFE)
    frame["driver_points_slow"] = _prior_ewm(frame, "driver", "_points", SLOW_HALFLIFE)
    frame["driver_finish_slow"] = _prior_ewm(frame, "driver", "_finish", SLOW_HALFLIFE)
    frame["driver_win_rate"] = _prior_ewm(frame, "driver", "_win", SLOW_HALFLIFE)
    frame["driver_podium_rate"] = _prior_ewm(frame, "driver", "_podium", SLOW_HALFLIFE)
    frame["driver_grid_fast"] = _prior_ewm(frame, "driver", "_grid", FAST_HALFLIFE)
    frame["driver_quali_gap_slow"] = _prior_ewm(frame, "driver", "_quali_gap", SLOW_HALFLIFE)
    frame["driver_gain_slow"] = _prior_ewm(frame, "driver", "_gain", SLOW_HALFLIFE)
    frame["driver_dnf_slow"] = _prior_ewm(frame, "driver", "_dnf", 2 * SLOW_HALFLIFE)
    frame["driver_starts"] = np.log1p(_prior_sum(frame, ["driver"], "_started"))
    team_mean_gap = frame.groupby(["race_id", "team_canonical"])["quali_gap_pct"].transform("mean")
    frame["_vs_teammate"] = frame["quali_gap_pct"] - team_mean_gap
    frame["driver_vs_teammate_quali"] = _prior_ewm(frame, "driver", "_vs_teammate", SLOW_HALFLIFE)
    track_mean = _prior_mean(frame, ["driver", "track"], "_finish")
    track_count = _prior_sum(frame, ["driver", "track"], "_started")
    # Shrink sparse circuit history towards the driver's general level.
    baseline = frame["driver_finish_slow"]
    frame["driver_track_finish"] = (
        (track_mean.fillna(baseline) * track_count + baseline * 2.0) / (track_count + 2.0)
    )

    # ---- team form (one row per team and race, then shifted) ---------------
    team_race = (
        frame.groupby(["race_id", "team_canonical"], as_index=False)
        .agg(
            _order=("_order", "first"),
            season=("season", "first"),
            _team_points=("_points", "sum"),
            _team_completed=("_started", "max"),
            _team_best_finish=("_finish", "min"),
            _team_quali_gap=("quali_gap_pct", "min"),
            _team_win=("_win", "max"),
            _team_dnf=("_dnf", "mean"),
        )
    )
    team_race.loc[team_race["_team_completed"].ne(1.0), "_team_points"] = np.nan
    team_race["team_points_fast"] = _prior_ewm(team_race, "team_canonical", "_team_points", FAST_HALFLIFE)
    team_race["team_points_slow"] = _prior_ewm(team_race, "team_canonical", "_team_points", SLOW_HALFLIFE)
    team_race["team_best_finish_fast"] = _prior_ewm(team_race, "team_canonical", "_team_best_finish", FAST_HALFLIFE)
    team_race["team_quali_gap_fast"] = _prior_ewm(team_race, "team_canonical", "_team_quali_gap", FAST_HALFLIFE)
    team_race["team_quali_gap_slow"] = _prior_ewm(team_race, "team_canonical", "_team_quali_gap", SLOW_HALFLIFE)
    team_race["team_win_rate"] = _prior_ewm(team_race, "team_canonical", "_team_win", SLOW_HALFLIFE)
    team_race["team_dnf_slow"] = _prior_ewm(team_race, "team_canonical", "_team_dnf", 2 * SLOW_HALFLIFE)
    team_race["_team_season_points"] = _prior_sum(team_race, ["team_canonical", "season"], "_team_points")
    team_columns = [
        "team_points_fast", "team_points_slow", "team_best_finish_fast", "team_quali_gap_fast",
        "team_quali_gap_slow", "team_win_rate", "team_dnf_slow", "_team_season_points",
    ]
    frame = frame.merge(
        team_race[["race_id", "team_canonical"] + team_columns],
        on=["race_id", "team_canonical"],
        how="left",
        validate="many_to_one",
    )

    # ---- season-to-date shares --------------------------------------------
    driver_season_points = _prior_sum(frame, ["driver", "season"], "_points")
    race = frame.groupby("race_id")
    total = driver_season_points.groupby(frame["race_id"]).transform("sum")
    frame["driver_season_points_share"] = (driver_season_points / total).where(total.gt(0), 0.0)
    team_total = frame.drop_duplicates(["race_id", "team_canonical"]).groupby("race_id")[
        "_team_season_points"
    ].sum()
    team_total = frame["race_id"].map(team_total)
    frame["team_season_points_share"] = (frame["_team_season_points"] / team_total).where(
        team_total.gt(0), 0.0
    )

    # ---- race-relative views of form --------------------------------------
    frame["driver_points_fast_rank"] = race["driver_points_fast"].rank(method="min", ascending=False)
    frame["team_quali_gap_fast_rank"] = race["team_quali_gap_fast"].rank(method="dense")
    frame["grid_vs_team_form"] = frame["grid_filled"] - frame["driver_grid_fast"]

    helper = [column for column in frame.columns if column.startswith("_")]
    return frame.drop(columns=helper)


def results_table(table: pd.DataFrame) -> pd.DataFrame:
    """Return completed-race rows only, with a status-derived finish flag."""
    finish = pd.to_numeric(table["finish_position"], errors="coerce")
    result = table[finish.gt(0)].copy()
    result["finished"] = finished_from_status(result["status"])
    return result
