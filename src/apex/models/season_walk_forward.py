"""Expanding race-by-race forecasting across multiple seasons."""
from __future__ import annotations

import pandas as pd

from apex.features.form_features import ORDER_MODEL_COLUMNS, build_order_features, race_order
from apex.models.forecast import form_history, season_outlook
from apex.models.race_order import walk_forward_order
from apex.models.walk_forward import MIN_TRAINING_RACES


def expanding_season_predictions(
    data: pd.DataFrame,
    *,
    seed_seasons: list[int],
    forecast_seasons: list[int],
    model_columns: list[str] | None = None,
    championship_simulations: int = 2000,
    calibration_races: int = 40,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Predict each race after the seed seasons and the title race before it.

    For every target race two things are produced from races strictly before
    it: the race forecast itself, and a championship checkpoint that simulates
    that race and every later round of the season. The checkpoint never sees
    the pace, grid or result of any race it simulates; only the calendar
    (race ids and circuits) and the entry list of the upcoming round are used.
    """
    required = {"race_id", "driver", "team", "finish_position"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"walk-forward data missing columns: {sorted(missing)}")
    table = data.copy()
    table["_order"] = race_order(table["race_id"])
    table["season"] = table["_order"] // 100
    seed_limit = table.loc[table["season"].isin(seed_seasons), "_order"].max()
    if pd.isna(seed_limit):
        raise ValueError("no races in the seed seasons")
    all_orders = sorted(table["_order"].unique())
    targets = [
        value for value in all_orders
        if value > seed_limit and value // 100 in set(forecast_seasons)
    ]
    if not targets:
        raise ValueError("no races available after the seed seasons")
    features = build_order_features(table.drop(columns="_order"))
    columns = list(model_columns or ORDER_MODEL_COLUMNS)
    earlier = [value for value in all_orders if value < targets[0]]
    warmup = [
        value for position, value in enumerate(earlier) if position >= MIN_TRAINING_RACES
    ][-calibration_races:]
    scored = walk_forward_order(features, warmup + targets, columns=columns, seed=seed)
    scored["_order"] = race_order(scored["race_id"])
    predictions = scored[scored["_order"].isin(targets)].copy()
    predictions["prediction_order"] = predictions["_order"].rank(method="dense").astype(int)

    history = form_history(features, warmup, seed=seed) if warmup else []
    checkpoints: list[pd.DataFrame] = []
    for number, target_order in enumerate(targets, start=1):
        season = target_order // 100
        known = table[table["_order"].lt(target_order)].drop(columns="_order")
        later = table[table["_order"].ge(target_order) & table["season"].eq(season)]
        remaining = later[["race_id", "track"]].drop_duplicates("race_id") if "track" in later else (
            later[["race_id"]].drop_duplicates()
        )
        roster = later[later["_order"].eq(target_order)][["driver", "team"]]
        outlook = season_outlook(
            known,
            season,
            remaining,
            roster=roster,
            history=history,
            simulations=championship_simulations,
            seed=seed + number,
        )
        board = outlook.championship.driver.copy()
        board["before_race_id"] = str(later.loc[later["_order"].eq(target_order), "race_id"].iloc[0])
        board["season"] = season
        board["prediction_order"] = number
        board["remaining_races"] = len(remaining)
        board["form_temperature"] = outlook.temperature
        board["form_drift_per_race"] = outlook.form_drift_per_race
        checkpoints.append(board)
        history = history + form_history(features, [target_order], seed=seed)
    return (
        predictions.drop(columns="_order").reset_index(drop=True),
        pd.concat(checkpoints, ignore_index=True),
    )
