"""Expanding race-by-race forecasting across multiple seasons."""
from __future__ import annotations

import numpy as np
import pandas as pd

from apex.models.championship import forecast_championship
from apex.models.walk_forward import make_walk_forward_model
from apex.models.race_monte_carlo import monte_carlo_race_probabilities


RACE_SPECIFIC_COLUMNS = {
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
}


def _cutoff_future_features(
    target: pd.DataFrame, completed: pd.DataFrame, model_columns: list[str]
) -> pd.DataFrame:
    """Replace future-race evidence with state known at the forecast cutoff."""
    result = target.copy()
    latest = (
        completed.sort_values("_race_order")
        .drop_duplicates("driver", keep="last")
        .set_index("driver")
    )
    result["team"] = result["driver"].map(latest["team"]).fillna("UNKNOWN")
    for column in model_columns:
        if column in RACE_SPECIFIC_COLUMNS:
            result[column] = np.nan
            continue
        if column in latest:
            result[column] = result["driver"].map(latest[column])
    return result


def _race_order(frame: pd.DataFrame) -> pd.Series:
    parts = frame["race_id"].astype(str).str.extract(r"^(?P<season>\d{4})_(?P<round>\d+)$")
    if parts.isna().any().any():
        raise ValueError("race_id values must have the form YYYY_round")
    return parts["season"].astype(int) * 100 + parts["round"].astype(int)


def expanding_season_predictions(
    data: pd.DataFrame,
    *,
    seed_seasons: list[int],
    forecast_seasons: list[int],
    model_columns: list[str],
    championship_simulations: int = 2000,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Predict each future race, appending its actual rows before the next.

    The returned checkpoint table contains WDC probabilities after every
    predicted race. Championship probabilities use predictions generated from
    that checkpoint's training cutoff, not later predictions.
    """
    required = {"season", "race_id", "driver", "team", "winner", "finish_position", "points"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"walk-forward data missing columns: {sorted(missing)}")
    missing_features = set(model_columns) - set(data.columns)
    if missing_features:
        raise ValueError(f"walk-forward data missing model columns: {sorted(missing_features)}")

    frame = data.copy()
    frame["_race_order"] = _race_order(frame)
    seed_limit = max(frame.loc[frame["season"].isin(seed_seasons), "_race_order"])
    target_orders = sorted(
        frame.loc[
            frame["season"].isin(forecast_seasons)
            & frame["_race_order"].gt(seed_limit),
            "_race_order",
        ].unique()
    )
    if not target_orders:
        raise ValueError("no races available after the seed seasons")

    completed = frame[frame["_race_order"].le(seed_limit)].copy()
    prediction_rows: list[pd.DataFrame] = []
    checkpoint_rows: list[pd.DataFrame] = []
    for race_number, race_order in enumerate(target_orders, start=1):
        target = frame[frame["_race_order"].eq(race_order)].copy()
        championship_season = int(target["season"].iloc[0])
        model = make_walk_forward_model(seed + race_number - 1)
        model.fit(completed[model_columns], completed["winner"])
        target["score"] = model.predict_proba(target[model_columns])[:, 1]
        target["race_probability"] = monte_carlo_race_probabilities(
            target, target["score"].to_numpy(), simulations=1000, seed=seed + race_number
        )
        target["predicted_position"] = (
            target["race_probability"].rank(method="first", ascending=False).astype(int)
        )
        target["training_races"] = int(completed["race_id"].nunique())
        target["prediction_order"] = race_number
        prediction_rows.append(target.drop(columns="_race_order"))

        completed_after = pd.concat(
            [
                completed,
                target[
                    [
                        column
                        for column in completed.columns
                        if column in target.columns
                    ]
                ],
            ],
            ignore_index=True,
        )
        completed_season = completed_after[
            completed_after["season"].eq(championship_season)
        ]
        completed_columns = ["race_id", "driver", "team", "points"]
        if "classified" in completed_season:
            completed_columns.extend(["finish_position", "classified"])
        completed_with_points = completed_season[completed_columns].copy()
        completed_with_points["team"] = (
            completed_with_points["team"]
            .replace({"None": pd.NA, "nan": pd.NA})
            .fillna("UNKNOWN")
        )
        future_predictions = frame[
            frame["_race_order"].gt(race_order)
            & frame["season"].eq(championship_season)
        ].copy()
        if not future_predictions.empty:
            future_predictions = _cutoff_future_features(
                future_predictions, completed_after, model_columns
            )
            future_predictions["score"] = model.predict_proba(
                future_predictions[model_columns]
            )[:, 1]
            future_predictions = future_predictions[
                ["race_id", "driver", "team", "score"]
            ]
        if future_predictions.empty:
            future_predictions = pd.DataFrame(
                columns=["race_id", "driver", "team", "score"]
            )
        team_lookup = (
            completed_after.assign(
                team=completed_after["team"].replace({"None": pd.NA, "nan": pd.NA})
            )
            .sort_values("_race_order")
            .dropna(subset=["team"])
            .drop_duplicates("driver", keep="last")
            .set_index("driver")["team"]
        )
        future_predictions["team"] = (
            future_predictions["team"]
            .replace({"None": pd.NA, "nan": pd.NA, "": pd.NA})
            .fillna(future_predictions["driver"].map(team_lookup))
            .fillna("UNKNOWN")
        )
        forecast = forecast_championship(
            completed_with_points,
            future_predictions,
            simulations=championship_simulations,
            seed=seed + race_number - 1,
        )
        table = forecast.driver.copy()
        table["after_race_id"] = str(target["race_id"].iloc[0])
        table["prediction_order"] = race_number
        checkpoint_rows.append(table)

        completed = completed_after

    return (
        pd.concat(prediction_rows, ignore_index=True),
        pd.concat(checkpoint_rows, ignore_index=True),
    )
