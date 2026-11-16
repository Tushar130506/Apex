"""Race-by-race walk-forward winner evaluation."""
from __future__ import annotations

import numpy as np
import pandas as pd

from apex.features.form_features import ORDER_MODEL_COLUMNS, build_order_features, race_order
from apex.models.plackett_luce import LinearPlackettLuce, build_layout, win_probabilities
from apex.models.race_order import evaluate_predictions, walk_forward_order

MIN_TRAINING_RACES = 20


def baseline_pick(data: pd.DataFrame, column: str, ascending: bool) -> pd.Series:
    """One deterministic pick per race from a single column.

    Races where the column is missing for everyone get no pick. Level values
    are separated by grid position and then driver code, never by row order.
    """
    if column not in data:
        return pd.Series(dtype=object)
    frame = data[["race_id", "driver"]].copy()
    frame["_value"] = pd.to_numeric(data[column], errors="coerce")
    if column == "grid_position":
        frame["_value"] = frame["_value"].where(frame["_value"].gt(0))
    grid = data["grid_position"] if "grid_position" in data else pd.Series(np.nan, index=data.index)
    frame["_grid"] = pd.to_numeric(grid, errors="coerce").fillna(np.inf)
    frame = frame.dropna(subset=["_value"]).sort_values(
        ["race_id", "_value", "_grid", "driver"], ascending=[True, ascending, True, True]
    )
    return frame.groupby("race_id")["driver"].first()


def _baseline_accuracy(data: pd.DataFrame, column: str, ascending: bool) -> float | None:
    picks = baseline_pick(data, column, ascending)
    if picks.empty:
        return None
    finish = pd.to_numeric(data["finish_position"], errors="coerce")
    winners = data.loc[finish.eq(1)].set_index("race_id")["driver"]
    joined = pd.concat([picks.rename("pick"), winners.rename("winner")], axis=1).dropna()
    return float((joined["pick"] == joined["winner"]).mean()) if len(joined) else None


def grid_only_probabilities(data: pd.DataFrame, target_orders: list[int]) -> pd.Series:
    """Walk-forward win probabilities from grid position alone.

    This is the honest probability baseline: a one-feature Plackett-Luce model
    on log grid position, refitted before every race exactly like the main model.
    """
    frame = data.assign(_order=race_order(data["race_id"]))
    output = pd.Series(np.nan, index=data.index)
    for target_order in target_orders:
        training = frame[frame["_order"].lt(target_order)]
        target = frame[frame["_order"].eq(target_order)]
        layout = build_layout(training["race_id"], training["finish_position"])
        model = LinearPlackettLuce(l2=0.01, stage_weights=(1.0,)).fit(training[["grid_log"]], layout)
        output.loc[target.index] = win_probabilities(
            target["race_id"], model.strengths(target[["grid_log"]])
        )
    return output


def prepare(data: pd.DataFrame, model_columns: list[str] | None = None) -> pd.DataFrame:
    """Add finish-order features unless the caller supplies its own columns."""
    columns = model_columns or ORDER_MODEL_COLUMNS
    if set(columns) <= set(data.columns):
        return data
    return build_order_features(data)


def walk_forward_predictions(
    data: pd.DataFrame,
    season: int,
    half_races: int | None = None,
    seed: int = 0,
    model_columns: list[str] | None = None,
    calibration_races: int = 40,
) -> tuple[pd.DataFrame, list[dict]]:
    """Predict post-half-season races and append each completed race in order.

    ``calibration_races`` earlier races are predicted the same way first (when
    enough history exists) so the temperature is already fitted on
    out-of-sample forecasts when the first scored race is reached.
    """
    frame = prepare(data, model_columns).copy()
    columns = list(model_columns or ORDER_MODEL_COLUMNS)
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"walk-forward data missing model columns: {sorted(missing)}")
    frame["_order"] = race_order(frame["race_id"])
    all_orders = sorted(frame["_order"].unique())
    season_orders = [value for value in all_orders if value // 100 == season]
    if len(season_orders) < 2:
        raise ValueError("selected season must contain at least two races")
    half = half_races if half_races is not None else len(season_orders) // 2
    if half < 1 or half >= len(season_orders):
        raise ValueError("half_races must be between 1 and the number of races minus 1")
    targets = season_orders[half:]
    earlier = [value for value in all_orders if value < targets[0]]
    warmup = [
        value for position, value in enumerate(earlier) if position >= MIN_TRAINING_RACES
    ][-calibration_races:] if calibration_races > 0 else []
    scored = walk_forward_order(
        frame.drop(columns="_order"), warmup + targets, columns=columns, seed=seed
    )
    scored["_order"] = race_order(scored["race_id"])
    predictions = scored[scored["_order"].isin(targets)].copy()
    predictions["score"] = predictions["strength"]
    predictions["uncalibrated_race_probability"] = win_probabilities(
        predictions["race_id"], predictions["raw_strength"].to_numpy()
    )
    metrics: list[dict] = []
    for target_order in targets:
        target = predictions[predictions["_order"].eq(target_order)]
        race_metrics = evaluate_predictions(target)
        race_metrics.update(
            {
                "race_id": str(target["race_id"].iloc[0]),
                "training_races": int(target["training_races"].iloc[0]),
                "temperature": float(target["temperature"].iloc[0]),
                "grid_baseline_accuracy": _baseline_accuracy(target, "grid_position", True),
                "qualifying_baseline_accuracy": _baseline_accuracy(
                    target, "qualifying_best_lap_s", True
                ),
                "form_baseline_accuracy": _baseline_accuracy(
                    target, "driver_points_fast", False
                ),
            }
        )
        metrics.append(race_metrics)
    return predictions.drop(columns="_order").reset_index(drop=True), metrics
