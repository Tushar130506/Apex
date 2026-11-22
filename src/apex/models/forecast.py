"""Genuine forecasts: a race that has not been run, and the rest of a season.

Nothing here reads a result of the race being predicted. A target race is a
set of rows with driver, team and (once known) grid and session pace; its
``finish_position`` is empty.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from apex.features.form_features import (
    FORM_COLUMNS,
    ORDER_MODEL_COLUMNS,
    build_order_features,
    canonical_team,
    race_order,
)
from apex.models.championship import ChampionshipForecast, forecast_championship
from apex.models.plackett_luce import SEASON_STAGE_WEIGHTS, build_layout, fit_temperature
from apex.models.race_order import (
    MIN_CALIBRATION_RACES,
    RaceOrderModel,
    walk_forward_order,
)

RESULT_COLUMNS = ("finish_position", "classified", "points", "derived_points", "status", "winner")
MIN_TRAINING_RACES = 20


def completed_orders(table: pd.DataFrame) -> list[int]:
    """Chronological keys of races that have at least one finishing position."""
    finish = pd.to_numeric(table["finish_position"], errors="coerce")
    order = race_order(table["race_id"])
    return sorted(order[finish.gt(0)].unique().tolist())


def latest_roster(table: pd.DataFrame) -> pd.DataFrame:
    """Drivers and teams entered in the most recent completed race."""
    orders = completed_orders(table)
    if not orders:
        raise ValueError("no completed race to take a roster from")
    last = table[race_order(table["race_id"]).eq(orders[-1])]
    roster = last[["driver", "team"]].copy()
    roster["team"] = canonical_team(roster["team"])
    if roster["team"].isna().any():
        # Fall back to the driver's most recent labeled team.
        labeled = table.assign(_team=canonical_team(table["team"]), _order=race_order(table["race_id"]))
        lookup = (
            labeled.dropna(subset=["_team"]).sort_values("_order")
            .drop_duplicates("driver", keep="last").set_index("driver")["_team"]
        )
        roster["team"] = roster["team"].fillna(roster["driver"].map(lookup))
    roster["team"] = roster["team"].fillna("UNKNOWN")
    return roster.reset_index(drop=True)


def add_upcoming_race(
    table: pd.DataFrame,
    race_id: str,
    track: str | None = None,
    entries: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Append rows for a race that has not been run.

    ``entries`` needs ``driver`` and may carry ``team``, ``grid_position`` and
    any pre-race session column (for example ``qualifying_best_lap_s``).
    Result columns are always blank, whatever is passed in.
    """
    if race_id in set(table["race_id"]):
        raise ValueError(f"{race_id} is already in the table")
    rows = latest_roster(table) if entries is None else entries.copy()
    if "driver" not in rows or rows["driver"].duplicated().any():
        raise ValueError("entries need one row per driver")
    if "team" not in rows:
        rows = rows.merge(latest_roster(table), on="driver", how="left")
    for column in RESULT_COLUMNS:
        if column in table:
            rows[column] = np.nan
    rows["race_id"] = race_id
    rows["season"] = int(str(race_id).split("_")[0])
    rows["track"] = track
    extra = [column for column in rows.columns if column not in table.columns]
    return pd.concat([table, rows.drop(columns=extra)], ignore_index=True)


def _target_has_weekend_data(target: pd.DataFrame) -> bool:
    grid = pd.to_numeric(target.get("grid_position"), errors="coerce")
    return bool(grid.gt(0).mean() >= 0.8) if grid is not None else False


def predict_race(
    table: pd.DataFrame,
    race_id: str,
    *,
    calibration_races: int = 40,
    seed: int = 0,
    simulations: int = 20000,
) -> pd.DataFrame:
    """Forecast one race from earlier races plus whatever is known pre-race.

    With a starting grid the full model is used; before qualifying only the
    prior-race form model is. The preceding ``calibration_races`` are predicted
    the same way first, purely to fit the temperature on out-of-sample races.
    """
    features = build_order_features(table)
    order = race_order(features["race_id"])
    target_mask = features["race_id"].eq(race_id)
    if not target_mask.any():
        raise ValueError(f"{race_id} is not in the table")
    target_order = int(order[target_mask].iloc[0])
    post_qualifying = _target_has_weekend_data(features[target_mask])
    columns = ORDER_MODEL_COLUMNS if post_qualifying else FORM_COLUMNS
    earlier = [value for value in completed_orders(features) if value < target_order]
    if len(earlier) < 2:
        raise ValueError("at least two earlier completed races are required")
    warmup = [
        value for position, value in enumerate(earlier)
        if position >= MIN_TRAINING_RACES
    ][-calibration_races:]
    predictions = walk_forward_order(
        features, warmup + [target_order], columns=columns, seed=seed, simulations=simulations
    )
    result = predictions[predictions["race_id"].eq(race_id)].copy()
    result["forecast_mode"] = "post-qualifying" if post_qualifying else "pre-weekend"
    result["calibration_races"] = len(warmup)
    return result.sort_values("predicted_position").reset_index(drop=True)


def estimate_form_drift(history: list[pd.DataFrame]) -> float:
    """Per-race standard deviation of change in a driver's relative strength.

    Measured between consecutive out-of-sample form forecasts. Strengths are
    only defined up to a per-race constant, so each race is centred on the
    drivers it shares with the next one.
    """
    changes = []
    for earlier, later in zip(history[:-1], history[1:]):
        merged = earlier[["driver", "raw_strength"]].merge(
            later[["driver", "raw_strength"]], on="driver", suffixes=("_a", "_b")
        )
        if len(merged) < 5:
            continue
        delta = merged["raw_strength_b"] - merged["raw_strength_a"]
        changes.append((delta - delta.mean()).to_numpy())
    if not changes:
        return 0.0
    return float(np.sqrt(np.mean(np.concatenate(changes) ** 2)))


def form_history(
    features: pd.DataFrame, orders: list[int], *, seed: int = 0
) -> list[pd.DataFrame]:
    """Out-of-sample form-only strengths for the given completed races."""
    frame = features.assign(_order=race_order(features["race_id"]))
    history = []
    for target_order in orders:
        training = frame[frame["_order"].lt(target_order)]
        target = frame[frame["_order"].eq(target_order)]
        model = RaceOrderModel(
            columns=list(FORM_COLUMNS), seed=seed, stage_weights=SEASON_STAGE_WEIGHTS
        ).fit(training)
        history.append(
            pd.DataFrame(
                {
                    "race_id": target["race_id"].to_numpy(),
                    "driver": target["driver"].to_numpy(),
                    "finish_position": pd.to_numeric(target["finish_position"], errors="coerce").to_numpy(),
                    "raw_strength": model.strengths(target),
                }
            )
        )
    return history


def form_calibration(history: list[pd.DataFrame]) -> tuple[float, float]:
    """Temperature and per-race drift from out-of-sample form forecasts."""
    if len(history) < MIN_CALIBRATION_RACES:
        return 1.0, 0.0
    scored = pd.concat(history, ignore_index=True)
    layout = build_layout(scored["race_id"], scored["finish_position"])
    temperature = fit_temperature(
        scored["raw_strength"].to_numpy(), layout, stage_weights=SEASON_STAGE_WEIGHTS
    )
    return temperature, estimate_form_drift(history) / temperature


@dataclass(frozen=True)
class SeasonOutlook:
    """Remaining-race strengths plus the championship simulation built on them."""

    races: pd.DataFrame
    championship: ChampionshipForecast
    temperature: float
    form_drift_per_race: float


def season_outlook(
    table: pd.DataFrame,
    season: int,
    remaining: pd.DataFrame,
    *,
    roster: pd.DataFrame | None = None,
    sprint_race_ids: list[str] | None = None,
    sprint_results: pd.DataFrame | None = None,
    history: list[pd.DataFrame] | None = None,
    calibration_races: int = 40,
    simulations: int = 5000,
    seed: int = 0,
) -> SeasonOutlook:
    """Forecast every remaining race of ``season`` and the two championships.

    ``table`` holds completed races only (anything later is ignored as an
    outcome because its results are blank). ``remaining`` lists ``race_id`` and
    optionally ``track`` for the rounds still to run.

    ``sprint_results`` holds completed sprint classifications (``race_id``,
    ``driver``, ``team``, ``finish_position``, ``classified``). Sprints of
    ``season`` held at rounds that are already complete add their points to
    the standings. A sprint at a round still listed in ``remaining`` is left
    out; list that round in ``sprint_race_ids`` to simulate it instead.

    ``sprint_race_ids=None`` means the remaining sprints are unknown: none is
    simulated, but the clinched/eliminated check assumes every remaining round
    could hold one. Pass ``[]`` when no sprints remain. A held sprint at a
    remaining round also counts towards that check, because its points are
    not in the standings.
    """
    if remaining.empty:
        raise ValueError("remaining must list at least one race")
    combined = table
    entries = latest_roster(table) if roster is None else roster
    for row in remaining.itertuples(index=False):
        combined = add_upcoming_race(
            combined, row.race_id, getattr(row, "track", None), entries
        )
    features = build_order_features(combined)
    order = race_order(features["race_id"])
    done = completed_orders(features)
    if history is None:
        warmup = [value for position, value in enumerate(done) if position >= MIN_TRAINING_RACES]
        history = form_history(features, warmup[-calibration_races:], seed=seed)
    temperature, drift = form_calibration(history)
    finish = pd.to_numeric(features["finish_position"], errors="coerce")
    model = RaceOrderModel(
        columns=list(FORM_COLUMNS), seed=seed, stage_weights=SEASON_STAGE_WEIGHTS
    ).fit(features[finish.gt(0).groupby(features["race_id"]).transform("any")])
    future = features[features["race_id"].isin(set(remaining["race_id"]))].copy()
    future["strength"] = model.strengths(future) / temperature
    future["team"] = future["team_canonical"]
    races = future[["race_id", "track", "driver", "team", "strength"]].reset_index(drop=True)

    season_done = features[order.floordiv(100).eq(season) & finish.gt(0)]
    completed = pd.DataFrame(
        {
            "race_id": season_done["race_id"],
            "driver": season_done["driver"],
            "team": season_done["team_canonical"],
            "points": pd.to_numeric(season_done.get("points"), errors="coerce"),
            "finish_position": pd.to_numeric(season_done["finish_position"], errors="coerce"),
            "classified": True,
        }
    )
    if "event_type" in season_done:
        completed["event_type"] = season_done["event_type"].fillna("race").to_numpy()
    if sprint_results is not None and not sprint_results.empty:
        completed = pd.concat(
            [completed.assign(event_type=completed.get("event_type", "race")),
             completed_sprints(sprint_results, set(completed["race_id"]))],
            ignore_index=True,
        )
    remaining_ids = set(remaining["race_id"])
    if sprint_race_ids is None:
        bound_sprints = len(remaining_ids)
    else:
        held = set() if sprint_results is None else set(sprint_results["race_id"]) & remaining_ids
        bound_sprints = len(set(sprint_race_ids) | held)
    championship = forecast_championship(
        completed,
        races,
        simulations=simulations,
        seed=seed,
        sprint_race_ids=sprint_race_ids,
        form_drift_per_race=drift,
        sprints_for_title_status=bound_sprints,
    )
    return SeasonOutlook(
        races=races, championship=championship, temperature=temperature, form_drift_per_race=drift
    )


def completed_sprints(sprint_results: pd.DataFrame, completed_race_ids: set[str]) -> pd.DataFrame:
    """Sprint rows, in the shape of completed results, for finished rounds only."""
    required = {"race_id", "driver", "team", "finish_position", "classified"}
    missing = required - set(sprint_results.columns)
    if missing:
        raise ValueError(f"sprint results missing columns: {sorted(missing)}")
    if sprint_results.duplicated(["race_id", "driver"]).any():
        raise ValueError("sprint results need one row per sprint and driver")
    rows = sprint_results[sprint_results["race_id"].isin(completed_race_ids)]
    return pd.DataFrame(
        {
            "race_id": rows["race_id"],
            "driver": rows["driver"],
            "team": canonical_team(rows["team"]),
            "points": np.nan,
            "finish_position": pd.to_numeric(rows["finish_position"], errors="coerce"),
            "classified": rows["classified"].fillna(False).astype(bool),
            "event_type": "sprint",
        }
    )


def append_results(table: pd.DataFrame, extra: pd.DataFrame) -> pd.DataFrame:
    """Append completed races that are missing from the feature table.

    ``extra`` needs ``race_id``, ``driver``, ``team``, ``grid_position``,
    ``finish_position`` and ``status``; any pre-race session column that is
    present (for example ``qualifying_best_lap_s``) is kept, the rest stay
    empty. Used when official results exist but lap data has not been
    processed yet.
    """
    required = {"race_id", "driver", "team", "grid_position", "finish_position", "status"}
    missing = required - set(extra.columns)
    if missing:
        raise ValueError(f"extra results missing columns: {sorted(missing)}")
    overlap = set(extra["race_id"]) & set(table["race_id"])
    if overlap:
        raise ValueError(f"races already in the table: {sorted(overlap)}")
    if extra.duplicated(["race_id", "driver"]).any():
        raise ValueError("extra results need one row per race and driver")
    rows = extra.copy()
    finish = pd.to_numeric(rows["finish_position"], errors="coerce")
    for race_id, positions in finish.groupby(rows["race_id"]):
        ranked = positions.dropna()
        if ranked.empty or ranked.duplicated().any() or ranked.min() != 1:
            raise ValueError(f"{race_id}: finishing positions must be unique and include a winner")
    rows["season"] = rows["race_id"].astype(str).str.split("_").str[0].astype(int)
    rows["classified"] = finish.notna()
    rows["winner"] = finish.eq(1).astype(int)
    for column in table.columns:
        if column not in rows:
            rows[column] = np.nan
    return pd.concat([table, rows[list(table.columns)]], ignore_index=True)
