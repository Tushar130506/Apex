"""Chronological Monte Carlo forecasts for driver and constructor championships.

Completed points are fixed. Every remaining race is sampled as a complete
Plackett-Luce finishing order, points are accumulated, and the final table is
ranked with the sporting-regulation countback (most wins, then most second
places, and so on), so a simulated season never ends in a shared title.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from apex.models.plackett_luce import sample_orders
from apex.models.points import GRAND_PRIX_POINTS, SPRINT_POINTS, derive_points

RACE_POINTS = GRAND_PRIX_POINTS
MISSING_TEAM_LABELS = {"", "None", "nan"}


@dataclass(frozen=True)
class ChampionshipForecast:
    """Aggregated championship probabilities and expected final points."""

    driver: pd.DataFrame
    constructor: pd.DataFrame
    simulations: int
    completed_races: int
    projected_races: int
    projected_sprints: int = 0


def _race_order(race_id: pd.Series) -> pd.Series:
    parts = race_id.astype(str).str.extract(r"^(?P<season>\d{4})_(?P<round>\d+)$")
    if parts.isna().any().any():
        raise ValueError("race_id values must have the form YYYY_round")
    return parts["season"].astype(int) * 100 + parts["round"].astype(int)


def _has_missing_team(team: pd.Series) -> bool:
    return bool(team.isna().any() or team.astype(str).str.strip().isin(MISSING_TEAM_LABELS).any())


def _validate_results(results: pd.DataFrame) -> None:
    required = {"race_id", "driver", "team", "points"}
    missing = required - set(results.columns)
    if missing:
        raise ValueError(f"results missing columns: {sorted(missing)}")
    if _has_missing_team(results["team"]):
        raise ValueError("results contain missing team labels")


def _validate_predictions(predictions: pd.DataFrame) -> None:
    required = {"race_id", "driver", "team"}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"predictions missing columns: {sorted(missing)}")
    if not ({"strength", "score", "race_probability"} & set(predictions.columns)):
        raise ValueError("predictions require strength, race_probability, or score")
    if _has_missing_team(predictions["team"]):
        raise ValueError("predictions contain missing team labels")
    if predictions.duplicated(["race_id", "driver"]).any():
        raise ValueError("predictions contain duplicate race/driver rows")


def _log_strengths(group: pd.DataFrame, temperature: float) -> np.ndarray:
    """Log-strengths for one race.

    ``strength`` is already a log-strength. ``race_probability`` and ``score``
    are proportional to strength, so their logarithm is used.
    """
    if temperature <= 0:
        raise ValueError("temperature must be greater than zero")
    if "strength" in group:
        raw = pd.to_numeric(group["strength"], errors="coerce").to_numpy(float)
        if np.isfinite(raw).all():
            return (raw - raw.max()) / temperature
    for column in ("race_probability", "score"):
        if column not in group:
            continue
        raw = pd.to_numeric(group[column], errors="coerce").to_numpy(float)
        if np.isfinite(raw).all() and (raw >= 0).all():
            logs = np.log(np.maximum(raw, 1e-12))
            return (logs - logs.max()) / temperature
    raise ValueError("prediction strengths must be finite and non-negative")


def countback_ranks(points: np.ndarray, position_counts: np.ndarray) -> np.ndarray:
    """Rank entrants (0 = champion) by points, then wins, seconds, thirds, ...

    ``points`` has shape ``(simulations, entrants)`` and ``position_counts``
    ``(simulations, entrants, positions)``. Entrants that are still level after
    the full countback keep their input order, which only happens when their
    simulated seasons are identical in every counted position.
    """
    simulations, entrants = points.shape
    ranks = np.empty((simulations, entrants), dtype=int)
    positions = position_counts.shape[2]
    for simulation in range(simulations):
        keys = [
            -position_counts[simulation, :, position]
            for position in range(positions - 1, -1, -1)
        ]
        keys.append(-points[simulation])
        order = np.lexsort(keys)
        ranks[simulation, order] = np.arange(entrants)
    return ranks


def _completed_position_counts(
    completed: pd.DataFrame, key: str, labels: list[str], positions: int
) -> np.ndarray:
    counts = np.zeros((len(labels), positions))
    if "finish_position" not in completed:
        return counts
    races = completed
    if "event_type" in races:
        races = races[races["event_type"].astype(str).str.lower().isin({"race", "grand prix", "gp"})]
    finish = pd.to_numeric(races["finish_position"], errors="coerce")
    index = {label: number for number, label in enumerate(labels)}
    for label, position in zip(races[key], finish):
        if label in index and pd.notna(position) and 1 <= position <= positions:
            counts[index[label], int(position) - 1] += 1
    return counts


def forecast_championship(
    completed_results: pd.DataFrame,
    future_predictions: pd.DataFrame,
    *,
    simulations: int = 5000,
    seed: int = 0,
    temperature: float = 1.0,
    sprint_race_ids: list[str] | None = None,
    form_drift_per_race: float = 0.0,
) -> ChampionshipForecast:
    """Forecast WDC/WCC from completed points and future race predictions.

    ``completed_results`` must contain only races known at the prediction
    timestamp. ``future_predictions`` must contain only races after that
    timestamp. The function never reads future result columns.

    ``sprint_race_ids`` lists future rounds that also hold a sprint; a second,
    independent order is sampled for each and scored on the sprint scale.
    ``form_drift_per_race`` is the standard deviation of a per-driver random
    walk added to log-strengths race after race. It widens long-horizon
    forecasts, because form measured today is a worse guide to a race ten
    rounds away than to the next one.
    """
    if simulations < 1:
        raise ValueError("simulations must be positive")
    if form_drift_per_race < 0:
        raise ValueError("form_drift_per_race must be non-negative")
    _validate_results(completed_results)
    _validate_predictions(future_predictions)
    if {"finish_position", "classified"} <= set(completed_results.columns):
        completed = derive_points(completed_results)
    else:
        completed = completed_results.copy()
        completed["derived_points"] = pd.to_numeric(completed["points"], errors="coerce").fillna(0.0)
    future = future_predictions.copy()
    completed["_order"] = _race_order(completed["race_id"])
    future["_order"] = _race_order(future["race_id"])
    if set(completed["race_id"]) & set(future["race_id"]):
        raise ValueError("completed and future races overlap")
    if not future.empty and not completed.empty and completed["_order"].max() >= future["_order"].min():
        raise ValueError("future predictions must be chronologically after completed races")
    sprint_ids = set(sprint_race_ids or [])
    unknown_sprints = sprint_ids - set(future["race_id"])
    if unknown_sprints:
        raise ValueError(f"sprint rounds without predictions: {sorted(unknown_sprints)}")

    drivers = sorted(set(completed["driver"]) | set(future["driver"]))
    driver_index = {driver: number for number, driver in enumerate(drivers)}
    # A driver's current team is the one from the latest known race.
    latest_team = (
        pd.concat([completed[["driver", "team", "_order"]], future[["driver", "team", "_order"]]])
        .sort_values("_order")
        .drop_duplicates("driver", keep="last")
        .set_index("driver")["team"]
        .to_dict()
    )
    teams = sorted(set(completed["team"]) | set(future["team"]))
    team_index = {team: number for number, team in enumerate(teams)}
    positions = max(
        int(future.groupby("race_id").size().max()) if not future.empty else 0,
        int(pd.to_numeric(completed.get("finish_position", pd.Series(dtype=float)), errors="coerce").max())
        if "finish_position" in completed and completed["finish_position"].notna().any()
        else 0,
        1,
    )

    base_driver = completed.groupby("driver")["derived_points"].sum()
    base_team = completed.groupby("team")["derived_points"].sum()
    driver_points = np.tile(
        np.array([float(base_driver.get(driver, 0.0)) for driver in drivers]), (simulations, 1)
    )
    team_points = np.tile(
        np.array([float(base_team.get(team, 0.0)) for team in teams]), (simulations, 1)
    )
    driver_counts = np.tile(
        _completed_position_counts(completed, "driver", drivers, positions), (simulations, 1, 1)
    )
    team_counts = np.tile(
        _completed_position_counts(completed, "team", teams, positions), (simulations, 1, 1)
    )

    rng = np.random.default_rng(seed)
    rows = np.arange(simulations)[:, None]
    drift = np.zeros((simulations, len(drivers)))
    race_scale = np.zeros(max(positions, len(RACE_POINTS)))
    race_scale[: len(RACE_POINTS)] = RACE_POINTS
    sprint_scale = np.zeros(max(positions, len(SPRINT_POINTS)))
    sprint_scale[: len(SPRINT_POINTS)] = SPRINT_POINTS
    projected_sprints = 0
    for _, group in future.groupby("_order", sort=True):
        group = group.sort_values("driver")
        entrants = np.array([driver_index[driver] for driver in group["driver"]])
        entrant_teams = np.array([team_index[team] for team in group["team"]])
        strengths = _log_strengths(group, temperature)
        if form_drift_per_race > 0:
            drift += rng.normal(0.0, form_drift_per_race, size=drift.shape)
        current = strengths[None, :] + drift[:, entrants]
        events = [("race", race_scale)]
        if group["race_id"].iloc[0] in sprint_ids:
            events.insert(0, ("sprint", sprint_scale))
            projected_sprints += 1
        for kind, scale in events:
            noise = rng.gumbel(size=current.shape)
            orders = np.argsort(-(current + noise), axis=1, kind="stable")
            width = orders.shape[1]
            earned = np.broadcast_to(scale[:width], orders.shape)
            np.add.at(driver_points, (rows, entrants[orders]), earned)
            np.add.at(team_points, (rows, entrant_teams[orders]), earned)
            if kind == "race":
                slots = np.broadcast_to(np.arange(width), orders.shape)
                np.add.at(driver_counts, (rows, entrants[orders], slots), 1.0)
                np.add.at(team_counts, (rows, entrant_teams[orders], slots), 1.0)

    driver_ranks = countback_ranks(driver_points, driver_counts)
    team_ranks = countback_ranks(team_points, team_counts)
    driver_frame = pd.DataFrame(
        {
            "driver": drivers,
            "team": [latest_team.get(driver) for driver in drivers],
            "current_points": [float(base_driver.get(driver, 0.0)) for driver in drivers],
            "expected_final_points": driver_points.mean(axis=0),
            "points_p05": np.quantile(driver_points, 0.05, axis=0),
            "points_p95": np.quantile(driver_points, 0.95, axis=0),
            "win_probability": (driver_ranks == 0).mean(axis=0),
            "top_three_probability": (driver_ranks < 3).mean(axis=0),
            "expected_final_position": driver_ranks.mean(axis=0) + 1.0,
        }
    ).sort_values(
        ["win_probability", "expected_final_points", "driver"], ascending=[False, False, True]
    )
    constructor_frame = pd.DataFrame(
        {
            "team": teams,
            "current_points": [float(base_team.get(team, 0.0)) for team in teams],
            "expected_final_points": team_points.mean(axis=0),
            "points_p05": np.quantile(team_points, 0.05, axis=0),
            "points_p95": np.quantile(team_points, 0.95, axis=0),
            "win_probability": (team_ranks == 0).mean(axis=0),
            "expected_final_position": team_ranks.mean(axis=0) + 1.0,
        }
    ).sort_values(
        ["win_probability", "expected_final_points", "team"], ascending=[False, False, True]
    )
    return ChampionshipForecast(
        driver=driver_frame.reset_index(drop=True),
        constructor=constructor_frame.reset_index(drop=True),
        simulations=simulations,
        completed_races=int(completed["race_id"].nunique()),
        projected_races=int(future["race_id"].nunique()),
        projected_sprints=projected_sprints,
    )


__all__ = [
    "ChampionshipForecast",
    "RACE_POINTS",
    "countback_ranks",
    "forecast_championship",
    "sample_orders",
]
