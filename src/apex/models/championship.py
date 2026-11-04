"""Chronological Monte Carlo forecasts for driver and constructor championships."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from apex.models.points import GRAND_PRIX_POINTS
from apex.models.points import derive_points

RACE_POINTS = GRAND_PRIX_POINTS


@dataclass(frozen=True)
class ChampionshipForecast:
    """Aggregated championship probabilities and expected final points."""

    driver: pd.DataFrame
    constructor: pd.DataFrame
    simulations: int
    completed_races: int
    projected_races: int


def _race_order(race_id: pd.Series) -> pd.Series:
    parts = race_id.astype(str).str.extract(r"^(?P<season>\d{4})_(?P<round>\d+)$")
    if parts.isna().any().any():
        raise ValueError("race_id values must have the form YYYY_round")
    return parts["season"].astype(int) * 100 + parts["round"].astype(int)


def _validate_results(results: pd.DataFrame) -> None:
    required = {"race_id", "driver", "team", "points"}
    missing = required - set(results.columns)
    if missing:
        raise ValueError(f"results missing columns: {sorted(missing)}")
    if results["team"].isna().any() or results["team"].astype(str).str.strip().isin({"", "None", "nan"}).any():
        raise ValueError("results contain missing team labels")


def _validate_predictions(predictions: pd.DataFrame) -> None:
    required = {"race_id", "driver", "team"}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"predictions missing columns: {sorted(missing)}")
    if not ({"score", "race_probability"} & set(predictions.columns)):
        raise ValueError("predictions require score or race_probability")
    if predictions["team"].isna().any() or predictions["team"].astype(str).str.strip().isin({"", "None", "nan"}).any():
        raise ValueError("predictions contain missing team labels")


def _strengths(group: pd.DataFrame, temperature: float) -> np.ndarray:
    if temperature <= 0:
        raise ValueError("temperature must be greater than zero")
    if "score" in group:
        raw = pd.to_numeric(group["score"], errors="coerce").to_numpy(float)
        if np.isfinite(raw).all():
            centered = raw - raw.max()
            return np.exp(centered / temperature)
    probability = pd.to_numeric(group["race_probability"], errors="coerce").to_numpy(float)
    if not np.isfinite(probability).all() or (probability < 0).any():
        raise ValueError("prediction strengths must be finite and non-negative")
    return np.maximum(probability, 1e-12) ** (1.0 / temperature)


def _sample_order(
    group: pd.DataFrame,
    rng: np.random.Generator,
    temperature: float,
    simulations: int = 1,
) -> np.ndarray:
    """Sample a Plackett-Luce order without replacing drivers."""
    if simulations < 1:
        raise ValueError("simulations must be positive")
    drivers = group["driver"].to_numpy()
    strengths = _strengths(group, temperature)
    log_strengths = np.log(np.maximum(strengths, 1e-12))
    available = np.ones((simulations, len(drivers)), dtype=bool)
    order_indices = np.empty((simulations, len(drivers)), dtype=int)
    for position in range(len(drivers)):
        gumbels = rng.gumbel(size=(simulations, len(drivers)))
        scores = np.where(available, log_strengths + gumbels, -np.inf)
        selected = scores.argmax(axis=1)
        order_indices[:, position] = selected
        available[np.arange(simulations), selected] = False
    orders = drivers[order_indices]
    return orders[0] if simulations == 1 else orders


def _points_for_order(order: Iterable[str]) -> dict[str, float]:
    return {
        driver: float(RACE_POINTS[position])
        for position, driver in enumerate(order)
        if position < len(RACE_POINTS)
    }


def forecast_championship(
    completed_results: pd.DataFrame,
    future_predictions: pd.DataFrame,
    *,
    simulations: int = 5000,
    seed: int = 0,
    temperature: float = 1.0,
) -> ChampionshipForecast:
    """Forecast WDC/WCC from completed points and future race predictions.

    ``completed_results`` must contain only races known at the prediction
    timestamp. ``future_predictions`` must contain only races after that
    timestamp. The function never reads future result columns.
    """
    if simulations < 1:
        raise ValueError("simulations must be positive")
    _validate_results(completed_results)
    _validate_predictions(future_predictions)
    if {"finish_position", "classified"} <= set(completed_results.columns):
        completed = derive_points(completed_results)
    elif "points" in completed_results:
        completed = completed_results.copy()
        completed["derived_points"] = pd.to_numeric(
            completed["points"], errors="coerce"
        ).fillna(0.0)
    else:
        raise ValueError(
            "completed results require finish_position/classified or derived points"
        )
    future = future_predictions.copy()
    completed["_order"] = _race_order(completed["race_id"])
    future["_order"] = _race_order(future["race_id"])
    if set(completed["race_id"]) & set(future["race_id"]):
        raise ValueError("completed and future races overlap")
    if not future.empty and completed["_order"].max() >= future["_order"].min():
        raise ValueError("future predictions must be chronologically after completed races")

    drivers = sorted(set(completed["driver"]) | set(future["driver"]))
    team_by_driver = (
        pd.concat(
            [completed[["driver", "team"]], future[["driver", "team"]]],
            ignore_index=True,
        )
        .dropna(subset=["driver", "team"])
        .drop_duplicates("driver")
        .set_index("driver")["team"]
        .to_dict()
    )
    base_driver = completed.groupby("driver")["derived_points"].sum().to_dict()
    base_team = completed.groupby("team")["derived_points"].sum().to_dict()
    driver_points = np.zeros((simulations, len(drivers)), dtype=float)
    constructor_points: dict[str, np.ndarray] = {
        team: np.zeros(simulations, dtype=float)
        for team in set(team_by_driver.values())
    }
    rng = np.random.default_rng(seed)
    future_groups = [group for _, group in future.groupby("race_id", sort=True)]
    sampled_orders = [
        _sample_order(group, rng, temperature, simulations=simulations)
        for group in future_groups
    ]

    for simulation in range(simulations):
        points = {driver: float(base_driver.get(driver, 0.0)) for driver in drivers}
        teams = {team: float(base_team.get(team, 0.0)) for team in constructor_points}
        for group, sampled_group_orders in zip(future_groups, sampled_orders):
            order = sampled_group_orders[simulation]
            future_team_by_driver = (
                group.dropna(subset=["driver", "team"])
                .drop_duplicates("driver")
                .set_index("driver")["team"]
                .to_dict()
            )
            for driver, earned in _points_for_order(order).items():
                points[driver] = points.get(driver, 0.0) + earned
                team = future_team_by_driver.get(driver, team_by_driver.get(driver))
                if team is not None:
                    teams[team] = teams.get(team, 0.0) + earned
        driver_points[simulation] = [points[driver] for driver in drivers]
        for team, values in constructor_points.items():
            values[simulation] = teams.get(team, 0.0)

    driver_frame = pd.DataFrame(
        {
            "driver": drivers,
            "current_points": [base_driver.get(driver, 0.0) for driver in drivers],
            "expected_final_points": driver_points.mean(axis=0),
            "win_probability": (
                (driver_points == driver_points.max(axis=1, keepdims=True))
                / (driver_points == driver_points.max(axis=1, keepdims=True)).sum(
                    axis=1, keepdims=True
                )
            ).mean(axis=0),
            "top_three_probability": (
                driver_points.argsort(axis=1)[:, -3:, None]
                == np.arange(len(drivers))[None, None, :]
            ).any(axis=1).mean(axis=0),
        }
    ).sort_values(["win_probability", "expected_final_points"], ascending=False)

    teams = sorted(constructor_points)
    team_matrix = np.column_stack([constructor_points[team] for team in teams])
    constructor_frame = pd.DataFrame(
        {
            "team": teams,
            "current_points": [base_team.get(team, 0.0) for team in teams],
            "expected_final_points": team_matrix.mean(axis=0),
            "win_probability": (
                (team_matrix == team_matrix.max(axis=1, keepdims=True))
                / (team_matrix == team_matrix.max(axis=1, keepdims=True)).sum(
                    axis=1, keepdims=True
                )
            ).mean(axis=0),
        }
    ).sort_values(["win_probability", "expected_final_points"], ascending=False)
    return ChampionshipForecast(
        driver=driver_frame.reset_index(drop=True),
        constructor=constructor_frame.reset_index(drop=True),
        simulations=simulations,
        completed_races=int(completed["race_id"].nunique()),
        projected_races=int(future["race_id"].nunique()),
    )
