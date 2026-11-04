"""Monte Carlo race-winner probabilities using pace, tyres, fuel, and safety."""
from __future__ import annotations

import numpy as np
import pandas as pd

from apex.features.fuel import sample_fuel_burn
from apex.features.tyre_degradation import CompoundDegradation
from apex.models.safety_hazard import sample_safety_car_starts
from apex.models.winner import race_probabilities


def monte_carlo_race_probabilities(
    data: pd.DataFrame,
    scores: np.ndarray,
    *,
    simulations: int = 3000,
    seed: int = 0,
    laps_remaining: int = 50,
    hazard_by_lap: dict[int, float] | None = None,
    degradation_model: CompoundDegradation | None = None,
) -> np.ndarray:
    """Simulate finish orders and return mutually exclusive win probabilities."""
    if simulations < 1 or laps_remaining < 1:
        raise ValueError("simulations and laps_remaining must be positive")
    if len(data) != len(scores):
        raise ValueError("data and scores must have the same length")
    rng = np.random.default_rng(seed)
    base = race_probabilities(data, scores)
    degradation = pd.to_numeric(
        data.get("degradation_s_per_lap", pd.Series(np.nan, index=data.index)),
        errors="coerce",
    ).fillna(0.05).to_numpy(float)
    if degradation_model is not None and {"compound", "tyre_age"} <= set(data.columns):
        model_degradation = []
        for compound, age, fallback_rate in zip(
            data["compound"], data["tyre_age"], degradation
        ):
            if pd.isna(compound) or pd.isna(age):
                model_degradation.append(fallback_rate)
                continue
            current_loss = degradation_model.loss(str(compound), float(age))
            next_loss = degradation_model.loss(str(compound), float(age) + 1.0)
            rate = float(np.asarray(next_loss - current_loss))
            model_degradation.append(rate if np.isfinite(rate) else fallback_rate)
        degradation = np.asarray(model_degradation, dtype=float)
    degradation = np.where(np.isfinite(degradation), degradation, 0.05)
    fuel_sensitivity = pd.to_numeric(
        data.get(
            "fuel_sensitivity_s_per_kg",
            pd.Series(0.032, index=data.index),
        ),
        errors="coerce",
    ).fillna(0.032).to_numpy(float)
    fuel_burn = sample_fuel_burn(rng, simulations * len(data)).reshape(
        simulations, len(data)
    )
    safety_starts = sample_safety_car_starts(
        rng, laps_remaining, hazard_by_lap or {}, simulations
    )
    wins = np.zeros(len(data), dtype=float)
    for simulation in range(simulations):
        tyre_noise = rng.lognormal(0.0, 0.18, len(data))
        fuel_penalty = (
            fuel_burn[simulation] * laps_remaining * fuel_sensitivity
        )
        safety_penalty = 0.0 if safety_starts[simulation] < 0 else rng.normal(0.0, 0.015)
        effective = np.log(np.clip(base, 1e-12, 1.0))
        effective -= degradation * tyre_noise * 0.01
        effective -= fuel_penalty * 0.001
        effective += safety_penalty * rng.normal(0.0, 1.0, len(data))
        effective -= effective.max()
        adjusted = np.exp(effective)
        adjusted /= adjusted.sum()
        wins[int(rng.choice(len(data), p=adjusted))] += 1
    return wins / simulations
