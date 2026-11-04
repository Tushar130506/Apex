"""Fuel-load assumptions and uncertainty handling."""
from __future__ import annotations

import numpy as np
import pandas as pd


def correct_lap_time(
    lap_time_s: pd.Series | np.ndarray,
    fuel_kg: pd.Series | np.ndarray,
    sensitivity_s_per_kg: float = 0.032,
) -> np.ndarray:
    """Remove the estimated fuel penalty from lap times."""
    if sensitivity_s_per_kg < 0:
        raise ValueError("sensitivity_s_per_kg must be non-negative")
    return np.asarray(lap_time_s, dtype=float) - sensitivity_s_per_kg * np.asarray(
        fuel_kg, dtype=float
    )


def sample_fuel_burn(
    rng: np.random.Generator,
    samples: int,
    mean_kg_per_lap: float = 1.6,
    sd_kg_per_lap: float = 0.08,
) -> np.ndarray:
    """Sample positive per-lap fuel burn for race simulations."""
    if samples < 1 or mean_kg_per_lap <= 0 or sd_kg_per_lap < 0:
        raise ValueError("invalid fuel-burn sampling parameters")
    return np.maximum(0.01, rng.normal(mean_kg_per_lap, sd_kg_per_lap, samples))
