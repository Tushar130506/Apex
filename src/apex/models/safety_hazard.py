"""Lap-by-lap safety-car hazard estimates from prior races."""
from __future__ import annotations

import numpy as np
import pandas as pd


def historical_lap_hazard(
    laps: pd.DataFrame,
    *,
    track: str | None = None,
    prior_only_before: int | None = None,
) -> dict[int, float]:
    """Estimate P(SC starts on lap) from earlier cached race observations."""
    required = {"lap_no", "track_status"}
    missing = required - set(laps.columns)
    if missing:
        raise ValueError(f"laps missing columns: {sorted(missing)}")
    frame = laps.copy()
    if track is not None and "track" in frame:
        frame = frame[frame["track"].eq(track)]
    if prior_only_before is not None and "season" in frame:
        frame = frame[frame["season"].lt(prior_only_before)]
    if frame.empty:
        return {}
    frame["sc"] = frame["track_status"].astype(str).str.contains(r"[46]")
    rates: dict[int, float] = {}
    for lap, group in frame.groupby("lap_no"):
        starts = int(group["sc"].any())
        races = max(1, int(group["session_id"].nunique()) if "session_id" in group else len(group))
        rates[int(lap)] = (starts + 1.0) / (races + 2.0)
    return rates


def sample_safety_car_starts(
    rng: np.random.Generator,
    laps_remaining: int,
    hazard_by_lap: dict[int, float],
    samples: int,
) -> np.ndarray:
    """Sample the first safety-car start lap, or -1 when none occurs."""
    if laps_remaining < 1 or samples < 1:
        raise ValueError("laps_remaining and samples must be positive")
    starts = np.full(samples, -1, dtype=int)
    for offset in range(laps_remaining):
        probability = float(np.clip(hazard_by_lap.get(offset + 1, 0.02), 0.0, 1.0))
        active = starts < 0
        occurred = active & (rng.random(samples) < probability)
        starts[occurred] = offset + 1
    return starts
