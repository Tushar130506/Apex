"""Chronological race-probability calibration."""
from __future__ import annotations

import numpy as np
import pandas as pd

from apex.models.winner import race_probabilities


def select_temperature(
    history: pd.DataFrame,
    *,
    score_column: str = "score",
    grid: tuple[float, ...] = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0),
) -> float:
    """Select temperature using only completed earlier races."""
    if history.empty:
        return 1.0
    losses = {}
    for temperature in grid:
        probabilities = race_probabilities(
            history, history[score_column].to_numpy(float), temperature=temperature
        )
        actual = history["winner"].to_numpy(bool)
        winners = probabilities[actual]
        losses[temperature] = float(-np.log(np.clip(winners, 1e-12, 1.0)).mean())
    return min(losses, key=losses.get)


def calibrated_race_probabilities(
    history: pd.DataFrame, target: pd.DataFrame, scores: np.ndarray
) -> np.ndarray:
    """Apply chronological temperature calibration to target scores."""
    temperature = select_temperature(history)
    return race_probabilities(
        target, np.asarray(scores, dtype=float), temperature=temperature
    )
