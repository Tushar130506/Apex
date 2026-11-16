"""Chronological race-probability calibration."""
from __future__ import annotations

import numpy as np
import pandas as pd

from apex.models.plackett_luce import build_layout, fit_temperature, win_probabilities


def _log_strengths(scores: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(scores, dtype=float), 1e-7, 1 - 1e-7)
    return np.log(clipped / (1.0 - clipped))


def select_temperature(
    history: pd.DataFrame, *, score_column: str = "score", prior_strength: float = 10.0
) -> float:
    """Fit a temperature on completed earlier races only.

    The search is continuous and shrunk towards 1, so a handful of races
    cannot push it to an extreme value (the earlier six-point grid search on a
    short rolling window did exactly that).
    """
    if history.empty:
        return 1.0
    finish = history["winner"].map({1: 1.0, True: 1.0}).astype(float)
    layout = build_layout(history["race_id"], finish)
    return fit_temperature(
        _log_strengths(history[score_column].to_numpy(float)), layout, prior_strength=prior_strength
    )


def calibrated_race_probabilities(
    history: pd.DataFrame, target: pd.DataFrame, scores: np.ndarray
) -> np.ndarray:
    """Apply chronological temperature calibration to target scores."""
    temperature = select_temperature(history)
    return win_probabilities(target["race_id"], _log_strengths(scores) / temperature)
