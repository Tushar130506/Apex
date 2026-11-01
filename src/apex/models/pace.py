"""Lap-pace model ladder: naive baseline -> gradient boosting (+ conformal intervals).

All models predict the residual y_delta = lap_time - previous clean lap; see features.lap_features.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
import pandas as pd

from apex.features.lap_features import FEATURES

try:  # LightGBM is optional; fall back to sklearn's histogram GBM
    from lightgbm import LGBMRegressor
except Exception:  # pragma: no cover
    LGBMRegressor = None
from sklearn.ensemble import HistGradientBoostingRegressor


class NaivePrevLap:
    """Baseline: next lap equals previous clean lap (delta = 0)."""
    name = "naive_prev_lap"
    def fit(self, X, y): return self
    def predict_delta(self, X): return np.zeros(len(X))


class StintMedianBaseline:
    """Baseline: lap time = median of the clean laps so far in the stint (via roll_mean_3 proxy)."""
    name = "rolling_mean_3"
    def fit(self, X, y): return self
    def predict_delta(self, X): return (X["roll_mean_3"] - X["prev_lap_time"]).to_numpy()


def _gbm(seed: int = 0, **kw):
    if LGBMRegressor is not None:
        return LGBMRegressor(n_estimators=200, learning_rate=0.03, num_leaves=7, min_child_samples=40,
                             subsample=0.8, subsample_freq=1, colsample_bytree=0.9, random_state=seed,
                             verbose=-1, **kw)
    return HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, max_leaf_nodes=7, min_samples_leaf=40, random_state=seed, **kw)


class GBMPace:
    name = "gbm"
    def __init__(self, seed: int = 0):
        self.m = _gbm(seed)
    def fit(self, X, y):
        self.m.fit(X[FEATURES], y); return self
    def predict_delta(self, X): return self.m.predict(X[FEATURES])


@dataclass
class ConformalPace:
    """Split-conformal wrapper: intervals calibrated on held-out *sessions* (never train rows)."""
    base: object
    alpha: float = 0.1
    q_: float = field(default=float("nan"), init=False)
    domain_: dict = field(default_factory=dict, init=False)

    def calibrate(self, X_train: pd.DataFrame, X_cal: pd.DataFrame, y_cal) -> "ConformalPace":
        res = np.abs(np.asarray(y_cal) - self.base.predict_delta(X_cal))
        n = len(res)
        level = min(1.0, np.ceil((n + 1) * (1 - self.alpha)) / n)
        self.q_ = float(np.quantile(res, level))
        self.domain_ = {c: (float(X_train[c].quantile(0.005)), float(X_train[c].quantile(0.995)))
                        for c in ("tyre_age", "fuel_kg", "track_temp") if c in X_train}
        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        d = self.base.predict_delta(X)
        p = X["prev_lap_time"].to_numpy() + d
        return pd.DataFrame({"pred": p, "lo": p - self.q_, "hi": p + self.q_}, index=X.index)

    def out_of_domain(self, row: dict) -> list[str]:
        return [f"{k}={row[k]:.1f} outside training range [{lo:.1f},{hi:.1f}]"
                for k, (lo, hi) in self.domain_.items() if k in row and not lo <= row[k] <= hi]
