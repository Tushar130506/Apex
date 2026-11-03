"""Tyre degradation: robust per-stint slope + forecast error at +k laps."""
from __future__ import annotations
import numpy as np
import pandas as pd


def _robust_line(age: np.ndarray, pace: np.ndarray) -> tuple[float, float]:
    """Fit a fast line after one median-absolute-deviation outlier pass."""
    finite = np.isfinite(age) & np.isfinite(pace)
    age, pace = age[finite], pace[finite]
    if len(age) < 2 or np.unique(age).size < 2:
        return 0.0, float(np.median(pace)) if len(pace) else float("nan")
    slope, intercept = np.polyfit(age, pace, 1)
    residual = pace - (slope * age + intercept)
    center = float(np.median(residual))
    mad = float(np.median(np.abs(residual - center)))
    if mad > 0:
        keep = np.abs(residual - center) <= 3.0 * 1.4826 * mad
        if keep.sum() >= 3:
            slope, intercept = np.polyfit(age[keep], pace[keep], 1)
    return float(slope), float(intercept)


def stint_slopes(df: pd.DataFrame, fuel_s_per_kg: float = 0.0, min_laps: int = 8) -> pd.DataFrame:
    """Robust slope (s/lap of tyre age) per stint on clean laps.

    fuel_s_per_kg is an ASSUMPTION (set 0 to skip). Fuel burn masks degradation, so run a sensitivity
    sweep over plausible values instead of treating one value as truth.
    """
    rows = []
    for (sid, drv, stint), g in df[df["valid"] & ~df["is_pit_lap"]].groupby(["session_id", "driver", "stint_id"]):
        g = g.dropna(subset=["tyre_age", "lap_time_s", "fuel_kg"])
        if len(g) < min_laps:
            continue
        y = g["lap_time_s"] - fuel_s_per_kg * g["fuel_kg"]
        slope, intercept = _robust_line(g["tyre_age"].to_numpy(float), y.to_numpy())
        rows.append(dict(session_id=sid, driver=drv, stint_id=stint, compound=g["compound"].iloc[0],
                         n_laps=len(g), slope_s_per_lap=slope, intercept=intercept))
    return pd.DataFrame(rows)


def forecast_error(df: pd.DataFrame, horizon: int = 3, fuel_s_per_kg: float = 0.0, min_fit: int = 8) -> pd.DataFrame:
    """Fit on the first laps of each stint, forecast +horizon laps by extrapolating the robust line."""
    rows = []
    for (sid, drv, stint), g in df[df["valid"] & ~df["is_pit_lap"]].groupby(["session_id", "driver", "stint_id"]):
        g = g.sort_values("tyre_age")
        g = g.dropna(subset=["tyre_age", "lap_time_s", "fuel_kg"])
        for cut in range(min_fit, len(g) - horizon):
            fit, tgt = g.iloc[:cut], g.iloc[cut + horizon - 1]
            y = fit["lap_time_s"] - fuel_s_per_kg * fit["fuel_kg"]
            slope, intercept = _robust_line(fit["tyre_age"].to_numpy(float), y.to_numpy())
            pred = slope * float(tgt["tyre_age"]) + intercept + fuel_s_per_kg * tgt["fuel_kg"]
            rows.append(dict(session_id=sid, stint_id=stint, horizon=horizon, err=float(tgt["lap_time_s"] - pred)))
    return pd.DataFrame(rows)
