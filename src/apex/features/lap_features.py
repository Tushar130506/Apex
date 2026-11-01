"""Lap/stint features. Only past information from the same stint is used (no look-ahead)."""
from __future__ import annotations
import numpy as np
import pandas as pd

COMPOUND_ORDER = ["SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET", "UNKNOWN"]
# Track-relative features only (no absolute lap times) so the model transfers across circuits.
FEATURES = ["tyre_age", "stint_lap", "fuel_kg", "track_temp", "air_temp", "compound_code",
            "trend_3", "dev_roll"]


def build_features(laps: pd.DataFrame) -> pd.DataFrame:
    """Adds clean-lap flags, previous-lap features and the residual target `y_delta`.

    Target is lap_time - previous clean lap time in the same stint (so a naive 'previous lap'
    baseline predicts delta=0 and any model must add information beyond it).
    """
    df = laps.sort_values(["session_id", "driver", "lap_no"]).copy()
    df["compound_code"] = df["compound"].map({c: i for i, c in enumerate(COMPOUND_ORDER)}).fillna(5).astype(int)
    g = df.groupby(["session_id", "driver", "stint_id"], sort=False)
    df["stint_lap"] = g.cumcount()
    clean = df["valid"].fillna(False)
    df["_t_clean"] = df["lap_time_s"].where(clean)
    df["_f_clean"] = df["fuel_kg"].where(clean)
    g = df.groupby(["session_id", "driver", "stint_id"], sort=False)
    # shift(1) of forward-filled clean lap times => previous clean lap, never the current one
    df["prev_lap_time"] = g["_t_clean"].transform(lambda s: s.ffill().shift(1))
    df["prev_fuel_kg"] = g["_f_clean"].transform(lambda s: s.ffill().shift(1))
    df["roll_mean_3"] = g["_t_clean"].transform(lambda s: s.ffill().shift(1).rolling(3, min_periods=1).mean())
    df["trend_3"] = g["_t_clean"].transform(lambda s: s.ffill().shift(1).diff().rolling(3, min_periods=1).mean())
    df["dev_roll"] = df["roll_mean_3"] - df["prev_lap_time"]   # denoised level vs last lap
    df["y_delta"] = df["lap_time_s"] - df["prev_lap_time"]
    df["model_row"] = clean & df["prev_lap_time"].notna() & ~df["is_pit_lap"] & df["trend_3"].notna()
    return df.drop(columns=["_t_clean", "_f_clean"])


def fuel_corrected(df: pd.DataFrame, s_per_kg: float, ref_fuel: float = 0.0) -> pd.Series:
    """Fuel-corrected lap time. s_per_kg is an *assumption* with a sensitivity range, not truth."""
    return df["lap_time_s"] - s_per_kg * (df["fuel_kg"] - ref_fuel)
