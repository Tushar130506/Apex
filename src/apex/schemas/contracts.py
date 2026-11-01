"""Data contracts (blueprint section 3). Every table carries an explicit `source`."""
from __future__ import annotations
import pandas as pd

SOURCES = {"fastf1_real", "test_fixture"}   # test_fixture is for unit tests only; training refuses it

LAP_COLUMNS = {
    "source": "object", "season": "int", "session_id": "object", "track": "object", "driver": "object",
    "lap_no": "int", "lap_time_s": "float", "valid": "bool", "exclusion_reason": "object",
    "compound": "object", "tyre_age": "int", "stint_id": "int", "fuel_kg": "float",
    "fuel_provenance": "object",  # 'estimated' | 'missing' (real-world fuel mass is never directly measured)
    "track_temp": "float", "air_temp": "float", "track_status": "object", "is_pit_lap": "bool",
}
AERO_COLUMNS = [
    "geometry_id", "provenance", "aoa_deg", "ride_height_mm", "yaw_deg", "rho", "velocity_ms",
    "cd", "cl", "mesh_cells", "solver_setup", "residual_max", "converged",
]


class ContractError(ValueError):
    pass


def validate_laps(df: pd.DataFrame) -> pd.DataFrame:
    missing = set(LAP_COLUMNS) - set(df.columns)
    if missing:
        raise ContractError(f"lap table missing columns: {sorted(missing)}")
    bad = set(df["source"].unique()) - SOURCES
    if bad:
        raise ContractError(f"unknown source labels: {bad}")
    if df.groupby("session_id")["source"].nunique().max() > 1:
        raise ContractError("a session must have exactly one source label")
    if (df["valid"] == False).any() and df.loc[~df["valid"], "exclusion_reason"].isna().any():  # noqa: E712
        raise ContractError("invalid laps must preserve an exclusion_reason")
    if df.duplicated(["session_id", "driver", "lap_no"]).any():
        raise ContractError("duplicate (session_id, driver, lap_no)")
    return df


def quality_report(df: pd.DataFrame) -> dict:
    """Small data-quality summary for the M2 notebook / docs."""
    return {
        "rows": len(df),
        "sessions": df["session_id"].nunique(),
        "stints": df.groupby(["session_id", "driver", "stint_id"]).ngroups,
        "sources": df.groupby("source")["session_id"].nunique().to_dict(),
        "sessions_per_season": df.groupby("season")["session_id"].nunique().to_dict(),
        "valid_fraction": float(df["valid"].mean()),
        "exclusions": df.loc[~df["valid"], "exclusion_reason"].value_counts().to_dict(),
        "fuel_provenance": df["fuel_provenance"].value_counts().to_dict(),
        "missing_fraction": df[["track_temp", "air_temp", "fuel_kg"]].isna().mean().round(3).to_dict(),
    }
