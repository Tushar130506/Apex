"""Compound-specific dynamic tyre degradation."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from apex.models.degradation import _robust_line


@dataclass(frozen=True)
class CompoundDegradation:
    """Linear and quadratic time loss by tyre compound and age."""

    coefficients: dict[str, tuple[float, float]]
    fallback: tuple[float, float] = (0.05, 0.001)

    def loss(self, compound: str, tyre_age: np.ndarray | float) -> np.ndarray:
        linear, quadratic = self.coefficients.get(compound.upper(), self.fallback)
        age = np.maximum(0.0, np.asarray(tyre_age, dtype=float))
        return linear * age + quadratic * age**2

    @classmethod
    def fit(
        cls, laps: pd.DataFrame, fuel_s_per_kg: float = 0.032, min_laps: int = 8
    ) -> "CompoundDegradation":
        required = {"valid", "is_pit_lap", "compound", "tyre_age", "lap_time_s", "fuel_kg"}
        missing = required - set(laps.columns)
        if missing:
            raise ValueError(f"laps missing columns: {sorted(missing)}")
        clean = laps[
            laps["valid"].fillna(False)
            & ~laps["is_pit_lap"].fillna(False)
            & laps["compound"].notna()
        ].dropna(subset=["tyre_age", "lap_time_s", "fuel_kg"])
        rows: dict[str, tuple[float, float]] = {}
        for compound, group in clean.groupby(clean["compound"].astype(str).str.upper()):
            if len(group) < min_laps:
                continue
            age = group["tyre_age"].to_numpy(float)
            pace = group["lap_time_s"].to_numpy(float) - fuel_s_per_kg * group["fuel_kg"]
            slope, _ = _robust_line(age, pace)
            centered = pace - slope * age
            quadratic = max(0.0, float(np.polyfit(age, centered, 2)[0])) if len(group) >= 3 else 0.0
            rows[compound] = (max(0.0, slope), quadratic)
        return cls(rows)
