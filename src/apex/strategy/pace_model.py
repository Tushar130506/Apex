"""Parametric pace model used by the strategy simulator (and as an interpretable degradation baseline)."""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
import pandas as pd

DRY = ("SOFT", "MEDIUM", "HARD")


@dataclass
class ParametricPace:
    """lap = session_base + offset[compound] + deg[compound] * tyre_age + fuel_s_per_kg * fuel_kg."""
    offset: dict = field(default_factory=lambda: {"SOFT": -0.7, "MEDIUM": 0.0, "HARD": 0.5})
    deg: dict = field(default_factory=lambda: {"SOFT": 0.09, "MEDIUM": 0.055, "HARD": 0.032})
    fuel_s_per_kg: float = 0.032
    burn_kg_per_lap: float = 1.6
    fit_provenance: str = "default_assumptions"

    def rel(self, compound: str, age, fuel_kg, deg_mult=1.0):
        return self.offset[compound] + self.deg[compound] * np.asarray(age) * deg_mult + self.fuel_s_per_kg * fuel_kg

    @classmethod
    def fit(cls, laps: pd.DataFrame, burn_kg_per_lap: float = 1.6) -> "ParametricPace":
        """Least squares with a per-(session, driver) intercept; MEDIUM offset fixed at 0. Uses clean laps only."""
        d = laps[laps["valid"] & ~laps["is_pit_lap"] & laps["compound"].isin(DRY)].copy()
        sess = pd.get_dummies(d["session_id"] + "|" + d["driver"].astype(str)).astype(float)  # per-driver base pace
        cols = [sess.to_numpy()]
        names = []
        for c in DRY:
            m = (d["compound"] == c).to_numpy().astype(float)
            if c != "MEDIUM":
                cols.append(m[:, None]); names.append(("offset", c))
            cols.append((m * d["tyre_age"].to_numpy())[:, None]); names.append(("deg", c))
        cols.append(d["fuel_kg"].to_numpy()[:, None]); names.append(("fuel", ""))
        X, y = np.hstack(cols), d["lap_time_s"].to_numpy()
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        coef = dict(zip(names, beta[sess.shape[1]:]))
        return cls(offset={"SOFT": coef[("offset", "SOFT")], "MEDIUM": 0.0, "HARD": coef[("offset", "HARD")]},
                   deg={c: float(coef[("deg", c)]) for c in DRY}, fuel_s_per_kg=float(coef[("fuel", "")]),
                   burn_kg_per_lap=burn_kg_per_lap, fit_provenance=f"fit:{d['source'].iloc[0]}:{d['session_id'].nunique()}sessions")

    def estimate_base(self, recent: pd.DataFrame) -> float:
        """Session base pace from recent clean laps (compound/age/fuel terms removed)."""
        r = recent[recent["valid"] & ~recent["is_pit_lap"]]
        if r.empty:
            raise ValueError("no clean laps to anchor pace")
        resid = [t - self.rel(c, a, f) for t, c, a, f in zip(r.lap_time_s, r.compound, r.tyre_age, r.fuel_kg)]
        return float(np.median(resid))
