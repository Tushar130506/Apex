"""Parametric pace model used by the strategy simulator (and as an interpretable degradation baseline)."""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
import pandas as pd

DRY = ("SOFT", "MEDIUM", "HARD")
# Per-lap gain from fuel burn and track grip against lap length, fitted on
# 2021-2026 dry races (runs/race_pace, runs/track_layout).
PROGRESS_INTERCEPT_S = 0.0248
PROGRESS_PER_KM_S = -0.0161


@dataclass
class ParametricPace:
    """lap = session_base + offset[compound] + deg[compound] * tyre_age + fuel_s_per_kg * fuel_kg."""
    offset: dict = field(default_factory=lambda: {"SOFT": -0.7, "MEDIUM": 0.0, "HARD": 0.5})
    # Defaults put a MEDIUM tyre 0.9 s slower after 20 laps, the median measured
    # from 2021-2026 race laps (runs/race_pace); SOFT and HARD keep their ratios.
    deg: dict = field(default_factory=lambda: {"SOFT": 0.054, "MEDIUM": 0.033, "HARD": 0.019})
    fuel_s_per_kg: float = 0.032
    burn_kg_per_lap: float = 1.6
    deg_quad: dict = field(default_factory=lambda: {"SOFT": 0.0009, "MEDIUM": 0.0006, "HARD": 0.00036})
    fit_provenance: str = "default_assumptions"

    def rel(self, compound: str, age, fuel_kg, deg_mult=1.0):
        age_array = np.asarray(age)
        return (
            self.offset[compound]
            + (
                self.deg[compound] * age_array
                + self.deg_quad.get(compound, 0.0) * age_array**2
            ) * deg_mult
            + self.fuel_s_per_kg * fuel_kg
        )

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
                   deg_quad={c: 0.0 for c in DRY},
                   burn_kg_per_lap=burn_kg_per_lap, fit_provenance=f"fit:{d['source'].iloc[0]}:{d['session_id'].nunique()}sessions")

    @classmethod
    def from_race_pace(
        cls,
        progress_s_per_lap: float,
        deg_s_per_lap: float,
        deg_quad_s_per_lap2: float = 0.0,
        burn_kg_per_lap: float = 1.6,
        provenance: str = "race_pace",
    ) -> "ParametricPace":
        """Pace model scaled to one circuit's measured race pace.

        The measurements (``apex.features.race_pace``) average over compounds,
        so they set the MEDIUM tyre and the other compounds keep their default
        ratios to it. The progress term mixes fuel burn with the track gaining
        grip; it is expressed as a fuel sensitivity because both scale with
        laps run, which is all the simulator needs.
        """
        if burn_kg_per_lap <= 0:
            raise ValueError("burn_kg_per_lap must be positive")
        base = cls()
        if deg_s_per_lap < 0 or deg_quad_s_per_lap2 < 0:
            # A curve that dips (graining clearing, track rubbering in) is
            # replaced by the straight line with the same 20-lap loss,
            # floored at no degradation.
            deg_s_per_lap = max(deg_s_per_lap * 20 + deg_quad_s_per_lap2 * 400, 0.0) / 20
            deg_quad_s_per_lap2 = 0.0
        scale = deg_s_per_lap / base.deg["MEDIUM"]
        quad_scale = deg_quad_s_per_lap2 / base.deg_quad["MEDIUM"]
        return cls(
            offset=dict(base.offset),
            deg={compound: value * scale for compound, value in base.deg.items()},
            deg_quad={compound: value * quad_scale for compound, value in base.deg_quad.items()},
            fuel_s_per_kg=max(-progress_s_per_lap, 0.0) / burn_kg_per_lap,
            burn_kg_per_lap=burn_kg_per_lap,
            fit_provenance=provenance,
        )

    @classmethod
    def for_lap_length(cls, length_km: float, burn_kg_per_lap: float = 1.6) -> "ParametricPace":
        """Pace model for a circuit with no measured races, from its lap length.

        Across 126 dry races (2021-2026) the per-lap gain from fuel burn and
        track grip followed lap length (Spearman -0.75; scripts/track_layout_study.py):
        ``progress = 0.0248 - 0.0161 * length_km`` s/lap. No layout measure
        predicted tyre wear, so wear stays at the default.
        """
        if length_km <= 0:
            raise ValueError("length_km must be positive")
        base = cls()
        progress = min(PROGRESS_INTERCEPT_S + PROGRESS_PER_KM_S * length_km, 0.0)
        return cls.from_race_pace(progress, base.deg["MEDIUM"], base.deg_quad["MEDIUM"], burn_kg_per_lap,
                                  provenance=f"lap_length:{length_km:.3f}km")

    @classmethod
    def for_circuit(cls, circuits_csv: str, circuit: str, burn_kg_per_lap: float = 1.6) -> "ParametricPace":
        """Pace model from ``runs/race_pace/circuits.csv`` (median of measured races)."""
        table = pd.read_csv(circuits_csv).set_index("circuit")
        if circuit not in table.index:
            raise KeyError(f"{circuit} not measured; known: {sorted(table.index)[:10]}...")
        row = table.loc[circuit]
        return cls.from_race_pace(
            float(row["progress_s_per_lap"]), float(row["deg_s_per_lap"]),
            float(row.get("deg_quad_s_per_lap2", 0.0)), burn_kg_per_lap,
            provenance=f"race_pace:{circuit}:{int(row['races'])}races",
        )

    def estimate_base(self, recent: pd.DataFrame) -> float:
        """Session base pace from recent clean laps (compound/age/fuel terms removed)."""
        r = recent[recent["valid"] & ~recent["is_pit_lap"]]
        if r.empty:
            raise ValueError("no clean laps to anchor pace")
        resid = [t - self.rel(c, a, f) for t, c, a, f in zip(r.lap_time_s, r.compound, r.tyre_age, r.fuel_kg)]
        return float(np.median(resid))
