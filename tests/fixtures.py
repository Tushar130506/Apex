"""TEST-ONLY lap generator (source='test_fixture'). Not part of the package, never used for training.

Ground truth is known, so tests can check that models recover degradation/fuel effects.
Never report metrics from this data as evidence about real racing.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

COMPOUNDS = {  # offset s vs MEDIUM, degradation s/lap
    "SOFT": (-0.7, 0.090), "MEDIUM": (0.0, 0.055), "HARD": (0.5, 0.032)}
FUEL_EFFECT_S_PER_KG = 0.032
BURN_KG_PER_LAP = 1.6
TRUTH = {"compounds": COMPOUNDS, "fuel_s_per_kg": FUEL_EFFECT_S_PER_KG, "burn": BURN_KG_PER_LAP}


def make_session(session_id: str, track: str, base: float, season: int, laps: int, pit_laps: list[int],
                 compounds: list[str], rng: np.random.Generator, noise: float = 0.18,
                 track_temp: float = 35.0) -> pd.DataFrame:
    rows, stint, age, fuel0 = [], 0, 0, 100.0
    comp = compounds[0]
    for lap in range(1, laps + 1):
        pit = lap in pit_laps
        off, deg = COMPOUNDS[comp]
        fuel = fuel0 - BURN_KG_PER_LAP * (lap - 1)
        temp = track_temp + 3 * np.sin(lap / 12)
        t = base + off + deg * age * (1 + 0.01 * (temp - 35)) + FUEL_EFFECT_S_PER_KG * fuel + rng.normal(0, noise)
        invalid = rng.random() < 0.03
        if pit:
            t += 21.5
        reason = "pit_lap" if pit else ("invalid_lap_flag" if invalid else None)
        rows.append(dict(source="test_fixture", season=season, session_id=session_id, track=track, driver="SIM",
                         lap_no=lap, lap_time_s=t, valid=reason is None, exclusion_reason=reason,
                         compound=comp, tyre_age=age, stint_id=stint, fuel_kg=fuel,
                         fuel_provenance="estimated", track_temp=temp, air_temp=temp - 8,
                         track_status="green", is_pit_lap=pit))
        age += 1
        if lap in pit_laps:
            stint += 1; age = 0; comp = compounds[min(stint, len(compounds) - 1)]
    return pd.DataFrame(rows)


def make_dataset(n_sessions: int = 48, seed: int = 0, seasons=(2022, 2023, 2024, 2025)) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    tracks = {"alpha": 88.0, "bravo": 95.5, "charlie": 79.2, "delta": 102.3}
    out = []
    for i in range(n_sessions):
        trk = list(tracks)[i % len(tracks)]
        laps = int(rng.integers(50, 66))
        pit = int(rng.integers(15, laps - 15))
        comps = list(rng.choice(list(COMPOUNDS), 2, replace=False))
        out.append(make_session(f"fx_{i:03d}", trk, tracks[trk] + rng.normal(0, 0.1), seasons[i % len(seasons)], laps, [pit], comps, rng,
                                track_temp=float(rng.uniform(25, 45))))
    return pd.concat(out, ignore_index=True)


def toy_cfd(designs, seed: int = 0):
    """NOT CFD: smooth analytic stand-in so the aero code can be unit-tested before ANSYS access exists."""
    import numpy as np
    from apex.schemas.contracts import AERO_COLUMNS
    r = np.random.default_rng(seed)
    a, h, y = designs["aoa_deg"], designs["ride_height_mm"], designs["yaw_deg"]
    cl = 1.2 + 0.11 * a - 0.006 * (h - 50) - 0.0035 * (a - 8) ** 2 - 0.03 * y
    cd = 0.35 + 0.012 * a + 0.00018 * (a ** 2) + 0.0008 * (h - 50) / 10 + 0.01 * y
    out = designs.copy()
    out["cd"] = cd + r.normal(0, 0.002, len(out)); out["cl"] = cl + r.normal(0, 0.004, len(out))
    out["rho"], out["velocity_ms"] = 1.225, 50.0
    out["mesh_cells"], out["solver_setup"] = 2_000_000, "toy_v1"
    out["residual_max"], out["converged"] = 1e-4, True
    out["provenance"] = "test_fixture"
    return out[AERO_COLUMNS]
