"""Design of experiments: Latin hypercube over bounded design variables (start with 2-4)."""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import qmc

DEFAULT_BOUNDS = {"aoa_deg": (-2.0, 12.0), "ride_height_mm": (30.0, 90.0), "yaw_deg": (0.0, 4.0)}


def latin_hypercube(n: int, bounds: dict[str, tuple[float, float]] = DEFAULT_BOUNDS, seed: int = 0,
                    prefix: str = "geom") -> pd.DataFrame:
    s = qmc.LatinHypercube(d=len(bounds), seed=seed, optimization="random-cd").random(n)
    lo = np.array([b[0] for b in bounds.values()]); hi = np.array([b[1] for b in bounds.values()])
    df = pd.DataFrame(qmc.scale(s, lo, hi), columns=list(bounds))
    df.insert(0, "geometry_id", [f"{prefix}_{i:03d}" for i in range(n)])
    return df.round(4)


def run_sheet(df: pd.DataFrame, rho: float = 1.225, velocity_ms: float = 50.0) -> pd.DataFrame:
    """Case list to hand to ANSYS Fluent (one row per run). Fill cd/cl/residual columns after solving."""
    out = df.copy()
    out["rho"], out["velocity_ms"] = rho, velocity_ms
    for c in ("mesh_cells", "solver_setup", "residual_max", "converged", "cd", "cl"):
        out[c] = np.nan
    out["provenance"] = "ansys_fluent_pending"
    return out
