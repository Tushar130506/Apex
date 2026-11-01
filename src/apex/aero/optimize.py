"""Surrogate-based candidate search. Winners MUST be re-run in CFD before any claim is made."""
from __future__ import annotations
import numpy as np
import pandas as pd
from apex.aero.doe import DEFAULT_BOUNDS, latin_hypercube
from apex.aero.vehicle_bridge import lap_time_delta
from apex.models.aero_surrogate import AeroSurrogate


def propose(surr: AeroSurrogate, baseline_cd_cl: tuple[float, float], n_search: int = 2000, top_k: int = 5,
            bounds: dict = DEFAULT_BOUNDS, min_cl: float | None = None, max_std: float = 0.05, seed: int = 1) -> pd.DataFrame:
    cand = latin_hypercube(n_search, bounds, seed=seed, prefix="cand")
    p = surr.predict(cand)
    cand["cd_pred"], cand["cl_pred"] = p.mean[:, 0], p.mean[:, 1]
    cand["cd_std"], cand["cl_std"] = p.std[:, 0], p.std[:, 1]
    cand["extrapolating"] = p.extrapolating
    ok = ~cand["extrapolating"] & (cand[["cd_std", "cl_std"]].max(axis=1) <= max_std)
    if min_cl is not None:
        ok &= cand["cl_pred"] >= min_cl
    cand = cand[ok].copy()
    # coarse prefilter on L/D-like proxy, then bridge estimate on the shortlist
    cand["proxy"] = cand["cl_pred"] / cand["cd_pred"]
    short = cand.nlargest(min(40, len(cand)), "proxy").copy()
    short["est_lap_delta_s"] = [lap_time_delta((cd, cl), baseline_cd_cl) for cd, cl in zip(short.cd_pred, short.cl_pred)]
    out = short.nsmallest(top_k, "est_lap_delta_s").reset_index(drop=True)
    out["status"] = "SURROGATE_ESTIMATE_NEEDS_CFD_RERUN"
    return out
