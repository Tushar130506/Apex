"""CFD result contract. Every CFD row links geometry, solver setup and convergence."""
from __future__ import annotations
import pandas as pd
from apex.schemas.contracts import AERO_COLUMNS, ContractError

REAL_PROVENANCE = ("ansys_fluent",)


def validate_cfd(df: pd.DataFrame, residual_tol: float = 1e-3, allowed_provenance=REAL_PROVENANCE) -> pd.DataFrame:
    miss = set(AERO_COLUMNS) - set(df.columns)
    if miss:
        raise ContractError(f"CFD table missing columns: {sorted(miss)}")
    bad_prov = set(df["provenance"]) - set(allowed_provenance)
    if bad_prov:
        raise ContractError(f"provenance {sorted(bad_prov)} not allowed (expected {list(allowed_provenance)})")
    if df["geometry_id"].duplicated().any():
        raise ContractError("duplicate geometry_id")
    if df[["cd", "cl", "rho", "velocity_ms"]].isna().any().any():
        raise ContractError("missing coefficients/conditions")
    if df["solver_setup"].nunique() > 1 or df["velocity_ms"].nunique() > 1 or df["rho"].nunique() > 1:
        raise ContractError("surrogate needs consistent solver setup and operating condition across rows")
    bad = df[(~df["converged"].astype(bool)) | (df["residual_max"] > residual_tol)]
    if len(bad):
        raise ContractError(f"{len(bad)} rows not converged / residual > {residual_tol}: {bad.geometry_id.tolist()[:5]}")
    return df
