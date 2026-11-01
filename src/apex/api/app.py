"""FastAPI service: strategy recommendation + aero bridge estimate. Run: uvicorn apex.api.app:app"""
from __future__ import annotations
import os
from pathlib import Path

import joblib
from fastapi import FastAPI
from pydantic import BaseModel, Field

from apex.aero.vehicle_bridge import lap_time_delta
from apex.strategy.pace_model import ParametricPace
from apex.strategy.race_simulator import Assumptions, RaceState, recommend

app = FastAPI(title="APEX", version="0.1.0")
def _load_pace() -> ParametricPace:
    """Use the pace fit from a training run: APEX_RUN=runs/<name>. Falls back to labelled default assumptions."""
    run = os.environ.get("APEX_RUN")
    if run and (Path(run) / "model.joblib").exists():
        return joblib.load(Path(run) / "model.joblib")["pace"]
    return ParametricPace()


PACE = _load_pace()


class StrategyRequest(BaseModel):
    lap: int = Field(ge=1); total_laps: int; compound: str; tyre_age: int = Field(ge=0)
    fuel_kg: float = Field(ge=0); base_pace: float
    data_ok: bool = True; data_msg: str = "ok"; ood_reasons: list[str] = []
    pit_loss_mean: float = 22.0; sc_probability: float = 0.25; n_samples: int = Field(1000, le=5000)


@app.get("/health")
def health():
    return {"status": "ok", "pace_provenance": PACE.fit_provenance}


@app.post("/strategy")
def strategy(r: StrategyRequest):
    st = RaceState(r.lap, r.total_laps, r.compound, r.tyre_age, r.fuel_kg, r.base_pace)
    a = Assumptions(pit_loss_mean=r.pit_loss_mean, sc_prob_per_race_remaining=r.sc_probability)
    out = recommend(st, PACE, a, r.n_samples, data_ok=r.data_ok, data_msg=r.data_msg,
                    ood_reasons=r.ood_reasons)
    if "table" in out:
        out["table"] = out["table"].round(3).to_dict("records")
    return out


class AeroRequest(BaseModel):
    cd: float; cl: float; base_cd: float; base_cl: float


@app.post("/aero/lap-delta")
def aero_lap_delta(r: AeroRequest):
    return {"est_lap_delta_s": lap_time_delta((r.cd, r.cl), (r.base_cd, r.base_cl)),
            "label": "ESTIMATE from unvalidated generic vehicle model"}
