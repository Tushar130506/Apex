"""Year-wise training run + prediction store.

Layout produced by `train_run`:
  runs/<name>/model.joblib        gbm + conformal wrapper + fitted pace model
  runs/<name>/metrics.json        baseline-vs-model metrics for validation AND test seasons
  runs/<name>/config.json         split years, seed, alpha, library versions
  data/predictions/<name>/predictions.parquet   one row per scored lap (train/val/test labelled)
"""
from __future__ import annotations
import json, platform
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from apex.evaluation.metrics import group_mae, interval_coverage, mae, rmse
from apex.evaluation.splits import season_split
from apex.features.lap_features import FEATURES, build_features
from apex.models.pace import ConformalPace, GBMPace, NaivePrevLap, StintMedianBaseline
from apex.strategy.pace_model import ParametricPace

ID_COLS = ["split", "season", "session_id", "track", "driver", "lap_no", "compound", "tyre_age"]


def _score(rows: pd.DataFrame, models: list, cp: ConformalPace, split: str) -> pd.DataFrame:
    out = rows[[c for c in ID_COLS if c != "split"]].copy()
    out.insert(0, "split", split)
    out["actual_s"] = rows["lap_time_s"].to_numpy()
    out["prev_lap_s"] = rows["prev_lap_time"].to_numpy()
    for m in models:
        out[f"pred_{m.name}_s"] = (rows["prev_lap_time"] + m.predict_delta(rows)).to_numpy()
    pi = cp.predict(rows)
    out["pred_lo_s"], out["pred_hi_s"] = pi["lo"].to_numpy(), pi["hi"].to_numpy()
    out["in_interval"] = (out.actual_s >= out.pred_lo_s) & (out.actual_s <= out.pred_hi_s)
    out["abs_err_gbm_s"] = (out.actual_s - out["pred_gbm_s"]).abs()
    return out


def train_run(laps: pd.DataFrame, train_years, val_years, test_years, run_name: str, runs_dir="runs",
              pred_dir="data/predictions", alpha: float = 0.1, seed: int = 0, allow_fixture: bool = False) -> dict:
    allowed = {"fastf1_real"} | ({"test_fixture"} if allow_fixture else set())
    if not set(laps.source.unique()) <= allowed:
        raise ValueError(f"training refuses sources {sorted(set(laps.source.unique()) - allowed)}")
    f = build_features(laps)
    m = f[f.model_row].copy()
    masks = season_split(m, train_years, val_years, test_years)
    tr, va, te = m[masks["train"]], m[masks["val"]], m[masks["test"]]

    gbm = GBMPace(seed).fit(tr, tr["y_delta"])
    models = [NaivePrevLap(), StintMedianBaseline(), gbm]
    cp = ConformalPace(gbm, alpha).calibrate(tr, va, va["y_delta"])      # calibrated on validation seasons only
    pace = ParametricPace.fit(f[f.season.isin(train_years)])             # strategy pace model: train seasons only

    scored = pd.concat([_score(tr, models, cp, "train"), _score(va, models, cp, "val"), _score(te, models, cp, "test")],
                       ignore_index=True)
    metrics = {"n_laps": {k: int(len(v)) for k, v in dict(train=tr, val=va, test=te).items()},
               "n_sessions": {k: int(v.session_id.nunique()) for k, v in dict(train=tr, val=va, test=te).items()}}
    for split in ("val", "test"):
        s = scored[scored.split == split]
        src = va if split == "val" else te
        sm = {}
        for mod in models:
            col = f"pred_{mod.name}_s"
            sm[mod.name] = dict(mae=mae(s.actual_s, s[col]), rmse=rmse(s.actual_s, s[col]),
                                mae_by_track=group_mae(s, "actual_s", col, "track"),
                                mae_by_compound=group_mae(s, "actual_s", col, "compound"),
                                mae_by_season=group_mae(s, "actual_s", col, "season"))
        sm["conformal"] = dict(target=1 - alpha, coverage=interval_coverage(s.actual_s, s.pred_lo_s, s.pred_hi_s),
                               mean_width_s=float((s.pred_hi_s - s.pred_lo_s).mean()))
        best_base = min(sm["naive_prev_lap"]["mae"], sm["rolling_mean_3"]["mae"])
        sm["gbm_beats_best_baseline"] = bool(sm["gbm"]["mae"] < best_base)
        metrics[split] = sm
    # leave drivers/teams/era caveats visible in the artefact itself
    metrics["notes"] = ["fuel is an estimate (FastF1 has no fuel mass)",
                        "test seasons should be evaluated once, after all model choices are frozen"]

    run, pdir = Path(runs_dir) / run_name, Path(pred_dir) / run_name
    run.mkdir(parents=True, exist_ok=True); pdir.mkdir(parents=True, exist_ok=True)
    joblib.dump(dict(gbm=gbm, cp=cp, pace=pace, features=FEATURES, train_years=list(train_years)), run / "model.joblib")
    cfg = dict(run_name=run_name, train_years=list(train_years), val_years=list(val_years), test_years=list(test_years),
               alpha=alpha, seed=seed, python=platform.python_version())
    (run / "config.json").write_text(json.dumps(cfg, indent=2))
    (run / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
    scored.to_parquet(pdir / "predictions.parquet")
    return metrics


def predict_with_run(run_dir: str | Path, laps: pd.DataFrame, out_path: str | Path | None = None) -> pd.DataFrame:
    """Score completed laps (one-step-ahead forecast of each lap from the laps before it) with a saved run."""
    art = joblib.load(Path(run_dir) / "model.joblib")
    f = build_features(laps)
    m = f[f.model_row].copy()
    scored = _score(m, [NaivePrevLap(), StintMedianBaseline(), art["gbm"]], art["cp"], "predict")
    scored["season_seen_in_training"] = scored.season.isin(art["train_years"])   # era-shift warning flag
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        scored.to_parquet(out_path)
    return scored
