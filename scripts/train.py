#!/usr/bin/env python
"""Train on train years, calibrate on validation years, report validation AND test.
  python scripts/train.py --run-name pace_v1                       # years from configs/split.json
  python scripts/train.py --run-name pace_v1 --train-years 2022 2023 --val-years 2024 --test-years 2025"""
import argparse, json, sys
sys.path.insert(0, "src")
from apex.data_io import load_processed
from apex.evaluation.reports import train_run

cfg = json.load(open("configs/split.json"))
ap = argparse.ArgumentParser()
ap.add_argument("--run-name", required=True)
ap.add_argument("--train-years", type=int, nargs="+", default=cfg["train_years"])
ap.add_argument("--val-years", type=int, nargs="+", default=cfg["val_years"])
ap.add_argument("--test-years", type=int, nargs="+", default=cfg["test_years"])
ap.add_argument("--data", default="data/processed")
a = ap.parse_args()
laps = load_processed(a.data, a.train_years + a.val_years + a.test_years)
m = train_run(laps, a.train_years, a.val_years, a.test_years, a.run_name)
print(json.dumps({k: m[k] for k in ("n_laps", "n_sessions")}, indent=2))
for split in ("val", "test"):
    print(f"\n== {split} ==")
    for k in ("naive_prev_lap", "rolling_mean_3", "gbm"):
        print(f"{k:16s} MAE {m[split][k]['mae']:.3f}s  RMSE {m[split][k]['rmse']:.3f}s")
    print("conformal:", {k: round(v, 3) for k, v in m[split]["conformal"].items()}, "| gbm beats baselines:", m[split]["gbm_beats_best_baseline"])
print(f"\nartefacts: runs/{a.run_name}/  predictions: data/predictions/{a.run_name}/predictions.parquet")
