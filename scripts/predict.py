#!/usr/bin/env python
"""Score laps with a saved run and store predictions.
  python scripts/predict.py --run runs/pace_v1 --years 2025 --out data/predictions/pace_v1/new_2025.parquet"""
import argparse, sys
sys.path.insert(0, "src")
from apex.data_io import load_processed
from apex.evaluation.reports import predict_with_run

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True); ap.add_argument("--years", type=int, nargs="+"); ap.add_argument("--out", required=True)
ap.add_argument("--data", default="data/processed")
a = ap.parse_args()
df = predict_with_run(a.run, load_processed(a.data, a.years), a.out)
print(f"{len(df)} laps scored -> {a.out}; MAE {df.abs_err_gbm_s.mean():.3f}s; interval coverage {df.in_interval.mean():.2f}")
if not df.season_seen_in_training.all():
    print("WARNING: some seasons were not in the training years (regulation-era shift risk)")
