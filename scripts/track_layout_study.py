#!/usr/bin/env python
"""Does circuit layout explain tyre wear, fuel/track gain, safety cars or overtaking?

  python scripts/track_layout_study.py

Layout features come from track centrelines (``apex.features.track_layout``).
Each question is answered leave-one-circuit-out: a ridge regression on the
layout features is fitted on every other circuit's races and predicts the
held-out circuit's races, against simply predicting the other circuits'
median. That is the situation layout is useful for: a circuit with no
history, such as a new venue. Writes runs/track_layout/.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from apex.features.track_layout import LAYOUT_COLUMNS, layout_table, race_circuits

RIDGE_ALPHA = 5.0


def _ridge(x: np.ndarray, y: np.ndarray, alpha: float) -> tuple[np.ndarray, float]:
    centre, scale = x.mean(axis=0), x.std(axis=0)
    scale[scale == 0] = 1.0
    z = (x - centre) / scale
    beta = np.linalg.solve(z.T @ z + alpha * np.eye(z.shape[1]), z.T @ (y - y.mean()))
    return beta / scale, y.mean() - (centre / scale) @ beta


def leave_one_circuit_out(data: pd.DataFrame, target: str, columns: list[str], alpha: float = RIDGE_ALPHA,
                          seed: int = 0) -> dict:
    """Mean absolute error of layout ridge vs the other circuits' median."""
    rows = []
    for circuit in data["circuit"].unique():
        train, test = data[data["circuit"] != circuit], data[data["circuit"] == circuit]
        coef, intercept = _ridge(train[columns].to_numpy(float), train[target].to_numpy(float), alpha)
        layout = test[columns].to_numpy(float) @ coef + intercept
        baseline = train[target].median()
        for value, guess in zip(test[target], layout):
            rows.append({"circuit": circuit, "layout_error": abs(guess - value), "median_error": abs(baseline - value)})
    errors = pd.DataFrame(rows)
    # Bootstrap over circuits, since races at one circuit are not independent.
    per_circuit = errors.groupby("circuit")[["layout_error", "median_error"]].mean()
    delta = (per_circuit["layout_error"] - per_circuit["median_error"]).to_numpy()
    rng = np.random.default_rng(seed)
    means = [rng.choice(delta, len(delta)).mean() for _ in range(5000)]
    coef, _ = _ridge(data[columns].to_numpy(float), data[target].to_numpy(float), alpha)
    circuit_means = data.groupby("circuit")[[target, *columns]].mean()
    return {
        "races": int(len(data)), "circuits": int(data["circuit"].nunique()),
        "layout_mae": float(errors["layout_error"].mean()),
        "median_mae": float(errors["median_error"].mean()),
        "difference_per_circuit": float(delta.mean()),
        "ci95": [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))],
        "share_of_circuits_better": float((delta < 0).mean()),
        "verdict": "helps" if np.quantile(means, 0.975) < 0 else "no measurable gain",
        "spearman_with_circuit_mean": {
            column: float(circuit_means[target].corr(circuit_means[column], method="spearman")) for column in columns
        },
        "ridge_coefficient_per_sd": {
            column: float(value * data[column].std()) for column, value in zip(columns, coef)
        },
    }


def grid_finish_correlation(features: pd.DataFrame) -> pd.DataFrame:
    """Per race: Spearman correlation of grid and finish among classified finishers."""
    frame = features.assign(
        grid=pd.to_numeric(features["grid_position"], errors="coerce").where(lambda g: g > 0),
        finish=pd.to_numeric(features["finish_position"], errors="coerce"),
    ).dropna(subset=["grid", "finish"])
    rows = [
        {"race_id": race_id, "grid_finish_spearman": group["grid"].corr(group["finish"], method="spearman")}
        for race_id, group in frame.groupby("race_id") if len(group) >= 10
    ]
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--races", default="runs/race_pace/races.csv")
    parser.add_argument("--features", default="runs/walk_forward_championship/input_features.parquet")
    parser.add_argument("--out", default="runs/track_layout")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    layouts = layout_table()
    layouts.to_csv(out / "layouts.csv", index=False, float_format="%.4f")
    features = pd.read_parquet(args.features)
    seasons = sorted(features["race_id"].str[:4].astype(int).unique())
    circuits = race_circuits(seasons)
    circuits.to_csv(out / "race_circuits.csv", index=False)

    pace = pd.read_csv(args.races).merge(layouts, on="circuit", how="inner")
    pace["safety_car_laps"] = pace["neutralised_laps"] / pace["drivers"]
    pace["stops_per_driver"] = pace["pit_stops"] / pace["drivers"]
    dry = pace[~pace["mixed_conditions"].astype(bool)]
    order = grid_finish_correlation(features).merge(circuits, on="race_id").merge(layouts, on="circuit")

    questions = {
        "tyre_wear_20_laps_s": (dry, "deg_at_20_laps_s"),
        "fuel_track_gain_s_per_lap": (dry, "progress_s_per_lap"),
        "pit_stops_per_driver": (dry, "stops_per_driver"),
        "safety_car_laps": (pace, "safety_car_laps"),
        "grid_finish_spearman": (order, "grid_finish_spearman"),
    }
    report = {name: leave_one_circuit_out(data, target, LAYOUT_COLUMNS) for name, (data, target) in questions.items()}
    (out / "study.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    for name, result in report.items():
        print(f"{name:28s} layout {result['layout_mae']:.3f}  median {result['median_mae']:.3f}  "
              f"diff {result['difference_per_circuit']:+.3f} [{result['ci95'][0]:+.3f}, {result['ci95'][1]:+.3f}]  "
              f"{result['verdict']}  ({result['circuits']} circuits, {result['races']} races)")


if __name__ == "__main__":
    main()
