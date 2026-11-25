#!/usr/bin/env python
"""Out-of-sample check of the per-circuit race-pace numbers.

  python scripts/benchmark_race_pace.py

For every dry race whose circuit was measured before (and with at least 20
earlier dry races), predicts its progress (fuel + track, s/lap) and its
20-lap tyre loss from earlier races only, five ways: the strategy model's
built-in defaults, the median of all earlier races, the median of earlier
races at the same circuit, the last race there, and the circuit median pulled
towards the all-track median as ``circuit_table`` does (shrinkage estimated
from earlier races). The built-in default wear was itself set from the
2021-2026 median, so its row is not out of sample. Reports mean absolute
errors and a paired bootstrap of "same circuit" against "all earlier races".
Reads runs/race_pace/races.csv from race_pace_study.py.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from apex.features.form_features import race_order
from apex.features.race_pace import shrink_weight, wear_shrinkage
from apex.strategy.pace_model import ParametricPace

MIN_EARLIER = 20


def predictions(races: pd.DataFrame) -> pd.DataFrame:
    dry = races[~races["mixed_conditions"].astype(bool)].assign(_order=lambda f: race_order(f["race_id"]))
    dry = dry.sort_values("_order")
    default = ParametricPace()
    default_progress = -default.fuel_s_per_kg * default.burn_kg_per_lap
    default_deg20 = default.deg["MEDIUM"] * 20 + default.deg_quad["MEDIUM"] * 400
    rows = []
    for _, race in dry.iterrows():
        past = dry[dry["_order"] < race["_order"]]
        same = past[past["circuit"] == race["circuit"]]
        if same.empty or len(past) < MIN_EARLIER:
            continue
        between, within = wear_shrinkage(past)
        weight = shrink_weight(len(same), between, within)
        all_deg20 = past["deg_at_20_laps_s"].median()
        methods = {
            "built-in default": (default_progress, default_deg20),
            "all earlier races": (past["progress_s_per_lap"].median(), past["deg_at_20_laps_s"].median()),
            "same circuit, earlier races": (same["progress_s_per_lap"].median(), same["deg_at_20_laps_s"].median()),
            "same circuit, last race": (same["progress_s_per_lap"].iloc[-1], same["deg_at_20_laps_s"].iloc[-1]),
            "same circuit, shrunk": (
                same["progress_s_per_lap"].median(),
                weight * same["deg_at_20_laps_s"].median() + (1 - weight) * all_deg20,
            ),
        }
        for method, (progress, deg20) in methods.items():
            rows.append({
                "race_id": race["race_id"], "season": int(race["season"]), "method": method,
                "progress_error": abs(progress - race["progress_s_per_lap"]),
                "deg20_error": abs(deg20 - race["deg_at_20_laps_s"]),
            })
    return pd.DataFrame(rows)


def paired(errors: pd.DataFrame, column: str, method: str = "same circuit, earlier races", seed: int = 0) -> dict:
    wide = errors.pivot(index="race_id", columns="method", values=column)
    delta = (wide[method] - wide["all earlier races"]).to_numpy()
    rng = np.random.default_rng(seed)
    means = [rng.choice(delta, len(delta)).mean() for _ in range(5000)]
    return {
        "mean_difference": float(delta.mean()),
        "ci95_low": float(np.quantile(means, 0.025)),
        "ci95_high": float(np.quantile(means, 0.975)),
        "share_circuit_better": float((delta < 0).mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--races", default="runs/race_pace/races.csv")
    parser.add_argument("--out", default="runs/race_pace/benchmark.json")
    args = parser.parse_args()
    errors = predictions(pd.read_csv(args.races))
    summary = errors.groupby("method").agg(
        races=("race_id", "nunique"),
        progress_mae_s_per_lap=("progress_error", "mean"),
        deg20_mae_s=("deg20_error", "mean"),
    ).sort_values("deg20_mae_s")
    report = {
        "races": int(errors["race_id"].nunique()),
        "methods": summary.round(4).to_dict(orient="index"),
        "same_circuit_minus_all_races": {
            "progress": paired(errors, "progress_error"),
            "deg20": paired(errors, "deg20_error"),
        },
        "shrunk_minus_all_races_deg20": paired(errors, "deg20_error", "same circuit, shrunk"),
        "deg20_mae_by_season": errors.groupby(["season", "method"])["deg20_error"].mean().unstack().round(4)
        .to_dict(orient="index"),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(summary.round(3).to_string())
    print(json.dumps({key: report[key] for key in ("same_circuit_minus_all_races", "shrunk_minus_all_races_deg20")},
                     indent=2))


if __name__ == "__main__":
    main()
