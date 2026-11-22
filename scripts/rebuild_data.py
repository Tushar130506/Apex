#!/usr/bin/env python
"""Rebuild every data file the models read, with one command.

  python scripts/rebuild_data.py                      # 2018 to the current season
  python scripts/rebuild_data.py --from-step results  # reuse downloaded laps
  python scripts/rebuild_data.py --dry-run            # only print the commands

Steps, in order:

  download  FastF1 sessions into the cache (resumable; new rounds only)
  laps      cache -> data/processed/laps_<year>.parquet
  results   official race results -> data/processed/race_results.parquet
  sprints   official sprint results -> data/processed/sprint_results.parquet
  features  race-level feature table -> <out>/input_features.parquet

Run it again after a race weekend to add the new round: sessions already
downloaded are skipped, and sessions not yet held are ignored. Needs network
access to the FastF1 data servers and `pip install -e ".[ml,fastf1]"`.
"""
from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
STEPS = ("download", "laps", "results", "sprints", "features")
FIRST_SPRINT_SEASON = 2021


def build_commands(args: argparse.Namespace) -> list[tuple[str, list[str]]]:
    """Every step as ``(name, command)``; paths are relative to the repo root."""
    years = [str(year) for year in range(args.first_year, args.last_year + 1)]
    sprint_years = [year for year in years if int(year) >= FIRST_SPRINT_SEASON]
    processed = Path(args.data)
    python = sys.executable
    download = [
        python, "download_fastf1.py", "--years", f"{args.first_year}-{args.last_year}",
        "--sessions", "all", "--cache-dir", args.cache_dir, "--allow-failed",
    ]
    if args.telemetry:
        download.append("--telemetry")
    commands = [
        ("download", download),
        ("laps", [python, "materialize_fastf1.py", "--output", str(processed), "--cache-dir", args.cache_dir]),
        ("results", [
            python, "scripts/build_race_results.py", *years, "--cache-dir", args.cache_dir,
            "--out", str(processed / "race_results.parquet"),
        ]),
    ]
    if sprint_years:
        commands.append(("sprints", [
            python, "scripts/build_race_results.py", *sprint_years, "--event-type", "sprint",
            "--cache-dir", args.cache_dir, "--out", str(processed / "sprint_results.parquet"),
        ]))
    commands.append(("features", [
        python, "scripts/walk_forward_winner.py", "--season", str(args.last_year),
        "--first-season", str(args.first_year), "--data", str(processed),
        "--results", str(processed / "race_results.parquet"), "--out", args.out,
    ]))
    start = STEPS.index(args.from_step)
    return [(name, command) for name, command in commands if STEPS.index(name) >= start]


def missing_rounds(results_path: Path, features_path: Path) -> list[str]:
    """Race ids that have official results but no row in the feature table."""
    import pandas as pd

    results = set(pd.read_parquet(results_path, columns=["race_id"])["race_id"])
    features = set(pd.read_parquet(features_path, columns=["race_id"])["race_id"])
    return sorted(results - features, key=lambda race_id: [int(part) for part in race_id.split("_")])


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--first-year", type=int, default=2018)
    parser.add_argument("--last-year", type=int, default=date.today().year)
    parser.add_argument("--from-step", choices=STEPS, default="download")
    parser.add_argument("--telemetry", action="store_true", help="also download car telemetry (large)")
    parser.add_argument("--data", default="data/processed")
    parser.add_argument("--cache-dir", default=".fastf1-cache")
    parser.add_argument("--out", default="runs/walk_forward_championship", help="folder for input_features.parquet")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.last_year < args.first_year:
        parser.error("--last-year must not be before --first-year")
    return args


def main() -> None:
    args = parse_args()
    commands = build_commands(args)
    for number, (name, command) in enumerate(commands, 1):
        print(f"\n[{number}/{len(commands)}] {name}: {' '.join(command)}", flush=True)
        if args.dry_run:
            continue
        (ROOT / args.data).mkdir(parents=True, exist_ok=True)
        done = subprocess.run(command, cwd=ROOT)
        if done.returncode != 0:
            raise SystemExit(
                f"step '{name}' failed (exit code {done.returncode}). "
                f"Fix the cause and continue with: python scripts/rebuild_data.py --from-step {name}"
            )
    if args.dry_run:
        return
    print(f"\nDone. Feature table: {args.out}/input_features.parquet")
    results = ROOT / args.data / "race_results.parquet"
    features = ROOT / args.out / "input_features.parquet"
    if results.exists() and features.exists():
        missing = missing_rounds(results, features)
        if missing:
            print(
                f"WARNING: {len(missing)} race(s) have results but no lap data, so they are not in the "
                f"feature table and their points are not counted: {', '.join(missing)}.\n"
                f"Add them with: python scripts/export_extra_results.py --results {Path(args.data) / 'race_results.parquet'} "
                f"--features {Path(args.out) / 'input_features.parquet'}, then pass the CSVs as --extra-results."
            )


if __name__ == "__main__":
    main()
