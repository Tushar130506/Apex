#!/usr/bin/env python
"""Download real race data with FastF1 -> data/processed/laps_<year>.parquet (one file per season).
  python scripts/build_dataset.py season 2022 2023 2024 2025
  python scripts/build_dataset.py season 2024 --max-events 3      # quick trial
  python scripts/build_dataset.py race 2024 Monza                 # -> laps_2024_Monza.parquet
  python scripts/build_dataset.py cache 2018 2019 2020 --sessions all --telemetry"""
import argparse, sys
sys.path.insert(0, "src")
from apex.schemas.contracts import quality_report, validate_laps

ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="kind", required=True)
r = sub.add_parser("race"); r.add_argument("year", type=int); r.add_argument("event"); r.add_argument("session", default="R", nargs="?")
s = sub.add_parser("season"); s.add_argument("years", type=int, nargs="+"); s.add_argument("--max-events", type=int)
c = sub.add_parser("cache"); c.add_argument("years", type=int, nargs="+")
c.add_argument("--sessions", choices=("race-only", "all"), default="all")
c.add_argument("--telemetry", action="store_true")
c.add_argument("--include-testing", action="store_true")
c.add_argument("--cache-dir", default=".fastf1-cache")
ap.add_argument("--out", default="data/processed")
a = ap.parse_args()
from pathlib import Path
Path(a.out).mkdir(parents=True, exist_ok=True)
if a.kind == "cache":
    from apex.ingestion.fastf1_loader import download_cache
    records = download_cache(
        a.years,
        cache_dir=a.cache_dir,
        sessions=a.sessions,
        telemetry=a.telemetry,
        include_testing=a.include_testing,
    )
    failed = sum(record["status"] == "failed" for record in records)
    print(f"cached {len(records) - failed}/{len(records)} sessions; {failed} failed")
elif a.kind == "race":
    from apex.ingestion.fastf1_loader import load_session
    jobs = {f"laps_{a.year}_{a.event}": load_session(a.year, a.event, a.session)}
else:
    from apex.ingestion.fastf1_loader import load_season
    jobs = {f"laps_{y}": load_season(y, max_events=a.max_events) for y in a.years}
for name, df in jobs.items():
    if df.empty:
        print(f"{name}: no data"); continue
    validate_laps(df); df.to_parquet(f"{a.out}/{name}.parquet")
    print(name); print(quality_report(df))
