#!/usr/bin/env python
"""Write the entry list for a race once qualifying is published.

  python scripts/fetch_weekend.py --race-id 2026_22
  python scripts/fetch_weekend.py --race-id 2026_22 --grid ANT=6 LEC=20

Writes ``data/manual/<race_id>_grid.csv`` (driver, team, grid_position,
qualifying_best_lap_s) for ``predict_race.py --entries``. The grid is the
qualifying order; grid penalties are announced separately, so pass them with
``--grid DRIVER=SLOT`` (the other cars then close up around them).
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, "src")
from apex.ingestion.jolpica import qualifying


def starting_grid(entries: pd.DataFrame, overrides: dict[str, int]) -> pd.Series:
    """Qualifying order with penalised cars moved to their slots."""
    unknown = set(overrides) - set(entries["driver"])
    if unknown:
        raise ValueError(f"not in qualifying: {sorted(unknown)}")
    slots = sorted(overrides.values())
    if len(set(slots)) != len(slots) or not all(1 <= slot <= len(entries) for slot in slots):
        raise ValueError("grid slots must be distinct and within the field")
    order = entries.sort_values("qualifying_position")["driver"]
    rest = iter(driver for driver in order if driver not in overrides)
    taken = {slot: driver for driver, slot in overrides.items()}
    grid = {taken[slot] if slot in taken else next(rest): slot for slot in range(1, len(entries) + 1)}
    return entries["driver"].map(grid)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--race-id", required=True, help="YYYY_round")
    parser.add_argument("--grid", nargs="*", default=[], help="penalties as DRIVER=SLOT, e.g. ANT=6")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    season, round_number = (int(part) for part in args.race_id.split("_"))
    entries = qualifying(season, round_number)
    if entries.empty:
        raise SystemExit(f"no qualifying published yet for {args.race_id}")
    overrides = {item.split("=")[0].upper(): int(item.split("=")[1]) for item in args.grid}
    entries["grid_position"] = starting_grid(entries, overrides)
    out = Path(args.out or f"data/manual/{args.race_id}_grid.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    table = entries.sort_values("grid_position")[["driver", "team", "grid_position", "qualifying_best_lap_s"]]
    table.to_csv(out, index=False)
    print(table.to_string(index=False))
    print(f"\nwrote {out}; grid = qualifying order" + (f" with {len(overrides)} penalty change(s)" if overrides else ""))


if __name__ == "__main__":
    main()
