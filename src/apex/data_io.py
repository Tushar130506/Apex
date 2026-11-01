"""Read the per-season FastF1 lap tables written by scripts/build_dataset.py."""
from __future__ import annotations
from pathlib import Path
import pandas as pd
from apex.schemas.contracts import validate_laps


def load_processed(directory: str | Path = "data/processed", years: list[int] | None = None) -> pd.DataFrame:
    files = sorted(Path(directory).glob("laps_*.parquet"))
    if not files:
        raise FileNotFoundError(f"no laps_*.parquet in {directory}; run scripts/build_dataset.py first")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df = df.drop_duplicates(["session_id", "driver", "lap_no"])
    if years is not None:
        df = df[df["season"].isin(years)]
    return validate_laps(df)
