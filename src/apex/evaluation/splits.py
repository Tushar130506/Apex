"""Leakage-safe, year-wise splitting: whole seasons, never adjacent rows."""
from __future__ import annotations
import numpy as np
import pandas as pd


class LeakageError(AssertionError):
    pass


def assert_no_leakage(df: pd.DataFrame, masks: dict[str, np.ndarray], group: str = "session_id") -> None:
    keys = list(masks)
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            shared = set(df.loc[masks[a], group]) & set(df.loc[masks[b], group])
            if shared:
                raise LeakageError(f"{group} shared between {a} and {b}: {sorted(shared)[:3]}")


def season_split(df: pd.DataFrame, train_years, val_years, test_years) -> dict[str, np.ndarray]:
    """Year-wise split: train / validation (calibration + model selection) / test (touch once).

    Seasons must not overlap and every split must be non-empty. A warning-free ordering
    (train < val < test) is recommended so you never train on the future.
    """
    sets = {"train": set(train_years), "val": set(val_years), "test": set(test_years)}
    names = list(sets)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if sets[a] & sets[b]:
                raise LeakageError(f"seasons shared between {a} and {b}: {sorted(sets[a] & sets[b])}")
    masks = {k: df["season"].isin(v).to_numpy() for k, v in sets.items()}
    empty = [k for k, m in masks.items() if not m.any()]
    if empty:
        raise ValueError(f"no laps for split(s) {empty}; available seasons: {sorted(df['season'].unique())}")
    assert_no_leakage(df, masks, "session_id")
    return masks
