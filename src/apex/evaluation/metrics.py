from __future__ import annotations
import numpy as np


def mae(y, p): return float(np.mean(np.abs(np.asarray(y) - np.asarray(p))))
def rmse(y, p): return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def interval_coverage(y, lo, hi) -> float:
    y, lo, hi = map(np.asarray, (y, lo, hi))
    return float(np.mean((y >= lo) & (y <= hi)))


def group_mae(df, y_col: str, p_col: str, by: str) -> dict:
    return {k: mae(g[y_col], g[p_col]) for k, g in df.groupby(by)}
