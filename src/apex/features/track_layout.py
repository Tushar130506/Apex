"""Circuit layout features from track centrelines.

The centrelines are the GeoJSON of github.com/bacinger/f1-circuits (MIT,
``data/reference/f1-circuits.geojson``), one line of longitude/latitude
points per circuit, typically every 40-70 m. Each line is resampled to 10 m
and smoothed before its curvature is measured, so corner counts describe
the shape of the lap, not each kerb. There is no elevation, banking or
surface data here.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

GEOJSON = "data/reference/f1-circuits.geojson"
STEP_M = 10.0
SMOOTH_M = 60.0            # moving-average window on the resampled line
CORNER_RADIUS_M = 250.0    # tighter than this counts as cornering
SLOW_RADIUS_M = 60.0       # tighter than this at the apex: a slow corner
STRAIGHT_RADIUS_M = 1000.0
MIN_CORNER_TURN_DEG = 20.0
EARTH_RADIUS_M = 6_371_000.0

# Jolpica circuitId -> f1-circuits id
CIRCUIT_IDS = {
    "albert_park": "au-1953", "bahrain": "bh-2002", "shanghai": "cn-2004", "catalunya": "es-1991",
    "monaco": "mc-1929", "villeneuve": "ca-1978", "ricard": "fr-1969", "red_bull_ring": "at-1969",
    "silverstone": "gb-1948", "hockenheimring": "de-1932", "hungaroring": "hu-1986", "spa": "be-1925",
    "monza": "it-1922", "marina_bay": "sg-2008", "sochi": "ru-2014", "suzuka": "jp-1962",
    "americas": "us-2012", "rodriguez": "mx-1962", "interlagos": "br-1940", "yas_marina": "ae-2009",
    "imola": "it-1953", "nurburgring": "de-1927", "portimao": "pt-2008", "mugello": "it-1914",
    "sepang": "my-1999", "istanbul": "tr-2005", "zandvoort": "nl-1948", "jeddah": "sa-2021",
    "miami": "us-2022", "losail": "qa-2004", "madring": "es-2026", "baku": "az-2016", "vegas": "us-2023",
}
LAYOUT_COLUMNS = [
    "length_km", "corners", "slow_corners", "corners_per_km", "turning_deg_per_km",
    "cornering_share", "straight_share", "longest_straight_m",
]


def _local_xy(coordinates: list) -> np.ndarray:
    points = np.asarray(coordinates, dtype=float)[:, :2]
    lon, lat = np.radians(points[:, 0]), np.radians(points[:, 1])
    x = (lon - lon.mean()) * np.cos(lat.mean()) * EARTH_RADIUS_M
    y = (lat - lat.mean()) * EARTH_RADIUS_M
    xy = np.column_stack([x, y])
    if np.hypot(*(xy[0] - xy[-1])) > 1.0:
        xy = np.vstack([xy, xy[:1]])          # close the lap
    return xy


def _resample(xy: np.ndarray, step: float) -> np.ndarray:
    seg = np.hypot(*np.diff(xy, axis=0).T)
    distance = np.concatenate([[0.0], np.cumsum(seg)])
    grid = np.arange(0.0, distance[-1], step)
    return np.column_stack([np.interp(grid, distance, xy[:, 0]), np.interp(grid, distance, xy[:, 1])])


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Start and end index of each run of True on a closed lap."""
    if mask.all():
        return [(0, len(mask))]
    shift = int(np.argmin(mask))               # start on a False so no run wraps
    rolled = np.roll(mask, -shift).astype(int)
    edges = np.diff(np.concatenate([[0], rolled, [0]]))
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    return [((s + shift) % len(mask), e - s) for s, e in zip(starts, ends)]


def layout_features(coordinates: list, length_m: float | None = None) -> dict:
    """Shape of one lap from its centreline coordinates (longitude, latitude)."""
    xy = _resample(_local_xy(coordinates), STEP_M)
    n = len(xy)
    window = max(int(SMOOTH_M / STEP_M), 1)
    kernel = np.ones(window) / window
    padded = np.vstack([xy[-window:], xy, xy[:window]])
    smooth = np.column_stack([np.convolve(padded[:, k], kernel, mode="same") for k in (0, 1)])[window:-window]
    heading = np.arctan2(*np.diff(np.vstack([smooth, smooth[:1]]), axis=0).T[::-1])
    turn = np.angle(np.exp(1j * np.diff(np.concatenate([heading, heading[:1]]))))   # per step, wrapped
    curvature = np.abs(turn) / STEP_M
    radius = np.where(curvature > 0, 1.0 / np.maximum(curvature, 1e-12), np.inf)
    corners = slow = 0
    for start, size in _runs(radius < CORNER_RADIUS_M):
        idx = (start + np.arange(size)) % n
        if np.degrees(np.abs(turn[idx].sum())) >= MIN_CORNER_TURN_DEG:
            corners += 1
            slow += int(radius[idx].min() < SLOW_RADIUS_M)
    straights = [size for _, size in _runs(radius > STRAIGHT_RADIUS_M)]
    measured = n * STEP_M
    length = float(length_m) if length_m else measured
    km = length / 1000.0
    return {
        "length_km": km,
        "corners": corners,
        "slow_corners": slow,
        "corners_per_km": corners / km,
        "turning_deg_per_km": float(np.degrees(np.abs(turn).sum()) / (measured / 1000.0)),
        "cornering_share": float((radius < CORNER_RADIUS_M).mean()),
        "straight_share": float((radius > STRAIGHT_RADIUS_M).mean()),
        "longest_straight_m": float(max(straights, default=0) * STEP_M * length / measured),
    }


def layout_table(path: str = GEOJSON) -> pd.DataFrame:
    """One row per Jolpica circuit id with :data:`LAYOUT_COLUMNS`."""
    features = json.loads(Path(path).read_text(encoding="utf-8"))["features"]
    by_id = {feature["properties"]["id"]: feature for feature in features}
    rows = []
    for circuit, source in CIRCUIT_IDS.items():
        if source not in by_id:
            continue
        feature = by_id[source]
        rows.append({"circuit": circuit, "name": feature["properties"]["Name"],
                     **layout_features(feature["geometry"]["coordinates"], feature["properties"].get("length"))})
    return pd.DataFrame(rows)


# Race-model candidates (experimental; not in ORDER_MODEL_COLUMNS).
LAYOUT_MODEL_COLUMNS = ["grid_x_twisty", "grid_x_straight", "team_layout_effect"]


def race_circuits(seasons) -> pd.DataFrame:
    """race_id -> Jolpica circuit id from the season schedules (cached)."""
    from apex.ingestion.jolpica import races

    return pd.DataFrame(
        [{"race_id": f"{season}_{race['round']}", "circuit": race["Circuit"]["circuitId"]}
         for season in seasons for race in races(f"{season}")]
    )


def add_layout_features(features: pd.DataFrame, circuits: pd.DataFrame, layouts: pd.DataFrame) -> pd.DataFrame:
    """Add :data:`LAYOUT_MODEL_COLUMNS` to the output of ``build_order_features``.

    ``grid_x_twisty`` and ``grid_x_straight`` multiply ``grid_log`` by the
    standardised corners per km and longest straight, so the model can let
    the grid matter more or less by layout. ``team_layout_effect`` is how
    much better or worse than its usual form the team has finished at twisty
    tracks in earlier races (shrunk towards zero), times this track's
    twistiness. Races at circuits without a layout get NaN.
    """
    from apex.features.form_features import SLOW_HALFLIFE, _heat_slope, _prior_ewm, race_order

    scaled = layouts.set_index("circuit")[["corners_per_km", "longest_straight_m"]]
    scaled = (scaled - scaled.mean()) / scaled.std(ddof=0)
    race = circuits.set_index("race_id")["circuit"]
    frame = features.copy()
    circuit = frame["race_id"].map(race)
    twisty = circuit.map(scaled["corners_per_km"])
    straight = circuit.map(scaled["longest_straight_m"])
    frame["grid_x_twisty"] = frame["grid_log"] * twisty
    frame["grid_x_straight"] = frame["grid_log"] * straight

    finish = pd.to_numeric(frame["finish_position"], errors="coerce")
    teams = (
        frame.assign(_finish=finish, _z=twisty)
        .groupby(["race_id", "team_canonical"], as_index=False)
        .agg(_best=("_finish", "min"), _z=("_z", "first"))
    )
    teams["_order"] = race_order(teams["race_id"])
    teams["_form"] = _prior_ewm(teams, "team_canonical", "_best", SLOW_HALFLIFE)
    teams["_resid"] = teams["_best"] - teams["_form"]
    teams["_slope"] = _heat_slope(teams, "team_canonical", "_resid", "_z")
    slope = frame.merge(teams[["race_id", "team_canonical", "_slope"]], on=["race_id", "team_canonical"],
                        how="left", validate="many_to_one")["_slope"].to_numpy()
    frame["team_layout_effect"] = slope * twisty.to_numpy()
    return frame


__all__ = [
    "CIRCUIT_IDS", "LAYOUT_COLUMNS", "LAYOUT_MODEL_COLUMNS", "add_layout_features",
    "layout_features", "layout_table", "race_circuits",
]
