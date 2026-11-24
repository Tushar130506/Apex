"""Fuel effect and tyre degradation measured from official race lap times.

Works from lap times and pit stops alone (the Jolpica API), so it needs no
F1TV account. What that data cannot give:

- **Compounds.** Degradation here is the average over whatever tyres were
  used; it is not split by compound.
- **Fuel load.** The race-progress effect mixes fuel burn with the track
  gaining grip; both make later laps faster. It is reported as one number
  per circuit, ``progress_s_per_lap`` (negative = faster each lap).
- **Track status.** Safety-car and red-flag laps are found from the data:
  a lap where the whole field is slow is dropped for everyone.
- **Weather.** A wet or drying race makes every lap faster as the track
  dries, which no fuel burn explains. Such races are flagged by the spread
  of the field's lap times (``field_spread``) and should be left out of
  circuit averages.

Model, fitted per race on clean laps with one intercept per driver::

    lap_time = driver_pace + progress * lap_no + deg * tyre_age + deg2 * tyre_age^2

Progress and degradation can be told apart because tyre age resets at each
pit stop while the lap number keeps rising.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

NEUTRALISED_RATIO = 1.07   # field median this much slower than usual -> safety car / VSC
OUTLIER_RATIO = 1.05       # a driver's lap this much slower than their own usual -> traffic or incident
MIN_CLEAN_LAPS = 150       # per race, across all drivers
MIN_DRIVERS = 8
MIXED_SPREAD = 1.06        # field_spread above this -> wet or drying race (dry races: 1.015-1.05)


def clean_race_laps(laps: pd.DataFrame, pit_stops: pd.DataFrame) -> pd.DataFrame:
    """Add stint, tyre age and an exclusion reason to one race's laps.

    ``laps``: race_id, driver, lap_no, lap_time_s. ``pit_stops``: driver,
    lap_no (the in-lap). The first stint's tyre age counts from the start,
    because the age of the starting set is not in the data.
    """
    frame = laps.sort_values(["driver", "lap_no"]).reset_index(drop=True).copy()
    stops = pit_stops.groupby("driver")["lap_no"].apply(sorted).to_dict()
    stint = np.zeros(len(frame), dtype=int)
    start = np.ones(len(frame), dtype=int)
    reason = pd.Series(pd.NA, index=frame.index, dtype="object")
    for driver, rows in frame.groupby("driver").groups.items():
        in_laps = stops.get(driver, [])
        lap_no = frame.loc[rows, "lap_no"].to_numpy()
        counts = np.searchsorted(np.array(in_laps), lap_no, side="left")  # stops before this lap
        stint[rows] = counts
        starts = np.array([1] + [lap + 1 for lap in in_laps])
        start[rows] = starts[counts]
        in_lap = np.isin(lap_no, in_laps)
        out_lap = np.isin(lap_no, [lap + 1 for lap in in_laps])
        reason.loc[rows[in_lap]] = "pit_in_lap"
        reason.loc[rows[out_lap & ~in_lap]] = "pit_out_lap"
    frame["stint_id"] = stint
    frame["tyre_age"] = frame["lap_no"] - start + 1
    reason[frame["lap_no"].eq(1) & reason.isna()] = "first_lap"
    reason[frame["lap_time_s"].isna() & reason.isna()] = "no_time"

    field = frame[reason.isna()].groupby("lap_no")["lap_time_s"].median()
    typical = field.median()
    neutralised = field.index[field > NEUTRALISED_RATIO * typical]
    reason[frame["lap_no"].isin(neutralised) & reason.isna()] = "neutralised"
    usual = frame[reason.isna()].groupby("driver")["lap_time_s"].median()
    slow = frame["lap_time_s"] > OUTLIER_RATIO * frame["driver"].map(usual)
    reason[slow & reason.isna()] = "slow_lap"
    frame["exclusion_reason"] = reason
    frame["valid"] = reason.isna()
    return frame


def field_spread(cleaned: pd.DataFrame) -> float:
    """95th over 5th percentile of the field's median clean lap, lap by lap.

    About 1.02-1.05 in a dry race (fuel burn and tyres); a drying track
    pushes it well past that.
    """
    field = cleaned[cleaned["valid"]].groupby("lap_no")["lap_time_s"].median()
    return float(field.quantile(0.95) / field.quantile(0.05))


def _design(clean: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
    drivers = sorted(clean["driver"].unique())
    dummies = (clean["driver"].to_numpy()[:, None] == np.array(drivers)[None, :]).astype(float)
    age = clean["tyre_age"].to_numpy(float)
    x = np.column_stack([dummies, clean["lap_no"].to_numpy(float), age, age**2])
    return x, clean["lap_time_s"].to_numpy(float), drivers


def _solve(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    residual = y - x @ beta
    centre = np.median(residual)
    mad = np.median(np.abs(residual - centre))
    if mad > 0:
        keep = np.abs(residual - centre) <= 3.0 * 1.4826 * mad
        if keep.sum() > x.shape[1] + 5:
            beta, *_ = np.linalg.lstsq(x[keep], y[keep], rcond=None)
    return beta


def fit_race_pace(cleaned: pd.DataFrame, *, bootstrap: int = 200, seed: int = 0) -> dict:
    """Race-progress and degradation estimates for one race.

    Intervals come from resampling drivers, so one driver's odd race cannot
    narrow them. ``deg_at_20_laps_s`` is the time a 20-lap-old tyre has lost
    against a new one, which is easier to read than the two coefficients.
    """
    clean = cleaned[cleaned["valid"]]
    drivers = clean["driver"].unique()
    if len(clean) < MIN_CLEAN_LAPS or len(drivers) < MIN_DRIVERS:
        raise ValueError(f"too few clean laps ({len(clean)}) or drivers ({len(drivers)})")
    x, y, names = _design(clean)
    beta = _solve(x, y)
    k = len(names)
    rng = np.random.default_rng(seed)
    draws = []
    by_driver = {driver: group for driver, group in clean.groupby("driver")}
    for _ in range(bootstrap):
        sample = rng.choice(drivers, size=len(drivers), replace=True)
        parts = [by_driver[driver].assign(driver=f"{driver}#{n}") for n, driver in enumerate(sample)]
        bx, by, bnames = _design(pd.concat(parts, ignore_index=True))
        b = _solve(bx, by)
        draws.append(b[len(bnames):])
    draws = np.array(draws)
    progress, deg, deg2 = beta[k:]
    loss20 = deg * 20 + deg2 * 400
    draw_loss20 = draws[:, 1] * 20 + draws[:, 2] * 400
    pace = pd.Series(beta[:k], index=names)
    spread = field_spread(cleaned)
    return {
        "race_id": str(clean["race_id"].iloc[0]),
        "clean_laps": int(len(clean)),
        "drivers": int(len(drivers)),
        "progress_s_per_lap": float(progress),
        "progress_ci95": [float(np.quantile(draws[:, 0], 0.025)), float(np.quantile(draws[:, 0], 0.975))],
        "deg_s_per_lap": float(deg),
        "deg_quad_s_per_lap2": float(deg2),
        "deg_at_20_laps_s": float(loss20),
        "deg_at_20_laps_ci95": [float(np.quantile(draw_loss20, 0.025)), float(np.quantile(draw_loss20, 0.975))],
        "field_spread": spread,
        "mixed_conditions": bool(spread > MIXED_SPREAD),
        "driver_pace_gap_s": (pace - pace.min()).round(3).to_dict(),
    }


TYPICAL_TYRE_AGE = 12   # laps; team pace is compared at this tyre age


def team_race_pace(cleaned: pd.DataFrame) -> pd.DataFrame:
    """Each team's race pace and tyre wear against the field, for one race.

    Fits driver intercepts, the common progress term and one linear
    tyre-age slope per team. ``pace_gap_pct`` is the team's mean driver pace
    at a 12-lap-old tyre against the field median, in percent of a lap
    (positive = slower). ``deg_rel_s_per_lap`` is the team's slope minus the
    median team's (positive = wears its tyres faster). Needs a ``team`` column.
    """
    clean = cleaned[cleaned["valid"]]
    drivers = sorted(clean["driver"].unique())
    teams = sorted(clean["team"].dropna().unique())
    if len(clean) < MIN_CLEAN_LAPS or len(drivers) < MIN_DRIVERS:
        raise ValueError(f"too few clean laps ({len(clean)}) or drivers ({len(drivers)})")
    driver = clean["driver"].to_numpy()
    team = clean["team"].to_numpy()
    age = clean["tyre_age"].to_numpy(float)
    x = np.column_stack([
        (driver[:, None] == np.array(drivers)[None, :]).astype(float),
        clean["lap_no"].to_numpy(float),
        (team[:, None] == np.array(teams)[None, :]) * age[:, None],
    ])
    beta = _solve(x, clean["lap_time_s"].to_numpy(float))
    k = len(drivers)
    slope = pd.Series(beta[k + 1:], index=teams)
    team_of = clean.drop_duplicates("driver").set_index("driver")["team"]
    pace = pd.Series(beta[:k], index=drivers) + team_of.reindex(drivers).map(slope).fillna(0) * TYPICAL_TYRE_AGE
    team_pace = pace.groupby(team_of.reindex(drivers)).mean()
    typical = float(clean["lap_time_s"].median())
    laps = clean.groupby("team").size()
    return pd.DataFrame({
        "race_id": str(clean["race_id"].iloc[0]),
        "team": teams,
        "pace_gap_pct": ((team_pace - team_pace.median()) / typical * 100).reindex(teams).to_numpy(),
        "deg_rel_s_per_lap": (slope - slope.median()).reindex(teams).to_numpy(),
        "clean_laps": laps.reindex(teams).fillna(0).astype(int).to_numpy(),
    })


def circuit_table(races: pd.DataFrame) -> pd.DataFrame:
    """One row per circuit from dry races: median progress and degradation.

    The degradation coefficients are those of the race with the median
    20-lap loss (the mean of the middle two for an even count), so the
    linear and quadratic terms stay consistent with each other.
    """
    dry = races[~races["mixed_conditions"].astype(bool)]
    rows = []
    for circuit, group in dry.groupby("circuit"):
        ordered = group.sort_values("deg_at_20_laps_s")
        n = len(ordered)
        middle = ordered.iloc[[(n - 1) // 2, n // 2]]
        rows.append({
            "circuit": circuit,
            "track": group["track"].iloc[-1],
            "races": n,
            "progress_s_per_lap": float(group["progress_s_per_lap"].median()),
            "deg_s_per_lap": float(middle["deg_s_per_lap"].mean()),
            "deg_quad_s_per_lap2": float(middle["deg_quad_s_per_lap2"].mean()),
            "deg_at_20_laps_s": float(middle["deg_at_20_laps_s"].mean()),
            "deg_at_20_laps_range": f"{group['deg_at_20_laps_s'].min():.2f} to {group['deg_at_20_laps_s'].max():.2f}",
            "pit_stops_per_race": float(group["pit_stops"].mean()),
        })
    columns = ["circuit", "track", "races", "progress_s_per_lap", "deg_s_per_lap", "deg_quad_s_per_lap2",
               "deg_at_20_laps_s", "deg_at_20_laps_range", "pit_stops_per_race"]
    return pd.DataFrame(rows, columns=columns).sort_values("deg_at_20_laps_s", ascending=False)


__all__ = ["circuit_table", "clean_race_laps", "field_spread", "fit_race_pace", "team_race_pace"]
