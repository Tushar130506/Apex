"""FIA-style race and sprint points derived from classified positions."""
from __future__ import annotations

import math
import pandas as pd


GRAND_PRIX_POINTS = (25, 18, 15, 12, 10, 8, 6, 4, 2, 1)
SPRINT_POINTS = (8, 7, 6, 5, 4, 3, 2, 1)


def _scale_points(points: tuple[int, ...], fraction: float | None) -> list[float]:
    if fraction is None or fraction >= 0.75:
        return [float(value) for value in points]
    if fraction < 0:
        raise ValueError("distance_fraction must be non-negative")
    if fraction < 0.25:
        return [0.0] * len(points)
    if fraction < 0.5:
        multiplier = 0.5
    else:
        multiplier = 0.75
    return [math.floor(value * multiplier) for value in points]


def points_for_position(
    position: object,
    *,
    event_type: str = "race",
    classified: bool = True,
    disqualified: bool = False,
    distance_fraction: float | None = None,
) -> float:
    """Return points from position and event conditions.

    The historical shortened-race scale is intentionally configurable and
    documented as approximate. Fastest-lap points are never awarded.
    """
    if not classified or disqualified:
        return 0.0
    numeric = pd.to_numeric(position, errors="coerce")
    if pd.isna(numeric) or numeric < 1:
        return 0.0
    normalized = event_type.strip().lower()
    if normalized in {"sprint", "sprint race"}:
        scale = SPRINT_POINTS
    elif normalized in {"race", "grand prix", "gp"}:
        scale = GRAND_PRIX_POINTS
    else:
        raise ValueError(f"unsupported event_type: {event_type}")
    values = _scale_points(scale, distance_fraction)
    index = int(numeric) - 1
    return values[index] if index < len(values) else 0.0


def derive_points(results: pd.DataFrame) -> pd.DataFrame:
    """Add derived event and total points without trusting source point fields."""
    required = {"finish_position", "classified"}
    missing = required - set(results.columns)
    if missing:
        raise ValueError(f"results missing columns: {sorted(missing)}")
    output = results.copy()
    event_type = output["event_type"] if "event_type" in output else pd.Series("race", index=output.index)
    distance = output["distance_fraction"] if "distance_fraction" in output else pd.Series(pd.NA, index=output.index)
    disqualified = output["disqualified"] if "disqualified" in output else pd.Series(False, index=output.index)
    output["derived_points"] = [
        points_for_position(
            position,
            event_type=str(kind),
            classified=bool(classified),
            disqualified=bool(dq),
            distance_fraction=None if pd.isna(fraction) else float(fraction),
        )
        for position, kind, classified, dq, fraction in zip(
            output["finish_position"], event_type, output["classified"], disqualified, distance
        )
    ]
    return output


def aggregate_championship_points(results: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate derived driver and constructor points, including sprints."""
    required = {"driver", "team", "derived_points"}
    missing = required - set(results.columns)
    if missing:
        raise ValueError(f"results missing columns: {sorted(missing)}")
    drivers = results.groupby("driver", as_index=False)["derived_points"].sum()
    drivers = drivers.rename(columns={"derived_points": "points"}).sort_values("points", ascending=False)
    constructors = results.groupby("team", as_index=False)["derived_points"].sum()
    constructors = constructors.rename(columns={"derived_points": "points"}).sort_values(
        "points", ascending=False
    )
    return drivers.reset_index(drop=True), constructors.reset_index(drop=True)
