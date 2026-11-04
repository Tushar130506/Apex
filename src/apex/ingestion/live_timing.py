"""Provider-neutral live timing contract for race-state updates."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Iterable


@dataclass(frozen=True)
class TimingEvent:
    timestamp: datetime
    race_id: str
    lap: int
    event_type: str
    driver: str | None = None
    compound: str | None = None
    tyre_age: int | None = None
    track_status: str | None = None


@dataclass
class LiveRaceState:
    """Mutable state reconstructed from an ordered event stream."""

    race_id: str
    lap: int = 0
    pit_stops: int = 0
    safety_car_active: bool = False
    driver_compounds: dict[str, str] | None = None
    driver_tyre_age: dict[str, int] | None = None

    def __post_init__(self) -> None:
        self.driver_compounds = self.driver_compounds or {}
        self.driver_tyre_age = self.driver_tyre_age or {}

    def apply(self, event: TimingEvent) -> None:
        if event.race_id != self.race_id or event.lap < self.lap:
            raise ValueError("events must belong to the race and arrive chronologically")
        self.lap = event.lap
        if event.event_type == "pit_stop":
            self.pit_stops += 1
        elif event.event_type == "safety_car_started":
            self.safety_car_active = True
        elif event.event_type == "safety_car_ended":
            self.safety_car_active = False
        if event.driver is not None and event.compound is not None:
            self.driver_compounds[event.driver] = event.compound
        if event.driver is not None and event.tyre_age is not None:
            self.driver_tyre_age[event.driver] = event.tyre_age


def replay_events(events: Iterable[TimingEvent], race_id: str) -> LiveRaceState:
    state = LiveRaceState(race_id)
    for event in events:
        state.apply(event)
    return state
