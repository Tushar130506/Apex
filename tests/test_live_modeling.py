from datetime import datetime

import numpy as np
import pandas as pd

from apex.features.tyre_degradation import CompoundDegradation
from apex.ingestion.live_timing import LiveRaceState, TimingEvent
from apex.models.race_monte_carlo import monte_carlo_race_probabilities
from apex.models.safety_hazard import sample_safety_car_starts


def test_compound_degradation_is_age_sensitive():
    model = CompoundDegradation({"SOFT": (0.1, 0.01), "HARD": (0.03, 0.001)})
    assert model.loss("SOFT", 10) > model.loss("HARD", 10)


def test_monte_carlo_probabilities_sum_to_one():
    data = pd.DataFrame(
        {
            "race_id": ["r", "r", "r"],
            "degradation_s_per_lap": [0.02, 0.05, 0.08],
        }
    )
    probabilities = monte_carlo_race_probabilities(
        data, np.array([0.6, 0.3, 0.1]), simulations=300, seed=2
    )
    assert probabilities.sum() == 1
    assert probabilities[0] > probabilities[2]


def test_safety_hazard_sampling_is_bounded():
    starts = sample_safety_car_starts(
        np.random.default_rng(1), 20, {1: 1.0}, 100
    )
    assert (starts == 1).all()


def test_live_timing_events_update_state():
    state = LiveRaceState("race")
    state.apply(TimingEvent(datetime.now(), "race", 5, "pit_stop", "VER", "HARD", 0))
    state.apply(TimingEvent(datetime.now(), "race", 5, "safety_car_started"))
    assert state.pit_stops == 1
    assert state.safety_car_active
    assert state.driver_compounds["VER"] == "HARD"
