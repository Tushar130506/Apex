"""Fuel/progress and degradation recovered from a simulated race."""
import numpy as np
import pandas as pd
import pytest

from apex.features.race_pace import clean_race_laps, fit_race_pace

PROGRESS, DEG, DEG2 = -0.06, 0.05, 0.001


def _race(seed=0, laps=57, drivers=16):
    rng = np.random.default_rng(seed)
    rows, stops = [], []
    for number in range(drivers):
        driver = f"D{number:02d}"
        base = 92.0 + 0.15 * number
        pits = sorted(rng.choice(np.arange(12, laps - 8), size=1 + number % 2, replace=False).tolist())
        if len(pits) == 2 and pits[1] - pits[0] < 6:
            pits = [pits[0]]
        stops += [{"driver": driver, "lap_no": lap} for lap in pits]
        start = 1
        for lap in range(1, laps + 1):
            age = lap - start + 1
            time = base + PROGRESS * lap + DEG * age + DEG2 * age**2 + rng.normal(0, 0.15)
            if lap == 1:
                time += 6
            if lap in pits:
                time += 20
                start = lap + 1
            if lap - 1 in pits:
                time += 4
            if lap in (30, 31):          # safety car for everyone
                time *= 1.35
            if driver == "D03" and lap == 40:
                time += 8                # traffic
            rows.append({"race_id": "2025_5", "driver": driver, "lap_no": lap, "lap_time_s": time})
    return pd.DataFrame(rows), pd.DataFrame(stops)


def test_cleaning_marks_pit_safety_car_and_slow_laps():
    laps, pits = _race()
    clean = clean_race_laps(laps, pits)
    reasons = clean.set_index(["driver", "lap_no"])["exclusion_reason"]
    first_stop = int(pits.loc[pits.driver.eq("D00"), "lap_no"].iloc[0])
    assert reasons[("D00", first_stop)] == "pit_in_lap" and reasons[("D00", first_stop + 1)] == "pit_out_lap"
    assert reasons[("D05", 1)] == "first_lap" and reasons[("D05", 30)] == "neutralised"
    assert reasons[("D03", 40)] == "slow_lap"
    after = clean[clean.driver.eq("D00") & clean.lap_no.eq(first_stop + 1)].iloc[0]
    assert after.tyre_age == 1 and after.stint_id == 1


def test_fit_recovers_progress_and_degradation():
    laps, pits = _race()
    result = fit_race_pace(clean_race_laps(laps, pits), bootstrap=50)
    assert result["progress_s_per_lap"] == pytest.approx(PROGRESS, abs=0.01)
    assert result["deg_at_20_laps_s"] == pytest.approx(DEG * 20 + DEG2 * 400, abs=0.15)
    low, high = result["progress_ci95"]
    assert low < PROGRESS < high
    assert result["driver_pace_gap_s"]["D00"] == 0 and result["driver_pace_gap_s"]["D15"] == pytest.approx(2.25, abs=0.15)


def test_too_little_data_is_refused():
    laps, pits = _race()
    few = ["D00", "D01", "D02", "D03"]
    laps, pits = laps[laps.driver.isin(few)], pits[pits.driver.isin(few)]
    with pytest.raises(ValueError):
        fit_race_pace(clean_race_laps(laps, pits))
