import numpy as np, pandas as pd
from apex.ingestion.fastf1_loader import laps_to_contract
from apex.schemas.contracts import validate_laps


def fake_laps():
    n = 6
    t = lambda s: pd.to_timedelta(s, unit="s")
    return pd.DataFrame({
        "Driver": ["VER"] * n, "LapNumber": np.arange(1, n + 1, dtype=float),
        "LapTime": [t(92.1), t(91.8), pd.NaT, t(90.9), t(120.0), t(91.0)],
        "PitInTime": [pd.NaT] * 4 + [t(500)] + [pd.NaT], "PitOutTime": [pd.NaT] * 5 + [t(520)],
        "TrackStatus": ["1", "1", "1", "4", "1", "1"], "IsAccurate": [True, True, False, True, False, True],
        "Compound": ["MEDIUM"] * 5 + ["HARD"], "TyreLife": [1, 2, 3, 4, 5, 1], "Stint": [1] * 5 + [2],
        "LapStartTime": [t(i * 90) for i in range(n)]})


def test_contract_mapping_and_reasons():
    w = pd.DataFrame({"Time": [pd.to_timedelta(0, unit="s")], "TrackTemp": [38.0], "AirTemp": [25.0]})
    df = laps_to_contract(fake_laps(), w, "fastf1_2023_1_R", "Test GP", 50, 2023)
    validate_laps(df)
    assert (df.source == "fastf1_real").all() and (df.season == 2023).all() and (df.fuel_provenance == "estimated").all()
    assert list(df.exclusion_reason.fillna("ok")) == ["ok", "ok", "no_time", "neutralised", "pit_lap", "pit_lap"]
    assert df.track_temp.eq(38.0).all() and df.fuel_kg.is_monotonic_decreasing
