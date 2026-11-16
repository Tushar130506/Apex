import numpy as np
import pandas as pd
import pytest

from apex.features.form_features import ORDER_MODEL_COLUMNS, build_order_features, canonical_team
from apex.models.championship import countback_ranks, forecast_championship
from apex.models.forecast import add_upcoming_race, append_results, predict_race
from apex.models.plackett_luce import (
    LinearPlackettLuce,
    break_ties,
    build_layout,
    fit_temperature,
    position_probabilities,
    win_probabilities,
)
from apex.models.race_order import describe_race

DRIVERS = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"]


def _table(races: int = 30, seed: int = 0) -> pd.DataFrame:
    """Synthetic seasons where grid position is informative but not decisive."""
    rng = np.random.default_rng(seed)
    skill = np.linspace(1.5, -1.5, len(DRIVERS))
    rows = []
    for number in range(races):
        season, rnd = 2020 + number // 10, number % 10 + 1
        quali = np.argsort(np.argsort(-(skill + rng.gumbel(size=len(DRIVERS)) * 0.5)))
        finish = np.argsort(np.argsort(-(skill + rng.gumbel(size=len(DRIVERS)))))
        for index, driver in enumerate(DRIVERS):
            rows.append(
                {
                    "race_id": f"{season}_{rnd}", "season": season, "track": f"Track {rnd}",
                    "driver": driver, "team": f"T{index // 2}",
                    "grid_position": float(quali[index] + 1),
                    "qualifying_best_lap_s": 80.0 + 0.1 * quali[index],
                    "finish_position": float(finish[index] + 1),
                    "status": "Finished" if finish[index] < 5 else "Engine",
                    "points": 0.0,
                }
            )
    return pd.DataFrame(rows)


def test_features_never_use_the_target_race_result():
    table = _table()
    changed = table.copy()
    last = changed["race_id"].eq("2022_10")
    changed.loc[last, "finish_position"] = changed.loc[last, "finish_position"].to_numpy()[::-1]
    changed.loc[last, "status"] = "Collision"
    first = build_order_features(table)
    second = build_order_features(changed)
    target = first["race_id"].eq("2022_10").to_numpy()
    pd.testing.assert_frame_equal(
        first.loc[target, ORDER_MODEL_COLUMNS], second.loc[target, ORDER_MODEL_COLUMNS]
    )


def test_retirements_are_read_from_status_not_the_classified_flag():
    features = build_order_features(_table().assign(classified=True))
    assert features["driver_dnf_slow"].dropna().between(0, 1).all()
    assert features["driver_dnf_slow"].max() > 0


def test_team_history_follows_renamed_constructors():
    teams = pd.Series(["Toro Rosso", "AlphaTauri", "RB", "Racing Bulls", "None", None])
    mapped = canonical_team(teams)
    assert mapped.iloc[:4].tolist() == ["Racing Bulls"] * 4
    assert mapped.iloc[4:].isna().all()


def test_win_probabilities_are_exact_positive_and_untied():
    race = pd.DataFrame(
        {"race_id": "2026_1", "driver": ["A", "B", "C", "D"], "grid_filled": [2, 1, 3, 4]}
    )
    described = describe_race(race, np.array([0.0, 0.0, -1.0, -1.0]), simulations=4000)
    assert described["race_probability"].sum() == pytest.approx(1.0)
    assert described["race_probability"].gt(0).all()
    assert not described["race_probability"].duplicated().any()
    # Level strengths are separated by the better grid slot, not by row order.
    assert described.set_index("driver")["predicted_position"].to_dict() == {"B": 1, "A": 2, "C": 3, "D": 4}
    matrix = position_probabilities(np.array([1.0, 0.0, -1.0]), simulations=4000)
    assert matrix.sum(axis=1) == pytest.approx(np.ones(3))
    assert matrix.sum(axis=0) == pytest.approx(np.ones(3), abs=0.02)


def test_break_ties_only_reorders_equal_strengths():
    adjusted = break_ties(np.array([1.0, 1.0, 0.5]), np.array([3, 1, 2]))
    assert adjusted[1] > adjusted[0] > adjusted[2]


def test_linear_model_recovers_grid_effect_and_temperature_is_stable():
    features = build_order_features(_table(60))
    layout = build_layout(features["race_id"], features["finish_position"])
    model = LinearPlackettLuce().fit(features[["grid_log"]], layout)
    assert model.coefficient_table().iloc[0]["coefficient"] < 0
    strengths = model.strengths(features[["grid_log"]])
    assert 0.6 < fit_temperature(strengths, layout) < 1.6
    # Three races cannot move the temperature far from 1.
    few = features[features["race_id"].isin(features["race_id"].unique()[:3])]
    short = build_layout(few["race_id"], few["finish_position"])
    assert 0.7 < fit_temperature(10 * model.strengths(few[["grid_log"]]), short) < 2.5
    probabilities = win_probabilities(features["race_id"], strengths)
    assert probabilities.min() > 0


def test_predict_race_before_and_after_qualifying():
    table = _table(40)
    pre = predict_race(add_upcoming_race(table, "2024_1", "Track 1"), "2024_1", calibration_races=5)
    assert pre["forecast_mode"].eq("pre-weekend").all()
    assert pre["finish_position"].isna().all()
    assert sorted(pre["predicted_position"]) == list(range(1, len(DRIVERS) + 1))
    entries = pd.DataFrame(
        {"driver": DRIVERS[::-1], "grid_position": range(1, len(DRIVERS) + 1), "finish_position": 1}
    )
    post = predict_race(
        add_upcoming_race(table, "2024_1", "Track 1", entries), "2024_1", calibration_races=5
    )
    assert post["forecast_mode"].eq("post-qualifying").all()
    assert post["finish_position"].isna().all()  # a result passed in by mistake is discarded
    assert post["race_probability"].sum() == pytest.approx(1.0)
    merged = pre.merge(post, on="driver", suffixes=("_pre", "_post"))
    assert not np.allclose(merged["race_probability_pre"], merged["race_probability_post"])


def test_append_results_validates_classification():
    table = _table(3)
    extra = table[table["race_id"].eq("2020_1")].assign(race_id="2020_4")
    assert append_results(table, extra)["race_id"].nunique() == 4
    with pytest.raises(ValueError, match="unique"):
        append_results(table, extra.assign(finish_position=2.0))


def test_title_ties_are_settled_by_countback():
    # Level on points: B has the win, A has two second places.
    points = np.array([[36.0, 36.0, 10.0]])
    counts = np.zeros((1, 3, 3))
    counts[0, 0, 1] = 2
    counts[0, 1, 0] = 1
    assert countback_ranks(points, counts)[0].tolist() == [1, 0, 2]
    completed = pd.DataFrame(
        [
            {"race_id": "2026_1", "driver": "A", "team": "X", "points": 25, "finish_position": 1, "classified": True},
            {"race_id": "2026_1", "driver": "B", "team": "Y", "points": 18, "finish_position": 2, "classified": True},
            {"race_id": "2026_2", "driver": "B", "team": "Y", "points": 25, "finish_position": 1, "classified": True},
            {"race_id": "2026_2", "driver": "A", "team": "X", "points": 18, "finish_position": 2, "classified": True},
        ]
    )
    future = pd.DataFrame(
        [
            {"race_id": "2026_3", "driver": "A", "team": "X", "strength": 0.0},
            {"race_id": "2026_3", "driver": "B", "team": "Y", "strength": 0.0},
        ]
    )
    forecast = forecast_championship(completed, future, simulations=4000, seed=1)
    assert forecast.driver["win_probability"].sum() == pytest.approx(1.0)
    assert forecast.driver.set_index("driver").loc["A", "win_probability"] == pytest.approx(0.5, abs=0.04)


def test_sprint_rounds_add_sprint_points():
    completed = pd.DataFrame([{"race_id": "2026_1", "driver": "A", "team": "X", "points": 0}])
    future = pd.DataFrame(
        [
            {"race_id": "2026_2", "driver": "A", "team": "X", "strength": 5.0},
            {"race_id": "2026_2", "driver": "B", "team": "Y", "strength": -5.0},
        ]
    )
    plain = forecast_championship(completed, future, simulations=500)
    sprint = forecast_championship(completed, future, simulations=500, sprint_race_ids=["2026_2"])
    gain = sprint.driver.set_index("driver")["expected_final_points"] - plain.driver.set_index("driver")["expected_final_points"]
    assert gain["A"] == pytest.approx(8.0, abs=0.1) and gain["B"] == pytest.approx(7.0, abs=0.1)
    assert sprint.projected_sprints == 1
