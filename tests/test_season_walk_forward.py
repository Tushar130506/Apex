import pandas as pd
import pytest

from apex.models.season_walk_forward import expanding_season_predictions


def _fixture() -> pd.DataFrame:
    rows = []
    for season, race in [(2018, 1), (2018, 2), (2025, 1), (2025, 2), (2025, 3)]:
        for position, driver in enumerate(("A", "B", "C"), start=1):
            rows.append(
                {
                    "season": season,
                    "race_id": f"{season}_{race}",
                    "track": f"Track {race}",
                    "driver": driver,
                    "team": {"A": "X", "B": "Y", "C": "Y"}[driver],
                    "grid_position": position,
                    "finish_position": position,
                    "status": "Finished",
                    "points": {1: 25, 2: 18, 3: 15}[position],
                }
            )
    return pd.DataFrame(rows)


def test_expanding_predictions_append_each_race():
    predictions, checkpoints = expanding_season_predictions(
        _fixture(),
        seed_seasons=[2018],
        forecast_seasons=[2025],
        championship_simulations=200,
    )
    assert predictions.groupby("race_id")["prediction_order"].first().tolist() == [1, 2, 3]
    assert predictions.groupby("race_id")["training_races"].first().tolist() == [2, 3, 4]
    assert checkpoints["before_race_id"].unique().tolist() == ["2025_1", "2025_2", "2025_3"]
    assert checkpoints.groupby("before_race_id")["remaining_races"].first().tolist() == [3, 2, 1]
    assert checkpoints.groupby("before_race_id")["win_probability"].sum().tolist() == pytest.approx([1.0] * 3)
    # Before the last round A leads 50-36: second place is enough, so A cannot lose.
    last = checkpoints[checkpoints["before_race_id"].eq("2025_3")].set_index("driver")
    assert last.loc["A", "win_probability"] == pytest.approx(1.0)


def test_checkpoint_ignores_results_of_simulated_races():
    base = _fixture()
    flipped = base.copy()
    final = flipped["race_id"].eq("2025_3")
    flipped.loc[final, "finish_position"] = flipped.loc[final, "finish_position"].map({1: 3, 2: 2, 3: 1})
    kwargs = dict(seed_seasons=[2018], forecast_seasons=[2025], championship_simulations=200)
    _, first = expanding_season_predictions(base, **kwargs)
    _, second = expanding_season_predictions(flipped, **kwargs)
    pd.testing.assert_frame_equal(first, second)
