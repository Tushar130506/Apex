import pandas as pd

from apex.models.season_walk_forward import expanding_season_predictions


def _fixture() -> pd.DataFrame:
    rows = []
    for season, race in [(2018, 1), (2025, 1), (2025, 2)]:
        for driver, score in [("A", 0.8), ("B", 0.2)]:
            rows.append(
                {
                    "season": season,
                    "race_id": f"{season}_{race}",
                    "driver": driver,
                    "team": "X" if driver == "A" else "Y",
                    "winner": int(driver == "A"),
                    "finish_position": 1 if driver == "A" else 2,
                    "points": 25 if driver == "A" else 18,
                    "feature": score,
                }
            )
    return pd.DataFrame(rows)


def test_expanding_predictions_append_each_race():
    predictions, checkpoints = expanding_season_predictions(
        _fixture(),
        seed_seasons=[2018],
        forecast_seasons=[2025],
        model_columns=["feature"],
        championship_simulations=100,
    )
    assert predictions["prediction_order"].tolist() == [1, 1, 2, 2]
    assert predictions.groupby("race_id")["training_races"].first().tolist() == [1, 2]
    assert checkpoints["after_race_id"].nunique() == 2
