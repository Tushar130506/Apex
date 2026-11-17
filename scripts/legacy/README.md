# Legacy scripts

These scripts belong to the earlier race-winner classifier
(`apex.models.winner`). The current race model is the Plackett-Luce
finish-order model described in `RACE_MODEL_REPORT.md`. They are kept so the
old baseline can be re-run for comparison; nothing in the current pipeline
calls them.

Run them from the repository root, for example
`python scripts/legacy/train_winner.py --help`.

| Script | What it did | Replaced by |
|---|---|---|
| `train_winner.py` | Trained the winner classifier on a fixed train/validation/test season split. | `scripts/evaluate_race_model.py` |
| `repeated_holdout_winner.py` | Expanding-holdout evaluation of the classifier and a 2026 forecast. | `scripts/evaluate_race_model.py`, `scripts/predict_race.py` |
| `forecast_championship.py` | Championship simulation from a stored file of classifier predictions. | `scripts/forecast_season.py`, `scripts/walk_forward_championship.py` |

Their stored outputs are the `runs/winner_*`, `runs/championship_forecast_*`
and `runs/fair_model_comparison_*` folders. The earlier write-ups are
`WINNER_WALK_FORWARD_2026_MEXICO_REPORT.md`, `WINNER_MODEL_FEATURE_SPEC.md`
and `APEX_RACE_WINNER_CONTEXT.md`.

`scripts/walk_forward_winner.py` is not legacy despite its name: it builds
`input_features.parquet`, the race-level table every current script reads.
