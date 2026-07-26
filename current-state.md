# Current State — Predict-XI

**Status:** Version 5 complete — recent-season data, Dixon-Coles features, stacking option,
independent cross-league matchups, leaner dashboard.

## Done (v5)
- **New primary data source**: `football-data.co.uk` (`csv_data_loader.py`) — actively
  maintained, unlike the footballcsv mirror which has no data past 2023-24 (verified 404s on
  every league). Brings in **2024-25 and 2025-26** (both complete seasons in this timeline) for
  every league it covers, plus richer per-match stats (shots, shots-on-target, corners, fouls,
  cards) across its *entire* history, not just the new seasons. Falls back to footballcsv only
  for leagues it doesn't carry (Russia, Poland, Austria, Switzerland, Denmark, Romania, Mexico).
  Dataset grew from 46,803 → **62,131 matches**, 640 → **669 teams**.
- **16 new rolling-average features** from the richer match stats (shots/SOT/corners/cards, for
  and against), same rolling-window pattern as the existing goals features.
- **Dixon-Coles attack/defense ratings** (new `dixon_coles.py`): a rolling, incremental
  approximation of the Dixon & Coles (1997) Poisson goal model, updated match-by-match with the
  same no-lookahead discipline as Elo. Feeds 5 new features (expected goals, Home/Draw/Away
  probabilities from the Poisson scoreline grid with low-score correlation adjustment) — these
  landed in the **top 3 most important features**, right behind `elo_diff`, and are the main
  reason draw recall improved again this round.
- **Stacking meta-learner** (`model_trainer.py`, `model_type="stacking"`): logistic-regression
  meta-learner over the same three calibrated base legs, replacing fixed voting weights with a
  learned combination. Benchmarked head-to-head against the voting ensemble on identical data.
- **Independent per-side matchups** (`templates/predict.html`): Predict page now has a separate
  league + team picker for home and away — any two teams, any two leagues, not constrained to a
  shared league like last round.
- **Leaner dashboard**: removed the confusion-matrix heatmap and top-features panels added last
  round (user didn't want them) and their supporting dead code in `app.py`/`static/style.css`.
- **Caught and fixed a real model-size bug**: the tuning search's winning RF candidate used
  `max_depth=None` (unbounded), which combined with `CalibratedClassifierCV(cv=3)` — storing 3
  full copies of the model — produced a **557MB** `model.joblib`. Traced it to the RF leg
  specifically, capped `max_depth` in `_RF_GRID`, and resized down to 300 trees/depth 10, landing
  at a shippable **64MB** with equivalent macro-F1 and 2x faster training. `_RF_GRID` now carries
  a comment explaining why unbounded depth is off the table.
- Added a `tree_params_override`/`voting_weights_override` path to `MatchPredictorModel.train()`
  — lets you re-fit with an already-known-good config without re-paying for the tuning search
  (used repeatedly this round to iterate on the RF-size fix in ~10-20 min instead of ~35+ min
  each time).

## Model comparison (temporal, purged 5-fold CV, 62,131 matches)
| Model | Macro F1 | Log-loss | Brier | Draw recall |
|---|---|---|---|---|
| **Ensemble (shipped)** | 0.430 | **1.018** | **0.204** | 24.6% |
| Stacking | 0.430 | 1.023 | 0.205 | 24.6% |

Stacking edged ensemble on macro-F1 by 0.002 — within CV noise (~0.005-0.006 std on both) — but
ensemble's log-loss/Brier were consistently better, so it shipped as the default. Full numbers
in `model_comparison.json`.

## Known limitations
- Russia, Poland, Austria, Switzerland, Denmark, Romania, and Mexico still cap out at 2023-24 —
  football-data.co.uk doesn't cover them and footballcsv has nothing more recent.
- Player-level data (squads, injuries, live form) was explicitly scoped **out** this round: free
  API tiers (~100 requests/day) can't backfill historical player data across 62k+ matches — that
  ceiling is a live/current-lookup budget, not a bulk-historical one. Live-prediction-time
  enrichment (a handful of calls per single fixture) remains a viable future idea; retroactively
  training on it does not.
- Draw recall at 24.6% is real, sustained progress across two rounds now (0.7% → 19.1% → 24.6%)
  but remains the hardest class — inherent to the problem, not obviously more fixable by more
  features alone.

## Next ideas
- A calibration/reliability chart, if the dashboard direction changes again (dropped last round
  and not re-added this round per explicit request).
- Blending Dixon-Coles' own outcome probabilities with the ensemble's at the probability level
  (not just as input features) — considered this round, deferred as an extra tuning axis on top
  of an already-large scope.
- Live-prediction-time player/injury enrichment via a rate-limited free API (see limitations).
