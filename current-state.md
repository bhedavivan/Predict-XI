# Current State — Predict-XI

**Status:** Version 2 complete — Elo + softmax model, 18k-match training, redesigned web UI.

## Done (v2)
- **Elo ratings** added as features (start 1500, home edge +65, K=24), updated per match.
- **Softmax (multinomial logistic) regression** implemented from scratch — standardized
  features, mini-batch gradient descent, L2. Selected as the default model; beats Naive Bayes
  on accuracy *and* calibration.
  - `--model-type {logreg,nb,ensemble}` to switch models.
- **Honest evaluation:** temporal hold-out (train on the past, test on the future) plus
  log-loss and Brier score. Log-loss dropped from 1.65 (NB) to 1.05 (calibrated).
- **More data:** trained on 18,256 matches — 12 top European divisions × 5 seasons
  (2019-20 → 2023-24) from footballcsv.
- **Fixed league-code bug:** footballcsv uses `es.1` / `sco.1`, not `esp.1` / `sc.1`; La Liga
  was silently missing before.
- **Redesigned Flask UI:** dashboard with live metrics + count-up animation, team-search
  prediction with Elo tiers and animated probability bars, league fixture browser.
- **Shipped artifacts:** `model.json`, `model_metrics.json`, `team_stats.json` committed so the
  app runs out-of-the-box; the large training matrix (`processed_data.json`) is git-ignored.
- All 26 unit tests pass.

## Model numbers (temporal hold-out)
- Accuracy 52.4% · baseline 43.5% · **+8.9 pts lift** · log-loss 1.048 · Brier 0.617.
- Draw recall stays low (~15%) — inherent to 3-way football prediction; draws are genuinely
  the hardest class.

## Known limitations
- **Elo is per-league, not cross-league** (teams only play within their league in the data),
  so cross-league Elo comparisons aren't meaningful. Within-league predictions are fine.
- 2024-25 season isn't in the footballcsv cache repo yet (404s), so training tops out at 2023-24.

## Deferred / next ideas
- Attack/defense (Dixon-Coles style) strength ratings and expected-goals features.
- Between-season Elo regression to the mean (to reflect roster turnover).
- Calibration curve / reliability diagram on the dashboard.
- Prediction history and a per-league standings view in the web UI.
