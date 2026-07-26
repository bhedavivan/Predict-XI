# Current State — Predict-XI

**Status:** Version 4 complete — balanced/calibrated ensemble, principled Elo, minimalist UI,
two-step league→team selector.

## Done (v4)
- **Fixed the draw-prediction bug**: v3's ensemble scored 48.0% accuracy by essentially never
  predicting a draw (0.7% recall) — only the LogisticRegression leg was balanced/calibrated.
  Now all three legs (`RandomForestClassifier`, `HistGradientBoostingClassifier`,
  `LogisticRegression`) train with balanced sample weights and are individually calibrated
  (`CalibratedClassifierCV`, isotonic). Draw recall is now **19.1%**.
- **Macro-F1-driven tuning**: hyperparameters (small grids for RF/HistGB/LR) and the
  `VotingClassifier` weights are chosen by a purged-CV search scored on macro-F1, not raw
  accuracy, so the search can't win by starving the draw class again. Results and the winning
  config are recorded in `model_metrics.json["tuning_notes"]`.
- **Rebuilt Elo on principled rules** (`data_processor.py`): removed `CLUB_PRIOR` and
  `LEAGUE_BASE_ELO` — hand-typed bonus Elo for 9 named clubs and 28 leagues. Every team now
  starts from the same 1500 anchor; added margin-of-victory K-factor scaling (the standard
  soccer-Elo improvement, per eloratings.net). Elo remains the single strongest feature
  (`elo_diff` alone is ~17.6% importance) — the old system wasn't wrong to lean on Elo, just
  wrong to fake cross-league comparability with hardcoded numbers.
- **Honest metrics**: `test_samples` was hardcoded to 0 in `get_metrics()` forever; now
  populated from the real holdout. Fixed a double-counting bug in `main.py::save_artifacts`
  that inflated the dashboard's "matches trained" stat by adding test_samples on top of
  train_samples (test_samples is a subset of train_samples, not additional data).
- **Minimalist UI rebuild**: `app.py`'s 700-line inline-HTML templates moved to real
  `templates/*.html` (Jinja2 `render_template`) + `static/style.css`. Dropped radial-gradient
  backgrounds, backdrop blur, glow shadows, gradient hero text, and animated count-up numbers
  in favor of flat panels, one accent color, and static numbers. Added a confusion-matrix
  heatmap to the dashboard.
- **Two-step team selector**: `/predict` now requires picking a league first; the team
  dropdowns start disabled and populate only with that league's teams once chosen. Deep links
  from Fixtures (`/predict?home=X&away=Y`) auto-infer the right league so they still land
  pre-populated.
- **RandomForest vs ensemble**: benchmarked during tuning (purged CV, macro-F1, subsample) —
  plain RandomForest (0.434) is essentially tied with the tuned ensemble (0.433), both ahead of
  LogisticRegression (0.425) and HistGB alone (0.413). Ensemble stays the shipped default for
  now (more robust calibration), but `rf` is a legitimate simpler alternative.
- Found and fixed an environment-specific performance issue: OpenMP/BLAS thread-pool spawn
  overhead made `n_jobs=-1` training ~6x *slower* in this sandbox; pinned to single-threaded
  execution (`model_trainer.py` sets `OMP_NUM_THREADS=1` etc. before importing numpy/sklearn,
  plus `threadpoolctl.threadpool_limits(1)` around fit calls).

## Model numbers (temporal, purged CV, 46,803 matches)
- Accuracy 46.3% · baseline 42.8% · macro-F1 0.425 · log-loss 1.013 · Brier 0.202.
- Draw recall 19.1% (was 0.7%) · Home Win recall 57.2% · Away Win recall 54.2%.

## Known limitations
- 2024-25 season still unavailable for most leagues (404s from footballcsv) — data caps out
  at 2023-24 for those.
- Cross-league Elo comparison is still not meaningful by design (no cross-league match data
  exists to anchor it) — this is now stated honestly instead of faked with hardcoded offsets.
- Draw recall at 19% is real progress but still the hardest class — draws are inherently the
  least predictable outcome in football from pre-match features alone.

## Next ideas
- Dixon-Coles attack/defense strength ratings and expected-goals features (raised in research
  during this pass — a Poisson goal-difference model tends to handle draws better than
  classification alone).
- A calibration/reliability chart on the dashboard (skipped this round to avoid a second
  multi-hour retrain in this slow sandbox — log-loss/Brier already summarize it).
- Player-level data aggregation (lineups, injuries, transfers).
- SHAP-based explainability for individual predictions.
