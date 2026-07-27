# Predict-XI — Handoff

Context for picking this project up in a fresh session. Everything below is
current as of commit `a4800bc`; all work is pushed to
`https://github.com/bhedavivan/Predict-XI` (branch `main`).

---

## What the project is

A football (soccer) match-outcome predictor with a Flask web UI. Given any
two teams — including teams from different leagues that never actually play
each other — it predicts Home Win / Draw / Away Win with calibrated
probabilities.

```bash
python app.py          # web UI on http://localhost:5000
python evaluate.py     # regenerate per-league + calibration analysis
python -m pytest       # 131 tests
```

Pages: Dashboard (metrics, per-league breakdown, calibration), Predict
(pick a league then a team, independently per side), Fixtures (live
upcoming matches from football-data.org).

---

## Where it stands

| Metric | Value |
|---|---|
| **Accuracy** | **57.8%** |
| Macro F1 | 0.562 |
| Draw recall | 52.4% |
| Baseline (always pick home) | ~43% |
| Training matches | 68,228 |
| Features | 82 |
| Leagues | 30 |
| Teams | 707 |

Measured on 6,823 held-out matches that the model was neither trained nor
calibrated on.

**Two numbers exist and they measure different things.** The 57.8% above is
the shipped pipeline (calibrated) on the recent holdout — what the app
actually does. `model_metrics.json` reports **46.1%**, which is purged
cross-validation of the *raw uncalibrated* pipeline averaged over earlier
time folds where Elo/Dixon-Coles ratings are still warming up. Don't quote
them interchangeably.

For context: bookmakers land around 50–55% on this exact 3-way problem, and
this model does not use their odds (see "Decisions" below).

Per-league accuracy ranges roughly 43–67%. Top divisions predict far better
than lower ones — bigger talent gaps mean less randomness. Full table in
`evaluation.json` and on the dashboard.

---

## Architecture

| File | Role |
|---|---|
| `app.py` | Flask UI, routes, league display names |
| `data_processor.py` | Feature engineering — Elo, form, H2H, evenness, squad value |
| `dixon_coles.py` | Rolling attack/defense ratings, Poisson scoreline model |
| `model_trainer.py` | sklearn ensemble, tuning, **calibration layer** |
| `csv_data_loader.py` | Match data: football-data.co.uk (2 feeds) + footballcsv fallback |
| `transfermarkt_data.py` | Squad market values (point-in-time + current) |
| `club_mapping.py` | Our club names → Transfermarkt club IDs |
| `team_aliases.py` | Live-API club names → our club names |
| `evaluate.py` | Per-league breakdown, reliability curve, draw-threshold analysis |

**Model**: soft-voting ensemble (RandomForest + HistGradientBoosting +
calibrated LogisticRegression), then a **post-hoc calibration layer**
(multinomial Platt scaling over log-probabilities), then a draw threshold
of 0.37.

**Data sources**
- `football-data.co.uk` main feed — results + shots/corners/cards, per season
- `football-data.co.uk` "new leagues" feed — one file per country, covers
  the leagues the main feed misses (incl. Brazil, calendar-year seasons)
- `footballcsv` — fallback only; **stops at 2023-24**, do not rely on it
- `transfermarkt-datasets` (CC0) — squad market values, refreshed weekly
- `football-data.org` API — live fixtures only (token in `.env`)

---

## Decisions already made — please don't silently reverse these

**No bookmaker odds as features.** Deliberate user decision. The app's
headline feature is predicting arbitrary matchups (Man City vs Real Madrid),
and odds only exist for real scheduled fixtures — so odds would break the
core feature. The user also wants the predictive work to be their own.
Odds are fine as a *validation benchmark*, never as model input.

**No fuzzy/similarity matching for club names.** Tried and rejected: it
produced confident wrong matches between genuinely different clubs —
`Ath Madrid → Real Madrid`, `Man City → Swansea City`, `U. Cluj → CFR Cluj`,
`Espanyol → Barcelona`. Meanwhile correct pairs scored *low*
(`Wolves → Wolverhampton Wanderers` at 0.43). Similarity is unsafe in both
directions. Matching is league-constrained exact/containment, with a
hand-verified table for the rest, and unmapped-is-better-than-mis-mapped.

**No hand-picked per-club or per-league constants.** An earlier version had
`CLUB_PRIOR` (invented Elo bonuses for named clubs) and `LEAGUE_BASE_ELO`
(28 hand-typed numbers). Both deleted. Per-league home advantage is now
*measured from data* — that's the acceptable version of the same idea.

**Retraining is manual by design**, not automated. User's explicit choice.

---

## ⚠ The recurring bug in this codebase

**Four separate times**, a feature was populated during training but
degenerate at prediction time. The model kept returning confident-looking
probabilities computed from values it had never trained on. Every one was
found by a human noticing a number that looked wrong, not by a test.

1. **Team names** — live API returns `"Manchester United FC"`, training data
   uses `"Man United"`. Unmatched teams silently fell back to neutral
   defaults, so *every* fixture-page prediction was identical.
2. **Head-to-head** — H2H describes a *pair*, but was stored per team, so it
   held whatever that team's last unrelated fixture showed. All 707 teams
   had `h2h_matches=0` at prediction time.
3. **Squad values** — missing for 24 mapped clubs; since the block requires
   both sides known, one missing club disabled it for the whole matchup.
4. **`season_progress` / venue-split form** — `season_progress` is
   match-level, so the per-team `home_`/`away_` prefix lookup could never
   find it (live 0.0 vs training always 1.0). Venue-split form was read from
   a team's most recent match of *any* venue. Caused draw probabilities to
   collapse to ~2%.

**There is now a guard**: `tests/test_data_processor.py::
TestLiveFeaturesMatchTrainingDistribution` compares live feature vectors
against the training range. **If you add a feature, make sure it is
populated on BOTH paths** (`add_form_features` for training,
`prepare_prediction_features` for serving) and that this test still passes.

---

## Other hard-won gotchas

- `clubs.csv` from transfermarkt has a `total_market_value` column that is
  **empty for all 796 rows**. It looks like exactly the shortcut you want.
  Aggregate from `player_valuations` instead.
- `CalibratedClassifierCV(cv=3)` stores **3 full copies** of the estimator.
  Use `ensemble=False`. An unbounded-depth RF once produced a **557MB**
  `model.joblib`; it's ~42MB now and ships via **Git LFS**.
- `n_jobs=-1` is ~11x faster than `n_jobs=1` at 65k rows. (It was set to 1
  after a 600-row smoke test where thread overhead dominated — that
  trade-off reverses completely at real scale.)
- Tuning a threshold and scoring it on the same slice produced a fake
  +3.6pt gain. Always tune and score on disjoint halves.
- Brazil plays a calendar-year season, so the feed labels it `2026` not
  `2025/2026`.
- Brazil has **three unrelated "Atlético" clubs** (Mineiro, Paranaense,
  Goianiense). Never partial-match on "atletico".
- `evaluate.py` must call `model.apply_calibration(...)`, not the raw
  pipeline — it once published numbers no user ever saw.

---

## What's next

In recommended order.

### 1. Re-run the retrain end-to-end
Calibration is currently *bolted on* after training via a separate
`fit_calibration()` call. Folding it into the training pipeline would mean
one command reproduces the shipped model. Also refresh `README.md` and
`current-state.md`, which still quote pre-calibration numbers.

### 2. Explainability on the Predict page
Now that probabilities are trustworthy, show *why*: Elo edge, Dixon-Coles
expected goals, squad-value gap, recent form, H2H record. Turns a bare
percentage into an argument. All the underlying values already exist in the
feature vector — this is presentation, not new modelling.

### 3. Dashboard refresh
`evaluation.json` is current, but verify the dashboard renders the new
per-league and calibration sections correctly, and that the headline stat
tiles show the calibrated numbers rather than the CV ones.

### 4. Squad-value coverage
Currently **48.2%** of training matches have squad values (needs *both*
clubs known; lower divisions have none). 33 clubs are unmapped and listed
explicitly in `club_mapping.KNOWN_ABSENT`. Improving coverage would
strengthen what is already the 5th most important feature.

### 5. Recency weighting
Raised by the user: the model weights a 2019 match the same as a 2025 one.
Sample weights decaying with age is a plausible real gain and was never
tried.

### Things deliberately NOT done
- **Per-match player data** (lineups, injuries) — free tiers cap at ~100
  requests/day; backfilling 68k matches would take years. Needs a paid API
  (~$20–40/mo).
- **Dixon-Coles probability blending** — tested in v6, negative result,
  documented in `dc_blend_analysis.json`. Don't redo it.
- **Brazil Série B** — removed from the Fixtures dropdown; the feed carries
  Série A only, and offering fixtures the model can't answer is worse than
  not listing them.
