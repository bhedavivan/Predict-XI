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
python -m pytest       # 250+ tests

# v5.0 data/model refresh (offline, manual):
python transfermarkt_scraper.py                 # (re)scrape 2nd-div squad values (optional)
python -c "from transfermarkt_scraper import build_tm_results; build_tm_results(range(2012,2027))"  # scrape new-league RESULTS
python main.py --all-seasons --all-leagues --force-retrain   # retrain on the ~48 top flights
python current_leagues.py                       # refresh current-season membership (Hull→PL etc.)
```

Pages: Dashboard, Predict (league→team per side), Simulate (Monte Carlo
season projection), Fixtures (live upcoming matches).

---

## New in v5.0 (see current-state.md for the full list)

- **`leagues.py`** — the single source of truth: every top-flight league's canonical name, strength
  **rank** (PL #1), country, data source, and simulator rules. Fixes the old "La Liga" vs "Primera
  Division" split; every dropdown is now strength-ordered; `team_display_name` (in `team_aliases.py`)
  makes a club read the same on every page.
- **Global top-flight scope (~48 leagues).** Lower divisions dropped from the app; 21 new top flights
  (Saudi, Ukraine, Croatia, Serbia, Czech, K-League, A-League, Colombia, Chile, Uruguay, Ecuador,
  Paraguay, Peru, Egypt, S.Africa, Morocco, Qatar, UAE, Israel, Hungary, India) scraped from
  Transfermarkt's `gesamtspielplan` (results) via `transfermarkt_scraper.py`, merged through
  `csv_data_loader.load_season_league_data`. ~53k new matches.
- **`pi_ratings.py`** — Constantinou & Fenton venue-split, goal-difference ratings, wired on both
  paths (feature count 90→95). **Dixon-Coles output blend** in `model_trainer` (weight auto-tuned on
  the leak-free holdout — self-selects 0 if it doesn't help).
- **Deeper simulator** (`simulate_season.py`): true DC scoreline sampling → goals-for tie-break,
  per-league tiers (Title/UCL/UEL/Playoff/Relegation), MC confidence intervals, GF/GA/GD + BTTS% +
  Over-2.5%, 20k sims.
- **`player_data.py` adapter**: drop a `data_cache/player_data.json` (availability + play-style per
  club) to feed live squads / play-styles with no code change; the `/predict` live nudge is wired.

---

## Where it stands (v5.0)

| Metric | v4.0 | **v5.0** |
|---|---|---|
| **RPS (holdout, leak-free)** | 0.2112 | **0.2059** |
| RPS (purged CV) | 0.2157 | **0.2088** |
| Log-loss (holdout) | 1.0234 | **1.009** |
| Accuracy (holdout, leak-free) | 48.9% | 48.0% |
| Baseline (always pick home) | 44.1% | 44.1% |
| Scored training matches | 168,154 | 161,501 |
| Features | 90 | **95** (pi-ratings) |
| Leagues | 38 (incl. lower) | **48 top flights** |

RPS improved 2.5% — pi-ratings (`pi_expected_gd`, `pi_diff` are the top two features) + the cleaner
top-flight training set. The DC output blend auto-tuned to 0 on the full holdout (self-protecting; no
regression). RPS is the metric — accuracy is at the no-odds ceiling. Model ~87 MB (under the 90 MB
commit guard; no LFS).
| Teams | 1,110 |
| Draw threshold | 0.29 (macro-F1 tuned) |

**RPS is the metric, not accuracy.** RPS 0.211 sits right at the de-vigged
bookmaker ceiling (0.2044) and in the published no-odds SOTA band (~0.19–0.21):
the model's *probabilities* are about as good as no-odds prediction gets. Top-1
accuracy (~49%) is dead-ceilinged on this 3-way problem — the bookmakers
themselves only manage ~50% on this broad league mix.

**Everything above is genuinely out-of-sample.** v4 fixed a leak the audit
caught: the model used to be scored (and calibrated) on rows it had trained on,
which inflated v8's "57.8% / 52.4% draw recall". Metrics and the calibrator now
come from a separate pipeline trained only on data before the holdout
(`holdout_eval.json`). The honest numbers are lower and correct.

**Draws:** out-of-sample the model can't push P(draw) above ~0.31, so argmax
never picks one. The draw threshold (0.29) is tuned to maximise macro-F1 within
a bounded accuracy budget, restoring draw recall 0 → ~28% (macro-F1 0.36 → 0.43)
for a ~3pt accuracy cost — draws at their true base rate, not the leaky v8 62.6%.

**v4.1 additions:** ClubElo cross-league ratings (`clubelo_data.py`, top-5
feature, fixes the cross-league blind spot; overall RPS flat because the holdout
is same-league-heavy), a "Why this prediction" explainability panel on the
Predict page (`data_processor.explain_prediction`), and a player-availability
adapter (`player_data.py`) that is NOT model-wired — no free source backfills
168k matches, so it's ready for live enrichment once `PLAYER_API_TOKEN` is set.
See `AUDIT-FINDINGS.md` and `RESEARCH-NOTES.md`.

Per-league accuracy ranges ~40% (Scottish lower tiers, Argentina) to ~57%
(Eliteserien, Russia, Portugal). Full table in `evaluation.json` / dashboard.

---

## Architecture

| File | Role |
|---|---|
| `app.py` | Flask UI, routes, league display names |
| `data_processor.py` | Feature engineering — Elo, form, H2H, evenness, squad value, draw-signal; `explain_prediction` |
| `dixon_coles.py` | Rolling attack/defense ratings, Poisson scoreline model |
| `model_trainer.py` | sklearn ensemble, tuning, RPS, **leak-free honest-holdout calibration** |
| `csv_data_loader.py` | Match data: football-data.co.uk (2 feeds) + footballcsv fallback |
| `transfermarkt_data.py` | Squad market values (point-in-time + current) |
| `clubelo_data.py` | ClubElo cross-league ratings (point-in-time index, safe name map) |
| `player_data.py` | Player availability adapter (needs `PLAYER_API_TOKEN`; not model-wired) |
| `player_stats.py` | Rolling squad goals+assists/cards from TM appearances (ablated OFF — `INCLUDE_PLAYER_STATS`) |
| `simulate_season.py` | Monte Carlo season projection: title/top-4/relegation odds (`/simulate` page) |
| `current_leagues.py` | Reassigns team_stats leagues to the CURRENT season via the live API (Hull → PL). Run post-retrain |
| `club_mapping.py` | Our club names → Transfermarkt club IDs |
| `team_aliases.py` | Live-API club names → our club names |
| `evaluate.py` | Per-league breakdown, reliability curve, RPS, draw-threshold analysis |
| `odds_benchmark.py` | Offline de-vigged bookmaker RPS/log-loss reference (never a feature) |

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

**Five separate times**, a feature was populated during training but
degenerate at prediction time. The model kept returning confident-looking
probabilities computed from values it had never trained on. The first four
were found by a human noticing a number that looked wrong, not by a test;
the fifth by comparing live feature-vector means against holdout means.

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
5. **The rest of the venue-split block** — the #4 fix covered `ppf_10` only.
   `form`, `goals_scored/conceded_avg`, `matches_played` and `gd_5` are also
   venue-specific in training but were still captured from the team's most
   recent match of *any* venue, so ~half of live predictions served
   wrong-venue values (live `home_gd_5` mean was **−1.47** vs **+1.23** in
   training — a sign flip). In-range values, so the range guard passed;
   caught by comparing live-vector means against holdout means. Fixed by
   extending the venue-matched capture to the whole block (venue-prefixed
   keys on `team_stats`, flat keys kept for backward compat).

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
  `model.joblib`; it's ~35–45MB now and is committed **directly (no Git LFS)** —
  keep it under GitHub's 100MB hard limit (the retrain prints a warning past 90MB).
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
- Transfermarkt's `domestic_competition_id` is NOT current-season membership —
  it retains recently-relegated clubs (its "GB1" lists ~37 clubs, not 20). Use
  the live football-data.org API for current leagues (`current_leagues.py`); the
  free TM bulk dataset is also first-tier only (no Serie B / Championship values
  — that needs a paid feed or ToS-violating scraping, deliberately not done).
- Squad-value club mapping keys off each team's **most-recent Transfermarkt-
  covered league**, NOT its first-seen one. First-seen tagged promoted/relegated
  clubs with a stale lower division (Monaco was in Ligue 2 in 2012-13, Leeds in
  the Championship until 2020) that has no TM competition, so they never matched
  despite being top-flight — +83 clubs (518→601) once fixed. TM ships first
  divisions only, so lower tiers stay structurally uncoverable. Full per-team
  classification in `SQUAD-COVERAGE.md`.

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
