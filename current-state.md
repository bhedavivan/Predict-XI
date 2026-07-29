# Current State — Predict-XI

**Status:** v5.0 — global top-flight focus (~48 leagues, scraped to reach the world's top ~50),
strength-ranked and consistently named everywhere; pi-ratings + a Dixon-Coles output blend added to
the model; a much deeper Monte Carlo simulator; and a bring-your-own play-style/live-squad adapter.

## API-Football live team-news (built, blocked on the free plan)

`player_sync.py` spends the API-Football budget (100 req/day) only on injuries for
imminent fixtures and writes `data_cache/player_data.json`, which the `/predict`
live nudge already reads (serving never calls the API — budget-safe). **But the
FREE plan is capped at seasons 2022-2024** (verified: "Free plans do not have
access to this season, try from 2022 to 2024"), so it cannot serve the current
season — the live nudge needs a plan with current-season access (or another
current source). The sync + adapter are dormant and switch on the moment that
exists: `python player_sync.py teams` then `python player_sync.py injuries`. A
*historical* injury feature from 2022-2024 was deliberately NOT built — it would
be populated in training but always zero at serving (no current data on free) =
the train/serve skew trap.

## What changed in v5.2

- **Live track record (`track_record.py`, `/track`).** Logs a dated forecast for every upcoming
  fixture BEFORE kickoff, settles it against the real result once played, and scores the running
  accuracy / RPS / log-loss / calibration — the honest, out-of-sample proof the model works in the
  wild (the holdout RPS is a past-data measurement; this accumulates forward). Keyed by the API's
  stable match id (exact settle, no name guessing). Run `python track_record.py log` / `settle`
  periodically; the `/track` page displays it. Seeded with 271 predictions for the 2026-27 season.

## What changed in v5.1

- **Squad values for every league.** The Transfermarkt value scraper now covers the 21 new top-flight
  leagues too (Al-Hilal €195m, Al Ahly €38m…); `apply_squad_values.py` fills them by exact TM-name
  match. Coverage **720 → 982 clubs**.
- **Promoted-team warm-up.** The big-five second divisions are loaded as rating warm-up only (fed
  through the feature loop, excluded from scoring), so a promoted club (Sunderland, Leeds) arrives in
  the top flight with real Elo/Dixon-Coles/pi history instead of a cold 1500 — and PL is a full 20
  again (was 19 when promoted clubs were missing).
- **Confederation-aware simulator tiers.** A Brazilian league shows "Libertadores/Sudamericana", an
  Asian one "AFC CL", not "UCL/UEL" (`leagues.tier_labels`).
- **Difficulty-coloured fixtures.** Each upcoming match shows the model's pick with a green/amber/grey
  chip (one-sided → coin-flip).
- **RPS-aware ensemble tuning.** Model selection factors in RPS (macro-F1 − weighted RPS), not just
  macro-F1. Holdout RPS held at 0.206 (at the no-odds ceiling; warm-up's gain is on promoted teams).
- **Encoding fix.** All JSON reads/writes now specify `encoding="utf-8"` — the accented new-league
  names had broken serving on Windows (cp1252).

## What changed in v5.0

- **Single source of truth for leagues (`leagues.py`).** Every league now has one canonical display
  name (killing the old "La Liga" vs "Primera Division" split between `app.py` and `config.py`), a
  strength **rank** (Premier League #1), a country, a data source, and simulator rules. `config.py`,
  `app.py`, `csv_data_loader.py` all read from it.
- **Top-flight only, strength-ordered.** The app shows/rates top-flight leagues only (the 11 lower
  divisions are dropped from the UI and from scored training — kept only as rating warm-up would be a
  follow-up). Every dropdown is ordered by league strength, not alphabetically. `team_display_name`
  makes a club read identically on Predict / Simulate / Fixtures (e.g. "Manchester United", not
  "Man United" on some pages and "Manchester United FC" on others).
- **Expanded toward the global top 50 by scraping (`transfermarkt_scraper.gesamtspielplan`).** 21 new
  top-flight leagues with no free bulk feed — Saudi, Ukraine, Croatia, Serbia, Czech, South Korea,
  Australia, Colombia, Chile, Uruguay, Ecuador, Paraguay, Peru, Egypt, South Africa, Morocco, Qatar,
  UAE, Israel, Hungary, India — are scraped politely (browser UA, 3–5 s delay, cached HTML + parsed
  results) and merged through the single loader branch in `csv_data_loader.load_season_league_data`.
- **pi-ratings (`pi_ratings.py`).** Constantinou & Fenton venue-split, goal-difference-based ratings —
  the biggest lever in the no-odds literature — wired on both the training and serving paths behind
  `INCLUDE_PI_RATINGS`, with the same `has_pi` both-sides gate + alignment tests as Elo/DC/ClubElo.
  Feature count 90 → 95.
- **Dixon-Coles output blend (`model_trainer`).** The DC H/D/A vector is blended into the ensemble
  output before calibration, with the weight **auto-tuned on the leak-free holdout** — so it self-
  selects 0 (a no-op) if it doesn't improve RPS, and can only help or do nothing.
- **Deeper Monte Carlo simulator.** True Dixon-Coles scoreline sampling (conditional on the calibrated
  outcome) → real goals-for, points→GD→GF tie-break; per-league tiers (Title / UCL / UEL / Playoff /
  Relegation from `leagues.LEAGUE_RULES`); Monte Carlo confidence intervals; projected GF/GA/GD,
  BTTS%, Over-2.5%; 20,000 sims. Kept the position heatmap + Re-simulate button.
- **Bring-your-own play-style / live-squad adapter (`player_data.py`).** A `FilePlayerDataSource`
  reads a user-maintained `data_cache/player_data.json` (availability + play-style per club) with no
  API needed; the live team-news nudge is wired into `/predict` (serving-only, inert without data).
  Play-style becomes a trained feature the moment historical, point-in-time style data is supplied.

_Headline metrics below refresh on the next retrain (`python main.py --all-seasons --all-leagues
--force-retrain`), which trains the expanded top-flight set with pi-ratings + the DC blend._

## Headline numbers (v5.0)

| Metric | v4.0 | **v5.0** | Reference |
|---|---|---|---|
| **RPS (holdout, leak-free OOS)** | 0.2112 | **0.2059** | market (de-vigged Bet365) 0.2044; no-odds SOTA ~0.19–0.21 |
| RPS (purged CV, warm folds) | 0.2157 | **0.2088** | same band |
| Log-loss (holdout) | 1.0234 | **1.009** | market 1.0036 |
| Accuracy (holdout, leak-free) | 48.9% | 48.0% | de-vigged market ~50% on this mix |

161,501 scored matches · **48 top-flight leagues** · 95 features · ~87 MB model (committed directly, no LFS).

**RPS improved 0.2112 → 0.2059 (−2.5%)** — a real, leak-free gain from pi-ratings + the cleaner
top-flight training set, landing squarely in the no-odds SOTA band and within ~0.002 of the de-vigged
market ceiling. `pi_expected_gd` and `pi_diff` are now the **top two features** (above `elo_diff`).

**Read RPS, not accuracy.** Accuracy is dominated by the home class and dead-ceilinged on this 3-way
problem; the de-vigged bookmakers themselves get only ~50% top-1. Accuracy is flat (48%) while RPS
improved — expected, and RPS is the metric that matters. The DC output blend **auto-tuned to weight 0
on the full holdout** (pi-ratings + the DC features already captured that signal) — its self-protecting
design means it can only help or no-op; here it no-oped, and pi-ratings carried the gain.

**Top features (v5.0):** `pi_expected_gd`, `pi_diff`, `elo_diff`, `dc_away_prob`, `dc_home_prob`,
`dc_away_exp_goals`, `dc_entropy`.

## ⚠ The v4.0 honesty correction (why the numbers look lower than v8's 53.2%)

The v8 "53.2% accuracy / 62.6% draw recall" was **optimistic**: the pipeline was refit on all data
and then evaluated on its own tail (in-sample), and the calibrator was fit on those same memorized
probabilities. A 16-dimension audit caught this; the fix evaluates and calibrates via a separate
pipeline trained only on data *before* the holdout. On genuinely out-of-sample matches:

- **Accuracy is ~49%, not 53–58%.** The gap was in-sample optimism plus a much harder league mix
  (v4 spans 38 leagues incl. lower/exotic divisions; v8's 68k was top-heavy). Per-league accuracy
  ranges 40% (Scottish tiers) to 57% (Eliteserien).
- **Draws** are hard without odds — argmax never picks one (P(draw) tops out ~0.31). Rather than
  leave the class dead, the draw threshold is tuned to maximise **macro-F1 within a bounded accuracy
  budget** (0.29): on the honest holdout this restores **draw recall 0 → ~28%** and macro-F1
  **0.36 → 0.43** at a ~3-point accuracy cost — draws predicted at roughly their true base rate. The
  old v8 "62.6% recall" was an in-sample-leak artifact, not real skill; ~28% at ~29% precision is
  the honest no-odds ceiling.

## What changed in v4.0

- **Data 68k → 168k** — seasons back to 2012-13 (long rating warm-up) + 8 calendar-year leagues
  (Norway, Sweden, Finland, Ireland, MLS, Japan, China, Argentina).
- **RPS + multiclass log-loss** are now first-class metrics (trainer, `evaluate.py`, dashboard),
  plus an offline `odds_benchmark.py` de-vigged market ceiling. Accuracy alone was misleading.
- **Leak-free honest holdout** — separate eval pipeline for metrics + calibrator; predictions
  persisted to `holdout_eval.json`.
- **New draw-signal features** (`dc_entropy`, `dc_total_exp_goals`, `tightness`) — `dc_entropy` is
  already the 5th most important feature — plus a regularized (selected) calibrator.
- **Recency weighting** was implemented and then **ablated out**: a leak-free test showed identical
  RPS across half-lives (the Elo/DC/EWMA features already encode recency). It ships off.
- **Audit fixes**: a 6th train/serve-skew bug (season-gap Elo regression missing at serving),
  Flask RCE, an Argentina season-label bug that dropped ~35% of its matches, XSS, nondeterministic
  feature selection, and more — see `AUDIT-FINDINGS.md`.

**Top features:** `dc_away_prob`, `elo_diff`, `dc_home_prob`, `clubelo_expected`, `clubelo_diff`,
`dc_entropy`.

## Latest additions (v4.1)

- **ClubElo cross-league ratings** (`clubelo_data.py`) — free, keyless, point-in-time. Fills the
  app's core blind spot: in-league Elo can't compare Man City to Bayern, but ClubElo can (1971 vs
  2001 on one scale). `clubelo_expected`/`clubelo_diff` are now top-5 features. Overall holdout RPS
  is flat (0.2109 → 0.2108) because that holdout is same-league-dominated — ClubElo's value is on
  the cross-league matchups the holdout barely contains. Safe league-constrained name matching +
  hand-verified overrides; `has_clubelo=0` for non-European/unmapped clubs.
- **Explainability on the Predict page** — a "Why this prediction" panel: a driver table (in-league
  Elo, cross-league ClubElo, Dixon-Coles xG, form, squad value, goal diff) with the favoured side
  highlighted, plus H2H and DC-scoreline chips. Presentation only; the underlying values are exactly
  what the model reads.
- **Draw recall restored** via the macro-F1 threshold (above).
- **Player-data adapter** (`player_data.py`) — pluggable injuries/lineups source + feature helper +
  a conservative live-only adjustment. Deliberately NOT wired into the trained model: no free source
  can backfill 168k historical matches, so it's ready for live enrichment once a paid API key is
  configured (`PLAYER_API_TOKEN`). The one item gated on a spend decision.

## v4.3 additions

- **Season simulator rebuilt to the Opta/FiveThirtyEight pattern** (`simulate_season.py`, `/simulate`).
  Research confirmed the pros project the REAL remaining fixtures on top of the ACTUAL current
  standings, not a from-scratch season. It now does exactly that:
  - **Live projection** — for the leagues the football-data.org API covers and while a season is
    under way, it seeds each simulation from the live table (points + goal difference already earned)
    and plays out only the real remaining fixtures, so the odds move as matches are played. Off-season
    / uncovered leagues fall back to a clearly-labelled hypothetical full season.
  - **Fast** — the ~380 fixture predictions are batched (`predict_proba_batch`) and the priced bundle
    is cached (`data_cache/sim_*.json`, keyed on model version + team_stats mtime), so a reload is
    ~0.3 s instead of ~4 s and doesn't re-hit the rate-limited API.
  - **Correct tie-breaking** — goal difference is sampled from the Dixon-Coles scoreline per fixture,
    so tables rank on points then GD like real ones (was arbitrary jitter).
  - **Richer output** — a full position-probability heatmap, projected-points range (P10–P90),
    current-vs-projected columns, and a **Re-simulate** button that rolls a fresh seed so a user can
    watch the Monte Carlo vary (the default stays deterministic per data-state, the trustworthy
    behaviour the pros use). Fixed an off-season bug that stacked next season's fixtures on last
    season's final table (Arsenal projected 149 pts, > the 114 max).
  - Still an aggregation of the model's calibrated match probabilities — it does not change any
    single-match forecast.
- **Current-season leagues (`current_leagues.py`).** Training data only knows a team's league up
  to the last completed season, so promoted/relegated sides showed in their old division. This
  fetches the real current membership from the live football-data.org API (`/competitions/{code}/
  teams`) and reassigns each covered team's league in `team_stats` — so Hull City shows in the
  Premier League, Cardiff in the Championship, etc. Serving-only; run after each retrain. NB:
  Transfermarkt's `domestic_competition_id` was tried and REJECTED — it retains recently-relegated
  clubs (its "GB1" lists ~37 clubs, not the current 20), so it can't distinguish current membership.
- **Squad values for secondary leagues — now scraped** (`transfermarkt_scraper.py` +
  `apply_squad_values.py`). The free CC0 dataset is first-tier only, so the user authorised scraping
  Transfermarkt's public market-value pages for the divisions it omits. Verified competition codes
  (Championship GB2, League One GB3, League Two GB4, Segunda ES2, Serie B IT2, 2.Bundesliga L2,
  Ligue 2 FR2) are scraped politely (browser UA within robots.txt, 3–5 s delay, on-disk HTML cache,
  429/503 backoff) across the last five seasons; the last completed season's total squad value is
  used (pre-season squads are in flux and understate value). Values are attached with the SAME safe,
  league-constrained matcher as the CC0 data (`club_mapping.match_team_name`, pooled by country so
  clubs that change tier still resolve), never a loose guess. This lifted squad-value coverage from
  **598 → 720 clubs** (+122). Serving-only enrichment, like `current_leagues.py`: rerun the two
  scripts after each retrain. Second-tier values feed the model's existing squad-value feature at
  serve time now; feeding them into TRAINING (point-in-time) is a retrain-gated follow-up.
- **Player-performance stats (`player_stats.py`) — built, ablated, shipped OFF.** Rolling squad
  goals+assists/cards per game from the free Transfermarkt `appearances` feed, point-in-time,
  covering clubs that lack a squad value (e.g. Monaco). Honest result: it did **not** help — a
  leak-free retrain moved RPS 0.2108 → 0.2111 and the features never entered the top-10, because
  goals+assists aggregated to team level is redundant with the scoring/form/Elo/DC signals already
  present, and it inflated the model 49→80 MB. `INCLUDE_PLAYER_STATS=False`; the module stays so a
  future *richer* source (tackles/pace/xG — not free) can flip it on. Same honest-gate discipline
  as recency weighting.

## Data sources
- Match results: football-data.co.uk (main + "new leagues" feeds), footballcsv fallback.
- Squad values: `dcaribou/transfermarkt-datasets` (CC0-1.0, weekly refresh) — a published dataset,
  not scraped. Point-in-time for training, current for live predictions.

## Known limitations
- **Squad coverage: 720/1,110 clubs** (was 518 → 598 via the most-recent-covered-league mapping fix,
  now → 720 by scraping the big-five second divisions + English League One/Two, see v4.3). The
  remaining ~390 gap is: (a) divisions not yet scraped — 3. Liga, Serie C, Segunda B, National League,
  Scottish 2-4, and non-CC0 first tiers CHN1/FIN1/IRL1 (the scraper extends to any of these by adding
  a verified TM competition code to `SECOND_DIVISION_COMPS`); (b) defunct/historical clubs (Bury,
  Macclesfield, Aalen, old Segunda sides) outside the scraped season window; (c) name-mismatch
  residuals in covered exotic leagues, hand-fixable via `MANUAL_CLUB_MAP`. Clubs still without a value
  are flagged `has_squad_value=0`. Full per-team breakdown in `SQUAD-COVERAGE.md`.
- **Draws are barely predictable without odds** — argmax essentially never picks Draw; forcing more
  draws costs accuracy at ~base-rate precision. This is a genuine no-odds ceiling, not a bug.
- **Two accuracy figures** (CV vs holdout) measure different slices; both published, and both are
  now genuinely out-of-sample (the in-sample leak was fixed).

## Next ideas — model RPS (evidence-backed, from the 2026 research sweep)

The no-odds SOTA is ~0.19 RPS (CatBoost + pi-ratings 0.1925; time-weighted Dixon-Coles ~0.191 on
Eredivisie); our leak-free holdout is ~0.211 vs the de-vigged market's 0.204. Prioritised, each
gated on a leak-free RPS check and (for a new feature) wired on BOTH train + serve paths with the
alignment test — the recurring skew bug:

1. **pi-ratings (home + away)** — Constantinou & Fenton. Update on goal margin, separate home/away
   ratings that cross-nudge. Beat Elo head-to-head (0.199 vs 0.204) and are the backbone of every
   SOTA result. **Biggest untapped gain: −0.003 to −0.008 RPS.** (medium effort)
2. **Blend the Dixon-Coles scoreline H/D/A vector into the ensemble output** (prob-space, ~70/30),
   not just as expected-goals features — structurally stabilises draws. −0.002 to −0.005 RPS. (low)
3. **Tune the boosting member** (CatBoost or a properly-tuned HistGBM) via time-series CV. −0.002 to
   −0.005 RPS. (medium)
4. **Prune the ~90 features** to a decorrelated core (a ~6-feature core matched a 205-feature pool in
   the literature). Variance reduction, −0.001 to −0.003. (medium)
5. **Regularise the Dirichlet calibrator (ODIR/L2) and A/B vs OvR isotonic** — do this last, on the
   improved model. −0.001 to −0.003. (low)
6. **Feed the scraped second-division squad values into TRAINING** (point-in-time) so the model
   learns from them, not just serves them. Marginal on RPS (value is secondary once ratings are
   present) but completes the "equal data for all clubs" goal. (medium, retrain-gated)

Note: the draw threshold only relabels the argmax pick, not the scored probability vector, so it does
NOT affect RPS/log-loss — keep it for the draw-recall UX. Full detail + sources in RESEARCH-NOTES.md.
