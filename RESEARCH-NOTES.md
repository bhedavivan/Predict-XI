# Research notes — how to actually improve this model

Produced from a parallel web-research sweep (7 independent agents: feature
engineering, draw prediction, recency weighting, calibration, model
architectures, benchmarks, data sources) plus a synthesis pass. Every
recommendation was checked against this repo's hard constraints (no bookmaker
odds as features, no fuzzy name matching, no hand-picked per-club constants,
free data only).

## The one thing to internalise: accuracy is already at the ceiling

No-odds cross-league prediction has a **practical ceiling of ~52-56% accuracy**
(RPS ~0.19-0.21, log-loss ~0.97). The best result ever published on this exact
setting is CatBoost + pi-ratings at **55.8%**. Raw bookmaker accuracy is ~53-55%.

This model's **57.8% recent-holdout accuracy is already at or above that ceiling.**
Bookmaker odds beat the best rating-only models by only ~0.001-0.004 RPS — so the
no-odds constraint costs almost nothing. **Any generic cross-league result pushing
past ~58-59% should be treated as a leakage red flag, not a win.**

The 46.1% purged-CV number is **not** the model's true capability. It averages
cold early folds where Elo/Dixon-Coles ratings are still warming up. Fixing the
burn-in (below) will lift the *reported* number several points toward the holdout
without changing true capability. Chasing the ~11pt gap as if it were real skill
would mean over-fitting to a measurement bug.

Draws are the single-most-likely outcome in only ~4% of matches (a structural 1X2
limit); the existing **52.4% draw recall is already exceptional** (most published
models get draw F1 ~0.11-0.20) and should be *protected*, not pushed higher. The
win on draws is precision/calibration, not more recall.

**Honest targets: better RPS/log-loss (~0.19-0.21 / ~0.97), better calibration,
better reliability on the cross-league and promoted-team subsets — NOT a higher
headline accuracy.**

## Prioritised plan (by gain-per-effort)

1. **RPS + multiclass log-loss as primary metrics** (low effort, enabling). Accuracy
   is dominated by the home class and dead-ceilinged; ordinal/probabilistic scoring
   rules are how the literature measures this problem. Also compute a de-vigged
   bookmaker-odds RPS on the same holdout as an OFFLINE-ONLY ceiling benchmark
   (football-data.co.uk ships B365/PS columns — never let them into X).
2. **ClubElo cross-league rating feed** (low-medium effort, highest genuine signal).
   Free keyless CSV, point-in-time (From/To date ranges). Fills the model's named
   blind spot — Elo is calibrated *within* a league, but arbitrary cross-league
   matchups are the app's whole purpose — and carries a promoted team's real
   cross-division strength in instead of the flat-1500 cold-start. Exact-match name
   table (mirror MANUAL_CLUB_MAP), `has_clubelo=0` for uncovered clubs.
3. **Rating burn-in / honest temporal CV** (low effort, corrects the reported number).
   Skip scoring the cold early period; pre-warm ratings from pre-2019 CC0 history
   (openfootball/engsoccerdata) so ratings are well-formed by scoring time. Report
   the recent temporal holdout as the headline; flag the cold fold as a warm-up floor.
4. **Regularise the calibrator** (medium effort, RPS/calibration not accuracy). Replace
   the hardcoded `CALIBRATION_SHRINKAGE=0.06` with an ODIR-penalised Dirichlet
   calibration selected by log-loss on an inner split; clip input probs to ~[0.01,0.99]
   to bound the tail extrapolation the code comment already complains about. Removes
   absurd 0.1%/99% displayed probabilities and makes the 0.37 draw threshold fire on
   the right matches.
5. **Re-tune the draw threshold** on a proper objective (macro-F1 / RPS, tracking draw
   *precision*), confirmed out-of-sample. Cheap once RPS exists.
6. **Cheap draw/value features off the existing Dixon-Coles grid** (low effort): total
   expected goals, low-score draw mass P(0,0)+P(1,1)+P(2,2), Shannon entropy of the
   H/D/A distribution, a tightness interaction, plus squad-value log-*ratio* and
   within-league value percentile, and fixture-congestion (matches in last 7/14 days).
7. **pi-ratings** (medium, uncertain): most-evidenced feature family, but overlaps the
   existing Elo+DC, so gate adoption on an RPS delta, not feature-importance rank.
8. **Data-driven xG proxy from the shots/SoT already stored** (medium, RPS not accuracy):
   fit `a*SoT + b*(shots-SoT)` conversion inside CV folds, run xG through the existing
   rolling/DC machinery. `has_shots`-gated (ragged coverage).
9. **Ordered-logit ensemble member** (medium, uncertain): the three outcomes are ordered
   on a strength axis; adds closeness-driven draw mass by construction. Gate on RPS.
10. **Optional, each gated on RPS, several likely null**: exponential time-decay sample
    weights (half-life grid {inf,730,540,365,270,180}d — the three ratings already bake
    in recency, so classifier-level decay may double-count); LightGBM/CatBoost with
    native categorical league handling (≈noise gain, but smaller model helps the
    file-size constraint); bivariate Poisson upgrade to Dixon-Coles.

## Data sources

- **ClubElo** (api.clubelo.com) — free keyless CSV, cross-competition, point-in-time.
  PRIMARY add. Attribution-only.
- **openfootball / engsoccerdata** (CC0) — pre-2019 results history for rating burn-in
  and priors ONLY (don't inflate the trained match count).
- **football-data.co.uk closing odds** (already ingested) — OFFLINE-ONLY market-ceiling
  benchmark in evaluate.py; never a feature.
- **Understat per-match xG** — OPTIONAL, ~6 top leagues, `has_xg=1` gated, live-capable.
- **FiveThirtyEight SPI `spi_matches.csv`** (CC BY 4.0) — OPTIONAL training-only xG/nsxG
  backfill, frozen at 2023. Ingest ONLY the xg/nsxg columns, never its model outputs.

## Do NOT bother (evidence says skip)

Referee / weather / derby features (no data for arbitrary matchups, or need banned
constants). Deep learning as the core model (loses to GBTs-on-ratings, heavy deps).
SMOTE / synthetic draw oversampling (corrupts calibration + base rates). Focal loss
(sklearn GBTs can't take it; fights calibration). FBref/StatsBomb *scraping* (ToS-banned,
IP-ban risk) — use the Understat snapshot / 538 static file instead. Isotonic/beta as
the shipped final calibrator (not natively 3-class / already the shipped map). Pushing
draw recall higher (already above the literature). 538 SPI's prob/rating columns as
features (another model's outputs, against the own-modelling principle).

## 2026 research sweep — simulator, Transfermarkt scraping, model RPS

A second sweep (4 agents + synthesis) grounded the simulator rebuild, the
second-division scrape, and the next model steps. Sources are inline below.

### Season simulators (Opta / FiveThirtyEight)
- They project the REAL remaining fixtures on top of the ACTUAL current standings
  (points/GD already earned), never a from-scratch season. Opta runs 10,000 sims;
  538 ran ~20,000 daily. (theanalyst.com; fivethirtyeight.com/features/how-our-club-soccer-predictions-work)
- 538 samples two Poisson goal distributions per match to get scorelines, so goal
  difference is available for tie-breaking. Core object is the position-probability
  matrix; title/top-4/relegation % are column-sums of it.
- Determinism per data-state is the trust anchor: reloading never reshuffles the
  table; freshness comes from new results, not new random draws. (→ we cache a
  seeded snapshot and expose a Re-simulate button for on-demand variance.)

### Transfermarkt scraping (for the second divisions the CC0 set omits)
- robots.txt allows the /startseite market-value paths for a generic UA but blocks
  named AI/bot UAs outright → use a plain Chrome UA, 3–5 s delay, disk cache.
- Verified competition codes: Championship GB2, League One GB3, League Two GB4,
  Segunda ES2, Serie B IT2, 2.Bundesliga L2, Ligue 2 FR2. Club id from
  /verein/<id>/; total squad value is the largest money cell in the row.
- The CC0 dcaribou dataset's seed list is first-tier only (grep 'second_tier' =
  none) — the second divisions genuinely must be self-scraped. (github.com/dcaribou)

### Model RPS — prioritised (see current-state.md "Next ideas")
- No-odds SOTA ~0.19 (CatBoost + pi-ratings 0.1925; time-weighted Dixon-Coles
  ~0.191). pi-ratings beat Elo 0.199 vs 0.204. (arxiv.org/html/2309.14807;
  pena.lt/y pi-ratings; Hubacek/Berrar 10.1007/s10994-018-5704-6)
- Biggest gains: (1) pi-ratings home/away features, (2) blend the Dixon-Coles
  H/D/A vector into the ensemble output, (3) tune the boosting member, (4) prune
  features, (5) regularise the calibrator. Each gated on leak-free RPS.
- RPS caveat (Wheatcroft, arxiv 1908.08980): post-hoc class reassignment can hurt
  proper scores — but our draw threshold only relabels the argmax pick, not the
  scored probability vector, so it does not affect RPS/log-loss.
- Squad market value is a real but secondary signal once strong ratings are present
  (correlation ~0.5–0.6 with strength) — worth having for all clubs, not a big RPS
  lever. (Peeters; Coates; paulrjohnson.net transfermarkt tests)
