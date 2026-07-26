# Predict-XI

Soccer match-outcome predictor. Rates every team with a running **Elo** and a rolling
**Dixon-Coles** attack/defense model, and predicts **Home Win / Draw / Away Win** with a
calibrated **sklearn ensemble** (RandomForest + HistGradientBoosting + LogisticRegression).

Trained on **65,700+ matches** across seven seasons (through 2025-26) and **all 29** divisions —
none still stuck on stale data.

---

## Model (v7)

| Metric | Value | Notes |
|---|---|---|
| Accuracy | **46.0%** | temporal, purged time-series CV |
| Baseline | 43.2% | always-predict-home |
| Macro F1 | **0.431** | balances all three classes, not just accuracy |
| Log-loss | 1.018 | below uniform (1.099) → still calibrated |
| Brier score | 0.204 | multiclass |
| Draw recall | **23.3%** | up from 0.7% pre-class-balancing |
| Training matches | 68,228 | 30 leagues × up to 7 seasons (2019-20 → 2025-26) |
| Teams rated | 707 | by Elo + Dixon-Coles |
| Features | 77 | 50 selected by tuned `SelectKBest` |
| Shipped model size | **26 MB** | (`model.joblib`, via Git LFS) — was 65 MB |

> **On reading the accuracy number.** An earlier version of this project reported 48%, which
> looks better than today's 46.0% but was not a better model: it predicted draws **0.7%** of the
> time, effectively ignoring a third of all possible outcomes to farm the majority classes.
> Today's model predicts draws at 23.3% recall. Compare macro-F1 (which weights all three classes
> equally) rather than raw accuracy when judging progress here.

> **Why not higher accuracy?** This isn't a bug — it's close to the ceiling for this problem.
> Football outcomes depend on things no pre-match feature set captures (a deflected shot, a red
> card, a bad refereeing call), and even bookmakers — with far more inputs than we have — land in
> the 50-55% range on this exact 3-way problem. 45-46% with genuine draw recall is a believable,
> honest number; anything claiming much higher on this task is usually leaking future information.

**Ensemble vs stacking** — trained head-to-head on identical data (purged 5-fold CV, on the
62,131-match dataset of the round they were compared in; ensemble has been carried forward and
re-validated on every dataset since):

| Model | Macro F1 | Log-loss | Brier | Draw recall |
|---|---|---|---|---|
| Ensemble (soft-voting, shipped) | 0.430 | **1.018** | **0.204** | 24.6% |
| Stacking (logistic meta-learner) | **0.430** | 1.023 | 0.205 | 24.6% |

Stacking edged out ensemble on macro-F1 by 0.002 — within the ~0.005-0.006 CV noise both models
showed, i.e. not a real difference. Ensemble's log-loss/Brier were consistently better, so it
ships as the default: the UI displays probabilities directly, and a noise-level classification
edge isn't worth worse-calibrated confidence numbers. `--model-type stacking` is available if
you want to try it yourself.

**Dixon-Coles output blending — tried, didn't help.** Beyond feeding DC's probabilities in as
input features (already in place), blending them with the ensemble's output probabilities at
prediction time was tested directly: macro-F1 on a held-out slice *drops* monotonically as the
blend weight increases (0.458 at 0% DC weight down to 0.397 at 50%), even though log-loss/Brier
improve marginally. The classifier already extracts DC's signal better through the input features
than a naive output blend can. No code change — full grid in `dc_blend_analysis.json`.

**How it works**
- **Elo ratings** — every team starts from the same 1500 anchor (no hand-picked per-league or
  per-club bonuses) and moves purely on results, with a margin-of-victory multiplier (bigger
  wins move the rating further, per the [World Football Elo Ratings](https://www.eloratings.net/about)
  method) on top of the standard home-edge (+65) and K-factor (24) update.
- **Dixon-Coles attack/defense** (`dixon_coles.py`) — a rolling, incremental approximation of the
  [Dixon & Coles (1997)](https://en.wikipedia.org/wiki/Dixon%E2%80%93Coles_model) Poisson goal
  model: each team carries an attack/defense strength nudged by the surprise in each result (same
  idea as Elo, applied to goals), feeding a Poisson scoreline grid with the low-score correlation
  adjustment. Its derived Home/Draw/Away probabilities are themselves top-3 most important
  features — this is specifically what pushed draw recall up this round.
- **Form features** — rolling 5/10/20-match form, EWMA form, win/loss streaks, clean-sheet and
  BTTS rates, strength of schedule, goals scored/conceded, head-to-head history, rest days, plus
  rolling shots/shots-on-target/corners/cards (for leagues football-data.co.uk covers) — 73
  features in total.
- **sklearn ensemble** — soft-voting `RandomForestClassifier` + `HistGradientBoostingClassifier`
  + calibrated `LogisticRegression`, all three legs individually probability-calibrated
  (`CalibratedClassifierCV`, isotonic) and trained with balanced sample weights so the model
  doesn't just learn to ignore draws. Hyperparameters and voting weights are picked by a search
  scored on **macro-F1** (not accuracy) over purged time-series cross-validation, so the search
  can't "win" by starving the minority draw class. A `stacking` model type (logistic-regression
  meta-learner instead of fixed voting weights) is also available — see comparison above.
- Alternative models (`rf`, `histgb`, `stacking`, `nb`, `logreg`) are selectable with
  `--model-type`.

> **Note on Elo across leagues:** teams only ever play within their own league in the training
> data, so Elo is calibrated *within* a league, not across leagues — comparing Arsenal to Celtic
> by raw Elo isn't meaningful, and we no longer fake it with manual per-league offsets. Cross-league
> signal instead comes from separate features (schedule strength, league identity, Dixon-Coles)
> the model learns from directly — which is what makes matching up any two teams from any two
> leagues in the Predict page meaningful rather than just a raw Elo comparison.

---

## Getting a football-data.org API key

The API key is only needed for the **live upcoming-fixtures** feature. Training and predictions
work without it (they use the bundled/cached historical data).

1. Go to **https://www.football-data.org/client/register**
2. Enter your name and email and submit the form — it's **free**.
3. Check your email; football-data.org sends your **API token** (a long string of letters/numbers).
4. Create a file named `.env` in the project folder (copy `.env.example`) and add the line:
   ```
   API_TOKEN=your_actual_token_here
   ```
5. Save. The `.env` file is git-ignored, so your token never gets committed.

The free tier allows 10 requests/minute and covers all the major leagues used here.

---

## Setup

```bash
pip install -r requirements.txt
```

That's it — Flask is the only third-party dependency (for the web UI). The model itself is
pure standard library.

---

## Web UI

```bash
python app.py
```

Open **http://localhost:5000**. The app ships with a pre-trained model, so it works immediately:

- **Dashboard** — live model performance stats.
- **Predict a matchup** — pick a league and team independently on each side (any two teams, any
  two leagues — no shared-league constraint) instead of scrolling a single 600+ team dropdown,
  and get calibrated Home/Draw/Away probabilities.
- **Browse live fixtures** — pick a league to pull upcoming fixtures from football-data.org
  (needs an API token) and predict any of them in one click.

The UI is deliberately plain: flat panels, one accent color, no gradients or blur — templates
live in `templates/*.html` with shared styling in `static/style.css`, not inlined in Python.

## CLI

```bash
# Predict a single match (uses the shipped model)
python main.py --predict "Man City" "Liverpool"

# Interactive prediction mode
python main.py --interactive

# Retrain on the top European leagues, 5 seasons
python main.py --source csv \
  --train eng.1 es.1 de.1 it.1 fr.1 nl.1 pt.1 be.1 tr.1 sco.1 at.1 ch.1 \
  --seasons 2019-20 2020-21 2021-22 2022-23 2023-24

# Retrain on everything, including the current 2025-26 season
python main.py --all-seasons --all-leagues --force-retrain

# Try a different model
python main.py --model-type stacking --train eng.1 es.1 de.1 it.1 fr.1

# List available leagues / seasons
python main.py --list-leagues
python main.py --list-seasons
```

Training prints accuracy, lift, log-loss, Brier, a confusion matrix, and per-class precision/recall.

---

## Keeping the model fresh

The shipped model is a snapshot — training data doesn't update itself. To pull in new results:

```bash
python main.py --all-seasons --all-leagues --force-retrain
```

This is a **manual, on-purpose step**, not automated. European domestic seasons run roughly
August-May, so re-running this once each close season (early summer) captures a full new
season's worth of finished matches. There's no scheduled job pushing model updates on its own —
every retrain is something you run and review before it ships, same as any other change to this
repo. A full retrain takes a while (tens of minutes) since it re-fetches/re-engineers features
across ~65k+ matches; football-data.co.uk responses are cached locally in `data_cache/` for 24h,
so re-running it again soon after (e.g. to try a different `--model-type`) is much faster.

---

## Project structure

| File | Purpose |
|------|---------|
| `main.py` | CLI: train, predict, interactive |
| `app.py` | Flask web UI (routes only — templates live under `templates/`) |
| `templates/*.html` | Jinja2 page templates (dashboard, predict, fixtures, training) |
| `static/style.css` | Shared design system |
| `model_trainer.py` | Production sklearn ensemble (RF + HistGB + calibrated LogReg, plus a stacking option), plus the original from-scratch Naive Bayes/softmax kept for tests |
| `data_processor.py` | Feature engineering: form, H2H, rest days, match stats, **Elo** |
| `dixon_coles.py` | Rolling Dixon-Coles attack/defense ratings |
| `csv_data_loader.py` | Loads historical CSVs from football-data.co.uk (primary) and footballcsv/cache.footballdata (fallback) |
| `api_client.py` | football-data.org live-fixtures client |
| `config.py` | API token loading + league codes |
| `model.json` / `model.joblib` | Shipped trained model (metadata + sklearn pipeline, via Git LFS) |
| `model_metrics.json` | Metrics shown on the dashboard |
| `model_comparison.json` | Ensemble vs stacking head-to-head, from the most recent full retrain |
| `team_stats.json` | Latest per-team stats/Elo/Dixon-Coles ratings for predictions |
| `tests/` | Unit tests (`python -m pytest`) |

## Data sources

- **Training:** [football-data.co.uk](https://www.football-data.co.uk/) — actively maintained,
  through the current season, no key needed. Two feeds: the main one (most major European
  leagues, richer per-match stats — shots/corners/cards) and a second "new leagues" feed (Russia,
  Poland, Austria, Switzerland, Denmark, Romania, Mexico — results only, no extra stats). Falls
  back to [footballcsv/cache.footballdata](https://github.com/footballcsv/cache.footballdata) only
  if a league/season is missing from both. As of this round, every training league is on one of
  the two football-data.co.uk feeds — footballcsv is a safety net, not load-bearing.
- **Live fixtures:** [football-data.org](https://www.football-data.org/) — free API key required.
  Note: the free tier doesn't cover every competition in `LEAGUE_CODES` (confirmed 403 on FL2,
  BL2, PD2, ELC2, ELC3, SA2, PPL2, DED2) — those will error or return empty on the Fixtures page
  regardless of training-data coverage.

## Requirements

- Python 3.8+
- Flask (web UI only)
- scikit-learn, numpy, pandas, joblib, threadpoolctl (see `requirements.txt`)
- [Git LFS](https://git-lfs.com/) to clone `model.joblib` (26 MB)
