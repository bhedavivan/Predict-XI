# Predict-XI

Soccer match-outcome predictor. Rates every team with a running **Elo** and predicts
**Home Win / Draw / Away Win** with a calibrated **sklearn ensemble**
(RandomForest + HistGradientBoosting + LogisticRegression).

Trained on **46,800+ matches** across six seasons and 29 divisions.

---

## Model (v4)

| Metric | Value | Notes |
|---|---|---|
| Accuracy | **46.3%** | temporal, purged time-series CV |
| Baseline | 42.8% | always-predict-home |
| Macro F1 | **0.425** | up from 0.36 — see note below |
| Log-loss | 1.013 | below uniform (1.099) → still calibrated |
| Brier score | 0.202 | multiclass |
| Draw recall | **19.1%** | up from 0.7% before the class-balancing fix |
| Training matches | 46,803 | 29 leagues × up to 6 seasons |
| Teams rated | 640 | by Elo |

> **Why accuracy went down but the model got better:** the previous version scored 48.0%
> accuracy by essentially never predicting a draw (0.7% draw recall) — always guessing
> home/away is a cheap way to inflate accuracy when draws are ~26% of matches. This version
> trains with balanced sample weights and picks hyperparameters/voting-weights by **macro-F1**
> instead of raw accuracy, so draw recall is now 19.1% — a real, usable signal — at the cost of
> a small amount of headline accuracy. Log-loss/Brier are still comfortably below the
> uninformative-uniform baseline, so probabilities stay meaningful.

**RandomForest vs HistGB vs ensemble** — evaluated during hyperparameter tuning (purged CV,
macro-F1, on a recency-capped subsample):

| Model | CV macro-F1 |
|---|---|
| RandomForest (tuned alone) | 0.434 |
| Ensemble (tuned voting weights) | 0.433 |
| LogisticRegression (tuned alone) | 0.425 |
| HistGradientBoosting (tuned alone) | 0.413 |

Plain **RandomForest is essentially tied with the full ensemble** — the extra HistGB/LR legs
buy almost nothing here. The shipped default stays `ensemble` for now (its calibration is more
robust and the gap is within noise), but `--model-type rf` is a legitimate, simpler alternative
worth trying if you want faster training.

**How it works**
- **Elo ratings** — every team starts from the same 1500 anchor (no hand-picked per-league or
  per-club bonuses) and moves purely on results, with a margin-of-victory multiplier (bigger
  wins move the rating further, per the [World Football Elo Ratings](https://www.eloratings.net/about)
  method) on top of the standard home-edge (+65) and K-factor (24) update.
- **Form features** — rolling 5/10/20-match form, EWMA form, win/loss streaks, clean-sheet and
  BTTS rates, strength of schedule, goals scored/conceded (home / away / overall), head-to-head
  history, and rest days — 52 features in total.
- **sklearn ensemble** — soft-voting `RandomForestClassifier` + `HistGradientBoostingClassifier`
  + calibrated `LogisticRegression`, all three legs individually probability-calibrated
  (`CalibratedClassifierCV`, isotonic) and trained with balanced sample weights so the model
  doesn't just learn to ignore draws. Hyperparameters and voting weights are picked by a small
  search scored on **macro-F1** (not accuracy) over purged time-series cross-validation, so the
  search can't "win" by starving the minority draw class.
- Alternative models (`rf`, `histgb`, `nb`, `logreg`) are selectable with `--model-type`; the
  shipped default is whichever scores best on held-out macro-F1 (see the dashboard for the
  current comparison).

> **Note on Elo across leagues:** teams only ever play within their own league in the training
> data, so Elo is calibrated *within* a league, not across leagues — comparing Arsenal to Celtic
> by raw Elo isn't meaningful, and we no longer fake it with manual per-league offsets. Cross-league
> signal instead comes from separate features (schedule strength, league identity) the model
> learns from directly.

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

- **Dashboard** — live model performance stats, a confusion-matrix heatmap, and the top
  predictive features.
- **Predict a matchup** — pick a league first, then choose the two teams inside it (no more
  scrolling a single 600+ team dropdown), and get calibrated Home/Draw/Away probabilities.
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

# Try a different model
python main.py --model-type nb --train eng.1 es.1 de.1 it.1 fr.1

# List available leagues / seasons
python main.py --list-leagues
python main.py --list-seasons
```

Training prints accuracy, lift, log-loss, Brier, a confusion matrix, and per-class precision/recall.

---

## Project structure

| File | Purpose |
|------|---------|
| `main.py` | CLI: train, predict, interactive |
| `app.py` | Flask web UI (routes only — templates live under `templates/`) |
| `templates/*.html` | Jinja2 page templates (dashboard, predict, fixtures, training) |
| `static/style.css` | Shared design system |
| `model_trainer.py` | Production sklearn ensemble (RF + HistGB + calibrated LogReg), plus the original from-scratch Naive Bayes/softmax kept for tests |
| `data_processor.py` | Feature engineering: form, H2H, rest days, **Elo** |
| `csv_data_loader.py` | Loads historical CSVs from footballcsv/cache.footballdata |
| `api_client.py` | football-data.org live-fixtures client |
| `config.py` | API token loading + league codes |
| `model.json` / `model.joblib` | Shipped trained model (metadata + sklearn pipeline) |
| `model_metrics.json` | Metrics shown on the dashboard |
| `team_stats.json` | Latest per-team stats/Elo for predictions |
| `tests/` | Unit tests (`python -m pytest`) |

## Data sources

- **Training:** [footballcsv/cache.footballdata](https://github.com/footballcsv/cache.footballdata) — historical results, no key needed.
- **Live fixtures:** [football-data.org](https://www.football-data.org/) — free API key required.

## Requirements

- Python 3.8+
- Flask (web UI only)
