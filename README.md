# Predict-XI

Soccer match-outcome predictor. Rates every team with a running **Elo** and predicts
**Home Win / Draw / Away Win** with a **multinomial softmax-regression** model —
both built from scratch using only the Python standard library (no scikit-learn, no numpy).

Trained on **18,000+ matches** across five seasons of Europe's top divisions.

---

## Model (v2)

| Metric | Value | Notes |
|---|---|---|
| Accuracy | **52.4%** | temporal hold-out (tested only on matches *after* training) |
| Baseline | 43.5% | always-predict-home |
| **Lift** | **+8.9 pts** | over the baseline |
| Log-loss | **1.048** | below uniform (1.099) → probabilities are calibrated |
| Brier score | 0.617 | multiclass |
| Training matches | 18,256 | 12 leagues × 5 seasons |
| Teams rated | 280 | by Elo |

**How it works**
- **Elo ratings** — every team carries a running strength rating (start 1500, home edge +65,
  K-factor 24), updated after each match. This is the single strongest feature.
- **Form features** — rolling 5-match form, goals scored/conceded (home / away / overall),
  head-to-head history, and rest days.
- **Softmax regression** — a 3-class logistic model trained with mini-batch gradient descent,
  feature standardization, and L2 regularization. Unlike Naive Bayes it doesn't assume features
  are independent, so its probabilities stay well-calibrated.
- Alternative models (`nb`, `ensemble`) are selectable with `--model-type`.

> **Note on Elo across leagues:** teams only ever play within their own league in the training
> data, so Elo is calibrated *within* a league, not across leagues. Compare Arsenal to Chelsea,
> not Arsenal to Celtic.

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
- **Predict a matchup** — type any two teams (autocomplete lists all 280 rated teams with their
  Elo) and get calibrated Home/Draw/Away probabilities with an animated breakdown.
- **Browse live fixtures** — pick a league to pull upcoming fixtures from football-data.org
  (needs an API token) and predict any of them in one click.

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
| `app.py` | Flask web UI |
| `model_trainer.py` | Gaussian Naive Bayes **and** softmax regression, both from scratch |
| `data_processor.py` | Feature engineering: form, H2H, rest days, **Elo** |
| `csv_data_loader.py` | Loads historical CSVs from footballcsv/cache.footballdata |
| `api_client.py` | football-data.org live-fixtures client |
| `config.py` | API token loading + league codes |
| `model.json` | Shipped trained model |
| `model_metrics.json` | Metrics shown on the dashboard |
| `team_stats.json` | Latest per-team stats/Elo for predictions |
| `tests/` | Unit tests (`python -m pytest`) |

## Data sources

- **Training:** [footballcsv/cache.footballdata](https://github.com/footballcsv/cache.footballdata) — historical results, no key needed.
- **Live fixtures:** [football-data.org](https://www.football-data.org/) — free API key required.

## Requirements

- Python 3.8+
- Flask (web UI only)
