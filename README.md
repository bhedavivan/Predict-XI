# Predict-XI

Soccer match outcome predictor using historical data from the [football-data.org](https://www.football-data.org/) API and a Gaussian Naive Bayes classifier built from scratch.

Predicts **Home Win / Draw / Away Win** for upcoming fixtures across 12 leagues.

## Setup

1. **Get an API token** — register for free at https://www.football-data.org/client/register
2. **Create a `.env` file** in the project directory:
   ```
   API_TOKEN=your_actual_token_here
   ```
   (Copy `.env.example` and replace the placeholder.)
3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## CLI Usage

Train a model and make predictions from the command line.

```bash
# Train on default leagues (PL, SA, BL1, PD, FL1) for season 2023
python main.py --train

# Train on specific leagues
python main.py --train PL SA BL1 --season 2023

# Predict a match (loads saved model.json if it exists, trains only if needed)
python main.py --predict "Manchester City" "Arsenal"

# Force retrain even if model.json exists
python main.py --predict "Manchester City" "Arsenal" --force-retrain

# Interactive prediction mode
python main.py --interactive

# List available league codes
python main.py --list-leagues
```

### How it works
- `--predict` loads the saved `model.json` if one exists. It only trains when there's no saved model or `--force-retrain` is passed.
- Training defaults to season **2023** (a completed season). If any fetch returns 0 finished matches, a clear error is shown telling you to pass `--season` with a completed season.
- If your API token is missing or invalid, you'll get a helpful error message instead of a crash.

## Web UI

A Flask web app is available at **http://localhost:5000**.

```bash
python app.py
```

### Features
- **Homepage:** Pick a league from a dropdown to see upcoming fixtures (fetched live from football-data.org)
- **Click a fixture** (or pick home + away team) to see predicted probabilities for Home Win / Draw / Away Win
- If no trained model exists, a "Train Model" button is shown
- API errors (bad token, no fixtures, network failure) are handled gracefully with user-friendly messages

## Project Structure

| File | Purpose |
|------|---------|
| `main.py` | CLI entry point (train, predict, interactive) |
| `app.py` | Flask web UI |
| `config.py` | API token loading, league codes |
| `api_client.py` | football-data.org API calls |
| `data_processor.py` | Feature engineering (form, goals) |
| `model_trainer.py` | Gaussian Naive Bayes from scratch |
| `model.json` | Saved trained model (generated) |
| `matches_data.json` | Cached processed match data (generated) |
| `.env` | Your API token (not committed) |

## Requirements

- Python 3.8+
- Flask (for web UI)
- No other external dependencies — the ML model is built from scratch using only the Python standard library

## Supported Leagues

| Code | League |
|------|--------|
| PL | Premier League |
| SA | Serie A |
| BL1 | Bundesliga |
| PD | Primera Division |
| FL1 | Ligue 1 |
| DED | Eredivisie |
| PPL | Primeira Liga |
| ELC | Championship |
| BSA | Campeonato Brasileiro Série A |
| CL | UEFA Champions League |
| EC | European Championship |
| WC | FIFA World Cup |