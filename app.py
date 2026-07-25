#!/usr/bin/env python3
"""
Predict-XI Web UI — Flask app served on http://localhost:5000
"""

import sys
import os
import urllib.parse
from datetime import datetime

from flask import Flask, render_template_string, request, redirect

# Ensure we resolve paths relative to this script's directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)

from config import LEAGUE_CODES
from api_client import fetch_upcoming_matches, MissingTokenError
from data_processor import prepare_prediction_features, load_data
from model_trainer import MatchPredictorModel

app = Flask(__name__)


def urlencode_filter(s):
    """Jinja2 filter for URL-encoding strings."""
    return urllib.parse.quote(str(s), safe='')


app.jinja_env.filters['urlencode'] = urlencode_filter


# ─── HTML Templates ───────────────────────────────────────────────

HOME_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Predict-XI</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #0f172a; color: #e2e8f0; min-height: 100vh; }
        .container { max-width: 900px; margin: 0 auto; padding: 2rem 1rem; }
        h1 { font-size: 2rem; margin-bottom: 0.5rem; color: #38bdf8; }
        .subtitle { color: #94a3b8; margin-bottom: 2rem; }
        .card { background: #1e293b; border-radius: 12px; padding: 1.5rem; margin-bottom: 1.5rem; border: 1px solid #334155; }
        .card h2 { font-size: 1.25rem; margin-bottom: 1rem; color: #f1f5f9; }
        label { display: block; margin-bottom: 0.5rem; color: #94a3b8; font-weight: 500; }
        select, button { width: 100%; padding: 0.75rem 1rem; border-radius: 8px; border: 1px solid #475569; background: #0f172a; color: #e2e8f0; font-size: 1rem; margin-bottom: 1rem; }
        select:focus, button:focus { outline: none; border-color: #38bdf8; }
        button { background: #38bdf8; color: #0f172a; font-weight: 600; cursor: pointer; border: none; }
        button:hover { background: #0ea5e9; }
        button:disabled { background: #475569; color: #94a3b8; cursor: not-allowed; }
        .error { background: #7f1d1d; border: 1px solid #dc2626; color: #fca5a5; padding: 1rem; border-radius: 8px; margin-bottom: 1rem; }
        .success { background: #14532d; border: 1px solid #22c55e; color: #86efac; padding: 1rem; border-radius: 8px; margin-bottom: 1rem; }
        .warning { background: #713f12; border: 1px solid #eab308; color: #fde047; padding: 1rem; border-radius: 8px; margin-bottom: 1rem; }
        .train-btn { background: #eab308; color: #0f172a; }
        .train-btn:hover { background: #ca8a04; }
        .model-status { display: flex; align-items: center; gap: 0.5rem; margin-bottom: 1rem; }
        .status-dot { width: 10px; height: 10px; border-radius: 50%; display: inline-block; }
        .status-dot.green { background: #22c55e; }
        .status-dot.red { background: #dc2626; }
    </style>
</head>
<body>
    <div class="container">
        <h1>⚽ Predict-XI</h1>
        <p class="subtitle">Soccer match outcome predictions</p>

        {% if error %}
        <div class="error">{{ error }}</div>
        {% endif %}

        {% if warning %}
        <div class="warning">{{ warning }}</div>
        {% endif %}

        <div class="card">
            <h2>Select a League</h2>
            <form method="GET" action="/fixtures">
                <label for="league">League</label>
                <select name="league" id="league">
                    <option value="">-- Choose a league --</option>
                    {% for code, name in leagues.items() %}
                    <option value="{{ code }}" {% if selected_league == code %}selected{% endif %}>{{ name }} ({{ code }})</option>
                    {% endfor %}
                </select>
                <button type="submit">View Upcoming Fixtures</button>
            </form>
        </div>

        {% if not model_exists %}
        <div class="card">
            <h2>No Trained Model Found</h2>
            <p style="color: #94a3b8; margin-bottom: 1rem;">A trained model is required to make predictions. Train one now using the CLI or click below.</p>
            <form method="POST" action="/train">
                <button type="submit" class="train-btn">Train Model (Season 2023)</button>
            </form>
        </div>
        {% else %}
        <div class="card">
            <div class="model-status">
                <span class="status-dot green"></span>
                <span style="color: #86efac;">Model loaded and ready</span>
            </div>
        </div>
        {% endif %}
    </div>
</body>
</html>"""

FIXTURES_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Fixtures - Predict-XI</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #0f172a; color: #e2e8f0; min-height: 100vh; }
        .container { max-width: 900px; margin: 0 auto; padding: 2rem 1rem; }
        h1 { font-size: 2rem; margin-bottom: 0.5rem; color: #38bdf8; }
        .subtitle { color: #94a3b8; margin-bottom: 2rem; }
        .card { background: #1e293b; border-radius: 12px; padding: 1.5rem; margin-bottom: 1.5rem; border: 1px solid #334155; }
        .card h2 { font-size: 1.25rem; margin-bottom: 1rem; color: #f1f5f9; }
        .fixture-list { list-style: none; }
        .fixture-item { display: flex; justify-content: space-between; align-items: center; padding: 1rem; border-bottom: 1px solid #334155; }
        .fixture-item:last-child { border-bottom: none; }
        .fixture-teams { font-weight: 600; font-size: 1.1rem; }
        .fixture-date { color: #94a3b8; font-size: 0.85rem; }
        .fixture-vs { color: #64748b; margin: 0 0.5rem; }
        .predict-btn { background: #38bdf8; color: #0f172a; border: none; padding: 0.5rem 1rem; border-radius: 6px; font-weight: 600; cursor: pointer; font-size: 0.9rem; text-decoration: none; display: inline-block; }
        .predict-btn:hover { background: #0ea5e9; }
        .back-link { display: inline-block; margin-top: 1rem; color: #38bdf8; text-decoration: none; }
        .back-link:hover { text-decoration: underline; }
        .error { background: #7f1d1d; border: 1px solid #dc2626; color: #fca5a5; padding: 1rem; border-radius: 8px; margin-bottom: 1rem; }
        .no-fixtures { text-align: center; padding: 2rem; color: #94a3b8; }
    </style>
</head>
<body>
    <div class="container">
        <h1>⚽ Upcoming Fixtures</h1>
        <p class="subtitle">{{ league_name }} ({{ league_code }})</p>

        {% if error %}
        <div class="error">{{ error }}</div>
        {% endif %}

        <div class="card">
            <h2>Fixtures</h2>
            {% if fixtures %}
            <ul class="fixture-list">
                {% for match in fixtures %}
                <li class="fixture-item">
                    <div>
                        <span class="fixture-teams">{{ match.home_team }}</span>
                        <span class="fixture-vs">vs</span>
                        <span class="fixture-teams">{{ match.away_team }}</span>
                        <div class="fixture-date">{{ match.date }}</div>
                    </div>
                    <a href="/predict?home={{ match.home_team|urlencode }}&away={{ match.away_team|urlencode }}" class="predict-btn">Predict</a>
                </li>
                {% endfor %}
            </ul>
            {% else %}
            <p class="no-fixtures">No upcoming fixtures found for this league. The season may be over or no matches are scheduled yet.</p>
            {% endif %}
        </div>

        <a href="/" class="back-link">← Back to league selection</a>
    </div>
</body>
</html>"""

PREDICT_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Prediction - Predict-XI</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #0f172a; color: #e2e8f0; min-height: 100vh; }
        .container { max-width: 700px; margin: 0 auto; padding: 2rem 1rem; }
        h1 { font-size: 2rem; margin-bottom: 0.5rem; color: #38bdf8; }
        .card { background: #1e293b; border-radius: 12px; padding: 1.5rem; margin-bottom: 1.5rem; border: 1px solid #334155; }
        .match-header { text-align: center; margin-bottom: 2rem; }
        .match-header .home { font-size: 1.5rem; font-weight: 700; color: #f1f5f9; }
        .match-header .away { font-size: 1.5rem; font-weight: 700; color: #f1f5f9; }
        .match-header .vs { font-size: 1.25rem; color: #64748b; margin: 0 1rem; }
        .prediction-outcome { text-align: center; font-size: 2rem; font-weight: 700; color: #38bdf8; margin-bottom: 1.5rem; }
        .probabilities { display: flex; gap: 1rem; justify-content: center; flex-wrap: wrap; }
        .prob-box { background: #0f172a; border-radius: 10px; padding: 1.25rem 1.5rem; text-align: center; min-width: 120px; border: 2px solid #334155; }
        .prob-box.winner { border-color: #38bdf8; }
        .prob-label { font-size: 0.85rem; color: #94a3b8; margin-bottom: 0.5rem; }
        .prob-value { font-size: 1.75rem; font-weight: 700; color: #f1f5f9; }
        .back-link { display: inline-block; margin-top: 1.5rem; color: #38bdf8; text-decoration: none; }
        .back-link:hover { text-decoration: underline; }
        .error { background: #7f1d1d; border: 1px solid #dc2626; color: #fca5a5; padding: 1rem; border-radius: 8px; margin-bottom: 1rem; }
        .warning { background: #713f12; border: 1px solid #eab308; color: #fde047; padding: 1rem; border-radius: 8px; margin-bottom: 1rem; }
        .no-model { text-align: center; padding: 2rem; }
        .no-model p { color: #94a3b8; margin-bottom: 1rem; }
        .train-btn { background: #eab308; color: #0f172a; border: none; padding: 0.75rem 1.5rem; border-radius: 8px; font-weight: 600; cursor: pointer; font-size: 1rem; }
        .train-btn:hover { background: #ca8a04; }
    </style>
</head>
<body>
    <div class="container">
        <h1>⚽ Match Prediction</h1>

        {% if error %}
        <div class="error">{{ error }}</div>
        {% endif %}

        {% if warning %}
        <div class="warning">{{ warning }}</div>
        {% endif %}

        <div class="card">
            <div class="match-header">
                <span class="home">{{ home_team }}</span>
                <span class="vs">vs</span>
                <span class="away">{{ away_team }}</span>
            </div>

            {% if prediction %}
            <div class="prediction-outcome">{{ prediction.prediction }}</div>
            <div class="probabilities">
                {% for outcome, prob in prediction.probabilities.items() %}
                <div class="prob-box {% if outcome == prediction.prediction %}winner{% endif %}">
                    <div class="prob-label">{{ outcome }}</div>
                    <div class="prob-value">{{ "%.1f"|format(prob * 100) }}%</div>
                </div>
                {% endfor %}
            </div>
            {% elif not model_exists %}
            <div class="no-model">
                <p>No trained model found. Train a model to make predictions.</p>
                <form method="POST" action="/train" style="display: inline;">
                    <input type="hidden" name="redirect" value="/predict?home={{ home_team|urlencode }}&away={{ away_team|urlencode }}">
                    <button type="submit" class="train-btn">Train Model Now</button>
                </form>
            </div>
            {% endif %}
        </div>

        <a href="/" class="back-link">← Back to league selection</a>
    </div>
</body>
</html>"""

TRAINING_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Training - Predict-XI</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #0f172a; color: #e2e8f0; min-height: 100vh; }
        .container { max-width: 700px; margin: 0 auto; padding: 2rem 1rem; }
        h1 { font-size: 2rem; margin-bottom: 0.5rem; color: #38bdf8; }
        .card { background: #1e293b; border-radius: 12px; padding: 1.5rem; margin-bottom: 1.5rem; border: 1px solid #334155; }
        .card h2 { font-size: 1.25rem; margin-bottom: 1rem; color: #f1f5f9; }
        .success { background: #14532d; border: 1px solid #22c55e; color: #86efac; padding: 1rem; border-radius: 8px; margin-bottom: 1rem; }
        .error { background: #7f1d1d; border: 1px solid #dc2626; color: #fca5a5; padding: 1rem; border-radius: 8px; margin-bottom: 1rem; }
        .back-link { display: inline-block; margin-top: 1.5rem; color: #38bdf8; text-decoration: none; }
        .back-link:hover { text-decoration: underline; }
        .metric { display: flex; justify-content: space-between; padding: 0.5rem 0; border-bottom: 1px solid #334155; }
        .metric:last-child { border-bottom: none; }
        .metric-label { color: #94a3b8; }
        .metric-value { font-weight: 600; }
        .btn { display: inline-block; background: #38bdf8; color: #0f172a; padding: 0.75rem 1.5rem; border-radius: 8px; font-weight: 600; text-decoration: none; margin-top: 1rem; }
        .btn:hover { background: #0ea5e9; }
    </style>
</head>
<body>
    <div class="container">
        <h1>⚽ Model Training</h1>

        {% if error %}
        <div class="error">{{ error }}</div>
        {% endif %}

        {% if success %}
        <div class="success">{{ success }}</div>
        <div class="card">
            <h2>Training Results</h2>
            <div class="metric">
                <span class="metric-label">Accuracy</span>
                <span class="metric-value">{{ "%.1f"|format(accuracy * 100) }}%</span>
            </div>
            <div class="metric">
                <span class="metric-label">Test Samples</span>
                <span class="metric-value">{{ test_samples }}</span>
            </div>
            <div class="metric">
                <span class="metric-label">Training Samples</span>
                <span class="metric-value">{{ train_samples }}</span>
            </div>
        </div>
        {% if redirect_url %}
        <a href="{{ redirect_url }}" class="btn">← Back to Prediction</a>
        {% endif %}
        {% endif %}

        <a href="/" class="back-link">← Back to league selection</a>
    </div>
</body>
</html>"""


# ─── Helpers ──────────────────────────────────────────────────────

def get_model():
    """Load the model if it exists, return None otherwise."""
    model = MatchPredictorModel()
    if model.load():
        return model
    return None


def model_exists():
    """Check if a saved model exists (relative to script directory)."""
    return os.path.exists(os.path.join(SCRIPT_DIR, "model.json"))


def build_team_stats():
    """Build team stats from saved match data, using the MOST RECENT form for each team."""
    rows = load_data()
    if not rows:
        return {}
    team_stats = {}
    for row in rows:
        for team_key, form_key, gs_key, gc_key, mp_key in [
            (row["home_team"], "home_form", "home_goals_scored_avg", "home_goals_conceded_avg", "home_matches_played"),
            (row["away_team"], "away_form", "away_goals_scored_avg", "away_goals_conceded_avg", "away_matches_played"),
        ]:
            # Always overwrite with the latest occurrence (rows are sorted by date)
            team_stats[team_key] = {
                "form": row.get(form_key, 0),
                "goals_scored_avg": row.get(gs_key, 0),
                "goals_conceded_avg": row.get(gc_key, 0),
                "matches_played": row.get(mp_key, 0),
            }
    return team_stats


# ─── Routes ───────────────────────────────────────────────────────

@app.route("/")
def home():
    """Homepage with league selection."""
    error = request.args.get("error", "")
    warning = request.args.get("warning", "")
    selected_league = request.args.get("league", "")
    return render_template_string(
        HOME_TEMPLATE,
        leagues=LEAGUE_CODES,
        model_exists=model_exists(),
        error=error,
        warning=warning,
        selected_league=selected_league,
    )


@app.route("/fixtures")
def fixtures():
    """Show upcoming fixtures for a selected league."""
    league_code = request.args.get("league", "").strip()

    if not league_code:
        return redirect("/?error=" + urllib.parse.quote("Please select a league."))

    if league_code not in LEAGUE_CODES:
        return redirect("/?error=" + urllib.parse.quote(f"Unknown league code: {league_code}"))

    try:
        raw_matches = fetch_upcoming_matches(league_code)
    except MissingTokenError as e:
        return redirect("/?error=" + urllib.parse.quote(str(e)))
    except Exception as e:
        return redirect("/?error=" + urllib.parse.quote(f"API error: {e}"))

    fixtures_list = []
    for m in raw_matches:
        home = m.get("homeTeam", {}).get("name", "Unknown")
        away = m.get("awayTeam", {}).get("name", "Unknown")
        date_str = m.get("utcDate", "")
        try:
            dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
            date_str = dt.strftime("%b %d, %Y at %H:%M")
        except (ValueError, AttributeError):
            pass
        fixtures_list.append({
            "home_team": home,
            "away_team": away,
            "date": date_str,
        })

    return render_template_string(
        FIXTURES_TEMPLATE,
        league_code=league_code,
        league_name=LEAGUE_CODES.get(league_code, league_code),
        fixtures=fixtures_list,
        error="",
    )


@app.route("/predict")
def predict():
    """Show prediction for a specific match."""
    home_team = request.args.get("home", "").strip()
    away_team = request.args.get("away", "").strip()

    if not home_team or not away_team:
        return render_template_string(
            PREDICT_TEMPLATE,
            home_team=home_team or "?",
            away_team=away_team or "?",
            prediction=None,
            model_exists=model_exists(),
            error="Please provide both home and away team names.",
            warning="",
        )

    model = get_model()
    if model is None:
        return render_template_string(
            PREDICT_TEMPLATE,
            home_team=home_team,
            away_team=away_team,
            prediction=None,
            model_exists=False,
            error="",
            warning="",
        )

    # Build team stats from saved data (most recent form)
    team_stats = build_team_stats()
    features = prepare_prediction_features(home_team, away_team, team_stats)
    result = model.predict(features)

    return render_template_string(
        PREDICT_TEMPLATE,
        home_team=home_team,
        away_team=away_team,
        prediction=result,
        model_exists=True,
        error="",
        warning="",
    )


@app.route("/train", methods=["POST"])
def train():
    """Train the model from the web UI."""
    from main import train_model, TrainingError

    redirect_url = request.form.get("redirect", "")

    try:
        model, metrics = train_model(["PL", "SA", "BL1", "PD", "FL1"], "2023")
        return render_template_string(
            TRAINING_TEMPLATE,
            success="Model trained successfully!",
            accuracy=metrics.get("accuracy", 0),
            test_samples=metrics.get("test_samples", 0),
            train_samples=metrics.get("train_samples", 0),
            redirect_url=redirect_url,
            error="",
        )
    except MissingTokenError as e:
        return render_template_string(
            TRAINING_TEMPLATE,
            success="",
            accuracy=0,
            test_samples=0,
            train_samples=0,
            redirect_url="",
            error=str(e),
        )
    except TrainingError as e:
        return render_template_string(
            TRAINING_TEMPLATE,
            success="",
            accuracy=0,
            test_samples=0,
            train_samples=0,
            redirect_url="",
            error=str(e),
        )
    except Exception as e:
        return render_template_string(
            TRAINING_TEMPLATE,
            success="",
            accuracy=0,
            test_samples=0,
            train_samples=0,
            redirect_url="",
            error=f"Training failed: {e}",
        )


if __name__ == "__main__":
    print("Starting Predict-XI web UI...")
    print("Open http://localhost:5000 in your browser")
    app.run(host="0.0.0.0", port=5000, debug=True)