#!/usr/bin/env python3
"""
Predict-XI — Flask web UI (multi-page).
Pages: Dashboard (/), Predict (/predict), Fixtures (/fixtures).
Serves on http://localhost:5000
"""

import os
import json
import urllib.parse
from datetime import datetime

from flask import Flask, render_template, request, redirect

# Resolve paths relative to this script so it runs from any working directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)

from config import LEAGUE_CODES
from api_client import fetch_upcoming_matches, MissingTokenError
from data_processor import prepare_prediction_features, load_data, compute_team_stats
from model_trainer import MatchPredictorModel

app = Flask(__name__)
app.jinja_env.filters['urlencode'] = lambda s: urllib.parse.quote(str(s), safe='')


# Human-readable names for every competition code used in training/predictions.
LEAGUE_NAMES = {
    "PL": "Premier League", "PD": "La Liga", "BL1": "Bundesliga", "SA": "Serie A",
    "FL1": "Ligue 1", "DED": "Eredivisie", "PPL": "Primeira Liga",
    "BEL1": "Belgian Pro League", "TUR1": "Super Lig", "GRE1": "Greek Super League",
    "RUS1": "Russian Premier League", "POL1": "Ekstraklasa", "AUT1": "Austrian Bundesliga",
    "SUI1": "Swiss Super League", "DEN1": "Danish Superliga", "ROU1": "Liga I",
    "MEX1": "Liga MX", "SCO1": "Scottish Premiership", "SCO2": "Scottish Championship",
    "SCO3": "Scottish League One", "SCO4": "Scottish League Two",
    "ELC": "Championship", "BL2": "2. Bundesliga", "PD2": "Segunda Division",
    "SA2": "Serie B", "FL2": "Ligue 2",
}

# Sequential single-hue ramp (light -> dark) for the confusion-matrix heatmap.
_SEQ_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
_CM_LABELS = {-1: "Away Win", 0: "Draw", 1: "Home Win"}
_CM_COLOR = {-1: "var(--away)", 0: "var(--draw)", 1: "var(--home)"}


# ─── Helpers ──────────────────────────────────────────────────────────────

def _load_json(name):
    try:
        with open(os.path.join(SCRIPT_DIR, name)) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def load_metrics():
    return _load_json("model_metrics.json") or {}


def build_confusion_view(metrics):
    """Prepare the confusion matrix for the heatmap template: per-cell color
    (sequential ramp keyed off the row max) and row-relative percentage."""
    cm = metrics.get("confusion_matrix")
    if not cm or not any(any(row) for row in cm):
        return None
    classes = [-1, 0, 1][:len(cm)]
    max_val = max(max(row) for row in cm) or 1
    rows = []
    for i, row in enumerate(cm):
        row_total = sum(row) or 1
        cells = []
        for val in row:
            frac = val / max_val
            step = min(int(frac * (len(_SEQ_RAMP) - 1)), len(_SEQ_RAMP) - 1)
            cells.append({
                "value": val,
                "pct": round(val / row_total * 100, 1),
                "bg": _SEQ_RAMP[step],
                "ink": "#0b1220" if step <= 2 else "#f4f8ff",
            })
        rows.append({
            "label": _CM_LABELS.get(classes[i], classes[i]),
            "color": _CM_COLOR.get(classes[i], "var(--muted)"),
            "cells": cells,
        })
    return {"rows": rows, "col_labels": [_CM_LABELS.get(c, c) for c in classes]}


def top_feature_importances(metrics, limit=12):
    """Top (name, importance) pairs, sorted descending, ready for the
    template to iterate directly (Jinja has no built-in `zip` filter)."""
    names = metrics.get("feature_names", [])
    imps = metrics.get("feature_importances") or []
    pairs = sorted(zip(names, imps), key=lambda t: -t[1])
    return pairs[:limit]


def build_team_stats():
    """Team stats for predictions. Prefer committed team_stats.json, then the
    full processed_data.json, then the API-path matches_data.json."""
    stats = _load_json("team_stats.json")
    if stats:
        return stats
    processed = _load_json("processed_data.json")
    if processed and processed.get("team_stats"):
        return processed["team_stats"]
    rows = load_data()
    return compute_team_stats(rows) if rows else {}


def model_exists():
    return os.path.exists(os.path.join(SCRIPT_DIR, "model.json"))


def get_model():
    model = MatchPredictorModel()
    return model if model.load() else None


def team_tier(elo):
    if elo >= 1650:
        return "Elite"
    if elo >= 1560:
        return "Strong"
    if elo >= 1470:
        return "Mid-table"
    return "Underdog"


def league_name(code):
    return LEAGUE_NAMES.get(code, code or "Other")


def team_list(stats):
    """Sorted teams with Elo, tier and league for the chooser."""
    out = []
    for name, s in stats.items():
        if not name or s.get("matches_played", 0) <= 0:
            continue
        elo = s.get("elo", 1500)
        lg = s.get("league", "")
        out.append({"name": name, "elo": round(elo, 1), "tier": team_tier(elo),
                    "league": lg, "league_name": league_name(lg)})
    out.sort(key=lambda t: (-t["elo"], t["name"]))
    return out


def team_info(stats, name):
    s = stats.get(name, {})
    elo = s.get("elo", 1500)
    lg = s.get("league", "")
    return {"elo": elo, "tier": team_tier(elo), "known": bool(s),
            "league": lg, "league_name": league_name(lg)}


def league_options(teams):
    """Distinct (code, name) leagues present, sorted by name."""
    seen = {}
    for t in teams:
        if t["league"]:
            seen[t["league"]] = t["league_name"]
    return sorted(seen.items(), key=lambda kv: kv[1])


# ─── Routes ───────────────────────────────────────────────────────────────

@app.route("/")
def dashboard():
    metrics = load_metrics()
    return render_template(
        "dashboard.html",
        active="dashboard",
        metrics=metrics,
        n_train_leagues=12,
        confusion=build_confusion_view(metrics),
        top_features=top_feature_importances(metrics),
        error=request.args.get("error", ""),
    )


@app.route("/predict")
def predict():
    stats = build_team_stats()
    teams = team_list(stats)
    home_team = request.args.get("home", "").strip()
    away_team = request.args.get("away", "").strip()

    # The league picker is the required first step. If we arrived with a
    # team already chosen (e.g. from the Fixtures "Predict" button) but no
    # explicit league, infer it from whichever team we know about so the
    # two-step flow still lands pre-populated instead of empty.
    sel_league = request.args.get("league", "").strip()
    if not sel_league:
        if home_team and stats.get(home_team):
            sel_league = stats[home_team].get("league", "")
        elif away_team and stats.get(away_team):
            sel_league = stats[away_team].get("league", "")

    common = dict(
        teams=teams,
        teams_json=json.dumps(teams),
        league_opts=league_options(teams),
        sel_home=home_team, sel_away=away_team, sel_league=sel_league,
        model_exists=model_exists(),
        error="", warning="",
    )

    # No matchup chosen yet → just show the chooser
    if not home_team or not away_team:
        return render_template("predict.html", active="predict", prediction=None, **common)

    model = get_model()
    if model is None:
        common["model_exists"] = False
        return render_template("predict.html", active="predict", prediction=None, **common)

    features = prepare_prediction_features(home_team, away_team, stats)
    result = model.predict(features)

    probs = result["probabilities"]
    p_home = probs.get("Home Win", 0.0)
    p_draw = probs.get("Draw", 0.0)
    p_away = probs.get("Away Win", 0.0)
    top_prob = max(p_home, p_draw, p_away)
    conf = "High" if top_prob >= 0.55 else ("Moderate" if top_prob >= 0.42 else "Low")

    h_info = team_info(stats, home_team)
    a_info = team_info(stats, away_team)

    return render_template(
        "predict.html", active="predict", prediction=result,
        home_team=home_team, away_team=away_team,
        p_home=p_home, p_draw=p_draw, p_away=p_away,
        top_prob=top_prob, confidence_label=conf,
        home_info=h_info, away_info=a_info,
        elo_diff=h_info["elo"] - a_info["elo"],
        both_known=h_info["known"] and a_info["known"],
        cross_league=h_info["league"] and a_info["league"] and h_info["league"] != a_info["league"],
        bars=[("Home Win", p_home, "var(--home)"), ("Draw", p_draw, "var(--draw)"),
              ("Away Win", p_away, "var(--away)")],
        **common,
    )


@app.route("/fixtures")
def fixtures():
    league_code = request.args.get("league", "").strip()
    if not league_code:
        return render_template("fixtures.html", active="fixtures", leagues=LEAGUE_CODES,
                                selected="", fixtures=[], league_name="", error="")
    if league_code not in LEAGUE_CODES:
        return render_template("fixtures.html", active="fixtures", leagues=LEAGUE_CODES, selected="",
                                fixtures=[], league_name="",
                                error=f"Unknown league code: {league_code}")
    try:
        raw = fetch_upcoming_matches(league_code)
        error = ""
    except MissingTokenError as e:
        raw, error = [], str(e)
    except Exception as e:
        raw, error = [], f"API error: {e}"

    fixtures_list = []
    for m in raw:
        date_str = m.get("utcDate", "")
        try:
            date_str = datetime.fromisoformat(date_str.replace("Z", "+00:00")).strftime("%b %d, %Y · %H:%M")
        except (ValueError, AttributeError):
            pass
        fixtures_list.append({
            "home_team": m.get("homeTeam", {}).get("name", "Unknown"),
            "away_team": m.get("awayTeam", {}).get("name", "Unknown"),
            "date": date_str,
        })

    return render_template(
        "fixtures.html", active="fixtures", leagues=LEAGUE_CODES, selected=league_code,
        league_name=LEAGUE_CODES.get(league_code, league_code),
        fixtures=fixtures_list, error=error,
    )


@app.route("/train", methods=["POST"])
def train():
    from main import train_model_csv, TrainingError
    top5 = ["eng.1", "es.1", "de.1", "it.1", "fr.1"]
    seasons = ["2021-22", "2022-23", "2023-24"]
    try:
        _, metrics = train_model_csv(seasons, top5, cv_folds=3, model_type="logreg")
        return render_template("training.html", active="", success="Model trained successfully!",
            accuracy=metrics.get("accuracy", 0), test_samples=metrics.get("test_samples", 0),
            train_samples=metrics.get("train_samples", 0), error="")
    except (TrainingError, MissingTokenError) as e:
        return render_template("training.html", active="", success="", accuracy=0,
            test_samples=0, train_samples=0, error=str(e))
    except Exception as e:
        return render_template("training.html", active="", success="", accuracy=0,
            test_samples=0, train_samples=0, error=f"Training failed: {e}")


if __name__ == "__main__":
    print("Starting Predict-XI web UI  ->  http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=True)
