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
from csv_data_loader import LEAGUE_CODE_MAP
from api_client import fetch_upcoming_matches, MissingTokenError
from data_processor import prepare_prediction_features, load_data, compute_team_stats
from model_trainer import MatchPredictorModel
from team_aliases import resolve_team_name

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
    # Trained-on competitions that had no display name, so the per-league
    # dashboard table was showing raw codes for them.
    "ELC2": "League One", "ELC3": "League Two", "ENG5": "National League",
    "BSA": "Brasileirao Serie A",
}

# ─── Helpers ──────────────────────────────────────────────────────────────

def _load_json(name):
    try:
        with open(os.path.join(SCRIPT_DIR, name)) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def load_metrics():
    return _load_json("model_metrics.json") or {}


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
    # evaluation.json is produced by evaluate.py after training. It's
    # optional: the dashboard still renders without it, just without the
    # per-league and calibration sections.
    evaluation = _load_json("evaluation.json")
    return render_template(
        "dashboard.html",
        active="dashboard",
        metrics=metrics,
        evaluation=evaluation,
        league_names=LEAGUE_NAMES,
        n_train_leagues=len(LEAGUE_CODE_MAP),
        error=request.args.get("error", ""),
    )


@app.route("/predict")
def predict():
    stats = build_team_stats()
    teams = team_list(stats)
    home_team = request.args.get("home", "").strip()
    away_team = request.args.get("away", "").strip()

    # The Fixtures page already resolves live-API names before linking here,
    # but resolve again as a fallback (e.g. a hand-typed/bookmarked URL using
    # an official long name) — never guess beyond the verified alias table,
    # an unresolved team just means we predict with neutral defaults and say
    # so clearly (see the "limited data" warning below), not a wrong team.
    home_lookup = home_team if home_team in stats else (resolve_team_name(home_team, stats) or home_team)
    away_lookup = away_team if away_team in stats else (resolve_team_name(away_team, stats) or away_team)

    # Each side's league picker is independent (any two teams can be
    # matched up). If we arrived with a team already chosen (e.g. from the
    # Fixtures "Predict" button) but no explicit league, infer it from that
    # team so the picker still lands pre-populated instead of empty.
    sel_home_league = stats.get(home_lookup, {}).get("league", "") if home_lookup else ""
    sel_away_league = stats.get(away_lookup, {}).get("league", "") if away_lookup else ""

    common = dict(
        teams=teams,
        teams_json=json.dumps(teams),
        league_opts=league_options(teams),
        sel_home=home_team, sel_away=away_team,
        sel_home_league=sel_home_league, sel_away_league=sel_away_league,
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

    features = prepare_prediction_features(home_lookup, away_lookup, stats)
    result = model.predict(features)

    probs = result["probabilities"]
    p_home = probs.get("Home Win", 0.0)
    p_draw = probs.get("Draw", 0.0)
    p_away = probs.get("Away Win", 0.0)
    top_prob = max(p_home, p_draw, p_away)
    conf = "High" if top_prob >= 0.55 else ("Moderate" if top_prob >= 0.42 else "Low")

    h_info = team_info(stats, home_lookup)
    a_info = team_info(stats, away_lookup)

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

    stats = build_team_stats()

    def _predict_name(team):
        """The live API uses official long names ('Manchester United FC')
        that don't match the training data's short names ('Man United').
        Resolve to whichever name our stats actually recognize, trying the
        official name then the API's shortName, so the Predict link lands on
        real stats instead of silently falling back to neutral defaults."""
        name = team.get("name", "")
        resolved = resolve_team_name(name, stats) or resolve_team_name(team.get("shortName", ""), stats)
        return resolved or name

    fixtures_list = []
    for m in raw:
        date_str = m.get("utcDate", "")
        try:
            date_str = datetime.fromisoformat(date_str.replace("Z", "+00:00")).strftime("%b %d, %Y · %H:%M")
        except (ValueError, AttributeError):
            pass
        home, away = m.get("homeTeam", {}), m.get("awayTeam", {})
        fixtures_list.append({
            "home_team": home.get("name", "Unknown"),
            "away_team": away.get("name", "Unknown"),
            "home_predict": _predict_name(home),
            "away_predict": _predict_name(away),
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
