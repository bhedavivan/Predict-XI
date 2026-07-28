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
from data_processor import prepare_prediction_features, load_data, compute_team_stats, explain_prediction
from model_trainer import MatchPredictorModel
from team_aliases import resolve_team_name, team_display_name
import leagues

app = Flask(__name__)
app.jinja_env.filters['urlencode'] = lambda s: urllib.parse.quote(str(s), safe='')


# ─── Helpers ──────────────────────────────────────────────────────────────

def _load_json(name):
    try:
        with open(os.path.join(SCRIPT_DIR, name), encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def load_metrics():
    return _load_json("model_metrics.json") or {}


def build_h2h_map():
    """Pair-keyed head-to-head history. Kept separate from team_stats because
    H2H describes two clubs together — storing it per team yielded whatever
    that team's last unrelated fixture showed, which read as 0 everywhere."""
    return _load_json("h2h_stats.json") or {}


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
    # Require BOTH the metadata and the pipeline: model.json alone doesn't make
    # a usable model (load() returns False without the .joblib), so checking
    # only the json disagreed with get_model() when the pipeline was missing.
    return (os.path.exists(os.path.join(SCRIPT_DIR, "model.json"))
            and os.path.exists(os.path.join(SCRIPT_DIR, "model.joblib")))


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
    return leagues.display_name(code)


def team_list(stats):
    """Teams for the chooser — TOP-FLIGHT leagues only, each with Elo, tier,
    league and a canonical display name. Sorted by Elo (valid within the app's
    grouping; leagues themselves are strength-ordered by league_options)."""
    out = []
    for name, s in stats.items():
        if not name or s.get("matches_played", 0) <= 0:
            continue
        lg = s.get("league", "")
        if lg not in leagues.TOP_FLIGHT_CODES:
            continue   # hide second/lower divisions from the whole app
        elo = s.get("elo", 1500)
        out.append({"name": name, "display": team_display_name(name),
                    "elo": round(elo, 1), "tier": team_tier(elo),
                    "league": lg, "league_name": league_name(lg)})
    out.sort(key=lambda t: (-t["elo"], t["display"]))
    return out


def team_info(stats, name):
    s = stats.get(name, {})
    elo = s.get("elo", 1500)
    lg = s.get("league", "")
    return {"elo": elo, "tier": team_tier(elo), "known": bool(s),
            "league": lg, "league_name": league_name(lg)}


def league_options(teams):
    """Distinct (code, name) leagues present, ordered by STRENGTH rank
    (Premier League first) rather than alphabetically."""
    seen = {}
    for t in teams:
        if t["league"]:
            seen[t["league"]] = t["league_name"]
    return sorted(seen.items(), key=lambda kv: leagues.league_rank(kv[0]))


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
        n_train_leagues=len(leagues.TOP_FLIGHT_CODES),
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

    # Guard the feature-build + predict: a stale artifact (model.joblib trained
    # with a different feature count than the current data_processor produces)
    # would otherwise raise a raw 500. Show a clear message instead.
    try:
        features = prepare_prediction_features(home_lookup, away_lookup, stats, build_h2h_map())
        result = model.predict(features)
    except Exception as e:
        common["error"] = (f"Prediction failed ({type(e).__name__}). The shipped model and the "
                           f"feature builder may be out of sync — retrain with "
                           f"`python main.py --all-seasons --all-leagues --force-retrain`.")
        return render_template("predict.html", active="predict", prediction=None, **common)

    # Live team-news nudge (serving-only, never trained): if a player-data
    # source is configured (a bring-your-own JSON file or an API token), shift
    # the forecast for injured/suspended key players. Fully inert otherwise, and
    # wrapped so it can never break a prediction.
    live_adjusted = False
    try:
        from player_data import get_source, live_availability_adjustment
        src = get_source()
        if src is not None:
            today = datetime.utcnow().strftime("%Y-%m-%d")
            ha, aa = src.availability(home_lookup, today), src.availability(away_lookup, today)
            adj = live_availability_adjustment(ha, aa, result["probabilities"])
            if adj != result["probabilities"]:
                result["probabilities"] = adj
                live_adjusted = True
    except Exception:
        pass

    probs = result["probabilities"]
    p_home = probs.get("Home Win", 0.0)
    p_draw = probs.get("Draw", 0.0)
    p_away = probs.get("Away Win", 0.0)
    top_prob = max(p_home, p_draw, p_away)
    conf = "High" if top_prob >= 0.55 else ("Moderate" if top_prob >= 0.42 else "Low")

    h_info = team_info(stats, home_lookup)
    a_info = team_info(stats, away_lookup)

    # "Why" drivers — the model reads these exact quantities, so surfacing them
    # turns a bare percentage into an argument (now that the probabilities are
    # honestly calibrated, showing the reasoning is worth a lot).
    try:
        explanation = explain_prediction(home_lookup, away_lookup, stats, build_h2h_map())
    except Exception:
        explanation = None

    return render_template(
        "predict.html", active="predict", prediction=result,
        home_team=team_display_name(home_lookup) if h_info["known"] else home_team,
        away_team=team_display_name(away_lookup) if a_info["known"] else away_team,
        p_home=p_home, p_draw=p_draw, p_away=p_away,
        top_prob=top_prob, confidence_label=conf,
        home_info=h_info, away_info=a_info,
        elo_diff=h_info["elo"] - a_info["elo"],
        both_known=h_info["known"] and a_info["known"],
        cross_league=h_info["league"] and a_info["league"] and h_info["league"] != a_info["league"],
        bars=[("Home Win", p_home, "var(--home)"), ("Draw", p_draw, "var(--draw)"),
              ("Away Win", p_away, "var(--away)")],
        explanation=explanation,
        live_adjusted=live_adjusted,
        **common,
    )


@app.route("/simulate")
def simulate():
    """Monte Carlo season projection. For leagues the live API covers, it
    projects the REAL remaining fixtures on top of the ACTUAL current standings
    (so the numbers move as matches are played — like the online supercomputers);
    otherwise it plays a hypothetical full season from current ratings. Either
    way it aggregates the model's calibrated match probabilities — it does not
    change any single match's forecast."""
    stats = build_team_stats()
    teams = team_list(stats)
    league_opts = league_options(teams)   # NB: don't shadow the `leagues` module
    league_code = request.args.get("league", "").strip()
    rows, positions = [], []
    err, lname, mode, as_of, remaining, live = "", "", "", "", 0, False
    rules, n_sims_used = {}, 0
    tier_top, tier_second = "UCL", "UEL"
    if league_code:
        lname = league_name(league_code)
        try:
            import numpy as np
            from simulate_season import simulate_league
            # A fresh seed (the Re-simulate button) lets the user watch the Monte
            # Carlo vary; the default is deterministic so a plain reload is stable.
            try:
                seed = int(request.args.get("seed", "42")) % (2**31)
            except ValueError:
                seed = 42
            out = simulate_league(league_code, seed=seed)
            mode = out.get("mode", "")
            live = mode == "live"
            as_of = out.get("as_of") or ""
            remaining = out.get("remaining", 0)
            rules = out.get("rules", {})
            n_sims_used = out.get("n_sims", 0)
            tier_top, tier_second = leagues.tier_labels(league_code)
            title = np.asarray(out["title_prob"])
            meanpos = np.asarray(out["mean_position"])
            order = list(np.argsort(-title + meanpos * 1e-6))
            posmat = np.asarray(out["position_matrix"])
            n = len(out["teams"])
            positions = list(range(1, n + 1))
            cur_pts, cur_pos = out.get("current_points"), out.get("current_position")

            def pct(arr, i):
                return round(float(out[arr][i]) * 100, 1)

            def ci(arr, i):
                return round(float(out[arr][i]) * 100 * 1.96, 1)   # ~95% half-width

            for i in order:
                rows.append({
                    "team": team_display_name(out["teams"][i]),
                    "now_pts": (int(cur_pts[i]) if cur_pts else None),
                    "now_pos": (int(cur_pos[i]) if cur_pos else None),
                    "exp_pts": round(float(out["expected_points"][i]), 1),
                    "p10": int(round(float(out["points_p10"][i]))),
                    "p90": int(round(float(out["points_p90"][i]))),
                    "title": pct("title_prob", i), "title_ci": ci("title_se", i),
                    "ucl": pct("ucl_prob", i), "ucl_ci": ci("ucl_se", i),
                    "uel": pct("uel_prob", i),
                    "playoff": pct("playoff_prob", i),
                    "releg": pct("relegation_prob", i), "releg_ci": ci("relegation_se", i),
                    "avg_pos": round(float(out["mean_position"][i]), 1),
                    "exp_gf": round(float(out["expected_gf"][i]), 1),
                    "exp_ga": round(float(out["expected_ga"][i]), 1),
                    "exp_gd": round(float(out["expected_gd"][i]), 1),
                    "btts": pct("btts_pct", i), "over25": pct("over25_pct", i),
                    "cells": [round(float(posmat[i][j]) * 100, 1) for j in range(n)],
                })
        except SystemExit as e:
            err = str(e)
        except Exception as e:
            err = f"Simulation failed: {e}"
    return render_template("simulate.html", active="simulate", leagues=league_opts,
                           selected=league_code, league_name=lname, rows=rows,
                           positions=positions, mode=mode, live=live, as_of=as_of,
                           remaining=remaining, rules=rules, n_sims=n_sims_used,
                           tier_top=tier_top, tier_second=tier_second, error=err)


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

    def _resolve(team):
        """The live API uses official long names ('Manchester United FC')
        that don't match the training data's short names ('Man United').
        Resolve to whichever name our stats recognize (official then shortName),
        so the Predict link lands on real stats, not neutral defaults, and the
        DISPLAYED name matches the rest of the app."""
        name = team.get("name", "")
        return resolve_team_name(name, stats) or resolve_team_name(team.get("shortName", ""), stats) or name

    fixtures_list = []
    for m in raw:
        date_str = m.get("utcDate", "")
        try:
            date_str = datetime.fromisoformat(date_str.replace("Z", "+00:00")).strftime("%b %d, %Y · %H:%M")
        except (ValueError, AttributeError):
            pass
        home, away = m.get("homeTeam", {}), m.get("awayTeam", {})
        h_res, a_res = _resolve(home), _resolve(away)
        fixtures_list.append({
            # Show the canonical name (resolved → pretty) so a club reads the
            # same on Fixtures as on Predict/Simulate.
            "home_team": team_display_name(h_res) if h_res in stats else h_res,
            "away_team": team_display_name(a_res) if a_res in stats else a_res,
            "home_predict": h_res,
            "away_predict": a_res,
            "date": date_str,
        })

    # Inline model prediction per fixture (batched) + a difficulty colour, so
    # the schedule reads at a glance which games are one-sided vs coin-flips.
    try:
        model = get_model()
        if model is not None:
            h2h = build_h2h_map()
            idx, feats = [], []
            for i, m in enumerate(fixtures_list):
                if m["home_predict"] in stats and m["away_predict"] in stats:
                    idx.append(i)
                    feats.append(prepare_prediction_features(m["home_predict"], m["away_predict"], stats, h2h))
            if feats:
                for i, pr in zip(idx, model.predict_proba_batch(feats)):
                    p = {"Home": pr.get("Home Win", 0.0), "Draw": pr.get("Draw", 0.0),
                         "Away": pr.get("Away Win", 0.0)}
                    fav = max(p, key=p.get)
                    top = p[fav]
                    fixtures_list[i].update({
                        "fav": fav, "fav_pct": round(top * 100),
                        "diff": "strong" if top >= 0.55 else ("mid" if top >= 0.42 else "open"),
                    })
    except Exception:
        pass   # predictions are a nice-to-have; never break the fixtures list

    return render_template(
        "fixtures.html", active="fixtures", leagues=LEAGUE_CODES, selected=league_code,
        league_name=LEAGUE_CODES.get(league_code, league_code),
        fixtures=fixtures_list, error=error,
    )


@app.route("/train", methods=["POST"])
def train():
    from main import train_model_csv, TrainingError
    # This endpoint OVERWRITES the shipped model/artifacts, so it must never be
    # reachable once a model exists: the app ships pretrained, retraining is a
    # deliberate offline step (`python main.py --all-seasons --all-leagues
    # --force-retrain`), and an unauthenticated POST that clobbers the ensemble
    # (and hammers upstream feeds on the single dev thread) is both a footgun
    # and a DoS vector. Only the genuine cold-start (no model) may bootstrap.
    if model_exists():
        return render_template("training.html", active="", success="", accuracy=0,
            test_samples=0, train_samples=0,
            error="A trained model already ships with the app. Retrain offline with "
                  "`python main.py --all-seasons --all-leagues --force-retrain`."), 403
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
    # debug/host are env-gated: the Werkzeug debugger is a remote-code-execution
    # console, so it must NOT default to on and must NOT bind to 0.0.0.0. Opt in
    # explicitly for local debugging (FLASK_DEBUG=1); default is loopback only.
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "5000"))
    debug = os.getenv("FLASK_DEBUG") == "1"
    print(f"Starting Predict-XI web UI  ->  http://{host}:{port}")
    app.run(host=host, port=port, debug=debug)
