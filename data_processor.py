import csv
import json
import math
from datetime import datetime, timedelta
from collections import defaultdict
from typing import List, Dict, Any, Tuple, Optional

import numpy as np


def process_matches(matches: list) -> list:
    """Convert raw match data into a list of dicts with engineered features."""
    rows = []
    for match in matches:
        if match.get("status") != "FINISHED":
            continue
        home_team = match["homeTeam"]["name"]
        away_team = match["awayTeam"]["name"]
        score = match.get("score", {}).get("fullTime", {})
        home_score = score.get("home")
        away_score = score.get("away")
        if home_score is None or away_score is None:
            continue
        if home_score > away_score:
            result = 1
        elif home_score < away_score:
            result = -1
        else:
            result = 0
        match_date = match.get("utcDate", "")
        competition = match.get("competition", {}).get("code", "")
        rows.append({
            "date": match_date,
            "competition": competition,
            "home_team": home_team,
            "away_team": away_team,
            "home_score": home_score,
            "away_score": away_score,
            "result": result,
            "total_goals": home_score + away_score,
            "goal_diff": home_score - away_score,
        })
    rows.sort(key=lambda r: r["date"])
    return rows


# --- Elo rating parameters -------------------------------------------------
# Standard, data-derived Elo: every team starts from the same anchor and
# moves purely on results. No hand-picked per-league or per-club bonuses —
# those were fabricated precision, not signal. Cross-league strength is left
# to the ML model itself, which already has schedule-strength (`home_sos` /
# `away_sos`) and league identity as separate features.
ELO_START = 1500.0
ELO_K = 24.0
ELO_HOME_ADV = 65.0
ELO_SEASON_GAP = 45
ELO_SEASON_REGRESS = 0.30


def _elo_expected(home_elo: float, away_elo: float) -> float:
    return 1.0 / (1.0 + 10 ** ((away_elo - (home_elo + ELO_HOME_ADV)) / 400.0))


def _elo_goal_diff_multiplier(goal_diff: int) -> float:
    """Margin-of-victory scaling for the Elo update, per the World Football
    Elo Ratings method (eloratings.net): bigger wins move ratings further,
    with diminishing returns, instead of a 1-0 and a 5-0 counting the same."""
    gd = abs(goal_diff)
    if gd <= 1:
        return 1.0
    if gd == 2:
        return 1.5
    return (11 + gd) / 8.0


def _days_between(date1: str, date2: str) -> int:
    try:
        d1 = datetime.fromisoformat(date1.replace("Z", "+00:00"))
        d2 = datetime.fromisoformat(date2.replace("Z", "+00:00"))
        return (d2 - d1).days
    except (ValueError, AttributeError):
        return 7


def add_form_features(rows: list, n_matches: int = 5) -> list:
    """Add rolling form features for each team (home/away specific), plus Elo."""
    if not rows:
        return rows

    team_home_history = defaultdict(list)
    team_away_history = defaultdict(list)
    team_overall_history = defaultdict(list)
    team_elo = {}
    team_season_matches = defaultdict(int)
    result_rows = []

    # Maintain team-pair histories for H2H without rescanning all prior matches
    team_pair_h2h = defaultdict(list)

    for row in rows:
        home = row["home_team"]
        away = row["away_team"]
        result = row["result"]
        match_date = row["date"]

        # Season progress
        team_season_matches[home] += 1
        team_season_matches[away] += 1
        season_match_num = team_season_matches[home] + team_season_matches[away]

        # Home-specific form
        home_recent = team_home_history[home][-n_matches:]
        home_form = sum(m["points"] for m in home_recent) / max(len(home_recent), 1)
        home_gs = sum(m["gs"] for m in home_recent)
        home_gc = sum(m["gc"] for m in home_recent)
        home_mp = len(home_recent)
        home_gd = sum(m["gd"] for m in home_recent)

        # Away-specific form
        away_recent = team_away_history[away][-n_matches:]
        away_form = sum(m["points"] for m in away_recent) / max(len(away_recent), 1)
        away_gs = sum(m["gs"] for m in away_recent)
        away_gc = sum(m["gc"] for m in away_recent)
        away_mp = len(away_recent)
        away_gd = sum(m["gd"] for m in away_recent)

        # Overall form (last n matches regardless of venue)
        home_overall = team_overall_history[home][-n_matches:]
        home_overall_form = sum(m["points"] for m in home_overall) / max(len(home_overall), 1)
        home_overall_gs = sum(m["gs"] for m in home_overall)
        home_overall_gc = sum(m["gc"] for m in home_overall)
        home_overall_gd = sum(m["gd"] for m in home_overall)

        away_overall = team_overall_history[away][-n_matches:]
        away_overall_form = sum(m["points"] for m in away_overall) / max(len(away_overall), 1)
        away_overall_gs = sum(m["gs"] for m in away_overall)
        away_overall_gc = sum(m["gc"] for m in away_overall)
        away_overall_gd = sum(m["gd"] for m in away_overall)

        # Longer windows (10 and 20 matches)
        home_10 = team_overall_history[home][-10:]
        home_20 = team_overall_history[home][-20:]
        away_10 = team_overall_history[away][-10:]
        away_20 = team_overall_history[away][-20:]

        home_ppf_10 = sum(m["points"] for m in home_10) / max(len(home_10), 1)
        home_ppf_20 = sum(m["points"] for m in home_20) / max(len(home_20), 1)
        away_ppf_10 = sum(m["points"] for m in away_10) / max(len(away_10), 1)
        away_ppf_20 = sum(m["points"] for m in away_20) / max(len(away_20), 1)

        # EWMA form (exponential moving average, alpha=0.3)
        def _ewma(history, key="points", alpha=0.3):
            vals = [m[key] for m in history]
            if not vals:
                return 0.0
            w = np.exp(np.arange(len(vals)) * np.log(alpha))
            w = w / w.sum()
            return float(np.dot(vals, w))

        home_ewma_form = _ewma(team_overall_history[home])
        away_ewma_form = _ewma(team_overall_history[away])
        home_ewma_gs = _ewma(team_overall_history[home], "gs")
        away_ewma_gs = _ewma(team_overall_history[away], "gs")
        home_ewma_gc = _ewma(team_overall_history[home], "gc")
        away_ewma_gc = _ewma(team_overall_history[away], "gc")

        # Streaks
        def _current_streak(history, good_value=3):
            if not history:
                return 0
            last = history[-1]["points"]
            if last == 0:
                return 0
            count = 0
            for m in reversed(history):
                if m["points"] == last:
                    count += 1
                else:
                    break
            return count if last == good_value else -count

        home_streak = _current_streak(team_overall_history[home])
        away_streak = _current_streak(team_overall_history[away])

        # Clean sheet rate (last 5, 10)
        home_cs_5 = sum(1 for m in team_overall_history[home][-5:] if m["gc"] == 0) / max(len(team_overall_history[home][-5:]), 1)
        home_cs_10 = sum(1 for m in team_overall_history[home][-10:] if m["gc"] == 0) / max(len(team_overall_history[home][-10:]), 1)
        away_cs_5 = sum(1 for m in team_overall_history[away][-5:] if m["gc"] == 0) / max(len(team_overall_history[away][-5:]), 1)
        away_cs_10 = sum(1 for m in team_overall_history[away][-10:] if m["gc"] == 0) / max(len(team_overall_history[away][-10:]), 1)

        # BTTS proxy (both teams scored)
        home_btts_5 = sum(1 for m in team_overall_history[home][-5:] if m["gs"] > 0 and m["gc"] > 0) / max(len(team_overall_history[home][-5:]), 1)
        away_btts_5 = sum(1 for m in team_overall_history[away][-5:] if m["gs"] > 0 and m["gc"] > 0) / max(len(team_overall_history[away][-5:]), 1)

        # Over 2.5 goals rate
        home_o25_5 = sum(1 for m in team_overall_history[home][-5:] if m["gd"] != 0 and (m["gs"] + m["gc"]) > 2.5) / max(len(team_overall_history[home][-5:]), 1)
        away_o25_5 = sum(1 for m in team_overall_history[away][-5:] if m["gd"] != 0 and (m["gs"] + m["gc"]) > 2.5) / max(len(team_overall_history[away][-5:]), 1)

        # Home/away split performance
        home_home_10 = team_home_history[home][-10:]
        home_home_ppf = sum(m["points"] for m in home_home_10) / max(len(home_home_10), 1)
        away_away_10 = team_away_history[away][-10:]
        away_away_ppf = sum(m["points"] for m in away_away_10) / max(len(away_away_10), 1)

        # Strength of schedule (average opponent Elo in last 5 matches)
        def _avg_opponent_elo(history, team_elo_map):
            if not history:
                return ELO_START
            elos = []
            for m in history[-5:]:
                opp = m.get("opponent")
                if opp and opp in team_elo_map:
                    elos.append(team_elo_map[opp])
            return sum(elos) / len(elos) if elos else ELO_START

        home_sos = _avg_opponent_elo(team_overall_history[home], team_elo)
        away_sos = _avg_opponent_elo(team_overall_history[away], team_elo)

        # Head-to-head (using pre-built pair history)
        pair_key = tuple(sorted([home, away]))
        h2h_all = team_pair_h2h[pair_key]
        h2h_recent = h2h_all[-5:]
        h2h_home_wins = sum(1 for m in h2h_recent if m.get("home_win") == 1)
        h2h_draws = sum(1 for m in h2h_recent if m.get("draw") == 1)
        h2h_away_wins = sum(1 for m in h2h_recent if m.get("away_win") == 1)

        # Rest days
        home_last_date = team_overall_history[home][-1]["date"] if team_overall_history[home] else None
        away_last_date = team_overall_history[away][-1]["date"] if team_overall_history[away] else None
        home_rest_days = _days_between(home_last_date, match_date) if home_last_date else 7
        away_rest_days = _days_between(away_last_date, match_date) if away_last_date else 7

        # Elo
        for team, last_date in ((home, home_last_date), (away, away_last_date)):
            if team not in team_elo:
                team_elo[team] = ELO_START
            elif last_date and _days_between(last_date, match_date) > ELO_SEASON_GAP:
                team_elo[team] = ELO_START + (team_elo[team] - ELO_START) * (1 - ELO_SEASON_REGRESS)
        home_earned = team_elo[home]
        away_earned = team_elo[away]
        home_elo = home_earned
        away_elo = away_earned

        new_row = dict(row)
        new_row.update({
            "home_form": home_form,
            "home_goals_scored_avg": home_gs / max(home_mp, 1),
            "home_goals_conceded_avg": home_gc / max(home_mp, 1),
            "home_matches_played": home_mp,
            "away_form": away_form,
            "away_goals_scored_avg": away_gs / max(away_mp, 1),
            "away_goals_conceded_avg": away_gc / max(away_mp, 1),
            "away_matches_played": away_mp,
            "home_overall_form": home_overall_form,
            "home_overall_goals_scored_avg": home_overall_gs / max(len(home_overall), 1),
            "home_overall_goals_conceded_avg": home_overall_gc / max(len(home_overall), 1),
            "away_overall_form": away_overall_form,
            "away_overall_goals_scored_avg": away_overall_gs / max(len(away_overall), 1),
            "away_overall_goals_conceded_avg": away_overall_gc / max(len(away_overall), 1),
            "h2h_matches": len(h2h_recent),
            "h2h_home_wins": h2h_home_wins,
            "h2h_draws": h2h_draws,
            "h2h_away_wins": h2h_away_wins,
            "home_rest_days": min(home_rest_days, 30),
            "away_rest_days": min(away_rest_days, 30),
            "home_elo": home_elo,
            "away_elo": away_elo,
            "elo_diff": home_elo - away_elo,
            # Advanced features
            "home_ppf_10": home_ppf_10,
            "home_ppf_20": home_ppf_20,
            "away_ppf_10": away_ppf_10,
            "away_ppf_20": away_ppf_20,
            "home_ewma_form": home_ewma_form,
            "away_ewma_form": away_ewma_form,
            "home_ewma_gs": home_ewma_gs,
            "away_ewma_gs": away_ewma_gs,
            "home_ewma_gc": home_ewma_gc,
            "away_ewma_gc": away_ewma_gc,
            "home_streak": home_streak,
            "away_streak": away_streak,
            "home_cs_rate_5": home_cs_5,
            "away_cs_rate_5": away_cs_5,
            "home_cs_rate_10": home_cs_10,
            "away_cs_rate_10": away_cs_10,
            "home_btts_rate_5": home_btts_5,
            "away_btts_rate_5": away_btts_5,
            "home_o25_rate_5": home_o25_5,
            "away_o25_rate_5": away_o25_5,
            "home_home_ppf_10": home_home_ppf,
            "away_away_ppf_10": away_away_ppf,
            "home_sos": home_sos,
            "away_sos": away_sos,
            "home_gd_5": home_gd,
            "away_gd_5": away_gd,
            "home_gd_10": home_overall_gd / max(len(home_overall), 1),
            "away_gd_10": away_overall_gd / max(len(away_overall), 1),
            "season_progress": min(season_match_num / 40.0, 1.0),
        })
        result_rows.append(new_row)

        # Update Elo (margin-of-victory scaled)
        exp_home = _elo_expected(home_earned, away_earned)
        actual_home = 1.0 if result == 1 else (0.5 if result == 0 else 0.0)
        mov = _elo_goal_diff_multiplier(row["goal_diff"])
        elo_delta = ELO_K * mov * (actual_home - exp_home)
        team_elo[home] = home_earned + elo_delta
        team_elo[away] = away_earned - elo_delta
        new_row["home_elo_post"] = team_elo[home]
        new_row["away_elo_post"] = team_elo[away]

        home_points = 3 if result == 1 else (1 if result == 0 else 0)
        away_points = 3 if result == -1 else (1 if result == 0 else 0)

        team_home_history[home].append({
            "points": home_points, "gs": row["home_score"], "gc": row["away_score"],
            "gd": row["goal_diff"], "date": match_date, "opponent": away, "result": result,
        })
        team_away_history[away].append({
            "points": away_points, "gs": row["away_score"], "gc": row["home_score"],
            "gd": -row["goal_diff"], "date": match_date, "opponent": home, "result": -result,
        })
        team_overall_history[home].append({
            "points": home_points, "gs": row["home_score"], "gc": row["away_score"],
            "gd": row["goal_diff"], "date": match_date, "opponent": away, "result": result,
        })
        team_overall_history[away].append({
            "points": away_points, "gs": row["away_score"], "gc": row["home_score"],
            "gd": -row["goal_diff"], "date": match_date, "opponent": home, "result": -result,
        })

        # Update H2H pair history
        team_pair_h2h[pair_key].append({
            "home_win": 1 if result == 1 else 0,
            "draw": 1 if result == 0 else 0,
            "away_win": 1 if result == -1 else 0,
        })

    return result_rows


def prepare_training_data(rows: list) -> Tuple[List[List[float]], List[int], List[str]]:
    base_form_cols = [
        "home_form", "home_goals_scored_avg", "home_goals_conceded_avg", "home_matches_played",
        "away_form", "away_goals_scored_avg", "away_goals_conceded_avg", "away_matches_played",
        "home_overall_form", "home_overall_goals_scored_avg", "home_overall_goals_conceded_avg",
        "away_overall_form", "away_overall_goals_scored_avg", "away_overall_goals_conceded_avg",
        "h2h_matches", "h2h_home_wins", "h2h_draws", "h2h_away_wins",
        "home_rest_days", "away_rest_days",
    ]
    advanced_form_cols = [
        "home_ppf_10", "home_ppf_20", "away_ppf_10", "away_ppf_20",
        "home_ewma_form", "away_ewma_form", "home_ewma_gs", "away_ewma_gs",
        "home_ewma_gc", "away_ewma_gc",
        "home_streak", "away_streak",
        "home_cs_rate_5", "away_cs_rate_5", "home_cs_rate_10", "away_cs_rate_10",
        "home_btts_rate_5", "away_btts_rate_5", "home_o25_rate_5", "away_o25_rate_5",
        "home_home_ppf_10", "away_away_ppf_10",
        "home_sos", "away_sos",
        "home_gd_5", "away_gd_5", "home_gd_10", "away_gd_10",
        "season_progress",
    ]
    elo_cols = ["home_elo", "away_elo", "elo_diff"]
    feature_cols = base_form_cols + advanced_form_cols + elo_cols

    X = []
    y = []
    for row in rows:
        if any(row.get(col) is None for col in base_form_cols + elo_cols):
            continue
        if all(row.get(col, 0) == 0 for col in base_form_cols):
            continue
        X.append([row.get(col, 0) for col in feature_cols])
        y.append(row["result"])

    return X, y, feature_cols


def prepare_prediction_features(home_team: str, away_team: str,
                                team_stats: dict) -> list:
    home = team_stats.get(home_team, {})
    away = team_stats.get(away_team, {})
    home_elo = home.get("elo", ELO_START)
    away_elo = away.get("elo", ELO_START)

    base_features = [
        home.get("form", 0), home.get("goals_scored_avg", 0), home.get("goals_conceded_avg", 0),
        home.get("matches_played", 0),
        away.get("form", 0), away.get("goals_scored_avg", 0), away.get("goals_conceded_avg", 0),
        away.get("matches_played", 0),
        home.get("overall_form", 0), home.get("overall_goals_scored_avg", 0), home.get("overall_goals_conceded_avg", 0),
        away.get("overall_form", 0), away.get("overall_goals_scored_avg", 0), away.get("overall_goals_conceded_avg", 0),
        home.get("h2h_matches", 0), home.get("h2h_home_wins", 0), home.get("h2h_draws", 0), home.get("h2h_away_wins", 0),
        home.get("rest_days", 7), away.get("rest_days", 7),
        # Advanced
        home.get("ppf_10", 0), home.get("ppf_20", 0), away.get("ppf_10", 0), away.get("ppf_20", 0),
        home.get("ewma_form", 0), away.get("ewma_form", 0),
        home.get("ewma_gs", 0), away.get("ewma_gs", 0),
        home.get("ewma_gc", 0), away.get("ewma_gc", 0),
        home.get("streak", 0), away.get("streak", 0),
        home.get("cs_rate_5", 0), away.get("cs_rate_5", 0),
        home.get("cs_rate_10", 0), away.get("cs_rate_10", 0),
        home.get("btts_rate_5", 0), away.get("btts_rate_5", 0),
        home.get("o25_rate_5", 0), away.get("o25_rate_5", 0),
        home.get("home_ppf_10", 0), away.get("away_ppf_10", 0),
        home.get("sos", ELO_START), away.get("sos", ELO_START),
        home.get("gd_5", 0), away.get("gd_5", 0),
        home.get("gd_10", 0), away.get("gd_10", 0),
        home.get("season_progress", 0),
    ]
    elo_features = [home_elo, away_elo, home_elo - away_elo]
    return base_features + elo_features


def compute_team_stats(rows: list) -> dict:
    if not rows:
        return {}

    team_stats = defaultdict(lambda: {
        "form": 0, "goals_scored_avg": 0, "goals_conceded_avg": 0, "matches_played": 0,
        "overall_form": 0, "overall_goals_scored_avg": 0, "overall_goals_conceded_avg": 0,
        "h2h_matches": 0, "h2h_home_wins": 0, "h2h_draws": 0, "h2h_away_wins": 0,
        "rest_days": 7, "elo": ELO_START, "league": "",
        "ppf_10": 0, "ppf_20": 0, "ewma_form": 0, "ewma_gs": 0, "ewma_gc": 0,
        "streak": 0, "cs_rate_5": 0, "cs_rate_10": 0, "btts_rate_5": 0, "o25_rate_5": 0,
        "home_ppf_10": 0, "away_ppf_10": 0, "sos": ELO_START,
        "gd_5": 0, "gd_10": 0, "season_progress": 0,
    })

    # Map team_stats keys to row prefixes
    prefix_map = {
        "form": "form", "goals_scored_avg": "goals_scored_avg", "goals_conceded_avg": "goals_conceded_avg",
        "matches_played": "matches_played", "overall_form": "overall_form",
        "overall_goals_scored_avg": "overall_goals_scored_avg", "overall_goals_conceded_avg": "overall_goals_conceded_avg",
        "h2h_matches": "h2h_matches", "h2h_home_wins": "h2h_home_wins", "h2h_draws": "h2h_draws", "h2h_away_wins": "h2h_away_wins",
        "rest_days": "rest_days", "ppf_10": "ppf_10", "ppf_20": "ppf_20",
        "ewma_form": "ewma_form", "ewma_gs": "ewma_gs", "ewma_gc": "ewma_gc",
        "streak": "streak", "cs_rate_5": "cs_rate_5", "cs_rate_10": "cs_rate_10",
        "btts_rate_5": "btts_rate_5", "o25_rate_5": "o25_rate_5",
        "home_ppf_10": "home_ppf_10", "away_ppf_10": "away_ppf_10",
        "sos": "sos", "gd_5": "gd_5", "gd_10": "gd_10", "season_progress": "season_progress",
    }

    for row in reversed(rows):
        home = row["home_team"]
        away = row["away_team"]
        if team_stats[home]["matches_played"] == 0:
            for key, base in prefix_map.items():
                row_key = f"home_{base}"
                if row_key in row:
                    team_stats[home][key] = row[row_key]
            team_stats[home]["elo"] = row.get("home_elo_post", ELO_START)
            team_stats[home]["league"] = row.get("competition", "")
        if team_stats[away]["matches_played"] == 0:
            for key, base in prefix_map.items():
                row_key = f"away_{base}"
                if row_key in row:
                    team_stats[away][key] = row[row_key]
            team_stats[away]["elo"] = row.get("away_elo_post", ELO_START)
            team_stats[away]["league"] = row.get("competition", "")
        if all(team_stats[t]["matches_played"] > 0 for t in [home, away]):
            pass

    return dict(team_stats)


def save_data(rows: list, path: str = "matches_data.json"):
    with open(path, "w") as f:
        json.dump(rows, f, indent=2)


def load_data(path: str = "matches_data.json") -> list:
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []
