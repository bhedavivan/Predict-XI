import csv
import json
from datetime import datetime, timedelta
from collections import defaultdict
from typing import List, Dict, Any, Tuple, Optional


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

    # Sort by date
    rows.sort(key=lambda r: r["date"])
    return rows


# --- Elo rating parameters -------------------------------------------------
ELO_START = 1500.0      # every team starts here
ELO_K = 24.0            # update magnitude per match
ELO_HOME_ADV = 65.0     # home-field advantage, in Elo points


def _elo_expected(home_elo: float, away_elo: float) -> float:
    """Expected score for the home team (0..1) given both ratings + home edge."""
    return 1.0 / (1.0 + 10 ** ((away_elo - (home_elo + ELO_HOME_ADV)) / 400.0))


def add_form_features(rows: list, n_matches: int = 5) -> list:
    """Add rolling form features for each team (home/away specific), plus Elo."""
    if not rows:
        return rows

    # Track history separately for home and away matches
    team_home_history = defaultdict(list)
    team_away_history = defaultdict(list)
    team_overall_history = defaultdict(list)
    team_elo = defaultdict(lambda: ELO_START)  # running Elo rating per team
    result_rows = []

    for row in rows:
        home = row["home_team"]
        away = row["away_team"]
        result = row["result"]
        match_date = row["date"]

        # Home team features (using home history)
        home_recent = team_home_history[home][-n_matches:] if home in team_home_history else []
        home_form = sum(m["points"] for m in home_recent) / max(len(home_recent), 1)
        home_gs = sum(m["gs"] for m in home_recent)
        home_gc = sum(m["gc"] for m in home_recent)
        home_mp = len(home_recent)

        # Away team features (using away history)
        away_recent = team_away_history[away][-n_matches:] if away in team_away_history else []
        away_form = sum(m["points"] for m in away_recent) / max(len(away_recent), 1)
        away_gs = sum(m["gs"] for m in away_recent)
        away_gc = sum(m["gc"] for m in away_recent)
        away_mp = len(away_recent)

        # Overall form (last n matches regardless of venue)
        home_overall = team_overall_history[home][-n_matches:] if home in team_overall_history else []
        home_overall_form = sum(m["points"] for m in home_overall) / max(len(home_overall), 1)
        home_overall_gs = sum(m["gs"] for m in home_overall)
        home_overall_gc = sum(m["gc"] for m in home_overall)

        away_overall = team_overall_history[away][-n_matches:] if away in team_overall_history else []
        away_overall_form = sum(m["points"] for m in away_overall) / max(len(away_overall), 1)
        away_overall_gs = sum(m["gs"] for m in away_overall)
        away_overall_gc = sum(m["gc"] for m in away_overall)

        # Head-to-head history
        h2h_matches = [m for m in team_overall_history[home] if m.get("opponent") == away]
        h2h_recent = h2h_matches[-5:] if h2h_matches else []
        h2h_home_wins = sum(1 for m in h2h_recent if m.get("result") == 1)
        h2h_draws = sum(1 for m in h2h_recent if m.get("result") == 0)
        h2h_away_wins = sum(1 for m in h2h_recent if m.get("result") == -1)

        # Rest days (days since last match)
        home_last_date = team_overall_history[home][-1]["date"] if team_overall_history[home] else None
        away_last_date = team_overall_history[away][-1]["date"] if team_overall_history[away] else None
        home_rest_days = _days_between(home_last_date, match_date) if home_last_date else 7
        away_rest_days = _days_between(away_last_date, match_date) if away_last_date else 7

        # Elo ratings (pre-match — no leakage)
        home_elo = team_elo[home]
        away_elo = team_elo[away]

        new_row = dict(row)
        new_row.update({
            # Home-specific form
            "home_form": home_form,
            "home_goals_scored_avg": home_gs / max(home_mp, 1),
            "home_goals_conceded_avg": home_gc / max(home_mp, 1),
            "home_matches_played": home_mp,
            # Away-specific form
            "away_form": away_form,
            "away_goals_scored_avg": away_gs / max(away_mp, 1),
            "away_goals_conceded_avg": away_gc / max(away_mp, 1),
            "away_matches_played": away_mp,
            # Overall form
            "home_overall_form": home_overall_form,
            "home_overall_goals_scored_avg": home_overall_gs / max(len(home_overall), 1),
            "home_overall_goals_conceded_avg": home_overall_gc / max(len(home_overall), 1),
            "away_overall_form": away_overall_form,
            "away_overall_goals_scored_avg": away_overall_gs / max(len(away_overall), 1),
            "away_overall_goals_conceded_avg": away_overall_gc / max(len(away_overall), 1),
            # Head-to-head
            "h2h_matches": len(h2h_recent),
            "h2h_home_wins": h2h_home_wins,
            "h2h_draws": h2h_draws,
            "h2h_away_wins": h2h_away_wins,
            # Rest days
            "home_rest_days": min(home_rest_days, 30),  # Cap at 30
            "away_rest_days": min(away_rest_days, 30),
            # Elo (pre-match ratings + gap)
            "home_elo": home_elo,
            "away_elo": away_elo,
            "elo_diff": home_elo - away_elo,
        })
        result_rows.append(new_row)

        # Update Elo ratings from this match result
        exp_home = _elo_expected(home_elo, away_elo)
        actual_home = 1.0 if result == 1 else (0.5 if result == 0 else 0.0)
        elo_delta = ELO_K * (actual_home - exp_home)
        team_elo[home] = home_elo + elo_delta
        team_elo[away] = away_elo - elo_delta
        # Post-match ratings, used later to read each team's current strength
        new_row["home_elo_post"] = team_elo[home]
        new_row["away_elo_post"] = team_elo[away]

        # Update team history
        home_points = 3 if result == 1 else (1 if result == 0 else 0)
        away_points = 3 if result == -1 else (1 if result == 0 else 0)

        # Home history
        team_home_history[home].append({
            "points": home_points,
            "gs": row["home_score"],
            "gc": row["away_score"],
            "date": match_date,
            "opponent": away,
            "result": result,
        })
        # Away history
        team_away_history[away].append({
            "points": away_points,
            "gs": row["away_score"],
            "gc": row["home_score"],
            "date": match_date,
            "opponent": home,
            "result": -result,  # From away perspective
        })
        # Overall history
        team_overall_history[home].append({
            "points": home_points,
            "gs": row["home_score"],
            "gc": row["away_score"],
            "date": match_date,
            "opponent": away,
            "result": result,
        })
        team_overall_history[away].append({
            "points": away_points,
            "gs": row["away_score"],
            "gc": row["home_score"],
            "date": match_date,
            "opponent": home,
            "result": -result,
        })

    return result_rows


def _days_between(date1: str, date2: str) -> int:
    """Calculate days between two ISO date strings."""
    try:
        d1 = datetime.fromisoformat(date1.replace("Z", "+00:00"))
        d2 = datetime.fromisoformat(date2.replace("Z", "+00:00"))
        return (d2 - d1).days
    except (ValueError, AttributeError):
        return 7  # Default to 7 days


def prepare_training_data(rows: list) -> Tuple[List[List[float]], List[int], List[str]]:
    """Prepare feature matrix X and target vector y for training.
    
    Returns:
        Tuple of (X, y, feature_names)
    """
    # Form-based features (zero when a team has no history yet)
    form_cols = [
        # Home-specific
        "home_form", "home_goals_scored_avg", "home_goals_conceded_avg",
        "home_matches_played",
        # Away-specific
        "away_form", "away_goals_scored_avg", "away_goals_conceded_avg",
        "away_matches_played",
        # Overall form
        "home_overall_form", "home_overall_goals_scored_avg", "home_overall_goals_conceded_avg",
        "away_overall_form", "away_overall_goals_scored_avg", "away_overall_goals_conceded_avg",
        # Head-to-head
        "h2h_matches", "h2h_home_wins", "h2h_draws", "h2h_away_wins",
        # Rest days
        "home_rest_days", "away_rest_days",
    ]
    # Elo features are appended last (default 1500, never "no history")
    elo_cols = ["home_elo", "away_elo", "elo_diff"]
    feature_cols = form_cols + elo_cols

    X = []
    y = []

    for row in rows:
        # Skip rows with missing features (first matches for each team)
        if any(row.get(col) is None for col in feature_cols):
            continue
        # Skip rows where all FORM features are zero (both teams have no history)
        if all(row.get(col, 0) == 0 for col in form_cols):
            continue

        X.append([row[col] for col in feature_cols])
        y.append(row["result"])

    return X, y, feature_cols


def prepare_prediction_features(home_team: str, away_team: str,
                                team_stats: dict) -> list:
    """Create feature vector for a single prediction."""
    home = team_stats.get(home_team, {})
    away = team_stats.get(away_team, {})

    home_elo = home.get("elo", ELO_START)
    away_elo = away.get("elo", ELO_START)

    return [
        home.get("form", 0),
        home.get("goals_scored_avg", 0),
        home.get("goals_conceded_avg", 0),
        home.get("matches_played", 0),
        away.get("form", 0),
        away.get("goals_scored_avg", 0),
        away.get("goals_conceded_avg", 0),
        away.get("matches_played", 0),
        home.get("overall_form", 0),
        home.get("overall_goals_scored_avg", 0),
        home.get("overall_goals_conceded_avg", 0),
        away.get("overall_form", 0),
        away.get("overall_goals_scored_avg", 0),
        away.get("overall_goals_conceded_avg", 0),
        home.get("h2h_matches", 0),
        home.get("h2h_home_wins", 0),
        home.get("h2h_draws", 0),
        home.get("h2h_away_wins", 0),
        home.get("rest_days", 7),
        away.get("rest_days", 7),
        # Elo features (must match feature_cols order in prepare_training_data)
        home_elo,
        away_elo,
        home_elo - away_elo,
    ]


def compute_team_stats(rows: list) -> dict:
    """Compute latest team statistics from processed match data."""
    if not rows:
        return {}
    
    team_stats = defaultdict(lambda: {
        "form": 0,
        "goals_scored_avg": 0,
        "goals_conceded_avg": 0,
        "matches_played": 0,
        "overall_form": 0,
        "overall_goals_scored_avg": 0,
        "overall_goals_conceded_avg": 0,
        "h2h_matches": 0,
        "h2h_home_wins": 0,
        "h2h_draws": 0,
        "h2h_away_wins": 0,
        "rest_days": 7,
        "elo": ELO_START,
    })

    # Get the most recent match for each team
    for row in reversed(rows):
        home = row["home_team"]
        away = row["away_team"]
        
        if team_stats[home]["matches_played"] == 0:
            team_stats[home] = {
                "form": row.get("home_form", 0),
                "goals_scored_avg": row.get("home_goals_scored_avg", 0),
                "goals_conceded_avg": row.get("home_goals_conceded_avg", 0),
                "matches_played": row.get("home_matches_played", 0),
                "overall_form": row.get("home_overall_form", 0),
                "overall_goals_scored_avg": row.get("home_overall_goals_scored_avg", 0),
                "overall_goals_conceded_avg": row.get("home_overall_goals_conceded_avg", 0),
                "h2h_matches": row.get("h2h_matches", 0),
                "h2h_home_wins": row.get("h2h_home_wins", 0),
                "h2h_draws": row.get("h2h_draws", 0),
                "h2h_away_wins": row.get("h2h_away_wins", 0),
                "rest_days": row.get("home_rest_days", 7),
                "elo": row.get("home_elo_post", ELO_START),
            }

        if team_stats[away]["matches_played"] == 0:
            team_stats[away] = {
                "form": row.get("away_form", 0),
                "goals_scored_avg": row.get("away_goals_scored_avg", 0),
                "goals_conceded_avg": row.get("away_goals_conceded_avg", 0),
                "matches_played": row.get("away_matches_played", 0),
                "overall_form": row.get("away_overall_form", 0),
                "overall_goals_scored_avg": row.get("away_overall_goals_scored_avg", 0),
                "overall_goals_conceded_avg": row.get("away_overall_goals_conceded_avg", 0),
                "h2h_matches": row.get("h2h_matches", 0),
                "h2h_home_wins": row.get("h2h_home_wins", 0),
                "h2h_draws": row.get("h2h_draws", 0),
                "h2h_away_wins": row.get("h2h_away_wins", 0),
                "rest_days": row.get("away_rest_days", 7),
                "elo": row.get("away_elo_post", ELO_START),
            }

        if all(team_stats[t]["matches_played"] > 0 for t in [home, away]):
            # Both teams have stats, we can stop
            pass
    
    return dict(team_stats)


def save_data(rows: list, path: str = "matches_data.json"):
    """Save processed data to JSON file."""
    with open(path, "w") as f:
        json.dump(rows, f, indent=2)


def load_data(path: str = "matches_data.json") -> list:
    """Load processed data from JSON file."""
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []
