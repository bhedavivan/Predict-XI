import csv
import json
from datetime import datetime
from collections import defaultdict


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


def add_form_features(rows: list, n_matches: int = 5) -> list:
    """Add rolling form features for each team."""
    if not rows:
        return rows

    team_history = defaultdict(list)
    result_rows = []

    for row in rows:
        home = row["home_team"]
        away = row["away_team"]
        result = row["result"]

        # Home team features
        home_recent = team_history[home][-n_matches:] if home in team_history else []
        home_form = sum(m["points"] for m in home_recent) / max(len(home_recent), 1)
        home_gs = sum(m["gs"] for m in home_recent)
        home_gc = sum(m["gc"] for m in home_recent)
        home_mp = len(home_recent)

        # Away team features
        away_recent = team_history[away][-n_matches:] if away in team_history else []
        away_form = sum(m["points"] for m in away_recent) / max(len(away_recent), 1)
        away_gs = sum(m["gs"] for m in away_recent)
        away_gc = sum(m["gc"] for m in away_recent)
        away_mp = len(away_recent)

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
        })
        result_rows.append(new_row)

        # Update team history
        home_points = 3 if result == 1 else (1 if result == 0 else 0)
        away_points = 3 if result == -1 else (1 if result == 0 else 0)

        team_history[home].append({
            "points": home_points,
            "gs": row["home_score"],
            "gc": row["away_score"],
        })
        team_history[away].append({
            "points": away_points,
            "gs": row["away_score"],
            "gc": row["home_score"],
        })

    return result_rows


def prepare_training_data(rows: list) -> tuple:
    """Prepare feature matrix X and target vector y for training."""
    feature_cols = [
        "home_form", "home_goals_scored_avg", "home_goals_conceded_avg",
        "home_matches_played",
        "away_form", "away_goals_scored_avg", "away_goals_conceded_avg",
        "away_matches_played",
    ]

    X = []
    y = []

    for row in rows:
        # Skip rows with missing features (first matches for each team)
        if any(row.get(col) is None for col in feature_cols):
            continue
        # Skip rows where all features are zero (no history)
        if all(row.get(col, 0) == 0 for col in feature_cols):
            continue

        X.append([row[col] for col in feature_cols])
        y.append(row["result"])

    return X, y


def prepare_prediction_features(home_team: str, away_team: str,
                                team_stats: dict) -> list:
    """Create feature vector for a single prediction."""
    home = team_stats.get(home_team, {})
    away = team_stats.get(away_team, {})

    return [
        home.get("form", 0),
        home.get("goals_scored_avg", 0),
        home.get("goals_conceded_avg", 0),
        home.get("matches_played", 0),
        away.get("form", 0),
        away.get("goals_scored_avg", 0),
        away.get("goals_conceded_avg", 0),
        away.get("matches_played", 0),
    ]


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