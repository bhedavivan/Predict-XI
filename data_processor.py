import csv
import json
import math
from datetime import datetime, timedelta
from collections import defaultdict
from typing import List, Dict, Any, Tuple, Optional

import numpy as np

import dixon_coles


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
            # Optional richer per-match stats (shots/corners/cards/fouls) —
            # only present for matches sourced from football-data.co.uk.
            "stats": match.get("stats") or {},
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
# ELO_K and ELO_SEASON_REGRESS were tuned against downstream purged-CV
# performance (see rating_constants_tuning.json), not hand-picked. The
# previous values (24.0 / 0.30) were too conservative: ratings adapted too
# slowly and were pulled too far back to the mean between seasons. Moving to
# 40.0 / 0.10 gained +0.017 macro-F1 and +0.011 accuracy on the probe.
# K=55 was also tested and scored WORSE than 40, so this is a real optimum
# rather than "faster is always better".
ELO_K = 40.0
ELO_HOME_ADV = 65.0
ELO_SEASON_GAP = 45
ELO_SEASON_REGRESS = 0.10

# Matches a competition must accumulate before its own measured goal
# baselines replace the global defaults. Below this the sample is too thin to
# beat the prior, so the global constants stay in use.
LEAGUE_BASELINE_MIN_MATCHES = 200


def _elo_expected(home_elo: float, away_elo: float, home_adv: float = None) -> float:
    adv = ELO_HOME_ADV if home_adv is None else home_adv
    return 1.0 / (1.0 + 10 ** ((away_elo - (home_elo + adv)) / 400.0))


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


# Squad market value is heavily right-skewed (a €1.4bn Real Madrid against a
# €15m promoted side), so raw euros would let a handful of superclubs dominate
# any distance-based split. Log-space keeps the ratio meaningful across the
# whole range.
_SQUAD_VALUE_SCALE = 1_000_000.0  # work in EUR millions


def _log_value(value: Optional[float]) -> float:
    if not value or value <= 0:
        return 0.0
    return math.log1p(value / _SQUAD_VALUE_SCALE)


def squad_value_feature_dict(home_value: Optional[float], away_value: Optional[float]) -> dict:
    """Build the squad-value feature block from two raw EUR values.

    Shared by the training path (add_form_features) and the prediction path
    (prepare_prediction_features) so the two cannot silently disagree — the
    same reason the evenness features are covered by an alignment test.

    `has_squad_value` is 0 unless BOTH sides are known: a comparison where
    only one value exists is worse than none, since the model would read the
    missing side as an infinitely weak squad.
    """
    both = bool(home_value) and bool(away_value)
    h, a = _log_value(home_value), _log_value(away_value)
    return {
        "home_squad_value": h if both else 0.0,
        "away_squad_value": a if both else 0.0,
        "squad_value_diff": (h - a) if both else 0.0,
        "abs_squad_value_diff": abs(h - a) if both else 0.0,
        "has_squad_value": 1.0 if both else 0.0,
    }


def _squad_value_features(home: str, away: str, match_date: str,
                           squad_values, club_map: Optional[dict]) -> dict:
    if squad_values is None or not club_map:
        return squad_value_feature_dict(None, None)
    hv = squad_values.value_at(club_map.get(home, ""), match_date)
    av = squad_values.value_at(club_map.get(away, ""), match_date)
    return squad_value_feature_dict(hv, av)


def add_form_features(rows: list, n_matches: int = 5,
                       elo_k: float = None, elo_home_adv: float = None,
                       elo_season_regress: float = None,
                       dc_k: float = None, dc_rho: float = None,
                       squad_values=None, club_map: dict = None) -> list:
    """Add rolling form features for each team (home/away specific), plus Elo.

    The rating constants are overridable so they can be tuned against actual
    predictive performance instead of staying hand-picked. They generate the
    model's most important features (elo_diff and the dc_* family), so their
    values matter as much as any classifier hyperparameter. Defaults are the
    module-level constants, keeping existing callers unchanged.

    `squad_values` (a transfermarkt_data.SquadValueIndex) and `club_map`
    ({our_team_name: transfermarkt_club_id}) are optional. When supplied,
    each match gets its two clubs' squad market value AS OF that match date.
    When absent — or for the ~11 leagues and 33 clubs Transfermarkt doesn't
    cover — the features fall back to 0 alongside a `has_squad_value` flag,
    so the model can learn to disregard them rather than treat 0 as "worthless
    squad".
    """
    if not rows:
        return rows

    elo_k = ELO_K if elo_k is None else elo_k
    elo_home_adv = ELO_HOME_ADV if elo_home_adv is None else elo_home_adv
    elo_season_regress = ELO_SEASON_REGRESS if elo_season_regress is None else elo_season_regress
    dc_k = dixon_coles.DC_K if dc_k is None else dc_k
    dc_rho = dixon_coles.DC_RHO if dc_rho is None else dc_rho

    team_home_history = defaultdict(list)
    team_away_history = defaultdict(list)
    team_overall_history = defaultdict(list)
    team_elo = {}
    dc_ratings = dixon_coles.DixonColesRatings(k=dc_k, rho=dc_rho)
    # Per-league goal baselines, accumulated as matches stream past rather
    # than precomputed over the whole file — using a full-dataset average
    # would leak later results into the features of earlier matches. Same
    # no-lookahead discipline as Elo and the Dixon-Coles ratings themselves.
    # [n_matches, home_goals, away_goals] per competition.
    league_goals = defaultdict(lambda: [0, 0.0, 0.0])
    team_season_matches = defaultdict(int)
    result_rows = []

    # Maintain team-pair histories for H2H without rescanning all prior matches
    team_pair_h2h = defaultdict(list)

    for row in rows:
        home = row["home_team"]
        away = row["away_team"]
        result = row["result"]
        match_date = row["date"]
        stats = row.get("stats") or {}

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

        # Match-stat rolling averages (shots/shots-on-target/corners/cards).
        # Only populated for matches sourced from football-data.co.uk; older
        # footballcsv-only rows leave these at 0 in team_overall_history, so
        # this degrades gracefully rather than crashing.
        def _stat_avg(history, key):
            recent = history[-5:]
            return sum(m.get(key, 0) for m in recent) / max(len(recent), 1)

        home_hist = team_overall_history[home]
        away_hist = team_overall_history[away]
        stat_features = {}
        for side, hist in (("home", home_hist), ("away", away_hist)):
            for label, key in (
                ("shots", "shots_for"), ("shots_against", "shots_against"),
                ("sot", "sot_for"), ("sot_against", "sot_against"),
                ("corners", "corners_for"), ("corners_against", "corners_against"),
                ("cards", "cards_for"), ("cards_against", "cards_against"),
            ):
                stat_features[f"{side}_{label}_avg"] = _stat_avg(hist, key)

        # League-specific goal baselines from matches seen SO FAR (never the
        # whole dataset — that would be lookahead). Falls back to the global
        # constants until the competition has enough history to beat them.
        comp = row.get("competition", "")
        lg_n, lg_hg, lg_ag = league_goals[comp]
        if lg_n >= LEAGUE_BASELINE_MIN_MATCHES:
            base_home_goals = lg_hg / lg_n
            base_away_goals = lg_ag / lg_n
        else:
            base_home_goals = dixon_coles.LEAGUE_AVG_HOME_GOALS
            base_away_goals = dixon_coles.LEAGUE_AVG_AWAY_GOALS

        # Elo
        for team, last_date in ((home, home_last_date), (away, away_last_date)):
            if team not in team_elo:
                team_elo[team] = ELO_START
            elif last_date and _days_between(last_date, match_date) > ELO_SEASON_GAP:
                team_elo[team] = ELO_START + (team_elo[team] - ELO_START) * (1 - elo_season_regress)
        home_earned = team_elo[home]
        away_earned = team_elo[away]
        home_elo = home_earned
        away_elo = away_earned

        # Dixon-Coles attack/defense derived probabilities (pre-match ratings
        # only — no lookahead, same discipline as Elo above).
        dc_p_home, dc_p_draw, dc_p_away, dc_exp_home, dc_exp_away = dc_ratings.predict(
            home, away, base_home_goals, base_away_goals)

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
            "dc_home_exp_goals": dc_exp_home,
            "dc_away_exp_goals": dc_exp_away,
            "dc_home_prob": dc_p_home,
            "dc_draw_prob": dc_p_draw,
            "dc_away_prob": dc_p_away,
            # "Evenness" features. A draw is the outcome of two closely
            # matched sides, but every strength feature above is *signed*
            # (elo_diff, goal-difference gaps), so "evenly matched" lives near
            # zero and a tree needs two splits to isolate it. These state
            # closeness directly, aimed at the model's weakest class
            # (draw recall was 24.6% at v6).
            "abs_elo_diff": abs(home_elo - away_elo),
            "abs_dc_exp_goals_diff": abs(dc_exp_home - dc_exp_away),
            "abs_form_diff": abs(home_overall_form - away_overall_form),
            "abs_ppf_10_diff": abs(home_ppf_10 - away_ppf_10),
            # Carried so compute_team_stats can hand the same baselines to
            # prepare_prediction_features — training and serving must build
            # dc_* from identical assumptions or the vectors silently skew.
            "league_base_home_goals": base_home_goals,
            "league_base_away_goals": base_away_goals,
        })
        new_row.update(stat_features)
        new_row.update(_squad_value_features(home, away, match_date, squad_values, club_map))
        result_rows.append(new_row)

        # Fold this match into its league's running goal baseline. Done only
        # AFTER its features were computed above, so a match never
        # contributes to the baseline used to predict itself.
        league_goals[comp][0] += 1
        league_goals[comp][1] += row["home_score"]
        league_goals[comp][2] += row["away_score"]

        # Update Dixon-Coles ratings from the real result, then persist the
        # post-match attack/defense so compute_team_stats can carry it
        # forward for predicting brand-new (untrained) matchups.
        dc_ratings.update(home, away, row["home_score"], row["away_score"],
                           base_home_goals, base_away_goals)
        new_row["home_dc_attack_post"] = dc_ratings.attack[home]
        new_row["home_dc_defense_post"] = dc_ratings.defense[home]
        new_row["away_dc_attack_post"] = dc_ratings.attack[away]
        new_row["away_dc_defense_post"] = dc_ratings.defense[away]

        # Update Elo (margin-of-victory scaled)
        league_home_adv = elo_home_adv * (base_home_goals / max(dixon_coles.LEAGUE_AVG_HOME_GOALS, 1e-6))
        exp_home = _elo_expected(home_earned, away_earned, league_home_adv)
        actual_home = 1.0 if result == 1 else (0.5 if result == 0 else 0.0)
        mov = _elo_goal_diff_multiplier(row["goal_diff"])
        elo_delta = elo_k * mov * (actual_home - exp_home)
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
            "shots_for": stats.get("home_shots", 0), "shots_against": stats.get("away_shots", 0),
            "sot_for": stats.get("home_shots_on_target", 0), "sot_against": stats.get("away_shots_on_target", 0),
            "corners_for": stats.get("home_corners", 0), "corners_against": stats.get("away_corners", 0),
            "cards_for": stats.get("home_yellow", 0) + stats.get("home_red", 0),
            "cards_against": stats.get("away_yellow", 0) + stats.get("away_red", 0),
        })
        team_overall_history[away].append({
            "points": away_points, "gs": row["away_score"], "gc": row["home_score"],
            "gd": -row["goal_diff"], "date": match_date, "opponent": home, "result": -result,
            "shots_for": stats.get("away_shots", 0), "shots_against": stats.get("home_shots", 0),
            "sot_for": stats.get("away_shots_on_target", 0), "sot_against": stats.get("home_shots_on_target", 0),
            "corners_for": stats.get("away_corners", 0), "corners_against": stats.get("home_corners", 0),
            "cards_for": stats.get("away_yellow", 0) + stats.get("away_red", 0),
            "cards_against": stats.get("home_yellow", 0) + stats.get("home_red", 0),
        })

        # Update H2H pair history
        team_pair_h2h[pair_key].append({
            "home_win": 1 if result == 1 else 0,
            "draw": 1 if result == 0 else 0,
            "away_win": 1 if result == -1 else 0,
        })

    # Expose the final pair histories so prediction can reuse them. Without
    # this, H2H is computed during training but fed as 0 for every live
    # prediction — the model learns a weight for a feature it then never
    # receives, which is a train/serve skew rather than a missing nicety.
    add_form_features.last_h2h = {
        f"{a}|{b}": list(v[-5:]) for (a, b), v in team_pair_h2h.items() if v
    }

    return result_rows


def h2h_pair_key(home: str, away: str) -> str:
    """Pair key for H2H lookups. Sorted, so the same two clubs map to one
    entry regardless of which is at home — matching how add_form_features
    accumulates them."""
    a, b = sorted([home or "", away or ""])
    return f"{a}|{b}"


def h2h_features_for(home: str, away: str, h2h_map: Optional[dict]) -> list:
    """[matches, home_wins, draws, away_wins] over the last 5 meetings.

    Semantics deliberately mirror training: `home_win` counts meetings won by
    whoever was at home in THAT match, not by the current home side. Keeping
    the definition identical matters more than making it more intuitive —
    training and serving must agree.
    """
    if not h2h_map:
        return [0, 0, 0, 0]
    recent = h2h_map.get(h2h_pair_key(home, away)) or []
    return [
        len(recent),
        sum(1 for m in recent if m.get("home_win") == 1),
        sum(1 for m in recent if m.get("draw") == 1),
        sum(1 for m in recent if m.get("away_win") == 1),
    ]


def prepare_training_data(rows: list, return_meta: bool = False):
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
    # Shots/shots-on-target/corners/cards rolling averages — 0 for rows
    # sourced from leagues/seasons without football-data.co.uk coverage.
    match_stat_cols = [
        "home_shots_avg", "home_shots_against_avg", "home_sot_avg", "home_sot_against_avg",
        "home_corners_avg", "home_corners_against_avg", "home_cards_avg", "home_cards_against_avg",
        "away_shots_avg", "away_shots_against_avg", "away_sot_avg", "away_sot_against_avg",
        "away_corners_avg", "away_corners_against_avg", "away_cards_avg", "away_cards_against_avg",
    ]
    elo_cols = ["home_elo", "away_elo", "elo_diff"]
    dc_cols = ["dc_home_exp_goals", "dc_away_exp_goals", "dc_home_prob", "dc_draw_prob", "dc_away_prob"]
    # Closeness features — see add_form_features; targeted at draw recall.
    evenness_cols = ["abs_elo_diff", "abs_dc_exp_goals_diff", "abs_form_diff", "abs_ppf_10_diff"]
    # Squad market value (Transfermarkt). 0 + has_squad_value=0 where the
    # league or club isn't covered — see squad_value_feature_dict.
    squad_cols = ["home_squad_value", "away_squad_value", "squad_value_diff",
                  "abs_squad_value_diff", "has_squad_value"]
    feature_cols = (base_form_cols + advanced_form_cols + match_stat_cols
                    + elo_cols + dc_cols + evenness_cols + squad_cols)

    X = []
    y = []
    meta = []
    for row in rows:
        if any(row.get(col) is None for col in base_form_cols + elo_cols):
            continue
        if all(row.get(col, 0) == 0 for col in base_form_cols):
            continue
        X.append([row.get(col, 0) for col in feature_cols])
        y.append(row["result"])
        # Kept parallel to X so downstream evaluation can slice results by
        # competition. Rows are filtered above, so leagues must be collected
        # here rather than reconstructed later.
        meta.append({"league": row.get("competition", ""), "date": row.get("date", "")})

    if return_meta:
        return X, y, feature_cols, meta
    return X, y, feature_cols


def prepare_prediction_features(home_team: str, away_team: str,
                                team_stats: dict, h2h_map: Optional[dict] = None) -> list:
    """Build the feature vector for one hypothetical match.

    `h2h_map` supplies head-to-head history for this specific PAIR. It is a
    separate argument because H2H is a property of two clubs together, not of
    either one — storing it on a team record (as was previously attempted)
    yields whatever that team's most recent unrelated fixture happened to
    show, which is why it silently read 0 everywhere.
    """
    home = team_stats.get(home_team, {})
    away = team_stats.get(away_team, {})
    h2h = h2h_features_for(home_team, away_team, h2h_map)
    home_elo = home.get("elo", ELO_START)
    away_elo = away.get("elo", ELO_START)

    base_features = [
        home.get("form", 0), home.get("goals_scored_avg", 0), home.get("goals_conceded_avg", 0),
        home.get("matches_played", 0),
        away.get("form", 0), away.get("goals_scored_avg", 0), away.get("goals_conceded_avg", 0),
        away.get("matches_played", 0),
        home.get("overall_form", 0), home.get("overall_goals_scored_avg", 0), home.get("overall_goals_conceded_avg", 0),
        away.get("overall_form", 0), away.get("overall_goals_scored_avg", 0), away.get("overall_goals_conceded_avg", 0),
        h2h[0], h2h[1], h2h[2], h2h[3],
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
    match_stat_features = [
        home.get("shots_avg", 0), home.get("shots_against_avg", 0),
        home.get("sot_avg", 0), home.get("sot_against_avg", 0),
        home.get("corners_avg", 0), home.get("corners_against_avg", 0),
        home.get("cards_avg", 0), home.get("cards_against_avg", 0),
        away.get("shots_avg", 0), away.get("shots_against_avg", 0),
        away.get("sot_avg", 0), away.get("sot_against_avg", 0),
        away.get("corners_avg", 0), away.get("corners_against_avg", 0),
        away.get("cards_avg", 0), away.get("cards_against_avg", 0),
    ]
    elo_features = [home_elo, away_elo, home_elo - away_elo]

    # Use the HOME team's league baselines — the match is played in their
    # competition, and these are the same values training used for that
    # league (carried through compute_team_stats). Falls back to the globals
    # for teams predating this field or from an unmeasured league.
    dc_p_home, dc_p_draw, dc_p_away, dc_exp_home, dc_exp_away = dixon_coles.match_probabilities(
        home.get("dc_attack", dixon_coles.DC_START), home.get("dc_defense", dixon_coles.DC_START),
        away.get("dc_attack", dixon_coles.DC_START), away.get("dc_defense", dixon_coles.DC_START),
        home.get("league_base_home_goals", dixon_coles.LEAGUE_AVG_HOME_GOALS),
        home.get("league_base_away_goals", dixon_coles.LEAGUE_AVG_AWAY_GOALS),
    )
    dc_features = [dc_exp_home, dc_exp_away, dc_p_home, dc_p_draw, dc_p_away]

    # Must mirror evenness_cols in prepare_training_data, same order.
    evenness_features = [
        abs(home_elo - away_elo),
        abs(dc_exp_home - dc_exp_away),
        abs(home.get("overall_form", 0) - away.get("overall_form", 0)),
        abs(home.get("ppf_10", 0) - away.get("ppf_10", 0)),
    ]

    # Squad value uses each team's CURRENT value (carried onto team_stats by
    # compute_team_stats), which is what lets new signings move a prediction:
    # a club that just bought five players has a higher current squad value,
    # and the model already learned from history what that is worth. Order
    # must match squad_cols in prepare_training_data — the alignment test
    # covers this.
    squad_block = squad_value_feature_dict(
        home.get("squad_value_eur"), away.get("squad_value_eur"))
    squad_features = [squad_block["home_squad_value"], squad_block["away_squad_value"],
                      squad_block["squad_value_diff"], squad_block["abs_squad_value_diff"],
                      squad_block["has_squad_value"]]

    return (base_features + match_stat_features + elo_features
            + dc_features + evenness_features + squad_features)


def compute_team_stats(rows: list, squad_values=None, club_map: dict = None) -> dict:
    """Latest per-team state for predictions.

    `squad_values`/`club_map` attach each team's CURRENT squad market value
    (not the point-in-time value used for training rows). That is what makes
    a transfer window visible to the UI: sign five players and the club's
    current value rises, so the next prediction differs — for a weight the
    model learned from history rather than a hand-set adjustment.
    """
    if not rows:
        return {}

    team_stats = defaultdict(lambda: {
        "form": 0, "goals_scored_avg": 0, "goals_conceded_avg": 0, "matches_played": 0,
        "overall_form": 0, "overall_goals_scored_avg": 0, "overall_goals_conceded_avg": 0,
        "h2h_matches": 0, "h2h_home_wins": 0, "h2h_draws": 0, "h2h_away_wins": 0,
        "rest_days": 7, "elo": ELO_START, "league": "", "squad_value_eur": None,
        "league_base_home_goals": dixon_coles.LEAGUE_AVG_HOME_GOALS,
        "league_base_away_goals": dixon_coles.LEAGUE_AVG_AWAY_GOALS,
        "ppf_10": 0, "ppf_20": 0, "ewma_form": 0, "ewma_gs": 0, "ewma_gc": 0,
        "streak": 0, "cs_rate_5": 0, "cs_rate_10": 0, "btts_rate_5": 0, "o25_rate_5": 0,
        "home_ppf_10": 0, "away_ppf_10": 0, "sos": ELO_START,
        "gd_5": 0, "gd_10": 0, "season_progress": 0,
        "shots_avg": 0, "shots_against_avg": 0, "sot_avg": 0, "sot_against_avg": 0,
        "corners_avg": 0, "corners_against_avg": 0, "cards_avg": 0, "cards_against_avg": 0,
        "dc_attack": dixon_coles.DC_START, "dc_defense": dixon_coles.DC_START,
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
        "shots_avg": "shots_avg", "shots_against_avg": "shots_against_avg",
        "sot_avg": "sot_avg", "sot_against_avg": "sot_against_avg",
        "corners_avg": "corners_avg", "corners_against_avg": "corners_against_avg",
        "cards_avg": "cards_avg", "cards_against_avg": "cards_against_avg",
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
            team_stats[home]["dc_attack"] = row.get("home_dc_attack_post", dixon_coles.DC_START)
            team_stats[home]["dc_defense"] = row.get("home_dc_defense_post", dixon_coles.DC_START)
            team_stats[home]["league"] = row.get("competition", "")
            team_stats[home]["league_base_home_goals"] = row.get(
                "league_base_home_goals", dixon_coles.LEAGUE_AVG_HOME_GOALS)
            team_stats[home]["league_base_away_goals"] = row.get(
                "league_base_away_goals", dixon_coles.LEAGUE_AVG_AWAY_GOALS)
        if team_stats[away]["matches_played"] == 0:
            for key, base in prefix_map.items():
                row_key = f"away_{base}"
                if row_key in row:
                    team_stats[away][key] = row[row_key]
            team_stats[away]["elo"] = row.get("away_elo_post", ELO_START)
            team_stats[away]["dc_attack"] = row.get("away_dc_attack_post", dixon_coles.DC_START)
            team_stats[away]["dc_defense"] = row.get("away_dc_defense_post", dixon_coles.DC_START)
            team_stats[away]["league"] = row.get("competition", "")
            team_stats[away]["league_base_home_goals"] = row.get(
                "league_base_home_goals", dixon_coles.LEAGUE_AVG_HOME_GOALS)
            team_stats[away]["league_base_away_goals"] = row.get(
                "league_base_away_goals", dixon_coles.LEAGUE_AVG_AWAY_GOALS)
        if all(team_stats[t]["matches_played"] > 0 for t in [home, away]):
            pass

    out = dict(team_stats)
    if squad_values is not None and club_map:
        for team, info in out.items():
            club_id = club_map.get(team)
            if club_id:
                info["squad_value_eur"] = squad_values.current_value(club_id)
    return out


def save_data(rows: list, path: str = "matches_data.json"):
    with open(path, "w") as f:
        json.dump(rows, f, indent=2)


def load_data(path: str = "matches_data.json") -> list:
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []
