"""Unit tests for data_processor module."""

import pytest
from datetime import datetime, timedelta
from data_processor import (
    process_matches,
    add_form_features,
    prepare_training_data,
    prepare_prediction_features,
    compute_team_stats,
    _days_between,
    _elo_expected,
    _elo_goal_diff_multiplier,
    ELO_START,
    ELO_HOME_ADV,
)


def _match_row(date, home, away, home_score, away_score):
    if home_score > away_score:
        result = 1
    elif home_score < away_score:
        result = -1
    else:
        result = 0
    return {
        "date": date,
        "competition": "PL",
        "home_team": home,
        "away_team": away,
        "home_score": home_score,
        "away_score": away_score,
        "result": result,
        "total_goals": home_score + away_score,
        "goal_diff": home_score - away_score,
    }


class TestProcessMatches:
    """Tests for process_matches function."""

    def test_process_finished_matches(self):
        """Test processing of finished matches."""
        matches = [
            {
                "status": "FINISHED",
                "homeTeam": {"name": "Team A"},
                "awayTeam": {"name": "Team B"},
                "score": {"fullTime": {"home": 2, "away": 1}},
                "utcDate": "2023-08-12T14:00:00Z",
                "competition": {"code": "PL"},
            },
            {
                "status": "FINISHED",
                "homeTeam": {"name": "Team C"},
                "awayTeam": {"name": "Team D"},
                "score": {"fullTime": {"home": 0, "away": 0}},
                "utcDate": "2023-08-13T14:00:00Z",
                "competition": {"code": "PL"},
            },
        ]
        rows = process_matches(matches)
        assert len(rows) == 2
        assert rows[0]["result"] == 1  # Home win
        assert rows[1]["result"] == 0  # Draw

    def test_skip_unfinished_matches(self):
        """Test that unfinished matches are skipped."""
        matches = [
            {
                "status": "SCHEDULED",
                "homeTeam": {"name": "Team A"},
                "awayTeam": {"name": "Team B"},
                "score": {"fullTime": {"home": None, "away": None}},
                "utcDate": "2023-08-12T14:00:00Z",
                "competition": {"code": "PL"},
            },
        ]
        rows = process_matches(matches)
        assert len(rows) == 0

    def test_skip_matches_without_scores(self):
        """Test that matches without scores are skipped."""
        matches = [
            {
                "status": "FINISHED",
                "homeTeam": {"name": "Team A"},
                "awayTeam": {"name": "Team B"},
                "score": {"fullTime": {"home": None, "away": None}},
                "utcDate": "2023-08-12T14:00:00Z",
                "competition": {"code": "PL"},
            },
        ]
        rows = process_matches(matches)
        assert len(rows) == 0

    def test_sort_by_date(self):
        """Test that rows are sorted by date."""
        matches = [
            {
                "status": "FINISHED",
                "homeTeam": {"name": "Team A"},
                "awayTeam": {"name": "Team B"},
                "score": {"fullTime": {"home": 1, "away": 0}},
                "utcDate": "2023-08-13T14:00:00Z",
                "competition": {"code": "PL"},
            },
            {
                "status": "FINISHED",
                "homeTeam": {"name": "Team C"},
                "awayTeam": {"name": "Team D"},
                "score": {"fullTime": {"home": 2, "away": 1}},
                "utcDate": "2023-08-12T14:00:00Z",
                "competition": {"code": "PL"},
            },
        ]
        rows = process_matches(matches)
        assert rows[0]["date"] < rows[1]["date"]


class TestDaysBetween:
    """Tests for _days_between helper function."""

    def test_same_day(self):
        """Test days between same day."""
        assert _days_between("2023-08-12T14:00:00Z", "2023-08-12T14:00:00Z") == 0

    def test_one_day(self):
        """Test days between consecutive days."""
        assert _days_between("2023-08-12T14:00:00Z", "2023-08-13T14:00:00Z") == 1

    def test_invalid_dates(self):
        """Test handling of invalid dates."""
        assert _days_between("invalid", "2023-08-13T14:00:00Z") == 7
        assert _days_between("2023-08-12T14:00:00Z", "invalid") == 7


class TestAddFormFeatures:
    """Tests for add_form_features function."""

    def test_add_form_features_basic(self):
        """Test basic form feature addition."""
        rows = [
            {
                "date": "2023-08-12T14:00:00Z",
                "home_team": "Team A",
                "away_team": "Team B",
                "home_score": 2,
                "away_score": 1,
                "result": 1,
                "total_goals": 3,
                "goal_diff": 1,
            },
            {
                "date": "2023-08-19T14:00:00Z",
                "home_team": "Team A",
                "away_team": "Team C",
                "home_score": 1,
                "away_score": 1,
                "result": 0,
                "total_goals": 2,
                "goal_diff": 0,
            },
        ]
        result = add_form_features(rows)
        assert len(result) == 2
        # First match should have no history
        assert result[0]["home_form"] == 0
        assert result[0]["away_form"] == 0
        # Second match should have history for Team A
        assert result[1]["home_form"] > 0

    def test_empty_rows(self):
        """Test with empty rows."""
        result = add_form_features([])
        assert result == []


class TestElo:
    """Tests for the Elo rating system: no hand-picked club/league bonuses,
    every team starts from the same anchor, and margin of victory scales
    the update."""

    def test_new_teams_start_at_elo_start(self):
        rows = [_match_row("2023-08-12T14:00:00Z", "Team A", "Team B", 1, 1)]
        result = add_form_features(rows)
        # A draw between two brand-new teams should barely move Elo away
        # from the shared 1500 anchor (home advantage still applies).
        assert result[0]["home_elo"] == ELO_START
        assert result[0]["away_elo"] == ELO_START

    def test_no_club_or_league_priors(self):
        """Historically Elo-heavy clubs (Real Madrid, Bayern, PSG, ...) must
        get no baked-in bonus — only match results move their rating."""
        rows = [_match_row("2023-08-12T14:00:00Z", "Real Madrid", "Paris SG", 0, 0)]
        result = add_form_features(rows)
        assert result[0]["home_elo"] == ELO_START
        assert result[0]["away_elo"] == ELO_START

    def test_home_win_raises_home_elo_lowers_away_elo(self):
        rows = [
            _match_row("2023-08-12T14:00:00Z", "Team A", "Team B", 2, 0),
            _match_row("2023-08-19T14:00:00Z", "Team A", "Team C", 1, 0),
        ]
        result = add_form_features(rows)
        assert result[1]["home_elo"] > ELO_START
        # Team B lost, so its post-match Elo should have dropped below start
        first = result[0]
        assert first["home_elo_post"] > ELO_START
        assert first["away_elo_post"] < ELO_START

    def test_larger_margin_moves_elo_further(self):
        """A 4-0 win should move the winner's Elo further than a 1-0 win,
        via the margin-of-victory multiplier."""
        small_margin = [_match_row("2023-08-12T14:00:00Z", "Team A", "Team B", 1, 0)]
        big_margin = [_match_row("2023-08-12T14:00:00Z", "Team A", "Team B", 4, 0)]
        small_result = add_form_features(small_margin)
        big_result = add_form_features(big_margin)
        small_gain = small_result[0]["home_elo_post"] - ELO_START
        big_gain = big_result[0]["home_elo_post"] - ELO_START
        assert big_gain > small_gain > 0

    def test_goal_diff_multiplier_monotonic(self):
        assert _elo_goal_diff_multiplier(0) == 1.0
        assert _elo_goal_diff_multiplier(1) == 1.0
        assert _elo_goal_diff_multiplier(2) == 1.5
        assert _elo_goal_diff_multiplier(3) > _elo_goal_diff_multiplier(2)
        assert _elo_goal_diff_multiplier(5) > _elo_goal_diff_multiplier(3)

    def test_elo_expected_home_advantage(self):
        """Equal-rated teams should favor the home side because of
        ELO_HOME_ADV."""
        p_home = _elo_expected(ELO_START, ELO_START)
        assert p_home > 0.5

    def test_season_gap_regresses_toward_start(self):
        """A team idle for more than ELO_SEASON_GAP days should regress
        partway back toward the shared 1500 anchor, not a league-specific
        one."""
        rows = [
            _match_row("2023-08-12T14:00:00Z", "Team A", "Team B", 3, 0),
            # Team A returns 200 days later against a new opponent.
            _match_row("2024-03-01T14:00:00Z", "Team A", "Team C", 0, 0),
        ]
        result = add_form_features(rows)
        elo_after_win = result[0]["home_elo_post"]
        elo_before_second_match = result[1]["home_elo"]
        assert ELO_START < elo_before_second_match < elo_after_win


class TestPrepareTrainingData:
    """Tests for prepare_training_data function."""

    def test_prepare_training_data(self):
        """Test preparing training data."""
        rows = [
            {
                "home_form": 1.0,
                "home_goals_scored_avg": 2.0,
                "home_goals_conceded_avg": 1.0,
                "home_matches_played": 5,
                "away_form": 0.5,
                "away_goals_scored_avg": 1.0,
                "away_goals_conceded_avg": 1.5,
                "away_matches_played": 5,
                "home_overall_form": 1.0,
                "home_overall_goals_scored_avg": 2.0,
                "home_overall_goals_conceded_avg": 1.0,
                "away_overall_form": 0.5,
                "away_overall_goals_scored_avg": 1.0,
                "away_overall_goals_conceded_avg": 1.5,
                "h2h_matches": 2,
                "h2h_home_wins": 1,
                "h2h_draws": 1,
                "h2h_away_wins": 0,
                "home_rest_days": 7,
                "away_rest_days": 7,
                "home_elo": 1520.0,
                "away_elo": 1480.0,
                "elo_diff": 40.0,
                "result": 1,
            },
        ]
        X, y, feature_names = prepare_training_data(rows)
        assert len(X) == 1
        assert len(y) == 1
        assert y[0] == 1
        assert len(feature_names) == 52  # 20 base + 29 advanced + 3 Elo


class TestPreparePredictionFeatures:
    """Tests for prepare_prediction_features function."""

    def test_prepare_prediction_features(self):
        """Test preparing prediction features."""
        team_stats = {
            "Team A": {
                "form": 1.0,
                "goals_scored_avg": 2.0,
                "goals_conceded_avg": 1.0,
                "matches_played": 5,
                "overall_form": 1.0,
                "overall_goals_scored_avg": 2.0,
                "overall_goals_conceded_avg": 1.0,
                "h2h_matches": 2,
                "h2h_home_wins": 1,
                "h2h_draws": 1,
                "h2h_away_wins": 0,
                "rest_days": 7,
            },
            "Team B": {
                "form": 0.5,
                "goals_scored_avg": 1.0,
                "goals_conceded_avg": 1.5,
                "matches_played": 5,
                "overall_form": 0.5,
                "overall_goals_scored_avg": 1.0,
                "overall_goals_conceded_avg": 1.5,
                "h2h_matches": 2,
                "h2h_home_wins": 1,
                "h2h_draws": 1,
                "h2h_away_wins": 0,
                "rest_days": 7,
            },
        }
        features = prepare_prediction_features("Team A", "Team B", team_stats)
        assert len(features) == 52  # 20 base + 29 advanced + 3 Elo
        assert features[0] == 1.0  # home form
        assert features[4] == 0.5  # away form


class TestComputeTeamStats:
    """Tests for compute_team_stats function."""

    def test_compute_team_stats(self):
        """Test computing team stats."""
        rows = [
            {
                "date": "2023-08-19T14:00:00Z",
                "home_team": "Team A",
                "away_team": "Team B",
                "home_form": 1.0,
                "home_goals_scored_avg": 2.0,
                "home_goals_conceded_avg": 1.0,
                "home_matches_played": 5,
                "away_form": 0.5,
                "away_goals_scored_avg": 1.0,
                "away_goals_conceded_avg": 1.5,
                "away_matches_played": 5,
                "home_overall_form": 1.0,
                "home_overall_goals_scored_avg": 2.0,
                "home_overall_goals_conceded_avg": 1.0,
                "away_overall_form": 0.5,
                "away_overall_goals_scored_avg": 1.0,
                "away_overall_goals_conceded_avg": 1.5,
                "h2h_matches": 2,
                "h2h_home_wins": 1,
                "h2h_draws": 1,
                "h2h_away_wins": 0,
                "home_rest_days": 7,
                "away_rest_days": 7,
            },
        ]
        stats = compute_team_stats(rows)
        assert "Team A" in stats
        assert "Team B" in stats
        assert stats["Team A"]["form"] == 1.0
        assert stats["Team B"]["form"] == 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])