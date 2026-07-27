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
    LEAGUE_BASELINE_MIN_MATCHES,
)
import dixon_coles


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
        assert len(feature_names) == 82  # 20 base + 29 advanced + 16 match-stat + 3 Elo + 5 Dixon-Coles + 4 evenness + 5 squad-value


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
        assert len(features) == 82  # 20 base + 29 advanced + 16 match-stat + 3 Elo + 5 Dixon-Coles + 4 evenness + 5 squad-value
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


class TestLeagueBaselinesNoLookahead:
    """Per-league goal baselines are accumulated as matches stream past. If
    they were ever computed over the whole dataset, every early match would
    carry information from results that hadn't happened yet — inflating
    offline scores while doing nothing for real predictions."""

    def test_early_matches_use_global_defaults(self):
        """Before a league reaches LEAGUE_BASELINE_MIN_MATCHES there isn't
        enough history, so the global constants must still be in force."""
        rows = [_match_row(f"2023-08-{d:02d}T14:00:00Z", f"T{d}", f"U{d}", 5, 0)
                for d in range(1, 11)]
        result = add_form_features(rows)
        for r in result:
            assert r["league_base_home_goals"] == dixon_coles.LEAGUE_AVG_HOME_GOALS
            assert r["league_base_away_goals"] == dixon_coles.LEAGUE_AVG_AWAY_GOALS

    def test_baseline_ignores_the_match_it_describes(self):
        """A match must never contribute to the baseline used to predict it.
        Feeding a long run of identical lopsided scores, the baseline seen by
        match N must reflect only matches 1..N-1."""
        n = LEAGUE_BASELINE_MIN_MATCHES + 50
        rows = [_match_row("2023-08-12T14:00:00Z", f"H{i}", f"A{i}", 3, 0) for i in range(n)]
        result = add_form_features(rows)
        # Once past the threshold the measured baseline should converge
        # toward the actual 3-0 pattern, but never reach it instantly at the
        # boundary — it only sees prior matches.
        at_threshold = result[LEAGUE_BASELINE_MIN_MATCHES]
        assert at_threshold["league_base_home_goals"] == pytest.approx(3.0, abs=0.01)
        assert at_threshold["league_base_away_goals"] == pytest.approx(0.0, abs=0.01)
        # The match just before the threshold still had too little history.
        just_before = result[LEAGUE_BASELINE_MIN_MATCHES - 1]
        assert just_before["league_base_home_goals"] == dixon_coles.LEAGUE_AVG_HOME_GOALS


class TestSquadValueFeatures:
    """Squad market value is what makes transfer activity visible to the
    model: sign five players, current squad value rises, prediction shifts —
    with a weight learned from history rather than a hand-set adjustment."""

    def test_signing_players_raises_squad_features(self):
        from data_processor import squad_value_feature_dict
        before = squad_value_feature_dict(500_000_000, 500_000_000)
        after = squad_value_feature_dict(900_000_000, 500_000_000)
        assert after["home_squad_value"] > before["home_squad_value"]
        assert after["squad_value_diff"] > before["squad_value_diff"]

    def test_missing_value_on_either_side_disables_the_block(self):
        """A one-sided comparison is worse than none — the model would read
        the missing side as an infinitely weak squad."""
        from data_processor import squad_value_feature_dict
        for h, a in ((500_000_000, None), (None, 500_000_000), (None, None)):
            f = squad_value_feature_dict(h, a)
            assert f["has_squad_value"] == 0.0
            assert f["home_squad_value"] == 0.0
            assert f["away_squad_value"] == 0.0
            assert f["squad_value_diff"] == 0.0

    def test_both_known_sets_the_indicator(self):
        from data_processor import squad_value_feature_dict
        assert squad_value_feature_dict(1, 1)["has_squad_value"] == 1.0

    def test_log_scaling_compresses_extreme_values(self):
        """Raw euros would let a handful of superclubs dominate any
        distance-based split; log space keeps the range usable."""
        from data_processor import squad_value_feature_dict
        small = squad_value_feature_dict(10_000_000, 1)["home_squad_value"]
        huge = squad_value_feature_dict(1_400_000_000, 1)["home_squad_value"]
        assert huge > small
        assert huge / small < 5  # 140x in euros, but far less in log space

    def test_equal_squads_give_zero_difference(self):
        from data_processor import squad_value_feature_dict
        f = squad_value_feature_dict(300_000_000, 300_000_000)
        assert f["squad_value_diff"] == pytest.approx(0.0)
        assert f["abs_squad_value_diff"] == pytest.approx(0.0)

    def test_abs_diff_is_symmetric(self):
        from data_processor import squad_value_feature_dict
        fwd = squad_value_feature_dict(900_000_000, 200_000_000)
        rev = squad_value_feature_dict(200_000_000, 900_000_000)
        assert fwd["abs_squad_value_diff"] == pytest.approx(rev["abs_squad_value_diff"])
        assert fwd["squad_value_diff"] == pytest.approx(-rev["squad_value_diff"])

    def test_features_absent_without_a_source(self):
        """No Transfermarkt data must degrade gracefully, not crash."""
        rows = [_match_row("2023-08-12T14:00:00Z", "Team A", "Team B", 1, 0)]
        result = add_form_features(rows, squad_values=None, club_map=None)
        assert result[0]["has_squad_value"] == 0.0


class TestTrainPredictFeatureAlignment:
    """prepare_training_data and prepare_prediction_features must build the
    SAME feature vector layout. If they drift, every prediction is silently
    computed against misaligned columns — the model still returns confident
    probabilities, they're just wrong. Nothing else in the codebase catches
    that, so it's guarded here."""

    def _built_rows(self):
        rows = [
            _match_row("2023-08-12T14:00:00Z", "Team A", "Team B", 2, 1),
            _match_row("2023-08-19T14:00:00Z", "Team A", "Team C", 1, 1),
            _match_row("2023-08-26T14:00:00Z", "Team B", "Team C", 0, 3),
            _match_row("2023-09-02T14:00:00Z", "Team C", "Team A", 2, 2),
        ]
        return add_form_features(rows)

    def test_vector_lengths_match(self):
        rows = self._built_rows()
        _, _, feature_names = prepare_training_data(rows)
        stats = compute_team_stats(rows)
        features = prepare_prediction_features("Team A", "Team B", stats)
        assert len(features) == len(feature_names), (
            f"training builds {len(feature_names)} features but prediction "
            f"builds {len(features)} — the two paths have drifted apart"
        )

    def test_evenness_features_are_non_negative(self):
        """Every abs_* feature is a magnitude; a negative one means a signed
        value leaked into an evenness slot."""
        rows = self._built_rows()
        _, _, feature_names = prepare_training_data(rows)
        stats = compute_team_stats(rows)
        features = prepare_prediction_features("Team A", "Team B", stats)
        for name, value in zip(feature_names, features):
            if name.startswith("abs_"):
                assert value >= 0, f"{name} is negative ({value})"

    def test_venue_independent_evenness_is_symmetric_under_swap(self):
        """Closeness measured from venue-independent quantities (Elo, form,
        points-per-fixture) must not change when the two teams swap sides.

        `abs_dc_exp_goals_diff` is deliberately excluded: it's derived from
        Dixon-Coles *expected goals*, which apply the home-advantage baseline
        (LEAGUE_AVG_HOME_GOALS vs LEAGUE_AVG_AWAY_GOALS) to whichever side is
        at home. Its gap genuinely depends on venue, so asymmetry there is
        correct behaviour rather than a leak."""
        venue_independent = {"abs_elo_diff", "abs_form_diff", "abs_ppf_10_diff"}
        rows = self._built_rows()
        _, _, feature_names = prepare_training_data(rows)
        stats = compute_team_stats(rows)
        forward = prepare_prediction_features("Team A", "Team B", stats)
        reverse = prepare_prediction_features("Team B", "Team A", stats)
        checked = 0
        for name, fwd, rev in zip(feature_names, forward, reverse):
            if name in venue_independent:
                assert fwd == pytest.approx(rev), f"{name} changed under swap"
                checked += 1
        assert checked == len(venue_independent), "an evenness feature went missing"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

class TestHeadToHeadAtPredictionTime:
    """H2H describes a PAIR of clubs, not either one alone. It was previously
    read off the home team's record, which held whatever that team's most
    recent unrelated fixture showed — so every live prediction silently got
    zeros while the model had trained on real H2H values."""

    def _rows(self):
        return add_form_features([
            _match_row("2023-08-12T14:00:00Z", "Team A", "Team B", 3, 0),
            _match_row("2023-09-12T14:00:00Z", "Team B", "Team A", 1, 1),
            _match_row("2023-10-12T14:00:00Z", "Team A", "Team C", 2, 0),
        ])

    def test_pair_key_is_order_independent(self):
        from data_processor import h2h_pair_key
        assert h2h_pair_key("Team A", "Team B") == h2h_pair_key("Team B", "Team A")

    def test_h2h_map_is_exposed_after_feature_building(self):
        self._rows()
        assert getattr(add_form_features, "last_h2h", None)

    def test_known_pair_has_history(self):
        from data_processor import h2h_features_for
        self._rows()
        h2h = add_form_features.last_h2h
        matches, home_wins, draws, away_wins = h2h_features_for("Team A", "Team B", h2h)
        assert matches == 2
        assert home_wins + draws + away_wins == matches

    def test_unseen_pair_is_zeros_not_an_error(self):
        from data_processor import h2h_features_for
        self._rows()
        assert h2h_features_for("Team A", "Nobody FC", add_form_features.last_h2h) == [0, 0, 0, 0]

    def test_missing_map_degrades_gracefully(self):
        from data_processor import h2h_features_for
        assert h2h_features_for("Team A", "Team B", None) == [0, 0, 0, 0]

    def test_prediction_features_reflect_pair_history(self):
        """The regression that started this: features must differ between a
        pair with history and a pair without."""
        rows = self._rows()
        stats = compute_team_stats(rows)
        h2h = add_form_features.last_h2h
        names = prepare_training_data(rows)[2]
        i = names.index("h2h_matches")
        with_hist = prepare_prediction_features("Team A", "Team B", stats, h2h)
        without = prepare_prediction_features("Team A", "Nobody FC", stats, h2h)
        assert with_hist[i] > 0
        assert without[i] == 0


class TestLiveFeaturesMatchTrainingDistribution:
    """Guard against the recurring failure mode in this project: a feature
    that is populated during training but degenerate at prediction time.

    It has bitten four times — team names, head-to-head, squad values, and
    season_progress — and each time the model kept returning confident
    probabilities computed from a value it had never seen while training.
    Checking that live vectors land inside the training range catches the
    whole class automatically.
    """

    def _built(self):
        rows = add_form_features([
            _match_row("2023-08-12T14:00:00Z", "Team A", "Team B", 2, 0),
            _match_row("2023-08-20T14:00:00Z", "Team B", "Team A", 1, 1),
            _match_row("2023-09-01T14:00:00Z", "Team A", "Team C", 0, 2),
            _match_row("2023-09-10T14:00:00Z", "Team C", "Team B", 3, 1),
            _match_row("2023-09-20T14:00:00Z", "Team B", "Team C", 1, 0),
        ])
        return rows, compute_team_stats(rows), add_form_features.last_h2h

    def test_season_progress_is_carried_to_prediction(self):
        """Regression: season_progress is match-level, so the per-team
        `home_`/`away_` prefix lookup could never find it and every live
        vector carried 0 — a value training effectively never contains."""
        _, stats, _ = self._built()
        assert stats["Team A"]["season_progress"] > 0

    def test_venue_split_form_comes_from_the_matching_venue(self):
        """Regression: these were read from a team's most recent match of ANY
        venue, so a side whose last game was away carried home_ppf_10 from
        nowhere (0) into every prediction.

        Asserts the mechanism rather than a positive value — a team can
        legitimately have 0 points at one venue, and asserting >0 would make
        the test pass or fail on the fixture's scorelines instead of on the
        carry-forward logic."""
        rows, stats, _ = self._built()
        last_home = [r for r in rows if r["home_team"] == "Team B"][-1]
        last_away = [r for r in rows if r["away_team"] == "Team B"][-1]
        assert stats["Team B"]["home_ppf_10"] == last_home["home_home_ppf_10"]
        assert stats["Team B"]["away_ppf_10"] == last_away["away_away_ppf_10"]

    def test_home_form_not_taken_from_an_away_fixture(self):
        """The specific corruption: Team A's last match is away, so before
        the fix its home_ppf_10 was 0 rather than its real home record."""
        rows, stats, _ = self._built()
        last_home = [r for r in rows if r["home_team"] == "Team A"][-1]
        assert stats["Team A"]["home_ppf_10"] == last_home["home_home_ppf_10"]
        assert stats["Team A"]["home_ppf_10"] > 0

    def test_no_live_feature_is_wildly_outside_training_range(self):
        rows, stats, h2h = self._built()
        X, _, names = prepare_training_data(rows)
        live = prepare_prediction_features("Team A", "Team B", stats, h2h)
        cols = list(zip(*X))
        offenders = []
        for i, name in enumerate(names):
            col = [c for c in cols[i]]
            lo, hi = min(col), max(col)
            if lo == hi:
                continue  # constant in this tiny fixture set, nothing to compare
            span = hi - lo
            if live[i] < lo - span or live[i] > hi + span:
                offenders.append((name, live[i], lo, hi))
        assert not offenders, f"live features far outside training range: {offenders}"
