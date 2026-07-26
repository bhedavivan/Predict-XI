"""Unit tests for the dixon_coles module."""

import pytest
from dixon_coles import (
    DixonColesRatings,
    match_probabilities,
    expected_goals,
    DC_START,
)


class TestExpectedGoals:
    def test_neutral_ratings_give_league_averages(self):
        exp_home, exp_away = expected_goals(DC_START, DC_START, DC_START, DC_START)
        from dixon_coles import LEAGUE_AVG_HOME_GOALS, LEAGUE_AVG_AWAY_GOALS
        assert exp_home == pytest.approx(LEAGUE_AVG_HOME_GOALS)
        assert exp_away == pytest.approx(LEAGUE_AVG_AWAY_GOALS)

    def test_stronger_attack_raises_expected_goals(self):
        base_home, _ = expected_goals(1.0, 1.0, 1.0, 1.0)
        boosted_home, _ = expected_goals(1.5, 1.0, 1.0, 1.0)
        assert boosted_home > base_home


class TestMatchProbabilities:
    def test_probabilities_sum_to_one(self):
        p_home, p_draw, p_away, _, _ = match_probabilities(1.0, 1.0, 1.0, 1.0)
        assert p_home + p_draw + p_away == pytest.approx(1.0, abs=1e-6)

    def test_equal_teams_favor_home(self):
        """Home advantage should still show up between two neutral teams."""
        p_home, p_draw, p_away, _, _ = match_probabilities(1.0, 1.0, 1.0, 1.0)
        assert p_home > p_away

    def test_strong_attack_beats_weak_defense(self):
        p_home_strong, _, _, _, _ = match_probabilities(1.8, 1.0, 0.6, 1.5)
        p_home_neutral, _, _, _, _ = match_probabilities(1.0, 1.0, 1.0, 1.0)
        assert p_home_strong > p_home_neutral

    def test_all_probabilities_non_negative(self):
        p_home, p_draw, p_away, _, _ = match_probabilities(2.5, 0.4, 0.4, 2.5)
        assert p_home >= 0 and p_draw >= 0 and p_away >= 0


class TestDixonColesRatings:
    def test_new_teams_start_neutral(self):
        r = DixonColesRatings()
        assert r.attack["Team A"] == DC_START
        assert r.defense["Team A"] == DC_START

    def test_repeated_big_wins_raise_attack_and_hurt_opponent_defense(self):
        r = DixonColesRatings()
        for _ in range(10):
            r.update("Strong", "Weak", 3, 0)
        assert r.attack["Strong"] > DC_START
        assert r.defense["Weak"] > DC_START  # higher = worse defense

    def test_ratings_stay_within_clip_bounds(self):
        from dixon_coles import DC_MIN, DC_MAX
        r = DixonColesRatings()
        for _ in range(200):
            r.update("Strong", "Weak", 6, 0)
        assert DC_MIN <= r.attack["Strong"] <= DC_MAX
        assert DC_MIN <= r.defense["Weak"] <= DC_MAX

    def test_predict_does_not_mutate_ratings(self):
        r = DixonColesRatings()
        r.update("A", "B", 2, 1)
        before = (r.attack["A"], r.defense["A"])
        r.predict("A", "B")
        assert (r.attack["A"], r.defense["A"]) == before


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
