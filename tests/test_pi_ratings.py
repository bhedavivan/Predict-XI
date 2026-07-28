"""Tests for the pi-ratings module."""

import math

from pi_ratings import (PiRatings, pi_feature_dict, expected_goals,
                        PI_MIN_MATCHES, PI_GAMMA)


class TestExpectedGoals:
    def test_zero_and_symmetry(self):
        assert expected_goals(0.0) == 0.0
        assert expected_goals(1.5) == -expected_goals(-1.5)

    def test_monotonic(self):
        assert expected_goals(2.0) > expected_goals(1.0) > expected_goals(0.5) > 0


class TestPiRatings:
    def test_home_win_raises_home_rating(self):
        r = PiRatings()
        r.update("A", "B", 2, 0)
        assert r.home_rating("A") > 0
        assert r.away_rating("B") < 0   # B under-performed away

    def test_cross_venue_transfer(self):
        r = PiRatings()
        before_away = r.away_rating("A")
        r.update("A", "B", 3, 0)
        d_home = r.home_rating("A")               # started at 0
        d_away = r.away_rating("A") - before_away
        # A's away rating moved by gamma * its home move.
        assert math.isclose(d_away, PI_GAMMA * d_home, rel_tol=1e-9)

    def test_big_margin_damped(self):
        r1 = PiRatings(); r1.update("A", "B", 1, 0)
        r5 = PiRatings(); r5.update("A", "B", 5, 0)
        # A 5-0 raises the rating more than a 1-0, but by far less than 5x.
        assert r5.home_rating("A") > r1.home_rating("A")
        assert r5.home_rating("A") < 5 * r1.home_rating("A")

    def test_games_counted(self):
        r = PiRatings()
        r.update("A", "B", 1, 1)
        assert r.games("A") == 1 and r.games("B") == 1

    def test_ratings_stay_finite(self):
        r = PiRatings()
        for _ in range(200):
            r.update("A", "B", 9, 0)   # relentless blowouts
        assert math.isfinite(r.home_rating("A")) and abs(r.home_rating("A")) <= 4.0


class TestPiFeatureDict:
    def test_gate_requires_both_sides_and_min_matches(self):
        # Not enough games -> zeroed block.
        d = pi_feature_dict(1.0, -1.0, PI_MIN_MATCHES - 1, PI_MIN_MATCHES)
        assert d["has_pi"] == 0.0 and d["pi_diff"] == 0.0
        # Both sides ready -> populated block, favouring the stronger home side.
        d = pi_feature_dict(1.0, -1.0, PI_MIN_MATCHES, PI_MIN_MATCHES)
        assert d["has_pi"] == 1.0 and d["pi_diff"] > 0 and d["pi_expected_gd"] > 0

    def test_missing_rating_is_zeroed(self):
        d = pi_feature_dict(None, -1.0, 10, 10)
        assert d["has_pi"] == 0.0 and d["home_pi"] == 0.0
