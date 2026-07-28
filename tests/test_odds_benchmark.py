"""Tests for odds_benchmark.py de-vig math — previously untested (audit finding).
Odds are used ONLY here as an offline benchmark, never as a model feature."""

import odds_benchmark as ob


class TestDevig:
    def test_normalizes_to_one(self):
        p = ob._devig(2.0, 3.5, 4.0)  # (home, draw, away) decimal odds
        assert p is not None
        assert abs(sum(p) - 1.0) < 1e-9

    def test_shorter_odds_mean_higher_probability(self):
        p_home, p_draw, p_away = ob._devig(1.5, 4.0, 7.0)
        assert p_home > p_draw > p_away  # 1.5 shortest -> most likely

    def test_removes_the_overround(self):
        # Raw implied probs sum to >1 (the vig); de-vigged must sum to exactly 1.
        p = ob._devig(2.0, 2.0, 2.0)  # implied 0.5+0.5+0.5 = 1.5 overround
        assert p == (1 / 3, 1 / 3, 1 / 3)

    def test_zero_or_bad_odds_return_none(self):
        assert ob._devig(0.0, 3.0, 4.0) is None
