"""Tests for the Monte Carlo season simulator's pure core."""

import itertools

import numpy as np

import leagues
from simulate_season import (simulate, simulate_projection, mc_stderr,
                             _dc_score_grid, _conditional_scoreline_tables)


def _round_robin(n, prob_fn):
    """Build fixtures for n teams; prob_fn(i,j)->(ph,pd,pa) for i home vs j."""
    fx = []
    for i in range(n):
        for j in range(n):
            if i != j:
                ph, pd, pa = prob_fn(i, j)
                fx.append((i, j, ph, pd, pa))
    return fx


class TestSimulate:
    def test_outputs_are_probabilities_that_sum_right(self):
        fx = _round_robin(6, lambda i, j: (0.4, 0.3, 0.3))
        out = simulate(fx, 6, n_sims=2000)
        for key in ("title_prob", "top4_prob", "relegation_prob"):
            assert np.all(out[key] >= 0) and np.all(out[key] <= 1)
        assert abs(out["title_prob"].sum() - 1.0) < 1e-9   # exactly one champion per sim
        assert abs(out["relegation_prob"].sum() - 3.0) < 1e-9  # exactly 3 relegated (n_teams-3)

    def test_a_dominant_team_wins_the_title_most_often(self):
        # Team 0 wins every home game and rarely loses away; others are even.
        def probs(i, j):
            if i == 0:
                return (0.9, 0.05, 0.05)   # team 0 at home: almost always wins
            if j == 0:
                return (0.1, 0.1, 0.8)     # team 0 away: usually wins
            return (0.4, 0.3, 0.3)
        out = simulate(_round_robin(8, probs), 8, n_sims=3000)
        assert out["title_prob"][0] > 0.8
        assert out["expected_points"][0] == max(out["expected_points"])
        assert out["relegation_prob"][0] < 0.01

    def test_symmetric_league_is_roughly_uniform(self):
        out = simulate(_round_robin(10, lambda i, j: (0.45, 0.27, 0.28)), 10, n_sims=4000)
        # No team should dominate when every fixture has identical odds.
        assert out["title_prob"].max() < 0.25

    def test_reproducible_with_seed(self):
        fx = _round_robin(6, lambda i, j: (0.4, 0.3, 0.3))
        a = simulate(fx, 6, n_sims=1000, seed=7)
        b = simulate(fx, 6, n_sims=1000, seed=7)
        assert np.allclose(a["expected_points"], b["expected_points"])


def _rr8(n, prob_fn, eh=1.4, ea=1.1, rho=-0.1):
    """Round-robin of 8-tuples (with rho) for the deeper projection core."""
    fx = []
    for i in range(n):
        for j in range(n):
            if i != j:
                ph, pd, pa = prob_fn(i, j)
                fx.append((i, j, ph, pd, pa, eh, ea, rho))
    return fx


class TestSimulateProjection:
    def test_calibrated_marginals_preserved(self):
        # The outcome must follow the calibrated (ph,pd,pa); conditional scoreline
        # sampling must not shift the home-win rate for a fixture.
        fx = [(0, 1, 0.6, 0.25, 0.15, 1.8, 1.0, -0.1)]
        # one fixture, seed points so ranking is irrelevant; check via a big sim
        rng_out = simulate_projection(fx * 1, 2, np.zeros(2), np.zeros(2),
                                      n_sims=40000, seed=3,
                                      rules=leagues.league_rules("PL", 2))
        # team 0 expected points ~= 3*0.6 + 1*0.25 = 2.05 over one match
        assert abs(rng_out["expected_points"][0] - 2.05) < 0.06

    def test_tiers_and_aggregates_present_and_sane(self):
        out = simulate_projection(_rr8(8, lambda i, j: (0.45, 0.27, 0.28)), 8,
                                  np.zeros(8), np.zeros(8), n_sims=4000,
                                  rules=leagues.league_rules("PL", 8),
                                  start_gf=np.zeros(8))
        for key in ("ucl_prob", "uel_prob", "relegation_prob", "btts_pct", "over25_pct",
                    "expected_gf", "expected_ga", "expected_gd", "title_se"):
            assert key in out
        assert abs(out["title_prob"].sum() - 1.0) < 1e-9
        # gd == gf - ga per team
        assert np.allclose(out["expected_gd"], out["expected_gf"] - out["expected_ga"], atol=1e-6)
        # aggregates are probabilities
        assert np.all(out["btts_pct"] >= 0) and np.all(out["btts_pct"] <= 1)

    def test_goals_for_tiebreak(self):
        # Two identical-strength teams, but team 0 scores more (higher eh in its
        # home fixtures) -> should out-rank on GF when points/GD are close.
        fx = ([(0, 1, 0.5, 0.0, 0.5, 3.0, 1.0, -0.1)]      # 0 wins big at home
              + [(1, 0, 0.5, 0.0, 0.5, 1.0, 1.0, -0.1)])   # away leg even
        out = simulate_projection(fx, 2, np.zeros(2), np.zeros(2), n_sims=4000,
                                  rules=leagues.league_rules("PL", 2))
        assert out["expected_gf"][0] > out["expected_gf"][1]

    def test_backcompat_seven_tuple(self):
        fx = [(0, 1, 0.5, 0.25, 0.25, 1.5, 1.0),
              (1, 0, 0.4, 0.3, 0.3, 1.3, 1.2)]
        out = simulate_projection(fx, 2, np.zeros(2), np.zeros(2), n_sims=2000)
        assert "title_prob" in out and abs(out["title_prob"].sum() - 1.0) < 1e-9

    def test_reproducible(self):
        fx = _rr8(6, lambda i, j: (0.4, 0.3, 0.3))
        a = simulate_projection(fx, 6, np.zeros(6), np.zeros(6), n_sims=2000, seed=7)
        b = simulate_projection(fx, 6, np.zeros(6), np.zeros(6), n_sims=2000, seed=7)
        assert np.allclose(a["expected_points"], b["expected_points"])
        assert np.allclose(a["expected_gf"], b["expected_gf"])


class TestScorelineGrid:
    def test_grid_normalized_and_nonneg(self):
        g = _dc_score_grid(1.6, 1.1, -0.1)
        assert abs(g.sum() - 1.0) < 1e-9 and np.all(g >= 0)

    def test_conditional_tables_cover_outcomes(self):
        t = _conditional_scoreline_tables(_dc_score_grid(1.5, 1.2, -0.1))
        for key in ("h", "d", "a"):
            hg, ag, cdf = t[key]
            assert len(hg) == len(ag) == len(cdf) and abs(cdf[-1] - 1.0) < 1e-9


class TestMcStderr:
    def test_formula_and_shrinks_with_n(self):
        assert abs(mc_stderr(0.5, 10000) - (0.25 / 10000) ** 0.5) < 1e-12
        assert mc_stderr(0.5, 40000) < mc_stderr(0.5, 10000)
