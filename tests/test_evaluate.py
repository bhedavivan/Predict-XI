"""Tests for evaluate.py — previously untested (audit finding)."""

import numpy as np

import evaluate as ev
from model_trainer import _rps as trainer_rps


class TestRpsIsShared:
    def test_evaluate_reuses_the_trainer_rps(self):
        # Deduped: evaluate imports _rps from model_trainer so the two can't drift.
        assert ev._rps is trainer_rps

    def test_rps_perfect_and_ordinality(self):
        cl = [-1, 0, 1]
        assert ev._rps([1], [[0.0, 0.0, 1.0]], cl) == 0.0
        far = ev._rps([1], [[1.0, 0.0, 0.0]], cl)       # away when home
        adjacent = ev._rps([1], [[0.0, 1.0, 0.0]], cl)  # draw when home
        assert far > adjacent


class TestLogLoss:
    def test_matches_hand_value(self):
        ll = ev._log_loss(np.array([1]), np.array([[0.1, 0.2, 0.7]]), [-1, 0, 1])
        assert abs(ll - (-np.log(0.7))) < 1e-9

    def test_empty_is_nan(self):
        assert np.isnan(ev._log_loss(np.array([]), np.zeros((0, 3)), [-1, 0, 1]))


class TestReliabilityCurve:
    def test_bins_have_confidence_and_accuracy(self):
        y = np.array([1, 1, -1, 0])
        P = np.array([[0.1, 0.2, 0.7], [0.2, 0.3, 0.5], [0.6, 0.2, 0.2], [0.3, 0.4, 0.3]])
        rows = ev.reliability_curve(y, P, [-1, 0, 1])
        assert rows and all("mean_confidence" in r and "observed_accuracy" in r for r in rows)


class TestPerLeagueBreakdown:
    def test_reports_a_big_league_and_pools_small_ones(self):
        y = np.array(([1, 0, -1] * 50) + [1, 0])       # 150 in PL, 2 in tiny
        p = y.copy()
        lg = (["PL"] * 150) + ["ZZZ", "ZZZ"]
        rows = ev.per_league_breakdown(y, p, lg, [-1, 0, 1], min_matches=100)
        names = [r["league"] for r in rows]
        assert "PL" in names
        assert any(r.get("pooled") for r in rows)      # the 2-match league is pooled


class TestDrawThresholdCurve:
    def test_structure_and_disjoint_split(self):
        rng = np.random.RandomState(0)
        n = 400
        P = rng.dirichlet([3, 2, 3], size=n)
        classes = [-1, 0, 1]
        y = np.array([classes[i] for i in P.argmax(1)])
        out = ev.draw_threshold_curve(y, P, classes)
        assert "chosen_threshold" in out and "survives_holdout" in out and "verdict" in out
        assert isinstance(out["survives_holdout"], bool)
