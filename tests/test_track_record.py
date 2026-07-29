"""Tests for the prediction track-record scoring (no network)."""

from track_record import score, _result_label


def _rec(mid, league, home, away, probs, pick, actual):
    return {"match_id": mid, "utc_date": "2026-08-15T14:00:00Z", "league": league,
            "home": home, "away": away, "home_display": home, "away_display": away,
            "probs": probs, "pick": pick, "logged_at": "2026-08-10",
            "status": "settled", "actual": actual, "actual_score": "1-0",
            "settled_at": "2026-08-16"}


def _p(h, d, a):
    return {"Home Win": h, "Draw": d, "Away Win": a}


class TestResultLabel:
    def test_labels(self):
        assert _result_label(2, 0) == "Home Win"
        assert _result_label(1, 1) == "Draw"
        assert _result_label(0, 2) == "Away Win"


class TestScore:
    def test_empty(self):
        s = score([])
        assert s["n_settled"] == 0 and s["accuracy"] is None

    def test_pending_counted_not_scored(self):
        recs = [dict(_rec(1, "PL", "A", "B", _p(.6, .25, .15), "Home Win", "Home Win"),
                     status="pending", actual=None)]
        s = score(recs)
        assert s["n_pending"] == 1 and s["n_settled"] == 0

    def test_accuracy_and_metrics(self):
        recs = [
            _rec(1, "PL", "A", "B", _p(.6, .25, .15), "Home Win", "Home Win"),   # correct
            _rec(2, "PL", "C", "D", _p(.5, .3, .2), "Home Win", "Away Win"),     # wrong
            _rec(3, "PD", "E", "F", _p(.2, .3, .5), "Away Win", "Away Win"),     # correct
            _rec(4, "PD", "G", "H", _p(.33, .34, .33), "Draw", "Draw"),          # correct
        ]
        s = score(recs)
        assert s["n_settled"] == 4
        assert abs(s["accuracy"] - 0.75) < 1e-9
        assert s["rps"] is not None and 0 <= s["rps"] <= 1
        assert s["log_loss"] is not None and s["brier"] is not None
        # per-league present with both leagues
        lgs = {r["league"] for r in s["per_league"]}
        assert lgs == {"PL", "PD"}

    def test_calibration_bands_and_recent(self):
        recs = [_rec(i, "PL", f"A{i}", f"B{i}", _p(.7, .2, .1), "Home Win",
                     "Home Win" if i % 2 == 0 else "Away Win") for i in range(6)]
        s = score(recs)
        # all picks sit in the 70-80% confidence band
        band = [c for c in s["calibration"] if c["n"] > 0][0]
        assert band["predicted"] == 70.0 and band["n"] == 6
        assert 0 <= band["actual"] <= 100
        assert len(s["recent"]) == 6
        assert all("correct" in r for r in s["recent"])

    def test_baseline_is_home_rate(self):
        recs = [_rec(1, "PL", "A", "B", _p(.6, .2, .2), "Home Win", "Home Win"),
                _rec(2, "PL", "C", "D", _p(.6, .2, .2), "Home Win", "Draw")]
        s = score(recs)
        assert abs(s["baseline_acc"] - 0.5) < 1e-9   # 1 of 2 were actually home wins
