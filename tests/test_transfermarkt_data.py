"""Tests for transfermarkt_data — point-in-time squad market values.

The critical property is no-lookahead: a squad value used as a feature for a
match must be computable from valuations published strictly before it.
Getting this wrong would inflate offline scores while doing nothing for real
predictions — the same trap the per-league goal baselines are guarded against
in test_data_processor.py.
"""

import pytest
from transfermarkt_data import SquadValueIndex, _month_key, _next_month


def _val(date, pid, value, club):
    return {"date": date, "player_id": pid,
            "market_value_in_eur": str(value), "current_club_id": club}


class TestMonthHelpers:
    def test_month_key(self):
        assert _month_key("2023-03-15") == "2023-03"

    def test_next_month_rolls_year(self):
        assert _next_month("2023-11") == "2023-12"
        assert _next_month("2023-12") == "2024-01"


class TestNoLookahead:
    def test_later_valuations_do_not_change_earlier_value(self):
        base = [_val("2023-01-10", "p1", 1_000_000, "A"),
                _val("2023-02-10", "p2", 2_000_000, "A")]
        before = SquadValueIndex().build(base).value_at("A", "2023-02-05")
        withfuture = base + [_val("2023-06-01", "p3", 900_000_000, "A")]
        after = SquadValueIndex().build(withfuture).value_at("A", "2023-02-05")
        assert before == after

    def test_future_valuation_does_appear_later(self):
        rows = [_val("2023-01-10", "p1", 1_000_000, "A"),
                _val("2023-06-01", "p2", 5_000_000, "A")]
        idx = SquadValueIndex().build(rows)
        assert idx.value_at("A", "2023-07-01") == 6_000_000

    def test_value_before_any_data_is_none(self):
        idx = SquadValueIndex().build([_val("2023-05-01", "p1", 1_000_000, "A")])
        assert idx.value_at("A", "2020-01-01") is None


class TestValuationSemantics:
    def test_latest_valuation_per_player_wins(self):
        """A re-valued player must not be double-counted."""
        rows = [_val("2023-01-10", "p1", 1_000_000, "A"),
                _val("2023-02-10", "p1", 4_000_000, "A")]
        idx = SquadValueIndex().build(rows)
        assert idx.value_at("A", "2023-06-01") == 4_000_000

    def test_transfer_moves_value_between_clubs(self):
        """This is what makes new signings register: a player's later
        valuation carries their new club, so value follows them."""
        rows = [_val("2023-01-10", "p1", 5_000_000, "A"),
                _val("2023-03-10", "p1", 5_000_000, "B")]
        idx = SquadValueIndex().build(rows)
        assert idx.value_at("A", "2023-02-01") == 5_000_000
        assert idx.value_at("B", "2023-02-01") is None
        assert idx.value_at("B", "2023-06-01") == 5_000_000
        assert not idx.value_at("A", "2023-06-01")

    def test_unknown_club_returns_none(self):
        idx = SquadValueIndex().build([_val("2023-01-10", "p1", 1, "A")])
        assert idx.value_at("ZZZ", "2023-06-01") is None

    def test_malformed_rows_are_skipped_not_fatal(self):
        rows = [_val("2023-01-10", "p1", 1_000_000, "A"),
                {"date": "", "player_id": "p2", "market_value_in_eur": "5", "current_club_id": "A"},
                {"date": "2023-01-11", "player_id": "p3", "market_value_in_eur": "notanumber",
                 "current_club_id": "A"}]
        idx = SquadValueIndex().build(rows)
        assert idx.value_at("A", "2023-06-01") == 1_000_000

    def test_empty_input_is_safe(self):
        idx = SquadValueIndex().build([])
        assert idx.value_at("A", "2023-01-01") is None


class TestCurrentValues:
    def test_current_comes_from_players_table(self):
        vals = [_val("2023-01-10", "p1", 1_000_000, "A")]
        players = [{"current_club_id": "A", "market_value_in_eur": "7000000"},
                   {"current_club_id": "A", "market_value_in_eur": "3000000"}]
        idx = SquadValueIndex().build(vals, players)
        assert idx.current_value("A") == 10_000_000
        assert idx.current_squad_size["A"] == 2

    def test_player_without_value_still_counts_toward_squad_size(self):
        players = [{"current_club_id": "A", "market_value_in_eur": ""}]
        idx = SquadValueIndex().build([_val("2023-01-10", "p1", 1, "A")], players)
        assert idx.current_squad_size["A"] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
