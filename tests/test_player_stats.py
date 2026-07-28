"""Tests for player_stats — the feature helper and point-in-time index logic
(offline; the appearances fetch/reduce is exercised by the training pipeline)."""

import player_stats as ps


class TestFeatureDict:
    def test_both_known(self):
        f = ps.player_stats_feature_dict(
            {"ga_per_game": 3.0, "cards_per_game": 1.5},
            {"ga_per_game": 1.5, "cards_per_game": 2.0})
        assert f["has_player_stats"] == 1.0
        assert f["ga_per_game_diff"] == 1.5
        assert f["home_cards_per_game"] == 1.5 and f["away_cards_per_game"] == 2.0

    def test_one_missing_disables_block(self):
        f = ps.player_stats_feature_dict({"ga_per_game": 3.0}, None)
        assert f["has_player_stats"] == 0.0
        assert f["home_ga_per_game"] == 0.0 and f["ga_per_game_diff"] == 0.0


class TestIndexPointInTime:
    def _idx(self):
        idx = ps.PlayerStatsIndex()
        # 12 games for club "1": goals+assists ramping, dated 2024-01..2024-12
        idx.series = {"1": [[f"2024-{m:02d}-01", float(m), 1.0, 2.0] for m in range(1, 13)]}
        return idx

    def test_no_lookahead(self):
        idx = self._idx()
        # As of 2024-07, only games before it count (Jan..Jun = 6 games < 10 -> None)
        assert idx.features_at("1", "2024-07-01") is None
        # As of 2024-12-15, 11 prior games (Jan..Nov, since Dec is 12-01 < 12-15)
        f = idx.features_at("1", "2024-12-15")
        assert f is not None
        # ga_per_game = mean(goals m + assists 1) over the window
        assert f["cards_per_game"] == 2.0

    def test_unknown_club_is_none(self):
        assert self._idx().features_at("999", "2025-01-01") is None

    def test_thin_history_is_none(self):
        idx = ps.PlayerStatsIndex()
        idx.series = {"1": [["2024-01-01", 1.0, 1.0, 1.0]]}  # 1 game < 10
        assert idx.features_at("1", "2025-01-01") is None
