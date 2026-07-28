"""Tests for the player-data adapter (offline — no API key needed)."""

import player_data as pd


class TestAvailabilityFeatures:
    def test_both_known(self):
        f = pd.player_availability_features(
            {"value_out_fraction": 0.05, "key_absences": 1},
            {"value_out_fraction": 0.20, "key_absences": 3})
        assert f["has_player_data"] == 1.0
        assert abs(f["availability_edge"] - 0.15) < 1e-9  # away more depleted -> favours home
        assert f["home_key_absences"] == 1 and f["away_key_absences"] == 3

    def test_one_missing_disables_block(self):
        f = pd.player_availability_features({"value_out_fraction": 0.05}, None)
        assert f["has_player_data"] == 0.0
        assert f["availability_edge"] == 0.0 and f["home_value_out_frac"] == 0.0


class TestLiveAdjustment:
    def test_no_data_leaves_probs_unchanged(self):
        probs = {"Home Win": 0.5, "Draw": 0.3, "Away Win": 0.2}
        assert pd.live_availability_adjustment(None, {"value_out_fraction": 0.1}, probs) == probs

    def test_home_less_depleted_shifts_toward_home_and_conserves_mass(self):
        probs = {"Home Win": 0.4, "Draw": 0.3, "Away Win": 0.3}
        out = pd.live_availability_adjustment(
            {"value_out_fraction": 0.0}, {"value_out_fraction": 0.2}, probs)
        assert out["Home Win"] > probs["Home Win"]
        assert out["Away Win"] < probs["Away Win"]
        assert abs(sum(out.values()) - 1.0) < 1e-9
        assert out["Draw"] == probs["Draw"]  # draw untouched

    def test_no_token_source_is_inert(self):
        src = pd.ApiFootballSource(token="")
        assert src.availability("Arsenal", "2026-08-22") is None


# ─── Bring-your-own-data adapter (Phase 5) ─────────────────────────────────
import json as _json
import os as _os
from player_data import (FilePlayerDataSource, style_feature_dict, get_source,
                         live_availability_adjustment)


class TestFilePlayerDataSource:
    def test_reads_availability_and_style(self, tmp_path):
        p = tmp_path / "pd.json"
        p.write_text(_json.dumps({
            "availability": {"Man City": {"value_out_fraction": 0.1, "key_absences": 1}},
            "style": {"Man City": {"pressing": 0.8, "tempo": 0.7}},
        }))
        src = FilePlayerDataSource(str(p))
        assert src.available()
        assert src.availability("Man City", "2026-01-01")["value_out_fraction"] == 0.1
        assert src.style("Man City", "2026-01-01")["pressing"] == 0.8
        assert src.availability("Nobody", "2026-01-01") is None

    def test_absent_file_is_inert(self, tmp_path):
        src = FilePlayerDataSource(str(tmp_path / "missing.json"))
        assert not src.available()
        assert src.availability("Man City", "2026-01-01") is None


class TestStyleFeatureDict:
    def test_both_sides_gate(self):
        d = style_feature_dict({"pressing": 0.8}, {"pressing": 0.3})
        assert d["has_style"] == 1.0 and d["home_pressing"] == 0.8
        assert style_feature_dict({"pressing": 0.8}, None)["has_style"] == 0.0


class TestLiveNudge:
    def test_shift_favours_less_depleted_side(self):
        probs = {"Home Win": 0.4, "Draw": 0.3, "Away Win": 0.3}
        # away more depleted -> home win prob rises, away falls, draw unchanged
        out = live_availability_adjustment({"value_out_fraction": 0.0},
                                           {"value_out_fraction": 0.2}, probs)
        assert out["Home Win"] > probs["Home Win"] and out["Away Win"] < probs["Away Win"]
        assert abs(out["Draw"] - probs["Draw"]) < 1e-9
        assert abs(sum(out.values()) - 1.0) < 1e-9

    def test_unknown_side_leaves_probs_unchanged(self):
        probs = {"Home Win": 0.4, "Draw": 0.3, "Away Win": 0.3}
        assert live_availability_adjustment(None, {"value_out_fraction": 0.2}, probs) == probs
