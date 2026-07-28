"""Tests for clubelo_data — offline (no network; the fetch/cache is exercised
by the training pipeline)."""

import clubelo_data as ce


class TestFeatureDict:
    def test_both_known(self):
        f = ce.clubelo_feature_dict(1900, 1500)
        assert f["has_clubelo"] == 1.0
        assert f["clubelo_diff"] == 400
        assert f["home_clubelo"] == 1900 and f["away_clubelo"] == 1500
        assert abs(f["clubelo_expected"] - 1 / (1 + 10 ** (-1))) < 1e-9  # ~0.909

    def test_one_missing_disables_block(self):
        f = ce.clubelo_feature_dict(1900, None)
        assert f["has_clubelo"] == 0.0
        assert f["home_clubelo"] == 0.0 and f["clubelo_diff"] == 0.0
        assert f["clubelo_expected"] == 0.5  # neutral when unknown

    def test_expected_symmetric_at_parity(self):
        assert abs(ce.clubelo_feature_dict(1600, 1600)["clubelo_expected"] - 0.5) < 1e-9


class TestNormalize:
    def test_accent_insensitive(self):
        assert ce._normalize("Atlético") == ce._normalize("Atletico")

    def test_drops_common_club_tokens_and_punct(self):
        assert ce._normalize("Arsenal FC") == ce._normalize("Arsenal")
        assert ce._normalize("Bayer 04") == ce._normalize("bayer04")


class TestResolveSafeMatching:
    def _idx(self, clubs=("Man City", "Man United", "Arsenal")):
        idx = ce.ClubEloIndex()
        for club in clubs:
            nn = ce._normalize(club)
            idx.snapshots["ENG"][nn]["2025-01"] = 1900.0
            idx.current["ENG"][nn] = 1900.0
        return idx

    def test_exact_match(self):
        assert self._idx().current_elo("Arsenal", "PL") == 1900.0

    def test_manual_override(self):
        idx = self._idx()
        nn = ce._normalize("Forest")
        idx.snapshots["ENG"][nn]["2025-01"] = 1800.0
        idx.current["ENG"][nn] = 1800.0
        assert idx.current_elo("Nott'm Forest", "PL") == 1800.0  # via CLUBELO_MANUAL

    def test_unmapped_returns_none(self):
        assert self._idx().current_elo("Nonexistent United", "PL") is None

    def test_non_european_league_returns_none(self):
        # ARG1 is intentionally absent from LEAGUE_TO_CLUBELO_COUNTRY.
        assert self._idx().current_elo("Boca Juniors", "ARG1") is None

    def test_value_at_has_no_lookahead(self):
        idx = ce.ClubEloIndex()
        nn = ce._normalize("Arsenal")
        idx.snapshots["ENG"][nn] = {"2024-01": 1800.0, "2025-01": 1900.0}
        assert idx.elo_at("Arsenal", "PL", "2024-06-15") == 1800.0   # most recent prior
        assert idx.elo_at("Arsenal", "PL", "2025-03-01") == 1900.0
        assert idx.elo_at("Arsenal", "PL", "2023-01-01") is None     # before first snapshot
