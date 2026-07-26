"""Unit tests for team_aliases — resolving live-API team names (official
long form) to the training data's short-name convention."""

import pytest
from team_aliases import resolve_team_name, strip_accents, TEAM_NAME_ALIASES


FAKE_STATS = {
    "Man United": {}, "Hull": {}, "Paris SG": {}, "Rennes": {},
    "Bayern Munich": {}, "Barcelona": {},
}


class TestStripAccents:
    def test_removes_diacritics(self):
        assert strip_accents("München") == "Munchen"
        assert strip_accents("Málaga") == "Malaga"

    def test_handles_empty(self):
        assert strip_accents("") == ""
        assert strip_accents(None) == ""


class TestResolveTeamName:
    def test_exact_match_short_circuits(self):
        assert resolve_team_name("Man United", FAKE_STATS) == "Man United"

    def test_resolves_official_long_name(self):
        assert resolve_team_name("Manchester United FC", FAKE_STATS) == "Man United"
        assert resolve_team_name("FC Bayern München", FAKE_STATS) == "Bayern Munich"
        assert resolve_team_name("Hull City AFC", FAKE_STATS) == "Hull"
        assert resolve_team_name("Paris Saint-Germain FC", FAKE_STATS) == "Paris SG"
        assert resolve_team_name("Stade Rennais FC 1901", FAKE_STATS) == "Rennes"

    def test_resolves_short_name_variant(self):
        assert resolve_team_name("PSG", FAKE_STATS) == "Paris SG"
        assert resolve_team_name("Barça", FAKE_STATS) == "Barcelona"

    def test_unresolvable_returns_none_not_a_guess(self):
        assert resolve_team_name("Some Totally Unknown FC", FAKE_STATS) is None

    def test_empty_name_returns_none(self):
        assert resolve_team_name("", FAKE_STATS) is None
        assert resolve_team_name(None, FAKE_STATS) is None

    def test_alias_target_must_exist_in_stats(self):
        """If the alias table points somewhere not actually in team_stats
        (e.g. a stale entry after retraining with different team names),
        resolution must fail closed, not return a dangling name."""
        assert resolve_team_name("FC Bayern München", {}) is None

    def test_no_dangerous_cross_club_collisions(self):
        """Regression guard for the exact failure mode fuzzy matching
        produced: rival/unrelated clubs must never resolve to each other."""
        stats = {"Barcelona": {}, "Real Madrid": {}}
        assert resolve_team_name("RCD Espanyol de Barcelona", stats) is None
        assert resolve_team_name("Rayo Vallecano de Madrid", stats) is None


class TestAliasTableIntegrity:
    def test_all_keys_are_normalized(self):
        for source in TEAM_NAME_ALIASES:
            assert source == strip_accents(source).lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
