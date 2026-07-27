"""Tests for club_mapping — joining our club names to Transfermarkt records.

The point of these tests is that a WRONG join is far worse than no join: it
silently attaches a different club's squad value and the model keeps
returning confident probabilities. Same failure shape as the Fixtures
team-name bug, so the known-dangerous pairs are pinned here.
"""

import pytest
from club_mapping import (
    build_club_mapping,
    normalize,
    LEAGUE_TO_TM_COMP,
    MANUAL_CLUB_MAP,
    KNOWN_ABSENT,
)


def _clubs(*rows):
    return [{"club_id": cid, "name": nm, "domestic_competition_id": comp}
            for cid, nm, comp in rows]


class TestNormalize:
    def test_strips_accents_and_noise_tokens(self):
        assert normalize("Atlético de Madrid") == normalize("Atletico Madrid")
        assert normalize("Arsenal FC") == normalize("Arsenal")

    def test_handles_empty(self):
        assert normalize("") == ""
        assert normalize(None) == ""


class TestDangerousPairsStayApart:
    """Each pair below was produced by string similarity during development
    and is WRONG. They must never map to each other."""

    def test_atletico_madrid_is_not_real_madrid(self):
        clubs = _clubs(("1", "Real Madrid", "ES1"), ("2", "Atlético de Madrid", "ES1"))
        m = build_club_mapping({"Ath Madrid": {"league": "PD"}}, clubs)
        assert m["Ath Madrid"] == "2"

    def test_man_city_is_not_swansea_city(self):
        clubs = _clubs(("1", "Swansea City", "GB1"), ("2", "Manchester City", "GB1"))
        m = build_club_mapping({"Man City": {"league": "PL"}}, clubs)
        assert m["Man City"] == "2"

    def test_universitatea_cluj_is_not_cfr_cluj(self):
        clubs = _clubs(("1", "CFR Cluj", "RO1"), ("2", "FC Universitatea Cluj", "RO1"))
        m = build_club_mapping({"U. Cluj": {"league": "ROU1"}}, clubs)
        assert m["U. Cluj"] == "2"

    def test_hearts_is_not_rangers(self):
        clubs = _clubs(("1", "Rangers FC", "SC1"), ("2", "Heart of Midlothian FC", "SC1"))
        m = build_club_mapping({"Hearts": {"league": "SCO1"}}, clubs)
        assert m["Hearts"] == "2"

    def test_three_brazilian_atleticos_stay_distinct(self):
        clubs = _clubs(("1", "Clube Atlético Mineiro", "BRA1"),
                       ("2", "Clube Atlético Paranaense", "BRA1"))
        m = build_club_mapping(
            {"Atletico-MG": {"league": "BSA"}, "Athletico-PR": {"league": "BSA"}}, clubs)
        assert m["Atletico-MG"] == "1"
        assert m["Athletico-PR"] == "2"

    def test_chivas_is_not_atlas(self):
        clubs = _clubs(("1", "Atlas Guadalajara", "MEX1"),
                       ("2", "Deportivo Guadalajara", "MEX1"))
        m = build_club_mapping({"Guadalajara Chivas": {"league": "MEX1"}}, clubs)
        assert m["Guadalajara Chivas"] == "2"


class TestCorrectButLowSimilarity:
    """These score badly on string similarity yet are genuinely correct —
    they're why a similarity threshold alone can't drive this."""

    @pytest.mark.parametrize("ours,tm,comp,league", [
        ("Wolves", "Wolverhampton Wanderers", "GB1", "PL"),
        ("M'gladbach", "Borussia Mönchengladbach", "L1", "BL1"),
        ("Sp Lisbon", "Sporting CP", "PO1", "PPL"),
        ("OFI Crete", "Omilos Filathlon Irakliou FC", "GR1", "GRE1"),
        ("AEK", "Athlitiki Enosi Konstantinoupoleos", "GR1", "GRE1"),
    ])
    def test_low_similarity_pairs_resolve(self, ours, tm, comp, league):
        m = build_club_mapping({ours: {"league": league}},
                                _clubs(("9", tm, comp), ("8", "Decoy Club", comp)))
        assert m.get(ours) == "9"


class TestSafetyBehaviour:
    def test_unknown_team_is_omitted_not_guessed(self):
        clubs = _clubs(("1", "Some Other Club", "GB1"))
        m = build_club_mapping({"Totally Fictional United": {"league": "PL"}}, clubs)
        assert "Totally Fictional United" not in m

    def test_uncovered_league_is_skipped(self):
        """Lower divisions have no Transfermarkt counterpart; teams there
        must be skipped rather than matched into some other league."""
        clubs = _clubs(("1", "Arsenal FC", "GB1"))
        m = build_club_mapping({"Arsenal": {"league": "ELC2"}}, clubs)
        assert m == {}

    def test_known_absent_teams_are_never_matched(self):
        clubs = _clubs(("1", "Austria Vienna", "A1"))
        m = build_club_mapping({"A. Lustenau": {"league": "AUT1"}}, clubs)
        assert "A. Lustenau" not in m


class TestTableIntegrity:
    def test_manual_and_absent_do_not_overlap(self):
        assert not (set(MANUAL_CLUB_MAP) & KNOWN_ABSENT)

    def test_league_map_has_no_empty_values(self):
        assert all(LEAGUE_TO_TM_COMP.values())


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
