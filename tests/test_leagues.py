"""Integrity tests for the canonical league registry."""

import leagues


class TestRegistry:
    def test_ranks_are_contiguous_and_unique(self):
        ranks = sorted(lg.rank for lg in leagues.REGISTRY)
        assert ranks == list(range(1, len(leagues.REGISTRY) + 1))

    def test_premier_league_is_rank_one(self):
        assert leagues.BY_CODE["PL"].rank == 1
        assert leagues.league_rank("PL") == 1

    def test_codes_and_csv_codes_unique(self):
        codes = [lg.code for lg in leagues.REGISTRY]
        csvs = [lg.csv_code for lg in leagues.REGISTRY]
        assert len(codes) == len(set(codes))
        assert len(csvs) == len(set(csvs))

    def test_no_lower_divisions(self):
        # Second/lower-division codes must never appear in the registry.
        lower = {"ELC", "ELC2", "ELC3", "ENG5", "PD2", "SA2", "BL2", "FL2",
                 "SCO2", "SCO3", "SCO4"}
        assert not (lower & leagues.TOP_FLIGHT_CODES)

    def test_scope_is_roughly_top_50(self):
        assert 40 <= len(leagues.REGISTRY) <= 55

    def test_live_api_subset(self):
        live = {lg.code for lg in leagues.REGISTRY if lg.live_api}
        assert live == {"PL", "PD", "SA", "BL1", "FL1", "DED", "PPL", "BSA"}

    def test_every_league_has_a_source(self):
        for lg in leagues.REGISTRY:
            assert lg.source in ("fd", "fdnew", "tm")
            if lg.source == "tm":
                assert lg.tm_code, f"{lg.code} tm source needs a tm_code"
            assert lg.name and lg.country and lg.csv_code


class TestHelpers:
    def test_display_name_resolves_league_and_cup_and_fallback(self):
        assert leagues.display_name("PD") == "La Liga"          # not "Primera Division"
        assert leagues.display_name("CL") == "UEFA Champions League"
        assert leagues.display_name("ZZ9") == "ZZ9"

    def test_unknown_rank_sorts_last(self):
        assert leagues.league_rank("ZZ9") == 999

    def test_fixtures_leagues_is_ordered_live_plus_cups(self):
        fx = list(leagues.fixtures_leagues())
        assert fx[0] == "PL"
        for cup in leagues.CUP_CODES:
            assert cup in fx

    def test_code_by_csv_matches_registry(self):
        assert leagues.CODE_BY_CSV["eng.1"] == "PL"
        assert leagues.CODE_BY_CSV["sau.1"] == "SAU1"


class TestRules:
    def test_default_and_known_rules(self):
        assert leagues.league_rules("PL", 20).cl_slots == 5
        assert leagues.league_rules("ZZ9", 20) == leagues.DEFAULT_RULES._replace(
            relegation_slots=3, cl_slots=4, uel_slots=2, playoff_slots=0)

    def test_rules_clamped_to_field(self):
        r = leagues.league_rules("PL", 8)
        assert r.cl_slots < 8 and r.relegation_slots >= 1
        assert r.cl_slots + r.uel_slots + r.relegation_slots < 8

    def test_tm_result_comps_present(self):
        assert leagues.TM_RESULT_COMPS["sau.1"] == "SA1"
        assert "eng.1" not in leagues.TM_RESULT_COMPS   # fd-sourced, not tm
