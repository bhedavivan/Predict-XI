"""Unit tests for csv_data_loader's football-data.co.uk parsing (offline,
no network calls — the fetch/cache functions are exercised separately by
manual smoke tests since they hit a live site)."""

import pytest
from csv_data_loader import parse_csv_matches, _fd_couk_season_code, FD_COUK_LEAGUE_MAP


FD_COUK_SAMPLE_CSV = (
    "Div,Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR,HS,AS,HST,AST,HC,AC,HF,AF,HY,AY,HR,AR\n"
    "E0,15/08/2025,Liverpool,Bournemouth,4,2,H,19,10,10,3,6,7,7,10,1,2,0,0\n"
    "E0,16/08/2025,Arsenal,Chelsea,1,1,D,12,9,5,4,5,3,8,9,2,1,0,0\n"
)


class TestSeasonCodeConversion:
    def test_season_code_format(self):
        assert _fd_couk_season_code("2025-26") == "2526"
        assert _fd_couk_season_code("2019-20") == "1920"
        assert _fd_couk_season_code("2023-24") == "2324"


class TestLeagueMap:
    def test_known_leagues_present(self):
        assert FD_COUK_LEAGUE_MAP["eng.1"] == "E0"
        assert FD_COUK_LEAGUE_MAP["es.1"] == "SP1"
        assert FD_COUK_LEAGUE_MAP["sco.1"] == "SC0"

    def test_uncovered_league_absent(self):
        # Leagues football-data.co.uk doesn't carry stay on the footballcsv
        # fallback path — they must NOT be in this map.
        assert "ru.1" not in FD_COUK_LEAGUE_MAP
        assert "mx.1" not in FD_COUK_LEAGUE_MAP


class TestParseFdCoukFormat:
    def test_parses_results_and_dates(self):
        matches = parse_csv_matches(FD_COUK_SAMPLE_CSV, "2025-26", "eng.1")
        assert len(matches) == 2
        m = matches[0]
        assert m["homeTeam"]["name"] == "Liverpool"
        assert m["awayTeam"]["name"] == "Bournemouth"
        assert m["score"]["fullTime"]["home"] == 4
        assert m["score"]["fullTime"]["away"] == 2
        assert m["utcDate"] == "2025-08-15"
        assert m["competition"]["code"] == "PL"

    def test_captures_extra_stats(self):
        matches = parse_csv_matches(FD_COUK_SAMPLE_CSV, "2025-26", "eng.1")
        stats = matches[0]["stats"]
        assert stats["home_shots"] == 19
        assert stats["away_shots"] == 10
        assert stats["home_shots_on_target"] == 10
        assert stats["home_corners"] == 6
        assert stats["home_yellow"] == 1
        assert stats["away_yellow"] == 2

    def test_draw_included(self):
        matches = parse_csv_matches(FD_COUK_SAMPLE_CSV, "2025-26", "eng.1")
        draw = matches[1]
        assert draw["score"]["fullTime"]["home"] == draw["score"]["fullTime"]["away"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
