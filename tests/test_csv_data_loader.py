"""Unit tests for csv_data_loader's football-data.co.uk parsing (offline,
no network calls — the fetch/cache functions are exercised separately by
manual smoke tests since they hit a live site)."""

import pytest
from csv_data_loader import (
    parse_csv_matches,
    parse_fd_couk_new_csv,
    _fd_couk_season_code,
    _fd_couk_new_season_label,
    FD_COUK_LEAGUE_MAP,
    FD_COUK_NEW_LEAGUE_MAP,
    LEAGUE_CODE_MAP,
    CALENDAR_YEAR_LEAGUES,
    NEW_LEAGUE_NAME_CANON,
)


NEW_LEAGUES = ["no.1", "se.1", "fi.1", "ie.1", "us.1", "jp.1", "cn.1", "ar.1"]


def _new_feed_csv(rows):
    header = "Country,League,Season,Date,Time,Home,Away,HG,AG,Res\n"
    return header + "".join(rows)


class TestNewCalendarLeagues:
    """The 8 new calendar-year leagues and the Argentina trailing-year fix."""

    def test_all_new_leagues_fully_wired(self):
        for lg in NEW_LEAGUES:
            assert lg in FD_COUK_NEW_LEAGUE_MAP, f"{lg} missing feed map"
            assert lg in LEAGUE_CODE_MAP, f"{lg} missing code map"
            assert lg in CALENDAR_YEAR_LEAGUES, f"{lg} not marked calendar-year"

    def test_argentina_trailing_year_captures_both_label_formats(self):
        # ar.1 mixes legacy split-year ('2016/2017') and calendar ('2017','2020')
        # labels in one file. Season 2016-17 -> target year 2017 must capture the
        # 2016/2017 AND 2017 rows, but never the 2020 row (which belongs to 2019-20).
        csv = _new_feed_csv([
            "Argentina,Liga,2016/2017,10/09/2016,,River,Boca,2,1,H\n",
            "Argentina,Liga,2017,05/03/2017,,Boca,River,0,0,D\n",
            "Argentina,Liga,2020,01/03/2020,,River,Boca,1,2,A\n",
        ])
        got = parse_fd_couk_new_csv(csv, "2016-17", "ar.1")
        assert len(got) == 2  # the two 2017-trailing rows, not the 2020 one
        got2020 = parse_fd_couk_new_csv(csv, "2019-20", "ar.1")
        assert len(got2020) == 1

    def test_name_canonicalization_merges_spelling_variants(self):
        csv = _new_feed_csv([
            "Norway,Elite,2024,01/05/2024,,Ham-Kam,Bodo,1,1,D\n",
        ])
        got = parse_fd_couk_new_csv(csv, "2023-24", "no.1")
        assert got and got[0]["homeTeam"]["name"] == NEW_LEAGUE_NAME_CANON["Ham-Kam"]

    def test_pure_calendar_league_unaffected(self):
        csv = _new_feed_csv([
            "Brazil,Serie A,2024,01/05/2024,,Flamengo,Vasco,2,0,H\n",
        ])
        assert len(parse_fd_couk_new_csv(csv, "2023-24", "br.1")) == 1


FD_COUK_SAMPLE_CSV = (
    "Div,Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR,HS,AS,HST,AST,HC,AC,HF,AF,HY,AY,HR,AR\n"
    "E0,15/08/2025,Liverpool,Bournemouth,4,2,H,19,10,10,3,6,7,7,10,1,2,0,0\n"
    "E0,16/08/2025,Arsenal,Chelsea,1,1,D,12,9,5,4,5,3,8,9,2,1,0,0\n"
)

FD_COUK_NEW_SAMPLE_CSV = (
    "Country,League,Season,Date,Time,Home,Away,HG,AG,Res,PSCH,PSCD,PSCA\n"
    "Mexico,Liga MX,2024/2025,12/07/2024,19:00,Puebla,Atlas,2,3,A,2.1,3.2,3.4\n"
    "Mexico,Liga MX,2025/2026,12/07/2025,19:00,Puebla,Atlas,2,3,A,2.1,3.2,3.4\n"
    "Mexico,Liga MX,2025/2026,13/07/2025,17:00,Leon,Pumas,1,1,D,2.5,3.1,2.9\n"
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


class TestNewLeaguesSeasonLabel:
    def test_season_label_format(self):
        assert _fd_couk_new_season_label("2025-26") == "2025/2026"
        assert _fd_couk_new_season_label("2019-20") == "2019/2020"
        assert _fd_couk_new_season_label("2023-24") == "2023/2024"

    def test_calendar_year_leagues_use_single_year(self):
        """Brazil plays Feb-Dec, so this feed labels its seasons with a
        single year. Passing the split-year label would silently match zero
        rows and look like "no data" rather than a mapping bug."""
        assert _fd_couk_new_season_label("2025-26", "br.1") == "2026"
        assert _fd_couk_new_season_label("2019-20", "br.1") == "2020"

    def test_european_leagues_unaffected_by_league_arg(self):
        assert _fd_couk_new_season_label("2025-26", "pl.1") == "2025/2026"


class TestNewLeaguesMap:
    def test_previously_stale_leagues_now_covered(self):
        for code in ("ru.1", "pl.1", "at.1", "ch.1", "dk.1", "ro.1", "mx.1"):
            assert code in FD_COUK_NEW_LEAGUE_MAP

    def test_brazil_covered(self):
        assert FD_COUK_NEW_LEAGUE_MAP.get("br.1") == "BRA"

    def test_no_overlap_with_main_feed_map(self):
        """A league should never be in both maps -- the main feed is
        preferred and tried first, so an overlap would mean the new-leagues
        entry is silently dead code."""
        assert not (set(FD_COUK_LEAGUE_MAP) & set(FD_COUK_NEW_LEAGUE_MAP))


class TestParseFdCoukNewFormat:
    def test_filters_to_requested_season(self):
        matches = parse_fd_couk_new_csv(FD_COUK_NEW_SAMPLE_CSV, "2025-26", "mx.1")
        assert len(matches) == 2
        assert all(m["season"] == "2025-26" for m in matches)

    def test_parses_results_and_dates(self):
        matches = parse_fd_couk_new_csv(FD_COUK_NEW_SAMPLE_CSV, "2025-26", "mx.1")
        m = matches[0]
        assert m["homeTeam"]["name"] == "Puebla"
        assert m["awayTeam"]["name"] == "Atlas"
        assert m["score"]["fullTime"]["home"] == 2
        assert m["score"]["fullTime"]["away"] == 3
        assert m["utcDate"] == "2025-07-12"
        assert m["competition"]["code"] == "MEX1"
        assert m["stats"] == {}

    def test_draw_included(self):
        matches = parse_fd_couk_new_csv(FD_COUK_NEW_SAMPLE_CSV, "2025-26", "mx.1")
        draw = next(m for m in matches if m["homeTeam"]["name"] == "Leon")
        assert draw["score"]["fullTime"]["home"] == draw["score"]["fullTime"]["away"]

    def test_other_season_excluded(self):
        matches = parse_fd_couk_new_csv(FD_COUK_NEW_SAMPLE_CSV, "2024-25", "mx.1")
        assert len(matches) == 1
        assert matches[0]["season"] == "2024-25"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
