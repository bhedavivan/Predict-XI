"""
Football-data.org CSV Data Loader

Loads historical football match data from the footballcsv/cache.footballdata
GitHub repository which contains CSV files for 30+ leagues across multiple seasons.
"""

import csv
import io
import json
import os
import re
import time
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError

from data_processor import process_matches, add_form_features, prepare_training_data, compute_team_stats


# GitHub raw content base URL for footballcsv/cache.footballdata
CSV_BASE_URL = "https://raw.githubusercontent.com/footballcsv/cache.footballdata/master"

# football-data.co.uk: actively maintained (unlike the footballcsv mirror above,
# which has no data past 2023-24), and carries richer per-match stats (shots,
# corners, cards) alongside results. Preferred source for any league it covers.
FD_COUK_BASE_URL = "https://www.football-data.co.uk/mmz4281"

# Available seasons. 2024-25/2025-26 only come from football-data.co.uk —
# the footballcsv mirror 404s on both for every league. Extended back to
# 2012-13 (the earliest the "new leagues" feed carries) to roughly double the
# match count and, just as importantly, give Elo / Dixon-Coles ratings a long
# warm-up so they are well-formed by the recent holdout the app is scored on.
# A season/league combo a source doesn't cover just yields no matches (the
# loader falls through and returns []), so extending the range never corrupts
# data — it only adds what exists. Recency weighting (see model_trainer)
# down-weights the older rows so they help rating warm-up and coverage without
# letting a 2013 regime dominate a 2026 prediction.
AVAILABLE_SEASONS = [
    "2012-13", "2013-14", "2014-15", "2015-16", "2016-17", "2017-18", "2018-19",
    "2019-20", "2020-21", "2021-22", "2022-23", "2023-24", "2024-25", "2025-26"
]

# Our "eng.1"-style league code -> football-data.co.uk's own code. Only
# leagues confirmed live (verified 200 OK, seasons 2019-20 through 2025-26)
# are listed; anything not here falls back to the footballcsv mirror, which
# caps out at 2023-24 for those leagues.
FD_COUK_LEAGUE_MAP = {
    "eng.1": "E0", "eng.2": "E1", "eng.3": "E2", "eng.4": "E3", "eng.5": "EC",
    "es.1": "SP1", "es.2": "SP2",
    "de.1": "D1", "de.2": "D2",
    "it.1": "I1", "it.2": "I2",
    "fr.1": "F1", "fr.2": "F2",
    "nl.1": "N1",
    "pt.1": "P1",
    "be.1": "B1",
    "tr.1": "T1",
    "gr.1": "G1",
    "sco.1": "SC0", "sco.2": "SC1", "sco.3": "SC2", "sco.4": "SC3",
}

# Extra per-match stat columns football-data.co.uk provides that the
# footballcsv mirror doesn't. Rolled up into rolling averages in
# data_processor.add_form_features the same way goals are.
FD_COUK_STAT_COLUMNS = {
    "home_shots": "HS", "away_shots": "AS",
    "home_shots_on_target": "HST", "away_shots_on_target": "AST",
    "home_corners": "HC", "away_corners": "AC",
    "home_fouls": "HF", "away_fouls": "AF",
    "home_yellow": "HY", "away_yellow": "AY",
    "home_red": "HR", "away_red": "AR",
}


def _fd_couk_season_code(season: str) -> str:
    """'2025-26' -> '2526' (football-data.co.uk's compact season format)."""
    start, end = season.split("-")
    return start[-2:] + end


# football-data.co.uk's "new leagues" feed — a second, differently-shaped
# feed covering leagues the main mmz4281 feed above doesn't (verified 200 OK
# for all seven). One CSV per country covering *every* season since 2012/13
# (not split per-season like the main feed), with a simpler column set
# (results only, no shots/corners/cards — add_form_features already treats
# those as optional/zero, so that's fine).
FD_COUK_NEW_BASE_URL = "https://www.football-data.co.uk/new"

FD_COUK_NEW_LEAGUE_MAP = {
    "ru.1": "RUS", "pl.1": "POL", "at.1": "AUT", "ch.1": "SWZ",
    "dk.1": "DNK", "ro.1": "ROU", "mx.1": "MEX", "br.1": "BRA",
    # Added leagues (all verified 200 OK, all calendar-year — see below).
    "no.1": "NOR",   # Eliteserien
    "se.1": "SWE",   # Allsvenskan
    "fi.1": "FIN",   # Veikkausliiga
    "ie.1": "IRL",   # League of Ireland Premier Division
    "us.1": "USA",   # MLS
    "jp.1": "JPN",   # J1 League
    "cn.1": "CHN",   # Chinese Super League
    "ar.1": "ARG",   # Argentine Primera (file pools Liga Profesional + Copa de la Liga)
}

# Leagues whose season is NOT Europe's split Aug-May year. For these the feed
# labels seasons by (mostly) a single calendar year. VERIFIED per league by
# inspecting the feed's Season column; a calendar-year league MISSING from this
# set silently ingests zero matches (recurring-bug shape), so every one added
# above is listed here. NB: matching for these leagues is by TRAILING YEAR (see
# parse_fd_couk_new_csv), not exact string — Argentina in particular mixes
# calendar labels ("2020") with legacy split-year labels ("2016/2017") in the
# SAME file, and trailing-year matching captures both without double-counting
# (each Season string's trailing year maps it to exactly one of our slots).
CALENDAR_YEAR_LEAGUES = {
    "br.1", "no.1", "se.1", "fi.1", "ie.1", "us.1", "jp.1", "cn.1", "ar.1",
}

# Hand-verified name canonicalization for the new-leagues feed, which ships a
# few clubs under two spellings (case / hyphen), splitting one club's Elo/form
# history across two team_stats keys. Explicit and verified — NOT fuzzy
# matching (which this project rejects). Extend only after confirming a real
# duplicate in team_stats.json.
NEW_LEAGUE_NAME_CANON = {
    "Ham-Kam": "HamKam",
    "Colon Santa FE": "Colon Santa Fe",
}


def _fd_couk_new_season_label(season: str, league_code: str = "") -> str:
    """'2025-26' -> '2025/2026' (this feed's Season column format), or
    -> '2026' for calendar-year leagues like Brazil.

    For a calendar-year league the label is the *end* year: our "2025-26"
    season spans Aug 2025-May 2026 in Europe, and the Brazilian campaign
    running inside that window is the one played across 2026."""
    start, end = season.split("-")
    if league_code in CALENDAR_YEAR_LEAGUES:
        return f"{start[:2]}{end}"
    return f"{start}/{start[:2]}{end}"

# League code mapping from CSV filename to standard codes.
# Keys MUST match the actual filenames in footballcsv/cache.footballdata
# (verified against the repo — e.g. Spain is "es.1" not "esp.1",
# Scotland is "sco.1" not "sc.1").
# The training league set is the canonical top-flight registry (leagues.py):
# csv_code -> our internal code. This is the single source of truth — lower
# divisions are intentionally absent (the app rates top flights only), and the
# ~21 Transfermarkt-sourced leagues (Saudi, Croatia, K League, …) are included.
import leagues as _leagues

LEAGUE_CODE_MAP = dict(_leagues.CODE_BY_CSV)
LEAGUE_NAME_MAP = {v: k for k, v in LEAGUE_CODE_MAP.items()}
# csv_code -> Transfermarkt gesamtspielplan comp code(s), for the scraped leagues.
_TM_RESULT_COMPS = dict(_leagues.TM_RESULT_COMPS)

# Second divisions used ONLY to warm up team ratings (Elo/Dixon-Coles/pi) so a
# promoted club arrives in the top flight with real form instead of a cold
# 1500 anchor. Loaded alongside the top flights, run through the feature loop
# chronologically, but EXCLUDED from the scored training set (data_processor.
# WARMUP_COMPETITION_CODES). Not in LEAGUE_CODE_MAP (they are not rated leagues).
WARMUP_CODE_MAP = {"eng.2": "ELC", "es.2": "PD2", "de.2": "BL2",
                   "it.2": "SA2", "fr.2": "FL2"}
WARMUP_CSV_CODES = list(WARMUP_CODE_MAP)
# Merged view used only to resolve a match's competition code when parsing.
_CODE_MAP = {**LEAGUE_CODE_MAP, **WARMUP_CODE_MAP}

# Cache directory
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_cache")
CACHE_TTL = 86400  # 24 hours in seconds


def ensure_cache_dir():
    """Ensure cache directory exists."""
    os.makedirs(CACHE_DIR, exist_ok=True)


def get_cache_path(season: str, league_code: str) -> str:
    """Get cache file path for a season/league."""
    return os.path.join(CACHE_DIR, f"{season}_{league_code}.csv")


def is_cache_valid(cache_path: str) -> bool:
    """Check if cache file exists and is not expired."""
    if not os.path.exists(cache_path):
        return False
    age = time.time() - os.path.getmtime(cache_path)
    return age < CACHE_TTL


def fetch_csv_from_github(season: str, league_code: str) -> Optional[str]:
    """
    Fetch CSV data from GitHub raw URL.
    
    Args:
        season: Season in format "2023-24"
        league_code: League code like "eng.1", "esp.1", etc.
    
    Returns:
        CSV content as string, or None if failed
    """
    url = f"{CSV_BASE_URL}/{season}/{league_code}.csv"
    try:
        req = Request(url, headers={'User-Agent': 'soccer-predictor/1.0'})
        with urlopen(req, timeout=30) as response:
            return response.read().decode('utf-8')
    except (URLError, HTTPError, UnicodeDecodeError) as e:
        print(f"Error fetching {url}: {e}")
        return None


def load_csv_from_cache_or_fetch(season: str, league_code: str) -> Optional[str]:
    """
    Load CSV data from cache or fetch from GitHub.
    
    Args:
        season: Season in format "2023-24"
        league_code: League code like "eng.1"
    
    Returns:
        CSV content as string, or None if failed
    """
    ensure_cache_dir()
    cache_path = get_cache_path(season, league_code)
    
    # Try cache first
    if is_cache_valid(cache_path):
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                return f.read()
        except (IOError, UnicodeDecodeError):
            pass  # Fall through to fetch
    
    # Fetch from GitHub
    csv_content = fetch_csv_from_github(season, league_code)
    if csv_content:
        try:
            with open(cache_path, 'w', encoding='utf-8') as f:
                f.write(csv_content)
        except IOError:
            pass  # Cache write failed, but we have the data
    
    return csv_content


def fetch_csv_from_fd_couk(season: str, fd_couk_code: str) -> Optional[str]:
    """Fetch CSV data from football-data.co.uk (results + shots/corners/cards,
    actively maintained through the current season, unlike the footballcsv
    mirror above)."""
    url = f"{FD_COUK_BASE_URL}/{_fd_couk_season_code(season)}/{fd_couk_code}.csv"
    try:
        req = Request(url, headers={'User-Agent': 'soccer-predictor/1.0'})
        with urlopen(req, timeout=30) as response:
            # football-data.co.uk CSVs are latin-1 encoded (accented referee/
            # team names), unlike the UTF-8 footballcsv mirror.
            return response.read().decode('latin-1')
    except (URLError, HTTPError, UnicodeDecodeError) as e:
        print(f"Error fetching {url}: {e}")
        return None


def load_fd_couk_csv_from_cache_or_fetch(season: str, league_code: str, fd_couk_code: str) -> Optional[str]:
    """Cache wrapper for football-data.co.uk, mirroring
    load_csv_from_cache_or_fetch. Cache key is prefixed so it can't collide
    with a footballcsv cache file for the same league_code."""
    ensure_cache_dir()
    cache_path = get_cache_path(season, f"fdcouk_{league_code}")

    if is_cache_valid(cache_path):
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                return f.read()
        except (IOError, UnicodeDecodeError):
            pass

    csv_content = fetch_csv_from_fd_couk(season, fd_couk_code)
    if csv_content:
        try:
            with open(cache_path, 'w', encoding='utf-8') as f:
                f.write(csv_content)
        except IOError:
            pass

    return csv_content


def fetch_csv_from_fd_couk_new(country_code: str) -> Optional[str]:
    """Fetch the whole-history CSV from football-data.co.uk's 'new leagues'
    feed (one file per country, all seasons since 2012/13 — not split per
    season like the main mmz4281 feed)."""
    url = f"{FD_COUK_NEW_BASE_URL}/{country_code}.csv"
    try:
        req = Request(url, headers={'User-Agent': 'soccer-predictor/1.0'})
        with urlopen(req, timeout=30) as response:
            return response.read().decode('latin-1')
    except (URLError, HTTPError, UnicodeDecodeError) as e:
        print(f"Error fetching {url}: {e}")
        return None


def load_fd_couk_new_csv_from_cache_or_fetch(league_code: str, country_code: str) -> Optional[str]:
    """Cache wrapper for the 'new leagues' feed. Cached under a season-less
    key ('all') since one file covers every season for that country."""
    ensure_cache_dir()
    cache_path = get_cache_path("all", f"fdcouk_new_{league_code}")

    if is_cache_valid(cache_path):
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                return f.read()
        except (IOError, UnicodeDecodeError):
            pass

    csv_content = fetch_csv_from_fd_couk_new(country_code)
    if csv_content:
        try:
            with open(cache_path, 'w', encoding='utf-8') as f:
                f.write(csv_content)
        except IOError:
            pass

    return csv_content


def parse_fd_couk_new_csv(csv_content: str, season: str, league_code: str) -> List[Dict[str, Any]]:
    """Parse the 'new leagues' feed's shape:
    Country,League,Season,Date,Time,Home,Away,HG,AG,Res,...odds...
    Filters to the requested season before converting to the same match-dict
    shape parse_csv_matches produces (no per-match stats in this feed)."""
    matches = []
    reader = csv.DictReader(io.StringIO(csv_content))
    season_label = _fd_couk_new_season_label(season, league_code)
    is_calendar = league_code in CALENDAR_YEAR_LEAGUES

    for row in reader:
        try:
            row_season = row.get('Season', '').strip()
            if is_calendar:
                # season_label is the target calendar year (e.g. "2017"). Match
                # by the Season string's TRAILING year so both "2017" and the
                # legacy "2016/2017" map to it — Argentina mixes both formats in
                # one file. Each Season string has one trailing year -> one
                # target, so no match is dropped and none is double-counted.
                m = re.search(r'(\d{4})\s*$', row_season)
                if not m or m.group(1) != season_label:
                    continue
            elif row_season != season_label:
                continue

            date_str = row.get('Date', '').strip()
            home_team = NEW_LEAGUE_NAME_CANON.get(row.get('Home', '').strip(), row.get('Home', '').strip())
            away_team = NEW_LEAGUE_NAME_CANON.get(row.get('Away', '').strip(), row.get('Away', '').strip())
            if not date_str or not home_team or not away_team:
                continue

            try:
                home_score = int(row.get('HG', '') or '')
                away_score = int(row.get('AG', '') or '')
            except ValueError:
                continue

            if row.get('Res', '').strip() not in ('H', 'A', 'D'):
                continue

            match_date = None
            for fmt in ('%d/%m/%Y', '%d/%m/%y'):
                try:
                    match_date = datetime.strptime(date_str, fmt).date().isoformat()
                    break
                except ValueError:
                    continue
            if not match_date:
                continue

            matches.append({
                "utcDate": match_date,
                "status": "FINISHED",
                "homeTeam": {"name": home_team},
                "awayTeam": {"name": away_team},
                "score": {"fullTime": {"home": home_score, "away": away_score}},
                "competition": {"code": _CODE_MAP.get(league_code, league_code)},
                "season": season,
                "league_code": league_code,
                "stats": {},
            })
        except Exception:
            continue

    return matches


def parse_csv_matches(csv_content: str, season: str, league_code: str) -> List[Dict[str, Any]]:
    """
    Parse CSV content into match dictionaries compatible with data_processor.
    
    Supports two formats:
    1. football-data.org format: Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR,HTHG,HTAG,HTR,...
    2. footballcsv/cache.footballdata format: Date,Team 1,FT,HT,Team 2
    """
    matches = []
    reader = csv.DictReader(io.StringIO(csv_content))
    
    # Detect format by checking column names
    fieldnames = reader.fieldnames or []
    
    # Check if it's the footballcsv format (Date,Team 1,FT,HT,Team 2)
    is_footballcsv_format = 'Team 1' in fieldnames and 'FT' in fieldnames and 'Team 2' in fieldnames
    
    for row in reader:
        try:
            stats = {}
            if is_footballcsv_format:
                # Parse footballcsv format: Date,Team 1,FT,HT,Team 2
                date_str = row.get('Date', '').strip()
                home_team = row.get('Team 1', '').strip()
                away_team = row.get('Team 2', '').strip()
                ft_score = row.get('FT', '').strip()
                
                if not date_str or not home_team or not away_team or not ft_score:
                    continue
                
                # Parse FT score (format: "2-1" or "0-3")
                try:
                    home_score_str, away_score_str = ft_score.split('-')
                    home_score = int(home_score_str)
                    away_score = int(away_score_str)
                except (ValueError, AttributeError):
                    continue
                
                # Determine result
                if home_score > away_score:
                    result = 'H'
                elif home_score < away_score:
                    result = 'A'
                else:
                    result = 'D'
            else:
                # Parse football-data.org format
                date_str = row.get('Date', '').strip()
                if not date_str:
                    continue
                
                home_team = row.get('HomeTeam', '').strip()
                away_team = row.get('AwayTeam', '').strip()
                
                if not home_team or not away_team:
                    continue
                
                # Parse scores
                try:
                    home_score = int(row.get('FTHG', 0) or 0)
                    away_score = int(row.get('FTAG', 0) or 0)
                except ValueError:
                    continue
                
                # Determine result
                if home_score > away_score:
                    result = 'H'
                elif home_score < away_score:
                    result = 'A'
                else:
                    result = 'D'
                
                # Only include finished matches
                if row.get('FTR', '').strip() not in ('H', 'A', 'D'):
                    continue

                # Extra match stats (football-data.co.uk only — absent/blank
                # on other football-data.org-shaped sources, which is fine,
                # add_form_features treats missing stats as 0).
                for feature_key, col in FD_COUK_STAT_COLUMNS.items():
                    raw = row.get(col, '').strip()
                    if raw:
                        try:
                            stats[feature_key] = int(raw)
                        except ValueError:
                            pass
            
            # Parse date (format: DD/MM/YY or DD/MM/YYYY or "Fri Aug 11 2023")
            match_date = None
            for fmt in ('%d/%m/%y', '%d/%m/%Y', '%Y-%m-%d', '%a %b %d %Y'):
                try:
                    match_date = datetime.strptime(date_str, fmt).date().isoformat()
                    break
                except ValueError:
                    continue
            
            if not match_date:
                continue
            
            match = {
                "utcDate": match_date,
                "status": "FINISHED",
                "homeTeam": {"name": home_team},
                "awayTeam": {"name": away_team},
                "score": {
                    "fullTime": {
                        "home": home_score,
                        "away": away_score
                    }
                },
                "competition": {"code": _CODE_MAP.get(league_code, league_code)},
                "season": season,
                "league_code": league_code,
                "stats": stats,
            }
            matches.append(match)
            
        except Exception as e:
            # Skip malformed rows
            continue
    
    return matches


def _tm_saison_id(season: str, calendar_year: bool) -> int:
    """Map our internal season string to a Transfermarkt saison_id (the season's
    start year). Calendar-year leagues (South American, K League, …) use the end
    year, mirroring the football-data.co.uk new-feed calendar handling: internal
    '2023-24' -> saison_id 2024 for a calendar league, 2023 otherwise."""
    start = int(season[:4])
    return start + 1 if calendar_year else start


def load_tm_results(season: str, league_code: str, tm_code: str) -> List[Dict[str, Any]]:
    """Match results for a Transfermarkt-sourced league-season, as normalized
    match dicts (the same shape the CSV parsers emit). Reads the scraper's
    on-disk cache; scrapes on a miss (polite, cached)."""
    from transfermarkt_scraper import results_for
    lg = _leagues.BY_CSV.get(league_code)
    year = _tm_saison_id(season, bool(lg and lg.calendar_year))
    try:
        rows = results_for(tm_code, year)
    except Exception:
        return []
    our_code = _CODE_MAP.get(league_code, league_code)
    out = []
    for date_iso, home, away, hg, ag in rows:
        out.append({
            "utcDate": date_iso,
            "status": "FINISHED",
            "homeTeam": {"name": home},
            "awayTeam": {"name": away},
            "score": {"fullTime": {"home": int(hg), "away": int(ag)}},
            "competition": {"code": our_code},
            "season": season,
            "league_code": league_code,
            "stats": {},
        })
    return out


def load_season_league_data(season: str, league_code: str) -> List[Dict[str, Any]]:
    """
    Load and parse match data for a specific season and league.
    
    Args:
        season: Season in format "2023-24"
        league_code: League code like "eng.1", "esp.1", etc.
    
    Returns:
        List of match dictionaries
    """
    fd_couk_code = FD_COUK_LEAGUE_MAP.get(league_code)
    if fd_couk_code:
        csv_content = load_fd_couk_csv_from_cache_or_fetch(season, league_code, fd_couk_code)
        if csv_content:
            return parse_csv_matches(csv_content, season, league_code)
        # football-data.co.uk had nothing for this season/league (e.g. a
        # season before either source existed) — fall through.

    new_country_code = FD_COUK_NEW_LEAGUE_MAP.get(league_code)
    if new_country_code:
        csv_content = load_fd_couk_new_csv_from_cache_or_fetch(league_code, new_country_code)
        if csv_content:
            matches = parse_fd_couk_new_csv(csv_content, season, league_code)
            if matches:
                return matches
        # Nothing for this season in the new-leagues feed either — fall
        # through to the stale-but-still-useful footballcsv mirror.

    # Leagues with no free bulk feed (Saudi, Croatia, K League, …) are scraped
    # from Transfermarkt's gesamtspielplan (results). The polite scraper caches
    # each season's HTML on disk, so training re-reads cache instantly.
    tm_code = _TM_RESULT_COMPS.get(league_code)
    if tm_code:
        matches = load_tm_results(season, league_code, tm_code)
        if matches:
            return matches

    csv_content = load_csv_from_cache_or_fetch(season, league_code)
    if not csv_content:
        return []

    return parse_csv_matches(csv_content, season, league_code)


def load_multiple_seasons_leagues(
    seasons: List[str], 
    league_codes: List[str],
    progress_callback: Optional[callable] = None
) -> List[Dict[str, Any]]:
    """
    Load match data for multiple seasons and leagues.
    
    Args:
        seasons: List of seasons like ["2022-23", "2023-24"]
        league_codes: List of league codes like ["eng.1", "esp.1"]
        progress_callback: Optional callback(season, league, count) for progress updates
    
    Returns:
        Combined list of all matches
    """
    all_matches = []
    
    for season in seasons:
        for league_code in league_codes:
            matches = load_season_league_data(season, league_code)
            all_matches.extend(matches)
            
            if progress_callback:
                progress_callback(season, league_code, len(matches))
    
    # Sort by date
    all_matches.sort(key=lambda m: m.get("utcDate", ""))
    return all_matches


def get_available_leagues_for_season(season: str) -> List[str]:
    """
    Get list of available league codes for a season by checking GitHub.
    This is a best-effort approach - we try known leagues.
    """
    available = []
    for league_code in LEAGUE_CODE_MAP.keys():
        # Quick check if file exists (HEAD request)
        url = f"{CSV_BASE_URL}/{season}/{league_code}.csv"
        try:
            req = Request(url, method='HEAD', headers={'User-Agent': 'soccer-predictor/1.0'})
            with urlopen(req, timeout=10) as response:
                if response.status == 200:
                    available.append(league_code)
        except (URLError, HTTPError, TimeoutError):
            pass
    return available


def get_league_display_name(league_code: str) -> str:
    """Get human-readable league name."""
    standard_code = _CODE_MAP.get(league_code, league_code)
    return standard_code


def load_and_process_data(
    seasons: List[str],
    league_codes: List[str],
    n_matches: int = 5
) -> Tuple[List[List[float]], List[int], List[str], Dict[str, Any]]:
    """
    Load CSV data for multiple seasons/leagues, process it, and prepare training data.
    
    Returns:
        Tuple of (X, y, feature_names, team_stats)
    """
    print(f"Loading data for seasons: {seasons}, leagues: {league_codes}")
    
    # Load all matches
    all_matches = load_multiple_seasons_leagues(seasons, league_codes)
    print(f"Total matches loaded: {len(all_matches)}")
    
    if not all_matches:
        return [], [], [], {}
    
    # Process matches
    rows = process_matches(all_matches)
    print(f"Finished matches: {len(rows)}")
    
    if not rows:
        return [], [], [], {}
    
    # Squad market values (Transfermarkt). club_mapping only needs each
    # team's league, which is available straight from the raw rows — building
    # it here avoids running the expensive feature pass twice just to learn
    # team->league. Any failure here is non-fatal: the squad-value features
    # fall back to 0 with has_squad_value=0 and everything else still trains.
    squad_values, club_map = None, None
    try:
        import transfermarkt_data
        import club_mapping
        # Tag each team with the most-recent league that Transfermarkt actually
        # COVERS (falling back to its latest league if none). Transfermarkt has
        # only first divisions, and a club's TM record sits in whichever top
        # flight it belongs to; a promoted/relegated club that has ever been
        # top-flight should still match (its point-in-time value_at then handles
        # each date). The old first-seen `setdefault` stranded such clubs in a
        # stale lower division with no TM comp (Monaco was in Ligue 2 in 2012-13,
        # Leeds in the Championship until 2020) so they never matched despite
        # being in TM; picking the covered league recovers ~40 top-flight clubs.
        from collections import defaultdict as _dd
        _covered = set(club_mapping.LEAGUE_TO_TM_COMP)
        _seq = _dd(list)
        for r in rows:  # chronological
            comp = r.get("competition", "")
            _seq[r["home_team"]].append(comp)
            _seq[r["away_team"]].append(comp)
        team_leagues = {}
        for _team, _cs in _seq.items():
            _cov = [c for c in _cs if c in _covered]
            team_leagues[_team] = {"league": _cov[-1] if _cov else _cs[-1]}
        _tm_clubs = transfermarkt_data.load_table("clubs")
        club_map = club_mapping.build_club_mapping(team_leagues, _tm_clubs)
        squad_values = transfermarkt_data.get_index()
        covered = sum(1 for t in team_leagues if t in club_map)
        print(f"Squad values: {len(club_map)} clubs mapped "
              f"({covered}/{len(team_leagues)} teams matched)")
    except Exception as e:  # noqa: BLE001 - optional enrichment, never fatal
        print(f"Squad values unavailable ({type(e).__name__}: {e}); "
              f"continuing without them")
        squad_values, club_map = None, None

    # ClubElo cross-league ratings (free, keyless). Same optional-enrichment
    # contract: any failure falls back to has_clubelo=0 and everything else
    # still trains. Only European leagues are covered.
    clubelo = None
    try:
        import clubelo_data
        clubelo = clubelo_data.get_index()
        print(f"ClubElo: index built ({len(clubelo.snapshots)} countries)")
    except Exception as e:  # noqa: BLE001 - optional enrichment, never fatal
        print(f"ClubElo unavailable ({type(e).__name__}: {e}); continuing without it")
        clubelo = None

    # Player-performance stats (Transfermarkt appearances) — resolved via the
    # same club_map as squad value. Same optional-enrichment contract.
    pstats = None
    try:
        import player_stats
        pstats = player_stats.get_index()
        print(f"Player stats: index built ({len(pstats.series)} club series)")
    except Exception as e:  # noqa: BLE001 - optional enrichment, never fatal
        print(f"Player stats unavailable ({type(e).__name__}: {e}); continuing without them")
        pstats = None

    # Add form features
    rows = add_form_features(rows, n_matches=n_matches,
                              squad_values=squad_values, club_map=club_map,
                              clubelo=clubelo, pstats=pstats)
    print(f"Features added. Total rows: {len(rows)}")

    # Prepare training data
    X, y, feature_names, row_meta = prepare_training_data(rows, return_meta=True)
    print(f"Training samples prepared: {len(X)}")

    # Compute team stats for prediction
    team_stats = compute_team_stats(rows, squad_values=squad_values, club_map=club_map,
                                     clubelo=clubelo, pstats=pstats)
    # NOTE: current-season league reassignment (promoted/relegated teams) is a
    # SEPARATE post-step — see current_leagues.py, which uses the live
    # football-data.org fixtures API. Transfermarkt's domestic_competition_id
    # was tried and rejected: it retains recently-relegated clubs (so GB1 lists
    # ~37 clubs, not the current 20), which would mislabel them as top-flight.

    # Per-row leagues are exposed as an attribute rather than a 5th return
    # value so the long-standing 4-tuple contract (main.py and the training
    # scripts all unpack it) keeps working. Callers that persist data read
    # this and hand it to save_processed_data(leagues=...).
    load_and_process_data.last_leagues = [m["league"] for m in row_meta]
    load_and_process_data.last_dates = [m["date"] for m in row_meta]
    load_and_process_data.last_h2h = getattr(add_form_features, "last_h2h", {})

    return X, y, feature_names, team_stats


def save_processed_data(
    X: List[List[float]], 
    y: List[int], 
    feature_names: List[str],
    team_stats: Dict[str, Any],
    path: str = "processed_data.json",
    leagues: Optional[List[str]] = None,
    dates: Optional[List[str]] = None,
):
    """Save processed training data and team stats to JSON file.

    `leagues` is parallel to X and lets evaluate.py break results down per
    competition — a single global accuracy across 30 leagues hides how
    differently the model performs in each. `dates` (also parallel to X) lets
    the trainer compute recency weights and lets evaluate.py reason about the
    holdout's time span.
    """
    data = {
        "X": X,
        "y": y,
        "feature_names": feature_names,
        "team_stats": team_stats,
        "leagues": leagues or [],
        "dates": dates or [],
        "saved_at": datetime.utcnow().isoformat() + "Z",
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_processed_data(path: str = "processed_data.json") -> Optional[Dict[str, Any]]:
    """Load processed training data from JSON file."""
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


if __name__ == "__main__":
    # Test loading a single league/season
    import sys
    
    season = sys.argv[1] if len(sys.argv) > 1 else "2023-24"
    league = sys.argv[2] if len(sys.argv) > 2 else "eng.1"
    
    print(f"Testing load for {season} {league}...")
    matches = load_season_league_data(season, league)
    print(f"Loaded {len(matches)} matches")
    
    if matches:
        print(f"First match: {matches[0]}")
        print(f"Last match: {matches[-1]}")
