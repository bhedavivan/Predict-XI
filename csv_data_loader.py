"""
Football-data.org CSV Data Loader

Loads historical football match data from the footballcsv/cache.footballdata
GitHub repository which contains CSV files for 30+ leagues across multiple seasons.
"""

import csv
import io
import json
import os
import time
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError

from data_processor import process_matches, add_form_features, prepare_training_data, compute_team_stats


# GitHub raw content base URL for footballcsv/cache.footballdata
CSV_BASE_URL = "https://raw.githubusercontent.com/footballcsv/cache.footballdata/master"

# Available seasons in the repository
AVAILABLE_SEASONS = [
    "2019-20", "2020-21", "2021-22", "2022-23", "2023-24", "2024-25"
]

# League code mapping from CSV filename to standard codes.
# Keys MUST match the actual filenames in footballcsv/cache.footballdata
# (verified against the repo — e.g. Spain is "es.1" not "esp.1",
# Scotland is "sco.1" not "sc.1").
LEAGUE_CODE_MAP = {
    # England
    "eng.1": "PL",      # Premier League
    "eng.2": "ELC",     # Championship
    "eng.3": "ELC2",    # League One
    "eng.4": "ELC3",    # League Two
    "eng.5": "ENG5",    # National League
    # Spain
    "es.1": "PD",       # La Liga
    "es.2": "PD2",      # Segunda Division
    # Germany
    "de.1": "BL1",      # Bundesliga
    "de.2": "BL2",      # 2. Bundesliga
    # Italy
    "it.1": "SA",       # Serie A
    "it.2": "SA2",      # Serie B
    # France
    "fr.1": "FL1",      # Ligue 1
    "fr.2": "FL2",      # Ligue 2
    # Other top divisions available in the repo
    "nl.1": "DED",      # Eredivisie
    "pt.1": "PPL",      # Primeira Liga
    "be.1": "BEL1",     # Belgian Pro League
    "tr.1": "TUR1",     # Turkish Super Lig
    "gr.1": "GRE1",     # Greek Super League
    "ru.1": "RUS1",     # Russian Premier League
    "pl.1": "POL1",     # Polish Ekstraklasa
    "at.1": "AUT1",     # Austrian Bundesliga
    "ch.1": "SUI1",     # Swiss Super League
    "dk.1": "DEN1",     # Danish Superliga
    "ro.1": "ROU1",     # Romanian Liga I
    "mx.1": "MEX1",     # Liga MX
    # Scotland
    "sco.1": "SCO1",    # Scottish Premiership
    "sco.2": "SCO2",    # Scottish Championship
    "sco.3": "SCO3",    # Scottish League One
    "sco.4": "SCO4",    # Scottish League Two
}

# Reverse mapping for display
LEAGUE_NAME_MAP = {v: k for k, v in LEAGUE_CODE_MAP.items()}

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
                "competition": {"code": LEAGUE_CODE_MAP.get(league_code, league_code)},
                "season": season,
                "league_code": league_code,
            }
            matches.append(match)
            
        except Exception as e:
            # Skip malformed rows
            continue
    
    return matches


def load_season_league_data(season: str, league_code: str) -> List[Dict[str, Any]]:
    """
    Load and parse match data for a specific season and league.
    
    Args:
        season: Season in format "2023-24"
        league_code: League code like "eng.1", "esp.1", etc.
    
    Returns:
        List of match dictionaries
    """
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
        except:
            pass
    return available


def get_league_display_name(league_code: str) -> str:
    """Get human-readable league name."""
    standard_code = LEAGUE_CODE_MAP.get(league_code, league_code)
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
    
    # Add form features
    rows = add_form_features(rows, n_matches=n_matches)
    print(f"Features added. Total rows: {len(rows)}")
    
    # Prepare training data
    X, y, feature_names = prepare_training_data(rows)
    print(f"Training samples prepared: {len(X)}")
    
    # Compute team stats for prediction
    team_stats = compute_team_stats(rows)
    
    return X, y, feature_names, team_stats


def save_processed_data(
    X: List[List[float]], 
    y: List[int], 
    feature_names: List[str],
    team_stats: Dict[str, Any],
    path: str = "processed_data.json"
):
    """Save processed training data and team stats to JSON file."""
    data = {
        "X": X,
        "y": y,
        "feature_names": feature_names,
        "team_stats": team_stats,
        "saved_at": datetime.utcnow().isoformat() + "Z",
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_processed_data(path: str = "processed_data.json") -> Optional[Dict[str, Any]]:
    """Load processed training data from JSON file."""
    try:
        with open(path) as f:
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
