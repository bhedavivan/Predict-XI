import json
import urllib.request
import urllib.error
from config import API_TOKEN, BASE_URL, LEAGUE_CODES


class MissingTokenError(Exception):
    """Raised when the API token is missing or invalid."""
    pass


def _make_request(url: str) -> dict:
    """Make an HTTP GET request with the API token."""
    if not API_TOKEN or API_TOKEN == "your_api_token_here":
        raise MissingTokenError(
            "API token is missing or not set. Please add your football-data.org API token "
            "to the .env file as: API_TOKEN=your_actual_token"
        )
    req = urllib.request.Request(url)
    req.add_header("X-Auth-Token", API_TOKEN)
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8") if e.fp else ""
        if e.code == 403 or e.code == 401:
            raise MissingTokenError(
                f"API authentication failed (HTTP {e.code}). "
                "Your API token may be invalid or expired. "
                "Get a free token at https://www.football-data.org/client/register"
            )
        raise RuntimeError(f"HTTP {e.code}: {e.reason} - {body}")
    except urllib.error.URLError as e:
        raise RuntimeError(f"Network error: {e.reason}. Check your internet connection.")


def fetch_matches(league_code: str, season: str = None) -> list:
    """Fetch matches for a given league code and optional season (year)."""
    if league_code not in LEAGUE_CODES:
        raise ValueError(f"Unknown league code: {league_code}. Valid: {list(LEAGUE_CODES.keys())}")

    url = f"{BASE_URL}/competitions/{league_code}/matches"
    if season:
        url += f"?season={season}"

    data = _make_request(url)
    return data.get("matches", [])


def fetch_standings(league_code: str, season: str = None) -> dict:
    """Fetch standings for a league."""
    url = f"{BASE_URL}/competitions/{league_code}/standings"
    if season:
        url += f"?season={season}"

    return _make_request(url)


def fetch_upcoming_matches(league_code: str) -> list:
    """Fetch upcoming (SCHEDULED/TIMED) matches for a league."""
    if league_code not in LEAGUE_CODES:
        raise ValueError(f"Unknown league code: {league_code}. Valid: {list(LEAGUE_CODES.keys())}")

    all_matches = []
    seen = set()

    for status in ("SCHEDULED", "TIMED"):
        try:
            url = f"{BASE_URL}/competitions/{league_code}/matches?status={status}"
            data = _make_request(url)
            matches = data.get("matches", [])
            for m in matches:
                mid = m.get("id")
                if mid and mid not in seen:
                    seen.add(mid)
                    all_matches.append(m)
        except MissingTokenError:
            raise
        except Exception:
            # If one status fails, try the other
            pass

    # Sort by date
    all_matches.sort(key=lambda m: m.get("utcDate", ""))
    return all_matches