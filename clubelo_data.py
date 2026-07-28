"""ClubElo cross-league club ratings (http://api.clubelo.com, free, keyless,
attribution-only ToS).

Why: this project's own Elo is calibrated WITHIN a league — teams from
different leagues only ever mix at the shared 1500 anchor — yet the app's whole
purpose is arbitrary cross-league matchups. ClubElo ratings are cross-
competition by construction, so they fill that named blind spot and, as a
bonus, a just-promoted or newly-covered club carries its real strength in
instead of a cold-start 1500.

Point-in-time, no lookahead: monthly snapshots are taken from ClubElo's dated
feed and a match only ever reads the snapshot for a month strictly on/before
its own — same discipline as the squad-value index. Coverage is European only;
our non-European leagues (Argentina, Japan, China, USA, Mexico, Brazil) have no
ClubElo counterpart and get has_clubelo=0.

Matching is league(country)-constrained normalized exact/containment plus a
small hand-verified override table — NEVER fuzzy similarity (this project
rejects that; unmapped is safer than mis-mapped).
"""

import csv
import io
import os
import re
import time
import unicodedata
from collections import defaultdict
from typing import Dict, Optional, Tuple
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError

CLUBELO_BASE = "http://api.clubelo.com"
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_cache")
CACHE_TTL = 3 * 86400  # snapshots are historical; only the latest month changes

# Our LEAGUE_CODE -> ClubElo Country code. Only European leagues ClubElo covers;
# a club is unique within its country regardless of tier (ClubElo tracks it
# continuously through promotion/relegation), so tier isn't needed here.
LEAGUE_TO_CLUBELO_COUNTRY = {
    "PL": "ENG", "ELC": "ENG", "ELC2": "ENG", "ELC3": "ENG", "ENG5": "ENG",
    "PD": "ESP", "PD2": "ESP",
    "SA": "ITA", "SA2": "ITA",
    "BL1": "GER", "BL2": "GER",
    "FL1": "FRA", "FL2": "FRA",
    "DED": "NED", "PPL": "POR", "BEL1": "BEL", "TUR1": "TUR", "GRE1": "GRE",
    "RUS1": "RUS", "POL1": "POL", "AUT1": "AUT", "SUI1": "SUI", "DEN1": "DEN",
    "ROU1": "ROU", "SCO1": "SCO", "SCO2": "SCO", "SCO3": "SCO", "SCO4": "SCO",
    "NOR1": "NOR", "SWE1": "SWE",
}

# Hand-verified our_name -> ClubElo club name, for clubs the normalized
# exact/containment matcher misses. Checked against the country's ClubElo roster
# by hand; never add an entry from a similarity score alone.
CLUBELO_MANUAL = {
    ("ENG", "Nott'm Forest"): "Forest",
    ("ENG", "Man United"): "Man United",
    ("ENG", "Wolves"): "Wolves",
    ("ENG", "Sheffield United"): "Sheffield Utd",
    ("ENG", "West Brom"): "West Brom",
    ("ESP", "Ath Madrid"): "Atletico",
    ("ESP", "Ath Bilbao"): "Bilbao",
    ("ESP", "Espanol"): "Espanyol",
    ("ESP", "Sociedad"): "Sociedad",
    ("ESP", "Vallecano"): "Rayo Vallecano",
    ("ESP", "Betis"): "Betis",
    ("ESP", "Celta"): "Celta",
    ("GER", "Ein Frankfurt"): "Frankfurt",
    ("GER", "M'gladbach"): "Gladbach",
    ("GER", "Leverkusen"): "Leverkusen",
    ("GER", "Dortmund"): "Dortmund",
    ("GER", "Bayern Munich"): "Bayern",
    ("ITA", "Verona"): "Verona",
    ("FRA", "Paris SG"): "Paris SG",
    ("FRA", "Rennes"): "Rennes",
    ("POR", "Sp Lisbon"): "Sporting",
    ("NED", "PSV Eindhoven"): "PSV",
    ("NED", "For Sittard"): "Fortuna Sittard",
    ("SCO", "Hearts"): "Hearts",
    ("SCO", "Rangers"): "Rangers",
    ("SCO", "Celtic"): "Celtic",
}


def _normalize(name: str) -> str:
    s = unicodedata.normalize("NFKD", name or "").encode("ascii", "ignore").decode()
    s = s.lower().replace("&", " and ").replace("-", " ").replace(".", " ")
    s = re.sub(r"\b(fc|afc|cf|sc|cd|ac|sv|ss|as|us|rc|sk|club|the)\b", " ", s)
    return re.sub(r"[^a-z0-9]", "", s)


def _months(start_year: int = 2012, end_year: int = 2026):
    for y in range(start_year, end_year + 1):
        for m in range(1, 13):
            yield f"{y}-{m:02d}-01"


def _fetch_snapshot(date_str: str) -> Optional[str]:
    url = f"{CLUBELO_BASE}/{date_str}"
    try:
        req = Request(url, headers={"User-Agent": "predict-xi/1.0 (+attribution: clubelo.com)"})
        with urlopen(req, timeout=30) as resp:
            return resp.read().decode("utf-8", errors="replace")
    except (URLError, HTTPError, OSError) as e:
        print(f"ClubElo fetch failed {url}: {e}")
        return None


def _cached_snapshot(date_str: str) -> Optional[str]:
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, f"clubelo_{date_str[:7]}.csv")
    # Historical months never change; only refresh a month younger than the TTL.
    if os.path.exists(path):
        fresh = (time.time() - os.path.getmtime(path)) < CACHE_TTL
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            if content and (fresh or date_str[:7] < time.strftime("%Y-%m")):
                return content
        except OSError:
            pass
    content = _fetch_snapshot(date_str)
    if content:
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
        except OSError:
            pass
    return content


class ClubEloIndex:
    """Month-granularity club Elo, keyed by (country, normalized name).

    snapshots[country][norm_name] = {month_key: elo}; a lookup for a match date
    returns the value from the latest month on/before it (no lookahead)."""

    def __init__(self):
        # country -> norm_name -> {"YYYY-MM": elo}
        self.snapshots: Dict[str, Dict[str, Dict[str, float]]] = defaultdict(lambda: defaultdict(dict))
        self.current: Dict[str, Dict[str, float]] = defaultdict(dict)
        # country -> {norm_name: original ClubElo display name} (for debugging)
        self.display: Dict[str, Dict[str, str]] = defaultdict(dict)
        self._resolve_cache: Dict[Tuple[str, str], Optional[str]] = {}

    def build(self, start_year: int = 2012, end_year: int = 2026):
        latest_month = None
        for date_str in _months(start_year, end_year):
            if date_str[:7] > time.strftime("%Y-%m"):
                break  # don't fetch future months
            content = _cached_snapshot(date_str)
            if not content:
                continue
            mk = date_str[:7]
            latest_month = mk
            for row in csv.DictReader(io.StringIO(content)):
                country = (row.get("Country") or "").strip()
                club = (row.get("Club") or "").strip()
                elo = (row.get("Elo") or "").strip()
                if not country or not club or not elo:
                    continue
                try:
                    elo_f = float(elo)
                except ValueError:
                    continue
                nn = _normalize(club)
                if not nn:
                    continue
                self.snapshots[country][nn][mk] = elo_f
                self.display[country][nn] = club
        # Present-day values = the last month seen.
        if latest_month:
            for country, clubs in self.snapshots.items():
                for nn, months in clubs.items():
                    if latest_month in months:
                        self.current[country][nn] = months[latest_month]
                    elif months:
                        self.current[country][nn] = months[max(months)]
        return self

    def _resolve(self, team: str, league_code: str) -> Optional[Tuple[str, str]]:
        country = LEAGUE_TO_CLUBELO_COUNTRY.get(league_code)
        if not country or country not in self.snapshots:
            return None
        key = (country, team)
        if key in self._resolve_cache:
            nn = self._resolve_cache[key]
            return (country, nn) if nn else None

        manual = CLUBELO_MANUAL.get((country, team))
        clubs = self.snapshots[country]
        nn = None
        if manual:
            mnn = _normalize(manual)
            if mnn in clubs:
                nn = mnn
        if nn is None:
            n = _normalize(team)
            if n in clubs:
                nn = n
            else:
                # unique containment either direction
                contained = [c for c in clubs if n and c and (n in c or c in n)]
                if len(contained) == 1:
                    nn = contained[0]
        self._resolve_cache[key] = nn
        return (country, nn) if nn else None

    def elo_at(self, team: str, league_code: str, on_date: str) -> Optional[float]:
        """Club Elo as known strictly on/before `on_date` (YYYY-MM-DD)."""
        res = self._resolve(team, league_code)
        if not res or not on_date:
            return None
        country, nn = res
        months = self.snapshots[country].get(nn)
        if not months:
            return None
        mk = on_date[:7]
        if mk in months:
            return months[mk]
        earlier = [m for m in months if m <= mk]
        return months[max(earlier)] if earlier else None

    def current_elo(self, team: str, league_code: str) -> Optional[float]:
        res = self._resolve(team, league_code)
        if not res:
            return None
        country, nn = res
        return self.current.get(country, {}).get(nn)


_INDEX: Optional[ClubEloIndex] = None


def get_index(force_reload: bool = False) -> ClubEloIndex:
    global _INDEX
    if _INDEX is None or force_reload:
        _INDEX = ClubEloIndex().build()
    return _INDEX


# --- Feature block (shared by training and serving, like squad_value_feature_dict) ---

def clubelo_feature_dict(home_elo: Optional[float], away_elo: Optional[float]) -> dict:
    """Build the ClubElo feature block from two raw ratings. has_clubelo is 0
    unless BOTH sides are known — a one-sided cross-league rating is misleading,
    the same rule the squad-value block uses."""
    both = home_elo is not None and away_elo is not None
    h = float(home_elo) if home_elo is not None else 0.0
    a = float(away_elo) if away_elo is not None else 0.0
    diff = (h - a) if both else 0.0
    expected = 1.0 / (1.0 + 10 ** (-diff / 400.0)) if both else 0.5
    return {
        "home_clubelo": h if both else 0.0,
        "away_clubelo": a if both else 0.0,
        "clubelo_diff": diff,
        "clubelo_expected": expected,
        "has_clubelo": 1.0 if both else 0.0,
    }
