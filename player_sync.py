"""
Spend the API-Football budget (100 requests/day) on the ONE thing it uniquely
provides: live team news (injuries / suspensions) for imminent fixtures.

Strategy (see current-state.md): never spend a request on results, fixtures,
standings or squad values — those are free elsewhere. Buy only injuries for the
next few days' matches, cache them, and let the /predict live nudge + the track
record read the cache. Two steps:

    python player_sync.py teams      # one-time: map our teams -> API-Football ids (~8 reqs)
    python player_sync.py injuries    # daily: injuries for upcoming fixtures (~8-25 reqs)

The injuries step writes data_cache/player_data.json — the exact file the
FilePlayerDataSource already reads — so the model's serving path adjusts for team
news with no further wiring, and never calls the API itself (that would blow the
budget). A local request cap keeps every run well under 100.

Honest signal: without a per-player value feed (which would need fuzzy name
matching this project rejects), availability is the FRACTION OF THE SQUAD
unavailable — crude per-player, but robust and it catches the depleted-team cases
that actually move a line. `Missing Fixture` counts full, `Questionable` half.
"""

import json
import os
import sys
import time
import urllib.parse
import urllib.request
import urllib.error

from config import load_env
load_env()

import leagues

_HOST = "https://v3.football.api-sports.io"
_TOKEN = os.getenv("PLAYER_API_TOKEN", "")
_DIR = os.path.dirname(os.path.abspath(__file__))
_CACHE = os.path.join(_DIR, "data_cache")
_TEAMS_MAP = os.path.join(_CACHE, "apifootball_teams.json")   # {api_team_id: our_key}
_OUT = os.path.join(_CACHE, "player_data.json")               # read by FilePlayerDataSource

# Our league code -> API-Football league id (their fixed ids; PL verified = 39).
API_LEAGUE_ID = {"PL": 39, "PD": 140, "SA": 135, "BL1": 78, "FL1": 61,
                 "DED": 88, "PPL": 94, "BSA": 71}
# Roughly the size of a matchday-relevant squad — the denominator for the
# "fraction of squad unavailable" signal. Bounded so no single game swings wildly.
_SQUAD_DENOM = 25.0


class _Budget:
    """Local request counter so a run never exceeds the daily cap."""
    def __init__(self, cap):
        self.cap, self.used = cap, 0

    def ok(self):
        return self.used < self.cap


_last_req = [0.0]
_MIN_INTERVAL = 7.0   # free tier caps ~10 requests/minute; ~8.5/min stays safe


def _get(path, params, budget):
    if not _TOKEN:
        raise SystemExit("PLAYER_API_TOKEN not set in .env.")
    if not budget.ok():
        raise RuntimeError("request budget exhausted")
    wait = _MIN_INTERVAL - (time.time() - _last_req[0])
    if wait > 0:
        time.sleep(wait)
    _last_req[0] = time.time()
    url = f"{_HOST}{path}?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"x-apisports-key": _TOKEN})
    budget.used += 1
    try:
        with urllib.request.urlopen(req, timeout=25) as resp:
            data = json.loads(resp.read())
    except (urllib.error.URLError, urllib.error.HTTPError, ValueError) as e:
        print(f"  request failed ({e})")
        return []
    if data.get("errors"):
        print(f"  API error: {data['errors']}")
    return data.get("response", [])


def _season_for(code, iso_date):
    """API-Football season (start year) for a league on a date, matching the
    academic/calendar convention used elsewhere in the project."""
    y = int(iso_date[:4])
    m = int(iso_date[5:7])
    lg = leagues.BY_CODE.get(code)
    if lg and lg.calendar_year:
        return y
    return y if m >= 7 else y - 1


def _load_team_stats():
    with open(os.path.join(_DIR, "team_stats.json"), encoding="utf-8") as f:
        return json.load(f)


# ─── One-time: map API-Football team ids to our team_stats keys ────────────

def build_team_map(cap=90):
    """GET /teams for each live league and resolve API names -> our keys via the
    verified alias table (never a fuzzy guess). ~8 requests, cached forever."""
    from team_aliases import resolve_team_name
    stats = _load_team_stats()
    budget = _Budget(cap)
    mapping = {}
    season = _season_for("PL", time.strftime("%Y-%m-%d"))
    for code, lid in API_LEAGUE_ID.items():
        resp = _get("/teams", {"league": lid, "season": season}, budget)
        hit = 0
        for t in resp:
            team = t.get("team", {})
            api_id, name = team.get("id"), team.get("name", "")
            key = resolve_team_name(name, stats) or (name if name in stats else None)
            if api_id and key:
                mapping[str(api_id)] = key
                hit += 1
        print(f"  {code} (league {lid}): {hit}/{len(resp)} teams mapped")
    os.makedirs(_CACHE, exist_ok=True)
    with open(_TEAMS_MAP, "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2, ensure_ascii=False)
    print(f"\nMapped {len(mapping)} teams -> {os.path.relpath(_TEAMS_MAP)} ({budget.used} requests used)")
    return mapping


def _load_team_map():
    try:
        with open(_TEAMS_MAP, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


# ─── Daily: injuries for imminent fixtures -> player_data.json ─────────────

def sync_injuries(days_ahead=4, cap=80):
    """Fetch injuries for the (league, date) pairs that actually have upcoming
    fixtures (dates come free from football-data.org), aggregate the fraction of
    each squad unavailable, and write player_data.json."""
    from api_client import fetch_upcoming_matches, MissingTokenError
    team_map = _load_team_map()
    if not team_map:
        raise SystemExit("Run `python player_sync.py teams` first.")

    # Which (league, date) pairs have fixtures soon — from the FREE feed.
    cutoff = time.time() + days_ahead * 86400
    pairs = set()
    for code in API_LEAGUE_ID:
        try:
            for m in fetch_upcoming_matches(code):
                d = (m.get("utcDate") or "")[:10]
                if not d:
                    continue
                try:
                    ts = time.mktime(time.strptime(d, "%Y-%m-%d"))
                except ValueError:
                    continue
                if ts <= cutoff:
                    pairs.add((code, d))
        except MissingTokenError:
            raise
        except Exception as e:
            print(f"  {code}: no fixtures ({e})")

    if not pairs:
        print("No upcoming fixtures in the window — nothing to sync.")
        return {}

    budget = _Budget(cap)
    weighted = {}    # our_key -> weighted absence count
    for code, date in sorted(pairs):
        if not budget.ok():
            print("  (request cap reached — stopping)")
            break
        lid = API_LEAGUE_ID[code]
        resp = _get("/injuries", {"league": lid, "season": _season_for(code, date), "date": date}, budget)
        for inj in resp:
            key = team_map.get(str(inj.get("team", {}).get("id")))
            if not key:
                continue
            typ = (inj.get("player", {}) or {}).get("type", "")
            w = 1.0 if "Missing" in typ else (0.5 if "Questionable" in typ else 0.0)
            weighted[key] = weighted.get(key, 0.0) + w
        print(f"  {code} {date}: {len(resp)} injury records")

    availability = {k: {"key_absences": round(v, 1),
                        "value_out_fraction": round(min(0.5, v / _SQUAD_DENOM), 3)}
                    for k, v in weighted.items()}
    out = {"availability": availability, "synced_at": time.strftime("%Y-%m-%d %H:%M"),
           "source": "api-football /injuries"}
    # Preserve any hand-provided "style" block the user dropped in.
    try:
        with open(_OUT, encoding="utf-8") as f:
            out["style"] = json.load(f).get("style", {})
    except (OSError, ValueError):
        pass
    os.makedirs(_CACHE, exist_ok=True)
    with open(_OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nWrote availability for {len(availability)} teams -> {os.path.relpath(_OUT)} "
          f"({budget.used} requests used).")
    return availability


_STYLES = os.path.join(_CACHE, "team_styles.json")   # {season: {our_key: {metrics}}}


def _style_metrics(st):
    """Derive durable play-style metrics from a /teams/statistics response."""
    games = (st.get("fixtures", {}).get("played", {}) or {}).get("total") or 0
    if not games:
        return None
    forms = st.get("lineups") or []
    primary = max(forms, key=lambda f: f.get("played", 0))["formation"] if forms else ""
    try:
        def_shape = int(primary.split("-")[0])          # 3/4/5-at-the-back tendency
    except (ValueError, IndexError):
        def_shape = 4

    def _sum(block):
        return sum((v or {}).get("total") or 0 for v in (block or {}).values())
    cards = _sum(st.get("cards", {}).get("yellow")) + _sum(st.get("cards", {}).get("red"))
    gmin = (st.get("goals", {}).get("for", {}) or {}).get("minute", {}) or {}
    total_for = _sum(gmin)
    late = ((gmin.get("76-90") or {}).get("total") or 0) + ((gmin.get("91-105") or {}).get("total") or 0)
    return {
        "def_shape": def_shape,
        "cards_pg": round(cards / games, 3),
        "late_share": round(late / total_for, 3) if total_for else 0.0,
    }


def sync_styles(league_codes, seasons, cap=95):
    """Collect durable team play-style profiles (formation, discipline, goal
    timing) from /teams/statistics for the given leagues+seasons (must be seasons
    the plan can access — free = 2022-2024). 1 request per team. Writes
    team_styles.json keyed by {season: {our_team_key: metrics}}."""
    from team_aliases import resolve_team_name
    stats = _load_team_stats()
    budget = _Budget(cap)
    try:
        with open(_STYLES, encoding="utf-8") as f:
            out = json.load(f)
    except (OSError, ValueError):
        out = {}
    for code in league_codes:
        lid = API_LEAGUE_ID.get(code)
        if not lid:
            continue
        for season in seasons:
            teams = _get("/teams", {"league": lid, "season": season}, budget)
            if not teams:
                print(f"  {code} {season}: no teams (season not on plan?)")
                continue
            got = 0
            for t in teams:
                if not budget.ok():
                    break
                team = t.get("team", {})
                key = resolve_team_name(team.get("name", ""), stats) or (team.get("name") if team.get("name") in stats else None)
                if not key:
                    continue
                if key in out.get(str(season), {}):
                    continue   # already collected (resume-friendly, saves requests)
                resp = _get("/teams/statistics", {"team": team.get("id"), "league": lid, "season": season}, budget)
                m = _style_metrics(resp) if resp else None
                if m:
                    out.setdefault(str(season), {})[key] = m
                    got += 1
            print(f"  {code} {season}: {got} team styles ({budget.used} reqs used)")
    os.makedirs(_CACHE, exist_ok=True)
    with open(_STYLES, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nWrote styles for {sum(len(v) for v in out.values())} team-seasons -> "
          f"{os.path.relpath(_STYLES)} ({budget.used} requests used)")
    return out


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "injuries"
    if cmd == "teams":
        build_team_map()
    elif cmd == "injuries":
        sync_injuries(int(sys.argv[2]) if len(sys.argv) > 2 else 4)
    elif cmd == "styles":
        codes = sys.argv[2].split(",") if len(sys.argv) > 2 else ["PL", "PD"]
        yrs = [int(y) for y in sys.argv[3].split(",")] if len(sys.argv) > 3 else [2023, 2024]
        sync_styles(codes, yrs)
    else:
        print("usage: python player_sync.py [teams | injuries [days] | styles [codes] [seasons]]")


if __name__ == "__main__":
    main()
