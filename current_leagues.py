"""Current-season league membership from the live football-data.org fixtures API.

Our training data only knows a team's league up to the last completed season,
so a promoted/relegated side shows in its old division (Hull City in the
Championship, when it's in the Premier League for 2026-27). The authoritative
source for THIS season's membership is the live API's /competitions/{code}/teams
endpoint — it returns exactly the current field (promoted in, relegated out).

(Transfermarkt's `domestic_competition_id` was tried and rejected: it retains
recently-relegated clubs, so its "GB1" lists ~37 clubs, not the current 20.)

Run `python current_leagues.py` AFTER a retrain/team_stats refresh: it reassigns
each covered team's league in team_stats.json and writes current_leagues.json
(used by the season simulator and the app's league groupings). Needs API_TOKEN.
Only the domestic competitions the free tier serves are updated; everything else
keeps its last-completed-season label.
"""

import json
import os
import time
import urllib.request
import urllib.error

from config import API_TOKEN, BASE_URL
from team_aliases import resolve_team_name

_DIR = os.path.dirname(os.path.abspath(__file__))

# Our league code -> football-data.org competition code (identical for the
# domestic leagues the free tier covers).
API_LEAGUES = {
    "PL": "PL", "ELC": "ELC", "PD": "PD", "SA": "SA", "BL1": "BL1",
    "FL1": "FL1", "DED": "DED", "PPL": "PPL", "BSA": "BSA",
}


def _resolve(team_stats, name, short):
    """Map an API team to our team_stats key (exact, then verified alias)."""
    if name in team_stats:
        return name
    if short in team_stats:
        return short
    return resolve_team_name(name, team_stats) or resolve_team_name(short, team_stats)


def fetch_current_membership(team_stats: dict, pause: float = 6.5) -> dict:
    """{our_league_code: [our_team_names]} for the current season. `pause`
    keeps us under the free tier's ~10 requests/minute limit."""
    out = {}
    for our_code, api_code in API_LEAGUES.items():
        try:
            req = urllib.request.Request(f"{BASE_URL}/competitions/{api_code}/teams",
                                         headers={"X-Auth-Token": API_TOKEN})
            data = json.loads(urllib.request.urlopen(req, timeout=25).read())
        except (urllib.error.URLError, urllib.error.HTTPError, ValueError) as e:
            print(f"  {api_code}: skipped ({e})")
            continue
        resolved, unresolved = [], []
        for t in data.get("teams", []):
            r = _resolve(team_stats, t.get("name", ""), t.get("shortName", ""))
            (resolved if r else unresolved).append(r or t.get("shortName") or t.get("name"))
        if resolved:
            out[our_code] = sorted(set(resolved))
        print(f"  {api_code}: {len(resolved)} resolved" +
              (f", {len(unresolved)} unresolved: {unresolved}" if unresolved else ""))
        time.sleep(pause)
    return out


def apply_to_team_stats(team_stats: dict, membership: dict) -> int:
    n = 0
    for lg, teams in membership.items():
        for t in teams:
            if t in team_stats and team_stats[t].get("league") != lg:
                team_stats[t]["league"] = lg
                n += 1
    return n


def main():
    with open(os.path.join(_DIR, "team_stats.json"), encoding="utf-8") as f:
        team_stats = json.load(f)
    print("Fetching current-season membership from football-data.org ...")
    membership = fetch_current_membership(team_stats)
    moved = apply_to_team_stats(team_stats, membership)
    with open(os.path.join(_DIR, "team_stats.json"), "w", encoding="utf-8") as f:
        json.dump(team_stats, f, indent=2)
    with open(os.path.join(_DIR, "current_leagues.json"), "w", encoding="utf-8") as f:
        json.dump(membership, f, indent=2)
    print(f"\nReassigned {moved} teams to their current league.")
    print("Sizes:", {k: len(v) for k, v in sorted(membership.items())})


if __name__ == "__main__":
    main()
