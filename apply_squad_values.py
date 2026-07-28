"""Attach self-scraped second-division squad values to team_stats.json.

Serving-only enrichment, in the same family as current_leagues.py: it fills the
squad_value_eur that the CC0 dataset can't provide for second/lower divisions
(Championship, Serie B, Ligue 2, 2. Bundesliga, Segunda, League One/Two), so the
app, predictions and season simulator show the SAME data for those clubs as for
top-flight ones. Run AFTER a retrain/team_stats refresh (a retrain regenerates
team_stats.json from the CC0 data and would drop these); pairs with:

    python transfermarkt_scraper.py     # (re)build data_cache/tm_scraped_values.json
    python apply_squad_values.py        # write the values into team_stats.json

Matching reuses the vetted, league-constrained matcher (club_mapping.match_team_name):
exact / unique-containment / hand-verified manual / >=0.90 similarity — never a
loose guess. Candidates are pooled by COUNTRY, not by exact division, because a
club's tier changes season to season (a side we label "League One" may be scraped
in the Championship); pooling England's GB2/GB3/GB4 together fixes that while
keeping countries isolated so names can't collide across borders.
"""

import json
import os

from club_mapping import match_team_name
from transfermarkt_scraper import load_scraped_values

_DIR = os.path.dirname(os.path.abspath(__file__))

# Our second/lower-division league code -> the TM competition codes to draw
# candidates from. English tiers share a pool (clubs move between them); the
# others are 1:1 with the single second division we scrape.
COUNTRY_POOL = {
    "ELC": ("GB2", "GB3", "GB4"), "ELC2": ("GB2", "GB3", "GB4"), "ELC3": ("GB2", "GB3", "GB4"),
    "PD2": ("ES2",), "SA2": ("IT2",), "BL2": ("L2",), "FL2": ("FR2",),
}

# Hand-verified second-division overrides, kept SEPARATE from club_mapping's
# CC0 map so they can never affect the first-division matching. our_name -> exact
# Transfermarkt name, each confirmed present in the scraped pool. Needed where
# our short name and TM's differ beyond containment — notably TM.co.uk's English
# exonyms (Nürnberg -> "1.FC Nuremberg").
SECOND_DIV_MANUAL = {
    "QPR": "Queens Park Rangers",
    "Sheffield Weds": "Sheffield Wednesday",
    "Bristol Rvs": "Bristol Rovers",
    "Nurnberg": "1.FC Nuremberg",
}


def match_scraped_values(team_stats: dict, store: dict, only_missing: bool = True) -> dict:
    """Return {our_team_name: value_eur} for second-division clubs resolved to a
    scraped club. `store` is transfermarkt_scraper's output."""
    # Candidate list per country pool: [(club_id, tm_name)].
    pools = {}
    for comps in {v for v in COUNTRY_POOL.values()}:
        pools[comps] = [(cid, rec["name"]) for cid, rec in store.items()
                        if rec.get("comp") in comps]

    out = {}
    for team, info in team_stats.items():
        lg = info.get("league", "")
        if lg not in COUNTRY_POOL or info.get("matches_played", 0) <= 0:
            continue
        if only_missing and info.get("squad_value_eur"):
            continue
        pool = pools[COUNTRY_POOL[lg]]
        override = SECOND_DIV_MANUAL.get(team)
        if override:
            cid = next((cid for cid, nm in pool if nm == override), None)
        else:
            cid = match_team_name(team, pool)
        if not cid:
            continue
        val = store.get(cid, {}).get("current")
        if val:
            out[team] = float(val)
    return out


def apply_new_league_values(team_stats: dict) -> int:
    """Fill squad values for the Transfermarkt-sourced top-flight leagues (Saudi,
    K-League, Egypt, …) that the CC0 dataset doesn't cover. Their team_stats keys
    ARE the TM club names, so this is a safe exact-name lookup — no fuzzy match.
    Returns the number of clubs filled."""
    from transfermarkt_scraper import load_new_league_values
    import leagues
    vals = load_new_league_values()
    if not vals:
        return 0
    tm_league_codes = {lg.code for lg in leagues.REGISTRY if lg.source == "tm"}
    n = 0
    for team, s in team_stats.items():
        if s.get("league") in tm_league_codes and s.get("matches_played", 0) > 0:
            v = vals.get(team)
            if v:
                s["squad_value_eur"] = float(v)
                n += 1
    return n


def main():
    with open(os.path.join(_DIR, "team_stats.json"), encoding="utf-8") as f:
        team_stats = json.load(f)

    # Second divisions (only present if the app still trains them).
    store = load_scraped_values()
    values = match_scraped_values(team_stats, store, only_missing=False) if store else {}
    for team, val in values.items():
        team_stats[team]["squad_value_eur"] = val

    # New top-flight leagues (Saudi, K-League, …) — exact TM-name lookup.
    n_new = apply_new_league_values(team_stats)

    with open(os.path.join(_DIR, "team_stats.json"), "w", encoding="utf-8") as f:
        json.dump(team_stats, f, indent=2, ensure_ascii=False)

    with_val = sum(1 for s in team_stats.values()
                   if s.get("matches_played", 0) > 0 and s.get("squad_value_eur"))
    print(f"Attached squad values: {len(values)} second-division + {n_new} new top-flight clubs.")
    print(f"team_stats now has squad values for {with_val} clubs.")


if __name__ == "__main__":
    main()
