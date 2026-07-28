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


def main():
    with open(os.path.join(_DIR, "team_stats.json"), encoding="utf-8") as f:
        team_stats = json.load(f)
    store = load_scraped_values()
    if not store:
        raise SystemExit("No scraped values found — run `python transfermarkt_scraper.py` first.")

    # only_missing=False: second-division values come solely from this scraper,
    # so always refresh them (idempotent, and lets corrected scrapes take effect).
    values = match_scraped_values(team_stats, store, only_missing=False)
    for team, val in values.items():
        team_stats[team]["squad_value_eur"] = val

    with open(os.path.join(_DIR, "team_stats.json"), "w", encoding="utf-8") as f:
        json.dump(team_stats, f, indent=2, ensure_ascii=False)

    with_val = sum(1 for s in team_stats.values()
                   if s.get("matches_played", 0) > 0 and s.get("squad_value_eur"))
    print(f"Attached second-division squad values to {len(values)} clubs.")
    print(f"team_stats now has squad values for {with_val} clubs.")


if __name__ == "__main__":
    main()
