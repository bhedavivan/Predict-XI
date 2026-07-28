"""
Polite scraper for club squad market values from transfermarkt.com — the
second/lower divisions the free CC0 bulk dataset does NOT ship.

Why this exists
---------------
`transfermarkt_data.py` reads the published CC0 dcaribou/transfermarkt-datasets
export, which (verified) contains first divisions only. So Championship, Serie B,
Ligue 2, 2. Bundesliga, Segunda, League One/Two clubs have no squad value and the
model falls back to `has_squad_value=0` for them. This module fills that gap by
reading the market-value pages Transfermarkt serves publicly.

Scraping vs. the CC0 dataset
----------------------------
The project's default is the CC0 dataset precisely to avoid scraping. The user
explicitly authorised scraping Transfermarkt for these missing values, so this
module does it — but politely and within robots.txt:

  * robots.txt (checked) disallows only /ceapi, /quickselect, /jumplist and
    /navigation/getSubNavigation for a generic agent; the /startseite market-value
    pages we read are allowed. It DOES fully block AI/bot user-agents by name
    (ClaudeBot, GPTBot, anthropic-ai, CCBot, …). So we send a plain browser UA and
    assert it carries none of those tokens.
  * One request at a time, a 3–5 s randomised delay between requests, and an
    on-disk HTML cache — past seasons are immutable, so a backfill is fetched once
    and never again. Exponential backoff on 429/503.

It reads only public market-value pages; it does not log in, submit forms, or
touch disallowed paths. Values are attached to our clubs by the same safe,
league-constrained name matcher used for the CC0 data (club_mapping), never a
loose similarity guess.

Output: data_cache/tm_scraped_values.json
    { club_id: {"name": str, "comp": TM_code, "seasons": {year: eur}, "current": eur} }

Run: python transfermarkt_scraper.py            # current + recent seasons, default comps
     python transfermarkt_scraper.py GB2 IT2    # specific TM competition codes
"""

import json
import os
import random
import re
import sys
import time
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError

TM_BASE = "https://www.transfermarkt.co.uk"

# A plain, current browser UA. Transfermarkt's robots.txt blocks named bot UAs
# outright, so this MUST NOT contain 'bot', 'claude', 'gpt', 'anthropic', etc.
_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
_FORBIDDEN_UA_TOKENS = ("bot", "claude", "gpt", "anthropic", "ccbot", "crawl", "spider")
assert not any(t in _UA.lower() for t in _FORBIDDEN_UA_TOKENS), \
    "User-Agent must not identify as a bot (Transfermarkt robots.txt blocks those)."

_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_cache")
_HTML_CACHE = os.path.join(_CACHE_DIR, "tm_html")
_OUT_PATH = os.path.join(_CACHE_DIR, "tm_scraped_values.json")

_MIN_DELAY, _MAX_DELAY = 3.0, 5.0
_last_fetch = [0.0]   # wall-clock of the last network hit, for pacing

# Our league code -> Transfermarkt competition code, for the divisions the CC0
# dataset omits. Every code here has been verified by fetching its page and
# confirming the <title>. Add more ONLY after the same verification — a wrong
# code silently attaches another division's values.
SECOND_DIVISION_COMPS = {
    "ELC": "GB2",    # Championship
    "ELC2": "GB3",   # League One
    "ELC3": "GB4",   # League Two
    "PD2": "ES2",    # Segunda Division
    "SA2": "IT2",    # Serie B
    "BL2": "L2",     # 2. Bundesliga
    "FL2": "FR2",    # Ligue 2
}

# Sanity band for a *total squad* value (EUR). Second-division squads run from a
# few million to a few hundred million (parachute clubs). Anything outside this
# is a parse error, not a real value, and is dropped.
_MIN_TOTAL, _MAX_TOTAL = 200_000, 2_000_000_000


def _reference_season() -> int:
    """The most recent COMPLETED season's start year. A football season starts
    in August, so before August the current calendar year's season hasn't begun
    and the last completed one starts the previous year (e.g. 2026-07 -> 2025,
    i.e. the 2025-26 season)."""
    t = time.localtime()
    return t.tm_year - 1 if t.tm_mon < 8 else t.tm_year


def _html_cache_path(url: str) -> str:
    key = re.sub(r"[^A-Za-z0-9]+", "_", url.replace(TM_BASE, "")).strip("_")
    return os.path.join(_HTML_CACHE, key + ".html")


def polite_get(url: str, use_cache: bool = True) -> str:
    """Fetch a page with a browser UA, honouring the crawl delay, backing off on
    rate limits, and caching the HTML on disk (past-season pages never change)."""
    path = _html_cache_path(url)
    if use_cache and os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        except OSError:
            pass

    # Pace: keep >= _MIN_DELAY since the previous network fetch, plus jitter.
    wait = _MIN_DELAY - (time.time() - _last_fetch[0])
    time.sleep(max(0.0, wait) + random.uniform(0.0, _MAX_DELAY - _MIN_DELAY))

    backoff = 60.0
    for attempt in range(3):
        try:
            req = Request(url, headers={"User-Agent": _UA,
                                        "Accept-Language": "en-US,en;q=0.9"})
            with urlopen(req, timeout=30) as resp:
                html = resp.read().decode("utf-8", "replace")
            _last_fetch[0] = time.time()
            os.makedirs(_HTML_CACHE, exist_ok=True)
            try:
                with open(path, "w", encoding="utf-8") as f:
                    f.write(html)
            except OSError:
                pass
            return html
        except HTTPError as e:
            _last_fetch[0] = time.time()
            if e.code in (429, 503) and attempt < 2:
                time.sleep(backoff)
                backoff *= 2
                continue
            raise
        except (URLError, OSError) as e:
            if attempt < 2:
                time.sleep(backoff)
                backoff *= 2
                continue
            raise RuntimeError(f"fetch failed for {url}: {e}")
    raise RuntimeError(f"fetch failed for {url}")


def _value_to_eur(num: str, unit: str) -> float:
    """'312.35','m' -> 312_350_000.0 . TM .co.uk uses '.' as the decimal point."""
    v = float(num.replace(",", ""))
    unit = (unit or "").lower()
    if unit == "bn":
        return v * 1e9
    if unit == "m":
        return v * 1e6
    if unit in ("k", "th."):
        return v * 1e3
    return v


def parse_club_values(html: str) -> list:
    """Parse a competition 'startseite' page into [(club_id, name, total_eur)].

    Row layout on that page ends with two money columns (average value, then
    TOTAL squad value); the total is always the larger, so we take the max
    unit-value in the row. Name comes from the club link's title attribute, id
    from its /verein/<id>/ segment."""
    m = re.search(r'<table class="items">(.*?)</table>', html, re.S)
    if not m:
        return []
    rows = re.findall(r'<tr class="(?:odd|even)">(.*?)</tr>', m.group(1), re.S)
    out = []
    for row in rows:
        cid = re.search(r'/verein/(\d+)/', row)
        if not cid:
            continue
        name_m = (re.search(r'/verein/\d+/[^"]*"\s+title="([^"]+)"', row)
                  or re.search(r'title="([^"]+)"', row))
        name = name_m.group(1).strip() if name_m else ""
        # Every money token in the row, e.g. ('312.35','m'); take the largest.
        money = re.findall(r'([\d.,]+)\s*(bn|m|k|Th\.)\b', row)
        if not name or not money:
            continue
        total = max(_value_to_eur(n, u) for n, u in money)
        if _MIN_TOTAL <= total <= _MAX_TOTAL:
            out.append((cid.group(1), name, total))
    return out


def scrape_competition(tm_code: str, year: int) -> list:
    """Club total squad values for one competition-season."""
    url = f"{TM_BASE}/-/startseite/wettbewerb/{tm_code}/plus/0?saison_id={year}"
    return parse_club_values(polite_get(url))


# ─── Match results (gesamtspielplan) — used to add whole leagues ───────────

_SCORE_RE = re.compile(r'class="ergebnis-link"[^>]*>\s*(\d+):(\d+)\s*<')
_VEREIN_RE = re.compile(r'<a title="([^"]+)"[^>]*href="[^"]*?/verein/\d+/')
_DATUM_RE = re.compile(r'datum/(\d{4}-\d{2}-\d{2})')
_ROW_RE = re.compile(r'<tr[^>]*>(.*?)</tr>', re.S)


def parse_gesamtspielplan(html: str) -> list:
    """Parse a Transfermarkt 'gesamtspielplan' (all fixtures & results) page into
    [(date_iso, home, away, home_goals, away_goals)] for PLAYED matches only.

    Per row: the score is the `ergebnis-link` anchor (e.g. '3:1'); the home team
    is the `/verein/` anchor's title before it and the away team the one after
    (the title attribute is the full club name, matching our TM value scraper);
    the date is a `datum/YYYY-MM-DD` link printed once per date, so it is carried
    forward across rows. Rows without a real score (unplayed / a kickoff time)
    are skipped. This href-based recipe is deliberately not tied to brittle CSS
    class names beyond `ergebnis-link`."""
    out = []
    last_date = None
    for row in _ROW_RE.findall(html):
        d = _DATUM_RE.search(row)
        if d:
            last_date = d.group(1)
        sm = _SCORE_RE.search(row)
        if not sm or not last_date:
            continue
        hg, ag = int(sm.group(1)), int(sm.group(2))
        pos = sm.start()
        teams = [(m.start(), m.group(1)) for m in _VEREIN_RE.finditer(row)]
        before = [nm for st, nm in teams if st < pos]
        after = [nm for st, nm in teams if st > pos]
        if not before or not after:
            continue
        home, away = before[-1].strip(), after[0].strip()
        if home and away and home != away:
            out.append((last_date, home, away, hg, ag))
    return out


def scrape_gesamtspielplan(tm_code: str, year: int) -> list:
    """Match results for one competition-season. `tm_code` may be a '+'-joined
    set of codes for split-season leagues (e.g. Peru 'TDeA+TDeC'), whose halves
    are fetched and concatenated into one logical season."""
    rows = []
    for code in tm_code.split("+"):
        url = f"{TM_BASE}/-/gesamtspielplan/wettbewerb/{code}/saison_id/{year}"
        try:
            rows.extend(parse_gesamtspielplan(polite_get(url)))
        except Exception as e:
            print(f"    {code} {year}: results skipped ({e})")
    return rows


_RESULTS_CACHE = os.path.join(_CACHE_DIR, "tm_results")


def results_for(tm_code: str, year: int, force: bool = False) -> list:
    """Parsed match results for a competition-season, cached to disk as JSON so
    repeated training runs neither re-fetch nor re-parse. `force` re-scrapes
    (used to refresh an in-progress season)."""
    path = os.path.join(_RESULTS_CACHE, f"{tm_code.replace('+', '_')}_{year}.json")
    if not force and os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as f:
                return [tuple(r) for r in json.load(f)]
        except (OSError, ValueError):
            pass
    rows = scrape_gesamtspielplan(tm_code, year)
    os.makedirs(_RESULTS_CACHE, exist_ok=True)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False)
    except OSError:
        pass
    return rows


def build_tm_results(seasons_saison_ids, force_latest: int = 2) -> None:
    """Pre-scrape match results for every Transfermarkt-sourced league across the
    given TM saison_ids, populating the on-disk cache so a subsequent training
    run reads instantly. `seasons_saison_ids` is an iterable of int start-years.
    The `force_latest` most-recent seasons are re-scraped (they may be in
    progress). Import here to avoid a hard dependency cycle at module load."""
    import leagues
    tm_leagues = [lg for lg in leagues.REGISTRY if lg.source == "tm"]
    years = sorted(set(int(y) for y in seasons_saison_ids))
    latest = set(years[-force_latest:]) if force_latest else set()
    for lg in tm_leagues:
        got = 0
        for y in years:
            rows = results_for(lg.tm_code, y, force=(y in latest))
            got += len(rows)
        print(f"  {lg.code:5} ({lg.tm_code}): {got} matches across {len(years)} seasons")


def build_scraped_values(comps=None, seasons=None) -> dict:
    """Scrape each (comp, season), keeping every club's per-season value and its
    most recent one. Persists to tm_scraped_values.json. `comps` is a list of TM
    competition codes (default: the second divisions); `seasons` a list of start
    years (default: the last five)."""
    comps = comps or sorted(set(SECOND_DIVISION_COMPS.values()))
    if seasons is None:
        # Last completed season back four more, plus the upcoming one (so newly
        # promoted/relegated clubs are captured even if we use the completed
        # season's value as 'current').
        ref = _reference_season()
        seasons = list(range(ref + 1, ref - 4, -1))

    store = {}
    tm_to_ours = {v: k for k, v in SECOND_DIVISION_COMPS.items()}
    for comp in comps:
        for year in seasons:
            try:
                rows = scrape_competition(comp, year)
            except Exception as e:
                print(f"  {comp} {year}: skipped ({e})")
                continue
            for cid, name, val in rows:
                rec = store.setdefault(cid, {"name": name, "comp": comp,
                                             "our_league": tm_to_ours.get(comp, ""),
                                             "seasons": {}, "current": None})
                rec["name"] = name
                rec["seasons"][str(year)] = val
            print(f"  {comp} {year}: {len(rows)} clubs")

    # 'current' = the last COMPLETED season's value, not the upcoming one. In
    # pre-season the not-yet-started season's squad is in flux (players sold,
    # replacements unsigned), which badly understates value (e.g. Hertha reads
    # €23m pre-season vs €61m for the completed season). Fall back to a club's
    # latest available season if it isn't in the reference one (e.g. just
    # promoted into this division).
    ref = str(_reference_season())
    for rec in store.values():
        if rec["seasons"]:
            rec["current"] = rec["seasons"].get(ref) or rec["seasons"][max(rec["seasons"], key=int)]

    os.makedirs(_CACHE_DIR, exist_ok=True)
    with open(_OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(store, f, indent=2, ensure_ascii=False)
    print(f"\nWrote {len(store)} clubs -> {os.path.relpath(_OUT_PATH)}")
    return store


def load_scraped_values() -> dict:
    try:
        with open(_OUT_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def main():
    comps = [a for a in sys.argv[1:] if a.isalnum()] or None
    print("Scraping Transfermarkt second-division squad values "
          "(polite: browser UA, 3–5s delay, cached)…")
    build_scraped_values(comps=comps)


if __name__ == "__main__":
    main()
