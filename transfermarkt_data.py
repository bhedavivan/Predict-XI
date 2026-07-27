"""
Squad market value from the transfermarkt-datasets bulk export.

Source: https://github.com/dcaribou/transfermarkt-datasets (CC0-1.0, public
domain — so this reads a published dataset rather than scraping Transfermarkt
directly, which their ToS would not permit). Refreshed weekly.

Why squad value: it is the strongest team-strength signal available to this
project that is a *cause* of results rather than a readout of someone else's
opinion (unlike bookmaker odds, which were deliberately rejected — see
README). Because valuations are dated, the model can learn how much squad
strength is worth from history, and current values then shift live
predictions for a learned reason instead of a hand-tuned fudge factor.

Two access paths:
  squad_value_at()  — point-in-time, for building training features
  current_squad_values() — present-day, so new signings register in the UI

IMPORTANT: clubs.csv has a `total_market_value` column that is empty for all
796 clubs. It looks like exactly the shortcut wanted here and is not; values
must be aggregated from player_valuations.
"""

import csv
import gzip
import io
import os
import time
from collections import defaultdict
from datetime import date
from typing import Dict, Optional, Tuple
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError

TM_BASE_URL = "https://pub-e682421888d945d684bcae8890b0ec20.r2.dev/data"

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_cache")
# The upstream dataset rebuilds weekly, so a 24h TTL (as used for match CSVs)
# would re-download ~10MB for nothing. Refresh every 3 days.
TM_CACHE_TTL = 3 * 86400

TM_FILES = ("clubs", "players", "player_valuations")


def _cache_path(name: str) -> str:
    return os.path.join(CACHE_DIR, f"transfermarkt_{name}.csv")


def _fetch(name: str) -> Optional[str]:
    url = f"{TM_BASE_URL}/{name}.csv.gz"
    try:
        req = Request(url, headers={"User-Agent": "soccer-predictor/1.0"})
        with urlopen(req, timeout=120) as resp:
            return gzip.decompress(resp.read()).decode("utf-8", errors="replace")
    except (URLError, HTTPError, OSError, EOFError) as e:
        print(f"Error fetching {url}: {e}")
        return None


def load_table(name: str) -> list:
    """Load one transfermarkt table as a list of dicts, cached on disk."""
    if name not in TM_FILES:
        raise ValueError(f"Unknown transfermarkt table: {name}")
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = _cache_path(name)

    content = None
    if os.path.exists(path) and (time.time() - os.path.getmtime(path)) < TM_CACHE_TTL:
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
        except (IOError, UnicodeDecodeError):
            content = None

    if content is None:
        content = _fetch(name)
        if content is None:
            # Fall back to a stale cache rather than failing outright — an
            # out-of-date squad value beats no squad value.
            if os.path.exists(path):
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        return list(csv.DictReader(io.StringIO(f.read())))
                except (IOError, UnicodeDecodeError):
                    pass
            return []
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
        except IOError:
            pass

    return list(csv.DictReader(io.StringIO(content)))


def _month_key(d: str) -> str:
    """'2023-03-15' -> '2023-03'."""
    return d[:7]


def _next_month(mk: str) -> str:
    y, m = int(mk[:4]), int(mk[5:7])
    return f"{y + 1}-01" if m == 12 else f"{y}-{m + 1:02d}"


class SquadValueIndex:
    """Month-granularity club squad values built from dated valuations.

    Snapshots are taken at the START of each month using only valuations
    dated strictly BEFORE that month, so a lookup for a match never sees a
    valuation published on or after the match date. Same no-lookahead
    discipline as Elo and the Dixon-Coles ratings.

    Month granularity is deliberate, not a shortcut: Transfermarkt revalues
    squads a handful of times a year, so finer resolution would carry no more
    information while costing far more memory.
    """

    def __init__(self):
        # club_id -> {month_key: total_value_eur}
        self.snapshots: Dict[str, Dict[str, float]] = defaultdict(dict)
        self._months: list = []
        self.current: Dict[str, float] = {}
        self.current_squad_size: Dict[str, int] = {}
        self.latest_valuation_date: Optional[str] = None

    def build(self, valuations: list, players: Optional[list] = None):
        rows = []
        for r in valuations:
            d = (r.get("date") or "").strip()
            v = (r.get("market_value_in_eur") or "").strip()
            cid = (r.get("current_club_id") or "").strip()
            if not d or not v or not cid:
                continue
            try:
                rows.append((d, r.get("player_id", ""), float(v), cid))
            except ValueError:
                continue
        rows.sort(key=lambda t: t[0])
        if not rows:
            return self
        self.latest_valuation_date = rows[-1][0]

        # Walk forward in time. player_state holds each player's most recent
        # (value, club) as of the cursor; at each month boundary we freeze a
        # snapshot from state built ONLY from strictly-earlier valuations.
        player_state: Dict[str, Tuple[float, str]] = {}
        cursor = _month_key(rows[0][0])
        i = 0
        end_month = _month_key(rows[-1][0])
        seen_clubs = set()
        last_recorded: Dict[str, float] = {}

        while True:
            while i < len(rows) and _month_key(rows[i][0]) < cursor:
                _, pid, val, cid = rows[i]
                player_state[pid] = (val, cid)
                seen_clubs.add(cid)
                i += 1
            totals: Dict[str, float] = defaultdict(float)
            for val, cid in player_state.values():
                totals[cid] += val
            # Iterate every club seen so far, not just those with players
            # now: a club whose squad emptied must record 0 rather than let
            # value_at() carry its last non-zero snapshot forward forever.
            # Only changes are stored, so lookups still carry a value across
            # quiet months while a real drop to zero is recorded.
            for cid in seen_clubs:
                tot = totals.get(cid, 0.0)
                if last_recorded.get(cid) != tot:
                    self.snapshots[cid][cursor] = tot
                    last_recorded[cid] = tot
            self._months.append(cursor)
            if cursor > end_month:
                break
            cursor = _next_month(cursor)

        # Present-day values come from players.csv, which carries each
        # player's CURRENT club — this is what makes new signings show up.
        if players:
            cur: Dict[str, float] = defaultdict(float)
            size: Dict[str, int] = defaultdict(int)
            for p in players:
                cid = (p.get("current_club_id") or "").strip()
                mv = (p.get("market_value_in_eur") or "").strip()
                if not cid:
                    continue
                size[cid] += 1
                if mv:
                    try:
                        cur[cid] += float(mv)
                    except ValueError:
                        pass
            self.current = dict(cur)
            self.current_squad_size = dict(size)
        return self

    def value_at(self, club_id: str, on_date: str) -> Optional[float]:
        """Squad value for `club_id` as known strictly before `on_date`."""
        if not club_id or not on_date:
            return None
        snaps = self.snapshots.get(club_id)
        if not snaps:
            return None
        mk = _month_key(on_date)
        if mk in snaps:
            return snaps[mk]
        earlier = [m for m in snaps if m <= mk]
        if not earlier:
            return None
        return snaps[max(earlier)]

    def current_value(self, club_id: str) -> Optional[float]:
        return self.current.get(club_id)


_INDEX: Optional[SquadValueIndex] = None


def get_index(force_reload: bool = False) -> SquadValueIndex:
    """Module-level cached index — building it parses ~500k rows, so callers
    should share one instance."""
    global _INDEX
    if _INDEX is None or force_reload:
        idx = SquadValueIndex()
        idx.build(load_table("player_valuations"), load_table("players"))
        _INDEX = idx
    return _INDEX
