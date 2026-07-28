"""Team-level player-performance signal from the Transfermarkt CC0 dataset
(the same free source as squad values — appearances.csv).

Motivation: a single squad market value is one noisy number, and many clubs
lack it. Actual on-pitch production adds a different, non-redundant signal —
in particular ASSISTS (creativity, which match results don't expose) and
disciplinary rate (cards/game, a style/aggression proxy) — aggregated to the
squad over a rolling window that spans league AND cup games Transfermarkt
tracks. Attached point-in-time (no lookahead) via the same team->TM-club_id
map squad value uses, with a `has_player_stats` flag for uncovered clubs.

What this is NOT: tackles, pace, or playstyle — those need StatsBomb/Opta
event data or FIFA/EA ratings (paid / licence-restricted), not available free
or historically here. This module ships the free subset only.

The raw appearances feed is ~42MB gzipped / millions of rows, so it is streamed
and reduced to a compact per-club game series that is cached; only the reduction
is kept in memory, not the raw rows.
"""

import csv
import gzip
import io
import json
import os
import time
from collections import defaultdict
from typing import Dict, List, Optional
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError

TM_BASE_URL = "https://pub-e682421888d945d684bcae8890b0ec20.r2.dev/data"
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_cache")
TM_CACHE_TTL = 3 * 86400
# Trailing games used for a club's rolling averages (~a season of league+cup).
WINDOW = 40


def _reduced_cache_path() -> str:
    return os.path.join(CACHE_DIR, "player_stats_reduced.json")


def _fetch_appearances_text() -> Optional[str]:
    url = f"{TM_BASE_URL}/appearances.csv.gz"
    try:
        req = Request(url, headers={"User-Agent": "predict-xi/1.0"})
        with urlopen(req, timeout=180) as resp:
            return gzip.decompress(resp.read()).decode("utf-8", errors="replace")
    except (URLError, HTTPError, OSError, EOFError) as e:
        print(f"appearances fetch failed {url}: {e}")
        return None


def _build_reduced() -> Dict[str, list]:
    """Reduce appearances to {club_id: [[date, goals, assists, cards], ...]}
    (one row per club-game, chronological). Streamed so the raw feed never all
    sits in memory as rows."""
    text = _fetch_appearances_text()
    if not text:
        return {}
    # (club_id, game_id) -> [date, goals, assists, cards]
    per_game: Dict[tuple, list] = {}
    reader = csv.DictReader(io.StringIO(text))
    for r in reader:
        cid = (r.get("player_club_id") or "").strip()
        gid = (r.get("game_id") or "").strip()
        d = (r.get("date") or "").strip()
        if not cid or not gid or not d:
            continue
        key = (cid, gid)
        rec = per_game.get(key)
        if rec is None:
            rec = per_game[key] = [d, 0.0, 0.0, 0.0]
        try:
            rec[1] += float(r.get("goals") or 0)
            rec[2] += float(r.get("assists") or 0)
            rec[3] += float(r.get("yellow_cards") or 0) + float(r.get("red_cards") or 0)
        except ValueError:
            pass
    per_club: Dict[str, list] = defaultdict(list)
    for (cid, _gid), rec in per_game.items():
        per_club[cid].append(rec)
    for cid in per_club:
        per_club[cid].sort(key=lambda x: x[0])  # chronological by date
    return dict(per_club)


class PlayerStatsIndex:
    """Rolling squad production per club, point-in-time (no lookahead)."""

    def __init__(self):
        self.series: Dict[str, list] = {}

    def build(self) -> "PlayerStatsIndex":
        os.makedirs(CACHE_DIR, exist_ok=True)
        path = _reduced_cache_path()
        if os.path.exists(path) and (time.time() - os.path.getmtime(path)) < TM_CACHE_TTL:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    self.series = json.load(f)
                return self
            except (OSError, json.JSONDecodeError):
                pass
        self.series = _build_reduced()
        if self.series:
            try:
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(self.series, f)
            except OSError:
                pass
        return self

    def features_at(self, club_id: str, on_date: str, window: int = WINDOW) -> Optional[Dict]:
        """Rolling per-game averages over the club's last `window` games
        strictly BEFORE `on_date`. None if the club/date isn't covered or has
        too little history to be meaningful."""
        if not club_id or not on_date:
            return None
        games = self.series.get(str(club_id))
        if not games:
            return None
        recent = [g for g in games if g[0] < on_date][-window:]
        if len(recent) < 10:  # too thin to trust
            return None
        n = len(recent)
        goals = sum(g[1] for g in recent)
        assists = sum(g[2] for g in recent)
        cards = sum(g[3] for g in recent)
        return {
            "ga_per_game": (goals + assists) / n,
            "cards_per_game": cards / n,
        }

    def current_features(self, club_id: str, window: int = WINDOW) -> Optional[Dict]:
        """Most recent `window` games (for serving). Uses a sentinel future
        date so `features_at` takes the tail."""
        return self.features_at(club_id, "9999-99-99", window)


_INDEX: Optional["PlayerStatsIndex"] = None


def get_index(force_reload: bool = False) -> "PlayerStatsIndex":
    global _INDEX
    if _INDEX is None or force_reload:
        _INDEX = PlayerStatsIndex().build()
    return _INDEX


def player_stats_feature_dict(home: Optional[Dict], away: Optional[Dict]) -> dict:
    """Feature block from two clubs' rolling stats. has_player_stats is 0
    unless BOTH sides are known (same both-sides rule as squad value/ClubElo —
    a one-sided view is misleading)."""
    both = bool(home) and bool(away)

    def _g(d, k):
        return float(d.get(k, 0.0)) if d else 0.0

    return {
        "home_ga_per_game": _g(home, "ga_per_game") if both else 0.0,
        "away_ga_per_game": _g(away, "ga_per_game") if both else 0.0,
        "ga_per_game_diff": (_g(home, "ga_per_game") - _g(away, "ga_per_game")) if both else 0.0,
        "home_cards_per_game": _g(home, "cards_per_game") if both else 0.0,
        "away_cards_per_game": _g(away, "cards_per_game") if both else 0.0,
        "has_player_stats": 1.0 if both else 0.0,
    }
