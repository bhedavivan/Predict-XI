"""Player-level availability (injuries / suspensions / expected lineups).

WHAT THIS IS AND ISN'T — read before wiring it into the model.

The single biggest thing separating this model from a bookmaker is real-time
squad availability: who's injured, suspended, or rested. Adding it is the honest
"catch the market" lever (see RESEARCH-NOTES.md). But there's a hard constraint:

  * No FREE source provides per-match historical lineups/injuries across 168k
    matches and 38 leagues. Free API tiers cap at ~100 requests/day, so you
    CANNOT backfill history — which means you cannot TRAIN a feature on it.
  * Therefore this data can only enrich LIVE predictions (adjust a forecast for
    a fixture happening soon, whose team-news is knowable), not the trained model.

So this module is deliberately:
  * an ADAPTER (pluggable source + a shared feature helper), ready to use the
    moment an API key is configured, and
  * NOT wired into the training feature vector — adding an always-empty
    `has_player_data=0` block would be a zero-variance feature and exactly the
    "populated in training, degenerate at serving" trap this project keeps
    hitting. Wire it in only once you have a source with real coverage.

To enable: set PLAYER_API_TOKEN in .env and instantiate a source (an
API-Football adapter skeleton is provided). Then call
`live_availability_adjustment(...)` from the /predict path to nudge a live
forecast, or, if you secure historical coverage, feed
`player_availability_features(...)` into add_form_features / prepare_prediction_
features behind a `has_player_data` flag exactly like the squad-value block.

No hand-picked per-club constants: any adjustment is derived from the fraction
of squad MARKET VALUE unavailable (a measured quantity), consistent with the
project's no-fudge-factors rule.
"""

import json
import os
from typing import Optional, Dict, List

try:
    from config import load_env  # reuses the same .env loader as the API client
    load_env()
except Exception:  # noqa: BLE001
    pass

PLAYER_API_TOKEN = os.getenv("PLAYER_API_TOKEN", "")
_DIR = os.path.dirname(os.path.abspath(__file__))


# ─── Source interface ───────────────────────────────────────────────────────

class PlayerDataSource:
    """Interface for a player-data provider. Implement `availability` to return,
    for a club around a date, which players are out and their share of the
    squad's market value; optionally `style` for play-style metrics. Return None
    when unknown (never guess)."""

    def availability(self, club: str, on_date: str) -> Optional[Dict]:
        raise NotImplementedError

    def style(self, club: str, on_date: str) -> Optional[Dict]:
        """Play-style metrics for a club (directness, pressing, tempo, …).
        Default None — override in a source that has them."""
        return None


class FilePlayerDataSource(PlayerDataSource):
    """Bring-your-own-data source: reads a JSON file the user maintains, so live
    squad availability AND play-styles can be fed WITHOUT any API or code change
    — exactly the "I'll feed the data/API when I have it" path. Shape:

        {
          "availability": {
            "Man City": {"value_out_fraction": 0.08, "key_absences": 1},
            ...
          },
          "style": {
            "Man City": {"directness": 0.3, "pressing": 0.8, "tempo": 0.7, "possession": 0.65},
            ...
          }
        }

    Keys are our team_stats names (e.g. "Man City", "Ath Madrid"). Values are the
    current snapshot. Drop the file at data_cache/player_data.json (or pass a
    path) and it activates automatically; absent → inert."""

    DEFAULT_PATH = os.path.join(_DIR, "data_cache", "player_data.json")

    def __init__(self, path: str = None):
        self.path = path or self.DEFAULT_PATH
        self._avail, self._style = {}, {}
        try:
            with open(self.path, encoding="utf-8") as f:
                data = json.load(f)
            self._avail = data.get("availability", {}) or {}
            self._style = data.get("style", {}) or {}
        except (OSError, ValueError):
            pass

    def available(self) -> bool:
        return bool(self._avail or self._style)

    def availability(self, club: str, on_date: str) -> Optional[Dict]:
        return self._avail.get(club)

    def style(self, club: str, on_date: str) -> Optional[Dict]:
        return self._style.get(club)


class ApiFootballSource(PlayerDataSource):
    """Skeleton adapter for API-Football (api-football.com), which exposes
    /injuries and /fixtures/lineups. Free tier ~100 req/day — enough for live
    fixtures, not for historical backfill. Fill in the endpoint calls once you
    have a key; without a token this is inert (returns None), by design."""

    BASE = "https://v3.football.api-sports.io"

    def __init__(self, token: str = PLAYER_API_TOKEN):
        self.token = token

    def availability(self, club: str, on_date: str) -> Optional[Dict]:
        if not self.token:
            return None
        # Intentionally not implemented against a live endpoint here: doing so
        # blind (no key to verify field names / rate limits) would be guesswork.
        # Wire the real /injuries + /players call when a key is available and
        # return the dict shape documented in `player_availability_features`.
        raise NotImplementedError(
            "ApiFootballSource.availability: implement the /injuries call once "
            "PLAYER_API_TOKEN is set and the response shape is verified.")


# ─── Shared feature helper (same pattern as squad_value_feature_dict) ─────────

def player_availability_features(home_avail: Optional[Dict],
                                  away_avail: Optional[Dict]) -> dict:
    """Feature block from two clubs' availability dicts. Each dict is expected
    to carry `value_out_fraction` (0..1: share of squad market value currently
    unavailable) and `key_absences` (count of unavailable players in, say, the
    top-10 by value). `has_player_data` is 0 unless BOTH sides are known — a
    one-sided view is misleading, same rule as squad value and ClubElo.

    Returns a fixed-width block so, if wired in, it aligns across train/serve.
    """
    both = bool(home_avail) and bool(away_avail)

    def _g(d, k):
        return float(d.get(k, 0.0)) if d else 0.0

    return {
        "home_value_out_frac": _g(home_avail, "value_out_fraction") if both else 0.0,
        "away_value_out_frac": _g(away_avail, "value_out_fraction") if both else 0.0,
        # Positive => away is MORE depleted than home (favours home).
        "availability_edge": (_g(away_avail, "value_out_fraction")
                              - _g(home_avail, "value_out_fraction")) if both else 0.0,
        "home_key_absences": _g(home_avail, "key_absences") if both else 0.0,
        "away_key_absences": _g(away_avail, "key_absences") if both else 0.0,
        "has_player_data": 1.0 if both else 0.0,
    }


STYLE_KEYS = ("directness", "pressing", "tempo", "possession")


def style_feature_dict(home_style: Optional[Dict], away_style: Optional[Dict]) -> dict:
    """Fixed-width play-style feature block, ready to wire into training the
    moment the user supplies historical style data with point-in-time coverage
    (same both-sides `has_style` gate as the other enrichment blocks). Until
    then it's used only for display. Generic axes so any provider maps onto it."""
    both = bool(home_style) and bool(away_style)
    out = {}
    for k in STYLE_KEYS:
        out[f"home_{k}"] = float(home_style.get(k, 0.0)) if both and home_style else 0.0
        out[f"away_{k}"] = float(away_style.get(k, 0.0)) if both and away_style else 0.0
    out["has_style"] = 1.0 if both else 0.0
    return out


def get_source() -> Optional[PlayerDataSource]:
    """Return the best available player-data source, or None. Prefers a local
    bring-your-own-data JSON file; falls back to the API-Football adapter when a
    token is set; otherwise None (the /predict path then runs unchanged)."""
    f = FilePlayerDataSource()
    if f.available():
        return f
    if PLAYER_API_TOKEN:
        return ApiFootballSource()
    return None


def live_availability_adjustment(home_avail: Optional[Dict], away_avail: Optional[Dict],
                                  probs: Dict[str, float]) -> Dict[str, float]:
    """OPTIONAL live-only nudge for the /predict path: shift Home/Away
    probability mass by the availability edge (share of squad value missing),
    leaving the model's trained output untouched when no data is present.

    Deliberately conservative and derived from a measured quantity (value-out
    fraction), not a hand-set per-club bonus. Returns probs unchanged if either
    side's availability is unknown. This is a *display-time* aid, not a model
    feature — keep it clearly separate from the trained probabilities.
    """
    if not home_avail or not away_avail:
        return probs
    edge = (float(away_avail.get("value_out_fraction", 0.0))
            - float(home_avail.get("value_out_fraction", 0.0)))
    if edge == 0.0:
        return probs
    # Move at most ~half the value-out gap into the favoured side's win prob,
    # taken proportionally from the other win outcome; draw left as-is.
    shift = max(-0.15, min(0.15, 0.5 * edge))
    p = dict(probs)
    if shift > 0:   # home less depleted
        take = min(shift, p.get("Away Win", 0.0))
        p["Home Win"] = p.get("Home Win", 0.0) + take
        p["Away Win"] = p.get("Away Win", 0.0) - take
    else:
        take = min(-shift, p.get("Home Win", 0.0))
        p["Away Win"] = p.get("Away Win", 0.0) + take
        p["Home Win"] = p.get("Home Win", 0.0) - take
    return p
