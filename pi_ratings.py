"""
pi-ratings (Constantinou & Fenton, 2013) — a dynamic team-strength rating that
the football-forecasting literature repeatedly finds beats Elo for match
prediction (RPS ~0.199 vs Elo ~0.204; the backbone of the CatBoost+pi SOTA).

Two things make pi-ratings different from the Elo already in this project:
  1. They update on the GOAL DIFFERENCE (with a diminishing-returns damping),
     not just win/draw/loss, so a 4-0 and a 1-0 move a team's rating by
     different amounts.
  2. Each team carries SEPARATE home and away ratings that cross-nudge each
     other, capturing that home and away form differ.

Used exactly like Elo / DixonColesRatings in data_processor.add_form_features:
a match's features are read from ratings as they stood BEFORE it (no lookahead),
then the ratings are updated from the real result. The learning rates are
overridable so they can be tuned against holdout RPS rather than hand-picked.

Reference: Constantinou & Fenton, "Determining the level of ability of football
teams by dynamic ratings based on the relative discrepancies in scores",
Journal of Quantitative Analysis in Sports (2013).
"""

import math
from collections import defaultdict

# Learning rates and scale. Anchors from the paper; tune on holdout RPS.
PI_LAMBDA = 0.035    # own-venue update rate
PI_GAMMA = 0.7       # fraction of the update transferred to the other venue
PI_C = 3.0           # rating -> expected-goal scale
PI_START = 0.0       # neutral rating
PI_CLIP = 4.0        # keep ratings (and thus expected goals) bounded
PI_MIN_MATCHES = 4   # below this a team's rating is too raw to trust (has_pi gate)


def expected_goals(rating: float, c: float = PI_C) -> float:
    """Constantinou-Fenton rating -> expected goals mapping: diminishing returns,
    sign-symmetric. e(0)=0; grows sub-linearly in |rating|."""
    sign = 1.0 if rating >= 0 else -1.0
    return sign * (10.0 ** (abs(rating) / c) - 1.0)


def _clip(r: float) -> float:
    return max(-PI_CLIP, min(PI_CLIP, r))


class PiRatings:
    """Rolling home/away pi-ratings per team, updated match by match."""

    def __init__(self, lam: float = None, gamma: float = None, c: float = None):
        self.home = defaultdict(lambda: PI_START)
        self.away = defaultdict(lambda: PI_START)
        self._games = defaultdict(int)
        self.lam = PI_LAMBDA if lam is None else lam
        self.gamma = PI_GAMMA if gamma is None else gamma
        self.c = PI_C if c is None else c

    def home_rating(self, team: str) -> float:
        return self.home[team]

    def away_rating(self, team: str) -> float:
        return self.away[team]

    def games(self, team: str) -> int:
        return self._games[team]

    def predict(self, home: str, away: str) -> tuple:
        """Pre-match expected goal difference (home perspective), no update.
        Returns (pred_gd, ghat_home, ghat_away)."""
        ghat_h = expected_goals(self.home[home], self.c)
        ghat_a = expected_goals(self.away[away], self.c)
        return ghat_h - ghat_a, ghat_h, ghat_a

    def update(self, home: str, away: str, home_goals: int, away_goals: int) -> None:
        """Fold in a real result. The rating of the acting venue moves by the
        damped goal-difference error; the team's OTHER venue rating gets a
        `gamma` fraction of that move (the cross-venue transfer)."""
        pred_gd, _, _ = self.predict(home, away)
        obs_gd = home_goals - away_goals
        err = obs_gd - pred_gd
        if err > 0:
            direction = 1.0
        elif err < 0:
            direction = -1.0
        else:
            direction = 0.0
        psi = self.c * math.log10(1.0 + abs(err))   # diminishing returns on big margins

        d_home = self.lam * psi * direction          # home over/under-performed
        self.home[home] = _clip(self.home[home] + d_home)
        self.away[home] = _clip(self.away[home] + self.gamma * d_home)

        d_away = -self.lam * psi * direction          # symmetric for the away side
        self.away[away] = _clip(self.away[away] + d_away)
        self.home[away] = _clip(self.home[away] + self.gamma * d_away)

        self._games[home] += 1
        self._games[away] += 1


def pi_feature_dict(home_pi, away_pi, home_games, away_games) -> dict:
    """Fixed-width pi feature block, shared by the training and serving paths so
    they cannot skew. `home_pi` is the HOME team's HOME rating; `away_pi` is the
    AWAY team's AWAY rating (the venue split is the whole point). `has_pi` is 1
    only when BOTH sides have enough history to trust — otherwise the block is
    zeroed, so an unrated team reads 0-with-flag, not a spuriously strong/weak
    rating."""
    both = (home_pi is not None and away_pi is not None
            and (home_games or 0) >= PI_MIN_MATCHES and (away_games or 0) >= PI_MIN_MATCHES)
    hp = float(home_pi) if home_pi is not None else 0.0
    ap = float(away_pi) if away_pi is not None else 0.0
    exp_gd = expected_goals(hp) - expected_goals(ap) if both else 0.0
    # Bound the non-linear expected-goal-difference feature defensively.
    exp_gd = max(-8.0, min(8.0, exp_gd))
    return {
        "home_pi": hp if both else 0.0,
        "away_pi": ap if both else 0.0,
        "pi_diff": (hp - ap) if both else 0.0,
        "pi_expected_gd": exp_gd,
        "has_pi": 1.0 if both else 0.0,
    }
