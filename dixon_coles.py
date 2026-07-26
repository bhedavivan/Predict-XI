"""
Rolling Dixon-Coles-style attack/defense ratings.

A lightweight, incremental approximation of the Dixon & Coles (1997) Poisson
goal model: instead of re-fitting maximum-likelihood attack/defense
parameters over the whole history on every match (expensive across tens of
thousands of rows), each team carries a running attack/defense strength that
nudges toward the surprise in each result — the same "K * (actual -
expected)" idea Elo uses, applied to goals instead of points.

Expected goals feed a Poisson scoreline grid, with the Dixon-Coles low-score
correlation adjustment (tau) for the 0-0/1-0/0-1/1-1 cells, to derive
Home/Draw/Away probabilities. This is specifically aimed at the model's
weakest class: literature on football prediction (Dixon & Coles 1997;
follow-up work on the model) attributes a chunk of its edge over plain
classifiers to modeling low-scoring/drawn outcomes directly rather than
learning them from aggregate form stats alone.

Used from data_processor.add_form_features the same way Elo is: features
for a match are computed from ratings as they stood *before* that match
(no lookahead), then the ratings are updated from the real result.
"""

import math
from collections import defaultdict

# Historical league-wide averages (goals/game) used as the Poisson baseline
# rate that attack/defense multipliers scale. Kept as fixed constants rather
# than fit from data — the classifier's own bias term absorbs any small
# systematic offset, and it avoids threading a data-dependent parameter
# through every caller.
LEAGUE_AVG_HOME_GOALS = 1.45
LEAGUE_AVG_AWAY_GOALS = 1.20

DC_START = 1.0          # neutral attack/defense strength
DC_K = 0.08              # learning rate for the rolling update
DC_MIN, DC_MAX = 0.4, 2.5  # clip so a small sample size can't run away
DC_RHO = -0.10            # Dixon-Coles low-score correlation adjustment
MAX_GOALS = 8             # scoreline grid cap for summing outcome probabilities


def _clip(value: float) -> float:
    return max(DC_MIN, min(DC_MAX, value))


def _poisson_pmf(k: int, lam: float) -> float:
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    return math.exp(-lam) * lam ** k / math.factorial(k)


def _tau(i: int, j: int, lam_home: float, lam_away: float, rho: float) -> float:
    """Dixon-Coles correlation adjustment for low-scoring cells."""
    if i == 0 and j == 0:
        return 1 - lam_home * lam_away * rho
    if i == 0 and j == 1:
        return 1 + lam_home * rho
    if i == 1 and j == 0:
        return 1 + lam_away * rho
    if i == 1 and j == 1:
        return 1 - rho
    return 1.0


def expected_goals(home_attack: float, home_defense: float,
                    away_attack: float, away_defense: float,
                    base_home: float = None, base_away: float = None) -> tuple:
    """`base_home`/`base_away` override the global goal baselines with
    league-specific ones (home advantage genuinely varies by competition —
    measured home-win rate spans 38.9%-46.7% across the leagues trained on).
    Defaults preserve the original global-constant behaviour."""
    bh = LEAGUE_AVG_HOME_GOALS if base_home is None else base_home
    ba = LEAGUE_AVG_AWAY_GOALS if base_away is None else base_away
    exp_home = bh * home_attack * away_defense
    exp_away = ba * away_attack * home_defense
    return exp_home, exp_away


def match_probabilities(home_attack: float, home_defense: float,
                         away_attack: float, away_defense: float,
                         base_home: float = None, base_away: float = None,
                         rho: float = None) -> tuple:
    """Return (p_home, p_draw, p_away, exp_home_goals, exp_away_goals)."""
    exp_home, exp_away = expected_goals(home_attack, home_defense,
                                        away_attack, away_defense,
                                        base_home, base_away)
    p_home = p_draw = p_away = 0.0
    for i in range(MAX_GOALS + 1):
        pi = _poisson_pmf(i, exp_home)
        for j in range(MAX_GOALS + 1):
            p = pi * _poisson_pmf(j, exp_away) * _tau(i, j, exp_home, exp_away,
                                                       DC_RHO if rho is None else rho)
            p = max(p, 0.0)
            if i > j:
                p_home += p
            elif i == j:
                p_draw += p
            else:
                p_away += p
    total = p_home + p_draw + p_away
    if total <= 0:
        return 1 / 3, 1 / 3, 1 / 3, exp_home, exp_away
    return p_home / total, p_draw / total, p_away / total, exp_home, exp_away


class DixonColesRatings:
    """Rolling attack/defense strength per team, updated match by match.

    `k` (learning rate) and `rho` (low-score correlation) default to the
    module constants but are overridable so they can be tuned against
    downstream predictive performance rather than staying hand-picked."""

    def __init__(self, k: float = None, rho: float = None):
        self.attack = defaultdict(lambda: DC_START)
        self.defense = defaultdict(lambda: DC_START)
        self.k = DC_K if k is None else k
        self.rho = DC_RHO if rho is None else rho

    def predict(self, home: str, away: str,
                 base_home: float = None, base_away: float = None) -> tuple:
        return match_probabilities(
            self.attack[home], self.defense[home],
            self.attack[away], self.defense[away],
            base_home, base_away, self.rho,
        )

    def update(self, home: str, away: str, home_goals: int, away_goals: int,
                base_home: float = None, base_away: float = None):
        bh = LEAGUE_AVG_HOME_GOALS if base_home is None else base_home
        ba = LEAGUE_AVG_AWAY_GOALS if base_away is None else base_away
        exp_home, exp_away = expected_goals(
            self.attack[home], self.defense[home],
            self.attack[away], self.defense[away],
            bh, ba,
        )
        # Home attack and away defense both move on the home-goals surprise
        # (a big home win means home attack understated it or away defense
        # overstated it — same evidence, two ratings); symmetric for away.
        # Normalising by the same baseline used to form the expectation keeps
        # the surprise scale consistent across leagues.
        home_surprise = (home_goals - exp_home) / max(bh, 1e-6)
        away_surprise = (away_goals - exp_away) / max(ba, 1e-6)

        self.attack[home] = _clip(self.attack[home] + self.k * home_surprise)
        self.defense[away] = _clip(self.defense[away] + self.k * home_surprise)
        self.attack[away] = _clip(self.attack[away] + self.k * away_surprise)
        self.defense[home] = _clip(self.defense[home] + self.k * away_surprise)
