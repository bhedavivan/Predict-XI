"""Monte Carlo season simulator.

The honest use of "many runs": a single match's probability is already the
model's best estimate — resampling it can't improve it. But feeding the
calibrated per-match probabilities into thousands of simulated SEASONS and
tallying the outcomes yields something the model doesn't output directly — the
distribution of final league tables: title odds, top-4 odds, relegation odds,
expected points, and the full position-probability matrix. This is exactly what
the Opta / FiveThirtyEight "supercomputers" do, built on the existing model.

Two projection modes:
  * LIVE  (`project_league`) — the real thing the online simulators do: start
    from the ACTUAL current standings (points + goal difference already earned
    this season, from the football-data.org API) and simulate only the REAL
    remaining fixtures. This is why a live simulator feels alive: its numbers
    move as matches are played, not because of RNG jitter. Requires an API
    token and one of the covered leagues.
  * HYPOTHETICAL (`run_league`) — for leagues the live API doesn't cover: play
    a full double round-robin from every team's current rating, from 0-0.

Both modes converge on `simulate_projection`, a vectorised NumPy core that
seeds each simulation from a starting points/GD vector, samples each fixture's
outcome from the model's calibrated H/D/A, and samples a Dixon-Coles scoreline
for a realistic goal difference (so tables break ties on GD the way real
leagues do, instead of an arbitrary jitter).

Run: python simulate_season.py PL [n_sims]
"""

import itertools
import json
import math
import os
import sys
import time
from typing import List, Optional, Tuple

import numpy as np

import dixon_coles
import leagues

_DIR = os.path.dirname(os.path.abspath(__file__))
_CACHE_DIR = os.path.join(_DIR, "data_cache")

# Cache-schema version — bump when the priced-bundle shape changes so stale
# bundles regenerate once instead of being served with a missing field.
BUNDLE_VERSION = 2

# Leagues the free football-data.org tier serves standings + fixtures for, so a
# LIVE projection is possible. Our league code == the API competition code for
# all of these. Everything else falls back to the hypothetical mode.
API_LEAGUES = {"PL", "ELC", "PD", "SA", "BL1", "FL1", "DED", "PPL", "BSA"}

# How long a priced projection (standings + model-priced fixtures) stays warm.
# Standings only change when a match finishes, so a few hours is plenty and it
# keeps the page instant on reload without re-hitting the rate-limited API or
# re-pricing hundreds of fixtures through the model.
SIM_CACHE_TTL = 6 * 3600


# ─── Pure Monte Carlo cores ────────────────────────────────────────────────

def simulate(fixtures: List[Tuple[int, int, float, float, float]], n_teams: int,
             n_sims: int = 10000, seed: int = 42) -> dict:
    """Points-only season simulation (kept for the unit tests and as the
    simplest possible core).

    `fixtures`: list of (home_idx, away_idx, p_home, p_draw, p_away).
    Returns per-team arrays: title/top4/relegation probabilities, expected
    points, and mean finishing position (1 = top)."""
    rng = np.random.default_rng(seed)
    pts = np.zeros((n_sims, n_teams), dtype=np.float64)
    for hi, ai, ph, pd, _pa in fixtures:
        r = rng.random(n_sims)
        home_win = r < ph
        draw = (r >= ph) & (r < ph + pd)
        away_win = ~home_win & ~draw
        pts[home_win, hi] += 3.0
        pts[draw, hi] += 1.0
        pts[draw, ai] += 1.0
        pts[away_win, ai] += 3.0
    # Tiny random jitter breaks point ties fairly instead of systematically
    # favouring a fixed team order.
    scores = pts + rng.random((n_sims, n_teams)) * 1e-6
    ranks = np.argsort(np.argsort(-scores, axis=1), axis=1)
    relegation_cut = n_teams - 3
    return {
        "title_prob": (ranks == 0).mean(axis=0),
        "top4_prob": (ranks < 4).mean(axis=0),
        "relegation_prob": (ranks >= relegation_cut).mean(axis=0),
        "expected_points": pts.mean(axis=0),
        "mean_position": ranks.mean(axis=0) + 1.0,
    }


def _poisson_pmf_vec(lam: float, kmax: int) -> np.ndarray:
    """Poisson pmf for k = 0..kmax, built by recurrence (no factorial overflow)."""
    lam = max(lam, 1e-6)
    pmf = np.empty(kmax + 1)
    pmf[0] = math.exp(-lam)
    for k in range(1, kmax + 1):
        pmf[k] = pmf[k - 1] * lam / k
    return pmf


def _dc_score_grid(eh: float, ea: float, rho: float,
                   max_goals: int = dixon_coles.MAX_GOALS) -> np.ndarray:
    """Normalized Dixon-Coles scoreline grid: grid[i,j] = P(home i, away j),
    with the low-score tau correction on the (0,0)/(0,1)/(1,0)/(1,1) cells."""
    grid = np.outer(_poisson_pmf_vec(eh, max_goals), _poisson_pmf_vec(ea, max_goals))
    for i, j in ((0, 0), (0, 1), (1, 0), (1, 1)):
        grid[i, j] *= dixon_coles._tau(i, j, eh, ea, rho)
    grid = np.clip(grid, 0.0, None)
    tot = grid.sum()
    return grid / tot if tot > 0 else grid


def _conditional_scoreline_tables(grid: np.ndarray) -> dict:
    """Split a scoreline grid into per-outcome (home/draw/away) lookup tables:
    {outcome: (home_goals[], away_goals[], cdf[])} for inverse-CDF sampling of a
    scoreline CONDITIONAL on the (calibrated) outcome."""
    G = grid.shape[0]
    ii, jj = np.meshgrid(np.arange(G), np.arange(G), indexing="ij")
    hg, ag, p = ii.ravel(), jj.ravel(), grid.ravel()
    tables = {}
    for key, mask, fallback in (("h", hg > ag, (1, 0)),
                                ("d", hg == ag, (0, 0)),
                                ("a", hg < ag, (0, 1))):
        pk = p[mask]
        if pk.sum() <= 0:
            tables[key] = (np.array([fallback[0]]), np.array([fallback[1]]), np.array([1.0]))
        else:
            tables[key] = (hg[mask], ag[mask], np.cumsum(pk) / pk.sum())
    return tables


def mc_stderr(p, n: int) -> np.ndarray:
    """Monte Carlo standard error of an estimated probability: sqrt(p(1-p)/n)."""
    p = np.asarray(p, dtype=np.float64)
    return np.sqrt(np.clip(p * (1.0 - p), 0.0, None) / max(n, 1))


def simulate_projection(fixtures, n_teams: int, start_points, start_gd,
                        n_sims: int = 10000, seed: int = 42,
                        rules=None, start_gf=None,
                        relegation_slots: int = 3, top_slots: int = 4) -> dict:
    """Season simulation that (a) starts from a given points/GD/GF vector — so it
    projects the remainder of a real season — and (b) samples a TRUE Dixon-Coles
    scoreline per fixture (conditional on the calibrated outcome), so goals-for
    is real and tables rank on points → goal difference → goals-for like real
    leagues.

    `fixtures`: (home_idx, away_idx, p_home, p_draw, p_away, exp_home_goals,
    exp_away_goals[, rho]). The 7-tuple form (no rho) is still accepted.
    `rules`: a leagues.LeagueRules for the tier bands (Title/UCL/UEL/Playoff/
    Relegation); when None, falls back to relegation_slots/top_slots.

    Returns the summary arrays, per-tier probabilities with Monte Carlo standard
    errors, projected goals aggregates (GF/GA/GD, BTTS%, Over-2.5%), a full
    position-probability matrix, and a 10th–90th points band."""
    rng = np.random.default_rng(seed)
    pts = np.tile(np.asarray(start_points, dtype=np.float64), (n_sims, 1))
    gd = np.tile(np.asarray(start_gd, dtype=np.float64), (n_sims, 1))
    gf = np.tile(np.asarray(start_gf if start_gf is not None else np.zeros(n_teams),
                            dtype=np.float64), (n_sims, 1))
    btts_sum = np.zeros(n_teams)
    over_sum = np.zeros(n_teams)
    mcount = np.zeros(n_teams)

    for fx in fixtures:
        hi, ai, ph, pd, _pa, eh, ea = fx[:7]
        rho = fx[7] if len(fx) > 7 else dixon_coles.DC_RHO
        tables = _conditional_scoreline_tables(_dc_score_grid(eh, ea, rho))

        r1 = rng.random(n_sims)
        home = r1 < ph
        draw = (r1 >= ph) & (r1 < ph + pd)
        away = ~home & ~draw

        r2 = rng.random(n_sims)
        hg = np.zeros(n_sims, dtype=np.int64)
        ag = np.zeros(n_sims, dtype=np.int64)
        for mask, key in ((home, "h"), (draw, "d"), (away, "a")):
            if not mask.any():
                continue
            cg, ca, cdf = tables[key]
            idx = np.searchsorted(cdf, r2[mask])
            np.clip(idx, 0, len(cdf) - 1, out=idx)
            hg[mask] = cg[idx]
            ag[mask] = ca[idx]

        m = hg - ag
        pts[home, hi] += 3.0
        pts[away, ai] += 3.0
        pts[draw, hi] += 1.0
        pts[draw, ai] += 1.0
        gd[:, hi] += m
        gd[:, ai] -= m
        gf[:, hi] += hg
        gf[:, ai] += ag
        btts = float(((hg > 0) & (ag > 0)).mean())
        over = float(((hg + ag) > 2).mean())
        btts_sum[hi] += btts; btts_sum[ai] += btts
        over_sum[hi] += over; over_sum[ai] += over
        mcount[hi] += 1; mcount[ai] += 1

    # Rank on points, then goal difference, then goals-for (real league order),
    # with a whisper of jitter for the rare exact three-way tie.
    score = pts * 1e6 + gd * 1e3 + gf + rng.random((n_sims, n_teams)) * 1e-6
    ranks = np.argsort(np.argsort(-score, axis=1), axis=1)   # 0 = champion

    position_matrix = np.zeros((n_teams, n_teams), dtype=np.float64)
    for t in range(n_teams):
        position_matrix[t] = np.bincount(ranks[:, t], minlength=n_teams) / n_sims

    if rules is not None:
        cl, uel, pl, rel = rules.cl_slots, rules.uel_slots, rules.playoff_slots, rules.relegation_slots
    else:
        cl, uel, pl, rel = top_slots, 0, 0, relegation_slots
    rel_cut = n_teams - rel
    title = (ranks == 0).mean(axis=0)
    ucl = (ranks < cl).mean(axis=0)
    uel_band = ((ranks >= cl) & (ranks < cl + uel)).mean(axis=0)
    playoff = (((ranks >= rel_cut - pl) & (ranks < rel_cut)).mean(axis=0)
               if pl else np.zeros(n_teams))
    releg = (ranks >= rel_cut).mean(axis=0)

    with np.errstate(invalid="ignore", divide="ignore"):
        btts_pct = np.where(mcount > 0, btts_sum / np.maximum(mcount, 1), 0.0)
        over_pct = np.where(mcount > 0, over_sum / np.maximum(mcount, 1), 0.0)
    p10, p90 = np.percentile(pts, [10, 90], axis=0)
    return {
        "title_prob": title,
        "ucl_prob": ucl,
        "uel_prob": uel_band,
        "playoff_prob": playoff,
        "relegation_prob": releg,
        "top4_prob": (ranks < 4).mean(axis=0),   # kept for back-compat
        "title_se": mc_stderr(title, n_sims),
        "ucl_se": mc_stderr(ucl, n_sims),
        "relegation_se": mc_stderr(releg, n_sims),
        "expected_points": pts.mean(axis=0),
        "points_p10": p10,
        "points_p90": p90,
        "expected_gf": gf.mean(axis=0),
        "expected_gd": gd.mean(axis=0),
        "expected_ga": gf.mean(axis=0) - gd.mean(axis=0),
        "btts_pct": btts_pct,
        "over25_pct": over_pct,
        "mean_position": ranks.mean(axis=0) + 1.0,
        "position_matrix": position_matrix,
        "n_sims": n_sims,
        "rules": {"relegation_slots": rel, "cl_slots": cl,
                  "uel_slots": uel, "playoff_slots": pl},
    }


# ─── Fixture pricing (shared by both modes) ────────────────────────────────

def _dc_expected_goals(home_key: str, away_key: str, stats: dict) -> Tuple[float, float, float]:
    """Dixon-Coles expected goals + rho for a fixture — the same computation
    prepare_prediction_features feeds the model, reused here for the scoreline
    sampling that gives goal difference. Falls back to neutral defaults for a
    team we have no ratings for."""
    home = stats.get(home_key, {})
    away = stats.get(away_key, {})
    rho = home.get("dc_rho", dixon_coles.DC_RHO)
    _, _, _, eh, ea = dixon_coles.match_probabilities(
        home.get("dc_attack", dixon_coles.DC_START), home.get("dc_defense", dixon_coles.DC_START),
        away.get("dc_attack", dixon_coles.DC_START), away.get("dc_defense", dixon_coles.DC_START),
        home.get("league_base_home_goals", dixon_coles.LEAGUE_AVG_HOME_GOALS),
        home.get("league_base_away_goals", dixon_coles.LEAGUE_AVG_AWAY_GOALS),
        rho=rho,
    )
    return eh, ea, rho


def _price_fixtures(pairs, stats, h2h, model, idx) -> list:
    """Price a list of (home_key, away_key, home_idx, away_idx) fixtures into
    the (hi, ai, ph, pd, pa, eh, ea, rho) tuples the simulator consumes. Outcome
    probabilities come from the trained model (in ONE batched call — hundreds
    of fixtures priced together rather than one predict() each); expected goals +
    rho from Dixon-Coles for the conditional scoreline sampling."""
    from data_processor import prepare_prediction_features
    feats = [prepare_prediction_features(h, a, stats, h2h) for h, a, _, _ in pairs]
    probs = model.predict_proba_batch(feats)
    priced = []
    for (home_key, away_key, hi, ai), r in zip(pairs, probs):
        eh, ea, rho = _dc_expected_goals(home_key, away_key, stats)
        priced.append((hi, ai,
                       r.get("Home Win", 0.0), r.get("Draw", 0.0), r.get("Away Win", 0.0),
                       eh, ea, rho))
    return priced


def _load_stats_h2h_model():
    from model_trainer import MatchPredictorModel
    with open(os.path.join(_DIR, "team_stats.json"), encoding="utf-8") as f:
        stats = json.load(f)
    try:
        with open(os.path.join(_DIR, "h2h_stats.json"), encoding="utf-8") as f:
            h2h = json.load(f)
    except (OSError, ValueError):
        h2h = {}
    model = MatchPredictorModel()
    if not model.load():
        raise SystemExit("No trained model found.")
    return stats, h2h, model


# ─── LIVE projection: current standings + real remaining fixtures ──────────

def _resolve_key(stats: dict, name: str, short: str):
    """Map an API team to our team_stats key (exact, short, then verified
    alias) — the same discipline current_leagues.py uses. Returns None if it
    can't be resolved; the caller still seeds the team's real points, it just
    prices its fixtures from neutral defaults."""
    from team_aliases import resolve_team_name
    if name in stats:
        return name
    if short and short in stats:
        return short
    return resolve_team_name(name, stats) or (resolve_team_name(short, stats) if short else None)


def _fetch_live_projection(league_code: str, stats: dict):
    """Pull the current standings + real remaining fixtures from the live API
    and build the priced-projection bundle. Joins standings to fixtures on the
    API's stable numeric team id (not names), so promoted/relegated sides and
    odd name spellings can't mis-align.

    Critically, it seeds from the standings of the SAME season the upcoming
    fixtures belong to. In the off-season the API's default standings are last
    season's *final* table while 'upcoming' fixtures are next season's — seeding
    one from the other stacks a fresh season's points on a completed one and
    yields impossible totals. Raises (→ caller falls back to the hypothetical
    full-season projection) when the target season hasn't kicked off yet, since
    'remaining fixtures on the current table' is then just a season from 0-0."""
    from api_client import fetch_standings, fetch_upcoming_matches

    matches = fetch_upcoming_matches(league_code)
    if not matches:
        raise RuntimeError("no upcoming fixtures")   # season complete

    # The fixtures we simulate belong to one season (the next match's); seed
    # from THAT season's standings, identified by its start year.
    target = matches[0].get("season", {}) or {}
    target_id = target.get("id")
    year = (target.get("startDate") or "")[:4]
    fixtures_raw = [m for m in matches
                    if (m.get("season", {}) or {}).get("id") == target_id] or matches

    data = fetch_standings(league_code, season=year or None)
    total = next((t for t in data.get("standings", []) if t.get("type") == "TOTAL"), None)
    table = (total or {}).get("table", []) if total else []
    if not table or sum(int(r.get("playedGames", 0) or 0) for r in table) == 0:
        # Target season hasn't started — defer to the balanced round-robin,
        # which also carries the correct promoted/relegated team set.
        raise RuntimeError("target season not started")

    order, meta = [], {}
    for row in table:
        team = row.get("team", {})
        tid = team.get("id")
        if tid is None:
            continue
        key = _resolve_key(stats, team.get("name", ""), team.get("shortName", ""))
        meta[tid] = {
            "display": key or team.get("shortName") or team.get("name") or str(tid),
            "key": key,
            "points": int(row.get("points", 0) or 0),
            "gd": int(row.get("goalDifference", 0) or 0),
            "gf": int(row.get("goalsFor", 0) or 0),
            "played": int(row.get("playedGames", 0) or 0),
            "position": int(row.get("position", 0) or 0),
        }
        order.append(tid)
    if len(order) < 4:
        raise RuntimeError(f"only {len(order)} teams in standings")

    idx = {tid: i for i, tid in enumerate(order)}
    pairs = []
    for m in fixtures_raw:
        h, a = m.get("homeTeam", {}).get("id"), m.get("awayTeam", {}).get("id")
        if h in idx and a in idx:
            pairs.append((meta[h]["key"] or meta[h]["display"],
                          meta[a]["key"] or meta[a]["display"], idx[h], idx[a]))
    if not pairs:
        raise RuntimeError("no simulatable remaining fixtures")

    return {
        "mode": "live",
        "teams": [meta[t]["display"] for t in order],
        "current_points": [meta[t]["points"] for t in order],
        "current_gd": [meta[t]["gd"] for t in order],
        "current_gf": [meta[t]["gf"] for t in order],
        "current_position": [meta[t]["position"] for t in order],
        "played": [meta[t]["played"] for t in order],
        "_pairs": pairs,   # (home_key, away_key, hi, ai), dropped after pricing
    }


def _sim_cache_path(league_code: str) -> str:
    return os.path.join(_CACHE_DIR, f"sim_{league_code}.json")


def _cache_signature(model) -> str:
    """Invalidate the cache when the model, team_stats, or bundle schema change."""
    try:
        mtime = int(os.path.getmtime(os.path.join(_DIR, "team_stats.json")))
    except OSError:
        mtime = 0
    return f"v{BUNDLE_VERSION}:{getattr(model, 'version', '?')}:{mtime}"


def _build_hypothetical_bundle(league_code, model, stats, h2h) -> dict:
    """Priced full double round-robin from every team's current rating (0-0).
    Team set from the authoritative current-membership list when available, else
    the team_stats league label."""
    current = {}
    try:
        with open(os.path.join(_DIR, "current_leagues.json"), encoding="utf-8") as f:
            current = json.load(f)
    except (OSError, ValueError):
        current = {}
    if current.get(league_code):
        teams = sorted(t for t in current[league_code]
                       if stats.get(t, {}).get("matches_played", 0) > 0)
    else:
        teams = sorted(t for t, s in stats.items()
                       if s.get("league") == league_code and s.get("matches_played", 0) > 0)
    if len(teams) < 4:
        raise SystemExit(f"League {league_code}: only {len(teams)} teams with data — need >= 4.")

    idx = {t: i for i, t in enumerate(teams)}
    pairs = [(h, a, idx[h], idx[a]) for h, a in itertools.permutations(teams, 2)]
    priced = _price_fixtures(pairs, stats, h2h, model, idx)
    n = len(teams)
    return {"mode": "hypothetical", "teams": teams, "fixtures": priced,
            "current_points": None, "current_gd": [0] * n, "current_gf": [0] * n,
            "current_position": None, "played": None}


def _build_live_bundle(league_code, model, stats, h2h) -> dict:
    """Priced remaining-fixtures projection seeded on the live standings.
    Raises if the league isn't in-season/covered (caller falls back)."""
    live = _fetch_live_projection(league_code, stats)
    pairs = live.pop("_pairs")
    idx = {t: i for i, t in enumerate(live["teams"])}
    live["fixtures"] = _price_fixtures(pairs, stats, h2h, model, idx)
    return live


def _build_bundle(league_code, model, stats, h2h) -> dict:
    """Live projection for a covered, in-season league; hypothetical otherwise."""
    if league_code in API_LEAGUES:
        try:
            return _build_live_bundle(league_code, model, stats, h2h)
        except SystemExit:
            raise
        except Exception:
            pass   # no token / API error / off-season → hypothetical
    return _build_hypothetical_bundle(league_code, model, stats, h2h)


def simulate_league(league_code: str, n_sims: int = 20000, force: bool = False,
                    seed: int = 42) -> dict:
    """Front door for the app. Builds (and caches) the priced projection bundle
    — live standings + real remaining fixtures where the API covers the league
    and the season is under way, otherwise a hypothetical full season — then
    runs the Monte Carlo. The priced bundle is cached so reloads are instant and
    don't re-hit the rate-limited API or re-price hundreds of fixtures; the fast
    vectorised sim itself re-runs each call. `seed` fixes the RNG so a reload
    shows the identical table (the trustworthy default); the app's Re-simulate
    button passes a fresh seed so a user can watch the Monte Carlo vary."""
    stats, h2h, model = _load_stats_h2h_model()
    sig = _cache_signature(model)
    path = _sim_cache_path(league_code)

    bundle = None
    if not force and os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as f:
                cached = json.load(f)
            if cached.get("sig") == sig and (time.time() - cached.get("computed_at", 0)) < SIM_CACHE_TTL:
                bundle = cached
        except (OSError, ValueError):
            bundle = None

    if bundle is None:
        bundle = _build_bundle(league_code, model, stats, h2h)
        bundle["sig"] = sig
        bundle["computed_at"] = time.time()
        os.makedirs(_CACHE_DIR, exist_ok=True)
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(bundle, f)
        except OSError:
            pass

    n_teams = len(bundle["teams"])
    start_points = bundle.get("current_points") or [0] * n_teams
    start_gd = bundle.get("current_gd") or [0] * n_teams
    start_gf = bundle.get("current_gf") or [0] * n_teams
    rules = leagues.league_rules(league_code, n_teams)
    res = simulate_projection(bundle["fixtures"], n_teams, start_points, start_gd,
                              n_sims=n_sims, seed=seed, rules=rules, start_gf=start_gf)
    return {
        "mode": bundle["mode"],
        "teams": bundle["teams"],
        "current_points": bundle.get("current_points"),
        "current_position": bundle.get("current_position"),
        "played": bundle.get("played"),
        "remaining": len(bundle["fixtures"]),
        "as_of": time.strftime("%d %b %Y", time.localtime(bundle.get("computed_at", time.time()))),
        **res,
    }


def run_league(league_code: str, n_sims: int = 10000) -> dict:
    """Back-compat wrapper: an uncached hypothetical full-season projection.
    (`simulate_league` is the cached front door the app uses.)"""
    stats, h2h, model = _load_stats_h2h_model()
    bundle = _build_hypothetical_bundle(league_code, model, stats, h2h)
    n = len(bundle["teams"])
    res = simulate_projection(bundle["fixtures"], n, np.zeros(n), np.zeros(n), n_sims=n_sims)
    return {"mode": "hypothetical", "teams": bundle["teams"], "remaining": len(bundle["fixtures"]),
            "current_points": None, "current_position": None, "played": None,
            "as_of": None, **res}


# ─── CLI ───────────────────────────────────────────────────────────────────

def _print_table(out: dict):
    teams = out["teams"]
    order = np.argsort(-np.asarray(out["title_prob"]) + np.asarray(out["mean_position"]) * 1e-6)
    live = out.get("mode") == "live"
    print(f"{'Team':22} {'Now':>4} {'ExpPts':>7} {'Title%':>7} {'Top4%':>7} {'Releg%':>7} {'AvgPos':>7}")
    for i in order:
        now = out["current_points"][i] if live and out.get("current_points") else 0
        print(f"{teams[i][:22]:22} {now:>4} {out['expected_points'][i]:7.1f} "
              f"{out['title_prob'][i]*100:7.1f} {out['top4_prob'][i]*100:7.1f} "
              f"{out['relegation_prob'][i]*100:7.1f} {out['mean_position'][i]:7.1f}")


if __name__ == "__main__":
    league = sys.argv[1] if len(sys.argv) > 1 else "PL"
    sims = int(sys.argv[2]) if len(sys.argv) > 2 else 10000
    out = simulate_league(league, sims)
    mode = "LIVE — current standings + remaining fixtures" if out["mode"] == "live" \
        else "hypothetical — full season from current ratings"
    hdr = f"Simulating {league} — {sims:,} seasons ({mode})"
    if out.get("as_of"):
        hdr += f"\nStandings as of {out['as_of']}; {out['remaining']} fixtures remaining"
    print(hdr + "\n")
    _print_table(out)
    print("\nNote: a Monte Carlo projection built on the model's calibrated match "
          "probabilities. It aggregates them across a season — it does not change any "
          "single-match forecast.")
