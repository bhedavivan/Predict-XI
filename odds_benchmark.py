"""OFFLINE-ONLY market-ceiling benchmark.

Computes de-vigged bookmaker (Bet365) RPS and log-loss over recent matches, so
the model's RPS can be read against what the market itself achieves. This is
the honest anchor the research turned up: published no-odds cross-league models
land only ~0.001-0.004 RPS above the de-vigged market, so if the model is near
this number there is essentially no accuracy headroom left.

CRITICAL: bookmaker odds are used ONLY here, as a yardstick. They are never a
model feature (see README "Decisions"): the app predicts arbitrary cross-league
matchups that have no odds, and the modelling is meant to be the owner's own.

Run: python odds_benchmark.py
"""
import csv
import glob
import io
import os

import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(_SCRIPT_DIR, "data_cache")

# football-data.co.uk main-feed 1X2 odds columns, in preference order. Bet365
# is the most complete; Pinnacle (PS) and the market average are fallbacks.
ODDS_TRIPLES = [("B365H", "B365D", "B365A"), ("PSH", "PSD", "PSA"),
                ("AvgH", "AvgD", "AvgA"), ("BbAvH", "BbAvD", "BbAvA")]


def _devig(oh, od, oa):
    try:
        ih, idr, ia = 1.0 / oh, 1.0 / od, 1.0 / oa
    except ZeroDivisionError:
        return None
    s = ih + idr + ia
    if s <= 0:
        return None
    return ih / s, idr / s, ia / s  # P(home), P(draw), P(away)


def market_probs_and_outcomes():
    """Yield (p_away, p_draw, p_home, outcome) for every recent match whose
    cached CSV carries usable 1X2 odds. Outcome is -1/0/1 (away/draw/home)."""
    rows = []
    files = glob.glob(os.path.join(CACHE_DIR, "*_fdcouk_*.csv"))
    for path in files:
        base = os.path.basename(path)
        if "fdcouk_new_" in base:
            continue  # new-leagues feed uses different column names; skip here
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
        except OSError:
            continue
        reader = csv.DictReader(io.StringIO(content))
        fields = reader.fieldnames or []
        triple = next((t for t in ODDS_TRIPLES if all(c in fields for c in t)), None)
        if triple is None:
            continue
        hc, dc, ac = triple
        for r in reader:
            ftr = (r.get("FTR") or "").strip()
            if ftr not in ("H", "D", "A"):
                continue
            try:
                oh, od, oa = float(r.get(hc, "")), float(r.get(dc, "")), float(r.get(ac, ""))
            except (ValueError, TypeError):
                continue
            dv = _devig(oh, od, oa)
            if dv is None:
                continue
            p_home, p_draw, p_away = dv
            outcome = {"H": 1, "D": 0, "A": -1}[ftr]
            rows.append((p_away, p_draw, p_home, outcome))
    return rows


def main():
    rows = market_probs_and_outcomes()
    if not rows:
        print("No cached CSVs with odds columns found. Run a training/data pass first "
              "(the cache is populated by csv_data_loader).")
        return
    P = np.array([[r[0], r[1], r[2]] for r in rows], dtype=np.float64)  # [away, draw, home]
    y = np.array([r[3] for r in rows])
    classes = np.array([-1, 0, 1])  # ordinal: away < draw < home
    onehot = (classes[None, :] == y[:, None]).astype(np.float64)

    cum_p = np.cumsum(P, axis=1)[:, :-1]
    cum_o = np.cumsum(onehot, axis=1)[:, :-1]
    rps = float(np.mean(np.sum((cum_p - cum_o) ** 2, axis=1) / (P.shape[1] - 1)))

    idx = {-1: 0, 0: 1, 1: 2}
    p_true = np.clip(np.array([P[i, idx[y[i]]] for i in range(len(y))]), 1e-15, 1.0)
    log_loss = float(np.mean(-np.log(p_true)))
    acc = float(np.mean(classes[P.argmax(1)] == y))

    print(f"De-vigged market benchmark over {len(y):,} recent matches with odds:")
    print(f"  RPS       = {rps:.4f}")
    print(f"  log-loss  = {log_loss:.4f}")
    print(f"  accuracy  = {acc:.4f}  (bookmaker top-1, for reference)")
    print("\nThe model's holdout RPS (evaluate.py / model_metrics.json) can be read "
          "against this. Published no-odds models land ~0.001-0.004 RPS above it.")


if __name__ == "__main__":
    main()
