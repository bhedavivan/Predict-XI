"""
Live prediction track record — the honest proof the model works out-of-sample.

The model reports a leak-free holdout RPS, but that is still a measurement on
past data. This logs a real, dated forecast for each UPCOMING fixture, then —
once the match is actually played — settles it against the real result and
scores the running accuracy / RPS / log-loss / calibration. Nothing here can
be optimistic: every prediction is recorded strictly BEFORE kickoff and graded
against reality.

Workflow (run periodically — e.g. a weekly cron, or the /schedule skill):
    python track_record.py log       # log predictions for the next fixtures
    python track_record.py settle    # grade any logged fixtures now finished
    python track_record.py score     # print the current record

The /track page displays it. Predictions are keyed by football-data.org's stable
numeric match id, so settling is an exact join — no name guessing. Covers the
eight top flights the free API serves live.
"""

import json
import os
import sys
import time

_DIR = os.path.dirname(os.path.abspath(__file__))
_LOG_PATH = os.path.join(_DIR, "predictions_log.json")

# Home Win / Draw / Away Win <-> the model's integer classes.
_LABEL_TO_CLASS = {"Home Win": 1, "Draw": 0, "Away Win": -1}
_CLASSES = [-1, 0, 1]   # away, draw, home (model order)


def _live_leagues():
    import leagues
    return [lg.code for lg in leagues.REGISTRY if lg.live_api]


def load_log() -> list:
    try:
        with open(_LOG_PATH, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return []


def save_log(records: list) -> None:
    with open(_LOG_PATH, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)


def _result_label(home_goals: int, away_goals: int) -> str:
    if home_goals > away_goals:
        return "Home Win"
    if home_goals < away_goals:
        return "Away Win"
    return "Draw"


# ─── Logging predictions on upcoming fixtures ──────────────────────────────

def log_predictions(days_ahead: int = 10, leagues_only=None) -> int:
    """Predict every upcoming fixture within `days_ahead` days (for the live
    leagues) and append the NEW ones to the log. Idempotent — a fixture already
    logged is skipped, so re-running never double-counts. Returns how many were
    added."""
    from api_client import fetch_upcoming_matches, MissingTokenError
    from model_trainer import MatchPredictorModel
    from data_processor import prepare_prediction_features
    from team_aliases import resolve_team_name, team_display_name

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

    records = load_log()
    seen = {r["match_id"] for r in records}
    cutoff = time.time() + days_ahead * 86400
    today = time.strftime("%Y-%m-%d")
    added = 0

    for code in (leagues_only or _live_leagues()):
        try:
            matches = fetch_upcoming_matches(code)
        except MissingTokenError:
            raise
        except Exception as e:
            print(f"  {code}: skipped ({e})")
            continue
        pending = []
        for m in matches:
            mid = m.get("id")
            if mid is None or mid in seen:
                continue
            utc = m.get("utcDate", "")
            try:
                ts = time.mktime(time.strptime(utc[:19], "%Y-%m-%dT%H:%M:%S"))
            except ValueError:
                continue
            if ts > cutoff:
                continue
            h_api, a_api = m.get("homeTeam", {}), m.get("awayTeam", {})
            home = resolve_team_name(h_api.get("name", ""), stats) or resolve_team_name(h_api.get("shortName", ""), stats)
            away = resolve_team_name(a_api.get("name", ""), stats) or resolve_team_name(a_api.get("shortName", ""), stats)
            if not home or not away:
                continue   # can't predict an unknown team honestly
            pending.append((mid, utc, code, home, away, m))
        if pending:
            probs = model.predict_proba_batch(
                [prepare_prediction_features(h, a, stats, h2h) for _, _, _, h, a, _ in pending])
            for (mid, utc, code, home, away, _m), pr in zip(pending, probs):
                p = {"Home Win": round(pr.get("Home Win", 0.0), 4),
                     "Draw": round(pr.get("Draw", 0.0), 4),
                     "Away Win": round(pr.get("Away Win", 0.0), 4)}
                records.append({
                    "match_id": mid, "utc_date": utc, "league": code,
                    "home": home, "away": away,
                    "home_display": team_display_name(home),
                    "away_display": team_display_name(away),
                    "probs": p, "pick": max(p, key=p.get),
                    "logged_at": today, "status": "pending",
                    "actual": None, "actual_score": None, "settled_at": None,
                })
                seen.add(mid)
                added += 1
        print(f"  {code}: {len(pending)} new predictions logged")

    save_log(records)
    print(f"\nLogged {added} new predictions ({len(records)} total).")
    return added


# ─── Settling logged predictions against real results ──────────────────────

def settle_predictions() -> int:
    """Grade any logged-but-pending fixtures that have now finished. Returns how
    many were settled."""
    from api_client import fetch_finished_matches, MissingTokenError

    records = load_log()
    pending = {r["match_id"]: r for r in records if r["status"] == "pending"}
    if not pending:
        print("Nothing pending to settle.")
        return 0

    leagues_with_pending = sorted({r["league"] for r in pending.values()})
    finished = {}
    for code in leagues_with_pending:
        try:
            for m in fetch_finished_matches(code):
                finished[m.get("id")] = m
        except MissingTokenError:
            raise
        except Exception as e:
            print(f"  {code}: skipped ({e})")

    today = time.strftime("%Y-%m-%d")
    settled = 0
    for mid, rec in pending.items():
        m = finished.get(mid)
        if not m:
            continue
        ft = m.get("score", {}).get("fullTime", {})
        hg, ag = ft.get("home"), ft.get("away")
        if hg is None or ag is None:
            continue
        rec["actual"] = _result_label(hg, ag)
        rec["actual_score"] = f"{hg}-{ag}"
        rec["status"] = "settled"
        rec["settled_at"] = today
        settled += 1

    save_log(records)
    print(f"Settled {settled} predictions.")
    return settled


# ─── Scoring the record ────────────────────────────────────────────────────

def score(records=None) -> dict:
    """Score the settled predictions: accuracy, RPS, log-loss, Brier, a
    confidence-calibration table, per-league breakdown, and the recent results."""
    from model_trainer import _rps
    import math

    records = records if records is not None else load_log()
    settled = [r for r in records if r.get("status") == "settled" and r.get("actual")]
    n_settled = len(settled)
    n_pending = sum(1 for r in records if r.get("status") == "pending")

    out = {"n_logged": len(records), "n_pending": n_pending, "n_settled": n_settled,
           "accuracy": None, "rps": None, "log_loss": None, "brier": None,
           "baseline_acc": None, "calibration": [], "per_league": [], "recent": []}
    if not settled:
        return out

    def _vec(r):   # [away, draw, home], matching _CLASSES order
        p = r["probs"]
        return [p["Away Win"], p["Draw"], p["Home Win"]]

    correct = sum(1 for r in settled if r["pick"] == r["actual"])
    home_actual = sum(1 for r in settled if r["actual"] == "Home Win")
    proba = [_vec(r) for r in settled]
    y_true = [_LABEL_TO_CLASS[r["actual"]] for r in settled]

    ll = brier = 0.0
    for r in settled:
        p = r["probs"]
        for lab in ("Home Win", "Draw", "Away Win"):
            hit = 1.0 if r["actual"] == lab else 0.0
            brier += (p[lab] - hit) ** 2
        ll += -math.log(max(p[r["actual"]], 1e-15))

    out["accuracy"] = round(correct / n_settled, 4)
    out["baseline_acc"] = round(home_actual / n_settled, 4)   # always-pick-home
    out["rps"] = round(_rps(y_true, proba, _CLASSES), 4)
    out["log_loss"] = round(ll / n_settled, 4)
    out["brier"] = round(brier / n_settled, 4)

    # Confidence calibration: bin by the pick's probability, compare stated
    # confidence to the rate it actually came in.
    bins = [(0.0, 0.40), (0.40, 0.50), (0.50, 0.60), (0.60, 0.70), (0.70, 0.80), (0.80, 1.01)]
    for lo, hi in bins:
        grp = [r for r in settled if lo <= max(r["probs"].values()) < hi]
        if grp:
            conf = sum(max(r["probs"].values()) for r in grp) / len(grp)
            acc = sum(1 for r in grp if r["pick"] == r["actual"]) / len(grp)
            out["calibration"].append({
                "band": f"{int(lo*100)}–{int(hi*100) if hi <= 1 else 100}%",
                "predicted": round(conf * 100, 1), "actual": round(acc * 100, 1), "n": len(grp)})

    by_lg = {}
    for r in settled:
        by_lg.setdefault(r["league"], []).append(r)
    import leagues as _lg
    for code, grp in by_lg.items():
        yv = [_LABEL_TO_CLASS[r["actual"]] for r in grp]
        pv = [_vec(r) for r in grp]
        out["per_league"].append({
            "league": code, "name": _lg.display_name(code), "n": len(grp),
            "accuracy": round(sum(1 for r in grp if r["pick"] == r["actual"]) / len(grp), 3),
            "rps": round(_rps(yv, pv, _CLASSES), 4)})
    out["per_league"].sort(key=lambda x: -x["n"])

    for r in sorted(settled, key=lambda r: r.get("settled_at", ""), reverse=True)[:20]:
        out["recent"].append({
            "home": r["home_display"], "away": r["away_display"], "league": r["league"],
            "pick": r["pick"], "actual": r["actual"], "score": r["actual_score"],
            "correct": r["pick"] == r["actual"],
            "conf": round(max(r["probs"].values()) * 100)})
    return out


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "score"
    if cmd == "log":
        log_predictions(int(sys.argv[2]) if len(sys.argv) > 2 else 10)
    elif cmd == "settle":
        settle_predictions()
    elif cmd == "score":
        s = score()
        print(f"Logged {s['n_logged']} | pending {s['n_pending']} | settled {s['n_settled']}")
        if s["n_settled"]:
            print(f"Accuracy {s['accuracy']:.1%} (always-home {s['baseline_acc']:.1%}) | "
                  f"RPS {s['rps']} | log-loss {s['log_loss']}")
    else:
        print("usage: python track_record.py [log [days] | settle | score]")


if __name__ == "__main__":
    main()
