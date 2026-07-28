"""
Honest evaluation of the shipped model: where it works, where it doesn't,
and whether its probabilities mean what they say.

A single global accuracy figure across 30 leagues hides a lot. This produces
three things from one held-out slice:

  1. Per-league accuracy / macro-F1 / draw recall — the model is not equally
     good everywhere, and saying so is more useful than one headline number.
  2. A reliability curve — when the model says 60%, does it happen 60% of the
     time? Calibration is what makes a displayed probability meaningful.
  3. A draw-threshold trade-off curve. Argmax on calibrated probabilities is
     accuracy-optimal, so predicting more draws necessarily costs accuracy.
     Rather than assume the size of that trade, this measures it, so the
     choice to keep argmax (or not) is evidence-based.

Writes evaluation.json, consumed by the dashboard.

Run: python evaluate.py
"""

import json
import os
from collections import defaultdict
from typing import Dict, List

import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Fraction of the (time-ordered) data held out for evaluation. Matches the
# tail-holdout convention used for the probabilistic metrics in
# model_trainer, so numbers here are comparable to those.
HOLDOUT_FRACTION = 0.2

DRAW_CLASS = 0
LABELS = {-1: "Away Win", 0: "Draw", 1: "Home Win"}


def _load_json(name):
    try:
        with open(os.path.join(_SCRIPT_DIR, name)) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray, classes: List[int]) -> float:
    f1s = []
    for c in classes:
        tp = int(np.sum((y_true == c) & (y_pred == c)))
        fp = int(np.sum((y_true != c) & (y_pred == c)))
        fn = int(np.sum((y_true == c) & (y_pred != c)))
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return float(np.mean(f1s))


def _recall(y_true: np.ndarray, y_pred: np.ndarray, cls: int) -> float:
    denom = int(np.sum(y_true == cls))
    if not denom:
        return float("nan")
    return float(np.sum((y_true == cls) & (y_pred == cls)) / denom)


# RPS is defined once, in model_trainer, and reused here so the training-time
# and evaluation-time scores can never drift apart (this project has repeatedly
# been bitten by two copies of the same logic diverging).
from model_trainer import _rps


def _log_loss(y_true: np.ndarray, proba: np.ndarray, classes) -> float:
    idx = {c: i for i, c in enumerate(classes)}
    y = np.asarray(y_true)
    if len(y) == 0:
        return float("nan")
    p = np.clip(proba[np.arange(len(y)), [idx[v] for v in y]], 1e-15, 1.0)
    return float(np.mean(-np.log(p)))


def per_league_breakdown(y_true, y_pred, leagues, classes, min_matches=100) -> List[dict]:
    """Accuracy/macro-F1/draw-recall per competition.

    Leagues below `min_matches` are pooled into one "other" row rather than
    reported individually — a 45% accuracy computed over 30 matches is noise
    dressed as a finding.
    """
    by = defaultdict(list)
    for i, lg in enumerate(leagues):
        by[lg or "unknown"].append(i)

    rows, small = [], []
    for lg, idx in by.items():
        if len(idx) < min_matches:
            small.extend(idx)
            continue
        idx = np.asarray(idx)
        yt, yp = y_true[idx], y_pred[idx]
        rows.append({
            "league": lg,
            "matches": int(len(idx)),
            "accuracy": round(float(np.mean(yt == yp)), 4),
            "macro_f1": round(_macro_f1(yt, yp, classes), 4),
            "draw_recall": round(_recall(yt, yp, DRAW_CLASS), 4),
            "home_win_rate": round(float(np.mean(yt == 1)), 4),
        })
    rows.sort(key=lambda r: -r["accuracy"])
    if small:
        idx = np.asarray(small)
        yt, yp = y_true[idx], y_pred[idx]
        rows.append({
            "league": f"(pooled: {len(by) - len(rows)} leagues under {min_matches} matches)",
            "matches": int(len(idx)),
            "accuracy": round(float(np.mean(yt == yp)), 4),
            "macro_f1": round(_macro_f1(yt, yp, classes), 4),
            "draw_recall": round(_recall(yt, yp, DRAW_CLASS), 4),
            "home_win_rate": round(float(np.mean(yt == 1)), 4),
            "pooled": True,
        })
    return rows


def reliability_curve(y_true, proba, classes, n_bins=10) -> List[dict]:
    """Bin every predicted class-probability and compare it to how often that
    class actually occurred. A well-calibrated model sits on the diagonal."""
    conf = proba.max(axis=1)
    pred = np.asarray(classes)[proba.argmax(axis=1)]
    correct = (pred == y_true).astype(float)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf >= lo) & (conf < hi if hi < 1.0 else conf <= hi)
        n = int(mask.sum())
        if not n:
            continue
        out.append({
            "bin_low": round(float(lo), 2),
            "bin_high": round(float(hi), 2),
            "n": n,
            "mean_confidence": round(float(conf[mask].mean()), 4),
            "observed_accuracy": round(float(correct[mask].mean()), 4),
        })
    return out


def _score_rule(y_true, proba, classes, draw_idx, non_draw, cls_arr, t=None) -> dict:
    if t is None:
        pred = cls_arr[proba.argmax(axis=1)]
    else:
        best_non_draw = cls_arr[np.asarray(non_draw)][proba[:, non_draw].argmax(axis=1)]
        pred = np.where(proba[:, draw_idx] >= t, DRAW_CLASS, best_non_draw)
    return {
        "threshold": t,
        "accuracy": round(float(np.mean(pred == y_true)), 4),
        "macro_f1": round(_macro_f1(y_true, pred, classes), 4),
        "draw_recall": round(_recall(y_true, pred, DRAW_CLASS), 4),
    }


# How much overall accuracy we'll trade to give the draw class a real voice.
# Without odds, draw probability tops out ~0.31, so a draw is never the argmax
# and plain accuracy is maximised by never predicting one — but that makes the
# draw class useless. Optimising MACRO-F1 (which counts the draw class equally)
# within this accuracy budget lets the model predict draws at roughly their
# true base rate; the budget stops it running away to "predict draws for half
# the fixtures" at barely-above-chance precision.
DRAW_ACC_TOLERANCE = 0.04


def draw_threshold_curve(y_true, proba, classes) -> dict:
    """Choose the draw-probability threshold that maximises MACRO-F1 within a
    small accuracy budget (DRAW_ACC_TOLERANCE), validated out-of-sample.

    Rule: predict Draw when P(draw) >= t, else argmax over the other two.

    The threshold is chosen on the FIRST half of this slice and scored on the
    SECOND half — essential, not ceremony: predicted draw probabilities are
    tightly clustered, so the boundary sits in a dense region where a 0.01
    shift swings thousands of predictions, and tuning+scoring on the same slice
    inflates the gain. Only a macro-F1 gain that survives on unseen data (while
    keeping accuracy within budget) is adopted.
    """
    classes = list(classes)
    draw_idx = classes.index(DRAW_CLASS)
    non_draw = [i for i in range(len(classes)) if i != draw_idx]
    cls_arr = np.asarray(classes)

    mid = len(y_true) // 2
    tune_y, tune_p = y_true[:mid], proba[:mid]
    test_y, test_p = y_true[mid:], proba[mid:]

    grid = [round(x, 2) for x in np.arange(0.20, 0.51, 0.01)]
    tune_pts = [_score_rule(tune_y, tune_p, classes, draw_idx, non_draw, cls_arr, t)
                for t in grid]
    tune_base = _score_rule(tune_y, tune_p, classes, draw_idx, non_draw, cls_arr, None)

    # Best macro-F1 on the tuning half, allowing accuracy to fall by at most the
    # budget (draws are worth a few points of accuracy; a model that never says
    # "draw" is not).
    viable = [p for p in tune_pts
              if p["accuracy"] >= tune_base["accuracy"] - DRAW_ACC_TOLERANCE
              and p["macro_f1"] > tune_base["macro_f1"]]
    viable.sort(key=lambda p: -p["macro_f1"])
    chosen = viable[0]["threshold"] if viable else None

    test_base = _score_rule(test_y, test_p, classes, draw_idx, non_draw, cls_arr, None)
    test_chosen = (_score_rule(test_y, test_p, classes, draw_idx, non_draw, cls_arr, chosen)
                   if chosen is not None else None)

    # Survives if macro-F1 still improves out-of-sample and accuracy stays
    # within budget there too.
    holds_up = bool(
        test_chosen
        and test_chosen["macro_f1"] > test_base["macro_f1"]
        and test_chosen["accuracy"] >= test_base["accuracy"] - DRAW_ACC_TOLERANCE
    )
    if chosen is None:
        verdict = ("keep argmax — no threshold improved macro-F1 on the tuning half "
                   "within the accuracy budget.")
    elif holds_up:
        verdict = (f"adopt threshold {chosen:.2f} — macro-F1 improved out-of-sample "
                   f"({test_base['macro_f1']} -> {test_chosen['macro_f1']}) for "
                   f"{test_base['accuracy']} -> {test_chosen['accuracy']} accuracy "
                   f"(draw recall {test_base['draw_recall']} -> {test_chosen['draw_recall']}).")
    else:
        verdict = (f"keep argmax — threshold {chosen:.2f} improved tuning macro-F1 but did "
                   f"NOT survive out-of-sample (macro-F1 {test_base['macro_f1']} -> "
                   f"{test_chosen['macro_f1']}). Selection bias, not signal.")

    return {
        "method": "threshold tuned on first half of holdout, scored on second half",
        "tuning_curve": tune_pts,
        "tuning_baseline": tune_base,
        "chosen_threshold": chosen,
        "test_argmax": test_base,
        "test_with_threshold": test_chosen,
        "survives_holdout": holds_up,
        "verdict": verdict,
    }


def main():
    from model_trainer import MatchPredictorModel
    from csv_data_loader import load_processed_data

    data = load_processed_data()
    if not data:
        raise SystemExit("processed_data.json not found — train the model first.")
    X = np.asarray(data["X"], dtype=np.float64)
    y = np.asarray(data["y"])
    leagues = data.get("leagues") or []

    model = MatchPredictorModel()
    if not model.load():
        raise SystemExit("No trained model found.")
    classes = list(model.classes)

    # Prefer the leak-free holdout predictions written at training time: those
    # come from an EVAL pipeline that never trained on these rows, so per-league
    # accuracy and the reliability curve reflect real out-of-sample behaviour.
    # Falling back to re-scoring the shipped (all-data) pipeline on its own tail
    # would be in-sample and optimistic — the whole point of the fix.
    holdout = _load_json("holdout_eval.json")
    if holdout and holdout.get("proba"):
        proba = np.asarray(holdout["proba"], dtype=np.float64)
        yte = np.asarray(holdout["y"])
        cal_hi = int(holdout.get("cal_hi", len(X) - len(yte)))
        n_test = len(yte)
        lg_te = leagues[cal_hi:cal_hi + n_test] if len(leagues) == len(X) else ["unknown"] * n_test
        print("Using leak-free holdout_eval.json (out-of-sample eval pipeline).")
    else:
        n_hold = max(int(len(X) * HOLDOUT_FRACTION), 1)
        half = n_hold // 2
        n_test = n_hold - half
        Xte, yte = X[-n_test:], y[-n_test:]
        lg_te = leagues[-n_test:] if len(leagues) == len(X) else ["unknown"] * n_test
        proba = model.apply_calibration(model.pipeline.predict_proba(Xte))
        print("WARNING: holdout_eval.json missing; falling back to IN-SAMPLE tail "
              "(optimistic). Retrain to regenerate the leak-free holdout.")

    pred = np.asarray(classes)[proba.argmax(axis=1)]

    threshold = getattr(model, "draw_threshold", None)
    if threshold is not None:
        di = classes.index(DRAW_CLASS)
        nd = [i for i in range(len(classes)) if i != di]
        best_nd = np.asarray(classes)[np.asarray(nd)][proba[:, nd].argmax(axis=1)]
        rule_pred = np.where(proba[:, di] >= threshold, DRAW_CLASS, best_nd)
    else:
        rule_pred = pred

    out = {
        "holdout_matches": int(n_test),
        "holdout_fraction": HOLDOUT_FRACTION,
        # The scalar threshold the shipped model actually uses. Named distinctly
        # from the "draw_threshold" analysis curve below — a plain
        # "draw_threshold" key appeared twice and the dict literal silently kept
        # only the last (the curve), so this scalar never reached evaluation.json.
        "draw_threshold_shipped": threshold,
        # "overall" reflects the SHIPPED decision rule (threshold if one is
        # set). The argmax figures are kept alongside so the effect of the
        # rule is visible rather than folded invisibly into one number.
        # RPS and log-loss are properties of the PROBABILITIES, so they are
        # the same under argmax or the draw-threshold rule (the rule only
        # changes the discrete pick). They are the primary metrics here:
        # accuracy is dominated by the home class and dead-ceilinged, whereas
        # RPS/log-loss are what the literature reports and what actually moves
        # when calibration or features improve.
        "rps": round(_rps(yte, proba, classes), 4),
        "log_loss": round(_log_loss(yte, proba, classes), 4),
        "overall": {
            "accuracy": round(float(np.mean(rule_pred == yte)), 4),
            "macro_f1": round(_macro_f1(yte, rule_pred, classes), 4),
            "draw_recall": round(_recall(yte, rule_pred, DRAW_CLASS), 4),
        },
        "overall_argmax": {
            "accuracy": round(float(np.mean(pred == yte)), 4),
            "macro_f1": round(_macro_f1(yte, pred, classes), 4),
            "draw_recall": round(_recall(yte, pred, DRAW_CLASS), 4),
        },
        "per_league": per_league_breakdown(yte, rule_pred, lg_te, classes),
        "reliability": reliability_curve(yte, proba, classes),
        "draw_threshold": draw_threshold_curve(yte, proba, classes),
    }
    with open(os.path.join(_SCRIPT_DIR, "evaluation.json"), "w") as f:
        json.dump(out, f, indent=2)

    print(f"Holdout: {n_test} matches")
    print(f"RPS={out['rps']}  log_loss={out['log_loss']}  "
          f"(primary; lower is better, ~0.19-0.21 is published no-odds SOTA)")
    print(f"Overall: acc={out['overall']['accuracy']} "
          f"macro_f1={out['overall']['macro_f1']} draw_recall={out['overall']['draw_recall']}")
    print("\nPer-league (best to worst):")
    for r in out["per_league"]:
        print(f"  {r['league']:12} n={r['matches']:5} acc={r['accuracy']:.3f} "
              f"f1={r['macro_f1']:.3f} draw_rec={r['draw_recall']:.3f}")
    print("\nDraw threshold:", out["draw_threshold"]["verdict"])
    print("Wrote evaluation.json")


if __name__ == "__main__":
    main()
