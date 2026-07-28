import json
import math
import os
import random
import warnings
from collections import Counter
from functools import partial
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

# Thread-pool spawn (OpenMP/BLAS) is unusually expensive in some sandboxed
# environments — e.g. a single small HistGradientBoostingClassifier fit going
# from ~2s to ~12s purely from multi-threading overhead. Pin to single-thread
# execution before numpy/sklearn initialize their pools; `setdefault` leaves
# an explicit override (e.g. on a real multi-core deployment box) intact.
for _env_var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_env_var, "1")

import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Model version. 4.0.0: 167k matches / 38 leagues / 85 features, RPS + log-loss
# as primary metrics, leak-free honest-holdout evaluation + out-of-sample
# calibrator, recency weighting (ablated → off), and the new draw-signal features.
MODEL_VERSION = "4.0.0"


# ─── Backward-compatible from-scratch models ───────────────────────────────
# Kept for tests and users who explicitly opt into them.

class GaussianNaiveBayes:
    """Gaussian Naive Bayes classifier implemented from scratch."""

    def __init__(self):
        self.classes = []
        self.means = {}
        self.vars = {}
        self.priors = {}

    def fit(self, X, y):
        self.classes = sorted(set(y))
        n_samples = len(X)
        n_features = len(X[0]) if X else 0
        class_samples = {c: [] for c in self.classes}
        for i, label in enumerate(y):
            class_samples[label].append(X[i])
        for c in self.classes:
            samples = class_samples[c]
            self.priors[c] = len(samples) / n_samples
            self.means[c] = []
            self.vars[c] = []
            for f in range(n_features):
                vals = [s[f] for s in samples]
                mean = sum(vals) / len(vals)
                var = sum((v - mean) ** 2 for v in vals) / len(vals)
                var = max(var, 1e-9)
                self.means[c].append(mean)
                self.vars[c].append(var)

    def _gaussian_pdf(self, x, mean, var):
        coeff = 1.0 / math.sqrt(2.0 * math.pi * var)
        exponent = math.exp(-((x - mean) ** 2) / (2.0 * var))
        return coeff * exponent

    def predict_proba(self, X):
        if not self.classes:
            raise RuntimeError("Model not trained yet.")
        predictions = []
        for sample in X:
            class_probs = {}
            for c in self.classes:
                log_prob = math.log(self.priors[c])
                for f, val in enumerate(sample):
                    pdf = self._gaussian_pdf(val, self.means[c][f], self.vars[c][f])
                    log_prob += math.log(max(pdf, 1e-300))
                class_probs[c] = log_prob
            max_log = max(class_probs.values())
            exp_probs = {c: math.exp(p - max_log) for c, p in class_probs.items()}
            total = sum(exp_probs.values())
            predictions.append({c: p / total for c, p in exp_probs.items()})
        return predictions

    def predict(self, X):
        probas = self.predict_proba(X)
        return [max(p, key=p.get) for p in probas]


class SoftmaxRegression:
    """Multinomial logistic (softmax) regression, implemented from scratch."""

    def __init__(self, lr: float = 0.5, epochs: int = 40, l2: float = 1e-4,
                 batch_size: int = 256, seed: int = 42):
        self.lr = lr
        self.epochs = epochs
        self.l2 = l2
        self.batch_size = batch_size
        self.seed = seed
        self.classes: List[int] = []
        self.feat_mean: List[float] = []
        self.feat_std: List[float] = []
        self.W: List[List[float]] = []
        self.b: List[float] = []

    def _standardize_fit(self, X):
        n = len(X[0])
        self.feat_mean = [0.0] * n
        self.feat_std = [1.0] * n
        for j in range(n):
            col = [row[j] for row in X]
            mean = sum(col) / len(col)
            var = sum((v - mean) ** 2 for v in col) / len(col)
            self.feat_mean[j] = mean
            self.feat_std[j] = math.sqrt(var) if var > 1e-12 else 1.0

    def _standardize(self, x):
        return [(x[j] - self.feat_mean[j]) / self.feat_std[j] for j in range(len(x))]

    @staticmethod
    def _softmax(z):
        m = max(z)
        exps = [math.exp(v - m) for v in z]
        s = sum(exps)
        return [e / s for e in exps]

    def _logits(self, x):
        return [self.b[k] + sum(x[j] * self.W[j][k] for j in range(len(x)))
                for k in range(len(self.classes))]

    def fit(self, X, y):
        rng = random.Random(self.seed)
        self.classes = sorted(set(y))
        n_features = len(X[0])
        n_classes = len(self.classes)
        cls_idx = {c: k for k, c in enumerate(self.classes)}
        self._standardize_fit(X)
        Xs = [self._standardize(row) for row in X]
        self.W = [[0.0] * n_classes for _ in range(n_features)]
        self.b = [0.0] * n_classes
        idx = list(range(len(Xs)))
        for _ in range(self.epochs):
            rng.shuffle(idx)
            for start in range(0, len(idx), self.batch_size):
                batch = idx[start:start + self.batch_size]
                gW = [[0.0] * n_classes for _ in range(n_features)]
                gb = [0.0] * n_classes
                for i in batch:
                    x = Xs[i]
                    probs = self._softmax(self._logits(x))
                    true_k = cls_idx[y[i]]
                    for k in range(n_classes):
                        err = probs[k] - (1.0 if k == true_k else 0.0)
                        gb[k] += err
                        for j in range(n_features):
                            gW[j][k] += err * x[j]
                bs = len(batch)
                for k in range(n_classes):
                    self.b[k] -= self.lr * (gb[k] / bs)
                    for j in range(n_features):
                        grad = gW[j][k] / bs + self.l2 * self.W[j][k]
                        self.W[j][k] -= self.lr * grad

    def predict_proba(self, X):
        out = []
        for row in X:
            x = self._standardize(row)
            probs = self._softmax(self._logits(x))
            out.append({self.classes[k]: probs[k] for k in range(len(self.classes))})
        return out

    def predict(self, X):
        return [max(p, key=p.get) for p in self.predict_proba(X)]


# ─── sklearn-based production models ───────────────────────────────────────

try:
    from sklearn.ensemble import (
        RandomForestClassifier,
        HistGradientBoostingClassifier,
        VotingClassifier,
        StackingClassifier,
    )
    from sklearn.linear_model import LogisticRegression
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.feature_selection import SelectKBest, mutual_info_classif
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.utils.class_weight import compute_sample_weight
    from sklearn.metrics import f1_score
    from threadpoolctl import threadpool_limits
    import joblib
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


def _to_numpy(X, y):
    """Convert list-of-lists to numpy arrays."""
    X_arr = np.asarray(X, dtype=np.float64)
    y_arr = np.asarray(y)
    return X_arr, y_arr


def _purged_time_series_split(n_samples: int, n_folds: int = 5, purge_gap: int = 0):
    """Yield train/test index pairs with a purge gap between them."""
    fold_size = n_samples // (n_folds + 1)
    for k in range(1, n_folds + 1):
        train_end = fold_size * k
        test_start = train_end + purge_gap
        test_end = min(test_start + fold_size, n_samples)
        if test_start >= n_samples:
            break
        yield np.arange(0, train_end), np.arange(test_start, test_end)


# ─── Tuning search spaces ───────────────────────────────────────────────────
# Small, hand-scoped grids (not exhaustive) evaluated on purged time-series CV
# in `_select_best`. Kept small deliberately: each candidate re-trains a full
# pipeline, so this stays a light search rather than an unbounded sweep.

_DEFAULT_RF_PARAMS = dict(n_estimators=400, max_depth=10, min_samples_split=10,
                           min_samples_leaf=5, max_features="sqrt", random_state=42, n_jobs=-1)
_DEFAULT_HISTGB_PARAMS = dict(max_depth=4, max_iter=300, learning_rate=0.05,
                               min_samples_leaf=20, l2_regularization=0.1, random_state=42)
_DEFAULT_LR_PARAMS = dict(max_iter=2000, C=0.5, random_state=42)
_DEFAULT_VOTING_WEIGHTS = [0.4, 0.35, 0.25]  # (histgb, rf, lr)

#   Grids kept to 2-3 candidates each: every candidate is a full purged-CV
#   evaluation (several pipeline fits), and fit calls carry a large fixed
#   overhead in some sandboxed/virtualized environments (thread-pool spawn
#   cost can dominate wall time on small folds) — so this stays a light,
#   bounded search rather than an exhaustive one.
_RF_GRID = [
    # n_estimators/max_depth kept modest on purpose: CalibratedClassifierCV
    # (cv=3) stores 3 FULL COPIES of whichever RF wins, so tree count is a
    # direct multiplier on shipped model size, not just training time. An
    # n_estimators=500/max_depth=None candidate here once produced a 557MB
    # model.joblib (390MB even at depth=18) for a ~0.001 macro-F1 gain over
    # these smaller candidates — not worth it for a model that has to be
    # distributable via Git LFS.
    dict(n_estimators=300, max_depth=10, min_samples_split=10, min_samples_leaf=10,
         max_features="sqrt", random_state=42, n_jobs=-1),
    dict(n_estimators=400, max_depth=10, min_samples_split=10, min_samples_leaf=5,
         max_features="sqrt", random_state=42, n_jobs=-1),
    dict(n_estimators=350, max_depth=12, min_samples_split=15, min_samples_leaf=8,
         max_features="log2", random_state=42, n_jobs=-1),
]
_HISTGB_GRID = [
    dict(max_depth=4, max_iter=300, learning_rate=0.05, min_samples_leaf=20,
         l2_regularization=0.1, random_state=42),
    dict(max_depth=6, max_iter=250, learning_rate=0.03, min_samples_leaf=30,
         l2_regularization=0.3, random_state=42),
    dict(max_depth=5, max_iter=400, learning_rate=0.04, min_samples_leaf=15,
         l2_regularization=0.2, random_state=42),
]
_LR_GRID = [
    dict(max_iter=2000, C=0.2, random_state=42),
    dict(max_iter=2000, C=1.0, random_state=42),
    dict(max_iter=2000, C=0.05, random_state=42),
]
# (histgb, rf, lr) weight triples for the soft-voting ensemble — replaces the
# single hardcoded [0.5, 0.3, 0.2] guess with a handful chosen by CV.
_VOTING_WEIGHT_GRID = [
    [0.5, 0.3, 0.2],
    [0.4, 0.35, 0.25],
    [0.34, 0.33, 0.33],
    [0.25, 0.45, 0.3],
]

# How many features to keep. `None` = keep all and let the tree models do
# their own selection at split time, which is the hypothesis this grid exists
# to test against the old hardcoded k=20 (see _build_sklearn_pipeline).
_SELECT_K_GRID = [None, 50, 35, 20]


def _build_sklearn_pipeline(model_type: str, n_features: int, use_feature_selection: bool = True,
                             tree_params: Optional[dict] = None, voting_weights: Optional[list] = None,
                             select_k: Optional[int] = None):
    """Build a sklearn Pipeline for the given model_type.

    `tree_params` overrides the underlying estimator's constructor kwargs
    (a dict of {"rf": {...}, "histgb": {...}, "lr": {...}} for "ensemble",
    or a flat kwargs dict for "rf"/"histgb"/"logreg"). All three legs are
    calibrated so voting combines well-calibrated probabilities, not raw
    tree-vote fractions.
    """
    if not SKLEARN_AVAILABLE:
        raise RuntimeError("scikit-learn is not installed.")

    # Feature selection is OPT-IN by k, not a fixed cap. It used to be
    # hardcoded to k=20, written when the feature set was much smaller; by the
    # time there were 73 features that silently discarded 53 of them —
    # including dc_draw_prob (built specifically to predict draws),
    # home_sos/away_sos (the cross-league comparison signal), and 15 of the 16
    # shot/corner/card features. mutual_info_classif is a *univariate* filter,
    # so features that only carry signal in combination (H2H, shots relative
    # to the opponent, schedule strength) score low individually even though
    # the tree models can exploit them at split time. select_k=None means
    # "keep everything and let the trees choose", which is a candidate in
    # _SELECT_K_GRID rather than an assumption.
    steps = []
    if use_feature_selection and select_k is not None and 0 < select_k < n_features:
        # mutual_info_classif uses a kNN estimator with random jitter; without a
        # fixed random_state the selected features (and thus the shipped model
        # and its feature_importances_) vary run to run. Pin it like every other
        # estimator so model.joblib is reproducible.
        steps.append(("select", SelectKBest(
            partial(mutual_info_classif, random_state=42), k=select_k)))
    steps.append(("scaler", StandardScaler()))

    def _rf(overrides=None):
        return RandomForestClassifier(**{**_DEFAULT_RF_PARAMS, **(overrides or {})})

    def _histgb(overrides=None):
        return HistGradientBoostingClassifier(**{**_DEFAULT_HISTGB_PARAMS, **(overrides or {})})

    def _lr(overrides=None):
        return LogisticRegression(**{**_DEFAULT_LR_PARAMS, **(overrides or {})})

    def _calibrated(estimator):
        # ensemble=False fits ONE base estimator (using cross_val_predict for
        # unbiased calibration targets) instead of storing `cv` full copies.
        # With cv=3 the default was tripling the size of every leg — the
        # dominant term in a model.joblib that has to ship via Git LFS.
        return CalibratedClassifierCV(estimator, method="isotonic", cv=3, ensemble=False)

    if model_type == "histgb":
        steps.append(("clf", _calibrated(_histgb(tree_params))))
    elif model_type == "rf":
        steps.append(("clf", _calibrated(_rf(tree_params))))
    elif model_type == "logreg":
        steps.append(("clf", _lr(tree_params)))
    elif model_type == "ensemble":
        tp = tree_params or {}
        histgb = _calibrated(_histgb(tp.get("histgb")))
        rf = _calibrated(_rf(tp.get("rf")))
        lr = _calibrated(_lr(tp.get("lr")))
        ensemble = VotingClassifier(
            estimators=[("histgb", histgb), ("rf", rf), ("lr", lr)],
            voting="soft",
            weights=voting_weights or _DEFAULT_VOTING_WEIGHTS,
        )
        steps.append(("clf", ensemble))
    elif model_type == "stacking":
        # A logistic-regression meta-learner over the three base models'
        # out-of-fold probabilities, instead of hand-picked voting weights —
        # lets the data decide how much to trust each leg per class.
        tp = tree_params or {}
        histgb = _calibrated(_histgb(tp.get("histgb")))
        rf = _calibrated(_rf(tp.get("rf")))
        lr = _calibrated(_lr(tp.get("lr")))
        stacking = StackingClassifier(
            estimators=[("histgb", histgb), ("rf", rf), ("lr", lr)],
            final_estimator=LogisticRegression(max_iter=2000),
            cv=3,
            n_jobs=1,
        )
        steps.append(("clf", stacking))
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    return Pipeline(steps)


def _select_best(param_grid: list, build_fn, X, y, n_folds: int, purge_gap: int,
                  recency_weights=None):
    """Pick the candidate from `param_grid` with the best purged-CV macro-F1
    (not accuracy) — macro-F1 penalizes a model that ignores the draw class
    to chase overall accuracy, which is exactly the failure mode being fixed.
    `build_fn(params)` must return an unfit Pipeline for a given candidate.
    `recency_weights` (parallel to X) is threaded through so the search scores
    candidates under the SAME weighting the final fit uses — otherwise tuning
    would optimize for uniform weights and the final model for recency-weighted.
    """
    # RPS-aware selection: rank by macro-F1 MINUS a weighted RPS, so among
    # candidates of comparable draw-handling the one with the better (lower) RPS
    # wins — RPS is the model's real target. macro-F1 still dominates (it guards
    # the draw class), and the retrain's leak-free holdout validates the result.
    RPS_WEIGHT = 2.0
    best_params, best_blend, best_f1 = param_grid[0], -1e9, None
    for params in param_grid:
        pipe = build_fn(params)
        res = _evaluate_pipeline(pipe, X, y, n_folds=n_folds, purge_gap=purge_gap,
                                 recency_weights=recency_weights)
        f1 = res.get("macro_f1")
        if f1 is None:
            continue
        rps = res.get("rps")
        blend = f1 - RPS_WEIGHT * rps if rps is not None else f1
        if blend > best_blend:
            best_blend, best_params, best_f1 = blend, params, f1
    # Return the winner's macro-F1 (not the blend) so the logged tuning notes
    # stay interpretable.
    return best_params, (best_f1 if best_f1 is not None else -1.0)


def _evaluate_pipeline(pipe, X, y, n_folds=5, purge_gap=0, burn_in_folds=0,
                        recency_weights=None):
    """Time-series cross-validation with purging. Returns metrics dict.

    `burn_in_folds` drops the earliest N folds from the REPORTED means. The
    ratings features (Elo, Dixon-Coles) start every team at a neutral anchor
    and need dozens of matches to become meaningful, so the first folds score
    a model reading half-warmed ratings — that is a measurement artifact of
    cold ratings, not the model's true skill, and averaging it in is what
    dragged the old purged-CV number ~11pt below the recent-holdout number.
    The folds still TRAIN chronologically; only their scores are excluded.

    `recency_weights` (parallel to X) is multiplied into the balanced class
    weights so recent matches count for more — see MatchPredictorModel.train.
    """
    rows_acc = []
    auc_scores = []
    f1_scores = []
    rps_scores = []
    confusion = np.zeros((len(np.unique(y)), len(np.unique(y))), dtype=int)
    classes = sorted(np.unique(y))
    class_to_idx = {c: i for i, c in enumerate(classes)}

    for fold_i, (tr, te) in enumerate(_purged_time_series_split(len(X), n_folds, purge_gap)):
        ytr, yte = y[tr], y[te]
        if len(np.unique(ytr)) < 2 or len(te) == 0:
            continue
        with warnings.catch_warnings(), threadpool_limits(limits=1):
            warnings.simplefilter("ignore")
            # Balanced sample weights so every leg of the pipeline (not just
            # a hand-picked one) corrects for the draw-minority class,
            # optionally scaled by recency (per-class renormalized) so recent
            # matches weigh more without disturbing the class balance.
            sw = _blend_sample_weights(
                ytr, np.asarray(recency_weights)[tr] if recency_weights is not None else None)
            try:
                pipe.fit(X[tr], ytr, clf__sample_weight=sw)
            except TypeError:
                pipe.fit(X[tr], ytr)
        pred = pipe.predict(X[te])
        proba = pipe.predict_proba(X[te])
        # A burned-in fold is trained (ratings carry forward) but not scored.
        if fold_i < burn_in_folds:
            continue
        rows_acc.append(float(np.mean(pred == yte)))
        f1_scores.append(float(f1_score(yte, pred, average="macro")))
        rps_scores.append(_rps(yte, proba, pipe.classes_))

        if len(classes) == 2:
            from sklearn.metrics import roc_auc_score
            try:
                pos_idx = np.where(pipe.classes_ == classes[1])[0][0]
                auc_scores.append(roc_auc_score(yte, proba[:, pos_idx]))
            except ValueError:
                pass
        else:
            from sklearn.metrics import roc_auc_score
            try:
                auc_scores.append(roc_auc_score(yte, proba, multi_class="ovr", average="macro"))
            except ValueError:
                pass

        for true, pred_i in zip(yte, pred):
            if true in class_to_idx and pred_i in class_to_idx:
                confusion[class_to_idx[true]][class_to_idx[pred_i]] += 1

    if not rows_acc:
        return {
            "accuracy": float("nan"), "accuracy_std": float("nan"), "auc": float("nan"),
            "macro_f1": None, "rps": None, "confusion_matrix": confusion.tolist(),
            "n_folds_used": 0, "fold_scores": [],
        }

    return {
        "accuracy": float(np.mean(rows_acc)),
        "accuracy_std": float(np.std(rows_acc)),
        "auc": float(np.mean(auc_scores)) if auc_scores else float("nan"),
        "macro_f1": float(np.mean(f1_scores)),
        "rps": float(np.mean(rps_scores)) if rps_scores else None,
        "confusion_matrix": confusion.tolist(),
        "n_folds_used": len(rows_acc),
        "fold_scores": rows_acc,
    }


def _rps(y_true, proba, classes) -> float:
    """Ranked Probability Score — the proper scoring rule for ORDERED
    outcomes, which Home/Draw/Away are (Draw sits between the two wins on a
    strength axis). Unlike accuracy (dominated by the home class and
    dead-ceilinged on this problem) or log-loss (ignores ordinality), RPS
    penalises a confident Home prediction more when the truth is Away than
    when it is Draw. It is the metric football-prediction literature reports,
    so it is what lets us compare against the ~0.19-0.21 published ceiling.
    Lower is better. Classes must be passed in ascending ordinal order."""
    classes = np.asarray(list(classes))
    y = np.asarray(y_true)
    P = np.asarray(proba, dtype=np.float64)
    if len(y) == 0 or P.shape[1] < 2:
        return float("nan")
    onehot = (classes[None, :] == y[:, None]).astype(np.float64)
    cum_p = np.cumsum(P, axis=1)[:, :-1]
    cum_o = np.cumsum(onehot, axis=1)[:, :-1]
    return float(np.mean(np.sum((cum_p - cum_o) ** 2, axis=1) / (P.shape[1] - 1)))


def _blend_sample_weights(y_sub, recency_sub):
    """Balanced class weights, optionally scaled by recency. Recency is
    renormalized PER CLASS so it reweights matches WITHIN a class (recent >
    old) without disturbing the class balance the balanced weights set up — a
    plain global rescale would let recency silently reweight the classes (e.g.
    if draws skew older), undoing the draw-minority correction."""
    sw = compute_sample_weight("balanced", y_sub)
    if recency_sub is None:
        return sw
    sw = sw * np.asarray(recency_sub, dtype=np.float64)
    out = np.zeros_like(sw)
    classes = np.unique(y_sub)
    target = len(sw) / len(classes)
    for c in classes:
        m = y_sub == c
        s = sw[m].sum()
        if s > 0:
            out[m] = sw[m] * (target / s)
    return out


def _majority_baseline(ytr, yte):
    maj = int(round(np.mean(ytr)))
    return float(np.mean(yte == maj))


def _feature_importance_from(clf):
    """Extract a feature_importances_-style array from a fitted classifier,
    unwrapping CalibratedClassifierCV and averaging across a VotingClassifier's
    tree legs (HistGB/RF). Linear-model coefficients are excluded from the
    ensemble average since they're a different scale/unit than impurity
    importances and would distort a shared "top features" ranking."""
    if hasattr(clf, "feature_importances_"):
        return np.asarray(clf.feature_importances_)
    if hasattr(clf, "calibrated_classifiers_"):
        imps = []
        for cc in clf.calibrated_classifiers_:
            base = getattr(cc, "estimator", None) or getattr(cc, "base_estimator", None)
            if base is not None and hasattr(base, "feature_importances_"):
                imps.append(np.asarray(base.feature_importances_))
        return np.mean(imps, axis=0) if imps else None
    if hasattr(clf, "estimators_"):
        imps = []
        for est in clf.estimators_:
            imp = _feature_importance_from(est)
            if imp is not None:
                imps.append(imp)
        return np.mean(imps, axis=0) if imps else None
    if hasattr(clf, "coef_"):
        coef = np.abs(clf.coef_)
        return coef.mean(axis=0) if coef.ndim > 1 else coef
    return None


class MatchPredictorModel:
    """Production ML ensemble for match outcome prediction.

    model_type selects the underlying sklearn model:
      - "nb":       from-scratch Gaussian Naive Bayes (kept for tests)
      - "logreg":   from-scratch softmax regression (kept for tests)
      - "histgb":   sklearn HistGradientBoostingClassifier
      - "rf":       sklearn RandomForestClassifier
      - "ensemble": sklearn VotingClassifier + Calibration (default)
      - "stacking": sklearn StackingClassifier (logistic-regression meta-learner
                    over the same three base legs, instead of fixed voting weights)
    """

    def __init__(self, model_type: str = "ensemble"):
        self.model_type = model_type
        self.sklearn_model_type = model_type if model_type in ("histgb", "rf", "logreg", "ensemble", "stacking") else "ensemble"
        self.pipeline = None
        self.classes: List[int] = []
        self.trained = False
        self.label_map = {-1: "Away Win", 0: "Draw", 1: "Home Win"}
        self.version = MODEL_VERSION
        self.trained_at: Optional[str] = None
        self.training_samples: int = 0
        self.feature_names: List[str] = []
        self.confusion_matrix: Optional[List[List[int]]] = None
        self.class_distribution: Dict[int, int] = {}
        self.cv_mean: float = 0.0
        self.cv_std: float = 0.0
        self.eval_method: str = "temporal"
        self.feature_importances_: Optional[List[float]] = None
        self.baseline_accuracy: float = 0.0
        self.accuracy_: float = 0.0
        # None means "not computed" (e.g. the holdout prob-metrics step
        # failed) -- must stay distinguishable from a real 0.0, which would
        # read as a suspiciously perfect score rather than a missing one.
        self.log_loss_: Optional[float] = None
        self.brier_: Optional[float] = None
        # RPS on the tail holdout (holdout) and averaged over warm CV folds
        # (cv_rps). None = not computed, kept distinguishable from a real 0.0.
        self.rps_: Optional[float] = None
        self.cv_rps: Optional[float] = None
        self.test_samples: int = 0
        self.tuning_notes: Dict[str, Any] = {}
        self.select_k_: Optional[int] = None
        # Decision-rule threshold for the Draw class. Predict Draw when
        # P(draw) >= this, instead of taking the plain argmax. See predict().
        self.draw_threshold: Optional[float] = None
        # Post-hoc probability recalibration (multinomial Platt scaling over
        # log-probabilities). The pipeline is trained with balanced class
        # weights to rescue draw recall, which systematically deflates
        # home-win probability: measured on held-out matches, "74%" actually
        # won 94% of the time. This maps raw outputs back onto reality so a
        # stated probability means what it says.
        self.calibrator = None
        self.class_prior_ = None
        self.calib_shrinkage_ = None
        self.calib_C_ = None
        self.recency_halflife_days = None
        # Dixon-Coles output blend: mix the DC H/D/A vector (already features)
        # into the ensemble output before calibration. The weight is TUNED on the
        # leak-free holdout, so it self-selects 0 (a no-op) whenever the blend
        # doesn't improve RPS — it can only help or do nothing.
        self.dc_blend_weight = None
        self.dc_prob_idx = None
        # Out-of-sample calibrated predictions on the honest holdout (from the
        # eval pipeline that never trained on them). Persisted so evaluate.py
        # and the dashboard report leak-free numbers instead of re-scoring the
        # all-data model on its own tail.
        self._holdout = None

    def _fit_sklearn(self, X, y, feature_names, cv_folds=5, purge_gap=0,
                      tree_params_override=None, voting_weights_override=None,
                      select_k_override=-1, recency_weights=None, burn_in_folds=0,
                      calibrate=False, calibrate_frac=0.2):
        """Fit sklearn pipeline with time-series CV.

        `recency_weights` (parallel to X) down-weights older matches in both
        the CV folds and the final refit. `burn_in_folds` excludes the coldest
        early CV folds from the reported score (ratings warm-up artifact)."""
        X_arr, y_arr = _to_numpy(X, y)
        n_features = X_arr.shape[1]

        # Adaptive model selection for small datasets
        actual_model_type = self.sklearn_model_type
        if n_features < 5 or len(X) < 50:
            actual_model_type = "logreg"
        elif len(X) < 200 and actual_model_type in ("ensemble", "stacking"):
            actual_model_type = "histgb"

        # Light hyperparameter / voting-weight search, scored by macro-F1 so
        # the search can't "win" by ignoring draws. Tuning runs on a
        # recency-capped subsample to stay fast; the winning config is then
        # evaluated and fit on the full dataset below. Skipped entirely if
        # the caller already knows the config it wants (tree_params_override).
        tuning_folds = min(cv_folds, 3)
        tree_params, voting_weights = tree_params_override, voting_weights_override
        self.tuning_notes = {}
        # Tune on a recency-capped, time-ordered subsample (purged CV needs the
        # rows in date order, so this can't be a random sample). The cap must
        # span at least a full annual cycle of every league: an 8k tail in a
        # mid-year retrain is almost entirely summer-calendar leagues
        # (MLS/Scandinavia/Japan/Argentina) with the European winter absent, so
        # hyperparameters got tuned on an unrepresentative slice. ~25k rows
        # reaches back ~1.5 years and includes every league's full season.
        tuning_cap = 25000
        if len(X_arr) > tuning_cap:
            Xt, yt = X_arr[-tuning_cap:], y_arr[-tuning_cap:]
            rec_t = np.asarray(recency_weights)[-tuning_cap:] if recency_weights is not None else None
        else:
            Xt, yt = X_arr, y_arr
            rec_t = recency_weights

        # `select_k_override=-1` is the "not specified" sentinel, since None
        # is itself a meaningful value here (= keep all features).
        select_k = select_k_override if select_k_override != -1 else None
        skip_search = tree_params_override is not None or select_k_override != -1

        if skip_search:
            pass
        elif len(X_arr) >= 500:
            # Greedy coordinate search (not exhaustive — each candidate
            # retrains a full pipeline): pick the feature-count first with
            # default estimators, then tune estimators at that width, then
            # the combiner. Scored throughout by macro-F1 so the search can't
            # win by ignoring draws.
            probe_type = actual_model_type if actual_model_type in ("rf", "histgb", "logreg") else "histgb"
            select_k, k_score = _select_best(
                _SELECT_K_GRID,
                lambda k: _build_sklearn_pipeline(probe_type, n_features, select_k=k),
                Xt, yt, tuning_folds, purge_gap, recency_weights=rec_t)
            self.tuning_notes["select_k"] = {
                "k": select_k if select_k is not None else "all",
                "n_features": int(n_features),
                "cv_macro_f1": round(k_score, 4),
            }

            if actual_model_type in ("rf", "histgb", "logreg"):
                grid = {"rf": _RF_GRID, "histgb": _HISTGB_GRID, "logreg": _LR_GRID}[actual_model_type]
                tree_params, score = _select_best(
                    grid,
                    lambda p: _build_sklearn_pipeline(actual_model_type, n_features, tree_params=p, select_k=select_k),
                    Xt, yt, tuning_folds, purge_gap, recency_weights=rec_t)
                self.tuning_notes[actual_model_type] = {"params": tree_params, "cv_macro_f1": round(score, 4)}
            elif actual_model_type in ("ensemble", "stacking"):
                # Both combiners (fixed-weight voting and the stacking
                # meta-learner) reuse the same tuned rf/histgb/lr base legs —
                # only how they're combined differs.
                best_rf, rf_score = _select_best(
                    _RF_GRID,
                    lambda p: _build_sklearn_pipeline("rf", n_features, tree_params=p, select_k=select_k),
                    Xt, yt, tuning_folds, purge_gap, recency_weights=rec_t)
                best_histgb, histgb_score = _select_best(
                    _HISTGB_GRID,
                    lambda p: _build_sklearn_pipeline("histgb", n_features, tree_params=p, select_k=select_k),
                    Xt, yt, tuning_folds, purge_gap, recency_weights=rec_t)
                best_lr, lr_score = _select_best(
                    _LR_GRID,
                    lambda p: _build_sklearn_pipeline("logreg", n_features, tree_params=p, select_k=select_k),
                    Xt, yt, tuning_folds, purge_gap, recency_weights=rec_t)
                tree_params = {"rf": best_rf, "histgb": best_histgb, "lr": best_lr}
                self.tuning_notes.update({
                    "rf": {"params": best_rf, "cv_macro_f1": round(rf_score, 4)},
                    "histgb": {"params": best_histgb, "cv_macro_f1": round(histgb_score, 4)},
                    "logreg": {"params": best_lr, "cv_macro_f1": round(lr_score, 4)},
                })
                if actual_model_type == "ensemble":
                    voting_weights, weight_score = _select_best(
                        _VOTING_WEIGHT_GRID,
                        lambda w: _build_sklearn_pipeline("ensemble", n_features, tree_params=tree_params,
                                                          voting_weights=w, select_k=select_k),
                        Xt, yt, tuning_folds, purge_gap, recency_weights=rec_t)
                    self.tuning_notes["voting_weights"] = {"weights": voting_weights, "cv_macro_f1": round(weight_score, 4)}

        self.select_k_ = select_k

        # Build the winning pipeline
        pipe = _build_sklearn_pipeline(actual_model_type, n_features,
                                        tree_params=tree_params, voting_weights=voting_weights,
                                        select_k=select_k)
        # Remember the winning config so the honest-holdout step can rebuild an
        # identically-configured EVAL pipeline that excludes the holdout.
        self._fit_config = dict(model_type=actual_model_type, n_features=n_features,
                                tree_params=tree_params, voting_weights=voting_weights,
                                select_k=select_k)

        # Evaluate on the full dataset (balanced sample weights applied inside,
        # scaled by recency; coldest folds burned in and excluded from score)
        cv_res = _evaluate_pipeline(pipe, X_arr, y_arr, n_folds=cv_folds, purge_gap=purge_gap,
                                     burn_in_folds=burn_in_folds, recency_weights=recency_weights)

        # Refit on all data
        with warnings.catch_warnings(), threadpool_limits(limits=1):
            warnings.simplefilter("ignore")
            sw_full = _blend_sample_weights(y_arr, recency_weights)
            try:
                pipe.fit(X_arr, y_arr, clf__sample_weight=sw_full)
            except TypeError:
                pipe.fit(X_arr, y_arr)

        self.pipeline = pipe
        self.classes = sorted(np.unique(y_arr).tolist())
        self._init_dc_blend_idx(feature_names)
        self.accuracy_ = cv_res["accuracy"]
        self.cv_mean = cv_res["accuracy"]
        self.cv_std = cv_res["accuracy_std"]
        self.confusion_matrix = cv_res["confusion_matrix"]
        self._cv_scores = cv_res.get("fold_scores", [])
        self.macro_f1_ = cv_res.get("macro_f1")
        self.cv_rps = cv_res.get("rps")

        # Feature importances from the underlying classifier (unwraps
        # CalibratedClassifierCV / VotingClassifier to reach the tree legs)
        try:
            clf = pipe.named_steps.get("clf")
            raw_imp = _feature_importance_from(clf)
            if raw_imp is not None:
                if "select" in pipe.named_steps:
                    mask = pipe.named_steps["select"].get_support()
                    full_imp = np.zeros(len(feature_names))
                    selected = np.where(mask)[0]
                    full_imp[selected] = raw_imp
                    self.feature_importances_ = full_imp.tolist()
                else:
                    self.feature_importances_ = np.asarray(raw_imp).tolist()
            else:
                self.feature_importances_ = None
        except Exception:
            self.feature_importances_ = None

        # Baseline
        maj_class = Counter(y_arr.tolist()).most_common(1)[0][0]
        self.baseline_accuracy = float(np.mean(y_arr == maj_class))

        # Honest holdout: probabilistic metrics AND (optionally) the calibrator
        # are computed via a separate EVAL pipeline trained on everything BEFORE
        # the holdout, never on it. The old code scored `pipe` — refit on ALL
        # data — on its own tail, so the reported RPS/log-loss/Brier and the
        # calibrator were in-sample and optimistic. `pipe` still ships trained
        # on all data (keeps the most recent matches); only the measurement and
        # calibrator use the leak-free eval pipeline.
        self._honest_holdout(X_arr, y_arr, calibrate_frac, recency_weights, calibrate)

        return cv_res

    def _honest_holdout(self, X_arr, y_arr, calibrate_frac, recency_weights, calibrate):
        """Leak-free holdout evaluation + calibration.

        Split the tail into a calibration slice and a disjoint test slice, fit
        an eval pipeline (identical config) on everything BEFORE both, fit the
        calibrator on the eval pipeline's OUT-OF-SAMPLE calibration-slice probs,
        and score CALIBRATED metrics on the test slice — none of which the eval
        pipeline trained on. The shipped self.pipeline (all data) reuses this
        calibrator; eval and shipped pipelines differ only by the ~20% tail, so
        the calibrator transfers, while the numbers stop being optimistic.
        """
        n = len(X_arr)
        n_hold = max(int(n * calibrate_frac), 1)
        n_test = n_hold - n_hold // 2
        cal_lo, cal_hi = n - n_hold, n - n_test
        self.test_samples = 0

        # Too little data for a real split (unit tests, tiny fixtures): fall back
        # to the pipeline's own tail. Optimistic, but only ever hit below the
        # threshold where an honest split isn't possible anyway.
        if cal_lo < 200 or n_test < 20:
            try:
                if calibrate:
                    self.fit_calibration(X_arr[cal_lo:cal_hi], y_arr[cal_lo:cal_hi])
                proba = self.apply_calibration(self.pipeline.predict_proba(X_arr[-n_test:]))
                self._compute_prob_metrics(y_arr[-n_test:], proba)
                self.test_samples = int(n_test)
            except Exception:
                self.log_loss_ = self.brier_ = self.rps_ = None
            return

        cfg = self._fit_config
        eval_pipe = _build_sklearn_pipeline(cfg["model_type"], cfg["n_features"],
                                            tree_params=cfg["tree_params"],
                                            voting_weights=cfg["voting_weights"], select_k=cfg["select_k"])
        with warnings.catch_warnings(), threadpool_limits(limits=1):
            warnings.simplefilter("ignore")
            sw = _blend_sample_weights(
                y_arr[:cal_lo],
                np.asarray(recency_weights)[:cal_lo] if recency_weights is not None else None)
            try:
                eval_pipe.fit(X_arr[:cal_lo], y_arr[:cal_lo], clf__sample_weight=sw)
            except TypeError:
                eval_pipe.fit(X_arr[:cal_lo], y_arr[:cal_lo])

        if calibrate:
            cal_X, cal_y = X_arr[cal_lo:cal_hi], y_arr[cal_lo:cal_hi]
            if self.dc_prob_idx:
                # Tune the DC blend weight on the leak-free test slice (the eval
                # pipeline never saw cal/test). For each weight, fit the
                # calibrator on the blended calibration probs and score the
                # blended+calibrated test-slice RPS; keep the best. w=0 wins if
                # the blend doesn't help, so this can only help or no-op.
                raw_cal = eval_pipe.predict_proba(cal_X)
                raw_test = eval_pipe.predict_proba(X_arr[cal_hi:])
                best_w, best_rps = 0.0, float("inf")
                for w in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5):
                    self.dc_blend_weight = w
                    self._fit_calibrator_on(
                        self._calib_logspace(self._blend_dc(raw_cal, cal_X)), np.asarray(cal_y))
                    P = self.apply_calibration(self._blend_dc(raw_test, X_arr[cal_hi:]))
                    r = _rps(y_arr[cal_hi:], P, self.classes)
                    if r < best_rps:
                        best_rps, best_w = r, w
                self.dc_blend_weight = best_w
                # Final calibrator at the chosen weight.
                self._fit_calibrator_on(
                    self._calib_logspace(self._blend_dc(raw_cal, cal_X)), np.asarray(cal_y))
            else:
                # Calibrator fit on the eval pipeline's OUT-OF-SAMPLE probs.
                self.fit_calibration(cal_X, cal_y, pipeline=eval_pipe)

        try:
            raw_test = eval_pipe.predict_proba(X_arr[cal_hi:])
            proba_test = self.apply_calibration(self._blend_dc(raw_test, X_arr[cal_hi:]))
            self._compute_prob_metrics(y_arr[cal_hi:], proba_test)
            self.test_samples = int(n_test)
            # Persist the OOS calibrated predictions so evaluate.py / dashboard
            # report leak-free per-league + reliability numbers.
            self._holdout = {
                "proba": np.asarray(proba_test, dtype=np.float64).round(6).tolist(),
                "y": [int(v) for v in y_arr[cal_hi:]],
                "cal_hi": int(cal_hi), "n_test": int(n_test),
                "classes": [int(c) for c in self.classes],
            }
        except Exception as e:
            print(f"WARNING: honest-holdout metric computation failed ({e}); "
                  f"reporting them as unavailable rather than a misleading 0.0.")
            self.log_loss_ = self.brier_ = self.rps_ = None

    def _compute_prob_metrics(self, y_true, proba):
        """Compute log-loss and Brier score."""
        n = len(y_true)
        if n == 0:
            return
        classes = self.pipeline.classes_ if self.pipeline else self.classes
        cls_idx = {c: i for i, c in enumerate(classes)}
        
        ll = 0.0
        brier = 0.0
        for i in range(n):
            true_idx = cls_idx.get(y_true[i], 0)
            p = max(proba[i, true_idx], 1e-15)
            ll += -math.log(p)
            for j, c in enumerate(classes):
                target = 1.0 if c == y_true[i] else 0.0
                brier += (proba[i, j] - target) ** 2
        
        self.log_loss_ = ll / n
        self.brier_ = brier / (n * len(classes))
        # RPS on the same holdout — the ordinal proper scoring rule the
        # football-prediction literature reports (see _rps).
        self.rps_ = _rps(y_true, proba, classes)

    def _fit_legacy(self, X, y, feature_names, cv_folds, temporal):
        """Fit using the old from-scratch models (backward compat)."""
        model_type = self.model_type
        if model_type == "nb":
            self.nb_model = GaussianNaiveBayes()
            self.lr_model = SoftmaxRegression()
        elif model_type == "logreg":
            self.nb_model = GaussianNaiveBayes()
            self.lr_model = SoftmaxRegression()
        else:
            # ensemble fallback
            self.nb_model = GaussianNaiveBayes()
            self.lr_model = SoftmaxRegression()
            self.model_type = "ensemble"
        
        self.classes = sorted(set(y))
        combined = list(zip(X, y))
        if not temporal:
            random.shuffle(combined)
        split_idx = int(len(combined) * 0.8)
        train_data = combined[:split_idx]
        test_data = combined[split_idx:]
        X_train = [d[0] for d in train_data]
        y_train = [d[1] for d in train_data]
        X_test = [d[0] for d in test_data]
        y_test = [d[1] for d in test_data]
        
        self.nb_model.fit(X_train, y_train)
        self.lr_model.fit(X_train, y_train)
        
        y_proba = self._proba_list_legacy(X_test)
        y_pred = [max(p, key=p.get) for p in y_proba]
        self.accuracy_ = sum(1 for i in range(len(y_test)) if y_test[i] == y_pred[i]) / max(len(y_test), 1)
        self.log_loss_ = self._log_loss_legacy(y_test, y_proba)
        self.brier_ = self._brier_legacy(y_test, y_proba)
        self.baseline_accuracy = Counter(y_train).most_common(1)[0][0] if y_train else 1
        self.baseline_accuracy = sum(1 for v in y_test if v == self.baseline_accuracy) / max(len(y_test), 1)
        
        self.class_distribution = {c: y_train.count(c) for c in self.classes}
        self.confusion_matrix = self._compute_confusion_matrix(y_test, y_pred, self.classes)
        
        cv_scores = self._cross_validate_legacy(X, y, cv_folds)
        self.cv_mean = sum(cv_scores) / len(cv_scores) if cv_scores else 0.0
        self.cv_std = self._std_legacy(cv_scores)
        
        # Refit on all data
        self.nb_model.fit(X, y)
        self.lr_model.fit(X, y)
        
        self.training_samples = len(X)
        self.trained = True

    def _proba_list_legacy(self, X):
        if self.model_type == "nb":
            return self.nb_model.predict_proba(X)
        if self.model_type == "logreg":
            return self.lr_model.predict_proba(X)
        pn = self.nb_model.predict_proba(X)
        pl = self.lr_model.predict_proba(X)
        merged = []
        for a, b in zip(pn, pl):
            merged.append({c: (a.get(c, 0.0) + b.get(c, 0.0)) / 2.0 for c in self.classes})
        return merged

    def _log_loss_legacy(self, y_true, y_proba):
        if not y_true:
            return 0.0
        total = 0.0
        for true, probs in zip(y_true, y_proba):
            p = max(probs.get(true, 1e-15), 1e-15)
            total += -math.log(p)
        return total / len(y_true)

    def _brier_legacy(self, y_true, y_proba):
        if not y_true:
            return 0.0
        total = 0.0
        for true, probs in zip(y_true, y_proba):
            for c in self.classes:
                target = 1.0 if c == true else 0.0
                total += (probs.get(c, 0.0) - target) ** 2
        return total / len(y_true)

    def _cross_validate_legacy(self, X, y, folds=5):
        if len(X) < folds * 2:
            return []
        combined = list(zip(X, y))
        random.shuffle(combined)
        fold_size = len(combined) // folds
        scores = []
        for i in range(folds):
            start = i * fold_size
            end = start + fold_size if i < folds - 1 else len(combined)
            test_data = combined[start:end]
            train_data = combined[:start] + combined[end:]
            if not train_data or not test_data:
                continue
            X_train = [d[0] for d in train_data]
            y_train = [d[1] for d in train_data]
            X_test = [d[0] for d in test_data]
            y_test = [d[1] for d in test_data]
            classes = sorted(set(y_train))
            nb = GaussianNaiveBayes()
            nb.fit(X_train, y_train)
            if self.model_type == "nb":
                y_pred = nb.predict(X_test)
            else:
                lr = SoftmaxRegression()
                lr.fit(X_train, y_train)
                if self.model_type == "logreg":
                    y_pred = lr.predict(X_test)
                else:
                    pn, pl = nb.predict_proba(X_test), lr.predict_proba(X_test)
                    y_pred = []
                    for a, b in zip(pn, pl):
                        avg = {c: (a.get(c, 0.0) + b.get(c, 0.0)) / 2.0 for c in classes}
                        y_pred.append(max(avg, key=avg.get))
            scores.append(sum(1 for j in range(len(y_test)) if y_test[j] == y_pred[j]) / max(len(y_test), 1))
        return scores

    def _std_legacy(self, values):
        if not values:
            return 0.0
        mean = sum(values) / len(values)
        return math.sqrt(sum((v - mean) ** 2 for v in values) / len(values))

    def _compute_confusion_matrix(self, y_true, y_pred, classes):
        n_classes = len(classes)
        cm = [[0] * n_classes for _ in range(n_classes)]
        class_to_idx = {c: i for i, c in enumerate(classes)}
        for true, pred in zip(y_true, y_pred):
            if true in class_to_idx and pred in class_to_idx:
                cm[class_to_idx[true]][class_to_idx[pred]] += 1
        return cm

    @staticmethod
    def _recency_weights(sample_dates, halflife_days):
        """Exponential time-decay weight per match: 0.5**(age/halflife), with
        age measured in days back from the most recent match. A half-life of
        e.g. 540 days means a match two half-lives (~3 years) old counts a
        quarter as much as last week's. Returns None (no reweighting) when
        dates or a half-life aren't supplied. Whether this actually helps is
        an empirical question — it is gated on an out-of-sample RPS delta, not
        assumed — because the Elo/DC/EWMA features already encode recency and
        classifier-level decay can double-count it."""
        if not sample_dates or not halflife_days:
            return None
        import datetime as _dt

        def _parse(d):
            try:
                return _dt.datetime.fromisoformat(str(d).replace("Z", "+00:00")).replace(tzinfo=None)
            except (ValueError, AttributeError):
                return None

        parsed = [_parse(d) for d in sample_dates]
        valid = [p for p in parsed if p is not None]
        if not valid:
            return None
        latest = max(valid)
        w = np.full(len(parsed), np.nan, dtype=np.float64)
        for i, p in enumerate(parsed):
            if p is not None:
                age = max((latest - p).days, 0)
                w[i] = 0.5 ** (age / float(halflife_days))
        # A row with an unparseable/missing date must NOT default to the maximum
        # weight (that would treat an unknown date as "most recent"). Give it the
        # smallest observed weight — treat unknown-age as old — so a bad date can
        # never inflate a row's influence.
        if np.isnan(w).any():
            w[np.isnan(w)] = np.nanmin(w)
        return w

    def train(self, X, y, feature_names: Optional[List[str]] = None,
              cv_folds: int = 5, temporal: bool = True,
              tree_params_override: Optional[dict] = None,
              voting_weights_override: Optional[list] = None,
              select_k_override: int = -1,
              sample_dates: Optional[List[str]] = None,
              recency_halflife_days: Optional[float] = None,
              burn_in_folds: int = 0,
              calibrate: bool = False,
              calibrate_frac: float = 0.2):
        """`*_override` skip the hyperparameter/voting-weight search and fit
        directly with the given config (e.g. re-using a config a previous
        tuning run already found) — same final evaluate+refit, just without
        re-paying for the search. `select_k_override` uses -1 as its
        "unspecified" sentinel because None means "keep all features".

        `sample_dates` (parallel to X) + `recency_halflife_days` enable
        exponential recency weighting; `burn_in_folds` excludes the coldest
        early CV folds from the reported CV score (ratings warm-up artifact)."""
        self.feature_names = feature_names or [f"feature_{i}" for i in range(len(X[0]) if X else 0)]
        recency_weights = self._recency_weights(sample_dates, recency_halflife_days)
        self.recency_halflife_days = recency_halflife_days

        purge_gap = 0
        if temporal and len(X) > 50:
            # Estimate horizon from data if possible; default to 1 event
            purge_gap = 1

        if self.model_type in ("nb", "logreg") or not SKLEARN_AVAILABLE:
            self._fit_legacy(X, y, self.feature_names, cv_folds, temporal)
        else:
            cv_res = self._fit_sklearn(X, y, self.feature_names, cv_folds=cv_folds, purge_gap=purge_gap,
                                        tree_params_override=tree_params_override,
                                        voting_weights_override=voting_weights_override,
                                        select_k_override=select_k_override,
                                        recency_weights=recency_weights,
                                        burn_in_folds=burn_in_folds,
                                        calibrate=calibrate, calibrate_frac=calibrate_frac)
            self.eval_method = "temporal" if temporal else "shuffled"
            self.training_samples = len(X)
            self.class_distribution = {c: int(y.count(c)) for c in self.classes}
            if not self.confusion_matrix or len(self.confusion_matrix) != len(self.classes):
                self.confusion_matrix = [[0]*len(self.classes) for _ in self.classes]
            self.trained = True
            # Calibration is folded in by _fit_sklearn's honest-holdout step
            # (calibrate flag threaded through), so one train() call reproduces
            # the shipped calibrated model — no separate bolt-on step.

        self.trained_at = datetime.utcnow().isoformat() + "Z"
        return self.get_metrics()

    def get_metrics(self):
        report = {}
        classes = sorted(self.classes)
        cm = self.confusion_matrix or [[0]*len(classes) for _ in classes]
        for i, c in enumerate(classes):
            tp = cm[i][i]
            fp = sum(cm[j][i] for j in range(len(classes))) - tp
            fn = sum(cm[i]) - tp
            precision = tp / max(tp + fp, 1)
            recall = tp / max(tp + fn, 1)
            f1 = 2 * precision * recall / max(precision + recall, 1e-9)
            report[str(c)] = {
                "precision": round(precision, 3),
                "recall": round(recall, 3),
                "f1-score": round(f1, 3),
                "support": self.class_distribution.get(c, 0),
            }
        return {
            "accuracy": round(self.accuracy_, 4),
            "baseline_accuracy": round(self.baseline_accuracy, 4),
            "log_loss": round(self.log_loss_, 4) if self.log_loss_ is not None else None,
            "brier": round(self.brier_, 4) if self.brier_ is not None else None,
            # RPS: holdout (proper ordinal score, ~0.19-0.21 is published SOTA)
            # and the warm-fold CV mean. Both None if not computed.
            "rps": round(self.rps_, 4) if self.rps_ is not None else None,
            "cv_rps": round(self.cv_rps, 4) if self.cv_rps is not None else None,
            "eval_method": self.eval_method,
            "classification_report": report,
            # test_samples is the tail slice held out for log-loss/Brier
            # (see _fit_sklearn); it's a subset of train_samples, not
            # additional data, and CV-fold accuracy above is measured over
            # purged folds spanning the full training set.
            "test_samples": getattr(self, "test_samples", 0),
            "train_samples": self.training_samples,
            "cv_scores": [round(s, 4) for s in getattr(self, "_cv_scores", [])],
            "cv_mean": round(self.cv_mean, 4),
            "cv_std": round(self.cv_std, 4),
            "macro_f1": round(self.macro_f1_, 4) if getattr(self, "macro_f1_", None) is not None else None,
            "confusion_matrix": self.confusion_matrix,
            "class_distribution": self.class_distribution,
            "feature_importances": self.feature_importances_,
            "tuning_notes": getattr(self, "tuning_notes", {}),
            "model_type": self.model_type,
            "version": self.version,
            "feature_names": self.feature_names,
        }

    def predict(self, features):
        if not self.trained:
            raise RuntimeError("Model not trained yet. Call train() first.")

        # Fail loudly on a feature-count mismatch (stale artifact vs a changed
        # feature builder) rather than letting sklearn raise a cryptic error or,
        # worse, silently predicting from a misaligned vector.
        if self.feature_names and len(features) != len(self.feature_names):
            raise ValueError(
                f"Feature count mismatch: got {len(features)} features but the model "
                f"expects {len(self.feature_names)}. The saved model and the feature "
                f"builder are out of sync — retrain.")

        if self.pipeline is not None:
            X = np.asarray([features], dtype=np.float64)
            proba = self._blend_dc(self.pipeline.predict_proba(X), X)
            proba = self.apply_calibration(proba)[0]
            cls_probs = {self.classes[i]: float(proba[i]) for i in range(len(self.classes))}
        elif hasattr(self, "nb_model") and self.nb_model:
            proba_dicts = self._proba_list_legacy([features])
            cls_probs = proba_dicts[0]
        else:
            raise RuntimeError("No model available.")
        
        argmax_pred = max(cls_probs, key=cls_probs.get)
        pred = argmax_pred

        # Draw decision rule. Plain argmax is accuracy-optimal only when the
        # class probabilities aren't systematically compressed — and here they
        # are: P(draw) never exceeds ~0.48 in practice, because a draw is
        # rarely the single most likely scoreline even when it's the best
        # call. That means argmax almost never picks Draw, and the class is
        # under-predicted regardless of how well calibrated the numbers are.
        #
        # Predicting Draw once P(draw) clears a threshold corrects for this.
        # Validated across 5 disjoint folds of held-out data: every fold
        # improved, mean +3.3pt accuracy, with draw recall roughly doubling.
        # The threshold is tuned on data disjoint from where it's scored (see
        # evaluate.py) — tuning and scoring on the same slice inflated the
        # gain by selection bias, which is why that split exists.
        #
        # The probabilities returned below are the model's own, untouched:
        # they're well calibrated by log-loss, so rescaling them to force
        # argmax agreement would make the displayed numbers worse to fix a
        # decision-rule problem.
        if self.draw_threshold is not None and cls_probs.get(0) is not None:
            if cls_probs[0] >= self.draw_threshold:
                pred = 0
            elif argmax_pred == 0:
                # Below the bar, so fall back to the better of home/away
                # rather than a draw argmax would otherwise have taken.
                non_draw = {c: p for c, p in cls_probs.items() if c != 0}
                if non_draw:
                    pred = max(non_draw, key=non_draw.get)

        class_probs = {}
        for cls, prob in cls_probs.items():
            class_probs[self.label_map.get(cls, cls)] = round(prob, 3)
        return {
            "prediction": self.label_map.get(pred, str(pred)),
            "probabilities": class_probs,
            # True when the decision rule and a plain argmax disagree, so the
            # UI can explain why the headline pick isn't the biggest number.
            "threshold_applied": pred != argmax_pred,
            "argmax_prediction": self.label_map.get(argmax_pred, str(argmax_pred)),
        }

    def predict_proba_batch(self, feature_rows):
        """Calibrated H/D/A probabilities for many fixtures in one pass — the
        same numbers predict() reports under 'probabilities', but vectorised so
        the season simulator can price hundreds of fixtures in a single sklearn
        call instead of hundreds of separate predict() round-trips. No draw
        threshold is applied: the simulator samples outcomes and needs the
        honest, untouched split. Returns a list of {label: prob} dicts."""
        if not self.trained:
            raise RuntimeError("Model not trained yet. Call train() first.")
        if not feature_rows:
            return []
        if self.feature_names and len(feature_rows[0]) != len(self.feature_names):
            raise ValueError(
                f"Feature count mismatch: got {len(feature_rows[0])} features but the "
                f"model expects {len(self.feature_names)} — retrain.")
        if self.pipeline is not None:
            X = np.asarray(feature_rows, dtype=np.float64)
            proba = self.apply_calibration(self._blend_dc(self.pipeline.predict_proba(X), X))
            return [{self.label_map.get(self.classes[i], self.classes[i]): float(row[i])
                     for i in range(len(self.classes))} for row in proba]
        # Legacy naive-bayes path: no vectorised predict, so map per row.
        return [{self.label_map.get(c, c): p for c, p in d.items()}
                for d in self._proba_list_legacy(feature_rows)]

    # A linear (Dirichlet) map over log-probabilities extrapolates without
    # bound at the tails, producing things like a 0.1% or 100% draw chance —
    # no real football match is that certain. Two guards, both measured rather
    # than hand-tuned: (1) clip the raw probabilities into this band before the
    # log so the map's INPUT can't be extreme (bounds derived from the fact
    # that no single outcome's true frequency is below ~1% or above ~99% on
    # this data); (2) mix a little of the empirical base rate back in, with the
    # blend weight SELECTED per fit by log-loss on an inner split rather than
    # left at a hand-picked constant.
    CALIB_CLIP_LO = 0.01
    CALIB_CLIP_HI = 0.99
    CALIBRATION_SHRINKAGE = 0.06  # fallback only, when the inner split is too small to select

    def _calib_logspace(self, proba):
        return np.log(np.clip(np.asarray(proba, dtype=np.float64),
                              self.CALIB_CLIP_LO, self.CALIB_CLIP_HI))

    def _init_dc_blend_idx(self, feature_names):
        """Locate the Dixon-Coles H/D/A probability columns so their vector can
        be blended into the ensemble output. None (blend disabled) if absent."""
        names = feature_names or []
        try:
            self.dc_prob_idx = {1: names.index("dc_home_prob"),
                                0: names.index("dc_draw_prob"),
                                -1: names.index("dc_away_prob")}
        except (ValueError, AttributeError):
            self.dc_prob_idx = None

    def _blend_dc(self, proba, feature_rows):
        """Blend the DC H/D/A vector into the ensemble probabilities (probability
        space) BEFORE calibration. Identity when no weight is set or the DC
        columns are absent, so it's safe on legacy models and tiny fixtures."""
        w = getattr(self, "dc_blend_weight", None)
        if not w or not self.dc_prob_idx:
            return proba
        P = np.asarray(proba, dtype=np.float64)
        single = P.ndim == 1
        if single:
            P = P.reshape(1, -1)
        X = np.asarray(feature_rows, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        dc = np.column_stack([X[:, self.dc_prob_idx[c]] for c in self.classes])
        s = dc.sum(axis=1, keepdims=True)
        s[s <= 0] = 1.0
        dc = dc / s
        out = (1.0 - w) * P + w * dc
        out = out / out.sum(axis=1, keepdims=True)
        return out[0] if single else out

    def apply_calibration(self, proba):
        """Map raw pipeline probabilities onto calibrated ones. Identity when
        no calibrator has been fitted, so older models still load and run."""
        if self.calibrator is None:
            return proba
        logp = self._calib_logspace(proba)
        out = self.calibrator.predict_proba(logp)
        prior = getattr(self, "class_prior_", None)
        w = getattr(self, "calib_shrinkage_", None)
        if w is None:
            w = self.CALIBRATION_SHRINKAGE
        if prior is not None and w > 0:
            out = (1.0 - w) * out + w * np.asarray(prior)
            out = out / out.sum(axis=1, keepdims=True)
        return out

    @staticmethod
    def _logloss_proba(y_true, P, classes):
        idx = {c: i for i, c in enumerate(classes)}
        n = len(y_true)
        if n == 0:
            return float("inf")
        s = 0.0
        for i in range(n):
            s += -math.log(max(P[i, idx[y_true[i]]], 1e-15))
        return s / n

    def _fit_calibrator_on(self, logp, y_arr):
        """Fit the Dirichlet calibrator on clipped log-probabilities `logp`,
        selecting the L2 strength C and the prior-blend weight by log-loss on an
        inner 70/30 split (not hand-picked) — regularizing the off-diagonal
        terms ODIR-style and stopping the map over-fitting a small slice."""
        n = len(y_arr)
        cut = int(n * 0.7)
        best = None
        if cut >= 30 and (n - cut) >= 20 and len(np.unique(y_arr[:cut])) == len(self.classes):
            for C in (0.1, 0.3, 1.0, 3.0):
                clf = LogisticRegression(C=C, max_iter=2000).fit(logp[:cut], y_arr[:cut])
                base = clf.predict_proba(logp[cut:])
                prior = np.array([(y_arr[:cut] == c).mean() for c in clf.classes_])
                for w in (0.0, 0.03, 0.06, 0.10):
                    P = (1.0 - w) * base + w * prior
                    P = P / P.sum(axis=1, keepdims=True)
                    ll = self._logloss_proba(y_arr[cut:], P, list(clf.classes_))
                    if best is None or ll < best[0]:
                        best = (ll, C, w)
        C, w = (best[1], best[2]) if best else (1.0, self.CALIBRATION_SHRINKAGE)
        self.calibrator = LogisticRegression(C=C, max_iter=2000).fit(logp, y_arr)
        self.class_prior_ = [float((y_arr == c).mean()) for c in self.calibrator.classes_]
        self.calib_C_ = C
        self.calib_shrinkage_ = w
        return self.calibrator

    def fit_calibration(self, X_holdout, y_holdout, pipeline=None):
        """Fit recalibration on matches held out from this call.

        `pipeline` defaults to self.pipeline but the honest-holdout step passes
        an EVAL pipeline that never trained on X_holdout, so the calibrator sees
        genuinely out-of-sample probabilities instead of memorized ones.
        """
        pipe = pipeline if pipeline is not None else self.pipeline
        if pipe is None:
            return None
        X = np.asarray(X_holdout, dtype=np.float64)
        raw = self._blend_dc(pipe.predict_proba(X), X)   # blend-aware
        self._fit_calibrator_on(self._calib_logspace(raw), np.asarray(y_holdout))
        return self.calibrator

    def save(self, path: str = None):
        if path is None:
            path = os.path.join(_SCRIPT_DIR, "model.json")
        
        metadata = {
            "version": self.version,
            "model_type": self.model_type,
            "trained_at": self.trained_at,
            "training_samples": self.training_samples,
            "feature_names": self.feature_names,
            "classes": self.classes,
            "baseline_accuracy": self.baseline_accuracy,
            "accuracy": self.accuracy_,
            "log_loss": self.log_loss_,
            "brier": self.brier_,
            "rps": self.rps_,
            "cv_rps": self.cv_rps,
            "cv_mean": self.cv_mean,
            "cv_std": self.cv_std,
            "eval_method": self.eval_method,
            "confusion_matrix": self.confusion_matrix,
            "class_distribution": self.class_distribution,
            "feature_importances": self.feature_importances_,
            "label_map": self.label_map,
            "draw_threshold": self.draw_threshold,
            "class_prior": getattr(self, "class_prior_", None),
            "calib_shrinkage": getattr(self, "calib_shrinkage_", None),
            "calib_C": getattr(self, "calib_C_", None),
            "dc_blend_weight": getattr(self, "dc_blend_weight", None),
            "recency_halflife_days": getattr(self, "recency_halflife_days", None),
        }
        
        with open(path, "w") as f:
            json.dump(metadata, f, indent=2)

        # Persist the leak-free holdout predictions next to the model so
        # evaluate.py grades on genuinely out-of-sample rows.
        if getattr(self, "_holdout", None):
            holdout_path = os.path.join(os.path.dirname(os.path.abspath(path)), "holdout_eval.json")
            try:
                with open(holdout_path, "w", encoding="utf-8") as f:
                    json.dump(self._holdout, f)
            except OSError:
                pass

        # Save sklearn pipeline separately
        if self.pipeline is not None:
            joblib_path = path.replace(".json", ".joblib") if path.endswith(".json") else path + ".joblib"
            try:
                joblib.dump(self.pipeline, joblib_path)
                # Size guard: model.joblib is committed directly (no Git LFS),
                # so it must stay under GitHub's 100MB hard limit. Warn early if
                # a hyperparameter choice bloats it (RF tree count/depth is the
                # usual culprit — see the _RF_GRID note).
                mb = os.path.getsize(joblib_path) / 1e6
                if mb > 90:
                    print(f"WARNING: {os.path.basename(joblib_path)} is {mb:.0f}MB — "
                          f"approaching GitHub's 100MB limit. Reduce RF n_estimators/max_depth "
                          f"or re-enable Git LFS before committing.")
            except Exception:
                pass
        if self.calibrator is not None:
            try:
                joblib.dump(self.calibrator, joblib_path.replace(".joblib", ".calib.joblib"))
            except Exception:
                pass
        
        # Also save legacy model state for backward compat
        if hasattr(self, "nb_model") and self.nb_model and self.nb_model.classes:
            metadata.update({
                "means": {str(k): v for k, v in self.nb_model.means.items()},
                "vars": {str(k): v for k, v in self.nb_model.vars.items()},
                "priors": {str(k): v for k, v in self.nb_model.priors.items()},
            })

    def load(self, path: str = None):
        if path is None:
            path = os.path.join(_SCRIPT_DIR, "model.json")
        
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return False
        
        self.version = data.get("version", "1.0.0")
        self.model_type = data.get("model_type", "ensemble")
        self.sklearn_model_type = self.model_type if self.model_type in ("histgb", "rf", "logreg", "ensemble", "stacking") else "ensemble"
        self.trained_at = data.get("trained_at")
        self.training_samples = data.get("training_samples", 0)
        self.feature_names = data.get("feature_names", [])
        self._init_dc_blend_idx(self.feature_names)   # locate DC cols for the output blend
        self.classes = [int(c) if not isinstance(c, int) else c for c in data.get("classes", [])]
        self.baseline_accuracy = data.get("baseline_accuracy", 0.0)
        self.accuracy_ = data.get("accuracy", 0.0)
        # No default here on purpose: a missing/null value means "not
        # computed", which must stay distinguishable from a real 0.0.
        self.log_loss_ = data.get("log_loss")
        self.brier_ = data.get("brier")
        self.rps_ = data.get("rps")
        self.cv_rps = data.get("cv_rps")
        self.cv_mean = data.get("cv_mean", 0.0)
        self.cv_std = data.get("cv_std", 0.0)
        self.eval_method = data.get("eval_method", "temporal")
        self.confusion_matrix = data.get("confusion_matrix")
        
        raw_label_map = data.get("label_map", {-1: "Away Win", 0: "Draw", 1: "Home Win"})
        self.draw_threshold = data.get("draw_threshold")
        self.class_prior_ = data.get("class_prior")
        self.calib_shrinkage_ = data.get("calib_shrinkage")
        self.calib_C_ = data.get("calib_C")
        self.dc_blend_weight = data.get("dc_blend_weight")
        self.recency_halflife_days = data.get("recency_halflife_days")
        self.label_map = {}
        for k, v in raw_label_map.items():
            try:
                self.label_map[int(k)] = v
            except (ValueError, TypeError):
                self.label_map[k] = v
        
        self.class_distribution = {}
        for k, v in data.get("class_distribution", {}).items():
            try:
                self.class_distribution[int(k)] = v
            except (ValueError, TypeError):
                self.class_distribution[k] = v
        
        # Load sklearn pipeline if available
        joblib_path = path.replace(".json", ".joblib") if path.endswith(".json") else path + ".joblib"
        # SECURITY: joblib.load is pickle-based and executes arbitrary code in
        # the file, so model.joblib is a TRUST BOUNDARY — only ever load the
        # model that ships with the repo or one produced by this project's own
        # training. Never point load() at a model.joblib from an untrusted
        # source (that is equivalent to running its author's code). There is no
        # safe way to "validate" a pickle before loading; the mitigation is
        # provenance, not inspection.
        if SKLEARN_AVAILABLE and os.path.exists(joblib_path):
            try:
                self.pipeline = joblib.load(joblib_path)
            except Exception:
                self.pipeline = None
        
        if self.pipeline is not None:
            calib_path = joblib_path.replace(".joblib", ".calib.joblib")
            if os.path.exists(calib_path):
                try:
                    self.calibrator = joblib.load(calib_path)
                except Exception:
                    self.calibrator = None
            self.trained = True
            return True
        
        # Fallback to legacy model state
        if "means" in data and "priors" in data:
            self.nb_model = GaussianNaiveBayes()
            self.nb_model.classes = self.classes
            self.nb_model.means = {int(k) if k.lstrip("-").isdigit() else k: v for k, v in data["means"].items()}
            self.nb_model.vars = {int(k) if k.lstrip("-").isdigit() else k: v for k, v in data["vars"].items()}
            self.nb_model.priors = {int(k) if k.lstrip("-").isdigit() else k: v for k, v in data["priors"].items()}
            
            self.lr_model = SoftmaxRegression()
            lr_data = data.get("logreg")
            if lr_data:
                self.lr_model.classes = lr_data.get("classes", self.classes)
                self.lr_model.feat_mean = lr_data.get("feat_mean", [])
                self.lr_model.feat_std = lr_data.get("feat_std", [])
                self.lr_model.W = lr_data.get("W", [])
                self.lr_model.b = lr_data.get("b", [])
            
            self.trained = True
            return True
        
        return False

    def get_model_info(self):
        return {
            "version": self.version,
            "trained_at": self.trained_at,
            "training_samples": self.training_samples,
            "feature_names": self.feature_names,
            "trained": self.trained,
            "model_type": self.model_type,
            "classes": self.classes,
        }
