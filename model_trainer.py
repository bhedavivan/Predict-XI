import json
import math
import os
import random
from collections import Counter
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Model version
MODEL_VERSION = "2.1.0"


class GaussianNaiveBayes:
    """Gaussian Naive Bayes classifier implemented from scratch."""

    def __init__(self):
        self.classes = []
        self.means = {}
        self.vars = {}
        self.priors = {}

    def fit(self, X, y):
        """Fit the model to training data."""
        self.classes = sorted(set(y))
        n_samples = len(X)
        n_features = len(X[0]) if X else 0

        # Group by class
        class_samples = {c: [] for c in self.classes}
        for i, label in enumerate(y):
            class_samples[label].append(X[i])

        for c in self.classes:
            samples = class_samples[c]
            self.priors[c] = len(samples) / n_samples

            # Calculate mean and variance for each feature
            self.means[c] = []
            self.vars[c] = []

            for f in range(n_features):
                vals = [s[f] for s in samples]
                mean = sum(vals) / len(vals)
                var = sum((v - mean) ** 2 for v in vals) / len(vals)
                # Add small epsilon to avoid zero variance
                var = max(var, 1e-9)
                self.means[c].append(mean)
                self.vars[c].append(var)

    def _gaussian_pdf(self, x, mean, var):
        """Compute Gaussian probability density function."""
        coeff = 1.0 / math.sqrt(2.0 * math.pi * var)
        exponent = math.exp(-((x - mean) ** 2) / (2.0 * var))
        return coeff * exponent

    def predict_proba(self, X):
        """Predict class probabilities for samples."""
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

            # Convert log probabilities to probabilities
            max_log = max(class_probs.values())
            exp_probs = {c: math.exp(p - max_log) for c, p in class_probs.items()}
            total = sum(exp_probs.values())
            predictions.append({c: p / total for c, p in exp_probs.items()})

        return predictions

    def predict(self, X):
        """Predict class labels for samples."""
        probas = self.predict_proba(X)
        return [max(p, key=p.get) for p in probas]


class SoftmaxRegression:
    """Multinomial logistic (softmax) regression, implemented from scratch.

    Unlike Naive Bayes, this does NOT assume features are independent, so its
    probability estimates stay well-calibrated even when features are
    correlated (form, Elo, goals all move together). Features are standardized
    and the weights are learned with mini-batch gradient descent + L2.
    """

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
        self.W: List[List[float]] = []   # [n_features][n_classes]
        self.b: List[float] = []         # [n_classes]

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


class MatchPredictorModel:
    """Ensemble of Gaussian Naive Bayes + Softmax Regression for match prediction.

    model_type selects how predictions are made:
      - "nb":       Gaussian Naive Bayes only
      - "logreg":   Softmax regression only
      - "ensemble": average of both probability distributions (default)
    """

    def __init__(self, model_type: str = "logreg"):
        self.model = GaussianNaiveBayes()       # kept name `model` for compatibility
        self.lr = SoftmaxRegression()
        self.model_type = model_type
        self.classes: List[int] = []
        self.trained = False
        self.label_map = {-1: "Away Win", 0: "Draw", 1: "Home Win"}
        self.version = MODEL_VERSION
        self.trained_at: Optional[str] = None
        self.training_samples: int = 0
        self.feature_names: List[str] = []
        self.confusion_matrix: Optional[List[List[int]]] = None
        self.class_distribution: Dict[int, int] = {}

    def _proba_list(self, X):
        """Probability dicts for each sample, per the active model_type."""
        if self.model_type == "nb":
            return self.model.predict_proba(X)
        if self.model_type == "logreg":
            return self.lr.predict_proba(X)
        # ensemble: average the two distributions
        pn = self.model.predict_proba(X)
        pl = self.lr.predict_proba(X)
        merged = []
        for a, b in zip(pn, pl):
            merged.append({c: (a.get(c, 0.0) + b.get(c, 0.0)) / 2.0 for c in self.classes})
        return merged

    def _predict_list(self, X):
        return [max(p, key=p.get) for p in self._proba_list(X)]

    def train(self, X, y, feature_names: Optional[List[str]] = None,
              cv_folds: int = 5, temporal: bool = True):
        """Train the model and return metrics.

        Args:
            X: Feature matrix (assumed date-ordered when temporal=True)
            y: Labels
            feature_names: Optional list of feature names
            cv_folds: Number of cross-validation folds (default 5)
            temporal: If True, hold out the most recent 20% by time (honest
                forward-test, no leakage). If False, shuffle then split.

        Returns:
            Dictionary with accuracy, log_loss, brier, classification_report,
            test_samples, cv_scores, confusion_matrix, baseline_accuracy
        """
        # Store feature names
        self.feature_names = feature_names or [f"feature_{i}" for i in range(len(X[0]) if X else 0)]

        # Split into train/test (80/20). Temporal split keeps the ordering so
        # the model is only ever tested on matches that happened *after* training.
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

        self.classes = sorted(set(y_train))
        self.model.fit(X_train, y_train)
        self.lr.fit(X_train, y_train)

        # Evaluate the active model (ensemble by default) on the holdout set
        y_proba = self._proba_list(X_test)
        y_pred = [max(p, key=p.get) for p in y_proba]
        accuracy = sum(1 for i in range(len(y_test)) if y_test[i] == y_pred[i]) / max(len(y_test), 1)

        # Probabilistic quality (log-loss + multiclass Brier) on the test set
        log_loss = self._log_loss(y_test, y_proba)
        brier = self._brier(y_test, y_proba)

        # Naive baseline: always predict the majority class from the training set
        majority = Counter(y_train).most_common(1)[0][0] if y_train else 1
        baseline_acc = sum(1 for v in y_test if v == majority) / max(len(y_test), 1)

        # Per-class metrics
        classes = sorted(set(y_train))
        report = {}
        for c in classes:
            tp = sum(1 for i in range(len(y_test)) if y_test[i] == c and y_pred[i] == c)
            fp = sum(1 for i in range(len(y_test)) if y_test[i] != c and y_pred[i] == c)
            fn = sum(1 for i in range(len(y_test)) if y_test[i] == c and y_pred[i] != c)
            precision = tp / max(tp + fp, 1)
            recall = tp / max(tp + fn, 1)
            f1 = 2 * precision * recall / max(precision + recall, 1e-9)
            report[str(c)] = {
                "precision": round(precision, 3),
                "recall": round(recall, 3),
                "f1-score": round(f1, 3),
                "support": sum(1 for v in y_test if v == c),
            }

        # Confusion matrix
        self.confusion_matrix = self._compute_confusion_matrix(y_test, y_pred, classes)
        
        # Class distribution
        self.class_distribution = {c: y_train.count(c) for c in classes}

        # Cross-validation
        cv_scores = self._cross_validate(X, y, cv_folds)

        # Refit both models on ALL data so the SAVED model uses every match
        # (metrics above stay honest — they came from the held-out future).
        self.model.fit(X, y)
        self.lr.fit(X, y)

        self.trained = True
        self.trained_at = datetime.utcnow().isoformat() + "Z"
        self.training_samples = len(X)

        return {
            "accuracy": round(accuracy, 4),
            "baseline_accuracy": round(baseline_acc, 4),
            "log_loss": round(log_loss, 4),
            "brier": round(brier, 4),
            "eval_method": "temporal" if temporal else "shuffled",
            "classification_report": report,
            "test_samples": len(y_test),
            "train_samples": len(X_train),
            "cv_scores": cv_scores,
            "cv_mean": round(sum(cv_scores) / len(cv_scores), 4) if cv_scores else 0,
            "cv_std": round(self._std(cv_scores), 4) if cv_scores else 0,
            "confusion_matrix": self.confusion_matrix,
            "class_distribution": self.class_distribution,
        }

    def _log_loss(self, y_true, y_proba) -> float:
        """Mean negative log-likelihood of the true class (lower is better)."""
        if not y_true:
            return 0.0
        total = 0.0
        for true, probs in zip(y_true, y_proba):
            p = max(probs.get(true, 1e-15), 1e-15)
            total += -math.log(p)
        return total / len(y_true)

    def _brier(self, y_true, y_proba) -> float:
        """Multiclass Brier score: mean squared error vs one-hot truth."""
        if not y_true:
            return 0.0
        total = 0.0
        for true, probs in zip(y_true, y_proba):
            for c in self.model.classes:
                target = 1.0 if c == true else 0.0
                total += (probs.get(c, 0.0) - target) ** 2
        return total / len(y_true)

    def _cross_validate(self, X, y, folds: int = 5) -> List[float]:
        """Perform k-fold cross-validation."""
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
                else:  # ensemble
                    pn, pl = nb.predict_proba(X_test), lr.predict_proba(X_test)
                    y_pred = []
                    for a, b in zip(pn, pl):
                        avg = {c: (a.get(c, 0.0) + b.get(c, 0.0)) / 2.0 for c in classes}
                        y_pred.append(max(avg, key=avg.get))

            accuracy = sum(1 for j in range(len(y_test)) if y_test[j] == y_pred[j]) / max(len(y_test), 1)
            scores.append(accuracy)

        return scores

    def _std(self, values: List[float]) -> float:
        """Calculate standard deviation."""
        if not values:
            return 0.0
        mean = sum(values) / len(values)
        return math.sqrt(sum((v - mean) ** 2 for v in values) / len(values))

    def predict(self, features):
        """Predict match outcome. Returns probabilities and class."""
        if not self.trained:
            raise RuntimeError("Model not trained yet. Call train() first.")

        probas = self._proba_list([features])[0]
        pred = max(probas, key=probas.get)

        class_probs = {}
        for cls, prob in probas.items():
            class_probs[self.label_map[cls]] = round(prob, 3)

        return {
            "prediction": self.label_map[pred],
            "probabilities": class_probs,
        }

    def save(self, path: str = None):
        """Save the trained model to disk as JSON."""
        if path is None:
            path = os.path.join(_SCRIPT_DIR, "model.json")
        data = {
            "version": self.version,
            "model_type": self.model_type,
            "trained_at": self.trained_at,
            "training_samples": self.training_samples,
            "feature_names": self.feature_names,
            "classes": self.model.classes,
            # Gaussian Naive Bayes parameters
            "nb": {
                "means": {str(k): v for k, v in self.model.means.items()},
                "vars": {str(k): v for k, v in self.model.vars.items()},
                "priors": {str(k): v for k, v in self.model.priors.items()},
            },
            # Softmax regression parameters
            "logreg": {
                "classes": self.lr.classes,
                "feat_mean": self.lr.feat_mean,
                "feat_std": self.lr.feat_std,
                "W": self.lr.W,
                "b": self.lr.b,
            },
            # Legacy top-level NB keys (backward compatibility with v2.0 loaders)
            "means": {str(k): v for k, v in self.model.means.items()},
            "vars": {str(k): v for k, v in self.model.vars.items()},
            "priors": {str(k): v for k, v in self.model.priors.items()},
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def load(self, path: str = None):
        """Load a trained model from disk."""
        if path is None:
            path = os.path.join(_SCRIPT_DIR, "model.json")
        try:
            with open(path) as f:
                data = json.load(f)

            def _int_keys(d):
                return {int(k) if k.lstrip("-").isdigit() else k: v for k, v in d.items()}

            self.model.classes = data["classes"]
            self.classes = list(data["classes"])

            # Gaussian Naive Bayes (supports both nested "nb" and legacy layout)
            nb = data.get("nb", data)
            self.model.means = _int_keys(nb["means"])
            self.model.vars = _int_keys(nb["vars"])
            self.model.priors = _int_keys(nb["priors"])

            # Softmax regression, if present
            lr = data.get("logreg")
            if lr:
                self.lr.classes = list(lr["classes"])
                self.lr.feat_mean = lr["feat_mean"]
                self.lr.feat_std = lr["feat_std"]
                self.lr.W = lr["W"]
                self.lr.b = lr["b"]
                self.model_type = data.get("model_type", "ensemble")
            else:
                # Old model file without softmax → fall back to NB only
                self.model_type = "nb"

            self.version = data.get("version", "1.0.0")
            self.trained_at = data.get("trained_at")
            self.training_samples = data.get("training_samples", 0)
            self.feature_names = data.get("feature_names", [])
            self.trained = True
            return True
        except (FileNotFoundError, json.JSONDecodeError, KeyError):
            return False

    def _compute_confusion_matrix(self, y_true, y_pred, classes):
        """Compute confusion matrix."""
        n_classes = len(classes)
        cm = [[0] * n_classes for _ in range(n_classes)]
        class_to_idx = {c: i for i, c in enumerate(classes)}
        
        for true, pred in zip(y_true, y_pred):
            if true in class_to_idx and pred in class_to_idx:
                cm[class_to_idx[true]][class_to_idx[pred]] += 1
        
        return cm

    def get_model_info(self) -> Dict[str, Any]:
        """Get model metadata."""
        return {
            "version": self.version,
            "trained_at": self.trained_at,
            "training_samples": self.training_samples,
            "feature_names": self.feature_names,
            "trained": self.trained,
        }
