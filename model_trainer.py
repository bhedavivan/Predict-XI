import json
import math
import os
import random
from collections import Counter

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


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


class MatchPredictorModel:
    """Wrapper around Gaussian Naive Bayes for match prediction."""

    def __init__(self):
        self.model = GaussianNaiveBayes()
        self.trained = False
        self.label_map = {-1: "Away Win", 0: "Draw", 1: "Home Win"}

    def train(self, X, y):
        """Train the model and return metrics."""
        # Split into train/test (80/20)
        combined = list(zip(X, y))
        random.shuffle(combined)
        split_idx = int(len(combined) * 0.8)
        train_data = combined[:split_idx]
        test_data = combined[split_idx:]

        X_train = [d[0] for d in train_data]
        y_train = [d[1] for d in train_data]
        X_test = [d[0] for d in test_data]
        y_test = [d[1] for d in test_data]

        self.model.fit(X_train, y_train)

        # Evaluate
        y_pred = self.model.predict(X_test)
        accuracy = sum(1 for i in range(len(y_test)) if y_test[i] == y_pred[i]) / max(len(y_test), 1)

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

        self.trained = True

        return {
            "accuracy": accuracy,
            "classification_report": report,
            "test_samples": len(y_test),
        }

    def predict(self, features):
        """Predict match outcome. Returns probabilities and class."""
        if not self.trained:
            raise RuntimeError("Model not trained yet. Call train() first.")

        probas = self.model.predict_proba([features])[0]
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
            "classes": self.model.classes,
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
            self.model.classes = data["classes"]
            self.model.means = {int(k) if k.startswith("-") or k.isdigit() else k: v
                                for k, v in data["means"].items()}
            self.model.vars = {int(k) if k.startswith("-") or k.isdigit() else k: v
                               for k, v in data["vars"].items()}
            self.model.priors = {int(k) if k.startswith("-") or k.isdigit() else k: v
                                 for k, v in data["priors"].items()}
            self.trained = True
            return True
        except (FileNotFoundError, json.JSONDecodeError, KeyError):
            return False