"""Unit tests for model_trainer module."""

import pytest
import os
import json
import numpy as np
from model_trainer import (
    GaussianNaiveBayes, MatchPredictorModel, MODEL_VERSION, _rps,
    _evaluate_pipeline, _build_sklearn_pipeline, SKLEARN_AVAILABLE,
)


class TestGaussianNaiveBayes:
    """Tests for GaussianNaiveBayes class."""

    def test_fit_and_predict(self):
        """Test basic fit and predict."""
        X = [
            [1.0, 2.0],
            [2.0, 3.0],
            [3.0, 4.0],
            [4.0, 5.0],
        ]
        y = [0, 0, 1, 1]

        model = GaussianNaiveBayes()
        model.fit(X, y)

        assert model.classes == [0, 1]
        assert len(model.means[0]) == 2
        assert len(model.vars[0]) == 2

        predictions = model.predict(X)
        assert len(predictions) == 4

    def test_predict_proba(self):
        """Test predict_proba returns probabilities."""
        X = [
            [1.0, 2.0],
            [2.0, 3.0],
            [3.0, 4.0],
            [4.0, 5.0],
        ]
        y = [0, 0, 1, 1]

        model = GaussianNaiveBayes()
        model.fit(X, y)

        probas = model.predict_proba(X)
        assert len(probas) == 4
        for p in probas:
            assert 0 in p
            assert 1 in p
            assert abs(sum(p.values()) - 1.0) < 1e-6

    def test_predict_before_fit_raises(self):
        """Test that predict before fit raises error."""
        model = GaussianNaiveBayes()
        with pytest.raises(RuntimeError):
            model.predict([[1.0, 2.0]])

    def test_single_class(self):
        """Test with single class."""
        X = [[1.0], [2.0], [3.0]]
        y = [0, 0, 0]

        model = GaussianNaiveBayes()
        model.fit(X, y)

        predictions = model.predict(X)
        assert all(p == 0 for p in predictions)


class TestMatchPredictorModel:
    """Tests for MatchPredictorModel class."""

    def test_init(self):
        """Test initialization."""
        model = MatchPredictorModel()
        assert not model.trained
        assert model.version == MODEL_VERSION
        assert model.label_map == {-1: "Away Win", 0: "Draw", 1: "Home Win"}

    def test_train_basic(self):
        """Test basic training."""
        X = [
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0],
            [3.0, 4.0, 5.0],
            [4.0, 5.0, 6.0],
            [5.0, 6.0, 7.0],
            [6.0, 7.0, 8.0],
        ]
        y = [-1, -1, 0, 0, 1, 1]

        model = MatchPredictorModel()
        metrics = model.train(X, y, feature_names=["f1", "f2", "f3"])

        assert model.trained
        assert "accuracy" in metrics
        assert "classification_report" in metrics
        assert "cv_scores" in metrics
        assert "confusion_matrix" in metrics
        assert "class_distribution" in metrics
        assert model.trained_at is not None
        assert model.training_samples > 0
        assert model.feature_names == ["f1", "f2", "f3"]

    def test_train_with_cv(self):
        """Test training with cross-validation."""
        X = [
            [1.0, 2.0],
            [2.0, 3.0],
            [3.0, 4.0],
            [4.0, 5.0],
            [5.0, 6.0],
            [6.0, 7.0],
            [7.0, 8.0],
            [8.0, 9.0],
            [9.0, 10.0],
            [10.0, 11.0],
        ]
        y = [-1, -1, 0, 0, 1, 1, 1, 1, 0, -1]

        model = MatchPredictorModel()
        metrics = model.train(X, y, cv_folds=3)

        assert "cv_scores" in metrics
        assert len(metrics["cv_scores"]) >= 1
        assert "cv_mean" in metrics
        assert "cv_std" in metrics

    def test_predict(self):
        """Test prediction."""
        X = [
            [1.0, 2.0],
            [2.0, 3.0],
            [3.0, 4.0],
            [4.0, 5.0],
        ]
        y = [-1, -1, 1, 1]

        model = MatchPredictorModel()
        model.train(X, y)

        result = model.predict([2.5, 3.5])
        assert "prediction" in result
        assert "probabilities" in result
        assert result["prediction"] in ["Home Win", "Draw", "Away Win"]
        assert sum(result["probabilities"].values()) == pytest.approx(1.0, rel=1e-2)

    def test_predict_before_train_raises(self):
        """Test that predict before train raises error."""
        model = MatchPredictorModel()
        with pytest.raises(RuntimeError):
            model.predict([1.0, 2.0])

    def test_save_and_load(self, tmp_path):
        """Test saving and loading model."""
        X = [
            [1.0, 2.0],
            [2.0, 3.0],
            [3.0, 4.0],
            [4.0, 5.0],
        ]
        y = [-1, -1, 1, 1]

        model = MatchPredictorModel()
        model.train(X, y)

        model_path = tmp_path / "test_model.json"
        model.save(str(model_path))

        # Load into new model
        new_model = MatchPredictorModel()
        loaded = new_model.load(str(model_path))

        assert loaded
        assert new_model.trained
        assert new_model.version == model.version
        assert new_model.trained_at == model.trained_at
        assert new_model.training_samples == model.training_samples
        assert new_model.feature_names == model.feature_names

        # Predictions should match
        pred1 = model.predict([2.5, 3.5])
        pred2 = new_model.predict([2.5, 3.5])
        assert pred1["prediction"] == pred2["prediction"]

    def test_load_nonexistent_file(self):
        """Test loading nonexistent file returns False."""
        model = MatchPredictorModel()
        loaded = model.load("/nonexistent/path/model.json")
        assert not loaded

    def test_get_model_info(self):
        """Test getting model info."""
        model = MatchPredictorModel()
        info = model.get_model_info()

        assert "version" in info
        assert "trained_at" in info
        assert "training_samples" in info
        assert "feature_names" in info
        assert "trained" in info
        assert info["trained"] is False

    def test_confusion_matrix(self):
        """Test confusion matrix computation."""
        # Use more samples to ensure all 3 classes appear in test set
        X = [
            [1.0, 2.0],
            [2.0, 3.0],
            [3.0, 4.0],
            [4.0, 5.0],
            [5.0, 6.0],
            [6.0, 7.0],
            [7.0, 8.0],
            [8.0, 9.0],
            [9.0, 10.0],
            [10.0, 11.0],
            [11.0, 12.0],
            [12.0, 13.0],
        ]
        y = [-1, -1, -1, -1, 0, 0, 0, 0, 1, 1, 1, 1]

        model = MatchPredictorModel()
        metrics = model.train(X, y)

        assert "confusion_matrix" in metrics
        cm = metrics["confusion_matrix"]
        assert len(cm) == 3  # 3 classes
        assert all(len(row) == 3 for row in cm)

    def test_prob_metrics_failure_reports_honestly(self, monkeypatch):
        """If holdout log-loss/Brier computation fails, they must come back
        as None/unavailable -- not a misleadingly perfect 0.0 (a silent
        fallback that reads as legitimate is exactly the bug class this
        guards against)."""
        X = [
            [1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0],
            [5.0, 6.0], [6.0, 7.0], [7.0, 8.0], [8.0, 9.0],
            [9.0, 10.0], [10.0, 11.0], [11.0, 12.0], [12.0, 13.0],
        ]
        y = [-1, -1, -1, -1, 0, 0, 0, 0, 1, 1, 1, 1]

        model = MatchPredictorModel()
        monkeypatch.setattr(
            model, "_compute_prob_metrics",
            lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("forced failure")),
        )
        metrics = model.train(X, y)

        assert metrics["log_loss"] is None
        assert metrics["brier"] is None
        assert model.log_loss_ is None
        assert model.brier_ is None

    def test_class_distribution(self):
        """Test class distribution tracking."""
        # Use more samples to ensure all classes appear in training set
        X = [
            [1.0, 2.0],
            [2.0, 3.0],
            [3.0, 4.0],
            [4.0, 5.0],
            [5.0, 6.0],
            [6.0, 7.0],
            [7.0, 8.0],
            [8.0, 9.0],
            [9.0, 10.0],
            [10.0, 11.0],
            [11.0, 12.0],
            [12.0, 13.0],
        ]
        y = [-1, -1, -1, -1, 0, 0, 0, 0, 1, 1, 1, 1]

        model = MatchPredictorModel()
        metrics = model.train(X, y)

        assert "class_distribution" in metrics
        dist = metrics["class_distribution"]
        # Distribution is computed from the holdout training set (80% of data).
        # The model is later refit on ALL data, so model.training_samples counts
        # every row — hence we compare against the reported train-split size.
        assert sum(dist.values()) == metrics["train_samples"]
        assert model.training_samples == len(X)
        assert all(v > 0 for v in dist.values())
        assert set(dist.keys()) == {-1, 0, 1}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

class TestDrawThreshold:
    """The draw decision rule. Plain argmax under-predicts draws because
    P(draw) is systematically compressed (never above ~0.48), so a draw is
    rarely the single most likely outcome even when it's the best call."""

    def _fitted(self, threshold=None):
        X = [[float(i), float(i % 3)] for i in range(60)]
        y = [(-1 if i % 3 == 0 else (0 if i % 3 == 1 else 1)) for i in range(60)]
        m = MatchPredictorModel()
        m.train(X, y)
        m.draw_threshold = threshold
        return m

    def test_no_threshold_keeps_argmax(self):
        m = self._fitted(threshold=None)
        r = m.predict([1.0, 1.0])
        assert r["prediction"] == r["argmax_prediction"]
        assert r["threshold_applied"] is False

    def test_low_threshold_forces_draw(self):
        """A threshold below any plausible P(draw) must always pick Draw."""
        m = self._fitted(threshold=0.0)
        r = m.predict([1.0, 1.0])
        assert r["prediction"] == "Draw"

    def test_high_threshold_never_picks_draw(self):
        """Above the achievable range, the rule must fall back to the better
        of home/away rather than leaving a draw that argmax would have taken."""
        m = self._fitted(threshold=1.01)
        r = m.predict([1.0, 1.0])
        assert r["prediction"] in ("Home Win", "Away Win")

    def test_probabilities_are_never_rewritten_by_the_rule(self):
        """The displayed numbers stay the model's own — the rule changes the
        pick, not the calibrated probabilities behind it."""
        base = self._fitted(threshold=None).predict([1.0, 1.0])["probabilities"]
        forced = self._fitted(threshold=0.0).predict([1.0, 1.0])["probabilities"]
        assert base == forced

    def test_threshold_survives_save_and_load(self, tmp_path):
        m = self._fitted(threshold=0.35)
        p = tmp_path / "m.json"
        m.save(str(p))
        m2 = MatchPredictorModel()
        assert m2.load(str(p))
        assert m2.draw_threshold == 0.35


class TestRPS:
    """Ranked Probability Score — the ordinal proper scoring rule now reported
    as the primary metric. Its whole point over log-loss is ordinality: a
    confident wrong pick two steps away (Home when Away) must be penalised more
    than one step away (Home when Draw)."""

    CLASSES = [-1, 0, 1]  # away < draw < home

    def test_perfect_confident_prediction_scores_zero(self):
        assert _rps([1], [[0.0, 0.0, 1.0]], self.CLASSES) == 0.0

    def test_uniform_is_between_zero_and_one(self):
        r = _rps([1], [[1 / 3, 1 / 3, 1 / 3]], self.CLASSES)
        assert 0.0 < r < 1.0

    def test_ordinality_far_error_worse_than_adjacent(self):
        far = _rps([1], [[1.0, 0.0, 0.0]], self.CLASSES)       # Away when Home
        adjacent = _rps([1], [[0.0, 1.0, 0.0]], self.CLASSES)  # Draw when Home
        assert far > adjacent
        assert far == 1.0 and adjacent == 0.5

    def test_reported_in_metrics_and_survives_save_load(self, tmp_path):
        X = [[float(i), float(i % 3)] for i in range(60)]
        y = [(-1 if i % 3 == 0 else (0 if i % 3 == 1 else 1)) for i in range(60)]
        m = MatchPredictorModel()
        metrics = m.train(X, y)
        assert "rps" in metrics  # present even if None on a tiny set
        m.rps_ = 0.21
        p = tmp_path / "m.json"
        m.save(str(p))
        m2 = MatchPredictorModel()
        assert m2.load(str(p))
        assert m2.rps_ == 0.21


class TestRecencyWeights:
    """Exponential time-decay weighting: recent matches must weigh more, and
    the whole thing must no-op cleanly when disabled (it is gated on RPS, not
    assumed to help)."""

    def test_recent_matches_weigh_more(self):
        dates = ["2020-01-01", "2023-01-01", "2026-01-01"]
        w = MatchPredictorModel._recency_weights(dates, halflife_days=365)
        assert w is not None
        assert w[2] > w[1] > w[0]
        assert abs(w[2] - 1.0) < 1e-9  # most recent match is the reference

    def test_disabled_returns_none(self):
        assert MatchPredictorModel._recency_weights(["2020-01-01"], None) is None
        assert MatchPredictorModel._recency_weights(None, 365) is None

    def test_training_with_recency_still_produces_a_working_model(self):
        X = [[float(i), float(i % 3)] for i in range(60)]
        y = [(-1 if i % 3 == 0 else (0 if i % 3 == 1 else 1)) for i in range(60)]
        dates = [f"20{20 + i // 30:02d}-{1 + i % 12:02d}-01" for i in range(60)]
        m = MatchPredictorModel()
        m.train(X, y, sample_dates=dates, recency_halflife_days=540)
        r = m.predict([1.0, 1.0])
        assert set(r["probabilities"]) == {"Home Win", "Draw", "Away Win"}

    def test_unparseable_date_gets_min_not_max_weight(self):
        # A bad/missing date must not be treated as the most-recent match.
        w = MatchPredictorModel._recency_weights(
            ["2020-01-01", "not-a-date", "2026-01-01"], halflife_days=365)
        assert w[1] == min(w)          # unparseable -> smallest weight
        assert w[1] < w[2]


@pytest.mark.skipif(not SKLEARN_AVAILABLE, reason="sklearn required")
class TestCalibration:
    """The reworked calibrator: selection, tail-clip, apply, and save/load
    round-trip of the calibration params (previously untested)."""

    def _data(self, n=180, feats=6):
        rng = np.random.RandomState(1)
        X, y = [], []
        for i in range(n):
            cls = i % 3            # 0,1,2 -> map to -1,0,1
            base = [rng.normal(cls, 1.0) for _ in range(feats)]
            X.append(base)
            y.append(cls - 1)
        return X, y

    def test_train_with_calibrate_fits_a_calibrator_and_predicts(self):
        X, y = self._data()
        m = MatchPredictorModel()
        m.train(X, y, calibrate=True)
        assert m.calibrator is not None
        r = m.predict(X[0])
        probs = list(r["probabilities"].values())
        assert abs(sum(probs) - 1.0) < 0.02
        assert all(0.0 <= p <= 1.0 for p in probs)

    def test_apply_calibration_is_identity_without_a_calibrator(self):
        m = MatchPredictorModel()
        m.classes = [-1, 0, 1]
        proba = np.array([[0.2, 0.3, 0.5]])
        assert np.allclose(m.apply_calibration(proba), proba)

    def test_input_clip_bounds_the_map(self):
        m = MatchPredictorModel()
        m.classes = [-1, 0, 1]
        logp = m._calib_logspace(np.array([[0.0, 0.0, 1.0]]))  # extreme 0/1 inputs
        assert np.isfinite(logp).all()  # clipped, no -inf from log(0)

    def test_calibration_params_survive_save_load(self, tmp_path):
        X, y = self._data()
        m = MatchPredictorModel()
        m.train(X, y, calibrate=True)
        p = tmp_path / "m.json"
        m.save(str(p))
        m2 = MatchPredictorModel()
        assert m2.load(str(p))
        assert m2.calib_C_ == m.calib_C_
        assert m2.calib_shrinkage_ == m.calib_shrinkage_
        assert m2.class_prior_ == m.class_prior_


@pytest.mark.skipif(not SKLEARN_AVAILABLE, reason="sklearn required")
class TestBurnInFolds:
    """burn_in_folds trains the earliest folds but excludes them from the
    reported score (the ratings warm-up fix) — previously untested."""

    def test_burn_in_reduces_scored_folds(self):
        rng = np.random.RandomState(2)
        X = rng.normal(size=(300, 6))
        y = np.array([(-1, 0, 1)[i % 3] for i in range(300)])
        pipe = _build_sklearn_pipeline("histgb", 6, select_k=None)
        r0 = _evaluate_pipeline(pipe, X, y, n_folds=4, burn_in_folds=0)
        pipe2 = _build_sklearn_pipeline("histgb", 6, select_k=None)
        r1 = _evaluate_pipeline(pipe2, X, y, n_folds=4, burn_in_folds=1)
        assert r1["n_folds_used"] == r0["n_folds_used"] - 1
