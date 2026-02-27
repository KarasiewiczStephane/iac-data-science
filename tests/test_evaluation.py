"""Tests for model evaluation and reporting."""

import json
from pathlib import Path

import pandas as pd
import pytest
from sklearn.datasets import make_classification

from src.models.evaluation import ClassificationMetrics, ModelEvaluator, RegressionMetrics
from src.models.implementations import get_model
from src.models.reporting import generate_report

# ---------------------------------------------------------------------------
# ClassificationMetrics
# ---------------------------------------------------------------------------


class TestClassificationMetrics:
    def test_perfect_classifier(self) -> None:
        y_true = pd.Series([0, 1, 1, 0])
        y_pred = pd.Series([0, 1, 1, 0])
        evaluator = ModelEvaluator("classification")
        metrics = evaluator.evaluate(y_true, y_pred)
        assert isinstance(metrics, ClassificationMetrics)
        assert metrics.accuracy == 1.0
        assert metrics.f1 == 1.0

    def test_random_classifier(self) -> None:
        y_true = pd.Series([0, 0, 1, 1])
        y_pred = pd.Series([1, 1, 0, 0])
        evaluator = ModelEvaluator("classification")
        metrics = evaluator.evaluate(y_true, y_pred)
        assert metrics.accuracy == 0.0

    def test_to_dict(self) -> None:
        m = ClassificationMetrics(accuracy=0.9, precision=0.88, recall=0.87, f1=0.85)
        d = m.to_dict()
        assert d["accuracy"] == 0.9
        assert "f1" in d

    def test_single_class(self) -> None:
        y_true = pd.Series([1, 1, 1])
        y_pred = pd.Series([1, 1, 1])
        evaluator = ModelEvaluator("classification")
        metrics = evaluator.evaluate(y_true, y_pred)
        assert metrics.accuracy == 1.0


# ---------------------------------------------------------------------------
# RegressionMetrics
# ---------------------------------------------------------------------------


class TestRegressionMetrics:
    def test_perfect_prediction(self) -> None:
        y_true = pd.Series([1.0, 2.0, 3.0])
        y_pred = pd.Series([1.0, 2.0, 3.0])
        evaluator = ModelEvaluator("regression")
        metrics = evaluator.evaluate(y_true, y_pred)
        assert isinstance(metrics, RegressionMetrics)
        assert metrics.mse == pytest.approx(0.0)
        assert metrics.r2 == pytest.approx(1.0)

    def test_known_error(self) -> None:
        y_true = pd.Series([1.0, 2.0, 3.0])
        y_pred = pd.Series([2.0, 3.0, 4.0])
        evaluator = ModelEvaluator("regression")
        metrics = evaluator.evaluate(y_true, y_pred)
        assert metrics.mse == pytest.approx(1.0)
        assert metrics.mae == pytest.approx(1.0)
        assert metrics.rmse == pytest.approx(1.0)

    def test_to_dict(self) -> None:
        m = RegressionMetrics(mse=1.0, mae=0.5, r2=0.9, rmse=1.0)
        d = m.to_dict()
        assert d["mse"] == 1.0


# ---------------------------------------------------------------------------
# ModelEvaluator
# ---------------------------------------------------------------------------


class TestModelEvaluator:
    def test_invalid_task_type_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown task type"):
            ModelEvaluator("clustering")

    def test_cross_validate(self) -> None:
        X, y = make_classification(n_samples=50, n_features=5, random_state=42)
        X_df = pd.DataFrame(X)
        y_s = pd.Series(y)
        model = get_model("random_forest", n_estimators=5, random_state=42)
        evaluator = ModelEvaluator("classification")
        result = evaluator.cross_validate(model, X_df, y_s, cv=3)
        assert "mean" in result
        assert "std" in result
        assert len(result["scores"]) == 3


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


class TestGenerateReport:
    def test_classification_report(self, tmp_path: Path) -> None:
        metrics = ClassificationMetrics(accuracy=0.95, precision=0.94, recall=0.93, f1=0.92)
        out = tmp_path / "report.json"
        generate_report(metrics, out)
        assert out.exists()
        data = json.loads(out.read_text())
        assert data["task_type"] == "ClassificationMetrics"
        assert data["metrics"]["accuracy"] == 0.95

    def test_regression_report(self, tmp_path: Path) -> None:
        metrics = RegressionMetrics(mse=0.5, mae=0.3, r2=0.9, rmse=0.707)
        out = tmp_path / "sub" / "report.json"
        generate_report(metrics, out)
        assert out.exists()
