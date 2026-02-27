"""Model evaluation utilities and metrics calculation."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import cross_val_score

from src.utils.logging import get_logger

logger = get_logger("models.evaluation")


@dataclass
class ClassificationMetrics:
    """Container for classification evaluation metrics.

    Attributes:
        accuracy: Fraction of correct predictions.
        precision: Weighted precision score.
        recall: Weighted recall score.
        f1: Weighted F1 score.
    """

    accuracy: float
    precision: float
    recall: float
    f1: float

    def to_dict(self) -> dict:
        """Convert metrics to a plain dictionary."""
        return asdict(self)


@dataclass
class RegressionMetrics:
    """Container for regression evaluation metrics.

    Attributes:
        mse: Mean squared error.
        mae: Mean absolute error.
        r2: R-squared coefficient of determination.
        rmse: Root mean squared error.
    """

    mse: float
    mae: float
    r2: float
    rmse: float

    def to_dict(self) -> dict:
        """Convert metrics to a plain dictionary."""
        return asdict(self)


class ModelEvaluator:
    """Evaluate models for classification or regression tasks.

    Args:
        task_type: Either 'classification' or 'regression'.
    """

    def __init__(self, task_type: str = "classification") -> None:
        if task_type not in {"classification", "regression"}:
            raise ValueError(f"Unknown task type: {task_type}")
        self.task_type = task_type

    def evaluate(
        self,
        y_true: pd.Series | np.ndarray,
        y_pred: pd.Series | np.ndarray,
    ) -> ClassificationMetrics | RegressionMetrics:
        """Compute metrics comparing true and predicted values.

        Args:
            y_true: Ground-truth labels or values.
            y_pred: Model predictions.

        Returns:
            A ClassificationMetrics or RegressionMetrics dataclass.
        """
        if self.task_type == "classification":
            metrics = ClassificationMetrics(
                accuracy=accuracy_score(y_true, y_pred),
                precision=precision_score(y_true, y_pred, average="weighted", zero_division=0),
                recall=recall_score(y_true, y_pred, average="weighted", zero_division=0),
                f1=f1_score(y_true, y_pred, average="weighted", zero_division=0),
            )
        else:
            mse = mean_squared_error(y_true, y_pred)
            metrics = RegressionMetrics(
                mse=mse,
                mae=mean_absolute_error(y_true, y_pred),
                r2=r2_score(y_true, y_pred),
                rmse=math.sqrt(mse),
            )
        logger.info("Evaluation (%s): %s", self.task_type, metrics)
        return metrics

    def cross_validate(
        self,
        model: object,
        X: pd.DataFrame,
        y: pd.Series,
        cv: int = 5,
    ) -> dict:
        """Run cross-validation and return summary statistics.

        Args:
            model: A sklearn-compatible estimator (or wrapper with .model attribute).
            X: Feature matrix.
            y: Target vector.
            cv: Number of cross-validation folds.

        Returns:
            Dict with 'mean', 'std', and 'scores' keys.
        """
        estimator = getattr(model, "model", model)
        scores = cross_val_score(estimator, X, y, cv=cv)
        result = {"mean": float(scores.mean()), "std": float(scores.std()), "scores": scores.tolist()}
        logger.info("Cross-validation (cv=%d): mean=%.4f std=%.4f", cv, result["mean"], result["std"])
        return result
