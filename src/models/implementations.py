"""Concrete model implementations wrapping sklearn estimators."""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier

from src.models.base import BaseModel
from src.utils.exceptions import ModelError
from src.utils.logging import get_logger

logger = get_logger("models.implementations")


class SklearnModelWrapper(BaseModel):
    """Wrapper that adapts an sklearn estimator to the BaseModel interface.

    Args:
        model: An sklearn estimator instance.
    """

    def __init__(self, model: Any) -> None:
        self.model = model

    def fit(self, X: pd.DataFrame, y: pd.Series) -> SklearnModelWrapper:
        """Train the underlying sklearn model.

        Args:
            X: Feature matrix.
            y: Target vector.

        Returns:
            The fitted wrapper.
        """
        logger.info("Fitting %s", type(self.model).__name__)
        self.model.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Generate predictions with the underlying sklearn model.

        Args:
            X: Feature matrix.

        Returns:
            Predictions as a Series.
        """
        preds = self.model.predict(X)
        return pd.Series(preds, index=X.index)


_MODEL_REGISTRY: dict[str, type] = {
    "random_forest": RandomForestClassifier,
    "gradient_boosting": GradientBoostingRegressor,
}


def get_model(model_type: str, **kwargs: Any) -> BaseModel:
    """Create a model instance by name.

    Args:
        model_type: Model identifier (e.g. 'random_forest', 'gradient_boosting').
        **kwargs: Hyperparameters forwarded to the sklearn constructor.

    Returns:
        A BaseModel-compatible instance.

    Raises:
        ModelError: If the model type is not recognized.
    """
    cls = _MODEL_REGISTRY.get(model_type)
    if cls is None:
        raise ModelError(f"Unknown model type: {model_type}. Available: {list(_MODEL_REGISTRY)}")
    logger.info("Creating model: %s", model_type)
    return SklearnModelWrapper(cls(**kwargs))
