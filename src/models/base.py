"""Abstract base model interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import joblib
import pandas as pd

from src.utils.exceptions import ModelError
from src.utils.logging import get_logger

logger = get_logger("models.base")


class BaseModel(ABC):
    """Abstract base class for all models.

    Subclasses must implement ``fit`` and ``predict``.
    """

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> BaseModel:
        """Train the model on the provided data.

        Args:
            X: Feature matrix.
            y: Target vector.

        Returns:
            The fitted model instance.
        """

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Generate predictions for the provided features.

        Args:
            X: Feature matrix.

        Returns:
            Predicted values as a Series.
        """

    def save(self, path: Path) -> None:
        """Persist the model to disk using joblib.

        Args:
            path: Destination file path.

        Raises:
            ModelError: If serialization fails.
        """
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(self, path)
            logger.info("Model saved to %s", path)
        except Exception as exc:
            raise ModelError(f"Failed to save model to {path}: {exc}") from exc

    @classmethod
    def load(cls, path: Path) -> BaseModel:
        """Load a model from disk.

        Args:
            path: Path to the serialized model.

        Returns:
            Deserialized model instance.

        Raises:
            ModelError: If the file doesn't exist or deserialization fails.
        """
        if not path.exists():
            raise ModelError(f"Model file not found: {path}")
        try:
            model = joblib.load(path)
            logger.info("Model loaded from %s", path)
            return model
        except Exception as exc:
            raise ModelError(f"Failed to load model from {path}: {exc}") from exc
