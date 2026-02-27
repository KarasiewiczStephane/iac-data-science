"""Model training orchestrator."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.models.base import BaseModel
from src.utils.logging import get_logger

logger = get_logger("models.trainer")


class ModelTrainer:
    """Orchestrates model training and optional persistence.

    Args:
        model: A BaseModel instance to train.
        config: Optional configuration dictionary (e.g. save path).
    """

    def __init__(self, model: BaseModel, config: dict | None = None) -> None:
        self.model = model
        self.config = config or {}

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
    ) -> BaseModel:
        """Train the model and optionally save it.

        Args:
            X_train: Training features.
            y_train: Training target.

        Returns:
            The trained model.
        """
        logger.info("Starting training with %d samples", len(X_train))
        self.model.fit(X_train, y_train)
        logger.info("Training complete")

        save_path = self.config.get("save_path")
        if save_path:
            self.model.save(Path(save_path))

        return self.model
