"""FastAPI dependency injection helpers."""

from __future__ import annotations

from pathlib import Path

from src.models.base import BaseModel as MLModel
from src.utils.logging import get_logger

logger = get_logger("api.dependencies")

_loaded_model: MLModel | None = None


def load_model(path: Path) -> None:
    """Load a model from disk into the global state.

    Args:
        path: Path to a serialized model file.
    """
    global _loaded_model
    _loaded_model = MLModel.load(path)
    logger.info("Model loaded from %s", path)


def get_model() -> MLModel | None:
    """Return the currently loaded model (or None).

    Returns:
        The loaded model instance, or None if no model is loaded.
    """
    return _loaded_model
