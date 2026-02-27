"""Feature engineering pipeline builder."""

from __future__ import annotations

from sklearn.pipeline import Pipeline

from src.data.preprocessors import FeatureScaler, MissingValueHandler, OutlierRemover
from src.utils.logging import get_logger

logger = get_logger("data.features")


def build_feature_pipeline(
    missing_strategy: str = "mean",
    outlier_method: str = "iqr",
    outlier_threshold: float = 1.5,
    scale: bool = True,
) -> Pipeline:
    """Build a sklearn Pipeline for preprocessing and feature engineering.

    Args:
        missing_strategy: Strategy for filling missing values ('mean', 'median', 'mode').
        outlier_method: Outlier detection method ('iqr' or 'zscore').
        outlier_threshold: Threshold for outlier detection.
        scale: Whether to include feature scaling.

    Returns:
        A configured sklearn Pipeline.
    """
    steps: list[tuple[str, object]] = [
        ("missing", MissingValueHandler(strategy=missing_strategy)),
        ("outliers", OutlierRemover(method=outlier_method, threshold=outlier_threshold)),
    ]
    if scale:
        steps.append(("scaler", FeatureScaler()))

    logger.info("Built feature pipeline with %d steps", len(steps))
    return Pipeline(steps)
