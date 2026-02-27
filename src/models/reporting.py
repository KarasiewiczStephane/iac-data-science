"""Model evaluation report generation."""

from __future__ import annotations

import json
from pathlib import Path

from src.models.evaluation import ClassificationMetrics, RegressionMetrics
from src.utils.logging import get_logger

logger = get_logger("models.reporting")


def generate_report(
    metrics: ClassificationMetrics | RegressionMetrics,
    output_path: Path,
) -> None:
    """Write evaluation metrics to a JSON report file.

    Args:
        metrics: A ClassificationMetrics or RegressionMetrics instance.
        output_path: Destination file path (will be created / overwritten).
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = {"task_type": type(metrics).__name__, "metrics": metrics.to_dict()}
    output_path.write_text(json.dumps(data, indent=2))
    logger.info("Report written to %s", output_path)
