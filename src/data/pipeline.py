"""Data pipeline orchestrating load and validation steps."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pandera as pa

from src.data.loaders import get_loader
from src.data.validators import validate_dataframe
from src.utils.config import Settings
from src.utils.exceptions import DataLoadError
from src.utils.logging import get_logger

logger = get_logger("data.pipeline")


class DataPipeline:
    """Orchestrates data loading and validation.

    Attributes:
        config: Application settings.
        schema: Optional Pandera schema for validation.
    """

    def __init__(
        self,
        config: Settings | None = None,
        schema: type[pa.DataFrameModel] | None = None,
    ) -> None:
        self.config = config or Settings()
        self.schema = schema

    def run(self, source: Path) -> pd.DataFrame:
        """Execute the data pipeline: load then optionally validate.

        Args:
            source: Path to the source data file.

        Returns:
            Loaded (and optionally validated) DataFrame.

        Raises:
            DataLoadError: If the file doesn't exist or can't be loaded.
        """
        if not source.exists():
            raise DataLoadError(f"Source file does not exist: {source}")

        file_type = source.suffix.lstrip(".")
        logger.info("Running pipeline for %s (type=%s)", source, file_type)

        loader = get_loader(file_type)
        df = loader.load(source)
        logger.info("Loaded %d rows, %d columns", len(df), len(df.columns))

        if self.schema is not None:
            df = validate_dataframe(df, self.schema)
            logger.info("Validation passed")

        return df
