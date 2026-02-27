"""Data loading utilities for various file formats."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

import pandas as pd

from src.utils.exceptions import DataLoadError
from src.utils.logging import get_logger

logger = get_logger("data.loaders")


class DataLoader(Protocol):
    """Protocol for data loaders."""

    def load(self, path: Path) -> pd.DataFrame:
        """Load data from a file path.

        Args:
            path: Path to the data file.

        Returns:
            Loaded data as a DataFrame.
        """
        ...


class CSVLoader:
    """Loader for CSV files.

    Attributes:
        kwargs: Extra keyword arguments forwarded to pandas.read_csv.
    """

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs

    def load(self, path: Path) -> pd.DataFrame:
        """Load a CSV file into a DataFrame.

        Args:
            path: Path to the CSV file.

        Returns:
            Loaded DataFrame.

        Raises:
            DataLoadError: If the file cannot be read.
        """
        logger.info("Loading CSV from %s", path)
        try:
            return pd.read_csv(path, **self.kwargs)
        except Exception as exc:
            raise DataLoadError(f"Failed to load CSV from {path}: {exc}") from exc


class ParquetLoader:
    """Loader for Parquet files."""

    def load(self, path: Path) -> pd.DataFrame:
        """Load a Parquet file into a DataFrame.

        Args:
            path: Path to the Parquet file.

        Returns:
            Loaded DataFrame.

        Raises:
            DataLoadError: If the file cannot be read.
        """
        logger.info("Loading Parquet from %s", path)
        try:
            return pd.read_parquet(path)
        except Exception as exc:
            raise DataLoadError(f"Failed to load Parquet from {path}: {exc}") from exc


_LOADERS: dict[str, type] = {
    "csv": CSVLoader,
    "parquet": ParquetLoader,
}


def get_loader(file_type: str, **kwargs: Any) -> DataLoader:
    """Get a data loader for the given file type.

    Args:
        file_type: File extension (e.g. 'csv', 'parquet').
        **kwargs: Extra arguments passed to the loader constructor.

    Returns:
        An instance of the appropriate loader.

    Raises:
        DataLoadError: If the file type is not supported.
    """
    loader_cls = _LOADERS.get(file_type)
    if loader_cls is None:
        raise DataLoadError(f"Unsupported file type: {file_type}. Supported: {list(_LOADERS)}")
    return loader_cls(**kwargs) if kwargs else loader_cls()
