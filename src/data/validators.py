"""Data validation utilities using Pandera schemas."""

from __future__ import annotations

import pandas as pd
import pandera as pa

from src.utils.exceptions import ValidationError
from src.utils.logging import get_logger

logger = get_logger("data.validators")


class BaseDataSchema(pa.DataFrameModel):
    """Base schema — subclass and add column annotations to validate datasets."""

    class Config:
        strict = False
        coerce = True


def validate_dataframe(df: pd.DataFrame, schema: type[pa.DataFrameModel]) -> pd.DataFrame:
    """Validate a DataFrame against a Pandera schema.

    Args:
        df: DataFrame to validate.
        schema: A Pandera DataFrameModel subclass defining expected columns.

    Returns:
        The validated (and possibly coerced) DataFrame.

    Raises:
        ValidationError: If the DataFrame fails schema checks.
    """
    logger.info("Validating DataFrame with schema %s", schema.__name__)
    try:
        return schema.validate(df)
    except pa.errors.SchemaError as exc:
        raise ValidationError(f"Schema validation failed: {exc}") from exc
