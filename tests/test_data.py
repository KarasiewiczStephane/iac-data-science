"""Tests for data loading, validation, and pipeline modules."""

from pathlib import Path

import pandas as pd
import pandera as pa
import pytest

from src.data.loaders import CSVLoader, ParquetLoader, get_loader
from src.data.pipeline import DataPipeline
from src.data.validators import BaseDataSchema, validate_dataframe
from src.utils.exceptions import DataLoadError, ValidationError

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_df() -> pd.DataFrame:
    return pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0], "label": ["x", "y", "x"]})


@pytest.fixture()
def csv_path(tmp_path: Path, sample_df: pd.DataFrame) -> Path:
    path = tmp_path / "data.csv"
    sample_df.to_csv(path, index=False)
    return path


@pytest.fixture()
def parquet_path(tmp_path: Path, sample_df: pd.DataFrame) -> Path:
    path = tmp_path / "data.parquet"
    sample_df.to_parquet(path, index=False)
    return path


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


class TestCSVLoader:
    def test_load(self, csv_path: Path, sample_df: pd.DataFrame) -> None:
        df = CSVLoader().load(csv_path)
        assert list(df.columns) == ["a", "b", "label"]
        assert len(df) == 3

    def test_load_with_kwargs(self, tmp_path: Path) -> None:
        path = tmp_path / "semi.csv"
        path.write_text("a;b\n1;2\n3;4\n")
        df = CSVLoader(sep=";").load(path)
        assert list(df.columns) == ["a", "b"]

    def test_load_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(DataLoadError, match="Failed to load CSV"):
            CSVLoader().load(tmp_path / "missing.csv")


class TestParquetLoader:
    def test_load(self, parquet_path: Path) -> None:
        df = ParquetLoader().load(parquet_path)
        assert len(df) == 3

    def test_round_trip_integrity(self, tmp_path: Path, sample_df: pd.DataFrame) -> None:
        path = tmp_path / "rt.parquet"
        sample_df.to_parquet(path, index=False)
        df = ParquetLoader().load(path)
        pd.testing.assert_frame_equal(df, sample_df)

    def test_load_missing_raises(self, tmp_path: Path) -> None:
        with pytest.raises(DataLoadError, match="Failed to load Parquet"):
            ParquetLoader().load(tmp_path / "missing.parquet")


class TestGetLoader:
    def test_csv(self) -> None:
        loader = get_loader("csv")
        assert isinstance(loader, CSVLoader)

    def test_parquet(self) -> None:
        loader = get_loader("parquet")
        assert isinstance(loader, ParquetLoader)

    def test_unsupported_type_raises(self) -> None:
        with pytest.raises(DataLoadError, match="Unsupported file type"):
            get_loader("xlsx")


# ---------------------------------------------------------------------------
# Validators
# ---------------------------------------------------------------------------


class _TestSchema(BaseDataSchema):
    a: pa.typing.Series[int]
    b: pa.typing.Series[float]


class TestValidateDataframe:
    def test_valid(self, sample_df: pd.DataFrame) -> None:
        result = validate_dataframe(sample_df, _TestSchema)
        assert len(result) == 3

    def test_missing_column_raises(self) -> None:
        df = pd.DataFrame({"a": [1, 2]})
        with pytest.raises(ValidationError, match="Schema validation failed"):
            validate_dataframe(df, _TestSchema)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class TestDataPipeline:
    def test_run_csv(self, csv_path: Path) -> None:
        pipeline = DataPipeline()
        df = pipeline.run(csv_path)
        assert len(df) == 3

    def test_run_parquet(self, parquet_path: Path) -> None:
        pipeline = DataPipeline()
        df = pipeline.run(parquet_path)
        assert len(df) == 3

    def test_run_with_schema(self, csv_path: Path) -> None:
        pipeline = DataPipeline(schema=_TestSchema)
        df = pipeline.run(csv_path)
        assert len(df) == 3

    def test_missing_source_raises(self, tmp_path: Path) -> None:
        pipeline = DataPipeline()
        with pytest.raises(DataLoadError, match="does not exist"):
            pipeline.run(tmp_path / "nope.csv")

    def test_empty_csv(self, tmp_path: Path) -> None:
        path = tmp_path / "empty.csv"
        path.write_text("a,b\n")
        pipeline = DataPipeline()
        df = pipeline.run(path)
        assert len(df) == 0
