"""Tests for data preprocessing and feature engineering."""

import numpy as np
import pandas as pd
import pytest

from src.data.features import build_feature_pipeline
from src.data.preprocessors import FeatureScaler, MissingValueHandler, OutlierRemover


@pytest.fixture()
def numeric_df() -> pd.DataFrame:
    np.random.seed(42)
    return pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0], "b": [10.0, 20.0, 30.0, 40.0, 50.0]})


@pytest.fixture()
def df_with_nans() -> pd.DataFrame:
    return pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": [np.nan, 5.0, 6.0], "c": ["x", "y", None]})


# ---------------------------------------------------------------------------
# MissingValueHandler
# ---------------------------------------------------------------------------


class TestMissingValueHandler:
    def test_mean_strategy(self, df_with_nans: pd.DataFrame) -> None:
        handler = MissingValueHandler(strategy="mean")
        result = handler.fit_transform(df_with_nans)
        assert not result["a"].isna().any()
        assert not result["b"].isna().any()
        assert result["a"].iloc[1] == pytest.approx(2.0)

    def test_median_strategy(self, df_with_nans: pd.DataFrame) -> None:
        handler = MissingValueHandler(strategy="median")
        result = handler.fit_transform(df_with_nans)
        assert not result["a"].isna().any()

    def test_mode_strategy(self, df_with_nans: pd.DataFrame) -> None:
        handler = MissingValueHandler(strategy="mode")
        result = handler.fit_transform(df_with_nans)
        assert result["c"].iloc[2] is not None

    def test_invalid_strategy_raises(self) -> None:
        handler = MissingValueHandler(strategy="invalid")
        with pytest.raises(ValueError, match="Unknown strategy"):
            handler.fit(pd.DataFrame({"a": [1.0]}))

    def test_fit_then_transform_matches_fit_transform(self, df_with_nans: pd.DataFrame) -> None:
        handler = MissingValueHandler(strategy="mean")
        r1 = handler.fit_transform(df_with_nans)
        handler2 = MissingValueHandler(strategy="mean")
        handler2.fit(df_with_nans)
        r2 = handler2.transform(df_with_nans)
        pd.testing.assert_frame_equal(r1, r2)


# ---------------------------------------------------------------------------
# OutlierRemover
# ---------------------------------------------------------------------------


class TestOutlierRemover:
    def test_iqr_removes_outliers(self) -> None:
        df = pd.DataFrame({"a": [1, 2, 3, 4, 5, 100]})
        remover = OutlierRemover(method="iqr", threshold=1.5)
        result = remover.fit_transform(df)
        assert 100 not in result["a"].values

    def test_zscore_removes_outliers(self) -> None:
        df = pd.DataFrame({"a": [1, 2, 3, 4, 5, 100]})
        remover = OutlierRemover(method="zscore", threshold=2.0)
        result = remover.fit_transform(df)
        assert 100 not in result["a"].values

    def test_no_outliers_preserves_data(self, numeric_df: pd.DataFrame) -> None:
        remover = OutlierRemover(method="iqr", threshold=3.0)
        result = remover.fit_transform(numeric_df)
        assert len(result) == len(numeric_df)

    def test_invalid_method_raises(self) -> None:
        remover = OutlierRemover(method="invalid")
        with pytest.raises(ValueError, match="Unknown method"):
            remover.fit(pd.DataFrame({"a": [1, 2]}))

    def test_fit_transform_consistency(self) -> None:
        df = pd.DataFrame({"a": [1, 2, 3, 4, 100]})
        r1 = OutlierRemover().fit_transform(df)
        r2 = OutlierRemover().fit(df).transform(df)
        pd.testing.assert_frame_equal(r1, r2)


# ---------------------------------------------------------------------------
# FeatureScaler
# ---------------------------------------------------------------------------


class TestFeatureScaler:
    def test_scales_to_zero_mean(self, numeric_df: pd.DataFrame) -> None:
        scaler = FeatureScaler()
        result = scaler.fit_transform(numeric_df)
        assert result["a"].mean() == pytest.approx(0.0, abs=1e-10)
        assert result["b"].mean() == pytest.approx(0.0, abs=1e-10)

    def test_scales_to_unit_variance(self, numeric_df: pd.DataFrame) -> None:
        scaler = FeatureScaler()
        result = scaler.fit_transform(numeric_df)
        assert result["a"].std(ddof=0) == pytest.approx(1.0, abs=1e-10)

    def test_preserves_non_numeric(self) -> None:
        df = pd.DataFrame({"a": [1.0, 2.0], "label": ["x", "y"]})
        scaler = FeatureScaler()
        result = scaler.fit_transform(df)
        assert list(result["label"]) == ["x", "y"]

    def test_fit_transform_consistency(self, numeric_df: pd.DataFrame) -> None:
        r1 = FeatureScaler().fit_transform(numeric_df)
        s = FeatureScaler()
        s.fit(numeric_df)
        r2 = s.transform(numeric_df)
        pd.testing.assert_frame_equal(r1, r2)


# ---------------------------------------------------------------------------
# Feature pipeline
# ---------------------------------------------------------------------------


class TestBuildFeaturePipeline:
    def test_default_pipeline_has_three_steps(self) -> None:
        pipeline = build_feature_pipeline()
        assert len(pipeline.steps) == 3

    def test_no_scaler(self) -> None:
        pipeline = build_feature_pipeline(scale=False)
        assert len(pipeline.steps) == 2

    def test_pipeline_fit_transform(self, numeric_df: pd.DataFrame) -> None:
        pipeline = build_feature_pipeline()
        result = pipeline.fit_transform(numeric_df)
        assert isinstance(result, pd.DataFrame)
        assert result["a"].mean() == pytest.approx(0.0, abs=1e-10)

    def test_pipeline_with_nans(self, df_with_nans: pd.DataFrame) -> None:
        pipeline = build_feature_pipeline(scale=False)
        result = pipeline.fit_transform(df_with_nans)
        assert not result.select_dtypes(include="number").isna().any().any()
