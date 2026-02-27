"""Sklearn-compatible data preprocessing transformers."""

from __future__ import annotations

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler

from src.utils.logging import get_logger

logger = get_logger("data.preprocessors")


class MissingValueHandler(BaseEstimator, TransformerMixin):
    """Fill missing values using a specified strategy.

    Args:
        strategy: One of 'mean', 'median', or 'mode'.
    """

    def __init__(self, strategy: str = "mean") -> None:
        self.strategy = strategy

    def fit(self, X: pd.DataFrame, y: object = None) -> MissingValueHandler:
        """Compute fill values from training data.

        Args:
            X: Training DataFrame.
            y: Ignored.

        Returns:
            Fitted transformer.
        """
        if self.strategy == "mean":
            self.fill_values_ = X.select_dtypes(include="number").mean()
        elif self.strategy == "median":
            self.fill_values_ = X.select_dtypes(include="number").median()
        elif self.strategy == "mode":
            self.fill_values_ = X.mode().iloc[0]
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
        logger.info("Fitted MissingValueHandler with strategy=%s", self.strategy)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply fill values to the DataFrame.

        Args:
            X: DataFrame to transform.

        Returns:
            DataFrame with missing values filled.
        """
        return X.fillna(self.fill_values_)


class OutlierRemover(BaseEstimator, TransformerMixin):
    """Remove outliers using IQR or z-score method.

    Args:
        method: Detection method ('iqr' or 'zscore').
        threshold: IQR multiplier or z-score cutoff.
    """

    def __init__(self, method: str = "iqr", threshold: float = 1.5) -> None:
        self.method = method
        self.threshold = threshold

    def fit(self, X: pd.DataFrame, y: object = None) -> OutlierRemover:
        """Compute outlier bounds from training data.

        Args:
            X: Training DataFrame.
            y: Ignored.

        Returns:
            Fitted transformer.
        """
        numeric = X.select_dtypes(include="number")
        if self.method == "iqr":
            q1 = numeric.quantile(0.25)
            q3 = numeric.quantile(0.75)
            iqr = q3 - q1
            self.lower_ = q1 - self.threshold * iqr
            self.upper_ = q3 + self.threshold * iqr
        elif self.method == "zscore":
            self.mean_ = numeric.mean()
            self.std_ = numeric.std()
        else:
            raise ValueError(f"Unknown method: {self.method}")
        self.numeric_cols_ = numeric.columns.tolist()
        logger.info("Fitted OutlierRemover with method=%s", self.method)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Remove rows containing outliers.

        Args:
            X: DataFrame to filter.

        Returns:
            DataFrame with outlier rows removed.
        """
        mask = pd.Series(True, index=X.index)
        numeric = X[self.numeric_cols_]
        if self.method == "iqr":
            mask = ((numeric >= self.lower_) & (numeric <= self.upper_)).all(axis=1)
        elif self.method == "zscore":
            z_scores = ((numeric - self.mean_) / self.std_).abs()
            mask = (z_scores <= self.threshold).all(axis=1)
        return X.loc[mask].reset_index(drop=True)


class FeatureScaler(BaseEstimator, TransformerMixin):
    """Standard-scale numeric columns while preserving non-numeric ones.

    Wraps sklearn's StandardScaler with column tracking.
    """

    def __init__(self) -> None:
        self._scaler = StandardScaler()

    def fit(self, X: pd.DataFrame, y: object = None) -> FeatureScaler:
        """Fit the scaler on numeric columns.

        Args:
            X: Training DataFrame.
            y: Ignored.

        Returns:
            Fitted transformer.
        """
        self.numeric_cols_ = X.select_dtypes(include="number").columns.tolist()
        if self.numeric_cols_:
            self._scaler.fit(X[self.numeric_cols_])
        logger.info("Fitted FeatureScaler on %d numeric columns", len(self.numeric_cols_))
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Scale numeric columns to zero mean and unit variance.

        Args:
            X: DataFrame to transform.

        Returns:
            DataFrame with scaled numeric columns.
        """
        result = X.copy()
        if self.numeric_cols_:
            scaled = self._scaler.transform(X[self.numeric_cols_])
            result[self.numeric_cols_] = scaled
        return result
