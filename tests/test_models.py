"""Tests for the models module."""

from pathlib import Path

import pandas as pd
import pytest
from sklearn.datasets import make_classification

from src.models.base import BaseModel
from src.models.implementations import SklearnModelWrapper, get_model
from src.models.trainer import ModelTrainer
from src.utils.exceptions import ModelError


@pytest.fixture()
def synthetic_data() -> tuple[pd.DataFrame, pd.Series]:
    X, y = make_classification(n_samples=100, n_features=5, random_state=42)
    return pd.DataFrame(X, columns=[f"f{i}" for i in range(5)]), pd.Series(y, name="target")


# ---------------------------------------------------------------------------
# BaseModel
# ---------------------------------------------------------------------------


class TestBaseModelSaveLoad:
    def test_save_and_load(self, tmp_path: Path, synthetic_data: tuple) -> None:
        X, y = synthetic_data
        model = get_model("random_forest", n_estimators=10, random_state=42)
        model.fit(X, y)

        path = tmp_path / "model.joblib"
        model.save(path)
        assert path.exists()

        loaded = BaseModel.load(path)
        pd.testing.assert_series_equal(model.predict(X), loaded.predict(X))

    def test_load_missing_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ModelError, match="not found"):
            BaseModel.load(tmp_path / "nope.joblib")

    def test_save_creates_parent_dirs(self, tmp_path: Path, synthetic_data: tuple) -> None:
        X, y = synthetic_data
        model = get_model("random_forest", n_estimators=5, random_state=42)
        model.fit(X, y)
        path = tmp_path / "sub" / "dir" / "model.joblib"
        model.save(path)
        assert path.exists()


# ---------------------------------------------------------------------------
# SklearnModelWrapper
# ---------------------------------------------------------------------------


class TestSklearnModelWrapper:
    def test_fit_predict(self, synthetic_data: tuple) -> None:
        X, y = synthetic_data
        model = get_model("random_forest", n_estimators=10, random_state=42)
        model.fit(X, y)
        preds = model.predict(X)
        assert isinstance(preds, pd.Series)
        assert len(preds) == len(X)

    def test_predictions_are_binary(self, synthetic_data: tuple) -> None:
        X, y = synthetic_data
        model = get_model("random_forest", n_estimators=10, random_state=42)
        model.fit(X, y)
        preds = model.predict(X)
        assert set(preds.unique()).issubset({0, 1})


# ---------------------------------------------------------------------------
# get_model factory
# ---------------------------------------------------------------------------


class TestGetModel:
    def test_random_forest(self) -> None:
        model = get_model("random_forest")
        assert isinstance(model, SklearnModelWrapper)

    def test_gradient_boosting(self) -> None:
        model = get_model("gradient_boosting")
        assert isinstance(model, SklearnModelWrapper)

    def test_unknown_raises(self) -> None:
        with pytest.raises(ModelError, match="Unknown model type"):
            get_model("xgboost")

    def test_kwargs_forwarded(self) -> None:
        model = get_model("random_forest", n_estimators=5)
        assert model.model.n_estimators == 5


# ---------------------------------------------------------------------------
# ModelTrainer
# ---------------------------------------------------------------------------


class TestModelTrainer:
    def test_train(self, synthetic_data: tuple) -> None:
        X, y = synthetic_data
        model = get_model("random_forest", n_estimators=10, random_state=42)
        trainer = ModelTrainer(model)
        trained = trainer.train(X, y)
        assert trained is model
        assert len(trained.predict(X)) == len(X)

    def test_train_with_save(self, tmp_path: Path, synthetic_data: tuple) -> None:
        X, y = synthetic_data
        path = tmp_path / "saved.joblib"
        model = get_model("random_forest", n_estimators=10, random_state=42)
        trainer = ModelTrainer(model, config={"save_path": str(path)})
        trainer.train(X, y)
        assert path.exists()
