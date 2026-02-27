"""Tests for the end-to-end demo script."""

from pathlib import Path

import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from src.data.pipeline import DataPipeline
from src.data.preprocessors import FeatureScaler, MissingValueHandler
from src.models.evaluation import ModelEvaluator
from src.models.implementations import get_model
from src.models.trainer import ModelTrainer


class TestEndToEndPipeline:
    """Integration test mirroring the demo script logic."""

    @pytest.fixture()
    def sample_data(self, tmp_path: Path) -> tuple[Path, Path]:
        train = pd.DataFrame(
            {
                "age": [25, 35, 45, 55, 30],
                "income": [40000, 60000, 80000, 50000, 70000],
                "credit_score": [650, 700, 750, 600, 720],
                "claim_amount": [1000, 8000, 3000, 12000, 2000],
                "num_claims": [1, 3, 2, 5, 1],
                "is_smoker": [0, 1, 0, 1, 0],
                "target": [0, 1, 0, 1, 0],
            }
        )
        test = train.copy()
        train_path = tmp_path / "train.csv"
        test_path = tmp_path / "test.csv"
        train.to_csv(train_path, index=False)
        test.to_csv(test_path, index=False)
        return train_path, test_path

    def test_full_pipeline(self, sample_data: tuple, tmp_path: Path) -> None:
        train_path, test_path = sample_data

        pipeline = DataPipeline()
        train_df = pipeline.run(train_path)
        test_df = pipeline.run(test_path)

        X_train = train_df.drop(columns=["target"])
        y_train = train_df["target"]
        X_test = test_df.drop(columns=["target"])
        y_test = test_df["target"]

        feat_pipeline = Pipeline(
            [
                ("missing", MissingValueHandler(strategy="mean")),
                ("scaler", FeatureScaler()),
            ]
        )
        X_train = feat_pipeline.fit_transform(X_train)
        X_test = feat_pipeline.transform(X_test)

        model = get_model("random_forest", n_estimators=10, random_state=42)
        trainer = ModelTrainer(model)
        trained = trainer.train(X_train, y_train)

        preds = trained.predict(X_test)
        assert len(preds) == len(y_test)

        evaluator = ModelEvaluator("classification")
        metrics = evaluator.evaluate(y_test, preds)
        assert 0 <= metrics.accuracy <= 1

        model_path = tmp_path / "model.joblib"
        trained.save(model_path)
        assert model_path.exists()
