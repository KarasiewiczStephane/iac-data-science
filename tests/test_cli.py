"""Tests for the CLI interface."""

from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest
from typer.testing import CliRunner

from src.cli import app
from src.models.implementations import get_model

runner = CliRunner()


@pytest.fixture()
def sample_csv(tmp_path: Path) -> Path:
    df = pd.DataFrame({"f0": [1, 2, 3, 4, 5], "f1": [10, 20, 30, 40, 50], "target": [0, 1, 0, 1, 0]})
    path = tmp_path / "train.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture()
def trained_model_path(tmp_path: Path, sample_csv: Path) -> Path:
    model = get_model("random_forest", n_estimators=5, random_state=42)
    df = pd.read_csv(sample_csv)
    X = df.drop(columns=["target"])
    y = df["target"]
    model.fit(X, y)
    path = tmp_path / "model.joblib"
    model.save(path)
    return path


class TestTrainCommand:
    def test_train_succeeds(self, tmp_path: Path, sample_csv: Path) -> None:
        out = tmp_path / "model.joblib"
        result = runner.invoke(app, ["train", str(sample_csv), "--output-path", str(out)])
        assert result.exit_code == 0
        assert out.exists()

    def test_train_missing_file(self, tmp_path: Path) -> None:
        result = runner.invoke(app, ["train", str(tmp_path / "nope.csv")])
        assert result.exit_code != 0


class TestPredictCommand:
    def test_predict_to_stdout(self, trained_model_path: Path, sample_csv: Path) -> None:
        # Create data without target column for prediction
        df = pd.read_csv(sample_csv).drop(columns=["target"])
        pred_csv = sample_csv.parent / "pred.csv"
        df.to_csv(pred_csv, index=False)

        result = runner.invoke(app, ["predict", str(trained_model_path), str(pred_csv)])
        assert result.exit_code == 0
        assert "predictions" in result.output.lower() or "0" in result.output


class TestEvaluateCommand:
    def test_evaluate(self, trained_model_path: Path, sample_csv: Path) -> None:
        result = runner.invoke(app, ["evaluate", str(trained_model_path), str(sample_csv)])
        assert result.exit_code == 0
        assert "accuracy" in result.output.lower() or "Metric" in result.output


class TestServeCommand:
    def test_serve_calls_uvicorn(self) -> None:
        with patch("uvicorn.run") as mock_run:
            runner.invoke(app, ["serve", "--port", "9999"])
            mock_run.assert_called_once()
            call_kwargs = mock_run.call_args
            assert call_kwargs.kwargs.get("port") == 9999 or call_kwargs[1].get("port") == 9999


class TestHelpMessages:
    def test_main_help(self) -> None:
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "Data Science Pipeline CLI" in result.output

    def test_train_help(self) -> None:
        result = runner.invoke(app, ["train", "--help"])
        assert result.exit_code == 0
        assert "Train" in result.output
