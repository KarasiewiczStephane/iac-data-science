"""CLI interface for iac-data-science."""

from __future__ import annotations

import logging
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from src.data.pipeline import DataPipeline
from src.models.base import BaseModel
from src.models.evaluation import ModelEvaluator
from src.models.implementations import get_model
from src.models.trainer import ModelTrainer
from src.utils.config import Settings

app = typer.Typer(name="iac-ds", help="Data Science Pipeline CLI")
console = Console()

settings = Settings.from_yaml()


@app.command()
def train(
    data_path: Path = typer.Argument(..., help="Path to training data (CSV)"),
    model_type: str = typer.Option(settings.model_type, help="Model algorithm"),
    target_column: str = typer.Option("target", help="Name of target column"),
    output_path: Path = typer.Option("models/model.joblib", help="Model output path"),
    random_state: int = typer.Option(settings.random_state, help="Random seed"),
) -> None:
    """Train a model on the provided data."""
    logging.basicConfig(level=settings.log_level)
    console.print(f"[bold]Training {model_type} on {data_path}[/bold]")

    pipeline = DataPipeline()
    df = pipeline.run(data_path)

    X = df.drop(columns=[target_column])
    y = df[target_column]

    model = get_model(model_type, random_state=random_state)
    trainer = ModelTrainer(model, config={"save_path": str(output_path)})
    trainer.train(X, y)

    console.print(f"[green]Model saved to {output_path}[/green]")


@app.command()
def predict(
    model_path: Path = typer.Argument(..., help="Path to trained model"),
    data_path: Path = typer.Argument(..., help="Path to data for prediction"),
    output_path: Path = typer.Option(None, help="Path to save predictions CSV"),
) -> None:
    """Run predictions with a trained model."""
    logging.basicConfig(level=settings.log_level)
    model = BaseModel.load(model_path)

    pipeline = DataPipeline()
    df = pipeline.run(data_path)

    preds = model.predict(df)
    console.print(f"Generated {len(preds)} predictions")

    if output_path:
        preds.to_csv(output_path, index=False, header=["prediction"])
        console.print(f"[green]Predictions saved to {output_path}[/green]")
    else:
        console.print(preds.to_string())


@app.command()
def evaluate(
    model_path: Path = typer.Argument(..., help="Path to trained model"),
    test_data: Path = typer.Argument(..., help="Path to test data (CSV)"),
    target_column: str = typer.Option("target", help="Name of target column"),
    task_type: str = typer.Option("classification", help="'classification' or 'regression'"),
) -> None:
    """Evaluate model performance on test data."""
    logging.basicConfig(level=settings.log_level)
    model = BaseModel.load(model_path)

    pipeline = DataPipeline()
    df = pipeline.run(test_data)

    X = df.drop(columns=[target_column])
    y = df[target_column]
    preds = model.predict(X)

    evaluator = ModelEvaluator(task_type)
    metrics = evaluator.evaluate(y, preds)

    table = Table(title="Evaluation Metrics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    for key, value in metrics.to_dict().items():
        table.add_row(key, f"{value:.4f}")
    console.print(table)


@app.command()
def serve(
    host: str = typer.Option(settings.api_host, help="API host"),
    port: int = typer.Option(settings.api_port, help="API port"),
    reload: bool = typer.Option(False, help="Enable auto-reload"),
) -> None:
    """Start the API server."""
    import uvicorn

    console.print(f"[bold]Starting server on {host}:{port}[/bold]")
    uvicorn.run("src.api.app:app", host=host, port=port, reload=reload)


if __name__ == "__main__":
    app()
