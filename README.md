# iac-data-science

> Insurance Analytics & Claims — Data Science Pipeline

## Overview

End-to-end data science pipeline for insurance claim risk classification. Includes data loading, validation, preprocessing, model training/evaluation, a REST API for inference, and a CLI for pipeline orchestration.

## Features

- **Data loading & validation** — CSV/Parquet loaders with Pandera schema enforcement
- **Preprocessing pipeline** — sklearn-compatible transformers for missing values, outliers, and scaling
- **Model training** — Pluggable model architecture (RandomForest, GradientBoosting) with serialization
- **Evaluation & reporting** — Classification/regression metrics with cross-validation and JSON reports
- **REST API** — FastAPI-based prediction endpoint with health checks
- **CLI** — Typer-powered interface for train, predict, evaluate, and serve commands
- **Docker** — Multi-stage production build with non-root user and health checks
- **CI/CD** — GitHub Actions with lint, test (80%+ coverage), and Docker build

## Quick Start

```bash
# Clone
git clone git@github.com:KarasiewiczStephane/iac-data-science.git
cd iac-data-science

# Install dependencies
pip install -r requirements.txt

# Run the demo
make demo

# Start the API server
make serve

# Run tests
make test
```

## CLI Usage

```bash
# Train a model
python -m src.cli train data/sample/train.csv --model-type random_forest

# Evaluate on test data
python -m src.cli evaluate models/model.joblib data/sample/test.csv

# Run predictions
python -m src.cli predict models/model.joblib data/sample/test.csv

# Start API server
python -m src.cli serve --port 8000
```

## API

Once running, Swagger docs are available at `http://localhost:8000/docs`.

| Endpoint | Method | Description |
|---|---|---|
| `/api/v1/health` | GET | Health check |
| `/api/v1/predict` | POST | Run model prediction |

## Project Structure

```
iac-data-science/
├── src/
│   ├── data/               # Loaders, validators, preprocessors, features
│   ├── models/             # Base model, implementations, trainer, evaluation
│   ├── api/                # FastAPI app, routes, schemas
│   ├── utils/              # Config, logging, exceptions
│   └── cli.py              # Typer CLI
├── tests/                  # Unit and integration tests
├── scripts/                # Demo and utility scripts
├── configs/                # YAML configuration
├── data/sample/            # Sample train/test datasets
├── .github/workflows/      # CI/CD pipeline
├── Dockerfile              # Multi-stage production build
├── docker-compose.yml      # Docker Compose setup
├── Makefile                # Common dev commands
├── requirements.txt        # Python dependencies
└── pyproject.toml          # Project metadata and tool config
```

## Docker

```bash
# Build and run
make docker

# Or with docker compose
docker compose up --build
```

## Development

```bash
# Install pre-commit hooks
pre-commit install

# Lint and format
make lint

# Run tests with coverage
make test
```

## Demo Results

Running the demo (`make demo`) on the sample insurance claims dataset:

| Metric | Value |
|---|---|
| Accuracy | 0.98 |
| Precision | 0.98 |
| Recall | 0.98 |
| F1 Score | 0.98 |
| CV Mean | 0.97 |

## License

MIT
