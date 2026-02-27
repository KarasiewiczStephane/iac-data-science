"""Pydantic request and response schemas for the API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """Request payload for the /predict endpoint.

    Attributes:
        features: Dictionary mapping feature names to their values.
    """

    features: dict[str, Any] = Field(..., description="Feature dictionary")


class PredictionResponse(BaseModel):
    """Response payload for the /predict endpoint.

    Attributes:
        prediction: The model's predicted value.
        model_version: Version string of the model that made the prediction.
    """

    prediction: Any
    model_version: str


class HealthResponse(BaseModel):
    """Response payload for the /health endpoint.

    Attributes:
        status: Service health status.
        version: Application version string.
    """

    status: str
    version: str
