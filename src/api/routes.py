"""API route definitions."""

from __future__ import annotations

import pandas as pd
from fastapi import APIRouter, HTTPException

from src.api.dependencies import get_model
from src.api.schemas import HealthResponse, PredictionRequest, PredictionResponse
from src.utils.config import Settings
from src.utils.logging import get_logger

logger = get_logger("api.routes")

router = APIRouter()

_settings = Settings.from_yaml()


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Return service health status."""
    return HealthResponse(status="healthy", version=_settings.version)


@router.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest) -> PredictionResponse:
    """Run a prediction using the loaded model.

    Args:
        request: JSON body containing a feature dictionary.

    Returns:
        PredictionResponse with the model output.

    Raises:
        HTTPException: 503 if no model is loaded, 500 on prediction error.
    """
    model = get_model()
    if model is None:
        raise HTTPException(status_code=503, detail="No model loaded")
    try:
        df = pd.DataFrame([request.features])
        prediction = model.predict(df)
        raw_value = prediction.iloc[0]
        # Convert numpy types to native Python for JSON serialization
        if hasattr(raw_value, "item"):
            raw_value = raw_value.item()
        return PredictionResponse(
            prediction=raw_value,
            model_version=_settings.version,
        )
    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
