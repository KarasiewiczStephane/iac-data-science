"""FastAPI application factory."""

from __future__ import annotations

from fastapi import FastAPI

from src.api.routes import router
from src.utils.exceptions import IACDataScienceError
from src.utils.logging import get_logger

logger = get_logger("api.app")


def _iac_exception_handler(request: object, exc: IACDataScienceError) -> dict:
    """Convert project exceptions to JSON error responses."""
    from fastapi.responses import JSONResponse

    logger.error("Handled exception: %s", exc)
    return JSONResponse(status_code=400, content={"detail": str(exc)})


def create_app() -> FastAPI:
    """Create and configure the FastAPI application.

    Returns:
        Configured FastAPI application instance.
    """
    app = FastAPI(title="iac-data-science API", version="1.0.0")
    app.include_router(router, prefix="/api/v1")
    app.add_exception_handler(IACDataScienceError, _iac_exception_handler)
    return app


app = create_app()
