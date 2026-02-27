"""Main entry point for iac-data-science."""

import logging

import uvicorn

from src.utils.config import Settings

logger = logging.getLogger(__name__)


def main() -> None:
    """Bootstrap and run the API server."""
    settings = Settings.from_yaml()
    logging.basicConfig(level=settings.log_level)
    logger.info("Starting %s v%s", settings.app_name, settings.version)
    uvicorn.run("src.api.app:app", host=settings.api_host, port=settings.api_port)


if __name__ == "__main__":
    main()
