"""Main entry point for iac-data-science."""

import logging

from src.utils.config import Settings

logger = logging.getLogger(__name__)


def main() -> None:
    """Bootstrap and run the application."""
    settings = Settings.from_yaml()
    logging.basicConfig(level=settings.log_level)
    logger.info("Starting %s v%s", settings.app_name, settings.version)


if __name__ == "__main__":
    main()
