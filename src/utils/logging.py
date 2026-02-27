"""Logging configuration for iac-data-science."""

import logging
import sys
from pathlib import Path

_DEFAULT_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


def setup_logging(
    level: str = "INFO",
    log_file: Path | None = None,
    format_string: str = _DEFAULT_FORMAT,
) -> logging.Logger:
    """Configure and return the root application logger.

    Args:
        level: Logging verbosity (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_file: Optional file path to write logs to.
        format_string: Format template for log messages.

    Returns:
        Configured root application logger.
    """
    logger = logging.getLogger("iac_data_science")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    if not logger.handlers:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(logging.Formatter(format_string))
        logger.addHandler(console_handler)

    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(format_string))
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a child logger under the application namespace.

    Args:
        name: Logger name to append to the base namespace.

    Returns:
        A child logger instance.
    """
    return logging.getLogger(f"iac_data_science.{name}")
