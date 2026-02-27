"""Tests for logging and exception utilities."""

import logging
from pathlib import Path

import pytest

from src.utils.exceptions import (
    ConfigError,
    DataLoadError,
    IACDataScienceError,
    ModelError,
    ValidationError,
)
from src.utils.logging import get_logger, setup_logging


class TestSetupLogging:
    """Tests for the setup_logging function."""

    def teardown_method(self) -> None:
        logger = logging.getLogger("iac_data_science")
        logger.handlers.clear()

    def test_returns_logger(self) -> None:
        logger = setup_logging()
        assert isinstance(logger, logging.Logger)
        assert logger.name == "iac_data_science"

    def test_sets_level(self) -> None:
        logger = setup_logging(level="DEBUG")
        assert logger.level == logging.DEBUG

    def test_console_handler_added(self) -> None:
        logger = setup_logging()
        assert len(logger.handlers) >= 1
        assert isinstance(logger.handlers[0], logging.StreamHandler)

    def test_file_handler_added(self, tmp_path: Path) -> None:
        log_file = tmp_path / "test.log"
        logger = setup_logging(log_file=log_file)
        file_handlers = [h for h in logger.handlers if isinstance(h, logging.FileHandler)]
        assert len(file_handlers) == 1

    def test_file_logging_writes(self, tmp_path: Path) -> None:
        log_file = tmp_path / "test.log"
        logger = setup_logging(level="INFO", log_file=log_file)
        logger.info("hello from test")
        # Flush handlers
        for h in logger.handlers:
            h.flush()
        content = log_file.read_text()
        assert "hello from test" in content

    def test_no_duplicate_handlers(self) -> None:
        setup_logging()
        setup_logging()
        logger = logging.getLogger("iac_data_science")
        console_handlers = [h for h in logger.handlers if isinstance(h, logging.StreamHandler)]
        assert len(console_handlers) == 1

    def test_custom_format(self) -> None:
        fmt = "%(levelname)s: %(message)s"
        logger = setup_logging(format_string=fmt)
        assert logger.handlers[0].formatter._fmt == fmt


class TestGetLogger:
    """Tests for the get_logger helper."""

    def test_returns_child_logger(self) -> None:
        logger = get_logger("data")
        assert logger.name == "iac_data_science.data"

    def test_different_names_different_loggers(self) -> None:
        a = get_logger("data")
        b = get_logger("models")
        assert a is not b


class TestExceptions:
    """Tests for custom exception hierarchy."""

    def test_base_exception_can_be_raised(self) -> None:
        with pytest.raises(IACDataScienceError, match="base error"):
            raise IACDataScienceError("base error")

    def test_data_load_error_is_subclass(self) -> None:
        assert issubclass(DataLoadError, IACDataScienceError)

    def test_validation_error_is_subclass(self) -> None:
        assert issubclass(ValidationError, IACDataScienceError)

    def test_model_error_is_subclass(self) -> None:
        assert issubclass(ModelError, IACDataScienceError)

    def test_config_error_is_subclass(self) -> None:
        assert issubclass(ConfigError, IACDataScienceError)

    def test_catch_specific_exception(self) -> None:
        try:
            raise DataLoadError("file not found")
        except DataLoadError as e:
            assert "file not found" in str(e)

    def test_catch_base_catches_children(self) -> None:
        try:
            raise ModelError("model broken")
        except IACDataScienceError as e:
            assert "model broken" in str(e)
