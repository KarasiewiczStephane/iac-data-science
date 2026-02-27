"""Application configuration loaded from environment and YAML."""

import logging
from pathlib import Path

import yaml
from pydantic import field_validator
from pydantic_settings import BaseSettings

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_CONFIG_PATH = _PROJECT_ROOT / "configs" / "config.yaml"


def _load_yaml(path: Path) -> dict:
    """Load a YAML file and return its contents as a dictionary.

    Args:
        path: Path to the YAML configuration file.

    Returns:
        Parsed YAML content as a dictionary.
    """
    if not path.exists():
        logger.warning("Config file not found at %s, using defaults", path)
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


class Settings(BaseSettings):
    """Application settings sourced from env vars, .env, and config.yaml.

    Attributes:
        app_name: Display name of the application.
        version: Semantic version string.
        log_level: Logging verbosity level.
        data_dir: Root directory for data files.
        sample_dir: Directory containing sample datasets.
        model_dir: Directory for saved model artifacts.
        model_type: Default model algorithm to use.
        random_state: Seed for reproducibility.
        api_host: Hostname for the API server.
        api_port: Port for the API server.
    """

    app_name: str = "iac-data-science"
    version: str = "1.0.0"
    log_level: str = "INFO"
    data_dir: Path = Path("data")
    sample_dir: Path = Path("data/sample")
    model_dir: Path = Path("models")
    model_type: str = "random_forest"
    random_state: int = 42
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Ensure the log level is a recognized Python logging level.

        Args:
            v: The log level string to validate.

        Returns:
            Uppercased log level string.

        Raises:
            ValueError: If the level is not a valid Python log level.
        """
        valid = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v.upper() not in valid:
            raise ValueError(f"Invalid log level: {v}. Must be one of {valid}")
        return v.upper()

    @classmethod
    def from_yaml(cls, path: Path | None = None) -> "Settings":
        """Create a Settings instance seeded from a YAML config file.

        Values in the YAML file are used as defaults; environment variables
        and .env entries still take precedence.

        Args:
            path: Path to the YAML config. Defaults to configs/config.yaml.

        Returns:
            A populated Settings instance.
        """
        config_path = path or _DEFAULT_CONFIG_PATH
        raw = _load_yaml(config_path)

        flat: dict = {}
        if "app" in raw:
            flat["app_name"] = raw["app"].get("name", "iac-data-science")
            flat["version"] = raw["app"].get("version", "1.0.0")
        if "logging" in raw:
            flat["log_level"] = raw["logging"].get("level", "INFO")
        if "data" in raw:
            flat["data_dir"] = raw["data"].get("raw_dir", "data")
            flat["sample_dir"] = raw["data"].get("sample_dir", "data/sample")
        if "model" in raw:
            flat["model_dir"] = raw["model"].get("output_dir", "models")
            flat["model_type"] = raw["model"].get("default_type", "random_forest")
            flat["random_state"] = raw["model"].get("random_state", 42)
        if "api" in raw:
            flat["api_host"] = raw["api"].get("host", "0.0.0.0")
            flat["api_port"] = raw["api"].get("port", 8000)

        return cls(**flat)
