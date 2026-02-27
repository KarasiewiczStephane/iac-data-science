"""Tests for application configuration."""

from pathlib import Path

import pytest
import yaml

from src.utils.config import Settings, _load_yaml


class TestLoadYaml:
    """Tests for YAML loading utility."""

    def test_load_valid_yaml(self, tmp_path: Path) -> None:
        cfg = {"app": {"name": "test"}}
        path = tmp_path / "config.yaml"
        path.write_text(yaml.dump(cfg))
        result = _load_yaml(path)
        assert result == cfg

    def test_load_missing_file_returns_empty(self, tmp_path: Path) -> None:
        result = _load_yaml(tmp_path / "missing.yaml")
        assert result == {}

    def test_load_empty_yaml_returns_empty(self, tmp_path: Path) -> None:
        path = tmp_path / "empty.yaml"
        path.write_text("")
        result = _load_yaml(path)
        assert result == {}


class TestSettings:
    """Tests for the Settings model."""

    def test_defaults(self) -> None:
        s = Settings()
        assert s.app_name == "iac-data-science"
        assert s.log_level == "INFO"
        assert s.data_dir == Path("data")
        assert s.model_dir == Path("models")
        assert s.api_port == 8000

    def test_override_via_kwargs(self) -> None:
        s = Settings(app_name="custom", log_level="DEBUG", api_port=9000)
        assert s.app_name == "custom"
        assert s.log_level == "DEBUG"
        assert s.api_port == 9000

    def test_invalid_log_level_raises(self) -> None:
        with pytest.raises(Exception):
            Settings(log_level="INVALID")

    def test_from_yaml(self, tmp_path: Path) -> None:
        cfg = {
            "app": {"name": "yaml-app", "version": "2.0.0"},
            "logging": {"level": "DEBUG"},
            "data": {"raw_dir": "mydata", "sample_dir": "mydata/sample"},
            "model": {
                "output_dir": "mymodels",
                "default_type": "gradient_boosting",
                "random_state": 7,
            },
            "api": {"host": "127.0.0.1", "port": 9090},
        }
        path = tmp_path / "config.yaml"
        path.write_text(yaml.dump(cfg))

        s = Settings.from_yaml(path)
        assert s.app_name == "yaml-app"
        assert s.version == "2.0.0"
        assert s.log_level == "DEBUG"
        assert s.data_dir == Path("mydata")
        assert s.model_type == "gradient_boosting"
        assert s.api_port == 9090

    def test_from_yaml_missing_file_uses_defaults(self, tmp_path: Path) -> None:
        s = Settings.from_yaml(tmp_path / "nope.yaml")
        assert s.app_name == "iac-data-science"

    def test_from_yaml_partial_config(self, tmp_path: Path) -> None:
        cfg = {"app": {"name": "partial"}}
        path = tmp_path / "config.yaml"
        path.write_text(yaml.dump(cfg))
        s = Settings.from_yaml(path)
        assert s.app_name == "partial"
        assert s.log_level == "INFO"

    def test_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("APP_NAME", "env-app")
        monkeypatch.setenv("LOG_LEVEL", "ERROR")
        s = Settings()
        assert s.app_name == "env-app"
        assert s.log_level == "ERROR"
