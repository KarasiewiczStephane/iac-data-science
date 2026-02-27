"""Custom exception hierarchy for iac-data-science."""


class IACDataScienceError(Exception):
    """Base exception for the project."""


class DataLoadError(IACDataScienceError):
    """Raised when data loading fails."""


class ValidationError(IACDataScienceError):
    """Raised when data validation fails."""


class ModelError(IACDataScienceError):
    """Raised when model operations fail."""


class ConfigError(IACDataScienceError):
    """Raised when configuration is invalid or missing."""
