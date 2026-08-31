"""Typed errors shared by the non-interactive command line entry points."""


class DynamicHistoryError(Exception):
    """Base class for expected, user-facing failures."""

    exit_code = 1


class ConfigError(DynamicHistoryError):
    exit_code = 2


class DataIntegrityError(DynamicHistoryError):
    exit_code = 3


class OutputExistsError(DynamicHistoryError):
    exit_code = 4


class ArtifactCompatibilityError(DynamicHistoryError):
    exit_code = 5
