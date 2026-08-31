from __future__ import annotations


class RLError(RuntimeError):
    """06 可预期错误基类。"""

    exit_code = 2


class ConfigError(RLError):
    pass


class DataIntegrityError(RLError):
    pass


class ArtifactCompatibilityError(RLError):
    pass


class OutputExistsError(RLError):
    pass


class IllegalActionError(RLError):
    pass

